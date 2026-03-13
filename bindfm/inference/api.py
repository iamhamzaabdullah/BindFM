"""
BindFM Inference API
---------------------
Clean, production-ready public interface for all BindFM prediction tasks.

Usage:
    from inference.api import BindFMPredictor

    predictor = BindFMPredictor.from_checkpoint("bindfm_full.pt")

    # Predict affinity (auto-detects input format)
    result = predictor.predict_affinity(
        binder="GGTTGGTGTGGTTGG",          # aptamer sequence (DNA)
        target="MKTLLLTLVVVTIVCLDLGYT",    # protein sequence
    )
    print(result)
    # Kd: 24.3 nM, P(bind): 0.91, t½: 4.2 min

    # Predict complex structure
    struct = predictor.predict_structure("CC(=O)Oc1ccccc1C(=O)O", "MKTL")
    struct.save_pdb("aspirin_complex.pdb")

    # Generate novel RNA aptamers for a target
    candidates = predictor.generate_binders(
        target="MKTLLLTLVVVTIVCLDLGYT",
        modality="aptamer",
        n_candidates=100,
        target_kd_nM=10.0,
    )
"""

from __future__ import annotations
import math
import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass

from model.bindfm import BindFM, BindFMConfig
from model.tokenizer import EntityType, MolecularGraph, ELEMENTS


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AffinityResult:
    """All binding affinity outputs from BindFM."""
    binding_probability: float
    log_kd_nM:           float
    kd_nM:               float
    kd_uM:               float
    log_kon:             float    # log10(kon) in M⁻¹s⁻¹
    log_koff:            float    # log10(koff) in s⁻¹
    half_life_s:         float    # ln2 / koff
    uncertainty:         float    # aleatoric ± in log_kd units

    def __str__(self) -> str:
        return (
            f"BindFM Affinity Prediction\n"
            f"  Binding probability:  {self.binding_probability:.3f}\n"
            f"  Kd:                   {self._fmt_kd()}\n"
            f"  kon:                  10^{self.log_kon:.2f} M⁻¹s⁻¹\n"
            f"  koff:                 10^{self.log_koff:.2f} s⁻¹\n"
            f"  Residence time (t½):  {self._fmt_t12()}\n"
            f"  Uncertainty:          ±{self.uncertainty:.2f} log units\n"
        )

    def _fmt_kd(self) -> str:
        if self.kd_nM < 0.001:
            return f"{self.kd_nM * 1e6:.1f} fM"
        elif self.kd_nM < 1.0:
            return f"{self.kd_nM * 1000:.1f} pM"
        elif self.kd_nM < 1000.0:
            return f"{self.kd_nM:.2f} nM"
        elif self.kd_uM < 1000.0:
            return f"{self.kd_uM:.2f} µM"
        else:
            return f"{self.kd_uM / 1000:.2f} mM"

    def _fmt_t12(self) -> str:
        t = self.half_life_s
        if t < 1:
            return f"{t * 1000:.0f} ms"
        elif t < 60:
            return f"{t:.1f} s"
        elif t < 3600:
            return f"{t / 60:.1f} min"
        elif t < 86400:
            return f"{t / 3600:.1f} h"
        else:
            return f"{t / 86400:.1f} days"


@dataclass
class StructureResult:
    """Predicted 3D complex coordinates."""
    coords_a:    np.ndarray    # [N_a, 3]  binder atoms (Å)
    coords_b:    np.ndarray    # [N_b, 3]  target atoms (Å)
    n_atoms_a:   int           = 0
    n_atoms_b:   int           = 0

    def __post_init__(self):
        self.n_atoms_a = len(self.coords_a)
        self.n_atoms_b = len(self.coords_b)

    def save_pdb(self, path: str):
        """Write predicted complex to a minimal PDB file."""
        lines = [
            "REMARK  BindFM predicted binding complex\n",
            "REMARK  Chain A = binder, Chain B = target\n",
        ]
        atom_n = 1
        for i, (x, y, z) in enumerate(self.coords_a):
            lines.append(
                f"ATOM  {atom_n:5d}  CA  UNK A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            atom_n += 1
        lines.append("TER\n")
        for j, (x, y, z) in enumerate(self.coords_b):
            lines.append(
                f"ATOM  {atom_n:5d}  CA  UNK B{j+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            atom_n += 1
        lines.append("END\n")
        with open(path, "w") as f:
            f.writelines(lines)
        print(f"Saved complex to {path}")

    def __str__(self) -> str:
        return (
            f"StructureResult: "
            f"{self.n_atoms_a} binder atoms, "
            f"{self.n_atoms_b} target atoms"
        )


@dataclass
class GeneratedBinder:
    """One de novo generated candidate binder."""
    rank:                int
    sequence:            Optional[str]       # nucleic acid / protein sequence
    smiles:              Optional[str]       # for small molecule outputs
    predicted_kd_nM:     float
    predicted_log_kd:    float
    binding_probability: float
    coords:              Optional[np.ndarray]  # [N, 3]

    def __str__(self) -> str:
        mol_str = self.sequence or self.smiles or "(no sequence decoded)"
        return (
            f"Candidate #{self.rank}: {mol_str[:40]}\n"
            f"  Pred. Kd:  {self.predicted_kd_nM:.2f} nM\n"
            f"  P(bind):   {self.binding_probability:.3f}\n"
        )


# ── Predictor ─────────────────────────────────────────────────────────────────

class BindFMPredictor:
    """
    High-level inference API for BindFM.
    Handles input parsing, format detection, and output formatting.

    Accepts any input format:
      - Protein sequence (uppercase ACDEFGHIKLMNPQRSTVWY)
      - RNA sequence (ACGU)
      - DNA sequence / aptamer (ACGT)
      - SMILES string (detected by chemistry characters)
      - PDB file path (*.pdb)
      - Pre-parsed MolecularGraph (passthrough)
    """

    MODALITY_MAP = {
        "protein":    EntityType.PROTEIN,
        "rna":        EntityType.RNA,
        "dna":        EntityType.DNA,
        "aptamer":    EntityType.RNA,     # RNA aptamer by default
        "dna_aptamer":EntityType.DNA,
        "small_mol":  EntityType.SMALL_MOL,
        "drug":       EntityType.SMALL_MOL,
        "smol":       EntityType.SMALL_MOL,
        "peptide":    EntityType.PROTEIN,
        "nucleic":    EntityType.RNA,
    }

    ASSAY_MAP = {
        "Kd": 0, "Ki": 1, "IC50": 2, "EC50": 3, "AC50": 4,
        "SPR_Kd": 5, "ITC": 6, "EMSA": 7, "Tm": 8,
        "kinact_KI": 9, "SELEX": 10, "unknown": 11,
    }

    def __init__(self, model: BindFM, device: str = "cpu"):
        self.device = device
        self.model  = model.to(device).eval()

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "BindFMPredictor":
        """Load from a saved BindFM checkpoint."""
        model = BindFM.load(path, device=device)
        return cls(model, device)

    @classmethod
    def from_config(cls, config: Optional[BindFMConfig] = None,
                    device: str = "cpu") -> "BindFMPredictor":
        """Initialize with random weights (for testing / architecture verification)."""
        model = BindFM(config or BindFMConfig.small())
        return cls(model, device)

    # ── Input parsing ─────────────────────────────────────────────────────────

    def _parse(self, entity: Union[str, MolecularGraph],
               hint: Optional[str] = None) -> MolecularGraph:
        """
        Auto-detect format and parse to MolecularGraph.
        hint: optional modality string from MODALITY_MAP keys.
        """
        from data.parsers import SMILESParser, SequenceParser

        if isinstance(entity, MolecularGraph):
            return entity

        if not isinstance(entity, str):
            raise TypeError(f"Expected str or MolecularGraph, got {type(entity)}")

        s = entity.strip()

        # Hint overrides detection
        if hint:
            etype = self.MODALITY_MAP.get(hint.lower())
            if etype == EntityType.SMALL_MOL:
                return SMILESParser.parse(s)
            elif etype in (EntityType.PROTEIN, EntityType.RNA, EntityType.DNA):
                return SequenceParser.parse(s, etype)

        # PDB file
        p = Path(s)
        if p.suffix.lower() == ".pdb" and p.exists():
            from data.parsers import PDBParser
            return PDBParser.parse_chain(str(p), "A")

        # SMILES: contains chemistry punctuation or lowercase aromatic atoms
        if any(c in s for c in "()[]=#@+-./%1234567890") or \
                any(c.islower() for c in s):
            return SMILESParser.parse(s)

        # Sequence: classify by alphabet
        clean = s.upper().replace("-", "").replace(" ", "")
        if not clean:
            raise ValueError("Empty input string")

        chars = set(clean)
        if chars <= set("ACGU") and "U" in chars:
            return SequenceParser.parse(clean, EntityType.RNA)
        if chars <= set("ACGT"):
            return SequenceParser.parse(clean, EntityType.DNA)
        if chars <= set("ACDEFGHIKLMNPQRSTVWY"):
            return SequenceParser.parse(clean, EntityType.PROTEIN)

        # Mixed — fall back to SMILES
        return SMILESParser.parse(s)

    def _to_device(self, mol: MolecularGraph) -> MolecularGraph:
        """Move MolecularGraph tensors to self.device."""
        mol.atom_feats = mol.atom_feats.to(self.device)
        mol.edge_index = mol.edge_index.to(self.device)
        mol.edge_feats = mol.edge_feats.to(self.device)
        if mol.coords is not None:
            mol.coords = mol.coords.to(self.device)
        if mol.atom_mask is not None:
            mol.atom_mask = mol.atom_mask.to(self.device)
        return mol

    # ── Affinity prediction ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict_affinity(
        self,
        binder:       Union[str, MolecularGraph],
        target:       Union[str, MolecularGraph],
        binder_hint:  Optional[str] = None,
        target_hint:  Optional[str] = None,
        assay_type:   str           = "Kd",
    ) -> AffinityResult:
        """
        Predict binding affinity between any two molecular entities.

        Args:
            binder:      SMILES / sequence / PDB path / MolecularGraph
            target:      SMILES / sequence / PDB path / MolecularGraph
            binder_hint: Optional type hint: "protein","rna","dna","aptamer","small_mol"
            target_hint: Same for target
            assay_type:  One of: Kd, Ki, IC50, EC50, SPR_Kd, ITC, EMSA

        Returns:
            AffinityResult with Kd, probability, kinetics, uncertainty
        """
        assay_idx = torch.tensor(
            self.ASSAY_MAP.get(assay_type, 0),
            dtype=torch.long, device=self.device,
        )
        mol_a = self._to_device(self._parse(binder, binder_hint))
        mol_b = self._to_device(self._parse(target, target_hint))

        raw = self.model.predict_binding(
            mol_a.atom_feats, mol_a.edge_index, mol_a.edge_feats, mol_a.coords,
            mol_b.atom_feats, mol_b.edge_index, mol_b.edge_feats, mol_b.coords,
            assay_type=assay_idx,
        )

        koff_rate = 10.0 ** raw["log_koff"]
        half_life = math.log(2) / max(koff_rate, 1e-12)

        return AffinityResult(
            binding_probability = raw["binding_probability"],
            log_kd_nM           = raw["log_kd_nM"],
            kd_nM               = raw["kd_nM"],
            kd_uM               = raw["kd_uM"],
            log_kon             = raw["log_kon"],
            log_koff            = raw["log_koff"],
            half_life_s         = half_life,
            uncertainty         = raw["uncertainty"],
        )

    def predict_affinity_batch(
        self,
        binders: List[Union[str, MolecularGraph]],
        target:  Union[str, MolecularGraph],
        **kwargs,
    ) -> List[AffinityResult]:
        """Screen a list of binders against one fixed target."""
        mol_b = self._to_device(self._parse(target, kwargs.pop("target_hint", None)))
        results = []
        for b in binders:
            try:
                results.append(self.predict_affinity(b, mol_b, **kwargs))
            except Exception as e:
                # Return placeholder on parse failure
                results.append(AffinityResult(
                    binding_probability=0.0, log_kd_nM=6.0, kd_nM=1e6,
                    kd_uM=1e3, log_kon=5.0, log_koff=-1.0,
                    half_life_s=0.0, uncertainty=9.9,
                ))
        return results

    # ── Structure prediction ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict_structure(
        self,
        binder:      Union[str, MolecularGraph],
        target:      Union[str, MolecularGraph],
        binder_hint: Optional[str] = None,
        target_hint: Optional[str] = None,
        n_steps:     int           = 100,
        output_pdb:  Optional[str] = None,
    ) -> StructureResult:
        """
        Predict 3D structure of the bound complex via flow matching.

        Args:
            binder, target: molecular inputs (any supported format)
            n_steps:   Euler ODE integration steps (more = more accurate, slower)
            output_pdb: optional path to write PDB file

        Returns:
            StructureResult with binder and target atom coordinates
        """
        mol_a = self._to_device(self._parse(binder, binder_hint))
        mol_b = self._to_device(self._parse(target, target_hint))

        coords_a, coords_b = self.model.predict_structure(
            mol_a.atom_feats, mol_a.edge_index, mol_a.edge_feats,
            mol_b.atom_feats, mol_b.edge_index, mol_b.edge_feats,
            n_steps=n_steps,
        )

        result = StructureResult(
            coords_a = coords_a.cpu().numpy(),
            coords_b = coords_b.cpu().numpy(),
        )

        if output_pdb:
            result.save_pdb(output_pdb)

        return result

    # ── De novo generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_binders(
        self,
        target:           Union[str, MolecularGraph],
        target_hint:      Optional[str]  = None,
        modality:         str            = "aptamer",
        n_candidates:     int            = 10,
        target_kd_nM:     Optional[float]= None,
        n_steps:          int            = 100,
        n_atoms:          int            = 50,
        rank_by_affinity: bool           = True,
        seed:             Optional[int]  = None,
    ) -> List[GeneratedBinder]:
        """
        Generate de novo binder candidates for a target molecule.

        Args:
            target:        Target molecule (any format)
            modality:      What to generate: "aptamer","dna_aptamer","protein","small_mol"
            n_candidates:  How many to generate
            target_kd_nM:  Desired affinity (nM) — conditions generation
            n_steps:       ODE integration steps
            n_atoms:       Approximate size of generated molecule
            rank_by_affinity: Sort results by predicted Kd

        Returns:
            List of GeneratedBinder, sorted by predicted Kd if rank_by_affinity
        """
        from data.parsers import SequenceParser

        entity_type = self.MODALITY_MAP.get(modality.lower(), EntityType.RNA)
        mol_b       = self._to_device(self._parse(target, target_hint))

        log_kd_t = (
            torch.tensor(math.log10(max(target_kd_nM, 1e-3)),
                         device=self.device, dtype=torch.float32)
            if target_kd_nM is not None else None
        )

        raw_list = self.model.generate_binder(
            b_atom_feats   = mol_b.atom_feats,
            b_edge_index   = mol_b.edge_index,
            b_edge_feats   = mol_b.edge_feats,
            b_coords       = mol_b.coords,
            modality       = entity_type,
            log_kd_target  = log_kd_t,
            n_candidates   = n_candidates,
            n_steps        = n_steps,
            n_atoms        = n_atoms,
            seed           = seed,
        )

        results = []
        for i, cand in enumerate(raw_list):
            # Decode atom features → sequence (best-effort)
            seq = self._decode_sequence(cand["atom_feats"], entity_type)

            # Score decoded sequence against target
            pred_kd, pred_lkd, pred_bp = self._score_candidate(
                seq, entity_type, mol_b
            )

            results.append(GeneratedBinder(
                rank                = i + 1,
                sequence            = seq,
                smiles              = None,   # atom→SMILES decoder: future work
                predicted_kd_nM     = pred_kd,
                predicted_log_kd    = pred_lkd,
                binding_probability = pred_bp,
                coords              = cand["coords"].cpu().numpy(),
            ))

        if rank_by_affinity:
            results.sort(key=lambda x: x.predicted_kd_nM)
            for i, r in enumerate(results):
                r.rank = i + 1

        return results

    def _decode_sequence(
        self,
        atom_feats:  torch.Tensor,   # [N, ATOM_FEAT_DIM]
        entity_type: EntityType,
    ) -> Optional[str]:
        """
        Decode generated atom feature vectors to a biological sequence.

        Strategy:
          1. For each atom, identify element from one-hot (dims 0:124)
          2. Group atoms into residues by the residue_idx feature (dim ~270)
          3. Assign nucleotide/AA identity from element composition per residue

        This is a post-hoc decoding step — the model generates at atom level,
        sequence is inferred from atomic composition.
        """
        if entity_type == EntityType.SMALL_MOL:
            return None   # SMILES reconstruction requires a dedicated graph decoder

        N = atom_feats.shape[0]
        if N == 0:
            return None

        # Element one-hot: first 124 dims
        elem_probs  = atom_feats[:, :124].softmax(dim=-1)
        elem_ids    = elem_probs.argmax(dim=-1).cpu().numpy()

        # Element index → symbol
        elem_syms   = [ELEMENTS[i] if i < len(ELEMENTS) else "C" for i in elem_ids]

        if entity_type == EntityType.PROTEIN:
            return self._decode_protein(elem_syms)
        elif entity_type in (EntityType.RNA, EntityType.DNA):
            return self._decode_nucleic(elem_syms, entity_type)

        return None

    def _decode_protein(self, elem_syms: list) -> str:
        """
        Decode atom element symbols to amino acid sequence.
        Uses mean atoms-per-residue (~7 heavy atoms) as window size.
        Identifies residue type from sidechain element composition.
        """
        atoms_per_res = 7
        N = len(elem_syms)
        seq = []
        for start in range(0, N, atoms_per_res):
            chunk = elem_syms[start:start + atoms_per_res]
            n_s   = chunk.count("S")
            n_n   = chunk.count("N")
            n_o   = chunk.count("O")
            n_c   = chunk.count("C")

            # Rule-based residue guessing from element composition
            if n_s >= 1:
                seq.append("C" if n_s == 1 else "M")   # Cys / Met
            elif n_n >= 3:
                seq.append("R")    # Arg (most N-rich)
            elif n_n == 2 and n_o == 1:
                seq.append("Q")    # Gln / Asn
            elif n_n == 2:
                seq.append("H")    # His
            elif n_o >= 2:
                seq.append("E")    # Glu / Asp
            elif n_o == 1 and n_n == 0:
                seq.append("S")    # Ser
            elif n_c >= 3 and n_n == 0 and n_o == 0:
                seq.append("L")    # Leu / Ile / Val
            else:
                seq.append("A")    # Ala as default

        return "".join(seq[:200])   # cap at 200 aa

    def _decode_nucleic(self, elem_syms: list, entity_type: EntityType) -> str:
        """
        Decode atom element symbols to nucleotide sequence.
        Uses ~20 heavy atoms per nucleotide as window size.
        Identifies base from nitrogen count (Purines > Pyrimidines).
        """
        atoms_per_nt = 20
        N = len(elem_syms)
        bases = (["A", "C", "G", "U"] if entity_type == EntityType.RNA
                 else ["A", "C", "G", "T"])
        seq = []
        for start in range(0, N, atoms_per_nt):
            chunk   = elem_syms[start:start + atoms_per_nt]
            n_n     = chunk.count("N")

            # Guanine has 5N (purine + extra amino), Adenine has 5N too
            # Cytosine has 3N, Uracil/Thymine has 2N
            if n_n >= 5:
                # Distinguish G (has O6) from A by oxygen count
                n_o = chunk.count("O")
                seq.append("G" if n_o >= 2 else "A")
            elif n_n == 3:
                seq.append("C")
            else:
                seq.append(bases[-1])   # U or T

        return "".join(seq[:200])   # cap at 200 nt

    def _score_candidate(
        self,
        seq:         Optional[str],
        entity_type: EntityType,
        mol_b:       MolecularGraph,
    ) -> tuple:
        """
        Score a generated candidate against the target.
        Returns (kd_nM, log_kd_nM, binding_probability).
        Falls back to neutral values on parse failure.
        """
        from data.parsers import SequenceParser

        if seq is None or len(seq) < 2:
            return 1000.0, 3.0, 0.5

        try:
            mol_a = self._to_device(SequenceParser.parse(seq, entity_type))
            raw   = self.model.predict_binding(
                mol_a.atom_feats, mol_a.edge_index, mol_a.edge_feats, mol_a.coords,
                mol_b.atom_feats, mol_b.edge_index, mol_b.edge_feats, mol_b.coords,
            )
            return raw["kd_nM"], raw["log_kd_nM"], raw["binding_probability"]
        except Exception:
            return 1000.0, 3.0, 0.5

    # ── Library screening ─────────────────────────────────────────────────────

    def screen_library(
        self,
        library:     List[Union[str, MolecularGraph]],
        target:      Union[str, MolecularGraph],
        target_hint: Optional[str] = None,
        top_k:       int           = 10,
        assay_type:  str           = "Kd",
        verbose:     bool          = True,
    ) -> List[Dict[str, Any]]:
        """
        Screen a compound library against a target.

        Args:
            library:   List of SMILES strings, sequences, or MolecularGraphs
            target:    Target molecule
            top_k:     Return only top K hits
            assay_type: Assay type for conditioning
            verbose:   Print progress

        Returns:
            List of dicts sorted by kd_nM (ascending = best binders first)
        """
        mol_b   = self._to_device(self._parse(target, target_hint))
        results = []

        for i, compound in enumerate(library):
            try:
                mol_a  = self._to_device(self._parse(compound))
                assay_idx = torch.tensor(
                    self.ASSAY_MAP.get(assay_type, 0),
                    dtype=torch.long, device=self.device,
                )
                raw = self.model.predict_binding(
                    mol_a.atom_feats, mol_a.edge_index, mol_a.edge_feats, mol_a.coords,
                    mol_b.atom_feats, mol_b.edge_index, mol_b.edge_feats, mol_b.coords,
                    assay_type=assay_idx,
                )
                results.append({
                    "compound":     compound if isinstance(compound, str) else "MolecularGraph",
                    "rank":         None,
                    "kd_nM":        raw["kd_nM"],
                    "log_kd_nM":    raw["log_kd_nM"],
                    "binding_prob": raw["binding_probability"],
                    "uncertainty":  raw["uncertainty"],
                })
            except Exception:
                pass   # skip compounds that fail to parse

            if verbose and (i + 1) % 100 == 0:
                print(f"  Screened {i+1}/{len(library)}...")

        results.sort(key=lambda x: x["kd_nM"])
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results[:top_k]
