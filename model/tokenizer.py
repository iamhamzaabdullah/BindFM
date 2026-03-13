"""
BindFM Tokenizer
----------------
Universal atomic tokenizer for all biomolecular binding modalities.
No pretrained vocabularies. No borrowed representations.
Everything is atoms, bonds, and geometry.

Handles:
  - Proteins (standard + modified AAs + PTMs)
  - DNA / RNA (standard + all chemical modifications)
  - Small molecules (arbitrary atom graphs from SMILES/SDF)
  - Hybrid / covalently modified entities
"""

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ──────────────────────────────────────────────
# 1. ELEMENT VOCABULARY
#    All 118 elements + special tokens
# ──────────────────────────────────────────────

ELEMENTS = [
    "<PAD>", "<MASK>", "<UNK>",          # special
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
    "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    # Biological pseudoatoms
    "Du",   # dummy / centroid
]

ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
N_ELEMENTS = len(ELEMENTS)  # 124


# ──────────────────────────────────────────────
# 2. ATOM FEATURE ENUMS
# ──────────────────────────────────────────────

class Hybridization(IntEnum):
    UNSPECIFIED = 0
    S           = 1
    SP          = 2
    SP2         = 3
    SP3         = 4
    SP3D        = 5
    SP3D2       = 6
    OTHER       = 7

class Chirality(IntEnum):
    NONE        = 0
    CW          = 1   # R
    CCW         = 2   # S
    OTHER       = 3

class BondType(IntEnum):
    NONE        = 0
    SINGLE      = 1
    DOUBLE      = 2
    TRIPLE      = 3
    AROMATIC    = 4
    COORDINATE  = 5   # metal coordination
    COVALENT_W  = 6   # covalent warhead bond
    HYDROGEN    = 7   # explicit H-bond (non-covalent)
    IONIC       = 8

class EntityType(IntEnum):
    PROTEIN     = 0
    DNA         = 1
    RNA         = 2
    SMALL_MOL   = 3
    LIPID       = 4
    GLYCAN      = 5
    ION         = 6
    WATER       = 7
    UNKNOWN     = 8

class StructuralContext(IntEnum):
    """For nucleic acids — secondary structure annotation"""
    NONE        = 0
    STEM        = 1
    LOOP        = 2
    BULGE       = 3
    JUNCTION    = 4
    G4_CORE     = 5
    G4_LOOP     = 6
    I_MOTIF     = 7
    TRIPLEX     = 8
    PSEUDOKNOT  = 9

class ModificationType(IntEnum):
    """Chemical modifications — nucleic acid and protein"""
    NONE            = 0
    # Nucleic acid backbone
    PHOSPHOROTHIOATE = 1
    BORANOPHOSPHATE  = 2
    METHYLPHOSPHONATE= 3
    # Nucleic acid sugar
    OMe_2PRIME      = 4
    F_2PRIME        = 5
    LNA             = 6   # locked nucleic acid
    ENA             = 7   # ethylene-bridged
    FANA            = 8   # 2'F arabino
    MORPHOLINO      = 9
    PNA_BACKBONE    = 10  # peptide nucleic acid
    # Nucleic acid base
    M6A             = 11  # N6-methyladenosine
    M5C             = 12  # 5-methylcytosine
    PSEUDOURIDINE   = 13
    M1A             = 14
    INOSINE         = 15
    # Protein PTMs
    PHOSPHO         = 16
    ACETYL          = 17
    METHYL          = 18
    UBIQUITIN_SITE  = 19
    GLYCOSYLATION   = 20
    # Terminus
    FIVE_PRIME_CAP  = 21
    THREE_POLY_A    = 22
    PEG_CONJUGATE   = 23
    FLUOROPHORE     = 24


# ──────────────────────────────────────────────
# 3. ATOM NODE FEATURE VECTOR
#    Fixed-dim, fully categorical + continuous
#    dim = 124 + 8 + 4 + 9 + 2 + 2 + 1 + 1 + 1 + 1 + 25 + 9 + 10 = ~197
#    Embedded later to d_model via linear projection
# ──────────────────────────────────────────────

@dataclass
class AtomFeatures:
    """
    All per-atom features for BindFM.
    One instance per atom in the molecular graph.
    """

    # Identity
    element_idx:        int     = 0     # into ELEMENTS vocab
    formal_charge:      int     = 0     # -4 to +4, clipped
    isotope:            int     = 0     # 0 = unspecified

    # Geometry
    hybridization:      int     = 0     # Hybridization enum
    chirality:          int     = 0     # Chirality enum
    num_hydrogens:      int     = 0     # explicit + implicit
    num_heavy_neighbors:int     = 0
    degree:             int     = 0     # graph degree

    # Ring membership
    in_ring:            bool    = False
    ring_size:          int     = 0     # 0 if not in ring, else smallest ring
    is_aromatic:        bool    = False

    # H-bonding
    is_hbd:             bool    = False  # H-bond donor
    is_hba:             bool    = False  # H-bond acceptor

    # Partial charge (float, Gasteiger or learned)
    partial_charge:     float   = 0.0

    # Hydrophobicity contribution
    is_hydrophobic:     bool    = False

    # Biological context
    entity_type:        int     = 0     # EntityType enum
    modification:       int     = 0     # ModificationType enum
    structural_context: int     = 0     # StructuralContext enum

    # Polymer context (protein/nucleic)
    is_backbone:        bool    = False  # backbone vs sidechain/base
    residue_idx:        int     = 0     # position in polymer chain
    chain_id:           int     = 0     # which chain (multi-chain inputs)

    # Binding annotations (optional labels for supervised tasks)
    is_interface:       bool    = False  # at known binding interface
    is_allosteric:      bool    = False  # in known allosteric site
    is_covalent_site:   bool    = False  # covalent attachment point

    def to_tensor(self) -> torch.Tensor:
        """
        Returns a 1D float tensor of atom features.
        Categoricals are one-hot encoded.
        """
        feats = []

        # Element one-hot [124]
        elem_oh = torch.zeros(N_ELEMENTS)
        elem_oh[self.element_idx] = 1.0
        feats.append(elem_oh)

        # Formal charge as scalar, normalized [-4,+4] → [-1,1] [1]
        feats.append(torch.tensor([self.formal_charge / 4.0]))

        # Hybridization one-hot [8]
        hyb_oh = torch.zeros(len(Hybridization))
        hyb_oh[self.hybridization] = 1.0
        feats.append(hyb_oh)

        # Chirality one-hot [4]
        chi_oh = torch.zeros(len(Chirality))
        chi_oh[self.chirality] = 1.0
        feats.append(chi_oh)

        # Continuous features [6]
        feats.append(torch.tensor([
            self.num_hydrogens        / 8.0,
            self.num_heavy_neighbors  / 8.0,
            self.degree               / 8.0,
            float(self.in_ring),
            self.ring_size            / 8.0,
            float(self.is_aromatic),
        ]))

        # H-bond [2]
        feats.append(torch.tensor([float(self.is_hbd), float(self.is_hba)]))

        # Partial charge [1]
        feats.append(torch.tensor([self.partial_charge]))

        # Hydrophobic [1]
        feats.append(torch.tensor([float(self.is_hydrophobic)]))

        # Entity type one-hot [9]
        ent_oh = torch.zeros(len(EntityType))
        ent_oh[self.entity_type] = 1.0
        feats.append(ent_oh)

        # Modification one-hot [25]
        mod_oh = torch.zeros(len(ModificationType))
        mod_oh[self.modification] = 1.0
        feats.append(mod_oh)

        # Structural context one-hot [10]
        sc_oh = torch.zeros(len(StructuralContext))
        sc_oh[self.structural_context] = 1.0
        feats.append(sc_oh)

        # Polymer context [3]
        feats.append(torch.tensor([
            float(self.is_backbone),
            self.residue_idx / 2000.0,  # normalize; clip large chains
            self.chain_id    / 8.0,
        ]))

        # Binding site annotations [3]
        feats.append(torch.tensor([
            float(self.is_interface),
            float(self.is_allosteric),
            float(self.is_covalent_site),
        ]))

        return torch.cat(feats, dim=0)  # ~197-dim

    @property
    def dim(self) -> int:
        return (
            N_ELEMENTS      # 124
            + 1             # charge
            + len(Hybridization)    # 8
            + len(Chirality)        # 4
            + 6             # ring/geometry
            + 2             # hbond
            + 1             # partial charge
            + 1             # hydrophobic
            + len(EntityType)       # 9
            + len(ModificationType) # 25
            + len(StructuralContext)# 10
            + 3             # polymer context
            + 3             # binding annotations
        )  # = 197


# Module-level constant — use this instead of hardcoding 197 everywhere
ATOM_FEAT_DIM: int = 197
BOND_FEAT_DIM: int = 14   # BondFeatures.dim


# ──────────────────────────────────────────────
# 4. BOND EDGE FEATURE VECTOR
# ──────────────────────────────────────────────

@dataclass
class BondFeatures:
    bond_type:          int     = 0     # BondType enum
    bond_length:        float   = 1.5   # Angstroms
    in_same_ring:       bool    = False
    is_conjugated:      bool    = False
    is_rotatable:       bool    = False
    stereo:             int     = 0     # 0=none, 1=E, 2=Z, 3=CIS, 4=TRANS

    def to_tensor(self) -> torch.Tensor:
        bt_oh = torch.zeros(len(BondType))
        bt_oh[self.bond_type] = 1.0
        cont = torch.tensor([
            self.bond_length / 5.0,   # normalize to ~0-1
            float(self.in_same_ring),
            float(self.is_conjugated),
            float(self.is_rotatable),
            self.stereo / 4.0,
        ])
        return torch.cat([bt_oh, cont], dim=0)  # 14-dim

    @property
    def dim(self) -> int:
        return len(BondType) + 5  # 14


# ──────────────────────────────────────────────
# 5. MOLECULAR GRAPH — the primary data structure
# ──────────────────────────────────────────────

@dataclass
class MolecularGraph:
    """
    Universal molecular representation for BindFM.
    Entity-agnostic: protein, RNA, small mol all look the same here.

    coords:     [N_atoms, 3]   — 3D Cartesian, None if unknown
    atom_feats: [N_atoms, 197] — per-atom feature vectors
    edge_index: [2, N_edges]   — COO sparse adjacency
    edge_feats: [N_edges, 14]  — per-bond feature vectors
    entity_type: EntityType
    n_atoms: int
    """
    atom_feats:  torch.Tensor               # [N, 197]
    edge_index:  torch.Tensor               # [2, E]
    edge_feats:  torch.Tensor               # [E, 14]
    entity_type: EntityType
    coords:      Optional[torch.Tensor] = None  # [N, 3]
    atom_mask:   Optional[torch.Tensor] = None  # [N] bool, True=valid
    sequence:    Optional[str]          = None  # raw sequence if applicable
    n_atoms:     int                    = 0

    def __post_init__(self):
        self.n_atoms = self.atom_feats.shape[0]
        if self.atom_mask is None:
            self.atom_mask = torch.ones(self.n_atoms, dtype=torch.bool)


@dataclass
class BindingPair:
    """
    A binding pair: two MolecularGraph entities + optional labels.
    This is the fundamental unit BindFM consumes.
    """
    entity_a:       MolecularGraph
    entity_b:       MolecularGraph

    # Labels (all optional — some data has structure, some only Kd, some neither)
    log_kd:         Optional[float]         = None  # log10(Kd in nM)
    kd_units:       Optional[str]           = None  # "Kd","Ki","IC50","Kd_SPR",...
    complex_coords: Optional[torch.Tensor]  = None  # [N_a+N_b, 3] bound complex
    contact_map:    Optional[torch.Tensor]  = None  # [N_a, N_b] float, known contacts
    is_binder:      Optional[bool]          = None  # positive/negative label
    kon:            Optional[float]         = None  # on-rate log10
    koff:           Optional[float]         = None  # off-rate log10
    is_covalent:    Optional[bool]          = False
    is_allosteric:  Optional[bool]          = False
    is_cooperative: Optional[bool]          = False


# ──────────────────────────────────────────────
# 6. PARSER STUBS
#    Full parsers implemented in data/parsers.py
#    Interfaces defined here for clarity
# ──────────────────────────────────────────────

class MolecularGraphBuilder:
    """
    Converts raw molecular representations into MolecularGraph.
    No RDKit monkey-patching — we own the featurization pipeline.
    """

    @staticmethod
    def from_smiles(smiles: str) -> MolecularGraph:
        """Parse SMILES → MolecularGraph. Requires RDKit."""
        from data.parsers import SMILESParser
        return SMILESParser.parse(smiles)

    @staticmethod
    def from_pdb_chain(pdb_str: str, chain_id: str,
                       entity_type: EntityType) -> MolecularGraph:
        """Parse PDB chain → MolecularGraph."""
        from data.parsers import PDBParser
        return PDBParser.parse_chain(pdb_str, chain_id, entity_type)

    @staticmethod
    def from_sequence(sequence: str,
                      entity_type: EntityType,
                      modifications: dict = None) -> MolecularGraph:
        """
        Build from sequence only (no 3D coords).
        entity_type determines whether to interpret as protein/DNA/RNA.
        modifications: dict of {position: ModificationType}
        """
        from data.parsers import SequenceParser
        return SequenceParser.parse(sequence, entity_type, modifications)
