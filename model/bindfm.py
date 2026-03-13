"""
BindFM — Complete Model
------------------------
The single end-to-end neural network for universal binding prediction.

Assembles:
  DualEntityEncoder      — shared SE(3)-equivariant atom encoder for both entities
  PairFormerTrunk        — binding interaction learning (pair representation)
  AffinityHead           — universal Kd / binding probability / kinetics
  StructureFlowMatchingHead — 3D complex structure prediction via flow matching
  GenerativeHead         — de novo binder generation

Clean interface throughout:
  Encoder  outputs: h [N, D_OUT], x [N, 3]
  Trunk    inputs:  h_a, x_a, h_b, x_b
  Trunk    outputs: single_a, single_b, pair
  Heads    inputs:  single_a, single_b, pair
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

from model.tokenizer import ATOM_FEAT_DIM, EntityType, MolecularGraph, BindingPair
from model.encoder import DualEntityEncoder, D_OUT
from model.trunk import PairFormerTrunk, D_SINGLE, D_PAIR
from model.heads import AffinityHead, StructureFlowMatchingHead, GenerativeHead, BindFMLoss


# ── Model Configuration ───────────────────────────────────────────────────────

@dataclass
class BindFMConfig:
    """Full model hyperparameters."""

    # Encoder
    n_encoder_layers:   int   = 8
    d_encoder_hidden:   int   = 256
    d_encoder_edge:     int   = 128
    d_encoder_out:      int   = 512
    n_rbf:              int   = 64
    cutoff_angst:       float = 8.0
    encoder_dropout:    float = 0.1

    # Trunk
    n_trunk_layers:     int   = 32
    d_single:           int   = 512
    d_pair:             int   = 128
    trunk_dropout:      float = 0.1

    # Heads
    n_assay_types:      int   = 12
    d_affinity_hidden:  int   = 512
    d_struct_hidden:    int   = 256
    d_gen_hidden:       int   = 512
    max_gen_atoms:      int   = 200

    @classmethod
    def small(cls) -> "BindFMConfig":
        """Development / testing config — fast forward pass on CPU."""
        return cls(
            n_encoder_layers  = 3,
            d_encoder_hidden  = 64,
            d_encoder_edge    = 32,
            d_encoder_out     = 128,
            n_rbf             = 32,
            n_trunk_layers    = 3,
            d_single          = 128,
            d_pair            = 32,
            d_affinity_hidden = 128,
            d_struct_hidden   = 64,
            d_gen_hidden      = 128,
        )

    @classmethod
    def medium(cls) -> "BindFMConfig":
        """Training config — good accuracy / speed balance."""
        return cls(
            n_encoder_layers  = 6,
            d_encoder_hidden  = 128,
            d_encoder_edge    = 64,
            d_encoder_out     = 256,
            n_rbf             = 64,
            n_trunk_layers    = 12,
            d_single          = 256,
            d_pair            = 64,
            d_affinity_hidden = 256,
            d_struct_hidden   = 128,
            d_gen_hidden      = 256,
        )

    @classmethod
    def full(cls) -> "BindFMConfig":
        """Full-scale BindFM — for training on A100 cluster."""
        return cls()   # defaults are full-scale


# ── Main Model ────────────────────────────────────────────────────────────────

class BindFM(nn.Module):
    """
    BindFM: Universal Binding Foundation Model.

    Supports all five binding modalities:
      protein ↔ small molecule
      protein ↔ protein
      protein ↔ nucleic acid
      nucleic ↔ small molecule
      nucleic ↔ nucleic  (aptamer self-structure, G4, etc.)

    Three co-trained output heads:
      1. Affinity head:   Kd, binary binding, kon/koff, uncertainty
      2. Structure head:  3D complex structure via flow matching
      3. Generative head: de novo binder generation conditioned on target
    """

    def __init__(self, config: Optional[BindFMConfig] = None):
        super().__init__()
        if config is None:
            config = BindFMConfig()
        self.config = config

        # 1. Shared equivariant atom encoder (same weights for both entities)
        self.encoder = DualEntityEncoder(
            atom_feat_dim = ATOM_FEAT_DIM,
            d_hidden      = config.d_encoder_hidden,
            d_edge        = config.d_encoder_edge,
            d_out         = config.d_encoder_out,
            n_rbf         = config.n_rbf,
            n_layers      = config.n_encoder_layers,
            cutoff        = config.cutoff_angst,
            dropout       = config.encoder_dropout,
        )

        # 2. PairFormer trunk
        self.trunk = PairFormerTrunk(
            d_single   = config.d_single,
            d_pair     = config.d_pair,
            n_layers   = config.n_trunk_layers,
            d_enc_out  = config.d_encoder_out,
            dropout    = config.trunk_dropout,
        )

        # 3. Output heads
        self.affinity_head = AffinityHead(
            d_single      = config.d_single,
            d_pair        = config.d_pair,
            n_assay_types = config.n_assay_types,
            d_hidden      = config.d_affinity_hidden,
        )

        self.structure_head = StructureFlowMatchingHead(
            d_single = config.d_single,
            d_pair   = config.d_pair,
            d_hidden = config.d_struct_hidden,
        )

        self.gen_head = GenerativeHead(
            d_single      = config.d_single,
            d_pair        = config.d_pair,
            atom_feat_dim = ATOM_FEAT_DIM,
            d_hidden      = config.d_gen_hidden,
            max_gen_atoms = config.max_gen_atoms,
        )

        self.loss_fn = BindFMLoss()

        self._log_param_count()

    def _log_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        enc   = sum(p.numel() for p in self.encoder.parameters())
        trunk = sum(p.numel() for p in self.trunk.parameters())
        heads = sum(p.numel() for p in self.affinity_head.parameters()) + \
                sum(p.numel() for p in self.structure_head.parameters()) + \
                sum(p.numel() for p in self.gen_head.parameters())
        print(f"BindFM parameters:")
        print(f"  Encoder (shared):  {enc:>12,}")
        print(f"  PairFormer trunk:  {trunk:>12,}")
        print(f"  Output heads:      {heads:>12,}")
        print(f"  TOTAL:             {total:>12,}")

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        a_atom_feats:   Tensor,              # [N_a, ATOM_FEAT_DIM]
        a_edge_index:   Tensor,              # [2, E_a]
        a_edge_feats:   Tensor,              # [E_a, 14]
        a_coords:       Optional[Tensor],    # [N_a, 3] or None
        b_atom_feats:   Tensor,              # [N_b, ATOM_FEAT_DIM]
        b_edge_index:   Tensor,              # [2, E_b]
        b_edge_feats:   Tensor,              # [E_b, 14]
        b_coords:       Optional[Tensor],    # [N_b, 3] or None

        # Head selection
        run_affinity:   bool = True,
        run_structure:  bool = False,
        run_gen:        bool = False,

        # Affinity head
        assay_type:     Optional[Tensor] = None,

        # Structure head (training only)
        noisy_coords_a: Optional[Tensor] = None,  # [N_a, 3]
        noisy_coords_b: Optional[Tensor] = None,  # [N_b, 3]
        flow_t:         Optional[Tensor] = None,   # [] scalar time

        # Generation head (training only)
        gen_modality:      Optional[Tensor] = None,
        gen_noisy_feats:   Optional[Tensor] = None,
        gen_noisy_coords:  Optional[Tensor] = None,
        gen_log_kd_target: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Unified forward pass. Returns dict with outputs from requested heads."""
        results = {}

        # 1. Encode both entities through shared encoder
        h_a, x_a, h_b, x_b = self.encoder(
            a_atom_feats, a_edge_index, a_edge_feats, a_coords, None,
            b_atom_feats, b_edge_index, b_edge_feats, b_coords, None,
        )

        # 2. PairFormer trunk: produce single and pair representations
        single_a, single_b, pair = self.trunk(h_a, x_a, h_b, x_b)

        results["single_a"] = single_a
        results["single_b"] = single_b
        results["pair"]     = pair

        # 3. Affinity head
        if run_affinity:
            results["affinity"] = self.affinity_head(
                single_a, single_b, pair, assay_type
            )

        # 4. Structure head (training: predict velocity field)
        if run_structure and noisy_coords_a is not None and flow_t is not None:
            vel_a, vel_b = self.structure_head(
                single_a, single_b, pair,
                noisy_coords_a, noisy_coords_b, flow_t,
            )
            results["structure"] = {"vel_a": vel_a, "vel_b": vel_b}

        # 5. Generative head (training: predict velocity in (feat, coord) space)
        if run_gen and gen_noisy_feats is not None and flow_t is not None:
            vel_feats, vel_coords = self.gen_head(
                single_b, gen_noisy_feats, gen_noisy_coords,
                flow_t, gen_modality, gen_log_kd_target,
            )
            results["generation"] = {
                "vel_feats":  vel_feats,
                "vel_coords": vel_coords,
            }

        return results

    # ── Inference: binding affinity ───────────────────────────────────────────

    @torch.no_grad()
    def predict_binding(
        self,
        a_atom_feats: Tensor, a_edge_index: Tensor,
        a_edge_feats: Tensor, a_coords: Optional[Tensor],
        b_atom_feats: Tensor, b_edge_index: Tensor,
        b_edge_feats: Tensor, b_coords: Optional[Tensor],
        assay_type:   Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """
        Predict binding affinity between two entities.
        Returns human-readable dict with Kd, probability, kinetics, uncertainty.
        """
        out = self.forward(
            a_atom_feats, a_edge_index, a_edge_feats, a_coords,
            b_atom_feats, b_edge_index, b_edge_feats, b_coords,
            run_affinity=True, run_structure=False, run_gen=False,
            assay_type=assay_type,
        )
        aff    = out["affinity"]
        log_kd = aff["log_kd"].item()
        return {
            "binding_probability": aff["binding_prob"].item(),
            "log_kd_nM":          log_kd,
            "kd_nM":              10 ** log_kd,
            "kd_uM":              10 ** (log_kd - 3.0),
            "log_kon":            aff["log_kon"].item(),
            "log_koff":           aff["log_koff"].item(),
            "uncertainty":        aff["uncertainty"].item(),
        }

    # ── Inference: structure prediction ──────────────────────────────────────

    @torch.no_grad()
    def predict_structure(
        self,
        a_atom_feats: Tensor, a_edge_index: Tensor, a_edge_feats: Tensor,
        b_atom_feats: Tensor, b_edge_index: Tensor, b_edge_feats: Tensor,
        n_steps: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict 3D complex structure.
        Returns (coords_a, coords_b) in Angstroms.
        """
        h_a, x_a, h_b, x_b = self.encoder(
            a_atom_feats, a_edge_index, a_edge_feats, None, None,
            b_atom_feats, b_edge_index, b_edge_feats, None, None,
        )
        single_a, single_b, pair = self.trunk(h_a, x_a, h_b, x_b)

        device = str(a_atom_feats.device)
        return self.structure_head.sample(
            single_a, single_b, pair,
            n_atoms_a = a_atom_feats.shape[0],
            n_atoms_b = b_atom_feats.shape[0],
            n_steps   = n_steps,
            device    = device,
        )

    # ── Inference: de novo generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate_binder(
        self,
        b_atom_feats:  Tensor,
        b_edge_index:  Tensor,
        b_edge_feats:  Tensor,
        b_coords:      Optional[Tensor] = None,
        modality:      EntityType = EntityType.SMALL_MOL,
        log_kd_target: Optional[float] = None,
        n_candidates:  int = 10,
        n_steps:       int = 100,
        n_atoms:       int = 50,
        seed:          Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate candidate binders for a target.
        Returns list of dicts with atom_feats and coords.
        """
        device = b_atom_feats.device

        # Encode target alone (pass as both A and B to get valid trunk output)
        h_b, x_b, _, _ = self.encoder(
            b_atom_feats, b_edge_index, b_edge_feats, b_coords, None,
        )
        # Use single-entity trunk: build single from B for generation conditioning
        single_b, _, _ = self.trunk(h_b, x_b, h_b, x_b)

        mod_tensor = torch.tensor(int(modality), device=device)
        kd_tensor  = (torch.tensor(log_kd_target, device=device, dtype=torch.float32)
                      if log_kd_target is not None else None)

        candidates = []
        for i in range(n_candidates):
            feats, coords = self.gen_head.generate(
                single_b    = single_b,
                modality    = mod_tensor,
                log_kd_target = kd_tensor,
                n_atoms     = n_atoms,
                n_steps     = n_steps,
                device      = str(device),
                seed        = None if seed is None else seed + i,
            )
            candidates.append({"atom_feats": feats, "coords": coords, "rank": i + 1})

        return candidates

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)
        print(f"BindFM saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BindFM":
        ckpt  = torch.load(path, map_location=device)
        model = cls(config=ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"BindFM loaded ← {path}")
        return model
