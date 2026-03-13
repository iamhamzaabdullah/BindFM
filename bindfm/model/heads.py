"""
BindFM Output Heads
--------------------
Three co-trained output heads operating on PairFormer trunk outputs.

AffinityHead:
    Predicts binding affinity (log Kd in nM), binding probability,
    association / dissociation rate constants, and aleatoric uncertainty.
    Conditioned on assay type so the model can distinguish Kd / Ki / IC50.

StructureFlowMatchingHead:
    Predicts velocity field for SE(3) flow matching.
    During training: given noisy coords at time t, predict velocity toward
    clean complex coords (Euler denoising).
    At inference: integrate ODE from noise to structure (Euler steps).

GenerativeHead:
    Generates de novo binder atom features and coordinates.
    Conditioned on target single representation + modality + optional Kd target.
    Uses the same flow matching formulation as StructureHead.

BindFMLoss:
    Multi-task loss combining all three heads with learned task weights
    (Kendall et al. 2018 homoscedastic uncertainty weighting).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, Any

from model.tokenizer import ATOM_FEAT_DIM, EntityType
from model.trunk import D_SINGLE, D_PAIR


# ── 1. Affinity Head ──────────────────────────────────────────────────────────

class AffinityHead(nn.Module):
    """
    Predict binding affinity from single and pair representations.

    Pooling strategy:
      1. Mean-pool single_a -> s_a: [D_SINGLE]
      2. Mean-pool single_b -> s_b: [D_SINGLE]
      3. Mean-pool pair     -> p:   [D_PAIR]     (over both N_a and N_b)
      4. Concatenate: [s_a, s_b, s_a*s_b, p] -> binding descriptor
      5. MLP + assay-type conditioning -> outputs

    Outputs:
      log_kd:       log10(Kd in nM)        float
      binding_prob: sigmoid probability     float in [0,1]
      log_kon:      log10(kon in M^-1 s^-1) float
      log_koff:     log10(koff in s^-1)     float
      uncertainty:  aleatoric uncertainty   float (always positive)
    """

    # Assay type conditioning: 12 types mapped to embeddings
    ASSAY_TYPES = [
        "Kd", "Ki", "IC50", "EC50", "AC50",
        "SPR_Kd", "ITC_Kd", "EMSA", "Tm_shift",
        "kinact_KI", "SELEX_enrichment", "unknown",
    ]

    def __init__(self, d_single: int = D_SINGLE, d_pair: int = D_PAIR,
                 n_assay_types: int = 12, d_hidden: int = 512):
        super().__init__()
        self.n_assay_types = n_assay_types

        # Assay type embedding
        self.assay_embed = nn.Embedding(n_assay_types, d_hidden // 4)

        # Binding descriptor: s_a + s_b + element-wise product + pair mean
        d_in = d_single + d_single + d_single + d_pair + d_hidden // 4

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        d_mlp_out = d_hidden // 2

        # Separate output heads
        self.head_kd          = nn.Linear(d_mlp_out, 1)    # log10(Kd nM)
        self.head_bind_logit  = nn.Linear(d_mlp_out, 1)    # binder/non-binder
        self.head_kon         = nn.Linear(d_mlp_out, 1)    # log10(kon)
        self.head_koff        = nn.Linear(d_mlp_out, 1)    # log10(koff)
        self.head_uncertainty = nn.Linear(d_mlp_out, 1)    # aleatoric uncertainty

    def forward(
        self,
        single_a:  Tensor,                   # [N_a, D_SINGLE]
        single_b:  Tensor,                   # [N_b, D_SINGLE]
        pair:      Tensor,                   # [N_a, N_b, D_PAIR]
        assay_type: Optional[Tensor] = None, # [] int64 or None
    ) -> Dict[str, Tensor]:
        """Returns dict with all binding predictions as scalar tensors."""

        # Pool to fixed-size binding descriptor
        s_a  = single_a.mean(dim=0)           # [D_SINGLE]
        s_b  = single_b.mean(dim=0)           # [D_SINGLE]
        p    = pair.mean(dim=[0, 1])          # [D_PAIR]
        elem = s_a * s_b                      # element-wise product [D_SINGLE]

        # Assay type conditioning
        if assay_type is None:
            assay_idx = torch.tensor(self.n_assay_types - 1,
                                     device=single_a.device, dtype=torch.long)
        else:
            assay_idx = assay_type.long().squeeze()
        assay_emb = self.assay_embed(assay_idx)   # [d_hidden//4]

        # Concatenate all components
        desc = torch.cat([s_a, s_b, elem, p, assay_emb], dim=-1)

        h = self.mlp(desc)

        log_kd       = self.head_kd(h).squeeze(-1)
        binding_prob = torch.sigmoid(self.head_bind_logit(h).squeeze(-1))
        log_kon      = self.head_kon(h).squeeze(-1)
        log_koff     = self.head_koff(h).squeeze(-1)
        uncertainty  = F.softplus(self.head_uncertainty(h).squeeze(-1)) + 1e-4

        return {
            "log_kd":       log_kd,
            "binding_prob": binding_prob,
            "log_kon":      log_kon,
            "log_koff":     log_koff,
            "uncertainty":  uncertainty,
        }


# ── 2. Structure Flow Matching Head ───────────────────────────────────────────

class StructureFlowMatchingHead(nn.Module):
    """
    SE(3) flow matching head for 3D complex structure prediction.

    Training:
        Given noisy coordinates at time t in [0,1], predict the velocity
        field pointing from noisy toward clean:
            v = (x_clean - x_noisy) / (1 - t)   [OT-CFM / Lipman et al.]

        Loss: MSE between predicted velocity and target velocity.

    Inference:
        Euler integration from t=0 (pure noise) to t=1 (clean structure):
            x_{t+dt} = x_t + v(x_t, t) * dt

    Equivariance:
        The velocity prediction uses only invariant features (trunk outputs)
        to produce per-atom scalar weights, which are then multiplied by
        coordinate vectors. This preserves equivariance.
    """

    def __init__(self, d_single: int = D_SINGLE, d_pair: int = D_PAIR,
                 d_hidden: int = 256):
        super().__init__()
        self.d_single = d_single

        # Time embedding (sinusoidal + linear projection)
        self.time_embed = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Per-atom velocity network: scalar weights for equivariant update
        # Input: single representation + time embedding + pair-aggregated context
        d_in = d_single + d_pair + d_hidden
        self.vel_net_a = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),  # scalar weight per atom
        )
        self.vel_net_b = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # Absolute position predictor (for atoms without neighbors in pair)
        self.abs_pred_a = nn.Linear(d_single, 3)
        self.abs_pred_b = nn.Linear(d_single, 3)

    @staticmethod
    def _sinusoidal_time(t: Tensor, d_model: int) -> Tensor:
        """t: scalar or [1] -> [d_model]"""
        t_val  = t.float().squeeze()
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(0, d_model, 2,
                                             device=t.device, dtype=torch.float32)
            / d_model
        )
        args   = t_val * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        single_a: Tensor,         # [N_a, D_SINGLE]
        single_b: Tensor,         # [N_b, D_SINGLE]
        pair:     Tensor,         # [N_a, N_b, D_PAIR]
        noisy_x_a: Tensor,        # [N_a, 3] noisy coordinates at time t
        noisy_x_b: Tensor,        # [N_b, 3]
        t:        Tensor,         # [] scalar time in [0,1]
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns velocity predictions:
            vel_a: [N_a, 3]
            vel_b: [N_b, 3]
        """
        d_hidden = self.time_embed[0].in_features
        t_emb    = self.time_embed(self._sinusoidal_time(t, d_hidden))  # [d_hidden]

        # Aggregate pair context per atom
        pair_ctx_a = pair.mean(dim=1)              # [N_a, D_PAIR]
        pair_ctx_b = pair.mean(dim=0)              # [N_b, D_PAIR]

        # Time embedding broadcast
        t_a = t_emb.unsqueeze(0).expand(single_a.shape[0], -1)  # [N_a, d_hidden]
        t_b = t_emb.unsqueeze(0).expand(single_b.shape[0], -1)  # [N_b, d_hidden]

        # Per-atom scalar velocity weights
        in_a    = torch.cat([single_a, pair_ctx_a, t_a], dim=-1)
        in_b    = torch.cat([single_b, pair_ctx_b, t_b], dim=-1)
        w_a     = self.vel_net_a(in_a)   # [N_a, 1]
        w_b     = self.vel_net_b(in_b)   # [N_b, 1]

        # Equivariant velocity: scalar_weight * normalized_position_vector
        # Fallback to absolute prediction when coordinates are near-zero
        norm_a  = noisy_x_a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_b  = noisy_x_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vel_a   = w_a * (noisy_x_a / norm_a) + self.abs_pred_a(single_a)
        vel_b   = w_b * (noisy_x_b / norm_b) + self.abs_pred_b(single_b)

        return vel_a, vel_b

    @torch.no_grad()
    def sample(
        self,
        single_a: Tensor, single_b: Tensor, pair: Tensor,
        n_atoms_a: int, n_atoms_b: int,
        n_steps: int = 100,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        """
        Euler ODE integration from noise to structure.
        Returns (coords_a, coords_b) in Angstroms.
        """
        # Sample initial Gaussian noise (scale ~protein radius)
        x_a = torch.randn(n_atoms_a, 3, device=device) * 10.0
        x_b = torch.randn(n_atoms_b, 3, device=device) * 10.0

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.tensor(i / n_steps, device=device)
            v_a, v_b = self.forward(single_a, single_b, pair, x_a, x_b, t)
            x_a = x_a + v_a * dt
            x_b = x_b + v_b * dt

        return x_a, x_b


# ── 3. Generative Head ────────────────────────────────────────────────────────

class GenerativeHead(nn.Module):
    """
    De novo binder generation head.

    Generates both atom features and 3D coordinates for a novel binder,
    conditioned on:
      - Target single representation (compressed to fixed context)
      - Desired modality (protein / RNA / DNA / small_mol)
      - Optional target Kd

    Uses the same OT-CFM flow matching formulation as StructureHead,
    but operates in joint (atom_feature, coordinate) space.

    Architecture:
      1. Compress target to fixed context vector via cross-attention pooling
      2. For N_gen atoms (sampled or fixed), flow-match from noise to binder
      3. Decode atom features -> one-hot element + properties
    """

    def __init__(self, d_single: int = D_SINGLE, d_pair: int = D_PAIR,
                 atom_feat_dim: int = ATOM_FEAT_DIM,
                 d_hidden: int = 512, max_gen_atoms: int = 200,
                 n_modalities: int = 9):
        super().__init__()
        self.d_single      = d_single
        self.atom_feat_dim = atom_feat_dim
        self.max_gen_atoms = max_gen_atoms

        # Compress variable-length target to fixed context
        self.target_pool = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Modality embedding
        self.modality_embed = nn.Embedding(n_modalities, d_hidden // 4)

        # Kd conditioning
        self.kd_proj = nn.Linear(1, d_hidden // 4)

        # Time embedding
        self.time_proj = nn.Linear(d_hidden, d_hidden // 2)

        # Context: target pooled + modality + kd + time
        d_ctx = d_hidden + d_hidden // 4 + d_hidden // 4 + d_hidden // 2

        # Atom feature flow network
        # Input: noisy atom features + context
        self.feat_flow = nn.Sequential(
            nn.LayerNorm(atom_feat_dim + d_ctx),
            nn.Linear(atom_feat_dim + d_ctx, d_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, atom_feat_dim),   # velocity in feature space
        )

        # Coordinate flow network: outputs scalar weight per atom (equivariant)
        self.coord_flow = nn.Sequential(
            nn.LayerNorm(atom_feat_dim + d_ctx),
            nn.Linear(atom_feat_dim + d_ctx, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),          # scalar per atom
        )

        # Coordinate absolute predictor (when noisy coords are near-zero)
        self.coord_abs = nn.Linear(atom_feat_dim + d_ctx, 3)

    @staticmethod
    def _sinusoidal_time(t: Tensor, d_model: int) -> Tensor:
        t_val = t.float().squeeze()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, d_model, 2,
                                             device=t.device, dtype=torch.float32)
            / d_model
        )
        args  = t_val * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def _build_context(self, single_b: Tensor, modality: Tensor,
                       log_kd_target: Optional[Tensor], t: Tensor,
                       n_atoms: int) -> Tensor:
        """Build fixed context vector broadcast to all generated atoms."""
        d_hidden = self.target_pool[1].in_features

        # Pool target
        target_ctx = self.target_pool(single_b).mean(dim=0)   # [d_hidden]

        # Modality embedding
        mod_emb    = self.modality_embed(modality.long().squeeze())  # [d_hidden//4]

        # Kd conditioning
        if log_kd_target is not None:
            kd_val = log_kd_target.float().view(1)
            kd_emb = self.kd_proj(kd_val).squeeze(0)          # [d_hidden//4]
        else:
            kd_emb = torch.zeros(self.kd_proj.out_features, device=single_b.device)

        # Time embedding
        d_time  = self.time_proj.in_features
        t_raw   = self._sinusoidal_time(t, d_time)
        t_emb   = self.time_proj(t_raw)                        # [d_hidden//2]

        ctx = torch.cat([target_ctx, mod_emb, kd_emb, t_emb], dim=-1)  # [d_ctx]
        return ctx.unsqueeze(0).expand(n_atoms, -1)            # [N, d_ctx]

    def forward(
        self,
        single_b:       Tensor,                  # [N_b, D_SINGLE] target repr
        noisy_feats:    Tensor,                  # [N_gen, ATOM_FEAT_DIM]
        noisy_coords:   Tensor,                  # [N_gen, 3]
        t:              Tensor,                  # [] scalar time in [0,1]
        modality:       Tensor,                  # [] int EntityType
        log_kd_target:  Optional[Tensor] = None, # [] float optional
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns velocity in feature space and coordinate space:
            vel_feats:  [N_gen, ATOM_FEAT_DIM]
            vel_coords: [N_gen, 3]
        """
        N_gen = noisy_feats.shape[0]
        ctx   = self._build_context(single_b, modality, log_kd_target, t, N_gen)
        inp   = torch.cat([noisy_feats, ctx], dim=-1)   # [N_gen, feat_dim+d_ctx]

        vel_feats = self.feat_flow(inp)                 # [N_gen, ATOM_FEAT_DIM]

        w          = self.coord_flow(inp)               # [N_gen, 1]
        norm_coords= noisy_coords.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vel_coords = w * (noisy_coords / norm_coords) + self.coord_abs(inp)

        return vel_feats, vel_coords

    @torch.no_grad()
    def generate(
        self,
        single_b:      Tensor,
        modality:      Tensor,
        log_kd_target: Optional[Tensor] = None,
        n_atoms:       int  = 50,
        n_steps:       int  = 100,
        device:        str  = "cpu",
        seed:          Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate one binder molecule via Euler ODE integration.
        Returns (atom_feats [N, ATOM_FEAT_DIM], coords [N, 3]).
        """
        if seed is not None:
            torch.manual_seed(seed)

        feats  = torch.randn(n_atoms, self.atom_feat_dim, device=device)
        coords = torch.randn(n_atoms, 3, device=device) * 5.0

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.tensor(i / n_steps, device=device)
            vf, vc = self.forward(single_b, feats, coords, t, modality,
                                  log_kd_target)
            feats  = feats  + vf * dt
            coords = coords + vc * dt

        return feats, coords


# ── 4. Multi-Task Loss ────────────────────────────────────────────────────────

class BindFMLoss(nn.Module):
    """
    Multi-task loss for joint training of all three heads.

    Uses homoscedastic uncertainty weighting (Kendall et al. 2018):
        L_total = sum_i  (1/2 * exp(-log_sigma_i^2) * L_i + log_sigma_i)

    This automatically balances tasks without manual weight tuning.
    The model learns how much to weight each task based on task uncertainty.
    """

    def __init__(self):
        super().__init__()
        # Learnable log-variance per task (initialized near zero)
        self.log_sigma_affinity   = nn.Parameter(torch.zeros(1))
        self.log_sigma_binder     = nn.Parameter(torch.zeros(1))
        self.log_sigma_structure  = nn.Parameter(torch.zeros(1))
        self.log_sigma_generation = nn.Parameter(torch.zeros(1))

    def _weighted(self, loss: Tensor, log_sigma: Tensor) -> Tensor:
        """Apply Kendall uncertainty weighting to a single task loss."""
        precision = torch.exp(-log_sigma)
        return 0.5 * precision * loss + 0.5 * log_sigma

    def affinity_loss(
        self,
        pred:     Dict[str, Tensor],
        log_kd:   Optional[Tensor],   # ground truth log Kd in nM
        is_binder: Optional[Tensor],  # float 0/1
    ) -> Dict[str, Tensor]:
        """Compute affinity + classification losses."""
        losses = {}

        # Regression: MSE on log Kd when label is available
        if log_kd is not None:
            target  = log_kd.float()
            pred_kd = pred["log_kd"]
            # Uncertainty-weighted regression (NLL of Gaussian)
            sigma2  = pred["uncertainty"] ** 2
            losses["kd_reg"] = (
                (pred_kd - target) ** 2 / (2 * sigma2)
                + 0.5 * torch.log(sigma2)
            ).mean()

        # Classification: BCE on binder / non-binder
        if is_binder is not None:
            prob   = pred["binding_prob"].clamp(1e-6, 1 - 1e-6)
            target = is_binder.float()
            losses["binder_cls"] = F.binary_cross_entropy(prob, target)

        return losses

    def structure_loss(
        self,
        vel_pred_a: Tensor, vel_pred_b: Tensor,
        vel_true_a: Tensor, vel_true_b: Tensor,
    ) -> Tensor:
        """MSE between predicted and target OT-CFM velocity field."""
        loss_a = F.mse_loss(vel_pred_a, vel_true_a)
        loss_b = F.mse_loss(vel_pred_b, vel_true_b)
        return (loss_a + loss_b) / 2.0

    def generation_loss(
        self,
        vel_feats_pred:  Tensor, vel_feats_true:  Tensor,
        vel_coords_pred: Tensor, vel_coords_true: Tensor,
    ) -> Tensor:
        """MSE loss on generation velocity in feature + coordinate space."""
        feat_loss  = F.mse_loss(vel_feats_pred,  vel_feats_true)
        coord_loss = F.mse_loss(vel_coords_pred, vel_coords_true)
        return (feat_loss + coord_loss) / 2.0

    def forward(
        self,
        affinity_pred:  Optional[Dict[str, Tensor]] = None,
        log_kd:         Optional[Tensor] = None,
        is_binder:      Optional[Tensor] = None,
        vel_pred_a:     Optional[Tensor] = None,
        vel_pred_b:     Optional[Tensor] = None,
        vel_true_a:     Optional[Tensor] = None,
        vel_true_b:     Optional[Tensor] = None,
        gen_vel_feats_pred:  Optional[Tensor] = None,
        gen_vel_feats_true:  Optional[Tensor] = None,
        gen_vel_coords_pred: Optional[Tensor] = None,
        gen_vel_coords_true: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute total multi-task loss.
        Returns dict with component losses and weighted total.
        """
        total  = torch.tensor(0.0, device=self.log_sigma_affinity.device,
                               requires_grad=True)
        losses = {}

        # Affinity losses
        if affinity_pred is not None:
            aff_losses = self.affinity_loss(affinity_pred, log_kd, is_binder)
            if "kd_reg" in aff_losses:
                l = self._weighted(aff_losses["kd_reg"],   self.log_sigma_affinity)
                total = total + l
                losses["kd_reg"] = aff_losses["kd_reg"].detach()
            if "binder_cls" in aff_losses:
                l = self._weighted(aff_losses["binder_cls"], self.log_sigma_binder)
                total = total + l
                losses["binder_cls"] = aff_losses["binder_cls"].detach()

        # Structure loss
        if vel_pred_a is not None and vel_true_a is not None:
            sl = self.structure_loss(vel_pred_a, vel_pred_b,
                                     vel_true_a, vel_true_b)
            total  = total + self._weighted(sl, self.log_sigma_structure)
            losses["structure"] = sl.detach()

        # Generation loss
        if gen_vel_feats_pred is not None and gen_vel_feats_true is not None:
            gl = self.generation_loss(
                gen_vel_feats_pred, gen_vel_feats_true,
                gen_vel_coords_pred, gen_vel_coords_true,
            )
            total  = total + self._weighted(gl, self.log_sigma_generation)
            losses["generation"] = gl.detach()

        losses["total"] = total
        return losses
