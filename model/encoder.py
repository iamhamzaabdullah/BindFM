"""
BindFM Equivariant Atom Encoder
================================
SE(3)-equivariant message passing encoder based on EGNN
(Satorras, Hoogeboom, Welling — ICML 2021).

Equivariance guarantee:
  - Scalar features h_i are INVARIANT under rotation / translation
  - Coordinates x_i transform COVARIANTLY under rotation / translation
  - Messages use only ||x_i - x_j|| (invariant) and unit vectors (equivariant)
  - weight * unit_vec  =>  invariant scalar * equivariant vector = equivariant

Output interface (used by trunk.py and bindfm.py):
  h: [N, D_OUT]   per-atom invariant scalar embeddings
  x: [N, 3]       per-atom equivariant coordinates

No phantom vector channel. No D_OUT_VECTOR. Clean interface throughout.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from model.tokenizer import ATOM_FEAT_DIM, BOND_FEAT_DIM

# ── Module-level dimension constants (imported by trunk.py, bindfm.py) ────────
D_HIDDEN:  int   = 256
D_EDGE:    int   = 128
D_OUT:     int   = 512
N_RBF:     int   = 64
CUTOFF:    float = 8.0
N_LAYERS:  int   = 8
DROPOUT:   float = 0.1


class RadialBasisExpansion(nn.Module):
    """Expand scalar distances into N_RBF Gaussian basis functions (fixed)."""

    def __init__(self, n_rbf: int = N_RBF, cutoff: float = CUTOFF):
        super().__init__()
        centers = torch.linspace(0.1, cutoff, n_rbf)
        widths  = torch.full((n_rbf,), (cutoff / n_rbf) ** 2)
        self.register_buffer("centers", centers)
        self.register_buffer("widths",  widths)

    def forward(self, dist: Tensor) -> Tensor:
        """dist: [E] -> [E, n_rbf]"""
        d  = dist.unsqueeze(-1)
        mu = self.centers.unsqueeze(0)
        w  = self.widths.unsqueeze(0)
        return torch.exp(-((d - mu) ** 2) / w.clamp(min=1e-8))


class CosineCutoff(nn.Module):
    """Smooth C-inf envelope that zeros contributions beyond cutoff."""

    def __init__(self, cutoff: float = CUTOFF):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: Tensor) -> Tensor:
        """dist: [E] -> [E] in [0,1]"""
        env = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)
        return env * (dist < self.cutoff).to(dist.dtype)


class AtomEmbedding(nn.Module):
    """Project 197-dim atom feature vector into d_hidden hidden space."""

    def __init__(self, in_dim: int = ATOM_FEAT_DIM, out_dim: int = D_HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeNetwork(nn.Module):
    """
    Compute edge embeddings from distance RBFs, bond features, and atom states.
    All inputs are invariant scalars.
    """

    def __init__(self, d_hidden: int = D_HIDDEN, d_edge: int = D_EDGE,
                 n_rbf: int = N_RBF, bond_feat_dim: int = BOND_FEAT_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        in_dim = n_rbf + 1 + bond_feat_dim + 2 * d_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_edge * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_edge * 2, d_edge),
            nn.SiLU(),
        )

    def forward(self, rbf: Tensor, envelope: Tensor, bond_feats: Tensor,
                h_src: Tensor, h_dst: Tensor) -> Tensor:
        inp = torch.cat([rbf, envelope.unsqueeze(-1), bond_feats, h_src, h_dst], dim=-1)
        return self.net(inp)


class EGNNLayer(nn.Module):
    """
    One round of SE(3)-equivariant EGNN message passing.

    SCALAR UPDATE (invariant):
        e_ij  = EdgeNetwork(dist, bond, h_src, h_dst)
        m_ij  = MsgNet(e_ij) * envelope(dist)
        agg_i = sum_j m_ij
        h_i'  = LayerNorm(h_i + UpdateNet(h_i, agg_i))

    COORD UPDATE (equivariant):
        w_ij  = CoordNet(m_ij)                         <- invariant scalar
        delta = w_ij * (x_i - x_j)/||x_i - x_j||      <- equivariant
        x_i'  = x_i + mean_j(delta_ij)

    Proof: distances invariant under SE(3), unit vector equivariant,
           product invariant*equivariant = equivariant. QED.
    """

    def __init__(self, d_hidden: int = D_HIDDEN, d_edge: int = D_EDGE,
                 n_rbf: int = N_RBF, bond_feat_dim: int = BOND_FEAT_DIM,
                 cutoff: float = CUTOFF, dropout: float = DROPOUT,
                 update_coords: bool = True):
        super().__init__()
        self.d_hidden      = d_hidden
        self.update_coords = update_coords

        self.rbf      = RadialBasisExpansion(n_rbf, cutoff)
        self.envelope = CosineCutoff(cutoff)
        self.edge_net = EdgeNetwork(d_hidden, d_edge, n_rbf, bond_feat_dim, dropout)

        self.msg_net = nn.Sequential(
            nn.Linear(d_edge, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
        )
        self.norm_h = nn.LayerNorm(d_hidden)

        if update_coords:
            self.coord_net = nn.Sequential(
                nn.Linear(d_hidden, d_hidden // 2),
                nn.SiLU(),
                nn.Linear(d_hidden // 2, 1),
                nn.Tanh(),
            )

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor,
                edge_feats: Tensor, atom_mask: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Tensor]:
        N   = h.shape[0]
        src = edge_index[0]
        dst = edge_index[1]

        diff     = x[src] - x[dst]
        dist     = diff.norm(dim=-1).clamp(min=1e-8)
        unit_vec = diff / dist.unsqueeze(-1)

        rbf      = self.rbf(dist)
        envelope = self.envelope(dist)

        e_ij = self.edge_net(rbf, envelope, edge_feats, h[src], h[dst])
        m_ij = self.msg_net(e_ij) * envelope.unsqueeze(-1)

        agg = h.new_zeros(N, self.d_hidden)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)

        h_new = self.norm_h(h + self.update_net(torch.cat([h, agg], dim=-1)))

        if self.update_coords:
            w         = self.coord_net(m_ij) * envelope.unsqueeze(-1)
            delta     = w * unit_vec
            coord_upd = x.new_zeros(N, 3)
            n_nbrs    = x.new_zeros(N)
            coord_upd.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), delta)
            n_nbrs.scatter_add_(0, dst, torch.ones(dst.shape[0], device=dst.device))
            x_new = x + coord_upd / n_nbrs.clamp(min=1.0).unsqueeze(-1)
        else:
            x_new = x

        if atom_mask is not None:
            m     = atom_mask.unsqueeze(-1).to(h.dtype)
            h_new = h_new * m
            x_new = x_new * m

        return h_new, x_new


class AtomEncoder(nn.Module):
    """
    Full SE(3)-equivariant encoder: N EGNN layers on the atom graph.
    Augments covalent bond graph with spatial radius-graph edges.

    Returns:
        h: [N, D_OUT]   per-atom invariant embeddings
        x: [N, 3]       per-atom equivariant coordinates
    """

    def __init__(self, atom_feat_dim: int = ATOM_FEAT_DIM,
                 d_hidden: int = D_HIDDEN, d_edge: int = D_EDGE,
                 d_out: int = D_OUT, n_rbf: int = N_RBF,
                 n_layers: int = N_LAYERS, cutoff: float = CUTOFF,
                 bond_feat_dim: int = BOND_FEAT_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.cutoff = cutoff

        self.embed = AtomEmbedding(atom_feat_dim, d_hidden, dropout)

        self.layers = nn.ModuleList([
            EGNNLayer(
                d_hidden      = d_hidden,
                d_edge        = d_edge,
                n_rbf         = n_rbf,
                bond_feat_dim = bond_feat_dim,
                cutoff        = cutoff,
                dropout       = dropout,
                update_coords = (i < n_layers - 1),
            )
            for i in range(n_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out * 2),
            nn.SiLU(),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

    def _add_spatial_edges(self, coords: Tensor, edge_index: Tensor,
                           edge_feats: Tensor,
                           atom_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """Add non-covalent spatial neighbor edges within cutoff radius."""
        N             = coords.shape[0]
        bond_feat_dim = edge_feats.shape[-1] if edge_feats.shape[0] > 0 else BOND_FEAT_DIM
        device        = coords.device
        dtype         = coords.dtype

        if N > 2000:
            return edge_index, edge_feats

        d      = (coords.unsqueeze(0) - coords.unsqueeze(1)).norm(dim=-1)
        within = (d < self.cutoff) & (d > 1e-4)

        if atom_mask is not None:
            within = within & (atom_mask.unsqueeze(0) & atom_mask.unsqueeze(1))

        sp_src, sp_dst = torch.where(within)
        if sp_src.shape[0] == 0:
            return edge_index, edge_feats

        sp_idx   = torch.stack([sp_src, sp_dst], dim=0)
        sp_feats = torch.zeros(sp_src.shape[0], bond_feat_dim, device=device, dtype=dtype)

        # Existing covalent edges get zero bond features in this combined tensor
        if edge_feats.shape[0] > 0:
            all_idx   = torch.cat([edge_index, sp_idx],   dim=1)
            all_feats = torch.cat([edge_feats, sp_feats], dim=0)
        else:
            all_idx   = sp_idx
            all_feats = sp_feats

        return all_idx, all_feats

    def forward(self, atom_feats: Tensor, edge_index: Tensor,
                edge_feats: Tensor, coords: Optional[Tensor] = None,
                atom_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        h = self.embed(atom_feats)
        x = coords.clone().float() if coords is not None \
            else atom_feats.new_zeros(atom_feats.shape[0], 3)

        if coords is not None and edge_index.shape[1] >= 0:
            edge_index, edge_feats = self._add_spatial_edges(
                x, edge_index, edge_feats, atom_mask
            )

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_feats, atom_mask)

        return self.out_proj(h), x


class DualEntityEncoder(nn.Module):
    """
    Encode both binding partners through THE SAME AtomEncoder.
    Shared weights = shared embedding space = universal binding.

    Returns: h_a [N_a, D_OUT], x_a [N_a, 3], h_b [N_b, D_OUT], x_b [N_b, 3]
    """

    def __init__(self, atom_feat_dim: int = ATOM_FEAT_DIM,
                 d_hidden: int = D_HIDDEN, d_edge: int = D_EDGE,
                 d_out: int = D_OUT, n_rbf: int = N_RBF,
                 n_layers: int = N_LAYERS, cutoff: float = CUTOFF,
                 bond_feat_dim: int = BOND_FEAT_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = AtomEncoder(
            atom_feat_dim=atom_feat_dim, d_hidden=d_hidden, d_edge=d_edge,
            d_out=d_out, n_rbf=n_rbf, n_layers=n_layers, cutoff=cutoff,
            bond_feat_dim=bond_feat_dim, dropout=dropout,
        )
        self.d_out = d_out

    def forward(self, a_atom_feats: Tensor, a_edge_index: Tensor,
                a_edge_feats: Tensor, a_coords: Optional[Tensor] = None,
                a_atom_mask: Optional[Tensor] = None,
                b_atom_feats: Optional[Tensor] = None,
                b_edge_index: Optional[Tensor] = None,
                b_edge_feats: Optional[Tensor] = None,
                b_coords: Optional[Tensor] = None,
                b_atom_mask: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        h_a, x_a = self.encoder(a_atom_feats, a_edge_index, a_edge_feats,
                                  a_coords, a_atom_mask)

        if b_atom_feats is not None and b_edge_index is not None \
                and b_edge_feats is not None:
            h_b, x_b = self.encoder(b_atom_feats, b_edge_index, b_edge_feats,
                                     b_coords, b_atom_mask)
        else:
            h_b = a_atom_feats.new_zeros(0, self.d_out)
            x_b = a_atom_feats.new_zeros(0, 3)

        return h_a, x_a, h_b, x_b

    @property
    def output_dim(self) -> int:
        return self.d_out
