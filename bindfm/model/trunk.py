"""
BindFM PairFormer Trunk
-----------------------
The binding interaction trunk. Takes per-atom embeddings for both entities
and learns what binding IS — not what molecules are.

Operates at ATOM level (not residue-level, not token-level).

Architecture per block:
  - Outer product mean: single → pair
  - Triangle multiplicative update (outgoing + incoming)
  - Triangle self-attention (row + column)
  - Single representation transition (FFN)

Input from encoder:
  h_a: [N_a, D_OUT]   entity A invariant embeddings
  x_a: [N_a, 3]       entity A equivariant coordinates (used only for RBF init)
  h_b: [N_b, D_OUT]   entity B invariant embeddings
  x_b: [N_b, 3]       entity B equivariant coordinates

Output:
  single_a: [N_a, D_SINGLE]
  single_b: [N_b, D_SINGLE]
  pair:     [N_a, N_b, D_PAIR]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from model.encoder import D_OUT

# ── Constants ─────────────────────────────────────────────────────────────────
D_SINGLE       = 512
D_PAIR         = 128
N_TRUNK_LAYERS = 32
N_HEADS_TRI    = 4
N_RBF_DIST     = 64


# ── 1. Single Representation Initialization ───────────────────────────────────

class SingleInitProjection(nn.Module):
    """
    Project encoder scalar outputs h [N, D_OUT] into trunk single repr [N, D_SINGLE].
    Also incorporates coordinate norms as additional invariant features.
    """

    def __init__(self, d_in: int = D_OUT, d_out: int = D_SINGLE):
        super().__init__()
        # d_in + 1 for coordinate norm (|x_i|, an invariant feature)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in + 1),
            nn.Linear(d_in + 1, d_out * 2),
            nn.SiLU(),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """
        h: [N, D_OUT]
        x: [N, 3]
        Returns: [N, D_SINGLE]
        """
        coord_norm = x.norm(dim=-1, keepdim=True)   # [N, 1] invariant
        inp = torch.cat([h, coord_norm], dim=-1)    # [N, D_OUT+1]
        return self.proj(inp)


# ── 2. Pair Representation Initialization ────────────────────────────────────

class PairInitProjection(nn.Module):
    """
    Initialize pair representation [N_a, N_b, D_PAIR] from:
      1. Low-rank outer product of single representations
      2. RBF-encoded pairwise distances (when coordinates available)
    """

    def __init__(self, d_single: int = D_SINGLE, d_pair: int = D_PAIR,
                 rank: int = 32, n_rbf: int = N_RBF_DIST):
        super().__init__()
        self.rank = rank

        self.left_proj  = nn.Linear(d_single, rank)
        self.right_proj = nn.Linear(d_single, rank)
        self.outer_proj = nn.Linear(rank * rank, d_pair)

        self.dist_rbf_centers = None  # registered as buffer in __init__
        self._n_rbf = n_rbf
        self.dist_proj = nn.Linear(n_rbf, d_pair)

        self.norm = nn.LayerNorm(d_pair)

        # Register RBF centers as buffer
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.5, 20.0, n_rbf)
        )

    def _rbf(self, dist: Tensor) -> Tensor:
        """dist: [N_a, N_b] -> [N_a, N_b, n_rbf]"""
        mu    = self.rbf_centers.view(1, 1, self._n_rbf)
        sigma = (20.0 / self._n_rbf) ** 2
        return torch.exp(-((dist.unsqueeze(-1) - mu) ** 2) / sigma)

    def forward(self, single_a: Tensor, single_b: Tensor,
                x_a: Optional[Tensor] = None,
                x_b: Optional[Tensor] = None) -> Tensor:
        """
        single_a: [N_a, D_SINGLE]
        single_b: [N_b, D_SINGLE]
        x_a:      [N_a, 3]  optional
        x_b:      [N_b, 3]  optional
        Returns:  [N_a, N_b, D_PAIR]
        """
        N_a, N_b = single_a.shape[0], single_b.shape[0]

        # Low-rank outer product
        left  = self.left_proj(single_a)   # [N_a, rank]
        right = self.right_proj(single_b)  # [N_b, rank]

        # For each (i, j): outer(left[i], right[j]) -> [rank, rank] -> flatten
        outer = (left.unsqueeze(1).unsqueeze(3) *
                 right.unsqueeze(0).unsqueeze(2))   # [N_a, N_b, rank, rank]
        outer = outer.reshape(N_a, N_b, self.rank * self.rank)
        pair  = self.outer_proj(outer)              # [N_a, N_b, D_PAIR]

        # Add distance embedding when coordinates are available
        if x_a is not None and x_b is not None:
            # [N_a, N_b] pairwise distances
            dist  = torch.cdist(x_a, x_b)          # [N_a, N_b]
            rbf   = self._rbf(dist)                 # [N_a, N_b, n_rbf]
            pair  = pair + self.dist_proj(rbf)

        return self.norm(pair)


# ── 3. Triangle Multiplicative Update ─────────────────────────────────────────

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Triangle multiplicative update from AlphaFold2.

    Outgoing:  pair(i,j) += Norm( Σ_k  a(i,k) ⊙ b(k,j) )
    Incoming:  pair(i,j) += Norm( Σ_k  a(k,i) ⊙ b(k,j) )

    Gated linear transformations on both factors.
    """

    def __init__(self, d_pair: int = D_PAIR, direction: str = "outgoing"):
        super().__init__()
        assert direction in ("outgoing", "incoming")
        self.direction = direction

        self.norm_in    = nn.LayerNorm(d_pair)
        self.left_proj  = nn.Linear(d_pair, d_pair)
        self.left_gate  = nn.Linear(d_pair, d_pair)
        self.right_proj = nn.Linear(d_pair, d_pair)
        self.right_gate = nn.Linear(d_pair, d_pair)
        self.center_norm= nn.LayerNorm(d_pair)
        self.out_proj   = nn.Linear(d_pair, d_pair)
        self.out_gate   = nn.Linear(d_pair, d_pair)
        self.norm_out   = nn.LayerNorm(d_pair)

    def forward(self, pair: Tensor) -> Tensor:
        """pair: [N_a, N_b, D_PAIR] -> [N_a, N_b, D_PAIR]"""
        z = self.norm_in(pair)

        left  = torch.sigmoid(self.left_gate(z))  * self.left_proj(z)
        right = torch.sigmoid(self.right_gate(z)) * self.right_proj(z)

        if self.direction == "outgoing":
            # pair(i,j) uses atoms k as intermediary: left[i,k], right[j,k]
            tri = torch.einsum("ikd,jkd->ijd", left, right)
        else:
            # incoming: left[k,i], right[k,j]
            tri = torch.einsum("kid,kjd->ijd", left, right)

        tri = self.center_norm(tri)
        tri = torch.sigmoid(self.out_gate(pair)) * self.out_proj(tri)

        return self.norm_out(pair + tri)


# ── 4. Triangle Self-Attention ────────────────────────────────────────────────

class TriangleAttention(nn.Module):
    """
    Row-wise or column-wise triangle self-attention on the pair matrix.

    Row-wise:    for each row i, attend over all j with pair[i,:] as queries
    Column-wise: for each column j, attend over all i with pair[:,j] as queries
    """

    def __init__(self, d_pair: int = D_PAIR, n_heads: int = N_HEADS_TRI,
                 axis: str = "row"):
        super().__init__()
        assert axis in ("row", "column")
        assert d_pair % n_heads == 0, \
            f"d_pair ({d_pair}) must be divisible by n_heads ({n_heads})"
        self.axis   = axis
        self.n_heads = n_heads
        self.d_head  = d_pair // n_heads

        self.norm     = nn.LayerNorm(d_pair)
        self.q_proj   = nn.Linear(d_pair, d_pair, bias=False)
        self.k_proj   = nn.Linear(d_pair, d_pair, bias=False)
        self.v_proj   = nn.Linear(d_pair, d_pair, bias=False)
        self.bias_proj= nn.Linear(d_pair, n_heads, bias=False)  # pair bias
        self.gate      = nn.Linear(d_pair, d_pair)
        self.out_proj  = nn.Linear(d_pair, d_pair)
        self.norm_out  = nn.LayerNorm(d_pair)

    def forward(self, pair: Tensor) -> Tensor:
        """pair: [N_a, N_b, D_PAIR] -> [N_a, N_b, D_PAIR]"""
        N_a, N_b, D = pair.shape
        z = self.norm(pair)

        # Transpose so attending dimension is always first
        if self.axis == "column":
            z = z.transpose(0, 1)   # [N_b, N_a, D]

        L, S, _ = z.shape

        Q = self.q_proj(z).view(L, S, self.n_heads, self.d_head)
        K = self.k_proj(z).view(L, S, self.n_heads, self.d_head)
        V = self.v_proj(z).view(L, S, self.n_heads, self.d_head)
        B = self.bias_proj(z)           # [L, S, n_heads] — pair bias

        # [L, n_heads, S, d_head]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 3, 1)      # [L, n_heads, d_head, S]
        V = V.permute(0, 2, 1, 3)
        B = B.permute(0, 2, 1).unsqueeze(2)  # [L, n_heads, 1, S]

        attn  = (Q @ K) / math.sqrt(self.d_head) + B
        attn  = F.softmax(attn, dim=-1)
        out   = (attn @ V).permute(0, 2, 1, 3).contiguous().view(L, S, D)

        gate  = torch.sigmoid(self.gate(z))
        out   = self.out_proj(gate * out)

        if self.axis == "column":
            out = out.transpose(0, 1)

        return self.norm_out(pair + out)


# ── 5. Outer Product Mean Update ──────────────────────────────────────────────

class OuterProductMeanUpdate(nn.Module):
    """
    Propagate single representation info into the pair representation.
    Low-rank outer product: pair(i,j) += f( single_a[i] ⊗ single_b[j] )
    """

    def __init__(self, d_single: int = D_SINGLE, d_pair: int = D_PAIR,
                 rank: int = 32):
        super().__init__()
        self.rank      = rank
        self.norm      = nn.LayerNorm(d_single)
        self.left_proj = nn.Linear(d_single, rank)
        self.right_proj= nn.Linear(d_single, rank)
        self.out_proj  = nn.Linear(rank * rank, d_pair)
        self.norm_out  = nn.LayerNorm(d_pair)

    def forward(self, single_a: Tensor, single_b: Tensor) -> Tensor:
        """Returns [N_a, N_b, D_PAIR] update."""
        a = self.left_proj(self.norm(single_a))    # [N_a, rank]
        b = self.right_proj(self.norm(single_b))   # [N_b, rank]

        outer = (a.unsqueeze(1).unsqueeze(3) *
                 b.unsqueeze(0).unsqueeze(2))       # [N_a, N_b, rank, rank]
        N_a, N_b = a.shape[0], b.shape[0]
        outer = outer.reshape(N_a, N_b, self.rank * self.rank)
        return self.norm_out(self.out_proj(outer))


# ── 6. Single Transition ──────────────────────────────────────────────────────

class SingleTransition(nn.Module):
    """Standard position-wise FFN update for the single representation."""

    def __init__(self, d_single: int = D_SINGLE, expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_single)
        self.ff   = nn.Sequential(
            nn.Linear(d_single, d_single * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_single * expansion, d_single),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ff(self.norm(x))


# ── 7. One PairFormer Block ───────────────────────────────────────────────────

class PairFormerBlock(nn.Module):

    def __init__(self, d_single: int = D_SINGLE, d_pair: int = D_PAIR,
                 dropout: float = 0.1):
        super().__init__()
        self.opm        = OuterProductMeanUpdate(d_single, d_pair)
        self.tri_out    = TriangleMultiplicativeUpdate(d_pair, "outgoing")
        self.tri_in     = TriangleMultiplicativeUpdate(d_pair, "incoming")
        self.tri_row    = TriangleAttention(d_pair, N_HEADS_TRI, "row")
        self.tri_col    = TriangleAttention(d_pair, N_HEADS_TRI, "column")
        self.single_a_ffn = SingleTransition(d_single, dropout=dropout)
        self.single_b_ffn = SingleTransition(d_single, dropout=dropout)
        self.pair_norm  = nn.LayerNorm(d_pair)

    def forward(self, single_a: Tensor, single_b: Tensor,
                pair: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # 1. Outer product mean: single -> pair
        pair = pair + self.opm(single_a, single_b)
        # 2. Triangle updates
        pair = self.tri_out(pair)
        pair = self.tri_in(pair)
        # 3. Triangle attention
        pair = self.tri_row(pair)
        pair = self.tri_col(pair)
        # 4. Single transitions (A and B independently)
        single_a = self.single_a_ffn(single_a)
        single_b = self.single_b_ffn(single_b)
        return single_a, single_b, self.pair_norm(pair)


# ── 8. Full PairFormer Trunk ──────────────────────────────────────────────────

class PairFormerTrunk(nn.Module):
    """
    Full PairFormer binding trunk.

    Takes encoder outputs (h, x) for both entities.
    Projects into trunk dimensions, then runs N_TRUNK_LAYERS PairFormerBlocks.

    Output:
        single_a: [N_a, D_SINGLE]
        single_b: [N_b, D_SINGLE]
        pair:     [N_a, N_b, D_PAIR]
    """

    def __init__(self, d_single: int = D_SINGLE, d_pair: int = D_PAIR,
                 n_layers: int = N_TRUNK_LAYERS, d_enc_out: int = D_OUT,
                 dropout: float = 0.1):
        super().__init__()

        self.single_init = SingleInitProjection(d_enc_out, d_single)
        self.pair_init   = PairInitProjection(d_single, d_pair)

        self.blocks = nn.ModuleList([
            PairFormerBlock(d_single, d_pair, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm_single = nn.LayerNorm(d_single)
        self.final_norm_pair   = nn.LayerNorm(d_pair)

    def forward(self, h_a: Tensor, x_a: Tensor,
                h_b: Tensor, x_b: Tensor,
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        h_a: [N_a, D_OUT]   entity A encoder output
        x_a: [N_a, 3]       entity A coords
        h_b: [N_b, D_OUT]   entity B encoder output
        x_b: [N_b, 3]       entity B coords

        Returns: single_a, single_b, pair
        """
        single_a = self.single_init(h_a, x_a)      # [N_a, D_SINGLE]
        single_b = self.single_init(h_b, x_b)      # [N_b, D_SINGLE]
        pair     = self.pair_init(single_a, single_b, x_a, x_b)  # [N_a, N_b, D_PAIR]

        for block in self.blocks:
            single_a, single_b, pair = block(single_a, single_b, pair)

        return (
            self.final_norm_single(single_a),
            self.final_norm_single(single_b),
            self.final_norm_pair(pair),
        )
