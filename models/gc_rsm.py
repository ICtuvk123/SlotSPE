"""Gene-conditioned low-rank modulation blocks."""

from __future__ import annotations

import torch
from torch import nn


class GeneConditionedRankSpaceModulation(nn.Module):
    """Low-rank residual adapter gated by a patient-level omics context."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gene_dim: int,
        rank: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or gene_dim <= 0:
            raise ValueError("in_dim, out_dim, and gene_dim must be positive")
        if rank <= 0 or hidden_dim <= 0:
            raise ValueError("rank and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.gene_dim = int(gene_dim)
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)

        self.down = nn.Linear(self.in_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.out_dim, bias=False)
        self.gene_modulator = nn.Sequential(
            nn.LayerNorm(self.gene_dim),
            nn.Linear(self.gene_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.rank),
            nn.Sigmoid(),
        )

        # Start as an exact residual no-op, so enabling the module is stable.
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        base_tokens: torch.Tensor,
        gene_context: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.in_dim:
            raise ValueError(f"tokens must be [B, N, {self.in_dim}]")
        if base_tokens.ndim != 3 or base_tokens.shape[-1] != self.out_dim:
            raise ValueError(f"base_tokens must be [B, N, {self.out_dim}]")
        if gene_context.ndim != 2 or gene_context.shape[-1] != self.gene_dim:
            raise ValueError(f"gene_context must be [B, {self.gene_dim}]")
        if tokens.shape[:2] != base_tokens.shape[:2]:
            raise ValueError("tokens and base_tokens must share batch and token dimensions")
        if tokens.shape[0] != gene_context.shape[0]:
            raise ValueError("tokens and gene_context must share batch dimension")

        rank_tokens = self.down(tokens.float())
        gates = self.gene_modulator(gene_context.float()).unsqueeze(1)
        delta = self.up(rank_tokens * gates)
        return base_tokens + self.residual_scale * delta
