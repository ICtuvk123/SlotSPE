"""DyKo-style trainable adapters for frozen visual and text tokens."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DyKoAdapter(nn.Module):
    """The independent 768 -> 192 -> 768 MLP used by DyKo."""

    def __init__(self, dim: int, reduction: int = 4, eps: float = 1e-8) -> None:
        super().__init__()
        if dim <= 0 or reduction <= 0 or dim % reduction != 0:
            raise ValueError("dim must be positive and divisible by reduction")
        self.dim = int(dim)
        self.reduction = int(reduction)
        self.eps = float(eps)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim not in (2, 3) or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must end in dimension {self.dim}")
        adapted = self.fc(tokens.float())
        return F.normalize(adapted, dim=-1, eps=self.eps)
