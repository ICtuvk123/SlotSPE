"""LoRA/GC-RSM adapters for CONCH-style combined QKV projections."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


class QVLoRAQKVLinear(nn.Module):
    """Wrap ``Linear(C, 3C)`` and add low-rank residuals only to Q and V.

    CONCH v1.5 follows the timm/open_clip convention where attention uses one
    combined projection whose output is laid out as ``[Q, K, V]``.  This wrapper
    preserves the frozen base QKV projection exactly and adds trainable low-rank
    updates to the Q and V slices only.
    """

    def __init__(
        self,
        base_qkv: nn.Linear,
        *,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_qkv, nn.Linear):
            raise TypeError("base_qkv must be nn.Linear")
        if base_qkv.out_features != base_qkv.in_features * 3:
            raise ValueError("base_qkv must have shape Linear(C, 3C)")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.base_qkv = base_qkv
        self.in_features = int(base_qkv.in_features)
        self.out_features = int(base_qkv.out_features)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)

        self.q_down = nn.Linear(self.in_features, self.rank, bias=False)
        self.q_up = nn.Linear(self.rank, self.in_features, bias=False)
        self.v_down = nn.Linear(self.in_features, self.rank, bias=False)
        self.v_up = nn.Linear(self.rank, self.in_features, bias=False)

        nn.init.kaiming_uniform_(self.q_down.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.v_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

        for parameter in self.base_qkv.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.base_qkv(x)
        q_delta = self.q_up(self.q_down(self.dropout(x))) * self.scaling
        v_delta = self.v_up(self.v_down(self.dropout(x))) * self.scaling
        q, k, v = qkv.chunk(3, dim=-1)
        return torch.cat((q + q_delta, k, v + v_delta), dim=-1)


class GeneConditionedQVLoRAQKVLinear(QVLoRAQKVLinear):
    """Q/V LoRA whose rank channels are gated by a patient gene context."""

    def __init__(
        self,
        base_qkv: nn.Linear,
        *,
        gene_dim: int,
        rank: int = 8,
        hidden_dim: int = 128,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(base_qkv, rank=rank, alpha=alpha, dropout=dropout)
        if gene_dim <= 0 or hidden_dim <= 0:
            raise ValueError("gene_dim and hidden_dim must be positive")
        self.gene_dim = int(gene_dim)
        self.hidden_dim = int(hidden_dim)
        self.gene_context: torch.Tensor | None = None
        self.gene_modulator = nn.Sequential(
            nn.LayerNorm(self.gene_dim),
            nn.Linear(self.gene_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.rank * 2),
            nn.Sigmoid(),
        )

    def set_gene_context(self, gene_context: torch.Tensor | None) -> None:
        if gene_context is not None:
            if gene_context.ndim != 2 or gene_context.shape[-1] != self.gene_dim:
                raise ValueError(f"gene_context must be [B, {self.gene_dim}]")
        self.gene_context = gene_context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gene_context is None:
            raise RuntimeError("gene_context must be set before GC-RSM CONCH LoRA forward")
        if x.shape[0] != self.gene_context.shape[0]:
            raise ValueError("token batch and gene_context batch must match")

        qkv = self.base_qkv(x)
        gates = self.gene_modulator(self.gene_context.to(device=x.device, dtype=x.dtype))
        q_gate, v_gate = gates.chunk(2, dim=-1)
        q_gate = q_gate.unsqueeze(1)
        v_gate = v_gate.unsqueeze(1)

        dropped = self.dropout(x)
        q_rank = self.q_down(dropped) * q_gate
        v_rank = self.v_down(dropped) * v_gate
        q_delta = self.q_up(q_rank) * self.scaling
        v_delta = self.v_up(v_rank) * self.scaling
        q, k, v = qkv.chunk(3, dim=-1)
        return torch.cat((q + q_delta, k, v + v_delta), dim=-1)


@dataclass(frozen=True)
class ConchQVLoraSummary:
    replaced: int
    target_names: tuple[str, ...]
    trainable_parameters: int


def _module_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _candidate_qkv_names(model: nn.Module) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if name.endswith(".qkv") and isinstance(module, nn.Linear):
            if module.out_features == module.in_features * 3:
                names.append(name)
    return names


def inject_conch_qv_lora(
    model: nn.Module,
    *,
    layers: int = 2,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    gene_dim: int | None = None,
    gene_hidden_dim: int = 128,
    freeze_non_lora: bool = True,
) -> ConchQVLoraSummary:
    """Replace the last ``layers`` combined QKV projections with Q/V LoRA."""
    if layers <= 0:
        raise ValueError("layers must be positive")
    names = _candidate_qkv_names(model)
    if not names:
        raise ValueError("No CONCH-style qkv Linear(C, 3C) modules found")
    target_names = tuple(names[-layers:])

    if freeze_non_lora:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    for name in target_names:
        parent, child_name = _module_parent(model, name)
        base_qkv = getattr(parent, child_name)
        if gene_dim is None:
            wrapped = QVLoRAQKVLinear(
                base_qkv, rank=rank, alpha=alpha, dropout=dropout
            )
        else:
            wrapped = GeneConditionedQVLoRAQKVLinear(
                base_qkv,
                gene_dim=gene_dim,
                hidden_dim=gene_hidden_dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        setattr(parent, child_name, wrapped)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ConchQVLoraSummary(
        replaced=len(target_names),
        target_names=target_names,
        trainable_parameters=trainable,
    )


def conch_qv_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only Q/V LoRA adapter weights."""
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if any(part in name for part in (".q_down.", ".q_up.", ".v_down.", ".v_up.", ".gene_modulator."))
    }


def iter_gene_conditioned_qv_lora(model: nn.Module) -> Iterator[GeneConditionedQVLoRAQKVLinear]:
    for module in model.modules():
        if isinstance(module, GeneConditionedQVLoRAQKVLinear):
            yield module


@contextmanager
def conch_qv_lora_gene_context(model: nn.Module, gene_context: torch.Tensor):
    modules = list(iter_gene_conditioned_qv_lora(model))
    if not modules:
        raise ValueError("model has no gene-conditioned CONCH Q/V LoRA modules")
    try:
        for module in modules:
            module.set_gene_context(gene_context)
        yield
    finally:
        for module in modules:
            module.set_gene_context(None)
