"""End-to-end CONCH v1.5 Q/V LoRA followed by SlotSPE."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tools.conch_qv_lora import (
    GeneConditionedQVLoRAQKVLinear,
    QVLoRAQKVLinear,
    conch_qv_lora_gene_context,
    inject_conch_qv_lora,
)
from tools.titan_local import load_local_titan


class OnlineConchSlotSPE(nn.Module):
    """Encode sampled raw patches while training only CONCH Q/V adapters."""

    def __init__(self, slotspe: nn.Module, args) -> None:
        super().__init__()
        if int(args.batch_size) != 1:
            raise ValueError("online CONCH training currently requires batch_size=1")
        self.slotspe = slotspe
        self.patch_batch_size = int(args.online_patch_batch_size)
        self.lora_mode = str(args.conch_qv_lora_mode).casefold()
        self.use_checkpoint = bool(args.conch_gradient_checkpointing)
        if self.patch_batch_size <= 0:
            raise ValueError("online_patch_batch_size must be positive")
        if self.lora_mode not in {"none", "static", "gene"}:
            raise ValueError("conch_qv_lora_mode must be 'none', 'static', or 'gene'")

        _, conch, _ = load_local_titan(
            args.online_conch_model_dir,
            device="cpu",
            return_conch=True,
        )
        if self.lora_mode == "none":
            for parameter in conch.parameters():
                parameter.requires_grad = False
            summary = {
                "mode": "none",
                "targets": [],
                "trainable_conch_params": 0,
            }
        else:
            summary = inject_conch_qv_lora(
                conch,
                layers=args.conch_qv_lora_layers,
                rank=args.conch_qv_lora_rank,
                alpha=args.conch_qv_lora_alpha,
                dropout=args.conch_qv_lora_dropout,
                gene_dim=args.wsi_projection_dim if self.lora_mode == "gene" else None,
                gene_hidden_dim=args.conch_gene_hidden_dim,
                freeze_non_lora=True,
            )
        self.conch = conch
        self.lora_summary = summary

    @property
    def last_loss_components(self):
        return self.slotspe.last_loss_components

    def train(self, mode: bool = True):
        super().train(mode)
        self.conch.eval()
        for module in self.conch.modules():
            if isinstance(module, (QVLoRAQKVLinear, GeneConditionedQVLoRAQKVLinear)):
                module.train(mode)
        return self

    def _encode_static(self, pixels: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.conch(pixels).float(), dim=-1)

    def _encode_gene(self, pixels: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        with conch_qv_lora_gene_context(self.conch, context):
            return F.normalize(self.conch(pixels).float(), dim=-1)

    def _encode_patches(
        self,
        pixels: torch.Tensor,
        gene_context: torch.Tensor | None,
    ) -> torch.Tensor:
        chunks = []
        for start in range(0, pixels.shape[0], self.patch_batch_size):
            chunk = pixels[start : start + self.patch_batch_size]
            if self.lora_mode == "gene":
                context = gene_context.expand(chunk.shape[0], -1)
                if self.use_checkpoint and self.training:
                    encoded = checkpoint(
                        self._encode_gene, chunk, context, use_reentrant=False
                    )
                else:
                    encoded = self._encode_gene(chunk, context)
            elif self.use_checkpoint and self.training:
                encoded = checkpoint(self._encode_static, chunk, use_reentrant=False)
            else:
                encoded = self._encode_static(chunk)
            chunks.append(encoded)
        return torch.cat(chunks, dim=0)

    def forward(self, **kwargs):
        pixels = kwargs["x_wsi"]
        if pixels.ndim != 5 or pixels.shape[0] != 1:
            raise ValueError("online raw patches must have shape [1,N,3,H,W]")
        encoded_omics = self.slotspe.encode_omics_from_kwargs(kwargs)
        gene_context = (
            self.slotspe.pool_omics_context(encoded_omics)
            if self.lora_mode == "gene"
            else None
        )
        patch_features = self._encode_patches(pixels[0], gene_context)

        slotspe_kwargs = dict(kwargs)
        slotspe_kwargs["x_wsi"] = patch_features.unsqueeze(0)
        slotspe_kwargs["encoded_omics"] = encoded_omics
        return self.slotspe(**slotspe_kwargs)

    def state_dict(self, *args, **kwargs):
        full = super().state_dict(*args, **kwargs)
        return OrderedDict(
            (name, value)
            for name, value in full.items()
            if name.startswith("slotspe.")
            or any(
                part in name
                for part in (
                    ".q_down.",
                    ".q_up.",
                    ".v_down.",
                    ".v_up.",
                    ".gene_modulator.",
                )
            )
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict:
            relevant_missing = [
                name
                for name in result.missing_keys
                if name.startswith("slotspe.")
                or any(
                    part in name
                    for part in (
                        ".q_down.",
                        ".q_up.",
                        ".v_down.",
                        ".v_up.",
                        ".gene_modulator.",
                    )
                )
            ]
            if relevant_missing or result.unexpected_keys:
                raise RuntimeError(
                    "online CONCH checkpoint mismatch: "
                    f"missing={relevant_missing}, unexpected={result.unexpected_keys}"
                )
        return result
