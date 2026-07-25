#!/usr/bin/env python3
"""Smoke-test CONCH v1.5 visual-backbone Q/V LoRA injection."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .conch_qv_lora import inject_conch_qv_lora
    from .titan_local import load_local_titan
except ImportError:  # pragma: no cover - direct script execution
    from conch_qv_lora import inject_conch_qv_lora
    from titan_local import load_local_titan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    _, conch, _preprocess = load_local_titan(
        args.model, device=device, return_conch=True
    )
    conch.eval()
    images = torch.randn(args.batch_size, 3, 448, 448, device=device)

    with torch.no_grad():
        before = conch.encode_image(images, proj_contrast=True, normalize=True)

    summary = inject_conch_qv_lora(
        conch,
        layers=args.layers,
        rank=args.rank,
        dropout=0.0,
        freeze_non_lora=True,
    )
    after = conch.encode_image(images, proj_contrast=True, normalize=True)
    if not torch.equal(before, after):
        raise RuntimeError("zero-initialized CONCH Q/V LoRA changed initial output")

    loss = after.float().sum()
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in conch.named_parameters()
        if parameter.requires_grad and any(part in name for part in (".q_up.", ".v_up."))
    ]
    if not gradients or not all(gradient is not None for gradient in gradients):
        raise RuntimeError("CONCH Q/V LoRA parameters did not receive gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("CONCH Q/V LoRA gradients are not finite")

    print(
        "[ok] CONCH v1.5 Q/V LoRA injected: "
        f"targets={list(summary.target_names)}, "
        f"trainable_params={summary.trainable_parameters}, "
        f"output_shape={tuple(after.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
