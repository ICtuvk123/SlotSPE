#!/usr/bin/env python3
"""Offline smoke test for local TITAN text and CONCH v1.5 patch encoders."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .titan_local import load_local_titan
except ImportError:
    from titan_local import load_local_titan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/data0/lfy_data/Pathology/checkpoints/TITAN"),
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")

    titan, conch, preprocess = load_local_titan(
        args.model, device=device, return_conch=True
    )
    image = Image.new("RGB", (512, 512), color=(180, 90, 130))
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    tokens = titan.text_encoder.tokenizer(["H&E histopathology showing clear cell renal carcinoma"])
    tokens = tokens.to(device)

    with torch.inference_mode():
        patch_embedding = F.normalize(conch(image_tensor).float(), dim=-1)
        text_embedding = F.normalize(titan.encode_text(tokens).float(), dim=-1)

    for name, embedding in (
        ("CONCH v1.5 patch", patch_embedding),
        ("TITAN text", text_embedding),
    ):
        if embedding.shape != (1, 768):
            raise RuntimeError(f"{name} shape is {tuple(embedding.shape)}, expected (1, 768)")
        if not torch.isfinite(embedding).all():
            raise RuntimeError(f"{name} contains NaN or infinity")
        norm = float(embedding.norm(dim=-1).item())
        if abs(norm - 1.0) > 1e-5:
            raise RuntimeError(f"{name} is not normalized: norm={norm:.8f}")
        print(f"[ok] {name}: shape={tuple(embedding.shape)}, norm={norm:.6f}")

    print("[ok] TITAN and CONCH v1.5 loaded fully from the local snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
