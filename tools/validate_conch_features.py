#!/usr/bin/env python3
"""Validate normalized CONCH contrastive tensors before deleting source WSIs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


def _manifest_slide_ids(path: Path, slide_ext: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"GDC manifest lacks a filename column: {path}")
        result = []
        for row in reader:
            filename = row["filename"]
            if filename.endswith(slide_ext):
                result.append(filename[: -len(slide_ext)])
    if not result:
        raise ValueError(f"No {slide_ext} slides found in manifest: {path}")
    return result


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("expected a plain [N,D] Tensor")
    return value


def _norm_range(tensor: torch.Tensor, chunk_size: int) -> tuple[float, float]:
    minimum = float("inf")
    maximum = float("-inf")
    for chunk in tensor.split(chunk_size, dim=0):
        values = chunk.float()
        if not torch.isfinite(values).all():
            raise ValueError("contains NaN or infinity")
        norms = values.norm(dim=-1)
        minimum = min(minimum, float(norms.min()))
        maximum = max(maximum, float(norms.max()))
    return minimum, maximum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--slide-ext", default=".svs")
    parser.add_argument("--expected-dim", type=int, default=512)
    parser.add_argument("--expected-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--norm-tolerance", type=float, default=5e-3)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--allow-missing", type=int, default=0)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        slide_ids = _manifest_slide_ids(args.manifest, args.slide_ext)
        expected_dtype = getattr(torch, args.expected_dtype)
        missing: list[str] = []
        invalid: dict[str, str] = {}
        patch_count = 0
        total_bytes = 0
        global_min = float("inf")
        global_max = float("-inf")

        for index, slide_id in enumerate(slide_ids, start=1):
            path = args.feature_dir / f"{slide_id}.pt"
            if not path.is_file():
                missing.append(slide_id)
                continue
            try:
                tensor = _load_tensor(path)
                if tensor.shape[0] < 1 or tensor.shape[1] != args.expected_dim:
                    raise ValueError(
                        f"shape {tuple(tensor.shape)} != [N,{args.expected_dim}]"
                    )
                if tensor.dtype != expected_dtype:
                    raise ValueError(f"dtype {tensor.dtype} != {expected_dtype}")
                norm_min, norm_max = _norm_range(tensor, args.chunk_size)
                if norm_min < 1.0 - args.norm_tolerance or norm_max > 1.0 + args.norm_tolerance:
                    raise ValueError(
                        f"L2 norm range [{norm_min:.6f}, {norm_max:.6f}] exceeds tolerance"
                    )
                patch_count += tensor.shape[0]
                total_bytes += path.stat().st_size
                global_min = min(global_min, norm_min)
                global_max = max(global_max, norm_max)
            except (OSError, RuntimeError, ValueError) as exc:
                invalid[slide_id] = str(exc)
            finally:
                if index % 50 == 0:
                    print(f"validated {index}/{len(slide_ids)} manifest entries")

        report = {
            "manifest": str(args.manifest),
            "feature_dir": str(args.feature_dir),
            "expected_slides": len(slide_ids),
            "valid_slides": len(slide_ids) - len(missing) - len(invalid),
            "missing_slides": missing,
            "invalid_slides": invalid,
            "total_patches": patch_count,
            "feature_bytes": total_bytes,
            "embedding_dim": args.expected_dim,
            "dtype": args.expected_dtype,
            "norm_min": None if patch_count == 0 else global_min,
            "norm_max": None if patch_count == 0 else global_max,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))

        if invalid:
            return 1
        if len(missing) > args.allow_missing:
            return 1
        return 0
    except (OSError, ValueError) as exc:
        print(f"CONCH validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
