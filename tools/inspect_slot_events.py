#!/usr/bin/env python3
"""Print a compact human-readable report from saved event-attention details."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("Details file must contain a dictionary")
    return value


def _last(value: Any) -> Any:
    return value[-1] if isinstance(value, list) else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--slide-id", default="unknown")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--open-threshold", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        details = _load(args.details)
        event_names = list(details["event_names"])
        gate = _last(details["event_gate"])[args.batch_index].squeeze(-1)
        dominant = _last(details["dominant_event"])[args.batch_index]
        semantic = _last(details.get("semantic_score", details["semantic_gate"]))[
            args.batch_index
        ].squeeze(-1)
        visual = _last(details.get("visual_score", details["visual_gate"]))[
            args.batch_index
        ].squeeze(-1)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Could not inspect details: {exc}", file=sys.stderr)
        return 1
    print(f"slide_id: {args.slide_id}")
    for slot_index in range(len(gate)):
        event_index = int(dominant[slot_index])
        print(f"slot {slot_index}:")
        print(f"  gate: {float(gate[slot_index]):.4f}")
        if float(gate[slot_index]) < args.open_threshold:
            print("  status: open slot")
            print(f"  best known event: {event_names[event_index]}")
        else:
            print(f"  dominant event: {event_names[event_index]}")
        print(f"  semantic score: {float(semantic[slot_index]):.4f}")
        print(f"  visual score: {float(visual[slot_index]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
