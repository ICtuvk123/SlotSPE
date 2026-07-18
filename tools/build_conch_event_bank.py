#!/usr/bin/env python3
"""Encode reviewed pathology events with the frozen CONCH text encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

try:
    from .conch_utils import CONCH_MODEL_NAME, conch_tokenize, load_frozen_conch
    from .event_schema import load_json, validate_event_payload
except ImportError:  # direct script execution
    from conch_utils import CONCH_MODEL_NAME, conch_tokenize, load_frozen_conch
    from event_schema import load_json, validate_event_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "third_party/CONCH/checkpoints/conch/pytorch_model.bin"
ENCODABLE_REVIEW_STATUSES = {
    "machine_checked_pending_expert_review",
    "machine_curated_pending_pathologist_review",
    "pathologist_reviewed",
}


def encode_prompt_ensemble(
    event_prompts: list[list[str]],
    *,
    encode_batch: Callable[[list[str]], torch.Tensor],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all four prompts independently, then mean and L2-normalize by event."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(len(prompts) != 4 for prompts in event_prompts):
        raise ValueError("Every event must contain exactly four prompt templates")
    flat_prompts = [prompt for prompts in event_prompts for prompt in prompts]
    batches: list[torch.Tensor] = []
    for start in range(0, len(flat_prompts), batch_size):
        features = encode_batch(flat_prompts[start : start + batch_size])
        if features.ndim != 2:
            raise ValueError(f"CONCH text encoder must return [batch,D], got {features.shape}")
        batches.append(F.normalize(features.float(), dim=-1).cpu())
    prompt_embeddings = torch.cat(batches, dim=0).reshape(len(event_prompts), 4, -1)
    event_embeddings = F.normalize(prompt_embeddings.mean(dim=1), dim=-1)
    return prompt_embeddings, event_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--from-hf", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = load_json(args.events)
        errors = validate_event_payload(payload)
        if errors:
            raise ValueError("Event validation failed:\n- " + "\n- ".join(errors))
        if payload.get("status") == "unreviewed_llm_candidates":
            raise ValueError(
                "Refusing to encode unreviewed raw candidates; run review_event_candidates.py first."
            )
        if payload.get("status") not in ENCODABLE_REVIEW_STATUSES:
            raise ValueError(
                f"Refusing to encode status {payload.get('status')!r}; expected one of "
                f"{sorted(ENCODABLE_REVIEW_STATUSES)}"
            )
        if payload.get("status") != "pathologist_reviewed":
            print("WARNING: candidates are still pending pathology expert review.")

        model, _preprocess, tokenizer, tokenize, device = load_frozen_conch(
            checkpoint=args.checkpoint,
            from_hf=args.from_hf,
            device=args.device,
        )
        events = payload["events"]
        event_prompts = [event["prompt_templates"] for event in events]

        def encode_batch(texts: list[str]) -> torch.Tensor:
            text_tokens = conch_tokenize(tokenize, tokenizer, texts).to(device)
            with torch.inference_mode():
                text_features = model.encode_text(text_tokens)
            return F.normalize(text_features.float(), dim=-1)

        prompt_embeddings, event_embeddings = encode_prompt_ensemble(
            event_prompts,
            encode_batch=encode_batch,
            batch_size=args.batch_size,
        )
        norms = event_embeddings.norm(dim=-1)
        if event_embeddings.shape[0] != len(events) or not torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError("Event embedding shape or normalization validation failed")

        artifact = {
            "model_name": CONCH_MODEL_NAME,
            "encoder_type": "CONCH text encoder",
            "feature_space": "conch_contrastive",
            "normalized": True,
            "prompt_ensemble": "mean_then_l2_normalize",
            "cancer_type": payload["cancer_type"],
            "dataset": payload["dataset"],
            "review_status": payload["status"],
            "event_ids": [event["event_id"] for event in events],
            "event_names": [event["event_name"] for event in events],
            "event_categories": [event["category"] for event in events],
            "event_definitions": [event["definition"] for event in events],
            "event_prompts": event_prompts,
            "schema_version": payload.get("schema_version", "1.0"),
            "ontology_version": payload.get("ontology_version", "legacy_v0"),
            "patch_spec": payload.get("patch_spec"),
            "generation_metadata": payload.get("generation_metadata"),
            "curation_metadata": payload.get("curation_metadata"),
            "sources": payload.get("sources", []),
            "event_metadata": [
                {
                    key: event.get(key)
                    for key in (
                        "event_id",
                        "event_name",
                        "category",
                        "evidence_tier",
                        "observation_scale",
                        "patch_observable",
                        "required_context",
                        "confounders",
                        "source_basis",
                        "semantic_group",
                    )
                    if key in event
                }
                for event in events
            ],
            "prompt_embeddings": prompt_embeddings.cpu(),
            "event_embeddings": event_embeddings.cpu(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.output)
        summary_path = args.output.with_name(args.output.stem + "_summary.json")
        summary = {
            key: value
            for key, value in artifact.items()
            if key not in {"prompt_embeddings", "event_embeddings"}
        }
        summary["prompt_embeddings_shape"] = list(prompt_embeddings.shape)
        summary["event_embeddings_shape"] = list(event_embeddings.shape)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Event-bank build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved event bank {tuple(event_embeddings.shape)} to {args.output}")
    print(f"Saved metadata summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
