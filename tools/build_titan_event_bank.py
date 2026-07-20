#!/usr/bin/env python3
"""Encode reviewed pathology events with TITAN's frozen text encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .build_conch_event_bank import (
        ENCODABLE_REVIEW_STATUSES,
        encode_prompt_ensemble,
    )
    from .event_schema import load_json, validate_event_payload
except ImportError:  # direct script execution
    from build_conch_event_bank import ENCODABLE_REVIEW_STATUSES, encode_prompt_ensemble
    from event_schema import load_json, validate_event_payload

try:
    from .titan_local import load_local_titan, validate_titan_snapshot
except ImportError:
    from titan_local import load_local_titan, validate_titan_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data0/lfy_data/Pathology/checkpoints/TITAN")
FEATURE_SPACE = "titan_text_768"
EXPECTED_DIM = 768
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Local TITAN directory, or MahmoodLab/TITAN with --allow-download.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download the gated model instead of using local files only.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-dim", type=int, default=EXPECTED_DIM)
    parser.add_argument(
        "--source-patch-size",
        type=int,
        default=512,
        help="Physical patch width/height at 20x used for v1.5 extraction.",
    )
    parser.add_argument("--encoder-input-size", type=int, default=448)
    return parser


def _event_metadata(events: list[dict]) -> list[dict]:
    keys = (
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
    return [{key: event.get(key) for key in keys if key in event} for event in events]


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = load_json(args.events)
        errors = validate_event_payload(payload)
        if errors:
            raise ValueError("Event validation failed:\n- " + "\n- ".join(errors))
        status = payload.get("status")
        if status == "unreviewed_llm_candidates":
            raise ValueError(
                "Refusing to encode unreviewed raw candidates; run review_event_candidates.py first."
            )
        if status not in ENCODABLE_REVIEW_STATUSES:
            raise ValueError(
                f"Refusing to encode status {status!r}; expected one of "
                f"{sorted(ENCODABLE_REVIEW_STATUSES)}"
            )
        if status != "pathologist_reviewed":
            print("WARNING: candidates are still pending pathology expert review.")
        if args.expected_dim <= 0 or args.source_patch_size <= 0 or args.encoder_input_size <= 0:
            raise ValueError("Embedding and patch dimensions must be positive")

        local_model_path = Path(args.model).expanduser()
        if local_model_path.exists():
            validate_titan_snapshot(local_model_path)

        try:
            import transformers
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required to load TITAN") from exc
        if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5:
            raise RuntimeError(
                "TITAN requires transformers<5 (the official environment uses 4.46.0); "
                f"found {transformers.__version__}. Use a separate TITAN preprocessing environment."
            )

        requested_device = torch.device(args.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; pass --device cpu")
        if not local_model_path.exists():
            if not args.allow_download:
                raise RuntimeError(f"Local TITAN model directory does not exist: {local_model_path}")
            titan = AutoModel.from_pretrained(
                args.model, trust_remote_code=True, local_files_only=False
            ).to(requested_device).eval()
            for parameter in titan.parameters():
                parameter.requires_grad_(False)
        else:
            titan = load_local_titan(local_model_path, device=requested_device)
        tokenizer = titan.text_encoder.tokenizer

        events = payload["events"]
        event_prompts = [event["prompt_templates"] for event in events]

        def encode_batch(texts: list[str]) -> torch.Tensor:
            tokens = tokenizer(texts).to(requested_device)
            with torch.inference_mode():
                features = titan.encode_text(tokens)
            if isinstance(features, tuple):
                features = features[0]
            if not isinstance(features, torch.Tensor):
                raise RuntimeError("TITAN encode_text did not return a Tensor")
            return F.normalize(features.float(), dim=-1)

        prompt_embeddings, event_embeddings = encode_prompt_ensemble(
            event_prompts,
            encode_batch=encode_batch,
            batch_size=args.batch_size,
        )
        if event_embeddings.shape != (len(events), args.expected_dim):
            raise RuntimeError(
                f"Expected event embeddings [{len(events)},{args.expected_dim}], "
                f"got {tuple(event_embeddings.shape)}"
            )
        norms = event_embeddings.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise RuntimeError("TITAN event embeddings are not L2-normalized")

        source_patch_spec = payload.get("patch_spec")
        patch_spec = dict(source_patch_spec or {})
        patch_spec.update(
            {
                "width_px": args.source_patch_size,
                "height_px": args.source_patch_size,
                "encoder_input_width_px": args.encoder_input_size,
                "encoder_input_height_px": args.encoder_input_size,
            }
        )
        artifact = {
            "model_name": "MahmoodLab/TITAN",
            "encoder_type": "TITAN text encoder",
            "feature_space": FEATURE_SPACE,
            "normalized": True,
            "prompt_ensemble": "mean_then_l2_normalize",
            "cancer_type": payload["cancer_type"],
            "dataset": payload["dataset"],
            "review_status": status,
            "event_ids": [event["event_id"] for event in events],
            "event_names": [event["event_name"] for event in events],
            "event_categories": [event["category"] for event in events],
            "event_definitions": [event["definition"] for event in events],
            "event_prompts": event_prompts,
            "schema_version": payload.get("schema_version", "1.0"),
            "ontology_version": payload.get("ontology_version", "legacy_v0"),
            "patch_spec": patch_spec,
            "source_event_patch_spec": source_patch_spec,
            "generation_metadata": payload.get("generation_metadata"),
            "curation_metadata": payload.get("curation_metadata"),
            "sources": payload.get("sources", []),
            "event_metadata": _event_metadata(events),
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
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"TITAN event-bank build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved TITAN event bank {tuple(event_embeddings.shape)} to {args.output}")
    print(f"Saved metadata summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
