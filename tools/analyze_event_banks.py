#!/usr/bin/env python3
"""Compare event banks in embedding space and against CONCH patch activations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_event_bank(path: Path) -> dict[str, Any]:
    artifact = _torch_load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Event bank must contain a dictionary: {path}")
    embeddings = artifact.get("event_embeddings")
    event_ids = artifact.get("event_ids")
    event_names = artifact.get("event_names")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"event_embeddings must be [M,D]: {path}")
    if not isinstance(event_ids, list) or len(event_ids) != embeddings.shape[0]:
        raise ValueError(f"event_ids do not align with embeddings: {path}")
    if not isinstance(event_names, list) or len(event_names) != embeddings.shape[0]:
        raise ValueError(f"event_names do not align with embeddings: {path}")
    values = embeddings.float()
    if not torch.isfinite(values).all():
        raise ValueError(f"Event bank contains NaN or infinity: {path}")
    norms = values.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=5e-3, rtol=5e-3):
        raise ValueError(f"Event embeddings are not L2-normalized: {path}")
    return {
        "path": str(path),
        "event_ids": event_ids,
        "event_names": event_names,
        "embeddings": F.normalize(values, dim=-1),
        "ontology_version": artifact.get("ontology_version", "legacy_v0"),
        "review_status": artifact.get("review_status"),
        "event_metadata": artifact.get("event_metadata", []),
    }


def load_patch_embeddings(path: Path) -> torch.Tensor:
    value = _torch_load(path)
    if isinstance(value, dict):
        value = value.get("conch_patch_embeddings", value.get("embeddings"))
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Patch feature file must contain a [N,D] tensor: {path}")
    values = value.float()
    if values.shape[0] < 1 or not torch.isfinite(values).all():
        raise ValueError(f"Patch features are empty or non-finite: {path}")
    norms = values.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-2, rtol=1e-2):
        raise ValueError(f"Patch embeddings are not L2-normalized: {path}")
    return F.normalize(values, dim=-1)


def patient_id_from_filename(path: Path) -> str:
    match = re.match(r"^(TCGA-[A-Za-z0-9]+-[A-Za-z0-9]+)", path.name)
    if match:
        return match.group(1).upper()
    return path.stem.split(".", 1)[0]


def _pair_records(
    matrix: torch.Tensor,
    left_ids: list[str],
    right_ids: list[str],
    *,
    same_bank: bool,
    limit: int = 25,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(left_ids):
        start = left_index + 1 if same_bank else 0
        for right_index in range(start, len(right_ids)):
            records.append(
                {
                    "left_event_id": left_id,
                    "right_event_id": right_ids[right_index],
                    "score": float(matrix[left_index, right_index]),
                }
            )
    records.sort(key=lambda item: item["score"], reverse=True)
    return records[:limit]


def embedding_comparison(
    v0: dict[str, Any], v2: dict[str, Any], threshold: float
) -> dict[str, Any]:
    v0_embeddings, v2_embeddings = v0["embeddings"], v2["embeddings"]
    within_v0 = v0_embeddings @ v0_embeddings.T
    within_v2 = v2_embeddings @ v2_embeddings.T
    cross = v2_embeddings @ v0_embeddings.T
    nearest_scores, nearest_indices = cross.max(dim=1)
    nearest = [
        {
            "v2_event_id": v2["event_ids"][index],
            "v0_event_id": v0["event_ids"][int(nearest_indices[index])],
            "cosine_similarity": float(nearest_scores[index]),
        }
        for index in range(len(v2["event_ids"]))
    ]
    nearest.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    return {
        "redundancy_threshold": threshold,
        "within_v0_top_pairs": _pair_records(
            within_v0, v0["event_ids"], v0["event_ids"], same_bank=True
        ),
        "within_v2_top_pairs": _pair_records(
            within_v2, v2["event_ids"], v2["event_ids"], same_bank=True
        ),
        "within_v2_flagged_pairs": [
            item
            for item in _pair_records(
                within_v2,
                v2["event_ids"],
                v2["event_ids"],
                same_bank=True,
                limit=len(v2["event_ids"]) ** 2,
            )
            if item["score"] >= threshold
        ],
        "v2_nearest_v0": nearest,
    }


def _jaccard_from_counts(counts: torch.Tensor, co_counts: torch.Tensor) -> torch.Tensor:
    union = counts[:, None] + counts[None, :] - co_counts
    return torch.where(union > 0, co_counts / union, torch.zeros_like(union))


def activation_comparison(
    v0: dict[str, Any],
    v2: dict[str, Any],
    *,
    feature_dir: Path,
    device: torch.device,
    batch_size: int,
    score_threshold: float,
    event_score_thresholds: torch.Tensor | None = None,
    threshold_label: str | None = None,
    min_slide_fraction: float,
    min_patient_fraction: float,
    max_files: int | None,
) -> dict[str, Any]:
    files = sorted(feature_dir.glob("*.pt"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No .pt patch feature files found in {feature_dir}")

    all_ids = v0["event_ids"] + v2["event_ids"]
    embeddings = torch.cat([v0["embeddings"], v2["embeddings"]], dim=0).to(device)
    event_count = len(all_ids)
    if event_score_thresholds is None:
        thresholds = torch.full(
            (event_count,), float(score_threshold), dtype=torch.float32, device=device
        )
    else:
        if event_score_thresholds.shape != (event_count,):
            raise ValueError(
                f"Per-event thresholds must have shape ({event_count},), "
                f"found {tuple(event_score_thresholds.shape)}"
            )
        thresholds = event_score_thresholds.float().to(device)
    activation_counts = torch.zeros(event_count, dtype=torch.long)
    coactivation_counts = torch.zeros(event_count, event_count, dtype=torch.long)
    slide_coverage_counts = torch.zeros(event_count, dtype=torch.long)
    patient_activation_counts: dict[str, torch.Tensor] = {}
    patient_patch_counts: dict[str, int] = {}
    total_patches = 0

    for file_index, path in enumerate(files, start=1):
        features = load_patch_embeddings(path)
        if features.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch in {path}: {features.shape[1]} != {embeddings.shape[1]}"
            )
        slide_counts = torch.zeros(event_count, dtype=torch.long)
        for chunk in features.split(batch_size, dim=0):
            scores = chunk.to(device) @ embeddings.T
            active = scores >= thresholds
            batch_counts = active.sum(dim=0).cpu().long()
            slide_counts += batch_counts
            activation_counts += batch_counts
            coactivation_counts += (
                active.float().T.matmul(active.float()).round().cpu().long()
            )
        patch_count = int(features.shape[0])
        total_patches += patch_count
        slide_coverage_counts += (slide_counts.float() / patch_count >= min_slide_fraction).long()
        patient_id = patient_id_from_filename(path)
        patient_activation_counts.setdefault(
            patient_id, torch.zeros(event_count, dtype=torch.long)
        ).add_(slide_counts)
        patient_patch_counts[patient_id] = patient_patch_counts.get(patient_id, 0) + patch_count
        if file_index % 50 == 0 or file_index == len(files):
            print(
                f"analyzed {file_index}/{len(files)} slides "
                f"({total_patches:,} patches)",
                flush=True,
            )

    patient_coverage_counts = torch.zeros(event_count, dtype=torch.long)
    for patient_id, counts in patient_activation_counts.items():
        fraction = counts.float() / patient_patch_counts[patient_id]
        patient_coverage_counts += (fraction >= min_patient_fraction).long()

    counts_float = activation_counts.double()
    jaccard = _jaccard_from_counts(counts_float, coactivation_counts.double())
    num_v0 = len(v0["event_ids"])
    num_slides = len(files)
    num_patients = len(patient_patch_counts)

    event_records = []
    for index, event_id in enumerate(all_ids):
        event_records.append(
            {
                "bank": "v0" if index < num_v0 else "v2",
                "event_id": event_id,
                "activated_patches": int(activation_counts[index]),
                "patch_activation_frequency": float(activation_counts[index] / total_patches),
                "covered_slides": int(slide_coverage_counts[index]),
                "slide_coverage": float(slide_coverage_counts[index] / num_slides),
                "covered_patients": int(patient_coverage_counts[index]),
                "patient_coverage": float(patient_coverage_counts[index] / num_patients),
            }
        )

    v2_jaccard = jaccard[num_v0:, num_v0:]
    cross_jaccard = jaccard[num_v0:, :num_v0]
    nearest_scores, nearest_indices = cross_jaccard.max(dim=1)
    nearest_v0 = [
        {
            "v2_event_id": v2["event_ids"][index],
            "v0_event_id": v0["event_ids"][int(nearest_indices[index])],
            "activation_jaccard": float(nearest_scores[index]),
        }
        for index in range(len(v2["event_ids"]))
    ]
    nearest_v0.sort(key=lambda item: item["activation_jaccard"], reverse=True)
    v2_records = [item for item in event_records if item["bank"] == "v2"]
    patch_frequencies = torch.tensor(
        [item["patch_activation_frequency"] for item in v2_records]
    )
    patient_coverages = torch.tensor([item["patient_coverage"] for item in v2_records])
    top_jaccard = _pair_records(
        v2_jaccard, v2["event_ids"], v2["event_ids"], same_bank=True
    )
    return {
        "feature_dir": str(feature_dir),
        "threshold_mode": "per_event" if event_score_thresholds is not None else "global",
        "threshold_label": threshold_label,
        "score_threshold": None if event_score_thresholds is not None else score_threshold,
        "event_score_thresholds": (
            None
            if event_score_thresholds is None
            else {
                event_id: float(event_score_thresholds[index])
                for index, event_id in enumerate(all_ids)
            }
        ),
        "min_slide_patch_fraction": min_slide_fraction,
        "min_patient_patch_fraction": min_patient_fraction,
        "num_slides": num_slides,
        "num_patients": num_patients,
        "num_patches": total_patches,
        "v2_summary": {
            "min_patch_activation_frequency": float(patch_frequencies.min()),
            "median_patch_activation_frequency": float(patch_frequencies.median()),
            "max_patch_activation_frequency": float(patch_frequencies.max()),
            "median_patient_coverage": float(patient_coverages.median()),
            "max_within_v2_activation_jaccard": (
                top_jaccard[0]["score"] if top_jaccard else 0.0
            ),
        },
        "events": event_records,
        "within_v2_top_activation_jaccard": top_jaccard,
        "v2_nearest_v0_activation": nearest_v0,
    }


def calibrate_quantile_thresholds(
    v0: dict[str, Any],
    v2: dict[str, Any],
    *,
    feature_dir: Path,
    device: torch.device,
    quantile: float,
    sample_per_slide: int,
    max_files: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate per-event score quantiles from a deterministic slide-balanced sample."""
    files = sorted(feature_dir.glob("*.pt"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No .pt patch feature files found in {feature_dir}")
    embeddings = torch.cat([v0["embeddings"], v2["embeddings"]], dim=0).to(device)
    samples: list[torch.Tensor] = []
    for path in files:
        features = load_patch_embeddings(path)
        if len(features) > sample_per_slide:
            indices = torch.linspace(
                0, len(features) - 1, steps=sample_per_slide, dtype=torch.long
            )
            features = features[indices]
        samples.append((features.to(device) @ embeddings.T).cpu())
    scores = torch.cat(samples, dim=0)
    thresholds = torch.quantile(scores, quantile, dim=0)
    return thresholds, {
        "quantile": quantile,
        "sample_per_slide": sample_per_slide,
        "sampled_patches": int(scores.shape[0]),
        "sampling": "deterministic evenly spaced patches per slide",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    embedding = report["embedding_comparison"]
    lines = [
        "# TCGA-KIRC v0/v2 event-bank audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- v0 events: {report['v0']['num_events']}",
        f"- v2 events: {report['v2']['num_events']}",
        f"- v2 embedding pairs at or above {embedding['redundancy_threshold']:.2f}: "
        f"{len(embedding['within_v2_flagged_pairs'])}",
    ]
    quality = report.get("quality_checks")
    if quality:
        lines.extend(
            [
                f"- embedding redundancy check: {'PASS' if quality['embedding_redundancy_pass'] else 'REVIEW'}",
                f"- offset-invariant activation-overlap check: "
                f"{'PASS' if quality['activation_overlap_pass'] else 'REVIEW'} "
                f"(maximum Jaccard {quality['max_quantile_activation_jaccard']:.4f})",
                f"- cosine 0.20 saturation warning: "
                f"{'YES' if quality['low_threshold_saturated'] else 'NO'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Highest within-v2 embedding similarities",
            "",
            "| Event A | Event B | Cosine |",
            "|---|---|---:|",
        ]
    )
    for item in embedding["within_v2_top_pairs"][:10]:
        lines.append(
            f"| {item['left_event_id']} | {item['right_event_id']} | {item['score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Closest v0 event for each v2 event",
            "",
            "| v2 event | nearest v0 event | Cosine |",
            "|---|---|---:|",
        ]
    )
    for item in embedding["v2_nearest_v0"]:
        lines.append(
            f"| {item['v2_event_id']} | {item['v0_event_id']} | "
            f"{item['cosine_similarity']:.4f} |"
        )

    sensitivity = report.get("activation_sensitivity", [])
    if sensitivity:
        lines.extend(
            [
                "",
                "## Activation-threshold sensitivity",
                "",
                "| Cosine threshold | Median patch frequency | Patch-frequency range | "
                "Median patient coverage | Max within-v2 Jaccard |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for item in sensitivity:
            summary = item["v2_summary"]
            lines.append(
                f"| {item['score_threshold']:.3f} | "
                f"{summary['median_patch_activation_frequency']:.4%} | "
                f"{summary['min_patch_activation_frequency']:.4%}-"
                f"{summary['max_patch_activation_frequency']:.4%} | "
                f"{summary['median_patient_coverage']:.4%} | "
                f"{summary['max_within_v2_activation_jaccard']:.4f} |"
            )

    activation = report.get("activation_comparison")
    if activation:
        lines.extend(
            [
                "",
                "## Primary patch activation audit",
                "",
                f"Scored {activation['num_patches']:,} patches from "
                f"{activation['num_slides']} slides and {activation['num_patients']} patients at "
                f"cosine >= {activation['score_threshold']:.3f}.",
                "",
                "| v2 event | Patch frequency | Slide coverage | Patient coverage |",
                "|---|---:|---:|---:|",
            ]
        )
        v2_events = [item for item in activation["events"] if item["bank"] == "v2"]
        v2_events.sort(key=lambda item: item["patch_activation_frequency"], reverse=True)
        for item in v2_events:
            lines.append(
                f"| {item['event_id']} | {item['patch_activation_frequency']:.4%} | "
                f"{item['slide_coverage']:.4%} | {item['patient_coverage']:.4%} |"
            )
        lines.extend(
            [
                "",
                "## Highest within-v2 activation overlaps",
                "",
                "| Event A | Event B | Activation Jaccard |",
                "|---|---|---:|",
            ]
        )
        for item in activation["within_v2_top_activation_jaccard"][:10]:
            lines.append(
                f"| {item['left_event_id']} | {item['right_event_id']} | {item['score']:.4f} |"
            )

    quantile_activation = report.get("event_quantile_activation")
    if quantile_activation:
        calibration = report["event_quantile_calibration"]
        lines.extend(
            [
                "",
                "## Per-event quantile overlap audit",
                "",
                f"Per-event thresholds were calibrated at the {calibration['quantile']:.1%} "
                f"score quantile from {calibration['sampled_patches']:,} slide-balanced "
                "sampled patches. This removes event-specific cosine offsets when checking "
                "whether different prompts select the same patches.",
                "",
                f"Maximum within-v2 activation Jaccard: "
                f"{quantile_activation['v2_summary']['max_within_v2_activation_jaccard']:.4f}.",
                "",
                "| Event A | Event B | Activation Jaccard |",
                "|---|---|---:|",
            ]
        )
        for item in quantile_activation["within_v2_top_activation_jaccard"][:10]:
            lines.append(
                f"| {item['left_event_id']} | {item['right_event_id']} | {item['score']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-bank", type=Path, required=True)
    parser.add_argument("--v2-bank", type=Path, required=True)
    parser.add_argument("--patch-feature-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--score-threshold",
        type=float,
        action="append",
        dest="score_thresholds",
        help="Repeat for a threshold sweep; defaults to 0.20, 0.30, 0.40, and 0.50.",
    )
    parser.add_argument(
        "--primary-score-threshold",
        type=float,
        help="Threshold whose detailed event table is primary; defaults to the largest threshold.",
    )
    parser.add_argument("--min-slide-patch-fraction", type=float, default=0.01)
    parser.add_argument("--min-patient-patch-fraction", type=float, default=0.01)
    parser.add_argument("--embedding-redundancy-threshold", type=float, default=0.85)
    parser.add_argument("--activation-redundancy-threshold", type=float, default=0.50)
    parser.add_argument(
        "--event-quantile",
        type=float,
        default=0.95,
        help="Per-event score quantile used for offset-invariant overlap analysis.",
    )
    parser.add_argument("--quantile-sample-per-slide", type=int, default=256)
    parser.add_argument("--max-files", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        score_thresholds = args.score_thresholds or [0.20, 0.30, 0.40, 0.50]
        score_thresholds = sorted(set(score_thresholds))
        primary_score_threshold = (
            max(score_thresholds)
            if args.primary_score_threshold is None
            else args.primary_score_threshold
        )
        for name, value in (
            ("minimum slide fraction", args.min_slide_patch_fraction),
            ("minimum patient fraction", args.min_patient_patch_fraction),
            ("embedding redundancy threshold", args.embedding_redundancy_threshold),
            ("activation redundancy threshold", args.activation_redundancy_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for value in score_thresholds:
            if not 0 <= value <= 1:
                raise ValueError("score thresholds must be in [0,1]")
        if primary_score_threshold not in score_thresholds:
            raise ValueError(
                "--primary-score-threshold must be one of the requested --score-threshold values"
            )
        if args.max_files is not None and args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        if not 0 < args.event_quantile < 1:
            raise ValueError("--event-quantile must be in (0,1)")
        if args.quantile_sample_per_slide <= 0:
            raise ValueError("--quantile-sample-per-slide must be positive")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        v0 = load_event_bank(args.v0_bank)
        v2 = load_event_bank(args.v2_bank)
        if v0["embeddings"].shape[1] != v2["embeddings"].shape[1]:
            raise ValueError("v0 and v2 event embedding dimensions differ")
        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "v0": {
                "path": v0["path"],
                "ontology_version": v0["ontology_version"],
                "review_status": v0["review_status"],
                "num_events": len(v0["event_ids"]),
            },
            "v2": {
                "path": v2["path"],
                "ontology_version": v2["ontology_version"],
                "review_status": v2["review_status"],
                "num_events": len(v2["event_ids"]),
            },
            "embedding_comparison": embedding_comparison(
                v0, v2, args.embedding_redundancy_threshold
            ),
        }
        if args.patch_feature_dir:
            sensitivity = []
            for score_threshold in score_thresholds:
                print(f"Running activation audit at cosine >= {score_threshold:.3f}")
                sensitivity.append(
                    activation_comparison(
                        v0,
                        v2,
                        feature_dir=args.patch_feature_dir,
                        device=device,
                        batch_size=args.batch_size,
                        score_threshold=score_threshold,
                        min_slide_fraction=args.min_slide_patch_fraction,
                        min_patient_fraction=args.min_patient_patch_fraction,
                        max_files=args.max_files,
                    )
                )
            report["activation_sensitivity"] = sensitivity
            report["activation_comparison"] = next(
                item
                for item in sensitivity
                if item["score_threshold"] == primary_score_threshold
            )
            print(
                f"Calibrating per-event {args.event_quantile:.1%} score thresholds "
                "for offset-invariant overlap"
            )
            thresholds, calibration = calibrate_quantile_thresholds(
                v0,
                v2,
                feature_dir=args.patch_feature_dir,
                device=device,
                quantile=args.event_quantile,
                sample_per_slide=args.quantile_sample_per_slide,
                max_files=args.max_files,
            )
            report["event_quantile_calibration"] = calibration
            report["event_quantile_activation"] = activation_comparison(
                v0,
                v2,
                feature_dir=args.patch_feature_dir,
                device=device,
                batch_size=args.batch_size,
                score_threshold=0.0,
                event_score_thresholds=thresholds,
                threshold_label=f"per_event_q{args.event_quantile:.3f}",
                min_slide_fraction=args.min_slide_patch_fraction,
                min_patient_fraction=args.min_patient_patch_fraction,
                max_files=args.max_files,
            )
            low_threshold_report = min(
                sensitivity, key=lambda item: item["score_threshold"]
            )
            max_quantile_jaccard = report["event_quantile_activation"]["v2_summary"][
                "max_within_v2_activation_jaccard"
            ]
            report["quality_checks"] = {
                "embedding_redundancy_pass": not report["embedding_comparison"][
                    "within_v2_flagged_pairs"
                ],
                "activation_redundancy_threshold": args.activation_redundancy_threshold,
                "max_quantile_activation_jaccard": max_quantile_jaccard,
                "activation_overlap_pass": (
                    max_quantile_jaccard < args.activation_redundancy_threshold
                ),
                "low_threshold_saturated": (
                    low_threshold_report["v2_summary"]["median_patient_coverage"]
                    >= 0.95
                ),
                "low_threshold": low_threshold_report["score_threshold"],
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        markdown_path = args.output.with_suffix(".md")
        write_markdown(report, markdown_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Event-bank analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved event-bank audit to {args.output}")
    print(f"Saved readable summary to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
