#!/usr/bin/env python3
"""Machine-check event candidates and prepare a file for expert review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .event_schema import (
        is_v2_payload,
        load_json,
        name_tokens,
        normalize_name,
        token_jaccard,
        validate_event_payload,
    )
except ImportError:  # direct script execution
    from event_schema import (
        is_v2_payload,
        load_json,
        name_tokens,
        normalize_name,
        token_jaccard,
        validate_event_payload,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "assets/event_bank/tcga_kirc/events_raw.json"
DEFAULT_OUTPUT = ROOT / "assets/event_bank/tcga_kirc/events_reviewed.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-event-id", action="append", default=[])
    parser.add_argument("--keep-event-ids-file", type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--content-similarity-threshold", type=float, default=0.68)
    parser.add_argument(
        "--curation-file",
        type=Path,
        help="V2 decision file with one core/context/reserve/exclude decision per event.",
    )
    parser.add_argument("--context-output", type=Path)
    parser.add_argument("--reserve-output", type=Path)
    return parser


def requested_ids(args: argparse.Namespace) -> set[str]:
    selected = set(args.keep_event_id)
    if args.keep_event_ids_file:
        for line in args.keep_event_ids_file.read_text(encoding="utf-8").splitlines():
            event_id = line.strip()
            if event_id and not event_id.startswith("#"):
                selected.add(event_id)
    return selected


def _content_tokens(event: dict[str, Any]) -> set[str]:
    fields = [event.get("event_name", ""), event.get("definition", "")]
    fields.extend(event.get("visual_cues", []))
    fields.extend(event.get("exclusion_cues", []))
    return name_tokens(" ".join(value for value in fields if isinstance(value, str)))


def content_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Lexical-content screen; CONCH similarity is checked after encoding."""
    left_tokens, right_tokens = _content_tokens(left), _content_tokens(right)
    union = left_tokens | right_tokens
    content_score = len(left_tokens & right_tokens) / len(union) if union else 1.0
    name_score = token_jaccard(left.get("event_name", ""), right.get("event_name", ""))
    group_score = float(
        bool(left.get("semantic_group"))
        and left.get("semantic_group") == right.get("semantic_group")
    )
    return 0.45 * name_score + 0.40 * content_score + 0.15 * group_score


def _load_curation(path: Path, event_ids: set[str]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read curation file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Curation file must be a top-level object")
    for field in ("curator", "review_date", "criteria", "decisions"):
        if field not in payload:
            raise ValueError(f"Curation file is missing {field!r}")
    if not isinstance(payload["criteria"], list) or not payload["criteria"]:
        raise ValueError("Curation criteria must be a non-empty list")
    decisions: dict[str, dict[str, str]] = {}
    allowed = {"core", "context", "reserve", "exclude"}
    if not isinstance(payload["decisions"], list):
        raise ValueError("Curation decisions must be a list")
    for index, item in enumerate(payload["decisions"]):
        if not isinstance(item, dict):
            raise ValueError(f"curation decisions[{index}] must be an object")
        event_id, decision, reason = (
            item.get("event_id"),
            item.get("decision"),
            item.get("reason"),
        )
        if not isinstance(event_id, str) or event_id not in event_ids:
            raise ValueError(f"Unknown curation event_id at decisions[{index}]: {event_id!r}")
        if event_id in decisions:
            raise ValueError(f"Duplicate curation decision for {event_id}")
        if decision not in allowed:
            raise ValueError(f"Invalid decision for {event_id}: {decision!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Curation reason is required for {event_id}")
        decisions[event_id] = {
            "event_id": event_id,
            "decision": decision,
            "reason": reason.strip(),
        }
    missing = event_ids - set(decisions)
    if missing:
        raise ValueError(f"Curation decisions are missing event IDs: {sorted(missing)}")
    return payload, decisions


def _curated_payload(
    payload: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    status: str,
    subset: str,
    curation: dict[str, Any],
    decisions: dict[str, dict[str, str]],
) -> dict[str, Any]:
    result = dict(payload)
    result["events"] = events
    result["status"] = status
    result["curation_metadata"] = {
        "subset": subset,
        "curator": curation["curator"],
        "review_date": curation["review_date"],
        "pathologist_reviewed": bool(curation.get("pathologist_reviewed", False)),
        "criteria": curation["criteria"],
        "decisions": [decisions[event["event_id"]] for event in events],
    }
    return result


def _write_payload(payload: dict[str, Any], path: Path) -> None:
    errors = validate_event_payload(payload)
    if errors:
        raise ValueError("Curated subset validation failed:\n- " + "\n- ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = load_json(args.input)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    errors = validate_event_payload(payload)
    events = payload.get("events", [])
    selected = requested_ids(args)
    if args.curation_file and selected:
        errors.append("--curation-file cannot be combined with keep-event ID options")
    if (args.context_output or args.reserve_output) and not args.curation_file:
        errors.append("--context-output/--reserve-output require --curation-file")
    all_ids = {event.get("event_id") for event in events if isinstance(event, dict)}
    unknown = selected - all_ids
    if unknown:
        errors.append(f"requested event IDs not found: {sorted(unknown)}")

    exact_names: dict[str, str] = {}
    near_duplicates: list[tuple[str, str, float]] = []
    content_duplicates: list[tuple[str, str, float]] = []
    semantic_groups: dict[str, list[str]] = {}
    for index, left in enumerate(events):
        normalized = normalize_name(left.get("event_name", ""))
        if normalized in exact_names:
            errors.append(
                f"duplicate normalized event_name: {exact_names[normalized]} and {left.get('event_id')}"
            )
        exact_names[normalized] = left.get("event_id", "")
        if left.get("semantic_group"):
            semantic_groups.setdefault(left["semantic_group"], []).append(
                left.get("event_id", "")
            )
        for right in events[index + 1 :]:
            score = token_jaccard(left.get("event_name", ""), right.get("event_name", ""))
            if score >= args.similarity_threshold:
                near_duplicates.append((left.get("event_id", ""), right.get("event_id", ""), score))
            content_score = content_similarity(left, right)
            if content_score >= args.content_similarity_threshold:
                content_duplicates.append(
                    (left.get("event_id", ""), right.get("event_id", ""), content_score)
                )

    print(f"Input events: {len(events)}")
    print(f"Schema errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Potential near-duplicate pairs: {len(near_duplicates)}")
    for left, right, score in near_duplicates:
        print(f"WARNING: {left} ~ {right} (token overlap={score:.3f})")
    print(f"Potential content-similar pairs: {len(content_duplicates)}")
    for left, right, score in content_duplicates:
        print(f"WARNING: {left} ~ {right} (content similarity={score:.3f})")
    related_groups = {
        group: event_ids
        for group, event_ids in semantic_groups.items()
        if len(event_ids) > 1
    }
    print(f"Multi-event semantic groups requiring explicit curation: {len(related_groups)}")
    for group, event_ids in sorted(related_groups.items()):
        print(f"REVIEW: {group}: {', '.join(event_ids)}")
    if errors:
        print("No reviewed file was written because validation failed.", file=sys.stderr)
        return 1

    try:
        if args.curation_file:
            if not is_v2_payload(payload):
                raise ValueError("--curation-file is supported only for schema v2 payloads")
            curation, decisions = _load_curation(args.curation_file, all_ids)
            buckets = {
                decision: [
                    event
                    for event in events
                    if decisions[event["event_id"]]["decision"] == decision
                ]
                for decision in ("core", "context", "reserve", "exclude")
            }
            core = buckets["core"]
            if not 20 <= len(core) <= 25:
                raise ValueError(f"V2 core must contain 20-25 events, found {len(core)}")
            invalid_core = [
                event["event_id"]
                for event in core
                if event.get("observation_scale") != "patch_local"
                or event.get("patch_observable") != "yes"
            ]
            if invalid_core:
                raise ValueError(
                    "V2 core may contain only directly patch-observable local events: "
                    f"{invalid_core}"
                )
            reviewed = _curated_payload(
                payload,
                core,
                status="machine_curated_pending_pathologist_review",
                subset="patch_core",
                curation=curation,
                decisions=decisions,
            )
            _write_payload(reviewed, args.output)
            if args.context_output:
                context = _curated_payload(
                    payload,
                    buckets["context"],
                    status="context_catalog_pending_pathologist_review",
                    subset="interface_and_wsi_context",
                    curation=curation,
                    decisions=decisions,
                )
                _write_payload(context, args.context_output)
            if args.reserve_output:
                reserve = _curated_payload(
                    payload,
                    buckets["reserve"],
                    status="reserve_catalog_pending_pathologist_review",
                    subset="reserve",
                    curation=curation,
                    decisions=decisions,
                )
                _write_payload(reserve, args.reserve_output)
            print(
                "Curation counts: "
                + ", ".join(f"{key}={len(value)}" for key, value in buckets.items())
            )
            print(f"Wrote v2 patch-local core to {args.output}")
        else:
            if selected:
                events = [event for event in events if event["event_id"] in selected]
                print(f"Keep-list retained {len(events)} events.")
            reviewed = dict(payload)
            reviewed["events"] = events
            reviewed["status"] = "machine_checked_pending_expert_review"
            _write_payload(reviewed, args.output)
            print(f"Wrote machine-checked candidates to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"Curation failed: {exc}", file=sys.stderr)
        return 1
    print("These events have NOT been reviewed by a pathology expert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
