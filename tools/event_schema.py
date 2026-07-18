"""Schema validation shared by pathology event-bank command line tools."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


ALLOWED_CATEGORIES = {
    "tumor_architecture",
    "proliferation",
    "necrosis",
    "invasion",
    "vasculature",
    "immune_microenvironment",
    "stroma",
    "differentiation",
    "tissue_interface",
    "other",
}

V2_SCHEMA_VERSION = "2.0"
V2_EVIDENCE_TIERS = {
    "tier_1_consensus",
    "tier_2_cohort_supported",
    "tier_3_exploratory",
}
V2_OBSERVATION_SCALES = {
    "patch_local",
    "interface_context",
    "wsi_anatomic_context",
}
V2_PATCH_OBSERVABILITY = {"yes", "conditional", "no"}

FORBIDDEN_PHRASES = (
    "poor prognosis",
    "good prognosis",
    "high risk",
    "low risk",
    "long survival",
    "short survival",
)

REQUIRED_EVENT_FIELDS = {
    "event_id",
    "event_name",
    "category",
    "definition",
    "visual_cues",
    "exclusion_cues",
    "prognostic_rationale",
    "prompt_templates",
}

V2_REQUIRED_EVENT_FIELDS = REQUIRED_EVENT_FIELDS | {
    "evidence_tier",
    "observation_scale",
    "patch_observable",
    "required_context",
    "confounders",
    "source_basis",
    "semantic_group",
}

V2_REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "ontology_version",
    "cancer_type",
    "dataset",
    "status",
    "patch_spec",
    "generation_metadata",
    "sources",
    "events",
}


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object with a concise, user-facing error."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Event JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level event JSON must be an object: {path}")
    return payload


def iter_strings(value: Any) -> Iterable[str]:
    """Yield every string nested inside a JSON-compatible value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def normalize_name(value: str) -> str:
    """Normalize a label for duplicate checks without external packages."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def name_tokens(value: str) -> set[str]:
    return set(normalize_name(value).split())


def token_jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = name_tokens(left), name_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def is_v2_payload(payload: dict[str, Any]) -> bool:
    """Return whether a payload opts into the scale/evidence-aware v2 schema."""
    return payload.get("schema_version") == V2_SCHEMA_VERSION


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_is_nonempty_string(item) for item in value)
    )


def validate_event_payload(
    payload: dict[str, Any],
    *,
    expected_count: int | None = None,
    expected_status: str | None = None,
) -> list[str]:
    """Return validation errors; an empty list means the payload is valid."""
    errors: list[str] = []
    v2 = is_v2_payload(payload)
    top_level_fields = (
        V2_REQUIRED_TOP_LEVEL_FIELDS
        if v2
        else {"cancer_type", "dataset", "status", "events"}
    )
    for field in sorted(top_level_fields):
        if field not in payload:
            errors.append(f"missing top-level field: {field}")

    if "schema_version" in payload and not v2:
        errors.append(
            f"unsupported schema_version: {payload.get('schema_version')!r}; "
            f"expected {V2_SCHEMA_VERSION!r}"
        )

    source_ids: set[str] = set()
    if v2:
        ontology_version = payload.get("ontology_version")
        if not _is_nonempty_string(ontology_version):
            errors.append("ontology_version must be a non-empty string")
        patch_spec = payload.get("patch_spec")
        if not isinstance(patch_spec, dict):
            errors.append("patch_spec must be an object")
        else:
            if patch_spec.get("width_px") != 448 or patch_spec.get("height_px") != 448:
                errors.append("patch_spec must describe 448x448-pixel inputs")
            if patch_spec.get("magnification") != 20:
                errors.append("patch_spec.magnification must be 20")
            if str(patch_spec.get("stain", "")).casefold() not in {"h&e", "he"}:
                errors.append("patch_spec.stain must be H&E")
        generation_metadata = payload.get("generation_metadata")
        if not isinstance(generation_metadata, dict):
            errors.append("generation_metadata must be an object")
        else:
            for field in ("prompt_version", "generator", "generated_at"):
                if not _is_nonempty_string(generation_metadata.get(field)):
                    errors.append(f"generation_metadata.{field} must be a non-empty string")
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append("sources must be a non-empty list")
        else:
            snake_case = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
            for index, source in enumerate(sources):
                prefix = f"sources[{index}]"
                if not isinstance(source, dict):
                    errors.append(f"{prefix} must be an object")
                    continue
                source_id = source.get("source_id")
                if not isinstance(source_id, str) or not snake_case.fullmatch(source_id):
                    errors.append(f"{prefix}.source_id must be lowercase snake_case")
                elif source_id in source_ids:
                    errors.append(f"duplicate source_id: {source_id}")
                else:
                    source_ids.add(source_id)
                for field in ("citation", "url", "evidence_scope"):
                    if not _is_nonempty_string(source.get(field)):
                        errors.append(f"{prefix}.{field} must be a non-empty string")

    events = payload.get("events")
    if not isinstance(events, list):
        return errors + ["events must be a list"]
    if expected_count is not None and len(events) != expected_count:
        errors.append(f"expected {expected_count} events, found {len(events)}")
    if expected_status is not None and payload.get("status") != expected_status:
        errors.append(
            f"status must be {expected_status!r}, found {payload.get('status')!r}"
        )

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    snake_case = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    for index, event in enumerate(events):
        prefix = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix} must be an object")
            continue
        required_fields = V2_REQUIRED_EVENT_FIELDS if v2 else REQUIRED_EVENT_FIELDS
        missing = required_fields - set(event)
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
        event_id = event.get("event_id")
        event_name = event.get("event_name")
        if not isinstance(event_id, str) or not snake_case.fullmatch(event_id):
            errors.append(f"{prefix}.event_id must be lowercase snake_case")
        elif event_id in seen_ids:
            errors.append(f"duplicate event_id: {event_id}")
        else:
            seen_ids.add(event_id)
        if not isinstance(event_name, str) or not event_name.strip():
            errors.append(f"{prefix}.event_name must be a non-empty string")
        else:
            normalized = normalize_name(event_name)
            if normalized in seen_names:
                errors.append(f"duplicate event_name: {event_name}")
            seen_names.add(normalized)
        if event.get("category") not in ALLOWED_CATEGORIES:
            errors.append(f"{prefix}.category is not allowed: {event.get('category')!r}")
        for text_field in ("definition", "prognostic_rationale"):
            if not _is_nonempty_string(event.get(text_field)):
                errors.append(f"{prefix}.{text_field} must be a non-empty string")
        for list_field in ("visual_cues", "exclusion_cues"):
            value = event.get(list_field)
            if not _is_nonempty_string_list(value):
                errors.append(f"{prefix}.{list_field} must be a non-empty string list")
        prompts = event.get("prompt_templates")
        if not isinstance(prompts, list) or len(prompts) != 4 or not all(
            isinstance(item, str) and item.strip() for item in (prompts or [])
        ):
            errors.append(f"{prefix}.prompt_templates must contain exactly four strings")

        if v2:
            evidence_tier = event.get("evidence_tier")
            observation_scale = event.get("observation_scale")
            patch_observable = event.get("patch_observable")
            if evidence_tier not in V2_EVIDENCE_TIERS:
                errors.append(f"{prefix}.evidence_tier is not allowed: {evidence_tier!r}")
            if observation_scale not in V2_OBSERVATION_SCALES:
                errors.append(
                    f"{prefix}.observation_scale is not allowed: {observation_scale!r}"
                )
            if patch_observable not in V2_PATCH_OBSERVABILITY:
                errors.append(
                    f"{prefix}.patch_observable is not allowed: {patch_observable!r}"
                )
            if observation_scale == "patch_local" and patch_observable == "no":
                errors.append(
                    f"{prefix} cannot be patch_local while patch_observable is 'no'"
                )
            if observation_scale == "wsi_anatomic_context" and patch_observable != "no":
                errors.append(
                    f"{prefix} WSI/anatomic events must set patch_observable to 'no'"
                )
            for list_field in ("required_context", "confounders", "source_basis"):
                if not _is_nonempty_string_list(event.get(list_field)):
                    errors.append(f"{prefix}.{list_field} must be a non-empty string list")
            if _is_nonempty_string_list(event.get("source_basis")):
                unknown_sources = set(event["source_basis"]) - source_ids
                if unknown_sources:
                    errors.append(
                        f"{prefix}.source_basis references unknown sources: "
                        f"{sorted(unknown_sources)}"
                    )
            semantic_group = event.get("semantic_group")
            if not isinstance(semantic_group, str) or not snake_case.fullmatch(semantic_group):
                errors.append(f"{prefix}.semantic_group must be lowercase snake_case")

        searchable = " ".join(iter_strings(event)).casefold()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in searchable:
                errors.append(f"{prefix} contains forbidden phrase: {phrase!r}")
    return errors


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a model response, tolerating only an outer JSON code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().casefold() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a top-level JSON object")
    return value
