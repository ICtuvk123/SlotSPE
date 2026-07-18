#!/usr/bin/env python3
"""Generate unreviewed pathology-event candidates with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from .event_schema import parse_json_response, validate_event_payload
except ImportError:  # direct script execution
    from event_schema import parse_json_response, validate_event_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "assets/event_bank/prompts/generate_pathology_events.md"
DEFAULT_OUTPUT = ROOT / "assets/event_bank/tcga_kirc/events_raw.json"


def render_prompt(path: Path, cancer_type: str, dataset: str, num_events: int) -> str:
    template = path.read_text(encoding="utf-8")
    return (
        template.replace("{CANCER_TYPE}", cancer_type)
        .replace("{DATASET_NAME}", dataset)
        .replace("{NUM_EVENTS}", str(num_events))
    )


def chat_completion(prompt: str, api_key: str, base_url: str, model: str) -> str:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    request_body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API returned HTTP {exc.code}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LLM API: {exc.reason}") from exc
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM API response lacks choices[0].message.content") from exc


def manual_instructions(output: Path, prompt: Path) -> None:
    print("LLM_API_KEY is not configured; no API request was made.")
    print(f"1. Open: {prompt}")
    print("2. Replace the placeholders and copy the prompt to ChatGPT, Claude, or another LLM.")
    print(f"3. Save the JSON response to: {output}")
    print("4. Re-run this command or continue with review_event_candidates.py.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer-type", default="kidney renal clear cell carcinoma")
    parser.add_argument("--dataset", default="TCGA-KIRC")
    parser.add_argument("--num-events", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_events <= 0:
        print("--num-events must be positive", file=sys.stderr)
        return 2
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        manual_instructions(args.output, args.prompt)
        return 2
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL")
    if not model:
        print("LLM_MODEL must be set when LLM_API_KEY is configured.", file=sys.stderr)
        return 2

    try:
        prompt = render_prompt(args.prompt, args.cancer_type, args.dataset, args.num_events)
        payload = parse_json_response(chat_completion(prompt, api_key, base_url, model))
        payload["status"] = "unreviewed_llm_candidates"
        if payload.get("schema_version") == "2.0":
            metadata = payload.setdefault("generation_metadata", {})
            metadata["generator"] = f"{model} via OpenAI-compatible chat completions"
            metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
            metadata["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            metadata["prompt_file"] = str(args.prompt)
        errors = validate_event_payload(
            payload,
            expected_count=args.num_events,
            expected_status="unreviewed_llm_candidates",
        )
        if errors:
            raise ValueError("Event validation failed:\n- " + "\n- ".join(errors))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Event generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(payload['events'])} unreviewed candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
