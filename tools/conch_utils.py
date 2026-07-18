"""Shared, strict loading helpers for the official CONCH package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


CONCH_MODEL_NAME = "conch_ViT-B-16"
CONCH_HF_REPOSITORY = "MahmoodLab/conch"
CONCH_ACCESS_URL = "https://huggingface.co/MahmoodLab/CONCH"


def import_conch() -> tuple[Any, Any, Any]:
    """Import the official public API or raise an actionable error."""
    try:
        from conch.open_clip_custom import (  # type: ignore[import-not-found]
            create_model_from_pretrained,
            get_tokenizer,
            tokenize,
        )
    except ImportError as exc:
        raise RuntimeError(
            "CONCH is not installed. Run `bash scripts/setup_conch.sh` first."
        ) from exc
    return create_model_from_pretrained, get_tokenizer, tokenize


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use --device cpu or run on a CUDA-enabled node."
        )
    return resolved


def load_frozen_conch(
    *,
    checkpoint: str | Path | None,
    from_hf: bool,
    device: str,
) -> tuple[torch.nn.Module, Any, Any, Any, torch.device]:
    """Load real CONCH weights and freeze all parameters.

    Random initialization is intentionally never available through this helper.
    """
    create_model_from_pretrained, get_tokenizer, tokenize = import_conch()
    resolved_device = resolve_device(device)
    try:
        if from_hf:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "HF_TOKEN is required for gated CONCH weights. Request access at "
                    f"{CONCH_ACCESS_URL}, export HF_TOKEN, then retry."
                )
            model, preprocess = create_model_from_pretrained(
                CONCH_MODEL_NAME,
                f"hf_hub:{CONCH_HF_REPOSITORY}",
                hf_auth_token=token,
            )
        else:
            if checkpoint is None:
                raise RuntimeError("A local --checkpoint is required unless --from-hf is used.")
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise RuntimeError(
                    f"CONCH checkpoint does not exist: {checkpoint_path}. "
                    f"Request access at {CONCH_ACCESS_URL}; no random weights will be used."
                )
            model, preprocess = create_model_from_pretrained(
                CONCH_MODEL_NAME,
                checkpoint_path=str(checkpoint_path),
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Failed to load gated CONCH weights. Verify access/checkpoint integrity at "
            f"{CONCH_ACCESS_URL}. Original error: {exc}"
        ) from exc

    model = model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess, get_tokenizer(), tokenize, resolved_device


def conch_tokenize(tokenize: Any, tokenizer: Any, texts: list[str]) -> torch.Tensor:
    """Call the documented tokenizer API with compatibility for older CONCH tags."""
    try:
        return tokenize(texts=texts, tokenizer=tokenizer)
    except TypeError:
        # Older CONCH releases used positional arguments.
        return tokenize(tokenizer, texts)
    except AttributeError as exc:
        # transformers 5 removed PreTrainedTokenizerFast.batch_encode_plus,
        # which the current public CONCH tokenizer wrapper still calls.  Use
        # the tokenizer's equivalent callable API while preserving CONCH's
        # 127-token context plus its reserved final placeholder token.
        if "batch_encode_plus" not in str(exc):
            raise
        encoded = tokenizer(
            texts,
            max_length=127,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        return torch.nn.functional.pad(input_ids, (0, 1), value=tokenizer.pad_token_id)
