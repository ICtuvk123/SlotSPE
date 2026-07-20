#!/usr/bin/env python3
"""Load a local TITAN snapshot without hidden Hugging Face network access."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch


TITAN_REQUIRED_FILES = (
    "config.json",
    "configuration_titan.py",
    "modeling_titan.py",
    "model.safetensors",
    "vision_transformer.py",
    "text_transformer.py",
    "conch_tokenizer.py",
    "conch_v1_5.py",
    "tokenizer.json",
)
CONCH_REQUIRED_FILES = (
    "conch_v1_5_pytorch_model.bin",
)


def validate_titan_snapshot(model_dir: str | Path, *, require_conch: bool = False) -> Path:
    """Return a resolved model directory after checking the local snapshot."""
    path = Path(model_dir).expanduser().resolve()
    required = TITAN_REQUIRED_FILES + (CONCH_REQUIRED_FILES if require_conch else ())
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete local TITAN snapshot at {path}; missing: {', '.join(missing)}"
        )
    return path


def load_local_titan(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    return_conch: bool = False,
):
    """Load TITAN and optionally its CONCH v1.5 patch tower fully offline.

    The public TITAN remote code hard-codes ``MahmoodLab/TITAN`` in both the
    tokenizer and CONCH checkpoint loaders.  Redirect only those two requests
    to the verified local snapshot for the duration of model construction.
    """
    model_path = validate_titan_snapshot(model_dir, require_conch=return_conch)

    try:
        import huggingface_hub
        import transformers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError("transformers and huggingface_hub are required for TITAN") from exc

    major = int(transformers.__version__.split(".", maxsplit=1)[0])
    if major >= 5:
        raise RuntimeError(
            "TITAN requires transformers<5 (official version: 4.46.0); "
            f"found {transformers.__version__}"
        )

    original_tokenizer_loader = PreTrainedTokenizerFast.from_pretrained
    original_hub_download = huggingface_hub.hf_hub_download

    def local_tokenizer_loader(model_ref, *args, **kwargs):
        if str(model_ref) == "MahmoodLab/TITAN":
            model_ref = str(model_path)
        kwargs["local_files_only"] = True
        return original_tokenizer_loader(model_ref, *args, **kwargs)

    def local_hub_download(repo_id, filename, *args, **kwargs):
        if str(repo_id) == "MahmoodLab/TITAN":
            local_file = model_path / str(filename)
            if local_file.is_file():
                return str(local_file)
            raise FileNotFoundError(f"Missing local TITAN asset: {local_file}")
        kwargs["local_files_only"] = True
        return original_hub_download(repo_id, filename, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                PreTrainedTokenizerFast,
                "from_pretrained",
                side_effect=local_tokenizer_loader,
            )
        )
        stack.enter_context(
            patch.object(
                huggingface_hub,
                "hf_hub_download",
                side_effect=local_hub_download,
            )
        )
        # Import the snapshot as an isolated local package. This avoids stale
        # ~/.cache/huggingface/modules entries created by an earlier incomplete
        # download while preserving the relative imports in modeling_titan.py.
        digest = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:12]
        package_name = f"_slotspe_titan_{digest}"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(model_path)]
            package.__package__ = package_name
            sys.modules[package_name] = package
        configuration_module = importlib.import_module(
            f"{package_name}.configuration_titan"
        )
        modeling_module = importlib.import_module(f"{package_name}.modeling_titan")
        config = configuration_module.TitanConfig.from_pretrained(
            str(model_path), local_files_only=True
        )
        titan = modeling_module.Titan.from_pretrained(
            str(model_path), config=config, local_files_only=True
        )
        conch_result = titan.return_conch() if return_conch else None

    target_device = torch.device(device)
    titan = titan.to(target_device).eval()
    for parameter in titan.parameters():
        parameter.requires_grad_(False)

    if not return_conch:
        return titan

    conch, preprocess = conch_result
    conch = conch.to(target_device).eval()
    for parameter in conch.parameters():
        parameter.requires_grad_(False)
    return titan, conch, preprocess
