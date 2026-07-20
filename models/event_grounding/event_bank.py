"""Frozen, metadata-preserving CONCH pathology event bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


class FrozenEventBank(nn.Module):
    """Load CONCH text embeddings while retaining human-readable metadata.

    Args:
        path: Artifact emitted by ``tools/build_conch_event_bank.py``.
        trainable: If true, register embeddings as a Parameter; otherwise a buffer.
    """

    def __init__(self, path: str | Path, trainable: bool = False, norm_tolerance: float = 1e-3):
        super().__init__()
        self.path = str(Path(path).expanduser().resolve())
        artifact_path = Path(self.path)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"CONCH event bank does not exist: {artifact_path}. "
                "Build it with tools/build_conch_event_bank.py."
            )
        artifact = _safe_torch_load(artifact_path)
        if not isinstance(artifact, dict):
            raise ValueError("Event-bank artifact must be a dictionary")
        embeddings = artifact.get("event_embeddings")
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError("event_embeddings must be a Tensor with shape [M,D]")
        embeddings = embeddings.detach().float()
        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise ValueError("event_embeddings must be non-empty")
        if not torch.isfinite(embeddings).all():
            raise ValueError("event_embeddings contain NaN or infinity")
        norms = embeddings.norm(dim=-1)
        if (norms <= 0).any() or not torch.allclose(
            norms, torch.ones_like(norms), atol=norm_tolerance, rtol=norm_tolerance
        ):
            raise ValueError("event_embeddings are not L2-normalized")
        feature_space = artifact.get("feature_space")
        if feature_space not in {
            "conch_contrastive",
            "conch_v1_5_contrastive",  # accepted for migration; direct use is rejected by SlotSPE
            "titan_text_768",
        }:
            raise ValueError("Event bank is not marked as a supported pathology text space")

        if trainable:
            self._embeddings = nn.Parameter(embeddings)
        else:
            self.register_buffer("_embeddings", embeddings, persistent=True)
        self.trainable = bool(trainable)
        self.model_name = str(artifact.get("model_name", ""))
        self.encoder_type = str(artifact.get("encoder_type", ""))
        self.feature_space = str(feature_space)
        self.review_status = str(artifact.get("review_status", "unknown"))
        self.cancer_type = str(artifact.get("cancer_type", ""))
        self.dataset = str(artifact.get("dataset", ""))
        self.event_ids = tuple(str(value) for value in artifact.get("event_ids", []))
        self.event_names = tuple(str(value) for value in artifact.get("event_names", []))
        if len(self.event_ids) != embeddings.shape[0] or len(self.event_names) != embeddings.shape[0]:
            raise ValueError("event_ids/event_names length does not match event_embeddings")

    @property
    def embeddings(self) -> torch.Tensor:
        if self.trainable:
            return F.normalize(self._embeddings, dim=-1)
        return self._embeddings

    @property
    def num_events(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self._embeddings.shape[1])

    def extra_repr(self) -> str:
        return (
            f"num_events={self.num_events}, embedding_dim={self.embedding_dim}, "
            f"trainable={self.trainable}, review_status={self.review_status!r}"
        )
