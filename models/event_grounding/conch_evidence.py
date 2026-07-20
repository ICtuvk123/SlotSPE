"""Patch-event evidence computed only in frozen CONCH contrastive space."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FeatureSpaceMismatchError(ValueError):
    """Raised before embeddings from incompatible encoders can be compared."""


class ConchPatchEventEvidence(nn.Module):
    """Compute absolute cosine evidence between CONCH image and text embeddings."""

    MODES = {"raw", "calibrated_sigmoid", "softmax"}
    FEATURE_SPACES = {
        "conch_contrastive",
        "conch_v1_5_contrastive",
        "conch_v1_5_titan_dyko_adapter_768",
    }

    def __init__(
        self,
        support_mode: str = "calibrated_sigmoid",
        delta_patch_event: float = 0.20,
        beta_patch_event: float = 0.10,
        tau_patch_event: float = 0.10,
        eps: float = 1e-8,
        norm_tolerance: float = 5e-3,
    ) -> None:
        super().__init__()
        if support_mode not in self.MODES:
            raise ValueError(f"support_mode must be one of {sorted(self.MODES)}")
        if beta_patch_event <= 0 or tau_patch_event <= 0:
            raise ValueError("beta_patch_event and tau_patch_event must be positive")
        self.support_mode = support_mode
        self.delta_patch_event = float(delta_patch_event)
        self.beta_patch_event = float(beta_patch_event)
        self.tau_patch_event = float(tau_patch_event)
        self.eps = float(eps)
        self.norm_tolerance = float(norm_tolerance)

    @staticmethod
    def validate_feature_spaces(z_feature_space: str, event_feature_space: str) -> None:
        if (
            z_feature_space not in ConchPatchEventEvidence.FEATURE_SPACES
            or event_feature_space not in ConchPatchEventEvidence.FEATURE_SPACES
            or z_feature_space != event_feature_space
        ):
            raise FeatureSpaceMismatchError(
                "Patch-event cosine requires matching CONCH image/text versions in a "
                "supported contrastive space; "
                f"got patch={z_feature_space!r}, event={event_feature_space!r}. "
                "CONCH v1 and v1.5, as well as UNI/ResNet/CTransPath, cannot be "
                "compared across encoders."
            )

    def _validate_normalized(
        self, tensor: torch.Tensor, name: str, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        tensor = tensor.float()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinity")
        norms = tensor.norm(dim=-1)
        checked_norms = norms if valid_mask is None else norms[valid_mask.bool()]
        if checked_norms.numel() == 0:
            raise ValueError(f"{name} contains no valid vectors")
        if (checked_norms <= self.eps).any():
            raise ValueError(f"{name} contains a zero vector")
        if not torch.allclose(
            checked_norms,
            torch.ones_like(checked_norms),
            atol=self.norm_tolerance,
            rtol=self.norm_tolerance,
        ):
            raise ValueError(f"{name} must be L2-normalized CONCH embeddings")
        return F.normalize(tensor, dim=-1, eps=self.eps)

    def forward(
        self,
        z_conch: torch.Tensor,
        event_embeddings: torch.Tensor,
        *,
        z_feature_space: str,
        event_feature_space: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(C_raw, C_support)`` with shape ``[B,M,N]``."""
        self.validate_feature_spaces(z_feature_space, event_feature_space)
        if z_conch.ndim != 3 or event_embeddings.ndim != 2:
            raise ValueError("z_conch must be [B,N,D] and event_embeddings must be [M,D]")
        if z_conch.shape[-1] != event_embeddings.shape[-1]:
            raise ValueError(
                f"CONCH dimension mismatch: patches={z_conch.shape[-1]}, events={event_embeddings.shape[-1]}"
            )
        if mask is not None and (mask.ndim != 2 or mask.shape != z_conch.shape[:2]):
            raise ValueError(f"mask must have shape {tuple(z_conch.shape[:2])}")
        z_conch = self._validate_normalized(z_conch, "z_conch", mask)
        event_embeddings = self._validate_normalized(event_embeddings, "event_embeddings")
        c_raw = torch.einsum("bnd,md->bmn", z_conch, event_embeddings)
        if self.support_mode == "raw":
            support = c_raw
        elif self.support_mode == "calibrated_sigmoid":
            support = torch.sigmoid(
                (c_raw - self.delta_patch_event) / self.beta_patch_event
            )
        else:
            support = torch.softmax(c_raw / self.tau_patch_event, dim=1)
        if mask is not None:
            valid = mask.bool().unsqueeze(1)
            c_raw = c_raw.masked_fill(~valid, 0.0)
            support = support.masked_fill(~valid, 0.0)
        return c_raw, support
