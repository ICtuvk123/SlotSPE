"""Structured semantic-and-visual gate for known pathology events."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class KnownEventGate(nn.Module):
    """Compute a gate that cannot bypass semantic or visual evidence."""

    def __init__(
        self,
        delta_sem: float = 0.20,
        beta_sem: float = 0.10,
        delta_vis: float = 0.30,
        beta_vis: float = 0.10,
        use_agreement_gate: bool = True,
        lambda_js: float = 1.0,
        support_mode: str = "calibrated_sigmoid",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if beta_sem <= 0 or beta_vis <= 0:
            raise ValueError("Gate beta values must be positive")
        if lambda_js < 0:
            raise ValueError("lambda_js must be non-negative")
        self.delta_sem = float(delta_sem)
        self.beta_sem = float(beta_sem)
        self.delta_vis = float(delta_vis)
        self.beta_vis = float(beta_vis)
        self.use_agreement_gate = bool(use_agreement_gate)
        self.lambda_js = float(lambda_js)
        self.support_mode = support_mode
        self.eps = float(eps)

    def forward(
        self,
        similarity: torch.Tensor,
        routing: torch.Tensor,
        visual_support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if similarity.ndim != 3 or routing.shape != similarity.shape or visual_support.shape != similarity.shape:
            raise ValueError("similarity, routing and visual_support must share shape [B,K,M]")
        dominant = similarity.argmax(dim=-1)
        r_star = similarity.gather(-1, dominant.unsqueeze(-1)).squeeze(-1)
        pi_star = visual_support.gather(-1, dominant.unsqueeze(-1)).squeeze(-1)
        q_sem = torch.sigmoid((r_star - self.delta_sem) / self.beta_sem)
        q_vis = torch.sigmoid((pi_star - self.delta_vis) / self.beta_vis)

        if self.use_agreement_gate:
            if self.support_mode == "raw":
                pi_positive = F.softplus(visual_support)
            else:
                pi_positive = visual_support.clamp_min(0.0)
            pi_positive = pi_positive + self.eps
            pi_prob = pi_positive / pi_positive.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            a_prob = routing.float().clamp_min(self.eps)
            a_prob = a_prob / a_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            midpoint = 0.5 * (a_prob + pi_prob.float())
            js = 0.5 * (
                (a_prob * (a_prob.log() - midpoint.clamp_min(self.eps).log())).sum(dim=-1)
                + (pi_prob * (pi_prob.log() - midpoint.clamp_min(self.eps).log())).sum(dim=-1)
            )
            q_agree = torch.exp(-self.lambda_js * js.clamp_min(0.0))
        else:
            q_agree = torch.ones_like(q_sem)
        gate = (q_sem * q_vis * q_agree).unsqueeze(-1)
        return {
            "dominant_event": dominant,
            "semantic_score": r_star,
            "visual_score": pi_star,
            "semantic_gate": q_sem.unsqueeze(-1),
            "visual_gate": q_vis.unsqueeze(-1),
            "agreement_gate": q_agree.unsqueeze(-1),
            "event_gate": gate,
        }
