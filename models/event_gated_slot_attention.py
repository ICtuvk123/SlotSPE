"""Event-gated extension of SlotSPE's original multi-head Slot Attention."""

from __future__ import annotations

from typing import Any

import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import pack, repeat, unpack

from models.event_grounding import ConchPatchEventEvidence, KnownEventGate
from models.slot_attention import MultiHeadSlotAttention


class EventGatedSlotAttention(MultiHeadSlotAttention):
    """Inject frozen CONCH event guidance inside every enabled slot iteration.

    The inherited visual assignment, value aggregation, GRU update, and MLP are
    preserved. Event guidance is a gated residual and therefore cannot replace
    an open slot's visual state.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        dim: int,
        event_embedding_dim: int,
        heads: int = 4,
        dim_head: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        event_projection_dim: int = 256,
        tau_event: float = 0.10,
        event_gate_start_iter: int = 1,
        patch_event_support_mode: str = "calibrated_sigmoid",
        delta_patch_event: float = 0.20,
        beta_patch_event: float = 0.10,
        tau_patch_event: float = 0.10,
        delta_sem: float = 0.20,
        beta_sem: float = 0.10,
        delta_vis: float = 0.30,
        beta_vis: float = 0.10,
        use_agreement_gate: bool = True,
        lambda_js: float = 1.0,
        lambda_event: float = 1.0,
        event_residual_dropout: float = 0.0,
        store_all_iterations: bool = False,
    ) -> None:
        super().__init__(
            num_slots=num_slots,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            iters=iters,
            eps=eps,
            hidden_dim=hidden_dim,
        )
        if event_projection_dim <= 0 or event_embedding_dim <= 0:
            raise ValueError("Event embedding/projection dimensions must be positive")
        if tau_event <= 0 or event_gate_start_iter < 0:
            raise ValueError("tau_event must be positive and event_gate_start_iter non-negative")
        if lambda_event < 0 or not 0.0 <= event_residual_dropout < 1.0:
            raise ValueError("lambda_event must be non-negative and dropout must be in [0,1)")
        self.event_embedding_dim = int(event_embedding_dim)
        self.event_projection_dim = int(event_projection_dim)
        self.tau_event = float(tau_event)
        self.event_gate_start_iter = int(event_gate_start_iter)
        self.lambda_event = float(lambda_event)
        self.store_all_iterations = bool(store_all_iterations)

        self.slot_semantic_projection = nn.Linear(dim, event_projection_dim)
        self.event_semantic_projection = nn.Linear(event_embedding_dim, event_projection_dim)
        self.event_value_projection = nn.Linear(event_embedding_dim, event_projection_dim)
        # No bias: a zero gate must produce an exactly zero residual.
        self.event_output_projection = nn.Linear(event_projection_dim, dim, bias=False)
        self.event_residual_dropout = nn.Dropout(event_residual_dropout)
        self.patch_event_evidence = ConchPatchEventEvidence(
            support_mode=patch_event_support_mode,
            delta_patch_event=delta_patch_event,
            beta_patch_event=beta_patch_event,
            tau_patch_event=tau_patch_event,
            eps=eps,
        )
        self.known_event_gate = KnownEventGate(
            delta_sem=delta_sem,
            beta_sem=beta_sem,
            delta_vis=delta_vis,
            beta_vis=beta_vis,
            use_agreement_gate=use_agreement_gate,
            lambda_js=lambda_js,
            support_mode=patch_event_support_mode,
            eps=eps,
        )

    def _initialize_slots(self, inputs: torch.Tensor, num_slots: int) -> torch.Tensor:
        batch = inputs.shape[0]
        mu = repeat(self.slots_mu, "1 1 d -> b s d", b=batch, s=num_slots)
        sigma = repeat(self.slots_logsigma.exp(), "1 1 d -> b s d", b=batch, s=num_slots)
        return mu + sigma * torch.randn(mu.shape, device=inputs.device, dtype=inputs.dtype)

    def _assign(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.split_heads(self.to_q(self.norm_slots(slots)))
        logits = einsum("... i d, ... j d -> ... i j", queries, keys) * self.scale
        competition = logits.softmax(dim=-2)
        if mask is None:
            aggregation = F.normalize(competition + self.eps, p=1, dim=-1)
        else:
            valid = mask.bool().unsqueeze(1).unsqueeze(1)
            competition = competition.masked_fill(~valid, 0.0)
            numerator = (competition + self.eps).masked_fill(~valid, 0.0)
            aggregation = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return competition, aggregation

    def _head_average(
        self, competition: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assignment = competition.mean(dim=1)
        if mask is not None:
            assignment = assignment.masked_fill(~mask.bool().unsqueeze(1), 0.0)
        normalized = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return assignment, normalized

    def _slot_event_match(
        self, slots: torch.Tensor, event_semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_semantic = F.normalize(
            self.slot_semantic_projection(slots).float(), dim=-1, eps=self.eps
        )
        similarity = torch.einsum("bkd,md->bkm", slot_semantic, event_semantic)
        routing = torch.softmax(similarity / self.tau_event, dim=-1)
        return similarity, routing

    def forward(
        self,
        inputs: torch.Tensor,
        z_conch: torch.Tensor | None = None,
        event_embeddings: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        num_slots: int | None = None,
        z_feature_space: str = "unknown",
        event_feature_space: str = "conch_contrastive",
        return_event_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if self.lambda_event == 0.0 and not return_event_details:
            return super().forward(inputs, num_slots=num_slots)
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"inputs must have shape [B,N,{self.dim}]")
        if mask is not None and (mask.ndim != 2 or mask.shape != inputs.shape[:2]):
            raise ValueError(f"mask must have shape {tuple(inputs.shape[:2])}")
        if z_conch is None or event_embeddings is None:
            raise ValueError("Event-gated attention requires z_conch and event_embeddings")
        if event_embeddings.ndim != 2 or event_embeddings.shape[1] != self.event_embedding_dim:
            raise ValueError(f"event_embeddings must have shape [M,{self.event_embedding_dim}]")

        n_slots = num_slots if num_slots is not None else self.num_slots
        slots = self._initialize_slots(inputs, n_slots)
        normalized_inputs = self.norm_input(inputs)
        keys, values = self.to_k(normalized_inputs), self.to_v(normalized_inputs)
        keys, values = map(self.split_heads, (keys, values))
        c_raw, patch_support = self.patch_event_evidence(
            z_conch,
            event_embeddings,
            z_feature_space=z_feature_space,
            event_feature_space=event_feature_space,
            mask=mask,
        )
        event_semantic = F.normalize(
            self.event_semantic_projection(event_embeddings).float(), dim=-1, eps=self.eps
        )
        event_values = self.event_value_projection(event_embeddings)

        iteration_details: list[dict[str, torch.Tensor]] = []
        for iteration in range(self.iters):
            slots_prev = slots
            pre_competition, pre_aggregation = self._assign(slots_prev, keys, mask)
            b_pre, bbar_pre = self._head_average(pre_competition, mask)
            similarity, routing = self._slot_event_match(slots_prev, event_semantic)
            visual_support = torch.einsum("bkn,bmn->bkm", bbar_pre.float(), patch_support)
            gates = self.known_event_gate(similarity, routing, visual_support)
            effective_gate = gates["event_gate"]
            if iteration < self.event_gate_start_iter:
                effective_gate = torch.zeros_like(effective_gate)

            guidance = torch.einsum(
                "bkm,md->bkd", routing.to(event_values.dtype), event_values
            )
            residual = self.event_output_projection(effective_gate.to(guidance.dtype) * guidance)
            residual = self.event_residual_dropout(residual)
            guided_slots = slots_prev + self.lambda_event * residual

            if iteration < self.event_gate_start_iter or self.lambda_event == 0.0:
                post_competition, post_aggregation = pre_competition, pre_aggregation
            else:
                post_competition, post_aggregation = self._assign(guided_slots, keys, mask)
            b_post, bbar_post = self._head_average(post_competition, mask)

            updates = einsum("... j d, ... i j -> ... i d", values, post_aggregation)
            updates = self.combine_heads(self.merge_heads(updates))
            updates, packed_shape = pack([updates], "* d")
            hidden, _ = pack([guided_slots], "* d")
            slots = self.gru(updates, hidden)
            slots, = unpack(slots, packed_shape, "* d")
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            if return_event_details:
                detail = {
                    "B_pre": b_pre,
                    "Bbar_pre": bbar_pre,
                    "B_post": b_post,
                    "Bbar_post": bbar_post,
                    "slot_event_similarity": similarity,
                    "slot_event_routing": routing,
                    "patch_event_similarity_raw": c_raw,
                    "patch_event_evidence": patch_support,
                    "slot_visual_support": visual_support,
                    "semantic_gate": gates["semantic_gate"],
                    "visual_gate": gates["visual_gate"],
                    "agreement_gate": gates["agreement_gate"],
                    "event_gate": effective_gate,
                    "dominant_event": gates["dominant_event"],
                    "semantic_score": gates["semantic_score"],
                    "visual_score": gates["visual_score"],
                    "slots_pre_guidance": slots_prev,
                    "S_tilde": guided_slots,
                    "slots": slots,
                }
                iteration_details.append(detail)

        if not return_event_details:
            return slots
        if self.store_all_iterations:
            details: dict[str, Any] = {
                key: [item[key] for item in iteration_details] for key in iteration_details[0]
            }
        else:
            details = iteration_details[-1]
        return slots, details
