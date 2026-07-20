import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.SlotSPE import SlotSPE
from models.event_gated_slot_attention import EventGatedSlotAttention
from models.slot_attention import MultiHeadSlotAttention


class EventGatedSlotAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.x = torch.randn(2, 32, 32)
        self.z = F.normalize(torch.randn(2, 32, 16), dim=-1)
        self.e = F.normalize(torch.randn(5, 16), dim=-1)

    def _module(self, **overrides):
        options = dict(
            num_slots=4,
            dim=32,
            event_embedding_dim=16,
            heads=4,
            dim_head=8,
            iters=2,
            event_projection_dim=16,
            event_gate_start_iter=0,
            use_agreement_gate=False,
        )
        options.update(overrides)
        return EventGatedSlotAttention(**options).eval()

    def test_shapes(self):
        slots, details = self._module()(
            self.x, self.z, self.e,
            z_feature_space="conch_contrastive", return_event_details=True)
        self.assertEqual(slots.shape, (2, 4, 32))
        expected = {
            "B_pre": (2, 4, 32), "B_post": (2, 4, 32),
            "slot_event_similarity": (2, 4, 5), "slot_event_routing": (2, 4, 5),
            "patch_event_evidence": (2, 5, 32), "slot_visual_support": (2, 4, 5),
            "event_gate": (2, 4, 1), "dominant_event": (2, 4),
        }
        for key, shape in expected.items():
            self.assertEqual(details[key].shape, shape, key)

    def test_open_slot_preserves_guided_state(self):
        module = self._module(delta_sem=100.0, delta_vis=100.0)
        _, details = module(self.x, self.z, self.e,
            z_feature_space="conch_contrastive", return_event_details=True)
        self.assertLess(float(details["event_gate"].max()), 1e-6)
        self.assertTrue(torch.equal(details["slots_pre_guidance"], details["S_tilde"]))
        self.assertTrue(torch.equal(details["B_pre"], details["B_post"]))

    def test_known_event_changes_assignment(self):
        module = self._module(delta_sem=-10.0, delta_vis=-10.0, lambda_event=5.0)
        _, details = module(self.x, self.z, self.e,
            z_feature_space="conch_contrastive", return_event_details=True)
        self.assertGreater(float(details["event_gate"].min()), 0.99)
        self.assertGreater(float((details["B_post"] - details["B_pre"]).abs().max()), 0.0)

    def test_store_all_iterations(self):
        _, details = self._module(store_all_iterations=True)(
            self.x, self.z, self.e,
            z_feature_space="conch_contrastive", return_event_details=True)
        self.assertIsInstance(details["event_gate"], list)
        self.assertEqual(len(details["event_gate"]), 2)

    def test_lambda_zero_matches_original(self):
        original = MultiHeadSlotAttention(
            num_slots=4, dim=32, heads=4, dim_head=8, iters=2
        ).eval()
        event = self._module(lambda_event=0.0)
        event.load_state_dict(original.state_dict(), strict=False)
        torch.manual_seed(11)
        expected = original(self.x)
        torch.manual_seed(11)
        actual = event(self.x)
        self.assertTrue(torch.equal(expected, actual))


class OriginalSlotSPECompatibilityTest(unittest.TestCase):
    def test_original_forward_stays_pair(self):
        args = SimpleNamespace(
            omic_sizes=[5, 7], n_classes=4, encoding_dim=32, wsi_projection_dim=16,
            rna_format="Pathways", slot_attention_type="original", slot_num_wsi=4,
            slot_num_omics=3, slot_iters=1, temperature=0.1, topk_ratio=0.5,
            top_k_method="parallel_topk_st", bag_loss="nll_surv", alpha_surv=0.5,
            lambda_recon_loss=0.01,
        )
        model = SlotSPE(args).eval()
        with torch.no_grad():
            output = model(
                x_wsi=torch.randn(2, 12, 32),
                x_omic1=torch.randn(2, 5),
                x_omic2=torch.randn(2, 7),
                omic_missing=False,
            )
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, (2, 4))

    def test_event_gated_slotspe_returns_debug_triple(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "bank.pt"
            event_embeddings = F.normalize(torch.randn(5, 16), dim=-1)
            torch.save(
                {
                    "model_name": "conch_ViT-B-16",
                    "encoder_type": "CONCH text encoder",
                    "feature_space": "conch_contrastive",
                    "review_status": "machine_checked_pending_expert_review",
                    "event_ids": [f"event_{index}" for index in range(5)],
                    "event_names": [f"Event {index}" for index in range(5)],
                    "event_embeddings": event_embeddings,
                },
                bank_path,
            )
            args = SimpleNamespace(
                omic_sizes=[5, 7], n_classes=4, encoding_dim=32, wsi_projection_dim=16,
                rna_format="Pathways", slot_attention_type="event_gated", slot_num_wsi=4,
                slot_num_omics=3, slot_iters=2, temperature=0.1, topk_ratio=0.5,
                top_k_method="parallel_topk_st", bag_loss="nll_surv", alpha_surv=0.5,
                lambda_recon_loss=0.01, event_bank_path=str(bank_path),
                slot_feature_encoder="conch",
                event_bank_trainable=False, event_projection_dim=16, tau_event=0.1,
                event_gate_start_iter=1, patch_event_support_mode="calibrated_sigmoid",
                delta_patch_event=0.2, beta_patch_event=0.1, tau_patch_event=0.1,
                delta_sem=0.2, beta_sem=0.1, delta_vis=0.3, beta_vis=0.1,
                use_agreement_gate=True, lambda_js=1.0, lambda_event=1.0,
                event_residual_dropout=0.0, store_all_iterations=False,
                return_event_details=True,
            )
            model = SlotSPE(args).eval()
            z_conch = F.normalize(torch.randn(2, 12, 16), dim=-1)
            with torch.no_grad():
                output = model(
                    x_wsi=torch.randn(2, 12, 32), z_conch=z_conch,
                    patch_mask=torch.ones(2, 12, dtype=torch.bool),
                    x_omic1=torch.randn(2, 5), x_omic2=torch.randn(2, 7),
                    omic_missing=False,
                )
            self.assertEqual(len(output), 3)
            self.assertEqual(output[0].shape, (2, 4))
            self.assertEqual(output[2]["event_gate"].shape, (2, 4, 1))
            self.assertEqual(len(output[2]["event_names"]), 5)


if __name__ == "__main__":
    unittest.main()
