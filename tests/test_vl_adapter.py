import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.SlotSPE import SlotSPE
from models.cross_modal_adapter import DyKoAdapter
from models.event_grounding.known_event_gate import KnownEventGate


def _write_titan_bank(path: Path, num_events: int = 5) -> None:
    torch.save(
        {
            "model_name": "MahmoodLab/TITAN",
            "encoder_type": "TITAN text encoder",
            "feature_space": "titan_text_768",
            "review_status": "machine_curated_pending_pathologist_review",
            "event_ids": [f"event_{index}" for index in range(num_events)],
            "event_names": [f"Event {index}" for index in range(num_events)],
            "event_embeddings": F.normalize(torch.randn(num_events, 768), dim=-1),
        },
        path,
    )


def _args(bank_path: Path, **overrides):
    options = dict(
        omic_sizes=[5, 7], n_classes=4, encoding_dim=768, wsi_projection_dim=16,
        rna_format="Pathways", slot_attention_type="event_gated", slot_num_wsi=4,
        slot_num_omics=3, slot_iters=2, temperature=0.1, topk_ratio=0.5,
        top_k_method="parallel_topk_st", bag_loss="nll_surv", alpha_surv=0.5,
        lambda_recon_loss=0.01, lambda_decoder_loss=1.0,
        event_bank_path=str(bank_path), slot_feature_encoder="conch_v1_5",
        reuse_slot_features_as_conch=True, event_bank_trainable=False,
        event_projection_dim=16, tau_event=0.1, event_gate_start_iter=1,
        patch_event_support_mode="calibrated_sigmoid", delta_patch_event=0.2,
        beta_patch_event=0.1, tau_patch_event=0.1, delta_sem=0.2,
        beta_sem=0.1, delta_vis=0.3, beta_vis=0.1,
        use_agreement_gate=True, lambda_js=1.0, lambda_event=1.0,
        event_residual_dropout=0.0, store_all_iterations=False,
        return_event_details=False, vl_adapter_type="dyko",
        vl_adapter_reduction=4, lambda_vl_alignment=0.1,
    )
    options.update(overrides)
    return SimpleNamespace(**options)


class DyKoAdapterTest(unittest.TestCase):
    def test_shape_norm_and_independent_parameters(self):
        visual = DyKoAdapter(768)
        text = DyKoAdapter(768)
        output = visual(torch.randn(2, 6, 768))
        self.assertEqual(output.shape, (2, 6, 768))
        self.assertTrue(torch.allclose(output.norm(dim=-1), torch.ones(2, 6), atol=1e-5))
        self.assertIsNot(visual.fc[0].weight, text.fc[0].weight)

    def test_zero_padding_stays_zero(self):
        output = DyKoAdapter(768)(torch.zeros(2, 3, 768))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))


class AlignmentLossTest(unittest.TestCase):
    def test_matching_distributions_have_zero_kl(self):
        gate = KnownEventGate(use_agreement_gate=True)
        probabilities = torch.softmax(torch.randn(2, 4, 5), dim=-1)
        result = gate(torch.randn(2, 4, 5), probabilities, probabilities)
        self.assertLess(float(result["alignment_kl"].abs().max()), 1e-6)

    def test_end_to_end_backward_updates_both_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "titan_bank.pt"
            _write_titan_bank(bank_path)
            model = SlotSPE(_args(bank_path)).train()
            patches = F.normalize(torch.randn(2, 12, 768), dim=-1)
            logits, auxiliary = model(
                x_wsi=patches,
                z_conch=patches.clone(),
                patch_mask=torch.ones(2, 12, dtype=torch.bool),
                x_omic1=torch.randn(2, 5),
                x_omic2=torch.randn(2, 7),
                y=torch.tensor([1, 2]),
                c=torch.tensor([0.0, 1.0]),
                omic_missing=False,
            )
            (logits.sum() + auxiliary).backward()
            for adapter in (model.path_adapter, model.text_adapter):
                gradients = [p.grad for p in adapter.parameters()]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)
            self.assertGreaterEqual(float(model.last_loss_components["vl_alignment_loss"]), 0.0)

    def test_raw_v15_and_titan_are_rejected_without_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "titan_bank.pt"
            _write_titan_bank(bank_path)
            with self.assertRaisesRegex(ValueError, "require --vl_adapter_type dyko"):
                SlotSPE(_args(bank_path, vl_adapter_type="none", lambda_vl_alignment=0.0))


if __name__ == "__main__":
    unittest.main()
