import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sksurv.util import Surv

from models.SlotSPE import SlotSPE
from models.gc_rsm import GeneConditionedRankSpaceModulation
from models.slot_attention import MultiHeadSlotAttention
from utils.core_utils import (
    _build_weight_decay_param_groups,
    _fixed_evaluation_rng,
    _fold_eval_slot_seed,
    _init_optim,
    _step,
    _summary,
)
from utils.loss_func import NLLSurvLoss


class FixedEvaluationRngTest(unittest.TestCase):
    def test_fixed_slot_draw_is_repeatable_and_restores_rng(self):
        module = MultiHeadSlotAttention(
            num_slots=4, dim=16, heads=4, dim_head=4, iters=1
        ).eval()
        inputs = torch.randn(2, 12, 16)
        device = torch.device("cpu")

        torch.manual_seed(19)
        state_before = torch.random.get_rng_state()
        with _fixed_evaluation_rng(100003, device):
            first = module(inputs)
        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))

        with _fixed_evaluation_rng(100003, device):
            second = module(inputs)
        with _fixed_evaluation_rng(100004, device):
            different = module(inputs)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))

    def test_fold_offset(self):
        self.assertEqual(_fold_eval_slot_seed(SimpleNamespace(eval_slot_seed=100000), 3), 100003)
        self.assertIsNone(_fold_eval_slot_seed(SimpleNamespace(eval_slot_seed=None), 3))

    class SummaryDataset(Dataset):
        def __init__(self):
            self.times = np.arange(1, 9, dtype=np.float32)
            self.censorship = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
            self.label_df = pd.DataFrame({
                "case id": [f"case_{index}" for index in range(8)],
                "survival_months_dss": self.times,
            })

        def __len__(self):
            return len(self.times)

        def __getitem__(self, index):
            return (
                torch.full((3, 4), float(index)),
                torch.full((5,), float(index)),
                torch.tensor(index % 4),
                torch.tensor(self.times[index]),
                torch.tensor(self.censorship[index]),
            )

    class StochasticSlotLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.zeros(4))

        def forward(self, **kwargs):
            batch_size = kwargs["x_wsi"].shape[0]
            logits = self.offset.expand(batch_size, -1) + torch.randn(batch_size, 4)
            return logits, 0.0

    class TrainableSlotLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = nn.Linear(1, 4)
            self.last_loss_components = {}

        def forward(self, **kwargs):
            feature = kwargs["x_wsi"].mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            logits = self.predictor(feature) + 0.01 * torch.randn(feature.shape[0], 4)
            auxiliary_loss = 0.01 * self.predictor.weight.square().mean()
            self.last_loss_components = {
                "decoder_loss": auxiliary_loss.detach(),
                "reconstruction_loss": torch.zeros_like(auxiliary_loss.detach()),
                "weighted_decoder_loss": auxiliary_loss.detach(),
                "weighted_reconstruction_loss": torch.zeros_like(auxiliary_loss.detach()),
            }
            return logits, auxiliary_loss

    def test_summary_repeats_patient_logits_and_metrics(self):
        dataset = self.SummaryDataset()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_factory = SimpleNamespace(
            label_col="survival_months_dss",
            bins=np.array([1.0, 3.0, 5.0, 8.0]),
        )
        args = SimpleNamespace(
            rna_format="RNASeq", cur_epoch=0, omic_missing=False,
            method="SlotSPE", bag_loss="nll_surv",
        )
        survival = Surv.from_arrays(
            event=(1 - dataset.censorship).astype(bool), time=dataset.times
        )
        model = self.StochasticSlotLikeModel()
        loss = NLLSurvLoss(alpha=0.5)

        first = _summary(
            args, dataset_factory, model, loader, loss, survival,
            eval_slot_seed=100002,
        )
        second = _summary(
            args, dataset_factory, model, loader, loss, survival,
            eval_slot_seed=100002,
        )

        for left, right in zip(first[1:], second[1:]):
            np.testing.assert_equal(left, right)
        for case_id in first[0]:
            self.assertTrue(
                np.array_equal(first[0][case_id]["logits"], second[0][case_id]["logits"])
            )

    def test_two_epoch_step_writes_reproducible_best_checkpoint_metrics(self):
        dataset = self.SummaryDataset()
        train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        clinical_df = pd.DataFrame({
            "survival_months_dss": dataset.times,
            "censorship": dataset.censorship,
        })
        dataset_factory = SimpleNamespace(
            label_col="survival_months_dss", censorship_var="censorship",
            clinical_df=clinical_df, bins=np.array([1.0, 3.0, 5.0, 8.0]),
        )
        model = self.TrainableSlotLikeModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        loss = NLLSurvLoss(alpha=0.5)

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                rna_format="RNASeq", cur_epoch=0, omic_missing=False,
                method="SlotSPE", bag_loss="nll_surv", opt="adam",
                batch_size=2, max_epochs=2, max_cindex=0.0,
                max_cindex_epoch=0, results_dir=directory, eval_slot_seed=100000,
            )
            results, metrics = _step(
                args, 0, loss, model, dataset_factory, optimizer, scheduler,
                train_loader, val_loader, io.StringIO(),
            )
            training_stats = metrics[-1]
            metrics_path = Path(directory) / "fold_0_epoch_metrics.csv"

            self.assertTrue((Path(directory) / "model_best_s0.pth").exists())
            self.assertEqual(len(results), len(dataset))
            self.assertEqual(training_stats["eval_slot_seed"], 100000)
            with metrics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertGreaterEqual(sum(int(row["is_best"]) for row in rows), 1)


class AdamWParameterGroupsTest(unittest.TestCase):
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(8, 8)
            self.norm = nn.LayerNorm(8)
            self.slots_mu = nn.Parameter(torch.randn(1, 1, 8))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, 8))

    def test_decay_exclusions(self):
        model = self.ToyModel()
        groups, names = _build_weight_decay_param_groups(model, 0.01)

        self.assertIn("projection.weight", names["decay"])
        self.assertIn("projection.bias", names["no_decay"])
        self.assertIn("norm.weight", names["no_decay"])
        self.assertIn("norm.bias", names["no_decay"])
        self.assertIn("slots_mu", names["no_decay"])
        self.assertIn("slots_logsigma", names["no_decay"])
        self.assertEqual({group["weight_decay"] for group in groups}, {0.0, 0.01})

    def test_adamw_uses_explicit_weight_decay(self):
        model = self.ToyModel()
        args = SimpleNamespace(opt="adamW", lr=5e-4, reg=1e-3, weight_decay=0.05)
        optimizer = _init_optim(args, model)

        decay_by_name = {
            group["group_name"]: group["weight_decay"] for group in optimizer.param_groups
        }
        self.assertEqual(decay_by_name, {"decay": 0.05, "no_decay": 0.0})

    def test_legacy_adam_still_ignores_reg_without_explicit_option(self):
        model = self.ToyModel()
        args = SimpleNamespace(opt="adam", lr=5e-4, reg=1e-3, weight_decay=None)
        optimizer = _init_optim(args, model)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)


class SlotSPERegularizationTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = dict(
            omic_sizes=[5, 7], n_classes=4, encoding_dim=32, wsi_projection_dim=16,
            rna_format="Pathways", slot_attention_type="original", slot_num_wsi=4,
            slot_num_omics=3, slot_iters=1, temperature=0.1, topk_ratio=0.5,
            top_k_method="parallel_topk_st", bag_loss="nll_surv", alpha_surv=0.5,
            lambda_recon_loss=0.01, lambda_decoder_loss=1.0,
            wsi_projection_dropout=0.0, fusion_dropout=0.0,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _inputs(training=False):
        inputs = dict(
            x_wsi=torch.randn(2, 12, 32),
            x_omic1=torch.randn(2, 5),
            x_omic2=torch.randn(2, 7),
            omic_missing=False,
        )
        if training:
            inputs.update(y=torch.tensor([0, 1]), c=torch.tensor([0.0, 1.0]))
        return inputs

    def test_zero_regularization_matches_legacy_defaults(self):
        legacy_args = self._args()
        del legacy_args.lambda_decoder_loss
        del legacy_args.wsi_projection_dropout
        del legacy_args.fusion_dropout
        legacy = SlotSPE(legacy_args).eval()
        explicit = SlotSPE(self._args()).eval()
        explicit.load_state_dict(legacy.state_dict())
        inputs = self._inputs()

        torch.manual_seed(31)
        expected = legacy(**inputs)[0]
        torch.manual_seed(31)
        actual = explicit(**inputs)[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_feature_gc_rsm_initializes_as_noop_and_backpropagates(self):
        model = SlotSPE(self._args(gc_rsm_mode="feature", gc_rsm_rank=4)).train()
        inputs = self._inputs(training=True)

        with torch.no_grad():
            x_omics = model._encode_omics(inputs)
            context = model._pool_omics_context(x_omics)
            base = model.wsi_mlp(inputs["x_wsi"])
            modulated = model.gc_rsm_feature(inputs["x_wsi"], base, context)
        self.assertTrue(torch.equal(base, modulated))

        logits, auxiliary = model(**inputs)
        (logits.sum() + auxiliary).backward()
        gradients = [p.grad for p in model.gc_rsm_feature.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_slot_qv_gc_rsm_initializes_as_noop_and_backpropagates(self):
        model = SlotSPE(self._args(gc_rsm_mode="slot_qv", gc_rsm_rank=4)).train()
        inputs = self._inputs(training=True)

        with torch.no_grad():
            x_omics = model._encode_omics(inputs)
            context = model._pool_omics_context(x_omics)
            projected = model.wsi_mlp(inputs["x_wsi"])
            modulated = model.gc_rsm_slot_qv(projected, projected, context)
        self.assertTrue(torch.equal(projected, modulated))

        logits, auxiliary = model(**inputs)
        (logits.sum() + auxiliary).backward()
        gradients = [p.grad for p in model.gc_rsm_slot_qv.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_auxiliary_loss_components_use_decoder_weight(self):
        model = SlotSPE(
            self._args(
                lambda_decoder_loss=0.5,
                wsi_projection_dropout=0.1,
                fusion_dropout=0.1,
            )
        ).train()
        _, auxiliary_loss = model(**self._inputs(training=True))
        components = model.last_loss_components

        expected = (
            0.5 * components["decoder_loss"]
            + 0.01 * components["reconstruction_loss"]
        )
        self.assertTrue(torch.allclose(auxiliary_loss.detach(), expected))
        self.assertTrue(torch.allclose(auxiliary_loss.detach(), components["aux_loss"]))


class GcRsmModuleTest(unittest.TestCase):
    def test_shape_validation_and_noop_initialization(self):
        module = GeneConditionedRankSpaceModulation(
            in_dim=8, out_dim=4, gene_dim=6, rank=3
        )
        tokens = torch.randn(2, 5, 8)
        base = torch.randn(2, 5, 4)
        context = torch.randn(2, 6)

        output = module(tokens, base, context)
        self.assertEqual(output.shape, base.shape)
        self.assertTrue(torch.equal(output, base))


if __name__ == "__main__":
    unittest.main()
