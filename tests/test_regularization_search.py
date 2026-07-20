import csv
import tempfile
import unittest
from pathlib import Path

from scripts.regularization_search import (
    CONFIG_FIELDS,
    make_task,
    plan_combine,
    plan_dropout,
    plan_optimizer,
    plan_phase,
    plan_reference,
    result_for_task,
    write_final_report,
    write_tasks,
)


class RegularizationSearchPlannerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.state_dir = self.root / "state"
        self.results_dir = self.root / "results"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_summary(self, task, cindex, gap, std=0.02, ipcw=0.77, ibs=0.20):
        directory = (
            self.results_dir / "kirc" / "SlotSPE" / f"experiment_sp_{task['tag']}"
        )
        directory.mkdir(parents=True)
        fields = [
            "folds", "val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc",
            "generalization_gap", "best_epoch", "best_to_final_val_drop",
        ]
        with (directory / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "folds": "mean", "val_cindex": cindex,
                "val_cindex_ipcw": ipcw, "val_IBS": ibs, "val_iauc": 0.76,
                "generalization_gap": gap, "best_epoch": 4,
                "best_to_final_val_drop": 0.05,
            })
            writer.writerow({"folds": "std", "val_cindex": std})

    def test_optimizer_grid_has_expected_nine_configs(self):
        tasks = plan_optimizer()
        self.assertEqual(len(tasks), 9)
        self.assertEqual({task["lr"] for task in tasks}, {"0.0004", "0.0005", "0.0006"})
        self.assertEqual(
            {task["weight_decay"] for task in tasks}, {"0.001", "0.01", "0.05"}
        )
        self.assertEqual({task["opt"] for task in tasks}, {"adamW"})

    def test_existing_manifest_is_reused_for_recovery(self):
        tasks = plan_reference()
        manifest = self.state_dir / "tasks" / "reference.tsv"
        write_tasks(manifest, tasks)
        recovered = plan_phase("reference", self.state_dir, self.results_dir)
        self.assertEqual([task["tag"] for task in recovered], [tasks[0]["tag"]])

    def test_dropout_inherits_best_optimizer_and_combine_uses_eligibility_rule(self):
        reference = plan_reference()
        write_tasks(self.state_dir / "tasks" / "reference.tsv", reference)
        self._write_summary(reference[0], 0.787, 0.22)

        optimizer = plan_optimizer()
        write_tasks(self.state_dir / "tasks" / "optimizer.tsv", optimizer)
        for index, task in enumerate(optimizer):
            self._write_summary(task, 0.78 + index * 0.001, 0.20)
        best_optimizer = optimizer[-1]

        dropout = plan_dropout(self.state_dir, self.results_dir)
        self.assertEqual(len(dropout), 7)
        self.assertTrue(all(task["lr"] == best_optimizer["lr"] for task in dropout))
        self.assertTrue(
            all(task["weight_decay"] == best_optimizer["weight_decay"] for task in dropout)
        )
        write_tasks(self.state_dir / "tasks" / "dropout.tsv", dropout)

        baseline_cindex = result_for_task(best_optimizer, self.results_dir)["mean_cindex"]
        for task in dropout:
            if task["variant"].startswith("wsi"):
                cindex, gap = baseline_cindex + 0.003, 0.20
            elif task["variant"].startswith("fusion"):
                cindex, gap = baseline_cindex - 0.001, 0.16
            else:
                cindex, gap = baseline_cindex - 0.01, 0.18
            self._write_summary(task, cindex, gap)

        combinations = plan_combine(self.state_dir, self.results_dir)
        self.assertEqual(len(combinations), 1)
        self.assertEqual(combinations[0]["variant"], "wsi_fusion")
        self.assertNotEqual(combinations[0]["wsi_projection_dropout"], "0.0")
        self.assertNotEqual(combinations[0]["fusion_dropout"], "0.0")
        self.assertEqual(combinations[0]["event_residual_dropout"], "0.0")

    def test_dropout_falls_back_to_reference_when_adamw_is_worse(self):
        reference = plan_reference()
        write_tasks(self.state_dir / "tasks" / "reference.tsv", reference)
        self._write_summary(reference[0], 0.79, 0.20, std=0.01)

        optimizer = plan_optimizer()
        write_tasks(self.state_dir / "tasks" / "optimizer.tsv", optimizer)
        for task in optimizer:
            self._write_summary(task, 0.78, 0.20, std=0.02)

        dropout = plan_dropout(self.state_dir, self.results_dir)
        self.assertTrue(all(task["source_tag"] == reference[0]["tag"] for task in dropout))
        self.assertEqual({task["opt"] for task in dropout}, {"adam"})
        self.assertEqual({task["weight_decay"] for task in dropout}, {"0"})

    def test_final_report_applies_metric_and_overfit_guardrails(self):
        reference = plan_reference()[0]
        candidate = plan_optimizer()[0]
        write_tasks(self.state_dir / "tasks" / "reference.tsv", [reference])
        write_tasks(self.state_dir / "tasks" / "optimizer.tsv", [candidate])
        self._write_summary(reference, 0.79, 0.20, ipcw=0.77, ibs=0.20)
        self._write_summary(candidate, 0.815, 0.16, ipcw=0.77, ibs=0.20)

        final_tasks = []
        for name, source in (("reference", reference), ("candidate1", candidate)):
            for seed in ("1", "5"):
                config = {key: source[key] for key in CONFIG_FIELDS}
                config["seed"] = seed
                task = make_task("final", name, config, source["tag"])
                final_tasks.append(task)
                if name == "reference":
                    self._write_summary(task, 0.79, 0.20, ipcw=0.77, ibs=0.20)
                else:
                    self._write_summary(task, 0.815, 0.16, ipcw=0.77, ibs=0.20)
        write_tasks(self.state_dir / "tasks" / "final.tsv", final_tasks)

        report = write_final_report(
            self.state_dir, self.results_dir, self.state_dir / "final_report.csv"
        )
        winner = next(row for row in report if not row["is_reference"])
        self.assertEqual(winner["three_seed_over_0p814"], 1)
        self.assertEqual(winner["passes_secondary_guardrails"], 1)
        self.assertEqual(winner["overfit_reduction_target"], 1)


if __name__ == "__main__":
    unittest.main()
