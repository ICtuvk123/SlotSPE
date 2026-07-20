#!/usr/bin/env python3
"""Plan, rank, and report the staged KIRC regularization search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import statistics
from pathlib import Path


CONFIG_FIELDS = [
    "seed", "opt", "lr", "weight_decay", "tau_event", "lambda_event",
    "topk_ratio", "wsi_projection_dropout", "fusion_dropout",
    "event_residual_dropout", "lambda_decoder_loss", "lambda_recon_loss",
    "event_projection_dim", "slot_iters",
]
TASK_FIELDS = ["task_id", "phase", "variant", "tag", "source_tag", *CONFIG_FIELDS]
SEED3_PHASES = ["reference", "optimizer", "dropout", "combine", "refine"]


def base_config():
    return {
        "seed": "3",
        "opt": "adam",
        "lr": "0.0005",
        "weight_decay": "0",
        "tau_event": "0.05",
        "lambda_event": "1.0",
        "topk_ratio": "0.50",
        "wsi_projection_dropout": "0.0",
        "fusion_dropout": "0.0",
        "event_residual_dropout": "0.0",
        "lambda_decoder_loss": "1.0",
        "lambda_recon_loss": "0.01",
        "event_projection_dim": "256",
        "slot_iters": "10",
    }


def config_fingerprint(config):
    payload = "|".join(f"{key}={config[key]}" for key in CONFIG_FIELDS)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def make_task(phase, variant, config, source_tag="-"):
    config = {key: str(config[key]) for key in CONFIG_FIELDS}
    tag = f"reg_{phase}_{variant}_{config_fingerprint(config)}_s{config['seed']}"
    return {
        "task_id": "",
        "phase": phase,
        "variant": variant,
        "tag": tag,
        "source_tag": source_tag,
        **config,
    }


def read_tasks(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tasks(path, tasks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=TASK_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for task_id, task in enumerate(tasks):
            task = dict(task)
            task["task_id"] = str(task_id)
            writer.writerow(task)


def summary_path_for_tag(results_dir, tag):
    root = Path(results_dir) / "kirc" / "SlotSPE"
    matches = sorted(root.glob(f"*_sp_{tag}/summary.csv"))
    return matches[0] if matches else None


def _summary_row(path, label):
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("folds")) == label:
                return row
    return None


def result_for_task(task, results_dir):
    summary = summary_path_for_tag(results_dir, task["tag"])
    if summary is None:
        return None
    mean = _summary_row(summary, "mean")
    std = _summary_row(summary, "std") or {}
    if mean is None:
        return None

    def value(row, key, default=float("nan")):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return default

    return {
        **task,
        "mean_cindex": value(mean, "val_cindex"),
        "std_cindex": value(std, "val_cindex"),
        "mean_ipcw": value(mean, "val_cindex_ipcw"),
        "mean_ibs": value(mean, "val_IBS"),
        "mean_iauc": value(mean, "val_iauc"),
        "mean_gap": value(mean, "generalization_gap"),
        "mean_best_epoch": value(mean, "best_epoch"),
        "mean_best_to_final_drop": value(mean, "best_to_final_val_drop"),
        "summary": str(summary),
    }


def completed_results(tasks, results_dir, require_all=True):
    results = []
    missing = []
    for task in tasks:
        result = result_for_task(task, results_dir)
        if result is None:
            missing.append(task["tag"])
        else:
            results.append(result)
    if require_all and missing:
        preview = ", ".join(missing[:3])
        raise SystemExit(
            f"Previous phase is incomplete: {len(missing)} result(s) missing, including {preview}"
        )
    return results


def ranking_key(result):
    std = result["std_cindex"] if math.isfinite(result["std_cindex"]) else float("inf")
    gap = result["mean_gap"] if math.isfinite(result["mean_gap"]) else float("inf")
    return (-result["mean_cindex"], std, gap)


def best_result(results):
    if not results:
        raise SystemExit("No completed result is available for selection")
    maximum_cindex = max(result["mean_cindex"] for result in results)
    contenders = [
        result for result in results
        if maximum_cindex - result["mean_cindex"] < 0.002
    ]

    def tie_breaker(result):
        std = result["std_cindex"] if math.isfinite(result["std_cindex"]) else float("inf")
        gap = result["mean_gap"] if math.isfinite(result["mean_gap"]) else float("inf")
        return (std, gap, -result["mean_cindex"])

    return min(contenders, key=tie_breaker)


def config_from_row(row):
    return {key: str(row[key]) for key in CONFIG_FIELDS}


def phase_tasks_path(state_dir, phase):
    return Path(state_dir) / "tasks" / f"{phase}.tsv"


def require_phase(state_dir, results_dir, phase):
    path = phase_tasks_path(state_dir, phase)
    if not path.exists():
        raise SystemExit(f"Missing {phase} task manifest: {path}")
    return completed_results(read_tasks(path), results_dir, require_all=True)


def best_optimizer_or_reference(state_dir, results_dir):
    reference = require_phase(state_dir, results_dir, "reference")
    optimizer = require_phase(state_dir, results_dir, "optimizer")
    return best_result([*reference, *optimizer])


def plan_reference():
    return [make_task("reference", "current_best", base_config())]


def plan_optimizer():
    tasks = []
    for lr, weight_decay in itertools.product(
        ("0.0004", "0.0005", "0.0006"), ("0.001", "0.01", "0.05")
    ):
        config = base_config()
        config.update(opt="adamW", lr=lr, weight_decay=weight_decay)
        variant = f"lr{lr.replace('.', 'p')}_wd{weight_decay.replace('.', 'p')}"
        tasks.append(make_task("optimizer", variant, config))
    return tasks


def plan_dropout(state_dir, results_dir):
    selected_base = best_optimizer_or_reference(state_dir, results_dir)
    base = config_from_row(selected_base)
    variants = [
        ("wsi_projection_dropout", "wsi", "0.1"),
        ("wsi_projection_dropout", "wsi", "0.2"),
        ("fusion_dropout", "fusion", "0.1"),
        ("fusion_dropout", "fusion", "0.2"),
        ("event_residual_dropout", "event", "0.1"),
        ("event_residual_dropout", "event", "0.2"),
        ("event_residual_dropout", "event", "0.3"),
    ]
    tasks = []
    for field, name, value in variants:
        config = dict(base)
        config[field] = value
        tasks.append(
            make_task(
                "dropout", f"{name}{value.replace('.', 'p')}", config,
                selected_base["tag"],
            )
        )
    return tasks


def plan_combine(state_dir, results_dir):
    selected_base = best_optimizer_or_reference(state_dir, results_dir)
    dropout_results = require_phase(state_dir, results_dir, "dropout")
    axis_by_variant = {
        "wsi": "wsi_projection_dropout",
        "fusion": "fusion_dropout",
        "event": "event_residual_dropout",
    }
    eligible = {}
    for axis, field in axis_by_variant.items():
        candidates = [row for row in dropout_results if row["variant"].startswith(axis)]
        candidate = best_result(candidates)
        cindex_gain = candidate["mean_cindex"] - selected_base["mean_cindex"]
        gap_reduction = selected_base["mean_gap"] - candidate["mean_gap"]
        if cindex_gain >= 0.002 or (cindex_gain >= -0.002 and gap_reduction >= 0.03):
            eligible[axis] = (field, candidate[field])

    tasks = []
    axes = list(axis_by_variant)
    combinations = []
    for size in (2, 3):
        combinations.extend(itertools.combinations([axis for axis in axes if axis in eligible], size))
    for combination in combinations[:4]:
        config = config_from_row(selected_base)
        for axis in combination:
            field, value = eligible[axis]
            config[field] = value
        tasks.append(
            make_task("combine", "_".join(combination), config, selected_base["tag"])
        )
    return tasks


def completed_seed3_search(state_dir, results_dir):
    tasks = []
    for phase in SEED3_PHASES:
        path = phase_tasks_path(state_dir, phase)
        if path.exists():
            tasks.extend(read_tasks(path))
    return completed_results(tasks, results_dir, require_all=True)


def plan_refine(state_dir, results_dir):
    completed = completed_seed3_search(state_dir, results_dir)
    current_best = best_result(completed)
    remaining_budget = max(0, 24 - len(completed))
    mutations = [
        ("decoder0p5", "lambda_decoder_loss", "0.5"),
        ("eventdim128", "event_projection_dim", "128"),
        ("slotiters5", "slot_iters", "5"),
        ("lambdaevent0p5", "lambda_event", "0.5"),
        ("recon0p05", "lambda_recon_loss", "0.05"),
    ]
    existing_fingerprints = {config_fingerprint(config_from_row(row)) for row in completed}
    tasks = []
    for variant, field, value in mutations:
        if len(tasks) >= remaining_budget:
            break
        config = config_from_row(current_best)
        config[field] = value
        fingerprint = config_fingerprint(config)
        if fingerprint in existing_fingerprints:
            continue
        existing_fingerprints.add(fingerprint)
        tasks.append(make_task("refine", variant, config, current_best["tag"]))
    return tasks


def plan_final(state_dir, results_dir):
    completed = completed_seed3_search(state_dir, results_dir)
    reference = best_result([row for row in completed if row["phase"] == "reference"])
    candidate_pool = [row for row in completed if row["phase"] != "reference"]
    first_candidate = best_result(candidate_pool)
    second_candidate = best_result(
        [row for row in candidate_pool if row["tag"] != first_candidate["tag"]]
    )
    candidates = [first_candidate, second_candidate]
    tasks = []
    sources = [("reference", reference), ("candidate1", candidates[0]), ("candidate2", candidates[1])]
    for name, source in sources:
        for seed in ("1", "5"):
            config = config_from_row(source)
            config["seed"] = seed
            tasks.append(make_task("final", name, config, source["tag"]))
    return tasks


def plan_phase(phase, state_dir, results_dir):
    existing_manifest = phase_tasks_path(state_dir, phase)
    if existing_manifest.exists():
        # A manifest is the recoverability boundary: relaunch exactly the same tasks.
        return read_tasks(existing_manifest)
    if phase == "reference":
        return plan_reference()
    if phase == "optimizer":
        require_phase(state_dir, results_dir, "reference")
        return plan_optimizer()
    if phase == "dropout":
        return plan_dropout(state_dir, results_dir)
    if phase == "combine":
        return plan_combine(state_dir, results_dir)
    if phase == "refine":
        return plan_refine(state_dir, results_dir)
    if phase == "final":
        require_phase(state_dir, results_dir, "refine")
        return plan_final(state_dir, results_dir)
    raise ValueError(phase)


RANKING_FIELDS = [
    *TASK_FIELDS, "mean_cindex", "std_cindex", "mean_ipcw", "mean_ibs",
    "mean_iauc", "mean_gap", "mean_best_epoch", "mean_best_to_final_drop", "summary",
]


def write_ranking(tasks_path, results_dir, output):
    results = completed_results(read_tasks(tasks_path), results_dir, require_all=False)
    results.sort(key=ranking_key)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    return results


def config_signature(row):
    return tuple((key, row[key]) for key in CONFIG_FIELDS if key != "seed")


def write_final_report(state_dir, results_dir, output):
    all_tasks = []
    for path in sorted((Path(state_dir) / "tasks").glob("*.tsv")):
        all_tasks.extend(read_tasks(path))
    results = completed_results(all_tasks, results_dir, require_all=False)
    groups = {}
    for result in results:
        groups.setdefault(config_signature(result), []).append(result)

    report_rows = []
    for grouped in groups.values():
        seeds = {int(row["seed"]) for row in grouped}
        if not {1, 3, 5}.issubset(seeds):
            continue
        by_seed = {int(row["seed"]): row for row in grouped}
        selected = [by_seed[seed] for seed in (1, 3, 5)]
        cindices = [row["mean_cindex"] for row in selected]
        report_rows.append({
            "source_seed3_tag": by_seed[3]["tag"],
            "is_reference": int(by_seed[3]["phase"] == "reference"),
            "seed3_cindex": by_seed[3]["mean_cindex"],
            "mean_cindex_3seed": statistics.mean(cindices),
            "std_cindex_across_seeds": statistics.stdev(cindices),
            "worst_seed_cindex": min(cindices),
            "mean_ipcw_3seed": statistics.mean(row["mean_ipcw"] for row in selected),
            "mean_ibs_3seed": statistics.mean(row["mean_ibs"] for row in selected),
            "mean_iauc_3seed": statistics.mean(row["mean_iauc"] for row in selected),
            "mean_gap_3seed": statistics.mean(row["mean_gap"] for row in selected),
            "mean_best_to_final_drop_3seed": statistics.mean(
                row["mean_best_to_final_drop"] for row in selected
            ),
            **{key: by_seed[3][key] for key in CONFIG_FIELDS if key != "seed"},
        })
    reference_rows = [row for row in report_rows if row["is_reference"]]
    if reference_rows:
        reference = reference_rows[0]
        for row in report_rows:
            row["delta_cindex_vs_reference"] = (
                row["mean_cindex_3seed"] - reference["mean_cindex_3seed"]
            )
            row["delta_ipcw_vs_reference"] = (
                row["mean_ipcw_3seed"] - reference["mean_ipcw_3seed"]
            )
            row["delta_ibs_vs_reference"] = (
                row["mean_ibs_3seed"] - reference["mean_ibs_3seed"]
            )
            row["gap_reduction_vs_reference"] = (
                reference["mean_gap_3seed"] - row["mean_gap_3seed"]
            )
            row["drop_reduction_vs_reference"] = (
                reference["mean_best_to_final_drop_3seed"]
                - row["mean_best_to_final_drop_3seed"]
            )
            row["passes_secondary_guardrails"] = int(
                row["delta_ipcw_vs_reference"] >= -0.01
                and row["delta_ibs_vs_reference"] <= 0.01
            )
            row["seed3_over_0p814"] = int(row["seed3_cindex"] > 0.814)
            row["three_seed_over_0p814"] = int(row["mean_cindex_3seed"] > 0.814)
            row["stable_improvement"] = int(
                row["delta_cindex_vs_reference"] > 0.0
                and row["passes_secondary_guardrails"]
            )
            row["overfit_reduction_target"] = int(
                row["gap_reduction_vs_reference"] >= 0.03
                or row["drop_reduction_vs_reference"] >= 0.02
            )
    report_rows.sort(key=lambda row: -row["mean_cindex_3seed"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(report_rows[0]) if report_rows else ["source_seed3_tag", "mean_cindex_3seed"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)
    return report_rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="create one phase task manifest")
    plan.add_argument("--phase", required=True, choices=[*SEED3_PHASES, "final"])
    plan.add_argument("--state-dir", required=True)
    plan.add_argument("--results-dir", required=True)
    plan.add_argument("--output", required=True)

    rank = subparsers.add_parser("rank", help="rank completed tasks from one manifest")
    rank.add_argument("--tasks", required=True)
    rank.add_argument("--results-dir", required=True)
    rank.add_argument("--output", required=True)

    report = subparsers.add_parser("report", help="aggregate completed 3-seed finalists")
    report.add_argument("--state-dir", required=True)
    report.add_argument("--results-dir", required=True)
    report.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "plan":
        tasks = plan_phase(args.phase, args.state_dir, args.results_dir)
        write_tasks(args.output, tasks)
        print(f"Planned {len(tasks)} {args.phase} task(s): {args.output}")
    elif args.command == "rank":
        results = write_ranking(args.tasks, args.results_dir, args.output)
        print(f"Ranked {len(results)} completed task(s): {args.output}")
    else:
        rows = write_final_report(args.state_dir, args.results_dir, args.output)
        print(f"Wrote {len(rows)} three-seed result(s): {args.output}")


if __name__ == "__main__":
    main()
