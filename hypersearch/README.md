# Event-gated hyperparameter search

Run the three-GPU sweep from the repository root:

```bash
bash scripts/launch_kirc_event_gated_sweep.sh
tmux attach -t slotspe_event_sweep
```

The default search has 30 groups: 5 learning rates x 2 event temperatures x
3 event-residual weights. Each group runs all five folds for 30 epochs. Based
on the existing KIRC runs (about 3.46-4.72 hours per group), each GPU receives
10 groups, or roughly 35-47 hours at historical throughput. Event-gating
overhead and GPU contention can change this estimate.

Directory layout after launch:

- `runs/<run-id>/tasks.tsv`: exact hyperparameters for every group
- `runs/<run-id>/logs/`: stdout/stderr for every group
- `runs/<run-id>/status/`: worker completion and failure records
- `runs/<run-id>/live_ranking.csv`: ranking refreshed after each completed group
- `runs/<run-id>/ranking.csv`: final ranking by mean validation c-index
- `results/kirc/SlotSPE/`: checkpoints, settings, and fold-level summaries

Use `DRY_RUN=1` to inspect the generated task table without starting tmux.
The search space and paths can be overridden with the environment variables
shown by `bash scripts/launch_kirc_event_gated_sweep.sh --help`.

To stop all three workers after seeing a satisfactory result, run
`tmux kill-session -t slotspe_event_sweep`. Completed groups remain on disk;
the currently interrupted groups will not have a final `summary.csv`.

Workers wait until their GPU has at least 18000 MiB free before starting a
group. Override this with `WAIT_FOR_GPU_FREE_MIB`; setting it to `0` disables
the guard.

## Regularization search

The anti-overfitting search is staged so later tasks inherit the best completed
configuration instead of requiring parameters to be copied by hand:

```bash
bash scripts/launch_kirc_regularization_search.sh reference
tmux attach -t slotspe_reg_reference

# After the monitor finishes, repeat with optimizer, dropout, combine, refine,
# and final, in that order.
```

Run the next phase only after the current monitor reports completion. A phase
with no eligible adaptive tasks exits successfully without creating tmux. Use
`DRY_RUN=1` with any phase to inspect its task manifest without starting tmux.
The search uses a fixed validation Slot seed, full five-fold runs, and at most
24 seed-3 configurations before evaluating the reference and two finalists at
seeds 1 and 5.

State is stored under `hypersearch/regularization_search/` by default:

- `tasks/<phase>.tsv`: exact, recoverable phase manifests
- `rankings/<phase>.csv`: C-index ranking with IPCW, IBS, generalization gap,
  best epoch, and best-to-final drop
- `final_3seed_report.csv`: seeds 1/3/5 aggregate for the reference and finalists

All new model controls default to legacy behavior. The regularization launcher
explicitly sets `eval_slot_seed=100000` and the selected AdamW/dropout/loss
settings for each task.

To let a controller advance all phases automatically, run it in a separate
tmux session after starting (or before starting) the reference phase:

```bash
tmux new-session -d -s slotspe_reg_driver \
  "GPU_IDS='5 6 7' bash scripts/drive_kirc_regularization_search.sh"
```

The controller never advances from an incomplete phase and stops on worker
failure, leaving phase logs and manifests intact for diagnosis and recovery.
