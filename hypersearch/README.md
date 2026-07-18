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
