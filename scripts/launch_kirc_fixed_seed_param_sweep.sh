#!/usr/bin/env bash
set -euo pipefail

cd /home/liufangyi/proj/SLotSPE/SlotSPE

# Fixed-seed sweep around the best completed setting:
#   seed=3, lr=5e-4, epochs=30, batch=32, slots=8/8, iters=10, topk=0.50, recon=0.01
#
# With 8 slots, topk_ratio is converted to int(8 * ratio), so these ratios test
# distinct keep-slot counts: 0.375 -> 3, 0.50 -> 4, 0.625 -> 5, 0.75 -> 6.

tmux new-session -d -s slotspe_kirc_fixed_lr4e4_topk50 './scripts/run_kirc_variant.sh 0 sweep_fixed_lr4e4_topk50_seed3 3 0.0004 30 32 8 8 10 0.50 0.01'
tmux new-session -d -s slotspe_kirc_fixed_lr6e4_topk50 './scripts/run_kirc_variant.sh 1 sweep_fixed_lr6e4_topk50_seed3 3 0.0006 30 32 8 8 10 0.50 0.01'
tmux new-session -d -s slotspe_kirc_fixed_lr7e4_topk50 './scripts/run_kirc_variant.sh 2 sweep_fixed_lr7e4_topk50_seed3 3 0.0007 30 32 8 8 10 0.50 0.01'
tmux new-session -d -s slotspe_kirc_fixed_lr1e3_topk50 './scripts/run_kirc_variant.sh 3 sweep_fixed_lr1e3_topk50_seed3 3 0.0010 30 32 8 8 10 0.50 0.01'

tmux new-session -d -s slotspe_kirc_fixed_topk375 './scripts/run_kirc_variant.sh 4 sweep_fixed_topk375_seed3 3 0.0005 30 32 8 8 10 0.375 0.01'
tmux new-session -d -s slotspe_kirc_fixed_topk625 './scripts/run_kirc_variant.sh 5 sweep_fixed_topk625_seed3 3 0.0005 30 32 8 8 10 0.625 0.01'
tmux new-session -d -s slotspe_kirc_fixed_topk75 './scripts/run_kirc_variant.sh 6 sweep_fixed_topk75_seed3 3 0.0005 30 32 8 8 10 0.75 0.01'

tmux new-session -d -s slotspe_kirc_fixed_recon02 './scripts/run_kirc_variant.sh 7 sweep_fixed_recon02_topk50_seed3 3 0.0005 30 32 8 8 10 0.50 0.02'

tmux ls
