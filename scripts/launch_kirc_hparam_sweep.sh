#!/usr/bin/env bash
set -euo pipefail

cd /home/liufangyi/proj/SLotSPE/SlotSPE

tmux new-session -d -s slotspe_kirc_sweep_seed1 './scripts/run_kirc_variant.sh 0 sweep_seed1_base 1 0.0005 30 32 8 8 10 0.25 0.01'
tmux new-session -d -s slotspe_kirc_sweep_seed2 './scripts/run_kirc_variant.sh 1 sweep_seed2_base 2 0.0005 30 32 8 8 10 0.25 0.01'
tmux new-session -d -s slotspe_kirc_sweep_lr3e4 './scripts/run_kirc_variant.sh 2 sweep_lr3e4_seed3 3 0.0003 30 32 8 8 10 0.25 0.01'
tmux new-session -d -s slotspe_kirc_sweep_lr1e4 './scripts/run_kirc_variant.sh 3 sweep_lr1e4_seed3 3 0.0001 30 32 8 8 10 0.25 0.01'
tmux new-session -d -s slotspe_kirc_sweep_topk50 './scripts/run_kirc_variant.sh 4 sweep_topk50_seed3 3 0.0005 30 32 8 8 10 0.50 0.01'
tmux new-session -d -s slotspe_kirc_sweep_topk125 './scripts/run_kirc_variant.sh 5 sweep_topk125_seed3 3 0.0005 30 32 8 8 10 0.125 0.01'
tmux new-session -d -s slotspe_kirc_sweep_slots12 './scripts/run_kirc_variant.sh 6 sweep_slots12_seed3 3 0.0005 30 32 12 12 10 0.25 0.01'
tmux new-session -d -s slotspe_kirc_sweep_recon005 './scripts/run_kirc_variant.sh 7 sweep_recon005_seed3 3 0.0005 30 32 8 8 10 0.25 0.005'

tmux ls
