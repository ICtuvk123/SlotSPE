#!/usr/bin/env python3
"""One-patient remote smoke test for raw WSI -> CONCH LoRA -> SlotSPE."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_survival import SurvivalDatasetFactory, _collate_pathways
from dataset.online_wsi_dataset import OnlineWSISurvivalDataset
from utils.core_utils import _init_loss_function, _process_data_and_forward
from utils.general_utils import _prepare_for_experiment
from utils.model_utils import _init_model
from utils.process_args import _process_args


def main():
    args = _process_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    args = _prepare_for_experiment(args)
    if not args.online_conch_model_dir:
        raise ValueError("--online_conch_model_dir is required")
    if args.batch_size != 1:
        raise ValueError("smoke test requires --batch_size 1")

    factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_classes,
        label_col=args.label_col,
        num_genes=args.num_genes,
        num_patches=args.num_patches,
    )
    dataset = OnlineWSISurvivalDataset(
        factory,
        raw_wsi_dir=args.raw_wsi_dir,
        patch_coords_dir=args.patch_coords_dir,
        split_key="train",
        fold=args.k_start,
        target_patch_size=args.online_target_patch_size,
        sample_seed=args.seed,
    )
    batch = _collate_pathways([dataset[0]])
    model = _init_model(args, factory)
    model.train()
    args.cur_epoch = 0

    output, y_disc, event_time, censorship = _process_data_and_forward(
        args, model, batch, args.device
    )
    logits, auxiliary_loss = output[:2]
    loss_fn = _init_loss_function(args)
    survival_loss = loss_fn(logits, y_disc, event_time, censorship)
    loss = survival_loss + auxiliary_loss
    loss.backward()

    if args.conch_qv_lora_mode == "none":
        trainable_grads = {
            name: float(parameter.grad.detach().norm().cpu())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and not name.startswith("conch.")
        }
        nonzero = {
            name: value for name, value in trainable_grads.items() if value > 0.0
        }
        if not nonzero:
            raise RuntimeError("online frozen CONCH baseline did not backpropagate")
    else:
        adapter_grads = {
            name: float(parameter.grad.detach().norm().cpu())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and any(part in name for part in (".q_up.", ".v_up."))
        }
        nonzero = {
            name: value for name, value in adapter_grads.items() if value > 0.0
        }
        if not nonzero:
            raise RuntimeError("CONCH Q/V LoRA did not receive a non-zero gradient")

    print("[ok] online CONCH GC-RSM smoke test passed")
    print(f"patches={tuple(batch[0].shape)}")
    print(f"logits={tuple(logits.shape)}")
    print(f"loss={float(loss.detach().cpu()):.6f}")
    print(f"nonzero_checked_gradients={len(nonzero)}")
    print(f"max_checked_gradient={max(nonzero.values()):.6f}")
    print(f"compact_checkpoint_tensors={len(model.state_dict())}")


if __name__ == "__main__":
    main()
