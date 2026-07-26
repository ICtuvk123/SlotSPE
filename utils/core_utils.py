#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: core_utils_rebuttal.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/21/24 3:56 PM
'''

from ast import Lambda
import csv
from contextlib import contextmanager
import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, brier_score, integrated_brier_score
from sksurv.util import Surv
from utils.general_utils import _save_pkl
from utils.loss_func import NLLSurvLoss, SurvPLE, RankLoss, SinkhornSurvLoss
import torch.optim as optim
import torch
from dataset.dataset_survival import SurvivalDataset, _collate_pathways
import os
from utils.model_utils import _init_model
import torch.nn.functional as F
import gc


def _resolved_weight_decay(args):
    """Resolve the explicit weight-decay option without changing legacy Adam runs."""
    explicit = getattr(args, "weight_decay", None)
    value = args.reg if explicit is None else explicit
    value = float(value)
    if value < 0.0:
        raise ValueError(f"weight_decay must be non-negative; got {value}")
    return value


def _build_weight_decay_param_groups(model, weight_decay):
    """Apply AdamW decay to matrix weights, but not offsets, norms, or Slot priors."""
    decay = []
    no_decay = []
    no_decay_names = []
    decay_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = name.casefold()
        exclude = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in normalized_name
            or "slots_mu" in normalized_name
            or "slots_logsigma" in normalized_name
        )
        if exclude:
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"})
    return groups, {"decay": decay_names, "no_decay": no_decay_names}


def _fold_eval_slot_seed(args, fold):
    base_seed = getattr(args, "eval_slot_seed", None)
    return None if base_seed is None else int(base_seed) + int(fold)


@contextmanager
def _fixed_evaluation_rng(seed, device):
    """Use common random numbers for validation without consuming the training RNG."""
    if seed is None:
        yield
        return

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        yield

def free_loader(loader):
    if loader is None:
        return
    # stop worker processes if an iterator exists (private API but effective)
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
    except Exception:
        pass
    # drop references and run GC
    del loader
    gc.collect()

def _get_split(args, dataset_factory, cur):
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    if (
        args.slot_attention_type == "event_gated"
        and args.lambda_event != 0.0
        and not args.conch_patch_feature_dir
        and not args.reuse_slot_features_as_conch
    ):
        raise ValueError(
            "Event-Gated SlotSPE requires --conch_patch_feature_dir, or explicitly verified "
            "--reuse_slot_features_as_conch. Cross-encoder cosine is forbidden."
        )
    event_dataset_args = {
        "conch_patch_feature_dir": args.conch_patch_feature_dir
        if args.slot_attention_type == "event_gated" else None,
        "reuse_slot_features_as_conch": args.reuse_slot_features_as_conch
        if args.slot_attention_type == "event_gated" else False,
        "slot_feature_encoder": args.slot_feature_encoder,
        "require_conch_alignment": args.require_conch_alignment,
    }
    if getattr(args, "online_conch_model_dir", None):
        from dataset.online_wsi_dataset import OnlineWSISurvivalDataset
        common = {
            "raw_wsi_dir": args.raw_wsi_dir,
            "patch_coords_dir": args.patch_coords_dir,
            "fold": cur,
            "target_patch_size": args.online_target_patch_size,
            "sample_seed": args.seed,
        }
        train_data = OnlineWSISurvivalDataset(
            dataset_factory, split_key="train", **common
        )
        test_data = OnlineWSISurvivalDataset(
            dataset_factory, split_key="val", **common
        )
    else:
        train_data = SurvivalDataset(dataset_factory, args.data_root_dir, 'train', cur, args.encoding_dim,
                                     **event_dataset_args)
        test_data = SurvivalDataset(dataset_factory, args.data_root_dir, 'val', cur, args.encoding_dim,
                                    **event_dataset_args)
    if args.rna_format == "Pathways" or args.rna_format == "RankedGenes":
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=_collate_pathways, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate_pathways, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,pin_memory=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0,pin_memory=False)
    print('Done!')
    print("Training on {} samples".format(len(train_data)))
    print("Validating on {} samples".format(len(test_data)))

    return train_data, test_data, train_loader, test_loader


def _init_loss_function(args):
    r"""
    Init the survival loss function

    Args:
        - args : argspace.Namespace

    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss

    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = SurvPLE() 
    elif args.bag_loss == 'rank_surv':
        loss_fn = RankLoss()
    elif args.bag_loss == 'sinkhorn_surv':
        loss_fn = SinkhornSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn


def _init_optim(args, model):
    r"""
    Init the optimizer

    Args:
        - args : argspace.Namespace
        - model : torch model

    Returns:
        - optimizer : torch optim
    """
    print('\nInit optimizer ...', end='\n')

    optimizer_name = args.opt.casefold()
    explicit_weight_decay = getattr(args, "weight_decay", None)

    conch_lora_lr = getattr(args, "conch_lora_lr", None)
    parameter_source = model.parameters()
    if conch_lora_lr is not None:
        if not getattr(args, "online_conch_model_dir", None):
            raise ValueError("--conch_lora_lr requires online CONCH training")
        if conch_lora_lr <= 0.0:
            raise ValueError("conch_lora_lr must be positive")
        conch_parameters = []
        downstream_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("conch."):
                conch_parameters.append(parameter)
            else:
                downstream_parameters.append(parameter)
        parameter_source = [
            {"params": downstream_parameters, "lr": args.lr, "group_name": "downstream"},
            {"params": conch_parameters, "lr": conch_lora_lr, "group_name": "conch_lora"},
        ]
        print(
            f"Optimizer learning rates: downstream={args.lr:g}, "
            f"conch_lora={conch_lora_lr:g}"
        )

    if optimizer_name == "adam":
        # Preserve legacy behavior unless the new explicit option is supplied.
        weight_decay = 0.0 if explicit_weight_decay is None else _resolved_weight_decay(args)
        optimizer = optim.Adam(parameter_source, lr=args.lr, weight_decay=weight_decay)

    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9,
            weight_decay=_resolved_weight_decay(args)
        )
    elif optimizer_name == "adamw":
        weight_decay = _resolved_weight_decay(args)
        parameter_groups, parameter_group_names = _build_weight_decay_param_groups(
            model, weight_decay
        )
        optimizer = optim.AdamW(parameter_groups, lr=args.lr, weight_decay=0.0)
        optimizer.parameter_group_names = parameter_group_names
        print(
            f"AdamW weight decay: {weight_decay:g} "
            f"({len(parameter_group_names['decay'])} decayed tensors, "
            f"{len(parameter_group_names['no_decay'])} excluded tensors)"
        )
    elif optimizer_name == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=_resolved_weight_decay(args))
    else:
        raise NotImplementedError

    return optimizer

def _init_scheduler(args, optimizer):
    r"""
    Init the scheduler

    Args:
        - args : argspace.Namespace
        - optimizer : torch optim

    Returns:
        - scheduler : torch.optim.lr_scheduler
    """
    print('\nInit scheduler ...', end='\n')

    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)
    else:
        raise NotImplementedError

    return scheduler

def _extract_survival_metadata(dataset_factory):

    all_censorships = dataset_factory.clinical_df[[dataset_factory.censorship_var]]
    #dataframe to numpy array
    all_censorships = all_censorships.to_numpy().flatten()

    all_event_times = dataset_factory.clinical_df[[dataset_factory.label_col]]
    #dataframe to numpy array
    all_event_times = all_event_times.to_numpy().flatten()

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)

    return all_survival
    
def _unpack_data(data, device, omics_format):
    # [img, omic_data_list, label, event_time, c]
    data_wsi = data[0].to(device)
    
    if omics_format == "Pathways" or omics_format == "RankedGenes":
        data_omics = [] # TODO: check
        for idx,item in enumerate(data[1]):
            for idy,omic in enumerate(item):
                omic = omic.to(device)
                omic = omic.unsqueeze(0)
                if idx == 0:
                    data_omics.append(omic)
                else:
                    data_omics[idy] = torch.cat((data_omics[idy],omic),dim=0)
    else:
        data_omics = data[1].to(device)
    
    y_disc = data[2].to(device)
    event_time = data[3].to(device)
    c = data[4].to(device)

    z_conch = data[5].to(device) if len(data) > 5 else None
    patch_mask = data[6].to(device) if len(data) > 6 else None

    return data_wsi, data_omics, y_disc, event_time, c, z_conch, patch_mask

def _process_data_and_forward(args, model, data, device, test=False):
    data_wsi, data_omics, y_disc, event_time, c, z_conch, patch_mask = _unpack_data(
        data, device, args.rna_format
    )
    
    input_args = {"x_wsi": data_wsi}
    if z_conch is not None:
        input_args["z_conch"] = z_conch
        input_args["patch_mask"] = patch_mask

    input_args["cur_epoch"] = args.cur_epoch
    input_args['omic_missing'] = False
    if test:
        input_args['y'] = None
        input_args['c'] = None

        input_args['omic_missing'] = args.omic_missing
    else:
        input_args['y'] = y_disc
        input_args['c'] = c


    if args.rna_format == "Pathways" or args.rna_format == "RankedGenes":
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i]

        out = model(**input_args)
    else:
        input_args['x_omics'] = data_omics
        out = model(**input_args)

    return out, y_disc, event_time, c


def _calculate_risk(h):
    hazards = torch.sigmoid(h) # h: the output of the model
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()


def _unpack_slotspe_output(output):
    """Accept the baseline pair and the optional event-debug triple."""
    if not isinstance(output, (tuple, list)) or len(output) not in (2, 3):
        raise ValueError("SlotSPE must return (logits, aux_loss[, event_details])")
    logits, aux_loss = output[:2]
    details = output[2] if len(output) == 3 else None
    return logits, aux_loss, details


def _update_arrays(all_risk_scores, all_censorships, all_event_times, event_time, censor, risk, clinical_data_list):

    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    return all_risk_scores, all_censorships, all_event_times

def _train_loop_survival(args, epoch, model,loader, optimizer, scheduler, loss_fn, log_file):
    if args.opt == "adam_seperate":
        optimizer_club = optimizer[1]
        optimizer = optimizer[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.0
    total_survival_loss = 0.0
    total_aux_loss = 0.0
    component_totals = {
        "decoder_loss": 0.0,
        "reconstruction_loss": 0.0,
        "weighted_decoder_loss": 0.0,
        "weighted_reconstruction_loss": 0.0,
        "vl_alignment_loss": 0.0,
        "weighted_vl_alignment_loss": 0.0,
    }
    num_batches = 0
    learning_rate = optimizer.param_groups[0]["lr"]
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    args.cur_epoch = epoch


    accumulation_steps = getattr(args, "gradient_accumulation_steps", None)
    if accumulation_steps is None:
        accumulation_steps = 32 if args.batch_size == 1 else 1
    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    gradient_clip_norm = float(getattr(args, "gradient_clip_norm", 0.0))
    optimizer.zero_grad(set_to_none=True)
    # one epoch
    for batch_idx, data in enumerate(loader):

        h, y_disc, event_time, c = _process_data_and_forward(args, model, data, device)

        if args.method.startswith("SlotSPE"):
            logits, slot_loss, _ = _unpack_slotspe_output(h)
        else:
            raise ValueError(f"Method {args.method} not implemented")

        if args.bag_loss == "cox_surv":
            loss_surv = loss_fn(logits, event_time, c) #y_hat, T, E
        else:
            loss_surv = loss_fn(logits, y_disc, event_time, c)
        loss_surv = loss_surv / y_disc.shape[0]

        if args.method.startswith("SlotSPE"):
            loss = loss_surv + slot_loss
        else:
            raise ValueError(f"Method {args.method} not implemented")

        (loss / accumulation_steps).backward()

        should_step = (
            (batch_idx + 1) % accumulation_steps == 0
            or (batch_idx + 1) == len(loader)
        )
        if should_step:
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        total_loss += loss.item()
        total_survival_loss += loss_surv.item()
        total_aux_loss += float(slot_loss.detach().item())
        latest_components = getattr(model, "last_loss_components", {})
        for key in component_totals:
            value = latest_components.get(key)
            if value is not None:
                component_totals[key] += float(value.detach().item())
        num_batches += 1
        risk, _ = _calculate_risk(logits)
        all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores, all_censorships,
                                                                           all_event_times, event_time, c, risk, data)

        if batch_idx % accumulation_steps == 0:
            print('batch:{}, loss:{:.4f}, loss_surv: {:.4f}'.format(batch_idx, loss.item(), loss_surv.item()))
            log_file.write('batch:{}, loss:{:.4f}, loss_surv: {:.4f}\n'.format(batch_idx, loss.item(), loss_surv.item()))

    scheduler.step()

    total_loss /= max(num_batches, 1)
    total_survival_loss /= max(num_batches, 1)
    total_aux_loss /= max(num_batches, 1)
    component_totals = {
        key: value / max(num_batches, 1) for key, value in component_totals.items()
    }
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores,tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))
    log_file.write('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}\n'.format(epoch, total_loss, c_index))

    return {
        "learning_rate": learning_rate,
        "train_loss": total_loss,
        "train_survival_loss": total_survival_loss,
        "train_aux_loss": total_aux_loss,
        "train_decoder_loss": component_totals["decoder_loss"],
        "train_reconstruction_loss": component_totals["reconstruction_loss"],
        "train_weighted_decoder_loss": component_totals["weighted_decoder_loss"],
        "train_weighted_reconstruction_loss": component_totals["weighted_reconstruction_loss"],
        "train_vl_alignment_loss": component_totals["vl_alignment_loss"],
        "train_weighted_vl_alignment_loss": component_totals["weighted_vl_alignment_loss"],
        "train_cindex": c_index,
    }


def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times,
                       all_risk_by_bin_scores):

    data = loader.dataset.label_df[dataset_factory.label_col]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    # ---> delete the nans and corresponding elements from other arrays
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    # <---

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times,
                                         all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics
    try:
        survival_test = Surv.from_arrays(event=(1 - all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc

    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.

    # brier score
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores,
                            times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.

    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores,
                                     times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1 - all_risk_by_bin_scores[:, 1:],
                                         times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.

    return c_index, c_index_ipcw, BS, IBS, iauc


def _summary(
    args, dataset_factory, model, loader, loss_fn, survival_train=None,
    eval_slot_seed=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.
    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_logits = []
    all_case_ids = []

    case_ids = loader.dataset.label_df["case id"]
    count = 0
    with _fixed_evaluation_rng(eval_slot_seed, device), torch.no_grad():
        for batch_idx, data in enumerate(loader):
            h, y_disc, event_time, c = _process_data_and_forward(args, model, data, device, test=True)

            if args.method.startswith("SlotSPE"):
                logits, _, _ = _unpack_slotspe_output(h)
            else:
                raise ValueError(f"Method {args.method} not implemented")

            if args.bag_loss == "cox_surv":
                loss = loss_fn(logits, event_time, c)  # y_hat, T, E
            elif args.bag_loss == "nll_surv":
                loss = loss_fn(logits, y_disc, event_time, c)
            else:
                raise ValueError(f"Loss function {args.bag_loss} not implemented")

            total_loss += loss.item()
            risk, risk_by_bin = _calculate_risk(logits)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores, all_censorships,
                                                                               all_event_times, event_time, c, risk, data)
            all_logits.append(logits.detach().cpu().numpy())
            all_case_ids.append(case_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)


    patient_results = {}
    for i in range(len(all_case_ids)):
        case_id = all_case_ids[i]
        patient_results[case_id] = {
            "risk": all_risk_scores[i],
            "censor": all_censorships[i],
            "time": all_event_times[i],
            "logits": all_logits[i]
        }

    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores,
                                                          all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss

def _save_results(cur, results_dict, args):
    filename = os.path.join(args.results_dir, "split_{}_results.pkl".format(cur))
    if os.path.exists(filename):
        os.remove(filename)
    print("Saving results...")
    _save_pkl(filename, results_dict)


_EPOCH_METRIC_FIELDS = [
    "fold", "epoch", "learning_rate",
    "train_loss", "train_survival_loss", "train_aux_loss",
    "train_decoder_loss", "train_reconstruction_loss",
    "train_weighted_decoder_loss", "train_weighted_reconstruction_loss",
    "train_vl_alignment_loss", "train_weighted_vl_alignment_loss",
    "train_cindex", "val_cindex", "val_cindex_ipcw", "val_BS", "val_IBS",
    "val_iauc", "val_loss", "generalization_gap", "is_best",
]


def _initialize_epoch_metrics(args, fold):
    path = os.path.join(args.results_dir, f"fold_{fold}_epoch_metrics.csv")
    with open(path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=_EPOCH_METRIC_FIELDS).writeheader()
    return path


def _append_epoch_metrics(path, row):
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_EPOCH_METRIC_FIELDS)
        writer.writerow({key: row[key] for key in _EPOCH_METRIC_FIELDS})


def _step(args, cur, loss_fn, model, dataset_factory, optimizer, scheduler, train_loader, val_loader, log_file):
    all_survival = _extract_survival_metadata(dataset_factory)
    eval_slot_seed = _fold_eval_slot_seed(args, cur)
    epoch_metrics_path = _initialize_epoch_metrics(args, cur)
    epoch_records = []
    best_record = None

    for epoch in range(args.max_epochs):
        train_metrics = _train_loop_survival(
            args, epoch, model, train_loader, optimizer, scheduler, loss_fn, log_file
        )
        results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss = _summary(
            args, dataset_factory, model, val_loader, loss_fn, all_survival,
            eval_slot_seed=eval_slot_seed,
        )
        print(
            'Epoch:{} Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
                epoch,
                val_cindex,
                val_cindex_ipcw,
                val_IBS,
                val_iauc
            ))
        log_file.write(
            'Epoch:{} Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}\n'.format(
                epoch,
                val_cindex,
                val_cindex_ipcw,
                val_IBS,
                val_iauc
            ))

        is_best = best_record is None or val_cindex > args.max_cindex
        record = {
            "fold": cur,
            "epoch": epoch,
            **train_metrics,
            "val_cindex": val_cindex,
            "val_cindex_ipcw": val_cindex_ipcw,
            "val_BS": val_BS,
            "val_IBS": val_IBS,
            "val_iauc": val_iauc,
            "val_loss": val_loss,
            "generalization_gap": train_metrics["train_cindex"] - val_cindex,
            "is_best": int(is_best),
        }
        epoch_records.append(record)
        _append_epoch_metrics(epoch_metrics_path, record)

        if is_best:
            args.max_cindex = val_cindex
            args.max_cindex_epoch = epoch
            best_record = record.copy()
            torch.save(model.state_dict(), os.path.join(args.results_dir, "model_best_s{}.pth".format(cur)))
            _save_results(cur, results_dict, args)

    # save the trained model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pth".format(cur)))

    final_record = epoch_records[-1]

    print(
        'Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
            final_record["val_cindex"],
            final_record["val_cindex_ipcw"],
            final_record["val_IBS"],
            final_record["val_iauc"]
        ))
    log_file.write(
        'Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}\n'.format(
            final_record["val_cindex"],
            final_record["val_cindex_ipcw"],
            final_record["val_IBS"],
            final_record["val_iauc"]
        ))

    best_model = torch.load(os.path.join(args.results_dir, "model_best_s{}.pth".format(cur)))
    model.load_state_dict(best_model)
    best_results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss = _summary(
        args, dataset_factory, model, val_loader, loss_fn, all_survival,
        eval_slot_seed=eval_slot_seed,
    )
    if eval_slot_seed is not None and not np.isclose(val_cindex, args.max_cindex, atol=1e-12):
        raise RuntimeError(
            "Deterministic best-checkpoint evaluation changed from "
            f"{args.max_cindex:.12f} to {val_cindex:.12f}"
        )
    print(
        'Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))
    log_file.write(
        'Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}\n'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))

    training_stats = {
        "best_epoch": int(args.max_cindex_epoch),
        "train_cindex_at_best": float(best_record["train_cindex"]),
        "generalization_gap": float(best_record["train_cindex"] - val_cindex),
        "final_epoch_val_cindex": float(final_record["val_cindex"]),
        "best_to_final_val_drop": float(val_cindex - final_record["val_cindex"]),
        "eval_slot_seed": eval_slot_seed,
    }

    return best_results_dict, (
        val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss,
        training_stats,
    )


def _train_val(args, dataset_factory, cur, log_file):
    # ---> get the splits and summarize the data
    train_data, test_data, train_loader, test_loader = _get_split(args, dataset_factory, cur)
    # ---> init the model, loss function and optimizer
    model = _init_model(args, dataset_factory)
    loss_fn = _init_loss_function(args)
    optimizer = _init_optim(args, model)
    scheduler = _init_scheduler(args, optimizer)
    # ---> train and validate
    results_dict, metrics = _step(args, cur, loss_fn, model, dataset_factory, optimizer, scheduler, train_loader, test_loader, log_file)
    return results_dict, metrics
