<div align="center">

<h1><a href="https://openreview.net/forum?id=WqCRSn2WAY">Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis</a></h1>

**[Yilan Zhang](https://scholar.google.com/citations?user=wZ4M4ecAAAAJ&hl), [Li Nanbo](https://scholar.google.com/citations?user=wa2S8OEAAAAJ&hl), [Changchun Yang](https://scholar.google.com/citations?user=pkxQnAYAAAAJ&hl), [J&uuml;rgen Schmidhuber](https://scholar.google.com/citations?user=gLnCTgIAAAAJ&hl), and [Xin Gao](https://scholar.google.com/citations?user=wqdK8ugAAAAJ&hl)**

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202026-8C1AFF)](https://openreview.net/forum?id=WqCRSn2WAY)
[![arXiv](https://img.shields.io/badge/arXiv-2512.01116-b31b1b.svg)](https://arxiv.org/abs/2512.01116)
![Visitors](https://komarev.com/ghpvc/?username=zylvemvetSlotSPE&label=visitors)
[![GitHub Stars](https://img.shields.io/github/stars/zylvemvet/SlotSPE?style=social)](https://github.com/zylvemvet/SlotSPE/stargazers)
![License](https://img.shields.io/badge/license-CC--BY--NC--4.0-blue)

</div>

<p align="center">
  <img src="./figures/figure1.svg" width="92%" alt="SlotSPE architecture overview">
</p>

## Overview

SlotSPE is the official implementation of **Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis**. The method models sparse, patient-specific structural prognostic events from whole-slide image (WSI) features and omics data, enabling compact multimodal interaction modeling for cancer survival prediction. The latent prognostic events are dynamically instantiated for each patient using a slot attention module, allowing the model to capture individualized prognostic patterns while preserving a concise representation.

<p align="center">
  <img src="./figures/visualization.svg" width="86%" alt="Interpretability of slots">
</p>


This repository contains:

- the main training and evaluation code for SlotSPE
- packaged clinical metadata, signatures, and fold splits under `dataset_csv/`
- externally hosted RNA feature tables for `dataset_csv/raw_rna_data_inter/`
- a separate preprocessing pipeline under `feature_extract/` for TCGA slide download, patching, and feature extraction

If you run into problems, open an issue or contact `yilan.zhang@kaust.edu.sa`.

## Repository Layout

- `survival.py`: main training and evaluation entrypoint
- `models/`: SlotSPE architecture, slot attention, multimodal fusion, and transformer modules
- `dataset/`: data loading for WSI features, omics inputs, and survival labels
- `dataset_csv/`: clinical tables, pathway/signature definitions, packaged fold splits, and the expected location for downloaded RNA feature tables
- `scripts/`: example cluster launchers; `scripts/SlotSPE.sh` is a SLURM/HPC example
- `feature_extract/`: preprocessing workflow for raw TCGA WSIs
- `figures/`: paper figures used in this README

## Environment Setup

Create a Python environment first:

```bash
conda create -n slotspe python=3.10
conda activate slotspe
```

Install PyTorch for your platform and CUDA setup, e.g.:

```bash
pip install torch torchvision
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is required by the training code but is not currently listed in `requirements.txt`, so install it separately using the official instructions for your system. The provided training workflow is GPU-oriented. The example launcher in [scripts/SlotSPE.sh](scripts/SlotSPE.sh) targets a SLURM cluster with CUDA available.

## Data Preparation

SlotSPE expects two main inputs:

1. Precomputed WSI feature tensors passed through `--data_root_dir`
2. Metadata and omics tables under `--data_path` (default: `./dataset_csv`)

### WSI feature files

- Each slide is loaded as a PyTorch tensor from `--data_root_dir`
- Filenames are expected to match the slide IDs in the clinical CSVs, with the slide suffix converted to `.pt`
- If a patient has multiple slides, the loader concatenates the corresponding tensors

Example layout:

```text
<data_root_dir>/
  TCGA-XX-XXXX-01Z-00-DX1.pt
  TCGA-YY-YYYY-01Z-00-DX1.pt
  ...
```

### Metadata under `dataset_csv/`

The current training code reads:

- `dataset_csv/clinical/all/<study>.csv` for case IDs, survival labels, censorship indicators, and slide IDs
- `dataset_csv/raw_rna_data_inter/<study>_rna_inter.csv` for omics inputs
- `dataset_csv/signatures/*.csv` for pathway or signature definitions
- `dataset_csv/splits/5fold/<study>/fold_<k>.csv` for train/validation splits

The large RNA feature tables under `dataset_csv/raw_rna_data_inter/` are **not stored in this GitHub repository**. Please download them from Google Drive and place them under `dataset_csv/raw_rna_data_inter/` before running training:

- [Google Drive: raw_rna_data_inter](https://drive.google.com/drive/folders/1RxCjSZYTWhJRnbYWAGySyvZk2RUKYb1t?usp=sharing)

Expected layout:

```text
dataset_csv/
  clinical/
  raw_rna_data_inter/
    blca_rna_inter.csv
    brca_rna_inter.csv
    coadread_rna_inter.csv
    hnsc_rna_inter.csv
    kirc_rna_inter.csv
    luad_rna_inter.csv
    lusc_rna_inter.csv
    skcm_rna_inter.csv
    stad_rna_inter.csv
    ucec_rna_inter.csv
  signatures/
  splits/
```

Packaged 5-fold splits are currently included for:

- `blca`
- `brca`
- `coadread`
- `hnsc`
- `kirc`
- `luad`
- `lusc`
- `skcm`
- `stad`
- `ucec`

## Feature Extraction Pipeline

The root training pipeline does not start from raw WSIs. It expects precomputed slide-level patch features saved as `.pt` tensors.

For preprocessing, use the dedicated materials in [feature_extract/README.md](feature_extract/README.md) or [Pipeline-Processing-TCGA-Slides-for-MIL](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL), especially:

- `S01-Downloading-Slides-from-TCGA.ipynb`
- `S02-Reorganizing-Slides-at-Patient-Level.ipynb`
- `S03-Segmenting-and-Patching-Slides.ipynb`
- `S04-Extracting-Patch-Features.ipynb`

That submodule also includes CLAM-based tooling and helper scripts for feature extraction with multiple pathology encoders.

## Quick Start

The public training entrypoint is:

```bash
python survival.py
```

A minimal example using the packaged metadata looks like this:

```bash
python survival.py \
  --data_root_dir /path/to/<study>/pt_files \
  --data_path ./dataset_csv \
  --results_dir ./results \
  --study brca \
  --rna_format Pathways \
  --signature combine \
  --label_col survival_months_dss \
  --encoding_dim 1024 \
  --num_patches 4096 \
  --slot_num_wsi 8 \
  --slot_num_omics 8 \
  --slot_iters 10 \
  --temperature 0.01 \
  --topk_ratio 0.25
```

Useful arguments:

- `--data_root_dir`: directory containing slide-level `.pt` feature tensors
- `--data_path`: root directory for `dataset_csv` metadata
- `--study`: cancer type to train on
- `--label_col`: survival endpoint, such as `survival_months_dss`
- `--encoding_dim`: WSI feature dimension expected in the `.pt` files
- `--num_patches`: number of patches sampled during training
- `--slot_num_wsi`: number of WSI slots
- `--slot_num_omics`: number of omics slots
- `--slot_iters`: slot attention iterations
- `--temperature`: gating temperature
- `--topk_ratio`: fraction of slots kept by the top-k gating module

The example script [scripts/SlotSPE.sh](scripts/SlotSPE.sh) shows how these arguments are combined for SLURM-based runs across studies and seeds.

## CONCH Event Grounding (TCGA-KIRC)

The fair primary comparison uses one normalized CONCH contrastive tensor for
both `X_slot` and `Z_conch`. Original SlotSPE and Event-Gated SlotSPE therefore
see the exact same patches and features; only event grounding differs. A legacy
dual-stream UNI+CONCH experiment remains possible, but UNI, ResNet, or
CTransPath vectors are rejected if passed as CONCH evidence. The default
`slot_attention_type=original` remains unchanged.

### 1. Install CONCH and build KIRC event text embeddings

CONCH is a gated model. Request access at
[MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH), then use either a
local checkpoint or `HF_TOKEN`; tokens and weights are never hard-coded.

```bash
bash scripts/setup_conch.sh

# Local checkpoint mode (large artifacts are kept on /data0 by default)
CONCH_CHECKPOINT=/data0/lfy_data/Pathology/checkpoints/conch/pytorch_model.bin \
  CONCH_DEVICE=cuda \
  bash scripts/build_tcga_kirc_event_bank.sh

# Hugging Face mode
export HF_TOKEN=<your_token>
CONCH_FROM_HF=1 CONCH_DEVICE=cuda \
  bash scripts/build_tcga_kirc_event_bank.sh
```

The checked-in `assets/event_bank/tcga_kirc/events_raw.json` contains 30
Codex-generated candidates. They are machine-checkable candidates, not expert
validated pathology labels. To generate another cancer-specific vocabulary,
use `tools/generate_event_candidates.py --cancer-type ... --dataset ...` and
review it with `tools/review_event_candidates.py`.

Those files are retained as the v0 baseline. A separate scale-aware v2 workflow
contains 38 evidence-tracked candidates, a 22-event patch-local core, and
separate interface/WSI-context and reserve catalogs:

```bash
CONCH_DEVICE=cuda bash scripts/build_tcga_kirc_event_bank_v2.sh

# Also audit v0/v2 activation on an existing patch-feature directory.
CONCH_DEVICE=cuda \
V2_PATCH_FEATURE_DIR=/data0/lfy_data/Pathology/CONCH/kirc/pt_files \
bash scripts/build_tcga_kirc_event_bank_v2.sh
```

See [the v2 event-bank documentation](assets/event_bank/tcga_kirc/v2/README.md)
for the schema, curation status, evidence sources, threshold sensitivity audit,
and required expert-review checkpoint. The v2 workflow never overwrites v0.

### 2. Stream TCGA-KIRC WSIs into CONCH PT files

The repository already contains the 939-slide GDC manifest. The disk-bounded
pipeline downloads one WSI at a time, creates non-overlapping 448x448 patches
at 20x, encodes them with the official frozen CONCH model, validates normalized
`[N,512]` float16 tensors, then deletes that raw WSI. Coordinate HDF5 files are
retained because they are small and preserve reproducibility.

```bash
mkdir -p /data0/lfy_data/Pathology/checkpoints/conch
# Manually place the gated pytorch_model.bin in the directory above.

bash scripts/run_kirc_conch_preprocess.sh
tmux attach -t kirc_conch
```

Defaults use `/data0/lfy_data/Pathology`, which currently has substantially
more free disk than the root filesystem. Override `PATHOLOGY_DATA_ROOT`,
`MODEL_CKPT`, `FEAT_DIR`, or `EXTRACT_BATCH_SIZE` through environment variables.
The process is resumable and writes PT files atomically. Use the read-only
storage report at any time:

```bash
bash scripts/audit_kirc_storage.sh
```

### 3. Run baseline or Event-Gated SlotSPE

Use the same CONCH PT directory for the fair baseline:

```bash
bash scripts/run_kirc_conch_baseline.sh
```

Event-Gated training reuses those exact tensors as CONCH evidence:

```bash
EVENT_BANK_PATH=assets/event_bank/tcga_kirc/conch_event_bank.pt \
  bash scripts/run_kirc_event_gated.sh
```

The historical UNI baseline remains available separately, but it is not the
primary architecture-only comparison against Event-Gated SlotSPE.

Set `--return_event_details` to return a third forward value containing
`B_pre`, `B_post`, raw/support patch-event evidence, slot-event routing, visual
support, all structured gates, and dominant event indices. Save that dictionary
with `torch.save` and inspect it with:

```bash
python tools/inspect_slot_events.py \
  --details /path/to/details.pt --slide-id TCGA-XX-XXXX --open-threshold 0.2
```

The inspection threshold only labels reports; it never replaces the continuous
training gate.

### 4. Tests

```bash
python -m unittest discover -s tests -v
```

Tests that require the official package or real gated weights report an explicit
skip when those external resources are unavailable. Unit tests never substitute
random weights for a claimed CONCH artifact.

### 5. Collaborating

Keep datasets, gated weights, checkpoints, logs, and experiment outputs outside
Git. Develop changes on a named feature branch and merge them through a reviewed
pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for the setup, branch,
testing, experiment-reporting, and data-security workflow.

### Expected Outputs

For each run, the code creates a study- and experiment-specific subdirectory under `--results_dir`. The pipeline writes:

- `experiment_settings.txt`: stored experiment configuration
- `log_start_<k_start>_end_<k_end>.txt` or `log_test.txt`: training or test logs
- `split_<fold>_results.pkl`: per-fold validation predictions used by the Kaplan-Meier utility
- `split_<fold>_results_final.pkl`: additional fold result pickle written by `survival.py`
- `summary.csv` or `summary_partial_<start>_<end>.csv`: aggregated fold metrics
- `KaplanMeier-r1.png`: Kaplan-Meier plot generated after cross-validation

## Citation

If this repository is useful in your work, please cite:

```bibtex
@inproceedings{zhang2026slotspe,
  title={Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis},
  author={Zhang, Yilan and Li, Nanbo and Yang, Changchun and Schmidhuber, J{\"u}rgen and Gao, Xin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Acknowledgement

The preprocessing resources under `feature_extract/` build on CLAM-based pathology tooling and related TCGA slide-processing workflows included in this repository.

We thank the following repositories for their contributions:

- [CLAM](https://github.com/mahmoodlab/CLAM)
- [Pipeline-Processing-TCGA-Slides-for-MIL](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL)
