# VMD-GeoBR-Mamba

<div align="center">

[![Target Journal](https://img.shields.io/badge/Target%20Journal-Computers%20%26%20Geosciences-brightgreen)](https://www.sciencedirect.com/journal/computers-and-geosciences)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Code%20Availability-orange)](#project-status)

**A self-supervised framework for geomagnetic background modeling and residual anomaly extraction**

[Installation](#installation) | [Quick Start](#quick-start) | [Repository Structure](#repository-structure) | [Data Availability](#data-availability)

</div>

---

## Overview

VMD-GeoBR-Mamba is a time-series modeling framework for geomagnetic background-residual decomposition. The released code trains and applies a self-supervised GeoBR-Mamba model on a VMD-derived IMF component, typically `IMF_3`.

The model learns a smooth geomagnetic background while separating localized residual disturbances without requiring manually labeled anomaly segments. Anomaly-related residuals can then be calculated against the estimated background rather than the complete reconstructed signal.

This repository focuses on the executable training and inference workflow used for code availability. Detailed scientific motivation, equations, case-study interpretation, and experimental discussion are provided in the associated manuscript.

## Main Features

```text
VMD-derived IMF input             -> Uses an IMF component such as IMF_3
Multi-scale causal convolution    -> Captures local temporal fluctuations
Mamba sequence modeling           -> Models long-range temporal dependency
Background-residual separation    -> Decomposes IMF = background + residual
Self-supervised objective         -> Requires no manually labeled anomalies
Reproducible outputs              -> Saves checkpoint, prediction tables, and figures
```

## Repository Structure

```text
VMD-GeoBR-Mamba/
  README.md
  requirements.txt
  configs/
    train_example.json
    infer_example.json
  models/
    __init__.py
    geobr_mamba.py   # GeoBR-Mamba model definition
    losses.py        # unified self-supervised decomposition objective
  data/
    __init__.py
    dataset.py       # CSV loading, scaling, and sliding-window datasets
  utils/
    __init__.py
    common.py        # seed and argument helpers
    evaluation.py    # evaluation loop and prediction table generation
    inference.py     # checkpoint metadata loading and inference loop
    plotting.py      # training and inference figure generation
  scripts/
    train.py         # model training entry point
    infer.py         # checkpoint inference entry point
  notebooks/
    README.md        # optional exploratory notebooks
```

## Installation

```bash
git clone https://github.com/<YOUR_USERNAME>/VMD-GeoBR-Mamba.git
cd VMD-GeoBR-Mamba

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

`mamba-ssm` may require a CUDA-compatible PyTorch environment. If installation fails, install the PyTorch build matching your CUDA/runtime first, then install `mamba-ssm`.

## Quick Start

### 1. Prepare Input Data

Prepare a CSV file containing the target IMF column. The default column name is:

```text
IMF_3
```

A date/time column is optional. If present, one of the following names is detected automatically:

```text
Date, date, time, Time, datetime, Datetime
```

If no date column exists, synthetic minute-level timestamps are generated.

### 2. Train

```bash
python scripts/train.py \
  --csv CD_train_300.csv \
  --column IMF_3 \
  --seq-len 300 \
  --pred-len 1 \
  --epochs 30 \
  --batch-size 512
```

Training outputs are saved to `mamba_usb_decomp_tgrs_outputs/` by default:

```text
mamba_usb_decomp_tgrs.pt
decomp_pred_long.csv
decomp_pred_agg.csv
decomp_bg_prediction.png
decomp_residual.png
decomp_gate.png
loss_history.csv
loss_history.png
config.json
```

### 3. Run Inference

```bash
python scripts/infer.py \
  --csv LS_CD_test_400.csv \
  --ckpt mamba_usb_decomp_tgrs_outputs/mamba_usb_decomp_tgrs.pt
```

The inference script reads the model hyperparameters, target column, sequence length, prediction length, and normalization statistics from the checkpoint. Use `--column` only when the input column should override the checkpoint metadata.

Inference outputs are saved to `mamba_usb_infer_outputs/` by default:

```text
decomp_pred_long.csv
decomp_pred_agg.csv
infer_bg_residual_recon.png
```

Use `--show` to display the inference figure after saving it.

## Example Configs

The `configs/` directory contains compact JSON examples documenting commonly used training and inference settings. The current command-line scripts take arguments directly, so these files are provided as reproducibility records rather than required runtime inputs.

## Key Arguments

| Argument | Default | Description |
|---|---:|---|
| `--csv` | script-specific | Input CSV path |
| `--column` | `IMF_3` for training | Target IMF column |
| `--seq-len` | `300` | Input window length |
| `--pred-len` | `1` | Prediction horizon |
| `--d-model` | `32` | Model hidden dimension |
| `--kernels` | `3,7,15` | Multi-scale convolution kernels |
| `--lambda-bg` | `0.40` | Background prior weight |
| `--lambda-res` | `0.01` | Residual prior weight |
| `--aggregate` | `mean` | Aggregation method for overlapping predictions |

## Data Availability

Raw geomagnetic observations may be subject to restrictions from the original data providers. This repository expects preprocessed CSV inputs with a VMD-derived IMF column, such as `IMF_3`.

For reproducibility, users should provide or document:

- the source and access conditions of raw geomagnetic observations;
- the VMD preprocessing procedure used to obtain IMF components;
- the CSV column names and sampling interval;
- the training/test split or station/event selection protocol;
- any released sample or synthetic data used for software verification.

## Citation

Please update the following entry after publication:

```bibtex
@misc{qin2026vmdgeobrmamba,
  title  = {VMD-GeoBR-Mamba},
  author = {Qin, Changfeng and Zhang, Jie and Mu, Feng and Sun, Minhao and Chi, Chengquan and Yu, Zining},
  year   = {2026},
  note   = {Software repository}
}
```

## License

This project is intended for release under the MIT License. Third-party libraries and datasets remain subject to their respective licenses and usage agreements.

## Project Status

This repository is being prepared for public code availability. The current version provides the core training and inference implementation for GeoBR-Mamba.
