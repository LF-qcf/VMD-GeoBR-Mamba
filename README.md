# VMD-GeoBR-Mamba

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Code](https://img.shields.io/badge/Code-Available-brightgreen)](#-quick-start)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange)](#-project-status)

**A self-supervised framework for geomagnetic background modeling and residual anomaly extraction**

[📌 Highlights](#-highlights) · [🎯 Core Innovation](#-core-innovation) · [💻 Quick Start](#-quick-start) · [🗂️ Structure](#%EF%B8%8F-repository-structure) · [🗃️ Data](#%EF%B8%8F-data-availability)

</div>

---

## 📑 Table of Contents

<details open>
<summary><b>Click to expand or collapse</b></summary>

- [📌 Highlights](#-highlights)
- [🔍 Overview](#-overview)
- [🎯 Core Innovation](#-core-innovation)
- [🧩 Implementation Features](#-implementation-features)
- [🏗️ Repository Structure](#%EF%B8%8F-repository-structure)
- [⚙️ Installation](#%EF%B8%8F-installation)
- [💻 Quick Start](#-quick-start)
  - [Prepare Input Data](#1-prepare-input-data)
  - [Train](#2-train)
  - [Run Inference](#3-run-inference)
- [🛠️ Key Arguments](#%EF%B8%8F-key-arguments)
- [♻️ Reproducibility](#%EF%B8%8F-reproducibility)
- [🗃️ Data Availability](#%EF%B8%8F-data-availability)
- [📚 Citation](#-citation)
- [📄 License](#-license)
- [🚧 Project Status](#-project-status)

</details>

---

## 📌 Highlights

- VMD isolates short-period geomagnetic variations for background modeling.
- GeoBR-Mamba separates smooth backgrounds from sparse disturbances.
- Background-only forecasts enhance residual anomaly distinguishability.

---

## 🔍 Overview

**VMD-GeoBR-Mamba** is a time-series modeling repository for geomagnetic background–disturbance decomposition.

The framework accepts a VMD-derived intrinsic mode function as input, learns a smooth background component, separates localized disturbances, and calculates residuals against the estimated background.

This repository focuses on the executable software workflow, including:

- data loading and sliding-window construction;
- model training and checkpoint management;
- background and disturbance inference;
- prediction aggregation;
- result export and visualization.

Detailed scientific discussion and case-specific interpretation are intentionally not included in this README.

---

## 🎯 Core Innovation

### Background-preserving self-supervised decomposition

Conventional forecasting models may reconstruct transient disturbances together with normal temporal variations. GeoBR-Mamba instead models two complementary components:

```text
Observed IMF component
        |
        +--> Smooth background component
        |
        +--> Sparse disturbance component
```

During inference, anomaly residuals are calculated using only the estimated background:

```text
Residual = Observation - Estimated Background
```

This design helps preserve localized departures that may otherwise be absorbed into the full reconstruction.

---

## 🧩 Implementation Features

```text
CSV data pipeline              -> Loads, scales, and windows IMF sequences
Multi-scale causal convolution -> Extracts local patterns at multiple scales
Mamba temporal encoder         -> Models long-range temporal dependencies
Configurable training          -> Supports adjustable windows and loss weights
Checkpoint-based inference     -> Restores model settings and normalization
Prediction aggregation         -> Combines overlapping window predictions
Output management              -> Saves checkpoints, tables, configs, and figures
```

---

## 🏗️ Repository Structure

```text
VMD-GeoBR-Mamba/
├── README.md
├── LICENSE
├── requirements.txt
│
├── configs/
│   ├── train_example.json
│   └── infer_example.json
│
├── models/
│   ├── __init__.py
│   ├── geobr_mamba.py
│   └── losses.py
│
├── data/
│   ├── __init__.py
│   └── dataset.py
│
├── utils/
│   ├── __init__.py
│   ├── common.py
│   ├── evaluation.py
│   ├── inference.py
│   └── plotting.py
│
├── scripts/
│   ├── train.py
│   └── infer.py
│
├── notebooks/
│   └── README.md
│
└── outputs/
    ├── train/
    └── inference/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/LF-qcf/VMD-GeoBR-Mamba.git
cd VMD-GeoBR-Mamba
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `mamba-ssm` may require a CUDA-compatible PyTorch environment. Install the PyTorch build matching the local CUDA runtime before installing `mamba-ssm`.

---

## 💻 Quick Start

### 1. Prepare Input Data

Prepare a CSV file containing the target IMF column.

Default column name:

```text
IMF_3
```

An optional date/time column may use one of the following names:

```text
Date, date, time, Time, datetime, Datetime
```

If no date/time column is provided, the code can generate synthetic minute-level timestamps for indexing.

Example:

```csv
Datetime,IMF_3
2026-01-01 00:00:00,0.0124
2026-01-01 00:01:00,0.0097
2026-01-01 00:02:00,-0.0041
```

### 2. Train

```bash
python scripts/train.py \
  --csv path/to/train.csv \
  --column IMF_3 \
  --seq-len 300 \
  --pred-len 1 \
  --epochs 30 \
  --batch-size 512 \
  --output-dir outputs/train
```

Typical training outputs:

```text
outputs/train/
├── best_model.pt
├── config.json
├── loss_history.csv
├── loss_history.png
├── decomp_pred_long.csv
├── decomp_pred_agg.csv
├── background_prediction.png
├── residual_prediction.png
└── gate_prediction.png
```

### 3. Run Inference

```bash
python scripts/infer.py \
  --csv path/to/test.csv \
  --ckpt outputs/train/best_model.pt \
  --output-dir outputs/inference
```

Typical inference outputs:

```text
outputs/inference/
├── decomp_pred_long.csv
├── decomp_pred_agg.csv
└── background_residual_reconstruction.png
```

Use `--show` to display the generated figure after saving it.

---

## 🛠️ Key Arguments

| Argument | Default | Description |
|---|---:|---|
| `--csv` | Required | Input CSV path |
| `--column` | `IMF_3` | Target IMF column |
| `--seq-len` | `300` | Input window length |
| `--pred-len` | `1` | Prediction horizon |
| `--d-model` | `32` | Hidden dimension |
| `--kernels` | `3,7,15` | Multi-scale convolution kernels |
| `--lambda-bg` | `0.40` | Background regularization weight |
| `--lambda-res` | `0.01` | Disturbance regularization weight |
| `--aggregate` | `mean` | Overlapping-prediction aggregation |
| `--output-dir` | Script-specific | Output directory |

View the complete command-line options with:

```bash
python scripts/train.py --help
python scripts/infer.py --help
```

---

## ♻️ Reproducibility

The workflow records the information required to rerun an experiment:

- random seed;
- model hyperparameters;
- input and prediction lengths;
- normalization statistics;
- target column;
- checkpoint metadata;
- training history;
- prediction tables;
- generated figures.

Recommended release checklist:

```text
[ ] requirements.txt is complete
[ ] public commands run without local path changes
[ ] sample or synthetic input data is included
[ ] configuration files match released checkpoints
[ ] all figures can be regenerated
[ ] repository includes an open-source license
[ ] a tagged release has been created
```

---

## 🗃️ Data Availability

Raw geomagnetic observations may be subject to restrictions imposed by the original data providers.

This repository expects preprocessed CSV files containing a VMD-derived IMF column. To support software verification, users should document:

- input column names;
- physical units;
- sampling interval;
- preprocessing steps;
- missing-value handling;
- train, validation, and test split rules.

When raw observations cannot be redistributed, include a small permitted sample or synthetic dataset that can run through the complete training and inference workflow.

---

## 📚 Citation

For the software repository, use:

```bibtex
@misc{vmdgeobrmamba,
  title  = {VMD-GeoBR-Mamba},
  year   = {2026},
  note   = {Software repository},
  url    = {https://github.com/LF-qcf/VMD-GeoBR-Mamba}
}
```

Update the entry when a permanent archive or publication record becomes available.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

Third-party libraries and datasets remain subject to their respective licenses and usage agreements.

---

## 🚧 Project Status

The repository provides the core implementation for:

- model training;
- checkpoint loading;
- background estimation;
- disturbance separation;
- residual calculation;
- prediction export;
- result visualization.

---

<div align="center">

### ⭐ VMD-GeoBR-Mamba

**Model the background, preserve localized departures, and keep the workflow reproducible.**

[⬆️ Back to Top](#vmd-geobr-mamba)

</div>
