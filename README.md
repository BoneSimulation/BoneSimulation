# BoneSimulation

> **Enterprise-grade computational pipeline for 3-D bone micro-structure analysis, finite-element mesh generation, and structural simulation using CalculiX.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Linter](https://github.com/BoneSimulation/BoneSimulation/actions/workflows/pylint.yml/badge.svg)](https://github.com/BoneSimulation/BoneSimulation/actions/workflows/pylint.yml)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Pipeline Modes](#pipeline-modes)
9. [Output Artifacts](#output-artifacts)
10. [Documentation](#documentation)
11. [Contributing](#contributing)
12. [License](#license)

---

## Overview

**BoneSimulation** is a scientific software platform developed as part of the **HLRS (High Performance Computing Center Stuttgart)** project. It transforms stacks of micro-CT TIFF images of trabecular bone into fully parameterised finite-element models ready for structural simulation.

The pipeline covers every processing stage end-to-end:

```
CT scan slices (TIFF)
     │
     ▼
Image Loading & Pre-processing
     │  (Otsu thresholding, Gaussian blur, morphological closing)
     ▼
3-D Volume Reconstruction & Cluster Analysis
     │  (connected-component labelling, largest-cluster extraction)
     ▼
Surface & Tetrahedral Mesh Generation
     │  (Marching Cubes → CGAL tetrahedralisation)
     ▼
Block Extraction
     │  (cubic sub-volumes with individual tetra meshes)
     ▼
CalculiX FEA Export
     │  (.inp file with boundary conditions and loading)
     ▼
Simulation & Reporting
     │  (automated run script, PDF analysis report)
     ▼
Statistical Analysis & ML Optimisation (XGBoost / Optuna)
```

---

## Key Features

| Feature | Description |
|---|---|
| **Chunk-based I/O** | Loads thousands of TIFF slices in memory-safe chunks using parallel workers |
| **Adaptive thresholding** | Otsu threshold calculated on a random sub-sample for large volumes |
| **3-D morphological operations** | Gaussian smoothing + morphological closing for noise reduction |
| **Marching Cubes meshing** | Chunk-aware surface extraction with seam-free vertex stitching |
| **CGAL tetrahedral meshing** | High-quality volumetric meshes via `ciclope` / `pygalmesh` |
| **Block extraction** | Systematic cubic sub-volume extraction with per-block tetra meshing |
| **CalculiX integration** | Automatic `.inp` generation with boundary conditions, loading, and run scripts |
| **PDF reporting** | Porosity, cluster statistics, and histograms exported to PDF |
| **XGBoost / Optuna ML** | Bayesian hyperparameter search for morphological feature regression |
| **Multiple run modes** | `none` (full), `small` (mini-volume), `dry` (no file writes), `complete` (all checks) |

---

## System Architecture

```
BoneSimulation/
├── src/                    ← Core pipeline source code
│   ├── main.py             ← Entry point & orchestration
│   ├── image_loading.py    ← Parallel TIFF loading (chunked)
│   ├── image_processing.py ← Filtering, thresholding, interpolation
│   ├── cluster_analysis.py ← Connected-component labelling
│   ├── mesh_generation.py  ← Marching Cubes + tetrahedral meshing
│   ├── block_extraction.py ← Raster block extraction pipeline
│   ├── calculix_export.py  ← CalculiX .inp writer
│   ├── file_io.py          ← TIFF stream I/O helpers
│   ├── reporting.py        ← PDF report generation
│   ├── visualization.py    ← Matplotlib histogram & slice plots
│   ├── optimize_xgb_pipeline.py ← XGBoost + Optuna optimisation
│   └── utils/              ← Shared utility helpers
├── data/                   ← Input CT datasets
│   ├── bigdataset/         ← Full-resolution TIFF slices
│   └── dataset/            ← Small/test dataset
├── docs/                   ← Technical documentation (Markdown)
├── calculix/               ← Generated .inp files & run scripts
├── pictures/               ← VTK meshes, TIFF cluster exports
├── report/                 ← PDF analysis reports
├── logs/                   ← Fault logs and session logs
├── additional_code/        ← Standalone visualisation scripts
├── bibliography/           ← Referenced literature (PDF)
├── environment.yml         ← Conda environment specification
├── requirements.txt        ← pip package list
└── installer.sh            ← One-shot setup script
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 64 GB or more |
| CPU | 4 cores | 16+ cores (OpenMP) |
| GPU | — | NVIDIA (CUDA, for XGBoost GPU mode) |
| Disk | 10 GB | 100 GB+ (CT data can be large) |

### Software

- **Operating System**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.12
- **Conda**: Anaconda or Miniconda
- **CalculiX** (`ccx`): version 2.22 (binary included as `cgx_2.22`)

---

## Installation

### Option 1 — Automated (recommended)

```bash
git clone https://github.com/BoneSimulation/BoneSimulation.git
cd BoneSimulation
bash installer.sh
```

The installer will:
1. Create a Conda environment named `py3818` from `environment.yml`
2. Activate the environment
3. Install all additional pip packages from `requirements.txt`

### Option 2 — Manual

```bash
# 1. Clone the repository
git clone https://github.com/BoneSimulation/BoneSimulation.git
cd BoneSimulation

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate py3818

# 3. Install pip packages
pip install -r requirements.txt
```

### Keeping the code up to date

```bash
git pull origin master
```

---

## Quick Start

```bash
conda activate py3818
cd src

# Full pipeline on the big dataset
python main.py

# Mini-volume test (fast, uses CHUNK_SIZE=5)
python main.py --test small

# Dry run — no files are written to disk
python main.py --test dry

# Complete mode — all steps and reports are verified
python main.py --test complete
```

---

## Pipeline Modes

| Mode flag | Description |
|---|---|
| *(none / default)* | Full pipeline on `data/bigdataset` |
| `--test small` | Reduces `CHUNK_SIZE` to 5 for rapid iteration |
| `--test dry` | Runs all computations but skips every file write |
| `--test fast_debug` | Alias for `small` |
| `--test complete` | Full pipeline with extra validation of all outputs |

---

## Output Artifacts

| Location | Content |
|---|---|
| `pictures/` | Surface meshes (`.vtk`), largest cluster (`.tif`), block sub-volumes |
| `pictures/blocks_<timestamp>/` | Per-block `.vti` raw volumes and `_tetra.vtk` meshes |
| `calculix/bone_sim_<timestamp>.inp` | CalculiX FEA input file |
| `calculix/run_simulation.sh` | Auto-generated shell script to execute CalculiX |
| `report/analysis_report_<timestamp>.pdf` | Porosity, cluster statistics, cluster-size histogram |
| `logs/fault_log.txt` | Low-level fault handler log |
| `logfile.log` | Full pipeline log (timestamped, INFO level) |

---

## Documentation

Detailed technical documentation is available in the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Full pipeline architecture and data-flow diagram |
| [`docs/modules.md`](docs/modules.md) | Per-module API reference |
| [`docs/installation.md`](docs/installation.md) | Detailed installation and troubleshooting guide |
| [`docs/usage.md`](docs/usage.md) | How to run the pipeline and customise parameters |
| [`docs/data-formats.md`](docs/data-formats.md) | Input/output file format specifications |
| [`docs/calculix-integration.md`](docs/calculix-integration.md) | CalculiX FEA export and simulation guide |
| [`docs/ml-optimization.md`](docs/ml-optimization.md) | XGBoost / Optuna hyperparameter optimisation pipeline |

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/my-feature`
2. Follow the existing code style (PEP 8, existing logging patterns)
3. Run pylint before opening a pull request: `pylint src/`
4. Open a pull request against `master`

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

*Developed as part of the HLRS computational science project. For questions, open an issue on GitHub.*
