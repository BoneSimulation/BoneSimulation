# Installation Guide

This guide covers every step required to set up BoneSimulation on a Linux or macOS system, including hardware recommendations, dependency installation, and common troubleshooting steps.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Clone the Repository](#clone-the-repository)
3. [Automated Installation](#automated-installation)
4. [Manual Installation](#manual-installation)
5. [Verifying the Installation](#verifying-the-installation)
6. [Installing CalculiX](#installing-calculix)
7. [GPU Support (Optional)](#gpu-support-optional)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 64 GB |
| CPU cores | 4 | 16+ |
| Disk space | 10 GB | 100 GB+ |
| GPU | — | NVIDIA (CUDA 11+) for XGBoost acceleration |

### Software

| Tool | Version | Notes |
|---|---|---|
| Linux (Ubuntu / Debian) or macOS | — | Windows is not supported |
| [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) | Latest | Required for environment management |
| Git | 2.x | For cloning and updates |
| `lscpu` | — | Used to detect CPU thread count (part of `util-linux`, standard on Linux) |

---

## Clone the Repository

```bash
git clone https://github.com/BoneSimulation/BoneSimulation.git
cd BoneSimulation
```

---

## Automated Installation

The `installer.sh` script handles environment creation and package installation in one step:

```bash
bash installer.sh
```

**What it does**:
1. Runs `conda env create -f environment.yml` — creates the `py3818` environment with all system-level and Python packages.
2. Activates the environment.
3. Runs `pip install -r requirements.txt` — installs additional packages not covered by the Conda spec.

The script exits with an error message if any step fails.

---

## Manual Installation

If you prefer to install step by step:

### Step 1 — Create the Conda environment

```bash
conda env create -f environment.yml
```

This creates an environment named `py3818` with Python 3.12 and all dependencies (NumPy, SciPy, scikit-image, VTK, PyVista, meshio, ciclope, etc.).

> **Note**: The `environment.yml` file pins exact versions for reproducibility. The `prefix` path at the bottom of the file is from the original developer's machine and is **ignored** during installation on other systems.

### Step 2 — Activate the environment

```bash
conda activate py3818
```

### Step 3 — Install additional pip packages

```bash
pip install -r requirements.txt
```

Key packages installed at this step:

| Package | Version | Purpose |
|---|---|---|
| `ciclope` | ~1.3.5 | CGAL-based tetrahedral meshing |
| `PyMCubes` | latest | Alternative Marching Cubes implementation |
| `xgboost` | ~3.0.2 | Gradient boosting for feature regression |
| `optuna` | ~4.4.0 | Bayesian hyperparameter optimisation |
| `fpdf2` | ~2.8.3 | PDF report generation |
| `numba` | ~0.61.2 | JIT compilation for performance-critical functions |

### Step 4 — Verify

```bash
conda activate py3818
python -c "import numpy, scipy, skimage, meshio, ciclope, pyvista, vtk, fpdf, optuna, xgboost; print('All imports OK')"
```

---

## Verifying the Installation

Run the pipeline in dry mode to confirm that all modules load correctly without writing any output files:

```bash
conda activate py3818
cd src
python main.py --test dry
```

Expected output (truncated):
```
INFO: Starte Processing Pipeline (test_mode = 'dry')
INFO: TESTMODE: Dry run (keine Dateien werden gespeichert!)
INFO: Loading images from: data/bigdataset
...
INFO: [Dry] Skip saving largest cluster
INFO: [Dry] Skip saving surface mesh
INFO: [Dry] Skip CalculiX export
INFO: [Dry] Skip saving analysis report
INFO: Processing completed successfully.
```

For a full end-to-end test on a small volume:

```bash
python main.py --test small
```

---

## Installing CalculiX

CalculiX (`ccx`) is required for the FEA simulation step. The repository includes the `cgx_2.22` pre-processor binary, but you need to install `ccx` separately.

### Option A — Apt (Ubuntu / Debian)

```bash
sudo apt-get install calculix-ccx
ccx --version
```

### Option B — Build from source

```bash
# Download CalculiX 2.22 source
wget http://www.dhondt.de/ccx_2.22.src.tar.bz2
tar -xjf ccx_2.22.src.tar.bz2
cd CalculiX/ccx_2.22/src
# Follow the build instructions in the README
```

### Option C — Use the included binary path

When running the pipeline you will be prompted:

```
Please enter the preferred solver you would like to use (default: ccx):
```

You can enter the full path to any CalculiX binary, for example `./ccx_2.22`.

---

## GPU Support (Optional)

The XGBoost optimisation pipeline (`optimize_xgb_pipeline.py`) can use GPU acceleration via `cupy` and XGBoost's `gpu_hist` tree method.

### Install cupy

Match the cupy version to your CUDA version:

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

Verify:

```bash
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
```

The pipeline auto-detects `cupy` availability and falls back to CPU gracefully if it is not installed.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ciclope'`

```bash
pip install ciclope[all]
```

### `ImportError: libGL.so.1: cannot open shared object file`

Install OpenGL libraries:

```bash
sudo apt-get install libgl1-mesa-glx libglu1-mesa
```

### `ciclope.core.tetraFE` not found

This means the optional CGAL backend is missing. Install `pygalmesh`:

```bash
pip install pygalmesh
```

### `lscpu: command not found`

On macOS, `lscpu` is not available. The `get_max_threads()` function in `main.py` will fail. Either install it (`brew install util-linux`) or replace the function with:

```python
import os
return os.cpu_count()
```

### Conda environment creation fails

If the `environment.yml` fails due to conflicting channels, try creating a minimal environment and installing via pip only:

```bash
conda create -n py3818 python=3.12
conda activate py3818
pip install -r requirements.txt
```

### Memory errors on large datasets

- Reduce `CHUNK_SIZE` in `main.py` (e.g., from 50 to 20).
- Lower `MEMORY_LIMIT_PERCENT` to trigger earlier wait cycles.
- Enable swap space on your system: `sudo fallocate -l 32G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.

### Pipeline hangs indefinitely

If `wait_for_memory()` never resolves, another process may be consuming RAM. Check with `htop` or `free -h` and kill the offending process.
