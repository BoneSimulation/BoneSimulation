# Usage Guide

This document explains how to run the BoneSimulation pipeline, describes all available run modes, and covers how to customise the key parameters.

---

## Table of Contents

1. [Running the Pipeline](#running-the-pipeline)
2. [Command-Line Arguments](#command-line-arguments)
3. [Run Modes](#run-modes)
4. [Customising Parameters](#customising-parameters)
5. [Typical Workflows](#typical-workflows)
6. [Monitoring Progress](#monitoring-progress)
7. [Standalone Scripts](#standalone-scripts)

---

## Running the Pipeline

All pipeline commands should be executed from the `src/` directory with the `py3818` Conda environment active.

```bash
conda activate py3818
cd /path/to/BoneSimulation/src

# Default: full pipeline on data/bigdataset
python main.py
```

> **Important**: The working directory must be `src/` because the pipeline uses relative paths like `data/bigdataset`, `pictures/`, and `calculix/`.

---

## Command-Line Arguments

```
python main.py [--test MODE]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--test` | `str` | `none` | Pipeline execution mode (see below) |

---

## Run Modes

### `none` (Default — full pipeline)

```bash
python main.py
```

Runs every stage on the full `data/bigdataset`:
1. Load all TIFF slices in chunks of 50
2. Pre-process (blur + binarise + close + interpolate)
3. Find the largest bone cluster
4. Generate surface mesh (VTK)
5. Generate tetrahedral mesh (VTK)
6. Extract cubic blocks with per-block tetra meshes
7. Export CalculiX `.inp` file (prompts for solver name)
8. Execute the CalculiX simulation
9. Generate PDF analysis report

### `small` / `fast_debug` — mini-volume test

```bash
python main.py --test small
```

Reduces `CHUNK_SIZE` to 5 so that only a small portion of slices is loaded. Useful for quickly verifying that the pipeline runs end-to-end without waiting hours for a full dataset.

### `dry` — no files written

```bash
python main.py --test dry
```

Runs all computations in memory but **skips every file write**, including:
- Cluster TIFF
- Surface mesh VTK
- Tetrahedral mesh VTK
- Block VTI and tetra files
- CalculiX `.inp` and run script
- PDF report

Use this mode to:
- Verify module imports and pipeline logic
- Profile memory usage without I/O overhead
- Run CI/CD checks quickly

### `complete` — full pipeline with verification

```bash
python main.py --test complete
```

Same as the default mode but logs extra verification messages at each stage. Intended for integration testing where you need confirmation that every output file was created correctly.

---

## Customising Parameters

The main configuration block is at the top of `src/main.py`. Edit these values before running.

### Chunk size

```python
CHUNK_SIZE = 50  # Number of TIFF slices per processing chunk
```

Reduce this if you run into memory errors. Increase it for faster I/O on systems with abundant RAM.

### Memory limit

```python
MEMORY_LIMIT_PERCENT = 80  # Pause pipeline if RAM usage exceeds this
```

Adjust based on your system and whether other processes are running concurrently.

### Output directories

```python
OUTPUT_DIR_MESHES    = "pictures"   # VTK meshes and TIFF cluster exports
PDF_REPORT_DIR       = "report"     # PDF analysis reports
CALCULIX_REPORT_DIR  = "calculix"   # .inp files and run scripts
LOGS_DIR             = "logs"       # Fault logs
```

### Dataset paths

```python
DATASET_BIG   = "data/bigdataset"   # Full CT dataset
DATASET_SMALL = "data/dataset"      # Small test dataset
```

The pipeline uses `DATASET_BIG` when the `directory` argument passed to `process_and_visualize` is `"data"`.

### Cropping (currently disabled)

```python
cropping_settings = {
    "z": (3, 0),
    "y": (50, 50),
    "x": (15, 35),
}
```

To enable cropping, uncomment the two lines in `process_and_visualize`:

```python
interpolated_stack = crop_stack_custom(interpolated_stack, cropping_settings)
logger.info(f"Stack cropped with settings: {cropping_settings}")
```

### Material properties (CalculiX)

```python
material_props = {'E': 18000.0, 'nu': 0.3}
```

- `E`: Young's modulus in MPa (18 000 MPa is typical for cortical bone)
- `nu`: Poisson's ratio (0.3 is standard for bone tissue)

### Block extraction size

In `src/block_extraction.py`, `improved_block_extraction_integration`:

```python
block_size_mm   = 30.0              # Physical block size in millimetres
voxel_spacing   = (0.05, 0.05, 0.05)  # Voxel size in mm (x, y, z)
```

Block size in voxels is computed as `int(block_size_mm / voxel_spacing[0])`.

---

## Typical Workflows

### Workflow 1 — First run on new CT data

```bash
# 1. Put your TIFF slices into data/bigdataset/
# 2. Dry run to check everything loads
python main.py --test dry

# 3. Small test to verify output quality
python main.py --test small

# 4. Full pipeline
python main.py
```

### Workflow 2 — Iterative parameter tuning

```bash
# Run dry to check memory behaviour
python main.py --test dry

# Edit cropping_settings in main.py to remove artefacts
# Re-run full pipeline
python main.py
```

### Workflow 3 — Only CalculiX re-export

If you already have a tetrahedral mesh `.vtk` and want to re-export the `.inp` with different material properties:

```python
# From a Python script or notebook
import meshio
from calculix_export import export_to_calculix

mesh = meshio.read("pictures/tetramesh_<timestamp>.vtk")
material_props = {"E": 20000.0, "nu": 0.28}
export_to_calculix(
    mesh.points,
    mesh.cells_dict["tetra"],
    material_props,
    "calculix/custom_bone_sim.inp"
)
```

### Workflow 4 — XGBoost hyperparameter optimisation

```bash
conda activate py3818
cd src

# Edit the volume_path at the bottom of optimize_xgb_pipeline.py
python optimize_xgb_pipeline.py
```

This will run 50 Optuna trials (configurable via `n_trials`) and print the best hyperparameters.

---

## Monitoring Progress

### Real-time log

The pipeline writes timestamped INFO / WARNING / ERROR messages to both stdout and `logfile.log`:

```bash
# Follow the log in real time
tail -f logfile.log
```

### Fault log

If the Python interpreter crashes (segfault, out-of-memory kill), the fault handler writes a stack trace to `logs/fault_log.txt`.

### Memory monitoring

While the pipeline is running, use a separate terminal:

```bash
watch -n 2 "free -h && ps aux --sort=-%mem | head -5"
```

---

## Standalone Scripts

### 3-D Mayavi visualisation

```bash
cd additional_code
python three_d_vis.py
```

Renders the raw CT dataset as a 3-D contour plot using Mayavi. Edit `DIRECTORY` inside the script to point to your dataset path.

### 2-D Matplotlib slice viewer

```bash
cd additional_code
python two_d_vis.py
```

Displays 2-D slices of the dataset.

### TIFF utilities

Located in `src/utils/`:

| Script | Purpose |
|---|---|
| `merge_tif_files.py` | Merges multiple single-page TIFF files into one multi-page stack |
| `splitt_tiff.py` | Splits a multi-page TIFF stack into individual slice files |
