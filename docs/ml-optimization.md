# ML Optimisation Pipeline

This document describes the machine-learning optimisation pipeline in `src/optimize_xgb_pipeline.py`, which uses XGBoost and Optuna to find the best morphological feature-engineering parameters for predicting bone structural properties.

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Engineering](#feature-engineering)
3. [Label Definition](#label-definition)
4. [XGBoost Model](#xgboost-model)
5. [Bayesian Optimisation with Optuna](#bayesian-optimisation-with-optuna)
6. [GPU Acceleration](#gpu-acceleration)
7. [Running the Optimisation](#running-the-optimisation)
8. [Interpreting Results](#interpreting-results)
9. [Extending the Pipeline](#extending-the-pipeline)

---

## Overview

The ML optimisation pipeline treats the **feature-engineering parameters** (threshold, kernel size, scaling) as hyperparameters alongside the **XGBoost training parameters** (learning rate, tree depth, number of estimators). Optuna's Tree-structured Parzen Estimator (TPE) searches this combined space to maximise 5-fold cross-validation score (negative MSE).

```
CT Volume (TIFF)
      │
      ▼
Block Extraction
      │  (non-overlapping sub-volumes, min 1 000 active voxels)
      ▼
Feature Engineering  ◄────── Optuna trial parameters
      │  (threshold, kernel_size, scaling)              │
      │  → porosity                                     │
      │  → num_clusters                                 │
      │  → largest_cluster_size                         │
      │  → surface_area                                 │
      │  → scaled_porosity                              │
      │  → kernel_size                                  │
      ▼                                                 │
XGBoost Regression   ◄────── Optuna trial parameters   │
      │  (learning_rate, max_depth, n_estimators)       │
      │                                                 │
      ▼                                                 │
5-Fold CV Score  ─────────────────────────────────────►┘
(maximise neg. MSE)
```

---

## Feature Engineering

Each 3-D block goes through the following steps to produce a 6-element feature vector:

### Step 1 — Thresholding

```python
bin_vol = (volume > threshold).astype(np.uint8)
```

`threshold` is a trial parameter in [0.1, 0.9].

### Step 2 — Morphological closing

```python
closed_vol = binary_closing(bin_vol,
    structure=np.ones((kernel_size, kernel_size, kernel_size)))
```

`kernel_size` is a trial parameter in {2, 3, …, 8}. Larger kernels fill bigger gaps but can merge separate trabeculae.

### Step 3 — Feature extraction

| Feature index | Name | Computation |
|---|---|---|
| 0 | `porosity` | `1 − mean(closed_vol)` |
| 1 | `num_clusters` | `scipy.ndimage.label` result |
| 2 | `largest_cluster_size` | `max(bincount(labels)[1:])` |
| 3 | `surface_area` | Sum of triangle areas from Marching Cubes |
| 4 | `scaled_feature` | `scaling × porosity` |
| 5 | `kernel_size` | The kernel size value itself |

`scaling` is a trial parameter in [0.3, 1.5] that acts as a weight on the porosity feature.

#### Surface area computation

Triangle areas are computed using a Numba-JIT-compiled function for performance:

```python
@njit
def fast_triangle_area(v0, v1, v2):
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.sqrt(np.dot(cross, cross))
```

---

## Label Definition

The regression target is a composite score that captures both the global porosity and the local structural complexity of each block:

```python
label = porosity × 1000 + largest_cluster_size × 0.01
```

The weights (1 000 and 0.01) are chosen to bring both terms into a similar numerical range. This is not a physically derived quantity — it is a proxy for the structural quality of the block, which XGBoost learns to predict from the feature vector.

> **Note**: This label definition can be replaced with physically meaningful targets such as apparent Young's modulus (from micro-FEA) or bone volume fraction (BV/TV) if ground-truth data is available.

---

## XGBoost Model

```python
XGBRegressor(
    tree_method  = 'gpu_hist' if GPU_AVAILABLE else 'hist',
    predictor    = 'gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
    learning_rate = ...,   # trial parameter: [0.01, 0.3], log scale
    max_depth     = ...,   # trial parameter: {3, ..., 10}
    n_estimators  = ...,   # trial parameter: {50, ..., 200}
    verbosity     = 0,
)
```

### Cross-validation

5-fold cross-validation with shuffling:

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, features, y_labels,
                         scoring="neg_mean_squared_error", cv=cv)
```

The Optuna objective returns `np.mean(scores)` (a negative number; higher = better).

---

## Bayesian Optimisation with Optuna

Optuna uses the **Tree-structured Parzen Estimator (TPE)** algorithm by default. TPE models the distribution of good and bad hyperparameter configurations separately and samples from the promising region.

### Search space

| Parameter | Type | Range |
|---|---|---|
| `threshold` | `float` | [0.1, 0.9] |
| `kernel_size` | `int` | [2, 8] |
| `scaling` | `float` | [0.3, 1.5] |
| `learning_rate` | `float` (log) | [0.01, 0.3] |
| `max_depth` | `int` | [3, 10] |
| `n_estimators` | `int` | [50, 200] |

### Objective direction

```python
study = optuna.create_study(direction="maximize")
```

We maximise `neg_mean_squared_error` (i.e., minimise MSE).

---

## GPU Acceleration

If `cupy` is installed, the pipeline automatically uses:
- `cupy` arrays for thresholding and porosity computation
- XGBoost's `gpu_hist` tree method
- XGBoost's `gpu_predictor`

The GPU path is selected here:

```python
extractor = extract_features_gpu if (use_gpu and GPU_AVAILABLE) else extract_features_cpu
```

`GPU_AVAILABLE` is set at module load:

```python
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
```

The pipeline falls back to CPU silently if `cupy` is not installed.

---

## Running the Optimisation

### 1. Prepare a TIFF volume

You need a 3-D TIFF file containing the processed bone volume (e.g., the largest-cluster TIFF output from the main pipeline):

```bash
# Use the output of the main pipeline
pictures/largest_cluster_<timestamp>.tif
```

### 2. Edit the script

Open `src/optimize_xgb_pipeline.py` and update the `__main__` block:

```python
if __name__ == "__main__":
    volume_path = "pictures/largest_cluster_<timestamp>.tif"
    block_shape = (128, 256, 256)   # adjust to your volume size
    n_trials = 50                   # increase for better results
```

### 3. Run

```bash
conda activate py3818
cd src
python optimize_xgb_pipeline.py
```

### 4. Output

```
Best parameters:
{
  'threshold': 0.452,
  'kernel_size': 4,
  'scaling': 0.871,
  'learning_rate': 0.043,
  'max_depth': 7,
  'n_estimators': 132
}
Best score (neg. MSE): -12345.67
```

---

## Interpreting Results

| Parameter | Effect if high | Effect if low |
|---|---|---|
| `threshold` | More aggressive binarisation — only very bright voxels are bone | Includes more grey-zone voxels as bone |
| `kernel_size` | Wider closing — merges nearby structures | Fine-grained closing — preserves thin trabeculae |
| `scaling` | Higher weight on porosity in the feature vector | Lower weight on porosity |
| `learning_rate` | Faster convergence, risk of overfitting | Slower convergence, more regularisation |
| `max_depth` | More complex trees, risk of overfitting | Simpler trees, more bias |
| `n_estimators` | More boosting rounds — better for low learning rates | Fewer rounds — faster but potentially weaker |

### Score interpretation

A typical score of `-1e4` to `-1e6` (neg. MSE) is expected, depending on the volume size and label scale. The absolute value matters less than the **relative improvement** between trials.

---

## Extending the Pipeline

### Using physically meaningful labels

Replace the composite label with CT-derived bone volume fraction (BV/TV):

```python
def compute_bvtv(block):
    return block.mean()   # BV/TV if binary volume

labels = [compute_bvtv((b > 0.5).astype(np.uint8)) for b in blocks]
```

### Adding more features

Extend `extract_features_cpu` to include:
- **Trabecular thickness** (Tb.Th): mean distance transform within bone voxels
- **Connectivity density** (Conn.D): number of connected components per unit volume
- **Anisotropy** (DA): eigenvalue ratio of the mean intercept length tensor

### Saving the trained model

After the study completes, re-train the best model and save it:

```python
import joblib

best = study.best_params
features = extract_features_parallel(X_blocks,
    threshold=best["threshold"],
    kernel_size=best["kernel_size"],
    scaling=best["scaling"])

model = XGBRegressor(
    learning_rate=best["learning_rate"],
    max_depth=best["max_depth"],
    n_estimators=best["n_estimators"]
)
model.fit(features, y_labels)
joblib.dump(model, "models/bone_xgb_model.pkl")
```

### Using Optuna dashboard

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_study.db
```

To persist the study to SQLite, create it with a storage argument:

```python
study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///optuna_study.db",
    study_name="bone_xgb"
)
```
