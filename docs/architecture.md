# Architecture & Pipeline Overview

This document describes the full end-to-end architecture of the BoneSimulation platform, covering data flow, module responsibilities, memory management strategies, and key design decisions.

---

## Table of Contents

1. [High-Level Data Flow](#high-level-data-flow)
2. [Processing Stages](#processing-stages)
3. [Memory Management](#memory-management)
4. [Directory Layout](#directory-layout)
5. [Concurrency Model](#concurrency-model)
6. [Design Decisions](#design-decisions)

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         data/bigdataset/                            │
│          (hundreds of micro-CT TIFF slices, grayscale)              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  load_images_in_chunks()
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CHUNK-BASED IMAGE LOADING  (image_loading.py)          │
│  • Sorts files alphanumerically                                     │
│  • Loads N slices at a time via multiprocessing.Pool                │
│  • Yields numpy arrays (uint8, shape: [chunk, H, W])                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│          IMAGE PRE-PROCESSING  (image_processing.py)                │
│  1. Otsu thresholding on a random 1 M-pixel sub-sample              │
│  2. Gaussian blur (σ=1) for noise suppression                       │
│  3. Binarisation: pixel > threshold → 1 (bone), else 0              │
│  4. Morphological closing (ball r=2) to seal micro-gaps             │
│  5. Spline interpolation: downscale by 0.5× (memory reduction)      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  All chunks concatenated → full 3-D volume
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│          CLUSTER ANALYSIS  (cluster_analysis.py)                    │
│  • Connected-component labelling (skimage.measure.label)            │
│  • Identifies and extracts the largest bone cluster                 │
│  • Memory-efficient path for volumes > 100 M voxels                 │
│  • Saves cluster as multi-page TIFF                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────────┐   ┌───────────────────────────────────────┐
│  SURFACE MESH           │   │  TETRAHEDRAL MESH                     │
│  (mesh_generation.py)   │   │  (mesh_generation.py)                 │
│  • Marching Cubes       │   │  • Binary volume inversion            │
│  • Chunk-aware for      │   │  • Optional downsampling for large    │
│    large volumes        │   │    volumes (> 100 M voxels)           │
│  • Saved as .vtk        │   │  • CGAL meshing via ciclope           │
│                         │   │  • Saved as .vtk                      │
└─────────────────────────┘   └────────────────┬──────────────────────┘
                                               │
                              ┌────────────────┴──────────────────────┐
                              ▼                                        ▼
              ┌───────────────────────────┐         ┌─────────────────────────────┐
              │  BLOCK EXTRACTION         │         │  CALCULIX EXPORT            │
              │  (block_extraction.py)    │         │  (calculix_export.py)       │
              │  • Raster scan of volume  │         │  • Writes .inp FEA file     │
              │  • Cubic sub-volumes      │         │  • Nodes, C3D4 elements     │
              │  • Per-block VTI + tetra  │         │  • Material: Bone           │
              │    mesh generation        │         │  • Auto boundary conditions │
              └───────────────────────────┘         │  • Auto load (z-max plane)  │
                                                    │  • Generates run_sim.sh     │
                                                    └─────────────┬───────────────┘
                                                                  │
                                                                  ▼
                                                    ┌─────────────────────────────┐
                                                    │  REPORTING  (reporting.py)  │
                                                    │  • Porosity calculation     │
                                                    │  • Cluster size stats       │
                                                    │  • PDF histogram report     │
                                                    └─────────────────────────────┘
```

---

## Processing Stages

### Stage 1 — Image Loading

**Module**: `src/image_loading.py`

The loader scans the dataset directory for `.tif` / `.tiff` files, sorts them alphanumerically (preserving slice order), and yields batches of size `CHUNK_SIZE` (default: 50) as NumPy arrays of shape `(chunk_size, H, W)`.

Parallelism is achieved via `multiprocessing.Pool.map`, so each CPU core decodes a different slice simultaneously.

### Stage 2 — Image Pre-processing

**Module**: `src/image_processing.py`

Each chunk passes through four steps:

| Step | Function | Notes |
|---|---|---|
| Otsu thresholding | `process_images_globally` | Sub-sampled at 1 M pixels to save memory |
| Gaussian blur | `process_images_globally` | σ=1, `preserve_range=True` |
| Morphological closing | `apply_morphological_closing` | 3-D ball (r=2) for small volumes; 2-D disk (r=2) per slice for large ones |
| Spatial interpolation | `interpolate_image_stack` | `scipy.ndimage.zoom`, scaling_factor=0.5, order=1 |

After processing, chunks are appended to a list and concatenated into a single 3-D array.

### Stage 3 — Cluster Analysis

**Module**: `src/cluster_analysis.py`

The binary 3-D volume is passed to `skimage.measure.label`. The largest foreground region (highest voxel count excluding background label 0) is extracted as a boolean mask. This isolates the primary trabecular bone structure from noise and debris.

For volumes exceeding 100 million voxels, a memory-efficient variant using `numpy.unique` instead of `numpy.bincount` is used to avoid peak memory spikes.

### Stage 4 — Mesh Generation

**Module**: `src/mesh_generation.py`

Two mesh types are generated from the largest cluster:

- **Surface mesh** (`marching_cubes`): Uses `skimage.measure.marching_cubes` with optional chunk-based processing and z-offset stitching. Output: triangular `.vtk`.
- **Tetrahedral mesh** (`generate_tetrahedral_mesh`): Binary volume is inverted (bone → void, void → bone for CGAL), optionally downsampled, then meshed with `ciclope.core.tetraFE.cgal_mesh`. Output: volumetric `.vtk`.

### Stage 5 — Block Extraction

**Module**: `src/block_extraction.py`

The full bone volume is subdivided into a regular 3-D raster of cubic blocks (default: 30 mm × 30 mm × 30 mm at 0.05 mm/voxel = 600 voxels per side). Each non-empty block is:

1. Saved as a VTI (VTK ImageData) file
2. Pre-processed independently (Gaussian → Otsu → closing → largest component)
3. Tetrahedral-meshed if it contains at least 100 active voxels

Blocks that are too small, empty, or fail meshing are skipped with a warning.

### Stage 6 — CalculiX Export

**Module**: `src/calculix_export.py`

The tetrahedral mesh is written as a CalculiX `.inp` file containing:

- `*NODE` section (all mesh vertices)
- `*ELEMENT, TYPE=C3D4` section (all tetrahedral elements)
- `*MATERIAL, NAME=Bone` with Young's modulus E = 18 000 MPa and Poisson's ratio ν = 0.3
- Automatically detected fixed nodes (z = z_min plane) as `*BOUNDARY`
- Automatically detected loaded nodes (z = z_max plane) as `*CLOAD` (100 N distributed)
- A static analysis `*STEP` / `*STATIC` / `*END STEP` block

A companion bash script `run_simulation.sh` is written that sets `OMP_NUM_THREADS` from `lscpu` and invokes the chosen CalculiX binary.

### Stage 7 — Reporting

**Module**: `src/reporting.py`

A PDF report is generated using `fpdf2` containing:
- Total cluster count
- Porosity (fraction of empty voxels)
- Max, min, and mean cluster sizes
- Cluster-size histogram (Matplotlib figure embedded as PNG)

---

## Memory Management

The pipeline is designed to operate on datasets that exceed available RAM. Key strategies:

| Strategy | Location | Effect |
|---|---|---|
| Chunk-based loading | `image_loading.py` | Only `CHUNK_SIZE` slices in RAM at once |
| `gc.collect()` after each chunk | `main.py` | Forces garbage collection between stages |
| `wait_for_memory()` guard | `main.py` | Pauses pipeline if RAM usage > 80% |
| Otsu sub-sampling | `image_processing.py` | Threshold computed on ≤ 1 M pixels |
| Morphological closing fallback | `image_processing.py` | 2-D per-slice closing for stacks > 100 slices |
| Chunked interpolation | `image_processing.py` | `scipy.ndimage.zoom` applied in z-chunks |
| Downsampling before tetra meshing | `mesh_generation.py` | Volume capped at 100 M voxels |
| `uint8` dtype | `image_processing.py`, `file_io.py` | 8× smaller than float64 |

---

## Directory Layout

```
BoneSimulation/
├── src/
│   ├── main.py                   ← Pipeline entry point
│   ├── image_loading.py
│   ├── image_processing.py
│   ├── cluster_analysis.py
│   ├── mesh_generation.py
│   ├── block_extraction.py
│   ├── calculix_export.py
│   ├── file_io.py
│   ├── reporting.py
│   ├── visualization.py
│   ├── optimize_xgb_pipeline.py
│   └── utils/
│       ├── utils.py
│       ├── merge_tif_files.py
│       └── splitt_tiff.py
├── data/
│   ├── bigdataset/               ← Primary CT dataset (TIFF slices)
│   └── dataset/                  ← Smaller test dataset
├── docs/                         ← This documentation
├── calculix/                     ← FEA input files and run scripts
├── pictures/                     ← Mesh and cluster output files
├── report/                       ← PDF analysis reports
├── logs/                         ← Fault logs
├── additional_code/              ← Standalone visualisation utilities
├── bibliography/                 ← Referenced literature
├── environment.yml
├── requirements.txt
└── installer.sh
```

---

## Concurrency Model

| Component | Mechanism | Benefit |
|---|---|---|
| TIFF loading | `multiprocessing.Pool.map` | CPU-parallel image decoding |
| Feature extraction (XGBoost pipeline) | `ProcessPoolExecutor` | CPU-parallel block feature computation |
| CalculiX solver | `OMP_NUM_THREADS` (set to max logical CPUs) | Multi-threaded FEA solver |
| GPU acceleration (optional) | `cupy` (XGBoost `gpu_hist`) | GPU-accelerated gradient boosting |

---

## Design Decisions

**Why chunk-based processing?**
Micro-CT datasets can contain thousands of high-resolution slices totalling tens of gigabytes. Processing the entire stack at once would exceed typical server RAM limits. The chunk-based approach trades CPU time (multiple passes) for a predictable memory footprint.

**Why Otsu thresholding?**
Otsu's method automatically finds the optimal global threshold separating bone (bright) from background (dark) without requiring manual calibration per scan. Sub-sampling the pixel pool to 1 M elements preserves statistical accuracy while reducing memory overhead.

**Why CGAL for tetrahedral meshing?**
CGAL provides guaranteed-quality Delaunay tetrahedralisations that satisfy user-specified facet-distance and circumradius bounds, which is critical for numerically stable finite-element analysis.

**Why CalculiX?**
CalculiX is an open-source FEA solver with an Abaqus-compatible input format. Its OpenMP parallelism makes it suitable for large meshes on standard Linux hardware without requiring commercial licences.
