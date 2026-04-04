# Module Reference

This document provides a detailed reference for every source module in the `src/` directory.

---

## Table of Contents

- [`main.py`](#mainpy) — Pipeline entry point
- [`image_loading.py`](#image_loadingpy) — Parallel TIFF loading
- [`image_processing.py`](#image_processingpy) — Filtering, thresholding, interpolation
- [`cluster_analysis.py`](#cluster_analysispy) — Connected-component labelling
- [`mesh_generation.py`](#mesh_generationpy) — Surface and tetrahedral meshing
- [`block_extraction.py`](#block_extractionpy) — Raster block extraction
- [`calculix_export.py`](#calculix_exportpy) — CalculiX FEA export
- [`file_io.py`](#file_iopy) — TIFF stream I/O
- [`reporting.py`](#reportingpy) — PDF report generation
- [`visualization.py`](#visualizationpy) — Matplotlib plots
- [`optimize_xgb_pipeline.py`](#optimize_xgb_pipelinepy) — XGBoost / Optuna optimisation
- [`utils/utils.py`](#utilsutilspy) — Shared utilities

---

## `main.py`

**Purpose**: Orchestrates the entire BoneSimulation pipeline. Parses command-line arguments, calls each stage in order, manages memory guards, and handles logging.

### Configuration constants

| Constant | Default | Description |
|---|---|---|
| `MEMORY_LIMIT_PERCENT` | `80` | RAM usage percentage that triggers a wait |
| `CHUNK_SIZE` | `50` | Number of TIFF slices loaded per chunk |
| `OUTPUT_DIR_MESHES` | `"pictures"` | Output directory for VTK/TIFF files |
| `PDF_REPORT_DIR` | `"report"` | Output directory for PDF reports |
| `CALCULIX_REPORT_DIR` | `"calculix"` | Output directory for `.inp` and run scripts |
| `LOGS_DIR` | `"logs"` | Directory for fault logs |
| `DATASET_BIG` | `"data/bigdataset"` | Path to the full CT dataset |

### Functions

#### `process_and_visualize(directory, test_mode="none")`

Main pipeline function. Executes all stages sequentially with memory guards between each stage.

**Parameters**:
- `directory` (`str`): Base data directory (usually `"data"`).
- `test_mode` (`str`): One of `"none"`, `"small"`, `"dry"`, `"fast_debug"`, `"complete"`.

#### `generate_run_script(input_inp_path, preferred_solver, output_dir)`

Writes a bash script that sets `OMP_NUM_THREADS` and invokes the chosen CalculiX binary on the given `.inp` file.

#### `get_max_threads()`

Reads `lscpu` output to compute the total number of logical CPU threads (`cores × threads_per_core`).

#### `check_memory() → bool`

Returns `True` if current RAM usage exceeds `MEMORY_LIMIT_PERCENT`.

#### `wait_for_memory()`

Blocks execution (with `gc.collect()` and `time.sleep(10)` loops) until RAM usage drops below the limit.

#### `ensure_output_dir(dir_path)`

Creates `dir_path` and all parent directories if they do not already exist.

### CLI

```
python src/main.py [--test {none,dry,small,fast_debug,complete}]
```

---

## `image_loading.py`

**Purpose**: Loads micro-CT TIFF slices from disk into NumPy arrays, either all at once or as a generator of chunks.

### Functions

#### `load_image(filepath) → np.ndarray | None`

Opens a single TIFF file as an 8-bit grayscale array. Returns `None` on failure.

**Parameters**:
- `filepath` (`str`): Absolute or relative path to a TIFF file.

**Returns**: `np.ndarray` with shape `(H, W)` and dtype `uint8`, or `None`.

#### `load_images(directory) → np.ndarray`

Loads all `.tif` files in `directory` in sorted order using `multiprocessing.Pool`. Returns a single 3-D array of shape `(N, H, W)`.

**Raises**: `ValueError` if no files are found or all loads fail.

#### `load_images_in_chunks(directory, chunk_size=50) → Generator[np.ndarray]`

Generator version of `load_images`. Yields chunks of shape `(chunk_size, H, W)` one at a time, allowing the caller to process data without loading the full volume into RAM.

**Parameters**:
- `directory` (`str`): Directory containing `.tif` / `.tiff` files.
- `chunk_size` (`int`): Number of slices per yielded chunk.

**Yields**: `np.ndarray` of shape `(chunk_size_actual, H, W)`.

---

## `image_processing.py`

**Purpose**: Applies pre-processing operations to 3-D image chunks: thresholding, smoothing, morphological closing, spatial interpolation, and optional cropping.

### Functions

#### `process_images_globally(data_array) → tuple[np.ndarray, np.ndarray, float]`

Computes an Otsu threshold on a random sub-sample of up to 1 million pixels, applies Gaussian smoothing (σ=1), and binarises the array.

**Returns**: `(blurred, binary, threshold)` where `blurred` and `binary` have the same shape as `data_array`.

#### `apply_morphological_closing(binary_images, ball_radius=2) → np.ndarray`

Closes small gaps in the binary volume.
- For stacks with ≤ 100 slices: applies 3-D ball closing.
- For larger stacks: applies 2-D disk closing slice-by-slice to limit peak memory.

#### `interpolate_image_stack(image_stack, scaling_factor=1.0, chunk_size=20) → np.ndarray`

Rescales the 3-D volume using `scipy.ndimage.zoom` with linear interpolation (order=1) applied in z-chunks to avoid large intermediate allocations.

**Returns**: Rescaled binary `uint8` array.

#### `crop_stack_custom(stack, crop: dict) → np.ndarray`

Removes a specified number of voxels from each face of the bounding box.

**Parameters**:
```python
crop = {
    "z": (top_slices, bottom_slices),
    "y": (front_rows, back_rows),
    "x": (left_cols, right_cols),
}
```

#### `save_blocks_as_npy(blocks, output_dir, timestamp)`

Saves each element of `blocks` as a separate `.npy` file.

---

## `cluster_analysis.py`

**Purpose**: Identifies distinct bone clusters in the binary 3-D volume using connected-component labelling and extracts the largest one.

### Functions

#### `find_largest_cluster(binary_image_stack, connectivity=1) → tuple`

Entry point for cluster analysis. Delegates to the memory-efficient variant for volumes > 100 M voxels.

**Parameters**:
- `binary_image_stack` (`np.ndarray`): 3-D binary array.
- `connectivity` (`int`): 1 = face connectivity, 2 = face+edge, 3 = face+edge+corner.

**Returns**: `(largest_cluster, num_clusters, size_of_largest_cluster)`
- `largest_cluster`: Boolean array of same shape, `True` where the largest cluster is.
- `num_clusters`: Total number of connected components found.
- `size_of_largest_cluster`: Voxel count of the largest component.

Returns `(None, 0, 0)` for empty or all-zero inputs.

#### `_find_largest_cluster_memory_efficient(binary_volume, connectivity=1) → tuple`

Internal variant using `numpy.unique` + `numpy.bincount` to minimise peak memory. Same return signature as `find_largest_cluster`.

---

## `mesh_generation.py`

**Purpose**: Generates surface (triangular) and volumetric (tetrahedral) meshes from binary 3-D arrays.

### Functions

#### `marching_cubes(binary_stack, spacing=(1,1,1), chunk_size=None) → tuple[np.ndarray, np.ndarray]`

Runs the Marching Cubes algorithm on the binary volume.
- For small volumes: uses `skimage.measure.marching_cubes` directly.
- For large volumes (when `chunk_size` is set): delegates to `_chunked_marching_cubes`.

**Returns**: `(verts, faces)` or `(None, None)` on failure.

#### `_chunked_marching_cubes(binary_stack, spacing, chunk_size=50, overlap=5) → tuple`

Processes the volume in overlapping z-chunks and stitches vertices and face indices across chunk boundaries.

#### `save_mesh_as_vtk(verts, faces, output_filename)`

Writes a triangular mesh to a VTK file using `meshio`.

#### `generate_tetrahedral_mesh(binary_volume, voxel_size, output_filename, use_downsampling=True, max_volume_size=100_000_000) → mesh | None`

Generates a high-quality tetrahedral mesh via CGAL (through `ciclope.core.tetraFE.cgal_mesh`).
- Automatically detects and corrects inverted binary volumes.
- Optionally downsamples large volumes with `scipy.ndimage.zoom`.

**Returns**: The mesh object (with a `.write()` method) or `None` on failure.

#### `reverse_binary(volume) → np.ndarray`

Inverts a binary volume: `1 → 0`, `0 → 1`. Used to prepare the volume for CGAL meshing (which expects the material region to be the interior void).

#### `extract_blocks(volume, block_size_voxels, step_size_voxels) → list[tuple]`

Slides a window over the volume and returns non-empty blocks as `(index, block_array)` tuples.

#### `save_blocks_as_vtk(blocks, voxel_spacing, output_dir, timestamp)`

Saves each block as a VTK `UniformGrid` file using PyVista.

---

## `block_extraction.py`

**Purpose**: Systematically extracts and meshes cubic sub-volumes (blocks) from the full bone volume.

### Functions

#### `numpy2vti(array, output_filename)`

Converts a 3-D NumPy array to a VTK ImageData (`.vti`) file using the VTK Python API.

**Raises**: `ValueError` if the array is not 3-D; `IOError` if the file cannot be written.

#### `debug_volume_info(volume, block_size_mm=30.0, voxel_size_mm=0.05)`

Prints diagnostic information about how many blocks can be extracted from the volume at the given physical size, and suggests smaller block sizes if zero blocks are possible.

#### `extract_blocks_raster(volume, block_size_voxels, output_dir, voxel_spacing=(0.05,0.05,0.05), write_tetra_mesh=True, min_voxels_threshold=100) → tuple[int, int]`

Main block extraction function. Iterates over a 3-D raster grid, saves each non-empty block as VTI, pre-processes it (Gaussian → Otsu → closing → largest component), and optionally generates a tetrahedral mesh.

**Returns**: `(total_blocks, successful_tetrahedral_meshes)`

#### `improved_block_extraction_integration(largest_cluster, timestamp, test_mode, output_base_dir="pictures") → tuple`

Wrapper that:
1. Calls `debug_volume_info` for diagnostics.
2. Computes the block size in voxels from physical parameters (default: 30 mm at 0.05 mm/voxel).
3. Automatically falls back to a smaller block size if the volume is too small.
4. Calls `extract_blocks_raster` (skipped in `dry` mode).

**Returns**: `(blocks_output_dir, num_blocks, successful_meshes)`

---

## `calculix_export.py`

**Purpose**: Writes CalculiX `.inp` FEA input files from tetrahedral mesh data.

### Functions

#### `export_to_calculix(mesh_points, mesh_elements, material_props, output_path, fix_z_plane=True, load_z_plane=True, load_value=100.0)`

Writes a complete CalculiX input file.

**Parameters**:
- `mesh_points` (`np.ndarray`): Shape `(N, 3)` — node coordinates.
- `mesh_elements` (`np.ndarray`): Shape `(M, 4)` — C3D4 tetrahedron node indices (1-based).
- `material_props` (`dict`): `{"E": float, "nu": float}` — Young's modulus (MPa) and Poisson's ratio.
- `output_path` (`str`): Path to write the `.inp` file.
- `fix_z_plane` (`bool`): If `True`, all nodes at `z = z_min` are fully fixed (`DOF 1–3 = 0`).
- `load_z_plane` (`bool`): If `True`, a distributed force of `load_value` N is applied at `z = z_max`.
- `load_value` (`float`): Total force (N) distributed equally over all loaded nodes.

**Output format excerpt**:
```
*HEADING
Knochen-Simulation
*NODE
1, x, y, z
...
*ELEMENT, TYPE=C3D4
1, n1, n2, n3, n4
...
*MATERIAL, NAME=Bone
*ELASTIC
18000.0, 0.3
*SOLID SECTION, ELSET=AllElements, MATERIAL=Bone
...
*BOUNDARY
node_id, 1, 3, 0.0
...
*CLOAD
node_id, 3, force_per_node
...
*STEP
*STATIC
*END STEP
```

---

## `file_io.py`

**Purpose**: Low-level TIFF file I/O helpers using `tifffile`.

### Functions

#### `load_tiff_stream(filepath) → np.ndarray | None`

Loads a multi-page TIFF file into a single 3-D NumPy array.

#### `save_tiff_in_chunks(image_stack, filename, chunk_size=100)`

Writes a 3-D array as a BigTIFF file in chunks to avoid loading the entire array into a `TiffWriter` buffer.

#### `load_tiff_in_chunks(filepath, chunk_size=100) → Generator[np.ndarray]`

Generator that reads a multi-page TIFF file `chunk_size` pages at a time, suitable for streaming very large stacks.

#### `save_largest_cluster_as_tiff(cluster_array, output_path)`

Saves a binary 3-D array as a multi-page TIFF using ImageJ-compatible metadata.

**Raises**: `ValueError` if the input is not a 3-D NumPy array.

---

## `reporting.py`

**Purpose**: Generates PDF analysis reports from cluster statistics.

### Functions

#### `generate_report(cluster_sizes, porosity, output_pdf_path)`

Creates a PDF containing:
- Title page with "Knochen Analyse Report"
- Number of clusters
- Porosity percentage
- Max / min / mean cluster size in voxels
- Cluster-size histogram (Matplotlib figure)

Uses `fpdf2` for PDF generation. The histogram image is saved temporarily as a PNG and embedded, then deleted.

#### `extract_statistics_from_volume(binary_volume) → tuple`

Computes cluster statistics from a binary 3-D array using 26-connectivity labelling (`scipy.ndimage.label`).

**Returns**: `(cluster_sizes, porosity, num_clusters)`
- `cluster_sizes`: List of voxel counts for each cluster.
- `porosity`: Fraction `(total_voxels − bone_voxels) / total_voxels`.
- `num_clusters`: Integer count of distinct clusters.

---

## `visualization.py`

**Purpose**: Simple Matplotlib visualisations for debugging and exploration.

### Functions

#### `plot_histogram(image_array, threshold=None)`

Displays a histogram of all pixel values (flattened to 1-D, cast to `uint8`). Optionally overlays a red vertical line at the Otsu threshold.

#### `plot_images(images, title)`

Displays up to 10 slices from a 3-D array in a horizontal row using `imshow` with a grey colormap.

---

## `optimize_xgb_pipeline.py`

**Purpose**: Bayesian hyperparameter optimisation for a morphological feature extraction → XGBoost regression pipeline using Optuna.

### Functions

#### `fast_triangle_area(v0, v1, v2) → float`

Numba-JIT compiled function that computes the area of a triangle from three 3-D vertex coordinates using the cross-product method.

#### `extract_features_cpu(volume, threshold, kernel_size, scaling) → np.ndarray`

Extracts a 6-element feature vector from a 3-D block using CPU:
1. Porosity (after binarisation and morphological closing)
2. Number of clusters
3. Largest cluster size
4. Surface area (Marching Cubes + triangle area summation)
5. `scaling × porosity`
6. `kernel_size`

#### `extract_features_gpu(volume, threshold, kernel_size, scaling) → np.ndarray`

GPU-accelerated version using `cupy`. Falls back to CPU automatically if `cupy` is not installed.

#### `extract_features_parallel(blocks, threshold, kernel_size, scaling, use_gpu=False, max_workers=8) → np.ndarray`

Distributes feature extraction across `max_workers` processes using `ProcessPoolExecutor`. Returns a 2-D array of shape `(n_blocks, 6)`.

#### `extract_blocks(volume, block_shape=(128,256,256), min_active_voxels=1000) → list`

Extracts non-overlapping sub-volumes of shape `block_shape` from `volume`, filtering out blocks with fewer than `min_active_voxels` active voxels.

#### `load_data(volume_path, block_shape=(128,256,256)) → tuple`

Loads a TIFF volume, extracts blocks, shuffles them, and computes regression labels:
```
label = porosity × 1000 + largest_cluster_size × 0.01
```

**Returns**: `(X_blocks, y_labels)`

#### `objective(trial) → float`

Optuna objective function. Suggests hyperparameters for feature engineering (`threshold`, `kernel_size`, `scaling`) and XGBoost training (`learning_rate`, `max_depth`, `n_estimators`). Evaluates using 5-fold cross-validation (negative MSE).

### `__main__` entry point

```python
volume_path = "..."   # path to a 3-D TIFF file
block_shape = (128, 256, 256)
n_trials = 50

X_blocks, y_labels = load_data(volume_path, block_shape)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)
print(study.best_params)
```

---

## `utils/utils.py`

**Purpose**: Shared utility helpers.

### Functions

#### `generate_timestamp() → str`

Returns the current datetime as a string formatted `YYYY-MM-DD_HH-MM-SS`. Used for unique file naming.

#### `check_os() → str`

Returns the operating system name (`"Linux"`, `"Darwin"`, `"Windows"`) from `platform.system()`.
