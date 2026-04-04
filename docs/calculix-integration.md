# CalculiX Integration

This document explains how BoneSimulation exports tetrahedral bone meshes to the CalculiX finite-element solver, what the generated `.inp` file contains, how to run the simulation, and how to interpret the results.

---

## Table of Contents

1. [Overview](#overview)
2. [Material Model](#material-model)
3. [Boundary Conditions and Loading](#boundary-conditions-and-loading)
4. [`.inp` File Structure](#inp-file-structure)
5. [Running the Simulation](#running-the-simulation)
6. [Result Files](#result-files)
7. [Post-processing in ParaView](#post-processing-in-paraview)
8. [Customising the Analysis](#customising-the-analysis)

---

## Overview

After the tetrahedral mesh is generated, BoneSimulation automatically exports it to a CalculiX input file (`.inp`) that encodes:

- All mesh nodes and elements
- Bone material properties (linear elastic)
- Boundary conditions (fixed bottom face)
- Applied loads (compressive force on top face)
- A static analysis step

A companion bash script (`run_simulation.sh`) is also written to invoke the CalculiX solver with the correct number of OpenMP threads.

The pipeline triggers this export and simulation automatically in non-dry modes:

```
Tetrahedral mesh (VTK)
         │
         ▼
  calculix_export.py
         │
         ├─→  bone_sim_<timestamp>.inp
         └─→  run_simulation.sh
                  │
                  ▼
            CalculiX solver (ccx)
                  │
                  ├─→  bone_sim_<timestamp>.dat  (text results)
                  └─→  bone_sim_<timestamp>.frd  (binary results for CGX)
```

---

## Material Model

The bone is modelled as a **linear isotropic elastic** material.

| Property | Symbol | Default value | Unit |
|---|---|---|---|
| Young's modulus | E | 18 000 | MPa |
| Poisson's ratio | ν | 0.3 | — |

These values are typical for human cortical bone. They can be customised in `main.py`:

```python
material_props = {'E': 18000.0, 'nu': 0.3}
```

The material is assigned to all elements via:

```
*MATERIAL, NAME=Bone
*ELASTIC
18000.0, 0.3
*SOLID SECTION, ELSET=AllElements, MATERIAL=Bone
```

> **Note**: The current model uses a single homogeneous material. For heterogeneous bone simulation, you would need to assign different material properties per element based on the local CT Hounsfield Unit density, which is not implemented in the current version.

---

## Boundary Conditions and Loading

BoneSimulation automatically detects boundary nodes by examining their z-coordinate:

### Fixed Nodes (Bottom Face)

All nodes at `z = z_min` (within a tolerance of 1e-5) are fully constrained in all three translational degrees of freedom:

```
*BOUNDARY
<node_id>, 1, 3, 0.0
```

This simulates the bone resting on a flat, rigid surface (e.g., a testing platen).

### Loaded Nodes (Top Face)

All nodes at `z = z_max` receive a distributed force in the z-direction:

```
*CLOAD
<node_id>, 3, <force_per_node>
```

The total applied force defaults to **100 N**, distributed equally over all top-face nodes:

```python
force_per_node = load_value / len(loaded_nodes)
```

This models a compressive load applied from the top — a typical uniaxial compression test setup used in bone biomechanics research.

---

## `.inp` File Structure

The complete file structure:

```
*HEADING
Knochen-Simulation

*NODE
1, x1, y1, z1
2, x2, y2, z2
...
N, xN, yN, zN

*ELEMENT, TYPE=C3D4
1, n1, n2, n3, n4
...
M, n1, n2, n3, n4

*MATERIAL, NAME=Bone
*ELASTIC
18000.0, 0.3

*SOLID SECTION, ELSET=AllElements, MATERIAL=Bone
AllElements

*ELSET, ELSET=AllElements, GENERATE
1, M, 1

*BOUNDARY
<fixed_node_1>, 1, 3, 0.0
<fixed_node_2>, 1, 3, 0.0
...

*CLOAD
<loaded_node_1>, 3, <force_per_node>
<loaded_node_2>, 3, <force_per_node>
...

*STEP
*STATIC
*END STEP
```

**Element type**: `C3D4` — a 4-node linear tetrahedral element (first-order). It has 3 translational DOFs per node and uses constant-strain interpolation.

---

## Running the Simulation

### Automatically (via pipeline)

When the pipeline runs in non-dry mode, it:
1. Prompts for the preferred solver binary name (default: `ccx`)
2. Generates `run_simulation.sh`
3. Executes `run_simulation.sh` with `os.system`

```
Please enter the preferred solver you would like to use (default: ccx):
```

Press Enter to use `ccx` or enter the path to a specific binary (e.g., `./ccx_2.22`).

### Manually

```bash
cd calculix/
bash run_simulation.sh
```

Or directly:

```bash
export OMP_NUM_THREADS=8
ccx -i bone_sim_<timestamp>
```

The `-i` flag specifies the input file name **without** the `.inp` extension.

### Multi-threading

The run script automatically sets `OMP_NUM_THREADS` to the maximum available logical CPU count (cores × threads/core) using `lscpu`. For a 16-core / 2-thread-per-core machine, this sets `OMP_NUM_THREADS=32`.

---

## Result Files

CalculiX writes results to the same directory as the `.inp` file:

| File | Content |
|---|---|
| `bone_sim_<timestamp>.dat` | ASCII summary: reaction forces, displacements at selected nodes |
| `bone_sim_<timestamp>.frd` | Binary result file (CGX format) — nodal displacements, element stresses |
| `bone_sim_<timestamp>.sta` | Convergence and step information |
| `bone_sim_<timestamp>.cvg` | Convergence monitor |

---

## Post-processing in ParaView

The `.frd` result file can be converted for ParaView using the `ccx2paraview` utility (included in the Conda environment):

```bash
cd calculix/
ccx2paraview bone_sim_<timestamp>.frd paraview
```

This generates a `.pvd` collection or `.vtu` files. Open the `.pvd` in ParaView and apply filters such as:

- **Warp By Vector** → to visualise displacements
- **Calculator** → to compute von Mises stress from stress tensor components

### Expected results

For a typical trabecular bone specimen under 100 N compression:

| Quantity | Typical range |
|---|---|
| Maximum displacement (z) | 0.001 – 0.01 mm |
| Von Mises stress | 0.1 – 100 MPa (depends on trabecular thickness) |
| Reaction force (z, bottom) | ~100 N (should equal applied load) |

---

## Customising the Analysis

### Changing the load magnitude

In `main.py`:

```python
# The 4th positional argument to export_to_calculix is load_value
export_to_calculix(mesh_points, mesh_elements, material_props, inp_path,
                   fix_z_plane=True, load_z_plane=True, load_value=500.0)
```

### Disabling automatic boundary conditions

```python
export_to_calculix(mesh_points, mesh_elements, material_props, inp_path,
                   fix_z_plane=False, load_z_plane=False)
```

Then add your custom `*BOUNDARY` and `*CLOAD` sections to the `.inp` file manually.

### Adding a dynamic analysis

Append the following to the `.inp` file after generation to run a frequency analysis:

```
*STEP
*FREQUENCY
10         ← number of eigenvalues to extract
*END STEP
```

### Heterogeneous material properties

To assign density-based material properties (Rho-E relationships) per element, you would need to extend `calculix_export.py` to:
1. Map each element's centroid back to a CT voxel
2. Look up the Hounsfield Unit value
3. Apply a bone-density-to-modulus relationship (e.g., Keyak et al. 1994)
4. Write per-element `*MATERIAL` sections or use `*USER MATERIAL` with UMAT

This is not currently implemented but is a natural extension of the pipeline.
