import os
import numpy as np
import logging
import gc
from mesh_generation import generate_tetrahedral_mesh, reverse_binary

logger = logging.getLogger(__name__)

def extract_and_mesh_blocks(volume, voxel_spacing, block_size_mm, step_size_mm, output_dir, timestamp):
    """
    Extract overlapping blocks from the largest cluster and generate tetrahedral meshes for each block.

    Args:
        volume (np.ndarray): 3D binary volume of the largest cluster.
        voxel_spacing (tuple): (z, y, x) spacing in mm.
        block_size_mm (float): Desired physical size of block in mm (cubic).
        step_size_mm (float): Step size for moving window in mm (for overlap).
        output_dir (str): Directory to save results.
        timestamp (str): For filenames.

    """
    logger.info("Starting block extraction and tetrahedral meshing...")

    # Compute block size and step size in voxel units
    voxel_spacing_z, voxel_spacing_y, voxel_spacing_x = voxel_spacing
    block_size_voxels = (
        int(np.round(block_size_mm / voxel_spacing_z)),
        int(np.round(block_size_mm / voxel_spacing_y)),
        int(np.round(block_size_mm / voxel_spacing_x))
    )
    step_size_voxels = (
        int(np.round(step_size_mm / voxel_spacing_z)),
        int(np.round(step_size_mm / voxel_spacing_y)),
        int(np.round(step_size_mm / voxel_spacing_x))
    )

    logger.info(f"Block size in voxels: {block_size_voxels}, Step size in voxels: {step_size_voxels}")

    z_dim, y_dim, x_dim = volume.shape
    num_blocks = 0

    # Loop over the volume in steps (rasterartig)
    for z in range(0, z_dim - block_size_voxels[0] + 1, step_size_voxels[0]):
        for y in range(0, y_dim - block_size_voxels[1] + 1, step_size_voxels[1]):
            for x in range(0, x_dim - block_size_voxels[2] + 1, step_size_voxels[2]):
                block = volume[
                    z : z + block_size_voxels[0],
                    y : y + block_size_voxels[1],
                    x : x + block_size_voxels[2]
                ]

                if np.count_nonzero(block) == 0:
                    # Skip empty block (no cluster inside)
                    continue

                # Reverse for tetra meshing
                reversed_block = reverse_binary(block)

                block_filename = f"block_z{z}_y{y}_x{x}_{timestamp}.vtk"
                block_output_path = os.path.join(output_dir, block_filename)

                # Generate tetrahedral mesh
                logger.info(f"Meshing block at (z={z}, y={y}, x={x})...")
                tetrahedral_mesh = generate_tetrahedral_mesh(reversed_block, 0.1, block_output_path)

                if tetrahedral_mesh:
                    logger.info(f"Saved tetrahedral mesh: {block_output_path}")
                    num_blocks += 1
                else:
                    logger.warning(f"Failed to mesh block at (z={z}, y={y}, x={x})")

                # Memory cleanup
                del block, reversed_block
                gc.collect()

    logger.info(f"Block meshing completed. Total blocks meshed: {num_blocks}")
