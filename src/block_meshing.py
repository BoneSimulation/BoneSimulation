# src/block_meshing.py

import logging
import os
import numpy as np
from mesh_generation import generate_tetrahedral_mesh, reverse_binary

logger = logging.getLogger(__name__)

def extract_and_mesh_blocks(volume, voxel_spacing, block_size_mm, step_size_mm, output_dir, timestamp):
    block_size_voxels = tuple(int(block_size_mm / s) for s in voxel_spacing)
    step_size_voxels = tuple(int(step_size_mm / s) for s in voxel_spacing)

    z_max, y_max, x_max = volume.shape
    z_block, y_block, x_block = block_size_voxels
    z_step, y_step, x_step = step_size_voxels

    block_idx = 0

    for z in range(0, z_max - z_block + 1, z_step):
        for y in range(0, y_max - y_block + 1, y_step):
            for x in range(0, x_max - x_block + 1, x_step):
                block = volume[z:z + z_block, y:y + y_block, x:x + x_block]

                if np.count_nonzero(block) == 0:
                    continue

                # Save block as npy
                block_save_path = os.path.join(output_dir, f"block_{block_idx:04d}_{timestamp}.npy")
                np.save(block_save_path, block)
                logger.info(f"Block {block_idx} saved as: {block_save_path}")

                logger.info(f"Processing block {block_idx}: position (z={z}, y={y}, x={x})")

                reversed_block = reverse_binary(block)

                tetra_output_path = os.path.join(
                    output_dir, f"block_{block_idx:04d}_{timestamp}.vtk"
                )

                try:
                    tetrahedral_mesh = generate_tetrahedral_mesh(reversed_block, 0.1, tetra_output_path)
                    if tetrahedral_mesh:
                        logger.info(f"Tetrahedral mesh saved as: {tetra_output_path}")
                    else:
                        logger.warning(f"Tetrahedral mesh for block {block_idx} could not be generated.")
                except Exception as e:
                    logger.error(f"Error meshing block {block_idx}: {e}", exc_info=True)

                block_idx += 1

    logger.info(f"Total blocks processed: {block_idx}")
