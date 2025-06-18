# src/main.py

import logging
import os
import sys
import datetime
import timeit
import psutil
import gc
import time
import numpy as np

from image_processing import (
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
    crop_stack_custom
)
from cluster_analysis import find_largest_cluster
from mesh_generation import (
    marching_cubes,
    save_mesh_as_vtk,
    generate_tetrahedral_mesh,
    reverse_binary
)
from image_loading import load_images_in_chunks
from file_io import save_largest_cluster_as_tiff
from src.block_meshing import extract_and_mesh_blocks
from src.calculix_export import export_to_calculix
from src.image_processing import save_blocks_as_npy
from src.reporting import extract_statistics_from_volume

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s : %(name)s : %(message)s",
    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configuration
MEMORY_LIMIT_PERCENT = 80
CHUNK_SIZE = 50
OUTPUT_DIR = "test_pictures"

# Crop settings
cropping_settings = {
    "z": (3, 0),
    "y": (50, 50),
    "x": (15, 35),
}


def generate_run_script(input_inp_path, output_dir):
    run_script_path = os.path.join(output_dir, "run_simulation.sh")
    with open(run_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Automatisch erzeugtes Run-Script für CalculiX\n")
        f.write(f"ccx -i {os.path.splitext(os.path.basename(input_inp_path))[0]}\n")

    # Ausführbar machen
    os.chmod(run_script_path, 0o755)
    print(f"Run-Script gespeichert: {run_script_path}")


def get_base_path():
    """Returns the base path of the data."""
    return "/home/mathias/PycharmProjects/BoneSimulation/data"

def check_memory():
    """Checks memory usage."""
    memory = psutil.virtual_memory()
    return memory.percent > MEMORY_LIMIT_PERCENT

def wait_for_memory():
    """Waits until enough memory is free."""
    while check_memory():
        logger.warning(f"Memory usage too high ({psutil.virtual_memory().percent}%). Waiting...")
        gc.collect()
        time.sleep(10)

def ensure_output_dir(dir_path):
    """Ensures that the output directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created output directory: {dir_path}")

def process_and_visualize(directory):
    """Main processing pipeline."""
    logger.info("Starting processing pipeline...")
    ensure_output_dir(OUTPUT_DIR)
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

    dataset_path = os.path.join(directory, "bigdataset")
    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    logger.info(f"Loading images from: {dataset_path}")
    data_generator = load_images_in_chunks(dataset_path, chunk_size=CHUNK_SIZE)

    interpolated_chunks = []

    try:
        # Process chunks
        for i, chunk in enumerate(data_generator):
            logger.info(f"Processing chunk {i + 1} with {len(chunk)} images")
            wait_for_memory()

            blurred_chunk, binary_chunk, _ = process_images_globally(chunk)
            del chunk
            gc.collect()

            wait_for_memory()
            closed_chunk = apply_morphological_closing(binary_chunk)
            del binary_chunk, blurred_chunk
            gc.collect()

            wait_for_memory()
            interpolated_chunk = interpolate_image_stack(closed_chunk, scaling_factor=0.5)
            del closed_chunk
            gc.collect()

            interpolated_chunks.append(interpolated_chunk)
            logger.info(f"Chunk {i + 1} processed and added to stack.")

        # Concatenate all chunks
        logger.info("Concatenating all interpolated chunks...")
        interpolated_stack = np.concatenate(interpolated_chunks, axis=0)
        del interpolated_chunks
        gc.collect()

        # Crop
        interpolated_stack = crop_stack_custom(interpolated_stack, cropping_settings)
        logger.info(f"Stack cropped with settings: {cropping_settings}")

        # Largest cluster
        wait_for_memory()
        logger.info("Finding largest cluster...")
        largest_cluster, num_clusters, cluster_size = find_largest_cluster(interpolated_stack)
        logger.info(f"Largest cluster: {cluster_size} voxels in {num_clusters} clusters")

        cluster_tiff_path = os.path.join(OUTPUT_DIR, f"largest_cluster_{timestamp}.tif")
        save_largest_cluster_as_tiff(largest_cluster, cluster_tiff_path)
        logger.info(f"Largest cluster saved: {cluster_tiff_path}")

        # Extract & mesh blocks
        blocks = extract_and_mesh_blocks(
            volume=largest_cluster,
            voxel_spacing=(0.05, 0.05, 0.05),
            block_size_mm=30.0,
            step_size_mm=15.0,
            output_dir=OUTPUT_DIR,
            timestamp=timestamp
        )

        save_blocks_as_npy(blocks, OUTPUT_DIR, timestamp)

        if largest_cluster is None:
            logger.error("No cluster found. Exiting.")
            return

        del interpolated_stack
        gc.collect()

        # Surface mesh
        wait_for_memory()
        logger.info("Generating surface mesh...")
        verts, faces = marching_cubes(largest_cluster)
        if verts is not None and faces is not None:
            mesh_path = os.path.join(OUTPUT_DIR, f"mesh_{timestamp}.vtk")
            save_mesh_as_vtk(verts, faces, mesh_path)
            logger.info(f"Surface mesh saved: {mesh_path}")
        else:
            logger.error("Surface mesh generation failed.")

        # Tetrahedral mesh
        wait_for_memory()
        logger.info("Generating tetrahedral mesh...")
        reversed_cluster = reverse_binary(largest_cluster)
        tetra_path = os.path.join(OUTPUT_DIR, f"tetramesh_{timestamp}.vtk")
        tetra_mesh = generate_tetrahedral_mesh(reversed_cluster, 0.1, tetra_path)
        if tetra_mesh:
            logger.info(f"Tetrahedral mesh saved: {tetra_path}")
        else:
            logger.warning("Tetrahedral mesh generation failed.")

        logger.info("Processing completed successfully.")

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

    material_props = {'E': 18000.0, 'nu': 0.3}

    inp_path = os.path.join(OUTPUT_DIR, f"bone_sim_{timestamp}.inp")

    export_to_calculix(mesh_points, mesh_elements, material_props, inp_path)

    generate_run_script(inp_path, OUTPUT_DIR)

    from reporting import generate_report
    cluster_sizes, porosity, num_clusters = extract_statistics_from_volume(largest_cluster)
    logger.info(f"Statistik extrahiert: {num_clusters} Cluster, Porosität {porosity:.2%}")

    report_path = os.path.join(OUTPUT_DIR, f"analysis_report_{timestamp}.pdf")
    generate_report(cluster_sizes, porosity, report_path)
    logger.info(f"Analyse-Report gespeichert: {report_path}")


if __name__ == "__main__":
    start_time = timeit.default_timer()
    try:
        directory = get_base_path()
        process_and_visualize(directory)
        elapsed = timeit.default_timer() - start_time
        logger.info(f"Total processing time: {elapsed:.2f} seconds.")
    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
