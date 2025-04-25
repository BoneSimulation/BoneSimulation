# main.py (optimized)

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
)
from cluster_analysis import (
    find_largest_cluster
)
from mesh_generation import (
    marching_cubes,
    save_mesh_as_vtk,
    generate_tetrahedral_mesh
)
from image_loading import (
    load_images_in_chunks
)
from file_io import (
    save_tiff_in_chunks
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s : %(name)s : %(message)s",
    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configuration parameters
MEMORY_LIMIT_PERCENT = 80  # Maximum memory usage in percent
CHUNK_SIZE = 50  # Number of images per chunk for processing
OUTPUT_DIR = "test_pictures"


def get_base_path():
    """Returns the base path of the data."""
    return "/home/mathias/PycharmProjects/BoneSimulation/data"


def check_memory():
    """Checks memory usage and returns True if it is too high."""
    memory = psutil.virtual_memory()
    return memory.percent > MEMORY_LIMIT_PERCENT


def wait_for_memory():
    """Waits until enough memory is free."""
    while check_memory():
        logger.warning(f"Memory usage too high ({psutil.virtual_memory().percent}%). Waiting 10 seconds...")
        gc.collect()  # Explicitly call garbage collection
        time.sleep(10)


def ensure_output_dir(dir_path):
    """Ensures that the output directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Output directory created: {dir_path}")


def process_and_visualize(directory):
    """Conducts the entire image processing and meshing process, memory-efficiently."""
    logger.info("Starting processing and visualization...")
    ensure_output_dir(OUTPUT_DIR)

    # Timestamp for output files
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

    # Set path directly for the large dataset
    dataset_path = os.path.join(directory, "bigdataset")

    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Load and process images in chunks
    logger.info(f"Loading images from: {dataset_path}")

    # Save the raw dataset in chunks
    raw_output_path = os.path.join(OUTPUT_DIR, f"raw_{timestamp}.tif")

    # Process images in chunks instead of loading everything at once
    try:
        data_generator = load_images_in_chunks(dataset_path, chunk_size=CHUNK_SIZE)
        all_processed_chunks = []

        for i, chunk in enumerate(data_generator):
            logger.info(f"Processing chunk {i + 1} with {len(chunk)} images")

            # Check memory before each processing step
            wait_for_memory()

            # Process chunk
            blurred_chunk, binary_chunk, _ = process_images_globally(chunk)
            del chunk  # Free up memory
            gc.collect()

            wait_for_memory()
            closed_binary_chunk = apply_morphological_closing(binary_chunk)
            del binary_chunk, blurred_chunk
            gc.collect()

            wait_for_memory()
            interpolated_chunk = interpolate_image_stack(closed_binary_chunk, scaling_factor=0.5)
            del closed_binary_chunk
            gc.collect()

            # Save processed chunk
            all_processed_chunks.append(interpolated_chunk)

            # Log progress
            logger.info(f"Chunk {i + 1} processed. Current memory usage: {psutil.virtual_memory().percent}%")

        # Concatenate all processed chunks
        wait_for_memory()
        interpolated_stack = np.concatenate(all_processed_chunks, axis=0)
        del all_processed_chunks
        gc.collect()

        # Find the largest cluster
        wait_for_memory()
        logger.info("Searching for the largest cluster...")
        largest_cluster, num_clusters, cluster_size = find_largest_cluster(interpolated_stack)
        logger.info(f"Largest cluster found: {cluster_size} voxels out of {num_clusters} clusters")

        # If no cluster was found, exit the program
        if largest_cluster is None:
            logger.error("No cluster found. Exiting the program.")
            return

        # Original interpolated_stack is no longer needed
        del interpolated_stack
        gc.collect()

        # Generate mesh
        wait_for_memory()
        logger.info("Generating mesh with Marching Cubes...")
        verts, faces = marching_cubes(largest_cluster)

        if verts is not None and faces is not None:
            mesh_output_path = os.path.join(OUTPUT_DIR, f"mesh_{timestamp}.vtk")
            save_mesh_as_vtk(verts, faces, mesh_output_path)
            logger.info(f"Mesh saved as: {mesh_output_path}")
        else:
            logger.error("Error during mesh generation. Skipping save.")

        # Generate tetrahedral mesh
        wait_for_memory()
        logger.info("Generating tetrahedral mesh...")
        tetra_output_path = os.path.join(OUTPUT_DIR, f"tetramesh_{timestamp}.vtk")
        tetrahedral_mesh = generate_tetrahedral_mesh(largest_cluster, 0.1, tetra_output_path)

        if tetrahedral_mesh:
            logger.info(f"Tetrahedral mesh successfully generated and saved as: {tetra_output_path}")
        else:
            logger.warning("Tetrahedral mesh could not be generated.")

        logger.info("Processing completed.")

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

    if __name__ == "__main__":
        starttime = timeit.default_timer()
        try:
            directory = get_base_path()
            process_and_visualize(directory)
            endtime = timeit.default_timer()
            logger.info(f"Processing completed in {endtime - starttime:.2f} seconds.")
        except KeyboardInterrupt:
            logger.info("Program interrupted by user.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)

