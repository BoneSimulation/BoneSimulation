import logging
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from src.image_processing import (
    load_images,
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
    find_largest_cluster,
    save_to_tiff_stack,
    save_raw_tiff_stack,
    marching_cubes,
    save_mesh_as_vtk,
    generate_tetrahedral_mesh
)
from utils.utils import generate_timestamp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s: %(levelname)s : %(name)s : %(message)s",
                    handlers=[
                        logging.FileHandler("logfile.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

def get_base_path():
    """Returns the base path for the dataset."""
    return "C:\\Users\\Mathias\\Documents\\BoneSimulation\\data\\dataset"

def process_and_visualize(directory):
    """Main function for processing and visualizing images."""
    logger.info("Starting processing and visualization...")
    data_array = load_images(directory)
    logger.info(f"Loaded images: {data_array.shape}")
    save_raw_tiff_stack(data_array, f"pictures/raw.tif")
    blurred_images, binary_images, global_threshold = process_images_globally(data_array)
    logger.info(f"Global threshold determined: {global_threshold}")
    closed_binary_images = apply_morphological_closing(binary_images)
    interpolated_stack = interpolate_image_stack(closed_binary_images, scaling_factor=0.5)
    logger.info(f"Interpolated stack created: {interpolated_stack.shape}")
    largest_cluster, _, cluster_size = find_largest_cluster(interpolated_stack)
    logger.info(f"Largest cluster found: {cluster_size} voxels")
    verts, faces = marching_cubes(interpolated_stack)
    save_mesh_as_vtk(verts, faces, f"test_pictures/mesh.vtk")
    generate_tetrahedral_mesh(largest_cluster, 0.1, f"test_pictures/tetramesh_.vtk")
    logger.info("Processing completed.")

if __name__ == "__main__":
    logger.debug("Running main script.")
    directory = get_base_path()
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    process_and_visualize(directory)