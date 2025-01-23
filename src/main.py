"""
main.py

Main script for processing, visualizing, and exporting image data.
"""

import logging
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from vtkmodules.util.numpy_support import numpy_to_vtk
from src.utils.utils import generate_timestamp, check_os
from src.image_processing import (
    load_images,
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
    find_largest_cluster,
    save_to_tiff_stack,
    save_raw_tiff_stack,
    numpy2vtk, marching_cubes, save_mesh_as_vtk, generate_tetrahedral_mesh, generate_calculix_input
)
from src.visualization import plot_histogram, plot_images

# Generate timestamp for filenames
timestamp = generate_timestamp()

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format=f"%(asctime)s: %(levelname)s : %(name)s : %(message)s",
                    handlers=[
                        logging.FileHandler(f"logfile_{timestamp}.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

# Suppress debug information from third-party libraries
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("tifffile").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def get_base_path():
    """Returns the base path for the dataset."""
    os_type = check_os()
    logger.info(f"Detected OS: {os_type}")
    return "data/dataset"


def process_and_visualize(directory):
    """
    Main function for processing and visualizing images.
    """
    logger.info("Starting processing and visualization...")

    # Load images
    data_array = load_images(directory)
    logger.info(f"Loaded images: {data_array.shape}")

    # Save raw data stack
    save_raw_tiff_stack(data_array, f"pictures/raw_{timestamp}.tif")

    # Process images with global threshold
    blurred_images, binary_images, global_threshold = process_images_globally(data_array)
    logger.info(f"Global threshold determined: {global_threshold}")

    # Morphological closing
    closed_binary_images = apply_morphological_closing(binary_images)

    # Validate active pixels after morphological closing
    active_pixels = np.sum(closed_binary_images)
    logger.info(f"Number of active pixels after closing: {active_pixels}")

    if active_pixels == 0:
        logger.error("No active pixels found after morphological closing.")
        return

    # Interpolation
    interpolated_stack = interpolate_image_stack(closed_binary_images, scaling_factor=0.5)
    logger.info(f"Interpolated stack created: {interpolated_stack.shape}")

    if np.sum(interpolated_stack) == 0:
        logger.error("Interpolated stack contains no active pixels.")
        return

    # Find clusters
    try:
        largest_cluster, num_clusters, cluster_size = find_largest_cluster(interpolated_stack)
        logger.info(f"Largest cluster found: {cluster_size} voxels")
    except ValueError as e:
        logger.error(f"Error finding the largest cluster: {e}")
        return

    # Generate mesh from the interpolated stack
    verts, faces = marching_cubes(interpolated_stack)
    logger.info(f"Mesh generated with {len(verts)} vertices and {len(faces)} faces.")

    # Save the mesh to a VTK file
    save_mesh_as_vtk(verts, faces, f"test_pictures/mesh_{timestamp}.vtk")

    # Save interpolated stack as VTK
    numpy2vtk(interpolated_stack, f"test_pictures/output_{timestamp}.vti")

    # Visualization
    plot_histogram(interpolated_stack, global_threshold)
    plot_images(interpolated_stack, title="Processed Images")

    # Schritt: Generiere das Tetraedernetz
    voxelsize = 0.1  # Beispielwert in mm
    tetrahedral_mesh_filename = f"test_pictures/tetramesh_{timestamp}.vtk"
    generate_tetrahedral_mesh(largest_cluster, voxelsize, tetrahedral_mesh_filename)

    # Schritt: Generiere die CalculiX Input-Datei
    calculix_input_filename = f"test_pictures/tetramesh_fe_{timestamp}.inp"
    input_template = "bone.inp"  # Vorlage mit Materialdefinition und Randbedingungen
    generate_calculix_input(tetrahedral_mesh_filename, input_template, calculix_input_filename)


    logger.info("Processing completed.")


if __name__ == "__main__":
    logger.debug("Running main script.")
    print("Running simulation")

    directory = get_base_path()
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    process_and_visualize(directory)
