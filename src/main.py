import logging
import os
import sys
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
    """
        Retrieves the base path for the dataset used in the Bone Simulation project.

        This function returns a string that represents the directory path where the dataset
        is stored. This path is typically used as a starting point for loading data files
        required for processing and analysis within the project.

        Args:
            None

        Returns:
            str: The base path to the dataset directory.
    """
    return "/home/mathias/PycharmProjects/BoneSimulation/data/dataset"

def process_and_visualize(directory):
    """
    Processes and visualizes images from a specified directory, performing a series of image processing steps
    and saving the results in various formats.

    This function includes the following steps:
    1. Loads images from the provided directory into a data array.
    2. Saves the raw image stack as a TIFF file.
    3. Applies global image processing techniques to generate blurred and binary images.
    4. Applies morphological closing to the binary images to enhance features.
    5. Interpolates the image stack to create a smoother representation.
    6. Identifies the largest cluster of connected components in the interpolated stack.
    7. Uses the Marching Cubes algorithm to create a 3D mesh from the processed data.
    8. Saves the generated mesh as a VTK file.
    9. Generates a tetrahedral mesh for the largest cluster and saves it as a VTK file.

    Args:
        directory (str): Path to the directory containing the images to be processed.

    Returns:
        None: This function does not return a value; it saves results to files, including:
            - A TIFF file of the raw image stack.
            - A VTK file of the generated 3D mesh.
            - A VTK file of the tetrahedral mesh for the largest cluster.

    Logging:
        The function logs various stages of processing, including the number of images loaded,
        the global threshold determined, and the size of the largest cluster found.
    """

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