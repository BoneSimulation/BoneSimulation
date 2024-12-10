"""This is the main file - every important thing is placed here"""

# pylint: disable=import-error
# pylint: disable=logging-fstring-interpolation

import logging
import os
import sys
import warnings
import numpy as np

from src.utils.utils import generate_timestamp, check_os
from src.image_processing import (
    load_images,
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
    find_largest_cluster,
    save_to_tiff_stack,
)
from src.visualization import plot_histogram, plot_images

timestamp = generate_timestamp()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("logfile.log")
formatter = logging.Formatter(f"{timestamp}: %(levelname)s : %(name)s : %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")


def get_base_path():
    os_type = check_os()
    if os_type == "Windows":
        return "data/dataset"
    if os_type == "Linux" or os_type == "MacOS":
        return "data/dataset"
    else:
        logger.warning("Unknown OS! Using default path.")
        return "data/dataset"


def process_and_visualize(directory):
    """
    Processes and visualizes all important images in the specified directory.
    """
    logger.info("Starting processing and visualization...")

    data_array = load_images(directory)
    logger.info(f"Loaded {data_array.shape[0]} images. Shape: {data_array.shape}")

    blurred_images, binary_images, global_threshold = process_images_globally(data_array)
    logger.info(f"Global threshold determined: {global_threshold}")

    closed_binary_images = apply_morphological_closing(binary_images)

    num_active_pixels = np.sum(closed_binary_images)
    logger.debug(f"Number of active pixels after closing: {num_active_pixels}")

    if num_active_pixels == 0:
        logger.error("No active pixels found after morphological closing.")
        for idx, binary_image in enumerate(binary_images[:5]):
            save_to_tiff_stack(
                binary_image.astype(np.uint8),
                f"debug/binary_image_before_closing_{idx}.tif"
            )
        return

    # Interpolation
    binary_image_array_interpolated = interpolate_image_stack(closed_binary_images, scaling_factor=0.5)
    logger.debug(f"Interpolated stack shape: {binary_image_array_interpolated.shape}")

    if np.sum(binary_image_array_interpolated) == 0:
        logger.error("Interpolated binary image stack is empty. No clusters to process.")
        return

    try:
        largest_cluster, num_clusters, largest_cluster_size = find_largest_cluster(binary_image_array_interpolated)
        logger.info(f"Found {num_clusters} clusters. Largest cluster size: {largest_cluster_size}")
    except ValueError as e:
        logger.error(f"Error finding largest cluster: {e}")
        return

    save_to_tiff_stack(largest_cluster.astype(np.uint8), f"pictures/{timestamp}_largest_cluster.tif")

    overall_average_intensity = np.mean(binary_images) * 255
    logger.info(f"Average Intensity of the whole stack: {overall_average_intensity}")

    save_to_tiff_stack(binary_image_array_interpolated, f"pictures/{timestamp}_binary_output_stack.tif")

    plot_histogram(binary_image_array_interpolated)
    plot_images(binary_image_array_interpolated, title="Processed Image")

    logger.info("Processing and visualization completed.")



if __name__ == "__main__":
    logger.debug("Running main script.")
    print("Running simulation")

    DIRECTORY = get_base_path()

    if not os.path.isdir(DIRECTORY):
        logger.error(f"Directory {DIRECTORY} does not exist!")
        sys.exit(f"Directory {DIRECTORY} does not exist.")

    logger.info(f"Using directory: {DIRECTORY}")
    process_and_visualize(DIRECTORY)
