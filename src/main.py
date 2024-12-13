# main.py
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

# Setup logging
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
    if os_type in ["Windows", "Linux", "MacOS"]:
        return "data/dataset"
    else:
        logger.warning("Unknown OS! Using default path.")
        return "data/dataset"


def validate_binary_images(binary_images):
    """
    Validates if binary images contain any active pixels.

    Args:
        binary_images (numpy.ndarray): Stack of binary images.

    Returns:
        numpy.ndarray: Validated binary images.
    """
    valid_images = []
    for i, binary_image in enumerate(binary_images):
        if np.sum(binary_image) > 0:
            valid_images.append(binary_image)
        else:
            logger.warning(f"Binary image at index {i} contains no active pixels.")

    if not valid_images:
        raise ValueError("No valid binary images found after validation.")

    return np.array(valid_images)


def process_and_visualize(directory):
    """
    Processes and visualizes all important images in the specified directory.
    """
    logger.info("Starting processing and visualization...")

    try:
        data_array = load_images(directory)
    except ValueError as e:
        logger.error(f"Error loading images: {e}")
        return

    logger.info(f"Loaded {data_array.shape[0]} images. Shape: {data_array.shape}")

    try:
        blurred_images, binary_images, global_threshold = process_images_globally(data_array)
        logger.info(f"Global threshold determined: {global_threshold}")
    except ValueError as e:
        logger.error(f"Error during global processing: {e}")
        return

    try:
        binary_images = validate_binary_images(binary_images)
        closed_binary_images = apply_morphological_closing(binary_images)
    except ValueError as e:
        logger.error(f"Error during validation or morphological closing: {e}")
        return

    num_active_pixels = np.sum(closed_binary_images)
    logger.debug(f"Number of active pixels after closing: {num_active_pixels}")

    if num_active_pixels == 0:
        logger.error("No active pixels found after morphological closing.")
        logger.info("Saving original binary images for debugging.")
        for idx, binary_image in enumerate(binary_images[:5]):
            save_to_tiff_stack(
                binary_image.astype(np.uint8),
                f"debug_binary_image_{idx}.tif"
            )
        return

    # Interpolation
    try:
        binary_image_array_interpolated = interpolate_image_stack(closed_binary_images, scaling_factor=0.5)
        logger.debug(f"Interpolated stack shape: {binary_image_array_interpolated.shape}")
    except Exception as e:
        logger.error(f"Error during interpolation: {e}")
        return

    if np.sum(binary_image_array_interpolated) == 0:
        logger.error("Interpolated binary image stack is empty. No clusters to process.")
        return

    try:
        largest_cluster, num_clusters, largest_cluster_size = find_largest_cluster(binary_image_array_interpolated)
        logger.info(f"Found {num_clusters} clusters. Largest cluster size: {largest_cluster_size}")
    except ValueError as e:
        logger.error(f"Error finding largest cluster: {e}")
        return

    save_to_tiff_stack(largest_cluster.astype(np.uint8), f"largest_cluster_{timestamp}.tif")

    overall_average_intensity = np.mean(binary_images) * 255
    logger.info(f"Average Intensity of the whole stack: {overall_average_intensity}")

    save_to_tiff_stack(binary_image_array_interpolated, f"binary_output_stack_{timestamp}.tif")

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
