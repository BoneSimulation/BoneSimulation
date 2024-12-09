# main.py

import logging
import os
import sys
import warnings
from multiprocessing import Pool
from PIL import Image
import numpy as np
from skimage import morphology, filters, measure

from src.utils.utils import generate_timestamp, check_os
from src.image_processing import (
    load_images,
    process_images,
    apply_morphological_closing,
    interpolate_image_stack,
    find_largest_cluster,
    save_to_tiff_stack,
)
from src.visualization import plot_histogram, plot_images

# Initialisiere Logger
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
    elif os_type == "Linux" or os_type == "MacOS":
        return "data/dataset"
    else:
        logger.warning("Unknown OS! Using default path.")
        return "data/dataset"



def process_and_visualize(directory):
    """
    Processes and visualizes all important images in the specified directory.
    """
    logger.info("Starting processing and visualization...")

    # Lade Bilder
    data_array = load_images(directory)
    logger.info(f"Loaded {data_array.shape[0]} images. Shape: {data_array.shape}")

    # Verarbeite Bilder
    image_blurred_array, binary_image_array, average_intensities = process_images(data_array)

    # Morphologische Schließung
    closed_binary_images = apply_morphological_closing(binary_image_array)

    # Interpolation auf geschlossene Bilder
    binary_image_array_interpolated = interpolate_image_stack(closed_binary_images, scaling_factor=0.5)

    # Größten Cluster finden
    largest_cluster, num_clusters, largest_cluster_size = find_largest_cluster(binary_image_array_interpolated)
    logger.info(f"Found {num_clusters} clusters. Largest cluster size: {largest_cluster_size}")

    # Speichern des größten Clusters
    save_to_tiff_stack(largest_cluster.astype(np.uint8), f"pictures/{timestamp}_largest_cluster.tif")

    # Durchschnittliche Intensität berechnen
    overall_average_intensity = np.mean(average_intensities) * 255
    logger.info(f"Average Intensity of the whole stack: {overall_average_intensity}")

    # Speichern des verarbeiteten Stacks
    save_to_tiff_stack(binary_image_array_interpolated, f"pictures/{timestamp}_binary_output_stack.tif")

    # Histogramm und Bilder plotten
    plot_histogram(binary_image_array_interpolated)
    plot_images(binary_image_array_interpolated, title="Processed Image")

    logger.info("Processing and visualization completed.")


if __name__ == "__main__":
    logger.debug("Running main script.")
    print("Running simulation")

    directory = get_base_path()

    if not os.path.isdir(directory):
        logger.error(f"Directory {directory} does not exist!")
        sys.exit(f"Directory {directory} does not exist.")

    logger.info(f"Using directory: {directory}")
    process_and_visualize(directory)
