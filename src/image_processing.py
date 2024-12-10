"""
Image processing module for handling and analyzing 3D image stacks.
"""

# pylint: disable=import-error

import os
import logging
from multiprocessing import Pool
from PIL import Image
from skimage import morphology, filters, measure
import scipy.ndimage
import tifffile as tiff
from src.utils.utils import generate_timestamp

timestamp = generate_timestamp()

logger = logging.getLogger(__name__)


def load_image(filepath):
    """
    Loads an image from the specified file and converts it to a grayscale image.

    Args:
        filepath (str): The path to the image file to be loaded.

    Returns:
        numpy.ndarray: A 2D array representing the grayscale image.
    """
    try:
        im = Image.open(filepath).convert("L")
        logger.info("Loaded image: %s", filepath)
        return np.array(im)
    except (IOError, ValueError) as e:
        logger.error("Error loading image %s: %s", filepath, e)
        return None


def load_images(directory):
    """
    Loads all TIFF images from the specified directory.
    """
    logger.info("Loading images from directory: %s", directory)

    filepaths = [
        os.path.join(directory, filename)
        for filename in sorted(os.listdir(directory))
        if filename.endswith(".tif")
    ]

    if not filepaths:
        logger.error("No valid images found in directory: %s", directory)
        raise ValueError(f"No valid images found in directory: {directory}")

    logger.info("Found %d image(s) in %s", len(filepaths), directory)

    with Pool() as pool:
        data_array = pool.map(load_image, filepaths)

    data_array = [img for img in data_array if img is not None]

    if not data_array:
        logger.error("No images could be successfully loaded from %s", directory)
        raise ValueError(f"No valid images could be successfully loaded from {directory}")

    return np.array(data_array)


from skimage import filters
import numpy as np
import matplotlib.pyplot as plt

def process_images_globally(data_array):
    """
    Applies global Otsu thresholding to the entire dataset and processes the images.

    Args:
        data_array (numpy.ndarray): 3D array containing the image stack.

    Returns:
        tuple: (blurred_images, binary_images, global_threshold)
    """
    if data_array.size == 0:
        raise ValueError("Input image stack is empty or invalid.")

    flattened_data = data_array.flatten()
    global_threshold = filters.threshold_otsu(flattened_data)

    plt.hist(flattened_data, bins=256, color='blue', alpha=0.7)
    plt.axvline(global_threshold, color='red', linestyle='dashed', linewidth=2)
    plt.title("Histogram of Image Data with Global Threshold")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.savefig(f"debug/histogram_with_threshold_{timestamp}.png")
    logger.info(f"Saved histogram with global threshold to debug/histogram_with_threshold_{timestamp}.png")
    plt.close()

    blurred_images = []
    binary_images = []

    for i, image in enumerate(data_array):
        try:
            image_blurred = filters.gaussian(image, sigma=1, mode="constant")
            blurred_images.append(image_blurred)

            binary_image = image_blurred > global_threshold
            binary_images.append(binary_image)
        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
            continue

    if not binary_images:
        raise ValueError("No valid images were processed!")

    return np.array(blurred_images), np.array(binary_images), global_threshold



def process_image(image):
    """
    Processes a single image by applying Gaussian blur and Otsu thresholding.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    image_blurred = filters.gaussian(image, sigma=1, mode="constant")
    binary_image = image_blurred > filters.threshold_otsu(image_blurred)
    average_intensity = np.mean(image_blurred)
    return image_blurred, binary_image, average_intensity


from skimage.morphology import closing, square

def apply_morphological_closing(binary_images):
    """
    Applies morphological closing to a stack of binary images.

    Args:
        binary_images (numpy.ndarray): 3D array of binary images.

    Returns:
        numpy.ndarray: Processed binary images after morphological closing.
    """
    closed_images = []
    for i, binary_image in enumerate(binary_images):
        try:
            closed_image = closing(binary_image, square(3))
            closed_images.append(closed_image)
        except Exception as e:
            logger.error(f"Error applying morphological closing to image {i}: {e}")
            continue

    return np.array(closed_images)



def interpolate_image_stack(image_stack, scaling_factor, order=1):
    """
    Interpolates a 3D image stack using the specified scaling factor.
    """
    return scipy.ndimage.zoom(image_stack,
                              (scaling_factor, scaling_factor, scaling_factor), order=order)


def find_largest_cluster(image_stack):
    labels, num_clusters = measure.label(image_stack, background=0, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.flatten())
    if len(cluster_sizes) <= 1:  # Prüfen, ob nur Hintergrund vorhanden ist
        raise ValueError("No clusters found in the image stack.")
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = (labels == largest_cluster_label)
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]



def save_to_tiff_stack(image_stack, filename):
    """
    Saves a stack of images as a TIFF file.

    Args:
        image_stack (numpy.ndarray): Stack of images to save.
        filename (str): Path to the output TIFF file.
    """
    try:
        tiff.imwrite(filename, image_stack, photometric='minisblack')
        logger.info(f"Saved image stack to {filename}")
    except Exception as e:
        logger.error(f"Error saving TIFF stack to {filename}: {e}")