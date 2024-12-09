"""
Image processing module for handling and analyzing 3D image stacks.
"""

import os
import logging
from multiprocessing import Pool
import numpy as np
from PIL import Image
from skimage import morphology, filters, measure
import scipy.ndimage

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


def process_images(data_array):
    """
    Applies Gaussian blur and binary thresholding to each image in the stack.
    """
    processed_results = []
    for i, image in enumerate(data_array):
        try:
            result = process_image(image)
            processed_results.append(result)
        except ValueError as e:
            logger.error("Error processing image %d: %s", i, e)

    if not processed_results:
        raise ValueError("No valid images were processed!")

    image_blurred_array, binary_image_array, average_intensities = zip(*processed_results)
    return (
        np.array(image_blurred_array),
        np.array(binary_image_array),
        np.array(average_intensities),
    )


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


def apply_morphological_closing(binary_images):
    """
    Applies morphological closing to all binary images in the stack.
    """
    return np.array([
        morphology.closing(image, footprint=morphology.disk(6))
        for image in binary_images
    ])


def interpolate_image_stack(image_stack, scaling_factor, order=1):
    """
    Interpolates a 3D image stack using the specified scaling factor.
    """
    return scipy.ndimage.zoom(image_stack, (scaling_factor, scaling_factor, scaling_factor), order=order)


def find_largest_cluster(image_stack):
    """
    Finds the largest voxel cluster in a 3D image stack.
    """
    labels, num_clusters = measure.label(image_stack, background=0, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.flatten())
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = (labels == largest_cluster_label)
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]


def save_to_tiff_stack(image_array, filepath):
    """
    Saves a 3D image stack as a multi-page TIFF file.
    """
    images = [Image.fromarray((image * 255).astype(np.uint8)) for image in image_array]
    images[0].save(filepath, save_all=True, append_images=images[1:])
