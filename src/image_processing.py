"""
image_processing.py

This file contains functions for loading, processing, and saving image data. It includes operations such as thresholding, morphological processing, and cluster analysis.
"""

import logging
import numpy as np
from multiprocessing import Pool
from skimage import morphology, filters, measure
import scipy.ndimage
import tifffile as tiff
from PIL import Image
import os

logger = logging.getLogger(__name__)

def load_image(filepath):
    """Loads a single image as a grayscale NumPy array."""
    try:
        im = Image.open(filepath).convert("L")
        return np.array(im)
    except Exception as e:
        logger.error("Error loading image %s: %s", filepath, e)
        return None

def load_images(directory):
    """Loads all .tif images in a directory into a 3D NumPy array."""
    filepaths = sorted([
        os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".tif")
    ])
    if not filepaths:
        raise ValueError("No valid images found in directory.")
    with Pool() as pool:
        images = pool.map(load_image, filepaths)
    images = [img for img in images if img is not None]
    if not images:
        raise ValueError("Failed to load any images.")
    logger.info(f"Loaded {len(images)} images from directory {directory}.")
    return np.array(images)

def process_images_globally(data_array):
    """Processes images using global Otsu thresholding."""
    if data_array.size == 0:
        raise ValueError("Input image stack is empty.")
    threshold = filters.threshold_otsu(data_array.flatten())
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold
    logger.info(f"Threshold applied: {threshold:.2f}")
    return blurred, binary, threshold

def apply_morphological_closing(binary_images):
    """Performs morphological closing on binary images."""
    closed = morphology.closing(binary_images, morphology.ball(3))
    logger.debug(f"Performed morphological closing. Active pixels: {np.sum(closed)}")
    return closed

def interpolate_image_stack(image_stack, scaling_factor=0.5):
    """Scales a 3D image stack using spline interpolation."""
    scaled = scipy.ndimage.zoom(image_stack, (scaling_factor, scaling_factor, scaling_factor), order=2)
    logger.info(f"Scaled image stack to shape {scaled.shape}.")
    return scaled

def find_largest_cluster(binary_image_stack):
    """Finds the largest connected voxel cluster."""
    labels, num_clusters = measure.label(binary_image_stack, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.ravel())
    if len(cluster_sizes) <= 1:
        raise ValueError("No clusters found.")
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = labels == largest_cluster_label
    logger.info(f"Found largest cluster with size {cluster_sizes[largest_cluster_label]}.")
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]

def save_to_tiff_stack(image_stack, filename):
    """Saves a 3D image stack as a TIFF file."""
    try:
        tiff.imwrite(filename, image_stack.astype(np.uint8))
        logger.info(f"Saved TIFF stack to {filename}.")
    except Exception as e:
        logger.error("Failed to save TIFF stack: %s", e)

def save_raw_tiff_stack(image_stack, filename):
    """
    Saves the TIFF stack before binarization.

    Args:
        image_stack (image_stack (numpy.ndarray): Stack of images.
        filename (str): Location to save the file.
    """
    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Raw data stack saved to: {filename}")
    except Exception as e:
        logger.error(f"Error saving raw data stack: {e}")

