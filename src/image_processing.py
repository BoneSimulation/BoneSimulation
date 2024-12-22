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

    # Debugging vor Verarbeitung
    logger.info(f"Input stack stats - Min: {data_array.min()}, Max: {data_array.max()}, Shape: {data_array.shape}")

    threshold = filters.threshold_otsu(data_array.flatten())
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold

    # Debugging nach Verarbeitung
    logger.info(f"Blurred stack stats - Min: {blurred.min()}, Max: {blurred.max()}, Shape: {blurred.shape}")
    logger.info(f"Binary image active pixels: {np.sum(binary)}")
    logger.info(f"Threshold applied: {threshold:.2f}")
    return blurred, binary, threshold


def apply_morphological_closing(binary_images):
    """Performs morphological closing on binary images."""
    closed = morphology.closing(binary_images, morphology.ball(3))

    # Log the number of active pixels after closing
    logger.debug(f"Performed morphological closing. Active pixels: {np.sum(closed)}")

    return closed


def interpolate_image_stack(image_stack, scaling_factor=0.5):
    """Scales a 3D image stack using spline interpolation."""
    # Log values before scaling
    logger.info(f"Before scaling: Min={image_stack.min()}, Max={image_stack.max()}")

    # Skaliere und konvertiere von Boolean zu Integer (True=1, False=0)
    scaled = scipy.ndimage.zoom(image_stack.astype(np.float32), (scaling_factor, scaling_factor, scaling_factor),
                                order=2)
    scaled = (scaled * 255).astype(np.uint8)  # Wandle das Bild zurück in 0 oder 1 (binär)

    # Log values after scaling
    logger.info(f"After scaling: Min={scaled.min()}, Max={scaled.max()}")

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


import tifffile as tiff

def save_to_tiff_stack(image_stack, filename):
    """Speichert ein 3D-Bildstack als TIFF-Datei."""
    try:
        # Debugging vor dem Speichern
        logger.info(f"Stack stats before saving - Min: {image_stack.min()}, Max: {image_stack.max()}, Shape: {image_stack.shape}")

        # Sicherstellen, dass der Stack richtig skaliert ist
        if image_stack.dtype != np.uint8:
            logger.warning("Converting image stack to uint8 format.")
            image_stack = (image_stack * 255).astype(np.uint8)

        logger.info(f"Saving stack to {filename}.")
        tiff.imwrite(filename, image_stack, photometric="minisblack")

        # Validierung nach dem Speichern
        reloaded_stack = tiff.imread(filename)
        logger.info(f"Reloaded stack stats - Min: {reloaded_stack.min()}, Max: {reloaded_stack.max()}, Shape: {reloaded_stack.shape}")
        logger.info(f"Saved TIFF stack to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack: {e}")



def save_raw_tiff_stack(image_stack, filename):
    """Saves the TIFF stack before binarization."""
    # Log values before saving
    logger.info(f"Raw image stack range: Min={image_stack.min()}, Max={image_stack.max()}")

    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Raw data stack saved to: {filename}")
    except Exception as e:
        logger.error(f"Error saving raw data stack: {e}")


from skimage.measure import label


def extract_largest_cluster(binary_stack):
    """Extracts the largest connected component (cluster) from the binary image stack."""
    labeled_stack, num_labels = label(binary_stack, connectivity=3, return_num=True)
    largest_cluster_label = np.argmax(np.bincount(labeled_stack.flat)[1:]) + 1  # Ignoriere Hintergrund (0)

    largest_cluster = (labeled_stack == largest_cluster_label).astype(np.uint8)
    logger.info(f"Largest cluster found: {np.sum(largest_cluster)} voxels")

    return largest_cluster

import imageio

def save_largest_cluster_stack(largest_cluster, filename):
    """Speichert den größten Cluster als TIFF-Datei."""
    try:
        # Debugging vor dem Speichern
        logger.info(f"Largest cluster stats before saving - Min: {largest_cluster.min()}, Max: {largest_cluster.max()}, Shape: {largest_cluster.shape}")

        # Sicherstellen, dass der Cluster korrekt skaliert ist
        if largest_cluster.dtype != np.uint8:
            logger.warning("Converting largest cluster to uint8 format.")
            largest_cluster = (largest_cluster * 255).astype(np.uint8)

        logger.info(f"Saving largest cluster to {filename}.")
        imageio.mimwrite(filename, largest_cluster, format="TIFF", bigtiff=True)

        # Validierung nach dem Speichern
        reloaded_cluster = tiff.imread(filename)
        logger.info(f"Reloaded cluster stats - Min: {reloaded_cluster.min()}, Max: {reloaded_cluster.max()}, Shape: {reloaded_cluster.shape}")
        logger.info(f"Saved largest cluster to {filename}.")
    except Exception as e:
        logger.error(f"Error saving largest cluster: {e}")
