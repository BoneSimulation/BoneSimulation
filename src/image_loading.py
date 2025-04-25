# image_loading.py (enhanced with chunk functionality)

import logging
import os
import numpy as np
from PIL import Image
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def load_image(filepath):
    """Loads a single image as a grayscale array."""
    try:
        im = Image.open(filepath).convert("L")
        return np.array(im)
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        return None


def load_images(directory):
    """Loads multiple images and combines them into a 3D array."""
    filepaths = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".tif")])

    if not filepaths:
        raise ValueError("No valid images found in directory.")

    with Pool() as pool:
        images = pool.map(load_image, filepaths)

    images = [img for img in images if img is not None]
    if not images:
        raise ValueError("Failed to load any images.")

    logger.info(f"Loaded {len(images)} images from directory {directory}.")
    return np.array(images)


def load_images_in_chunks(directory, chunk_size=50):
    """
    Generator function that loads images in chunks.

    Args:
        directory: Directory containing the images
        chunk_size: Number of images per chunk

    Yields:
        numpy.ndarray: Chunk of images as a 3D array
    """
    filepaths = sorted([os.path.join(directory, file) for file in os.listdir(directory)
                        if file.lower().endswith((".tif", ".tiff"))])

    if not filepaths:
        raise ValueError(f"No valid images found in directory: {directory}")

    logger.info(f"Total of {len(filepaths)} images found. Processing in chunks of {chunk_size}.")

    for i in range(0, len(filepaths), chunk_size):
        chunk_paths = filepaths[i:i + chunk_size]
        logger.info(
            f"Loading chunk {i // chunk_size + 1}/{(len(filepaths) - 1) // chunk_size + 1} ({len(chunk_paths)} images)")

        with Pool() as pool:
            chunk_images = pool.map(load_image, chunk_paths)

        # Remove failed loads
        chunk_images = [img for img in chunk_images if img is not None]
        if not chunk_images:
            logger.warning(f"No images loaded in chunk {i // chunk_size + 1}. Skipping.")
            continue

        yield np.array(chunk_images)
