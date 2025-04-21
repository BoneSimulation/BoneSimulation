import logging
import os
import numpy as np
from PIL import Image
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def load_image(filepath):
    """Lädt ein einzelnes Bild als Graustufen-Array."""
    try:
        im = Image.open(filepath).convert("L")
        return np.array(im)
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        return None


def load_images(directory):
    """Lädt mehrere Bilder und kombiniert sie zu einem 3D-Array."""
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
