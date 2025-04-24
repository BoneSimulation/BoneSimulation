# image_loading.py (ergänzt mit Chunk-Funktionalität)

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


def load_images_in_chunks(directory, chunk_size=50):
    """
    Generator-Funktion, die Bilder in Chunks lädt.

    Args:
        directory: Verzeichnis mit den Bildern
        chunk_size: Anzahl der Bilder pro Chunk

    Yields:
        numpy.ndarray: Chunk mit Bildern als 3D-Array
    """
    filepaths = sorted([os.path.join(directory, file) for file in os.listdir(directory)
                        if file.lower().endswith((".tif", ".tiff"))])

    if not filepaths:
        raise ValueError(f"Keine gültigen Bilder im Verzeichnis gefunden: {directory}")

    logger.info(f"Insgesamt {len(filepaths)} Bilder gefunden. Verarbeite in Chunks von {chunk_size}")

    for i in range(0, len(filepaths), chunk_size):
        chunk_paths = filepaths[i:i + chunk_size]
        logger.info(
            f"Lade Chunk {i // chunk_size + 1}/{(len(filepaths) - 1) // chunk_size + 1} ({len(chunk_paths)} Bilder)")

        with Pool() as pool:
            chunk_images = pool.map(load_image, chunk_paths)

        # Entferne fehlgeschlagene Ladevorgänge
        chunk_images = [img for img in chunk_images if img is not None]
        if not chunk_images:
            logger.warning(f"Keine Bilder im Chunk {i // chunk_size + 1} geladen. Überspringe.")
            continue

        yield np.array(chunk_images)