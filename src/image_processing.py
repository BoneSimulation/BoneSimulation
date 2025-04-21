import logging
import numpy as np
from skimage import morphology, filters
import scipy.ndimage

logger = logging.getLogger(__name__)


def process_images_globally(data_array):
    """Wendet Otsu-Thresholding an, erzeugt binäre Bilder und gibt den Schwellwert zurück."""
    if data_array.size == 0:
        raise ValueError("Input image stack is empty.")

    threshold = filters.threshold_otsu(data_array.flatten())
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold
    logger.info(f"Threshold applied: {threshold:.2f}")

    return blurred, binary, threshold


def apply_morphological_closing(binary_images):
    """Führt morphologisches Closing durch, um kleine Lücken zu schließen."""
    return morphology.closing(binary_images, morphology.ball(3))


def interpolate_image_stack(image_stack, scaling_factor=0.5, chunk_size=100):
    """
    Interpoliert das Bild mittels Spline-Interpolation in kleineren Chunks,
    um Speicherprobleme zu vermeiden.

    Args:
        image_stack (np.ndarray): 3D-Bildstapel (z, y, x).
        scaling_factor (float): Skalierungsfaktor für die Interpolation.
        chunk_size (int): Größe der Blöcke für die Verarbeitung (z-Richtung).

    Returns:
        np.ndarray: Interpoliertes 3D-Bild als Binärmaske.
    """
    new_shape = tuple(int(dim * scaling_factor) for dim in image_stack.shape)
    scaled_stack = np.zeros(new_shape, dtype=np.float32)  # Speicher sparen

    for i in range(0, image_stack.shape[0], chunk_size):
        chunk = image_stack[i:i + chunk_size].astype(np.float32)
        zoom_factors = (scaling_factor,) * 3
        scaled_chunk = scipy.ndimage.zoom(chunk, zoom_factors, order=2)

        end_idx = min(i + scaled_chunk.shape[0], new_shape[0])
        scaled_stack[i:end_idx, :, :] = scaled_chunk[:end_idx - i, :, :]

    return (scaled_stack > 0.5).astype(np.uint8)
