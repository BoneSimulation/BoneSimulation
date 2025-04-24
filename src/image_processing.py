# image_processing.py

import logging
import numpy as np
from skimage import morphology, filters
import scipy.ndimage

logger = logging.getLogger(__name__)


def process_images_globally(data_array):
    """Wendet Otsu-Thresholding an, erzeugt binäre Bilder und gibt den Schwellwert zurück."""
    if data_array.size == 0:
        raise ValueError("Input image stack is empty.")

    # Wir berechnen den Threshold auf einem reduzierten Sample, um Speicher zu sparen
    sample_size = min(1000000, data_array.size)
    sample_indices = np.random.choice(data_array.size, sample_size, replace=False)
    sample = data_array.flatten()[sample_indices]

    threshold = filters.threshold_otsu(sample)
    logger.info(f"Otsu-Threshold berechnet: {threshold:.2f}")

    # Speichereffiziente Gaußsche Glättung
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold

    logger.info(f"Binärisierung abgeschlossen mit Schwellwert: {threshold:.2f}")
    return blurred, binary, threshold


def apply_morphological_closing(binary_images, ball_radius=3):
    """Führt morphologisches Closing durch, um kleine Lücken zu schließen."""
    logger.info(f"Starte morphologisches Closing mit Radius {ball_radius}...")

    # Erstelle den Ball-Strukturelement für das Closing
    ball = morphology.ball(ball_radius)

    # Für sehr große Bilder verwenden wir slice-by-slice Closing statt 3D
    if binary_images.shape[0] > 100:
        logger.info("Großes Volumen erkannt, führe Closing Schicht für Schicht durch...")
        result = np.zeros_like(binary_images)

        for i in range(binary_images.shape[0]):
            # 2D Closing für jede Schicht
            result[i] = morphology.closing(binary_images[i], morphology.disk(ball_radius))

            # Fortschritt loggen
            if i % 20 == 0:
                logger.info(f"Closing: {i}/{binary_images.shape[0]} Schichten verarbeitet")
    else:
        # 3D Closing für kleinere Volumen
        result = morphology.closing(binary_images, ball)

    logger.info("Morphologisches Closing abgeschlossen.")
    return result


def interpolate_image_stack(image_stack, scaling_factor=0.5, chunk_size=20):
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
    logger.info(f"Starte Interpolation mit Skalierungsfaktor {scaling_factor}...")

    new_shape = tuple(int(dim * scaling_factor) for dim in image_stack.shape)
    scaled_stack = np.zeros(new_shape, dtype=np.uint8)  # Speicher sparen mit uint8

    total_chunks = (image_stack.shape[0] + chunk_size - 1) // chunk_size

    for i in range(0, image_stack.shape[0], chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, image_stack.shape[0])

        logger.info(f"Interpoliere Chunk {i // chunk_size + 1}/{total_chunks}...")

        # Chunk extrahieren
        chunk = image_stack[chunk_start:chunk_end].astype(np.float32)

        # Skalierungsfaktoren berechnen
        # Für den z-Faktor berücksichtigen wir das tatsächliche Chunk-Verhältnis
        chunk_scale_z = (scaling_factor * (chunk_end - chunk_start)) / chunk.shape[0]
        zoom_factors = (chunk_scale_z, scaling_factor, scaling_factor)

        # Chunk interpolieren
        scaled_chunk = scipy.ndimage.zoom(chunk, zoom_factors, order=1)  # order=1 für lineare Interpolation (schneller)

        # Zielpositionen im skalierten Array berechnen
        target_start = int(chunk_start * scaling_factor)
        target_end = target_start + scaled_chunk.shape[0]
        target_end = min(target_end, scaled_stack.shape[0])

        # Chunk in das Ergebnisarray einfügen
        scaled_stack[target_start:target_end] = (scaled_chunk[:target_end - target_start] > 0.5).astype(np.uint8)

        logger.info(f"Chunk {i // chunk_size + 1}/{total_chunks} interpoliert.")

    logger.info(f"Interpolation abgeschlossen. Neues Volumen: {scaled_stack.shape}")
    return scaled_stack