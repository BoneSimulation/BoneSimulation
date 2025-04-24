# file_io.py

import logging
import numpy as np
import tifffile as tiff
import os

logger = logging.getLogger(__name__)


def load_tiff_stream(filepath):
    """Lädt einen einzelnen TIFF-Stream als 3D-Array."""
    try:
        image_stack = tiff.imread(filepath)
        logger.info(f"Loaded TIFF stream with shape {image_stack.shape} from {filepath}")
        return image_stack
    except Exception as e:
        logger.error(f"Error loading TIFF stream {filepath}: {e}")
        return None


def save_tiff_in_chunks(image_stack, filename, chunk_size=100):
    """Speichert ein großes 3D-Bild als TIFF-Stack in Chunks."""
    try:
        # Stellen Sie sicher, dass das Ausgabeverzeichnis existiert
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Ausgabeverzeichnis erstellt: {output_dir}")

        logger.info(f"Speichere TIFF-Stack mit Shape {image_stack.shape} in Chunks der Größe {chunk_size}...")

        with tiff.TiffWriter(filename, bigtiff=True) as tif:
            for i in range(0, image_stack.shape[0], chunk_size):
                end_idx = min(i + chunk_size, image_stack.shape[0])
                chunk = image_stack[i:end_idx]

                # Konvertiere zu uint8, wenn es float-Daten sind, um Speicher zu sparen
                if chunk.dtype.kind == 'f':
                    chunk = (chunk * 255).astype(np.uint8)

                tif.write(chunk)
                logger.info(f"Chunk {i // chunk_size + 1}/{(image_stack.shape[0] - 1) // chunk_size + 1} gespeichert.")

        total_size_mb = os.path.getsize(filename) / (1024 * 1024)
        logger.info(f"TIFF-Stack gespeichert nach {filename} ({total_size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Error saving TIFF stack in chunks: {e}")


def load_tiff_in_chunks(filepath, chunk_size=100):
    """
    Generator-Funktion zum Laden eines TIFF-Stacks in Chunks.

    Args:
        filepath: Pfad zur TIFF-Datei
        chunk_size: Anzahl der Bilder pro Chunk

    Yields:
        numpy.ndarray: Chunk des Bildstapels
    """
    try:
        with tiff.TiffFile(filepath) as tif:
            # Gesamtzahl der Bilder im Stack
            num_images = len(tif.pages)
            logger.info(f"TIFF-Datei mit {num_images} Bildern gefunden: {filepath}")

            # Laden in Chunks
            for i in range(0, num_images, chunk_size):
                end_idx = min(i + chunk_size, num_images)
                chunk = tif.asarray(range(i, end_idx))
                logger.info(f"Chunk {i // chunk_size + 1}/{(num_images - 1) // chunk_size + 1} geladen: "
                            f"Bilder {i}-{end_idx - 1}")
                yield chunk

    except Exception as e:
        logger.error(f"Fehler beim Laden der TIFF-Datei {filepath} in Chunks: {e}")
        raise