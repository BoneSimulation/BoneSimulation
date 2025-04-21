import logging
import numpy as np
import tifffile as tiff

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
        with tiff.TiffWriter(filename, bigtiff=True) as tif:
            for i in range(0, image_stack.shape[0], chunk_size):
                tif.write(image_stack[i:i + chunk_size])
        logger.info(f"Saved TIFF stack in chunks to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack in chunks: {e}")
