import logging
import ciclope.core.tetraFE
import numpy as np
from skimage import morphology, filters, measure
import scipy.ndimage
import tifffile as tiff
from PIL import Image
import vtk
import meshio
import os
from multiprocessing import Pool

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


def load_tiff_stream_lazy(filepath):
    """Lädt einen TIFF-Stream schrittweise, um Speicher zu sparen."""
    try:
        with tiff.TiffFile(filepath) as tif:
            image_stack = np.array([page.asarray() for page in tif.pages])
        logger.info(f"Lazy-loaded TIFF stream with shape {image_stack.shape} from {filepath}")
        return image_stack
    except Exception as e:
        logger.error(f"Error loading TIFF stream lazily {filepath}: {e}")
        return None


def save_tiff_in_chunks(image_stack, filename, chunk_size=100):
    """Speichert ein großes 3D-Bild als TIFF-Stack in Chunks."""
    try:
        with tiff.TiffWriter(filename, bigtiff=True) as tif:
            for i in range(0, image_stack.shape[0], chunk_size):
                tif.write(image_stack[i:i + chunk_size])
        logger.info(f"Saved TIFF stack in chunks to {filename}.")
        print("saved in Chunks")
    except Exception as e:
        logger.error(f"Error saving TIFF stack in chunks: {e}")

def save_to_tiff_stack(image_stack, filename):
    """Speichert ein 3D-Bild als TIFF-Stack."""
    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Saved TIFF stack to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack: {e}")