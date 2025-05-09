# src/file_io.py

import logging
import numpy as np
import tifffile as tiff
import os

logger = logging.getLogger(__name__)


def load_tiff_stream(filepath):
    """Loads a single TIFF stream as a 3D array."""
    try:
        image_stack = tiff.imread(filepath)
        logger.info(f"Loaded TIFF stream with shape {image_stack.shape} from {filepath}")
        return image_stack
    except Exception as e:
        logger.error(f"Error loading TIFF stream {filepath}: {e}")
        return None


def save_tiff_in_chunks(image_stack, filename, chunk_size=100):
    """Saves a large 3D image as a TIFF stack in chunks."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Output directory created: {output_dir}")

        logger.info(f"Saving TIFF stack with shape {image_stack.shape} in chunks of size {chunk_size}...")

        with tiff.TiffWriter(filename, bigtiff=True) as tif:
            for i in range(0, image_stack.shape[0], chunk_size):
                end_idx = min(i + chunk_size, image_stack.shape[0])
                chunk = image_stack[i:end_idx]

                # Convert to uint8 if it is float data to save memory
                if chunk.dtype.kind == 'f':
                    chunk = (chunk * 255).astype(np.uint8)

                tif.write(chunk)
                logger.info(f"Chunk {i // chunk_size + 1}/{(image_stack.shape[0] - 1) // chunk_size + 1} saved.")

        total_size_mb = os.path.getsize(filename) / (1024 * 1024)
        logger.info(f"TIFF stack saved to {filename} ({total_size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Error saving TIFF stack in chunks: {e}")


def load_tiff_in_chunks(filepath, chunk_size=100):
    """
    Generator function to load a TIFF stack in chunks.

    Args:
        filepath: Path to the TIFF file
        chunk_size: Number of images per chunk

    Yields:
        numpy.ndarray: Chunk of the image stack
    """
    try:
        with tiff.TiffFile(filepath) as tif:
            # Total number of images in the stack
            num_images = len(tif.pages)
            logger.info(f"TIFF file with {num_images} images found: {filepath}")

            # Load in chunks
            for i in range(0, num_images, chunk_size):
                end_idx = min(i + chunk_size, num_images)
                chunk = tif.asarray(range(i, end_idx))
                logger.info(f"Chunk {i // chunk_size + 1}/{(num_images - 1) // chunk_size + 1} loaded: "
                            f"Images {i}-{end_idx - 1}")
                yield chunk

    except Exception as e:
        logger.error(f"Error loading TIFF file {filepath} in chunks: {e}")
        raise
