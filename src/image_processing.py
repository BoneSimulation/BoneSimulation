# src/image_processing.py

import logging
import numpy as np
from skimage import morphology, filters
import scipy.ndimage

logger = logging.getLogger(__name__)


def process_images_globally(data_array):
    """Applies Otsu thresholding, generates binary images, and returns the threshold."""
    if data_array.size == 0:
        raise ValueError("Input image stack is empty.")

    # Calculate the threshold on a reduced sample to save memory
    sample_size = min(1000000, data_array.size)
    sample_indices = np.random.choice(data_array.size, sample_size, replace=False)
    sample = data_array.flatten()[sample_indices]

    threshold = filters.threshold_otsu(sample)
    logger.info(f"Otsu threshold calculated: {threshold:.2f}")

    # Memory-efficient Gaussian smoothing
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold

    logger.info(f"Binarization completed with threshold: {threshold:.2f}")
    return blurred, binary, threshold


def apply_morphological_closing(binary_images, ball_radius=3):
    """Performs morphological closing to close small gaps."""
    logger.info(f"Starting morphological closing with radius {ball_radius}...")

    # Create the ball structuring element for closing
    ball = morphology.ball(ball_radius)

    # For very large images, use slice-by-slice closing instead of 3D
    if binary_images.shape[0] > 100:
        logger.info("Large volume detected, performing closing slice by slice...")
        result = np.zeros_like(binary_images)

        for i in range(binary_images.shape[0]):
            # 2D closing for each slice
            result[i] = morphology.closing(binary_images[i], morphology.disk(ball_radius))

            # Log progress
            if i % 20 == 0:
                logger.info(f"Closing: {i}/{binary_images.shape[0]} slices processed")
    else:
        # 3D closing for smaller volumes
        result = morphology.closing(binary_images, ball)

    logger.info("Morphological closing completed.")
    return result


def interpolate_image_stack(image_stack, scaling_factor=0.5, chunk_size=20):
    """
    Interpolates the image using spline interpolation in smaller chunks
    to avoid memory issues.

    Args:
        image_stack (np.ndarray): 3D image stack (z, y, x).
        scaling_factor (float): Scaling factor for interpolation.
        chunk_size (int): Size of the blocks for processing (z-direction).

    Returns:
        np.ndarray: Interpolated 3D image as a binary mask.
    """
    logger.info(f"Starting interpolation with scaling factor {scaling_factor}...")

    new_shape = tuple(int(dim * scaling_factor) for dim in image_stack.shape)
    scaled_stack = np.zeros(new_shape, dtype=np.uint8)  # Save memory with uint8

    total_chunks = (image_stack.shape[0] + chunk_size - 1) // chunk_size

    for i in range(0, image_stack.shape[0], chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, image_stack.shape[0])

        logger.info(f"Interpolating chunk {i // chunk_size + 1}/{total_chunks}...")

        # Extract chunk
        chunk = image_stack[chunk_start:chunk_end].astype(np.float32)

        # Calculate scaling factors
        # For the z-factor, consider the actual chunk ratio
        chunk_scale_z = (scaling_factor * (chunk_end - chunk_start)) / chunk.shape[0]
        zoom_factors = (chunk_scale_z, scaling_factor, scaling_factor)

        # Interpolate chunk
        scaled_chunk = scipy.ndimage.zoom(chunk, zoom_factors, order=1)  # order=1 for linear interpolation (faster)

        # Calculate target positions in the scaled array
        target_start = int(chunk_start * scaling_factor)
        target_end = target_start + scaled_chunk.shape[0]
        target_end = min(target_end, scaled_stack.shape[0])

        # Insert chunk into the result array
        scaled_stack[target_start:target_end] = (scaled_chunk[:target_end - target_start] > 0.5).astype(np.uint8)

        logger.info(f"Chunk {i // chunk_size + 1}/{total_chunks} interpolated.")

    logger.info(f"Interpolation completed. New volume: {scaled_stack.shape}")
    return scaled_stack


def crop_stack_custom(stack, crop: dict):
    """
    Crops a 3D stack by removing specific voxel counts from each border.

    crop = {
        'z': (top, bottom),
        'y': (front, back),
        'x': (left, right)
    }
    """
    z0, z1 = crop.get("z", (0, 0))
    y0, y1 = crop.get("y", (0, 0))
    x0, x1 = crop.get("x", (0, 0))

    z, y, x = stack.shape
    return stack[
           z0: z - z1,
           y0: y - y1,
           x0: x - x1
           ]
