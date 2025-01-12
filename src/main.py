"""
main.py

This is the main script for processing and visualizing image data. It orchestrates the loading, processing, and saving of images, as well as logging the workflow.
"""

import logging
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from vtkmodules.util.numpy_support import numpy_to_vtk

from src.utils.utils import generate_timestamp, check_os
from src.image_processing import (
    load_images,
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
    find_largest_cluster,
    save_to_tiff_stack,
    save_raw_tiff_stack
)
from src.visualization import plot_histogram, plot_images

# Generate timestamp for filenames
timestamp = generate_timestamp()

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Change level from DEBUG to INFO
                    format=f"%(asctime)s: %(levelname)s : %(name)s : %(message)s",
                    handlers=[
                        logging.FileHandler("logfile.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

# Suppress debug information from third-party libraries
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("tifffile").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def get_base_path():
    """Returns the base path for the dataset."""
    os_type = check_os()
    logger.info(f"Detected OS: {os_type}")
    return "data/dataset"


from scipy.ndimage import gaussian_filter


def smooth_image_stack(image_stack, sigma=1.0):
    """
    Smoothes a 3D image stack with a Gaussian filter.

    Args:
        image_stack (numpy.ndarray): 3D image stack.
        sigma (float): Standard deviation for the Gaussian filter.

    Returns:
        numpy.ndarray: Smoothed 3D image stack.
    """
    logger.info(f"Applying Gaussian smoothing with sigma={sigma}.")
    smoothed_stack = gaussian_filter(image_stack, sigma=sigma)
    logger.info(f"Smoothed image stack stats - Min: {smoothed_stack.min()}, Max: {smoothed_stack.max()}")

    return smoothed_stack


import vtk

def save_as_vtk(image_stack, output_path):
    """
    Saves a 3D image stack as a VTK file.

    Args:
        image_stack (numpy.ndarray): 3D image stack (z, y, x).
        output_path (str): Path to the output VTK file.
    """
    logger.info(f"Saving image stack as VTK to {output_path}.")

    # Convert the numpy array to a VTK-compatible array
    vtk_data_array = numpy_to_vtk(image_stack.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    # Create a VTK ImageData object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image_stack.shape[2], image_stack.shape[1], image_stack.shape[0])  # (x, y, z)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)  # Pixel spacing (adjustable)
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # Write the VTK file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(vtk_image)
    writer.Write()

    logger.info(f"VTK file saved successfully to {output_path}.")


def process_and_visualize(directory):
    """
    Main function for processing and visualizing images.
    """
    logger.info("Starting processing and visualization...")

    # Load images
    data_array = load_images(directory)
    logger.info(f"Loaded images: {data_array.shape}")

    # Save raw data stack
    save_raw_tiff_stack(data_array, f"pictures/raw_{timestamp}.tif")

    # Process images with global threshold
    blurred_images, binary_images, global_threshold = process_images_globally(data_array)
    logger.info(f"Global threshold determined: {global_threshold}")

    # Morphological closing
    closed_binary_images = apply_morphological_closing(binary_images)

    # Validate active pixels after morphological closing
    active_pixels = np.sum(closed_binary_images)
    logger.info(f"Number of active pixels after closing: {active_pixels}")

    if active_pixels == 0:
        logger.error("No active pixels found after morphological closing.")
        return

    # Interpolation
    interpolated_stack = interpolate_image_stack(closed_binary_images, scaling_factor=0.5)
    logger.info(f"Interpolated stack created: {interpolated_stack.shape}")

    if np.sum(interpolated_stack) == 0:
        logger.error("Interpolated stack contains no active pixels.")
        return

    # Find clusters
    try:
        largest_cluster, num_clusters, cluster_size = find_largest_cluster(interpolated_stack)
        logger.info(f"Largest cluster found: {cluster_size} voxels")
    except ValueError as e:
        logger.error(f"Error finding the largest cluster: {e}")
        return

    logger.info(f"Processed stack shape before saving: {interpolated_stack.shape}")
    logger.info(f"Largest cluster stack shape before saving: {largest_cluster.shape}")

    # Visualize the largest cluster
    plt.imshow(largest_cluster[:, :, largest_cluster.shape[2] // 2], cmap='gray')
    plt.title("Largest Cluster - Slice")
    plt.show()

    # Visualize the interpolated image
    plt.imshow(interpolated_stack[:, :, interpolated_stack.shape[2] // 2], cmap='gray')
    plt.title("Interpolated Image - Slice")
    plt.show()

    # Save results
    save_to_tiff_stack(largest_cluster.astype(np.uint32), f"pictures/cluster_{timestamp}.tif")
    # save_to_tiff_stack(interpolated_stack, f"pictures/processed_{timestamp}.tif")

    # Visualization
    plot_histogram(interpolated_stack, global_threshold)
    plot_images(interpolated_stack, title="Processed Images")

    logger.info("Processing completed.")


if __name__ == "__main__":
    logger.debug("Running main script.")
    print("Running simulation")

    directory = get_base_path()
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    process_and_visualize(directory)