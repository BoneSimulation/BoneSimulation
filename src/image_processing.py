"""
image_processing.py

This file contains functions for loading, processing, and saving image data. It includes operations such as thresholding, morphological processing, and cluster analysis.
"""

import logging
import numpy as np
from multiprocessing import Pool
from skimage import morphology, filters, measure
import scipy.ndimage
import tifffile as tiff
from PIL import Image
import vtk
from vtk.util import numpy_support
import os
from skimage.measure import label
import imageio


logger = logging.getLogger(__name__)

def load_image(filepath):
    """Loads a single image as a grayscale NumPy array."""
    try:
        im = Image.open(filepath).convert("L")
        return np.array(im)
    except Exception as e:
        logger.error("Error loading image %s: %s", filepath, e)
        return None


def load_images(directory):
    """Loads all .tif images in a directory into a 3D NumPy array."""
    filepaths = sorted([
        os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".tif")
    ])
    if not filepaths:
        raise ValueError("No valid images found in directory.")
    with Pool() as pool:
        images = pool.map(load_image, filepaths)
    images = [img for img in images if img is not None]
    if not images:
        raise ValueError("Failed to load any images.")
    logger.info(f"Loaded {len(images)} images from directory {directory}.")
    return np.array(images)


def process_images_globally(data_array):
    """Processes images using global Otsu thresholding."""
    if data_array.size == 0:
        raise ValueError("Input image stack is empty.")

    # Debugging before processing
    logger.info(f"Input stack stats - Min: {data_array.min()}, Max: {data_array.max()}, Shape: {data_array.shape}")

    threshold = filters.threshold_otsu(data_array.flatten())
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold

    # Debugging after processing
    logger.info(f"Blurred stack stats - Min: {blurred.min()}, Max: {blurred.max()}, Shape: {blurred.shape}")
    logger.info(f"Binary image active pixels: {np.sum(binary)}")
    logger.info(f"Threshold applied: {threshold:.2f}")
    return blurred, binary, threshold


def apply_morphological_closing(binary_images):
    """Performs morphological closing on binary images."""
    closed = morphology.closing(binary_images, morphology.ball(3))

    # Log the number of active pixels after closing
    logger.debug(f"Performed morphological closing. Active pixels: {np.sum(closed)}")

    return closed


def interpolate_image_stack(image_stack, scaling_factor=0.5, threshold=0.5):
    """
    Scales a 3D binary image stack using spline interpolation and applies a threshold to binarize the result.

    Args:
        image_stack (numpy.ndarray): 3D binary image stack.
        scaling_factor (float): Scaling factor for interpolation.
        threshold (float): Threshold value to binarize the interpolated image (default is 0.5).

    Returns:
        numpy.ndarray: Interpolated and binarized 3D image stack.
    """
    logger.info(f"Before scaling: Min={image_stack.min()}, Max={image_stack.max()}")

    # Scale the image with interpolation (float needed for interpolation)
    scaled = scipy.ndimage.zoom(image_stack.astype(np.float32),
                                (scaling_factor, scaling_factor, scaling_factor),
                                order=2)

    # Binarize the image directly after interpolation
    binary_scaled = (scaled >= threshold).astype(np.uint8)

    logger.info(f"After scaling: Min={binary_scaled.min()}, Max={binary_scaled.max()}")
    logger.info(f"Scaled image stack to shape {binary_scaled.shape}.")

    return binary_scaled


def find_largest_cluster(binary_image_stack):
    """Finds the largest connected voxel cluster."""
    labels, num_clusters = measure.label(binary_image_stack, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.ravel())
    if len(cluster_sizes) <= 1:
        raise ValueError("No clusters found.")
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = labels == largest_cluster_label
    logger.info(f"Found largest cluster with size {cluster_sizes[largest_cluster_label]}.")
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]


def save_to_tiff_stack(image_stack, filename):
    """Saves a 3D image stack as a TIFF file."""
    try:
        # Debugging before saving
        logger.info(f"Stack stats before saving - Min: {image_stack.min()}, Max: {image_stack.max()}, Shape: {image_stack.shape}")

        # Ensure the stack is properly scaled
        if image_stack.dtype != np.uint8:
            logger.warning("Converting image stack to uint8 format.")
            image_stack = (image_stack * 255).astype(np.uint8)

        logger.info(f"Saving stack to {filename}.")
        tiff.imwrite(filename, image_stack, photometric="minisblack")

        # Validation after saving
        reloaded_stack = tiff.imread(filename)
        logger.info(f"Reloaded stack stats - Min: {reloaded_stack.min()}, Max: {reloaded_stack.max()}, Shape: {reloaded_stack.shape}")
        logger.info(f"Saved TIFF stack to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack: {e}")


def save_raw_tiff_stack(image_stack, filename):
    """Saves the TIFF stack before binarization."""
    # Log values before saving
    logger.info(f"Raw image stack range: Min={image_stack.min()}, Max={image_stack.max()}")

    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Raw data stack saved to: {filename}")
    except Exception as e:
        logger.error(f"Error saving raw data stack: {e}")


def extract_largest_cluster(binary_stack):
    """Extracts the largest connected component (cluster) from the binary image stack."""
    labeled_stack, num_labels = label(binary_stack, connectivity=3, return_num=True)
    largest_cluster_label = np.argmax(np.bincount(labeled_stack.flat)[1:]) + 1  # Ignore background (0)

    largest_cluster = (labeled_stack == largest_cluster_label).astype(np.uint8)
    logger.info(f"Largest cluster found: {np.sum(largest_cluster)} voxels")

    return largest_cluster


def save_largest_cluster_stack(largest_cluster, filename):
    """Saves the largest cluster as a TIFF file."""
    try:
        # Debugging before saving
        logger.info(f"Largest cluster stats before saving - Min: {largest_cluster.min()}, Max: {largest_cluster.max()}, Shape: {largest_cluster.shape}")

        # Ensure the cluster is correctly scaled
        if largest_cluster.dtype != np.uint8:
            logger.warning("Converting largest cluster to uint8 format.")
            largest_cluster = (largest_cluster * 255).astype(np.uint8)

        logger.info(f"Saving largest cluster to {filename}.")
        imageio.mimwrite(filename, largest_cluster, format="TIFF", bigtiff=True)

        # Validation after saving
        reloaded_cluster = tiff.imread(filename)
        logger.info(f"Reloaded cluster stats - Min: {reloaded_cluster.min()}, Max: {reloaded_cluster.max()}, Shape: {reloaded_cluster.shape}")
        logger.info(f"Saved largest cluster to {filename}.")
    except Exception as e:
        logger.error(f"Error saving largest cluster: {e}")



def numpy2vtk(array: np.ndarray, output_filename: str):
    """
    Convert a 3D NumPy array to a VTK file and save it.

    Parameters:
        array (np.ndarray): A 3D NumPy array.
        output_filename (str): The path where the VTK file will be saved.
    """
    # Ensure the array is 3D
    if array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional.")

    # Get the dimensions of the array
    dimensions = array.shape

    # Create a vtkImageData object
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(dimensions)
    vtk_image_data.SetSpacing(1.0, 1.0, 1.0)  # Adjust spacing as needed
    vtk_image_data.SetOrigin(0.0, 0.0, 0.0)  # Adjust origin as needed

    # Convert the NumPy array to a VTK array
    vtk_array = numpy_support.numpy_to_vtk(num_array=array.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    # Add the VTK array to the vtkImageData object
    vtk_image_data.GetPointData().SetScalars(vtk_array)

    # Write the VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(vtk_image_data)

    # Ensure the file is written
    if not writer.Write():
        raise IOError(f"Failed to write VTK file to {output_filename}")
    logger.info(f"VTK file written to {output_filename}")


from skimage import measure
import numpy as np


def marching_cubes(binary_stack, spacing=(1, 1, 1)):
    """
    Erstellt ein Mesh aus einem binarisierten 3D-Bildstack mittels Marching Cubes.

    Args:
        binary_stack (numpy.ndarray): Binarisierter 3D-Bildstack.
        spacing (tuple): Abstand der Voxels entlang der Achsen.

    Returns:
        verts, faces: Punkte und Flächen des generierten Meshes.
    """
    # Normierung des Volumens auf den Bereich [0, 1]
    normalized_stack = (binary_stack - binary_stack.min()) / (binary_stack.max() - binary_stack.min())

    # Marching Cubes auf der normierten Datenbasis mit level=0.5
    verts, faces, _, _ = measure.marching_cubes(normalized_stack, level=0.5, spacing=spacing)

    return verts, faces
