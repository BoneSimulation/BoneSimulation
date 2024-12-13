import os
import logging
from multiprocessing import Pool
import numpy as np
from PIL import Image
from skimage import filters, measure, morphology
import scipy.ndimage
import tifffile as tiff
from src.utils.utils import generate_timestamp

timestamp = generate_timestamp()
logger = logging.getLogger(__name__)

def load_image(filepath):
    """
    Loads an image from the specified file and converts it to a grayscale image.

    Args:
        filepath (str): The path to the image file to be loaded.

    Returns:
        numpy.ndarray: A 2D array representing the grayscale image.
    """
    try:
        im = Image.open(filepath).convert("L")
        logger.info("Loaded image: %s", filepath)
        return np.array(im)
    except (IOError, ValueError) as e:
        logger.error("Error loading image %s: %s", filepath, e)
        return None

def load_images(directory):
    """
    Loads all TIFF images from the specified directory.

    Args:
        directory (str): Path to the directory containing TIFF images.

    Returns:
        numpy.ndarray: A 3D array containing the stacked images.
    """
    logger.info("Loading images from directory: %s", directory)

    filepaths = [
        os.path.join(directory, filename)
        for filename in sorted(os.listdir(directory))
        if filename.endswith(".tif")
    ]

    if not filepaths:
        logger.error("No valid images found in directory: %s", directory)
        raise ValueError(f"No valid images found in directory: {directory}")

    with Pool() as pool:
        data_array = pool.map(load_image, filepaths)

    data_array = [img for img in data_array if img is not None]

    if not data_array:
        logger.error("No images could be successfully loaded from %s", directory)
        raise ValueError(f"No valid images could be successfully loaded from {directory}")

    return np.array(data_array)

def process_images_globally(data_array):
    """
    Applies global Otsu thresholding to the entire dataset and processes the images.

    Args:
        data_array (numpy.ndarray): 3D array containing the image stack.

    Returns:
        tuple: (blurred_images, binary_images, global_threshold)
    """
    if data_array.size == 0:
        raise ValueError("Input image stack is empty or invalid.")

    global_threshold = filters.threshold_otsu(data_array.flatten())
    blurred_images = [filters.gaussian(image, sigma=1) for image in data_array]
    binary_images = [(image > global_threshold).astype(np.uint8) for image in blurred_images]

    return np.array(blurred_images), np.array(binary_images), global_threshold

def apply_morphological_closing(binary_images):
    """
    Applies morphological closing to a stack of binary images.

    Args:
        binary_images (numpy.ndarray): 3D array of binary images.

    Returns:
        numpy.ndarray: Processed binary images after morphological closing.
    """
    closed_images = [morphology.closing(image, morphology.square(3)) for image in binary_images]
    return np.array(closed_images)

def interpolate_image_stack(image_stack, scaling_factor, order=1):
    """
    Interpolates a 3D image stack using the specified scaling factor.

    Args:
        image_stack (numpy.ndarray): The input 3D image stack.
        scaling_factor (float): The scaling factor for resizing.
        order (int): The order of the spline interpolation.

    Returns:
        numpy.ndarray: The resized image stack.
    """
    return scipy.ndimage.zoom(image_stack, (scaling_factor, scaling_factor, scaling_factor), order=order)

def find_largest_cluster(image_stack):
    """
    Identifies the largest cluster in the binary image stack.

    Args:
        image_stack (numpy.ndarray): The binary 3D image stack.

    Returns:
        tuple: (largest_cluster, num_clusters, largest_cluster_size)
    """
    labels, num_clusters = measure.label(image_stack, background=0, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.flatten())
    if len(cluster_sizes) <= 1:
        raise ValueError("No clusters found in the image stack.")

    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = (labels == largest_cluster_label)
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]

def save_to_tiff_stack(image_stack, filename):
    """
    Saves a stack of images as a TIFF file.

    Args:
        image_stack (numpy.ndarray): Stack of images to save.
        filename (str): Path to the output TIFF file.
    """
    try:
        tiff.imwrite(filename, image_stack, photometric='minisblack')
        logger.info(f"Saved image stack to {filename}")
    except Exception as e:
        logger.error(f"Error saving TIFF stack to {filename}: {e}")
