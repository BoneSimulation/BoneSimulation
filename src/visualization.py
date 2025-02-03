"""
visualization.py

This file contains functions for visualizing image data, including plotting histograms and displaying slices of 3D image stacks.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(image_array, threshold=None):
    """
    Plots a histogram of pixel values from the image stack.

    This function takes a 3D array representing an image stack, flattens it into a 1D array,
    and generates a histogram of the pixel values. If a global threshold is provided, it is
    marked on the histogram for reference.

    Args:
        image_array (numpy.ndarray): A 3D array representing the image stack, where each pixel
                                     value is typically in the range of 0 to 255.
        threshold (float, optional): A global threshold value to mark on the histogram. If provided,
                                     a vertical line will be drawn at this value.

    Returns:
        None: This function does not return a value; it displays the histogram plot.
    """

    flattened_data = image_array.flatten().astype(np.uint8)

    plt.figure(figsize=(10, 6))
    plt.hist(flattened_data, bins=256, color="blue", alpha=0.7, label="Pixel Values")

    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label=f"Threshold: {threshold:.2f}")

    plt.title("Histogram of Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()


def plot_images(images, title):
    """
    Displays slices of 3D image stacks.

    This function takes a 3D array representing an image stack and displays a specified number
    of slices (up to 10) in a single figure. Each slice is shown in grayscale, and the function
    sets titles for each slice and a main title for the figure.

    Args:
        images (numpy.ndarray): A 3D array representing the image stack, where each slice is
                               a 2D array. The array should have a shape of (num_slices, height, width).
        title (str): The title to be displayed above the figure.

    Returns:
        None: This function does not return a value; it displays the image slices in a plot.
    """

    slices = min(10, images.shape[0])
    fig, axes = plt.subplots(1, slices, figsize=(20, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Slice {i + 1}")
    plt.suptitle(title)
    plt.show()
