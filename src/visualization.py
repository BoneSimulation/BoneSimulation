"""
visualization.py

This file contains functions for visualizing image data, including plotting histograms and displaying slices of 3D image stacks.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(image_array, threshold=None):
    """
    Plots a histogram of pixel values from the image stack.

    Args:
        image_array (numpy.ndarray): 3D array of image stack.
        threshold (float, optional): Global threshold value to mark on the histogram.
    """
    # 3D-Array in 1D umwandeln und in uint8 konvertieren
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
    """Displays slices of 3D image stacks."""
    slices = min(10, images.shape[0])
    fig, axes = plt.subplots(1, slices, figsize=(20, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Slice {i + 1}")
    plt.suptitle(title)
    plt.show()
