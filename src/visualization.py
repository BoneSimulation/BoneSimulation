"""
visualization.py

This file contains functions for visualizing image data, including plotting histograms and displaying slices of 3D image stacks.
"""

import matplotlib.pyplot as plt

def plot_histogram(data, threshold):
    """Plots a histogram with a threshold marker."""
    plt.hist(data, bins=256, color="blue", alpha=0.7)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2)
    plt.title("Histogram with Threshold")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
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
