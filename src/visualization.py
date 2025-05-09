# src/visualization.py

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(image_array, threshold=None):
    """Displays the histogram of pixel values of an image stack."""
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
    """Displays up to 10 images from a 3D stack."""
    slices = min(10, images.shape[0])
    fig, axes = plt.subplots(1, slices, figsize=(20, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Slice {i + 1}")
    plt.suptitle(title)
    plt.show()
