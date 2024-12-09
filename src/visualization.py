# visualization.py

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(image_array):
    """
    Plots a histogram of pixel values from the image stack.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(image_array.flatten(), bins=256, range=(0, 1), color="gray", alpha=0.7)
    plt.title("Histogram of Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def plot_images(image_array, title):
    """
    Plots images in a grid format.
    """
    num_images = image_array.shape[0]
    cols = 3
    rows = (num_images // cols) + (num_images % cols > 0)

    plt.figure(figsize=(15, 5 * rows))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_array[i], cmap="gray")
        plt.title(f"{title} {i + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
