import matplotlib.pyplot as plt

def plot_histogram(image_array, filename=None):
    """
    Plots a histogram of pixel values from the image stack.

    Args:
        image_array (numpy.ndarray): The image stack to analyze.
        filename (str, optional): If provided, saves the plot to the specified file.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(image_array.flatten(), bins=256, color="gray", alpha=0.7)
    plt.title("Histogram of Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid()

    if filename:
        plt.savefig(filename)
        plt.close()
        print(f"Histogram saved to {filename}")
    else:
        plt.show()


def plot_images(image_array, title, filename=None):
    """
    Plots images in a grid format.

    Args:
        image_array (numpy.ndarray): The image stack to visualize.
        title (str): Title for the images.
        filename (str, optional): If provided, saves the plot to the specified file.
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

    if filename:
        plt.savefig(filename)
        plt.close()
        print(f"Images saved to {filename}")
    else:
        plt.show()


def visualize_cluster(image_stack, largest_cluster, filename=None):
    """
    Visualizes the largest cluster overlaid on the original images.

    Args:
        image_stack (numpy.ndarray): The original image stack.
        largest_cluster (numpy.ndarray): The binary mask of the largest cluster.
        filename (str, optional): If provided, saves the plot to the specified file.
    """
    num_images = image_stack.shape[0]
    cols = 3
    rows = (num_images // cols) + (num_images % cols > 0)

    plt.figure(figsize=(15, 5 * rows))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_stack[i], cmap="gray", alpha=0.5)
        plt.imshow(largest_cluster[i], cmap="Reds", alpha=0.5)
        plt.title(f"Cluster Overlay {i + 1}")
        plt.axis("off")

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
        print(f"Cluster visualization saved to {filename}")
    else:
        plt.show()
