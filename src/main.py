import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from PIL import Image
from skimage import filters
import matplotlib.pyplot as plt
from mayavi import mlab
import warnings
import logging
from multiprocessing import Pool

from src.utils.utils import generate_timestamp, check_os

timestamp = generate_timestamp()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logfile.log')
formatter = logging.Formatter(f'{timestamp}: %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")


def load_image(filepath):
    im = Image.open(filepath).convert("L")
    return np.array(im)


def process_image(image):
    image_blurred = filters.gaussian(image, sigma=2, mode='constant', cval=0)
    binary_image = image_blurred > filters.threshold_otsu(image_blurred)
    return image_blurred, binary_image


def process_and_visualize(directory):
    # Bilder laden und als 3D-Array speichern
    filepaths = [os.path.join(directory, filename) for filename in sorted(os.listdir(directory)) if
                 filename.endswith(".tif")]
    with Pool() as pool:
        data_array = pool.map(load_image, filepaths)

    data_array = np.array(data_array)
    print("Datenarray hat die Form:", data_array.shape)
    with Pool() as pool:
        results = pool.map(process_image, data_array)
    image_blurred_array, binary_image_array = zip(*results)
    image_blurred_array = np.array(image_blurred_array)
    binary_image_array = np.array(binary_image_array)
    plot_images(image_blurred_array, "Überarbeitete Bilder")
    plot_images(binary_image_array, "Binarisierte Bilder")
    plot_histogram(image_blurred_array)

    visualize_3d(image_blurred_array)


def plot_images(image_array, title):
    num_images = image_array.shape[0]
    cols = 3
    rows = (num_images // cols) + (num_images % cols > 0)

    plt.figure(figsize=(15, 5 * rows))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_array[i], cmap='gray')
        plt.title(f'{title} {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/home/mathias/PycharmProjects/BoneSimulation/pictures/bone/images/plot_new_{timestamp}.png')
    plt.show()
    print("plot images were loaded")


def plot_histogram(image_array):
    plt.figure(figsize=(10, 6))
    all_pixel_values = image_array.flatten()
    plt.hist(all_pixel_values, bins=256, range=(0, 1), color='gray', alpha=0.7)
    plt.title('Histogramm der Pixelwerte der überarbeiteten Bilder')
    plt.xlabel('Pixelwerte')
    plt.ylabel('Häufigkeit')
    plt.xlim(0, 1)
    plt.grid()
    plt.savefig(f'/home/mathias/PycharmProjects/BoneSimulation/pictures/plot/plot_binary_{timestamp}.png')
    print("plot histogram were found")


def visualize_3d(image_array):
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
    mlab.contour3d(image_array, contours=8, opacity=0.5, colormap='bone')
    mlab.savefig(f'/home/mathias/PycharmProjects/BoneSimulation/pictures/bone/mesh/plot_3d_{timestamp}.png')
    mlab.show()


if __name__ == "__main__":
    logger.debug("Running")
    print("Running simulation")

    if check_os() == "Windows":
        directory = "..\\BoneSimulation\\data\\dataset"
    elif check_os() == "Linux":
        directory = "/home/mathias/PycharmProjects/BoneSimulation/data/dataset"
    elif check_os() == "MacOS":
        directory = "/Users/mathias/PycharmProjects/BoneSimulation/data/dataset"  # not familiar with macOS, please check
    else:
        print("Unknown OS!")
        directory = None

    if directory is not None:
        print(f"Directory found: {directory}")
        process_and_visualize(directory)
    else:
        print("No valid directory found. Exiting.")
