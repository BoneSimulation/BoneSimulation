import logging

import ciclope.core.tetraFE
import numpy as np
from skimage import morphology, filters, measure
import scipy.ndimage
import tifffile as tiff
from PIL import Image
import vtk
from vtk.util import numpy_support
import meshio
import os
from multiprocessing import Pool


logger = logging.getLogger(__name__)

def load_image(filepath):
    """
    Loads a single image from the specified file path and converts it to a grayscale NumPy array.

    This function attempts to open an image file, convert it to grayscale, and return it as a NumPy array.
    If the image cannot be loaded due to an error (e.g., file not found, unsupported format),
    it logs the error and returns None.

    Args:
        filepath (str): The path to the image file to be loaded.

    Returns:
        np.ndarray or None: A grayscale image represented as a NumPy array if successful;
                            otherwise, None if an error occurs.
    """

    try:
        im = Image.open(filepath).convert("L")
        return np.array(im)
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        return None


def load_images(directory):
    """
    Loads all valid TIFF images from the specified directory and returns them as a NumPy array.

    This function scans the given directory for files with a ".tif" extension,
    loads each image in parallel using multiprocessing, and converts them to grayscale NumPy arrays.
    If no valid images are found or if all images fail to load, it raises a ValueError.

    Args:
        directory (str): The path to the directory containing the TIFF images to be loaded.

    Returns:
        np.ndarray: A NumPy array containing the loaded grayscale images. Each image is represented
                        as a 2D array, and the resulting array has a shape of (num_images, height, width).

    Raises:
        ValueError: If no valid images are found in the directory or if all images fail to load.
    """

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
    """
    Processes a stack of images using global Otsu thresholding to generate blurred and binary images.

    This function takes a 3D NumPy array representing a stack of images, applies Gaussian
    blurring to reduce noise, and then uses Otsu's method to determine a global threshold
    for binarization. The function returns the blurred images, the binary mask, and the
    calculated threshold value.

    Args:
        data_array (np.ndarray): A 3D NumPy array containing the image stack to be processed.
                                 The array should have a shape of (num_images, height, width).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The blurred images as a 3D NumPy array.
            - np.ndarray: The binary mask as a 3D NumPy array, where pixels are True if they
                        exceed the threshold and False otherwise.
            - float: The global threshold value determined by Otsu's method.

    Raises:
        ValueError: If the input image stack is empty.
    """

    if data_array.size == 0:
        raise ValueError("Input image stack is empty.")
    threshold = filters.threshold_otsu(data_array.flatten())
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold
    logger.info(f"Threshold applied: {threshold:.2f}")
    return blurred, binary, threshold


def apply_morphological_closing(binary_images):
    """
    Performs morphological closing on a stack of binary images to fill small holes and connect
    nearby objects.

    This function applies morphological closing using a spherical structuring element with a radius
    of 3 pixels. Closing is useful for enhancing the structure of binary images by closing small
    gaps and holes in the foreground objects.

    Args:
        binary_images (np.ndarray): A 3D NumPy array containing binary images, where each image
                                    is represented as a 2D array. The array should have a shape
                                    of (num_images, height, width).

    Returns:
        np.ndarray: A 3D NumPy array containing the closed binary images, with the same shape as
                    the input array.
    """

    closed = morphology.closing(binary_images, morphology.ball(3))
    return closed


def interpolate_image_stack(image_stack, scaling_factor=0.5):
    """
    Scales a 3D image stack using spline interpolation.

    This function applies spline interpolation to resize a 3D image stack by a specified scaling factor.
    The input image stack is first converted to a float32 type to ensure precision during interpolation.
    The function returns a binary mask where pixels are set to 1 if their value exceeds 0.5 after scaling.

    Args:
        image_stack (np.ndarray): A 3D NumPy array representing the image stack to be scaled.
                                  The array should have a shape of (num_images, height, width).
        scaling_factor (float, optional): The factor by which to scale the image dimensions.
                                           Default is 0.5, which reduces the size by half.

    Returns:
        np.ndarray: A 3D NumPy array containing the scaled binary images, where each pixel is
                    represented as an 8-bit unsigned integer (0 or 1).
    """

    scaled = scipy.ndimage.zoom(image_stack.astype(np.float32),
                                (scaling_factor, scaling_factor, scaling_factor),
                                order=2)
    binary_scaled = (scaled > 0.5).astype(np.uint8)
    return binary_scaled


def find_largest_cluster(binary_image_stack):
    """
    Finds the largest connected voxel cluster in a binary image stack.

    This function labels connected components in the binary image stack and identifies the largest
    cluster based on the number of connected voxels. It uses a connectivity of 1 to define
    adjacency. The function returns a binary mask of the largest cluster, the total number of clusters
    found, and the size of the largest cluster.

    Args:
        binary_image_stack (np.ndarray): A 3D NumPy array representing the binary image stack,
                                         where each voxel is either 0 or 1. The array should have
                                         a shape of (num_images, height, width).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A binary mask of the largest cluster, with the same shape as the input array.
            - int: The total number of clusters found in the image stack.
            - int: The size of the largest cluster (number of connected voxels).
    """

    labels, num_clusters = measure.label(binary_image_stack, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.ravel())
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = labels == largest_cluster_label
    logger.info(f"Found largest cluster with size {cluster_sizes[largest_cluster_label]}.")
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]


def save_to_tiff_stack(image_stack, filename):
    """
    Saves a 3D image stack as a TIFF file.

    This function takes a 3D NumPy array representing an image stack and saves it to a specified
    TIFF file. The function uses the 'minisblack' photometric interpretation, which is suitable
    for grayscale images. If the save operation fails, it logs an error message.

    Args:
        image_stack (np.ndarray): A 3D NumPy array representing the image stack to be saved.
                                  The array should have a shape of (num_images, height, width).
        filename (str): The path to the output TIFF file where the image stack will be saved.

    Returns:
        None: This function does not return a value; it either saves the image stack to a file
              or logs an error if the operation fails.
    """

    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Saved TIFF stack to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack: {e}")


def save_raw_tiff_stack(image_stack, filename):
    """
    Saves the TIFF stack of raw image data before binarization.

    This function takes a 3D NumPy array representing an image stack and saves it to a specified
    TIFF file. The function uses the 'minisblack' photometric interpretation, which is suitable
    for grayscale images. If the save operation fails, it logs an error message.

    Args:
        image_stack (np.ndarray): A 3D NumPy array representing the raw image stack to be saved.
                                  The array should have a shape of (num_images, height, width).
        filename (str): The path to the output TIFF file where the raw image stack will be saved.

    Returns:
        None: This function does not return a value; it either saves the image stack to a file
              or logs an error if the operation fails.
    """

    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Raw data stack saved to: {filename}")
    except Exception as e:
        logger.error(f"Error saving raw data stack: {e}")


def marching_cubes(binary_stack, spacing=(1, 1, 1)):
    """
    Generates a surface mesh from a binary image stack using the Marching Cubes algorithm.

    This function takes a 3D binary image stack and applies the Marching Cubes algorithm to
    extract a surface mesh. The input binary stack is normalized to ensure that the values
    are within the range [0, 1]. The function returns the vertices and faces of the generated
    mesh.

    Args:
        binary_stack (np.ndarray): A 3D NumPy array representing the binary image stack,
                                    where each voxel is either 0 or 1. The array should have
                                    a shape of (num_images, height, width).
        spacing (tuple, optional): A tuple representing the spacing between voxels in each
                                    dimension (x, y, z). Default is (1, 1, 1).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of vertices of the generated mesh.
            - np.ndarray: An array of faces of the generated mesh, where each face is defined
                            by indices into the vertex array.
        """

    normalized_stack = (binary_stack - binary_stack.min()) / (binary_stack.max() - binary_stack.min())
    verts, faces, _, _ = measure.marching_cubes(normalized_stack, level=0.5, spacing=spacing)
    return verts, faces


def save_mesh_as_vtk(verts, faces, output_filename):
    """
    Saves a mesh represented by vertices and faces as a VTK file.

    This function takes the vertices and faces of a mesh and writes them to a specified
    VTK file using the `meshio` library. The mesh is saved as a collection of triangular
    cells defined by the provided faces.

    Args:
        verts (np.ndarray): An array of vertices of the mesh, where each vertex is represented
                            by its coordinates (x, y, z).
        faces (np.ndarray): An array of faces of the mesh, where each face is defined by
                            indices into the vertex array. Each face should correspond to a
                            triangle.
        output_filename (str): The path to the output VTK file where the mesh will be saved.

    Returns:
        None: This function does not return a value; it saves the mesh to a file and logs
                the operation.
        """

    cells = [("triangle", faces)]
    meshio.write_points_cells(output_filename, verts, cells)
    logger.info(f"Mesh saved as VTK file: {output_filename}")


import logging
import numpy as np
import os

logger = logging.getLogger(__name__)


# Hilfsklasse, um Mesh-Daten in einem Objekt zu kapseln (sodass vars() funktioniert)
class MeshDataContainer:
    """
    Ein einfacher Container, der Mesh-Daten hält und ein __dict__-Attribut besitzt.
    """

    def __init__(self, points, cells, point_data, cell_data, field_data, point_sets, cell_sets):
        self.points = points
        self.cells = cells
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data
        self.point_sets = point_sets
        self.cell_sets = cell_sets


# Hilfsklasse, die ein CellBlock-ähnliches Objekt simuliert
class CellBlock:
    """
    Ein einfacher Container für Zell-Daten, der ein __dict__-Attribut besitzt.
    """

    def __init__(self, cell_type, data):
        self.type = cell_type
        self.data = data


def generate_tetrahedral_mesh(binary_volume: np.ndarray, voxel_size: float, output_filename: str):
    """
    Creates a tetrahedral mesh from a binary 3D image and saves it as a VTK file.

    This function takes a binary volume represented as a 3D NumPy array and generates a
    tetrahedral mesh using specified voxel size parameters. The function logs the parameters
    used for mesh creation and handles any exceptions that may occur during the process.

    Args:
        binary_volume (np.ndarray): A binary 3D image stack, where voxels are either 0 or 1.
        voxel_size (float): The size of a voxel, which influences the mesh resolution.
        output_filename (str): The path to the output VTK file where the tetrahedral mesh will be saved.

    Returns:
        mesh: The generated tetrahedral mesh object, or None if an error occurs.
    """
    try:
        import ciclope.core.tetraFE as tetraFE

        # Create a voxel size array (iterable)
        vs = np.ones(3) * voxel_size

        # Parameters for mesh generation
        mesh_size_factor = 0.8
        max_facet_distance = mesh_size_factor * np.min(vs)
        max_cell_circumradius = 2 * mesh_size_factor * np.min(vs)

        logger.info(f"Creating tetrahedral mesh with voxel_size={voxel_size}, mesh_size_factor={mesh_size_factor}, "
                    f"max_facet_distance={max_facet_distance}, max_cell_circumradius={max_cell_circumradius}")

        # Debugging the input parameters
        logger.debug(f"binary_volume shape: {binary_volume.shape}, type: {type(binary_volume)}")
        logger.debug(f"voxel_size array: {vs}, type: {type(vs)}")

        # Generate the tetrahedral mesh
        mesh = tetraFE.cgal_mesh(binary_volume, vs, 'tetra',
                                 max_facet_distance,
                                 max_cell_circumradius)

        # Save the tetrahedral mesh as a VTK file
        mesh.write(output_filename)
        logger.info(f"Tetrahedral mesh saved to {output_filename}")

        # Bereite Mesh-Daten in einem Container vor, damit tetraFE.mesh2tetrafe korrekt darauf zugreifen kann.
        # Anstatt ein Tupel ("tetra", ...) zu übergeben, erstellen wir ein CellBlock-Objekt.
        mesh_data_fixed = MeshDataContainer(
            points=mesh.points,
            cells=[CellBlock("tetra", mesh.cells[0].data)],
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=mesh.field_data,
            point_sets=mesh.point_sets,
            cell_sets=mesh.cell_sets,
        )

        # Define template and output for CalculiX input file conversion
        input_template = "C:\\Users\\Mathias\\Documents\\BoneSimulation\\data\\bone.inp"
        filename_putput = "test_pictures/final_bone.inp"

        # Convert the tetrahedral mesh to a CalculiX input file
        tetraFE.mesh2tetrafe(mesh_data_fixed, input_template, filename_putput)
        logger.info(f"CalculiX input file saved to {filename_putput}")

        return mesh

    except ImportError as e:
        logger.error(f"Failed to import tetraFE module: {e}")
    except Exception as e:
        logger.error(f"Error generating tetrahedral mesh: {e}")
    return None


def write_tetrahedral_mesh(mesh, output_filename: str):
    """
    Saves the tetrahedral mesh as a VTK file and converts it to a specified input format.

    Args:
        mesh: The tetrahedral mesh to be saved and converted.
        output_filename (str): The path to the output VTK file where the mesh will be saved.

    Returns:
        None
    """
    try:
        mesh.write(output_filename)
        logger.info(f"Tetrahedral mesh for simulation saved to {output_filename}")

        # Convert the mesh to a CalculiX input file using a template
        input_template = "C:\\Users\\Mathias\\Documents\\BoneSimulation\\data\\bone.inp"
        ciclope.core.tetraFE.mesh2tetrafe(mesh, input_template, output_filename)
    except Exception as e:
        logger.error(f"Error converting tetrahedral mesh: {e}")
