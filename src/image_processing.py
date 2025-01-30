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

logger = logging.getLogger(__name__)

def load_image(filepath):
    """Loads a single image as a grayscale NumPy array."""
    try:
        im = Image.open(filepath).convert("L")
        return np.array(im)
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        return None

def load_images(directory):
    """Loads all .tif images in a directory into a 3D NumPy array."""
    import os
    from multiprocessing import Pool

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
    threshold = filters.threshold_otsu(data_array.flatten())
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold
    logger.info(f"Threshold applied: {threshold:.2f}")
    return blurred, binary, threshold

def apply_morphological_closing(binary_images):
    """Performs morphological closing on binary images."""
    closed = morphology.closing(binary_images, morphology.ball(3))
    return closed

def interpolate_image_stack(image_stack, scaling_factor=0.5):
    """Scales a 3D image stack using spline interpolation."""
    scaled = scipy.ndimage.zoom(image_stack.astype(np.float32),
                                (scaling_factor, scaling_factor, scaling_factor),
                                order=2)
    binary_scaled = (scaled > 0.5).astype(np.uint8)
    return binary_scaled

def find_largest_cluster(binary_image_stack):
    """Finds the largest connected voxel cluster."""
    labels, num_clusters = measure.label(binary_image_stack, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.ravel())
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = labels == largest_cluster_label
    logger.info(f"Found largest cluster with size {cluster_sizes[largest_cluster_label]}.")
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]

def save_to_tiff_stack(image_stack, filename):
    """Saves a 3D image stack as a TIFF file."""
    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Saved TIFF stack to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack: {e}")

def save_raw_tiff_stack(image_stack, filename):
    """Saves the TIFF stack before binarization."""
    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Raw data stack saved to: {filename}")
    except Exception as e:
        logger.error(f"Error saving raw data stack: {e}")

def marching_cubes(binary_stack, spacing=(1, 1, 1)):
    """Generates a surface mesh using Marching Cubes."""
    normalized_stack = (binary_stack - binary_stack.min()) / (binary_stack.max() - binary_stack.min())
    verts, faces, _, _ = measure.marching_cubes(normalized_stack, level=0.5, spacing=spacing)
    return verts, faces

def save_mesh_as_vtk(verts, faces, output_filename):
    """Saves a mesh as a VTK file."""
    cells = [("triangle", faces)]
    meshio.write_points_cells(output_filename, verts, cells)
    logger.info(f"Mesh saved as VTK file: {output_filename}")

def generate_tetrahedral_mesh(binary_volume: np.ndarray, voxel_size: float, output_filename: str):
    """
    Erstellt ein Tetraedernetz aus einem binarisierten 3D-Bild und speichert es als VTK-Datei.

    Args:
        binary_volume (np.ndarray): Binarisierter 3D-Bildstack.
        voxel_size (float): Größe eines Voxels.
        output_filename (str): Pfad zur Ausgabe-VTK-Datei.
    """
    try:
        import ciclope.core.tetraFE as tetraFE

        # Voxelgrößenarray erstellen
        vs = np.ones(3) * voxel_size

        # Parameter für die Mesh-Erstellung
        mesh_size_factor = 0.8
        max_facet_distance = mesh_size_factor * np.min(vs)
        max_cell_circumradius = 2 * mesh_size_factor * np.min(vs)

        logger.info(f"Creating tetrahedral mesh with voxel_size={voxel_size}, mesh_size_factor={mesh_size_factor}, "
                    f"max_facet_distance={max_facet_distance}, max_cell_circumradius={max_cell_circumradius}")

        # Debugging der Eingabeparameter
        logger.debug(f"binary_volume shape: {binary_volume.shape}, type: {type(binary_volume)}")
        logger.debug(f"voxel_size array: {vs}, type: {type(vs)}")

        # Erstellen des Tetraedernetzes
        mesh = tetraFE.cgal_mesh(binary_volume, vs, 'tetra',
                                 max_facet_distance,
                                 max_cell_circumradius)

        # Speichern des Tetraedernetzes als VTK-Datei
        input_template = "C:\\Users\\Mathias\\Documents\\BoneSimulation\\data\\bone.inp"
        filename_putput = "final_bone.inp"
        mesh.write(output_filename)
        logger.info(f"Tetrahedral mesh saved to {output_filename}")
        ciclope.core.tetraFE.mesh2tetrafe(mesh, input_template, filename_putput)

    except ImportError as e:
        logger.error(f"Failed to import tetraFE module: {e}")
    except Exception as e:
        logger.error(f"Error generating tetrahedral mesh: {e}")
