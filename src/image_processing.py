import logging
import numpy as np
import os
import tifffile as tiff
import scipy.ndimage
from skimage import morphology, filters, measure
from multiprocessing import Pool
from PIL import Image
import meshio

logger = logging.getLogger(__name__)
# Variable zur Steuerung des Ladeverhaltens
USE_TIFF_STREAM = True  # True: Lade einen einzelnen TIFF-Stream, False: Lade mehrere Bilder

# ---------------- TIFF-Lade- und Speicherfunktionen ----------------
def load_tiff_stream(filepath):
    """Lädt einen einzelnen TIFF-Stream als 3D-Array."""
    try:
        image_stack = tiff.imread(filepath)
        logger.info(f"Loaded TIFF stream with shape {image_stack.shape} from {filepath}")
        return image_stack
    except Exception as e:
        logger.error(f"Error loading TIFF stream {filepath}: {e}")
        return None

def load_tiff_stream_lazy(filepath):
    """Lädt einen TIFF-Stream schrittweise, um Speicher zu sparen."""
    try:
        with tiff.TiffFile(filepath) as tif:
            image_stack = np.array([page.asarray() for page in tif.pages])
        logger.info(f"Lazy-loaded TIFF stream with shape {image_stack.shape} from {filepath}")
        return image_stack
    except Exception as e:
        logger.error(f"Error loading TIFF stream lazily {filepath}: {e}")
        return None

def save_tiff_in_chunks(image_stack, filename, chunk_size=100):
    """Speichert ein großes 3D-Bild als TIFF-Stack in Chunks."""
    try:
        with tiff.TiffWriter(filename, bigtiff=True) as tif:
            for i in range(0, image_stack.shape[0], chunk_size):
                tif.write(image_stack[i:i + chunk_size])
        logger.info(f"Saved TIFF stack in chunks to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack in chunks: {e}")

def save_to_tiff_stack(image_stack, filename):
    """Speichert ein 3D-Bild als TIFF-Stack."""
    try:
        tiff.imwrite(filename, image_stack, photometric="minisblack")
        logger.info(f"Saved TIFF stack to {filename}.")
    except Exception as e:
        logger.error(f"Error saving TIFF stack: {e}")

# ---------------- Bildladefunktionen ----------------
def load_image(filepath):
    """Lädt ein einzelnes Bild als Graustufen-Array."""
    try:
        im = Image.open(filepath).convert("L")
        return np.array(im)
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        return None

# zendodo

def load_images(directory):
    """Lädt mehrere Bilder und kombiniert sie zu einem 3D-Array."""
    filepaths = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".tif")])
    if not filepaths:
        raise ValueError("No valid images found in directory.")
    with Pool() as pool:
        images = pool.map(load_image, filepaths)
    images = [img for img in images if img is not None]
    if not images:
        raise ValueError("Failed to load any images.")
    logger.info(f"Loaded {len(images)} images from directory {directory}.")
    return np.array(images)

# ---------------- Bildverarbeitung ----------------
def process_images_globally(data_array):
    """Anwendet Otsu-Thresholding, erzeugt binäre Bilder und gibt den Schwellwert zurück."""
    if data_array.size == 0:
        raise ValueError("Input image stack is empty.")
    threshold = filters.threshold_otsu(data_array.flatten())
    blurred = filters.gaussian(data_array, sigma=1, preserve_range=True)
    binary = blurred > threshold
    logger.info(f"Threshold applied: {threshold:.2f}")
    return blurred, binary, threshold

def apply_morphological_closing(binary_images):
    """Führt morphologisches Closing durch, um kleine Lücken zu schließen."""
    closed = morphology.closing(binary_images, morphology.ball(1))
    return closed

def interpolate_image_stack(image_stack, scaling_factor=0.5, chunk_size=100):
    """Interpoliert das Bild mittels Spline-Interpolation in Chunks."""
    new_shape = tuple(int(dim * scaling_factor) for dim in image_stack.shape)
    scaled_stack = np.zeros(new_shape, dtype=np.float32)
    for i in range(0, image_stack.shape[0], chunk_size):
        chunk = image_stack[i:i + chunk_size].astype(np.float32)
        zoom_factors = (scaling_factor, scaling_factor, scaling_factor)
        scaled_chunk = scipy.ndimage.zoom(chunk, zoom_factors, order=2)
        end_idx = min(i + scaled_chunk.shape[0], new_shape[0])
        scaled_stack[i:end_idx, :, :] = scaled_chunk[:end_idx - i, :, :]
    return (scaled_stack > 0.5).astype(np.uint8)

# dpi value

def find_largest_cluster(binary_image_stack):
    """Findet den größten verbundenen Cluster im binären Bild."""
    labels, num_clusters = measure.label(binary_image_stack, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.ravel())
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = labels == largest_cluster_label
    logger.info(f"Found largest cluster with size {cluster_sizes[largest_cluster_label]}.")
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]

# ---------------- Mesh-Generierung ----------------
def marching_cubes(binary_stack, spacing=(1, 1, 1)):
    """Erstellt ein Mesh aus dem binären Stack mit dem Marching Cubes Algorithmus."""
    normalized_stack = (binary_stack - binary_stack.min()) / (binary_stack.max() - binary_stack.min())
    verts, faces, _, _ = measure.marching_cubes(normalized_stack, level=0.5, spacing=spacing)
    return verts, faces

def save_mesh_as_vtk(verts, faces, output_filename):
    """Speichert ein Mesh als VTK-Datei."""
    cells = [("triangle", faces)]
    meshio.write_points_cells(output_filename, verts, cells)
    logger.info(f"Mesh saved as VTK file: {output_filename}")

def generate_tetrahedral_mesh(binary_volume: np.ndarray, voxel_size: float, output_filename: str):
    """Erstellt ein Tetraedernetz und speichert es als VTK-Datei."""
    try:
        import ciclope.core.tetraFE as tetraFE
        vs = np.ones(3) * voxel_size
        mesh_size_factor = 1.4
        max_facet_distance = mesh_size_factor * np.min(vs)
        max_cell_circumradius = 2 * mesh_size_factor * np.min(vs)
        mesh = tetraFE.cgal_mesh(binary_volume, vs, 'tetra', max_facet_distance, max_cell_circumradius)
        mesh.write(output_filename)
        logger.info(f"Tetrahedral mesh saved to {output_filename}")
        return mesh
    except Exception as e:
        logger.error(f"Error generating tetrahedral mesh: {e}")
        return None
