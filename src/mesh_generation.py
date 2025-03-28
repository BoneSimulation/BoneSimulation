import logging
import ciclope.core.tetraFE
import numpy as np
from skimage import morphology, filters, measure
import scipy.ndimage
import tifffile as tiff
from PIL import Image
import vtk
import meshio
import os
from multiprocessing import Pool

logger = logging.getLogger(__name__)


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
