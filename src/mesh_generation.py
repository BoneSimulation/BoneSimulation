import logging
import numpy as np
from skimage import measure
import meshio

try:
    import ciclope.core.tetraFE as tetraFE
except ImportError:
    tetraFE = None
    logging.error("Modul 'ciclope.core.tetraFE' konnte nicht importiert werden. Bitte installieren!")

logger = logging.getLogger(__name__)


def marching_cubes(binary_stack: np.ndarray, spacing=(1, 1, 1)):
    """
    Erstellt ein Mesh aus dem binären Stack mit dem Marching Cubes Algorithmus.

    :param binary_stack: 3D-Binärvolumen (numpy.ndarray)
    :param spacing: Pixelabstand in x, y, z (Tuple)
    :return: (Vertices, Faces) oder (None, None) bei Fehler
    """
    try:
        if binary_stack.size == 0:
            raise ValueError("Leerer Eingabestack für Marching Cubes.")

        min_val, max_val = binary_stack.min(), binary_stack.max()
        if max_val - min_val == 0:
            raise ValueError("Konstantes Bild erkannt. Keine Mesh-Generierung möglich.")

        normalized_stack = (binary_stack - min_val) / (max_val - min_val)
        verts, faces, _, _ = measure.marching_cubes(normalized_stack, level=0.5, spacing=spacing)

        logger.info(f"Marching Cubes erfolgreich ausgeführt. {len(verts)} Punkte, {len(faces)} Flächen generiert.")
        return verts, faces

    except Exception as e:
        logger.error(f"Fehler bei Marching Cubes: {e}")
        return None, None


def save_mesh_as_vtk(verts: np.ndarray, faces: np.ndarray, output_filename: str):
    """
    Speichert ein Mesh als VTK-Datei.

    :param verts: Vertices des Meshes
    :param faces: Dreiecksflächen des Meshes
    :param output_filename: Pfad zur Ausgabedatei
    """
    try:
        if verts is None or faces is None:
            raise ValueError("Ungültige Eingabedaten für VTK-Speicherung.")

        cells = [("triangle", faces)]
        meshio.write_points_cells(output_filename, verts, cells)

        logger.info(f"Mesh erfolgreich als VTK-Datei gespeichert: {output_filename}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der VTK-Datei '{output_filename}': {e}")


def generate_tetrahedral_mesh(binary_volume: np.ndarray, voxel_size: float, output_filename: str):
    """
    Erstellt ein Tetraedernetz und speichert es als VTK-Datei.

    :param binary_volume: 3D-Binärvolumen (numpy.ndarray)
    :param voxel_size: Voxelgröße als Float
    :param output_filename: Name der VTK-Datei
    :return: Mesh-Objekt oder None bei Fehler
    """
    if tetraFE is None:
        logger.error("Tetrahedral Meshing kann nicht ausgeführt werden – 'ciclope.core.tetraFE' nicht verfügbar.")
        return None

    try:
        if binary_volume.size == 0:
            raise ValueError("Leeres Eingangsvolumen für Tetrahedral Meshing.")

        vs = np.ones(3) * voxel_size
        mesh_size_factor = 1.4
        max_facet_distance = mesh_size_factor * np.min(vs)
        max_cell_circumradius = 2 * mesh_size_factor * np.min(vs)

        mesh = tetraFE.cgal_mesh(binary_volume, vs, 'tetra', max_facet_distance, max_cell_circumradius)
        mesh.write(output_filename)

        logger.info(f"Tetrahedral Mesh erfolgreich gespeichert: {output_filename}")
        return mesh

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Tetrahedral Meshes: {e}")
        return None
