# mesh_generation.py

import logging
import numpy as np
from skimage import measure
import meshio
import gc

try:
    import ciclope.core.tetraFE as tetraFE
except ImportError:
    tetraFE = None
    logging.error("Modul 'ciclope.core.tetraFE' konnte nicht importiert werden. Bitte installieren!")

logger = logging.getLogger(__name__)


def marching_cubes(binary_stack: np.ndarray, spacing=(1, 1, 1), chunk_size=None):
    """
    Erstellt ein Mesh aus dem binären Stack mit dem Marching Cubes Algorithmus.
    Verwendet einen Chunk-basierten Ansatz für große Volumen.

    :param binary_stack: 3D-Binärvolumen (numpy.ndarray)
    :param spacing: Pixelabstand in x, y, z (Tuple)
    :param chunk_size: Größe der Chunks für die Verarbeitung (z-Richtung)
    :return: (Vertices, Faces) oder (None, None) bei Fehler
    """
    try:
        if binary_stack.size == 0:
            raise ValueError("Leerer Eingabestack für Marching Cubes.")

        min_val, max_val = binary_stack.min(), binary_stack.max()
        if max_val - min_val == 0:
            raise ValueError("Konstantes Bild erkannt. Keine Mesh-Generierung möglich.")

        # Für sehr große Volumen verwenden wir einen Chunk-basierten Ansatz
        if chunk_size is not None and binary_stack.shape[0] > chunk_size:
            return _chunked_marching_cubes(binary_stack, spacing=spacing, chunk_size=chunk_size)

        # Für kleinere Volumen verwenden wir den Standard-Ansatz
        normalized_stack = binary_stack.astype(np.float32)  # float32 für Speichereffizienz
        verts, faces, _, _ = measure.marching_cubes(normalized_stack, level=0.5, spacing=spacing)

        logger.info(f"Marching Cubes erfolgreich ausgeführt. {len(verts)} Punkte, {len(faces)} Flächen generiert.")
        return verts, faces

    except Exception as e:
        logger.error(f"Fehler bei Marching Cubes: {e}")
        return None, None


def _chunked_marching_cubes(binary_stack, spacing=(1, 1, 1), chunk_size=50, overlap=5):
    """
    Führt Marching Cubes in Chunks aus und fügt die Ergebnisse zusammen.

    :param binary_stack: 3D-Binärvolumen
    :param spacing: Pixelabstand in x, y, z
    :param chunk_size: Anzahl der Schichten pro Chunk
    :param overlap: Überlappung zwischen Chunks, um Nahtstellen zu vermeiden
    :return: (Vertices, Faces) oder (None, None) bei Fehler
    """
    logger.info(f"Starte Chunk-basierten Marching Cubes mit Chunkgröße {chunk_size} und Überlappung {overlap}...")

    all_verts = []
    all_faces = []
    vert_offset = 0

    # Verarbeite jeden Chunk
    for z_start in range(0, binary_stack.shape[0], chunk_size - overlap):
        z_end = min(z_start + chunk_size, binary_stack.shape[0])

        # Überspringe den letzten Chunk, wenn er zu klein ist
        if z_end - z_start < 10:
            continue

        logger.info(f"Verarbeite Chunk von z={z_start} bis z={z_end}")

        # Extrahiere den Chunk
        chunk = binary_stack[z_start:z_end].copy()

        # Wende Marching Cubes auf den Chunk an
        try:
            verts, faces, _, _ = measure.marching_cubes(chunk, level=0.5, spacing=spacing)

            if len(verts) == 0:
                logger.info(f"Keine Oberfläche im Chunk von z={z_start} bis z={z_end} gefunden")
                continue

            # Verschiebe die z-Koordinaten basierend auf der Chunk-Position
            verts[:, 0] += z_start * spacing[0]

            # Korrigiere die Vertex-Indizes der Dreiecke
            faces = faces + vert_offset if vert_offset > 0 else faces

            # Füge zu den Gesamtergebnissen hinzu
            all_verts.append(verts)
            all_faces.append(faces)

            # Aktualisiere den Vertex-Offset für den nächsten Chunk
            vert_offset += len(verts)

            # Speicher freigeben
            del verts, faces, chunk
            gc.collect()

        except Exception as e:
            logger.error(f"Fehler bei Marching Cubes für Chunk z={z_start} bis z={z_end}: {e}")

    # Wenn keine Vertices erstellt wurden
    if not all_verts:
        logger.warning("Keine Vertices durch Marching Cubes gefunden.")
        return None, None

    # Kombiniere die Ergebnisse
    try:
        combined_verts = np.vstack(all_verts)
        combined_faces = np.vstack(all_faces)

        logger.info(f"Chunk-basierter Marching Cubes abgeschlossen. {len(combined_verts)} Punkte, "
                    f"{len(combined_faces)} Flächen generiert.")

        return combined_verts, combined_faces

    except Exception as e:
        logger.error(f"Fehler beim Kombinieren der Mesh-Chunks: {e}")
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

        logger.info(f"Speichere Mesh mit {len(verts)} Vertices und {len(faces)} Dreiecken als VTK...")

        # Verkleinere die Datengröße durch Konvertierung zu float32
        verts = verts.astype(np.float32)

        # Erstelle die Zellen für meshio
        cells = [("triangle", faces)]

        # Speichere das Mesh
        meshio.write_points_cells(output_filename, verts, cells)

        logger.info(f"Mesh erfolgreich als VTK-Datei gespeichert: {output_filename}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der VTK-Datei '{output_filename}': {e}")


def generate_tetrahedral_mesh(binary_volume: np.ndarray, voxel_size: float, output_filename: str,
                              use_downsampling=True, max_volume_size=50_000_000):
    """
    Erstellt ein Tetraedernetz und speichert es als VTK-Datei.
    Bei großen Volumen wird optional Downsampling angewendet.

    :param binary_volume: 3D-Binärvolumen (numpy.ndarray)
    :param voxel_size: Voxelgröße als Float
    :param output_filename: Name der VTK-Datei
    :param use_downsampling: Ob Downsampling für große Volumen verwendet werden soll
    :param max_volume_size: Maximale Volumengröße (in Voxeln) bevor Downsampling angewendet wird
    :return: Mesh-Objekt oder None bei Fehler
    """
    if tetraFE is None:
        logger.error("Tetrahedral Meshing kann nicht ausgeführt werden – 'ciclope.core.tetraFE' nicht verfügbar.")
        return None

    try:
        if binary_volume.size == 0:
            raise ValueError("Leeres Eingangsvolumen für Tetrahedral Meshing.")

        # Prüfe Volumengröße und wende ggf. Downsampling an
        if use_downsampling and binary_volume.size > max_volume_size:
            logger.info(f"Volumen zu groß für direktes Tetrahedral Meshing ({binary_volume.size} > {max_volume_size}).")
            logger.info("Wende Downsampling an...")

            # Berechne Downsampling-Faktor
            reduction_factor = (max_volume_size / binary_volume.size) ** (1 / 3)
            # Mindestens Faktor 0.5 (Halbierung)
            reduction_factor = min(reduction_factor, 0.5)

            from scipy.ndimage import zoom
            downsampled = zoom(binary_volume, reduction_factor, order=0)  # order=0 für nächste-Nachbar-Interpolation

            logger.info(f"Volumen auf {downsampled.shape} verkleinert (Faktor: {reduction_factor:.3f})")
            binary_volume = downsampled

            # Passe Voxelgröße entsprechend an
            voxel_size = voxel_size / reduction_factor
            logger.info(f"Voxelgröße angepasst auf {voxel_size:.6f}")

        vs = np.ones(3) * voxel_size
        mesh_size_factor = 1.4
        max_facet_distance = mesh_size_factor * np.min(vs)
        max_cell_circumradius = 2 * mesh_size_factor * np.min(vs)

        logger.info("Starte CGAL Tetrahedral Meshing...")
        mesh = tetraFE.cgal_mesh(binary_volume, vs, 'tetra', max_facet_distance, max_cell_circumradius)

        # Fix: Check the mesh object attributes and structure
        try:
            # Try different methods to get cell count
            if hasattr(mesh, 'n_cells'):
                if callable(mesh.n_cells):
                    cell_count = mesh.n_cells()
                else:
                    cell_count = mesh.n_cells
            elif hasattr(mesh, 'cells'):
                cell_count = len(mesh.cells)
            elif hasattr(mesh, 'num_cells'):
                if callable(mesh.num_cells):
                    cell_count = mesh.num_cells()
                else:
                    cell_count = mesh.num_cells
            else:
                # If no method works, just report success without a count
                logger.info("Tetrahedral Mesh erfolgreich erstellt.")
                cell_count = None

            # Log with cell count if available
            if cell_count is not None:
                logger.info(f"Tetrahedral Mesh erstellt mit {cell_count} Tetraedern.")
        except Exception as attribute_error:
            # If accessing attributes fails, just continue without reporting count
            logger.warning(f"Konnte Anzahl der Tetraeder nicht ermitteln: {attribute_error}")

        # Save the mesh regardless of count retrieval
        mesh.write(output_filename)
        logger.info(f"Tetrahedral Mesh erfolgreich gespeichert: {output_filename}")
        return mesh

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Tetrahedral Meshes: {e}")
        return None