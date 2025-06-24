# src/mesh_generation.py

import logging
import os

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

        # Fix: Convert the binary data to float before checking min/max
        # to avoid boolean subtraction issues
        normalized_stack = binary_stack.astype(np.float32)

        min_val, max_val = normalized_stack.min(), normalized_stack.max()
        if max_val - min_val == 0:
            raise ValueError("Konstantes Bild erkannt. Keine Mesh-Generierung möglich.")

        # Für sehr große Volumen verwenden wir einen Chunk-basierten Ansatz
        if chunk_size is not None and binary_stack.shape[0] > chunk_size:
            return _chunked_marching_cubes(binary_stack, spacing=spacing, chunk_size=chunk_size)

        # Für kleinere Volumen verwenden wir den Standard-Ansatz
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

        # Extrahiere den Chunk und wandele zu float32 um
        chunk = binary_stack[z_start:z_end].copy().astype(np.float32)

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
                              use_downsampling=True, max_volume_size=100_000_000):
    """
    Erstellt ein Tetraedernetz und speichert es als VTK-Datei.
    Bei großen Volumen wird optional Downsampling angewendet.
    Schutz gegen invertierte oder verrauschte Binärvolumina ist eingebaut.
    """
    if tetraFE is None:
        logger.error("Tetrahedral Meshing kann nicht ausgeführt werden – 'ciclope.core.tetraFE' nicht verfügbar.")
        return None

    try:
        if binary_volume.size == 0:
            raise ValueError("Leeres Eingangsvolumen für Tetrahedral Meshing.")

        # --- Sicherheit: Volume binär machen ---
        binary_volume = (binary_volume > 0.5).astype(np.uint8)

        # --- Optional: Invertieren, wenn es wahrscheinlich falsch herum ist ---
        if np.mean(binary_volume) > 0.5:
            logger.warning("Binärvolumen scheint invertiert – führe Invertierung durch.")
            binary_volume = 1 - binary_volume

        # --- Downsampling bei großen Volumina ---
        if use_downsampling and binary_volume.size > max_volume_size:
            logger.info(f"Volumen zu groß für direktes Tetrahedral Meshing ({binary_volume.size} > {max_volume_size}).")
            logger.info("Wende Downsampling an...")

            reduction_factor = (max_volume_size / binary_volume.size) ** (1 / 3)
            reduction_factor = min(reduction_factor, 0.5)

            from scipy.ndimage import zoom
            downsampled = zoom(binary_volume, reduction_factor, order=0)

            # Nach Downsampling erneut binär machen
            binary_volume = (downsampled > 0.5).astype(np.uint8)

            logger.info(f"Volumen auf {binary_volume.shape} verkleinert (Faktor: {reduction_factor:.3f})")
            voxel_size = voxel_size / reduction_factor
            logger.info(f"Voxelgröße angepasst auf {voxel_size:.6f}")


        # --- Mesh-Parameter berechnen ---
        vs = np.ones(3) * voxel_size
        mesh_size_factor = 1.4
        max_facet_distance = mesh_size_factor * np.min(vs)
        max_cell_circumradius = 2 * mesh_size_factor * np.min(vs)

        logger.info("Starte CGAL Tetrahedral Meshing...")

        mesh = tetraFE.cgal_mesh(binary_volume, vs, 'tetra', max_facet_distance, max_cell_circumradius)


        # --- Optionale Log-Ausgabe über Zellanzahl ---
        try:
            if hasattr(mesh, 'n_cells'):
                cell_count = mesh.n_cells() if callable(mesh.n_cells) else mesh.n_cells
            elif hasattr(mesh, 'cells'):
                cell_count = len(mesh.cells)
            elif hasattr(mesh, 'num_cells'):
                cell_count = mesh.num_cells() if callable(mesh.num_cells) else mesh.num_cells
            else:
                cell_count = None

            if cell_count is not None:
                logger.info(f"Tetrahedral Mesh erstellt mit {cell_count} Tetraedern.")
        except Exception as attribute_error:
            logger.warning(f"Konnte Anzahl der Tetraeder nicht ermitteln: {attribute_error}")

        mesh.write(output_filename)
        logger.info(f"Tetrahedral Mesh erfolgreich gespeichert: {output_filename}")
        return mesh

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Tetrahedral Meshes: {e}")
        return None


def reverse_binary(volume: np.ndarray) -> np.ndarray:
    """
    Kehrt die Werte eines binären Volumens um (1 → 0, 0 → 1).
    Gibt eine Fehlermeldung aus, wenn das Volumen nicht binär ist.
    """
    unique_values = np.unique(volume)
    if not np.all(np.isin(unique_values, [0, 1])):
        logger.warning(f"Nicht-binäre Werte im Volumen gefunden: {unique_values}. Erwarte nur 0 und 1.")

    return 1 - volume


import os
import numpy as np
import pyvista as pv  # pyvista als VTK-Wrapper

def extract_blocks(volume, block_size_voxels, step_size_voxels):
    """Extract as many non-empty blocks as possible from the volume."""
    blocks = []

    z_max, y_max, x_max = volume.shape
    z_block, y_block, x_block = block_size_voxels
    z_step, y_step, x_step = step_size_voxels

    block_idx = 0

    for z in range(0, z_max - z_block + 1, z_step):
        for y in range(0, y_max - y_block + 1, y_step):
            for x in range(0, x_max - x_block + 1, x_step):
                block = volume[z:z + z_block, y:y + y_block, x:x + x_block]

                if np.count_nonzero(block) == 0:
                    continue  # skip empty block

                blocks.append((block_idx, block))
                block_idx += 1

    print(f"Total blocks extracted: {block_idx}")
    return blocks


def save_blocks_as_vtk(blocks, voxel_spacing, output_dir, timestamp):
    """Save each block as VTK file."""
    os.makedirs(output_dir, exist_ok=True)

    for block_idx, block in blocks:
        # Convert numpy block to PyVista grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(block.shape)[::-1] + 1  # (x, y, z) + 1
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = voxel_spacing[::-1]  # (x, y, z)

        # Add binary data as scalar field
        grid.cell_data["values"] = block.flatten(order="F")

        # Save VTK
        vtk_filename = os.path.join(output_dir, f"block_{block_idx:04d}_{timestamp}.vtk")
        grid.save(vtk_filename)

        logger.info(f"Saved block {block_idx} as VTK: {vtk_filename}")
