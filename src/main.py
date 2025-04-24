# main.py (optimiert)

import logging
import os
import sys
import datetime
import timeit
import psutil
import gc
import time
import numpy as np
from image_processing import (
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
)
from cluster_analysis import (
    find_largest_cluster
)
from mesh_generation import (
    marching_cubes,
    save_mesh_as_vtk,
    generate_tetrahedral_mesh
)
from image_loading import (
    load_images_in_chunks
)
from file_io import (
    save_tiff_in_chunks
)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s : %(name)s : %(message)s",
    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Konfigurationsparameter
MEMORY_LIMIT_PERCENT = 80  # Maximale Speicherauslastung in Prozent
CHUNK_SIZE = 50  # Anzahl der Bilder pro Chunk für die Verarbeitung
OUTPUT_DIR = "test_pictures"


def get_base_path():
    """Gibt den Basis-Pfad der Daten zurück."""
    return "/home/mathias/PycharmProjects/BoneSimulation/data"


def check_memory():
    """Überprüft den Speicherverbrauch und gibt True zurück wenn zu hoch."""
    memory = psutil.virtual_memory()
    return memory.percent > MEMORY_LIMIT_PERCENT


def wait_for_memory():
    """Wartet bis genügend Speicher frei ist."""
    while check_memory():
        logger.warning(f"Speicherverbrauch zu hoch ({psutil.virtual_memory().percent}%). Warte 10 Sekunden...")
        gc.collect()  # Garbage Collection explizit aufrufen
        time.sleep(10)


def ensure_output_dir(dir_path):
    """Stellt sicher, dass das Ausgabeverzeichnis existiert."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Ausgabeverzeichnis erstellt: {dir_path}")


def process_and_visualize(directory):
    """Führt den gesamten Bildverarbeitungs- und Meshing-Prozess durch, speichereffizient."""
    logger.info("Starting processing and visualization...")
    ensure_output_dir(OUTPUT_DIR)

    # Timestamp für die Ausgabedateien
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pfad direkt für den großen Datensatz setzen
    dataset_path = os.path.join(directory, "bigdataset")

    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Bilder in Chunks laden und verarbeiten
    logger.info(f"Lade Bilder aus: {dataset_path}")

    # Speichern des Rohdatensatzes in Chunks
    raw_output_path = os.path.join(OUTPUT_DIR, f"raw_{timestamp}.tif")

    # Wir verarbeiten die Bilder in Chunks statt alles auf einmal zu laden
    try:
        data_generator = load_images_in_chunks(dataset_path, chunk_size=CHUNK_SIZE)
        all_processed_chunks = []

        for i, chunk in enumerate(data_generator):
            logger.info(f"Verarbeite Chunk {i + 1} mit {len(chunk)} Bildern")

            # Vor jedem Verarbeitungsschritt Speicher prüfen
            wait_for_memory()

            # Chunk verarbeiten
            blurred_chunk, binary_chunk, _ = process_images_globally(chunk)
            del chunk  # Freigeben des Speicherplatzes
            gc.collect()

            wait_for_memory()
            closed_binary_chunk = apply_morphological_closing(binary_chunk)
            del binary_chunk, blurred_chunk
            gc.collect()

            wait_for_memory()
            interpolated_chunk = interpolate_image_stack(closed_binary_chunk, scaling_factor=0.5)
            del closed_binary_chunk
            gc.collect()

            # Verarbeitetes Chunk speichern
            all_processed_chunks.append(interpolated_chunk)

            # Fortschritt protokollieren
            logger.info(f"Chunk {i + 1} verarbeitet. Aktueller Speicherverbrauch: {psutil.virtual_memory().percent}%")

        # Alle verarbeiteten Chunks zusammenführen
        wait_for_memory()
        interpolated_stack = np.concatenate(all_processed_chunks, axis=0)
        del all_processed_chunks
        gc.collect()

        # Größtes Cluster finden
        wait_for_memory()
        logger.info("Suche nach dem größten Cluster...")
        largest_cluster, num_clusters, cluster_size = find_largest_cluster(interpolated_stack)
        logger.info(f"Größtes Cluster gefunden: {cluster_size} Voxel von insgesamt {num_clusters} Clustern")

        # Wenn kein Cluster gefunden wurde, Programm beenden
        if largest_cluster is None:
            logger.error("Kein Cluster gefunden. Beende das Programm.")
            return

        # Originales interpolated_stack nicht mehr benötigt
        del interpolated_stack
        gc.collect()

        # Mesh generieren
        wait_for_memory()
        logger.info("Generiere Mesh mit Marching Cubes...")
        verts, faces = marching_cubes(largest_cluster)

        if verts is not None and faces is not None:
            mesh_output_path = os.path.join(OUTPUT_DIR, f"mesh_{timestamp}.vtk")
            save_mesh_as_vtk(verts, faces, mesh_output_path)
            logger.info(f"Mesh gespeichert als: {mesh_output_path}")
        else:
            logger.error("Fehler bei der Mesh-Generierung. Überspringe Speicherung.")

        # Tetrahedrales Mesh generieren
        wait_for_memory()
        logger.info("Generiere tetrahedrales Mesh...")
        tetra_output_path = os.path.join(OUTPUT_DIR, f"tetramesh_{timestamp}.vtk")
        tetrahedral_mesh = generate_tetrahedral_mesh(largest_cluster, 0.1, tetra_output_path)

        if tetrahedral_mesh:
            logger.info(f"Tetrahedrales Mesh erfolgreich generiert und gespeichert als: {tetra_output_path}")
        else:
            logger.warning("Tetrahedrales Mesh konnte nicht generiert werden.")

        logger.info("Verarbeitung abgeschlossen.")

    except Exception as e:
        logger.error(f"Fehler während der Verarbeitung: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    starttime = timeit.default_timer()
    try:
        directory = get_base_path()
        process_and_visualize(directory)
        endtime = timeit.default_timer()
        logger.info(f"Verarbeitung in {endtime - starttime:.2f} Sekunden abgeschlossen.")
    except KeyboardInterrupt:
        logger.info("Programm durch Benutzer unterbrochen.")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}", exc_info=True)
