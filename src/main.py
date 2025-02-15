import logging
import os
import sys
import datetime
import numpy as np
from src.image_processing import (
    load_tiff_stream_lazy,
    load_images,
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
    find_largest_cluster,
    save_tiff_in_chunks,
    marching_cubes,
    save_mesh_as_vtk,
    generate_tetrahedral_mesh,
)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s: %(levelname)s : %(name)s : %(message)s",
                    handlers=[logging.FileHandler("logfile.log"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Variable zur Steuerung des Ladeverhaltens
USE_TIFF_STREAM = True  # True: Lade TIFF-Stream, False: Lade einzelne Bilder als Stack

def get_base_path():
    """Gibt den Basis-Pfad der Daten zurück."""
    return "/home/mathias/PycharmProjects/BoneSimulation/data"

def process_and_visualize(directory):
    """Führt den gesamten Bildverarbeitungs- und Meshing-Prozess durch, speicherschonend."""
    logger.info("Starting processing and visualization...")

    if USE_TIFF_STREAM:
        filepath = os.path.join(directory, "Knochenprobe2stream.tiff")
        data_array = load_tiff_stream_lazy(filepath)
    else:
        data_array = load_images(directory)

    if data_array is None:
        logger.error("Failed to load images. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded images: {data_array.shape}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_tiff_in_chunks(data_array, f"pictures/raw_{timestamp}.tif")

    # Bildverarbeitungsschritte
    blurred_images, binary_images, global_threshold = process_images_globally(data_array)
    closed_binary_images = apply_morphological_closing(binary_images)
    interpolated_stack = interpolate_image_stack(closed_binary_images, scaling_factor=0.5)

    largest_cluster, _, cluster_size = find_largest_cluster(interpolated_stack)
    logger.info(f"Largest cluster found: {cluster_size} voxels")

    verts, faces = marching_cubes(interpolated_stack)
    save_mesh_as_vtk(verts, faces, f"test_pictures/mesh_{timestamp}.vtk")

    tetrahedral_mesh = generate_tetrahedral_mesh(largest_cluster, 0.1, f"test_pictures/tetramesh_{timestamp}.vtk")

    if tetrahedral_mesh:
        logger.info("Tetrahedral mesh successfully generated.")

    logger.info("Processing completed.")

if __name__ == "__main__":
    directory = get_base_path()
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    process_and_visualize(directory)
