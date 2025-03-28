import logging
import os
import sys
import datetime
import timeit
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
    load_image,
    load_images,
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

# Variable zur Steuerung des Datensatzes
USE_LARGE_DATASET = (
    True  # True: Großer Datensatz-Ordner, False: Kleiner Datensatz-Ordner
)


def get_base_path():
    """Gibt den Basis-Pfad der Daten zurück."""
    return "/home/mathias/PycharmProjects/BoneSimulation/data"


def process_and_visualize(directory):
    """Führt den gesamten Bildverarbeitungs- und Meshing-Prozess durch, speicherschonend."""
    logger.info("Starting processing and visualization...")

    # Pfad setzen abhängig von der Datensatzgröße
    dataset_path = os.path.join(
        directory, "bigdataset" if USE_LARGE_DATASET else "dataset"
    )

    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Bilder aus dem entsprechenden Ordner laden
    data_array = load_images(dataset_path)

    if data_array is None:
        logger.error("Failed to load images. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded images: {data_array.shape}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_tiff_in_chunks(data_array, f"pictures/raw_{timestamp}.tif")

    # Bildverarbeitungsschritte
    blurred_images, binary_images, global_threshold = process_images_globally(
        data_array
    )
    closed_binary_images = apply_morphological_closing(binary_images)
    print("2")
    interpolated_stack = interpolate_image_stack(
        closed_binary_images, scaling_factor=0.5
    )
    print("5")

    largest_cluster, _, cluster_size = find_largest_cluster(interpolated_stack)
    logger.info(f"Largest cluster found: {cluster_size} voxels")

    verts, faces = marching_cubes(interpolated_stack)
    save_mesh_as_vtk(verts, faces, f"test_pictures/mesh_{timestamp}.vtk")

    tetrahedral_mesh = generate_tetrahedral_mesh(
        largest_cluster, 0.1, f"test_pictures/tetramesh_{timestamp}.vtk"
    )

    if tetrahedral_mesh:
        logger.info("Tetrahedral mesh successfully generated.")

    logger.info("Processing completed.")


if __name__ == "__main__":
    starttime = timeit.default_timer()
    directory = get_base_path()
    process_and_visualize(directory)
    endtime = timeit.default_timer()
    print(f"It took {endtime - starttime} seconds for the whole simulation to complete.")

