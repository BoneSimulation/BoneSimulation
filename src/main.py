# src/main.py

"""
python src/main.py --test complete - ganz normal (default)
python src/main.py --test small - mini-Volume, kleiner Test
python src/main.py --test dry - dry run, kein Speichern
"""

import logging
import os
import sys
import datetime
import timeit
import psutil
import gc
import time
import numpy as np
import argparse
import stat
import faulthandler
import subprocess
import meshio

from image_processing import (
    process_images_globally,
    apply_morphological_closing,
    interpolate_image_stack,
    crop_stack_custom
)
from cluster_analysis import find_largest_cluster
from mesh_generation import (
    marching_cubes,
    save_mesh_as_vtk,
    generate_tetrahedral_mesh,
    reverse_binary, extract_blocks, save_blocks_as_vtk
)
from image_loading import load_images_in_chunks
from file_io import save_largest_cluster_as_tiff
from src.calculix_export import export_to_calculix
from src.reporting import extract_statistics_from_volume
from src.block_extraction import extract_blocks_raster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s : %(name)s : %(message)s",
    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configuration
MEMORY_LIMIT_PERCENT = 80
CHUNK_SIZE = 50
OUTPUT_DIR_MESHES = "pictures"

PDF_REPORT_DIR = "report"
CALCULIX_REPORT_DIR = "calculix"
LOGS_DIR = "logs"

DATASET_BIG = "data/bigdataset"
DATASET_SMALL = "data/dataset"

# Crop settings
cropping_settings = {
    "z": (3, 0),
    "y": (50, 50),
    "x": (15, 35),
}


def export_to_calculix(mesh_points, mesh_elements, material_props, inp_path):
    mesh = meshio.Mesh(
        points=mesh_points,
        cells={"tetra": mesh_elements}
    )
    meshio.write(inp_path, mesh)
    with open(inp_path, "a") as f:
        f.write("*MATERIAL, NAME=Material1\n")
        f.write("*ELASTIC\n")
        f.write(f"{material_props['E']}, {material_props['nu']}\n")
        f.write("*END MATERIAL\n")


def generate_run_script(input_inp_path, preferred_solver, output_dir):
    run_script_path = os.path.join(output_dir, "run_simulation.sh")

    if preferred_solver == "":
        preferred_solver = "./cgx_2.22"

    with open(run_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Automatisch erzeugtes Run-Script für CalculiX\n")
        num_threads = get_max_threads()
        f.write(f"export OMP_NUM_THREADS={num_threads}\n")
        f.write(f"{preferred_solver} -i {os.path.splitext(os.path.basename(input_inp_path))[0]} \n")

    os.chmod(run_script_path, 0o755)
    logger.info(f"Run-Script gespeichert: {run_script_path}")
    print(f"Run-Script gespeichert: {run_script_path}")
    print(f"Verwendete Threads: {num_threads}")


def get_max_threads():
    output = subprocess.check_output(['lscpu'], universal_newlines=True)

    cores = 0
    threads_per_core = 0

    for line in output.splitlines():
        if "Core(s) per socket:" in line:
            cores = int(line.split(":")[1].strip())
        elif "Thread(s) per core:" in line:
            threads_per_core = int(line.split(":")[1].strip())

    max_threads = cores * threads_per_core
    print(f"Maximale Anzahl an Threads: {max_threads}")
    return max_threads



def get_base_path():
    return "data"

def check_memory():
    memory = psutil.virtual_memory()
    return memory.percent > MEMORY_LIMIT_PERCENT

def wait_for_memory():
    while check_memory():
        logger.warning(f"Memory usage too high ({psutil.virtual_memory().percent}%). Waiting...")
        gc.collect()
        time.sleep(10)

def ensure_output_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created output directory: {dir_path}")

def process_and_visualize(directory, test_mode="none"):
    logger.info(f"Starte Processing Pipeline (test_mode = '{test_mode}')")

    ensure_output_dir(OUTPUT_DIR_MESHES)
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

    dataset_path = os.path.join(directory, "bigdataset")
    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    global CHUNK_SIZE
    if test_mode in ("small", "fast_debug"):
        CHUNK_SIZE = 5
        logger.warning(f"TESTMODE: Reduziere CHUNK_SIZE auf {CHUNK_SIZE}")

    if test_mode == "dry":
        logger.warning("TESTMODE: Dry run (keine Dateien werden gespeichert!)")

    if test_mode == "complete":
        logger.warning("TESTMODE: Complete (alle Schritte und Reports werden geprüft)")

    logger.info(f"CHUNK_SIZE = {CHUNK_SIZE}")
    logger.info(f"Output dir  = {OUTPUT_DIR_MESHES}")

    logger.info(f"Loading images from: {dataset_path}")
    data_generator = load_images_in_chunks(dataset_path, chunk_size=CHUNK_SIZE)
    interpolated_chunks = []

    try:
        for i, chunk in enumerate(data_generator):
            logger.info(f"Processing chunk {i + 1} with {len(chunk)} images")
            wait_for_memory()

            blurred_chunk, binary_chunk, _ = process_images_globally(chunk)
            del chunk
            gc.collect()

            wait_for_memory()
            closed_chunk = apply_morphological_closing(binary_chunk)
            del binary_chunk, blurred_chunk
            gc.collect()

            wait_for_memory()
            interpolated_chunk = interpolate_image_stack(closed_chunk, scaling_factor=0.5)
            del closed_chunk
            gc.collect()

            interpolated_chunks.append(interpolated_chunk)
            logger.info(f"Chunk {i + 1} processed and added to stack.")

        logger.info("Concatenating all interpolated chunks...")
        interpolated_stack = np.concatenate(interpolated_chunks, axis=0)
        del interpolated_chunks
        gc.collect()

        interpolated_stack = crop_stack_custom(interpolated_stack, cropping_settings)
        logger.info(f"Stack cropped with settings: {cropping_settings}")

        wait_for_memory()
        logger.info("Finding largest cluster...")
        largest_cluster, num_clusters, cluster_size = find_largest_cluster(interpolated_stack)
        logger.info(f"Largest cluster: {cluster_size} voxels in {num_clusters} clusters")

        cluster_tiff_path = os.path.join(OUTPUT_DIR_MESHES, f"largest_cluster_{timestamp}.tif")
        if test_mode not in ("dry",):
            save_largest_cluster_as_tiff(largest_cluster, cluster_tiff_path)
            logger.info(f"Largest cluster saved: {cluster_tiff_path}")
        else:
            logger.info(f"[Dry] Skip saving largest cluster ({cluster_tiff_path})")

        if largest_cluster is None:
            logger.error("No cluster found. Exiting.")
            return

        del interpolated_stack
        gc.collect()

        wait_for_memory()
        logger.info("Generating surface mesh...")
        verts, faces = marching_cubes(largest_cluster)
        mesh_path = os.path.join(OUTPUT_DIR_MESHES, f"mesh_{timestamp}.vtk")
        if verts is not None and faces is not None:
            if test_mode not in ("dry",):
                save_mesh_as_vtk(verts, faces, mesh_path)
                logger.info(f"Surface mesh saved: {mesh_path}")
            else:
                logger.info(f"[Dry] Skip saving surface mesh ({mesh_path})")
        else:
            logger.error("Surface mesh generation failed.")

        wait_for_memory()
        logger.info("Generating tetrahedral mesh...")
        reversed_cluster = reverse_binary(largest_cluster)
        tetra_path = os.path.join(OUTPUT_DIR_MESHES, f"tetramesh_{timestamp}.vtk")
        tetra_mesh = generate_tetrahedral_mesh(reversed_cluster, 0.1, tetra_path)
        if tetra_mesh:
            logger.info(f"Tetrahedral mesh saved: {tetra_path}")
        else:
            logger.warning("Tetrahedral mesh generation failed.")

        block_size_mm = 30.0
        voxel_spacing = (0.05, 0.05, 0.05)
        block_size_voxels = int(block_size_mm / voxel_spacing[0])

        blocks_output_dir = os.path.join(OUTPUT_DIR_MESHES, f"blocks_{timestamp}")
        if test_mode != "dry":
            extract_blocks_raster(
                volume=largest_cluster,
                block_size_voxels=block_size_voxels,
                output_dir="output_blocks",
                voxel_spacing=voxel_spacing,
                write_tetra_mesh=True
            )
        else:
            logger.info("[Dry] Skipping block extraction.")

        # Export to CalculiX
        material_props = {'E': 18000.0, 'nu': 0.3}
        inp_path = os.path.join(CALCULIX_REPORT_DIR, f"bone_sim_{timestamp}.inp")

        # --- Hier: Dummy points/elements ersetzen ---
        mesh = meshio.read(tetra_path)
        mesh_points = mesh.points
        mesh_elements = mesh.cells_dict['tetra']

        if test_mode not in ("dry",):
            preferred_solver = input("Please enter the preferred solver you would like to use: (default ccx)")
            export_to_calculix(mesh_points, mesh_elements, material_props, inp_path)
            generate_run_script(inp_path, preferred_solver, CALCULIX_REPORT_DIR)
            os.system("/home/mathias/PycharmProjects/BoneSimulation/test_pictures/run_simulation.sh")
        else:
            logger.info(f"[Dry] Skip CalculiX export.")

        # Report
        cluster_sizes, porosity, num_clusters = extract_statistics_from_volume(largest_cluster)
        logger.info(f"Statistik extrahiert: {num_clusters} Cluster, Porosität {porosity:.2%}")

        report_path = os.path.join(PDF_REPORT_DIR, f"analysis_report_{timestamp}.pdf")
        if test_mode not in ("dry",):
            from src.reporting import generate_report
            generate_report(cluster_sizes, porosity, report_path)
            logger.info(f"Analyse-Report gespeichert: {report_path}")
        else:
            logger.info(f"[Dry] Skip saving analysis report.")

        logger.info("Processing completed successfully.")

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    start_time = timeit.default_timer()
    with open(os.path.join(LOGS_DIR, "fault_log.txt"), "w") as f:
        faulthandler.enable(file=f)
    faulthandler.enable()
    try:
        parser = argparse.ArgumentParser(description="BoneSimulation Pipeline")
        parser.add_argument("--test", type=str, default="none",
                            help="Testmodus: none, dry, small, fast_debug, complete")
        args = parser.parse_args()

        directory = get_base_path()
        process_and_visualize(directory, test_mode=args.test)

        elapsed = timeit.default_timer() - start_time
        logger.info(f"Total processing time: {elapsed:.2f} seconds.")
    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
