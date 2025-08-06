import os
import numpy as np
import vtk
import vtkmodules.util.numpy_support as numpy_support

from skimage.filters import threshold_otsu, gaussian
from skimage import morphology, measure
from scipy import ndimage

from mesh_generation import reverse_binary, generate_tetrahedral_mesh


def numpy2vti(array: np.ndarray, output_filename: str):
    """Convert a 3D NumPy array to VTI (VTK ImageData) and save it."""
    if array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional.")

    dims = array.shape
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(dims[::-1])  # (z, y, x) → (x, y, z)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)
    vtk_image.SetOrigin(0.0, 0.0, 0.0)

    vtk_array = numpy_support.numpy_to_vtk(
        num_array=array.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )
    vtk_image.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(vtk_image)
    if not writer.Write():
        raise IOError(f"Failed to write VTI file: {output_filename}")


def debug_volume_info(volume: np.ndarray, block_size_mm: float = 30.0, voxel_size_mm: float = 0.05):
    """Debug-Funktion um Volume-Informationen anzuzeigen"""
    print("=== VOLUME DEBUG INFO ===")
    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Volume min/max: {volume.min():.2f} / {volume.max():.2f}")
    print(f"Volume mean: {volume.mean():.2f}")

    # Berechne Block-Größe in Voxeln
    block_size_voxels = int(block_size_mm / voxel_size_mm)
    print(f"Block size: {block_size_mm}mm = {block_size_voxels} voxels")

    # Berechne mögliche Blöcke
    z_max, y_max, x_max = volume.shape
    blocks_z = z_max // block_size_voxels
    blocks_y = y_max // block_size_voxels
    blocks_x = x_max // block_size_voxels

    print(f"Possible blocks per dimension: z={blocks_z}, y={blocks_y}, x={blocks_x}")
    print(f"Total possible blocks: {blocks_z * blocks_y * blocks_x}")

    # Empfehlungen
    if blocks_z * blocks_y * blocks_x == 0:
        print("\n=== RECOMMENDATIONS ===")
        print("No blocks can be extracted with current settings!")

        # Berechne maximale mögliche Block-Größe
        max_block_size_voxels = min(z_max, y_max, x_max)
        max_block_size_mm = max_block_size_voxels * voxel_size_mm
        print(f"Maximum possible block size: {max_block_size_voxels} voxels = {max_block_size_mm:.1f}mm")

        # Empfehle kleinere Block-Größen
        for suggested_mm in [25.0, 20.0, 15.0, 10.0, 5.0]:
            suggested_voxels = int(suggested_mm / voxel_size_mm)
            if suggested_voxels <= max_block_size_voxels:
                blocks_possible = (z_max // suggested_voxels) * (y_max // suggested_voxels) * (
                            x_max // suggested_voxels)
                print(f"With {suggested_mm}mm ({suggested_voxels} voxels): {blocks_possible} blocks possible")

    print("========================\n")


def extract_blocks_raster(
        volume: np.ndarray,
        block_size_voxels: int,
        output_dir: str,
        voxel_spacing=(0.05, 0.05, 0.05),
        write_tetra_mesh=True,
        min_voxels_threshold: int = 100
):
    """Extract cubic blocks from volume, save as VTI, optionally tetra-mesh."""
    os.makedirs(output_dir, exist_ok=True)

    # Volume-Dimensionen
    z_max, y_max, x_max = volume.shape
    print(f"Volume dimensions: {volume.shape} (z, y, x)")
    print(f"Block size in voxels: {block_size_voxels}")

    # Berechne wie viele komplette Blöcke in jede Dimension passen
    blocks_z = z_max // block_size_voxels
    blocks_y = y_max // block_size_voxels
    blocks_x = x_max // block_size_voxels

    print(f"Blocks per dimension: z={blocks_z}, y={blocks_y}, x={blocks_x}")

    total_blocks = blocks_z * blocks_y * blocks_x
    print(f"Total cubic blocks to extract: {total_blocks}")

    if total_blocks == 0:
        print(f"ERROR: No blocks can be extracted!")
        print(f"Volume size: {volume.shape}")
        print(f"Block size: {block_size_voxels}")
        print(f"Consider reducing block_size_voxels or check your volume dimensions")
        return 0, 0

    counter = 0
    successful_blocks = 0

    # Iteriere über alle möglichen Blöcke
    for i in range(blocks_z):
        z_start = i * block_size_voxels
        z_end = z_start + block_size_voxels

        for j in range(blocks_y):
            y_start = j * block_size_voxels
            y_end = y_start + block_size_voxels

            for k in range(blocks_x):
                x_start = k * block_size_voxels
                x_end = x_start + block_size_voxels

                # Extrahiere den Block
                block = volume[z_start:z_end, y_start:y_end, x_start:x_end]

                # Überprüfe die Block-Größe
                expected_shape = (block_size_voxels, block_size_voxels, block_size_voxels)
                if block.shape != expected_shape:
                    print(f"WARNING: Block {counter} has wrong shape {block.shape}, expected {expected_shape}")
                    print(f"Position: z={z_start}:{z_end}, y={y_start}:{y_end}, x={x_start}:{x_end}")
                    counter += 1
                    continue

                # Speichere rohen Block als VTI
                vti_path = os.path.join(output_dir, f"block_{counter:04d}.vti")
                try:
                    numpy2vti(block.astype(np.float32), vti_path)
                    print(f"[{counter:04d}] Saved VTI: {vti_path}")
                except Exception as e:
                    print(f"[{counter:04d}] ERROR saving VTI: {e}")
                    counter += 1
                    continue

                # Preprocessing für Tetraeder-Meshing
                if write_tetra_mesh:
                    try:
                        # Gaussian smoothing
                        smoothed = gaussian(block, sigma=1, preserve_range=True)

                        # Otsu threshold
                        threshold = threshold_otsu(smoothed)
                        binary = smoothed < threshold  # Annahme: dunkle Bereiche sind Material

                        # Morphological closing
                        binary = morphology.closing(binary, footprint=morphology.ball(3))

                        # Finde größte zusammenhängende Komponente
                        labels, num_labels = measure.label(binary, return_num=True, connectivity=1)

                        if num_labels == 0:
                            print(f"[{counter:04d}] No components found after thresholding.")
                            counter += 1
                            continue

                        # Finde die größte Komponente (außer Hintergrund)
                        counts = np.bincount(labels.ravel())
                        if len(counts) <= 1:
                            print(f"[{counter:04d}] Only background found.")
                            counter += 1
                            continue

                        # Index 0 ist Hintergrund, also nehme den größten aus 1:
                        largest_label = counts[1:].argmax() + 1
                        mask = (labels == largest_label)

                        voxels_in_mask = np.sum(mask)
                        print(f"[{counter:04d}] Largest component: {voxels_in_mask} voxels")

                        if voxels_in_mask < min_voxels_threshold:
                            print(
                                f"[{counter:04d}] Skipped: {voxels_in_mask} voxels (below threshold {min_voxels_threshold})")
                            counter += 1
                            continue

                        # Tetraeder-Meshing
                        reversed_mask = reverse_binary(mask.astype(np.uint8))
                        tetra_path = os.path.join(output_dir, f"block_{counter:04d}_tetra.vtk")

                        tetra_mesh = generate_tetrahedral_mesh(reversed_mask, 0.1, tetra_path)
                        if tetra_mesh:
                            print(f"[{counter:04d}] Tetra mesh saved: {tetra_path}")
                            successful_blocks += 1
                        else:
                            print(f"[{counter:04d}] Tetra mesh FAILED!")

                    except Exception as e:
                        print(f"[{counter:04d}] ERROR in tetra meshing: {e}")

                counter += 1

    print(f"\nFinished extracting {counter} cubic blocks.")
    print(f"Successfully created {successful_blocks} tetrahedral meshes.")

    return counter, successful_blocks


def improved_block_extraction_integration(largest_cluster, timestamp, test_mode, output_base_dir="pictures"):
    """Verbesserte Integration der Block-Extraktion in die Hauptpipeline"""

    # Debug-Informationen
    debug_volume_info(largest_cluster)

    # Konfiguration
    block_size_mm = 30.0
    voxel_spacing = (0.05, 0.05, 0.05)  # mm
    voxel_size_mm = voxel_spacing[0]  # Annahme: isotrope Voxel

    # Berechne Block-Größe in Voxeln
    block_size_voxels = int(block_size_mm / voxel_size_mm)

    # Überprüfe ob Block-Extraktion möglich ist
    z_max, y_max, x_max = largest_cluster.shape
    possible_blocks = (z_max // block_size_voxels) * (y_max // block_size_voxels) * (x_max // block_size_voxels)

    if possible_blocks == 0:
        print(f"WARNING: Cannot extract {block_size_mm}mm blocks from volume {largest_cluster.shape}")
        # Automatisch kleinere Block-Größe wählen
        max_possible_size = min(z_max, y_max, x_max)
        new_block_size_mm = (max_possible_size * voxel_size_mm) * 0.8  # 80% der maximalen Größe
        block_size_voxels = int(new_block_size_mm / voxel_size_mm)
        print(f"Automatically using smaller block size: {new_block_size_mm:.1f}mm ({block_size_voxels} voxels)")

    # Erstelle Output-Verzeichnis
    blocks_output_dir = os.path.join(output_base_dir, f"blocks_{timestamp}")

    if test_mode != "dry":
        num_blocks, successful_meshes = extract_blocks_raster(
            volume=largest_cluster,
            block_size_voxels=block_size_voxels,
            output_dir=blocks_output_dir,
            voxel_spacing=voxel_spacing,
            write_tetra_mesh=True,
            min_voxels_threshold=100  # Mindestanzahl Voxel für Meshing
        )
        print(f"Block extraction completed: {num_blocks} total, {successful_meshes} with meshes")
        return blocks_output_dir, num_blocks, successful_meshes
    else:
        print("[Dry] Skipping block extraction.")
        return blocks_output_dir, 0, 0