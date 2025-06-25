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

def extract_blocks_raster(
    volume: np.ndarray,
    block_size_voxels: int,
    output_dir: str,
    voxel_spacing=(0.05, 0.05, 0.05),
    write_tetra_mesh=True,
    min_voxels_threshold: int = 99999
):
    """Extract cubic blocks from volume, save as VTI, optionally tetra-mesh."""
    os.makedirs(output_dir, exist_ok=True)

    z_max, y_max, x_max = volume.shape
    blocks_per_dim = (
        z_max // block_size_voxels,
        y_max // block_size_voxels,
        x_max // block_size_voxels
    )
    total_blocks = blocks_per_dim[0] * blocks_per_dim[1] * blocks_per_dim[2]
    print(f"Total cubic blocks to extract: {total_blocks}")

    counter = 0
    for i in range(blocks_per_dim[0]):
        z_start = i * block_size_voxels
        z_end = z_start + block_size_voxels
        for j in range(blocks_per_dim[1]):
            y_start = j * block_size_voxels
            y_end = y_start + block_size_voxels
            for k in range(blocks_per_dim[2]):
                x_start = k * block_size_voxels
                x_end = x_start + block_size_voxels

                block = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                if block.shape != (block_size_voxels, block_size_voxels, block_size_voxels):
                    raise RuntimeError(
                        f"Wrong block shape {block.shape} at (i={i}, j={j}, k={k})"
                    )

                # Save raw block
                vti_path = os.path.join(output_dir, f"block_{counter:04d}.vti")
                numpy2vti(block, vti_path)
                print(f"[{counter:04d}] Saved VTI: {vti_path}")

                # Preprocessing: Gaussian smoothing → Otsu threshold → morph closing → largest component
                smoothed = gaussian(block, sigma=1, preserve_range=True)
                threshold = threshold_otsu(smoothed)
                binary = smoothed < threshold
                binary = morphology.closing(binary, footprint=morphology.ball(3))

                labels, _ = measure.label(binary, return_num=True, connectivity=1)
                counts = np.bincount(labels.ravel())

                if len(counts) <= 1:
                    print(f"[{counter:04d}] No components found.")
                    counter += 1
                    continue

                largest_label = counts[1:].argmax() + 1
                mask = labels == largest_label

                voxels_in_mask = np.sum(mask)
                if voxels_in_mask < min_voxels_threshold:
                    print(f"[{counter:04d}] Skipped: {voxels_in_mask} voxels (too small)")
                    counter += 1
                    continue

                # Tetra-Meshing
                if write_tetra_mesh:
                    reversed_mask = reverse_binary(mask.astype(np.uint8))  # or float if required
                    tetra_path = os.path.join(output_dir, f"block_{counter:04d}_tetra.vtk")
                    tetra_mesh = generate_tetrahedral_mesh(reversed_mask, 0.1, tetra_path)
                    if tetra_mesh:
                        print(f"[{counter:04d}] Tetra mesh saved: {tetra_path}")
                    else:
                        print(f"[{counter:04d}] Tetra mesh FAILED!")

                counter += 1

    print(f"Finished extracting {counter} cubic blocks.")
