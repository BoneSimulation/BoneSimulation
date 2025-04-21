####################################################
# High Performance Computing Center Stuttgart (HLRS)
####################################################
#
# Created on 31/10/2024
#
# Created by Benjamin Schnabel
#
####################################################

# %%
# Imports

import sys

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mcubes
import pygalmesh

from scipy import ndimage, misc
from skimage.filters import threshold_otsu, gaussian
from skimage import measure, morphology
import meshio

from ciclope.utils.recon_utils import read_tiff_stack, plot_midplanes
from ciclope.utils.preprocess import remove_unconnected
from ciclope.core import tetraFE
import ciclope

from tifffile import imread

import vtk
from vtk.util import numpy_support
import itk
from itkwidgets import view

import ccx2paraview

# %%
# Style

plt.style.use("presentation.mplstyle")

####################################################
# Pre-Processing
####################################################

# %%
# Load input data

print("Load input data")

# Path to dataset
input_file = "../dataset/3155_D_4_bc_0000.tif"

# Voxel size of the dataset [mm]
voxel_size = 19.5e-3

# %%
# Functions

def numpy2vtk(array: np.ndarray, output_filename: str):
    """
    Convert a 3D NumPy array to a VTK file and save it.

    Parameters:
        array (np.ndarray): A 3D NumPy array.
        output_filename (str): The path where the VTK file will be saved.
    """

    # Ensure the array is 3D
    if array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional.")

    # Get the dimensions of the array
    dimensions = array.shape

    # Create a vtkImageData object
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(dimensions)
    # Adjust spacing as needed
    vtk_image_data.SetSpacing(1.0, 1.0, 1.0)
    # Adjust origin as needed
    vtk_image_data.SetOrigin(0.0, 0.0, 0.0)

    # Convert the NumPy array to a VTK array
    vtk_array = numpy_support.numpy_to_vtk(num_array=array.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT)

    # Add the VTK array to the vtkImageData object
    vtk_image_data.GetPointData().SetScalars(vtk_array)

    # Write the VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(vtk_image_data)

    # Ensure the file is written
    if not writer.Write():
        raise IOError(f"Failed to write VTK file to {output_filename}")

# %%
# Read the input data and define an array of the voxelsize

print("Read the input data and define an array of the voxelsize")

input_dir = os.path.dirname(input_file)
stack_files = []

for f in os.listdir(input_dir):
    if os.path.isfile:
        stack_files.append(os.path.join(input_dir, f))

stack_files.sort()
data_3D = imread(stack_files)

vs = np.ones(3)*voxel_size

# %%
# Inspect the dataset and plot the first tiff image

plt.figure()
plt.imshow(data_3D[0, :, :], cmap="bone")
list_of_ticks = np.linspace(0, 200, 5)
plt.xticks(list_of_ticks)
plt.yticks(list_of_ticks)
plt.colorbar()
plt.savefig("slice.pdf", dpi = "figure", format="pdf")
#plt.show()
plt.close()

# Save the array to a VTK file
numpy2vtk(data_3D, "slice.vti")

# %%
# Gaussian smooth

print("Gaussian smooth")

data_3D = gaussian(data_3D, sigma=1, preserve_range=True)

# %%
# Inspect the dataset and plot the first tiff image

plt.figure()
plt.imshow(data_3D[0, :, :], cmap="bone")
list_of_ticks = np.linspace(0, 200, 5)
plt.xticks(list_of_ticks)
plt.yticks(list_of_ticks)
plt.colorbar()
plt.savefig("slice_smoothed.pdf", dpi = "figure", format="pdf")
#plt.show()
plt.close()

# Save the array to a VTK file
numpy2vtk(data_3D, "slice_smoothed.vti")

# %%
# Thresholding

print("Thresholding")

# Use Otsu's method
T = threshold_otsu(data_3D)
print("Threshold: {}".format(T))

# Plot the histogram
plt.figure()
plt.hist(data_3D.ravel(), edgecolor="white", linewidth=0.1, bins=50)
plt.axvline(T, color="k", linestyle="dashed", linewidth=0.5, label="Threshold: {:10.2f}".format(T))
plt.xlim(left = 0, right = 255)
plt.legend()
plt.xlabel("Grayscale values")
plt.ylabel("Number of Voxels")
plt.savefig("histogram.pdf", dpi = "figure", format = "pdf")
#plt.show()
plt.close()

# %%
# Resize (optional)

print("Resize (optional)")

resampling = 2

# resize the 3D data using spline interpolation of order 2
data_3D = ndimage.zoom(data_3D, 1/resampling, output=None, order=2)

# correct voxelsize
vs = vs * resampling

# %%
# Inspect the dataset and plot the first tiff image

plt.figure()
plt.imshow(data_3D[0, :, :], cmap="bone")
list_of_ticks = np.linspace(0, 100, 5)
plt.xticks(list_of_ticks)
plt.yticks(list_of_ticks)
plt.colorbar()
plt.savefig("slice_resized.pdf", dpi = "figure", format="pdf")
#plt.show()
plt.close()

# Save the array to a VTK file
numpy2vtk(data_3D, "slice_resized.vti")

# %%
# Binarize

print("Binarize")

# Apply the threshold

BW = data_3D > T

# %%
# Inspect the dataset and plot the first tiff image

plt.figure()
plt.imshow(BW[0, :, :], cmap="gray")
list_of_ticks = np.linspace(0, 100, 5)
plt.xticks(list_of_ticks)
plt.yticks(list_of_ticks)
plt.colorbar()
plt.savefig("slice_bw.pdf", dpi = "figure", format="pdf")
#plt.show()
plt.close()

# Save the array to a VTK file
numpy2vtk(BW, "slice_bw.vti")

# %%
# Morphological close of binary image (optional)

print("Morphological close of binary image (optional)")

BW = morphology.closing(BW, footprint=morphology.ball(3))

# %%
# Inspect the dataset and plot the first tiff image

plt.figure()
plt.imshow(BW[0, :, :], cmap="gray")
list_of_ticks = np.linspace(0, 100, 5)
plt.xticks(list_of_ticks)
plt.yticks(list_of_ticks)
plt.colorbar()
plt.savefig("slice_closed.pdf", dpi = "figure", format="pdf")
#plt.show()
plt.close()

# Save the array to a VTK file
numpy2vtk(BW, "slice_closed.vti")

# %%
# Detect largest isolated cluster of voxels

print("Detect largest isolated cluster of voxels")

# 1. Label the BW 3D image
[labels, n_labels] = measure.label(BW, background=None, return_num=True, connectivity=1)

# 2. Count the occurrences of each label
occurrences = np.bincount(labels.reshape(labels.size))

# 3. Find the largest unconnected label
largest_label_id = occurrences[1:].argmax() + 1
L = labels == largest_label_id

# %%
# Inspect the dataset and plot the first tiff image

plt.figure()
plt.imshow(L[0, :, :], cmap="gray")
list_of_ticks = np.linspace(0, 100, 5)
plt.xticks(list_of_ticks)
plt.yticks(list_of_ticks)
plt.colorbar()
plt.savefig("slice_cluster.pdf", dpi = "figure", format = "pdf")
#plt.show()
plt.close()

# Save the array to a VTK file
numpy2vtk(L, "slice_cluster.vti")

# %%
# Shell mesh for visualization (optional)

print("Shell mesh for visualization (optional)")

#smoothed_L = mcubes.smooth(L)
#vertices, triangles = mcubes.marching_cubes(np.transpose(smoothed_L, [2, 1, 0]), 0)
#filename_shellmesh_out = "shellmesh.vtk"
#meshio.write_points_cells(filename_shellmesh_out, vertices.tolist(), [("triangle", triangles.tolist())])

# %%
# Generate tetrahedra mesh

print("Generate tetrahedra mesh")

filename_mesh_out = 'tetramesh.vtk'

mesh_size_factor = 1.2

mesh = tetraFE.cgal_mesh(L, vs, 'tetra', mesh_size_factor * min(vs), 2 * mesh_size_factor * min(vs))

mesh.write(filename_mesh_out)

# %%
# Generate tetrahedra-FE model with constant material properties

print("Generate tetrahedra-FE model with constant material properties")

filename_out = "tetramesh_fe.inp"

input_template = "bone.inp"

ciclope.core.tetraFE.mesh2tetrafe(mesh, input_template, filename_out)

