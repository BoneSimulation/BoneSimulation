from mayavi import mlab
import numpy as np
from PIL import Image
import os
from skimage import filters
import matplotlib.pyplot as plt

directory = "/home/mathias/PycharmProjects/BoneSimulation/data/dataset"

data_list = []

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".tif"):
        filepath = os.path.join(directory, filename)
        im = Image.open(filepath)
        im_array = np.array(im)

        data_list.append(im_array)

# 3d array
data_array = np.array(data_list)

print("Datenarray hat die Form:", data_array.shape)

# Gaußscher Filter
image_blurred_list = [filters.gaussian(image, sigma=2, mode='constant', cval=0) for image in data_array]

# back to numpy array
image_blurred_array = np.array(image_blurred_list)

# mayavi
mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
mlab.volume_slice(image_blurred_array, plane_orientation='x_axes', slice_index=100)
mlab.volume_slice(image_blurred_array, plane_orientation='y_axes', slice_index=100)
mlab.volume_slice(image_blurred_array, plane_orientation='z_axes', slice_index=100)

mlab.show()
