from mayavi import mlab
import numpy as np
from PIL import Image
import os
from skimage import filters
import warnings

# Warnungen für falsche Farbprofile unterdrücken
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")

directory = "/home/mathias/PycharmProjects/BoneSimulation/data/dataset"

data_list = []

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".tif"):
        filepath = os.path.join(directory, filename)
        im = Image.open(filepath).convert("L")  # "L" = Graustufen
        im_array = np.array(im)
        data_list.append(im_array)

# 3d array
data_array = np.array(data_list)
print("Datenarray hat die Form:", data_array.shape)

# Gaußscher Filter
image_blurred_array = np.array([filters.gaussian(image, sigma=2, mode='constant', cval=0) for image in data_array])


mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

mlab.contour3d(image_blurred_array, contours=8, opacity=0.5, colormap='cool')

mlab.show()
