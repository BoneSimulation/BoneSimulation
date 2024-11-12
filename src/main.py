from PIL import Image
import numpy as np
from skimage import filters
import os
import matplotlib.pyplot as plt

directory = "/home/mathias/PycharmProjects/BoneSimulation/data/dataset"

data_list = []

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".tif"):
        filepath = os.path.join(directory, filename)

        im = Image.open(filepath)
        im_array = np.array(im)

        data_list.append(im_array)

data_array = np.array(data_list)

print("Shape of array:", data_array.shape)

image_blurred_list = [filters.gaussian(image, sigma=2, mode='constant', cval=0) for image in data_array]

image_blurred_array = np.array(image_blurred_list)

#plt.imshow(image_blurred_array[0], cmap='gray')
#plt.axis('off')
#plt.show()
