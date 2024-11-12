from mayavi import mlab
import numpy as np
from PIL import Image
import os

directory = "/home/mathias/PycharmProjects/BoneSimulation/data/dataset/dataset"

data_list = []

# Lade alle Bilder in ein NumPy-Array
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".tif"):
        filepath = os.path.join(directory, filename)

        # Bild öffnen und in ein NumPy-Array konvertieren
        im = Image.open(filepath)
        im_array = np.array(im)

        data_list.append(im_array)

# Die Liste in ein NumPy-Array konvertieren (3D-Array: [Bilder, Höhe, Breite])
data_array = np.array(data_list)

print("Datenarray hat die Form:", data_array.shape)

# Gaußscher Filter auf jedes Bild anwenden
image_blurred_list = [filters.gaussian(image, sigma=2, mode='constant', cval=0) for image in data_array]

# Umwandeln der Liste zurück in ein NumPy-Array
image_blurred_array = np.array(image_blurred_list)

# Volumenvisualisierung mit mayavi
mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
mlab.volume_slice(image_blurred_array, plane_orientation='x_axes', slice_index=100)
mlab.volume_slice(image_blurred_array, plane_orientation='y_axes', slice_index=100)
mlab.volume_slice(image_blurred_array, plane_orientation='z_axes', slice_index=100)

mlab.show()
