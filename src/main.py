from PIL import Image
import numpy as np
from skimage import filters
import os
import matplotlib.pyplot as plt

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

# Zeige das erste Bild nach der Weichzeichnung
plt.imshow(image_blurred_array[0], cmap='gray')
plt.axis('off')
plt.show()
