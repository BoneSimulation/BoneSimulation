import os
import numpy as np
from PIL import Image
from skimage import filters, io
import matplotlib.pyplot as plt

directory = "/home/mathias/PycharmProjects/BoneSimulation/data/dataset"

data_list = []

# Laden der Bilder
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".tif"):
        filepath = os.path.join(directory, filename)
        im = Image.open(filepath)
        im_array = np.array(im)
        data_list.append(im_array)

# 3D-Array erstellen
data_array = np.array(data_list)
print("Datenarray hat die Form:", data_array.shape)

# Gaußscher Filter
image_blurred_list = [filters.gaussian(image, sigma=2, mode='constant', cval=0) for image in data_array]
image_blurred_array = np.array(image_blurred_list)

# Binarisierung
binary_images = [image > filters.threshold_otsu(image) for image in image_blurred_array]
binary_image_array = np.array(binary_images)

# Anzeigen der überarbeiteten Bilder
num_images = image_blurred_array.shape[0]
cols = 3
rows = (num_images // cols) + (num_images % cols > 0)

#/ plt.figure(figsize=(15, 5 * rows))

for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(image_blurred_array[i], cmap='gray')  # Verwende cmap='gray' für Graustufenbilder
    plt.title(f'Überarbeitetes Bild {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Anzeigen der binarisierten Bilder
plt.figure(figsize=(15, 5 * rows))

for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(binary_image_array[i], cmap='gray')  # Verwende cmap='gray' für Graustufenbilder
    plt.title(f'Binarisiertes Bild {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Histogramm der Pixelwerte plotten
plt.figure(figsize=(10, 6))

# Alle Pixelwerte in einem Array sammeln
all_pixel_values = image_blurred_array.flatten()

# Histogramm erstellen
plt.hist(all_pixel_values, bins=256, range=(0, 1), color='gray', alpha=0.7)

# Achsen und Titel hinzufügen
plt.title('Histogramm der Pixelwerte der überarbeiteten Bilder')
plt.xlabel('Pixelwerte')
plt.ylabel('Häufigkeit')
plt.xlim(0, 1)

plt.grid()
plt.show()


def plotting():



def main():
    print("Program started")
