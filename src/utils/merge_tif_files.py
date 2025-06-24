import os
from PIL import Image


def create_3d_tiff(input_folder, output_file):
    # Liste der TIFF-Dateien im Eingangsordner
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') or f.endswith('.tiff')]

    # Sortiere die Dateien, falls nötig
    tiff_files.sort()

    # Liste zum Speichern der Bilder
    images = []

    # Lade jede TIFF-Datei und füge sie zur Liste hinzu
    for tiff_file in tiff_files:
        file_path = os.path.join(input_folder, tiff_file)
        img = Image.open(file_path)
        images.append(img)

    # Speichere die Bilder als 3D-TIFF
    if images:
        images[0].save(output_file, save_all=True, append_images=images[1:], compression='tiff_lzw')
        print(f"3D TIFF-Datei wurde erfolgreich erstellt: {output_file}")
    else:
        print("Keine TIFF-Dateien gefunden.")


# Beispielaufruf
input_folder = '/home/mathias/PycharmProjects/BoneSimulation/data/bigdataset'  # Ersetze dies durch den Pfad zu deinem Ordner
output_file = 'output_3d.tiff'  # Ersetze dies durch den gewünschten Ausgabepfad
create_3d_tiff(input_folder, output_file)
