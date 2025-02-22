import os
import tifffile as tiff
import numpy as np
import logging
from PIL import Image

# Logging einrichten
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)

def split_tiff(input_tiff: str, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)

        # TIFF-Stream laden
        image_stack = tiff.imread(input_tiff)
        logger.info(f"Loaded TIFF file {input_tiff} with {image_stack.shape[0]} slices.")

        # Einzelne Bilder speichern
        for i, img in enumerate(image_stack):
            img_path = os.path.join(output_dir, f"slice_{i:04d}.tif")
            Image.fromarray(img).save(img_path)
            logger.info(f"Saved slice {i} to {img_path}")

    except Exception as e:
        logger.error(f"Error splitting TIFF file: {e}")

if __name__ == "__main__":
    input_tiff_path = "/home/mathias/PycharmProjects/BoneSimulation/data/Knochenprobe2stream.tiff"
    output_folder = "/home/mathias/PycharmProjects/BoneSimulation/data/bigdataset"
    split_tiff(input_tiff_path, output_folder)
