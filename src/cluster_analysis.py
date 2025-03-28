import logging
import ciclope.core.tetraFE
import numpy as np
from skimage import morphology, filters, measure
import scipy.ndimage
import tifffile as tiff
from PIL import Image
import vtk
import meshio
import os
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def find_largest_cluster(binary_image_stack):
    """Findet den größten verbundenen Cluster im binären Bild."""
    labels, num_clusters = measure.label(binary_image_stack, return_num=True, connectivity=1)
    cluster_sizes = np.bincount(labels.ravel())
    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = labels == largest_cluster_label
    logger.info(f"Found largest cluster with size {cluster_sizes[largest_cluster_label]}.")
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]