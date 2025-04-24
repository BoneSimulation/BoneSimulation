import logging
import numpy as np
from skimage import measure


logger = logging.getLogger(__name__)


def find_largest_cluster(binary_image_stack):
    """Find the largest connected cluster in the binary image.

    Args:
        binary_image_stack: Binary image data to analyze.

    Returns:
        tuple: (largest_cluster, num_clusters, size_of_largest_cluster)
            - largest_cluster: Binary array marking the largest cluster
            - num_clusters: Total number of clusters found
            - size_of_largest_cluster: Size of the largest cluster
    """
    labels, num_clusters = measure.label(
        binary_image_stack, 
        return_num=True, 
        connectivity=1
    )
    cluster_sizes = np.bincount(labels.ravel())

    if len(cluster_sizes) <= 1:
        logger.warning("No clusters found.")
        return None, num_clusters, 0

    largest_cluster_label = cluster_sizes[1:].argmax() + 1
    largest_cluster = labels == largest_cluster_label
    logger.info(f"Found largest cluster with size {cluster_sizes[largest_cluster_label]}.")
    return largest_cluster, num_clusters, cluster_sizes[largest_cluster_label]
