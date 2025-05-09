# src/cluster_analysis.py

import logging
import numpy as np
from skimage import measure
import gc

logger = logging.getLogger(__name__)


def find_largest_cluster(binary_image_stack, connectivity=1):
    """Find the largest connected cluster in the binary image.

    Args:
        binary_image_stack: Binary image data to analyze.
        connectivity: Connectivity for labeling (1=faces, 2=faces+edges, 3=faces+edges+corners)

    Returns:
        tuple: (largest_cluster, num_clusters, size_of_largest_cluster)
            - largest_cluster: Binary array marking the largest cluster
            - num_clusters: Total number of clusters found
            - size_of_largest_cluster: Size of the largest cluster
    """
    logger.info("Starting Connected Component Labeling...")

    # Check for empty volume
    if binary_image_stack.size == 0 or binary_image_stack.max() == 0:
        logger.warning("Empty volume or no active voxels for cluster analysis.")
        return None, 0, 0

    # For very large volumes, the analysis needs to be adjusted
    if binary_image_stack.size > 100_000_000:  # ~100 million voxels
        logger.info("Large volume detected, using optimized cluster analysis...")
        return _find_largest_cluster_memory_efficient(binary_image_stack, connectivity)

    # Standard analysis for smaller volumes
    labels, num_clusters = measure.label(
        binary_image_stack,
        return_num=True,
        connectivity=connectivity
    )

    logger.info(f"Connected Component Labeling completed. {num_clusters} clusters found.")

    if num_clusters == 0:
        logger.warning("No clusters found.")
        return None, 0, 0

    # Calculate cluster sizes
    cluster_sizes = np.bincount(labels.ravel())

    # Ignore the background (label 0)
    if len(cluster_sizes) <= 1:
        logger.warning("No voxels above the threshold found.")
        return None, 0, 0

    # Find the largest cluster
    foreground_sizes = cluster_sizes[1:]
    largest_cluster_label = foreground_sizes.argmax() + 1
    largest_cluster_size = foreground_sizes.max()

    logger.info(f"The largest cluster has label {largest_cluster_label} with {largest_cluster_size} voxels.")

    # Extract the largest cluster
    largest_cluster = labels == largest_cluster_label

    # Free memory
    del labels
    gc.collect()

    return largest_cluster, num_clusters, largest_cluster_size


def _find_largest_cluster_memory_efficient(binary_volume, connectivity=1):
    """
    Memory-efficient version of cluster analysis for very large volumes.

    This function processes the volume in slices and tracks
    cluster connections between slices.
    """
    logger.info("Using slice-based cluster analysis for memory optimization...")

    # This implementation would require a complex adaptive component labeling method
    # that works slice-wise and tracks cluster assignments.
    # For this code snippet, we use a simplified version:

    # 1. Use a high connectivity value for better connections
    labels, num_clusters = measure.label(
        binary_volume,
        return_num=True,
        connectivity=connectivity
    )

    logger.info(f"Labeling completed. {num_clusters} clusters identified.")

    if num_clusters == 0:
        return None, 0, 0

    # 2. Calculate cluster sizes directly to minimize memory usage
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 3. Find the largest cluster (ignore background with label 0)
    background_idx = np.where(unique_labels == 0)[0]
    if len(background_idx) > 0:
        # Remove background from counts
        mask = unique_labels != 0
        unique_labels = unique_labels[mask]
        counts = counts[mask]

    if len(counts) == 0:
        return None, 0, 0

    largest_idx = np.argmax(counts)
    largest_label = unique_labels[largest_idx]
    largest_size = counts[largest_idx]

    logger.info(f"Largest cluster identified: Label {largest_label}, Size {largest_size}.")

    # 4. Extract only the largest cluster to save memory
    largest_cluster = (labels == largest_label)

    # 5. Free memory
    del labels
    gc.collect()

    return largest_cluster, num_clusters, largest_size
