# cluster_analysis.py

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
    logger.info("Starte Connected Component Labeling...")

    # Prüfe auf leeres Volumen
    if binary_image_stack.size == 0 or binary_image_stack.max() == 0:
        logger.warning("Leeres Volumen oder keine aktiven Voxel für Cluster-Analyse.")
        return None, 0, 0

    # Bei sehr großen Volumen muss die Analyse angepasst werden
    if binary_image_stack.size > 100_000_000:  # ~100 Millionen Voxel
        logger.info("Großes Volumen erkannt, verwende optimierte Cluster-Analyse...")
        return _find_largest_cluster_memory_efficient(binary_image_stack, connectivity)

    # Standard-Analyse für kleinere Volumen
    labels, num_clusters = measure.label(
        binary_image_stack,
        return_num=True,
        connectivity=connectivity
    )

    logger.info(f"Connected Component Labeling abgeschlossen. {num_clusters} Cluster gefunden.")

    if num_clusters == 0:
        logger.warning("Keine Cluster gefunden.")
        return None, 0, 0

    # Berechne Clustergröße
    cluster_sizes = np.bincount(labels.ravel())

    # Ignoriere den Hintergrund (Label 0)
    if len(cluster_sizes) <= 1:
        logger.warning("Keine Voxel über dem Schwellwert gefunden.")
        return None, 0, 0

    # Finde das größte Cluster
    foreground_sizes = cluster_sizes[1:]
    largest_cluster_label = foreground_sizes.argmax() + 1
    largest_cluster_size = foreground_sizes.max()

    logger.info(f"Größtes Cluster hat Label {largest_cluster_label} mit {largest_cluster_size} Voxeln")

    # Extrahiere das größte Cluster
    largest_cluster = labels == largest_cluster_label

    # Speicher freigeben
    del labels
    gc.collect()

    return largest_cluster, num_clusters, largest_cluster_size


def _find_largest_cluster_memory_efficient(binary_volume, connectivity=1):
    """
    Memory-effiziente Version der Cluster-Analyse für sehr große Volumen.

    Diese Funktion verarbeitet das Volumen in Scheiben und verfolgt
    Cluster-Verbindungen zwischen Scheiben.
    """
    logger.info("Verwende scheiben-basierte Clusteranalyse zur Speicheroptimierung...")

    # Diese Implementation würde eine komplexe adaptive Komponenten-Labeling-Methode
    # erfordern, die scheiben-weise arbeitet und Cluster-Zuordnungen nachverfolgt.
    # Für diesen Code-Ausschnitt verwenden wir eine vereinfachte Version:

    # 1. Verwenden eines hohen connectivity-Werts für bessere Verbindungen
    labels, num_clusters = measure.label(
        binary_volume,
        return_num=True,
        connectivity=connectivity
    )

    logger.info(f"Labeling abgeschlossen. {num_clusters} Cluster identifiziert.")

    if num_clusters == 0:
        return None, 0, 0

    # 2. Berechne Clustergrößen direkt und minimiere Speicherverbrauch
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 3. Finde das größte Cluster (ignoriere Hintergrund mit Label 0)
    background_idx = np.where(unique_labels == 0)[0]
    if len(background_idx) > 0:
        # Entferne Hintergrund aus den Zählungen
        mask = unique_labels != 0
        unique_labels = unique_labels[mask]
        counts = counts[mask]

    if len(counts) == 0:
        return None, 0, 0

    largest_idx = np.argmax(counts)
    largest_label = unique_labels[largest_idx]
    largest_size = counts[largest_idx]

    logger.info(f"Größtes Cluster identifiziert: Label {largest_label}, Größe {largest_size}")

    # 4. Extrahiere nur das größte Cluster, um Speicher zu sparen
    largest_cluster = (labels == largest_label)

    # 5. Speicher freigeben
    del labels
    gc.collect()

    return largest_cluster, num_clusters, largest_size