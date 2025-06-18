import matplotlib.pyplot as plt
from fpdf import FPDF

import matplotlib.pyplot as plt
from fpdf import FPDF
import os

def generate_report(cluster_sizes, porosity, output_pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Knochen Analyse Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Anzahl Cluster: {len(cluster_sizes)}", ln=True)
    pdf.cell(0, 10, f"Porosität: {porosity:.2%}", ln=True)
    pdf.cell(0, 10, f"Maximale Clustergröße: {max(cluster_sizes) if cluster_sizes else 0} Voxel", ln=True)
    pdf.cell(0, 10, f"Minimale Clustergröße: {min(cluster_sizes) if cluster_sizes else 0} Voxel", ln=True)
    pdf.cell(0, 10, f"Mittlere Clustergröße: {np.mean(cluster_sizes) if cluster_sizes else 0:.1f} Voxel", ln=True)

    # Erstelle Histogramm und speichere temporär
    plt.figure(figsize=(5,3))
    plt.hist(cluster_sizes, bins=20, color='skyblue')
    plt.title("Clustergrößenverteilung")
    plt.xlabel("Voxel")
    plt.ylabel("Anzahl Cluster")
    plt.tight_layout()
    hist_path = output_pdf_path.replace(".pdf", "_hist.png")
    plt.savefig(hist_path)
    plt.close()

    pdf.image(hist_path, x=10, y=70, w=pdf.w - 20)
    pdf.output(output_pdf_path)

    # Lösche temporäre Bilddatei
    if os.path.exists(hist_path):
        os.remove(hist_path)


import numpy as np
from scipy.ndimage import label

def extract_statistics_from_volume(binary_volume):
    """
    Berechnet aus einem binären 3D-Volume die Clustergrößen und Porosität.

    Args:
        binary_volume (np.ndarray): 3D Binärmaske (1 = Knochen, 0 = Hintergrund)

    Returns:
        cluster_sizes (list of int): Liste der Clustergrößen in Voxelanzahl
        porosity (float): Anteil leerer Voxel (zwischen 0 und 1)
        num_clusters (int): Anzahl der Cluster
    """
    structure = np.ones((3,3,3))  # 26er Nachbarschaft
    labeled_array, num_clusters = label(binary_volume, structure=structure)

    cluster_sizes = [np.sum(labeled_array == i) for i in range(1, num_clusters + 1)]

    total_voxels = binary_volume.size
    bone_voxels = np.sum(binary_volume)
    porosity = 1.0 - (bone_voxels / total_voxels)

    return cluster_sizes, porosity, num_clusters
