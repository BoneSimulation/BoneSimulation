import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import optuna
import tifffile
from scipy.ndimage import binary_closing, label
from skimage.measure import marching_cubes
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm
from numba import njit
import os

from xgboost import XGBRegressor

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit
def fast_triangle_area(v0, v1, v2):
    # NumPy-Array Inputs (3D Punkte)
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.sqrt(np.dot(cross, cross))

def extract_features_cpu(volume, threshold, kernel_size, scaling):
    bin_vol = (volume > threshold).astype(np.uint8)
    closed_vol = binary_closing(bin_vol, structure=np.ones((kernel_size, kernel_size, kernel_size)))
    porosity = 1.0 - closed_vol.mean()

    labeled_array, num_clusters = label(closed_vol)
    cluster_sizes = np.bincount(labeled_array.flatten())[1:]
    largest_cluster_size = cluster_sizes.max() if cluster_sizes.size > 0 else 0

    try:
        verts, faces, _, _ = marching_cubes(closed_vol, level=0.5)
        surface_area = 0.0
        for f in faces:
            surface_area += fast_triangle_area(verts[f[0]], verts[f[1]], verts[f[2]])
    except Exception as e:
        logger.warning(f"Surface area calculation failed: {e}")
        surface_area = 0.0

    scaled_feature = scaling * porosity

    return np.array([
        porosity,
        num_clusters,
        largest_cluster_size,
        surface_area,
        scaled_feature,
        kernel_size
    ])

def extract_features_gpu(volume, threshold, kernel_size, scaling):
    # GPU-Array
    gpu_vol = cp.asarray(volume)

    # 1. Thresholding
    bin_vol = (gpu_vol > threshold).astype(cp.uint8)

    # 2. Morphologisches Closing (CPU → GPU)
    closed_vol_cpu = binary_closing(cp.asnumpy(bin_vol), structure=np.ones((kernel_size, kernel_size, kernel_size)))
    closed_vol_gpu = cp.asarray(closed_vol_cpu)

    # 3. Porosität (GPU)
    porosity = 1.0 - cp.mean(closed_vol_gpu)

    # 4. Cluster-Analyse → CPU
    labeled_array_cpu, num_clusters = label(closed_vol_cpu)
    cluster_sizes = np.bincount(labeled_array_cpu.flatten())[1:]
    largest_cluster_size = cluster_sizes.max() if cluster_sizes.size > 0 else 0

    # 5. Surface Area (CPU)
    try:
        verts, faces, _, _ = marching_cubes(closed_vol_cpu, level=0.5)
        surface_area = 0.0
        for f in faces:
            surface_area += fast_triangle_area(verts[f[0]], verts[f[1]], verts[f[2]])
    except Exception as e:
        logger.warning(f"Surface area calculation failed: {e}")
        surface_area = 0.0

    # 6. Features zusammenbauen
    features = np.array([
        scaling * float(porosity),
        num_clusters,
        largest_cluster_size,
        surface_area,
        scaling * float(porosity),
        kernel_size
    ])

    return features



def extract_features_parallel(blocks, threshold, kernel_size, scaling, use_gpu=False, max_workers=8):
    extractor = extract_features_gpu if (use_gpu and GPU_AVAILABLE) else extract_features_cpu
    features = []

    logger.info(f"Starte Feature-Extraktion mit {'GPU' if use_gpu else 'CPU'} und {max_workers} Arbeitern...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extractor, block, threshold, kernel_size, scaling): idx for idx, block in enumerate(blocks)}
        for future in tqdm(as_completed(futures), total=len(blocks), desc="Features extrahieren"):
            features.append(future.result())

    features = np.vstack(features)
    return features

def extract_blocks(volume, block_shape=(128, 256, 256), min_active_voxels=1000):
    z_max, y_max, x_max = volume.shape
    bz, by, bx = block_shape
    blocks = []
    logger.info(f"Starte Block-Extraktion mit Blockgröße {block_shape}...")
    for z in range(0, z_max - bz + 1, bz):
        for y in range(0, y_max - by + 1, by):
            for x in range(0, x_max - bx + 1, bx):
                block = volume[z:z+bz, y:y+by, x:x+bx]
                if np.sum(block) > min_active_voxels:
                    blocks.append(block)
    logger.info(f"{len(blocks)} Blöcke extrahiert aus Volume.")
    return blocks

def load_data(volume_path, block_shape=(128, 256, 256)):
    logger.info(f"Lade Volume von '{volume_path}' ...")
    vol = tifffile.imread(volume_path)
    logger.info(f"Volume geladen: shape = {vol.shape}")

    blocks = extract_blocks(vol, block_shape=block_shape)

    logger.info("Shuffle Blöcke...")
    np.random.shuffle(blocks)

    logger.info("Berechne Labels für Blöcke...")
    labels = []
    for block in tqdm(blocks, desc="Labels berechnen"):
        bin_vol = (block > 0.5).astype(np.uint8)
        closed_vol = binary_closing(bin_vol, structure=np.ones((3, 3, 3)))
        porosity = 1.0 - closed_vol.mean()

        labeled_array, _ = label(closed_vol)
        cluster_sizes = np.bincount(labeled_array.flatten())[1:]
        largest_cluster_size = cluster_sizes.max() if cluster_sizes.size > 0 else 0

        label_value = porosity * 1000 + largest_cluster_size * 0.01
        labels.append(label_value)

    y_labels = np.array(labels)
    logger.info(f"{len(blocks)} Blöcke, {len(y_labels)} Labels berechnet.")
    return blocks, y_labels

def objective(trial):
    threshold = trial.suggest_float("threshold", 0.1, 0.9)
    kernel_size = trial.suggest_int("kernel_size", 2, 8)
    scaling = trial.suggest_float("scaling", 0.3, 1.5)

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    n_estimators = trial.suggest_int("n_estimators", 50, 200)

    logger.info(f"Feature-Engineering mit params: threshold={threshold:.3f}, kernel_size={kernel_size}, scaling={scaling:.3f}")

    features = extract_features_parallel(X_blocks, threshold, kernel_size, scaling, use_gpu=True, max_workers=8)

    model = XGBRegressor(
        tree_method='gpu_hist' if GPU_AVAILABLE else 'hist',
        predictor='gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        verbosity=0,
        use_label_encoder=False
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, features, y_labels, scoring="neg_mean_squared_error", cv=cv)
    mean_score = np.mean(scores)

    logger.info(f"Trial: score={mean_score:.4f}, params={trial.params}")
    return mean_score

if __name__ == "__main__":
    volume_path = "/home/mathias/PycharmProjects/BoneSimulation/src/utils/output_3d.tiff"
    block_shape = (128, 256, 256)
    n_trials = 50

    X_blocks, y_labels = load_data(volume_path, block_shape=block_shape)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\nBeste Parameter:")
    print(study.best_params)
    print(f"Bestes Score (neg. MSE): {study.best_value:.4f}")
