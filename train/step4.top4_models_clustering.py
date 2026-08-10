"""
run_top4_pipeline.py
Pipeline for running the top-4 classification models on clustering analysis.

Steps:
  1. Read top-4 models from evaluation_results
  2. Find the best fold (highest test_mcc) for each model
  3. Extract features, run PCA + KMeans clustering, and save visualizations

Requirements:
    pip install torch scikit-learn pandas numpy tqdm
"""

import os
import shutil
import traceback
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm

from cross_cluster_matching.models.models import get_model_and_transform
from cross_cluster_matching.data.data_utils import collect_reference_data, collect_image_paths, CellImageDataset
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.clustering.K_optimizer import optimize_k_for_candidate_clustering
from cross_cluster_matching.clustering.calculate_consensus_score import calculate_consensus_score
from cross_cluster_matching.clustering.matching_rule_application import identify_matching_clusters, save_matched_cells, check_matching_rules
from cross_cluster_matching.visualization.visualization import (
    count_non_background_pixels,
    visualize_all_clusters_without_images,
    visualize_all_clusters_without_images_1,
    improved_visualize_clusters_with_images,
    visualize_matching_clusters,
    analyze_cluster_composition,
    plot_similarity_heatmap,
    visualize_filtered_clusters,
)

# ── Path config ───────────────────────────────
MODELS_FOLDER      = r"/path/to/your/train_val_models"
EVALUATION_FOLDER  = r"/path/to/your/evaluation_results"
OUTPUT_BASE_DIR    = r"/path/to/your/Top4_clustering_analysis"
MALIGNANT_CELLS_DIR = r"/path/to/your/cell_data/malignant"
BENIGN_CELLS_DIR    = r"/path/to/your/cell_data/benign"

# ── Pipeline hyperparameters ──────────────────
BATCH_SIZE        = 8
NUM_WORKERS       = 0
CELL_SIZE_WEIGHT  = 1.0
FEATURE_LAYER     = "penultimate"
PCA_DIM           = 32
REFERENCE_K       = 10
SAVE_CLUSTER_IMGS = True

# Source label → category name mapping
CATEGORY_NAMES = {
    0: "Candidate",
    1: "LUAD",
    2: "LUSC",
    3: "SCLC",
    4: "Benign",
    5: "Malignant",
}


# ─────────────────────────────────────────────
def find_best_model_for_top4(models_folder: str, evaluation_folder: str) -> dict:
    """Load top-4 model summary and find the best fold (highest test_mcc) for each model."""
    top4_path = os.path.join(evaluation_folder, "top4_models_summary.csv")
    if not os.path.exists(top4_path):
        raise FileNotFoundError(f"Top-4 summary not found: {top4_path}")

    top4_models = pd.read_csv(top4_path)["model"].tolist()[:4]
    print(f"Top-4 models: {top4_models}")

    best_model_paths = {}
    for model_name in top4_models:
        detail_path = os.path.join(evaluation_folder, model_name, f"{model_name}_detailed_results.csv")
        if not os.path.exists(detail_path):
            print(f"Warning: detailed results not found for {model_name}, skipped")
            continue

        df = pd.read_csv(detail_path)
        best_idx  = df["test_mcc"].idxmax()
        best_fold = df.loc[best_idx, "fold"]
        model_path = os.path.join(models_folder, model_name, f"{model_name.lower()}_fold_{best_fold}.pth")

        if os.path.exists(model_path):
            best_model_paths[model_name] = {
                "path": model_path,
                "fold": best_fold,
                "mcc":  df.loc[best_idx, "test_mcc"],
            }
            print(f"  {model_name}: fold={best_fold}, MCC={df.loc[best_idx, 'test_mcc']:.4f}")
        else:
            print(f"Warning: model weight not found: {model_path}")

    return best_model_paths


def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_norm_stats(log_path: str):
    """Try to read mean/std from a training log file; fall back to ImageNet defaults."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if not os.path.exists(log_path):
        return mean, std
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if ("均值:" in line or "Mean:" in line) and "[" in line:
                    mean = [float(x) for x in line[line.find("[")+1:line.find("]")].split(",")]
                elif ("标准差:" in line or "Std:" in line) and "[" in line:
                    std  = [float(x) for x in line[line.find("[")+1:line.find("]")].split(",")]
    except Exception:
        pass
    return mean, std


def run_pipeline_for_model(
    model_name: str,
    model_path: str,
    malignant_cells_dir: str,
    benign_cells_dir: str,
    output_base_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    cell_size_weight: float = 1.0,
    feature_layer: str = "penultimate",
    pca_dim: int = 32,
    reference_k: int = 10,
    save_cluster_images: bool = True,
) -> bool:
    """Run the full feature-extraction + clustering pipeline for one model."""
    print(f"Running pipeline for {model_name} ...")
    _set_seed()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean, std = _read_norm_stats(model_path.replace(".pth", ".log"))
        model, transform, _ = get_model_and_transform(model_name, device, model_path, mean=mean, std=std)

        # ── Reference domain ─────────────────────
        print("--- Processing reference cells ---")
        ref_paths, ref_sources = collect_reference_data(malignant_cells_dir, benign_cells_dir)
        ref_sizes = np.log1p([count_non_background_pixels(p) for p in tqdm(ref_paths, desc="Ref sizes")])

        ref_loader = DataLoader(
            CellImageDataset(ref_paths, transform=transform, source_labels=ref_sources),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        ref_feats, ref_sources, ref_paths = extract_features_with_cell_size(
            model, ref_loader, device, ref_sizes, feature_layer, cell_size_weight
        )

        # PCA + KMeans
        pca = PCA(n_components=min(pca_dim, ref_feats.shape[0], ref_feats.shape[1]))
        ref_norm = normalize(pca.fit_transform(ref_feats))
        kmeans   = KMeans(n_clusters=reference_k, random_state=42).fit(ref_norm)

        ref_results = {
            "features":        ref_norm,
            "labels":          kmeans.labels_,
            "source_labels":   ref_sources,
            "image_paths":     ref_paths,
            "cluster_centers": kmeans.cluster_centers_,
            "n_clusters":      reference_k,
            "pca":             pca,
        }

        # ── Visualization ─────────────────────────
        ref_vis_dir = os.path.join(output_base_dir, model_name, f"reference_visualization_pca{pca_dim}")
        os.makedirs(ref_vis_dir, exist_ok=True)

        visualize_all_clusters_without_images(
            ref_results["features"], ref_results["labels"], ref_results["source_labels"], "Reference", ref_vis_dir)
        visualize_all_clusters_without_images_1(
            ref_results["features"], ref_results["labels"], ref_results["source_labels"], "Reference", ref_vis_dir)
        improved_visualize_clusters_with_images(
            ref_results["features"], ref_results["labels"], ref_results["source_labels"],
            ref_results["image_paths"], "Reference", ref_vis_dir, ref_sizes)
        analyze_cluster_composition(
            ref_results["labels"], ref_results["source_labels"],
            ref_results["image_paths"], "Reference", ref_vis_dir)

        # ── Save cluster images (optional) ────────
        if save_cluster_images:
            clusters_dir = os.path.join(ref_vis_dir, "cluster_images")
            for cluster in sorted(set(ref_results["labels"])):
                for cat in CATEGORY_NAMES.values():
                    os.makedirs(os.path.join(clusters_dir, f"cluster_{cluster}", cat), exist_ok=True)

            print("Copying images to cluster folders ...")
            for img_path, label, source in tqdm(
                zip(ref_results["image_paths"], ref_results["labels"], ref_results["source_labels"]),
                total=len(ref_results["image_paths"]), desc="Copying",
            ):
                dst_dir = os.path.join(clusters_dir, f"cluster_{label}", CATEGORY_NAMES[source])
                shutil.copy2(img_path, os.path.join(dst_dir, Path(img_path).name))

            print(f"Cluster images saved to: {clusters_dir}")

        return True

    except Exception as e:
        print(f"Pipeline failed for {model_name}: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────
def main():
    print("Top-4 Model Clustering Pipeline")
    print("=" * 50)

    # Validate input paths
    for path, name in [
        (MODELS_FOLDER,       "models_folder"),
        (EVALUATION_FOLDER,   "evaluation_folder"),
        (MALIGNANT_CELLS_DIR, "malignant_cells_dir"),
        (BENIGN_CELLS_DIR,    "benign_cells_dir"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Step 1: Find best fold for each top-4 model
    print("\n[Step 1] Finding best fold for each top-4 model ...")
    best_model_paths = find_best_model_for_top4(MODELS_FOLDER, EVALUATION_FOLDER)
    if not best_model_paths:
        print("No valid model weights found. Exiting.")
        return
    print(f"Found {len(best_model_paths)} valid model(s).\n")

    # Step 2: Run pipeline for each model
    print("[Step 2] Running pipeline for each model ...")
    successful, failed = [], []

    for model_name, info in best_model_paths.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | fold: {info['fold']} | MCC: {info['mcc']:.4f}")
        print(f"Weight: {info['path']}")
        print("=" * 60)

        ok = run_pipeline_for_model(
            model_name=model_name,
            model_path=info["path"],
            malignant_cells_dir=MALIGNANT_CELLS_DIR,
            benign_cells_dir=BENIGN_CELLS_DIR,
            output_base_dir=OUTPUT_BASE_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            cell_size_weight=CELL_SIZE_WEIGHT,
            feature_layer=FEATURE_LAYER,
            pca_dim=PCA_DIM,
            reference_k=REFERENCE_K,
            save_cluster_images=SAVE_CLUSTER_IMGS,
        )
        (successful if ok else failed).append(model_name)
        print(f"{'✓' if ok else '✗'} {model_name} {'succeeded' if ok else 'failed'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(best_model_paths)} model(s) processed | "
          f"{len(successful)} succeeded | {len(failed)} failed")
    if successful:
        print("Succeeded:", ", ".join(successful))
    if failed:
        print("Failed:   ", ", ".join(failed))
    print("=" * 60)


if __name__ == "__main__":
    print(f"Start time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()