# pipeline.py

import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import json
import hashlib
from pathlib import Path
import joblib
import umap
from tqdm import tqdm

from cross_cluster_matching.models.models import get_model_and_transform
from cross_cluster_matching.data.data_utils import collect_reference_data, collect_image_paths, CellImageDataset, extract_patient_id
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.clustering.K_optimizer import optimize_k_for_candidate_clustering
from cross_cluster_matching.clustering.matching_rule_application import identify_matching_clusters, save_matched_cells, check_matching_rules
from cross_cluster_matching.visualization.visualization import (
    count_non_background_pixels,
    visualize_all_clusters_without_images,
    visualize_all_clusters_without_images_1,
    improved_visualize_clusters_with_images,
    visualize_matching_clusters,
    analyze_cluster_composition,
    plot_similarity_heatmap,
    visualize_filtered_clusters
)


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _sha256_file(path):
    """Return the SHA256 recorded in reference_cache metadata."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_cache_value(name, actual, expected):
    if actual != expected:
        raise ValueError(
            f"reference_cache parameter mismatch: {name}, cache={actual!r}, current config={expected!r}"
        )


def _assert_cache_float_list(name, actual, expected):
    if expected is None:
        return
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array, expected_array, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"reference_cache parameter mismatch: {name}, "
            f"cache={actual_array.tolist()}, current config={expected_array.tolist()}"
        )


def load_reference_cache(
    reference_cache_dir, model_path, model_type, feature_layer,
    mean, std, reference_k, pca_dim, cell_size_weight, malignant_ref_clusters,
    reference_umap_mode="runtime_refit",
):
    """Load a fixed Top4 reference cache without touching reference images.

    ``runtime_refit`` keeps all inference-critical cache artefacts fixed, then
    fits the UMAP visualisation reducer in the current Python process.  A
    serialised UMAP reducer contains Numba internals and is not portable across
    otherwise similar local and server environments.
    """
    cache_dir = Path(reference_cache_dir)
    required_files = {
        "metadata": cache_dir / "metadata.json",
        "pca":      cache_dir / "pca.joblib",
        "kmeans":   cache_dir / "kmeans.joblib",
        "features": cache_dir / "reference_features.npz",
        "manifest": cache_dir / "reference_manifest.csv",
    }
    if reference_umap_mode == "cached":
        required_files["umap"] = cache_dir / "reference_umap_by_cluster.joblib"
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"reference_cache is incomplete: {cache_dir}, missing {', '.join(missing)}."
        )

    with required_files["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported reference_cache schema_version: {metadata.get('schema_version')!r}")

    cache_model      = metadata.get("model", {})
    cache_preprocess = metadata.get("preprocessing", {})
    cache_parameters = metadata.get("reference_parameters", {})
    _assert_cache_value("model_type",    cache_model.get("model_type"),    model_type)
    _assert_cache_value("feature_layer", cache_model.get("feature_layer"), feature_layer)
    _assert_cache_float_list("mean", cache_preprocess.get("mean"), mean)
    _assert_cache_float_list("std",  cache_preprocess.get("std"),  std)
    _assert_cache_value("reference_k",      int(cache_parameters.get("reference_k")),      int(reference_k))
    _assert_cache_value("effective_pca_dim", int(cache_parameters.get("effective_pca_dim")), int(pca_dim))
    _assert_cache_value(
        "cell_size_weight", float(cache_parameters.get("cell_size_weight")), float(cell_size_weight)
    )
    _assert_cache_value(
        "malignant_ref_clusters", list(metadata.get("malignant_ref_clusters", [])),
        [int(cluster) for cluster in malignant_ref_clusters],
    )

    current_model_hash = _sha256_file(model_path)
    _assert_cache_value("model_sha256", cache_model.get("sha256"), current_model_hash)

    pca      = joblib.load(required_files["pca"])
    kmeans   = joblib.load(required_files["kmeans"])
    arrays   = np.load(required_files["features"], allow_pickle=False)
    manifest = pd.read_csv(required_files["manifest"])
    features      = np.asarray(arrays["normalized_features"])
    labels        = np.asarray(arrays["cluster_labels"],  dtype=int)
    source_labels = np.asarray(arrays["source_labels"],   dtype=int)
    centers       = np.asarray(arrays["cluster_centers"])
    n_reference   = len(features)
    if not (len(labels) == len(source_labels) == len(manifest) == n_reference):
        raise ValueError("Internal array length mismatch in reference_cache; aborting.")
    if centers.shape[0] != int(reference_k) or kmeans.n_clusters != int(reference_k):
        raise ValueError("Cluster centre count in reference_cache does not match reference_k.")
    if not np.allclose(centers, kmeans.cluster_centers_, rtol=0.0, atol=1e-12):
        raise ValueError("cluster_centers in reference_cache is inconsistent with kmeans.joblib.")

    if reference_umap_mode == "runtime_refit":
        # Only the two-dimensional visualisation is re-fitted. The input is
        # the cache's frozen article-version reference features, so this does
        # not alter PCA, KMeans, reference labels/centres or matching results.
        print("[INFO] Re-fitting UMAP in the current process for visualisation only.")
        reference_umap = umap.UMAP(random_state=42).fit(features)
    elif reference_umap_mode == "cached":
        # Legacy option. Cross-environment deserialisation can fail only when
        # transform() first invokes Numba, even if joblib.load() succeeds.
        print("[WARN] Using serialised UMAP from cache; this is not portable across environments.")
        reference_umap = joblib.load(required_files["umap"])
    else:
        raise ValueError(
            f"Unknown reference_umap_mode={reference_umap_mode!r}; "
            "expected 'runtime_refit' or 'cached'."
        )

    image_paths = manifest["relative_path"].astype(str).tolist()
    ref_results = {
        "features":        features,
        "labels":          labels,
        "source_labels":   source_labels,
        "image_paths":     image_paths,
        "cluster_centers": centers,
        "n_clusters":      int(reference_k),
        "pca":             pca,
        "kmeans":          kmeans,
    }
    print(f"[INFO] reference_cache loaded and verified: {cache_dir}")
    print(f"[INFO] reference images={n_reference}, malignant ref clusters={metadata['malignant_ref_clusters']}")
    return ref_results, reference_umap


def run_pipeline(
    positive_folder_path, negative_folder_path, malignant_cells_dir=None, benign_cells_dir=None,
    model_path=None, base_save_dir=None,
    feature_layer='penultimate', reference_k=10, max_candidate_k=20,
    batch_size=16, num_workers=0, cell_size_weight=1.0, model_type='ResNeXt', pca_dim=32, mean=None, std=None,
    malignant_ref_clusters=None, matching_rules=None,
    use_cell_level_rule_filter=False, reference_cache_dir=None,
    patient_id=None, stats_path=None, reference_umap_mode="runtime_refit",
    save_rule_diagnostics=False,
):
    """Run cross-domain cluster matching for candidate cell images."""

    if malignant_ref_clusters is None:
        raise ValueError("malignant_ref_clusters cannot be None")
    if model_path is None or base_save_dir is None:
        raise ValueError("model_path and base_save_dir are required")
    if reference_cache_dir is None and (not malignant_cells_dir or not benign_cells_dir):
        raise ValueError("malignant_cells_dir and benign_cells_dir are required when reference_cache_dir is not provided.")

    n_rules = len(malignant_ref_clusters)

    if matching_rules is None:
        matching_rules = [
            {
                "target":           malignant_ref_clusters[i],
                "avoid":            [c for j, c in enumerate(malignant_ref_clusters) if j != i],
                "threshold_target": 0.5,
                "threshold_ratio":  3,
            }
            for i in range(n_rules)
        ]

    assert len(matching_rules) == n_rules, \
        f"len(matching_rules)={len(matching_rules)} must equal len(malignant_ref_clusters)={n_rules}"

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, _ = get_model_and_transform(model_type, device, model_path, mean=mean, std=std)

    # Reference domain: load reviewed Top4 cache when supplied; otherwise
    # keep the original re-clustering path for legacy configurations.
    if reference_cache_dir:
        ref_results, reference_umap = load_reference_cache(
            reference_cache_dir, model_path, model_type, feature_layer,
            mean, std, reference_k, pca_dim, cell_size_weight, malignant_ref_clusters,
            reference_umap_mode=reference_umap_mode,
        )
        pca = ref_results['pca']
    else:
        print("======= Processing reference cells =======")
        ref_paths, ref_sources = collect_reference_data(malignant_cells_dir, benign_cells_dir)
        ref_sizes = np.log1p([count_non_background_pixels(p) for p in tqdm(ref_paths, desc="Ref Sizes")])
        ref_loader = DataLoader(
            CellImageDataset(ref_paths, transform=transform, source_labels=ref_sources),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        ref_feats, ref_sources, ref_paths = extract_features_with_cell_size(
            model, ref_loader, device, ref_sizes, feature_layer, cell_size_weight
        )
        pca = PCA(n_components=min(pca_dim, ref_feats.shape[0], ref_feats.shape[1]))
        ref_norm = normalize(pca.fit_transform(ref_feats))
        kmeans = KMeans(n_clusters=reference_k, random_state=42).fit(ref_norm)
        ref_results = {
            'features':        ref_norm,
            'labels':          kmeans.labels_,
            'source_labels':   ref_sources,
            'image_paths':     ref_paths,
            'cluster_centers': kmeans.cluster_centers_,
            'n_clusters':      reference_k,
            'pca':             pca,
        }
        ref_vis_dir = os.path.join(base_save_dir, f'reference_visualization_pca{pca_dim}')
        os.makedirs(ref_vis_dir, exist_ok=True)
        reference_umap = visualize_all_clusters_without_images(
            ref_results['features'], ref_results['labels'], ref_results['source_labels'], "Reference", ref_vis_dir
        )
        visualize_all_clusters_without_images_1(
            ref_results['features'], ref_results['labels'], ref_results['source_labels'], "Reference", ref_vis_dir
        )
        improved_visualize_clusters_with_images(
            ref_results['features'], ref_results['labels'], ref_results['source_labels'],
            ref_results['image_paths'], "Reference", ref_vis_dir, ref_sizes
        )
        analyze_cluster_composition(
            ref_results['labels'], ref_results['source_labels'], ref_results['image_paths'], "Reference", ref_vis_dir
        )

    patient_folders = []
    patient_folders.extend([(entry.path, 'positive') for entry in os.scandir(positive_folder_path) if entry.is_dir()])
    patient_folders.extend([(entry.path, 'negative') for entry in os.scandir(negative_folder_path) if entry.is_dir()])

    seen_pids = set()
    unique_patient_folders = []
    for path, status in patient_folders:
        pid = extract_patient_id(path)
        if pid not in seen_pids:
            seen_pids.add(pid)
            unique_patient_folders.append((path, status))
        else:
            print(f"[INFO] Skipping duplicate sample: {pid} (path: {path}, label: {status})")
    patient_folders = unique_patient_folders

    if patient_id:
        patient_folders = [
            (path, status) for path, status in patient_folders
            if extract_patient_id(path) == patient_id
        ]
        if not patient_folders:
            raise ValueError(
                f"patient_id={patient_id!r} not found in candidate domain input directories."
            )

    results_stats = []

    for folder, status in patient_folders:
        pid     = extract_patient_id(folder)
        save_dir = os.path.join(base_save_dir, pid)
        os.makedirs(save_dir, exist_ok=True)

        cand_paths = collect_image_paths(os.path.join(folder, "malignant_images"))
        if not cand_paths:
            continue

        cand_sources = [0] * len(cand_paths)
        cand_sizes   = np.log1p([count_non_background_pixels(p) for p in cand_paths])

        cand_loader = DataLoader(
            CellImageDataset(cand_paths, transform=transform, source_labels=cand_sources),
            batch_size=batch_size, shuffle=False
        )
        cand_feats, _, cand_paths = extract_features_with_cell_size(
            model, cand_loader, device, cand_sizes, feature_layer, cell_size_weight
        )

        cand_norm = normalize(pca.transform(cand_feats))

        matched_counts = [0] * n_rules

        dirs = {f'rule{i+1}': os.path.join(save_dir, f'rule{i+1}_matched_cells') for i in range(n_rules)}
        dirs['total'] = os.path.join(save_dir, 'matched_malignant_cells')
        output_dirs = [dirs['total']]
        if save_rule_diagnostics:
            output_dirs.extend(dirs[f'rule{i+1}'] for i in range(n_rules))
        for d in output_dirs:
            os.makedirs(d, exist_ok=True)

        if len(cand_norm) >= pca_dim:

            cand_results, best_k = optimize_k_for_candidate_clustering(
                cand_norm, cand_sources, cand_paths, ref_results, save_dir, pid, max_candidate_k, malignant_ref_clusters
            )
            if not cand_results:
                continue

            df_cluster = pd.DataFrame({
                'labels':      cand_results['labels'],
                'image_paths': cand_results['image_paths'],
            })
            df_cluster.to_csv(os.path.join(save_dir, 'cluster_detail.csv'), index=False, encoding='utf-8-sig')

            *rule_clusters_list, similarity_matrix = identify_matching_clusters(
                cand_results['cluster_centers'], ref_results['cluster_centers'], matching_rules
            )

            cell_sims = cosine_similarity(cand_results['features'], ref_results['cluster_centers'])

            passed_indices_dict = {}

            for i, rule_clusters in enumerate(rule_clusters_list):
                rule_name = f'rule{i+1}'
                indices   = [j for j, lbl in enumerate(cand_results['labels']) if lbl in rule_clusters]

                rejected_dir = None
                if save_rule_diagnostics:
                    rejected_dir = os.path.join(save_dir, f'rule{i+1}_rejected_cells')
                    os.makedirs(rejected_dir, exist_ok=True)

                cell_filter_rule = matching_rules[i] if use_cell_level_rule_filter else None

                count, kept_indices = save_matched_cells(
                    indices, cand_paths, cand_results['labels'], cell_sims, rule_name,
                    {'rule_dir': dirs[rule_name], 'total_dir': dirs['total']},
                    malignant_ref_clusters,
                    rule             = cell_filter_rule,
                    save_rejected    = save_rule_diagnostics,
                    rejected_dir     = rejected_dir,
                    save_rule_copies = save_rule_diagnostics,
                )
                matched_counts[i] = count
                passed_indices_dict[rule_name] = kept_indices

            matching_pairs = []
            for rule, rule_clusters in zip(matching_rules, rule_clusters_list):
                target_ref = rule["target"]
                for cand_idx in rule_clusters:
                    matching_pairs.append(
                        (target_ref, cand_idx, similarity_matrix[cand_idx, target_ref])
                    )

            rule_matched_clusters = {
                f'rule{i+1}': clusters for i, clusters in enumerate(rule_clusters_list)
            }

            visualize_matching_clusters(
                ref_results, cand_results, matching_pairs,
                save_dir, reference_umap, rule_matched_clusters, output_format="png"
            )
            visualize_all_clusters_without_images(
                cand_results['features'], cand_results['labels'], cand_results['source_labels'],
                pid, save_dir, reference_umap, output_format="png"
            )
            improved_visualize_clusters_with_images(
                cand_results['features'], cand_results['labels'], cand_results['source_labels'],
                cand_paths, pid, save_dir, cand_sizes, output_format="png"
            )
            visualize_filtered_clusters(
                ref_results, cand_results, matching_pairs, save_dir,
                rule_matched_clusters = rule_matched_clusters,
                target_ref_indices    = malignant_ref_clusters,
                passed_indices_dict   = passed_indices_dict,
            )
            xlabel = [f"C{i}" for i in range(cand_results['n_clusters'])]
            ylabel = [f"R{i}" for i in range(ref_results['n_clusters'])]
            plot_similarity_heatmap(
                similarity_matrix.T, xlabel, ylabel,
                f"{pid} Similarity Matrix",
                os.path.join(save_dir, 'similarity_heatmap.png'), save_tiff=False
            )

        else:

            cell_sims = cosine_similarity(cand_norm, ref_results['cluster_centers'])

            processed = set()
            for i, rule in enumerate(matching_rules):
                rule_name = f'rule{i+1}'
                indices = [
                    j for j, s in enumerate(cell_sims)
                    if j not in processed and check_matching_rules(
                        s,
                        ref_idx_target         = rule["target"],
                        ref_idx_avoid          = rule["avoid"],
                        threshold_target       = rule["threshold_target"],
                        threshold_avoid_others = rule.get("threshold_avoid_others"),
                        threshold_ratio        = rule.get("threshold_ratio"),
                    )
                ]
                count, newly_processed = save_matched_cells(
                    indices, cand_paths, [0] * len(cand_paths), cell_sims, rule_name,
                    {'rule_dir': dirs[rule_name], 'total_dir': dirs['total']},
                    malignant_ref_clusters,
                    save_rule_copies=save_rule_diagnostics,
                )
                matched_counts[i] = count
                processed.update(newly_processed)

            xlabel = [f"Cell{i}" for i in range(len(cell_sims))]
            ylabel = [f"R{i}" for i in range(ref_results['n_clusters'])]
            plot_similarity_heatmap(
                cell_sims.T, xlabel, ylabel,
                f"Patient {pid} Cell Similarity Matrix",
                os.path.join(save_dir, 'Cell_similarity_heatmap.png'), save_tiff=False
            )

        stat = {
            'Patient_ID': pid,
            'Status':     status,
            **{f'Rule{i+1}': matched_counts[i] for i in range(n_rules)},
            'Total':      sum(matched_counts),
        }
        results_stats.append(stat)
        print(f"Patient {pid} Done. Matched: {sum(matched_counts)}")

    final_stats_path = stats_path or os.path.join(base_save_dir, 'stats.xlsx')
    stats_parent = os.path.dirname(os.path.abspath(final_stats_path))
    if stats_parent:
        os.makedirs(stats_parent, exist_ok=True)
    pd.DataFrame(results_stats).to_excel(final_stats_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-domain cluster matching for false positive removal")

    parser.add_argument("--positive_folder_path", required=True)
    parser.add_argument("--negative_folder_path", required=True)
    parser.add_argument("--malignant_cells_dir",  default=None)
    parser.add_argument("--benign_cells_dir",     default=None)
    parser.add_argument(
        "--reference_cache_dir", default=None,
        help="Fixed Top4 reference_cache directory; when provided, reference images are not re-read or re-clustered.",
    )
    parser.add_argument(
        "--reference_umap_mode", choices=("runtime_refit", "cached"),
        default="runtime_refit",
        help=(
            "UMAP strategy for a reference cache. runtime_refit (default) fits "
            "UMAP from frozen cache features in this process and affects only "
            "visualisation; cached uses the legacy serialised UMAP reducer."
        ),
    )
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--base_save_dir", required=True)
    parser.add_argument(
        "--patient_id", default=None,
        help="Process only this patient ID; intended for single-SVS or Slurm array mode.",
    )
    parser.add_argument(
        "--stats_path", default=None,
        help="Explicit output path for the statistics Excel file; defaults to base_save_dir/stats.xlsx.",
    )

    parser.add_argument("--model_type",       default="ResNeXt")
    parser.add_argument("--feature_layer",    default="penultimate")
    parser.add_argument("--reference_k",      type=int,   default=10)
    parser.add_argument("--max_candidate_k",  type=int,   default=20)
    parser.add_argument("--pca_dim",          type=int,   default=32)
    parser.add_argument("--cell_size_weight", type=float, default=1.0)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--workers",          type=int,   default=0)

    parser.add_argument("--mean", nargs="+", type=float, default=None)
    parser.add_argument("--std",  nargs="+", type=float, default=None)

    parser.add_argument("--malignant_ref_clusters", required=True)

    parser.add_argument("--rules_str", default="")

    parser.add_argument("--use_cell_level_rule_filter", action="store_true", default=False)
    parser.add_argument("--save_rule_diagnostics", action="store_true", default=False,
                        help="Save rule-level matched/rejected image copies for debugging.")

    parser.add_argument("--cell_type", default="unknown")

    args = parser.parse_args()

    if not args.reference_cache_dir and (not args.malignant_cells_dir or not args.benign_cells_dir):
        parser.error("--malignant_cells_dir and --benign_cells_dir are both required when --reference_cache_dir is not provided.")

    mal_clusters = [int(x.strip()) for x in args.malignant_ref_clusters.split(",")]

    matching_rules = None
    if args.rules_str:
        matching_rules = []
        for rule_str in args.rules_str.split("@"):
            parts = [p.strip() for p in rule_str.split(":")]
            if len(parts) != 5:
                raise ValueError(f"Invalid rule format (expected 5 fields): {rule_str}")
            target, avoid, thr_target, thr_ratio, thr_avoid = parts

            avoid_parsed = [int(x) for x in avoid.split(",")] if "," in avoid else int(avoid)

            matching_rules.append({
                "target":                 int(target),
                "avoid":                  avoid_parsed,
                "threshold_target":       float(thr_target),
                "threshold_ratio":        float(thr_ratio) if thr_ratio != "none" else None,
                "threshold_avoid_others": float(thr_avoid) if thr_avoid != "none" else None,
            })

    print(f"[INFO] cell_type              = {args.cell_type}")
    print(f"[INFO] model_type             = {args.model_type}")
    print(f"[INFO] malignant_ref_clusters = {mal_clusters}")
    print(f"[INFO] reference_umap_mode    = {args.reference_umap_mode}")
    print(f"[INFO] use_cell_level_filter  = {args.use_cell_level_rule_filter}")
    print(f"[INFO] save_rule_diagnostics  = {args.save_rule_diagnostics}")
    print(f"[INFO] matching_rules         = {matching_rules}")

    run_pipeline(
        positive_folder_path       = args.positive_folder_path,
        negative_folder_path       = args.negative_folder_path,
        malignant_cells_dir        = args.malignant_cells_dir,
        benign_cells_dir           = args.benign_cells_dir,
        reference_cache_dir        = args.reference_cache_dir,
        model_path                 = args.model_path,
        base_save_dir              = args.base_save_dir,
        model_type                 = args.model_type,
        feature_layer              = args.feature_layer,
        reference_k                = args.reference_k,
        max_candidate_k            = args.max_candidate_k,
        pca_dim                    = args.pca_dim,
        cell_size_weight           = args.cell_size_weight,
        batch_size                 = args.batch_size,
        num_workers                = args.workers,
        mean                       = args.mean,
        std                        = args.std,
        malignant_ref_clusters     = mal_clusters,
        matching_rules             = matching_rules,
        use_cell_level_rule_filter = args.use_cell_level_rule_filter,
        patient_id                 = args.patient_id,
        stats_path                 = args.stats_path,
        reference_umap_mode        = args.reference_umap_mode,
        save_rule_diagnostics      = args.save_rule_diagnostics,
    )