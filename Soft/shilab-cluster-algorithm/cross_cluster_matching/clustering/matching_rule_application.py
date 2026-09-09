# clustering/matching_rule_application.py


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import shutil

def check_matching_rules(similarities, ref_idx_target, ref_idx_avoid, threshold_target, threshold_avoid_others=None, threshold_ratio=None):
    """Return whether similarities satisfy one matching rule."""
    # Keep the established rule behaviour: compare two-decimal cosine
    # similarities, so configured thresholds use the same displayed precision.
    similarities = [round(float(s), 2) for s in similarities]
    if similarities[ref_idx_target] < threshold_target:
        return False
        

    avoid_indices = ref_idx_avoid if isinstance(ref_idx_avoid, list) else [ref_idx_avoid]
    

    other_indices = [i for i in range(len(similarities)) if i != ref_idx_target and i not in avoid_indices]
    
    if threshold_ratio is not None:
        limit = similarities[ref_idx_target] / threshold_ratio
        if any(similarities[i] >= limit for i in other_indices):
            return False
            
    if threshold_avoid_others is not None:
        if any(similarities[i] >= threshold_avoid_others for i in other_indices):
            return False
            
    return True


# def identify_matching_clusters(candidate_centers, reference_centers):

#     similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
#     rule1_clusters = []
#     rule2_clusters = []
    
#     for i, sims in enumerate(similarity_matrix):

#         if check_matching_rules(sims, 5, 0, 0.5, threshold_ratio=3):
#             rule1_clusters.append(i)

#         elif check_matching_rules(sims, 0, 5, 0.4, threshold_avoid_others=0.2):
#             rule2_clusters.append(i)
        
            
#     return rule1_clusters, rule2_clusters, similarity_matrix

def identify_matching_clusters(candidate_centers, reference_centers, rules):
    """
    Identify candidate clusters that satisfy ordered matching rules.

    Parameters
    ----------
    candidate_centers : array-like, shape (n_candidates, n_features)
    reference_centers : array-like, shape (n_references, n_features)
    rules : list of dict
        Each dictionary defines one rule with the following fields:
        {
            "target"               : int,               # target reference-cluster index
            "avoid"                : int | list[int],   # reference-cluster index or indices to avoid
            "threshold_target"     : float,             # target similarity threshold
            "threshold_avoid_others": float | None,     # absolute threshold for other clusters
            "threshold_ratio"      : float | None,      # relative threshold: target / ratio
        }

    Returns
    -------
    rule_clusters : list of list[int]
        One list per rule, containing the matching candidate-cluster indices.
        Rules are evaluated in order; a cluster matched by an earlier rule is
        not evaluated against later rules.
    similarity_matrix : ndarray, shape (n_candidates, n_references)

    Examples
    --------
    # Equivalent to the former two-rule matching configuration.
    rules = [
        {"target": 5, "avoid": 0, "threshold_target": 0.5, "threshold_ratio": 3},
        {"target": 0, "avoid": 5, "threshold_target": 0.4, "threshold_avoid_others": 0.2},
    ]
    rule1_clusters, rule2_clusters, sim = identify_matching_clusters(cand, ref, rules)

    # Equivalent to the former three-rule urine configuration.
    rules = [
        {"target": 0, "avoid": [4, 9], "threshold_target": 0.5, "threshold_ratio": 3},
        {"target": 4, "avoid": [0, 9], "threshold_target": 0.5, "threshold_ratio": 3},
        {"target": 9, "avoid": [0, 4], "threshold_target": 0.5, "threshold_ratio": 3},
    ]
    rule1, rule2, rule3, sim = identify_matching_clusters(cand, ref, rules)
    """
    similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
    rule_clusters = [[] for _ in rules]

    for i, sims in enumerate(similarity_matrix):
        for rule_idx, rule in enumerate(rules):
            matched = check_matching_rules(
                sims,
                ref_idx_target=rule["target"],
                ref_idx_avoid=rule["avoid"],
                threshold_target=rule["threshold_target"],
                threshold_avoid_others=rule.get("threshold_avoid_others"),
                threshold_ratio=rule.get("threshold_ratio"),
            )
            if matched:
                rule_clusters[rule_idx].append(i)
                break

    return (*rule_clusters, similarity_matrix)

def identify_matching_clusters_urine(candidate_centers, reference_centers):
    """Apply the legacy three-rule urine matching configuration."""
    similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
    rule1_clusters = []
    rule2_clusters = []
    rule3_clusters = []
    
    for i, sims in enumerate(similarity_matrix):

        if check_matching_rules(sims, 0, [4, 9], 0.5, threshold_ratio=3):
            rule1_clusters.append(i)

        elif check_matching_rules(sims, 4, [0, 9], 0.5, threshold_ratio=3):
            rule2_clusters.append(i)
        elif check_matching_rules(sims, 9, [0, 4], 0.5, threshold_ratio=3):
            rule3_clusters.append(i)
            
    return rule1_clusters, rule2_clusters, rule3_clusters, similarity_matrix


def identify_matching_clusters_urine_v2(candidate_centers, reference_centers):
    """Apply the legacy two-rule urine matching configuration."""
    similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
    rule1_clusters = []
    rule2_clusters = []
    
    for i, sims in enumerate(similarity_matrix):

        if check_matching_rules(sims, 7, 0, 0.5, threshold_ratio=3):
            rule1_clusters.append(i)

        elif check_matching_rules(sims, 0, 7, 0.5, threshold_avoid_others=3):
            rule2_clusters.append(i)
        
            
    return rule1_clusters, rule2_clusters, similarity_matrix

# def identify_matching_clusters_lyy(candidate_centers, reference_centers):

#     similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
#     rule1_clusters = []
#     rule2_clusters = []
    
#     for i, sims in enumerate(similarity_matrix):

#         if check_matching_rules(sims, 0, 2, 0.5, threshold_ratio=3):
#             rule1_clusters.append(i)

#         elif check_matching_rules(sims, 2, 0, 0.5, threshold_avoid_others=3):
#             rule2_clusters.append(i)
        
            
#     return rule1_clusters, rule2_clusters, similarity_matrix


def save_matched_cells(
    indices,
    image_paths,
    labels,
    similarity_matrix,
    rule_type,
    save_dirs,
    target_ref_indices,
    rule=None,
    save_rejected=True,
    rejected_dir=None,
    rejected_l0_dir=None,
    save_rule_copies=True,
):
    """Copy cells that pass matching rules and return their source indices."""
    count = 0
    processed_indices = []

    for idx in indices:
        cell_sim_vec = similarity_matrix[idx]


        most_similar_ref = int(np.argmax(cell_sim_vec))
        if most_similar_ref not in target_ref_indices:

            if save_rejected and rejected_l0_dir is not None:
                _save_rejected_cell(
                    image_paths[idx], labels[idx], most_similar_ref,
                    cell_sim_vec[most_similar_ref],
                    rule_type, rejected_l0_dir, layer="L0"
                )
            continue


        mal_sims     = {i: cell_sim_vec[i] for i in target_ref_indices}
        best_ref_idx = max(mal_sims, key=mal_sims.get)
        best_sim     = mal_sims[best_ref_idx]


        if rule is not None:
            passed = check_matching_rules(
                cell_sim_vec,
                ref_idx_target         = rule["target"],
                ref_idx_avoid          = rule["avoid"],
                threshold_target       = rule["threshold_target"],
                threshold_avoid_others = rule.get("threshold_avoid_others"),
                threshold_ratio        = rule.get("threshold_ratio"),
            )
            if not passed:

                if save_rejected and rejected_dir is not None:
                    _save_rejected_cell(
                        image_paths[idx], labels[idx], best_ref_idx,
                        best_sim,
                        rule_type, rejected_dir, layer="L1"
                    )
                continue


        src_path      = image_paths[idx]
        filename      = os.path.basename(src_path)
        cluster_label = labels[idx]

        prefix   = f"{rule_type}_cluster{cluster_label}_ref{best_ref_idx}_sim{best_sim:.4f}"
        new_name = f"{prefix}_{filename}"

        if save_rule_copies:
            shutil.copy(src_path, os.path.join(save_dirs['rule_dir'], new_name))
        shutil.copy(src_path, os.path.join(save_dirs['total_dir'], new_name))

        count += 1
        processed_indices.append(idx)

    return count, processed_indices


def _save_rejected_cell(src_path, cluster_label, ref_idx, sim_score,
                        rule_type, rejected_dir, layer="L1"):
    """Copy one rejected cell with rule and similarity metadata in its filename."""
    filename = os.path.basename(src_path)
    prefix = f"rejected_{layer}_{rule_type}_cluster{cluster_label}_ref{ref_idx}_sim{sim_score:.4f}"
    new_name = f"{prefix}_{filename}"
    os.makedirs(rejected_dir, exist_ok=True)
    shutil.copy(src_path, os.path.join(rejected_dir, new_name))
