# clustering/K_optimizer.py


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import shutil



def plot_k_optimization_curve(k_values, scores, best_k, patient_id, save_dir):
    """Plot clustering-quality metrics across candidate K values."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, 'o-', linewidth=2, markersize=8)
    

    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    
    plt.xlabel('k (Number of Clusters)', fontsize=24, fontweight='bold', labelpad = 20)
    plt.ylabel('Size-weighted Score(K)', fontsize=24, fontweight='bold', labelpad = 20)
    plt.title(f'{patient_id} K-Optimization', fontsize=28, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16)
    plt.margins(0.1)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    save_path = os.path.join(save_dir, 'k_optimization_plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def optimize_k_for_candidate_clustering(
    patient_candidate_normalized, 
    patient_candidate_source_labels, 
    patient_candidate_image_paths,
    reference_results,
    patient_save_dir,
    patient_id,
    max_k=20,
    malignant_ref_clusters=None
):
    """Select a candidate-clustering K value and return its results."""
    print(
        f"  Candidate cell count: {len(patient_candidate_normalized)}; "
        "starting K optimization..."
    )
    

    k_optimization_dir = os.path.join(patient_save_dir, 'k_optimization')
    os.makedirs(k_optimization_dir, exist_ok=True)
    

    k_scores = []
    

    max_k = min(max_k, len(patient_candidate_normalized) // 3)
    # Equation (7): evaluate every integer K in [2, Kmax], not only even K.
    k_values = range(2, max_k + 1)



    benign_ref_clusters = [i for i in range(reference_results['n_clusters']) if i not in malignant_ref_clusters]
    

    reference_centers = reference_results['cluster_centers']
    cell_similarity_matrix = cosine_similarity(patient_candidate_normalized, reference_centers)
    

    cell_nearest_ref_cluster = np.argmin(1 - cell_similarity_matrix, axis=1)
    
    for k in k_values:
        print(f"  Trying k={k}")
        
        try:

            k_save_dir = os.path.join(k_optimization_dir, f'k{k}')
            os.makedirs(k_save_dir, exist_ok=True)


            patient_candidate_kmeans = KMeans(
                n_clusters=k, random_state=42, n_init=20, max_iter=300
            )
            patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)
            

            cluster_counts = np.bincount(patient_candidate_labels, minlength=k)
            if np.any(cluster_counts == 0):
                print(f"  Warning: k={k} produced an empty cluster; skipping it")
                continue
            

            patient_candidate_distances = np.linalg.norm(
                patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels], 
                axis=1
            )
            

            current_candidate_results = {
                'features': patient_candidate_normalized,
                'labels': patient_candidate_labels,
                'distances': patient_candidate_distances,
                'source_labels': patient_candidate_source_labels,
                'image_paths': patient_candidate_image_paths,
                'kmeans': patient_candidate_kmeans,
                'cluster_centers': patient_candidate_kmeans.cluster_centers_,
                'n_clusters': k
            }
            

            candidate_centers = current_candidate_results['cluster_centers']
            similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
            

            distance_matrix = 1 - similarity_matrix
            

            cluster_nearest_ref = np.argmin(distance_matrix, axis=1)
            

            cluster_scores = []
            
            for cand_idx in range(k):

                cluster_mask = (patient_candidate_labels == cand_idx)
                cluster_size = np.sum(cluster_mask)
                

                malignant_distances = [distance_matrix[cand_idx, ref_idx] for ref_idx in malignant_ref_clusters]
                Dm = min(malignant_distances)
                

                benign_distances = [distance_matrix[cand_idx, ref_idx] for ref_idx in benign_ref_clusters]
                Db = min(benign_distances) if benign_distances else 1.0
                

                epsilon = 1e-10
                Ssep = abs((Dm - Db) / (Dm + Db + epsilon))
                

                cluster_cells_mask = (patient_candidate_labels == cand_idx)
                cluster_cells_nearest_ref = cell_nearest_ref_cluster[cluster_cells_mask]
                

                cluster_nearest_ref_cluster = cluster_nearest_ref[cand_idx]
                

                matching_cells_count = np.sum(cluster_cells_nearest_ref == cluster_nearest_ref_cluster)
                

                # Equation (10-4): P(i) is a within-cluster consistency.
                # Its denominator is N(i), not all candidate cells.
                P_i = matching_cells_count / cluster_size
                

                cluster_scores.append({
                    'separation': Ssep,
                    'consistency': P_i,
                    'cluster_size': int(cluster_size),
                    'matching_cells_count': int(matching_cells_count),
                })
            
            # Use a size-normalised K score so values from different K are on
            # the same [0, 1] scale.  P(i) remains Eq. (10-4), while N(i)/N
            # makes each cluster's contribution proportional to its cells:
            # Score(K) = sum_i [N(i)/N] * Ssep(i) * P(i).
            total_cells = len(patient_candidate_normalized)
            score_k = sum(
                (item['cluster_size'] / total_cells)
                * item['separation'] * item['consistency']
                for item in cluster_scores
            )
            
            print(f"  k={k} size-weighted Score(K): {score_k:.4f}")
            k_scores.append((k, score_k))
            

            np.save(os.path.join(k_save_dir, 'candidate_cluster_centers.npy'), candidate_centers)
            np.save(os.path.join(k_save_dir, 'candidate_labels.npy'), patient_candidate_labels)
            np.save(
                os.path.join(k_save_dir, 'cluster_scores.npy'),
                np.array([
                    [
                        item['separation'], item['consistency'],
                        item['cluster_size'], item['matching_cells_count'],
                        item['cluster_size'] / total_cells,
                    ]
                    for item in cluster_scores
                ]),
            )
                
        except Exception as e:
            print(f"  Error while trying k={k}: {e}; skipping it")
            continue
    

    if not k_scores:
        print("  Warning: all candidate K values failed; falling back to k=2")

        try:
            k = 2
            k_save_dir = os.path.join(k_optimization_dir, f'k{k}')
            os.makedirs(k_save_dir, exist_ok=True)
            
            patient_candidate_kmeans = KMeans(
                n_clusters=k, random_state=42, n_init=20, max_iter=300
            )
            patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)
            
            patient_candidate_distances = np.linalg.norm(
                patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels], 
                axis=1
            )
            
            best_candidate_results = {
                'features': patient_candidate_normalized,
                'labels': patient_candidate_labels,
                'distances': patient_candidate_distances,
                'source_labels': patient_candidate_source_labels,
                'image_paths': patient_candidate_image_paths,
                'kmeans': patient_candidate_kmeans,
                'cluster_centers': patient_candidate_kmeans.cluster_centers_,
                'n_clusters': k
            }
            
            best_k = k
            print(f"  Clustering completed with fallback k={k}")
            
        except Exception as e:
            print(f"  Fallback k=2 also failed: {e}; skipping this slide")
            return None, None
    else:

        k_scores.sort(key=lambda x: x[1], reverse=True)
        best_k, best_score = k_scores[0]
        

        k_df = pd.DataFrame(k_scores, columns=['k', 'size_weighted_score_k'])
        k_df.to_csv(os.path.join(k_optimization_dir, 'k_optimization_results.csv'), index=False)
        

        k_vals = [x[0] for x in k_scores]
        scores = [x[1] for x in k_scores]

        

        plot_k_optimization_curve(k_vals, scores, best_k, patient_id, k_optimization_dir)
        

        patient_candidate_kmeans = KMeans(
            n_clusters=best_k, random_state=42, n_init=20, max_iter=300
        )
        patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)
        
        patient_candidate_distances = np.linalg.norm(
            patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels], 
            axis=1
        )
        
        best_candidate_results = {
            'features': patient_candidate_normalized,
            'labels': patient_candidate_labels,
            'distances': patient_candidate_distances,
            'source_labels': patient_candidate_source_labels,
            'image_paths': patient_candidate_image_paths,
            'kmeans': patient_candidate_kmeans,
            'cluster_centers': patient_candidate_kmeans.cluster_centers_,
            'n_clusters': best_k
        }
        
        print(
            f"  Optimization complete, best K={best_k}, "
            f"best size-weighted Score(K)={best_score:.4f}"
        )
    
    return best_candidate_results, best_k
