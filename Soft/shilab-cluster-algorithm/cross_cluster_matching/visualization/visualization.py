# visualization/visualization.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from tqdm import tqdm
import umap
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


category_names = {
    0: 'Candidate',
    1: 'LUAD',
    2: 'LUSC',
    3: 'SCLC',
    4: 'Benign',
    5: 'Malignant'
}

category_colors = {
    0: (40, 120, 181),
    1: (255, 136, 132),
    2: (200, 36, 35),
    3: (255, 165, 0),
    4: (154, 201, 219),
    5: (185, 89, 184)
}

category_markers = {
    0: 's',
    1: 'v',
    2: '<',
    3: '>',
    4: 'o',
    5: '^'
}


def set_plot_font():
    """Configure Matplotlib fonts used by visualization functions."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'bold'


def count_non_background_pixels(image_path):
    """Count non-background pixels in an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        background_mask = np.all(img_array == [225, 225, 225], axis=2)
        non_background_count = np.sum(~background_mask)
        return non_background_count
    except Exception as e:
        print(f"Error counting non-background pixels for {image_path}: {e}")
        return 0


def analyze_cluster_composition(labels, source_labels, image_paths, method_name, save_dir):
    """Summarize cluster composition by source label."""

    color_map = {
        i: f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for i, c in category_colors.items()
    }

    unique_clusters = sorted(set(labels))

    cluster_composition = {}

    results_df = pd.DataFrame({
        'image_path': image_paths,
        'source_label': source_labels,
        'cluster': labels
    })

    results_df['category'] = results_df['source_label'].map(category_names)

    existing_categories = set(results_df['source_label'].unique())

    for cluster in unique_clusters:
        cluster_samples = results_df[results_df['cluster'] == cluster]
        category_counts = cluster_samples['source_label'].value_counts().to_dict()
        all_possible_categories = set(category_names.keys())
        cluster_composition[cluster] = {
            'total': len(cluster_samples),
            'categories': {category_names[i]: category_counts.get(i, 0) for i in all_possible_categories if i in existing_categories}
        }

    summary_data = []
    for cluster, data in cluster_composition.items():
        row = {'Cluster': cluster, 'Total': data['total']}
        row.update(data['categories'])
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    csv_path = os.path.join(save_dir, f"{method_name}_cluster_composition.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Cluster composition summary saved to: {csv_path}")

    set_plot_font()

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])

    clusters = summary_df['Cluster'].tolist()
    bottom = np.zeros(len(clusters))
    legend_elements = []

    for i in sorted(category_names.keys()):
        category = category_names[i]
        if i in existing_categories and category in summary_df.columns:
            values = summary_df[category].values
            ax.bar(clusters, values, bottom=bottom, color=color_map[i])
            bottom += values
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color_map[i], label=category)
            )

    ax.set_xlabel('Cluster', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('Number of Samples', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_title(f'{method_name} - Cluster Composition', fontsize=32, fontweight='bold', pad=20)

    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters, rotation=90 if len(clusters) > 10 else 0)
    ax.set_xlim(-0.6, len(clusters) - 0.4)

    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    legend_ax = fig.add_axes([0.76, 0.5, 0.1, 0.4])
    legend_ax.axis('off')

    legend = legend_ax.legend(
        handles=legend_elements,
        loc='center left',
        title="Categories",
        title_fontsize=18,
        prop={'weight': 'bold', 'size': 18},
        labelspacing=0.8,
        borderpad=1.0,
        handletextpad=0.5,
        frameon=True
    )
    legend_ax.add_artist(legend)

    plt.savefig(os.path.join(save_dir, f"{method_name}_cluster_composition.tif"), dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    plt.close()

    return cluster_composition


def visualize_all_clusters_without_images(features, labels, source_labels, method_name, save_dir,
                                          trained_umap=None, output_format="tif"):
    """Plot candidate clusters without image thumbnails."""
    print(f"Creating global UMAP visualization without thumbnails for {method_name}...")

    if trained_umap is None:
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        embedding = trained_umap.transform(features)

    unique_clusters = sorted(set(labels))
    cluster_colors = cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

    set_plot_font()

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])

    for cluster_idx, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embedding = embedding[cluster_indices]
        cluster_source_labels = [source_labels[i] for i in cluster_indices]

        for category in set(source_labels):
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            if category_indices:
                category_embedding = cluster_embedding[category_indices]
                ax.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=[cluster_colors[cluster_idx]],
                    marker=category_markers.get(category, 'o'),
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )

    cluster_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cluster_colors[i],
                   markersize=16, label=f'Cluster {cluster}')
        for i, cluster in enumerate(unique_clusters)
    ]

    category_legend_elements = [
        plt.Line2D([0], [0], marker=marker, color='black',
                   markersize=16, label=category_names[cat])
        for cat, marker in category_markers.items()
        if cat in source_labels
    ]

    legend_ax = fig.add_axes([0.76, 0.5, 0.1, 0.4])
    legend_ax.axis('off')

    legend1 = legend_ax.legend(
        handles=cluster_legend_elements,
        title="Clusters",
        title_fontsize=18,
        prop={'weight': 'bold', 'size': 18},
        labelspacing=0.8,
        borderpad=1.0,
        handletextpad=0.5,
        frameon=True,
        loc='center left'
    )
    legend_ax.add_artist(legend1)

    legend_ax2 = fig.add_axes([0.76, 0.1, 0.1, 0.4])
    legend_ax2.axis('off')

    legend2 = legend_ax2.legend(
        handles=category_legend_elements,
        title="Categories",
        title_fontsize=18,
        prop={'weight': 'bold', 'size': 18},
        labelspacing=0.8,
        borderpad=1.0,
        handletextpad=0.5,
        frameon=True,
        loc='center left'
    )
    legend_ax2.add_artist(legend2)

    ax.set_title(f"{method_name} - All Clusters Distribution", fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)
    ax.margins(x=0.08, y=0.08)
    ax.set_aspect('equal', adjustable='datalim')

    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    os.makedirs(os.path.join(save_dir, 'cluster_images'), exist_ok=True)
    output_path = os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution.{output_format}")
    save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.20}
    if output_format == 'tif':
        save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
    plt.savefig(output_path, **save_kwargs)
    plt.close()

    print(f"  Global visualization without thumbnails completed for {method_name}")

    if trained_umap is None:
        return reducer
    else:
        return trained_umap


def visualize_all_clusters_without_images_1(features, labels, source_labels, method_name, save_dir,
                                            trained_umap=None, output_format="tif"):
    """Plot candidate clusters using the alternate visualization layout."""
    print(f"Creating global UMAP visualization without thumbnails for {method_name}...")

    if trained_umap is None:
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        embedding = trained_umap.transform(features)

    unique_clusters = sorted(set(labels))

    color_map = {
        i: f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for i, c in category_colors.items()
    }

    set_plot_font()

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])

    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embedding = embedding[cluster_indices]
        cluster_source_labels = [source_labels[i] for i in cluster_indices]

        for category in set(source_labels):
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            if category_indices:
                category_embedding = cluster_embedding[category_indices]
                ax.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=color_map[category],
                    marker='o',
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )

    category_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map[cat],
                   markersize=16, label=category_names[cat])
        for cat in sorted(category_colors.keys())
        if cat in source_labels
    ]

    legend_ax = fig.add_axes([0.76, 0.5, 0.1, 0.4])
    legend_ax.axis('off')

    legend = legend_ax.legend(
        handles=category_legend_elements,
        loc='center left',
        title="Categories",
        title_fontsize=18,
        prop={'weight': 'bold', 'size': 18},
        labelspacing=0.5,
        borderpad=1.0,
        handletextpad=0.5,
        framealpha=0.9
    )
    legend_ax.add_artist(legend)

    ax.set_title(f"{method_name} - All Clusters Distribution", fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)

    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    os.makedirs(os.path.join(save_dir, 'cluster_images'), exist_ok=True)
    ax.margins(x=0.08, y=0.08)
    ax.set_aspect('equal', adjustable='datalim')
    output_path = os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution_1.{output_format}")
    save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.20}
    if output_format == 'tif':
        save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
    plt.savefig(output_path, **save_kwargs)
    plt.close()

    print(f"  Global visualization without thumbnails completed for {method_name}")

    if trained_umap is None:
        return reducer
    else:
        return trained_umap


def improved_visualize_clusters_with_images(features, labels, source_labels, image_paths, method_name, save_dir,
                                            log_sizes=None, thumbnail_size=30, max_images=1000,
                                            output_format="tif"):
    """Plot candidate clusters with image thumbnails."""
    print(f"Creating per-cluster visualizations for {method_name}...")

    if log_sizes is None:
        print("Computing cell sizes...")
        cell_sizes = []
        for img_path in tqdm(image_paths, desc="Computing cell sizes"):
            size = count_non_background_pixels(img_path)
            cell_sizes.append(size)
        log_sizes = np.log1p(np.array(cell_sizes))

    if np.max(log_sizes) > np.min(log_sizes):
        normalized_log_sizes = (log_sizes - np.min(log_sizes)) / (np.max(log_sizes) - np.min(log_sizes))
        size_scale_factors = 0.5 + 1.5 * normalized_log_sizes
    else:
        size_scale_factors = np.ones_like(log_sizes)

    unique_clusters = sorted(set(labels))

    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]

        if len(cluster_indices) > max_images:
            np.random.seed(42)
            selected_indices = np.random.choice(cluster_indices, max_images, replace=False)
        else:
            selected_indices = cluster_indices

        cluster_features = features[selected_indices]
        cluster_source_labels = [source_labels[i] for i in selected_indices]
        cluster_image_paths = [image_paths[i] for i in selected_indices]
        cluster_size_factors = [size_scale_factors[i] for i in selected_indices]

        if len(cluster_features) == 1:
            cluster_embedding = np.array([[0.0, 0.0]])
            print(f"  Cluster {cluster} has only 1 sample, placing at origin")
        elif len(cluster_features) == 2:
            cluster_embedding = np.array([[-0.5, 0.0], [0.5, 0.0]])
            print(f"  Cluster {cluster} has only 2 samples, placing on x-axis")
        elif len(cluster_features) <= 5:
            try:
                cluster_embedding = PCA(n_components=2).fit_transform(cluster_features)
                print(f"  Cluster {cluster} has {len(cluster_features)} samples, using PCA")
            except Exception as e:
                print(f"  Cluster {cluster} PCA failed: {e}, using random projection")
                cluster_embedding = np.random.randn(len(cluster_features), 2) * 0.5
        else:
            cluster_reducer = umap.UMAP(
                n_neighbors=min(15, len(cluster_features) - 1),
                min_dist=0.1,
                n_components=2,
                metric='euclidean',
                spread=5.0,
                random_state=42
            )
            cluster_embedding = cluster_reducer.fit_transform(cluster_features)
            print(f"  Cluster {cluster} has {len(cluster_features)} samples, using UMAP")

        set_plot_font()

        plt.figure(figsize=(16, 16))

        plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1],
                    c=[f'#{category_colors[sl][0]:02x}{category_colors[sl][1]:02x}{category_colors[sl][2]:02x}'
                       for sl in cluster_source_labels],
                    alpha=0.3, s=50)

        if len(cluster_embedding) > 1:
            distances = pdist(cluster_embedding)
            avg_distance = np.mean(distances)
            base_size = min(thumbnail_size, avg_distance * 10)
            base_size = max(base_size, 20)
        else:
            base_size = thumbnail_size

        for i, (x, y, source_label, img_path, size_factor) in enumerate(zip(
            cluster_embedding[:, 0], cluster_embedding[:, 1],
            cluster_source_labels, cluster_image_paths, cluster_size_factors
        )):
            try:
                img = Image.open(img_path).convert('RGB')
                width, height = img.size
                aspect_ratio = width / height
                img_size = int(base_size * size_factor)

                if aspect_ratio >= 1:
                    new_width = img_size
                    new_height = int(img_size / aspect_ratio)
                else:
                    new_height = img_size
                    new_width = int(img_size * aspect_ratio)

                img = img.resize((new_width, new_height), Image.LANCZOS)
                img_array = np.array(img)

                border_color = category_colors.get(source_label, (0, 0, 0))
                border_width = 2
                img_array[:border_width, :, :] = border_color
                img_array[-border_width:, :, :] = border_color
                img_array[:, :border_width, :] = border_color
                img_array[:, -border_width:, :] = border_color

                bordered_img = Image.fromarray(img_array)
                imagebox = OffsetImage(bordered_img, zoom=1.0)
                imagebox.image.axes = plt.gca()
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
                plt.gca().add_artist(ab)

            except Exception as e:
                print(f"Cannot display image {img_path}: {e}")
                continue

        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}',
                       markersize=18, label=category_names[label])
            for label, c in category_colors.items()
            if label in cluster_source_labels
        ]
        plt.legend(handles=legend_elements, loc='upper right', prop={'weight': 'bold', 'size': 18},
                   labelspacing=0.5,
                   borderpad=1.0,
                   handletextpad=0.5,
                   framealpha=0.9)

        plt.title(f"{method_name} - Cluster {cluster} ({len(cluster_indices)} samples)", fontsize=32, fontweight='bold', pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
        plt.ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)

        plt.axis('equal')
        plt.margins(0.2)
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=18, pad=10)
        for tick in plt.gca().get_xticklabels():
            tick.set_fontweight('bold')
        for tick in plt.gca().get_yticklabels():
            tick.set_fontweight('bold')

        os.makedirs(os.path.join(save_dir, 'cluster_images'), exist_ok=True)
        output_path = os.path.join(save_dir, 'cluster_images', f"{method_name}_cluster{cluster}_images_improved.{output_format}")
        save_kwargs = {'dpi': 150, 'bbox_inches': 'tight'}
        if output_format == 'tif':
            save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
        plt.savefig(output_path, **save_kwargs)
        plt.close()

    print(f"  Per-cluster visualization completed for {method_name}")


def visualize_matching_clusters(reference_results, candidate_results, matching_pairs, save_dir,
                                trained_umap=None, rule_matched_clusters=None, output_format="tif"):
    """Visualize matched candidate and reference clusters."""

    reference_embedding = reference_results['features']
    reference_labels = reference_results['labels']
    reference_source_labels = reference_results['source_labels']

    candidate_embedding = candidate_results['features']
    candidate_labels = candidate_results['labels']
    candidate_source_labels = candidate_results['source_labels']

    n_ref_clusters = reference_results['n_clusters']
    n_cand_clusters = candidate_results['n_clusters']

    ref_colors = plt.cm.tab10(np.linspace(0, 1, n_ref_clusters))
    cand_colors = plt.cm.tab20(np.linspace(0, 1, n_cand_clusters))

    if trained_umap is None:
        reducer = umap.UMAP(random_state=42)
        reference_umap = reducer.fit_transform(reference_embedding)
        candidate_umap = reducer.transform(candidate_embedding)
    else:
        reference_umap = trained_umap.transform(reference_embedding)
        candidate_umap = trained_umap.transform(candidate_embedding)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    for cluster in range(n_ref_clusters):
        cluster_indices = np.where(reference_labels == cluster)[0]
        cluster_embedding = reference_umap[cluster_indices]
        cluster_source_labels = [reference_source_labels[i] for i in cluster_indices]

        for category in set(reference_source_labels):
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            if category_indices:
                category_embedding = cluster_embedding[category_indices]
                plt.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=[ref_colors[cluster]],
                    marker=category_markers.get(category, 'o'),
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )

    plt.title('Reference Domain Clusters', fontsize=32, fontweight='bold', pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold', labelpad=20)
    plt.ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold', labelpad=20)

    plt.subplot(1, 2, 2)
    for cluster in range(n_cand_clusters):
        cluster_indices = np.where(candidate_labels == cluster)[0]
        cluster_embedding = candidate_umap[cluster_indices]
        cluster_source_labels = [candidate_source_labels[i] for i in cluster_indices]

        for category in set(candidate_source_labels):
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            if category_indices:
                category_embedding = cluster_embedding[category_indices]
                plt.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=cand_colors[cluster],
                    marker=category_markers.get(category, 'o'),
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )

    plt.title('Candidate Domain Clusters', fontsize=32, fontweight='bold', pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold', labelpad=20)
    plt.ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold', labelpad=20)

    if rule_matched_clusters:
        rule_info = []
        for rule_name, clusters in rule_matched_clusters.items():
            rule_info.append(f"{rule_name}: {len(clusters)}")
        rule_text = ", ".join(rule_info)
        plt.figtext(0.5, 0.01, f'Matched Pairs: {len(matching_pairs)} ({rule_text})',
                    ha='center', fontsize=14)
    else:
        plt.figtext(0.5, 0.01, f'Matched Pairs: {len(matching_pairs)}',
                    ha='center', fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(save_dir, f'matching_visualization.{output_format}')
    save_kwargs = {'dpi': 300, 'bbox_inches': 'tight'}
    if output_format == 'tif':
        save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
    plt.savefig(output_path, **save_kwargs)
    plt.close()

    print(f"Matching visualization saved to '{output_path}'")

    if trained_umap is None:
        return reducer
    else:
        return trained_umap


def plot_similarity_heatmap(similarity_matrix, x_labels, y_labels, title, save_path, save_tiff=True):
    """Save a labeled similarity heatmap."""
    plt.figure(figsize=(12, 10))

    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=x_labels,
                yticklabels=y_labels)

    plt.title(title, fontsize=28, fontweight='bold', pad=20)
    plt.xlabel("Candidate", fontsize=24, fontweight='bold', labelpad=20)
    plt.ylabel("Reference", fontsize=24, fontweight='bold', labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=10)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    if save_tiff:
        plt.savefig(save_path.replace('.png', '.tif'), dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    plt.close()
    print(f"  Heatmap saved to: {save_path}")


def visualize_filtered_clusters(
    reference_results,
    candidate_results,
    matching_pairs,
    save_dir,
    reference_umap=None,
    rule_matched_clusters=None,
    target_ref_indices=None,
    thumbnail_size=30,
    max_images=1000,
    passed_indices_dict=None,
):
    """Visualize cells retained by the configured matching rules."""

    print("Creating visualization for matched clusters...")

    features = candidate_results['features']
    labels = candidate_results['labels']
    source_labels = candidate_results['source_labels']
    image_paths = candidate_results['image_paths']

    cell_sims = cosine_similarity(features, reference_results['cluster_centers'])

    if target_ref_indices is None:
        target_ref_indices = [0, 5]

    matched_clusters = []

    if rule_matched_clusters:
        for rule_name, clusters in rule_matched_clusters.items():
            for cluster in clusters:
                matched_clusters.append((cluster, rule_name))
    elif not matched_clusters:
        for ref_idx, cand_idx, _ in matching_pairs:
            if ref_idx == 5:
                matched_clusters.append((cand_idx, 'rule1'))
            elif ref_idx == 0:
                matched_clusters.append((cand_idx, 'rule2'))

    if hasattr(candidate_results, 'log_sizes'):
        log_sizes = candidate_results['log_sizes']
    else:
        print("Computing cell sizes...")
        cell_sizes = []
        for img_path in tqdm(image_paths, desc="Computing cell sizes"):
            size = count_non_background_pixels(img_path)
            cell_sizes.append(size)
        log_sizes = np.log1p(np.array(cell_sizes))

    if np.max(log_sizes) > np.min(log_sizes):
        normalized_log_sizes = (log_sizes - np.min(log_sizes)) / (np.max(log_sizes) - np.min(log_sizes))
        size_scale_factors = 0.5 + 1.5 * normalized_log_sizes
    else:
        size_scale_factors = np.ones_like(log_sizes)

    matched_vis_dir = os.path.join(save_dir, 'matched_clusters_visualization')
    os.makedirs(matched_vis_dir, exist_ok=True)

    for cluster_idx, rule_type in matched_clusters:
        cluster_indices = np.where(labels == cluster_idx)[0]

        if passed_indices_dict is not None:
            rule_passed = set(passed_indices_dict.get(rule_type, []))
            filtered_indices = [
                idx for idx in cluster_indices
                if idx in rule_passed
            ]
        else:
            filtered_indices = []
            for idx in cluster_indices:
                most_similar_ref = np.argmax(cell_sims[idx])
                if most_similar_ref in target_ref_indices:
                    filtered_indices.append(idx)

        if not filtered_indices:
            print(f"  Cluster {cluster_idx} ({rule_type}) has no cells matching the specified reference clusters, skipping")
            continue

        if len(filtered_indices) > max_images:
            np.random.seed(42)
            selected_indices = np.random.choice(filtered_indices, max_images, replace=False)
        else:
            selected_indices = filtered_indices

        cluster_features = features[cluster_indices]
        filtered_features = features[filtered_indices]

        cluster_source_labels = [source_labels[i] for i in cluster_indices]
        cluster_image_paths = [image_paths[i] for i in cluster_indices]
        cluster_size_factors = [size_scale_factors[i] for i in cluster_indices]

        filtered_source_labels = [source_labels[i] for i in filtered_indices]
        filtered_image_paths = [image_paths[i] for i in filtered_indices]
        filtered_size_factors = [size_scale_factors[i] for i in filtered_indices]

        if len(cluster_features) == 1:
            cluster_embedding = np.array([[0.0, 0.0]])
        elif len(cluster_features) == 2:
            cluster_embedding = np.array([[-0.5, 0.0], [0.5, 0.0]])
        elif len(cluster_features) <= 5:
            try:
                cluster_embedding = PCA(n_components=2).fit_transform(cluster_features)
            except Exception as e:
                print(f"  Cluster {cluster_idx} PCA failed: {e}, using random projection")
                cluster_embedding = np.random.randn(len(cluster_features), 2) * 0.5
        else:
            cluster_reducer = umap.UMAP(
                n_neighbors=min(15, len(cluster_features) - 1),
                min_dist=0.1,
                n_components=2,
                metric='euclidean',
                spread=5.0,
                random_state=42
            )
            cluster_embedding = cluster_reducer.fit_transform(cluster_features)

        filtered_mask = np.zeros(len(cluster_indices), dtype=bool)
        for idx in filtered_indices:
            pos = np.where(cluster_indices == idx)[0]
            if len(pos) > 0:
                filtered_mask[pos[0]] = True

        set_plot_font()

        plt.figure(figsize=(16, 16))

        plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1],
                    c='lightgray', alpha=0.2, s=30)

        plt.scatter(cluster_embedding[filtered_mask, 0], cluster_embedding[filtered_mask, 1],
                    c=[f'#{category_colors[sl][0]:02x}{category_colors[sl][1]:02x}{category_colors[sl][2]:02x}'
                       for sl in filtered_source_labels],
                    alpha=0.8, s=80)

        if len(cluster_embedding) > 1:
            distances = pdist(cluster_embedding)
            avg_distance = np.mean(distances)
            base_size = min(thumbnail_size, avg_distance * 10)
            base_size = max(base_size, 20)
        else:
            base_size = thumbnail_size

        for i, idx in enumerate(np.where(filtered_mask)[0]):
            x, y = cluster_embedding[idx, 0], cluster_embedding[idx, 1]
            source_label = cluster_source_labels[idx]
            img_path = cluster_image_paths[idx]
            size_factor = cluster_size_factors[idx]

            try:
                img = Image.open(img_path).convert('RGB')
                width, height = img.size
                aspect_ratio = width / height
                img_size = int(base_size * size_factor)

                if aspect_ratio >= 1:
                    new_width = img_size
                    new_height = int(img_size / aspect_ratio)
                else:
                    new_height = img_size
                    new_width = int(img_size * aspect_ratio)

                img = img.resize((new_width, new_height), Image.LANCZOS)
                img_array = np.array(img)

                border_color = category_colors.get(source_label, (0, 0, 0))
                border_width = 2
                img_array[:border_width, :, :] = border_color
                img_array[-border_width:, :, :] = border_color
                img_array[:, :border_width, :] = border_color
                img_array[:, -border_width:, :] = border_color

                bordered_img = Image.fromarray(img_array)
                imagebox = OffsetImage(bordered_img, zoom=1.0)
                imagebox.image.axes = plt.gca()
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
                plt.gca().add_artist(ab)

            except Exception as e:
                print(f"Cannot display image {img_path}: {e}")
                continue

        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}',
                       markersize=16, label=category_names[label])
            for label, c in category_colors.items()
            if label in filtered_source_labels
        ]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='lightgray',
                       markersize=16, label='filtered cells')
        )
        plt.legend(handles=legend_elements,
                   loc='upper right',
                   prop={'weight': 'bold', 'size': 16},
                   labelspacing=0.5,
                   borderpad=1.0,
                   handletextpad=0.5,
                   framealpha=0.9)

        plt.title(f"{rule_type} - Cluster {cluster_idx} Matched Cells ({len(filtered_indices)}/{len(cluster_indices)} cells)",
                  fontsize=32, fontweight='bold', pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
        plt.ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)

        plt.axis('equal')
        plt.margins(0.2)
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=18, pad=10)
        for tick in plt.gca().get_xticklabels():
            tick.set_fontweight('bold')
        for tick in plt.gca().get_yticklabels():
            tick.set_fontweight('bold')

        plt.savefig(os.path.join(matched_vis_dir, f"{rule_type}_cluster{cluster_idx}_matched_cells.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("  Matched cluster visualization completed")
    return matched_vis_dir