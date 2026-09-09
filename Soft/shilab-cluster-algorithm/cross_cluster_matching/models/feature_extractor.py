# models/feature_extractor.py

import numpy as np
import torch
from tqdm import tqdm


def extract_features_with_cell_size(model, dataloader, device, cell_sizes, feature_layer='penultimate', cell_size_weight=1.0):
    """Extract model features and append weighted cell-size information."""
    model.eval()
    features_list = []
    source_labels_list = []
    indices_list = []

    with torch.no_grad():
        for batch_images, batch_labels, batch_indices in tqdm(dataloader, desc="Extracting features"):
            batch_images = batch_images.to(device)
            batch_features = model(batch_images, layer=feature_layer)

            if len(batch_features.shape) == 4:
                batch_features = torch.mean(batch_features, dim=(2, 3))

            batch_features = batch_features.cpu().numpy()
            features_list.append(batch_features)
            source_labels_list.append(batch_labels.numpy())
            indices_list.append(batch_indices.numpy())

    features = np.vstack(features_list)
    source_labels = np.concatenate(source_labels_list)
    indices = np.concatenate(indices_list)

    image_paths = [dataloader.dataset.image_paths[idx] for idx in indices]
    batch_cell_sizes = np.array([cell_sizes[idx] for idx in indices])

    log_cell_sizes = np.log1p(batch_cell_sizes)
    if np.max(log_cell_sizes) > np.min(log_cell_sizes):
        normalized_log_cell_sizes = (log_cell_sizes - np.min(log_cell_sizes)) / (np.max(log_cell_sizes) - np.min(log_cell_sizes))
    else:
        normalized_log_cell_sizes = np.zeros_like(log_cell_sizes)

    cell_size_features = cell_size_weight * normalized_log_cell_sizes.reshape(-1, 1)
    features_with_size = np.hstack((features, cell_size_features))

    return features_with_size, source_labels, image_paths