# data/data_utils.py

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os


def align_cell_by_major_axis(image):
    """Align a cell image to its estimated major axis."""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest_contour)
    center, (width, height), angle = rect

    if height > width:
        angle += 90

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    h, w = image.shape[:2]
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    aligned_image = cv2.warpAffine(image, M, (new_w, new_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
    return aligned_image


class CellImageDataset(Dataset):
    """Dataset of cell images with optional major-axis alignment."""

    def __init__(self, image_paths, transform=None, source_labels=None, align_cells=False):
        """Initialize image paths, transforms, source labels, and alignment settings."""
        self.image_paths = image_paths
        self.transform = transform
        self.source_labels = source_labels if source_labels is not None else [0] * len(image_paths)
        self.align_cells = align_cells
        self.aligned_images_cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        source_label = self.source_labels[idx]

        try:
            if self.align_cells:
                if img_path not in self.aligned_images_cache:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Failed to read image: {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(align_cell_by_major_axis(img))
                    self.aligned_images_cache[img_path] = img
                else:
                    img = self.aligned_images_cache[img_path]
            else:
                img = Image.open(img_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img, source_label, idx

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            dummy_img = torch.zeros((3, 224, 224)) if self.transform else np.zeros((224, 224, 3), dtype=np.uint8)
            return dummy_img, source_label, idx


def collect_image_paths(directory, extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']):
    """Collect supported image paths from one directory."""
    paths = []
    if os.path.exists(directory):
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(directory, ext)))
    return paths


def collect_reference_data(malignant_dir, benign_dir):
    """Collect malignant and benign reference image paths with source labels."""
    image_paths = []
    labels = []

    categories = {
        "LUAD": 1,
        "LUSC": 2,
        "SCLC": 3,
    }

    subdirs = [d for d in os.listdir(malignant_dir)
               if os.path.isdir(os.path.join(malignant_dir, d))]

    if subdirs and any(subdir in categories for subdir in subdirs):
        # Structured malignant directory: one sub-folder per cancer type
        for cat_name, label in categories.items():
            subdir_path = os.path.join(malignant_dir, cat_name)
            if os.path.exists(subdir_path):
                paths = collect_image_paths(subdir_path)
                image_paths.extend(paths)
                labels.extend([label] * len(paths))
    else:
        # Flat malignant directory
        malignant_paths = collect_image_paths(malignant_dir)
        image_paths.extend(malignant_paths)
        labels.extend([5] * len(malignant_paths))

    benign_paths = collect_image_paths(benign_dir)
    image_paths.extend(benign_paths)
    labels.extend([4] * len(benign_paths))

    return image_paths, labels


def extract_patient_id(patient_folder):
    """Return the patient identifier encoded by a folder name."""
    return os.path.basename(patient_folder)