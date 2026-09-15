#!/usr/bin/env python
"""Inference utilities."""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import shutil
import argparse

from shilab_classifier.models.model_definitions import create_model
from shilab_classifier.training.preprocessing import pad_to_square_transform, pad_to_square_299_transform


class UnlabeledImageDataset(Dataset):
    """Unlabeled image dataset."""

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform   = transform
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(exts)
        ])
        self.samples = [(os.path.join(folder_path, f), 0) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.samples[idx][0]
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0


def load_model_weights(model, model_path, device):
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint['model_state_dict']")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint['state_dict']")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict directly")
    return model


def process_malignant_predictions(results_df, malignant_output_dir, confidence_threshold, output_dir):
    """Save and summarize malignant predictions."""
    malignant_predictions = results_df[
        (results_df['predicted_class'] == 'malignant') &
        (results_df['prob_malignant'] >= confidence_threshold)
    ]

    print(f"Found {len(malignant_predictions)} malignant prediction(s) "
          f"with confidence >= {confidence_threshold}")

    if len(malignant_predictions) == 0:
        print("No malignant predictions meet the confidence threshold")
        return

    processed_count    = 0
    renamed_files_info = []

    for _, row in malignant_predictions.iterrows():
        original_path  = row['image_path']
        confidence     = row['prob_malignant']
        name, ext      = os.path.splitext(os.path.basename(original_path))
        new_filename   = f"{name}_malignant_{confidence:.4f}{ext}"
        new_path       = os.path.join(malignant_output_dir, new_filename)

        try:
            shutil.copy2(original_path, new_path)
            processed_count += 1
            renamed_files_info.append({
                'original_path': original_path,
                'new_path'     : new_path,
                'confidence'   : confidence,
            })
        except Exception as e:
            print(f"  Warning: error processing {original_path}: {e}")

    if renamed_files_info:
        renamed_df   = pd.DataFrame(renamed_files_info)
        renamed_path = os.path.join(output_dir, 'malignant_images_with_confidence.csv')
        renamed_df.to_csv(renamed_path, index=False, encoding='utf-8-sig')
        print(f"Processed {processed_count} malignant image(s); details saved to {renamed_path}")
        print(f"Malignant images with confidence scores saved to {malignant_output_dir}")


def predict_new_data(
    model_name="DenseNet161",
    model_path=None,
    input_dir=None,
    output_dir=None,
    batch_size=16,
    num_workers=0,
    confidence_threshold=0.5,
    device=None,
    mean=None,
    std=None
):
    """Run binary classification on unlabelled images."""
    if model_path is None or input_dir is None or output_dir is None:
        raise ValueError("model_path, input_dir, and output_dir are all required")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes  = 2
    idx_to_class = {0: "benign", 1: "malignant"}

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std  = [0.229, 0.224, 0.225]

    malignant_output_dir = os.path.join(output_dir, 'malignant_images')
    os.makedirs(output_dir,          exist_ok=True)
    os.makedirs(malignant_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Predicting with model: {model_name}")
    print(f"{'='*60}")
    print(f"  Model path:          {model_path}")
    print(f"  Input directory:     {input_dir}")
    print(f"  Output directory:    {output_dir}")
    print(f"  Device:              {device}")
    print(f"  Batch size:          {batch_size}")
    print(f"  Confidence threshold:{confidence_threshold}")

    print(f"Loading model weights: {model_path}")
    model = create_model(model_name, num_classes=num_classes)
    model = load_model_weights(model, model_path, device)
    model = model.to(device)
    model.eval()

    print(f"Normalisation  mean={mean}  std={std}")
    pad_transform = (
        pad_to_square_299_transform if 'Inception' in model_name
        else pad_to_square_transform
    )
    transform = transforms.Compose([
        transforms.Lambda(pad_transform),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    print(f"Loading data from: {input_dir}")
    has_subfolders = any(
        os.path.isdir(os.path.join(input_dir, d))
        for d in os.listdir(input_dir)
        if not d.startswith('.')
    )

    if has_subfolders:
        print("Subfolder structure detected; using ImageFolder.")
        print("Warning: verify that the class mapping below matches training.")
        try:
            dataset    = datasets.ImageFolder(input_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
            print(f"Found {len(dataset)} image(s) across {len(dataset.class_to_idx)} class(es):")
            for class_name, idx in dataset.class_to_idx.items():
                print(f"  - folder class {idx}: {class_name}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    else:
        print("Flat folder structure detected; using UnlabeledImageDataset.")
        dataset    = UnlabeledImageDataset(input_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        print(f"Found {len(dataset)} image(s)")

    if len(dataset) == 0:
        print("Error: no images found. Check the path and file formats.")
        return None

    print("\nRunning inference...")
    all_paths  = []
    all_logits = []
    all_probs  = []
    all_preds  = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}")

            batch_size_actual = inputs.size(0)
            start_idx   = batch_idx * batch_size
            batch_paths = [
                dataset.samples[i][0]
                for i in range(start_idx, min(start_idx + batch_size_actual, len(dataset)))
            ]

            inputs  = inputs.to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_paths.extend(batch_paths)
            all_logits.append(outputs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)

    results_df = pd.DataFrame({
        'image_path'     : all_paths,
        'filename'       : [os.path.basename(p) for p in all_paths],
        'predicted_idx'  : all_preds,
        'predicted_class': [idx_to_class[int(idx)] for idx in all_preds],
        'logit_benign'   : all_logits[:, 0],
        'logit_malignant': all_logits[:, 1],
        'prob_benign'    : all_probs[:, 0],
        'prob_malignant' : all_probs[:, 1],
    })

    results_path = os.path.join(output_dir, 'prediction_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\nPrediction results saved to {results_path}")

    class_counts = results_df['predicted_class'].value_counts()
    print("\nPrediction summary:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} ({count / len(results_df) * 100:.1f}%)")

    print("\nProcessing malignant predictions...")
    process_malignant_predictions(results_df, malignant_output_dir, confidence_threshold, output_dir)

    plt.figure(figsize=(10, 6))
    plt.hist(results_df['prob_malignant'], bins=20, alpha=0.7,
             color='steelblue', edgecolor='white')
    plt.axvline(x=confidence_threshold, color='red', linestyle='--',
                label=f'Threshold = {confidence_threshold}')
    plt.title('Malignant Prediction Probability Distribution')
    plt.xlabel('Malignant probability')
    plt.ylabel('Number of images')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prob_hist_path = os.path.join(output_dir, 'malignant_probability_distribution.png')
    plt.savefig(prob_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Probability distribution plot saved to {prob_hist_path}")

    print("\nInference complete.")
    return results_df


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference on new data using a trained model'
    )
    parser.add_argument('-m', '--model',      type=str, default="DenseNet161",
                        help='Model name (default: DenseNet161)')
    parser.add_argument('-w', '--weights',    type=str, required=True,
                        help='Path to model weights file (.pth)')
    parser.add_argument('-i', '--input',      type=str, required=True,
                        help='Input data directory')
    parser.add_argument('-o', '--output',     type=str, required=True,
                        help='Output results directory')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('-j', '--workers',    type=int, default=0,
                        help='Number of data-loader workers (default: 0)')
    parser.add_argument('-t', '--threshold',  type=float, default=0.5,
                        help='Malignant confidence threshold (default: 0.5)')
    parser.add_argument('--mean', type=float, nargs=3, default=None,
                        help='Normalisation mean [R G B] (default: ImageNet [0.485, 0.456, 0.406])')
    parser.add_argument('--std',  type=float, nargs=3, default=None,
                        help='Normalisation std  [R G B] (default: ImageNet [0.229, 0.224, 0.225])')
    return parser.parse_args(argv)


def main(argv=None):
    """Run the command-line application."""
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict_new_data(
        model_name=args.model,
        model_path=args.weights,
        input_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
        confidence_threshold=args.threshold,
        device=device,
        mean=args.mean,
        std=args.std,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
