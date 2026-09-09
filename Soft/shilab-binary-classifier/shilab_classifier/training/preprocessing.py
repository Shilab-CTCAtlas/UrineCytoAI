"""Model-training utilities."""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import traceback


def pad_to_square(image, fill_color=(255, 255, 255)):
    """Pad an image to a square canvas and resize to 224×224."""
    try:
        width, height = image.size
        max_side  = max(width, height)
        new_image = Image.new('RGB', (max_side, max_side), fill_color)
        new_image.paste(image, ((max_side - width) // 2, (max_side - height) // 2))
        return new_image.resize((224, 224), Image.LANCZOS)
    except Exception as e:
        print(f"Error in pad_to_square: {e}  image={image}")
        return Image.new('RGB', (224, 224), fill_color)


def pad_to_square_transform(img):
    """Lambda-compatible wrapper for pad_to_square (224×224)."""
    try:
        return pad_to_square(img)
    except Exception as e:
        print(f"Error in pad_to_square_transform: {e}")
        return Image.new('RGB', (224, 224), (255, 255, 255))


def pad_to_square_299(image, fill_color=(255, 255, 255)):
    """Pad an image to a square canvas and resize to 299×299."""
    try:
        width, height = image.size
        max_side  = max(width, height)
        new_image = Image.new('RGB', (max_side, max_side), fill_color)
        new_image.paste(image, ((max_side - width) // 2, (max_side - height) // 2))
        return new_image.resize((299, 299), Image.LANCZOS)
    except Exception as e:
        print(f"Error in pad_to_square_299: {e}  image={image}")
        return Image.new('RGB', (299, 299), fill_color)


def pad_to_square_299_transform(img):
    """Lambda-compatible wrapper for pad_to_square_299 (299×299)."""
    try:
        return pad_to_square_299(img)
    except Exception as e:
        print(f"Error in pad_to_square_299_transform: {e}")
        return Image.new('RGB', (299, 299), (255, 255, 255))


def calculate_mean_std(data_dir, model_name=None):
    """Calculate per-channel mean and std for a dataset.

    Falls back to ImageNet statistics if the directory is missing or an
    error occurs during computation.
    """
    print(f"Computing mean and std for: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: directory not found: {data_dir}. Using ImageNet defaults.")
        return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

    try:
        if model_name and 'Inception' in model_name:
            transform = transforms.Compose([
                transforms.Lambda(pad_to_square_299_transform),
                transforms.ToTensor()
            ])
            print(f"Using 299×299 input size for {model_name}")
        else:
            transform = transforms.Compose([
                transforms.Lambda(pad_to_square_transform),
                transforms.ToTensor()
            ])
            print("Using 224×224 input size")

        dataset    = datasets.ImageFolder(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        print(f"Dataset loaded: {len(dataset)} image(s)")

        mean         = torch.zeros(3)
        std          = torch.zeros(3)
        total_images = 0

        for i, (images, _) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"  Processing batch {i}/{len(dataloader)}")
            batch_size    = images.size(0)
            images_flat   = images.view(batch_size, images.size(1), -1)
            mean         += images_flat.mean(2).sum(0)
            std          += images_flat.std(2).sum(0)
            total_images += batch_size

        mean /= total_images
        std  /= total_images

        print(f"Done: mean={mean}, std={std}")
        return mean, std

    except Exception as e:
        print(f"Error computing mean and std: {e}")
        traceback.print_exc()
        return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])