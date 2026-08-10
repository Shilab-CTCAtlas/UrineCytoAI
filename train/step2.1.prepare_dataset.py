"""
prepare_dataset.py
Convert LabelMe JSON annotations to YOLO format and split into train/val/test sets.

Input structure:
    input_dir/
        image1.jpg, image1.json, image2.jpg, ...

Output structure:
    output_dir/
        images/{train,val,test}/
        labels/{train,val,test}/
        labels_json/{train,val,test}/  (optional)
        classes.txt, custom.yaml
"""

import os
import json
import shutil
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# ── Path config ──────────────────────────────
input_dir  = r"/path/to/your/input_data"
output_dir = r"/path/to/your/output_yolo_data"

# ── Split ratios & settings ───────────────────
train_ratio = 0.7
val_ratio   = 0.2
test_ratio  = 0.1
seed        = 42
keep_json   = True  # whether to copy original JSON files

# ── Class mapping ─────────────────────────────
class_mapping = {
    "single_cell": 0,
    "cluster":     1,
    "impurity":    2,
    "part":        3,
    "vague":       4,
}
class_names = {v: k for k, v in class_mapping.items()}


def create_empty_json(image_path: str, json_path: str) -> bool:
    """Create an empty LabelMe-format JSON for images without annotations."""
    try:
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2] if img is not None else (0, 0)
        empty = {
            "version": "5.0.1", "flags": {}, "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": img_height, "imageWidth": img_width,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(empty, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Failed to create empty JSON: {e}")
        return False


def convert_json_to_yolo(json_path: str, output_path: str, class_mapping: dict) -> None:
    """Convert a LabelMe JSON annotation to YOLO TXT format."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth",  0)
        img_h = data.get("imageHeight", 0)

        if img_w == 0 or img_h == 0:
            print(f"Warning: invalid image size in {json_path}")
            open(output_path, "w").close()
            return

        lines = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue
            class_id = class_mapping.get(shape.get("label"), -1)
            if class_id == -1:
                print(f"Warning: unknown label '{shape.get('label')}', skipped")
                continue

            pts = shape.get("points", [])
            if len(pts) == 4:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            elif len(pts) == 2:
                x_min, x_max = min(pts[0][0], pts[1][0]), max(pts[0][0], pts[1][0])
                y_min, y_max = min(pts[0][1], pts[1][1]), max(pts[0][1], pts[1][1])
            else:
                print(f"Warning: unexpected point count {len(pts)} in {json_path}")
                continue

            cx = (x_min + x_max) / 2.0 / img_w
            cy = (y_min + y_max) / 2.0 / img_h
            w  = (x_max - x_min) / img_w
            h  = (y_max - y_min) / img_h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        open(output_path, "w").close()


def plot_split_pie(train_n: int, val_n: int, test_n: int) -> None:
    """Save a pie chart of the dataset split to output_dir."""
    plt.figure(figsize=(7, 6))
    plt.pie(
        [train_n, val_n, test_n],
        labels=["Train", "Val", "Test"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#2196F3", "#FFC107"],
    )
    plt.title("Dataset Split")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "split_ratio.png"), dpi=150)
    plt.show()


def main():
    random.seed(seed)
    np.random.seed(seed)

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
        if keep_json:
            os.makedirs(os.path.join(output_dir, "labels_json", split), exist_ok=True)

    # Collect image-JSON pairs
    print("\n[1/4] Collecting image-JSON pairs ...")
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    print(f"Found {len(image_files)} images")

    pairs, created = [], 0
    for img_file in image_files:
        img_path  = os.path.join(input_dir, img_file)
        stem      = os.path.splitext(img_file)[0]
        json_path = os.path.join(input_dir, stem + ".json")
        if not os.path.exists(json_path):
            if create_empty_json(img_path, json_path):
                created += 1
        pairs.append((img_path, json_path, stem))
    print(f"Total pairs: {len(pairs)}, empty JSON created: {created}")

    # Split dataset
    print("\n[2/4] Splitting dataset ...")
    random.shuffle(pairs)
    total     = len(pairs)
    train_end = int(total * train_ratio)
    val_end   = train_end + int(total * val_ratio)
    splits = {
        "train": pairs[:train_end],
        "val":   pairs[train_end:val_end],
        "test":  pairs[val_end:],
    }
    for name, data in splits.items():
        print(f"  {name}: {len(data)} ({len(data)/total*100:.1f}%)")

    try:
        plot_split_pie(len(splits["train"]), len(splits["val"]), len(splits["test"]))
    except Exception as e:
        print(f"Pie chart skipped: {e}")

    # Copy files and convert labels
    print("\n[3/4] Processing files ...")
    for split_name, split_data in splits.items():
        for img_path, json_path, stem in tqdm(split_data, desc=split_name):
            ext = os.path.splitext(img_path)[1]
            shutil.copy2(img_path, os.path.join(output_dir, "images", split_name, stem + ext))
            convert_json_to_yolo(
                json_path,
                os.path.join(output_dir, "labels", split_name, stem + ".txt"),
                class_mapping,
            )
            if keep_json:
                shutil.copy2(json_path, os.path.join(output_dir, "labels_json", split_name, stem + ".json"))

    # Generate config files
    print("\n[4/4] Generating config files ...")
    with open(os.path.join(output_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names[i] for i in sorted(class_names)))

    names_block = "\n".join(f"  {i}: {class_names[i]}" for i in sorted(class_names))
    yaml_content = f"""# YOLOv12 dataset configuration
path: {output_dir}
train: images/train
val:   images/val
test:  images/test

names:
{names_block}
"""
    with open(os.path.join(output_dir, "custom.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("Done! Dataset is ready for YOLOv12 training.")


if __name__ == "__main__":
    main()