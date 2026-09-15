#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import random
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import openslide
import torch
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming SVS YOLO inference: combines Steps 1 and 2 without writing intermediate patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Path arguments
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to a single input SVS file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for inference outputs (the original Step 2 output_dir), e.g. data_svsModle")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to YOLO model weights")

    # Step 1 patch parameters
    parser.add_argument("--size", type=int, default=1024,
                        help="Patch size")
    parser.add_argument("--overlap", type=float, default=0.05,
                        help="Patch overlap ratio")
    parser.add_argument("--grid_size", type=int, default=3,
                        help="Center-sampling grid size; matches the original Step 1")
    parser.add_argument("--read_workers", type=int, default=4,
                        help="Number of OpenSlide reader threads; each thread opens its own slide object")

    # YOLO inference parameters
    parser.add_argument("--infer_conf", type=float, default=0.1,
                        help="YOLO confidence threshold (lower values retain more candidates)")
    parser.add_argument("--infer_iou", type=float, default=0.3,
                        help="YOLO NMS IoU threshold")
    parser.add_argument("--save_conf", type=float, default=0.4,
                        help="Confidence threshold for saving crops, JSON, and confidence visualizations")
    parser.add_argument("--yolo_batch_size", type=int, default=8,
                        help="YOLO batch size")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="YOLO inference image size; defaults to the patch size")
    parser.add_argument("--half", action="store_true",
                        help="Enable FP16 inference; disabled by default for output consistency")

    # Legacy Step 2 completeness-filtering parameters
    parser.add_argument("--border_threshold", type=int, default=5,
                        help="Border threshold for single_cell and cluster detections")
    parser.add_argument("--crop_padding", type=int, default=5,
                        help="Padding in pixels added around saved crops")
    parser.add_argument("--cluster_area_ratio_max", type=float, default=0.20,
                        help="Maximum area ratio for filtering border-touching cluster fragments")
    parser.add_argument("--cluster_aspect_ratio_max", type=float, default=3.0,
                        help="Maximum aspect ratio for filtering border-touching cluster fragments")

    # Output controls: preserve the original Step 2 defaults
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="Save patch-level JSON files; enabled by default for legacy Step 2 compatibility")
    parser.add_argument("--no_save_json", dest="save_json", action="store_false",
                        help="Do not save patch-level JSON files; save crops and statistics only")
    parser.add_argument("--save_annotation", action="store_true", default=True,
                        help="Save annotation visualizations; enabled by default for legacy Step 2 compatibility")
    parser.add_argument("--no_save_annotation", dest="save_annotation", action="store_false",
                        help="Do not save annotation visualizations to reduce I/O")

    # Optional safeguards: disabled by default to avoid changing results
    parser.add_argument("--skip_background", action="store_true",
                        help="Skip obvious background patches; disabled by default to avoid missed detections")
    parser.add_argument("--bg_sat_thr", type=int, default=15,
                        help="HSV saturation threshold for background filtering")
    parser.add_argument("--bg_val_thr", type=int, default=230,
                        help="HSV brightness threshold for background filtering")
    parser.add_argument("--bg_ratio_thr", type=float, default=0.95,
                        help="Skip a patch when its background-pixel ratio exceeds this threshold")

    return parser.parse_args()


def generate_center_patch_coords(width, height, patch_size=1024, overlap_rate=0.05, grid_size=3):
    overlap_pixels = int(patch_size * overlap_rate)
    step_size = patch_size - overlap_pixels

    num_cols = (width - overlap_pixels) // step_size
    num_rows = (height - overlap_pixels) // step_size

    grid_rows = (num_rows + grid_size - 1) // grid_size
    grid_cols = (num_cols + grid_size - 1) // grid_size

    coords = []
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            center_row = grid_row * grid_size + grid_size // 2
            center_col = grid_col * grid_size + grid_size // 2

            if center_row >= num_rows or center_col >= num_cols:
                continue

            x = center_col * step_size
            y = center_row * step_size
            coords.append((x, y, center_row, center_col))

    return coords, num_rows, num_cols, grid_rows, grid_cols


# Optional background filtering: disabled by default
def is_background_patch_bgr(img_bgr, sat_thr=15, val_thr=230, ratio_thr=0.95):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bg_mask = (s < sat_thr) & (v > val_thr)
    return float(np.mean(bg_mask)) >= ratio_thr

_thread_local = threading.local()

def get_slide(slide_path):    
    if not hasattr(_thread_local, "slide") or _thread_local.slide_path != slide_path:
        if hasattr(_thread_local, "slide"):
            try:
                _thread_local.slide.close()
            except Exception:
                pass
        _thread_local.slide = openslide.OpenSlide(slide_path)
        _thread_local.slide_path = slide_path
    return _thread_local.slide

def read_patch_worker(slide_path, coord_item, patch_size):
    x, y, row, col = coord_item
    slide = get_slide(slide_path) 
    region = slide.read_region((x, y), level=0, size=(patch_size, patch_size))

    region_rgb = region.convert("RGB")
    region_np = np.ascontiguousarray(np.asarray(region_rgb, dtype=np.uint8))
    image_bgr = cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR)

    image_name = f"tile_{row}_{col}"
    return {
        "image": image_bgr,
        "image_name": image_name,
        "row": row,
        "col": col,
        "x": x,
        "y": y,
    }


# Legacy Step 2 single-cell completeness filtering
def is_complete_single_cell(xyxy, image_shape, border_threshold=5):
    """Return whether a single-cell detection stays inside the patch border."""
    x1, y1, x2, y2 = map(int, xyxy)
    img_h, img_w = image_shape[:2]

    if x1 < border_threshold:          return False
    if y1 < border_threshold:          return False
    if x2 > img_w - border_threshold:  return False
    if y2 > img_h - border_threshold:  return False

    return True


# Legacy Step 2 cluster completeness filtering
def is_complete_cluster(
    xyxy,
    image_shape,
    border_threshold=5,
    area_ratio_max=0.20,
    aspect_ratio_max=3.0
):
    x1, y1, x2, y2 = map(int, xyxy)
    img_h, img_w = image_shape[:2]

    touches_border = (
        x1 < border_threshold or
        y1 < border_threshold or
        x2 > img_w - border_threshold or
        y2 > img_h - border_threshold
    )

    if not touches_border:
        return True

    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h
    patch_area = img_h * img_w
    area_ratio = box_area / patch_area
    aspect_ratio = max(box_w, box_h) / max(min(box_w, box_h), 1)

    if area_ratio < area_ratio_max or aspect_ratio > aspect_ratio_max:
        return False

    return True


# Filter incomplete detections
def filter_incomplete_detections(
    detections,
    class_names,
    image_shape,
    border_threshold=5,
    cluster_area_ratio_max=0.20,
    cluster_aspect_ratio_max=3.0
):
    if len(detections) == 0:
        return detections, 0, 0

    valid_mask = []
    skipped_single_cell_count = 0
    skipped_cluster_count = 0

    for xyxy, class_id in zip(detections.xyxy, detections.class_id):
        class_name = class_names[class_id]

        if class_name == "single_cell":
            if not is_complete_single_cell(xyxy, image_shape, border_threshold):
                valid_mask.append(False)
                skipped_single_cell_count += 1
                continue

        elif class_name == "cluster":
            if not is_complete_cluster(
                xyxy,
                image_shape,
                border_threshold=border_threshold,
                area_ratio_max=cluster_area_ratio_max,
                aspect_ratio_max=cluster_aspect_ratio_max
            ):
                valid_mask.append(False)
                skipped_cluster_count += 1
                continue

        valid_mask.append(True)

    valid_mask = np.array(valid_mask, dtype=bool)
    filtered_detections = detections[valid_mask]

    return filtered_detections, skipped_single_cell_count, skipped_cluster_count


# JSON template compatible with the original Step 2 format
def make_json_template(image_name, image_ext, image_shape, model_path, description_suffix=""):
    return {
        "version": "3.0.3",
        "flags": {},
        "shapes": [],
        "imagePath": f"{image_name}{image_ext}",
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1],
        "description": (
            f"Detected by YOLOv12 model, "
            f"incomplete single_cell & cluster removed: {model_path}"
            + description_suffix
        )
    }


# Process and save results for one patch
def process_and_save_one_patch(
    image,
    image_name,
    sample_output_dir,
    model_path,
    result,
    class_names,
    box_annotator,
    label_annotator,
    save_conf,
    border_threshold,
    crop_padding,
    cluster_area_ratio_max,
    cluster_aspect_ratio_max,
    save_json=True,
    save_annotation=True,
):
    image_ext   = ".png"
    image_shape = image.shape

    sample_annotation_dir = os.path.join(sample_output_dir, "annotation")
    sample_json_dir       = os.path.join(sample_output_dir, "json")
    conf_dir              = os.path.join(sample_output_dir, f"confidence_{save_conf}")

    if save_annotation:
        os.makedirs(sample_annotation_dir, exist_ok=True)
    if save_json:
        os.makedirs(sample_json_dir, exist_ok=True)
    for class_name_val in class_names.values():
        os.makedirs(os.path.join(conf_dir, class_name_val), exist_ok=True)

    detections = sv.Detections.from_ultralytics(result)

    stats_record = {
        "image_name":                                image_name,
        "total_detections_raw":                      0,
        "total_detections_after_filter":             0,
        f"detections_conf_{save_conf}_raw":          0,
        f"detections_conf_{save_conf}_after_filter": 0,
        "skipped_incomplete_single_cell":            0,
        "skipped_incomplete_cluster":                0,
    }

    # Handle patches with no detections
    if len(detections) == 0:
        if save_json:
            json_empty = make_json_template(image_name, image_ext, image_shape, model_path)
            with open(os.path.join(sample_json_dir, f"{image_name}.json"), "w", encoding="utf-8") as f:
                json.dump(json_empty, f, indent=2, ensure_ascii=False)
            json_conf_empty = make_json_template(
                image_name, image_ext, image_shape, model_path,
                f", conf>={save_conf}, no detections"
            )
            with open(os.path.join(sample_json_dir, f"{image_name}_conf{save_conf}.json"), "w", encoding="utf-8") as f:
                json.dump(json_conf_empty, f, indent=2, ensure_ascii=False)
        if save_annotation:
            cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_all_annotated.png"), image)
            cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_conf{save_conf}_annotated.png"), image)
        return stats_record

    # Completeness filtering
    filtered_detections, skipped_sc_count, skipped_cl_count = filter_incomplete_detections(
        detections=detections,
        class_names=class_names,
        image_shape=image_shape,
        border_threshold=border_threshold,
        cluster_area_ratio_max=cluster_area_ratio_max,
        cluster_aspect_ratio_max=cluster_aspect_ratio_max,
    )

    raw_total      = len(detections)
    raw_conf_count = int(np.sum(detections.confidence >= save_conf))
    filtered_total = len(filtered_detections)
    if filtered_total == 0:
        stats_record.update({
            "total_detections_raw":                      raw_total,
            "total_detections_after_filter":             0,
            f"detections_conf_{save_conf}_raw":          raw_conf_count,
            f"detections_conf_{save_conf}_after_filter": 0,
            "skipped_incomplete_single_cell":            skipped_sc_count,
            "skipped_incomplete_cluster":                skipped_cl_count,
        })
        if save_json:
            json_empty = make_json_template(image_name, image_ext, image_shape, model_path)
            with open(os.path.join(sample_json_dir, f"{image_name}.json"), "w", encoding="utf-8") as f:
                json.dump(json_empty, f, indent=2, ensure_ascii=False)
            json_conf_empty = make_json_template(
                image_name, image_ext, image_shape, model_path,
                f", conf>={save_conf}, no detections after filter"
            )
            with open(os.path.join(sample_json_dir, f"{image_name}_conf{save_conf}.json"), "w", encoding="utf-8") as f:
                json.dump(json_conf_empty, f, indent=2, ensure_ascii=False)
        if save_annotation:
            cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_all_annotated.png"), image)
            cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_conf{save_conf}_annotated.png"), image)
        return stats_record

    # Initialize JSON templates
    json_data_all  = make_json_template(image_name, image_ext, image_shape, model_path)
    json_data_conf = make_json_template(
        image_name, image_ext, image_shape, model_path,
        f", conf>={save_conf}"
    )

    json_all_i  = 0
    crop_conf_i = 0

    for xyxy, class_id, confidence in zip(
        filtered_detections.xyxy,
        filtered_detections.class_id,
        filtered_detections.confidence,
    ):
        x1, y1, x2, y2 = map(float, xyxy)
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        class_name = class_names[class_id]

        x1_pad = max(0,              x1i - crop_padding)
        y1_pad = max(0,              y1i - crop_padding)
        x2_pad = min(image_shape[1], x2i + crop_padding)
        y2_pad = min(image_shape[0], y2i + crop_padding)

        shape_base = {
            "kie_linking": [],
            "label":       class_name,
            "score":       float(confidence),
            "points": [
                [x1_pad, y1_pad],
                [x2_pad, y1_pad],
                [x2_pad, y2_pad],
                [x1_pad, y2_pad],
            ],
            "group_id":    None,
            "description": f"Confidence: {confidence:.4f}",
            "difficult":   False,
            "shape_type":  "rectangle",
            "flags":       {},
        }

        # Write all-detection JSON
        if save_json:
            shape_all = {
                **shape_base,
                "attributes": {
                    "confidence":   float(confidence),
                    "class_id":     int(class_id),
                    "detection_id": json_all_i,
                    "filtered":     False,
                },
            }
            json_data_all["shapes"].append(shape_all)
            json_all_i += 1

        if confidence < save_conf:
            continue

        if save_json:
            shape_conf = {
                **shape_base,
                "attributes": {
                    "confidence":   float(confidence),
                    "class_id":     int(class_id),
                    "detection_id": crop_conf_i,
                    "filtered":     False,
                },
            }
            json_data_conf["shapes"].append(shape_conf)

        SAVE_CROP_CLASSES = {"single_cell", "cluster"}
        if class_name in SAVE_CROP_CLASSES:
            cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
            cv2.imwrite(
                os.path.join(
                    conf_dir, class_name,
                    f"{image_name}_{crop_conf_i}_conf{confidence:.2f}.png"
                ),
                cropped
            )
        crop_conf_i += 1

    # Save JSON files
    if save_json:
        with open(os.path.join(sample_json_dir, f"{image_name}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_all, f, indent=2, ensure_ascii=False)
        with open(os.path.join(sample_json_dir, f"{image_name}_conf{save_conf}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_conf, f, indent=2, ensure_ascii=False)

    # Save annotation images for all detections
    if save_annotation:
        annotated_all = image.copy()
        labels_all = [
            f"{class_names[cid]} {cf:.2f}"
            for cid, cf in zip(filtered_detections.class_id, filtered_detections.confidence)
        ]
        annotated_all = box_annotator.annotate(scene=annotated_all, detections=filtered_detections)
        annotated_all = label_annotator.annotate(
            scene=annotated_all, detections=filtered_detections, labels=labels_all
        )
        cv2.imwrite(
            os.path.join(sample_annotation_dir, f"{image_name}_all_annotated.png"),
            annotated_all
        )

        # conf annotation
        detections_conf = filtered_detections[filtered_detections.confidence >= save_conf]
        if len(detections_conf) > 0:
            annotated_conf = image.copy()            
            labels_conf = [
                f"{class_names[cid]} {cf:.2f}"
                for cid, cf in zip(detections_conf.class_id, detections_conf.confidence)
            ]
            annotated_conf = box_annotator.annotate(scene=annotated_conf, detections=detections_conf)
            annotated_conf = label_annotator.annotate(
                scene=annotated_conf, detections=detections_conf, labels=labels_conf
            )
        else:
            annotated_conf = image.copy()
        cv2.imwrite(
            os.path.join(sample_annotation_dir, f"{image_name}_conf{save_conf}_annotated.png"),
            annotated_conf
        )

    # Update the statistics record
    stats_record.update({
        "total_detections_raw":                      raw_total,
        "total_detections_after_filter":             filtered_total,
        f"detections_conf_{save_conf}_raw":          raw_conf_count,
        f"detections_conf_{save_conf}_after_filter": crop_conf_i,   # Use crop_conf_i directly for the filtered confidence count
        "skipped_incomplete_single_cell":            skipped_sc_count,
        "skipped_incomplete_cluster":                skipped_cl_count,
    })

    return stats_record


# After inference, summarize confidence-file counts for each patient
# Preserve the statistics logic used by the updated urine Step 2
def build_confidence_statistics(output_dir, save_conf):
    conf_folder_name = f"confidence_{save_conf}"
    records = []

    for dirpath, dirnames, _ in os.walk(output_dir):
        if os.path.basename(dirpath) != conf_folder_name:
            continue

        rel_path = os.path.relpath(dirpath, output_dir)
        patient_id = os.path.dirname(rel_path)
        if patient_id in ("", "."):
            patient_id = "(root)"

        record = {"patient_id": patient_id}

        try:
            class_dirs = sorted([
                d for d in os.listdir(dirpath)
                if os.path.isdir(os.path.join(dirpath, d))
            ])
        except PermissionError:
            print(f"  Warning: permission denied: {dirpath}")
            continue

        if not class_dirs:
            print(f"  Warning: no class subdirectories under {rel_path}; skipping")
            continue

        for cls in class_dirs:
            cls_path = os.path.join(dirpath, cls)
            try:
                file_count = len([
                    f for f in os.listdir(cls_path)
                    if os.path.isfile(os.path.join(cls_path, f))
                ])
            except PermissionError:
                print(f"  Warning: permission denied: {cls_path}")
                file_count = -1
            record[cls] = file_count

        records.append(record)

    if not records:
        print(f"  Warning: no {conf_folder_name} directories under {output_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    class_cols = sorted([c for c in df.columns if c != "patient_id"])
    df = df[["patient_id"] + class_cols]

    for col in class_cols:
        df[col] = df[col].fillna(0).astype(int)

    df = df.sort_values("patient_id").reset_index(drop=True)

    sum_row = {"patient_id": "Total"}
    for col in class_cols:
        sum_row[col] = df[col].sum()

    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
    return df


# Main workflow
def main():
    args = parse_args()

    svs_path = args.input_file
    sample_name = os.path.splitext(os.path.basename(svs_path))[0]
    sample_output_dir = os.path.join(args.output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    print("============================================================")
    print("  Step1+Step2")
    print(f"  SVS file         : {svs_path}")
    print(f"  Sample name      : {sample_name}")
    print(f"  Output directory : {sample_output_dir}")
    print(f"  patch size     : {args.size}")
    print(f"  overlap        : {args.overlap}")
    print(f"  grid_size      : {args.grid_size}")
    print(f"  read_workers   : {args.read_workers}")
    print(f"  YOLO batch     : {args.yolo_batch_size}")
    print(f"  imgsz          : {args.imgsz}")
    print(f"  half           : {args.half}")
    print(f"  save_json      : {args.save_json}")
    print(f"  save_annotation: {args.save_annotation}")
    print(f"  skip_background: {args.skip_background}")
    print("============================================================")

    total_start = time.time()

    # Read slide dimensions once for metadata
    slide = openslide.OpenSlide(svs_path)
    width, height = slide.dimensions
    slide.close()

    coords, num_rows, num_cols, grid_rows, grid_cols = generate_center_patch_coords(
        width=width,
        height=height,
        patch_size=args.size,
        overlap_rate=args.overlap,
        grid_size=args.grid_size
    )

    print(f"Slide dimensions: {width} x {height}")
    print(f"Original grid would be {num_rows} x {num_cols} = {num_rows * num_cols} patches")
    print(f"Will process center patches: approximately {grid_rows} x {grid_cols} = {len(coords)} patches")

    # Load the YOLO model
    model = YOLO(args.model_path)
    class_names = model.names
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    stats_records = []
    skipped_bg = 0

    # Read patches with a thread pool and run batched YOLO inference on the main thread
    batch_items = []

    def flush_batch(items):
        if not items:
            return

        images = [it["image"] for it in items]

        # Keep FP32 as the default for consistency with the legacy implementation
        results = model(
            images,
            verbose=False,
            conf=args.infer_conf,
            iou=args.infer_iou,
            imgsz=args.imgsz,
            half=args.half
        )

        for item, result in zip(items, results):
            rec = process_and_save_one_patch(
                image=item["image"],
                image_name=item["image_name"],
                sample_output_dir=sample_output_dir,
                model_path=args.model_path,
                result=result,
                class_names=class_names,
                box_annotator=box_annotator,
                label_annotator=label_annotator,
                save_conf=args.save_conf,
                border_threshold=args.border_threshold,
                crop_padding=args.crop_padding,
                cluster_area_ratio_max=args.cluster_area_ratio_max,
                cluster_aspect_ratio_max=args.cluster_aspect_ratio_max,
                save_json=args.save_json,
                save_annotation=args.save_annotation,
            )
            stats_records.append(rec)         

    with ThreadPoolExecutor(max_workers=args.read_workers) as executor:
        # Submit futures in coordinate order
        futures = [
            executor.submit(read_patch_worker, svs_path, coord, args.size)
            for coord in coords
        ]

        # Iterate in submission order rather than completion order
        for future in tqdm(futures, total=len(futures), desc="OpenSlide reading + YOLO inference"):
            try:
                item = future.result()   # Handle patch-read failures and continue
            except Exception as e:
                print(f"  Warning: failed to read patch: {e}")
                continue

            if args.skip_background and is_background_patch_bgr(
                item["image"],
                sat_thr=args.bg_sat_thr, val_thr=args.bg_val_thr, ratio_thr=args.bg_ratio_thr
            ):
                skipped_bg += 1
                continue

            batch_items.append(item)

            if len(batch_items) >= args.yolo_batch_size:
                flush_batch(batch_items)
                batch_items = []


    # Flush the remaining partial batch
    flush_batch(batch_items)

    # Save patch-level statistics in the legacy-compatible format
    if stats_records:
        stats_df = pd.DataFrame(stats_records)        
        stats_path = os.path.join(sample_output_dir, "detection_statistics.csv")
        stats_df.to_csv(stats_path, index=False)        

        total_images = len(stats_df)
        total_raw = stats_df["total_detections_raw"].sum()
        total_filtered = stats_df["total_detections_after_filter"].sum()
        total_conf_raw = stats_df[f"detections_conf_{args.save_conf}_raw"].sum()
        total_conf_filtered = stats_df[f"detections_conf_{args.save_conf}_after_filter"].sum()
        total_skipped_sc = stats_df["skipped_incomplete_single_cell"].sum()
        total_skipped_cl = stats_df["skipped_incomplete_cluster"].sum()

        ratio_raw = (total_conf_raw / total_raw * 100) if total_raw > 0 else 0.0
        ratio_filter = (total_conf_filtered / total_filtered * 100) if total_filtered > 0 else 0.0

        print("\nDetection statistics:")
        print(f"- Processed patches                         : {total_images}")
        print(f"- Skipped background patches                : {skipped_bg}")
        print(f"- Raw total detections                      : {total_raw}")
        print(f"- Filtered total detections                 : {total_filtered}")
        print(f"- Raw detections with conf >= {args.save_conf}       : {total_conf_raw} ({ratio_raw:.1f}%)")
        print(f"- Filtered detections with conf >= {args.save_conf}  : {total_conf_filtered} ({ratio_filter:.1f}%)")
        print(f"- Removed incomplete single_cell detections : {total_skipped_sc}")
        print(f"- Removed incomplete cluster detections     : {total_skipped_cl}")
        print(f"- Patch-level statistics saved to           : {stats_path}")

    # Save per-patient confidence statistics
    conf_df = build_confidence_statistics(args.output_dir, args.save_conf)
    if not conf_df.empty:
        conf_stats_csv = os.path.join(args.output_dir, f"confidence_{args.save_conf}_patient_statistics.csv")
        conf_df.to_csv(conf_stats_csv, index=False)
        print(f"- Patient-level confidence statistics saved to: {conf_stats_csv}")

    elapsed = time.time() - total_start
    print("\nProcessing complete!")
    print(f"Results saved to: {sample_output_dir}")
    print(f"Total elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
