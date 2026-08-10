#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 + Step2 合并优化版：SVS → OpenSlide内存patch → YOLO batch推理 → 原Step2输出结构

目标：
  1. 不再把中间 patch 保存为 png，减少 Step1/Step2 重复 IO
  2. YOLO 支持 batch 推理，提高 GPU 利用率
  3. 保留原 step2.yolo_modele_infer_urine.py 的医学处理逻辑：
     - single_cell 完整性过滤
     - cluster 边缘残片过滤
     - save_conf 筛选
     - crop_padding 裁剪
     - JSON 保存
     - annotation 保存
     - confidence_{save_conf}/{cell_type}/ 输出结构
  4. 输出保持与原 Step2 一致，确保 Step3 可直接衔接

输出结构：
  output_dir/
    {sample_name}/
      annotation/
      json/
      confidence_{save_conf}/
        single_cell/
        cluster/
        ...

注意：
  - 默认 half=False，避免 FP16 数值差异影响推理结果
  - 默认 save_json=True, save_annotation=True，保持与原 Step2 行为一致
  - 默认不做背景过滤，避免漏检风险
"""
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


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SVS流式YOLO推理：合并Step1和Step2，避免中间patch落盘",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 路径参数
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入单个SVS文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出结果根目录，对应原Step2的 output_dir，例如 data_svsModle")
    parser.add_argument("--model_path", type=str, required=True,
                        help="YOLO模型权重路径")

    # Step1 patch 参数
    parser.add_argument("--size", type=int, default=1024,
                        help="patch大小")
    parser.add_argument("--overlap", type=float, default=0.05,
                        help="patch overlap比例")
    parser.add_argument("--grid_size", type=int, default=3,
                        help="中心采样grid大小，与原Step1一致")
    parser.add_argument("--read_workers", type=int, default=4,
                        help="OpenSlide读取线程数。每个线程独立打开OpenSlide对象")

    # YOLO 推理参数
    parser.add_argument("--infer_conf", type=float, default=0.1,
                        help="YOLO推理置信度阈值")
    parser.add_argument("--infer_iou", type=float, default=0.3,
                        help="YOLO NMS IoU阈值")
    parser.add_argument("--save_conf", type=float, default=0.4,
                        help="保存/裁剪/JSON/conf版可视化阈值")
    parser.add_argument("--yolo_batch_size", type=int, default=8,
                        help="YOLO batch size")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="YOLO推理输入尺寸，默认与patch size一致")
    parser.add_argument("--half", action="store_true",
                        help="启用FP16推理。默认关闭，以最大程度保持与原FP32结果一致")

    # 原 Step2 完整性过滤参数
    parser.add_argument("--border_threshold", type=int, default=5,
                        help="single_cell / cluster 贴边判断阈值")
    parser.add_argument("--crop_padding", type=int, default=5,
                        help="裁剪保存时四周扩展padding")
    parser.add_argument("--cluster_area_ratio_max", type=float, default=0.20,
                        help="cluster贴边残片过滤：面积占比阈值")
    parser.add_argument("--cluster_aspect_ratio_max", type=float, default=3.0,
                        help="cluster贴边残片过滤：长宽比阈值")

    # 输出控制：默认保持原Step2行为
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="保存patch级JSON。默认保存，保持与原Step2一致")
    parser.add_argument("--no_save_json", dest="save_json", action="store_false",
                        help="不保存patch级JSON，仅保存crop和统计")
    parser.add_argument("--save_annotation", action="store_true", default=True,
                        help="保存annotation可视化图。默认保存，保持与原Step2一致")
    parser.add_argument("--no_save_annotation", dest="save_annotation", action="store_false",
                        help="不保存annotation可视化图，加速IO")

    # 安全可选项：默认关闭，避免影响结果
    parser.add_argument("--skip_background", action="store_true",
                        help="跳过明显背景patch。默认关闭，避免漏检")
    parser.add_argument("--bg_sat_thr", type=int, default=15,
                        help="背景过滤HSV饱和度阈值")
    parser.add_argument("--bg_val_thr", type=int, default=230,
                        help="背景过滤HSV亮度阈值")
    parser.add_argument("--bg_ratio_thr", type=float, default=0.95,
                        help="背景像素比例超过该值则跳过patch")

    return parser.parse_args()


# ============================================================
# 与原 Step1 一致的中心patch坐标生成
# ============================================================
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


# ============================================================
# 可选背景过滤：默认不启用
# ============================================================
def is_background_patch_bgr(img_bgr, sat_thr=15, val_thr=230, ratio_thr=0.95):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bg_mask = (s < sat_thr) & (v > val_thr)
    return float(np.mean(bg_mask)) >= ratio_thr

_thread_local = threading.local()

def get_slide(slide_path):
    """每个线程只打开一次 OpenSlide，线程内复用"""
    if not hasattr(_thread_local, "slide") or _thread_local.slide_path != slide_path:
        if hasattr(_thread_local, "slide"):
            try:
                _thread_local.slide.close()
            except Exception:
                pass                        # close 失败不阻止重新 open
        _thread_local.slide = openslide.OpenSlide(slide_path)
        _thread_local.slide_path = slide_path
    return _thread_local.slide

def read_patch_worker(slide_path, coord_item, patch_size):
    x, y, row, col = coord_item
    slide = get_slide(slide_path)           # ← 线程内复用，不重复 open/close
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


# ============================================================
# 原 Step2：single_cell 完整性过滤
# ============================================================
def is_complete_single_cell(xyxy, image_shape, border_threshold=5):
    x1, y1, x2, y2 = map(int, xyxy)
    img_h, img_w = image_shape[:2]

    if x1 < border_threshold:          return False
    if y1 < border_threshold:          return False
    if x2 > img_w - border_threshold:  return False
    if y2 > img_h - border_threshold:  return False

    return True


# ============================================================
# 原 Step2：cluster 完整性过滤
# ============================================================
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


# ============================================================
# 原 Step2：过滤不完整目标
# ============================================================
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


# ============================================================
# JSON模板：保持原Step2格式
# ============================================================
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


# ============================================================
# 单patch结果处理与保存
# ============================================================
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

    # ── 无检测结果 ────────────────────────────────────────────
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

    # ── 完整性过滤 ──────────────────────────────────────────── 
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

    # ── 初始化 JSON 模板 ──────────────────────────────────────
    json_data_all  = make_json_template(image_name, image_ext, image_shape, model_path)
    json_data_conf = make_json_template(
        image_name, image_ext, image_shape, model_path,
        f", conf>={save_conf}"
    )

    # ── 连续编号计数器 ────────────────────────────────────────
    # json_all_i  : 全量 JSON detection_id（所有 conf 级别）
    # crop_conf_i : conf JSON detection_id，与裁剪文件名严格对齐（ISSUE-9 修复）
    json_all_i  = 0
    crop_conf_i = 0

    # ── 单次遍历：全量JSON + conf JSON + 裁剪保存 ────────────
    for xyxy, class_id, confidence in zip(
        filtered_detections.xyxy,
        filtered_detections.class_id,
        filtered_detections.confidence,
    ):
        x1, y1, x2, y2 = map(float, xyxy)
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        class_name = class_names[class_id]

        # padding 坐标计算一次，JSON 和裁剪共用
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

        # 写入全量 JSON
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

        # 裁剪保存（只保存 single_cell 和 cluster 两类）
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

    # ── 保存 JSON ─────────────────────────────────────────────
    if save_json:
        with open(os.path.join(sample_json_dir, f"{image_name}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_all, f, indent=2, ensure_ascii=False)
        with open(os.path.join(sample_json_dir, f"{image_name}_conf{save_conf}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_conf, f, indent=2, ensure_ascii=False)

    # ── 全量 annotation ───────────────────────────────────────
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

    # ── 更新统计记录 ──────────────────────────────────────────
    stats_record.update({
        "total_detections_raw":                      raw_total,
        "total_detections_after_filter":             filtered_total,
        f"detections_conf_{save_conf}_raw":          raw_conf_count,
        f"detections_conf_{save_conf}_after_filter": crop_conf_i,  # 直接用 crop_conf_i
        "skipped_incomplete_single_cell":            skipped_sc_count,
        "skipped_incomplete_cluster":                skipped_cl_count,
    })

    return stats_record


# ============================================================
# 推理完成后：遍历输出目录统计每个病人confidence文件数
# 保留尿液新版step2的统计逻辑
# ============================================================
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
            print(f"  ⚠️ 无权限访问: {dirpath}")
            continue

        if not class_dirs:
            print(f"  ⚠️ {rel_path} 下没有类别子文件夹，跳过")
            continue

        for cls in class_dirs:
            cls_path = os.path.join(dirpath, cls)
            try:
                file_count = len([
                    f for f in os.listdir(cls_path)
                    if os.path.isfile(os.path.join(cls_path, f))
                ])
            except PermissionError:
                print(f"  ⚠️ 无权限访问: {cls_path}")
                file_count = -1
            record[cls] = file_count

        records.append(record)

    if not records:
        print(f"  ⚠️ 未在 {output_dir} 下找到任何 '{conf_folder_name}' 文件夹")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    class_cols = sorted([c for c in df.columns if c != "patient_id"])
    df = df[["patient_id"] + class_cols]

    for col in class_cols:
        df[col] = df[col].fillna(0).astype(int)

    df = df.sort_values("patient_id").reset_index(drop=True)

    sum_row = {"patient_id": "【合计】"}
    for col in class_cols:
        sum_row[col] = df[col].sum()

    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
    return df


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    svs_path = args.input_file
    sample_name = os.path.splitext(os.path.basename(svs_path))[0]
    sample_output_dir = os.path.join(args.output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    print("============================================================")
    print("  Step1+Step2 合并流式推理启动")
    print(f"  SVS文件        : {svs_path}")
    print(f"  样本名         : {sample_name}")
    print(f"  输出目录       : {sample_output_dir}")
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

    # 读取slide尺寸，只打开一次用于取metadata
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

    # 加载YOLO
    model = YOLO(args.model_path)
    class_names = model.names
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    stats_records = []
    skipped_bg = 0

    # 读取patch采用线程池，YOLO在主线程batch推理
    batch_items = []

    def flush_batch(items):
        if not items:
            return

        images = [it["image"] for it in items]

        # half 默认False，确保与原Step2 FP32更一致
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
        # 按顺序提交，保留 future 列表（有序）
        futures = [
            executor.submit(read_patch_worker, svs_path, coord, args.size)
            for coord in coords
        ]

        # ✅ 按提交顺序迭代，而不是 as_completed
        for future in tqdm(futures, total=len(futures), desc="OpenSlide读取+YOLO推理"):
            try:
                item = future.result()   # 会等待该 future 完成
            except Exception as e:
                print(f"  ⚠️ 读取patch失败: {e}")
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


    # flush剩余batch
    flush_batch(batch_items)

    # 保存patch级统计，字段与原旧版统计兼容
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

        print("\n检测统计信息:")
        print(f"- 处理patch数                         : {total_images}")
        print(f"- 跳过背景patch数                     : {skipped_bg}")
        print(f"- 原始总检测数                        : {total_raw}")
        print(f"- 过滤后总检测数                      : {total_filtered}")
        print(f"- 原始 conf>={args.save_conf} 检测数   : {total_conf_raw} ({ratio_raw:.1f}%)")
        print(f"- 过滤后 conf>={args.save_conf} 检测数 : {total_conf_filtered} ({ratio_filter:.1f}%)")
        print(f"- 删除的不完整 single_cell 数         : {total_skipped_sc}")
        print(f"- 删除的不完整 cluster 数             : {total_skipped_cl}")
        print(f"- patch级统计保存到                   : {stats_path}")

    # 保存病人级confidence统计，和尿液新版step2逻辑保持一致
    conf_df = build_confidence_statistics(args.output_dir, args.save_conf)
    if not conf_df.empty:
        conf_stats_csv = os.path.join(args.output_dir, f"confidence_{args.save_conf}_patient_statistics.csv")
        conf_df.to_csv(conf_stats_csv, index=False)
        print(f"- 病人级confidence统计保存到           : {conf_stats_csv}")

    elapsed = time.time() - total_start
    print("\n处理完成!")
    print(f"结果已保存到: {sample_output_dir}")
    print(f"总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
