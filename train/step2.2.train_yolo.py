"""
train_yolo.py
Train a YOLOv12 model using the Ultralytics library.

Usage:
    python step2.2.train_yolo.py
"""

from ultralytics import YOLO

# ── Path config ──────────────────────────────
YAML_PATH    = r"/path/to/your/output_yolo_data/custom.yaml"
MODEL_CONFIG = "yolov12s.yaml"                           # train from scratch
# MODEL_CONFIG = "/path/to/your/pretrained/yolov12x.pt" # or use pretrained weights

# ── Training hyperparameters ──────────────────
TRAIN_ARGS = dict(
    data    = YAML_PATH,
    epochs  = 400,
    imgsz   = 1024,
    batch   = 8,
    device  = 0,
    iou     = 0.7,
    project = r"/path/to/your/results/yolo",
    name    = "train_exp01",
)


def main():
    model = YOLO(MODEL_CONFIG)
    print(f"Starting training | data: {YAML_PATH} | epochs: {TRAIN_ARGS['epochs']}")
    results = model.train(**TRAIN_ARGS)
    print(f"Training complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()