"""
Binary classification model (benign/malignant) training pipeline.
Includes dataset splitting, cross-validation preparation, training, and evaluation.

"""

import os
import shutil
import torch
from datetime import datetime
from shilab_classifier import (
    split_dataset,
    set_seed,
    prepare_cross_validation,
    train_all_models,
    evaluate_all_models,
    SUPPORTED_MODELS,
)

# ── Path config ───────────────────────────────
RAW_DATA_DIR            = r"/path/to/your/raw_data"               # raw image data root
BASE_OUTPUT_DIR         = r"/path/to/your/output"                 # experiment output root
ORIGINAL_BENIGN_TEST_DIR = r"/path/to/your/external_benign_test"  # external benign test set (optional)

# ── Experiment directory ──────────────────────
EXPERIMENT_DIR   = os.path.join(BASE_OUTPUT_DIR, "")
SPLIT_DATA_DIR   = os.path.join(EXPERIMENT_DIR, "split_data")
CV_DATA_DIR      = os.path.join(EXPERIMENT_DIR, "cross_validation_data")
MODELS_DIR       = os.path.join(EXPERIMENT_DIR, "train_val_models")
RESULTS_DIR      = os.path.join(EXPERIMENT_DIR, "evaluation_results")
TRAIN_VAL_DIR    = os.path.join(SPLIT_DATA_DIR, "train_val")
TEST_DIR         = os.path.join(SPLIT_DATA_DIR, "test")
TARGET_BENIGN_TEST_DIR = os.path.join(EXPERIMENT_DIR, "split_data", "test", "benign")

for d in [EXPERIMENT_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Training hyperparameters ──────────────────
SEED        = 42
NUM_FOLDS   = 5
NUM_EPOCHS  = 50
BATCH_SIZE  = 16
INITIAL_LR  = 0.001
PATIENCE    = 10
NUM_WORKERS = 0
NUM_CLASSES = 2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_TO_TRAIN = SUPPORTED_MODELS  # or specify a subset, e.g. ["ResNet50"]


def copy_benign_test_data(source_dir: str, target_dir: str) -> bool:
    """Copy external benign test images into the experiment test directory."""
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return False

    os.makedirs(target_dir, exist_ok=True)
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    if not files:
        print(f"No files found in {source_dir}")
        return False

    for file_name in files:
        dst = os.path.join(target_dir, file_name)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(os.path.join(source_dir, file_name), dst)

    print(f"Copied {len(files)} files from {source_dir} to {target_dir}")
    return True


def main():
    set_seed(SEED)

    # ── Step 0: Validate input directories ───────
    benign_path    = os.path.join(RAW_DATA_DIR, "benign")
    malignant_path = os.path.join(RAW_DATA_DIR, "malignant")
    for path, name in [(RAW_DATA_DIR, "raw data"), (benign_path, "benign"), (malignant_path, "malignant")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} directory not found: {path}")
    print("Input directories verified.")

    # ── Step 1: Split dataset ─────────────────────
    split_dataset(
        benign_path=benign_path,
        malignant_path=malignant_path,
        output_dir=SPLIT_DATA_DIR,
        split_ratio=0.9,
        random_seed=SEED,
    )
    print("Dataset split complete.")

    # ── Step 2: Prepare cross-validation folds ────
    print(f"Preparing {NUM_FOLDS}-fold cross-validation ...")
    prepare_cross_validation(
        data_path=TRAIN_VAL_DIR,
        output_base=CV_DATA_DIR,
        n_splits=NUM_FOLDS,
        random_state=SEED,
    )
    print("Cross-validation data ready.")

    # ── Step 3: Train models ──────────────────────
    print(f"Training {len(MODELS_TO_TRAIN)} model(s) × {NUM_FOLDS} folds "
          f"= {len(MODELS_TO_TRAIN) * NUM_FOLDS} runs | start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    train_all_models(
        model_list=MODELS_TO_TRAIN,
        cv_data_dir=CV_DATA_DIR,
        output_dir=MODELS_DIR,
        num_folds=NUM_FOLDS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        initial_lr=INITIAL_LR,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        num_workers=NUM_WORKERS,
        patience=PATIENCE,
    )
    print("All models trained.")

    # ── (Optional) Copy external benign test set ──
    # copy_benign_test_data(ORIGINAL_BENIGN_TEST_DIR, TARGET_BENIGN_TEST_DIR)

    # ── Step 4: Evaluate models ───────────────────
    print(f"Evaluating models | start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    eval_result = evaluate_all_models(
        model_list=MODELS_TO_TRAIN,
        cv_data_dir=CV_DATA_DIR,
        test_dir=TEST_DIR,
        models_dir=MODELS_DIR,
        output_dir=RESULTS_DIR,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        save_summary=True,
    )

    if eval_result:
        print(f"Evaluation complete. Total time: {eval_result['total_time']:.2f} min")
    else:
        print("Evaluation returned no results.")


if __name__ == "__main__":
    main()