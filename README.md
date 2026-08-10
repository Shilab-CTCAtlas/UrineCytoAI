# UrineCytoAI

`UrineCytoAI` is a research workflow for urine cytology whole-slide images. It
contains data preparation and training scripts for cell detection and
benign/malignant classification, plus an SVS-to-YOLO streaming inference
script that extracts detected single-cell and cluster crops.

> **Research use only.** This repository is not a medical device and must not
> be used as the sole basis for clinical diagnosis or treatment decisions.

## Repository layout

```text
.
├── train/
│   ├── step1.svsSplit.py                 # SVS centre-patch extraction
│   ├── step2.1.prepare_dataset.py        # LabelMe-to-YOLO dataset conversion
│   ├── step2.2.train_yolo.py             # YOLO detector training
│   ├── step3.train_classifier.py         # Benign/malignant classifier training
│   └── step4.top4_models_clustering.py   # Top-four model clustering analysis
├── Soft/shilab_pipeline/application/
│   └── step1_2_yolo_model_urine.py       # Streaming SVS + YOLO inference
├── models/
│   ├── yolo/                             # YOLO checkpoint
│   ├── binary-classifier/                # Binary-classification checkpoints
│   └── cluster/                          # Checkpoints used for clustering analysis
└── requirements.txt
```

## Installation

Use Python 3.10 or a compatible Python version. Install a PyTorch/TorchVision
build that matches your CUDA driver first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

`openslide-python` requires the native OpenSlide library. On Windows, install
the OpenSlide binaries and add their DLL directory to `PATH` before reading SVS
files.

The Step 3 and Step 4 training scripts import the shared packages
`shilab_classifier` and `cross_cluster_matching`. They are not included in the
current UrineCytoAI directory, so install compatible copies before using these
two scripts:

```bash
pip install -e /path/to/shilab-binary-classifier
pip install -e /path/to/shilab-cluster-algorithm
```

## Inference: SVS to detected cell crops

`Soft/shilab_pipeline/application/step1_2_yolo_model_urine.py` combines centre
patch sampling and batched YOLO inference without saving intermediate full-size
patches. It filters incomplete cells at patch borders, writes LabelMe-compatible
JSON and annotation images, and saves high-confidence `single_cell` and
`cluster` crops.

```bash
python Soft/shilab_pipeline/application/step1_2_yolo_model_urine.py \
  --input_file /path/to/sample.svs \
  --output_dir /path/to/inference_output \
  --model_path models/yolo/best.pt \
  --size 1024 --overlap 0.05 --grid_size 3 \
  --read_workers 4 --yolo_batch_size 8 \
  --infer_conf 0.1 --infer_iou 0.3 --save_conf 0.4
```

The command creates `/path/to/inference_output/<sample_name>/annotation/`,
`json/`, `confidence_0.4/single_cell/`, and `confidence_0.4/cluster/`. It also
writes patch-level `detection_statistics.csv` and an output-root-level
`confidence_0.4_patient_statistics.csv`. Use `--no_save_json` or
`--no_save_annotation` when those artifacts are not required; use
`--skip_background` only after validating that it does not reduce sensitivity.

## Training workflow

All paths embedded in the training scripts use the public placeholder form
`/path/to/your/...`. Replace those constants with local paths before running
scripts that do not accept command-line path arguments.

### Step 1 — extract representative SVS patches

```bash
python train/step1.svsSplit.py \
  --input_file /path/to/svs_files \
  --output_dir /path/to/patches \
  --size 1024 --overlap 0.05 --grid_size 3 --threads 8
```

The script accepts one SVS file or a directory and extracts only the centre
patch from each `grid_size x grid_size` region.

### Step 2.1 — prepare YOLO data

Set `input_dir` and `output_dir` in `train/step2.1.prepare_dataset.py`. The
input directory must contain images and same-basename LabelMe JSON files.

```bash
python train/step2.1.prepare_dataset.py
```

It creates a 70%/20%/10% train/validation/test split, converts rectangular
annotations to YOLO labels, and writes `custom.yaml`. The class IDs are
`single_cell`, `cluster`, `impurity`, `part`, and `vague`. Missing JSON files
are created as empty annotations in the input directory.

### Step 2.2 — train the YOLO detector

Set `YAML_PATH` to the `custom.yaml` from Step 2.1 and set
`TRAIN_ARGS['project']` to an experiment-output directory in
`train/step2.2.train_yolo.py`.

```bash
python train/step2.2.train_yolo.py
```

The supplied defaults train `yolov12s.yaml` for 400 epochs at image size 1024
and batch size 8. Ultralytics saves `weights/best.pt` inside the run directory.

### Step 3 — train the benign/malignant classifier

Set `RAW_DATA_DIR` and `BASE_OUTPUT_DIR` in
`train/step3.train_classifier.py`. The input must contain:

```text
/path/to/your/raw_data/
├── benign/
└── malignant/
```

```bash
python train/step3.train_classifier.py
```

The script makes a 90% train-validation split, prepares five folds, trains the
architectures in `SUPPORTED_MODELS`, and writes `split_data/`,
`cross_validation_data/`, `train_val_models/`, and `evaluation_results/`.

### Step 4 — top-four model clustering analysis

Set `MODELS_FOLDER`, `EVALUATION_FOLDER`, `OUTPUT_BASE_DIR`,
`MALIGNANT_CELLS_DIR`, and `BENIGN_CELLS_DIR` in
`train/step4.top4_models_clustering.py`.

```bash
python train/step4.top4_models_clustering.py
```

This script reads `top4_models_summary.csv`, selects the best fold by
`test_mcc` for each top-four model, extracts reference-cell features, applies
PCA and KMeans, and saves visualizations. The malignant reference directory
may contain `LUAD/`, `LUSC/`, and `SCLC/` subfolders; the benign directory
contains benign cell images.

## Models and Git LFS

The repository includes two 102.21 MiB DenseNet161 checkpoints, which are over
GitHub's normal Git file-size limit. Git LFS is required before committing:

```bash
git lfs install
git lfs track "*.pt" "*.pth"
git add .gitattributes models/
```

The binary-classifier and cluster folders contain identical DenseNet161 weights
and identical MobileNetV2 weights. They are retained under both paths for
configuration compatibility; Git LFS stores each repeated object only once per
content hash. Do not upload raw urine images, patient identifiers, or private
reference-cell libraries.

## Before public release

- Add the shared binary-classifier and clustering packages if this repository
  is intended to run standalone.
- Add a single end-to-end runner/configuration template if you want one-command
  inference beyond the provided YOLO stage.
- Verify data, model, and third-party-code redistribution rights.
- Add a license only after every relevant rights holder agrees to its terms.
