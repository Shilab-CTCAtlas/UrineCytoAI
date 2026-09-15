# UrineCytoAI

`UrineCytoAI` is a research pipeline for urine cytology whole-slide images.
It covers slide tiling, cell detection, binary benign/malignant
classification, and clustering-based false-positive removal.

## Repository layout

```text
.
├── demo/                          # Example SVS inputs
├── train/                         # Training scripts, from SVS tiling to clustering
├── Soft/
│   ├── shilab_pipeline/           # YOLO-based SVS-to-cell-image inference
│   ├── shilab-binary-classifier/  # Benign/malignant classifier package
│   └── shilab-cluster-algorithm/  # Cluster matching / false-positive removal package
├── models/
│   ├── yolo/                      # Cell detection checkpoint
│   ├── binary-classifier/         # Binary-classification checkpoints
│   └── cluster/                   # Checkpoints used by the matching workflow
├── environment.yml                # Reproducible server inference environment
└── requirements.txt               # Pinned direct pip dependencies
```

## Workflow

```text
SVS whole-slide image
  → patch extraction and YOLO detection
  → benign/malignant classification
  → cluster matching and false-positive removal
  → matched_malignant_cells/*.png
```

The training scripts are ordered by stage:

| Stage | Script | Purpose |
| --- | --- | --- |
| 1 | `train/step1.svsSplit.py` | Extract representative patches from SVS slides |
| 2.1 | `train/step2.1.prepare_dataset.py` | Convert LabelMe annotations into a YOLO dataset |
| 2.2 | `train/step2.2.train_yolo.py` | Train the cell detector |
| 3 | `train/step3.train_classifier.py` | Train the binary classifier |
| 4 | `train/step4.top4_models_clustering.py` | Build and evaluate reference-cell clustering models |

## Installation

The released end-to-end inference workflow was run on the server in a single
`patho_pipeline` environment. Create the same environment with:

```bash
conda env create -f environment.yml
conda activate patho_pipeline
```

Install the repository versions of the two local packages:

```bash
pip install -e Soft/shilab-binary-classifier
pip install -e Soft/shilab-cluster-algorithm
```

`environment.yml` is the authoritative specification for the released
inference workflow. `requirements.txt` contains the corresponding pinned
direct pip dependencies for users who manage the native libraries separately.

`openslide-python` also requires the native OpenSlide library. The Conda
environment installs OpenSlide 4.0.1; users creating a pip-only environment
must install this native library separately.

The server environment uses Python 3.10.20, PyTorch 2.2.1 with CUDA 11.8,
TorchVision 0.17.1, Ultralytics 8.3.63, NumPy 1.26.4, pandas 2.0.3,
scikit-learn 1.3.0, SciPy 1.10.1, OpenCV 4.9.0, OpenSlide 4.0.1,
openslide-python 1.4.1, UMAP-learn 0.5.7, and Supervision 0.25.1.

## Run the included demo

The repository includes `demo/malignant.svs` and `demo/benign.svs`. The
commands below use `malignant.svs`; replace `malignant` with `benign` in the
input and output paths to process the other slide. Run all commands from the
repository root.

The slides are example inputs only.

## End-to-end inference

### 1. SVS to candidate cell crops

```bash

python Soft/shilab_pipeline/application/step1_2_yolo_model_urine.py \
  --input_file demo/malignant.svs \
  --output_dir demo/run_malignant \
  --model_path models/yolo/best.pt \
  --grid_size 1 \
  --no_save_json \
  --no_save_annotation
```
The directories passed to the next stage are:

```text
demo/run_malignant/malignant/confidence_0.4/single_cell/
demo/run_malignant/malignant/confidence_0.4/cluster/
```

### 2. Binary benign/malignant classification

```bash
python -m shilab_classifier.infer.binary_infer \
  --model DenseNet161 \
  --weights models/binary-classifier/densenet161.pth \
  --input demo/run_malignant/malignant/confidence_0.4/single_cell \
  --output demo/run_malignant/binary_infer/single_cell/malignant \
  --mean 0.5920 0.5683 0.7187 \
  --std 0.2453 0.2345 0.1264

python -m shilab_classifier.infer.binary_infer \
  --model MobileNetV2 \
  --weights models/binary-classifier/mobilenetv2.pth \
  --input demo/run_malignant/malignant/confidence_0.4/cluster \
  --output demo/run_malignant/binary_infer/cluster/malignant \
  --mean 0.5884 0.5755 0.7305 \
  --std 0.2668 0.2546 0.1377
```

Each output directory contains `prediction_results.csv`, a probability plot,
and `malignant_images/` containing images retained as malignant.

### 3. Cluster matching and false-positive removal

The matching program expects each immediate child of
`positive_folder_path` or `negative_folder_path` to be a sample directory
containing `malignant_images/`. For an unlabelled demo sample, place it below
either root and supply an existing empty directory for the other root. The two
roots label rows in the summary statistics and do not change the matching
algorithm.

```bash
python -m cross_cluster_matching.application.pipeline \
  --positive_folder_path demo/run_malignant/binary_infer/single_cell \
  --negative_folder_path demo/empty_cases \
  --reference_cache_dir models/cluster/reference_cache/single_cell \
  --model_path models/cluster/densenet161.pth \
  --base_save_dir demo/run_malignant/cluster_fp_removal/single_cell \
  --model_type DenseNet161 \
  --mean 0.5920 0.5683 0.7187 \
  --std 0.2453 0.2345 0.1264 \
  --malignant_ref_clusters 8,0 \
  --rules_str "8:0:0.6:2:none@0:8:0.9:none:0.3" \
  --use_cell_level_rule_filter

python -m cross_cluster_matching.application.pipeline \
  --positive_folder_path demo/run_malignant/binary_infer/cluster \
  --negative_folder_path demo/empty_cases \
  --reference_cache_dir models/cluster/reference_cache/cluster \
  --model_path models/cluster/mobilenetv2.pth \
  --base_save_dir demo/run_malignant/cluster_fp_removal/cluster \
  --model_type MobileNetV2 \
  --pca_dim 16 --max_candidate_k 10 \
  --mean 0.5884 0.5755 0.7305 \
  --std 0.2668 0.2546 0.1377 \
  --malignant_ref_clusters 5,0 \
  --rules_str "5:0:0.85:4:none@0:5:0.85:4:none" \
  --use_cell_level_rule_filter
```

The final retained images are written to:

```text
demo/run_malignant/cluster_fp_removal/single_cell/malignant/matched_malignant_cells/
demo/run_malignant/cluster_fp_removal/cluster/malignant/matched_malignant_cells/
```

## Quick start: training

The released models were trained locally. YOLO development used Python 3.11.11,
PyTorch 2.5.1 with CUDA 12.1, and Ultralytics 8.3.63; binary-classifier and
reference-clustering development used Python 3.8.19, PyTorch 2.2.1 with CUDA
11.8, and TorchVision 0.17.1.

```bash
# 1. Extract representative SVS patches
python train/step1.svsSplit.py --input_file /path/to/svs --output_dir /path/to/patches

# 2. Prepare annotations and train the YOLO detector
python train/step2.1.prepare_dataset.py
python train/step2.2.train_yolo.py

# 3. Train the binary benign/malignant classifier
python train/step3.train_classifier.py

# 4. Build and evaluate reference-cell clustering
python train/step4.top4_models_clustering.py
```

| Script | Configure before running |
| --- | --- |
| `step2.1.prepare_dataset.py` | `input_dir`, `output_dir` |
| `step2.2.train_yolo.py` | `YAML_PATH`, `TRAIN_ARGS['project']` |
| `step3.train_classifier.py` | raw binary dataset and output paths |
| `step4.top4_models_clustering.py` | trained binary models, evaluation results, and urine reference-cell paths |

Step 3 expects a binary dataset with `benign/` and `malignant/` directories.
Step 4 selects the best fold of each top-performing model, extracts reference
features, applies PCA and KMeans, and writes the clustering results and
visualizations.

## Data, configuration, and checkpoints

The repository supplies example SVS inputs, model checkpoints, and frozen
reference caches for both single cells and cell clusters. It does not include
training source images, clinical metadata, or the original malignant and
benign reference-cell images. The inference commands above additionally
specify:

- the checkpoint architecture and normalization values shown above;
- the matching rules and reference-cluster IDs shown above; and

The caches contain only the fixed numerical representation required by the
matching algorithm; the source reference images are not required for the
released inference workflow.