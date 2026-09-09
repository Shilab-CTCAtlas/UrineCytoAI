# ShiLab Binary Classifier

`shilab-binary-classifier` provides reproducible utilities for binary classification of cytology cell images. It supports dataset splitting, cross-validation preparation, model training and evaluation, and inference on unlabelled images.

## Installation

```bash
cd path/to/shilab-binary-classifier
pip install -e .
```

Install the package's declared runtime requirements with `pip install -r requirements.txt` when editable installation is not used. Use a PyTorch build compatible with your CUDA driver when GPU inference or training is required.

## Data layout

For splitting and training, arrange input images as follows:

```text
raw_data/
├── benign/
└── malignant/
```

Images may be PNG, JPG, JPEG, BMP, TIF, or TIFF files.

## Inference

The installed command copies malignant predictions above the selected threshold and writes `prediction_results.csv` plus summary plots to the output directory.

```bash
shilab-binary-infer \
  --model DenseNet161 \
  --weights /path/to/binary_model.pth \
  --input /path/to/cell_images \
  --output /path/to/binary_inference \
  --mean 0.485 0.456 0.406 \
  --std 0.229 0.224 0.225 \
  --threshold 0.5
```

`--mean` and `--std` must be the normalization values used for training. The input can be a flat image directory or an `ImageFolder`-style directory.

## Training

For the documented end-to-end training workflow, use the repository-level `train/` scripts and README. Model architectures available to the training API are exposed as `shilab_classifier.SUPPORTED_MODELS`.

## Reproducibility

Record the package version, model architecture, class order, random seed, image normalization, software environment, and the exact weight file used for each experiment. Do not commit patient images or model weights to source control unless their distribution has been approved.
