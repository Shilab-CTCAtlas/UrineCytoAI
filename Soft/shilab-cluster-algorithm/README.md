# ShiLab Cross-Cluster Matching

`shilab-cluster-algorithm` removes false-positive candidate cells by matching candidate-image clusters to malignant and benign reference clusters. It produces `matched_malignant_cells/`, the input consumed by the cancer-classifier inference workflow.

## Installation

```bash
cd path/to/shilab-cluster-algorithm
pip install -e .
```

The editable install reads the complete runtime dependency list from
`requirements.txt`. Install a PyTorch build compatible with the local CUDA
driver before running the workflow on a GPU.

## Primary interface

Use the Python API from a project-specific Step-4 wrapper, which should provide reference paths, model weights, normalization statistics, the relevant malignant reference-cluster IDs, and matching rules:

```python
from cross_cluster_matching.application.pipeline import run_pipeline

run_pipeline(
    positive_folder_path="/path/to/candidate_positive_folders",
    negative_folder_path="/path/to/candidate_negative_folders",
    reference_cache_dir="/path/to/reference_cache",
    model_path="/path/to/feature_extractor.pth",
    base_save_dir="/path/to/step4_results",
    malignant_ref_clusters=[0],
    matching_rules=[{
        "target": 0,
        "avoid": [1],
        "threshold_target": 0.5,
        "threshold_ratio": 3.0,
        "threshold_avoid_others": None,
    }],
)
```

The command-line interface is available through:

```bash
python -m cross_cluster_matching.application.pipeline --help
```

When `--reference_cache_dir` is not supplied, both `--malignant_cells_dir` and `--benign_cells_dir` are required. Save and reuse the generated reference cache for a fixed reference cohort whenever possible.

## Outputs

For each patient, the pipeline writes cluster diagnostics and `matched_malignant_cells/`. The latter contains only candidate images that pass the configured matching rules. It is the designated input for downstream cancer-type inference.

## Reproducibility

Keep the reference cache, feature-extractor weight checksum, normalization values, PCA dimension, matching rules, and package version together with each analysis result. Do not treat reference-cluster indices or thresholds as universal constants: they are specific to the reference cohort and model.
