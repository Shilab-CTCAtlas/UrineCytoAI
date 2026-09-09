"""ShiLab research software utilities."""

__version__ = "0.3.0"
__author__ = "ShiLab"

from shilab_classifier.data.dataset_splitter import split_dataset, clear_directory
from shilab_classifier.data.cross_validation import prepare_cross_validation
from shilab_classifier.training.trainer import train_single_model_fold, train_all_models, set_seed
from shilab_classifier.models.model_definitions import create_model, SUPPORTED_MODELS
from shilab_classifier.evaluation.evaluator import evaluate_single_model, evaluate_all_models

__all__ = [
    'split_dataset',
    'clear_directory',
    'prepare_cross_validation',
    'train_single_model_fold',
    'train_all_models',
    'set_seed',
    'create_model',
    'SUPPORTED_MODELS',
    'evaluate_single_model',
    'evaluate_all_models'
]
