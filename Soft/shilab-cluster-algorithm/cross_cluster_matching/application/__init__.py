"""
Application module for cross-cluster matching pipeline implementation.

This module contains the main implementation functions for the cross-domain
cluster matching pipeline. The pipeline orchestrates the entire workflow
from data processing to cluster matching and visualization.

Modules:
- pipeline: Main pipeline function for running the complete analysis

Note: This module is under active development. New functionality will be added
as the project evolves. Please check the documentation for updates.
"""

# Import main pipeline function for easy access
# from .qrr_pipeline import run_pipeline
from .pipeline import run_pipeline

# TODO: Add new application functions here as they are developed
# Example structure for future additions:
# from .new_module import new_function_1, new_function_2

__all__ = [
    'run_pipeline',
    ]