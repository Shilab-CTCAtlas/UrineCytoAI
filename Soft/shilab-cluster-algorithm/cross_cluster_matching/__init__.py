"""
Cross-domain Cluster Matching Pipeline

A comprehensive pipeline for analyzing cell images through cross-domain cluster matching.
This package provides tools for data processing, feature extraction, clustering analysis,
and visualization of cell image data.

Modules:
- clustering: Clustering algorithms and optimization functions
- data: Image data processing utilities
- models: Feature extraction models and model classes
- visualization: Visualization functions for cluster analysis
- application: Main implementation functions and pipeline orchestration

Usage:
    from cross_cluster_matching.application.pipeline import run_pipeline
"""

# Package version
__version__ = "1.0.0"

__all__ = ["run_pipeline"]


def __getattr__(name):
    """Load the full pipeline only when its public entry point is requested."""
    if name == "run_pipeline":
        from .application.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
