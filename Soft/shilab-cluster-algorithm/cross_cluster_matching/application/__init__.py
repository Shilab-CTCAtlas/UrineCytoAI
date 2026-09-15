"""Application entry points for the cross-cluster matching pipeline."""

__all__ = ["run_pipeline"]


def __getattr__(name):
    """Load the pipeline only when its public entry point is requested."""
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
