__all__ = ["predict_new_data"]


def __getattr__(name):
    """Load inference helpers only when requested."""
    if name == "predict_new_data":
        from .binary_infer import predict_new_data

        return predict_new_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
