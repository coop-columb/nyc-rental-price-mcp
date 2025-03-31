(none – file is moved)

"""Data processing module."""

from .preprocessing import (
    load_data,
    handle_missing_values,
    encode_categorical_variables,
    normalize_features,
    preprocess_data,
)

__all__ = [
    "load_data",
    "handle_missing_values",
    "encode_categorical_variables",
    "normalize_features",
    "preprocess_data",
]
