"""Data processing module for NYC rental price prediction.

This module exports functions for data loading, preprocessing, and cleaning.
"""

from src.nyc_rental_price.data.preprocessing import (
    preprocess_data,
    clean_data,
    handle_missing_values,
    encode_categorical_features
)
from src.nyc_rental_price.data.scraper import scrape_data

__all__ = [
    "preprocess_data",
    "clean_data",
    "handle_missing_values",
    "encode_categorical_features",
    "scrape_data"
]

"""
Data processing and acquisition module for NYC rental price prediction.

This module provides utilities for data loading, preprocessing, and web scraping
of rental listings.
"""

from .preprocessing import (
    load_data,
    handle_missing_values,
    encode_categorical_variables,
    normalize_features,
    preprocess_data,
)
from .scraper import Scraper

__all__ = [
    "load_data",
    "handle_missing_values",
    "encode_categorical_variables",
    "normalize_features",
    "preprocess_data",
    "Scraper",
]

"""
Data processing module for NYC rental price prediction.

This module contains utilities for data loading, preprocessing, cleaning,
and scraping rental listings data for the NYC rental price prediction model.
"""

from nyc_rental_price.data.preprocessing import (
    load_data,
    handle_missing_values,
    encode_categorical_variables,
    normalize_features,
    preprocess_data,
)
from nyc_rental_price.data.scraper import Scraper

__all__ = [
    "load_data",
    "handle_missing_values",
    "encode_categorical_variables",
    "normalize_features",
    "preprocess_data",
    "Scraper",
]

