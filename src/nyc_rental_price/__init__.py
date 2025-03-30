"""NYC Rental Price Prediction Package."""

__version__ = "0.1.0"

from .data import preprocessing
from .data.scraper import Scraper

__all__ = [
    "__version__",
    "preprocessing",
    "Scraper",
]
