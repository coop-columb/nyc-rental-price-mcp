(none – file is moved)

"""API module for NYC rental price prediction service.

This module exports API endpoints for the rental price prediction service.
"""

from src.nyc_rental_price.api.main import app, predict, health_check

__all__ = ["app", "predict", "health_check"]

"""NYC Rental Price API package.

This package provides the API for predicting NYC rental prices.
"""

from .main import app

__all__ = ["app"]

