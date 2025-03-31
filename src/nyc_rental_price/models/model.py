"""Model implementations for NYC rental price prediction."""

__all__ = [
    "BaseModel",
    "GradientBoostingModel",
    "LightGBMModel",
    "XGBoostModel",
    "NeuralNetworkModel",
    "ModelEnsemble",
]

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up conditional imports for optional dependencies
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all rental price prediction models."""

    def __init__(self, model_name="base_model"):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        pass

    def evaluate(self, X, y):
        """Evaluate model performance."""
        y_pred = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, y_pred),
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score(y, y_pred),
        }

        return metrics

    def save_model(self, filepath):
        """Save model to disk."""
        import joblib

        joblib.dump(self.model, filepath)
        return filepath

    def load_model(self, filepath):
        """Load model from disk."""
        import joblib

        self.model = joblib.load(filepath)
        return True


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for rental price prediction."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__("gradient_boosting")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )

    def fit(self, X, y):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)


class LightGBMModel(BaseModel):
    """LightGBM model for rental price prediction."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__("lightgbm")

        if not LIGHTGBM_AVAILABLE:
            logger.warning(
                "LightGBM not available, using GradientBoostingRegressor instead"
            )
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )

    def fit(self, X, y):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost model for rental price prediction."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__("xgboost")

        if not XGBOOST_AVAILABLE:
            logger.warning(
                "XGBoost not available, using GradientBoostingRegressor instead"
            )
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )

    def fit(self, X, y):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)


class NeuralNetworkModel(BaseModel):
    """Neural Network model for rental price prediction."""

    def __init__(self, hidden_layers=[64, 32], activation="relu", learning_rate=0.001):
        super().__init__("neural_network")
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.model = None

    def _build_model(self, input_dim):
        """Build and compile the neural network."""
        try:
            import tensorflow as tf

            model = tf.keras.Sequential()

            # Input layer
            model.add(
                tf.keras.layers.Dense(
                    self.hidden_layers[0],
                    activation=self.activation,
                    input_dim=input_dim,
                )
            )

            # Hidden layers
            for units in self.hidden_layers[1:]:
                model.add(tf.keras.layers.Dense(units, activation=self.activation))
                model.add(tf.keras.layers.Dropout(0.2))

            # Output layer
            model.add(tf.keras.layers.Dense(1))

            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="mse",
                metrics=["mae"],
            )

            return model
        except ImportError:
            logger.warning(
                "TensorFlow not available, using GradientBoostingRegressor instead"
            )
            return GradientBoostingRegressor(random_state=42)

    def fit(self, X, y):
        """Fit the model to training data."""
        if self.model is None:
            self.model = self._build_model(X.shape[1])

        # Check if model is a TensorFlow model or sklearn fallback
        if hasattr(self.model, "fit_generator"):
            # TensorFlow model
            self.model.fit(
                X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0
            )
        else:
            # sklearn fallback
            self.model.fit(X, y)

        return self

    def predict(self, X):
        """Make predictions on new data."""
        return (
            self.model.predict(X).flatten()
            if hasattr(self.model, "predict_on_batch")
            else self.model.predict(X)
        )


class ModelEnsemble(BaseModel):
    """Ensemble of multiple models for rental price prediction."""

    def __init__(self, models=None):
        super().__init__("model_ensemble")
        self.models = models or []

    def add_model(self, model):
        """Add a model to the ensemble."""
        self.models.append(model)
        return self

    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """Average predictions from all models."""
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
