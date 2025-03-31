"""Target encoder for categorical variables.

This module provides functionality for encoding categorical variables using target statistics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables using target statistics."""
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        target_column: str = "price",
        min_samples_leaf: int = 10,
        smoothing: float = 10.0,
        noise_level: float = 0.01,
        cv: int = 5,
        data_dir: str = "data/interim",
        random_state: int = 42,
    ):
        """Initialize the target encoder.
        
        Args:
            columns: List of categorical columns to encode
            target_column: Name of the target column
            min_samples_leaf: Minimum samples required for a separate category level
            smoothing: Smoothing factor for regularization
            noise_level: Level of noise to add during training
            cv: Number of cross-validation folds
            data_dir: Directory to save/load encoding data
            random_state: Random seed for reproducibility
        """
        self.columns = columns
        self.target_column = target_column
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.noise_level = noise_level
        self.cv = cv
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoding mappings
        self.encoding_mappings = {}
        self.global_mean = None
        
        logger.info(
            f"Initialized TargetEncoder with smoothing={smoothing}, "
            f"min_samples_leaf={min_samples_leaf}, cv={cv}"
        )
    
    def _encode_column(
        self, X: pd.DataFrame, y: pd.Series, column: str, is_train: bool = True
    ) -> pd.Series:
        """Encode a single categorical column.
        
        Args:
            X: Input DataFrame
            y: Target series
            column: Column to encode
            is_train: Whether this is training data
        
        Returns:
            Series with encoded values
        """
        if is_train:
            # Initialize encoding mapping for this column
            self.encoding_mappings[column] = {}
            
            # Calculate global mean
            self.global_mean = y.mean()
            
            # Calculate target statistics for each category
            category_stats = (
                pd.DataFrame({column: X[column], "target": y})
                .groupby(column)["target"]
                .agg(["mean", "count"])
            )
            
            # Apply smoothing
            smoothed_means = (
                category_stats["count"] * category_stats["mean"]
                + self.smoothing * self.global_mean
            ) / (category_stats["count"] + self.smoothing)
            
            # Create mapping
            for category, mean in smoothed_means.items():
                self.encoding_mappings[column][category] = mean
            
            # For training, use cross-validation to avoid target leakage
            encoded_series = pd.Series(index=X.index, dtype=float)
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            
            for train_idx, val_idx in kf.split(X):
                # Get train/validation splits
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                # Calculate means for this fold
                fold_means = (
                    pd.DataFrame({column: X_train[column], "target": y_train})
                    .groupby(column)["target"]
                    .agg(["mean", "count"])
                )
                
                # Apply smoothing
                fold_smoothed_means = (
                    fold_means["count"] * fold_means["mean"]
                    + self.smoothing * self.global_mean
                ) / (fold_means["count"] + self.smoothing)
                
                # Map validation data
                for idx in val_idx:
                    category = X.iloc[idx][column]
                    if category in fold_smoothed_means:
                        encoded_value = fold_smoothed_means[category]
                    else:
                        encoded_value = self.global_mean
                    
                    # Add noise during training
                    if self.noise_level > 0:
                        noise = np.random.normal(0, self.noise_level * encoded_value)
                        encoded_value += noise
                    
                    encoded_series.iloc[idx] = encoded_value
            
            return encoded_series
        else:
            # For test data, use the precomputed mappings
            encoded_series = pd.Series(index=X.index, dtype=float)
            
            for idx, category in enumerate(X[column]):
                if category in self.encoding_mappings[column]:
                    encoded_series.iloc[idx] = self.encoding_mappings[column][category]
                else:
                    encoded_series.iloc[idx] = self.global_mean
            
            return encoded_series
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """Fit the target encoder on training data.
        
        Args:
            X: Input DataFrame
            y: Target series
        
        Returns:
            Self for method chaining
        """
        # Determine columns to encode if not specified
        if self.columns is None:
            self.columns = [
                col for col in X.columns if X[col].dtype == "object" or X[col].dtype == "category"
            ]
        
        if not self.columns:
            logger.warning("No categorical columns found for encoding")
            return self
        
        logger.info(f"Fitting target encoder on {len(self.columns)} columns")
        
        # Fit each column
        for column in self.columns:
            if column in X.columns:
                self._encode_column(X, y, column, is_train=True)
            else:
                logger.warning(f"Column {column} not found in input data")
        
        # Save encodings
        self.save_encodings()
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Transform categorical columns using target encoding.
        
        Args:
            X: Input DataFrame
            y: Optional target series (not used in transform)
        
        Returns:
            DataFrame with encoded categorical columns
        """
        X_encoded = X.copy()
        
        for column in self.columns:
            if column in X.columns and column in self.encoding_mappings:
                X_encoded[f"{column}_encoded"] = self._encode_column(
                    X, None, column, is_train=False
                )
            else:
                logger.warning(f"Column {column} not found or not fitted")
        
        logger.info(f"Transformed {len(self.columns)} columns with target encoding")
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            X: Input DataFrame
            y: Target series
        
        Returns:
            DataFrame with encoded categorical columns
        """
        self.fit(X, y)
        
        X_encoded = X.copy()
        
        for column in self.columns:
            if column in X.columns:
                X_encoded[f"{column}_encoded"] = self._encode_column(
                    X, y, column, is_train=True
                )
        
        return X_encoded
    
    def save_encodings(self, filename: Optional[str] = None) -> None:
        """Save encoding mappings to disk.
        
        Args:
            filename: Optional filename to save encodings to
        """
        import json
        
        if filename is None:
            filename = "target_encodings.json"
        
        filepath = self.data_dir / filename
        
        # Convert mappings to serializable format
        serializable_mappings = {}
        for column, mapping in self.encoding_mappings.items():
            serializable_mappings[column] = {
                str(k): float(v) for k, v in mapping.items()
            }
        
        # Add global mean
        if self.global_mean is not None:
            serializable_mappings["_global_mean"] = float(self.global_mean)
        
        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(serializable_mappings, f, indent=2)
        
        logger.info(f"Saved target encodings to {filepath}")
    
    def load_encodings(self, filename: Optional[str] = None) -> bool:
        """Load encoding mappings from disk.
        
        Args:
            filename: Optional filename to load encodings from
        
        Returns:
            True if encodings were loaded successfully, False otherwise
        """
        import json
        
        if filename is None:
            filename = "target_encodings.json"
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Encodings file not found: {filepath}")
            return False
        
        try:
            with open(filepath, "r") as f:
                serialized_mappings = json.load(f)
            
            # Extract global mean
            if "_global_mean" in serialized_mappings:
                self.global_mean = float(serialized_mappings["_global_mean"])
                del serialized_mappings["_global_mean"]
            
            # Convert mappings back to original format
            self.encoding_mappings = {}
            for column, mapping in serialized_mappings.items():
                self.encoding_mappings[column] = {
                    k: float(v) for k, v in mapping.items()
                }
            
            # Update columns list
            self.columns = list(self.encoding_mappings.keys())
            
            logger.info(f"Loaded target encodings for {len(self.columns)} columns")
            return True
        except Exception as e:
            logger.error(f"Error loading encodings: {str(e)}")
            return False