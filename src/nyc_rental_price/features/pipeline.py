"""Feature pipeline for NYC rental price prediction.

This module provides a pipeline for orchestrating feature generation for rental listings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from .distance import DistanceFeatureGenerator
from .encoder import TargetEncoder
from .neighborhood import NeighborhoodEmbedding
from .temporal import TemporalFeatureGenerator
from .text import TextFeatureExtractor

logger = logging.getLogger(__name__)


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Pipeline for generating and transforming features for rental listings."""
    
    def __init__(
        self,
        data_dir: str = "data/interim",
        target_column: str = "price",
        text_column: str = "description",
        date_column: str = "posted_date",
        random_state: int = 42,
    ):
        """Initialize the feature pipeline.
        
        Args:
            data_dir: Directory to save/load feature data
            target_column: Name of the target column
            text_column: Name of the text description column
            date_column: Name of the date column
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.target_column = target_column
        self.text_column = text_column
        self.date_column = date_column
        self.random_state = random_state
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature generators
        self.neighborhood_embedding = NeighborhoodEmbedding(
            data_dir=str(self.data_dir),
            random_state=random_state,
        )
        
        self.distance_generator = DistanceFeatureGenerator(
            data_dir=str(self.data_dir),
        )
        
        self.text_extractor = TextFeatureExtractor(
            method="tfidf",
            max_features=1000,
            n_components=20,
            data_dir=str(self.data_dir),
            random_state=random_state,
        )
        
        self.temporal_generator = TemporalFeatureGenerator(
            data_dir=str(self.data_dir),
        )
        
        self.target_encoder = TargetEncoder(
            target_column=target_column,
            data_dir=str(self.data_dir),
            random_state=random_state,
        )
        
        # Initialize feature scaling
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        logger.info("Initialized FeaturePipeline")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data before feature generation.
        
        Args:
            df: Raw DataFrame with rental listings
        
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Convert price to numeric if it's not already
        if self.target_column in result_df.columns:
            if result_df[self.target_column].dtype == "object":
                # Extract numeric price from string (e.g., "$1,500/month" -> 1500)
                result_df[self.target_column] = (
                    result_df[self.target_column]
                    .str.replace(r"[^\d]", "", regex=True)
                    .replace("", np.nan)
                )
            
            # Convert to float
            result_df[self.target_column] = pd.to_numeric(
                result_df[self.target_column], errors="coerce"
            )
        
        # Handle missing values
        for col in result_df.columns:
            if result_df[col].dtype == "object":
                result_df[col] = result_df[col].fillna("")
        
        # Remove outliers in the target variable
        if self.target_column in result_df.columns:
            # Get valid price values
            valid_prices = result_df[self.target_column].dropna()
            
            if len(valid_prices) > 0:
                # Calculate price statistics
                q1 = valid_prices.quantile(0.01)
                q3 = valid_prices.quantile(0.99)
                
                # Filter out extreme outliers
                price_mask = (
                    (result_df[self.target_column] >= q1)
                    & (result_df[self.target_column] <= q3)
                )
                
                result_df = result_df[price_mask].reset_index(drop=True)
                logger.info(
                    f"Removed price outliers outside range [{q1:.2f}, {q3:.2f}], "
                    f"remaining: {len(result_df)}"
                )
        
        # Convert numeric columns
        for col in ["bedrooms", "bathrooms", "sqft"]:
            if col in result_df.columns:
                if result_df[col].dtype == "object":
                    # Extract numeric values
                    result_df[col] = (
                        result_df[col]
                        .str.replace(r"[^\d\.]", "", regex=True)
                        .replace("", np.nan)
                    )
                
                # Convert to float
                result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
        
        return result_df
    
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturePipeline":
        """Fit the feature pipeline on training data.
        
        Args:
            df: Input DataFrame with rental listings
            y: Optional target series (not used, as target is in df)
        
        Returns:
            Self for method chaining
        """
        # Preprocess the data
        processed_df = self._preprocess_data(df)
        
        if len(processed_df) == 0:
            logger.warning("No data available after preprocessing")
            return self
        
        # Extract target if available
        target = None
        if self.target_column in processed_df.columns:
            target = processed_df[self.target_column]
        
        # Fit neighborhood embeddings
        logger.info("Fitting neighborhood embeddings")
        self.neighborhood_embedding.fit(processed_df)
        
        # Fit text feature extractor
        if self.text_column in processed_df.columns:
            logger.info("Fitting text feature extractor")
            self.text_extractor.fit(processed_df, self.text_column)
        
        # Generate all features
        logger.info("Generating features for fitting")
        
        # Neighborhood features
        neighborhood_df = self._generate_neighborhood_features(processed_df)
        
        # Distance features
        distance_df = self.distance_generator.generate_features(processed_df)
        
        # Text features
        if self.text_column in processed_df.columns:
            text_df = self.text_extractor.transform(processed_df, self.text_column)
        else:
            text_df = processed_df.copy()
        
        # Temporal features
        if self.date_column in processed_df.columns:
            temporal_df = self.temporal_generator.generate_features(
                processed_df, self.date_column
            )
        else:
            temporal_df = processed_df.copy()
        
        # Combine all features
        combined_df = processed_df.copy()
        
        # Add neighborhood features
        for col in neighborhood_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = neighborhood_df[col]
        
        # Add distance features
        for col in distance_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = distance_df[col]
        
        # Add text features
        for col in text_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = text_df[col]
        
        # Add temporal features
        for col in temporal_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = temporal_df[col]
        
        # Fit target encoder if target is available
        if target is not None:
            logger.info("Fitting target encoder")
            self.target_encoder.fit(combined_df, target)
            
            # Apply target encoding
            encoded_df = self.target_encoder.transform(combined_df)
            
            # Add encoded features
            for col in encoded_df.columns:
                if col not in combined_df.columns:
                    combined_df[col] = encoded_df[col]
        
        # Identify numeric feature columns for scaling
        self.feature_columns = [
            col for col in combined_df.columns
            if col != self.target_column
            and pd.api.types.is_numeric_dtype(combined_df[col])
            and not col.startswith("is_")  # Skip boolean features
        ]
        
        logger.info(f"Identified {len(self.feature_columns)} numeric feature columns")
        
        # Fit scaler on numeric features
        if self.feature_columns:
            logger.info("Fitting feature scaler")
            self.scaler.fit(combined_df[self.feature_columns])
        
        # Save the pipeline state
        self.save_pipeline()
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted feature pipeline.
        
        Args:
            df: Input DataFrame with rental listings
        
        Returns:
            DataFrame with generated features
        """
        # Preprocess the data
        processed_df = self._preprocess_data(df)
        
        if len(processed_df) == 0:
            logger.warning("No data available after preprocessing")
            return pd.DataFrame()
        
        # Generate all features
        logger.info("Generating features for transformation")
        
        # Neighborhood features
        neighborhood_df = self._generate_neighborhood_features(processed_df)
        
        # Distance features
        distance_df = self.distance_generator.generate_features(processed_df)
        
        # Text features
        if self.text_column in processed_df.columns:
            text_df = self.text_extractor.transform(processed_df, self.text_column)
        else:
            text_df = processed_df.copy()
        
        # Temporal features
        if self.date_column in processed_df.columns:
            temporal_df = self.temporal_generator.generate_features(
                processed_df, self.date_column
            )
        else:
            temporal_df = processed_df.copy()
        
        # Combine all features
        combined_df = processed_df.copy()
        
        # Add neighborhood features
        for col in neighborhood_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = neighborhood_df[col]
        
        # Add distance features
        for col in distance_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = distance_df[col]
        
        # Add text features
        for col in text_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = text_df[col]
        
        # Add temporal features
        for col in temporal_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = temporal_df[col]
        
        # Apply target encoding
        encoded_df = self.target_encoder.transform(combined_df)
        
        # Add encoded features
        for col in encoded_df.columns:
            if col not in combined_df.columns:
                combined_df[col] = encoded_df[col]
        
        # Apply scaling to numeric features
        if self.feature_columns:
            # Get intersection of available feature columns
            available_features = [
                col for col in self.feature_columns if col in combined_df.columns
            ]
            
            if available_features:
                # Scale features
                scaled_features = self.scaler.transform(combined_df[available_features])
                
                # Update DataFrame with scaled values
                for i, col in enumerate(available_features):
                    combined_df[f"{col}_scaled"] = scaled_features[:, i]
        
        logger.info(f"Transformed data with {combined_df.shape[1]} features")
        
        return combined_df
    
    def _generate_neighborhood_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate neighborhood embedding features.
        
        Args:
            df: Input DataFrame with rental listings
        
        Returns:
            DataFrame with neighborhood features
        """
        result_df = df.copy()
        
        # Check if neighborhood column exists
        if "neighborhood" not in result_df.columns:
            logger.warning("No neighborhood column found for embedding")
            return result_df
        
        # Generate embedding for each listing
        embedding_cols = []
        for idx, listing in result_df.iterrows():
            neighborhood = listing.get("neighborhood", "")
            
            if neighborhood:
                # Get embedding vector
                embedding = self.neighborhood_embedding.transform(neighborhood)
                
                # Add embedding components to result DataFrame
                for i, value in enumerate(embedding):
                    col_name = f"neighborhood_emb_{i+1}"
                    result_df.loc[idx, col_name] = value
                    
                    if col_name not in embedding_cols:
                        embedding_cols.append(col_name)
        
        # Fill missing values with zeros
        for col in embedding_cols:
            result_df[col] = result_df[col].fillna(0)
        
        logger.info(f"Generated {len(embedding_cols)} neighborhood embedding features")
        
        return result_df
    
    def save_pipeline(self, filename: Optional[str] = None) -> None:
        """Save the pipeline state to disk.
        
        Args:
            filename: Optional filename to save pipeline to
        """
        import joblib
        
        if filename is None:
            filename = "feature_pipeline.joblib"
        
        filepath = self.data_dir / filename
        
        # Save pipeline state
        state = {
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
        }
        
        joblib.dump(state, filepath)
        logger.info(f"Saved feature pipeline state to {filepath}")
        
        # Save individual components
        self.neighborhood_embedding.save_embeddings()
        self.text_extractor.save_model()
        self.target_encoder.save_encodings()
    
    def load_pipeline(self, filename: Optional[str] = None) -> bool:
        """Load the pipeline state from disk.
        
        Args:
            filename: Optional filename to load pipeline from
        
        Returns:
            True if pipeline was loaded successfully, False otherwise
        """
        import joblib
        
        if filename is None:
            filename = "feature_pipeline.joblib"
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Pipeline state file not found: {filepath}")
            return False
        
        try:
            # Load pipeline state
            state = joblib.load(filepath)
            
            self.feature_columns = state["feature_columns"]
            self.scaler = state["scaler"]
            
            logger.info(f"Loaded feature pipeline state from {filepath}")
            
            # Load individual components
            self.neighborhood_embedding.load_embeddings()
            self.text_extractor.load_model()
            self.target_encoder.load_encodings()
            
            return True
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            return False