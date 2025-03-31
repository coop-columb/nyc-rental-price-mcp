"""Text feature extractor for rental listings.

This module provides functionality for extracting features from listing text descriptions.
"""

import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract features from listing text descriptions."""
    
    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 100,
        n_components: int = 10,
        data_dir: str = "data/interim",
        random_state: int = 42,
    ):
        """Initialize the text feature extractor.
        
        Args:
            method: Feature extraction method ('tfidf', 'count', or 'word2vec')
            max_features: Maximum number of features to extract
            n_components: Number of components for dimensionality reduction
            data_dir: Directory to save/load model data
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.max_features = max_features
        self.n_components = n_components
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vectorizers
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
                lowercase=True,
                analyzer="word",
                token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
                min_df=5,
                ngram_range=(1, 2),
            )
        elif method == "count":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words="english",
                lowercase=True,
                analyzer="word",
                token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
                min_df=5,
                ngram_range=(1, 2),
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Initialize dimensionality reduction
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        
        # Initialize amenity detector
        self.amenities = self._load_amenities()
        
        logger.info(
            f"Initialized TextFeatureExtractor with method={method}, "
            f"max_features={max_features}, n_components={n_components}"
        )
    
    def _load_amenities(self) -> Set[str]:
        """Load or initialize the set of common rental amenities.
        
        Returns:
            Set of amenity terms
        """
        amenities_file = self.data_dir / "amenities.txt"
        
        if amenities_file.exists():
            # Load existing amenities list
            with open(amenities_file, "r") as f:
                amenities = set(line.strip().lower() for line in f if line.strip())
            logger.info(f"Loaded {len(amenities)} amenities from {amenities_file}")
            return amenities
        
        # Initialize with common rental amenities
        amenities = {
            # Appliances
            "dishwasher", "washer", "dryer", "refrigerator", "microwave", "stove",
            "oven", "garbage disposal", "freezer", "range", "cooktop",
            
            # Features
            "hardwood", "stainless steel", "granite", "marble", "walk-in closet",
            "ceiling fan", "fireplace", "balcony", "patio", "deck", "yard",
            "garden", "pool", "jacuzzi", "hot tub", "sauna", "storage",
            
            # Utilities
            "heat", "water", "gas", "electricity", "trash", "cable", "internet",
            "wifi", "central air", "air conditioning", "ac", "heating",
            
            # Building amenities
            "elevator", "doorman", "concierge", "security", "gym", "fitness",
            "laundry", "bike storage", "garage", "parking", "roof deck",
            "rooftop", "courtyard", "lounge", "pet friendly", "cats", "dogs",
            
            # Services
            "package", "maintenance", "superintendent", "super", "valet",
            "shuttle", "cleaning", "maid", "housekeeping",
        }
        
        # Save amenities for future use
        with open(amenities_file, "w") as f:
            for amenity in sorted(amenities):
                f.write(f"{amenity}\n")
        
        logger.info(f"Initialized {len(amenities)} common amenities")
        return amenities
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for feature extraction.
        
        Args:
            text: Raw text to preprocess
        
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        
        # Remove phone numbers
        text = re.sub(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "", text)
        
        # Remove special characters and digits
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def _detect_amenities(self, text: str) -> Dict[str, bool]:
        """Detect amenities mentioned in the text.
        
        Args:
            text: Preprocessed text
        
        Returns:
            Dictionary mapping amenity names to boolean values
        """
        if not isinstance(text, str):
            return {}
        
        text = text.lower()
        result = {}
        
        for amenity in self.amenities:
            result[f"has_{amenity.replace(' ', '_')}"] = amenity in text
        
        return result
    
    def fit(self, listings_df: pd.DataFrame, text_column: str = "description") -> "TextFeatureExtractor":
        """Fit the text feature extractor on listing descriptions.
        
        Args:
            listings_df: DataFrame with listing data
            text_column: Name of the column containing text descriptions
        
        Returns:
            Self for method chaining
        """
        # Extract and preprocess text descriptions
        texts = listings_df[text_column].fillna("").apply(self._preprocess_text).values
        
        if len(texts) == 0:
            logger.warning("No text data available for fitting")
            return self
        
        # Fit the vectorizer
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Fitted vectorizer on {X.shape[0]} descriptions with {X.shape[1]} features")
        
        # Fit SVD for dimensionality reduction
        if X.shape[1] > self.n_components:
            self.svd.fit(X)
            logger.info(
                f"Fitted SVD with {self.n_components} components, "
                f"explained variance: {self.svd.explained_variance_ratio_.sum():.4f}"
            )
        
        # Save the model
        self.save_model()
        
        return self
    
    def transform(
        self, listings_df: pd.DataFrame, text_column: str = "description"
    ) -> pd.DataFrame:
        """Transform listing descriptions into features.
        
        Args:
            listings_df: DataFrame with listing data
            text_column: Name of the column containing text descriptions
        
        Returns:
            DataFrame with text features
        """
        # Create a copy of the input DataFrame
        result_df = listings_df.copy()
        
        # Extract and preprocess text descriptions
        texts = result_df[text_column].fillna("").apply(self._preprocess_text).values
        
        if len(texts) == 0:
            logger.warning("No text data available for transformation")
            return result_df
        
        # Transform texts to feature vectors
        try:
            X = self.vectorizer.transform(texts)
            
            # Apply dimensionality reduction if needed
            if hasattr(self, "svd") and X.shape[1] > self.n_components:
                X_reduced = self.svd.transform(X)
                
                # Add reduced features to the result DataFrame
                for i in range(self.n_components):
                    result_df[f"text_feature_{i+1}"] = X_reduced[:, i]
            else:
                # If SVD not applied, use the raw features
                X_dense = X.toarray()
                feature_names = self.vectorizer.get_feature_names_out()
                
                # Add top features to the result DataFrame
                for i, name in enumerate(feature_names[:self.n_components]):
                    result_df[f"text_{name}"] = X_dense[:, i]
            
            logger.info(f"Generated {self.n_components} text features")
        except Exception as e:
            logger.error(f"Error transforming text: {str(e)}")
        
        # Detect amenities in descriptions
        for idx, text in enumerate(texts):
            amenities = self._detect_amenities(text)
            for amenity, present in amenities.items():
                result_df.loc[idx, amenity] = present
        
        # Count amenities
        amenity_columns = [col for col in result_df.columns if col.startswith("has_")]
        result_df["amenity_count"] = result_df[amenity_columns].sum(axis=1)
        
        logger.info(f"Detected {len(amenity_columns)} amenities in descriptions")
        
        return result_df
    
    def save_model(self, prefix: Optional[str] = None) -> None:
        """Save the model to disk.
        
        Args:
            prefix: Optional prefix for the saved files
        """
        import joblib
        
        prefix = prefix or self.method
        vectorizer_path = self.data_dir / f"{prefix}_vectorizer.joblib"
        svd_path = self.data_dir / f"{prefix}_svd.joblib"
        
        # Save vectorizer
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save SVD if it has been fitted
        if hasattr(self, "svd") and hasattr(self.svd, "components_"):
            joblib.dump(self.svd, svd_path)
        
        logger.info(f"Saved text feature extractor model to {self.data_dir}")
    
    def load_model(self, prefix: Optional[str] = None) -> bool:
        """Load the model from disk.
        
        Args:
            prefix: Optional prefix for the saved files
        
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        import joblib
        
        prefix = prefix or self.method
        vectorizer_path = self.data_dir / f"{prefix}_vectorizer.joblib"
        svd_path = self.data_dir / f"{prefix}_svd.joblib"
        
        if not vectorizer_path.exists():
            logger.warning(f"Vectorizer file not found: {vectorizer_path}")
            return False
        
        try:
            # Load vectorizer
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load SVD if available
            if svd_path.exists():
                self.svd = joblib.load(svd_path)
            
            logger.info(f"Loaded text feature extractor model from {self.data_dir}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False