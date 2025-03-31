"""Neighborhood embedding feature generator.

This module provides functionality for generating embeddings for NYC neighborhoods.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class NeighborhoodEmbedding:
    """Generate embeddings for NYC neighborhoods based on various features."""
    
    def __init__(
        self,
        n_components: int = 10,
        data_dir: str = "data/interim",
        random_state: int = 42,
    ):
        """Initialize the neighborhood embedding generator.
        
        Args:
            n_components: Number of embedding dimensions
            data_dir: Directory to save/load embedding data
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        
        # NYC neighborhood metadata
        self.neighborhood_data = {}
        self.neighborhood_mapping = {}
        
        logger.info(f"Initialized NeighborhoodEmbedding with {n_components} dimensions")
    
    def _preprocess_neighborhoods(self, df: pd.DataFrame) -> List[str]:
        """Preprocess and standardize neighborhood names.
        
        Args:
            df: DataFrame with a 'neighborhood' column
        
        Returns:
            List of standardized neighborhood names
        """
        # Standardize neighborhood names
        neighborhoods = []
        
        for hood in df["neighborhood"].fillna("").unique():
            if not hood:
                continue
            
            # Clean and standardize the neighborhood name
            clean_hood = hood.lower().strip()
            
            # Remove common prefixes/suffixes
            clean_hood = clean_hood.replace("new york", "").strip()
            clean_hood = clean_hood.replace("ny", "").strip()
            clean_hood = clean_hood.replace("nyc", "").strip()
            clean_hood = clean_hood.replace(",", "").strip()
            
            # Map to standard neighborhood names
            if clean_hood in self.neighborhood_mapping:
                standardized = self.neighborhood_mapping[clean_hood]
            else:
                standardized = clean_hood
                self.neighborhood_mapping[clean_hood] = standardized
            
            neighborhoods.append(standardized)
        
        return sorted(list(set(neighborhoods)))
    
    def _load_neighborhood_metadata(self) -> None:
        """Load or initialize neighborhood metadata.
        
        This could include data like:
        - Geographic coordinates
        - Demographics
        - Crime rates
        - School ratings
        - Transit accessibility
        """
        metadata_file = self.data_dir / "neighborhood_metadata.csv"
        
        if metadata_file.exists():
            # Load existing metadata
            metadata_df = pd.read_csv(metadata_file)
            self.neighborhood_data = metadata_df.set_index("neighborhood").to_dict("index")
            logger.info(f"Loaded metadata for {len(self.neighborhood_data)} neighborhoods")
        else:
            # Initialize with sample data for key NYC neighborhoods
            # In a real implementation, this would be loaded from external sources
            self.neighborhood_data = {
                "manhattan": {
                    "latitude": 40.7831,
                    "longitude": -73.9712,
                    "median_income": 85000,
                    "population_density": 70000,
                    "transit_score": 90,
                },
                "brooklyn": {
                    "latitude": 40.6782,
                    "longitude": -73.9442,
                    "median_income": 65000,
                    "population_density": 35000,
                    "transit_score": 85,
                },
                "queens": {
                    "latitude": 40.7282,
                    "longitude": -73.7949,
                    "median_income": 70000,
                    "population_density": 20000,
                    "transit_score": 75,
                },
                "bronx": {
                    "latitude": 40.8448,
                    "longitude": -73.8648,
                    "median_income": 40000,
                    "population_density": 32000,
                    "transit_score": 80,
                },
                "staten island": {
                    "latitude": 40.5795,
                    "longitude": -74.1502,
                    "median_income": 80000,
                    "population_density": 8000,
                    "transit_score": 60,
                },
            }
            
            # Add more specific neighborhoods
            # This is a simplified version for demonstration
            for borough, neighborhoods in {
                "manhattan": [
                    "upper east side", "upper west side", "midtown", "chelsea",
                    "greenwich village", "soho", "tribeca", "financial district",
                    "harlem", "east village", "west village",
                ],
                "brooklyn": [
                    "williamsburg", "park slope", "dumbo", "brooklyn heights",
                    "bushwick", "bedford-stuyvesant", "crown heights", "flatbush",
                    "greenpoint", "boerum hill", "fort greene",
                ],
                "queens": [
                    "astoria", "long island city", "flushing", "jackson heights",
                    "forest hills", "jamaica", "bayside", "ridgewood", "sunnyside",
                ],
                "bronx": [
                    "riverdale", "fordham", "concourse", "mott haven",
                    "pelham bay", "throgs neck", "kingsbridge",
                ],
                "staten island": [
                    "st. george", "todt hill", "great kills", "new dorp",
                    "port richmond", "tottenville",
                ],
            }.items():
                # Generate synthetic metadata for each neighborhood
                # In a real implementation, this would use actual data
                for neighborhood in neighborhoods:
                    # Start with borough data and add some random variation
                    np.random.seed(hash(neighborhood) % 2**32)
                    
                    self.neighborhood_data[neighborhood] = {
                        "latitude": self.neighborhood_data[borough]["latitude"] + np.random.normal(0, 0.02),
                        "longitude": self.neighborhood_data[borough]["longitude"] + np.random.normal(0, 0.02),
                        "median_income": self.neighborhood_data[borough]["median_income"] * np.random.uniform(0.8, 1.2),
                        "population_density": self.neighborhood_data[borough]["population_density"] * np.random.uniform(0.7, 1.3),
                        "transit_score": min(100, max(0, self.neighborhood_data[borough]["transit_score"] + np.random.normal(0, 10))),
                    }
            
            # Save metadata for future use
            metadata_df = pd.DataFrame.from_dict(self.neighborhood_data, orient="index")
            metadata_df.index.name = "neighborhood"
            metadata_df.reset_index().to_csv(metadata_file, index=False)
            
            logger.info(f"Initialized metadata for {len(self.neighborhood_data)} neighborhoods")
    
    def _compute_neighborhood_features(self) -> pd.DataFrame:
        """Compute features for neighborhoods based on available metadata.
        
        Returns:
            DataFrame with neighborhood features
        """
        features = []
        
        for neighborhood, data in self.neighborhood_data.items():
            # Create a feature vector for each neighborhood
            feature = {
                "neighborhood": neighborhood,
                "latitude": data.get("latitude", 0),
                "longitude": data.get("longitude", 0),
                "median_income": data.get("median_income", 0),
                "population_density": data.get("population_density", 0),
                "transit_score": data.get("transit_score", 0),
            }
            
            # Add more features as needed
            features.append(feature)
        
        return pd.DataFrame(features)
    
    def fit(self, listings_df: Optional[pd.DataFrame] = None) -> "NeighborhoodEmbedding":
        """Fit the neighborhood embedding model.
        
        Args:
            listings_df: Optional DataFrame with listings data
        
        Returns:
            Self for method chaining
        """
        # Load neighborhood metadata
        self._load_neighborhood_metadata()
        
        # If listings data is provided, extract additional information
        if listings_df is not None:
            neighborhoods = self._preprocess_neighborhoods(listings_df)
            logger.info(f"Found {len(neighborhoods)} unique neighborhoods in the data")
            
            # Compute neighborhood statistics from listings
            for neighborhood in neighborhoods:
                if neighborhood not in self.neighborhood_data:
                    # Initialize with default values
                    self.neighborhood_data[neighborhood] = {
                        "latitude": 0,
                        "longitude": 0,
                        "median_income": 0,
                        "population_density": 0,
                        "transit_score": 0,
                    }
                
                # Filter listings for this neighborhood
                hood_listings = listings_df[
                    listings_df["neighborhood"].str.lower().str.contains(neighborhood, na=False)
                ]
                
                if len(hood_listings) > 0:
                    # Update neighborhood data with statistics from listings
                    price_stats = hood_listings["price"].replace("", np.nan).dropna()
                    price_stats = pd.to_numeric(price_stats, errors="coerce")
                    
                    if len(price_stats) > 0:
                        self.neighborhood_data[neighborhood]["median_price"] = price_stats.median()
                        self.neighborhood_data[neighborhood]["mean_price"] = price_stats.mean()
                        self.neighborhood_data[neighborhood]["std_price"] = price_stats.std()
                        self.neighborhood_data[neighborhood]["listing_count"] = len(price_stats)
        
        # Compute features for each neighborhood
        features_df = self._compute_neighborhood_features()
        
        if len(features_df) == 0:
            logger.warning("No neighborhood features available for embedding")
            return self
        
        # Extract feature matrix (excluding the neighborhood name)
        X = features_df.drop("neighborhood", axis=1).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA to get embeddings
        embeddings = self.pca.fit_transform(X_scaled)
        
        # Store embeddings
        for i, neighborhood in enumerate(features_df["neighborhood"]):
            self.embeddings[neighborhood] = embeddings[i]
        
        logger.info(
            f"Generated {self.n_components}-dimensional embeddings for {len(self.embeddings)} neighborhoods"
        )
        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Save embeddings
        self.save_embeddings()
        
        return self
    
    def transform(self, neighborhood: str) -> np.ndarray:
        """Get the embedding for a specific neighborhood.
        
        Args:
            neighborhood: Name of the neighborhood
        
        Returns:
            Embedding vector for the neighborhood
        """
        # Standardize the neighborhood name
        neighborhood = neighborhood.lower().strip()
        
        # Check if we have a direct embedding
        if neighborhood in self.embeddings:
            return self.embeddings[neighborhood]
        
        # Check if we have a mapping
        if neighborhood in self.neighborhood_mapping:
            mapped = self.neighborhood_mapping[neighborhood]
            if mapped in self.embeddings:
                return self.embeddings[mapped]
        
        # If not found, return a zero vector
        logger.warning(f"No embedding found for neighborhood: {neighborhood}")
        return np.zeros(self.n_components)
    
    def save_embeddings(self, filename: Optional[str] = None) -> None:
        """Save embeddings to a file.
        
        Args:
            filename: Optional filename to save embeddings to
        """
        if filename is None:
            filename = "neighborhood_embeddings.npz"
        
        filepath = self.data_dir / filename
        
        # Convert embeddings to a format that can be saved
        embeddings_dict = {
            "neighborhoods": np.array(list(self.embeddings.keys())),
            "embeddings": np.array(list(self.embeddings.values())),
        }
        
        np.savez(filepath, **embeddings_dict)
        logger.info(f"Saved neighborhood embeddings to {filepath}")
    
    def load_embeddings(self, filename: Optional[str] = None) -> bool:
        """Load embeddings from a file.
        
        Args:
            filename: Optional filename to load embeddings from
        
        Returns:
            True if embeddings were loaded successfully, False otherwise
        """
        if filename is None:
            filename = "neighborhood_embeddings.npz"
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Embeddings file not found: {filepath}")
            return False
        
        try:
            data = np.load(filepath)
            neighborhoods = data["neighborhoods"]
            embeddings = data["embeddings"]
            
            self.embeddings = {}
            for i, neighborhood in enumerate(neighborhoods):
                self.embeddings[str(neighborhood)] = embeddings[i]
            
            logger.info(f"Loaded embeddings for {len(self.embeddings)} neighborhoods")
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return False