"""Distance-based feature generator.

This module provides functionality for generating distance-based features for NYC rental listings.
"""

import logging
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DistanceFeatureGenerator:
    """Generate distance-based features for NYC rental listings."""
    
    def __init__(
        self,
        data_dir: str = "data/interim",
        poi_file: Optional[str] = None,
    ):
        """Initialize the distance feature generator.
        
        Args:
            data_dir: Directory to save/load POI (points of interest) data
            poi_file: Optional filename for POI data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.poi_file = poi_file or "nyc_poi.csv"
        self.poi_data = None
        
        logger.info("Initialized DistanceFeatureGenerator")
    
    def _load_poi_data(self) -> pd.DataFrame:
        """Load points of interest data.
        
        Returns:
            DataFrame with POI data
        """
        poi_path = self.data_dir / self.poi_file
        
        if poi_path.exists():
            # Load existing POI data
            self.poi_data = pd.read_csv(poi_path)
            logger.info(f"Loaded {len(self.poi_data)} points of interest")
            return self.poi_data
        
        # Generate sample POI data if file doesn't exist
        # In a real implementation, this would be loaded from external sources
        logger.info("Generating sample POI data")
        
        # Define some key NYC locations as POIs
        pois = [
            # Subway stations
            {"name": "Times Square", "type": "subway", "latitude": 40.7590, "longitude": -73.9845},
            {"name": "Grand Central", "type": "subway", "latitude": 40.7527, "longitude": -73.9772},
            {"name": "Union Square", "type": "subway", "latitude": 40.7356, "longitude": -73.9906},
            {"name": "Penn Station", "type": "subway", "latitude": 40.7506, "longitude": -73.9939},
            {"name": "Brooklyn Bridge", "type": "subway", "latitude": 40.7127, "longitude": -74.0059},
            {"name": "Atlantic Ave", "type": "subway", "latitude": 40.6845, "longitude": -73.9799},
            {"name": "Jackson Heights", "type": "subway", "latitude": 40.7470, "longitude": -73.8914},
            {"name": "Yankee Stadium", "type": "subway", "latitude": 40.8296, "longitude": -73.9262},
            {"name": "St. George", "type": "subway", "latitude": 40.6434, "longitude": -74.0739},
            
            # Parks
            {"name": "Central Park", "type": "park", "latitude": 40.7812, "longitude": -73.9665},
            {"name": "Prospect Park", "type": "park", "latitude": 40.6602, "longitude": -73.9690},
            {"name": "Bryant Park", "type": "park", "latitude": 40.7536, "longitude": -73.9832},
            {"name": "Washington Square Park", "type": "park", "latitude": 40.7308, "longitude": -73.9973},
            {"name": "Flushing Meadows", "type": "park", "latitude": 40.7461, "longitude": -73.8458},
            {"name": "Van Cortlandt Park", "type": "park", "latitude": 40.8968, "longitude": -73.8871},
            {"name": "Clove Lakes Park", "type": "park", "latitude": 40.6214, "longitude": -74.1162},
            
            # Schools
            {"name": "Columbia University", "type": "school", "latitude": 40.8075, "longitude": -73.9626},
            {"name": "NYU", "type": "school", "latitude": 40.7295, "longitude": -73.9965},
            {"name": "CUNY City College", "type": "school", "latitude": 40.8192, "longitude": -73.9494},
            {"name": "Fordham University", "type": "school", "latitude": 40.8614, "longitude": -73.8827},
            {"name": "St. John's University", "type": "school", "latitude": 40.7241, "longitude": -73.7962},
            {"name": "Wagner College", "type": "school", "latitude": 40.6156, "longitude": -74.0955},
            
            # Shopping
            {"name": "5th Avenue", "type": "shopping", "latitude": 40.7546, "longitude": -73.9691},
            {"name": "SoHo", "type": "shopping", "latitude": 40.7233, "longitude": -73.9985},
            {"name": "Atlantic Terminal Mall", "type": "shopping", "latitude": 40.6843, "longitude": -73.9783},
            {"name": "Queens Center Mall", "type": "shopping", "latitude": 40.7337, "longitude": -73.8694},
            {"name": "Bronx Terminal Market", "type": "shopping", "latitude": 40.8256, "longitude": -73.9308},
            {"name": "Staten Island Mall", "type": "shopping", "latitude": 40.5827, "longitude": -74.1662},
        ]
        
        self.poi_data = pd.DataFrame(pois)
        
        # Save POI data for future use
        self.poi_data.to_csv(poi_path, index=False)
        logger.info(f"Generated and saved {len(self.poi_data)} points of interest")
        
        return self.poi_data
    
    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate the great circle distance between two points in kilometers.
        
        Args:
            lat1: Latitude of the first point
            lon1: Longitude of the first point
            lat2: Latitude of the second point
            lon2: Longitude of the second point
        
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def _extract_coordinates(
        self, listing: pd.Series
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract latitude and longitude from a listing.
        
        Args:
            listing: Series representing a rental listing
        
        Returns:
            Tuple of (latitude, longitude) or (None, None) if not available
        """
        lat = lon = None
        
        # Try different field names for coordinates
        for lat_field in ["latitude", "lat"]:
            if lat_field in listing and pd.notna(listing[lat_field]):
                try:
                    lat = float(listing[lat_field])
                    break
                except (ValueError, TypeError):
                    pass
        
        for lon_field in ["longitude", "lon", "lng"]:
            if lon_field in listing and pd.notna(listing[lon_field]):
                try:
                    lon = float(listing[lon_field])
                    break
                except (ValueError, TypeError):
                    pass
        
        return lat, lon
    
    def generate_features(self, listings_df: pd.DataFrame) -> pd.DataFrame:
        """Generate distance-based features for rental listings.
        
        Args:
            listings_df: DataFrame with rental listings data
        
        Returns:
            DataFrame with additional distance-based features
        """
        # Load POI data
        self._load_poi_data()
        
        # Create a copy of the input DataFrame
        result_df = listings_df.copy()
        
        # Initialize distance feature columns
        distance_features = [
            "dist_nearest_subway",
            "dist_nearest_park",
            "dist_nearest_school",
            "dist_nearest_shopping",
            "dist_nearest_poi",
        ]
        
        for feature in distance_features:
            result_df[feature] = np.nan
        
        # Group POIs by type
        poi_by_type = self.poi_data.groupby("type")
        
        # Process each listing
        for idx, listing in result_df.iterrows():
            lat, lon = self._extract_coordinates(listing)
            
            if lat is None or lon is None:
                # Skip listings without coordinates
                continue
            
            # Calculate distances to each POI type
            min_distances = {}
            
            for poi_type, pois in poi_by_type:
                distances = []
                
                for _, poi in pois.iterrows():
                    distance = self.haversine_distance(
                        lat, lon, poi["latitude"], poi["longitude"]
                    )
                    distances.append(distance)
                
                if distances:
                    min_distances[f"dist_nearest_{poi_type}"] = min(distances)
            
            # Calculate distance to nearest POI of any type
            all_distances = []
            for _, poi in self.poi_data.iterrows():
                distance = self.haversine_distance(
                    lat, lon, poi["latitude"], poi["longitude"]
                )
                all_distances.append(distance)
            
            if all_distances:
                min_distances["dist_nearest_poi"] = min(all_distances)
            
            # Update the result DataFrame
            for feature, distance in min_distances.items():
                result_df.loc[idx, feature] = distance
        
        # Fill missing values with median
        for feature in distance_features:
            median_value = result_df[feature].median()
            result_df[feature] = result_df[feature].fillna(median_value)
        
        logger.info(f"Generated {len(distance_features)} distance-based features")
        
        return result_df