"""Temporal feature generator for rental listings.

This module provides functionality for generating time-based features for NYC rental listings.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalFeatureGenerator:
    """Generate time-based features for NYC rental listings."""
    
    def __init__(self, data_dir: str = "data/interim"):
        """Initialize the temporal feature generator.
        
        Args:
            data_dir: Directory to save/load temporal data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize seasonal patterns
        self.seasonal_patterns = self._load_seasonal_patterns()
        
        logger.info("Initialized TemporalFeatureGenerator")
    
    def _load_seasonal_patterns(self) -> pd.DataFrame:
        """Load or initialize seasonal rental price patterns.
        
        Returns:
            DataFrame with seasonal patterns
        """
        patterns_file = self.data_dir / "seasonal_patterns.csv"
        
        if patterns_file.exists():
            # Load existing patterns
            patterns = pd.read_csv(patterns_file)
            logger.info(f"Loaded seasonal patterns from {patterns_file}")
            return patterns
        
        # Initialize with synthetic seasonal patterns
        # In a real implementation, this would be derived from historical data
        months = list(range(1, 13))
        
        # Create synthetic seasonal factors
        # NYC rental market typically peaks in summer and early fall
        seasonal_factors = [
            0.92,  # January
            0.90,  # February
            0.93,  # March
            0.96,  # April
            0.98,  # May
            1.02,  # June
            1.05,  # July
            1.08,  # August
            1.07,  # September
            1.03,  # October
            0.98,  # November
            0.94,  # December
        ]
        
        # Create day-of-week factors
        # Weekends typically have higher search activity
        dow_factors = [
            0.97,  # Monday
            0.96,  # Tuesday
            0.98,  # Wednesday
            1.00,  # Thursday
            1.03,  # Friday
            1.04,  # Saturday
            1.02,  # Sunday
        ]
        
        # Create patterns DataFrame
        patterns = pd.DataFrame({
            "month": months,
            "seasonal_factor": seasonal_factors,
        })
        
        # Add day of week patterns
        patterns["dow"] = list(range(7))
        patterns["dow_factor"] = dow_factors
        
        # Save patterns for future use
        patterns.to_csv(patterns_file, index=False)
        logger.info(f"Initialized seasonal patterns and saved to {patterns_file}")
        
        return patterns
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object.
        
        Args:
            date_str: Date string to parse
        
        Returns:
            Datetime object or None if parsing fails
        """
        if not date_str or not isinstance(date_str, str):
            return None
        
        # Try different date formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%b %d, %Y",
            "%B %d, %Y",
            "%Y%m%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def generate_features(
        self,
        listings_df: pd.DataFrame,
        date_column: str = "posted_date",
        reference_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate temporal features for rental listings.
        
        Args:
            listings_df: DataFrame with rental listings data
            date_column: Name of the column containing dates
            reference_date: Optional reference date for relative calculations
        
        Returns:
            DataFrame with additional temporal features
        """
        # Create a copy of the input DataFrame
        result_df = listings_df.copy()
        
        # Parse the reference date if provided
        ref_date = None
        if reference_date:
            ref_date = self._parse_date(reference_date)
        
        if ref_date is None:
            # Use current date as reference if not provided
            ref_date = datetime.now()
        
        # Initialize temporal feature columns
        result_df["listing_month"] = np.nan
        result_df["listing_day"] = np.nan
        result_df["listing_dow"] = np.nan
        result_df["days_since_posted"] = np.nan
        result_df["is_weekend"] = False
        result_df["seasonal_factor"] = 1.0
        result_df["dow_factor"] = 1.0
        
        # Process each listing
        for idx, listing in result_df.iterrows():
            # Get the listing date
            date_value = listing.get(date_column)
            listing_date = self._parse_date(date_value)
            
            if listing_date:
                # Extract date components
                result_df.loc[idx, "listing_month"] = listing_date.month
                result_df.loc[idx, "listing_day"] = listing_date.day
                result_df.loc[idx, "listing_dow"] = listing_date.weekday()
                
                # Calculate days since posted
                days_since = (ref_date - listing_date).days
                result_df.loc[idx, "days_since_posted"] = max(0, days_since)
                
                # Set weekend flag
                result_df.loc[idx, "is_weekend"] = listing_date.weekday() >= 5
                
                # Apply seasonal factors
                month_idx = listing_date.month - 1
                dow_idx = listing_date.weekday()
                
                result_df.loc[idx, "seasonal_factor"] = self.seasonal_patterns.iloc[month_idx]["seasonal_factor"]
                result_df.loc[idx, "dow_factor"] = self.seasonal_patterns.iloc[dow_idx]["dow_factor"]
        
        # Fill missing values
        for col in ["listing_month", "listing_day", "listing_dow"]:
            result_df[col] = result_df[col].fillna(-1).astype(int)
        
        result_df["days_since_posted"] = result_df["days_since_posted"].fillna(0).astype(int)
        
        # Create month and day of week cyclical features
        result_df["month_sin"] = np.sin(2 * np.pi * result_df["listing_month"] / 12)
        result_df["month_cos"] = np.cos(2 * np.pi * result_df["listing_month"] / 12)
        result_df["dow_sin"] = np.sin(2 * np.pi * result_df["listing_dow"] / 7)
        result_df["dow_cos"] = np.cos(2 * np.pi * result_df["listing_dow"] / 7)
        
        # Create listing age features
        result_df["is_new_listing"] = result_df["days_since_posted"] <= 7
        result_df["listing_age_factor"] = np.exp(-0.05 * result_df["days_since_posted"])
        
        logger.info("Generated temporal features for rental listings")
        
        return result_df