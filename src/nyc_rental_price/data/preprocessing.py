"""Data preprocessing module for NYC rental price prediction.

This module provides functions for cleaning and preparing rental listings data.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.nyc_rental_price.features import FeaturePipeline

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load rental listings data from a CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with rental listings data
    """
    logger.info(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} listings from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean rental listings data.

    Args:
        df: DataFrame with rental listings data

    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning {len(df)} rental listings")

    # Create a copy of the input DataFrame
    result_df = df.copy()

    # Standardize column names
    result_df.columns = [col.lower().replace(" ", "_") for col in result_df.columns]

    # Clean price column
    if "price" in result_df.columns:
        # Extract numeric price
        result_df["price"] = (
            result_df["price"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", np.nan)
        )

        # Convert to numeric
        result_df["price"] = pd.to_numeric(result_df["price"], errors="coerce")

        # Remove rows with invalid prices
        valid_price_mask = (
            result_df["price"].notna()
            & (result_df["price"] > 0)
            & (result_df["price"] < 100000)  # Upper limit for NYC rentals
        )

        result_df = result_df[valid_price_mask].reset_index(drop=True)
        logger.info(f"Removed {len(df) - len(result_df)} listings with invalid prices")

    # Clean bedrooms and bathrooms
    for col in ["bedrooms", "bathrooms"]:
        if col in result_df.columns:
            # Extract numeric values
            result_df[col] = (
                result_df[col]
                .astype(str)
                .str.replace(r"[^\d\.]", "", regex=True)
                .replace("", np.nan)
            )

            # Convert to numeric
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

            # Set reasonable limits
            min_val = 0
            max_val = 10 if col == "bedrooms" else 8

            # Apply limits
            result_df[col] = result_df[col].clip(min_val, max_val)

    # Clean square footage
    if "sqft" in result_df.columns:
        # Extract numeric values
        result_df["sqft"] = (
            result_df["sqft"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", np.nan)
        )

        # Convert to numeric
        result_df["sqft"] = pd.to_numeric(result_df["sqft"], errors="coerce")

        # Set reasonable limits for NYC apartments
        result_df["sqft"] = result_df["sqft"].clip(100, 10000)

    # Clean neighborhood
    if "neighborhood" in result_df.columns:
        # Convert to string and lowercase
        result_df["neighborhood"] = (
            result_df["neighborhood"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace("nan", np.nan)
        )

        # Remove rows with missing neighborhoods
        result_df = result_df[result_df["neighborhood"].notna()].reset_index(drop=True)
        logger.info(
            f"Removed {len(df) - len(result_df)} listings with missing neighborhoods"
        )

    # Remove duplicate listings
    if "id" in result_df.columns:
        # Remove duplicates by ID
        result_df = result_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        logger.info(f"Removed duplicate listings, remaining: {len(result_df)}")

    # Fill missing values
    for col in result_df.columns:
        if result_df[col].dtype == "object":
            result_df[col] = result_df[col].fillna("")
        elif pd.api.types.is_numeric_dtype(result_df[col]):
            # For numeric columns, fill with median
            result_df[col] = result_df[col].fillna(result_df[col].median())

    logger.info(f"Cleaned data contains {len(result_df)} listings")

    return result_df


def combine_data_sources(filepaths: List[str]) -> pd.DataFrame:
    """Combine rental listings data from multiple sources.

    Args:
        filepaths: List of paths to CSV files

    Returns:
        Combined DataFrame
    """
    logger.info(f"Combining data from {len(filepaths)} sources")

    all_data = []

    for filepath in filepaths:
        # Load data from this source
        df = load_data(filepath)

        if len(df) > 0:
            # Add source column if not present
            if "source" not in df.columns:
                source_name = Path(filepath).stem.split("_")[0]
                df["source"] = source_name

            all_data.append(df)

    if not all_data:
        logger.warning("No data loaded from any source")
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data contains {len(combined_df)} listings")

    return combined_df


def generate_features(
    df: pd.DataFrame, pipeline: Optional[FeaturePipeline] = None
) -> pd.DataFrame:
    """Generate features for rental listings using the feature pipeline.

    Args:
        df: DataFrame with rental listings data
        pipeline: Optional feature pipeline instance

    Returns:
        DataFrame with generated features
    """
    logger.info(f"Generating features for {len(df)} listings")

    if pipeline is None:
        # Create a new pipeline
        pipeline = FeaturePipeline()

    # Generate features
    try:
        # Check if pipeline is already fitted
        is_fitted = hasattr(pipeline, "feature_columns") and pipeline.feature_columns

        if is_fitted:
            # Transform data
            features_df = pipeline.transform(df)
        else:
            # Fit and transform
            features_df = pipeline.fit(df).transform(df)

        logger.info(f"Generated features, result shape: {features_df.shape}")
        return features_df
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return df


def split_data(
    df: pd.DataFrame,
    target_column: str = "price",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into training, validation, and test sets.

    Args:
        df: DataFrame with rental listings data
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split

    logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}")

    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # First split: training + validation vs. test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: training vs. validation
    # Adjust validation size to account for the first split
    adjusted_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, random_state=random_state
    )

    logger.info(
        f"Split data into train ({len(X_train)}), validation ({len(X_val)}), "
        f"and test ({len(X_test)}) sets"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(filepath: str) -> pd.DataFrame:
    """Load, clean, and preprocess rental listings data.

    Args:
        filepath: Path to the CSV file

    Returns:
        Preprocessed DataFrame
    """
    # Load data
    df = load_data(filepath)

    if len(df) == 0:
        return pd.DataFrame()

    # Clean data
    cleaned_df = clean_data(df)

    # Generate features
    pipeline = FeaturePipeline()
    features_df = generate_features(cleaned_df, pipeline)

    return features_df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Process command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess rental listings data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/listings_processed.csv",
        help="Path to output CSV file",
    )

    args = parser.parse_args()

    # Check if input is a directory or file
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        # Process all CSV files in the directory
        csv_files = list(input_path.glob("*.csv"))

        if not csv_files:
            logger.error(f"No CSV files found in {input_path}")
            exit(1)

        # Combine data from all files
        df = combine_data_sources([str(file) for file in csv_files])
    else:
        # Process a single file
        df = load_data(str(input_path))

    if len(df) == 0:
        logger.error("No data to process")
        exit(1)

    # Clean and preprocess data
    cleaned_df = clean_data(df)

    # Generate features
    pipeline = FeaturePipeline()
    features_df = generate_features(cleaned_df, pipeline)

    # Save preprocessed data
    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed data to {output_path}")
