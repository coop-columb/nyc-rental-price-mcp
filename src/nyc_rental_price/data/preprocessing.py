"""Data preprocessing module for NYC rental price prediction."""

__all__ = [
    "load_data",
    "preprocess_data",
    "clean_data",
    "engineer_additional_features",
    "process_features",
]

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        data_path: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)


def preprocess_data(input_path, output_path, engineer_features=False):
    """Preprocess raw rental data for model training.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
        engineer_features: Whether to create additional features

    Returns:
        Processed DataFrame
    """
    logger.info(f"Loading data from {input_path}")
    df = load_data(input_path)

    # Basic cleaning
    logger.info("Cleaning data")
    df = clean_data(df)

    # Feature engineering
    if engineer_features:
        logger.info("Engineering features")
        df = engineer_additional_features(df)

    # Process features
    logger.info("Processing features")
    df_processed = process_features(df)

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

    return df_processed


def clean_data(df):
    """Clean the raw data.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Make a copy
    df = df.copy()

    # Drop rows where price is NaN
    if "price" in df.columns:
        df = df.dropna(subset=["price"])
        logger.info(
            f"Dropped {len(df) - len(df.dropna(subset=['price']))} rows with missing price values"
        )

    # Handle missing values
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Convert categorical columns to string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    # Remove outliers (simple method)
    for col in ["price", "sqft"]:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df = df[(df[col] >= q1) & (df[col] <= q3)]

    return df


def engineer_additional_features(df):
    """Create additional features.

    Args:
        df: DataFrame with basic features

    Returns:
        DataFrame with additional features
    """
    # Make a copy
    df = df.copy()

    # Price per square foot
    if "price" in df.columns and "sqft" in df.columns:
        # Convert sqft to float and handle NaN/zero values
        df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce").astype(np.float64)
        df["price"] = df["price"].astype(np.float64)

        # Calculate price_per_sqft only for valid entries
        mask = (df["sqft"] > 0) & df["sqft"].notna() & df["price"].notna()

        # Initialize price_per_sqft with NaN
        df["price_per_sqft"] = pd.Series(np.nan, index=df.index, dtype=np.float64)

        if mask.any():
            # Calculate for valid entries
            df.loc[mask, "price_per_sqft"] = (
                df.loc[mask, "price"] / df.loc[mask, "sqft"]
            ).astype(np.float64)

            # Get median for valid entries
            median_price_per_sqft = df.loc[mask, "price_per_sqft"].median()

            # Fill NaN values with median
            df["price_per_sqft"] = df["price_per_sqft"].fillna(median_price_per_sqft)

        logger.info(
            f"Created price_per_sqft feature with {df['price_per_sqft'].isna().sum()} NaN values"
        )
        logger.info(f"Price per sqft dtype: {df['price_per_sqft'].dtype}")

        if not df["price_per_sqft"].isna().all():
            logger.info(
                f"Price per sqft range: {df['price_per_sqft'].min():.2f} - {df['price_per_sqft'].max():.2f}"
            )
        else:
            logger.warning("Price per sqft has all NaN values")

    # Total rooms
    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["bedrooms"] = (
            pd.to_numeric(df["bedrooms"], errors="coerce").fillna(0).astype(np.float64)
        )
        df["bathrooms"] = (
            pd.to_numeric(df["bathrooms"], errors="coerce").fillna(0).astype(np.float64)
        )
        df["total_rooms"] = (df["bedrooms"] + df["bathrooms"]).astype(np.float64)
        logger.info(
            f"Created total_rooms feature with range: {df['total_rooms'].min():.1f} - {df['total_rooms'].max():.1f}"
        )

    # Amenities
    all_amenity_cols = [
        "has_doorman",
        "has_elevator",
        "has_dishwasher",
        "has_washer_dryer",
        "is_furnished",
        "has_balcony",
        "has_parking",
    ]

    # Find existing amenity columns
    existing_amenities = [col for col in all_amenity_cols if col in df.columns]

    if existing_amenities:
        # Process each amenity
        for col in existing_amenities:
            df[col] = df[col].fillna(False).astype(bool).astype(np.float64)

        # Calculate total amenities
        df["amenities_count"] = df[existing_amenities].sum(axis=1).astype(np.float64)

        logger.info(
            f"Created amenities_count feature from {len(existing_amenities)} amenities"
        )
        logger.info(
            f"Amenities count range: {df['amenities_count'].min():.0f} - {df['amenities_count'].max():.0f}"
        )

    logger.info(f"Engineered features shape: {df.shape}")
    return df


def process_features(df):
    """Process features for model training.

    Args:
        df: DataFrame with features

    Returns:
        Processed DataFrame with features ready for model training
    """
    # Make a copy
    df = df.copy()

    # Extract target variable first and preserve index
    if "price" in df.columns:
        target = df["price"].astype(np.float64)
        X = df.drop(columns=["price"])
        logger.info(f"Extracted price column with {target.isna().sum()} NaN values")
    else:
        target = None
        X = df.copy()
        logger.info("No price column found")

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Log feature types
    logger.info(
        f"Processing {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features"
    )

    # Create preprocessing pipelines
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Get feature names for one-hot encoded columns
    onehot_cols = []
    for col in categorical_cols:
        unique_values = X[col].unique()
        onehot_cols.extend([f"{col}_{val}" for val in unique_values if pd.notna(val)])

    # Fit and transform
    if categorical_cols and numerical_cols:
        # Data quality check before preprocessing
        nan_before = X.isna().sum().sum()
        if nan_before > 0:
            logger.warning(
                f"Found {nan_before} NaN values in features before preprocessing"
            )
            for col, count in X.isna().sum()[X.isna().sum() > 0].items():
                logger.warning(f"{col}: {count} NaN values")

        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)

        # Create DataFrame with processed features with explicit dtype
        processed_cols = numerical_cols + onehot_cols
        df_processed = pd.DataFrame(
            X_processed, columns=processed_cols, dtype=np.float64
        )

        # Add target back if it exists
        if target is not None:
            df_processed["price"] = target

            # Data quality check after adding price
            if df_processed["price"].isna().any():
                logger.warning(
                    f"Found {df_processed['price'].isna().sum()} NaN values in price column after processing"
                )

        # Final data quality check
        nan_counts = df_processed.isna().sum()
        if nan_counts.any():
            logger.warning("Found NaN values in processed features:")
            for col, count in nan_counts[nan_counts > 0].items():
                logger.warning(f"{col}: {count} NaN values")
    else:
        logger.warning("No categorical or numerical columns found for preprocessing")
        df_processed = X.copy()
        if target is not None:
            df_processed["price"] = target

    return df_processed


def main():
    """Main function to parse arguments and preprocess data."""
    parser = argparse.ArgumentParser(description="Preprocess rental data")

    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Path to save processed data"
    )

    parser.add_argument(
        "--engineer-features", action="store_true", help="Create additional features"
    )

    args = parser.parse_args()

    # Preprocess data
    preprocess_data(args.input, args.output, args.engineer_features)


if __name__ == "__main__":
    main()
