"""Training module for NYC rental price prediction models."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.nyc_rental_price.models.model import (
    GradientBoostingModel,
    LightGBMModel,
    ModelEnsemble,
    NeuralNetworkModel,
    XGBoostModel,
)

logger = logging.getLogger(__name__)


def load_data(data_path):
    """Load and prepare data for model training."""
    df = pd.read_csv(data_path)
    logger.info(
        f"Loaded data from {data_path} with {df.shape[0]} rows and {df.shape[1]} columns"
    )

    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.any():
        logger.warning("Found NaN values in the following columns:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"{col}: {count} NaN values")

    # Handle NaN values in target column
    if "price" in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=["price"])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(
                f"Dropped {dropped_rows} rows with NaN values in price column"
            )

    return df


def train_model(args):
    """Train a model with the specified parameters."""
    # Load data
    data = load_data(args.data_path)

    # Extract target variable
    if args.target_column not in data.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in data")

    y = data[args.target_column]
    X = data.drop(columns=[args.target_column])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")

    # Validate data after split
    logger.info("Validating data quality after split...")

    # Check target variables
    if y_train.isna().any():
        logger.warning(f"Found {y_train.isna().sum()} NaN values in training target")
        y_train = y_train.dropna()
        X_train = X_train[y_train.index]

    if y_test.isna().any():
        logger.warning(f"Found {y_test.isna().sum()} NaN values in test target")
        y_test = y_test.dropna()
        X_test = X_test[y_test.index]

    # Check feature variables
    train_nan_counts = X_train.isna().sum()
    test_nan_counts = X_test.isna().sum()

    if train_nan_counts.any():
        logger.warning("Found NaN values in training features:")
        for col, count in train_nan_counts[train_nan_counts > 0].items():
            logger.warning(f"{col}: {count} NaN values")
        # Fill NaN values with median for numerical columns
        for col in X_train.select_dtypes(include=["number"]).columns:
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(
                X_train[col].median()
            )  # Use training median for test

    # Create model based on model_type
    if args.model_type == "gradient_boosting":
        model = GradientBoostingModel(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
    elif args.model_type == "lightgbm":
        model = LightGBMModel(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
    elif args.model_type == "xgboost":
        model = XGBoostModel(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
    elif args.model_type == "neural_network":
        model = NeuralNetworkModel(
            hidden_layers=[64, 32], activation="relu", learning_rate=0.001
        )
    elif args.model_type == "ensemble":
        # Create an ensemble of multiple models
        ensemble = ModelEnsemble()

        # Add different models to the ensemble
        ensemble.add_model(GradientBoostingModel())
        ensemble.add_model(LightGBMModel())
        ensemble.add_model(XGBoostModel())

        model = ensemble
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Train model
    logger.info(f"Training {args.model_type} model...")
    model.fit(X_train, y_train)

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Model evaluation metrics: {metrics}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    return model, metrics


def main():
    """Main function to parse arguments and train a model."""
    parser = argparse.ArgumentParser(
        description="Train a rental price prediction model"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the processed data CSV file",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="gradient_boosting",
        choices=[
            "gradient_boosting",
            "lightgbm",
            "xgboost",
            "neural_network",
            "ensemble",
        ],
        help="Type of model to train",
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="price",
        help="Name of the target column in the data",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators for tree-based models",
    )

    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate for models"
    )

    parser.add_argument(
        "--max-depth", type=int, default=3, help="Maximum depth for tree-based models"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Train model
    train_model(args)


if __name__ == "__main__":
    main()
