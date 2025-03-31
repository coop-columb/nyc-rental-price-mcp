#!/usr/bin/env python
"""
Advanced model training example for NYC rental price prediction.

This script demonstrates how to use the advanced ML capabilities:
1. LightGBM and XGBoost models
2. Bayesian hyperparameter optimization
3. K-fold cross-validation
4. Model ensembling
5. Feature importance analysis
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.nyc_rental_price.data.preprocessing import (
    clean_data,
    generate_features,
    load_data,
    split_data,
)
from src.nyc_rental_price.features import FeaturePipeline
from src.nyc_rental_price.models.model import (
    BaseModel,
    BayesianOptimizer,
    CrossValidator,
    GradientBoostingModel,
    LightGBMModel,
    ModelEnsemble,
    NeuralNetworkModel,
    XGBoostModel,
)
from src.nyc_rental_price.models.train import (
    create_model,
    cross_validate_model,
    evaluate_model,
    train_model,
    tune_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_advanced_training(
    data_path: str,
    output_dir: str = "models/advanced",
    tune_hyperparams: bool = True,
    cross_validate: bool = True,
    n_folds: int = 5,
    create_ensemble: bool = True,
    random_state: int = 42,
):
    """Run advanced model training with multiple models and techniques.

    Args:
        data_path: Path to the processed data file
        output_dir: Directory to save models and results
        tune_hyperparams: Whether to perform hyperparameter tuning
        cross_validate: Whether to perform cross-validation
        n_folds: Number of folds for cross-validation
        create_ensemble: Whether to create an ensemble of models
        random_state: Random seed for reproducibility
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)

    if len(df) == 0:
        logger.error(f"No data loaded from {data_path}")
        return

    # Clean data
    df = clean_data(df)

    # Generate features
    pipeline = FeaturePipeline()
    df = generate_features(df, pipeline)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=0.2, val_size=0.1, random_state=random_state
    )

    # Define models to train
    model_types = ["gradient_boosting", "lightgbm", "xgboost", "neural_network"]
    models = {}
    metrics = {}

    # Train each model type
    for model_type in model_types:
        logger.info(f"Processing {model_type} model")
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(exist_ok=True)

        # Create model
        model = create_model(
            model_type=model_type,
            model_dir=str(model_output_dir),
            model_name=f"{model_type}_model",
            random_state=random_state,
        )

        # Hyperparameter tuning if requested
        if tune_hyperparams:
            logger.info(f"Tuning hyperparameters for {model_type}")
            tuned_model = tune_model(
                model=model,
                X=X_train,
                y=y_train,
                cv=3,
                n_iter=20,
                random_state=random_state,
            )
            model = tuned_model

        # Cross-validation if requested
        if cross_validate:
            logger.info(f"Performing {n_folds}-fold cross-validation for {model_type}")
            cv_results = cross_validate_model(
                model=model,
                X=pd.concat([X_train, X_val]),
                y=pd.concat([y_train, y_val]),
                cv=n_folds,
                random_state=random_state,
            )
            
            # Save cross-validation results
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv(model_output_dir / "cv_results.csv", index=False)
            
            # Plot cross-validation results
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=cv_results_df)
            plt.title(f"{model_type} Cross-Validation Results")
            plt.ylabel("Value")
            plt.xlabel("Metric")
            plt.tight_layout()
            plt.savefig(model_output_dir / "cv_results.png", dpi=300)
            plt.close()

        # Train the model
        trained_model = train_model(model, X_train, y_train, X_val, y_val)
        
        # Evaluate model
        model_metrics = evaluate_model(
            trained_model, X_test, y_test, str(model_output_dir)
        )
        
        # Store model and metrics
        models[model_type] = trained_model
        metrics[model_type] = model_metrics

    # Create ensemble if requested
    if create_ensemble and len(models) > 1:
        logger.info("Creating model ensemble")
        ensemble_dir = output_dir / "ensemble"
        ensemble_dir.mkdir(exist_ok=True)

        # Create ensemble with all models
        ensemble = ModelEnsemble(
            models=list(models.values()),
            model_dir=str(ensemble_dir),
            model_name="advanced_ensemble",
            random_state=random_state,
        )

        # Evaluate ensemble
        ensemble_metrics = evaluate_model(
            ensemble, X_test, y_test, str(ensemble_dir)
        )
        metrics["ensemble"] = ensemble_metrics

    # Compare model performance
    compare_models(metrics, output_dir)


def compare_models(metrics: dict, output_dir: Path):
    """Compare performance of different models.

    Args:
        metrics: Dictionary of model metrics
        output_dir: Directory to save comparison results
    """
    logger.info("Comparing model performance")

    # Create comparison dataframe
    comparison = []
    for model_name, model_metrics in metrics.items():
        row = {"model": model_name}
        row.update(model_metrics)
        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)
    
    # Save comparison results
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot MAE
    plt.subplot(2, 2, 1)
    sns.barplot(x="model", y="mae", data=comparison_df)
    plt.title("Mean Absolute Error (lower is better)")
    plt.xticks(rotation=45)
    
    # Plot RMSE
    plt.subplot(2, 2, 2)
    sns.barplot(x="model", y="rmse", data=comparison_df)
    plt.title("Root Mean Squared Error (lower is better)")
    plt.xticks(rotation=45)
    
    # Plot R²
    plt.subplot(2, 2, 3)
    sns.barplot(x="model", y="r2", data=comparison_df)
    plt.title("R² Score (higher is better)")
    plt.xticks(rotation=45)
    
    # Plot MAPE
    plt.subplot(2, 2, 4)
    sns.barplot(x="model", y="mape", data=comparison_df)
    plt.title("Mean Absolute Percentage Error (lower is better)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close()

    # Log best model
    best_model_rmse = comparison_df.loc[comparison_df["rmse"].idxmin()]["model"]
    best_model_r2 = comparison_df.loc[comparison_df["r2"].idxmax()]["model"]
    
    logger.info(f"Best model by RMSE: {best_model_rmse}")
    logger.info(f"Best model by R²: {best_model_r2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run advanced model training for NYC rental price prediction"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the processed data file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/advanced",
        help="Directory to save models and results",
    )
    
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Perform cross-validation",
    )
    
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    
    parser.add_argument(
        "--create-ensemble",
        action="store_true",
        help="Create an ensemble of models",
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    run_advanced_training(
        data_path=args.data_path,
        output_dir=args.output_dir,
        tune_hyperparams=args.tune_hyperparams,
        cross_validate=args.cross_validate,
        n_folds=args.n_folds,
        create_ensemble=args.create_ensemble,
        random_state=args.random_state,
    )