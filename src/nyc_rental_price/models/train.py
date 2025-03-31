"""Training module for NYC rental price prediction models.

This module provides functions for training and evaluating rental price prediction models.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    GradientBoostingModel,
    LightGBMModel,
    XGBoostModel,
    ModelEnsemble,
    NeuralNetworkModel,
    BayesianOptimizer,
    CrossValidator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cross_validate_model(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """Perform k-fold cross-validation for model evaluation.

    Args:
        model: Model to evaluate
        X: Feature data
        y: Target data
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of evaluation metrics for each fold
    """
    logger.info(f"Performing {n_folds}-fold cross-validation for {model.model_name}")
    
    validator = CrossValidator(
        model=model,
        n_folds=n_folds,
        random_state=random_state,
    )
    
    cv_metrics = validator.validate(X, y)
    
    # Log average metrics
    for metric, values in cv_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        logger.info(f"CV {metric}: {mean_value:.4f} (+/- {std_value:.4f})")
    
    return cv_metrics


def train_model(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> BaseModel:
    """Train a model on the training data.

    Args:
        model: Model to train
        X_train: Training features
        y_train: Training target
        X_val: Optional validation features
        y_val: Optional validation target

    Returns:
        Trained model
    """
    logger.info(f"Training {model.model_name}")

    # Train the model
    if (
        isinstance(model, NeuralNetworkModel)
        and X_val is not None
        and y_val is not None
    ):
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    # Save the model
    model.save_model()

    return model


def evaluate_model(
    model: BaseModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a model on the test data.

    Args:
        model: Model to evaluate
        X_test: Test features
        y_test: Test target
        output_dir: Optional directory to save evaluation results

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model.model_name}")

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Create evaluation plots
    if output_dir:
        # Create a figure for the plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Actual vs. Predicted plot
        axes[0].scatter(y_test, y_pred, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
        axes[0].set_xlabel("Actual Price")
        axes[0].set_ylabel("Predicted Price")
        axes[0].set_title("Actual vs. Predicted Rental Prices")

        # Residuals plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color="k", linestyle="--")
        axes[1].set_xlabel("Predicted Price")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residuals vs. Predicted Rental Prices")

        # Add metrics as text
        metrics_text = "\n".join(
            [
                f"MAE: ${metrics['mae']:.2f}",
                f"RMSE: ${metrics['rmse']:.2f}",
                f"R²: {metrics['r2']:.4f}",
                f"MAPE: {metrics['mape']:.2f}%",
            ]
        )

        fig.text(0.02, 0.02, metrics_text, fontsize=12)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f"{model.model_name}_evaluation.png", dpi=300)
        plt.close()

        # Save predictions and actual values
        results_df = pd.DataFrame(
            {
                "actual": y_test,
                "predicted": y_pred,
                "residual": residuals,
            }
        )

        results_df.to_csv(
            output_dir / f"{model.model_name}_predictions.csv", index=False
        )

        # If the model is a gradient boosting model, plot feature importances
        if isinstance(model, GradientBoostingModel) and hasattr(
            model, "feature_importances_"
        ):
            plt.figure(figsize=(10, 8))

            # Get top 20 features
            top_features = model.feature_importances_.head(20)

            # Plot feature importances
            sns.barplot(x=top_features.values, y=top_features.index)
            plt.title(f"Top 20 Feature Importances - {model.model_name}")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{model.model_name}_feature_importances.png", dpi=300
            )
            plt.close()

            # Save feature importances
            model.feature_importances_.to_csv(
                output_dir / f"{model.model_name}_feature_importances.csv"
            )

    return metrics


def tune_model(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_space: Dict[str, Union[float, int, str]],
    n_trials: int = 50,
    cv_folds: int = 5,
) -> Tuple[BaseModel, Dict[str, Union[float, int, str]]]:
    """Tune model hyperparameters using Bayesian optimization.

    Args:
        model: Model to tune
        X_train: Training features
        y_train: Training target
        param_space: Dictionary defining the hyperparameter search space
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (tuned model, best parameters)
    """
    logger.info(f"Tuning hyperparameters for {model.model_name}")
    
    optimizer = BayesianOptimizer(
        model=model,
        param_space=param_space,
        n_trials=n_trials,
        cv_folds=cv_folds,
    )
    
    best_params = optimizer.optimize(X_train, y_train)
    tuned_model = create_model(model.model_type, **best_params)
    
    return tuned_model, best_params


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Create a model of the specified type.

    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Created model
    """
    if model_type == "gradient_boosting":
        return GradientBoostingModel(**kwargs)
    elif model_type == "lightgbm":
        return LightGBMModel(**kwargs)
    elif model_type == "xgboost":
        return XGBoostModel(**kwargs)
    elif model_type == "neural_network":
        return NeuralNetworkModel(**kwargs)
    elif model_type == "ensemble":
        # Create individual models
        gb_model = GradientBoostingModel(model_name="gb_for_ensemble")
        nn_model = NeuralNetworkModel(model_name="nn_for_ensemble")

        # Create ensemble
        ensemble = ModelEnsemble(
            models=[gb_model, nn_model],
            weights=[0.6, 0.4],  # Assign higher weight to gradient boosting
            **kwargs,
        )

        return ensemble
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_and_evaluate(
    data_path: str,
    model_type: str = "gradient_boosting",
    output_dir: str = "models",
    model_name: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
    param_space: Optional[Dict[str, Union[float, int, str]]] = None,
    n_trials: int = 50,
    perform_cv: bool = False,
    cv_folds: int = 5,
) -> Dict[str, float]:
    """Train and evaluate a model on the specified data.

    Args:
        data_path: Path to the data file
        model_type: Type of model to train
        output_dir: Directory to save model and evaluation results
        model_name: Optional name for the model
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of evaluation metrics
    """
    # Load and preprocess data
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)

    if len(df) == 0:
        logger.error(f"No data loaded from {data_path}")
        return {}

    # Clean data
    df = clean_data(df)

    # Generate features
    pipeline = FeaturePipeline()
    df = generate_features(df, pipeline)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )

    # Create model
    if model_name is None:
        model_name = f"{model_type}_model"

    model = create_model(
        model_type=model_type,
        model_dir=output_dir,
        model_name=model_name,
        random_state=random_state,
    )

    # Train model
    trained_model = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate model
    metrics = evaluate_model(trained_model, X_test, y_test, output_dir)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a rental price prediction model"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the processed data file",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble",
        choices=["gradient_boosting", "neural_network", "ensemble"],
        help="Type of model to train",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model and evaluation results",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for the model",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Train and evaluate model
    metrics = train_and_evaluate(
        data_path=args.data_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        model_name=args.model_name,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    # Print metrics
    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
