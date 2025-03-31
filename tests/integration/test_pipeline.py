"""Integration test for the NYC rental price prediction pipeline.

This test verifies that the entire pipeline works correctly, from data loading to prediction.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.nyc_rental_price.data.preprocessing import preprocess_data, split_data
from src.nyc_rental_price.features import FeaturePipeline
from src.nyc_rental_price.models.model import GradientBoostingModel, NeuralNetworkModel, ModelEnsemble


@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    # Check if sample data exists
    sample_file = "data/raw/sample_listings.csv"
    if not os.path.exists(sample_file):
        pytest.skip(f"Sample data file not found: {sample_file}")
    
    return pd.read_csv(sample_file)


@pytest.fixture
def model_artifacts_path(tmp_path):
    """Create a temporary directory for model artifacts."""
    artifacts_path = tmp_path / "models"
    artifacts_path.mkdir(exist_ok=True)
    return artifacts_path


def test_full_pipeline(sample_data, model_artifacts_path):
    """Test the full machine learning pipeline from preprocessing to evaluation."""
    # Step 1: Preprocess the data
    processed_data = preprocess_data("data/raw/sample_listings.csv")
    
    # Check that we have processed data
    assert len(processed_data) > 0
    assert "price" in processed_data.columns
    
    # Step 2: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        processed_data, test_size=0.2, val_size=0.1
    )
    
    # Check that splits are correct
    assert len(X_train) + len(X_val) + len(X_test) == len(processed_data)
    
    # Step 3: Create and train gradient boosting model
    gb_model = GradientBoostingModel(
        model_dir=str(model_artifacts_path),
        model_name="test_gb_model",
        n_estimators=10,  # Small number for quick testing
    )
    
    # Train model
    gb_model.fit(X_train, y_train)
    
    # Evaluate model
    gb_metrics = gb_model.evaluate(X_test, y_test)
    
    # Check that we have metrics
    assert "mae" in gb_metrics
    assert "rmse" in gb_metrics
    assert "r2" in gb_metrics
    
    # Make predictions
    gb_predictions = gb_model.predict(X_test)
    
    # Check that we have predictions
    assert len(gb_predictions) == len(X_test)
    assert not np.isnan(gb_predictions).any()
    
    # Step 4: Create and train neural network model
    nn_model = NeuralNetworkModel(
        model_dir=str(model_artifacts_path),
        model_name="test_nn_model",
        hidden_layers=[16, 8],  # Small architecture for quick testing
        epochs=5,  # Small number for quick testing
    )
    
    # Train model
    nn_model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Evaluate model
    nn_metrics = nn_model.evaluate(X_test, y_test)
    
    # Check that we have metrics
    assert "mae" in nn_metrics
    assert "rmse" in nn_metrics
    assert "r2" in nn_metrics
    
    # Make predictions
    nn_predictions = nn_model.predict(X_test)
    
    # Check that we have predictions
    assert len(nn_predictions) == len(X_test)
    assert not np.isnan(nn_predictions).any()
    
    # Step 5: Create and train ensemble model
    ensemble_model = ModelEnsemble(
        models=[gb_model, nn_model],
        weights=[0.6, 0.4],
        model_dir=str(model_artifacts_path),
        model_name="test_ensemble_model",
    )
    
    # No need to train the ensemble since its component models are already trained
    
    # Evaluate model
    ensemble_metrics = ensemble_model.evaluate(X_test, y_test)
    
    # Check that we have metrics
    assert "mae" in ensemble_metrics
    assert "rmse" in ensemble_metrics
    assert "r2" in ensemble_metrics
    
    # Make predictions
    ensemble_predictions = ensemble_model.predict(X_test)
    
    # Check that we have predictions
    assert len(ensemble_predictions) == len(X_test)
    assert not np.isnan(ensemble_predictions).any()
    
    # Check that ensemble predictions are a weighted average of component predictions
    expected_predictions = gb_predictions * 0.6 + nn_predictions * 0.4
    np.testing.assert_allclose(ensemble_predictions, expected_predictions, rtol=1e-5)


def test_preprocessing_to_training_compatibility(sample_data, model_artifacts_path):
    """Test that preprocessing output is compatible with model training input."""
    # Preprocess the data
    processed_data = preprocess_data("data/raw/sample_listings.csv")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)
    
    # Create a model
    model = GradientBoostingModel(
        model_dir=str(model_artifacts_path),
        model_name="compatibility_test_model",
        n_estimators=5,  # Small number for quick testing
    )
    
    # Try to fit the model
    try:
        model.fit(X_train, y_train)
        fit_successful = True
    except Exception as e:
        fit_successful = False
        print(f"Fit failed with error: {e}")
    
    # Assert that fit was successful
    assert fit_successful, "Model fitting failed, indicating incompatibility between preprocessing and model"
    
    # Try to make predictions
    try:
        predictions = model.predict(X_test)
        predict_successful = True
    except Exception as e:
        predict_successful = False
        print(f"Prediction failed with error: {e}")
    
    # Assert that prediction was successful
    assert predict_successful, "Model prediction failed, indicating incompatibility between preprocessing and model"


def test_save_load_prediction_consistency(sample_data, model_artifacts_path):
    """Test that model predictions are consistent before and after saving/loading."""
    # Preprocess the data
    processed_data = preprocess_data("data/raw/sample_listings.csv")
    
    # Split data
    X_train, _, X_test, y_train, _, _ = split_data(processed_data)
    
    # Create and train a model
    model = GradientBoostingModel(
        model_dir=str(model_artifacts_path),
        model_name="consistency_test_model",
        n_estimators=5,  # Small number for quick testing
    )
    model.fit(X_train, y_train)
    
    # Generate predictions before saving
    predictions_before = model.predict(X_test)
    
    # Save the model
    model.save_model()
    
    # Create a new model instance
    loaded_model = GradientBoostingModel(
        model_dir=str(model_artifacts_path),
        model_name="consistency_test_model",
    )
    
    # Load the model
    loaded_model.load_model()
    
    # Generate predictions after loading
    predictions_after = loaded_model.predict(X_test)
    
    # Check that predictions are the same (within tolerance)
    np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5)