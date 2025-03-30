import pytest
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

from nyc_rental_price.data.preprocessing import preprocess_data
from nyc_rental_price.models.model import create_model, compile_model, save_model, load_model
from nyc_rental_price.models.train import train_model, evaluate_model


def test_full_pipeline(sample_data, model_artifacts_path):
    """Test the full machine learning pipeline from preprocessing to evaluation."""
    # Step 1: Preprocess the data
    X, y = preprocess_data(sample_data)
    
    # Split data into train and test sets
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Step 2: Create and compile the model
    input_dim = X.shape[1]
    model = create_model(input_dim=input_dim, hidden_layers=[32, 16])
    compiled_model = compile_model(model, optimizer='adam', loss='mse', metrics=['mae'])
    
    # Step 3: Train the model
    model_path = model_artifacts_path / "pipeline_test_model"
    history = train_model(
        compiled_model,
        X_train,
        y_train,
        validation_split=0.2,
        epochs=3,
        batch_size=32,
        model_path=model_path,
        verbose=0
    )
    
    # Check that training history was recorded
    assert 'loss' in history.history
    assert len(history.history['loss']) == 3  # 3 epochs
    
    # Step 4: Load the saved model
    loaded_model = load_model(model_path)
    
    # Step 5: Evaluate the model
    metrics = evaluate_model(loaded_model, X_test, y_test)
    
    # Check that evaluation metrics were calculated
    assert 'loss' in metrics
    assert 'mae' in metrics
    
    # Step 6: Generate predictions
    predictions = loaded_model.predict(X_test, verbose=0)
    
    # Check predictions shape and values
    assert predictions.shape == (len(X_test), 1)
    assert np.all(predictions > 0)  # Rental prices should be positive


def test_preprocessing_to_training_compatibility(sample_data):
    """Test that preprocessing output is compatible with model training input."""
    # Preprocess the data
    X, y = preprocess_data(sample_data)
    
    # Create a model with matching input dimensions
    input_dim = X.shape[1]
    model = create_model(input_dim=input_dim)
    compiled_model = compile_model(model)
    
    # Try to fit the model for 1 epoch
    # This will fail if there's a shape mismatch or data type issues
    try:
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        fit_successful = True
    except Exception as e:
        fit_successful = False
        print(f"Fit failed with error: {e}")
    
    # Assert that fit was successful
    assert fit_successful, "Model fitting failed, indicating incompatibility between preprocessing and model"


def test_save_load_prediction_consistency(sample_data, model_artifacts_path):
    """Test that model predictions are consistent before and after saving/loading."""
    # Preprocess the data
    X, y = preprocess_data(sample_data)
    
    # Create and train a simple model
    input_dim = X.shape[1]
    model = create_model(input_dim=input_dim, hidden_layers=[16, 8])
    compiled_model = compile_model(model)
    compiled_model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    
    # Generate predictions before saving
    predictions_before = compiled_model.predict(X, verbose=0)
    
    # Save the model
    model_path = model_artifacts_path / "consistency_test_model"
    save_model(compiled_model, model_path)
    
    # Load the model
    loaded_model = load_model(model_path)
    
    # Generate predictions after loading
    predictions_after = loaded_model.predict(X, verbose=0)
    
    # Check that predictions are the same (within tolerance)
    np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5)

