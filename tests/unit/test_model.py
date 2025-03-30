import pytest
import tensorflow as tf
import numpy as np

from nyc_rental_price.models.model import (
    create_model,
    compile_model,
    save_model,
    load_model
)


def test_create_model():
    """Test that the model creation function works as expected."""
    # Test with default parameters
    model = create_model(input_dim=10)
    
    # Check that it's a TensorFlow model
    assert isinstance(model, tf.keras.Model)
    
    # Check input shape
    assert model.input_shape == (None, 10)
    
    # Check that it has layers
    assert len(model.layers) > 1
    
    # Test with custom parameters
    model_custom = create_model(
        input_dim=15,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.3,
        activation='relu'
    )
    
    # Check the custom input shape
    assert model_custom.input_shape == (None, 15)
    
    # Count the dense layers (excluding input layer)
    dense_layers = [layer for layer in model_custom.layers if isinstance(layer, tf.keras.layers.Dense)]
    assert len(dense_layers) == 4  # 3 hidden + 1 output


def test_compile_model():
    """Test that the model compilation function works as expected."""
    model = create_model(input_dim=10)
    compiled_model = compile_model(model)
    
    # Check that the model has been compiled
    assert compiled_model.optimizer is not None
    assert compiled_model.loss is not None
    
    # Test with custom parameters
    compiled_model_custom = compile_model(
        model,
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Check optimizer type
    assert isinstance(compiled_model_custom.optimizer, tf.keras.optimizers.Adam)
    
    # Check metrics
    assert 'mae' in [m.name for m in compiled_model_custom.metrics]


def test_model_save_load(model_artifacts_path, processed_data):
    """Test saving and loading model functionality."""
    X, y = processed_data
    
    # Create and train a simple model
    model = create_model(input_dim=X.shape[1])
    model.fit(X, y, epochs=1, verbose=0)
    
    # Get predictions before saving
    predictions_before = model.predict(X, verbose=0)
    
    # Save the model
    model_path = model_artifacts_path / "test_model"
    save_model(model, model_path)
    
    # Check that model files were created
    assert (model_path / "saved_model.pb").exists()
    
    # Load the model
    loaded_model = load_model(model_path)
    
    # Check that it's a TensorFlow model
    assert isinstance(loaded_model, tf.keras.Model)
    
    # Get predictions after loading
    predictions_after = loaded_model.predict(X, verbose=0)
    
    # Check that the predictions are the same
    np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5)


def test_model_predict(trained_model, processed_data):
    """Test model prediction functionality."""
    X, _ = processed_data
    
    # Generate predictions
    predictions = trained_model.predict(X, verbose=0)
    
    # Check that predictions shape is correct
    assert predictions.shape == (len(X), 1)
    
    # Check that predictions are within reasonable range for rental prices
    assert np.all(predictions > 0)  # Rental prices should be positive

