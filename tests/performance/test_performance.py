import time
import pytest
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

def test_model_loading_time():
    """Test the time it takes to load the model."""
    start_time = time.time()
    
    # Load the model if it exists, otherwise create a simple one for testing
    model_path = os.path.join('models', 'best_mcp_model.h5')
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        # Create a simple model for testing
        model = keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
    
    end_time = time.time()
    load_time = end_time - start_time
    
    print(f"Model loading time: {load_time:.4f} seconds")
    
    # Assert that model loading takes less than 5 seconds
    # This threshold can be adjusted based on your requirements
    assert load_time < 5.0, f"Model loading took too long: {load_time:.4f} seconds"

def test_prediction_speed():
    """Test the speed of model predictions."""
    # Load or create model
    model_path = os.path.join('models', 'best_mcp_model.h5')
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        # Create a simple model for testing
        model = keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
    
    # Create sample input data
    num_samples = 100
    input_data = np.random.random((num_samples, 5))
    
    # Warm-up run
    _ = model.predict(input_data[:1])
    
    # Measure prediction time
    start_time = time.time()
    predictions = model.predict(input_data)
    end_time = time.time()
    
    prediction_time = end_time - start_time
    avg_time_per_sample = prediction_time / num_samples
    
    print(f"Total prediction time for {num_samples} samples: {prediction_time:.4f} seconds")
    print(f"Average prediction time per sample: {avg_time_per_sample:.6f} seconds")
    
    # Assert that predictions are reasonably fast
    # This threshold can be adjusted based on your requirements
    assert avg_time_per_sample < 0.01, f"Predictions too slow: {avg_time_per_sample:.6f} seconds per sample"

def test_model_memory_usage():
    """Test the memory usage of the model."""
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Load or create model
    model_path = os.path.join('models', 'best_mcp_model.h5')
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        # Create a simple model for testing
        model = keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
    
    memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
    memory_used = memory_after - memory_before
    
    print(f"Memory used by model: {memory_used:.2f} MB")
    
    # This test will always pass, it's just for monitoring
    # You can add assertions later if you want to enforce memory limits
    assert True

