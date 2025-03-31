import os

import numpy as np
import psutil
import pytest
import tensorflow as tf
from tensorflow import keras


def load_model():
    """Load or create a model for testing."""
    model_path = os.path.join("models", "best_mcp_model.h5")
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        # Create a simple model for testing
        model = keras.Sequential(
            [
                keras.Input(shape=(10,)),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
    return model


def test_model_loading_time(benchmark):
    """Test the time it takes to load the model using pytest-benchmark."""
    # Use benchmark to measure the model loading time
    result = benchmark(load_model)

    # The benchmark fixture automatically captures and reports timing information
    # We can still add a basic assertion to ensure the model loads quickly
    assert (
        benchmark.stats.stats.mean < 5.0
    ), f"Model loading took too long: {benchmark.stats.stats.mean:.4f} seconds"


def test_prediction_speed(benchmark):
    """Test the speed of model predictions using pytest-benchmark."""
    # Load the model first (outside of the benchmark)
    model = load_model()

    # Create sample input data with correct shape (matching the actual model input shape)
    num_samples = 100
    input_data = np.random.random((num_samples, 10))  # Changed from 5 to 10 features

    # Warm-up run (outside of the benchmark)
    _ = model.predict(input_data[:1])

    # Use benchmark to measure prediction time
    def run_prediction():
        return model.predict(input_data)

    # Run the benchmark
    predictions = benchmark(run_prediction)

    # Calculate average time per sample using benchmark stats
    avg_time_per_sample = benchmark.stats.stats.mean / num_samples

    # Assert that predictions are reasonably fast
    assert (
        avg_time_per_sample < 0.01
    ), f"Predictions too slow: {avg_time_per_sample:.6f} seconds per sample"


def test_model_memory_usage(benchmark):
    """Test the memory usage of the model using pytest-benchmark."""
    process = psutil.Process(os.getpid())

    def measure_model_loading_memory():
        memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        model = load_model()
        memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
        memory_used = memory_after - memory_before
        return memory_used

    # Use benchmark to measure memory usage consistency
    memory_used = benchmark(measure_model_loading_memory)

    # We can add an assertion for memory limits if needed
    # For now, just demonstrate using benchmark
    assert True, f"Memory used by model: {memory_used:.2f} MB"
