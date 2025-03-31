import os
import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

from data_processing.preprocessing import preprocess_data
from nyc_rental_price.models.model import create_model


@pytest.fixture
def sample_data():
    """Generate sample rental data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'bedrooms': np.random.randint(0, 5, 100),
        'bathrooms': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], 100),
        'sqft': np.random.randint(400, 2000, 100),
        'neighborhood': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], 100),
        'building_age_years': np.random.randint(0, 100, 100),
        'floor': np.random.randint(1, 30, 100),
        'has_elevator': np.random.choice([0, 1], 100),
        'has_doorman': np.random.choice([0, 1], 100),
        'price': np.random.randint(1500, 5000, 100)
    })


@pytest.fixture
def processed_data(sample_data):
    """Return processed X and y data for model testing."""
    X, y = preprocess_data(sample_data)
    # Convert y to numpy array if it's a pandas Series
    if isinstance(y, pd.Series):
        y = y.values
    return X, y


@pytest.fixture
def trained_model(processed_data):
    """Create a small model and train it minimally for testing."""
    X, y = processed_data
    
    # Get input shape from processed features
    input_dim = X.shape[1]
    
    # Create a simple model for testing
    model = create_model(input_dim=input_dim, hidden_layers=[10, 5])
    
    # Train for just 2 epochs
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    
    return model


@pytest.fixture
def model_artifacts_path():
    """Return a temporary path for model artifacts."""
    path = Path("test_artifacts/models")
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Clean up can be performed here if needed

