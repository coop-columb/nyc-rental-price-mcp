import pytest
import pandas as pd
import numpy as np

from nyc_rental_price.data.preprocessing import (
    preprocess_data,
    handle_missing_values,
    encode_categorical_features,
    normalize_numerical_features
)


def test_preprocess_data(sample_data):
    """Test that preprocessing returns the expected X and y data structures."""
    X, y = preprocess_data(sample_data)
    
    # Check that X is a DataFrame or numpy array with the right shape
    assert X is not None
    assert X.shape[0] == sample_data.shape[0]  # Same number of samples
    
    # Check that y contains the target values
    assert y is not None
    assert len(y) == sample_data.shape[0]  # Same number of target values
    
    # Check that the target column is not in X
    if isinstance(X, pd.DataFrame):
        assert 'price' not in X.columns


def test_handle_missing_values():
    """Test handling of missing values in the dataset."""
    # Create test data with missing values
    data = pd.DataFrame({
        'bedrooms': [1, np.nan, 3, 2],
        'bathrooms': [1.0, 2.0, np.nan, 1.5],
        'neighborhood': ['Manhattan', None, 'Brooklyn', 'Queens'],
        'price': [2500, 3000, 3500, 2800]
    })
    
    # Process the data
    processed_data = handle_missing_values(data)
    
    # Check that there are no more missing values
    assert processed_data.isnull().sum().sum() == 0
    
    # Check that the data has the same number of rows
    assert len(processed_data) == len(data)


def test_encode_categorical_features():
    """Test encoding of categorical features."""
    # Create test data with categorical features
    data = pd.DataFrame({
        'neighborhood': ['Manhattan', 'Brooklyn', 'Queens', 'Manhattan', 'Bronx'],
        'building_type': ['Condo', 'Co-op', 'Rental', 'Condo', 'Rental']
    })
    
    # Encode the categorical features
    encoded_data = encode_categorical_features(data)
    
    # Check that the categorical columns are encoded (e.g., with one-hot encoding)
    assert 'neighborhood' not in encoded_data.columns  # Original column should be removed
    assert 'building_type' not in encoded_data.columns  # Original column should be removed
    
    # There should be new columns for the categories
    assert encoded_data.shape[1] > data.shape[1]
    
    # Check that all values are now numeric
    assert encoded_data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()


def test_normalize_numerical_features():
    """Test normalization of numerical features."""
    # Create test data with numerical features
    data = pd.DataFrame({
        'bedrooms': [1, 2, 3, 4, 5],
        'sqft': [500, 750, 1000, 1250, 1500],
        'price': [2000, 2500, 3000, 3500, 4000]
    })
    
    # Normalize the numerical features
    normalized_data = normalize_numerical_features(data)
    
    # Check that the data has been normalized (values between 0 and 1 or -1 and 1)
    for col in ['bedrooms', 'sqft']:
        if col in normalized_data.columns:
            assert normalized_data[col].min() >= -1
            assert normalized_data[col].max() <= 1
    
    # Price column should remain unchanged if it's the target
    if 'price' in normalized_data.columns:
        assert normalized_data['price'].equals(data['price'])

