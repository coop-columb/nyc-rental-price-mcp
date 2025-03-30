"""Model creation and management for NYC rental price prediction."""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as tf_load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_dim, hidden_layers=[64, 32], dropout_rate=0.2):
    """Create a neural network model for rental price prediction.
    
    Args:
        input_dim (int): Number of input features
        hidden_layers (list): List of integers for number of units in each hidden layer
        dropout_rate (float): Dropout rate between layers
    
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))  # Single output for price prediction
    
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def save_model(model, path):
    """Save a trained model to disk.
    
    Args:
        model (tf.keras.Model): Trained model to save
        path (str): Path where to save the model
    """
    model.save(path)

def load_model(path):
    """Load a trained model from disk.
    
    Args:
        path (str): Path to the saved model
    
    Returns:
        tf.keras.Model: Loaded model
    """
    return tf_load_model(path)

import tensorflow as tf

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
