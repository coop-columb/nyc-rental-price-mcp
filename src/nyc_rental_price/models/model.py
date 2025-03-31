(none – file is moved)

"""Model creation and management for NYC rental price prediction."""

from tensorflow.keras import models, layers


def create_model(
    input_dim, hidden_layers=[64, 32], dropout_rate=0.2, activation="relu"
):
    """Create a neural network model for rental price prediction.

    Args:
        input_dim (int): Number of input features
        hidden_layers (list): List of integers for number of units in each
            hidden layer
        dropout_rate (float): Dropout rate between layers
        activation (str): Activation function to use for hidden layers

    Returns:
        tensorflow.keras.Model: Neural network model (uncompiled)
    """
    model = models.Sequential()

    # Input layer
    model.add(
        layers.Dense(
            hidden_layers[0],
            activation=activation,
            input_dim=input_dim,
        )
    )
    model.add(layers.Dropout(dropout_rate))

    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(1))  # Single output for price prediction

    return model


def compile_model(model, optimizer="adam", loss="mse", metrics=["mae"]):
    """Compile a neural network model with specified parameters.

    Args:
        model (tensorflow.keras.Model): The model to compile
        optimizer (str or tensorflow.keras.optimizers.Optimizer):
            Optimizer to use
        loss (str or tensorflow.keras.losses.Loss): Loss function
        metrics (list): List of metrics to track

    Returns:
        tensorflow.keras.Model: Compiled neural network model
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def save_model(model, path):
    """Save a trained model to disk.

    Args:
        model (tensorflow.keras.Model): Trained model to save
        path (str): Path where to save the model
    """
    model.save(path)


def load_model(path):
    """Load a trained model from disk.

    Args:
        path (str): Path to the saved model

    Returns:
        tensorflow.keras.Model: Loaded model
    """
    return models.load_model(path)


def build_model(input_shape):
    """Build a neural network model with the specified input shape.

    Args:
        input_shape (int): Number of input features

    Returns:
        tensorflow.keras.Model: Compiled neural network model
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
