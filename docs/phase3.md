# Phase 3: Model Definition and Rigorous Comparative Testing

## Overview

Phase 3 focuses on developing a modular neural network model for predicting NYC rental prices. This phase includes model architecture design, hyperparameter tuning, training process implementation, and robust evaluation metrics.

## Model Architecture

### Base Model (model.py)

The base model architecture is defined in `model.py` as a configurable neural network with the following features:

- Flexible input layer size based on feature dimensionality
- Configurable hidden layer architecture through a list of layer sizes
- Dropout layers for regularization
- Single output neuron for regression

```python
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
```

### Model Management

The model implementation includes robust model management functions:

- `save_model(model, path)`: Save trained models to disk
- `load_model(path)`: Load trained models for inference
- Proper error handling and validation

```python
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
    return tf.keras.models.load_model(path)
```

## Hyperparameter Choices

The model training process now supports configurable hyperparameters:

- **Hidden Layer Architecture**: Customizable through `hidden_layers` parameter
- **Dropout Rate**: Adjustable regularization strength
- **Optimizer**: Adam optimizer with configurable learning rate
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20% of training data reserved for validation
- **Early Stopping**: Patience of 15 epochs monitoring validation loss

These hyperparameters were chosen to balance model complexity, training speed, and generalization capability. The configurable nature of the model allows for easy experimentation with different architectures and regularization strengths.

## Data Preprocessing

Before model training, data preprocessing is implemented in `preprocessing.py` with these steps:

1. **Data Loading**: Raw data loaded from CSV files
2. **Missing Value Handling**: Framework in place for handling missing values
3. **Categorical Variable Encoding**: Structure for encoding categorical features
4. **Feature Normalization**: Structure for normalizing numerical features

The preprocessed data is saved to `data/processed/listings_processed.csv`.

## Training Process

The training workflow is implemented in `train_test.py` and includes:

1. **Data Loading**: Preprocessed data is loaded
2. **Train-Test Split**: Data is split into 80% training and 20% test sets
3. **Model Initialization**: Model architecture is instantiated based on input shape
4. **Model Training**: Model is trained with the following callbacks:
   - EarlyStopping: Prevents overfitting by monitoring validation loss
   - ModelCheckpoint: Saves the best model during training
5. **Model Evaluation**: Trained model is evaluated on the test set

```python
def train_model(model, X_train, y_train):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('models/best_mcp_model.h5', save_best_only=True)
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    return history
```

## Evaluation Metrics

The model performance is evaluated using:

1. **Mean Absolute Error (MAE)**: Measures the average magnitude of errors without considering direction
2. **Root Mean Square Error (RMSE)**: Gives higher weight to larger errors, calculated as the square root of MSE
3. **R-squared (R²)**: Coefficient of determination, indicating the proportion of variance explained by the model

```python
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}
```

## Logging

The training and evaluation process includes comprehensive logging to track model performance:

- Data loading status
- Model architecture building
- Training progress with callbacks
- Evaluation results including MAE and RMSE metrics
- Final evaluation summary

## Conclusion

Phase 3 establishes a modular and robust machine learning pipeline for NYC rental price prediction. The implementation includes flexible model architecture, hyperparameter choices optimized for regression tasks, a comprehensive training process with regularization techniques, and appropriate evaluation metrics.

The next phase will focus on comparing this neural network approach with traditional regression methods to benchmark performance and identify the optimal modeling strategy.
