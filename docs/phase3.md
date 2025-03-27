# Phase 3: Model Definition and Rigorous Comparative Testing

## Overview

Phase 3 focuses on developing a modular neural network model for predicting NYC rental prices. This phase includes model architecture design, hyperparameter tuning, training process implementation, and robust evaluation metrics.

## Model Architecture

### Base Model (model.py)

The base model architecture is defined in `model.py` as a sequential neural network with the following layers:

- Input layer: Dense layer with 64 neurons and ReLU activation
- Hidden layer: Dense layer with 32 neurons and ReLU activation
- Output layer: Single neuron for regression (predicting rental price)

```python
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### Enhanced MCP Model (train_test.py)

For the training and evaluation pipeline, we implemented an enhanced version of the model in `train_test.py` with:

- Input layer: Dense layer with 128 neurons and ReLU activation
- Dropout layer (30% dropout rate) for regularization
- Hidden layer: Dense layer with 64 neurons and ReLU activation
- Output layer: Single neuron for regression

```python
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model
```

## Hyperparameter Choices

The model training process incorporates several key hyperparameters:

- **Optimizer**: Adam optimizer with a learning rate of 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20% of training data reserved for validation
- **Early Stopping**: Patience of 15 epochs monitoring validation loss
- **Dropout Rate**: 30% dropout for regularization

These hyperparameters were chosen to balance model complexity, training speed, and generalization capability.

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

```python
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {'mae': mae, 'rmse': rmse}
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
