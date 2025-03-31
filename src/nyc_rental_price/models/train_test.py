(none – file is moved)

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_processed_data(filepath):
    logging.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)


def split_data(df, target="price"):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info("Data split into training and testing sets")
    return X_train, X_test, y_train, y_test


def build_model(input_shape):
    logging.info("Building MCP model architecture")
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(model, X_train, y_train):
    logging.info("Training model with EarlyStopping and ModelCheckpoint callbacks")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_mcp_model.h5", save_best_only=True
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model performance")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    logging.info(f"Evaluation Results - MAE: {mae}, RMSE: {rmse}")
    return {"mae": mae, "rmse": rmse}


if __name__ == "__main__":
    df = load_processed_data("data/processed/listings_processed.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_model(X_train.shape[1])
    train_model(model, X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    logging.info(f"Final evaluation metrics: {results}")
