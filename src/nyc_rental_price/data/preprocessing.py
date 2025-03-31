(none – file is moved)

import numpy as np
import pandas as pd


def load_data(filepath):
    return pd.read_csv(filepath)


def handle_missing_values(df):
    # Implement missing value handling
    return df


def encode_categorical_variables(df):
    # Implement encoding
    return df


def normalize_features(df):
    # Implement normalization
    return df


def preprocess_data(filepath):
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    df = normalize_features(df)
    return df


if __name__ == "__main__":
    raw_data_path = "data/raw/listings.csv"
    processed_data = preprocess_data(raw_data_path)
    processed_data.to_csv("data/processed/listings_processed.csv", index=False)
