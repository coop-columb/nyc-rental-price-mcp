#!/bin/bash

# Create directories
mkdir -p data/raw data/processed models

# Generate sample data
echo "Generating sample data..."
python -m src.nyc_rental_price.examples.generate_sample_data --num-samples 1000

# Preprocess data
echo "Preprocessing data..."
python -m src.nyc_rental_price.data.preprocessing --input data/raw/sample_listings.csv --output data/processed/sample_processed.csv --engineer-features

# Train model
echo "Training model..."
python -m src.nyc_rental_price.models.train --data-path data/processed/sample_processed.csv --model-type gradient_boosting --output-dir models

echo "Pipeline complete! Model saved to models/gradient_boosting_model.pkl"
