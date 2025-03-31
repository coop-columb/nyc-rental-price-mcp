# NYC Rental Price Prediction Examples

This directory contains example scripts demonstrating how to use the NYC rental price prediction system.

## Available Examples

### Sample Data Generator

`generate_sample_data.py` generates synthetic rental listings data for testing:

1. Creates realistic NYC rental listings with prices
2. Includes neighborhoods, amenities, and descriptions
3. Generates data with realistic distributions and relationships

#### Usage

```bash
# Generate 1000 sample listings (default)
python -m src.nyc_rental_price.examples.generate_sample_data

# Generate custom number of listings
python -m src.nyc_rental_price.examples.generate_sample_data --num-samples 5000

# Specify output path
python -m src.nyc_rental_price.examples.generate_sample_data --output-path data/raw/my_sample_data.csv

# Set random seed for reproducibility
python -m src.nyc_rental_price.examples.generate_sample_data --random-state 123
```

### Advanced Model Training

`advanced_model_training.py` demonstrates how to use the advanced ML capabilities:

1. Multiple model types (Gradient Boosting, LightGBM, XGBoost, Neural Networks)
2. Bayesian hyperparameter optimization
3. K-fold cross-validation
4. Model ensembling
5. Feature importance analysis
6. Model comparison

#### Usage

```bash
# Basic usage
python -m src.nyc_rental_price.examples.advanced_model_training --data-path data/processed/listings_processed.csv

# With hyperparameter tuning
python -m src.nyc_rental_price.examples.advanced_model_training --data-path data/processed/listings_processed.csv --tune-hyperparams

# With cross-validation
python -m src.nyc_rental_price.examples.advanced_model_training --data-path data/processed/listings_processed.csv --cross-validate --n-folds 5

# Create model ensemble
python -m src.nyc_rental_price.examples.advanced_model_training --data-path data/processed/listings_processed.csv --create-ensemble

# Full advanced training
python -m src.nyc_rental_price.examples.advanced_model_training --data-path data/processed/listings_processed.csv --tune-hyperparams --cross-validate --create-ensemble
```

## End-to-End Example

Here's how to run a complete end-to-end example:

```bash
# 1. Generate sample data
python -m src.nyc_rental_price.examples.generate_sample_data --num-samples 2000

# 2. Preprocess the data
python -m src.nyc_rental_price.data.preprocessing --input data/raw/sample_listings.csv --output data/processed/sample_processed.csv

# 3. Train and evaluate models
python -m src.nyc_rental_price.examples.advanced_model_training --data-path data/processed/sample_processed.csv --create-ensemble
```

## Adding New Examples

When adding new examples, follow these guidelines:

1. Create a new Python file in this directory
2. Add a detailed docstring explaining the purpose of the example
3. Include command-line arguments for customization
4. Update this README.md with usage instructions