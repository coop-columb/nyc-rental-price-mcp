# NYC Rental Price Prediction Project

This project provides a machine learning model for predicting NYC rental prices.

## Overview

The project has been restructured to improve clarity and maintainability. The new structure is as follows:

Root Directory:

- README.md              : Project overview and structure description.
- docs/                  : Additional documentation.
- data/                  : Raw and processed data.
- experiments/           : Exploratory notebooks and analysis.
- src/                   : Core application code.

Inside src/:

- api/                   : API endpoints and server logic.
- data_processing/       : Modules for data collection, scraping, and preprocessing.
- models/                : Model training, inference, and related utilities.

## Getting Started

Follow instructions in each directory's README to set up and run the components.

## Project Details

### Project Overview

This project implements a machine learning system for predicting rental prices in New York City. Using historical rental data, the system leverages neural networks to provide accurate price predictions based on apartment features and location data.

### Key Features

- **Data Processing Pipeline**: Robust preprocessing of NYC rental data with missing value handling
- **Neural Network Model**: TensorFlow/Keras-based sequential model with dense layers
- **Model Training**: Complete pipeline with early stopping and checkpointing
- **REST API**: FastAPI implementation for real-time price predictions
- **Testing Framework**: Comprehensive unit, integration, and performance tests

### New Features

- **Enhanced Scraper**: Robust web scraping with error handling and data validation
- **Neural Network Model**: Configurable architecture with dropout regularization
- **Comprehensive Testing**: Organized test suite with fixtures and mocks
- **Structured Data Processing**: Standardized pipeline for data cleaning and feature engineering

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment tool (venv, conda, etc.)
- TensorFlow 2.x
- scikit-learn

### Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/nyc-rental-price-mcp.git
   cd nyc-rental-price-mcp
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:

   ```bash
   pip install -e .
   ```

4. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

## Project Structure

```bash
nyc-rental-price-mcp/
├── data/                      # Data storage
│   ├── raw/                   # Original, immutable data
│   ├── interim/               # Intermediate processing data
│   └── processed/             # Final data for modeling
├── models/                    # Trained model files
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code package
│   └── nyc_rental_price/      # Main package
│       ├── data/              # Data processing modules
│       │   ├── preprocessing.py  # Data cleaning and transformation
│       │   └── scraper.py     # Data collection utilities
│       ├── models/            # Model definition and training
│       │   ├── model.py       # Neural network architecture
│       │   └── train.py       # Training and evaluation pipeline
│       ├── features/          # Feature engineering
│       └── api/               # API implementation
│           └── main.py        # FastAPI endpoints
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── performance/           # Performance tests
├── docs/                      # Documentation
├── setup.py                   # Package installation configuration
├── pyproject.toml             # Development tool configuration
└── README.md                  # Project documentation
```

## Usage Examples

### Data Processing

```python
from nyc_rental_price.data.preprocessing import preprocess_rental_data

# Load and preprocess data
processed_data = preprocess_rental_data('data/raw/rental_listings.csv')
processed_data.to_csv('data/processed/cleaned_data.csv', index=False)
```

### Model Training

```python
from nyc_rental_price.models.train import train_model
from nyc_rental_price.data.preprocessing import load_processed_data

# Load processed data
X_train, X_test, y_train, y_test = load_processed_data('data/processed/cleaned_data.csv')

# Train the model
model, history, metrics = train_model(
    X_train, y_train, X_test, y_test,
    epochs=100,
    batch_size=32,
    model_path='models/rental_price_model.h5'
)

print(f"Model evaluation: {metrics}")
```

## Model Components

The project includes a robust neural network model implementation:

```python
from nyc_rental_price.models.model import create_model

# Create a model with custom architecture
model = create_model(
    input_dim=10,
    hidden_layers=[64, 32],
    dropout_rate=0.2
)
```

### Using the API

#### Running the API server

```bash
uvicorn nyc_rental_price.api.main:app --reload
```

#### Making predictions

```python
import requests
import json

# Example rental property data
property_data = {
    "bedrooms": 2,
    "bathrooms": 1,
    "sqft": 850,
    "neighborhood": "Williamsburg",
    "has_elevator": True,
    "has_doorman": False,
    "has_dishwasher": True,
    "has_laundry": True
}

# Send request to API
response = requests.post(
    "http://localhost:8000/predict",
    json=property_data
)

# View predicted price
result = response.json()
print(f"Predicted monthly rent: ${result['predicted_price']:.2f}")
```

## Development Guidelines

### Setting Up Development Environment

1. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

### Code Style

This project uses:

- Black for code formatting
- Flake8 for linting
- Type hints for better code quality

Run formatting and linting:

```bash
black src tests
flake8 src tests
```

### Testing

Run tests:

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit
pytest tests/integration
pytest tests/performance
```

## Test Organization

```bash
tests/
├── unit/               # Unit tests for individual components
│   ├── test_scraper.py
│   └── test_model.py
├── integration/        # Integration tests
├── performance/        # Performance benchmarks
└── conftest.py         # Shared test fixtures
# Code block with blank lines around it

### Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Add or update tests as necessary
4. Ensure all tests pass
5. Submit a pull request

### Documentation

When adding new features, be sure to:

- Add docstrings to all functions and classes
- Update the README.md if necessary
- Consider adding a notebook example if applicable
