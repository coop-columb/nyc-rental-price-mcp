# NYC Rental Price Prediction Project

This project provides a machine learning system for predicting NYC rental prices using advanced ML techniques.

## Overview

The NYC Rental Price Prediction system is a comprehensive machine learning solution that predicts rental prices in New York City based on property features, location, and market trends. The system uses advanced feature engineering and ensemble modeling techniques to provide accurate price predictions.

## Key Features

- **Multi-source Data Collection**: Scrapes rental listings from StreetEasy, Zillow, and Craigslist
- **Advanced Feature Engineering**:
  - Neighborhood embeddings
  - Distance-based features (proximity to subway, parks, etc.)
  - Text feature extraction from descriptions
  - Temporal features (seasonality, listing age)
  - Target encoding for categorical variables
- **Sophisticated ML Models**:
  - Gradient boosting ensemble (XGBoost, LightGBM)
  - Neural networks with embedding layers
  - Model stacking approach
  - Bayesian hyperparameter optimization
  - K-fold cross-validation
- **Comprehensive Evaluation**:
  - Cross-validation with stratification
  - Feature importance analysis
  - Error analysis by neighborhood and price range
- **Production-ready API**:
  - FastAPI implementation with validation
  - Confidence intervals for predictions
  - Explanation endpoints
  - Batch prediction support

## Project Structure

```bash
nyc-rental-price-mcp/
├── data/                      # Data storage
│   ├── raw/                   # Original, immutable data
│   ├── interim/               # Intermediate processing data
│   └── processed/             # Final data for modeling
├── models/                    # Trained model files
├── src/                       # Source code package
│   └── nyc_rental_price/      # Main package
│       ├── data/              # Data processing modules
│       │   ├── preprocessing.py  # Data cleaning and transformation
│       │   ├── scrape_listings.py # Data collection CLI
│       │   └── scrapers/      # Web scrapers for different sources
│       │       ├── base.py    # Base scraper functionality
│       │       ├── streeteasy.py # StreetEasy-specific scraper
│       │       ├── zillow.py  # Zillow-specific scraper
│       │       └── craigslist.py # Craigslist-specific scraper
│       ├── features/          # Feature engineering
│       │   ├── neighborhood.py # Neighborhood embeddings
│       │   ├── distance.py    # Distance-based features
│       │   ├── text.py        # Text feature extraction
│       │   ├── temporal.py    # Time-based features
│       │   ├── encoder.py     # Target encoding
│       │   └── pipeline.py    # Feature pipeline orchestration
│       ├── models/            # Model definition and training
│       │   ├── model.py       # Model implementations
│       │   └── train.py       # Training and evaluation pipeline
│       ├── examples/          # Example scripts
│       │   ├── generate_sample_data.py # Sample data generation
│       │   └── advanced_model_training.py # Advanced ML example
│       ├── api/               # API implementation
│       │   └── main.py        # FastAPI endpoints
│       └── main.py            # Main entry point for running the pipeline
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── performance/           # Performance tests
├── docs/                      # Documentation
├── setup.py                   # Package installation configuration
├── pyproject.toml             # Development tool configuration
└── README.md                  # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment tool (venv, conda, etc.)

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

## Usage

### Sample Data Generation

For testing without real data, generate synthetic NYC rental listings:

```bash
python -m src.nyc_rental_price.examples.generate_sample_data \
    --num-samples=2000 \
    --output-path=data/raw/sample_listings.csv
```

### Data Collection

Collect rental listings from various sources:

```bash
python -m src.nyc_rental_price.data.scrape_listings \
    --sources=streeteasy,zillow,craigslist \
    --output-dir=data/raw \
    --max-pages=5
```

### Data Preprocessing

Process the collected data:

```bash
python -m src.nyc_rental_price.data.preprocessing \
    --input=data/raw \
    --output=data/processed/listings_processed.csv
```

### Model Training

#### Basic Training

Train a prediction model:

```bash
python -m src.nyc_rental_price.models.train \
    --data-path=data/processed/listings_processed.csv \
    --model-type=ensemble \
    --output-dir=models
```

#### Advanced Training

Train with advanced ML techniques:

```bash
python -m src.nyc_rental_price.examples.advanced_model_training \
    --data-path=data/processed/listings_processed.csv \
    --tune-hyperparams \
    --cross-validate \
    --create-ensemble
```

### Running the API

Start the prediction API:

```bash
python -m src.nyc_rental_price.api.main \
    --port=8000
```

### Running the Complete Pipeline

Run the entire pipeline with a single command:

```bash
python -m src.nyc_rental_price.main --all
```

Or run specific steps:

```bash
python -m src.nyc_rental_price.main \
    --collect-data \
    --sources=streeteasy,zillow \
    --preprocess-data \
    --train-model \
    --model-type=ensemble
```

## Example Workflows

### Quick Start with Sample Data

For a quick start using synthetic data:

```bash
# 1. Generate sample data
python -m src.nyc_rental_price.examples.generate_sample_data --num-samples 2000

# 2. Preprocess the data
python -m src.nyc_rental_price.data.preprocessing \
    --input data/raw/sample_listings.csv \
    --output data/processed/sample_processed.csv

# 3. Train and evaluate models
python -m src.nyc_rental_price.examples.advanced_model_training \
    --data-path data/processed/sample_processed.csv \
    --create-ensemble
```

### Advanced Model Comparison

Compare different model types:

```bash
# Train with multiple models and create comparison visualizations
python -m src.nyc_rental_price.examples.advanced_model_training \
    --data-path data/processed/listings_processed.csv \
    --output-dir models/comparison \
    --cross-validate \
    --create-ensemble
```

## API Usage

### Making Predictions

```python
import requests

# Example rental property data
property_data = {
    "bedrooms": 2,
    "bathrooms": 1,
    "sqft": 850,
    "neighborhood": "Williamsburg",
    "has_doorman": False,
    "has_elevator": True,
    "has_dishwasher": True,
    "has_washer_dryer": False,
    "is_furnished": False
}

# Send request to API
response = requests.post(
    "http://localhost:8000/predict",
    json=property_data
)

# View predicted price
result = response.json()
print(f"Predicted monthly rent: ${result['predicted_price']:.2f}")
print(f"Confidence interval: ${result['confidence_interval']['lower_bound']:.2f} - ${result['confidence_interval']['upper_bound']:.2f}")
```

### Batch Predictions

```python
import requests

# Multiple properties
properties = {
    "properties": [
        {
            "bedrooms": 2,
            "bathrooms": 1,
            "sqft": 850,
            "neighborhood": "Williamsburg"
        },
        {
            "bedrooms": 1,
            "bathrooms": 1,
            "sqft": 600,
            "neighborhood": "East Village"
        }
    ]
}

# Send batch request
response = requests.post(
    "http://localhost:8000/batch-predict",
    json=properties
)

# View results
results = response.json()
for i, prediction in enumerate(results["predictions"]):
    print(f"Property {i+1}: ${prediction['predicted_price']:.2f}")
```

### Getting Prediction Explanations

```python
import requests

# Property data
property_data = {
    "bedrooms": 2,
    "bathrooms": 1,
    "sqft": 850,
    "neighborhood": "Williamsburg",
    "has_doorman": False,
    "has_elevator": True
}

# Get explanation
response = requests.post(
    "http://localhost:8000/explain",
    json=property_data
)

# View explanation
explanation = response.json()
for feature, contribution in explanation["feature_contributions"].items():
    print(f"{feature}: ${contribution:.2f}")
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
- Ruff for linting
- Type hints for better code quality

Run formatting and linting:

```bash
black src tests
ruff src tests
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

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Add or update tests as necessary
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.