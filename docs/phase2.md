# Phase 2: Data Preprocessing

## Objectives
- Clean and prepare raw rental listing data for modeling
- Handle missing values, outliers, and inconsistencies
- Transform categorical variables into model-ready formats
- Normalize numerical features to improve model performance

## Preprocessing Steps

### 1. Data Loading and Initial Exploration
- Loaded raw data from `data/raw/listings.csv`
- Performed exploratory data analysis to understand data structure
- Identified key features and potential issues requiring attention

### 2. Handling Missing Values
- Analyzed patterns of missing data across all features
- Applied appropriate strategies based on the nature of missing data:
  - Numerical features: Imputed with median values for skewed distributions, mean for normal distributions
  - Categorical features: Imputed with mode or created "Unknown" category
  - Features with >30% missing values: Evaluated for removal or more sophisticated imputation

### 3. Encoding Categorical Variables
- Identified all categorical features requiring encoding
- Applied one-hot encoding for nominal variables (e.g., neighborhood, building type)
- Applied ordinal encoding for ordered categories (e.g., condition ratings)
- Created binary flags for boolean features

### 4. Feature Normalization
- Applied min-max scaling to bound features between 0 and 1
- Used standardization (z-score normalization) for features used in distance-based calculations
- Log-transformed heavily skewed numerical features (e.g., square footage, certain amenity counts)

### 5. Feature Engineering
- Created interaction terms for potentially related features
- Generated distance-based features (e.g., proximity to subway stations, parks)
- Developed aggregate neighborhood statistics
- Extracted time-based features from listing dates

## Implementation Details
The preprocessing pipeline is implemented in `preprocessing.py` and follows a modular approach:
- `load_data()`: Handles data import
- `handle_missing_values()`: Implements missing data strategies
- `encode_categorical_variables()`: Transforms categorical data
- `normalize_features()`: Scales numerical features
- `preprocess_data()`: Orchestrates the entire preprocessing workflow

## Rationale for Key Decisions
- **Robust scaling over standard scaling**: More resistant to outliers common in NYC rental data
- **Neighborhood grouping**: Consolidated low-frequency neighborhoods to reduce dimensionality
- **Price outlier treatment**: Applied winsorization at 1% and 99% to preserve distribution shape while removing extreme outliers
- **Feature selection**: Removed highly correlated features (>0.85 correlation) to reduce multicollinearity

## Output
- Processed dataset saved to `data/processed/listings_processed.csv`
- Data dimensionality: [X features × Y samples] after preprocessing
- All features ready for direct input to modeling phase

## Next Steps
The preprocessed data will be used in Phase 3 for model development and evaluation.

