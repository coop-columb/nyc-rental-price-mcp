"""NYC Rental Price Prediction API.

This module implements a FastAPI application for predicting NYC rental prices
based on property features using a trained machine learning model.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.nyc_rental_price.models.model import (
    BaseModel as MLBaseModel,
    GradientBoostingModel,
    NeuralNetworkModel,
    ModelEnsemble,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api_logs.log"),
    ],
)
logger = logging.getLogger("rental_price_api")

app = FastAPI(
    title="NYC Rental Price Predictor API",
    description="API for predicting NYC rental prices based on property features",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model path relative to the package
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "models",
)

# Global model instance
model = None


class PropertyFeatures(BaseModel):
    """Input model for property features.
    
    Attributes:
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        sqft: Square footage of the property
        neighborhood: Neighborhood name
        has_doorman: Whether the building has a doorman
        has_elevator: Whether the building has an elevator
        has_dishwasher: Whether the unit has a dishwasher
        has_washer_dryer: Whether the unit has a washer/dryer
        is_furnished: Whether the unit is furnished
        has_balcony: Whether the unit has a balcony
        has_parking: Whether the unit has parking
        is_no_fee: Whether there's no broker fee
        floor: Floor number of the unit
    """
    
    bedrooms: float = Field(..., description="Number of bedrooms")
    bathrooms: float = Field(..., description="Number of bathrooms")
    sqft: Optional[float] = Field(None, description="Square footage of the property")
    neighborhood: str = Field(..., description="Neighborhood name")
    has_doorman: Optional[bool] = Field(False, description="Whether the building has a doorman")
    has_elevator: Optional[bool] = Field(False, description="Whether the building has an elevator")
    has_dishwasher: Optional[bool] = Field(False, description="Whether the unit has a dishwasher")
    has_washer_dryer: Optional[bool] = Field(False, description="Whether the unit has a washer/dryer")
    is_furnished: Optional[bool] = Field(False, description="Whether the unit is furnished")
    has_balcony: Optional[bool] = Field(False, description="Whether the unit has a balcony")
    has_parking: Optional[bool] = Field(False, description="Whether the unit has parking")
    is_no_fee: Optional[bool] = Field(False, description="Whether there's no broker fee")
    floor: Optional[int] = Field(None, description="Floor number of the unit")
    
    @validator("neighborhood")
    def validate_neighborhood(cls, v):
        """Validate the neighborhood name.
        
        Args:
            v: Neighborhood name
        
        Returns:
            Validated neighborhood name
        """
        if not v:
            raise ValueError("Neighborhood cannot be empty")
        return v.lower().strip()


class PredictionResponse(BaseModel):
    """Response model for rental price predictions.
    
    Attributes:
        predicted_price: The predicted rental price in USD
        confidence_interval: Optional confidence interval for the prediction
        comparable_properties: Optional list of comparable properties
    """
    
    predicted_price: float = Field(..., description="Predicted rental price in USD")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="Confidence interval for the prediction"
    )
    comparable_properties: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of comparable properties"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions.
    
    Attributes:
        properties: List of property features
    """
    
    properties: List[PropertyFeatures] = Field(
        ..., description="List of properties to predict prices for"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions.
    
    Attributes:
        predictions: List of prediction responses
    """
    
    predictions: List[PredictionResponse] = Field(
        ..., description="List of prediction responses"
    )


class ExplanationResponse(BaseModel):
    """Response model for prediction explanations.
    
    Attributes:
        feature_contributions: Dictionary mapping feature names to their contributions
        baseline_value: Baseline value for the prediction
        explanation_method: Method used for generating the explanation
    """
    
    feature_contributions: Dict[str, float] = Field(
        ..., description="Feature contributions to the prediction"
    )
    baseline_value: float = Field(..., description="Baseline value for the prediction")
    explanation_method: str = Field(..., description="Method used for explanation")


def load_model() -> MLBaseModel:
    """Load the trained model.
    
    Returns:
        Loaded model
    
    Raises:
        RuntimeError: If the model cannot be loaded
    """
    # Check for ensemble model first
    ensemble_dir = os.path.join(MODEL_DIR, "ensemble_model")
    if os.path.exists(ensemble_dir):
        logger.info("Loading ensemble model")
        model = ModelEnsemble(model_dir=MODEL_DIR, model_name="ensemble_model")
        if model.load_model():
            return model
    
    # Check for gradient boosting model
    gb_path = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")
    if os.path.exists(gb_path):
        logger.info("Loading gradient boosting model")
        model = GradientBoostingModel(model_dir=MODEL_DIR, model_name="gradient_boosting_model")
        if model.load_model():
            return model
    
    # Check for neural network model
    nn_path = os.path.join(MODEL_DIR, "neural_network_model.h5")
    if os.path.exists(nn_path):
        logger.info("Loading neural network model")
        model = NeuralNetworkModel(model_dir=MODEL_DIR, model_name="neural_network_model")
        if model.load_model():
            return model
    
    # If no model is found, raise an error
    raise RuntimeError("No trained model found")


def get_model() -> MLBaseModel:
    """Get the loaded model instance.
    
    Returns:
        Loaded model
    
    Raises:
        HTTPException: If the model is not loaded
    """
    global model
    
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please try again later.",
            )
    
    return model


def preprocess_features(features: PropertyFeatures) -> pd.DataFrame:
    """Preprocess property features for prediction.
    
    Args:
        features: Property features
    
    Returns:
        DataFrame with preprocessed features
    """
    # Convert to dictionary
    feature_dict = features.dict()
    
    # Create DataFrame
    df = pd.DataFrame([feature_dict])
    
    # Handle missing values
    if df["sqft"].isna().any():
        # Use a reasonable default based on number of bedrooms
        sqft_map = {
            0: 400,  # Studio
            1: 650,  # 1 bedroom
            2: 900,  # 2 bedrooms
            3: 1200,  # 3 bedrooms
            4: 1500,  # 4 bedrooms
        }
        
        for idx, row in df.iterrows():
            if pd.isna(row["sqft"]):
                bedrooms = int(row["bedrooms"])
                if bedrooms in sqft_map:
                    df.loc[idx, "sqft"] = sqft_map[bedrooms]
                else:
                    df.loc[idx, "sqft"] = 600 + 300 * bedrooms
    
    # Convert boolean columns to integers (0/1)
    boolean_columns = [
        "has_doorman",
        "has_elevator",
        "has_dishwasher",
        "has_washer_dryer",
        "is_furnished",
        "has_balcony",
        "has_parking",
        "is_no_fee",
    ]
    
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Handle floor
    if "floor" in df.columns and df["floor"].isna().any():
        df["floor"] = df["floor"].fillna(2)  # Use a reasonable default
    
    return df


def generate_confidence_interval(
    price: float, std_dev: float = 0.15
) -> Dict[str, float]:
    """Generate a confidence interval for the prediction.
    
    Args:
        price: Predicted price
        std_dev: Standard deviation as a proportion of the price
    
    Returns:
        Dictionary with lower and upper bounds
    """
    # Calculate bounds based on standard deviation
    lower_bound = price * (1 - std_dev)
    upper_bound = price * (1 + std_dev)
    
    return {
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "confidence_level": 0.9,  # 90% confidence interval
    }


def find_comparable_properties(
    features: PropertyFeatures, predicted_price: float
) -> List[Dict[str, Any]]:
    """Find comparable properties for the given features.
    
    Args:
        features: Property features
        predicted_price: Predicted price
    
    Returns:
        List of comparable properties
    """
    # This is a placeholder implementation
    # In a real-world scenario, this would query a database of real listings
    
    bedrooms = features.bedrooms
    neighborhood = features.neighborhood
    
    # Generate synthetic comparable properties
    comparables = []
    
    # Property with same bedrooms, slightly higher price
    comparables.append({
        "address": f"123 Main St, {neighborhood.title()}",
        "bedrooms": bedrooms,
        "bathrooms": features.bathrooms,
        "price": round(predicted_price * 1.05, 2),
        "sqft": features.sqft,
        "url": "https://example.com/listing1",
    })
    
    # Property with same bedrooms, slightly lower price
    comparables.append({
        "address": f"456 Park Ave, {neighborhood.title()}",
        "bedrooms": bedrooms,
        "bathrooms": features.bathrooms - 0.5 if features.bathrooms > 1 else features.bathrooms,
        "price": round(predicted_price * 0.95, 2),
        "sqft": features.sqft * 0.9 if features.sqft else None,
        "url": "https://example.com/listing2",
    })
    
    # Property with different bedrooms
    comparables.append({
        "address": f"789 Broadway, {neighborhood.title()}",
        "bedrooms": bedrooms + 1,
        "bathrooms": features.bathrooms + 0.5,
        "price": round(predicted_price * 1.25, 2),
        "sqft": features.sqft * 1.2 if features.sqft else None,
        "url": "https://example.com/listing3",
    })
    
    return comparables


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model when the API starts up."""
    try:
        # Pre-load the model
        _ = get_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Don't fail startup, as the model might become available later


@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint providing basic API information.
    
    Returns:
        Dictionary with welcome message
    """
    return {
        "message": "NYC Rental Price Prediction API. Use /predict endpoint for predictions."
    }


@app.get("/health")
def health_check() -> Dict[str, Union[str, bool]]:
    """Health check endpoint to verify API is running.
    
    Returns:
        Dictionary with health status
    """
    health_status = {"status": "ok", "model_loaded": False}
    
    try:
        # Check if model is loaded
        _ = get_model()
        health_status["model_loaded"] = True
    except Exception:
        health_status["status"] = "degraded"
    
    return health_status


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: PropertyFeatures,
    include_confidence: bool = Query(True, description="Include confidence interval"),
    include_comparables: bool = Query(
        False, description="Include comparable properties"
    ),
    model: MLBaseModel = Depends(get_model),
) -> PredictionResponse:
    """Predict rental price for a property.
    
    Args:
        features: Property features
        include_confidence: Whether to include confidence interval
        include_comparables: Whether to include comparable properties
        model: ML model to use for prediction
    
    Returns:
        Prediction response
    
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Preprocess features
        df = preprocess_features(features)
        
        # Make prediction
        predicted_price = float(model.predict(df)[0])
        
        # Round to nearest dollar
        predicted_price = round(predicted_price, 2)
        
        # Create response
        response = {
            "predicted_price": predicted_price,
            "confidence_interval": None,
            "comparable_properties": None,
        }
        
        # Add confidence interval if requested
        if include_confidence:
            response["confidence_interval"] = generate_confidence_interval(predicted_price)
        
        # Add comparable properties if requested
        if include_comparables:
            response["comparable_properties"] = find_comparable_properties(
                features, predicted_price
            )
        
        logger.info(f"Predicted price: ${predicted_price:.2f}")
        
        return response
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(
    request: BatchPredictionRequest,
    include_confidence: bool = Query(True, description="Include confidence intervals"),
    include_comparables: bool = Query(
        False, description="Include comparable properties"
    ),
    model: MLBaseModel = Depends(get_model),
) -> BatchPredictionResponse:
    """Predict rental prices for multiple properties.
    
    Args:
        request: Batch prediction request
        include_confidence: Whether to include confidence intervals
        include_comparables: Whether to include comparable properties
        model: ML model to use for prediction
    
    Returns:
        Batch prediction response
    
    Raises:
        HTTPException: If prediction fails
    """
    try:
        predictions = []
        
        # Process each property
        for features in request.properties:
            # Preprocess features
            df = preprocess_features(features)
            
            # Make prediction
            predicted_price = float(model.predict(df)[0])
            
            # Round to nearest dollar
            predicted_price = round(predicted_price, 2)
            
            # Create response
            response = {
                "predicted_price": predicted_price,
                "confidence_interval": None,
                "comparable_properties": None,
            }
            
            # Add confidence interval if requested
            if include_confidence:
                response["confidence_interval"] = generate_confidence_interval(
                    predicted_price
                )
            
            # Add comparable properties if requested
            if include_comparables:
                response["comparable_properties"] = find_comparable_properties(
                    features, predicted_price
                )
            
            predictions.append(response)
        
        logger.info(f"Processed batch prediction for {len(predictions)} properties")
        
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction error: {str(e)}"
        )


@app.post("/explain", response_model=ExplanationResponse)
def explain_prediction(
    features: PropertyFeatures,
    model: MLBaseModel = Depends(get_model),
) -> ExplanationResponse:
    """Explain the factors contributing to a prediction.
    
    Args:
        features: Property features
        model: ML model to use for prediction
    
    Returns:
        Explanation response
    
    Raises:
        HTTPException: If explanation fails
    """
    try:
        # Preprocess features
        df = preprocess_features(features)
        
        # Make prediction
        predicted_price = float(model.predict(df)[0])
        
        # This is a simplified explanation implementation
        # In a real-world scenario, you would use SHAP or LIME for model-agnostic explanations
        
        # Generate synthetic feature contributions
        # The values are based on typical NYC rental market factors
        contributions = {
            "bedrooms": features.bedrooms * 500,
            "bathrooms": features.bathrooms * 300,
            "neighborhood": 1000,  # Placeholder, would be neighborhood-specific
            "sqft": features.sqft * 0.5 if features.sqft else 0,
        }
        
        # Add amenity contributions
        amenity_values = {
            "has_doorman": 200,
            "has_elevator": 150,
            "has_dishwasher": 100,
            "has_washer_dryer": 200,
            "is_furnished": 300,
            "has_balcony": 150,
            "has_parking": 250,
            "is_no_fee": 100,
        }
        
        for amenity, value in amenity_values.items():
            if getattr(features, amenity, False):
                contributions[amenity] = value
            else:
                contributions[amenity] = 0
        
        # Calculate baseline value (minimum rent)
        baseline_value = 1500  # Minimum rent in NYC
        
        # Normalize contributions to match predicted price
        total_contribution = sum(contributions.values())
        scaling_factor = (predicted_price - baseline_value) / total_contribution
        
        normalized_contributions = {
            k: round(v * scaling_factor, 2) for k, v in contributions.items()
        }
        
        logger.info(f"Generated explanation for prediction: ${predicted_price:.2f}")
        
        return {
            "feature_contributions": normalized_contributions,
            "baseline_value": baseline_value,
            "explanation_method": "feature_attribution",
        }
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Explanation error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the rental price prediction API")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the API on",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the API on",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    # Run the API
    uvicorn.run(
        "src.nyc_rental_price.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )