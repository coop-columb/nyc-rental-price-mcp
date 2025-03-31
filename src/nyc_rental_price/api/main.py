(none – file is moved)

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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("api_logs.log")],
)
logger = logging.getLogger("rental_price_api")

app = FastAPI(
    title="NYC Rental Price Predictor API",
    description="API for predicting NYC rental prices based on property features",
    version="1.0.0",
)

# Define model path relative to the package
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                         "models/best_mcp_model.h5")


class PredictionRequest(BaseModel):
    """Request model for rental price predictions.
    
    Attributes:
        features: List of float values representing property features.
    """
    features: List[float]

    @validator("features")
    def validate_features(cls, features: List[float]) -> List[float]:
        """Validate the features list is not empty.
        
        Args:
            features: The list of features to validate
            
        Returns:
            The validated features list
            
        Raises:
            ValueError: If the features list is empty
        """
        # This will be updated once we know the exact feature count expected by the model
        if len(features) < 1:
            raise ValueError("Features list cannot be empty")
        return features


class PredictionResponse(BaseModel):
    """Response model for rental price predictions.
    
    Attributes:
        predicted_price: The predicted rental price in USD.
        confidence: Optional confidence score for the prediction.
    """
    predicted_price: float
    confidence: Optional[float] = None


# Load model on startup if it exists
@app.on_event("startup")
async def startup_event() -> None:
    """Load the ML model when the API starts up.
    
    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    if os.path.exists(MODEL_PATH):
        logger.info("Loading model from %s", MODEL_PATH)
        try:
            global model
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model: %s", str(e))
            raise RuntimeError(f"Could not load model: {str(e)}")
    else:
        logger.warning(
            "Model file %s not found. API will not be able to make predictions until model is available.",
            MODEL_PATH,
        )


@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint providing basic API information.
    
    Returns:
        A dictionary with a welcome message.
    """
    return {
        "message": "NYC Rental Price Prediction API. Use /predict endpoint for predictions."
    }


@app.get("/health")
def health_check() -> Dict[str, Union[str, bool]]:
    """Health check endpoint to verify API is running.
    
    Returns:
        A dictionary containing the API health status and model loading state.
    """
    health_status = {"status": "ok", "model_loaded": False}

    try:
        if "model" in globals():
            health_status["model_loaded"] = True
    except:
        pass

    return health_status


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Make a rental price prediction based on provided features.
    
    Args:
        request: A PredictionRequest object containing the property features.
        
    Returns:
        A PredictionResponse object containing the predicted price.
        
    Raises:
        HTTPException: If the model is not loaded or if there's an error making the prediction.
    """
    logger.info("Received prediction request")

    # Check if model is loaded
    if "model" not in globals():
        logger.error("Model not loaded, cannot make prediction")
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later."
        )

    try:
        # Convert features to DataFrame
        # Note: Column names would need to be added if the model expects them
        input_df = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(input_df).flatten()[0]

        # Log and return the prediction
        logger.info(f"Prediction made: {prediction}")
        return {"predicted_price": float(prediction)}

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

