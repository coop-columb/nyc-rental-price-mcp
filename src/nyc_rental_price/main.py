#!/usr/bin/env python3
"""NYC Rental Price Prediction Pipeline.

This script runs the entire NYC rental price prediction pipeline:
1. Data collection
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation
6. API deployment

Example usage:
    python -m src.nyc_rental_price.main --collect-data --train-model --start-api
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("nyc_rental_price.log"),
    ],
)
logger = logging.getLogger("nyc_rental_price")


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the NYC rental price prediction pipeline"
    )
    
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Collect rental listings data",
    )
    
    parser.add_argument(
        "--sources",
        type=str,
        default="streeteasy,zillow,craigslist",
        help="Comma-separated list of data sources",
    )
    
    parser.add_argument(
        "--preprocess-data",
        action="store_true",
        help="Preprocess collected data",
    )
    
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train prediction model",
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble",
        choices=["gradient_boosting", "neural_network", "ensemble"],
        help="Type of model to train",
    )
    
    parser.add_argument(
        "--start-api",
        action="store_true",
        help="Start the prediction API",
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the API server",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the entire pipeline",
    )
    
    return parser.parse_args()


def collect_data(sources):
    """Collect rental listings data.
    
    Args:
        sources: Comma-separated list of data sources
    
    Returns:
        True if data collection was successful, False otherwise
    """
    logger.info(f"Collecting data from sources: {sources}")
    
    try:
        # Run the data collection script
        cmd = [
            sys.executable,
            "-m",
            "src.nyc_rental_price.data.scrape_listings",
            f"--sources={sources}",
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Data collection completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error collecting data: {str(e)}")
        return False


def preprocess_data():
    """Preprocess collected data.
    
    Returns:
        True if preprocessing was successful, False otherwise
    """
    logger.info("Preprocessing collected data")
    
    try:
        # Get all CSV files in the raw data directory
        raw_dir = Path("data/raw")
        csv_files = list(raw_dir.glob("*.csv"))
        
        if not csv_files:
            logger.error("No CSV files found in data/raw")
            return False
        
        # Run the preprocessing script
        cmd = [
            sys.executable,
            "-m",
            "src.nyc_rental_price.data.preprocessing",
            f"--input={raw_dir}",
            "--output=data/processed/listings_processed.csv",
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Data preprocessing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return False


def train_model(model_type):
    """Train prediction model.
    
    Args:
        model_type: Type of model to train
    
    Returns:
        True if training was successful, False otherwise
    """
    logger.info(f"Training {model_type} model")
    
    try:
        # Check if processed data exists
        processed_file = Path("data/processed/listings_processed.csv")
        if not processed_file.exists():
            logger.error("Processed data file not found")
            return False
        
        # Run the model training script
        cmd = [
            sys.executable,
            "-m",
            "src.nyc_rental_price.models.train",
            f"--data-path={processed_file}",
            f"--model-type={model_type}",
            "--output-dir=models",
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Model training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error training model: {str(e)}")
        return False


def start_api(port):
    """Start the prediction API.
    
    Args:
        port: Port for the API server
    
    Returns:
        True if the API was started successfully, False otherwise
    """
    logger.info(f"Starting API on port {port}")
    
    try:
        # Check if model exists
        model_dir = Path("models")
        if not model_dir.exists() or not any(model_dir.iterdir()):
            logger.error("No trained models found")
            return False
        
        # Run the API server
        cmd = [
            sys.executable,
            "-m",
            "src.nyc_rental_price.api.main",
            f"--port={port}",
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)  # This will block until the API is stopped
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting API: {str(e)}")
        return False
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
        return True


def main():
    """Run the NYC rental price prediction pipeline."""
    args = parse_args()
    
    # If --all is specified, run the entire pipeline
    if args.all:
        args.collect_data = True
        args.preprocess_data = True
        args.train_model = True
        args.start_api = True
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run the pipeline steps
    if args.collect_data:
        if not collect_data(args.sources):
            logger.error("Data collection failed, stopping pipeline")
            return 1
    
    if args.preprocess_data:
        if not preprocess_data():
            logger.error("Data preprocessing failed, stopping pipeline")
            return 1
    
    if args.train_model:
        if not train_model(args.model_type):
            logger.error("Model training failed, stopping pipeline")
            return 1
    
    if args.start_api:
        if not start_api(args.api_port):
            logger.error("API startup failed")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())