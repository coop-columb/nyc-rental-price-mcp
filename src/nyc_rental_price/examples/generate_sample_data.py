#!/usr/bin/env python
"""
Sample data generator for NYC rental price prediction.

This script generates synthetic rental listings data for testing the NYC rental price
prediction system without requiring real data.
"""

import argparse
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_sample_data(
    num_samples: int = 1000,
    output_path: str = "data/raw/sample_listings.csv",
    random_state: int = 42,
):
    """Generate synthetic rental listings data.

    Args:
        num_samples: Number of listings to generate
        output_path: Path to save the generated data
        random_state: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)

    # Define NYC neighborhoods
    neighborhoods = [
        "Upper East Side", "Upper West Side", "Midtown", "Chelsea", "Greenwich Village",
        "SoHo", "Tribeca", "Financial District", "Harlem", "East Village", "West Village",
        "Williamsburg", "Park Slope", "DUMBO", "Brooklyn Heights", "Bushwick",
        "Bedford-Stuyvesant", "Crown Heights", "Flatbush", "Greenpoint", "Astoria",
        "Long Island City", "Flushing", "Jackson Heights", "Forest Hills"
    ]

    # Define building types
    building_types = [
        "Apartment", "Condo", "Co-op", "Townhouse", "Duplex", "Loft", "Studio",
        "Walk-up", "Elevator Building", "Doorman Building", "Luxury Building"
    ]

    # Define amenities
    amenities = [
        "Elevator", "Doorman", "Gym", "Laundry", "Dishwasher", "Hardwood Floors",
        "High Ceilings", "Stainless Steel Appliances", "Roof Deck", "Balcony",
        "Terrace", "Outdoor Space", "Parking", "Storage", "Pets Allowed", "Furnished",
        "Central Air", "Fireplace", "Swimming Pool", "Concierge", "Bike Room"
    ]

    # Generate random listings
    listings = []
    for i in range(num_samples):
        # Basic listing details
        neighborhood = random.choice(neighborhoods)
        bedrooms = random.choice([0, 1, 1, 1, 2, 2, 3, 4, 5])  # More weight to 1-2 bedrooms
        bathrooms = random.choice([1, 1, 1, 1.5, 1.5, 2, 2, 2.5, 3])  # More weight to 1-2 bathrooms
        
        # Square footage - depends on bedrooms
        sqft_base = 350 if bedrooms == 0 else 500  # Studio base is 350 sqft
        sqft_per_bedroom = 250  # Additional sqft per bedroom
        sqft_variation = 100  # Random variation
        sqft = sqft_base + (bedrooms * sqft_per_bedroom) + random.randint(-sqft_variation, sqft_variation)
        
        # Price - depends on neighborhood, bedrooms, bathrooms, sqft
        # Base price factors
        neighborhood_factor = {
            "Upper East Side": 1.2, "Upper West Side": 1.2, "Midtown": 1.3, 
            "Chelsea": 1.4, "Greenwich Village": 1.5, "SoHo": 1.6, "Tribeca": 1.7,
            "Financial District": 1.3, "Harlem": 0.8, "East Village": 1.3, 
            "West Village": 1.5, "Williamsburg": 1.1, "Park Slope": 1.0,
            "DUMBO": 1.4, "Brooklyn Heights": 1.3, "Bushwick": 0.9,
            "Bedford-Stuyvesant": 0.8, "Crown Heights": 0.8, "Flatbush": 0.7,
            "Greenpoint": 1.0, "Astoria": 0.9, "Long Island City": 1.1,
            "Flushing": 0.8, "Jackson Heights": 0.7, "Forest Hills": 0.8
        }.get(neighborhood, 1.0)
        
        # Calculate base price
        base_price = 2000 + (bedrooms * 1000) + (bathrooms * 500) + (sqft * 1.5)
        
        # Apply neighborhood factor and add random variation
        price = int(base_price * neighborhood_factor * random.uniform(0.85, 1.15))
        
        # Building type
        building_type = random.choice(building_types)
        
        # Random amenities
        num_amenities = random.randint(0, 10)
        listing_amenities = random.sample(amenities, num_amenities)
        amenities_str = ", ".join(listing_amenities)
        
        # Posted date - random date in the last 60 days
        days_ago = random.randint(0, 60)
        posted_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Description
        description = generate_description(
            neighborhood, bedrooms, bathrooms, sqft, building_type, listing_amenities
        )
        
        # Create listing
        listing = {
            "id": f"L{i+1:06d}",
            "neighborhood": neighborhood,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": sqft,
            "price": price,
            "building_type": building_type,
            "amenities": amenities_str,
            "posted_date": posted_date,
            "description": description,
            "source": "sample_data"
        }
        
        listings.append(listing)
    
    # Create DataFrame
    df = pd.DataFrame(listings)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Generated {num_samples} sample listings and saved to {output_path}")
    
    return df


def generate_description(
    neighborhood, bedrooms, bathrooms, sqft, building_type, amenities
):
    """Generate a realistic listing description.

    Args:
        neighborhood: Neighborhood name
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        sqft: Square footage
        building_type: Type of building
        amenities: List of amenities

    Returns:
        Generated description
    """
    # Intro phrases
    intros = [
        f"Beautiful {building_type.lower()} in the heart of {neighborhood}.",
        f"Spacious {building_type.lower()} located in {neighborhood}.",
        f"Charming {building_type.lower()} in prime {neighborhood} location.",
        f"Luxury {building_type.lower()} in desirable {neighborhood}.",
        f"Stunning {building_type.lower()} in the vibrant neighborhood of {neighborhood}."
    ]
    
    # Room descriptions
    if bedrooms == 0:
        bedroom_desc = "Studio apartment with sleeping area"
    elif bedrooms == 1:
        bedroom_desc = "One bedroom with ample closet space"
    else:
        bedroom_desc = f"{bedrooms} bedrooms with generous closet space"
    
    bathroom_desc = f"{bathrooms} {'bathroom' if bathrooms == 1 else 'bathrooms'}"
    
    # Amenity descriptions
    amenity_text = ""
    if amenities:
        amenity_list = ", ".join(amenities[:-1]) + (f" and {amenities[-1]}" if len(amenities) > 1 else amenities[0])
        amenity_text = f" Building features include {amenity_list}."
    
    # Location benefits
    location_benefits = [
        f"Close to subway lines.",
        f"Near restaurants and shopping.",
        f"Steps from parks and recreation.",
        f"Easy access to public transportation.",
        f"Convenient location with nearby cafes and shops."
    ]
    
    # Assemble description
    description = (
        f"{random.choice(intros)} This {sqft} square foot home features {bedroom_desc} "
        f"and {bathroom_desc}. {amenity_text} {random.choice(location_benefits)}"
    )
    
    return description


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sample rental listings data"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of listings to generate",
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/raw/sample_listings.csv",
        help="Path to save the generated data",
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    generate_sample_data(
        num_samples=args.num_samples,
        output_path=args.output_path,
        random_state=args.random_state,
    )