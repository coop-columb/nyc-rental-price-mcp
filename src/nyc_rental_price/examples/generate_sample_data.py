"""Generate sample data for NYC rental price prediction."""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(num_samples=1000, output_path=None):
    """Generate synthetic NYC rental listings data.

    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the generated data

    Returns:
        DataFrame with synthetic data
    """
    # Define neighborhoods by borough
    neighborhoods = {
        "Manhattan": [
            "Upper East Side",
            "Upper West Side",
            "Midtown",
            "Chelsea",
            "Greenwich Village",
            "East Village",
            "SoHo",
            "Tribeca",
            "Financial District",
            "Harlem",
        ],
        "Brooklyn": [
            "Williamsburg",
            "Park Slope",
            "Brooklyn Heights",
            "DUMBO",
            "Bushwick",
            "Bedford-Stuyvesant",
            "Crown Heights",
            "Greenpoint",
        ],
        "Queens": [
            "Astoria",
            "Long Island City",
            "Jackson Heights",
            "Forest Hills",
            "Flushing",
            "Sunnyside",
            "Ridgewood",
        ],
        "Bronx": ["Riverdale", "Fordham", "Concourse", "Mott Haven"],
        "Staten Island": ["St. George", "Tompkinsville", "Stapleton"],
    }

    # Flatten neighborhoods
    all_neighborhoods = [n for sublist in neighborhoods.values() for n in sublist]

    # Generate data
    data = []
    for _ in range(num_samples):
        # Select neighborhood and borough
        neighborhood = random.choice(all_neighborhoods)
        borough = next(b for b, n in neighborhoods.items() if neighborhood in n)

        # Generate property details
        bedrooms = random.choice([0, 1, 1, 1, 2, 2, 2, 3, 3, 4])
        bathrooms = min(bedrooms + random.choice([-1, 0, 0, 0, 1]), 3)
        bathrooms = max(1, bathrooms)  # Ensure at least 1 bathroom

        # Square footage based on bedrooms
        base_sqft = 400 + bedrooms * 250
        sqft = base_sqft + random.randint(-100, 200)

        # Amenities
        has_doorman = random.choice([0, 0, 0, 1])
        has_elevator = random.choice([0, 0, 1, 1])
        has_dishwasher = random.choice([0, 1, 1])
        has_washer_dryer = random.choice([0, 0, 0, 1])
        is_furnished = random.choice([0, 0, 0, 0, 1])
        has_balcony = random.choice([0, 0, 0, 1])
        has_parking = random.choice([0, 0, 0, 1])

        # Price calculation with some randomness
        # Base price by borough
        borough_multiplier = {
            "Manhattan": 1.5,
            "Brooklyn": 1.2,
            "Queens": 1.0,
            "Bronx": 0.8,
            "Staten Island": 0.7,
        }

        # Calculate price
        base_price = 1500 + (bedrooms * 800) + (bathrooms * 400) + (sqft * 2)
        price = base_price * borough_multiplier[borough]

        # Add amenity premiums
        if has_doorman:
            price *= 1.1
        if has_elevator:
            price *= 1.05
        if has_washer_dryer:
            price *= 1.08
        if is_furnished:
            price *= 1.15
        if has_balcony:
            price *= 1.07
        if has_parking:
            price *= 1.12

        # Add some random variation
        price *= random.uniform(0.9, 1.1)

        # Round price
        price = round(price / 50) * 50

        # Create listing
        listing = {
            "neighborhood": neighborhood,
            "borough": borough,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": sqft,
            "has_doorman": has_doorman,
            "has_elevator": has_elevator,
            "has_dishwasher": has_dishwasher,
            "has_washer_dryer": has_washer_dryer,
            "is_furnished": is_furnished,
            "has_balcony": has_balcony,
            "has_parking": has_parking,
            "price": price,
        }

        data.append(listing)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save data if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Generated {num_samples} samples and saved to {output_path}")

    return df


def main():
    """Main function to parse arguments and generate sample data."""
    parser = argparse.ArgumentParser(description="Generate sample NYC rental data")

    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to generate"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/sample_listings.csv",
        help="Path to save the generated data",
    )

    args = parser.parse_args()

    # Generate data
    generate_sample_data(args.num_samples, args.output)


if __name__ == "__main__":
    main()
