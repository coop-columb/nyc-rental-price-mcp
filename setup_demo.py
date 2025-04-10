import os
import random

import numpy as np
import pandas as pd

# Create data directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("models", exist_ok=True)

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
num_samples = 2000
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

# Save raw data
df.to_csv("data/raw/sample_listings.csv", index=False)
print(f"Generated {num_samples} samples and saved to data/raw/sample_listings.csv")

# Create a processed version with basic preprocessing
processed_df = df.copy()

# Add some derived features
processed_df["price_per_sqft"] = processed_df["price"] / processed_df["sqft"]
processed_df["total_rooms"] = processed_df["bedrooms"] + processed_df["bathrooms"]
processed_df["amenities_count"] = (
    processed_df["has_doorman"]
    + processed_df["has_elevator"]
    + processed_df["has_dishwasher"]
    + processed_df["has_washer_dryer"]
    + processed_df["is_furnished"]
    + processed_df["has_balcony"]
    + processed_df["has_parking"]
)

# One-hot encode borough
borough_dummies = pd.get_dummies(processed_df["borough"], prefix="borough")
processed_df = pd.concat([processed_df, borough_dummies], axis=1)

# Save processed data
processed_df.to_csv("data/processed/listings_processed.csv", index=False)
print(f"Processed data saved to data/processed/listings_processed.csv")

# Create a simple model file to simulate trained model
with open("models/gradient_boosting_model_20250410_000000.pkl", "w") as f:
    f.write("This is a placeholder for a trained model file")

print("Created placeholder model file")
