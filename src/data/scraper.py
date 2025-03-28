import json
import logging
import os
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_page(url):
    headers = {
        "User-Agent": "NYC Rental Price Prediction Project - [Your Name/Contact Info]"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None


def parse_listings(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # Implement parsing logic here
    listings = []  # Placeholder: return empty list until parsing logic is implemented
    return listings


def scrape_listings(base_url, pages):
    all_listings = []
    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        html_content = fetch_page(url)
        if html_content:
            listings = parse_listings(html_content)
            all_listings.extend(listings)
            logging.info(f"Page {page} scraped successfully.")
        else:
            logging.warning(f"Skipping page {page} due to fetch failure.")
        time.sleep(5)  # Polite scraping
    return all_listings


if __name__ == "__main__":
    base_url = "https://streeteasy.com/for-rent/nyc"
    pages_to_scrape = 10
    listings = scrape_listings(base_url, pages_to_scrape)

    # Create data/raw directory if it doesn't exist
    raw_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw"
    )
    os.makedirs(raw_data_dir, exist_ok=True)

    # Save listings to data/raw/ with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(raw_data_dir, f"rental_listings_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(listings, f, indent=4)

    logging.info(f"Saved {len(listings)} listings to {output_file}")
