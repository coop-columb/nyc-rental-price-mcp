"""Command-line interface for running scrapers.

This module provides a command-line interface for running the rental listings scrapers.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.nyc_rental_price.data.scrapers import ScraperFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Scrape rental listings from various sources"
    )
    
    parser.add_argument(
        "--sources",
        type=str,
        default="streeteasy,zillow,craigslist",
        help="Comma-separated list of sources to scrape (streeteasy,zillow,craigslist)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save scraped data",
    )
    
    parser.add_argument(
        "--min-price",
        type=int,
        default=None,
        help="Minimum rental price",
    )
    
    parser.add_argument(
        "--max-price",
        type=int,
        default=None,
        help="Maximum rental price",
    )
    
    parser.add_argument(
        "--min-beds",
        type=int,
        default=None,
        help="Minimum number of bedrooms",
    )
    
    parser.add_argument(
        "--max-beds",
        type=int,
        default=None,
        help="Maximum number of bedrooms",
    )
    
    parser.add_argument(
        "--neighborhoods",
        type=str,
        default=None,
        help="Comma-separated list of neighborhoods to search in",
    )
    
    parser.add_argument(
        "--no-fee",
        action="store_true",
        help="Search for no-fee listings only",
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of pages to scrape per source",
    )
    
    parser.add_argument(
        "--proxies",
        type=str,
        default=None,
        help="Comma-separated list of proxy URLs to rotate through",
    )
    
    return parser.parse_args()


def main():
    """Run the scrapers based on command-line arguments."""
    args = parse_args()
    
    # Parse arguments
    sources = [s.strip() for s in args.sources.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    neighborhoods = None
    if args.neighborhoods:
        neighborhoods = [n.strip() for n in args.neighborhoods.split(",")]
    
    proxies = None
    if args.proxies:
        proxies = [p.strip() for p in args.proxies.split(",")]
    
    # Run scrapers for each source
    for source in sources:
        try:
            logger.info(f"Starting scraper for {source}")
            
            # Create scraper for the source
            scraper = ScraperFactory.create_scraper(
                source,
                output_dir=str(output_dir),
                proxies=proxies,
            )
            
            # Search for listings
            listings = scraper.search(
                min_price=args.min_price,
                max_price=args.max_price,
                min_beds=args.min_beds,
                max_beds=args.max_beds,
                neighborhoods=neighborhoods,
                no_fee=args.no_fee,
                max_pages=args.max_pages,
            )
            
            # Save listings to CSV
            if listings:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{source}_listings_{timestamp}.csv"
                scraper.save_to_csv(listings, filename)
                logger.info(f"Saved {len(listings)} listings from {source}")
            else:
                logger.warning(f"No listings found for {source}")
        
        except Exception as e:
            logger.error(f"Error running scraper for {source}: {str(e)}")
    
    logger.info("Scraping completed")


if __name__ == "__main__":
    main()