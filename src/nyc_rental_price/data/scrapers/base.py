"""Base scraper module for rental listings data collection.

This module provides the foundation for all scrapers used in the NYC rental price
prediction system. It includes rate limiting, proxy rotation, and robust error handling.
"""

import csv
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# Configure logging
logger = logging.getLogger(__name__)


def rate_limit(min_delay: float = 1.0, max_delay: float = 3.0):
    """Decorator to rate limit requests to avoid being blocked.
    
    Args:
        min_delay: Minimum delay in seconds between requests
        max_delay: Maximum delay in seconds between requests
    
    Returns:
        Decorated function with rate limiting
    """
    def decorator(func):
        last_call_time = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get a unique key for the function call
            key = f"{func.__name__}_{id(args[0])}"
            
            # Calculate time since last call
            now = time.time()
            time_since_last_call = now - last_call_time.get(key, 0)
            
            # Generate a random delay within the specified range
            delay = random.uniform(min_delay, max_delay)
            
            # Sleep if needed
            if time_since_last_call < delay:
                sleep_time = delay - time_since_last_call
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            # Update last call time
            last_call_time[key] = time.time()
            
            # Call the original function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.5):
    """Decorator to retry failed requests with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplicative factor for backoff between retries
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    retries += 1
                    
                    if retries <= max_retries:
                        sleep_time = backoff_factor ** retries
                        logger.warning(
                            f"Request failed: {str(e)}. Retrying in {sleep_time:.2f} seconds "
                            f"(attempt {retries}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"Request failed after {max_retries} retries: {str(e)}"
                        )
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    
    return decorator


class Scraper(ABC):
    """Base scraper class with common functionality for all data sources."""
    
    def __init__(
        self,
        base_url: str,
        output_dir: str = "data/raw",
        proxies: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the base scraper.
        
        Args:
            base_url: The base URL for the data source
            output_dir: Directory to save scraped data
            proxies: List of proxy URLs to rotate through
            headers: Custom headers to use for requests
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.proxies = proxies or []
        self.current_proxy_index = 0
        
        # Create a requests session for connection pooling
        self.session = requests.Session()
        
        # Set default headers if none provided
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} with base URL: {base_url}")
    
    def _get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next proxy from the rotation.
        
        Returns:
            Dictionary with proxy configuration or None if no proxies available
        """
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        
        return {"http": proxy, "https": proxy}
    
    @rate_limit(min_delay=2.0, max_delay=5.0)
    @retry_on_failure(max_retries=3)
    def fetch_page(self, url: str) -> str:
        """Fetch the HTML content of a webpage with rate limiting and retries.
        
        Args:
            url: The URL to fetch
        
        Returns:
            The HTML content of the page as a string
        
        Raises:
            RequestException: If the request fails after all retries
        """
        proxies = self._get_next_proxy()
        
        logger.debug(f"Fetching URL: {url}")
        response = self.session.get(
            url, headers=self.headers, proxies=proxies, timeout=30
        )
        response.raise_for_status()
        
        logger.debug(f"Successfully fetched URL: {url}")
        return response.text
    
    def save_to_csv(self, listings: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save scraped listings to a CSV file.
        
        Args:
            listings: List of listing dictionaries
            filename: Optional filename, defaults to source_YYYYMMDD_HHMMSS.csv
        
        Returns:
            Path to the saved CSV file
        """
        if not listings:
            logger.warning("No listings to save")
            return ""
        
        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_name = self.__class__.__name__.replace("Scraper", "").lower()
            filename = f"{source_name}_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Get all unique keys from the listings to use as CSV headers
        fieldnames = set()
        for listing in listings:
            fieldnames.update(listing.keys())
        
        # Sort fieldnames for consistent column order
        fieldnames = sorted(fieldnames)
        
        # Write to CSV
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(listings)
        
        logger.info(f"Saved {len(listings)} listings to {filepath}")
        return str(filepath)
    
    @abstractmethod
    def search(self, **kwargs) -> List[Dict[str, Any]]:
        """Search for rental listings with given parameters.
        
        Args:
            **kwargs: Search parameters specific to the data source
        
        Returns:
            A list of rental listings matching the search criteria
        """
        pass
    
    @abstractmethod
    def parse_listings(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse rental listings from HTML content.
        
        Args:
            html_content: HTML content to parse
        
        Returns:
            List of dictionaries containing listing details
        """
        pass


class ScraperFactory:
    """Factory class for creating appropriate scrapers based on source."""
    
    @staticmethod
    def create_scraper(source: str, **kwargs) -> Scraper:
        """Create a scraper instance for the specified source.
        
        Args:
            source: The data source name (streeteasy, zillow, craigslist)
            **kwargs: Additional arguments to pass to the scraper constructor
        
        Returns:
            A scraper instance for the specified source
        
        Raises:
            ValueError: If the source is not supported
        """
        # Import here to avoid circular imports
        from .streeteasy import StreetEasyScraper
        from .zillow import ZillowScraper
        from .craigslist import CraigslistScraper
        
        source = source.lower()
        
        if source == "streeteasy":
            return StreetEasyScraper(**kwargs)
        elif source == "zillow":
            return ZillowScraper(**kwargs)
        elif source == "craigslist":
            return CraigslistScraper(**kwargs)
        else:
            raise ValueError(f"Unsupported source: {source}")