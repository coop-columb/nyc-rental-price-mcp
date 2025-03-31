"""StreetEasy scraper for NYC rental listings.

This module provides a scraper for StreetEasy rental listings in New York City.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from .base import Scraper

logger = logging.getLogger(__name__)


class StreetEasyScraper(Scraper):
    """Scraper for StreetEasy rental listings."""
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        proxies: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the StreetEasy scraper.
        
        Args:
            output_dir: Directory to save scraped data
            proxies: List of proxy URLs to rotate through
            headers: Custom headers to use for requests
        """
        super().__init__(
            base_url="https://streeteasy.com",
            output_dir=output_dir,
            proxies=proxies,
            headers=headers,
        )
    
    def search(
        self,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        min_beds: Optional[int] = None,
        max_beds: Optional[int] = None,
        neighborhoods: Optional[List[str]] = None,
        no_fee: bool = False,
        page: int = 1,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for rental listings on StreetEasy.
        
        Args:
            min_price: Minimum rental price
            max_price: Maximum rental price
            min_beds: Minimum number of bedrooms
            max_beds: Maximum number of bedrooms
            neighborhoods: List of neighborhoods to search in
            no_fee: Whether to include only no-fee listings
            page: Starting page number
            max_pages: Maximum number of pages to scrape
        
        Returns:
            List of rental listings
        """
        all_listings = []
        current_page = page
        
        # Build the search URL with filters
        url = f"{self.base_url}/for-rent/nyc"
        params = []
        
        if min_price:
            params.append(f"price_min={min_price}")
        if max_price:
            params.append(f"price_max={max_price}")
        if min_beds:
            params.append(f"beds_min={min_beds}")
        if max_beds:
            params.append(f"beds_max={max_beds}")
        if neighborhoods:
            # Format neighborhoods for StreetEasy URL (e.g., "east-village|west-village")
            formatted_neighborhoods = "|".join(
                n.lower().replace(" ", "-") for n in neighborhoods
            )
            params.append(f"area={formatted_neighborhoods}")
        if no_fee:
            params.append("no_fee=1")
        
        base_search_url = f"{url}?{'&'.join(params)}"
        
        while current_page <= max_pages:
            search_url = f"{base_search_url}&page={current_page}"
            logger.info(f"Scraping StreetEasy page {current_page}: {search_url}")
            
            try:
                html_content = self.fetch_page(search_url)
                page_listings = self.parse_listings(html_content)
                
                if not page_listings:
                    logger.info(f"No more listings found on page {current_page}")
                    break
                
                all_listings.extend(page_listings)
                logger.info(f"Found {len(page_listings)} listings on page {current_page}")
                
                current_page += 1
            except Exception as e:
                logger.error(f"Error scraping page {current_page}: {str(e)}")
                break
        
        logger.info(f"Scraped a total of {len(all_listings)} listings from StreetEasy")
        return all_listings
    
    def parse_listings(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse rental listings from StreetEasy HTML content.
        
        Args:
            html_content: HTML content to parse
        
        Returns:
            List of dictionaries containing listing details
        """
        soup = BeautifulSoup(html_content, "html.parser")
        listings = []
        
        # Find all listing cards
        listing_cards = soup.select("div.searchCardList article")
        
        for card in listing_cards:
            try:
                # Extract listing URL and ID
                listing_link = card.select_one("a.listingCard-globalLink")
                if not listing_link:
                    continue
                
                listing_url = listing_link.get("href", "")
                if not listing_url.startswith("http"):
                    listing_url = f"{self.base_url}{listing_url}"
                
                # Extract listing ID from URL
                listing_id_match = re.search(r"/(\d+)_", listing_url)
                listing_id = listing_id_match.group(1) if listing_id_match else ""
                
                # Extract basic details
                title_elem = card.select_one("div.listingCard-content h3")
                title = title_elem.text.strip() if title_elem else ""
                
                address_elem = card.select_one("div.listingCard-content .listingCard-addressLabel")
                address = address_elem.text.strip() if address_elem else ""
                
                price_elem = card.select_one("div.listingCard-content .listingCard-priceLabel")
                price_text = price_elem.text.strip() if price_elem else ""
                # Extract numeric price
                price_match = re.search(r"\$?([\d,]+)", price_text)
                price = price_match.group(1).replace(",", "") if price_match else ""
                
                # Extract details like beds, baths, sqft
                details_elem = card.select_one("div.listingCard-details")
                beds = baths = sqft = ""
                
                if details_elem:
                    beds_elem = details_elem.select_one(".listingDetailDefinitions .listingDetail-rooms")
                    beds = beds_elem.text.strip() if beds_elem else ""
                    
                    baths_elem = details_elem.select_one(".listingDetailDefinitions .listingDetail-baths")
                    baths = baths_elem.text.strip() if baths_elem else ""
                    
                    sqft_elem = details_elem.select_one(".listingDetailDefinitions .listingDetail-sqft")
                    sqft = sqft_elem.text.strip() if sqft_elem else ""
                
                # Extract neighborhood
                neighborhood_elem = card.select_one("div.listingCard-content .listingCard-upperInfo .listingCard-upperShortInfo")
                neighborhood = neighborhood_elem.text.strip() if neighborhood_elem else ""
                
                # Check if no-fee
                no_fee = bool(card.select_one(".noFee"))
                
                # Create listing object
                listing = {
                    "id": listing_id,
                    "title": title,
                    "address": address,
                    "neighborhood": neighborhood,
                    "price": price,
                    "bedrooms": beds,
                    "bathrooms": baths,
                    "sqft": sqft,
                    "no_fee": no_fee,
                    "url": listing_url,
                    "source": "streeteasy",
                }
                
                listings.append(listing)
            except Exception as e:
                logger.error(f"Error parsing listing: {str(e)}")
                continue
        
        return listings
    
    def get_listing_details(self, listing_url: str) -> Dict[str, Any]:
        """Get detailed information for a specific listing.
        
        Args:
            listing_url: URL of the listing to get details for
        
        Returns:
            Dictionary containing detailed listing information
        """
        logger.info(f"Fetching details for listing: {listing_url}")
        
        try:
            html_content = self.fetch_page(listing_url)
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract listing ID from URL
            listing_id_match = re.search(r"/(\d+)_", listing_url)
            listing_id = listing_id_match.group(1) if listing_id_match else ""
            
            # Extract basic details
            title_elem = soup.select_one("h1.building-title")
            title = title_elem.text.strip() if title_elem else ""
            
            address_elem = soup.select_one("p.building-address")
            address = address_elem.text.strip() if address_elem else ""
            
            price_elem = soup.select_one("div.detail-cell .price")
            price_text = price_elem.text.strip() if price_elem else ""
            price_match = re.search(r"\$?([\d,]+)", price_text)
            price = price_match.group(1).replace(",", "") if price_match else ""
            
            # Extract details
            details = {}
            detail_groups = soup.select("div.detail-cell")
            
            for group in detail_groups:
                label_elem = group.select_one(".label")
                value_elem = group.select_one(".value")
                
                if label_elem and value_elem:
                    label = label_elem.text.strip().lower().replace(" ", "_")
                    value = value_elem.text.strip()
                    details[label] = value
            
            # Extract description
            description_elem = soup.select_one("div.description")
            description = description_elem.text.strip() if description_elem else ""
            
            # Extract amenities
            amenities = []
            amenities_elems = soup.select("div.amenities li")
            for amenity in amenities_elems:
                amenities.append(amenity.text.strip())
            
            # Extract images
            images = []
            image_elems = soup.select("div.image-gallery img")
            for img in image_elems:
                src = img.get("src", "")
                if src:
                    images.append(src)
            
            # Create detailed listing object
            detailed_listing = {
                "id": listing_id,
                "title": title,
                "address": address,
                "price": price,
                "bedrooms": details.get("bedrooms", ""),
                "bathrooms": details.get("bathrooms", ""),
                "sqft": details.get("square_feet", ""),
                "description": description,
                "amenities": amenities,
                "images": images,
                "url": listing_url,
                "source": "streeteasy",
                **details,  # Include all other details
            }
            
            return detailed_listing
        except Exception as e:
            logger.error(f"Error getting listing details: {str(e)}")
            return {}