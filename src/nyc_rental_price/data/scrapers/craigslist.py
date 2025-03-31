"""Craigslist scraper for NYC rental listings.

This module provides a scraper for Craigslist rental listings in New York City.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .base import Scraper

logger = logging.getLogger(__name__)


class CraigslistScraper(Scraper):
    """Scraper for Craigslist rental listings."""
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        proxies: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the Craigslist scraper.
        
        Args:
            output_dir: Directory to save scraped data
            proxies: List of proxy URLs to rotate through
            headers: Custom headers to use for requests
        """
        super().__init__(
            base_url="https://newyork.craigslist.org",
            output_dir=output_dir,
            proxies=proxies,
            headers=headers,
        )
    
    def search(
        self,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        min_bedrooms: Optional[int] = None,
        max_bedrooms: Optional[int] = None,
        neighborhoods: Optional[List[str]] = None,
        no_fee: bool = False,
        has_image: bool = True,
        page: int = 0,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for rental listings on Craigslist.
        
        Args:
            min_price: Minimum rental price
            max_price: Maximum rental price
            min_bedrooms: Minimum number of bedrooms
            max_bedrooms: Maximum number of bedrooms
            neighborhoods: List of neighborhoods to search in
            no_fee: Whether to include only no-fee listings
            has_image: Whether to include only listings with images
            page: Starting page number (0-indexed)
            max_pages: Maximum number of pages to scrape
        
        Returns:
            List of rental listings
        """
        all_listings = []
        current_page = page
        
        # Build the search URL with filters
        url = f"{self.base_url}/search/apa"  # apa = apartments
        params = []
        
        if min_price:
            params.append(f"min_price={min_price}")
        if max_price:
            params.append(f"max_price={max_price}")
        if min_bedrooms:
            params.append(f"min_bedrooms={min_bedrooms}")
        if max_bedrooms:
            params.append(f"max_bedrooms={max_bedrooms}")
        if no_fee:
            params.append("broker_fee=1")  # 1 = no fee
        if has_image:
            params.append("hasPic=1")
        
        # If neighborhoods are specified, use the Craigslist area codes
        if neighborhoods:
            # This would need a mapping of neighborhood names to Craigslist area codes
            # For now, we'll use a simple approach of searching for these terms in the results
            neighborhood_filter = "|".join(neighborhoods)
        else:
            neighborhood_filter = None
        
        # Construct the base search URL
        base_search_url = f"{url}?{'&'.join(params)}"
        
        while current_page < max_pages:
            # Craigslist pagination uses s=0, s=120, s=240, etc. (120 results per page)
            offset = current_page * 120
            search_url = f"{base_search_url}&s={offset}" if offset > 0 else base_search_url
            
            logger.info(f"Scraping Craigslist page {current_page + 1}: {search_url}")
            
            try:
                html_content = self.fetch_page(search_url)
                page_listings = self.parse_listings(html_content)
                
                if not page_listings:
                    logger.info(f"No more listings found on page {current_page + 1}")
                    break
                
                # Filter by neighborhood if specified
                if neighborhood_filter:
                    filtered_listings = []
                    for listing in page_listings:
                        # Check if any neighborhood term is in the listing title or neighborhood field
                        listing_text = (
                            f"{listing.get('title', '')} {listing.get('neighborhood', '')}"
                        ).lower()
                        if re.search(neighborhood_filter.lower(), listing_text):
                            filtered_listings.append(listing)
                    
                    page_listings = filtered_listings
                
                all_listings.extend(page_listings)
                logger.info(
                    f"Found {len(page_listings)} listings on page {current_page + 1}"
                )
                
                current_page += 1
            except Exception as e:
                logger.error(f"Error scraping page {current_page + 1}: {str(e)}")
                break
        
        logger.info(f"Scraped a total of {len(all_listings)} listings from Craigslist")
        return all_listings
    
    def parse_listings(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse rental listings from Craigslist HTML content.
        
        Args:
            html_content: HTML content to parse
        
        Returns:
            List of dictionaries containing listing details
        """
        soup = BeautifulSoup(html_content, "html.parser")
        listings = []
        
        # Find all listing rows
        listing_rows = soup.select("li.cl-static-search-result")
        
        for row in listing_rows:
            try:
                # Extract listing URL and ID
                link_elem = row.select_one("a.cl-app-anchor")
                if not link_elem:
                    continue
                
                listing_url = link_elem.get("href", "")
                
                # Extract listing ID from URL
                listing_id_match = re.search(r"/(\d+)\.html", listing_url)
                listing_id = listing_id_match.group(1) if listing_id_match else ""
                
                # Extract price
                price_elem = row.select_one("div.price")
                price_text = price_elem.text.strip() if price_elem else ""
                price_match = re.search(r"\$?([\d,]+)", price_text)
                price = price_match.group(1).replace(",", "") if price_match else ""
                
                # Extract title
                title_elem = row.select_one("div.title")
                title = title_elem.text.strip() if title_elem else ""
                
                # Extract neighborhood
                hood_elem = row.select_one("div.hood")
                neighborhood = hood_elem.text.strip() if hood_elem else ""
                
                # Extract bedrooms from title or separate element
                bedrooms = ""
                beds_match = re.search(r"(\d+)\s*(?:br|bed)", title.lower())
                if beds_match:
                    bedrooms = beds_match.group(1)
                
                # Extract bathrooms from title
                bathrooms = ""
                baths_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ba|bath)", title.lower())
                if baths_match:
                    bathrooms = baths_match.group(1)
                
                # Extract square footage from title
                sqft = ""
                sqft_match = re.search(r"(\d+)\s*(?:ft|sq\s*ft)", title.lower())
                if sqft_match:
                    sqft = sqft_match.group(1)
                
                # Check if listing has images
                has_image = bool(row.select_one("div.imagebox"))
                
                # Check if no fee
                no_fee = "no fee" in title.lower() or "no broker fee" in title.lower()
                
                # Extract posting date
                date_elem = row.select_one("div.meta time")
                posted_date = date_elem.get("datetime", "") if date_elem else ""
                
                # Create listing object
                listing = {
                    "id": listing_id,
                    "title": title,
                    "neighborhood": neighborhood,
                    "price": price,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "sqft": sqft,
                    "url": listing_url,
                    "has_image": has_image,
                    "no_fee": no_fee,
                    "posted_date": posted_date,
                    "source": "craigslist",
                    "scraped_date": datetime.now().strftime("%Y-%m-%d"),
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
            listing_id_match = re.search(r"/(\d+)\.html", listing_url)
            listing_id = listing_id_match.group(1) if listing_id_match else ""
            
            # Extract title
            title_elem = soup.select_one("h1.posting-title span.postingtitletext")
            title = title_elem.text.strip() if title_elem else ""
            
            # Extract price
            price_elem = soup.select_one("span.price")
            price_text = price_elem.text.strip() if price_elem else ""
            price_match = re.search(r"\$?([\d,]+)", price_text)
            price = price_match.group(1).replace(",", "") if price_match else ""
            
            # Extract neighborhood
            hood_elem = soup.select_one("small.housing-location")
            neighborhood = hood_elem.text.strip() if hood_elem else ""
            
            # Extract posting date
            date_elem = soup.select_one("time.date.timeago")
            posted_date = date_elem.get("datetime", "") if date_elem else ""
            
            # Extract attributes from the posting body
            attributes = {}
            attr_groups = soup.select("p.attrgroup")
            
            for group in attr_groups:
                spans = group.select("span")
                
                for span in spans:
                    # Check for bedroom/bathroom/sqft info
                    if "BR" in span.text or "Ba" in span.text:
                        br_match = re.search(r"(\d+)BR", span.text)
                        ba_match = re.search(r"(\d+)Ba", span.text)
                        sqft_match = re.search(r"(\d+)ft", span.text)
                        
                        if br_match:
                            attributes["bedrooms"] = br_match.group(1)
                        if ba_match:
                            attributes["bathrooms"] = ba_match.group(1)
                        if sqft_match:
                            attributes["sqft"] = sqft_match.group(1)
                    else:
                        # Other attributes like "cats are OK - purrr" or "apartment"
                        attributes[span.text.strip()] = True
            
            # Extract description
            description_elem = soup.select_one("#postingbody")
            description = ""
            if description_elem:
                # Remove the "QR Code Link to This Post" text
                for element in description_elem.select("div.print-information"):
                    element.decompose()
                description = description_elem.text.strip()
            
            # Extract images
            images = []
            image_elems = soup.select("div.gallery img")
            for img in image_elems:
                src = img.get("src", "")
                if src:
                    images.append(src)
            
            # Check if no fee
            no_fee = "no fee" in title.lower() or "no broker fee" in title.lower()
            
            # Extract map coordinates if available
            latitude = longitude = ""
            map_elem = soup.select_one("#map")
            if map_elem:
                lat_match = re.search(r'data-latitude="([^"]+)"', str(map_elem))
                lng_match = re.search(r'data-longitude="([^"]+)"', str(map_elem))
                
                if lat_match:
                    latitude = lat_match.group(1)
                if lng_match:
                    longitude = lng_match.group(1)
            
            # Create detailed listing object
            detailed_listing = {
                "id": listing_id,
                "title": title,
                "neighborhood": neighborhood,
                "price": price,
                "bedrooms": attributes.get("bedrooms", ""),
                "bathrooms": attributes.get("bathrooms", ""),
                "sqft": attributes.get("sqft", ""),
                "description": description,
                "images": images,
                "url": listing_url,
                "posted_date": posted_date,
                "no_fee": no_fee,
                "latitude": latitude,
                "longitude": longitude,
                "attributes": attributes,
                "source": "craigslist",
                "scraped_date": datetime.now().strftime("%Y-%m-%d"),
            }
            
            return detailed_listing
        except Exception as e:
            logger.error(f"Error getting listing details: {str(e)}")
            return {}