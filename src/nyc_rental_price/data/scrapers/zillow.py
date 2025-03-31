"""Zillow scraper for NYC rental listings.

This module provides a scraper for Zillow rental listings in New York City.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from .base import Scraper

logger = logging.getLogger(__name__)


class ZillowScraper(Scraper):
    """Scraper for Zillow rental listings."""
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        proxies: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the Zillow scraper.
        
        Args:
            output_dir: Directory to save scraped data
            proxies: List of proxy URLs to rotate through
            headers: Custom headers to use for requests
        """
        # Zillow is more aggressive with anti-scraping, so we need more robust headers
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        
        if headers:
            default_headers.update(headers)
        
        super().__init__(
            base_url="https://www.zillow.com",
            output_dir=output_dir,
            proxies=proxies,
            headers=default_headers,
        )
    
    def search(
        self,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        min_beds: Optional[int] = None,
        max_beds: Optional[int] = None,
        min_baths: Optional[float] = None,
        home_type: Optional[str] = None,
        page: int = 1,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for rental listings on Zillow.
        
        Args:
            min_price: Minimum rental price
            max_price: Maximum rental price
            min_beds: Minimum number of bedrooms
            max_beds: Maximum number of bedrooms
            min_baths: Minimum number of bathrooms
            home_type: Type of home (apartment, house, etc.)
            page: Starting page number
            max_pages: Maximum number of pages to scrape
        
        Returns:
            List of rental listings
        """
        all_listings = []
        current_page = page
        
        # Build the search URL with filters
        url = f"{self.base_url}/new-york-ny/rentals"
        params = []
        
        if min_price:
            params.append(f"price={min_price}-")
        if max_price:
            if "price=" in "".join(params):
                # Update existing price parameter
                for i, param in enumerate(params):
                    if param.startswith("price="):
                        params[i] = f"{param}{max_price}"
            else:
                params.append(f"price=-{max_price}")
        
        if min_beds:
            params.append(f"beds={min_beds}-")
        if max_beds:
            if "beds=" in "".join(params):
                # Update existing beds parameter
                for i, param in enumerate(params):
                    if param.startswith("beds="):
                        params[i] = f"{param}{max_beds}"
            else:
                params.append(f"beds=-{max_beds}")
        
        if min_baths:
            params.append(f"baths={min_baths}-")
        
        if home_type:
            params.append(f"home-type={home_type}")
        
        # Join parameters with underscore for Zillow URL format
        if params:
            url = f"{url}/{'/'.join(params)}"
        
        while current_page <= max_pages:
            if current_page > 1:
                search_url = f"{url}/{current_page}_p"
            else:
                search_url = url
            
            logger.info(f"Scraping Zillow page {current_page}: {search_url}")
            
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
        
        logger.info(f"Scraped a total of {len(all_listings)} listings from Zillow")
        return all_listings
    
    def parse_listings(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse rental listings from Zillow HTML content.
        
        Args:
            html_content: HTML content to parse
        
        Returns:
            List of dictionaries containing listing details
        """
        soup = BeautifulSoup(html_content, "html.parser")
        listings = []
        
        # Zillow loads data via JavaScript, so we need to extract it from script tags
        # Look for the script containing the listing data
        data_script = None
        for script in soup.find_all("script", type="application/json"):
            if "data-zrr-shared-data-key" in script.attrs:
                data_script = script
                break
        
        if not data_script:
            # Try alternative method - direct HTML parsing
            return self._parse_listings_from_html(soup)
        
        try:
            # Extract and parse JSON data
            json_data = json.loads(data_script.string)
            search_results = json_data.get("cat1", {}).get("searchResults", {}).get("listResults", [])
            
            for result in search_results:
                try:
                    # Extract listing ID
                    listing_id = result.get("zpid", "")
                    
                    # Extract address components
                    address = result.get("address", {})
                    street_address = address.get("streetAddress", "")
                    city = address.get("city", "")
                    state = address.get("state", "")
                    zipcode = address.get("zipcode", "")
                    full_address = f"{street_address}, {city}, {state} {zipcode}"
                    
                    # Extract price
                    price_text = result.get("price", "")
                    price_match = re.search(r"\$?([\d,]+)", price_text)
                    price = price_match.group(1).replace(",", "") if price_match else ""
                    
                    # Extract basic details
                    bedrooms = result.get("beds", "")
                    bathrooms = result.get("baths", "")
                    sqft = result.get("area", "")
                    
                    # Extract URL
                    detail_url = result.get("detailUrl", "")
                    if detail_url and not detail_url.startswith("http"):
                        detail_url = f"https://www.zillow.com{detail_url}"
                    
                    # Extract listing type and status
                    listing_type = result.get("hdpData", {}).get("homeInfo", {}).get("propertyType", "")
                    status = result.get("statusType", "")
                    
                    # Extract latitude and longitude
                    lat = result.get("latLong", {}).get("latitude", "")
                    lng = result.get("latLong", {}).get("longitude", "")
                    
                    # Create listing object
                    listing = {
                        "id": listing_id,
                        "address": full_address,
                        "street_address": street_address,
                        "city": city,
                        "state": state,
                        "zipcode": zipcode,
                        "price": price,
                        "bedrooms": bedrooms,
                        "bathrooms": bathrooms,
                        "sqft": sqft,
                        "url": detail_url,
                        "property_type": listing_type,
                        "status": status,
                        "latitude": lat,
                        "longitude": lng,
                        "source": "zillow",
                        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                    }
                    
                    listings.append(listing)
                except Exception as e:
                    logger.error(f"Error parsing listing: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error parsing JSON data: {str(e)}")
            # Fall back to HTML parsing
            listings = self._parse_listings_from_html(soup)
        
        return listings
    
    def _parse_listings_from_html(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse listings directly from HTML as a fallback method.
        
        Args:
            soup: BeautifulSoup object of the page HTML
        
        Returns:
            List of dictionaries containing listing details
        """
        listings = []
        
        # Find all listing cards
        listing_cards = soup.select("div[data-test='property-card']")
        
        for card in listing_cards:
            try:
                # Extract listing URL and ID
                link_elem = card.select_one("a[data-test='property-card-link']")
                detail_url = link_elem.get("href", "") if link_elem else ""
                
                if detail_url and not detail_url.startswith("http"):
                    detail_url = f"https://www.zillow.com{detail_url}"
                
                # Extract listing ID from URL
                listing_id_match = re.search(r"/(\d+)_zpid", detail_url)
                listing_id = listing_id_match.group(1) if listing_id_match else ""
                
                # Extract price
                price_elem = card.select_one("[data-test='property-card-price']")
                price_text = price_elem.text.strip() if price_elem else ""
                price_match = re.search(r"\$?([\d,]+)", price_text)
                price = price_match.group(1).replace(",", "") if price_match else ""
                
                # Extract address
                address_elem = card.select_one("[data-test='property-card-addr']")
                address = address_elem.text.strip() if address_elem else ""
                
                # Extract details
                details_elem = card.select_one("[data-test='property-card-details']")
                details_text = details_elem.text.strip() if details_elem else ""
                
                # Parse beds, baths, sqft from details text
                bedrooms = bathrooms = sqft = ""
                
                beds_match = re.search(r"(\d+)\s*bd", details_text)
                if beds_match:
                    bedrooms = beds_match.group(1)
                
                baths_match = re.search(r"(\d+(?:\.\d+)?)\s*ba", details_text)
                if baths_match:
                    bathrooms = baths_match.group(1)
                
                sqft_match = re.search(r"([\d,]+)\s*sqft", details_text)
                if sqft_match:
                    sqft = sqft_match.group(1).replace(",", "")
                
                # Create listing object
                listing = {
                    "id": listing_id,
                    "address": address,
                    "price": price,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "sqft": sqft,
                    "url": detail_url,
                    "source": "zillow",
                    "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                }
                
                listings.append(listing)
            except Exception as e:
                logger.error(f"Error parsing listing from HTML: {str(e)}")
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
            listing_id_match = re.search(r"/(\d+)_zpid", listing_url)
            listing_id = listing_id_match.group(1) if listing_id_match else ""
            
            # Try to extract data from JSON in script tags
            data_script = None
            for script in soup.find_all("script", type="application/json"):
                if "data-zrr-shared-data-key" in script.attrs:
                    data_script = script
                    break
            
            if data_script:
                try:
                    json_data = json.loads(data_script.string)
                    property_data = json_data.get("apiCache", {})
                    
                    # Find the property data in the API cache
                    for key, value in property_data.items():
                        if "property" in key.lower() and isinstance(value, dict):
                            property_info = value
                            
                            # Extract basic details
                            address = property_info.get("address", {})
                            street_address = address.get("streetAddress", "")
                            city = address.get("city", "")
                            state = address.get("state", "")
                            zipcode = address.get("zipcode", "")
                            full_address = f"{street_address}, {city}, {state} {zipcode}"
                            
                            # Extract price
                            price = property_info.get("price", "")
                            if isinstance(price, str):
                                price_match = re.search(r"\$?([\d,]+)", price)
                                price = price_match.group(1).replace(",", "") if price_match else ""
                            
                            # Extract details
                            bedrooms = property_info.get("bedrooms", "")
                            bathrooms = property_info.get("bathrooms", "")
                            sqft = property_info.get("livingArea", "")
                            
                            # Extract description
                            description = property_info.get("description", "")
                            
                            # Extract features and amenities
                            features = property_info.get("resoFacts", {}).get("atAGlanceFacts", [])
                            amenities = []
                            for feature in features:
                                amenities.append(feature.get("factValue", ""))
                            
                            # Extract images
                            images = []
                            media_items = property_info.get("mediaItems", [])
                            for item in media_items:
                                if item.get("type") == "image":
                                    images.append(item.get("url", ""))
                            
                            # Create detailed listing object
                            detailed_listing = {
                                "id": listing_id,
                                "address": full_address,
                                "street_address": street_address,
                                "city": city,
                                "state": state,
                                "zipcode": zipcode,
                                "price": price,
                                "bedrooms": bedrooms,
                                "bathrooms": bathrooms,
                                "sqft": sqft,
                                "description": description,
                                "amenities": amenities,
                                "images": images,
                                "url": listing_url,
                                "source": "zillow",
                                "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                            }
                            
                            return detailed_listing
                except Exception as e:
                    logger.error(f"Error parsing JSON data for details: {str(e)}")
            
            # Fall back to HTML parsing if JSON extraction fails
            return self._parse_listing_details_from_html(soup, listing_id, listing_url)
        except Exception as e:
            logger.error(f"Error getting listing details: {str(e)}")
            return {}
    
    def _parse_listing_details_from_html(
        self, soup: BeautifulSoup, listing_id: str, listing_url: str
    ) -> Dict[str, Any]:
        """Parse listing details from HTML as a fallback method.
        
        Args:
            soup: BeautifulSoup object of the page HTML
            listing_id: ID of the listing
            listing_url: URL of the listing
        
        Returns:
            Dictionary containing detailed listing information
        """
        try:
            # Extract address
            address_elem = soup.select_one("h1[data-test='home-details-summary-address']")
            address = address_elem.text.strip() if address_elem else ""
            
            # Extract price
            price_elem = soup.select_one("[data-test='home-details-price']")
            price_text = price_elem.text.strip() if price_elem else ""
            price_match = re.search(r"\$?([\d,]+)", price_text)
            price = price_match.group(1).replace(",", "") if price_match else ""
            
            # Extract basic details
            beds_elem = soup.select_one("[data-test='home-details-beds']")
            bedrooms = beds_elem.text.strip() if beds_elem else ""
            bedrooms = re.sub(r"[^\d]", "", bedrooms)
            
            baths_elem = soup.select_one("[data-test='home-details-baths']")
            bathrooms = baths_elem.text.strip() if baths_elem else ""
            bathrooms = re.sub(r"[^\d\.]", "", bathrooms)
            
            sqft_elem = soup.select_one("[data-test='home-details-sqft']")
            sqft_text = sqft_elem.text.strip() if sqft_elem else ""
            sqft_match = re.search(r"([\d,]+)", sqft_text)
            sqft = sqft_match.group(1).replace(",", "") if sqft_match else ""
            
            # Extract description
            description_elem = soup.select_one("[data-test='home-description']")
            description = description_elem.text.strip() if description_elem else ""
            
            # Extract features and amenities
            amenities = []
            features_section = soup.select_one("[data-test='home-features']")
            if features_section:
                feature_items = features_section.select("li")
                for item in feature_items:
                    amenities.append(item.text.strip())
            
            # Extract images
            images = []
            image_elems = soup.select("picture img")
            for img in image_elems:
                src = img.get("src", "")
                if src and "images-cf" in src:
                    images.append(src)
            
            # Create detailed listing object
            detailed_listing = {
                "id": listing_id,
                "address": address,
                "price": price,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft": sqft,
                "description": description,
                "amenities": amenities,
                "images": images,
                "url": listing_url,
                "source": "zillow",
                "scraped_date": datetime.now().strftime("%Y-%m-%d"),
            }
            
            return detailed_listing
        except Exception as e:
            logger.error(f"Error parsing listing details from HTML: {str(e)}")
            return {}