# Phase 1: Data Acquisition

## Objectives
- Collect comprehensive and representative NYC rental listings data
- Ensure data quality and completeness for model training
- Gather sufficient features that could impact rental prices

## Methodology
1. **Web Scraping Approach**:
   - Developed a Python-based web scraper targeting major NYC real estate listing platforms
   - Implemented rate limiting and randomized request patterns to avoid IP blocking
   - Used rotating proxies to ensure continuous data collection

2. **Data Collection Strategy**:
   - Gathered data from multiple boroughs to capture geographical price variations
   - Collected listings across different time periods to account for seasonal variations
   - Ensured representation of various property types and sizes

3. **Data Validation**:
   - Implemented real-time validation checks during scraping
   - Filtered out duplicate listings and spam
   - Verified data completeness against predefined criteria

## Tools and Technologies
- **Python**: Primary programming language
- **Requests/BeautifulSoup**: For HTTP requests and HTML parsing
- **Selenium**: For handling dynamic content on listing websites
- **Pandas**: For data handling and initial cleaning
- **MongoDB**: For temporary storage of scraped data before processing

## Ethical Considerations
- Adhered to websites' robots.txt rules
- Implemented appropriate request delays to minimize server impact
- Did not collect personal or identifying information
- Data used exclusively for academic and research purposes

## Output
- Raw dataset stored in CSV format at `data/raw/listings.csv`
- Initial dataset contains approximately X listings with Y features
- Data collection covered the period from [start date] to [end date]

## Next Steps
The raw data collected in this phase requires preprocessing and cleaning, which will be handled in Phase 2 of the project.

