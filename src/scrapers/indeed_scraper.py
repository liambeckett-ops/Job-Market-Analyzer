"""
Indeed job scraper implementation.
Scrapes job listings from Indeed.com with respect to rate limits and robots.txt.
"""

from typing import List, Dict, Optional
from urllib.parse import quote_plus
import logging
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

class IndeedScraper(BaseScraper):
    """
    Scraper for Indeed.com job listings.
    """
    
    def __init__(self):
        super().__init__("https://www.indeed.com", delay_range=(2, 4))
        
    def scrape_jobs(self, query: str, location: str, num_pages: int = 5) -> List[Dict]:
        """
        Scrape job listings from Indeed.
        
        Args:
            query: Job search query
            location: Location to search
            num_pages: Number of pages to scrape
            
        Returns:
            List of job dictionaries
        """
        jobs = []
        
        for page in range(num_pages):
            logger.info(f"Scraping Indeed page {page + 1} for '{query}' in '{location}'")
            
            # Build search URL
            params = {
                'q': query,
                'l': location,
                'start': page * 10  # Indeed shows 10 jobs per page
            }
            
            search_url = f"{self.base_url}/jobs"
            soup = self._make_request(search_url, params)
            
            if not soup:
                logger.warning(f"Failed to fetch page {page + 1}")
                continue
                
            # Find job cards
            job_cards = soup.find_all('div', {'class': lambda x: x and 'job_seen_beacon' in x})
            
            if not job_cards:
                # Try alternative selectors
                job_cards = soup.find_all('a', {'data-jk': True})
            
            logger.info(f"Found {len(job_cards)} jobs on page {page + 1}")
            
            for card in job_cards:
                try:
                    job_data = self._parse_job_card(card)
                    if job_data:
                        jobs.append(job_data)
                except Exception as e:
                    logger.error(f"Error parsing job card: {e}")
                    continue
                    
        logger.info(f"Successfully scraped {len(jobs)} jobs from Indeed")
        return jobs
        
    def _parse_job_card(self, card) -> Optional[Dict]:
        """
        Parse individual job card from Indeed search results.
        
        Args:
            card: BeautifulSoup element representing a job card
            
        Returns:
            Dictionary with job information or None if parsing fails
        """
        try:
            # Extract job title
            title_elem = card.find('h2', {'class': lambda x: x and 'jobTitle' in x if x else False})
            if not title_elem:
                title_elem = card.find('a', {'data-jk': True})
            
            if not title_elem:
                return None
                
            title = self._clean_text(title_elem.get_text())
            
            # Extract job URL
            link_elem = title_elem.find('a') if title_elem.name != 'a' else title_elem
            job_url = link_elem.get('href', '') if link_elem else ''
            if job_url and not job_url.startswith('http'):
                job_url = f"{self.base_url}{job_url}"
                
            # Extract company name
            company_elem = card.find('span', {'class': lambda x: x and 'companyName' in x if x else False})
            company = self._clean_text(company_elem.get_text()) if company_elem else "Unknown"
            
            # Extract location
            location_elem = card.find('div', {'class': lambda x: x and 'companyLocation' in x if x else False})
            location = self._clean_text(location_elem.get_text()) if location_elem else "Unknown"
            
            # Extract salary (if available)
            salary_elem = card.find('span', {'class': lambda x: x and 'salary' in x.lower() if x else False})
            salary_info = {'min_salary': None, 'max_salary': None, 'salary_type': None}
            if salary_elem:
                salary_text = self._clean_text(salary_elem.get_text())
                salary_info = self._extract_salary(salary_text)
                
            # Extract job description snippet
            description_elem = card.find('div', {'class': lambda x: x and 'job-snippet' in x if x else False})
            description = self._clean_text(description_elem.get_text()) if description_elem else ""
            
            # Extract posting date
            date_elem = card.find('span', {'class': lambda x: x and 'date' in x.lower() if x else False})
            posting_date = self._clean_text(date_elem.get_text()) if date_elem else "Unknown"
            
            return {
                'title': title,
                'company': company,
                'location': location,
                'description': description,
                'url': job_url,
                'source': 'Indeed',
                'posting_date': posting_date,
                'skills': self._extract_skills(description),
                **salary_info
            }
            
        except Exception as e:
            logger.error(f"Error parsing job card: {e}")
            return None
            
    def scrape_job_details(self, job_url: str) -> Dict:
        """
        Scrape detailed information for a specific Indeed job posting.
        
        Args:
            job_url: URL of the job posting
            
        Returns:
            Dictionary with detailed job information
        """
        soup = self._make_request(job_url)
        
        if not soup:
            return {'error': 'Failed to fetch job details'}
            
        try:
            # Extract full job description
            description_elem = soup.find('div', {'class': lambda x: x and 'jobsearch-jobDescriptionText' in x if x else False})
            full_description = self._clean_text(description_elem.get_text()) if description_elem else ""
            
            # Extract job requirements
            requirements = []
            req_sections = soup.find_all(['ul', 'ol'], string=lambda text: 'requirement' in text.lower() if text else False)
            for section in req_sections:
                req_items = section.find_all('li')
                requirements.extend([self._clean_text(item.get_text()) for item in req_items])
                
            # Extract benefits (if available)
            benefits = []
            benefit_sections = soup.find_all(['ul', 'ol'], string=lambda text: 'benefit' in text.lower() if text else False)
            for section in benefit_sections:
                benefit_items = section.find_all('li')
                benefits.extend([self._clean_text(item.get_text()) for item in benefit_items])
                
            # Extract company information
            company_elem = soup.find('div', {'class': lambda x: x and 'company' in x.lower() if x else False})
            company_info = self._clean_text(company_elem.get_text()) if company_elem else ""
            
            return {
                'full_description': full_description,
                'requirements': requirements,
                'benefits': benefits,
                'company_info': company_info,
                'skills': self._extract_skills(full_description),
                'detailed_skills': self._extract_skills(full_description + ' ' + ' '.join(requirements))
            }
            
        except Exception as e:
            logger.error(f"Error scraping job details: {e}")
            return {'error': str(e)}
            
    def search_by_salary_range(self, query: str, location: str, min_salary: int, max_salary: int = None) -> List[Dict]:
        """
        Search for jobs within a specific salary range.
        
        Args:
            query: Job search query
            location: Location to search
            min_salary: Minimum salary
            max_salary: Maximum salary (optional)
            
        Returns:
            List of job dictionaries within the salary range
        """
        # Indeed's salary filter format
        salary_filter = f"${min_salary:,}"
        if max_salary:
            salary_filter += f"-${max_salary:,}"
            
        params = {
            'q': query,
            'l': location,
            'salary': salary_filter
        }
        
        search_url = f"{self.base_url}/jobs"
        soup = self._make_request(search_url, params)
        
        if not soup:
            return []
            
        jobs = []
        job_cards = soup.find_all('div', {'class': lambda x: x and 'job_seen_beacon' in x})
        
        for card in job_cards:
            job_data = self._parse_job_card(card)
            if job_data:
                jobs.append(job_data)
                
        return jobs
