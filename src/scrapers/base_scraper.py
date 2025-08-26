<<<<<<< HEAD
"""
Base scraper class for job market data extraction.
Provides common functionality for all job board scrapers.
"""

import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """
    Abstract base class for job scrapers.
    Implements common functionality like rate limiting, error handling, and data cleaning.
    """
    
    def __init__(self, base_url: str, delay_range: tuple = (1, 3)):
        """
        Initialize the base scraper.
        
        Args:
            base_url: The base URL of the job site
            delay_range: Tuple of (min, max) seconds to wait between requests
        """
        self.base_url = base_url
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _make_request(self, url: str, params: Dict = None) -> Optional[BeautifulSoup]:
        """
        Make a HTTP request with rate limiting and error handling.
        
        Args:
            url: The URL to request
            params: Query parameters
            
        Returns:
            BeautifulSoup object or None if request failed
        """
        try:
            # Rate limiting
            delay = random.uniform(*self.delay_range)
            time.sleep(delay)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return BeautifulSoup(response.content, 'html.parser')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
            
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        return text.strip()
        
    def _extract_salary(self, salary_text: str) -> Dict[str, Optional[float]]:
        """
        Extract salary information from text.
        
        Args:
            salary_text: Text containing salary information
            
        Returns:
            Dictionary with min_salary, max_salary, and salary_type
        """
        import re
        
        if not salary_text:
            return {'min_salary': None, 'max_salary': None, 'salary_type': None}
            
        # Remove common prefixes
        salary_text = re.sub(r'^(salary|pay|compensation):?\s*', '', salary_text.lower())
        
        # Extract salary ranges
        salary_pattern = r'[\$£€]?([\d,]+)(?:\s*[-–]\s*[\$£€]?([\d,]+))?'
        matches = re.findall(salary_pattern, salary_text.replace(',', ''))
        
        if matches:
            match = matches[0]
            min_salary = float(match[0]) if match[0] else None
            max_salary = float(match[1]) if match[1] else min_salary
            
            # Determine if it's hourly, yearly, etc.
            salary_type = 'yearly'
            if any(word in salary_text for word in ['hour', 'hr', '/h']):
                salary_type = 'hourly'
            elif any(word in salary_text for word in ['month', '/m']):
                salary_type = 'monthly'
                
            return {
                'min_salary': min_salary,
                'max_salary': max_salary,
                'salary_type': salary_type
            }
            
        return {'min_salary': None, 'max_salary': None, 'salary_type': None}
        
    def _extract_skills(self, description: str) -> List[str]:
        """
        Extract technical skills from job description.
        
        Args:
            description: Job description text
            
        Returns:
            List of identified skills
        """
        if not description:
            return []
            
        # Common technical skills to look for
        skills_keywords = [
            # Programming Languages
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'typescript', 'scala', 'kotlin', 'swift', 'objective-c', 'r', 'matlab',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'fastapi', 'spring', 'asp.net', 'bootstrap', 'jquery',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'oracle', 'sqlite', 'nosql', 'dynamodb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'terraform', 'ansible', 'linux', 'bash', 'ci/cd',
            
            # Data Science & ML
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'jupyter',
            'matplotlib', 'seaborn', 'tableau', 'power bi', 'spark', 'hadoop',
            
            # Other Technologies
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum', 'jira'
        ]
        
        description_lower = description.lower()
        found_skills = []
        
        for skill in skills_keywords:
            if skill in description_lower:
                found_skills.append(skill.title())
                
        return list(set(found_skills))  # Remove duplicates
        
    @abstractmethod
    def scrape_jobs(self, query: str, location: str, num_pages: int = 5) -> List[Dict]:
        """
        Scrape job listings for a given query and location.
        
        Args:
            query: Job search query (e.g., "software engineer")
            location: Location to search (e.g., "New York, NY")
            num_pages: Number of pages to scrape
            
        Returns:
            List of job dictionaries
        """
        pass
        
    @abstractmethod
    def scrape_job_details(self, job_url: str) -> Dict:
        """
        Scrape detailed information for a specific job posting.
        
        Args:
            job_url: URL of the job posting
            
        Returns:
            Dictionary with detailed job information
        """
=======
import abc

class BaseScraper(abc.ABC):
    """Abstract base class for job site scrapers."""
    @abc.abstractmethod
    def fetch_jobs(self, query, location):
>>>>>>> d2a851d (Initial project refactor: add scrapers, database manager, and analysis features)
        pass
