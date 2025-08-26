from ..database.db_manager import DatabaseManager
from ..scrapers.indeed_scraper import IndeedScraper
from ..scrapers.linkedin_scraper import LinkedInScraper
from ..scrapers.glassdoor_scraper import GlassdoorScraper

class JobAnalyzer:
    def __init__(self):
        self.db = DatabaseManager()
        self.scrapers = [
            ('Indeed', IndeedScraper()),
            ('LinkedIn', LinkedInScraper()),
            ('Glassdoor', GlassdoorScraper()),
        ]

    def fetch_and_store_jobs(self, query, location):
        for source, scraper in self.scrapers:
            jobs = scraper.fetch_jobs(query, location)
            for job in jobs:
                self.db.insert_job(job, source)

    def analyze(self):
        print('Jobs by company:', self.db.count_jobs_by_company())
        print('Jobs by location:', self.db.count_jobs_by_location())

if __name__ == "__main__":
    analyzer = JobAnalyzer()
    analyzer.fetch_and_store_jobs('engineer', 'remote')
    analyzer.analyze()
