from .base_scraper import BaseScraper

class GlassdoorScraper(BaseScraper):
    def fetch_jobs(self, query, location):
        # Placeholder: Implement actual scraping logic here
        return [{
            'title': 'Product Manager',
            'company': 'Glassdoor',
            'location': location,
            'description': 'Sample job from Glassdoor',
        }]

if __name__ == "__main__":
    scraper = GlassdoorScraper()
    print(scraper.fetch_jobs('manager', 'remote'))
