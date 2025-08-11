"""
Main application for Job Market Analyzer.
Provides command-line interface for scraping and analyzing job market data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

# Import our modules
from scrapers.indeed_scraper import IndeedScraper
from analyzers.salary_analyzer import SalaryAnalyzer
from database.db_manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('job_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class JobMarketAnalyzer:
    """
    Main application class for job market analysis.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager('data/jobs.db')
        self.scrapers = {
            'indeed': IndeedScraper()
        }
        self.analyzers = {
            'salary': SalaryAnalyzer()
        }
        
    def scrape_jobs(self, site: str, query: str, location: str, pages: int = 5) -> List[Dict]:
        """
        Scrape jobs from specified site.
        
        Args:
            site: Job site to scrape ('indeed', 'linkedin')
            query: Job search query
            location: Location to search
            pages: Number of pages to scrape
            
        Returns:
            List of scraped job data
        """
        if site not in self.scrapers:
            raise ValueError(f"Unsupported site: {site}. Available: {list(self.scrapers.keys())}")
            
        scraper = self.scrapers[site]
        logger.info(f"Starting to scrape {site} for '{query}' in '{location}'")
        
        try:
            jobs = scraper.scrape_jobs(query, location, pages)
            
            # Save to database
            self.db_manager.save_jobs(jobs)
            
            logger.info(f"Successfully scraped and saved {len(jobs)} jobs")
            return jobs
            
        except Exception as e:
            logger.error(f"Error scraping jobs: {e}")
            raise
            
    def analyze_salary_data(self, job_title: str = None, location: str = None, output_dir: str = 'reports') -> Dict:
        """
        Analyze salary data from database.
        
        Args:
            job_title: Filter by job title (optional)
            location: Filter by location (optional)
            output_dir: Directory to save reports
            
        Returns:
            Analysis results
        """
        # Load data from database
        jobs_data = self.db_manager.get_jobs(job_title=job_title, location=location)
        
        if not jobs_data:
            logger.warning("No job data found in database")
            return {'error': 'No data available'}
            
        # Perform salary analysis
        analyzer = self.analyzers['salary']
        analyzer.load_data(jobs_data)
        
        # Generate comprehensive report
        output_path = Path(output_dir) / 'salary_analysis.json'
        output_path.parent.mkdir(exist_ok=True)
        
        report = analyzer.generate_salary_report(str(output_path))
        
        logger.info(f"Salary analysis complete. Report saved to {output_path}")
        return report
        
    def get_market_insights(self, query: str, location: str) -> Dict:
        """
        Get comprehensive market insights for a specific job query.
        
        Args:
            query: Job search query
            location: Location to analyze
            
        Returns:
            Market insights dictionary
        """
        jobs_data = self.db_manager.get_jobs(job_title=query, location=location)
        
        if not jobs_data:
            return {'error': 'No data available for the specified query'}
            
        insights = {
            'total_jobs': len(jobs_data),
            'unique_companies': len(set(job['company'] for job in jobs_data if job['company'] != 'Unknown')),
            'locations': {},
            'skills_demand': {},
            'salary_insights': {}
        }
        
        # Analyze locations
        location_counts = {}
        for job in jobs_data:
            loc = job.get('location', 'Unknown')
            location_counts[loc] = location_counts.get(loc, 0) + 1
        insights['locations'] = dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Analyze skills
        all_skills = []
        for job in jobs_data:
            if job.get('skills'):
                all_skills.extend(job['skills'])
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        insights['skills_demand'] = dict(skill_counts.most_common(15))
        
        # Salary insights
        salary_analyzer = SalaryAnalyzer()
        salary_analyzer.load_data(jobs_data)
        insights['salary_insights'] = salary_analyzer.get_salary_statistics()
        
        return insights
        
    def export_data(self, format: str, output_path: str, filters: Dict = None):
        """
        Export job data in specified format.
        
        Args:
            format: Export format ('csv', 'json', 'excel')
            output_path: Output file path
            filters: Optional filters to apply
        """
        jobs_data = self.db_manager.get_jobs(**(filters or {}))
        
        if not jobs_data:
            logger.warning("No data to export")
            return
            
        import pandas as pd
        df = pd.DataFrame(jobs_data)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Data exported to {output_path}")

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(description='Job Market Analyzer')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape job data')
    scrape_parser.add_argument('--site', required=True, choices=['indeed'], help='Job site to scrape')
    scrape_parser.add_argument('--query', required=True, help='Job search query')
    scrape_parser.add_argument('--location', required=True, help='Location to search')
    scrape_parser.add_argument('--pages', type=int, default=5, help='Number of pages to scrape')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze job data')
    analyze_parser.add_argument('--type', required=True, choices=['salary'], help='Analysis type')
    analyze_parser.add_argument('--job-title', help='Filter by job title')
    analyze_parser.add_argument('--location', help='Filter by location')
    analyze_parser.add_argument('--output', default='reports', help='Output directory')
    
    # Insights command
    insights_parser = subparsers.add_parser('insights', help='Get market insights')
    insights_parser.add_argument('--query', required=True, help='Job query')
    insights_parser.add_argument('--location', required=True, help='Location')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export job data')
    export_parser.add_argument('--format', required=True, choices=['csv', 'json', 'excel'], help='Export format')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--job-title', help='Filter by job title')
    export_parser.add_argument('--location', help='Filter by location')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    analyzer = JobMarketAnalyzer()
    
    try:
        if args.command == 'scrape':
            jobs = analyzer.scrape_jobs(args.site, args.query, args.location, args.pages)
            print(f"✅ Successfully scraped {len(jobs)} jobs")
            
        elif args.command == 'analyze':
            if args.type == 'salary':
                report = analyzer.analyze_salary_data(args.job_title, args.location, args.output)
                if 'error' not in report:
                    print(f"✅ Salary analysis complete. Report saved to {args.output}/salary_analysis.json")
                    
                    # Print summary
                    overview = report.get('overview', {})
                    print(f"\n📊 Quick Summary:")
                    print(f"Jobs analyzed: {overview.get('count', 0)}")
                    print(f"Median salary: ${overview.get('median_salary', 0):,.0f}")
                    print(f"Mean salary: ${overview.get('mean_salary', 0):,.0f}")
                else:
                    print(f"❌ {report['error']}")
                    
        elif args.command == 'insights':
            insights = analyzer.get_market_insights(args.query, args.location)
            if 'error' not in insights:
                print(f"\n🎯 Market Insights for '{args.query}' in '{args.location}':")
                print(f"Total jobs found: {insights['total_jobs']}")
                print(f"Unique companies: {insights['unique_companies']}")
                
                print(f"\n🏆 Top Skills in Demand:")
                for skill, count in list(insights['skills_demand'].items())[:5]:
                    print(f"  {skill}: {count} jobs")
                    
                salary_info = insights.get('salary_insights', {})
                if salary_info.get('median_salary'):
                    print(f"\n💰 Salary Info:")
                    print(f"  Median: ${salary_info['median_salary']:,.0f}")
                    print(f"  Range: ${salary_info['min_salary']:,.0f} - ${salary_info['max_salary']:,.0f}")
            else:
                print(f"❌ {insights['error']}")
                
        elif args.command == 'export':
            filters = {}
            if args.job_title:
                filters['job_title'] = args.job_title
            if args.location:
                filters['location'] = args.location
                
            analyzer.export_data(args.format, args.output, filters)
            print(f"✅ Data exported to {args.output}")
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
