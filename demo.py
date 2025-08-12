"""
Demo script for Job Market Analyzer.
Shows basic usage and functionality testing.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from scrapers.indeed_scraper import IndeedScraper
from analyzers.salary_analyzer import SalaryAnalyzer
from database.db_manager import DatabaseManager

def demo_job_scraping():
    """Demonstrate job scraping functionality."""
    print("Job Market Analyzer Demo")
    print("=" * 50)
    
    # Initialize scraper
    print("\n1. Initializing Indeed scraper...")
    scraper = IndeedScraper()
    
    # Scrape some jobs (limited for demo)
    print("\n2. Scraping sample jobs...")
    print("Query: 'python developer'")
    print("Location: 'remote'")
    print("Pages: 2 (for demo)")
    
    try:
        jobs = scraper.scrape_jobs("python developer", "remote", num_pages=2)
        print(f"Found {len(jobs)} jobs")
        
        if jobs:
            # Show sample job
            sample_job = jobs[0]
            print(f"\n Sample Job:")
            print(f"Title: {sample_job.get('title', 'N/A')}")
            print(f"Company: {sample_job.get('company', 'N/A')}")
            print(f"Location: {sample_job.get('location', 'N/A')}")
            print(f"Skills: {', '.join(sample_job.get('skills', [])[:5])}")
            
            # Save to database
            print(f"\n3. Saving to database...")
            db_manager = DatabaseManager()
            saved_count = db_manager.save_jobs(jobs)
            print(f"Saved {saved_count} new jobs")
            
            # Analyze salary data
            print(f"\n4. Analyzing salary data...")
            analyzer = SalaryAnalyzer()
            analyzer.load_data(jobs)
            
            stats = analyzer.get_salary_statistics()
            if stats.get('count', 0) > 0:
                print(f"Salary Analysis:")
                print(f"Jobs with salary: {stats['count']}")
                print(f"Median salary: ${stats.get('median_salary', 0):,.0f}")
                print(f"Salary range: ${stats.get('min_salary', 0):,.0f} - ${stats.get('max_salary', 0):,.0f}")
            else:
                print(f"No salary data available in this sample")
                
            # Skills analysis
            all_skills = []
            for job in jobs:
                all_skills.extend(job.get('skills', []))
            
            if all_skills:
                from collections import Counter
                skill_counts = Counter(all_skills)
                print(f"\n5. Top Skills in Demand:")
                for skill, count in skill_counts.most_common(5):
                    print(f"   {skill}: {count} jobs")
            
            # Database stats
            print(f"\n6. Database Statistics:")
            stats = db_manager.get_database_stats()
            print(f"Total jobs in DB: {stats['total_jobs']}")
            print(f"Unique companies: {stats['unique_companies']}")
            print(f"Jobs with salary: {stats['jobs_with_salary']}")
            
        else:
            print("No jobs found - this might be due to rate limiting or site changes")
            
    except Exception as e:
        print(f"Error during scraping: {e}")
        print("This might be due to:")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- Network issues")
        print("- Changes in Indeed's website structure")
        
    print(f"\n Demo complete!")
    print(f"\nTo use the full application:")
    print(f"python src/main.py scrape --site indeed --query 'data scientist' --location 'New York'")
    print(f"python src/main.py analyze --type salary")
    print(f"python src/main.py insights --query 'software engineer' --location 'remote'")

if __name__ == '__main__':
    demo_job_scraping()