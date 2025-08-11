"""
Demo with sample data for Job Market Analyzer.
Shows analysis capabilities using generated sample data.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from analyzers.salary_analyzer import SalaryAnalyzer
from database.db_manager import DatabaseManager

def demo_with_sample_data():
    """Demonstrate job market analysis with sample data."""
    print("🎯 Job Market Analyzer - Analysis Demo")
    print("=" * 50)
    
    # Load sample data
    print("\n1. Loading sample job data...")
    with open('data/sample_jobs.json', 'r') as f:
        jobs = json.load(f)
    
    print(f"   ✅ Loaded {len(jobs)} sample jobs")
    
    # Save to database
    print("\n2. Saving to database...")
    db_manager = DatabaseManager()
    saved_count = db_manager.save_jobs(jobs)
    print(f"   ✅ Saved {saved_count} jobs to database")
    
    # Perform salary analysis
    print("\n3. Performing salary analysis...")
    analyzer = SalaryAnalyzer()
    analyzer.load_data(jobs)
    
    # Overall statistics
    stats = analyzer.get_salary_statistics()
    print(f"\n📊 Overall Salary Statistics:")
    print(f"   Jobs analyzed: {stats['count']}")
    print(f"   Average salary: ${stats['mean_salary']:,.0f}")
    print(f"   Median salary: ${stats['median_salary']:,.0f}")
    print(f"   Salary range: ${stats['min_salary']:,.0f} - ${stats['max_salary']:,.0f}")
    print(f"   75th percentile: ${stats['percentile_75']:,.0f}")
    
    # Salary ranges breakdown
    print(f"\n💰 Salary Distribution:")
    for range_name, count in stats['salary_ranges'].items():
        percentage = (count / stats['count']) * 100
        print(f"   {range_name}: {count} jobs ({percentage:.1f}%)")
    
    # Analysis by location
    print(f"\n🌍 Top Paying Locations:")
    location_stats = analyzer.analyze_by_location()
    for i, row in location_stats.head(5).iterrows():
        print(f"   {row['city']}: ${row['mean_salary']:,.0f} avg ({row['job_count']} jobs)")
    
    # Analysis by job title
    print(f"\n👩‍💻 Top Paying Roles:")
    title_stats = analyzer.analyze_by_job_title()
    for i, row in title_stats.head(5).iterrows():
        role_name = row.name if hasattr(row, 'name') else title_stats.index[i]
        print(f"   {role_name}: ${row['mean_salary']:,.0f} avg ({row['job_count']} jobs)")
    
    # Top companies
    print(f"\n🏢 Top Paying Companies:")
    company_stats = analyzer.get_top_paying_companies()
    for i, row in company_stats.head(5).iterrows():
        print(f"   {row['company']}: ${row['mean_salary']:,.0f} avg ({row['salary_postings']} jobs)")
    
    # Skills analysis
    print(f"\n🛠️ Most In-Demand Skills:")
    all_skills = []
    for job in jobs:
        all_skills.extend(job.get('skills', []))
    
    from collections import Counter
    skill_counts = Counter(all_skills)
    for skill, count in skill_counts.most_common(8):
        percentage = (count / len(jobs)) * 100
        print(f"   {skill}: {count} jobs ({percentage:.1f}%)")
    
    # Database statistics
    print(f"\n📈 Database Statistics:")
    db_stats = db_manager.get_database_stats()
    print(f"   Total jobs: {db_stats['total_jobs']}")
    print(f"   Unique companies: {db_stats['unique_companies']}")
    print(f"   Unique locations: {db_stats['unique_locations']}")
    print(f"   Jobs with salary data: {db_stats['jobs_with_salary']}")
    
    # Generate comprehensive report
    print(f"\n📋 Generating analysis report...")
    report = analyzer.generate_salary_report('reports/salary_analysis_demo.json')
    print(f"   ✅ Report saved to: reports/salary_analysis_demo.json")
    
    # Specific insights
    print(f"\n🎯 Key Market Insights:")
    
    # Tech vs Non-tech
    tech_titles = ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'Machine Learning Engineer']
    tech_jobs = [job for job in jobs if any(title in job['title'] for title in tech_titles)]
    if tech_jobs:
        tech_analyzer = SalaryAnalyzer()
        tech_analyzer.load_data(tech_jobs)
        tech_stats = tech_analyzer.get_salary_statistics()
        print(f"   Tech roles average: ${tech_stats['mean_salary']:,.0f} ({len(tech_jobs)} jobs)")
    
    # Remote vs On-site
    remote_jobs = [job for job in jobs if 'Remote' in job['location']]
    if remote_jobs:
        remote_analyzer = SalaryAnalyzer()
        remote_analyzer.load_data(remote_jobs)
        remote_stats = remote_analyzer.get_salary_statistics()
        print(f"   Remote jobs average: ${remote_stats['mean_salary']:,.0f} ({len(remote_jobs)} jobs)")
    
    print(f"\n✨ Analysis complete!")
    print(f"\nTo run your own analysis:")
    print(f"   python src/main.py analyze --type salary")
    print(f"   python src/main.py insights --query 'Data Scientist' --location 'San Francisco'")

if __name__ == '__main__':
    demo_with_sample_data()
