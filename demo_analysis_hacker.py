"""
Demo with sample data for Job Market Analyzer.
Shows analysis capabilities using generated sample data.
Built with hacker aesthetic + tech minimalist vibes.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from analyzers.salary_analyzer import SalaryAnalyzer
from database.db_manager import DatabaseManager

# ANSI Color codes for terminal styling
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def print_banner():
    """Display hacker-style ASCII banner."""
    banner = f"""{Colors.CYAN}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════════╗
    ║                    JOB MARKET ANALYZER                    ║
    ║                   {Colors.DIM}[ RECON MODE ACTIVE ]{Colors.BOLD}                   ║
    ╚═══════════════════════════════════════════════════════════╝{Colors.RESET}
    {Colors.DIM}> Initiating market intelligence gathering...{Colors.RESET}
    """
    print(banner)

def status_msg(msg, status="INFO"):
    """Print status message with hacker styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "SUCCESS":
        print(f"{Colors.GREEN}[{timestamp}] >> {msg}{Colors.RESET}")
    elif status == "ERROR":
        print(f"{Colors.RED}[{timestamp}] !! {msg}{Colors.RESET}")
    elif status == "WARN":
        print(f"{Colors.YELLOW}[{timestamp}] ?? {msg}{Colors.RESET}")
    else:
        print(f"{Colors.CYAN}[{timestamp}] -- {msg}{Colors.RESET}")

def section_header(title):
    """Print section header with style."""
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'─' * 60}")
    print(f"│ {title.upper()}")
    print(f"{'─' * 60}{Colors.RESET}")

def demo_with_sample_data():
    """Demonstrate job market analysis with sample data."""
    print_banner()
    
    # Load sample data
    section_header("DATA ACQUISITION")
    status_msg("Accessing sample job dataset...")
    with open('data/sample_jobs.json', 'r') as f:
        jobs = json.load(f)
    
    status_msg(f"Successfully loaded {len(jobs)} job records", "SUCCESS")
    
    # Save to database
    section_header("DATABASE INTEGRATION")
    status_msg("Establishing database connection...")
    db_manager = DatabaseManager()
    saved_count = db_manager.save_jobs(jobs)
    status_msg(f"Stored {saved_count} records in secure database", "SUCCESS")
    
    # Perform salary analysis
    section_header("MARKET INTELLIGENCE ANALYSIS")
    status_msg("Initializing salary analysis engine...")
    analyzer = SalaryAnalyzer()
    analyzer.load_data(jobs)
    
    # Overall statistics
    stats = analyzer.get_salary_statistics()
    print(f"\n{Colors.WHITE}{Colors.BOLD}FINANCIAL INTELLIGENCE REPORT{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.WHITE}Target Records Analyzed: {Colors.GREEN}{stats['count']}{Colors.RESET}")
    print(f"{Colors.WHITE}Average Compensation: {Colors.GREEN}${stats['mean_salary']:,.0f}{Colors.RESET}")
    print(f"{Colors.WHITE}Median Baseline: {Colors.YELLOW}${stats['median_salary']:,.0f}{Colors.RESET}")
    print(f"{Colors.WHITE}Range Spectrum: {Colors.CYAN}${stats['min_salary']:,.0f} - ${stats['max_salary']:,.0f}{Colors.RESET}")
    print(f"{Colors.WHITE}Elite Threshold (75%): {Colors.MAGENTA}${stats['percentile_75']:,.0f}{Colors.RESET}")
    
    # Salary ranges breakdown
    print(f"\n{Colors.WHITE}{Colors.BOLD}COMPENSATION DISTRIBUTION{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    for range_name, count in stats['salary_ranges'].items():
        percentage = (count / stats['count']) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{Colors.WHITE}{range_name:15}: {Colors.CYAN}{count:3d} {Colors.DIM}[{bar[:20]}] {percentage:4.1f}%{Colors.RESET}")
    
    # Analysis by location
    print(f"\n{Colors.WHITE}{Colors.BOLD}HIGH-VALUE LOCATION INTEL{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    location_stats = analyzer.analyze_by_location()
    for i, row in location_stats.head(5).iterrows():
        print(f"{Colors.CYAN}► {Colors.WHITE}{row['city']:20}: {Colors.GREEN}${row['mean_salary']:,.0f} {Colors.DIM}({row['job_count']} positions){Colors.RESET}")
    
    # Analysis by job title
    print(f"\n{Colors.WHITE}{Colors.BOLD}PREMIUM ROLE CLASSIFICATION{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    title_stats = analyzer.analyze_by_job_title()
    for i, row in title_stats.head(5).iterrows():
        role_name = row.name if hasattr(row, 'name') else title_stats.index[i]
        print(f"{Colors.MAGENTA}#{i+1:2d} {Colors.WHITE}{str(role_name)[:25]:25}: {Colors.GREEN}${row['mean_salary']:,.0f} {Colors.DIM}({row['job_count']} roles){Colors.RESET}")
    
    # Top companies
    print(f"\n{Colors.WHITE}{Colors.BOLD}CORPORATE COMPENSATION LEADERS{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    company_stats = analyzer.get_top_paying_companies()
    for i, row in company_stats.head(5).iterrows():
        print(f"{Colors.YELLOW}▲ {Colors.WHITE}{row['company']:15}: {Colors.GREEN}${row['mean_salary']:,.0f} {Colors.DIM}({row['salary_postings']} postings){Colors.RESET}")
    
    # Top skills analysis
    print(f"\n{Colors.WHITE}{Colors.BOLD}CRITICAL SKILL MATRIX{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    from collections import Counter
    all_skills = []
    for job in jobs:
        if 'skills' in job and job['skills']:
            all_skills.extend(job['skills'])
    
    skill_counts = Counter(all_skills)
    for skill, count in skill_counts.most_common(8):
        percentage = (count / len(jobs)) * 100
        if percentage > 25:
            demand_level = f"{Colors.RED}🔥 CRITICAL{Colors.RESET}"
        elif percentage > 20:
            demand_level = f"{Colors.YELLOW}⚡ HIGH{Colors.RESET}"
        else:
            demand_level = f"{Colors.DIM}● MODERATE{Colors.RESET}"
        print(f"{Colors.CYAN}[{percentage:4.1f}%] {Colors.WHITE}{skill:15}: {Colors.YELLOW}{count:3d} requests {demand_level}")
    
    # Database statistics
    print(f"\n{Colors.WHITE}{Colors.BOLD}SYSTEM STATUS REPORT{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    db_stats = db_manager.get_database_stats()
    print(f"{Colors.WHITE}Total Records: {Colors.GREEN}{db_stats['total_jobs']}{Colors.RESET}")
    print(f"{Colors.WHITE}Unique Entities: {Colors.GREEN}{db_stats['unique_companies']}{Colors.RESET}")
    print(f"{Colors.WHITE}Location Coverage: {Colors.GREEN}{db_stats['unique_locations']}{Colors.RESET}")
    print(f"{Colors.WHITE}Salary Data Points: {Colors.GREEN}{db_stats['jobs_with_salary']}{Colors.RESET}")
    
    # Generate comprehensive report
    section_header("INTELLIGENCE EXPORT")
    status_msg("Generating comprehensive analysis report...")
    report = analyzer.generate_salary_report('reports/salary_analysis_demo.json')
    status_msg("Report exported to: reports/salary_analysis_demo.json", "SUCCESS")
    
    # Specific insights
    print(f"\n{Colors.WHITE}{Colors.BOLD}MARKET INTELLIGENCE SUMMARY{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    
    # Tech vs Non-tech
    tech_titles = ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'Machine Learning Engineer']
    tech_jobs = [job for job in jobs if any(title in job['title'] for title in tech_titles)]
    if tech_jobs:
        tech_analyzer = SalaryAnalyzer()
        tech_analyzer.load_data(tech_jobs)
        tech_stats = tech_analyzer.get_salary_statistics()
        print(f"{Colors.CYAN}▶ Tech Sector Average: {Colors.GREEN}${tech_stats['mean_salary']:,.0f} {Colors.DIM}({len(tech_jobs)} positions){Colors.RESET}")
    
    # Remote vs On-site
    remote_jobs = [job for job in jobs if 'Remote' in job['location']]
    if remote_jobs:
        remote_analyzer = SalaryAnalyzer()
        remote_analyzer.load_data(remote_jobs)
        remote_stats = remote_analyzer.get_salary_statistics()
        print(f"{Colors.CYAN}▶ Remote Work Premium: {Colors.GREEN}${remote_stats['mean_salary']:,.0f} {Colors.DIM}({len(remote_jobs)} positions){Colors.RESET}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'─' * 60}")
    print(f"│ MARKET ANALYSIS COMPLETE - STANDING BY")
    print(f"{'─' * 60}{Colors.RESET}")
    
    print(f"\n{Colors.DIM}> Run your own deep dive analysis:")
    print(f"  python src/main.py analyze --type salary")
    print(f"  python src/main.py insights --query 'Data Scientist' --location 'San Francisco'{Colors.RESET}")

if __name__ == '__main__':
    demo_with_sample_data()
