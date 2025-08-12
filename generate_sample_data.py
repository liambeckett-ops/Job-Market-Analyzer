"""
Sample data generator for Job Market Analyzer demo.
Creates realistic job data for testing analysis features.
"""

import json
import random
from datetime import datetime, timedelta

def generate_sample_jobs(num_jobs=100):
    """Generate sample job data for demonstration."""
    
    # Sample data pools
    job_titles = [
        "Software Engineer", "Senior Software Engineer", "Python Developer",
        "Data Scientist", "Machine Learning Engineer", "DevOps Engineer",
        "Frontend Developer", "Backend Developer", "Full Stack Developer",
        "Product Manager", "QA Engineer", "Security Engineer"
    ]
    
    companies = [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix",
        "Uber", "Airbnb", "Spotify", "Tesla", "SpaceX", "Twitter",
        "LinkedIn", "Salesforce", "Adobe", "Intel", "Cisco", "IBM"
    ]
    
    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
        "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
        "Remote", "Remote - US", "Remote - Global"
    ]
    
    skills_pool = [
        "Python", "JavaScript", "Java", "C++", "React", "Angular", "Vue.js",
        "Node.js", "Django", "Flask", "FastAPI", "SQL", "PostgreSQL", "MongoDB",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "Linux",
        "Machine Learning", "TensorFlow", "PyTorch", "Pandas", "NumPy"
    ]
    
    job_descriptions = [
        "Join our team to build scalable web applications using modern technologies.",
        "We're looking for a passionate developer to work on cutting-edge projects.",
        "Help us revolutionize the industry with innovative software solutions.",
        "Work with a talented team to deliver high-quality products to millions of users.",
        "Be part of our mission to transform how people interact with technology."
    ]
    
    jobs = []
    
    for i in range(num_jobs):
        # Generate random job data
        title = random.choice(job_titles)
        company = random.choice(companies)
        location = random.choice(locations)
        description = random.choice(job_descriptions)
        
        # Generate salary based on role level
        base_salary = {
            "Software Engineer": (70000, 120000),
            "Senior Software Engineer": (120000, 180000),
            "Python Developer": (75000, 130000),
            "Data Scientist": (90000, 160000),
            "Machine Learning Engineer": (110000, 190000),
            "DevOps Engineer": (85000, 150000),
            "Frontend Developer": (70000, 125000),
            "Backend Developer": (80000, 140000),
            "Full Stack Developer": (75000, 135000),
            "Product Manager": (100000, 170000),
            "QA Engineer": (60000, 110000),
            "Security Engineer": (90000, 160000)
        }
        
        salary_range = base_salary.get(title, (70000, 120000))
        min_salary = random.randint(salary_range[0], salary_range[1] - 20000)
        max_salary = min_salary + random.randint(10000, 30000)
        
        # Location salary adjustment
        if "San Francisco" in location or "New York" in location:
            min_salary = int(min_salary * 1.3)
            max_salary = int(max_salary * 1.3)
        elif "Remote" in location:
            min_salary = int(min_salary * 1.1)
            max_salary = int(max_salary * 1.1)
        
        # Generate skills for this job
        num_skills = random.randint(3, 8)
        job_skills = random.sample(skills_pool, num_skills)
        
        # Generate posting date
        days_ago = random.randint(1, 30)
        posting_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        job = {
            "title": title,
            "company": company,
            "location": location,
            "description": description,
            "url": f"https://example.com/job/{i}",
            "source": "Sample Data",
            "posting_date": posting_date,
            "min_salary": min_salary,
            "max_salary": max_salary,
            "salary_type": "yearly",
            "skills": job_skills
        }
        
        jobs.append(job)
    
    return jobs

def main():
    """Generate and save sample data."""
    print("Generating sample job data...")
    
    jobs = generate_sample_jobs(150)
    
    # Save to JSON file
    with open('data/sample_jobs.json', 'w') as f:
        json.dump(jobs, f, indent=2)
    
    print(f"Generated {len(jobs)} sample jobs")
    print("Saved to: data/sample_jobs.json")
    
    # Quick stats
    companies = set(job['company'] for job in jobs)
    locations = set(job['location'] for job in jobs)
    all_skills = []
    for job in jobs:
        all_skills.extend(job['skills'])
    
    from collections import Counter
    top_skills = Counter(all_skills).most_common(5)
    
    print(f"\n Sample Data Overview:")
    print(f"Companies: {len(companies)}")
    print(f"Locations: {len(locations)}")
    print(f"Top skills: {', '.join([skill for skill, count in top_skills])}")
    
    # Calculate salary stats
    salaries = [(job['min_salary'] + job['max_salary']) / 2 for job in jobs]
    avg_salary = sum(salaries) / len(salaries)
    print(f"Average salary: ${avg_salary:,.0f}")

if __name__ == '__main__':
    main()
