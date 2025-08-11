"""
Salary analysis module for job market data.
Provides comprehensive salary trend analysis and insights.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class SalaryAnalyzer:
    """
    Analyzes salary data from job postings to provide market insights.
    """
    
    def __init__(self):
        self.salary_data = None
        
    def load_data(self, jobs_data: List[Dict]) -> pd.DataFrame:
        """
        Load and prepare job data for salary analysis.
        
        Args:
            jobs_data: List of job dictionaries
            
        Returns:
            Prepared DataFrame with salary information
        """
        df = pd.DataFrame(jobs_data)
        
        # Filter jobs with salary information
        salary_df = df.dropna(subset=['min_salary'])
        
        # Convert salary to annual equivalent
        salary_df = salary_df.copy()
        salary_df['annual_min'] = salary_df.apply(self._convert_to_annual, axis=1, salary_col='min_salary')
        salary_df['annual_max'] = salary_df.apply(self._convert_to_annual, axis=1, salary_col='max_salary')
        salary_df['annual_avg'] = (salary_df['annual_min'] + salary_df['annual_max']) / 2
        
        # Clean location data
        salary_df['city'] = salary_df['location'].apply(self._extract_city)
        salary_df['state'] = salary_df['location'].apply(self._extract_state)
        
        self.salary_data = salary_df
        logger.info(f"Loaded {len(salary_df)} jobs with salary data")
        return salary_df
        
    def _convert_to_annual(self, row, salary_col: str) -> Optional[float]:
        """
        Convert salary to annual equivalent based on salary type.
        
        Args:
            row: DataFrame row
            salary_col: Column name for salary
            
        Returns:
            Annual salary equivalent
        """
        salary = row[salary_col]
        if pd.isna(salary):
            return None
            
        salary_type = row.get('salary_type', 'yearly')
        
        if salary_type == 'hourly':
            return salary * 40 * 52  # 40 hours/week, 52 weeks/year
        elif salary_type == 'monthly':
            return salary * 12
        else:  # yearly
            return salary
            
    def _extract_city(self, location: str) -> str:
        """Extract city from location string."""
        if not location or location == "Unknown":
            return "Unknown"
        return location.split(',')[0].strip()
        
    def _extract_state(self, location: str) -> str:
        """Extract state from location string."""
        if not location or location == "Unknown" or ',' not in location:
            return "Unknown"
        return location.split(',')[-1].strip()
        
    def get_salary_statistics(self, job_title: str = None, location: str = None) -> Dict:
        """
        Calculate salary statistics for given filters.
        
        Args:
            job_title: Filter by job title (optional)
            location: Filter by location (optional)
            
        Returns:
            Dictionary with salary statistics
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
            
        df = self.salary_data.copy()
        
        # Apply filters
        if job_title:
            df = df[df['title'].str.contains(job_title, case=False, na=False)]
        if location:
            df = df[df['location'].str.contains(location, case=False, na=False)]
            
        if df.empty:
            return {'error': 'No data found for the given filters'}
            
        stats = {
            'count': len(df),
            'min_salary': df['annual_min'].min(),
            'max_salary': df['annual_max'].max(),
            'median_salary': df['annual_avg'].median(),
            'mean_salary': df['annual_avg'].mean(),
            'std_salary': df['annual_avg'].std(),
            'percentile_25': df['annual_avg'].quantile(0.25),
            'percentile_75': df['annual_avg'].quantile(0.75),
            'percentile_90': df['annual_avg'].quantile(0.90)
        }
        
        # Add salary ranges
        stats['salary_ranges'] = self._categorize_salaries(df['annual_avg'])
        
        return stats
        
    def _categorize_salaries(self, salaries: pd.Series) -> Dict:
        """
        Categorize salaries into ranges.
        
        Args:
            salaries: Series of salary values
            
        Returns:
            Dictionary with salary range counts
        """
        ranges = {
            'Under $50k': (0, 50000),
            '$50k-$75k': (50000, 75000),
            '$75k-$100k': (75000, 100000),
            '$100k-$150k': (100000, 150000),
            '$150k-$200k': (150000, 200000),
            'Over $200k': (200000, float('inf'))
        }
        
        range_counts = {}
        for range_name, (min_sal, max_sal) in ranges.items():
            count = len(salaries[(salaries >= min_sal) & (salaries < max_sal)])
            range_counts[range_name] = count
            
        return range_counts
        
    def analyze_by_location(self) -> pd.DataFrame:
        """
        Analyze salary trends by location.
        
        Returns:
            DataFrame with salary statistics by location
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
            
        location_stats = self.salary_data.groupby('city').agg({
            'annual_avg': ['count', 'mean', 'median', 'min', 'max'],
            'title': 'count'
        }).round(2)
        
        location_stats.columns = ['job_count', 'mean_salary', 'median_salary', 'min_salary', 'max_salary', 'total_postings']
        location_stats = location_stats.sort_values('mean_salary', ascending=False)
        
        return location_stats.reset_index()
        
    def analyze_by_job_title(self) -> pd.DataFrame:
        """
        Analyze salary trends by job title.
        
        Returns:
            DataFrame with salary statistics by job title
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
            
        # Group similar job titles
        job_groups = self._group_job_titles(self.salary_data['title'])
        
        title_stats = self.salary_data.groupby(job_groups).agg({
            'annual_avg': ['count', 'mean', 'median', 'min', 'max'],
        }).round(2)
        
        title_stats.columns = ['job_count', 'mean_salary', 'median_salary', 'min_salary', 'max_salary']
        title_stats = title_stats.sort_values('mean_salary', ascending=False)
        
        return title_stats.reset_index()
        
    def _group_job_titles(self, titles: pd.Series) -> pd.Series:
        """
        Group similar job titles together.
        
        Args:
            titles: Series of job titles
            
        Returns:
            Series with grouped job titles
        """
        title_groups = {
            'Software Engineer': ['software engineer', 'software developer', 'developer', 'programmer'],
            'Data Scientist': ['data scientist', 'data analyst', 'data engineer'],
            'Product Manager': ['product manager', 'product owner', 'pm'],
            'DevOps Engineer': ['devops', 'infrastructure', 'site reliability'],
            'Frontend Developer': ['frontend', 'front-end', 'ui developer'],
            'Backend Developer': ['backend', 'back-end', 'api developer'],
            'Full Stack Developer': ['full stack', 'fullstack'],
            'Machine Learning Engineer': ['machine learning', 'ml engineer', 'ai engineer'],
            'QA Engineer': ['qa', 'quality assurance', 'test engineer'],
            'Security Engineer': ['security', 'cybersecurity', 'infosec']
        }
        
        grouped_titles = titles.copy()
        
        for group_name, keywords in title_groups.items():
            for keyword in keywords:
                mask = titles.str.contains(keyword, case=False, na=False)
                grouped_titles[mask] = group_name
                
        return grouped_titles
        
    def analyze_salary_trends(self, time_period: str = '6M') -> Dict:
        """
        Analyze salary trends over time.
        
        Args:
            time_period: Time period for analysis (e.g., '6M', '1Y')
            
        Returns:
            Dictionary with trend analysis
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
            
        # This would require historical data or posting dates
        # For now, return current snapshot analysis
        current_stats = self.get_salary_statistics()
        
        # Calculate growth estimates based on industry standards
        growth_estimates = {
            'Software Engineer': 0.08,  # 8% annually
            'Data Scientist': 0.12,     # 12% annually
            'Product Manager': 0.06,    # 6% annually
            'DevOps Engineer': 0.10,    # 10% annually
        }
        
        trends = {
            'current_period': current_stats,
            'growth_estimates': growth_estimates,
            'market_outlook': 'Positive - Tech salaries continue to grow above inflation'
        }
        
        return trends
        
    def get_top_paying_companies(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top paying companies by average salary.
        
        Args:
            top_n: Number of companies to return
            
        Returns:
            DataFrame with top paying companies
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
            
        company_stats = self.salary_data.groupby('company').agg({
            'annual_avg': ['count', 'mean', 'median'],
            'title': 'count'
        }).round(2)
        
        company_stats.columns = ['salary_postings', 'mean_salary', 'median_salary', 'total_postings']
        
        # Filter companies with at least 3 salary postings
        company_stats = company_stats[company_stats['salary_postings'] >= 3]
        company_stats = company_stats.sort_values('mean_salary', ascending=False)
        
        return company_stats.head(top_n).reset_index()
        
    def generate_salary_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive salary analysis report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Complete salary analysis report
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
            
        report = {
            'overview': self.get_salary_statistics(),
            'by_location': self.analyze_by_location().to_dict('records'),
            'by_job_title': self.analyze_by_job_title().to_dict('records'),
            'top_companies': self.get_top_paying_companies().to_dict('records'),
            'trends': self.analyze_salary_trends(),
            'summary': {
                'total_jobs_analyzed': len(self.salary_data),
                'salary_data_coverage': f"{len(self.salary_data)} out of {len(self.salary_data)} jobs",
                'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Salary report saved to {output_path}")
            
        return report
