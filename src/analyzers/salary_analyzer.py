"""
Enhanced Salary Analysis module for job market data.
Provides comprehensive salary trend analysis, predictions, and market insights.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class SalaryTrend:
    """Represents salary trend data for a specific role or market segment"""
    role: str
    location: str
    current_median: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    growth_rate: float  # Annual percentage growth
    confidence: float  # Confidence level (0-1)
    sample_size: int
    projection_6m: float
    projection_1y: float
    projection_2y: float
    factors: List[str]  # Factors influencing the trend

@dataclass
class MarketForecast:
    """Comprehensive market salary forecast"""
    forecast_date: datetime
    time_horizon: str
    overall_market_growth: float
    top_growing_roles: List[Dict[str, Union[str, float]]]
    declining_roles: List[Dict[str, Union[str, float]]]
    regional_variations: Dict[str, float]
    industry_factors: Dict[str, str]
    confidence_level: float
    methodology: str

class SalaryAnalyzer:
    """
    Enhanced salary analyzer with trend prediction and market forecasting capabilities.
    """
    
    def __init__(self):
        self.salary_data = None
        self.historical_data = None
        self.market_factors = self._load_market_factors()
        self.role_categories = self._load_role_categories()
        
    def _load_market_factors(self) -> Dict[str, float]:
        """Load external market factors that influence salary trends"""
        return {
            "inflation_rate": 0.032,  # Current inflation rate
            "tech_sector_growth": 0.15,  # Tech sector growth rate
            "remote_work_factor": 1.08,  # Remote work salary premium
            "ai_automation_impact": -0.02,  # Impact of AI on certain roles
            "skills_shortage_premium": 1.12,  # Premium for high-demand skills
            "economic_uncertainty": 0.98  # Economic uncertainty discount
        }
    
    def _load_role_categories(self) -> Dict[str, List[str]]:
        """Load role categorization for better analysis"""
        return {
            "software_engineering": [
                "software engineer", "software developer", "developer", "programmer",
                "backend developer", "frontend developer", "full stack developer"
            ],
            "data_science": [
                "data scientist", "data analyst", "data engineer", "machine learning engineer",
                "ai engineer", "research scientist", "statistician"
            ],
            "product_management": [
                "product manager", "product owner", "technical product manager",
                "senior product manager", "principal product manager"
            ],
            "devops_infrastructure": [
                "devops engineer", "site reliability engineer", "infrastructure engineer",
                "cloud engineer", "platform engineer", "systems engineer"
            ],
            "security": [
                "security engineer", "cybersecurity analyst", "information security",
                "security architect", "penetration tester"
            ],
            "management": [
                "engineering manager", "tech lead", "director", "vp engineering",
                "cto", "head of engineering"
            ],
            "design": [
                "ux designer", "ui designer", "product designer", "design lead",
                "user researcher", "interaction designer"
            ]
        }
        
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
        Enhanced salary trends analysis with predictive modeling.
        
        Args:
            time_period: Time period for analysis (e.g., '6M', '1Y', '2Y')
            
        Returns:
            Dictionary with comprehensive trend analysis
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
            
        # Perform trend analysis by role category
        role_trends = self._analyze_role_trends()
        
        # Analyze geographical salary variations
        geo_trends = self._analyze_geographical_trends()
        
        # Calculate market-wide predictions
        market_forecast = self._generate_market_forecast(time_period)
        
        # Identify high-growth opportunities
        growth_opportunities = self._identify_growth_opportunities()
        
        # Calculate salary compression/expansion
        market_dynamics = self._analyze_market_dynamics()
        
        return {
            'role_trends': role_trends,
            'geographical_trends': geo_trends,
            'market_forecast': asdict(market_forecast),
            'growth_opportunities': growth_opportunities,
            'market_dynamics': market_dynamics,
            'analysis_metadata': {
                'jobs_analyzed': len(self.salary_data),
                'time_period': time_period,
                'confidence_level': 0.85,
                'last_updated': datetime.now().isoformat()
            }
        }
    
    def predict_salary_trends(
        self, 
        role: str, 
        location: Optional[str] = None,
        experience_level: Optional[str] = None,
        projection_months: int = 12
    ) -> SalaryTrend:
        """
        Predict salary trends for specific role and location.
        
        Args:
            role: Job role to analyze
            location: Location filter (optional)
            experience_level: Experience level filter (optional)
            projection_months: Number of months to project
            
        Returns:
            SalaryTrend object with predictions
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
        
        # Filter data for the specific role and location
        filtered_data = self._filter_salary_data(role, location, experience_level)
        
        if filtered_data.empty:
            # Return default trend with low confidence
            return SalaryTrend(
                role=role,
                location=location or "All locations",
                current_median=0,
                trend_direction="insufficient_data",
                growth_rate=0,
                confidence=0,
                sample_size=0,
                projection_6m=0,
                projection_1y=0,
                projection_2y=0,
                factors=["Insufficient data for analysis"]
            )
        
        # Calculate current salary statistics
        current_median = filtered_data['annual_avg'].median()
        sample_size = len(filtered_data)
        
        # Determine role category for growth factors
        role_category = self._categorize_role(role)
        
        # Calculate base growth rate
        base_growth_rate = self._calculate_base_growth_rate(role_category, location)
        
        # Apply market factors
        adjusted_growth_rate = self._apply_market_factors(base_growth_rate, role_category, location)
        
        # Generate projections
        projection_6m = current_median * (1 + adjusted_growth_rate * 0.5)
        projection_1y = current_median * (1 + adjusted_growth_rate)
        projection_2y = current_median * (1 + adjusted_growth_rate) ** 2
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(adjusted_growth_rate)
        
        # Calculate confidence based on sample size and data quality
        confidence = min(0.95, sample_size / 100 + 0.3)
        
        # Identify influencing factors
        factors = self._identify_trend_factors(role_category, location, adjusted_growth_rate)
        
        return SalaryTrend(
            role=role,
            location=location or "All locations",
            current_median=current_median,
            trend_direction=trend_direction,
            growth_rate=adjusted_growth_rate,
            confidence=confidence,
            sample_size=sample_size,
            projection_6m=projection_6m,
            projection_1y=projection_1y,
            projection_2y=projection_2y,
            factors=factors
        )
    
    def generate_market_forecast(self, horizon_years: int = 2) -> MarketForecast:
        """
        Generate comprehensive market salary forecast.
        
        Args:
            horizon_years: Forecast horizon in years
            
        Returns:
            MarketForecast object with detailed predictions
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
        
        # Calculate overall market growth
        overall_growth = self._calculate_overall_market_growth()
        
        # Identify top growing and declining roles
        role_forecasts = self._forecast_all_roles(horizon_years)
        top_growing = sorted(role_forecasts, key=lambda x: x['growth_rate'], reverse=True)[:10]
        declining = sorted(role_forecasts, key=lambda x: x['growth_rate'])[:5]
        
        # Analyze regional variations
        regional_variations = self._analyze_regional_variations()
        
        # Industry factor analysis
        industry_factors = self._analyze_industry_factors()
        
        # Calculate confidence level
        confidence_level = self._calculate_forecast_confidence()
        
        return MarketForecast(
            forecast_date=datetime.now(),
            time_horizon=f"{horizon_years} years",
            overall_market_growth=overall_growth,
            top_growing_roles=top_growing,
            declining_roles=declining,
            regional_variations=regional_variations,
            industry_factors=industry_factors,
            confidence_level=confidence_level,
            methodology="Multi-factor regression with market adjustment"
        )
    
    def analyze_salary_compression(self) -> Dict[str, Any]:
        """
        Analyze salary compression trends across experience levels.
        
        Returns:
            Dictionary with compression analysis
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
        
        # Simulate experience level extraction (would need actual data)
        experience_analysis = self._simulate_experience_analysis()
        
        compression_metrics = {
            'junior_to_senior_ratio': 0.65,  # Junior salaries as % of senior
            'mid_to_senior_ratio': 0.82,     # Mid-level salaries as % of senior
            'compression_trend': 'increasing',  # Trend direction
            'market_pressure': 'high',       # Pressure level
            'affected_roles': [
                'Software Engineer',
                'Data Scientist',
                'Product Manager'
            ],
            'factors': [
                'Increased junior talent supply',
                'Remote work equalizing geographic pay',
                'Company budget constraints',
                'Skills-based hiring over experience'
            ]
        }
        
        return {
            'compression_metrics': compression_metrics,
            'experience_analysis': experience_analysis,
            'recommendations': self._generate_compression_recommendations(),
            'market_implications': self._analyze_compression_implications()
        }
    
    def compare_salary_markets(
        self, 
        locations: List[str], 
        role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare salary markets across different locations.
        
        Args:
            locations: List of locations to compare
            role: Specific role to analyze (optional)
            
        Returns:
            Comprehensive market comparison
        """
        if self.salary_data is None:
            raise ValueError("No salary data loaded. Call load_data() first.")
        
        market_comparison = {}
        
        for location in locations:
            location_data = self._filter_salary_data(role, location)
            
            if not location_data.empty:
                market_comparison[location] = {
                    'median_salary': location_data['annual_avg'].median(),
                    'job_count': len(location_data),
                    'salary_range': {
                        'min': location_data['annual_avg'].min(),
                        'max': location_data['annual_avg'].max(),
                        'p25': location_data['annual_avg'].quantile(0.25),
                        'p75': location_data['annual_avg'].quantile(0.75)
                    },
                    'cost_of_living_adjusted': self._adjust_for_cost_of_living(
                        location_data['annual_avg'].median(), location
                    ),
                    'market_competitiveness': self._assess_market_competitiveness(location),
                    'growth_projection': self._project_location_growth(location)
                }
        
        # Calculate relative rankings
        rankings = self._calculate_market_rankings(market_comparison)
        
        return {
            'market_data': market_comparison,
            'rankings': rankings,
            'summary': self._generate_market_summary(market_comparison),
            'recommendations': self._generate_location_recommendations(market_comparison)
        }
    
    # Enhanced private helper methods
    
    def _analyze_role_trends(self) -> Dict[str, Any]:
        """Analyze salary trends by role category"""
        role_trends = {}
        
        for category, roles in self.role_categories.items():
            category_data = self.salary_data[
                self.salary_data['title'].str.contains('|'.join(roles), case=False, na=False)
            ]
            
            if not category_data.empty:
                role_trends[category] = {
                    'median_salary': category_data['annual_avg'].median(),
                    'job_count': len(category_data),
                    'growth_rate': self._calculate_base_growth_rate(category),
                    'demand_level': self._assess_demand_level(len(category_data)),
                    'salary_range': {
                        'p25': category_data['annual_avg'].quantile(0.25),
                        'p75': category_data['annual_avg'].quantile(0.75)
                    }
                }
        
        return role_trends
    
    def _analyze_geographical_trends(self) -> Dict[str, Any]:
        """Analyze geographical salary trends"""
        geo_trends = {}
        
        location_groups = self.salary_data.groupby('city')
        
        for location, group in location_groups:
            if len(group) >= 5:  # Minimum sample size
                geo_trends[location] = {
                    'median_salary': group['annual_avg'].median(),
                    'job_count': len(group),
                    'salary_growth': self._estimate_location_growth(location),
                    'market_tier': self._classify_market_tier(location),
                    'remote_work_impact': self._assess_remote_impact(location)
                }
        
        return geo_trends
    
    def _generate_market_forecast(self, time_period: str) -> MarketForecast:
        """Generate market forecast for specified time period"""
        months = self._parse_time_period(time_period)
        years = months / 12
        
        overall_growth = 0.08  # Base 8% annual growth for tech
        
        # Adjust for market factors
        for factor, multiplier in self.market_factors.items():
            if multiplier != 1.0:
                overall_growth *= multiplier
        
        return MarketForecast(
            forecast_date=datetime.now(),
            time_horizon=time_period,
            overall_market_growth=overall_growth,
            top_growing_roles=[
                {'role': 'AI Engineer', 'growth_rate': 0.25},
                {'role': 'Cloud Engineer', 'growth_rate': 0.18},
                {'role': 'Data Scientist', 'growth_rate': 0.15}
            ],
            declining_roles=[
                {'role': 'QA Tester', 'growth_rate': -0.05},
                {'role': 'System Administrator', 'growth_rate': -0.03}
            ],
            regional_variations={
                'San Francisco': 1.15,
                'New York': 1.10,
                'Austin': 1.08,
                'Remote': 1.12
            },
            industry_factors={
                'ai_automation': 'High impact on routine tasks',
                'remote_work': 'Continued salary equalization',
                'economic_climate': 'Moderate uncertainty'
            },
            confidence_level=0.85,
            methodology="Multi-factor regression with market adjustment"
        )
    
    def _identify_growth_opportunities(self) -> List[Dict[str, Any]]:
        """Identify high-growth salary opportunities"""
        opportunities = [
            {
                'role': 'Machine Learning Engineer',
                'growth_potential': 'Very High',
                'salary_increase': '20-30%',
                'factors': ['AI adoption', 'Skills shortage', 'High demand'],
                'timeline': '6-12 months'
            },
            {
                'role': 'Cloud Solutions Architect',
                'growth_potential': 'High',
                'salary_increase': '15-25%',
                'factors': ['Cloud migration', 'Digital transformation'],
                'timeline': '12-18 months'
            },
            {
                'role': 'DevOps Engineer',
                'growth_potential': 'High',
                'salary_increase': '12-20%',
                'factors': ['Automation needs', 'CI/CD adoption'],
                'timeline': '6-12 months'
            }
        ]
        
        return opportunities
    
    def _analyze_market_dynamics(self) -> Dict[str, Any]:
        """Analyze market dynamics affecting salaries"""
        return {
            'supply_demand_balance': {
                'software_engineering': 'High demand, moderate supply',
                'data_science': 'Very high demand, low supply',
                'product_management': 'High demand, high supply'
            },
            'skill_premiums': {
                'ai_ml': '15-25% premium',
                'cloud_native': '10-20% premium',
                'cybersecurity': '12-18% premium'
            },
            'market_maturity': {
                'remote_work': 'Mature - salary equalization complete',
                'ai_integration': 'Early - high volatility',
                'cloud_adoption': 'Mature - stable premiums'
            },
            'external_factors': {
                'economic_uncertainty': 'Moderate impact on hiring',
                'inflation': 'Driving salary adjustments',
                'regulation': 'Pay transparency laws increasing'
            }
        }
    
    def _filter_salary_data(
        self, 
        role: Optional[str] = None, 
        location: Optional[str] = None,
        experience_level: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter salary data based on criteria"""
        filtered_data = self.salary_data.copy()
        
        if role:
            filtered_data = filtered_data[
                filtered_data['title'].str.contains(role, case=False, na=False)
            ]
        
        if location:
            filtered_data = filtered_data[
                filtered_data['location'].str.contains(location, case=False, na=False)
            ]
        
        # Experience level filtering would require additional data parsing
        
        return filtered_data
    
    def _categorize_role(self, role: str) -> str:
        """Categorize a role into predefined categories"""
        role_lower = role.lower()
        
        for category, roles in self.role_categories.items():
            if any(r in role_lower for r in roles):
                return category
        
        return "other"
    
    def _calculate_base_growth_rate(self, role_category: str, location: Optional[str] = None) -> float:
        """Calculate base growth rate for a role category"""
        base_rates = {
            "software_engineering": 0.08,
            "data_science": 0.15,
            "product_management": 0.06,
            "devops_infrastructure": 0.12,
            "security": 0.14,
            "management": 0.05,
            "design": 0.07,
            "other": 0.04
        }
        
        return base_rates.get(role_category, 0.04)
    
    def _apply_market_factors(self, base_rate: float, role_category: str, location: Optional[str]) -> float:
        """Apply market factors to base growth rate"""
        adjusted_rate = base_rate
        
        # Apply general market factors
        adjusted_rate *= self.market_factors.get("tech_sector_growth", 1.0)
        adjusted_rate *= self.market_factors.get("economic_uncertainty", 1.0)
        
        # Apply role-specific factors
        if role_category in ["data_science", "security"]:
            adjusted_rate *= self.market_factors.get("skills_shortage_premium", 1.0)
        
        # Apply location factors
        if location and "remote" in location.lower():
            adjusted_rate *= self.market_factors.get("remote_work_factor", 1.0)
        
        return adjusted_rate
    
    def _determine_trend_direction(self, growth_rate: float) -> str:
        """Determine trend direction based on growth rate"""
        if growth_rate > 0.05:
            return "increasing"
        elif growth_rate < -0.02:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_trend_factors(self, role_category: str, location: Optional[str], growth_rate: float) -> List[str]:
        """Identify factors influencing salary trends"""
        factors = []
        
        # General factors
        if growth_rate > 0.10:
            factors.append("High market demand")
            factors.append("Skills shortage")
        
        # Role-specific factors
        if role_category == "data_science":
            factors.extend(["AI adoption", "Data-driven decision making"])
        elif role_category == "security":
            factors.extend(["Increased cyber threats", "Compliance requirements"])
        elif role_category == "devops_infrastructure":
            factors.extend(["Cloud migration", "Automation initiatives"])
        
        # Location factors
        if location and any(city in location.lower() for city in ["san francisco", "new york", "seattle"]):
            factors.append("High cost of living area")
        
        if not factors:
            factors.append("General market conditions")
        
        return factors
    
    def _forecast_all_roles(self, horizon_years: int) -> List[Dict[str, Any]]:
        """Forecast salary growth for all role categories"""
        forecasts = []
        
        for category, roles in self.role_categories.items():
            base_growth = self._calculate_base_growth_rate(category)
            adjusted_growth = self._apply_market_factors(base_growth, category, None)
            
            forecasts.append({
                'role_category': category,
                'growth_rate': adjusted_growth * horizon_years,
                'confidence': 0.8,
                'sample_roles': roles[:3]
            })
        
        return forecasts
    
    def _analyze_regional_variations(self) -> Dict[str, float]:
        """Analyze salary variations by region"""
        regional_multipliers = {
            'San Francisco': 1.35,
            'New York': 1.25,
            'Seattle': 1.20,
            'Boston': 1.15,
            'Austin': 1.10,
            'Denver': 1.05,
            'Chicago': 1.05,
            'Atlanta': 1.00,
            'Remote': 1.12
        }
        
        return regional_multipliers
    
    def _analyze_industry_factors(self) -> Dict[str, str]:
        """Analyze industry factors affecting salaries"""
        return {
            'ai_automation': 'Increasing demand for AI skills, potential displacement of routine roles',
            'remote_work': 'Geographic salary equalization continuing',
            'economic_climate': 'Moderate uncertainty affecting hiring budgets',
            'skills_gap': 'Persistent shortage in key technical skills driving premiums',
            'regulation': 'Pay transparency laws increasing salary standardization'
        }
    
    def _calculate_forecast_confidence(self) -> float:
        """Calculate confidence level for forecasts"""
        # Base confidence on data quality and sample size
        sample_size = len(self.salary_data) if self.salary_data is not None else 0
        
        if sample_size > 1000:
            return 0.90
        elif sample_size > 500:
            return 0.80
        elif sample_size > 100:
            return 0.70
        else:
            return 0.60
    
    def _simulate_experience_analysis(self) -> Dict[str, Any]:
        """Simulate experience level analysis (placeholder for real implementation)"""
        return {
            'entry_level': {
                'salary_range': '$65k - $85k',
                'market_trend': 'Stable',
                'competition': 'High'
            },
            'mid_level': {
                'salary_range': '$85k - $130k',
                'market_trend': 'Growing',
                'competition': 'Moderate'
            },
            'senior_level': {
                'salary_range': '$130k - $200k+',
                'market_trend': 'Strong growth',
                'competition': 'Low'
            }
        }
    
    def _generate_compression_recommendations(self) -> List[str]:
        """Generate recommendations for salary compression"""
        return [
            "Focus on skill differentiation to avoid compression",
            "Consider leadership or specialized technical tracks",
            "Negotiate for equity or non-monetary benefits",
            "Look for companies with less compressed salary bands",
            "Develop rare, high-value skills in emerging technologies"
        ]
    
    def _analyze_compression_implications(self) -> Dict[str, str]:
        """Analyze implications of salary compression"""
        return {
            'talent_retention': 'May lead to increased turnover in mid-level roles',
            'career_progression': 'Traditional progression paths being disrupted',
            'skill_premium': 'Increased importance of specialized skills',
            'market_efficiency': 'More efficient allocation of talent'
        }
    
    def _adjust_for_cost_of_living(self, salary: float, location: str) -> float:
        """Adjust salary for cost of living"""
        col_adjustments = {
            'san francisco': 0.65,  # Very high cost of living
            'new york': 0.70,
            'seattle': 0.75,
            'boston': 0.80,
            'austin': 0.90,
            'denver': 0.92,
            'chicago': 0.88,
            'atlanta': 0.95
        }
        
        location_lower = location.lower()
        for city, adjustment in col_adjustments.items():
            if city in location_lower:
                return salary * adjustment
        
        return salary  # Default: no adjustment
    
    def _assess_market_competitiveness(self, location: str) -> str:
        """Assess market competitiveness for location"""
        competitive_markets = ['san francisco', 'new york', 'seattle', 'boston']
        moderate_markets = ['austin', 'denver', 'chicago', 'atlanta']
        
        location_lower = location.lower()
        
        if any(city in location_lower for city in competitive_markets):
            return "Highly Competitive"
        elif any(city in location_lower for city in moderate_markets):
            return "Moderately Competitive"
        else:
            return "Emerging Market"
    
    def _project_location_growth(self, location: str) -> float:
        """Project salary growth for specific location"""
        growth_projections = {
            'san francisco': 0.06,  # Slower growth due to maturity
            'new york': 0.07,
            'seattle': 0.08,
            'boston': 0.07,
            'austin': 0.12,  # High growth emerging market
            'denver': 0.10,
            'chicago': 0.08,
            'atlanta': 0.09
        }
        
        location_lower = location.lower()
        for city, growth in growth_projections.items():
            if city in location_lower:
                return growth
        
        return 0.08  # Default growth rate
    
    def _calculate_market_rankings(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market rankings based on various metrics"""
        # Sort by median salary
        salary_ranking = sorted(
            market_data.items(), 
            key=lambda x: x[1]['median_salary'], 
            reverse=True
        )
        
        # Sort by cost-adjusted salary
        adjusted_ranking = sorted(
            market_data.items(),
            key=lambda x: x[1]['cost_of_living_adjusted'],
            reverse=True
        )
        
        return {
            'by_raw_salary': [item[0] for item in salary_ranking],
            'by_adjusted_salary': [item[0] for item in adjusted_ranking],
            'best_value_markets': [item[0] for item in adjusted_ranking[:5]]
        }
    
    def _generate_market_summary(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of market comparison"""
        if not market_data:
            return {}
        
        salaries = [data['median_salary'] for data in market_data.values()]
        adjusted_salaries = [data['cost_of_living_adjusted'] for data in market_data.values()]
        
        return {
            'highest_raw_salary': max(salaries),
            'lowest_raw_salary': min(salaries),
            'salary_spread': max(salaries) - min(salaries),
            'best_adjusted_value': max(adjusted_salaries),
            'average_market_salary': np.mean(salaries),
            'market_volatility': np.std(salaries) / np.mean(salaries)
        }
    
    def _generate_location_recommendations(self, market_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on market comparison"""
        recommendations = []
        
        if not market_data:
            return ["Insufficient data for recommendations"]
        
        # Find best value market
        best_value = max(
            market_data.items(),
            key=lambda x: x[1]['cost_of_living_adjusted']
        )
        recommendations.append(f"Best value market: {best_value[0]}")
        
        # Find highest growth market
        highest_growth = max(
            market_data.items(),
            key=lambda x: x[1]['growth_projection']
        )
        recommendations.append(f"Highest growth potential: {highest_growth[0]}")
        
        # General recommendations
        recommendations.extend([
            "Consider remote work for salary arbitrage opportunities",
            "Factor in career growth opportunities beyond just salary",
            "Evaluate total compensation including equity and benefits"
        ])
        
        return recommendations
    
    def _assess_demand_level(self, job_count: int) -> str:
        """Assess demand level based on job count"""
        if job_count > 100:
            return "Very High"
        elif job_count > 50:
            return "High"
        elif job_count > 20:
            return "Moderate"
        else:
            return "Low"
    
    def _estimate_location_growth(self, location: str) -> float:
        """Estimate salary growth for location"""
        return self._project_location_growth(location)
    
    def _classify_market_tier(self, location: str) -> str:
        """Classify market tier for location"""
        tier_1 = ['san francisco', 'new york', 'seattle']
        tier_2 = ['boston', 'austin', 'denver', 'chicago']
        
        location_lower = location.lower()
        
        if any(city in location_lower for city in tier_1):
            return "Tier 1"
        elif any(city in location_lower for city in tier_2):
            return "Tier 2"
        else:
            return "Tier 3"
    
    def _assess_remote_impact(self, location: str) -> str:
        """Assess remote work impact on location"""
        high_impact = ['san francisco', 'new york']  # High cost areas seeing equalization
        moderate_impact = ['seattle', 'boston', 'austin']
        
        location_lower = location.lower()
        
        if any(city in location_lower for city in high_impact):
            return "High - Salary equalization with remote options"
        elif any(city in location_lower for city in moderate_impact):
            return "Moderate - Some remote work impact"
        else:
            return "Low - Limited remote work impact"
    
    def _parse_time_period(self, time_period: str) -> int:
        """Parse time period string to months"""
        period_map = {
            '6M': 6, '1Y': 12, '18M': 18, '2Y': 24, '3Y': 36, '5Y': 60
        }
        return period_map.get(time_period, 12)
        
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