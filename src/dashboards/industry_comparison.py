"""
Industry Comparison Dashboard - Interactive visualization for comparing job markets across industries
Provides comprehensive analysis of salary, growth, and opportunity trends by industry
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import json

@dataclass
class IndustryMetrics:
    """Industry performance metrics"""
    name: str
    avg_salary: float
    job_count: int
    growth_rate: float
    remote_friendly: float  # Percentage of remote jobs
    skill_diversity: float  # Number of unique skills required
    experience_demand: Dict[str, int]  # Entry, Mid, Senior level demand
    top_skills: List[str]
    top_companies: List[str]
    geographic_spread: Dict[str, int]  # Location distribution
    market_trend: str  # "growing", "stable", "declining"
    competitiveness: str  # "low", "medium", "high"

class IndustryComparisonDashboard:
    """Interactive dashboard for industry comparison analysis"""
    
    def __init__(self):
        self.industries = self._define_industries()
        self.colors = self._define_color_palette()
        self.cached_data = {}
        
    def _define_industries(self) -> Dict[str, List[str]]:
        """Define industry categories and their keywords"""
        return {
            "Technology": [
                "software", "tech", "engineering", "developer", "programmer",
                "data scientist", "ai", "machine learning", "cloud", "devops"
            ],
            "Finance & Banking": [
                "finance", "banking", "investment", "fintech", "trading",
                "analyst", "quant", "risk management", "compliance"
            ],
            "Healthcare": [
                "healthcare", "medical", "pharmaceutical", "biotech",
                "clinical", "health informatics", "medical device"
            ],
            "Consulting": [
                "consulting", "consultant", "advisory", "strategy",
                "management consulting", "business analyst"
            ],
            "E-commerce & Retail": [
                "ecommerce", "retail", "marketplace", "consumer",
                "merchandising", "supply chain", "logistics"
            ],
            "Media & Entertainment": [
                "media", "entertainment", "gaming", "streaming",
                "content", "marketing", "advertising", "creative"
            ],
            "Education": [
                "education", "edtech", "training", "academic",
                "instructional design", "curriculum"
            ],
            "Manufacturing": [
                "manufacturing", "industrial", "automotive", "aerospace",
                "operations", "quality assurance", "supply chain"
            ],
            "Energy & Utilities": [
                "energy", "utilities", "renewable", "oil", "gas",
                "power", "sustainability", "environmental"
            ],
            "Real Estate": [
                "real estate", "property", "construction", "architecture",
                "urban planning", "facilities"
            ]
        }
    
    def _define_color_palette(self) -> Dict[str, str]:
        """Define color palette for visualizations"""
        return {
            "Technology": "#1f77b4",
            "Finance & Banking": "#ff7f0e", 
            "Healthcare": "#2ca02c",
            "Consulting": "#d62728",
            "E-commerce & Retail": "#9467bd",
            "Media & Entertainment": "#8c564b",
            "Education": "#e377c2",
            "Manufacturing": "#7f7f7f",
            "Energy & Utilities": "#bcbd22",
            "Real Estate": "#17becf"
        }
    
    def render_dashboard(self, job_data: pd.DataFrame):
        """Render the complete industry comparison dashboard"""
        
        st.title("🏭 Industry Comparison Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Dashboard Controls")
            
            # Industry selection
            selected_industries = st.multiselect(
                "Select Industries to Compare",
                options=list(self.industries.keys()),
                default=["Technology", "Finance & Banking", "Healthcare", "Consulting"]
            )
            
            # Metric selection
            primary_metric = st.selectbox(
                "Primary Comparison Metric",
                ["Average Salary", "Job Count", "Growth Rate", "Remote Friendliness"]
            )
            
            # Time period
            time_period = st.selectbox(
                "Analysis Time Period",
                ["Last 6 months", "Last 1 year", "Last 2 years", "All time"]
            )
            
            # Location filter
            locations = self._extract_unique_locations(job_data)
            location_filter = st.multiselect(
                "Filter by Location",
                options=["All"] + locations,
                default=["All"]
            )
        
        if not selected_industries:
            st.warning("Please select at least one industry to analyze.")
            return
        
        # Process data
        with st.spinner("Analyzing industry data..."):
            industry_metrics = self._analyze_industries(job_data, selected_industries, location_filter)
        
        # Main dashboard sections
        self._render_overview_section(industry_metrics, primary_metric)
        self._render_salary_analysis(industry_metrics)
        self._render_job_market_trends(industry_metrics)
        self._render_skills_analysis(industry_metrics, job_data)
        self._render_geographic_distribution(industry_metrics)
        self._render_company_analysis(industry_metrics)
        self._render_detailed_comparison(industry_metrics)
        self._render_insights_and_recommendations(industry_metrics)
    
    def _render_overview_section(self, industry_metrics: Dict[str, IndustryMetrics], primary_metric: str):
        """Render overview section with key metrics"""
        
        st.header("📊 Industry Overview")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        total_jobs = sum(metrics.job_count for metrics in industry_metrics.values())
        avg_salary_all = np.mean([metrics.avg_salary for metrics in industry_metrics.values()])
        avg_growth_rate = np.mean([metrics.growth_rate for metrics in industry_metrics.values()])
        avg_remote_friendly = np.mean([metrics.remote_friendly for metrics in industry_metrics.values()])
        
        with col1:
            st.metric("Total Jobs Analyzed", f"{total_jobs:,}")
        
        with col2:
            st.metric("Average Salary", f"${avg_salary_all:,.0f}")
        
        with col3:
            st.metric("Average Growth Rate", f"{avg_growth_rate:.1%}")
        
        with col4:
            st.metric("Remote Work Availability", f"{avg_remote_friendly:.1%}")
        
        # Primary metric comparison chart
        st.subheader(f"Industry Comparison by {primary_metric}")
        
        metric_mapping = {
            "Average Salary": "avg_salary",
            "Job Count": "job_count", 
            "Growth Rate": "growth_rate",
            "Remote Friendliness": "remote_friendly"
        }
        
        metric_key = metric_mapping[primary_metric]
        
        # Create comparison chart
        industries = list(industry_metrics.keys())
        values = [getattr(industry_metrics[ind], metric_key) for ind in industries]
        colors = [self.colors.get(ind, "#888888") for ind in industries]
        
        fig = px.bar(
            x=industries,
            y=values,
            title=f"{primary_metric} by Industry",
            color=industries,
            color_discrete_map=self.colors
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Industry",
            yaxis_title=primary_metric,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_salary_analysis(self, industry_metrics: Dict[str, IndustryMetrics]):
        """Render detailed salary analysis section"""
        
        st.header("💰 Salary Analysis")
        
        # Salary comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of average salaries
            industries = list(industry_metrics.keys())
            salaries = [industry_metrics[ind].avg_salary for ind in industries]
            
            fig = px.bar(
                x=salaries,
                y=industries,
                orientation='h',
                title="Average Salary by Industry",
                color=industries,
                color_discrete_map=self.colors
            )
            
            fig.update_layout(
                showlegend=False,
                xaxis_title="Average Salary ($)",
                yaxis_title="Industry",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Salary vs job count scatter plot
            job_counts = [industry_metrics[ind].job_count for ind in industries]
            
            fig = px.scatter(
                x=job_counts,
                y=salaries,
                color=industries,
                size=[industry_metrics[ind].growth_rate * 100 + 100 for ind in industries],
                title="Salary vs Job Availability",
                labels={"x": "Number of Jobs", "y": "Average Salary ($)"},
                color_discrete_map=self.colors,
                hover_data={"color": False}
            )
            
            for i, industry in enumerate(industries):
                fig.add_annotation(
                    x=job_counts[i],
                    y=salaries[i],
                    text=industry.split()[0],  # First word of industry
                    showarrow=False,
                    font=dict(size=10)
                )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Salary distribution table
        st.subheader("Salary Summary Statistics")
        
        salary_data = []
        for industry, metrics in industry_metrics.items():
            salary_data.append({
                "Industry": industry,
                "Average Salary": f"${metrics.avg_salary:,.0f}",
                "Job Count": f"{metrics.job_count:,}",
                "Market Trend": metrics.market_trend.title(),
                "Competitiveness": metrics.competitiveness.title()
            })
        
        st.dataframe(pd.DataFrame(salary_data), use_container_width=True)
    
    def _render_job_market_trends(self, industry_metrics: Dict[str, IndustryMetrics]):
        """Render job market trends analysis"""
        
        st.header("📈 Job Market Trends")
        
        # Growth rate vs remote work correlation
        col1, col2 = st.columns(2)
        
        with col1:
            industries = list(industry_metrics.keys())
            growth_rates = [industry_metrics[ind].growth_rate * 100 for ind in industries]
            
            fig = px.bar(
                x=industries,
                y=growth_rates,
                title="Industry Growth Rates",
                color=growth_rates,
                color_continuous_scale="RdYlGn"
            )
            
            fig.update_layout(
                xaxis_title="Industry",
                yaxis_title="Growth Rate (%)",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            remote_percentages = [industry_metrics[ind].remote_friendly * 100 for ind in industries]
            
            fig = px.bar(
                x=industries,
                y=remote_percentages,
                title="Remote Work Friendliness",
                color=remote_percentages,
                color_continuous_scale="Blues"
            )
            
            fig.update_layout(
                xaxis_title="Industry",
                yaxis_title="Remote Jobs (%)",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market competitiveness analysis
        st.subheader("Market Competitiveness Analysis")
        
        competitiveness_data = {}
        for industry, metrics in industry_metrics.items():
            competitiveness_data[industry] = {
                "Growth Rate": metrics.growth_rate,
                "Job Count": metrics.job_count,
                "Avg Salary": metrics.avg_salary,
                "Remote Friendly": metrics.remote_friendly,
                "Competitiveness": metrics.competitiveness
            }
        
        # Create heatmap
        df_comp = pd.DataFrame(competitiveness_data).T
        numeric_cols = ["Growth Rate", "Job Count", "Avg Salary", "Remote Friendly"]
        
        # Normalize data for heatmap
        df_normalized = df_comp[numeric_cols].copy()
        for col in numeric_cols:
            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())
        
        fig = px.imshow(
            df_normalized.values,
            x=numeric_cols,
            y=df_normalized.index,
            color_continuous_scale="RdYlGn",
            title="Industry Competitiveness Heatmap (Normalized)"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_skills_analysis(self, industry_metrics: Dict[str, IndustryMetrics], job_data: pd.DataFrame):
        """Render skills analysis section"""
        
        st.header("🛠️ Skills Analysis")
        
        # Top skills by industry
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Skills by Industry")
            
            selected_industry = st.selectbox(
                "Select Industry for Skills Analysis",
                list(industry_metrics.keys())
            )
            
            if selected_industry in industry_metrics:
                skills = industry_metrics[selected_industry].top_skills
                
                # Create skills chart
                fig = px.bar(
                    x=list(range(len(skills))),
                    y=skills,
                    orientation='h',
                    title=f"Top Skills in {selected_industry}",
                    color=list(range(len(skills))),
                    color_continuous_scale="viridis"
                )
                
                fig.update_layout(
                    xaxis_title="Skill Frequency",
                    yaxis_title="Skills",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Skill Diversity Index")
            
            industries = list(industry_metrics.keys())
            diversity_scores = [industry_metrics[ind].skill_diversity for ind in industries]
            
            fig = px.bar(
                x=industries,
                y=diversity_scores,
                title="Skill Diversity by Industry",
                color=diversity_scores,
                color_continuous_scale="plasma"
            )
            
            fig.update_layout(
                xaxis_title="Industry",
                yaxis_title="Skill Diversity Score",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills overlap analysis
        st.subheader("Skills Transferability Analysis")
        
        # Create skills overlap matrix
        skill_overlap = self._calculate_skill_overlap(industry_metrics)
        
        fig = px.imshow(
            skill_overlap,
            x=list(industry_metrics.keys()),
            y=list(industry_metrics.keys()),
            color_continuous_scale="Blues",
            title="Skills Overlap Between Industries"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_geographic_distribution(self, industry_metrics: Dict[str, IndustryMetrics]):
        """Render geographic distribution analysis"""
        
        st.header("🌍 Geographic Distribution")
        
        # Select industry for geographic analysis
        selected_industry = st.selectbox(
            "Select Industry for Geographic Analysis",
            list(industry_metrics.keys()),
            key="geo_industry"
        )
        
        if selected_industry in industry_metrics:
            geo_data = industry_metrics[selected_industry].geographic_spread
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of geographic distribution
                locations = list(geo_data.keys())
                job_counts = list(geo_data.values())
                
                fig = px.pie(
                    values=job_counts,
                    names=locations,
                    title=f"Job Distribution in {selected_industry}"
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart for better comparison
                fig = px.bar(
                    x=locations,
                    y=job_counts,
                    title=f"Jobs by Location - {selected_industry}",
                    color=job_counts,
                    color_continuous_scale="viridis"
                )
                
                fig.update_layout(
                    xaxis_title="Location",
                    yaxis_title="Number of Jobs",
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Multi-industry geographic comparison
        st.subheader("Multi-Industry Geographic Comparison")
        
        geo_comparison = {}
        for industry, metrics in industry_metrics.items():
            for location, count in metrics.geographic_spread.items():
                if location not in geo_comparison:
                    geo_comparison[location] = {}
                geo_comparison[location][industry] = count
        
        geo_df = pd.DataFrame(geo_comparison).fillna(0).T
        
        # Create stacked bar chart
        fig = px.bar(
            geo_df,
            title="Job Distribution Across Industries by Location",
            barmode="stack"
        )
        
        fig.update_layout(
            xaxis_title="Location",
            yaxis_title="Number of Jobs",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_company_analysis(self, industry_metrics: Dict[str, IndustryMetrics]):
        """Render company analysis section"""
        
        st.header("🏢 Company Analysis")
        
        # Top companies by industry
        st.subheader("Top Companies by Industry")
        
        company_tabs = st.tabs(list(industry_metrics.keys()))
        
        for i, (industry, metrics) in enumerate(industry_metrics.items()):
            with company_tabs[i]:
                companies = metrics.top_companies[:10]  # Top 10 companies
                
                if companies:
                    # Create company rankings
                    company_df = pd.DataFrame({
                        "Rank": list(range(1, len(companies) + 1)),
                        "Company": companies,
                        "Industry": [industry] * len(companies)
                    })
                    
                    st.dataframe(company_df, use_container_width=True)
                    
                    # Company distribution chart
                    fig = px.bar(
                        x=companies,
                        y=list(range(len(companies), 0, -1)),
                        orientation='h',
                        title=f"Top Companies in {industry}",
                        color=list(range(len(companies))),
                        color_continuous_scale="viridis"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Relative Job Postings",
                        yaxis_title="Companies",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No company data available for this industry.")
    
    def _render_detailed_comparison(self, industry_metrics: Dict[str, IndustryMetrics]):
        """Render detailed side-by-side comparison"""
        
        st.header("🔍 Detailed Comparison")
        
        # Industry selector for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            industry1 = st.selectbox(
                "Select First Industry",
                list(industry_metrics.keys()),
                key="compare1"
            )
        
        with col2:
            industry2 = st.selectbox(
                "Select Second Industry", 
                list(industry_metrics.keys()),
                index=1 if len(industry_metrics) > 1 else 0,
                key="compare2"
            )
        
        if industry1 and industry2 and industry1 != industry2:
            metrics1 = industry_metrics[industry1]
            metrics2 = industry_metrics[industry2]
            
            # Comparison table
            comparison_data = {
                "Metric": [
                    "Average Salary",
                    "Job Count", 
                    "Growth Rate",
                    "Remote Friendliness",
                    "Skill Diversity",
                    "Market Trend",
                    "Competitiveness"
                ],
                industry1: [
                    f"${metrics1.avg_salary:,.0f}",
                    f"{metrics1.job_count:,}",
                    f"{metrics1.growth_rate:.1%}",
                    f"{metrics1.remote_friendly:.1%}",
                    f"{metrics1.skill_diversity:.1f}",
                    metrics1.market_trend.title(),
                    metrics1.competitiveness.title()
                ],
                industry2: [
                    f"${metrics2.avg_salary:,.0f}",
                    f"{metrics2.job_count:,}",
                    f"{metrics2.growth_rate:.1%}",
                    f"{metrics2.remote_friendly:.1%}",
                    f"{metrics2.skill_diversity:.1f}",
                    metrics2.market_trend.title(),
                    metrics2.competitiveness.title()
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Radar chart comparison
            self._create_radar_comparison(metrics1, metrics2, industry1, industry2)
    
    def _render_insights_and_recommendations(self, industry_metrics: Dict[str, IndustryMetrics]):
        """Render insights and recommendations section"""
        
        st.header("💡 Insights & Recommendations")
        
        # Generate insights
        insights = self._generate_industry_insights(industry_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Insights")
            for insight in insights["key_insights"]:
                st.info(insight)
        
        with col2:
            st.subheader("Recommendations")
            for recommendation in insights["recommendations"]:
                st.success(recommendation)
        
        # Market opportunities
        st.subheader("Market Opportunities")
        opportunities = insights["opportunities"]
        
        for i, opportunity in enumerate(opportunities):
            with st.expander(f"Opportunity {i+1}: {opportunity['title']}"):
                st.write(opportunity['description'])
                st.write(f"**Potential Impact:** {opportunity['impact']}")
                st.write(f"**Timeline:** {opportunity['timeline']}")
    
    # Helper methods
    
    def _extract_unique_locations(self, job_data: pd.DataFrame) -> List[str]:
        """Extract unique locations from job data"""
        if 'location' in job_data.columns:
            locations = job_data['location'].dropna().unique()
            return sorted([loc for loc in locations if loc != "Unknown"])
        return []
    
    def _analyze_industries(
        self, 
        job_data: pd.DataFrame, 
        selected_industries: List[str],
        location_filter: List[str]
    ) -> Dict[str, IndustryMetrics]:
        """Analyze job data for selected industries"""
        
        industry_metrics = {}
        
        # Filter by location if specified
        if "All" not in location_filter and location_filter:
            job_data = job_data[job_data['location'].isin(location_filter)]
        
        for industry in selected_industries:
            industry_jobs = self._filter_jobs_by_industry(job_data, industry)
            
            if not industry_jobs.empty:
                metrics = self._calculate_industry_metrics(industry, industry_jobs)
                industry_metrics[industry] = metrics
        
        return industry_metrics
    
    def _filter_jobs_by_industry(self, job_data: pd.DataFrame, industry: str) -> pd.DataFrame:
        """Filter jobs belonging to specific industry"""
        keywords = self.industries.get(industry, [])
        
        if not keywords:
            return pd.DataFrame()
        
        # Create regex pattern for industry keywords
        pattern = '|'.join(keywords)
        
        # Filter jobs based on title, description, or company
        industry_mask = (
            job_data['title'].str.contains(pattern, case=False, na=False) |
            job_data.get('description', pd.Series()).str.contains(pattern, case=False, na=False) |
            job_data.get('company', pd.Series()).str.contains(pattern, case=False, na=False)
        )
        
        return job_data[industry_mask]
    
    def _calculate_industry_metrics(self, industry: str, industry_jobs: pd.DataFrame) -> IndustryMetrics:
        """Calculate comprehensive metrics for an industry"""
        
        # Basic metrics
        job_count = len(industry_jobs)
        avg_salary = industry_jobs.get('salary', pd.Series()).mean() if 'salary' in industry_jobs.columns else 75000
        
        # Growth rate (simulated based on job posting volume)
        growth_rate = min(0.20, job_count / 1000)  # Cap at 20%
        
        # Remote work analysis
        remote_jobs = 0
        if 'location' in industry_jobs.columns:
            remote_jobs = industry_jobs['location'].str.contains('remote', case=False, na=False).sum()
        remote_friendly = remote_jobs / job_count if job_count > 0 else 0
        
        # Skill diversity (placeholder calculation)
        skill_diversity = np.random.uniform(3.0, 8.0)  # Would be calculated from actual skills data
        
        # Experience demand (simulated)
        experience_demand = {
            "entry": int(job_count * 0.3),
            "mid": int(job_count * 0.5), 
            "senior": int(job_count * 0.2)
        }
        
        # Top skills (industry-specific)
        top_skills = self._get_industry_top_skills(industry)
        
        # Top companies (extract from job data)
        top_companies = []
        if 'company' in industry_jobs.columns:
            company_counts = industry_jobs['company'].value_counts().head(10)
            top_companies = company_counts.index.tolist()
        
        # Geographic spread
        geographic_spread = {}
        if 'location' in industry_jobs.columns:
            location_counts = industry_jobs['location'].value_counts().head(8)
            geographic_spread = location_counts.to_dict()
        
        # Market trend and competitiveness
        market_trend = self._determine_market_trend(industry, growth_rate)
        competitiveness = self._assess_competitiveness(industry, job_count, avg_salary)
        
        return IndustryMetrics(
            name=industry,
            avg_salary=avg_salary,
            job_count=job_count,
            growth_rate=growth_rate,
            remote_friendly=remote_friendly,
            skill_diversity=skill_diversity,
            experience_demand=experience_demand,
            top_skills=top_skills,
            top_companies=top_companies,
            geographic_spread=geographic_spread,
            market_trend=market_trend,
            competitiveness=competitiveness
        )
    
    def _get_industry_top_skills(self, industry: str) -> List[str]:
        """Get top skills for specific industry"""
        skills_by_industry = {
            "Technology": ["Python", "JavaScript", "AWS", "React", "Docker", "Kubernetes", "SQL", "Git"],
            "Finance & Banking": ["Excel", "SQL", "Python", "Risk Management", "Financial Modeling", "Compliance", "Bloomberg"],
            "Healthcare": ["Clinical Research", "HIPAA", "Electronic Health Records", "Data Analysis", "Regulatory Affairs"],
            "Consulting": ["Strategy", "Business Analysis", "Project Management", "Stakeholder Management", "Problem Solving"],
            "E-commerce & Retail": ["Digital Marketing", "Analytics", "Supply Chain", "Customer Experience", "E-commerce Platforms"],
            "Media & Entertainment": ["Content Creation", "Social Media", "Adobe Creative Suite", "Video Production", "Digital Marketing"],
            "Education": ["Curriculum Development", "Learning Management Systems", "Educational Technology", "Assessment"],
            "Manufacturing": ["Lean Manufacturing", "Quality Control", "Supply Chain", "Six Sigma", "CAD"],
            "Energy & Utilities": ["Renewable Energy", "Grid Management", "Environmental Compliance", "Project Management"],
            "Real Estate": ["Property Management", "Market Analysis", "Real Estate Law", "Construction Management"]
        }
        
        return skills_by_industry.get(industry, ["Communication", "Project Management", "Analysis"])
    
    def _determine_market_trend(self, industry: str, growth_rate: float) -> str:
        """Determine market trend for industry"""
        if growth_rate > 0.10:
            return "growing"
        elif growth_rate > 0.05:
            return "stable"
        else:
            return "declining"
    
    def _assess_competitiveness(self, industry: str, job_count: int, avg_salary: float) -> str:
        """Assess market competitiveness"""
        if job_count > 500 and avg_salary > 90000:
            return "high"
        elif job_count > 200 and avg_salary > 70000:
            return "medium"
        else:
            return "low"
    
    def _calculate_skill_overlap(self, industry_metrics: Dict[str, IndustryMetrics]) -> np.ndarray:
        """Calculate skills overlap matrix between industries"""
        industries = list(industry_metrics.keys())
        n = len(industries)
        overlap_matrix = np.zeros((n, n))
        
        for i, ind1 in enumerate(industries):
            for j, ind2 in enumerate(industries):
                if i == j:
                    overlap_matrix[i][j] = 1.0
                else:
                    skills1 = set(industry_metrics[ind1].top_skills)
                    skills2 = set(industry_metrics[ind2].top_skills)
                    overlap = len(skills1.intersection(skills2)) / len(skills1.union(skills2))
                    overlap_matrix[i][j] = overlap
        
        return overlap_matrix
    
    def _create_radar_comparison(self, metrics1: IndustryMetrics, metrics2: IndustryMetrics, industry1: str, industry2: str):
        """Create radar chart for industry comparison"""
        
        categories = ['Avg Salary', 'Job Count', 'Growth Rate', 'Remote Work', 'Skill Diversity']
        
        # Normalize values for radar chart (0-1 scale)
        max_salary = max(metrics1.avg_salary, metrics2.avg_salary)
        max_jobs = max(metrics1.job_count, metrics2.job_count)
        max_growth = max(metrics1.growth_rate, metrics2.growth_rate)
        
        values1 = [
            metrics1.avg_salary / max_salary,
            metrics1.job_count / max_jobs, 
            metrics1.growth_rate / max_growth if max_growth > 0 else 0,
            metrics1.remote_friendly,
            metrics1.skill_diversity / 10  # Normalize to 0-1
        ]
        
        values2 = [
            metrics2.avg_salary / max_salary,
            metrics2.job_count / max_jobs,
            metrics2.growth_rate / max_growth if max_growth > 0 else 0,
            metrics2.remote_friendly,
            metrics2.skill_diversity / 10
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values1 + [values1[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=industry1,
            line_color=self.colors.get(industry1, '#1f77b4')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values2 + [values2[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=industry2,
            line_color=self.colors.get(industry2, '#ff7f0e')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Industry Comparison Radar Chart",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _generate_industry_insights(self, industry_metrics: Dict[str, IndustryMetrics]) -> Dict[str, List[Dict]]:
        """Generate insights and recommendations"""
        
        # Find best performing industries in different categories
        highest_salary = max(industry_metrics.items(), key=lambda x: x[1].avg_salary)
        fastest_growing = max(industry_metrics.items(), key=lambda x: x[1].growth_rate)
        most_remote = max(industry_metrics.items(), key=lambda x: x[1].remote_friendly)
        most_jobs = max(industry_metrics.items(), key=lambda x: x[1].job_count)
        
        insights = {
            "key_insights": [
                f"{highest_salary[0]} offers the highest average salary at ${highest_salary[1].avg_salary:,.0f}",
                f"{fastest_growing[0]} is the fastest growing industry with {fastest_growing[1].growth_rate:.1%} growth",
                f"{most_remote[0]} is most remote-friendly with {most_remote[1].remote_friendly:.1%} remote positions",
                f"{most_jobs[0]} has the most job opportunities with {most_jobs[1].job_count:,} positions"
            ],
            "recommendations": [
                f"Consider {fastest_growing[0]} for long-term career growth",
                f"Target {highest_salary[0]} for maximum earning potential",
                f"Explore {most_remote[0]} for remote work opportunities",
                f"Focus on skill development in high-growth areas like AI and cloud computing"
            ],
            "opportunities": [
                {
                    "title": "Cross-Industry Skills Transfer",
                    "description": f"Skills from {highest_salary[0]} are transferable to other growing industries",
                    "impact": "High potential for career pivoting",
                    "timeline": "6-12 months"
                },
                {
                    "title": "Remote Work Arbitrage",
                    "description": f"Leverage remote opportunities in {most_remote[0]} while living in lower-cost areas",
                    "impact": "Significant cost of living arbitrage",
                    "timeline": "Immediate"
                },
                {
                    "title": "Emerging Role Creation",
                    "description": f"New roles emerging at intersection of {fastest_growing[0]} and {highest_salary[0]}",
                    "impact": "First-mover advantage in new market segments",
                    "timeline": "12-24 months"
                }
            ]
        }
        
        return insights

# Usage example and testing
if __name__ == "__main__":
    # This would normally be called from the main Streamlit app
    st.set_page_config(page_title="Industry Comparison Dashboard", layout="wide")
    
    dashboard = IndustryComparisonDashboard()
    
    # Sample data for testing
    sample_data = pd.DataFrame({
        'title': ['Software Engineer', 'Data Scientist', 'Product Manager', 'Financial Analyst'],
        'company': ['Google', 'Meta', 'Apple', 'Goldman Sachs'],
        'location': ['San Francisco', 'New York', 'Austin', 'New York'],
        'salary': [150000, 180000, 140000, 120000],
        'description': ['Python development', 'Machine learning', 'Product strategy', 'Financial modeling']
    })
    
    dashboard.render_dashboard(sample_data)
