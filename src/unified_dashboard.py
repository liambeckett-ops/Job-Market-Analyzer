"""
Unified Job Market Analysis Dashboard
Integrates all analysis features with user authentication and personalized experiences
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our enhanced analyzers
from analyzers.skills_gap_analyzer import SkillsGapAnalyzer
from analyzers.salary_analyzer import SalaryAnalyzer
from analyzers.resume_optimizer import ResumeOptimizer
from dashboards.industry_comparison import IndustryComparisonDashboard
from auth.user_auth import AuthenticationManager, CacheManager
from database.db_manager import DatabaseManager

# Configure Streamlit page
st.set_page_config(
    page_title="Job Market Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

class UnifiedDashboard:
    """Main unified dashboard class"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        self.skills_analyzer = SkillsGapAnalyzer()
        self.salary_analyzer = SalaryAnalyzer()
        self.resume_optimizer = ResumeOptimizer()
        self.industry_dashboard = IndustryComparisonDashboard()
        
        # Load job market data
        self.job_data = self._load_job_data()
    
    def _load_job_data(self) -> pd.DataFrame:
        """Load job data from database"""
        try:
            jobs = self.db_manager.get_jobs(limit=10000)  # Load recent jobs
            if jobs:
                return pd.DataFrame(jobs)
            else:
                # Return sample data if no real data available
                return self._generate_sample_data()
        except Exception as e:
            st.error(f"Error loading job data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        companies = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Uber", "Airbnb", "Spotify", "Stripe"]
        roles = ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", "UX Designer"]
        locations = ["San Francisco", "New York", "Seattle", "Austin", "Boston", "Remote"]
        industries = ["Technology", "Finance", "Healthcare", "E-commerce", "Media"]
        
        sample_data = []
        for i in range(1000):
            sample_data.append({
                'id': i + 1,
                'title': np.random.choice(roles),
                'company': np.random.choice(companies),
                'location': np.random.choice(locations),
                'industry': np.random.choice(industries),
                'salary': np.random.normal(120000, 30000),
                'description': f"Sample job description for {np.random.choice(roles)}",
                'skills': np.random.choice([
                    ["Python", "SQL", "AWS"],
                    ["JavaScript", "React", "Node.js"],
                    ["Java", "Spring", "Docker"],
                    ["Python", "Machine Learning", "TensorFlow"],
                    ["Product Strategy", "Analytics", "Agile"]
                ]),
                'posted_date': datetime.now() - timedelta(days=np.random.randint(1, 90)),
                'experience_level': np.random.choice(["Entry", "Mid", "Senior"])
            })
        
        return pd.DataFrame(sample_data)
    
    def render_authentication(self):
        """Render authentication interface"""
        
        if st.session_state.authenticated:
            return True
        
        st.title("🔐 Job Market Analyzer Pro")
        st.markdown("**Enterprise-grade job market analysis with personalized insights**")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            self._render_login_form()
        
        with tab2:
            self._render_registration_form()
        
        return False
    
    def _render_login_form(self):
        """Render login form"""
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login", type="primary"):
                if username and password:
                    result = self.auth_manager.authenticate_user(username, password)
                    
                    if result['success']:
                        st.session_state.authenticated = True
                        st.session_state.user_info = {
                            'user_id': result['user_id'],
                            'username': result['username'],
                            'token': result['token']
                        }
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result['error'])
                else:
                    st.warning("Please enter both username and password")
    
    def _render_registration_form(self):
        """Render registration form"""
        st.subheader("Create New Account")
        
        with st.form("registration_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                full_name = st.text_input("Full Name")
            
            with col2:
                target_roles = st.multiselect(
                    "Target Roles",
                    ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", 
                     "UX Designer", "Marketing Manager", "Sales Manager"]
                )
                
                target_industries = st.multiselect(
                    "Target Industries",
                    ["Technology", "Finance", "Healthcare", "E-commerce", "Media", "Consulting"]
                )
                
                experience_level = st.selectbox(
                    "Experience Level",
                    ["Entry", "Mid", "Senior", "Executive"]
                )
                
                preferred_locations = st.multiselect(
                    "Preferred Locations",
                    ["San Francisco", "New York", "Seattle", "Austin", "Boston", "Remote", "Los Angeles"]
                )
            
            current_skills = st.text_area(
                "Current Skills (comma-separated)",
                placeholder="e.g., Python, SQL, Machine Learning, Project Management"
            )
            
            if st.form_submit_button("Register", type="primary"):
                if username and email and password:
                    skills_list = [skill.strip() for skill in current_skills.split(",") if skill.strip()]
                    
                    result = self.auth_manager.register_user(
                        username=username,
                        email=email,
                        password=password,
                        full_name=full_name,
                        target_roles=target_roles,
                        target_industries=target_industries,
                        experience_level=experience_level.lower(),
                        preferred_locations=preferred_locations,
                        current_skills=skills_list
                    )
                    
                    if result['success']:
                        st.success("Registration successful! Please login with your credentials.")
                    else:
                        st.error(result['error'])
                else:
                    st.warning("Please fill in all required fields")
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        
        with st.sidebar:
            # User info
            if st.session_state.authenticated:
                user_profile = self.auth_manager.get_user_profile(st.session_state.user_info['user_id'])
                
                st.markdown("---")
                st.markdown(f"**Welcome, {user_profile.username}!**")
                
                if st.button("Logout", type="secondary"):
                    self.auth_manager.logout_user(st.session_state.user_info['token'])
                    st.session_state.authenticated = False
                    st.session_state.user_info = None
                    st.rerun()
                
                st.markdown("---")
                
                # Navigation
                st.subheader("🧭 Navigation")
                
                pages = [
                    ("📊", "Dashboard"),
                    ("🎯", "Skills Gap Analysis"),
                    ("💰", "Salary Analysis"),
                    ("🏭", "Industry Comparison"),
                    ("📄", "Resume Optimizer"),
                    ("👤", "Profile"),
                    ("📈", "Analysis History")
                ]
                
                for icon, page in pages:
                    if st.button(f"{icon} {page}", key=f"nav_{page}"):
                        st.session_state.current_page = page
                        st.rerun()
                
                st.markdown("---")
                
                # Quick stats
                st.subheader("📊 Quick Stats")
                analysis_history = self.auth_manager.get_analysis_history(
                    st.session_state.user_info['user_id'], limit=10
                )
                
                st.metric("Total Analyses", len(analysis_history))
                st.metric("Jobs in Database", len(self.job_data))
                
                if user_profile.target_roles:
                    st.metric("Target Roles", len(user_profile.target_roles))
    
    def render_dashboard(self):
        """Render main dashboard overview"""
        
        user_profile = self.auth_manager.get_user_profile(st.session_state.user_info['user_id'])
        
        st.title("📊 Job Market Dashboard")
        st.markdown(f"**Personalized insights for {user_profile.full_name or user_profile.username}**")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_jobs = len(self.job_data)
            st.metric("Total Jobs", f"{total_jobs:,}")
        
        with col2:
            avg_salary = self.job_data['salary'].mean() if 'salary' in self.job_data.columns else 0
            st.metric("Average Salary", f"${avg_salary:,.0f}")
        
        with col3:
            unique_companies = self.job_data['company'].nunique() if 'company' in self.job_data.columns else 0
            st.metric("Companies", unique_companies)
        
        with col4:
            recent_jobs = len(self.job_data[self.job_data['posted_date'] > datetime.now() - timedelta(days=7)]) if 'posted_date' in self.job_data.columns else 0
            st.metric("New This Week", recent_jobs)
        
        # Personalized recommendations
        st.subheader("🎯 Personalized Recommendations")
        
        if user_profile.target_roles:
            # Filter jobs for user's target roles
            target_jobs = self.job_data[
                self.job_data['title'].str.contains('|'.join(user_profile.target_roles), case=False, na=False)
            ]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Opportunities in Your Target Roles:**")
                for role in user_profile.target_roles:
                    role_count = len(self.job_data[self.job_data['title'].str.contains(role, case=False, na=False)])
                    st.write(f"• {role}: {role_count} jobs")
            
            with col2:
                if len(target_jobs) > 0:
                    avg_target_salary = target_jobs['salary'].mean()
                    st.metric("Average Salary (Target Roles)", f"${avg_target_salary:,.0f}")
                    
                    # Top companies for target roles
                    top_companies = target_jobs['company'].value_counts().head(5)
                    st.write("**Top Hiring Companies:**")
                    for company, count in top_companies.items():
                        st.write(f"• {company}: {count} positions")
        
        # Market trends visualization
        st.subheader("📈 Market Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary distribution by role
            if 'title' in self.job_data.columns and 'salary' in self.job_data.columns:
                role_salaries = self.job_data.groupby('title')['salary'].mean().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=role_salaries.values,
                    y=role_salaries.index,
                    orientation='h',
                    title="Average Salary by Role",
                    labels={'x': 'Average Salary ($)', 'y': 'Role'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Job postings by location
            if 'location' in self.job_data.columns:
                location_counts = self.job_data['location'].value_counts().head(10)
                
                fig = px.pie(
                    values=location_counts.values,
                    names=location_counts.index,
                    title="Job Distribution by Location"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent analysis history
        st.subheader("📋 Recent Analysis")
        
        analysis_history = self.auth_manager.get_analysis_history(
            st.session_state.user_info['user_id'], limit=5
        )
        
        if analysis_history:
            for analysis in analysis_history:
                with st.expander(f"{analysis.analysis_type} - {analysis.created_at.strftime('%Y-%m-%d %H:%M')}"):
                    st.json(analysis.parameters)
        else:
            st.info("No analysis history yet. Try running some analyses!")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Analyze Skills Gap", type="primary"):
                st.session_state.current_page = "Skills Gap Analysis"
                st.rerun()
        
        with col2:
            if st.button("💰 Salary Analysis", type="primary"):
                st.session_state.current_page = "Salary Analysis"
                st.rerun()
        
        with col3:
            if st.button("📄 Optimize Resume", type="primary"):
                st.session_state.current_page = "Resume Optimizer"
                st.rerun()
    
    def render_skills_gap_analysis(self):
        """Render skills gap analysis page"""
        
        st.title("🎯 Skills Gap Analysis")
        st.markdown("Identify skill gaps and get personalized learning recommendations")
        
        user_profile = self.auth_manager.get_user_profile(st.session_state.user_info['user_id'])
        
        # Configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_role = st.selectbox(
                "Target Role",
                user_profile.target_roles + ["Other"],
                index=0 if user_profile.target_roles else 0
            )
            
            if target_role == "Other":
                target_role = st.text_input("Specify target role")
            
            target_industry = st.selectbox(
                "Target Industry",
                user_profile.target_industries + ["Other"],
                index=0 if user_profile.target_industries else 0
            )
            
            location = st.selectbox(
                "Location",
                user_profile.preferred_locations + ["All Locations"],
                index=0 if user_profile.preferred_locations else 0
            )
        
        with col2:
            st.markdown("**Your Current Skills:**")
            current_skills_display = user_profile.current_skills[:10]  # Show first 10
            for skill in current_skills_display:
                st.write(f"✅ {skill}")
            
            if len(user_profile.current_skills) > 10:
                st.write(f"... and {len(user_profile.current_skills) - 10} more")
        
        if st.button("🔍 Analyze Skills Gap", type="primary"):
            with st.spinner("Analyzing skills gap..."):
                
                # Perform skills gap analysis
                analysis = self.skills_analyzer.analyze_skills_gap(
                    self.job_data,
                    user_skills=user_profile.current_skills,
                    target_roles=[target_role] if target_role else None,
                    location=location if location != "All Locations" else None
                )
                
                # Save analysis to history
                self.auth_manager.save_analysis_history(
                    user_profile.user_id,
                    "Skills Gap Analysis",
                    {
                        "target_role": target_role,
                        "target_industry": target_industry,
                        "location": location
                    },
                    {
                        "total_skills_identified": analysis.total_skills_identified,
                        "critical_gaps_count": len(analysis.critical_skill_gaps),
                        "overall_score": len(user_profile.current_skills) / max(analysis.total_skills_identified, 1) * 100
                    }
                )
                
                # Display results
                st.success("Analysis complete!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Skills Analyzed", analysis.total_skills_identified)
                
                with col2:
                    st.metric("Critical Gaps", len(analysis.critical_skill_gaps))
                
                with col3:
                    st.metric("Your Skills", len(user_profile.current_skills))
                
                with col4:
                    coverage = len(user_profile.current_skills) / max(analysis.total_skills_identified, 1) * 100
                    st.metric("Skills Coverage", f"{coverage:.0f}%")
                
                # Critical skill gaps
                st.subheader("🚨 Critical Skill Gaps")
                
                if analysis.critical_skill_gaps:
                    for i, skill in enumerate(analysis.critical_skill_gaps[:10], 1):
                        with st.expander(f"{i}. {skill.name} (Priority: {skill.priority_level})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Demand Score:** {skill.demand_score:.2f}")
                                st.write(f"**Job Count:** {skill.job_count}")
                                st.write(f"**Average Salary Impact:** ${skill.avg_salary:,.0f}")
                            
                            with col2:
                                st.write("**Learning Resources:**")
                                for resource in skill.learning_resources[:3]:
                                    st.write(f"• {resource}")
                                
                                if skill.related_skills:
                                    st.write("**Related Skills:**")
                                    st.write(", ".join(skill.related_skills[:5]))
                else:
                    st.info("Great! No critical skill gaps identified.")
                
                # Learning roadmap
                st.subheader("🗺️ Learning Roadmap")
                
                if analysis.critical_skill_gaps:
                    roadmap = self.skills_analyzer.generate_learning_roadmap(
                        user_profile.current_skills,
                        [skill.name for skill in analysis.critical_skill_gaps[:5]],
                        timeline_months=12
                    )
                    
                    if roadmap["learning_roadmap"]:
                        timeline_df = pd.DataFrame(roadmap["learning_roadmap"])
                        
                        # Create Gantt chart
                        fig = px.timeline(
                            timeline_df,
                            x_start="start_week",
                            x_end="end_week",
                            y="skill",
                            color="difficulty",
                            title="12-Month Learning Timeline"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Difficulty breakdown
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            difficulty_breakdown = roadmap["difficulty_breakdown"]
                            fig = px.pie(
                                values=list(difficulty_breakdown.values()),
                                names=list(difficulty_breakdown.keys()),
                                title="Learning Difficulty Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write("**Recommended Resources:**")
                            resources = roadmap["recommended_resources"]
                            for skill, skill_resources in resources.items():
                                st.write(f"**{skill}:**")
                                for resource in skill_resources[:2]:
                                    st.write(f"• {resource}")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                for i, recommendation in enumerate(analysis.recommendations, 1):
                    st.write(f"{i}. {recommendation}")
    
    def render_salary_analysis(self):
        """Render salary analysis page"""
        
        st.title("💰 Salary Analysis & Trends")
        st.markdown("Get comprehensive salary insights and market predictions")
        
        user_profile = self.auth_manager.get_user_profile(st.session_state.user_info['user_id'])
        
        # Analysis configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Market Overview", "Role-Specific", "Trend Prediction", "Geographic Comparison"]
            )
        
        with col2:
            target_role = st.selectbox(
                "Target Role",
                ["All Roles"] + user_profile.target_roles,
                index=1 if user_profile.target_roles else 0
            )
        
        with col3:
            location = st.selectbox(
                "Location",
                ["All Locations"] + user_profile.preferred_locations,
                index=1 if user_profile.preferred_locations else 0
            )
        
        if st.button("📊 Analyze Salaries", type="primary"):
            with st.spinner("Analyzing salary data..."):
                
                # Load salary data
                self.salary_analyzer.load_data(self.job_data.to_dict('records'))
                
                if analysis_type == "Market Overview":
                    # General market analysis
                    stats = self.salary_analyzer.get_salary_statistics(
                        job_title=target_role if target_role != "All Roles" else None,
                        location=location if location != "All Locations" else None
                    )
                    
                    if 'error' not in stats:
                        # Display key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Jobs Analyzed", f"{stats['count']:,}")
                        
                        with col2:
                            st.metric("Median Salary", f"${stats['median_salary']:,.0f}")
                        
                        with col3:
                            st.metric("Average Salary", f"${stats['mean_salary']:,.0f}")
                        
                        with col4:
                            st.metric("Salary Range", f"${stats['min_salary']:,.0f} - ${stats['max_salary']:,.0f}")
                        
                        # Salary distribution
                        st.subheader("💰 Salary Distribution")
                        
                        ranges = stats['salary_ranges']
                        fig = px.bar(
                            x=list(ranges.keys()),
                            y=list(ranges.values()),
                            title="Salary Range Distribution",
                            labels={'x': 'Salary Range', 'y': 'Number of Jobs'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Percentile analysis
                        st.subheader("📊 Salary Percentiles")
                        
                        percentile_data = {
                            "25th Percentile": stats['percentile_25'],
                            "Median (50th)": stats['median_salary'],
                            "75th Percentile": stats['percentile_75'],
                            "90th Percentile": stats['percentile_90']
                        }
                        
                        fig = px.bar(
                            x=list(percentile_data.keys()),
                            y=list(percentile_data.values()),
                            title="Salary Percentile Breakdown",
                            labels={'x': 'Percentile', 'y': 'Salary ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                elif analysis_type == "Trend Prediction":
                    # Salary trend analysis
                    st.subheader("📈 Salary Trend Predictions")
                    
                    if target_role != "All Roles":
                        trend = self.salary_analyzer.predict_salary_trends(
                            role=target_role,
                            location=location if location != "All Locations" else None,
                            projection_months=24
                        )
                        
                        # Display trend information
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Median", f"${trend.current_median:,.0f}")
                        
                        with col2:
                            st.metric("12-Month Projection", f"${trend.projection_1y:,.0f}")
                        
                        with col3:
                            growth_rate_percent = trend.growth_rate * 100
                            st.metric("Growth Rate", f"{growth_rate_percent:+.1f}%")
                        
                        # Trend chart
                        months = ["Current", "6 Months", "12 Months", "24 Months"]
                        projections = [trend.current_median, trend.projection_6m, trend.projection_1y, trend.projection_2y]
                        
                        fig = px.line(
                            x=months,
                            y=projections,
                            title=f"Salary Trend Projection: {target_role}",
                            labels={'x': 'Time Period', 'y': 'Projected Salary ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Influencing factors
                        st.subheader("🔍 Trend Factors")
                        
                        for factor in trend.factors:
                            st.write(f"• {factor}")
                        
                        # Confidence indicator
                        confidence_percentage = trend.confidence * 100
                        st.info(f"**Prediction Confidence:** {confidence_percentage:.0f}% (based on {trend.sample_size} job samples)")
                
                elif analysis_type == "Geographic Comparison":
                    # Geographic salary comparison
                    locations_to_compare = user_profile.preferred_locations[:5] if user_profile.preferred_locations else [
                        "San Francisco", "New York", "Seattle", "Austin", "Boston"
                    ]
                    
                    comparison = self.salary_analyzer.compare_salary_markets(
                        locations=locations_to_compare,
                        role=target_role if target_role != "All Roles" else None
                    )
                    
                    st.subheader("🌍 Geographic Salary Comparison")
                    
                    # Create comparison chart
                    if comparison["market_data"]:
                        locations = list(comparison["market_data"].keys())
                        salaries = [data["median_salary"] for data in comparison["market_data"].values()]
                        adjusted_salaries = [data["cost_of_living_adjusted"] for data in comparison["market_data"].values()]
                        
                        comparison_df = pd.DataFrame({
                            "Location": locations,
                            "Raw Salary": salaries,
                            "COL Adjusted": adjusted_salaries
                        })
                        
                        fig = px.bar(
                            comparison_df,
                            x="Location",
                            y=["Raw Salary", "COL Adjusted"],
                            title="Salary Comparison by Location",
                            barmode="group"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Market rankings
                        st.subheader("🏆 Market Rankings")
                        
                        rankings = comparison["rankings"]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**By Raw Salary:**")
                            for i, location in enumerate(rankings["by_raw_salary"], 1):
                                st.write(f"{i}. {location}")
                        
                        with col2:
                            st.write("**By Cost-Adjusted Salary:**")
                            for i, location in enumerate(rankings["by_adjusted_salary"], 1):
                                st.write(f"{i}. {location}")
                
                # Save analysis to history
                self.auth_manager.save_analysis_history(
                    user_profile.user_id,
                    "Salary Analysis",
                    {
                        "analysis_type": analysis_type,
                        "target_role": target_role,
                        "location": location
                    },
                    {"timestamp": datetime.now().isoformat()}
                )
    
    def render_industry_comparison(self):
        """Render industry comparison page"""
        
        st.title("🏭 Industry Comparison")
        st.markdown("Compare job markets across different industries")
        
        # Use the existing industry comparison dashboard
        self.industry_dashboard.render_dashboard(self.job_data)
    
    def render_resume_optimizer(self):
        """Render resume optimizer page"""
        
        st.title("📄 Resume Optimizer")
        st.markdown("Get AI-powered resume optimization suggestions")
        
        user_profile = self.auth_manager.get_user_profile(st.session_state.user_info['user_id'])
        
        # Resume input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📄 Resume Input")
            
            input_method = st.radio(
                "Choose input method:",
                ["Paste Text", "Upload File"]
            )
            
            resume_text = ""
            
            if input_method == "Paste Text":
                resume_text = st.text_area(
                    "Paste your resume text here:",
                    height=300,
                    placeholder="Copy and paste your resume content..."
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload your resume",
                    type=['pdf', 'docx', 'txt'],
                    help="Supported formats: PDF, DOCX, TXT"
                )
                
                if uploaded_file:
                    # In a real implementation, you would extract text from the file
                    st.success("File uploaded successfully!")
                    resume_text = "Sample resume text would be extracted here..."
        
        with col2:
            st.subheader("🎯 Analysis Configuration")
            
            target_role = st.selectbox(
                "Target Role",
                user_profile.target_roles + ["Other"],
                index=0 if user_profile.target_roles else 0
            )
            
            if target_role == "Other":
                target_role = st.text_input("Specify target role")
            
            target_industry = st.selectbox(
                "Target Industry",
                user_profile.target_industries + ["Other"],
                index=0 if user_profile.target_industries else 0
            )
            
            experience_level = st.selectbox(
                "Experience Level",
                ["Entry Level", "Mid Level", "Senior Level", "Executive"],
                index=["entry", "mid", "senior", "executive"].index(user_profile.experience_level)
            )
        
        if resume_text and target_role:
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("Analyzing your resume..."):
                    
                    # Initialize resume optimizer with job market data
                    optimizer = ResumeOptimizer(self.job_data)
                    
                    # Perform analysis
                    analysis = optimizer.analyze_resume(
                        resume_text,
                        target_role,
                        target_industry.lower(),
                        experience_level.lower().replace(" level", "")
                    )
                    
                    # Save analysis to history
                    self.auth_manager.save_analysis_history(
                        user_profile.user_id,
                        "Resume Analysis",
                        {
                            "target_role": target_role,
                            "target_industry": target_industry,
                            "experience_level": experience_level
                        },
                        {
                            "overall_score": analysis.overall_score,
                            "ats_compatibility": analysis.ats_compatibility,
                            "role_fit_score": analysis.role_fit_score
                        }
                    )
                    
                    # Display results
                    st.success("Resume analysis complete!")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Overall Score", f"{analysis.overall_score:.0f}/100")
                    
                    with col2:
                        st.metric("ATS Compatibility", f"{analysis.ats_compatibility:.0f}/100")
                    
                    with col3:
                        st.metric("Keyword Density", f"{analysis.keyword_density:.0f}%")
                    
                    with col4:
                        st.metric("Role Fit", f"{analysis.role_fit_score:.0f}/100")
                    
                    # Section analysis
                    st.subheader("📝 Section Analysis")
                    
                    for section_name, section in analysis.sections.items():
                        with st.expander(f"{section_name.title()} Section (Score: {section.score:.0f}/100)"):
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if section.suggestions:
                                    st.write("**Suggestions:**")
                                    for suggestion in section.suggestions:
                                        st.write(f"• {suggestion}")
                            
                            with col2:
                                if section.missing_elements:
                                    st.write("**Missing Elements:**")
                                    for element in section.missing_elements:
                                        st.write(f"• {element}")
                    
                    # Skills analysis
                    st.subheader("🛠️ Skills Analysis")
                    
                    present_skills = [sm for sm in analysis.skill_matches if sm.present]
                    missing_critical_skills = [sm for sm in analysis.skill_matches 
                                             if not sm.present and sm.priority == "critical"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Present Skills:**")
                        for skill in present_skills[:10]:
                            st.write(f"✅ {skill.skill}")
                    
                    with col2:
                        st.write("**Missing Critical Skills:**")
                        for skill in missing_critical_skills[:10]:
                            st.write(f"❌ {skill.skill}")
                    
                    # Optimization suggestions
                    st.subheader("💡 Optimization Suggestions")
                    
                    for i, suggestion in enumerate(analysis.optimization_suggestions, 1):
                        st.write(f"{i}. {suggestion}")
                    
                    # Improvement priorities
                    if analysis.improvement_priorities:
                        st.subheader("🎯 Improvement Priorities")
                        
                        priorities_df = pd.DataFrame(analysis.improvement_priorities)
                        st.dataframe(priorities_df, use_container_width=True)
                    
                    # Generate detailed suggestions
                    if st.button("📋 Generate Detailed Optimization Plan"):
                        with st.spinner("Creating optimization plan..."):
                            detailed_suggestions = optimizer.generate_optimized_suggestions(analysis)
                            
                            st.subheader("📋 Detailed Optimization Plan")
                            
                            # Immediate actions
                            st.write("**🔥 Immediate Actions:**")
                            for action in detailed_suggestions["immediate_actions"]:
                                st.write(f"• {action}")
                            
                            # Section enhancements
                            if detailed_suggestions["section_enhancements"]:
                                st.write("**📝 Section Enhancements:**")
                                for section, details in detailed_suggestions["section_enhancements"].items():
                                    st.write(f"**{section.title()}** (Priority: {details['priority']})")
                                    for suggestion in details['suggestions'][:3]:
                                        st.write(f"  • {suggestion}")
    
    def render_profile(self):
        """Render user profile page"""
        
        st.title("👤 User Profile")
        st.markdown("Manage your profile and preferences")
        
        user_profile = self.auth_manager.get_user_profile(st.session_state.user_info['user_id'])
        
        # Profile information
        with st.form("profile_form"):
            st.subheader("📋 Profile Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input("Full Name", value=user_profile.full_name)
                experience_level = st.selectbox(
                    "Experience Level",
                    ["entry", "mid", "senior", "executive"],
                    index=["entry", "mid", "senior", "executive"].index(user_profile.experience_level)
                )
            
            with col2:
                email_display = st.text_input("Email", value=user_profile.email, disabled=True)
                username_display = st.text_input("Username", value=user_profile.username, disabled=True)
            
            # Target roles
            st.subheader("🎯 Career Goals")
            
            target_roles = st.multiselect(
                "Target Roles",
                ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", 
                 "UX Designer", "Marketing Manager", "Sales Manager", "Other"],
                default=user_profile.target_roles
            )
            
            target_industries = st.multiselect(
                "Target Industries",
                ["Technology", "Finance", "Healthcare", "E-commerce", "Media", "Consulting", "Other"],
                default=user_profile.target_industries
            )
            
            preferred_locations = st.multiselect(
                "Preferred Locations",
                ["San Francisco", "New York", "Seattle", "Austin", "Boston", "Remote", 
                 "Los Angeles", "Chicago", "Denver", "Other"],
                default=user_profile.preferred_locations
            )
            
            # Skills
            st.subheader("🛠️ Skills")
            
            current_skills_text = ", ".join(user_profile.current_skills)
            current_skills = st.text_area(
                "Current Skills (comma-separated)",
                value=current_skills_text,
                help="List your current skills separated by commas"
            )
            
            # Salary expectations
            st.subheader("💰 Salary Expectations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_salary = st.number_input(
                    "Minimum Salary",
                    value=user_profile.salary_expectations.get('min', 50000),
                    min_value=0,
                    step=5000
                )
            
            with col2:
                max_salary = st.number_input(
                    "Maximum Salary",
                    value=user_profile.salary_expectations.get('max', 150000),
                    min_value=0,
                    step=5000
                )
            
            if st.form_submit_button("💾 Update Profile", type="primary"):
                # Update profile
                skills_list = [skill.strip() for skill in current_skills.split(",") if skill.strip()]
                
                updates = {
                    'full_name': full_name,
                    'experience_level': experience_level,
                    'target_roles': target_roles,
                    'target_industries': target_industries,
                    'preferred_locations': preferred_locations,
                    'current_skills': skills_list,
                    'salary_expectations': {'min': min_salary, 'max': max_salary}
                }
                
                if self.auth_manager.update_user_profile(user_profile.user_id, **updates):
                    st.success("Profile updated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to update profile")
        
        # Account statistics
        st.subheader("📊 Account Statistics")
        
        analysis_history = self.auth_manager.get_analysis_history(user_profile.user_id)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Analyses", len(analysis_history))
        
        with col2:
            st.metric("Member Since", user_profile.created_at.strftime("%B %Y"))
        
        with col3:
            st.metric("Last Login", user_profile.last_login.strftime("%Y-%m-%d"))
    
    def render_analysis_history(self):
        """Render analysis history page"""
        
        st.title("📈 Analysis History")
        st.markdown("View and manage your past analyses")
        
        user_profile = self.auth_manager.get_user_profile(st.session_state.user_info['user_id'])
        analysis_history = self.auth_manager.get_analysis_history(user_profile.user_id, limit=100)
        
        if not analysis_history:
            st.info("No analysis history yet. Start analyzing to see your history here!")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_types = list(set([a.analysis_type for a in analysis_history]))
            selected_type = st.selectbox("Filter by Type", ["All"] + analysis_types)
        
        with col2:
            date_range = st.selectbox(
                "Date Range",
                ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
            )
        
        with col3:
            sort_order = st.selectbox("Sort By", ["Newest First", "Oldest First"])
        
        # Apply filters
        filtered_history = analysis_history
        
        if selected_type != "All":
            filtered_history = [a for a in filtered_history if a.analysis_type == selected_type]
        
        if date_range != "All Time":
            days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
            cutoff_date = datetime.now() - timedelta(days=days_map[date_range])
            filtered_history = [a for a in filtered_history if a.created_at >= cutoff_date]
        
        if sort_order == "Oldest First":
            filtered_history = list(reversed(filtered_history))
        
        # Display statistics
        st.subheader("📊 Analysis Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(analysis_history))
        
        with col2:
            st.metric("This Month", len([a for a in analysis_history 
                                       if a.created_at >= datetime.now() - timedelta(days=30)]))
        
        with col3:
            analysis_type_counts = {}
            for a in analysis_history:
                analysis_type_counts[a.analysis_type] = analysis_type_counts.get(a.analysis_type, 0) + 1
            most_common_type = max(analysis_type_counts.items(), key=lambda x: x[1])[0]
            st.metric("Most Used", most_common_type)
        
        with col4:
            favorites_count = len([a for a in analysis_history if a.is_favorite])
            st.metric("Favorites", favorites_count)
        
        # Analysis type distribution
        if analysis_history:
            fig = px.pie(
                values=list(analysis_type_counts.values()),
                names=list(analysis_type_counts.keys()),
                title="Analysis Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display analysis history
        st.subheader("📋 Analysis History")
        
        if filtered_history:
            for analysis in filtered_history:
                with st.expander(f"{analysis.analysis_type} - {analysis.created_at.strftime('%Y-%m-%d %H:%M')}"):
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Parameters:**")
                        st.json(analysis.parameters)
                        
                        if analysis.results:
                            st.write("**Key Results:**")
                            # Display simplified results
                            for key, value in list(analysis.results.items())[:5]:
                                if isinstance(value, (int, float)):
                                    st.write(f"• {key}: {value}")
                                elif isinstance(value, str):
                                    st.write(f"• {key}: {value}")
                    
                    with col2:
                        st.write(f"**Created:** {analysis.created_at.strftime('%Y-%m-%d %H:%M')}")
                        
                        if st.button(f"⭐ {'Remove from' if analysis.is_favorite else 'Add to'} Favorites", 
                                   key=f"fav_{analysis.id}"):
                            # In a real implementation, you would update the favorite status
                            st.success("Favorite status updated!")
                        
                        if st.button(f"📊 Re-run Analysis", key=f"rerun_{analysis.id}"):
                            # Set session state to navigate to appropriate analysis page
                            if analysis.analysis_type == "Skills Gap Analysis":
                                st.session_state.current_page = "Skills Gap Analysis"
                            elif analysis.analysis_type == "Salary Analysis":
                                st.session_state.current_page = "Salary Analysis"
                            elif analysis.analysis_type == "Resume Analysis":
                                st.session_state.current_page = "Resume Optimizer"
                            st.rerun()
        else:
            st.info("No analyses found matching your filters.")
    
    def run(self):
        """Main application runner"""
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            border-left: 4px solid #1f77b4;
        }
        
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Check authentication
        if not self.render_authentication():
            return
        
        # Render sidebar navigation
        self.render_sidebar()
        
        # Route to appropriate page
        current_page = st.session_state.current_page
        
        if current_page == "Dashboard":
            self.render_dashboard()
        elif current_page == "Skills Gap Analysis":
            self.render_skills_gap_analysis()
        elif current_page == "Salary Analysis":
            self.render_salary_analysis()
        elif current_page == "Industry Comparison":
            self.render_industry_comparison()
        elif current_page == "Resume Optimizer":
            self.render_resume_optimizer()
        elif current_page == "Profile":
            self.render_profile()
        elif current_page == "Analysis History":
            self.render_analysis_history()
        else:
            self.render_dashboard()


# Main application entry point
def main():
    """Main entry point for the unified dashboard"""
    
    dashboard = UnifiedDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
