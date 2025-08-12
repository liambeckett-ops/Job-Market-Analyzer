"""
Skills Gap Analyzer - Advanced skills analysis with gap identification and recommendations
Analyzes skill demand vs supply, identifies skill gaps, and provides learning recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
import logging
from dataclasses import dataclass, asdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SkillProfile:
    """Represents a skill profile with demand and supply metrics"""
    name: str
    category: str
    demand_score: float
    supply_score: float
    gap_score: float  # Positive = high demand, low supply
    growth_trend: float
    salary_impact: float
    job_count: int
    avg_salary: float
    related_skills: List[str]
    learning_resources: List[str]
    priority_level: str  # "critical", "high", "medium", "low"

@dataclass
class SkillGapAnalysis:
    """Complete skills gap analysis results"""
    analysis_date: datetime
    total_jobs_analyzed: int
    total_skills_identified: int
    top_skills_in_demand: List[SkillProfile]
    critical_skill_gaps: List[SkillProfile]
    emerging_skills: List[SkillProfile]
    declining_skills: List[SkillProfile]
    skill_categories: Dict[str, List[SkillProfile]]
    recommendations: List[str]
    market_insights: Dict[str, Any]

class SkillsGapAnalyzer:
    """Advanced skills analyzer with gap identification capabilities"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.skill_categories = self._load_skill_categories()
        self.skill_patterns = self._load_skill_patterns()
        self.learning_resources = self._load_learning_resources()
        
    def _load_skill_categories(self) -> Dict[str, List[str]]:
        """Load predefined skill categories"""
        return {
            "programming_languages": [
                "python", "javascript", "java", "c++", "c#", "go", "rust", "swift", "kotlin",
                "typescript", "php", "ruby", "scala", "r", "matlab", "sql", "html", "css"
            ],
            "frameworks_libraries": [
                "react", "angular", "vue", "django", "flask", "spring", "express", "node.js",
                "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "keras", "opencv"
            ],
            "cloud_platforms": [
                "aws", "azure", "gcp", "google cloud", "kubernetes", "docker", "terraform",
                "ansible", "jenkins", "gitlab ci", "github actions"
            ],
            "databases": [
                "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
                "oracle", "sql server", "sqlite", "dynamodb"
            ],
            "data_science": [
                "machine learning", "deep learning", "data analysis", "statistics",
                "data visualization", "big data", "etl", "data mining", "nlp", "computer vision"
            ],
            "devops_tools": [
                "git", "linux", "bash", "powershell", "monitoring", "logging", "ci/cd",
                "infrastructure as code", "microservices", "api development"
            ],
            "soft_skills": [
                "communication", "leadership", "problem solving", "teamwork", "project management",
                "agile", "scrum", "analytical thinking", "creativity", "adaptability"
            ],
            "security": [
                "cybersecurity", "information security", "network security", "encryption",
                "penetration testing", "vulnerability assessment", "compliance"
            ]
        }
    
    def _load_skill_patterns(self) -> Dict[str, str]:
        """Load regex patterns for skill extraction"""
        return {
            "python": r"\\bpython\\b",
            "javascript": r"\\b(javascript|js)\\b",
            "react": r"\\breact(\\.js)?\\b",
            "aws": r"\\b(aws|amazon web services)\\b",
            "machine learning": r"\\b(machine learning|ml)\\b",
            "data science": r"\\b(data science|data scientist)\\b",
            # Add more patterns as needed
        }
    
    def _load_learning_resources(self) -> Dict[str, List[str]]:
        """Load learning resources for skills"""
        return {
            "python": [
                "Python.org official tutorial",
                "Codecademy Python Course",
                "Real Python",
                "Python for Everybody (Coursera)"
            ],
            "javascript": [
                "MDN Web Docs",
                "freeCodeCamp",
                "JavaScript.info",
                "Eloquent JavaScript"
            ],
            "react": [
                "React Official Documentation",
                "React Tutorial for Beginners",
                "The Complete React Developer Course",
                "React Hooks Course"
            ],
            "aws": [
                "AWS Training and Certification",
                "A Cloud Guru",
                "AWS Solutions Architect Course",
                "AWS Free Tier Hands-on Labs"
            ],
            "machine learning": [
                "Coursera Machine Learning Course",
                "Fast.ai Practical Deep Learning",
                "Kaggle Learn",
                "Machine Learning Yearning"
            ]
        }
    
    async def analyze_skills_gap(
        self,
        job_data: pd.DataFrame,
        user_skills: Optional[List[str]] = None,
        target_roles: Optional[List[str]] = None,
        location: Optional[str] = None
    ) -> SkillGapAnalysis:
        """Perform comprehensive skills gap analysis"""
        
        logger.info("Starting skills gap analysis...")
        
        # Extract and normalize skills from job data
        extracted_skills = await self._extract_skills_from_jobs(job_data)
        
        # Calculate skill metrics
        skill_profiles = await self._calculate_skill_metrics(extracted_skills, job_data)
        
        # Identify skill gaps
        skill_gaps = await self._identify_skill_gaps(skill_profiles, user_skills)
        
        # Analyze trends
        emerging_skills, declining_skills = await self._analyze_skill_trends(extracted_skills, job_data)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(skill_gaps, user_skills, target_roles)
        
        # Create market insights
        market_insights = await self._generate_market_insights(skill_profiles, job_data)
        
        # Categorize skills
        categorized_skills = await self._categorize_skills(skill_profiles)
        
        analysis = SkillGapAnalysis(
            analysis_date=datetime.now(),
            total_jobs_analyzed=len(job_data),
            total_skills_identified=len(skill_profiles),
            top_skills_in_demand=sorted(skill_profiles, key=lambda x: x.demand_score, reverse=True)[:20],
            critical_skill_gaps=sorted(skill_gaps, key=lambda x: x.gap_score, reverse=True)[:10],
            emerging_skills=emerging_skills[:10],
            declining_skills=declining_skills[:10],
            skill_categories=categorized_skills,
            recommendations=recommendations,
            market_insights=market_insights
        )
        
        logger.info(f"Skills gap analysis completed. Analyzed {len(skill_profiles)} skills.")
        return analysis
    
    async def compare_user_profile(
        self,
        user_skills: List[str],
        target_role: str,
        job_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare user's skills against target role requirements"""
        
        # Filter jobs for target role
        role_jobs = job_data[
            job_data['title'].str.contains(target_role, case=False, na=False)
        ]
        
        if role_jobs.empty:
            return {"error": f"No jobs found for role: {target_role}"}
        
        # Extract required skills for target role
        role_skills = await self._extract_skills_from_jobs(role_jobs)
        
        # Normalize user skills
        user_skills_normalized = [skill.lower().strip() for skill in user_skills]
        
        # Calculate matches and gaps
        matching_skills = []
        missing_skills = []
        
        for skill, count in role_skills.items():
            frequency = count / len(role_jobs)
            
            if skill.lower() in user_skills_normalized:
                matching_skills.append({
                    "skill": skill,
                    "frequency": frequency,
                    "job_count": count
                })
            else:
                missing_skills.append({
                    "skill": skill,
                    "frequency": frequency,
                    "job_count": count,
                    "priority": "high" if frequency > 0.5 else "medium" if frequency > 0.2 else "low"
                })
        
        # Calculate compatibility score
        total_skill_weight = sum(skill['frequency'] for skill in matching_skills + missing_skills)
        match_weight = sum(skill['frequency'] for skill in matching_skills)
        compatibility_score = (match_weight / total_skill_weight * 100) if total_skill_weight > 0 else 0
        
        return {
            "target_role": target_role,
            "jobs_analyzed": len(role_jobs),
            "compatibility_score": compatibility_score,
            "matching_skills": sorted(matching_skills, key=lambda x: x['frequency'], reverse=True),
            "missing_skills": sorted(missing_skills, key=lambda x: x['frequency'], reverse=True),
            "recommendations": await self._generate_profile_recommendations(missing_skills[:5]),
            "analysis_date": datetime.now().isoformat()
        }
    
    async def predict_skill_demand(
        self,
        job_data: pd.DataFrame,
        prediction_months: int = 6
    ) -> Dict[str, Any]:
        """Predict future skill demand trends"""
        
        # Extract skills with timestamps
        skills_timeline = await self._extract_skills_timeline(job_data)
        
        predictions = {}
        
        for skill, timeline in skills_timeline.items():
            if len(timeline) < 3:  # Need minimum data points
                continue
            
            # Calculate trend
            trend = await self._calculate_trend(timeline)
            
            # Simple linear projection
            current_demand = timeline[-1]['count'] if timeline else 0
            projected_demand = max(0, current_demand + (trend * prediction_months))
            
            predictions[skill] = {
                "current_demand": current_demand,
                "projected_demand": projected_demand,
                "trend": trend,
                "confidence": min(len(timeline) / 12, 1.0),  # Higher confidence with more data
                "category": await self._get_skill_category(skill)
            }
        
        # Sort by projected growth
        sorted_predictions = dict(
            sorted(predictions.items(), key=lambda x: x[1]['trend'], reverse=True)
        )
        
        return {
            "prediction_period_months": prediction_months,
            "skills_analyzed": len(predictions),
            "predictions": sorted_predictions,
            "top_growing_skills": dict(list(sorted_predictions.items())[:10]),
            "declining_skills": dict(list(sorted(predictions.items(), key=lambda x: x[1]['trend'])[:10])),
            "generated_at": datetime.now().isoformat()
        }
    
    async def generate_learning_roadmap(
        self,
        current_skills: List[str],
        target_skills: List[str],
        timeline_months: int = 12
    ) -> Dict[str, Any]:
        """Generate a personalized learning roadmap"""
        
        # Identify skill gaps
        skill_gaps = [skill for skill in target_skills if skill.lower() not in [s.lower() for s in current_skills]]
        
        # Prioritize skills based on difficulty and dependencies
        prioritized_skills = await self._prioritize_learning_skills(skill_gaps)
        
        # Create timeline
        roadmap = await self._create_learning_timeline(prioritized_skills, timeline_months)
        
        return {
            "current_skills": current_skills,
            "target_skills": target_skills,
            "skill_gaps": skill_gaps,
            "learning_roadmap": roadmap,
            "estimated_completion": timeline_months,
            "total_skills_to_learn": len(skill_gaps),
            "difficulty_breakdown": await self._analyze_learning_difficulty(skill_gaps),
            "recommended_resources": await self._get_learning_resources_for_skills(skill_gaps),
            "created_at": datetime.now().isoformat()
        }
    
    # Private helper methods
    
    async def _extract_skills_from_jobs(self, job_data: pd.DataFrame) -> Dict[str, int]:
        """Extract skills from job descriptions"""
        skills_counter = Counter()
        
        for _, job in job_data.iterrows():
            # Combine title, description, and requirements
            text = " ".join([
                str(job.get('title', '')),
                str(job.get('description', '')),
                str(job.get('requirements', ''))
            ]).lower()
            
            # Extract skills using patterns and categories
            for category, skills in self.skill_categories.items():
                for skill in skills:
                    pattern = self.skill_patterns.get(skill, f"\\\\b{re.escape(skill)}\\\\b")
                    if re.search(pattern, text, re.IGNORECASE):
                        skills_counter[skill] += 1
        
        return dict(skills_counter)
    
    async def _calculate_skill_metrics(
        self,
        extracted_skills: Dict[str, int],
        job_data: pd.DataFrame
    ) -> List[SkillProfile]:
        """Calculate comprehensive metrics for each skill"""
        
        skill_profiles = []
        total_jobs = len(job_data)
        
        for skill, count in extracted_skills.items():
            # Calculate demand score (frequency in jobs)
            demand_score = count / total_jobs
            
            # Estimate supply score (inverse of demand for gap analysis)
            supply_score = 1.0 - min(demand_score, 1.0)
            
            # Calculate gap score
            gap_score = demand_score - supply_score
            
            # Calculate average salary for jobs requiring this skill
            skill_jobs = job_data[
                job_data.apply(
                    lambda row: skill.lower() in str(row.get('description', '')).lower() or 
                               skill.lower() in str(row.get('title', '')).lower(),
                    axis=1
                )
            ]
            
            avg_salary = skill_jobs['salary'].mean() if 'salary' in skill_jobs.columns and not skill_jobs.empty else 0
            
            # Get related skills
            related_skills = await self._find_related_skills(skill, extracted_skills)
            
            # Determine priority level
            priority_level = await self._determine_priority_level(demand_score, gap_score)
            
            # Get category
            category = await self._get_skill_category(skill)
            
            profile = SkillProfile(
                name=skill,
                category=category,
                demand_score=demand_score,
                supply_score=supply_score,
                gap_score=gap_score,
                growth_trend=0.0,  # Will be calculated in trend analysis
                salary_impact=avg_salary,
                job_count=count,
                avg_salary=avg_salary,
                related_skills=related_skills,
                learning_resources=self.learning_resources.get(skill, []),
                priority_level=priority_level
            )
            
            skill_profiles.append(profile)
        
        return skill_profiles
    
    async def _identify_skill_gaps(
        self,
        skill_profiles: List[SkillProfile],
        user_skills: Optional[List[str]] = None
    ) -> List[SkillProfile]:
        """Identify critical skill gaps"""
        
        # If user skills provided, filter for gaps in user's profile
        if user_skills:
            user_skills_lower = [skill.lower() for skill in user_skills]
            gaps = [
                profile for profile in skill_profiles
                if profile.name.lower() not in user_skills_lower and profile.gap_score > 0.1
            ]
        else:
            # General market gaps
            gaps = [profile for profile in skill_profiles if profile.gap_score > 0.2]
        
        return sorted(gaps, key=lambda x: x.gap_score, reverse=True)
    
    async def _analyze_skill_trends(
        self,
        extracted_skills: Dict[str, int],
        job_data: pd.DataFrame
    ) -> Tuple[List[SkillProfile], List[SkillProfile]]:
        """Analyze emerging and declining skills"""
        
        # Simple trend analysis based on recent vs older job postings
        if 'date_posted' not in job_data.columns:
            return [], []
        
        # Split data into recent and older periods
        cutoff_date = datetime.now() - timedelta(days=90)
        recent_jobs = job_data[pd.to_datetime(job_data['date_posted']) >= cutoff_date]
        older_jobs = job_data[pd.to_datetime(job_data['date_posted']) < cutoff_date]
        
        recent_skills = await self._extract_skills_from_jobs(recent_jobs)
        older_skills = await self._extract_skills_from_jobs(older_jobs)
        
        emerging_skills = []
        declining_skills = []
        
        for skill in extracted_skills.keys():
            recent_freq = recent_skills.get(skill, 0) / max(len(recent_jobs), 1)
            older_freq = older_skills.get(skill, 0) / max(len(older_jobs), 1)
            
            trend = recent_freq - older_freq
            
            # Create simplified skill profile for trending skills
            profile = SkillProfile(
                name=skill,
                category=await self._get_skill_category(skill),
                demand_score=recent_freq,
                supply_score=0.5,  # Placeholder
                gap_score=recent_freq - 0.5,
                growth_trend=trend,
                salary_impact=0,
                job_count=recent_skills.get(skill, 0),
                avg_salary=0,
                related_skills=[],
                learning_resources=self.learning_resources.get(skill, []),
                priority_level="medium"
            )
            
            if trend > 0.05:  # Emerging threshold
                emerging_skills.append(profile)
            elif trend < -0.05:  # Declining threshold
                declining_skills.append(profile)
        
        emerging_skills.sort(key=lambda x: x.growth_trend, reverse=True)
        declining_skills.sort(key=lambda x: x.growth_trend)
        
        return emerging_skills, declining_skills
    
    async def _generate_recommendations(
        self,
        skill_gaps: List[SkillProfile],
        user_skills: Optional[List[str]] = None,
        target_roles: Optional[List[str]] = None
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if skill_gaps:
            top_gaps = skill_gaps[:5]
            recommendations.append(
                f"Focus on developing these high-demand skills: {', '.join([s.name for s in top_gaps])}"
            )
            
            for gap in top_gaps[:3]:
                if gap.learning_resources:
                    recommendations.append(
                        f"For {gap.name}: Start with {gap.learning_resources[0]}"
                    )
        
        if user_skills and target_roles:
            recommendations.append(
                f"Based on your current skills, consider roles that leverage: {', '.join(user_skills[:5])}"
            )
        
        # Add general market recommendations
        recommendations.extend([
            "Stay updated with cloud computing trends (AWS, Azure, GCP)",
            "Develop data analysis skills to stay competitive",
            "Consider learning containerization technologies (Docker, Kubernetes)",
            "Strengthen your soft skills, especially communication and leadership"
        ])
        
        return recommendations
    
    async def _generate_market_insights(
        self,
        skill_profiles: List[SkillProfile],
        job_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate market insights from skill analysis"""
        
        insights = {
            "total_unique_skills": len(skill_profiles),
            "most_in_demand_category": None,
            "highest_paying_skills": [],
            "market_saturation": {},
            "skill_diversity_index": 0.0
        }
        
        # Category analysis
        category_demand = defaultdict(float)
        for profile in skill_profiles:
            category_demand[profile.category] += profile.demand_score
        
        if category_demand:
            insights["most_in_demand_category"] = max(category_demand, key=category_demand.get)
        
        # Highest paying skills
        paying_skills = sorted(
            [p for p in skill_profiles if p.avg_salary > 0],
            key=lambda x: x.avg_salary,
            reverse=True
        )[:10]
        
        insights["highest_paying_skills"] = [
            {"skill": p.name, "avg_salary": p.avg_salary} for p in paying_skills
        ]
        
        # Market saturation analysis
        for profile in skill_profiles:
            saturation = "oversaturated" if profile.gap_score < -0.3 else \
                        "balanced" if -0.3 <= profile.gap_score <= 0.3 else \
                        "undersupplied"
            insights["market_saturation"][profile.name] = saturation
        
        # Skill diversity index (Shannon entropy)
        if skill_profiles:
            total_demand = sum(p.demand_score for p in skill_profiles)
            if total_demand > 0:
                probabilities = [p.demand_score / total_demand for p in skill_profiles]
                insights["skill_diversity_index"] = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return insights
    
    async def _categorize_skills(self, skill_profiles: List[SkillProfile]) -> Dict[str, List[SkillProfile]]:
        """Categorize skills by their category"""
        categorized = defaultdict(list)
        
        for profile in skill_profiles:
            categorized[profile.category].append(profile)
        
        # Sort each category by demand score
        for category in categorized:
            categorized[category].sort(key=lambda x: x.demand_score, reverse=True)
        
        return dict(categorized)
    
    async def _get_skill_category(self, skill: str) -> str:
        """Get the category for a skill"""
        skill_lower = skill.lower()
        
        for category, skills in self.skill_categories.items():
            if skill_lower in [s.lower() for s in skills]:
                return category
        
        return "other"
    
    async def _find_related_skills(self, skill: str, all_skills: Dict[str, int]) -> List[str]:
        """Find skills that commonly appear with the given skill"""
        # Simplified implementation - in practice, this would use job-level co-occurrence
        category = await self._get_skill_category(skill)
        related = [s for s in self.skill_categories.get(category, []) if s != skill]
        return related[:5]  # Top 5 related skills
    
    async def _determine_priority_level(self, demand_score: float, gap_score: float) -> str:
        """Determine priority level based on demand and gap scores"""
        if demand_score > 0.5 and gap_score > 0.3:
            return "critical"
        elif demand_score > 0.3 and gap_score > 0.2:
            return "high"
        elif demand_score > 0.1:
            return "medium"
        else:
            return "low"
    
    async def _extract_skills_timeline(self, job_data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Extract skills with timeline data"""
        # Simplified implementation
        return {}
    
    async def _calculate_trend(self, timeline: List[Dict]) -> float:
        """Calculate trend from timeline data"""
        if len(timeline) < 2:
            return 0.0
        
        # Simple linear trend
        counts = [point['count'] for point in timeline]
        return (counts[-1] - counts[0]) / len(counts)
    
    async def _prioritize_learning_skills(self, skills: List[str]) -> List[Dict[str, Any]]:
        """Prioritize skills for learning based on difficulty and dependencies"""
        prioritized = []
        
        skill_difficulty = {
            "html": 1, "css": 1, "javascript": 2, "python": 2, "sql": 2,
            "react": 3, "aws": 3, "machine learning": 4, "kubernetes": 4
        }
        
        for skill in skills:
            difficulty = skill_difficulty.get(skill.lower(), 3)
            prioritized.append({
                "skill": skill,
                "difficulty": difficulty,
                "estimated_weeks": difficulty * 2,
                "prerequisites": []  # Would be populated with actual prerequisites
            })
        
        return sorted(prioritized, key=lambda x: x['difficulty'])
    
    async def _create_learning_timeline(
        self,
        prioritized_skills: List[Dict[str, Any]],
        timeline_months: int
    ) -> List[Dict[str, Any]]:
        """Create a learning timeline"""
        timeline = []
        current_week = 0
        weeks_available = timeline_months * 4
        
        for skill_info in prioritized_skills:
            if current_week + skill_info['estimated_weeks'] <= weeks_available:
                timeline.append({
                    "skill": skill_info['skill'],
                    "start_week": current_week + 1,
                    "end_week": current_week + skill_info['estimated_weeks'],
                    "difficulty": skill_info['difficulty'],
                    "resources": self.learning_resources.get(skill_info['skill'], [])
                })
                current_week += skill_info['estimated_weeks']
            else:
                break
        
        return timeline
    
    async def _analyze_learning_difficulty(self, skills: List[str]) -> Dict[str, int]:
        """Analyze difficulty breakdown of skills to learn"""
        difficulty_count = {"beginner": 0, "intermediate": 0, "advanced": 0}
        
        skill_levels = {
            "html": "beginner", "css": "beginner", "javascript": "intermediate",
            "python": "intermediate", "react": "intermediate", "aws": "advanced",
            "machine learning": "advanced", "kubernetes": "advanced"
        }
        
        for skill in skills:
            level = skill_levels.get(skill.lower(), "intermediate")
            difficulty_count[level] += 1
        
        return difficulty_count
    
    async def _get_learning_resources_for_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Get learning resources for a list of skills"""
        resources = {}
        for skill in skills:
            resources[skill] = self.learning_resources.get(skill, [
                f"Online tutorials for {skill}",
                f"Official {skill} documentation",
                f"{skill} beginner course"
            ])
        return resources
    
    async def _generate_profile_recommendations(self, missing_skills: List[Dict]) -> List[str]:
        """Generate recommendations based on missing skills"""
        recommendations = []
        
        high_priority = [s for s in missing_skills if s['priority'] == 'high']
        if high_priority:
            recommendations.append(
                f"Immediately focus on learning: {', '.join([s['skill'] for s in high_priority[:3]])}"
            )
        
        medium_priority = [s for s in missing_skills if s['priority'] == 'medium']
        if medium_priority:
            recommendations.append(
                f"Consider developing: {', '.join([s['skill'] for s in medium_priority[:3]])}"
            )
        
        recommendations.append("Take online courses or bootcamps to accelerate learning")
        recommendations.append("Build projects to demonstrate your new skills")
        
        return recommendations
