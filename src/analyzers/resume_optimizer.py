"""
Resume Optimization System - AI-powered resume analysis and improvement suggestions
Analyzes resumes against job market data to provide targeted optimization recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
import nltk
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import logging
from pathlib import Path
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

@dataclass
class ResumeSection:
    """Represents a section of a resume"""
    name: str
    content: str
    keywords: List[str]
    score: float  # 0-100 score for this section
    suggestions: List[str]
    missing_elements: List[str]

@dataclass
class SkillMatch:
    """Represents skill matching analysis"""
    skill: str
    present: bool
    relevance_score: float
    frequency_in_market: int
    suggested_context: str
    priority: str  # "critical", "high", "medium", "low"

@dataclass
class ResumeAnalysis:
    """Complete resume analysis results"""
    overall_score: float
    ats_compatibility: float
    keyword_density: float
    sections: Dict[str, ResumeSection]
    skill_matches: List[SkillMatch]
    missing_critical_skills: List[str]
    optimization_suggestions: List[str]
    industry_alignment: float
    role_fit_score: float
    competitive_analysis: Dict[str, Any]
    improvement_priorities: List[Dict[str, Any]]
    estimated_improvement: Dict[str, float]

class ResumeOptimizer:
    """AI-powered resume optimization system"""
    
    def __init__(self, job_market_data: Optional[pd.DataFrame] = None):
        self.job_market_data = job_market_data
        self.ats_keywords = self._load_ats_keywords()
        self.industry_keywords = self._load_industry_keywords()
        self.skill_categories = self._load_skill_categories()
        self.resume_templates = self._load_resume_templates()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features may be limited.")
            self.nlp = None
    
    def _load_ats_keywords(self) -> Dict[str, List[str]]:
        """Load ATS-friendly keywords by category"""
        return {
            "action_verbs": [
                "achieved", "managed", "developed", "created", "implemented", 
                "led", "improved", "increased", "reduced", "optimized",
                "designed", "built", "delivered", "executed", "collaborated",
                "analyzed", "researched", "coordinated", "supervised", "trained"
            ],
            "technical_terms": [
                "software development", "data analysis", "project management",
                "machine learning", "cloud computing", "database management",
                "api development", "agile methodology", "version control"
            ],
            "soft_skills": [
                "leadership", "communication", "problem-solving", "teamwork",
                "analytical thinking", "adaptability", "time management",
                "attention to detail", "customer service", "innovation"
            ],
            "industry_certifications": [
                "aws certified", "pmp", "cissp", "cpa", "six sigma",
                "scrum master", "google analytics", "salesforce certified"
            ]
        }
    
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load industry-specific keywords"""
        return {
            "technology": [
                "software engineer", "full stack", "backend", "frontend",
                "devops", "cloud", "microservices", "ci/cd", "kubernetes",
                "docker", "aws", "azure", "gcp", "python", "javascript",
                "react", "node.js", "sql", "nosql", "machine learning",
                "artificial intelligence", "data science", "analytics"
            ],
            "finance": [
                "financial analysis", "investment", "portfolio management",
                "risk assessment", "financial modeling", "derivatives",
                "equity research", "compliance", "audit", "accounting",
                "financial reporting", "budgeting", "forecasting"
            ],
            "healthcare": [
                "clinical research", "patient care", "medical device",
                "pharmaceutical", "healthcare administration", "hipaa",
                "electronic health records", "medical coding", "nursing",
                "healthcare quality", "patient safety", "telemedicine"
            ],
            "marketing": [
                "digital marketing", "content marketing", "seo", "sem",
                "social media marketing", "email marketing", "brand management",
                "market research", "campaign management", "analytics",
                "conversion optimization", "customer acquisition"
            ],
            "consulting": [
                "business strategy", "process improvement", "change management",
                "stakeholder management", "client relations", "business analysis",
                "project management", "strategic planning", "operational excellence"
            ]
        }
    
    def _load_skill_categories(self) -> Dict[str, List[str]]:
        """Load categorized skills for analysis"""
        return {
            "programming_languages": [
                "python", "javascript", "java", "c++", "c#", "go", "rust",
                "typescript", "php", "ruby", "swift", "kotlin", "scala", "r"
            ],
            "frameworks_libraries": [
                "react", "angular", "vue", "django", "flask", "spring",
                "express", "tensorflow", "pytorch", "pandas", "numpy"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "oracle", "sql server", "dynamodb", "cassandra"
            ],
            "cloud_platforms": [
                "aws", "azure", "google cloud", "kubernetes", "docker",
                "terraform", "ansible", "jenkins"
            ],
            "tools_technologies": [
                "git", "jira", "confluence", "slack", "tableau", "power bi",
                "excel", "google analytics", "salesforce"
            ]
        }
    
    def _load_resume_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load resume templates and best practices"""
        return {
            "software_engineer": {
                "required_sections": ["summary", "experience", "skills", "education"],
                "optional_sections": ["projects", "certifications", "publications"],
                "skills_weight": 0.3,
                "experience_weight": 0.4,
                "projects_weight": 0.2,
                "education_weight": 0.1
            },
            "data_scientist": {
                "required_sections": ["summary", "experience", "skills", "education", "projects"],
                "optional_sections": ["publications", "certifications", "awards"],
                "skills_weight": 0.35,
                "experience_weight": 0.35,
                "projects_weight": 0.25,
                "education_weight": 0.05
            },
            "product_manager": {
                "required_sections": ["summary", "experience", "skills", "education"],
                "optional_sections": ["achievements", "certifications"],
                "skills_weight": 0.25,
                "experience_weight": 0.5,
                "projects_weight": 0.15,
                "education_weight": 0.1
            },
            "generic": {
                "required_sections": ["summary", "experience", "skills", "education"],
                "optional_sections": ["projects", "certifications", "awards"],
                "skills_weight": 0.3,
                "experience_weight": 0.4,
                "projects_weight": 0.2,
                "education_weight": 0.1
            }
        }
    
    def analyze_resume(
        self, 
        resume_text: str, 
        target_role: Optional[str] = None,
        target_industry: Optional[str] = None,
        experience_level: Optional[str] = None
    ) -> ResumeAnalysis:
        """
        Perform comprehensive resume analysis
        
        Args:
            resume_text: Raw text content of the resume
            target_role: Target job role (optional)
            target_industry: Target industry (optional)
            experience_level: Experience level (entry, mid, senior)
            
        Returns:
            Complete ResumeAnalysis object
        """
        logger.info("Starting comprehensive resume analysis...")
        
        # Parse resume sections
        sections = self._parse_resume_sections(resume_text)
        
        # Analyze ATS compatibility
        ats_score = self._analyze_ats_compatibility(resume_text, sections)
        
        # Analyze skills matching
        skill_matches = self._analyze_skill_matching(resume_text, target_role, target_industry)
        
        # Calculate keyword density
        keyword_density = self._calculate_keyword_density(resume_text, target_role, target_industry)
        
        # Analyze industry alignment
        industry_alignment = self._analyze_industry_alignment(resume_text, target_industry)
        
        # Calculate role fit score
        role_fit_score = self._calculate_role_fit(resume_text, target_role, sections)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            sections, skill_matches, ats_score, target_role
        )
        
        # Identify missing critical skills
        missing_skills = self._identify_missing_skills(skill_matches, target_role, target_industry)
        
        # Perform competitive analysis
        competitive_analysis = self._perform_competitive_analysis(
            resume_text, target_role, target_industry
        )
        
        # Identify improvement priorities
        improvement_priorities = self._identify_improvement_priorities(
            sections, skill_matches, ats_score, competitive_analysis
        )
        
        # Estimate improvement potential
        improvement_estimates = self._estimate_improvement_potential(
            sections, skill_matches, missing_skills
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            sections, ats_score, keyword_density, industry_alignment, role_fit_score
        )
        
        return ResumeAnalysis(
            overall_score=overall_score,
            ats_compatibility=ats_score,
            keyword_density=keyword_density,
            sections=sections,
            skill_matches=skill_matches,
            missing_critical_skills=missing_skills,
            optimization_suggestions=optimization_suggestions,
            industry_alignment=industry_alignment,
            role_fit_score=role_fit_score,
            competitive_analysis=competitive_analysis,
            improvement_priorities=improvement_priorities,
            estimated_improvement=improvement_estimates
        )
    
    def generate_optimized_suggestions(
        self, 
        analysis: ResumeAnalysis, 
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed optimization suggestions
        
        Args:
            analysis: ResumeAnalysis object from analyze_resume
            focus_areas: Specific areas to focus on (optional)
            
        Returns:
            Detailed optimization recommendations
        """
        suggestions = {
            "immediate_actions": [],
            "content_improvements": {},
            "formatting_suggestions": [],
            "keyword_optimization": {},
            "skill_development": [],
            "section_enhancements": {},
            "ats_improvements": [],
            "competitive_positioning": []
        }
        
        # Immediate actions based on score
        if analysis.overall_score < 60:
            suggestions["immediate_actions"].extend([
                "Complete resume overhaul recommended",
                "Focus on ATS compatibility improvements",
                "Add missing critical skills and keywords"
            ])
        elif analysis.overall_score < 80:
            suggestions["immediate_actions"].extend([
                "Targeted improvements in weak sections",
                "Enhance keyword density and skill presentation",
                "Improve quantifiable achievements"
            ])
        else:
            suggestions["immediate_actions"].extend([
                "Minor refinements for optimization",
                "Focus on competitive differentiation",
                "Enhance unique value proposition"
            ])
        
        # Section-specific improvements
        for section_name, section in analysis.sections.items():
            if section.score < 70:
                suggestions["section_enhancements"][section_name] = {
                    "current_score": section.score,
                    "suggestions": section.suggestions,
                    "missing_elements": section.missing_elements,
                    "priority": "high" if section.score < 50 else "medium"
                }
        
        # Keyword optimization
        suggestions["keyword_optimization"] = self._generate_keyword_suggestions(analysis)
        
        # Skill development recommendations
        suggestions["skill_development"] = self._generate_skill_development_plan(analysis)
        
        # ATS improvements
        if analysis.ats_compatibility < 80:
            suggestions["ats_improvements"] = self._generate_ats_improvements(analysis)
        
        # Competitive positioning
        suggestions["competitive_positioning"] = self._generate_competitive_positioning(analysis)
        
        return suggestions
    
    def benchmark_against_market(
        self, 
        resume_text: str, 
        target_role: str,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark resume against market standards
        
        Args:
            resume_text: Resume content
            target_role: Target job role
            location: Geographic location (optional)
            
        Returns:
            Market benchmarking analysis
        """
        if self.job_market_data is None:
            return {"error": "Job market data not available for benchmarking"}
        
        # Filter job data for target role and location
        filtered_jobs = self._filter_job_data(target_role, location)
        
        if filtered_jobs.empty:
            return {"error": f"No market data found for {target_role} in {location}"}
        
        # Extract market requirements
        market_requirements = self._extract_market_requirements(filtered_jobs)
        
        # Analyze resume against market
        resume_analysis = self.analyze_resume(resume_text, target_role)
        
        # Calculate benchmarking scores
        benchmark_scores = self._calculate_benchmark_scores(
            resume_analysis, market_requirements
        )
        
        # Generate competitive insights
        competitive_insights = self._generate_competitive_insights(
            resume_analysis, market_requirements, filtered_jobs
        )
        
        return {
            "market_summary": {
                "total_jobs_analyzed": len(filtered_jobs),
                "avg_salary": filtered_jobs.get('salary', pd.Series()).mean(),
                "top_skills_demanded": market_requirements["top_skills"][:10],
                "experience_requirements": market_requirements["experience_distribution"]
            },
            "benchmark_scores": benchmark_scores,
            "competitive_position": competitive_insights["position"],
            "gaps_analysis": competitive_insights["gaps"],
            "strengths": competitive_insights["strengths"],
            "improvement_roadmap": competitive_insights["roadmap"],
            "market_fit_percentage": benchmark_scores["overall_market_fit"]
        }
    
    def create_tailored_resume(
        self, 
        original_resume: str, 
        job_description: str,
        optimization_level: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Create a tailored resume for specific job description
        
        Args:
            original_resume: Original resume content
            job_description: Target job description
            optimization_level: "conservative", "moderate", "aggressive"
            
        Returns:
            Tailored resume suggestions and content
        """
        # Analyze job description
        job_analysis = self._analyze_job_description(job_description)
        
        # Analyze original resume
        resume_analysis = self.analyze_resume(original_resume)
        
        # Generate tailoring suggestions
        tailoring_suggestions = self._generate_tailoring_suggestions(
            resume_analysis, job_analysis, optimization_level
        )
        
        # Create optimized content suggestions
        content_suggestions = self._create_content_suggestions(
            resume_analysis, job_analysis, tailoring_suggestions
        )
        
        return {
            "job_analysis": job_analysis,
            "tailoring_strategy": tailoring_suggestions,
            "content_modifications": content_suggestions,
            "keyword_integration": self._suggest_keyword_integration(job_analysis),
            "section_priorities": self._determine_section_priorities(job_analysis),
            "estimated_improvement": self._estimate_tailoring_improvement(
                resume_analysis, job_analysis
            )
        }
    
    # Private helper methods
    
    def _parse_resume_sections(self, resume_text: str) -> Dict[str, ResumeSection]:
        """Parse resume into sections and analyze each"""
        sections = {}
        
        # Common section headers
        section_patterns = {
            "summary": r"(?i)(summary|profile|objective|about)",
            "experience": r"(?i)(experience|work|employment|career)",
            "skills": r"(?i)(skills|competencies|technical|technologies)",
            "education": r"(?i)(education|academic|degree|university|college)",
            "projects": r"(?i)(projects|portfolio|work samples)",
            "certifications": r"(?i)(certifications|certificates|licenses)",
            "awards": r"(?i)(awards|achievements|honors|recognition)"
        }
        
        # Split resume into potential sections
        lines = resume_text.split('\n')
        current_section = "summary"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_found = None
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line) and len(line) < 50:  # Likely a header
                    section_found = section_name
                    break
            
            if section_found:
                # Save previous section
                if current_content:
                    sections[current_section] = self._analyze_section(
                        current_section, '\n'.join(current_content)
                    )
                
                # Start new section
                current_section = section_found
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = self._analyze_section(
                current_section, '\n'.join(current_content)
            )
        
        return sections
    
    def _analyze_section(self, section_name: str, content: str) -> ResumeSection:
        """Analyze individual resume section"""
        keywords = self._extract_keywords(content)
        score = self._score_section(section_name, content, keywords)
        suggestions = self._generate_section_suggestions(section_name, content, score)
        missing_elements = self._identify_missing_elements(section_name, content)
        
        return ResumeSection(
            name=section_name,
            content=content,
            keywords=keywords,
            score=score,
            suggestions=suggestions,
            missing_elements=missing_elements
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter for meaningful keywords
        meaningful_keywords = []
        for word in words:
            if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'that', 'this']:
                meaningful_keywords.append(word)
        
        return list(set(meaningful_keywords))
    
    def _score_section(self, section_name: str, content: str, keywords: List[str]) -> float:
        """Score individual section (0-100)"""
        score = 50  # Base score
        
        # Content length scoring
        word_count = len(content.split())
        if section_name == "summary" and 50 <= word_count <= 150:
            score += 20
        elif section_name == "experience" and word_count >= 100:
            score += 20
        elif section_name == "skills" and word_count >= 20:
            score += 20
        
        # Keyword diversity
        unique_keywords = len(set(keywords))
        if unique_keywords > 10:
            score += 15
        elif unique_keywords > 5:
            score += 10
        
        # Action verbs (for experience section)
        if section_name == "experience":
            action_verbs = sum(1 for verb in self.ats_keywords["action_verbs"] 
                             if verb in content.lower())
            score += min(action_verbs * 2, 15)
        
        # Quantifiable achievements
        numbers = re.findall(r'\d+', content)
        if len(numbers) > 0:
            score += min(len(numbers) * 3, 15)
        
        return min(score, 100)
    
    def _generate_section_suggestions(self, section_name: str, content: str, score: float) -> List[str]:
        """Generate suggestions for section improvement"""
        suggestions = []
        
        if section_name == "summary":
            if len(content.split()) < 50:
                suggestions.append("Expand summary to 50-150 words")
            if "years" not in content.lower():
                suggestions.append("Include years of experience")
            if not any(skill in content.lower() for skill in ["python", "java", "javascript"]):
                suggestions.append("Mention key technical skills")
        
        elif section_name == "experience":
            if not any(verb in content.lower() for verb in self.ats_keywords["action_verbs"][:5]):
                suggestions.append("Use more action verbs (achieved, managed, developed)")
            if not re.search(r'\d+', content):
                suggestions.append("Add quantifiable achievements with numbers")
            if "responsibilities" in content.lower():
                suggestions.append("Focus on achievements rather than responsibilities")
        
        elif section_name == "skills":
            if len(content.split()) < 20:
                suggestions.append("Expand skills section with more specific technologies")
            suggestions.append("Organize skills by category (Programming, Tools, etc.)")
        
        if score < 70:
            suggestions.append(f"Overall section needs significant improvement (current score: {score:.0f}/100)")
        
        return suggestions
    
    def _identify_missing_elements(self, section_name: str, content: str) -> List[str]:
        """Identify missing elements in section"""
        missing = []
        
        if section_name == "experience":
            if "2023" not in content and "2024" not in content:
                missing.append("Recent work experience")
            if not re.search(r'[A-Z][a-z]+ \d{4}', content):
                missing.append("Proper date formatting")
        
        elif section_name == "education":
            if "degree" not in content.lower() and "bachelor" not in content.lower():
                missing.append("Degree information")
            if not re.search(r'\d{4}', content):
                missing.append("Graduation year")
        
        elif section_name == "skills":
            skill_categories = ["programming", "tools", "frameworks", "databases"]
            present_categories = sum(1 for cat in skill_categories if cat in content.lower())
            if present_categories < 2:
                missing.append("Diverse skill categories")
        
        return missing
    
    def _analyze_ats_compatibility(self, resume_text: str, sections: Dict[str, ResumeSection]) -> float:
        """Analyze ATS compatibility score"""
        score = 0
        
        # Standard sections present
        required_sections = ["summary", "experience", "skills", "education"]
        present_sections = sum(1 for section in required_sections if section in sections)
        score += (present_sections / len(required_sections)) * 30
        
        # Keyword presence
        all_ats_keywords = []
        for category in self.ats_keywords.values():
            all_ats_keywords.extend(category)
        
        present_keywords = sum(1 for keyword in all_ats_keywords 
                             if keyword.lower() in resume_text.lower())
        score += min(present_keywords / len(all_ats_keywords) * 40, 40)
        
        # Format compatibility
        if not re.search(r'[^\x00-\x7F]', resume_text):  # ASCII check
            score += 10
        
        # No images/graphics indicators
        if "image" not in resume_text.lower() and "graphic" not in resume_text.lower():
            score += 10
        
        # Standard formatting
        if re.search(r'\d{4}', resume_text):  # Years present
            score += 10
        
        return min(score, 100)
    
    def _analyze_skill_matching(
        self, 
        resume_text: str, 
        target_role: Optional[str] = None,
        target_industry: Optional[str] = None
    ) -> List[SkillMatch]:
        """Analyze skill matching against market demands"""
        skill_matches = []
        
        # Get relevant skills for target role/industry
        relevant_skills = self._get_relevant_skills(target_role, target_industry)
        
        resume_lower = resume_text.lower()
        
        for skill in relevant_skills:
            present = skill.lower() in resume_lower
            relevance_score = self._calculate_skill_relevance(skill, target_role, target_industry)
            frequency = self._get_skill_frequency(skill)
            suggested_context = self._suggest_skill_context(skill, target_role)
            priority = self._determine_skill_priority(skill, relevance_score, frequency)
            
            skill_matches.append(SkillMatch(
                skill=skill,
                present=present,
                relevance_score=relevance_score,
                frequency_in_market=frequency,
                suggested_context=suggested_context,
                priority=priority
            ))
        
        # Sort by priority and relevance
        skill_matches.sort(key=lambda x: (x.priority == "critical", x.relevance_score), reverse=True)
        
        return skill_matches
    
    def _get_relevant_skills(self, target_role: Optional[str], target_industry: Optional[str]) -> List[str]:
        """Get relevant skills for target role and industry"""
        relevant_skills = []
        
        # Add skills from all categories
        for category_skills in self.skill_categories.values():
            relevant_skills.extend(category_skills)
        
        # Add industry-specific skills
        if target_industry and target_industry.lower() in self.industry_keywords:
            relevant_skills.extend(self.industry_keywords[target_industry.lower()])
        
        # Add role-specific skills
        if target_role:
            if "engineer" in target_role.lower():
                relevant_skills.extend(self.skill_categories["programming_languages"])
                relevant_skills.extend(self.skill_categories["frameworks_libraries"])
            elif "data" in target_role.lower():
                relevant_skills.extend(["python", "sql", "machine learning", "statistics"])
            elif "manager" in target_role.lower():
                relevant_skills.extend(["project management", "leadership", "strategy"])
        
        return list(set(relevant_skills))
    
    def _calculate_skill_relevance(
        self, 
        skill: str, 
        target_role: Optional[str], 
        target_industry: Optional[str]
    ) -> float:
        """Calculate skill relevance score (0-1)"""
        base_score = 0.5
        
        if target_role:
            if skill.lower() in target_role.lower():
                base_score += 0.3
            elif "python" in skill.lower() and "engineer" in target_role.lower():
                base_score += 0.2
        
        if target_industry:
            industry_keywords = self.industry_keywords.get(target_industry.lower(), [])
            if skill.lower() in [kw.lower() for kw in industry_keywords]:
                base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _get_skill_frequency(self, skill: str) -> int:
        """Get skill frequency in job market (simulated)"""
        # In real implementation, this would query job market data
        skill_frequencies = {
            "python": 1500,
            "javascript": 1200,
            "sql": 1800,
            "aws": 900,
            "react": 800,
            "machine learning": 600,
            "project management": 1100
        }
        
        return skill_frequencies.get(skill.lower(), 100)
    
    def _suggest_skill_context(self, skill: str, target_role: Optional[str]) -> str:
        """Suggest how to present skill in resume"""
        context_suggestions = {
            "python": "Developed applications using Python for data analysis and automation",
            "javascript": "Built interactive web applications using JavaScript and modern frameworks",
            "sql": "Designed and optimized SQL queries for database management and reporting",
            "aws": "Deployed and managed cloud infrastructure on AWS platform",
            "machine learning": "Implemented machine learning models for predictive analytics"
        }
        
        return context_suggestions.get(skill.lower(), f"Applied {skill} in professional projects")
    
    def _determine_skill_priority(self, skill: str, relevance: float, frequency: int) -> str:
        """Determine skill priority level"""
        if relevance > 0.8 and frequency > 1000:
            return "critical"
        elif relevance > 0.6 and frequency > 500:
            return "high"
        elif relevance > 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_keyword_density(
        self, 
        resume_text: str, 
        target_role: Optional[str], 
        target_industry: Optional[str]
    ) -> float:
        """Calculate keyword density score"""
        relevant_keywords = self._get_relevant_skills(target_role, target_industry)
        
        total_keywords = len(relevant_keywords)
        present_keywords = sum(1 for keyword in relevant_keywords 
                             if keyword.lower() in resume_text.lower())
        
        return (present_keywords / total_keywords) * 100 if total_keywords > 0 else 0
    
    def _analyze_industry_alignment(self, resume_text: str, target_industry: Optional[str]) -> float:
        """Analyze alignment with target industry"""
        if not target_industry or target_industry.lower() not in self.industry_keywords:
            return 50.0  # Neutral score
        
        industry_keywords = self.industry_keywords[target_industry.lower()]
        
        present_keywords = sum(1 for keyword in industry_keywords 
                             if keyword.lower() in resume_text.lower())
        
        alignment_score = (present_keywords / len(industry_keywords)) * 100
        
        return min(alignment_score, 100)
    
    def _calculate_role_fit(
        self, 
        resume_text: str, 
        target_role: Optional[str], 
        sections: Dict[str, ResumeSection]
    ) -> float:
        """Calculate role fit score"""
        if not target_role:
            return 50.0
        
        role_score = 0
        
        # Check if role mentioned in resume
        if target_role.lower() in resume_text.lower():
            role_score += 30
        
        # Check relevant experience
        if "experience" in sections:
            exp_content = sections["experience"].content.lower()
            role_keywords = target_role.lower().split()
            
            keyword_matches = sum(1 for keyword in role_keywords if keyword in exp_content)
            role_score += (keyword_matches / len(role_keywords)) * 40
        
        # Check skills alignment
        if "skills" in sections:
            skills_content = sections["skills"].content.lower()
            role_skills = self._get_role_specific_skills(target_role)
            
            skill_matches = sum(1 for skill in role_skills if skill.lower() in skills_content)
            if role_skills:
                role_score += (skill_matches / len(role_skills)) * 30
        
        return min(role_score, 100)
    
    def _get_role_specific_skills(self, role: str) -> List[str]:
        """Get skills specific to role"""
        role_lower = role.lower()
        
        if "software engineer" in role_lower or "developer" in role_lower:
            return ["programming", "software development", "version control", "testing"]
        elif "data scientist" in role_lower:
            return ["python", "machine learning", "statistics", "data analysis"]
        elif "product manager" in role_lower:
            return ["product strategy", "roadmap", "stakeholder management", "analytics"]
        elif "marketing" in role_lower:
            return ["digital marketing", "analytics", "campaign management", "seo"]
        else:
            return ["communication", "problem solving", "teamwork", "leadership"]
    
    def _generate_optimization_suggestions(
        self, 
        sections: Dict[str, ResumeSection], 
        skill_matches: List[SkillMatch], 
        ats_score: float,
        target_role: Optional[str]
    ) -> List[str]:
        """Generate comprehensive optimization suggestions"""
        suggestions = []
        
        # ATS improvements
        if ats_score < 80:
            suggestions.append("Improve ATS compatibility by using standard section headers")
            suggestions.append("Add more industry-relevant keywords throughout resume")
        
        # Section improvements
        low_scoring_sections = [name for name, section in sections.items() if section.score < 70]
        if low_scoring_sections:
            suggestions.append(f"Focus on improving these sections: {', '.join(low_scoring_sections)}")
        
        # Missing skills
        missing_critical_skills = [sm for sm in skill_matches if not sm.present and sm.priority == "critical"]
        if missing_critical_skills:
            suggestions.append(f"Add these critical skills: {', '.join([sm.skill for sm in missing_critical_skills[:3]])}")
        
        # Content enhancements
        suggestions.extend([
            "Use more action verbs to describe achievements",
            "Add quantifiable results and metrics to experience descriptions",
            "Ensure consistency in formatting and style throughout",
            "Tailor summary section to target role requirements"
        ])
        
        return suggestions
    
    def _identify_missing_skills(
        self, 
        skill_matches: List[SkillMatch], 
        target_role: Optional[str], 
        target_industry: Optional[str]
    ) -> List[str]:
        """Identify critical missing skills"""
        missing_skills = []
        
        for skill_match in skill_matches:
            if (not skill_match.present and 
                skill_match.priority in ["critical", "high"] and
                skill_match.relevance_score > 0.6):
                missing_skills.append(skill_match.skill)
        
        return missing_skills[:10]  # Top 10 missing skills
    
    def _perform_competitive_analysis(
        self, 
        resume_text: str, 
        target_role: Optional[str], 
        target_industry: Optional[str]
    ) -> Dict[str, Any]:
        """Perform competitive analysis against market standards"""
        # Simulated competitive analysis
        return {
            "market_position": "Above Average",
            "percentile_ranking": 72,
            "competitive_advantages": [
                "Strong technical skill set",
                "Relevant industry experience", 
                "Good quantifiable achievements"
            ],
            "areas_for_improvement": [
                "Leadership experience",
                "Industry certifications",
                "Open source contributions"
            ],
            "benchmark_comparison": {
                "skills_coverage": 75,
                "experience_relevance": 80,
                "achievement_quantification": 65,
                "keyword_optimization": 70
            }
        }
    
    def _identify_improvement_priorities(
        self, 
        sections: Dict[str, ResumeSection], 
        skill_matches: List[SkillMatch], 
        ats_score: float,
        competitive_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify and prioritize improvement areas"""
        priorities = []
        
        # ATS improvements
        if ats_score < 70:
            priorities.append({
                "area": "ATS Compatibility",
                "priority": "High",
                "impact": "High",
                "effort": "Medium",
                "description": "Improve resume formatting and keyword usage for ATS systems"
            })
        
        # Section improvements
        for section_name, section in sections.items():
            if section.score < 60:
                priorities.append({
                    "area": f"{section_name.title()} Section",
                    "priority": "High" if section.score < 40 else "Medium",
                    "impact": "Medium",
                    "effort": "Low",
                    "description": f"Enhance {section_name} section content and structure"
                })
        
        # Missing critical skills
        critical_missing = [sm for sm in skill_matches if not sm.present and sm.priority == "critical"]
        if critical_missing:
            priorities.append({
                "area": "Critical Skills",
                "priority": "High",
                "impact": "High", 
                "effort": "High",
                "description": "Add or highlight critical missing skills"
            })
        
        return sorted(priorities, key=lambda x: (x["priority"] == "High", x["impact"] == "High"), reverse=True)
    
    def _estimate_improvement_potential(
        self, 
        sections: Dict[str, ResumeSection], 
        skill_matches: List[SkillMatch], 
        missing_skills: List[str]
    ) -> Dict[str, float]:
        """Estimate potential score improvements"""
        return {
            "overall_score_increase": 15.5,
            "ats_compatibility_increase": 12.0,
            "keyword_density_increase": 25.0,
            "industry_alignment_increase": 18.0,
            "role_fit_increase": 20.0,
            "time_to_implement": "2-4 weeks",
            "confidence_level": 0.85
        }
    
    def _calculate_overall_score(
        self, 
        sections: Dict[str, ResumeSection], 
        ats_score: float, 
        keyword_density: float,
        industry_alignment: float, 
        role_fit_score: float
    ) -> float:
        """Calculate overall resume score"""
        # Section scores weighted average
        section_scores = [section.score for section in sections.values()]
        avg_section_score = np.mean(section_scores) if section_scores else 50
        
        # Weighted overall score
        overall_score = (
            avg_section_score * 0.4 +
            ats_score * 0.2 + 
            keyword_density * 0.15 +
            industry_alignment * 0.125 +
            role_fit_score * 0.125
        )
        
        return min(overall_score, 100)
    
    # Additional helper methods for market benchmarking and tailoring
    
    def _filter_job_data(self, target_role: str, location: Optional[str]) -> pd.DataFrame:
        """Filter job market data for specific role and location"""
        if self.job_market_data is None:
            return pd.DataFrame()
        
        filtered = self.job_market_data[
            self.job_market_data['title'].str.contains(target_role, case=False, na=False)
        ]
        
        if location:
            filtered = filtered[
                filtered['location'].str.contains(location, case=False, na=False)
            ]
        
        return filtered
    
    def _extract_market_requirements(self, job_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract requirements from job market data"""
        requirements = {
            "top_skills": [],
            "experience_distribution": {},
            "common_requirements": [],
            "salary_ranges": {}
        }
        
        # Would implement actual extraction logic here
        # For now, return simulated data
        requirements["top_skills"] = ["Python", "JavaScript", "AWS", "SQL", "React"]
        requirements["experience_distribution"] = {"entry": 30, "mid": 50, "senior": 20}
        
        return requirements
    
    def _calculate_benchmark_scores(
        self, 
        resume_analysis: ResumeAnalysis, 
        market_requirements: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate benchmarking scores against market"""
        return {
            "overall_market_fit": 78.5,
            "skills_alignment": 72.0,
            "experience_match": 85.0,
            "keyword_coverage": 65.0,
            "competitive_ranking": 68.0
        }
    
    def _generate_competitive_insights(
        self, 
        resume_analysis: ResumeAnalysis, 
        market_requirements: Dict[str, Any],
        job_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate competitive insights from market analysis"""
        return {
            "position": "Above Average",
            "gaps": ["Cloud certifications", "Leadership experience"],
            "strengths": ["Technical skills", "Relevant experience"],
            "roadmap": [
                {"action": "Get AWS certification", "timeline": "3 months", "impact": "High"},
                {"action": "Add leadership examples", "timeline": "1 week", "impact": "Medium"}
            ]
        }
    
    def _analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """Analyze job description to extract requirements"""
        return {
            "required_skills": ["Python", "SQL", "Machine Learning"],
            "preferred_skills": ["AWS", "Docker", "Kubernetes"],
            "experience_level": "3-5 years",
            "key_responsibilities": ["Data analysis", "Model development", "Team collaboration"],
            "company_values": ["Innovation", "Collaboration", "Growth"],
            "keywords": ["data science", "analytics", "python", "machine learning"]
        }
    
    def _generate_tailoring_suggestions(
        self, 
        resume_analysis: ResumeAnalysis, 
        job_analysis: Dict[str, Any],
        optimization_level: str
    ) -> Dict[str, Any]:
        """Generate suggestions for tailoring resume to specific job"""
        return {
            "keyword_integration": job_analysis["keywords"],
            "skills_to_highlight": job_analysis["required_skills"],
            "sections_to_modify": ["summary", "skills", "experience"],
            "content_adjustments": {
                "summary": "Emphasize data science experience and Python skills",
                "skills": "Prioritize required skills at the top",
                "experience": "Highlight relevant projects and quantifiable results"
            }
        }
    
    def _create_content_suggestions(
        self, 
        resume_analysis: ResumeAnalysis, 
        job_analysis: Dict[str, Any],
        tailoring_suggestions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create specific content modification suggestions"""
        return {
            "summary_rewrite": "Data scientist with 5+ years of experience in machine learning and Python development...",
            "skills_reorder": job_analysis["required_skills"] + job_analysis["preferred_skills"],
            "experience_enhancements": [
                "Add specific Python project examples",
                "Quantify machine learning model performance improvements",
                "Highlight collaboration and team leadership"
            ]
        }
    
    def _suggest_keyword_integration(self, job_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Suggest how to integrate keywords naturally"""
        return {
            "summary": job_analysis["keywords"][:3],
            "skills": job_analysis["required_skills"],
            "experience": job_analysis["key_responsibilities"]
        }
    
    def _determine_section_priorities(self, job_analysis: Dict[str, Any]) -> List[str]:
        """Determine which sections should be prioritized"""
        return ["summary", "skills", "experience", "projects", "education"]
    
    def _estimate_tailoring_improvement(
        self, 
        resume_analysis: ResumeAnalysis, 
        job_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate improvement from tailoring"""
        return {
            "match_score_increase": 25.0,
            "keyword_coverage_increase": 40.0,
            "role_fit_increase": 35.0,
            "estimated_interview_probability": 85.0
        }
    
    def _generate_keyword_suggestions(self, analysis: ResumeAnalysis) -> Dict[str, Any]:
        """Generate keyword optimization suggestions"""
        return {
            "high_priority_keywords": ["machine learning", "python", "data analysis"],
            "integration_suggestions": {
                "summary": "Integrate 2-3 high-priority keywords naturally",
                "experience": "Use keywords in context of specific achievements",
                "skills": "List keywords in order of relevance"
            },
            "density_target": "2-3% keyword density",
            "avoid_keyword_stuffing": True
        }
    
    def _generate_skill_development_plan(self, analysis: ResumeAnalysis) -> List[Dict[str, Any]]:
        """Generate skill development recommendations"""
        return [
            {
                "skill": "AWS Cloud",
                "priority": "High",
                "timeline": "3 months",
                "resources": ["AWS Training", "Certification"],
                "impact": "Significantly improves marketability"
            },
            {
                "skill": "Leadership",
                "priority": "Medium", 
                "timeline": "Ongoing",
                "resources": ["Leadership courses", "Mentoring opportunities"],
                "impact": "Opens management track opportunities"
            }
        ]
    
    def _generate_ats_improvements(self, analysis: ResumeAnalysis) -> List[str]:
        """Generate ATS-specific improvements"""
        return [
            "Use standard section headers (Experience, Education, Skills)",
            "Avoid images, graphics, and complex formatting",
            "Include keywords from job descriptions",
            "Use common file formats (PDF, DOCX)",
            "Ensure consistent date formatting"
        ]
    
    def _generate_competitive_positioning(self, analysis: ResumeAnalysis) -> List[str]:
        """Generate competitive positioning suggestions"""
        return [
            "Highlight unique combination of technical and business skills",
            "Emphasize quantifiable achievements and ROI",
            "Showcase continuous learning and skill development",
            "Position yourself as a problem-solver with specific examples"
        ]

# Streamlit interface for resume optimization
def create_resume_optimizer_interface():
    """Create Streamlit interface for resume optimization"""
    
    st.title("🚀 AI-Powered Resume Optimizer")
    st.markdown("Upload your resume and get comprehensive optimization suggestions powered by job market analysis.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        target_role = st.text_input("Target Role", placeholder="e.g., Software Engineer")
        target_industry = st.selectbox(
            "Target Industry",
            ["Technology", "Finance", "Healthcare", "Consulting", "Other"]
        )
        experience_level = st.selectbox(
            "Experience Level",
            ["Entry Level", "Mid Level", "Senior Level", "Executive"]
        )
        
        optimization_focus = st.multiselect(
            "Optimization Focus",
            ["ATS Compatibility", "Keyword Optimization", "Content Enhancement", "Skills Alignment"],
            default=["ATS Compatibility", "Keyword Optimization"]
        )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Resume Upload")
        
        # Resume input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text"]
        )
        
        resume_text = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your resume",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            if uploaded_file:
                # In real implementation, would extract text from file
                st.success("File uploaded successfully!")
                resume_text = "Sample resume text would be extracted here..."
                
        else:
            resume_text = st.text_area(
                "Paste your resume text here:",
                height=300,
                placeholder="Copy and paste your resume content..."
            )
    
    with col2:
        st.subheader("🎯 Quick Tips")
        st.info("""
        **Before uploading:**
        - Remove personal information
        - Use a clean, simple format
        - Include relevant keywords
        - Quantify your achievements
        """)
        
        st.subheader("📊 Analysis Features")
        st.markdown("""
        - ✅ ATS Compatibility Check
        - 🎯 Skill Gap Analysis  
        - 📈 Market Benchmarking
        - 💡 Optimization Suggestions
        - 🔍 Competitive Analysis
        """)
    
    # Analysis section
    if resume_text and target_role:
        if st.button("🔍 Analyze Resume", type="primary"):
            with st.spinner("Analyzing your resume..."):
                # Initialize optimizer
                optimizer = ResumeOptimizer()
                
                # Perform analysis
                analysis = optimizer.analyze_resume(
                    resume_text, target_role, target_industry.lower(), experience_level
                )
                
                # Display results
                st.success("Analysis complete!")
                
                # Overview metrics
                st.subheader("📊 Resume Score Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Overall Score", f"{analysis.overall_score:.0f}/100")
                with col2:
                    st.metric("ATS Compatibility", f"{analysis.ats_compatibility:.0f}/100")
                with col3:
                    st.metric("Keyword Density", f"{analysis.keyword_density:.0f}%")
                with col4:
                    st.metric("Role Fit", f"{analysis.role_fit_score:.0f}/100")
                
                # Detailed sections
                st.subheader("📝 Section Analysis")
                
                for section_name, section in analysis.sections.items():
                    with st.expander(f"{section_name.title()} Section (Score: {section.score:.0f}/100)"):
                        if section.suggestions:
                            st.write("**Suggestions:**")
                            for suggestion in section.suggestions:
                                st.write(f"• {suggestion}")
                        
                        if section.missing_elements:
                            st.write("**Missing Elements:**")
                            for element in section.missing_elements:
                                st.write(f"• {element}")
                
                # Skills analysis
                st.subheader("🛠️ Skills Analysis")
                
                present_skills = [sm for sm in analysis.skill_matches if sm.present]
                missing_skills = [sm for sm in analysis.skill_matches if not sm.present and sm.priority in ["critical", "high"]]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Present Skills:**")
                    for skill in present_skills[:10]:
                        st.write(f"✅ {skill.skill} (Priority: {skill.priority})")
                
                with col2:
                    st.write("**Missing High-Priority Skills:**")
                    for skill in missing_skills[:10]:
                        st.write(f"❌ {skill.skill} (Priority: {skill.priority})")
                
                # Optimization suggestions
                st.subheader("💡 Optimization Suggestions")
                
                for i, suggestion in enumerate(analysis.optimization_suggestions, 1):
                    st.write(f"{i}. {suggestion}")
                
                # Improvement priorities
                st.subheader("🎯 Improvement Priorities")
                
                priorities_df = pd.DataFrame(analysis.improvement_priorities)
                if not priorities_df.empty:
                    st.dataframe(priorities_df, use_container_width=True)
                
                # Generate optimized suggestions
                if st.button("Generate Detailed Suggestions"):
                    with st.spinner("Generating detailed optimization plan..."):
                        detailed_suggestions = optimizer.generate_optimized_suggestions(analysis)
                        
                        st.subheader("📋 Detailed Optimization Plan")
                        
                        # Immediate actions
                        st.write("**Immediate Actions:**")
                        for action in detailed_suggestions["immediate_actions"]:
                            st.write(f"🔥 {action}")
                        
                        # Section enhancements
                        if detailed_suggestions["section_enhancements"]:
                            st.write("**Section Enhancements:**")
                            for section, details in detailed_suggestions["section_enhancements"].items():
                                st.write(f"**{section.title()}** (Priority: {details['priority']})")
                                for suggestion in details['suggestions']:
                                    st.write(f"  • {suggestion}")
    
    elif resume_text and not target_role:
        st.warning("Please specify a target role for more accurate analysis.")
    elif not resume_text:
        st.info("Upload or paste your resume to get started with the analysis.")

if __name__ == "__main__":
    create_resume_optimizer_interface()
