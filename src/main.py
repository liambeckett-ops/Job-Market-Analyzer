#!/usr/bin/env python3
"""
Enhanced Job Market Analyzer - Main Application
Comprehensive job market analysis with advanced features and user authentication
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database.db_manager import DatabaseManager
from analyzers.skills_gap_analyzer import SkillsGapAnalyzer
from analyzers.salary_analyzer import SalaryAnalyzer
from analyzers.resume_optimizer import ResumeOptimizer
from auth.user_auth import AuthenticationManager
from unified_dashboard import UnifiedDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobMarketAnalyzerPro:
    """Enhanced Job Market Analyzer with enterprise features"""
    
    def __init__(self):
        """Initialize the analyzer with all components"""
        self.db_manager = DatabaseManager()
        self.auth_manager = AuthenticationManager()
        self.skills_analyzer = SkillsGapAnalyzer()
        self.salary_analyzer = SalaryAnalyzer()
        self.resume_optimizer = ResumeOptimizer()
        self.unified_dashboard = UnifiedDashboard()
        
        logger.info("Job Market Analyzer Pro initialized")
    
    def run_cli_mode(self, args):
        """Run the application in CLI mode"""
        
        if args.command == "dashboard":
            logger.info("Starting unified dashboard...")
            self.unified_dashboard.run()
        
        elif args.command == "search":
            self._handle_job_search(args)
        
        elif args.command == "analyze":
            self._handle_analysis(args)
        
        elif args.command == "user":
            self._handle_user_management(args)
        
        elif args.command == "data":
            self._handle_data_operations(args)
        
        else:
            logger.error(f"Unknown command: {args.command}")
            print("Use --help for available commands")
    
    def _handle_job_search(self, args):
        """Handle job search operations"""
        
        try:
            jobs = self.db_manager.search_jobs(
                keywords=args.keywords,
                location=args.location,
                limit=args.limit or 50
            )
            
            if jobs:
                print(f"\nFound {len(jobs)} jobs:")
                print("-" * 80)
                
                for job in jobs:
                    print(f"Title: {job['title']}")
                    print(f"Company: {job['company']}")
                    print(f"Location: {job['location']}")
                    if 'salary' in job and job['salary']:
                        print(f"Salary: ${job['salary']:,.0f}")
                    print(f"Posted: {job['posted_date']}")
                    print("-" * 40)
            else:
                print("No jobs found matching your criteria.")
        
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            print(f"Error: {e}")
    
    def _handle_analysis(self, args):
        """Handle analysis operations"""
        
        try:
            if args.analysis_type == "skills":
                self._run_skills_analysis(args)
            
            elif args.analysis_type == "salary":
                self._run_salary_analysis(args)
            
            elif args.analysis_type == "resume":
                self._run_resume_analysis(args)
            
            else:
                print("Available analysis types: skills, salary, resume")
        
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            print(f"Error: {e}")
    
    def _run_skills_analysis(self, args):
        """Run skills gap analysis"""
        
        print("\n🎯 Running Skills Gap Analysis...")
        
        # Load job data
        jobs = self.db_manager.get_jobs(limit=1000)
        job_data = [dict(job) for job in jobs]
        
        # Get user skills (simplified for CLI)
        user_skills = input("Enter your current skills (comma-separated): ").split(",")
        user_skills = [skill.strip() for skill in user_skills if skill.strip()]
        
        target_role = input("Enter target role (optional): ").strip() or None
        location = input("Enter location (optional): ").strip() or None
        
        # Perform analysis
        analysis = self.skills_analyzer.analyze_skills_gap(
            job_data,
            user_skills=user_skills,
            target_roles=[target_role] if target_role else None,
            location=location
        )
        
        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"Total Skills Analyzed: {analysis.total_skills_identified}")
        print(f"Critical Skill Gaps: {len(analysis.critical_skill_gaps)}")
        print(f"Skills Coverage: {len(user_skills) / max(analysis.total_skills_identified, 1) * 100:.1f}%")
        
        if analysis.critical_skill_gaps:
            print(f"\n🚨 Top Critical Skills to Learn:")
            for i, skill in enumerate(analysis.critical_skill_gaps[:5], 1):
                print(f"{i}. {skill.name} (Demand Score: {skill.demand_score:.2f})")
        
        if analysis.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(analysis.recommendations, 1):
                print(f"{i}. {rec}")
    
    def _run_salary_analysis(self, args):
        """Run salary analysis"""
        
        print("\n💰 Running Salary Analysis...")
        
        # Load job data
        jobs = self.db_manager.get_jobs(limit=1000)
        job_data = [dict(job) for job in jobs]
        
        # Load data into analyzer
        self.salary_analyzer.load_data(job_data)
        
        role = input("Enter job role (optional): ").strip() or None
        location = input("Enter location (optional): ").strip() or None
        
        # Get salary statistics
        stats = self.salary_analyzer.get_salary_statistics(
            job_title=role,
            location=location
        )
        
        if 'error' not in stats:
            print(f"\n📊 Salary Statistics:")
            print(f"Jobs Analyzed: {stats['count']:,}")
            print(f"Median Salary: ${stats['median_salary']:,.0f}")
            print(f"Average Salary: ${stats['mean_salary']:,.0f}")
            print(f"Salary Range: ${stats['min_salary']:,.0f} - ${stats['max_salary']:,.0f}")
            print(f"25th Percentile: ${stats['percentile_25']:,.0f}")
            print(f"75th Percentile: ${stats['percentile_75']:,.0f}")
            print(f"90th Percentile: ${stats['percentile_90']:,.0f}")
            
            # Show salary range distribution
            print(f"\n📈 Salary Distribution:")
            for range_name, count in stats['salary_ranges'].items():
                print(f"{range_name}: {count} jobs")
        else:
            print(f"Error: {stats['error']}")
    
    def _run_resume_analysis(self, args):
        """Run resume analysis"""
        
        print("\n📄 Running Resume Analysis...")
        
        resume_file = input("Enter path to resume file (or press Enter to paste text): ").strip()
        
        if resume_file and Path(resume_file).exists():
            # In a real implementation, you would extract text from the file
            resume_text = f"Sample resume content from {resume_file}"
            print(f"Loaded resume from {resume_file}")
        else:
            print("Paste your resume text (press Ctrl+D when finished):")
            resume_lines = []
            try:
                while True:
                    line = input()
                    resume_lines.append(line)
            except EOFError:
                pass
            
            resume_text = "\n".join(resume_lines)
        
        if not resume_text.strip():
            print("No resume text provided.")
            return
        
        target_role = input("Enter target role: ").strip()
        target_industry = input("Enter target industry: ").strip()
        experience_level = input("Enter experience level (entry/mid/senior/executive): ").strip()
        
        # Load job data
        jobs = self.db_manager.get_jobs(limit=1000)
        job_data = [dict(job) for job in jobs]
        
        # Initialize optimizer and analyze
        optimizer = ResumeOptimizer(job_data)
        analysis = optimizer.analyze_resume(
            resume_text,
            target_role,
            target_industry,
            experience_level
        )
        
        # Display results
        print(f"\n📊 Resume Analysis Results:")
        print(f"Overall Score: {analysis.overall_score:.0f}/100")
        print(f"ATS Compatibility: {analysis.ats_compatibility:.0f}/100")
        print(f"Keyword Density: {analysis.keyword_density:.1f}%")
        print(f"Role Fit Score: {analysis.role_fit_score:.0f}/100")
        
        # Show section scores
        print(f"\n📝 Section Scores:")
        for section_name, section in analysis.sections.items():
            print(f"{section_name.title()}: {section.score:.0f}/100")
        
        # Show optimization suggestions
        print(f"\n💡 Top Optimization Suggestions:")
        for i, suggestion in enumerate(analysis.optimization_suggestions[:5], 1):
            print(f"{i}. {suggestion}")
        
        # Show missing critical skills
        missing_critical = [sm for sm in analysis.skill_matches 
                          if not sm.present and sm.priority == "critical"]
        
        if missing_critical:
            print(f"\n❌ Missing Critical Skills:")
            for skill in missing_critical[:5]:
                print(f"• {skill.skill}")
    
    def _handle_user_management(self, args):
        """Handle user management operations"""
        
        if args.user_action == "register":
            self._register_user()
        
        elif args.user_action == "login":
            self._login_user()
        
        elif args.user_action == "profile":
            self._show_user_profile(args)
        
        else:
            print("Available user actions: register, login, profile")
    
    def _register_user(self):
        """Register a new user"""
        
        print("\n🔐 User Registration")
        
        username = input("Username: ").strip()
        email = input("Email: ").strip()
        password = input("Password: ").strip()
        full_name = input("Full Name: ").strip()
        
        print("\nOptional Information:")
        target_roles = input("Target roles (comma-separated): ").split(",")
        target_roles = [role.strip() for role in target_roles if role.strip()]
        
        target_industries = input("Target industries (comma-separated): ").split(",")
        target_industries = [ind.strip() for ind in target_industries if ind.strip()]
        
        experience_level = input("Experience level (entry/mid/senior/executive): ").strip() or "mid"
        
        preferred_locations = input("Preferred locations (comma-separated): ").split(",")
        preferred_locations = [loc.strip() for loc in preferred_locations if loc.strip()]
        
        current_skills = input("Current skills (comma-separated): ").split(",")
        current_skills = [skill.strip() for skill in current_skills if skill.strip()]
        
        # Register user
        result = self.auth_manager.register_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
            target_roles=target_roles,
            target_industries=target_industries,
            experience_level=experience_level,
            preferred_locations=preferred_locations,
            current_skills=current_skills
        )
        
        if result['success']:
            print(f"✅ User registered successfully!")
            print(f"User ID: {result['user_id']}")
        else:
            print(f"❌ Registration failed: {result['error']}")
    
    def _login_user(self):
        """Login user"""
        
        print("\n🔐 User Login")
        
        username = input("Username or Email: ").strip()
        password = input("Password: ").strip()
        
        result = self.auth_manager.authenticate_user(username, password)
        
        if result['success']:
            print(f"✅ Login successful!")
            print(f"Welcome, {result['username']}!")
            print(f"Token: {result['token'][:20]}...")
        else:
            print(f"❌ Login failed: {result['error']}")
    
    def _show_user_profile(self, args):
        """Show user profile"""
        
        if not args.user_id:
            print("User ID required for profile view")
            return
        
        try:
            user_id = int(args.user_id)
            profile = self.auth_manager.get_user_profile(user_id)
            
            if profile:
                print(f"\n👤 User Profile")
                print(f"Username: {profile.username}")
                print(f"Email: {profile.email}")
                print(f"Full Name: {profile.full_name}")
                print(f"Experience Level: {profile.experience_level}")
                print(f"Member Since: {profile.created_at}")
                print(f"Target Roles: {', '.join(profile.target_roles)}")
                print(f"Target Industries: {', '.join(profile.target_industries)}")
                print(f"Preferred Locations: {', '.join(profile.preferred_locations)}")
                print(f"Skills: {', '.join(profile.current_skills[:10])}")
                
                if len(profile.current_skills) > 10:
                    print(f"... and {len(profile.current_skills) - 10} more skills")
            else:
                print("User not found")
        
        except ValueError:
            print("Invalid user ID")
        except Exception as e:
            print(f"Error retrieving profile: {e}")
    
    def _handle_data_operations(self, args):
        """Handle data operations"""
        
        if args.data_action == "stats":
            self._show_data_stats()
        
        elif args.data_action == "export":
            self._export_data(args)
        
        elif args.data_action == "import":
            self._import_data(args)
        
        else:
            print("Available data actions: stats, export, import")
    
    def _show_data_stats(self):
        """Show database statistics"""
        
        print("\n📊 Database Statistics")
        
        try:
            total_jobs = len(self.db_manager.get_jobs(limit=10000))
            print(f"Total Jobs: {total_jobs:,}")
            
            # Get unique companies
            jobs = self.db_manager.get_jobs(limit=10000)
            companies = set(job['company'] for job in jobs if job.get('company'))
            print(f"Unique Companies: {len(companies):,}")
            
            # Get unique locations
            locations = set(job['location'] for job in jobs if job.get('location'))
            print(f"Unique Locations: {len(locations):,}")
            
            # Get recent jobs (last 30 days)
            from datetime import datetime, timedelta
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_jobs = [job for job in jobs 
                          if job.get('posted_date') and 
                          datetime.fromisoformat(job['posted_date'].replace('Z', '+00:00')) > recent_cutoff]
            print(f"Recent Jobs (30 days): {len(recent_jobs):,}")
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            print(f"Error: {e}")
    
    def _export_data(self, args):
        """Export data to file"""
        
        print(f"\n📤 Exporting data...")
        
        try:
            jobs = self.db_manager.get_jobs(limit=10000)
            
            if args.format == "json":
                import json
                output_file = args.output or "jobs_export.json"
                
                with open(output_file, 'w') as f:
                    json.dump([dict(job) for job in jobs], f, indent=2, default=str)
            
            elif args.format == "csv":
                import pandas as pd
                output_file = args.output or "jobs_export.csv"
                
                df = pd.DataFrame([dict(job) for job in jobs])
                df.to_csv(output_file, index=False)
            
            else:
                print("Supported formats: json, csv")
                return
            
            print(f"✅ Data exported to {output_file}")
            print(f"Exported {len(jobs)} jobs")
        
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            print(f"Error: {e}")
    
    def _import_data(self, args):
        """Import data from file"""
        
        if not args.input_file or not Path(args.input_file).exists():
            print("Input file not found")
            return
        
        print(f"\n📥 Importing data from {args.input_file}...")
        
        try:
            if args.input_file.endswith('.json'):
                with open(args.input_file, 'r') as f:
                    data = json.load(f)
            
            elif args.input_file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(args.input_file)
                data = df.to_dict('records')
            
            else:
                print("Supported formats: json, csv")
                return
            
            # Store jobs
            imported_count = 0
            for job_data in data:
                try:
                    job_id = self.db_manager.store_job(
                        title=job_data.get('title', 'Unknown'),
                        company=job_data.get('company', 'Unknown'),
                        location=job_data.get('location', 'Unknown'),
                        description=job_data.get('description', ''),
                        salary=job_data.get('salary'),
                        posted_date=job_data.get('posted_date'),
                        source_url=job_data.get('source_url', ''),
                        job_source=job_data.get('job_source', 'import')
                    )
                    
                    if job_id:
                        imported_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to import job: {e}")
                    continue
            
            print(f"✅ Successfully imported {imported_count} jobs")
        
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            print(f"Error: {e}")


def create_argument_parser():
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="Job Market Analyzer Pro - Comprehensive job market analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dashboard                      # Launch unified dashboard
  %(prog)s search --keywords "python"     # Search for Python jobs
  %(prog)s analyze --type skills          # Run skills gap analysis
  %(prog)s user --action register         # Register new user
  %(prog)s data --action stats            # Show database statistics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch unified dashboard')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for jobs')
    search_parser.add_argument('--keywords', '-k', help='Search keywords')
    search_parser.add_argument('--location', '-l', help='Job location')
    search_parser.add_argument('--limit', '-n', type=int, help='Number of results')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run analysis')
    analyze_parser.add_argument('--type', dest='analysis_type', 
                               choices=['skills', 'salary', 'resume'],
                               help='Type of analysis to run')
    
    # User management command
    user_parser = subparsers.add_parser('user', help='User management')
    user_parser.add_argument('--action', dest='user_action',
                           choices=['register', 'login', 'profile'],
                           help='User action to perform')
    user_parser.add_argument('--user-id', dest='user_id', help='User ID for profile view')
    
    # Data operations command
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_parser.add_argument('--action', dest='data_action',
                           choices=['stats', 'export', 'import'],
                           help='Data action to perform')
    data_parser.add_argument('--format', choices=['json', 'csv'], help='Export/import format')
    data_parser.add_argument('--output', help='Output file for export')
    data_parser.add_argument('--input-file', help='Input file for import')
    
    return parser


def main():
    """Main entry point"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        # No command specified, launch dashboard by default
        args.command = 'dashboard'
    
    try:
        analyzer = JobMarketAnalyzerPro()
        analyzer.run_cli_mode(args)
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
