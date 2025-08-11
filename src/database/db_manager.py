"""
Database manager for job market data storage and retrieval.
Uses SQLite for local data persistence.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages SQLite database operations for job market data.
    """
    
    def __init__(self, db_path: str = 'data/jobs.db'):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    company TEXT,
                    location TEXT,
                    description TEXT,
                    url TEXT,
                    source TEXT,
                    posting_date TEXT,
                    min_salary REAL,
                    max_salary REAL,
                    salary_type TEXT,
                    skills TEXT,  -- JSON array
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(title, company, location, url)
                )
            ''')
            
            # Create skills table for normalized storage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    skill_name TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (id),
                    UNIQUE(job_id, skill_name)
                )
            ''')
            
            # Create search_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    location TEXT,
                    source TEXT,
                    jobs_found INTEGER,
                    search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(skill_name)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    def save_jobs(self, jobs_data: List[Dict]) -> int:
        """
        Save job data to database.
        
        Args:
            jobs_data: List of job dictionaries
            
        Returns:
            Number of jobs saved
        """
        if not jobs_data:
            return 0
            
        saved_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for job in jobs_data:
                try:
                    # Prepare job data
                    skills_json = json.dumps(job.get('skills', []))
                    
                    # Insert job (ignore duplicates)
                    cursor.execute('''
                        INSERT OR IGNORE INTO jobs 
                        (title, company, location, description, url, source, posting_date,
                         min_salary, max_salary, salary_type, skills)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        job.get('title', ''),
                        job.get('company', ''),
                        job.get('location', ''),
                        job.get('description', ''),
                        job.get('url', ''),
                        job.get('source', ''),
                        job.get('posting_date', ''),
                        job.get('min_salary'),
                        job.get('max_salary'),
                        job.get('salary_type'),
                        skills_json
                    ))
                    
                    if cursor.rowcount > 0:
                        job_id = cursor.lastrowid
                        saved_count += 1
                        
                        # Save skills separately
                        skills = job.get('skills', [])
                        for skill in skills:
                            cursor.execute('''
                                INSERT OR IGNORE INTO skills (job_id, skill_name)
                                VALUES (?, ?)
                            ''', (job_id, skill))
                            
                except Exception as e:
                    logger.error(f"Error saving job: {e}")
                    continue
                    
            conn.commit()
            
        logger.info(f"Saved {saved_count} new jobs to database")
        return saved_count
        
    def get_jobs(self, job_title: str = None, location: str = None, 
                 company: str = None, source: str = None, 
                 limit: int = None) -> List[Dict]:
        """
        Retrieve jobs from database with optional filters.
        
        Args:
            job_title: Filter by job title
            location: Filter by location
            company: Filter by company
            source: Filter by source
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries
        """
        query = 'SELECT * FROM jobs WHERE 1=1'
        params = []
        
        if job_title:
            query += ' AND title LIKE ?'
            params.append(f'%{job_title}%')
            
        if location:
            query += ' AND location LIKE ?'
            params.append(f'%{location}%')
            
        if company:
            query += ' AND company LIKE ?'
            params.append(f'%{company}%')
            
        if source:
            query += ' AND source = ?'
            params.append(source)
            
        query += ' ORDER BY scraped_at DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
            
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            jobs = []
            for row in cursor.fetchall():
                job = dict(row)
                # Parse skills JSON
                job['skills'] = json.loads(job.get('skills', '[]'))
                jobs.append(job)
                
        return jobs
        
    def get_job_count(self) -> int:
        """Get total number of jobs in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM jobs')
            return cursor.fetchone()[0]
            
    def get_unique_companies(self) -> List[str]:
        """Get list of unique companies in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT company FROM jobs WHERE company != "" ORDER BY company')
            return [row[0] for row in cursor.fetchall()]
            
    def get_unique_locations(self) -> List[str]:
        """Get list of unique locations in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT location FROM jobs WHERE location != "" ORDER BY location')
            return [row[0] for row in cursor.fetchall()]
            
    def get_skill_statistics(self) -> List[Dict]:
        """
        Get skill demand statistics.
        
        Returns:
            List of skills with job counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT skill_name, COUNT(*) as job_count
                FROM skills
                GROUP BY skill_name
                ORDER BY job_count DESC
                LIMIT 50
            ''')
            
            return [{'skill': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
    def get_salary_statistics(self) -> Dict:
        """
        Get salary statistics from database.
        
        Returns:
            Dictionary with salary statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as jobs_with_salary,
                    AVG((min_salary + max_salary) / 2) as avg_salary,
                    MIN(min_salary) as min_salary,
                    MAX(max_salary) as max_salary
                FROM jobs 
                WHERE min_salary IS NOT NULL
            ''')
            
            row = cursor.fetchone()
            return {
                'jobs_with_salary': row[0],
                'average_salary': row[1],
                'minimum_salary': row[2],
                'maximum_salary': row[3]
            }
            
    def save_search_history(self, query: str, location: str, source: str, jobs_found: int):
        """
        Save search history for analytics.
        
        Args:
            query: Search query used
            location: Location searched
            source: Job source used
            jobs_found: Number of jobs found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_history (query, location, source, jobs_found)
                VALUES (?, ?, ?, ?)
            ''', (query, location, source, jobs_found))
            conn.commit()
            
    def get_database_stats(self) -> Dict:
        """
        Get comprehensive database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) FROM jobs')
            total_jobs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT company) FROM jobs WHERE company != ""')
            unique_companies = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT location) FROM jobs WHERE location != ""')
            unique_locations = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM jobs WHERE min_salary IS NOT NULL')
            jobs_with_salary = cursor.fetchone()[0]
            
            # Source distribution
            cursor.execute('SELECT source, COUNT(*) FROM jobs GROUP BY source')
            source_distribution = dict(cursor.fetchall())
            
            # Recent activity
            cursor.execute('SELECT COUNT(*) FROM jobs WHERE scraped_at >= date("now", "-7 days")')
            jobs_last_week = cursor.fetchone()[0]
            
            return {
                'total_jobs': total_jobs,
                'unique_companies': unique_companies,
                'unique_locations': unique_locations,
                'jobs_with_salary': jobs_with_salary,
                'salary_coverage': f"{jobs_with_salary}/{total_jobs}" if total_jobs > 0 else "0/0",
                'source_distribution': source_distribution,
                'jobs_last_week': jobs_last_week,
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024)
            }
            
    def cleanup_old_data(self, days_old: int = 30):
        """
        Remove job data older than specified days.
        
        Args:
            days_old: Number of days after which to remove data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Remove old jobs and their associated skills
            cursor.execute('''
                DELETE FROM skills WHERE job_id IN (
                    SELECT id FROM jobs WHERE scraped_at < date("now", "-{} days")
                )
            '''.format(days_old))
            
            cursor.execute('DELETE FROM jobs WHERE scraped_at < date("now", "-{} days")'.format(days_old))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old job records")
            return deleted_count
            
    def export_to_csv(self, output_path: str, filters: Dict = None):
        """
        Export job data to CSV file.
        
        Args:
            output_path: Path to output CSV file
            filters: Optional filters to apply
        """
        jobs = self.get_jobs(**(filters or {}))
        
        if not jobs:
            logger.warning("No data to export")
            return
            
        import pandas as pd
        df = pd.DataFrame(jobs)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(jobs)} jobs to {output_path}")
        
    def backup_database(self, backup_path: str):
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup file
        """
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        
    def close(self):
        """Close database connection."""
        # SQLite connections are automatically closed when context exits
        pass
