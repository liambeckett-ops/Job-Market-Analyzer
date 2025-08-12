"""
User Authentication and Profile Management System
Handles user registration, login, and profile data for personalized experiences
"""

import hashlib
import sqlite3
import jwt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: int
    username: str
    email: str
    full_name: str
    target_roles: List[str]
    target_industries: List[str]
    current_skills: List[str]
    experience_level: str
    preferred_locations: List[str]
    salary_expectations: Dict[str, float]
    created_at: datetime
    last_login: datetime
    preferences: Dict[str, Any]

@dataclass
class AnalysisHistory:
    """Analysis history record"""
    id: int
    user_id: int
    analysis_type: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    created_at: datetime
    is_favorite: bool

class AuthenticationManager:
    """Manages user authentication and profiles"""
    
    def __init__(self, db_path: str = 'data/users.db', secret_key: str = 'your-secret-key'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.secret_key = secret_key
        self._init_database()
    
    def _init_database(self):
        """Initialize user database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    target_roles TEXT,  -- JSON array
                    target_industries TEXT,  -- JSON array
                    current_skills TEXT,  -- JSON array
                    experience_level TEXT,
                    preferred_locations TEXT,  -- JSON array
                    salary_expectations TEXT,  -- JSON object
                    preferences TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Analysis history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    analysis_type TEXT NOT NULL,
                    parameters TEXT,  -- JSON object
                    results TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_favorite BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User preferences/cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    cache_key TEXT,
                    cache_data TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(user_id, cache_key)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_user ON analysis_history(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_user ON user_cache(user_id)')
            
            conn.commit()
            logger.info("User database initialized successfully")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self, user_id: int, username: str) -> str:
        """Generate JWT token for user session"""
        payload = {
            'user_id': user_id,
            'username': username,
            'exp': datetime.utcnow() + timedelta(days=7)  # Token expires in 7 days
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str = "",
        **profile_data
    ) -> Dict[str, Any]:
        """
        Register a new user
        
        Args:
            username: Unique username
            email: User email
            password: User password
            full_name: User's full name
            **profile_data: Additional profile information
            
        Returns:
            Registration result with user info or error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if username or email already exists
                cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
                if cursor.fetchone():
                    return {'success': False, 'error': 'Username or email already exists'}
                
                # Hash password
                password_hash = self._hash_password(password)
                
                # Prepare profile data
                target_roles = json.dumps(profile_data.get('target_roles', []))
                target_industries = json.dumps(profile_data.get('target_industries', []))
                current_skills = json.dumps(profile_data.get('current_skills', []))
                preferred_locations = json.dumps(profile_data.get('preferred_locations', []))
                salary_expectations = json.dumps(profile_data.get('salary_expectations', {}))
                preferences = json.dumps(profile_data.get('preferences', {}))
                
                # Insert user
                cursor.execute('''
                    INSERT INTO users 
                    (username, email, password_hash, full_name, target_roles, 
                     target_industries, current_skills, experience_level, 
                     preferred_locations, salary_expectations, preferences)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    username, email, password_hash, full_name, target_roles,
                    target_industries, current_skills, profile_data.get('experience_level', 'mid'),
                    preferred_locations, salary_expectations, preferences
                ))
                
                user_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"User {username} registered successfully")
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'username': username,
                    'message': 'User registered successfully'
                }
                
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return {'success': False, 'error': 'Registration failed'}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user and return session token
        
        Args:
            username: Username or email
            password: User password
            
        Returns:
            Authentication result with token or error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Find user by username or email
                cursor.execute('''
                    SELECT * FROM users 
                    WHERE (username = ? OR email = ?) AND is_active = 1
                ''', (username, username))
                
                user = cursor.fetchone()
                if not user:
                    return {'success': False, 'error': 'User not found'}
                
                # Verify password
                password_hash = self._hash_password(password)
                if user['password_hash'] != password_hash:
                    return {'success': False, 'error': 'Invalid password'}
                
                # Generate session token
                token = self._generate_token(user['id'], user['username'])
                
                # Update last login
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
                
                # Save session
                cursor.execute('''
                    INSERT INTO user_sessions (user_id, session_token, expires_at)
                    VALUES (?, ?, ?)
                ''', (user['id'], token, datetime.utcnow() + timedelta(days=7)))
                
                conn.commit()
                
                logger.info(f"User {username} authenticated successfully")
                
                return {
                    'success': True,
                    'token': token,
                    'user_id': user['id'],
                    'username': user['username'],
                    'full_name': user['full_name']
                }
                
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {'success': False, 'error': 'Authentication failed'}
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token and return user info
        
        Args:
            token: JWT token
            
        Returns:
            User info if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if session is still active
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM user_sessions 
                    WHERE session_token = ? AND is_active = 1 AND expires_at > CURRENT_TIMESTAMP
                ''', (token,))
                
                if cursor.fetchone():
                    return {
                        'user_id': payload['user_id'],
                        'username': payload['username']
                    }
            
            return None
            
        except jwt.ExpiredSignatureError:
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """
        Get complete user profile
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile object or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM users WHERE id = ? AND is_active = 1', (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    return None
                
                return UserProfile(
                    user_id=user['id'],
                    username=user['username'],
                    email=user['email'],
                    full_name=user['full_name'] or '',
                    target_roles=json.loads(user['target_roles'] or '[]'),
                    target_industries=json.loads(user['target_industries'] or '[]'),
                    current_skills=json.loads(user['current_skills'] or '[]'),
                    experience_level=user['experience_level'] or 'mid',
                    preferred_locations=json.loads(user['preferred_locations'] or '[]'),
                    salary_expectations=json.loads(user['salary_expectations'] or '{}'),
                    created_at=datetime.fromisoformat(user['created_at']),
                    last_login=datetime.fromisoformat(user['last_login']) if user['last_login'] else datetime.now(),
                    preferences=json.loads(user['preferences'] or '{}')
                )
                
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def update_user_profile(self, user_id: int, **updates) -> bool:
        """
        Update user profile
        
        Args:
            user_id: User ID
            **updates: Fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query dynamically
                update_fields = []
                update_values = []
                
                for field, value in updates.items():
                    if field in ['target_roles', 'target_industries', 'current_skills', 
                                'preferred_locations', 'salary_expectations', 'preferences']:
                        update_fields.append(f"{field} = ?")
                        update_values.append(json.dumps(value))
                    elif field in ['full_name', 'experience_level']:
                        update_fields.append(f"{field} = ?")
                        update_values.append(value)
                
                if not update_fields:
                    return False
                
                update_values.append(user_id)
                query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
                
                cursor.execute(query, update_values)
                conn.commit()
                
                logger.info(f"Updated profile for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False
    
    def save_analysis_history(
        self,
        user_id: int,
        analysis_type: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any]
    ) -> int:
        """
        Save analysis to user's history
        
        Args:
            user_id: User ID
            analysis_type: Type of analysis performed
            parameters: Analysis parameters
            results: Analysis results
            
        Returns:
            Analysis history ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO analysis_history (user_id, analysis_type, parameters, results)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, analysis_type, json.dumps(parameters), json.dumps(results)))
                
                analysis_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Saved analysis history for user {user_id}")
                return analysis_id
                
        except Exception as e:
            logger.error(f"Error saving analysis history: {e}")
            return 0
    
    def get_analysis_history(
        self,
        user_id: int,
        analysis_type: Optional[str] = None,
        limit: int = 50
    ) -> List[AnalysisHistory]:
        """
        Get user's analysis history
        
        Args:
            user_id: User ID
            analysis_type: Filter by analysis type (optional)
            limit: Maximum number of records
            
        Returns:
            List of AnalysisHistory objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = 'SELECT * FROM analysis_history WHERE user_id = ?'
                params = [user_id]
                
                if analysis_type:
                    query += ' AND analysis_type = ?'
                    params.append(analysis_type)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                
                history = []
                for row in cursor.fetchall():
                    history.append(AnalysisHistory(
                        id=row['id'],
                        user_id=row['user_id'],
                        analysis_type=row['analysis_type'],
                        parameters=json.loads(row['parameters']),
                        results=json.loads(row['results']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        is_favorite=bool(row['is_favorite'])
                    ))
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []
    
    def logout_user(self, token: str) -> bool:
        """
        Logout user by deactivating session
        
        Args:
            token: Session token
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE session_token = ?', (token,))
                conn.commit()
                return True
        except:
            return False


class CacheManager:
    """Manages user-specific caching for improved performance"""
    
    def __init__(self, db_path: str = 'data/users.db'):
        self.db_path = Path(db_path)
    
    def set_cache(
        self,
        user_id: int,
        cache_key: str,
        data: Any,
        expires_minutes: int = 60
    ) -> bool:
        """
        Set cache data for user
        
        Args:
            user_id: User ID
            cache_key: Cache key
            data: Data to cache
            expires_minutes: Cache expiration in minutes
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_cache (user_id, cache_key, cache_data, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, cache_key, json.dumps(data), expires_at))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def get_cache(self, user_id: int, cache_key: str) -> Optional[Any]:
        """
        Get cached data for user
        
        Args:
            user_id: User ID
            cache_key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT cache_data FROM user_cache 
                    WHERE user_id = ? AND cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                ''', (user_id, cache_key))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    def clear_cache(self, user_id: int, cache_key: Optional[str] = None) -> bool:
        """
        Clear cache for user
        
        Args:
            user_id: User ID
            cache_key: Specific cache key (optional, clears all if None)
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if cache_key:
                    cursor.execute('DELETE FROM user_cache WHERE user_id = ? AND cache_key = ?', (user_id, cache_key))
                else:
                    cursor.execute('DELETE FROM user_cache WHERE user_id = ?', (user_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def cleanup_expired_cache(self) -> int:
        """
        Remove expired cache entries
        
        Returns:
            Number of entries removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM user_cache WHERE expires_at <= CURRENT_TIMESTAMP')
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
