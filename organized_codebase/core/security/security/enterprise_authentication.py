"""
MetaGPT Derived Enterprise Authentication
Extracted from MetaGPT enterprise patterns and multi-user security
Enhanced for comprehensive enterprise authentication and access management
"""

import logging
import jwt
import hashlib
import secrets
import time
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from .error_handler import SecurityError, AuthenticationError, AuthorizationError, security_error_handler


class UserRole(Enum):
    """Enterprise user roles based on MetaGPT patterns"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"
    READONLY = "readonly"


class AuthenticationMethod(Enum):
    """Authentication methods supported"""
    PASSWORD = "password"
    JWT_TOKEN = "jwt_token"
    API_KEY = "api_key"
    LDAP = "ldap"
    OAUTH2 = "oauth2"
    MFA = "mfa"


class SessionStatus(Enum):
    """User session status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    LOCKED = "locked"


@dataclass
class UserProfile:
    """Enterprise user profile with comprehensive attributes"""
    user_id: str
    username: str
    email: str
    role: UserRole
    organization: str = "default"
    department: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_hash: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        return self.locked_until and datetime.utcnow() < self.locked_until
    
    @property
    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return self.role in [UserRole.SUPER_ADMIN, UserRole.ADMIN]
    
    @property
    def display_name(self) -> str:
        """Get user display name"""
        return f"{self.username} ({self.role.value})"


@dataclass
class EnterpriseSession:
    """Secure enterprise session management"""
    session_id: str
    user_id: str
    user_profile: UserProfile
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    permissions_cache: Set[str] = field(default_factory=set)
    activity_log: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        """Check if session is active"""
        return self.status == SessionStatus.ACTIVE and not self.is_expired
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired"""
        max_session_age = timedelta(hours=8)  # 8 hour default session
        return (datetime.utcnow() - self.last_activity) > max_session_age
    
    @property
    def session_age_minutes(self) -> float:
        """Get session age in minutes"""
        return (datetime.utcnow() - self.created_at).total_seconds() / 60
    
    def update_activity(self, activity_type: str = "access", details: Dict[str, Any] = None):
        """Update session activity"""
        self.last_activity = datetime.utcnow()
        
        activity_record = {
            'timestamp': self.last_activity.isoformat(),
            'type': activity_type,
            'details': details or {}
        }
        
        self.activity_log.append(activity_record)
        
        # Limit activity log size
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[-50:]


class PasswordManager:
    """Secure password management for enterprise users"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_password_length = 12
        self.require_complexity = True
        self.password_history_size = 5
        self.password_histories: Dict[str, List[str]] = {}
    
    def hash_password(self, password: str, salt: str = None) -> str:
        """Hash password with salt"""
        try:
            if not salt:
                salt = secrets.token_hex(32)
            
            # Use PBKDF2 for password hashing
            password_bytes = password.encode('utf-8')
            salt_bytes = salt.encode('utf-8')
            
            # 100,000 iterations for security
            hashed = hashlib.pbkdf2_hmac('sha256', password_bytes, salt_bytes, 100000)
            
            return f"{salt}:{hashed.hex()}"
            
        except Exception as e:
            raise SecurityError(f"Password hashing failed: {str(e)}", "PWD_HASH_001")
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            if ':' not in password_hash:
                return False
            
            salt, stored_hash = password_hash.split(':', 1)
            
            # Hash the provided password with the stored salt
            password_bytes = password.encode('utf-8')
            salt_bytes = salt.encode('utf-8')
            
            computed_hash = hashlib.pbkdf2_hmac('sha256', password_bytes, salt_bytes, 100000)
            
            return secrets.compare_digest(computed_hash.hex(), stored_hash)
            
        except Exception as e:
            self.logger.error(f"Password verification failed: {e}")
            return False
    
    def validate_password_strength(self, password: str, username: str = None) -> Dict[str, Any]:
        """Validate password strength according to enterprise policy"""
        validation_result = {
            'valid': True,
            'issues': [],
            'score': 0,
            'strength': 'weak'
        }
        
        try:
            # Length check
            if len(password) < self.min_password_length:
                validation_result['issues'].append(f"Password must be at least {self.min_password_length} characters")
                validation_result['valid'] = False
            else:
                validation_result['score'] += 20
            
            # Complexity checks if required
            if self.require_complexity:
                has_upper = any(c.isupper() for c in password)
                has_lower = any(c.islower() for c in password)
                has_digit = any(c.isdigit() for c in password)
                has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
                
                if not has_upper:
                    validation_result['issues'].append("Password must contain uppercase letters")
                    validation_result['valid'] = False
                else:
                    validation_result['score'] += 15
                
                if not has_lower:
                    validation_result['issues'].append("Password must contain lowercase letters")
                    validation_result['valid'] = False
                else:
                    validation_result['score'] += 15
                
                if not has_digit:
                    validation_result['issues'].append("Password must contain numbers")
                    validation_result['valid'] = False
                else:
                    validation_result['score'] += 15
                
                if not has_special:
                    validation_result['issues'].append("Password must contain special characters")
                    validation_result['valid'] = False
                else:
                    validation_result['score'] += 15
            
            # Username similarity check
            if username and username.lower() in password.lower():
                validation_result['issues'].append("Password cannot contain username")
                validation_result['valid'] = False
                validation_result['score'] -= 20
            
            # Common password check
            common_passwords = ['password', '123456', 'admin', 'welcome', 'letmein']
            if password.lower() in common_passwords:
                validation_result['issues'].append("Password is too common")
                validation_result['valid'] = False
                validation_result['score'] -= 30
            
            # Determine strength
            if validation_result['score'] >= 80:
                validation_result['strength'] = 'very_strong'
            elif validation_result['score'] >= 60:
                validation_result['strength'] = 'strong'
            elif validation_result['score'] >= 40:
                validation_result['strength'] = 'medium'
            else:
                validation_result['strength'] = 'weak'
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Password validation error: {e}")
            validation_result['valid'] = False
            validation_result['issues'].append("Password validation error")
            return validation_result
    
    def check_password_history(self, user_id: str, new_password: str) -> bool:
        """Check if password was used recently"""
        try:
            if user_id not in self.password_histories:
                return True
            
            new_hash = self.hash_password(new_password)
            
            for old_hash in self.password_histories[user_id]:
                if self.verify_password(new_password, old_hash):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Password history check failed: {e}")
            return True  # Allow on error to prevent lockout
    
    def store_password_history(self, user_id: str, password_hash: str):
        """Store password in history for reuse prevention"""
        try:
            if user_id not in self.password_histories:
                self.password_histories[user_id] = []
            
            self.password_histories[user_id].append(password_hash)
            
            # Limit history size
            if len(self.password_histories[user_id]) > self.password_history_size:
                self.password_histories[user_id] = self.password_histories[user_id][-self.password_history_size:]
                
        except Exception as e:
            self.logger.error(f"Error storing password history: {e}")


class JWTManager:
    """Enterprise JWT token management"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(64)
        self.algorithm = 'HS256'
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=7)
        self.logger = logging.getLogger(__name__)
    
    def generate_access_token(self, user_profile: UserProfile, 
                            permissions: Set[str] = None) -> str:
        """Generate JWT access token"""
        try:
            now = datetime.utcnow()
            
            payload = {
                'user_id': user_profile.user_id,
                'username': user_profile.username,
                'email': user_profile.email,
                'role': user_profile.role.value,
                'organization': user_profile.organization,
                'permissions': list(permissions or user_profile.permissions),
                'iat': now,
                'exp': now + self.access_token_expire,
                'type': 'access'
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
            
        except Exception as e:
            raise SecurityError(f"Token generation failed: {str(e)}", "JWT_GEN_001")
    
    def generate_refresh_token(self, user_profile: UserProfile) -> str:
        """Generate JWT refresh token"""
        try:
            now = datetime.utcnow()
            
            payload = {
                'user_id': user_profile.user_id,
                'username': user_profile.username,
                'iat': now,
                'exp': now + self.refresh_token_expire,
                'type': 'refresh'
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
            
        except Exception as e:
            raise SecurityError(f"Refresh token generation failed: {str(e)}", "JWT_REFRESH_001")
    
    def verify_token(self, token: str, token_type: str = 'access') -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get('type') != token_type:
                raise AuthenticationError("Invalid token type")
            
            # Check expiration
            if datetime.utcnow() > datetime.fromisoformat(payload['exp']):
                raise AuthenticationError("Token expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            raise AuthenticationError(f"Token verification failed: {str(e)}")


class EnterpriseAuthenticationManager:
    """Comprehensive enterprise authentication management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.users: Dict[str, UserProfile] = {}
        self.sessions: Dict[str, EnterpriseSession] = {}
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager()
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self._lock = threading.RLock()
        
        # Initialize built-in admin user
        self._create_default_admin()
    
    def create_user(self, username: str, email: str, password: str,
                   role: UserRole = UserRole.USER,
                   organization: str = "default") -> bool:
        """Create new enterprise user"""
        try:
            with self._lock:
                # Check if user already exists
                if any(user.username == username or user.email == email 
                      for user in self.users.values()):
                    raise ValidationError("User already exists")
                
                # Validate password
                password_validation = self.password_manager.validate_password_strength(password, username)
                if not password_validation['valid']:
                    raise ValidationError(f"Password validation failed: {'; '.join(password_validation['issues'])}")
                
                # Create user profile
                user_id = secrets.token_urlsafe(16)
                password_hash = self.password_manager.hash_password(password)
                
                user_profile = UserProfile(
                    user_id=user_id,
                    username=username,
                    email=email,
                    role=role,
                    organization=organization,
                    password_hash=password_hash
                )
                
                # Set default permissions based on role
                user_profile.permissions = self._get_default_permissions(role)
                
                # Store user
                self.users[user_id] = user_profile
                
                # Store password history
                self.password_manager.store_password_history(user_id, password_hash)
                
                self.logger.info(f"Created user: {username} ({role.value})")
                return True
                
        except Exception as e:
            error = SecurityError(f"Failed to create user: {str(e)}", "USER_CREATE_001")
            security_error_handler.handle_error(error)
            return False
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = None, user_agent: str = None) -> Optional[EnterpriseSession]:
        """Authenticate user and create session"""
        try:
            with self._lock:
                # Find user by username or email
                user_profile = None
                for user in self.users.values():
                    if user.username == username or user.email == username:
                        user_profile = user
                        break
                
                if not user_profile:
                    self._record_failed_attempt(username)
                    raise AuthenticationError("Invalid credentials")
                
                # Check account lockout
                if user_profile.is_locked:
                    raise AuthenticationError("Account is locked")
                
                # Check failed attempts
                if self._is_user_locked_out(username):
                    raise AuthenticationError("Too many failed attempts")
                
                # Verify password
                if not self.password_manager.verify_password(password, user_profile.password_hash):
                    self._record_failed_attempt(username)
                    user_profile.failed_attempts += 1
                    
                    # Lock account after max attempts
                    if user_profile.failed_attempts >= self.max_failed_attempts:
                        user_profile.locked_until = datetime.utcnow() + self.lockout_duration
                        self.logger.warning(f"Account locked: {username}")
                    
                    raise AuthenticationError("Invalid credentials")
                
                # Successful authentication - reset failed attempts
                user_profile.failed_attempts = 0
                user_profile.last_login = datetime.utcnow()
                self._clear_failed_attempts(username)
                
                # Create session
                session = self._create_session(user_profile, ip_address, user_agent)
                
                self.logger.info(f"User authenticated: {username}")
                return session
                
        except Exception as e:
            if isinstance(e, (AuthenticationError, ValidationError)):
                raise
            
            error = AuthenticationError(f"Authentication failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def authenticate_token(self, token: str) -> Optional[EnterpriseSession]:
        """Authenticate user using JWT token"""
        try:
            # Verify token
            payload = self.jwt_manager.verify_token(token)
            user_id = payload['user_id']
            
            if user_id not in self.users:
                raise AuthenticationError("User not found")
            
            user_profile = self.users[user_id]
            
            # Check if user is locked
            if user_profile.is_locked:
                raise AuthenticationError("Account is locked")
            
            # Create session from token
            session_id = secrets.token_urlsafe(32)
            session = EnterpriseSession(
                session_id=session_id,
                user_id=user_id,
                user_profile=user_profile,
                permissions_cache=set(payload.get('permissions', []))
            )
            
            self.sessions[session_id] = session
            
            self.logger.info(f"Token authenticated: {user_profile.username}")
            return session
            
        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            
            error = AuthenticationError(f"Token authentication failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def authorize_action(self, session: EnterpriseSession, 
                        required_permission: str) -> bool:
        """Authorize user action based on permissions"""
        try:
            if not session.is_active:
                return False
            
            # Super admin has all permissions
            if session.user_profile.role == UserRole.SUPER_ADMIN:
                return True
            
            # Check cached permissions
            if required_permission in session.permissions_cache:
                session.update_activity("permission_check", {"permission": required_permission})
                return True
            
            # Check user permissions
            if required_permission in session.user_profile.permissions:
                session.permissions_cache.add(required_permission)
                session.update_activity("permission_check", {"permission": required_permission})
                return True
            
            self.logger.warning(f"Access denied: {session.user_profile.username} lacks {required_permission}")
            return False
            
        except Exception as e:
            self.logger.error(f"Authorization check failed: {e}")
            return False
    
    def get_authentication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive authentication system statistics"""
        try:
            with self._lock:
                total_users = len(self.users)
                active_sessions = sum(1 for s in self.sessions.values() if s.is_active)
                locked_users = sum(1 for u in self.users.values() if u.is_locked)
                
                # Role distribution
                role_distribution = {}
                for user in self.users.values():
                    role = user.role.value
                    role_distribution[role] = role_distribution.get(role, 0) + 1
                
                # Recent activity
                recent_logins = sum(1 for u in self.users.values() 
                                  if u.last_login and (datetime.utcnow() - u.last_login).total_seconds() < 3600)
                
                # Failed attempt statistics
                recent_failed = sum(1 for attempts in self.failed_login_attempts.values()
                                  for attempt in attempts
                                  if (datetime.utcnow() - attempt).total_seconds() < 3600)
                
                return {
                    'user_stats': {
                        'total_users': total_users,
                        'locked_users': locked_users,
                        'role_distribution': role_distribution,
                        'recent_logins_1h': recent_logins
                    },
                    'session_stats': {
                        'active_sessions': active_sessions,
                        'total_sessions': len(self.sessions),
                        'average_session_age_minutes': self._calculate_average_session_age()
                    },
                    'security_stats': {
                        'failed_attempts_1h': recent_failed,
                        'locked_accounts': locked_users,
                        'authentication_methods': ['password', 'jwt_token']
                    },
                    'overall_security_score': self._calculate_authentication_security_score()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating authentication statistics: {e}")
            return {'error': str(e)}
    
    def _create_session(self, user_profile: UserProfile, 
                       ip_address: str = None, user_agent: str = None) -> EnterpriseSession:
        """Create new user session"""
        session_id = secrets.token_urlsafe(32)
        
        session = EnterpriseSession(
            session_id=session_id,
            user_id=user_profile.user_id,
            user_profile=user_profile,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions_cache=user_profile.permissions.copy()
        )
        
        session.update_activity("login", {
            'ip_address': ip_address,
            'user_agent': user_agent
        })
        
        self.sessions[session_id] = session
        return session
    
    def _get_default_permissions(self, role: UserRole) -> Set[str]:
        """Get default permissions for role"""
        base_permissions = {
            UserRole.SUPER_ADMIN: {
                'admin.users.create', 'admin.users.delete', 'admin.users.modify',
                'admin.system.configure', 'admin.logs.view', 'admin.security.manage',
                'user.profile.view', 'user.profile.edit', 'content.read', 'content.write'
            },
            UserRole.ADMIN: {
                'admin.users.create', 'admin.users.modify', 'admin.logs.view',
                'user.profile.view', 'user.profile.edit', 'content.read', 'content.write'
            },
            UserRole.MANAGER: {
                'user.profile.view', 'user.profile.edit', 'content.read', 'content.write',
                'team.view', 'team.manage'
            },
            UserRole.USER: {
                'user.profile.view', 'user.profile.edit', 'content.read', 'content.write'
            },
            UserRole.GUEST: {
                'content.read'
            },
            UserRole.READONLY: {
                'content.read', 'user.profile.view'
            }
        }
        
        return base_permissions.get(role, set())
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_login_attempts:
            self.failed_login_attempts[username] = []
        
        self.failed_login_attempts[username].append(datetime.utcnow())
        
        # Clean old attempts (older than 1 hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.failed_login_attempts[username] = [
            attempt for attempt in self.failed_login_attempts[username]
            if attempt > cutoff_time
        ]
    
    def _clear_failed_attempts(self, username: str):
        """Clear failed login attempts for user"""
        self.failed_login_attempts.pop(username, None)
    
    def _is_user_locked_out(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts"""
        if username not in self.failed_login_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_login_attempts[username]
            if (datetime.utcnow() - attempt) < self.lockout_duration
        ]
        
        return len(recent_attempts) >= self.max_failed_attempts
    
    def _calculate_average_session_age(self) -> float:
        """Calculate average session age in minutes"""
        if not self.sessions:
            return 0.0
        
        total_age = sum(session.session_age_minutes for session in self.sessions.values())
        return total_age / len(self.sessions)
    
    def _calculate_authentication_security_score(self) -> float:
        """Calculate overall authentication security score"""
        try:
            score = 0.0
            
            # Password policy compliance
            if self.password_manager.min_password_length >= 12:
                score += 20
            if self.password_manager.require_complexity:
                score += 20
            
            # Account lockout configuration
            if self.max_failed_attempts <= 5:
                score += 15
            if self.lockout_duration >= timedelta(minutes=15):
                score += 15
            
            # User security status
            if self.users:
                locked_rate = sum(1 for u in self.users.values() if u.is_locked) / len(self.users)
                score += (1 - locked_rate) * 15
            
            # Session security
            if self.sessions:
                active_rate = sum(1 for s in self.sessions.values() if s.is_active) / len(self.sessions)
                score += active_rate * 15
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating authentication security score: {e}")
            return 0.0
    
    def _create_default_admin(self):
        """Create default administrator account"""
        try:
            admin_username = "admin"
            admin_password = os.getenv('PASSWORD')  # Should be changed immediately in production
            
            if not any(user.username == admin_username for user in self.users.values()):
                self.create_user(
                    username=admin_username,
                    email="admin@testmaster.local",
                    password=admin_password,
                    role=UserRole.SUPER_ADMIN,
                    organization="system"
                )
                
                self.logger.info("Default admin user created (change password immediately)")
                
        except Exception as e:
            self.logger.error(f"Failed to create default admin: {e}")


# Global enterprise authentication manager
enterprise_auth = EnterpriseAuthenticationManager()


# Convenience functions
def create_enterprise_user(username: str, email: str, password: str, 
                          role: UserRole = UserRole.USER) -> bool:
    """Convenience function to create enterprise user"""
    return enterprise_auth.create_user(username, email, password, role)


def authenticate_enterprise_user(username: str, password: str) -> Optional[EnterpriseSession]:
    """Convenience function to authenticate enterprise user"""
    return enterprise_auth.authenticate_user(username, password)