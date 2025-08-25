"""
Agency-Swarm Derived Authentication Security Module
Extracted from agency-swarm FastAPI integration patterns
Enhanced for comprehensive authentication and authorization
"""

import os
import jwt
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from .error_handler import AuthenticationError, AuthorizationError, security_error_handler


@dataclass
class UserSession:
    """User session data structure"""
    user_id: str
    username: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    session_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    last_activity: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuthenticationContext:
    """Authentication context for requests"""
    is_authenticated: bool = False
    user_session: Optional[UserSession] = None
    token: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TokenManager:
    """JWT token management based on agency-swarm patterns"""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
        
    def generate_token(self, user_session: UserSession, expires_hours: int = 24) -> str:
        """Generate JWT token for user session"""
        try:
            payload = {
                'user_id': user_session.user_id,
                'username': user_session.username,
                'roles': user_session.roles,
                'session_token': user_session.session_token,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=expires_hours),
                'iss': 'testmaster-security'
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            self.logger.info(f"Generated token for user: {user_session.username}")
            return token
            
        except Exception as e:
            error = AuthenticationError(f"Token generation failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token expiration
            if datetime.utcfromtimestamp(payload['exp']) < datetime.utcnow():
                raise AuthenticationError("Token has expired")
            
            self.logger.debug(f"Token verified for user: {payload.get('username')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            error = AuthenticationError("Token has expired")
            security_error_handler.handle_error(error)
            raise error
        except jwt.InvalidTokenError as e:
            error = AuthenticationError(f"Invalid token: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
        except Exception as e:
            error = AuthenticationError(f"Token verification failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error


class AuthenticationManager:
    """Central authentication management system"""
    
    def __init__(self, app_token: Optional[str] = None):
        self.app_token = app_token or os.getenv("APP_TOKEN")
        self.token_manager = TokenManager()
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.logger = logging.getLogger(__name__)
        
    def authenticate_token(self, token: str, context: Dict[str, Any] = None) -> AuthenticationContext:
        """Authenticate using token - based on agency-swarm verify_token pattern"""
        auth_context = AuthenticationContext(
            token=token,
            ip_address=context.get('ip_address') if context else None,
            user_agent=context.get('user_agent') if context else None
        )
        
        try:
            # Handle app token authentication (agency-swarm pattern)
            if self.app_token and token == self.app_token:
                auth_context.is_authenticated = True
                self.logger.info("App token authentication successful")
                return auth_context
            
            # Handle JWT token authentication
            payload = self.token_manager.verify_token(token)
            session_token = payload.get('session_token')
            
            # Verify session exists and is valid
            if session_token in self.active_sessions:
                session = self.active_sessions[session_token]
                if session.expires_at > datetime.utcnow():
                    # Update last activity
                    session.last_activity = datetime.utcnow()
                    auth_context.is_authenticated = True
                    auth_context.user_session = session
                    self.logger.info(f"JWT authentication successful for: {session.username}")
                    return auth_context
                else:
                    # Session expired
                    del self.active_sessions[session_token]
                    
            raise AuthenticationError("Invalid or expired session")
            
        except AuthenticationError:
            raise
        except Exception as e:
            error = AuthenticationError(f"Authentication failed: {str(e)}")
            security_error_handler.handle_error(error, context)
            raise error
    
    def create_user_session(self, username: str, password: str, roles: List[str] = None) -> UserSession:
        """Create authenticated user session"""
        try:
            # Check for account lockout
            if self._is_account_locked(username):
                raise AuthenticationError(f"Account locked due to too many failed attempts")
            
            # Simulate password verification (replace with real auth)
            if not self._verify_password(username, password):
                self._record_failed_attempt(username)
                raise AuthenticationError("Invalid credentials")
            
            # Create session
            user_session = UserSession(
                user_id=hashlib.sha256(username.encode()).hexdigest()[:16],
                username=username,
                roles=roles or ['user']
            )
            
            # Store active session
            self.active_sessions[user_session.session_token] = user_session
            
            # Clear failed attempts
            if username in self.failed_attempts:
                del self.failed_attempts[username]
            
            self.logger.info(f"Created session for user: {username}")
            return user_session
            
        except AuthenticationError:
            raise
        except Exception as e:
            error = AuthenticationError(f"Session creation failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def terminate_session(self, session_token: str):
        """Terminate user session"""
        if session_token in self.active_sessions:
            username = self.active_sessions[session_token].username
            del self.active_sessions[session_token]
            self.logger.info(f"Terminated session for user: {username}")
    
    def check_permission(self, session: UserSession, required_permission: str) -> bool:
        """Check if user has required permission"""
        return (
            required_permission in session.permissions or 
            'admin' in session.roles or
            'superuser' in session.roles
        )
    
    def _verify_password(self, username: str, password: str) -> bool:
        """Password verification placeholder - implement with real auth system"""
        # This is a placeholder - implement with proper password hashing/verification
        return len(password) >= 8
    
    def _record_failed_attempt(self, username: str):
        """Record failed authentication attempt"""
        now = datetime.utcnow()
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(now)
        
        # Clean old attempts
        cutoff = now - self.lockout_duration
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username] 
            if attempt > cutoff
        ]
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username not in self.failed_attempts:
            return False
        
        recent_attempts = len(self.failed_attempts[username])
        return recent_attempts >= self.max_failed_attempts
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        active_count = len(self.active_sessions)
        locked_accounts = sum(1 for username in self.failed_attempts 
                             if self._is_account_locked(username))
        
        return {
            'active_sessions': active_count,
            'locked_accounts': locked_accounts,
            'total_failed_attempts': sum(len(attempts) for attempts in self.failed_attempts.values())
        }


class AuthorizationManager:
    """Role-based access control system"""
    
    def __init__(self):
        self.role_permissions: Dict[str, List[str]] = {
            'admin': ['*'],  # All permissions
            'user': ['read', 'write_own'],
            'guest': ['read']
        }
        self.logger = logging.getLogger(__name__)
    
    def authorize_request(self, session: UserSession, resource: str, action: str) -> bool:
        """Authorize request based on user roles and permissions"""
        try:
            # Check if user has admin privileges
            if 'admin' in session.roles:
                return True
            
            # Check specific permissions
            required_permission = f"{action}:{resource}"
            
            for role in session.roles:
                if role in self.role_permissions:
                    permissions = self.role_permissions[role]
                    if '*' in permissions or required_permission in permissions:
                        return True
            
            # Check direct permissions
            if required_permission in session.permissions:
                return True
            
            self.logger.warning(
                f"Authorization denied for user {session.username}: "
                f"{action} on {resource}"
            )
            return False
            
        except Exception as e:
            error = AuthorizationError(f"Authorization check failed: {str(e)}")
            security_error_handler.handle_error(error)
            return False


# Global instances
auth_manager = AuthenticationManager()
authz_manager = AuthorizationManager()