"""
API Authentication Manager - Handles authentication and authorization

This module provides comprehensive authentication services including:
- API key authentication
- JWT token management
- OAuth integration
- Permission management
- Session handling
"""

import secrets
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import hmac
import jwt

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.api_models import APIUser, APISession, AuthenticationLevel, create_api_user

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """
    Manages API authentication and authorization
    
    Provides multiple authentication methods and comprehensive
    user management capabilities.
    """
    
    def __init__(self, jwt_secret: Optional[str] = None):
        """
        Initialize authentication manager
        
        Args:
            jwt_secret: Global JWT secret (optional, users can have individual secrets)
        """
        self.users: Dict[str, APIUser] = {}
        self.active_sessions: Dict[str, APISession] = {}
        self.api_key_index: Dict[str, str] = {}  # api_key -> user_id mapping
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        
        # Security configuration
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        self.session_duration_minutes = 60
        self.token_expiry_hours = 24
        
        # OAuth providers (mock)
        self.oauth_providers = {}
        
        logger.info("Authentication Manager initialized")
    
    def create_user(
        self,
        username: str,
        email: str,
        permissions: List[str],
        password: Optional[str] = None,
        is_admin: bool = False
    ) -> APIUser:
        """
        Create new API user
        
        Args:
            username: User's username
            email: User's email
            permissions: List of permissions
            password: Optional password for basic auth
            is_admin: Whether user has admin privileges
            
        Returns:
            Created APIUser instance
        """
        # Check for duplicate username or email
        for user in self.users.values():
            if user.username == username:
                raise ValueError(f"Username '{username}' already exists")
            if user.email == email:
                raise ValueError(f"Email '{email}' already registered")
        
        # Create user
        user = create_api_user(username, email, permissions)
        user.is_admin = is_admin
        
        # Hash password if provided
        if password:
            user.password_hash = self._hash_password(password)
        
        # Store user and create index
        self.users[user.user_id] = user
        self.api_key_index[user.api_key] = user.user_id
        
        logger.info(f"Created user: {username} (ID: {user.user_id})")
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[APIUser]:
        """
        Authenticate using API key
        
        Args:
            api_key: API key to authenticate
            
        Returns:
            Authenticated user or None
        """
        user_id = self.api_key_index.get(api_key)
        if not user_id:
            logger.warning(f"Invalid API key attempted")
            return None
        
        user = self.users.get(user_id)
        if not user:
            logger.error(f"User not found for valid API key")
            return None
        
        if not user.is_active:
            logger.warning(f"Inactive user attempted access: {user.username}")
            return None
        
        if user.is_locked():
            logger.warning(f"Locked user attempted access: {user.username}")
            return None
        
        user.update_last_access()
        logger.debug(f"API key authentication successful for {user.username}")
        return user
    
    def authenticate_jwt(self, token: str) -> Optional[APIUser]:
        """
        Authenticate using JWT token
        
        Args:
            token: JWT token to authenticate
            
        Returns:
            Authenticated user or None
        """
        try:
            # Decode without verification first to get user_id
            unverified = jwt.decode(token, options={"verify_signature": False})
            user_id = unverified.get("user_id")
            
            if not user_id or user_id not in self.users:
                logger.warning("JWT contains invalid user_id")
                return None
            
            user = self.users[user_id]
            
            # Verify with user's secret
            payload = jwt.decode(
                token,
                user.jwt_secret,
                algorithms=["HS256"]
            )
            
            # Validate token claims
            if payload.get("user_id") != user_id:
                logger.warning("JWT user_id mismatch")
                return None
            
            # Check if user is still active
            if not user.is_active or user.is_locked():
                logger.warning(f"JWT authentication failed - user inactive: {user.username}")
                return None
            
            user.update_last_access()
            logger.debug(f"JWT authentication successful for {user.username}")
            return user
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT authentication error: {e}")
            return None
    
    def authenticate_basic(self, username: str, password: str) -> Optional[APIUser]:
        """
        Authenticate using username and password
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Authenticated user or None
        """
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return None
        
        # Check if account is locked
        if user.is_locked():
            logger.warning(f"Login attempt for locked account: {username}")
            return None
        
        # Verify password
        if not user.password_hash:
            logger.warning(f"No password set for user: {username}")
            return None
        
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.now() + timedelta(
                    minutes=self.lockout_duration_minutes
                )
                logger.warning(f"Account locked due to failed attempts: {username}")
            
            return None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.update_last_access()
        logger.info(f"Basic authentication successful for {username}")
        return user
    
    def generate_jwt_token(
        self,
        user_id: str,
        expires_in_hours: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate JWT token for user
        
        Args:
            user_id: User ID
            expires_in_hours: Token expiry time in hours
            
        Returns:
            JWT token or None
        """
        if user_id not in self.users:
            logger.error(f"Cannot generate token for non-existent user: {user_id}")
            return None
        
        user = self.users[user_id]
        expires_in = expires_in_hours or self.token_expiry_hours
        
        payload = {
            "user_id": user_id,
            "username": user.username,
            "permissions": user.permissions,
            "is_admin": user.is_admin,
            "exp": datetime.utcnow() + timedelta(hours=expires_in),
            "iat": datetime.utcnow(),
            "iss": "TestMaster API Gateway"
        }
        
        token = jwt.encode(payload, user.jwt_secret, algorithm="HS256")
        logger.debug(f"Generated JWT token for {user.username}")
        return token
    
    def create_session(self, user_id: str) -> Optional[APISession]:
        """
        Create new session for user
        
        Args:
            user_id: User ID
            
        Returns:
            Created session or None
        """
        if user_id not in self.users:
            return None
        
        session = APISession(
            session_id=secrets.token_urlsafe(32),
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=self.session_duration_minutes)
        )
        
        self.active_sessions[session.session_id] = session
        logger.debug(f"Created session for user {user_id}")
        return session
    
    def validate_session(self, session_id: str) -> Optional[APIUser]:
        """
        Validate session and return associated user
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            User if session is valid, None otherwise
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        if not session.is_valid():
            # Remove invalid session
            del self.active_sessions[session_id]
            return None
        
        # Update session activity
        session.last_activity = datetime.now()
        
        # Get associated user
        user = self.users.get(session.user_id)
        if user and user.is_active and not user.is_locked():
            return user
        
        return None
    
    def check_permission(self, user: APIUser, required_permission: str) -> bool:
        """
        Check if user has required permission
        
        Args:
            user: User to check
            required_permission: Permission required
            
        Returns:
            True if user has permission
        """
        return user.has_permission(required_permission)
    
    def revoke_user_access(self, user_id: str) -> bool:
        """
        Revoke user access
        
        Args:
            user_id: User ID to revoke
            
        Returns:
            True if successful
        """
        if user_id in self.users:
            user = self.users[user_id]
            user.is_active = False
            
            # Remove API key from index
            if user.api_key in self.api_key_index:
                del self.api_key_index[user.api_key]
            
            # Invalidate all user sessions
            sessions_to_remove = [
                sid for sid, session in self.active_sessions.items()
                if session.user_id == user_id
            ]
            for sid in sessions_to_remove:
                del self.active_sessions[sid]
            
            logger.info(f"Revoked access for user {user.username}")
            return True
        
        return False
    
    def rotate_api_key(self, user_id: str) -> Optional[str]:
        """
        Rotate user's API key
        
        Args:
            user_id: User ID
            
        Returns:
            New API key or None
        """
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        
        # Remove old key from index
        if user.api_key in self.api_key_index:
            del self.api_key_index[user.api_key]
        
        # Generate new key
        new_key = secrets.token_urlsafe(32)
        user.api_key = new_key
        self.api_key_index[new_key] = user_id
        
        logger.info(f"Rotated API key for user {user.username}")
        return new_key
    
    def get_user_by_id(self, user_id: str) -> Optional[APIUser]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[APIUser]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def list_active_users(self) -> List[APIUser]:
        """List all active users"""
        return [u for u in self.users.values() if u.is_active and not u.is_locked()]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired = [
            sid for sid, session in self.active_sessions.items()
            if session.is_expired()
        ]
        
        for sid in expired:
            del self.active_sessions[sid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return f"{salt}${pwd_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split('$')
            pwd_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return pwd_hash.hex() == hash_hex
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        return {
            "total_users": len(self.users),
            "active_users": len(self.list_active_users()),
            "locked_users": len([u for u in self.users.values() if u.is_locked()]),
            "active_sessions": len(self.active_sessions),
            "admin_users": len([u for u in self.users.values() if u.is_admin])
        }


# Factory function
def create_authentication_manager(jwt_secret: Optional[str] = None) -> AuthenticationManager:
    """
    Create and configure authentication manager
    
    Args:
        jwt_secret: Global JWT secret
        
    Returns:
        Configured authentication manager
    """
    return AuthenticationManager(jwt_secret)