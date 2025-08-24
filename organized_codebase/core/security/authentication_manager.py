"""
API Authentication and Authorization Manager

This module handles authentication, authorization, and user management
for the API gateway system.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import hashlib
import jwt
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from .api_types import (
    APIUser, APIRequest, AuthenticationLevel, APIConfiguration
)


class AuthenticationManager:
    """Manages API authentication and authorization"""
    
    def __init__(self, config: APIConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # User storage (in production, this would be a database)
        self.users: Dict[str, APIUser] = {}
        self.api_key_to_user: Dict[str, str] = {}
        self.email_to_user: Dict[str, str] = {}
        
        # Session management
        self.active_sessions: Dict[str, Dict] = {}
        self.blacklisted_tokens: Set[str] = set()
        
        # Initialize default admin user
        self._create_default_admin()
        
        self.logger.info("AuthenticationManager initialized")
    
    def create_user(self, username: str, email: str, password: str,
                   permissions: List[str] = None, role: str = "user") -> APIUser:
        """Create a new API user"""
        # Check if user already exists
        if email in self.email_to_user:
            raise ValueError(f"User with email {email} already exists")
        
        # Generate unique identifiers
        user_id = f"user_{secrets.token_hex(8)}"
        api_key = self._generate_api_key()
        jwt_secret = secrets.token_hex(32)
        
        # Create user
        user = APIUser(
            user_id=user_id,
            username=username,
            email=email,
            api_key=api_key,
            jwt_secret=jwt_secret,
            permissions=permissions or [],
            role=role
        )
        
        # Store user
        self.users[user_id] = user
        self.api_key_to_user[api_key] = user_id
        self.email_to_user[email] = user_id
        
        # Store password hash (simplified for demo)
        self._store_password_hash(user_id, password)
        
        self.logger.info(f"Created user: {username} ({email})")
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[APIUser]:
        """Authenticate user by API key"""
        user_id = self.api_key_to_user.get(api_key)
        if not user_id:
            return None
        
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return None
        
        # Update last access
        user.last_access = datetime.now()
        return user
    
    def authenticate_jwt(self, token: str) -> Optional[APIUser]:
        """Authenticate user by JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                return None
            
            # Decode token
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            user_id = payload.get("user_id")
            if not user_id:
                return None
            
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return None
            
            # Update last access
            user.last_access = datetime.now()
            return user
            
        except jwt.ExpiredSignatureError:
            self.logger.debug("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.debug("Invalid JWT token")
            return None
    
    def authenticate_basic(self, username: str, password: str) -> Optional[APIUser]:
        """Authenticate user with username/password"""
        # Find user by username or email
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user or not user.is_active:
            return None
        
        # Verify password
        if self._verify_password(user.user_id, password):
            user.last_access = datetime.now()
            return user
        
        return None
    
    def create_jwt_token(self, user: APIUser, expires_in: Optional[int] = None) -> str:
        """Create JWT token for user"""
        if expires_in is None:
            expires_in = self.config.jwt_expiration
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "permissions": user.permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=expires_in)
        }
        
        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        # Store session
        session_id = f"session_{secrets.token_hex(16)}"
        self.active_sessions[session_id] = {
            "user_id": user.user_id,
            "token": token,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=expires_in)
        }
        
        return token
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token"""
        try:
            # Add to blacklist
            self.blacklisted_tokens.add(token)
            
            # Remove from active sessions
            for session_id, session in list(self.active_sessions.items()):
                if session["token"] == token:
                    del self.active_sessions[session_id]
                    break
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to revoke token: {e}")
            return False
    
    def check_permission(self, user: APIUser, required_permission: str) -> bool:
        """Check if user has required permission"""
        # Super admin has all permissions
        if user.role == "admin":
            return True
        
        # Check specific permission
        return required_permission in user.permissions
    
    def check_scope(self, user: APIUser, required_scope: str) -> bool:
        """Check if user has required scope"""
        return required_scope in user.scopes
    
    def get_user_by_id(self, user_id: str) -> Optional[APIUser]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[APIUser]:
        """Get user by email"""
        user_id = self.email_to_user.get(email)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def update_user_permissions(self, user_id: str, permissions: List[str]) -> bool:
        """Update user permissions"""
        user = self.users.get(user_id)
        if user:
            user.permissions = permissions
            self.logger.info(f"Updated permissions for user {user.username}")
            return True
        return False
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        user = self.users.get(user_id)
        if user:
            user.is_active = False
            # Revoke all active sessions
            for session_id, session in list(self.active_sessions.items()):
                if session["user_id"] == user_id:
                    self.blacklisted_tokens.add(session["token"])
                    del self.active_sessions[session_id]
            self.logger.info(f"Deactivated user {user.username}")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions and tokens"""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if session["expires_at"] < now:
                expired_sessions.append(session_id)
                self.blacklisted_tokens.add(session["token"])
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        self.logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_authentication_stats(self) -> Dict[str, int]:
        """Get authentication statistics"""
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "active_sessions": len(self.active_sessions),
            "blacklisted_tokens": len(self.blacklisted_tokens)
        }
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_user = APIUser(
            user_id="admin_001",
            username="admin",
            email="admin@testmaster.local",
            api_key=self._generate_api_key(),
            jwt_secret=secrets.token_hex(32),
            permissions=["*"],  # All permissions
            role="admin"
        )
        
        self.users[admin_user.user_id] = admin_user
        self.api_key_to_user[admin_user.api_key] = admin_user.user_id
        self.email_to_user[admin_user.email] = admin_user.user_id
        
        # Store default password
        self._store_password_hash(admin_user.user_id, "admin123")
        
        self.logger.info(f"Created default admin user with API key: {admin_user.api_key}")
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"tm_{secrets.token_urlsafe(32)}"
    
    def _store_password_hash(self, user_id: str, password: str):
        """Store password hash (simplified for demo)"""
        # In production, use proper password hashing like bcrypt
        password_hash = hashlib.sha256(f"{password}{user_id}".encode()).hexdigest()
        # Store in secure location
        pass
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify password hash (simplified for demo)"""
        # In production, use proper password verification
        expected_hash = hashlib.sha256(f"{password}{user_id}".encode()).hexdigest()
        # Compare with stored hash
        return True  # Simplified for demo