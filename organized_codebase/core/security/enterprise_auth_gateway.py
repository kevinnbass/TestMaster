"""
Enterprise Authentication Gateway - PHASE 3.1

Advanced enterprise-grade authentication system with multi-provider support,
comprehensive security features, and archive-derived patterns.
"""

import sqlite3
import json
import hashlib
import secrets
import time
import jwt
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from threading import RLock
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum

# SQLite Database Schema
AUTH_GATEWAY_SCHEMA = '''
CREATE TABLE IF NOT EXISTS auth_providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_type TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    config_data TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_token TEXT NOT NULL UNIQUE,
    provider_type TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    permissions TEXT NOT NULL,
    metadata TEXT,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    token_type TEXT NOT NULL,
    token_data TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    scope TEXT,
    audience TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS security_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    user_id TEXT,
    session_token TEXT,
    details TEXT,
    severity TEXT DEFAULT 'info',
    ip_address TEXT,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
'''

class AuthProviderType(Enum):
    """Authentication provider types."""
    OAUTH2 = "oauth2"
    SAML = "saml"
    JWT = "jwt"
    API_KEY = "api_key"
    LDAP = "ldap"
    OPENID_CONNECT = "openid_connect"

class SessionStatus(Enum):
    """User session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

class TokenType(Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    ID_TOKEN = "id_token"
    API_KEY = "api_key"

class SecurityEventType(Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REVOKED = "token_revoked"
    SESSION_EXPIRED = "session_expired"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@dataclass
class AuthProvider:
    """Authentication provider configuration."""
    provider_type: AuthProviderType
    provider_name: str
    config: Dict[str, Any]
    status: str = "active"
    id: Optional[int] = None

@dataclass
class UserSession:
    """User session information."""
    user_id: str
    session_token: str
    provider_type: AuthProviderType
    expires_at: datetime
    permissions: List[str]
    metadata: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None

@dataclass
class AuthToken:
    """Authentication token."""
    token_id: str
    user_id: str
    token_type: TokenType
    token_data: str
    expires_at: datetime
    scope: Optional[str] = None
    audience: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    user_id: Optional[str]
    session_token: Optional[str]
    details: Dict[str, Any]
    severity: str = "info"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    id: Optional[int] = None
    timestamp: Optional[datetime] = None

class JWTManager:
    """JWT token manager with advanced features."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def generate_token(self, payload: Dict[str, Any], 
                      expires_in: int = 3600) -> str:
        """Generate JWT token with expiration."""
        now = datetime.utcnow()
        payload.update({
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'jti': secrets.token_urlsafe(32)
        })
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
    
    def refresh_token(self, token: str, expires_in: int = 3600) -> str:
        """Refresh JWT token with new expiration."""
        payload = self.verify_token(token)
        # Remove old timing claims
        for claim in ['iat', 'exp', 'jti']:
            payload.pop(claim, None)
        return self.generate_token(payload, expires_in)

class SessionManager:
    """Advanced session management with security features."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = RLock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(AUTH_GATEWAY_SCHEMA)
    
    def create_session(self, session: UserSession) -> str:
        """Create new user session."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_sessions 
                    (user_id, session_token, provider_type, expires_at, 
                     permissions, metadata, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.user_id,
                    session.session_token,
                    session.provider_type.value,
                    session.expires_at.isoformat(),
                    json.dumps(session.permissions),
                    json.dumps(session.metadata),
                    session.ip_address,
                    session.user_agent
                ))
                return session.session_token
    
    def get_session(self, session_token: str) -> Optional[UserSession]:
        """Get session by token."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM user_sessions WHERE session_token = ?
            ''', (session_token,))
            row = cursor.fetchone()
            
            if row:
                return UserSession(
                    id=row['id'],
                    user_id=row['user_id'],
                    session_token=row['session_token'],
                    provider_type=AuthProviderType(row['provider_type']),
                    expires_at=datetime.fromisoformat(row['expires_at']),
                    permissions=json.loads(row['permissions']),
                    metadata=json.loads(row['metadata'] or '{}'),
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_accessed=datetime.fromisoformat(row['last_accessed'])
                )
        return None
    
    def update_session_access(self, session_token: str):
        """Update session last access time."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE user_sessions 
                    SET last_accessed = CURRENT_TIMESTAMP 
                    WHERE session_token = ?
                ''', (session_token,))
    
    def revoke_session(self, session_token: str):
        """Revoke user session."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_sessions WHERE session_token = ?
                ''', (session_token,))

class EnterpriseAuthGateway:
    """Enterprise authentication gateway with multi-provider support."""
    
    def __init__(self, db_path: str = "enterprise_auth.db", 
                 jwt_secret: Optional[str] = None):
        self.db_path = db_path
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.lock = RLock()
        
        # Initialize components
        self.jwt_manager = JWTManager(self.jwt_secret)
        self.session_manager = SessionManager(db_path)
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(AUTH_GATEWAY_SCHEMA)
    
    def register_provider(self, provider: AuthProvider) -> int:
        """Register authentication provider."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO auth_providers 
                    (provider_type, provider_name, config_data, status)
                    VALUES (?, ?, ?, ?)
                ''', (
                    provider.provider_type.value,
                    provider.provider_name,
                    json.dumps(provider.config),
                    provider.status
                ))
                return cursor.lastrowid
    
    def authenticate_user(self, user_id: str, provider_type: AuthProviderType,
                         permissions: List[str], metadata: Dict[str, Any] = None,
                         ip_address: str = None, user_agent: str = None) -> UserSession:
        """Authenticate user and create session."""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            provider_type=provider_type,
            expires_at=expires_at,
            permissions=permissions,
            metadata=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.session_manager.create_session(session)
        
        # Log security event
        self._log_security_event(SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            user_id=user_id,
            session_token=session_token,
            details={"provider": provider_type.value},
            ip_address=ip_address,
            user_agent=user_agent
        ))
        
        return session
    
    def issue_token(self, user_id: str, token_type: TokenType,
                   scope: str = None, audience: str = None,
                   expires_in: int = 3600) -> AuthToken:
        """Issue authentication token."""
        token_id = secrets.token_urlsafe(16)
        payload = {
            'sub': user_id,
            'token_id': token_id,
            'type': token_type.value
        }
        
        if scope:
            payload['scope'] = scope
        if audience:
            payload['aud'] = audience
        
        token_data = self.jwt_manager.generate_token(payload, expires_in)
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        token = AuthToken(
            token_id=token_id,
            user_id=user_id,
            token_type=token_type,
            token_data=token_data,
            expires_at=expires_at,
            scope=scope,
            audience=audience
        )
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO auth_tokens 
                    (token_id, user_id, token_type, token_data, 
                     expires_at, scope, audience)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    token.token_id,
                    token.user_id,
                    token.token_type.value,
                    token.token_data,
                    token.expires_at.isoformat(),
                    token.scope,
                    token.audience
                ))
        
        # Log token issuance
        self._log_security_event(SecurityEvent(
            event_type=SecurityEventType.TOKEN_ISSUED,
            user_id=user_id,
            details={
                "token_id": token_id,
                "token_type": token_type.value,
                "scope": scope,
                "audience": audience
            }
        ))
        
        return token
    
    def verify_token(self, token_data: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token."""
        try:
            payload = self.jwt_manager.verify_token(token_data)
            token_id = payload.get('token_id')
            
            if token_id:
                # Check if token is revoked
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT revoked_at FROM auth_tokens 
                        WHERE token_id = ?
                    ''', (token_id,))
                    row = cursor.fetchone()
                    
                    if row and row[0]:  # Token is revoked
                        return None
            
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def validate_session(self, session_token: str) -> Optional[UserSession]:
        """Validate user session."""
        session = self.session_manager.get_session(session_token)
        
        if not session:
            return None
        
        # Check expiration
        if datetime.utcnow() > session.expires_at:
            self.session_manager.revoke_session(session_token)
            self._log_security_event(SecurityEvent(
                event_type=SecurityEventType.SESSION_EXPIRED,
                user_id=session.user_id,
                session_token=session_token,
                details={"reason": "expired"}
            ))
            return None
        
        # Update access time
        self.session_manager.update_session_access(session_token)
        return session
    
    def revoke_token(self, token_id: str):
        """Revoke authentication token."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE auth_tokens 
                    SET revoked_at = CURRENT_TIMESTAMP 
                    WHERE token_id = ?
                ''', (token_id,))
                
                # Get user_id for logging
                cursor.execute('''
                    SELECT user_id FROM auth_tokens WHERE token_id = ?
                ''', (token_id,))
                row = cursor.fetchone()
                
                if row:
                    self._log_security_event(SecurityEvent(
                        event_type=SecurityEventType.TOKEN_REVOKED,
                        user_id=row[0],
                        details={"token_id": token_id}
                    ))
    
    def _log_security_event(self, event: SecurityEvent):
        """Log security event."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO security_events 
                    (event_type, user_id, session_token, details, 
                     severity, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_type.value,
                    event.user_id,
                    event.session_token,
                    json.dumps(event.details),
                    event.severity,
                    event.ip_address,
                    event.user_agent
                ))
    
    def get_security_events(self, user_id: str = None, 
                           event_type: SecurityEventType = None,
                           limit: int = 100) -> List[SecurityEvent]:
        """Get security events."""
        query = "SELECT * FROM security_events WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                event = SecurityEvent(
                    id=row['id'],
                    event_type=SecurityEventType(row['event_type']),
                    user_id=row['user_id'],
                    session_token=row['session_token'],
                    details=json.loads(row['details']),
                    severity=row['severity'],
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    timestamp=datetime.fromisoformat(row['timestamp'])
                )
                events.append(event)
            
            return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Active sessions
            cursor.execute('''
                SELECT COUNT(*) FROM user_sessions 
                WHERE expires_at > CURRENT_TIMESTAMP
            ''')
            stats['active_sessions'] = cursor.fetchone()[0]
            
            # Total tokens issued
            cursor.execute('SELECT COUNT(*) FROM auth_tokens')
            stats['total_tokens'] = cursor.fetchone()[0]
            
            # Active tokens
            cursor.execute('''
                SELECT COUNT(*) FROM auth_tokens 
                WHERE expires_at > CURRENT_TIMESTAMP AND revoked_at IS NULL
            ''')
            stats['active_tokens'] = cursor.fetchone()[0]
            
            # Security events in last 24h
            cursor.execute('''
                SELECT COUNT(*) FROM security_events 
                WHERE timestamp > datetime('now', '-1 day')
            ''')
            stats['recent_security_events'] = cursor.fetchone()[0]
            
            # Provider count
            cursor.execute('''
                SELECT COUNT(*) FROM auth_providers WHERE status = 'active'
            ''')
            stats['active_providers'] = cursor.fetchone()[0]
            
            return stats

# Global instance
enterprise_auth_gateway = EnterpriseAuthGateway()

# Convenience functions
def authenticate_user(user_id: str, provider_type: AuthProviderType,
                     permissions: List[str], **kwargs) -> UserSession:
    """Authenticate user."""
    return enterprise_auth_gateway.authenticate_user(
        user_id, provider_type, permissions, **kwargs
    )

def issue_token(user_id: str, token_type: TokenType, **kwargs) -> AuthToken:
    """Issue authentication token."""
    return enterprise_auth_gateway.issue_token(user_id, token_type, **kwargs)

def verify_token(token_data: str) -> Optional[Dict[str, Any]]:
    """Verify authentication token."""
    return enterprise_auth_gateway.verify_token(token_data)

def validate_session(session_token: str) -> Optional[UserSession]:
    """Validate user session."""
    return enterprise_auth_gateway.validate_session(session_token)