"""
API Gateway Models - Core data structures for TestMaster API Gateway

This module provides data models for:
- API endpoints and routing
- Authentication and authorization
- Request/response handling
- Rate limiting and security
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
import uuid


class HTTPMethod(Enum):
    """HTTP request methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthenticationLevel(Enum):
    """Authentication requirement levels"""
    NONE = "none"
    BASIC = "basic"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    MULTI_FACTOR = "multi_factor"


class RateLimitType(Enum):
    """Rate limiting time windows"""
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_WEEK = "per_week"


class ResponseFormat(Enum):
    """API response formats"""
    JSON = "json"
    XML = "xml"
    HTML = "html"
    TEXT = "text"
    BINARY = "binary"


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: HTTPMethod
    handler: Callable
    auth_required: AuthenticationLevel = AuthenticationLevel.API_KEY
    rate_limit: Optional[Tuple[int, RateLimitType]] = None
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    responses: Dict[int, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    version: str = "1.0.0"
    permissions: List[str] = field(default_factory=list)
    timeout: int = 30  # seconds
    cache_ttl: int = 0  # seconds (0 = no cache)
    
    def __post_init__(self):
        """Validate endpoint configuration"""
        if not self.path.startswith('/'):
            self.path = '/' + self.path
        
        # Default responses if not provided
        if not self.responses:
            self.responses = {
                200: "Success",
                400: "Bad Request",
                401: "Unauthorized",
                500: "Internal Server Error"
            }


@dataclass
class APIUser:
    """API user account"""
    user_id: str
    username: str
    email: str
    api_key: str
    jwt_secret: str
    permissions: List[str]
    rate_limits: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_access: Optional[datetime] = None
    is_active: bool = True
    is_admin: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Security
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_hash: Optional[str] = None
    two_factor_enabled: bool = False
    
    # Usage tracking
    request_count: int = 0
    last_request: Optional[datetime] = None
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return self.is_admin or permission in self.permissions or "admin" in self.permissions
    
    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until:
            return datetime.now() < self.locked_until
        return False
    
    def update_last_access(self):
        """Update last access timestamp"""
        self.last_access = datetime.now()
        self.last_request = datetime.now()
        self.request_count += 1


@dataclass
class APIRequest:
    """API request context"""
    request_id: str
    user_id: Optional[str]
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: str = "unknown"
    user_agent: str = ""
    
    # Request metadata
    api_version: str = "1.0.0"
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Performance tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        """Initialize request ID if not provided"""
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = self.request_id
    
    def get_duration(self) -> Optional[float]:
        """Get request processing duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class APIResponse:
    """API response"""
    status_code: int
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Response metadata
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    # Pagination
    pagination: Optional[Dict[str, Any]] = None
    
    # Performance
    processing_time: Optional[float] = None
    
    def is_success(self) -> bool:
        """Check if response indicates success"""
        return 200 <= self.status_code < 300
    
    def is_error(self) -> bool:
        """Check if response indicates error"""
        return self.status_code >= 400
    
    def add_error(self, error: str):
        """Add error message"""
        self.errors.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        result = {
            "status_code": self.status_code,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version
        }
        
        if self.data is not None:
            result["data"] = self.data
        if self.message:
            result["message"] = self.message
        if self.errors:
            result["errors"] = self.errors
        if self.metadata:
            result["metadata"] = self.metadata
        if self.request_id:
            result["request_id"] = self.request_id
        if self.pagination:
            result["pagination"] = self.pagination
        if self.processing_time is not None:
            result["processing_time"] = self.processing_time
        
        return result


@dataclass
class RateLimitStatus:
    """Rate limit status for a user/endpoint"""
    user_id: str
    endpoint: str
    limit: int
    limit_type: RateLimitType
    current_count: int
    remaining_requests: int
    reset_time: datetime
    is_limited: bool = False
    
    def can_proceed(self) -> bool:
        """Check if request can proceed"""
        return not self.is_limited and self.remaining_requests > 0


@dataclass
class APISession:
    """API session information"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid"""
        return self.is_active and not self.is_expired()
    
    def refresh(self, duration_minutes: int = 60):
        """Refresh session expiration"""
        self.last_activity = datetime.now()
        self.expires_at = datetime.now() + timedelta(minutes=duration_minutes)


@dataclass
class APIMetrics:
    """API usage metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_users: int = 0
    active_users: int = 0
    
    # Performance metrics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_count: int = 0
    rate_limit_hits: int = 0
    
    # Endpoint metrics
    endpoint_usage: Dict[str, int] = field(default_factory=dict)
    method_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Time-based metrics
    requests_per_minute: float = 0.0
    requests_per_hour: float = 0.0
    peak_load: int = 0
    
    def calculate_error_rate(self):
        """Calculate error rate"""
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
        else:
            self.error_rate = 0.0
    
    def update_endpoint_usage(self, endpoint: str):
        """Update endpoint usage count"""
        self.endpoint_usage[endpoint] = self.endpoint_usage.get(endpoint, 0) + 1
    
    def update_method_distribution(self, method: str):
        """Update method distribution"""
        self.method_distribution[method] = self.method_distribution.get(method, 0) + 1


# Factory functions
def create_api_endpoint(
    path: str,
    method: str,
    handler: Callable,
    **kwargs
) -> APIEndpoint:
    """Factory function to create API endpoint"""
    return APIEndpoint(
        path=path,
        method=HTTPMethod(method.upper()),
        handler=handler,
        **kwargs
    )


def create_api_user(
    username: str,
    email: str,
    permissions: List[str] = None
) -> APIUser:
    """Factory function to create API user"""
    import secrets
    
    return APIUser(
        user_id=str(uuid.uuid4()),
        username=username,
        email=email,
        api_key=secrets.token_urlsafe(32),
        jwt_secret=secrets.token_urlsafe(32),
        permissions=permissions or ["read"]
    )


def create_api_request(
    method: str,
    path: str,
    **kwargs
) -> APIRequest:
    """Factory function to create API request"""
    return APIRequest(
        request_id=str(uuid.uuid4()),
        user_id=kwargs.get('user_id'),
        method=method,
        path=path,
        headers=kwargs.get('headers', {}),
        query_params=kwargs.get('query_params', {}),
        body=kwargs.get('body'),
        ip_address=kwargs.get('ip_address', 'unknown')
    )


def create_api_response(
    status_code: int,
    **kwargs
) -> APIResponse:
    """Factory function to create API response"""
    return APIResponse(
        status_code=status_code,
        data=kwargs.get('data'),
        message=kwargs.get('message'),
        errors=kwargs.get('errors', []),
        metadata=kwargs.get('metadata', {}),
        request_id=kwargs.get('request_id')
    )


# Import for timedelta
from datetime import timedelta