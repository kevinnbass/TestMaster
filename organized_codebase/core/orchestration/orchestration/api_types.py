"""
API Gateway Types and Data Structures

This module defines the core data types, enumerations, and data structures
used throughout the API gateway system.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple


class HTTPMethod(Enum):
    """HTTP methods supported by the API"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class AuthenticationLevel(Enum):
    """Authentication levels for API endpoints"""
    NONE = "none"
    BASIC = "basic"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    BEARER_TOKEN = "bearer_token"


class RateLimitType(Enum):
    """Rate limiting time windows"""
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"


class APIStatus(Enum):
    """API operation status"""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


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
    status: APIStatus = APIStatus.ACTIVE
    version: str = "v1"
    deprecated_since: Optional[datetime] = None
    cache_ttl: Optional[int] = None


@dataclass
class APIUser:
    """API user account and permissions"""
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
    role: str = "user"
    scopes: List[str] = field(default_factory=list)


@dataclass
class APIRequest:
    """API request context and metadata"""
    request_id: str
    user_id: Optional[str]
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: str = "unknown"
    user_agent: str = "unknown"
    auth_token: Optional[str] = None


@dataclass
class APIResponse:
    """API response structure"""
    status_code: int
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "status_code": self.status_code,
            "data": self.data,
            "message": self.message,
            "errors": self.errors,
            "metadata": self.metadata,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RateLimit:
    """Rate limiting configuration and state"""
    limit: int
    window: RateLimitType
    remaining: int = 0
    reset_at: Optional[datetime] = None
    burst_limit: Optional[int] = None
    
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded"""
        return self.remaining <= 0
    
    def reset_if_needed(self) -> None:
        """Reset rate limit if window has passed"""
        if self.reset_at and datetime.now() >= self.reset_at:
            self.remaining = self.limit
            self._calculate_next_reset()
    
    def _calculate_next_reset(self) -> None:
        """Calculate next reset time"""
        now = datetime.now()
        if self.window == RateLimitType.PER_SECOND:
            self.reset_at = now.replace(microsecond=0) + timedelta(seconds=1)
        elif self.window == RateLimitType.PER_MINUTE:
            self.reset_at = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        elif self.window == RateLimitType.PER_HOUR:
            self.reset_at = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif self.window == RateLimitType.PER_DAY:
            self.reset_at = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


@dataclass
class APIMetrics:
    """API performance and usage metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    rate_limited_requests: int = 0
    authentication_failures: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    active_connections: int = 0
    bytes_transferred: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return (self.cache_hits / total_cache_requests) * 100


@dataclass
class APIConfiguration:
    """API gateway configuration"""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    api_version: str = "v1"
    api_prefix: str = "/api"
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting_enabled: bool = True
    authentication_enabled: bool = True
    jwt_secret_key: str = "default-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # seconds
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30  # seconds
    enable_swagger: bool = True
    log_requests: bool = True
    enable_caching: bool = True
    cache_default_ttl: int = 300  # seconds