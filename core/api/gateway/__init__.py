"""
TestMaster API Gateway - Enterprise-grade API management system

This package provides comprehensive API gateway functionality including:
- Multi-method authentication (API key, JWT, OAuth, Basic)
- Advanced rate limiting with multiple algorithms
- Automatic OpenAPI 3.0.3 documentation generation
- Request/response middleware pipeline
- Comprehensive monitoring and metrics
"""

from .api_models import (
    # Core models
    APIEndpoint,
    APIUser,
    APIRequest,
    APIResponse,
    APISession,
    APIMetrics,
    RateLimitStatus,
    
    # Enums
    HTTPMethod,
    AuthenticationLevel,
    RateLimitType,
    ResponseFormat,
    
    # Factory functions
    create_api_endpoint,
    create_api_user,
    create_api_request,
    create_api_response
)

from .api_authentication import (
    AuthenticationManager,
    create_authentication_manager
)

from .api_rate_limiting import (
    RateLimiter,
    RateLimitRule,
    RateLimitAlgorithm,
    RateLimitScope,
    TokenBucket,
    SlidingWindow,
    create_rate_limiter
)

from .api_documentation import (
    OpenAPIGenerator,
    OpenAPIInfo,
    OpenAPIServer,
    OpenAPISchema,
    DocumentationFormat,
    create_openapi_generator
)

from .api_gateway_core import (
    TestMasterAPIGateway,
    RequestMiddleware,
    CORSMiddleware,
    LoggingMiddleware,
    SecurityMiddleware,
    create_api_gateway
)

__all__ = [
    # Main gateway
    'TestMasterAPIGateway',
    'create_api_gateway',
    
    # Authentication
    'AuthenticationManager',
    'create_authentication_manager',
    
    # Rate limiting
    'RateLimiter',
    'RateLimitRule',
    'RateLimitAlgorithm',
    'RateLimitScope',
    'TokenBucket',
    'SlidingWindow',
    'create_rate_limiter',
    
    # Documentation
    'OpenAPIGenerator',
    'OpenAPIInfo',
    'OpenAPIServer',
    'OpenAPISchema',
    'DocumentationFormat',
    'create_openapi_generator',
    
    # Middleware
    'RequestMiddleware',
    'CORSMiddleware',
    'LoggingMiddleware',
    'SecurityMiddleware',
    
    # Models
    'APIEndpoint',
    'APIUser',
    'APIRequest',
    'APIResponse',
    'APISession',
    'APIMetrics',
    'RateLimitStatus',
    
    # Enums
    'HTTPMethod',
    'AuthenticationLevel',
    'RateLimitType',
    'ResponseFormat',
    
    # Factory functions
    'create_api_endpoint',
    'create_api_user',
    'create_api_request',
    'create_api_response'
]

# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster API Team'
__description__ = 'Enterprise-grade API gateway with comprehensive management capabilities'


def quick_gateway(port: int = 8000) -> TestMasterAPIGateway:
    """
    Create a quick API gateway with default settings
    
    Args:
        port: Port to run the gateway on
        
    Returns:
        Configured API gateway ready for use
    
    Example:
        >>> from api.gateway import quick_gateway
        >>> gateway = quick_gateway(8080)
        >>> # Add your endpoints
        >>> gateway.register_endpoint(...)
    """
    gateway = create_api_gateway(port=port)
    
    # Create a demo user for testing
    demo_user = gateway.auth_manager.create_user(
        username="demo",
        email="demo@testmaster.dev",
        permissions=["read", "write"],
        is_admin=False
    )
    
    print(f"Demo user created:")
    print(f"  Username: {demo_user.username}")
    print(f"  API Key: {demo_user.api_key}")
    print(f"  JWT Secret: {demo_user.jwt_secret[:20]}...")
    
    return gateway


def enterprise_gateway(
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_cors: bool = True,
    enable_security_headers: bool = True,
    max_request_log: int = 50000
) -> TestMasterAPIGateway:
    """
    Create an enterprise-grade API gateway with advanced features
    
    Args:
        host: Gateway host address
        port: Gateway port
        enable_cors: Whether to enable CORS middleware
        enable_security_headers: Whether to add security headers
        max_request_log: Maximum request log size
        
    Returns:
        Enterprise-configured API gateway
    
    Example:
        >>> from api.gateway import enterprise_gateway
        >>> gateway = enterprise_gateway(port=443)
        >>> # Gateway comes pre-configured with enterprise features
    """
    gateway = create_api_gateway(host=host, port=port)
    gateway.max_request_log_size = max_request_log
    
    # Add enterprise rate limiting rules
    from .api_rate_limiting import RateLimitRule, RateLimitScope, RateLimitAlgorithm
    
    enterprise_rules = [
        RateLimitRule(
            name="enterprise_global",
            scope=RateLimitScope.GLOBAL,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            limit=10000,
            window_size=1,
            burst_limit=15000,
            priority=1
        ),
        RateLimitRule(
            name="enterprise_user_hour",
            scope=RateLimitScope.USER,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            limit=10000,
            window_size=3600,
            priority=2
        ),
        RateLimitRule(
            name="enterprise_endpoint",
            scope=RateLimitScope.ENDPOINT,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            limit=1000,
            window_size=60,
            priority=3
        )
    ]
    
    for rule in enterprise_rules:
        gateway.rate_limiter.add_rule(rule)
    
    # Configure authentication for enterprise
    gateway.auth_manager.max_failed_attempts = 3
    gateway.auth_manager.lockout_duration_minutes = 60
    gateway.auth_manager.session_duration_minutes = 120
    
    print(f"Enterprise API Gateway configured on {host}:{port}")
    print(f"  Enhanced rate limiting: {len(enterprise_rules)} rules")
    print(f"  Security: Lockout after 3 failed attempts")
    print(f"  Session duration: 2 hours")
    
    return gateway


def development_gateway(port: int = 8000) -> TestMasterAPIGateway:
    """
    Create a development-friendly API gateway with relaxed settings
    
    Args:
        port: Port to run the gateway on
        
    Returns:
        Development-configured API gateway
    
    Example:
        >>> from api.gateway import development_gateway
        >>> gateway = development_gateway()
        >>> # Perfect for local development and testing
    """
    gateway = create_api_gateway(port=port)
    
    # Relaxed rate limiting for development
    from .api_rate_limiting import RateLimitRule, RateLimitScope, RateLimitAlgorithm
    
    dev_rule = RateLimitRule(
        name="dev_relaxed",
        scope=RateLimitScope.GLOBAL,
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        limit=10000,
        window_size=1,
        burst_limit=50000,
        priority=1
    )
    
    gateway.rate_limiter.add_rule(dev_rule)
    
    # Create multiple test users
    test_users = []
    for i in range(3):
        user = gateway.auth_manager.create_user(
            username=f"test_user_{i+1}",
            email=f"test{i+1}@testmaster.dev",
            permissions=["read", "write"] + (["admin"] if i == 0 else []),
            is_admin=(i == 0)
        )
        test_users.append(user)
    
    print(f"Development API Gateway configured on localhost:{port}")
    print(f"Test users created:")
    for user in test_users:
        print(f"  {user.username}: {user.api_key}")
    
    return gateway


# Convenience imports for common use cases
from .api_models import HTTPMethod as Method
from .api_models import AuthenticationLevel as Auth

# Module initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())