"""
API Gateway Core - Main gateway engine for request routing and processing

This module provides the core API gateway functionality including:
- Request routing and processing
- Middleware pipeline execution
- Error handling and recovery
- Comprehensive logging and monitoring
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from collections import deque
import json

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.api_models import (
    APIEndpoint, APIRequest, APIResponse, APIUser, APIMetrics,
    HTTPMethod, AuthenticationLevel, create_api_request, create_api_response
)
from .api_authentication import AuthenticationManager, create_authentication_manager
from .api_rate_limiting import RateLimiter, create_rate_limiter
from .api_documentation import OpenAPIGenerator, create_openapi_generator

logger = logging.getLogger(__name__)


class RequestMiddleware:
    """Base class for request middleware"""
    
    async def process_request(self, request: APIRequest) -> Optional[APIResponse]:
        """
        Process incoming request
        
        Args:
            request: Incoming API request
            
        Returns:
            Response if middleware wants to short-circuit, None otherwise
        """
        return None
    
    async def process_response(self, request: APIRequest, response: APIResponse) -> APIResponse:
        """
        Process outgoing response
        
        Args:
            request: Original API request
            response: API response
            
        Returns:
            Modified response
        """
        return response


class CORSMiddleware(RequestMiddleware):
    """CORS (Cross-Origin Resource Sharing) middleware"""
    
    def __init__(self, allowed_origins: List[str] = None):
        self.allowed_origins = allowed_origins or ["*"]
    
    async def process_response(self, request: APIRequest, response: APIResponse) -> APIResponse:
        """Add CORS headers to response"""
        if not response.metadata:
            response.metadata = {}
        
        response.metadata.update({
            "Access-Control-Allow-Origin": "*" if "*" in self.allowed_origins else self.allowed_origins[0],
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
            "Access-Control-Max-Age": "86400"
        })
        
        return response


class LoggingMiddleware(RequestMiddleware):
    """Request/response logging middleware"""
    
    def __init__(self):
        self.logger = logging.getLogger("api_gateway.requests")
    
    async def process_request(self, request: APIRequest) -> Optional[APIResponse]:
        """Log incoming request"""
        request.start_time = time.time()
        self.logger.info(
            f"REQUEST: {request.method} {request.path} "
            f"[{request.request_id}] from {request.ip_address}"
        )
        return None
    
    async def process_response(self, request: APIRequest, response: APIResponse) -> APIResponse:
        """Log outgoing response"""
        request.end_time = time.time()
        duration = request.get_duration() or 0
        
        self.logger.info(
            f"RESPONSE: {response.status_code} [{request.request_id}] "
            f"in {duration*1000:.2f}ms"
        )
        
        if response.processing_time is None:
            response.processing_time = duration
        
        return response


class SecurityMiddleware(RequestMiddleware):
    """Security headers and validation middleware"""
    
    async def process_response(self, request: APIRequest, response: APIResponse) -> APIResponse:
        """Add security headers to response"""
        if not response.metadata:
            response.metadata = {}
        
        response.metadata.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        })
        
        return response


class TestMasterAPIGateway:
    """
    Main API Gateway for TestMaster framework
    
    Provides comprehensive API management including routing, authentication,
    rate limiting, documentation, and monitoring.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_request_log_size: int = 10000
    ):
        """
        Initialize API Gateway
        
        Args:
            host: Gateway host address
            port: Gateway port
            max_request_log_size: Maximum request log size
        """
        self.host = host
        self.port = port
        self.max_request_log_size = max_request_log_size
        
        # Core components
        self.auth_manager = create_authentication_manager()
        self.rate_limiter = create_rate_limiter()
        self.openapi_generator = create_openapi_generator()
        
        # Gateway state
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.request_log: deque = deque(maxlen=max_request_log_size)
        self.is_running = False
        self.metrics = APIMetrics()
        
        # Middleware pipeline
        self.middleware: List[RequestMiddleware] = [
            LoggingMiddleware(),
            SecurityMiddleware(),
            CORSMiddleware()
        ]
        
        # Error handlers
        self.error_handlers: Dict[int, Callable] = {}
        
        # Initialize core endpoints
        self._register_core_endpoints()
        
        logger.info(f"TestMaster API Gateway initialized on {host}:{port}")
    
    def add_middleware(self, middleware: RequestMiddleware):
        """
        Add middleware to the pipeline
        
        Args:
            middleware: Middleware instance to add
        """
        self.middleware.append(middleware)
        logger.info(f"Added middleware: {middleware.__class__.__name__}")
    
    def register_endpoint(self, endpoint: APIEndpoint):
        """
        Register new API endpoint
        
        Args:
            endpoint: API endpoint to register
        """
        endpoint_key = f"{endpoint.method.value}:{endpoint.path}"
        self.endpoints[endpoint_key] = endpoint
        self.openapi_generator.add_endpoint(endpoint)
        
        logger.info(f"Registered endpoint: {endpoint.method.value} {endpoint.path}")
    
    def register_error_handler(self, status_code: int, handler: Callable):
        """
        Register custom error handler
        
        Args:
            status_code: HTTP status code to handle
            handler: Error handler function
        """
        self.error_handlers[status_code] = handler
        logger.info(f"Registered error handler for status code: {status_code}")
    
    async def handle_request(self, request: APIRequest) -> APIResponse:
        """
        Handle incoming API request
        
        Args:
            request: Incoming API request
            
        Returns:
            API response
        """
        try:
            # Process through middleware (request phase)
            for middleware in self.middleware:
                early_response = await middleware.process_request(request)
                if early_response:
                    return await self._process_response_middleware(request, early_response)
            
            # Find matching endpoint
            endpoint = self._find_endpoint(request)
            if not endpoint:
                return await self._handle_not_found(request)
            
            # Authenticate request
            user = await self._authenticate_request(request, endpoint.auth_required)
            if endpoint.auth_required != AuthenticationLevel.NONE and not user:
                return await self._handle_unauthorized(request)
            
            # Check permissions
            if endpoint.permissions and user:
                if not any(self.auth_manager.check_permission(user, perm) for perm in endpoint.permissions):
                    return await self._handle_forbidden(request)
            
            # Check rate limits
            if endpoint.rate_limit and user:
                allowed, statuses = self.rate_limiter.check_rate_limit(
                    user.user_id, endpoint.path, request.ip_address,
                    {endpoint.path: endpoint.rate_limit}
                )
                if not allowed:
                    return await self._handle_rate_limited(request, statuses)
            
            # Update metrics
            self._update_request_metrics(request, endpoint)
            
            # Execute endpoint handler
            response = await self._execute_endpoint(endpoint, request, user)
            
            # Process through middleware (response phase)
            return await self._process_response_middleware(request, response)
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return await self._handle_internal_error(request, e)
    
    def _find_endpoint(self, request: APIRequest) -> Optional[APIEndpoint]:
        """Find matching endpoint for request"""
        endpoint_key = f"{request.method}:{request.path}"
        endpoint = self.endpoints.get(endpoint_key)
        
        if not endpoint:
            # Try path parameter matching
            for key, ep in self.endpoints.items():
                method, path = key.split(':', 1)
                if method == request.method and self._match_path_pattern(request.path, path):
                    return ep
        
        return endpoint
    
    def _match_path_pattern(self, request_path: str, pattern_path: str) -> bool:
        """Match request path against pattern with parameters"""
        request_parts = request_path.strip('/').split('/')
        pattern_parts = pattern_path.strip('/').split('/')
        
        if len(request_parts) != len(pattern_parts):
            return False
        
        for req_part, pattern_part in zip(request_parts, pattern_parts):
            if pattern_part.startswith('{') and pattern_part.endswith('}'):
                continue  # Path parameter - matches anything
            elif req_part != pattern_part:
                return False
        
        return True
    
    async def _authenticate_request(
        self,
        request: APIRequest,
        auth_level: AuthenticationLevel
    ) -> Optional[APIUser]:
        """Authenticate API request"""
        if auth_level == AuthenticationLevel.NONE:
            return None
        
        # Try API key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user = self.auth_manager.authenticate_api_key(api_key)
            if user:
                return user
        
        # Try JWT authentication
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = self.auth_manager.authenticate_jwt(token)
            if user:
                return user
        
        # Try session authentication
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            user = self.auth_manager.validate_session(session_id)
            if user:
                return user
        
        return None
    
    async def _execute_endpoint(
        self,
        endpoint: APIEndpoint,
        request: APIRequest,
        user: Optional[APIUser]
    ) -> APIResponse:
        """Execute endpoint handler"""
        try:
            # Set correlation ID for tracking
            request.correlation_id = request.request_id
            
            # Execute handler with timeout
            response = await asyncio.wait_for(
                endpoint.handler(request, user),
                timeout=endpoint.timeout
            )
            
            # Ensure response has request ID
            if not response.request_id:
                response.request_id = request.request_id
            
            self.metrics.successful_requests += 1
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Endpoint timeout: {endpoint.path}")
            self.metrics.timeout_count += 1
            return create_api_response(
                status_code=504,
                message="Request timeout",
                errors=["The request took too long to process"],
                request_id=request.request_id
            )
        except Exception as e:
            logger.error(f"Endpoint execution error: {e}")
            self.metrics.failed_requests += 1
            return create_api_response(
                status_code=500,
                message="Internal server error",
                errors=["An unexpected error occurred during processing"],
                request_id=request.request_id
            )
    
    async def _process_response_middleware(
        self,
        request: APIRequest,
        response: APIResponse
    ) -> APIResponse:
        """Process response through middleware pipeline"""
        for middleware in reversed(self.middleware):
            response = await middleware.process_response(request, response)
        
        # Log request
        self.request_log.append({
            "request_id": request.request_id,
            "method": request.method,
            "path": request.path,
            "status_code": response.status_code,
            "timestamp": request.timestamp.isoformat(),
            "duration": request.get_duration(),
            "user_id": request.user_id,
            "ip_address": request.ip_address
        })
        
        return response
    
    def _update_request_metrics(self, request: APIRequest, endpoint: APIEndpoint):
        """Update request metrics"""
        self.metrics.total_requests += 1
        self.metrics.update_endpoint_usage(endpoint.path)
        self.metrics.update_method_distribution(request.method)
        
        # Update user count
        if request.user_id:
            self.metrics.active_users = len(self.auth_manager.list_active_users())
    
    async def _handle_not_found(self, request: APIRequest) -> APIResponse:
        """Handle 404 Not Found"""
        if 404 in self.error_handlers:
            return await self.error_handlers[404](request)
        
        return create_api_response(
            status_code=404,
            message="Endpoint not found",
            errors=["The requested endpoint does not exist"],
            request_id=request.request_id
        )
    
    async def _handle_unauthorized(self, request: APIRequest) -> APIResponse:
        """Handle 401 Unauthorized"""
        if 401 in self.error_handlers:
            return await self.error_handlers[401](request)
        
        return create_api_response(
            status_code=401,
            message="Authentication required",
            errors=["Valid authentication credentials are required"],
            request_id=request.request_id
        )
    
    async def _handle_forbidden(self, request: APIRequest) -> APIResponse:
        """Handle 403 Forbidden"""
        if 403 in self.error_handlers:
            return await self.error_handlers[403](request)
        
        return create_api_response(
            status_code=403,
            message="Insufficient permissions",
            errors=["You do not have permission to access this resource"],
            request_id=request.request_id
        )
    
    async def _handle_rate_limited(self, request: APIRequest, statuses: List) -> APIResponse:
        """Handle 429 Rate Limited"""
        if 429 in self.error_handlers:
            return await self.error_handlers[429](request)
        
        self.metrics.rate_limit_hits += 1
        
        # Get rate limit info from first status
        rate_info = statuses[0] if statuses else None
        metadata = {}
        if rate_info:
            metadata = {
                "retry_after": 60,
                "limit": rate_info.limit,
                "remaining": rate_info.remaining_requests,
                "reset_time": rate_info.reset_time.isoformat()
            }
        
        return create_api_response(
            status_code=429,
            message="Rate limit exceeded",
            errors=["Too many requests - please try again later"],
            metadata=metadata,
            request_id=request.request_id
        )
    
    async def _handle_internal_error(self, request: APIRequest, error: Exception) -> APIResponse:
        """Handle 500 Internal Server Error"""
        if 500 in self.error_handlers:
            return await self.error_handlers[500](request)
        
        self.metrics.failed_requests += 1
        
        return create_api_response(
            status_code=500,
            message="Internal server error",
            errors=["An unexpected error occurred"],
            request_id=request.request_id
        )
    
    def _register_core_endpoints(self):
        """Register core API endpoints"""
        # Health check endpoint
        self.register_endpoint(APIEndpoint(
            path="/health",
            method=HTTPMethod.GET,
            handler=self._handle_health_check,
            auth_required=AuthenticationLevel.NONE,
            description="API health check",
            responses={200: "API is healthy"},
            tags=["System"]
        ))
        
        # Status endpoint
        self.register_endpoint(APIEndpoint(
            path="/status",
            method=HTTPMethod.GET,
            handler=self._handle_status,
            auth_required=AuthenticationLevel.API_KEY,
            description="Get system status and metrics",
            responses={200: "System status retrieved"},
            tags=["System"]
        ))
        
        # OpenAPI spec endpoint
        self.register_endpoint(APIEndpoint(
            path="/openapi.json",
            method=HTTPMethod.GET,
            handler=self._handle_openapi_spec,
            auth_required=AuthenticationLevel.NONE,
            description="Get OpenAPI specification",
            responses={200: "OpenAPI spec retrieved"},
            tags=["Documentation"]
        ))
        
        # Metrics endpoint
        self.register_endpoint(APIEndpoint(
            path="/metrics",
            method=HTTPMethod.GET,
            handler=self._handle_metrics,
            auth_required=AuthenticationLevel.API_KEY,
            description="Get API metrics",
            responses={200: "Metrics retrieved"},
            tags=["Monitoring"]
        ))
    
    async def _handle_health_check(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle health check request"""
        return create_api_response(
            status_code=200,
            data={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "uptime": time.time()
            },
            message="API is healthy"
        )
    
    async def _handle_status(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle status request"""
        auth_stats = self.auth_manager.get_statistics()
        rate_stats = self.rate_limiter.get_statistics()
        
        return create_api_response(
            status_code=200,
            data={
                "api_gateway": {
                    "status": "running",
                    "endpoints_registered": len(self.endpoints),
                    "requests_processed": self.metrics.total_requests,
                    "success_rate": (
                        self.metrics.successful_requests / max(self.metrics.total_requests, 1)
                    )
                },
                "authentication": auth_stats,
                "rate_limiting": rate_stats,
                "middleware_count": len(self.middleware)
            },
            message="System status retrieved"
        )
    
    async def _handle_openapi_spec(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle OpenAPI specification request"""
        spec = self.openapi_generator.get_spec()
        return create_api_response(
            status_code=200,
            data=spec,
            message="OpenAPI specification retrieved"
        )
    
    async def _handle_metrics(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle metrics request"""
        # Update calculated metrics
        self.metrics.calculate_error_rate()
        
        return create_api_response(
            status_code=200,
            data={
                "api_metrics": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "error_rate": self.metrics.error_rate,
                    "timeout_count": self.metrics.timeout_count,
                    "rate_limit_hits": self.metrics.rate_limit_hits
                },
                "endpoint_usage": dict(self.metrics.endpoint_usage),
                "method_distribution": dict(self.metrics.method_distribution),
                "active_users": self.metrics.active_users
            },
            message="API metrics retrieved"
        )
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """Get OpenAPI specification"""
        return self.openapi_generator.get_spec()
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get comprehensive API statistics"""
        return {
            "gateway_info": {
                "host": self.host,
                "port": self.port,
                "is_running": self.is_running,
                "endpoints_count": len(self.endpoints),
                "middleware_count": len(self.middleware)
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": (
                    self.metrics.successful_requests / max(self.metrics.total_requests, 1)
                ),
                "error_rate": self.metrics.error_rate,
                "average_response_time": self.metrics.avg_response_time
            },
            "authentication": self.auth_manager.get_statistics(),
            "rate_limiting": self.rate_limiter.get_statistics(),
            "documentation": self.openapi_generator.get_statistics()
        }
    
    def cleanup(self):
        """Cleanup gateway resources"""
        logger.info("Cleaning up API Gateway resources")
        self.auth_manager.cleanup_expired_sessions()
        self.rate_limiter.cleanup_expired_data()


# Factory function
def create_api_gateway(
    host: str = "0.0.0.0",
    port: int = 8000
) -> TestMasterAPIGateway:
    """
    Create and configure API gateway
    
    Args:
        host: Gateway host
        port: Gateway port
        
    Returns:
        Configured API gateway
    """
    return TestMasterAPIGateway(host=host, port=port)