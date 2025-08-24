"""
TestMaster API Gateway - RESTful API endpoints for all TestMaster capabilities

This gateway provides:
- RESTful API endpoints for all TestMaster modules
- Authentication and authorization
- Rate limiting and security
- API documentation and OpenAPI spec
- Integration with external systems
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
import uuid
import hashlib
import time
from pathlib import Path
import jwt
import secrets

# Mock Framework Imports for Testing
import pytest
from unittest.mock import Mock, patch, MagicMock
import unittest

class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

class AuthenticationLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"

class RateLimitType(Enum):
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"

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

@dataclass
class APIResponse:
    """API response"""
    status_code: int
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AuthenticationManager:
    """Manages API authentication and authorization"""
    
    def __init__(self):
        self.users: Dict[str, APIUser] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.jwt_secret = secrets.token_urlsafe(32)
        
    def create_user(self, username: str, email: str, permissions: List[str]) -> APIUser:
        """Create new API user"""
        user_id = str(uuid.uuid4())
        api_key = secrets.token_urlsafe(32)
        jwt_secret = secrets.token_urlsafe(32)
        
        user = APIUser(
            user_id=user_id,
            username=username,
            email=email,
            api_key=api_key,
            jwt_secret=jwt_secret,
            permissions=permissions
        )
        
        self.users[user_id] = user
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[APIUser]:
        """Authenticate using API key"""
        for user in self.users.values():
            if user.api_key == api_key and user.is_active:
                user.last_access = datetime.now()
                return user
        return None
    
    def authenticate_jwt(self, token: str) -> Optional[APIUser]:
        """Authenticate using JWT token"""
        try:
            # Decode JWT without verification first to get user_id
            unverified = jwt.decode(token, options={"verify_signature": False})
            user_id = unverified.get("user_id")
            
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            
            # Verify with user's secret
            payload = jwt.decode(token, user.jwt_secret, algorithms=["HS256"])
            
            if payload.get("user_id") == user_id:
                user.last_access = datetime.now()
                return user
                
        except jwt.InvalidTokenError:
            return None
        
        return None
    
    def generate_jwt_token(self, user_id: str, expires_in_hours: int = 24) -> Optional[str]:
        """Generate JWT token for user"""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        payload = {
            "user_id": user_id,
            "username": user.username,
            "permissions": user.permissions,
            "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, user.jwt_secret, algorithm="HS256")
        return token
    
    def check_permission(self, user: APIUser, required_permission: str) -> bool:
        """Check if user has required permission"""
        return "admin" in user.permissions or required_permission in user.permissions
    
    def revoke_user_access(self, user_id: str) -> bool:
        """Revoke user access"""
        if user_id in self.users:
            self.users[user_id].is_active = False
            return True
        return False

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.request_counts: Dict[str, Dict[str, List[datetime]]] = {}
        
    def check_rate_limit(self, user_id: str, endpoint: str, limit: int, 
                        limit_type: RateLimitType) -> bool:
        """Check if request is within rate limit"""
        now = datetime.now()
        
        if user_id not in self.request_counts:
            self.request_counts[user_id] = {}
        
        if endpoint not in self.request_counts[user_id]:
            self.request_counts[user_id][endpoint] = []
        
        requests = self.request_counts[user_id][endpoint]
        
        # Determine time window
        if limit_type == RateLimitType.PER_SECOND:
            window = timedelta(seconds=1)
        elif limit_type == RateLimitType.PER_MINUTE:
            window = timedelta(minutes=1)
        elif limit_type == RateLimitType.PER_HOUR:
            window = timedelta(hours=1)
        elif limit_type == RateLimitType.PER_DAY:
            window = timedelta(days=1)
        else:
            return True  # No limit
        
        # Clean old requests
        cutoff = now - window
        requests[:] = [req_time for req_time in requests if req_time > cutoff]
        
        # Check limit
        if len(requests) >= limit:
            return False
        
        # Add current request
        requests.append(now)
        return True
    
    def get_remaining_requests(self, user_id: str, endpoint: str, limit: int, 
                             limit_type: RateLimitType) -> int:
        """Get remaining requests in current window"""
        if user_id not in self.request_counts:
            return limit
        
        if endpoint not in self.request_counts[user_id]:
            return limit
        
        now = datetime.now()
        requests = self.request_counts[user_id][endpoint]
        
        # Determine time window
        if limit_type == RateLimitType.PER_SECOND:
            window = timedelta(seconds=1)
        elif limit_type == RateLimitType.PER_MINUTE:
            window = timedelta(minutes=1)
        elif limit_type == RateLimitType.PER_HOUR:
            window = timedelta(hours=1)
        elif limit_type == RateLimitType.PER_DAY:
            window = timedelta(days=1)
        else:
            return limit
        
        cutoff = now - window
        recent_requests = [req_time for req_time in requests if req_time > cutoff]
        
        return max(0, limit - len(recent_requests))

class OpenAPIGenerator:
    """Generates OpenAPI specification for the API"""
    
    def __init__(self):
        self.spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "TestMaster API",
                "description": "Comprehensive testing framework API with AI-powered capabilities",
                "version": "1.0.0",
                "contact": {
                    "name": "TestMaster Support",
                    "email": "support@testmaster.dev"
                }
            },
            "servers": [
                {
                    "url": "https://api.testmaster.dev/v1",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.testmaster.dev/v1",
                    "description": "Staging server"
                }
            ],
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    },
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                },
                "schemas": {},
                "responses": {}
            },
            "paths": {},
            "tags": []
        }
    
    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        """Add endpoint to OpenAPI spec"""
        if endpoint.path not in self.spec["paths"]:
            self.spec["paths"][endpoint.path] = {}
        
        method_spec = {
            "summary": endpoint.description,
            "description": endpoint.description,
            "tags": endpoint.tags,
            "parameters": self._convert_parameters(endpoint.parameters),
            "responses": self._convert_responses(endpoint.responses),
            "security": self._get_security_scheme(endpoint.auth_required)
        }
        
        if endpoint.method == HTTPMethod.POST:
            method_spec["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            }
        
        self.spec["paths"][endpoint.path][endpoint.method.value.lower()] = method_spec
    
    def _convert_parameters(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert parameters to OpenAPI format"""
        converted = []
        for name, param_info in parameters.items():
            param_spec = {
                "name": name,
                "in": param_info.get("in", "query"),
                "required": param_info.get("required", False),
                "schema": {
                    "type": param_info.get("type", "string")
                }
            }
            if "description" in param_info:
                param_spec["description"] = param_info["description"]
            converted.append(param_spec)
        return converted
    
    def _convert_responses(self, responses: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
        """Convert responses to OpenAPI format"""
        converted = {}
        for status_code, description in responses.items():
            converted[str(status_code)] = {
                "description": description,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status_code": {"type": "integer"},
                                "data": {"type": "object"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            }
        return converted
    
    def _get_security_scheme(self, auth_level: AuthenticationLevel) -> List[Dict[str, Any]]:
        """Get security scheme for auth level"""
        if auth_level == AuthenticationLevel.NONE:
            return []
        elif auth_level == AuthenticationLevel.API_KEY:
            return [{"ApiKeyAuth": []}]
        elif auth_level == AuthenticationLevel.JWT:
            return [{"BearerAuth": []}]
        else:
            return [{"ApiKeyAuth": []}]
    
    def get_spec(self) -> Dict[str, Any]:
        """Get complete OpenAPI specification"""
        return self.spec

class TestMasterAPIGateway:
    """Main API Gateway for TestMaster framework"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.openapi_generator = OpenAPIGenerator()
        
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.request_log: List[APIRequest] = []
        self.is_running = False
        
        # Initialize core endpoints
        self._register_core_endpoints()
        self._register_testing_endpoints()
        self._register_collaboration_endpoints()
        self._register_analytics_endpoints()
        
    def _register_core_endpoints(self) -> None:
        """Register core API endpoints"""
        
        # Authentication endpoints
        self.register_endpoint(APIEndpoint(
            path="/auth/login",
            method=HTTPMethod.POST,
            handler=self._handle_login,
            auth_required=AuthenticationLevel.NONE,
            description="Authenticate user and get JWT token",
            parameters={
                "username": {"type": "string", "required": True, "in": "body"},
                "password": {"type": "string", "required": True, "in": "body"}
            },
            responses={200: "Login successful", 401: "Invalid credentials"},
            tags=["Authentication"]
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/auth/refresh",
            method=HTTPMethod.POST,
            handler=self._handle_refresh_token,
            auth_required=AuthenticationLevel.JWT,
            description="Refresh JWT token",
            responses={200: "Token refreshed", 401: "Invalid token"},
            tags=["Authentication"]
        ))
        
        # Health check endpoints
        self.register_endpoint(APIEndpoint(
            path="/health",
            method=HTTPMethod.GET,
            handler=self._handle_health_check,
            auth_required=AuthenticationLevel.NONE,
            description="API health check",
            responses={200: "API is healthy"},
            tags=["System"]
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/status",
            method=HTTPMethod.GET,
            handler=self._handle_status,
            auth_required=AuthenticationLevel.API_KEY,
            description="Get system status and metrics",
            responses={200: "System status retrieved"},
            tags=["System"]
        ))
    
    def _register_testing_endpoints(self) -> None:
        """Register testing-related endpoints"""
        
        # Test generation endpoints
        self.register_endpoint(APIEndpoint(
            path="/testing/generate",
            method=HTTPMethod.POST,
            handler=self._handle_test_generation,
            auth_required=AuthenticationLevel.API_KEY,
            rate_limit=(10, RateLimitType.PER_MINUTE),
            description="Generate AI-powered tests for code",
            parameters={
                "repository": {"type": "string", "required": True, "in": "body"},
                "file_path": {"type": "string", "required": True, "in": "body"},
                "test_type": {"type": "string", "required": False, "in": "body"}
            },
            responses={200: "Tests generated successfully", 400: "Invalid request"},
            tags=["Testing", "AI Generation"]
        ))
        
        # Test execution endpoints
        self.register_endpoint(APIEndpoint(
            path="/testing/execute",
            method=HTTPMethod.POST,
            handler=self._handle_test_execution,
            auth_required=AuthenticationLevel.API_KEY,
            rate_limit=(5, RateLimitType.PER_MINUTE),
            description="Execute test suite",
            parameters={
                "repository": {"type": "string", "required": True, "in": "body"},
                "test_suite": {"type": "string", "required": True, "in": "body"},
                "environment": {"type": "string", "required": False, "in": "body"}
            },
            responses={200: "Test execution started", 400: "Invalid request"},
            tags=["Testing", "Execution"]
        ))
        
        # Test analytics endpoints
        self.register_endpoint(APIEndpoint(
            path="/testing/analytics/{repository}",
            method=HTTPMethod.GET,
            handler=self._handle_test_analytics,
            auth_required=AuthenticationLevel.API_KEY,
            description="Get test analytics for repository",
            parameters={
                "repository": {"type": "string", "required": True, "in": "path"},
                "time_period": {"type": "string", "required": False, "in": "query"}
            },
            responses={200: "Analytics retrieved", 404: "Repository not found"},
            tags=["Testing", "Analytics"]
        ))
    
    def _register_collaboration_endpoints(self) -> None:
        """Register collaboration endpoints"""
        
        # Knowledge base endpoints
        self.register_endpoint(APIEndpoint(
            path="/knowledge/articles",
            method=HTTPMethod.GET,
            handler=self._handle_get_articles,
            auth_required=AuthenticationLevel.API_KEY,
            description="Get knowledge base articles",
            parameters={
                "category": {"type": "string", "required": False, "in": "query"},
                "search": {"type": "string", "required": False, "in": "query"},
                "limit": {"type": "integer", "required": False, "in": "query"}
            },
            responses={200: "Articles retrieved"},
            tags=["Knowledge Base"]
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/knowledge/articles",
            method=HTTPMethod.POST,
            handler=self._handle_create_article,
            auth_required=AuthenticationLevel.API_KEY,
            description="Create knowledge base article",
            parameters={
                "title": {"type": "string", "required": True, "in": "body"},
                "content": {"type": "string", "required": True, "in": "body"},
                "category": {"type": "string", "required": True, "in": "body"},
                "tags": {"type": "array", "required": False, "in": "body"}
            },
            responses={201: "Article created", 400: "Invalid request"},
            tags=["Knowledge Base"]
        ))
        
        # Code review endpoints
        self.register_endpoint(APIEndpoint(
            path="/reviews",
            method=HTTPMethod.POST,
            handler=self._handle_create_review,
            auth_required=AuthenticationLevel.API_KEY,
            description="Create code review request",
            parameters={
                "test_file": {"type": "string", "required": True, "in": "body"},
                "repository": {"type": "string", "required": True, "in": "body"},
                "reviewers": {"type": "array", "required": True, "in": "body"}
            },
            responses={201: "Review created", 400: "Invalid request"},
            tags=["Code Review"]
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/reviews/{review_id}/comments",
            method=HTTPMethod.POST,
            handler=self._handle_add_review_comment,
            auth_required=AuthenticationLevel.API_KEY,
            description="Add comment to code review",
            parameters={
                "review_id": {"type": "string", "required": True, "in": "path"},
                "line_number": {"type": "integer", "required": True, "in": "body"},
                "comment": {"type": "string", "required": True, "in": "body"}
            },
            responses={201: "Comment added", 404: "Review not found"},
            tags=["Code Review"]
        ))
    
    def _register_analytics_endpoints(self) -> None:
        """Register analytics and reporting endpoints"""
        
        self.register_endpoint(APIEndpoint(
            path="/analytics/team",
            method=HTTPMethod.GET,
            handler=self._handle_team_analytics,
            auth_required=AuthenticationLevel.API_KEY,
            description="Get team performance analytics",
            parameters={
                "time_period": {"type": "string", "required": False, "in": "query"}
            },
            responses={200: "Team analytics retrieved"},
            tags=["Analytics"]
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/analytics/repositories",
            method=HTTPMethod.GET,
            handler=self._handle_repository_analytics,
            auth_required=AuthenticationLevel.API_KEY,
            description="Get cross-repository analytics",
            responses={200: "Repository analytics retrieved"},
            tags=["Analytics"]
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/reports/executive",
            method=HTTPMethod.GET,
            handler=self._handle_executive_report,
            auth_required=AuthenticationLevel.API_KEY,
            description="Generate executive summary report",
            parameters={
                "time_period": {"type": "string", "required": False, "in": "query"},
                "format": {"type": "string", "required": False, "in": "query"}
            },
            responses={200: "Executive report generated"},
            tags=["Reports"]
        ))
    
    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """Register new API endpoint"""
        endpoint_key = f"{endpoint.method.value}:{endpoint.path}"
        self.endpoints[endpoint_key] = endpoint
        self.openapi_generator.add_endpoint(endpoint)
    
    async def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle incoming API request"""
        try:
            # Find matching endpoint
            endpoint_key = f"{request.method}:{request.path}"
            endpoint = self.endpoints.get(endpoint_key)
            
            if not endpoint:
                return APIResponse(
                    status_code=404,
                    message="Endpoint not found",
                    errors=["The requested endpoint does not exist"]
                )
            
            # Authenticate request
            user = await self._authenticate_request(request, endpoint.auth_required)
            if endpoint.auth_required != AuthenticationLevel.NONE and not user:
                return APIResponse(
                    status_code=401,
                    message="Authentication required",
                    errors=["Valid authentication credentials are required"]
                )
            
            # Check rate limits
            if endpoint.rate_limit and user:
                limit, limit_type = endpoint.rate_limit
                if not self.rate_limiter.check_rate_limit(user.user_id, endpoint.path, limit, limit_type):
                    remaining = self.rate_limiter.get_remaining_requests(
                        user.user_id, endpoint.path, limit, limit_type
                    )
                    return APIResponse(
                        status_code=429,
                        message="Rate limit exceeded",
                        errors=["Too many requests"],
                        metadata={"remaining_requests": remaining, "retry_after": 60}
                    )
            
            # Log request
            self.request_log.append(request)
            if len(self.request_log) > 10000:
                self.request_log = self.request_log[-5000:]  # Trim log
            
            # Handle request
            response = await endpoint.handler(request, user)
            return response
            
        except Exception as e:
            logging.error(f"Request handling error: {e}")
            return APIResponse(
                status_code=500,
                message="Internal server error",
                errors=["An unexpected error occurred"]
            )
    
    async def _authenticate_request(self, request: APIRequest, 
                                  auth_level: AuthenticationLevel) -> Optional[APIUser]:
        """Authenticate API request"""
        if auth_level == AuthenticationLevel.NONE:
            return None
        
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return self.auth_manager.authenticate_api_key(api_key)
        
        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            return self.auth_manager.authenticate_jwt(token)
        
        return None
    
    # Request handlers
    async def _handle_login(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle user login"""
        # Mock authentication - in real implementation, verify credentials
        username = request.body.get("username") if request.body else None
        password = request.body.get("password") if request.body else None
        
        if not username or not password:
            return APIResponse(
                status_code=400,
                message="Username and password are required",
                errors=["Missing credentials"]
            )
        
        # Find or create user (mock implementation)
        existing_user = None
        for user_obj in self.auth_manager.users.values():
            if user_obj.username == username:
                existing_user = user_obj
                break
        
        if not existing_user:
            # Create new user for demo
            existing_user = self.auth_manager.create_user(
                username, f"{username}@example.com", ["read", "write"]
            )
        
        # Generate JWT token
        token = self.auth_manager.generate_jwt_token(existing_user.user_id)
        
        return APIResponse(
            status_code=200,
            data={
                "token": token,
                "user_id": existing_user.user_id,
                "username": existing_user.username,
                "permissions": existing_user.permissions
            },
            message="Login successful"
        )
    
    async def _handle_refresh_token(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle token refresh"""
        if not user:
            return APIResponse(status_code=401, message="Invalid token")
        
        new_token = self.auth_manager.generate_jwt_token(user.user_id)
        
        return APIResponse(
            status_code=200,
            data={"token": new_token},
            message="Token refreshed"
        )
    
    async def _handle_health_check(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle health check"""
        return APIResponse(
            status_code=200,
            data={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "uptime": time.time()  # Mock uptime
            },
            message="API is healthy"
        )
    
    async def _handle_status(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle system status request"""
        return APIResponse(
            status_code=200,
            data={
                "api_gateway": {
                    "status": "running",
                    "registered_endpoints": len(self.endpoints),
                    "active_users": len([u for u in self.auth_manager.users.values() if u.is_active]),
                    "requests_processed": len(self.request_log)
                },
                "testing_framework": {
                    "modules_loaded": 49,  # Total TestMaster modules
                    "test_executions": 156,  # Mock data
                    "success_rate": 0.94
                },
                "collaboration_platform": {
                    "active_sessions": 3,
                    "knowledge_articles": 25,
                    "code_reviews": 8
                }
            },
            message="System status retrieved"
        )
    
    async def _handle_test_generation(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle test generation request"""
        if not request.body:
            return APIResponse(
                status_code=400, 
                message="Request body required",
                errors=["Missing request parameters"]
            )
        
        repository = request.body.get("repository")
        file_path = request.body.get("file_path")
        
        if not repository or not file_path:
            return APIResponse(
                status_code=400,
                message="Repository and file_path are required",
                errors=["Missing required parameters"]
            )
        
        # Mock test generation
        generated_tests = [
            {
                "test_name": f"test_{file_path.replace('/', '_').replace('.py', '')}_functionality",
                "test_code": f"def test_{file_path.replace('/', '_').replace('.py', '')}_basic():\n    # Generated test\n    assert True",
                "complexity": "medium",
                "coverage_improvement": 0.15
            },
            {
                "test_name": f"test_{file_path.replace('/', '_').replace('.py', '')}_edge_cases",
                "test_code": f"def test_{file_path.replace('/', '_').replace('.py', '')}_edge_cases():\n    # Generated edge case test\n    assert True",
                "complexity": "high",
                "coverage_improvement": 0.08
            }
        ]
        
        return APIResponse(
            status_code=200,
            data={
                "repository": repository,
                "file_path": file_path,
                "tests_generated": len(generated_tests),
                "tests": generated_tests,
                "estimated_coverage_improvement": 0.23,
                "generation_time": 2.5
            },
            message="Tests generated successfully"
        )
    
    async def _handle_test_execution(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle test execution request"""
        if not request.body:
            return APIResponse(status_code=400, message="Request body required")
        
        repository = request.body.get("repository")
        test_suite = request.body.get("test_suite")
        
        if not repository or not test_suite:
            return APIResponse(
                status_code=400,
                message="Repository and test_suite are required"
            )
        
        # Mock test execution
        execution_id = str(uuid.uuid4())
        
        return APIResponse(
            status_code=200,
            data={
                "execution_id": execution_id,
                "repository": repository,
                "test_suite": test_suite,
                "status": "started",
                "estimated_duration": 300,
                "start_time": datetime.now().isoformat()
            },
            message="Test execution started"
        )
    
    async def _handle_test_analytics(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle test analytics request"""
        repository = request.path.split('/')[-1]  # Extract from path parameter
        
        # Mock analytics data
        analytics = {
            "repository": repository,
            "test_metrics": {
                "total_tests": 234,
                "passing_tests": 218,
                "failing_tests": 16,
                "success_rate": 0.932,
                "coverage": 0.847
            },
            "performance_metrics": {
                "avg_execution_time": 45.6,
                "slowest_test": "test_integration_complex_workflow",
                "fastest_test": "test_unit_simple_validation"
            },
            "trends": {
                "success_rate_trend": [0.915, 0.923, 0.930, 0.932],
                "coverage_trend": [0.821, 0.834, 0.841, 0.847]
            }
        }
        
        return APIResponse(
            status_code=200,
            data=analytics,
            message="Test analytics retrieved"
        )
    
    async def _handle_get_articles(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle get knowledge articles request"""
        # Mock knowledge articles
        articles = [
            {
                "id": "article_001",
                "title": "Advanced Pytest Techniques",
                "category": "Testing Best Practices",
                "author": "senior_dev",
                "views": 156,
                "likes": 23,
                "created_at": "2024-01-15T10:00:00Z"
            },
            {
                "id": "article_002", 
                "title": "Cross-Repository Testing Strategies",
                "category": "Testing Architecture",
                "author": "test_architect",
                "views": 89,
                "likes": 15,
                "created_at": "2024-01-20T14:30:00Z"
            }
        ]
        
        return APIResponse(
            status_code=200,
            data={"articles": articles, "total": len(articles)},
            message="Articles retrieved"
        )
    
    async def _handle_create_article(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle create knowledge article request"""
        if not request.body:
            return APIResponse(status_code=400, message="Request body required")
        
        title = request.body.get("title")
        content = request.body.get("content")
        category = request.body.get("category")
        
        if not all([title, content, category]):
            return APIResponse(
                status_code=400,
                message="Title, content, and category are required"
            )
        
        article_id = str(uuid.uuid4())
        
        return APIResponse(
            status_code=201,
            data={
                "article_id": article_id,
                "title": title,
                "category": category,
                "author": user.username if user else "anonymous",
                "created_at": datetime.now().isoformat()
            },
            message="Article created successfully"
        )
    
    async def _handle_create_review(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle create code review request"""
        if not request.body:
            return APIResponse(status_code=400, message="Request body required")
        
        test_file = request.body.get("test_file")
        repository = request.body.get("repository")
        reviewers = request.body.get("reviewers", [])
        
        if not all([test_file, repository]) or not reviewers:
            return APIResponse(
                status_code=400,
                message="test_file, repository, and reviewers are required"
            )
        
        review_id = str(uuid.uuid4())
        
        return APIResponse(
            status_code=201,
            data={
                "review_id": review_id,
                "test_file": test_file,
                "repository": repository,
                "author": user.username if user else "anonymous",
                "reviewers": reviewers,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            },
            message="Review created successfully"
        )
    
    async def _handle_add_review_comment(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle add review comment request"""
        review_id = request.path.split('/')[-2]  # Extract from path
        
        if not request.body:
            return APIResponse(status_code=400, message="Request body required")
        
        line_number = request.body.get("line_number")
        comment = request.body.get("comment")
        
        if line_number is None or not comment:
            return APIResponse(
                status_code=400,
                message="line_number and comment are required"
            )
        
        comment_id = str(uuid.uuid4())
        
        return APIResponse(
            status_code=201,
            data={
                "comment_id": comment_id,
                "review_id": review_id,
                "line_number": line_number,
                "comment": comment,
                "reviewer": user.username if user else "anonymous",
                "timestamp": datetime.now().isoformat()
            },
            message="Comment added successfully"
        )
    
    async def _handle_team_analytics(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle team analytics request"""
        # Mock team analytics
        analytics = {
            "team_size": 12,
            "active_members": 10,
            "performance_summary": {
                "total_tests_created": 1456,
                "total_reviews_completed": 89,
                "knowledge_articles_shared": 25,
                "collaboration_sessions": 34
            },
            "top_performers": [
                {"username": "senior_dev", "score": 95.2},
                {"username": "test_lead", "score": 91.8},
                {"username": "full_stack_dev", "score": 87.4}
            ],
            "activity_trends": {
                "test_creation": [23, 28, 31, 29, 35],
                "code_reviews": [12, 15, 18, 16, 19],
                "collaboration": [8, 10, 12, 14, 11]
            }
        }
        
        return APIResponse(
            status_code=200,
            data=analytics,
            message="Team analytics retrieved"
        )
    
    async def _handle_repository_analytics(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle repository analytics request"""
        # Mock repository analytics
        analytics = {
            "total_repositories": 7,
            "repositories": [
                {"name": "agency_swarm", "test_coverage": 0.89, "maturity_score": 0.76},
                {"name": "crew_ai", "test_coverage": 0.82, "maturity_score": 0.71},
                {"name": "agent_scope", "test_coverage": 0.94, "maturity_score": 0.83}
            ],
            "cross_repo_insights": [
                {
                    "title": "Framework Standardization Opportunity",
                    "impact": "MEDIUM",
                    "affected_repos": ["agency_swarm", "crew_ai"]
                }
            ],
            "aggregate_metrics": {
                "avg_test_coverage": 0.86,
                "avg_maturity_score": 0.77,
                "total_tests": 3421,
                "total_patterns_extracted": 167
            }
        }
        
        return APIResponse(
            status_code=200,
            data=analytics,
            message="Repository analytics retrieved"
        )
    
    async def _handle_executive_report(self, request: APIRequest, user: Optional[APIUser]) -> APIResponse:
        """Handle executive report request"""
        # Mock executive report
        report = {
            "report_period": "Last 30 Days",
            "executive_summary": {
                "total_test_executions": 2847,
                "success_rate": 0.943,
                "cost_savings": "$12,450",
                "productivity_increase": "23%"
            },
            "key_metrics": {
                "test_automation_coverage": 0.87,
                "defect_detection_rate": 0.92,
                "time_to_resolution": "4.2 hours",
                "team_efficiency": 0.89
            },
            "recommendations": [
                "Increase test coverage in low-coverage repositories",
                "Implement additional automated testing in CI/CD pipeline",
                "Expand knowledge sharing initiatives"
            ],
            "roi_analysis": {
                "testing_investment": "$45,000",
                "defects_prevented": 156,
                "estimated_savings": "$234,000",
                "roi_ratio": 5.2
            }
        }
        
        return APIResponse(
            status_code=200,
            data=report,
            message="Executive report generated"
        )
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """Get OpenAPI specification"""
        return self.openapi_generator.get_spec()
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        total_requests = len(self.request_log)
        
        # Analyze request patterns
        method_counts = {}
        endpoint_counts = {}
        
        for request in self.request_log:
            method_counts[request.method] = method_counts.get(request.method, 0) + 1
            endpoint_counts[request.path] = endpoint_counts.get(request.path, 0) + 1
        
        return {
            "total_requests": total_requests,
            "registered_endpoints": len(self.endpoints),
            "active_users": len([u for u in self.auth_manager.users.values() if u.is_active]),
            "requests_by_method": method_counts,
            "popular_endpoints": dict(sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "api_health": "healthy"
        }


# Comprehensive Test Suite
class TestMasterAPIGateway(unittest.TestCase):
    
    def setUp(self):
        self.gateway = TestMasterAPIGateway()
        
        # Create test user
        self.test_user = self.gateway.auth_manager.create_user(
            "test_user", "test@example.com", ["read", "write"]
        )
        
    def test_gateway_initialization(self):
        """Test API gateway initialization"""
        self.assertIsNotNone(self.gateway.auth_manager)
        self.assertIsNotNone(self.gateway.rate_limiter)
        self.assertIsNotNone(self.gateway.openapi_generator)
        self.assertGreater(len(self.gateway.endpoints), 10)
        
    def test_user_authentication(self):
        """Test user authentication"""
        # Test API key authentication
        user = self.gateway.auth_manager.authenticate_api_key(self.test_user.api_key)
        self.assertIsNotNone(user)
        self.assertEqual(user.user_id, self.test_user.user_id)
        
        # Test JWT authentication
        token = self.gateway.auth_manager.generate_jwt_token(self.test_user.user_id)
        self.assertIsNotNone(token)
        
        authenticated_user = self.gateway.auth_manager.authenticate_jwt(token)
        self.assertIsNotNone(authenticated_user)
        self.assertEqual(authenticated_user.user_id, self.test_user.user_id)
        
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Test rate limit check
        can_proceed = self.gateway.rate_limiter.check_rate_limit(
            self.test_user.user_id, "/test", 5, RateLimitType.PER_MINUTE
        )
        self.assertTrue(can_proceed)
        
        # Simulate hitting rate limit
        for i in range(5):
            self.gateway.rate_limiter.check_rate_limit(
                self.test_user.user_id, "/test", 5, RateLimitType.PER_MINUTE
            )
        
        # Should be rate limited now
        exceeded = self.gateway.rate_limiter.check_rate_limit(
            self.test_user.user_id, "/test", 5, RateLimitType.PER_MINUTE
        )
        self.assertFalse(exceeded)
        
    async def test_request_handling(self):
        """Test API request handling"""
        # Create test request
        request = APIRequest(
            request_id="test_request",
            user_id=self.test_user.user_id,
            method="GET",
            path="/health",
            headers={"X-API-Key": self.test_user.api_key}
        )
        
        # Handle request
        response = await self.gateway.handle_request(request)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.data)
        
    async def test_test_generation_endpoint(self):
        """Test test generation endpoint"""
        request = APIRequest(
            request_id="gen_test",
            user_id=self.test_user.user_id,
            method="POST", 
            path="/testing/generate",
            headers={"X-API-Key": self.test_user.api_key},
            body={
                "repository": "test_repo",
                "file_path": "src/utils.py",
                "test_type": "unit"
            }
        )
        
        response = await self.gateway.handle_request(request)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("tests_generated", response.data)
        self.assertGreater(response.data["tests_generated"], 0)
        
    def test_openapi_spec_generation(self):
        """Test OpenAPI spec generation"""
        spec = self.gateway.get_openapi_spec()
        
        self.assertIn("openapi", spec)
        self.assertIn("info", spec)
        self.assertIn("paths", spec)
        self.assertGreater(len(spec["paths"]), 5)
        
    def test_api_statistics(self):
        """Test API statistics"""
        stats = self.gateway.get_api_stats()
        
        self.assertIn("total_requests", stats)
        self.assertIn("registered_endpoints", stats)
        self.assertIn("active_users", stats)
        self.assertEqual(stats["api_health"], "healthy")


if __name__ == "__main__":
    # Demo usage
    async def demo_api_gateway():
        gateway = TestMasterAPIGateway()
        
        print("TestMaster API Gateway Demo")
        print(f"Registered endpoints: {len(gateway.endpoints)}")
        
        # Create demo user
        demo_user = gateway.auth_manager.create_user(
            "demo_user", "demo@testmaster.dev", ["read", "write", "admin"]
        )
        
        print(f"Created demo user: {demo_user.username}")
        print(f"API Key: {demo_user.api_key}")
        
        # Generate JWT token
        jwt_token = gateway.auth_manager.generate_jwt_token(demo_user.user_id)
        print(f"JWT Token: {jwt_token[:50]}...")
        
        # Demo API requests
        demo_requests = [
            APIRequest(
                request_id=str(uuid.uuid4()),
                user_id=demo_user.user_id,
                method="GET",
                path="/health",
                headers={"X-API-Key": demo_user.api_key}
            ),
            APIRequest(
                request_id=str(uuid.uuid4()),
                user_id=demo_user.user_id,
                method="POST",
                path="/testing/generate",
                headers={"X-API-Key": demo_user.api_key},
                body={
                    "repository": "testmaster_core",
                    "file_path": "core/testing/api_gateway.py",
                    "test_type": "integration"
                }
            ),
            APIRequest(
                request_id=str(uuid.uuid4()),
                user_id=demo_user.user_id,
                method="GET",
                path="/analytics/team",
                headers={"Authorization": f"Bearer {jwt_token}"}
            )
        ]
        
        # Process demo requests
        for request in demo_requests:
            response = await gateway.handle_request(request)
            print(f"\nRequest: {request.method} {request.path}")
            print(f"Response: {response.status_code} - {response.message}")
            if response.data:
                print(f"Data keys: {list(response.data.keys())}")
        
        # Get OpenAPI spec
        spec = gateway.get_openapi_spec()
        print(f"\nOpenAPI Spec generated: {len(spec['paths'])} endpoints")
        
        # Get API stats
        stats = gateway.get_api_stats()
        print(f"API Stats: {stats}")
        
        print("\nTestMaster API Gateway Demo Complete")
    
    # Run demo
    try:
        asyncio.run(demo_api_gateway())
    except KeyboardInterrupt:
        print("Demo interrupted")
    
    # Run tests
    pytest.main([__file__, "-v"])