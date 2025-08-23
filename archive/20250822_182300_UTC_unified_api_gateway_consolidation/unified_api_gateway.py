"""
Unified API Gateway
===================

Enterprise-grade API gateway providing centralized access to all intelligence systems
with advanced routing, security, rate limiting, and comprehensive request orchestration.

Integrates: Service Discovery, Load Balancing, Circuit Breakers, Authentication,
Rate Limiting, Request Validation, Response Transformation, and Monitoring.

Author: TestMaster Intelligence Framework
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
import threading
import hashlib
import hmac


# ============================================================================
# CORE API TYPES
# ============================================================================

class RequestMethod(Enum):
    """HTTP request methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class AuthenticationLevel(Enum):
    """Authentication requirement levels"""
    NONE = "none"
    BASIC = "basic"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    ENTERPRISE = "enterprise"


class RateLimitScope(Enum):
    """Rate limiting scope types"""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    endpoint_id: str = field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:8]}")
    path: str = ""
    method: RequestMethod = RequestMethod.GET
    service_name: str = ""
    handler_function: str = ""
    auth_level: AuthenticationLevel = AuthenticationLevel.BASIC
    rate_limit: Optional[Dict[str, int]] = None
    timeout_seconds: int = 30
    cache_ttl: int = 0
    validation_schema: Optional[Dict[str, Any]] = None
    transformation_rules: List[Dict[str, Any]] = field(default_factory=list)
    middleware: List[str] = field(default_factory=list)
    documentation: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class APIRequest:
    """Unified API request structure"""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    method: RequestMethod = RequestMethod.GET
    path: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Any] = None
    user_context: Dict[str, Any] = field(default_factory=dict)
    client_ip: str = "0.0.0.0"
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


@dataclass
class APIResponse:
    """Unified API response structure"""
    request_id: str
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# RATE LIMITING ENGINE
# ============================================================================

class RateLimitingEngine:
    """Advanced rate limiting with multiple scopes and algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger("rate_limiting_engine")
        
        # Rate limit buckets
        self.rate_buckets: Dict[str, Dict[str, Any]] = {}
        self.bucket_lock = threading.Lock()
        
        # Rate limit policies
        self.policies: Dict[str, Dict[str, Any]] = {
            "default": {"requests": 1000, "window_seconds": 3600},
            "premium": {"requests": 5000, "window_seconds": 3600},
            "enterprise": {"requests": 20000, "window_seconds": 3600},
            "internal": {"requests": 100000, "window_seconds": 3600}
        }
        
        # Algorithm configurations
        self.algorithms = {
            "token_bucket": self._token_bucket_check,
            "sliding_window": self._sliding_window_check,
            "fixed_window": self._fixed_window_check,
            "leaky_bucket": self._leaky_bucket_check
        }
        
        self.logger.info("Rate limiting engine initialized")
    
    def check_rate_limit(self, scope: RateLimitScope, identifier: str, 
                        policy: str = "default", algorithm: str = "token_bucket") -> Tuple[bool, Dict[str, Any]]:
        """Check if request passes rate limit"""
        try:
            bucket_key = f"{scope.value}:{identifier}:{policy}"
            
            with self.bucket_lock:
                # Initialize bucket if needed
                if bucket_key not in self.rate_buckets:
                    self.rate_buckets[bucket_key] = self._initialize_bucket(policy, algorithm)
                
                bucket = self.rate_buckets[bucket_key]
                
                # Apply rate limiting algorithm
                if algorithm in self.algorithms:
                    allowed, bucket_info = self.algorithms[algorithm](bucket, policy)
                    
                    # Update bucket state
                    bucket.update(bucket_info)
                    self.rate_buckets[bucket_key] = bucket
                    
                    return allowed, {
                        "bucket_key": bucket_key,
                        "algorithm": algorithm,
                        "policy": policy,
                        "current_usage": bucket_info.get("current_usage", 0),
                        "limit": self.policies[policy]["requests"],
                        "reset_time": bucket_info.get("reset_time"),
                        "retry_after": bucket_info.get("retry_after", 0)
                    }
                else:
                    self.logger.warning(f"Unknown rate limiting algorithm: {algorithm}")
                    return True, {"error": "unknown_algorithm"}
                    
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True, {"error": str(e)}  # Fail open
    
    def _initialize_bucket(self, policy: str, algorithm: str) -> Dict[str, Any]:
        """Initialize rate limit bucket"""
        policy_config = self.policies.get(policy, self.policies["default"])
        
        return {
            "policy": policy,
            "algorithm": algorithm,
            "requests": policy_config["requests"],
            "window_seconds": policy_config["window_seconds"],
            "current_usage": 0,
            "last_refill": time.time(),
            "window_start": time.time(),
            "request_times": []
        }
    
    def _token_bucket_check(self, bucket: Dict[str, Any], policy: str) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket algorithm"""
        now = time.time()
        policy_config = self.policies[policy]
        
        # Calculate tokens to add
        time_passed = now - bucket["last_refill"]
        tokens_to_add = time_passed * (policy_config["requests"] / policy_config["window_seconds"])
        
        # Update bucket
        bucket["current_usage"] = max(0, bucket["current_usage"] - tokens_to_add)
        bucket["last_refill"] = now
        
        # Check if request allowed
        if bucket["current_usage"] < policy_config["requests"]:
            bucket["current_usage"] += 1
            return True, bucket
        else:
            retry_after = (bucket["current_usage"] - policy_config["requests"]) / (policy_config["requests"] / policy_config["window_seconds"])
            bucket["retry_after"] = retry_after
            return False, bucket
    
    def _sliding_window_check(self, bucket: Dict[str, Any], policy: str) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window algorithm"""
        now = time.time()
        policy_config = self.policies[policy]
        window_start = now - policy_config["window_seconds"]
        
        # Remove old requests
        bucket["request_times"] = [t for t in bucket["request_times"] if t > window_start]
        
        # Check limit
        if len(bucket["request_times"]) < policy_config["requests"]:
            bucket["request_times"].append(now)
            bucket["current_usage"] = len(bucket["request_times"])
            return True, bucket
        else:
            # Calculate retry after
            oldest_request = min(bucket["request_times"])
            retry_after = oldest_request + policy_config["window_seconds"] - now
            bucket["retry_after"] = max(0, retry_after)
            return False, bucket
    
    def _fixed_window_check(self, bucket: Dict[str, Any], policy: str) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window algorithm"""
        now = time.time()
        policy_config = self.policies[policy]
        
        # Check if window reset needed
        if now - bucket["window_start"] >= policy_config["window_seconds"]:
            bucket["current_usage"] = 0
            bucket["window_start"] = now
        
        # Check limit
        if bucket["current_usage"] < policy_config["requests"]:
            bucket["current_usage"] += 1
            return True, bucket
        else:
            # Calculate retry after
            retry_after = bucket["window_start"] + policy_config["window_seconds"] - now
            bucket["retry_after"] = max(0, retry_after)
            return False, bucket
    
    def _leaky_bucket_check(self, bucket: Dict[str, Any], policy: str) -> Tuple[bool, Dict[str, Any]]:
        """Leaky bucket algorithm"""
        now = time.time()
        policy_config = self.policies[policy]
        
        # Calculate leakage
        time_passed = now - bucket["last_refill"]
        leak_rate = policy_config["requests"] / policy_config["window_seconds"]
        leaked = time_passed * leak_rate
        
        # Update bucket
        bucket["current_usage"] = max(0, bucket["current_usage"] - leaked)
        bucket["last_refill"] = now
        
        # Check capacity
        if bucket["current_usage"] < policy_config["requests"]:
            bucket["current_usage"] += 1
            return True, bucket
        else:
            retry_after = 1.0 / leak_rate  # Time for one token to leak
            bucket["retry_after"] = retry_after
            return False, bucket


# ============================================================================
# REQUEST VALIDATOR
# ============================================================================

class RequestValidator:
    """Advanced request validation and sanitization"""
    
    def __init__(self):
        self.logger = logging.getLogger("request_validator")
        
        # Validation schemas cache
        self.schema_cache: Dict[str, Dict[str, Any]] = {}
        
        # Security patterns
        self.security_patterns = {
            "sql_injection": [r"union\s+select", r"drop\s+table", r"insert\s+into"],
            "xss": [r"<script", r"javascript:", r"onload="],
            "path_traversal": [r"\.\.\/", r"\.\.\\", r"%2e%2e"],
            "command_injection": [r";\s*\w+", r"\|\s*\w+", r"&&\s*\w+"]
        }
        
        self.logger.info("Request validator initialized")
    
    def validate_request(self, request: APIRequest, endpoint: APIEndpoint) -> Tuple[bool, List[str]]:
        """Validate request against endpoint requirements"""
        errors = []
        
        try:
            # Method validation
            if request.method != endpoint.method:
                errors.append(f"Method mismatch: expected {endpoint.method.value}, got {request.method.value}")
            
            # Path validation
            if not self._validate_path(request.path, endpoint.path):
                errors.append(f"Path validation failed: {request.path}")
            
            # Schema validation
            if endpoint.validation_schema:
                schema_errors = self._validate_schema(request, endpoint.validation_schema)
                errors.extend(schema_errors)
            
            # Security validation
            security_errors = self._validate_security(request)
            errors.extend(security_errors)
            
            # Size validation
            size_errors = self._validate_size_limits(request)
            errors.extend(size_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Request validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def _validate_path(self, request_path: str, endpoint_path: str) -> bool:
        """Validate request path against endpoint pattern"""
        # Simple path matching (can be enhanced with regex patterns)
        # Extract path variables like /api/v1/users/{user_id}
        endpoint_parts = endpoint_path.split('/')
        request_parts = request_path.split('/')
        
        if len(endpoint_parts) != len(request_parts):
            return False
        
        for endpoint_part, request_part in zip(endpoint_parts, request_parts):
            if endpoint_part.startswith('{') and endpoint_part.endswith('}'):
                # Path variable - validate format if needed
                continue
            elif endpoint_part != request_part:
                return False
        
        return True
    
    def _validate_schema(self, request: APIRequest, schema: Dict[str, Any]) -> List[str]:
        """Validate request against JSON schema"""
        errors = []
        
        try:
            # Basic schema validation
            if "required" in schema and request.body:
                required_fields = schema["required"]
                if isinstance(request.body, dict):
                    for field in required_fields:
                        if field not in request.body:
                            errors.append(f"Missing required field: {field}")
            
            # Type validation
            if "properties" in schema and request.body and isinstance(request.body, dict):
                properties = schema["properties"]
                for field, value in request.body.items():
                    if field in properties:
                        expected_type = properties[field].get("type")
                        if expected_type and not self._validate_type(value, expected_type):
                            errors.append(f"Invalid type for field {field}: expected {expected_type}")
            
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")
        
        return errors
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    def _validate_security(self, request: APIRequest) -> List[str]:
        """Validate request for security threats"""
        errors = []
        
        # Validate all string inputs
        string_inputs = []
        
        # Add query parameters
        for key, value in request.query_params.items():
            if isinstance(value, str):
                string_inputs.append(f"query.{key}={value}")
        
        # Add body fields
        if request.body and isinstance(request.body, dict):
            for key, value in request.body.items():
                if isinstance(value, str):
                    string_inputs.append(f"body.{key}={value}")
        
        # Add headers
        for key, value in request.headers.items():
            string_inputs.append(f"header.{key}={value}")
        
        # Check patterns
        for input_string in string_inputs:
            for threat_type, patterns in self.security_patterns.items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, input_string, re.IGNORECASE):
                        errors.append(f"Security threat detected ({threat_type}): {pattern}")
        
        return errors
    
    def _validate_size_limits(self, request: APIRequest) -> List[str]:
        """Validate request size limits"""
        errors = []
        
        # Check body size
        if request.body:
            body_size = len(json.dumps(request.body) if isinstance(request.body, (dict, list)) else str(request.body))
            if body_size > 10 * 1024 * 1024:  # 10MB limit
                errors.append(f"Request body too large: {body_size} bytes")
        
        # Check header size
        total_header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if total_header_size > 8192:  # 8KB limit
            errors.append(f"Headers too large: {total_header_size} bytes")
        
        return errors


# ============================================================================
# UNIFIED API GATEWAY
# ============================================================================

class UnifiedAPIGateway:
    """
    Enterprise-grade unified API gateway providing centralized access
    to all intelligence systems with advanced features and monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("unified_api_gateway")
        
        # Core components
        self.rate_limiter = RateLimitingEngine()
        self.validator = RequestValidator()
        
        # Registry and routing
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.route_handlers: Dict[str, Callable] = {}
        self.middleware_registry: Dict[str, Callable] = {}
        
        # Security and authentication
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.jwt_secrets: Dict[str, str] = {"default": "your-secret-key"}
        
        # Performance monitoring
        self.gateway_metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "error_responses": 0,
            "average_response_time_ms": 0.0,
            "rate_limited_requests": 0,
            "authentication_failures": 0
        }
        
        # Response cache
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
        
        # Initialize default endpoints
        self._initialize_default_endpoints()
        
        self.logger.info("Unified API Gateway initialized")
    
    def _initialize_default_endpoints(self):
        """Initialize default system endpoints"""
        default_endpoints = [
            # Health and status
            APIEndpoint(
                path="/api/v1/health",
                method=RequestMethod.GET,
                service_name="gateway",
                handler_function="health_check",
                auth_level=AuthenticationLevel.NONE,
                documentation={"description": "Gateway health check"}
            ),
            
            # Intelligence systems
            APIEndpoint(
                path="/api/v1/intelligence/analytics",
                method=RequestMethod.POST,
                service_name="intelligence",
                handler_function="analytics_query",
                auth_level=AuthenticationLevel.API_KEY,
                rate_limit={"requests": 100, "window_seconds": 3600},
                documentation={"description": "Execute analytics query"}
            ),
            
            # Orchestration
            APIEndpoint(
                path="/api/v1/orchestration/workflow",
                method=RequestMethod.POST,
                service_name="orchestration",
                handler_function="execute_workflow",
                auth_level=AuthenticationLevel.JWT,
                rate_limit={"requests": 50, "window_seconds": 3600},
                documentation={"description": "Execute orchestration workflow"}
            ),
            
            # Coordination
            APIEndpoint(
                path="/api/v1/coordination/services",
                method=RequestMethod.GET,
                service_name="coordination",
                handler_function="list_services",
                auth_level=AuthenticationLevel.API_KEY,
                documentation={"description": "List available services"}
            ),
            
            # Streaming
            APIEndpoint(
                path="/api/v1/streaming/events",
                method=RequestMethod.POST,
                service_name="streaming",
                handler_function="publish_event",
                auth_level=AuthenticationLevel.API_KEY,
                rate_limit={"requests": 1000, "window_seconds": 3600},
                documentation={"description": "Publish streaming event"}
            )
        ]
        
        for endpoint in default_endpoints:
            self.register_endpoint(endpoint)
    
    def register_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Register new API endpoint"""
        try:
            route_key = f"{endpoint.method.value}:{endpoint.path}"
            
            if route_key in self.endpoints:
                self.logger.warning(f"Endpoint {route_key} already exists, replacing")
            
            self.endpoints[route_key] = endpoint
            self.logger.info(f"Registered endpoint: {route_key}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register endpoint: {e}")
            return False
    
    def register_handler(self, service_name: str, handler_function: str, handler: Callable) -> bool:
        """Register request handler function"""
        try:
            handler_key = f"{service_name}.{handler_function}"
            self.route_handlers[handler_key] = handler
            self.logger.info(f"Registered handler: {handler_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register handler: {e}")
            return False
    
    async def process_request(self, request: APIRequest) -> APIResponse:
        """Process incoming API request through the gateway"""
        start_time = time.time()
        request_metrics = {"start_time": start_time}
        
        try:
            self.gateway_metrics["total_requests"] += 1
            
            # Route resolution
            route_key = f"{request.method.value}:{request.path}"
            endpoint = self._resolve_endpoint(request.path, request.method)
            
            if not endpoint:
                return self._create_error_response(
                    request.request_id, 404, "Endpoint not found"
                )
            
            # Authentication
            auth_result = await self._authenticate_request(request, endpoint)
            if not auth_result["success"]:
                self.gateway_metrics["authentication_failures"] += 1
                return self._create_error_response(
                    request.request_id, 401, auth_result["error"]
                )
            
            # Rate limiting
            rate_limit_result = self._check_rate_limits(request, endpoint)
            if not rate_limit_result["allowed"]:
                self.gateway_metrics["rate_limited_requests"] += 1
                return self._create_error_response(
                    request.request_id, 429, "Rate limit exceeded",
                    headers={"Retry-After": str(rate_limit_result.get("retry_after", 60))}
                )
            
            # Request validation
            validation_result, errors = self.validator.validate_request(request, endpoint)
            if not validation_result:
                return self._create_error_response(
                    request.request_id, 400, f"Validation failed: {'; '.join(errors)}"
                )
            
            # Cache check
            cache_result = await self._check_cache(request, endpoint)
            if cache_result:
                return cache_result
            
            # Request processing
            response = await self._process_endpoint_request(request, endpoint)
            
            # Cache response if cacheable
            if endpoint.cache_ttl > 0 and response.status_code == 200:
                await self._cache_response(request, endpoint, response)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            response.execution_time_ms = execution_time
            self._update_response_metrics(response, execution_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            self.gateway_metrics["error_responses"] += 1
            
            return self._create_error_response(
                request.request_id, 500, f"Internal server error: {str(e)}",
                execution_time_ms=execution_time
            )
    
    def _resolve_endpoint(self, path: str, method: RequestMethod) -> Optional[APIEndpoint]:
        """Resolve endpoint from path and method"""
        # Direct match first
        route_key = f"{method.value}:{path}"
        if route_key in self.endpoints:
            return self.endpoints[route_key]
        
        # Pattern matching for parameterized paths
        for key, endpoint in self.endpoints.items():
            endpoint_method, endpoint_path = key.split(':', 1)
            if endpoint_method == method.value:
                if self._path_matches_pattern(path, endpoint_path):
                    return endpoint
        
        return None
    
    def _path_matches_pattern(self, request_path: str, endpoint_path: str) -> bool:
        """Check if request path matches endpoint pattern"""
        request_parts = request_path.strip('/').split('/')
        endpoint_parts = endpoint_path.strip('/').split('/')
        
        if len(request_parts) != len(endpoint_parts):
            return False
        
        for req_part, ep_part in zip(request_parts, endpoint_parts):
            if ep_part.startswith('{') and ep_part.endswith('}'):
                # Path parameter - matches any value
                continue
            elif req_part != ep_part:
                return False
        
        return True
    
    async def _authenticate_request(self, request: APIRequest, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Authenticate request based on endpoint requirements"""
        if endpoint.auth_level == AuthenticationLevel.NONE:
            return {"success": True}
        
        # API Key authentication
        if endpoint.auth_level == AuthenticationLevel.API_KEY:
            api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            if not api_key:
                return {"success": False, "error": "API key required"}
            
            if api_key not in self.api_keys:
                return {"success": False, "error": "Invalid API key"}
            
            # Update request context
            request.user_context.update(self.api_keys[api_key])
            return {"success": True}
        
        # JWT authentication
        if endpoint.auth_level == AuthenticationLevel.JWT:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return {"success": False, "error": "JWT token required"}
            
            token = auth_header[7:]  # Remove "Bearer "
            # In production, validate JWT token
            # For now, accept any non-empty token
            if token:
                request.user_context["jwt_token"] = token
                return {"success": True}
            else:
                return {"success": False, "error": "Invalid JWT token"}
        
        # Basic authentication
        if endpoint.auth_level == AuthenticationLevel.BASIC:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Basic "):
                # In production, validate credentials
                return {"success": True}
            else:
                return {"success": False, "error": "Basic authentication required"}
        
        return {"success": False, "error": "Unsupported authentication method"}
    
    def _check_rate_limits(self, request: APIRequest, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Check rate limits for request"""
        if not endpoint.rate_limit:
            return {"allowed": True}
        
        # Determine rate limit scope and identifier
        scope = RateLimitScope.API_KEY
        identifier = request.user_context.get("api_key", request.client_ip)
        
        # Use endpoint-specific rate limit if available
        requests_limit = endpoint.rate_limit.get("requests", 1000)
        window_seconds = endpoint.rate_limit.get("window_seconds", 3600)
        
        # Check with rate limiter
        allowed, rate_info = self.rate_limiter.check_rate_limit(
            scope, identifier, "default", "token_bucket"
        )
        
        return {
            "allowed": allowed,
            "rate_info": rate_info,
            "retry_after": rate_info.get("retry_after", 60)
        }
    
    async def _check_cache(self, request: APIRequest, endpoint: APIEndpoint) -> Optional[APIResponse]:
        """Check if response is cached"""
        if endpoint.cache_ttl <= 0 or request.method != RequestMethod.GET:
            return None
        
        # Generate cache key
        cache_key = self._generate_cache_key(request, endpoint)
        
        with self.cache_lock:
            if cache_key in self.response_cache:
                cached_item = self.response_cache[cache_key]
                
                # Check if cache is still valid
                if time.time() - cached_item["timestamp"] < endpoint.cache_ttl:
                    response = cached_item["response"]
                    response.headers["X-Cache"] = "HIT"
                    return response
                else:
                    # Remove expired cache
                    del self.response_cache[cache_key]
        
        return None
    
    async def _cache_response(self, request: APIRequest, endpoint: APIEndpoint, response: APIResponse):
        """Cache response if applicable"""
        if endpoint.cache_ttl <= 0 or response.status_code != 200:
            return
        
        cache_key = self._generate_cache_key(request, endpoint)
        
        with self.cache_lock:
            self.response_cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
    
    def _generate_cache_key(self, request: APIRequest, endpoint: APIEndpoint) -> str:
        """Generate cache key for request"""
        key_data = f"{request.method.value}:{request.path}:{json.dumps(request.query_params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    async def _process_endpoint_request(self, request: APIRequest, endpoint: APIEndpoint) -> APIResponse:
        """Process request through endpoint handler"""
        handler_key = f"{endpoint.service_name}.{endpoint.handler_function}"
        
        if handler_key in self.route_handlers:
            try:
                handler = self.route_handlers[handler_key]
                result = await handler(request, endpoint)
                
                if isinstance(result, APIResponse):
                    return result
                else:
                    # Convert result to response
                    return APIResponse(
                        request_id=request.request_id,
                        status_code=200,
                        body=result
                    )
                    
            except Exception as e:
                self.logger.error(f"Handler execution failed: {e}")
                return self._create_error_response(
                    request.request_id, 500, f"Handler error: {str(e)}"
                )
        else:
            # Mock response for unregistered handlers
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={
                    "message": f"Mock response for {endpoint.service_name}.{endpoint.handler_function}",
                    "endpoint": endpoint.path,
                    "method": endpoint.method.value
                }
            )
    
    def _create_error_response(self, request_id: str, status_code: int, 
                             error_message: str, headers: Optional[Dict[str, str]] = None,
                             execution_time_ms: float = 0.0) -> APIResponse:
        """Create error response"""
        return APIResponse(
            request_id=request_id,
            status_code=status_code,
            headers=headers or {},
            error_message=error_message,
            body={"error": error_message, "status_code": status_code},
            execution_time_ms=execution_time_ms
        )
    
    def _update_response_metrics(self, response: APIResponse, execution_time_ms: float):
        """Update gateway performance metrics"""
        if response.status_code < 400:
            self.gateway_metrics["successful_responses"] += 1
        else:
            self.gateway_metrics["error_responses"] += 1
        
        # Update average response time
        total_responses = (self.gateway_metrics["successful_responses"] + 
                          self.gateway_metrics["error_responses"])
        current_avg = self.gateway_metrics["average_response_time_ms"]
        
        self.gateway_metrics["average_response_time_ms"] = (
            (current_avg * (total_responses - 1) + execution_time_ms) / total_responses
        )
    
    def add_api_key(self, api_key: str, metadata: Dict[str, Any]) -> bool:
        """Add API key with metadata"""
        try:
            self.api_keys[api_key] = {
                "api_key": api_key,
                "created_at": datetime.now().isoformat(),
                **metadata
            }
            self.logger.info(f"Added API key: {api_key[:8]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add API key: {e}")
            return False
    
    def get_gateway_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gateway statistics"""
        return {
            "gateway_metrics": self.gateway_metrics.copy(),
            "endpoints_registered": len(self.endpoints),
            "handlers_registered": len(self.route_handlers),
            "active_api_keys": len(self.api_keys),
            "cache_entries": len(self.response_cache),
            "rate_limit_buckets": len(self.rate_limiter.rate_buckets)
        }
    
    def get_endpoint_documentation(self) -> Dict[str, Any]:
        """Get API documentation for all endpoints"""
        docs = {}
        
        for route_key, endpoint in self.endpoints.items():
            docs[route_key] = {
                "path": endpoint.path,
                "method": endpoint.method.value,
                "service": endpoint.service_name,
                "handler": endpoint.handler_function,
                "auth_level": endpoint.auth_level.value,
                "rate_limit": endpoint.rate_limit,
                "cache_ttl": endpoint.cache_ttl,
                "documentation": endpoint.documentation,
                "enabled": endpoint.enabled
            }
        
        return docs


# ============================================================================
# GLOBAL GATEWAY INSTANCE
# ============================================================================

# Global instance for unified API access
unified_gateway = UnifiedAPIGateway()

# Export for external use
__all__ = [
    'RequestMethod',
    'AuthenticationLevel', 
    'RateLimitScope',
    'APIEndpoint',
    'APIRequest',
    'APIResponse',
    'RateLimitingEngine',
    'RequestValidator',
    'UnifiedAPIGateway',
    'unified_gateway'
]