"""
Enhanced API Gateway - Consolidated implementation with advanced features

This module extends the TestMaster API Gateway with advanced features
extracted from the unified gateway during IRONCLAD consolidation.

Features added:
- Advanced rate limiting algorithms (token bucket, sliding window, etc.)
- Enhanced request validation and sanitization
- Improved performance monitoring
- Extended middleware capabilities

Author: Agent E - IRONCLAD Consolidation Result
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from collections import deque

# Import base gateway components
from .api_gateway_core import (
    TestMasterAPIGateway, RequestMiddleware, 
    CORSMiddleware, LoggingMiddleware, SecurityMiddleware
)
from .api_models import APIEndpoint, APIRequest, APIResponse, APIUser, HTTPMethod
from .enhanced_rate_limiting import EnhancedRateLimitingEngine, RateLimitPolicy, RateLimitAlgorithm

logger = logging.getLogger(__name__)


class RequestValidationMiddleware(RequestMiddleware):
    """
    Enhanced request validation middleware
    
    Extracted and enhanced from unified gateway implementation
    """
    
    def __init__(self):
        self.logger = logging.getLogger("request_validation")
        
        # Security patterns for validation
        self.security_patterns = {
            "sql_injection": [r"union\s+select", r"drop\s+table", r"insert\s+into"],
            "xss": [r"<script", r"javascript:", r"onload="],
            "path_traversal": [r"\.\.\/", r"\.\.\\", r"%2e%2e"],
            "command_injection": [r";.*rm\s", r"\|\s*rm\s", r"&&.*rm\s"]
        }
    
    async def process_request(self, request: APIRequest) -> Optional[APIResponse]:
        """Validate request for security threats"""
        # Check for malicious patterns
        threats_detected = self._scan_for_threats(request)
        if threats_detected:
            self.logger.warning(f"Security threats detected: {threats_detected}")
            from .api_models import create_api_response
            return create_api_response(
                status_code=403,
                message="Request blocked due to security policy",
                errors=[f"Security violation: {', '.join(threats_detected)}"],
                request_id=request.request_id
            )
        
        # Check request size limits
        if hasattr(request, 'body') and request.body:
            body_size = len(str(request.body))
            if body_size > 10 * 1024 * 1024:  # 10MB limit
                from .api_models import create_api_response
                return create_api_response(
                    status_code=413,
                    message="Request entity too large",
                    errors=["Request body exceeds 10MB limit"],
                    request_id=request.request_id
                )
        
        return None  # Allow request to continue
    
    def _scan_for_threats(self, request: APIRequest) -> List[str]:
        """Scan request for security threats"""
        import re
        threats = []
        
        # Combine all request data for scanning
        scan_data = []
        if hasattr(request, 'path'):
            scan_data.append(request.path)
        if hasattr(request, 'query_params'):
            scan_data.extend(str(v) for v in request.query_params.values())
        if hasattr(request, 'body') and request.body:
            scan_data.append(str(request.body))
        
        # Scan for each threat type
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                for data in scan_data:
                    if re.search(pattern, data, re.IGNORECASE):
                        threats.append(threat_type)
                        break
        
        return list(set(threats))


class PerformanceMonitoringMiddleware(RequestMiddleware):
    """
    Enhanced performance monitoring middleware
    
    Provides detailed performance metrics and alerting
    """
    
    def __init__(self):
        self.logger = logging.getLogger("performance_monitoring")
        self.performance_data: deque = deque(maxlen=1000)
        self.slow_request_threshold_ms = 1000
        self.error_rate_threshold = 0.05
    
    async def process_request(self, request: APIRequest) -> Optional[APIResponse]:
        """Mark request start time"""
        request.start_time = time.time()
        return None
    
    async def process_response(self, request: APIRequest, response: APIResponse) -> APIResponse:
        """Calculate and record performance metrics"""
        end_time = time.time()
        if hasattr(request, 'start_time'):
            duration_ms = (end_time - request.start_time) * 1000
            
            # Record performance data
            perf_record = {
                "timestamp": datetime.now(),
                "path": request.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "is_error": response.status_code >= 400,
                "is_slow": duration_ms > self.slow_request_threshold_ms
            }
            self.performance_data.append(perf_record)
            
            # Log slow requests
            if duration_ms > self.slow_request_threshold_ms:
                self.logger.warning(f"Slow request: {request.method} {request.path} took {duration_ms:.2f}ms")
            
            # Add performance headers
            if not response.metadata:
                response.metadata = {}
            response.metadata["X-Response-Time"] = f"{duration_ms:.2f}ms"
            response.metadata["X-Request-ID"] = request.request_id
        
        return response
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.performance_data:
            return {"error": "No performance data available"}
        
        total_requests = len(self.performance_data)
        error_requests = sum(1 for r in self.performance_data if r["is_error"])
        slow_requests = sum(1 for r in self.performance_data if r["is_slow"])
        avg_duration = sum(r["duration_ms"] for r in self.performance_data) / total_requests
        
        return {
            "total_requests": total_requests,
            "error_rate": error_requests / total_requests,
            "slow_request_rate": slow_requests / total_requests,
            "average_response_time_ms": avg_duration,
            "p95_response_time_ms": self._calculate_percentile(95),
            "p99_response_time_ms": self._calculate_percentile(99),
            "recent_errors": error_requests,
            "recent_slow_requests": slow_requests
        }
    
    def _calculate_percentile(self, percentile: int) -> float:
        """Calculate response time percentile"""
        durations = sorted([r["duration_ms"] for r in self.performance_data])
        index = int((percentile / 100) * len(durations))
        return durations[min(index, len(durations) - 1)]


class EnhancedTestMasterAPIGateway(TestMasterAPIGateway):
    """
    Enhanced API Gateway with advanced features from unified gateway
    
    This class extends the base TestMaster API Gateway with:
    - Advanced rate limiting algorithms
    - Enhanced request validation
    - Performance monitoring and alerting
    - Improved middleware capabilities
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_request_log_size: int = 50000,
        enable_advanced_features: bool = True
    ):
        """
        Initialize Enhanced API Gateway
        
        Args:
            host: Gateway host address
            port: Gateway port
            max_request_log_size: Maximum request log size
            enable_advanced_features: Enable advanced consolidated features
        """
        # Initialize base gateway
        super().__init__(host, port, max_request_log_size)
        
        # Replace rate limiter with enhanced version
        if enable_advanced_features:
            self.enhanced_rate_limiter = EnhancedRateLimitingEngine()
            
            # Add enhanced middleware
            self.validation_middleware = RequestValidationMiddleware()
            self.performance_middleware = PerformanceMonitoringMiddleware()
            
            # Insert enhanced middleware at the beginning of pipeline
            self.middleware.insert(0, self.validation_middleware)
            self.middleware.insert(1, self.performance_middleware)
            
            logger.info("Enhanced API Gateway features enabled")
        
        # Enhanced metrics
        self.enhanced_metrics = {
            "advanced_rate_limiting_enabled": enable_advanced_features,
            "validation_blocks": 0,
            "performance_alerts": 0,
            "algorithm_usage": {}
        }
        
        logger.info(f"Enhanced TestMaster API Gateway initialized on {host}:{port}")
    
    def add_rate_limit_policy(self, policy: RateLimitPolicy):
        """
        Add custom rate limiting policy
        
        Args:
            policy: Rate limiting policy configuration
        """
        if hasattr(self, 'enhanced_rate_limiter'):
            self.enhanced_rate_limiter.add_policy(policy)
            logger.info(f"Added rate limit policy: {policy.name}")
    
    async def handle_request_enhanced(self, request: APIRequest) -> APIResponse:
        """
        Enhanced request handling with advanced features
        
        This method extends the base request handling with:
        - Advanced rate limiting
        - Enhanced validation
        - Performance monitoring
        """
        start_time = time.time()
        
        try:
            # Use enhanced rate limiting if available
            if hasattr(self, 'enhanced_rate_limiter'):
                # Determine rate limit scope and policy
                user_id = getattr(request, 'user_id', None)
                scope = f"user:{user_id}" if user_id else f"ip:{request.ip_address}"
                
                # Check rate limit with enhanced engine
                allowed, rate_info = self.enhanced_rate_limiter.check_rate_limit(
                    scope=scope,
                    identifier=user_id or request.ip_address,
                    policy_name="default",
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET
                )
                
                if not allowed:
                    self.enhanced_metrics["algorithm_usage"][rate_info.get("algorithm", "unknown")] = \
                        self.enhanced_metrics["algorithm_usage"].get(rate_info.get("algorithm", "unknown"), 0) + 1
                    
                    from .api_models import create_api_response
                    return create_api_response(
                        status_code=429,
                        message="Rate limit exceeded (enhanced)",
                        errors=["Advanced rate limiting policy enforced"],
                        metadata={
                            "retry_after": rate_info.get("retry_after", 60),
                            "algorithm": rate_info.get("algorithm"),
                            "policy": rate_info.get("policy")
                        },
                        request_id=request.request_id
                    )
            
            # Use base gateway request handling for everything else
            response = await super().handle_request(request)
            
            # Add enhanced headers
            if not response.metadata:
                response.metadata = {}
            response.metadata["X-Enhanced-Gateway"] = "true"
            response.metadata["X-Features"] = "advanced-rate-limiting,validation,monitoring"
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced request handling error: {e}")
            # Fall back to base gateway handling
            return await super().handle_request(request)
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gateway statistics including enhanced features"""
        base_stats = super().get_api_stats()
        
        enhanced_stats = {
            "enhanced_features": self.enhanced_metrics,
            "performance_monitoring": {},
            "advanced_rate_limiting": {}
        }
        
        # Get performance statistics
        if hasattr(self, 'performance_middleware'):
            enhanced_stats["performance_monitoring"] = self.performance_middleware.get_performance_summary()
        
        # Get enhanced rate limiting statistics
        if hasattr(self, 'enhanced_rate_limiter'):
            enhanced_stats["advanced_rate_limiting"] = self.enhanced_rate_limiter.get_statistics()
        
        return {**base_stats, **enhanced_stats}
    
    def cleanup_enhanced(self):
        """Enhanced cleanup including consolidated features"""
        # Base gateway cleanup
        super().cleanup()
        
        # Clean up enhanced features
        if hasattr(self, 'enhanced_rate_limiter'):
            self.enhanced_rate_limiter.cleanup_expired_buckets()
        
        logger.info("Enhanced API Gateway cleanup completed")


# Factory functions for enhanced gateway
def create_enhanced_api_gateway(
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_advanced_features: bool = True
) -> EnhancedTestMasterAPIGateway:
    """
    Create enhanced API gateway with consolidated features
    
    Args:
        host: Gateway host
        port: Gateway port
        enable_advanced_features: Enable advanced consolidated features
        
    Returns:
        Enhanced API gateway with consolidated functionality
    """
    return EnhancedTestMasterAPIGateway(
        host=host, 
        port=port, 
        enable_advanced_features=enable_advanced_features
    )


def create_enterprise_enhanced_gateway(
    host: str = "0.0.0.0",
    port: int = 8000
) -> EnhancedTestMasterAPIGateway:
    """
    Create enterprise-grade enhanced API gateway
    
    Args:
        host: Gateway host
        port: Gateway port
        
    Returns:
        Enterprise-configured enhanced API gateway
    """
    gateway = EnhancedTestMasterAPIGateway(host=host, port=port, enable_advanced_features=True)
    
    # Add enterprise rate limiting policies
    enterprise_policies = [
        RateLimitPolicy("enterprise_premium", 50000, 3600, RateLimitAlgorithm.TOKEN_BUCKET, 75000),
        RateLimitPolicy("enterprise_standard", 20000, 3600, RateLimitAlgorithm.SLIDING_WINDOW),
        RateLimitPolicy("enterprise_basic", 10000, 3600, RateLimitAlgorithm.FIXED_WINDOW),
        RateLimitPolicy("development_unlimited", 1000000, 3600, RateLimitAlgorithm.LEAKY_BUCKET)
    ]
    
    for policy in enterprise_policies:
        gateway.add_rate_limit_policy(policy)
    
    logger.info(f"Enterprise Enhanced API Gateway configured with {len(enterprise_policies)} policies")
    return gateway