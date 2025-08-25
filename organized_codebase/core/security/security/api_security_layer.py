
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
Agency-Swarm Derived API Security Layer
Comprehensive API security based on FastAPI integration patterns
Enhanced with CORS, headers, middleware, and request validation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from .error_handler import SecurityError, AuthenticationError, security_error_handler
from .authentication_system import auth_manager, AuthenticationContext
from .rate_limiter import rate_limit_manager, get_client_identifier
from .validation_framework import global_validator, sanitize_input


@dataclass
class SecurityConfig:
    """API security configuration"""
    cors_origins: List[str] = None
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = None
    cors_allow_headers: List[str] = None
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    require_https: bool = True
    security_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]  # Default from agency-swarm pattern
        if self.cors_allow_methods is None:
            self.cors_allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        if self.cors_allow_headers is None:
            self.cors_allow_headers = ["*"]
        if self.security_headers is None:
            self.security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY", 
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            }


class RequestSecurityValidator:
    """Request-level security validation"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_request_size(self, content_length: Optional[int]) -> bool:
        """Validate request size"""
        if content_length and content_length > self.config.max_request_size:
            raise SecurityError(
                f"Request too large: {content_length} bytes. Max allowed: {self.config.max_request_size}",
                "REQ_SIZE_001"
            )
        return True
    
    def validate_content_type(self, content_type: Optional[str], allowed_types: List[str]) -> bool:
        """Validate request content type"""
        if not content_type:
            return True  # Allow requests without content-type
        
        # Extract main content type (ignore charset, etc.)
        main_type = content_type.split(';')[0].strip().lower()
        
        if allowed_types and main_type not in allowed_types:
            raise SecurityError(
                f"Invalid content type: {content_type}. Allowed types: {allowed_types}",
                "REQ_CONTENT_001"
            )
        return True
    
    def validate_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Validate and sanitize request headers"""
        sanitized_headers = {}
        
        for key, value in headers.items():
            # Sanitize header values
            sanitized_value = sanitize_input(value)
            
            # Check for suspicious headers
            key_lower = key.lower()
            if key_lower in ['x-forwarded-for', 'x-real-ip']:
                # Validate IP addresses in forwarding headers
                if not self._is_valid_ip_list(sanitized_value):
                    self.logger.warning(f"Suspicious IP forwarding header: {key}={value}")
                    continue
            
            sanitized_headers[key] = sanitized_value
        
        return sanitized_headers
    
    def _is_valid_ip_list(self, ip_string: str) -> bool:
        """Validate comma-separated IP address list"""
        import ipaddress
        try:
            ips = [ip.strip() for ip in ip_string.split(',')]
            for ip in ips:
                ipaddress.ip_address(ip)
            return True
        except (ValueError, ipaddress.AddressValueError):
            return False


class APISecurityMiddleware:
    """Comprehensive API security middleware"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.request_validator = RequestSecurityValidator(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Request tracking
        self.active_requests: Dict[str, datetime] = {}
        
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and secure incoming request"""
        try:
            request_id = request_data.get('request_id', 'unknown')
            self.active_requests[request_id] = datetime.utcnow()
            
            # Extract request components
            method = request_data.get('method', 'GET')
            path = request_data.get('path', '/')
            headers = request_data.get('headers', {})
            body = request_data.get('body')
            ip_address = request_data.get('ip_address')
            
            # Security validations
            
            # 1. HTTPS enforcement
            if self.config.require_https and not request_data.get('is_https', False):
                raise SecurityError("HTTPS required for secure communication", "HTTPS_001")
            
            # 2. Request size validation
            content_length = headers.get('content-length')
            if content_length:
                self.request_validator.validate_request_size(int(content_length))
            
            # 3. Content type validation for POST/PUT requests
            if method in ['POST', 'PUT', 'PATCH']:
                allowed_types = ['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data']
                self.request_validator.validate_content_type(
                    headers.get('content-type'), 
                    allowed_types
                )
            
            # 4. Header validation and sanitization
            sanitized_headers = self.request_validator.validate_headers(headers)
            
            # 5. Rate limiting
            client_id = get_client_identifier({
                'ip_address': ip_address,
                'user_agent': headers.get('user-agent')
            })
            
            # Determine rate limit rule based on endpoint
            rule_name = self._get_rate_limit_rule(path, method)
            rate_limit_manager.check_limit(client_id, rule_name, {
                'method': method,
                'path': path,
                'ip_address': ip_address
            })
            
            # 6. Body sanitization if present
            sanitized_body = body
            if body and isinstance(body, (str, dict)):
                sanitized_body = sanitize_input(body)
            
            # 7. Input validation
            if sanitized_body:
                validation_result = global_validator.validate_input(sanitized_body)
                if not validation_result.is_valid:
                    raise SecurityError(
                        f"Request validation failed: {validation_result.reason}",
                        "REQ_VAL_001"
                    )
            
            # Return sanitized request data
            secured_request = {
                **request_data,
                'headers': sanitized_headers,
                'body': sanitized_body,
                'security_validated': True,
                'validation_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Request security validation passed: {method} {path}")
            return secured_request
            
        except SecurityError:
            raise
        except Exception as e:
            error = SecurityError(f"Request security processing failed: {str(e)}", "REQ_PROC_001")
            security_error_handler.handle_error(error, request_data)
            raise error
        finally:
            # Cleanup request tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and secure outgoing response"""
        try:
            # Add security headers
            headers = response_data.get('headers', {})
            headers.update(self.config.security_headers)
            
            # Add CORS headers if needed
            if self.config.cors_origins:
                headers.update(self._get_cors_headers())
            
            # Sanitize response body if it contains user data
            body = response_data.get('body')
            if body and isinstance(body, (str, dict)):
                # Only sanitize if response contains potentially unsafe content
                if isinstance(body, str) and any(pattern in body.lower() for pattern in ['<script', 'javascript:', 'data:text/html']):
                    body = sanitize_input(body)
                elif isinstance(body, dict) and 'message' in body:
                    body['message'] = sanitize_input(body['message'])
            
            response_data.update({
                'headers': headers,
                'body': body,
                'security_processed': True
            })
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Response security processing failed: {e}")
            # Don't fail the response for security processing errors
            return response_data
    
    def _get_rate_limit_rule(self, path: str, method: str) -> str:
        """Determine appropriate rate limit rule for endpoint"""
        if '/auth/' in path or path.endswith('/login'):
            return 'api_auth'
        elif '/upload' in path or method in ['POST', 'PUT'] and 'file' in path:
            return 'api_upload'
        elif '/ws/' in path:
            return 'websocket'
        else:
            return 'api_general'
    
    def _get_cors_headers(self) -> Dict[str, str]:
        """Generate CORS headers based on configuration"""
        return {
            'Access-Control-Allow-Origin': ','.join(self.config.cors_origins),
            'Access-Control-Allow-Methods': ','.join(self.config.cors_allow_methods),
            'Access-Control-Allow-Headers': ','.join(self.config.cors_allow_headers),
            'Access-Control-Allow-Credentials': 'true' if self.config.cors_allow_credentials else 'false'
        }


class APISecurityManager:
    """Central API security management"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.middleware = APISecurityMiddleware(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Security event tracking
        self.security_events: List[Dict[str, Any]] = []
        self.max_events = 1000
    
    def secure_endpoint(self, endpoint_func: Callable, 
                       require_auth: bool = True,
                       required_permissions: List[str] = None) -> Callable:
        """Decorator to secure API endpoints"""
        
        def security_wrapper(*args, **kwargs):
            try:
                # Extract request context from kwargs
                request_context = kwargs.get('request_context', {})
                
                # Apply security middleware
                secured_request = self.middleware.process_request(request_context)
                kwargs['request_context'] = secured_request
                
                # Authentication check
                auth_context = None
                if require_auth:
                    token = secured_request.get('headers', {}).get('authorization')
                    if not token:
                        raise AuthenticationError("Authentication required")
                    
                    # Remove 'Bearer ' prefix if present
                    if token.startswith('Bearer '):
                        token = token[7:]
                    
                    auth_context = auth_manager.authenticate_token(token, secured_request)
                    if not auth_context.is_authenticated:
                        raise AuthenticationError("Invalid authentication")
                
                # Authorization check
                if required_permissions and auth_context and auth_context.user_session:
                    from .authentication_system import authz_manager
                    for permission in required_permissions:
                        resource, action = permission.split(':', 1)
                        if not authz_manager.authorize_request(auth_context.user_session, resource, action):
                            from .error_handler import AuthorizationError
                            raise AuthorizationError(f"Permission denied: {permission}")
                
                # Call the actual endpoint
                response = endpoint_func(*args, **kwargs)
                
                # Process response through security middleware
                if isinstance(response, dict):
                    secured_response = self.middleware.process_response(response)
                    return secured_response
                
                return response
                
            except (SecurityError, AuthenticationError):
                raise
            except Exception as e:
                error = SecurityError(f"Endpoint security wrapper failed: {str(e)}", "EP_SEC_001")
                security_error_handler.handle_error(error)
                raise error
        
        return security_wrapper
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        
        self.logger.info(f"Security event logged: {event_type}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        now = datetime.utcnow()
        
        # Count events by type in last hour
        hour_ago = now - timedelta(hours=1)
        recent_events = [
            e for e in self.security_events 
            if datetime.fromisoformat(e['timestamp']) > hour_ago
        ]
        
        event_counts = {}
        for event in recent_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'total_events': len(self.security_events),
            'recent_events_1h': len(recent_events),
            'event_types_1h': event_counts,
            'active_requests': len(self.middleware.active_requests),
            'rate_limiter_stats': rate_limit_manager.limiter.get_stats() if hasattr(rate_limit_manager.limiter, 'get_stats') else {}
        }


# Global security manager instance
api_security_manager = APISecurityManager()


def secure_api_endpoint(require_auth: bool = True, required_permissions: List[str] = None):
    """Decorator for securing API endpoints"""
    def decorator(func):
        return api_security_manager.secure_endpoint(func, require_auth, required_permissions)
    return decorator