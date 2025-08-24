"""
Enhanced Error Handling Module
==============================

Centralized error handling for TestMaster Dashboard APIs.
Provides consistent error responses, logging, and monitoring.

Author: TestMaster Team
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from functools import wraps
from flask import jsonify, request
import sys
import os

logger = logging.getLogger(__name__)

class ErrorCode:
    """Standard error codes for the dashboard."""
    
    # Client errors (4xx)
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    
    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    MONITOR_ERROR = "MONITOR_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"

class DashboardError(Exception):
    """Base exception for dashboard-specific errors."""
    
    def __init__(self, message: str, error_code: str = ErrorCode.INTERNAL_ERROR, 
                 status_code: int = 500, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

class ValidationError(DashboardError):
    """Error for input validation failures."""
    
    def __init__(self, message: str, field: str = None, details: Optional[Dict] = None):
        details = details or {}
        if field:
            details['field'] = field
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details
        )

class MonitorError(DashboardError):
    """Error for monitoring system failures."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.MONITOR_ERROR,
            status_code=503,
            details=details
        )

class CacheError(DashboardError):
    """Error for cache system failures."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.CACHE_ERROR,
            status_code=503,
            details=details
        )

def create_error_response(error: Exception, request_id: str = None) -> Tuple[Dict[str, Any], int]:
    """
    Create a standardized error response.
    
    Args:
        error: The exception that occurred
        request_id: Optional request ID for tracking
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    if isinstance(error, DashboardError):
        response = {
            'status': 'error',
            'error_code': error.error_code,
            'message': error.message,
            'timestamp': error.timestamp,
            'details': error.details
        }
        status_code = error.status_code
    else:
        # Handle unexpected errors
        response = {
            'status': 'error',
            'error_code': ErrorCode.INTERNAL_ERROR,
            'message': 'An unexpected error occurred',
            'timestamp': datetime.now().isoformat(),
            'details': {'type': type(error).__name__}
        }
        status_code = 500
    
    if request_id:
        response['request_id'] = request_id
    
    # Add debug info in development
    if os.getenv('ENVIRONMENT') == 'development':
        response['debug'] = {
            'exception_type': type(error).__name__,
            'exception_message': str(error),
            'traceback': traceback.format_exc().split('\n')[-10:]  # Last 10 lines
        }
    
    return response, status_code

def handle_api_error(func):
    """
    Decorator to handle API errors consistently.
    
    Usage:
        @performance_bp.route('/endpoint')
        @handle_api_error
        def my_endpoint():
            # Your code here
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_id = getattr(request, 'id', None) or f"{func.__name__}_{datetime.now().timestamp()}"
        
        try:
            logger.debug(f"API call started: {func.__name__} [Request ID: {request_id}]")
            result = func(*args, **kwargs)
            logger.debug(f"API call completed: {func.__name__} [Request ID: {request_id}]")
            return result
            
        except DashboardError as e:
            logger.warning(f"Dashboard error in {func.__name__}: {e.message} [Request ID: {request_id}]")
            response, status_code = create_error_response(e, request_id)
            return jsonify(response), status_code
            
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e} [Request ID: {request_id}]")
            logger.error(f"Traceback: {traceback.format_exc()}")
            response, status_code = create_error_response(e, request_id)
            return jsonify(response), status_code
    
    return wrapper

def validate_request_params(required_params: list = None, optional_params: dict = None):
    """
    Decorator to validate request parameters.
    
    Args:
        required_params: List of required parameter names
        optional_params: Dict of optional params with default values
        
    Usage:
        @validate_request_params(['codebase'], {'hours': 1, 'format': 'json'})
        def my_endpoint():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            errors = []
            
            # Check required parameters
            if required_params:
                for param in required_params:
                    if not request.args.get(param):
                        errors.append(f"Missing required parameter: {param}")
            
            if errors:
                raise ValidationError(
                    message="Request validation failed",
                    details={'validation_errors': errors}
                )
            
            # Add optional parameters with defaults
            if optional_params:
                for param, default in optional_params.items():
                    if param not in request.args:
                        request.args = request.args.copy()
                        request.args[param] = str(default)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def monitor_performance(func):
    """
    Decorator to monitor API endpoint performance.
    
    Usage:
        @monitor_performance
        def my_endpoint():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"API Performance: {func.__name__} completed in {duration:.3f}s")
            
            # Add performance headers
            if hasattr(result, 'headers'):
                result.headers['X-Response-Time'] = f"{duration:.3f}s"
                result.headers['X-Endpoint'] = func.__name__
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"API Performance: {func.__name__} failed after {duration:.3f}s")
            raise
    
    return wrapper

# Convenience decorator that combines all enhancements
def enhanced_api_endpoint(required_params: list = None, optional_params: dict = None):
    """
    Combined decorator for full API enhancement.
    
    Usage:
        @enhanced_api_endpoint(['codebase'], {'hours': 1})
        def my_endpoint():
            pass
    """
    def decorator(func):
        # Apply decorators in reverse order (innermost first)
        enhanced_func = func
        enhanced_func = handle_api_error(enhanced_func)
        enhanced_func = monitor_performance(enhanced_func)
        if required_params or optional_params:
            enhanced_func = validate_request_params(required_params, optional_params)(enhanced_func)
        return enhanced_func
    return decorator

def log_error_stats():
    """Log error statistics for monitoring."""
    # This could be extended to track error rates, patterns, etc.
    pass