"""
Agency-Swarm Derived Error Handling Security Module
Extracted from agency-swarm/agency_swarm/util/errors.py
Enhanced for comprehensive security error handling
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Type
from contextlib import contextmanager


class SecurityError(Exception):
    """Base security exception class"""
    def __init__(self, message: str, error_code: str = "SEC_001", details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class AuthenticationError(SecurityError):
    """Authentication related security errors"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "AUTH_001", details)


class AuthorizationError(SecurityError):
    """Authorization related security errors"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "AUTHZ_001", details)


class ValidationError(SecurityError):
    """Input validation security errors"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "VAL_001", details)


class RateLimitError(SecurityError):
    """Rate limiting security errors"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "RATE_001", details)


class RefusalError(SecurityError):
    """Content refusal security errors - derived from agency-swarm pattern"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "REF_001", details)


class SecurityErrorHandler:
    """Centralized security error handling system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle security errors with logging and response formatting"""
        error_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {}
        }
        
        if isinstance(error, SecurityError):
            error_info.update({
                'error_code': error.error_code,
                'details': error.details,
                'security_error': True
            })
        
        # Log the error
        self.logger.error(f"Security Error: {error_info}", exc_info=True)
        
        # Track error frequency
        error_key = f"{type(error).__name__}:{getattr(error, 'error_code', 'UNKNOWN')}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        return error_info
    
    @contextmanager
    def secure_operation(self, operation_name: str):
        """Context manager for secure operations with error handling"""
        try:
            self.logger.info(f"Starting secure operation: {operation_name}")
            yield
            self.logger.info(f"Completed secure operation: {operation_name}")
        except SecurityError as e:
            self.handle_error(e, {'operation': operation_name})
            raise
        except Exception as e:
            # Wrap non-security exceptions in SecurityError
            security_error = SecurityError(
                f"Unexpected error in {operation_name}: {str(e)}",
                "SEC_999",
                {'original_error': type(e).__name__, 'operation': operation_name}
            )
            self.handle_error(security_error, {'operation': operation_name})
            raise security_error
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error frequency statistics"""
        return self.error_counts.copy()
    
    def reset_error_stats(self):
        """Reset error frequency counters"""
        self.error_counts.clear()


# Global error handler instance
security_error_handler = SecurityErrorHandler()


def handle_security_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for handling security errors"""
    return security_error_handler.handle_error(error, context)


def secure_operation(operation_name: str):
    """Decorator for secure operations"""
    return security_error_handler.secure_operation(operation_name)