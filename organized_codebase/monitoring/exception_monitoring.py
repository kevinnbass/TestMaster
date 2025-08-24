"""
AgentOps Derived Exception Monitoring Security Module
Extracted from AgentOps exceptions.py patterns and secure exception handling
Enhanced for comprehensive exception tracking and security monitoring
"""

import sys
import traceback
import logging
import hashlib
from typing import Dict, List, Optional, Any, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from .error_handler import SecurityError, security_error_handler


class ExceptionSeverity(Enum):
    """Exception severity levels based on AgentOps patterns"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExceptionCategory(Enum):
    """Exception categories for security classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION = "configuration"
    API_ERROR = "api_error"
    CLIENT_ERROR = "client_error"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


@dataclass
class SecureExceptionInfo:
    """Secure exception information structure"""
    exception_id: str
    exception_type: str
    exception_message: str
    category: ExceptionCategory
    severity: ExceptionSeverity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    frequency_count: int = 1
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def age_hours(self) -> float:
        """Get age of exception in hours"""
        return (datetime.utcnow() - self.first_seen).total_seconds() / 3600
    
    @property
    def is_recurring(self) -> bool:
        """Check if exception is recurring"""
        return self.frequency_count > 1


# AgentOps-inspired security exception classes
class SecureMultiSessionException(SecurityError):
    """Secure version of MultiSessionException with enhanced tracking"""
    def __init__(self, message: str = "Multiple sessions detected - security risk"):
        super().__init__(message, "MULTI_SESSION_001")
        self.category = ExceptionCategory.SECURITY_VIOLATION
        self.severity = ExceptionSeverity.HIGH


class SecureNoSessionException(SecurityError):
    """Secure version of NoSessionException with context tracking"""
    def __init__(self, message: str = "No session found - authentication required"):
        super().__init__(message, "NO_SESSION_001")
        self.category = ExceptionCategory.AUTHENTICATION
        self.severity = ExceptionSeverity.MEDIUM


class SecureNoApiKeyException(SecurityError):
    """Secure version of NoApiKeyException with enhanced security"""
    def __init__(self, message: str = "API key missing - authentication failure", 
                 endpoint: str = "https://api.agentops.ai"):
        enhanced_message = (
            f"{message}\n"
            f"Find your API key at {endpoint}/settings/projects"
        )
        super().__init__(enhanced_message, "NO_API_KEY_001")
        self.category = ExceptionCategory.AUTHENTICATION
        self.severity = ExceptionSeverity.CRITICAL


class SecureInvalidApiKeyException(SecurityError):
    """Secure version of InvalidApiKeyException with masking"""
    def __init__(self, api_key: str, endpoint: str = "https://api.agentops.ai"):
        # Mask API key for security
        masked_key = f"{api_key[:4]}***{api_key[-4:]}" if len(api_key) >= 8 else "***"
        message = (
            f"API Key is invalid: {masked_key}\n"
            f"Find your API key at {endpoint}/settings/projects"
        )
        super().__init__(message, "INVALID_API_KEY_001")
        self.category = ExceptionCategory.AUTHENTICATION
        self.severity = ExceptionSeverity.HIGH


class SecureApiServerException(SecurityError):
    """Secure version of ApiServerException with enhanced logging"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message, "API_SERVER_001")
        self.category = ExceptionCategory.API_ERROR
        self.severity = ExceptionSeverity.MEDIUM
        self.status_code = status_code


class SecureClientNotInitializedException(SecurityError):
    """Secure version of client initialization exception"""
    def __init__(self, message: str = "Client must be initialized before use"):
        super().__init__(message, "CLIENT_INIT_001")
        self.category = ExceptionCategory.CLIENT_ERROR
        self.severity = ExceptionSeverity.HIGH


class SecureJwtExpiredException(SecurityError):
    """Secure version of JWT expiration exception"""
    def __init__(self, message: str = "JWT token has expired - reauthentication required"):
        super().__init__(message, "JWT_EXPIRED_001")
        self.category = ExceptionCategory.AUTHENTICATION
        self.severity = ExceptionSeverity.MEDIUM


class ExceptionSecurityAnalyzer:
    """Security-focused exception analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Security-sensitive patterns in exception messages
        self.sensitive_patterns = [
            r'api[_\s]*key[:\s]*([a-zA-Z0-9\-]+)',
            r'token[:\s]*([a-zA-Z0-9\.\-_]+)',
            r'password[:\s]*([^\s]+)',
            r'secret[:\s]*([^\s]+)',
            r'bearer[:\s]*([a-zA-Z0-9\.\-_]+)',
        ]
        
        # Dangerous exception patterns that indicate security issues
        self.dangerous_patterns = [
            r'SQL.*injection',
            r'command.*injection',
            r'path.*traversal',
            r'directory.*traversal',
            r'XSS',
            r'cross.*site.*scripting',
            r'CSRF',
            r'privilege.*escalation',
        ]
    
    def analyze_exception(self, exc: Exception, context: Dict[str, Any] = None) -> SecureExceptionInfo:
        """Analyze exception for security implications"""
        try:
            # Generate unique ID for this exception type + message
            exception_signature = f"{type(exc).__name__}:{str(exc)}"
            exception_id = hashlib.sha256(exception_signature.encode()).hexdigest()[:16]
            
            # Categorize exception
            category = self._categorize_exception(exc)
            severity = self._assess_severity(exc, category)
            
            # Sanitize message for security
            sanitized_message = self._sanitize_message(str(exc))
            
            # Get stack trace (sanitized)
            stack_trace = None
            try:
                stack_trace = self._sanitize_stack_trace(traceback.format_exc())
            except Exception:
                pass
            
            exception_info = SecureExceptionInfo(
                exception_id=exception_id,
                exception_type=type(exc).__name__,
                exception_message=sanitized_message,
                category=category,
                severity=severity,
                stack_trace=stack_trace,
                context=context or {}
            )
            
            # Check for dangerous patterns
            if self._contains_dangerous_patterns(str(exc)):
                exception_info.severity = ExceptionSeverity.CRITICAL
                exception_info.category = ExceptionCategory.SECURITY_VIOLATION
                self.logger.critical(f"Security-sensitive exception detected: {exception_info.exception_type}")
            
            return exception_info
            
        except Exception as e:
            # Fallback exception info if analysis fails
            self.logger.error(f"Exception analysis failed: {e}")
            return SecureExceptionInfo(
                exception_id="analysis_failed",
                exception_type=type(exc).__name__,
                exception_message="Analysis failed - message sanitized",
                category=ExceptionCategory.UNKNOWN,
                severity=ExceptionSeverity.HIGH
            )
    
    def _categorize_exception(self, exc: Exception) -> ExceptionCategory:
        """Categorize exception based on type and AgentOps patterns"""
        exc_name = type(exc).__name__.lower()
        exc_message = str(exc).lower()
        
        # Authentication-related
        if any(auth_term in exc_name for auth_term in ['auth', 'token', 'jwt', 'api', 'key']):
            return ExceptionCategory.AUTHENTICATION
        
        # API-related
        if any(api_term in exc_name for api_term in ['api', 'server', 'http', 'request']):
            return ExceptionCategory.API_ERROR
        
        # Client-related
        if any(client_term in exc_name for client_term in ['client', 'init', 'session']):
            return ExceptionCategory.CLIENT_ERROR
        
        # Configuration-related
        if any(config_term in exc_name for config_term in ['config', 'setting', 'parameter']):
            return ExceptionCategory.CONFIGURATION
        
        # Check message content
        if any(sec_term in exc_message for sec_term in ['unauthorized', 'forbidden', 'denied']):
            return ExceptionCategory.AUTHORIZATION
        
        return ExceptionCategory.SYSTEM_ERROR
    
    def _assess_severity(self, exc: Exception, category: ExceptionCategory) -> ExceptionSeverity:
        """Assess exception severity"""
        exc_name = type(exc).__name__.lower()
        exc_message = str(exc).lower()
        
        # Critical severity patterns
        if any(critical_term in exc_message for critical_term in [
            'security', 'attack', 'breach', 'unauthorized access', 'injection'
        ]):
            return ExceptionSeverity.CRITICAL
        
        # High severity by category
        if category in [ExceptionCategory.AUTHENTICATION, ExceptionCategory.SECURITY_VIOLATION]:
            return ExceptionSeverity.HIGH
        
        # High severity by type
        if any(high_term in exc_name for high_term in ['runtime', 'system', 'memory', 'timeout']):
            return ExceptionSeverity.HIGH
        
        # Medium severity by category
        if category in [ExceptionCategory.API_ERROR, ExceptionCategory.CLIENT_ERROR]:
            return ExceptionSeverity.MEDIUM
        
        return ExceptionSeverity.LOW
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize exception message to remove sensitive information"""
        sanitized = message
        
        # Remove sensitive patterns
        import re
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, lambda m: f"{m.group().split(':')[0]}:***", sanitized, flags=re.IGNORECASE)
        
        # Limit message length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000] + "...[truncated]"
        
        return sanitized
    
    def _sanitize_stack_trace(self, stack_trace: str) -> str:
        """Sanitize stack trace to remove sensitive paths and information"""
        if not stack_trace:
            return None
        
        lines = stack_trace.split('\n')
        sanitized_lines = []
        
        for line in lines:
            # Remove full file paths, keep only filename
            if 'File "' in line:
                import os
                try:
                    start = line.find('File "') + 6
                    end = line.find('"', start)
                    if start > 5 and end > start:
                        file_path = line[start:end]
                        filename = os.path.basename(file_path)
                        line = line.replace(file_path, filename)
                except Exception:
                    pass
            
            # Remove sensitive information from line content
            for pattern in self.sensitive_patterns:
                import re
                line = re.sub(pattern, "***", line, flags=re.IGNORECASE)
            
            sanitized_lines.append(line)
        
        # Limit stack trace length
        if len(sanitized_lines) > 50:
            sanitized_lines = sanitized_lines[:25] + ["...[truncated]..."] + sanitized_lines[-25:]
        
        return '\n'.join(sanitized_lines)
    
    def _contains_dangerous_patterns(self, text: str) -> bool:
        """Check if exception contains dangerous security patterns"""
        import re
        text_lower = text.lower()
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False


class ExceptionMonitoringSystem:
    """Comprehensive exception monitoring system"""
    
    def __init__(self):
        self.analyzer = ExceptionSecurityAnalyzer()
        self.exception_registry: Dict[str, SecureExceptionInfo] = {}
        self.recent_exceptions: List[SecureExceptionInfo] = []
        self.exception_hooks: List[Callable[[SecureExceptionInfo], None]] = []
        self.max_recent_exceptions = 1000
        self.logger = logging.getLogger(__name__)
        
        # Security thresholds
        self.critical_threshold = 5  # Max critical exceptions per hour
        self.high_threshold = 20     # Max high severity exceptions per hour
        
    def register_exception(self, exc: Exception, context: Dict[str, Any] = None) -> SecureExceptionInfo:
        """Register and analyze an exception"""
        try:
            # Analyze exception
            exc_info = self.analyzer.analyze_exception(exc, context)
            
            # Update frequency if we've seen this exception before
            if exc_info.exception_id in self.exception_registry:
                existing = self.exception_registry[exc_info.exception_id]
                existing.frequency_count += 1
                existing.last_seen = datetime.utcnow()
                exc_info = existing
            else:
                # New exception
                self.exception_registry[exc_info.exception_id] = exc_info
            
            # Add to recent exceptions
            self.recent_exceptions.append(exc_info)
            if len(self.recent_exceptions) > self.max_recent_exceptions:
                self.recent_exceptions = self.recent_exceptions[-self.max_recent_exceptions:]
            
            # Trigger security hooks
            for hook in self.exception_hooks:
                try:
                    hook(exc_info)
                except Exception as hook_error:
                    self.logger.error(f"Exception hook failed: {hook_error}")
            
            # Check security thresholds
            self._check_security_thresholds()
            
            # Log based on severity
            if exc_info.severity == ExceptionSeverity.CRITICAL:
                self.logger.critical(f"Critical exception: {exc_info.exception_type} - {exc_info.exception_message}")
            elif exc_info.severity == ExceptionSeverity.HIGH:
                self.logger.error(f"High severity exception: {exc_info.exception_type}")
            elif exc_info.severity == ExceptionSeverity.MEDIUM:
                self.logger.warning(f"Medium severity exception: {exc_info.exception_type}")
            else:
                self.logger.info(f"Low severity exception: {exc_info.exception_type}")
            
            return exc_info
            
        except Exception as e:
            self.logger.error(f"Failed to register exception: {e}")
            # Create minimal exception info as fallback
            return SecureExceptionInfo(
                exception_id="registration_failed",
                exception_type=type(exc).__name__,
                exception_message="Registration failed",
                category=ExceptionCategory.SYSTEM_ERROR,
                severity=ExceptionSeverity.HIGH
            )
    
    def add_security_hook(self, hook: Callable[[SecureExceptionInfo], None]):
        """Add a security hook for exception notifications"""
        self.exception_hooks.append(hook)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get exception monitoring security summary"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            # Recent exceptions by severity
            recent_by_severity = {severity.value: 0 for severity in ExceptionSeverity}
            recent_by_category = {category.value: 0 for category in ExceptionCategory}
            
            for exc_info in self.recent_exceptions:
                if exc_info.timestamp > hour_ago:
                    recent_by_severity[exc_info.severity.value] += 1
                    recent_by_category[exc_info.category.value] += 1
            
            # Top recurring exceptions
            recurring_exceptions = [
                {
                    'exception_type': exc_info.exception_type,
                    'frequency': exc_info.frequency_count,
                    'severity': exc_info.severity.value,
                    'last_seen': exc_info.last_seen.isoformat()
                }
                for exc_info in sorted(
                    self.exception_registry.values(),
                    key=lambda x: x.frequency_count,
                    reverse=True
                )[:10]
                if exc_info.is_recurring
            ]
            
            return {
                'total_exceptions': len(self.exception_registry),
                'recent_exceptions_1h': len([e for e in self.recent_exceptions if e.timestamp > hour_ago]),
                'severity_distribution_1h': recent_by_severity,
                'category_distribution_1h': recent_by_category,
                'recurring_exceptions': recurring_exceptions,
                'security_hooks_registered': len(self.exception_hooks),
                'critical_threshold': self.critical_threshold,
                'high_threshold': self.high_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error generating security summary: {e}")
            return {'error': str(e)}
    
    def _check_security_thresholds(self):
        """Check if security thresholds are exceeded"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            # Count recent exceptions by severity
            critical_count = sum(1 for exc in self.recent_exceptions 
                               if exc.timestamp > hour_ago and exc.severity == ExceptionSeverity.CRITICAL)
            high_count = sum(1 for exc in self.recent_exceptions 
                           if exc.timestamp > hour_ago and exc.severity == ExceptionSeverity.HIGH)
            
            # Check thresholds
            if critical_count >= self.critical_threshold:
                security_error = SecurityError(
                    f"Critical exception threshold exceeded: {critical_count} in last hour",
                    "THRESHOLD_CRITICAL_001"
                )
                security_error_handler.handle_error(security_error)
            
            if high_count >= self.high_threshold:
                security_error = SecurityError(
                    f"High severity exception threshold exceeded: {high_count} in last hour", 
                    "THRESHOLD_HIGH_001"
                )
                security_error_handler.handle_error(security_error)
                
        except Exception as e:
            self.logger.error(f"Error checking security thresholds: {e}")


# Global exception monitoring system
exception_monitor = ExceptionMonitoringSystem()


# Convenience functions
def monitor_exception(exc: Exception, context: Dict[str, Any] = None) -> SecureExceptionInfo:
    """Convenience function to monitor an exception"""
    return exception_monitor.register_exception(exc, context)


def add_exception_security_hook(hook: Callable[[SecureExceptionInfo], None]):
    """Convenience function to add security hook"""
    exception_monitor.add_security_hook(hook)


# Exception decorator for automatic monitoring
def monitor_exceptions(func):
    """Decorator to automatically monitor exceptions in functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            monitor_exception(e, {'function': func.__name__, 'args_count': len(args)})
            raise
    return wrapper