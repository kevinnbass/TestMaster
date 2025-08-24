"""
Agency-Swarm Derived Validation Security Module  
Extracted from agency-swarm/agency_swarm/util/validators.py
Enhanced for comprehensive input validation security
"""

import re
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Type
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from .error_handler import ValidationError, security_error_handler


class ValidationResult(BaseModel):
    """Validation result structure based on agency-swarm pattern"""
    reason: str = Field(..., description="Detailed validation reasoning")
    is_valid: bool = Field(..., description="Whether input passes validation")
    fixed_value: Optional[str] = Field(None, description="Suggested fix if invalid")
    risk_level: str = Field("low", description="Security risk level: low, medium, high, critical")
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class SecurityValidator:
    """Enhanced security validation framework based on agency-swarm patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = {}
        self.blocked_patterns = []
        self.allowed_patterns = []
        
    def register_rule(self, name: str, validator: Callable[[Any], ValidationResult]):
        """Register a validation rule"""
        self.validation_rules[name] = validator
        
    def add_blocked_pattern(self, pattern: str, description: str = ""):
        """Add a blocked regex pattern for security"""
        self.blocked_patterns.append({
            'pattern': re.compile(pattern, re.IGNORECASE),
            'description': description
        })
        
    def add_allowed_pattern(self, pattern: str, description: str = ""):
        """Add an allowed regex pattern"""
        self.allowed_patterns.append({
            'pattern': re.compile(pattern, re.IGNORECASE),
            'description': description
        })
        
    def validate_input(self, value: Any, rules: List[str] = None) -> ValidationResult:
        """Validate input against specified rules"""
        try:
            # Convert to string for pattern matching
            str_value = str(value) if value is not None else ""
            
            # Check blocked patterns first
            for blocked in self.blocked_patterns:
                if blocked['pattern'].search(str_value):
                    return ValidationResult(
                        reason=f"Input contains blocked pattern: {blocked['description']}",
                        is_valid=False,
                        risk_level="high"
                    )
            
            # Apply specific validation rules
            if rules:
                for rule_name in rules:
                    if rule_name in self.validation_rules:
                        result = self.validation_rules[rule_name](value)
                        if not result.is_valid:
                            return result
            
            return ValidationResult(
                reason="Input passed all validation checks",
                is_valid=True,
                risk_level="low"
            )
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult(
                reason=f"Validation failed with error: {str(e)}",
                is_valid=False,
                risk_level="medium"
            )


class InputSanitizer:
    """Input sanitization utilities"""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return str(value)
            
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        return sanitized
    
    @staticmethod
    def sanitize_json(value: str) -> Optional[Dict[str, Any]]:
        """Safely parse and sanitize JSON"""
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return {k: InputSanitizer.sanitize_string(str(v)) for k, v in data.items()}
            return None
        except (json.JSONDecodeError, TypeError):
            return None
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security"""
        # Remove path traversal attempts
        filename = re.sub(r'\.\./', '', filename)
        filename = re.sub(r'\.\.\\', '', filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Limit length
        return filename[:255]


def create_security_validator(
    statement: str,
    risk_level: str = "medium",
    allow_override: bool = False
) -> Callable[[Any], Any]:
    """Create a security validator function based on agency-swarm pattern"""
    
    validator = SecurityValidator()
    
    def validate_value(value: Any) -> Any:
        """Validate a value against the security statement"""
        try:
            # Basic security checks
            str_value = str(value) if value is not None else ""
            
            # Check for injection attempts
            injection_patterns = [
                r'<script[^>]*>.*?</script>',  # XSS
                r'javascript:',  # JavaScript injection
                r'data:text/html',  # Data URL XSS
                r'(union|select|insert|update|delete|drop)\s+',  # SQL injection
                r'(\.\./){2,}',  # Path traversal
                r'proc_open|exec|system|shell_exec',  # Command injection
            ]
            
            for pattern in injection_patterns:
                if re.search(pattern, str_value, re.IGNORECASE):
                    error = ValidationError(
                        f"Input validation failed: {statement}. Detected potential injection attack.",
                        details={'pattern_matched': pattern, 'input_value': str_value[:100]}
                    )
                    security_error_handler.handle_error(error)
                    
                    if not allow_override:
                        raise error
                    
                    # Return sanitized version if override allowed
                    return InputSanitizer.sanitize_string(str_value)
            
            # Statement-specific validation would go here
            # For now, return the value if it passes basic security
            return value
            
        except ValidationError:
            raise
        except Exception as e:
            error = ValidationError(
                f"Validation error: {str(e)}",
                details={'statement': statement, 'input_value': str(value)[:100]}
            )
            security_error_handler.handle_error(error)
            raise error
    
    return validate_value


# Global validator instance
global_validator = SecurityValidator()

# Add default security patterns
global_validator.add_blocked_pattern(
    r'<script[^>]*>.*?</script>', 
    "XSS script injection"
)
global_validator.add_blocked_pattern(
    r'(union|select|insert|update|delete|drop)\s+.*from', 
    "SQL injection attempt"
)
global_validator.add_blocked_pattern(
    r'(\.\./){2,}', 
    "Path traversal attempt"
)


def validate_security_input(value: Any, rules: List[str] = None) -> ValidationResult:
    """Convenience function for security input validation"""
    return global_validator.validate_input(value, rules)


def sanitize_input(value: Any) -> Any:
    """Convenience function for input sanitization"""
    if isinstance(value, str):
        return InputSanitizer.sanitize_string(value)
    elif isinstance(value, dict):
        return {k: sanitize_input(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_input(item) for item in value]
    return value