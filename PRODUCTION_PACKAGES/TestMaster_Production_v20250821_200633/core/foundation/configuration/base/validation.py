"""
Configuration Validation
========================

Comprehensive validation framework for hierarchical configuration management.

Author: Agent E - Infrastructure Consolidation
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    key: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.key}: {self.message}"


class ConfigurationValidator:
    """
    Comprehensive configuration validation framework.
    
    Provides validation rules, schema validation, and hierarchical
    configuration validation across all TestMaster layers.
    """
    
    def __init__(self):
        self.validators: Dict[str, List[Callable]] = {}
        self.global_validators: List[Callable] = []
        self.schema_cache: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in validators
        self._register_builtin_validators()
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def validate(self, config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate configuration against all applicable rules."""
        results = []
        
        # Schema validation if provided
        if schema:
            schema_results = self._validate_schema(config, schema)
            results.extend(schema_results)
        
        # Key-specific validation
        for key, value in config.items():
            key_results = self._validate_key(key, value)
            results.extend(key_results)
        
        # Global validation rules
        global_results = self._validate_global(config)
        results.extend(global_results)
        
        self.logger.debug(f"Validation completed: {len(results)} issues found")
        return results
    
    def add_validator(self, key: str, validator: Callable[[Any], Optional[str]]):
        """Add validator for specific configuration key."""
        if key not in self.validators:
            self.validators[key] = []
        self.validators[key].append(validator)
        self.logger.debug(f"Added validator for key: {key}")
    
    def add_global_validator(self, validator: Callable[[Dict[str, Any]], List[ValidationResult]]):
        """Add global configuration validator."""
        self.global_validators.append(validator)
        self.logger.debug("Added global validator")
    
    def validate_type(self, key: str, value: Any, expected_type: type) -> Optional[ValidationResult]:
        """Validate value type."""
        if not isinstance(value, expected_type):
            return ValidationResult(
                key=key,
                message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value,
                expected=expected_type.__name__
            )
        return None
    
    def validate_range(self, key: str, value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                      max_val: Optional[Union[int, float]] = None) -> Optional[ValidationResult]:
        """Validate numeric range."""
        if min_val is not None and value < min_val:
            return ValidationResult(
                key=key,
                message=f"Value {value} is below minimum {min_val}",
                severity=ValidationSeverity.ERROR,
                value=value,
                expected=f">= {min_val}"
            )
        
        if max_val is not None and value > max_val:
            return ValidationResult(
                key=key,
                message=f"Value {value} is above maximum {max_val}",
                severity=ValidationSeverity.ERROR,
                value=value,
                expected=f"<= {max_val}"
            )
        
        return None
    
    def validate_pattern(self, key: str, value: str, pattern: str) -> Optional[ValidationResult]:
        """Validate string pattern."""
        if not isinstance(value, str):
            return ValidationResult(
                key=key,
                message=f"Pattern validation requires string, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            )
        
        if not re.match(pattern, value):
            return ValidationResult(
                key=key,
                message=f"Value '{value}' does not match pattern '{pattern}'",
                severity=ValidationSeverity.ERROR,
                value=value,
                expected=pattern
            )
        
        return None
    
    def validate_choices(self, key: str, value: Any, choices: List[Any]) -> Optional[ValidationResult]:
        """Validate value is in allowed choices."""
        if value not in choices:
            return ValidationResult(
                key=key,
                message=f"Value '{value}' not in allowed choices: {choices}",
                severity=ValidationSeverity.ERROR,
                value=value,
                expected=choices
            )
        return None
    
    def validate_required(self, config: Dict[str, Any], required_keys: List[str]) -> List[ValidationResult]:
        """Validate required keys are present."""
        results = []
        for key in required_keys:
            if key not in config:
                results.append(ValidationResult(
                    key=key,
                    message=f"Required configuration key '{key}' is missing",
                    severity=ValidationSeverity.ERROR
                ))
        return results
    
    def validate_dependencies(self, config: Dict[str, Any], dependencies: Dict[str, List[str]]) -> List[ValidationResult]:
        """Validate configuration dependencies."""
        results = []
        
        for key, deps in dependencies.items():
            if key in config:
                for dep in deps:
                    if dep not in config:
                        results.append(ValidationResult(
                            key=key,
                            message=f"Configuration '{key}' requires '{dep}' to be set",
                            severity=ValidationSeverity.ERROR,
                            value=config.get(key)
                        ))
        
        return results
    
    def validate_mutually_exclusive(self, config: Dict[str, Any], 
                                   exclusive_groups: List[List[str]]) -> List[ValidationResult]:
        """Validate mutually exclusive configuration groups."""
        results = []
        
        for group in exclusive_groups:
            present_keys = [key for key in group if key in config]
            if len(present_keys) > 1:
                results.append(ValidationResult(
                    key = os.getenv('KEY').join(present_keys),
                    message=f"Mutually exclusive keys cannot be set together: {present_keys}",
                    severity=ValidationSeverity.ERROR,
                    value=present_keys
                ))
        
        return results
    
    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[ValidationResult]:
        """Validate configuration against schema."""
        results = []
        
        # Validate required fields
        required = schema.get('required', [])
        results.extend(self.validate_required(config, required))
        
        # Validate field properties
        properties = schema.get('properties', {})
        for key, prop_schema in properties.items():
            if key in config:
                field_results = self._validate_field_schema(key, config[key], prop_schema)
                results.extend(field_results)
        
        # Validate dependencies
        dependencies = schema.get('dependencies', {})
        results.extend(self.validate_dependencies(config, dependencies))
        
        # Validate mutually exclusive groups
        exclusive_groups = schema.get('mutually_exclusive', [])
        results.extend(self.validate_mutually_exclusive(config, exclusive_groups))
        
        return results
    
    def _validate_field_schema(self, key: str, value: Any, schema: Dict[str, Any]) -> List[ValidationResult]:
        """Validate individual field against schema."""
        results = []
        
        # Type validation
        if 'type' in schema:
            expected_type = self._get_type_from_string(schema['type'])
            if expected_type:
                type_result = self.validate_type(key, value, expected_type)
                if type_result:
                    results.append(type_result)
        
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            if 'minimum' in schema or 'maximum' in schema:
                range_result = self.validate_range(
                    key, value, 
                    schema.get('minimum'), 
                    schema.get('maximum')
                )
                if range_result:
                    results.append(range_result)
        
        # Pattern validation for strings
        if isinstance(value, str) and 'pattern' in schema:
            pattern_result = self.validate_pattern(key, value, schema['pattern'])
            if pattern_result:
                results.append(pattern_result)
        
        # Choices validation
        if 'enum' in schema:
            choice_result = self.validate_choices(key, value, schema['enum'])
            if choice_result:
                results.append(choice_result)
        
        return results
    
    def _validate_key(self, key: str, value: Any) -> List[ValidationResult]:
        """Validate specific key using registered validators."""
        results = []
        
        if key in self.validators:
            for validator in self.validators[key]:
                try:
                    error_message = validator(value)
                    if error_message:
                        results.append(ValidationResult(
                            key=key,
                            message=error_message,
                            severity=ValidationSeverity.ERROR,
                            value=value
                        ))
                except Exception as e:
                    results.append(ValidationResult(
                        key=key,
                        message=f"Validator error: {str(e)}",
                        severity=ValidationSeverity.WARNING,
                        value=value
                    ))
        
        return results
    
    def _validate_global(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Run global validation rules."""
        results = []
        
        for validator in self.global_validators:
            try:
                validator_results = validator(config)
                if validator_results:
                    results.extend(validator_results)
            except Exception as e:
                results.append(ValidationResult(
                    key = os.getenv('KEY'),
                    message=f"Global validator error: {str(e)}",
                    severity=ValidationSeverity.WARNING
                ))
        
        return results
    
    def _register_builtin_validators(self):
        """Register built-in validation rules."""
        # API Key validation
        self.add_validator('api_key', lambda x: None if isinstance(x, str) and len(x) > 10 
                          else "API key must be a string with more than 10 characters")
        
        # Port validation
        self.add_validator('port', lambda x: None if isinstance(x, int) and 1 <= x <= 65535 
                          else "Port must be an integer between 1 and 65535")
        
        # URL validation
        self.add_validator('url', self._validate_url)
        
        # Timeout validation
        self.add_validator('timeout', lambda x: None if isinstance(x, (int, float)) and x > 0 
                          else "Timeout must be a positive number")
        
        # Boolean validation for common flags
        for flag in ['debug', 'verbose', 'enabled', 'disabled']:
            self.add_validator(flag, lambda x: None if isinstance(x, bool) 
                              else f"{flag} must be a boolean")
    
    def _validate_url(self, url: str) -> Optional[str]:
        """Validate URL format."""
        if not isinstance(url, str):
            return "URL must be a string"
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, url):
            return "Invalid URL format"
        
        return None
    
    def _get_type_from_string(self, type_string: str) -> Optional[type]:
        """Convert type string to Python type."""
        type_map = {
            'string': str,
            'str': str,
            'integer': int,
            'int': int,
            'number': float,
            'float': float,
            'boolean': bool,
            'bool': bool,
            'list': list,
            'array': list,
            'dict': dict,
            'object': dict
        }
        return type_map.get(type_string.lower())


# Export key classes
__all__ = [
    'ValidationSeverity',
    'ValidationResult',
    'ConfigurationValidator'
]