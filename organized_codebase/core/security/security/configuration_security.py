"""
AgentOps Derived Configuration Security Module
Extracted from AgentOps config.py patterns and secure configuration management
Enhanced for comprehensive configuration validation and security
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Set, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID
from .error_handler import SecurityError, ValidationError, security_error_handler


class ConfigSecurityLevel(Enum):
    """Configuration security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


@dataclass
class ConfigValidationRule:
    """Configuration validation rule"""
    name: str
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required: bool = True
    security_level: ConfigSecurityLevel = ConfigSecurityLevel.INTERNAL
    allowed_values: Optional[Set[str]] = None
    description: str = ""


@dataclass
class SecureConfigField:
    """Secure configuration field with validation and metadata"""
    name: str
    value: Any
    security_level: ConfigSecurityLevel = ConfigSecurityLevel.INTERNAL
    is_masked: bool = False
    validation_rules: List[ConfigValidationRule] = field(default_factory=list)
    last_validated: Optional[datetime] = None
    validation_errors: List[str] = field(default_factory=list)
    source: str = "default"  # default, env, file, runtime
    
    def __post_init__(self):
        # Automatically mask sensitive fields
        if self.security_level in [ConfigSecurityLevel.CONFIDENTIAL, ConfigSecurityLevel.SECRET]:
            self.is_masked = True
    
    @property
    def display_value(self) -> str:
        """Get display-safe value (masked if sensitive)"""
        if self.is_masked and self.value:
            if isinstance(self.value, str) and len(self.value) > 8:
                return f"{self.value[:4]}{'*' * (len(self.value) - 8)}{self.value[-4:]}"
            else:
                return "***masked***"
        return str(self.value) if self.value is not None else "None"


class ConfigurationValidator:
    """Configuration security validator based on AgentOps patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # AgentOps-inspired validation rules
        self.default_rules = {
            'api_key': ConfigValidationRule(
                name='api_key',
                pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                required=True,
                security_level=ConfigSecurityLevel.SECRET,
                description="API key must be a valid UUID format"
            ),
            'endpoint': ConfigValidationRule(
                name='endpoint',
                pattern=r'^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})+(?:/.*)?$',
                required=True,
                security_level=ConfigSecurityLevel.INTERNAL,
                description="Endpoint must be a valid HTTPS URL"
            ),
            'max_wait_time': ConfigValidationRule(
                name='max_wait_time',
                min_length=100,
                max_length=30000,
                security_level=ConfigSecurityLevel.PUBLIC,
                description="Wait time must be between 100ms and 30s"
            ),
            'max_queue_size': ConfigValidationRule(
                name='max_queue_size',
                min_length=1,
                max_length=10000,
                security_level=ConfigSecurityLevel.INTERNAL,
                description="Queue size must be between 1 and 10000"
            ),
            'log_level': ConfigValidationRule(
                name='log_level',
                allowed_values={'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
                security_level=ConfigSecurityLevel.PUBLIC,
                description="Log level must be a valid logging level"
            )
        }
        
        # Dangerous configuration patterns
        self.dangerous_patterns = [
            r'password.*=.*[^*]',  # Exposed passwords
            r'secret.*=.*[^*]',    # Exposed secrets
            r'token.*=.*[^*]',     # Exposed tokens
            r'eval\s*\(',          # Code injection
            r'exec\s*\(',          # Code execution
            r'__import__',         # Import hijacking
        ]
    
    def validate_field(self, field: SecureConfigField) -> bool:
        """Validate a single configuration field"""
        try:
            field.validation_errors = []
            field.last_validated = datetime.utcnow()
            
            # Check if field has specific rules
            rules = field.validation_rules or []
            if field.name in self.default_rules:
                rules.append(self.default_rules[field.name])
            
            # Apply validation rules
            for rule in rules:
                rule_errors = self._apply_rule(field, rule)
                field.validation_errors.extend(rule_errors)
            
            # Check for dangerous patterns
            if isinstance(field.value, str):
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, field.value, re.IGNORECASE):
                        field.validation_errors.append(f"Contains dangerous pattern: {pattern}")
            
            # Special validation for API keys (UUID format check)
            if field.name == 'api_key' and field.value:
                try:
                    UUID(field.value)
                except ValueError:
                    field.validation_errors.append("API key format is invalid (not a valid UUID)")
            
            is_valid = len(field.validation_errors) == 0
            if not is_valid:
                self.logger.warning(f"Configuration field '{field.name}' failed validation: {field.validation_errors}")
            
            return is_valid
            
        except Exception as e:
            field.validation_errors.append(f"Validation error: {str(e)}")
            self.logger.error(f"Error validating field '{field.name}': {e}")
            return False
    
    def _apply_rule(self, field: SecureConfigField, rule: ConfigValidationRule) -> List[str]:
        """Apply a specific validation rule to a field"""
        errors = []
        
        try:
            # Required field check
            if rule.required and (field.value is None or field.value == ""):
                errors.append(f"Field '{field.name}' is required")
                return errors
            
            # Skip further validation if field is empty and not required
            if not rule.required and (field.value is None or field.value == ""):
                return errors
            
            value_str = str(field.value)
            
            # Pattern validation
            if rule.pattern and not re.match(rule.pattern, value_str):
                errors.append(f"Field '{field.name}' does not match required pattern")
            
            # Length validation for strings
            if isinstance(field.value, str):
                if rule.min_length and len(field.value) < rule.min_length:
                    errors.append(f"Field '{field.name}' is too short (min: {rule.min_length})")
                if rule.max_length and len(field.value) > rule.max_length:
                    errors.append(f"Field '{field.name}' is too long (max: {rule.max_length})")
            
            # Numeric range validation
            elif isinstance(field.value, (int, float)):
                if rule.min_length and field.value < rule.min_length:
                    errors.append(f"Field '{field.name}' is too small (min: {rule.min_length})")
                if rule.max_length and field.value > rule.max_length:
                    errors.append(f"Field '{field.name}' is too large (max: {rule.max_length})")
            
            # Allowed values validation
            if rule.allowed_values:
                if isinstance(field.value, str):
                    if field.value.upper() not in rule.allowed_values:
                        errors.append(f"Field '{field.name}' has invalid value. Allowed: {rule.allowed_values}")
                else:
                    if str(field.value) not in rule.allowed_values:
                        errors.append(f"Field '{field.name}' has invalid value. Allowed: {rule.allowed_values}")
            
            return errors
            
        except Exception as e:
            return [f"Rule application error: {str(e)}"]


class SecureConfigurationManager:
    """Secure configuration management system based on AgentOps patterns"""
    
    def __init__(self):
        self.validator = ConfigurationValidator()
        self.config_fields: Dict[str, SecureConfigField] = {}
        self.config_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Track configuration changes
        self.change_audit: List[Dict[str, Any]] = []
        
        # Environment variable patterns to scan
        self.env_patterns = [
            'AGENTOPS_*',
            '*_API_KEY',
            '*_SECRET',
            '*_TOKEN',
            '*_PASSWORD'
        ]
    
    def load_from_environment(self, prefix: str = "AGENTOPS_") -> Dict[str, SecureConfigField]:
        """Load configuration from environment variables"""
        loaded_fields = {}
        
        try:
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    field_name = key[len(prefix):].lower()
                    
                    # Determine security level based on field name
                    security_level = self._determine_security_level(field_name)
                    
                    field = SecureConfigField(
                        name=field_name,
                        value=value,
                        security_level=security_level,
                        source="env"
                    )
                    
                    # Validate field
                    is_valid = self.validator.validate_field(field)
                    
                    loaded_fields[field_name] = field
                    self.config_fields[field_name] = field
                    
                    if is_valid:
                        self.logger.info(f"Loaded valid config field: {field_name}")
                    else:
                        self.logger.warning(f"Loaded invalid config field: {field_name} - {field.validation_errors}")
            
            # Audit the load operation
            self._audit_change("environment_load", {
                'fields_loaded': len(loaded_fields),
                'source': 'environment'
            })
            
            return loaded_fields
            
        except Exception as e:
            error = SecurityError(f"Failed to load configuration from environment: {str(e)}", "CONFIG_ENV_001")
            security_error_handler.handle_error(error)
            return {}
    
    def set_field(self, name: str, value: Any, 
                  security_level: ConfigSecurityLevel = ConfigSecurityLevel.INTERNAL,
                  source: str = "runtime") -> bool:
        """Set a configuration field with validation"""
        try:
            # Create field
            field = SecureConfigField(
                name=name,
                value=value,
                security_level=security_level,
                source=source
            )
            
            # Validate field
            is_valid = self.validator.validate_field(field)
            
            # Store field even if invalid (for audit purposes)
            old_value = None
            if name in self.config_fields:
                old_value = self.config_fields[name].display_value
            
            self.config_fields[name] = field
            
            # Audit the change
            self._audit_change("field_set", {
                'field_name': name,
                'old_value': old_value,
                'new_value': field.display_value,
                'is_valid': is_valid,
                'source': source
            })
            
            if is_valid:
                self.logger.info(f"Successfully set configuration field: {name}")
            else:
                self.logger.warning(f"Set invalid configuration field: {name} - {field.validation_errors}")
            
            return is_valid
            
        except Exception as e:
            error = SecurityError(f"Failed to set configuration field '{name}': {str(e)}", "CONFIG_SET_001")
            security_error_handler.handle_error(error)
            return False
    
    def get_field(self, name: str) -> Optional[SecureConfigField]:
        """Get a configuration field"""
        return self.config_fields.get(name)
    
    def get_value(self, name: str, default: Any = None) -> Any:
        """Get a configuration value"""
        field = self.config_fields.get(name)
        return field.value if field else default
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration fields"""
        validation_results = {}
        
        try:
            for name, field in self.config_fields.items():
                is_valid = self.validator.validate_field(field)
                validation_results[name] = field.validation_errors
            
            # Count invalid fields
            invalid_count = sum(1 for errors in validation_results.values() if errors)
            
            self.logger.info(f"Configuration validation complete. {invalid_count} invalid fields found.")
            
            return validation_results
            
        except Exception as e:
            error = SecurityError(f"Configuration validation failed: {str(e)}", "CONFIG_VAL_001")
            security_error_handler.handle_error(error)
            return {"validation_error": [str(e)]}
    
    def export_safe_config(self) -> Dict[str, Any]:
        """Export configuration with sensitive values masked"""
        try:
            safe_config = {}
            
            for name, field in self.config_fields.items():
                safe_config[name] = {
                    'value': field.display_value,
                    'security_level': field.security_level.value,
                    'source': field.source,
                    'is_valid': len(field.validation_errors) == 0,
                    'last_validated': field.last_validated.isoformat() if field.last_validated else None
                }
            
            return safe_config
            
        except Exception as e:
            self.logger.error(f"Error exporting safe config: {e}")
            return {}
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get configuration security summary"""
        try:
            total_fields = len(self.config_fields)
            
            # Count by security level
            level_counts = {}
            invalid_count = 0
            
            for field in self.config_fields.values():
                level = field.security_level.value
                level_counts[level] = level_counts.get(level, 0) + 1
                
                if field.validation_errors:
                    invalid_count += 1
            
            # Recent changes
            recent_changes = len([
                change for change in self.change_audit
                if datetime.fromisoformat(change['timestamp']) > datetime.utcnow().replace(hour=datetime.utcnow().hour-1)
            ])
            
            return {
                'total_fields': total_fields,
                'invalid_fields': invalid_count,
                'security_level_distribution': level_counts,
                'recent_changes_1h': recent_changes,
                'total_changes': len(self.change_audit),
                'validation_rules': len(self.validator.default_rules)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating security summary: {e}")
            return {'error': str(e)}
    
    def _determine_security_level(self, field_name: str) -> ConfigSecurityLevel:
        """Determine security level based on field name"""
        field_lower = field_name.lower()
        
        if any(sensitive in field_lower for sensitive in ['api_key', 'secret', 'token', 'password']):
            return ConfigSecurityLevel.SECRET
        elif any(conf in field_lower for conf in ['endpoint', 'url', 'host']):
            return ConfigSecurityLevel.CONFIDENTIAL
        elif any(internal in field_lower for internal in ['queue', 'timeout', 'retry', 'max_']):
            return ConfigSecurityLevel.INTERNAL
        else:
            return ConfigSecurityLevel.PUBLIC
    
    def _audit_change(self, action: str, details: Dict[str, Any]):
        """Audit configuration changes"""
        self.change_audit.append({
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'details': details
        })
        
        # Keep audit log manageable
        if len(self.change_audit) > 1000:
            self.change_audit = self.change_audit[-500:]


# Global secure configuration manager
secure_config_manager = SecureConfigurationManager()


def load_secure_config(prefix: str = "AGENTOPS_") -> Dict[str, SecureConfigField]:
    """Convenience function to load secure configuration"""
    return secure_config_manager.load_from_environment(prefix)


def validate_config_security() -> Dict[str, List[str]]:
    """Convenience function to validate configuration security"""
    return secure_config_manager.validate_all()