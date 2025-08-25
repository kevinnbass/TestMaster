"""
AgentOps Derived Validation Security Module
Extracted from AgentOps validation.py patterns and secure validation systems
Enhanced for comprehensive trace validation and security monitoring
"""

import re
import time
import json
import hmac
import hashlib
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from .error_handler import SecurityError, ValidationError, security_error_handler


class ValidationLevel(Enum):
    """Validation security levels"""
    BASIC = "basic"
    STANDARD = "standard" 
    ENHANCED = "enhanced"
    STRICT = "strict"


class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ValidationRule:
    """Security validation rule definition"""
    name: str
    description: str
    validation_function: Callable[[Any], Tuple[bool, str]]
    level: ValidationLevel = ValidationLevel.STANDARD
    enabled: bool = True
    max_retries: int = 3
    timeout_seconds: float = 10.0


@dataclass 
class ValidationResult:
    """Comprehensive validation result based on AgentOps patterns"""
    rule_name: str
    status: ValidationStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0
    data_validated: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    security_flags: List[str] = field(default_factory=list)
    
    @property
    def is_successful(self) -> bool:
        """Check if validation was successful"""
        return self.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]


class TraceSecurityValidator:
    """Trace security validation engine based on AgentOps patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Security validation patterns
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'system\s*\(',
            r'subprocess\.',
        ]
        
        # LLM span security indicators
        self.llm_security_indicators = [
            'prompt_injection',
            'jailbreak',
            'system_override',
            'instruction_bypass',
            'context_manipulation'
        ]
    
    def validate_trace_id(self, trace_id: str) -> ValidationResult:
        """Validate trace ID format and security"""
        start_time = time.time()
        
        try:
            # Check format (32 character hex string)
            if not re.match(r'^[0-9a-f]{32}$', trace_id.lower()):
                return ValidationResult(
                    rule_name="trace_id_format",
                    status=ValidationStatus.FAILED,
                    message=f"Invalid trace ID format: {trace_id[:8]}...",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check for suspicious patterns
            security_flags = []
            for pattern in self.suspicious_patterns:
                if re.search(pattern, trace_id, re.IGNORECASE):
                    security_flags.append(f"suspicious_pattern: {pattern}")
            
            if security_flags:
                return ValidationResult(
                    rule_name="trace_id_format",
                    status=ValidationStatus.FAILED,
                    message="Trace ID contains suspicious patterns",
                    security_flags=security_flags,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            return ValidationResult(
                rule_name="trace_id_format",
                status=ValidationStatus.PASSED,
                message="Trace ID format is valid",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="trace_id_format",
                status=ValidationStatus.ERROR,
                message=f"Validation error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def validate_jwt_token(self, token: str) -> ValidationResult:
        """Validate JWT token structure and security"""
        start_time = time.time()
        
        try:
            if not token:
                return ValidationResult(
                    rule_name="jwt_token_validation",
                    status=ValidationStatus.FAILED,
                    message="JWT token is empty or None",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Basic JWT structure check (3 parts separated by dots)
            parts = token.split('.')
            if len(parts) != 3:
                return ValidationResult(
                    rule_name="jwt_token_validation",
                    status=ValidationStatus.FAILED,
                    message="JWT token does not have correct structure (3 parts)",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check each part is base64-like
            for i, part in enumerate(parts):
                if not re.match(r'^[A-Za-z0-9_-]+$', part):
                    return ValidationResult(
                        rule_name="jwt_token_validation", 
                        status=ValidationStatus.FAILED,
                        message=f"JWT token part {i+1} contains invalid characters",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
            
            # Check for reasonable length (not too short or suspiciously long)
            if len(token) < 20 or len(token) > 2048:
                return ValidationResult(
                    rule_name="jwt_token_validation",
                    status=ValidationStatus.WARNING,
                    message=f"JWT token length unusual: {len(token)} characters",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            return ValidationResult(
                rule_name="jwt_token_validation",
                status=ValidationStatus.PASSED,
                message="JWT token structure is valid",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="jwt_token_validation",
                status=ValidationStatus.ERROR,
                message=f"JWT validation error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def validate_span_data(self, span_data: Dict[str, Any]) -> ValidationResult:
        """Validate span data for security issues"""
        start_time = time.time()
        
        try:
            security_flags = []
            
            # Check required fields
            required_fields = ['span_name', 'span_attributes']
            missing_fields = [field for field in required_fields if field not in span_data]
            
            if missing_fields:
                return ValidationResult(
                    rule_name="span_data_validation",
                    status=ValidationStatus.FAILED,
                    message=f"Missing required fields: {missing_fields}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Validate span name
            span_name = span_data.get('span_name', '')
            if not isinstance(span_name, str) or len(span_name) == 0:
                security_flags.append("invalid_span_name")
            
            # Check for suspicious content in span attributes
            attributes = span_data.get('span_attributes', {})
            if isinstance(attributes, dict):
                attributes_str = json.dumps(attributes)
                
                for pattern in self.suspicious_patterns:
                    if re.search(pattern, attributes_str, re.IGNORECASE):
                        security_flags.append(f"suspicious_content: {pattern}")
                
                # Check for LLM security indicators
                for indicator in self.llm_security_indicators:
                    if indicator in attributes_str.lower():
                        security_flags.append(f"llm_security_risk: {indicator}")
            
            # Validate data size (prevent DoS)
            data_size = len(json.dumps(span_data))
            if data_size > 1024 * 1024:  # 1MB limit
                security_flags.append(f"oversized_data: {data_size} bytes")
            
            # Check for nested depth (prevent stack overflow)
            max_depth = self._get_nested_depth(span_data)
            if max_depth > 10:
                security_flags.append(f"excessive_nesting: {max_depth} levels")
            
            if security_flags:
                return ValidationResult(
                    rule_name="span_data_validation",
                    status=ValidationStatus.FAILED,
                    message="Span data contains security issues",
                    security_flags=security_flags,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            return ValidationResult(
                rule_name="span_data_validation",
                status=ValidationStatus.PASSED,
                message="Span data validation passed",
                data_validated={'span_count': 1, 'data_size': data_size},
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="span_data_validation",
                status=ValidationStatus.ERROR,
                message=f"Span validation error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def validate_llm_spans(self, spans: List[Dict[str, Any]]) -> ValidationResult:
        """Validate LLM spans for security based on AgentOps patterns"""
        start_time = time.time()
        
        try:
            llm_spans = []
            security_flags = []
            
            for span in spans:
                span_attributes = span.get('span_attributes', {})
                is_llm_span = False
                
                # Check for LLM span kind (AgentOps pattern)
                span_kind = span_attributes.get('agentops.span.kind', '')
                if not span_kind:
                    # Check nested structure
                    agentops_attrs = span_attributes.get('agentops', {})
                    if isinstance(agentops_attrs, dict):
                        span_attrs = agentops_attrs.get('span', {})
                        if isinstance(span_attrs, dict):
                            span_kind = span_attrs.get('kind', '')
                
                is_llm_span = span_kind == 'llm'
                
                # Alternative check: Look for gen_ai attributes
                if not is_llm_span:
                    gen_ai_attrs = span_attributes.get('gen_ai', {})
                    if isinstance(gen_ai_attrs, dict):
                        if 'prompt' in gen_ai_attrs or 'completion' in gen_ai_attrs:
                            is_llm_span = True
                
                # Check for request type
                if not is_llm_span:
                    request_type = span_attributes.get('gen_ai.request.type', '')
                    if not request_type:
                        request_type = span_attributes.get('llm.request.type', '')
                    if request_type in ['chat', 'completion']:
                        is_llm_span = True
                
                if is_llm_span:
                    llm_spans.append(span)
                    
                    # Security validation for LLM spans
                    self._validate_llm_span_security(span, security_flags)
            
            if not llm_spans:
                return ValidationResult(
                    rule_name="llm_span_validation",
                    status=ValidationStatus.WARNING,
                    message="No LLM spans found in trace",
                    data_validated={'llm_spans_found': 0},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            if security_flags:
                return ValidationResult(
                    rule_name="llm_span_validation",
                    status=ValidationStatus.FAILED,
                    message=f"LLM spans contain security issues",
                    security_flags=security_flags,
                    data_validated={'llm_spans_found': len(llm_spans)},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            return ValidationResult(
                rule_name="llm_span_validation",
                status=ValidationStatus.PASSED,
                message=f"Found {len(llm_spans)} valid LLM spans",
                data_validated={'llm_spans_found': len(llm_spans)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="llm_span_validation",
                status=ValidationStatus.ERROR,
                message=f"LLM span validation error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _validate_llm_span_security(self, span: Dict[str, Any], security_flags: List[str]):
        """Validate individual LLM span for security issues"""
        try:
            attributes = span.get('span_attributes', {})
            
            # Check for prompt injection indicators
            gen_ai_attrs = attributes.get('gen_ai', {})
            if isinstance(gen_ai_attrs, dict):
                prompt = gen_ai_attrs.get('prompt', '')
                completion = gen_ai_attrs.get('completion', '')
                
                # Check for security indicators in prompt/completion
                for text in [prompt, completion]:
                    if isinstance(text, str):
                        text_lower = text.lower()
                        for indicator in self.llm_security_indicators:
                            if indicator in text_lower:
                                security_flags.append(f"llm_security_indicator: {indicator}")
                        
                        # Check for injection patterns
                        for pattern in self.suspicious_patterns:
                            if re.search(pattern, text, re.IGNORECASE):
                                security_flags.append(f"injection_pattern: {pattern}")
        
        except Exception as e:
            security_flags.append(f"validation_error: {str(e)}")
    
    def _get_nested_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate maximum nested depth of object"""
        if depth > 20:  # Prevent infinite recursion
            return depth
        
        if isinstance(obj, dict):
            return max([self._get_nested_depth(v, depth + 1) for v in obj.values()], default=depth)
        elif isinstance(obj, list):
            return max([self._get_nested_depth(item, depth + 1) for item in obj], default=depth)
        else:
            return depth


class ValidationSecurityManager:
    """Central validation security management system"""
    
    def __init__(self):
        self.validator = TraceSecurityValidator()
        self.validation_history: List[ValidationResult] = []
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.security_thresholds = {
            'max_failures_per_hour': 100,
            'max_security_flags_per_hour': 50,
            'critical_failure_threshold': 10
        }
        self.logger = logging.getLogger(__name__)
        
        # Register default validation rules
        self._register_default_rules()
    
    def validate_comprehensive(self, data: Dict[str, Any], 
                             validation_level: ValidationLevel = ValidationLevel.STANDARD) -> List[ValidationResult]:
        """Perform comprehensive validation based on AgentOps patterns"""
        results = []
        
        try:
            # Validate based on data type
            if 'trace_id' in data:
                result = self.validator.validate_trace_id(data['trace_id'])
                results.append(result)
                self.validation_history.append(result)
            
            if 'jwt_token' in data:
                result = self.validator.validate_jwt_token(data['jwt_token'])
                results.append(result)
                self.validation_history.append(result)
            
            if 'spans' in data:
                result = self.validator.validate_llm_spans(data['spans'])
                results.append(result)
                self.validation_history.append(result)
                
                # Validate individual spans
                for span in data['spans']:
                    span_result = self.validator.validate_span_data(span)
                    results.append(span_result)
                    self.validation_history.append(span_result)
            
            # Apply custom validation rules
            for rule_name, rule in self.validation_rules.items():
                if rule.enabled and rule.level.value <= validation_level.value:
                    try:
                        is_valid, message = rule.validation_function(data)
                        status = ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED
                        
                        result = ValidationResult(
                            rule_name=rule_name,
                            status=status,
                            message=message
                        )
                        results.append(result)
                        self.validation_history.append(result)
                        
                    except Exception as e:
                        error_result = ValidationResult(
                            rule_name=rule_name,
                            status=ValidationStatus.ERROR,
                            message=f"Rule execution error: {str(e)}"
                        )
                        results.append(error_result)
                        self.validation_history.append(error_result)
            
            # Limit history size
            if len(self.validation_history) > 10000:
                self.validation_history = self.validation_history[-5000:]
            
            # Check security thresholds
            self._check_security_thresholds(results)
            
            return results
            
        except Exception as e:
            error_result = ValidationResult(
                rule_name="comprehensive_validation",
                status=ValidationStatus.ERROR,
                message=f"Comprehensive validation error: {str(e)}"
            )
            results.append(error_result)
            return results
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add custom validation rule"""
        self.validation_rules[rule.name] = rule
        self.logger.info(f"Added validation rule: {rule.name}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation security summary"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            # Recent validations
            recent_validations = [v for v in self.validation_history if v.timestamp > hour_ago]
            
            # Count by status
            status_counts = {}
            for result in recent_validations:
                status = result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Security flags summary
            all_flags = []
            for result in recent_validations:
                all_flags.extend(result.security_flags)
            
            flag_counts = {}
            for flag in all_flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            
            # Performance metrics
            execution_times = [v.execution_time_ms for v in recent_validations if v.execution_time_ms > 0]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            return {
                'total_validations': len(self.validation_history),
                'recent_validations_1h': len(recent_validations),
                'status_distribution_1h': status_counts,
                'security_flags_1h': flag_counts,
                'avg_execution_time_ms': round(avg_execution_time, 2),
                'validation_rules': len(self.validation_rules),
                'security_thresholds': self.security_thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Error generating validation summary: {e}")
            return {'error': str(e)}
    
    def _register_default_rules(self):
        """Register default security validation rules"""
        def validate_data_size(data: Dict[str, Any]) -> Tuple[bool, str]:
            """Validate data size is reasonable"""
            data_str = json.dumps(data)
            size = len(data_str)
            max_size = 10 * 1024 * 1024  # 10MB
            
            if size > max_size:
                return False, f"Data too large: {size} bytes (max: {max_size})"
            return True, f"Data size OK: {size} bytes"
        
        def validate_no_secrets(data: Dict[str, Any]) -> Tuple[bool, str]:
            """Validate no secrets are exposed in data"""
            data_str = json.dumps(data).lower()
            secret_patterns = ['password', 'secret', 'private_key', 'bearer']
            
            for pattern in secret_patterns:
                if pattern in data_str:
                    return False, f"Potential secret exposed: {pattern}"
            return True, "No secrets detected"
        
        self.validation_rules.update({
            'data_size_check': ValidationRule(
                name='data_size_check',
                description='Validate data size is within limits',
                validation_function=validate_data_size,
                level=ValidationLevel.BASIC
            ),
            'secret_exposure_check': ValidationRule(
                name='secret_exposure_check',
                description='Check for exposed secrets in data',
                validation_function=validate_no_secrets,
                level=ValidationLevel.ENHANCED
            )
        })
    
    def _check_security_thresholds(self, results: List[ValidationResult]):
        """Check if security thresholds are exceeded"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            recent_results = [r for r in self.validation_history if r.timestamp > hour_ago]
            
            # Count failures
            failure_count = sum(1 for r in recent_results if r.status == ValidationStatus.FAILED)
            
            # Count security flags
            security_flag_count = sum(len(r.security_flags) for r in recent_results)
            
            # Check thresholds
            if failure_count >= self.security_thresholds['max_failures_per_hour']:
                security_error = SecurityError(
                    f"Validation failure threshold exceeded: {failure_count} failures in last hour",
                    "VAL_THRESHOLD_001"
                )
                security_error_handler.handle_error(security_error)
            
            if security_flag_count >= self.security_thresholds['max_security_flags_per_hour']:
                security_error = SecurityError(
                    f"Security flag threshold exceeded: {security_flag_count} flags in last hour",
                    "VAL_THRESHOLD_002"
                )
                security_error_handler.handle_error(security_error)
                
        except Exception as e:
            self.logger.error(f"Error checking security thresholds: {e}")


# Global validation security manager
validation_security_manager = ValidationSecurityManager()


def validate_secure_data(data: Dict[str, Any], 
                        level: ValidationLevel = ValidationLevel.STANDARD) -> List[ValidationResult]:
    """Convenience function for secure data validation"""
    return validation_security_manager.validate_comprehensive(data, level)


def add_security_validation_rule(rule: ValidationRule):
    """Convenience function to add validation rule"""
    validation_security_manager.add_validation_rule(rule)