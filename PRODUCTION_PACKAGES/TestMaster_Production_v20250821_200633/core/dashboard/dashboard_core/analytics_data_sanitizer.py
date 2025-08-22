"""
Analytics Data Sanitization and Validation Engine
================================================

Advanced real-time data sanitization, validation, and cleaning system
for analytics data to ensure only clean, validated data reaches the dashboard.

Author: TestMaster Team
"""

import logging
import re
import json
import html
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

class DataType(Enum):
    NUMERIC = "numeric"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    SQL_QUERY = "sql_query"
    FILE_PATH = "file_path"
    IDENTIFIER = "identifier"

@dataclass
class ValidationRule:
    """Data validation rule definition."""
    rule_id: str
    field_path: str
    data_type: DataType
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    sanitizer: Optional[Callable[[Any], Any]] = None
    description: str = ""

@dataclass
class ValidationIssue:
    """Data validation issue found during processing."""
    issue_id: str
    rule_id: str
    field_path: str
    severity: str  # error, warning, info
    message: str
    original_value: Any
    suggested_fix: Optional[Any] = None
    timestamp: datetime = None

class AnalyticsDataSanitizer:
    """
    Advanced data sanitization and validation engine for analytics data.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE,
                 max_issues_per_batch: int = 1000):
        """
        Initialize analytics data sanitizer.
        
        Args:
            validation_level: Level of validation strictness
            max_issues_per_batch: Maximum validation issues to collect per batch
        """
        self.validation_level = validation_level
        self.max_issues_per_batch = max_issues_per_batch
        
        # Validation rules
        self.validation_rules = {}
        self.field_sanitizers = {}
        
        # Issue tracking
        self.validation_issues = deque(maxlen=max_issues_per_batch)
        self.issue_patterns = defaultdict(int)
        self.sanitization_stats = {
            'records_processed': 0,
            'issues_found': 0,
            'issues_fixed': 0,
            'records_rejected': 0,
            'data_sanitized_kb': 0,
            'start_time': datetime.now()
        }
        
        # Threat detection
        self.threat_patterns = self._setup_threat_patterns()
        self.suspicious_patterns = defaultdict(int)
        
        # Data cleansing operations
        self.cleansing_operations = []
        
        # Threading for background processing
        self.sanitizer_active = False
        self.background_thread = None
        self.pending_data = deque(maxlen=10000)
        
        # Setup default rules
        self._setup_default_validation_rules()
        self._setup_default_sanitizers()
        
        logger.info(f"Analytics Data Sanitizer initialized: {validation_level.value}")
    
    def start_background_processing(self):
        """Start background data sanitization processing."""
        if self.sanitizer_active:
            return
        
        self.sanitizer_active = True
        self.background_thread = threading.Thread(target=self._background_processing_loop, daemon=True)
        self.background_thread.start()
        
        logger.info("Analytics data sanitizer background processing started")
    
    def stop_background_processing(self):
        """Stop background processing."""
        self.sanitizer_active = False
        
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)
        
        logger.info("Analytics data sanitizer background processing stopped")
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.validation_rules[rule.rule_id] = rule
        logger.info(f"Added validation rule: {rule.rule_id}")
    
    def add_field_sanitizer(self, field_path: str, sanitizer: Callable[[Any], Any]):
        """Add a field-specific sanitizer function."""
        self.field_sanitizers[field_path] = sanitizer
        logger.info(f"Added field sanitizer for: {field_path}")
    
    def sanitize_and_validate(self, data: Dict[str, Any], 
                            source: str = "unknown") -> Tuple[Dict[str, Any], List[ValidationIssue]]:
        """
        Sanitize and validate analytics data.
        
        Args:
            data: Raw analytics data
            source: Data source identifier
        
        Returns:
            Tuple of (sanitized_data, validation_issues)
        """
        issues = []
        sanitized_data = {}
        
        try:
            # Start with deep copy of data
            import copy
            sanitized_data = copy.deepcopy(data)
            
            # Phase 1: Threat detection and removal
            threat_issues = self._detect_and_remove_threats(sanitized_data, source)
            issues.extend(threat_issues)
            
            # Phase 2: Data type validation and conversion
            type_issues = self._validate_and_convert_types(sanitized_data, source)
            issues.extend(type_issues)
            
            # Phase 3: Field-specific sanitization
            sanitization_issues = self._apply_field_sanitizers(sanitized_data, source)
            issues.extend(sanitization_issues)
            
            # Phase 4: Business rule validation
            business_issues = self._validate_business_rules(sanitized_data, source)
            issues.extend(business_issues)
            
            # Phase 5: Data consistency checks
            consistency_issues = self._check_data_consistency(sanitized_data, source)
            issues.extend(consistency_issues)
            
            # Phase 6: Final cleanup
            sanitized_data = self._final_data_cleanup(sanitized_data)
            
            # Update statistics
            self.sanitization_stats['records_processed'] += 1
            self.sanitization_stats['issues_found'] += len(issues)
            self.sanitization_stats['data_sanitized_kb'] += len(str(sanitized_data)) / 1024
            
            # Track issue patterns
            for issue in issues:
                self.issue_patterns[f"{issue.rule_id}:{issue.severity}"] += 1
            
            # Store issues for analysis
            self.validation_issues.extend(issues)
            
            logger.debug(f"Sanitized data from {source}: {len(issues)} issues found")
            
        except Exception as e:
            error_issue = ValidationIssue(
                issue_id=f"sanitization_error_{int(time.time())}",
                rule_id="sanitization_engine",
                field_path="root",
                severity="error",
                message=f"Sanitization failed: {str(e)}",
                original_value=data,
                timestamp=datetime.now()
            )
            issues.append(error_issue)
            logger.error(f"Sanitization error for {source}: {e}")
        
        return sanitized_data, issues
    
    def queue_for_sanitization(self, data: Dict[str, Any], source: str = "unknown"):
        """Queue data for background sanitization."""
        self.pending_data.append({
            'data': data,
            'source': source,
            'timestamp': datetime.now()
        })
    
    def get_sanitization_summary(self) -> Dict[str, Any]:
        """Get sanitization system summary."""
        uptime = (datetime.now() - self.sanitization_stats['start_time']).total_seconds()
        
        # Calculate recent issue rates
        recent_issues = [i for i in self.validation_issues 
                        if (datetime.now() - (i.timestamp or datetime.now())).total_seconds() < 3600]
        
        issue_severity_counts = defaultdict(int)
        for issue in recent_issues:
            issue_severity_counts[issue.severity] += 1
        
        return {
            'validation_level': self.validation_level.value,
            'sanitizer_active': self.sanitizer_active,
            'uptime_seconds': uptime,
            'statistics': self.sanitization_stats.copy(),
            'validation_rules': {
                'total_rules': len(self.validation_rules),
                'field_sanitizers': len(self.field_sanitizers)
            },
            'recent_activity': {
                'issues_last_hour': len(recent_issues),
                'issue_severity_distribution': dict(issue_severity_counts),
                'pending_queue_size': len(self.pending_data),
                'top_issue_patterns': dict(sorted(self.issue_patterns.items(), 
                                                key=lambda x: x[1], reverse=True)[:10])
            },
            'threat_detection': {
                'suspicious_patterns_detected': len(self.suspicious_patterns),
                'threat_patterns_configured': len(self.threat_patterns)
            }
        }
    
    def _setup_default_validation_rules(self):
        """Setup default validation rules for common analytics fields."""
        
        # Numeric metrics validation
        self.validation_rules['cpu_usage'] = ValidationRule(
            rule_id='cpu_usage',
            field_path='cpu_usage_percent',
            data_type=DataType.NUMERIC,
            min_value=0.0,
            max_value=100.0,
            description='CPU usage percentage validation'
        )
        
        self.validation_rules['memory_usage'] = ValidationRule(
            rule_id='memory_usage',
            field_path='memory_usage_percent',
            data_type=DataType.NUMERIC,
            min_value=0.0,
            max_value=100.0,
            description='Memory usage percentage validation'
        )
        
        self.validation_rules['response_time'] = ValidationRule(
            rule_id='response_time',
            field_path='response_time_ms',
            data_type=DataType.NUMERIC,
            min_value=0.0,
            max_value=300000.0,  # 5 minutes max
            description='Response time validation'
        )
        
        # String field validation
        self.validation_rules['component_name'] = ValidationRule(
            rule_id='component_name',
            field_path='component',
            data_type=DataType.STRING,
            required=True,
            min_length=1,
            max_length=100,
            pattern=r'^[a-zA-Z0-9_\-\.]+$',
            description='Component name validation'
        )
        
        # Timestamp validation
        self.validation_rules['timestamp'] = ValidationRule(
            rule_id='timestamp',
            field_path='timestamp',
            data_type=DataType.DATE,
            required=True,
            description='Timestamp validation'
        )
        
        # Email validation (if present)
        self.validation_rules['email_fields'] = ValidationRule(
            rule_id='email_fields',
            field_path='*email*',
            data_type=DataType.EMAIL,
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            description='Email field validation'
        )
        
        # URL validation
        self.validation_rules['url_fields'] = ValidationRule(
            rule_id='url_fields',
            field_path='*url*',
            data_type=DataType.URL,
            max_length=2000,
            description='URL field validation'
        )
    
    def _setup_default_sanitizers(self):
        """Setup default field sanitizers."""
        
        # HTML sanitizer
        def html_sanitizer(value):
            if isinstance(value, str):
                return html.escape(value)
            return value
        
        # SQL injection prevention
        def sql_sanitizer(value):
            if isinstance(value, str):
                # Remove common SQL injection patterns
                dangerous_patterns = [
                    r"('\s*(or|and)\s*'.*')",
                    r"(\s*(union|select|insert|update|delete|drop|create|alter|exec|execute)\s+)",
                    r"(-{2,}|/\*|\*/)",
                    r"(;|\||&)"
                ]
                for pattern in dangerous_patterns:
                    value = re.sub(pattern, '', value, flags=re.IGNORECASE)
            return value
        
        # Numeric sanitizer
        def numeric_sanitizer(value):
            if isinstance(value, str):
                # Extract numeric value
                numeric_match = re.search(r'[-+]?\d*\.?\d+', value)
                if numeric_match:
                    try:
                        return float(numeric_match.group())
                    except ValueError:
                        pass
            return value
        
        # String length sanitizer
        def string_length_sanitizer(value, max_length=1000):
            if isinstance(value, str) and len(value) > max_length:
                return value[:max_length] + "..."
            return value
        
        # Register sanitizers
        self.field_sanitizers['*message*'] = html_sanitizer
        self.field_sanitizers['*description*'] = html_sanitizer
        self.field_sanitizers['*query*'] = sql_sanitizer
        self.field_sanitizers['*text*'] = lambda v: string_length_sanitizer(v, 5000)
        self.field_sanitizers['*name*'] = lambda v: string_length_sanitizer(v, 200)
    
    def _setup_threat_patterns(self) -> Dict[str, List[str]]:
        """Setup threat detection patterns."""
        return {
            'sql_injection': [
                r"'\s*(or|and)\s+'.*'",
                r"union\s+select",
                r"drop\s+table",
                r"insert\s+into",
                r"delete\s+from",
                r"update\s+.*set"
            ],
            'xss_attempts': [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"eval\s*\(",
                r"document\.cookie"
            ],
            'command_injection': [
                r";\s*(rm|del|format|shutdown)",
                r"\|\s*(cat|type|more)",
                r"&&\s*(rm|del|format)",
                r"`.*`",
                r"\$\(.*\)"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"/etc/passwd",
                r"windows/system32"
            ]
        }
    
    def _detect_and_remove_threats(self, data: Dict[str, Any], source: str) -> List[ValidationIssue]:
        """Detect and remove potential security threats."""
        issues = []
        
        def check_value(value, field_path):
            if not isinstance(value, str):
                return
            
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        issue = ValidationIssue(
                            issue_id=f"threat_{threat_type}_{int(time.time())}",
                            rule_id=f"threat_detection_{threat_type}",
                            field_path=field_path,
                            severity="error",
                            message=f"Potential {threat_type} detected",
                            original_value=value,
                            suggested_fix="[REDACTED - SECURITY THREAT]",
                            timestamp=datetime.now()
                        )
                        issues.append(issue)
                        self.suspicious_patterns[threat_type] += 1
                        
                        # Remove or sanitize the threatening content
                        if self.validation_level == ValidationLevel.STRICT:
                            return "[REDACTED - SECURITY THREAT]"
                        else:
                            # Just remove the pattern
                            value = re.sub(pattern, '', value, flags=re.IGNORECASE)
            
            return value
        
        # Recursively check all string values
        def sanitize_dict(d, path=""):
            if isinstance(d, dict):
                for key, value in d.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)):
                        sanitize_dict(value, current_path)
                    else:
                        sanitized = check_value(value, current_path)
                        if sanitized != value:
                            d[key] = sanitized
            elif isinstance(d, list):
                for i, item in enumerate(d):
                    current_path = f"{path}[{i}]"
                    if isinstance(item, (dict, list)):
                        sanitize_dict(item, current_path)
                    else:
                        sanitized = check_value(item, current_path)
                        if sanitized != item:
                            d[i] = sanitized
        
        sanitize_dict(data)
        return issues
    
    def _validate_and_convert_types(self, data: Dict[str, Any], source: str) -> List[ValidationIssue]:
        """Validate and convert data types."""
        issues = []
        
        for rule_id, rule in self.validation_rules.items():
            field_value = self._get_field_value(data, rule.field_path)
            
            if field_value is None:
                if rule.required:
                    issue = ValidationIssue(
                        issue_id=f"missing_required_{rule_id}_{int(time.time())}",
                        rule_id=rule_id,
                        field_path=rule.field_path,
                        severity="error",
                        message=f"Required field missing: {rule.field_path}",
                        original_value=None,
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
                continue
            
            # Type-specific validation
            converted_value, type_issues = self._convert_and_validate_type(
                field_value, rule, source
            )
            issues.extend(type_issues)
            
            # Update the value if converted
            if converted_value != field_value:
                self._set_field_value(data, rule.field_path, converted_value)
        
        return issues
    
    def _apply_field_sanitizers(self, data: Dict[str, Any], source: str) -> List[ValidationIssue]:
        """Apply field-specific sanitizers."""
        issues = []
        
        def apply_sanitizers(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for matching sanitizers
                    for sanitizer_pattern, sanitizer_func in self.field_sanitizers.items():
                        if self._field_matches_pattern(current_path, sanitizer_pattern):
                            try:
                                sanitized_value = sanitizer_func(value)
                                if sanitized_value != value:
                                    obj[key] = sanitized_value
                                    issue = ValidationIssue(
                                        issue_id=f"sanitized_{key}_{int(time.time())}",
                                        rule_id="field_sanitization",
                                        field_path=current_path,
                                        severity="info",
                                        message="Field sanitized",
                                        original_value=value,
                                        suggested_fix=sanitized_value,
                                        timestamp=datetime.now()
                                    )
                                    issues.append(issue)
                                    self.sanitization_stats['issues_fixed'] += 1
                            except Exception as e:
                                issue = ValidationIssue(
                                    issue_id=f"sanitization_error_{key}_{int(time.time())}",
                                    rule_id="field_sanitization",
                                    field_path=current_path,
                                    severity="warning",
                                    message=f"Sanitization failed: {str(e)}",
                                    original_value=value,
                                    timestamp=datetime.now()
                                )
                                issues.append(issue)
                    
                    # Recurse into nested structures
                    if isinstance(value, (dict, list)):
                        apply_sanitizers(value, current_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        apply_sanitizers(item, f"{path}[{i}]")
        
        apply_sanitizers(data)
        return issues
    
    def _validate_business_rules(self, data: Dict[str, Any], source: str) -> List[ValidationIssue]:
        """Validate business logic rules."""
        issues = []
        
        # Custom business rule validations can be added here
        # Example: CPU usage should be reasonable
        cpu_usage = self._get_field_value(data, 'cpu_usage_percent')
        if cpu_usage is not None and isinstance(cpu_usage, (int, float)):
            if cpu_usage > 99.9:
                issue = ValidationIssue(
                    issue_id=f"suspicious_cpu_{int(time.time())}",
                    rule_id="business_logic_cpu",
                    field_path="cpu_usage_percent",
                    severity="warning",
                    message="Suspiciously high CPU usage",
                    original_value=cpu_usage,
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        return issues
    
    def _check_data_consistency(self, data: Dict[str, Any], source: str) -> List[ValidationIssue]:
        """Check data consistency across fields."""
        issues = []
        
        # Example consistency check: timestamps should be recent
        timestamp_fields = ['timestamp', 'created_at', 'updated_at']
        current_time = datetime.now()
        
        for field in timestamp_fields:
            timestamp_value = self._get_field_value(data, field)
            if timestamp_value:
                try:
                    if isinstance(timestamp_value, str):
                        parsed_time = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                    elif isinstance(timestamp_value, datetime):
                        parsed_time = timestamp_value
                    else:
                        continue
                    
                    # Check if timestamp is too far in the future or past
                    time_diff = abs((current_time - parsed_time).total_seconds())
                    if time_diff > 86400 * 30:  # More than 30 days
                        issue = ValidationIssue(
                            issue_id=f"inconsistent_timestamp_{field}_{int(time.time())}",
                            rule_id="data_consistency_timestamp",
                            field_path=field,
                            severity="warning",
                            message=f"Timestamp is {time_diff/86400:.1f} days old",
                            original_value=timestamp_value,
                            timestamp=datetime.now()
                        )
                        issues.append(issue)
                
                except Exception as e:
                    issue = ValidationIssue(
                        issue_id=f"timestamp_parse_error_{field}_{int(time.time())}",
                        rule_id="data_consistency_timestamp",
                        field_path=field,
                        severity="warning",
                        message=f"Could not parse timestamp: {str(e)}",
                        original_value=timestamp_value,
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
        
        return issues
    
    def _final_data_cleanup(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform final data cleanup operations."""
        
        def cleanup_dict(obj):
            if isinstance(obj, dict):
                # Remove null/empty values if configured
                cleaned = {}
                for key, value in obj.items():
                    if value is not None and value != "":
                        if isinstance(value, (dict, list)):
                            cleaned_value = cleanup_dict(value)
                            if cleaned_value:  # Only add if not empty
                                cleaned[key] = cleaned_value
                        else:
                            cleaned[key] = value
                return cleaned
            elif isinstance(obj, list):
                return [cleanup_dict(item) for item in obj if item is not None]
            else:
                return obj
        
        return cleanup_dict(data)
    
    def _convert_and_validate_type(self, value: Any, rule: ValidationRule, source: str) -> Tuple[Any, List[ValidationIssue]]:
        """Convert and validate a value according to its rule."""
        issues = []
        converted_value = value
        
        try:
            if rule.data_type == DataType.NUMERIC:
                if not isinstance(value, (int, float)):
                    converted_value = float(str(value))
                
                # Range validation
                if rule.min_value is not None and converted_value < rule.min_value:
                    issue = ValidationIssue(
                        issue_id=f"numeric_range_min_{int(time.time())}",
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        severity="error",
                        message=f"Value {converted_value} below minimum {rule.min_value}",
                        original_value=value,
                        suggested_fix=rule.min_value,
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
                    converted_value = rule.min_value
                
                if rule.max_value is not None and converted_value > rule.max_value:
                    issue = ValidationIssue(
                        issue_id=f"numeric_range_max_{int(time.time())}",
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        severity="error",
                        message=f"Value {converted_value} above maximum {rule.max_value}",
                        original_value=value,
                        suggested_fix=rule.max_value,
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
                    converted_value = rule.max_value
            
            elif rule.data_type == DataType.STRING:
                converted_value = str(value)
                
                # Length validation
                if rule.min_length is not None and len(converted_value) < rule.min_length:
                    issue = ValidationIssue(
                        issue_id=f"string_length_min_{int(time.time())}",
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        severity="error",
                        message=f"String too short: {len(converted_value)} < {rule.min_length}",
                        original_value=value,
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
                
                if rule.max_length is not None and len(converted_value) > rule.max_length:
                    issue = ValidationIssue(
                        issue_id=f"string_length_max_{int(time.time())}",
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        severity="warning",
                        message=f"String truncated: {len(converted_value)} > {rule.max_length}",
                        original_value=value,
                        suggested_fix=converted_value[:rule.max_length],
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
                    converted_value = converted_value[:rule.max_length]
                
                # Pattern validation
                if rule.pattern and not re.match(rule.pattern, converted_value):
                    issue = ValidationIssue(
                        issue_id=f"pattern_mismatch_{int(time.time())}",
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        severity="error",
                        message=f"Value doesn't match pattern: {rule.pattern}",
                        original_value=value,
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
            
            elif rule.data_type == DataType.EMAIL:
                converted_value = str(value)
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, converted_value):
                    issue = ValidationIssue(
                        issue_id=f"invalid_email_{int(time.time())}",
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        severity="error",
                        message="Invalid email format",
                        original_value=value,
                        timestamp=datetime.now()
                    )
                    issues.append(issue)
            
            elif rule.data_type == DataType.DATE:
                if isinstance(value, str):
                    try:
                        converted_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        converted_value = datetime.now()
                        issue = ValidationIssue(
                            issue_id=f"invalid_date_{int(time.time())}",
                            rule_id=rule.rule_id,
                            field_path=rule.field_path,
                            severity="warning",
                            message="Invalid date format, using current time",
                            original_value=value,
                            suggested_fix=converted_value.isoformat(),
                            timestamp=datetime.now()
                        )
                        issues.append(issue)
            
            # Custom validator
            if rule.custom_validator and not rule.custom_validator(converted_value):
                issue = ValidationIssue(
                    issue_id=f"custom_validation_{int(time.time())}",
                    rule_id=rule.rule_id,
                    field_path=rule.field_path,
                    severity="error",
                    message="Custom validation failed",
                    original_value=value,
                    timestamp=datetime.now()
                )
                issues.append(issue)
        
        except Exception as e:
            issue = ValidationIssue(
                issue_id=f"conversion_error_{int(time.time())}",
                rule_id=rule.rule_id,
                field_path=rule.field_path,
                severity="error",
                message=f"Type conversion failed: {str(e)}",
                original_value=value,
                timestamp=datetime.now()
            )
            issues.append(issue)
        
        return converted_value, issues
    
    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get field value using dot notation path."""
        if '.' not in field_path and '[' not in field_path:
            return data.get(field_path)
        
        # Handle complex paths (nested objects, arrays)
        current = data
        parts = field_path.replace('[', '.').replace(']', '').split('.')
        
        for part in parts:
            if not part:
                continue
            try:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    return None
            except (KeyError, IndexError, TypeError):
                return None
        
        return current
    
    def _set_field_value(self, data: Dict[str, Any], field_path: str, value: Any):
        """Set field value using dot notation path."""
        if '.' not in field_path and '[' not in field_path:
            data[field_path] = value
            return
        
        # Handle complex paths
        current = data
        parts = field_path.replace('[', '.').replace(']', '').split('.')
        
        for i, part in enumerate(parts[:-1]):
            if not part:
                continue
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set the final value
        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = value
    
    def _field_matches_pattern(self, field_path: str, pattern: str) -> bool:
        """Check if field path matches a pattern (supports wildcards)."""
        if pattern == field_path:
            return True
        
        if '*' in pattern:
            # Convert pattern to regex
            regex_pattern = pattern.replace('*', '.*')
            return bool(re.match(regex_pattern, field_path, re.IGNORECASE))
        
        return False
    
    def _background_processing_loop(self):
        """Background processing loop for queued data."""
        while self.sanitizer_active:
            try:
                if self.pending_data:
                    # Process a batch of data
                    batch_size = min(10, len(self.pending_data))
                    for _ in range(batch_size):
                        if not self.pending_data:
                            break
                        
                        item = self.pending_data.popleft()
                        sanitized_data, issues = self.sanitize_and_validate(
                            item['data'], item['source']
                        )
                        
                        # Store results for later retrieval if needed
                        # This could be enhanced to store results somewhere
                
                time.sleep(1)  # Process every second
            except Exception as e:
                logger.error(f"Background processing error: {e}")
    
    def get_validation_issues(self, severity: Optional[str] = None, 
                            limit: int = 100) -> List[ValidationIssue]:
        """Get recent validation issues, optionally filtered by severity."""
        issues = list(self.validation_issues)
        
        if severity:
            issues = [i for i in issues if i.severity == severity]
        
        # Sort by timestamp (most recent first)
        issues.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)
        
        return issues[:limit]
    
    def shutdown(self):
        """Shutdown sanitization engine."""
        self.stop_background_processing()
        logger.info("Analytics Data Sanitizer shutdown")