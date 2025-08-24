"""
Analytics Data Normalizer
=========================

Comprehensive data normalization and standardization system for ensuring
consistent data formats, units, and structures across all analytics components.

Author: TestMaster Team
"""

import re
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import copy

logger = logging.getLogger(__name__)

class DataType(Enum):
    NUMERIC = "numeric"
    TIMESTAMP = "timestamp"
    PERCENTAGE = "percentage"
    BYTES = "bytes"
    DURATION = "duration"
    BOOLEAN = "boolean"
    STRING = "string"
    ENUM = "enum"

class NormalizationRule(Enum):
    UNIT_CONVERSION = "unit_conversion"
    FORMAT_STANDARDIZATION = "format_standardization"
    RANGE_NORMALIZATION = "range_normalization"
    TYPE_CONVERSION = "type_conversion"
    FIELD_MAPPING = "field_mapping"
    VALIDATION = "validation"

@dataclass
class FieldSchema:
    """Schema definition for a data field."""
    name: str
    data_type: DataType
    required: bool = False
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    unit: Optional[str] = None
    format_pattern: Optional[str] = None
    description: Optional[str] = None

class AnalyticsDataNormalizer:
    """
    Comprehensive data normalization and standardization system.
    """
    
    def __init__(self):
        """Initialize the data normalizer."""
        # Schema registry
        self.schemas = {}
        self.field_mappings = {}
        
        # Normalization rules
        self.normalization_rules = []
        self.custom_validators = {}
        
        # Unit conversion tables
        self.unit_conversions = {
            'bytes': {
                'b': 1,
                'byte': 1,
                'bytes': 1,
                'kb': 1024,
                'kilobyte': 1024,
                'kilobytes': 1024,
                'mb': 1024 * 1024,
                'megabyte': 1024 * 1024,
                'megabytes': 1024 * 1024,
                'gb': 1024 * 1024 * 1024,
                'gigabyte': 1024 * 1024 * 1024,
                'gigabytes': 1024 * 1024 * 1024,
                'tb': 1024 * 1024 * 1024 * 1024,
                'terabyte': 1024 * 1024 * 1024 * 1024,
                'terabytes': 1024 * 1024 * 1024 * 1024
            },
            'duration': {
                'ms': 0.001,
                'millisecond': 0.001,
                'milliseconds': 0.001,
                's': 1,
                'sec': 1,
                'second': 1,
                'seconds': 1,
                'm': 60,
                'min': 60,
                'minute': 60,
                'minutes': 60,
                'h': 3600,
                'hr': 3600,
                'hour': 3600,
                'hours': 3600,
                'd': 86400,
                'day': 86400,
                'days': 86400
            }
        }
        
        # Timestamp format patterns
        self.timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.?\d*Z?',  # ISO 8601
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Standard datetime
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # US format
            r'\d{10}',  # Unix timestamp
            r'\d{13}'   # Unix timestamp (milliseconds)
        ]
        
        # Statistics tracking
        self.normalization_stats = {
            'total_normalizations': 0,
            'successful_normalizations': 0,
            'failed_normalizations': 0,
            'validation_errors': 0,
            'unit_conversions': 0,
            'type_conversions': 0,
            'field_mappings': 0,
            'start_time': datetime.now()
        }
        
        # Setup default schemas
        self._setup_default_schemas()
        
        logger.info("Analytics Data Normalizer initialized")
    
    def register_schema(self, schema_name: str, fields: List[FieldSchema]):
        """Register a data schema."""
        self.schemas[schema_name] = {field.name: field for field in fields}
        logger.info(f"Registered schema: {schema_name} with {len(fields)} fields")
    
    def add_field_mapping(self, source_field: str, target_field: str, 
                         transformation: Optional[Callable] = None):
        """Add a field mapping rule."""
        self.field_mappings[source_field] = {
            'target': target_field,
            'transformation': transformation
        }
    
    def add_custom_validator(self, field_name: str, validator: Callable[[Any], bool]):
        """Add a custom validator for a field."""
        self.custom_validators[field_name] = validator
    
    def normalize(self, data: Dict[str, Any], schema_name: str = None) -> Dict[str, Any]:
        """
        Normalize data according to schema and rules.
        
        Args:
            data: Data to normalize
            schema_name: Schema to apply (optional)
        
        Returns:
            Normalized data
        """
        start_time = datetime.now()
        self.normalization_stats['total_normalizations'] += 1
        
        try:
            # Deep copy to avoid modifying original data
            normalized = copy.deepcopy(data)
            
            # Apply field mappings
            normalized = self._apply_field_mappings(normalized)
            
            # Normalize timestamps
            normalized = self._normalize_timestamps(normalized)
            
            # Normalize units
            normalized = self._normalize_units(normalized)
            
            # Normalize data types
            normalized = self._normalize_data_types(normalized)
            
            # Normalize numeric ranges
            normalized = self._normalize_numeric_ranges(normalized)
            
            # Apply schema validation if provided
            if schema_name and schema_name in self.schemas:
                normalized = self._apply_schema(normalized, schema_name)
            
            # Add normalization metadata
            normalized['_normalization'] = {
                'normalized_at': datetime.now().isoformat(),
                'schema_applied': schema_name,
                'normalization_version': '1.0',
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
            
            self.normalization_stats['successful_normalizations'] += 1
            return normalized
            
        except Exception as e:
            self.normalization_stats['failed_normalizations'] += 1
            logger.error(f"Normalization failed: {e}")
            
            # Return original data with error information
            error_data = copy.deepcopy(data)
            error_data['_normalization'] = {
                'error': str(e),
                'normalized_at': datetime.now().isoformat(),
                'success': False
            }
            return error_data
    
    def validate(self, data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema_name: Schema name
        
        Returns:
            Validation result
        """
        if schema_name not in self.schemas:
            return {'valid': False, 'error': f'Schema {schema_name} not found'}
        
        schema = self.schemas[schema_name]
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'field_validations': {}
        }
        
        # Check required fields
        for field_name, field_schema in schema.items():
            if field_schema.required and field_name not in data:
                validation_result['errors'].append(f'Required field missing: {field_name}')
                validation_result['valid'] = False
            
            if field_name in data:
                field_validation = self._validate_field(data[field_name], field_schema)
                validation_result['field_validations'][field_name] = field_validation
                
                if not field_validation['valid']:
                    validation_result['errors'].extend(field_validation['errors'])
                    validation_result['valid'] = False
                
                if field_validation['warnings']:
                    validation_result['warnings'].extend(field_validation['warnings'])
        
        # Check for unexpected fields
        for field_name in data:
            if field_name not in schema and not field_name.startswith('_'):
                validation_result['warnings'].append(f'Unexpected field: {field_name}')
        
        if not validation_result['valid']:
            self.normalization_stats['validation_errors'] += 1
        
        return validation_result
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        uptime = (datetime.now() - self.normalization_stats['start_time']).total_seconds()
        success_rate = (self.normalization_stats['successful_normalizations'] / 
                       max(self.normalization_stats['total_normalizations'], 1)) * 100
        
        return {
            'total_normalizations': self.normalization_stats['total_normalizations'],
            'successful_normalizations': self.normalization_stats['successful_normalizations'],
            'failed_normalizations': self.normalization_stats['failed_normalizations'],
            'success_rate': success_rate,
            'validation_errors': self.normalization_stats['validation_errors'],
            'unit_conversions': self.normalization_stats['unit_conversions'],
            'type_conversions': self.normalization_stats['type_conversions'],
            'field_mappings': self.normalization_stats['field_mappings'],
            'registered_schemas': len(self.schemas),
            'field_mappings_count': len(self.field_mappings),
            'custom_validators': len(self.custom_validators),
            'uptime_seconds': uptime,
            'normalizations_per_second': self.normalization_stats['total_normalizations'] / max(uptime, 1)
        }
    
    def _setup_default_schemas(self):
        """Set up default schemas for common data types."""
        # System metrics schema
        system_metrics_fields = [
            FieldSchema('cpu_usage_percent', DataType.PERCENTAGE, True, min_value=0, max_value=100),
            FieldSchema('memory_usage_percent', DataType.PERCENTAGE, True, min_value=0, max_value=100),
            FieldSchema('memory_used_mb', DataType.NUMERIC, unit='mb', min_value=0),
            FieldSchema('memory_total_mb', DataType.NUMERIC, unit='mb', min_value=0),
            FieldSchema('disk_usage_percent', DataType.PERCENTAGE, min_value=0, max_value=100),
            FieldSchema('disk_used_gb', DataType.NUMERIC, unit='gb', min_value=0),
            FieldSchema('disk_total_gb', DataType.NUMERIC, unit='gb', min_value=0),
            FieldSchema('timestamp', DataType.TIMESTAMP, True)
        ]
        
        # Test analytics schema
        test_analytics_fields = [
            FieldSchema('total_tests', DataType.NUMERIC, True, min_value=0),
            FieldSchema('passed_tests', DataType.NUMERIC, True, min_value=0),
            FieldSchema('failed_tests', DataType.NUMERIC, True, min_value=0),
            FieldSchema('skipped_tests', DataType.NUMERIC, min_value=0),
            FieldSchema('coverage_percent', DataType.PERCENTAGE, min_value=0, max_value=100),
            FieldSchema('quality_score', DataType.NUMERIC, min_value=0, max_value=100),
            FieldSchema('pass_rate', DataType.PERCENTAGE, min_value=0, max_value=100),
            FieldSchema('execution_time_seconds', DataType.DURATION, unit='seconds', min_value=0),
            FieldSchema('timestamp', DataType.TIMESTAMP, True)
        ]
        
        # Performance metrics schema
        performance_metrics_fields = [
            FieldSchema('response_time_ms', DataType.DURATION, unit='ms', min_value=0),
            FieldSchema('throughput_per_second', DataType.NUMERIC, min_value=0),
            FieldSchema('error_rate_percent', DataType.PERCENTAGE, min_value=0, max_value=100),
            FieldSchema('latency_p50_ms', DataType.DURATION, unit='ms', min_value=0),
            FieldSchema('latency_p95_ms', DataType.DURATION, unit='ms', min_value=0),
            FieldSchema('latency_p99_ms', DataType.DURATION, unit='ms', min_value=0),
            FieldSchema('timestamp', DataType.TIMESTAMP, True)
        ]
        
        self.register_schema('system_metrics', system_metrics_fields)
        self.register_schema('test_analytics', test_analytics_fields)
        self.register_schema('performance_metrics', performance_metrics_fields)
        
        # Setup common field mappings
        self._setup_default_field_mappings()
    
    def _setup_default_field_mappings(self):
        """Set up default field mappings."""
        # CPU usage mappings
        self.add_field_mapping('cpu_percent', 'cpu_usage_percent')
        self.add_field_mapping('cpu_usage', 'cpu_usage_percent')
        self.add_field_mapping('cpu', 'cpu_usage_percent')
        
        # Memory usage mappings
        self.add_field_mapping('memory_percent', 'memory_usage_percent')
        self.add_field_mapping('mem_usage', 'memory_usage_percent')
        self.add_field_mapping('memory_usage', 'memory_usage_percent')
        self.add_field_mapping('memory_used', 'memory_used_mb')
        self.add_field_mapping('mem_used', 'memory_used_mb')
        
        # Test result mappings
        self.add_field_mapping('tests_total', 'total_tests')
        self.add_field_mapping('num_tests', 'total_tests')
        self.add_field_mapping('test_count', 'total_tests')
        self.add_field_mapping('tests_passed', 'passed_tests')
        self.add_field_mapping('passed', 'passed_tests')
        self.add_field_mapping('tests_failed', 'failed_tests')
        self.add_field_mapping('failed', 'failed_tests')
        
        # Time mappings
        self.add_field_mapping('created_at', 'timestamp')
        self.add_field_mapping('updated_at', 'timestamp')
        self.add_field_mapping('time', 'timestamp')
        self.add_field_mapping('datetime', 'timestamp')
    
    def _apply_field_mappings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field mappings to data."""
        mapped_data = {}
        
        for key, value in data.items():
            if key in self.field_mappings:
                mapping = self.field_mappings[key]
                target_field = mapping['target']
                transformation = mapping['transformation']
                
                # Apply transformation if provided
                if transformation:
                    try:
                        value = transformation(value)
                    except Exception as e:
                        logger.warning(f"Field transformation failed for {key}: {e}")
                
                mapped_data[target_field] = value
                self.normalization_stats['field_mappings'] += 1
            else:
                mapped_data[key] = value
        
        # Handle nested dictionaries
        for key, value in mapped_data.items():
            if isinstance(value, dict):
                mapped_data[key] = self._apply_field_mappings(value)
        
        return mapped_data
    
    def _normalize_timestamps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize timestamp fields to ISO 8601 format."""
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'last_access', 'datetime']
        
        def normalize_timestamp_value(value):
            if value is None:
                return value
            
            # If already a datetime object
            if isinstance(value, datetime):
                return value.isoformat()
            
            # If string, try to parse
            if isinstance(value, str):
                # Try ISO format first
                try:
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return dt.isoformat()
                except ValueError:
                    pass
                
                # Try other formats
                for pattern in self.timestamp_patterns:
                    if re.match(pattern, value):
                        try:
                            if len(value) == 10:  # Unix timestamp
                                dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
                            elif len(value) == 13:  # Unix timestamp (ms)
                                dt = datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
                            else:
                                # Try standard parsing
                                dt = datetime.fromisoformat(value.replace('/', '-'))
                            return dt.isoformat()
                        except ValueError:
                            continue
            
            # If numeric, assume Unix timestamp
            if isinstance(value, (int, float)):
                try:
                    if value > 1e12:  # Milliseconds
                        dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
                    else:  # Seconds
                        dt = datetime.fromtimestamp(value, tz=timezone.utc)
                    return dt.isoformat()
                except (ValueError, OSError):
                    pass
            
            return value  # Return original if can't normalize
        
        def normalize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in timestamp_fields:
                        obj[key] = normalize_timestamp_value(value)
                    elif isinstance(value, (dict, list)):
                        normalize_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    normalize_recursive(item)
        
        normalize_recursive(data)
        return data
    
    def _normalize_units(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize units to standard formats."""
        def normalize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Memory fields - normalize to MB
                    if 'memory' in key.lower() and ('mb' in key.lower() or 'gb' in key.lower() or 'kb' in key.lower()):
                        if isinstance(value, (int, float)):
                            if 'kb' in key.lower():
                                obj[key] = value / 1024  # Convert KB to MB
                                self.normalization_stats['unit_conversions'] += 1
                            elif 'gb' in key.lower():
                                obj[key] = value * 1024  # Convert GB to MB
                                self.normalization_stats['unit_conversions'] += 1
                    
                    # Duration fields - normalize to seconds
                    elif ('time' in key.lower() or 'duration' in key.lower()) and isinstance(value, (int, float)):
                        if 'ms' in key.lower() or 'millisecond' in key.lower():
                            obj[key.replace('_ms', '_seconds').replace('_milliseconds', '_seconds')] = value / 1000
                            if key != key.replace('_ms', '_seconds').replace('_milliseconds', '_seconds'):
                                del obj[key]
                            self.normalization_stats['unit_conversions'] += 1
                        elif 'minutes' in key.lower() or '_min' in key.lower():
                            obj[key.replace('_minutes', '_seconds').replace('_min', '_seconds')] = value * 60
                            if key != key.replace('_minutes', '_seconds').replace('_min', '_seconds'):
                                del obj[key]
                            self.normalization_stats['unit_conversions'] += 1
                    
                    # Percentage fields - ensure 0-100 range
                    elif 'percent' in key.lower() and isinstance(value, (int, float)):
                        if value > 1 and value <= 100:
                            pass  # Already in percentage
                        elif 0 <= value <= 1:
                            obj[key] = value * 100  # Convert from ratio to percentage
                            self.normalization_stats['unit_conversions'] += 1
                    
                    elif isinstance(value, (dict, list)):
                        normalize_recursive(value)
            
            elif isinstance(obj, list):
                for item in obj:
                    normalize_recursive(item)
        
        normalize_recursive(data)
        return data
    
    def _normalize_data_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data types to expected formats."""
        def normalize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Boolean fields
                    if key.lower() in ['active', 'enabled', 'success', 'healthy', 'valid', 'compressed']:
                        if isinstance(value, str):
                            if value.lower() in ['true', '1', 'yes', 'on', 'enabled']:
                                obj[key] = True
                                self.normalization_stats['type_conversions'] += 1
                            elif value.lower() in ['false', '0', 'no', 'off', 'disabled']:
                                obj[key] = False
                                self.normalization_stats['type_conversions'] += 1
                        elif isinstance(value, (int, float)):
                            obj[key] = bool(value)
                            self.normalization_stats['type_conversions'] += 1
                    
                    # Numeric fields that might be strings
                    elif key.lower() in ['count', 'total', 'size', 'duration', 'percent', 'score', 'rate']:
                        if isinstance(value, str) and value.isdigit():
                            try:
                                obj[key] = int(value) if '.' not in value else float(value)
                                self.normalization_stats['type_conversions'] += 1
                            except ValueError:
                                pass
                    
                    elif isinstance(value, (dict, list)):
                        normalize_recursive(value)
            
            elif isinstance(obj, list):
                for item in obj:
                    normalize_recursive(item)
        
        normalize_recursive(data)
        return data
    
    def _normalize_numeric_ranges(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize numeric values to appropriate ranges."""
        def normalize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (int, float)):
                        # Percentage fields should be 0-100
                        if 'percent' in key.lower() and (value < 0 or value > 100):
                            obj[key] = max(0, min(100, value))
                        
                        # Usage fields should not be negative
                        elif 'usage' in key.lower() and value < 0:
                            obj[key] = 0
                        
                        # Count fields should not be negative
                        elif 'count' in key.lower() and value < 0:
                            obj[key] = 0
                        
                        # Size fields should not be negative
                        elif 'size' in key.lower() and value < 0:
                            obj[key] = 0
                    
                    elif isinstance(value, (dict, list)):
                        normalize_recursive(value)
            
            elif isinstance(obj, list):
                for item in obj:
                    normalize_recursive(item)
        
        normalize_recursive(data)
        return data
    
    def _apply_schema(self, data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Apply schema validation and default values."""
        schema = self.schemas[schema_name]
        normalized = copy.deepcopy(data)
        
        # Apply default values for missing fields
        for field_name, field_schema in schema.items():
            if field_name not in normalized and field_schema.default_value is not None:
                normalized[field_name] = field_schema.default_value
        
        # Validate and potentially fix field values
        for field_name, value in normalized.items():
            if field_name in schema:
                field_schema = schema[field_name]
                normalized[field_name] = self._normalize_field_value(value, field_schema)
        
        return normalized
    
    def _normalize_field_value(self, value: Any, field_schema: FieldSchema) -> Any:
        """Normalize a single field value according to its schema."""
        if value is None:
            return field_schema.default_value
        
        # Type-specific normalization
        if field_schema.data_type == DataType.PERCENTAGE:
            if isinstance(value, (int, float)):
                return max(0, min(100, float(value)))
        
        elif field_schema.data_type == DataType.NUMERIC:
            if isinstance(value, (int, float)):
                if field_schema.min_value is not None:
                    value = max(field_schema.min_value, value)
                if field_schema.max_value is not None:
                    value = min(field_schema.max_value, value)
                return value
            elif isinstance(value, str) and value.replace('.', '').isdigit():
                return float(value) if '.' in value else int(value)
        
        elif field_schema.data_type == DataType.BOOLEAN:
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            else:
                return bool(value)
        
        elif field_schema.data_type == DataType.TIMESTAMP:
            # Already handled in _normalize_timestamps
            return value
        
        return value
    
    def _validate_field(self, value: Any, field_schema: FieldSchema) -> Dict[str, Any]:
        """Validate a single field against its schema."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        # Type validation
        if field_schema.data_type == DataType.NUMERIC and not isinstance(value, (int, float)):
            validation['errors'].append(f'Expected numeric value, got {type(value).__name__}')
            validation['valid'] = False
        
        elif field_schema.data_type == DataType.BOOLEAN and not isinstance(value, bool):
            validation['errors'].append(f'Expected boolean value, got {type(value).__name__}')
            validation['valid'] = False
        
        # Range validation
        if isinstance(value, (int, float)):
            if field_schema.min_value is not None and value < field_schema.min_value:
                validation['errors'].append(f'Value {value} below minimum {field_schema.min_value}')
                validation['valid'] = False
            
            if field_schema.max_value is not None and value > field_schema.max_value:
                validation['errors'].append(f'Value {value} above maximum {field_schema.max_value}')
                validation['valid'] = False
        
        # Allowed values validation
        if field_schema.allowed_values and value not in field_schema.allowed_values:
            validation['errors'].append(f'Value {value} not in allowed values: {field_schema.allowed_values}')
            validation['valid'] = False
        
        # Custom validation
        if field_schema.name in self.custom_validators:
            try:
                if not self.custom_validators[field_schema.name](value):
                    validation['errors'].append(f'Custom validation failed for field {field_schema.name}')
                    validation['valid'] = False
            except Exception as e:
                validation['warnings'].append(f'Custom validator error: {e}')
        
        return validation