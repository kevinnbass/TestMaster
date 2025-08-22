"""
TestMaster Intelligence Hub Request Validators
==============================================

Request validation for API endpoints.
Ensures data integrity and proper formatting.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re


class ValidationError(Exception):
    """Custom validation error with detailed context.
    
    Enhanced exception class for API request validation errors.
    Provides structured error information including field-specific
    details and error codes for programmatic handling.
    
    Attributes:
        message: Human-readable error description
        field: Name of the field that failed validation (if applicable)
        code: Machine-readable error code for programmatic handling
        
    Error Codes:
        - MISSING_REQUIRED_FIELDS: Required field not provided
        - INVALID_TYPE: Value type doesn't match expected
        - VALUE_TOO_LOW: Numeric value below minimum
        - VALUE_TOO_HIGH: Numeric value above maximum
        - INVALID_ENUM_VALUE: Value not in allowed enumeration
        - LIST_TOO_SHORT: List has fewer items than required
        - LIST_TOO_LONG: List has more items than allowed
        - INVALID_URL: URL format validation failed
        - INVALID_EMAIL: Email format validation failed
        
    Example:
        >>> raise ValidationError(
        ...     "Field 'age' must be >= 0",
        ...     field="age",
        ...     code="VALUE_TOO_LOW"
        ... )
        >>> # Can be caught and handled programmatically
        >>> try:
        ...     validate_user(data)
        >>> except ValidationError as e:
        ...     if e.code == "MISSING_REQUIRED_FIELDS":
        ...         return {"error": "Required fields missing"}
    """
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        """Initialize validation error with context.
        
        Args:
            message: Human-readable error description
            field: Name of the field that failed validation
            code: Machine-readable error code for programmatic handling
        """
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)


class BaseValidator:
    """Base validator with common validation methods.
    
    Comprehensive validation framework providing reusable validation
    methods for API request data. Supports type checking, range validation,
    enumeration validation, and format validation with detailed error reporting.
    
    Key Validation Types:
    - **Required Fields**: Ensures mandatory fields are present
    - **Type Validation**: Verifies data types match expectations
    - **Range Validation**: Checks numeric values within bounds
    - **Enumeration Validation**: Validates against allowed values
    - **List Validation**: Checks list length constraints
    - **Format Validation**: URL, email, and other format validation
    
    Usage Pattern:
        All methods are static and can be used independently or chained
        for comprehensive validation. Each method raises ValidationError
        with detailed context when validation fails.
        
    Example:
        >>> data = {"name": "John", "age": 25, "role": "user"}
        >>> BaseValidator.validate_required(data, ["name", "age"])
        >>> BaseValidator.validate_type(data["age"], int, "age")
        >>> BaseValidator.validate_range(data["age"], min_val=0, max_val=120, field_name="age")
        >>> BaseValidator.validate_enum(data["role"], ["user", "admin"], "role")
        >>> # If any validation fails, ValidationError is raised
    """
    
    @staticmethod
    def validate_required(data: Dict[str, Any], required_fields: List[str]):
        """Validate that required fields are present and not None.
        
        Checks that all specified required fields exist in the data
        dictionary and have non-None values. Collects all missing
        fields and reports them together for better user experience.
        
        Args:
            data: Dictionary to validate
            required_fields: List of field names that must be present
            
        Raises:
            ValidationError: If any required fields are missing,
                           with code "MISSING_REQUIRED_FIELDS"
                           
        Example:
            >>> data = {"name": "John", "email": None}
            >>> BaseValidator.validate_required(data, ["name", "email", "age"])
            ValidationError: Missing required fields: email, age
        """
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}",
                code="MISSING_REQUIRED_FIELDS"
            )
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, field_name: str):
        """Validate that value is of expected type."""
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Field '{field_name}' must be of type {expected_type.__name__}",
                field=field_name,
                code="INVALID_TYPE"
            )
    
    @staticmethod
    def validate_range(value: float, min_val: Optional[float] = None, 
                      max_val: Optional[float] = None, field_name: str = "value"):
        """Validate that value is within range."""
        if min_val is not None and value < min_val:
            raise ValidationError(
                f"Field '{field_name}' must be >= {min_val}",
                field=field_name,
                code="VALUE_TOO_LOW"
            )
        
        if max_val is not None and value > max_val:
            raise ValidationError(
                f"Field '{field_name}' must be <= {max_val}",
                field=field_name,
                code="VALUE_TOO_HIGH"
            )
    
    @staticmethod
    def validate_enum(value: str, allowed_values: List[str], field_name: str):
        """Validate that value is one of allowed values."""
        if value not in allowed_values:
            raise ValidationError(
                f"Field '{field_name}' must be one of: {', '.join(allowed_values)}",
                field=field_name,
                code="INVALID_ENUM_VALUE"
            )
    
    @staticmethod
    def validate_list_length(lst: List, min_length: Optional[int] = None,
                           max_length: Optional[int] = None, field_name: str = "list"):
        """Validate list length."""
        if min_length is not None and len(lst) < min_length:
            raise ValidationError(
                f"Field '{field_name}' must have at least {min_length} items",
                field=field_name,
                code="LIST_TOO_SHORT"
            )
        
        if max_length is not None and len(lst) > max_length:
            raise ValidationError(
                f"Field '{field_name}' must have at most {max_length} items",
                field=field_name,
                code="LIST_TOO_LONG"
            )
    
    @staticmethod
    def validate_url(url: str, field_name: str = "url"):
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        
        if not url_pattern.match(url):
            raise ValidationError(
                f"Field '{field_name}' must be a valid URL",
                field=field_name,
                code="INVALID_URL"
            )


class AnalysisValidator(BaseValidator):
    """Validator for analysis requests."""
    
    @staticmethod
    def validate_analysis_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis request data."""
        # Check analysis type
        if 'analysis_type' in data:
            BaseValidator.validate_enum(
                data['analysis_type'],
                ['comprehensive', 'statistical', 'ml', 'testing', 'integration', 
                 'predictive', 'anomaly'],
                'analysis_type'
            )
        
        # Validate data field if present
        if 'data' in data:
            BaseValidator.validate_type(data.get('data'), dict, 'data')
        
        # Validate options if present
        if 'options' in data:
            BaseValidator.validate_type(data.get('options'), dict, 'options')
        
        return data
    
    @staticmethod
    def validate_metrics(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate metrics data."""
        if not isinstance(metrics, list):
            raise ValidationError("Metrics must be a list", code="INVALID_METRICS_FORMAT")
        
        for i, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                raise ValidationError(
                    f"Metric at index {i} must be a dictionary",
                    code="INVALID_METRIC_FORMAT"
                )
            
            # Validate metric has required fields
            if 'value' not in metric:
                raise ValidationError(
                    f"Metric at index {i} missing 'value' field",
                    code="MISSING_METRIC_VALUE"
                )
        
        return metrics


class TestingValidator(BaseValidator):
    """Validator for testing requests."""
    
    @staticmethod
    def validate_coverage_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate coverage analysis request."""
        # Validate test results
        if 'test_results' in data:
            TestingValidator.validate_test_results(data['test_results'])
        
        # Validate confidence level
        if 'confidence_level' in data:
            BaseValidator.validate_range(
                data['confidence_level'], 0.0, 1.0, 'confidence_level'
            )
        
        return data
    
    @staticmethod
    def validate_test_results(test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate test results data."""
        if not isinstance(test_results, list):
            raise ValidationError(
                "Test results must be a list",
                code="INVALID_TEST_RESULTS_FORMAT"
            )
        
        for i, result in enumerate(test_results):
            if not isinstance(result, dict):
                raise ValidationError(
                    f"Test result at index {i} must be a dictionary",
                    code="INVALID_TEST_RESULT_FORMAT"
                )
            
            # Validate required fields
            required_fields = ['test_id', 'status']
            for field in required_fields:
                if field not in result:
                    raise ValidationError(
                        f"Test result at index {i} missing '{field}' field",
                        code=f"MISSING_{field.upper()}"
                    )
            
            # Validate status
            BaseValidator.validate_enum(
                result['status'],
                ['passed', 'failed', 'skipped', 'error', 'running'],
                f"test_results[{i}].status"
            )
        
        return test_results
    
    @staticmethod
    def validate_optimization_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test optimization request."""
        # Validate strategy
        if 'strategy' in data:
            BaseValidator.validate_enum(
                data['strategy'],
                ['comprehensive', 'latency', 'throughput', 'reliability', 'balanced'],
                'strategy'
            )
        
        return data


class IntegrationValidator(BaseValidator):
    """Validator for integration requests."""
    
    @staticmethod
    def validate_system_analysis_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system analysis request."""
        # Validate systems list
        if 'systems' in data:
            BaseValidator.validate_type(data['systems'], list, 'systems')
            BaseValidator.validate_list_length(
                data['systems'], min_length=1, field_name='systems'
            )
        
        # Validate time window
        if 'time_window_hours' in data:
            BaseValidator.validate_range(
                data['time_window_hours'], 1, 720, 'time_window_hours'
            )
        
        return data
    
    @staticmethod
    def validate_endpoint_registration(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate endpoint registration data."""
        # Required fields
        BaseValidator.validate_required(
            data, ['endpoint_id', 'name', 'url', 'integration_type']
        )
        
        # Validate URL
        BaseValidator.validate_url(data['url'], 'url')
        
        # Validate integration type
        BaseValidator.validate_enum(
            data['integration_type'],
            ['api_gateway', 'database_sync', 'event_stream', 'file_transfer',
             'real_time_sync', 'batch_processing', 'authentication', 'monitoring'],
            'integration_type'
        )
        
        return data
    
    @staticmethod
    def validate_event_publish(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate event publish request."""
        # Required fields
        BaseValidator.validate_required(
            data, ['source_system', 'event_type']
        )
        
        # Validate payload if present
        if 'payload' in data:
            BaseValidator.validate_type(data['payload'], dict, 'payload')
        
        return data


class BatchValidator(BaseValidator):
    """Validator for batch operations."""
    
    @staticmethod
    def validate_batch_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate batch analysis request."""
        # Validate analyses list
        BaseValidator.validate_required(data, ['analyses'])
        BaseValidator.validate_type(data['analyses'], list, 'analyses')
        BaseValidator.validate_list_length(
            data['analyses'], 
            min_length=1, 
            max_length=100,  # Limit batch size
            field_name='analyses'
        )
        
        # Validate each analysis in batch
        for i, analysis in enumerate(data['analyses']):
            if not isinstance(analysis, dict):
                raise ValidationError(
                    f"Analysis at index {i} must be a dictionary",
                    code="INVALID_BATCH_ITEM"
                )
            
            if 'type' not in analysis:
                raise ValidationError(
                    f"Analysis at index {i} missing 'type' field",
                    code="MISSING_ANALYSIS_TYPE"
                )
            
            BaseValidator.validate_enum(
                analysis['type'],
                ['analytics', 'testing', 'integration'],
                f"analyses[{i}].type"
            )
        
        return data


class RequestValidator:
    """Main request validator that routes to specific validators."""
    
    @staticmethod
    def validate(endpoint: str, data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate request data for specific endpoint.
        
        Returns:
            Tuple of (is_valid, validated_data, error_message)
        """
        try:
            # Route to appropriate validator based on endpoint
            if '/analyze' in endpoint:
                validated = AnalysisValidator.validate_analysis_request(data)
            elif '/testing/coverage' in endpoint:
                validated = TestingValidator.validate_coverage_request(data)
            elif '/testing/optimize' in endpoint:
                validated = TestingValidator.validate_optimization_request(data)
            elif '/integration/systems/analyze' in endpoint:
                validated = IntegrationValidator.validate_system_analysis_request(data)
            elif '/integration/events/publish' in endpoint:
                validated = IntegrationValidator.validate_event_publish(data)
            elif '/batch' in endpoint:
                validated = BatchValidator.validate_batch_request(data)
            else:
                # Default validation - just return data
                validated = data
            
            return True, validated, None
            
        except ValidationError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"


# Export validators
__all__ = [
    'ValidationError',
    'BaseValidator',
    'AnalysisValidator',
    'TestingValidator',
    'IntegrationValidator',
    'BatchValidator',
    'RequestValidator'
]