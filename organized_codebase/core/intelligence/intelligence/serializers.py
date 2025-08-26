"""
TestMaster Intelligence Hub Data Serializers
============================================

Data serialization and transformation for API responses.
Converts complex objects to JSON-serializable formats.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import asdict, is_dataclass
from enum import Enum
import json


class IntelligenceSerializer:
    """Base serializer for intelligence hub data structures.
    
    Comprehensive serialization framework that converts complex intelligence
    data structures into JSON-compatible formats for API responses. Handles
    nested objects, dataclasses, enums, and custom objects recursively.
    
    Key Features:
    - **Recursive Serialization**: Deep serialization of nested structures
    - **Type Safety**: Proper handling of datetime, enum, and dataclass types
    - **Custom Object Support**: Serialization of arbitrary Python objects
    - **JSON Compatibility**: All output is JSON-serializable
    - **Error Resilience**: Graceful handling of serialization failures
    
    Supported Types:
    - Primitives (str, int, float, bool, None)
    - Collections (list, tuple, dict)
    - Dataclasses (automatic field extraction)
    - Enums (value extraction)
    - Datetime objects (ISO format conversion)
    - Custom objects (attribute extraction)
    
    Example:
        >>> from dataclasses import dataclass
        >>> from datetime import datetime
        >>> 
        >>> @dataclass
        >>> class TestResult:
        ...     name: str
        ...     timestamp: datetime
        ...     passed: bool
        ...
        >>> result = TestResult("test_calc", datetime.now(), True)
        >>> serialized = IntelligenceSerializer.serialize_value(result)
        >>> print(serialized['timestamp'])  # ISO format string
        2024-08-21T23:30:00.123456
    """
    
    @staticmethod
    def serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format string.
        
        Args:
            dt: Datetime object to serialize, or None
            
        Returns:
            ISO format string representation or None if input is None
            
        Example:
            >>> from datetime import datetime
            >>> dt = datetime(2024, 8, 21, 23, 30, 0)
            >>> iso_str = IntelligenceSerializer.serialize_datetime(dt)
            >>> print(iso_str)
            2024-08-21T23:30:00
        """
        return dt.isoformat() if dt else None
    
    @staticmethod
    def serialize_enum(enum_val: Optional[Enum]) -> Optional[str]:
        """Serialize enum to string value."""
        return enum_val.value if enum_val else None
    
    @staticmethod
    def serialize_dataclass(obj: Any) -> Dict[str, Any]:
        """Serialize dataclass to dictionary."""
        if is_dataclass(obj):
            result = {}
            for field_name, field_value in asdict(obj).items():
                result[field_name] = IntelligenceSerializer.serialize_value(field_value)
            return result
        return {}
    
    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Recursively serialize any value to JSON-compatible format."""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return IntelligenceSerializer.serialize_datetime(value)
        elif isinstance(value, Enum):
            return IntelligenceSerializer.serialize_enum(value)
        elif is_dataclass(value):
            return IntelligenceSerializer.serialize_dataclass(value)
        elif isinstance(value, dict):
            return {k: IntelligenceSerializer.serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [IntelligenceSerializer.serialize_value(item) for item in value]
        elif hasattr(value, '__dict__'):
            # Handle custom objects
            return IntelligenceSerializer.serialize_object(value)
        else:
            # Primitive types
            return value
    
    @staticmethod
    def serialize_object(obj: Any) -> Dict[str, Any]:
        """Serialize custom object to dictionary."""
        result = {}
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):  # Skip methods
                        result[attr_name] = IntelligenceSerializer.serialize_value(attr_value)
                except:
                    pass  # Skip attributes that can't be accessed
        return result


class AnalysisSerializer(IntelligenceSerializer):
    """Specialized serializer for analysis results and data science outputs.
    
    Extends the base IntelligenceSerializer with domain-specific knowledge
    for serializing analysis results, cross-system data, and ML insights.
    Handles complex analysis objects with statistical data and recommendations.
    
    Key Capabilities:
    - **Analysis Results**: Comprehensive analysis object serialization
    - **Statistical Data**: Confidence scores and quality metrics
    - **Recommendations**: Structured recommendation serialization  
    - **Cross-System Analysis**: Multi-system integration data
    - **ML Insights**: Machine learning model outputs and features
    - **Enhanced Features**: Advanced analysis capabilities
    
    Supported Analysis Types:
    - UnifiedAnalysis (comprehensive analysis results)
    - CrossSystemAnalysis (multi-system integration analysis)
    - Statistical analysis with confidence intervals
    - ML model outputs with feature importance
    - Recommendation engines with scoring
    
    Example:
        >>> analysis = UnifiedAnalysis(
        ...     analysis_id="analysis_001",
        ...     confidence_score=0.95,
        ...     results={"accuracy": 0.87, "precision": 0.92}
        ... )
        >>> serialized = AnalysisSerializer.serialize_unified_analysis(analysis)
        >>> print(serialized['confidence_score'])
        0.95
    """
    
    @staticmethod
    def serialize_unified_analysis(analysis) -> Dict[str, Any]:
        """Serialize UnifiedAnalysis object to JSON-compatible dictionary.
        
        Converts a UnifiedAnalysis object into a structured dictionary
        suitable for API responses, preserving all analysis metadata,
        results, and recommendations.
        
        Args:
            analysis: UnifiedAnalysis object containing analysis results
            
        Returns:
            Dictionary with all analysis data in JSON-compatible format
            
        Structure:
            - analysis_id: Unique identifier for the analysis
            - timestamp: When the analysis was performed (ISO format)
            - analysis_type: Type of analysis performed
            - source_data: Input data used for analysis
            - results: Main analysis results and metrics
            - confidence_score: Statistical confidence (0.0-1.0)
            - quality_metrics: Analysis quality indicators
            - recommendations: Structured recommendations list
            - enhanced_features: Advanced analysis capabilities used
        """
        return {
            'analysis_id': getattr(analysis, 'analysis_id', getattr(analysis, 'id', '')),
            'timestamp': IntelligenceSerializer.serialize_datetime(analysis.timestamp),
            'analysis_type': IntelligenceSerializer.serialize_enum(analysis.analysis_type),
            'source_data': getattr(analysis, 'source_data', {}),
            'results': IntelligenceSerializer.serialize_value(analysis.results),
            'confidence_score': analysis.confidence_score,
            'quality_metrics': getattr(analysis, 'quality_metrics', {}),
            'recommendations': getattr(analysis, 'recommendations', []),
            'enhanced_features': getattr(analysis, 'enhanced_features', {})
        }
    
    @staticmethod
    def serialize_cross_system_analysis(analysis) -> Dict[str, Any]:
        """Serialize CrossSystemAnalysis object."""
        return {
            'analysis_id': analysis.analysis_id,
            'timestamp': IntelligenceSerializer.serialize_datetime(analysis.timestamp),
            'systems_analyzed': analysis.systems_analyzed,
            'analysis_duration': analysis.analysis_duration,
            'system_correlations': analysis.system_correlations,
            'performance_correlations': analysis.performance_correlations,
            'error_correlations': analysis.error_correlations,
            'health_scores': analysis.system_health_scores,
            'bottleneck_analysis': analysis.bottleneck_analysis,
            'optimization_opportunities': analysis.optimization_opportunities,
            'predicted_failures': analysis.predicted_failures,
            'capacity_forecasts': analysis.capacity_forecasts
        }


class TestSerializer(IntelligenceSerializer):
    """Serializer for test-related data."""
    
    @staticmethod
    def serialize_test_execution_result(result) -> Dict[str, Any]:
        """Serialize TestExecutionResult object."""
        return {
            'test_id': result.test_id,
            'test_name': result.test_name,
            'status': result.status,
            'execution_time': result.execution_time,
            'timestamp': IntelligenceSerializer.serialize_datetime(result.timestamp),
            'coverage_data': result.coverage_data,
            'performance_metrics': result.performance_metrics,
            'error_message': result.error_message,
            'error_category': result.error_category,
            'optimization_score': result.optimization_score,
            'predicted_failure_probability': result.predicted_failure_probability
        }
    
    @staticmethod
    def serialize_test_suite_analysis(analysis) -> Dict[str, Any]:
        """Serialize TestSuiteAnalysis object."""
        return {
            'total_tests': analysis.total_tests,
            'passed_tests': analysis.passed_tests,
            'failed_tests': analysis.failed_tests,
            'skipped_tests': analysis.skipped_tests,
            'total_execution_time': analysis.total_execution_time,
            'coverage_metrics': {
                'line_coverage': analysis.line_coverage,
                'branch_coverage': analysis.branch_coverage,
                'function_coverage': analysis.function_coverage,
                'class_coverage': analysis.class_coverage,
                'file_coverage': analysis.file_coverage
            },
            'statistical_analysis': {
                'coverage_confidence': analysis.coverage_confidence,
                'statistical_power': analysis.statistical_power,
                'sample_adequacy': analysis.sample_adequacy
            },
            'coverage_gaps': analysis.coverage_gaps,
            'optimization_opportunities': analysis.optimization_opportunities,
            'critical_paths': analysis.critical_paths,
            'circular_dependencies': analysis.circular_dependencies
        }
    
    @staticmethod
    def serialize_unified_test(test) -> Dict[str, Any]:
        """Serialize UnifiedTest object."""
        return {
            'test_id': test.test_id,
            'test_name': test.test_name,
            'test_type': IntelligenceSerializer.serialize_enum(test.test_type),
            'description': getattr(test, 'description', ''),
            'dependencies': getattr(test, 'dependencies', []),
            'tags': getattr(test, 'tags', []),
            'complexity_level': getattr(test, 'complexity_level', 'medium'),
            'estimated_execution_time': getattr(test, 'estimated_execution_time', 0),
            'test_data': getattr(test, 'test_data', {})
        }


class IntegrationSerializer(IntelligenceSerializer):
    """Serializer for integration-related data."""
    
    @staticmethod
    def serialize_integration_endpoint(endpoint) -> Dict[str, Any]:
        """Serialize IntegrationEndpoint object."""
        return {
            'endpoint_id': endpoint.endpoint_id,
            'name': endpoint.name,
            'url': endpoint.url,
            'integration_type': IntelligenceSerializer.serialize_enum(endpoint.integration_type),
            'status': IntelligenceSerializer.serialize_enum(endpoint.status),
            'availability_percentage': endpoint.availability_percentage,
            'last_successful_connection': IntelligenceSerializer.serialize_datetime(
                endpoint.last_successful_connection
            ),
            'last_error': endpoint.last_error,
            'response_times': endpoint.response_times[-10:] if endpoint.response_times else [],  # Last 10
            'error_rates': endpoint.error_rates,
            'features': {
                'correlation_tracking': endpoint.correlation_tracking,
                'circuit_breaker_enabled': endpoint.circuit_breaker_enabled,
                'websocket_enabled': endpoint.websocket_enabled,
                'event_streaming': endpoint.event_streaming
            }
        }
    
    @staticmethod
    def serialize_integration_event(event) -> Dict[str, Any]:
        """Serialize IntegrationEvent object."""
        return {
            'event_id': event.event_id,
            'timestamp': IntelligenceSerializer.serialize_datetime(event.timestamp),
            'source_system': event.source_system,
            'target_system': event.target_system,
            'event_type': event.event_type,
            'status': event.status,
            'payload': event.payload,
            'correlation_id': event.correlation_id,
            'processing_time': event.processing_time,
            'error_message': event.error_message,
            'retry_count': event.retry_count
        }


class MetricsSerializer(IntelligenceSerializer):
    """Serializer for metrics data."""
    
    @staticmethod
    def serialize_unified_metric(metric) -> Dict[str, Any]:
        """Serialize UnifiedMetric object."""
        return {
            'metric_id': metric.metric_id,
            'name': metric.name,
            'value': metric.value,
            'timestamp': IntelligenceSerializer.serialize_datetime(metric.timestamp),
            'unit': metric.unit,
            'metadata': metric.metadata,
            'tags': metric.tags,
            'aggregation_type': metric.aggregation_type,
            'statistical_properties': {
                'mean': getattr(metric, 'mean', None),
                'std_dev': getattr(metric, 'std_dev', None),
                'min': getattr(metric, 'min_value', None),
                'max': getattr(metric, 'max_value', None)
            },
            'ml_features': getattr(metric, 'ml_features', {}),
            'anomaly_score': getattr(metric, 'anomaly_score', 0.0)
        }


class ResponseFormatter:
    """Format API responses consistently."""
    
    @staticmethod
    def success_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
        """Format successful response."""
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': IntelligenceSerializer.serialize_value(data)
        }
        if message:
            response['message'] = message
        return response
    
    @staticmethod
    def error_response(error: str, code: Optional[str] = None, details: Optional[Dict] = None) -> Dict[str, Any]:
        """Format error response."""
        response = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'error': error
        }
        if code:
            response['error_code'] = code
        if details:
            response['details'] = details
        return response
    
    @staticmethod
    def paginated_response(data: List[Any], page: int, page_size: int, total: int) -> Dict[str, Any]:
        """Format paginated response."""
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': [IntelligenceSerializer.serialize_value(item) for item in data],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': (total + page_size - 1) // page_size
            }
        }


# Export all serializers
__all__ = [
    'IntelligenceSerializer',
    'AnalysisSerializer',
    'TestSerializer',
    'IntegrationSerializer',
    'MetricsSerializer',
    'ResponseFormatter'
]