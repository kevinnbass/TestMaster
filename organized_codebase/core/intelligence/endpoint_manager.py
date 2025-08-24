"""
Focused Endpoint Manager

Handles API endpoint registration, management, and configuration for the API orchestrator.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class EndpointCategory(Enum):
    """API endpoint categories."""
    GENERATION = "generation"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"
    MANAGEMENT = "management"
    REPORTING = "reporting"
    INTEGRATION = "integration"


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"


@dataclass
class EndpointParameter:
    """API endpoint parameter definition."""
    name: str
    param_type: str  # string, number, boolean, object, array
    required: bool = True
    description: str = ""
    default_value: Any = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointExample:
    """API endpoint usage example."""
    name: str
    description: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    status_code: int = 200


@dataclass
class APIEndpoint:
    """Complete API endpoint definition."""
    endpoint_id: str
    service_type: str
    category: EndpointCategory
    path: str
    method: HTTPMethod
    description: str
    service_module: str
    service_function: str
    parameters: List[EndpointParameter] = field(default_factory=list)
    examples: List[EndpointExample] = field(default_factory=list)
    rate_limit: Optional[int] = None
    authentication_required: bool = True
    deprecated: bool = False
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class EndpointManager:
    """
    Focused endpoint manager for API orchestration.
    Handles endpoint registration, configuration, and management.
    """
    
    def __init__(self):
        """Initialize endpoint manager with configuration."""
        try:
            # Endpoint storage
            self.endpoints = {}  # endpoint_id -> APIEndpoint
            self.endpoints_by_path = {}  # path -> APIEndpoint
            self.endpoints_by_service = defaultdict(list)  # service_type -> List[APIEndpoint]
            self.endpoints_by_category = defaultdict(list)  # category -> List[APIEndpoint]
            
            # Configuration
            self.default_rate_limit = 100  # requests per minute
            self.default_authentication = True
            self.api_version = "1.0.0"
            
            # Statistics
            self.endpoint_stats = {
                'total_endpoints': 0,
                'endpoints_by_method': defaultdict(int),
                'endpoints_by_service': defaultdict(int),
                'endpoints_by_category': defaultdict(int),
                'deprecated_endpoints': 0
            }
            
            # Initialize default endpoints
            self._initialize_default_endpoints()
            
            logger.info("Endpoint Manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize endpoint manager: {e}")
            raise
    
    def register_endpoint(self, endpoint: APIEndpoint) -> bool:
        """
        Register a new API endpoint.
        
        Args:
            endpoint: Endpoint configuration to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate endpoint
            if not self._validate_endpoint(endpoint):
                logger.error(f"Invalid endpoint configuration: {endpoint.endpoint_id}")
                return False
            
            # Check for duplicates
            if endpoint.endpoint_id in self.endpoints:
                logger.warning(f"Endpoint {endpoint.endpoint_id} already registered")
                return False
            
            if endpoint.path in self.endpoints_by_path:
                logger.warning(f"Path {endpoint.path} already registered")
                return False
            
            # Register endpoint
            self.endpoints[endpoint.endpoint_id] = endpoint
            self.endpoints_by_path[endpoint.path] = endpoint
            self.endpoints_by_service[endpoint.service_type].append(endpoint)
            self.endpoints_by_category[endpoint.category].append(endpoint)
            
            # Update statistics
            self._update_endpoint_stats()
            
            logger.info(f"Endpoint registered: {endpoint.endpoint_id} -> {endpoint.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register endpoint {endpoint.endpoint_id}: {e}")
            return False
    
    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """
        Unregister an API endpoint.
        
        Args:
            endpoint_id: ID of endpoint to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if endpoint_id not in self.endpoints:
                logger.warning(f"Endpoint {endpoint_id} not found for unregistration")
                return False
            
            endpoint = self.endpoints[endpoint_id]
            
            # Remove from all indices
            del self.endpoints[endpoint_id]
            del self.endpoints_by_path[endpoint.path]
            
            self.endpoints_by_service[endpoint.service_type].remove(endpoint)
            self.endpoints_by_category[endpoint.category].remove(endpoint)
            
            # Update statistics
            self._update_endpoint_stats()
            
            logger.info(f"Endpoint unregistered: {endpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister endpoint {endpoint_id}: {e}")
            return False
    
    def get_endpoint(self, endpoint_id: str) -> Optional[APIEndpoint]:
        """
        Get endpoint by ID.
        
        Args:
            endpoint_id: Endpoint ID to lookup
            
        Returns:
            Endpoint configuration if found, None otherwise
        """
        try:
            return self.endpoints.get(endpoint_id)
        except Exception as e:
            logger.error(f"Failed to get endpoint {endpoint_id}: {e}")
            return None
    
    def get_endpoint_by_path(self, path: str) -> Optional[APIEndpoint]:
        """
        Get endpoint by path.
        
        Args:
            path: Endpoint path to lookup
            
        Returns:
            Endpoint configuration if found, None otherwise
        """
        try:
            return self.endpoints_by_path.get(path)
        except Exception as e:
            logger.error(f"Failed to get endpoint by path {path}: {e}")
            return None
    
    def list_endpoints(self, service_type: Optional[str] = None, 
                      category: Optional[EndpointCategory] = None,
                      include_deprecated: bool = True) -> List[APIEndpoint]:
        """
        List endpoints with optional filtering.
        
        Args:
            service_type: Filter by service type (optional)
            category: Filter by category (optional)
            include_deprecated: Include deprecated endpoints (default: True)
            
        Returns:
            List of matching endpoints
        """
        try:
            endpoints = []
            
            if service_type:
                endpoints = self.endpoints_by_service.get(service_type, [])
            elif category:
                endpoints = self.endpoints_by_category.get(category, [])
            else:
                endpoints = list(self.endpoints.values())
            
            # Filter deprecated endpoints if requested
            if not include_deprecated:
                endpoints = [ep for ep in endpoints if not ep.deprecated]
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []
    
    def update_endpoint(self, endpoint_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update endpoint configuration.
        
        Args:
            endpoint_id: Endpoint ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if endpoint_id not in self.endpoints:
                logger.warning(f"Endpoint {endpoint_id} not found for update")
                return False
            
            endpoint = self.endpoints[endpoint_id]
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(endpoint, field):
                    setattr(endpoint, field, value)
            
            # Update timestamp
            endpoint.updated_at = datetime.utcnow()
            
            # Update statistics
            self._update_endpoint_stats()
            
            logger.info(f"Endpoint updated: {endpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update endpoint {endpoint_id}: {e}")
            return False
    
    def deprecate_endpoint(self, endpoint_id: str, deprecation_message: str = "") -> bool:
        """
        Mark an endpoint as deprecated.
        
        Args:
            endpoint_id: Endpoint ID to deprecate
            deprecation_message: Optional deprecation message
            
        Returns:
            True if deprecation successful, False otherwise
        """
        try:
            updates = {
                'deprecated': True,
                'metadata': {'deprecation_message': deprecation_message}
            }
            return self.update_endpoint(endpoint_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to deprecate endpoint {endpoint_id}: {e}")
            return False
    
    def generate_api_documentation(self, service_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate API documentation for endpoints.
        
        Args:
            service_type: Generate docs for specific service type (optional)
            
        Returns:
            API documentation in OpenAPI-like format
        """
        try:
            endpoints = self.list_endpoints(service_type=service_type, include_deprecated=False)
            
            # Group endpoints by path for documentation
            paths = {}
            for endpoint in endpoints:
                if endpoint.path not in paths:
                    paths[endpoint.path] = {}
                
                # Add method to path
                method_info = {
                    'summary': endpoint.description,
                    'operationId': endpoint.endpoint_id,
                    'tags': [endpoint.service_type],
                    'parameters': self._format_parameters_for_docs(endpoint.parameters),
                    'responses': self._generate_response_docs(endpoint),
                    'examples': self._format_examples_for_docs(endpoint.examples)
                }
                
                if endpoint.authentication_required:
                    method_info['security'] = [{'bearerAuth': []}]
                
                if endpoint.rate_limit:
                    method_info['x-rate-limit'] = endpoint.rate_limit
                
                paths[endpoint.path][endpoint.method.value.lower()] = method_info
            
            return {
                'openapi': '3.0.0',
                'info': {
                    'title': 'Enterprise API',
                    'version': self.api_version,
                    'description': 'Enterprise API Documentation'
                },
                'paths': paths,
                'components': {
                    'securitySchemes': {
                        'bearerAuth': {
                            'type': 'http',
                            'scheme': 'bearer'
                        }
                    }
                },
                'statistics': self.get_endpoint_statistics()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate API documentation: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def validate_request(self, endpoint_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a request against endpoint parameters.
        
        Args:
            endpoint_id: Endpoint ID to validate against
            request_data: Request data to validate
            
        Returns:
            Validation result with errors if any
        """
        try:
            endpoint = self.get_endpoint(endpoint_id)
            if not endpoint:
                return {
                    'valid': False,
                    'errors': [f'Endpoint {endpoint_id} not found']
                }
            
            errors = []
            
            # Check required parameters
            for param in endpoint.parameters:
                if param.required and param.name not in request_data:
                    errors.append(f'Required parameter missing: {param.name}')
                    continue
                
                # Validate parameter type and rules
                if param.name in request_data:
                    value = request_data[param.name]
                    param_errors = self._validate_parameter(param, value)
                    errors.extend(param_errors)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'endpoint_id': endpoint_id
            }
            
        except Exception as e:
            logger.error(f"Failed to validate request for {endpoint_id}: {e}")
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}']
            }
    
    def get_endpoint_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive endpoint statistics.
        
        Returns:
            Dictionary containing endpoint statistics
        """
        try:
            self._update_endpoint_stats()
            
            # Calculate additional metrics
            total_endpoints = len(self.endpoints)
            active_endpoints = total_endpoints - self.endpoint_stats['deprecated_endpoints']
            
            return {
                'total_endpoints': total_endpoints,
                'active_endpoints': active_endpoints,
                'deprecated_endpoints': self.endpoint_stats['deprecated_endpoints'],
                'endpoints_by_method': dict(self.endpoint_stats['endpoints_by_method']),
                'endpoints_by_service': dict(self.endpoint_stats['endpoints_by_service']),
                'endpoints_by_category': dict(self.endpoint_stats['endpoints_by_category']),
                'api_version': self.api_version,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get endpoint statistics: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }
    
    # Private helper methods
    def _initialize_default_endpoints(self) -> None:
        """Initialize default enterprise endpoints."""
        try:
            # Health check endpoint
            health_endpoint = APIEndpoint(
                endpoint_id="system_health",
                service_type="system",
                category=EndpointCategory.MONITORING,
                path="/api/v1/health",
                method=HTTPMethod.GET,
                description="System health check",
                service_module="core.health",
                service_function="get_health_status",
                authentication_required=False
            )
            self.register_endpoint(health_endpoint)
            
            # API documentation endpoint
            docs_endpoint = APIEndpoint(
                endpoint_id="api_documentation",
                service_type="system",
                category=EndpointCategory.INTEGRATION,
                path="/api/v1/docs",
                method=HTTPMethod.GET,
                description="API documentation",
                service_module="core.docs",
                service_function="get_api_docs",
                authentication_required=False
            )
            self.register_endpoint(docs_endpoint)
            
        except Exception as e:
            logger.error(f"Failed to initialize default endpoints: {e}")
    
    def _validate_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Validate endpoint configuration."""
        try:
            # Required fields
            if not endpoint.endpoint_id:
                return False
            if not endpoint.path:
                return False
            if not endpoint.service_module:
                return False
            if not endpoint.service_function:
                return False
            
            # Path format validation
            if not endpoint.path.startswith('/'):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating endpoint: {e}")
            return False
    
    def _update_endpoint_stats(self) -> None:
        """Update endpoint statistics."""
        try:
            # Reset stats
            self.endpoint_stats = {
                'total_endpoints': len(self.endpoints),
                'endpoints_by_method': defaultdict(int),
                'endpoints_by_service': defaultdict(int),
                'endpoints_by_category': defaultdict(int),
                'deprecated_endpoints': 0
            }
            
            # Calculate stats
            for endpoint in self.endpoints.values():
                self.endpoint_stats['endpoints_by_method'][endpoint.method.value] += 1
                self.endpoint_stats['endpoints_by_service'][endpoint.service_type] += 1
                self.endpoint_stats['endpoints_by_category'][endpoint.category.value] += 1
                
                if endpoint.deprecated:
                    self.endpoint_stats['deprecated_endpoints'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to update endpoint stats: {e}")
    
    def _validate_parameter(self, param: EndpointParameter, value: Any) -> List[str]:
        """Validate a parameter value."""
        errors = []
        
        try:
            # Type validation
            if param.param_type == "string" and not isinstance(value, str):
                errors.append(f"Parameter {param.name} must be a string")
            elif param.param_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Parameter {param.name} must be a number")
            elif param.param_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Parameter {param.name} must be a boolean")
            elif param.param_type == "object" and not isinstance(value, dict):
                errors.append(f"Parameter {param.name} must be an object")
            elif param.param_type == "array" and not isinstance(value, list):
                errors.append(f"Parameter {param.name} must be an array")
            
            # Validation rules
            for rule, rule_value in param.validation_rules.items():
                if rule == "min_length" and isinstance(value, str):
                    if len(value) < rule_value:
                        errors.append(f"Parameter {param.name} must be at least {rule_value} characters")
                elif rule == "max_length" and isinstance(value, str):
                    if len(value) > rule_value:
                        errors.append(f"Parameter {param.name} must be at most {rule_value} characters")
                elif rule == "min_value" and isinstance(value, (int, float)):
                    if value < rule_value:
                        errors.append(f"Parameter {param.name} must be at least {rule_value}")
                elif rule == "max_value" and isinstance(value, (int, float)):
                    if value > rule_value:
                        errors.append(f"Parameter {param.name} must be at most {rule_value}")
            
        except Exception as e:
            errors.append(f"Parameter validation error for {param.name}: {str(e)}")
        
        return errors
    
    def _format_parameters_for_docs(self, parameters: List[EndpointParameter]) -> List[Dict[str, Any]]:
        """Format parameters for API documentation."""
        try:
            formatted = []
            for param in parameters:
                param_doc = {
                    'name': param.name,
                    'in': 'body',  # Simplified - could be query, path, etc.
                    'required': param.required,
                    'schema': {'type': param.param_type}
                }
                
                if param.description:
                    param_doc['description'] = param.description
                
                if param.default_value is not None:
                    param_doc['default'] = param.default_value
                
                formatted.append(param_doc)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting parameters for docs: {e}")
            return []
    
    def _generate_response_docs(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate response documentation for endpoint."""
        try:
            responses = {
                '200': {
                    'description': 'Successful response',
                    'content': {
                        'application/json': {
                            'schema': {'type': 'object'}
                        }
                    }
                },
                '400': {
                    'description': 'Bad request'
                },
                '500': {
                    'description': 'Internal server error'
                }
            }
            
            if endpoint.authentication_required:
                responses['401'] = {'description': 'Unauthorized'}
            
            if endpoint.rate_limit:
                responses['429'] = {'description': 'Rate limit exceeded'}
            
            return responses
            
        except Exception as e:
            logger.error(f"Error generating response docs: {e}")
            return {}
    
    def _format_examples_for_docs(self, examples: List[EndpointExample]) -> List[Dict[str, Any]]:
        """Format examples for API documentation."""
        try:
            formatted = []
            for example in examples:
                formatted.append({
                    'name': example.name,
                    'description': example.description,
                    'request': example.request,
                    'response': example.response
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting examples for docs: {e}")
            return []