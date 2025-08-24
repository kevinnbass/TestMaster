"""
API Validation Framework
=======================
Comprehensive API documentation and validation system for TestMaster Intelligence Hub.

This module provides automated API endpoint validation, OpenAPI/Swagger integration,
and comprehensive REST API health checking capabilities.

Author: Agent D - Documentation & Validation Excellence  
Phase: Hour 2 - API Documentation & Validation Systems
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import asyncio
import aiohttp
import requests
from urllib.parse import urljoin
import yaml
import re
import time
import logging

logger = logging.getLogger(__name__)

class APIEndpointType(Enum):
    """Types of API endpoints."""
    REST_GET = "rest_get"
    REST_POST = "rest_post"
    REST_PUT = "rest_put"
    REST_DELETE = "rest_delete"
    REST_PATCH = "rest_patch"
    WEBSOCKET = "websocket"
    GRAPHQL = "graphql"

class ValidationStatus(Enum):
    """Validation status for API endpoints."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"

class HTTPStatus(Enum):
    """HTTP status categories."""
    SUCCESS = "2xx"
    REDIRECT = "3xx"
    CLIENT_ERROR = "4xx"
    SERVER_ERROR = "5xx"

@dataclass
class APIEndpoint:
    """Represents an API endpoint for validation."""
    path: str
    methods: List[str]
    endpoint_name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    authentication_required: bool = False
    rate_limit: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Result of API endpoint validation."""
    endpoint: APIEndpoint
    status: ValidationStatus
    response_time: float
    status_code: Optional[int]
    response_data: Optional[Dict[str, Any]]
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    health_score: float = 0.0

@dataclass
class APIValidationReport:
    """Comprehensive API validation report."""
    base_url: str
    total_endpoints: int
    healthy_endpoints: int
    warning_endpoints: int
    error_endpoints: int
    average_response_time: float
    validation_results: List[ValidationResult]
    overall_health_score: float
    recommendations: List[str]
    generation_timestamp: datetime = field(default_factory=datetime.now)

class OpenAPIDocumentationGenerator:
    """Generates OpenAPI/Swagger documentation for API endpoints."""
    
    def __init__(self, base_info: Dict[str, Any] = None):
        self.base_info = base_info or {
            "title": "TestMaster Intelligence Hub API",
            "version": "1.0.0",
            "description": "Comprehensive API for TestMaster intelligence capabilities",
            "contact": {
                "name": "Agent D - Documentation & Validation",
                "email": "agent-d@testmaster.ai"
            }
        }
    
    def generate_openapi_spec(self, endpoints: List[APIEndpoint], 
                            base_url: str = "http://localhost:5000") -> Dict[str, Any]:
        """Generate complete OpenAPI 3.0 specification."""
        
        openapi_spec = {
            "openapi": "3.0.3",
            "info": self.base_info,
            "servers": [{"url": base_url, "description": "TestMaster Intelligence Hub"}],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": self._generate_security_schemes()
            },
            "tags": self._generate_tags(endpoints)
        }
        
        # Generate path specifications
        for endpoint in endpoints:
            path_spec = self._generate_path_spec(endpoint)
            openapi_spec["paths"][endpoint.path] = path_spec
        
        return openapi_spec
    
    def _generate_path_spec(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate OpenAPI path specification for an endpoint."""
        path_spec = {}
        
        for method in endpoint.methods:
            if method.upper() in ['OPTIONS', 'HEAD']:
                continue  # Skip standard HTTP methods
                
            method_spec = {
                "summary": endpoint.description or f"{method.upper()} {endpoint.path}",
                "description": f"Endpoint: {endpoint.endpoint_name}",
                "operationId": endpoint.endpoint_name.replace('.', '_'),
                "tags": endpoint.tags or ["intelligence"],
                "responses": self._generate_responses()
            }
            
            # Add parameters if present
            if endpoint.parameters:
                method_spec["parameters"] = self._generate_parameters(endpoint.parameters)
            
            # Add request body for POST/PUT/PATCH
            if method.upper() in ['POST', 'PUT', 'PATCH'] and endpoint.request_schema:
                method_spec["requestBody"] = self._generate_request_body(endpoint.request_schema)
            
            # Add security if required
            if endpoint.authentication_required:
                method_spec["security"] = [{"bearerAuth": []}]
            
            path_spec[method.lower()] = method_spec
        
        return path_spec
    
    def _generate_parameters(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate OpenAPI parameters specification."""
        parameters = []
        
        for param_name, param_info in params.items():
            param_spec = {
                "name": param_name,
                "in": param_info.get("location", "query"),
                "required": param_info.get("required", False),
                "schema": {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", f"Parameter {param_name}")
                }
            }
            parameters.append(param_spec)
        
        return parameters
    
    def _generate_request_body(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OpenAPI request body specification."""
        return {
            "required": True,
            "content": {
                "application/json": {
                    "schema": schema
                }
            }
        }
    
    def _generate_responses(self) -> Dict[str, Any]:
        """Generate standard API responses."""
        return {
            "200": {
                "description": "Successful response",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "data": {"type": "object"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "400": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "details": {"type": "object"}
                            }
                        }
                    }
                }
            },
            "500": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "timestamp": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    
    def _generate_security_schemes(self) -> Dict[str, Any]:
        """Generate security scheme definitions."""
        return {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            },
            "apiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            }
        }
    
    def _generate_tags(self, endpoints: List[APIEndpoint]) -> List[Dict[str, Any]]:
        """Generate API tags from endpoints."""
        all_tags = set()
        for endpoint in endpoints:
            all_tags.update(endpoint.tags or ["intelligence"])
        
        return [
            {
                "name": tag,
                "description": f"Operations related to {tag}"
            }
            for tag in sorted(all_tags)
        ]

class APIEndpointValidator:
    """Validates API endpoints for functionality and performance."""
    
    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    async def validate_endpoint(self, endpoint: APIEndpoint) -> ValidationResult:
        """Validate a single API endpoint."""
        start_time = time.time()
        
        try:
            # Test GET endpoints first (safest)
            method_to_test = self._select_test_method(endpoint.methods)
            url = urljoin(self.base_url, endpoint.path.lstrip('/'))
            
            # Prepare test request
            test_data = self._generate_test_data(endpoint)
            
            # Make request
            if method_to_test == 'GET':
                response = self.session.get(url, timeout=self.timeout, params=test_data)
            elif method_to_test == 'POST':
                response = self.session.post(url, json=test_data, timeout=self.timeout)
            else:
                # For other methods, try GET first as fallback
                response = self.session.get(url, timeout=self.timeout)
            
            response_time = time.time() - start_time
            
            # Analyze response
            status = self._determine_status(response.status_code, response_time)
            health_score = self._calculate_health_score(response.status_code, response_time)
            
            try:
                response_data = response.json() if response.text else None
            except:
                response_data = {"raw_response": response.text[:500]}
            
            return ValidationResult(
                endpoint=endpoint,
                status=status,
                response_time=response_time,
                status_code=response.status_code,
                response_data=response_data,
                health_score=health_score
            )
            
        except requests.exceptions.Timeout:
            return ValidationResult(
                endpoint=endpoint,
                status=ValidationStatus.TIMEOUT,
                response_time=self.timeout,
                status_code=None,
                response_data=None,
                error_message="Request timeout",
                health_score=0.0
            )
            
        except requests.exceptions.ConnectionError:
            return ValidationResult(
                endpoint=endpoint,
                status=ValidationStatus.UNAVAILABLE,
                response_time=time.time() - start_time,
                status_code=None,
                response_data=None,
                error_message="Connection error",
                health_score=0.0
            )
            
        except Exception as e:
            return ValidationResult(
                endpoint=endpoint,
                status=ValidationStatus.ERROR,
                response_time=time.time() - start_time,
                status_code=None,
                response_data=None,
                error_message=str(e),
                health_score=0.0
            )
    
    def _select_test_method(self, methods: List[str]) -> str:
        """Select the safest method to test."""
        method_priority = ['GET', 'HEAD', 'OPTIONS', 'POST', 'PUT', 'PATCH', 'DELETE']
        
        for method in method_priority:
            if method in [m.upper() for m in methods]:
                return method
        
        return methods[0] if methods else 'GET'
    
    def _generate_test_data(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate safe test data for endpoint validation."""
        test_data = {}
        
        # Generate test data based on parameters
        if endpoint.parameters:
            for param_name, param_info in endpoint.parameters.items():
                param_type = param_info.get("type", "string")
                
                if param_type == "string":
                    test_data[param_name] = "test_value"
                elif param_type == "integer":
                    test_data[param_name] = 1
                elif param_type == "boolean":
                    test_data[param_name] = True
                elif param_type == "array":
                    test_data[param_name] = ["test_item"]
        
        return test_data
    
    def _determine_status(self, status_code: int, response_time: float) -> ValidationStatus:
        """Determine validation status based on response."""
        if status_code is None:
            return ValidationStatus.UNAVAILABLE
        
        if 200 <= status_code < 300:
            return ValidationStatus.HEALTHY if response_time < 2.0 else ValidationStatus.WARNING
        elif 300 <= status_code < 400:
            return ValidationStatus.WARNING
        elif 400 <= status_code < 500:
            return ValidationStatus.WARNING  # Client errors might be expected for test requests
        else:
            return ValidationStatus.ERROR
    
    def _calculate_health_score(self, status_code: int, response_time: float) -> float:
        """Calculate health score (0.0 - 1.0) for endpoint."""
        if status_code is None:
            return 0.0
        
        # Base score from status code
        if 200 <= status_code < 300:
            status_score = 1.0
        elif 300 <= status_code < 400:
            status_score = 0.8
        elif 400 <= status_code < 500:
            status_score = 0.6  # Might be expected for test requests
        else:
            status_score = 0.2
        
        # Response time factor
        if response_time < 0.5:
            time_factor = 1.0
        elif response_time < 2.0:
            time_factor = 0.8
        elif response_time < 5.0:
            time_factor = 0.6
        else:
            time_factor = 0.3
        
        return status_score * time_factor
    
    async def validate_all_endpoints(self, endpoints: List[APIEndpoint]) -> APIValidationReport:
        """Validate all endpoints and generate comprehensive report."""
        
        logger.info(f"Starting validation of {len(endpoints)} endpoints")
        
        # Validate endpoints concurrently (with limited concurrency)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def validate_with_semaphore(endpoint):
            async with semaphore:
                return await self.validate_endpoint(endpoint)
        
        validation_results = []
        
        # For now, run synchronously to avoid async complexity with requests
        for endpoint in endpoints:
            result = await self.validate_endpoint(endpoint)
            validation_results.append(result)
        
        # Generate comprehensive report
        return self._generate_validation_report(validation_results)
    
    def _generate_validation_report(self, results: List[ValidationResult]) -> APIValidationReport:
        """Generate comprehensive validation report."""
        
        total_endpoints = len(results)
        healthy_endpoints = sum(1 for r in results if r.status == ValidationStatus.HEALTHY)
        warning_endpoints = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        error_endpoints = sum(1 for r in results if r.status in [ValidationStatus.ERROR, 
                                                              ValidationStatus.TIMEOUT, 
                                                              ValidationStatus.UNAVAILABLE])
        
        # Calculate average response time (only for successful requests)
        successful_results = [r for r in results if r.status_code is not None]
        avg_response_time = (sum(r.response_time for r in successful_results) / 
                           len(successful_results)) if successful_results else 0.0
        
        # Calculate overall health score
        overall_health_score = (sum(r.health_score for r in results) / 
                              total_endpoints) if total_endpoints > 0 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return APIValidationReport(
            base_url=self.base_url,
            total_endpoints=total_endpoints,
            healthy_endpoints=healthy_endpoints,
            warning_endpoints=warning_endpoints,
            error_endpoints=error_endpoints,
            average_response_time=avg_response_time,
            validation_results=results,
            overall_health_score=overall_health_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate improvement recommendations based on validation results."""
        recommendations = []
        
        # Check for slow endpoints
        slow_endpoints = [r for r in results if r.response_time > 2.0 and r.status_code]
        if slow_endpoints:
            recommendations.append(f"Optimize performance for {len(slow_endpoints)} slow endpoints")
        
        # Check for error endpoints
        error_endpoints = [r for r in results if r.status in [ValidationStatus.ERROR, 
                                                             ValidationStatus.UNAVAILABLE]]
        if error_endpoints:
            recommendations.append(f"Fix {len(error_endpoints)} non-functional endpoints")
        
        # Check for timeout endpoints
        timeout_endpoints = [r for r in results if r.status == ValidationStatus.TIMEOUT]
        if timeout_endpoints:
            recommendations.append(f"Address timeout issues in {len(timeout_endpoints)} endpoints")
        
        # Overall health recommendations
        overall_health = sum(r.health_score for r in results) / len(results) if results else 0
        if overall_health < 0.8:
            recommendations.append("Overall API health below 80% - comprehensive review needed")
        elif overall_health < 0.9:
            recommendations.append("API health good but could be improved")
        
        return recommendations

class APIValidationFramework:
    """Main framework coordinating all API validation and documentation capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.openapi_generator = OpenAPIDocumentationGenerator()
        self.validator = APIEndpointValidator(base_url)
        self.discovered_endpoints: List[APIEndpoint] = []
    
    def discover_intelligence_endpoints(self) -> List[APIEndpoint]:
        """Discover TestMaster Intelligence Hub API endpoints."""
        # Based on the analysis from Hour 1, define all intelligence endpoints
        endpoints = [
            APIEndpoint(
                path="/api/intelligence/status",
                methods=["GET"],
                endpoint_name="intelligence_api.get_intelligence_status",
                description="Get overall intelligence system status",
                tags=["status", "health"]
            ),
            APIEndpoint(
                path="/api/intelligence/analyze",
                methods=["POST"],
                endpoint_name="intelligence_api.analyze",
                description="Perform unified intelligence analysis",
                tags=["analysis"],
                request_schema={
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string"},
                        "target": {"type": "string"},
                        "options": {"type": "object"}
                    },
                    "required": ["analysis_type", "target"]
                }
            ),
            APIEndpoint(
                path="/api/intelligence/analytics/analyze",
                methods=["POST"],
                endpoint_name="intelligence_api.analyze_metrics",
                description="Analyze metrics and generate insights",
                tags=["analytics"],
                request_schema={
                    "type": "object",
                    "properties": {
                        "metrics_source": {"type": "string"},
                        "time_range": {"type": "string"},
                        "filters": {"type": "object"}
                    }
                }
            ),
            APIEndpoint(
                path="/api/intelligence/analytics/correlations",
                methods=["POST"],
                endpoint_name="intelligence_api.find_correlations",
                description="Find correlations in data",
                tags=["analytics", "correlations"]
            ),
            APIEndpoint(
                path="/api/intelligence/analytics/predict",
                methods=["POST"],
                endpoint_name="intelligence_api.predict_trends",
                description="Predict trends based on historical data",
                tags=["analytics", "prediction"]
            ),
            APIEndpoint(
                path="/api/intelligence/testing/coverage",
                methods=["POST"],
                endpoint_name="intelligence_api.analyze_coverage",
                description="Analyze test coverage metrics",
                tags=["testing", "coverage"]
            ),
            APIEndpoint(
                path="/api/intelligence/testing/optimize",
                methods=["POST"],
                endpoint_name="intelligence_api.optimize_tests",
                description="Optimize test execution and selection",
                tags=["testing", "optimization"]
            ),
            APIEndpoint(
                path="/api/intelligence/testing/predict-failures",
                methods=["POST"],
                endpoint_name="intelligence_api.predict_failures",
                description="Predict potential test failures",
                tags=["testing", "prediction"]
            ),
            APIEndpoint(
                path="/api/intelligence/testing/generate",
                methods=["POST"],
                endpoint_name="intelligence_api.generate_tests",
                description="Generate integration tests",
                tags=["testing", "generation"]
            ),
            APIEndpoint(
                path="/api/intelligence/integration/systems/analyze",
                methods=["POST"],
                endpoint_name="intelligence_api.analyze_systems",
                description="Analyze system integration patterns",
                tags=["integration", "analysis"]
            ),
            APIEndpoint(
                path="/api/intelligence/integration/endpoints",
                methods=["GET"],
                endpoint_name="intelligence_api.get_endpoints",
                description="Get all registered endpoints",
                tags=["integration", "discovery"]
            ),
            APIEndpoint(
                path="/api/intelligence/integration/endpoints/<endpoint_id>/health",
                methods=["GET"],
                endpoint_name="intelligence_api.get_endpoint_health",
                description="Check health of specific endpoint",
                tags=["integration", "health"],
                parameters={
                    "endpoint_id": {
                        "type": "string",
                        "location": "path",
                        "required": True,
                        "description": "Unique identifier for the endpoint"
                    }
                }
            ),
            APIEndpoint(
                path="/api/intelligence/integration/events/publish",
                methods=["POST"],
                endpoint_name="intelligence_api.publish_event",
                description="Publish integration event",
                tags=["integration", "events"]
            ),
            APIEndpoint(
                path="/api/intelligence/integration/performance",
                methods=["GET"],
                endpoint_name="intelligence_api.get_performance",
                description="Get integration performance metrics",
                tags=["integration", "performance"]
            ),
            APIEndpoint(
                path="/api/intelligence/monitoring/realtime",
                methods=["GET"],
                endpoint_name="intelligence_api.get_realtime_metrics",
                description="Get real-time monitoring metrics",
                tags=["monitoring", "realtime"]
            ),
            APIEndpoint(
                path="/api/intelligence/batch/analyze",
                methods=["POST"],
                endpoint_name="intelligence_api.batch_analyze",
                description="Perform batch analysis operations",
                tags=["batch", "analysis"]
            ),
            APIEndpoint(
                path="/api/intelligence/health",
                methods=["GET"],
                endpoint_name="intelligence_api.health_check",
                description="Comprehensive health check of all intelligence systems",
                tags=["health", "status"]
            )
        ]
        
        self.discovered_endpoints = endpoints
        return endpoints
    
    async def generate_complete_api_documentation(self) -> Dict[str, Any]:
        """Generate complete API documentation with validation."""
        
        # Discover endpoints if not already done
        if not self.discovered_endpoints:
            self.discover_intelligence_endpoints()
        
        # Generate OpenAPI specification
        openapi_spec = self.openapi_generator.generate_openapi_spec(
            self.discovered_endpoints, 
            self.base_url
        )
        
        # Validate all endpoints
        validation_report = await self.validator.validate_all_endpoints(
            self.discovered_endpoints
        )
        
        return {
            "openapi_specification": openapi_spec,
            "validation_report": {
                "summary": {
                    "total_endpoints": validation_report.total_endpoints,
                    "healthy_endpoints": validation_report.healthy_endpoints,
                    "warning_endpoints": validation_report.warning_endpoints,
                    "error_endpoints": validation_report.error_endpoints,
                    "overall_health_score": validation_report.overall_health_score,
                    "average_response_time": validation_report.average_response_time
                },
                "recommendations": validation_report.recommendations,
                "detailed_results": [
                    {
                        "endpoint": result.endpoint.path,
                        "status": result.status.value,
                        "response_time": result.response_time,
                        "status_code": result.status_code,
                        "health_score": result.health_score,
                        "error_message": result.error_message
                    }
                    for result in validation_report.validation_results
                ]
            },
            "generation_timestamp": datetime.now().isoformat()
        }
    
    def export_openapi_yaml(self, openapi_spec: Dict[str, Any], 
                           output_path: str = "api_documentation.yaml") -> str:
        """Export OpenAPI specification to YAML file."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)
        
        return output_path
    
    def export_validation_report(self, validation_data: Dict[str, Any], 
                               output_path: str = "api_validation_report.json") -> str:
        """Export validation report to JSON file."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, default=str)
        
        return output_path

# Global API validation framework instance
_api_validation_framework = APIValidationFramework()

def get_api_validation_framework() -> APIValidationFramework:
    """Get the global API validation framework instance."""
    return _api_validation_framework

async def validate_intelligence_apis(base_url: str = "http://localhost:5000") -> Dict[str, Any]:
    """High-level function to validate all intelligence APIs."""
    framework = APIValidationFramework(base_url)
    return await framework.generate_complete_api_documentation()

def generate_openapi_documentation(output_path: str = "openapi_spec.yaml") -> str:
    """Generate and export OpenAPI documentation."""
    framework = get_api_validation_framework()
    endpoints = framework.discover_intelligence_endpoints()
    openapi_spec = framework.openapi_generator.generate_openapi_spec(endpoints)
    return framework.export_openapi_yaml(openapi_spec, output_path)