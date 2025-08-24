
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
Frontend Data Contracts & Standardization
=========================================

Ensures all backend endpoints return consistent, frontend-friendly data formats.
Provides data validation, transformation, and contract enforcement.

Author: TestMaster Team
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from flask import Blueprint, jsonify, request, make_response
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class ResponseStatus(Enum):
    """Standardized response statuses."""
    SUCCESS = "success"
    ERROR = "error"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    LOADING = "loading"

@dataclass
class StandardResponse:
    """Standard response format for all endpoints."""
    status: str
    message: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class PaginatedResponse:
    """Paginated response format."""
    items: List[Any]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

@dataclass
class MetricsData:
    """Standardized metrics format."""
    name: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    timestamp: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.tags is None:
            self.tags = {}

@dataclass
class HealthStatus:
    """Standardized health status format."""
    service: str
    status: str
    uptime: Optional[float] = None
    last_check: Optional[str] = None
    dependencies: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[List[MetricsData]] = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now().isoformat()
        if self.dependencies is None:
            self.dependencies = []
        if self.metrics is None:
            self.metrics = []

class FrontendDataContract:
    """
    Ensures consistent data contracts between backend and frontend.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('FrontendDataContract')
        
        # Standard field mappings
        self.field_mappings = {
            'id': 'id',
            'name': 'name', 
            'status': 'status',
            'timestamp': 'timestamp',
            'created_at': 'created_at',
            'updated_at': 'updated_at'
        }
        
        # Required fields for different data types
        self.required_fields = {
            'agent': ['id', 'name', 'role', 'status'],
            'crew': ['id', 'name', 'swarm_type', 'agent_count', 'status'],
            'swarm': ['id', 'name', 'architecture', 'status'],
            'task': ['id', 'description', 'status'],
            'metric': ['name', 'value', 'timestamp'],
            'health': ['service', 'status', 'timestamp']
        }
        
    def create_standard_response(self, 
                               status: str, 
                               data: Any = None, 
                               message: str = None,
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a standardized response."""
        response = StandardResponse(
            status=status,
            message=message,
            data=data,
            metadata=metadata or {}
        )
        return asdict(response)
        
    def create_success_response(self, data: Any = None, message: str = None) -> Dict[str, Any]:
        """Create a success response."""
        return self.create_standard_response(ResponseStatus.SUCCESS.value, data, message)
        
    def create_error_response(self, error: str, details: Any = None) -> Dict[str, Any]:
        """Create an error response."""
        return self.create_standard_response(
            ResponseStatus.ERROR.value, 
            details, 
            error
        )
        
    def create_health_response(self, service: str, 
                             status: str = "healthy",
                             metrics: List[MetricsData] = None,
                             dependencies: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized health response."""
        health = HealthStatus(
            service=service,
            status=status,
            metrics=metrics or [],
            dependencies=dependencies or []
        )
        return self.create_success_response(asdict(health))
        
    def create_paginated_response(self, 
                                items: List[Any],
                                page: int = 1,
                                page_size: int = 20,
                                total_items: int = None) -> Dict[str, Any]:
        """Create a paginated response."""
        if total_items is None:
            total_items = len(items)
            
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        
        paginated = PaginatedResponse(
            items=items,
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )
        
        return self.create_success_response(asdict(paginated))
        
    def validate_agent_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and standardize agent data."""
        required = self.required_fields['agent']
        validated = {}
        
        for field in required:
            if field not in data:
                if field == 'status':
                    validated[field] = 'unknown'
                else:
                    raise ValueError(f"Missing required field: {field}")
            else:
                validated[field] = data[field]
                
        # Add optional fields
        optional_fields = ['goal', 'backstory', 'capabilities', 'performance_metrics']
        for field in optional_fields:
            if field in data:
                validated[field] = data[field]
                
        return validated
        
    def validate_crew_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and standardize crew data."""
        required = self.required_fields['crew']
        validated = {}
        
        for field in required:
            if field not in data:
                if field == 'status':
                    validated[field] = 'unknown'
                elif field == 'agent_count':
                    validated[field] = 0
                else:
                    raise ValueError(f"Missing required field: {field}")
            else:
                validated[field] = data[field]
                
        # Add optional fields
        optional_fields = ['description', 'created_at', 'performance_metrics', 'agents']
        for field in optional_fields:
            if field in data:
                validated[field] = data[field]
                
        return validated
        
    def validate_metrics_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and standardize metrics data."""
        validated_metrics = []
        
        for metric in data:
            validated = {}
            required = self.required_fields['metric']
            
            for field in required:
                if field not in metric:
                    if field == 'timestamp':
                        validated[field] = datetime.now().isoformat()
                    else:
                        raise ValueError(f"Missing required field in metric: {field}")
                else:
                    validated[field] = metric[field]
                    
            # Add optional fields
            optional_fields = ['unit', 'tags', 'description']
            for field in optional_fields:
                if field in metric:
                    validated[field] = metric[field]
                    
            validated_metrics.append(validated)
            
        return validated_metrics
        
    def ensure_response_format(self, response_data: Any) -> Dict[str, Any]:
        """Ensure response follows standard format."""
        if isinstance(response_data, dict):
            # If it already has status, validate it
            if 'status' in response_data:
                valid_statuses = [s.value for s in ResponseStatus]
                if response_data['status'] not in valid_statuses:
                    response_data['status'] = ResponseStatus.SUCCESS.value
                    
                # Ensure timestamp exists
                if 'timestamp' not in response_data:
                    response_data['timestamp'] = datetime.now().isoformat()
                    
                return response_data
            else:
                # Wrap in standard format
                return self.create_success_response(response_data)
        else:
            # Wrap non-dict responses
            return self.create_success_response(response_data)
            
    def add_response_headers(self, response, cache_seconds: int = 30):
        """Add standard response headers."""
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = f'public, max-age={cache_seconds}'
        response.headers['X-API-Version'] = '2.0'
        response.headers['X-Timestamp'] = datetime.now().isoformat()
        return response

# Global data contract instance
data_contract = FrontendDataContract()

# Flask Blueprint for data contract API
data_contract_bp = Blueprint('data_contract', __name__)

@data_contract_bp.route('/validate', methods=['POST'])
def validate_data():
    """Validate data against frontend contracts."""
    try:
        request_data = request.get_json() or {}
        data_type = request_data.get('type')
        data = request_data.get('data')
        
        if not data_type or not data:
            return jsonify(data_contract.create_error_response(
                "Missing required fields: type, data"
            )), 400
            
        if data_type == 'agent':
            validated = data_contract.validate_agent_data(data)
        elif data_type == 'crew':
            validated = data_contract.validate_crew_data(data)
        elif data_type == 'metrics':
            validated = data_contract.validate_metrics_data(data)
        else:
            return jsonify(data_contract.create_error_response(
                f"Unsupported data type: {data_type}"
            )), 400
            
        return jsonify(data_contract.create_success_response(
            validated,
            "Data validation successful"
        ))
        
    except ValueError as e:
        return jsonify(data_contract.create_error_response(str(e))), 400
    except Exception as e:
        return jsonify(data_contract.create_error_response(str(e))), 500

@data_contract_bp.route('/schemas', methods=['GET'])
def get_data_schemas():
    """Get available data schemas and their required fields."""
    try:
        schemas = {}
        
        for data_type, fields in data_contract.required_fields.items():
            schemas[data_type] = {
                'required_fields': fields,
                'description': f"Schema for {data_type} data"
            }
            
        return jsonify(data_contract.create_success_response(schemas))
        
    except Exception as e:
        return jsonify(data_contract.create_error_response(str(e))), 500

@data_contract_bp.route('/transform', methods=['POST'])
def transform_response():
    """Transform data to standard response format."""
    try:
        request_data = request.get_json() or {}
        
        # Transform to standard format
        standardized = data_contract.ensure_response_format(request_data)
        
        return jsonify(standardized)
        
    except Exception as e:
        return jsonify(data_contract.create_error_response(str(e))), 500

@data_contract_bp.route('/health', methods=['GET'])
def data_contract_health():
    """Health check for data contract service."""
    return jsonify(data_contract.create_health_response(
        "Frontend Data Contracts",
        "healthy",
        [
            MetricsData("schemas_available", len(data_contract.required_fields)),
            MetricsData("field_mappings", len(data_contract.field_mappings))
        ]
    ))

# Middleware function to standardize all responses
def standardize_response_middleware():
    """Middleware to automatically standardize all API responses."""
    def middleware(response):
        if response.content_type == 'application/json':
            try:
                data = json.loads(response.get_data(as_text=True))
                standardized = data_contract.ensure_response_format(data)
                response.set_data(json.dumps(standardized))
                response = data_contract.add_response_headers(response)
            except (json.JSONDecodeError, Exception):
                # Leave response as-is if we can't process it
                pass
        return response
    
    return middleware

# Export key components
__all__ = [
    'FrontendDataContract',
    'StandardResponse',
    'PaginatedResponse', 
    'MetricsData',
    'HealthStatus',
    'ResponseStatus',
    'data_contract',
    'data_contract_bp',
    'standardize_response_middleware'
]