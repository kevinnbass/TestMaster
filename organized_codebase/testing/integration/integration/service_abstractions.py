"""
Service Abstractions
===================

Unified service abstractions for integration with external systems,
APIs, databases, and enterprise services.

Author: Agent E - Infrastructure Consolidation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import json

from .integration_base import (
    IntegrationBase, IntegrationConfiguration, IntegrationContext,
    IntegrationStatus, IntegrationMode, IntegrationMetrics
)

T = TypeVar('T')
R = TypeVar('R')


class ServiceType(Enum):
    """Service type enumeration."""
    REST_API = "rest_api"
    SOAP_SERVICE = "soap_service"
    GRAPHQL_API = "graphql_api"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FILE_SERVICE = "file_service"
    CACHE_SERVICE = "cache_service"
    AUTHENTICATION = "authentication"
    MONITORING = "monitoring"
    LOGGING = "logging"
    STORAGE = "storage"
    COMPUTATION = "computation"
    WORKFLOW = "workflow"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class ServiceProtocol(Enum):
    """Service communication protocol."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MQTT = "mqtt"
    AMQP = "amqp"
    KAFKA = "kafka"
    REDIS = "redis"
    FTP = "ftp"
    SFTP = "sftp"
    SSH = "ssh"
    CUSTOM = "custom"


class AuthenticationType(Enum):
    """Authentication type enumeration."""
    NONE = "none"
    BASIC = "basic"
    BEARER_TOKEN = "bearer_token"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CERTIFICATE = "certificate"
    CUSTOM = "custom"


@dataclass
class ServiceCredentials:
    """Service authentication credentials."""
    
    auth_type: AuthenticationType = AuthenticationType.NONE
    
    # Basic authentication
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Token-based authentication
    token: Optional[str] = None
    api_key: Optional[str] = None
    
    # OAuth2 credentials
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    
    # Certificate authentication
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    
    # Custom credentials
    custom_credentials: Dict[str, Any] = field(default_factory=dict)
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {}
        
        if self.auth_type == AuthenticationType.BASIC and self.username and self.password:
            import base64
            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            headers['Authorization'] = f'Basic {credentials}'
        
        elif self.auth_type == AuthenticationType.BEARER_TOKEN and self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        elif self.auth_type == AuthenticationType.API_KEY and self.api_key:
            headers['X-API-Key'] = self.api_key
        
        elif self.auth_type == AuthenticationType.JWT and self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        return headers


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    
    # Basic endpoint information
    name: str
    url: str
    protocol: ServiceProtocol = ServiceProtocol.HTTPS
    method: str = "GET"
    
    # Path and query parameters
    path_parameters: Dict[str, str] = field(default_factory=dict)
    query_parameters: Dict[str, str] = field(default_factory=dict)
    
    # Headers and body
    headers: Dict[str, str] = field(default_factory=dict)
    body_template: Optional[str] = None
    
    # Request/response configuration
    content_type: str = "application/json"
    accept_type: str = "application/json"
    encoding: str = "utf-8"
    
    # Timeout and retry settings
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Validation settings
    validate_ssl: bool = True
    expected_status_codes: List[int] = field(default_factory=lambda: [200])
    
    def build_url(self, **kwargs) -> str:
        """Build complete URL with parameters."""
        url = self.url
        
        # Replace path parameters
        for key, value in self.path_parameters.items():
            if key in kwargs:
                url = url.replace(f"{{{key}}}", str(kwargs[key]))
        
        # Add query parameters
        query_params = {**self.query_parameters}
        for key, value in kwargs.items():
            if key not in self.path_parameters:
                query_params[key] = str(value)
        
        if query_params:
            import urllib.parse
            query_string = urllib.parse.urlencode(query_params)
            url += f"?{query_string}"
        
        return url
    
    def build_headers(self, credentials: Optional[ServiceCredentials] = None) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            'Content-Type': self.content_type,
            'Accept': self.accept_type,
            **self.headers
        }
        
        if credentials:
            headers.update(credentials.get_auth_headers())
        
        return headers


@dataclass
class ServiceRequest:
    """Service request data structure."""
    
    endpoint: ServiceEndpoint
    data: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    context: Optional[IntegrationContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            'endpoint': self.endpoint.name,
            'url': self.endpoint.build_url(**self.parameters),
            'method': self.endpoint.method,
            'headers': self.endpoint.build_headers(),
            'data': self.data,
            'parameters': self.parameters
        }


@dataclass
class ServiceResponse:
    """Service response data structure."""
    
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    data: Optional[Union[Dict[str, Any], List, str]] = None
    raw_content: Optional[bytes] = None
    
    # Response metadata
    response_time: float = 0.0
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    
    # Request context
    request: Optional[ServiceRequest] = None
    context: Optional[IntegrationContext] = None
    
    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return 200 <= self.status_code < 300
    
    @property
    def is_client_error(self) -> bool:
        """Check if response indicates client error."""
        return 400 <= self.status_code < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return 500 <= self.status_code < 600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'status_code': self.status_code,
            'headers': self.headers,
            'data': self.data,
            'response_time': self.response_time,
            'content_type': self.content_type,
            'content_length': self.content_length,
            'is_success': self.is_success
        }


class ServiceBase(IntegrationBase[ServiceRequest, ServiceResponse]):
    """
    Abstract base class for service integrations.
    
    Provides unified interface for service communication with
    authentication, error handling, and monitoring.
    """
    
    def __init__(self, 
                 config: IntegrationConfiguration,
                 service_type: ServiceType,
                 credentials: Optional[ServiceCredentials] = None):
        super().__init__(config)
        self.service_type = service_type
        self.credentials = credentials
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self._connection_pool = None
    
    # Abstract methods for service-specific implementation
    
    @abstractmethod
    async def _create_connection(self) -> Any:
        """Create service connection."""
        pass
    
    @abstractmethod
    async def _close_connection(self, connection: Any):
        """Close service connection."""
        pass
    
    @abstractmethod
    async def _send_request(self, 
                           request: ServiceRequest, 
                           connection: Any) -> ServiceResponse:
        """Send request using service-specific protocol."""
        pass
    
    # Service endpoint management
    
    def add_endpoint(self, endpoint: ServiceEndpoint):
        """Add service endpoint."""
        self.endpoints[endpoint.name] = endpoint
    
    def get_endpoint(self, name: str) -> Optional[ServiceEndpoint]:
        """Get service endpoint by name."""
        return self.endpoints.get(name)
    
    def remove_endpoint(self, name: str):
        """Remove service endpoint."""
        if name in self.endpoints:
            del self.endpoints[name]
    
    def list_endpoints(self) -> List[str]:
        """List available endpoint names."""
        return list(self.endpoints.keys())
    
    # Integration implementation
    
    async def initialize(self) -> bool:
        """Initialize service integration."""
        try:
            # Validate configuration
            if not self.config.name:
                raise ValueError("Service name is required")
            
            # Initialize connection pool if needed
            if self.config.max_concurrent_connections > 1:
                self._connection_pool = []
                for _ in range(self.config.connection_pool_size):
                    connection = await self._create_connection()
                    self._connection_pool.append(connection)
            
            return True
            
        except Exception as e:
            await self._emit_event("service_initialization_error", {"error": str(e)})
            return False
    
    async def connect(self) -> bool:
        """Establish service connection."""
        try:
            if not self._connection_pool:
                # Create single connection for non-pooled mode
                self._connection = await self._create_connection()
            
            self.status = IntegrationStatus.CONNECTED
            return True
            
        except Exception as e:
            await self._emit_event("service_connection_error", {"error": str(e)})
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from service."""
        try:
            if self._connection_pool:
                # Close all pooled connections
                for connection in self._connection_pool:
                    await self._close_connection(connection)
                self._connection_pool = []
            elif hasattr(self, '_connection'):
                # Close single connection
                await self._close_connection(self._connection)
                delattr(self, '_connection')
            
            self.status = IntegrationStatus.DISCONNECTED
            return True
            
        except Exception as e:
            await self._emit_event("service_disconnection_error", {"error": str(e)})
            return False
    
    async def send_request(self, 
                          request: ServiceRequest, 
                          context: IntegrationContext) -> ServiceResponse:
        """Send request to service."""
        try:
            # Get connection
            if self._connection_pool:
                connection = self._connection_pool.pop(0)
                try:
                    response = await self._send_request(request, connection)
                    return response
                finally:
                    # Return connection to pool
                    self._connection_pool.append(connection)
            else:
                response = await self._send_request(request, self._connection)
                return response
                
        except Exception as e:
            await self._emit_event("service_request_error", {
                "error": str(e),
                "endpoint": request.endpoint.name
            })
            raise
    
    async def receive_response(self, context: IntegrationContext) -> ServiceResponse:
        """Receive response from service (for async patterns)."""
        # Default implementation for synchronous services
        raise NotImplementedError("Async response handling not implemented")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform service health check."""
        try:
            # Try to establish connection
            if not self.is_connected:
                await self.connect()
            
            health_info = {
                "service_type": self.service_type.value,
                "status": self.status.value,
                "endpoints_count": len(self.endpoints),
                "connection_pool_size": len(self._connection_pool) if self._connection_pool else 0,
                "healthy": self.is_connected
            }
            
            # Perform service-specific health check
            service_health = await self._perform_service_health_check()
            health_info.update(service_health)
            
            return health_info
            
        except Exception as e:
            return {
                "service_type": self.service_type.value,
                "status": "error",
                "error": str(e),
                "healthy": False
            }
    
    async def _perform_service_health_check(self) -> Dict[str, Any]:
        """Perform service-specific health check."""
        # Default implementation - can be overridden by subclasses
        return {"service_specific_health": "ok"}
    
    # Convenience methods for common operations
    
    async def call_endpoint(self, 
                           endpoint_name: str,
                           data: Optional[Dict[str, Any]] = None,
                           parameters: Optional[Dict[str, Any]] = None,
                           context: Optional[IntegrationContext] = None) -> ServiceResponse:
        """Call service endpoint by name."""
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        
        request = ServiceRequest(
            endpoint=endpoint,
            data=data,
            parameters=parameters or {},
            context=context
        )
        
        if not context:
            context = IntegrationContext(
                integration_id=self.integration_id,
                session_id=f"call_{int(datetime.now().timestamp())}"
            )
        
        return await self.execute(request, context)
    
    async def batch_call(self, 
                        calls: List[Dict[str, Any]],
                        context: Optional[IntegrationContext] = None) -> List[ServiceResponse]:
        """Execute multiple endpoint calls."""
        responses = []
        
        for call in calls:
            try:
                response = await self.call_endpoint(**call)
                responses.append(response)
            except Exception as e:
                # Create error response
                error_response = ServiceResponse(
                    status_code=500,
                    data={"error": str(e)},
                    context=context
                )
                responses.append(error_response)
        
        return responses
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"service_type={self.service_type.value}, "
                f"endpoints={len(self.endpoints)}, "
                f"status={self.status.value})")


# Service registry for managing service instances
class ServiceRegistry:
    """Registry for managing service instances."""
    
    def __init__(self):
        self._services: Dict[str, ServiceBase] = {}
        self._service_types: Dict[str, type] = {}
    
    def register_service_type(self, service_type: str, service_class: type):
        """Register service class for type."""
        self._service_types[service_type] = service_class
    
    def create_service(self, 
                      service_id: str,
                      service_type: str,
                      config: IntegrationConfiguration,
                      credentials: Optional[ServiceCredentials] = None) -> ServiceBase:
        """Create and register service instance."""
        if service_type not in self._service_types:
            raise ValueError(f"Unknown service type: {service_type}")
        
        service_class = self._service_types[service_type]
        service = service_class(config, ServiceType(service_type), credentials)
        
        self._services[service_id] = service
        return service
    
    def get_service(self, service_id: str) -> Optional[ServiceBase]:
        """Get service instance by ID."""
        return self._services.get(service_id)
    
    def remove_service(self, service_id: str):
        """Remove service from registry."""
        if service_id in self._services:
            del self._services[service_id]
    
    def list_services(self) -> List[str]:
        """List registered service IDs."""
        return list(self._services.keys())
    
    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceBase]:
        """Get services by type."""
        return [service for service in self._services.values() 
                if service.service_type == service_type]
    
    async def start_all_services(self) -> Dict[str, bool]:
        """Start all registered services."""
        results = {}
        for service_id, service in self._services.items():
            results[service_id] = await service.start()
        return results
    
    async def stop_all_services(self) -> Dict[str, bool]:
        """Stop all registered services."""
        results = {}
        for service_id, service in self._services.items():
            results[service_id] = await service.stop()
        return results


__all__ = [
    'ServiceType',
    'ServiceProtocol',
    'AuthenticationType',
    'ServiceCredentials',
    'ServiceEndpoint',
    'ServiceRequest',
    'ServiceResponse',
    'ServiceBase',
    'ServiceRegistry'
]