"""
Streamlined API Orchestrator

Enterprise API orchestration orchestrator - now modularized for simplicity.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

# Import modular API orchestration components
from .api import (
    ServiceRegistry, ServiceInfo, ServiceType, ServiceStatus,
    RequestRouter, RoutingRequest, RequestPriority,
    EndpointManager, APIEndpoint, EndpointCategory, HTTPMethod
)

logger = logging.getLogger(__name__)


class APIOrchestrator:
    """
    Streamlined API orchestration orchestrator.
    Coordinates service registration, request routing, and endpoint management through modular components.
    """
    
    def __init__(self):
        """Initialize the API orchestrator with modular components."""
        try:
            # Initialize core orchestration components
            self.service_registry = ServiceRegistry()
            self.request_router = RequestRouter()
            self.endpoint_manager = EndpointManager()
            
            # Initialize orchestration state
            self.orchestration_metrics = {
                'total_requests_processed': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_processing_time': 0.0
            }
            self.active_sessions = {}
            
            # Initialize enterprise API endpoints
            self._initialize_enterprise_endpoints()
            
            logger.info("API Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize API orchestrator: {e}")
            raise
    
    # High-level orchestration operations
    async def process_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an API request through the complete orchestration pipeline.
        Handles routing, validation, and service coordination.
        """
        try:
            start_time = datetime.utcnow()
            self.orchestration_metrics['total_requests_processed'] += 1
            
            # Extract request information
            endpoint_path = request_data.get('path', '')
            method = request_data.get('method', 'GET')
            request_id = request_data.get('request_id', f"req_{datetime.utcnow().timestamp()}")
            
            # Get endpoint configuration
            endpoint = self.endpoint_manager.get_endpoint_by_path(endpoint_path)
            if not endpoint:
                self.orchestration_metrics['failed_requests'] += 1
                return {
                    'status': 'endpoint_not_found',
                    'error': f'Endpoint {endpoint_path} not found',
                    'request_id': request_id
                }
            
            # Validate request data
            validation_result = self.endpoint_manager.validate_request(
                endpoint.endpoint_id, request_data.get('payload', {})
            )
            if not validation_result['valid']:
                self.orchestration_metrics['failed_requests'] += 1
                return {
                    'status': 'validation_failed',
                    'errors': validation_result['errors'],
                    'request_id': request_id
                }
            
            # Create routing request
            routing_request = RoutingRequest(
                request_id=request_id,
                service_type=endpoint.service_type,
                endpoint_path=endpoint_path,
                method=method,
                priority=self._determine_request_priority(request_data),
                headers=request_data.get('headers', {}),
                payload=request_data.get('payload', {}),
                timeout=request_data.get('timeout', 30.0)
            )
            
            # Route request to service
            routing_result = await self.request_router.route_request(routing_request)
            if routing_result['status'] != 'routed':
                self.orchestration_metrics['failed_requests'] += 1
                return {
                    'status': 'routing_failed',
                    'error': routing_result.get('error', 'Unknown routing error'),
                    'request_id': request_id
                }
            
            # Process request (mock - in real implementation, would call actual service)
            processing_result = await self._process_service_request(endpoint, routing_result, request_data)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_processing_metrics(processing_time, processing_result['status'] == 'success')
            
            if processing_result['status'] == 'success':
                self.orchestration_metrics['successful_requests'] += 1
            else:
                self.orchestration_metrics['failed_requests'] += 1
            
            return {
                'status': processing_result['status'],
                'data': processing_result.get('data', {}),
                'error': processing_result.get('error'),
                'processing_time_ms': processing_time,
                'endpoint_id': endpoint.endpoint_id,
                'service_id': routing_result.get('target_service_id'),
                'request_id': request_id
            }
            
        except Exception as e:
            logger.error(f"Failed to process API request: {e}")
            self.orchestration_metrics['failed_requests'] += 1
            return {
                'status': 'orchestration_error',
                'error': str(e),
                'request_id': request_data.get('request_id', 'unknown')
            }
    
    # Service management operations (delegate to service registry)
    async def register_service(self, service_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new service - delegates to service registry.
        """
        try:
            # Convert dict to ServiceInfo object
            service_obj = ServiceInfo(
                service_id=service_info['service_id'],
                service_type=ServiceType(service_info['service_type']),
                name=service_info['name'],
                version=service_info.get('version', '1.0.0'),
                endpoint_url=service_info['endpoint_url'],
                health_check_url=service_info.get('health_check_url', service_info['endpoint_url'] + '/health'),
                metadata=service_info.get('metadata', {})
            )
            
            success = await self.service_registry.register_service(service_obj)
            
            if success:
                # Register routing targets for service endpoints
                await self._register_service_routing_targets(service_obj)
            
            return {
                'status': 'registered' if success else 'registration_failed',
                'service_id': service_info['service_id']
            }
            
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return {
                'status': 'registration_error',
                'error': str(e)
            }
    
    async def unregister_service(self, service_id: str) -> Dict[str, Any]:
        """
        Unregister a service - delegates to service registry.
        """
        try:
            success = await self.service_registry.unregister_service(service_id)
            
            if success:
                # Remove routing targets
                await self._unregister_service_routing_targets(service_id)
            
            return {
                'status': 'unregistered' if success else 'unregistration_failed',
                'service_id': service_id
            }
            
        except Exception as e:
            logger.error(f"Failed to unregister service: {e}")
            return {
                'status': 'unregistration_error',
                'error': str(e)
            }
    
    def discover_services(self, service_type: Optional[str] = None, 
                         status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discover services - delegates to service registry.
        """
        try:
            service_type_enum = ServiceType(service_type) if service_type else None
            status_enum = ServiceStatus(status_filter) if status_filter else None
            
            services = self.service_registry.discover_services(service_type_enum, status_enum)
            
            return [
                {
                    'service_id': service.service_id,
                    'service_type': service.service_type.value,
                    'name': service.name,
                    'version': service.version,
                    'endpoint_url': service.endpoint_url,
                    'status': service.status.value,
                    'response_time_ms': service.response_time_ms,
                    'success_count': service.success_count,
                    'error_count': service.error_count
                }
                for service in services
            ]
            
        except Exception as e:
            logger.error(f"Failed to discover services: {e}")
            return []
    
    # Endpoint management operations (delegate to endpoint manager)
    def register_endpoint(self, endpoint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register API endpoint - delegates to endpoint manager.
        """
        try:
            # Convert dict to APIEndpoint object
            endpoint = APIEndpoint(
                endpoint_id=endpoint_config['endpoint_id'],
                service_type=endpoint_config['service_type'],
                category=EndpointCategory(endpoint_config['category']),
                path=endpoint_config['path'],
                method=HTTPMethod(endpoint_config['method']),
                description=endpoint_config['description'],
                service_module=endpoint_config['service_module'],
                service_function=endpoint_config['service_function'],
                rate_limit=endpoint_config.get('rate_limit'),
                authentication_required=endpoint_config.get('authentication_required', True),
                version=endpoint_config.get('version', '1.0.0'),
                metadata=endpoint_config.get('metadata', {})
            )
            
            success = self.endpoint_manager.register_endpoint(endpoint)
            
            return {
                'status': 'registered' if success else 'registration_failed',
                'endpoint_id': endpoint_config['endpoint_id']
            }
            
        except Exception as e:
            logger.error(f"Failed to register endpoint: {e}")
            return {
                'status': 'registration_error',
                'error': str(e)
            }
    
    def get_api_documentation(self, service_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate API documentation - delegates to endpoint manager.
        """
        try:
            return self.endpoint_manager.generate_api_documentation(service_type)
        except Exception as e:
            logger.error(f"Failed to get API documentation: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def list_endpoints(self, service_type: Optional[str] = None, 
                      category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List endpoints - delegates to endpoint manager.
        """
        try:
            category_enum = EndpointCategory(category) if category else None
            endpoints = self.endpoint_manager.list_endpoints(service_type, category_enum)
            
            return [
                {
                    'endpoint_id': ep.endpoint_id,
                    'service_type': ep.service_type,
                    'category': ep.category.value,
                    'path': ep.path,
                    'method': ep.method.value,
                    'description': ep.description,
                    'rate_limit': ep.rate_limit,
                    'authentication_required': ep.authentication_required,
                    'deprecated': ep.deprecated,
                    'version': ep.version
                }
                for ep in endpoints
            ]
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []
    
    # Routing operations (delegate to request router)
    def configure_routing(self, routing_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure request routing - delegates to request router.
        """
        try:
            # Set routing strategy
            if 'strategy' in routing_config:
                from .api.request_router import RoutingStrategy
                strategy = RoutingStrategy(routing_config['strategy'])
                success = self.request_router.set_routing_strategy(strategy)
                
                if not success:
                    return {
                        'status': 'configuration_failed',
                        'error': 'Failed to set routing strategy'
                    }
            
            # Configure features
            if 'enable_load_balancing' in routing_config:
                self.request_router.enable_load_balancing = routing_config['enable_load_balancing']
            
            if 'enable_circuit_breaker' in routing_config:
                self.request_router.enable_circuit_breaker = routing_config['enable_circuit_breaker']
            
            if 'enable_rate_limiting' in routing_config:
                self.request_router.enable_rate_limiting = routing_config['enable_rate_limiting']
            
            return {
                'status': 'configured',
                'applied_settings': routing_config
            }
            
        except Exception as e:
            logger.error(f"Failed to configure routing: {e}")
            return {
                'status': 'configuration_error',
                'error': str(e)
            }
    
    # Monitoring and statistics
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestration metrics from all components.
        """
        try:
            # Get metrics from all components
            service_stats = self.service_registry.get_registry_statistics()
            routing_stats = self.request_router.get_routing_statistics()
            endpoint_stats = self.endpoint_manager.get_endpoint_statistics()
            
            return {
                'orchestration_metrics': self.orchestration_metrics,
                'service_registry': service_stats,
                'request_router': routing_stats,
                'endpoint_manager': endpoint_stats,
                'active_sessions': len(self.active_sessions),
                'system_health': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get orchestration metrics: {e}")
            return {
                'system_health': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        """
        try:
            # Check component health
            registry_health = len(self.service_registry.services) > 0
            router_health = self.request_router.routing_metrics['total_requests'] >= 0
            endpoint_health = len(self.endpoint_manager.endpoints) > 0
            
            overall_health = registry_health and router_health and endpoint_health
            
            return {
                'status': 'healthy' if overall_health else 'degraded',
                'components': {
                    'service_registry': 'healthy' if registry_health else 'degraded',
                    'request_router': 'healthy' if router_health else 'degraded',
                    'endpoint_manager': 'healthy' if endpoint_health else 'degraded'
                },
                'metrics_summary': {
                    'total_services': len(self.service_registry.services),
                    'total_endpoints': len(self.endpoint_manager.endpoints),
                    'total_requests_processed': self.orchestration_metrics['total_requests_processed'],
                    'success_rate': self._calculate_success_rate()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Private helper methods
    def _initialize_enterprise_endpoints(self) -> None:
        """Initialize default enterprise API endpoints."""
        try:
            # Documentation service endpoints
            doc_endpoints = [
                {
                    'endpoint_id': 'doc_generate_api',
                    'service_type': 'documentation',
                    'category': 'generation',
                    'path': '/api/v1/documentation/generate',
                    'method': 'POST',
                    'description': 'Generate comprehensive API documentation',
                    'service_module': 'core.intelligence.documentation.auto_generator',
                    'service_function': 'generate_documentation'
                },
                {
                    'endpoint_id': 'doc_analyze_quality',
                    'service_type': 'documentation',
                    'category': 'analysis',
                    'path': '/api/v1/documentation/analyze/quality',
                    'method': 'POST',
                    'description': 'Analyze documentation quality with AI insights',
                    'service_module': 'core.intelligence.documentation.enterprise.documentation_intelligence',
                    'service_function': 'analyze_documentation_intelligence'
                }
            ]
            
            # Security service endpoints
            security_endpoints = [
                {
                    'endpoint_id': 'sec_vulnerability_scan',
                    'service_type': 'security',
                    'category': 'analysis',
                    'path': '/api/v1/security/scan/vulnerabilities',
                    'method': 'POST',
                    'description': 'Comprehensive vulnerability scanning',
                    'service_module': 'core.intelligence.security.enterprise.security_validator',
                    'service_function': 'run_vulnerability_scan'
                },
                {
                    'endpoint_id': 'sec_compliance_check',
                    'service_type': 'security',
                    'category': 'analysis',
                    'path': '/api/v1/security/compliance/check',
                    'method': 'POST',
                    'description': 'Compliance validation and reporting',
                    'service_module': 'core.intelligence.security.enterprise.security_validator',
                    'service_function': 'validate_compliance'
                }
            ]
            
            # Register all endpoints
            all_endpoints = doc_endpoints + security_endpoints
            for endpoint_config in all_endpoints:
                self.register_endpoint(endpoint_config)
            
        except Exception as e:
            logger.error(f"Failed to initialize enterprise endpoints: {e}")
    
    async def _register_service_routing_targets(self, service_info: ServiceInfo) -> None:
        """Register routing targets for a service."""
        try:
            # Get endpoints for this service type
            endpoints = self.endpoint_manager.list_endpoints(service_type=service_info.service_type.value)
            
            for endpoint in endpoints:
                self.request_router.register_routing_target(
                    service_id=service_info.service_id,
                    service_type=service_info.service_type.value,
                    endpoint_path=endpoint.path,
                    endpoint_url=f"{service_info.endpoint_url}{endpoint.path}",
                    weight=1.0
                )
                
        except Exception as e:
            logger.error(f"Failed to register routing targets for {service_info.service_id}: {e}")
    
    async def _unregister_service_routing_targets(self, service_id: str) -> None:
        """Unregister routing targets for a service."""
        try:
            # Remove all routing targets for this service
            for endpoint_path in list(self.request_router.routing_table.keys()):
                self.request_router.unregister_routing_target(service_id, endpoint_path)
                
        except Exception as e:
            logger.error(f"Failed to unregister routing targets for {service_id}: {e}")
    
    def _determine_request_priority(self, request_data: Dict[str, Any]) -> RequestPriority:
        """Determine request priority based on request data."""
        try:
            # Check for explicit priority
            priority_str = request_data.get('priority', 'normal').upper()
            
            if priority_str == 'CRITICAL':
                return RequestPriority.CRITICAL
            elif priority_str == 'HIGH':
                return RequestPriority.HIGH
            elif priority_str == 'LOW':
                return RequestPriority.LOW
            else:
                return RequestPriority.NORMAL
                
        except Exception as e:
            logger.error(f"Error determining request priority: {e}")
            return RequestPriority.NORMAL
    
    async def _process_service_request(self, endpoint: APIEndpoint, routing_result: Dict, 
                                     request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through service (mock implementation)."""
        try:
            # Mock service processing - in real implementation, would make actual service call
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Mock successful response
            return {
                'status': 'success',
                'data': {
                    'message': f'Request processed successfully by {endpoint.service_function}',
                    'endpoint_id': endpoint.endpoint_id,
                    'service_module': endpoint.service_module,
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Service request processing failed: {e}")
            return {
                'status': 'service_error',
                'error': str(e)
            }
    
    def _update_processing_metrics(self, processing_time_ms: float, success: bool) -> None:
        """Update processing metrics."""
        try:
            # Update average processing time
            total_requests = self.orchestration_metrics['total_requests_processed']
            current_avg = self.orchestration_metrics['average_processing_time']
            
            if total_requests > 1:
                self.orchestration_metrics['average_processing_time'] = (
                    (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
                )
            else:
                self.orchestration_metrics['average_processing_time'] = processing_time_ms
                
        except Exception as e:
            logger.error(f"Error updating processing metrics: {e}")
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        try:
            total = self.orchestration_metrics['total_requests_processed']
            successful = self.orchestration_metrics['successful_requests']
            
            if total == 0:
                return 100.0
            
            return (successful / total) * 100.0
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return 0.0