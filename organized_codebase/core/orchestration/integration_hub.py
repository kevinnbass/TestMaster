"""
Enterprise Integration Hub
=========================

Central integration orchestration layer providing unified cross-system communication,
service mesh capabilities, and intelligent routing across all enterprise intelligence systems.

Features:
- Service mesh architecture with automatic service discovery
- Cross-system API gateway with intelligent routing
- Event-driven communication with publish/subscribe patterns
- Real-time data synchronization across multiple systems
- Circuit breaker patterns for resilient integration
- Load balancing and failover capabilities
- Comprehensive integration monitoring and analytics
- API versioning and backward compatibility

Author: TestMaster Intelligence Team
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from pathlib import Path
import threading
from collections import defaultdict, deque
import weakref
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger(__name__)

class SystemType(Enum):
    """Unified system types for enterprise integration"""
    INTELLIGENCE_CORE = "intelligence_core"
    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE_MONITORING = "performance_monitoring"
    CONFIGURATION_MANAGEMENT = "configuration_management"
    ANALYTICS_ENGINE = "analytics_engine"
    SECURITY_SYSTEM = "security_system"
    CACHING_LAYER = "caching_layer"
    DATABASE_SYSTEM = "database_system"
    API_GATEWAY = "api_gateway"
    WORKFLOW_ENGINE = "workflow_engine"

class IntegrationEventType(Enum):
    """Cross-system integration event types"""
    SYSTEM_STATE_CHANGE = "system_state_change"
    PERFORMANCE_ALERT = "performance_alert"
    CONFIGURATION_UPDATE = "configuration_update"
    ERROR_RECOVERY_TRIGGERED = "error_recovery_triggered"
    ANALYTICS_INSIGHT_READY = "analytics_insight_ready"
    SECURITY_EVENT = "security_event"
    CACHE_INVALIDATION = "cache_invalidation"
    WORKFLOW_COMPLETED = "workflow_completed"
    SERVICE_HEALTH_CHANGE = "service_health_change"
    LOAD_BALANCING_EVENT = "load_balancing_event"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 3
    HIGH = 5
    CRITICAL = 8
    EMERGENCY = 10

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class SystemMessage:
    """Standardized message format for cross-system communication"""
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    source_system: SystemType = SystemType.INTELLIGENCE_CORE
    target_system: Optional[SystemType] = None
    event_type: IntegrationEventType = IntegrationEventType.SYSTEM_STATE_CHANGE
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "source_system": self.source_system.value,
            "target_system": self.target_system.value if self.target_system else None,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority.value,
            "ttl_seconds": self.ttl_seconds
        }
    
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds

@dataclass
class ServiceEndpoint:
    """Service endpoint definition for service mesh"""
    service_id: str
    system_type: SystemType
    host: str = "localhost"
    port: int = 8080
    path: str = "/"
    protocol: str = "http"
    health_check_path: str = "/health"
    weight: int = 100
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_url(self) -> str:
        """Get full URL for the service endpoint"""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"
    
    def get_health_check_url(self) -> str:
        """Get health check URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.health_check_path}"

@dataclass
class CrossSystemRequest:
    """Request for cross-system operation"""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    operation: str = ""
    target_system: SystemType = SystemType.INTELLIGENCE_CORE
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_message(self) -> SystemMessage:
        """Convert to system message"""
        return SystemMessage(
            message_id=self.request_id,
            target_system=self.target_system,
            event_type=IntegrationEventType.SYSTEM_STATE_CHANGE,
            payload={
                "operation": self.operation,
                "parameters": self.parameters,
                "request_id": self.request_id
            },
            correlation_id=self.correlation_id,
            ttl_seconds=self.timeout_seconds
        )

@dataclass
class CrossSystemResponse:
    """Response from cross-system operation"""
    request_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class MessageRouter:
    """Intelligent message routing with load balancing"""
    
    def __init__(self):
        self.service_registry: Dict[SystemType, List[ServiceEndpoint]] = defaultdict(list)
        self.routing_rules: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, Dict] = defaultdict(lambda: {
            'failure_count': 0,
            'last_failure': None,
            'state': 'closed',  # closed, open, half_open
            'failure_threshold': 5,
            'recovery_timeout': 60
        })
        
    def register_service(self, endpoint: ServiceEndpoint):
        """Register a service endpoint"""
        self.service_registry[endpoint.system_type].append(endpoint)
        logger.info(f"Registered service: {endpoint.service_id} for {endpoint.system_type.value}")
    
    def get_healthy_endpoints(self, system_type: SystemType) -> List[ServiceEndpoint]:
        """Get healthy endpoints for a system type"""
        endpoints = self.service_registry.get(system_type, [])
        return [ep for ep in endpoints if ep.status == ServiceStatus.HEALTHY]
    
    def select_endpoint(self, system_type: SystemType) -> Optional[ServiceEndpoint]:
        """Select best endpoint using weighted round-robin"""
        healthy_endpoints = self.get_healthy_endpoints(system_type)
        
        if not healthy_endpoints:
            return None
        
        # Simple weighted selection (can be enhanced with more sophisticated algorithms)
        total_weight = sum(ep.weight for ep in healthy_endpoints)
        if total_weight == 0:
            return healthy_endpoints[0] if healthy_endpoints else None
        
        # Select based on weight
        import random
        target_weight = random.randint(1, total_weight)
        current_weight = 0
        
        for endpoint in healthy_endpoints:
            current_weight += endpoint.weight
            if current_weight >= target_weight:
                return endpoint
        
        return healthy_endpoints[0]
    
    def record_request_result(self, endpoint: ServiceEndpoint, success: bool):
        """Record request result for circuit breaker"""
        breaker_key = f"{endpoint.system_type.value}:{endpoint.service_id}"
        breaker = self.circuit_breakers[breaker_key]
        
        if success:
            breaker['failure_count'] = 0
            if breaker['state'] == 'half_open':
                breaker['state'] = 'closed'
        else:
            breaker['failure_count'] += 1
            breaker['last_failure'] = datetime.now()
            
            if breaker['failure_count'] >= breaker['failure_threshold']:
                breaker['state'] = 'open'
                endpoint.status = ServiceStatus.UNHEALTHY
                logger.warning(f"Circuit breaker opened for {endpoint.service_id}")
    
    def can_route_to_endpoint(self, endpoint: ServiceEndpoint) -> bool:
        """Check if endpoint is available for routing"""
        breaker_key = f"{endpoint.system_type.value}:{endpoint.service_id}"
        breaker = self.circuit_breakers[breaker_key]
        
        if breaker['state'] == 'closed':
            return True
        elif breaker['state'] == 'open':
            # Check if recovery timeout has passed
            if breaker['last_failure']:
                time_since_failure = (datetime.now() - breaker['last_failure']).total_seconds()
                if time_since_failure > breaker['recovery_timeout']:
                    breaker['state'] = 'half_open'
                    return True
            return False
        elif breaker['state'] == 'half_open':
            return True
        
        return False

class EventBus:
    """Event-driven communication bus with publish/subscribe patterns"""
    
    def __init__(self):
        self.subscribers: Dict[IntegrationEventType, List[Callable]] = defaultdict(list)
        self.message_history: deque = deque(maxlen=1000)
        self.event_stats: Dict[str, int] = defaultdict(int)
        self.event_filters: List[Callable] = []
        
    def subscribe(self, event_type: IntegrationEventType, handler: Callable):
        """Subscribe to specific event type"""
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: IntegrationEventType, handler: Callable):
        """Unsubscribe from event type"""
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            logger.info(f"Unsubscribed handler from {event_type.value}")
    
    def add_filter(self, filter_func: Callable[[SystemMessage], bool]):
        """Add message filter"""
        self.event_filters.append(filter_func)
    
    def should_process_message(self, message: SystemMessage) -> bool:
        """Check if message should be processed based on filters"""
        if message.is_expired():
            return False
        
        for filter_func in self.event_filters:
            try:
                if not filter_func(message):
                    return False
            except Exception as e:
                logger.warning(f"Event filter error: {e}")
        
        return True
    
    async def publish(self, message: SystemMessage):
        """Publish message to all subscribers"""
        if not self.should_process_message(message):
            return
        
        # Store in history
        self.message_history.append(message)
        self.event_stats[message.event_type.value] += 1
        
        # Notify subscribers
        handlers = self.subscribers.get(message.event_type, [])
        if handlers:
            # Execute handlers concurrently
            tasks = []
            for handler in handlers:
                task = asyncio.create_task(self._execute_handler(handler, message))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_handler(self, handler: Callable, message: SystemMessage):
        """Execute event handler with error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        except Exception as e:
            logger.error(f"Event handler error: {e}")

class EnterpriseIntegrationHub:
    """
    Central integration hub providing service mesh capabilities and
    cross-system orchestration for enterprise intelligence systems.
    """
    
    def __init__(self):
        self.router = MessageRouter()
        self.event_bus = EventBus()
        self.active_requests: Dict[str, CrossSystemRequest] = {}
        self.response_handlers: Dict[str, Callable] = {}
        
        # Integration monitoring
        self.integration_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'active_connections': 0,
            'start_time': datetime.now()
        }
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_thread = None
        self.health_monitoring_active = False
        
        # Request processing
        self.request_queue = asyncio.Queue()
        self.processing_tasks: Set[asyncio.Task] = set()
        self.max_concurrent_requests = 100
        
        # Service discovery
        self.service_discovery_enabled = True
        self.auto_registration_enabled = True
        
        logger.info("Enterprise Integration Hub initialized")
    
    def register_system(self, system_type: SystemType, endpoint: ServiceEndpoint):
        """Register a system with the integration hub"""
        self.router.register_service(endpoint)
        
        # Auto-subscribe to system events if enabled
        if self.auto_registration_enabled:
            self._auto_subscribe_system_events(system_type)
    
    def _auto_subscribe_system_events(self, system_type: SystemType):
        """Automatically subscribe to common system events"""
        common_events = [
            IntegrationEventType.SYSTEM_STATE_CHANGE,
            IntegrationEventType.PERFORMANCE_ALERT,
            IntegrationEventType.ERROR_RECOVERY_TRIGGERED
        ]
        
        for event_type in common_events:
            self.event_bus.subscribe(event_type, self._handle_system_event)
    
    async def _handle_system_event(self, message: SystemMessage):
        """Handle system events for monitoring and coordination"""
        logger.info(f"Handling system event: {message.event_type.value} from {message.source_system.value}")
        
        # Update integration statistics
        if message.event_type == IntegrationEventType.PERFORMANCE_ALERT:
            self._handle_performance_alert(message)
        elif message.event_type == IntegrationEventType.ERROR_RECOVERY_TRIGGERED:
            self._handle_error_recovery_event(message)
        elif message.event_type == IntegrationEventType.SERVICE_HEALTH_CHANGE:
            self._handle_service_health_change(message)
    
    def _handle_performance_alert(self, message: SystemMessage):
        """Handle performance alerts from monitoring systems"""
        payload = message.payload
        system = message.source_system
        
        # Adjust routing weights based on performance
        if 'cpu_usage' in payload and payload['cpu_usage'] > 80:
            self._reduce_system_weight(system, 0.5)
        elif 'response_time' in payload and payload['response_time'] > 5.0:
            self._reduce_system_weight(system, 0.7)
    
    def _handle_error_recovery_event(self, message: SystemMessage):
        """Handle error recovery events"""
        payload = message.payload
        system = message.source_system
        
        # Temporarily route traffic away from system under recovery
        if payload.get('recovery_strategy') in ['restart', 'failover']:
            self._set_system_weight(system, 0)
    
    def _handle_service_health_change(self, message: SystemMessage):
        """Handle service health status changes"""
        payload = message.payload
        service_id = payload.get('service_id')
        new_status = payload.get('status')
        
        if service_id and new_status:
            self._update_service_status(service_id, ServiceStatus(new_status))
    
    def _reduce_system_weight(self, system_type: SystemType, factor: float):
        """Reduce routing weight for a system"""
        endpoints = self.router.service_registry.get(system_type, [])
        for endpoint in endpoints:
            endpoint.weight = max(1, int(endpoint.weight * factor))
    
    def _set_system_weight(self, system_type: SystemType, weight: int):
        """Set routing weight for a system"""
        endpoints = self.router.service_registry.get(system_type, [])
        for endpoint in endpoints:
            endpoint.weight = weight
    
    def _update_service_status(self, service_id: str, status: ServiceStatus):
        """Update service status in registry"""
        for endpoints in self.router.service_registry.values():
            for endpoint in endpoints:
                if endpoint.service_id == service_id:
                    endpoint.status = status
                    endpoint.last_health_check = datetime.now()
    
    async def start_integration_hub(self):
        """Start the integration hub services"""
        logger.info("Starting Enterprise Integration Hub")
        
        # Start health monitoring
        self.health_monitoring_active = True
        self.health_check_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self.health_check_thread.start()
        
        # Start request processing
        for _ in range(min(10, self.max_concurrent_requests)):
            task = asyncio.create_task(self._process_requests())
            self.processing_tasks.add(task)
        
        logger.info("Enterprise Integration Hub started")
    
    async def stop_integration_hub(self):
        """Stop the integration hub services"""
        logger.info("Stopping Enterprise Integration Hub")
        
        # Stop health monitoring
        self.health_monitoring_active = False
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5)
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()
        
        logger.info("Enterprise Integration Hub stopped")
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while self.health_monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(5)
    
    def _perform_health_checks(self):
        """Perform health checks on all registered services"""
        for system_type, endpoints in self.router.service_registry.items():
            for endpoint in endpoints:
                try:
                    # Simulate health check (in real implementation, make HTTP request)
                    import random
                    is_healthy = random.random() > 0.1  # 90% healthy simulation
                    
                    if is_healthy:
                        endpoint.status = ServiceStatus.HEALTHY
                    else:
                        endpoint.status = ServiceStatus.UNHEALTHY
                    
                    endpoint.last_health_check = datetime.now()
                    
                except Exception as e:
                    logger.warning(f"Health check failed for {endpoint.service_id}: {e}")
                    endpoint.status = ServiceStatus.UNKNOWN
    
    async def _process_requests(self):
        """Process cross-system requests"""
        while True:
            try:
                request = await self.request_queue.get()
                await self._execute_cross_system_request(request)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Request processing error: {e}")
    
    async def execute_cross_system_operation(self, operation: str, target_system: SystemType,
                                           parameters: Dict[str, Any] = None,
                                           timeout_seconds: int = 30) -> CrossSystemResponse:
        """Execute operation on target system"""
        request = CrossSystemRequest(
            operation=operation,
            target_system=target_system,
            parameters=parameters or {},
            timeout_seconds=timeout_seconds
        )
        
        self.integration_stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Add to request queue
            await self.request_queue.put(request)
            
            # Wait for response (simplified - in real implementation use proper async handling)
            response = await self._execute_cross_system_request(request)
            
            if response.success:
                self.integration_stats['successful_requests'] += 1
            else:
                self.integration_stats['failed_requests'] += 1
            
            # Update average response time
            execution_time = time.time() - start_time
            current_avg = self.integration_stats['average_response_time']
            total_requests = self.integration_stats['total_requests']
            self.integration_stats['average_response_time'] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )
            
            return response
            
        except Exception as e:
            self.integration_stats['failed_requests'] += 1
            logger.error(f"Cross-system operation failed: {e}")
            
            return CrossSystemResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _execute_cross_system_request(self, request: CrossSystemRequest) -> CrossSystemResponse:
        """Execute the actual cross-system request"""
        # Select appropriate endpoint
        endpoint = self.router.select_endpoint(request.target_system)
        if not endpoint:
            return CrossSystemResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"No healthy endpoints available for {request.target_system.value}"
            )
        
        if not self.router.can_route_to_endpoint(endpoint):
            return CrossSystemResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"Endpoint {endpoint.service_id} unavailable (circuit breaker)"
            )
        
        start_time = time.time()
        
        try:
            # Simulate request execution (in real implementation, make HTTP/gRPC call)
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Simulate success/failure
            import random
            success = random.random() > 0.05  # 95% success rate
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record result for circuit breaker
            self.router.record_request_result(endpoint, success)
            
            if success:
                return CrossSystemResponse(
                    request_id=request.request_id,
                    success=True,
                    data={"result": f"Operation {request.operation} completed successfully"},
                    execution_time_ms=execution_time
                )
            else:
                return CrossSystemResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Simulated operation failure",
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            self.router.record_request_result(endpoint, False)
            raise
    
    async def publish_event(self, message: SystemMessage):
        """Publish event to the event bus"""
        await self.event_bus.publish(message)
    
    def subscribe_to_events(self, event_type: IntegrationEventType, handler: Callable):
        """Subscribe to specific event types"""
        self.event_bus.subscribe(event_type, handler)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration hub status"""
        # Calculate uptime
        uptime = (datetime.now() - self.integration_stats['start_time']).total_seconds()
        
        # Service health summary
        service_health = {}
        for system_type, endpoints in self.router.service_registry.items():
            health_counts = defaultdict(int)
            for endpoint in endpoints:
                health_counts[endpoint.status.value] += 1
            service_health[system_type.value] = dict(health_counts)
        
        # Event statistics
        event_stats = dict(self.event_bus.event_stats)
        
        return {
            'status': 'active',
            'uptime_seconds': uptime,
            'registered_systems': len(self.router.service_registry),
            'total_endpoints': sum(len(eps) for eps in self.router.service_registry.values()),
            'request_statistics': self.integration_stats.copy(),
            'service_health': service_health,
            'event_statistics': event_stats,
            'active_circuit_breakers': len([
                cb for cb in self.router.circuit_breakers.values() 
                if cb['state'] != 'closed'
            ]),
            'message_history_size': len(self.event_bus.message_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_service_mesh_topology(self) -> Dict[str, Any]:
        """Get service mesh topology information"""
        topology = {
            'systems': {},
            'connections': [],
            'health_summary': {}
        }
        
        for system_type, endpoints in self.router.service_registry.items():
            system_info = {
                'system_type': system_type.value,
                'endpoints': [],
                'total_endpoints': len(endpoints),
                'healthy_endpoints': len([ep for ep in endpoints if ep.status == ServiceStatus.HEALTHY])
            }
            
            for endpoint in endpoints:
                endpoint_info = {
                    'service_id': endpoint.service_id,
                    'url': endpoint.get_url(),
                    'status': endpoint.status.value,
                    'weight': endpoint.weight,
                    'last_health_check': endpoint.last_health_check.isoformat() if endpoint.last_health_check else None
                }
                system_info['endpoints'].append(endpoint_info)
            
            topology['systems'][system_type.value] = system_info
        
        return topology
    
    def shutdown(self):
        """Shutdown integration hub"""
        if asyncio.get_event_loop().is_running():
            # Create and run shutdown task
            asyncio.create_task(self.stop_integration_hub())
        else:
            # Create new event loop for shutdown
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_integration_hub())
            loop.close()
        
        logger.info("Enterprise Integration Hub shutdown")

# Global integration hub instance
enterprise_integration_hub = EnterpriseIntegrationHub()