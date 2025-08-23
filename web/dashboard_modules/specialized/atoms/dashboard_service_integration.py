#!/usr/bin/env python3
"""
Dashboard Service Integration - Atomic Component
Dashboard and service integration layer
Agent Z - STEELCLAD Frontend Atomization
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict


class IntegrationType(Enum):
    """Types of service integrations"""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    GRAPHQL = "graphql"
    EVENT_STREAM = "event_stream"
    MESSAGE_QUEUE = "message_queue"


class ServiceType(Enum):
    """Types of services"""
    DATA_SERVICE = "data_service"
    AUTH_SERVICE = "auth_service"
    MONITORING_SERVICE = "monitoring_service"
    ANALYTICS_SERVICE = "analytics_service"
    NOTIFICATION_SERVICE = "notification_service"


@dataclass
class ServiceConnection:
    """Service connection configuration"""
    service_id: str
    service_type: ServiceType
    integration_type: IntegrationType
    endpoint: str
    active: bool = False
    last_connected: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'service_id': self.service_id,
            'service_type': self.service_type.value,
            'integration_type': self.integration_type.value,
            'endpoint': self.endpoint,
            'active': self.active,
            'last_connected': self.last_connected.isoformat() if self.last_connected else None
        }


class DashboardServiceIntegration:
    """
    Dashboard service integration component
    Manages integration between dashboard and backend services
    """
    
    def __init__(self):
        self.service_connections: Dict[str, ServiceConnection] = {}
        self.service_handlers: Dict[str, Callable] = {}
        self.integration_callbacks: Dict[IntegrationType, List[Callable]] = defaultdict(list)
        
        # Service data cache
        self.service_data_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 60  # seconds
        
        # Integration metrics
        self.integration_metrics = {
            'total_integrations': 0,
            'active_connections': 0,
            'data_exchanges': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'integration_errors': 0,
            'avg_integration_time': 0.0
        }
        
        # Service health tracking
        self.service_health: Dict[str, Dict[str, Any]] = {}
        
        # Event subscriptions
        self.event_subscriptions: Dict[str, Set[str]] = defaultdict(set)
    
    def register_service(self, service_id: str, service_type: ServiceType,
                        integration_type: IntegrationType, endpoint: str) -> bool:
        """
        Register a service for dashboard integration
        Main interface for service registration
        """
        if service_id in self.service_connections:
            return False
        
        connection = ServiceConnection(
            service_id=service_id,
            service_type=service_type,
            integration_type=integration_type,
            endpoint=endpoint
        )
        
        self.service_connections[service_id] = connection
        self.integration_metrics['total_integrations'] += 1
        
        # Initialize health tracking
        self.service_health[service_id] = {
            'status': 'registered',
            'last_check': datetime.now().isoformat(),
            'response_time_ms': 0,
            'error_count': 0
        }
        
        return True
    
    async def connect_service(self, service_id: str) -> bool:
        """Connect to a registered service"""
        if service_id not in self.service_connections:
            return False
        
        connection = self.service_connections[service_id]
        
        try:
            start_time = time.time()
            
            # Simulate connection based on integration type
            if connection.integration_type == IntegrationType.WEBSOCKET:
                await self._connect_websocket(connection)
            elif connection.integration_type == IntegrationType.REST_API:
                await self._connect_rest_api(connection)
            elif connection.integration_type == IntegrationType.EVENT_STREAM:
                await self._connect_event_stream(connection)
            else:
                await asyncio.sleep(0.01)  # Default connection simulation
            
            connection.active = True
            connection.last_connected = datetime.now()
            self.integration_metrics['active_connections'] += 1
            
            # Update health
            response_time = (time.time() - start_time) * 1000
            self.service_health[service_id].update({
                'status': 'connected',
                'last_check': datetime.now().isoformat(),
                'response_time_ms': response_time
            })
            
            # Update average integration time
            self._update_avg_integration_time(response_time)
            
            return True
            
        except Exception:
            self.integration_metrics['integration_errors'] += 1
            self.service_health[service_id]['error_count'] += 1
            return False
    
    async def _connect_websocket(self, connection: ServiceConnection):
        """Connect via WebSocket"""
        await asyncio.sleep(0.005)  # Simulate WebSocket connection
    
    async def _connect_rest_api(self, connection: ServiceConnection):
        """Connect via REST API"""
        await asyncio.sleep(0.01)  # Simulate REST connection
    
    async def _connect_event_stream(self, connection: ServiceConnection):
        """Connect via event stream"""
        await asyncio.sleep(0.008)  # Simulate event stream connection
    
    def disconnect_service(self, service_id: str) -> bool:
        """Disconnect from a service"""
        if service_id not in self.service_connections:
            return False
        
        connection = self.service_connections[service_id]
        
        if connection.active:
            connection.active = False
            self.integration_metrics['active_connections'] -= 1
            
            self.service_health[service_id]['status'] = 'disconnected'
        
        return True
    
    async def exchange_data(self, service_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Exchange data with a service"""
        if service_id not in self.service_connections:
            return {'error': 'Service not registered'}
        
        connection = self.service_connections[service_id]
        
        if not connection.active:
            return {'error': 'Service not connected'}
        
        # Check cache first
        cache_key = f"{service_id}:{hash(str(data))}"
        cached_result = self._get_cached_data(cache_key)
        
        if cached_result is not None:
            self.integration_metrics['cache_hits'] += 1
            return cached_result
        
        self.integration_metrics['cache_misses'] += 1
        
        try:
            start_time = time.time()
            
            # Process based on service type
            if service_id in self.service_handlers:
                handler = self.service_handlers[service_id]
                
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(data)
                else:
                    result = handler(data)
            else:
                # Default processing
                result = {
                    'service_id': service_id,
                    'processed': True,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }
            
            # Cache result
            self._cache_data(cache_key, result)
            
            # Update metrics
            integration_time = (time.time() - start_time) * 1000
            self._update_avg_integration_time(integration_time)
            self.integration_metrics['data_exchanges'] += 1
            
            return result
            
        except Exception as e:
            self.integration_metrics['integration_errors'] += 1
            return {'error': str(e)}
    
    def register_handler(self, service_id: str, handler: Callable):
        """Register a handler for service data processing"""
        self.service_handlers[service_id] = handler
    
    def subscribe_to_events(self, service_id: str, event_types: List[str]):
        """Subscribe to service events"""
        for event_type in event_types:
            self.event_subscriptions[event_type].add(service_id)
    
    async def broadcast_event(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast event to subscribed services"""
        subscribed_services = self.event_subscriptions.get(event_type, set())
        
        for service_id in subscribed_services:
            if service_id in self.service_connections:
                connection = self.service_connections[service_id]
                
                if connection.active:
                    # Process event for service
                    await self._process_event_for_service(service_id, event_type, event_data)
    
    async def _process_event_for_service(self, service_id: str, 
                                        event_type: str, event_data: Dict[str, Any]):
        """Process event for specific service"""
        # Notify integration callbacks
        for callback in self.integration_callbacks[IntegrationType.EVENT_STREAM]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(service_id, event_type, event_data)
                else:
                    callback(service_id, event_type, event_data)
            except Exception:
                pass
    
    def add_integration_callback(self, integration_type: IntegrationType, callback: Callable):
        """Add callback for integration events"""
        self.integration_callbacks[integration_type].append(callback)
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data if available"""
        if cache_key not in self.service_data_cache:
            return None
        
        # Check if cache expired
        if time.time() - self.cache_timestamps.get(cache_key, 0) > self.cache_ttl:
            del self.service_data_cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
        
        return self.service_data_cache[cache_key]
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache service data"""
        self.service_data_cache[cache_key] = data
        self.cache_timestamps[cache_key] = time.time()
    
    def _update_avg_integration_time(self, integration_time_ms: float):
        """Update average integration time"""
        current_avg = self.integration_metrics['avg_integration_time']
        exchanges = max(self.integration_metrics['data_exchanges'], 1)
        
        self.integration_metrics['avg_integration_time'] = (
            (current_avg * (exchanges - 1) + integration_time_ms) / exchanges
        )
    
    async def health_check_all_services(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        health_results = {}
        
        for service_id, connection in self.service_connections.items():
            if connection.active:
                health = await self._check_service_health(service_id)
                health_results[service_id] = health
        
        return health_results
    
    async def _check_service_health(self, service_id: str) -> Dict[str, Any]:
        """Check health of specific service"""
        start_time = time.time()
        
        try:
            # Simulate health check
            await asyncio.sleep(0.005)
            
            response_time = (time.time() - start_time) * 1000
            
            health = {
                'healthy': True,
                'response_time_ms': response_time,
                'status': 'operational',
                'last_check': datetime.now().isoformat()
            }
            
            self.service_health[service_id].update(health)
            
            return health
            
        except Exception:
            return {
                'healthy': False,
                'status': 'error',
                'last_check': datetime.now().isoformat()
            }
    
    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific service"""
        if service_id not in self.service_connections:
            return None
        
        connection = self.service_connections[service_id]
        health = self.service_health.get(service_id, {})
        
        return {
            **connection.to_dict(),
            'health': health,
            'handlers_registered': service_id in self.service_handlers,
            'event_subscriptions': [
                event for event, services in self.event_subscriptions.items()
                if service_id in services
            ]
        }
    
    def get_all_services(self) -> List[Dict[str, Any]]:
        """Get status of all services"""
        return [
            self.get_service_status(service_id)
            for service_id in self.service_connections.keys()
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        return {
            **self.integration_metrics,
            'cache_size': len(self.service_data_cache),
            'registered_services': len(self.service_connections),
            'registered_handlers': len(self.service_handlers),
            'event_subscriptions': sum(len(s) for s in self.event_subscriptions.values()),
            'latency_target_met': self.integration_metrics['avg_integration_time'] < 50
        }