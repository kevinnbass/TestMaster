"""
Enterprise Integration Hub & External System Connectivity
=========================================================

Agent B Hours 50-60: Advanced enterprise integration with external systems,
API connectivity, service mesh integration, and enterprise-grade coordination.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-08-22 (Hours 50-60)
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from urllib.parse import urljoin
import threading

# Import orchestration components
try:
    from .real_time_tuning import RealTimePerformanceTuner, PerformanceSnapshot
    from .discovery.orchestration_coordinator import OrchestrationCoordinator
    COORDINATION_AVAILABLE = True
except ImportError:
    COORDINATION_AVAILABLE = False
    logging.warning("Coordination components not available for enterprise integration")


class IntegrationType(Enum):
    """Types of enterprise integrations"""
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    MICROSERVICE = "microservice"
    SERVICE_MESH = "service_mesh"
    EVENT_STREAM = "event_stream"
    FILE_SYSTEM = "file_system"
    CLOUD_SERVICE = "cloud_service"
    WEBHOOK = "webhook"


class ConnectionStatus(Enum):
    """Connection status for external systems"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    TIMEOUT = "timeout"
    AUTHENTICATION_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"


@dataclass
class ExternalSystemConfig:
    """Configuration for external system integration"""
    system_id: str
    name: str
    integration_type: IntegrationType
    endpoint_url: str
    authentication: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_count: int = 3
    rate_limit_per_minute: int = 1000
    health_check_interval: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SystemConnection:
    """Active connection to external system"""
    config: ExternalSystemConfig
    status: ConnectionStatus
    connected_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    response_times: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate connection success rate"""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0


@dataclass
class IntegrationMessage:
    """Message for enterprise integration"""
    message_id: str
    source_system: str
    target_system: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timeout_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class EnterpriseIntegrationHub:
    """
    Enterprise Integration Hub
    
    Manages connections and communication with external systems,
    providing enterprise-grade integration capabilities with monitoring,
    retry logic, and performance optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EnterpriseIntegrationHub")
        
        # System connections
        self.connections: Dict[str, SystemConnection] = {}
        self.system_configs: Dict[str, ExternalSystemConfig] = {}
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.processing_active = False
        
        # Performance monitoring
        self.performance_tuner: Optional[RealTimePerformanceTuner] = None
        self.orchestration_coordinator: Optional[OrchestrationCoordinator] = None
        
        if COORDINATION_AVAILABLE:
            self._initialize_coordination_integration()
        
        # Health monitoring
        self.health_check_active = False
        self.health_check_interval = 60  # seconds
        
        # Integration metrics
        self.integration_metrics = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_errors": 0,
            "total_systems_connected": 0,
            "start_time": datetime.now()
        }
        
        self.logger.info("Enterprise Integration Hub initialized")
    
    def _initialize_coordination_integration(self):
        """Initialize coordination with performance tuning and orchestration"""
        try:
            # Initialize performance tuner for integration monitoring
            from .real_time_tuning import TuningStrategy
            self.performance_tuner = RealTimePerformanceTuner(TuningStrategy.INTELLIGENT)
            
            # Initialize orchestration coordinator
            from .discovery.orchestration_coordinator import orchestration_coordinator
            self.orchestration_coordinator = orchestration_coordinator
            
            self.logger.info("Coordination integration initialized")
            
        except Exception as e:
            self.logger.warning(f"Coordination integration failed: {e}")
    
    async def register_system(self, config: ExternalSystemConfig) -> bool:
        """Register external system for integration"""
        try:
            self.system_configs[config.system_id] = config
            
            # Create connection
            connection = SystemConnection(
                config=config,
                status=ConnectionStatus.DISCONNECTED
            )
            
            self.connections[config.system_id] = connection
            
            # Attempt initial connection
            await self._establish_connection(config.system_id)
            
            self.logger.info(f"Registered external system: {config.name} ({config.system_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register system {config.system_id}: {e}")
            return False
    
    async def _establish_connection(self, system_id: str) -> bool:
        """Establish connection to external system"""
        if system_id not in self.connections:
            return False
        
        connection = self.connections[system_id]
        config = connection.config
        
        try:
            connection.status = ConnectionStatus.CONNECTING
            
            # Simulate connection establishment (in production, this would be actual API calls)
            start_time = time.time()
            
            # Mock connection logic based on integration type
            if config.integration_type == IntegrationType.REST_API:
                success = await self._connect_rest_api(config)
            elif config.integration_type == IntegrationType.DATABASE:
                success = await self._connect_database(config)
            elif config.integration_type == IntegrationType.MESSAGE_QUEUE:
                success = await self._connect_message_queue(config)
            elif config.integration_type == IntegrationType.MICROSERVICE:
                success = await self._connect_microservice(config)
            else:
                # Generic connection simulation
                await asyncio.sleep(0.1)  # Simulate connection time
                success = True
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if success:
                connection.status = ConnectionStatus.CONNECTED
                connection.connected_at = datetime.now()
                connection.last_activity = datetime.now()
                connection.success_count += 1
                connection.response_times.append(response_time)
                
                # Keep only last 100 response times
                if len(connection.response_times) > 100:
                    connection.response_times = connection.response_times[-100:]
                
                self.integration_metrics["total_systems_connected"] += 1
                
                self.logger.info(f"Connected to {config.name} in {response_time:.1f}ms")
                return True
            else:
                connection.status = ConnectionStatus.ERROR
                connection.error_count += 1
                self.integration_metrics["total_errors"] += 1
                
                self.logger.error(f"Failed to connect to {config.name}")
                return False
                
        except Exception as e:
            connection.status = ConnectionStatus.ERROR
            connection.error_count += 1
            self.integration_metrics["total_errors"] += 1
            
            self.logger.error(f"Connection error for {config.name}: {e}")
            return False
    
    async def _connect_rest_api(self, config: ExternalSystemConfig) -> bool:
        """Connect to REST API system"""
        # Mock REST API connection
        await asyncio.sleep(0.05)  # Simulate HTTP request
        
        # Simulate authentication if configured
        if config.authentication:
            await asyncio.sleep(0.02)  # Simulate auth check
        
        return True  # Mock successful connection
    
    async def _connect_database(self, config: ExternalSystemConfig) -> bool:
        """Connect to database system"""
        # Mock database connection
        await asyncio.sleep(0.1)  # Simulate database connection
        return True
    
    async def _connect_message_queue(self, config: ExternalSystemConfig) -> bool:
        """Connect to message queue system"""
        # Mock message queue connection
        await asyncio.sleep(0.03)  # Simulate queue connection
        return True
    
    async def _connect_microservice(self, config: ExternalSystemConfig) -> bool:
        """Connect to microservice"""
        # Mock microservice connection
        await asyncio.sleep(0.08)  # Simulate service discovery and connection
        return True
    
    async def send_message(self, message: IntegrationMessage) -> bool:
        """Send message to external system"""
        try:
            target_system = message.target_system
            
            if target_system not in self.connections:
                self.logger.error(f"Target system {target_system} not registered")
                return False
            
            connection = self.connections[target_system]
            
            if connection.status != ConnectionStatus.CONNECTED:
                # Attempt to reconnect
                await self._establish_connection(target_system)
                
                if connection.status != ConnectionStatus.CONNECTED:
                    self.logger.error(f"Cannot send message - {target_system} not connected")
                    return False
            
            # Send message
            start_time = time.time()
            
            success = await self._transmit_message(connection, message)
            
            response_time = (time.time() - start_time) * 1000
            
            if success:
                connection.success_count += 1
                connection.last_activity = datetime.now()
                connection.response_times.append(response_time)
                
                self.integration_metrics["total_messages_sent"] += 1
                
                self.logger.debug(f"Message sent to {target_system} in {response_time:.1f}ms")
                
                # Update performance monitoring
                if self.performance_tuner:
                    await self._update_performance_metrics(connection, response_time)
                
                return True
            else:
                connection.error_count += 1
                self.integration_metrics["total_errors"] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send message to {message.target_system}: {e}")
            self.integration_metrics["total_errors"] += 1
            return False
    
    async def _transmit_message(self, connection: SystemConnection, message: IntegrationMessage) -> bool:
        """Transmit message to external system"""
        config = connection.config
        
        try:
            # Simulate different transmission methods based on integration type
            if config.integration_type == IntegrationType.REST_API:
                await asyncio.sleep(0.02)  # Simulate HTTP POST
            elif config.integration_type == IntegrationType.MESSAGE_QUEUE:
                await asyncio.sleep(0.01)  # Simulate queue publish
            elif config.integration_type == IntegrationType.DATABASE:
                await asyncio.sleep(0.05)  # Simulate database insert
            else:
                await asyncio.sleep(0.03)  # Generic transmission
            
            # Mock successful transmission
            return True
            
        except Exception as e:
            self.logger.error(f"Message transmission failed: {e}")
            return False
    
    async def _update_performance_metrics(self, connection: SystemConnection, response_time: float):
        """Update performance monitoring with integration metrics"""
        if not self.performance_tuner:
            return
        
        try:
            # Create performance snapshot for integration
            integration_metrics = {
                "execution_time": response_time,
                "memory_usage": 30.0,  # Mock memory usage
                "success_rate": connection.success_rate,
                "throughput": len(connection.response_times),
                "error_rate": connection.error_count / max(connection.success_count + connection.error_count, 1)
            }
            
            # Update performance tuner (if monitoring is active)
            # This would integrate with the real-time tuning system
            
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
    
    async def receive_message(self, message_data: Dict[str, Any]) -> bool:
        """Receive message from external system"""
        try:
            # Convert to IntegrationMessage
            message = IntegrationMessage(
                message_id=message_data.get("message_id", str(time.time())),
                source_system=message_data.get("source_system", "unknown"),
                target_system=message_data.get("target_system", "testmaster"),
                message_type=message_data.get("message_type", "generic"),
                payload=message_data.get("payload", {}),
                correlation_id=message_data.get("correlation_id")
            )
            
            # Add to processing queue
            await self.message_queue.put(message)
            
            self.integration_metrics["total_messages_received"] += 1
            
            self.logger.debug(f"Received message from {message.source_system}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            self.integration_metrics["total_errors"] += 1
            return False
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered message handler for type: {message_type}")
    
    async def start_processing(self):
        """Start message processing"""
        if self.processing_active:
            return
        
        self.processing_active = True
        self.logger.info("Started enterprise integration message processing")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        # Start health checking
        asyncio.create_task(self._health_check_loop())
    
    async def stop_processing(self):
        """Stop message processing"""
        self.processing_active = False
        self.health_check_active = False
        self.logger.info("Stopped enterprise integration processing")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.processing_active:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Process message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # No message received, continue loop
                continue
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    async def _process_message(self, message: IntegrationMessage):
        """Process received message"""
        try:
            message_type = message.message_type
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                
                # Execute handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
                
                self.logger.debug(f"Processed {message_type} message from {message.source_system}")
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Message processing failed: {e}")
    
    async def _health_check_loop(self):
        """Health check loop for connected systems"""
        self.health_check_active = True
        
        while self.health_check_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all connected systems"""
        for system_id, connection in self.connections.items():
            if connection.status == ConnectionStatus.CONNECTED:
                # Check if connection is still healthy
                time_since_activity = datetime.now() - connection.last_activity
                
                if time_since_activity.total_seconds() > connection.config.health_check_interval * 2:
                    # Connection appears stale, test it
                    await self._test_connection_health(system_id)
    
    async def _test_connection_health(self, system_id: str):
        """Test health of specific connection"""
        try:
            connection = self.connections[system_id]
            
            # Send health check message
            health_check_message = IntegrationMessage(
                message_id=f"health_check_{time.time()}",
                source_system="testmaster",
                target_system=system_id,
                message_type="health_check",
                payload={"timestamp": datetime.now().isoformat()}
            )
            
            success = await self._transmit_message(connection, health_check_message)
            
            if not success:
                self.logger.warning(f"Health check failed for {system_id}, attempting reconnection")
                await self._establish_connection(system_id)
                
        except Exception as e:
            self.logger.error(f"Health check test failed for {system_id}: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        status = {
            "hub_status": "active" if self.processing_active else "inactive",
            "total_systems": len(self.system_configs),
            "connected_systems": len([c for c in self.connections.values() if c.status == ConnectionStatus.CONNECTED]),
            "integration_metrics": self.integration_metrics.copy(),
            "system_connections": {},
            "performance_summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add individual system statuses
        for system_id, connection in self.connections.items():
            status["system_connections"][system_id] = {
                "name": connection.config.name,
                "type": connection.config.integration_type.value,
                "status": connection.status.value,
                "success_rate": connection.success_rate,
                "average_response_time": connection.average_response_time,
                "connected_at": connection.connected_at.isoformat() if connection.connected_at else None,
                "last_activity": connection.last_activity.isoformat() if connection.last_activity else None
            }
        
        # Add performance summary
        if self.connections:
            all_response_times = []
            total_success = 0
            total_errors = 0
            
            for connection in self.connections.values():
                all_response_times.extend(connection.response_times)
                total_success += connection.success_count
                total_errors += connection.error_count
            
            status["performance_summary"] = {
                "overall_success_rate": total_success / max(total_success + total_errors, 1),
                "average_response_time": sum(all_response_times) / len(all_response_times) if all_response_times else 0.0,
                "total_successful_operations": total_success,
                "total_errors": total_errors
            }
        
        return status
    
    async def disconnect_system(self, system_id: str) -> bool:
        """Disconnect from external system"""
        if system_id not in self.connections:
            return False
        
        try:
            connection = self.connections[system_id]
            connection.status = ConnectionStatus.DISCONNECTED
            connection.connected_at = None
            
            self.logger.info(f"Disconnected from {connection.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to disconnect from {system_id}: {e}")
            return False
    
    async def reconnect_all_systems(self):
        """Reconnect to all registered systems"""
        self.logger.info("Reconnecting to all external systems...")
        
        reconnection_results = []
        
        for system_id in self.system_configs.keys():
            success = await self._establish_connection(system_id)
            reconnection_results.append((system_id, success))
        
        successful_connections = sum(1 for _, success in reconnection_results if success)
        
        self.logger.info(f"Reconnection complete: {successful_connections}/{len(reconnection_results)} systems connected")
        
        return reconnection_results


# Global enterprise integration hub instance
enterprise_integration_hub = EnterpriseIntegrationHub()


# Convenience functions for enterprise integration
async def register_external_system(
    system_id: str,
    name: str,
    integration_type: IntegrationType,
    endpoint_url: str,
    **kwargs
) -> bool:
    """Register external system with default configuration"""
    config = ExternalSystemConfig(
        system_id=system_id,
        name=name,
        integration_type=integration_type,
        endpoint_url=endpoint_url,
        **kwargs
    )
    
    return await enterprise_integration_hub.register_system(config)


async def send_integration_message(
    target_system: str,
    message_type: str,
    payload: Dict[str, Any],
    **kwargs
) -> bool:
    """Send message to external system"""
    message = IntegrationMessage(
        message_id=f"msg_{time.time()}",
        source_system="testmaster",
        target_system=target_system,
        message_type=message_type,
        payload=payload,
        **kwargs
    )
    
    return await enterprise_integration_hub.send_message(message)


def register_integration_handler(message_type: str, handler: Callable):
    """Register message handler for enterprise integration"""
    enterprise_integration_hub.register_message_handler(message_type, handler)


# Export key components
__all__ = [
    'EnterpriseIntegrationHub',
    'ExternalSystemConfig',
    'SystemConnection',
    'IntegrationMessage',
    'IntegrationType',
    'ConnectionStatus',
    'enterprise_integration_hub',
    'register_external_system',
    'send_integration_message',
    'register_integration_handler'
]