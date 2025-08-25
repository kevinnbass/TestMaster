"""
Layer Management System

Manages system architecture layers and their interactions, providing
clear separation of concerns and controlled communication between layers.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
from pathlib import Path


class SystemLayer(Enum):
    """System architecture layers."""
    PRESENTATION = "presentation"           # UI/API endpoints
    APPLICATION = "application"            # Business logic coordination  
    BUSINESS = "business"                  # Core business logic
    INTEGRATION = "integration"           # External system integration
    DATA_ACCESS = "data_access"           # Data persistence layer
    INFRASTRUCTURE = "infrastructure"      # System infrastructure


class LayerStatus(Enum):
    """Layer operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class CommunicationPattern(Enum):
    """Communication patterns between layers."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    EVENT_DRIVEN = "event_driven"
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"


@dataclass
class LayerConfiguration:
    """Configuration for a system layer."""
    layer_id: str
    layer_type: SystemLayer
    description: str
    dependencies: List[str] = field(default_factory=list)  # Layer IDs this depends on
    allowed_communications: Dict[str, CommunicationPattern] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    initialization_order: int = 0
    health_check_interval: int = 30  # seconds
    enable_monitoring: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type.value,
            "description": self.description,
            "dependencies": self.dependencies,
            "allowed_communications": {k: v.value for k, v in self.allowed_communications.items()},
            "resource_limits": self.resource_limits,
            "initialization_order": self.initialization_order,
            "health_check_interval": self.health_check_interval,
            "enable_monitoring": self.enable_monitoring
        }


@dataclass
class LayerInstance:
    """Runtime instance of a system layer."""
    config: LayerConfiguration
    status: LayerStatus = LayerStatus.INACTIVE
    instance: Optional[Any] = None
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    error_count: int = 0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "performance_metrics": self.performance_metrics
        }


@dataclass
class LayerCommunication:
    """Represents communication between layers."""
    from_layer: str
    to_layer: str
    pattern: CommunicationPattern
    message_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0


class LayerManager:
    """
    Manages system layers and their interactions.
    
    Responsibilities:
    - Layer lifecycle management (initialization, startup, shutdown)
    - Communication routing and validation
    - Dependency resolution and ordering
    - Health monitoring and error recovery
    - Performance tracking and optimization
    - Security and access control between layers
    """
    
    def __init__(self, enable_strict_mode: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_strict_mode = enable_strict_mode  # Enforce communication rules
        
        # Layer management
        self.layer_configs: Dict[str, LayerConfiguration] = {}
        self.layer_instances: Dict[str, LayerInstance] = {}
        self.layer_factories: Dict[str, Callable] = {}
        
        # Communication tracking
        self.communication_history: List[LayerCommunication] = []
        self.communication_handlers: Dict[Tuple[str, str], Callable] = {}
        
        # System state
        self.is_running = False
        self.initialization_order: List[str] = []
        self.health_monitor_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.system_metrics = {
            "total_communications": 0,
            "failed_communications": 0,
            "average_response_time": 0.0,
            "layer_uptime": {},
            "error_rates": {}
        }
        
        # Initialize standard layers
        self._initialize_standard_layers()
    
    def _initialize_standard_layers(self):
        """Initialize standard system layers."""
        # Presentation Layer
        self.register_layer(LayerConfiguration(
            layer_id="presentation",
            layer_type=SystemLayer.PRESENTATION,
            description="User interface and API endpoints",
            dependencies=[],
            allowed_communications={
                "application": CommunicationPattern.REQUEST_RESPONSE,
                "integration": CommunicationPattern.ASYNCHRONOUS
            },
            initialization_order=1
        ))
        
        # Application Layer  
        self.register_layer(LayerConfiguration(
            layer_id="application",
            layer_type=SystemLayer.APPLICATION,
            description="Application logic coordination",
            dependencies=["business", "integration"],
            allowed_communications={
                "business": CommunicationPattern.SYNCHRONOUS,
                "integration": CommunicationPattern.EVENT_DRIVEN,
                "data_access": CommunicationPattern.REQUEST_RESPONSE
            },
            initialization_order=2
        ))
        
        # Business Layer
        self.register_layer(LayerConfiguration(
            layer_id="business",
            layer_type=SystemLayer.BUSINESS,
            description="Core business logic and rules",
            dependencies=["data_access"],
            allowed_communications={
                "data_access": CommunicationPattern.SYNCHRONOUS,
                "integration": CommunicationPattern.EVENT_DRIVEN
            },
            initialization_order=4
        ))
        
        # Integration Layer
        self.register_layer(LayerConfiguration(
            layer_id="integration",
            layer_type=SystemLayer.INTEGRATION,
            description="External system integration",
            dependencies=["infrastructure"],
            allowed_communications={
                "infrastructure": CommunicationPattern.ASYNCHRONOUS,
                "data_access": CommunicationPattern.REQUEST_RESPONSE
            },
            initialization_order=3
        ))
        
        # Data Access Layer
        self.register_layer(LayerConfiguration(
            layer_id="data_access",
            layer_type=SystemLayer.DATA_ACCESS,
            description="Data persistence and retrieval",
            dependencies=["infrastructure"],
            allowed_communications={
                "infrastructure": CommunicationPattern.SYNCHRONOUS
            },
            initialization_order=5
        ))
        
        # Infrastructure Layer
        self.register_layer(LayerConfiguration(
            layer_id="infrastructure",
            layer_type=SystemLayer.INFRASTRUCTURE,
            description="System infrastructure and utilities",
            dependencies=[],
            allowed_communications={},
            initialization_order=6
        ))
    
    def register_layer(self, config: LayerConfiguration):
        """Register a new layer configuration."""
        self.layer_configs[config.layer_id] = config
        self.layer_instances[config.layer_id] = LayerInstance(config=config)
        self.logger.info(f"Registered layer: {config.layer_id}")
    
    def register_layer_factory(self, layer_id: str, factory: Callable):
        """Register a factory function for creating layer instances."""
        self.layer_factories[layer_id] = factory
        self.logger.info(f"Registered factory for layer: {layer_id}")
    
    def register_communication_handler(self, 
                                       from_layer: str, 
                                       to_layer: str, 
                                       handler: Callable):
        """Register a handler for communication between specific layers."""
        key = (from_layer, to_layer)
        self.communication_handlers[key] = handler
        self.logger.info(f"Registered communication handler: {from_layer} -> {to_layer}")
    
    async def start_layers(self):
        """Start all layers in dependency order."""
        if self.is_running:
            self.logger.warning("Layers are already running")
            return
        
        # Calculate initialization order
        self._calculate_initialization_order()
        
        # Initialize layers in order
        for layer_id in self.initialization_order:
            await self._initialize_layer(layer_id)
        
        # Start health monitoring
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        self.is_running = True
        self.logger.info("All layers started successfully")
    
    async def stop_layers(self):
        """Stop all layers in reverse dependency order."""
        if not self.is_running:
            return
        
        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop layers in reverse order
        for layer_id in reversed(self.initialization_order):
            await self._shutdown_layer(layer_id)
        
        self.is_running = False
        self.logger.info("All layers stopped")
    
    def _calculate_initialization_order(self):
        """Calculate the order to initialize layers based on dependencies."""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(layer_id: str):
            if layer_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {layer_id}")
            if layer_id in visited:
                return
            
            temp_visited.add(layer_id)
            
            config = self.layer_configs[layer_id]
            for dep in config.dependencies:
                if dep in self.layer_configs:
                    visit(dep)
            
            temp_visited.remove(layer_id)
            visited.add(layer_id)
            order.append(layer_id)
        
        for layer_id in self.layer_configs:
            if layer_id not in visited:
                visit(layer_id)
        
        # Sort by explicit initialization order if provided
        order.sort(key=lambda lid: self.layer_configs[lid].initialization_order)
        
        self.initialization_order = order
        self.logger.info(f"Layer initialization order: {order}")
    
    async def _initialize_layer(self, layer_id: str):
        """Initialize a specific layer."""
        instance = self.layer_instances[layer_id]
        
        try:
            self.logger.info(f"Initializing layer: {layer_id}")
            instance.status = LayerStatus.INITIALIZING
            
            # Create layer instance if factory is available
            if layer_id in self.layer_factories:
                factory = self.layer_factories[layer_id]
                instance.instance = await factory() if asyncio.iscoroutinefunction(factory) else factory()
            
            # Start the layer if it has a start method
            if instance.instance and hasattr(instance.instance, 'start'):
                start_method = instance.instance.start
                if asyncio.iscoroutinefunction(start_method):
                    await start_method()
                else:
                    start_method()
            
            instance.status = LayerStatus.ACTIVE
            instance.started_at = datetime.now()
            instance.health_status = "healthy"
            
            self.logger.info(f"Layer {layer_id} initialized successfully")
            
        except Exception as e:
            instance.status = LayerStatus.ERROR
            instance.last_error = str(e)
            instance.error_count += 1
            self.logger.error(f"Failed to initialize layer {layer_id}: {e}")
            raise
    
    async def _shutdown_layer(self, layer_id: str):
        """Shutdown a specific layer."""
        instance = self.layer_instances[layer_id]
        
        try:
            self.logger.info(f"Shutting down layer: {layer_id}")
            
            # Stop the layer if it has a stop method
            if instance.instance and hasattr(instance.instance, 'stop'):
                stop_method = instance.instance.stop
                if asyncio.iscoroutinefunction(stop_method):
                    await stop_method()
                else:
                    stop_method()
            
            instance.status = LayerStatus.INACTIVE
            instance.instance = None
            
            self.logger.info(f"Layer {layer_id} shut down successfully")
            
        except Exception as e:
            instance.status = LayerStatus.ERROR
            instance.last_error = str(e)
            instance.error_count += 1
            self.logger.error(f"Failed to shutdown layer {layer_id}: {e}")
    
    async def communicate(self, 
                         from_layer: str, 
                         to_layer: str, 
                         message_type: str, 
                         data: Any = None,
                         pattern: Optional[CommunicationPattern] = None) -> Any:
        """Send a message from one layer to another."""
        start_time = datetime.now()
        
        # Validate communication
        if self.enable_strict_mode:
            self._validate_communication(from_layer, to_layer, pattern)
        
        # Create communication record
        comm = LayerCommunication(
            from_layer=from_layer,
            to_layer=to_layer,
            pattern=pattern or CommunicationPattern.REQUEST_RESPONSE,
            message_type=message_type,
            data=data
        )
        
        try:
            # Get target layer instance
            target_instance = self.layer_instances[to_layer]
            if target_instance.status != LayerStatus.ACTIVE:
                raise RuntimeError(f"Target layer {to_layer} is not active")
            
            # Check for custom handler
            handler_key = (from_layer, to_layer)
            if handler_key in self.communication_handlers:
                handler = self.communication_handlers[handler_key]
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(message_type, data)
                else:
                    result = handler(message_type, data)
            else:
                # Default handling - call method on target instance
                result = await self._default_communication_handler(
                    target_instance, message_type, data
                )
            
            # Record success
            comm.success = True
            comm.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            self.system_metrics["total_communications"] += 1
            
            return result
            
        except Exception as e:
            # Record failure
            comm.success = False
            comm.error = str(e)
            comm.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            self.system_metrics["total_communications"] += 1
            self.system_metrics["failed_communications"] += 1
            
            self.logger.error(f"Communication failed {from_layer} -> {to_layer}: {e}")
            raise
        
        finally:
            # Store communication record
            self.communication_history.append(comm)
            
            # Limit history size
            if len(self.communication_history) > 10000:
                self.communication_history = self.communication_history[-5000:]
    
    def _validate_communication(self, 
                                from_layer: str, 
                                to_layer: str, 
                                pattern: Optional[CommunicationPattern]):
        """Validate that communication is allowed between layers."""
        if from_layer not in self.layer_configs:
            raise ValueError(f"Unknown source layer: {from_layer}")
        
        if to_layer not in self.layer_configs:
            raise ValueError(f"Unknown target layer: {to_layer}")
        
        from_config = self.layer_configs[from_layer]
        
        # Check if communication is allowed
        if to_layer not in from_config.allowed_communications:
            raise ValueError(f"Communication not allowed: {from_layer} -> {to_layer}")
        
        # Check communication pattern
        if pattern:
            allowed_pattern = from_config.allowed_communications[to_layer]
            if pattern != allowed_pattern:
                raise ValueError(
                    f"Communication pattern mismatch: expected {allowed_pattern}, got {pattern}"
                )
    
    async def _default_communication_handler(self, 
                                             target_instance: LayerInstance, 
                                             message_type: str, 
                                             data: Any) -> Any:
        """Default handler for layer communication."""
        if not target_instance.instance:
            raise RuntimeError("Target layer instance is not available")
        
        # Try to find a handler method
        handler_method_name = f"handle_{message_type}"
        if hasattr(target_instance.instance, handler_method_name):
            handler = getattr(target_instance.instance, handler_method_name)
            if asyncio.iscoroutinefunction(handler):
                return await handler(data)
            else:
                return handler(data)
        
        # Try generic handler
        if hasattr(target_instance.instance, 'handle_message'):
            handler = target_instance.instance.handle_message
            if asyncio.iscoroutinefunction(handler):
                return await handler(message_type, data)
            else:
                return handler(message_type, data)
        
        raise NotImplementedError(f"No handler for message type: {message_type}")
    
    async def _health_monitor_loop(self):
        """Monitor layer health continuously."""
        while self.is_running:
            try:
                for layer_id, instance in self.layer_instances.items():
                    if instance.status == LayerStatus.ACTIVE:
                        await self._check_layer_health(layer_id)
                
                # Update system metrics
                self._update_system_metrics()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)  # Longer pause on error
    
    async def _check_layer_health(self, layer_id: str):
        """Check the health of a specific layer."""
        instance = self.layer_instances[layer_id]
        
        try:
            # Check if layer has health check method
            if instance.instance and hasattr(instance.instance, 'health_check'):
                health_check = instance.instance.health_check
                if asyncio.iscoroutinefunction(health_check):
                    health_status = await health_check()
                else:
                    health_status = health_check()
                
                instance.health_status = str(health_status)
            else:
                # Basic health check - just verify instance exists
                instance.health_status = "healthy" if instance.instance else "no_instance"
            
            instance.last_health_check = datetime.now()
            
        except Exception as e:
            instance.health_status = f"error: {str(e)}"
            instance.error_count += 1
            instance.last_error = str(e)
            self.logger.warning(f"Health check failed for layer {layer_id}: {e}")
    
    def _update_system_metrics(self):
        """Update overall system metrics."""
        # Calculate average response time
        if self.communication_history:
            recent_comms = self.communication_history[-1000:]  # Last 1000 communications
            successful_comms = [c for c in recent_comms if c.success]
            
            if successful_comms:
                avg_time = sum(c.duration_ms for c in successful_comms) / len(successful_comms)
                self.system_metrics["average_response_time"] = avg_time
        
        # Calculate layer uptime
        current_time = datetime.now()
        for layer_id, instance in self.layer_instances.items():
            if instance.started_at and instance.status == LayerStatus.ACTIVE:
                uptime = (current_time - instance.started_at).total_seconds()
                self.system_metrics["layer_uptime"][layer_id] = uptime
        
        # Calculate error rates
        for layer_id, instance in self.layer_instances.items():
            if instance.started_at:
                runtime_hours = (current_time - instance.started_at).total_seconds() / 3600
                error_rate = instance.error_count / max(runtime_hours, 1)
                self.system_metrics["error_rates"][layer_id] = error_rate
    
    def get_layer_status(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific layer."""
        if layer_id not in self.layer_instances:
            return None
        
        return self.layer_instances[layer_id].to_dict()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        layer_statuses = {}
        for layer_id, instance in self.layer_instances.items():
            layer_statuses[layer_id] = instance.to_dict()
        
        return {
            "is_running": self.is_running,
            "total_layers": len(self.layer_instances),
            "active_layers": sum(1 for i in self.layer_instances.values() if i.status == LayerStatus.ACTIVE),
            "initialization_order": self.initialization_order,
            "system_metrics": self.system_metrics.copy(),
            "layer_statuses": layer_statuses,
            "communication_stats": {
                "total_communications": len(self.communication_history),
                "recent_success_rate": self._calculate_recent_success_rate()
            }
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate for recent communications."""
        if not self.communication_history:
            return 1.0
        
        recent_comms = self.communication_history[-100:]  # Last 100 communications
        successful = sum(1 for c in recent_comms if c.success)
        return successful / len(recent_comms)
    
    def get_communication_history(self, 
                                  from_layer: Optional[str] = None,
                                  to_layer: Optional[str] = None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get communication history with optional filtering."""
        filtered_history = self.communication_history
        
        if from_layer:
            filtered_history = [c for c in filtered_history if c.from_layer == from_layer]
        
        if to_layer:
            filtered_history = [c for c in filtered_history if c.to_layer == to_layer]
        
        # Return most recent entries
        recent_history = filtered_history[-limit:] if limit else filtered_history
        
        return [
            {
                "from_layer": c.from_layer,
                "to_layer": c.to_layer,
                "pattern": c.pattern.value,
                "message_type": c.message_type,
                "timestamp": c.timestamp.isoformat(),
                "success": c.success,
                "error": c.error,
                "duration_ms": c.duration_ms
            }
            for c in recent_history
        ]
    
    async def restart_layer(self, layer_id: str):
        """Restart a specific layer."""
        if layer_id not in self.layer_instances:
            raise ValueError(f"Unknown layer: {layer_id}")
        
        self.logger.info(f"Restarting layer: {layer_id}")
        
        # Shutdown first
        await self._shutdown_layer(layer_id)
        
        # Wait briefly
        await asyncio.sleep(1.0)
        
        # Restart
        await self._initialize_layer(layer_id)
        
        self.logger.info(f"Layer {layer_id} restarted successfully")


# Factory function
def create_layer_manager(enable_strict_mode: bool = True) -> LayerManager:
    """Create a layer manager with default configuration."""
    return LayerManager(enable_strict_mode=enable_strict_mode)