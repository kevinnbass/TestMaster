"""
Cross-System Integration APIs
============================

Unified API layer enabling seamless communication between all consolidated systems.
Provides standardized interfaces for cross-system workflows and data exchange.

Built on Phase 1A consolidated foundation:
- Observability: unified_monitor.py
- State/Config: unified_state_manager.py + yaml_config_enhancer.py  
- Orchestration: unified_orchestrator.py + swarm_router_enhancement.py
- UI/Dashboard: unified_dashboard.py + nocode_enhancement.py

Author: TestMaster Phase 1B Integration System
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
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
import threading


# ============================================================================
# CORE INTEGRATION TYPES
# ============================================================================

class SystemType(Enum):
    """Unified system types"""
    OBSERVABILITY = "observability"
    STATE_CONFIG = "state_config"
    ORCHESTRATION = "orchestration"
    UI_DASHBOARD = "ui_dashboard"


class IntegrationEventType(Enum):
    """Cross-system integration event types"""
    SYSTEM_STATE_CHANGE = "system_state_change"
    WORKFLOW_STEP_COMPLETE = "workflow_step_complete"
    PERFORMANCE_ALERT = "performance_alert"
    CONFIGURATION_UPDATE = "configuration_update"
    UI_ACTION_TRIGGERED = "ui_action_triggered"
    ORCHESTRATION_DECISION = "orchestration_decision"
    ANALYTICS_INSIGHT = "analytics_insight"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class SystemMessage:
    """Standardized message format for cross-system communication"""
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    source_system: SystemType = SystemType.OBSERVABILITY
    target_system: Optional[SystemType] = None
    event_type: IntegrationEventType = IntegrationEventType.SYSTEM_STATE_CHANGE
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=high, 10=critical
    
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
            "priority": self.priority
        }


@dataclass
class CrossSystemRequest:
    """Request for cross-system operation"""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    operation: str = ""
    target_system: SystemType = SystemType.OBSERVABILITY
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    correlation_id: Optional[str] = None
    callback: Optional[Callable] = None
    
    def to_message(self) -> SystemMessage:
        """Convert to system message"""
        return SystemMessage(
            message_id=self.request_id,
            target_system=self.target_system,
            event_type=IntegrationEventType.UI_ACTION_TRIGGERED,
            payload={
                "operation": self.operation,
                "parameters": self.parameters,
                "timeout_seconds": self.timeout_seconds
            },
            correlation_id=self.correlation_id,
            priority=5
        )


@dataclass
class CrossSystemResponse:
    """Response from cross-system operation"""
    request_id: str
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# ABSTRACT SYSTEM INTERFACE
# ============================================================================

class UnifiedSystemInterface(ABC):
    """Abstract interface that all unified systems must implement"""
    
    @abstractmethod
    def get_system_type(self) -> SystemType:
        """Get the system type identifier"""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health"""
        pass
    
    @abstractmethod
    def handle_cross_system_message(self, message: SystemMessage) -> CrossSystemResponse:
        """Handle incoming cross-system message"""
        pass
    
    @abstractmethod
    def get_available_operations(self) -> List[str]:
        """Get list of available cross-system operations"""
        pass
    
    @abstractmethod
    def validate_operation_parameters(self, operation: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for a given operation"""
        pass


# ============================================================================
# CROSS-SYSTEM MESSAGE BUS
# ============================================================================

class CrossSystemMessageBus:
    """High-performance message bus for cross-system communication"""
    
    def __init__(self):
        self.logger = logging.getLogger("cross_system_message_bus")
        
        # System registry
        self.registered_systems: Dict[SystemType, UnifiedSystemInterface] = {}
        
        # Message routing
        self.message_handlers: Dict[IntegrationEventType, List[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_requests: Dict[str, CrossSystemRequest] = {}
        
        # Performance tracking
        self.message_stats = {
            "total_messages": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "average_delivery_time_ms": 0.0
        }
        
        # Event loops and workers
        self.message_processor_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        self.logger.info("Cross-system message bus initialized")
    
    def register_system(self, system: UnifiedSystemInterface) -> bool:
        """Register a unified system with the message bus"""
        try:
            system_type = system.get_system_type()
            
            if system_type in self.registered_systems:
                self.logger.warning(f"System {system_type.value} already registered, replacing")
            
            self.registered_systems[system_type] = system
            self.logger.info(f"Registered system: {system_type.value}")
            
            # Validate system interface
            status = system.get_system_status()
            operations = system.get_available_operations()
            
            self.logger.info(f"System {system_type.value} status: {status.get('status', 'unknown')}")
            self.logger.info(f"System {system_type.value} operations: {len(operations)} available")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register system: {e}")
            return False
    
    def register_event_handler(self, event_type: IntegrationEventType, handler: Callable):
        """Register handler for specific event type"""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        
        self.message_handlers[event_type].append(handler)
        self.logger.info(f"Registered handler for {event_type.value}")
    
    async def send_message(self, message: SystemMessage) -> bool:
        """Send message through the bus"""
        try:
            await self.message_queue.put(message)
            self.message_stats["total_messages"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue message {message.message_id}: {e}")
            return False
    
    async def send_request(self, request: CrossSystemRequest) -> CrossSystemResponse:
        """Send request and wait for response"""
        try:
            # Store request for correlation
            self.pending_requests[request.request_id] = request
            
            # Convert to message and send
            message = request.to_message()
            await self.send_message(message)
            
            # Wait for response with timeout
            start_time = time.time()
            timeout_time = start_time + request.timeout_seconds
            
            while time.time() < timeout_time:
                # Check if we have a response (this would be updated by message processor)
                # For now, simulate processing
                await asyncio.sleep(0.1)
                
                # Simulate response after short delay
                if time.time() - start_time > 0.5:  # Simulate 500ms processing
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Clean up
                    self.pending_requests.pop(request.request_id, None)
                    
                    return CrossSystemResponse(
                        request_id=request.request_id,
                        success=True,
                        result={"status": "processed", "operation": request.operation},
                        execution_time_ms=execution_time
                    )
            
            # Timeout
            self.pending_requests.pop(request.request_id, None)
            return CrossSystemResponse(
                request_id=request.request_id,
                success=False,
                error_message="Request timeout",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            self.pending_requests.pop(request.request_id, None)
            
            return CrossSystemResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def start_message_processor(self):
        """Start the message processing loop"""
        self.logger.info("Starting message processor")
        
        while not self.shutdown_event.is_set():
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # Normal timeout, continue
                continue
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info("Message processor stopped")
    
    async def _process_message(self, message: SystemMessage):
        """Process a single message"""
        start_time = time.time()
        
        try:
            # Route to target system if specified
            if message.target_system and message.target_system in self.registered_systems:
                target_system = self.registered_systems[message.target_system]
                response = target_system.handle_cross_system_message(message)
                
                if response.success:
                    self.message_stats["successful_deliveries"] += 1
                else:
                    self.message_stats["failed_deliveries"] += 1
                    self.logger.warning(f"Message delivery failed: {response.error_message}")
            
            # Route to event handlers
            if message.event_type in self.message_handlers:
                for handler in self.message_handlers[message.event_type]:
                    try:
                        await handler(message)
                    except Exception as e:
                        self.logger.error(f"Event handler failed: {e}")
            
            # Update performance stats
            delivery_time = (time.time() - start_time) * 1000
            self._update_delivery_stats(delivery_time)
            
        except Exception as e:
            self.logger.error(f"Failed to process message {message.message_id}: {e}")
            self.message_stats["failed_deliveries"] += 1
    
    def _update_delivery_stats(self, delivery_time_ms: float):
        """Update delivery time statistics"""
        current_avg = self.message_stats["average_delivery_time_ms"]
        total_deliveries = (self.message_stats["successful_deliveries"] + 
                          self.message_stats["failed_deliveries"])
        
        if total_deliveries > 0:
            self.message_stats["average_delivery_time_ms"] = (
                (current_avg * (total_deliveries - 1) + delivery_time_ms) / total_deliveries
            )
    
    def get_bus_statistics(self) -> Dict[str, Any]:
        """Get message bus performance statistics"""
        return {
            "registered_systems": list(self.registered_systems.keys()),
            "message_stats": self.message_stats.copy(),
            "pending_requests": len(self.pending_requests),
            "event_handlers": {
                event_type.value: len(handlers) 
                for event_type, handlers in self.message_handlers.items()
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the message bus"""
        self.logger.info("Shutting down message bus")
        self.shutdown_event.set()
        
        if self.message_processor_task:
            await self.message_processor_task


# ============================================================================
# CROSS-SYSTEM API COORDINATOR
# ============================================================================

class CrossSystemAPICoordinator:
    """
    Main coordinator for cross-system API operations.
    Provides high-level interface for cross-system workflows.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("cross_system_api_coordinator")
        self.message_bus = CrossSystemMessageBus()
        
        # System discovery and integration
        self.system_adapters: Dict[SystemType, Any] = {}
        self.integration_health = {
            "overall_status": "initializing",
            "system_statuses": {},
            "last_health_check": None
        }
        
        # Workflow coordination
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Cross-system API coordinator initialized")
    
    async def initialize_integration(self) -> bool:
        """Initialize cross-system integration"""
        try:
            self.logger.info("Initializing cross-system integration")
            
            # Start message bus
            self.message_bus.message_processor_task = asyncio.create_task(
                self.message_bus.start_message_processor()
            )
            
            # Discover and integrate systems
            await self._discover_and_integrate_systems()
            
            # Setup health monitoring
            await self._setup_health_monitoring()
            
            self.integration_health["overall_status"] = "operational"
            self.integration_health["last_health_check"] = datetime.now()
            
            self.logger.info("Cross-system integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration: {e}")
            self.integration_health["overall_status"] = "failed"
            return False
    
    async def _discover_and_integrate_systems(self):
        """Discover and integrate all unified systems"""
        # Import and integrate unified systems
        
        # Note: In production, these would be dynamic imports
        # For now, we'll create placeholder adapters
        
        system_configs = {
            SystemType.OBSERVABILITY: {
                "module": "observability.unified_monitor",
                "class": "UnifiedObservabilitySystem"
            },
            SystemType.STATE_CONFIG: {
                "module": "state.unified_state_manager", 
                "class": "UnifiedStateManager"
            },
            SystemType.ORCHESTRATION: {
                "module": "orchestration.unified_orchestrator",
                "class": "UnifiedOrchestrator"
            },
            SystemType.UI_DASHBOARD: {
                "module": "ui.unified_dashboard",
                "class": "UnifiedDashboard"
            }
        }
        
        for system_type, config in system_configs.items():
            try:
                # Create system adapter
                adapter = self._create_system_adapter(system_type, config)
                
                if adapter:
                    # Register with message bus
                    success = self.message_bus.register_system(adapter)
                    
                    if success:
                        self.system_adapters[system_type] = adapter
                        self.integration_health["system_statuses"][system_type.value] = "integrated"
                        self.logger.info(f"Successfully integrated {system_type.value}")
                    else:
                        self.integration_health["system_statuses"][system_type.value] = "registration_failed"
                        
            except Exception as e:
                self.logger.error(f"Failed to integrate {system_type.value}: {e}")
                self.integration_health["system_statuses"][system_type.value] = "integration_failed"
    
    def _create_system_adapter(self, system_type: SystemType, config: Dict[str, str]) -> Optional[UnifiedSystemInterface]:
        """Create adapter for a unified system"""
        
        # For Phase 1B, create mock adapters that implement the interface
        # In production, these would dynamically load the actual systems
        
        class MockSystemAdapter(UnifiedSystemInterface):
            def __init__(self, sys_type: SystemType):
                self.sys_type = sys_type
                self.status = "operational"
            
            def get_system_type(self) -> SystemType:
                return self.sys_type
            
            def get_system_status(self) -> Dict[str, Any]:
                return {
                    "status": self.status,
                    "system_type": self.sys_type.value,
                    "last_updated": datetime.now().isoformat(),
                    "operations_available": len(self.get_available_operations())
                }
            
            def handle_cross_system_message(self, message: SystemMessage) -> CrossSystemResponse:
                # Simulate message processing
                return CrossSystemResponse(
                    request_id=message.message_id,
                    success=True,
                    result={"processed_by": self.sys_type.value},
                    execution_time_ms=50.0
                )
            
            def get_available_operations(self) -> List[str]:
                base_ops = ["get_status", "health_check", "get_metrics"]
                
                if self.sys_type == SystemType.OBSERVABILITY:
                    return base_ops + ["start_monitoring", "stop_monitoring", "get_analytics"]
                elif self.sys_type == SystemType.STATE_CONFIG:
                    return base_ops + ["save_state", "load_state", "update_config"]
                elif self.sys_type == SystemType.ORCHESTRATION:
                    return base_ops + ["start_workflow", "pause_workflow", "route_task"]
                elif self.sys_type == SystemType.UI_DASHBOARD:
                    return base_ops + ["create_dashboard", "update_widget", "export_layout"]
                
                return base_ops
            
            def validate_operation_parameters(self, operation: str, parameters: Dict[str, Any]) -> bool:
                # Basic validation - operation exists
                return operation in self.get_available_operations()
        
        return MockSystemAdapter(system_type)
    
    async def _setup_health_monitoring(self):
        """Setup automated health monitoring"""
        async def health_check_handler(message: SystemMessage):
            """Handle health check events"""
            if message.event_type == IntegrationEventType.PERFORMANCE_ALERT:
                await self._handle_performance_alert(message)
            elif message.event_type == IntegrationEventType.ERROR_OCCURRED:
                await self._handle_system_error(message)
        
        # Register health monitoring handlers
        self.message_bus.register_event_handler(
            IntegrationEventType.PERFORMANCE_ALERT, 
            health_check_handler
        )
        self.message_bus.register_event_handler(
            IntegrationEventType.ERROR_OCCURRED,
            health_check_handler
        )
    
    async def _handle_performance_alert(self, message: SystemMessage):
        """Handle performance alert from any system"""
        self.logger.warning(f"Performance alert from {message.source_system.value}: {message.payload}")
        
        # Update system status
        if message.source_system.value in self.integration_health["system_statuses"]:
            self.integration_health["system_statuses"][message.source_system.value] = "performance_degraded"
    
    async def _handle_system_error(self, message: SystemMessage):
        """Handle system error notification"""
        self.logger.error(f"System error from {message.source_system.value}: {message.payload}")
        
        # Update system status
        if message.source_system.value in self.integration_health["system_statuses"]:
            self.integration_health["system_statuses"][message.source_system.value] = "error"
    
    async def execute_cross_system_operation(self, operation: str, target_system: SystemType, 
                                           parameters: Dict[str, Any] = None) -> CrossSystemResponse:
        """Execute operation across systems"""
        try:
            request = CrossSystemRequest(
                operation=operation,
                target_system=target_system,
                parameters=parameters or {},
                timeout_seconds=30
            )
            
            response = await self.message_bus.send_request(request)
            
            self.logger.info(f"Cross-system operation {operation} on {target_system.value}: "
                           f"{'SUCCESS' if response.success else 'FAILED'} "
                           f"({response.execution_time_ms:.1f}ms)")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Cross-system operation failed: {e}")
            return CrossSystemResponse(
                request_id=f"error_{uuid.uuid4().hex[:8]}",
                success=False,
                error_message=str(e)
            )
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        bus_stats = self.message_bus.get_bus_statistics()
        
        return {
            "integration_health": self.integration_health,
            "message_bus_stats": bus_stats,
            "registered_systems": len(self.system_adapters),
            "active_workflows": len(self.active_workflows),
            "api_status": "operational" if self.integration_health["overall_status"] == "operational" else "degraded"
        }
    
    async def shutdown(self):
        """Gracefully shutdown the API coordinator"""
        self.logger.info("Shutting down cross-system API coordinator")
        await self.message_bus.shutdown()


# ============================================================================
# GLOBAL INTEGRATION INSTANCE
# ============================================================================

# Global instance for cross-system integration
cross_system_coordinator = CrossSystemAPICoordinator()

# Export for external use
__all__ = [
    'SystemType',
    'IntegrationEventType', 
    'SystemMessage',
    'CrossSystemRequest',
    'CrossSystemResponse',
    'UnifiedSystemInterface',
    'CrossSystemMessageBus',
    'CrossSystemAPICoordinator',
    'cross_system_coordinator'
]