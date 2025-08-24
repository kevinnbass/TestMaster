"""
Integration Base Abstractions
============================

Core integration abstractions providing the foundation for all
integration capabilities in TestMaster.

Author: Agent E - Infrastructure Consolidation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from datetime import datetime

T = TypeVar('T')
R = TypeVar('R')


class IntegrationStatus(Enum):
    """Integration status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    FAILED = "failed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class IntegrationMode(Enum):
    """Integration mode enumeration."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"
    REAL_TIME = "real_time"
    ADAPTIVE = "adaptive"


class IntegrationPattern(Enum):
    """Integration pattern enumeration."""
    POINT_TO_POINT = "point_to_point"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    MESSAGE_QUEUE = "message_queue"
    EVENT_DRIVEN = "event_driven"
    SERVICE_MESH = "service_mesh"


class IntegrationPriority(Enum):
    """Integration priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class IntegrationMetrics:
    """Integration performance metrics."""
    
    # Connection metrics
    connection_time: float = 0.0
    reconnection_count: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: Optional[datetime] = None
    
    # Performance metrics
    requests_sent: int = 0
    requests_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    average_latency: float = 0.0
    success_rate: float = 0.0
    
    # Error metrics
    error_count: int = 0
    timeout_count: int = 0
    retry_count: int = 0
    circuit_breaker_trips: int = 0
    
    # Resource metrics
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    thread_count: int = 0
    connection_pool_size: int = 0
    
    def update_success_rate(self):
        """Update success rate calculation."""
        total_requests = self.requests_sent + self.requests_received
        if total_requests > 0:
            successful_requests = total_requests - self.error_count
            self.success_rate = successful_requests / total_requests
    
    def record_request(self, latency: float, success: bool = True):
        """Record a request with latency."""
        self.requests_sent += 1
        if not success:
            self.error_count += 1
        
        # Update average latency
        if self.requests_sent == 1:
            self.average_latency = latency
        else:
            self.average_latency = (
                (self.average_latency * (self.requests_sent - 1) + latency) / 
                self.requests_sent
            )
        
        self.update_success_rate()


@dataclass
class IntegrationConfiguration:
    """Integration configuration settings."""
    
    # Basic configuration
    name: str
    description: str = ""
    enabled: bool = True
    auto_start: bool = True
    
    # Connection configuration
    connection_timeout: int = 30
    read_timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Integration mode
    mode: IntegrationMode = IntegrationMode.ASYNCHRONOUS
    pattern: IntegrationPattern = IntegrationPattern.REQUEST_RESPONSE
    priority: IntegrationPriority = IntegrationPriority.NORMAL
    
    # Performance settings
    max_concurrent_connections: int = 10
    connection_pool_size: int = 5
    buffer_size: int = 8192
    batch_size: int = 100
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60
    
    # Monitoring settings
    health_check_enabled: bool = True
    health_check_interval: int = 30
    metrics_enabled: bool = True
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get custom setting value."""
        return self.custom_settings.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """Set custom setting value."""
        self.custom_settings[key] = value


@dataclass
class IntegrationContext:
    """Integration execution context."""
    
    integration_id: str
    session_id: str
    request_id: Optional[str] = None
    
    # Context data
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    timeout: Optional[int] = None
    priority: IntegrationPriority = IntegrationPriority.NORMAL
    retry_enabled: bool = True
    
    # Tracing information
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    
    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def start_execution(self):
        """Mark execution as started."""
        self.started_at = datetime.now()
    
    def complete_execution(self):
        """Mark execution as completed."""
        self.completed_at = datetime.now()


class IntegrationBase(ABC, Generic[T, R]):
    """
    Abstract base class for all integration implementations.
    
    Provides unified interface for integration lifecycle management,
    monitoring, and execution.
    """
    
    def __init__(self, config: IntegrationConfiguration):
        self.config = config
        self.status = IntegrationStatus.INITIALIZING
        self.metrics = IntegrationMetrics()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._started_at: Optional[datetime] = None
        self._circuit_breaker_state = "closed"
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: Optional[datetime] = None
    
    @property
    def integration_id(self) -> str:
        """Get unique integration identifier."""
        return f"{self.__class__.__name__}_{self.config.name}"
    
    @property
    def is_active(self) -> bool:
        """Check if integration is active."""
        return self.status == IntegrationStatus.ACTIVE
    
    @property
    def is_connected(self) -> bool:
        """Check if integration is connected."""
        return self.status == IntegrationStatus.CONNECTED
    
    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        if self._started_at:
            return (datetime.now() - self._started_at).total_seconds()
        return 0.0
    
    # Abstract methods - must be implemented by subclasses
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the integration."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from target system."""
        pass
    
    @abstractmethod
    async def send_request(self, request: T, context: IntegrationContext) -> R:
        """Send request to target system."""
        pass
    
    @abstractmethod
    async def receive_response(self, context: IntegrationContext) -> R:
        """Receive response from target system."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass
    
    # Lifecycle management
    
    async def start(self) -> bool:
        """Start the integration."""
        try:
            self.status = IntegrationStatus.INITIALIZING
            
            if not await self.initialize():
                self.status = IntegrationStatus.FAILED
                return False
            
            if not await self.connect():
                self.status = IntegrationStatus.FAILED
                return False
            
            self.status = IntegrationStatus.ACTIVE
            self._started_at = datetime.now()
            await self._emit_event("integration_started", {"integration_id": self.integration_id})
            
            return True
            
        except Exception as e:
            self.status = IntegrationStatus.ERROR
            await self._emit_event("integration_error", {"error": str(e)})
            return False
    
    async def stop(self) -> bool:
        """Stop the integration."""
        try:
            await self.disconnect()
            self.status = IntegrationStatus.TERMINATED
            await self._emit_event("integration_stopped", {"integration_id": self.integration_id})
            return True
            
        except Exception as e:
            await self._emit_event("integration_error", {"error": str(e)})
            return False
    
    async def restart(self) -> bool:
        """Restart the integration."""
        await self.stop()
        await asyncio.sleep(1)  # Brief pause
        return await self.start()
    
    # Integration execution
    
    async def execute(self, request: T, context: Optional[IntegrationContext] = None) -> R:
        """Execute integration request."""
        if not context:
            context = IntegrationContext(
                integration_id=self.integration_id,
                session_id=f"session_{int(time.time())}"
            )
        
        context.start_execution()
        
        try:
            # Check circuit breaker
            if not self._check_circuit_breaker():
                raise Exception("Circuit breaker is open")
            
            # Execute request
            start_time = time.time()
            response = await self.send_request(request, context)
            latency = time.time() - start_time
            
            # Record success
            self.metrics.record_request(latency, success=True)
            self._reset_circuit_breaker()
            
            context.complete_execution()
            return response
            
        except Exception as e:
            # Record failure
            latency = time.time() - start_time if 'start_time' in locals() else 0.0
            self.metrics.record_request(latency, success=False)
            self._record_circuit_breaker_failure()
            
            context.complete_execution()
            await self._emit_event("integration_error", {"error": str(e), "context": context})
            raise
    
    # Circuit breaker implementation
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows request."""
        if not self.config.circuit_breaker_enabled:
            return True
        
        if self._circuit_breaker_state == "open":
            # Check if recovery timeout has passed
            if (self._circuit_breaker_last_failure and 
                (datetime.now() - self._circuit_breaker_last_failure).total_seconds() > 
                self.config.recovery_timeout):
                self._circuit_breaker_state = "half_open"
                return True
            return False
        
        return True
    
    def _record_circuit_breaker_failure(self):
        """Record circuit breaker failure."""
        if not self.config.circuit_breaker_enabled:
            return
        
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.now()
        
        if self._circuit_breaker_failures >= self.config.failure_threshold:
            self._circuit_breaker_state = "open"
            self.metrics.circuit_breaker_trips += 1
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker on success."""
        self._circuit_breaker_state = "closed"
        self._circuit_breaker_failures = 0
    
    # Event handling
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler."""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].remove(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered handlers."""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, data)
                    else:
                        handler(event_type, data)
                except Exception as e:
                    # Log error but don't fail the integration
                    print(f"Event handler error: {e}")
    
    # Monitoring and metrics
    
    def get_metrics(self) -> IntegrationMetrics:
        """Get current metrics."""
        self.metrics.uptime_seconds = self.uptime
        return self.metrics
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information."""
        return {
            "integration_id": self.integration_id,
            "status": self.status.value,
            "uptime": self.uptime,
            "metrics": {
                "success_rate": self.metrics.success_rate,
                "average_latency": self.metrics.average_latency,
                "error_count": self.metrics.error_count,
                "requests_sent": self.metrics.requests_sent
            },
            "circuit_breaker": {
                "state": self._circuit_breaker_state,
                "failures": self._circuit_breaker_failures,
                "trips": self.metrics.circuit_breaker_trips
            },
            "configuration": {
                "mode": self.config.mode.value,
                "pattern": self.config.pattern.value,
                "priority": self.config.priority.value
            }
        }
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"id={self.integration_id}, "
                f"status={self.status.value}, "
                f"uptime={self.uptime:.1f}s)")


# Integration factory for creating specific integration types
class IntegrationFactory:
    """Factory for creating integration instances."""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, integration_type: str, integration_class: type):
        """Register integration class."""
        cls._registry[integration_type] = integration_class
    
    @classmethod
    def create(cls, integration_type: str, config: IntegrationConfiguration) -> IntegrationBase:
        """Create integration instance."""
        if integration_type not in cls._registry:
            raise ValueError(f"Unknown integration type: {integration_type}")
        
        integration_class = cls._registry[integration_type]
        return integration_class(config)
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of registered integration types."""
        return list(cls._registry.keys())


__all__ = [
    'IntegrationStatus',
    'IntegrationMode', 
    'IntegrationPattern',
    'IntegrationPriority',
    'IntegrationMetrics',
    'IntegrationConfiguration',
    'IntegrationContext',
    'IntegrationBase',
    'IntegrationFactory'
]