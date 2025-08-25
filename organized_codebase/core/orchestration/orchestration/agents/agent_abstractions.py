"""
Agent Abstractions
=================

Unified agent abstractions providing consistent agent representation
across all orchestration systems in TestMaster.

Author: Agent E - Infrastructure Consolidation
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Set
from abc import ABC, abstractmethod


class AgentStatus(Enum):
    """Agent operational status."""
    INITIALIZING = "initializing"
    AVAILABLE = "available"
    BUSY = "busy"
    PAUSED = "paused"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentCapability(Enum):
    """Agent capability types."""
    TASK_EXECUTION = "task_execution"
    DATA_PROCESSING = "data_processing"
    ANALYSIS = "analysis"
    INTELLIGENCE = "intelligence"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    TESTING = "testing"
    INTEGRATION = "integration"
    WORKFLOW = "workflow"
    SWARM = "swarm"


class AgentType(Enum):
    """Agent type classification."""
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    HYBRID = "hybrid"
    INTELLIGENCE = "intelligence"
    SUPERVISOR = "supervisor"


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    current_load: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 100.0
    uptime_seconds: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class AgentConfiguration:
    """Agent configuration and settings."""
    max_concurrent_tasks: int = 5
    execution_timeout: timedelta = field(default_factory=lambda: timedelta(hours=1))
    retry_attempts: int = 3
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    specialized_config: Dict[str, Any] = field(default_factory=dict)
    communication_protocols: List[str] = field(default_factory=list)


@dataclass
class AgentResources:
    """Agent resource allocation and usage."""
    cpu_cores: int = 1
    memory_mb: int = 512
    gpu_access: bool = False
    network_bandwidth: Optional[int] = None
    storage_mb: int = 1024
    allocated_resources: Dict[str, Any] = field(default_factory=dict)
    resource_reservations: List[str] = field(default_factory=list)


class AgentCommunication:
    """Agent communication management."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_queue: List[Dict[str, Any]] = []
        self.sent_messages: List[Dict[str, Any]] = []
        self.communication_channels: Dict[str, Any] = {}
        self.protocol_handlers: Dict[str, Callable] = {}
    
    async def send_message(self, target_agent: str, message: Dict[str, Any]) -> bool:
        """Send message to another agent."""
        try:
            message['sender'] = self.agent_id
            message['target'] = target_agent
            message['timestamp'] = datetime.now()
            self.sent_messages.append(message)
            return True
        except Exception:
            return False
    
    def receive_message(self, message: Dict[str, Any]):
        """Receive message from another agent."""
        self.message_queue.append(message)
    
    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get pending messages."""
        messages = self.message_queue.copy()
        self.message_queue.clear()
        return messages


class UnifiedAgent(ABC):
    """
    Unified agent abstraction for all orchestration systems.
    
    Provides consistent interface for agent definition, lifecycle management,
    and coordination across workflow, swarm, intelligence, and hybrid orchestrators.
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_type: AgentType = AgentType.WORKER,
        capabilities: List[AgentCapability] = None,
        agent_id: Optional[str] = None
    ):
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.capabilities = set(capabilities or [AgentCapability.TASK_EXECUTION])
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.now()
        self.last_heartbeat = datetime.now()
        
        # Agent configuration and resources
        self.configuration = AgentConfiguration()
        self.resources = AgentResources()
        self.metrics = AgentMetrics()
        
        # Task management
        self.active_tasks: Dict[str, Any] = {}
        self.task_queue: List[Any] = []
        self.task_history: List[Dict[str, Any]] = []
        
        # Communication
        self.communication = AgentCommunication(self.agent_id)
        self.registered_orchestrators: Set[str] = set()
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.status_change_handlers: List[Callable] = []
        
        # Specialized functionality
        self.plugins: Dict[str, Any] = {}
        self.extensions: Dict[str, Any] = {}
    
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by agent implementations
    # ========================================================================
    
    @abstractmethod
    async def execute_task(self, task: Any) -> Any:
        """Execute assigned task."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize agent systems and capabilities."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown agent gracefully."""
        pass
    
    @abstractmethod
    def validate_task(self, task: Any) -> bool:
        """Validate if agent can execute the given task."""
        pass
    
    # ========================================================================
    # AGENT LIFECYCLE MANAGEMENT
    # ========================================================================
    
    async def start(self) -> bool:
        """Start the agent."""
        try:
            self.status = AgentStatus.INITIALIZING
            
            # Initialize agent systems
            if not await self.initialize():
                self.status = AgentStatus.ERROR
                return False
            
            # Start heartbeat
            self._start_heartbeat()
            
            self.status = AgentStatus.AVAILABLE
            self._emit_status_change('agent_started')
            return True
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self._emit_status_change('agent_error', {'error': str(e)})
            return False
    
    async def stop(self) -> bool:
        """Stop the agent."""
        try:
            self.status = AgentStatus.OFFLINE
            
            # Complete active tasks
            await self._complete_active_tasks()
            
            # Shutdown agent systems
            await self.shutdown()
            
            self._emit_status_change('agent_stopped')
            return True
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return False
    
    def pause(self):
        """Pause agent operations."""
        if self.status == AgentStatus.AVAILABLE or self.status == AgentStatus.BUSY:
            self.status = AgentStatus.PAUSED
            self._emit_status_change('agent_paused')
    
    def resume(self):
        """Resume agent operations."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.AVAILABLE
            self._emit_status_change('agent_resumed')
    
    def set_maintenance_mode(self, enabled: bool):
        """Set agent maintenance mode."""
        if enabled:
            self.status = AgentStatus.MAINTENANCE
            self._emit_status_change('agent_maintenance_start')
        else:
            self.status = AgentStatus.AVAILABLE
            self._emit_status_change('agent_maintenance_end')
    
    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================
    
    async def assign_task(self, task: Any) -> bool:
        """Assign task to agent."""
        try:
            # Validate task
            if not self.validate_task(task):
                return False
            
            # Check capacity
            if len(self.active_tasks) >= self.configuration.max_concurrent_tasks:
                self.task_queue.append(task)
                return True
            
            # Execute task
            task_id = getattr(task, 'task_id', f"task_{uuid.uuid4().hex[:8]}")
            self.active_tasks[task_id] = task
            self.status = AgentStatus.BUSY
            
            # Execute asynchronously
            result = await self.execute_task(task)
            
            # Record completion
            await self._complete_task(task_id, result)
            
            return True
            
        except Exception as e:
            self._emit_event('task_failed', {'task': task, 'error': str(e)})
            return False
    
    async def _complete_task(self, task_id: str, result: Any):
        """Complete task execution."""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self._update_metrics()
            
            # Record in history
            self.task_history.append({
                'task_id': task_id,
                'completed_at': datetime.now(),
                'result': result,
                'success': True
            })
            
            # Process queued tasks
            if self.task_queue and len(self.active_tasks) < self.configuration.max_concurrent_tasks:
                next_task = self.task_queue.pop(0)
                await self.assign_task(next_task)
            
            # Update status
            if not self.active_tasks:
                self.status = AgentStatus.AVAILABLE
            
            self._emit_event('task_completed', {'task_id': task_id, 'result': result})
    
    async def _complete_active_tasks(self):
        """Complete all active tasks during shutdown."""
        # Simple implementation - wait for tasks to complete naturally
        # Subclasses can override for more sophisticated handling
        pass
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of assigned task."""
        if task_id in self.active_tasks:
            return "running"
        
        for task_record in self.task_history:
            if task_record['task_id'] == task_id:
                return "completed" if task_record['success'] else "failed"
        
        return None
    
    # ========================================================================
    # CAPABILITY MANAGEMENT
    # ========================================================================
    
    def add_capability(self, capability: AgentCapability):
        """Add agent capability."""
        self.capabilities.add(capability)
        self._emit_event('capability_added', {'capability': capability.value})
    
    def remove_capability(self, capability: AgentCapability):
        """Remove agent capability."""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            self._emit_event('capability_removed', {'capability': capability.value})
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability."""
        return capability in self.capabilities
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get all agent capabilities."""
        return list(self.capabilities)
    
    # ========================================================================
    # COMMUNICATION MANAGEMENT
    # ========================================================================
    
    async def send_message(self, target_agent: str, message: Dict[str, Any]) -> bool:
        """Send message to another agent."""
        return await self.communication.send_message(target_agent, message)
    
    def receive_message(self, message: Dict[str, Any]):
        """Receive message from another agent."""
        self.communication.receive_message(message)
        self._emit_event('message_received', {'message': message})
    
    def process_messages(self):
        """Process pending messages."""
        messages = self.communication.get_pending_messages()
        for message in messages:
            self._handle_message(message)
    
    def _handle_message(self, message: Dict[str, Any]):
        """Handle received message."""
        # Default implementation - subclasses can override
        message_type = message.get('type', 'unknown')
        if message_type == 'task_assignment':
            # Handle task assignment message
            pass
        elif message_type == 'coordination':
            # Handle coordination message
            pass
    
    # ========================================================================
    # ORCHESTRATOR INTEGRATION
    # ========================================================================
    
    def register_with_orchestrator(self, orchestrator_id: str):
        """Register agent with orchestrator."""
        self.registered_orchestrators.add(orchestrator_id)
        self._emit_event('orchestrator_registered', {'orchestrator_id': orchestrator_id})
    
    def unregister_from_orchestrator(self, orchestrator_id: str):
        """Unregister agent from orchestrator."""
        if orchestrator_id in self.registered_orchestrators:
            self.registered_orchestrators.remove(orchestrator_id)
            self._emit_event('orchestrator_unregistered', {'orchestrator_id': orchestrator_id})
    
    def get_registered_orchestrators(self) -> List[str]:
        """Get list of registered orchestrators."""
        return list(self.registered_orchestrators)
    
    # ========================================================================
    # CONFIGURATION AND RESOURCES
    # ========================================================================
    
    def configure(self, configuration: Dict[str, Any]):
        """Update agent configuration."""
        for key, value in configuration.items():
            if hasattr(self.configuration, key):
                setattr(self.configuration, key, value)
        
        self._emit_event('configuration_updated', {'configuration': configuration})
    
    def allocate_resources(self, resources: Dict[str, Any]):
        """Allocate resources to agent."""
        for key, value in resources.items():
            if hasattr(self.resources, key):
                setattr(self.resources, key, value)
        
        self._emit_event('resources_allocated', {'resources': resources})
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            'cpu_utilization': self.metrics.resource_utilization.get('cpu', 0.0),
            'memory_usage': self.metrics.resource_utilization.get('memory', 0.0),
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'current_load': self.metrics.current_load
        }
    
    # ========================================================================
    # METRICS AND MONITORING
    # ========================================================================
    
    def _update_metrics(self):
        """Update agent metrics."""
        self.metrics.last_activity = datetime.now()
        
        # Calculate success rate
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks > 0:
            self.metrics.success_rate = (self.metrics.tasks_completed / total_tasks) * 100
        
        # Calculate current load
        self.metrics.current_load = len(self.active_tasks) / max(self.configuration.max_concurrent_tasks, 1)
        
        # Calculate uptime
        self.metrics.uptime_seconds = (datetime.now() - self.created_at).total_seconds()
    
    def get_performance_metrics(self) -> AgentMetrics:
        """Get current performance metrics."""
        self._update_metrics()
        return self.metrics
    
    def _start_heartbeat(self):
        """Start agent heartbeat."""
        self.last_heartbeat = datetime.now()
    
    def heartbeat(self):
        """Send heartbeat signal."""
        self.last_heartbeat = datetime.now()
        self._emit_event('heartbeat', {'timestamp': self.last_heartbeat})
    
    # ========================================================================
    # EVENT HANDLING
    # ========================================================================
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def add_status_change_handler(self, handler: Callable):
        """Add status change handler."""
        self.status_change_handlers.append(handler)
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any] = None):
        """Emit event to registered handlers."""
        event_data = event_data or {}
        event_data.update({
            'agent_id': self.agent_id,
            'timestamp': datetime.now()
        })
        
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event_type, event_data)
            except Exception:
                pass  # Silent fail for event handlers
    
    def _emit_status_change(self, event_type: str, event_data: Dict[str, Any] = None):
        """Emit status change event."""
        event_data = event_data or {}
        event_data.update({
            'agent_id': self.agent_id,
            'old_status': getattr(self, '_previous_status', None),
            'new_status': self.status.value,
            'timestamp': datetime.now()
        })
        
        for handler in self.status_change_handlers:
            try:
                handler(event_type, event_data)
            except Exception:
                pass  # Silent fail for event handlers
        
        self._emit_event(event_type, event_data)
    
    # ========================================================================
    # AGENT INFORMATION
    # ========================================================================
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type.value,
            'status': self.status.value,
            'capabilities': [cap.value for cap in self.capabilities],
            'created_at': self.created_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'completed_tasks': self.metrics.tasks_completed,
            'failed_tasks': self.metrics.tasks_failed,
            'success_rate': self.metrics.success_rate,
            'current_load': self.metrics.current_load,
            'uptime_seconds': self.metrics.uptime_seconds,
            'registered_orchestrators': list(self.registered_orchestrators)
        }
    
    def __str__(self) -> str:
        return f"{self.agent_name} ({self.status.value})"
    
    def __repr__(self) -> str:
        return (f"UnifiedAgent(id={self.agent_id}, name={self.agent_name}, "
                f"type={self.agent_type.value}, status={self.status.value})")


# Export key classes
__all__ = [
    'AgentStatus',
    'AgentCapability',
    'AgentType',
    'AgentMetrics',
    'AgentConfiguration',
    'AgentResources',
    'AgentCommunication',
    'UnifiedAgent'
]