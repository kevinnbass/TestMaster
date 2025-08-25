"""
Orchestrator Base Abstractions
==============================

Core orchestrator interface and base implementations providing unified
abstractions for all orchestration systems in TestMaster.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Set
from pathlib import Path


logger = logging.getLogger(__name__)


class OrchestratorType(Enum):
    """Types of orchestrators in the hierarchy."""
    MASTER = "master"
    WORKFLOW = "workflow"
    SWARM = "swarm"
    INTELLIGENCE = "intelligence"
    HYBRID = "hybrid"
    SPECIALIZED = "specialized"


class OrchestratorStatus(Enum):
    """Orchestrator operational status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ExecutionStrategy(Enum):
    """Execution strategies for orchestration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    OPTIMIZED = "optimized"
    INTELLIGENT = "intelligent"
    WORKFLOW_DRIVEN = "workflow_driven"
    SEMANTIC_AWARE = "semantic_aware"
    CROSS_SYSTEM = "cross_system"


@dataclass
class OrchestratorMetrics:
    """Metrics for orchestrator performance."""
    total_tasks_executed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    peak_concurrent_tasks: int = 0
    resource_utilization: float = 0.0
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass 
class OrchestratorCapabilities:
    """Capabilities of an orchestrator."""
    max_concurrent_tasks: int = 10
    supports_dag_execution: bool = True
    supports_agent_coordination: bool = True
    supports_ml_optimization: bool = False
    supports_real_time_monitoring: bool = True
    supports_auto_scaling: bool = False
    supports_workflow_design: bool = False
    supports_workflow_optimization: bool = False
    supports_semantic_learning: bool = False
    supports_cross_system_coordination: bool = False
    supports_adaptive_execution: bool = False
    supports_intelligent_routing: bool = False
    supported_task_types: Set[str] = field(default_factory=set)
    integration_protocols: Set[str] = field(default_factory=set)
    workflow_patterns: Set[str] = field(default_factory=set)
    semantic_capabilities: Set[str] = field(default_factory=set)


class OrchestratorBase(ABC):
    """
    Abstract base class for all orchestrators in TestMaster.
    
    Provides unified interface and common functionality for:
    - Master orchestrators
    - Workflow execution engines
    - Swarm coordination systems
    - Intelligence orchestrators
    - Hybrid orchestration systems
    """
    
    def __init__(
        self,
        orchestrator_id: Optional[str] = None,
        orchestrator_type: OrchestratorType = OrchestratorType.SPECIALIZED,
        name: Optional[str] = None
    ):
        self.orchestrator_id = orchestrator_id or f"orch_{uuid.uuid4().hex[:8]}"
        self.orchestrator_type = orchestrator_type
        self.name = name or f"{orchestrator_type.value}_orchestrator"
        
        # Orchestrator state
        self.status = OrchestratorStatus.INITIALIZING
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Capabilities and configuration
        self.capabilities = OrchestratorCapabilities()
        self.configuration: Dict[str, Any] = {}
        
        # Metrics and monitoring
        self.metrics = OrchestratorMetrics()
        self.performance_history: List[OrchestratorMetrics] = []
        
        # Task and agent management
        self.active_tasks: Dict[str, Any] = {}
        self.task_queue: List[Any] = []
        self.registered_agents: Dict[str, Any] = {}
        
        # Workflow and semantic management
        self.active_workflows: Dict[str, Any] = {}
        self.workflow_templates: Dict[str, Any] = {}
        self.semantic_patterns: Dict[str, Any] = {}
        self.cross_system_connections: Dict[str, Any] = {}
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.execution_listeners: List[Callable] = []
        
        # Integration and coordination
        self.parent_orchestrator: Optional['OrchestratorBase'] = None
        self.child_orchestrators: List['OrchestratorBase'] = []
        self.integration_points: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"Orchestrator initialized: {self.name} ({self.orchestrator_id})")
    
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================
    
    @abstractmethod
    async def execute_task(self, task: Any) -> Any:
        """Execute a single task."""
        pass
    
    @abstractmethod
    async def execute_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Execute a batch of tasks."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the orchestrator."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the orchestrator."""
        pass
    
    @abstractmethod
    def get_supported_capabilities(self) -> OrchestratorCapabilities:
        """Get orchestrator capabilities."""
        pass
    
    # ========================================================================
    # WORKFLOW ORCHESTRATION METHODS (Optional - implement if workflow-enabled)
    # ========================================================================
    
    async def design_workflow(self, requirements: Dict[str, Any]) -> Optional[Any]:
        """Design a workflow based on requirements. Override if workflow-enabled."""
        if not self.capabilities.supports_workflow_design:
            return None
        # Subclasses can implement workflow design logic
        return None
    
    async def optimize_workflow(self, workflow: Any, performance_data: Dict[str, Any]) -> Optional[Any]:
        """Optimize workflow based on performance data. Override if optimization-enabled."""
        if not self.capabilities.supports_workflow_optimization:
            return None
        # Subclasses can implement workflow optimization logic
        return None
    
    async def execute_workflow(self, workflow: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow with context. Override for workflow-specific execution."""
        # Default implementation treats workflow as a complex task
        return await self.execute_task(workflow)
    
    async def learn_semantic_patterns(self, data: Any) -> Dict[str, Any]:
        """Learn semantic patterns from data. Override if semantic-aware."""
        if not self.capabilities.supports_semantic_learning:
            return {}
        # Subclasses can implement semantic learning logic
        return {}
    
    async def coordinate_cross_system(self, systems: List[str], coordination_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate across multiple systems. Override if cross-system enabled."""
        if not self.capabilities.supports_cross_system_coordination:
            return {"error": "Cross-system coordination not supported"}
        # Subclasses can implement cross-system coordination logic
        return {}
    
    # ========================================================================
    # ORCHESTRATOR LIFECYCLE MANAGEMENT
    # ========================================================================
    
    async def initialize(self, configuration: Dict[str, Any] = None) -> bool:
        """Initialize the orchestrator with configuration."""
        try:
            self.status = OrchestratorStatus.INITIALIZING
            
            if configuration:
                self.configuration.update(configuration)
            
            # Initialize capabilities
            self.capabilities = self.get_supported_capabilities()
            
            # Initialize subsystems
            await self._initialize_subsystems()
            
            # Register event handlers
            self._register_default_event_handlers()
            
            self.status = OrchestratorStatus.ACTIVE
            self.logger.info(f"Orchestrator {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.status = OrchestratorStatus.ERROR
            self.logger.error(f"Failed to initialize orchestrator {self.name}: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the orchestrator gracefully."""
        try:
            self.status = OrchestratorStatus.STOPPING
            
            # Complete active tasks
            await self._complete_active_tasks()
            
            # Cleanup resources
            await self._cleanup_resources()
            
            # Disconnect from integrations
            await self._disconnect_integrations()
            
            self.status = OrchestratorStatus.STOPPED
            self.logger.info(f"Orchestrator {self.name} shutdown successfully")
            return True
            
        except Exception as e:
            self.status = OrchestratorStatus.ERROR
            self.logger.error(f"Failed to shutdown orchestrator {self.name}: {e}")
            return False
    
    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================
    
    def submit_task(self, task: Any) -> str:
        """Submit a task for execution."""
        task_id = getattr(task, 'task_id', f"task_{uuid.uuid4().hex[:8]}")
        
        self.task_queue.append(task)
        self.active_tasks[task_id] = task
        
        self.logger.debug(f"Task submitted: {task_id}")
        self._emit_event('task_submitted', {'task_id': task_id, 'task': task})
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a specific task."""
        task = self.active_tasks.get(task_id)
        if task:
            return getattr(task, 'status', 'unknown')
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            # Implement task cancellation logic
            self.logger.info(f"Task cancelled: {task_id}")
            self._emit_event('task_cancelled', {'task_id': task_id})
            return True
        return False
    
    # ========================================================================
    # AGENT MANAGEMENT
    # ========================================================================
    
    def register_agent(self, agent: Any) -> str:
        """Register an agent with the orchestrator."""
        agent_id = getattr(agent, 'agent_id', f"agent_{uuid.uuid4().hex[:8]}")
        
        self.registered_agents[agent_id] = agent
        self.logger.info(f"Agent registered: {agent_id}")
        self._emit_event('agent_registered', {'agent_id': agent_id, 'agent': agent})
        
        return agent_id
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
            self.logger.info(f"Agent unregistered: {agent_id}")
            self._emit_event('agent_unregistered', {'agent_id': agent_id})
            return True
        return False
    
    def get_available_agents(self) -> List[Any]:
        """Get list of available agents."""
        return [agent for agent in self.registered_agents.values() 
                if getattr(agent, 'status', None) == 'available']
    
    # ========================================================================
    # WORKFLOW MANAGEMENT
    # ========================================================================
    
    def register_workflow_template(self, template_id: str, template: Any) -> bool:
        """Register a workflow template."""
        self.workflow_templates[template_id] = template
        self.logger.info(f"Workflow template registered: {template_id}")
        self._emit_event('workflow_template_registered', {'template_id': template_id})
        return True
    
    def create_workflow_instance(self, template_id: str, instance_id: str = None, 
                                context: Dict[str, Any] = None) -> Optional[str]:
        """Create a workflow instance from template."""
        if template_id not in self.workflow_templates:
            return None
        
        instance_id = instance_id or f"workflow_{len(self.active_workflows)}"
        template = self.workflow_templates[template_id]
        
        # Create workflow instance (implementation depends on workflow system)
        workflow_instance = {
            'template_id': template_id,
            'instance_id': instance_id,
            'template': template,
            'context': context or {},
            'status': 'created',
            'created_at': datetime.now()
        }
        
        self.active_workflows[instance_id] = workflow_instance
        self.logger.info(f"Workflow instance created: {instance_id}")
        self._emit_event('workflow_instance_created', {'instance_id': instance_id})
        
        return instance_id
    
    def get_workflow_status(self, instance_id: str) -> Optional[str]:
        """Get workflow instance status."""
        workflow = self.active_workflows.get(instance_id)
        return workflow.get('status') if workflow else None
    
    def stop_workflow(self, instance_id: str) -> bool:
        """Stop a workflow instance."""
        if instance_id in self.active_workflows:
            workflow = self.active_workflows[instance_id]
            workflow['status'] = 'stopped'
            self.logger.info(f"Workflow stopped: {instance_id}")
            self._emit_event('workflow_stopped', {'instance_id': instance_id})
            return True
        return False
    
    # ========================================================================
    # SEMANTIC PATTERN MANAGEMENT
    # ========================================================================
    
    def register_semantic_pattern(self, pattern_id: str, pattern: Any) -> bool:
        """Register a semantic pattern."""
        self.semantic_patterns[pattern_id] = {
            'pattern': pattern,
            'registered_at': datetime.now(),
            'usage_count': 0
        }
        self.logger.info(f"Semantic pattern registered: {pattern_id}")
        self._emit_event('semantic_pattern_registered', {'pattern_id': pattern_id})
        return True
    
    def apply_semantic_pattern(self, pattern_id: str, data: Any) -> Optional[Any]:
        """Apply a semantic pattern to data."""
        if pattern_id not in self.semantic_patterns:
            return None
        
        pattern_info = self.semantic_patterns[pattern_id]
        pattern_info['usage_count'] += 1
        
        # Pattern application logic would be implemented by subclasses
        self.logger.debug(f"Applied semantic pattern: {pattern_id}")
        return data  # Placeholder - subclasses implement actual pattern application
    
    def get_semantic_insights(self) -> Dict[str, Any]:
        """Get semantic learning insights."""
        return {
            'total_patterns': len(self.semantic_patterns),
            'most_used_patterns': sorted(
                [(pid, info['usage_count']) for pid, info in self.semantic_patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            'cross_system_connections': len(self.cross_system_connections)
        }
    
    # ========================================================================
    # ORCHESTRATOR HIERARCHY MANAGEMENT
    # ========================================================================
    
    def set_parent_orchestrator(self, parent: 'OrchestratorBase'):
        """Set parent orchestrator in hierarchy."""
        self.parent_orchestrator = parent
        parent.child_orchestrators.append(self)
        self.logger.info(f"Parent orchestrator set: {parent.name}")
    
    def add_child_orchestrator(self, child: 'OrchestratorBase'):
        """Add child orchestrator."""
        child.parent_orchestrator = self
        self.child_orchestrators.append(child)
        self.logger.info(f"Child orchestrator added: {child.name}")
    
    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get information about orchestrator hierarchy."""
        return {
            'orchestrator_id': self.orchestrator_id,
            'name': self.name,
            'type': self.orchestrator_type.value,
            'parent': self.parent_orchestrator.name if self.parent_orchestrator else None,
            'children': [child.name for child in self.child_orchestrators],
            'hierarchy_depth': self._get_hierarchy_depth()
        }
    
    # ========================================================================
    # CONFIGURATION AND CAPABILITIES
    # ========================================================================
    
    def configure(self, configuration: Dict[str, Any]):
        """Update orchestrator configuration."""
        self.configuration.update(configuration)
        self.logger.info("Orchestrator configuration updated")
        self._emit_event('configuration_updated', {'configuration': configuration})
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current orchestrator configuration."""
        return self.configuration.copy()
    
    def update_capabilities(self, capabilities: Dict[str, Any]):
        """Update orchestrator capabilities."""
        for key, value in capabilities.items():
            if hasattr(self.capabilities, key):
                setattr(self.capabilities, key, value)
        
        self.logger.info("Orchestrator capabilities updated")
        self._emit_event('capabilities_updated', {'capabilities': capabilities})
    
    # ========================================================================
    # METRICS AND MONITORING
    # ========================================================================
    
    def update_metrics(self, metrics_update: Dict[str, Any]):
        """Update orchestrator metrics."""
        for key, value in metrics_update.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        
        self.metrics.last_updated = datetime.now()
        self.last_activity = datetime.now()
    
    def get_performance_metrics(self) -> OrchestratorMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            'orchestrator_id': self.orchestrator_id,
            'name': self.name,
            'type': self.orchestrator_type.value,
            'status': self.status.value,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'registered_agents': len(self.registered_agents),
            'active_workflows': len(self.active_workflows),
            'workflow_templates': len(self.workflow_templates),
            'semantic_patterns': len(self.semantic_patterns),
            'cross_system_connections': len(self.cross_system_connections),
            'capabilities': {
                'workflow_design': self.capabilities.supports_workflow_design,
                'workflow_optimization': self.capabilities.supports_workflow_optimization,
                'semantic_learning': self.capabilities.supports_semantic_learning,
                'cross_system_coordination': self.capabilities.supports_cross_system_coordination,
                'adaptive_execution': self.capabilities.supports_adaptive_execution
            },
            'metrics': {
                'total_tasks_executed': self.metrics.total_tasks_executed,
                'success_rate': self.metrics.successful_tasks / max(self.metrics.total_tasks_executed, 1),
                'average_execution_time': self.metrics.average_execution_time,
                'throughput_per_minute': self.metrics.throughput_per_minute
            },
            'last_activity': self.last_activity.isoformat(),
            'uptime_seconds': (datetime.now() - self.created_at).total_seconds()
        }
    
    # ========================================================================
    # EVENT HANDLING
    # ========================================================================
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit event to registered handlers."""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event_type, event_data)
            except Exception as e:
                self.logger.warning(f"Event handler error for {event_type}: {e}")
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    async def _initialize_subsystems(self):
        """Initialize orchestrator subsystems."""
        # Subclasses can override to initialize specific subsystems
        pass
    
    async def _complete_active_tasks(self):
        """Complete or cancel active tasks during shutdown."""
        # Wait for active tasks to complete or implement cancellation
        pass
    
    async def _cleanup_resources(self):
        """Cleanup orchestrator resources."""
        # Subclasses can override to cleanup specific resources
        pass
    
    async def _disconnect_integrations(self):
        """Disconnect from integration points."""
        # Subclasses can override to handle specific integrations
        pass
    
    def _register_default_event_handlers(self):
        """Register default event handlers."""
        self.add_event_handler('task_completed', self._handle_task_completed)
        self.add_event_handler('task_failed', self._handle_task_failed)
    
    def _handle_task_completed(self, event_type: str, event_data: Dict[str, Any]):
        """Handle task completion event."""
        self.metrics.successful_tasks += 1
        self.metrics.total_tasks_executed += 1
    
    def _handle_task_failed(self, event_type: str, event_data: Dict[str, Any]):
        """Handle task failure event."""
        self.metrics.failed_tasks += 1
        self.metrics.total_tasks_executed += 1
    
    def _get_hierarchy_depth(self) -> int:
        """Get depth in orchestrator hierarchy."""
        depth = 0
        current = self.parent_orchestrator
        while current:
            depth += 1
            current = current.parent_orchestrator
        return depth
    
    def __str__(self) -> str:
        return f"{self.name} ({self.orchestrator_type.value})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.orchestrator_id}, "
                f"type={self.orchestrator_type.value}, status={self.status.value})")


# Export key classes
__all__ = [
    'OrchestratorType',
    'OrchestratorStatus',
    'ExecutionStrategy',
    'OrchestratorMetrics',
    'OrchestratorCapabilities',
    'OrchestratorBase'
]