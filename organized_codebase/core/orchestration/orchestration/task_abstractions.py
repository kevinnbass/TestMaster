"""
Task Abstractions
================

Unified task abstractions providing consistent task representation
across all orchestration systems in TestMaster.

Author: Agent E - Infrastructure Consolidation
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Set
from abc import ABC, abstractmethod


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class TaskType(Enum):
    """Task type classification."""
    WORKFLOW = "workflow"
    SWARM = "swarm"
    INTELLIGENCE = "intelligence"
    HYBRID = "hybrid"
    ATOMIC = "atomic"
    COMPOSITE = "composite"


@dataclass
class TaskMetadata:
    """Task metadata for tracking and analysis."""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    created_at: datetime
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    labels: Dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None
    owner: Optional[str] = None


@dataclass
class TaskDependencies:
    """Task dependency management."""
    depends_on: List[str] = field(default_factory=list)
    dependent_tasks: List[str] = field(default_factory=list)
    soft_dependencies: List[str] = field(default_factory=list)
    conditional_dependencies: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskResources:
    """Task resource requirements and allocation."""
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    gpu_required: bool = False
    network_bandwidth: Optional[int] = None
    storage_mb: Optional[int] = None
    execution_timeout: Optional[timedelta] = None
    max_retries: int = 3
    resource_tags: Set[str] = field(default_factory=set)


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result_data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[timedelta] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    output_artifacts: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class TaskExecutionContext:
    """Task execution context and environment."""
    
    def __init__(
        self,
        execution_id: Optional[str] = None,
        orchestrator_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        self.execution_id = execution_id or f"exec_{uuid.uuid4().hex[:8]}"
        self.orchestrator_id = orchestrator_id
        self.session_id = session_id
        self.environment_vars: Dict[str, str] = {}
        self.execution_params: Dict[str, Any] = {}
        self.shared_state: Dict[str, Any] = {}
        self.created_at = datetime.now()


class UnifiedTask(ABC):
    """
    Unified task abstraction for all orchestration systems.
    
    Provides consistent interface for task definition, execution,
    and management across workflow, swarm, intelligence, and hybrid orchestrators.
    """
    
    def __init__(
        self,
        task_name: str,
        task_type: TaskType = TaskType.ATOMIC,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None
    ):
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        self.task_name = task_name
        self.task_type = task_type
        self.priority = priority
        
        # Task state
        self.status = TaskStatus.PENDING
        self.current_retry = 0
        
        # Task metadata
        self.metadata = TaskMetadata(
            task_id=self.task_id,
            task_type=task_type,
            priority=priority,
            created_at=datetime.now()
        )
        
        # Task configuration
        self.dependencies = TaskDependencies()
        self.resources = TaskResources()
        self.execution_context: Optional[TaskExecutionContext] = None
        
        # Task data
        self.input_data: Dict[str, Any] = {}
        self.output_data: Dict[str, Any] = {}
        self.task_config: Dict[str, Any] = {}
        
        # Event handlers
        self.status_change_handlers: List[Callable] = []
        self.progress_handlers: List[Callable] = []
        
        # Results and metrics
        self.result: Optional[TaskResult] = None
        self.execution_history: List[TaskResult] = []
    
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by task implementations
    # ========================================================================
    
    @abstractmethod
    async def execute(self, context: TaskExecutionContext) -> TaskResult:
        """Execute the task and return result."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate task configuration and dependencies."""
        pass
    
    @abstractmethod
    def estimate_resources(self) -> TaskResources:
        """Estimate required resources for execution."""
        pass
    
    # ========================================================================
    # TASK LIFECYCLE MANAGEMENT
    # ========================================================================
    
    def submit(self, execution_context: TaskExecutionContext = None):
        """Submit task for execution."""
        self.status = TaskStatus.QUEUED
        self.metadata.submitted_at = datetime.now()
        self.execution_context = execution_context or TaskExecutionContext()
        self._emit_status_change('task_submitted')
    
    async def start(self) -> bool:
        """Start task execution."""
        try:
            if not self.validate():
                self.status = TaskStatus.FAILED
                return False
            
            self.status = TaskStatus.RUNNING
            self.metadata.started_at = datetime.now()
            self._emit_status_change('task_started')
            
            return True
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            self._emit_status_change('task_failed', {'error': str(e)})
            return False
    
    async def complete(self, result: TaskResult):
        """Complete task execution."""
        self.status = result.status
        self.result = result
        self.metadata.completed_at = datetime.now()
        self.execution_history.append(result)
        
        self._emit_status_change('task_completed', {'result': result})
    
    def pause(self):
        """Pause task execution."""
        if self.status == TaskStatus.RUNNING:
            self.status = TaskStatus.PAUSED
            self._emit_status_change('task_paused')
    
    def resume(self):
        """Resume paused task."""
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
            self._emit_status_change('task_resumed')
    
    def cancel(self):
        """Cancel task execution."""
        self.status = TaskStatus.CANCELLED
        self._emit_status_change('task_cancelled')
    
    # ========================================================================
    # DEPENDENCY MANAGEMENT
    # ========================================================================
    
    def add_dependency(self, task_id: str, dependency_type: str = "hard"):
        """Add task dependency."""
        if dependency_type == "hard":
            self.dependencies.depends_on.append(task_id)
        elif dependency_type == "soft":
            self.dependencies.soft_dependencies.append(task_id)
        elif dependency_type == "conditional":
            condition = f"task_{task_id}_completed"
            self.dependencies.conditional_dependencies[task_id] = condition
    
    def remove_dependency(self, task_id: str):
        """Remove task dependency."""
        if task_id in self.dependencies.depends_on:
            self.dependencies.depends_on.remove(task_id)
        if task_id in self.dependencies.soft_dependencies:
            self.dependencies.soft_dependencies.remove(task_id)
        if task_id in self.dependencies.conditional_dependencies:
            del self.dependencies.conditional_dependencies[task_id]
    
    def check_dependencies_satisfied(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        # Check hard dependencies
        for dep in self.dependencies.depends_on:
            if dep not in completed_tasks:
                return False
        
        # Conditional dependencies could be checked here with custom logic
        return True
    
    # ========================================================================
    # RESOURCE MANAGEMENT
    # ========================================================================
    
    def set_resource_requirements(
        self,
        cpu_cores: Optional[int] = None,
        memory_mb: Optional[int] = None,
        gpu_required: bool = False,
        execution_timeout: Optional[timedelta] = None
    ):
        """Set task resource requirements."""
        if cpu_cores is not None:
            self.resources.cpu_cores = cpu_cores
        if memory_mb is not None:
            self.resources.memory_mb = memory_mb
        self.resources.gpu_required = gpu_required
        if execution_timeout is not None:
            self.resources.execution_timeout = execution_timeout
    
    def get_resource_requirements(self) -> TaskResources:
        """Get current resource requirements."""
        return self.resources
    
    # ========================================================================
    # DATA MANAGEMENT
    # ========================================================================
    
    def set_input_data(self, data: Dict[str, Any]):
        """Set task input data."""
        self.input_data.update(data)
    
    def get_input_data(self, key: Optional[str] = None) -> Any:
        """Get task input data."""
        if key:
            return self.input_data.get(key)
        return self.input_data
    
    def set_output_data(self, data: Dict[str, Any]):
        """Set task output data."""
        self.output_data.update(data)
    
    def get_output_data(self, key: Optional[str] = None) -> Any:
        """Get task output data."""
        if key:
            return self.output_data.get(key)
        return self.output_data
    
    # ========================================================================
    # EVENT HANDLING
    # ========================================================================
    
    def add_status_change_handler(self, handler: Callable):
        """Add status change event handler."""
        self.status_change_handlers.append(handler)
    
    def add_progress_handler(self, handler: Callable):
        """Add progress update handler."""
        self.progress_handlers.append(handler)
    
    def _emit_status_change(self, event_type: str, event_data: Dict[str, Any] = None):
        """Emit status change event."""
        event_data = event_data or {}
        event_data.update({
            'task_id': self.task_id,
            'status': self.status.value,
            'timestamp': datetime.now()
        })
        
        for handler in self.status_change_handlers:
            try:
                handler(event_type, event_data)
            except Exception:
                pass  # Silent fail for event handlers
    
    def _emit_progress(self, progress: float, message: str = ""):
        """Emit progress update."""
        for handler in self.progress_handlers:
            try:
                handler(self.task_id, progress, message)
            except Exception:
                pass  # Silent fail for event handlers
    
    # ========================================================================
    # TASK INFORMATION
    # ========================================================================
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get comprehensive task information."""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.metadata.created_at.isoformat(),
            'submitted_at': self.metadata.submitted_at.isoformat() if self.metadata.submitted_at else None,
            'started_at': self.metadata.started_at.isoformat() if self.metadata.started_at else None,
            'completed_at': self.metadata.completed_at.isoformat() if self.metadata.completed_at else None,
            'dependencies': {
                'hard': self.dependencies.depends_on,
                'soft': self.dependencies.soft_dependencies,
                'conditional': self.dependencies.conditional_dependencies
            },
            'resources': {
                'cpu_cores': self.resources.cpu_cores,
                'memory_mb': self.resources.memory_mb,
                'gpu_required': self.resources.gpu_required,
                'timeout': str(self.resources.execution_timeout) if self.resources.execution_timeout else None
            },
            'retry_count': self.current_retry,
            'max_retries': self.resources.max_retries
        }
    
    def __str__(self) -> str:
        return f"{self.task_name} ({self.status.value})"
    
    def __repr__(self) -> str:
        return (f"UnifiedTask(id={self.task_id}, name={self.task_name}, "
                f"type={self.task_type.value}, status={self.status.value})")


# Export key classes
__all__ = [
    'TaskStatus',
    'TaskPriority', 
    'TaskType',
    'TaskMetadata',
    'TaskDependencies',
    'TaskResources',
    'TaskResult',
    'TaskExecutionContext',
    'UnifiedTask'
]