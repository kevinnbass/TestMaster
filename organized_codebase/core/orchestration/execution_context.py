"""
Execution Context
================

Unified execution context management providing consistent execution
environment across all orchestration systems in TestMaster.

Author: Agent E - Infrastructure Consolidation
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Set
from pathlib import Path


class ExecutionMode(Enum):
    """Execution mode for orchestration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"
    INTELLIGENT = "intelligent"
    HYBRID = "hybrid"


class ExecutionPriority(Enum):
    """Execution priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class ExecutionEnvironment(Enum):
    """Execution environment types."""
    LOCAL = "local"
    CONTAINER = "container"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    EDGE = "edge"


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_duration: Optional[timedelta] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: float = 0.0
    disk_io: float = 0.0
    tasks_executed: int = 0
    tasks_successful: int = 0
    tasks_failed: int = 0
    error_count: int = 0
    warnings_count: int = 0


@dataclass
class ExecutionResources:
    """Execution resource allocation."""
    cpu_cores: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    disk_space_mb: Optional[int] = None
    network_bandwidth: Optional[int] = None
    gpu_allocation: Optional[int] = None
    custom_resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionConstraints:
    """Execution constraints and limits."""
    max_execution_time: Optional[timedelta] = None
    max_memory_usage: Optional[int] = None
    max_cpu_usage: Optional[float] = None
    allowed_network_access: bool = True
    allowed_file_access: bool = True
    security_restrictions: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)


class ExecutionState:
    """Execution state management."""
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.shared_data: Dict[str, Any] = {}
        self.temporary_data: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
    
    def set_variable(self, name: str, value: Any, persistent: bool = True):
        """Set execution variable."""
        self.variables[name] = value
        if persistent:
            self._save_checkpoint(f"var_{name}")
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get execution variable."""
        return self.variables.get(name, default)
    
    def set_shared_data(self, key: str, value: Any):
        """Set shared data accessible by all components."""
        self.shared_data[key] = value
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data."""
        return self.shared_data.get(key, default)
    
    def create_checkpoint(self, name: str):
        """Create state checkpoint."""
        self.checkpoints[name] = {
            'variables': self.variables.copy(),
            'shared_data': self.shared_data.copy(),
            'timestamp': datetime.now()
        }
    
    def restore_checkpoint(self, name: str) -> bool:
        """Restore state from checkpoint."""
        if name in self.checkpoints:
            checkpoint = self.checkpoints[name]
            self.variables = checkpoint['variables'].copy()
            self.shared_data = checkpoint['shared_data'].copy()
            return True
        return False
    
    def _save_checkpoint(self, identifier: str):
        """Save automatic checkpoint."""
        self.state_history.append({
            'identifier': identifier,
            'variables': self.variables.copy(),
            'timestamp': datetime.now()
        })


class ExecutionContext:
    """
    Unified execution context providing consistent execution environment
    across all orchestration systems.
    
    Manages execution state, resources, constraints, and environmental
    configuration for orchestrated task execution.
    """
    
    def __init__(
        self,
        context_id: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL
    ):
        self.context_id = context_id or f"ctx_{uuid.uuid4().hex[:8]}"
        self.execution_mode = execution_mode
        self.priority = priority
        self.environment = environment
        
        # Context metadata
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.name: Optional[str] = None
        self.description: Optional[str] = None
        self.tags: Set[str] = set()
        self.labels: Dict[str, str] = {}
        
        # Execution configuration
        self.resources = ExecutionResources()
        self.constraints = ExecutionConstraints()
        self.metrics = ExecutionMetrics()
        
        # State management
        self.state = ExecutionState()
        self.configuration: Dict[str, Any] = {}
        self.environment_vars: Dict[str, str] = {}
        
        # Context hierarchy
        self.parent_context: Optional['ExecutionContext'] = None
        self.child_contexts: List['ExecutionContext'] = []
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.lifecycle_handlers: List[Callable] = []
        
        # Execution tracking
        self.execution_log: List[Dict[str, Any]] = []
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Working directory and file management
        self.working_directory: Optional[Path] = None
        self.input_files: List[Path] = []
        self.output_files: List[Path] = []
        self.temporary_files: List[Path] = []
    
    # ========================================================================
    # CONTEXT LIFECYCLE MANAGEMENT
    # ========================================================================
    
    async def initialize(self) -> bool:
        """Initialize execution context."""
        try:
            self.started_at = datetime.now()
            self.metrics.start_time = self.started_at
            
            # Initialize working directory
            if not self.working_directory:
                self.working_directory = Path.cwd()
            
            # Initialize environment
            await self._initialize_environment()
            
            # Setup resource allocation
            await self._allocate_resources()
            
            # Initialize state
            self.state.create_checkpoint('initialization')
            
            self._emit_lifecycle_event('context_initialized')
            self._log_execution_event('context_initialized')
            
            return True
            
        except Exception as e:
            self._log_execution_event('initialization_failed', {'error': str(e)})
            return False
    
    async def finalize(self) -> bool:
        """Finalize execution context."""
        try:
            self.completed_at = datetime.now()
            self.metrics.end_time = self.completed_at
            
            if self.started_at:
                self.metrics.execution_duration = self.completed_at - self.started_at
            
            # Cleanup resources
            await self._cleanup_resources()
            
            # Cleanup temporary files
            await self._cleanup_temporary_files()
            
            # Finalize child contexts
            for child_context in self.child_contexts:
                await child_context.finalize()
            
            self._emit_lifecycle_event('context_finalized')
            self._log_execution_event('context_finalized')
            
            return True
            
        except Exception as e:
            self._log_execution_event('finalization_failed', {'error': str(e)})
            return False
    
    # ========================================================================
    # CONTEXT HIERARCHY MANAGEMENT
    # ========================================================================
    
    def create_child_context(
        self,
        name: Optional[str] = None,
        execution_mode: Optional[ExecutionMode] = None,
        inherit_state: bool = True
    ) -> 'ExecutionContext':
        """Create child execution context."""
        child_context = ExecutionContext(
            execution_mode=execution_mode or self.execution_mode,
            priority=self.priority,
            environment=self.environment
        )
        
        child_context.name = name
        child_context.parent_context = self
        self.child_contexts.append(child_context)
        
        if inherit_state:
            # Inherit configuration and environment
            child_context.configuration.update(self.configuration)
            child_context.environment_vars.update(self.environment_vars)
            
            # Inherit state variables (copy, not reference)
            for key, value in self.state.variables.items():
                child_context.state.set_variable(key, value)
        
        self._emit_event('child_context_created', {'child_id': child_context.context_id})
        return child_context
    
    def get_root_context(self) -> 'ExecutionContext':
        """Get root context in hierarchy."""
        current = self
        while current.parent_context:
            current = current.parent_context
        return current
    
    def get_context_depth(self) -> int:
        """Get depth in context hierarchy."""
        depth = 0
        current = self.parent_context
        while current:
            depth += 1
            current = current.parent_context
        return depth
    
    # ========================================================================
    # RESOURCE MANAGEMENT
    # ========================================================================
    
    def set_resource_limits(
        self,
        cpu_cores: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        execution_timeout: Optional[timedelta] = None
    ):
        """Set resource limits for execution."""
        if cpu_cores is not None:
            self.resources.cpu_cores = cpu_cores
        if memory_limit_mb is not None:
            self.resources.memory_limit_mb = memory_limit_mb
        if execution_timeout is not None:
            self.constraints.max_execution_time = execution_timeout
    
    def allocate_custom_resource(self, resource_name: str, allocation: Any):
        """Allocate custom resource."""
        self.resources.custom_resources[resource_name] = allocation
    
    async def _allocate_resources(self):
        """Allocate execution resources."""
        # Resource allocation logic would be implemented here
        # This is a placeholder for actual resource allocation
        pass
    
    async def _cleanup_resources(self):
        """Cleanup allocated resources."""
        # Resource cleanup logic would be implemented here
        pass
    
    # ========================================================================
    # EXECUTION MANAGEMENT
    # ========================================================================
    
    def start_task(self, task_id: str):
        """Mark task as started."""
        self.active_tasks.add(task_id)
        self.metrics.tasks_executed += 1
        self._log_execution_event('task_started', {'task_id': task_id})
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark task as completed."""
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
            
            if success:
                self.completed_tasks.add(task_id)
                self.metrics.tasks_successful += 1
            else:
                self.failed_tasks.add(task_id)
                self.metrics.tasks_failed += 1
            
            self._log_execution_event(
                'task_completed',
                {'task_id': task_id, 'success': success}
            )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            'context_id': self.context_id,
            'name': self.name,
            'execution_mode': self.execution_mode.value,
            'priority': self.priority.value,
            'environment': self.environment.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration': str(self.metrics.execution_duration) if self.metrics.execution_duration else None,
            'tasks': {
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks),
                'total': self.metrics.tasks_executed
            },
            'child_contexts': len(self.child_contexts),
            'context_depth': self.get_context_depth()
        }
    
    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================
    
    def set_configuration(self, config: Dict[str, Any]):
        """Set execution configuration."""
        self.configuration.update(config)
        self._emit_event('configuration_updated', {'config': config})
    
    def get_configuration(self, key: Optional[str] = None) -> Any:
        """Get configuration value."""
        if key:
            return self.configuration.get(key)
        return self.configuration.copy()
    
    def set_environment_variable(self, name: str, value: str):
        """Set environment variable."""
        self.environment_vars[name] = value
    
    def get_environment_variable(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable."""
        return self.environment_vars.get(name, default)
    
    # ========================================================================
    # FILE AND DIRECTORY MANAGEMENT
    # ========================================================================
    
    def set_working_directory(self, path: Union[str, Path]):
        """Set working directory."""
        self.working_directory = Path(path)
        self._emit_event('working_directory_changed', {'path': str(path)})
    
    def add_input_file(self, file_path: Union[str, Path]):
        """Add input file."""
        self.input_files.append(Path(file_path))
    
    def add_output_file(self, file_path: Union[str, Path]):
        """Add output file."""
        self.output_files.append(Path(file_path))
    
    def create_temporary_file(self, suffix: str = "") -> Path:
        """Create temporary file."""
        temp_file = Path.cwd() / f"temp_{self.context_id}_{len(self.temporary_files)}{suffix}"
        self.temporary_files.append(temp_file)
        return temp_file
    
    async def _cleanup_temporary_files(self):
        """Cleanup temporary files."""
        for temp_file in self.temporary_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors
    
    # ========================================================================
    # EVENT HANDLING
    # ========================================================================
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def add_lifecycle_handler(self, handler: Callable):
        """Add lifecycle event handler."""
        self.lifecycle_handlers.append(handler)
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any] = None):
        """Emit event to registered handlers."""
        event_data = event_data or {}
        event_data.update({
            'context_id': self.context_id,
            'timestamp': datetime.now()
        })
        
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event_type, event_data)
            except Exception:
                pass  # Silent fail for event handlers
    
    def _emit_lifecycle_event(self, event_type: str, event_data: Dict[str, Any] = None):
        """Emit lifecycle event."""
        event_data = event_data or {}
        
        for handler in self.lifecycle_handlers:
            try:
                handler(event_type, event_data)
            except Exception:
                pass  # Silent fail for event handlers
        
        self._emit_event(event_type, event_data)
    
    # ========================================================================
    # LOGGING AND MONITORING
    # ========================================================================
    
    def _log_execution_event(self, event_type: str, event_data: Dict[str, Any] = None):
        """Log execution event."""
        self.execution_log.append({
            'timestamp': datetime.now(),
            'event_type': event_type,
            'event_data': event_data or {},
            'context_id': self.context_id
        })
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log."""
        return self.execution_log.copy()
    
    async def _initialize_environment(self):
        """Initialize execution environment."""
        # Environment initialization logic would be implemented here
        pass
    
    def __str__(self) -> str:
        return f"ExecutionContext({self.context_id})"
    
    def __repr__(self) -> str:
        return (f"ExecutionContext(id={self.context_id}, mode={self.execution_mode.value}, "
                f"priority={self.priority.value}, environment={self.environment.value})")


# Export key classes
__all__ = [
    'ExecutionMode',
    'ExecutionPriority',
    'ExecutionEnvironment',
    'ExecutionMetrics',
    'ExecutionResources',
    'ExecutionConstraints',
    'ExecutionState',
    'ExecutionContext'
]