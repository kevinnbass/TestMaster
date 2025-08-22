"""
Workflow Models - Core data structures for intelligent workflow management

This module provides comprehensive data models for the intelligent workflow engine,
including workflow definitions, task specifications, execution contexts, and 
performance metrics with autonomous optimization capabilities.

Key Components:
- Workflow definition structures with dependency management
- Task execution models with priority and resource allocation
- Performance metrics and optimization tracking
- Execution context and constraint management
- System capability modeling and load balancing
- Advanced scheduling algorithms with adaptive optimization
"""

import uuid
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json


class WorkflowStatus(Enum):
    """Status of workflow execution"""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OPTIMIZING = "optimizing"
    RETRYING = "retrying"


class TaskStatus(Enum):
    """Status of individual task execution"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    WAITING_DEPENDENCIES = "waiting_dependencies"


class TaskPriority(Enum):
    """Priority levels for task execution"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class OptimizationObjective(Enum):
    """Optimization objectives for workflow execution"""
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_RESOURCE_USAGE = "minimize_resource_usage"
    MAXIMIZE_QUALITY = "maximize_quality"
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    BALANCE_ALL = "balance_all"
    MINIMIZE_COST = "minimize_cost"


class SystemCapability(Enum):
    """System capabilities for workflow planning"""
    DATA_ANALYSIS = "data_analysis"
    PATTERN_DETECTION = "pattern_detection"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"
    RECOMMENDATION = "recommendation"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    REPORTING = "reporting"
    MONITORING = "monitoring"


class ResourceType(Enum):
    """Types of resources used by tasks"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"
    DATABASE = "database"
    API_CALLS = "api_calls"
    CONCURRENT_SLOTS = "concurrent_slots"


@dataclass
class TaskResource:
    """Resource requirements and usage for a task"""
    resource_type: ResourceType
    required_amount: float
    max_amount: Optional[float] = None
    current_usage: float = 0.0
    reserved_amount: float = 0.0
    unit: str = "units"
    
    def is_available(self, available_amount: float) -> bool:
        """Check if required resources are available"""
        return available_amount >= self.required_amount
    
    def allocate(self, amount: float) -> bool:
        """Allocate resources for task execution"""
        if amount >= self.required_amount:
            self.reserved_amount = min(amount, self.max_amount or amount)
            return True
        return False


@dataclass
class TaskConstraint:
    """Constraints that affect task execution"""
    constraint_type: str
    constraint_value: Any
    is_hard_constraint: bool = True
    violation_penalty: float = 0.0
    description: str = ""


@dataclass
class TaskDefinition:
    """Definition of a workflow task"""
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    task_type: str = "generic"
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Execution parameters
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Dependencies and relationships
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    can_run_parallel: bool = True
    
    # Resource requirements
    required_resources: List[TaskResource] = field(default_factory=list)
    required_capabilities: List[SystemCapability] = field(default_factory=list)
    estimated_duration_seconds: Optional[float] = None
    
    # Constraints
    constraints: List[TaskConstraint] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "workflow_engine"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Execution state and results of a task"""
    task_id: str
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    status: TaskStatus = TaskStatus.PENDING
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    assigned_system: Optional[str] = None
    
    # Results and output
    result: Any = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Retry tracking
    attempt_number: int = 1
    retry_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    success_score: float = 0.0
    quality_score: float = 0.0
    confidence_score: float = 0.0
    
    def mark_started(self, system_id: str = None):
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.assigned_system = system_id
    
    def mark_completed(self, result: Any = None, metrics: Dict = None):
        """Mark task as completed with results"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        if metrics:
            self.performance_metrics.update(metrics)
    
    def mark_failed(self, error: str, details: Dict = None):
        """Mark task as failed with error information"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error
        self.error_details = details or {}
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


@dataclass
class WorkflowDefinition:
    """Complete definition of a workflow"""
    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Workflow structure
    tasks: Dict[str, TaskDefinition] = field(default_factory=dict)
    task_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution configuration
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCE_ALL
    max_parallel_tasks: Optional[int] = None
    total_timeout_seconds: Optional[int] = None
    
    # Requirements and constraints
    required_capabilities: List[SystemCapability] = field(default_factory=list)
    workflow_constraints: List[TaskConstraint] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "workflow_designer"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_task(self, task: TaskDefinition, dependencies: List[str] = None):
        """Add a task to the workflow"""
        self.tasks[task.task_id] = task
        if dependencies:
            self.task_dependencies[task.task_id] = dependencies
            task.depends_on = dependencies
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """Get tasks that are ready to execute"""
        ready_tasks = []
        for task_id, task in self.tasks.items():
            if task_id not in completed_tasks:
                dependencies = self.task_dependencies.get(task_id, [])
                if all(dep in completed_tasks for dep in dependencies):
                    ready_tasks.append(task_id)
        return ready_tasks
    
    def validate(self) -> List[str]:
        """Validate workflow definition and return issues"""
        issues = []
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id):
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep in self.task_dependencies.get(task_id, []):
                if has_cycle(dep):
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.tasks:
            if has_cycle(task_id):
                issues.append(f"Circular dependency detected involving task {task_id}")
        
        # Check for missing dependencies
        for task_id, dependencies in self.task_dependencies.items():
            for dep in dependencies:
                if dep not in self.tasks:
                    issues.append(f"Task {task_id} depends on non-existent task {dep}")
        
        return issues


@dataclass
class WorkflowExecution:
    """Execution state and results of a workflow"""
    workflow_id: str
    execution_id: str = field(default_factory=lambda: f"wf_exec_{uuid.uuid4().hex[:12]}")
    status: WorkflowStatus = WorkflowStatus.CREATED
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Task executions
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    
    # Performance metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_task_count: int = 0
    average_task_duration: float = 0.0
    total_resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    overall_success_rate: float = 0.0
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    
    # Error tracking
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Results
    workflow_result: Any = None
    output_artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self):
        """Mark workflow execution as started"""
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self):
        """Mark workflow execution as completed"""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self._calculate_final_metrics()
    
    def mark_failed(self, error_message: str):
        """Mark workflow execution as failed"""
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.now()
        self.error_messages.append(error_message)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self._calculate_final_metrics()
    
    def add_task_execution(self, task_execution: TaskExecution):
        """Add a task execution to the workflow"""
        self.task_executions[task_execution.task_id] = task_execution
        if task_execution.status == TaskStatus.COMPLETED:
            self.completed_tasks.add(task_execution.task_id)
            self.successful_tasks += 1
        elif task_execution.status == TaskStatus.FAILED:
            self.failed_tasks.add(task_execution.task_id)
            self.failed_task_count += 1
    
    def _calculate_final_metrics(self):
        """Calculate final workflow metrics"""
        self.total_tasks = len(self.task_executions)
        
        if self.total_tasks > 0:
            self.overall_success_rate = self.successful_tasks / self.total_tasks
            
            # Calculate average task duration
            completed_durations = [
                exec.duration_seconds for exec in self.task_executions.values()
                if exec.duration_seconds is not None
            ]
            if completed_durations:
                self.average_task_duration = sum(completed_durations) / len(completed_durations)
            
            # Calculate quality score (weighted by success rate and performance)
            self.quality_score = (
                self.overall_success_rate * 0.6 +
                min(1.0, self.successful_tasks / max(1, self.total_tasks)) * 0.4
            )
            
            # Calculate efficiency score
            if self.duration_seconds and self.duration_seconds > 0:
                expected_duration = sum(
                    task.estimated_duration_seconds or 60
                    for task in self.task_executions.values()
                )
                self.efficiency_score = min(1.0, expected_duration / self.duration_seconds)


@dataclass
class SystemStatus:
    """Status and capabilities of a system that can execute tasks"""
    system_id: str
    system_name: str = ""
    is_available: bool = True
    current_load: float = 0.0
    max_load: float = 1.0
    
    # Capabilities
    supported_capabilities: List[SystemCapability] = field(default_factory=list)
    supported_task_types: List[str] = field(default_factory=list)
    
    # Resource availability
    available_resources: Dict[ResourceType, float] = field(default_factory=dict)
    reserved_resources: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Performance metrics
    average_task_duration: float = 60.0
    success_rate: float = 0.95
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Current assignments
    assigned_tasks: Set[str] = field(default_factory=set)
    task_queue: List[str] = field(default_factory=list)
    
    def can_execute_task(self, task: TaskDefinition) -> bool:
        """Check if system can execute the given task"""
        if not self.is_available:
            return False
        
        # Check capabilities
        for capability in task.required_capabilities:
            if capability not in self.supported_capabilities:
                return False
        
        # Check task type support
        if task.task_type not in self.supported_task_types and self.supported_task_types:
            return False
        
        # Check resource availability
        for resource in task.required_resources:
            available = self.available_resources.get(resource.resource_type, 0.0)
            if not resource.is_available(available):
                return False
        
        return True
    
    def get_load_score(self) -> float:
        """Get current load score (0.0 = no load, 1.0 = fully loaded)"""
        base_load = self.current_load / max(self.max_load, 0.01)
        queue_load = len(self.task_queue) * 0.1
        return min(1.0, base_load + queue_load)


@dataclass
class OptimizationMetrics:
    """Metrics for workflow optimization tracking"""
    optimization_id: str = field(default_factory=lambda: f"opt_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Before optimization metrics
    before_duration: Optional[float] = None
    before_resource_usage: Dict[str, float] = field(default_factory=dict)
    before_success_rate: float = 0.0
    
    # After optimization metrics
    after_duration: Optional[float] = None
    after_resource_usage: Dict[str, float] = field(default_factory=dict)
    after_success_rate: float = 0.0
    
    # Improvement calculations
    duration_improvement: float = 0.0
    resource_improvement: float = 0.0
    success_rate_improvement: float = 0.0
    overall_improvement: float = 0.0
    
    # Optimization details
    optimization_type: str = ""
    optimization_parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_description: str = ""
    
    def calculate_improvements(self):
        """Calculate improvement metrics"""
        if self.before_duration and self.after_duration:
            self.duration_improvement = (
                (self.before_duration - self.after_duration) / self.before_duration
            ) * 100
        
        self.success_rate_improvement = (
            self.after_success_rate - self.before_success_rate
        ) * 100
        
        # Calculate overall improvement as weighted average
        self.overall_improvement = (
            self.duration_improvement * 0.4 +
            self.success_rate_improvement * 0.6
        )


# Factory Functions
def create_task_definition(
    name: str,
    function: Callable = None,
    parameters: Dict = None,
    dependencies: List[str] = None,
    priority: TaskPriority = TaskPriority.MEDIUM,
    capabilities: List[SystemCapability] = None
) -> TaskDefinition:
    """Create a task definition with common parameters"""
    task = TaskDefinition(
        name=name,
        function=function,
        parameters=parameters or {},
        depends_on=dependencies or [],
        priority=priority,
        required_capabilities=capabilities or []
    )
    return task


def create_workflow_definition(
    name: str,
    description: str = "",
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCE_ALL
) -> WorkflowDefinition:
    """Create a workflow definition with basic configuration"""
    return WorkflowDefinition(
        name=name,
        description=description,
        optimization_objective=optimization_objective
    )


def create_system_status(
    system_id: str,
    system_name: str = "",
    capabilities: List[SystemCapability] = None,
    resources: Dict[ResourceType, float] = None
) -> SystemStatus:
    """Create a system status with basic configuration"""
    return SystemStatus(
        system_id=system_id,
        system_name=system_name or system_id,
        supported_capabilities=capabilities or [],
        available_resources=resources or {}
    )


# Utility Functions
def calculate_task_priority_score(task: TaskDefinition, context: Dict[str, Any] = None) -> float:
    """Calculate numerical priority score for task scheduling"""
    base_scores = {
        TaskPriority.CRITICAL: 1000.0,
        TaskPriority.HIGH: 800.0,
        TaskPriority.MEDIUM: 500.0,
        TaskPriority.LOW: 200.0,
        TaskPriority.BACKGROUND: 50.0
    }
    
    base_score = base_scores.get(task.priority, 500.0)
    
    # Adjust for deadline urgency if available
    if context and 'current_time' in context:
        if task.metadata.get('deadline'):
            deadline = task.metadata['deadline']
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline)
            time_remaining = (deadline - context['current_time']).total_seconds()
            if time_remaining > 0:
                urgency_multiplier = max(0.1, min(2.0, 3600 / time_remaining))
                base_score *= urgency_multiplier
    
    return base_score


def validate_workflow_consistency(workflow: WorkflowDefinition) -> List[str]:
    """Validate workflow for consistency and completeness"""
    issues = workflow.validate()
    
    # Additional validation checks
    if not workflow.tasks:
        issues.append("Workflow has no tasks defined")
    
    if not workflow.name:
        issues.append("Workflow has no name")
    
    # Check for orphaned tasks (tasks with no path to completion)
    all_dependencies = set()
    for deps in workflow.task_dependencies.values():
        all_dependencies.update(deps)
    
    entry_points = set(workflow.tasks.keys()) - all_dependencies
    if not entry_points:
        issues.append("Workflow has no entry point tasks (all tasks have dependencies)")
    
    return issues


# Constants and Configuration
DEFAULT_TASK_TIMEOUT = 300  # 5 minutes
DEFAULT_WORKFLOW_TIMEOUT = 3600  # 1 hour
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5

PRIORITY_WEIGHTS = {
    TaskPriority.CRITICAL: 1.0,
    TaskPriority.HIGH: 0.8,
    TaskPriority.MEDIUM: 0.6,
    TaskPriority.LOW: 0.4,
    TaskPriority.BACKGROUND: 0.2
}

RESOURCE_LIMITS = {
    ResourceType.CPU: 100.0,
    ResourceType.MEMORY: 32768.0,  # MB
    ResourceType.DISK_IO: 1000.0,  # MB/s
    ResourceType.NETWORK_IO: 1000.0,  # MB/s
    ResourceType.CONCURRENT_SLOTS: 10.0
}