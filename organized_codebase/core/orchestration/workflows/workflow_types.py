"""
Workflow Type Definitions and Enumerations

This module defines the core data types, enumerations, and data classes used
throughout the workflow system for status tracking, task management, and
workflow configuration.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import networkx as nx


class WorkflowStatus(Enum):
    """Status of workflow execution"""
    PENDING = "pending"
    DESIGNING = "designing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OPTIMIZING = "optimizing"


class TaskStatus(Enum):
    """Status of individual workflow tasks"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowPriority(Enum):
    """Priority levels for workflows"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class OptimizationObjective(Enum):
    """Objectives for workflow optimization"""
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCE_ALL = "balance_all"


@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    task_id: str
    name: str
    task_type: str
    target_system: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    estimated_duration: float = 0.0  # seconds
    actual_duration: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0  # seconds
    priority: int = 5
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    execution_graph: nx.DiGraph
    priority: WorkflowPriority
    optimization_objective: OptimizationObjective
    max_parallel_tasks: int = 5
    total_timeout: float = 3600.0  # seconds
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowExecution:
    """Runtime execution state of a workflow"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    task_results: Dict[str, Any] = field(default_factory=dict)
    execution_metrics: Dict[str, float] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0


@dataclass
class WorkflowOptimization:
    """Optimization results for a workflow"""
    optimization_id: str
    workflow_id: str
    objective: OptimizationObjective
    original_performance: Dict[str, float]
    optimized_performance: Dict[str, float]
    optimization_changes: List[str]
    improvement_percentage: float
    implementation_effort: float
    created_at: datetime = field(default_factory=datetime.now)