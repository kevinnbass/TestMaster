"""
Workflow Types - Advanced Workflow Management Type Definitions
==============================================================

Comprehensive type definitions and data structures for intelligent workflow management,
task orchestration, and optimization with enterprise-grade execution patterns.
Implements advanced workflow management types for autonomous intelligence systems.

This module provides all type definitions, enums, and dataclasses required for
sophisticated workflow design, execution, and optimization across intelligence frameworks.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: workflow_types.py (220 lines)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


class WorkflowStatus(Enum):
    """Comprehensive workflow execution status with enterprise states"""
    PENDING = "pending"
    DESIGNING = "designing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OPTIMIZING = "optimizing"
    RECOVERING = "recovering"
    SUSPENDED = "suspended"


class TaskStatus(Enum):
    """Individual workflow task status with comprehensive tracking"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class WorkflowPriority(Enum):
    """Enterprise workflow priority levels with numeric ordering"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class OptimizationObjective(Enum):
    """Advanced optimization objectives for workflow performance"""
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_RESOURCES = "minimize_resources"
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    BALANCE_ALL = "balance_all"
    CUSTOM = "custom"


class WorkflowDesignMode(Enum):
    """Workflow design mode for intelligent generation"""
    MANUAL = "manual"
    TEMPLATE_BASED = "template_based"
    AI_GENERATED = "ai_generated"
    HYBRID = "hybrid"
    LEARNED = "learned"


class ExecutionStrategy(Enum):
    """Workflow execution strategies for different scenarios"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    PRIORITY_DRIVEN = "priority_driven"
    RESOURCE_OPTIMIZED = "resource_optimized"
    FAULT_TOLERANT = "fault_tolerant"


@dataclass
class WorkflowTask:
    """Advanced individual task within workflow with enterprise features"""
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
    timeout: Optional[float] = None
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    
    # Execution tracking
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Optimization metadata
    optimization_score: float = 0.0
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_efficiency(self) -> float:
        """Calculate task execution efficiency"""
        if self.estimated_duration <= 0 or self.actual_duration is None:
            return 0.0
        
        # Efficiency = estimated / actual (higher is better)
        efficiency = self.estimated_duration / self.actual_duration
        
        # Adjust for retry penalty
        retry_penalty = 1.0 - (self.retry_count * 0.1)
        
        return max(0.0, min(2.0, efficiency * retry_penalty))
    
    def is_ready_to_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready for execution based on dependencies"""
        return (self.status == TaskStatus.PENDING and 
                all(dep in completed_tasks for dep in self.dependencies))


@dataclass
class WorkflowDefinition:
    """Comprehensive workflow definition with enterprise capabilities"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCE_ALL
    design_mode: WorkflowDesignMode = WorkflowDesignMode.MANUAL
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    
    # Execution constraints
    max_parallel_tasks: int = 5
    total_timeout: Optional[float] = None
    max_retries: int = 3
    
    # Success criteria
    success_criteria: List[str] = field(default_factory=list)
    minimum_success_rate: float = 0.8
    required_capabilities: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resource requirements
    resource_budget: Dict[str, float] = field(default_factory=dict)
    cost_constraints: Dict[str, float] = field(default_factory=dict)
    
    def calculate_complexity_score(self) -> float:
        """Calculate workflow complexity score"""
        base_complexity = len(self.tasks)
        
        # Add complexity for dependencies
        total_dependencies = sum(len(task.dependencies) for task in self.tasks)
        dependency_complexity = total_dependencies * 0.5
        
        # Add complexity for different task types
        unique_systems = len(set(task.target_system for task in self.tasks))
        system_complexity = unique_systems * 0.3
        
        # Add complexity for resource requirements
        resource_complexity = len(self.resource_budget) * 0.2
        
        return base_complexity + dependency_complexity + system_complexity + resource_complexity
    
    def get_critical_path_duration(self) -> float:
        """Calculate critical path duration estimate"""
        # Simplified critical path calculation
        # In practice, this would use network analysis algorithms
        
        if not self.tasks:
            return 0.0
        
        # For now, assume sequential execution of longest path
        max_duration = 0.0
        for task in self.tasks:
            # Add task duration plus dependency chain
            task_path_duration = task.estimated_duration
            # This is simplified - would need proper topological analysis
            max_duration = max(max_duration, task_path_duration)
        
        return max_duration * len(self.tasks) * 0.7  # Estimate with parallelism


@dataclass
class WorkflowExecution:
    """Comprehensive workflow execution tracking with enterprise monitoring"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    progress: float = 0.0
    
    # Execution tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Task tracking
    scheduled_tasks: List[str] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    skipped_tasks: List[str] = field(default_factory=list)
    
    # Results and errors
    task_results: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    execution_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    resource_usage: Dict[str, float] = field(default_factory=dict)
    cost_incurred: float = 0.0
    
    # Performance tracking
    throughput_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def calculate_success_rate(self) -> float:
        """Calculate overall execution success rate"""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks == 0:
            return 0.0
        
        return len(self.completed_tasks) / total_tasks
    
    def get_execution_duration(self) -> Optional[float]:
        """Get total execution duration in seconds"""
        if self.started_at is None:
            return None
        
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall execution efficiency score"""
        success_rate = self.calculate_success_rate()
        
        # Time efficiency (if available)
        time_efficiency = 1.0
        if 'estimated_duration' in self.execution_metrics and 'total_duration' in self.execution_metrics:
            estimated = self.execution_metrics['estimated_duration']
            actual = self.execution_metrics['total_duration']
            if actual > 0:
                time_efficiency = estimated / actual
        
        # Resource efficiency
        resource_efficiency = 1.0
        if self.resource_usage:
            # Simplified efficiency calculation
            avg_utilization = np.mean(list(self.resource_usage.values()))
            resource_efficiency = min(1.0, avg_utilization / 0.8)  # Target 80% utilization
        
        # Combined efficiency score
        return (success_rate * 0.5 + time_efficiency * 0.3 + resource_efficiency * 0.2)


@dataclass
class WorkflowOptimization:
    """Comprehensive workflow optimization result with enterprise insights"""
    optimization_id: str
    workflow_id: str
    objective: OptimizationObjective
    created_at: datetime = field(default_factory=datetime.now)
    
    # Performance comparison
    original_performance: Dict[str, float] = field(default_factory=dict)
    optimized_performance: Dict[str, float] = field(default_factory=dict)
    improvement_percentage: float = 0.0
    
    # Optimization details
    optimization_changes: List[str] = field(default_factory=list)
    implementation_effort: float = 0.0  # 0-1 scale
    confidence_score: float = 0.0  # 0-1 scale
    
    # Impact analysis
    cost_impact: float = 0.0
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    resource_impact: Dict[str, float] = field(default_factory=dict)
    
    # Recommendation metadata
    recommended_actions: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    rollback_plan: List[str] = field(default_factory=list)
    
    def calculate_value_score(self) -> float:
        """Calculate overall optimization value score"""
        # Combine improvement with implementation difficulty
        improvement_value = self.improvement_percentage
        implementation_penalty = self.implementation_effort * 0.3
        
        value_score = improvement_value - implementation_penalty
        
        # Adjust for confidence
        confidence_factor = self.confidence_score
        
        return max(0.0, value_score * confidence_factor)
    
    def is_recommended(self, min_improvement: float = 0.1, max_effort: float = 0.7) -> bool:
        """Check if optimization is recommended based on criteria"""
        return (self.improvement_percentage >= min_improvement and 
                self.implementation_effort <= max_effort and
                self.confidence_score >= 0.6)


@dataclass
class WorkflowTemplate:
    """Reusable workflow template for common patterns"""
    template_id: str
    name: str
    description: str
    category: str
    task_templates: List[Dict[str, Any]]
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[str] = field(default_factory=list)
    optimization_hints: List[str] = field(default_factory=list)
    
    # Template metadata
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    average_duration: float = 0.0
    
    def instantiate_workflow(self, parameters: Dict[str, Any] = None) -> WorkflowDefinition:
        """Create workflow instance from template"""
        merged_params = {**self.default_parameters, **(parameters or {})}
        
        # Create tasks from templates
        tasks = []
        for i, task_template in enumerate(self.task_templates):
            task = WorkflowTask(
                task_id=f"{self.template_id}_task_{i}",
                name=task_template.get('name', f"Task {i+1}"),
                task_type=task_template.get('task_type', 'generic'),
                target_system=task_template.get('target_system', 'default'),
                parameters={**task_template.get('parameters', {}), **merged_params},
                dependencies=task_template.get('dependencies', []),
                estimated_duration=task_template.get('estimated_duration', 10.0)
            )
            tasks.append(task)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=f"workflow_{uuid.uuid4().hex[:8]}",
            name=f"{self.name} Instance",
            description=f"Instance of {self.name} template",
            tasks=tasks,
            required_capabilities=self.required_capabilities.copy()
        )
        
        return workflow


@dataclass
class SystemCapabilities:
    """System capabilities for workflow task assignment"""
    system_id: str
    available_capabilities: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_capacity: Dict[str, float] = field(default_factory=dict)
    current_load: float = 0.0
    reliability_score: float = 1.0
    cost_per_task: float = 1.0
    
    def can_execute_task(self, task: WorkflowTask) -> bool:
        """Check if system can execute the given task"""
        # Check if system has required capabilities
        task_capabilities = task.metadata.get('required_capabilities', [])
        return all(cap in self.available_capabilities for cap in task_capabilities)
    
    def calculate_suitability_score(self, task: WorkflowTask, objective: OptimizationObjective) -> float:
        """Calculate suitability score for executing task based on objective"""
        if not self.can_execute_task(task):
            return 0.0
        
        base_score = 1.0
        
        # Adjust based on optimization objective
        if objective == OptimizationObjective.MINIMIZE_TIME:
            speed_score = self.performance_metrics.get('speed', 1.0)
            load_penalty = self.current_load * 0.5
            base_score = speed_score * (1.0 - load_penalty)
        elif objective == OptimizationObjective.MINIMIZE_COST:
            cost_score = 1.0 / max(0.1, self.cost_per_task)
            base_score = cost_score
        elif objective == OptimizationObjective.MAXIMIZE_RELIABILITY:
            base_score = self.reliability_score * (1.0 - self.current_load * 0.3)
        else:  # Balanced approach
            speed = self.performance_metrics.get('speed', 1.0)
            cost = 1.0 / max(0.1, self.cost_per_task)
            reliability = self.reliability_score
            load_factor = 1.0 - self.current_load * 0.4
            
            base_score = (speed * 0.3 + cost * 0.2 + reliability * 0.3 + load_factor * 0.2)
        
        return max(0.0, min(1.0, base_score))


# Export all workflow types and enums
__all__ = [
    'WorkflowStatus', 'TaskStatus', 'WorkflowPriority', 'OptimizationObjective',
    'WorkflowDesignMode', 'ExecutionStrategy', 'WorkflowTask', 'WorkflowDefinition',
    'WorkflowExecution', 'WorkflowOptimization', 'WorkflowTemplate', 'SystemCapabilities'
]