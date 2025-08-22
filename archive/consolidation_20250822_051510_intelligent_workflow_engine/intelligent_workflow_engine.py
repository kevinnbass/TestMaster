"""
Intelligent Workflow Engine - Automated Workflow Management

This module implements the IntelligentWorkflowEngine system, providing autonomous
workflow management that coordinates complex multi-system intelligence operations
with self-optimization and adaptive execution capabilities.

Features:
- Self-designing workflows based on requirements and system capabilities
- Adaptive workflow execution with real-time optimization
- Intelligent task distribution and load balancing
- Automated error recovery and retry mechanisms
- Performance-driven workflow optimization
- Cross-system result aggregation and synthesis

Author: Agent A - Hour 36 - Intelligent Workflow Engine
Created: 2025-01-21
Enhanced with: Autonomous workflow intelligence, adaptive execution
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable, Type
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import yaml
import threading
from collections import defaultdict, deque
import statistics
import hashlib
import time
import uuid
from queue import PriorityQueue
import heapq

# Configure logging for intelligent workflow engine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class WorkflowDesigner:
    """Designs workflows based on requirements and available systems"""
    
    def __init__(self):
        self.design_templates: Dict[str, Dict[str, Any]] = {}
        self.system_capabilities: Dict[str, List[str]] = {}
        self.performance_models: Dict[str, Dict[str, float]] = {}
        
        logger.info("WorkflowDesigner initialized")
    
    def register_system_capabilities(self, system_id: str, capabilities: List[str],
                                   performance_metrics: Dict[str, float]):
        """Register system capabilities for workflow design"""
        self.system_capabilities[system_id] = capabilities
        self.performance_models[system_id] = performance_metrics
        logger.info(f"Registered capabilities for {system_id}: {len(capabilities)} capabilities")
    
    async def design_workflow(self, requirements: Dict[str, Any],
                            constraints: Dict[str, Any] = None) -> WorkflowDefinition:
        """Design a workflow based on requirements"""
        logger.info(f"Designing workflow for: {requirements.get('objective', 'unknown')}")
        
        if constraints is None:
            constraints = {}
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        # Analyze requirements
        required_capabilities = self._analyze_requirements(requirements)
        
        # Select systems and create tasks
        tasks = await self._create_tasks(required_capabilities, requirements, constraints)
        
        # Build execution graph
        execution_graph = self._build_execution_graph(tasks, requirements)
        
        # Determine optimization objective
        optimization_objective = self._determine_optimization_objective(requirements)
        
        # Set workflow properties
        priority = self._determine_priority(requirements)
        max_parallel_tasks = constraints.get("max_parallel_tasks", 5)
        total_timeout = constraints.get("total_timeout", 3600.0)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=requirements.get("name", f"Workflow_{workflow_id}"),
            description=requirements.get("description", "Auto-generated workflow"),
            tasks=tasks,
            execution_graph=execution_graph,
            priority=priority,
            optimization_objective=optimization_objective,
            max_parallel_tasks=max_parallel_tasks,
            total_timeout=total_timeout,
            retry_policy=self._create_retry_policy(requirements),
            success_criteria=self._define_success_criteria(requirements),
            metadata=requirements.get("metadata", {})
        )
        
        logger.info(f"Designed workflow {workflow_id} with {len(tasks)} tasks")
        return workflow
    
    def _analyze_requirements(self, requirements: Dict[str, Any]) -> List[str]:
        """Analyze requirements to determine needed capabilities"""
        required_capabilities = []
        
        objective = requirements.get("objective", "").lower()
        
        # Map objectives to capabilities
        if "analyze" in objective or "analysis" in objective:
            required_capabilities.extend(["data_analysis", "statistical_analysis"])
        
        if "predict" in objective or "forecast" in objective:
            required_capabilities.extend(["prediction", "forecasting", "modeling"])
        
        if "optimize" in objective or "improvement" in objective:
            required_capabilities.extend(["optimization", "tuning"])
        
        if "pattern" in objective or "detect" in objective:
            required_capabilities.extend(["pattern_detection", "anomaly_detection"])
        
        if "recommend" in objective or "suggest" in objective:
            required_capabilities.extend(["recommendation", "decision_support"])
        
        # Add explicit capability requirements
        if "required_capabilities" in requirements:
            required_capabilities.extend(requirements["required_capabilities"])
        
        # Remove duplicates and return
        return list(set(required_capabilities))
    
    async def _create_tasks(self, required_capabilities: List[str],
                          requirements: Dict[str, Any],
                          constraints: Dict[str, Any]) -> List[WorkflowTask]:
        """Create tasks based on required capabilities"""
        tasks = []
        task_counter = 1
        
        for capability in required_capabilities:
            # Find systems that provide this capability
            suitable_systems = []
            for system_id, capabilities in self.system_capabilities.items():
                if capability in capabilities:
                    suitable_systems.append(system_id)
            
            if suitable_systems:
                # Select best system for this capability
                best_system = self._select_best_system(suitable_systems, capability, constraints)
                
                # Create task
                task = WorkflowTask(
                    task_id=f"task_{task_counter:03d}",
                    name=f"{capability}_task",
                    task_type=capability,
                    target_system=best_system,
                    parameters=self._create_task_parameters(capability, requirements),
                    estimated_duration=self._estimate_task_duration(best_system, capability),
                    timeout=constraints.get("task_timeout", 300.0),
                    max_retries=constraints.get("max_retries", 3),
                    priority=self._calculate_task_priority(capability, requirements)
                )
                
                tasks.append(task)
                task_counter += 1
        
        # Add dependencies between tasks
        tasks = self._add_task_dependencies(tasks, requirements)
        
        return tasks
    
    def _select_best_system(self, suitable_systems: List[str], capability: str,
                          constraints: Dict[str, Any]) -> str:
        """Select the best system for a capability"""
        if len(suitable_systems) == 1:
            return suitable_systems[0]
        
        # Score systems based on performance metrics
        best_system = suitable_systems[0]
        best_score = 0.0
        
        for system_id in suitable_systems:
            score = self._calculate_system_score(system_id, capability, constraints)
            if score > best_score:
                best_score = score
                best_system = system_id
        
        return best_system
    
    def _calculate_system_score(self, system_id: str, capability: str,
                              constraints: Dict[str, Any]) -> float:
        """Calculate score for a system for a specific capability"""
        if system_id not in self.performance_models:
            return 0.5  # Default score
        
        performance = self.performance_models[system_id]
        score = 0.0
        
        # Factor in response time (lower is better)
        response_time = performance.get("response_time", 1000)
        score += max(0, (2000 - response_time) / 2000) * 0.3
        
        # Factor in availability (higher is better)
        availability = performance.get("availability", 0.95)
        score += availability * 0.3
        
        # Factor in throughput (higher is better)
        throughput = performance.get("throughput", 100)
        score += min(throughput / 1000, 1.0) * 0.2
        
        # Factor in error rate (lower is better)
        error_rate = performance.get("error_rate", 0.05)
        score += max(0, (0.1 - error_rate) / 0.1) * 0.2
        
        # Apply constraint penalties
        if constraints.get("max_response_time") and response_time > constraints["max_response_time"]:
            score *= 0.5
        
        if constraints.get("min_availability") and availability < constraints["min_availability"]:
            score *= 0.3
        
        return score
    
    def _create_task_parameters(self, capability: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create parameters for a task based on capability and requirements"""
        base_parameters = {
            "capability": capability,
            "timeout": 300,
            "priority": "medium"
        }
        
        # Add capability-specific parameters
        if capability in ["data_analysis", "statistical_analysis"]:
            base_parameters.update({
                "analysis_type": requirements.get("analysis_type", "comprehensive"),
                "include_visualization": requirements.get("include_visualization", False)
            })
        
        elif capability in ["prediction", "forecasting"]:
            base_parameters.update({
                "prediction_horizon": requirements.get("prediction_horizon", "short_term"),
                "confidence_level": requirements.get("confidence_level", 0.95)
            })
        
        elif capability in ["optimization", "tuning"]:
            base_parameters.update({
                "optimization_target": requirements.get("optimization_target", "performance"),
                "max_iterations": requirements.get("max_iterations", 100)
            })
        
        # Add common parameters from requirements
        if "input_data" in requirements:
            base_parameters["input_data"] = requirements["input_data"]
        
        if "output_format" in requirements:
            base_parameters["output_format"] = requirements["output_format"]
        
        return base_parameters
    
    def _estimate_task_duration(self, system_id: str, capability: str) -> float:
        """Estimate duration for a task"""
        if system_id not in self.performance_models:
            return 60.0  # Default 1 minute
        
        performance = self.performance_models[system_id]
        base_time = performance.get("response_time", 1000) / 1000  # Convert to seconds
        
        # Adjust based on capability complexity
        complexity_multipliers = {
            "data_analysis": 2.0,
            "prediction": 3.0,
            "optimization": 4.0,
            "pattern_detection": 1.5,
            "recommendation": 1.0
        }
        
        multiplier = complexity_multipliers.get(capability, 2.0)
        return base_time * multiplier
    
    def _calculate_task_priority(self, capability: str, requirements: Dict[str, Any]) -> int:
        """Calculate priority for a task"""
        base_priority = 5  # Medium priority
        
        # Adjust based on capability importance
        critical_capabilities = ["optimization", "decision_support", "prediction"]
        if capability in critical_capabilities:
            base_priority = 2  # High priority
        
        # Adjust based on requirements
        if requirements.get("priority") == "critical":
            base_priority = 1
        elif requirements.get("priority") == "high":
            base_priority = 2
        elif requirements.get("priority") == "low":
            base_priority = 4
        
        return base_priority
    
    def _add_task_dependencies(self, tasks: List[WorkflowTask],
                             requirements: Dict[str, Any]) -> List[WorkflowTask]:
        """Add dependencies between tasks based on logical flow"""
        
        # Create dependency mapping based on task types
        dependency_rules = {
            "data_analysis": [],  # Usually first
            "pattern_detection": ["data_analysis"],
            "prediction": ["data_analysis", "pattern_detection"],
            "optimization": ["data_analysis", "prediction"],
            "recommendation": ["optimization", "prediction"]
        }
        
        # Build task lookup
        task_lookup = {task.task_type: task for task in tasks}
        
        # Apply dependency rules
        for task in tasks:
            required_predecessors = dependency_rules.get(task.task_type, [])
            for predecessor_type in required_predecessors:
                if predecessor_type in task_lookup:
                    predecessor_task = task_lookup[predecessor_type]
                    if predecessor_task.task_id not in task.dependencies:
                        task.dependencies.append(predecessor_task.task_id)
        
        return tasks
    
    def _build_execution_graph(self, tasks: List[WorkflowTask],
                             requirements: Dict[str, Any]) -> nx.DiGraph:
        """Build execution graph from tasks and dependencies"""
        graph = nx.DiGraph()
        
        # Add nodes for tasks
        for task in tasks:
            graph.add_node(task.task_id, **{
                "name": task.name,
                "task_type": task.task_type,
                "target_system": task.target_system,
                "estimated_duration": task.estimated_duration,
                "priority": task.priority
            })
        
        # Add edges for dependencies
        for task in tasks:
            for dependency in task.dependencies:
                graph.add_edge(dependency, task.task_id, relationship="dependency")
        
        # Add start and end nodes
        graph.add_node("start", type="control")
        graph.add_node("end", type="control")
        
        # Connect start to tasks with no dependencies
        for task in tasks:
            if not task.dependencies:
                graph.add_edge("start", task.task_id, relationship="start")
        
        # Connect tasks with no successors to end
        for task in tasks:
            has_successors = any(task.task_id in other_task.dependencies for other_task in tasks)
            if not has_successors:
                graph.add_edge(task.task_id, "end", relationship="end")
        
        return graph
    
    def _determine_optimization_objective(self, requirements: Dict[str, Any]) -> OptimizationObjective:
        """Determine optimization objective from requirements"""
        explicit_objective = requirements.get("optimization_objective")
        if explicit_objective:
            return OptimizationObjective(explicit_objective)
        
        # Infer from requirements
        if requirements.get("time_critical", False):
            return OptimizationObjective.MINIMIZE_TIME
        elif requirements.get("accuracy_critical", False):
            return OptimizationObjective.MAXIMIZE_ACCURACY
        elif requirements.get("cost_sensitive", False):
            return OptimizationObjective.MINIMIZE_COST
        elif requirements.get("high_throughput", False):
            return OptimizationObjective.MAXIMIZE_THROUGHPUT
        else:
            return OptimizationObjective.BALANCE_ALL
    
    def _determine_priority(self, requirements: Dict[str, Any]) -> WorkflowPriority:
        """Determine workflow priority from requirements"""
        priority_str = requirements.get("priority", "medium").lower()
        
        priority_mapping = {
            "critical": WorkflowPriority.CRITICAL,
            "high": WorkflowPriority.HIGH,
            "medium": WorkflowPriority.MEDIUM,
            "low": WorkflowPriority.LOW,
            "background": WorkflowPriority.BACKGROUND
        }
        
        return priority_mapping.get(priority_str, WorkflowPriority.MEDIUM)
    
    def _create_retry_policy(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create retry policy based on requirements"""
        return {
            "max_retries": requirements.get("max_retries", 3),
            "retry_delay": requirements.get("retry_delay", 5.0),
            "exponential_backoff": requirements.get("exponential_backoff", True),
            "retry_on_timeout": requirements.get("retry_on_timeout", True),
            "retry_on_system_error": requirements.get("retry_on_system_error", True)
        }
    
    def _define_success_criteria(self, requirements: Dict[str, Any]) -> List[str]:
        """Define success criteria for the workflow"""
        criteria = [
            "All tasks completed successfully",
            "No critical errors occurred",
            "Execution time within acceptable limits"
        ]
        
        # Add specific criteria from requirements
        if "success_criteria" in requirements:
            criteria.extend(requirements["success_criteria"])
        
        if requirements.get("accuracy_threshold"):
            criteria.append(f"Results accuracy >= {requirements['accuracy_threshold']}")
        
        if requirements.get("max_execution_time"):
            criteria.append(f"Execution time <= {requirements['max_execution_time']} seconds")
        
        return criteria

class WorkflowScheduler:
    """Schedules and manages workflow execution with intelligent load balancing"""
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.max_concurrent_workflows = max_concurrent_workflows
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_queue = PriorityQueue()
        self.task_queue = PriorityQueue()
        self.system_load: Dict[str, float] = {}
        self.scheduler_running = False
        
        logger.info(f"WorkflowScheduler initialized with max {max_concurrent_workflows} concurrent workflows")
    
    async def schedule_workflow(self, workflow: WorkflowDefinition) -> str:
        """Schedule a workflow for execution"""
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        # Create workflow execution
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.workflow_id,
            status=WorkflowStatus.PENDING
        )
        
        # Add to queue with priority
        priority = workflow.priority.value
        self.workflow_queue.put((priority, execution_id, workflow, execution))
        
        logger.info(f"Scheduled workflow {workflow.workflow_id} for execution as {execution_id}")
        return execution_id
    
    async def start_scheduler(self):
        """Start the workflow scheduler"""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        self.scheduler_running = True
        logger.info("Starting workflow scheduler")
        
        # Start scheduler tasks
        asyncio.create_task(self._workflow_scheduler_loop())
        asyncio.create_task(self._task_scheduler_loop())
        asyncio.create_task(self._system_monitor_loop())
    
    async def _workflow_scheduler_loop(self):
        """Main workflow scheduling loop"""
        while self.scheduler_running:
            try:
                # Check if we can start more workflows
                if len(self.active_workflows) < self.max_concurrent_workflows:
                    if not self.workflow_queue.empty():
                        priority, execution_id, workflow, execution = self.workflow_queue.get()
                        
                        # Start workflow execution
                        await self._start_workflow_execution(workflow, execution)
                
                await asyncio.sleep(1.0)  # Check every second
            
            except Exception as e:
                logger.error(f"Error in workflow scheduler loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _task_scheduler_loop(self):
        """Task scheduling and execution loop"""
        while self.scheduler_running:
            try:
                # Process ready tasks
                await self._process_ready_tasks()
                await asyncio.sleep(0.5)  # Check twice per second
            
            except Exception as e:
                logger.error(f"Error in task scheduler loop: {e}")
                await asyncio.sleep(2.0)
    
    async def _system_monitor_loop(self):
        """Monitor system load and performance"""
        while self.scheduler_running:
            try:
                # Update system load metrics
                await self._update_system_load()
                await asyncio.sleep(10.0)  # Update every 10 seconds
            
            except Exception as e:
                logger.error(f"Error in system monitor loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _start_workflow_execution(self, workflow: WorkflowDefinition, execution: WorkflowExecution):
        """Start execution of a workflow"""
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.now()
        self.active_workflows[execution.execution_id] = execution
        
        # Schedule initial tasks (those with no dependencies)
        ready_tasks = self._get_ready_tasks(workflow, execution)
        for task in ready_tasks:
            await self._schedule_task(workflow, execution, task)
        
        logger.info(f"Started workflow execution {execution.execution_id} with {len(ready_tasks)} initial tasks")
    
    def _get_ready_tasks(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> List[WorkflowTask]:
        """Get tasks that are ready to execute"""
        ready_tasks = []
        
        for task in workflow.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_completed = all(
                    dep_id in execution.completed_tasks for dep_id in task.dependencies
                )
                
                if dependencies_completed:
                    ready_tasks.append(task)
        
        return ready_tasks
    
    async def _schedule_task(self, workflow: WorkflowDefinition, execution: WorkflowExecution, task: WorkflowTask):
        """Schedule a task for execution"""
        # Check system load for target system
        system_load = self.system_load.get(task.target_system, 0.0)
        
        # Calculate task priority (lower number = higher priority)
        priority = task.priority + (system_load * 10)  # Penalize loaded systems
        
        # Add to task queue
        self.task_queue.put((priority, datetime.now(), workflow, execution, task))
        
        task.status = TaskStatus.SCHEDULED
        logger.debug(f"Scheduled task {task.task_id} for system {task.target_system}")
    
    async def _process_ready_tasks(self):
        """Process tasks that are ready to execute"""
        if self.task_queue.empty():
            return
        
        # Get next task
        priority, scheduled_time, workflow, execution, task = self.task_queue.get()
        
        # Check if system is available
        system_load = self.system_load.get(task.target_system, 0.0)
        if system_load > 0.8:  # System overloaded
            # Re-queue with delay
            await asyncio.sleep(1.0)
            self.task_queue.put((priority + 1, datetime.now(), workflow, execution, task))
            return
        
        # Execute task
        await self._execute_task(workflow, execution, task)
    
    async def _execute_task(self, workflow: WorkflowDefinition, execution: WorkflowExecution, task: WorkflowTask):
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        execution.current_tasks.append(task.task_id)
        
        logger.info(f"Executing task {task.task_id} on system {task.target_system}")
        
        try:
            # Simulate task execution
            await self._simulate_task_execution(task)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration = (task.completed_at - task.started_at).total_seconds()
            
            # Move to completed tasks
            execution.current_tasks.remove(task.task_id)
            execution.completed_tasks.append(task.task_id)
            execution.task_results[task.task_id] = task.results
            
            # Update progress
            execution.progress = len(execution.completed_tasks) / len(workflow.tasks)
            
            logger.info(f"Task {task.task_id} completed successfully in {task.actual_duration:.2f}s")
            
            # Check if workflow is complete
            if len(execution.completed_tasks) == len(workflow.tasks):
                await self._complete_workflow(workflow, execution)
            else:
                # Schedule next ready tasks
                ready_tasks = self._get_ready_tasks(workflow, execution)
                for ready_task in ready_tasks:
                    await self._schedule_task(workflow, execution, ready_task)
        
        except Exception as e:
            # Task failed
            await self._handle_task_failure(workflow, execution, task, str(e))
    
    async def _simulate_task_execution(self, task: WorkflowTask):
        """Simulate task execution (replace with actual system calls)"""
        # Simulate variable execution time
        execution_time = task.estimated_duration * (0.8 + 0.4 * np.random.random())
        await asyncio.sleep(min(execution_time, 0.1))  # Cap simulation time
        
        # Simulate success/failure based on system reliability
        failure_probability = 0.05  # 5% failure rate
        if np.random.random() < failure_probability:
            raise Exception("Simulated system error")
        
        # Generate mock results
        task.results = {
            "status": "success",
            "output": f"Results from {task.task_type} on {task.target_system}",
            "metrics": {
                "accuracy": 0.85 + 0.1 * np.random.random(),
                "processing_time": execution_time,
                "confidence": 0.9 + 0.1 * np.random.random()
            }
        }
    
    async def _handle_task_failure(self, workflow: WorkflowDefinition, execution: WorkflowExecution, 
                                 task: WorkflowTask, error_message: str):
        """Handle task failure with retry logic"""
        task.error_message = error_message
        task.retry_count += 1
        
        logger.warning(f"Task {task.task_id} failed: {error_message} (retry {task.retry_count}/{task.max_retries})")
        
        if task.retry_count <= task.max_retries:
            # Retry task
            task.status = TaskStatus.RETRYING
            await asyncio.sleep(2.0 ** task.retry_count)  # Exponential backoff
            await self._schedule_task(workflow, execution, task)
        else:
            # Task permanently failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            execution.current_tasks.remove(task.task_id)
            execution.failed_tasks.append(task.task_id)
            execution.error_messages.append(f"Task {task.task_id}: {error_message}")
            
            # Check if workflow should fail
            await self._check_workflow_failure(workflow, execution)
    
    async def _check_workflow_failure(self, workflow: WorkflowDefinition, execution: WorkflowExecution):
        """Check if workflow should fail due to task failures"""
        # For now, fail workflow if any critical task fails
        # This could be made more sophisticated based on task importance
        
        if len(execution.failed_tasks) > 0:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.now()
            
            logger.error(f"Workflow {execution.execution_id} failed due to task failures")
            
            # Remove from active workflows
            if execution.execution_id in self.active_workflows:
                del self.active_workflows[execution.execution_id]
    
    async def _complete_workflow(self, workflow: WorkflowDefinition, execution: WorkflowExecution):
        """Complete workflow execution"""
        execution.status = WorkflowStatus.COMPLETED
        execution.completed_at = datetime.now()
        execution.progress = 1.0
        
        # Calculate execution metrics
        total_duration = (execution.completed_at - execution.started_at).total_seconds()
        execution.execution_metrics = {
            "total_duration": total_duration,
            "tasks_completed": len(execution.completed_tasks),
            "tasks_failed": len(execution.failed_tasks),
            "success_rate": len(execution.completed_tasks) / len(workflow.tasks),
            "average_task_duration": statistics.mean([
                task.actual_duration for task in workflow.tasks 
                if task.actual_duration is not None
            ]) if any(task.actual_duration for task in workflow.tasks) else 0
        }
        
        logger.info(f"Workflow {execution.execution_id} completed successfully in {total_duration:.2f}s")
        
        # Remove from active workflows
        if execution.execution_id in self.active_workflows:
            del self.active_workflows[execution.execution_id]
    
    async def _update_system_load(self):
        """Update system load metrics"""
        # Simulate system load updates
        for system_id in self.system_load:
            # Add some random variation
            current_load = self.system_load[system_id]
            variation = (np.random.random() - 0.5) * 0.2  # ±10% variation
            new_load = max(0.0, min(1.0, current_load + variation))
            self.system_load[system_id] = new_load
        
        # Add load for systems with active tasks
        for execution in self.active_workflows.values():
            for task_id in execution.current_tasks:
                # Find the task and its target system
                # This is simplified - in practice you'd track this better
                pass
    
    def register_system(self, system_id: str, initial_load: float = 0.0):
        """Register a system for load monitoring"""
        self.system_load[system_id] = initial_load
        logger.info(f"Registered system {system_id} with initial load {initial_load:.1%}")
    
    def stop_scheduler(self):
        """Stop the workflow scheduler"""
        self.scheduler_running = False
        logger.info("Workflow scheduler stopped")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            "scheduler_running": self.scheduler_running,
            "active_workflows": len(self.active_workflows),
            "queued_workflows": self.workflow_queue.qsize(),
            "queued_tasks": self.task_queue.qsize(),
            "system_loads": self.system_load.copy(),
            "max_concurrent_workflows": self.max_concurrent_workflows
        }

class WorkflowOptimizer:
    """Optimizes workflow performance based on execution history and patterns"""
    
    def __init__(self):
        self.execution_history: List[WorkflowExecution] = []
        self.optimization_history: List[WorkflowOptimization] = []
        self.performance_patterns: Dict[str, Dict[str, float]] = {}
        
        logger.info("WorkflowOptimizer initialized")
    
    def record_execution(self, execution: WorkflowExecution):
        """Record workflow execution for optimization analysis"""
        self.execution_history.append(execution)
        logger.debug(f"Recorded execution {execution.execution_id} for optimization analysis")
    
    async def optimize_workflow(self, workflow: WorkflowDefinition,
                              objective: OptimizationObjective = None) -> WorkflowOptimization:
        """Optimize a workflow based on historical data and objective"""
        if objective is None:
            objective = workflow.optimization_objective
        
        logger.info(f"Optimizing workflow {workflow.workflow_id} for {objective.value}")
        
        # Analyze current performance
        current_performance = self._analyze_current_performance(workflow)
        
        # Generate optimization recommendations
        optimizations = await self._generate_optimizations(workflow, objective)
        
        # Apply optimizations
        optimized_workflow = self._apply_optimizations(workflow, optimizations)
        
        # Predict optimized performance
        optimized_performance = self._predict_optimized_performance(optimized_workflow, optimizations)
        
        # Calculate improvement
        improvement = self._calculate_improvement(current_performance, optimized_performance, objective)
        
        # Create optimization record
        optimization = WorkflowOptimization(
            optimization_id=f"opt_{uuid.uuid4().hex[:8]}",
            workflow_id=workflow.workflow_id,
            objective=objective,
            original_performance=current_performance,
            optimized_performance=optimized_performance,
            optimization_changes=optimizations,
            improvement_percentage=improvement,
            implementation_effort=self._estimate_implementation_effort(optimizations)
        )
        
        self.optimization_history.append(optimization)
        
        logger.info(f"Workflow optimization completed. Improvement: {improvement:.1%}")
        return optimization
    
    def _analyze_current_performance(self, workflow: WorkflowDefinition) -> Dict[str, float]:
        """Analyze current performance of a workflow"""
        # Find executions of this workflow
        workflow_executions = [
            exec for exec in self.execution_history 
            if exec.workflow_id == workflow.workflow_id and exec.status == WorkflowStatus.COMPLETED
        ]
        
        if not workflow_executions:
            # Estimate based on task estimates
            estimated_duration = sum(task.estimated_duration for task in workflow.tasks)
            return {
                "duration": estimated_duration,
                "success_rate": 0.9,  # Optimistic estimate
                "resource_usage": 50.0,  # Medium usage
                "cost": estimated_duration * 0.1  # Simple cost model
            }
        
        # Calculate metrics from executions
        durations = [exec.execution_metrics.get("total_duration", 0) for exec in workflow_executions]
        success_rates = [exec.execution_metrics.get("success_rate", 0) for exec in workflow_executions]
        
        return {
            "duration": statistics.mean(durations),
            "success_rate": statistics.mean(success_rates),
            "resource_usage": 50.0,  # Would be calculated from actual system metrics
            "cost": statistics.mean(durations) * 0.1
        }
    
    async def _generate_optimizations(self, workflow: WorkflowDefinition,
                                    objective: OptimizationObjective) -> List[str]:
        """Generate optimization recommendations"""
        optimizations = []
        
        if objective == OptimizationObjective.MINIMIZE_TIME:
            optimizations.extend(self._generate_time_optimizations(workflow))
        elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            optimizations.extend(self._generate_accuracy_optimizations(workflow))
        elif objective == OptimizationObjective.MINIMIZE_COST:
            optimizations.extend(self._generate_cost_optimizations(workflow))
        elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            optimizations.extend(self._generate_throughput_optimizations(workflow))
        else:  # BALANCE_ALL
            optimizations.extend(self._generate_balanced_optimizations(workflow))
        
        return optimizations
    
    def _generate_time_optimizations(self, workflow: WorkflowDefinition) -> List[str]:
        """Generate optimizations to minimize execution time"""
        optimizations = []
        
        # Increase parallelism
        if workflow.max_parallel_tasks < 10:
            optimizations.append("Increase maximum parallel tasks to improve concurrency")
        
        # Optimize task scheduling
        optimizations.append("Reorder tasks to minimize critical path length")
        
        # Optimize system selection
        optimizations.append("Select fastest systems for time-critical tasks")
        
        # Reduce timeouts for non-critical tasks
        optimizations.append("Reduce task timeouts where appropriate")
        
        return optimizations
    
    def _generate_accuracy_optimizations(self, workflow: WorkflowDefinition) -> List[str]:
        """Generate optimizations to maximize accuracy"""
        optimizations = []
        
        # Use ensemble approaches
        optimizations.append("Implement ensemble voting for critical decisions")
        
        # Select most accurate systems
        optimizations.append("Prioritize high-accuracy systems over speed")
        
        # Add validation tasks
        optimizations.append("Add cross-validation tasks for important results")
        
        # Increase retry attempts
        optimizations.append("Increase retry attempts for accuracy-critical tasks")
        
        return optimizations
    
    def _generate_cost_optimizations(self, workflow: WorkflowDefinition) -> List[str]:
        """Generate optimizations to minimize cost"""
        optimizations = []
        
        # Use cheaper systems
        optimizations.append("Select cost-effective systems where performance allows")
        
        # Reduce resource allocation
        optimizations.append("Optimize resource allocation to reduce waste")
        
        # Batch processing
        optimizations.append("Implement batching to improve resource utilization")
        
        # Eliminate redundant tasks
        optimizations.append("Remove or combine redundant tasks")
        
        return optimizations
    
    def _generate_throughput_optimizations(self, workflow: WorkflowDefinition) -> List[str]:
        """Generate optimizations to maximize throughput"""
        optimizations = []
        
        # Maximize parallelism
        optimizations.append("Maximize parallel task execution")
        
        # Pipeline optimization
        optimizations.append("Implement pipelining for sequential tasks")
        
        # Load balancing
        optimizations.append("Implement dynamic load balancing across systems")
        
        # Caching
        optimizations.append("Implement result caching for repeated operations")
        
        return optimizations
    
    def _generate_balanced_optimizations(self, workflow: WorkflowDefinition) -> List[str]:
        """Generate balanced optimizations across all objectives"""
        optimizations = []
        
        # Moderate parallelism
        optimizations.append("Balance parallel execution with resource constraints")
        
        # Smart system selection
        optimizations.append("Select systems based on multi-criteria optimization")
        
        # Adaptive timeouts
        optimizations.append("Implement adaptive timeouts based on task importance")
        
        # Quality gates
        optimizations.append("Add quality gates to balance speed and accuracy")
        
        return optimizations
    
    def _apply_optimizations(self, workflow: WorkflowDefinition, 
                           optimizations: List[str]) -> WorkflowDefinition:
        """Apply optimizations to create an optimized workflow"""
        # Create a copy of the workflow
        optimized_workflow = WorkflowDefinition(
            workflow_id=f"{workflow.workflow_id}_optimized",
            name=f"{workflow.name} (Optimized)",
            description=f"Optimized version of {workflow.name}",
            tasks=workflow.tasks.copy(),
            execution_graph=workflow.execution_graph.copy(),
            priority=workflow.priority,
            optimization_objective=workflow.optimization_objective,
            max_parallel_tasks=workflow.max_parallel_tasks,
            total_timeout=workflow.total_timeout,
            retry_policy=workflow.retry_policy.copy(),
            success_criteria=workflow.success_criteria.copy(),
            metadata=workflow.metadata.copy()
        )
        
        # Apply optimizations
        for optimization in optimizations:
            if "parallel tasks" in optimization.lower():
                optimized_workflow.max_parallel_tasks = min(15, optimized_workflow.max_parallel_tasks + 3)
            
            elif "timeout" in optimization.lower() and "reduce" in optimization.lower():
                for task in optimized_workflow.tasks:
                    if task.priority > 3:  # Low priority tasks
                        task.timeout *= 0.8
            
            elif "retry" in optimization.lower() and "increase" in optimization.lower():
                for task in optimized_workflow.tasks:
                    if task.priority <= 2:  # High priority tasks
                        task.max_retries += 1
        
        return optimized_workflow
    
    def _predict_optimized_performance(self, optimized_workflow: WorkflowDefinition,
                                     optimizations: List[str]) -> Dict[str, float]:
        """Predict performance of optimized workflow"""
        # Start with current performance
        base_performance = self._analyze_current_performance(optimized_workflow)
        
        # Apply optimization factors
        improvement_factors = {
            "duration": 1.0,
            "success_rate": 1.0,
            "resource_usage": 1.0,
            "cost": 1.0
        }
        
        for optimization in optimizations:
            if "parallel" in optimization.lower():
                improvement_factors["duration"] *= 0.85  # 15% improvement
                improvement_factors["resource_usage"] *= 1.1  # 10% increase
            
            elif "timeout" in optimization.lower():
                improvement_factors["duration"] *= 0.95  # 5% improvement
            
            elif "accuracy" in optimization.lower():
                improvement_factors["success_rate"] *= 1.05  # 5% improvement
                improvement_factors["duration"] *= 1.1  # 10% slower
            
            elif "cost" in optimization.lower():
                improvement_factors["cost"] *= 0.9  # 10% cost reduction
        
        # Apply improvements
        optimized_performance = {}
        for metric, value in base_performance.items():
            factor = improvement_factors.get(metric, 1.0)
            optimized_performance[metric] = value * factor
        
        return optimized_performance
    
    def _calculate_improvement(self, current: Dict[str, float], optimized: Dict[str, float],
                             objective: OptimizationObjective) -> float:
        """Calculate improvement percentage based on objective"""
        if objective == OptimizationObjective.MINIMIZE_TIME:
            return (current["duration"] - optimized["duration"]) / current["duration"]
        elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            return (optimized["success_rate"] - current["success_rate"]) / current["success_rate"]
        elif objective == OptimizationObjective.MINIMIZE_COST:
            return (current["cost"] - optimized["cost"]) / current["cost"]
        elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            return (optimized["duration"] - current["duration"]) / current["duration"]  # Faster = higher throughput
        else:  # BALANCE_ALL
            # Weighted average of improvements
            time_improvement = (current["duration"] - optimized["duration"]) / current["duration"]
            accuracy_improvement = (optimized["success_rate"] - current["success_rate"]) / current["success_rate"]
            cost_improvement = (current["cost"] - optimized["cost"]) / current["cost"]
            
            return (time_improvement + accuracy_improvement + cost_improvement) / 3
    
    def _estimate_implementation_effort(self, optimizations: List[str]) -> float:
        """Estimate effort required to implement optimizations"""
        base_effort = 0.1  # Base effort
        
        effort_per_optimization = {
            "parallel": 0.3,
            "ensemble": 0.5,
            "caching": 0.4,
            "timeout": 0.1,
            "retry": 0.1,
            "system": 0.2
        }
        
        total_effort = base_effort
        for optimization in optimizations:
            for keyword, effort in effort_per_optimization.items():
                if keyword in optimization.lower():
                    total_effort += effort
                    break
        
        return min(1.0, total_effort)

class IntelligentWorkflowEngine:
    """
    Main workflow engine that coordinates design, execution, and optimization
    """
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.designer = WorkflowDesigner()
        self.scheduler = WorkflowScheduler(max_concurrent_workflows)
        self.optimizer = WorkflowOptimizer()
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.engine_metrics: Dict[str, Any] = {}
        
        logger.info("IntelligentWorkflowEngine initialized with comprehensive automation capabilities")
    
    async def start_engine(self):
        """Start the workflow engine"""
        logger.info("Starting Intelligent Workflow Engine")
        await self.scheduler.start_scheduler()
        self._update_engine_metrics()
    
    def register_intelligence_system(self, system_id: str, capabilities: List[str],
                                   performance_metrics: Dict[str, float]):
        """Register an intelligence system with the workflow engine"""
        logger.info(f"Registering intelligence system: {system_id}")
        
        # Register with designer
        self.designer.register_system_capabilities(system_id, capabilities, performance_metrics)
        
        # Register with scheduler
        self.scheduler.register_system(system_id, performance_metrics.get("load", 0.0))
        
        logger.info(f"Successfully registered {system_id} with {len(capabilities)} capabilities")
    
    async def create_and_execute_workflow(self, requirements: Dict[str, Any],
                                        constraints: Dict[str, Any] = None) -> Tuple[str, str]:
        """Create and execute a workflow based on requirements"""
        logger.info(f"Creating and executing workflow for: {requirements.get('objective', 'unknown')}")
        
        # Design workflow
        workflow = await self.designer.design_workflow(requirements, constraints)
        self.workflows[workflow.workflow_id] = workflow
        
        # Schedule for execution
        execution_id = await self.scheduler.schedule_workflow(workflow)
        
        logger.info(f"Created workflow {workflow.workflow_id} and scheduled execution {execution_id}")
        return workflow.workflow_id, execution_id
    
    async def optimize_workflow(self, workflow_id: str, 
                              objective: OptimizationObjective = None) -> WorkflowOptimization:
        """Optimize an existing workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        optimization = await self.optimizer.optimize_workflow(workflow, objective)
        
        logger.info(f"Optimized workflow {workflow_id}. Improvement: {optimization.improvement_percentage:.1%}")
        return optimization
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow"""
        if workflow_id not in self.workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow = self.workflows[workflow_id]
        
        # Find executions of this workflow
        executions = [exec for exec in self.executions.values() if exec.workflow_id == workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.priority.value,
            "tasks_count": len(workflow.tasks),
            "executions": len(executions),
            "last_execution": executions[-1].status.value if executions else "never_executed",
            "optimization_objective": workflow.optimization_objective.value,
            "created_at": workflow.created_at.isoformat()
        }
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of a workflow execution"""
        if execution_id in self.scheduler.active_workflows:
            execution = self.scheduler.active_workflows[execution_id]
        elif execution_id in self.executions:
            execution = self.executions[execution_id]
        else:
            return {"error": f"Execution {execution_id} not found"}
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress": execution.progress,
            "current_tasks": len(execution.current_tasks),
            "completed_tasks": len(execution.completed_tasks),
            "failed_tasks": len(execution.failed_tasks),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "execution_metrics": execution.execution_metrics
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the workflow engine"""
        scheduler_status = self.scheduler.get_scheduler_status()
        
        return {
            "engine_metrics": self.engine_metrics,
            "scheduler_status": scheduler_status,
            "total_workflows": len(self.workflows),
            "total_executions": len(self.executions),
            "optimization_history": len(self.optimizer.optimization_history),
            "registered_systems": len(self.designer.system_capabilities),
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_engine_metrics(self):
        """Update engine performance metrics"""
        self.engine_metrics = {
            "workflows_created": len(self.workflows),
            "workflows_executed": len(self.executions),
            "total_optimizations": len(self.optimizer.optimization_history),
            "average_workflow_tasks": statistics.mean([len(w.tasks) for w in self.workflows.values()]) if self.workflows else 0,
            "success_rate": self._calculate_success_rate(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall workflow success rate"""
        completed_executions = [exec for exec in self.executions.values() 
                              if exec.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]]
        
        if not completed_executions:
            return 1.0
        
        successful_executions = [exec for exec in completed_executions 
                               if exec.status == WorkflowStatus.COMPLETED]
        
        return len(successful_executions) / len(completed_executions)
    
    def stop_engine(self):
        """Stop the workflow engine"""
        logger.info("Stopping Intelligent Workflow Engine")
        self.scheduler.stop_scheduler()

async def main():
    """Main function to demonstrate IntelligentWorkflowEngine capabilities"""
    
    # Initialize the intelligent workflow engine
    workflow_engine = IntelligentWorkflowEngine(max_concurrent_workflows=5)
    
    print("⚙️ Intelligent Workflow Engine - Automated Workflow Management")
    print("=" * 80)
    
    # Start the engine
    await workflow_engine.start_engine()
    
    # Register intelligence systems
    systems = [
        {
            "system_id": "analytics_hub",
            "capabilities": ["data_analysis", "statistical_analysis", "trend_detection"],
            "performance_metrics": {"response_time": 150.0, "throughput": 1000.0, "availability": 0.99, "load": 0.3}
        },
        {
            "system_id": "ml_orchestrator",
            "capabilities": ["prediction", "forecasting", "optimization", "model_training"],
            "performance_metrics": {"response_time": 300.0, "throughput": 500.0, "availability": 0.98, "load": 0.5}
        },
        {
            "system_id": "pattern_recognizer",
            "capabilities": ["pattern_detection", "anomaly_detection", "classification"],
            "performance_metrics": {"response_time": 200.0, "throughput": 800.0, "availability": 0.97, "load": 0.2}
        },
        {
            "system_id": "decision_engine",
            "capabilities": ["recommendation", "decision_support", "risk_assessment"],
            "performance_metrics": {"response_time": 100.0, "throughput": 1200.0, "availability": 0.995, "load": 0.1}
        }
    ]
    
    print("\n1. Intelligence System Registration")
    print("-" * 40)
    
    for system in systems:
        workflow_engine.register_intelligence_system(
            system["system_id"], 
            system["capabilities"], 
            system["performance_metrics"]
        )
        print(f"✅ Registered {system['system_id']} with {len(system['capabilities'])} capabilities")
    
    print("\n\n2. Workflow Creation and Execution")
    print("-" * 40)
    
    # Create workflows with different requirements
    workflow_examples = [
        {
            "name": "Comprehensive Data Analysis Workflow",
            "objective": "Analyze data patterns and generate predictions",
            "description": "End-to-end data analysis with pattern detection and forecasting",
            "required_capabilities": ["data_analysis", "pattern_detection", "prediction"],
            "priority": "high",
            "optimization_objective": "balance_all",
            "accuracy_threshold": 0.85,
            "max_execution_time": 600
        },
        {
            "name": "Real-time Decision Support Workflow", 
            "objective": "Provide real-time recommendations based on current data",
            "description": "Fast decision support for time-critical operations",
            "required_capabilities": ["data_analysis", "recommendation"],
            "priority": "critical",
            "time_critical": True,
            "optimization_objective": "minimize_time",
            "max_execution_time": 120
        },
        {
            "name": "Advanced ML Optimization Workflow",
            "objective": "Optimize ML models for maximum accuracy",
            "description": "Complex ML workflow with multiple optimization stages",
            "required_capabilities": ["model_training", "optimization", "forecasting"],
            "priority": "medium",
            "accuracy_critical": True,
            "optimization_objective": "maximize_accuracy",
            "accuracy_threshold": 0.95
        }
    ]
    
    created_workflows = []
    for workflow_req in workflow_examples:
        workflow_id, execution_id = await workflow_engine.create_and_execute_workflow(workflow_req)
        created_workflows.append((workflow_id, execution_id))
        print(f"✅ Created and scheduled: {workflow_req['name']}")
        print(f"   Workflow ID: {workflow_id}")
        print(f"   Execution ID: {execution_id}")
    
    print("\n\n3. Workflow Execution Monitoring")
    print("-" * 40)
    
    # Wait for some execution to complete
    await asyncio.sleep(3.0)
    
    # Check workflow statuses
    for workflow_id, execution_id in created_workflows:
        workflow_status = workflow_engine.get_workflow_status(workflow_id)
        execution_status = workflow_engine.get_execution_status(execution_id)
        
        print(f"\nWorkflow: {workflow_status['name']}")
        print(f"  Status: {execution_status['status']}")
        print(f"  Progress: {execution_status['progress']:.1%}")
        print(f"  Completed Tasks: {execution_status['completed_tasks']}")
        print(f"  Failed Tasks: {execution_status['failed_tasks']}")
    
    print("\n\n4. Workflow Optimization")
    print("-" * 40)
    
    # Optimize workflows for different objectives
    optimization_objectives = [
        (OptimizationObjective.MINIMIZE_TIME, "Speed Optimization"),
        (OptimizationObjective.MAXIMIZE_ACCURACY, "Accuracy Optimization"),
        (OptimizationObjective.MINIMIZE_COST, "Cost Optimization")
    ]
    
    for objective, description in optimization_objectives:
        if created_workflows:
            workflow_id = created_workflows[0][0]  # Optimize first workflow
            optimization = await workflow_engine.optimize_workflow(workflow_id, objective)
            
            print(f"\n{description} for {workflow_id}:")
            print(f"  Improvement: {optimization.improvement_percentage:.1%}")
            print(f"  Implementation Effort: {optimization.implementation_effort:.1f}")
            print(f"  Changes: {len(optimization.optimization_changes)}")
            
            # Show top optimization changes
            for change in optimization.optimization_changes[:2]:
                print(f"    • {change}")
    
    print("\n\n5. Engine Performance Analytics")
    print("-" * 40)
    
    # Get comprehensive engine status
    engine_status = workflow_engine.get_engine_status()
    
    print("Workflow Engine Status:")
    print(f"  Total Workflows Created: {engine_status['total_workflows']}")
    print(f"  Total Executions: {engine_status['total_executions']}")
    print(f"  Active Workflows: {engine_status['scheduler_status']['active_workflows']}")
    print(f"  Queued Workflows: {engine_status['scheduler_status']['queued_workflows']}")
    print(f"  Queued Tasks: {engine_status['scheduler_status']['queued_tasks']}")
    print(f"  Registered Systems: {engine_status['registered_systems']}")
    print(f"  Optimization History: {engine_status['optimization_history']}")
    
    print(f"\nSystem Load Distribution:")
    for system_id, load in engine_status['scheduler_status']['system_loads'].items():
        print(f"  {system_id}: {load:.1%}")
    
    print(f"\nEngine Metrics:")
    metrics = engine_status['engine_metrics']
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
    print(f"  Average Tasks per Workflow: {metrics['average_workflow_tasks']:.1f}")
    print(f"  Total Optimizations: {metrics['total_optimizations']}")
    
    print("\n\n6. Advanced Workflow Features Demo")
    print("-" * 40)
    
    # Create a complex workflow with custom requirements
    complex_workflow_req = {
        "name": "Multi-Stage Intelligence Pipeline",
        "objective": "Execute complete intelligence analysis pipeline with optimization",
        "description": "Complex multi-stage workflow demonstrating full engine capabilities",
        "required_capabilities": ["data_analysis", "pattern_detection", "prediction", "optimization", "recommendation"],
        "priority": "high",
        "optimization_objective": "balance_all",
        "constraints": {
            "max_parallel_tasks": 3,
            "total_timeout": 900,
            "max_retries": 5
        },
        "success_criteria": [
            "All analysis stages completed successfully",
            "Prediction accuracy > 90%",
            "Recommendations generated with confidence > 85%"
        ],
        "metadata": {
            "created_by": "intelligent_workflow_engine",
            "complexity": "high",
            "estimated_value": "high"
        }
    }
    
    complex_workflow_id, complex_execution_id = await workflow_engine.create_and_execute_workflow(
        complex_workflow_req, complex_workflow_req.get("constraints", {})
    )
    
    print(f"✅ Created complex workflow: {complex_workflow_id}")
    
    # Monitor complex workflow
    await asyncio.sleep(2.0)
    complex_status = workflow_engine.get_execution_status(complex_execution_id)
    print(f"Complex Workflow Status: {complex_status['status']} ({complex_status['progress']:.1%} complete)")
    
    print("\n✅ Intelligent Workflow Engine demonstration completed successfully!")
    print("Advanced autonomous workflow management achieved with comprehensive optimization!")
    
    # Stop the engine
    workflow_engine.stop_engine()

if __name__ == "__main__":
    asyncio.run(main())