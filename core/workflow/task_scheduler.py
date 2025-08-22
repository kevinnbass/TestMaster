"""
Task Scheduler - Advanced intelligent task scheduling and resource allocation

This module implements sophisticated task scheduling algorithms with load balancing,
resource optimization, and adaptive execution strategies for the intelligent
workflow engine with real-time performance optimization.

Key Capabilities:
- Priority-based task scheduling with dynamic rebalancing
- Resource-aware task allocation with constraint satisfaction
- Load balancing across multiple execution systems
- Adaptive scheduling based on system performance and availability
- Dependency resolution with parallel execution optimization
- Real-time system monitoring and capacity management
"""

import asyncio
import heapq
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import statistics

from .workflow_models import (
    TaskDefinition, TaskExecution, TaskStatus, TaskPriority,
    WorkflowDefinition, WorkflowExecution, WorkflowStatus,
    SystemStatus, SystemCapability, ResourceType, TaskResource,
    OptimizationObjective, OptimizationMetrics,
    calculate_task_priority_score, PRIORITY_WEIGHTS
)

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Different scheduling strategies for task execution"""
    PRIORITY_FIRST = "priority_first"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    RESOURCE_OPTIMIZED = "resource_optimized"
    DEADLINE_AWARE = "deadline_aware"


class TaskScheduler:
    """
    Advanced intelligent task scheduler with resource optimization
    
    Manages task execution across multiple systems with sophisticated
    scheduling algorithms and real-time performance optimization.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """Initialize task scheduler"""
        self.max_concurrent_tasks = max_concurrent_tasks
        self.systems: Dict[str, SystemStatus] = {}
        self.task_queue = []  # Priority queue (heapq)
        self.running_tasks: Dict[str, TaskExecution] = {}
        self.completed_tasks: Dict[str, TaskExecution] = {}
        self.failed_tasks: Dict[str, TaskExecution] = {}
        
        # Scheduling configuration
        self.scheduling_strategy = SchedulingStrategy.LOAD_BALANCED
        self.load_balancing_enabled = True
        self.resource_optimization_enabled = True
        self.adaptive_scheduling_enabled = True
        
        # Performance tracking
        self.scheduling_metrics = {
            'total_tasks_scheduled': 0,
            'average_scheduling_time': 0.0,
            'system_utilization': {},
            'resource_efficiency': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        self._system_lock = threading.Lock()
        
        logger.info("Task Scheduler initialized with %d max concurrent tasks", max_concurrent_tasks)
    
    def register_system(self, system: SystemStatus):
        """Register a system for task execution"""
        with self._system_lock:
            self.systems[system.system_id] = system
            self.scheduling_metrics['system_utilization'][system.system_id] = 0.0
        logger.info("Registered system: %s", system.system_id)
    
    def unregister_system(self, system_id: str):
        """Unregister a system"""
        with self._system_lock:
            if system_id in self.systems:
                # Reassign running tasks if possible
                self._reassign_system_tasks(system_id)
                del self.systems[system_id]
                if system_id in self.scheduling_metrics['system_utilization']:
                    del self.scheduling_metrics['system_utilization'][system_id]
        logger.info("Unregistered system: %s", system_id)
    
    def schedule_task(self, task: TaskDefinition, workflow_context: Dict[str, Any] = None) -> str:
        """
        Schedule a task for execution
        
        Args:
            task: Task definition to schedule
            workflow_context: Context information from workflow
            
        Returns:
            Task execution ID
        """
        start_time = time.time()
        
        # Create task execution
        task_execution = TaskExecution(task_id=task.task_id)
        task_execution.status = TaskStatus.QUEUED
        
        # Calculate priority score
        priority_score = self._calculate_dynamic_priority(task, workflow_context)
        
        # Add to priority queue (negative score for min-heap behavior as max-priority)
        with self._lock:
            heapq.heappush(self.task_queue, (-priority_score, task_execution.execution_id, task, task_execution))
            self.scheduling_metrics['total_tasks_scheduled'] += 1
        
        # Update scheduling time metrics
        scheduling_time = time.time() - start_time
        self._update_scheduling_metrics(scheduling_time)
        
        logger.debug("Scheduled task %s with priority %.2f", task.task_id, priority_score)
        return task_execution.execution_id
    
    def schedule_workflow_tasks(self, workflow: WorkflowDefinition) -> Dict[str, str]:
        """
        Schedule all tasks in a workflow with dependency consideration
        
        Args:
            workflow: Workflow definition containing tasks
            
        Returns:
            Dictionary mapping task IDs to execution IDs
        """
        execution_ids = {}
        
        # Create dependency graph and schedule in topological order
        dependency_graph = self._build_dependency_graph(workflow)
        scheduled_tasks = set()
        
        while len(scheduled_tasks) < len(workflow.tasks):
            # Find tasks ready to be scheduled
            ready_tasks = []
            for task_id, task in workflow.tasks.items():
                if task_id not in scheduled_tasks:
                    dependencies = workflow.task_dependencies.get(task_id, [])
                    if all(dep in scheduled_tasks for dep in dependencies):
                        ready_tasks.append((task_id, task))
            
            # Schedule ready tasks
            for task_id, task in ready_tasks:
                execution_id = self.schedule_task(task, {
                    'workflow_id': workflow.workflow_id,
                    'workflow_context': workflow.metadata
                })
                execution_ids[task_id] = execution_id
                scheduled_tasks.add(task_id)
            
            # Safety check to prevent infinite loop
            if not ready_tasks and len(scheduled_tasks) < len(workflow.tasks):
                logger.error("Circular dependency or missing tasks in workflow %s", workflow.workflow_id)
                break
        
        logger.info("Scheduled %d tasks for workflow %s", len(execution_ids), workflow.workflow_id)
        return execution_ids
    
    async def execute_next_tasks(self) -> List[TaskExecution]:
        """
        Execute the next available tasks based on scheduling strategy
        
        Returns:
            List of task executions started
        """
        started_executions = []
        
        with self._lock:
            # Determine how many tasks we can start
            available_slots = self.max_concurrent_tasks - len(self.running_tasks)
            if available_slots <= 0:
                return started_executions
            
            # Get tasks ready for execution
            tasks_to_start = []
            temp_queue = []
            
            while self.task_queue and len(tasks_to_start) < available_slots:
                priority, exec_id, task, task_execution = heapq.heappop(self.task_queue)
                
                # Find suitable system for this task
                suitable_system = self._find_best_system_for_task(task)
                
                if suitable_system:
                    tasks_to_start.append((task, task_execution, suitable_system))
                else:
                    # Put back in queue if no suitable system available
                    temp_queue.append((priority, exec_id, task, task_execution))
            
            # Put back tasks that couldn't be scheduled
            for item in temp_queue:
                heapq.heappush(self.task_queue, item)
            
            # Start the selected tasks
            for task, task_execution, system in tasks_to_start:
                self._start_task_execution(task, task_execution, system)
                started_executions.append(task_execution)
        
        # Execute tasks asynchronously
        if started_executions:
            await self._execute_tasks_async(started_executions)
        
        return started_executions
    
    def get_system_recommendations(self, task: TaskDefinition) -> List[Tuple[str, float]]:
        """
        Get system recommendations for a task with suitability scores
        
        Args:
            task: Task to find systems for
            
        Returns:
            List of (system_id, suitability_score) tuples, sorted by score
        """
        recommendations = []
        
        with self._system_lock:
            for system_id, system in self.systems.items():
                if system.can_execute_task(task):
                    score = self._calculate_system_suitability_score(system, task)
                    recommendations.append((system_id, score))
        
        # Sort by suitability score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def optimize_scheduling(self, optimization_objective: OptimizationObjective) -> OptimizationMetrics:
        """
        Optimize scheduling parameters based on historical performance
        
        Args:
            optimization_objective: Objective to optimize for
            
        Returns:
            Optimization metrics and results
        """
        start_time = time.time()
        
        # Collect current metrics
        before_metrics = self._collect_performance_metrics()
        
        # Apply optimization based on objective
        if optimization_objective == OptimizationObjective.MINIMIZE_TIME:
            self._optimize_for_speed()
        elif optimization_objective == OptimizationObjective.MINIMIZE_RESOURCE_USAGE:
            self._optimize_for_resources()
        elif optimization_objective == OptimizationObjective.MAXIMIZE_QUALITY:
            self._optimize_for_quality()
        elif optimization_objective == OptimizationObjective.BALANCE_ALL:
            self._optimize_balanced()
        
        # Collect after metrics
        after_metrics = self._collect_performance_metrics()
        
        # Create optimization result
        optimization = OptimizationMetrics(
            optimization_type=optimization_objective.value,
            before_duration=before_metrics.get('average_task_duration', 0.0),
            before_resource_usage=before_metrics.get('resource_usage', {}),
            before_success_rate=before_metrics.get('success_rate', 0.0),
            after_duration=after_metrics.get('average_task_duration', 0.0),
            after_resource_usage=after_metrics.get('resource_usage', {}),
            after_success_rate=after_metrics.get('success_rate', 0.0),
            optimization_description=f"Optimized scheduler for {optimization_objective.value}"
        )
        optimization.calculate_improvements()
        
        optimization_time = time.time() - start_time
        logger.info("Scheduling optimization completed in %.2f seconds", optimization_time)
        
        return optimization
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of task queues and systems"""
        with self._lock:
            queue_length = len(self.task_queue)
            running_count = len(self.running_tasks)
            completed_count = len(self.completed_tasks)
            failed_count = len(self.failed_tasks)
        
        with self._system_lock:
            system_loads = {
                system_id: system.get_load_score()
                for system_id, system in self.systems.items()
            }
        
        return {
            'queued_tasks': queue_length,
            'running_tasks': running_count,
            'completed_tasks': completed_count,
            'failed_tasks': failed_count,
            'available_systems': len(self.systems),
            'system_loads': system_loads,
            'scheduling_metrics': self.scheduling_metrics.copy()
        }
    
    def _calculate_dynamic_priority(self, task: TaskDefinition, context: Dict[str, Any] = None) -> float:
        """Calculate dynamic priority score for a task"""
        base_score = calculate_task_priority_score(task, context)
        
        # Apply scheduling strategy adjustments
        if self.scheduling_strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
            if task.estimated_duration_seconds:
                # Favor shorter tasks
                duration_factor = max(0.1, min(2.0, 300 / task.estimated_duration_seconds))
                base_score *= duration_factor
        
        elif self.scheduling_strategy == SchedulingStrategy.RESOURCE_OPTIMIZED:
            # Favor tasks that use resources efficiently
            resource_efficiency = self._calculate_resource_efficiency(task)
            base_score *= (1.0 + resource_efficiency)
        
        elif self.scheduling_strategy == SchedulingStrategy.DEADLINE_AWARE:
            # Boost priority for tasks approaching deadline
            if context and 'deadline' in task.metadata:
                deadline_urgency = self._calculate_deadline_urgency(task, context)
                base_score *= (1.0 + deadline_urgency)
        
        return base_score
    
    def _find_best_system_for_task(self, task: TaskDefinition) -> Optional[SystemStatus]:
        """Find the best available system for executing a task"""
        suitable_systems = []
        
        with self._system_lock:
            for system in self.systems.values():
                if system.can_execute_task(task):
                    suitability_score = self._calculate_system_suitability_score(system, task)
                    suitable_systems.append((system, suitability_score))
        
        if not suitable_systems:
            return None
        
        # Sort by suitability and return the best system
        suitable_systems.sort(key=lambda x: x[1], reverse=True)
        return suitable_systems[0][0]
    
    def _calculate_system_suitability_score(self, system: SystemStatus, task: TaskDefinition) -> float:
        """Calculate how suitable a system is for executing a task"""
        if not system.can_execute_task(task):
            return 0.0
        
        score = 1.0
        
        # Factor in current load (prefer less loaded systems)
        load_factor = max(0.1, 1.0 - system.get_load_score())
        score *= load_factor
        
        # Factor in success rate
        score *= system.success_rate
        
        # Factor in estimated duration vs system performance
        if task.estimated_duration_seconds and system.average_task_duration > 0:
            duration_factor = min(2.0, system.average_task_duration / task.estimated_duration_seconds)
            score *= duration_factor
        
        # Factor in resource availability
        resource_availability_score = 1.0
        for resource in task.required_resources:
            available = system.available_resources.get(resource.resource_type, 0.0)
            if available > 0:
                utilization = min(1.0, resource.required_amount / available)
                resource_availability_score *= (1.0 - utilization * 0.5)
        
        score *= resource_availability_score
        
        return score
    
    def _start_task_execution(self, task: TaskDefinition, task_execution: TaskExecution, system: SystemStatus):
        """Start executing a task on a system"""
        task_execution.mark_started(system.system_id)
        self.running_tasks[task_execution.execution_id] = task_execution
        
        # Update system status
        system.assigned_tasks.add(task_execution.task_id)
        system.current_load += 0.1  # Approximate load increase
        
        # Reserve resources
        for resource in task.required_resources:
            available = system.available_resources.get(resource.resource_type, 0.0)
            resource.allocate(available)
            system.reserved_resources[resource.resource_type] = (
                system.reserved_resources.get(resource.resource_type, 0.0) + 
                resource.reserved_amount
            )
        
        logger.debug("Started task %s on system %s", task.task_id, system.system_id)
    
    async def _execute_tasks_async(self, task_executions: List[TaskExecution]):
        """Execute multiple tasks asynchronously"""
        async def execute_single_task(task_execution: TaskExecution):
            try:
                # Simulate task execution (in real implementation, this would call the actual function)
                if hasattr(task_execution, 'task') and task_execution.task.function:
                    result = await task_execution.task.function(**task_execution.task.parameters)
                    task_execution.mark_completed(result)
                else:
                    # Simulate execution time
                    await asyncio.sleep(0.1)
                    task_execution.mark_completed({"status": "simulated_success"})
                
                # Move to completed
                self._complete_task_execution(task_execution)
                
            except Exception as e:
                task_execution.mark_failed(str(e))
                self._complete_task_execution(task_execution)
        
        # Execute all tasks concurrently
        await asyncio.gather(*[execute_single_task(execution) for execution in task_executions])
    
    def _complete_task_execution(self, task_execution: TaskExecution):
        """Complete a task execution and update system status"""
        with self._lock:
            if task_execution.execution_id in self.running_tasks:
                del self.running_tasks[task_execution.execution_id]
                
                if task_execution.status == TaskStatus.COMPLETED:
                    self.completed_tasks[task_execution.execution_id] = task_execution
                else:
                    self.failed_tasks[task_execution.execution_id] = task_execution
        
        # Update system status
        if task_execution.assigned_system:
            with self._system_lock:
                system = self.systems.get(task_execution.assigned_system)
                if system:
                    system.assigned_tasks.discard(task_execution.task_id)
                    system.current_load = max(0.0, system.current_load - 0.1)
                    
                    # Update system metrics
                    if task_execution.duration_seconds:
                        system.average_task_duration = (
                            system.average_task_duration * 0.9 + 
                            task_execution.duration_seconds * 0.1
                        )
                    
                    # Update success rate
                    success = 1.0 if task_execution.status == TaskStatus.COMPLETED else 0.0
                    system.success_rate = system.success_rate * 0.95 + success * 0.05
    
    def _build_dependency_graph(self, workflow: WorkflowDefinition) -> Dict[str, Set[str]]:
        """Build dependency graph for workflow tasks"""
        graph = defaultdict(set)
        for task_id, dependencies in workflow.task_dependencies.items():
            for dep in dependencies:
                graph[dep].add(task_id)
        return dict(graph)
    
    def _reassign_system_tasks(self, system_id: str):
        """Reassign tasks from a system that's being removed"""
        tasks_to_reassign = []
        
        # Find tasks assigned to this system
        for execution in self.running_tasks.values():
            if execution.assigned_system == system_id:
                tasks_to_reassign.append(execution)
        
        # Attempt to reassign tasks
        for task_execution in tasks_to_reassign:
            # Find alternative system
            # In a real implementation, this would involve more sophisticated reassignment logic
            logger.warning("Task %s reassignment needed (system %s unavailable)", 
                         task_execution.task_id, system_id)
    
    def _optimize_for_speed(self):
        """Optimize scheduler for minimum execution time"""
        self.scheduling_strategy = SchedulingStrategy.PRIORITY_FIRST
        self.max_concurrent_tasks = min(20, self.max_concurrent_tasks * 2)
        logger.info("Optimized scheduler for speed")
    
    def _optimize_for_resources(self):
        """Optimize scheduler for minimum resource usage"""
        self.scheduling_strategy = SchedulingStrategy.RESOURCE_OPTIMIZED
        self.max_concurrent_tasks = max(5, self.max_concurrent_tasks // 2)
        logger.info("Optimized scheduler for resource efficiency")
    
    def _optimize_for_quality(self):
        """Optimize scheduler for maximum quality"""
        self.scheduling_strategy = SchedulingStrategy.LOAD_BALANCED
        self.resource_optimization_enabled = True
        logger.info("Optimized scheduler for quality")
    
    def _optimize_balanced(self):
        """Optimize scheduler for balanced performance"""
        self.scheduling_strategy = SchedulingStrategy.LOAD_BALANCED
        self.load_balancing_enabled = True
        self.resource_optimization_enabled = True
        logger.info("Optimized scheduler for balanced performance")
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        completed_durations = [
            execution.duration_seconds for execution in self.completed_tasks.values()
            if execution.duration_seconds is not None
        ]
        
        avg_duration = statistics.mean(completed_durations) if completed_durations else 0.0
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        success_rate = len(self.completed_tasks) / max(1, total_tasks)
        
        return {
            'average_task_duration': avg_duration,
            'success_rate': success_rate,
            'resource_usage': self._calculate_total_resource_usage(),
            'total_tasks': total_tasks
        }
    
    def _calculate_total_resource_usage(self) -> Dict[str, float]:
        """Calculate total resource usage across all systems"""
        total_usage = defaultdict(float)
        
        with self._system_lock:
            for system in self.systems.values():
                for resource_type, usage in system.reserved_resources.items():
                    total_usage[resource_type.value] += usage
        
        return dict(total_usage)
    
    def _calculate_resource_efficiency(self, task: TaskDefinition) -> float:
        """Calculate resource efficiency score for a task"""
        if not task.required_resources:
            return 0.5  # Neutral score
        
        # Simple efficiency calculation based on resource requirements
        total_resources = sum(resource.required_amount for resource in task.required_resources)
        estimated_duration = task.estimated_duration_seconds or 60.0
        
        # Efficiency is inverse of resource*time
        efficiency = 1.0 / max(1.0, total_resources * estimated_duration / 3600.0)
        return min(1.0, efficiency)
    
    def _calculate_deadline_urgency(self, task: TaskDefinition, context: Dict[str, Any]) -> float:
        """Calculate deadline urgency factor"""
        deadline = task.metadata.get('deadline')
        if not deadline:
            return 0.0
        
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        
        current_time = context.get('current_time', datetime.now())
        time_remaining = (deadline - current_time).total_seconds()
        
        if time_remaining <= 0:
            return 2.0  # Maximum urgency for overdue tasks
        elif time_remaining < 3600:  # Less than 1 hour
            return 1.5
        elif time_remaining < 86400:  # Less than 1 day
            return 1.0
        else:
            return 0.0
    
    def _update_scheduling_metrics(self, scheduling_time: float):
        """Update internal scheduling performance metrics"""
        current_avg = self.scheduling_metrics['average_scheduling_time']
        total_scheduled = self.scheduling_metrics['total_tasks_scheduled']
        
        # Update running average
        self.scheduling_metrics['average_scheduling_time'] = (
            (current_avg * (total_scheduled - 1) + scheduling_time) / total_scheduled
        )


# Factory function
def create_task_scheduler(max_concurrent_tasks: int = 10) -> TaskScheduler:
    """Create and configure a task scheduler"""
    return TaskScheduler(max_concurrent_tasks=max_concurrent_tasks)