"""
Workflow Scheduler - Advanced Workflow Execution and Task Orchestration
=======================================================================

Enterprise-grade workflow scheduler with intelligent task orchestration, load balancing,
and adaptive execution strategies for autonomous intelligence systems.
Implements advanced scheduling algorithms and real-time optimization.

This module provides comprehensive workflow execution capabilities including priority-based
scheduling, load balancing, fault tolerance, and performance optimization.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: workflow_scheduler.py (450 lines)
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from queue import PriorityQueue
import heapq
import threading
import statistics

from .workflow_types import (
    WorkflowDefinition, WorkflowExecution, WorkflowTask, SystemCapabilities,
    WorkflowStatus, TaskStatus, WorkflowPriority, OptimizationObjective
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentWorkflowScheduler:
    """Advanced workflow scheduler with intelligent orchestration"""
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.max_concurrent_workflows = max_concurrent_workflows
        self.scheduler_running = False
        
        # Execution tracking
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_queue = PriorityQueue()
        self.task_queue = PriorityQueue()
        
        # System management
        self.system_load: Dict[str, float] = defaultdict(float)
        self.system_capabilities: Dict[str, SystemCapabilities] = {}
        
        # Performance tracking
        self.execution_history: List[WorkflowExecution] = []
        self.performance_metrics: Dict[str, float] = defaultdict(float)
        
        # Scheduling optimization
        self.load_balancing_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.scheduling_policies: Dict[str, Any] = {
            'priority_weight': 0.4,
            'load_weight': 0.3,
            'performance_weight': 0.3,
            'max_retries': 3,
            'retry_delay': 2.0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Intelligent Workflow Scheduler initialized")
    
    async def start_scheduler(self):
        """Start the workflow scheduler with background processing"""
        if self.scheduler_running:
            self.logger.warning("Scheduler already running")
            return
        
        self.scheduler_running = True
        
        # Start background tasks
        scheduler_tasks = [
            asyncio.create_task(self._workflow_execution_loop()),
            asyncio.create_task(self._task_scheduling_loop()),
            asyncio.create_task(self._load_monitoring_loop()),
            asyncio.create_task(self._performance_optimization_loop())
        ]
        
        self.logger.info("Workflow scheduler started with background processing")
        return scheduler_tasks
    
    async def schedule_workflow(self, workflow: WorkflowDefinition, 
                              constraints: Dict[str, Any] = None) -> str:
        """Schedule workflow for execution"""
        try:
            # Create workflow execution
            execution = WorkflowExecution(
                execution_id=f"exec_{workflow.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                workflow_id=workflow.workflow_id
            )
            
            # Apply constraints
            if constraints:
                execution.execution_constraints = constraints
            
            # Calculate priority score for scheduling
            priority_score = self._calculate_priority_score(workflow)
            
            # Add to scheduling queue
            self.workflow_queue.put((priority_score, datetime.now(), execution, workflow))
            
            self.logger.info(f"Scheduled workflow {workflow.workflow_id} with priority {priority_score}")
            return execution.execution_id
            
        except Exception as e:
            self.logger.error(f"Error scheduling workflow {workflow.workflow_id}: {e}")
            raise
    
    def _calculate_priority_score(self, workflow: WorkflowDefinition) -> float:
        """Calculate priority score for workflow scheduling (lower = higher priority)"""
        # Base priority from enum value
        base_priority = workflow.priority.value
        
        # Adjust for optimization objective urgency
        urgency_adjustment = 0.0
        if workflow.optimization_objective == OptimizationObjective.MINIMIZE_TIME:
            urgency_adjustment = -0.5  # Higher priority for time-critical
        elif workflow.optimization_objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            urgency_adjustment = -0.3  # Moderate priority increase
        
        # Adjust for workflow complexity (simpler tasks get slight priority)
        complexity_score = workflow.calculate_complexity_score()
        complexity_adjustment = complexity_score * 0.1
        
        # Calculate final score
        priority_score = base_priority + urgency_adjustment + complexity_adjustment
        
        return max(1.0, priority_score)  # Ensure positive priority
    
    async def _workflow_execution_loop(self):
        """Main workflow execution loop"""
        self.logger.info("Starting workflow execution loop")
        
        while self.scheduler_running:
            try:
                # Check if we can start new workflows
                if (len(self.active_workflows) < self.max_concurrent_workflows and 
                    not self.workflow_queue.empty()):
                    
                    # Get next workflow from queue
                    priority_score, queued_at, execution, workflow = self.workflow_queue.get()
                    
                    # Start workflow execution
                    await self._start_workflow_execution(workflow, execution)
                
                # Monitor active workflows
                await self._monitor_active_workflows()
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in workflow execution loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _start_workflow_execution(self, workflow: WorkflowDefinition, 
                                      execution: WorkflowExecution):
        """Start executing a workflow"""
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            
            # Add to active workflows
            self.active_workflows[execution.execution_id] = execution
            
            # Schedule initial ready tasks
            ready_tasks = self._get_ready_tasks(workflow, execution)
            for task in ready_tasks:
                await self._schedule_task(workflow, execution, task)
            
            self.logger.info(f"Started workflow execution: {execution.execution_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting workflow execution: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_messages.append(str(e))
    
    def _get_ready_tasks(self, workflow: WorkflowDefinition, 
                        execution: WorkflowExecution) -> List[WorkflowTask]:
        """Get tasks ready for execution"""
        ready_tasks = []
        completed_task_ids = set(execution.completed_tasks)
        
        for task in workflow.tasks:
            if (task.status == TaskStatus.PENDING and 
                task.is_ready_to_execute(completed_task_ids) and
                len(execution.current_tasks) < workflow.max_parallel_tasks):
                ready_tasks.append(task)
        
        return ready_tasks
    
    async def _schedule_task(self, workflow: WorkflowDefinition, 
                           execution: WorkflowExecution, task: WorkflowTask):
        """Schedule individual task for execution"""
        try:
            # Select optimal system for task
            target_system = await self._select_optimal_system_for_task(task, workflow)
            task.target_system = target_system
            
            # Update task status
            task.status = TaskStatus.SCHEDULED
            task.scheduled_at = datetime.now()
            
            # Add to execution tracking
            execution.scheduled_tasks.append(task.task_id)
            
            # Calculate task priority
            task_priority = self._calculate_task_priority(task, workflow)
            
            # Add to task queue
            self.task_queue.put((task_priority, datetime.now(), task, workflow, execution))
            
            self.logger.debug(f"Scheduled task {task.task_id} on {target_system}")
            
        except Exception as e:
            self.logger.error(f"Error scheduling task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
    
    async def _select_optimal_system_for_task(self, task: WorkflowTask, 
                                            workflow: WorkflowDefinition) -> str:
        """Select optimal system for task execution"""
        if not self.system_capabilities:
            return task.target_system or 'default'
        
        best_system = None
        best_score = 0.0
        
        for system_id, capabilities in self.system_capabilities.items():
            if capabilities.can_execute_task(task):
                score = capabilities.calculate_suitability_score(task, workflow.optimization_objective)
                
                # Adjust for current load
                current_load = self.system_load.get(system_id, 0.0)
                load_penalty = current_load * 0.3
                adjusted_score = score * (1.0 - load_penalty)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_system = system_id
        
        return best_system or task.target_system or 'default'
    
    def _calculate_task_priority(self, task: WorkflowTask, 
                               workflow: WorkflowDefinition) -> float:
        """Calculate task priority for scheduling"""
        # Base priority from workflow
        base_priority = workflow.priority.value
        
        # Adjust for task-specific priority
        task_priority_adjustment = 0.0
        if hasattr(task, 'priority'):
            task_priority_adjustment = task.priority.value * 0.2
        
        # Adjust for critical path position
        critical_path_adjustment = 0.0
        if len(task.dependencies) == 0:  # Start tasks get higher priority
            critical_path_adjustment = -0.3
        
        # Adjust for estimated duration (shorter tasks get slight priority)
        duration_adjustment = min(0.2, task.estimated_duration / 300.0)  # Max 0.2 for 5min+ tasks
        
        priority_score = base_priority + task_priority_adjustment + critical_path_adjustment + duration_adjustment
        return max(1.0, priority_score)
    
    async def _task_scheduling_loop(self):
        """Task scheduling and execution loop"""
        self.logger.info("Starting task scheduling loop")
        
        while self.scheduler_running:
            try:
                if not self.task_queue.empty():
                    # Get next task from queue
                    priority, queued_at, task, workflow, execution = self.task_queue.get()
                    
                    # Execute task
                    await self._execute_task(workflow, execution, task)
                
                await asyncio.sleep(0.5)  # Check every half second
                
            except Exception as e:
                self.logger.error(f"Error in task scheduling loop: {e}")
                await asyncio.sleep(2.0)
    
    async def _execute_task(self, workflow: WorkflowDefinition, 
                          execution: WorkflowExecution, task: WorkflowTask):
        """Execute individual task"""
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Add to current tasks
            execution.current_tasks.append(task.task_id)
            
            # Update system load
            self.system_load[task.target_system] += 0.1
            
            self.logger.debug(f"Executing task {task.task_id} on {task.target_system}")
            
            # Simulate task execution (replace with actual system calls)
            await self._simulate_task_execution(task)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration = (task.completed_at - task.started_at).total_seconds()
            
            # Update execution tracking
            execution.current_tasks.remove(task.task_id)
            execution.completed_tasks.append(task.task_id)
            execution.task_results[task.task_id] = task.results
            
            # Update progress
            execution.progress = len(execution.completed_tasks) / len(workflow.tasks)
            
            # Update system load
            self.system_load[task.target_system] = max(0.0, self.system_load[task.target_system] - 0.1)
            
            self.logger.debug(f"Task {task.task_id} completed in {task.actual_duration:.2f}s")
            
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
        await asyncio.sleep(min(execution_time, 0.1))  # Cap simulation time for demo
        
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
            },
            "performance_score": np.random.uniform(0.7, 1.0)
        }
        
        # Update task performance metrics
        task.performance_metrics = {
            "execution_efficiency": task.calculate_efficiency(),
            "quality_score": task.results["metrics"]["accuracy"],
            "resource_utilization": np.random.uniform(0.6, 0.9)
        }
    
    async def _handle_task_failure(self, workflow: WorkflowDefinition, 
                                 execution: WorkflowExecution, 
                                 task: WorkflowTask, error_message: str):
        """Handle task failure with intelligent retry logic"""
        task.error_message = error_message
        task.retry_count += 1
        
        self.logger.warning(f"Task {task.task_id} failed: {error_message} (retry {task.retry_count}/{task.max_retries})")
        
        # Update system load
        self.system_load[task.target_system] = max(0.0, self.system_load[task.target_system] - 0.1)
        
        if task.retry_count <= task.max_retries:
            # Retry task with exponential backoff
            task.status = TaskStatus.RETRYING
            retry_delay = self.scheduling_policies['retry_delay'] ** task.retry_count
            await asyncio.sleep(min(retry_delay, 30.0))  # Cap at 30 seconds
            
            # Optionally select different system for retry
            if task.retry_count > 1:
                alternative_system = await self._select_alternative_system(task, workflow)
                if alternative_system:
                    task.target_system = alternative_system
            
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
    
    async def _select_alternative_system(self, task: WorkflowTask, 
                                       workflow: WorkflowDefinition) -> Optional[str]:
        """Select alternative system for task retry"""
        current_system = task.target_system
        
        for system_id, capabilities in self.system_capabilities.items():
            if (system_id != current_system and 
                capabilities.can_execute_task(task) and 
                self.system_load.get(system_id, 0.0) < 0.8):
                return system_id
        
        return None
    
    async def _check_workflow_failure(self, workflow: WorkflowDefinition, 
                                    execution: WorkflowExecution):
        """Check if workflow should fail due to task failures"""
        failure_threshold = 1.0 - workflow.minimum_success_rate
        current_failure_rate = len(execution.failed_tasks) / len(workflow.tasks)
        
        if current_failure_rate > failure_threshold:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.now()
            
            self.logger.error(f"Workflow {execution.execution_id} failed due to excessive task failures")
            
            # Remove from active workflows
            if execution.execution_id in self.active_workflows:
                del self.active_workflows[execution.execution_id]
            
            # Add to execution history
            self.execution_history.append(execution)
    
    async def _complete_workflow(self, workflow: WorkflowDefinition, 
                               execution: WorkflowExecution):
        """Complete workflow execution with comprehensive metrics"""
        execution.status = WorkflowStatus.COMPLETED
        execution.completed_at = datetime.now()
        execution.progress = 1.0
        
        # Calculate comprehensive execution metrics
        total_duration = (execution.completed_at - execution.started_at).total_seconds()
        
        # Task performance metrics
        completed_tasks = [task for task in workflow.tasks if task.status == TaskStatus.COMPLETED]
        
        execution.execution_metrics = {
            "total_duration": total_duration,
            "estimated_duration": sum(task.estimated_duration for task in workflow.tasks),
            "tasks_completed": len(execution.completed_tasks),
            "tasks_failed": len(execution.failed_tasks),
            "success_rate": len(execution.completed_tasks) / len(workflow.tasks),
            "average_task_duration": statistics.mean([
                task.actual_duration for task in completed_tasks 
                if task.actual_duration is not None
            ]) if completed_tasks else 0,
            "efficiency_score": execution.calculate_efficiency_score(),
            "quality_score": np.mean([
                task.performance_metrics.get('quality_score', 0.8) 
                for task in completed_tasks
            ]) if completed_tasks else 0.8
        }
        
        # Resource utilization metrics
        execution.resource_usage = self._calculate_resource_usage(workflow, execution)
        execution.cost_incurred = self._calculate_execution_cost(workflow, execution)
        
        self.logger.info(f"Workflow {execution.execution_id} completed successfully in {total_duration:.2f}s")
        
        # Remove from active workflows and add to history
        if execution.execution_id in self.active_workflows:
            del self.active_workflows[execution.execution_id]
        
        self.execution_history.append(execution)
        
        # Update performance metrics
        self._update_performance_metrics(execution)
    
    def _calculate_resource_usage(self, workflow: WorkflowDefinition, 
                                execution: WorkflowExecution) -> Dict[str, float]:
        """Calculate total resource usage for workflow execution"""
        resource_usage = defaultdict(float)
        
        for task in workflow.tasks:
            if task.status == TaskStatus.COMPLETED:
                for resource, amount in task.resource_requirements.items():
                    duration = task.actual_duration or task.estimated_duration
                    resource_usage[resource] += amount * (duration / 3600.0)  # Convert to hours
        
        return dict(resource_usage)
    
    def _calculate_execution_cost(self, workflow: WorkflowDefinition, 
                                execution: WorkflowExecution) -> float:
        """Calculate total execution cost"""
        total_cost = 0.0
        
        for task in workflow.tasks:
            if task.status == TaskStatus.COMPLETED:
                # Simple cost model based on duration and resource usage
                duration_hours = (task.actual_duration or task.estimated_duration) / 3600.0
                base_cost = duration_hours * 0.10  # $0.10 per hour base rate
                
                # Add resource costs
                resource_cost = sum(task.resource_requirements.values()) * 0.05
                
                total_cost += base_cost + resource_cost
        
        return total_cost
    
    def _update_performance_metrics(self, execution: WorkflowExecution):
        """Update overall scheduler performance metrics"""
        self.performance_metrics['total_executions'] += 1
        
        if execution.status == WorkflowStatus.COMPLETED:
            self.performance_metrics['successful_executions'] += 1
        else:
            self.performance_metrics['failed_executions'] += 1
        
        # Update success rate
        total = self.performance_metrics['total_executions']
        successful = self.performance_metrics['successful_executions']
        self.performance_metrics['success_rate'] = successful / total if total > 0 else 0.0
        
        # Update average execution time
        if execution.execution_metrics:
            duration = execution.execution_metrics.get('total_duration', 0)
            current_avg = self.performance_metrics.get('average_execution_time', 0)
            self.performance_metrics['average_execution_time'] = (
                (current_avg * (total - 1) + duration) / total
            )
    
    async def _monitor_active_workflows(self):
        """Monitor active workflows for timeouts and issues"""
        current_time = datetime.now()
        
        workflows_to_timeout = []
        
        for execution_id, execution in self.active_workflows.items():
            # Check for workflow timeout
            if execution.started_at:
                runtime = (current_time - execution.started_at).total_seconds()
                
                # Find the original workflow definition (simplified lookup)
                workflow_timeout = 3600.0  # Default 1 hour timeout
                
                if runtime > workflow_timeout:
                    workflows_to_timeout.append(execution_id)
        
        # Handle timeouts
        for execution_id in workflows_to_timeout:
            execution = self.active_workflows[execution_id]
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = current_time
            execution.error_messages.append("Workflow timeout exceeded")
            
            del self.active_workflows[execution_id]
            self.execution_history.append(execution)
            
            self.logger.warning(f"Workflow {execution_id} timed out")
    
    async def _load_monitoring_loop(self):
        """Monitor and update system load metrics"""
        self.logger.info("Starting load monitoring loop")
        
        while self.scheduler_running:
            try:
                await self._update_system_load()
                await self._balance_system_loads()
                
                await asyncio.sleep(30.0)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in load monitoring loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _update_system_load(self):
        """Update system load metrics"""
        # Simulate system load decay over time
        for system_id in list(self.system_load.keys()):
            current_load = self.system_load[system_id]
            # Gradual load decay
            new_load = max(0.0, current_load * 0.95)
            self.system_load[system_id] = new_load
    
    async def _balance_system_loads(self):
        """Balance loads across systems"""
        if not self.system_load:
            return
        
        # Simple load balancing - reduce weights for overloaded systems
        for system_id, load in self.system_load.items():
            if load > 0.8:  # High load threshold
                self.load_balancing_weights[system_id] = max(0.1, self.load_balancing_weights[system_id] * 0.9)
            elif load < 0.3:  # Low load - increase weight
                self.load_balancing_weights[system_id] = min(2.0, self.load_balancing_weights[system_id] * 1.1)
    
    async def _performance_optimization_loop(self):
        """Continuous performance optimization"""
        self.logger.info("Starting performance optimization loop")
        
        while self.scheduler_running:
            try:
                await self._optimize_scheduling_policies()
                await self._analyze_execution_patterns()
                
                await asyncio.sleep(300.0)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance optimization loop: {e}")
                await asyncio.sleep(600.0)
    
    async def _optimize_scheduling_policies(self):
        """Optimize scheduling policies based on performance data"""
        if len(self.execution_history) < 10:
            return
        
        # Analyze recent execution performance
        recent_executions = self.execution_history[-50:]  # Last 50 executions
        
        # Calculate metrics
        success_rates = [exec.calculate_success_rate() for exec in recent_executions]
        avg_success_rate = np.mean(success_rates)
        
        # Adjust policies based on performance
        if avg_success_rate < 0.8:
            # Increase retry attempts for low success rate
            self.scheduling_policies['max_retries'] = min(5, self.scheduling_policies['max_retries'] + 1)
            self.logger.info("Increased max retries due to low success rate")
        elif avg_success_rate > 0.95:
            # Decrease retries for high success rate
            self.scheduling_policies['max_retries'] = max(1, self.scheduling_policies['max_retries'] - 1)
    
    async def _analyze_execution_patterns(self):
        """Analyze execution patterns for optimization opportunities"""
        if len(self.execution_history) < 20:
            return
        
        # Analyze system performance patterns
        system_performance = defaultdict(list)
        
        for execution in self.execution_history[-100:]:  # Last 100 executions
            for task_id, results in execution.task_results.items():
                if isinstance(results, dict) and 'metrics' in results:
                    # Extract system from task (simplified)
                    system_id = 'default'  # Would be extracted from task info
                    performance = results['metrics'].get('accuracy', 0.8)
                    system_performance[system_id].append(performance)
        
        # Update system reliability scores
        for system_id, performances in system_performance.items():
            if system_id in self.system_capabilities:
                avg_performance = np.mean(performances)
                self.system_capabilities[system_id].reliability_score = avg_performance
    
    # Public API methods
    def register_system_capabilities(self, system_id: str, capabilities: SystemCapabilities):
        """Register system capabilities for scheduling"""
        self.system_capabilities[system_id] = capabilities
        self.system_load[system_id] = capabilities.current_load
        self.logger.info(f"Registered system capabilities: {system_id}")
    
    def stop_scheduler(self):
        """Stop the workflow scheduler"""
        self.scheduler_running = False
        self.logger.info("Workflow scheduler stopped")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        return {
            "scheduler_running": self.scheduler_running,
            "active_workflows": len(self.active_workflows),
            "queued_workflows": self.workflow_queue.qsize(),
            "queued_tasks": self.task_queue.qsize(),
            "system_loads": dict(self.system_load),
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "performance_metrics": dict(self.performance_metrics),
            "scheduling_policies": self.scheduling_policies.copy(),
            "load_balancing_weights": dict(self.load_balancing_weights)
        }
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific workflow execution"""
        execution = self.active_workflows.get(execution_id)
        if not execution:
            # Check execution history
            for hist_exec in self.execution_history:
                if hist_exec.execution_id == execution_id:
                    execution = hist_exec
                    break
        
        if not execution:
            return None
        
        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress": execution.progress,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "scheduled_tasks": execution.scheduled_tasks,
            "current_tasks": execution.current_tasks,
            "completed_tasks": execution.completed_tasks,
            "failed_tasks": execution.failed_tasks,
            "execution_metrics": execution.execution_metrics,
            "error_messages": execution.error_messages
        }


# Export main classes
__all__ = [
    'IntelligentWorkflowScheduler'
]