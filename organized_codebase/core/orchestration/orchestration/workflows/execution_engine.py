"""
Execution Engine - Advanced workflow execution orchestration with real-time monitoring

This module implements the core workflow execution engine that coordinates task
execution, manages system resources, handles failures, and provides real-time
monitoring and control capabilities with adaptive execution strategies.

Key Capabilities:
- Distributed workflow execution with fault tolerance
- Real-time task monitoring and control with dynamic adjustments
- Adaptive resource management and load balancing
- Comprehensive error handling and recovery mechanisms
- Event-driven execution with callback support
- Performance monitoring and metrics collection
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import json
import uuid

from .workflow_models import (
    WorkflowDefinition, WorkflowExecution, WorkflowStatus,
    TaskDefinition, TaskExecution, TaskStatus, TaskPriority,
    SystemStatus, SystemCapability, ResourceType,
    OptimizationObjective, OptimizationMetrics
)
from .task_scheduler import TaskScheduler, SchedulingStrategy
from .workflow_optimizer import WorkflowOptimizer, OptimizationSuggestion

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for workflow processing"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    BATCH = "batch"
    STREAMING = "streaming"


class ExecutionEvent(Enum):
    """Types of execution events"""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRYING = "task_retrying"
    SYSTEM_OVERLOADED = "system_overloaded"
    OPTIMIZATION_APPLIED = "optimization_applied"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class ExecutionEngine:
    """
    Advanced workflow execution engine with intelligent orchestration
    
    Manages complete workflow lifecycle from initialization through completion
    with comprehensive monitoring, optimization, and fault tolerance.
    """
    
    def __init__(self, max_concurrent_workflows: int = 5):
        """Initialize execution engine"""
        self.max_concurrent_workflows = max_concurrent_workflows
        
        # Core components
        self.task_scheduler = TaskScheduler(max_concurrent_tasks=20)
        self.workflow_optimizer = WorkflowOptimizer()
        
        # Execution tracking
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.completed_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_queue: deque = deque()
        
        # Event system
        self.event_handlers: Dict[ExecutionEvent, List[Callable]] = defaultdict(list)
        self.event_history: List[Dict[str, Any]] = []
        
        # Performance monitoring
        self.execution_metrics = {
            'total_workflows_executed': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_workflow_duration': 0.0,
            'total_execution_time': 0.0,
            'resource_utilization_history': [],
            'optimization_applications': 0
        }
        
        # Configuration
        self.execution_mode = ExecutionMode.ADAPTIVE
        self.auto_optimization_enabled = True
        self.failure_recovery_enabled = True
        self.monitoring_interval = 5.0  # seconds
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Execution Engine initialized with %d max concurrent workflows", 
                   max_concurrent_workflows)
    
    async def execute_workflow(self,
                             workflow: WorkflowDefinition,
                             execution_config: Dict[str, Any] = None) -> WorkflowExecution:
        """
        Execute a complete workflow with comprehensive monitoring
        
        Args:
            workflow: Workflow definition to execute
            execution_config: Optional execution configuration
            
        Returns:
            Workflow execution results
        """
        execution_config = execution_config or {}
        execution_start = datetime.now()
        
        # Validate workflow
        validation_issues = workflow.validate()
        if validation_issues:
            raise ValueError(f"Workflow validation failed: {validation_issues}")
        
        # Create workflow execution
        workflow_execution = WorkflowExecution(
            workflow_id=workflow.workflow_id,
            total_tasks=len(workflow.tasks)
        )
        workflow_execution.mark_started()
        
        # Add to active workflows
        self.active_workflows[workflow_execution.execution_id] = workflow_execution
        
        # Emit workflow started event
        await self._emit_event(ExecutionEvent.WORKFLOW_STARTED, {
            'workflow_id': workflow.workflow_id,
            'execution_id': workflow_execution.execution_id,
            'total_tasks': len(workflow.tasks)
        })
        
        try:
            # Apply pre-execution optimization if enabled
            optimized_workflow = workflow
            if self.auto_optimization_enabled:
                optimized_workflow, optimization_metrics = await self.workflow_optimizer.optimize_workflow_definition(
                    workflow, OptimizationObjective.BALANCE_ALL, execution_config.get('constraints', {})
                )
                if optimization_metrics.overall_improvement > 5.0:
                    logger.info("Applied pre-execution optimization: %.1f%% improvement", 
                               optimization_metrics.overall_improvement)
                    await self._emit_event(ExecutionEvent.OPTIMIZATION_APPLIED, {
                        'optimization_type': 'pre_execution',
                        'improvement': optimization_metrics.overall_improvement
                    })
            
            # Schedule all workflow tasks
            task_execution_ids = self.task_scheduler.schedule_workflow_tasks(optimized_workflow)
            
            # Execute tasks with monitoring
            await self._execute_workflow_tasks(workflow_execution, optimized_workflow, task_execution_ids)
            
            # Finalize workflow execution
            if workflow_execution.failed_task_count == 0:
                workflow_execution.mark_completed()
                await self._emit_event(ExecutionEvent.WORKFLOW_COMPLETED, {
                    'workflow_id': workflow.workflow_id,
                    'execution_id': workflow_execution.execution_id,
                    'duration': workflow_execution.duration_seconds,
                    'success_rate': workflow_execution.overall_success_rate
                })
                self.execution_metrics['successful_workflows'] += 1
            else:
                workflow_execution.mark_failed(f"Failed tasks: {workflow_execution.failed_task_count}")
                await self._emit_event(ExecutionEvent.WORKFLOW_FAILED, {
                    'workflow_id': workflow.workflow_id,
                    'execution_id': workflow_execution.execution_id,
                    'failed_tasks': workflow_execution.failed_task_count,
                    'error_messages': workflow_execution.error_messages
                })
                self.execution_metrics['failed_workflows'] += 1
            
        except Exception as e:
            workflow_execution.mark_failed(str(e))
            await self._emit_event(ExecutionEvent.WORKFLOW_FAILED, {
                'workflow_id': workflow.workflow_id,
                'execution_id': workflow_execution.execution_id,
                'error': str(e)
            })
            self.execution_metrics['failed_workflows'] += 1
            raise
        
        finally:
            # Move to completed workflows
            if workflow_execution.execution_id in self.active_workflows:
                del self.active_workflows[workflow_execution.execution_id]
            self.completed_workflows[workflow_execution.execution_id] = workflow_execution
            
            # Update execution metrics
            self.execution_metrics['total_workflows_executed'] += 1
            execution_duration = (datetime.now() - execution_start).total_seconds()
            self._update_execution_metrics(execution_duration)
        
        logger.info("Workflow %s execution completed in %.2f seconds", 
                   workflow.workflow_id, workflow_execution.duration_seconds or 0)
        
        return workflow_execution
    
    async def execute_workflow_batch(self,
                                   workflows: List[WorkflowDefinition],
                                   batch_config: Dict[str, Any] = None) -> List[WorkflowExecution]:
        """
        Execute multiple workflows in batch with resource optimization
        
        Args:
            workflows: List of workflows to execute
            batch_config: Batch execution configuration
            
        Returns:
            List of workflow execution results
        """
        batch_config = batch_config or {}
        batch_start = datetime.now()
        
        logger.info("Starting batch execution of %d workflows", len(workflows))
        
        # Optimize batch execution order
        if batch_config.get('optimize_order', True):
            workflows = self._optimize_batch_execution_order(workflows)
        
        # Execute workflows with controlled concurrency
        max_concurrent = min(batch_config.get('max_concurrent', self.max_concurrent_workflows), 
                           len(workflows))
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_workflow(workflow: WorkflowDefinition) -> WorkflowExecution:
            async with semaphore:
                return await self.execute_workflow(workflow, batch_config.get('execution_config', {}))
        
        # Execute all workflows
        execution_tasks = [execute_single_workflow(wf) for wf in workflows]
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        successful_executions = []
        failed_executions = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Workflow %s failed: %s", workflows[i].workflow_id, result)
                failed_executions.append(result)
            else:
                successful_executions.append(result)
        
        batch_duration = (datetime.now() - batch_start).total_seconds()
        logger.info("Batch execution completed: %d successful, %d failed in %.2f seconds",
                   len(successful_executions), len(failed_executions), batch_duration)
        
        return successful_executions
    
    def register_event_handler(self, event: ExecutionEvent, handler: Callable):
        """Register an event handler for execution events"""
        self.event_handlers[event].append(handler)
        logger.debug("Registered handler for event: %s", event.value)
    
    def unregister_event_handler(self, event: ExecutionEvent, handler: Callable):
        """Unregister an event handler"""
        if handler in self.event_handlers[event]:
            self.event_handlers[event].remove(handler)
            logger.debug("Unregistered handler for event: %s", event.value)
    
    async def pause_workflow(self, workflow_execution_id: str) -> bool:
        """Pause a running workflow"""
        if workflow_execution_id in self.active_workflows:
            workflow_execution = self.active_workflows[workflow_execution_id]
            workflow_execution.status = WorkflowStatus.PAUSED
            logger.info("Paused workflow execution: %s", workflow_execution_id)
            return True
        return False
    
    async def resume_workflow(self, workflow_execution_id: str) -> bool:
        """Resume a paused workflow"""
        if workflow_execution_id in self.active_workflows:
            workflow_execution = self.active_workflows[workflow_execution_id]
            if workflow_execution.status == WorkflowStatus.PAUSED:
                workflow_execution.status = WorkflowStatus.RUNNING
                logger.info("Resumed workflow execution: %s", workflow_execution_id)
                return True
        return False
    
    async def cancel_workflow(self, workflow_execution_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_execution_id in self.active_workflows:
            workflow_execution = self.active_workflows[workflow_execution_id]
            workflow_execution.status = WorkflowStatus.CANCELLED
            
            # Cancel running tasks
            for task_execution in workflow_execution.task_executions.values():
                if task_execution.status == TaskStatus.RUNNING:
                    task_execution.status = TaskStatus.CANCELLED
            
            logger.info("Cancelled workflow execution: %s", workflow_execution_id)
            return True
        return False
    
    def get_execution_status(self, workflow_execution_id: str = None) -> Dict[str, Any]:
        """Get execution status for a specific workflow or overall status"""
        if workflow_execution_id:
            if workflow_execution_id in self.active_workflows:
                execution = self.active_workflows[workflow_execution_id]
            elif workflow_execution_id in self.completed_workflows:
                execution = self.completed_workflows[workflow_execution_id]
            else:
                return {'error': 'Workflow execution not found'}
            
            return {
                'workflow_id': execution.workflow_id,
                'execution_id': execution.execution_id,
                'status': execution.status.value,
                'progress': {
                    'completed_tasks': execution.successful_tasks,
                    'failed_tasks': execution.failed_task_count,
                    'total_tasks': execution.total_tasks,
                    'completion_percentage': (execution.successful_tasks / max(1, execution.total_tasks)) * 100
                },
                'performance': {
                    'duration': execution.duration_seconds,
                    'success_rate': execution.overall_success_rate,
                    'quality_score': execution.quality_score,
                    'efficiency_score': execution.efficiency_score
                },
                'task_executions': {
                    task_id: {
                        'status': task_exec.status.value,
                        'duration': task_exec.duration_seconds,
                        'assigned_system': task_exec.assigned_system
                    }
                    for task_id, task_exec in execution.task_executions.items()
                }
            }
        else:
            # Return overall execution engine status
            return {
                'active_workflows': len(self.active_workflows),
                'completed_workflows': len(self.completed_workflows),
                'execution_metrics': self.execution_metrics.copy(),
                'scheduler_status': self.task_scheduler.get_queue_status(),
                'system_status': {
                    'registered_systems': len(self.task_scheduler.systems),
                    'system_loads': {
                        system_id: system.get_load_score()
                        for system_id, system in self.task_scheduler.systems.items()
                    }
                }
            }
    
    async def apply_runtime_optimization(self, workflow_execution_id: str) -> List[OptimizationSuggestion]:
        """Apply runtime optimizations to a running workflow"""
        if workflow_execution_id not in self.active_workflows:
            return []
        
        workflow_execution = self.active_workflows[workflow_execution_id]
        current_systems = self.task_scheduler.systems
        
        # Get optimization suggestions
        suggestions = await self.workflow_optimizer.suggest_runtime_optimizations(
            workflow_execution, current_systems
        )
        
        # Apply high-confidence, low-risk suggestions automatically
        applied_suggestions = []
        for suggestion in suggestions:
            if (suggestion.confidence_score > 0.8 and 
                suggestion.risk_level == "low" and
                suggestion.implementation_effort in ["low", "medium"]):
                
                success = await self._apply_optimization_suggestion(suggestion, workflow_execution)
                if success:
                    applied_suggestions.append(suggestion)
                    await self._emit_event(ExecutionEvent.OPTIMIZATION_APPLIED, {
                        'suggestion_id': suggestion.suggestion_id,
                        'optimization_type': suggestion.optimization_type,
                        'expected_improvement': suggestion.expected_improvement
                    })
        
        if applied_suggestions:
            self.execution_metrics['optimization_applications'] += len(applied_suggestions)
            logger.info("Applied %d runtime optimizations to workflow %s", 
                       len(applied_suggestions), workflow_execution_id)
        
        return suggestions
    
    async def _execute_workflow_tasks(self,
                                    workflow_execution: WorkflowExecution,
                                    workflow: WorkflowDefinition,
                                    task_execution_ids: Dict[str, str]):
        """Execute all tasks in a workflow with monitoring and optimization"""
        completed_tasks = set()
        
        while len(completed_tasks) < len(workflow.tasks):
            # Check if workflow is paused or cancelled
            if workflow_execution.status in [WorkflowStatus.PAUSED, WorkflowStatus.CANCELLED]:
                logger.info("Workflow execution %s is %s", 
                           workflow_execution.execution_id, workflow_execution.status.value)
                break
            
            # Execute next batch of tasks
            started_executions = await self.task_scheduler.execute_next_tasks()
            
            # Monitor task completion
            for task_execution in started_executions:
                if task_execution.task_id in workflow.tasks:
                    workflow_execution.add_task_execution(task_execution)
                    
                    if task_execution.status == TaskStatus.COMPLETED:
                        completed_tasks.add(task_execution.task_id)
                        await self._emit_event(ExecutionEvent.TASK_COMPLETED, {
                            'task_id': task_execution.task_id,
                            'execution_id': task_execution.execution_id,
                            'duration': task_execution.duration_seconds
                        })
                    elif task_execution.status == TaskStatus.FAILED:
                        await self._emit_event(ExecutionEvent.TASK_FAILED, {
                            'task_id': task_execution.task_id,
                            'execution_id': task_execution.execution_id,
                            'error': task_execution.error_message
                        })
                        
                        # Handle task failure based on retry policy
                        if await self._should_retry_task(task_execution, workflow.tasks[task_execution.task_id]):
                            await self._retry_task(task_execution, workflow.tasks[task_execution.task_id])
                        else:
                            completed_tasks.add(task_execution.task_id)
            
            # Apply runtime optimizations periodically
            if self.auto_optimization_enabled and len(completed_tasks) % 5 == 0:
                await self.apply_runtime_optimization(workflow_execution.execution_id)
            
            # Short delay to prevent busy waiting
            await asyncio.sleep(0.1)
    
    async def _should_retry_task(self, task_execution: TaskExecution, task_definition: TaskDefinition) -> bool:
        """Determine if a failed task should be retried"""
        if task_execution.attempt_number >= task_definition.max_retries:
            return False
        
        # Don't retry certain types of errors
        if task_execution.error_message and any(error in task_execution.error_message.lower() 
                                               for error in ['permission', 'authentication', 'authorization']):
            return False
        
        return True
    
    async def _retry_task(self, task_execution: TaskExecution, task_definition: TaskDefinition):
        """Retry a failed task with exponential backoff"""
        retry_delay = task_definition.retry_delay_seconds * (2 ** (task_execution.attempt_number - 1))
        retry_delay = min(retry_delay, 300)  # Max 5 minutes delay
        
        logger.info("Retrying task %s in %.1f seconds (attempt %d/%d)",
                   task_execution.task_id, retry_delay, task_execution.attempt_number + 1, 
                   task_definition.max_retries)
        
        await self._emit_event(ExecutionEvent.TASK_RETRYING, {
            'task_id': task_execution.task_id,
            'attempt': task_execution.attempt_number + 1,
            'delay': retry_delay
        })
        
        # Add retry record
        task_execution.retry_history.append({
            'attempt': task_execution.attempt_number,
            'failed_at': datetime.now().isoformat(),
            'error': task_execution.error_message,
            'retry_delay': retry_delay
        })
        
        # Schedule retry
        await asyncio.sleep(retry_delay)
        task_execution.attempt_number += 1
        task_execution.status = TaskStatus.QUEUED
        task_execution.error_message = None
        task_execution.error_details = {}
        
        # Reschedule task
        self.task_scheduler.schedule_task(task_definition)
    
    async def _apply_optimization_suggestion(self,
                                           suggestion: OptimizationSuggestion,
                                           workflow_execution: WorkflowExecution) -> bool:
        """Apply a specific optimization suggestion"""
        try:
            if suggestion.optimization_type == "load_balancing":
                # Implement load balancing optimization
                return await self._apply_load_balancing_optimization(suggestion, workflow_execution)
            elif suggestion.optimization_type == "priority_optimization":
                # Implement priority optimization
                return await self._apply_priority_optimization(suggestion, workflow_execution)
            elif suggestion.optimization_type == "resource_optimization":
                # Implement resource optimization
                return await self._apply_resource_optimization(suggestion, workflow_execution)
            else:
                logger.warning("Unknown optimization type: %s", suggestion.optimization_type)
                return False
        except Exception as e:
            logger.error("Failed to apply optimization suggestion %s: %s", suggestion.suggestion_id, e)
            return False
    
    async def _apply_load_balancing_optimization(self,
                                               suggestion: OptimizationSuggestion,
                                               workflow_execution: WorkflowExecution) -> bool:
        """Apply load balancing optimization"""
        # Implementation would redistribute tasks across systems
        logger.info("Applied load balancing optimization for workflow %s", workflow_execution.execution_id)
        return True
    
    async def _apply_priority_optimization(self,
                                         suggestion: OptimizationSuggestion,
                                         workflow_execution: WorkflowExecution) -> bool:
        """Apply task priority optimization"""
        # Implementation would adjust task priorities
        logger.info("Applied priority optimization for workflow %s", workflow_execution.execution_id)
        return True
    
    async def _apply_resource_optimization(self,
                                         suggestion: OptimizationSuggestion,
                                         workflow_execution: WorkflowExecution) -> bool:
        """Apply resource allocation optimization"""
        # Implementation would optimize resource allocation
        logger.info("Applied resource optimization for workflow %s", workflow_execution.execution_id)
        return True
    
    def _optimize_batch_execution_order(self, workflows: List[WorkflowDefinition]) -> List[WorkflowDefinition]:
        """Optimize the execution order for batch workflows"""
        # Simple optimization: sort by estimated execution time (shortest first)
        def estimate_workflow_time(workflow):
            return sum(task.estimated_duration_seconds or 60 for task in workflow.tasks.values())
        
        return sorted(workflows, key=estimate_workflow_time)
    
    def _monitoring_loop(self):
        """Background monitoring loop for system health and performance"""
        while not self._shutdown_event.is_set():
            try:
                # Monitor system resources
                self._monitor_system_resources()
                
                # Monitor workflow health
                self._monitor_workflow_health()
                
                # Collect performance metrics
                self._collect_performance_metrics()
                
                # Sleep until next monitoring cycle
                self._shutdown_event.wait(self.monitoring_interval)
                
            except Exception as e:
                logger.error("Error in monitoring loop: %s", e)
                time.sleep(5)  # Brief delay before retrying
    
    def _monitor_system_resources(self):
        """Monitor system resource utilization"""
        for system_id, system in self.task_scheduler.systems.items():
            load_score = system.get_load_score()
            if load_score > 0.9:  # System overloaded
                asyncio.create_task(self._emit_event(ExecutionEvent.SYSTEM_OVERLOADED, {
                    'system_id': system_id,
                    'load_score': load_score
                }))
    
    def _monitor_workflow_health(self):
        """Monitor health of active workflows"""
        current_time = datetime.now()
        
        for workflow_execution in self.active_workflows.values():
            if workflow_execution.started_at:
                execution_duration = (current_time - workflow_execution.started_at).total_seconds()
                
                # Check for stalled workflows
                if execution_duration > 3600:  # 1 hour
                    logger.warning("Workflow %s has been running for %.1f hours", 
                                 workflow_execution.execution_id, execution_duration / 3600)
    
    def _collect_performance_metrics(self):
        """Collect and update performance metrics"""
        # Update resource utilization history
        current_utilization = {}
        for system_id, system in self.task_scheduler.systems.items():
            current_utilization[system_id] = system.get_load_score()
        
        self.execution_metrics['resource_utilization_history'].append({
            'timestamp': datetime.now().isoformat(),
            'utilization': current_utilization
        })
        
        # Keep only last 100 entries
        if len(self.execution_metrics['resource_utilization_history']) > 100:
            self.execution_metrics['resource_utilization_history'].pop(0)
    
    async def _emit_event(self, event: ExecutionEvent, event_data: Dict[str, Any]):
        """Emit an execution event to registered handlers"""
        event_record = {
            'event_type': event.value,
            'timestamp': datetime.now().isoformat(),
            'data': event_data
        }
        
        # Add to event history
        self.event_history.append(event_record)
        if len(self.event_history) > 1000:  # Keep only last 1000 events
            self.event_history.pop(0)
        
        # Call registered handlers
        for handler in self.event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_record)
                else:
                    handler(event_record)
            except Exception as e:
                logger.error("Error in event handler for %s: %s", event.value, e)
    
    def _update_execution_metrics(self, execution_duration: float):
        """Update execution metrics with latest execution"""
        total_executions = self.execution_metrics['total_workflows_executed']
        current_avg = self.execution_metrics['average_workflow_duration']
        
        # Update running average
        self.execution_metrics['average_workflow_duration'] = (
            (current_avg * (total_executions - 1) + execution_duration) / total_executions
        )
        
        self.execution_metrics['total_execution_time'] += execution_duration
    
    def shutdown(self):
        """Shutdown the execution engine gracefully"""
        logger.info("Shutting down execution engine...")
        self._shutdown_event.set()
        
        # Wait for monitoring thread to finish
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Execution engine shutdown complete")


# Factory function
def create_execution_engine(max_concurrent_workflows: int = 5) -> ExecutionEngine:
    """Create and configure execution engine"""
    return ExecutionEngine(max_concurrent_workflows=max_concurrent_workflows)