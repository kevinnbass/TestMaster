"""
Workflow Scheduler - Intelligent Workflow Scheduling and Execution

This module implements the WorkflowScheduler component that manages workflow
execution with intelligent load balancing, task scheduling, and system monitoring.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import logging
import statistics
import uuid
from datetime import datetime
from queue import PriorityQueue
from typing import Dict, List, Any

import numpy as np

from .workflow_types import (
    WorkflowDefinition, WorkflowExecution, WorkflowTask,
    WorkflowStatus, TaskStatus
)

logger = logging.getLogger(__name__)


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