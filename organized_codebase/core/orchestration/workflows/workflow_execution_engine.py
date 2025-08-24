"""
Workflow Execution Engine
========================

High-performance workflow execution engine that coordinates workflow steps
across all unified systems with intelligent scheduling, error recovery,
and real-time monitoring.

Integrates with:
- Cross-System APIs for system communication
- Workflow Framework for workflow definitions
- Visual Designer for execution monitoring
- Unified State Manager for execution state persistence

Author: TestMaster Phase 1B Integration System
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import threading
from queue import Queue, PriorityQueue

# Import workflow framework
from .workflow_framework import (
    WorkflowDefinition, WorkflowExecution, WorkflowStep, 
    WorkflowStatus, StepStatus, WorkflowStepType
)

# Import cross-system APIs
from .cross_system_apis import (
    SystemType, SystemMessage, CrossSystemRequest, CrossSystemResponse,
    cross_system_coordinator
)


# ============================================================================
# EXECUTION ENGINE TYPES
# ============================================================================

class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class ExecutionPriority(Enum):
    """Execution priority levels"""
    LOW = 1
    NORMAL = 3
    HIGH = 5
    CRITICAL = 8
    EMERGENCY = 10


@dataclass
class ExecutionContext:
    """Context for workflow execution"""
    execution_id: str
    workflow_definition: WorkflowDefinition
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    
    # Execution state
    current_step: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_context: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    start_time: Optional[datetime] = None
    step_timings: Dict[str, float] = field(default_factory=dict)
    
    # Execution control
    pause_requested: bool = False
    cancel_requested: bool = False
    
    def get_variable(self, name: str) -> Any:
        """Get variable value with fallback to workflow defaults"""
        if name in self.variables:
            return self.variables[name]
        
        # Check workflow default variables
        if self.workflow_definition:
            for var in self.workflow_definition.variables:
                if var.name == name:
                    return var.default_value
        
        return None
    
    def set_variable(self, name: str, value: Any):
        """Set variable value"""
        self.variables[name] = value
    
    def interpolate_value(self, value: Any) -> Any:
        """Interpolate variables in value"""
        if isinstance(value, str) and "{{" in value and "}}" in value:
            # Simple template interpolation
            for var_name, var_value in self.variables.items():
                placeholder = f"{{{{{var_name}}}}}"
                if placeholder in value:
                    value = value.replace(placeholder, str(var_value))
            
            # Interpolate step results
            for step_id, result in self.step_results.items():
                placeholder = f"{{{{{step_id}.result}}}}"
                if placeholder in value:
                    value = value.replace(placeholder, str(result))
        
        elif isinstance(value, dict):
            return {k: self.interpolate_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.interpolate_value(item) for item in value]
        
        return value


@dataclass
class StepExecution:
    """Individual step execution tracking"""
    step: WorkflowStep
    context: ExecutionContext
    
    # Execution state
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    
    # Retry state
    current_attempt: int = 0
    max_attempts: int = 3
    next_retry_time: Optional[datetime] = None
    
    # Dependencies
    waiting_for: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    def can_execute(self) -> bool:
        """Check if step can be executed"""
        return (len(self.waiting_for) == 0 and 
                self.step.status == StepStatus.WAITING and
                not self.context.pause_requested and
                not self.context.cancel_requested)
    
    def needs_retry(self) -> bool:
        """Check if step needs retry"""
        return (self.step.status == StepStatus.FAILED and
                self.current_attempt < self.max_attempts and
                (not self.next_retry_time or datetime.now() >= self.next_retry_time))


# ============================================================================
# WORKFLOW EXECUTION ENGINE
# ============================================================================

class WorkflowExecutionEngine:
    """
    High-performance workflow execution engine with intelligent scheduling,
    parallel execution, error recovery, and real-time monitoring.
    """
    
    def __init__(self, max_concurrent_workflows: int = 50, max_concurrent_steps: int = 100):
        self.logger = logging.getLogger("workflow_execution_engine")
        
        # Execution management
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_concurrent_steps = max_concurrent_steps
        
        # Active executions
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.step_executions: Dict[str, Dict[str, StepExecution]] = {}
        
        # Execution queues
        self.execution_queue: PriorityQueue = PriorityQueue()
        self.step_queue: PriorityQueue = PriorityQueue()
        
        # Thread pools
        self.workflow_executor = ThreadPoolExecutor(max_workers=max_concurrent_workflows, 
                                                   thread_name_prefix="workflow")
        self.step_executor = ThreadPoolExecutor(max_workers=max_concurrent_steps,
                                               thread_name_prefix="step")
        
        # Monitoring and statistics
        self.execution_stats = {
            "total_workflows_executed": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_steps_executed": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "average_workflow_time": 0.0,
            "average_step_time": 0.0
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "workflow_started": [],
            "workflow_completed": [],
            "workflow_failed": [],
            "step_started": [],
            "step_completed": [],
            "step_failed": []
        }
        
        # Engine state
        self.is_running = False
        self.shutdown_requested = False
        
        # Background tasks
        self.scheduler_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info("Workflow execution engine initialized")
    
    async def start_engine(self):
        """Start the workflow execution engine"""
        if self.is_running:
            return
        
        self.logger.info("Starting workflow execution engine")
        self.is_running = True
        
        # Start background tasks
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        self.logger.info("Workflow execution engine started")
    
    async def stop_engine(self):
        """Stop the workflow execution engine"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping workflow execution engine")
        self.shutdown_requested = True
        
        # Cancel background tasks
        if self.scheduler_task:
            self.scheduler_task.cancel()
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Wait for active executions to complete (with timeout)
        timeout = 30.0
        start_time = time.time()
        
        while self.active_executions and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)
        
        # Force stop remaining executions
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_workflow(execution_id)
        
        # Shutdown thread pools
        self.workflow_executor.shutdown(wait=True)
        self.step_executor.shutdown(wait=True)
        
        self.is_running = False
        self.logger.info("Workflow execution engine stopped")
    
    async def execute_workflow(self, workflow_definition: WorkflowDefinition,
                             variables: Dict[str, Any] = None,
                             execution_mode: ExecutionMode = ExecutionMode.HYBRID,
                             priority: ExecutionPriority = ExecutionPriority.NORMAL) -> str:
        """Start workflow execution"""
        try:
            # Create execution context
            execution_id = f"exec_{uuid.uuid4().hex[:12]}"
            
            context = ExecutionContext(
                execution_id=execution_id,
                workflow_definition=workflow_definition,
                variables=variables or {},
                execution_mode=execution_mode,
                priority=priority
            )
            
            # Create workflow execution
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_definition=workflow_definition,
                status=WorkflowStatus.PENDING,
                variables=context.variables.copy()
            )
            
            # Store execution state
            self.active_executions[execution_id] = execution
            self.execution_contexts[execution_id] = context
            
            # Initialize step executions
            self._initialize_step_executions(execution_id, context)
            
            # Queue for execution
            priority_value = 10 - priority.value  # Lower number = higher priority
            self.execution_queue.put((priority_value, time.time(), execution_id))
            
            self.logger.info(f"Queued workflow execution: {execution_id}")
            
            # Fire workflow started event
            await self._fire_event("workflow_started", execution_id, context)
            
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Failed to start workflow execution: {e}")
            raise
    
    def _initialize_step_executions(self, execution_id: str, context: ExecutionContext):
        """Initialize step execution tracking"""
        step_executions = {}
        
        # Create step execution objects
        for step in context.workflow_definition.steps:
            step_exec = StepExecution(step=step, context=context)
            
            # Configure retry settings
            retry_config = step.retry_config
            step_exec.max_attempts = retry_config.get("max_attempts", 3)
            
            # Set up dependencies
            step_exec.waiting_for = set(step.depends_on)
            
            step_executions[step.step_id] = step_exec
        
        # Set up dependent relationships
        for step_id, step_exec in step_executions.items():
            for dep_id in step_exec.step.depends_on:
                if dep_id in step_executions:
                    step_executions[dep_id].dependents.add(step_id)
        
        self.step_executions[execution_id] = step_executions
    
    async def _scheduler_loop(self):
        """Main scheduler loop for workflow and step execution"""
        while not self.shutdown_requested:
            try:
                # Process workflow queue
                await self._process_workflow_queue()
                
                # Process step queue
                await self._process_step_queue()
                
                # Schedule ready steps
                await self._schedule_ready_steps()
                
                # Handle retries
                await self._handle_step_retries()
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_workflow_queue(self):
        """Process queued workflows"""
        processed = 0
        max_batch = 10
        
        while not self.execution_queue.empty() and processed < max_batch:
            try:
                priority, timestamp, execution_id = self.execution_queue.get_nowait()
                
                if execution_id in self.active_executions:
                    # Start workflow execution
                    await self._start_workflow_execution(execution_id)
                    processed += 1
                
            except Exception as e:
                self.logger.error(f"Error processing workflow queue: {e}")
                break
    
    async def _start_workflow_execution(self, execution_id: str):
        """Start actual workflow execution"""
        execution = self.active_executions.get(execution_id)
        context = self.execution_contexts.get(execution_id)
        
        if not execution or not context:
            return
        
        try:
            # Update execution status
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()
            context.start_time = execution.start_time
            
            # Schedule initial steps (those with no dependencies)
            for step_id, step_exec in self.step_executions[execution_id].items():
                if step_exec.can_execute():
                    await self._queue_step_execution(execution_id, step_id)
            
            self.logger.info(f"Started workflow execution: {execution_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start workflow execution {execution_id}: {e}")
            await self._fail_workflow(execution_id, str(e))
    
    async def _queue_step_execution(self, execution_id: str, step_id: str):
        """Queue step for execution"""
        context = self.execution_contexts.get(execution_id)
        if not context:
            return
        
        # Priority: workflow priority + step urgency
        priority_value = 10 - context.priority.value
        
        self.step_queue.put((priority_value, time.time(), execution_id, step_id))
        
        # Update step status
        step_exec = self.step_executions[execution_id][step_id]
        step_exec.step.status = StepStatus.RUNNING
    
    async def _process_step_queue(self):
        """Process queued steps"""
        processed = 0
        max_batch = 20
        
        while not self.step_queue.empty() and processed < max_batch:
            try:
                priority, timestamp, execution_id, step_id = self.step_queue.get_nowait()
                
                if (execution_id in self.active_executions and 
                    step_id in self.step_executions.get(execution_id, {})):
                    
                    # Execute step asynchronously
                    asyncio.create_task(self._execute_step(execution_id, step_id))
                    processed += 1
                
            except Exception as e:
                self.logger.error(f"Error processing step queue: {e}")
                break
    
    async def _execute_step(self, execution_id: str, step_id: str):
        """Execute individual workflow step"""
        try:
            step_exec = self.step_executions[execution_id][step_id]
            step = step_exec.step
            context = step_exec.context
            
            # Fire step started event
            await self._fire_event("step_started", execution_id, context, step_id)
            
            # Record start time
            step_exec.start_time = datetime.now()
            step.start_time = step_exec.start_time
            
            # Execute step based on type
            result = await self._execute_step_by_type(step, context)
            
            # Record completion
            step_exec.end_time = datetime.now()
            step.end_time = step_exec.end_time
            step_exec.execution_time_ms = (step_exec.end_time - step_exec.start_time).total_seconds() * 1000
            
            # Store result
            step.result = result
            step.status = StepStatus.COMPLETED
            context.step_results[step_id] = result
            context.step_timings[step_id] = step_exec.execution_time_ms
            
            # Update statistics
            self.execution_stats["total_steps_executed"] += 1
            self.execution_stats["successful_steps"] += 1
            
            # Fire step completed event
            await self._fire_event("step_completed", execution_id, context, step_id)
            
            # Check for workflow completion
            await self._check_workflow_completion(execution_id)
            
            # Schedule dependent steps
            await self._schedule_dependent_steps(execution_id, step_id)
            
            self.logger.info(f"Step {step_id} completed in {step_exec.execution_time_ms:.1f}ms")
            
        except Exception as e:
            await self._fail_step(execution_id, step_id, str(e))
    
    async def _execute_step_by_type(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute step based on its type"""
        
        if step.type == WorkflowStepType.SYSTEM_OPERATION:
            return await self._execute_system_operation(step, context)
        
        elif step.type == WorkflowStepType.CONDITIONAL:
            return await self._execute_conditional(step, context)
        
        elif step.type == WorkflowStepType.PARALLEL:
            return await self._execute_parallel(step, context)
        
        elif step.type == WorkflowStepType.LOOP:
            return await self._execute_loop(step, context)
        
        elif step.type == WorkflowStepType.DELAY:
            return await self._execute_delay(step, context)
        
        elif step.type == WorkflowStepType.HUMAN_APPROVAL:
            return await self._execute_human_approval(step, context)
        
        elif step.type == WorkflowStepType.DATA_TRANSFORM:
            return await self._execute_data_transform(step, context)
        
        elif step.type == WorkflowStepType.EXTERNAL_API:
            return await self._execute_external_api(step, context)
        
        else:
            raise ValueError(f"Unknown step type: {step.type}")
    
    async def _execute_system_operation(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute system operation step"""
        if not step.target_system or not step.operation:
            raise ValueError("System operation step must specify target_system and operation")
        
        # Interpolate parameters
        parameters = context.interpolate_value(step.parameters)
        
        # Execute cross-system operation
        response = await cross_system_coordinator.execute_cross_system_operation(
            operation=step.operation,
            target_system=step.target_system,
            parameters=parameters
        )
        
        if not response.success:
            raise RuntimeError(f"System operation failed: {response.error_message}")
        
        return response.result
    
    async def _execute_conditional(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute conditional step"""
        conditions = step.conditions
        if not conditions:
            return True
        
        # Simple condition evaluation
        for condition_name, condition_expr in conditions.items():
            # Interpolate condition expression
            expr = context.interpolate_value(condition_expr)
            
            # Basic condition evaluation (would need more sophisticated parser in production)
            if isinstance(expr, bool):
                result = expr
            elif isinstance(expr, str):
                # Simple comparisons
                if " > " in expr:
                    left, right = expr.split(" > ")
                    result = float(left.strip()) > float(right.strip())
                elif " < " in expr:
                    left, right = expr.split(" < ")
                    result = float(left.strip()) < float(right.strip())
                elif " == " in expr:
                    left, right = expr.split(" == ")
                    result = left.strip() == right.strip()
                else:
                    result = bool(expr)
            else:
                result = bool(expr)
            
            # Store condition result
            context.set_variable(f"{step.step_id}.{condition_name}", result)
        
        return True
    
    async def _execute_parallel(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute parallel step (placeholder)"""
        # In production, this would manage parallel execution branches
        return {"parallel_completed": True}
    
    async def _execute_loop(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute loop step (placeholder)"""
        # In production, this would manage loop iteration
        return {"loop_completed": True}
    
    async def _execute_delay(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute delay step"""
        delay_seconds = step.parameters.get("seconds", 1)
        delay_seconds = context.interpolate_value(delay_seconds)
        
        await asyncio.sleep(float(delay_seconds))
        return {"delayed_seconds": delay_seconds}
    
    async def _execute_human_approval(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute human approval step (placeholder)"""
        # In production, this would create approval request
        return {"approval_status": "auto_approved"}
    
    async def _execute_data_transform(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute data transformation step"""
        transformations = step.parameters.get("transformations", [])
        input_data = context.interpolate_value(step.parameters.get("input_data", {}))
        
        # Simple data transformations
        result = input_data
        
        for transform in transformations:
            if transform == "normalize_timestamps":
                # Placeholder transformation
                result["normalized_at"] = datetime.now().isoformat()
            elif transform == "aggregate_metrics":
                # Placeholder aggregation
                result["aggregated"] = True
            elif transform == "enrich_metadata":
                # Placeholder enrichment
                result["enriched"] = True
        
        return result
    
    async def _execute_external_api(self, step: WorkflowStep, context: ExecutionContext) -> Any:
        """Execute external API step (placeholder)"""
        # In production, this would make actual HTTP requests
        return {"api_response": "success"}
    
    async def _fail_step(self, execution_id: str, step_id: str, error_message: str):
        """Handle step failure"""
        step_exec = self.step_executions[execution_id][step_id]
        step = step_exec.step
        context = step_exec.context
        
        step.status = StepStatus.FAILED
        step.error_message = error_message
        step_exec.current_attempt += 1
        
        # Update statistics
        self.execution_stats["failed_steps"] += 1
        
        self.logger.error(f"Step {step_id} failed: {error_message}")
        
        # Fire step failed event
        await self._fire_event("step_failed", execution_id, context, step_id)
        
        # Check if step should be retried
        if step_exec.needs_retry():
            retry_delay = step.retry_config.get("delay_seconds", 5)
            step_exec.next_retry_time = datetime.now() + timedelta(seconds=retry_delay)
            step.status = StepStatus.RETRYING
            
            self.logger.info(f"Step {step_id} will retry in {retry_delay} seconds")
        else:
            # Fail workflow if step cannot be retried
            await self._fail_workflow(execution_id, f"Step {step_id} failed: {error_message}")
    
    async def _schedule_dependent_steps(self, execution_id: str, completed_step_id: str):
        """Schedule steps that depend on the completed step"""
        step_executions = self.step_executions.get(execution_id, {})
        completed_step = step_executions.get(completed_step_id)
        
        if not completed_step:
            return
        
        # Update dependent steps
        for dependent_step_id in completed_step.dependents:
            dependent_step = step_executions.get(dependent_step_id)
            
            if dependent_step and completed_step_id in dependent_step.waiting_for:
                dependent_step.waiting_for.remove(completed_step_id)
                
                # Schedule if all dependencies are satisfied
                if dependent_step.can_execute():
                    await self._queue_step_execution(execution_id, dependent_step_id)
    
    async def _handle_step_retries(self):
        """Handle step retries"""
        current_time = datetime.now()
        
        for execution_id, step_executions in self.step_executions.items():
            for step_id, step_exec in step_executions.items():
                if (step_exec.needs_retry() and 
                    step_exec.next_retry_time and 
                    current_time >= step_exec.next_retry_time):
                    
                    # Reset step status and queue for retry
                    step_exec.step.status = StepStatus.WAITING
                    step_exec.next_retry_time = None
                    
                    await self._queue_step_execution(execution_id, step_id)
                    
                    self.logger.info(f"Retrying step {step_id} (attempt {step_exec.current_attempt})")
    
    async def _check_workflow_completion(self, execution_id: str):
        """Check if workflow has completed"""
        execution = self.active_executions.get(execution_id)
        step_executions = self.step_executions.get(execution_id, {})
        
        if not execution:
            return
        
        # Check if all steps are completed or failed
        completed_steps = 0
        failed_steps = 0
        
        for step_exec in step_executions.values():
            if step_exec.step.status == StepStatus.COMPLETED:
                completed_steps += 1
            elif step_exec.step.status == StepStatus.FAILED:
                failed_steps += 1
        
        total_steps = len(step_executions)
        
        if completed_steps == total_steps:
            # Workflow completed successfully
            await self._complete_workflow(execution_id)
        elif completed_steps + failed_steps == total_steps:
            # Workflow failed (some steps failed and cannot be retried)
            await self._fail_workflow(execution_id, "Some workflow steps failed")
    
    async def _complete_workflow(self, execution_id: str):
        """Complete workflow execution"""
        execution = self.active_executions.get(execution_id)
        context = self.execution_contexts.get(execution_id)
        
        if not execution or not context:
            return
        
        # Update execution status
        execution.status = WorkflowStatus.COMPLETED
        execution.end_time = datetime.now()
        execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
        
        # Update statistics
        self.execution_stats["total_workflows_executed"] += 1
        self.execution_stats["successful_workflows"] += 1
        
        # Fire workflow completed event
        await self._fire_event("workflow_completed", execution_id, context)
        
        self.logger.info(f"Workflow {execution_id} completed in {execution.total_execution_time:.2f}s")
        
        # Clean up (could be moved to background cleanup task)
        # For now, keep execution data for monitoring
    
    async def _fail_workflow(self, execution_id: str, error_message: str):
        """Fail workflow execution"""
        execution = self.active_executions.get(execution_id)
        context = self.execution_contexts.get(execution_id)
        
        if not execution or not context:
            return
        
        # Update execution status
        execution.status = WorkflowStatus.FAILED
        execution.end_time = datetime.now()
        
        if execution.start_time:
            execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
        
        # Store error context
        context.error_context["error_message"] = error_message
        context.error_context["failed_at"] = datetime.now().isoformat()
        
        # Update statistics
        self.execution_stats["total_workflows_executed"] += 1
        self.execution_stats["failed_workflows"] += 1
        
        # Fire workflow failed event
        await self._fire_event("workflow_failed", execution_id, context)
        
        self.logger.error(f"Workflow {execution_id} failed: {error_message}")
    
    async def _schedule_ready_steps(self):
        """Schedule steps that are ready to execute"""
        for execution_id, step_executions in self.step_executions.items():
            for step_id, step_exec in step_executions.items():
                if (step_exec.can_execute() and 
                    step_exec.step.status == StepStatus.WAITING):
                    await self._queue_step_execution(execution_id, step_id)
    
    async def _monitor_loop(self):
        """Monitor loop for execution health and cleanup"""
        while not self.shutdown_requested:
            try:
                # Update execution progress
                self._update_execution_progress()
                
                # Clean up completed executions (older than 1 hour)
                self._cleanup_old_executions()
                
                # Update statistics
                self._update_statistics()
                
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5.0)
    
    def _update_execution_progress(self):
        """Update execution progress for all active workflows"""
        for execution_id, execution in self.active_executions.items():
            step_executions = self.step_executions.get(execution_id, {})
            
            if step_executions:
                completed = sum(1 for s in step_executions.values() 
                              if s.step.status == StepStatus.COMPLETED)
                total = len(step_executions)
                
                # Update execution progress
                execution.completed_steps = set(s.step.step_id for s in step_executions.values() 
                                              if s.step.status == StepStatus.COMPLETED)
                execution.failed_steps = set(s.step.step_id for s in step_executions.values() 
                                           if s.step.status == StepStatus.FAILED)
                execution.running_steps = set(s.step.step_id for s in step_executions.values() 
                                            if s.step.status == StepStatus.RUNNING)
    
    def _cleanup_old_executions(self):
        """Clean up old completed executions"""
        cleanup_time = datetime.now() - timedelta(hours=1)
        
        to_remove = []
        for execution_id, execution in self.active_executions.items():
            if (execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and
                execution.end_time and execution.end_time < cleanup_time):
                to_remove.append(execution_id)
        
        for execution_id in to_remove:
            self.active_executions.pop(execution_id, None)
            self.execution_contexts.pop(execution_id, None)
            self.step_executions.pop(execution_id, None)
            
            self.logger.debug(f"Cleaned up old execution: {execution_id}")
    
    def _update_statistics(self):
        """Update execution statistics"""
        # Calculate average times
        completed_workflows = [e for e in self.active_executions.values() 
                             if e.status == WorkflowStatus.COMPLETED and e.total_execution_time > 0]
        
        if completed_workflows:
            total_time = sum(e.total_execution_time for e in completed_workflows)
            self.execution_stats["average_workflow_time"] = total_time / len(completed_workflows)
        
        # Calculate average step time
        all_step_timings = []
        for context in self.execution_contexts.values():
            all_step_timings.extend(context.step_timings.values())
        
        if all_step_timings:
            self.execution_stats["average_step_time"] = sum(all_step_timings) / len(all_step_timings)
    
    async def _fire_event(self, event_name: str, execution_id: str, context: ExecutionContext, 
                         step_id: Optional[str] = None):
        """Fire event to registered handlers"""
        handlers = self.event_handlers.get(event_name, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(execution_id, context, step_id)
                else:
                    handler(execution_id, context, step_id)
            except Exception as e:
                self.logger.error(f"Event handler error for {event_name}: {e}")
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def register_event_handler(self, event_name: str, handler: Callable):
        """Register event handler"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        
        self.event_handlers[event_name].append(handler)
        self.logger.info(f"Registered handler for event: {event_name}")
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause workflow execution"""
        context = self.execution_contexts.get(execution_id)
        execution = self.active_executions.get(execution_id)
        
        if context and execution:
            context.pause_requested = True
            execution.status = WorkflowStatus.PAUSED
            
            self.logger.info(f"Paused workflow: {execution_id}")
            return True
        
        return False
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume workflow execution"""
        context = self.execution_contexts.get(execution_id)
        execution = self.active_executions.get(execution_id)
        
        if context and execution and execution.status == WorkflowStatus.PAUSED:
            context.pause_requested = False
            execution.status = WorkflowStatus.RUNNING
            
            # Reschedule ready steps
            await self._schedule_ready_steps()
            
            self.logger.info(f"Resumed workflow: {execution_id}")
            return True
        
        return False
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        context = self.execution_contexts.get(execution_id)
        execution = self.active_executions.get(execution_id)
        
        if context and execution:
            context.cancel_requested = True
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            
            self.logger.info(f"Cancelled workflow: {execution_id}")
            return True
        
        return False
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        execution = self.active_executions.get(execution_id)
        context = self.execution_contexts.get(execution_id)
        step_executions = self.step_executions.get(execution_id, {})
        
        if not execution:
            return None
        
        return {
            "execution_id": execution_id,
            "status": execution.status.value,
            "progress_percentage": execution.get_progress_percentage(),
            "start_time": execution.start_time.isoformat() if execution.start_time else None,
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "total_execution_time": execution.total_execution_time,
            "completed_steps": len(execution.completed_steps),
            "failed_steps": len(execution.failed_steps),
            "running_steps": len(execution.running_steps),
            "total_steps": len(step_executions),
            "current_step": context.current_step if context else None,
            "variables": context.variables if context else {},
            "error_context": context.error_context if context else {}
        }
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get engine statistics and health"""
        return {
            "engine_status": "running" if self.is_running else "stopped",
            "active_workflows": len(self.active_executions),
            "queued_workflows": self.execution_queue.qsize(),
            "queued_steps": self.step_queue.qsize(),
            "execution_statistics": self.execution_stats.copy(),
            "resource_usage": {
                "max_concurrent_workflows": self.max_concurrent_workflows,
                "max_concurrent_steps": self.max_concurrent_steps,
                "workflow_thread_pool_size": self.workflow_executor._max_workers,
                "step_thread_pool_size": self.step_executor._max_workers
            }
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active workflow executions"""
        return [
            {
                "execution_id": execution_id,
                "workflow_name": execution.workflow_definition.name if execution.workflow_definition else "Unknown",
                "status": execution.status.value,
                "progress": execution.get_progress_percentage(),
                "start_time": execution.start_time.isoformat() if execution.start_time else None,
                "duration": (datetime.now() - execution.start_time).total_seconds() if execution.start_time else 0
            }
            for execution_id, execution in self.active_executions.items()
        ]


# ============================================================================
# GLOBAL EXECUTION ENGINE INSTANCE
# ============================================================================

# Global instance for workflow execution engine
workflow_execution_engine = WorkflowExecutionEngine()

# Export for external use
__all__ = [
    'ExecutionMode',
    'ExecutionPriority',
    'ExecutionContext',
    'StepExecution',
    'WorkflowExecutionEngine',
    'workflow_execution_engine'
]