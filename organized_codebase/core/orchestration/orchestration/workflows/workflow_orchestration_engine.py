from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
Workflow Orchestration Engine
============================

Advanced workflow orchestration system providing intelligent workflow execution,
distributed coordination, and adaptive scheduling across enterprise systems.

Features:
- Multi-step workflow execution with dependencies
- Distributed workflow coordination across systems
- Adaptive scheduling with priority-based execution
- Workflow versioning and rollback capabilities
- Real-time workflow monitoring and debugging
- Conditional branching and parallel execution
- Error recovery and retry strategies
- Workflow templates and composition patterns

Author: TestMaster Intelligence Team
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import heapq

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class StepType(Enum):
    """Types of workflow steps"""
    SYSTEM_OPERATION = "system_operation"
    DATA_PROCESSING = "data_processing"
    CONDITION_CHECK = "condition_check"
    PARALLEL_GROUP = "parallel_group"
    SUBPROCESS = "subprocess"
    HUMAN_TASK = "human_task"
    API_CALL = "api_call"
    SCRIPT_EXECUTION = "script_execution"

class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    CONDITIONAL = "conditional"

class Priority(Enum):
    """Workflow execution priority"""
    LOW = 1
    NORMAL = 3
    HIGH = 5
    CRITICAL = 8
    EMERGENCY = 10

@dataclass
class WorkflowVariable:
    """Workflow variable definition"""
    name: str
    value: Any
    variable_type: str = "string"
    description: str = ""
    is_secret: bool = False
    is_input: bool = False
    is_output: bool = False

@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    name: str
    step_type: StepType
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 5
    on_failure: str = "fail"  # fail, continue, retry, rollback
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    execution_log: List[str] = field(default_factory=list)

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str = ""
    version: str = "1.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: List[WorkflowVariable] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout_seconds: int = 3600
    max_parallel_steps: int = 10
    priority: Priority = Priority.NORMAL
    tags: List[str] = field(default_factory=list)
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID"""
        return next((step for step in self.steps if step.step_id == step_id), None)
    
    def get_variable(self, name: str) -> Optional[WorkflowVariable]:
        """Get variable by name"""
        return next((var for var in self.variables if var.name == name), None)

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    workflow_definition: WorkflowDefinition = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    
    def get_execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.start_time:
            end_time = self.end_time or datetime.now()
            return (end_time - self.start_time).total_seconds()
        return None
    
    def calculate_progress(self) -> float:
        """Calculate execution progress percentage"""
        if not self.workflow_definition or not self.workflow_definition.steps:
            return 0.0
        
        total_steps = len(self.workflow_definition.steps)
        completed = len(self.completed_steps)
        
        self.progress_percentage = (completed / total_steps) * 100
        return self.progress_percentage

class WorkflowScheduler:
    """Intelligent workflow scheduler with priority-based execution"""
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.max_concurrent_workflows = max_concurrent_workflows
        self.pending_queue: List[Tuple[int, float, WorkflowExecution]] = []
        self.running_workflows: Dict[str, WorkflowExecution] = {}
        self.scheduler_lock = threading.Lock()
        
    def schedule_workflow(self, execution: WorkflowExecution):
        """Schedule workflow for execution"""
        with self.scheduler_lock:
            # Use negative priority for max heap behavior (higher priority first)
            priority_score = -execution.workflow_definition.priority.value
            timestamp = time.time()
            
            heapq.heappush(self.pending_queue, (priority_score, timestamp, execution))
            logger.info(f"Scheduled workflow {execution.execution_id} with priority {execution.workflow_definition.priority.value}")
    
    def get_next_workflow(self) -> Optional[WorkflowExecution]:
        """Get next workflow to execute"""
        with self.scheduler_lock:
            if len(self.running_workflows) >= self.max_concurrent_workflows:
                return None
            
            if not self.pending_queue:
                return None
            
            _, _, execution = heapq.heappop(self.pending_queue)
            self.running_workflows[execution.execution_id] = execution
            return execution
    
    def complete_workflow(self, execution_id: str):
        """Mark workflow as completed"""
        with self.scheduler_lock:
            if execution_id in self.running_workflows:
                del self.running_workflows[execution_id]
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        with self.scheduler_lock:
            return {
                'pending_workflows': len(self.pending_queue),
                'running_workflows': len(self.running_workflows),
                'max_concurrent': self.max_concurrent_workflows,
                'running_workflow_ids': list(self.running_workflows.keys())
            }

class StepExecutor:
    """Executes individual workflow steps"""
    
    def __init__(self):
        self.step_handlers: Dict[StepType, Callable] = {}
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Register default step handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default step execution handlers"""
        self.step_handlers[StepType.SYSTEM_OPERATION] = self._execute_system_operation
        self.step_handlers[StepType.DATA_PROCESSING] = self._execute_data_processing
        self.step_handlers[StepType.CONDITION_CHECK] = self._execute_condition_check
        self.step_handlers[StepType.API_CALL] = self._execute_api_call
        self.step_handlers[StepType.SCRIPT_EXECUTION] = self._execute_script
    
    async def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute a workflow step"""
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        
        try:
            # Get appropriate handler
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"No handler for step type: {step.step_type}")
            
            # Execute step with timeout
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self.executor,
                handler,
                step,
                context
            )
            
            try:
                success, result = await asyncio.wait_for(future, timeout=step.timeout_seconds)
                
                step.status = StepStatus.COMPLETED
                step.result = result
                step.end_time = datetime.now()
                
                step.execution_log.append(f"Step completed successfully at {step.end_time}")
                return success, result
                
            except asyncio.TimeoutError:
                step.status = StepStatus.FAILED
                step.error_message = f"Step timed out after {step.timeout_seconds} seconds"
                step.end_time = datetime.now()
                
                step.execution_log.append(f"Step timed out at {step.end_time}")
                return False, {"error": step.error_message}
                
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.end_time = datetime.now()
            
            step.execution_log.append(f"Step failed with error: {e}")
            return False, {"error": str(e)}
    
    def _execute_system_operation(self, step: WorkflowStep, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute system operation step"""
        operation = step.operation
        parameters = step.parameters
        
        # Simulate system operation execution
        time.sleep(0.1)  # Simulate work
        
        if operation == "health_check":
            return True, {"status": "healthy", "timestamp": datetime.now().isoformat()}
        elif operation == "restart_service":
            return True, {"service": parameters.get("service_name", "unknown"), "status": "restarted"}
        elif operation == "backup_data":
            return True, {"backup_location": f"/backups/{uuid.uuid4().hex[:8]}.bak"}
        else:
            return True, {"operation": operation, "result": "completed"}
    
    def _execute_data_processing(self, step: WorkflowStep, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute data processing step"""
        operation = step.operation
        parameters = step.parameters
        
        # Simulate data processing
        time.sleep(0.2)  # Simulate processing time
        
        processed_records = parameters.get("record_count", 100)
        return True, {
            "processed_records": processed_records,
            "processing_time_ms": 200,
            "output_location": f"/data/processed/{uuid.uuid4().hex[:8]}.json"
        }
    
    def _execute_condition_check(self, step: WorkflowStep, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute condition check step"""
        condition = step.condition or step.parameters.get("condition", "true")
        
        # Simple condition evaluation (can be enhanced with proper expression parser)
        try:
            # For safety, only allow basic comparisons
            if ">" in condition or "<" in condition or "==" in condition:
                # Evaluate simple conditions
                result = SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval(condition, {"__builtins__": {}}, context)
            else:
                result = condition.lower() in ["true", "1", "yes"]
            
            return True, {"condition_result": result, "condition": condition}
            
        except Exception as e:
            return False, {"error": f"Condition evaluation failed: {e}"}
    
    def _execute_api_call(self, step: WorkflowStep, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute API call step"""
        url = step.parameters.get("url", "")
        method = step.parameters.get("method", "GET")
        
        # Simulate API call
        time.sleep(0.3)  # Simulate network delay
        
        # Simulate success/failure
        import random
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            return True, {
                "status_code": 200,
                "response": {"message": "API call successful"},
                "url": url,
                "method": method
            }
        else:
            return False, {"error": "API call failed", "status_code": 500}
    
    def _execute_script(self, step: WorkflowStep, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute script step"""
        script = step.parameters.get("script", "")
        
        # For safety, only allow very basic script execution
        if script == "echo test":
            return True, {"output": "test", "exit_code": 0}
        elif script.startswith("echo "):
            message = script[5:]  # Remove "echo "
            return True, {"output": message, "exit_code": 0}
        else:
            return False, {"error": "Script execution not allowed for security reasons"}

class WorkflowOrchestrationEngine:
    """
    Advanced workflow orchestration system providing intelligent workflow execution
    and distributed coordination across enterprise systems.
    """
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.scheduler = WorkflowScheduler(max_concurrent_workflows)
        self.step_executor = StepExecutor()
        
        # Workflow storage
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: deque = deque(maxlen=1000)
        
        # Orchestration state
        self.orchestration_active = False
        self.orchestration_tasks: Set[asyncio.Task] = set()
        
        # Monitoring and metrics
        self.orchestration_stats = {
            'workflows_executed': 0,
            'workflows_successful': 0,
            'workflows_failed': 0,
            'average_execution_time': 0.0,
            'start_time': datetime.now()
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("Workflow Orchestration Engine initialized")
    
    def register_workflow_definition(self, workflow_def: WorkflowDefinition):
        """Register a workflow definition"""
        self.workflow_definitions[workflow_def.workflow_id] = workflow_def
        logger.info(f"Registered workflow definition: {workflow_def.workflow_id}")
    
    def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID"""
        return self.workflow_definitions.get(workflow_id)
    
    async def start_orchestration(self):
        """Start the workflow orchestration engine"""
        if self.orchestration_active:
            return
        
        logger.info("Starting Workflow Orchestration Engine")
        self.orchestration_active = True
        
        # Start workflow execution loop
        execution_task = asyncio.create_task(self._workflow_execution_loop())
        self.orchestration_tasks.add(execution_task)
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.orchestration_tasks.add(monitoring_task)
        
        logger.info("Workflow Orchestration Engine started")
    
    async def stop_orchestration(self):
        """Stop the workflow orchestration engine"""
        if not self.orchestration_active:
            return
        
        logger.info("Stopping Workflow Orchestration Engine")
        self.orchestration_active = False
        
        # Cancel all orchestration tasks
        for task in self.orchestration_tasks:
            task.cancel()
        
        await asyncio.gather(*self.orchestration_tasks, return_exceptions=True)
        self.orchestration_tasks.clear()
        
        logger.info("Workflow Orchestration Engine stopped")
    
    async def execute_workflow(self, workflow_id: str, variables: Dict[str, Any] = None) -> str:
        """Execute a workflow"""
        workflow_def = self.get_workflow_definition(workflow_id)
        if not workflow_def:
            raise ValueError(f"Workflow definition not found: {workflow_id}")
        
        # Create execution instance
        execution = WorkflowExecution(
            workflow_definition=workflow_def,
            execution_context=variables or {}
        )
        
        self.workflow_executions[execution.execution_id] = execution
        
        # Schedule for execution
        self.scheduler.schedule_workflow(execution)
        
        logger.info(f"Scheduled workflow execution: {execution.execution_id}")
        return execution.execution_id
    
    async def _workflow_execution_loop(self):
        """Main workflow execution loop"""
        while self.orchestration_active:
            try:
                # Get next workflow to execute
                execution = self.scheduler.get_next_workflow()
                
                if execution:
                    # Execute workflow in background
                    execution_task = asyncio.create_task(self._execute_workflow_instance(execution))
                    self.orchestration_tasks.add(execution_task)
                else:
                    # No workflows to execute, wait briefly
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Workflow execution loop error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_workflow_instance(self, execution: WorkflowExecution):
        """Execute a specific workflow instance"""
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()
            
            await self._emit_event("workflow_started", execution)
            
            # Execute steps based on execution mode
            if execution.workflow_definition.execution_mode == ExecutionMode.SEQUENTIAL:
                success = await self._execute_sequential_workflow(execution)
            elif execution.workflow_definition.execution_mode == ExecutionMode.PARALLEL:
                success = await self._execute_parallel_workflow(execution)
            elif execution.workflow_definition.execution_mode == ExecutionMode.HYBRID:
                success = await self._execute_hybrid_workflow(execution)
            else:
                success = await self._execute_conditional_workflow(execution)
            
            # Update final status
            if success:
                execution.status = WorkflowStatus.COMPLETED
                self.orchestration_stats['workflows_successful'] += 1
            else:
                execution.status = WorkflowStatus.FAILED
                self.orchestration_stats['workflows_failed'] += 1
            
            execution.end_time = datetime.now()
            execution.calculate_progress()
            
            # Update statistics
            self.orchestration_stats['workflows_executed'] += 1
            
            # Calculate average execution time
            execution_time = execution.get_execution_duration()
            if execution_time:
                current_avg = self.orchestration_stats['average_execution_time']
                total_workflows = self.orchestration_stats['workflows_executed']
                self.orchestration_stats['average_execution_time'] = (
                    (current_avg * (total_workflows - 1) + execution_time) / total_workflows
                )
            
            await self._emit_event("workflow_completed", execution)
            
            # Move to history
            self.execution_history.append(execution)
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            
            self.orchestration_stats['workflows_failed'] += 1
            logger.error(f"Workflow execution failed {execution.execution_id}: {e}")
            
            await self._emit_event("workflow_failed", execution)
        
        finally:
            # Mark workflow as completed in scheduler
            self.scheduler.complete_workflow(execution.execution_id)
    
    async def _execute_sequential_workflow(self, execution: WorkflowExecution) -> bool:
        """Execute workflow steps sequentially"""
        steps = execution.workflow_definition.steps
        
        for step in steps:
            if not await self._execute_single_step(step, execution):
                return False
        
        return True
    
    async def _execute_parallel_workflow(self, execution: WorkflowExecution) -> bool:
        """Execute workflow steps in parallel"""
        steps = execution.workflow_definition.steps
        
        # Execute all steps concurrently
        tasks = []
        for step in steps:
            task = asyncio.create_task(self._execute_single_step(step, execution))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if all steps succeeded
        return all(result is True for result in results if not isinstance(result, Exception))
    
    async def _execute_hybrid_workflow(self, execution: WorkflowExecution) -> bool:
        """Execute workflow with hybrid sequential/parallel approach"""
        steps = execution.workflow_definition.steps
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(steps)
        
        # Execute steps respecting dependencies
        completed_steps = set()
        
        while len(completed_steps) < len(steps):
            # Find steps that can be executed (dependencies satisfied)
            ready_steps = []
            for step in steps:
                if (step.step_id not in completed_steps and
                    all(dep in completed_steps for dep in step.depends_on)):
                    ready_steps.append(step)
            
            if not ready_steps:
                # No more steps can be executed
                break
            
            # Execute ready steps in parallel
            tasks = []
            for step in ready_steps:
                task = asyncio.create_task(self._execute_single_step(step, execution))
                tasks.append((step, task))
            
            # Wait for completion
            for step, task in tasks:
                try:
                    success = await task
                    if success:
                        completed_steps.add(step.step_id)
                        execution.completed_steps.append(step.step_id)
                    else:
                        execution.failed_steps.append(step.step_id)
                        if step.on_failure == "fail":
                            return False
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {e}")
                    execution.failed_steps.append(step.step_id)
                    if step.on_failure == "fail":
                        return False
        
        return len(completed_steps) == len(steps)
    
    async def _execute_conditional_workflow(self, execution: WorkflowExecution) -> bool:
        """Execute workflow with conditional branching"""
        steps = execution.workflow_definition.steps
        
        for step in steps:
            # Check step condition
            if step.condition:
                condition_met = await self._evaluate_condition(step.condition, execution.execution_context)
                if not condition_met:
                    step.status = StepStatus.SKIPPED
                    continue
            
            if not await self._execute_single_step(step, execution):
                return False
        
        return True
    
    async def _execute_single_step(self, step: WorkflowStep, execution: WorkflowExecution) -> bool:
        """Execute a single workflow step with retry logic"""
        execution.current_step = step.step_id
        
        for attempt in range(step.max_retries + 1):
            if attempt > 0:
                step.status = StepStatus.RETRYING
                await asyncio.sleep(step.retry_delay_seconds)
            
            success, result = await self.step_executor.execute_step(step, execution.execution_context)
            
            if success:
                # Update execution context with step result
                execution.execution_context[f"step_{step.step_id}_result"] = result
                execution.completed_steps.append(step.step_id)
                return True
            else:
                step.retry_count = attempt + 1
                if attempt < step.max_retries:
                    logger.warning(f"Step {step.step_id} failed, retrying ({attempt + 1}/{step.max_retries})")
                else:
                    logger.error(f"Step {step.step_id} failed after {step.max_retries} retries")
                    execution.failed_steps.append(step.step_id)
                    return False
        
        return False
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for steps"""
        graph = {}
        for step in steps:
            graph[step.step_id] = step.depends_on
        return graph
    
    async def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate workflow condition"""
        try:
            # Simple condition evaluation (can be enhanced)
            return SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False
    
    async def _emit_event(self, event_type: str, execution: WorkflowExecution):
        """Emit workflow event"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(execution)
                else:
                    handler(execution)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")
    
    def subscribe_to_events(self, event_type: str, handler: Callable):
        """Subscribe to workflow events"""
        self.event_handlers[event_type].append(handler)
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.orchestration_active:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Check for timed out workflows
                current_time = datetime.now()
                for execution in list(self.workflow_executions.values()):
                    if (execution.status == WorkflowStatus.RUNNING and
                        execution.start_time and
                        (current_time - execution.start_time).total_seconds() > execution.workflow_definition.timeout_seconds):
                        
                        execution.status = WorkflowStatus.TIMEOUT
                        execution.end_time = current_time
                        logger.warning(f"Workflow {execution.execution_id} timed out")
                        
                        await self._emit_event("workflow_timeout", execution)
                
                # Clean up completed executions
                self._cleanup_completed_executions()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _cleanup_completed_executions(self):
        """Clean up old completed executions"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        completed_executions = []
        for execution_id, execution in self.workflow_executions.items():
            if (execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                execution.end_time and execution.end_time < cutoff_time):
                completed_executions.append(execution_id)
        
        for execution_id in completed_executions:
            del self.workflow_executions[execution_id]
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        execution = self.workflow_executions.get(execution_id)
        if not execution:
            return None
        
        step_statuses = []
        for step in execution.workflow_definition.steps:
            step_statuses.append({
                'step_id': step.step_id,
                'name': step.name,
                'status': step.status.value,
                'start_time': step.start_time.isoformat() if step.start_time else None,
                'end_time': step.end_time.isoformat() if step.end_time else None,
                'error_message': step.error_message,
                'retry_count': step.retry_count
            })
        
        return {
            'execution_id': execution.execution_id,
            'workflow_id': execution.workflow_definition.workflow_id,
            'status': execution.status.value,
            'progress_percentage': execution.calculate_progress(),
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'current_step': execution.current_step,
            'completed_steps': execution.completed_steps,
            'failed_steps': execution.failed_steps,
            'step_details': step_statuses,
            'execution_duration': execution.get_execution_duration(),
            'error_message': execution.error_message
        }
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration engine status"""
        uptime = (datetime.now() - self.orchestration_stats['start_time']).total_seconds()
        
        # Execution status summary
        status_counts = defaultdict(int)
        for execution in self.workflow_executions.values():
            status_counts[execution.status.value] += 1
        
        return {
            'status': 'active' if self.orchestration_active else 'inactive',
            'uptime_seconds': uptime,
            'statistics': self.orchestration_stats.copy(),
            'scheduler_status': self.scheduler.get_scheduler_status(),
            'active_executions': len(self.workflow_executions),
            'execution_status_distribution': dict(status_counts),
            'registered_workflows': len(self.workflow_definitions),
            'execution_history_size': len(self.execution_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown workflow orchestration engine"""
        if self.orchestration_active:
            # Create and run shutdown task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_orchestration())
            loop.close()
        
        logger.info("Workflow Orchestration Engine shutdown")

# Global workflow orchestration engine instance
workflow_orchestration_engine = WorkflowOrchestrationEngine()