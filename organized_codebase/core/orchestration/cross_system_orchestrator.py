"""
Cross-System Orchestrator
=========================

Advanced orchestration engine managing complex workflows across multiple intelligence
systems with sophisticated execution patterns, dependency resolution, and error recovery.

Integrates: Workflow Engine, Dependency Graph, Circuit Breakers, Event Choreography,
State Management, Resource Allocation, and Real-time Monitoring.

Author: TestMaster Intelligence Framework
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import networkx as nx


# ============================================================================
# CORE ORCHESTRATION TYPES
# ============================================================================

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskType(Enum):
    """Task execution types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    FUNCTION = "function"
    SERVICE_CALL = "service_call"
    EVENT_TRIGGER = "event_trigger"
    WORKFLOW = "workflow"


class ExecutionStrategy(Enum):
    """Workflow execution strategies"""
    EAGER = "eager"           # Execute as soon as dependencies ready
    LAZY = "lazy"             # Execute only when output needed
    BATCH = "batch"           # Group similar tasks
    ADAPTIVE = "adaptive"     # Adjust based on performance
    PRIORITY = "priority"     # Execute by priority order
    RESOURCE_AWARE = "resource_aware"  # Consider resource availability


@dataclass
class WorkflowTask:
    """Individual task in workflow execution"""
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    name: str = ""
    task_type: TaskType = TaskType.FUNCTION
    function_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay: float = 1.0
    priority: int = 5
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_attempts: int = 0


@dataclass
class OrchestrationWorkflow:
    """Complete workflow definition and state"""
    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    tasks: List[WorkflowTask] = field(default_factory=list)
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.EAGER
    max_parallel_tasks: int = 10
    workflow_timeout: int = 3600
    on_failure: str = "stop"  # stop, continue, retry
    on_success: str = "complete"
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Runtime state
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    execution_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of workflow/task execution"""
    execution_id: str
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    tasks_executed: int = 0
    tasks_failed: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DEPENDENCY GRAPH MANAGER
# ============================================================================

class DependencyGraphManager:
    """Advanced dependency graph with cycle detection and optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger("dependency_graph_manager")
        
        # Graph storage
        self.graph = nx.DiGraph()
        self.task_registry: Dict[str, WorkflowTask] = {}
        
        # Analysis cache
        self.topology_cache: Dict[str, List[str]] = {}
        self.critical_path_cache: Dict[str, List[str]] = {}
        
        self.logger.info("Dependency graph manager initialized")
    
    def add_workflow(self, workflow: OrchestrationWorkflow) -> bool:
        """Add workflow tasks to dependency graph"""
        try:
            workflow_prefix = f"{workflow.workflow_id}."
            
            # Add tasks to registry
            for task in workflow.tasks:
                full_task_id = f"{workflow_prefix}{task.task_id}"
                self.task_registry[full_task_id] = task
                self.graph.add_node(full_task_id, task=task)
            
            # Add dependencies
            for task in workflow.tasks:
                full_task_id = f"{workflow_prefix}{task.task_id}"
                
                for dep_id in task.dependencies:
                    full_dep_id = f"{workflow_prefix}{dep_id}"
                    if full_dep_id in self.task_registry:
                        self.graph.add_edge(full_dep_id, full_task_id)
                    else:
                        self.logger.warning(f"Dependency not found: {dep_id} for task {task.task_id}")
            
            # Validate graph
            if not self._validate_graph(workflow_prefix):
                return False
            
            # Clear caches
            self._clear_caches()
            
            self.logger.info(f"Added workflow {workflow.workflow_id} with {len(workflow.tasks)} tasks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add workflow: {e}")
            return False
    
    def _validate_graph(self, workflow_prefix: str) -> bool:
        """Validate graph for cycles and connectivity"""
        workflow_nodes = [n for n in self.graph.nodes() if n.startswith(workflow_prefix)]
        workflow_subgraph = self.graph.subgraph(workflow_nodes)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(workflow_subgraph):
            cycles = list(nx.simple_cycles(workflow_subgraph))
            self.logger.error(f"Workflow contains cycles: {cycles}")
            return False
        
        # Check connectivity
        if len(workflow_nodes) > 1:
            undirected = workflow_subgraph.to_undirected()
            if not nx.is_connected(undirected):
                components = list(nx.connected_components(undirected))
                self.logger.warning(f"Workflow has disconnected components: {len(components)}")
        
        return True
    
    def get_execution_order(self, workflow_id: str) -> List[List[str]]:
        """Get topological execution order with parallelization"""
        try:
            workflow_prefix = f"{workflow_id}."
            workflow_nodes = [n for n in self.graph.nodes() if n.startswith(workflow_prefix)]
            workflow_subgraph = self.graph.subgraph(workflow_nodes)
            
            # Use cached result if available
            cache_key = f"{workflow_id}_topo"
            if cache_key in self.topology_cache:
                return self._group_parallel_tasks(self.topology_cache[cache_key])
            
            # Generate topological order
            topo_order = list(nx.topological_sort(workflow_subgraph))
            self.topology_cache[cache_key] = topo_order
            
            # Group tasks that can run in parallel
            return self._group_parallel_tasks(topo_order)
            
        except Exception as e:
            self.logger.error(f"Failed to get execution order: {e}")
            return []
    
    def _group_parallel_tasks(self, topo_order: List[str]) -> List[List[str]]:
        """Group tasks that can execute in parallel"""
        if not topo_order:
            return []
        
        execution_levels = []
        remaining_tasks = set(topo_order)
        completed_tasks = set()
        
        while remaining_tasks:
            # Find tasks with all dependencies completed
            ready_tasks = []
            for task_id in remaining_tasks:
                dependencies = set(self.graph.predecessors(task_id))
                if dependencies.issubset(completed_tasks):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # This shouldn't happen with valid DAG
                self.logger.error("No ready tasks found - possible dependency issue")
                break
            
            # Sort by priority
            ready_tasks.sort(key=lambda tid: self.task_registry[tid].priority, reverse=True)
            
            execution_levels.append(ready_tasks)
            completed_tasks.update(ready_tasks)
            remaining_tasks -= set(ready_tasks)
        
        return execution_levels
    
    def get_critical_path(self, workflow_id: str) -> Tuple[List[str], float]:
        """Get critical path through workflow"""
        try:
            cache_key = f"{workflow_id}_critical"
            if cache_key in self.critical_path_cache:
                path = self.critical_path_cache[cache_key]
                duration = sum(self.task_registry[tid].timeout_seconds for tid in path)
                return path, duration
            
            workflow_prefix = f"{workflow_id}."
            workflow_nodes = [n for n in self.graph.nodes() if n.startswith(workflow_prefix)]
            workflow_subgraph = self.graph.subgraph(workflow_nodes)
            
            # Calculate longest path (critical path)
            longest_path = []
            max_duration = 0.0
            
            # Find all possible paths and select longest
            root_nodes = [n for n in workflow_nodes if workflow_subgraph.in_degree(n) == 0]
            leaf_nodes = [n for n in workflow_nodes if workflow_subgraph.out_degree(n) == 0]
            
            for root in root_nodes:
                for leaf in leaf_nodes:
                    try:
                        paths = list(nx.all_simple_paths(workflow_subgraph, root, leaf))
                        for path in paths:
                            duration = sum(self.task_registry[tid].timeout_seconds for tid in path)
                            if duration > max_duration:
                                max_duration = duration
                                longest_path = path
                    except nx.NetworkXNoPath:
                        continue
            
            self.critical_path_cache[cache_key] = longest_path
            return longest_path, max_duration
            
        except Exception as e:
            self.logger.error(f"Failed to get critical path: {e}")
            return [], 0.0
    
    def get_task_dependencies(self, task_id: str) -> Dict[str, List[str]]:
        """Get task dependencies and dependents"""
        try:
            predecessors = list(self.graph.predecessors(task_id))
            successors = list(self.graph.successors(task_id))
            
            return {
                "dependencies": predecessors,
                "dependents": successors,
                "depth": nx.shortest_path_length(self.graph, source=task_id) if predecessors else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get task dependencies: {e}")
            return {"dependencies": [], "dependents": [], "depth": 0}
    
    def _clear_caches(self):
        """Clear analysis caches"""
        self.topology_cache.clear()
        self.critical_path_cache.clear()


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class OrchestrationExecutionEngine:
    """Advanced execution engine with sophisticated scheduling and monitoring"""
    
    def __init__(self, max_workers: int = 10):
        self.logger = logging.getLogger("orchestration_execution_engine")
        
        # Execution infrastructure
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.dependency_manager = DependencyGraphManager()
        
        # Active executions
        self.active_workflows: Dict[str, OrchestrationWorkflow] = {}
        self.active_tasks: Dict[str, Future] = {}
        self.execution_lock = threading.Lock()
        
        # Task handlers
        self.task_handlers: Dict[TaskType, Callable] = {
            TaskType.FUNCTION: self._execute_function_task,
            TaskType.SERVICE_CALL: self._execute_service_call,
            TaskType.CONDITIONAL: self._execute_conditional_task,
            TaskType.LOOP: self._execute_loop_task,
            TaskType.EVENT_TRIGGER: self._execute_event_trigger,
            TaskType.WORKFLOW: self._execute_sub_workflow
        }
        
        # Performance monitoring
        self.execution_metrics = {
            "workflows_executed": 0,
            "tasks_executed": 0,
            "average_workflow_time": 0.0,
            "average_task_time": 0.0,
            "success_rate": 100.0,
            "resource_utilization": 0.0
        }
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Orchestration execution engine initialized")
    
    async def execute_workflow(self, workflow: OrchestrationWorkflow) -> ExecutionResult:
        """Execute complete workflow with advanced orchestration"""
        start_time = time.time()
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        
        try:
            # Initialize workflow
            workflow.status = WorkflowStatus.RUNNING
            workflow.start_time = datetime.now()
            workflow.execution_context = {"execution_id": execution_id}
            
            with self.execution_lock:
                self.active_workflows[workflow.workflow_id] = workflow
            
            # Add to dependency graph
            if not self.dependency_manager.add_workflow(workflow):
                raise Exception("Failed to add workflow to dependency graph")
            
            # Get execution plan
            execution_levels = self.dependency_manager.get_execution_order(workflow.workflow_id)
            critical_path, estimated_duration = self.dependency_manager.get_critical_path(workflow.workflow_id)
            
            self.logger.info(f"Executing workflow {workflow.workflow_id} with {len(execution_levels)} levels")
            self.logger.info(f"Critical path: {len(critical_path)} tasks, estimated duration: {estimated_duration}s")
            
            # Execute workflow levels
            tasks_executed = 0
            tasks_failed = 0
            
            for level_idx, task_level in enumerate(execution_levels):
                self.logger.info(f"Executing level {level_idx + 1}/{len(execution_levels)}: {len(task_level)} tasks")
                
                # Execute tasks in parallel within level
                level_results = await self._execute_task_level(workflow, task_level)
                
                # Process results
                for task_id, success, result, error in level_results:
                    if success:
                        tasks_executed += 1
                        workflow.completed_tasks.add(task_id)
                    else:
                        tasks_failed += 1
                        workflow.failed_tasks.add(task_id)
                        
                        # Check failure handling
                        if workflow.on_failure == "stop":
                            raise Exception(f"Task {task_id} failed: {error}")
                
                # Check timeout
                if time.time() - start_time > workflow.workflow_timeout:
                    raise Exception("Workflow timeout exceeded")
            
            # Complete workflow
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            
            execution_time = (time.time() - start_time) * 1000
            self._update_execution_metrics(execution_time, True)
            
            return ExecutionResult(
                execution_id=execution_id,
                success=True,
                result={"workflow_id": workflow.workflow_id, "status": "completed"},
                execution_time_ms=execution_time,
                tasks_executed=tasks_executed,
                tasks_failed=tasks_failed
            )
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            
            execution_time = (time.time() - start_time) * 1000
            self._update_execution_metrics(execution_time, False)
            
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )
        
        finally:
            # Cleanup
            with self.execution_lock:
                self.active_workflows.pop(workflow.workflow_id, None)
    
    async def _execute_task_level(self, workflow: OrchestrationWorkflow, 
                                 task_ids: List[str]) -> List[Tuple[str, bool, Any, Optional[str]]]:
        """Execute tasks in parallel within a level"""
        # Respect max parallel limit
        max_parallel = min(len(task_ids), workflow.max_parallel_tasks)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_parallel)
        
        # Execute tasks concurrently
        tasks = []
        for task_id in task_ids:
            task = asyncio.create_task(self._execute_single_task(workflow, task_id, semaphore))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            task_id = task_ids[i]
            
            if isinstance(result, Exception):
                processed_results.append((task_id, False, None, str(result)))
            else:
                success, task_result, error = result
                processed_results.append((task_id, success, task_result, error))
        
        return processed_results
    
    async def _execute_single_task(self, workflow: OrchestrationWorkflow, 
                                  task_id: str, semaphore: asyncio.Semaphore) -> Tuple[bool, Any, Optional[str]]:
        """Execute single task with error handling and retries"""
        full_task_id = f"{workflow.workflow_id}.{task_id}"
        task = self.dependency_manager.task_registry.get(full_task_id)
        
        if not task:
            return False, None, f"Task not found: {task_id}"
        
        async with semaphore:
            # Update task state
            task.status = WorkflowStatus.RUNNING
            task.start_time = datetime.now()
            workflow.current_tasks.add(task_id)
            
            try:
                # Check circuit breaker
                if not self._check_circuit_breaker(task.function_name):
                    return False, None, f"Circuit breaker open for {task.function_name}"
                
                # Execute with retries
                for attempt in range(task.retry_count + 1):
                    try:
                        start_time = time.time()
                        
                        # Execute task based on type
                        if task.task_type in self.task_handlers:
                            result = await self.task_handlers[task.task_type](task, workflow)
                        else:
                            result = await self._execute_function_task(task, workflow)
                        
                        # Update task success state
                        task.status = WorkflowStatus.COMPLETED
                        task.end_time = datetime.now()
                        task.execution_time_ms = (time.time() - start_time) * 1000
                        task.result = result
                        
                        # Update circuit breaker
                        self._record_circuit_breaker_success(task.function_name)
                        
                        return True, result, None
                        
                    except Exception as e:
                        task.retry_attempts += 1
                        error_msg = str(e)
                        
                        # Record circuit breaker failure
                        self._record_circuit_breaker_failure(task.function_name)
                        
                        if attempt < task.retry_count:
                            self.logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {error_msg}, retrying...")
                            await asyncio.sleep(task.retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            # Final failure
                            task.status = WorkflowStatus.FAILED
                            task.end_time = datetime.now()
                            task.error_message = error_msg
                            return False, None, error_msg
                
            finally:
                workflow.current_tasks.discard(task_id)
        
        return False, None, "Task execution failed"
    
    async def _execute_function_task(self, task: WorkflowTask, workflow: OrchestrationWorkflow) -> Any:
        """Execute function task"""
        # Mock function execution
        self.logger.info(f"Executing function task: {task.function_name}")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Return mock result
        return {
            "task_id": task.task_id,
            "function": task.function_name,
            "parameters": task.parameters,
            "timestamp": datetime.now().isoformat(),
            "result": "success"
        }
    
    async def _execute_service_call(self, task: WorkflowTask, workflow: OrchestrationWorkflow) -> Any:
        """Execute service call task"""
        self.logger.info(f"Executing service call: {task.function_name}")
        
        # Mock service call
        await asyncio.sleep(0.2)
        
        return {
            "service": task.function_name,
            "response": "service_response",
            "status": "completed"
        }
    
    async def _execute_conditional_task(self, task: WorkflowTask, workflow: OrchestrationWorkflow) -> Any:
        """Execute conditional task"""
        conditions = task.conditions
        
        # Evaluate conditions (mock evaluation)
        condition_result = conditions.get("condition", True)
        
        if condition_result:
            return await self._execute_function_task(task, workflow)
        else:
            return {"skipped": True, "reason": "condition_not_met"}
    
    async def _execute_loop_task(self, task: WorkflowTask, workflow: OrchestrationWorkflow) -> Any:
        """Execute loop task"""
        loop_config = task.parameters.get("loop", {})
        iterations = loop_config.get("iterations", 1)
        
        results = []
        for i in range(iterations):
            # Mock iteration
            await asyncio.sleep(0.05)
            results.append(f"iteration_{i}_result")
        
        return {"loop_results": results, "iterations": iterations}
    
    async def _execute_event_trigger(self, task: WorkflowTask, workflow: OrchestrationWorkflow) -> Any:
        """Execute event trigger task"""
        event_type = task.parameters.get("event_type", "generic")
        
        # Mock event triggering
        return {
            "event_triggered": event_type,
            "timestamp": datetime.now().isoformat(),
            "workflow_id": workflow.workflow_id
        }
    
    async def _execute_sub_workflow(self, task: WorkflowTask, workflow: OrchestrationWorkflow) -> Any:
        """Execute sub-workflow task"""
        sub_workflow_id = task.parameters.get("workflow_id")
        
        # Mock sub-workflow execution
        return {
            "sub_workflow_id": sub_workflow_id,
            "status": "completed",
            "parent_workflow": workflow.workflow_id
        }
    
    def _check_circuit_breaker(self, function_name: str) -> bool:
        """Check circuit breaker state"""
        if function_name not in self.circuit_breakers:
            self.circuit_breakers[function_name] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": None,
                "failure_threshold": 5,
                "recovery_timeout": 60
            }
        
        breaker = self.circuit_breakers[function_name]
        
        if breaker["state"] == "open":
            # Check if recovery timeout has passed
            if (breaker["last_failure_time"] and 
                time.time() - breaker["last_failure_time"] > breaker["recovery_timeout"]):
                breaker["state"] = "half-open"
                breaker["failure_count"] = 0
                return True
            return False
        
        return True
    
    def _record_circuit_breaker_success(self, function_name: str):
        """Record successful execution for circuit breaker"""
        if function_name in self.circuit_breakers:
            breaker = self.circuit_breakers[function_name]
            breaker["failure_count"] = 0
            breaker["state"] = "closed"
    
    def _record_circuit_breaker_failure(self, function_name: str):
        """Record failed execution for circuit breaker"""
        if function_name not in self.circuit_breakers:
            self._check_circuit_breaker(function_name)  # Initialize
        
        breaker = self.circuit_breakers[function_name]
        breaker["failure_count"] += 1
        breaker["last_failure_time"] = time.time()
        
        if breaker["failure_count"] >= breaker["failure_threshold"]:
            breaker["state"] = "open"
            self.logger.warning(f"Circuit breaker opened for {function_name}")
    
    def _update_execution_metrics(self, execution_time_ms: float, success: bool):
        """Update execution performance metrics"""
        self.execution_metrics["workflows_executed"] += 1
        
        # Update average execution time
        total_workflows = self.execution_metrics["workflows_executed"]
        current_avg = self.execution_metrics["average_workflow_time"]
        self.execution_metrics["average_workflow_time"] = (
            (current_avg * (total_workflows - 1) + execution_time_ms) / total_workflows
        )
        
        # Update success rate
        if success:
            success_count = total_workflows * (self.execution_metrics["success_rate"] / 100.0)
            self.execution_metrics["success_rate"] = ((success_count + 1) / total_workflows) * 100.0
        else:
            success_count = total_workflows * (self.execution_metrics["success_rate"] / 100.0)
            self.execution_metrics["success_rate"] = (success_count / total_workflows) * 100.0
    
    def get_execution_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current execution status"""
        with self.execution_lock:
            workflow = self.active_workflows.get(workflow_id)
            
            if not workflow:
                return {"error": "Workflow not found or not active"}
            
            return {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
                "current_tasks": list(workflow.current_tasks),
                "completed_tasks": len(workflow.completed_tasks),
                "failed_tasks": len(workflow.failed_tasks),
                "total_tasks": len(workflow.tasks),
                "progress_percentage": (len(workflow.completed_tasks) / len(workflow.tasks)) * 100 if workflow.tasks else 0
            }
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics"""
        return {
            "execution_metrics": self.execution_metrics.copy(),
            "active_workflows": len(self.active_workflows),
            "active_tasks": len(self.active_tasks),
            "circuit_breakers": {
                name: breaker["state"] 
                for name, breaker in self.circuit_breakers.items()
            },
            "dependency_graph_size": len(self.dependency_manager.task_registry)
        }


# ============================================================================
# CROSS-SYSTEM ORCHESTRATOR
# ============================================================================

class CrossSystemOrchestrator:
    """
    Master orchestrator coordinating complex workflows across all intelligence systems
    with advanced execution strategies and comprehensive monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("cross_system_orchestrator")
        
        # Core execution engine
        self.execution_engine = OrchestrationExecutionEngine(max_workers=20)
        
        # Workflow registry
        self.workflow_templates: Dict[str, OrchestrationWorkflow] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # System integration
        self.system_connectors: Dict[str, Any] = {}
        self.global_context: Dict[str, Any] = {}
        
        # Performance optimization
        self.optimization_strategies = {
            "adaptive_scheduling": True,
            "resource_balancing": True,
            "intelligent_caching": True,
            "predictive_scaling": True
        }
        
        # Initialize default workflows
        self._initialize_default_workflows()
        
        self.logger.info("Cross-system orchestrator initialized")
    
    def _initialize_default_workflows(self):
        """Initialize default cross-system workflows"""
        # Intelligence Analysis Workflow
        intelligence_workflow = OrchestrationWorkflow(
            name="Intelligence Analysis Pipeline",
            description="Comprehensive intelligence analysis across all systems",
            execution_strategy=ExecutionStrategy.ADAPTIVE,
            max_parallel_tasks=8,
            workflow_timeout=1800
        )
        
        # Add intelligence analysis tasks
        tasks = [
            WorkflowTask(
                task_id="collect_data",
                name="Data Collection",
                task_type=TaskType.SERVICE_CALL,
                function_name="intelligence.collect_data",
                parameters={"sources": ["analytics", "streaming", "coordination"]},
                priority=10
            ),
            WorkflowTask(
                task_id="analyze_patterns",
                name="Pattern Analysis",
                task_type=TaskType.FUNCTION,
                function_name="analytics.analyze_patterns",
                dependencies=["collect_data"],
                priority=8
            ),
            WorkflowTask(
                task_id="detect_anomalies",
                name="Anomaly Detection",
                task_type=TaskType.FUNCTION,
                function_name="analytics.detect_anomalies",
                dependencies=["collect_data"],
                priority=8
            ),
            WorkflowTask(
                task_id="generate_insights",
                name="Insight Generation",
                task_type=TaskType.FUNCTION,
                function_name="intelligence.generate_insights",
                dependencies=["analyze_patterns", "detect_anomalies"],
                priority=6
            ),
            WorkflowTask(
                task_id="create_report",
                name="Report Creation",
                task_type=TaskType.SERVICE_CALL,
                function_name="reporting.create_intelligence_report",
                dependencies=["generate_insights"],
                priority=4
            )
        ]
        
        intelligence_workflow.tasks = tasks
        self.register_workflow_template("intelligence_analysis", intelligence_workflow)
        
        # System Health Monitoring Workflow
        health_workflow = OrchestrationWorkflow(
            name="System Health Monitoring",
            description="Monitor and maintain system health across all components",
            execution_strategy=ExecutionStrategy.PRIORITY,
            max_parallel_tasks=12,
            workflow_timeout=600
        )
        
        health_tasks = [
            WorkflowTask(
                task_id="check_services",
                name="Service Health Check",
                task_type=TaskType.SERVICE_CALL,
                function_name="coordination.health_check_services",
                priority=10
            ),
            WorkflowTask(
                task_id="monitor_resources",
                name="Resource Monitoring",
                task_type=TaskType.FUNCTION,
                function_name="coordination.monitor_resources",
                priority=9
            ),
            WorkflowTask(
                task_id="analyze_performance",
                name="Performance Analysis",
                task_type=TaskType.FUNCTION,
                function_name="analytics.analyze_performance",
                dependencies=["check_services", "monitor_resources"],
                priority=7
            ),
            WorkflowTask(
                task_id="optimize_allocation",
                name="Resource Optimization",
                task_type=TaskType.CONDITIONAL,
                function_name="coordination.optimize_resources",
                dependencies=["analyze_performance"],
                conditions={"performance_degraded": True},
                priority=8
            )
        ]
        
        health_workflow.tasks = health_tasks
        self.register_workflow_template("system_health", health_workflow)
    
    def register_workflow_template(self, template_name: str, workflow: OrchestrationWorkflow) -> bool:
        """Register workflow template"""
        try:
            self.workflow_templates[template_name] = workflow
            self.logger.info(f"Registered workflow template: {template_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register workflow template: {e}")
            return False
    
    async def execute_workflow_template(self, template_name: str, 
                                       parameters: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute workflow from template"""
        try:
            if template_name not in self.workflow_templates:
                raise Exception(f"Workflow template not found: {template_name}")
            
            # Clone template
            template = self.workflow_templates[template_name]
            workflow = OrchestrationWorkflow(
                name=template.name,
                description=template.description,
                tasks=template.tasks.copy(),
                global_parameters={**template.global_parameters, **(parameters or {})},
                execution_strategy=template.execution_strategy,
                max_parallel_tasks=template.max_parallel_tasks,
                workflow_timeout=template.workflow_timeout
            )
            
            # Execute workflow
            result = await self.execution_engine.execute_workflow(workflow)
            
            # Store execution history
            self.execution_history.append(result)
            
            # Keep only last 100 executions
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow template execution failed: {e}")
            return ExecutionResult(
                execution_id=f"error_{uuid.uuid4().hex[:8]}",
                success=False,
                error_message=str(e)
            )
    
    async def execute_custom_workflow(self, workflow: OrchestrationWorkflow) -> ExecutionResult:
        """Execute custom workflow definition"""
        result = await self.execution_engine.execute_workflow(workflow)
        self.execution_history.append(result)
        return result
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status"""
        # Get execution engine metrics
        engine_metrics = self.execution_engine.get_orchestration_metrics()
        
        # Calculate success statistics
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for result in self.execution_history if result.success)
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 100.0
        
        return {
            "orchestrator_status": "operational",
            "registered_templates": len(self.workflow_templates),
            "execution_history": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": success_rate
            },
            "engine_metrics": engine_metrics,
            "optimization_strategies": self.optimization_strategies,
            "system_connectors": len(self.system_connectors)
        }
    
    def get_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered workflow templates"""
        templates = {}
        
        for name, workflow in self.workflow_templates.items():
            templates[name] = {
                "name": workflow.name,
                "description": workflow.description,
                "task_count": len(workflow.tasks),
                "execution_strategy": workflow.execution_strategy.value,
                "max_parallel_tasks": workflow.max_parallel_tasks,
                "workflow_timeout": workflow.workflow_timeout
            }
        
        return templates


# ============================================================================
# GLOBAL ORCHESTRATOR INSTANCE
# ============================================================================

# Global instance for cross-system orchestration
cross_system_orchestrator = CrossSystemOrchestrator()

# Export for external use
__all__ = [
    'WorkflowStatus',
    'TaskType',
    'ExecutionStrategy',
    'WorkflowTask',
    'OrchestrationWorkflow',
    'ExecutionResult',
    'DependencyGraphManager',
    'OrchestrationExecutionEngine',
    'CrossSystemOrchestrator',
    'cross_system_orchestrator'
]