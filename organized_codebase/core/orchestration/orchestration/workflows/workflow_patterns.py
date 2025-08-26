"""
Workflow Patterns
================

Workflow orchestration patterns for DAG-based task execution
and dependency-driven workflow management.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from collections import defaultdict, deque


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """Workflow node status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowNode:
    """Workflow node representing a task or operation."""
    node_id: str
    name: str
    task_definition: Any
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowPattern(ABC):
    """Abstract base class for workflow patterns."""
    
    def __init__(self, pattern_name: str):
        self.pattern_name = pattern_name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.status = WorkflowStatus.PENDING
        self.execution_order: List[str] = []
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow pattern."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate workflow structure."""
        pass
    
    def add_node(self, node: WorkflowNode):
        """Add node to workflow."""
        self.nodes[node.node_id] = node
    
    def add_dependency(self, dependent_id: str, dependency_id: str):
        """Add dependency between nodes."""
        if dependent_id in self.nodes and dependency_id in self.nodes:
            self.nodes[dependent_id].dependencies.add(dependency_id)
            self.nodes[dependency_id].dependents.add(dependent_id)
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit event to handlers."""
        for handler in self.event_handlers[event_type]:
            try:
                handler(event_type, event_data)
            except Exception:
                pass


class DAGPattern(WorkflowPattern):
    """
    Directed Acyclic Graph (DAG) workflow pattern.
    
    Executes tasks based on dependency relationships in a DAG structure.
    Supports parallel execution of independent tasks and ensures proper
    dependency ordering.
    """
    
    def __init__(self, dag_name: str = "dag_workflow"):
        super().__init__(dag_name)
        self.parallel_execution = True
        self.max_parallel_tasks = 10
        self.topological_order: List[str] = []
    
    def validate(self) -> bool:
        """Validate DAG structure (no cycles, valid dependencies)."""
        try:
            # Check for cycles using DFS
            visited = set()
            rec_stack = set()
            
            def has_cycle(node_id: str) -> bool:
                visited.add(node_id)
                rec_stack.add(node_id)
                
                for dependent_id in self.nodes[node_id].dependents:
                    if dependent_id not in visited:
                        if has_cycle(dependent_id):
                            return True
                    elif dependent_id in rec_stack:
                        return True
                
                rec_stack.remove(node_id)
                return False
            
            # Check all nodes for cycles
            for node_id in self.nodes:
                if node_id not in visited:
                    if has_cycle(node_id):
                        return False
            
            # Generate topological order
            self.topological_order = self._topological_sort()
            return True
            
        except Exception:
            return False
    
    def _topological_sort(self) -> List[str]:
        """Generate topological ordering of nodes."""
        in_degree = {node_id: len(node.dependencies) for node_id, node in self.nodes.items()}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        topo_order = []
        
        while queue:
            node_id = queue.popleft()
            topo_order.append(node_id)
            
            for dependent_id in self.nodes[node_id].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        return topo_order
    
    async def execute(self) -> Dict[str, Any]:
        """Execute DAG workflow."""
        if not self.validate():
            return {"status": "failed", "error": "Invalid DAG structure"}
        
        self.status = WorkflowStatus.RUNNING
        self._emit_event("workflow_started", {"dag_name": self.pattern_name})
        
        try:
            if self.parallel_execution:
                result = await self._execute_parallel()
            else:
                result = await self._execute_sequential()
            
            self.status = WorkflowStatus.COMPLETED
            self._emit_event("workflow_completed", {"result": result})
            return result
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            error_result = {"status": "failed", "error": str(e)}
            self._emit_event("workflow_failed", error_result)
            return error_result
    
    async def _execute_sequential(self) -> Dict[str, Any]:
        """Execute nodes sequentially in topological order."""
        results = {}
        
        for node_id in self.topological_order:
            node = self.nodes[node_id]
            
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(node_id):
                node.status = NodeStatus.SKIPPED
                continue
            
            try:
                node.status = NodeStatus.RUNNING
                node.start_time = datetime.now()
                
                # Execute task
                result = await self._execute_node(node)
                
                node.result = result
                node.status = NodeStatus.COMPLETED
                node.end_time = datetime.now()
                results[node_id] = result
                
                self._emit_event("node_completed", {
                    "node_id": node_id,
                    "result": result
                })
                
            except Exception as e:
                node.error = str(e)
                node.status = NodeStatus.FAILED
                node.end_time = datetime.now()
                
                self._emit_event("node_failed", {
                    "node_id": node_id,
                    "error": str(e)
                })
                
                # Handle failure based on strategy
                if not await self._handle_node_failure(node):
                    break
        
        return {"status": "completed", "results": results}
    
    async def _execute_parallel(self) -> Dict[str, Any]:
        """Execute nodes in parallel respecting dependencies."""
        results = {}
        running_tasks = {}
        completed_nodes = set()
        failed_nodes = set()
        
        # Initialize ready nodes (no dependencies)
        ready_nodes = [
            node_id for node_id, node in self.nodes.items()
            if len(node.dependencies) == 0
        ]
        
        while ready_nodes or running_tasks:
            # Start new tasks for ready nodes
            while ready_nodes and len(running_tasks) < self.max_parallel_tasks:
                node_id = ready_nodes.pop(0)
                node = self.nodes[node_id]
                
                node.status = NodeStatus.RUNNING
                node.start_time = datetime.now()
                
                # Create and start task
                task = asyncio.create_task(self._execute_node(node))
                running_tasks[node_id] = task
                
                self._emit_event("node_started", {"node_id": node_id})
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    # Find node_id for completed task
                    node_id = None
                    for nid, t in running_tasks.items():
                        if t == task:
                            node_id = nid
                            break
                    
                    if node_id:
                        node = self.nodes[node_id]
                        del running_tasks[node_id]
                        
                        try:
                            result = await task
                            node.result = result
                            node.status = NodeStatus.COMPLETED
                            node.end_time = datetime.now()
                            results[node_id] = result
                            completed_nodes.add(node_id)
                            
                            self._emit_event("node_completed", {
                                "node_id": node_id,
                                "result": result
                            })
                            
                            # Check for newly ready nodes
                            self._update_ready_nodes(
                                node_id, ready_nodes, completed_nodes, failed_nodes
                            )
                            
                        except Exception as e:
                            node.error = str(e)
                            node.status = NodeStatus.FAILED
                            node.end_time = datetime.now()
                            failed_nodes.add(node_id)
                            
                            self._emit_event("node_failed", {
                                "node_id": node_id,
                                "error": str(e)
                            })
                            
                            # Handle failure
                            if not await self._handle_node_failure(node):
                                # Cancel remaining tasks
                                for remaining_task in running_tasks.values():
                                    remaining_task.cancel()
                                return {"status": "failed", "error": str(e)}
        
        return {"status": "completed", "results": results}
    
    def _dependencies_satisfied(self, node_id: str) -> bool:
        """Check if node dependencies are satisfied."""
        node = self.nodes[node_id]
        for dep_id in node.dependencies:
            dep_node = self.nodes[dep_id]
            if dep_node.status != NodeStatus.COMPLETED:
                return False
        return True
    
    def _update_ready_nodes(
        self,
        completed_node_id: str,
        ready_nodes: List[str],
        completed_nodes: Set[str],
        failed_nodes: Set[str]
    ):
        """Update ready nodes list after node completion."""
        for dependent_id in self.nodes[completed_node_id].dependents:
            if dependent_id not in ready_nodes and dependent_id not in completed_nodes:
                # Check if all dependencies are satisfied
                node = self.nodes[dependent_id]
                all_deps_satisfied = True
                
                for dep_id in node.dependencies:
                    if dep_id not in completed_nodes:
                        all_deps_satisfied = False
                        break
                
                if all_deps_satisfied:
                    ready_nodes.append(dependent_id)
    
    async def _execute_node(self, node: WorkflowNode) -> Any:
        """Execute individual node task."""
        # This would be implemented based on task definition
        # For now, return a placeholder
        await asyncio.sleep(0.1)  # Simulate task execution
        return f"Result for {node.name}"
    
    async def _handle_node_failure(self, node: WorkflowNode) -> bool:
        """Handle node execution failure."""
        if node.retry_count < node.max_retries:
            node.retry_count += 1
            node.status = NodeStatus.PENDING
            return True
        return False


class PipelinePattern(WorkflowPattern):
    """
    Pipeline workflow pattern for sequential data processing.
    
    Executes tasks in a linear pipeline where each stage processes
    the output of the previous stage.
    """
    
    def __init__(self, pipeline_name: str = "pipeline_workflow"):
        super().__init__(pipeline_name)
        self.stages: List[str] = []
        self.stage_results: Dict[str, Any] = {}
    
    def add_stage(self, node: WorkflowNode, position: Optional[int] = None):
        """Add stage to pipeline."""
        self.add_node(node)
        
        if position is None:
            self.stages.append(node.node_id)
        else:
            self.stages.insert(position, node.node_id)
        
        # Setup dependencies
        self._setup_pipeline_dependencies()
    
    def _setup_pipeline_dependencies(self):
        """Setup sequential dependencies for pipeline stages."""
        for i in range(1, len(self.stages)):
            current_stage = self.stages[i]
            previous_stage = self.stages[i - 1]
            self.add_dependency(current_stage, previous_stage)
    
    def validate(self) -> bool:
        """Validate pipeline structure."""
        return len(self.stages) > 0 and all(stage_id in self.nodes for stage_id in self.stages)
    
    async def execute(self) -> Dict[str, Any]:
        """Execute pipeline stages sequentially."""
        if not self.validate():
            return {"status": "failed", "error": "Invalid pipeline structure"}
        
        self.status = WorkflowStatus.RUNNING
        self._emit_event("pipeline_started", {"pipeline_name": self.pattern_name})
        
        try:
            pipeline_input = None
            
            for stage_id in self.stages:
                node = self.nodes[stage_id]
                
                node.status = NodeStatus.RUNNING
                node.start_time = datetime.now()
                
                # Execute stage with previous stage output as input
                try:
                    result = await self._execute_pipeline_stage(node, pipeline_input)
                    
                    node.result = result
                    node.status = NodeStatus.COMPLETED
                    node.end_time = datetime.now()
                    self.stage_results[stage_id] = result
                    
                    # Output becomes input for next stage
                    pipeline_input = result
                    
                    self._emit_event("stage_completed", {
                        "stage_id": stage_id,
                        "result": result
                    })
                    
                except Exception as e:
                    node.error = str(e)
                    node.status = NodeStatus.FAILED
                    node.end_time = datetime.now()
                    
                    self._emit_event("stage_failed", {
                        "stage_id": stage_id,
                        "error": str(e)
                    })
                    
                    # Pipeline fails if any stage fails
                    self.status = WorkflowStatus.FAILED
                    return {"status": "failed", "error": str(e), "failed_stage": stage_id}
            
            self.status = WorkflowStatus.COMPLETED
            result = {
                "status": "completed",
                "final_result": pipeline_input,
                "stage_results": self.stage_results
            }
            
            self._emit_event("pipeline_completed", result)
            return result
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            error_result = {"status": "failed", "error": str(e)}
            self._emit_event("pipeline_failed", error_result)
            return error_result
    
    async def _execute_pipeline_stage(self, node: WorkflowNode, stage_input: Any) -> Any:
        """Execute individual pipeline stage."""
        # This would be implemented based on task definition
        # For now, return a placeholder that incorporates input
        await asyncio.sleep(0.1)  # Simulate stage processing
        return f"Stage {node.name} processed: {stage_input}"


class ConditionalPattern(WorkflowPattern):
    """
    Conditional workflow pattern for branching execution.
    
    Executes different branches based on conditions and allows
    for conditional task execution and dynamic workflow routing.
    """
    
    def __init__(self, conditional_name: str = "conditional_workflow"):
        super().__init__(conditional_name)
        self.conditions: Dict[str, Callable] = {}
        self.branches: Dict[str, List[str]] = {}
        self.active_branch: Optional[str] = None
    
    def add_condition(self, condition_name: str, condition_func: Callable):
        """Add condition for branching."""
        self.conditions[condition_name] = condition_func
    
    def add_branch(self, branch_name: str, node_ids: List[str]):
        """Add execution branch."""
        self.branches[branch_name] = node_ids
    
    def validate(self) -> bool:
        """Validate conditional workflow structure."""
        # Check that all branch nodes exist
        for branch_nodes in self.branches.values():
            for node_id in branch_nodes:
                if node_id not in self.nodes:
                    return False
        return True
    
    async def execute(self) -> Dict[str, Any]:
        """Execute conditional workflow."""
        if not self.validate():
            return {"status": "failed", "error": "Invalid conditional structure"}
        
        self.status = WorkflowStatus.RUNNING
        self._emit_event("conditional_started", {"workflow_name": self.pattern_name})
        
        try:
            # Evaluate conditions to determine active branch
            context = {"nodes": self.nodes, "workflow": self}
            
            for condition_name, condition_func in self.conditions.items():
                try:
                    if await self._evaluate_condition(condition_func, context):
                        self.active_branch = condition_name
                        break
                except Exception as e:
                    self._emit_event("condition_error", {
                        "condition": condition_name,
                        "error": str(e)
                    })
            
            if not self.active_branch:
                return {"status": "failed", "error": "No condition satisfied"}
            
            # Execute active branch
            branch_nodes = self.branches[self.active_branch]
            results = {}
            
            for node_id in branch_nodes:
                node = self.nodes[node_id]
                
                node.status = NodeStatus.RUNNING
                node.start_time = datetime.now()
                
                try:
                    result = await self._execute_node(node)
                    
                    node.result = result
                    node.status = NodeStatus.COMPLETED
                    node.end_time = datetime.now()
                    results[node_id] = result
                    
                except Exception as e:
                    node.error = str(e)
                    node.status = NodeStatus.FAILED
                    node.end_time = datetime.now()
                    
                    return {
                        "status": "failed",
                        "error": str(e),
                        "failed_node": node_id,
                        "active_branch": self.active_branch
                    }
            
            self.status = WorkflowStatus.COMPLETED
            result = {
                "status": "completed",
                "active_branch": self.active_branch,
                "results": results
            }
            
            self._emit_event("conditional_completed", result)
            return result
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            error_result = {"status": "failed", "error": str(e)}
            self._emit_event("conditional_failed", error_result)
            return error_result
    
    async def _evaluate_condition(self, condition_func: Callable, context: Dict[str, Any]) -> bool:
        """Evaluate condition function."""
        if asyncio.iscoroutinefunction(condition_func):
            return await condition_func(context)
        else:
            return condition_func(context)


# Export key classes
__all__ = [
    'WorkflowStatus',
    'NodeStatus',
    'WorkflowNode',
    'WorkflowPattern',
    'DAGPattern',
    'PipelinePattern',
    'ConditionalPattern'
]