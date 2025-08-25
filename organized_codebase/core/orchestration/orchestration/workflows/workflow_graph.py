"""
Graph-Based Workflow Management for TestMaster

Inspired by LangGraph's supervisor and state graph patterns, this provides
a flexible graph-based workflow system for orchestrating TestMaster operations
including test generation, monitoring, and verification workflows.

Features:
- State-based workflow definitions
- Conditional edges and parallel execution
- Integration with existing TestMaster components
- Toggleable via feature flags
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import copy
import threading
from pathlib import Path

from .feature_flags import FeatureFlags
from .shared_state import get_shared_state
from .tracking_manager import get_tracking_manager
from .monitoring_decorators import monitor_performance


class WorkflowState(Enum):
    """Possible states in a workflow."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Types of nodes in the workflow graph."""
    ACTION = "action"           # Executes an action
    CONDITION = "condition"     # Makes a decision
    PARALLEL = "parallel"       # Runs multiple branches in parallel
    MERGE = "merge"            # Merges parallel branches
    START = "start"            # Start node
    END = "end"                # End node


@dataclass
class WorkflowContext:
    """Context passed through workflow execution."""
    workflow_id: str
    execution_id: str
    start_time: datetime
    current_node: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)
    parallel_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowNode:
    """A node in the workflow graph."""
    name: str
    node_type: NodeType
    handler: Optional[Callable] = None
    condition: Optional[Callable] = None
    parallel_branches: List[str] = field(default_factory=list)
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEdge:
    """An edge connecting nodes in the workflow graph."""
    from_node: str
    to_node: str
    condition: Optional[Callable] = None
    condition_key: Optional[str] = None  # For conditional edges
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowGraph:
    """
    Graph-based workflow management system for TestMaster.
    
    Provides state-based workflow orchestration with conditional routing,
    parallel execution, and integration with TestMaster's monitoring systems.
    """
    
    def __init__(self, name: str = "testmaster_workflow", max_parallel_branches: int = 4):
        """
        Initialize workflow graph.
        
        Args:
            name: Name of the workflow
            max_parallel_branches: Maximum number of parallel branches
        """
        self.enabled = FeatureFlags.is_enabled('layer2_monitoring', 'graph_workflows')
        
        if not self.enabled:
            return
        
        self.name = name
        config = FeatureFlags.get_config('layer2_monitoring', 'graph_workflows')
        self.max_parallel_branches = config.get('max_parallel_branches', max_parallel_branches)
        
        # Graph structure
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self.conditional_edges: Dict[str, Dict[str, str]] = {}  # node -> {condition_result -> target_node}
        
        # Runtime state
        self.current_executions: Dict[str, WorkflowContext] = {}
        self.execution_lock = threading.RLock()
        
        # Integration components
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
            
        if FeatureFlags.is_enabled('layer2_monitoring', 'tracking_manager'):
            self.tracking_manager = get_tracking_manager()
        else:
            self.tracking_manager = None
        
        # Default nodes
        self._add_default_nodes()
        
        print("Graph-based workflow system initialized")
        print(f"   Workflow: {self.name}")
        print(f"   Max parallel branches: {self.max_parallel_branches}")
    
    def _add_default_nodes(self):
        """Add default START and END nodes."""
        self.add_node("START", NodeType.START)
        self.add_node("END", NodeType.END)
    
    def add_node(self, name: str, node_type: NodeType, handler: Callable = None,
                 condition: Callable = None, parallel_branches: List[str] = None,
                 timeout_seconds: float = None, retry_count: int = 0,
                 metadata: Dict[str, Any] = None):
        """
        Add a node to the workflow graph.
        
        Args:
            name: Unique name for the node
            node_type: Type of the node
            handler: Function to execute for action nodes
            condition: Function to evaluate for condition nodes
            parallel_branches: List of branches for parallel nodes
            timeout_seconds: Timeout for node execution
            retry_count: Number of retries on failure
            metadata: Additional metadata for the node
        """
        if not self.enabled:
            return
        
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in workflow")
        
        self.nodes[name] = WorkflowNode(
            name=name,
            node_type=node_type,
            handler=handler,
            condition=condition,
            parallel_branches=parallel_branches or [],
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            metadata=metadata or {}
        )
    
    def add_edge(self, from_node: str, to_node: str, metadata: Dict[str, Any] = None):
        """
        Add a simple edge between two nodes.
        
        Args:
            from_node: Source node name
            to_node: Target node name
            metadata: Additional metadata for the edge
        """
        if not self.enabled:
            return
        
        self._validate_nodes_exist([from_node, to_node])
        
        self.edges.append(WorkflowEdge(
            from_node=from_node,
            to_node=to_node,
            metadata=metadata or {}
        ))
    
    def add_conditional_edge(self, from_node: str, condition: Callable,
                           condition_map: Dict[str, str], default: str = "END"):
        """
        Add a conditional edge that routes based on condition result.
        
        Args:
            from_node: Source node name
            condition: Function that returns a string key
            condition_map: Mapping of condition results to target nodes
            default: Default target node if condition result not in map
        """
        if not self.enabled:
            return
        
        nodes_to_validate = [from_node] + list(condition_map.values()) + [default]
        self._validate_nodes_exist(nodes_to_validate)
        
        # Store conditional logic
        self.conditional_edges[from_node] = {
            'condition': condition,
            'map': condition_map,
            'default': default
        }
    
    def _validate_nodes_exist(self, node_names: List[str]):
        """Validate that all specified nodes exist."""
        for node_name in node_names:
            if node_name not in self.nodes and node_name != "END":
                raise ValueError(f"Node '{node_name}' does not exist in workflow")
    
    @monitor_performance(name="workflow_invoke")
    def invoke(self, initial_data: Dict[str, Any] = None, execution_id: str = None) -> Dict[str, Any]:
        """
        Execute the workflow synchronously.
        
        Args:
            initial_data: Initial data to pass to the workflow
            execution_id: Optional execution ID for tracking
            
        Returns:
            Final workflow context data
        """
        if not self.enabled:
            # Fallback to simple linear execution
            return self._execute_linear_fallback(initial_data or {})
        
        execution_id = execution_id or f"exec_{int(time.time() * 1000)}"
        
        # Start tracking if enabled
        chain_id = None
        if self.tracking_manager:
            chain_id = self.tracking_manager.start_chain(
                chain_name=f"workflow_{self.name}",
                inputs={
                    'execution_id': execution_id,
                    'initial_data_keys': list((initial_data or {}).keys()),
                    'workflow_name': self.name
                }
            )
        
        try:
            context = WorkflowContext(
                workflow_id=self.name,
                execution_id=execution_id,
                start_time=datetime.now(),
                data=copy.deepcopy(initial_data or {}),
                metadata={'chain_id': chain_id}
            )
            
            with self.execution_lock:
                self.current_executions[execution_id] = context
            
            # Execute workflow
            result = self._execute_workflow(context)
            
            # Track completion
            if self.tracking_manager:
                self.tracking_manager.end_chain(
                    chain_id=chain_id,
                    outputs={
                        'execution_id': execution_id,
                        'final_state': result.get('final_state', 'unknown'),
                        'nodes_executed': len(context.execution_path),
                        'success': result.get('success', False)
                    },
                    success=result.get('success', False)
                )
            
            return result
            
        except Exception as e:
            if self.tracking_manager and chain_id:
                self.tracking_manager.end_chain(
                    chain_id=chain_id,
                    success=False,
                    error=f"Workflow execution failed: {str(e)}"
                )
            raise
        finally:
            with self.execution_lock:
                self.current_executions.pop(execution_id, None)
    
    async def ainvoke(self, initial_data: Dict[str, Any] = None, execution_id: str = None) -> Dict[str, Any]:
        """
        Execute the workflow asynchronously.
        
        Args:
            initial_data: Initial data to pass to the workflow
            execution_id: Optional execution ID for tracking
            
        Returns:
            Final workflow context data
        """
        if not self.enabled:
            return self._execute_linear_fallback(initial_data or {})
        
        # For now, run synchronous version in thread pool
        # Can be enhanced for true async execution later
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, initial_data, execution_id)
    
    def _execute_workflow(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute the workflow starting from START node."""
        current_node = "START"
        context.current_node = current_node
        context.execution_path.append(current_node)
        
        while current_node != "END":
            try:
                # Execute current node
                node = self.nodes[current_node]
                
                # Track node execution
                if self.tracking_manager:
                    self.tracking_manager.track_operation(
                        run_id=f"node_{current_node}_{int(time.time() * 1000)}",
                        component="workflow_graph",
                        operation=f"execute_node_{node.node_type.value}",
                        inputs={
                            'node_name': current_node,
                            'node_type': node.node_type.value,
                            'execution_id': context.execution_id
                        },
                        parent_run_id=context.metadata.get('chain_id')
                    )
                
                # Execute node based on type
                if node.node_type == NodeType.ACTION:
                    self._execute_action_node(node, context)
                elif node.node_type == NodeType.CONDITION:
                    pass  # Condition evaluation happens in routing
                elif node.node_type == NodeType.PARALLEL:
                    self._execute_parallel_node(node, context)
                elif node.node_type == NodeType.START:
                    pass  # START node is just a marker
                
                # Determine next node
                next_node = self._get_next_node(current_node, context)
                
                if next_node == current_node:
                    raise RuntimeError(f"Infinite loop detected at node '{current_node}'")
                
                current_node = next_node
                context.current_node = current_node
                context.execution_path.append(current_node)
                
                # Update shared state if enabled
                if self.shared_state:
                    self.shared_state.set(f"workflow_current_{context.execution_id}", current_node)
                
            except Exception as e:
                error_msg = f"Error executing node '{current_node}': {str(e)}"
                context.errors.append(error_msg)
                print(f"Workflow error: {error_msg}")
                
                # Track error
                if self.tracking_manager:
                    self.tracking_manager.track_operation(
                        run_id=f"error_{current_node}_{int(time.time() * 1000)}",
                        component="workflow_graph",
                        operation="node_execution_error",
                        error=error_msg,
                        parent_run_id=context.metadata.get('chain_id'),
                        success=False
                    )
                
                return {
                    'success': False,
                    'error': error_msg,
                    'final_state': 'FAILED',
                    'execution_path': context.execution_path,
                    'context': context.data
                }
        
        return {
            'success': True,
            'final_state': 'COMPLETED',
            'execution_path': context.execution_path,
            'context': context.data,
            'parallel_results': context.parallel_results
        }
    
    def _execute_action_node(self, node: WorkflowNode, context: WorkflowContext):
        """Execute an action node."""
        if node.handler:
            try:
                result = node.handler(context)
                if result is not None:
                    context.data.update(result)
            except Exception as e:
                if node.retry_count > 0:
                    # Implement retry logic if needed
                    pass
                raise
    
    def _execute_parallel_node(self, node: WorkflowNode, context: WorkflowContext):
        """Execute a parallel node with multiple branches."""
        if not node.parallel_branches:
            return
        
        # Limit parallel branches
        branches = node.parallel_branches[:self.max_parallel_branches]
        
        # For now, execute sequentially (can be enhanced for true parallelism)
        for branch in branches:
            if branch in self.nodes:
                branch_node = self.nodes[branch]
                if branch_node.handler:
                    try:
                        result = branch_node.handler(context)
                        if result is not None:
                            context.parallel_results[branch] = result
                    except Exception as e:
                        context.parallel_results[branch] = {'error': str(e)}
    
    def _get_next_node(self, current_node: str, context: WorkflowContext) -> str:
        """Determine the next node based on edges and conditions."""
        # Check for conditional edges first
        if current_node in self.conditional_edges:
            conditional_logic = self.conditional_edges[current_node]
            condition = conditional_logic['condition']
            condition_map = conditional_logic['map']
            default = conditional_logic['default']
            
            try:
                condition_result = condition(context)
                return condition_map.get(str(condition_result), default)
            except Exception as e:
                print(f"Condition evaluation failed for node '{current_node}': {e}")
                return default
        
        # Check for simple edges
        for edge in self.edges:
            if edge.from_node == current_node:
                return edge.to_node
        
        # Default to END if no edges found
        return "END"
    
    def _execute_linear_fallback(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback linear execution when graph workflows are disabled."""
        return {
            'success': True,
            'final_state': 'COMPLETED_LINEAR',
            'execution_path': ['START', 'END'],
            'context': initial_data,
            'fallback': True
        }
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        with self.execution_lock:
            active_executions = len(self.current_executions)
            execution_states = [ctx.current_node for ctx in self.current_executions.values()]
        
        stats = {
            'enabled': True,
            'workflow_name': self.name,
            'nodes_count': len(self.nodes),
            'edges_count': len(self.edges),
            'conditional_edges_count': len(self.conditional_edges),
            'max_parallel_branches': self.max_parallel_branches,
            'active_executions': active_executions,
            'current_states': execution_states
        }
        
        # Add shared state statistics if available
        if self.shared_state:
            # SharedState doesn't have get_all_keys, so we'll just count what we know
            stats['shared_state_keys'] = 0  # Could be enhanced if SharedState adds key enumeration
        
        return stats
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific execution."""
        if not self.enabled:
            return None
        
        with self.execution_lock:
            context = self.current_executions.get(execution_id)
            if not context:
                return None
            
            return {
                'execution_id': execution_id,
                'workflow_id': context.workflow_id,
                'current_node': context.current_node,
                'start_time': context.start_time.isoformat(),
                'execution_path': context.execution_path,
                'errors': context.errors,
                'parallel_results_count': len(context.parallel_results)
            }
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        if not self.enabled:
            return False
        
        with self.execution_lock:
            if execution_id in self.current_executions:
                del self.current_executions[execution_id]
                
                # Track cancellation
                if self.tracking_manager:
                    self.tracking_manager.track_operation(
                        run_id=f"cancel_{execution_id}",
                        component="workflow_graph",
                        operation="execution_cancelled",
                        inputs={'execution_id': execution_id}
                    )
                
                return True
        
        return False


# Factory functions for creating common workflow patterns

def create_test_monitoring_workflow() -> WorkflowGraph:
    """Create a workflow for test monitoring operations."""
    workflow = WorkflowGraph("test_monitoring")
    
    if not workflow.enabled:
        return workflow
    
    # Add monitoring nodes
    workflow.add_node("detect_change", NodeType.ACTION, 
                     handler=lambda ctx: {"change_detected": True})
    workflow.add_node("analyze_impact", NodeType.ACTION,
                     handler=lambda ctx: {"impact_level": "medium"})
    workflow.add_node("should_generate", NodeType.CONDITION)
    workflow.add_node("generate_tests", NodeType.ACTION,
                     handler=lambda ctx: {"tests_generated": True})
    workflow.add_node("verify_tests", NodeType.ACTION,
                     handler=lambda ctx: {"tests_verified": True})
    
    # Add edges
    workflow.add_edge("START", "detect_change")
    workflow.add_edge("detect_change", "analyze_impact")
    workflow.add_conditional_edge("analyze_impact", 
                                 lambda ctx: "generate" if ctx.data.get("impact_level") != "low" else "skip",
                                 {"generate": "generate_tests", "skip": "END"})
    workflow.add_edge("generate_tests", "verify_tests")
    workflow.add_edge("verify_tests", "END")
    
    return workflow


def create_parallel_test_generation_workflow() -> WorkflowGraph:
    """Create a workflow for parallel test generation."""
    workflow = WorkflowGraph("parallel_test_generation")
    
    if not workflow.enabled:
        return workflow
    
    # Add nodes for parallel processing
    workflow.add_node("prepare_modules", NodeType.ACTION,
                     handler=lambda ctx: {"modules_prepared": True})
    workflow.add_node("parallel_generate", NodeType.PARALLEL,
                     parallel_branches=["unit_tests", "integration_tests", "performance_tests"])
    workflow.add_node("unit_tests", NodeType.ACTION,
                     handler=lambda ctx: {"unit_tests": "generated"})
    workflow.add_node("integration_tests", NodeType.ACTION,
                     handler=lambda ctx: {"integration_tests": "generated"})
    workflow.add_node("performance_tests", NodeType.ACTION,
                     handler=lambda ctx: {"performance_tests": "generated"})
    workflow.add_node("merge_results", NodeType.MERGE)
    workflow.add_node("validate_all", NodeType.ACTION,
                     handler=lambda ctx: {"all_validated": True})
    
    # Add edges
    workflow.add_edge("START", "prepare_modules")
    workflow.add_edge("prepare_modules", "parallel_generate")
    workflow.add_edge("parallel_generate", "merge_results")
    workflow.add_edge("merge_results", "validate_all")
    workflow.add_edge("validate_all", "END")
    
    return workflow


# Global workflow manager instance
_workflow_graphs: Dict[str, WorkflowGraph] = {}


def get_workflow_graph(name: str = "default") -> WorkflowGraph:
    """Get or create a workflow graph instance."""
    global _workflow_graphs
    if name not in _workflow_graphs:
        _workflow_graphs[name] = WorkflowGraph(name)
    return _workflow_graphs[name]


def register_workflow_graph(name: str, workflow: WorkflowGraph):
    """Register a named workflow graph."""
    global _workflow_graphs
    _workflow_graphs[name] = workflow