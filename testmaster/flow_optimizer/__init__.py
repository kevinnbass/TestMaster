"""
Execution Flow Optimizer for TestMaster

Intelligent execution flow optimization system including:
- Flow analysis and bottleneck detection
- Adaptive execution algorithms
- Performance-based routing
- Resource allocation optimization
- Dependency graph optimization
- Parallel execution strategies
"""

from typing import Dict, Any, List, Optional, Union
from .flow_analyzer import (
    FlowAnalyzer, FlowAnalysis, BottleneckInfo,
    FlowMetric, AnalysisType, get_flow_analyzer
)
from .execution_router import (
    ExecutionRouter, Route, RoutingStrategy,
    RouteWeight, PerformanceData, get_execution_router
)
from .resource_optimizer import (
    ResourceOptimizer, ResourceAllocation, ResourceType,
    OptimizationPolicy, get_resource_optimizer
)
from .dependency_resolver import (
    DependencyResolver, DependencyGraph, TaskNode,
    DependencyType, get_dependency_resolver
)
from .parallel_executor import (
    ParallelExecutor, ExecutionPlan, ParallelStrategy,
    TaskBatch, get_parallel_executor
)
from ..core.feature_flags import FeatureFlags

# Global instances
_flow_analyzer = None
_execution_router = None
_resource_optimizer = None
_dependency_resolver = None
_parallel_executor = None

def is_flow_optimizer_enabled() -> bool:
    """Check if execution flow optimizer is enabled."""
    return FeatureFlags.is_enabled('layer3_orchestration', 'flow_optimizer')

def configure_flow_optimizer(
    learning_rate: float = 0.1,
    enable_adaptive_routing: bool = True,
    enable_parallel_execution: bool = True,
    optimization_interval: int = 60
) -> Dict[str, Any]:
    """
    Configure execution flow optimizer.
    
    Args:
        learning_rate: Learning rate for adaptive algorithms
        enable_adaptive_routing: Enable adaptive routing strategies
        enable_parallel_execution: Enable parallel execution
        optimization_interval: Optimization interval in seconds
        
    Returns:
        Configuration status
    """
    if not is_flow_optimizer_enabled():
        return {"status": "disabled", "reason": "flow_optimizer feature not enabled"}
    
    global _flow_analyzer, _execution_router, _resource_optimizer
    global _dependency_resolver, _parallel_executor
    
    # Initialize components
    _flow_analyzer = get_flow_analyzer()
    _execution_router = get_execution_router()
    _resource_optimizer = get_resource_optimizer()
    _dependency_resolver = get_dependency_resolver()
    
    if enable_parallel_execution:
        _parallel_executor = get_parallel_executor()
        _parallel_executor.configure(learning_rate=learning_rate)
    
    if enable_adaptive_routing:
        _execution_router.enable_adaptive_routing(learning_rate=learning_rate)
    
    config = {
        "learning_rate": learning_rate,
        "adaptive_routing": enable_adaptive_routing,
        "parallel_execution": enable_parallel_execution,
        "optimization_interval": optimization_interval
    }
    
    print(f"Execution flow optimizer configured: {config}")
    return {"status": "configured", "config": config}

def analyze_execution_flow(
    workflow_id: str,
    execution_data: List[Dict[str, Any]],
    include_dependencies: bool = True
) -> FlowAnalysis:
    """
    Analyze execution flow for optimization opportunities.
    
    Args:
        workflow_id: Workflow identifier
        execution_data: Historical execution data
        include_dependencies: Include dependency analysis
        
    Returns:
        Flow analysis with bottlenecks and recommendations
    """
    if not is_flow_optimizer_enabled():
        return FlowAnalysis(workflow_id, [], [], 0.0, "disabled")
    
    analyzer = get_flow_analyzer()
    return analyzer.analyze_flow(workflow_id, execution_data, include_dependencies)

def optimize_execution_route(
    task_id: str,
    available_resources: List[Dict[str, Any]],
    performance_history: Dict[str, Any] = None
) -> Route:
    """
    Optimize execution route for a task.
    
    Args:
        task_id: Task identifier
        available_resources: Available execution resources
        performance_history: Historical performance data
        
    Returns:
        Optimized execution route
    """
    if not is_flow_optimizer_enabled():
        return Route(task_id, [], "disabled", 0.0)
    
    router = get_execution_router()
    return router.find_optimal_route(task_id, available_resources, performance_history)

def optimize_resource_allocation(
    workflow_id: str,
    tasks: List[Dict[str, Any]],
    available_resources: Dict[str, Any],
    constraints: Dict[str, Any] = None
) -> ResourceAllocation:
    """
    Optimize resource allocation for workflow execution.
    
    Args:
        workflow_id: Workflow identifier
        tasks: List of tasks to execute
        available_resources: Available system resources
        constraints: Resource allocation constraints
        
    Returns:
        Optimized resource allocation plan
    """
    if not is_flow_optimizer_enabled():
        return ResourceAllocation(workflow_id, {}, "disabled")
    
    optimizer = get_resource_optimizer()
    return optimizer.optimize_allocation(workflow_id, tasks, available_resources, constraints)

def resolve_dependencies(
    tasks: List[Dict[str, Any]],
    dependency_rules: List[Dict[str, Any]] = None
) -> DependencyGraph:
    """
    Resolve task dependencies and create execution graph.
    
    Args:
        tasks: List of tasks with dependencies
        dependency_rules: Custom dependency rules
        
    Returns:
        Dependency graph with execution order
    """
    if not is_flow_optimizer_enabled():
        return DependencyGraph([], [], "disabled")
    
    resolver = get_dependency_resolver()
    return resolver.resolve_dependencies(tasks, dependency_rules)

def create_parallel_execution_plan(
    workflow_id: str,
    dependency_graph: DependencyGraph,
    resource_allocation: ResourceAllocation,
    strategy: str = "balanced"
) -> ExecutionPlan:
    """
    Create parallel execution plan for workflow.
    
    Args:
        workflow_id: Workflow identifier
        dependency_graph: Task dependency graph
        resource_allocation: Resource allocation plan
        strategy: Parallel execution strategy
        
    Returns:
        Parallel execution plan with batched tasks
    """
    if not is_flow_optimizer_enabled():
        return ExecutionPlan(workflow_id, [], "disabled")
    
    executor = get_parallel_executor()
    return executor.create_execution_plan(workflow_id, dependency_graph, resource_allocation, strategy)

def get_optimization_status() -> Dict[str, Any]:
    """
    Get current optimization status.
    
    Returns:
        Optimization status summary
    """
    if not is_flow_optimizer_enabled():
        return {"status": "disabled"}
    
    status = {
        "flow_analyzer": _flow_analyzer is not None,
        "execution_router": _execution_router is not None,
        "resource_optimizer": _resource_optimizer is not None,
        "dependency_resolver": _dependency_resolver is not None,
        "parallel_executor": _parallel_executor is not None
    }
    
    return {"status": "active", "components": status}

def shutdown_flow_optimizer():
    """Shutdown execution flow optimizer."""
    global _flow_analyzer, _execution_router, _resource_optimizer
    global _dependency_resolver, _parallel_executor
    
    if _parallel_executor:
        _parallel_executor.shutdown()
    
    if _execution_router:
        _execution_router.shutdown()
    
    # Reset instances
    _flow_analyzer = None
    _execution_router = None
    _resource_optimizer = None
    _dependency_resolver = None
    _parallel_executor = None
    
    print("Execution flow optimizer shutdown completed")

# Convenience aliases
analyze_flow = analyze_execution_flow
optimize_route = optimize_execution_route
optimize_resources = optimize_resource_allocation
resolve_deps = resolve_dependencies
create_execution_plan = create_parallel_execution_plan