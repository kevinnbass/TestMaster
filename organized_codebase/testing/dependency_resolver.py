"""
Dependency Resolver for TestMaster Flow Optimizer

Resolves task dependencies and creates optimized execution graphs.
"""

import threading
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

from core.feature_flags import FeatureFlags

class DependencyType(Enum):
    """Types of dependencies."""
    SEQUENTIAL = "sequential"
    DATA = "data"
    RESOURCE = "resource"
    TEMPORAL = "temporal"
    CONDITIONAL = "conditional"

@dataclass
class TaskNode:
    """Task node in dependency graph."""
    task_id: str
    dependencies: List[str]
    dependents: List[str]
    priority: int
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    
    def __post_init__(self):
        if not self.dependencies:
            self.dependencies = []
        if not self.dependents:
            self.dependents = []
        if not self.resource_requirements:
            self.resource_requirements = {}

@dataclass
class DependencyGraph:
    """Dependency graph with execution order."""
    nodes: List[TaskNode]
    execution_levels: List[List[str]]
    status: str
    critical_path: List[str] = None
    total_estimated_time: float = 0.0
    
    def __post_init__(self):
        if self.critical_path is None:
            self.critical_path = []

class DependencyResolver:
    """Dependency resolver for task execution."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'flow_optimizer')
        self.lock = threading.RLock()
        self.resolution_cache: Dict[str, DependencyGraph] = {}
        
        if not self.enabled:
            return
        
        print("Dependency resolver initialized")
        print("   Dependency types: sequential, data, resource, temporal, conditional")
    
    def resolve_dependencies(
        self,
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
        if not self.enabled:
            return DependencyGraph([], [], "disabled")
        
        # Create task nodes
        nodes = self._create_task_nodes(tasks)
        
        # Apply dependency rules
        if dependency_rules:
            self._apply_dependency_rules(nodes, dependency_rules)
        
        # Build dependency graph
        dependency_map = self._build_dependency_map(nodes)
        
        # Perform topological sort
        execution_levels = self._topological_sort(nodes, dependency_map)
        
        # Find critical path
        critical_path = self._find_critical_path(nodes, dependency_map)
        
        # Calculate total estimated time
        total_time = self._calculate_total_time(execution_levels, nodes)
        
        graph = DependencyGraph(
            nodes=nodes,
            execution_levels=execution_levels,
            status="resolved",
            critical_path=critical_path,
            total_estimated_time=total_time
        )
        
        print(f"Dependencies resolved: {len(nodes)} tasks, {len(execution_levels)} levels, critical path: {len(critical_path)} tasks")
        
        return graph
    
    def _create_task_nodes(self, tasks: List[Dict[str, Any]]) -> List[TaskNode]:
        """Create task nodes from task specifications."""
        nodes = []
        
        for task in tasks:
            node = TaskNode(
                task_id=task.get('id', f"task_{len(nodes)}"),
                dependencies=task.get('dependencies', []),
                dependents=[],  # Will be populated later
                priority=task.get('priority', 5),
                estimated_duration=task.get('estimated_duration', 100.0),
                resource_requirements=task.get('resource_requirements', {})
            )
            nodes.append(node)
        
        # Populate dependents
        node_map = {node.task_id: node for node in nodes}
        for node in nodes:
            for dep_id in node.dependencies:
                if dep_id in node_map:
                    node_map[dep_id].dependents.append(node.task_id)
        
        return nodes
    
    def _apply_dependency_rules(self, nodes: List[TaskNode], dependency_rules: List[Dict[str, Any]]):
        """Apply custom dependency rules to tasks."""
        node_map = {node.task_id: node for node in nodes}
        
        for rule in dependency_rules:
            rule_type = rule.get('type', 'sequential')
            source_pattern = rule.get('source_pattern', '')
            target_pattern = rule.get('target_pattern', '')
            
            # Find matching tasks
            source_tasks = [node for node in nodes if source_pattern in node.task_id]
            target_tasks = [node for node in nodes if target_pattern in node.task_id]
            
            # Apply rule
            if rule_type == 'sequential':
                for source in source_tasks:
                    for target in target_tasks:
                        if target.task_id not in source.dependents:
                            source.dependents.append(target.task_id)
                        if source.task_id not in target.dependencies:
                            target.dependencies.append(source.task_id)
    
    def _build_dependency_map(self, nodes: List[TaskNode]) -> Dict[str, Set[str]]:
        """Build dependency map for graph algorithms."""
        dependency_map = defaultdict(set)
        
        for node in nodes:
            for dep_id in node.dependencies:
                dependency_map[node.task_id].add(dep_id)
        
        return dependency_map
    
    def _topological_sort(self, nodes: List[TaskNode], dependency_map: Dict[str, Set[str]]) -> List[List[str]]:
        """Perform topological sort to determine execution levels."""
        # Calculate in-degrees
        in_degree = defaultdict(int)
        node_ids = {node.task_id for node in nodes}
        
        for node_id in node_ids:
            in_degree[node_id] = len(dependency_map[node_id])
        
        # Initialize queue with nodes having no dependencies
        queue = deque([node_id for node_id in node_ids if in_degree[node_id] == 0])
        execution_levels = []
        
        while queue:
            # Process all nodes at current level
            current_level = []
            level_size = len(queue)
            
            for _ in range(level_size):
                current_node = queue.popleft()
                current_level.append(current_node)
                
                # Find node object
                node = next((n for n in nodes if n.task_id == current_node), None)
                if node:
                    # Reduce in-degree of dependents
                    for dependent_id in node.dependents:
                        if dependent_id in in_degree:
                            in_degree[dependent_id] -= 1
                            if in_degree[dependent_id] == 0:
                                queue.append(dependent_id)
            
            if current_level:
                execution_levels.append(current_level)
        
        return execution_levels
    
    def _find_critical_path(self, nodes: List[TaskNode], dependency_map: Dict[str, Set[str]]) -> List[str]:
        """Find critical path through the dependency graph."""
        node_map = {node.task_id: node for node in nodes}
        
        # Calculate earliest start times
        earliest_start = {}
        for node in nodes:
            earliest_start[node.task_id] = 0.0
        
        # Forward pass - calculate earliest start times
        for node in nodes:
            for dep_id in node.dependencies:
                if dep_id in node_map:
                    dep_node = node_map[dep_id]
                    earliest_start[node.task_id] = max(
                        earliest_start[node.task_id],
                        earliest_start[dep_id] + dep_node.estimated_duration
                    )
        
        # Calculate latest start times
        latest_start = {}
        
        # Find end nodes (nodes with no dependents)
        end_nodes = [node for node in nodes if not node.dependents]
        
        # Initialize latest start times for end nodes
        for node in end_nodes:
            latest_start[node.task_id] = earliest_start[node.task_id]
        
        # Backward pass - calculate latest start times
        # Process nodes in reverse topological order
        processed = set()
        for node in reversed(nodes):
            if node.task_id not in latest_start:
                latest_start[node.task_id] = float('inf')
                
                for dependent_id in node.dependents:
                    if dependent_id in node_map and dependent_id in latest_start:
                        latest_start[node.task_id] = min(
                            latest_start[node.task_id],
                            latest_start[dependent_id] - node.estimated_duration
                        )
                
                if latest_start[node.task_id] == float('inf'):
                    latest_start[node.task_id] = earliest_start[node.task_id]
        
        # Find critical path (nodes where earliest_start == latest_start)
        critical_nodes = [
            node.task_id for node in nodes
            if abs(earliest_start[node.task_id] - latest_start[node.task_id]) < 0.01
        ]
        
        # Order critical nodes by earliest start time
        critical_path = sorted(critical_nodes, key=lambda x: earliest_start[x])
        
        return critical_path
    
    def _calculate_total_time(self, execution_levels: List[List[str]], nodes: List[TaskNode]) -> float:
        """Calculate total estimated execution time."""
        node_map = {node.task_id: node for node in nodes}
        total_time = 0.0
        
        for level in execution_levels:
            # Find maximum duration in this level (parallel execution)
            level_max_duration = 0.0
            for task_id in level:
                if task_id in node_map:
                    level_max_duration = max(level_max_duration, node_map[task_id].estimated_duration)
            total_time += level_max_duration
        
        return total_time
    
    def optimize_dependency_graph(self, graph: DependencyGraph) -> DependencyGraph:
        """Optimize dependency graph for better parallelization."""
        if not self.enabled:
            return graph
        
        # Create optimized copy
        optimized_nodes = []
        node_map = {node.task_id: node for node in graph.nodes}
        
        for node in graph.nodes:
            # Optimize dependencies by removing redundant ones
            optimized_deps = self._remove_redundant_dependencies(node, node_map)
            
            optimized_node = TaskNode(
                task_id=node.task_id,
                dependencies=optimized_deps,
                dependents=node.dependents.copy(),
                priority=node.priority,
                estimated_duration=node.estimated_duration,
                resource_requirements=node.resource_requirements.copy()
            )
            optimized_nodes.append(optimized_node)
        
        # Rebuild dependency map and execution levels
        dependency_map = self._build_dependency_map(optimized_nodes)
        execution_levels = self._topological_sort(optimized_nodes, dependency_map)
        critical_path = self._find_critical_path(optimized_nodes, dependency_map)
        total_time = self._calculate_total_time(execution_levels, optimized_nodes)
        
        optimized_graph = DependencyGraph(
            nodes=optimized_nodes,
            execution_levels=execution_levels,
            status="optimized",
            critical_path=critical_path,
            total_estimated_time=total_time
        )
        
        print(f"Dependency graph optimized: {len(execution_levels)} levels, {total_time:.1f}ms total time")
        
        return optimized_graph
    
    def _remove_redundant_dependencies(self, node: TaskNode, node_map: Dict[str, TaskNode]) -> List[str]:
        """Remove redundant transitive dependencies."""
        if not node.dependencies:
            return []
        
        # Find all transitive dependencies
        all_transitive = set()
        for dep_id in node.dependencies:
            if dep_id in node_map:
                transitive = self._get_all_dependencies(dep_id, node_map, set())
                all_transitive.update(transitive)
        
        # Remove dependencies that are already covered transitively
        direct_deps = []
        for dep_id in node.dependencies:
            # Check if this dependency is covered by another direct dependency
            is_redundant = False
            for other_dep in node.dependencies:
                if other_dep != dep_id and other_dep in node_map:
                    other_transitive = self._get_all_dependencies(other_dep, node_map, set())
                    if dep_id in other_transitive:
                        is_redundant = True
                        break
            
            if not is_redundant:
                direct_deps.append(dep_id)
        
        return direct_deps
    
    def _get_all_dependencies(self, task_id: str, node_map: Dict[str, TaskNode], visited: Set[str]) -> Set[str]:
        """Get all transitive dependencies of a task."""
        if task_id in visited or task_id not in node_map:
            return set()
        
        visited.add(task_id)
        all_deps = set()
        
        node = node_map[task_id]
        for dep_id in node.dependencies:
            all_deps.add(dep_id)
            all_deps.update(self._get_all_dependencies(dep_id, node_map, visited))
        
        return all_deps
    
    def get_parallelization_opportunities(self, graph: DependencyGraph) -> Dict[str, Any]:
        """Analyze parallelization opportunities in dependency graph."""
        if not graph.execution_levels:
            return {"max_parallel_tasks": 0, "parallelization_ratio": 0.0}
        
        total_tasks = len(graph.nodes)
        max_parallel_tasks = max(len(level) for level in graph.execution_levels)
        avg_parallel_tasks = sum(len(level) for level in graph.execution_levels) / len(graph.execution_levels)
        
        # Calculate parallelization ratio
        sequential_time = sum(node.estimated_duration for node in graph.nodes)
        parallel_time = graph.total_estimated_time
        parallelization_ratio = (sequential_time - parallel_time) / max(sequential_time, 1.0)
        
        return {
            "total_tasks": total_tasks,
            "execution_levels": len(graph.execution_levels),
            "max_parallel_tasks": max_parallel_tasks,
            "avg_parallel_tasks": avg_parallel_tasks,
            "parallelization_ratio": parallelization_ratio,
            "critical_path_length": len(graph.critical_path),
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup_factor": sequential_time / max(parallel_time, 1.0)
        }

def get_dependency_resolver() -> DependencyResolver:
    """Get dependency resolver instance."""
    return DependencyResolver()