"""
Parallel Executor for TestMaster Flow Optimizer

Creates and executes parallel execution plans for optimized workflow performance.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import asyncio

from core.feature_flags import FeatureFlags
from .dependency_resolver import DependencyGraph
from .resource_optimizer import ResourceAllocation

class ParallelStrategy(Enum):
    """Parallel execution strategies."""
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    RESOURCE_AWARE = "resource_aware"

@dataclass
class TaskBatch:
    """Batch of tasks for parallel execution."""
    batch_id: str
    task_ids: List[str]
    estimated_duration: float
    resource_allocation: Dict[str, float]
    dependencies_satisfied: bool = True

@dataclass
class ExecutionPlan:
    """Parallel execution plan."""
    workflow_id: str
    batches: List[TaskBatch]
    status: str
    total_estimated_time: float = 0.0
    parallelization_factor: float = 1.0
    resource_efficiency: float = 0.0

class ParallelExecutor:
    """Parallel executor for workflow optimization."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'flow_optimizer')
        self.lock = threading.RLock()
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.thread_pool: ThreadPoolExecutor = None
        self.learning_rate = 0.1
        self.strategy_performance: Dict[str, List[float]] = {}
        
        if not self.enabled:
            return
        
        self.thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="flow_optimizer")
        
        print("Parallel executor initialized")
        print(f"   Thread pool workers: 8")
        print(f"   Parallel strategies: {[s.value for s in ParallelStrategy]}")
    
    def configure(self, learning_rate: float = 0.1, max_workers: int = 8):
        """Configure parallel executor."""
        self.learning_rate = learning_rate
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="flow_optimizer")
        print(f"Parallel executor configured: {max_workers} workers, learning rate: {learning_rate}")
    
    def create_execution_plan(
        self,
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
        if not self.enabled:
            return ExecutionPlan(workflow_id, [], "disabled")
        
        try:
            strategy_enum = ParallelStrategy(strategy)
        except ValueError:
            strategy_enum = ParallelStrategy.BALANCED
        
        # Create execution batches based on dependency levels
        batches = self._create_execution_batches(
            dependency_graph,
            resource_allocation,
            strategy_enum
        )
        
        # Calculate plan metrics
        total_time = sum(batch.estimated_duration for batch in batches)
        sequential_time = sum(node.estimated_duration for node in dependency_graph.nodes)
        parallelization_factor = sequential_time / max(total_time, 1.0)
        
        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency(batches, resource_allocation)
        
        plan = ExecutionPlan(
            workflow_id=workflow_id,
            batches=batches,
            status="created",
            total_estimated_time=total_time,
            parallelization_factor=parallelization_factor,
            resource_efficiency=resource_efficiency
        )
        
        # Store plan
        with self.lock:
            self.execution_plans[workflow_id] = plan
        
        print(f"Execution plan created for {workflow_id}: {len(batches)} batches, {parallelization_factor:.2f}x speedup")
        
        return plan
    
    def _create_execution_batches(
        self,
        dependency_graph: DependencyGraph,
        resource_allocation: ResourceAllocation,
        strategy: ParallelStrategy
    ) -> List[TaskBatch]:
        """Create execution batches based on strategy."""
        batches = []
        node_map = {node.task_id: node for node in dependency_graph.nodes}
        
        for level_index, level_tasks in enumerate(dependency_graph.execution_levels):
            if strategy == ParallelStrategy.AGGRESSIVE:
                # Single batch per level for maximum parallelization
                batch = self._create_single_batch(f"batch_{level_index}", level_tasks, node_map, resource_allocation)
                batches.append(batch)
                
            elif strategy == ParallelStrategy.CONSERVATIVE:
                # Multiple smaller batches to reduce resource contention
                batch_size = max(1, len(level_tasks) // 4)
                for i in range(0, len(level_tasks), batch_size):
                    batch_tasks = level_tasks[i:i+batch_size]
                    batch = self._create_single_batch(f"batch_{level_index}_{i//batch_size}", batch_tasks, node_map, resource_allocation)
                    batches.append(batch)
                    
            elif strategy == ParallelStrategy.RESOURCE_AWARE:
                # Create batches based on resource requirements
                batches.extend(self._create_resource_aware_batches(f"level_{level_index}", level_tasks, node_map, resource_allocation))
                
            elif strategy == ParallelStrategy.ADAPTIVE:
                # Use adaptive batching based on historical performance
                batches.extend(self._create_adaptive_batches(f"level_{level_index}", level_tasks, node_map, resource_allocation))
                
            else:  # BALANCED
                # Balanced approach - moderate batch sizes
                batch_size = max(1, min(len(level_tasks), 8))
                for i in range(0, len(level_tasks), batch_size):
                    batch_tasks = level_tasks[i:i+batch_size]
                    batch = self._create_single_batch(f"batch_{level_index}_{i//batch_size}", batch_tasks, node_map, resource_allocation)
                    batches.append(batch)
        
        return batches
    
    def _create_single_batch(
        self,
        batch_id: str,
        task_ids: List[str],
        node_map: Dict[str, Any],
        resource_allocation: ResourceAllocation
    ) -> TaskBatch:
        """Create a single task batch."""
        # Calculate estimated duration (max of all tasks in batch)
        estimated_duration = 0.0
        total_resource_req = {}
        
        for task_id in task_ids:
            if task_id in node_map:
                node = node_map[task_id]
                estimated_duration = max(estimated_duration, node.estimated_duration)
                
                # Aggregate resource requirements
                for res_type, amount in node.resource_requirements.items():
                    total_resource_req[res_type] = total_resource_req.get(res_type, 0.0) + amount
        
        return TaskBatch(
            batch_id=batch_id,
            task_ids=task_ids,
            estimated_duration=estimated_duration,
            resource_allocation=total_resource_req
        )
    
    def _create_resource_aware_batches(
        self,
        level_prefix: str,
        task_ids: List[str],
        node_map: Dict[str, Any],
        resource_allocation: ResourceAllocation
    ) -> List[TaskBatch]:
        """Create batches based on resource requirements."""
        batches = []
        
        # Group tasks by similar resource requirements
        resource_groups = self._group_tasks_by_resources(task_ids, node_map)
        
        batch_index = 0
        for resource_profile, group_tasks in resource_groups.items():
            batch = self._create_single_batch(f"{level_prefix}_res_{batch_index}", group_tasks, node_map, resource_allocation)
            batches.append(batch)
            batch_index += 1
        
        return batches if batches else [self._create_single_batch(f"{level_prefix}_default", task_ids, node_map, resource_allocation)]
    
    def _create_adaptive_batches(
        self,
        level_prefix: str,
        task_ids: List[str],
        node_map: Dict[str, Any],
        resource_allocation: ResourceAllocation
    ) -> List[TaskBatch]:
        """Create batches using adaptive algorithm."""
        # Use historical performance to determine optimal batch size
        avg_performance = self._get_average_strategy_performance()
        
        if avg_performance > 0.8:
            # High performance - use larger batches
            batch_size = min(len(task_ids), 12)
        elif avg_performance > 0.6:
            # Medium performance - use moderate batches
            batch_size = min(len(task_ids), 6)
        else:
            # Low performance - use smaller batches
            batch_size = min(len(task_ids), 3)
        
        batches = []
        for i in range(0, len(task_ids), batch_size):
            batch_tasks = task_ids[i:i+batch_size]
            batch = self._create_single_batch(f"{level_prefix}_adaptive_{i//batch_size}", batch_tasks, node_map, resource_allocation)
            batches.append(batch)
        
        return batches
    
    def _group_tasks_by_resources(self, task_ids: List[str], node_map: Dict[str, Any]) -> Dict[str, List[str]]:
        """Group tasks by similar resource requirements."""
        groups = {}
        
        for task_id in task_ids:
            if task_id not in node_map:
                continue
            
            node = node_map[task_id]
            
            # Create resource profile signature
            profile_parts = []
            for res_type in ["cpu", "memory", "network", "storage"]:
                amount = node.resource_requirements.get(res_type, 0.0)
                # Quantize to reduce profile variations
                quantized = round(amount / 10.0) * 10
                profile_parts.append(f"{res_type}:{quantized}")
            
            profile = ",".join(profile_parts)
            
            if profile not in groups:
                groups[profile] = []
            groups[profile].append(task_id)
        
        return groups
    
    def _calculate_resource_efficiency(self, batches: List[TaskBatch], resource_allocation: ResourceAllocation) -> float:
        """Calculate resource efficiency of execution plan."""
        if not batches:
            return 0.0
        
        # Calculate resource utilization efficiency
        total_allocated = 0.0
        total_required = 0.0
        
        for batch in batches:
            for res_type, amount in batch.resource_allocation.items():
                total_required += amount
        
        for res_type, type_allocations in resource_allocation.allocations.items():
            for pool_id, amount in type_allocations.items():
                total_allocated += amount
        
        # Efficiency is how well we match requirements to allocation
        if total_allocated > 0:
            efficiency = min(1.0, total_required / total_allocated)
        else:
            efficiency = 0.0
        
        return efficiency
    
    def _get_average_strategy_performance(self) -> float:
        """Get average performance across all strategies."""
        all_scores = []
        for scores in self.strategy_performance.values():
            all_scores.extend(scores)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.5
    
    def record_strategy_performance(self, strategy: str, performance_score: float):
        """Record performance for strategy learning."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        
        self.strategy_performance[strategy].append(performance_score)
        
        # Keep only recent performance data
        if len(self.strategy_performance[strategy]) > 50:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-50:]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_plans = len(self.execution_plans)
        total_batches = sum(len(plan.batches) for plan in self.execution_plans.values())
        
        avg_parallelization = 0.0
        avg_efficiency = 0.0
        
        if self.execution_plans:
            avg_parallelization = sum(plan.parallelization_factor for plan in self.execution_plans.values()) / total_plans
            avg_efficiency = sum(plan.resource_efficiency for plan in self.execution_plans.values()) / total_plans
        
        return {
            "total_plans": total_plans,
            "total_batches": total_batches,
            "average_parallelization_factor": avg_parallelization,
            "average_resource_efficiency": avg_efficiency,
            "strategy_performance": dict(self.strategy_performance),
            "thread_pool_active": self.thread_pool is not None
        }
    
    def shutdown(self):
        """Shutdown parallel executor."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        print("Parallel executor shutdown completed")

def get_parallel_executor() -> ParallelExecutor:
    """Get parallel executor instance."""
    return ParallelExecutor()