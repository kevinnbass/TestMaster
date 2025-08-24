"""
Resource Optimizer for TestMaster Flow Optimizer

Optimizes resource allocation for efficient workflow execution.
"""

import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import heapq

from core.feature_flags import FeatureFlags

class ResourceType(Enum):
    """Types of resources."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    GPU = "gpu"
    THREAD_POOL = "thread_pool"

class OptimizationPolicy(Enum):
    """Resource optimization policies."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    BALANCED = "balanced"
    ENERGY_EFFICIENT = "energy_efficient"

@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    resource_type: ResourceType
    amount: float
    priority: int
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}

@dataclass
class ResourcePool:
    """Resource pool information."""
    pool_id: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    cost_per_unit: float
    performance_rating: float

@dataclass
class ResourceAllocation:
    """Resource allocation result."""
    workflow_id: str
    allocations: Dict[str, Dict[str, float]]
    status: str
    total_cost: float = 0.0
    efficiency_score: float = 0.0
    constraints_satisfied: bool = True

class ResourceOptimizer:
    """Resource optimizer for workflow execution."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'flow_optimizer')
        self.lock = threading.RLock()
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.allocation_history: Dict[str, List[ResourceAllocation]] = {}
        self.optimization_policies: Dict[str, OptimizationPolicy] = {}
        
        if not self.enabled:
            return
        
        self._initialize_default_pools()
        
        print("Resource optimizer initialized")
        print(f"   Resource types: {[rt.value for rt in ResourceType]}")
        print(f"   Default pools: {len(self.resource_pools)}")
    
    def _initialize_default_pools(self):
        """Initialize default resource pools."""
        default_pools = [
            ResourcePool("cpu_pool_1", ResourceType.CPU, 100.0, 80.0, 0.01, 0.9),
            ResourcePool("memory_pool_1", ResourceType.MEMORY, 1000.0, 750.0, 0.005, 0.85),
            ResourcePool("network_pool_1", ResourceType.NETWORK, 1000.0, 900.0, 0.002, 0.95),
            ResourcePool("storage_pool_1", ResourceType.STORAGE, 10000.0, 8000.0, 0.001, 0.8),
            ResourcePool("thread_pool_1", ResourceType.THREAD_POOL, 50.0, 35.0, 0.02, 0.9),
        ]
        
        for pool in default_pools:
            self.resource_pools[pool.pool_id] = pool
    
    def optimize_allocation(
        self,
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
        if not self.enabled:
            return ResourceAllocation(workflow_id, {}, "disabled")
        
        constraints = constraints or {}
        
        # Extract resource requirements from tasks
        requirements = self._extract_requirements(tasks)
        
        # Select optimization policy
        policy = self._select_optimization_policy(workflow_id, constraints)
        
        # Generate allocation plan
        allocations = self._generate_allocation_plan(requirements, policy, constraints)
        
        # Calculate metrics
        total_cost = self._calculate_total_cost(allocations)
        efficiency_score = self._calculate_efficiency_score(allocations, requirements)
        constraints_satisfied = self._check_constraints(allocations, constraints)
        
        allocation = ResourceAllocation(
            workflow_id=workflow_id,
            allocations=allocations,
            status="optimized",
            total_cost=total_cost,
            efficiency_score=efficiency_score,
            constraints_satisfied=constraints_satisfied
        )
        
        # Store in history
        with self.lock:
            if workflow_id not in self.allocation_history:
                self.allocation_history[workflow_id] = []
            self.allocation_history[workflow_id].append(allocation)
        
        print(f"Resource allocation optimized for {workflow_id}: {efficiency_score:.3f} efficiency, cost: {total_cost:.2f}")
        
        return allocation
    
    def _extract_requirements(self, tasks: List[Dict[str, Any]]) -> List[ResourceRequirement]:
        """Extract resource requirements from tasks."""
        requirements = []
        
        for task in tasks:
            # Extract CPU requirements
            cpu_req = task.get('cpu_requirement', 1.0)
            if cpu_req > 0:
                requirements.append(ResourceRequirement(
                    resource_type=ResourceType.CPU,
                    amount=cpu_req,
                    priority=task.get('priority', 5),
                    constraints=task.get('cpu_constraints', {})
                ))
            
            # Extract memory requirements
            memory_req = task.get('memory_requirement', 100.0)
            if memory_req > 0:
                requirements.append(ResourceRequirement(
                    resource_type=ResourceType.MEMORY,
                    amount=memory_req,
                    priority=task.get('priority', 5),
                    constraints=task.get('memory_constraints', {})
                ))
            
            # Extract network requirements
            network_req = task.get('network_requirement', 10.0)
            if network_req > 0:
                requirements.append(ResourceRequirement(
                    resource_type=ResourceType.NETWORK,
                    amount=network_req,
                    priority=task.get('priority', 5),
                    constraints=task.get('network_constraints', {})
                ))
        
        return requirements
    
    def _select_optimization_policy(self, workflow_id: str, constraints: Dict[str, Any]) -> OptimizationPolicy:
        """Select optimization policy based on constraints and history."""
        # Check for explicit policy in constraints
        policy_name = constraints.get('optimization_policy')
        if policy_name:
            try:
                return OptimizationPolicy(policy_name)
            except ValueError:
                pass
        
        # Use stored policy for workflow
        if workflow_id in self.optimization_policies:
            return self.optimization_policies[workflow_id]
        
        # Default policy selection logic
        budget_constraint = constraints.get('max_cost')
        performance_requirement = constraints.get('min_performance')
        
        if budget_constraint and not performance_requirement:
            return OptimizationPolicy.MINIMIZE_COST
        elif performance_requirement and not budget_constraint:
            return OptimizationPolicy.MAXIMIZE_PERFORMANCE
        elif constraints.get('energy_efficient', False):
            return OptimizationPolicy.ENERGY_EFFICIENT
        else:
            return OptimizationPolicy.BALANCED
    
    def _generate_allocation_plan(
        self,
        requirements: List[ResourceRequirement],
        policy: OptimizationPolicy,
        constraints: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Generate resource allocation plan based on policy."""
        allocations = {}
        
        # Group requirements by resource type
        grouped_requirements = {}
        for req in requirements:
            res_type = req.resource_type.value
            if res_type not in grouped_requirements:
                grouped_requirements[res_type] = []
            grouped_requirements[res_type].append(req)
        
        # Allocate resources for each type
        for resource_type, reqs in grouped_requirements.items():
            type_allocations = self._allocate_resource_type(resource_type, reqs, policy, constraints)
            allocations[resource_type] = type_allocations
        
        return allocations
    
    def _allocate_resource_type(
        self,
        resource_type: str,
        requirements: List[ResourceRequirement],
        policy: OptimizationPolicy,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Allocate resources for a specific resource type."""
        # Get available pools for this resource type
        available_pools = [
            pool for pool in self.resource_pools.values()
            if pool.resource_type.value == resource_type and pool.available_capacity > 0
        ]
        
        if not available_pools:
            return {}
        
        # Sort requirements by priority
        sorted_requirements = sorted(requirements, key=lambda r: r.priority, reverse=True)
        
        # Sort pools based on optimization policy
        if policy == OptimizationPolicy.MINIMIZE_COST:
            available_pools.sort(key=lambda p: p.cost_per_unit)
        elif policy == OptimizationPolicy.MAXIMIZE_PERFORMANCE:
            available_pools.sort(key=lambda p: p.performance_rating, reverse=True)
        elif policy == OptimizationPolicy.ENERGY_EFFICIENT:
            available_pools.sort(key=lambda p: p.cost_per_unit * p.performance_rating)
        else:  # BALANCED
            available_pools.sort(key=lambda p: (p.cost_per_unit + (1.0 - p.performance_rating)) / 2)
        
        allocations = {}
        
        # Allocate resources using greedy approach
        for requirement in sorted_requirements:
            remaining_amount = requirement.amount
            
            for pool in available_pools:
                if remaining_amount <= 0:
                    break
                
                # Calculate how much can be allocated from this pool
                allocatable = min(remaining_amount, pool.available_capacity)
                
                if allocatable > 0:
                    if pool.pool_id not in allocations:
                        allocations[pool.pool_id] = 0.0
                    
                    allocations[pool.pool_id] += allocatable
                    pool.available_capacity -= allocatable
                    remaining_amount -= allocatable
        
        return allocations
    
    def _calculate_total_cost(self, allocations: Dict[str, Dict[str, float]]) -> float:
        """Calculate total cost of resource allocation."""
        total_cost = 0.0
        
        for resource_type, type_allocations in allocations.items():
            for pool_id, amount in type_allocations.items():
                pool = self.resource_pools.get(pool_id)
                if pool:
                    total_cost += amount * pool.cost_per_unit
        
        return total_cost
    
    def _calculate_efficiency_score(
        self,
        allocations: Dict[str, Dict[str, float]],
        requirements: List[ResourceRequirement]
    ) -> float:
        """Calculate efficiency score of resource allocation."""
        # Calculate satisfaction ratio
        total_required = sum(req.amount for req in requirements)
        total_allocated = sum(
            sum(type_allocations.values()) for type_allocations in allocations.values()
        )
        
        satisfaction_ratio = min(1.0, total_allocated / max(total_required, 0.1))
        
        # Calculate performance-weighted efficiency
        performance_score = 0.0
        total_allocation = 0.0
        
        for resource_type, type_allocations in allocations.items():
            for pool_id, amount in type_allocations.items():
                pool = self.resource_pools.get(pool_id)
                if pool:
                    performance_score += amount * pool.performance_rating
                    total_allocation += amount
        
        if total_allocation > 0:
            avg_performance = performance_score / total_allocation
        else:
            avg_performance = 0.0
        
        # Combine satisfaction and performance
        efficiency_score = (satisfaction_ratio * 0.6) + (avg_performance * 0.4)
        
        return efficiency_score
    
    def _check_constraints(self, allocations: Dict[str, Dict[str, float]], constraints: Dict[str, Any]) -> bool:
        """Check if allocation satisfies constraints."""
        # Check cost constraint
        max_cost = constraints.get('max_cost')
        if max_cost is not None:
            total_cost = self._calculate_total_cost(allocations)
            if total_cost > max_cost:
                return False
        
        # Check resource limits
        resource_limits = constraints.get('resource_limits', {})
        for resource_type, limit in resource_limits.items():
            type_allocations = allocations.get(resource_type, {})
            total_allocated = sum(type_allocations.values())
            if total_allocated > limit:
                return False
        
        return True
    
    def add_resource_pool(self, pool: ResourcePool):
        """Add a new resource pool."""
        self.resource_pools[pool.pool_id] = pool
        print(f"Resource pool added: {pool.pool_id} ({pool.resource_type.value})")
    
    def update_pool_capacity(self, pool_id: str, new_capacity: float):
        """Update available capacity of a resource pool."""
        if pool_id in self.resource_pools:
            self.resource_pools[pool_id].available_capacity = new_capacity
            print(f"Pool capacity updated: {pool_id} -> {new_capacity}")
    
    def set_optimization_policy(self, workflow_id: str, policy: OptimizationPolicy):
        """Set optimization policy for a workflow."""
        self.optimization_policies[workflow_id] = policy
        print(f"Optimization policy set for {workflow_id}: {policy.value}")
    
    def get_allocation_history(self, workflow_id: str) -> List[ResourceAllocation]:
        """Get allocation history for a workflow."""
        with self.lock:
            return self.allocation_history.get(workflow_id, [])
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization statistics."""
        utilization = {}
        
        for pool_id, pool in self.resource_pools.items():
            utilization_ratio = (pool.total_capacity - pool.available_capacity) / pool.total_capacity
            utilization[pool_id] = {
                "resource_type": pool.resource_type.value,
                "utilization_ratio": utilization_ratio,
                "available_capacity": pool.available_capacity,
                "total_capacity": pool.total_capacity
            }
        
        return utilization
    
    def optimize_pool_sizes(self, historical_demand: Dict[str, List[float]]):
        """Optimize pool sizes based on historical demand."""
        for resource_type, demands in historical_demand.items():
            if not demands:
                continue
            
            # Calculate recommended capacity based on demand patterns
            avg_demand = sum(demands) / len(demands)
            peak_demand = max(demands)
            recommended_capacity = avg_demand * 1.2 + (peak_demand - avg_demand) * 0.5
            
            # Update pools of this resource type
            for pool in self.resource_pools.values():
                if pool.resource_type.value == resource_type:
                    old_capacity = pool.total_capacity
                    pool.total_capacity = max(pool.total_capacity, recommended_capacity)
                    pool.available_capacity += (pool.total_capacity - old_capacity)
                    
                    if pool.total_capacity != old_capacity:
                        print(f"Pool {pool.pool_id} capacity optimized: {old_capacity:.1f} -> {pool.total_capacity:.1f}")

def get_resource_optimizer() -> ResourceOptimizer:
    """Get resource optimizer instance."""
    return ResourceOptimizer()