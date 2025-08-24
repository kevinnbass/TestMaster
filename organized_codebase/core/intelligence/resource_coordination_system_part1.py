"""
Resource Coordination System
===========================
"""Core Module - Split from resource_coordination_system.py"""


import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
import threading
from collections import defaultdict, deque
import statistics


logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DATABASE_CONNECTIONS = "database_connections"
    API_RATE_LIMIT = "api_rate_limit"
    THREAD_POOL = "thread_pool"
    CACHE_SPACE = "cache_space"
    FILE_HANDLES = "file_handles"
    CUSTOM = "custom"

class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    FIRST_COME_FIRST_SERVED = "fcfs"
    PRIORITY_BASED = "priority"
    FAIR_SHARE = "fair_share"
    WEIGHTED_FAIR_QUEUING = "wfq"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PROPORTIONAL_SHARE = "proportional"

class ResourceStatus(Enum):
    """Resource allocation status"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    EXHAUSTED = "exhausted"
    OVERCOMMITTED = "overcommitted"
    DEGRADED = "degraded"

class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    PRIORITY_BASED = "priority"
    TEMPORAL_ORDERING = "temporal"
    RESOURCE_SHARING = "sharing"
    PREEMPTION = "preemption"
    ALTERNATIVE_RESOURCE = "alternative"
    DEGRADED_SERVICE = "degraded"

@dataclass
class ResourceDefinition:
    """Resource definition and configuration"""
    resource_id: str
    resource_type: ResourceType
    name: str
    description: str = ""
    total_capacity: float = 100.0
    allocation_unit: str = "units"
    min_allocation: float = 1.0
    max_allocation: float = 100.0
    is_sharable: bool = True
    is_preemptible: bool = False
    cost_per_unit: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate_allocation(self, amount: float) -> bool:
        """Validate allocation amount"""
        return self.min_allocation <= amount <= min(self.max_allocation, self.total_capacity)

@dataclass
class ResourceQuota:
    """Resource quota for tenant/user"""
    tenant_id: str
    resource_type: ResourceType
    max_allocation: float
    current_allocation: float = 0.0
    reserved_allocation: float = 0.0
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    
    def available_quota(self) -> float:
        """Get available quota"""
        return max(0, self.max_allocation - self.current_allocation - self.reserved_allocation)
    
    def utilization_percentage(self) -> float:
        """Get quota utilization percentage"""
        if self.max_allocation == 0:
            return 0.0
        return (self.current_allocation / self.max_allocation) * 100

@dataclass
class ResourceAllocation:
    """Resource allocation record"""
    allocation_id: str = field(default_factory=lambda: f"alloc_{uuid.uuid4().hex[:12]}")
    resource_id: str = ""
    tenant_id: str = ""
    requester_id: str = ""
    amount: float = 1.0
    priority: int = 1
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIRST_COME_FIRST_SERVED
    expires_at: Optional[datetime] = None
    allocated_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if allocation has expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

@dataclass
class ResourceRequest:
    """Resource allocation request"""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    resource_type: ResourceType = ResourceType.CPU
    tenant_id: str = ""
    requester_id: str = ""
    amount: float = 1.0
    duration_seconds: Optional[int] = None
    priority: int = 1
    deadline: Optional[datetime] = None
    can_be_preempted: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_expired(self) -> bool:
        """Check if request has expired"""
        if self.deadline:
            return datetime.now() > self.deadline
        return False

@dataclass
class ResourceConflict:
    """Resource allocation conflict"""
    conflict_id: str = field(default_factory=lambda: f"conflict_{uuid.uuid4().hex[:12]}")
    resource_id: str = ""
    conflicting_requests: List[str] = field(default_factory=list)
    conflict_type: str = "capacity_exceeded"
    total_demand: float = 0.0
    available_capacity: float = 0.0
    detected_at: datetime = field(default_factory=datetime.now)
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class ResourcePool:
    """Manages a pool of specific resources"""
    
    def __init__(self, definition: ResourceDefinition):
        self.definition = definition
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.available_capacity = definition.total_capacity
        self.reserved_capacity = 0.0
        self.allocation_history: deque = deque(maxlen=1000)
        self.usage_stats = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'peak_usage': 0.0,
            'average_utilization': 0.0
        }
    
    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        used_capacity = self.definition.total_capacity - self.available_capacity
        if self.definition.total_capacity == 0:
            return 0.0
        return (used_capacity / self.definition.total_capacity) * 100
    
    def can_allocate(self, amount: float) -> bool:
        """Check if allocation is possible"""
        return (self.available_capacity >= amount and 
                self.definition.validate_allocation(amount))
    
    def allocate(self, allocation: ResourceAllocation) -> bool:
        """Allocate resource"""
        if not self.can_allocate(allocation.amount):
            return False
        
        self.allocations[allocation.allocation_id] = allocation
        self.available_capacity -= allocation.amount
        
        # Update statistics
        self.usage_stats['successful_allocations'] += 1
        current_usage = self.definition.total_capacity - self.available_capacity
        self.usage_stats['peak_usage'] = max(self.usage_stats['peak_usage'], current_usage)
        
        # Record in history
        self.allocation_history.append({
            'action': 'allocated',
            'allocation_id': allocation.allocation_id,
            'amount': allocation.amount,
            'timestamp': datetime.now().isoformat()
        })
        
        return True
    
    def deallocate(self, allocation_id: str) -> bool:
        """Deallocate resource"""
        if allocation_id not in self.allocations:
            return False
        
        allocation = self.allocations[allocation_id]
        del self.allocations[allocation_id]
        self.available_capacity += allocation.amount
        
        # Record in history
        self.allocation_history.append({
            'action': 'deallocated',
            'allocation_id': allocation_id,
            'amount': allocation.amount,
            'timestamp': datetime.now().isoformat()
        })
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status"""
        return {
            'resource_id': self.definition.resource_id,
            'resource_type': self.definition.resource_type.value,
            'total_capacity': self.definition.total_capacity,
            'available_capacity': self.available_capacity,
            'reserved_capacity': self.reserved_capacity,
            'utilization_percentage': self.get_utilization(),
            'active_allocations': len(self.allocations),
            'usage_statistics': self.usage_stats.copy()
        }

class AllocationScheduler:
    """Schedules resource allocations using various strategies"""
    
    def __init__(self):
        self.strategies = {
            AllocationStrategy.FIRST_COME_FIRST_SERVED: self._fcfs_schedule,
            AllocationStrategy.PRIORITY_BASED: self._priority_schedule,
            AllocationStrategy.FAIR_SHARE: self._fair_share_schedule,
            AllocationStrategy.WEIGHTED_FAIR_QUEUING: self._wfq_schedule,
            AllocationStrategy.ROUND_ROBIN: self._round_robin_schedule,
            AllocationStrategy.LEAST_LOADED: self._least_loaded_schedule,
            AllocationStrategy.PROPORTIONAL_SHARE: self._proportional_schedule
        }
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
    
    def schedule(self, requests: List[ResourceRequest], 
                pools: Dict[str, ResourcePool],
                strategy: AllocationStrategy) -> List[Tuple[ResourceRequest, str]]:
        """Schedule requests to resource pools"""
        scheduler_func = self.strategies.get(strategy, self._fcfs_schedule)
        return scheduler_func(requests, pools)
    
    def _fcfs_schedule(self, requests: List[ResourceRequest], 
                      pools: Dict[str, ResourcePool]) -> List[Tuple[ResourceRequest, str]]:
        """First Come First Served scheduling"""
        scheduled = []
        sorted_requests = sorted(requests, key=lambda r: r.created_at)
        
        for request in sorted_requests:
            # Find first available pool of matching type
            for pool_id, pool in pools.items():
                if (pool.definition.resource_type == request.resource_type and
                    pool.can_allocate(request.amount)):
                    scheduled.append((request, pool_id))
                    break
        
        return scheduled
    
    def _priority_schedule(self, requests: List[ResourceRequest],
                          pools: Dict[str, ResourcePool]) -> List[Tuple[ResourceRequest, str]]:
        """Priority-based scheduling"""
        scheduled = []
        sorted_requests = sorted(requests, key=lambda r: (-r.priority, r.created_at))
        
        for request in sorted_requests:
            # Find best available pool
            best_pool = None
            best_pool_id = None
            
            for pool_id, pool in pools.items():
                if (pool.definition.resource_type == request.resource_type and
                    pool.can_allocate(request.amount)):
                    if best_pool is None or pool.get_utilization() < best_pool.get_utilization():
                        best_pool = pool
                        best_pool_id = pool_id
            
            if best_pool_id:
                scheduled.append((request, best_pool_id))
        
        return scheduled
    
    def _fair_share_schedule(self, requests: List[ResourceRequest],
                           pools: Dict[str, ResourcePool]) -> List[Tuple[ResourceRequest, str]]:
        """Fair share scheduling"""
        scheduled = []
        
        # Group requests by tenant
        tenant_requests = defaultdict(list)
        for request in requests:
            tenant_requests[request.tenant_id].append(request)
        
        # Round-robin between tenants
        tenant_queues = list(tenant_requests.values())
        while any(tenant_queues):
            for tenant_queue in tenant_queues[:]:
                if not tenant_queue:
                    tenant_queues.remove(tenant_queue)
                    continue
                
                request = tenant_queue.pop(0)
                
                # Find available pool
                for pool_id, pool in pools.items():
                    if (pool.definition.resource_type == request.resource_type and
                        pool.can_allocate(request.amount)):
                        scheduled.append((request, pool_id))
                        break
        
        return scheduled
    
    def _wfq_schedule(self, requests: List[ResourceRequest],
                     pools: Dict[str, ResourcePool]) -> List[Tuple[ResourceRequest, str]]:
        """Weighted Fair Queuing scheduling"""
        # Similar to fair share but with priority weights
        return self._priority_schedule(requests, pools)
    
    def _round_robin_schedule(self, requests: List[ResourceRequest],
                            pools: Dict[str, ResourcePool]) -> List[Tuple[ResourceRequest, str]]:
        """Round-robin scheduling across pools"""
        scheduled = []
        
        for request in requests:
            # Get pools of matching type
            matching_pools = [
                (pool_id, pool) for pool_id, pool in pools.items()
                if pool.definition.resource_type == request.resource_type
            ]
            
            if not matching_pools:
                continue
            
            # Round-robin selection
            pool_type = request.resource_type.value
            counter = self.round_robin_counters[pool_type]
            
            for i in range(len(matching_pools)):
                pool_index = (counter + i) % len(matching_pools)
                pool_id, pool = matching_pools[pool_index]
                
                if pool.can_allocate(request.amount):
                    scheduled.append((request, pool_id))
                    self.round_robin_counters[pool_type] = (pool_index + 1) % len(matching_pools)
                    break
        
        return scheduled
    
    def _least_loaded_schedule(self, requests: List[ResourceRequest],
                             pools: Dict[str, ResourcePool]) -> List[Tuple[ResourceRequest, str]]:
        """Least loaded scheduling"""
        scheduled = []
        
        for request in requests:
            # Find least loaded pool of matching type
            best_pool = None
            best_pool_id = None
            lowest_utilization = float('inf')
            
            for pool_id, pool in pools.items():
                if (pool.definition.resource_type == request.resource_type and
                    pool.can_allocate(request.amount)):
                    utilization = pool.get_utilization()
                    if utilization < lowest_utilization:
                        lowest_utilization = utilization
                        best_pool = pool
                        best_pool_id = pool_id
            
            if best_pool_id:
                scheduled.append((request, best_pool_id))
        
        return scheduled
    
    def _proportional_schedule(self, requests: List[ResourceRequest],
                             pools: Dict[str, ResourcePool]) -> List[Tuple[ResourceRequest, str]]:
        """Proportional share scheduling"""
        # Similar to priority-based but considers request amounts
        scheduled = []
        sorted_requests = sorted(requests, key=lambda r: (r.amount / r.priority, r.created_at))
        
        for request in sorted_requests:
            for pool_id, pool in pools.items():
                if (pool.definition.resource_type == request.resource_type and
                    pool.can_allocate(request.amount)):
                    scheduled.append((request, pool_id))
                    break
        
        return scheduled

class ConflictResolver:
    """Resolves resource allocation conflicts"""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictResolutionStrategy.PRIORITY_BASED: self._resolve_by_priority,
            ConflictResolutionStrategy.TEMPORAL_ORDERING: self._resolve_by_time,
            ConflictResolutionStrategy.RESOURCE_SHARING: self._resolve_by_sharing,
            ConflictResolutionStrategy.PREEMPTION: self._resolve_by_preemption,
            ConflictResolutionStrategy.ALTERNATIVE_RESOURCE: self._resolve_alternative,
            ConflictResolutionStrategy.DEGRADED_SERVICE: self._resolve_degraded
        }
    
    def resolve_conflict(self, conflict: ResourceConflict, 
                        requests: List[ResourceRequest],
                        pools: Dict[str, ResourcePool]) -> List[ResourceRequest]:
        """Resolve resource conflict"""
        strategy = conflict.resolution_strategy or ConflictResolutionStrategy.PRIORITY_BASED
        resolver_func = self.resolution_strategies.get(strategy, self._resolve_by_priority)
        
        return resolver_func(conflict, requests, pools)
    
    def _resolve_by_priority(self, conflict: ResourceConflict,
                           requests: List[ResourceRequest],
                           pools: Dict[str, ResourcePool]) -> List[ResourceRequest]:
        """Resolve by priority ordering"""
        # Sort by priority and allocate in order until capacity is exhausted
        conflicting_requests = [r for r in requests if r.request_id in conflict.conflicting_requests]
        sorted_requests = sorted(conflicting_requests, key=lambda r: (-r.priority, r.created_at))
        
        resolved = []
        remaining_capacity = conflict.available_capacity
        
        for request in sorted_requests:
            if request.amount <= remaining_capacity:
                resolved.append(request)
                remaining_capacity -= request.amount
        
        return resolved
    
    def _resolve_by_time(self, conflict: ResourceConflict,
                        requests: List[ResourceRequest],
                        pools: Dict[str, ResourcePool]) -> List[ResourceRequest]:
        """Resolve by temporal ordering (FCFS)"""
        conflicting_requests = [r for r in requests if r.request_id in conflict.conflicting_requests]
        sorted_requests = sorted(conflicting_requests, key=lambda r: r.created_at)
        
        resolved = []
        remaining_capacity = conflict.available_capacity
        
        for request in sorted_requests:
            if request.amount <= remaining_capacity:
                resolved.append(request)
                remaining_capacity -= request.amount
        
        return resolved
    
    def _resolve_by_sharing(self, conflict: ResourceConflict,
                          requests: List[ResourceRequest],
                          pools: Dict[str, ResourcePool]) -> List[ResourceRequest]:
        """Resolve by proportional sharing"""
        conflicting_requests = [r for r in requests if r.request_id in conflict.conflicting_requests]
        
        # Calculate proportional shares
        total_demand = sum(r.amount for r in conflicting_requests)
        share_factor = conflict.available_capacity / total_demand
        
        resolved = []
        for request in conflicting_requests:
            # Reduce request amount proportionally
            new_amount = request.amount * share_factor
            if new_amount >= 1.0:  # Minimum allocation
                request.amount = new_amount
                resolved.append(request)
        
        return resolved
    
    def _resolve_by_preemption(self, conflict: ResourceConflict,
                             requests: List[ResourceRequest],
                             pools: Dict[str, ResourcePool]) -> List[ResourceRequest]:
        """Resolve by preempting lower priority allocations"""
        # Find pool and existing allocations
        pool = None
        for p in pools.values():
            if p.definition.resource_id == conflict.resource_id:
                pool = p
                break
        
        if not pool:
            return []
        
        conflicting_requests = [r for r in requests if r.request_id in conflict.conflicting_requests]
        high_priority_requests = [r for r in conflicting_requests if r.priority >= 5]
        
        # Try to preempt existing allocations
        preemptable_allocations = [
            alloc for alloc in pool.allocations.values()
            if any(req.priority > 3 for req in high_priority_requests)  # Can preempt lower priority
        ]
