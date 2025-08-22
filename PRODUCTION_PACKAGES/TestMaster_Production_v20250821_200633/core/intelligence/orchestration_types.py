"""
Orchestration Types and Data Structures
=======================================

Core type definitions and data structures for the Intelligence Command Center
orchestration system. Provides enterprise-grade type safety and data modeling.

This module contains all Enum definitions and dataclass structures used throughout
the orchestration system, following enterprise architectural patterns.

Author: Agent A - Hours 30-40
Created: 2025-08-22
Module: orchestration_types.py (150 lines)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class FrameworkType(Enum):
    """Types of intelligence frameworks"""
    ANALYTICS = "analytics"
    ML = "ml"
    API = "api"
    ANALYSIS = "analysis"


class OrchestrationPriority(Enum):
    """Priority levels for orchestration tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class OrchestrationStatus(Enum):
    """Status of orchestration operations"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ResourceType(Enum):
    """Types of resources managed by orchestration"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    THREADS = "threads"
    CONNECTIONS = "connections"


@dataclass
class FrameworkCapability:
    """Represents a capability of an intelligence framework"""
    name: str
    framework_type: FrameworkType
    capability_type: str
    performance_metrics: Dict[str, float]
    resource_requirements: Dict[ResourceType, float]
    availability: float  # 0-1
    load_score: float   # 0-1, current load
    priority: int       # Higher number = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationTask:
    """Represents a task to be orchestrated across frameworks"""
    task_id: str
    task_type: str
    framework_targets: List[FrameworkType]
    priority: OrchestrationPriority
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime = None
    completed_at: datetime = None
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    result: Dict[str, Any] = None
    error_info: str = None


@dataclass
class ResourceAllocation:
    """Represents resource allocation across frameworks"""
    resource_type: ResourceType
    total_available: float
    total_allocated: float
    framework_allocations: Dict[FrameworkType, float]
    utilization_percentage: float
    predicted_demand: float
    scaling_recommendation: str
    
    def remaining_capacity(self) -> float:
        """Calculate remaining resource capacity"""
        return max(0.0, self.total_available - self.total_allocated)
    
    def is_overallocated(self) -> bool:
        """Check if resource is overallocated"""
        return self.total_allocated > self.total_available


@dataclass
class FrameworkHealthStatus:
    """Health status of an intelligence framework"""
    framework_type: FrameworkType
    overall_health_score: float  # 0-1
    performance_metrics: Dict[str, float]
    resource_utilization: Dict[ResourceType, float]
    error_rate: float
    response_time: float
    throughput: float
    availability: float
    last_health_check: datetime
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# Export all public types
__all__ = [
    'FrameworkType',
    'OrchestrationPriority', 
    'OrchestrationStatus',
    'ResourceType',
    'FrameworkCapability',
    'OrchestrationTask',
    'ResourceAllocation',
    'FrameworkHealthStatus'
]