"""
Resource Intelligence Data Models
=================================

Core data structures and enums for the revolutionary resource allocation system.
Extracted from intelligent_resource_allocator.py for enterprise modular architecture.

Agent D Implementation - Hour 10-11: Revolutionary Intelligence Modularization
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class AllocationStrategy(Enum):
    """Resource allocation strategies for intelligent optimization"""
    FAIR_SHARE = "fair_share"
    PERFORMANCE_BASED = "performance_based"
    DEMAND_BASED = "demand_based"
    PRIORITY_BASED = "priority_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    COST_OPTIMIZED = "cost_optimized"


class ScalingDirection(Enum):
    """Scaling direction options for predictive scaling"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class LoadBalancingAlgorithm(Enum):
    """Advanced load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin" 
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    PERFORMANCE_BASED = "performance_based"
    PREDICTIVE = "predictive"


@dataclass
class ResourceConstraint:
    """Resource constraint definition with elasticity modeling"""
    resource_type: str
    min_allocation: float
    max_allocation: float
    priority: int
    cost_per_unit: float = 1.0
    elasticity: float = 1.0  # How easily this resource can be scaled
    
    def is_satisfied(self, current_allocation: float) -> bool:
        """Check if current allocation satisfies constraint"""
        return self.min_allocation <= current_allocation <= self.max_allocation


@dataclass
class AllocationRequest:
    """Resource allocation request with priority and urgency modeling"""
    request_id: str
    framework_id: str
    resource_requirements: Dict[str, float]
    priority: int
    urgency: float  # 0-1
    duration: timedelta
    deadline: Optional[datetime] = None
    constraints: List[ResourceConstraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            **asdict(self),
            'duration': str(self.duration),
            'deadline': self.deadline.isoformat() if self.deadline else None
        }


@dataclass
class AllocationDecision:
    """Resource allocation decision with confidence scoring"""
    decision_id: str
    request_id: str
    framework_id: str
    allocated_resources: Dict[str, float]
    allocation_strategy: AllocationStrategy
    confidence: float  # 0-1
    cost_estimate: float
    performance_impact: float
    allocation_time: datetime
    expiration_time: datetime
    success: bool = True
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            **asdict(self),
            'allocation_strategy': self.allocation_strategy.value,
            'allocation_time': self.allocation_time.isoformat(),
            'expiration_time': self.expiration_time.isoformat()
        }


@dataclass
class LoadBalancingMetrics:
    """Load balancing performance metrics with health scoring"""
    framework_id: str
    current_load: float  # 0-1
    capacity: float
    utilization: float  # 0-1
    response_time: float
    throughput: float
    error_rate: float
    health_score: float  # 0-1
    weight: float = 1.0  # Load balancing weight
    
    def effective_capacity(self) -> float:
        """Calculate effective capacity considering health"""
        return self.capacity * self.health_score


@dataclass
class PredictiveScalingSignal:
    """Signal for predictive scaling decisions with confidence modeling"""
    resource_type: str
    predicted_demand: float
    current_supply: float
    confidence: float  # 0-1
    time_horizon: timedelta
    trend_direction: ScalingDirection
    urgency: float  # 0-1
    cost_impact: float
    
    def scaling_needed(self) -> bool:
        """Determine if scaling action is needed based on demand ratio and confidence"""
        demand_ratio = self.predicted_demand / self.current_supply if self.current_supply > 0 else float('inf')
        
        if demand_ratio > 1.2 and self.confidence > 0.7:
            return True
        elif demand_ratio < 0.6 and self.confidence > 0.8:
            return True
        
        return False


@dataclass
class ResourceAllocationMetrics:
    """Comprehensive resource allocation performance metrics"""
    total_requests: int = 0
    successful_allocations: int = 0
    failed_allocations: int = 0
    total_resources_allocated: Dict[str, float] = field(default_factory=dict)
    average_response_time: float = 0.0
    average_confidence: float = 0.0
    cost_efficiency: float = 0.0
    prediction_accuracy: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate allocation success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_allocations / self.total_requests
    
    def failure_rate(self) -> float:
        """Calculate allocation failure rate"""
        return 1.0 - self.success_rate()


@dataclass
class ScalingAction:
    """Scaling action with rationale and expected impact"""
    action_id: str
    resource_type: str
    current_allocation: float
    target_allocation: float
    scaling_direction: ScalingDirection
    confidence: float
    rationale: str
    expected_cost_impact: float
    expected_performance_impact: float
    execution_timestamp: datetime
    completion_timestamp: Optional[datetime] = None
    success: bool = True
    actual_impact: Optional[Dict[str, float]] = None