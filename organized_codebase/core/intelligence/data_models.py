"""
Resource Allocation Data Models
================================

Core data models and enums for the resource allocation system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    FAIR_SHARE = "fair_share"
    PERFORMANCE_BASED = "performance_based"
    DEMAND_BASED = "demand_based"
    PRIORITY_BASED = "priority_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    COST_OPTIMIZED = "cost_optimized"


class ScalingDirection(Enum):
    """Scaling direction options"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin" 
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    PERFORMANCE_BASED = "performance_based"
    PREDICTIVE = "predictive"


@dataclass
class ResourceConstraint:
    """Resource constraint definition"""
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
    """Resource allocation request"""
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
    """Resource allocation decision"""
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
    """Load balancing performance metrics"""
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
    """Signal for predictive scaling decisions"""
    resource_type: str
    predicted_demand: float
    current_supply: float
    confidence: float  # 0-1
    time_horizon: timedelta
    trend_direction: ScalingDirection
    urgency: float  # 0-1
    cost_impact: float
    
    def scaling_needed(self) -> bool:
        """Determine if scaling action is needed"""
        demand_ratio = self.predicted_demand / self.current_supply if self.current_supply > 0 else float('inf')
        
        if demand_ratio > 1.2 and self.confidence > 0.7:
            return True
        elif demand_ratio < 0.6 and self.confidence > 0.8:
            return True
        
        return False


__all__ = [
    'AllocationStrategy',
    'ScalingDirection',
    'LoadBalancingAlgorithm',
    'ResourceConstraint',
    'AllocationRequest',
    'AllocationDecision',
    'LoadBalancingMetrics',
    'PredictiveScalingSignal'
]