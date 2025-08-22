"""
Resource Allocation Types - Advanced Resource Management Type Definitions
=========================================================================

Comprehensive type definitions and data structures for intelligent resource allocation,
load balancing, and predictive scaling with enterprise-grade optimization patterns.
Implements advanced resource management types for enterprise intelligence systems.

This module provides all type definitions, enums, and dataclasses required for
sophisticated resource allocation and optimization across intelligence frameworks.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: resource_allocation_types.py (200 lines)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum


class AllocationStrategy(Enum):
    """Advanced resource allocation strategies with optimization patterns"""
    FAIR_SHARE = "fair_share"
    PERFORMANCE_BASED = "performance_based"
    DEMAND_BASED = "demand_based"
    PRIORITY_BASED = "priority_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    COST_OPTIMIZED = "cost_optimized"
    ML_OPTIMIZED = "ml_optimized"
    MULTI_OBJECTIVE = "multi_objective"


class ScalingDirection(Enum):
    """Scaling direction options with enterprise patterns"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    AUTO_SCALE = "auto_scale"
    PREDICTIVE_SCALE = "predictive_scale"


class LoadBalancingAlgorithm(Enum):
    """Advanced load balancing algorithms for enterprise systems"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin" 
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    PERFORMANCE_BASED = "performance_based"
    PREDICTIVE = "predictive"
    MACHINE_LEARNING = "machine_learning"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"


class ResourceType(Enum):
    """Resource types for comprehensive allocation management"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    BANDWIDTH = "bandwidth"
    THREADS = "threads"
    CONNECTIONS = "connections"
    COMPUTE_UNITS = "compute_units"
    CUSTOM = "custom"


class AllocationPriority(Enum):
    """Allocation priority levels with enterprise hierarchy"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class OptimizationObjective(Enum):
    """Optimization objectives for resource allocation"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class ResourceConstraint:
    """Advanced resource constraint definition with enterprise validation"""
    resource_type: str
    min_allocation: float
    max_allocation: float
    priority: int
    cost_per_unit: float = 1.0
    elasticity: float = 1.0  # How easily this resource can be scaled
    constraint_type: str = "hard"  # hard, soft, or preference
    penalty_factor: float = 1.0  # Penalty for constraint violations
    
    def is_satisfied(self, current_allocation: float) -> bool:
        """Check if current allocation satisfies constraint"""
        return self.min_allocation <= current_allocation <= self.max_allocation
    
    def calculate_violation_penalty(self, current_allocation: float) -> float:
        """Calculate penalty for constraint violations"""
        if self.is_satisfied(current_allocation):
            return 0.0
        
        if current_allocation < self.min_allocation:
            violation = self.min_allocation - current_allocation
        else:
            violation = current_allocation - self.max_allocation
        
        return violation * self.penalty_factor


@dataclass
class AllocationRequest:
    """Comprehensive resource allocation request with enterprise features"""
    request_id: str
    framework_id: str
    resource_requirements: Dict[str, float]
    priority: AllocationPriority
    urgency: float  # 0-1
    duration: timedelta
    deadline: Optional[datetime] = None
    constraints: List[ResourceConstraint] = field(default_factory=list)
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED
    max_cost: Optional[float] = None
    preferred_strategy: Optional[AllocationStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation with serialization"""
        return {
            **asdict(self),
            'priority': self.priority.value,
            'optimization_objective': self.optimization_objective.value,
            'preferred_strategy': self.preferred_strategy.value if self.preferred_strategy else None,
            'duration': str(self.duration),
            'deadline': self.deadline.isoformat() if self.deadline else None
        }
    
    def calculate_urgency_score(self) -> float:
        """Calculate comprehensive urgency score"""
        base_urgency = self.urgency
        
        # Priority factor
        priority_weights = {
            AllocationPriority.CRITICAL: 1.0,
            AllocationPriority.HIGH: 0.8,
            AllocationPriority.NORMAL: 0.6,
            AllocationPriority.LOW: 0.4,
            AllocationPriority.BACKGROUND: 0.2
        }
        priority_factor = priority_weights.get(self.priority, 0.6)
        
        # Deadline factor
        deadline_factor = 1.0
        if self.deadline:
            time_remaining = (self.deadline - datetime.now()).total_seconds()
            if time_remaining > 0:
                deadline_factor = max(0.1, 1.0 - (time_remaining / 3600))  # Increase urgency as deadline approaches
        
        return min(1.0, base_urgency * priority_factor * deadline_factor)


@dataclass
class AllocationDecision:
    """Comprehensive resource allocation decision with enterprise tracking"""
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
    optimization_score: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    alternative_decisions: List['AllocationDecision'] = field(default_factory=list)
    success: bool = True
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation with comprehensive serialization"""
        return {
            **asdict(self),
            'allocation_strategy': self.allocation_strategy.value,
            'allocation_time': self.allocation_time.isoformat(),
            'expiration_time': self.expiration_time.isoformat(),
            'alternative_decisions': [decision.to_dict() for decision in self.alternative_decisions]
        }
    
    def is_expired(self) -> bool:
        """Check if allocation decision has expired"""
        return datetime.now() > self.expiration_time
    
    def calculate_efficiency_score(self) -> float:
        """Calculate allocation efficiency score"""
        if not self.success:
            return 0.0
        
        # Combine confidence, performance impact, and cost efficiency
        cost_efficiency = max(0.0, 1.0 - min(1.0, self.cost_estimate / 100.0))
        performance_score = max(0.0, self.performance_impact)
        
        return (self.confidence * 0.4 + performance_score * 0.4 + cost_efficiency * 0.2)


@dataclass
class LoadBalancingMetrics:
    """Advanced load balancing performance metrics with enterprise monitoring"""
    framework_id: str
    current_load: float  # 0-1
    capacity: float
    utilization: float  # 0-1
    response_time: float
    throughput: float
    error_rate: float
    health_score: float  # 0-1
    weight: float = 1.0  # Load balancing weight
    queue_length: int = 0
    active_connections: int = 0
    resource_efficiency: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def effective_capacity(self) -> float:
        """Calculate effective capacity considering health and efficiency"""
        return self.capacity * self.health_score * self.resource_efficiency
    
    def calculate_load_score(self) -> float:
        """Calculate comprehensive load score for balancing decisions"""
        # Combine multiple factors for load scoring
        load_factor = 1.0 - self.current_load
        health_factor = self.health_score
        efficiency_factor = self.resource_efficiency
        response_factor = max(0.0, 1.0 - min(1.0, self.response_time / 1000.0))
        
        return (load_factor * 0.3 + health_factor * 0.3 + 
                efficiency_factor * 0.2 + response_factor * 0.2)


@dataclass
class PredictiveScalingSignal:
    """Advanced predictive scaling signal with ML-enhanced predictions"""
    resource_type: str
    predicted_demand: float
    current_supply: float
    confidence: float  # 0-1
    time_horizon: timedelta
    trend_direction: ScalingDirection
    urgency: float  # 0-1
    cost_impact: float
    prediction_model: str = "heuristic"
    historical_accuracy: float = 0.0
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    external_factors: Dict[str, float] = field(default_factory=dict)
    
    def scaling_needed(self, threshold_up: float = 1.2, threshold_down: float = 0.6) -> bool:
        """Determine if scaling action is needed with configurable thresholds"""
        if self.current_supply <= 0:
            return True
        
        demand_ratio = self.predicted_demand / self.current_supply
        confidence_threshold = 0.7 if self.trend_direction == ScalingDirection.SCALE_UP else 0.8
        
        if demand_ratio > threshold_up and self.confidence > confidence_threshold:
            return True
        elif demand_ratio < threshold_down and self.confidence > confidence_threshold:
            return True
        
        return False
    
    def calculate_scaling_magnitude(self) -> float:
        """Calculate recommended scaling magnitude based on prediction"""
        if not self.scaling_needed():
            return 0.0
        
        demand_ratio = self.predicted_demand / self.current_supply if self.current_supply > 0 else 2.0
        confidence_factor = self.confidence
        urgency_factor = self.urgency
        
        # Base scaling magnitude
        if demand_ratio > 1.0:
            base_magnitude = (demand_ratio - 1.0) * 0.5  # Scale up
        else:
            base_magnitude = (demand_ratio - 1.0) * 0.3  # Scale down (more conservative)
        
        # Apply modifying factors
        adjusted_magnitude = base_magnitude * confidence_factor * (1.0 + urgency_factor * 0.5)
        
        # Limit scaling magnitude to reasonable bounds
        return max(-0.5, min(1.0, adjusted_magnitude))


@dataclass
class OptimizationResult:
    """Comprehensive optimization result with detailed analysis"""
    optimization_id: str
    objective_value: float
    solution: Dict[str, Dict[str, float]]
    convergence_status: str
    iterations: int
    execution_time: float
    constraint_satisfaction: Dict[str, bool]
    alternative_solutions: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourcePool:
    """Resource pool definition for advanced allocation management"""
    pool_id: str
    pool_name: str
    resource_types: Set[str]
    total_capacity: Dict[str, float]
    available_capacity: Dict[str, float]
    reserved_capacity: Dict[str, float]
    pool_priority: int
    access_policies: List[str] = field(default_factory=list)
    usage_constraints: List[ResourceConstraint] = field(default_factory=list)
    
    def get_utilization(self, resource_type: str) -> float:
        """Calculate utilization for a specific resource type"""
        total = self.total_capacity.get(resource_type, 0.0)
        available = self.available_capacity.get(resource_type, 0.0)
        
        if total <= 0:
            return 0.0
        
        utilized = total - available
        return utilized / total


# Export all types and enums
__all__ = [
    'AllocationStrategy', 'ScalingDirection', 'LoadBalancingAlgorithm',
    'ResourceType', 'AllocationPriority', 'OptimizationObjective',
    'ResourceConstraint', 'AllocationRequest', 'AllocationDecision',
    'LoadBalancingMetrics', 'PredictiveScalingSignal', 'OptimizationResult',
    'ResourcePool'
]