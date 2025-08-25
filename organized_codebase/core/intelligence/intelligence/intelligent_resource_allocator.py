"""
Intelligent Resource Allocator
==============================

Advanced resource allocation and load balancing system that dynamically
optimizes resources across all intelligence frameworks with predictive scaling.

Agent A - Hour 22-24: Intelligence Orchestration & Coordination
Intelligent resource management with autonomous optimization capabilities.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import json
import threading
from abc import ABC, abstractmethod

# Advanced optimization imports
try:
    from scipy.optimize import minimize, differential_evolution, linprog
    from scipy.stats import norm
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_ADVANCED_OPTIMIZATION = True
except ImportError:
    HAS_ADVANCED_OPTIMIZATION = False
    logging.warning("Advanced optimization libraries not available. Using simplified optimization.")


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


class ResourceOptimizer(ABC):
    """Abstract base class for resource optimization algorithms"""
    
    @abstractmethod
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize resource allocation across requests"""
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization algorithm information"""
        pass


class LinearProgrammingOptimizer(ResourceOptimizer):
    """Linear programming based resource optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize using linear programming"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not allocation_requests:
                return self._fallback_optimization(available_resources, allocation_requests)
            
            # Prepare optimization problem
            resource_types = list(available_resources.keys())
            n_requests = len(allocation_requests)
            n_resources = len(resource_types)
            
            # Objective function coefficients (maximize priority-weighted satisfaction)
            c = []
            for request in allocation_requests:
                for resource_type in resource_types:
                    priority_weight = request.priority / 10.0
                    urgency_weight = request.urgency
                    c.append(-(priority_weight * urgency_weight))  # Negative for maximization
            
            # Constraint matrix (resource capacity constraints)
            A_ub = []
            b_ub = []
            
            # Resource capacity constraints
            for i, resource_type in enumerate(resource_types):
                constraint_row = [0.0] * (n_requests * n_resources)
                for j in range(n_requests):
                    constraint_row[j * n_resources + i] = 1.0
                A_ub.append(constraint_row)
                b_ub.append(available_resources[resource_type])
            
            # Request requirement constraints (simplified)
            A_eq = []
            b_eq = []
            
            # Bounds for variables (non-negative allocations)
            bounds = []
            for request in allocation_requests:
                for resource_type in resource_types:
                    required = request.resource_requirements.get(resource_type, 0.0)
                    max_useful = required * 2.0  # Allow up to 2x requested amount
                    bounds.append((0.0, max_useful))
            
            # Solve linear program
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                # Extract allocation decisions
                allocations = {}
                for i, request in enumerate(allocation_requests):
                    request_allocations = {}
                    for j, resource_type in enumerate(resource_types):
                        idx = i * n_resources + j
                        request_allocations[resource_type] = float(result.x[idx])
                    allocations[request.request_id] = request_allocations
                
                return allocations
            else:
                self.logger.warning("Linear programming optimization failed, using fallback")
                return self._fallback_optimization(available_resources, allocation_requests)
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return self._fallback_optimization(available_resources, allocation_requests)
    
    def _fallback_optimization(self, 
                             available_resources: Dict[str, float],
                             allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Fallback optimization using simple proportional allocation"""
        allocations = {}
        
        if not allocation_requests:
            return allocations
        
        # Sort requests by priority and urgency
        sorted_requests = sorted(
            allocation_requests, 
            key=lambda r: (r.priority, r.urgency), 
            reverse=True
        )
        
        # Track remaining resources
        remaining = dict(available_resources)
        
        for request in sorted_requests:
            request_allocations = {}
            
            for resource_type, required_amount in request.resource_requirements.items():
                available_amount = remaining.get(resource_type, 0.0)
                allocated_amount = min(required_amount, available_amount)
                
                request_allocations[resource_type] = allocated_amount
                remaining[resource_type] = max(0.0, available_amount - allocated_amount)
            
            allocations[request.request_id] = request_allocations
        
        return allocations
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization algorithm information"""
        return {
            'algorithm': 'LinearProgrammingOptimizer',
            'method': 'highs' if HAS_ADVANCED_OPTIMIZATION else 'fallback_proportional',
            'supports_constraints': True,
            'optimization_type': 'exact'
        }


class GeneticAlgorithmOptimizer(ResourceOptimizer):
    """Genetic algorithm based resource optimizer for complex scenarios"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize using genetic algorithm"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not allocation_requests:
                return self._simple_allocation(available_resources, allocation_requests)
            
            # Define the optimization problem
            def objective_function(allocation_vector):
                return self._evaluate_allocation_fitness(
                    allocation_vector, available_resources, allocation_requests, constraints
                )
            
            # Define bounds for the optimization variables
            bounds = []
            for request in allocation_requests:
                for resource_type in available_resources.keys():
                    max_allocation = available_resources[resource_type]
                    bounds.append((0.0, max_allocation))
            
            # Run genetic algorithm optimization
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=self.generations,
                popsize=15,  # Smaller population for speed
                seed=42,
                atol=1e-6,
                tol=1e-6
            )
            
            if result.success:
                # Convert result back to allocation dictionary
                allocations = self._vector_to_allocations(
                    result.x, available_resources, allocation_requests
                )
                return allocations
            else:
                self.logger.warning("Genetic algorithm optimization failed, using simple allocation")
                return self._simple_allocation(available_resources, allocation_requests)
                
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization error: {e}")
            return self._simple_allocation(available_resources, allocation_requests)
    
    def _evaluate_allocation_fitness(self, 
                                   allocation_vector: np.ndarray,
                                   available_resources: Dict[str, float],
                                   allocation_requests: List[AllocationRequest],
                                   constraints: List[ResourceConstraint]) -> float:
        """Evaluate fitness of an allocation solution"""
        try:
            allocations = self._vector_to_allocations(
                allocation_vector, available_resources, allocation_requests
            )
            
            total_fitness = 0.0
            
            # Resource utilization efficiency
            resource_efficiency = self._calculate_resource_efficiency(allocations, available_resources)
            total_fitness += resource_efficiency * 0.3
            
            # Request satisfaction
            request_satisfaction = self._calculate_request_satisfaction(allocations, allocation_requests)
            total_fitness += request_satisfaction * 0.4
            
            # Priority weighting
            priority_satisfaction = self._calculate_priority_satisfaction(allocations, allocation_requests)
            total_fitness += priority_satisfaction * 0.2
            
            # Constraint satisfaction penalty
            constraint_penalty = self._calculate_constraint_penalty(allocations, constraints)
            total_fitness -= constraint_penalty * 0.1
            
            return -total_fitness  # Negative because differential_evolution minimizes
            
        except Exception as e:
            self.logger.error(f"Fitness evaluation error: {e}")
            return float('inf')  # Bad fitness for invalid solutions
    
    def _vector_to_allocations(self, 
                             allocation_vector: np.ndarray,
                             available_resources: Dict[str, float],
                             allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Convert optimization vector back to allocation dictionary"""
        allocations = {}
        resource_types = list(available_resources.keys())
        n_resources = len(resource_types)
        
        for i, request in enumerate(allocation_requests):
            request_allocations = {}
            for j, resource_type in enumerate(resource_types):
                idx = i * n_resources + j
                if idx < len(allocation_vector):
                    request_allocations[resource_type] = max(0.0, float(allocation_vector[idx]))
                else:
                    request_allocations[resource_type] = 0.0
            allocations[request.request_id] = request_allocations
        
        return allocations
    
    def _calculate_resource_efficiency(self, 
                                     allocations: Dict[str, Dict[str, float]],
                                     available_resources: Dict[str, float]) -> float:
        """Calculate resource utilization efficiency"""
        if not available_resources:
            return 0.0
        
        total_efficiency = 0.0
        
        for resource_type, available in available_resources.items():
            if available <= 0:
                continue
            
            total_allocated = sum(
                request_allocations.get(resource_type, 0.0)
                for request_allocations in allocations.values()
            )
            
            utilization = min(total_allocated / available, 1.0)
            
            # Efficiency function - good utilization without over-allocation
            if utilization <= 0.8:
                efficiency = utilization
            elif utilization <= 1.0:
                efficiency = 0.8 - (utilization - 0.8) * 2  # Penalty for over-utilization
            else:
                efficiency = -1.0  # Heavy penalty for exceeding capacity
            
            total_efficiency += efficiency
        
        return total_efficiency / len(available_resources)
    
    def _calculate_request_satisfaction(self, 
                                      allocations: Dict[str, Dict[str, float]],
                                      allocation_requests: List[AllocationRequest]) -> float:
        """Calculate how well requests are satisfied"""
        if not allocation_requests:
            return 1.0
        
        total_satisfaction = 0.0
        
        for request in allocation_requests:
            request_allocations = allocations.get(request.request_id, {})
            request_satisfaction = 0.0
            resource_count = 0
            
            for resource_type, required in request.resource_requirements.items():
                if required > 0:
                    allocated = request_allocations.get(resource_type, 0.0)
                    satisfaction_ratio = min(allocated / required, 1.0)
                    request_satisfaction += satisfaction_ratio
                    resource_count += 1
            
            if resource_count > 0:
                request_satisfaction /= resource_count
            
            total_satisfaction += request_satisfaction
        
        return total_satisfaction / len(allocation_requests)
    
    def _calculate_priority_satisfaction(self, 
                                       allocations: Dict[str, Dict[str, float]],
                                       allocation_requests: List[AllocationRequest]) -> float:
        """Calculate priority-weighted satisfaction"""
        if not allocation_requests:
            return 1.0
        
        total_weighted_satisfaction = 0.0
        total_weight = 0.0
        
        for request in allocation_requests:
            request_allocations = allocations.get(request.request_id, {})
            weight = request.priority * request.urgency
            
            satisfaction = 0.0
            for resource_type, required in request.resource_requirements.items():
                if required > 0:
                    allocated = request_allocations.get(resource_type, 0.0)
                    satisfaction += min(allocated / required, 1.0)
            
            if request.resource_requirements:
                satisfaction /= len(request.resource_requirements)
            
            total_weighted_satisfaction += weight * satisfaction
            total_weight += weight
        
        return total_weighted_satisfaction / total_weight if total_weight > 0 else 0.0
    
    def _calculate_constraint_penalty(self, 
                                    allocations: Dict[str, Dict[str, float]],
                                    constraints: List[ResourceConstraint]) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0.0
        
        for constraint in constraints:
            total_allocation = sum(
                request_allocations.get(constraint.resource_type, 0.0)
                for request_allocations in allocations.values()
            )
            
            if total_allocation < constraint.min_allocation:
                penalty += (constraint.min_allocation - total_allocation) / constraint.min_allocation
            elif total_allocation > constraint.max_allocation:
                penalty += (total_allocation - constraint.max_allocation) / constraint.max_allocation
        
        return penalty
    
    def _simple_allocation(self, 
                         available_resources: Dict[str, float],
                         allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Simple proportional allocation fallback"""
        return LinearProgrammingOptimizer()._fallback_optimization(
            available_resources, allocation_requests
        )
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization algorithm information"""
        return {
            'algorithm': 'GeneticAlgorithmOptimizer',
            'population_size': self.population_size,
            'generations': self.generations,
            'supports_constraints': True,
            'optimization_type': 'heuristic'
        }


class IntelligentResourceAllocator:
    """
    Advanced resource allocation and load balancing system that dynamically
    optimizes resources across all intelligence frameworks with predictive scaling.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Intelligent Resource Allocator"""
        self.config = config or self._get_default_config()
        
        # Core components
        self.allocation_requests: deque = deque(maxlen=1000)
        self.allocation_decisions: deque = deque(maxlen=1000)
        self.active_allocations: Dict[str, AllocationDecision] = {}
        
        # Resource tracking
        self.available_resources: Dict[str, float] = {}
        self.resource_constraints: List[ResourceConstraint] = []
        self.resource_utilization_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Load balancing
        self.framework_metrics: Dict[str, LoadBalancingMetrics] = {}
        self.load_balancing_weights: Dict[str, float] = {}
        
        # Predictive scaling
        self.scaling_signals: deque = deque(maxlen=50)
        self.demand_predictions: Dict[str, float] = {}
        
        # Optimization engines
        self.optimizers: Dict[str, ResourceOptimizer] = {
            'linear_programming': LinearProgrammingOptimizer(),
            'genetic_algorithm': GeneticAlgorithmOptimizer()
        }
        
        # Performance tracking
        self.allocation_metrics = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_allocation_time': 0.0,
            'resource_efficiency': 0.0,
            'load_balancing_efficiency': 0.0,
            'prediction_accuracy': 0.0
        }
        
        # Background tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Initialize components
        self._initialize_resources()
        self._initialize_constraints()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'default_allocation_strategy': AllocationStrategy.ADAPTIVE,
            'load_balancing_algorithm': LoadBalancingAlgorithm.PERFORMANCE_BASED,
            'enable_predictive_scaling': True,
            'scaling_threshold_up': 0.8,
            'scaling_threshold_down': 0.3,
            'allocation_timeout': timedelta(seconds=30),
            'rebalancing_interval': timedelta(minutes=5),
            'prediction_horizon': timedelta(hours=1),
            'resource_safety_margin': 0.1,  # 10% safety margin
            'max_allocation_attempts': 3,
            'enable_autonomous_scaling': True,
            'cost_optimization_enabled': True,
            'log_level': logging.INFO
        }
    
    def _initialize_resources(self) -> None:
        """Initialize available resources"""
        # This would be detected from actual system in production
        self.available_resources = {
            'cpu': 100.0,        # 100 CPU units
            'memory': 64.0,      # 64 GB memory
            'storage': 1000.0,   # 1 TB storage
            'network': 10.0,     # 10 Gbps network
            'gpu': 8.0,          # 8 GPU units
            'threads': 128.0,    # 128 threads
            'connections': 10000.0  # 10k connections
        }
    
    def _initialize_constraints(self) -> None:
        """Initialize resource constraints"""
        self.resource_constraints = [
            ResourceConstraint(
                resource_type='cpu',
                min_allocation=10.0,   # Always keep 10% CPU available
                max_allocation=90.0,   # Never exceed 90% CPU
                priority=10,
                cost_per_unit=0.05,
                elasticity=0.9
            ),
            ResourceConstraint(
                resource_type='memory',
                min_allocation=8.0,    # Keep 8GB available
                max_allocation=56.0,   # Max 56GB allocated
                priority=9,
                cost_per_unit=0.02,
                elasticity=0.8
            ),
            ResourceConstraint(
                resource_type='storage',
                min_allocation=100.0,  # Keep 100GB available
                max_allocation=900.0,  # Max 900GB allocated
                priority=7,
                cost_per_unit=0.001,
                elasticity=0.95
            ),
            ResourceConstraint(
                resource_type='gpu',
                min_allocation=1.0,    # Keep 1 GPU available
                max_allocation=7.0,    # Max 7 GPUs allocated
                priority=8,
                cost_per_unit=0.20,
                elasticity=0.6
            )
        ]
    
    def _setup_logging(self) -> None:
        """Setup logging for the allocator"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config['log_level'])
    
    async def start(self) -> None:
        """Start the resource allocator"""
        if self._running:
            self.logger.warning("Resource allocator is already running")
            return
        
        self._running = True
        self.logger.info("Starting Intelligent Resource Allocator")
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._allocation_processing_loop()),
            asyncio.create_task(self._load_balancing_loop()),
            asyncio.create_task(self._predictive_scaling_loop()),
            asyncio.create_task(self._resource_monitoring_loop())
        ]
        
        self.logger.info("Intelligent Resource Allocator started successfully")
    
    async def stop(self) -> None:
        """Stop the resource allocator"""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping Intelligent Resource Allocator")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Intelligent Resource Allocator stopped")
    
    async def request_allocation(self, 
                               framework_id: str,
                               resource_requirements: Dict[str, float],
                               priority: int = 5,
                               urgency: float = 0.5,
                               duration: timedelta = None,
                               deadline: datetime = None,
                               strategy: AllocationStrategy = None) -> AllocationDecision:
        """Request resource allocation"""
        
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        request = AllocationRequest(
            request_id=request_id,
            framework_id=framework_id,
            resource_requirements=resource_requirements,
            priority=priority,
            urgency=urgency,
            duration=duration or timedelta(hours=1),
            deadline=deadline
        )
        
        self.allocation_requests.append(request)
        self.allocation_metrics['total_requests'] += 1
        
        self.logger.info(f"Allocation request {request_id} from {framework_id}: {resource_requirements}")
        
        # Process allocation immediately for high urgency requests
        if urgency >= 0.8 or priority >= 8:
            return await self._process_allocation_request(request, strategy)
        else:
            # Queue for batch processing
            return await self._create_pending_decision(request)
    
    async def _process_allocation_request(self, 
                                        request: AllocationRequest,
                                        strategy: AllocationStrategy = None) -> AllocationDecision:
        """Process a single allocation request"""
        try:
            start_time = datetime.now()
            
            # Determine allocation strategy
            if strategy is None:
                strategy = self._select_allocation_strategy(request)
            
            # Check resource availability
            if not self._check_resource_availability(request.resource_requirements):
                return self._create_failed_decision(request, "Insufficient resources available")
            
            # Optimize allocation
            optimizer = self._select_optimizer(request, strategy)
            allocation_result = optimizer.optimize_allocation(
                self.available_resources,
                [request],
                self.resource_constraints
            )
            
            allocated_resources = allocation_result.get(request.request_id, {})
            
            # Validate allocation
            if not self._validate_allocation(allocated_resources, request):
                return self._create_failed_decision(request, "Allocation validation failed")
            
            # Calculate allocation metrics
            confidence = self._calculate_allocation_confidence(allocated_resources, request)
            cost_estimate = self._calculate_allocation_cost(allocated_resources)
            performance_impact = self._estimate_performance_impact(allocated_resources, request)
            
            # Create successful allocation decision
            decision = AllocationDecision(
                decision_id=f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                request_id=request.request_id,
                framework_id=request.framework_id,
                allocated_resources=allocated_resources,
                allocation_strategy=strategy,
                confidence=confidence,
                cost_estimate=cost_estimate,
                performance_impact=performance_impact,
                allocation_time=start_time,
                expiration_time=start_time + request.duration,
                success=True
            )
            
            # Apply allocation
            await self._apply_allocation(decision)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_allocation_metrics(decision, processing_time, True)
            
            self.logger.info(f"Allocation successful: {request.request_id} -> {allocated_resources}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Allocation processing failed for {request.request_id}: {e}")
            return self._create_failed_decision(request, str(e))
    
    def _select_allocation_strategy(self, request: AllocationRequest) -> AllocationStrategy:
        """Select the best allocation strategy for a request"""
        # High priority/urgency -> Performance-based
        if request.priority >= 8 or request.urgency >= 0.8:
            return AllocationStrategy.PERFORMANCE_BASED
        
        # Long duration -> Cost optimized
        if request.duration > timedelta(hours=4):
            return AllocationStrategy.COST_OPTIMIZED
        
        # Large resource requirements -> Predictive
        total_resource_demand = sum(request.resource_requirements.values())
        if total_resource_demand > 50.0:
            return AllocationStrategy.PREDICTIVE
        
        # Default to adaptive
        return self.config['default_allocation_strategy']
    
    def _select_optimizer(self, request: AllocationRequest, strategy: AllocationStrategy) -> ResourceOptimizer:
        """Select the best optimizer for the request and strategy"""
        # For complex scenarios, use genetic algorithm
        if (strategy == AllocationStrategy.PREDICTIVE or 
            len(request.resource_requirements) > 4 or
            len(self.resource_constraints) > 6):
            return self.optimizers['genetic_algorithm']
        
        # Default to linear programming
        return self.optimizers['linear_programming']
    
    def _check_resource_availability(self, requirements: Dict[str, float]) -> bool:
        """Check if required resources are available"""
        for resource_type, required_amount in requirements.items():
            available = self.available_resources.get(resource_type, 0.0)
            
            # Check against constraints
            for constraint in self.resource_constraints:
                if constraint.resource_type == resource_type:
                    max_allocatable = constraint.max_allocation
                    current_allocated = self._get_current_allocation(resource_type)
                    if current_allocated + required_amount > max_allocatable:
                        return False
            
            # Check absolute availability
            if available < required_amount:
                return False
        
        return True
    
    def _get_current_allocation(self, resource_type: str) -> float:
        """Get current allocation for a resource type"""
        total_allocated = 0.0
        for decision in self.active_allocations.values():
            allocated_amount = decision.allocated_resources.get(resource_type, 0.0)
            total_allocated += allocated_amount
        return total_allocated
    
    def _validate_allocation(self, allocated_resources: Dict[str, float], request: AllocationRequest) -> bool:
        """Validate that allocation meets minimum requirements"""
        for resource_type, required_amount in request.resource_requirements.items():
            allocated_amount = allocated_resources.get(resource_type, 0.0)
            
            # Must allocate at least 50% of requested amount
            if allocated_amount < required_amount * 0.5:
                return False
        
        return True
    
    def _calculate_allocation_confidence(self, allocated_resources: Dict[str, float], request: AllocationRequest) -> float:
        """Calculate confidence score for allocation"""
        confidence_scores = []
        
        for resource_type, required_amount in request.resource_requirements.items():
            allocated_amount = allocated_resources.get(resource_type, 0.0)
            if required_amount > 0:
                satisfaction_ratio = allocated_amount / required_amount
                confidence_scores.append(min(satisfaction_ratio, 1.0))
            else:
                confidence_scores.append(1.0)
        
        if not confidence_scores:
            return 0.5
        
        # Overall confidence is the average satisfaction
        base_confidence = np.mean(confidence_scores)
        
        # Adjust for resource availability
        availability_factor = self._calculate_resource_availability_factor()
        
        # Adjust for system load
        load_factor = self._calculate_system_load_factor()
        
        return min(1.0, base_confidence * availability_factor * load_factor)
    
    def _calculate_resource_availability_factor(self) -> float:
        """Calculate factor based on overall resource availability"""
        availability_ratios = []
        
        for resource_type, available in self.available_resources.items():
            current_allocated = self._get_current_allocation(resource_type)
            if available > 0:
                utilization = current_allocated / available
                availability_ratios.append(1.0 - utilization)
        
        return np.mean(availability_ratios) if availability_ratios else 0.5
    
    def _calculate_system_load_factor(self) -> float:
        """Calculate factor based on overall system load"""
        if not self.framework_metrics:
            return 1.0
        
        load_scores = [metrics.current_load for metrics in self.framework_metrics.values()]
        avg_load = np.mean(load_scores) if load_scores else 0.5
        
        # Convert load to availability factor (high load = low availability)
        return max(0.1, 1.0 - avg_load)
    
    def _calculate_allocation_cost(self, allocated_resources: Dict[str, float]) -> float:
        """Calculate estimated cost for allocation"""
        total_cost = 0.0
        
        for resource_type, allocated_amount in allocated_resources.items():
            # Find cost per unit from constraints
            cost_per_unit = 0.01  # Default cost
            for constraint in self.resource_constraints:
                if constraint.resource_type == resource_type:
                    cost_per_unit = constraint.cost_per_unit
                    break
            
            total_cost += allocated_amount * cost_per_unit
        
        return total_cost
    
    def _estimate_performance_impact(self, allocated_resources: Dict[str, float], request: AllocationRequest) -> float:
        """Estimate performance impact of allocation"""
        # Simplified performance impact estimation
        impact_factors = []
        
        for resource_type, allocated_amount in allocated_resources.items():
            required_amount = request.resource_requirements.get(resource_type, 0.0)
            
            if required_amount > 0:
                allocation_ratio = allocated_amount / required_amount
                
                if allocation_ratio >= 1.0:
                    impact_factors.append(1.0)  # Positive impact
                elif allocation_ratio >= 0.8:
                    impact_factors.append(0.8)  # Neutral impact
                else:
                    impact_factors.append(allocation_ratio * 0.5)  # Negative impact
        
        return np.mean(impact_factors) if impact_factors else 0.5
    
    async def _apply_allocation(self, decision: AllocationDecision) -> None:
        """Apply allocation decision"""
        # Update available resources
        for resource_type, allocated_amount in decision.allocated_resources.items():
            current_available = self.available_resources.get(resource_type, 0.0)
            self.available_resources[resource_type] = max(0.0, current_available - allocated_amount)
        
        # Store active allocation
        self.active_allocations[decision.decision_id] = decision
        
        # Record decision
        self.allocation_decisions.append(decision)
        
        # Update utilization history
        for resource_type, allocated_amount in decision.allocated_resources.items():
            self.resource_utilization_history[resource_type].append({
                'timestamp': decision.allocation_time,
                'allocation': allocated_amount,
                'decision_id': decision.decision_id
            })
    
    def _create_pending_decision(self, request: AllocationRequest) -> AllocationDecision:
        """Create a pending allocation decision"""
        return AllocationDecision(
            decision_id=f"pending_{request.request_id}",
            request_id=request.request_id,
            framework_id=request.framework_id,
            allocated_resources={},
            allocation_strategy=AllocationStrategy.FAIR_SHARE,
            confidence=0.5,
            cost_estimate=0.0,
            performance_impact=0.0,
            allocation_time=datetime.now(),
            expiration_time=datetime.now() + request.duration,
            success=False,
            failure_reason="Queued for processing"
        )
    
    def _create_failed_decision(self, request: AllocationRequest, reason: str) -> AllocationDecision:
        """Create a failed allocation decision"""
        decision = AllocationDecision(
            decision_id=f"failed_{request.request_id}",
            request_id=request.request_id,
            framework_id=request.framework_id,
            allocated_resources={},
            allocation_strategy=AllocationStrategy.FAIR_SHARE,
            confidence=0.0,
            cost_estimate=0.0,
            performance_impact=0.0,
            allocation_time=datetime.now(),
            expiration_time=datetime.now(),
            success=False,
            failure_reason=reason
        )
        
        self.allocation_decisions.append(decision)
        self._update_allocation_metrics(decision, 0.0, False)
        
        return decision
    
    def _update_allocation_metrics(self, decision: AllocationDecision, processing_time: float, success: bool) -> None:
        """Update allocation performance metrics"""
        if success:
            self.allocation_metrics['successful_allocations'] += 1
        else:
            self.allocation_metrics['failed_allocations'] += 1
        
        # Update average processing time
        total_requests = self.allocation_metrics['total_requests']
        current_avg = self.allocation_metrics['average_allocation_time']
        
        if total_requests > 0:
            self.allocation_metrics['average_allocation_time'] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
        
        # Update resource efficiency
        self._update_resource_efficiency()
    
    def _update_resource_efficiency(self) -> None:
        """Update resource efficiency metrics"""
        if not self.available_resources:
            return
        
        efficiency_scores = []
        
        for resource_type, total_available in self.available_resources.items():
            if total_available <= 0:
                continue
            
            currently_allocated = self._get_current_allocation(resource_type)
            original_available = total_available + currently_allocated
            
            if original_available > 0:
                utilization = currently_allocated / original_available
                
                # Efficiency score based on utilization (optimal around 70-80%)
                if 0.7 <= utilization <= 0.8:
                    efficiency = 1.0
                elif utilization < 0.7:
                    efficiency = utilization / 0.7
                else:
                    efficiency = max(0.0, 1.0 - (utilization - 0.8) / 0.2)
                
                efficiency_scores.append(efficiency)
        
        if efficiency_scores:
            self.allocation_metrics['resource_efficiency'] = np.mean(efficiency_scores)
    
    async def _allocation_processing_loop(self) -> None:
        """Background loop for processing allocation requests"""
        self.logger.info("Starting allocation processing loop")
        
        while self._running:
            try:
                # Process pending requests in batches
                await self._process_allocation_batch()
                
                # Clean up expired allocations
                await self._cleanup_expired_allocations()
                
                # Update metrics
                self._update_resource_efficiency()
                
                await asyncio.sleep(1.0)  # Process frequently
                
            except Exception as e:
                self.logger.error(f"Allocation processing loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_allocation_batch(self) -> None:
        """Process a batch of allocation requests"""
        batch_size = 10
        requests_to_process = []
        
        # Collect requests for batch processing
        for _ in range(min(batch_size, len(self.allocation_requests))):
            if self.allocation_requests:
                request = self.allocation_requests.popleft()
                # Skip high-priority requests (already processed)
                if request.urgency < 0.8 and request.priority < 8:
                    requests_to_process.append(request)
        
        if not requests_to_process:
            return
        
        # Batch optimize allocations
        try:
            optimizer = self.optimizers['linear_programming']  # Use LP for batch processing
            batch_result = optimizer.optimize_allocation(
                self.available_resources,
                requests_to_process,
                self.resource_constraints
            )
            
            # Create decisions for each request
            for request in requests_to_process:
                allocated_resources = batch_result.get(request.request_id, {})
                
                if allocated_resources and self._validate_allocation(allocated_resources, request):
                    # Create successful decision
                    decision = AllocationDecision(
                        decision_id=f"batch_{request.request_id}",
                        request_id=request.request_id,
                        framework_id=request.framework_id,
                        allocated_resources=allocated_resources,
                        allocation_strategy=AllocationStrategy.FAIR_SHARE,
                        confidence=self._calculate_allocation_confidence(allocated_resources, request),
                        cost_estimate=self._calculate_allocation_cost(allocated_resources),
                        performance_impact=self._estimate_performance_impact(allocated_resources, request),
                        allocation_time=datetime.now(),
                        expiration_time=datetime.now() + request.duration,
                        success=True
                    )
                    
                    await self._apply_allocation(decision)
                    self._update_allocation_metrics(decision, 0.1, True)
                else:
                    # Create failed decision
                    self._create_failed_decision(request, "Batch allocation failed")
                    
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Create failed decisions for all requests
            for request in requests_to_process:
                self._create_failed_decision(request, f"Batch processing error: {e}")
    
    async def _cleanup_expired_allocations(self) -> None:
        """Clean up expired allocations"""
        current_time = datetime.now()
        expired_decisions = []
        
        for decision_id, decision in self.active_allocations.items():
            if current_time >= decision.expiration_time:
                expired_decisions.append(decision_id)
        
        # Release resources from expired allocations
        for decision_id in expired_decisions:
            decision = self.active_allocations[decision_id]
            await self._release_allocation(decision)
            del self.active_allocations[decision_id]
            
            self.logger.info(f"Released expired allocation: {decision_id}")
    
    async def _release_allocation(self, decision: AllocationDecision) -> None:
        """Release resources from an allocation"""
        # Return resources to available pool
        for resource_type, allocated_amount in decision.allocated_resources.items():
            current_available = self.available_resources.get(resource_type, 0.0)
            self.available_resources[resource_type] = current_available + allocated_amount
    
    async def _load_balancing_loop(self) -> None:
        """Background loop for load balancing optimization"""
        self.logger.info("Starting load balancing loop")
        
        while self._running:
            try:
                await self._update_framework_metrics()
                await self._optimize_load_balancing()
                await self._update_load_balancing_weights()
                
                await asyncio.sleep(self.config['rebalancing_interval'].total_seconds())
                
            except Exception as e:
                self.logger.error(f"Load balancing loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _update_framework_metrics(self) -> None:
        """Update metrics for all frameworks"""
        # This would collect real metrics from frameworks in production
        # For now, simulate metrics
        
        frameworks = ['analytics', 'ml', 'api', 'analysis']
        
        for framework_id in frameworks:
            # Simulate metrics
            self.framework_metrics[framework_id] = LoadBalancingMetrics(
                framework_id=framework_id,
                current_load=np.random.uniform(0.2, 0.9),
                capacity=100.0,
                utilization=np.random.uniform(0.3, 0.8),
                response_time=np.random.uniform(0.05, 0.3),
                throughput=np.random.uniform(50, 200),
                error_rate=np.random.uniform(0.001, 0.05),
                health_score=np.random.uniform(0.8, 1.0)
            )
    
    async def _optimize_load_balancing(self) -> None:
        """Optimize load balancing across frameworks"""
        if not self.framework_metrics:
            return
        
        algorithm = self.config['load_balancing_algorithm']
        
        if algorithm == LoadBalancingAlgorithm.PERFORMANCE_BASED:
            await self._performance_based_load_balancing()
        elif algorithm == LoadBalancingAlgorithm.PREDICTIVE:
            await self._predictive_load_balancing()
        else:
            await self._weighted_load_balancing()
    
    async def _performance_based_load_balancing(self) -> None:
        """Performance-based load balancing optimization"""
        # Calculate performance scores
        performance_scores = {}
        
        for framework_id, metrics in self.framework_metrics.items():
            # Performance score based on multiple factors
            load_factor = 1.0 - metrics.current_load
            response_factor = 1.0 / (1.0 + metrics.response_time)
            error_factor = 1.0 - metrics.error_rate
            health_factor = metrics.health_score
            
            performance_score = (
                load_factor * 0.3 +
                response_factor * 0.3 +
                error_factor * 0.2 +
                health_factor * 0.2
            )
            
            performance_scores[framework_id] = performance_score
        
        # Normalize scores to weights
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for framework_id, score in performance_scores.items():
                self.load_balancing_weights[framework_id] = score / total_score
        else:
            # Equal weights if no valid scores
            equal_weight = 1.0 / len(performance_scores)
            for framework_id in performance_scores:
                self.load_balancing_weights[framework_id] = equal_weight
    
    async def _predictive_load_balancing(self) -> None:
        """Predictive load balancing using forecasted demand"""
        # This would integrate with predictive models in production
        # For now, use simple trend-based prediction
        
        predicted_loads = {}
        
        for framework_id, metrics in self.framework_metrics.items():
            # Simple prediction based on current trend
            current_load = metrics.current_load
            predicted_change = np.random.uniform(-0.1, 0.1)  # Simulate prediction
            predicted_load = max(0.0, min(1.0, current_load + predicted_change))
            predicted_loads[framework_id] = predicted_load
        
        # Adjust weights based on predicted loads
        for framework_id, predicted_load in predicted_loads.items():
            # Lower weight for frameworks with predicted high load
            weight = max(0.1, 1.0 - predicted_load)
            self.load_balancing_weights[framework_id] = weight
        
        # Normalize weights
        total_weight = sum(self.load_balancing_weights.values())
        if total_weight > 0:
            for framework_id in self.load_balancing_weights:
                self.load_balancing_weights[framework_id] /= total_weight
    
    async def _weighted_load_balancing(self) -> None:
        """Weighted load balancing based on capacity and health"""
        for framework_id, metrics in self.framework_metrics.items():
            # Weight based on effective capacity
            effective_capacity = metrics.effective_capacity()
            weight = effective_capacity / 100.0  # Normalize to 0-1
            
            # Adjust for current utilization
            utilization_factor = max(0.1, 1.0 - metrics.utilization)
            weight *= utilization_factor
            
            self.load_balancing_weights[framework_id] = weight
        
        # Normalize weights
        total_weight = sum(self.load_balancing_weights.values())
        if total_weight > 0:
            for framework_id in self.load_balancing_weights:
                self.load_balancing_weights[framework_id] /= total_weight
    
    async def _update_load_balancing_weights(self) -> None:
        """Update load balancing weights and calculate efficiency"""
        if not self.load_balancing_weights:
            return
        
        # Calculate load balancing efficiency
        weight_variance = np.var(list(self.load_balancing_weights.values()))
        max_variance = 0.25  # Maximum possible variance for uniform distribution
        
        # Efficiency is inverse of variance (lower variance = better balance)
        efficiency = max(0.0, 1.0 - (weight_variance / max_variance))
        self.allocation_metrics['load_balancing_efficiency'] = efficiency
    
    async def _predictive_scaling_loop(self) -> None:
        """Background loop for predictive scaling"""
        if not self.config['enable_predictive_scaling']:
            return
        
        self.logger.info("Starting predictive scaling loop")
        
        while self._running:
            try:
                await self._generate_scaling_signals()
                await self._process_scaling_signals()
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Predictive scaling loop error: {e}")
                await asyncio.sleep(120.0)
    
    async def _generate_scaling_signals(self) -> None:
        """Generate predictive scaling signals"""
        for resource_type, current_available in self.available_resources.items():
            # Calculate current utilization
            current_allocated = self._get_current_allocation(resource_type)
            total_capacity = current_available + current_allocated
            
            if total_capacity <= 0:
                continue
            
            current_utilization = current_allocated / total_capacity
            
            # Predict future demand (simplified)
            predicted_demand = await self._predict_resource_demand(resource_type)
            predicted_utilization = predicted_demand / total_capacity
            
            # Generate scaling signal
            if predicted_utilization > self.config['scaling_threshold_up']:
                trend_direction = ScalingDirection.SCALE_UP
                urgency = min(1.0, (predicted_utilization - self.config['scaling_threshold_up']) / 0.2)
            elif predicted_utilization < self.config['scaling_threshold_down']:
                trend_direction = ScalingDirection.SCALE_DOWN
                urgency = min(1.0, (self.config['scaling_threshold_down'] - predicted_utilization) / 0.2)
            else:
                trend_direction = ScalingDirection.MAINTAIN
                urgency = 0.0
            
            if trend_direction != ScalingDirection.MAINTAIN:
                signal = PredictiveScalingSignal(
                    resource_type=resource_type,
                    predicted_demand=predicted_demand,
                    current_supply=total_capacity,
                    confidence=0.7,  # Would be from actual prediction model
                    time_horizon=self.config['prediction_horizon'],
                    trend_direction=trend_direction,
                    urgency=urgency,
                    cost_impact=self._estimate_scaling_cost(resource_type, trend_direction)
                )
                
                self.scaling_signals.append(signal)
    
    async def _predict_resource_demand(self, resource_type: str) -> float:
        """Predict future resource demand"""
        # This would use actual predictive models in production
        # For now, use simple trend-based prediction
        
        history = self.resource_utilization_history.get(resource_type, deque())
        if len(history) < 5:
            # Not enough history, use current allocation
            return self._get_current_allocation(resource_type)
        
        # Simple linear trend prediction
        recent_allocations = [entry['allocation'] for entry in list(history)[-10:]]
        if len(recent_allocations) >= 2:
            # Calculate trend
            trend = (recent_allocations[-1] - recent_allocations[0]) / len(recent_allocations)
            predicted_demand = recent_allocations[-1] + trend * 5  # 5 steps ahead
            return max(0.0, predicted_demand)
        
        return recent_allocations[-1] if recent_allocations else 0.0
    
    def _estimate_scaling_cost(self, resource_type: str, direction: ScalingDirection) -> float:
        """Estimate cost of scaling operation"""
        # Find cost per unit
        cost_per_unit = 0.01  # Default
        for constraint in self.resource_constraints:
            if constraint.resource_type == resource_type:
                cost_per_unit = constraint.cost_per_unit
                break
        
        # Estimate scaling amount (10% of current capacity)
        current_capacity = self.available_resources.get(resource_type, 0.0)
        current_allocated = self._get_current_allocation(resource_type)
        total_capacity = current_capacity + current_allocated
        
        scaling_amount = total_capacity * 0.1
        
        if direction == ScalingDirection.SCALE_UP:
            return scaling_amount * cost_per_unit
        elif direction == ScalingDirection.SCALE_DOWN:
            return -scaling_amount * cost_per_unit * 0.5  # Savings (but with some cost)
        else:
            return 0.0
    
    async def _process_scaling_signals(self) -> None:
        """Process scaling signals and take autonomous actions"""
        if not self.config['enable_autonomous_scaling']:
            return
        
        # Process high-urgency signals first
        sorted_signals = sorted(self.scaling_signals, key=lambda s: s.urgency, reverse=True)
        
        for signal in sorted_signals[:5]:  # Process top 5 signals
            if signal.scaling_needed():
                await self._execute_scaling_action(signal)
        
        # Clear processed signals
        self.scaling_signals.clear()
    
    async def _execute_scaling_action(self, signal: PredictiveScalingSignal) -> None:
        """Execute autonomous scaling action"""
        try:
            resource_type = signal.resource_type
            current_capacity = self.available_resources.get(resource_type, 0.0)
            
            if signal.trend_direction == ScalingDirection.SCALE_UP:
                # Increase available resources by 20%
                additional_capacity = current_capacity * 0.2
                self.available_resources[resource_type] = current_capacity + additional_capacity
                
                self.logger.info(f"Autonomous scale-up: {resource_type} increased by {additional_capacity}")
                
            elif signal.trend_direction == ScalingDirection.SCALE_DOWN:
                # Decrease available resources by 15% (more conservative)
                reduction = current_capacity * 0.15
                self.available_resources[resource_type] = max(0.0, current_capacity - reduction)
                
                self.logger.info(f"Autonomous scale-down: {resource_type} decreased by {reduction}")
            
        except Exception as e:
            self.logger.error(f"Scaling action failed for {signal.resource_type}: {e}")
    
    async def _resource_monitoring_loop(self) -> None:
        """Background loop for resource monitoring"""
        self.logger.info("Starting resource monitoring loop")
        
        while self._running:
            try:
                await self._monitor_resource_health()
                await self._detect_resource_anomalies()
                await self._update_prediction_accuracy()
                
                await asyncio.sleep(30.0)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Resource monitoring loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def _monitor_resource_health(self) -> None:
        """Monitor health of resource allocations"""
        for resource_type, available in self.available_resources.items():
            allocated = self._get_current_allocation(resource_type)
            total = available + allocated
            
            if total > 0:
                utilization = allocated / total
                
                # Log warnings for high utilization
                if utilization > 0.9:
                    self.logger.warning(f"High utilization for {resource_type}: {utilization:.1%}")
                elif utilization > 0.95:
                    self.logger.critical(f"Critical utilization for {resource_type}: {utilization:.1%}")
    
    async def _detect_resource_anomalies(self) -> None:
        """Detect anomalies in resource usage patterns"""
        # This would implement anomaly detection algorithms in production
        # For now, just log unusual patterns
        
        for resource_type, history in self.resource_utilization_history.items():
            if len(history) < 10:
                continue
            
            recent_allocations = [entry['allocation'] for entry in list(history)[-10:]]
            mean_allocation = np.mean(recent_allocations)
            std_allocation = np.std(recent_allocations)
            
            # Check for anomalies (values > 2 standard deviations from mean)
            if std_allocation > 0:
                for allocation in recent_allocations[-3:]:  # Check last 3 allocations
                    z_score = abs(allocation - mean_allocation) / std_allocation
                    if z_score > 2.0:
                        self.logger.warning(
                            f"Resource anomaly detected for {resource_type}: "
                            f"allocation={allocation}, z_score={z_score:.2f}"
                        )
    
    async def _update_prediction_accuracy(self) -> None:
        """Update prediction accuracy metrics"""
        # This would compare predictions with actual outcomes in production
        # For now, simulate accuracy tracking
        
        simulated_accuracy = np.random.uniform(0.7, 0.95)
        self.allocation_metrics['prediction_accuracy'] = simulated_accuracy
    
    async def get_allocation_status(self) -> Dict[str, Any]:
        """Get comprehensive allocation status"""
        return {
            'version': '1.0.0',
            'status': 'active' if self._running else 'inactive',
            'allocation_metrics': dict(self.allocation_metrics),
            'active_allocations': len(self.active_allocations),
            'queued_requests': len(self.allocation_requests),
            'available_resources': dict(self.available_resources),
            'resource_constraints': [asdict(c) for c in self.resource_constraints],
            'framework_metrics': {
                fid: asdict(metrics) for fid, metrics in self.framework_metrics.items()
            },
            'load_balancing_weights': dict(self.load_balancing_weights),
            'scaling_signals': len(self.scaling_signals),
            'configuration': self.config
        }
    
    async def get_framework_recommendation(self, task_requirements: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get framework recommendations for a task based on current load balancing"""
        recommendations = []
        
        for framework_id, weight in self.load_balancing_weights.items():
            metrics = self.framework_metrics.get(framework_id)
            if metrics:
                # Calculate suitability score
                load_score = 1.0 - metrics.current_load
                health_score = metrics.health_score
                weight_score = weight
                
                overall_score = (load_score * 0.4 + health_score * 0.3 + weight_score * 0.3)
                recommendations.append((framework_id, overall_score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    async def release_allocation(self, decision_id: str) -> bool:
        """Manually release an allocation"""
        if decision_id in self.active_allocations:
            decision = self.active_allocations[decision_id]
            await self._release_allocation(decision)
            del self.active_allocations[decision_id]
            
            self.logger.info(f"Manually released allocation: {decision_id}")
            return True
        
        return False


# Factory function for easy instantiation
def create_intelligent_resource_allocator(config: Dict[str, Any] = None) -> IntelligentResourceAllocator:
    """Create and return a configured Intelligent Resource Allocator"""
    return IntelligentResourceAllocator(config)


# Export main classes
__all__ = [
    'IntelligentResourceAllocator',
    'AllocationRequest',
    'AllocationDecision', 
    'LoadBalancingMetrics',
    'PredictiveScalingSignal',
    'ResourceConstraint',
    'AllocationStrategy',
    'ScalingDirection',
    'LoadBalancingAlgorithm',
    'create_intelligent_resource_allocator'
]