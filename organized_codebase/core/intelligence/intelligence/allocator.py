"""
Resource Allocator Core Module
===============================

Main resource allocator that coordinates optimization, load balancing, and
predictive scaling for intelligent resource management.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import asdict

from .data_models import (
    AllocationRequest, AllocationDecision, AllocationStrategy,
    ResourceConstraint, ScalingDirection
)
from .optimizers import LinearProgrammingOptimizer, GeneticAlgorithmOptimizer, ResourceOptimizer
from .load_balancer import LoadBalancer
from .predictive_scaler import PredictiveScaler


class ResourceAllocator:
    """
    Core resource allocator that manages allocation requests, decisions,
    and coordinates with optimization engines.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Resource Allocator"""
        self.config = config or self._get_default_config()
        
        # Core components
        self.allocation_requests: deque = deque(maxlen=1000)
        self.allocation_decisions: deque = deque(maxlen=1000)
        self.active_allocations: Dict[str, AllocationDecision] = {}
        
        # Resource tracking
        self.available_resources: Dict[str, float] = {}
        self.resource_constraints: List[ResourceConstraint] = []
        
        # Sub-modules
        self.load_balancer = LoadBalancer(config.get('load_balancing', {}))
        self.predictive_scaler = PredictiveScaler(config.get('predictive_scaling', {}))
        
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
            'resource_efficiency': 0.0
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
            'allocation_timeout': timedelta(seconds=30),
            'rebalancing_interval': timedelta(minutes=5),
            'resource_safety_margin': 0.1,
            'max_allocation_attempts': 3,
            'batch_size': 10,
            'log_level': logging.INFO
        }
    
    def _initialize_resources(self) -> None:
        """Initialize available resources"""
        self.available_resources = {
            'cpu': 100.0,
            'memory': 64.0,
            'storage': 1000.0,
            'network': 10.0,
            'gpu': 8.0,
            'threads': 128.0,
            'connections': 10000.0
        }
        
        # Share with predictive scaler
        self.predictive_scaler.available_resources = self.available_resources
    
    def _initialize_constraints(self) -> None:
        """Initialize resource constraints"""
        self.resource_constraints = [
            ResourceConstraint(
                resource_type='cpu',
                min_allocation=10.0,
                max_allocation=90.0,
                priority=10,
                cost_per_unit=0.05,
                elasticity=0.9
            ),
            ResourceConstraint(
                resource_type='memory',
                min_allocation=8.0,
                max_allocation=56.0,
                priority=9,
                cost_per_unit=0.02,
                elasticity=0.8
            ),
            ResourceConstraint(
                resource_type='storage',
                min_allocation=100.0,
                max_allocation=900.0,
                priority=7,
                cost_per_unit=0.001,
                elasticity=0.95
            ),
            ResourceConstraint(
                resource_type='gpu',
                min_allocation=1.0,
                max_allocation=7.0,
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
        
        self.logger.info(f"Allocation request {request_id} from {framework_id}")
        
        # Process allocation immediately for high urgency requests
        if urgency >= 0.8 or priority >= 8:
            return await self._process_allocation_request(request, strategy)
        else:
            # Queue for batch processing
            return self._create_pending_decision(request)
    
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
            
            self.logger.info(f"Allocation successful: {request.request_id}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Allocation processing failed for {request.request_id}: {e}")
            return self._create_failed_decision(request, str(e))
    
    def _select_allocation_strategy(self, request: AllocationRequest) -> AllocationStrategy:
        """Select the best allocation strategy for a request"""
        if request.priority >= 8 or request.urgency >= 0.8:
            return AllocationStrategy.PERFORMANCE_BASED
        
        if request.duration > timedelta(hours=4):
            return AllocationStrategy.COST_OPTIMIZED
        
        total_resource_demand = sum(request.resource_requirements.values())
        if total_resource_demand > 50.0:
            return AllocationStrategy.PREDICTIVE
        
        return self.config['default_allocation_strategy']
    
    def _select_optimizer(self, request: AllocationRequest, strategy: AllocationStrategy) -> ResourceOptimizer:
        """Select the best optimizer for the request and strategy"""
        if (strategy == AllocationStrategy.PREDICTIVE or 
            len(request.resource_requirements) > 4 or
            len(self.resource_constraints) > 6):
            return self.optimizers['genetic_algorithm']
        
        return self.optimizers['linear_programming']
    
    def _check_resource_availability(self, requirements: Dict[str, float]) -> bool:
        """Check if required resources are available"""
        for resource_type, required_amount in requirements.items():
            available = self.available_resources.get(resource_type, 0.0)
            
            for constraint in self.resource_constraints:
                if constraint.resource_type == resource_type:
                    max_allocatable = constraint.max_allocation
                    current_allocated = self._get_current_allocation(resource_type)
                    if current_allocated + required_amount > max_allocatable:
                        return False
            
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
        
        return np.mean(confidence_scores)
    
    def _calculate_allocation_cost(self, allocated_resources: Dict[str, float]) -> float:
        """Calculate estimated cost for allocation"""
        total_cost = 0.0
        
        for resource_type, allocated_amount in allocated_resources.items():
            cost_per_unit = 0.01  # Default cost
            for constraint in self.resource_constraints:
                if constraint.resource_type == resource_type:
                    cost_per_unit = constraint.cost_per_unit
                    break
            
            total_cost += allocated_amount * cost_per_unit
        
        return total_cost
    
    def _estimate_performance_impact(self, allocated_resources: Dict[str, float], request: AllocationRequest) -> float:
        """Estimate performance impact of allocation"""
        impact_factors = []
        
        for resource_type, allocated_amount in allocated_resources.items():
            required_amount = request.resource_requirements.get(resource_type, 0.0)
            
            if required_amount > 0:
                allocation_ratio = allocated_amount / required_amount
                
                if allocation_ratio >= 1.0:
                    impact_factors.append(1.0)
                elif allocation_ratio >= 0.8:
                    impact_factors.append(0.8)
                else:
                    impact_factors.append(allocation_ratio * 0.5)
        
        return np.mean(impact_factors) if impact_factors else 0.5
    
    async def _apply_allocation(self, decision: AllocationDecision) -> None:
        """Apply allocation decision"""
        for resource_type, allocated_amount in decision.allocated_resources.items():
            current_available = self.available_resources.get(resource_type, 0.0)
            self.available_resources[resource_type] = max(0.0, current_available - allocated_amount)
            
            # Update predictive scaler
            self.predictive_scaler.update_resource_utilization(
                resource_type, allocated_amount, decision.allocation_time
            )
        
        self.active_allocations[decision.decision_id] = decision
        self.allocation_decisions.append(decision)
    
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
    
    async def release_allocation(self, decision_id: str) -> bool:
        """Release an allocation"""
        if decision_id in self.active_allocations:
            decision = self.active_allocations[decision_id]
            
            # Return resources to available pool
            for resource_type, allocated_amount in decision.allocated_resources.items():
                current_available = self.available_resources.get(resource_type, 0.0)
                self.available_resources[resource_type] = current_available + allocated_amount
            
            del self.active_allocations[decision_id]
            self.logger.info(f"Released allocation: {decision_id}")
            return True
        
        return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive allocator status"""
        return {
            'allocation_metrics': dict(self.allocation_metrics),
            'active_allocations': len(self.active_allocations),
            'queued_requests': len(self.allocation_requests),
            'available_resources': dict(self.available_resources),
            'resource_constraints': [asdict(c) for c in self.resource_constraints],
            'load_balancer_metrics': self.load_balancer.get_metrics(),
            'predictive_scaler_metrics': self.predictive_scaler.get_metrics()
        }
    
    async def get_framework_recommendation(self, task_requirements: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get framework recommendations for a task"""
        return self.load_balancer.get_framework_recommendation(task_requirements)


__all__ = ['ResourceAllocator']