"""
Intelligent Resource Allocator Core - Master Resource Management Orchestration
==============================================================================

Master orchestration system for intelligent resource allocation with advanced
optimization, load balancing, and predictive scaling capabilities.
Implements enterprise-grade resource management coordination and monitoring.

This module serves as the main coordination hub for all resource allocation
capabilities, integrating optimization engines, load balancers, and predictive scaling.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: intelligent_resource_allocator_core.py (480 lines)
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import defaultdict, deque
from dataclasses import asdict, field
import json
import threading

from .resource_allocation_types import (
    AllocationRequest, AllocationDecision, LoadBalancingMetrics,
    PredictiveScalingSignal, ResourceConstraint, AllocationStrategy,
    ScalingDirection, LoadBalancingAlgorithm, AllocationPriority
)
from .optimization_engine import (
    create_optimizer, LinearProgrammingOptimizer, GeneticAlgorithmOptimizer
)
from .load_balancer import LoadBalancingEngine, PredictiveScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentResourceAllocator:
    """Master intelligent resource allocation system with enterprise orchestration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Configuration with intelligent defaults
        default_config = {
            'optimization_algorithm': 'linear_programming',
            'load_balancing_algorithm': LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN,
            'enable_predictive_scaling': True,
            'enable_autonomous_scaling': True,
            'enable_async_processing': True,
            'max_concurrent_requests': 50,
            'allocation_timeout': timedelta(seconds=30),
            'rebalancing_interval': timedelta(minutes=5),
            'scaling_check_interval': timedelta(minutes=2),
            'resource_utilization_threshold': 0.8,
            'enable_batch_processing': True,
            'batch_size': 10
        }
        
        self.config = {**default_config, **(config or {})}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core components
        self.optimization_engine = create_optimizer(self.config['optimization_algorithm'])
        self.load_balancer = LoadBalancingEngine(self.config['load_balancing_algorithm'])
        self.predictive_scaler = PredictiveScaler()
        
        # Resource management state
        self.available_resources: Dict[str, float] = {}
        self.resource_constraints: List[ResourceConstraint] = []
        self.allocation_requests: deque = deque()
        self.active_allocations: Dict[str, AllocationDecision] = {}
        self.allocation_decisions: List[AllocationDecision] = []
        
        # Framework management
        self.framework_metrics: Dict[str, LoadBalancingMetrics] = {}
        self.load_balancing_weights: Dict[str, float] = {}
        
        # Predictive scaling
        self.scaling_signals: List[PredictiveScalingSignal] = []
        self.resource_utilization_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance metrics
        self.allocation_metrics = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_allocation_time': 0.0,
            'resource_efficiency': 0.0,
            'prediction_accuracy': 0.0
        }
        
        # Background processing
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        self.logger.info("Intelligent Resource Allocator initialized with enterprise configuration")
    
    async def start(self) -> None:
        """Start the resource allocator with background processing"""
        if self._running:
            self.logger.warning("Resource allocator already running")
            return
        
        self._running = True
        
        # Start background processing tasks
        if self.config['enable_async_processing']:
            self._background_tasks = [
                asyncio.create_task(self._allocation_processing_loop()),
                asyncio.create_task(self._load_balancing_loop()),
                asyncio.create_task(self._predictive_scaling_loop()),
                asyncio.create_task(self._resource_monitoring_loop())
            ]
        
        self.logger.info("Intelligent Resource Allocator started with background processing")
    
    async def stop(self) -> None:
        """Stop the resource allocator and cleanup background tasks"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self.logger.info("Intelligent Resource Allocator stopped")
    
    async def allocate_resources(self, request: AllocationRequest) -> AllocationDecision:
        """Allocate resources for a single request with comprehensive optimization"""
        try:
            start_time = datetime.now()
            self.allocation_metrics['total_requests'] += 1
            
            # Validate request
            if not self._validate_request(request):
                return self._create_failed_decision(request, "Invalid request")
            
            # High-priority requests get immediate processing
            if request.priority in [AllocationPriority.CRITICAL, AllocationPriority.HIGH] or request.urgency > 0.8:
                decision = await self._process_high_priority_request(request)
            else:
                # Queue for batch processing
                self.allocation_requests.append(request)
                decision = AllocationDecision(
                    decision_id=f"queued_{request.request_id}",
                    request_id=request.request_id,
                    framework_id=request.framework_id,
                    allocated_resources={},
                    allocation_strategy=AllocationStrategy.DEMAND_BASED,
                    confidence=0.5,
                    cost_estimate=0.0,
                    performance_impact=0.0,
                    allocation_time=datetime.now(),
                    expiration_time=datetime.now() + timedelta(minutes=1),
                    success=False,
                    failure_reason="Queued for batch processing"
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_allocation_metrics(decision, processing_time, decision.success)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error allocating resources for {request.request_id}: {e}")
            return self._create_failed_decision(request, str(e))
    
    async def batch_allocate_resources(self, requests: List[AllocationRequest]) -> Dict[str, AllocationDecision]:
        """Batch allocate resources with enterprise optimization"""
        try:
            if not requests:
                return {}
            
            self.logger.info(f"Processing batch allocation for {len(requests)} requests")
            
            # Use optimization engine for batch processing
            optimization_result = self.optimization_engine.optimize_allocation(
                self.available_resources, requests, self.resource_constraints
            )
            
            decisions = {}
            
            if optimization_result.convergence_status == "optimal":
                # Create decisions from optimization results
                for request in requests:
                    allocated_resources = optimization_result.solution.get(request.request_id, {})
                    
                    if allocated_resources and self._validate_allocation(allocated_resources, request):
                        decision = AllocationDecision(
                            decision_id=f"batch_{request.request_id}",
                            request_id=request.request_id,
                            framework_id=request.framework_id,
                            allocated_resources=allocated_resources,
                            allocation_strategy=AllocationStrategy.PREDICTIVE,
                            confidence=self._calculate_allocation_confidence(allocated_resources, request),
                            cost_estimate=self._calculate_allocation_cost(allocated_resources),
                            performance_impact=self._estimate_performance_impact(allocated_resources, request),
                            allocation_time=datetime.now(),
                            expiration_time=datetime.now() + request.duration,
                            optimization_score=optimization_result.objective_value,
                            success=True
                        )
                        
                        await self._apply_allocation(decision)
                        decisions[request.request_id] = decision
                    else:
                        decisions[request.request_id] = self._create_failed_decision(
                            request, "Batch optimization failed validation"
                        )
            else:
                # Fallback to individual processing
                for request in requests:
                    decisions[request.request_id] = await self._process_fallback_allocation(request)
            
            return decisions
            
        except Exception as e:
            self.logger.error(f"Batch allocation failed: {e}")
            return {req.request_id: self._create_failed_decision(req, str(e)) for req in requests}
    
    async def _process_high_priority_request(self, request: AllocationRequest) -> AllocationDecision:
        """Process high-priority request immediately"""
        try:
            # Use genetic algorithm for complex high-priority requests
            ga_optimizer = GeneticAlgorithmOptimizer(population_size=30, generations=50)
            
            optimization_result = ga_optimizer.optimize_allocation(
                self.available_resources, [request], self.resource_constraints
            )
            
            allocated_resources = optimization_result.solution.get(request.request_id, {})
            
            if allocated_resources and self._validate_allocation(allocated_resources, request):
                decision = AllocationDecision(
                    decision_id=f"priority_{request.request_id}",
                    request_id=request.request_id,
                    framework_id=request.framework_id,
                    allocated_resources=allocated_resources,
                    allocation_strategy=AllocationStrategy.PRIORITY_BASED,
                    confidence=self._calculate_allocation_confidence(allocated_resources, request),
                    cost_estimate=self._calculate_allocation_cost(allocated_resources),
                    performance_impact=self._estimate_performance_impact(allocated_resources, request),
                    allocation_time=datetime.now(),
                    expiration_time=datetime.now() + request.duration,
                    optimization_score=optimization_result.objective_value,
                    success=True
                )
                
                await self._apply_allocation(decision)
                return decision
            else:
                return self._create_failed_decision(request, "High-priority allocation validation failed")
                
        except Exception as e:
            self.logger.error(f"High-priority allocation failed: {e}")
            return self._create_failed_decision(request, str(e))
    
    async def _process_fallback_allocation(self, request: AllocationRequest) -> AllocationDecision:
        """Process allocation with simple fallback strategy"""
        try:
            # Simple greedy allocation as fallback
            allocated_resources = {}
            
            for resource_type, required in request.resource_requirements.items():
                available = self.available_resources.get(resource_type, 0.0)
                allocated = min(required, available)
                allocated_resources[resource_type] = allocated
            
            if self._validate_allocation(allocated_resources, request):
                decision = AllocationDecision(
                    decision_id=f"fallback_{request.request_id}",
                    request_id=request.request_id,
                    framework_id=request.framework_id,
                    allocated_resources=allocated_resources,
                    allocation_strategy=AllocationStrategy.FAIR_SHARE,
                    confidence=0.6,
                    cost_estimate=self._calculate_allocation_cost(allocated_resources),
                    performance_impact=self._estimate_performance_impact(allocated_resources, request),
                    allocation_time=datetime.now(),
                    expiration_time=datetime.now() + request.duration,
                    success=True
                )
                
                await self._apply_allocation(decision)
                return decision
            else:
                return self._create_failed_decision(request, "Fallback allocation insufficient")
                
        except Exception as e:
            return self._create_failed_decision(request, f"Fallback error: {e}")
    
    def _validate_request(self, request: AllocationRequest) -> bool:
        """Validate allocation request comprehensively"""
        try:
            # Check required fields
            if not request.request_id or not request.framework_id:
                return False
            
            # Check resource requirements
            if not request.resource_requirements:
                return False
            
            # Check for negative requirements
            for resource_type, amount in request.resource_requirements.items():
                if amount < 0:
                    return False
            
            # Check deadline validity
            if request.deadline and request.deadline < datetime.now():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Request validation error: {e}")
            return False
    
    def _validate_allocation(self, allocation: Dict[str, float], request: AllocationRequest) -> bool:
        """Validate allocation against request and constraints"""
        try:
            # Check if allocation meets minimum requirements
            satisfaction_ratio = 0.0
            total_requirements = sum(request.resource_requirements.values())
            
            if total_requirements > 0:
                satisfied_requirements = 0.0
                for resource_type, required in request.resource_requirements.items():
                    allocated = allocation.get(resource_type, 0.0)
                    satisfied_requirements += min(required, allocated)
                
                satisfaction_ratio = satisfied_requirements / total_requirements
            
            # Require at least 50% satisfaction for validation
            return satisfaction_ratio >= 0.5
            
        except Exception as e:
            self.logger.error(f"Allocation validation error: {e}")
            return False
    
    def _calculate_allocation_confidence(self, allocation: Dict[str, float], 
                                       request: AllocationRequest) -> float:
        """Calculate confidence score for allocation"""
        try:
            total_required = sum(request.resource_requirements.values())
            total_allocated = sum(allocation.values())
            
            if total_required <= 0:
                return 0.0
            
            # Base confidence on satisfaction ratio
            satisfaction_ratio = min(1.0, total_allocated / total_required)
            
            # Adjust for resource availability
            availability_factor = 1.0
            for resource_type, allocated in allocation.items():
                available = self.available_resources.get(resource_type, 0.0)
                if available > 0:
                    utilization = allocated / available
                    if utilization > 0.9:  # High utilization reduces confidence
                        availability_factor *= (1.0 - (utilization - 0.9) / 0.1 * 0.3)
            
            confidence = satisfaction_ratio * availability_factor
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _calculate_allocation_cost(self, allocation: Dict[str, float]) -> float:
        """Calculate estimated cost of allocation"""
        try:
            total_cost = 0.0
            
            # Simple cost model (would be more sophisticated in production)
            cost_per_unit = {'cpu': 1.0, 'memory': 0.5, 'storage': 0.1, 'network': 0.3}
            
            for resource_type, amount in allocation.items():
                unit_cost = cost_per_unit.get(resource_type, 1.0)
                total_cost += amount * unit_cost
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Cost calculation error: {e}")
            return 0.0
    
    def _estimate_performance_impact(self, allocation: Dict[str, float], 
                                   request: AllocationRequest) -> float:
        """Estimate performance impact of allocation"""
        try:
            # Calculate resource adequacy
            adequacy_scores = []
            
            for resource_type, required in request.resource_requirements.items():
                allocated = allocation.get(resource_type, 0.0)
                if required > 0:
                    adequacy = min(1.0, allocated / required)
                    adequacy_scores.append(adequacy)
            
            if adequacy_scores:
                average_adequacy = np.mean(adequacy_scores)
                # Performance impact is positive when allocation is adequate
                return average_adequacy
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Performance impact estimation error: {e}")
            return 0.5
    
    async def _apply_allocation(self, decision: AllocationDecision) -> None:
        """Apply allocation by updating resource availability"""
        try:
            # Reduce available resources
            for resource_type, allocated_amount in decision.allocated_resources.items():
                current_available = self.available_resources.get(resource_type, 0.0)
                self.available_resources[resource_type] = max(0.0, current_available - allocated_amount)
            
            # Record active allocation
            self.active_allocations[decision.decision_id] = decision
            self.allocation_decisions.append(decision)
            
            # Update resource utilization history for predictive scaling
            for resource_type, allocated in decision.allocated_resources.items():
                total_capacity = self.available_resources.get(resource_type, 0.0) + allocated
                utilization = allocated / total_capacity if total_capacity > 0 else 0.0
                
                self.predictive_scaler.record_resource_usage(
                    resource_type, allocated, total_capacity
                )
            
        except Exception as e:
            self.logger.error(f"Error applying allocation {decision.decision_id}: {e}")
    
    async def _allocation_processing_loop(self) -> None:
        """Background loop for processing queued allocation requests"""
        self.logger.info("Starting allocation processing loop")
        
        while self._running:
            try:
                # Process pending requests in batches
                await self._process_allocation_batch()
                
                # Clean up expired allocations
                await self._cleanup_expired_allocations()
                
                # Update resource efficiency metrics
                self._update_resource_efficiency()
                
                await asyncio.sleep(1.0)  # Process frequently
                
            except Exception as e:
                self.logger.error(f"Allocation processing loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_allocation_batch(self) -> None:
        """Process a batch of queued allocation requests"""
        if not self.allocation_requests:
            return
        
        batch_size = self.config['batch_size']
        requests_to_process = []
        
        # Collect requests for batch processing
        for _ in range(min(batch_size, len(self.allocation_requests))):
            if self.allocation_requests:
                request = self.allocation_requests.popleft()
                requests_to_process.append(request)
        
        if requests_to_process:
            self.logger.debug(f"Processing batch of {len(requests_to_process)} requests")
            batch_decisions = await self.batch_allocate_resources(requests_to_process)
            
            # Update metrics for batch
            for decision in batch_decisions.values():
                self._update_allocation_metrics(decision, 0.1, decision.success)
    
    async def _cleanup_expired_allocations(self) -> None:
        """Clean up expired allocations and return resources"""
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
        """Release resources from an allocation back to available pool"""
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
                
                await asyncio.sleep(self.config['rebalancing_interval'].total_seconds())
                
            except Exception as e:
                self.logger.error(f"Load balancing loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _update_framework_metrics(self) -> None:
        """Update metrics for all registered frameworks"""
        # In production, this would collect real metrics from frameworks
        # For demonstration, simulate realistic metrics
        
        frameworks = ['analytics', 'ml', 'api', 'analysis']
        
        for framework_id in frameworks:
            # Simulate realistic metrics
            metrics = LoadBalancingMetrics(
                framework_id=framework_id,
                current_load=np.random.uniform(0.2, 0.9),
                capacity=100.0,
                utilization=np.random.uniform(0.3, 0.8),
                response_time=np.random.uniform(0.05, 0.3),
                throughput=np.random.uniform(50, 200),
                error_rate=np.random.uniform(0.001, 0.05),
                health_score=np.random.uniform(0.8, 1.0),
                weight=1.0
            )
            
            self.framework_metrics[framework_id] = metrics
            self.load_balancer.update_framework_metrics(framework_id, metrics)
    
    async def _optimize_load_balancing(self) -> None:
        """Optimize load balancing weights across frameworks"""
        try:
            # Get recommendations from load balancer
            status = await self.load_balancer.get_load_balancing_status()
            
            # Update internal weights based on load balancer recommendations
            for framework_id, framework_status in status.get('framework_status', {}).items():
                performance_score = framework_status.get('performance_score', 1.0)
                self.load_balancing_weights[framework_id] = performance_score
            
            # Normalize weights
            total_weight = sum(self.load_balancing_weights.values())
            if total_weight > 0:
                for framework_id in self.load_balancing_weights:
                    self.load_balancing_weights[framework_id] /= total_weight
                    
        except Exception as e:
            self.logger.error(f"Load balancing optimization error: {e}")
    
    async def _predictive_scaling_loop(self) -> None:
        """Background loop for predictive scaling analysis"""
        self.logger.info("Starting predictive scaling loop")
        
        while self._running:
            try:
                # Generate scaling signals
                signals = await self.predictive_scaler.generate_scaling_signals()
                self.scaling_signals.extend(signals)
                
                # Process scaling signals if autonomous scaling enabled
                if self.config['enable_autonomous_scaling']:
                    await self._process_scaling_signals()
                
                await asyncio.sleep(self.config['scaling_check_interval'].total_seconds())
                
            except Exception as e:
                self.logger.error(f"Predictive scaling loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def _process_scaling_signals(self) -> None:
        """Process scaling signals and take autonomous actions"""
        # Sort by urgency and process top signals
        sorted_signals = sorted(self.scaling_signals, key=lambda s: s.urgency, reverse=True)
        
        for signal in sorted_signals[:5]:  # Process top 5 signals
            if signal.scaling_needed():
                await self._execute_scaling_action(signal)
        
        # Clear processed signals
        self.scaling_signals.clear()
    
    async def _execute_scaling_action(self, signal: PredictiveScalingSignal) -> None:
        """Execute autonomous scaling action based on signal"""
        try:
            resource_type = signal.resource_type
            current_capacity = self.available_resources.get(resource_type, 0.0)
            
            if signal.trend_direction == ScalingDirection.SCALE_UP:
                # Increase available resources
                scaling_magnitude = signal.calculate_scaling_magnitude()
                additional_capacity = current_capacity * abs(scaling_magnitude)
                self.available_resources[resource_type] = current_capacity + additional_capacity
                
                self.logger.info(f"Autonomous scale-up: {resource_type} increased by {additional_capacity}")
                
            elif signal.trend_direction == ScalingDirection.SCALE_DOWN:
                # Decrease available resources (more conservative)
                scaling_magnitude = signal.calculate_scaling_magnitude()
                reduction = current_capacity * abs(scaling_magnitude) * 0.5  # Conservative scaling down
                self.available_resources[resource_type] = max(0.0, current_capacity - reduction)
                
                self.logger.info(f"Autonomous scale-down: {resource_type} decreased by {reduction}")
            
        except Exception as e:
            self.logger.error(f"Scaling action failed for {signal.resource_type}: {e}")
    
    async def _resource_monitoring_loop(self) -> None:
        """Background loop for resource health monitoring"""
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
        # In production, this would compare predictions with actual outcomes
        # For now, simulate accuracy tracking
        simulated_accuracy = np.random.uniform(0.7, 0.95)
        self.allocation_metrics['prediction_accuracy'] = simulated_accuracy
    
    def _get_current_allocation(self, resource_type: str) -> float:
        """Get current allocation for a resource type"""
        return sum(
            decision.allocated_resources.get(resource_type, 0.0)
            for decision in self.active_allocations.values()
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
        return decision
    
    def _update_allocation_metrics(self, decision: AllocationDecision, 
                                 processing_time: float, success: bool) -> None:
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
    
    # Public API methods
    def set_available_resources(self, resources: Dict[str, float]) -> None:
        """Set available resources for allocation"""
        self.available_resources = resources.copy()
        self.logger.info(f"Updated available resources: {resources}")
    
    def add_resource_constraint(self, constraint: ResourceConstraint) -> None:
        """Add a resource constraint"""
        self.resource_constraints.append(constraint)
        self.logger.info(f"Added resource constraint for {constraint.resource_type}")
    
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
    
    async def get_framework_recommendation(self, task_requirements: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """Get framework recommendations using load balancer"""
        return await self.load_balancer.select_framework(task_requirements)


# Factory function for easy instantiation
def create_intelligent_resource_allocator(config: Dict[str, Any] = None) -> IntelligentResourceAllocator:
    """Create and return a configured Intelligent Resource Allocator"""
    return IntelligentResourceAllocator(config)


# Export main classes and functions
__all__ = [
    'IntelligentResourceAllocator',
    'create_intelligent_resource_allocator'
]