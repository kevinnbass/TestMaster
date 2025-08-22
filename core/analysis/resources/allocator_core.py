"""
Resource Intelligence Allocator Core
===================================

Main orchestration system for intelligent resource allocation with predictive scaling.
Extracted from intelligent_resource_allocator.py for enterprise modular architecture.

Agent D Implementation - Hour 11-12: Revolutionary Intelligence Modularization
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque

from .data_models import (
    AllocationStrategy, AllocationRequest, AllocationDecision, 
    ResourceConstraint, ResourceAllocationMetrics, ScalingAction
)
from .optimizers import LinearProgrammingOptimizer, GeneticAlgorithmOptimizer, ResourceOptimizer
from .load_balancer import LoadBalancer, LoadBalancingAlgorithm
from .predictive_scaler import PredictiveScaler
from .resource_monitor import ResourceMonitor


class IntelligentResourceAllocator:
    """Master intelligent resource allocation system with autonomous optimization"""
    
    def __init__(self, 
                 resource_types: List[str] = None,
                 default_strategy: AllocationStrategy = AllocationStrategy.PERFORMANCE_BASED,
                 enable_predictive_scaling: bool = True,
                 enable_load_balancing: bool = True,
                 enable_monitoring: bool = True):
        
        # Core configuration
        self.resource_types = resource_types or ['cpu', 'memory', 'storage', 'network', 'gpu', 'threads', 'connections']
        self.default_strategy = default_strategy
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_load_balancing = enable_load_balancing
        self.enable_monitoring = enable_monitoring
        
        # Resource state
        self.available_resources = {rt: 100.0 for rt in self.resource_types}  # Default capacity
        self.allocated_resources = {rt: 0.0 for rt in self.resource_types}
        self.reserved_resources = {rt: 0.0 for rt in self.resource_types}
        
        # Request management
        self.pending_requests = deque()
        self.active_allocations = {}  # request_id -> AllocationDecision
        self.allocation_history = deque(maxlen=1000)
        
        # Optimization components
        self.optimizers = {
            'linear_programming': LinearProgrammingOptimizer(),
            'genetic_algorithm': GeneticAlgorithmOptimizer()
        }
        self.current_optimizer = self.optimizers['linear_programming']
        
        # Advanced components
        self.load_balancer = LoadBalancer(LoadBalancingAlgorithm.PERFORMANCE_BASED) if enable_load_balancing else None
        self.predictive_scaler = PredictiveScaler() if enable_predictive_scaling else None
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
        
        # Metrics and monitoring
        self.allocation_metrics = ResourceAllocationMetrics()
        self.performance_history = deque(maxlen=100)
        
        # Background tasks
        self.background_tasks = set()
        self.is_running = False
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        if self.enable_monitoring and self.resource_monitor:
            self._setup_monitoring_callbacks()
    
    async def start(self):
        """Start the resource allocator and background tasks"""
        if self.is_running:
            self.logger.warning("Resource allocator is already running")
            return
        
        self.is_running = True
        
        # Start monitoring
        if self.enable_monitoring and self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        # Start background tasks
        if self.enable_predictive_scaling:
            self.background_tasks.add(asyncio.create_task(self._predictive_scaling_loop()))
        
        self.background_tasks.add(asyncio.create_task(self._allocation_processing_loop()))
        self.background_tasks.add(asyncio.create_task(self._resource_optimization_loop()))
        
        self.logger.info("Intelligent Resource Allocator started successfully")
    
    async def stop(self):
        """Stop the resource allocator and cleanup"""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Stop monitoring
        if self.enable_monitoring and self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        self.logger.info("Intelligent Resource Allocator stopped")
    
    async def request_resources(self, request: AllocationRequest) -> AllocationDecision:
        """Request resource allocation with intelligent optimization"""
        try:
            self.logger.info(f"Processing allocation request: {request.request_id}")
            
            # Validate request
            if not self._validate_request(request):
                return self._create_failed_decision(request, "Invalid request parameters")
            
            # Check resource availability
            if not self._check_resource_availability(request):
                # Try to queue for later processing
                self.pending_requests.append(request)
                return self._create_failed_decision(request, "Insufficient resources - queued for later")
            
            # Select optimization strategy
            optimizer = self._select_optimizer(request)
            
            # Optimize allocation
            allocation_result = optimizer.optimize_allocation(
                self._get_available_resources_for_allocation(),
                [request],
                request.constraints
            )
            
            if request.request_id in allocation_result:
                allocated_resources = allocation_result[request.request_id]
                
                # Create allocation decision
                decision = AllocationDecision(
                    decision_id=str(uuid.uuid4()),
                    request_id=request.request_id,
                    framework_id=request.framework_id,
                    allocated_resources=allocated_resources,
                    allocation_strategy=self._determine_strategy(request),
                    confidence=self._calculate_allocation_confidence(request, allocated_resources),
                    cost_estimate=self._estimate_allocation_cost(allocated_resources),
                    performance_impact=self._estimate_performance_impact(allocated_resources),
                    allocation_time=datetime.now(),
                    expiration_time=datetime.now() + request.duration,
                    success=True
                )
                
                # Apply allocation
                self._apply_allocation(decision)
                
                # Update metrics
                self._update_allocation_metrics(decision, True)
                
                self.logger.info(f"Successfully allocated resources for request {request.request_id}")
                return decision
            else:
                return self._create_failed_decision(request, "Optimization failed to allocate resources")
                
        except Exception as e:
            self.logger.error(f"Error processing allocation request {request.request_id}: {e}")
            return self._create_failed_decision(request, f"Internal error: {str(e)}")
    
    async def release_resources(self, allocation_decision: AllocationDecision):
        """Release allocated resources"""
        try:
            if allocation_decision.decision_id in self.active_allocations:
                # Release resources
                for resource_type, amount in allocation_decision.allocated_resources.items():
                    if resource_type in self.allocated_resources:
                        self.allocated_resources[resource_type] -= amount
                        self.allocated_resources[resource_type] = max(0, self.allocated_resources[resource_type])
                
                # Remove from active allocations
                del self.active_allocations[allocation_decision.decision_id]
                
                # Add to history
                self.allocation_history.append({
                    'decision': allocation_decision,
                    'released_at': datetime.now(),
                    'duration': datetime.now() - allocation_decision.allocation_time
                })
                
                # Release connection from load balancer
                if self.enable_load_balancing and self.load_balancer:
                    self.load_balancer.release_connection(allocation_decision.framework_id)
                
                self.logger.info(f"Released resources for allocation {allocation_decision.decision_id}")
                
                # Process pending requests
                await self._process_pending_requests()
                
        except Exception as e:
            self.logger.error(f"Error releasing resources for {allocation_decision.decision_id}: {e}")
    
    def update_resource_capacity(self, resource_type: str, new_capacity: float):
        """Update resource capacity"""
        if resource_type in self.available_resources:
            old_capacity = self.available_resources[resource_type]
            self.available_resources[resource_type] = new_capacity
            
            self.logger.info(f"Updated {resource_type} capacity from {old_capacity} to {new_capacity}")
            
            # Update monitoring
            if self.enable_monitoring and self.resource_monitor:
                self.resource_monitor.update_resource_metrics(resource_type, {
                    'capacity': new_capacity,
                    'utilization': self.allocated_resources.get(resource_type, 0) / new_capacity if new_capacity > 0 else 0
                })
    
    def _validate_request(self, request: AllocationRequest) -> bool:
        """Validate allocation request parameters"""
        try:
            # Check required fields
            if not request.request_id or not request.framework_id:
                return False
            
            # Check resource requirements
            if not request.resource_requirements:
                return False
            
            # Check if requested resources are valid
            for resource_type in request.resource_requirements:
                if resource_type not in self.resource_types:
                    self.logger.warning(f"Unknown resource type requested: {resource_type}")
                    return False
            
            # Check for negative resource requests
            for amount in request.resource_requirements.values():
                if amount < 0:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating request: {e}")
            return False
    
    def _check_resource_availability(self, request: AllocationRequest) -> bool:
        """Check if requested resources are available"""
        try:
            for resource_type, requested_amount in request.resource_requirements.items():
                available = self.available_resources.get(resource_type, 0)
                allocated = self.allocated_resources.get(resource_type, 0)
                remaining = available - allocated
                
                if requested_amount > remaining:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking resource availability: {e}")
            return False
    
    def _get_available_resources_for_allocation(self) -> Dict[str, float]:
        """Get currently available resources for allocation"""
        available = {}
        for resource_type in self.resource_types:
            total_capacity = self.available_resources.get(resource_type, 0)
            current_allocation = self.allocated_resources.get(resource_type, 0)
            available[resource_type] = max(0, total_capacity - current_allocation)
        
        return available
    
    def _select_optimizer(self, request: AllocationRequest) -> ResourceOptimizer:
        """Select best optimizer for the request"""
        # Simple selection logic - would be more sophisticated in production
        if len(self.pending_requests) > 10:
            # Use genetic algorithm for complex multi-request optimization
            return self.optimizers['genetic_algorithm']
        else:
            # Use linear programming for simpler cases
            return self.optimizers['linear_programming']
    
    def _determine_strategy(self, request: AllocationRequest) -> AllocationStrategy:
        """Determine allocation strategy for request"""
        # Use request metadata or default strategy
        if 'preferred_strategy' in request.metadata:
            try:
                return AllocationStrategy(request.metadata['preferred_strategy'])
            except ValueError:
                pass
        
        return self.default_strategy
    
    def _calculate_allocation_confidence(self, request: AllocationRequest, 
                                       allocated_resources: Dict[str, float]) -> float:
        """Calculate confidence in allocation decision"""
        try:
            confidence_factors = []
            
            # Resource satisfaction factor
            for resource_type, requested in request.resource_requirements.items():
                allocated = allocated_resources.get(resource_type, 0)
                if requested > 0:
                    satisfaction = min(1.0, allocated / requested)
                    confidence_factors.append(satisfaction)
            
            # System load factor
            total_utilization = sum(
                self.allocated_resources.get(rt, 0) / self.available_resources.get(rt, 1)
                for rt in self.resource_types
            ) / len(self.resource_types)
            
            load_factor = 1.0 - min(0.5, total_utilization)  # Lower utilization = higher confidence
            confidence_factors.append(load_factor)
            
            # Historical success factor
            if self.allocation_metrics.total_requests > 0:
                success_rate = self.allocation_metrics.success_rate()
                confidence_factors.append(success_rate)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating allocation confidence: {e}")
            return 0.5
    
    def _estimate_allocation_cost(self, allocated_resources: Dict[str, float]) -> float:
        """Estimate cost of resource allocation"""
        # Simplified cost model - would be more sophisticated in production
        base_cost_per_unit = 1.0
        total_cost = sum(amount * base_cost_per_unit for amount in allocated_resources.values())
        return total_cost
    
    def _estimate_performance_impact(self, allocated_resources: Dict[str, float]) -> float:
        """Estimate performance impact of allocation"""
        # Simplified impact model
        total_allocation = sum(allocated_resources.values())
        total_capacity = sum(self.available_resources.values())
        
        if total_capacity > 0:
            impact_ratio = total_allocation / total_capacity
            return min(1.0, impact_ratio)
        
        return 0.0
    
    def _apply_allocation(self, decision: AllocationDecision):
        """Apply allocation decision to resource state"""
        # Update allocated resources
        for resource_type, amount in decision.allocated_resources.items():
            if resource_type in self.allocated_resources:
                self.allocated_resources[resource_type] += amount
            else:
                self.allocated_resources[resource_type] = amount
        
        # Add to active allocations
        self.active_allocations[decision.decision_id] = decision
    
    def _create_failed_decision(self, request: AllocationRequest, reason: str) -> AllocationDecision:
        """Create failed allocation decision"""
        decision = AllocationDecision(
            decision_id=str(uuid.uuid4()),
            request_id=request.request_id,
            framework_id=request.framework_id,
            allocated_resources={},
            allocation_strategy=self.default_strategy,
            confidence=0.0,
            cost_estimate=0.0,
            performance_impact=0.0,
            allocation_time=datetime.now(),
            expiration_time=datetime.now(),
            success=False,
            failure_reason=reason
        )
        
        self._update_allocation_metrics(decision, False)
        return decision
    
    def _update_allocation_metrics(self, decision: AllocationDecision, success: bool):
        """Update allocation performance metrics"""
        self.allocation_metrics.total_requests += 1
        
        if success:
            self.allocation_metrics.successful_allocations += 1
            
            # Update resource allocation totals
            for resource_type, amount in decision.allocated_resources.items():
                if resource_type in self.allocation_metrics.total_resources_allocated:
                    self.allocation_metrics.total_resources_allocated[resource_type] += amount
                else:
                    self.allocation_metrics.total_resources_allocated[resource_type] = amount
        else:
            self.allocation_metrics.failed_allocations += 1
    
    async def _allocation_processing_loop(self):
        """Background loop for processing allocation requests"""
        while self.is_running:
            try:
                await self._process_pending_requests()
                await asyncio.sleep(5.0)  # Process every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in allocation processing loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_pending_requests(self):
        """Process pending allocation requests"""
        processed_requests = []
        
        while self.pending_requests:
            request = self.pending_requests.popleft()
            
            if self._check_resource_availability(request):
                # Process the request
                decision = await self.request_resources(request)
                if decision.success:
                    processed_requests.append(request)
            else:
                # Put back in queue if still within deadline
                if request.deadline and datetime.now() < request.deadline:
                    self.pending_requests.append(request)
                else:
                    # Request expired
                    self._create_failed_decision(request, "Request deadline exceeded")
        
        if processed_requests:
            self.logger.info(f"Processed {len(processed_requests)} pending requests")
    
    async def _predictive_scaling_loop(self):
        """Background loop for predictive scaling"""
        if not self.enable_predictive_scaling or not self.predictive_scaler:
            return
        
        while self.is_running:
            try:
                # Update resource metrics for scaling
                for resource_type in self.resource_types:
                    current_demand = self.allocated_resources.get(resource_type, 0)
                    current_supply = self.available_resources.get(resource_type, 0)
                    
                    self.predictive_scaler.update_resource_metrics(
                        resource_type, current_demand, current_supply
                    )
                
                # Generate and process scaling signals
                scaling_signals = self.predictive_scaler.generate_scaling_signals(self.resource_types)
                
                for signal in scaling_signals:
                    if signal.scaling_needed():
                        await self._execute_scaling_action(signal)
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in predictive scaling loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _execute_scaling_action(self, signal):
        """Execute scaling action based on signal"""
        try:
            # Calculate target allocation
            if signal.trend_direction.value == 'scale_up':
                target_allocation = signal.predicted_demand * 1.2  # 20% buffer
                scale_factor = 1.2
            elif signal.trend_direction.value == 'scale_down':
                target_allocation = signal.predicted_demand * 1.1  # 10% buffer
                scale_factor = 0.9
            else:
                return
            
            # Execute scaling
            current_capacity = self.available_resources.get(signal.resource_type, 0)
            new_capacity = current_capacity * scale_factor
            
            self.update_resource_capacity(signal.resource_type, new_capacity)
            
            # Record scaling action
            action = self.predictive_scaler.execute_scaling_action(signal, new_capacity)
            
            self.logger.info(f"Executed scaling action: {action.action_id}")
            
        except Exception as e:
            self.logger.error(f"Error executing scaling action: {e}")
    
    async def _resource_optimization_loop(self):
        """Background loop for resource optimization"""
        while self.is_running:
            try:
                # Check for expired allocations
                current_time = datetime.now()
                expired_allocations = [
                    decision for decision in self.active_allocations.values()
                    if decision.expiration_time <= current_time
                ]
                
                for decision in expired_allocations:
                    await self.release_resources(decision)
                
                # Optimize resource distribution
                await self._optimize_resource_distribution()
                
                await asyncio.sleep(30.0)  # Optimize every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resource optimization loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _optimize_resource_distribution(self):
        """Optimize current resource distribution"""
        try:
            # Simple rebalancing logic - would be more sophisticated in production
            if len(self.active_allocations) > 5:
                # Check for over-utilized resources
                for resource_type in self.resource_types:
                    capacity = self.available_resources.get(resource_type, 0)
                    allocated = self.allocated_resources.get(resource_type, 0)
                    
                    if capacity > 0:
                        utilization = allocated / capacity
                        if utilization > 0.9:  # Over 90% utilized
                            self.logger.warning(f"Resource {resource_type} is over-utilized: {utilization:.1%}")
                            # Could trigger scaling or rebalancing here
                            
        except Exception as e:
            self.logger.error(f"Error optimizing resource distribution: {e}")
    
    def _setup_monitoring_callbacks(self):
        """Setup monitoring callbacks for resource events"""
        if not self.resource_monitor:
            return
        
        # Register anomaly callback
        self.resource_monitor.register_callback('anomaly', self._handle_anomaly_event)
        
        # Register performance degradation callback
        self.resource_monitor.register_callback('performance_degradation', self._handle_performance_degradation)
    
    def _handle_anomaly_event(self, anomaly_event: Dict):
        """Handle anomaly detection events"""
        self.logger.warning(f"Anomaly detected: {anomaly_event}")
        # Could trigger automatic remediation actions here
    
    def _handle_performance_degradation(self, degradation_event: Dict):
        """Handle performance degradation events"""
        self.logger.warning(f"Performance degradation detected: {degradation_event}")
        # Could trigger resource scaling or optimization here
    
    def get_allocation_status(self) -> Dict:
        """Get comprehensive allocation status"""
        return {
            'is_running': self.is_running,
            'available_resources': self.available_resources.copy(),
            'allocated_resources': self.allocated_resources.copy(),
            'resource_utilization': {
                rt: self.allocated_resources.get(rt, 0) / self.available_resources.get(rt, 1)
                for rt in self.resource_types
            },
            'active_allocations': len(self.active_allocations),
            'pending_requests': len(self.pending_requests),
            'allocation_metrics': {
                'total_requests': self.allocation_metrics.total_requests,
                'success_rate': self.allocation_metrics.success_rate(),
                'average_confidence': self.allocation_metrics.average_confidence
            }
        }


def create_intelligent_resource_allocator(
    resource_types: List[str] = None,
    default_strategy: AllocationStrategy = AllocationStrategy.PERFORMANCE_BASED,
    enable_predictive_scaling: bool = True,
    enable_load_balancing: bool = True,
    enable_monitoring: bool = True
) -> IntelligentResourceAllocator:
    """Factory function to create intelligent resource allocator"""
    return IntelligentResourceAllocator(
        resource_types=resource_types,
        default_strategy=default_strategy,
        enable_predictive_scaling=enable_predictive_scaling,
        enable_load_balancing=enable_load_balancing,
        enable_monitoring=enable_monitoring
    )