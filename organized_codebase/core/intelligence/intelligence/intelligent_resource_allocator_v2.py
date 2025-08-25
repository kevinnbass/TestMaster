"""
Intelligent Resource Allocator V2 - Facade
===========================================

Backward-compatible facade for the modularized resource allocation system.
This file provides the same interface as the original monolithic version
while delegating to the new modular components.

Agent D - Hour 0-4: Resource Allocator Modularization Complete
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

from resource_allocation import (
    ResourceAllocator,
    LoadBalancer,
    PredictiveScaler,
    AllocationRequest,
    AllocationDecision,
    LoadBalancingMetrics,
    PredictiveScalingSignal,
    ResourceConstraint,
    AllocationStrategy,
    ScalingDirection,
    LoadBalancingAlgorithm
)


class IntelligentResourceAllocator:
    """
    Facade class that provides backward compatibility with the original
    monolithic IntelligentResourceAllocator while using the new modular system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Intelligent Resource Allocator"""
        self.config = config or self._get_default_config()
        
        # Initialize the modular components
        self.resource_allocator = ResourceAllocator(self._extract_allocator_config())
        self.load_balancer = self.resource_allocator.load_balancer
        self.predictive_scaler = self.resource_allocator.predictive_scaler
        
        # Expose internal state for backward compatibility
        self.allocation_requests = self.resource_allocator.allocation_requests
        self.allocation_decisions = self.resource_allocator.allocation_decisions
        self.active_allocations = self.resource_allocator.active_allocations
        self.available_resources = self.resource_allocator.available_resources
        self.resource_constraints = self.resource_allocator.resource_constraints
        
        # Expose metrics
        self.allocation_metrics = self.resource_allocator.allocation_metrics
        self.framework_metrics = self.load_balancer.framework_metrics
        self.load_balancing_weights = self.load_balancer.load_balancing_weights
        self.scaling_signals = self.predictive_scaler.scaling_signals
        
        # Background task management
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
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
            'resource_safety_margin': 0.1,
            'max_allocation_attempts': 3,
            'enable_autonomous_scaling': True,
            'cost_optimization_enabled': True,
            'log_level': logging.INFO
        }
    
    def _extract_allocator_config(self) -> Dict[str, Any]:
        """Extract configuration for ResourceAllocator"""
        return {
            'default_allocation_strategy': self.config.get('default_allocation_strategy'),
            'allocation_timeout': self.config.get('allocation_timeout'),
            'rebalancing_interval': self.config.get('rebalancing_interval'),
            'resource_safety_margin': self.config.get('resource_safety_margin'),
            'max_allocation_attempts': self.config.get('max_allocation_attempts'),
            'log_level': self.config.get('log_level'),
            'load_balancing': {
                'load_balancing_algorithm': self.config.get('load_balancing_algorithm'),
                'rebalancing_interval': self.config.get('rebalancing_interval')
            },
            'predictive_scaling': {
                'enable_predictive_scaling': self.config.get('enable_predictive_scaling'),
                'enable_autonomous_scaling': self.config.get('enable_autonomous_scaling'),
                'scaling_threshold_up': self.config.get('scaling_threshold_up'),
                'scaling_threshold_down': self.config.get('scaling_threshold_down'),
                'prediction_horizon': self.config.get('prediction_horizon')
            }
        }
    
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
        return await self.resource_allocator.request_allocation(
            framework_id, resource_requirements, priority, urgency,
            duration, deadline, strategy
        )
    
    async def _allocation_processing_loop(self) -> None:
        """Background loop for processing allocation requests"""
        self.logger.info("Starting allocation processing loop")
        
        while self._running:
            try:
                # Process pending requests
                await self._process_allocation_batch()
                
                # Clean up expired allocations
                await self._cleanup_expired_allocations()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Allocation processing loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_allocation_batch(self) -> None:
        """Process a batch of allocation requests"""
        # Delegate to ResourceAllocator's internal batch processing
        # This maintains backward compatibility while using new modular system
        batch_size = 10
        requests_to_process = []
        
        for _ in range(min(batch_size, len(self.allocation_requests))):
            if self.allocation_requests:
                request = self.allocation_requests.popleft()
                if request.urgency < 0.8 and request.priority < 8:
                    requests_to_process.append(request)
        
        for request in requests_to_process:
            await self.resource_allocator._process_allocation_request(request)
    
    async def _cleanup_expired_allocations(self) -> None:
        """Clean up expired allocations"""
        current_time = datetime.now()
        expired_decisions = []
        
        for decision_id, decision in self.active_allocations.items():
            if current_time >= decision.expiration_time:
                expired_decisions.append(decision_id)
        
        for decision_id in expired_decisions:
            await self.resource_allocator.release_allocation(decision_id)
    
    async def _load_balancing_loop(self) -> None:
        """Background loop for load balancing optimization"""
        self.logger.info("Starting load balancing loop")
        
        while self._running:
            try:
                # Simulate framework metrics for testing
                frameworks = ['analytics', 'ml', 'api', 'analysis']
                await self.load_balancer.simulate_framework_metrics(frameworks)
                
                # Optimize load balancing
                await self.load_balancer.optimize_load_balancing()
                
                await asyncio.sleep(self.config['rebalancing_interval'].total_seconds())
                
            except Exception as e:
                self.logger.error(f"Load balancing loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _predictive_scaling_loop(self) -> None:
        """Background loop for predictive scaling"""
        if not self.config['enable_predictive_scaling']:
            return
        
        self.logger.info("Starting predictive scaling loop")
        
        while self._running:
            try:
                # Generate scaling signals
                for resource_type in self.available_resources:
                    current_allocated = self.resource_allocator._get_current_allocation(resource_type)
                    signal = self.predictive_scaler.generate_scaling_signal(
                        resource_type,
                        self.available_resources[resource_type],
                        current_allocated,
                        self.resource_constraints
                    )
                
                # Process scaling signals
                if self.config['enable_autonomous_scaling']:
                    actions = self.predictive_scaler.process_scaling_signals()
                    for action in actions:
                        direction = ScalingDirection(action['direction'])
                        self.predictive_scaler.execute_scaling_action(
                            action['resource_type'], direction
                        )
                
                await asyncio.sleep(60.0)
                
            except Exception as e:
                self.logger.error(f"Predictive scaling loop error: {e}")
                await asyncio.sleep(120.0)
    
    async def _resource_monitoring_loop(self) -> None:
        """Background loop for resource monitoring"""
        self.logger.info("Starting resource monitoring loop")
        
        while self._running:
            try:
                # Monitor resource health
                for resource_type in self.available_resources:
                    allocated = self.resource_allocator._get_current_allocation(resource_type)
                    total = self.available_resources[resource_type] + allocated
                    
                    if total > 0:
                        utilization = allocated / total
                        
                        if utilization > 0.9:
                            self.logger.warning(f"High utilization for {resource_type}: {utilization:.1%}")
                        elif utilization > 0.95:
                            self.logger.critical(f"Critical utilization for {resource_type}: {utilization:.1%}")
                
                await asyncio.sleep(30.0)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def get_allocation_status(self) -> Dict[str, Any]:
        """Get comprehensive allocation status"""
        status = await self.resource_allocator.get_status()
        
        # Add additional fields for backward compatibility
        status.update({
            'version': '2.0.0',
            'status': 'active' if self._running else 'inactive',
            'scaling_signals': len(self.scaling_signals),
            'configuration': self.config
        })
        
        return status
    
    async def get_framework_recommendation(self, task_requirements: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get framework recommendations for a task based on current load balancing"""
        return await self.resource_allocator.get_framework_recommendation(task_requirements)
    
    async def release_allocation(self, decision_id: str) -> bool:
        """Manually release an allocation"""
        return await self.resource_allocator.release_allocation(decision_id)


# Factory function for easy instantiation
def create_intelligent_resource_allocator(config: Dict[str, Any] = None) -> IntelligentResourceAllocator:
    """Create and return a configured Intelligent Resource Allocator"""
    return IntelligentResourceAllocator(config)


# Export main classes for backward compatibility
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