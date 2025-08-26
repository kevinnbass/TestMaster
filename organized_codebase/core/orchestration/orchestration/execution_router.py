"""
Execution Router for TestMaster Flow Optimizer

Intelligent routing of tasks to optimal execution paths based on performance data.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import random

from core.feature_flags import FeatureFlags

class RoutingStrategy(Enum):
    """Routing strategies."""
    ROUND_ROBIN = "round_robin"
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"
    SHORTEST_QUEUE = "shortest_queue"

@dataclass
class RouteWeight:
    """Route weight configuration."""
    resource_id: str
    weight: float
    performance_score: float
    load_factor: float

@dataclass
class PerformanceData:
    """Performance data for routing decisions."""
    resource_id: str
    avg_response_time: float
    success_rate: float
    current_load: float
    availability: float
    last_updated: datetime

@dataclass
class Route:
    """Execution route."""
    task_id: str
    path: List[str]
    strategy: str
    confidence_score: float
    estimated_completion_time: float = 0.0
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = {}

class ExecutionRouter:
    """Execution router for optimal task routing."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'flow_optimizer')
        self.lock = threading.RLock()
        self.performance_data: Dict[str, PerformanceData] = {}
        self.routing_history: Dict[str, List[Route]] = {}
        self.adaptive_enabled = False
        self.learning_rate = 0.1
        self.route_weights: Dict[str, RouteWeight] = {}
        
        if not self.enabled:
            return
        
        print("Execution router initialized")
        print(f"   Available strategies: {[s.value for s in RoutingStrategy]}")
    
    def enable_adaptive_routing(self, learning_rate: float = 0.1):
        """Enable adaptive routing with machine learning."""
        self.adaptive_enabled = True
        self.learning_rate = learning_rate
        print(f"Adaptive routing enabled with learning rate: {learning_rate}")
    
    def find_optimal_route(
        self,
        task_id: str,
        available_resources: List[Dict[str, Any]],
        performance_history: Dict[str, Any] = None
    ) -> Route:
        """
        Find optimal execution route for a task.
        
        Args:
            task_id: Task identifier
            available_resources: Available execution resources
            performance_history: Historical performance data
            
        Returns:
            Optimal execution route
        """
        if not self.enabled:
            return Route(task_id, [], "disabled", 0.0)
        
        # Update performance data
        self._update_performance_data(available_resources, performance_history)
        
        # Select routing strategy
        strategy = self._select_routing_strategy(task_id, available_resources)
        
        # Generate route based on strategy
        route = self._generate_route(task_id, available_resources, strategy)
        
        # Store in history
        with self.lock:
            if task_id not in self.routing_history:
                self.routing_history[task_id] = []
            self.routing_history[task_id].append(route)
        
        print(f"Optimal route found for {task_id}: {strategy.value} strategy, confidence: {route.confidence_score:.3f}")
        
        return route
    
    def _update_performance_data(self, available_resources: List[Dict[str, Any]], performance_history: Dict[str, Any] = None):
        """Update performance data for routing decisions."""
        current_time = datetime.now()
        
        for resource in available_resources:
            resource_id = resource.get('id', 'unknown')
            
            # Extract performance metrics
            response_time = resource.get('response_time', 100.0)
            success_rate = resource.get('success_rate', 0.95)
            current_load = resource.get('current_load', 0.5)
            availability = resource.get('availability', 1.0)
            
            # Update performance data
            self.performance_data[resource_id] = PerformanceData(
                resource_id=resource_id,
                avg_response_time=response_time,
                success_rate=success_rate,
                current_load=current_load,
                availability=availability,
                last_updated=current_time
            )
    
    def _select_routing_strategy(self, task_id: str, available_resources: List[Dict[str, Any]]) -> RoutingStrategy:
        """Select optimal routing strategy."""
        if self.adaptive_enabled:
            return self._adaptive_strategy_selection(task_id, available_resources)
        
        # Default strategy selection logic
        resource_count = len(available_resources)
        avg_load = sum(r.get('current_load', 0.5) for r in available_resources) / max(resource_count, 1)
        
        if resource_count == 1:
            return RoutingStrategy.ROUND_ROBIN
        elif avg_load > 0.8:
            return RoutingStrategy.LOAD_BALANCED
        elif any(r.get('performance_score', 0.5) > 0.9 for r in available_resources):
            return RoutingStrategy.PERFORMANCE_BASED
        else:
            return RoutingStrategy.SHORTEST_QUEUE
    
    def _adaptive_strategy_selection(self, task_id: str, available_resources: List[Dict[str, Any]]) -> RoutingStrategy:
        """Adaptive strategy selection using performance history."""
        # Get historical performance for each strategy
        history = self.routing_history.get(task_id, [])
        strategy_performance = {}
        
        for route in history[-10:]:  # Consider last 10 routes
            strategy = route.strategy
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(route.confidence_score)
        
        # Select best performing strategy
        best_strategy = RoutingStrategy.PERFORMANCE_BASED  # Default
        best_score = 0.0
        
        for strategy_name, scores in strategy_performance.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                try:
                    best_strategy = RoutingStrategy(strategy_name)
                except ValueError:
                    # Handle case where strategy name doesn't match enum
                    continue
        
        return best_strategy
    
    def _generate_route(self, task_id: str, available_resources: List[Dict[str, Any]], strategy: RoutingStrategy) -> Route:
        """Generate route based on selected strategy."""
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_route(task_id, available_resources)
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._performance_based_route(task_id, available_resources)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return self._load_balanced_route(task_id, available_resources)
        elif strategy == RoutingStrategy.SHORTEST_QUEUE:
            return self._shortest_queue_route(task_id, available_resources)
        else:
            return self._adaptive_route(task_id, available_resources)
    
    def _round_robin_route(self, task_id: str, available_resources: List[Dict[str, Any]]) -> Route:
        """Generate round-robin route."""
        if not available_resources:
            return Route(task_id, [], "round_robin", 0.0)
        
        # Simple round-robin selection
        resource_index = hash(task_id) % len(available_resources)
        selected_resource = available_resources[resource_index]
        
        path = [selected_resource.get('id', 'unknown')]
        confidence = 0.7  # Moderate confidence for round-robin
        
        return Route(
            task_id=task_id,
            path=path,
            strategy="round_robin",
            confidence_score=confidence,
            estimated_completion_time=selected_resource.get('response_time', 100.0)
        )
    
    def _performance_based_route(self, task_id: str, available_resources: List[Dict[str, Any]]) -> Route:
        """Generate performance-based route."""
        if not available_resources:
            return Route(task_id, [], "performance_based", 0.0)
        
        # Select resource with best performance score
        best_resource = max(available_resources, key=lambda r: r.get('performance_score', 0.0))
        
        path = [best_resource.get('id', 'unknown')]
        confidence = best_resource.get('performance_score', 0.5)
        
        return Route(
            task_id=task_id,
            path=path,
            strategy="performance_based",
            confidence_score=confidence,
            estimated_completion_time=best_resource.get('response_time', 100.0)
        )
    
    def _load_balanced_route(self, task_id: str, available_resources: List[Dict[str, Any]]) -> Route:
        """Generate load-balanced route."""
        if not available_resources:
            return Route(task_id, [], "load_balanced", 0.0)
        
        # Select resource with lowest current load
        best_resource = min(available_resources, key=lambda r: r.get('current_load', 1.0))
        
        path = [best_resource.get('id', 'unknown')]
        confidence = 1.0 - best_resource.get('current_load', 0.5)
        
        return Route(
            task_id=task_id,
            path=path,
            strategy="load_balanced",
            confidence_score=confidence,
            estimated_completion_time=best_resource.get('response_time', 100.0) * (1 + best_resource.get('current_load', 0.5))
        )
    
    def _shortest_queue_route(self, task_id: str, available_resources: List[Dict[str, Any]]) -> Route:
        """Generate shortest queue route."""
        if not available_resources:
            return Route(task_id, [], "shortest_queue", 0.0)
        
        # Select resource with shortest queue
        best_resource = min(available_resources, key=lambda r: r.get('queue_length', 0))
        
        path = [best_resource.get('id', 'unknown')]
        queue_length = best_resource.get('queue_length', 0)
        confidence = max(0.1, 1.0 - (queue_length / 10.0))  # Confidence decreases with queue length
        
        return Route(
            task_id=task_id,
            path=path,
            strategy="shortest_queue",
            confidence_score=confidence,
            estimated_completion_time=best_resource.get('response_time', 100.0) * (1 + queue_length * 0.1)
        )
    
    def _adaptive_route(self, task_id: str, available_resources: List[Dict[str, Any]]) -> Route:
        """Generate adaptive route using multiple factors."""
        if not available_resources:
            return Route(task_id, [], "adaptive", 0.0)
        
        # Score each resource based on multiple factors
        resource_scores = []
        
        for resource in available_resources:
            performance_score = resource.get('performance_score', 0.5)
            load_factor = 1.0 - resource.get('current_load', 0.5)
            availability_factor = resource.get('availability', 1.0)
            response_time_factor = max(0.1, 1.0 - (resource.get('response_time', 100.0) / 1000.0))
            
            # Weighted combination of factors
            combined_score = (
                performance_score * 0.3 +
                load_factor * 0.3 +
                availability_factor * 0.2 +
                response_time_factor * 0.2
            )
            
            resource_scores.append((resource, combined_score))
        
        # Select best scoring resource
        best_resource, best_score = max(resource_scores, key=lambda x: x[1])
        
        path = [best_resource.get('id', 'unknown')]
        
        return Route(
            task_id=task_id,
            path=path,
            strategy="adaptive",
            confidence_score=best_score,
            estimated_completion_time=best_resource.get('response_time', 100.0)
        )
    
    def update_route_performance(self, task_id: str, route: Route, actual_performance: Dict[str, Any]):
        """Update route performance for adaptive learning."""
        if not self.adaptive_enabled:
            return
        
        actual_time = actual_performance.get('completion_time', route.estimated_completion_time)
        success = actual_performance.get('success', True)
        
        # Update route weights based on performance
        for resource_id in route.path:
            if resource_id in self.route_weights:
                weight = self.route_weights[resource_id]
                
                # Adjust weight based on performance
                if success and actual_time <= route.estimated_completion_time:
                    weight.weight += self.learning_rate * 0.1
                else:
                    weight.weight -= self.learning_rate * 0.05
                
                # Keep weight in reasonable bounds
                weight.weight = max(0.1, min(2.0, weight.weight))
        
        print(f"Route performance updated for {task_id}: success={success}, time={actual_time:.1f}ms")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_routes = sum(len(routes) for routes in self.routing_history.values())
        
        strategy_counts = {}
        avg_confidence = 0.0
        
        for routes in self.routing_history.values():
            for route in routes:
                strategy = route.strategy
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                avg_confidence += route.confidence_score
        
        if total_routes > 0:
            avg_confidence /= total_routes
        
        return {
            "total_routes": total_routes,
            "strategy_distribution": strategy_counts,
            "average_confidence": avg_confidence,
            "adaptive_enabled": self.adaptive_enabled,
            "performance_data_points": len(self.performance_data)
        }
    
    def shutdown(self):
        """Shutdown execution router."""
        self.adaptive_enabled = False
        print("Execution router shutdown completed")

def get_execution_router() -> ExecutionRouter:
    """Get execution router instance."""
    return ExecutionRouter()