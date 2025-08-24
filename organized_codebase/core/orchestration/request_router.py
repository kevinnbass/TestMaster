"""
Focused Request Router

Handles request routing, load balancing, and traffic management for the API orchestrator.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import random

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Request routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    HEALTH_BASED = "health_based"


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RoutingRequest:
    """Request routing information."""
    request_id: str
    service_type: str
    endpoint_path: str
    method: str
    priority: RequestPriority = RequestPriority.NORMAL
    headers: Dict[str, str] = field(default_factory=dict)
    payload: Any = None
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingTarget:
    """Target service for routing."""
    service_id: str
    endpoint_url: str
    weight: float = 1.0
    current_connections: int = 0
    total_requests: int = 0
    success_rate: float = 100.0
    response_time_ms: float = 0.0
    last_used: Optional[datetime] = None


class RequestRouter:
    """
    Focused request router for API orchestration.
    Handles intelligent routing, load balancing, and traffic management.
    """
    
    def __init__(self):
        """Initialize request router with routing configurations."""
        try:
            # Routing configuration
            self.routing_strategy = RoutingStrategy.HEALTH_BASED
            self.enable_load_balancing = True
            self.enable_circuit_breaker = True
            self.enable_rate_limiting = True
            
            # Request queues by priority
            self.request_queues = {
                RequestPriority.CRITICAL: deque(),
                RequestPriority.HIGH: deque(),
                RequestPriority.NORMAL: deque(),
                RequestPriority.LOW: deque()
            }
            
            # Routing tables
            self.routing_table = {}  # endpoint_path -> List[RoutingTarget]
            self.service_targets = defaultdict(list)  # service_type -> List[RoutingTarget]
            
            # Load balancing state
            self.round_robin_counters = defaultdict(int)
            self.connection_counts = defaultdict(int)
            
            # Circuit breaker state
            self.circuit_breakers = {}
            self.circuit_breaker_config = {
                'failure_threshold': 5,
                'recovery_timeout': 60,
                'half_open_max_calls': 3
            }
            
            # Rate limiting
            self.rate_limiters = {}
            self.rate_limit_windows = defaultdict(deque)
            
            # Metrics
            self.routing_metrics = {
                'total_requests': 0,
                'successful_routes': 0,
                'failed_routes': 0,
                'average_route_time': 0.0,
                'circuit_breaker_trips': 0,
                'rate_limit_hits': 0
            }
            
            logger.info("Request Router initialized")
        except Exception as e:
            logger.error(f"Failed to initialize request router: {e}")
            raise
    
    async def route_request(self, request: RoutingRequest) -> Dict[str, Any]:
        """
        Route a request to the best available service.
        
        Args:
            request: Request to route
            
        Returns:
            Routing result with target service information
        """
        try:
            start_time = time.time()
            self.routing_metrics['total_requests'] += 1
            
            # Check rate limiting
            if not self._check_rate_limit(request):
                self.routing_metrics['rate_limit_hits'] += 1
                return {
                    'status': 'rate_limited',
                    'error': 'Request rate limit exceeded',
                    'request_id': request.request_id
                }
            
            # Add to appropriate queue
            await self._enqueue_request(request)
            
            # Find routing targets
            targets = self._find_routing_targets(request)
            if not targets:
                self.routing_metrics['failed_routes'] += 1
                return {
                    'status': 'no_targets',
                    'error': 'No available routing targets',
                    'request_id': request.request_id
                }
            
            # Select best target using routing strategy
            target = await self._select_routing_target(targets, request)
            if not target:
                self.routing_metrics['failed_routes'] += 1
                return {
                    'status': 'target_selection_failed',
                    'error': 'Could not select routing target',
                    'request_id': request.request_id
                }
            
            # Check circuit breaker
            if not self._check_circuit_breaker(target.service_id):
                # Try alternative target
                alternative_targets = [t for t in targets if t.service_id != target.service_id]
                if alternative_targets:
                    target = await self._select_routing_target(alternative_targets, request)
                else:
                    self.routing_metrics['failed_routes'] += 1
                    return {
                        'status': 'circuit_breaker_open',
                        'error': 'Circuit breaker open for all targets',
                        'request_id': request.request_id
                    }
            
            # Update connection count and metrics
            self._update_routing_metrics(target, start_time)
            
            # Successful routing
            self.routing_metrics['successful_routes'] += 1
            
            return {
                'status': 'routed',
                'target_service_id': target.service_id,
                'target_endpoint': target.endpoint_url,
                'routing_strategy': self.routing_strategy.value,
                'route_time_ms': (time.time() - start_time) * 1000,
                'request_id': request.request_id
            }
            
        except Exception as e:
            logger.error(f"Failed to route request {request.request_id}: {e}")
            self.routing_metrics['failed_routes'] += 1
            return {
                'status': 'routing_error',
                'error': str(e),
                'request_id': request.request_id
            }
    
    def register_routing_target(self, service_id: str, service_type: str, 
                               endpoint_path: str, endpoint_url: str, 
                               weight: float = 1.0) -> bool:
        """
        Register a new routing target.
        
        Args:
            service_id: Unique service identifier
            service_type: Type of service
            endpoint_path: API endpoint path
            endpoint_url: Full endpoint URL
            weight: Routing weight for load balancing
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            target = RoutingTarget(
                service_id=service_id,
                endpoint_url=endpoint_url,
                weight=weight
            )
            
            # Add to routing table
            if endpoint_path not in self.routing_table:
                self.routing_table[endpoint_path] = []
            self.routing_table[endpoint_path].append(target)
            
            # Add to service targets
            self.service_targets[service_type].append(target)
            
            # Initialize circuit breaker
            self.circuit_breakers[service_id] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': None,
                'half_open_calls': 0
            }
            
            logger.info(f"Registered routing target: {service_id} -> {endpoint_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register routing target {service_id}: {e}")
            return False
    
    def unregister_routing_target(self, service_id: str, endpoint_path: str) -> bool:
        """
        Unregister a routing target.
        
        Args:
            service_id: Service identifier to unregister
            endpoint_path: Endpoint path to remove
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            # Remove from routing table
            if endpoint_path in self.routing_table:
                self.routing_table[endpoint_path] = [
                    t for t in self.routing_table[endpoint_path] 
                    if t.service_id != service_id
                ]
                
                # Clean up empty entries
                if not self.routing_table[endpoint_path]:
                    del self.routing_table[endpoint_path]
            
            # Remove from service targets
            for service_type, targets in self.service_targets.items():
                self.service_targets[service_type] = [
                    t for t in targets if t.service_id != service_id
                ]
            
            # Clean up circuit breaker
            if service_id in self.circuit_breakers:
                del self.circuit_breakers[service_id]
            
            logger.info(f"Unregistered routing target: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister routing target {service_id}: {e}")
            return False
    
    def set_routing_strategy(self, strategy: RoutingStrategy) -> bool:
        """
        Set the routing strategy for load balancing.
        
        Args:
            strategy: New routing strategy to use
            
        Returns:
            True if strategy set successfully, False otherwise
        """
        try:
            self.routing_strategy = strategy
            logger.info(f"Routing strategy set to: {strategy.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to set routing strategy: {e}")
            return False
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing statistics.
        
        Returns:
            Dictionary containing routing statistics
        """
        try:
            # Queue statistics
            queue_stats = {}
            for priority, queue in self.request_queues.items():
                queue_stats[priority.name.lower()] = len(queue)
            
            # Target statistics
            target_stats = {
                'total_targets': sum(len(targets) for targets in self.routing_table.values()),
                'targets_by_endpoint': {
                    endpoint: len(targets) 
                    for endpoint, targets in self.routing_table.items()
                }
            }
            
            # Circuit breaker statistics
            circuit_breaker_stats = {
                'total_breakers': len(self.circuit_breakers),
                'open_breakers': sum(
                    1 for cb in self.circuit_breakers.values() 
                    if cb['state'] == 'open'
                ),
                'half_open_breakers': sum(
                    1 for cb in self.circuit_breakers.values() 
                    if cb['state'] == 'half_open'
                )
            }
            
            return {
                'routing_metrics': self.routing_metrics,
                'queue_statistics': queue_stats,
                'target_statistics': target_stats,
                'circuit_breaker_statistics': circuit_breaker_stats,
                'current_strategy': self.routing_strategy.value,
                'features_enabled': {
                    'load_balancing': self.enable_load_balancing,
                    'circuit_breaker': self.enable_circuit_breaker,
                    'rate_limiting': self.enable_rate_limiting
                },
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get routing statistics: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }
    
    # Private helper methods
    async def _enqueue_request(self, request: RoutingRequest) -> None:
        """Add request to appropriate priority queue."""
        try:
            self.request_queues[request.priority].append(request)
        except Exception as e:
            logger.error(f"Failed to enqueue request {request.request_id}: {e}")
    
    def _find_routing_targets(self, request: RoutingRequest) -> List[RoutingTarget]:
        """Find available routing targets for a request."""
        try:
            # Look up by endpoint path first
            targets = self.routing_table.get(request.endpoint_path, [])
            
            # If no specific endpoint targets, try service type
            if not targets and request.service_type:
                targets = self.service_targets.get(request.service_type, [])
            
            # Filter out unhealthy targets
            healthy_targets = [
                target for target in targets 
                if self._is_target_healthy(target)
            ]
            
            return healthy_targets
            
        except Exception as e:
            logger.error(f"Failed to find routing targets: {e}")
            return []
    
    async def _select_routing_target(self, targets: List[RoutingTarget], 
                                   request: RoutingRequest) -> Optional[RoutingTarget]:
        """Select the best routing target using the configured strategy."""
        try:
            if not targets:
                return None
            
            if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(targets, request.endpoint_path)
            elif self.routing_strategy == RoutingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(targets)
            elif self.routing_strategy == RoutingStrategy.WEIGHTED_RANDOM:
                return self._weighted_random_selection(targets)
            elif self.routing_strategy == RoutingStrategy.HEALTH_BASED:
                return self._health_based_selection(targets)
            else:
                # Default to first available
                return targets[0]
                
        except Exception as e:
            logger.error(f"Failed to select routing target: {e}")
            return None
    
    def _round_robin_selection(self, targets: List[RoutingTarget], endpoint_path: str) -> RoutingTarget:
        """Select target using round-robin strategy."""
        counter = self.round_robin_counters[endpoint_path]
        target = targets[counter % len(targets)]
        self.round_robin_counters[endpoint_path] = counter + 1
        return target
    
    def _least_connections_selection(self, targets: List[RoutingTarget]) -> RoutingTarget:
        """Select target with least connections."""
        return min(targets, key=lambda t: t.current_connections)
    
    def _weighted_random_selection(self, targets: List[RoutingTarget]) -> RoutingTarget:
        """Select target using weighted random strategy."""
        weights = [target.weight for target in targets]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(targets)
        
        rand_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for target, weight in zip(targets, weights):
            cumulative_weight += weight
            if rand_value <= cumulative_weight:
                return target
        
        return targets[-1]  # Fallback
    
    def _health_based_selection(self, targets: List[RoutingTarget]) -> RoutingTarget:
        """Select target based on health and performance metrics."""
        # Score targets based on success rate and response time
        scored_targets = []
        for target in targets:
            # Health score (higher is better)
            health_score = target.success_rate / 100.0
            
            # Response time score (lower is better, normalized)
            if target.response_time_ms > 0:
                time_score = max(0, 1.0 - (target.response_time_ms / 1000.0))
            else:
                time_score = 1.0
            
            # Combined score
            combined_score = (health_score * 0.7) + (time_score * 0.3)
            scored_targets.append((target, combined_score))
        
        # Select target with highest score
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        return scored_targets[0][0]
    
    def _is_target_healthy(self, target: RoutingTarget) -> bool:
        """Check if a routing target is healthy."""
        try:
            # Check circuit breaker state
            if target.service_id in self.circuit_breakers:
                cb_state = self.circuit_breakers[target.service_id]['state']
                if cb_state == 'open':
                    return False
            
            # Check success rate
            if target.success_rate < 50.0:
                return False
            
            # Check response time
            if target.response_time_ms > 5000:  # 5 seconds
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking target health: {e}")
            return False
    
    def _check_rate_limit(self, request: RoutingRequest) -> bool:
        """Check if request passes rate limiting."""
        try:
            if not self.enable_rate_limiting:
                return True
            
            # Simple rate limiting implementation
            current_time = datetime.utcnow()
            window_key = f"{request.service_type}:{request.endpoint_path}"
            
            # Clean old entries (60-second window)
            cutoff_time = current_time - timedelta(seconds=60)
            window = self.rate_limit_windows[window_key]
            
            while window and window[0] < cutoff_time:
                window.popleft()
            
            # Check rate limit (default: 100 requests per minute)
            rate_limit = 100
            if len(window) >= rate_limit:
                return False
            
            # Add current request
            window.append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error
    
    def _check_circuit_breaker(self, service_id: str) -> bool:
        """Check if circuit breaker allows the request."""
        try:
            if not self.enable_circuit_breaker:
                return True
            
            if service_id not in self.circuit_breakers:
                return True
            
            cb = self.circuit_breakers[service_id]
            current_time = datetime.utcnow()
            
            if cb['state'] == 'closed':
                return True
            elif cb['state'] == 'open':
                # Check if recovery timeout has passed
                if (cb['last_failure_time'] and 
                    current_time - cb['last_failure_time'] > timedelta(seconds=self.circuit_breaker_config['recovery_timeout'])):
                    cb['state'] = 'half_open'
                    cb['half_open_calls'] = 0
                    return True
                return False
            elif cb['state'] == 'half_open':
                # Allow limited calls in half-open state
                if cb['half_open_calls'] < self.circuit_breaker_config['half_open_max_calls']:
                    cb['half_open_calls'] += 1
                    return True
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return True  # Allow on error
    
    def _update_routing_metrics(self, target: RoutingTarget, start_time: float) -> None:
        """Update routing and target metrics."""
        try:
            # Update connection count
            target.current_connections += 1
            target.total_requests += 1
            target.last_used = datetime.utcnow()
            
            # Update average route time
            route_time = (time.time() - start_time) * 1000
            current_avg = self.routing_metrics['average_route_time']
            total_requests = self.routing_metrics['total_requests']
            
            if total_requests > 1:
                self.routing_metrics['average_route_time'] = (
                    (current_avg * (total_requests - 1) + route_time) / total_requests
                )
            else:
                self.routing_metrics['average_route_time'] = route_time
                
        except Exception as e:
            logger.error(f"Error updating routing metrics: {e}")