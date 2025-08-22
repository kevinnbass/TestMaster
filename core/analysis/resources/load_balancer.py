"""
Resource Intelligence Load Balancer
==================================

Advanced load balancing algorithms with predictive and performance-based routing.
Extracted from intelligent_resource_allocator.py for enterprise modular architecture.

Agent D Implementation - Hour 10-11: Revolutionary Intelligence Modularization
"""

import logging
import random
import time
from typing import Dict, List, Optional
from collections import defaultdict

from .data_models import LoadBalancingAlgorithm, LoadBalancingMetrics


class LoadBalancer:
    """Advanced load balancer with multiple algorithms and health scoring"""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.PERFORMANCE_BASED):
        self.algorithm = algorithm
        self.framework_metrics = {}
        self.round_robin_index = 0
        self.connection_counts = defaultdict(int)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Algorithm dispatch table
        self.algorithm_handlers = {
            LoadBalancingAlgorithm.ROUND_ROBIN: self._round_robin,
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: self._weighted_round_robin,
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: self._least_connections,
            LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS: self._weighted_least_connections,
            LoadBalancingAlgorithm.PERFORMANCE_BASED: self._performance_based,
            LoadBalancingAlgorithm.PREDICTIVE: self._predictive_routing
        }
    
    def update_framework_metrics(self, framework_id: str, metrics: LoadBalancingMetrics):
        """Update metrics for a framework"""
        self.framework_metrics[framework_id] = metrics
        self.logger.debug(f"Updated metrics for framework {framework_id}: "
                         f"load={metrics.current_load:.2f}, health={metrics.health_score:.2f}")
    
    def select_framework(self, available_frameworks: List[str], 
                        task_requirements: Optional[Dict] = None) -> Optional[str]:
        """Select best framework based on current algorithm"""
        if not available_frameworks:
            return None
        
        # Filter healthy frameworks
        healthy_frameworks = [
            fw for fw in available_frameworks 
            if self._is_framework_healthy(fw)
        ]
        
        if not healthy_frameworks:
            self.logger.warning("No healthy frameworks available")
            return None
        
        # Use algorithm-specific selection
        handler = self.algorithm_handlers.get(self.algorithm, self._round_robin)
        selected = handler(healthy_frameworks, task_requirements)
        
        if selected:
            self.connection_counts[selected] += 1
            self.logger.debug(f"Selected framework {selected} using {self.algorithm.value}")
        
        return selected
    
    def _is_framework_healthy(self, framework_id: str) -> bool:
        """Check if framework is healthy enough for load balancing"""
        if framework_id not in self.framework_metrics:
            return True  # Assume healthy if no metrics available
        
        metrics = self.framework_metrics[framework_id]
        
        # Health thresholds
        min_health_score = 0.3
        max_error_rate = 0.2
        max_utilization = 0.95
        
        return (metrics.health_score >= min_health_score and
                metrics.error_rate <= max_error_rate and
                metrics.utilization <= max_utilization)
    
    def _round_robin(self, frameworks: List[str], task_requirements: Optional[Dict] = None) -> str:
        """Simple round-robin selection"""
        if not frameworks:
            return None
        
        selected = frameworks[self.round_robin_index % len(frameworks)]
        self.round_robin_index = (self.round_robin_index + 1) % len(frameworks)
        return selected
    
    def _weighted_round_robin(self, frameworks: List[str], task_requirements: Optional[Dict] = None) -> str:
        """Weighted round-robin based on framework capacity"""
        if not frameworks:
            return None
        
        # Calculate weights based on effective capacity
        weights = []
        for fw in frameworks:
            if fw in self.framework_metrics:
                metrics = self.framework_metrics[fw]
                weight = metrics.effective_capacity() * metrics.weight
            else:
                weight = 1.0  # Default weight
            weights.append(weight)
        
        # Weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return self._round_robin(frameworks, task_requirements)
        
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return frameworks[i]
        
        return frameworks[-1]  # Fallback
    
    def _least_connections(self, frameworks: List[str], task_requirements: Optional[Dict] = None) -> str:
        """Select framework with least active connections"""
        if not frameworks:
            return None
        
        min_connections = float('inf')
        selected = frameworks[0]
        
        for fw in frameworks:
            connections = self.connection_counts[fw]
            if connections < min_connections:
                min_connections = connections
                selected = fw
        
        return selected
    
    def _weighted_least_connections(self, frameworks: List[str], task_requirements: Optional[Dict] = None) -> str:
        """Weighted least connections considering framework capacity"""
        if not frameworks:
            return None
        
        best_ratio = float('inf')
        selected = frameworks[0]
        
        for fw in frameworks:
            connections = self.connection_counts[fw]
            
            if fw in self.framework_metrics:
                metrics = self.framework_metrics[fw]
                capacity = metrics.effective_capacity()
                weight = metrics.weight
                
                # Connection ratio considering capacity and weight
                if capacity > 0 and weight > 0:
                    ratio = connections / (capacity * weight)
                else:
                    ratio = connections
            else:
                ratio = connections
            
            if ratio < best_ratio:
                best_ratio = ratio
                selected = fw
        
        return selected
    
    def _performance_based(self, frameworks: List[str], task_requirements: Optional[Dict] = None) -> str:
        """Select framework based on performance metrics"""
        if not frameworks:
            return None
        
        best_score = -1
        selected = frameworks[0]
        
        for fw in frameworks:
            if fw in self.framework_metrics:
                metrics = self.framework_metrics[fw]
                
                # Calculate performance score
                # Higher is better: health, throughput, available capacity
                # Lower is better: response_time, error_rate, utilization
                
                health_score = metrics.health_score
                utilization_score = 1.0 - metrics.utilization  # Lower utilization is better
                response_time_score = 1.0 / (1.0 + metrics.response_time)  # Lower response time is better
                error_rate_score = 1.0 - metrics.error_rate  # Lower error rate is better
                throughput_score = min(metrics.throughput / 1000.0, 1.0)  # Normalize throughput
                
                # Weighted combination
                performance_score = (
                    health_score * 0.3 +
                    utilization_score * 0.25 +
                    response_time_score * 0.2 +
                    error_rate_score * 0.15 +
                    throughput_score * 0.1
                )
                
                # Apply framework weight
                performance_score *= metrics.weight
                
            else:
                performance_score = 0.5  # Default score for unknown frameworks
            
            if performance_score > best_score:
                best_score = performance_score
                selected = fw
        
        return selected
    
    def _predictive_routing(self, frameworks: List[str], task_requirements: Optional[Dict] = None) -> str:
        """Predictive routing based on anticipated load and performance"""
        if not frameworks:
            return None
        
        current_time = time.time()
        best_predicted_score = -1
        selected = frameworks[0]
        
        for fw in frameworks:
            if fw in self.framework_metrics:
                metrics = self.framework_metrics[fw]
                
                # Predict future load based on current trends
                current_utilization = metrics.utilization
                current_response_time = metrics.response_time
                
                # Simple trend prediction (would be enhanced with ML in production)
                predicted_utilization = current_utilization
                predicted_response_time = current_response_time
                
                # Adjust predictions based on connection counts
                active_connections = self.connection_counts[fw]
                capacity = metrics.effective_capacity()
                
                if capacity > 0:
                    predicted_load_increase = active_connections / capacity * 0.1
                    predicted_utilization = min(1.0, current_utilization + predicted_load_increase)
                    predicted_response_time = current_response_time * (1 + predicted_load_increase)
                
                # Calculate predicted performance score
                health_score = metrics.health_score
                predicted_utilization_score = 1.0 - predicted_utilization
                predicted_response_score = 1.0 / (1.0 + predicted_response_time)
                error_rate_score = 1.0 - metrics.error_rate
                
                predicted_score = (
                    health_score * 0.3 +
                    predicted_utilization_score * 0.3 +
                    predicted_response_score * 0.25 +
                    error_rate_score * 0.15
                )
                
                # Apply framework weight
                predicted_score *= metrics.weight
                
            else:
                predicted_score = 0.5  # Default score
            
            if predicted_score > best_predicted_score:
                best_predicted_score = predicted_score
                selected = fw
        
        return selected
    
    def release_connection(self, framework_id: str):
        """Release a connection from framework"""
        if framework_id in self.connection_counts:
            self.connection_counts[framework_id] = max(0, self.connection_counts[framework_id] - 1)
    
    def get_load_balancing_stats(self) -> Dict:
        """Get current load balancing statistics"""
        stats = {
            'algorithm': self.algorithm.value,
            'total_frameworks': len(self.framework_metrics),
            'healthy_frameworks': len([fw for fw in self.framework_metrics 
                                     if self._is_framework_healthy(fw)]),
            'connection_distribution': dict(self.connection_counts),
            'framework_health': {}
        }
        
        for fw_id, metrics in self.framework_metrics.items():
            stats['framework_health'][fw_id] = {
                'health_score': metrics.health_score,
                'utilization': metrics.utilization,
                'response_time': metrics.response_time,
                'error_rate': metrics.error_rate,
                'effective_capacity': metrics.effective_capacity(),
                'is_healthy': self._is_framework_healthy(fw_id)
            }
        
        return stats
    
    def set_algorithm(self, algorithm: LoadBalancingAlgorithm):
        """Change load balancing algorithm"""
        self.algorithm = algorithm
        self.logger.info(f"Load balancing algorithm changed to {algorithm.value}")


def create_load_balancer(algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.PERFORMANCE_BASED) -> LoadBalancer:
    """Factory function to create load balancer"""
    return LoadBalancer(algorithm)