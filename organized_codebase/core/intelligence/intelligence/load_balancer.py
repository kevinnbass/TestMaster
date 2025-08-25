"""
Load Balancing Module
=====================

Handles load balancing optimization across frameworks including performance-based,
predictive, and weighted algorithms.
"""

import logging
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
from dataclasses import asdict

from .data_models import LoadBalancingMetrics, LoadBalancingAlgorithm


class LoadBalancer:
    """Load balancing manager for optimizing resource distribution"""
    
    def __init__(self, config: Dict = None):
        """Initialize the load balancer"""
        self.config = config or self._get_default_config()
        self.framework_metrics: Dict[str, LoadBalancingMetrics] = {}
        self.load_balancing_weights: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.load_balancing_efficiency = 0.0
    
    def _get_default_config(self) -> Dict:
        """Get default load balancing configuration"""
        return {
            'load_balancing_algorithm': LoadBalancingAlgorithm.PERFORMANCE_BASED,
            'rebalancing_interval': timedelta(minutes=5),
            'min_weight': 0.1,
            'max_weight': 10.0
        }
    
    async def update_framework_metrics(self, framework_id: str, metrics: LoadBalancingMetrics) -> None:
        """Update metrics for a framework"""
        self.framework_metrics[framework_id] = metrics
        self.logger.debug(f"Updated metrics for framework {framework_id}")
    
    async def optimize_load_balancing(self) -> None:
        """Optimize load balancing across frameworks"""
        if not self.framework_metrics:
            return
        
        algorithm = self.config['load_balancing_algorithm']
        
        if algorithm == LoadBalancingAlgorithm.PERFORMANCE_BASED:
            await self._performance_based_load_balancing()
        elif algorithm == LoadBalancingAlgorithm.PREDICTIVE:
            await self._predictive_load_balancing()
        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
            await self._weighted_least_connections()
        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            await self._least_connections()
        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            await self._weighted_round_robin()
        else:
            await self._round_robin()
        
        # Update efficiency metrics
        await self._update_load_balancing_efficiency()
    
    async def _performance_based_load_balancing(self) -> None:
        """Performance-based load balancing optimization"""
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
        self._normalize_weights(performance_scores)
    
    async def _predictive_load_balancing(self) -> None:
        """Predictive load balancing using forecasted demand"""
        predicted_loads = {}
        
        for framework_id, metrics in self.framework_metrics.items():
            # Simple prediction based on current trend
            current_load = metrics.current_load
            predicted_change = np.random.uniform(-0.1, 0.1)  # Simulate prediction
            predicted_load = max(0.0, min(1.0, current_load + predicted_change))
            predicted_loads[framework_id] = predicted_load
        
        # Adjust weights based on predicted loads
        weights = {}
        for framework_id, predicted_load in predicted_loads.items():
            # Lower weight for frameworks with predicted high load
            weight = max(self.config['min_weight'], 1.0 - predicted_load)
            weights[framework_id] = weight
        
        self._normalize_weights(weights)
    
    async def _weighted_least_connections(self) -> None:
        """Weighted least connections load balancing"""
        weights = {}
        
        for framework_id, metrics in self.framework_metrics.items():
            # Weight based on capacity and current connections
            effective_capacity = metrics.effective_capacity()
            connection_ratio = 1.0 - metrics.utilization
            
            weight = effective_capacity * connection_ratio
            weights[framework_id] = max(self.config['min_weight'], weight)
        
        self._normalize_weights(weights)
    
    async def _least_connections(self) -> None:
        """Simple least connections load balancing"""
        weights = {}
        
        for framework_id, metrics in self.framework_metrics.items():
            # Weight inversely proportional to current load
            weight = 1.0 - metrics.current_load
            weights[framework_id] = max(self.config['min_weight'], weight)
        
        self._normalize_weights(weights)
    
    async def _weighted_round_robin(self) -> None:
        """Weighted round robin based on capacity and health"""
        weights = {}
        
        for framework_id, metrics in self.framework_metrics.items():
            # Weight based on effective capacity
            effective_capacity = metrics.effective_capacity()
            weight = effective_capacity / 100.0  # Normalize to 0-1
            
            # Adjust for current utilization
            utilization_factor = max(0.1, 1.0 - metrics.utilization)
            weight *= utilization_factor
            
            weights[framework_id] = weight
        
        self._normalize_weights(weights)
    
    async def _round_robin(self) -> None:
        """Simple round robin - equal weights"""
        equal_weight = 1.0 / len(self.framework_metrics) if self.framework_metrics else 0.0
        
        for framework_id in self.framework_metrics:
            self.load_balancing_weights[framework_id] = equal_weight
    
    def _normalize_weights(self, weights: Dict[str, float]) -> None:
        """Normalize weights to sum to 1.0"""
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            for framework_id, weight in weights.items():
                normalized_weight = weight / total_weight
                # Apply min/max bounds
                normalized_weight = max(
                    self.config['min_weight'] / len(weights),
                    min(self.config['max_weight'] / len(weights), normalized_weight)
                )
                self.load_balancing_weights[framework_id] = normalized_weight
        else:
            # Equal weights if no valid scores
            equal_weight = 1.0 / len(weights) if weights else 0.0
            for framework_id in weights:
                self.load_balancing_weights[framework_id] = equal_weight
    
    async def _update_load_balancing_efficiency(self) -> None:
        """Update load balancing efficiency metrics"""
        if not self.load_balancing_weights:
            return
        
        # Calculate load balancing efficiency
        weight_variance = np.var(list(self.load_balancing_weights.values()))
        max_variance = 0.25  # Maximum possible variance for uniform distribution
        
        # Efficiency is inverse of variance (lower variance = better balance)
        efficiency = max(0.0, 1.0 - (weight_variance / max_variance))
        self.load_balancing_efficiency = efficiency
    
    def get_framework_recommendation(self, task_requirements: Dict[str, float]) -> List[Tuple[str, float]]:
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
    
    def get_metrics(self) -> Dict:
        """Get current load balancing metrics"""
        return {
            'framework_metrics': {
                fid: asdict(metrics) for fid, metrics in self.framework_metrics.items()
            },
            'load_balancing_weights': dict(self.load_balancing_weights),
            'load_balancing_efficiency': self.load_balancing_efficiency,
            'algorithm': self.config['load_balancing_algorithm'].value
        }
    
    async def simulate_framework_metrics(self, frameworks: List[str]) -> None:
        """Simulate framework metrics for testing"""
        for framework_id in frameworks:
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


__all__ = ['LoadBalancer']