"""
Load Balancer - Advanced Load Balancing and Predictive Scaling Engine
=====================================================================

Enterprise-grade load balancing system with multiple algorithms, predictive scaling,
and intelligent health monitoring for distributed intelligence frameworks.
Implements advanced load balancing patterns for high-availability systems.

This module provides comprehensive load balancing capabilities including weighted
algorithms, predictive scaling, and autonomous health management.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: load_balancer.py (400 lines)
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import asdict
import json

from .resource_allocation_types import (
    LoadBalancingMetrics, PredictiveScalingSignal, ScalingDirection,
    LoadBalancingAlgorithm, AllocationStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadBalancingEngine:
    """Advanced load balancing engine with multiple algorithms and prediction"""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN):
        self.algorithm = algorithm
        self.framework_metrics: Dict[str, LoadBalancingMetrics] = {}
        self.request_history: deque = deque(maxlen=1000)
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.current_index = 0  # For round-robin algorithms
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Algorithm-specific state
        self.connection_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    def register_framework(self, framework_id: str, initial_metrics: LoadBalancingMetrics) -> None:
        """Register a framework for load balancing"""
        self.framework_metrics[framework_id] = initial_metrics
        self.load_history[framework_id] = deque(maxlen=100)
        self.connection_counts[framework_id] = 0
        self.response_times[framework_id] = deque(maxlen=50)
        
        self.logger.info(f"Registered framework for load balancing: {framework_id}")
    
    def update_framework_metrics(self, framework_id: str, metrics: LoadBalancingMetrics) -> None:
        """Update metrics for a framework"""
        if framework_id in self.framework_metrics:
            # Store historical data
            old_metrics = self.framework_metrics[framework_id]
            self.load_history[framework_id].append({
                'timestamp': datetime.now(),
                'load': old_metrics.current_load,
                'utilization': old_metrics.utilization,
                'response_time': old_metrics.response_time
            })
            
            # Update current metrics
            self.framework_metrics[framework_id] = metrics
            
            # Update response time tracking
            self.response_times[framework_id].append(metrics.response_time)
            
        else:
            self.logger.warning(f"Framework {framework_id} not registered for load balancing")
    
    async def select_framework(self, task_requirements: Dict[str, Any] = None) -> Optional[str]:
        """Select optimal framework based on load balancing algorithm"""
        available_frameworks = [
            fid for fid, metrics in self.framework_metrics.items()
            if metrics.health_score > 0.3  # Only consider healthy frameworks
        ]
        
        if not available_frameworks:
            return None
        
        # Apply load balancing algorithm
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_selection(available_frameworks)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_frameworks)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_frameworks)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
            return self._weighted_least_connections_selection(available_frameworks)
        elif self.algorithm == LoadBalancingAlgorithm.PERFORMANCE_BASED:
            return self._performance_based_selection(available_frameworks)
        elif self.algorithm == LoadBalancingAlgorithm.PREDICTIVE:
            return await self._predictive_selection(available_frameworks, task_requirements)
        elif self.algorithm == LoadBalancingAlgorithm.MACHINE_LEARNING:
            return await self._ml_based_selection(available_frameworks, task_requirements)
        else:
            return self._adaptive_weighted_selection(available_frameworks)
    
    def _round_robin_selection(self, frameworks: List[str]) -> str:
        """Simple round-robin selection"""
        if not frameworks:
            return None
        
        selected = frameworks[self.current_index % len(frameworks)]
        self.current_index += 1
        return selected
    
    def _weighted_round_robin_selection(self, frameworks: List[str]) -> str:
        """Weighted round-robin based on framework weights"""
        if not frameworks:
            return None
        
        # Calculate weights based on capacity and health
        weights = []
        for fid in frameworks:
            metrics = self.framework_metrics[fid]
            weight = metrics.effective_capacity() * metrics.weight
            weights.append(weight)
        
        # Weighted selection
        if sum(weights) > 0:
            weights = np.array(weights) / sum(weights)
            selected_idx = np.random.choice(len(frameworks), p=weights)
            return frameworks[selected_idx]
        else:
            return frameworks[0]
    
    def _least_connections_selection(self, frameworks: List[str]) -> str:
        """Select framework with least active connections"""
        min_connections = float('inf')
        selected_framework = None
        
        for fid in frameworks:
            connections = self.connection_counts[fid]
            if connections < min_connections:
                min_connections = connections
                selected_framework = fid
        
        return selected_framework
    
    def _weighted_least_connections_selection(self, frameworks: List[str]) -> str:
        """Weighted least connections considering capacity"""
        best_ratio = float('inf')
        selected_framework = None
        
        for fid in frameworks:
            metrics = self.framework_metrics[fid]
            connections = self.connection_counts[fid]
            effective_capacity = metrics.effective_capacity()
            
            if effective_capacity > 0:
                ratio = connections / effective_capacity
                if ratio < best_ratio:
                    best_ratio = ratio
                    selected_framework = fid
        
        return selected_framework or frameworks[0]
    
    def _performance_based_selection(self, frameworks: List[str]) -> str:
        """Select based on comprehensive performance metrics"""
        best_score = float('-inf')
        selected_framework = None
        
        for fid in frameworks:
            metrics = self.framework_metrics[fid]
            score = metrics.calculate_load_score()
            
            if score > best_score:
                best_score = score
                selected_framework = fid
        
        return selected_framework
    
    async def _predictive_selection(self, frameworks: List[str], 
                                  task_requirements: Dict[str, Any] = None) -> str:
        """Predictive selection based on anticipated load"""
        predictions = {}
        
        for fid in frameworks:
            # Predict future load based on trends
            predicted_load = await self._predict_future_load(fid)
            predictions[fid] = predicted_load
        
        # Select framework with lowest predicted load
        best_framework = min(predictions.keys(), key=lambda fid: predictions[fid])
        return best_framework
    
    async def _ml_based_selection(self, frameworks: List[str],
                                task_requirements: Dict[str, Any] = None) -> str:
        """Machine learning based selection (simplified)"""
        # In a full implementation, this would use trained ML models
        # For now, use sophisticated heuristics
        
        scores = {}
        
        for fid in frameworks:
            metrics = self.framework_metrics[fid]
            
            # Calculate multi-factor score
            load_factor = 1.0 - metrics.current_load
            health_factor = metrics.health_score
            performance_factor = 1.0 / (1.0 + metrics.response_time / 1000.0)
            utilization_factor = 1.0 - metrics.utilization
            
            # Weighted combination
            score = (load_factor * 0.3 + health_factor * 0.3 + 
                    performance_factor * 0.25 + utilization_factor * 0.15)
            
            scores[fid] = score
        
        # Select framework with highest score
        best_framework = max(scores.keys(), key=lambda fid: scores[fid])
        return best_framework
    
    def _adaptive_weighted_selection(self, frameworks: List[str]) -> str:
        """Adaptive weighted selection that learns from performance"""
        adaptive_weights = {}
        
        for fid in frameworks:
            metrics = self.framework_metrics[fid]
            
            # Base weight from metrics
            base_weight = metrics.weight
            
            # Adjust based on recent performance
            recent_performance = self._calculate_recent_performance(fid)
            adaptive_weight = base_weight * recent_performance
            
            adaptive_weights[fid] = adaptive_weight
        
        # Weighted selection
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            weights = [adaptive_weights[fid] / total_weight for fid in frameworks]
            selected_idx = np.random.choice(len(frameworks), p=weights)
            return frameworks[selected_idx]
        else:
            return frameworks[0]
    
    def _calculate_recent_performance(self, framework_id: str) -> float:
        """Calculate recent performance score for adaptive weighting"""
        if framework_id not in self.response_times:
            return 1.0
        
        recent_times = list(self.response_times[framework_id])
        if not recent_times:
            return 1.0
        
        # Lower response times = better performance
        avg_response_time = np.mean(recent_times)
        performance_score = max(0.1, 1.0 - min(1.0, avg_response_time / 1000.0))
        
        return performance_score
    
    async def _predict_future_load(self, framework_id: str, 
                                 horizon_minutes: int = 5) -> float:
        """Predict future load for a framework"""
        if framework_id not in self.load_history:
            return 0.5  # Default prediction
        
        history = list(self.load_history[framework_id])
        if len(history) < 3:
            return self.framework_metrics[framework_id].current_load
        
        # Simple trend analysis
        recent_loads = [entry['load'] for entry in history[-10:]]
        
        # Linear trend prediction
        x = np.arange(len(recent_loads))
        coeffs = np.polyfit(x, recent_loads, 1)
        predicted_load = coeffs[0] * len(recent_loads) + coeffs[1]
        
        # Ensure reasonable bounds
        return max(0.0, min(1.0, predicted_load))
    
    def record_request(self, framework_id: str, request_data: Dict[str, Any]) -> None:
        """Record a request for analytics and learning"""
        self.request_history.append({
            'timestamp': datetime.now(),
            'framework_id': framework_id,
            'request_data': request_data
        })
        
        # Increment connection count
        self.connection_counts[framework_id] += 1
    
    def record_completion(self, framework_id: str, response_time: float, 
                         success: bool = True) -> None:
        """Record request completion"""
        # Decrement connection count
        if framework_id in self.connection_counts:
            self.connection_counts[framework_id] = max(0, 
                self.connection_counts[framework_id] - 1)
        
        # Update response time tracking
        if framework_id in self.response_times:
            self.response_times[framework_id].append(response_time)
    
    async def get_load_balancing_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancing status"""
        framework_status = {}
        
        for fid, metrics in self.framework_metrics.items():
            framework_status[fid] = {
                'metrics': asdict(metrics),
                'connections': self.connection_counts[fid],
                'avg_response_time': np.mean(list(self.response_times[fid])) if self.response_times[fid] else 0.0,
                'recent_load_trend': await self._predict_future_load(fid),
                'performance_score': self._calculate_recent_performance(fid)
            }
        
        return {
            'algorithm': self.algorithm.value,
            'total_frameworks': len(self.framework_metrics),
            'active_connections': sum(self.connection_counts.values()),
            'total_requests_processed': len(self.request_history),
            'framework_status': framework_status,
            'overall_health': self._calculate_overall_health()
        }
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score"""
        if not self.framework_metrics:
            return 0.0
        
        health_scores = [metrics.health_score for metrics in self.framework_metrics.values()]
        return np.mean(health_scores)


class PredictiveScaler:
    """Predictive scaling engine for autonomous resource management"""
    
    def __init__(self, prediction_window: timedelta = timedelta(minutes=15)):
        self.prediction_window = prediction_window
        self.scaling_history: List[Dict[str, Any]] = []
        self.resource_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Scaling thresholds
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.confidence_threshold = 0.7
    
    def record_resource_usage(self, resource_type: str, usage: float, 
                            capacity: float, timestamp: datetime = None) -> None:
        """Record resource usage for trend analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        utilization = usage / capacity if capacity > 0 else 0.0
        
        self.resource_trends[resource_type].append({
            'timestamp': timestamp,
            'usage': usage,
            'capacity': capacity,
            'utilization': utilization
        })
    
    async def generate_scaling_signals(self) -> List[PredictiveScalingSignal]:
        """Generate predictive scaling signals"""
        signals = []
        
        for resource_type, trend_data in self.resource_trends.items():
            if len(trend_data) < 5:  # Need minimum data for prediction
                continue
            
            signal = await self._analyze_resource_trend(resource_type, list(trend_data))
            if signal and signal.scaling_needed():
                signals.append(signal)
        
        return signals
    
    async def _analyze_resource_trend(self, resource_type: str, 
                                    trend_data: List[Dict[str, Any]]) -> Optional[PredictiveScalingSignal]:
        """Analyze trend for a specific resource type"""
        try:
            # Extract recent utilization data
            recent_data = trend_data[-20:]  # Last 20 data points
            utilizations = [entry['utilization'] for entry in recent_data]
            timestamps = [entry['timestamp'] for entry in recent_data]
            
            # Calculate trend
            current_utilization = utilizations[-1]
            predicted_utilization = await self._predict_utilization(utilizations, timestamps)
            
            # Determine scaling direction
            scaling_direction = ScalingDirection.MAINTAIN
            urgency = 0.0
            
            if predicted_utilization > self.scale_up_threshold:
                scaling_direction = ScalingDirection.SCALE_UP
                urgency = min(1.0, (predicted_utilization - self.scale_up_threshold) / 0.2)
            elif predicted_utilization < self.scale_down_threshold:
                scaling_direction = ScalingDirection.SCALE_DOWN
                urgency = min(1.0, (self.scale_down_threshold - predicted_utilization) / 0.3)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(utilizations)
            
            if confidence >= self.confidence_threshold and scaling_direction != ScalingDirection.MAINTAIN:
                current_capacity = recent_data[-1]['capacity']
                
                return PredictiveScalingSignal(
                    resource_type=resource_type,
                    predicted_demand=predicted_utilization * current_capacity,
                    current_supply=current_capacity,
                    confidence=confidence,
                    time_horizon=self.prediction_window,
                    trend_direction=scaling_direction,
                    urgency=urgency,
                    cost_impact=self._estimate_scaling_cost(resource_type, scaling_direction),
                    prediction_model="trend_analysis",
                    historical_accuracy=self._get_historical_accuracy(resource_type)
                )
        
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {resource_type}: {e}")
        
        return None
    
    async def _predict_utilization(self, utilizations: List[float], 
                                 timestamps: List[datetime]) -> float:
        """Predict future utilization based on trend analysis"""
        if len(utilizations) < 3:
            return utilizations[-1] if utilizations else 0.0
        
        # Simple linear trend prediction
        x = np.arange(len(utilizations))
        coeffs = np.polyfit(x, utilizations, 1)
        
        # Predict for next time step
        predicted = coeffs[0] * len(utilizations) + coeffs[1]
        
        # Apply bounds and smoothing
        current = utilizations[-1]
        smoothing_factor = 0.3
        predicted = current + smoothing_factor * (predicted - current)
        
        return max(0.0, min(1.0, predicted))
    
    def _calculate_prediction_confidence(self, utilizations: List[float]) -> float:
        """Calculate confidence in prediction based on data quality"""
        if len(utilizations) < 3:
            return 0.5
        
        # Factors affecting confidence
        data_points = len(utilizations)
        variance = np.var(utilizations)
        trend_consistency = self._calculate_trend_consistency(utilizations)
        
        # Combine factors
        data_factor = min(1.0, data_points / 10.0)
        variance_factor = max(0.1, 1.0 - min(1.0, variance))
        consistency_factor = trend_consistency
        
        confidence = (data_factor * 0.3 + variance_factor * 0.4 + consistency_factor * 0.3)
        return max(0.1, min(1.0, confidence))
    
    def _calculate_trend_consistency(self, utilizations: List[float]) -> float:
        """Calculate consistency of trend direction"""
        if len(utilizations) < 3:
            return 0.5
        
        # Calculate direction changes
        directions = []
        for i in range(1, len(utilizations)):
            if utilizations[i] > utilizations[i-1]:
                directions.append(1)
            elif utilizations[i] < utilizations[i-1]:
                directions.append(-1)
            else:
                directions.append(0)
        
        if not directions:
            return 0.5
        
        # Count consistent directions
        most_common_direction = max(set(directions), key=directions.count)
        consistency_ratio = directions.count(most_common_direction) / len(directions)
        
        return consistency_ratio
    
    def _estimate_scaling_cost(self, resource_type: str, 
                             direction: ScalingDirection) -> float:
        """Estimate cost impact of scaling action"""
        # Simplified cost estimation
        base_cost = 10.0  # Base cost per scaling unit
        
        if direction == ScalingDirection.SCALE_UP:
            return base_cost * 1.0  # Cost to scale up
        elif direction == ScalingDirection.SCALE_DOWN:
            return -base_cost * 0.7  # Savings from scaling down
        else:
            return 0.0
    
    def _get_historical_accuracy(self, resource_type: str) -> float:
        """Get historical prediction accuracy for resource type"""
        # In a full implementation, this would track actual vs predicted
        # For now, return a reasonable default
        return 0.75
    
    def record_scaling_action(self, signal: PredictiveScalingSignal, 
                            action_taken: bool, actual_outcome: Dict[str, Any] = None) -> None:
        """Record scaling action for learning and improvement"""
        self.scaling_history.append({
            'timestamp': datetime.now(),
            'signal': asdict(signal),
            'action_taken': action_taken,
            'actual_outcome': actual_outcome or {}
        })
        
        # Keep history manageable
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]


# Export main classes
__all__ = [
    'LoadBalancingEngine', 'PredictiveScaler'
]