"""
Predictive Scaling Module
=========================

Handles predictive scaling decisions and autonomous resource scaling based on
demand forecasting and resource utilization patterns.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
from collections import deque, defaultdict

from .data_models import PredictiveScalingSignal, ScalingDirection, ResourceConstraint


class PredictiveScaler:
    """Predictive scaling manager for autonomous resource adjustment"""
    
    def __init__(self, config: Dict = None):
        """Initialize the predictive scaler"""
        self.config = config or self._get_default_config()
        
        # Resource tracking
        self.available_resources: Dict[str, float] = {}
        self.resource_utilization_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.scaling_signals: deque = deque(maxlen=50)
        self.demand_predictions: Dict[str, float] = {}
        
        # Performance tracking
        self.prediction_accuracy = 0.7  # Default accuracy
        self.scaling_actions_taken = 0
        self.successful_predictions = 0
        self.total_predictions = 0
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _get_default_config(self) -> Dict:
        """Get default predictive scaling configuration"""
        return {
            'enable_predictive_scaling': True,
            'enable_autonomous_scaling': True,
            'scaling_threshold_up': 0.8,
            'scaling_threshold_down': 0.3,
            'prediction_horizon': timedelta(hours=1),
            'min_history_for_prediction': 5,
            'scale_up_percentage': 0.2,
            'scale_down_percentage': 0.15,
            'confidence_threshold': 0.7
        }
    
    def update_resource_utilization(self, resource_type: str, allocation: float, timestamp: datetime) -> None:
        """Update resource utilization history"""
        self.resource_utilization_history[resource_type].append({
            'timestamp': timestamp,
            'allocation': allocation,
            'decision_id': f"util_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        })
    
    def generate_scaling_signal(self, 
                               resource_type: str,
                               current_available: float,
                               current_allocated: float,
                               constraints: Optional[List[ResourceConstraint]] = None) -> Optional[PredictiveScalingSignal]:
        """Generate a predictive scaling signal for a resource"""
        total_capacity = current_available + current_allocated
        
        if total_capacity <= 0:
            return None
        
        current_utilization = current_allocated / total_capacity
        
        # Predict future demand
        predicted_demand = self.predict_resource_demand(resource_type)
        predicted_utilization = predicted_demand / total_capacity
        
        # Determine scaling direction
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
                confidence=self._calculate_prediction_confidence(resource_type),
                time_horizon=self.config['prediction_horizon'],
                trend_direction=trend_direction,
                urgency=urgency,
                cost_impact=self._estimate_scaling_cost(resource_type, trend_direction, constraints)
            )
            
            self.scaling_signals.append(signal)
            return signal
        
        return None
    
    def predict_resource_demand(self, resource_type: str) -> float:
        """Predict future resource demand"""
        history = self.resource_utilization_history.get(resource_type, deque())
        
        if len(history) < self.config['min_history_for_prediction']:
            # Not enough history, return current or zero
            return history[-1]['allocation'] if history else 0.0
        
        # Extract recent allocations
        recent_allocations = [entry['allocation'] for entry in list(history)[-10:]]
        
        # Simple prediction methods
        prediction_method = self._select_prediction_method(recent_allocations)
        
        if prediction_method == 'linear_trend':
            return self._predict_linear_trend(recent_allocations)
        elif prediction_method == 'exponential_smoothing':
            return self._predict_exponential_smoothing(recent_allocations)
        elif prediction_method == 'moving_average':
            return self._predict_moving_average(recent_allocations)
        else:
            return recent_allocations[-1] if recent_allocations else 0.0
    
    def _select_prediction_method(self, data: List[float]) -> str:
        """Select the best prediction method based on data characteristics"""
        if len(data) < 2:
            return 'last_value'
        
        # Calculate variance and trend
        variance = np.var(data)
        trend = (data[-1] - data[0]) / len(data) if len(data) > 1 else 0
        
        # Select method based on characteristics
        if abs(trend) > 0.1 and variance < 10:
            return 'linear_trend'
        elif variance > 5:
            return 'exponential_smoothing'
        else:
            return 'moving_average'
    
    def _predict_linear_trend(self, data: List[float]) -> float:
        """Predict using linear trend extrapolation"""
        if len(data) < 2:
            return data[-1] if data else 0.0
        
        # Calculate trend
        x = np.arange(len(data))
        coefficients = np.polyfit(x, data, 1)
        
        # Predict next value
        next_x = len(data)
        predicted = coefficients[0] * next_x + coefficients[1]
        
        return max(0.0, predicted)
    
    def _predict_exponential_smoothing(self, data: List[float], alpha: float = 0.3) -> float:
        """Predict using exponential smoothing"""
        if not data:
            return 0.0
        
        # Simple exponential smoothing
        smoothed = data[0]
        for value in data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return max(0.0, smoothed)
    
    def _predict_moving_average(self, data: List[float], window: int = 3) -> float:
        """Predict using moving average"""
        if not data:
            return 0.0
        
        window = min(window, len(data))
        return np.mean(data[-window:])
    
    def _calculate_prediction_confidence(self, resource_type: str) -> float:
        """Calculate confidence in prediction"""
        history = self.resource_utilization_history.get(resource_type, deque())
        
        if len(history) < self.config['min_history_for_prediction']:
            return 0.5  # Low confidence with insufficient history
        
        # Base confidence on data consistency
        recent_allocations = [entry['allocation'] for entry in list(history)[-10:]]
        
        if len(recent_allocations) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        mean_allocation = np.mean(recent_allocations)
        std_allocation = np.std(recent_allocations)
        
        if mean_allocation > 0:
            cv = std_allocation / mean_allocation
            # Lower CV means more consistent data, higher confidence
            confidence = max(0.3, min(0.95, 1.0 - cv))
        else:
            confidence = 0.5
        
        # Adjust based on overall prediction accuracy
        confidence *= self.prediction_accuracy
        
        return confidence
    
    def _estimate_scaling_cost(self, 
                              resource_type: str,
                              direction: ScalingDirection,
                              constraints: Optional[List[ResourceConstraint]] = None) -> float:
        """Estimate cost of scaling operation"""
        # Find cost per unit from constraints
        cost_per_unit = 0.01  # Default
        
        if constraints:
            for constraint in constraints:
                if constraint.resource_type == resource_type:
                    cost_per_unit = constraint.cost_per_unit
                    break
        
        # Estimate scaling amount
        current_capacity = self.available_resources.get(resource_type, 100.0)
        
        if direction == ScalingDirection.SCALE_UP:
            scaling_amount = current_capacity * self.config['scale_up_percentage']
            return scaling_amount * cost_per_unit
        elif direction == ScalingDirection.SCALE_DOWN:
            scaling_amount = current_capacity * self.config['scale_down_percentage']
            return -scaling_amount * cost_per_unit * 0.5  # Savings (but with some cost)
        else:
            return 0.0
    
    def process_scaling_signals(self) -> List[Dict]:
        """Process scaling signals and return recommended actions"""
        if not self.config['enable_autonomous_scaling']:
            return []
        
        actions = []
        
        # Process high-urgency signals first
        sorted_signals = sorted(self.scaling_signals, key=lambda s: s.urgency, reverse=True)
        
        for signal in sorted_signals[:5]:  # Process top 5 signals
            if signal.scaling_needed() and signal.confidence >= self.config['confidence_threshold']:
                action = {
                    'resource_type': signal.resource_type,
                    'direction': signal.trend_direction.value,
                    'urgency': signal.urgency,
                    'confidence': signal.confidence,
                    'cost_impact': signal.cost_impact,
                    'predicted_demand': signal.predicted_demand,
                    'current_supply': signal.current_supply
                }
                actions.append(action)
        
        return actions
    
    def execute_scaling_action(self, resource_type: str, direction: ScalingDirection) -> bool:
        """Execute autonomous scaling action"""
        try:
            current_capacity = self.available_resources.get(resource_type, 0.0)
            
            if direction == ScalingDirection.SCALE_UP:
                # Increase available resources
                additional_capacity = current_capacity * self.config['scale_up_percentage']
                self.available_resources[resource_type] = current_capacity + additional_capacity
                
                self.logger.info(f"Scaled up {resource_type} by {additional_capacity}")
                self.scaling_actions_taken += 1
                return True
                
            elif direction == ScalingDirection.SCALE_DOWN:
                # Decrease available resources
                reduction = current_capacity * self.config['scale_down_percentage']
                self.available_resources[resource_type] = max(0.0, current_capacity - reduction)
                
                self.logger.info(f"Scaled down {resource_type} by {reduction}")
                self.scaling_actions_taken += 1
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Scaling action failed for {resource_type}: {e}")
            return False
    
    def update_prediction_accuracy(self, predicted_demand: float, actual_demand: float) -> None:
        """Update prediction accuracy based on actual vs predicted"""
        self.total_predictions += 1
        
        if predicted_demand > 0:
            error_rate = abs(actual_demand - predicted_demand) / predicted_demand
            if error_rate < 0.2:  # Within 20% is considered successful
                self.successful_predictions += 1
        
        if self.total_predictions > 0:
            self.prediction_accuracy = self.successful_predictions / self.total_predictions
    
    def get_metrics(self) -> Dict:
        """Get predictive scaling metrics"""
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'scaling_actions_taken': self.scaling_actions_taken,
            'successful_predictions': self.successful_predictions,
            'total_predictions': self.total_predictions,
            'pending_signals': len(self.scaling_signals),
            'config': self.config
        }
    
    def clear_signals(self) -> None:
        """Clear processed scaling signals"""
        self.scaling_signals.clear()


__all__ = ['PredictiveScaler']