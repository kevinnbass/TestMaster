"""
Resource Intelligence Predictive Scaler
=======================================

Predictive scaling system with demand forecasting and autonomous scaling capabilities.
Extracted from intelligent_resource_allocator.py for enterprise modular architecture.

Agent D Implementation - Hour 10-11: Revolutionary Intelligence Modularization
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import PredictiveScalingSignal, ScalingDirection, ScalingAction


class PredictiveScaler:
    """Advanced predictive scaling system with demand forecasting"""
    
    def __init__(self, prediction_window_hours: int = 24, 
                 history_retention_hours: int = 168,  # 1 week
                 scaling_threshold_up: float = 0.8,
                 scaling_threshold_down: float = 0.4):
        self.prediction_window = timedelta(hours=prediction_window_hours)
        self.history_retention = timedelta(hours=history_retention_hours)
        self.scaling_threshold_up = scaling_threshold_up
        self.scaling_threshold_down = scaling_threshold_down
        
        # Historical data storage
        self.demand_history = {}  # resource_type -> deque of (timestamp, demand) tuples
        self.supply_history = {}  # resource_type -> deque of (timestamp, supply) tuples
        self.scaling_history = {}  # resource_type -> deque of ScalingAction
        
        # Prediction models (simplified - would use ML in production)
        self.trend_weights = {'recent': 0.5, 'medium': 0.3, 'old': 0.2}
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def update_resource_metrics(self, resource_type: str, demand: float, supply: float):
        """Update demand and supply metrics for a resource type"""
        current_time = datetime.now()
        
        # Initialize if not exists
        if resource_type not in self.demand_history:
            self.demand_history[resource_type] = deque()
            self.supply_history[resource_type] = deque()
            self.scaling_history[resource_type] = deque()
        
        # Add new data points
        self.demand_history[resource_type].append((current_time, demand))
        self.supply_history[resource_type].append((current_time, supply))
        
        # Clean old data
        self._cleanup_old_data(resource_type, current_time)
        
        self.logger.debug(f"Updated metrics for {resource_type}: demand={demand:.2f}, supply={supply:.2f}")
    
    def _cleanup_old_data(self, resource_type: str, current_time: datetime):
        """Remove data older than retention period"""
        cutoff_time = current_time - self.history_retention
        
        # Clean demand history
        while (self.demand_history[resource_type] and 
               self.demand_history[resource_type][0][0] < cutoff_time):
            self.demand_history[resource_type].popleft()
        
        # Clean supply history
        while (self.supply_history[resource_type] and 
               self.supply_history[resource_type][0][0] < cutoff_time):
            self.supply_history[resource_type].popleft()
        
        # Clean scaling history
        while (self.scaling_history[resource_type] and 
               self.scaling_history[resource_type][0].execution_timestamp < cutoff_time):
            self.scaling_history[resource_type].popleft()
    
    def generate_scaling_signals(self, resource_types: List[str]) -> List[PredictiveScalingSignal]:
        """Generate predictive scaling signals for resource types"""
        signals = []
        
        for resource_type in resource_types:
            signal = self._generate_resource_scaling_signal(resource_type)
            if signal and signal.scaling_needed():
                signals.append(signal)
        
        return signals
    
    def _generate_resource_scaling_signal(self, resource_type: str) -> Optional[PredictiveScalingSignal]:
        """Generate scaling signal for a specific resource type"""
        try:
            if resource_type not in self.demand_history:
                return None
            
            current_time = datetime.now()
            
            # Get current metrics
            current_demand = self._get_latest_demand(resource_type)
            current_supply = self._get_latest_supply(resource_type)
            
            if current_demand is None or current_supply is None:
                return None
            
            # Predict future demand
            predicted_demand, confidence = self._predict_demand(resource_type)
            
            # Determine scaling direction and urgency
            demand_ratio = predicted_demand / current_supply if current_supply > 0 else float('inf')
            
            if demand_ratio > self.scaling_threshold_up:
                trend_direction = ScalingDirection.SCALE_UP
                urgency = min(1.0, (demand_ratio - self.scaling_threshold_up) / (1.2 - self.scaling_threshold_up))
            elif demand_ratio < self.scaling_threshold_down:
                trend_direction = ScalingDirection.SCALE_DOWN
                urgency = min(1.0, (self.scaling_threshold_down - demand_ratio) / self.scaling_threshold_down)
            else:
                trend_direction = ScalingDirection.MAINTAIN
                urgency = 0.0
            
            # Calculate cost impact (simplified)
            cost_impact = self._estimate_scaling_cost(resource_type, trend_direction, predicted_demand, current_supply)
            
            signal = PredictiveScalingSignal(
                resource_type=resource_type,
                predicted_demand=predicted_demand,
                current_supply=current_supply,
                confidence=confidence,
                time_horizon=self.prediction_window,
                trend_direction=trend_direction,
                urgency=urgency,
                cost_impact=cost_impact
            )
            
            self.logger.debug(f"Generated scaling signal for {resource_type}: "
                            f"direction={trend_direction.value}, urgency={urgency:.2f}, confidence={confidence:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating scaling signal for {resource_type}: {e}")
            return None
    
    def _get_latest_demand(self, resource_type: str) -> Optional[float]:
        """Get latest demand value for resource type"""
        if (resource_type in self.demand_history and 
            self.demand_history[resource_type]):
            return self.demand_history[resource_type][-1][1]
        return None
    
    def _get_latest_supply(self, resource_type: str) -> Optional[float]:
        """Get latest supply value for resource type"""
        if (resource_type in self.supply_history and 
            self.supply_history[resource_type]):
            return self.supply_history[resource_type][-1][1]
        return None
    
    def _predict_demand(self, resource_type: str) -> Tuple[float, float]:
        """Predict future demand using time series analysis"""
        try:
            if resource_type not in self.demand_history:
                return 0.0, 0.0
            
            demand_data = list(self.demand_history[resource_type])
            if len(demand_data) < 3:
                # Not enough data for prediction
                latest_demand = demand_data[-1][1] if demand_data else 0.0
                return latest_demand, 0.3  # Low confidence
            
            # Extract demand values and timestamps
            demands = np.array([d[1] for d in demand_data])
            timestamps = np.array([d[0].timestamp() for d in demand_data])
            
            # Simple trend analysis
            recent_period = len(demands) // 3 if len(demands) > 6 else len(demands)
            medium_period = len(demands) * 2 // 3 if len(demands) > 6 else len(demands)
            
            # Calculate trends for different periods
            recent_trend = self._calculate_trend(demands[-recent_period:])
            medium_trend = self._calculate_trend(demands[-medium_period:])
            overall_trend = self._calculate_trend(demands)
            
            # Weighted prediction
            predicted_demand = (
                demands[-1] + 
                recent_trend * self.trend_weights['recent'] +
                medium_trend * self.trend_weights['medium'] +
                overall_trend * self.trend_weights['old']
            )
            
            # Ensure non-negative
            predicted_demand = max(0.0, predicted_demand)
            
            # Calculate confidence based on data consistency
            confidence = self._calculate_prediction_confidence(demands)
            
            return predicted_demand, confidence
            
        except Exception as e:
            self.logger.error(f"Error predicting demand for {resource_type}: {e}")
            return 0.0, 0.0
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend using simple linear regression"""
        if len(data) < 2:
            return 0.0
        
        try:
            x = np.arange(len(data))
            y = data
            
            # Simple linear regression: y = mx + b
            n = len(data)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_prediction_confidence(self, data: np.ndarray) -> float:
        """Calculate confidence in prediction based on data stability"""
        if len(data) < 2:
            return 0.2
        
        try:
            # Calculate coefficient of variation
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if mean_val == 0:
                return 0.3
            
            cv = std_val / mean_val
            
            # Higher stability (lower CV) = higher confidence
            # CV of 0 = confidence 1.0, CV of 1 = confidence 0.5
            confidence = max(0.2, min(1.0, 1.0 - cv * 0.5))
            
            # Boost confidence with more data points
            data_boost = min(0.2, len(data) / 50)  # Up to 0.2 boost for 50+ data points
            confidence = min(1.0, confidence + data_boost)
            
            return confidence
            
        except Exception:
            return 0.3
    
    def _estimate_scaling_cost(self, resource_type: str, direction: ScalingDirection, 
                             predicted_demand: float, current_supply: float) -> float:
        """Estimate cost impact of scaling action"""
        try:
            if direction == ScalingDirection.MAINTAIN:
                return 0.0
            
            # Simple cost model (would be more sophisticated in production)
            base_cost_per_unit = 1.0  # Would come from resource configuration
            
            if direction == ScalingDirection.SCALE_UP:
                additional_supply_needed = max(0, predicted_demand - current_supply)
                cost_impact = additional_supply_needed * base_cost_per_unit
            else:  # SCALE_DOWN
                supply_reduction = max(0, current_supply - predicted_demand)
                cost_impact = -supply_reduction * base_cost_per_unit * 0.8  # Savings with efficiency factor
            
            return cost_impact
            
        except Exception:
            return 0.0
    
    def execute_scaling_action(self, signal: PredictiveScalingSignal, 
                             target_allocation: float) -> ScalingAction:
        """Execute scaling action based on signal"""
        action = ScalingAction(
            action_id=f"scale_{signal.resource_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            resource_type=signal.resource_type,
            current_allocation=signal.current_supply,
            target_allocation=target_allocation,
            scaling_direction=signal.trend_direction,
            confidence=signal.confidence,
            rationale=f"Predicted demand: {signal.predicted_demand:.2f}, "
                     f"Current supply: {signal.current_supply:.2f}, "
                     f"Confidence: {signal.confidence:.2f}",
            expected_cost_impact=signal.cost_impact,
            expected_performance_impact=self._estimate_performance_impact(signal, target_allocation),
            execution_timestamp=datetime.now()
        )
        
        # Record scaling action
        if signal.resource_type not in self.scaling_history:
            self.scaling_history[signal.resource_type] = deque()
        
        self.scaling_history[signal.resource_type].append(action)
        
        self.logger.info(f"Executed scaling action: {action.action_id} for {signal.resource_type}")
        
        return action
    
    def _estimate_performance_impact(self, signal: PredictiveScalingSignal, 
                                   target_allocation: float) -> float:
        """Estimate performance impact of scaling action"""
        try:
            current_utilization = signal.predicted_demand / signal.current_supply if signal.current_supply > 0 else 1.0
            target_utilization = signal.predicted_demand / target_allocation if target_allocation > 0 else 1.0
            
            # Performance improvement is roughly inverse to utilization change
            if signal.trend_direction == ScalingDirection.SCALE_UP:
                # Lower utilization = better performance
                return max(0.0, (current_utilization - target_utilization) * 100)
            elif signal.trend_direction == ScalingDirection.SCALE_DOWN:
                # Higher utilization = worse performance, but cost savings
                return min(0.0, (current_utilization - target_utilization) * 100)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def get_scaling_recommendations(self, resource_types: List[str]) -> Dict[str, Dict]:
        """Get scaling recommendations for resource types"""
        recommendations = {}
        
        for resource_type in resource_types:
            signals = self.generate_scaling_signals([resource_type])
            
            if signals:
                signal = signals[0]
                
                # Calculate recommended target allocation
                if signal.trend_direction == ScalingDirection.SCALE_UP:
                    target_allocation = signal.predicted_demand * 1.2  # 20% buffer
                elif signal.trend_direction == ScalingDirection.SCALE_DOWN:
                    target_allocation = signal.predicted_demand * 1.1  # 10% buffer
                else:
                    target_allocation = signal.current_supply
                
                recommendations[resource_type] = {
                    'current_supply': signal.current_supply,
                    'predicted_demand': signal.predicted_demand,
                    'recommended_allocation': target_allocation,
                    'scaling_direction': signal.trend_direction.value,
                    'confidence': signal.confidence,
                    'urgency': signal.urgency,
                    'cost_impact': signal.cost_impact,
                    'action_needed': signal.scaling_needed()
                }
            else:
                recommendations[resource_type] = {
                    'current_supply': self._get_latest_supply(resource_type) or 0.0,
                    'predicted_demand': self._get_latest_demand(resource_type) or 0.0,
                    'recommended_allocation': self._get_latest_supply(resource_type) or 0.0,
                    'scaling_direction': 'maintain',
                    'confidence': 0.0,
                    'urgency': 0.0,
                    'cost_impact': 0.0,
                    'action_needed': False
                }
        
        return recommendations


def create_predictive_scaler(prediction_window_hours: int = 24,
                           history_retention_hours: int = 168,
                           scaling_threshold_up: float = 0.8,
                           scaling_threshold_down: float = 0.4) -> PredictiveScaler:
    """Factory function to create predictive scaler"""
    return PredictiveScaler(
        prediction_window_hours=prediction_window_hours,
        history_retention_hours=history_retention_hours,
        scaling_threshold_up=scaling_threshold_up,
        scaling_threshold_down=scaling_threshold_down
    )