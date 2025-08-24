"""
System Growth Modeler
====================

Models system growth patterns and predicts future growth based on historical data.
Extracted from architectural_evolution_predictor.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

from .data_models import SystemGrowthPattern

# Configure logging
logger = logging.getLogger(__name__)


class SystemGrowthModeler:
    """Models system growth patterns and predicts future growth"""
    
    def __init__(self):
        self.growth_indicators = {
            'user_metrics': ['active_users', 'new_registrations', 'user_sessions'],
            'technical_metrics': ['api_calls', 'database_queries', 'response_times'],
            'business_metrics': ['revenue', 'feature_usage', 'support_tickets'],
            'development_metrics': ['commits', 'deployments', 'bug_reports']
        }
        
        self.growth_models = {
            'linear': self._linear_growth_model,
            'exponential': self._exponential_growth_model,
            'logarithmic': self._logarithmic_growth_model,
            'sigmoid': self._sigmoid_growth_model,
            'seasonal': self._seasonal_growth_model
        }
    
    def analyze_growth_patterns(self, historical_data: Dict[str, List[float]], 
                               time_periods: List[datetime]) -> SystemGrowthPattern:
        """Analyze historical data to identify growth patterns"""
        try:
            growth_pattern = SystemGrowthPattern()
            
            if not historical_data or not time_periods:
                return growth_pattern
            
            # Calculate growth rates for different metrics
            growth_rates = {}
            for metric, values in historical_data.items():
                if len(values) >= 2:
                    growth_rate = self._calculate_growth_rate(values)
                    growth_rates[metric] = growth_rate
            
            # Map metrics to growth pattern attributes
            growth_pattern.component_growth_rate = growth_rates.get('component_count', 0.0)
            growth_pattern.user_growth_rate = growth_rates.get('active_users', 0.0)
            growth_pattern.data_growth_rate = growth_rates.get('data_volume', 0.0)
            growth_pattern.transaction_growth_rate = growth_rates.get('transaction_count', 0.0)
            growth_pattern.feature_addition_rate = growth_rates.get('feature_count', 0.0)
            growth_pattern.complexity_growth_rate = growth_rates.get('complexity_score', 0.0)
            
            # Analyze seasonal patterns
            growth_pattern.seasonal_patterns = self._detect_seasonal_patterns(historical_data, time_periods)
            
            # Calculate growth acceleration
            growth_pattern.growth_acceleration = self._calculate_growth_acceleration(growth_rates)
            
            # Assess growth sustainability
            growth_pattern.growth_sustainability = self._assess_growth_sustainability(growth_rates)
            
            # Determine time period
            if time_periods:
                time_span = time_periods[-1] - time_periods[0]
                growth_pattern.time_period = f"{time_span.days} days"
            
            return growth_pattern
            
        except Exception as e:
            logger.error(f"Error analyzing growth patterns: {e}")
            return SystemGrowthPattern()
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate compound annual growth rate (CAGR)"""
        try:
            if len(values) < 2 or values[0] == 0:
                return 0.0
            
            # Simple growth rate calculation
            start_value = values[0]
            end_value = values[-1]
            periods = len(values) - 1
            
            if start_value <= 0:
                return 0.0
            
            growth_rate = (end_value / start_value) ** (1 / periods) - 1
            return growth_rate
            
        except Exception as e:
            logger.error(f"Error calculating growth rate: {e}")
            return 0.0
    
    def _detect_seasonal_patterns(self, historical_data: Dict[str, List[float]], 
                                 time_periods: List[datetime]) -> Dict[str, float]:
        """Detect seasonal patterns in growth data"""
        try:
            seasonal_patterns = {}
            
            if len(time_periods) < 12:  # Need at least a year of data
                return seasonal_patterns
            
            # Group data by month
            monthly_data = defaultdict(list)
            for i, period in enumerate(time_periods):
                month = period.month
                for metric, values in historical_data.items():
                    if i < len(values):
                        monthly_data[f"{metric}_{month}"].append(values[i])
            
            # Calculate seasonal indices
            for key, values in monthly_data.items():
                if len(values) > 1:
                    avg_value = np.mean(values)
                    overall_avg = np.mean([v for vals in historical_data.values() for v in vals])
                    if overall_avg > 0:
                        seasonal_index = avg_value / overall_avg
                        seasonal_patterns[key] = seasonal_index
            
            return seasonal_patterns
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {e}")
            return {}
    
    def _calculate_growth_acceleration(self, growth_rates: Dict[str, float]) -> float:
        """Calculate overall growth acceleration"""
        try:
            if not growth_rates:
                return 0.0
            
            # Weight different growth rates by importance
            weights = {
                'user_growth_rate': 0.3,
                'transaction_growth_rate': 0.25,
                'data_growth_rate': 0.2,
                'feature_addition_rate': 0.15,
                'complexity_growth_rate': -0.1  # Negative because complexity growth is bad
            }
            
            weighted_growth = 0.0
            total_weight = 0.0
            
            for metric, rate in growth_rates.items():
                weight = weights.get(metric, 0.1)
                weighted_growth += rate * weight
                total_weight += abs(weight)
            
            if total_weight > 0:
                return weighted_growth / total_weight
            else:
                return np.mean(list(growth_rates.values()))
                
        except Exception as e:
            logger.error(f"Error calculating growth acceleration: {e}")
            return 0.0
    
    def _assess_growth_sustainability(self, growth_rates: Dict[str, float]) -> float:
        """Assess sustainability of current growth rates"""
        try:
            sustainability_factors = []
            
            # Check for balanced growth
            rate_variance = np.var(list(growth_rates.values()))
            if rate_variance < 0.1:  # Low variance indicates balanced growth
                sustainability_factors.append(0.8)
            else:
                sustainability_factors.append(0.4)
            
            # Check for reasonable growth rates
            avg_growth = np.mean(list(growth_rates.values()))
            if 0.1 <= avg_growth <= 0.5:  # 10-50% growth is sustainable
                sustainability_factors.append(0.9)
            elif avg_growth > 0.5:  # Very high growth may not be sustainable
                sustainability_factors.append(0.3)
            else:
                sustainability_factors.append(0.6)
            
            # Check complexity growth
            complexity_growth = growth_rates.get('complexity_growth_rate', 0)
            if complexity_growth < 0.1:  # Low complexity growth is good
                sustainability_factors.append(0.8)
            else:
                sustainability_factors.append(0.4)
            
            return np.mean(sustainability_factors)
            
        except Exception as e:
            logger.error(f"Error assessing growth sustainability: {e}")
            return 0.5
    
    def predict_future_growth(self, growth_pattern: SystemGrowthPattern, 
                             months_ahead: int = 12) -> Dict[str, List[float]]:
        """Predict future growth based on identified patterns"""
        try:
            predictions = {}
            
            # Base growth rates
            base_rates = {
                'component_count': growth_pattern.component_growth_rate,
                'user_count': growth_pattern.user_growth_rate,
                'data_volume': growth_pattern.data_growth_rate,
                'transaction_volume': growth_pattern.transaction_growth_rate
            }
            
            # Generate predictions for each metric
            for metric, base_rate in base_rates.items():
                monthly_predictions = []
                current_value = 100  # Assume base of 100 units
                
                for month in range(months_ahead):
                    # Apply base growth rate
                    monthly_growth = base_rate
                    
                    # Apply seasonal adjustment if available
                    season_key = f"{metric}_{(month % 12) + 1}"
                    seasonal_factor = growth_pattern.seasonal_patterns.get(season_key, 1.0)
                    monthly_growth *= seasonal_factor
                    
                    # Apply growth acceleration/deceleration
                    acceleration_factor = 1.0 + (growth_pattern.growth_acceleration * month / 12)
                    monthly_growth *= acceleration_factor
                    
                    # Apply sustainability factor
                    sustainability_factor = growth_pattern.growth_sustainability
                    monthly_growth *= sustainability_factor
                    
                    current_value *= (1 + monthly_growth)
                    monthly_predictions.append(current_value)
                
                predictions[metric] = monthly_predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting future growth: {e}")
            return {}
    
    def _linear_growth_model(self, data: List[float]) -> Dict[str, float]:
        """Linear growth model"""
        try:
            if len(data) < 2:
                return {'slope': 0, 'intercept': 0, 'r_squared': 0}
            
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            slope, intercept = coeffs
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {'slope': slope, 'intercept': intercept, 'r_squared': r_squared}
            
        except Exception as e:
            logger.error(f"Error in linear growth model: {e}")
            return {'slope': 0, 'intercept': 0, 'r_squared': 0}
    
    def _exponential_growth_model(self, data: List[float]) -> Dict[str, float]:
        """Exponential growth model"""
        try:
            if len(data) < 2 or any(d <= 0 for d in data):
                return {'growth_rate': 0, 'initial_value': data[0] if data else 0, 'r_squared': 0}
            
            x = np.arange(len(data))
            log_data = np.log(data)
            coeffs = np.polyfit(x, log_data, 1)
            growth_rate, log_initial = coeffs
            initial_value = np.exp(log_initial)
            
            # Calculate R-squared
            y_pred = initial_value * np.exp(growth_rate * x)
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {'growth_rate': growth_rate, 'initial_value': initial_value, 'r_squared': r_squared}
            
        except Exception as e:
            logger.error(f"Error in exponential growth model: {e}")
            return {'growth_rate': 0, 'initial_value': data[0] if data else 0, 'r_squared': 0}
    
    def _logarithmic_growth_model(self, data: List[float]) -> Dict[str, float]:
        """Logarithmic growth model"""
        try:
            if len(data) < 2:
                return {'coefficient': 0, 'constant': 0, 'r_squared': 0}
            
            x = np.arange(1, len(data) + 1)  # Start from 1 to avoid log(0)
            log_x = np.log(x)
            coeffs = np.polyfit(log_x, data, 1)
            coefficient, constant = coeffs
            
            # Calculate R-squared
            y_pred = coefficient * log_x + constant
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {'coefficient': coefficient, 'constant': constant, 'r_squared': r_squared}
            
        except Exception as e:
            logger.error(f"Error in logarithmic growth model: {e}")
            return {'coefficient': 0, 'constant': 0, 'r_squared': 0}
    
    def _sigmoid_growth_model(self, data: List[float]) -> Dict[str, float]:
        """Sigmoid (S-curve) growth model"""
        try:
            # Simplified sigmoid approximation
            if len(data) < 4:
                return {'carrying_capacity': max(data) if data else 0, 'growth_rate': 0, 'r_squared': 0}
            
            carrying_capacity = max(data) * 1.2  # Estimate carrying capacity
            midpoint = len(data) / 2
            
            # Estimate growth rate from steepest part of curve
            max_slope = 0
            for i in range(1, len(data)):
                slope = data[i] - data[i-1]
                max_slope = max(max_slope, slope)
            
            growth_rate = max_slope / (carrying_capacity / 4) if carrying_capacity > 0 else 0
            
            return {'carrying_capacity': carrying_capacity, 'growth_rate': growth_rate, 'r_squared': 0.7}
            
        except Exception as e:
            logger.error(f"Error in sigmoid growth model: {e}")
            return {'carrying_capacity': 0, 'growth_rate': 0, 'r_squared': 0}
    
    def _seasonal_growth_model(self, data: List[float]) -> Dict[str, float]:
        """Seasonal growth model with trend and seasonality"""
        try:
            if len(data) < 12:  # Need at least a year of data
                return {'trend': 0, 'seasonal_amplitude': 0, 'r_squared': 0}
            
            # Decompose into trend and seasonal components
            x = np.arange(len(data))
            
            # Linear trend
            trend_coeffs = np.polyfit(x, data, 1)
            trend = trend_coeffs[0]
            
            # Remove trend to isolate seasonal component
            detrended = data - (trend_coeffs[0] * x + trend_coeffs[1])
            
            # Calculate seasonal amplitude
            seasonal_amplitude = np.std(detrended)
            
            return {'trend': trend, 'seasonal_amplitude': seasonal_amplitude, 'r_squared': 0.8}
            
        except Exception as e:
            logger.error(f"Error in seasonal growth model: {e}")
            return {'trend': 0, 'seasonal_amplitude': 0, 'r_squared': 0}


__all__ = ['SystemGrowthModeler']