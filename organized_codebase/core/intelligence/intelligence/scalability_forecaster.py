"""
Scalability Forecaster
=====================

Forecasts system scalability needs and bottlenecks based on growth predictions.
Extracted from architectural_evolution_predictor.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import logging
import math
import numpy as np
from typing import Dict, List, Any

from .data_models import ArchitecturalMetrics, ScalabilityForecast, ScalingPattern

# Configure logging
logger = logging.getLogger(__name__)


class ScalabilityForecaster:
    """Forecasts system scalability needs and bottlenecks"""
    
    def __init__(self):
        self.capacity_metrics = {
            'cpu_utilization': {'threshold': 0.8, 'growth_impact': 1.5},
            'memory_utilization': {'threshold': 0.85, 'growth_impact': 1.3},
            'disk_io': {'threshold': 0.7, 'growth_impact': 2.0},
            'network_bandwidth': {'threshold': 0.75, 'growth_impact': 1.8},
            'database_connections': {'threshold': 0.9, 'growth_impact': 2.5},
            'api_response_time': {'threshold': 500, 'growth_impact': 1.2, 'unit': 'ms'},
            'concurrent_users': {'threshold': 1000, 'growth_impact': 1.1}
        }
        
        self.scaling_strategies = {
            ScalingPattern.HORIZONTAL: {
                'cost_factor': 1.0,
                'complexity_factor': 1.2,
                'effectiveness': 0.9
            },
            ScalingPattern.VERTICAL: {
                'cost_factor': 1.5,
                'complexity_factor': 0.8,
                'effectiveness': 0.7
            },
            ScalingPattern.FUNCTIONAL: {
                'cost_factor': 1.3,
                'complexity_factor': 1.5,
                'effectiveness': 0.8
            }
        }
    
    def forecast_scalability(self, current_metrics: ArchitecturalMetrics,
                           growth_predictions: Dict[str, List[float]],
                           forecast_months: int = 12) -> ScalabilityForecast:
        """Forecast scalability needs based on growth predictions"""
        try:
            forecast = ScalabilityForecast(forecast_horizon=forecast_months)
            
            # Set current capacity baseline
            forecast.current_capacity = self._extract_current_capacity(current_metrics)
            
            # Predict future load based on growth patterns
            forecast.predicted_load = self._predict_future_load(growth_predictions, forecast_months)
            
            # Identify capacity gaps
            forecast.capacity_gaps = self._identify_capacity_gaps(
                forecast.current_capacity, forecast.predicted_load
            )
            
            # Predict bottlenecks
            forecast.bottleneck_predictions = self._predict_bottlenecks(
                forecast.capacity_gaps, forecast.predicted_load
            )
            
            # Generate scaling recommendations
            forecast.scaling_recommendations = self._generate_scaling_recommendations(
                forecast.bottleneck_predictions, forecast.capacity_gaps
            )
            
            # Calculate resource requirements
            forecast.resource_requirements = self._calculate_resource_requirements(
                forecast.scaling_recommendations, forecast.predicted_load
            )
            
            # Estimate costs
            forecast.cost_implications = self._estimate_scaling_costs(
                forecast.resource_requirements, forecast.scaling_recommendations
            )
            
            # Assess risks
            forecast.risk_factors = self._assess_scaling_risks(
                forecast.scaling_recommendations, current_metrics
            )
            
            # Calculate confidence level
            forecast.confidence_level = self._calculate_forecast_confidence(
                growth_predictions, current_metrics
            )
            
            # Recommend primary scaling strategy
            forecast.recommended_scaling_strategy = self._recommend_scaling_strategy(
                forecast.bottleneck_predictions, current_metrics
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting scalability: {e}")
            return ScalabilityForecast()
    
    def _extract_current_capacity(self, metrics: ArchitecturalMetrics) -> Dict[str, float]:
        """Extract current capacity metrics"""
        try:
            return {
                'cpu_capacity': 100.0,  # Assume 100% base capacity
                'memory_capacity': 100.0,
                'storage_capacity': 100.0,
                'network_capacity': 100.0,
                'api_capacity': metrics.api_endpoint_count * 1000,  # requests per second
                'database_capacity': metrics.database_count * 500,  # connections
                'service_capacity': metrics.service_count * 100    # concurrent requests
            }
        except Exception as e:
            logger.error(f"Error extracting current capacity: {e}")
            return {}
    
    def _predict_future_load(self, growth_predictions: Dict[str, List[float]], 
                           forecast_months: int) -> Dict[str, float]:
        """Predict future load based on growth patterns"""
        try:
            if not growth_predictions:
                return {}
            
            predicted_load = {}
            
            # Map growth predictions to load metrics
            mapping = {
                'user_count': 'user_load',
                'transaction_volume': 'transaction_load',
                'data_volume': 'data_load',
                'component_count': 'system_complexity_load'
            }
            
            for growth_metric, load_metric in mapping.items():
                if growth_metric in growth_predictions:
                    # Take the value at the forecast horizon
                    values = growth_predictions[growth_metric]
                    if len(values) >= forecast_months:
                        predicted_load[load_metric] = values[forecast_months - 1]
                    elif values:
                        predicted_load[load_metric] = values[-1]
            
            # Convert to resource-specific load predictions
            user_load = predicted_load.get('user_load', 100)
            transaction_load = predicted_load.get('transaction_load', 100)
            data_load = predicted_load.get('data_load', 100)
            
            resource_load = {
                'cpu_load': user_load * 0.8 + transaction_load * 0.6,
                'memory_load': user_load * 0.7 + data_load * 0.9,
                'storage_load': data_load * 1.2,
                'network_load': transaction_load * 0.9 + user_load * 0.3,
                'api_load': transaction_load * 1.1,
                'database_load': transaction_load * 0.8 + data_load * 0.7
            }
            
            return resource_load
            
        except Exception as e:
            logger.error(f"Error predicting future load: {e}")
            return {}
    
    def _identify_capacity_gaps(self, current_capacity: Dict[str, float], 
                              predicted_load: Dict[str, float]) -> Dict[str, float]:
        """Identify gaps between current capacity and predicted load"""
        try:
            gaps = {}
            
            for resource in current_capacity:
                capacity = current_capacity[resource]
                load_key = resource.replace('_capacity', '_load')
                predicted = predicted_load.get(load_key, 0)
                
                if predicted > capacity:
                    gap_ratio = (predicted - capacity) / capacity
                    gaps[resource] = gap_ratio
                else:
                    gaps[resource] = 0.0
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error identifying capacity gaps: {e}")
            return {}
    
    def _predict_bottlenecks(self, capacity_gaps: Dict[str, float], 
                           predicted_load: Dict[str, float]) -> List[Dict[str, Any]]:
        """Predict likely bottlenecks based on capacity analysis"""
        try:
            bottlenecks = []
            
            # Sort capacity gaps by severity
            sorted_gaps = sorted(capacity_gaps.items(), key=lambda x: x[1], reverse=True)
            
            for resource, gap_ratio in sorted_gaps:
                if gap_ratio > 0.1:  # 10% gap threshold
                    severity = 'critical' if gap_ratio > 0.5 else 'high' if gap_ratio > 0.3 else 'medium'
                    
                    bottleneck = {
                        'resource': resource,
                        'severity': severity,
                        'gap_ratio': gap_ratio,
                        'predicted_impact': self._predict_bottleneck_impact(resource, gap_ratio),
                        'timeline_to_bottleneck': self._estimate_bottleneck_timeline(gap_ratio),
                        'mitigation_urgency': severity
                    }
                    bottlenecks.append(bottleneck)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error predicting bottlenecks: {e}")
            return []
    
    def _predict_bottleneck_impact(self, resource: str, gap_ratio: float) -> Dict[str, str]:
        """Predict impact of bottleneck on system"""
        try:
            impact_map = {
                'cpu_capacity': {
                    'performance': 'severe_degradation' if gap_ratio > 0.5 else 'degradation',
                    'availability': 'reduced' if gap_ratio > 0.3 else 'stable',
                    'user_experience': 'poor' if gap_ratio > 0.4 else 'degraded'
                },
                'memory_capacity': {
                    'performance': 'severe_degradation' if gap_ratio > 0.3 else 'degradation',
                    'availability': 'at_risk' if gap_ratio > 0.2 else 'stable',
                    'user_experience': 'poor' if gap_ratio > 0.25 else 'degraded'
                },
                'database_capacity': {
                    'performance': 'severe_degradation' if gap_ratio > 0.4 else 'degradation',
                    'availability': 'critical_risk' if gap_ratio > 0.3 else 'at_risk',
                    'user_experience': 'unacceptable' if gap_ratio > 0.4 else 'poor'
                }
            }
            
            return impact_map.get(resource, {
                'performance': 'degradation',
                'availability': 'stable',
                'user_experience': 'degraded'
            })
            
        except Exception as e:
            logger.error(f"Error predicting bottleneck impact: {e}")
            return {}
    
    def _estimate_bottleneck_timeline(self, gap_ratio: float) -> str:
        """Estimate when bottleneck will occur"""
        try:
            if gap_ratio > 0.8:
                return 'immediate'
            elif gap_ratio > 0.5:
                return '1-3 months'
            elif gap_ratio > 0.3:
                return '3-6 months'
            elif gap_ratio > 0.1:
                return '6-12 months'
            else:
                return '12+ months'
        except:
            return 'unknown'
    
    def _generate_scaling_recommendations(self, bottlenecks: List[Dict[str, Any]], 
                                        capacity_gaps: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate scaling recommendations based on bottleneck analysis"""
        try:
            recommendations = []
            
            for bottleneck in bottlenecks:
                resource = bottleneck['resource']
                gap_ratio = bottleneck['gap_ratio']
                
                # Determine appropriate scaling strategy
                if 'cpu' in resource or 'memory' in resource:
                    if gap_ratio > 0.5:
                        strategy = ScalingPattern.HORIZONTAL
                        recommendation = {
                            'resource': resource,
                            'strategy': strategy,
                            'scale_factor': math.ceil(1 + gap_ratio),
                            'priority': 'high',
                            'implementation_complexity': 'medium',
                            'cost_impact': 'medium'
                        }
                    else:
                        strategy = ScalingPattern.VERTICAL
                        recommendation = {
                            'resource': resource,
                            'strategy': strategy,
                            'scale_factor': 1 + gap_ratio,
                            'priority': 'medium',
                            'implementation_complexity': 'low',
                            'cost_impact': 'low'
                        }
                    
                elif 'database' in resource:
                    strategy = ScalingPattern.DATA_PARTITIONING
                    recommendation = {
                        'resource': resource,
                        'strategy': strategy,
                        'scale_factor': math.ceil(1 + gap_ratio),
                        'priority': 'high',
                        'implementation_complexity': 'high',
                        'cost_impact': 'medium'
                    }
                    
                else:
                    strategy = ScalingPattern.HORIZONTAL
                    recommendation = {
                        'resource': resource,
                        'strategy': strategy,
                        'scale_factor': 1 + gap_ratio,
                        'priority': 'medium',
                        'implementation_complexity': 'medium',
                        'cost_impact': 'medium'
                    }
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating scaling recommendations: {e}")
            return []
    
    def _calculate_resource_requirements(self, scaling_recommendations: List[Dict[str, Any]], 
                                       predicted_load: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate specific resource requirements for scaling"""
        try:
            requirements = {}
            
            for recommendation in scaling_recommendations:
                resource = recommendation['resource']
                scale_factor = recommendation['scale_factor']
                strategy = recommendation['strategy']
                
                if resource not in requirements:
                    requirements[resource] = {}
                
                # Calculate specific requirements based on strategy
                if strategy == ScalingPattern.HORIZONTAL:
                    requirements[resource]['additional_instances'] = scale_factor - 1
                    requirements[resource]['total_instances'] = scale_factor
                    requirements[resource]['cpu_per_instance'] = 2.0  # cores
                    requirements[resource]['memory_per_instance'] = 4.0  # GB
                    
                elif strategy == ScalingPattern.VERTICAL:
                    requirements[resource]['cpu_increase'] = scale_factor - 1
                    requirements[resource]['memory_increase'] = scale_factor - 1
                    requirements[resource]['storage_increase'] = scale_factor - 1
                    
                elif strategy == ScalingPattern.DATA_PARTITIONING:
                    requirements[resource]['partition_count'] = int(scale_factor)
                    requirements[resource]['replication_factor'] = 2
                    requirements[resource]['storage_per_partition'] = 100  # GB
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error calculating resource requirements: {e}")
            return {}
    
    def _estimate_scaling_costs(self, resource_requirements: Dict[str, Dict[str, float]], 
                              scaling_recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate costs associated with scaling recommendations"""
        try:
            costs = {}
            
            # Cost factors (simplified)
            cost_factors = {
                'cpu_hour': 0.05,  # $ per core hour
                'memory_gb_hour': 0.01,  # $ per GB hour
                'storage_gb_month': 0.10,  # $ per GB month
                'instance_hour': 0.20,  # $ per instance hour
                'network_gb': 0.05  # $ per GB transfer
            }
            
            monthly_hours = 24 * 30  # 720 hours per month
            
            for resource, requirements in resource_requirements.items():
                resource_cost = 0.0
                
                # Calculate costs based on requirements
                if 'additional_instances' in requirements:
                    instances = requirements['additional_instances']
                    cpu_per_instance = requirements.get('cpu_per_instance', 2)
                    memory_per_instance = requirements.get('memory_per_instance', 4)
                    
                    resource_cost += instances * monthly_hours * cost_factors['instance_hour']
                    resource_cost += instances * cpu_per_instance * monthly_hours * cost_factors['cpu_hour']
                    resource_cost += instances * memory_per_instance * monthly_hours * cost_factors['memory_gb_hour']
                
                elif 'cpu_increase' in requirements:
                    cpu_increase = requirements['cpu_increase']
                    memory_increase = requirements.get('memory_increase', 0)
                    
                    resource_cost += cpu_increase * monthly_hours * cost_factors['cpu_hour']
                    resource_cost += memory_increase * monthly_hours * cost_factors['memory_gb_hour']
                
                costs[resource] = resource_cost
            
            # Calculate total costs
            costs['total_monthly'] = sum(costs.values())
            costs['total_annual'] = costs['total_monthly'] * 12
            
            return costs
            
        except Exception as e:
            logger.error(f"Error estimating scaling costs: {e}")
            return {}
    
    def _assess_scaling_risks(self, scaling_recommendations: List[Dict[str, Any]], 
                            current_metrics: ArchitecturalMetrics) -> List[str]:
        """Assess risks associated with scaling recommendations"""
        try:
            risks = []
            
            # Analyze implementation complexity
            high_complexity_count = len([r for r in scaling_recommendations 
                                       if r.get('implementation_complexity') == 'high'])
            
            if high_complexity_count > 2:
                risks.append("Multiple high-complexity scaling operations increase implementation risk")
            
            # Analyze cost impact
            high_cost_count = len([r for r in scaling_recommendations 
                                 if r.get('cost_impact') == 'high'])
            
            if high_cost_count > 1:
                risks.append("High cost impact may strain budget and require significant investment")
            
            # Analyze architectural impact
            if current_metrics.coupling_score > 0.7:
                risks.append("High system coupling may complicate horizontal scaling")
            
            if current_metrics.monitoring_coverage < 0.6:
                risks.append("Insufficient monitoring may hide scaling issues")
            
            # Analyze data consistency risks
            data_scaling = any('database' in r['resource'] for r in scaling_recommendations)
            if data_scaling and current_metrics.database_count > 1:
                risks.append("Database scaling may introduce data consistency challenges")
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing scaling risks: {e}")
            return []
    
    def _calculate_forecast_confidence(self, growth_predictions: Dict[str, List[float]], 
                                     current_metrics: ArchitecturalMetrics) -> float:
        """Calculate confidence level for scalability forecast"""
        try:
            confidence_factors = []
            
            # Data quality factor
            if growth_predictions and all(len(values) >= 6 for values in growth_predictions.values()):
                confidence_factors.append(0.8)  # Good historical data
            else:
                confidence_factors.append(0.4)  # Limited data
            
            # System maturity factor
            if current_metrics.lines_of_code > 10000 and current_metrics.component_count > 5:
                confidence_factors.append(0.7)  # Mature system
            else:
                confidence_factors.append(0.5)  # Early stage system
            
            # Monitoring factor
            if current_metrics.monitoring_coverage > 0.8:
                confidence_factors.append(0.9)  # Good monitoring
            elif current_metrics.monitoring_coverage > 0.5:
                confidence_factors.append(0.7)  # Adequate monitoring
            else:
                confidence_factors.append(0.3)  # Poor monitoring
            
            # Architecture complexity factor
            if current_metrics.complexity_score < 0.5:
                confidence_factors.append(0.8)  # Simple architecture
            elif current_metrics.complexity_score < 0.7:
                confidence_factors.append(0.6)  # Moderate complexity
            else:
                confidence_factors.append(0.4)  # Complex architecture
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error calculating forecast confidence: {e}")
            return 0.5
    
    def _recommend_scaling_strategy(self, bottlenecks: List[Dict[str, Any]], 
                                  current_metrics: ArchitecturalMetrics) -> ScalingPattern:
        """Recommend primary scaling strategy based on analysis"""
        try:
            if not bottlenecks:
                return ScalingPattern.HORIZONTAL
            
            # Analyze bottleneck types
            cpu_memory_bottlenecks = len([b for b in bottlenecks 
                                        if 'cpu' in b['resource'] or 'memory' in b['resource']])
            database_bottlenecks = len([b for b in bottlenecks if 'database' in b['resource']])
            
            # Consider system characteristics
            if current_metrics.coupling_score > 0.8:
                # High coupling favors vertical scaling
                return ScalingPattern.VERTICAL
            elif database_bottlenecks > cpu_memory_bottlenecks:
                # Database bottlenecks suggest data partitioning
                return ScalingPattern.DATA_PARTITIONING
            elif current_metrics.service_count > 10:
                # Many services suggest functional scaling
                return ScalingPattern.FUNCTIONAL
            else:
                # Default to horizontal scaling
                return ScalingPattern.HORIZONTAL
                
        except Exception as e:
            logger.error(f"Error recommending scaling strategy: {e}")
            return ScalingPattern.HORIZONTAL


__all__ = ['ScalabilityForecaster']