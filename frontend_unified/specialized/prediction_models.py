#!/usr/bin/env python3
"""
Prediction Models Module
========================

Machine learning prediction models extracted from predictive_analytics_integration.py
for STEELCLAD modularization (Agent Y STEELCLAD Protocol)

Contains simplified prediction models for:
- Linear trend analysis
- Service failure prediction
- Performance degradation prediction
- Resource utilization forecasting
"""

import numpy as np
from typing import Dict, Any, List


class SimpleLinearTrendModel:
    """
    Simple linear trend prediction model
    
    Uses polynomial fitting to predict linear trends in time series data.
    Suitable for basic trend analysis of system metrics.
    """
    
    def predict(self, data: List[float]) -> float:
        """
        Predict trend direction and magnitude
        
        Args:
            data: List of numerical values representing time series
            
        Returns:
            Slope of trend line (positive = increasing, negative = decreasing)
        """
        if not data or len(data) < 2:
            return 0.0
        
        try:
            # Use numpy polyfit to calculate linear trend
            slope = np.polyfit(range(len(data)), data, 1)[0]
            return float(slope)
        except Exception:
            return 0.0
    
    def predict_with_confidence(self, data: List[float]) -> Dict[str, float]:
        """
        Predict trend with confidence measure
        
        Args:
            data: List of numerical values representing time series
            
        Returns:
            Dictionary with trend and confidence metrics
        """
        if not data or len(data) < 3:
            return {'trend': 0.0, 'confidence': 0.0, 'r_squared': 0.0}
        
        try:
            x = np.array(range(len(data)))
            y = np.array(data)
            
            # Calculate linear fit
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            # Calculate R-squared for confidence
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Convert R-squared to confidence percentage
            confidence = min(abs(r_squared) * 100, 100.0)
            
            return {
                'trend': float(slope),
                'confidence': float(confidence),
                'r_squared': float(r_squared)
            }
        except Exception:
            return {'trend': 0.0, 'confidence': 0.0, 'r_squared': 0.0}


class ServiceFailurePredictionModel:
    """
    Service failure prediction model
    
    Predicts probability of service failures based on service success rates
    and historical performance metrics.
    """
    
    def predict(self, metrics: Dict[str, Any]) -> float:
        """
        Predict service failure probability
        
        Args:
            metrics: Dictionary containing service metrics
            
        Returns:
            Failure probability (0.0 = no failure, 1.0 = certain failure)
        """
        try:
            success_rate = float(metrics.get('service_success_rate', 100.0))
            
            # Convert success rate to failure probability
            failure_probability = max(0.0, (100.0 - success_rate) / 100.0)
            
            # Apply additional factors if available
            if 'validation_errors' in metrics:
                error_count = int(metrics['validation_errors'])
                error_factor = min(error_count * 0.1, 0.5)  # Up to 50% increase
                failure_probability = min(failure_probability + error_factor, 1.0)
            
            return failure_probability
        except Exception:
            return 0.0
    
    def predict_by_service_type(self, metrics: Dict[str, Any], service_history: List[Dict]) -> Dict[str, float]:
        """
        Predict failure probability by service type
        
        Args:
            metrics: Current service metrics
            service_history: Historical service performance data
            
        Returns:
            Dictionary mapping service types to failure probabilities
        """
        service_predictions = {}
        
        try:
            # Group services by type and analyze
            service_types = {}
            for entry in service_history:
                svc_type = entry.get('service_type', 'unknown')
                if svc_type not in service_types:
                    service_types[svc_type] = []
                service_types[svc_type].append(entry.get('success_rate', 100.0))
            
            # Calculate failure probability for each type
            for svc_type, success_rates in service_types.items():
                if success_rates:
                    avg_success = sum(success_rates) / len(success_rates)
                    trend = SimpleLinearTrendModel().predict(success_rates)
                    
                    base_failure_prob = max(0.0, (100.0 - avg_success) / 100.0)
                    trend_penalty = max(0.0, -trend * 0.01)  # Penalty for declining trend
                    
                    service_predictions[svc_type] = min(base_failure_prob + trend_penalty, 1.0)
                else:
                    service_predictions[svc_type] = 0.5  # Unknown = moderate risk
        except Exception:
            pass
        
        return service_predictions


class PerformanceDegradationModel:
    """
    Performance degradation prediction model
    
    Predicts system performance degradation based on health metrics
    and resource utilization patterns.
    """
    
    def predict(self, metrics: Dict[str, Any]) -> float:
        """
        Predict performance degradation probability
        
        Args:
            metrics: Dictionary containing system health metrics
            
        Returns:
            Degradation probability (0.0 = no degradation, 1.0 = severe degradation)
        """
        try:
            health = float(metrics.get('overall_health', 100.0))
            
            # Convert health to degradation probability
            degradation_prob = max(0.0, (100.0 - health) / 100.0)
            
            # Apply additional factors
            factors = []
            
            # Dependency health factor
            if 'dependency_health' in metrics:
                dep_health = float(metrics['dependency_health'])
                dep_factor = max(0.0, (100.0 - dep_health) / 200.0)  # Half weight
                factors.append(dep_factor)
            
            # Import success rate factor
            if 'import_success_rate' in metrics:
                import_rate = float(metrics['import_success_rate'])
                import_factor = max(0.0, (100.0 - import_rate) / 200.0)  # Half weight
                factors.append(import_factor)
            
            # Layer compliance factor
            if 'layer_compliance' in metrics:
                layer_compliance = float(metrics['layer_compliance'])
                compliance_factor = max(0.0, (100.0 - layer_compliance) / 200.0)  # Half weight
                factors.append(compliance_factor)
            
            # Combine all factors
            if factors:
                additional_risk = sum(factors) / len(factors)
                degradation_prob = min(degradation_prob + additional_risk, 1.0)
            
            return degradation_prob
        except Exception:
            return 0.0
    
    def predict_critical_areas(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict degradation risk by system area
        
        Args:
            metrics: System health metrics
            
        Returns:
            Dictionary mapping system areas to degradation risks
        """
        area_risks = {}
        
        try:
            # Architecture layer risks
            layer_compliance = float(metrics.get('layer_compliance', 100.0))
            area_risks['architecture_layers'] = max(0.0, (100.0 - layer_compliance) / 100.0)
            
            # Dependency risks  
            dependency_health = float(metrics.get('dependency_health', 100.0))
            area_risks['dependencies'] = max(0.0, (100.0 - dependency_health) / 100.0)
            
            # Service layer risks
            service_rate = float(metrics.get('service_success_rate', 100.0))
            area_risks['services'] = max(0.0, (100.0 - service_rate) / 100.0)
            
            # Import system risks
            import_rate = float(metrics.get('import_success_rate', 100.0))
            area_risks['imports'] = max(0.0, (100.0 - import_rate) / 100.0)
            
        except Exception:
            pass
        
        return area_risks


class ResourceUtilizationModel:
    """
    Resource utilization prediction model
    
    Predicts future resource utilization based on current usage patterns
    and system growth trends.
    """
    
    def predict(self, metrics: Dict[str, Any]) -> float:
        """
        Predict normalized resource utilization
        
        Args:
            metrics: Dictionary containing resource metrics
            
        Returns:
            Normalized utilization (0.0 = minimal, 1.0 = maximum)
        """
        try:
            components = int(metrics.get('registered_components', 0))
            services = int(metrics.get('services_registered', 0))
            dependencies = int(metrics.get('dependency_count', 0))
            
            # Calculate composite utilization score
            utilization_factors = []
            
            if components > 0:
                utilization_factors.append(min(components / 100.0, 1.0))
            
            if services > 0:
                utilization_factors.append(min(services / 50.0, 1.0))  # 50 services = full
            
            if dependencies > 0:
                utilization_factors.append(min(dependencies / 200.0, 1.0))  # 200 deps = full
            
            # Average the factors
            if utilization_factors:
                return sum(utilization_factors) / len(utilization_factors)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def predict_growth_trend(self, historical_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Predict resource growth trends
        
        Args:
            historical_metrics: List of historical metric snapshots
            
        Returns:
            Dictionary with growth predictions for different resource types
        """
        growth_trends = {}
        
        try:
            if len(historical_metrics) < 3:
                return growth_trends
            
            # Extract time series for different resources
            components_history = [m.get('registered_components', 0) for m in historical_metrics]
            services_history = [m.get('services_registered', 0) for m in historical_metrics]
            dependencies_history = [m.get('dependency_count', 0) for m in historical_metrics]
            
            # Calculate trends using linear model
            trend_model = SimpleLinearTrendModel()
            
            if components_history:
                growth_trends['components'] = trend_model.predict(components_history)
            
            if services_history:
                growth_trends['services'] = trend_model.predict(services_history)
            
            if dependencies_history:
                growth_trends['dependencies'] = trend_model.predict(dependencies_history)
            
            # Calculate overall growth trend
            if growth_trends:
                growth_trends['overall'] = sum(growth_trends.values()) / len(growth_trends)
            
        except Exception:
            pass
        
        return growth_trends


# Factory function for model creation
def create_prediction_models() -> Dict[str, Any]:
    """
    Create and return all prediction models
    
    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        'health_trend': SimpleLinearTrendModel(),
        'service_failure': ServiceFailurePredictionModel(),
        'performance': PerformanceDegradationModel(),
        'resource_utilization': ResourceUtilizationModel()
    }