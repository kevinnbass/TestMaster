#!/usr/bin/env python3
"""
Predictive Analytics Integration - Agent A Hour 7
Enhanced dashboard with predictive analytics and trend analysis

Integrates machine learning predictions and trend analysis into the
architecture monitoring dashboard for proactive system management.
"""

import logging
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import time

# Import architecture components
from core.architecture.architecture_integration import get_architecture_framework
from core.services.service_registry import get_service_registry


class PredictionType(Enum):
    """Types of predictions available"""
    HEALTH_TREND = "health_trend"
    SERVICE_FAILURE = "service_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_UTILIZATION = "resource_utilization"
    DEPENDENCY_ISSUES = "dependency_issues"


class ConfidenceLevel(Enum):
    """Prediction confidence levels"""
    HIGH = "high"      # >80% confidence
    MEDIUM = "medium"  # 60-80% confidence
    LOW = "low"       # 40-60% confidence
    VERY_LOW = "very_low"  # <40% confidence


@dataclass
class PredictiveMetric:
    """Predictive analytics metric"""
    name: str
    current_value: float
    predicted_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    confidence: ConfidenceLevel
    prediction_horizon: int  # minutes into future
    factors: List[str]  # Contributing factors
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PredictionResult:
    """Result of predictive analysis"""
    prediction_type: PredictionType
    target_metric: str
    probability: float
    confidence: ConfidenceLevel
    time_to_event: Optional[int]  # minutes
    contributing_factors: List[str]
    recommended_actions: List[str]
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PredictiveAnalyticsEngine:
    """
    Predictive Analytics Engine
    
    Provides machine learning-based predictions and trend analysis
    for architecture monitoring and system health management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Architecture components
        self.framework = get_architecture_framework()
        self.service_registry = get_service_registry()
        
        # Historical data storage
        self.historical_data_dir = Path("data/predictive_analytics")
        self.historical_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Prediction models (simplified for demonstration)
        self.models = {
            'health_trend': SimpleLinearTrendModel(),
            'service_failure': ServiceFailurePredictionModel(),
            'performance': PerformanceDegradationModel(),
            'resource_utilization': ResourceUtilizationModel()
        }
        
        # Historical metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Prediction cache
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Load historical data
        self._load_historical_data()
        
        self.logger.info("Predictive Analytics Engine initialized")
    
    def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics for analysis"""
        try:
            # Get architecture health
            health = self.framework.validate_architecture()
            
            # Get service registry status
            service_report = self.service_registry.get_registration_report()
            
            # Get architecture metrics
            arch_metrics = self.framework.get_architecture_metrics()
            
            current_metrics = {
                'timestamp': datetime.now().isoformat(),
                'overall_health': health.overall_health,
                'layer_compliance': health.layer_compliance,
                'dependency_health': health.dependency_health,
                'import_success_rate': health.import_success_rate,
                'services_registered': service_report.get('services_registered', 0),
                'services_active': service_report.get('services_active', 0),
                'service_success_rate': service_report.get('success_rate', 0.0),
                'architecture_layers': len(arch_metrics.get('layers', [])),
                'registered_components': len(arch_metrics.get('components', [])),
                'dependency_count': len(arch_metrics.get('dependencies', [])),
                'validation_errors': len(health.recommendations)
            }
            
            # Add to historical data
            self._add_to_history(current_metrics)
            
            return current_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect current metrics: {e}")
            return {}
    
    def generate_predictive_metrics(self) -> List[PredictiveMetric]:
        """Generate predictive metrics for dashboard"""
        predictive_metrics = []
        
        try:
            current_metrics = self.collect_current_metrics()
            
            if not current_metrics or len(self.metrics_history) < 5:
                self.logger.warning("Insufficient data for predictions")
                return predictive_metrics
            
            # Health trend prediction
            health_trend = self._predict_health_trend()
            if health_trend:
                predictive_metrics.append(health_trend)
            
            # Service performance prediction
            service_performance = self._predict_service_performance()
            if service_performance:
                predictive_metrics.append(service_performance)
            
            # Resource utilization prediction
            resource_prediction = self._predict_resource_utilization()
            if resource_prediction:
                predictive_metrics.append(resource_prediction)
            
            # Dependency health prediction
            dependency_prediction = self._predict_dependency_health()
            if dependency_prediction:
                predictive_metrics.append(dependency_prediction)
            
        except Exception as e:
            self.logger.error(f"Failed to generate predictive metrics: {e}")
        
        return predictive_metrics
    
    def generate_predictions(self) -> List[PredictionResult]:
        """Generate comprehensive system predictions"""
        predictions = []
        
        try:
            # Check cache first
            cache_key = "system_predictions"
            cached = self._get_cached_prediction(cache_key)
            if cached:
                return [cached]
            
            current_metrics = self.collect_current_metrics()
            
            if not current_metrics or len(self.metrics_history) < 10:
                return predictions
            
            # Service failure prediction
            service_failure = self._predict_service_failures(current_metrics)
            if service_failure:
                predictions.append(service_failure)
            
            # Performance degradation prediction
            performance_degradation = self._predict_performance_issues(current_metrics)
            if performance_degradation:
                predictions.append(performance_degradation)
            
            # System health deterioration prediction
            health_deterioration = self._predict_health_deterioration(current_metrics)
            if health_deterioration:
                predictions.append(health_deterioration)
            
            # Dependency issues prediction
            dependency_issues = self._predict_dependency_issues(current_metrics)
            if dependency_issues:
                predictions.append(dependency_issues)
            
            # Cache predictions
            if predictions:
                self._cache_predictions(cache_key, predictions[0])
            
        except Exception as e:
            self.logger.error(f"Failed to generate predictions: {e}")
        
        return predictions
    
    def _predict_health_trend(self) -> Optional[PredictiveMetric]:
        """Predict health trend using historical data"""
        try:
            if len(self.metrics_history) < 5:
                return None
            
            # Get recent health values
            health_values = [m.get('overall_health', 0) for m in self.metrics_history[-10:]]
            current_health = health_values[-1] if health_values else 0
            
            # Simple linear trend calculation
            if len(health_values) >= 5:
                x = np.array(range(len(health_values)))
                y = np.array(health_values)
                
                # Calculate linear regression
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                
                # Predict 30 minutes ahead (assuming 1 data point per 3 minutes)
                future_health = current_health + (slope * 10)
                
                # Determine trend direction and confidence
                if abs(slope) < 0.01:
                    trend = "stable"
                    confidence = ConfidenceLevel.MEDIUM
                elif slope > 0:
                    trend = "increasing"
                    confidence = ConfidenceLevel.HIGH if slope > 0.05 else ConfidenceLevel.MEDIUM
                else:
                    trend = "decreasing"
                    confidence = ConfidenceLevel.HIGH if slope < -0.05 else ConfidenceLevel.MEDIUM
                
                return PredictiveMetric(
                    name="architecture_health_trend",
                    current_value=current_health,
                    predicted_value=min(max(future_health, 0), 100),
                    trend_direction=trend,
                    confidence=confidence,
                    prediction_horizon=30,
                    factors=["historical_trend", "system_stability", "recent_changes"]
                )
                
        except Exception as e:
            self.logger.error(f"Health trend prediction failed: {e}")
        
        return None
    
    def _predict_service_performance(self) -> Optional[PredictiveMetric]:
        """Predict service performance metrics"""
        try:
            if len(self.metrics_history) < 5:
                return None
            
            # Get service success rates
            success_rates = [m.get('service_success_rate', 0) for m in self.metrics_history[-8:]]
            current_rate = success_rates[-1] if success_rates else 0
            
            # Calculate average and trend
            if len(success_rates) >= 4:
                recent_avg = np.mean(success_rates[-4:])
                overall_avg = np.mean(success_rates)
                
                trend = "stable"
                confidence = ConfidenceLevel.MEDIUM
                
                if recent_avg > overall_avg + 5:
                    trend = "increasing"
                    confidence = ConfidenceLevel.HIGH
                elif recent_avg < overall_avg - 5:
                    trend = "decreasing" 
                    confidence = ConfidenceLevel.HIGH
                
                predicted_rate = recent_avg + (recent_avg - overall_avg) * 0.5
                
                return PredictiveMetric(
                    name="service_success_rate",
                    current_value=current_rate,
                    predicted_value=min(max(predicted_rate, 0), 100),
                    trend_direction=trend,
                    confidence=confidence,
                    prediction_horizon=20,
                    factors=["service_reliability", "load_patterns", "system_health"]
                )
                
        except Exception as e:
            self.logger.error(f"Service performance prediction failed: {e}")
        
        return None
    
    def _predict_resource_utilization(self) -> Optional[PredictiveMetric]:
        """Predict resource utilization trends"""
        try:
            if len(self.metrics_history) < 5:
                return None
            
            # Use component count as proxy for resource utilization
            component_counts = [m.get('registered_components', 0) for m in self.metrics_history[-6:]]
            current_count = component_counts[-1] if component_counts else 0
            
            if len(component_counts) >= 4:
                # Simple growth rate calculation
                growth_rate = (component_counts[-1] - component_counts[0]) / max(len(component_counts), 1)
                predicted_count = current_count + growth_rate * 3
                
                trend = "stable"
                if growth_rate > 1:
                    trend = "increasing"
                elif growth_rate < -1:
                    trend = "decreasing"
                
                confidence = ConfidenceLevel.MEDIUM if abs(growth_rate) > 0.5 else ConfidenceLevel.LOW
                
                return PredictiveMetric(
                    name="component_utilization",
                    current_value=current_count,
                    predicted_value=max(predicted_count, 0),
                    trend_direction=trend,
                    confidence=confidence,
                    prediction_horizon=15,
                    factors=["component_growth", "system_expansion", "architecture_evolution"]
                )
                
        except Exception as e:
            self.logger.error(f"Resource utilization prediction failed: {e}")
        
        return None
    
    def _predict_dependency_health(self) -> Optional[PredictiveMetric]:
        """Predict dependency health trends"""
        try:
            if len(self.metrics_history) < 5:
                return None
            
            dependency_health = [m.get('dependency_health', 0) for m in self.metrics_history[-7:]]
            current_health = dependency_health[-1] if dependency_health else 0
            
            if len(dependency_health) >= 5:
                # Calculate variance to determine stability
                variance = np.var(dependency_health)
                avg_health = np.mean(dependency_health)
                
                if variance < 100:  # Low variance = stable
                    trend = "stable"
                    confidence = ConfidenceLevel.HIGH
                    predicted_health = avg_health
                else:  # High variance = unstable
                    recent_trend = dependency_health[-3:][0] - dependency_health[-1]
                    if recent_trend > 0:
                        trend = "increasing"
                    else:
                        trend = "decreasing"
                    confidence = ConfidenceLevel.MEDIUM
                    predicted_health = current_health + recent_trend
                
                return PredictiveMetric(
                    name="dependency_health_trend",
                    current_value=current_health,
                    predicted_value=min(max(predicted_health, 0), 100),
                    trend_direction=trend,
                    confidence=confidence,
                    prediction_horizon=25,
                    factors=["dependency_stability", "import_resolution", "service_connectivity"]
                )
                
        except Exception as e:
            self.logger.error(f"Dependency health prediction failed: {e}")
        
        return None
    
    def _predict_service_failures(self, current_metrics: Dict[str, Any]) -> Optional[PredictionResult]:
        """Predict potential service failures"""
        try:
            service_rate = current_metrics.get('service_success_rate', 100)
            validation_errors = current_metrics.get('validation_errors', 0)
            
            # Simple rule-based prediction
            failure_probability = 0.0
            
            if service_rate < 80:
                failure_probability += 0.3
            if validation_errors > 5:
                failure_probability += 0.2
            if service_rate < 60:
                failure_probability += 0.4
            
            if failure_probability > 0.5:
                confidence = ConfidenceLevel.HIGH if failure_probability > 0.8 else ConfidenceLevel.MEDIUM
                severity = "high" if failure_probability > 0.8 else "medium"
                
                return PredictionResult(
                    prediction_type=PredictionType.SERVICE_FAILURE,
                    target_metric="service_success_rate",
                    probability=failure_probability,
                    confidence=confidence,
                    time_to_event=int(60 / max(failure_probability, 0.1)),  # Inverse relationship
                    contributing_factors=[
                        f"Current success rate: {service_rate}%",
                        f"Validation errors: {validation_errors}",
                        "Historical performance trends"
                    ],
                    recommended_actions=[
                        "Review service configurations",
                        "Check dependency health", 
                        "Monitor resource utilization",
                        "Run diagnostic tests"
                    ],
                    severity=severity
                )
                
        except Exception as e:
            self.logger.error(f"Service failure prediction failed: {e}")
        
        return None
    
    def _predict_performance_issues(self, current_metrics: Dict[str, Any]) -> Optional[PredictionResult]:
        """Predict performance degradation"""
        try:
            overall_health = current_metrics.get('overall_health', 100)
            dependency_health = current_metrics.get('dependency_health', 100)
            
            # Performance risk factors
            performance_risk = 0.0
            
            if overall_health < 70:
                performance_risk += 0.4
            if dependency_health < 80:
                performance_risk += 0.3
            if len(self.metrics_history) > 5:
                recent_health_trend = np.mean([m.get('overall_health', 100) for m in self.metrics_history[-3:]])
                if recent_health_trend < overall_health - 10:
                    performance_risk += 0.3
            
            if performance_risk > 0.4:
                return PredictionResult(
                    prediction_type=PredictionType.PERFORMANCE_DEGRADATION,
                    target_metric="overall_health",
                    probability=performance_risk,
                    confidence=ConfidenceLevel.MEDIUM,
                    time_to_event=int(120 / max(performance_risk, 0.1)),
                    contributing_factors=[
                        f"Overall health: {overall_health}%",
                        f"Dependency health: {dependency_health}%",
                        "Recent performance trends"
                    ],
                    recommended_actions=[
                        "Optimize critical components",
                        "Check resource allocation",
                        "Review dependency chains",
                        "Consider architecture refactoring"
                    ],
                    severity="medium" if performance_risk > 0.7 else "low"
                )
                
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
        
        return None
    
    def _predict_health_deterioration(self, current_metrics: Dict[str, Any]) -> Optional[PredictionResult]:
        """Predict overall system health deterioration"""
        try:
            if len(self.metrics_history) < 8:
                return None
            
            health_values = [m.get('overall_health', 100) for m in self.metrics_history[-8:]]
            current_health = health_values[-1]
            
            # Calculate health trend
            recent_avg = np.mean(health_values[-4:])
            earlier_avg = np.mean(health_values[:4])
            
            deterioration_rate = (earlier_avg - recent_avg) / 4
            
            if deterioration_rate > 2:  # Health declining by >2% per measurement
                time_to_critical = int(max((current_health - 30) / deterioration_rate, 10))
                
                return PredictionResult(
                    prediction_type=PredictionType.HEALTH_TREND,
                    target_metric="overall_health",
                    probability=min(deterioration_rate / 10, 1.0),
                    confidence=ConfidenceLevel.HIGH if deterioration_rate > 5 else ConfidenceLevel.MEDIUM,
                    time_to_event=time_to_critical,
                    contributing_factors=[
                        f"Health deterioration rate: {deterioration_rate:.1f}%/measurement",
                        f"Current health: {current_health}%",
                        "Sustained downward trend"
                    ],
                    recommended_actions=[
                        "Immediate system review required",
                        "Check for recent changes",
                        "Validate all components",
                        "Consider rollback if applicable"
                    ],
                    severity="high" if deterioration_rate > 8 else "medium"
                )
                
        except Exception as e:
            self.logger.error(f"Health deterioration prediction failed: {e}")
        
        return None
    
    def _predict_dependency_issues(self, current_metrics: Dict[str, Any]) -> Optional[PredictionResult]:
        """Predict dependency-related issues"""
        try:
            dependency_health = current_metrics.get('dependency_health', 100)
            import_success = current_metrics.get('import_success_rate', 100)
            
            dependency_risk = 0.0
            
            if dependency_health < 90:
                dependency_risk += 0.3
            if import_success < 95:
                dependency_risk += 0.4
            if dependency_health < 70:
                dependency_risk += 0.3
            
            if dependency_risk > 0.4:
                return PredictionResult(
                    prediction_type=PredictionType.DEPENDENCY_ISSUES,
                    target_metric="dependency_health",
                    probability=dependency_risk,
                    confidence=ConfidenceLevel.MEDIUM,
                    time_to_event=int(90 / max(dependency_risk, 0.1)),
                    contributing_factors=[
                        f"Dependency health: {dependency_health}%",
                        f"Import success rate: {import_success}%",
                        "Dependency chain complexity"
                    ],
                    recommended_actions=[
                        "Review import dependencies",
                        "Update dependency mappings",
                        "Check for circular dependencies",
                        "Validate service connections"
                    ],
                    severity="medium" if dependency_risk > 0.6 else "low"
                )
                
        except Exception as e:
            self.logger.error(f"Dependency issues prediction failed: {e}")
        
        return None
    
    def _add_to_history(self, metrics: Dict[str, Any]):
        """Add metrics to historical data"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        # Periodically save to disk
        if len(self.metrics_history) % 10 == 0:
            self._save_historical_data()
    
    def _load_historical_data(self):
        """Load historical data from disk"""
        try:
            history_file = self.historical_data_dir / "metrics_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.metrics_history = json.load(f)
                self.logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
        except Exception as e:
            self.logger.warning(f"Could not load historical data: {e}")
    
    def _save_historical_data(self):
        """Save historical data to disk"""
        try:
            history_file = self.historical_data_dir / "metrics_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history[-500:], f)  # Save last 500 entries
        except Exception as e:
            self.logger.warning(f"Could not save historical data: {e}")
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction if still valid"""
        if cache_key in self.prediction_cache:
            prediction = self.prediction_cache[cache_key]
            if (datetime.now() - prediction.timestamp).total_seconds() < self.cache_ttl:
                return prediction
        return None
    
    def _cache_predictions(self, cache_key: str, prediction: PredictionResult):
        """Cache prediction result"""
        self.prediction_cache[cache_key] = prediction
    
    def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive analytics data for dashboard"""
        try:
            predictive_metrics = self.generate_predictive_metrics()
            predictions = self.generate_predictions()
            current_metrics = self.collect_current_metrics()
            
            return {
                'current_metrics': current_metrics,
                'predictive_metrics': [asdict(m) for m in predictive_metrics],
                'predictions': [asdict(p) for p in predictions],
                'analytics_status': {
                    'historical_data_points': len(self.metrics_history),
                    'predictions_cached': len(self.prediction_cache),
                    'models_active': len(self.models),
                    'last_analysis': datetime.now().isoformat()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


# Simplified prediction model classes (for demonstration)
class SimpleLinearTrendModel:
    """Simple linear trend prediction model"""
    def predict(self, data):
        return np.polyfit(range(len(data)), data, 1)[0] if len(data) > 1 else 0


class ServiceFailurePredictionModel:
    """Service failure prediction model"""
    def predict(self, metrics):
        success_rate = metrics.get('service_success_rate', 100)
        return max(0, (100 - success_rate) / 100)


class PerformanceDegradationModel:
    """Performance degradation prediction model"""
    def predict(self, metrics):
        health = metrics.get('overall_health', 100)
        return max(0, (100 - health) / 100)


class ResourceUtilizationModel:
    """Resource utilization prediction model"""
    def predict(self, metrics):
        components = metrics.get('registered_components', 0)
        return min(components / 100, 1.0)  # Normalize to 0-1


# Global analytics engine instance
_analytics_engine: Optional[PredictiveAnalyticsEngine] = None


def get_predictive_analytics_engine() -> PredictiveAnalyticsEngine:
    """Get global predictive analytics engine instance"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = PredictiveAnalyticsEngine()
    return _analytics_engine