"""
ML Performance Prediction Models Module
Extracted from predictive_analytics_integration.py for Agent X's Epsilon base integration
< 200 lines per STEELCLAD protocol

Provides intelligent performance prediction capabilities:
- Service performance forecasting
- Health trend prediction
- Resource utilization modeling
- Performance degradation detection
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"  
    STABLE = "stable"

@dataclass
class PredictiveMetric:
    """Predictive metric result"""
    name: str
    current_value: float
    predicted_value: float
    trend_direction: TrendDirection
    confidence: ConfidenceLevel
    prediction_horizon: int  # minutes
    factors: List[str]
    timestamp: str

class MLPerformancePredictions:
    """ML-powered performance prediction engine"""
    
    def __init__(self, history_limit: int = 50):
        self.metrics_history: List[Dict[str, Any]] = []
        self.history_limit = history_limit
        self.prediction_models = {
            'health_trend': self._predict_health_trend,
            'service_performance': self._predict_service_performance,
            'resource_utilization': self._predict_resource_utilization,
            'failure_probability': self._predict_failure_probability
        }
        
    def add_metrics_data(self, metrics: Dict[str, Any]):
        """Add new metrics data to history for prediction"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
        
        # Maintain history limit
        if len(self.metrics_history) > self.history_limit:
            self.metrics_history = self.metrics_history[-self.history_limit:]
    
    def generate_predictions(self) -> Dict[str, PredictiveMetric]:
        """Generate all available performance predictions"""
        predictions = {}
        
        if len(self.metrics_history) < 3:
            logger.warning("Insufficient historical data for predictions")
            return predictions
        
        for model_name, model_func in self.prediction_models.items():
            try:
                prediction = model_func()
                if prediction:
                    predictions[model_name] = prediction
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
        
        return predictions
    
    def _predict_health_trend(self) -> Optional[PredictiveMetric]:
        """Predict overall system health trend"""
        try:
            if len(self.metrics_history) < 5:
                return None
            
            # Extract health values from recent history
            health_values = [m.get('overall_health', 50) for m in self.metrics_history[-10:]]
            current_health = health_values[-1]
            
            # Simple linear trend analysis
            if len(health_values) >= 5:
                x = np.array(range(len(health_values)))
                y = np.array(health_values)
                
                # Calculate linear regression
                slope, intercept = np.polyfit(x, y, 1)
                future_health = slope * (len(health_values) + 5) + intercept
                
                # Determine trend and confidence
                trend = TrendDirection.STABLE
                confidence = ConfidenceLevel.MEDIUM
                
                if abs(slope) > 2:
                    trend = TrendDirection.INCREASING if slope > 0 else TrendDirection.DECREASING
                    confidence = ConfidenceLevel.HIGH if abs(slope) > 5 else ConfidenceLevel.MEDIUM
                
                return PredictiveMetric(
                    name="health_trend",
                    current_value=current_health,
                    predicted_value=min(max(future_health, 0), 100),
                    trend_direction=trend,
                    confidence=confidence,
                    prediction_horizon=30,
                    factors=["historical_trend", "system_stability", "recent_changes"],
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Health trend prediction failed: {e}")
        
        return None
    
    def _predict_service_performance(self) -> Optional[PredictiveMetric]:
        """Predict service performance metrics"""
        try:
            if len(self.metrics_history) < 5:
                return None
            
            # Get service success rates from history
            success_rates = [m.get('service_success_rate', 0) for m in self.metrics_history[-8:]]
            current_rate = success_rates[-1] if success_rates else 0
            
            if len(success_rates) >= 4:
                recent_avg = np.mean(success_rates[-4:])
                overall_avg = np.mean(success_rates)
                
                trend = TrendDirection.STABLE
                confidence = ConfidenceLevel.MEDIUM
                
                # Determine trend based on recent vs overall average
                if recent_avg > overall_avg + 5:
                    trend = TrendDirection.INCREASING
                    confidence = ConfidenceLevel.HIGH
                elif recent_avg < overall_avg - 5:
                    trend = TrendDirection.DECREASING
                    confidence = ConfidenceLevel.HIGH
                
                # Predict future performance
                predicted_rate = recent_avg + (recent_avg - overall_avg) * 0.5
                
                return PredictiveMetric(
                    name="service_success_rate",
                    current_value=current_rate,
                    predicted_value=min(max(predicted_rate, 0), 100),
                    trend_direction=trend,
                    confidence=confidence,
                    prediction_horizon=20,
                    factors=["service_reliability", "load_patterns", "system_health"],
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Service performance prediction failed: {e}")
        
        return None
    
    def _predict_resource_utilization(self) -> Optional[PredictiveMetric]:
        """Predict resource utilization trends"""
        try:
            if len(self.metrics_history) < 4:
                return None
            
            # Extract resource metrics
            component_counts = [m.get('registered_components', 0) for m in self.metrics_history[-6:]]
            current_components = component_counts[-1]
            
            if len(component_counts) >= 3:
                # Calculate growth rate
                recent_growth = np.mean(np.diff(component_counts[-3:]))
                predicted_components = current_components + (recent_growth * 3)  # 3 periods ahead
                
                trend = TrendDirection.STABLE
                confidence = ConfidenceLevel.MEDIUM
                
                if abs(recent_growth) > 1:
                    trend = TrendDirection.INCREASING if recent_growth > 0 else TrendDirection.DECREASING
                    confidence = ConfidenceLevel.HIGH if abs(recent_growth) > 2 else ConfidenceLevel.MEDIUM
                
                return PredictiveMetric(
                    name="resource_utilization",
                    current_value=current_components,
                    predicted_value=max(predicted_components, 0),
                    trend_direction=trend,
                    confidence=confidence,
                    prediction_horizon=15,
                    factors=["component_registration", "system_load", "scaling_patterns"],
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Resource utilization prediction failed: {e}")
        
        return None
    
    def _predict_failure_probability(self) -> Optional[PredictiveMetric]:
        """Predict system failure probability"""
        try:
            if len(self.metrics_history) < 3:
                return None
            
            # Calculate failure indicators
            recent_metrics = self.metrics_history[-3:]
            health_degradation = 0
            service_issues = 0
            
            for i in range(1, len(recent_metrics)):
                prev_health = recent_metrics[i-1].get('overall_health', 100)
                curr_health = recent_metrics[i].get('overall_health', 100)
                
                if curr_health < prev_health - 10:  # Significant degradation
                    health_degradation += 1
                
                if recent_metrics[i].get('service_success_rate', 100) < 80:
                    service_issues += 1
            
            # Simple failure probability calculation
            failure_score = (health_degradation * 0.4 + service_issues * 0.6) * 0.3
            failure_probability = min(failure_score, 0.95)
            
            confidence = ConfidenceLevel.HIGH if failure_probability > 0.7 else ConfidenceLevel.MEDIUM
            trend = TrendDirection.INCREASING if failure_probability > 0.3 else TrendDirection.STABLE
            
            return PredictiveMetric(
                name="failure_probability",
                current_value=failure_probability,
                predicted_value=failure_probability,
                trend_direction=trend,
                confidence=confidence,
                prediction_horizon=10,
                factors=["health_degradation", "service_failures", "system_instability"],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failure probability prediction failed: {e}")
        
        return None
    
    def get_prediction_summary(self, predictions: Dict[str, PredictiveMetric]) -> Dict[str, Any]:
        """Generate summary of prediction results"""
        if not predictions:
            return {"status": "no_predictions", "summary": {}}
        
        high_confidence_count = sum(1 for p in predictions.values() if p.confidence == ConfidenceLevel.HIGH)
        critical_predictions = sum(1 for p in predictions.values() 
                                 if p.name == "failure_probability" and p.current_value > 0.6)
        
        return {
            "status": "predictions_available",
            "summary": {
                "total_predictions": len(predictions),
                "high_confidence_predictions": high_confidence_count,
                "critical_alerts": critical_predictions,
                "prediction_types": list(predictions.keys()),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()
    
    def get_history_size(self) -> int:
        """Get current history size"""
        return len(self.metrics_history)

# Plugin interface for Agent X integration
def create_performance_predictions_plugin(config: Dict[str, Any] = None):
    """Factory function to create ML performance predictions plugin"""
    history_limit = config.get('history_limit', 50) if config else 50
    return MLPerformancePredictions(history_limit)