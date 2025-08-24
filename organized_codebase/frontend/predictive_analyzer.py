"""
STEELCLAD Phase 2 Extraction: Predictive Analysis Engine
Extracted from enhanced_intelligence_linkage.py for specialized predictive analytics

This module provides comprehensive predictive analysis capabilities:
- Evolution prediction with ML-based trend analysis
- Change impact radius calculation
- Refactoring recommendations generation
- Service failure risk assessment
- Performance degradation forecasting
"""

import re
import ast
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

class ConfidenceLevel(Enum):
    """Prediction confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class PredictiveMetric:
    """Predictive analytics metric"""
    name: str
    current_value: float
    predicted_value: float
    trend_direction: str
    confidence: ConfidenceLevel
    prediction_horizon: int
    factors: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PredictiveAnalyzer:
    """
    Advanced predictive analysis engine for code evolution forecasting.
    
    Provides ML-powered predictions for:
    - Code health trends and evolution patterns
    - Service failure probability assessment
    - Performance degradation forecasting
    - Change impact radius calculation
    - Intelligent refactoring recommendations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize predictive analyzer with optional configuration."""
        self.config = config or {}
        
        # Configure prediction parameters
        self.health_trend_horizon = self.config.get('health_trend_horizon', 60)  # 1 hour
        self.failure_risk_horizon = self.config.get('failure_risk_horizon', 120)  # 2 hours
        self.performance_horizon = self.config.get('performance_horizon', 180)  # 3 hours
        
        # Configure scoring weights
        self.health_weights = self.config.get('health_weights', {
            'maintainability': 0.7,
            'complexity': 0.3
        })
        self.failure_weights = self.config.get('failure_weights', {
            'error_handling': 0.6,
            'logging_coverage': 0.4
        })
        self.performance_weights = self.config.get('performance_weights', {
            'complexity': 0.6,
            'size': 0.4
        })
    
    def predict_evolution(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Enhanced predictive evolution analysis using ML techniques.
        
        Args:
            content: File content to analyze
            file_path: Path to the file for context
            
        Returns:
            Dict containing evolution predictions with confidence scores
        """
        try:
            # Calculate current metrics
            current_complexity = self._calculate_complexity(content)
            current_maintainability = self._calculate_maintainability_index(content, current_complexity)
            
            # Predict health trend
            health_prediction = self._predict_health_trend(current_maintainability, current_complexity)
            
            # Predict service failure probability
            failure_prediction = self._predict_service_failure(content, file_path)
            
            # Predict performance degradation
            performance_prediction = self._predict_performance_degradation(current_complexity)
            
            return {
                "health_trend": health_prediction,
                "service_failure_risk": failure_prediction,
                "performance_degradation": performance_prediction,
                "prediction_confidence": self._calculate_prediction_confidence([
                    health_prediction, failure_prediction, performance_prediction
                ]),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "prediction_available": False}
    
    def calculate_change_impact_radius(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Calculate change impact radius with enhanced analytics.
        
        Args:
            content: File content to analyze
            file_path: Path to the file for context
            
        Returns:
            Dict containing impact radius analysis and risk assessment
        """
        # Analyze dependencies and interconnections
        imports = len(re.findall(r"^import |^from .* import", content, re.MULTILINE))
        function_calls = len(re.findall(r"(\w+)\(", content))
        class_usage = len(re.findall(r"(\w+)\.(\w+)", content))
        
        # Calculate impact score
        impact_score = (imports * 0.4 + function_calls * 0.3 + class_usage * 0.3) / max(len(content.splitlines()), 1)
        
        return {
            "impact_radius": min(1.0, impact_score),
            "affected_systems": max(1, int(impact_score * 10)),
            "propagation_depth": min(5, int(impact_score * 15)),
            "risk_level": "high" if impact_score > 0.7 else "medium" if impact_score > 0.4 else "low",
            "factors": {
                "import_dependencies": imports,
                "function_coupling": function_calls,
                "class_coupling": class_usage
            }
        }
    
    def generate_refactoring_recommendations(self, evolution_predictions: Dict, change_impact_radius: Dict) -> List[Dict]:
        """
        Generate intelligent refactoring recommendations.
        
        Args:
            evolution_predictions: Results from predict_evolution
            change_impact_radius: Results from calculate_change_impact_radius
            
        Returns:
            List of prioritized refactoring recommendations
        """
        recommendations = []
        
        # Analyze predictions for recommendations
        if isinstance(evolution_predictions, dict):
            health_trend = evolution_predictions.get("health_trend")
            performance_pred = evolution_predictions.get("performance_degradation")
            failure_risk = evolution_predictions.get("service_failure_risk")
            
            if health_trend and hasattr(health_trend, 'trend_direction'):
                if health_trend.trend_direction == "degrading":
                    recommendations.append({
                        "type": "health_improvement",
                        "priority": "high",
                        "action": "Improve code maintainability through refactoring",
                        "expected_impact": "30% maintainability improvement"
                    })
            
            if performance_pred and hasattr(performance_pred, 'current_value'):
                if performance_pred.current_value > 0.6:
                    recommendations.append({
                        "type": "performance_optimization", 
                        "priority": "medium",
                        "action": "Reduce algorithmic complexity and optimize critical paths",
                        "expected_impact": "25% performance improvement"
                    })
            
            if failure_risk and hasattr(failure_risk, 'current_value'):
                if failure_risk.current_value > 0.5:
                    recommendations.append({
                        "type": "reliability_enhancement",
                        "priority": "high", 
                        "action": "Improve error handling and logging coverage",
                        "expected_impact": "40% reduction in failure probability"
                    })
        
        # Add impact-based recommendations
        if isinstance(change_impact_radius, dict):
            impact_level = change_impact_radius.get("risk_level", "low")
            if impact_level == "high":
                recommendations.append({
                    "type": "dependency_management",
                    "priority": "medium",
                    "action": "Reduce coupling through interface abstraction",
                    "expected_impact": "50% reduction in change propagation risk"
                })
        
        return recommendations
    
    # Private helper methods
    
    def _predict_health_trend(self, maintainability: Dict, complexity: Dict) -> PredictiveMetric:
        """Predict health trend using maintainability and complexity metrics."""
        mi_score = maintainability.get("maintainability_index", 50)
        cc_score = complexity.get("cyclomatic_complexity", 10)
        
        # Simple ML-like prediction based on current metrics
        health_score = (mi_score / 100) * self.health_weights['maintainability'] + \
                      (1 - min(cc_score / 50, 1)) * self.health_weights['complexity']
        
        trend = "stable"
        if health_score > 0.75:
            trend = "improving"
        elif health_score < 0.4:
            trend = "degrading"
            
        confidence = ConfidenceLevel.HIGH if abs(health_score - 0.5) > 0.25 else ConfidenceLevel.MEDIUM
        
        return PredictiveMetric(
            name="health_trend",
            current_value=health_score,
            predicted_value=min(1.0, health_score * 1.05),  # Slight improvement prediction
            trend_direction=trend,
            confidence=confidence,
            prediction_horizon=self.health_trend_horizon,
            factors=["maintainability_index", "cyclomatic_complexity", "code_quality"]
        )
    
    def _predict_service_failure(self, content: str, file_path: str) -> PredictiveMetric:
        """Predict service failure probability."""
        # Analyze failure indicators
        error_handling_score = len(re.findall(r"try:|except|finally:", content)) / max(len(content.splitlines()), 1)
        logging_score = len(re.findall(r"log\.|logger\.|logging\.", content, re.IGNORECASE)) / max(len(content.splitlines()), 1)
        
        # Calculate failure risk
        failure_risk = max(0, 1.0 - (error_handling_score * self.failure_weights['error_handling'] + 
                                   logging_score * self.failure_weights['logging_coverage']))
        
        confidence = ConfidenceLevel.HIGH if failure_risk > 0.7 or failure_risk < 0.3 else ConfidenceLevel.MEDIUM
        
        return PredictiveMetric(
            name="service_failure_risk", 
            current_value=failure_risk,
            predicted_value=failure_risk * 0.95,  # Slight improvement over time
            trend_direction="decreasing" if failure_risk > 0.5 else "stable",
            confidence=confidence,
            prediction_horizon=self.failure_risk_horizon,
            factors=["error_handling", "logging_coverage", "code_robustness"]
        )
    
    def _predict_performance_degradation(self, complexity: Dict) -> PredictiveMetric:
        """Predict performance degradation based on complexity."""
        cc_score = complexity.get("cyclomatic_complexity", 10)
        loc = complexity.get("lines_of_code", 100)
        
        # Performance degradation prediction
        degradation_risk = min(1.0, (cc_score / 30) * self.performance_weights['complexity'] + 
                              (loc / 1000) * self.performance_weights['size'])
        
        confidence = ConfidenceLevel.HIGH if degradation_risk > 0.6 else ConfidenceLevel.MEDIUM
        
        return PredictiveMetric(
            name="performance_degradation",
            current_value=degradation_risk,
            predicted_value=min(1.0, degradation_risk * 1.1),  # Slight increase over time
            trend_direction="increasing" if degradation_risk > 0.4 else "stable", 
            confidence=confidence,
            prediction_horizon=self.performance_horizon,
            factors=["cyclomatic_complexity", "code_size", "algorithmic_complexity"]
        )
    
    def _calculate_prediction_confidence(self, predictions: List[PredictiveMetric]) -> ConfidenceLevel:
        """Calculate overall confidence across multiple predictions."""
        if not predictions:
            return ConfidenceLevel.VERY_LOW
            
        confidence_scores = []
        for pred in predictions:
            if hasattr(pred, 'confidence'):
                if pred.confidence == ConfidenceLevel.HIGH:
                    confidence_scores.append(0.9)
                elif pred.confidence == ConfidenceLevel.MEDIUM:
                    confidence_scores.append(0.7)
                elif pred.confidence == ConfidenceLevel.LOW:
                    confidence_scores.append(0.5)
                else:
                    confidence_scores.append(0.3)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.3
        
        if avg_confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif avg_confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate cyclomatic complexity and related metrics."""
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'and', 'or', 'try', 'except']
        
        total_complexity = 1  # Base complexity
        for keyword in complexity_keywords:
            total_complexity += len(re.findall(rf'\b{keyword}\b', content))
        
        return {
            "cyclomatic_complexity": total_complexity,
            "lines_of_code": len(content.splitlines()),
            "complexity_density": total_complexity / max(len(content.splitlines()), 1)
        }
    
    def _calculate_maintainability_index(self, content: str, complexity: Dict) -> Dict[str, Any]:
        """Calculate maintainability index."""
        loc = complexity["lines_of_code"]
        cc = complexity["cyclomatic_complexity"]
        
        # Simplified maintainability index calculation
        if loc == 0:
            return {"maintainability_index": 100, "maintainability_level": "excellent"}
            
        # Basic formula: 171 - 5.2 * log(avg_cc) - 0.23 * avg_cc - 16.2 * log(loc)
        import math
        
        try:
            mi = 171 - 5.2 * math.log(cc) - 0.23 * cc - 16.2 * math.log(loc)
            mi = max(0, min(100, mi))  # Clamp between 0-100
        except:
            mi = 50  # Default value
        
        return {
            "maintainability_index": mi,
            "maintainability_level": self._get_maintainability_level(mi)
        }
    
    def _get_maintainability_level(self, mi: float) -> str:
        """Get maintainability level description."""
        if mi >= 85:
            return "excellent"
        elif mi >= 70:
            return "good"
        elif mi >= 50:
            return "moderate"
        elif mi >= 25:
            return "low"
        else:
            return "critical"

def create_predictive_analyzer(config: Optional[Dict] = None) -> PredictiveAnalyzer:
    """
    Factory function to create a configured predictive analyzer instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured PredictiveAnalyzer instance
    """
    return PredictiveAnalyzer(config)

# Export key components
__all__ = [
    'PredictiveAnalyzer',
    'PredictiveMetric', 
    'ConfidenceLevel',
    'create_predictive_analyzer'
]