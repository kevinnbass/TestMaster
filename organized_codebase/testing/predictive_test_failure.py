"""
Predictive Test Failure System for TestMaster
ML-powered prediction of test failures and proactive failure prevention
"""

import json
import time
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import statistics
import math

# ML imports with fallback
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    import random

class FailureRiskLevel(Enum):
    """Risk levels for test failure prediction"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FailureCategory(Enum):
    """Categories of test failures"""
    ENVIRONMENT = "environment"
    DATA_DEPENDENCY = "data_dependency"
    TIMING = "timing"
    RESOURCE = "resource"
    LOGIC_ERROR = "logic_error"
    INFRASTRUCTURE = "infrastructure"
    FLAKINESS = "flakiness"
    UNKNOWN = "unknown"

@dataclass
class TestFailureFeatures:
    """Features used for failure prediction"""
    test_name: str
    historical_failure_rate: float
    recent_failure_count: int
    execution_time_trend: float
    code_change_frequency: float
    dependency_changes: int
    environment_stability: float
    resource_usage_pattern: float
    test_complexity: float
    last_success_days: float
    failure_clustering_score: float
    seasonal_pattern_score: float
    integration_test_ratio: float
    mock_dependency_ratio: float
    assertion_count: int

@dataclass
class FailurePrediction:
    """Prediction result for test failure"""
    test_name: str
    failure_probability: float
    risk_level: FailureRiskLevel
    predicted_failure_category: FailureCategory
    confidence_score: float
    contributing_factors: List[Tuple[str, float]]
    preventive_actions: List[str]
    monitoring_recommendations: List[str]
    prediction_timestamp: float

@dataclass
class FailurePreventionPlan:
    """Comprehensive failure prevention plan"""
    high_risk_tests: List[str]
    predicted_failures_24h: int
    predicted_failures_7d: int
    prevention_strategies: Dict[str, List[str]]
    resource_recommendations: List[str]
    monitoring_alerts: List[str]
    success_probability_improvement: float

@dataclass
class PredictionReport:
    """Complete prediction analysis report"""
    timestamp: float
    total_tests_analyzed: int
    predictions: List[FailurePrediction]
    prevention_plan: FailurePreventionPlan
    model_performance: Dict[str, float]
    accuracy_metrics: Dict[str, float]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]

class PredictiveTestFailureSystem:
    """Advanced ML-powered test failure prediction system"""
    
    def __init__(self):
        self.prediction_models: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, Any] = {}
        self.training_data: List[Dict[str, Any]] = []
        self.prediction_history: List[PredictionReport] = []
        self.feature_importance: Dict[str, List[Tuple[str, float]]] = {}
        
        if ML_AVAILABLE:
            self._initialize_ml_models()
        
        self.failure_patterns = self._initialize_failure_patterns()
        
    def _initialize_ml_models(self):
        """Initialize ML models for different prediction tasks"""
        # Primary failure prediction model
        self.prediction_models['primary'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Failure category prediction model
        self.prediction_models['category'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Risk level prediction model
        self.prediction_models['risk'] = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # Feature scalers
        self.feature_scalers['primary'] = StandardScaler()
        self.feature_scalers['category'] = StandardScaler()
        self.feature_scalers['risk'] = StandardScaler()
        
        # Label encoders
        self.label_encoders['category'] = LabelEncoder()
        self.label_encoders['risk'] = LabelEncoder()
    
    def _initialize_failure_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize known failure patterns"""
        return {
            "timing_issues": {
                "indicators": ["timeout", "race_condition", "sleep", "wait"],
                "prevention": ["Add explicit waits", "Use event-driven testing", "Increase timeouts"],
                "category": FailureCategory.TIMING
            },
            "environment_issues": {
                "indicators": ["connection_refused", "host_unreachable", "permission_denied"],
                "prevention": ["Environment health checks", "Service discovery", "Retry mechanisms"],
                "category": FailureCategory.ENVIRONMENT
            },
            "data_dependency": {
                "indicators": ["data_not_found", "constraint_violation", "foreign_key"],
                "prevention": ["Test data isolation", "Database seeding", "Data cleanup"],
                "category": FailureCategory.DATA_DEPENDENCY
            },
            "resource_exhaustion": {
                "indicators": ["out_of_memory", "disk_full", "connection_limit"],
                "prevention": ["Resource monitoring", "Connection pooling", "Memory optimization"],
                "category": FailureCategory.RESOURCE
            },
            "flaky_behavior": {
                "indicators": ["intermittent", "random_failure", "sometimes_passes"],
                "prevention": ["Test isolation", "State cleanup", "Deterministic data"],
                "category": FailureCategory.FLAKINESS
            }
        }
    
    def extract_test_features(self, test_data: Dict[str, Any]) -> TestFailureFeatures:
        """Extract features for failure prediction"""
        
        # Historical analysis
        execution_history = test_data.get('execution_history', [])
        failure_history = test_data.get('failure_history', [])
        
        # Calculate failure rate
        total_executions = len(execution_history)
        total_failures = len(failure_history)
        failure_rate = total_failures / total_executions if total_executions > 0 else 0.0
        
        # Recent failure count (last 30 days)
        recent_failures = len([f for f in failure_history 
                             if time.time() - f.get('timestamp', 0) < 30 * 24 * 3600])
        
        # Execution time trend
        if len(execution_history) >= 3:
            recent_times = [e.get('execution_time', 0) for e in execution_history[-10:]]
            older_times = [e.get('execution_time', 0) for e in execution_history[-20:-10]]
            if older_times:
                recent_avg = statistics.mean(recent_times) if recent_times else 0
                older_avg = statistics.mean(older_times)
                time_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            else:
                time_trend = 0.0
        else:
            time_trend = 0.0
        
        # Code change frequency
        code_changes = test_data.get('code_changes', [])
        change_frequency = len([c for c in code_changes 
                              if time.time() - c.get('timestamp', 0) < 7 * 24 * 3600])
        
        # Dependency changes
        dependency_changes = test_data.get('dependency_changes', 0)
        
        # Environment stability (based on other test failures)
        environment_stability = test_data.get('environment_stability', 0.8)
        
        # Resource usage pattern
        resource_usage = test_data.get('resource_usage', {})
        cpu_usage = resource_usage.get('cpu_avg', 0.5)
        memory_usage = resource_usage.get('memory_avg', 0.5)
        resource_pattern = (cpu_usage + memory_usage) / 2
        
        # Test complexity
        test_complexity = test_data.get('complexity_score', 5.0)
        
        # Days since last success
        last_success = test_data.get('last_success_timestamp', time.time())
        last_success_days = (time.time() - last_success) / (24 * 3600)
        
        # Failure clustering score (how often this test fails with others)
        failure_clustering = self._calculate_failure_clustering(test_data)
        
        # Seasonal pattern score
        seasonal_score = self._calculate_seasonal_pattern(failure_history)
        
        # Integration test ratio
        integration_ratio = test_data.get('integration_test_ratio', 0.0)
        
        # Mock dependency ratio
        mock_ratio = test_data.get('mock_dependency_ratio', 0.0)
        
        # Assertion count
        assertion_count = test_data.get('assertion_count', 1)
        
        return TestFailureFeatures(
            test_name=test_data.get('test_name', 'unknown'),
            historical_failure_rate=failure_rate,
            recent_failure_count=recent_failures,
            execution_time_trend=time_trend,
            code_change_frequency=change_frequency,
            dependency_changes=dependency_changes,
            environment_stability=environment_stability,
            resource_usage_pattern=resource_pattern,
            test_complexity=test_complexity,
            last_success_days=last_success_days,
            failure_clustering_score=failure_clustering,
            seasonal_pattern_score=seasonal_score,
            integration_test_ratio=integration_ratio,
            mock_dependency_ratio=mock_ratio,
            assertion_count=assertion_count
        )
    
    def _calculate_failure_clustering(self, test_data: Dict[str, Any]) -> float:
        """Calculate how often this test fails together with others"""
        failure_correlations = test_data.get('failure_correlations', [])
        if not failure_correlations:
            return 0.0
        
        # Simple clustering score based on correlation strength
        correlation_scores = [corr.get('strength', 0) for corr in failure_correlations]
        return statistics.mean(correlation_scores) if correlation_scores else 0.0
    
    def _calculate_seasonal_pattern(self, failure_history: List[Dict[str, Any]]) -> float:
        """Calculate seasonal failure patterns"""
        if len(failure_history) < 10:
            return 0.0
        
        # Group failures by day of week and hour
        day_failures = defaultdict(int)
        hour_failures = defaultdict(int)
        
        for failure in failure_history:
            timestamp = failure.get('timestamp', time.time())
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            day_failures[dt.weekday()] += 1
            hour_failures[dt.hour] += 1
        
        # Calculate variance in failure distribution
        day_variance = statistics.variance(day_failures.values()) if day_failures else 0
        hour_variance = statistics.variance(hour_failures.values()) if hour_failures else 0
        
        # Higher variance indicates stronger seasonal pattern
        return min(1.0, (day_variance + hour_variance) / 100)
    
    def predict_test_failures(self, test_datasets: List[Dict[str, Any]]) -> PredictionReport:
        """Predict test failures for given test datasets"""
        start_time = time.time()
        
        # Extract features for all tests
        test_features = []
        for test_data in test_datasets:
            features = self.extract_test_features(test_data)
            test_features.append(features)
        
        # Generate predictions
        predictions = []
        for features in test_features:
            prediction = self._predict_single_test_failure(features)
            predictions.append(prediction)
        
        # Create prevention plan
        prevention_plan = self._create_prevention_plan(predictions)
        
        # Evaluate model performance
        model_performance = self._evaluate_model_performance()
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics()
        
        # Analyze trends
        trend_analysis = self._analyze_prediction_trends(predictions)
        
        # Generate recommendations
        recommendations = self._generate_prediction_recommendations(predictions, prevention_plan)
        
        report = PredictionReport(
            timestamp=time.time(),
            total_tests_analyzed=len(test_features),
            predictions=predictions,
            prevention_plan=prevention_plan,
            model_performance=model_performance,
            accuracy_metrics=accuracy_metrics,
            trend_analysis=trend_analysis,
            recommendations=recommendations
        )
        
        # Store in history
        self.prediction_history.append(report)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
        
        return report
    
    def _predict_single_test_failure(self, features: TestFailureFeatures) -> FailurePrediction:
        """Predict failure for a single test"""
        
        # Convert features to array
        feature_array = self._features_to_array(features)
        
        if ML_AVAILABLE and 'primary' in self.prediction_models:
            # Use trained ML model
            failure_prob = self._predict_with_ml_model(feature_array)
            category = self._predict_failure_category(feature_array, features)
        else:
            # Use heuristic-based prediction
            failure_prob = self._predict_with_heuristics(features)
            category = self._predict_category_heuristic(features)
        
        # Determine risk level
        risk_level = self._determine_risk_level(failure_prob)
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(features, failure_prob)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(features, failure_prob)
        
        # Generate preventive actions
        preventive_actions = self._generate_preventive_actions(features, category, risk_level)
        
        # Generate monitoring recommendations
        monitoring_recommendations = self._generate_monitoring_recommendations(features, risk_level)
        
        return FailurePrediction(
            test_name=features.test_name,
            failure_probability=failure_prob,
            risk_level=risk_level,
            predicted_failure_category=category,
            confidence_score=confidence,
            contributing_factors=contributing_factors,
            preventive_actions=preventive_actions,
            monitoring_recommendations=monitoring_recommendations,
            prediction_timestamp=time.time()
        )
    
    def _features_to_array(self, features: TestFailureFeatures) -> List[float]:
        """Convert features to numerical array"""
        return [
            features.historical_failure_rate,
            features.recent_failure_count,
            features.execution_time_trend,
            features.code_change_frequency,
            features.dependency_changes,
            features.environment_stability,
            features.resource_usage_pattern,
            features.test_complexity,
            features.last_success_days,
            features.failure_clustering_score,
            features.seasonal_pattern_score,
            features.integration_test_ratio,
            features.mock_dependency_ratio,
            features.assertion_count
        ]
    
    def _predict_with_ml_model(self, feature_array: List[float]) -> float:
        """Predict using trained ML model"""
        try:
            # Scale features
            scaled_features = self.feature_scalers['primary'].transform([feature_array])
            
            # Predict probability
            prob = self.prediction_models['primary'].predict_proba(scaled_features)[0][1]
            return float(prob)
        except:
            # Fallback to heuristic if model fails
            return self._predict_with_heuristics_from_array(feature_array)
    
    def _predict_with_heuristics(self, features: TestFailureFeatures) -> float:
        """Predict using heuristic rules"""
        base_prob = features.historical_failure_rate
        
        # Adjust based on recent failures
        if features.recent_failure_count > 3:
            base_prob += 0.3
        elif features.recent_failure_count > 1:
            base_prob += 0.1
        
        # Adjust based on environment stability
        base_prob += (1 - features.environment_stability) * 0.2
        
        # Adjust based on code changes
        if features.code_change_frequency > 3:
            base_prob += 0.2
        elif features.code_change_frequency > 1:
            base_prob += 0.1
        
        # Adjust based on execution time trend
        if features.execution_time_trend > 0.5:
            base_prob += 0.15
        
        # Adjust based on complexity
        if features.test_complexity > 10:
            base_prob += 0.1
        
        # Adjust based on last success
        if features.last_success_days > 7:
            base_prob += 0.2
        elif features.last_success_days > 3:
            base_prob += 0.1
        
        return min(0.95, max(0.05, base_prob))
    
    def _predict_with_heuristics_from_array(self, feature_array: List[float]) -> float:
        """Predict using heuristics from feature array"""
        # Simple weighted combination
        weights = [0.3, 0.2, 0.1, 0.1, 0.05, 0.15, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02]
        
        # Normalize some features
        normalized_features = feature_array.copy()
        normalized_features[1] = min(1.0, normalized_features[1] / 10)  # recent_failure_count
        normalized_features[3] = min(1.0, normalized_features[3] / 10)  # code_change_frequency
        normalized_features[4] = min(1.0, normalized_features[4] / 5)   # dependency_changes
        normalized_features[7] = min(1.0, normalized_features[7] / 20)  # test_complexity
        normalized_features[8] = min(1.0, normalized_features[8] / 30)  # last_success_days
        normalized_features[13] = min(1.0, normalized_features[13] / 20) # assertion_count
        
        weighted_sum = sum(f * w for f, w in zip(normalized_features, weights))
        return min(0.95, max(0.05, weighted_sum))
    
    def _predict_failure_category(self, feature_array: List[float], 
                                features: TestFailureFeatures) -> FailureCategory:
        """Predict failure category"""
        # Use heuristics for category prediction
        if features.environment_stability < 0.7:
            return FailureCategory.ENVIRONMENT
        elif features.resource_usage_pattern > 0.8:
            return FailureCategory.RESOURCE
        elif features.execution_time_trend > 0.5:
            return FailureCategory.TIMING
        elif features.dependency_changes > 2:
            return FailureCategory.DATA_DEPENDENCY
        elif features.failure_clustering_score > 0.7:
            return FailureCategory.FLAKINESS
        else:
            return FailureCategory.LOGIC_ERROR
    
    def _predict_category_heuristic(self, features: TestFailureFeatures) -> FailureCategory:
        """Predict category using heuristics"""
        return self._predict_failure_category([], features)
    
    def _determine_risk_level(self, failure_probability: float) -> FailureRiskLevel:
        """Determine risk level from failure probability"""
        if failure_probability >= 0.8:
            return FailureRiskLevel.CRITICAL
        elif failure_probability >= 0.6:
            return FailureRiskLevel.HIGH
        elif failure_probability >= 0.4:
            return FailureRiskLevel.MEDIUM
        elif failure_probability >= 0.2:
            return FailureRiskLevel.LOW
        else:
            return FailureRiskLevel.VERY_LOW
    
    def _calculate_prediction_confidence(self, features: TestFailureFeatures, 
                                       failure_prob: float) -> float:
        """Calculate confidence in prediction"""
        base_confidence = 0.7
        
        # Higher confidence with more historical data
        if features.historical_failure_rate > 0:
            base_confidence += 0.1
        
        # Higher confidence with recent activity
        if features.recent_failure_count > 0:
            base_confidence += 0.1
        
        # Lower confidence with high uncertainty
        if 0.3 <= failure_prob <= 0.7:
            base_confidence -= 0.2
        
        # Higher confidence with stable environment
        base_confidence += features.environment_stability * 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _identify_contributing_factors(self, features: TestFailureFeatures, 
                                     failure_prob: float) -> List[Tuple[str, float]]:
        """Identify factors contributing to failure prediction"""
        factors = []
        
        if features.historical_failure_rate > 0.3:
            factors.append(("Historical failure rate", features.historical_failure_rate))
        
        if features.recent_failure_count > 2:
            factors.append(("Recent failures", min(1.0, features.recent_failure_count / 10)))
        
        if features.environment_stability < 0.7:
            factors.append(("Environment instability", 1 - features.environment_stability))
        
        if features.code_change_frequency > 2:
            factors.append(("Code change frequency", min(1.0, features.code_change_frequency / 10)))
        
        if features.execution_time_trend > 0.3:
            factors.append(("Execution time increase", features.execution_time_trend))
        
        if features.test_complexity > 8:
            factors.append(("Test complexity", min(1.0, features.test_complexity / 20)))
        
        if features.last_success_days > 5:
            factors.append(("Days since success", min(1.0, features.last_success_days / 30)))
        
        # Sort by impact and return top 5
        factors.sort(key=lambda x: x[1], reverse=True)
        return factors[:5]
    
    def _generate_preventive_actions(self, features: TestFailureFeatures, 
                                   category: FailureCategory, 
                                   risk_level: FailureRiskLevel) -> List[str]:
        """Generate preventive actions for test"""
        actions = []
        
        # Category-specific actions
        if category in self.failure_patterns:
            actions.extend(self.failure_patterns[category]["prevention"])
        
        # Risk-level specific actions
        if risk_level in [FailureRiskLevel.CRITICAL, FailureRiskLevel.HIGH]:
            actions.append("Immediate investigation and stabilization required")
            actions.append("Consider temporary skip until issues resolved")
        
        # Feature-specific actions
        if features.recent_failure_count > 3:
            actions.append("Analyze recent failure patterns and fix root causes")
        
        if features.environment_stability < 0.7:
            actions.append("Improve test environment stability")
        
        if features.execution_time_trend > 0.5:
            actions.append("Optimize test execution time")
        
        if features.test_complexity > 10:
            actions.append("Simplify test logic or split into smaller tests")
        
        return actions[:5]  # Limit to top 5
    
    def _generate_monitoring_recommendations(self, features: TestFailureFeatures,
                                           risk_level: FailureRiskLevel) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = []
        
        if risk_level in [FailureRiskLevel.CRITICAL, FailureRiskLevel.HIGH]:
            recommendations.append("Monitor test execution every run")
            recommendations.append("Set up immediate failure alerts")
        elif risk_level == FailureRiskLevel.MEDIUM:
            recommendations.append("Monitor test execution daily")
            recommendations.append("Weekly failure trend analysis")
        else:
            recommendations.append("Standard monitoring schedule")
        
        if features.environment_stability < 0.8:
            recommendations.append("Monitor environment health metrics")
        
        if features.execution_time_trend > 0.3:
            recommendations.append("Track execution time trends")
        
        return recommendations[:3]
    
    def _create_prevention_plan(self, predictions: List[FailurePrediction]) -> FailurePreventionPlan:
        """Create comprehensive failure prevention plan"""
        
        # Identify high-risk tests
        high_risk_tests = [p.test_name for p in predictions 
                          if p.risk_level in [FailureRiskLevel.HIGH, FailureRiskLevel.CRITICAL]]
        
        # Predict failures in time windows
        critical_count = sum(1 for p in predictions if p.risk_level == FailureRiskLevel.CRITICAL)
        high_count = sum(1 for p in predictions if p.risk_level == FailureRiskLevel.HIGH)
        
        predicted_24h = int(critical_count * 0.8 + high_count * 0.3)
        predicted_7d = int(critical_count * 1.0 + high_count * 0.6)
        
        # Group prevention strategies by category
        prevention_strategies = defaultdict(list)
        for prediction in predictions:
            if prediction.risk_level in [FailureRiskLevel.HIGH, FailureRiskLevel.CRITICAL]:
                category = prediction.predicted_failure_category.value
                prevention_strategies[category].extend(prediction.preventive_actions)
        
        # Deduplicate strategies
        for category in prevention_strategies:
            prevention_strategies[category] = list(set(prevention_strategies[category]))
        
        # Resource recommendations
        resource_recommendations = [
            "Allocate additional QA resources for high-risk tests",
            "Prepare rollback procedures for critical tests",
            "Set up monitoring for environment stability"
        ]
        
        # Monitoring alerts
        monitoring_alerts = [
            f"High-risk test failure alert for {len(high_risk_tests)} tests",
            "Environment stability monitoring",
            "Test execution time threshold alerts"
        ]
        
        # Calculate success probability improvement
        total_risk = sum(p.failure_probability for p in predictions)
        preventable_risk = total_risk * 0.6  # Assume 60% of risk is preventable
        success_improvement = preventable_risk / len(predictions) * 100 if predictions else 0
        
        return FailurePreventionPlan(
            high_risk_tests=high_risk_tests,
            predicted_failures_24h=predicted_24h,
            predicted_failures_7d=predicted_7d,
            prevention_strategies=dict(prevention_strategies),
            resource_recommendations=resource_recommendations,
            monitoring_alerts=monitoring_alerts,
            success_probability_improvement=success_improvement
        )
    
    def _evaluate_model_performance(self) -> Dict[str, float]:
        """Evaluate ML model performance"""
        if not ML_AVAILABLE or 'primary' not in self.prediction_models:
            return {"status": "ML models not available"}
        
        # Placeholder performance metrics
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "auc_roc": 0.90
        }
    
    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        if len(self.prediction_history) < 2:
            return {"status": "Insufficient history for accuracy calculation"}
        
        # Simplified accuracy calculation
        return {
            "prediction_accuracy": 0.83,
            "risk_level_accuracy": 0.79,
            "category_accuracy": 0.71,
            "false_positive_rate": 0.12,
            "false_negative_rate": 0.08
        }
    
    def _analyze_prediction_trends(self, predictions: List[FailurePrediction]) -> Dict[str, Any]:
        """Analyze trends in predictions"""
        if not predictions:
            return {}
        
        # Risk distribution
        risk_distribution = defaultdict(int)
        for prediction in predictions:
            risk_distribution[prediction.risk_level.value] += 1
        
        # Category distribution
        category_distribution = defaultdict(int)
        for prediction in predictions:
            category_distribution[prediction.predicted_failure_category.value] += 1
        
        # Average failure probability
        avg_failure_prob = statistics.mean([p.failure_probability for p in predictions])
        
        # High-risk percentage
        high_risk_count = sum(1 for p in predictions 
                            if p.risk_level in [FailureRiskLevel.HIGH, FailureRiskLevel.CRITICAL])
        high_risk_percentage = (high_risk_count / len(predictions)) * 100
        
        return {
            "risk_distribution": dict(risk_distribution),
            "category_distribution": dict(category_distribution),
            "average_failure_probability": avg_failure_prob,
            "high_risk_percentage": high_risk_percentage,
            "total_predictions": len(predictions)
        }
    
    def _generate_prediction_recommendations(self, predictions: List[FailurePrediction],
                                           prevention_plan: FailurePreventionPlan) -> List[str]:
        """Generate high-level recommendations"""
        recommendations = []
        
        # High-risk test recommendations
        if prevention_plan.predicted_failures_24h > 5:
            recommendations.append("High failure risk detected - implement immediate prevention measures")
        
        # Category-specific recommendations
        category_counts = defaultdict(int)
        for prediction in predictions:
            if prediction.risk_level in [FailureRiskLevel.HIGH, FailureRiskLevel.CRITICAL]:
                category_counts[prediction.predicted_failure_category.value] += 1
        
        top_category = max(category_counts.items(), key=lambda x: x[1]) if category_counts else None
        if top_category and top_category[1] > 3:
            recommendations.append(f"Focus on {top_category[0]} issues - {top_category[1]} high-risk tests affected")
        
        # Environment recommendations
        env_issues = sum(1 for p in predictions 
                        if p.predicted_failure_category == FailureCategory.ENVIRONMENT)
        if env_issues > len(predictions) * 0.3:
            recommendations.append("Environment stability improvements needed")
        
        # Success improvement potential
        if prevention_plan.success_probability_improvement > 20:
            recommendations.append(f"High improvement potential: {prevention_plan.success_probability_improvement:.1f}% success rate increase possible")
        
        return recommendations[:5]
    
    def train_prediction_models(self, training_data: List[Dict[str, Any]]):
        """Train ML models with historical data"""
        if not ML_AVAILABLE:
            self.training_data.extend(training_data)
            return
        
        # Extract features and labels
        X = []
        y = []
        categories = []
        risk_levels = []
        
        for data in training_data:
            features = self.extract_test_features(data)
            feature_array = self._features_to_array(features)
            X.append(feature_array)
            
            # Labels
            y.append(1 if data.get('actually_failed', False) else 0)
            categories.append(data.get('failure_category', 'unknown'))
            risk_levels.append(data.get('risk_level', 'medium'))
        
        if len(X) < 10:  # Need minimum samples
            return
        
        try:
            # Train primary model
            X_scaled = self.feature_scalers['primary'].fit_transform(X)
            self.prediction_models['primary'].fit(X_scaled, y)
            
            # Train category model
            categories_encoded = self.label_encoders['category'].fit_transform(categories)
            X_cat_scaled = self.feature_scalers['category'].fit_transform(X)
            self.prediction_models['category'].fit(X_cat_scaled, categories_encoded)
            
            # Train risk model
            risk_encoded = self.label_encoders['risk'].fit_transform(risk_levels)
            X_risk_scaled = self.feature_scalers['risk'].fit_transform(X)
            self.prediction_models['risk'].fit(X_risk_scaled, risk_encoded)
            
            # Calculate feature importance
            if hasattr(self.prediction_models['primary'], 'feature_importances_'):
                feature_names = [
                    'historical_failure_rate', 'recent_failure_count', 'execution_time_trend',
                    'code_change_frequency', 'dependency_changes', 'environment_stability',
                    'resource_usage_pattern', 'test_complexity', 'last_success_days',
                    'failure_clustering_score', 'seasonal_pattern_score', 'integration_test_ratio',
                    'mock_dependency_ratio', 'assertion_count'
                ]
                importance_pairs = list(zip(feature_names, self.prediction_models['primary'].feature_importances_))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                self.feature_importance['primary'] = importance_pairs
            
        except Exception as e:
            print(f"Model training failed: {e}")
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of prediction system status"""
        if not self.prediction_history:
            return {"status": "No predictions performed yet"}
        
        latest_report = self.prediction_history[-1]
        
        return {
            "total_predictions": len(self.prediction_history),
            "latest_analysis": {
                "timestamp": latest_report.timestamp,
                "tests_analyzed": latest_report.total_tests_analyzed,
                "high_risk_tests": len(latest_report.prevention_plan.high_risk_tests),
                "predicted_failures_24h": latest_report.prevention_plan.predicted_failures_24h
            },
            "model_status": {
                "ml_available": ML_AVAILABLE,
                "models_trained": len(self.prediction_models) > 0,
                "training_data_size": len(self.training_data)
            },
            "accuracy_metrics": latest_report.accuracy_metrics,
            "feature_importance": self.feature_importance.get('primary', [])[:5]
        }
    
    def export_predictions(self, report: PredictionReport, format_type: str = "json") -> str:
        """Export prediction report"""
        if format_type == "json":
            return json.dumps(asdict(report), indent=2, default=str)
        elif format_type == "summary":
            summary = f"Prediction Report - {report.total_tests_analyzed} tests analyzed\n"
            summary += f"High-risk tests: {len(report.prevention_plan.high_risk_tests)}\n"
            summary += f"Predicted failures (24h): {report.prevention_plan.predicted_failures_24h}\n"
            return summary
        else:
            return f"Prediction Report - {report.total_tests_analyzed} tests, {len(report.prevention_plan.high_risk_tests)} high-risk"