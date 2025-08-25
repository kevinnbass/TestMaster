"""
ML-Powered Test Intelligence Engine for TestMaster
Advanced test optimization, defect prediction, and intelligent recommendations
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class IntelligenceModel(Enum):
    """Available ML models"""
    DEFECT_PREDICTOR = "defect_predictor"
    TEST_PRIORITIZER = "test_prioritizer"
    COVERAGE_OPTIMIZER = "coverage_optimizer"
    MAINTENANCE_PREDICTOR = "maintenance_predictor"
    FLAKINESS_DETECTOR = "flakiness_detector"


class OptimizationGoal(Enum):
    """Optimization objectives"""
    MAXIMIZE_COVERAGE = "maximize_coverage"
    MINIMIZE_EXECUTION_TIME = "minimize_execution_time"
    MAXIMIZE_DEFECT_DETECTION = "maximize_defect_detection"
    MINIMIZE_MAINTENANCE = "minimize_maintenance"
    BALANCE_ALL = "balance_all"


@dataclass
class TestFeatures:
    """Features extracted from test cases"""
    test_id: str
    lines_of_code: int
    cyclomatic_complexity: int
    assertion_count: int
    mock_usage: bool
    execution_time: float
    failure_rate: float
    code_coverage: float
    dependencies_count: int
    last_modified_days: int
    defects_found: int
    maintenance_hours: float
    test_type: str  # unit, integration, system
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML"""
        return [
            self.lines_of_code,
            self.cyclomatic_complexity,
            self.assertion_count,
            1.0 if self.mock_usage else 0.0,
            self.execution_time,
            self.failure_rate,
            self.code_coverage,
            self.dependencies_count,
            self.last_modified_days,
            self.defects_found,
            self.maintenance_hours,
            {'unit': 0, 'integration': 1, 'system': 2}.get(self.test_type, 0)
        ]


@dataclass
class Prediction:
    """ML prediction result"""
    model_type: IntelligenceModel
    prediction: Any
    confidence: float
    explanation: List[str]
    feature_importance: Dict[str, float]


@dataclass
class OptimizationRecommendation:
    """Test optimization recommendation"""
    test_id: str
    recommendation_type: str
    priority: int
    description: str
    estimated_impact: Dict[str, float]
    implementation_effort: str  # low, medium, high


class DefectPredictor:
    """Predicts likelihood of defects in code areas"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'lines_of_code', 'cyclomatic_complexity', 'assertion_count',
            'mock_usage', 'execution_time', 'failure_rate', 'code_coverage',
            'dependencies_count', 'last_modified_days', 'defects_found',
            'maintenance_hours', 'test_type_encoded'
        ]
    
    def train(self, test_features: List[TestFeatures], defect_labels: List[bool]):
        """Train defect prediction model"""
        if len(test_features) < 10:
            return False  # Need minimum data
        
        # Convert to feature matrix
        X = np.array([tf.to_feature_vector() for tf in test_features])
        y = np.array(defect_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        return accuracy > 0.7  # Require decent accuracy
    
    def predict_defect_risk(self, test_features: TestFeatures) -> Prediction:
        """Predict defect risk for test area"""
        if not self.is_trained:
            return Prediction(
                model_type=IntelligenceModel.DEFECT_PREDICTOR,
                prediction=0.5,
                confidence=0.0,
                explanation=["Model not trained"],
                feature_importance={}
            )
        
        # Convert to feature vector
        X = np.array([test_features.to_feature_vector()])
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        prob = self.model.predict_proba(X_scaled)[0][1]  # Probability of defect
        
        # Get feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Generate explanation
        explanations = self._generate_defect_explanation(test_features, importance)
        
        return Prediction(
            model_type=IntelligenceModel.DEFECT_PREDICTOR,
            prediction=prob,
            confidence=max(abs(prob - 0.5) * 2, 0.1),  # Higher confidence for extreme predictions
            explanation=explanations,
            feature_importance=importance
        )
    
    def _generate_defect_explanation(self, features: TestFeatures, 
                                   importance: Dict[str, float]) -> List[str]:
        """Generate human-readable explanation"""
        explanations = []
        
        # Top importance factors
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, imp in sorted_importance[:3]:
            if imp > 0.1:  # Only explain important features
                if feature == 'failure_rate' and features.failure_rate > 0.1:
                    explanations.append(f"High failure rate ({features.failure_rate:.1%}) increases defect risk")
                elif feature == 'cyclomatic_complexity' and features.cyclomatic_complexity > 10:
                    explanations.append(f"High complexity ({features.cyclomatic_complexity}) indicates risk")
                elif feature == 'code_coverage' and features.code_coverage < 0.8:
                    explanations.append(f"Low coverage ({features.code_coverage:.1%}) suggests gaps")
                elif feature == 'last_modified_days' and features.last_modified_days < 30:
                    explanations.append("Recent changes increase instability risk")
        
        return explanations[:3]


class TestPrioritizer:
    """ML-based test prioritization"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, test_features: List[TestFeatures], priority_scores: List[float]):
        """Train prioritization model"""
        if len(test_features) < 10:
            return False
        
        X = np.array([tf.to_feature_vector() for tf in test_features])
        y = np.array(priority_scores)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training MSE
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        
        return mse < 0.1  # Require low error
    
    def predict_priority(self, test_features: TestFeatures) -> float:
        """Predict test priority score"""
        if not self.is_trained:
            return 0.5  # Default priority
        
        X = np.array([test_features.to_feature_vector()])
        X_scaled = self.scaler.transform(X)
        
        priority = self.model.predict(X_scaled)[0]
        return max(0, min(1, priority))  # Clamp to [0, 1]


class CoverageOptimizer:
    """Optimizes test selection for maximum coverage"""
    
    def __init__(self):
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        
    def optimize_test_selection(self, test_features: List[TestFeatures],
                               coverage_matrix: List[List[bool]],
                               time_budget: float) -> List[str]:
        """Select optimal tests within time budget"""
        
        if not test_features or not coverage_matrix:
            return []
        
        # Convert to numpy arrays
        feature_matrix = np.array([tf.to_feature_vector() for tf in test_features])
        coverage_array = np.array(coverage_matrix)
        
        # Cluster tests by similarity
        clusters = self.clustering_model.fit_predict(feature_matrix)
        
        # Select representative tests from each cluster
        selected_tests = []
        remaining_time = time_budget
        
        for cluster_id in range(max(clusters) + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Find test in cluster with best coverage/time ratio
            best_ratio = -1
            best_test_idx = None
            
            for idx in cluster_indices:
                test = test_features[idx]
                if test.execution_time <= remaining_time:
                    # Calculate coverage gained
                    coverage_gained = np.sum(coverage_array[idx])
                    ratio = coverage_gained / max(test.execution_time, 0.1)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_test_idx = idx
            
            if best_test_idx is not None:
                selected_tests.append(test_features[best_test_idx].test_id)
                remaining_time -= test_features[best_test_idx].execution_time
        
        return selected_tests


class TestIntelligence:
    """Main ML-powered test intelligence engine"""
    
    def __init__(self):
        self.defect_predictor = DefectPredictor()
        self.test_prioritizer = TestPrioritizer()
        self.coverage_optimizer = CoverageOptimizer()
        self.models_trained = False
        
    def train_models(self, historical_data: Dict[str, Any]) -> Dict[str, bool]:
        """Train all ML models with historical data"""
        results = {}
        
        test_features = historical_data.get('test_features', [])
        defect_labels = historical_data.get('defect_labels', [])
        priority_scores = historical_data.get('priority_scores', [])
        
        # Train defect predictor
        if test_features and defect_labels:
            results['defect_predictor'] = self.defect_predictor.train(test_features, defect_labels)
        
        # Train test prioritizer
        if test_features and priority_scores:
            results['test_prioritizer'] = self.test_prioritizer.train(test_features, priority_scores)
        
        # Coverage optimizer doesn't need training (uses clustering)
        results['coverage_optimizer'] = True
        
        self.models_trained = any(results.values())
        return results
    
    def generate_test_recommendations(self, test_features: List[TestFeatures],
                                    optimization_goal: OptimizationGoal) -> List[OptimizationRecommendation]:
        """Generate intelligent test recommendations"""
        recommendations = []
        
        for test in test_features:
            # Predict defect risk
            defect_prediction = self.defect_predictor.predict_defect_risk(test)
            
            # Generate recommendations based on analysis
            if defect_prediction.prediction > 0.7:
                recommendations.append(OptimizationRecommendation(
                    test_id=test.test_id,
                    recommendation_type="increase_coverage",
                    priority=1,
                    description=f"High defect risk ({defect_prediction.prediction:.1%}) - add more comprehensive tests",
                    estimated_impact={'defect_reduction': 0.3, 'coverage_increase': 0.2},
                    implementation_effort="medium"
                ))
            
            if test.execution_time > 30.0:  # Slow test
                recommendations.append(OptimizationRecommendation(
                    test_id=test.test_id,
                    recommendation_type="optimize_performance",
                    priority=2,
                    description=f"Slow execution ({test.execution_time:.1f}s) - consider optimization or mocking",
                    estimated_impact={'time_savings': test.execution_time * 0.5},
                    implementation_effort="medium"
                ))
            
            if test.failure_rate > 0.1:  # Flaky test
                recommendations.append(OptimizationRecommendation(
                    test_id=test.test_id,
                    recommendation_type="improve_reliability",
                    priority=1,
                    description=f"High failure rate ({test.failure_rate:.1%}) - investigate flakiness",
                    estimated_impact={'reliability_improvement': 0.8},
                    implementation_effort="high"
                ))
            
            if test.assertion_count < 2 and test.lines_of_code > 20:
                recommendations.append(OptimizationRecommendation(
                    test_id=test.test_id,
                    recommendation_type="improve_assertions",
                    priority=3,
                    description="Low assertion count - add more specific validations",
                    estimated_impact={'quality_improvement': 0.2},
                    implementation_effort="low"
                ))
        
        # Sort by priority and impact
        recommendations.sort(key=lambda r: (r.priority, -sum(r.estimated_impact.values())))
        return recommendations[:20]  # Top 20 recommendations
    
    def recommend_test_generation(self, code_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend new tests to generate"""
        recommendations = []
        
        uncovered_functions = code_analysis.get('uncovered_functions', [])
        complex_functions = code_analysis.get('complex_functions', [])
        security_patterns = code_analysis.get('security_patterns', [])
        
        # Recommend tests for uncovered functions
        for func in uncovered_functions:
            recommendations.append({
                'type': 'unit_test',
                'target': func,
                'priority': 'high' if func in complex_functions else 'medium',
                'reason': 'No test coverage',
                'test_types': ['unit', 'edge_cases']
            })
        
        # Recommend security tests
        for pattern in security_patterns:
            recommendations.append({
                'type': 'security_test',
                'target': pattern,
                'priority': 'high',
                'reason': 'Security-sensitive code detected',
                'test_types': ['security', 'fuzzing']
            })
        
        # Recommend integration tests for high-dependency modules
        high_dependency_modules = code_analysis.get('high_dependency_modules', [])
        for module in high_dependency_modules:
            recommendations.append({
                'type': 'integration_test',
                'target': module,
                'priority': 'medium',
                'reason': 'High dependency complexity',
                'test_types': ['integration', 'contract']
            })
        
        return recommendations
    
    def predict_maintenance_effort(self, test_features: TestFeatures) -> float:
        """Predict future maintenance effort for test"""
        # Heuristic-based prediction (could be ML model)
        base_effort = test_features.lines_of_code * 0.1  # Base effort
        
        # Complexity factor
        complexity_factor = min(test_features.cyclomatic_complexity / 10, 2.0)
        
        # Dependency factor
        dependency_factor = min(test_features.dependencies_count / 5, 1.5)
        
        # Flakiness factor
        flakiness_factor = 1 + test_features.failure_rate
        
        predicted_effort = base_effort * complexity_factor * dependency_factor * flakiness_factor
        return min(predicted_effort, 100.0)  # Cap at 100 hours
    
    def suggest_test_optimizations(self, test_suite_features: List[TestFeatures]) -> Dict[str, Any]:
        """Suggest overall test suite optimizations"""
        if not test_suite_features:
            return {}
        
        # Analyze current state
        total_execution_time = sum(t.execution_time for t in test_suite_features)
        avg_coverage = statistics.mean(t.code_coverage for t in test_suite_features)
        flaky_tests = [t for t in test_suite_features if t.failure_rate > 0.05]
        slow_tests = [t for t in test_suite_features if t.execution_time > 10.0]
        
        optimizations = {
            'current_metrics': {
                'total_execution_time': total_execution_time,
                'average_coverage': avg_coverage,
                'flaky_test_count': len(flaky_tests),
                'slow_test_count': len(slow_tests)
            },
            'optimizations': [],
            'estimated_improvements': {}
        }
        
        # Parallel execution opportunities
        if total_execution_time > 300:  # 5 minutes
            parallelizable = len([t for t in test_suite_features if t.dependencies_count < 2])
            time_savings = total_execution_time * 0.6 * (parallelizable / len(test_suite_features))
            
            optimizations['optimizations'].append({
                'type': 'parallel_execution',
                'description': f'Run {parallelizable} independent tests in parallel',
                'estimated_time_savings': time_savings
            })
        
        # Test consolidation opportunities
        redundant_tests = self._identify_redundant_tests(test_suite_features)
        if redundant_tests:
            optimizations['optimizations'].append({
                'type': 'test_consolidation',
                'description': f'Consolidate {len(redundant_tests)} redundant tests',
                'estimated_time_savings': sum(t.execution_time for t in redundant_tests) * 0.7
            })
        
        # Smart test selection
        if len(test_suite_features) > 50:
            optimizations['optimizations'].append({
                'type': 'smart_selection',
                'description': 'Use ML-based test selection for faster feedback',
                'estimated_time_savings': total_execution_time * 0.4
            })
        
        return optimizations
    
    def _identify_redundant_tests(self, test_features: List[TestFeatures]) -> List[TestFeatures]:
        """Identify potentially redundant tests"""
        redundant = []
        
        # Simple heuristic: tests with similar coverage and low defect detection
        for i, test1 in enumerate(test_features):
            for test2 in test_features[i+1:]:
                # Check similarity
                coverage_diff = abs(test1.code_coverage - test2.code_coverage)
                execution_diff = abs(test1.execution_time - test2.execution_time)
                
                if (coverage_diff < 0.1 and execution_diff < 5.0 and 
                    test1.defects_found == 0 and test2.defects_found == 0):
                    redundant.append(test2)
        
        return redundant
    
    def generate_intelligence_report(self, test_suite_features: List[TestFeatures]) -> Dict[str, Any]:
        """Generate comprehensive intelligence report"""
        recommendations = self.generate_test_recommendations(
            test_suite_features, OptimizationGoal.BALANCE_ALL
        )
        
        optimizations = self.suggest_test_optimizations(test_suite_features)
        
        # Calculate intelligence scores
        avg_defect_risk = statistics.mean(
            self.defect_predictor.predict_defect_risk(t).prediction 
            for t in test_suite_features
        ) if test_suite_features else 0
        
        return {
            'intelligence_summary': {
                'total_tests_analyzed': len(test_suite_features),
                'average_defect_risk': avg_defect_risk,
                'models_trained': self.models_trained,
                'recommendations_generated': len(recommendations)
            },
            'recommendations': [
                {
                    'test_id': r.test_id,
                    'type': r.recommendation_type,
                    'priority': r.priority,
                    'description': r.description,
                    'impact': r.estimated_impact,
                    'effort': r.implementation_effort
                }
                for r in recommendations[:10]
            ],
            'optimizations': optimizations,
            'next_actions': self._generate_next_actions(recommendations, optimizations)
        }
    
    def _generate_next_actions(self, recommendations: List[OptimizationRecommendation],
                              optimizations: Dict[str, Any]) -> List[str]:
        """Generate prioritized next actions"""
        actions = []
        
        # High priority recommendations
        high_priority = [r for r in recommendations if r.priority == 1]
        if high_priority:
            actions.append(f"Address {len(high_priority)} high-priority test issues immediately")
        
        # Performance optimizations
        if optimizations.get('optimizations'):
            total_savings = sum(
                opt.get('estimated_time_savings', 0) 
                for opt in optimizations['optimizations']
            )
            if total_savings > 60:  # > 1 minute savings
                actions.append(f"Implement performance optimizations for {total_savings:.0f}s time savings")
        
        # Model training
        if not self.models_trained:
            actions.append("Collect more historical data to train ML models")
        
        return actions[:5]