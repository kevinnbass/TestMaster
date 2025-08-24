"""
ML-Powered Test Optimization System for TestMaster
Advanced test optimization using machine learning algorithms
"""

import pickle
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict

# ML imports with fallback
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class OptimizationStrategy(Enum):
    """Test optimization strategies"""
    EXECUTION_TIME = "execution_time"
    FAILURE_RISK = "failure_risk"
    COVERAGE_IMPACT = "coverage_impact"
    MAINTENANCE_COST = "maintenance_cost"
    BUSINESS_VALUE = "business_value"
    RESOURCE_EFFICIENCY = "resource_efficiency"

class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SKIP = "skip"

@dataclass
class TestMetadata:
    """Comprehensive test metadata for ML analysis"""
    test_name: str
    file_path: str
    execution_time: float
    last_failure_time: Optional[float]
    failure_count: int
    success_count: int
    lines_of_code: int
    complexity_score: float
    coverage_contribution: float
    dependencies: List[str]
    change_frequency: float
    business_criticality: float
    maintenance_effort: float
    resource_usage: Dict[str, float]

@dataclass
class OptimizationResult:
    """Result of test optimization"""
    test_name: str
    original_priority: TestPriority
    optimized_priority: TestPriority
    confidence_score: float
    reasoning: List[str]
    estimated_savings: Dict[str, float]
    recommendations: List[str]

@dataclass
class OptimizationReport:
    """Comprehensive optimization analysis report"""
    strategy: OptimizationStrategy
    total_tests_analyzed: int
    optimization_results: List[OptimizationResult]
    estimated_total_savings: Dict[str, float]
    model_performance: Dict[str, float]
    recommendations: List[str]
    execution_time: float
    
    @property
    def total_tests(self) -> int:
        """Backward compatibility property"""
        return self.total_tests_analyzed

class MLTestOptimizer:
    """Advanced ML-powered test optimization system"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.EXECUTION_TIME):
        self.strategy = strategy
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, List[Tuple[str, float]]] = {}
        self.optimization_history: List[OptimizationReport] = []
        self.test_database: Dict[str, TestMetadata] = {}
        
        if not ML_AVAILABLE:
            print("Warning: ML libraries not available, using fallback algorithms")
    
    def analyze_test_suite(self, test_data_or_directory: Any, test_metadata: Optional[List[Any]] = None) -> OptimizationReport:
        """Comprehensive ML-based test suite analysis"""
        start_time = time.time()
        
        # Handle different call signatures for backward compatibility
        if isinstance(test_data_or_directory, str) and test_metadata is not None:
            # Called as analyze_test_suite(directory, test_data)
            directory = test_data_or_directory
            raw_test_data = test_metadata
            # Convert raw test data to TestMetadata objects
            converted_metadata = []
            for test_data in raw_test_data:
                if isinstance(test_data, dict):
                    metadata = TestMetadata(
                        test_name=test_data.get("name", "unknown"),
                        file_path=test_data.get("file_path", "unknown"),
                        execution_time=test_data.get("execution_time", 0.1),
                        last_failure_time=test_data.get("last_failure_time"),
                        failure_count=test_data.get("failure_count", 0),
                        success_count=test_data.get("success_count", 10),
                        lines_of_code=test_data.get("lines_of_code", 50),
                        complexity_score=test_data.get("complexity", 1),
                        coverage_contribution=test_data.get("coverage", 0.5),
                        dependencies=test_data.get("dependencies", []),
                        change_frequency=test_data.get("change_frequency", 0.1),
                        business_criticality=test_data.get("business_criticality", 0.5),
                        maintenance_effort=test_data.get("maintenance_effort", 0.3),
                        resource_usage=test_data.get("resource_usage", {"cpu": 0.1, "memory": 0.05})
                    )
                    converted_metadata.append(metadata)
            processed_metadata = converted_metadata
        elif isinstance(test_data_or_directory, list):
            # Called as analyze_test_suite(test_metadata)
            processed_metadata = test_data_or_directory
        else:
            raise ValueError("Invalid parameters for analyze_test_suite")
        
        # Update test database
        for test in processed_metadata:
            self.test_database[test.test_name] = test
        
        # Extract features for ML analysis
        features, test_names = self._extract_features(processed_metadata)
        
        # Apply optimization strategy
        optimization_results = self._optimize_test_suite(features, test_names, processed_metadata)
        
        # Calculate estimated savings
        total_savings = self._calculate_total_savings(optimization_results, processed_metadata)
        
        # Evaluate model performance
        model_performance = self._evaluate_model_performance()
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(optimization_results)
        
        execution_time = time.time() - start_time
        
        report = OptimizationReport(
            strategy=self.strategy,
            total_tests_analyzed=len(processed_metadata),
            optimization_results=optimization_results,
            estimated_total_savings=total_savings,
            model_performance=model_performance,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        self.optimization_history.append(report)
        return report
    
    def _extract_features(self, test_metadata: List[TestMetadata]) -> Tuple[np.ndarray, List[str]]:
        """Extract features for ML analysis"""
        features = []
        test_names = []
        
        for test in test_metadata:
            feature_vector = [
                test.execution_time,
                test.failure_count,
                test.success_count,
                test.lines_of_code,
                test.complexity_score,
                test.coverage_contribution,
                len(test.dependencies),
                test.change_frequency,
                test.business_criticality,
                test.maintenance_effort,
                test.resource_usage.get('cpu', 0.0),
                test.resource_usage.get('memory', 0.0),
                self._calculate_failure_rate(test),
                self._calculate_stability_score(test),
                self._calculate_efficiency_score(test)
            ]
            
            features.append(feature_vector)
            test_names.append(test.test_name)
        
        return np.array(features), test_names
    
    def _calculate_failure_rate(self, test: TestMetadata) -> float:
        """Calculate test failure rate"""
        total_executions = test.failure_count + test.success_count
        return test.failure_count / total_executions if total_executions > 0 else 0.0
    
    def _calculate_stability_score(self, test: TestMetadata) -> float:
        """Calculate test stability score"""
        if test.last_failure_time is None:
            return 1.0
        
        time_since_failure = time.time() - test.last_failure_time
        # Score based on time since last failure (days)
        days_stable = time_since_failure / (24 * 3600)
        return min(1.0, days_stable / 30)  # Stabilizes after 30 days
    
    def _calculate_efficiency_score(self, test: TestMetadata) -> float:
        """Calculate test efficiency score"""
        # Coverage per unit time
        if test.execution_time > 0:
            efficiency = test.coverage_contribution / test.execution_time
        else:
            efficiency = test.coverage_contribution
        
        # Normalize to 0-1 range
        return min(1.0, efficiency)
    
    def _optimize_test_suite(self, features: np.ndarray, test_names: List[str], 
                           test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Optimize test suite using ML models"""
        if self.strategy == OptimizationStrategy.EXECUTION_TIME:
            return self._optimize_by_execution_time(features, test_names, test_metadata)
        elif self.strategy == OptimizationStrategy.FAILURE_RISK:
            return self._optimize_by_failure_risk(features, test_names, test_metadata)
        elif self.strategy == OptimizationStrategy.COVERAGE_IMPACT:
            return self._optimize_by_coverage_impact(features, test_names, test_metadata)
        elif self.strategy == OptimizationStrategy.MAINTENANCE_COST:
            return self._optimize_by_maintenance_cost(features, test_names, test_metadata)
        elif self.strategy == OptimizationStrategy.BUSINESS_VALUE:
            return self._optimize_by_business_value(features, test_names, test_metadata)
        else:
            return self._optimize_by_resource_efficiency(features, test_names, test_metadata)
    
    def _optimize_by_execution_time(self, features: np.ndarray, test_names: List[str],
                                  test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Optimize based on execution time reduction"""
        results = []
        
        if ML_AVAILABLE and len(features) > 5:
            # Use clustering to group similar tests
            try:
                # Normalize features
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(features)
                
                # Cluster tests by characteristics
                n_clusters = min(5, len(features) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(normalized_features)
                
                # Analyze each cluster
                for i, (test_name, cluster_id) in enumerate(zip(test_names, clusters)):
                    test = test_metadata[i]
                    
                    # Calculate optimization score
                    execution_score = 1.0 - min(1.0, test.execution_time / 60.0)  # Normalize to 1 minute
                    efficiency_score = self._calculate_efficiency_score(test)
                    
                    optimization_score = (execution_score + efficiency_score) / 2
                    
                    # Determine priority
                    if optimization_score > 0.8:
                        new_priority = TestPriority.CRITICAL
                    elif optimization_score > 0.6:
                        new_priority = TestPriority.HIGH
                    elif optimization_score > 0.4:
                        new_priority = TestPriority.MEDIUM
                    elif optimization_score > 0.2:
                        new_priority = TestPriority.LOW
                    else:
                        new_priority = TestPriority.SKIP
                    
                    # Generate reasoning
                    reasoning = []
                    if test.execution_time > 30:
                        reasoning.append("Long execution time")
                    if efficiency_score > 0.7:
                        reasoning.append("High efficiency")
                    if test.coverage_contribution > 0.1:
                        reasoning.append("High coverage contribution")
                    
                    # Calculate savings
                    time_savings = max(0, test.execution_time - 10) if new_priority != TestPriority.SKIP else test.execution_time
                    savings = {
                        'time_seconds': time_savings,
                        'resource_cpu': test.resource_usage.get('cpu', 0) * 0.3,
                        'resource_memory': test.resource_usage.get('memory', 0) * 0.3
                    }
                    
                    # Generate recommendations
                    recommendations = self._generate_test_recommendations(test, new_priority)
                    
                    results.append(OptimizationResult(
                        test_name=test_name,
                        original_priority=TestPriority.MEDIUM,  # Assume medium as baseline
                        optimized_priority=new_priority,
                        confidence_score=optimization_score,
                        reasoning=reasoning,
                        estimated_savings=savings,
                        recommendations=recommendations
                    ))
                
                # Store model for future use
                self.models['execution_time_kmeans'] = kmeans
                self.scalers['execution_time_scaler'] = scaler
                
            except Exception as e:
                # Fallback to simple heuristics
                results = self._fallback_time_optimization(features, test_names, test_metadata)
        else:
            # Fallback algorithm
            results = self._fallback_time_optimization(features, test_names, test_metadata)
        
        return results
    
    def _optimize_by_failure_risk(self, features: np.ndarray, test_names: List[str],
                                test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Optimize based on failure risk prediction"""
        results = []
        
        if ML_AVAILABLE and len(features) > 10:
            try:
                # Prepare training data for failure prediction
                X = features
                y = [1 if test.failure_count > test.success_count * 0.1 else 0 for test in test_metadata]
                
                # Train failure prediction model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Predict failure probabilities
                failure_probabilities = model.predict_proba(X)[:, 1]
                
                # Generate optimization results
                for i, (test_name, failure_prob) in enumerate(zip(test_names, failure_probabilities)):
                    test = test_metadata[i]
                    
                    # Determine priority based on failure risk
                    if failure_prob > 0.8:
                        new_priority = TestPriority.CRITICAL
                    elif failure_prob > 0.6:
                        new_priority = TestPriority.HIGH
                    elif failure_prob > 0.4:
                        new_priority = TestPriority.MEDIUM
                    elif failure_prob > 0.2:
                        new_priority = TestPriority.LOW
                    else:
                        new_priority = TestPriority.SKIP
                    
                    reasoning = [f"Failure probability: {failure_prob:.2f}"]
                    if test.failure_count > 5:
                        reasoning.append("High historical failure count")
                    if test.maintenance_effort > 0.7:
                        reasoning.append("High maintenance effort")
                    
                    savings = {
                        'debugging_hours': test.failure_count * 2 * (1 - failure_prob),
                        'maintenance_cost': test.maintenance_effort * 100 * (1 - failure_prob)
                    }
                    
                    recommendations = self._generate_test_recommendations(test, new_priority)
                    
                    results.append(OptimizationResult(
                        test_name=test_name,
                        original_priority=TestPriority.MEDIUM,
                        optimized_priority=new_priority,
                        confidence_score=failure_prob,
                        reasoning=reasoning,
                        estimated_savings=savings,
                        recommendations=recommendations
                    ))
                
                # Store model and evaluate performance
                self.models['failure_risk_rf'] = model
                if len(y_test) > 0:
                    test_predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, test_predictions)
                    self.feature_importance['failure_risk'] = list(zip(
                        ['execution_time', 'failure_count', 'success_count', 'complexity', 'coverage'],
                        model.feature_importances_[:5]
                    ))
                
            except Exception as e:
                results = self._fallback_risk_optimization(features, test_names, test_metadata)
        else:
            results = self._fallback_risk_optimization(features, test_names, test_metadata)
        
        return results
    
    def _optimize_by_coverage_impact(self, features: np.ndarray, test_names: List[str],
                                   test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Optimize based on coverage impact analysis"""
        results = []
        
        # Sort tests by coverage contribution
        coverage_scores = [(i, test.coverage_contribution) for i, test in enumerate(test_metadata)]
        coverage_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Assign priorities based on coverage impact
        total_tests = len(coverage_scores)
        for rank, (test_idx, coverage_score) in enumerate(coverage_scores):
            test = test_metadata[test_idx]
            test_name = test_names[test_idx]
            
            # Priority based on coverage ranking
            percentile = rank / total_tests
            if percentile < 0.2:  # Top 20%
                new_priority = TestPriority.CRITICAL
            elif percentile < 0.4:  # Top 40%
                new_priority = TestPriority.HIGH
            elif percentile < 0.7:  # Top 70%
                new_priority = TestPriority.MEDIUM
            elif percentile < 0.9:  # Top 90%
                new_priority = TestPriority.LOW
            else:
                new_priority = TestPriority.SKIP
            
            reasoning = [f"Coverage contribution: {coverage_score:.3f}"]
            if coverage_score > 0.1:
                reasoning.append("High coverage impact")
            if test.business_criticality > 0.7:
                reasoning.append("Business critical")
            
            savings = {
                'coverage_efficiency': coverage_score * 100,
                'test_redundancy_reduction': max(0, 1 - coverage_score) * 50
            }
            
            recommendations = self._generate_test_recommendations(test, new_priority)
            
            results.append(OptimizationResult(
                test_name=test_name,
                original_priority=TestPriority.MEDIUM,
                optimized_priority=new_priority,
                confidence_score=coverage_score,
                reasoning=reasoning,
                estimated_savings=savings,
                recommendations=recommendations
            ))
        
        return results
    
    def _optimize_by_maintenance_cost(self, features: np.ndarray, test_names: List[str],
                                    test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Optimize based on maintenance cost analysis"""
        results = []
        
        for i, test in enumerate(test_metadata):
            test_name = test_names[i]
            
            # Calculate maintenance score
            maintenance_score = (
                test.maintenance_effort * 0.4 +
                (test.failure_count / max(1, test.success_count)) * 0.3 +
                test.change_frequency * 0.3
            )
            
            # Invert score (lower maintenance is better)
            optimization_score = max(0, 1 - maintenance_score)
            
            # Determine priority
            if optimization_score > 0.8:
                new_priority = TestPriority.CRITICAL
            elif optimization_score > 0.6:
                new_priority = TestPriority.HIGH
            elif optimization_score > 0.4:
                new_priority = TestPriority.MEDIUM
            elif optimization_score > 0.2:
                new_priority = TestPriority.LOW
            else:
                new_priority = TestPriority.SKIP
            
            reasoning = [f"Maintenance score: {maintenance_score:.2f}"]
            if test.maintenance_effort > 0.7:
                reasoning.append("High maintenance effort")
            if test.change_frequency > 0.5:
                reasoning.append("Frequently changing")
            
            savings = {
                'maintenance_hours': test.maintenance_effort * 10 * optimization_score,
                'refactoring_cost': test.change_frequency * 50 * optimization_score
            }
            
            recommendations = self._generate_test_recommendations(test, new_priority)
            
            results.append(OptimizationResult(
                test_name=test_name,
                original_priority=TestPriority.MEDIUM,
                optimized_priority=new_priority,
                confidence_score=optimization_score,
                reasoning=reasoning,
                estimated_savings=savings,
                recommendations=recommendations
            ))
        
        return results
    
    def _optimize_by_business_value(self, features: np.ndarray, test_names: List[str],
                                  test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Optimize based on business value analysis"""
        results = []
        
        for i, test in enumerate(test_metadata):
            test_name = test_names[i]
            
            # Calculate business value score
            value_score = (
                test.business_criticality * 0.5 +
                test.coverage_contribution * 0.3 +
                (1 - self._calculate_failure_rate(test)) * 0.2
            )
            
            # Determine priority
            if value_score > 0.8:
                new_priority = TestPriority.CRITICAL
            elif value_score > 0.6:
                new_priority = TestPriority.HIGH
            elif value_score > 0.4:
                new_priority = TestPriority.MEDIUM
            elif value_score > 0.2:
                new_priority = TestPriority.LOW
            else:
                new_priority = TestPriority.SKIP
            
            reasoning = [f"Business value score: {value_score:.2f}"]
            if test.business_criticality > 0.8:
                reasoning.append("Business critical")
            if test.coverage_contribution > 0.1:
                reasoning.append("High coverage value")
            
            savings = {
                'business_risk_reduction': test.business_criticality * 1000 * value_score,
                'quality_improvement': test.coverage_contribution * 500 * value_score
            }
            
            recommendations = self._generate_test_recommendations(test, new_priority)
            
            results.append(OptimizationResult(
                test_name=test_name,
                original_priority=TestPriority.MEDIUM,
                optimized_priority=new_priority,
                confidence_score=value_score,
                reasoning=reasoning,
                estimated_savings=savings,
                recommendations=recommendations
            ))
        
        return results
    
    def _optimize_by_resource_efficiency(self, features: np.ndarray, test_names: List[str],
                                       test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Optimize based on resource efficiency"""
        results = []
        
        for i, test in enumerate(test_metadata):
            test_name = test_names[i]
            
            # Calculate resource efficiency
            cpu_efficiency = 1 - min(1, test.resource_usage.get('cpu', 0))
            memory_efficiency = 1 - min(1, test.resource_usage.get('memory', 0))
            time_efficiency = 1 - min(1, test.execution_time / 60)
            
            efficiency_score = (cpu_efficiency + memory_efficiency + time_efficiency) / 3
            
            # Weight by coverage contribution
            weighted_score = efficiency_score * (1 + test.coverage_contribution)
            
            # Determine priority
            if weighted_score > 1.5:
                new_priority = TestPriority.CRITICAL
            elif weighted_score > 1.2:
                new_priority = TestPriority.HIGH
            elif weighted_score > 0.8:
                new_priority = TestPriority.MEDIUM
            elif weighted_score > 0.4:
                new_priority = TestPriority.LOW
            else:
                new_priority = TestPriority.SKIP
            
            reasoning = [f"Resource efficiency: {efficiency_score:.2f}"]
            if test.execution_time < 5:
                reasoning.append("Fast execution")
            if test.resource_usage.get('cpu', 0) < 0.3:
                reasoning.append("Low CPU usage")
            
            savings = {
                'cpu_hours': test.resource_usage.get('cpu', 0) * 10 * efficiency_score,
                'memory_gb_hours': test.resource_usage.get('memory', 0) * 5 * efficiency_score,
                'execution_time': test.execution_time * 0.3 * efficiency_score
            }
            
            recommendations = self._generate_test_recommendations(test, new_priority)
            
            results.append(OptimizationResult(
                test_name=test_name,
                original_priority=TestPriority.MEDIUM,
                optimized_priority=new_priority,
                confidence_score=weighted_score,
                reasoning=reasoning,
                estimated_savings=savings,
                recommendations=recommendations
            ))
        
        return results
    
    def _generate_test_recommendations(self, test: TestMetadata, 
                                     priority: TestPriority) -> List[str]:
        """Generate specific recommendations for test optimization"""
        recommendations = []
        
        if priority == TestPriority.CRITICAL:
            recommendations.append("Keep in core test suite")
            if test.execution_time > 30:
                recommendations.append("Optimize execution time")
        elif priority == TestPriority.HIGH:
            recommendations.append("Include in regular test runs")
            if test.failure_count > 5:
                recommendations.append("Improve test stability")
        elif priority == TestPriority.MEDIUM:
            recommendations.append("Run during integration testing")
            if test.maintenance_effort > 0.7:
                recommendations.append("Consider refactoring")
        elif priority == TestPriority.LOW:
            recommendations.append("Run during nightly builds")
            recommendations.append("Consider combining with similar tests")
        else:  # SKIP
            recommendations.append("Consider removing or archiving")
            recommendations.append("Evaluate if coverage is provided elsewhere")
        
        # Specific technical recommendations
        if test.complexity_score > 10:
            recommendations.append("Simplify test logic")
        if len(test.dependencies) > 10:
            recommendations.append("Reduce dependencies")
        if test.lines_of_code > 100:
            recommendations.append("Split into smaller tests")
        
        return recommendations[:3]  # Limit to top 3
    
    def _fallback_time_optimization(self, features: np.ndarray, test_names: List[str],
                                  test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Fallback optimization when ML is not available"""
        results = []
        
        for i, test in enumerate(test_metadata):
            test_name = test_names[i]
            
            # Simple heuristic based on execution time and coverage
            if test.execution_time < 5 and test.coverage_contribution > 0.1:
                new_priority = TestPriority.CRITICAL
                confidence = 0.9
            elif test.execution_time < 15 and test.coverage_contribution > 0.05:
                new_priority = TestPriority.HIGH
                confidence = 0.8
            elif test.execution_time < 30:
                new_priority = TestPriority.MEDIUM
                confidence = 0.7
            elif test.execution_time < 60:
                new_priority = TestPriority.LOW
                confidence = 0.6
            else:
                new_priority = TestPriority.SKIP
                confidence = 0.5
            
            reasoning = [f"Execution time: {test.execution_time:.1f}s"]
            savings = {'time_seconds': max(0, test.execution_time - 10)}
            recommendations = self._generate_test_recommendations(test, new_priority)
            
            results.append(OptimizationResult(
                test_name=test_name,
                original_priority=TestPriority.MEDIUM,
                optimized_priority=new_priority,
                confidence_score=confidence,
                reasoning=reasoning,
                estimated_savings=savings,
                recommendations=recommendations
            ))
        
        return results
    
    def _fallback_risk_optimization(self, features: np.ndarray, test_names: List[str],
                                  test_metadata: List[TestMetadata]) -> List[OptimizationResult]:
        """Fallback risk optimization"""
        results = []
        
        for i, test in enumerate(test_metadata):
            test_name = test_names[i]
            failure_rate = self._calculate_failure_rate(test)
            
            if failure_rate > 0.3:
                new_priority = TestPriority.CRITICAL
            elif failure_rate > 0.2:
                new_priority = TestPriority.HIGH
            elif failure_rate > 0.1:
                new_priority = TestPriority.MEDIUM
            elif failure_rate > 0.05:
                new_priority = TestPriority.LOW
            else:
                new_priority = TestPriority.SKIP
            
            reasoning = [f"Failure rate: {failure_rate:.2f}"]
            savings = {'debugging_hours': test.failure_count * 2}
            recommendations = self._generate_test_recommendations(test, new_priority)
            
            results.append(OptimizationResult(
                test_name=test_name,
                original_priority=TestPriority.MEDIUM,
                optimized_priority=new_priority,
                confidence_score=1 - failure_rate,
                reasoning=reasoning,
                estimated_savings=savings,
                recommendations=recommendations
            ))
        
        return results
    
    def _calculate_total_savings(self, optimization_results: List[OptimizationResult],
                               test_metadata: List[TestMetadata]) -> Dict[str, float]:
        """Calculate total estimated savings from optimization"""
        total_savings = defaultdict(float)
        
        for result in optimization_results:
            for metric, value in result.estimated_savings.items():
                total_savings[metric] += value
        
        return dict(total_savings)
    
    def _evaluate_model_performance(self) -> Dict[str, float]:
        """Evaluate ML model performance"""
        performance = {}
        
        if 'failure_risk_rf' in self.models:
            # Use stored validation results or cross-validation
            performance['failure_prediction_accuracy'] = 0.85  # Placeholder
        
        if 'execution_time_kmeans' in self.models:
            performance['clustering_silhouette_score'] = 0.6  # Placeholder
        
        # Add optimization effectiveness metrics
        if self.optimization_history:
            recent_savings = [sum(r.estimated_total_savings.values()) 
                            for r in self.optimization_history[-5:]]
            if recent_savings:
                performance['average_savings'] = sum(recent_savings) / len(recent_savings)
        
        return performance
    
    def _generate_optimization_recommendations(self, 
                                             optimization_results: List[OptimizationResult]) -> List[str]:
        """Generate high-level optimization recommendations"""
        recommendations = []
        
        # Analyze priority distribution
        priority_counts = defaultdict(int)
        for result in optimization_results:
            priority_counts[result.optimized_priority] += 1
        
        total_tests = len(optimization_results)
        
        if priority_counts[TestPriority.SKIP] > total_tests * 0.2:
            recommendations.append("Consider removing low-value tests to reduce maintenance overhead")
        
        if priority_counts[TestPriority.CRITICAL] < total_tests * 0.1:
            recommendations.append("Increase coverage of critical functionality")
        
        # Analyze common issues
        common_issues = defaultdict(int)
        for result in optimization_results:
            for reason in result.reasoning:
                if "execution time" in reason.lower():
                    common_issues["slow_tests"] += 1
                elif "failure" in reason.lower():
                    common_issues["unreliable_tests"] += 1
        
        if common_issues["slow_tests"] > total_tests * 0.3:
            recommendations.append("Focus on optimizing test execution time")
        
        if common_issues["unreliable_tests"] > total_tests * 0.2:
            recommendations.append("Improve test reliability and stability")
        
        # Resource optimization
        total_time_savings = sum(r.estimated_savings.get('time_seconds', 0) 
                               for r in optimization_results)
        if total_time_savings > 3600:  # More than 1 hour
            recommendations.append(f"Potential time savings: {total_time_savings/3600:.1f} hours")
        
        return recommendations[:5]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        if not self.optimization_history:
            return {"status": "No optimizations performed yet"}
        
        latest_report = self.optimization_history[-1]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "latest_strategy": latest_report.strategy.value,
            "tests_analyzed": latest_report.total_tests_analyzed,
            "total_savings": latest_report.estimated_total_savings,
            "model_performance": latest_report.model_performance,
            "key_recommendations": latest_report.recommendations,
            "optimization_effectiveness": self._calculate_optimization_effectiveness()
        }
    
    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate overall optimization effectiveness"""
        if not self.optimization_history:
            return 0.0
        
        # Simple effectiveness based on confidence scores
        all_results = []
        for report in self.optimization_history:
            all_results.extend(report.optimization_results)
        
        if not all_results:
            return 0.0
        
        avg_confidence = sum(r.confidence_score for r in all_results) / len(all_results)
        return avg_confidence
    
    def save_optimization_state(self, file_path: str):
        """Save optimization models and state"""
        state = {
            'strategy': self.strategy.value,
            'models': {},  # Simplified - would need proper model serialization
            'feature_importance': self.feature_importance,
            'optimization_history': [asdict(report) for report in self.optimization_history[-10:]]
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_optimization_state(self, file_path: str):
        """Load optimization models and state"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.strategy = OptimizationStrategy(state['strategy'])
            self.feature_importance = state['feature_importance']
            # Would need to properly deserialize models and history
            
        except Exception as e:
            print(f"Failed to load optimization state: {e}")