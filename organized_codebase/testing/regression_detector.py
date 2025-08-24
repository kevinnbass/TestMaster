"""
Automatic Regression Detection Framework for TestMaster
Identifies regressions through historical analysis and ML prediction
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import statistics


class RegressionType(Enum):
    """Types of regressions"""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    RELIABILITY = "reliability"
    SECURITY = "security"


@dataclass
class TestResult:
    """Single test execution result"""
    test_id: str
    timestamp: str
    passed: bool
    execution_time: float
    memory_usage: Optional[float] = None
    output_hash: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None


@dataclass
class Regression:
    """Detected regression"""
    type: RegressionType
    test_id: str
    baseline_value: Any
    current_value: Any
    confidence: float  # 0.0 to 1.0
    severity: str  # critical, high, medium, low
    first_detected: str
    description: str


@dataclass
class RegressionReport:
    """Regression analysis report"""
    total_tests: int
    regressions_found: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    regression_rate: float
    recommendations: List[str]


class BaselineManager:
    """Manages test baselines for comparison"""
    
    def __init__(self):
        self.baselines = {}
        self.history = []
        
    def establish_baseline(self, results: List[TestResult]) -> Dict[str, Any]:
        """Establish baseline from test results"""
        baseline = {}
        
        for result in results:
            baseline[result.test_id] = {
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'output_hash': result.output_hash,
                'passed': result.passed,
                'metrics': result.metrics or {}
            }
            
        self.baselines = baseline
        return baseline
    
    def update_baseline(self, test_id: str, result: TestResult):
        """Update baseline for specific test"""
        if test_id not in self.baselines:
            self.baselines[test_id] = {}
            
        self.baselines[test_id].update({
            'execution_time': result.execution_time,
            'memory_usage': result.memory_usage,
            'output_hash': result.output_hash,
            'passed': result.passed,
            'metrics': result.metrics or {}
        })
    
    def get_baseline(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get baseline for test"""
        return self.baselines.get(test_id)
    
    def calculate_statistics(self, test_id: str, metric: str) -> Dict[str, float]:
        """Calculate statistical measures for metric"""
        history = [r for r in self.history if r.test_id == test_id]
        
        if not history:
            return {}
            
        values = []
        for result in history:
            if metric == 'execution_time':
                values.append(result.execution_time)
            elif metric == 'memory_usage' and result.memory_usage:
                values.append(result.memory_usage)
            elif result.metrics and metric in result.metrics:
                values.append(result.metrics[metric])
                
        if not values:
            return {}
            
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values)
        }


class RegressionDetector:
    """Main regression detection engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.baseline_manager = BaselineManager()
        self.regressions = []
        self.thresholds = self._initialize_thresholds()
        
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize detection thresholds"""
        return {
            'performance_threshold': self.config.get('performance_threshold', 1.2),  # 20% slower
            'memory_threshold': self.config.get('memory_threshold', 1.3),  # 30% more memory
            'accuracy_threshold': self.config.get('accuracy_threshold', 0.95),  # 5% accuracy loss
            'reliability_threshold': self.config.get('reliability_threshold', 0.98),  # 2% more failures
        }
    
    def detect_regressions(self, current_results: List[TestResult]) -> List[Regression]:
        """Detect regressions in test results"""
        regressions = []
        
        for result in current_results:
            baseline = self.baseline_manager.get_baseline(result.test_id)
            
            if not baseline:
                # No baseline, can't detect regression
                continue
                
            # Check different types of regressions
            regressions.extend(self._check_functional_regression(result, baseline))
            regressions.extend(self._check_performance_regression(result, baseline))
            regressions.extend(self._check_memory_regression(result, baseline))
            regressions.extend(self._check_accuracy_regression(result, baseline))
            regressions.extend(self._check_reliability_regression(result, baseline))
            
        self.regressions.extend(regressions)
        return regressions
    
    def _check_functional_regression(self, result: TestResult, 
                                    baseline: Dict) -> List[Regression]:
        """Check for functional regression"""
        regressions = []
        
        # Test failure when it used to pass
        if baseline['passed'] and not result.passed:
            regressions.append(Regression(
                type=RegressionType.FUNCTIONAL,
                test_id=result.test_id,
                baseline_value="passed",
                current_value="failed",
                confidence=1.0,
                severity="critical",
                first_detected=result.timestamp,
                description=f"Test started failing: {result.error_message}"
            ))
            
        # Output changed
        if baseline['output_hash'] and result.output_hash:
            if baseline['output_hash'] != result.output_hash:
                regressions.append(Regression(
                    type=RegressionType.FUNCTIONAL,
                    test_id=result.test_id,
                    baseline_value=baseline['output_hash'],
                    current_value=result.output_hash,
                    confidence=0.9,
                    severity="high",
                    first_detected=result.timestamp,
                    description="Test output changed"
                ))
                
        return regressions
    
    def _check_performance_regression(self, result: TestResult,
                                     baseline: Dict) -> List[Regression]:
        """Check for performance regression"""
        regressions = []
        
        baseline_time = baseline.get('execution_time')
        if baseline_time and result.execution_time:
            ratio = result.execution_time / baseline_time
            
            if ratio > self.thresholds['performance_threshold']:
                # Calculate confidence based on magnitude
                confidence = min((ratio - 1.0) / 0.5, 1.0)
                
                # Determine severity
                if ratio > 2.0:
                    severity = "critical"
                elif ratio > 1.5:
                    severity = "high"
                elif ratio > 1.3:
                    severity = "medium"
                else:
                    severity = "low"
                    
                regressions.append(Regression(
                    type=RegressionType.PERFORMANCE,
                    test_id=result.test_id,
                    baseline_value=baseline_time,
                    current_value=result.execution_time,
                    confidence=confidence,
                    severity=severity,
                    first_detected=result.timestamp,
                    description=f"Performance degraded by {(ratio-1)*100:.1f}%"
                ))
                
        return regressions
    
    def _check_memory_regression(self, result: TestResult,
                                baseline: Dict) -> List[Regression]:
        """Check for memory regression"""
        regressions = []
        
        baseline_memory = baseline.get('memory_usage')
        if baseline_memory and result.memory_usage:
            ratio = result.memory_usage / baseline_memory
            
            if ratio > self.thresholds['memory_threshold']:
                confidence = min((ratio - 1.0) / 0.5, 1.0)
                
                if ratio > 2.0:
                    severity = "critical"
                elif ratio > 1.5:
                    severity = "high"
                else:
                    severity = "medium"
                    
                regressions.append(Regression(
                    type=RegressionType.MEMORY,
                    test_id=result.test_id,
                    baseline_value=baseline_memory,
                    current_value=result.memory_usage,
                    confidence=confidence,
                    severity=severity,
                    first_detected=result.timestamp,
                    description=f"Memory usage increased by {(ratio-1)*100:.1f}%"
                ))
                
        return regressions
    
    def _check_accuracy_regression(self, result: TestResult,
                                  baseline: Dict) -> List[Regression]:
        """Check for accuracy regression"""
        regressions = []
        
        if result.metrics and baseline.get('metrics'):
            for metric_name, current_value in result.metrics.items():
                if 'accuracy' in metric_name.lower() or 'score' in metric_name.lower():
                    baseline_value = baseline['metrics'].get(metric_name)
                    
                    if baseline_value and current_value < baseline_value:
                        ratio = current_value / baseline_value
                        
                        if ratio < self.thresholds['accuracy_threshold']:
                            confidence = min((1.0 - ratio) / 0.1, 1.0)
                            
                            regressions.append(Regression(
                                type=RegressionType.ACCURACY,
                                test_id=result.test_id,
                                baseline_value=baseline_value,
                                current_value=current_value,
                                confidence=confidence,
                                severity="high" if ratio < 0.9 else "medium",
                                first_detected=result.timestamp,
                                description=f"{metric_name} dropped by {(1-ratio)*100:.1f}%"
                            ))
                            
        return regressions
    
    def _check_reliability_regression(self, result: TestResult,
                                     baseline: Dict) -> List[Regression]:
        """Check for reliability regression using historical data"""
        regressions = []
        
        # Calculate failure rate from history
        stats = self.baseline_manager.calculate_statistics(result.test_id, 'passed')
        
        if stats and 'mean' in stats:
            historical_pass_rate = stats['mean']
            
            # Need multiple failures to detect reliability issue
            recent_results = [r for r in self.baseline_manager.history[-10:]
                            if r.test_id == result.test_id]
            
            if len(recent_results) >= 5:
                recent_pass_rate = sum(1 for r in recent_results if r.passed) / len(recent_results)
                
                if recent_pass_rate < historical_pass_rate * self.thresholds['reliability_threshold']:
                    regressions.append(Regression(
                        type=RegressionType.RELIABILITY,
                        test_id=result.test_id,
                        baseline_value=f"{historical_pass_rate*100:.1f}% pass rate",
                        current_value=f"{recent_pass_rate*100:.1f}% pass rate",
                        confidence=0.8,
                        severity="high",
                        first_detected=result.timestamp,
                        description="Test becoming unreliable"
                    ))
                    
        return regressions
    
    def predict_regression(self, test_id: str, metrics: Dict[str, float]) -> float:
        """Predict regression probability using ML"""
        # Simplified prediction based on trend analysis
        stats = self.baseline_manager.calculate_statistics(test_id, 'execution_time')
        
        if not stats or 'mean' not in stats:
            return 0.0
            
        # Check if trending upward
        recent_history = self.baseline_manager.history[-10:]
        if len(recent_history) < 3:
            return 0.0
            
        times = [r.execution_time for r in recent_history if r.test_id == test_id]
        if len(times) < 3:
            return 0.0
            
        # Calculate trend
        trend = (times[-1] - times[0]) / times[0] if times[0] > 0 else 0
        
        # Probability based on trend
        if trend > 0.3:
            return min(trend, 1.0)
        return 0.0
    
    def generate_report(self) -> RegressionReport:
        """Generate regression analysis report"""
        if not self.regressions:
            return RegressionReport(
                total_tests=0,
                regressions_found=0,
                by_type={},
                by_severity={},
                regression_rate=0.0,
                recommendations=["No regressions detected"]
            )
            
        by_type = {}
        by_severity = {}
        
        for regression in self.regressions:
            # Count by type
            type_key = regression.type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # Count by severity
            by_severity[regression.severity] = by_severity.get(regression.severity, 0) + 1
            
        unique_tests = len(set(r.test_id for r in self.regressions))
        
        return RegressionReport(
            total_tests=unique_tests,
            regressions_found=len(self.regressions),
            by_type=by_type,
            by_severity=by_severity,
            regression_rate=len(self.regressions) / max(unique_tests, 1),
            recommendations=self._generate_recommendations()
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on regressions"""
        recommendations = []
        
        # Check severity distribution
        critical_count = sum(1 for r in self.regressions if r.severity == "critical")
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical regressions immediately")
            
        # Check regression types
        perf_regressions = [r for r in self.regressions if r.type == RegressionType.PERFORMANCE]
        if perf_regressions:
            recommendations.append("Profile code to identify performance bottlenecks")
            
        memory_regressions = [r for r in self.regressions if r.type == RegressionType.MEMORY]
        if memory_regressions:
            recommendations.append("Check for memory leaks and optimize memory usage")
            
        functional_regressions = [r for r in self.regressions if r.type == RegressionType.FUNCTIONAL]
        if functional_regressions:
            recommendations.append("Review recent code changes for breaking changes")
            
        reliability_regressions = [r for r in self.regressions if r.type == RegressionType.RELIABILITY]
        if reliability_regressions:
            recommendations.append("Investigate flaky tests and race conditions")
            
        return recommendations[:5]