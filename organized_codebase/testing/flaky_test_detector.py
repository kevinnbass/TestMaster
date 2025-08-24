"""
Flaky Test Detection Framework for TestMaster
Identifies and analyzes unreliable tests using statistical methods
"""

import statistics
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import math


class FlakinessType(Enum):
    """Types of test flakiness"""
    RANDOM = "random"  # Random failures
    TIMING = "timing"  # Time-dependent failures
    ORDER = "order"  # Order-dependent failures
    RESOURCE = "resource"  # Resource contention
    ENVIRONMENT = "environment"  # Environment-specific
    CONCURRENCY = "concurrency"  # Race conditions
    NETWORK = "network"  # Network-related


@dataclass
class TestRun:
    """Single test execution"""
    test_id: str
    run_id: str
    passed: bool
    execution_time: float
    timestamp: str
    environment: Dict[str, str]
    error_type: Optional[str] = None
    retry_count: int = 0


@dataclass
class FlakinessAnalysis:
    """Flakiness analysis for a test"""
    test_id: str
    flakiness_score: float  # 0.0 (stable) to 1.0 (very flaky)
    failure_rate: float
    confidence: float
    flakiness_type: FlakinessType
    pattern: str
    contributing_factors: List[str]
    recommended_fixes: List[str]


@dataclass
class FlakinessReport:
    """Overall flakiness report"""
    total_tests: int
    flaky_tests: int
    flakiness_rate: float
    by_type: Dict[str, int]
    top_flaky_tests: List[Tuple[str, float]]
    estimated_impact: Dict[str, Any]
    recommendations: List[str]


class StatisticalAnalyzer:
    """Statistical analysis for flakiness detection"""
    
    def calculate_failure_rate(self, runs: List[TestRun]) -> float:
        """Calculate failure rate"""
        if not runs:
            return 0.0
        failures = sum(1 for r in runs if not r.passed)
        return failures / len(runs)
    
    def calculate_consistency(self, runs: List[TestRun]) -> float:
        """Calculate result consistency (0=inconsistent, 1=consistent)"""
        if len(runs) < 2:
            return 1.0
            
        # Check for alternating patterns
        transitions = 0
        for i in range(1, len(runs)):
            if runs[i].passed != runs[i-1].passed:
                transitions += 1
                
        max_transitions = len(runs) - 1
        consistency = 1.0 - (transitions / max_transitions)
        return consistency
    
    def detect_pattern(self, runs: List[TestRun]) -> str:
        """Detect failure pattern"""
        if len(runs) < 5:
            return "insufficient_data"
            
        # Convert to binary sequence (1=pass, 0=fail)
        sequence = [1 if r.passed else 0 for r in runs]
        
        # Check for patterns
        if self._is_alternating(sequence):
            return "alternating"
        elif self._is_periodic(sequence):
            return "periodic"
        elif self._is_clustered(sequence):
            return "clustered"
        elif self._is_random(sequence):
            return "random"
        else:
            return "irregular"
    
    def _is_alternating(self, sequence: List[int]) -> bool:
        """Check for alternating pattern"""
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                return False
        return True
    
    def _is_periodic(self, sequence: List[int], max_period: int = 10) -> bool:
        """Check for periodic pattern"""
        for period in range(2, min(max_period, len(sequence)//2)):
            is_periodic = True
            for i in range(period, len(sequence)):
                if sequence[i] != sequence[i % period]:
                    is_periodic = False
                    break
            if is_periodic:
                return True
        return False
    
    def _is_clustered(self, sequence: List[int]) -> bool:
        """Check for clustered failures"""
        clusters = []
        current_cluster = []
        
        for i, val in enumerate(sequence):
            if val == 0:  # Failure
                current_cluster.append(i)
            elif current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
                
        if current_cluster:
            clusters.append(current_cluster)
            
        # Check if failures are clustered
        if len(clusters) < 2:
            return False
            
        # Calculate average gap between clusters
        gaps = []
        for i in range(1, len(clusters)):
            gap = clusters[i][0] - clusters[i-1][-1]
            gaps.append(gap)
            
        avg_gap = statistics.mean(gaps)
        return avg_gap > len(sequence) / (len(clusters) * 2)
    
    def _is_random(self, sequence: List[int]) -> bool:
        """Check for random pattern using runs test"""
        if len(sequence) < 10:
            return False
            
        # Count runs (sequences of same value)
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
                
        # Expected runs for random sequence
        n1 = sum(sequence)
        n2 = len(sequence) - n1
        
        if n1 == 0 or n2 == 0:
            return False
            
        expected_runs = (2 * n1 * n2) / len(sequence) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - len(sequence))) / (
            len(sequence)**2 * (len(sequence) - 1))
            
        if variance <= 0:
            return False
            
        # Z-score for runs test
        z_score = abs(runs - expected_runs) / math.sqrt(variance)
        
        # Random if z-score < 1.96 (95% confidence)
        return z_score < 1.96


class FlakyTestDetector:
    """Main flaky test detection engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analyzer = StatisticalAnalyzer()
        self.test_history = defaultdict(list)
        self.flakiness_threshold = self.config.get('flakiness_threshold', 0.3)
        self.min_runs = self.config.get('min_runs', 5)
        
    def add_test_run(self, run: TestRun):
        """Add test run to history"""
        self.test_history[run.test_id].append(run)
    
    def analyze_test(self, test_id: str) -> Optional[FlakinessAnalysis]:
        """Analyze test for flakiness"""
        runs = self.test_history.get(test_id, [])
        
        if len(runs) < self.min_runs:
            return None
            
        # Calculate metrics
        failure_rate = self.analyzer.calculate_failure_rate(runs)
        consistency = self.analyzer.calculate_consistency(runs)
        pattern = self.analyzer.detect_pattern(runs)
        
        # Skip if test always passes or always fails
        if failure_rate == 0.0 or failure_rate == 1.0:
            return None
            
        # Calculate flakiness score
        flakiness_score = self._calculate_flakiness_score(
            failure_rate, consistency, pattern
        )
        
        if flakiness_score < self.flakiness_threshold:
            return None
            
        # Determine flakiness type
        flakiness_type = self._determine_flakiness_type(runs, pattern)
        
        # Identify contributing factors
        factors = self._identify_contributing_factors(runs, flakiness_type)
        
        # Generate recommendations
        fixes = self._generate_fix_recommendations(flakiness_type, factors)
        
        # Calculate confidence
        confidence = min(len(runs) / 20, 1.0)  # More runs = higher confidence
        
        return FlakinessAnalysis(
            test_id=test_id,
            flakiness_score=flakiness_score,
            failure_rate=failure_rate,
            confidence=confidence,
            flakiness_type=flakiness_type,
            pattern=pattern,
            contributing_factors=factors,
            recommended_fixes=fixes
        )
    
    def _calculate_flakiness_score(self, failure_rate: float, 
                                  consistency: float, pattern: str) -> float:
        """Calculate overall flakiness score"""
        # Flaky tests have moderate failure rate and low consistency
        optimal_failure_rate = 0.3  # Most flaky around 30% failure
        
        # Distance from optimal failure rate
        failure_score = 1.0 - abs(failure_rate - optimal_failure_rate) / 0.7
        
        # Inconsistency score
        inconsistency_score = 1.0 - consistency
        
        # Pattern score
        pattern_scores = {
            'random': 0.9,
            'alternating': 0.8,
            'periodic': 0.7,
            'clustered': 0.6,
            'irregular': 0.5,
            'insufficient_data': 0.0
        }
        pattern_score = pattern_scores.get(pattern, 0.5)
        
        # Weighted average
        flakiness = (
            failure_score * 0.3 +
            inconsistency_score * 0.4 +
            pattern_score * 0.3
        )
        
        return round(flakiness, 3)
    
    def _determine_flakiness_type(self, runs: List[TestRun], 
                                 pattern: str) -> FlakinessType:
        """Determine type of flakiness"""
        
        # Check timing variations
        times = [r.execution_time for r in runs]
        if times:
            time_variance = statistics.stdev(times) / statistics.mean(times) if len(times) > 1 else 0
            if time_variance > 0.5:
                return FlakinessType.TIMING
        
        # Check for order dependencies
        if pattern == 'alternating' or pattern == 'periodic':
            return FlakinessType.ORDER
        
        # Check error types for patterns
        error_types = [r.error_type for r in runs if r.error_type]
        if error_types:
            if any('timeout' in e.lower() for e in error_types if e):
                return FlakinessType.TIMING
            elif any('connection' in e.lower() or 'network' in e.lower() for e in error_types if e):
                return FlakinessType.NETWORK
            elif any('lock' in e.lower() or 'concurrent' in e.lower() for e in error_types if e):
                return FlakinessType.CONCURRENCY
            elif any('resource' in e.lower() or 'memory' in e.lower() for e in error_types if e):
                return FlakinessType.RESOURCE
        
        # Check environment variations
        envs = [r.environment for r in runs if r.environment]
        if envs and len(set(str(e) for e in envs)) > 1:
            # Different results in different environments
            env_failures = defaultdict(list)
            for run in runs:
                if run.environment:
                    env_key = str(run.environment)
                    env_failures[env_key].append(run.passed)
            
            # Check if failure rate varies by environment
            failure_rates = []
            for env, results in env_failures.items():
                rate = sum(1 for r in results if not r) / len(results)
                failure_rates.append(rate)
            
            if failure_rates and max(failure_rates) - min(failure_rates) > 0.3:
                return FlakinessType.ENVIRONMENT
        
        return FlakinessType.RANDOM
    
    def _identify_contributing_factors(self, runs: List[TestRun],
                                      flakiness_type: FlakinessType) -> List[str]:
        """Identify factors contributing to flakiness"""
        factors = []
        
        # Timing analysis
        times = [r.execution_time for r in runs]
        if times and len(times) > 1:
            time_cv = statistics.stdev(times) / statistics.mean(times)
            if time_cv > 0.3:
                factors.append(f"High execution time variance (CV={time_cv:.2f})")
        
        # Retry analysis
        retries = [r.retry_count for r in runs]
        if any(r > 0 for r in retries):
            avg_retries = statistics.mean(retries)
            factors.append(f"Requires retries (avg={avg_retries:.1f})")
        
        # Error pattern analysis
        error_types = [r.error_type for r in runs if r.error_type]
        if error_types:
            unique_errors = set(error_types)
            if len(unique_errors) > 1:
                factors.append(f"Multiple error types ({len(unique_errors)} different)")
        
        # Type-specific factors
        if flakiness_type == FlakinessType.TIMING:
            factors.append("Timing-sensitive operations detected")
        elif flakiness_type == FlakinessType.ORDER:
            factors.append("Test order dependency suspected")
        elif flakiness_type == FlakinessType.CONCURRENCY:
            factors.append("Potential race condition")
        elif flakiness_type == FlakinessType.NETWORK:
            factors.append("Network-dependent operations")
        elif flakiness_type == FlakinessType.RESOURCE:
            factors.append("Resource contention issues")
        elif flakiness_type == FlakinessType.ENVIRONMENT:
            factors.append("Environment-specific behavior")
            
        return factors
    
    def _generate_fix_recommendations(self, flakiness_type: FlakinessType,
                                     factors: List[str]) -> List[str]:
        """Generate recommendations to fix flakiness"""
        fixes = []
        
        if flakiness_type == FlakinessType.TIMING:
            fixes.extend([
                "Add explicit waits instead of sleep",
                "Use proper synchronization mechanisms",
                "Increase timeout values",
                "Mock time-dependent operations"
            ])
        elif flakiness_type == FlakinessType.ORDER:
            fixes.extend([
                "Ensure proper test isolation",
                "Reset shared state in setup/teardown",
                "Avoid test interdependencies",
                "Use fresh test data for each run"
            ])
        elif flakiness_type == FlakinessType.CONCURRENCY:
            fixes.extend([
                "Add proper locking mechanisms",
                "Use thread-safe operations",
                "Serialize concurrent operations in tests",
                "Mock concurrent dependencies"
            ])
        elif flakiness_type == FlakinessType.NETWORK:
            fixes.extend([
                "Mock network calls",
                "Add retry logic with backoff",
                "Use local test servers",
                "Implement proper error handling"
            ])
        elif flakiness_type == FlakinessType.RESOURCE:
            fixes.extend([
                "Ensure resource cleanup in teardown",
                "Increase resource limits for tests",
                "Use resource pooling",
                "Mock resource-intensive operations"
            ])
        elif flakiness_type == FlakinessType.ENVIRONMENT:
            fixes.extend([
                "Standardize test environment",
                "Use containerization for consistency",
                "Mock environment-specific dependencies",
                "Add environment validation"
            ])
        else:  # RANDOM
            fixes.extend([
                "Add logging to identify failure causes",
                "Increase test data variation",
                "Check for hidden dependencies",
                "Review test assertions"
            ])
            
        return fixes[:5]  # Return top 5 recommendations
    
    def generate_report(self) -> FlakinessReport:
        """Generate flakiness report for all tests"""
        flaky_tests = []
        by_type = defaultdict(int)
        
        for test_id in self.test_history:
            analysis = self.analyze_test(test_id)
            if analysis:
                flaky_tests.append((test_id, analysis.flakiness_score))
                by_type[analysis.flakiness_type.value] += 1
        
        # Sort by flakiness score
        flaky_tests.sort(key=lambda x: x[1], reverse=True)
        
        total_tests = len(self.test_history)
        flaky_count = len(flaky_tests)
        
        return FlakinessReport(
            total_tests=total_tests,
            flaky_tests=flaky_count,
            flakiness_rate=flaky_count / total_tests if total_tests > 0 else 0,
            by_type=dict(by_type),
            top_flaky_tests=flaky_tests[:10],
            estimated_impact=self._estimate_impact(flaky_tests),
            recommendations=self._generate_overall_recommendations(by_type)
        )
    
    def _estimate_impact(self, flaky_tests: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Estimate impact of flaky tests"""
        if not flaky_tests:
            return {"wasted_time": 0, "false_positives": 0}
            
        # Estimate wasted time (assumes reruns and investigation)
        avg_flakiness = statistics.mean(score for _, score in flaky_tests)
        wasted_hours = len(flaky_tests) * avg_flakiness * 2  # 2 hours per flaky test
        
        # Estimate false positives
        false_positive_rate = avg_flakiness * 0.3  # 30% of flakiness leads to false positives
        
        return {
            "wasted_time_hours": round(wasted_hours, 1),
            "false_positive_rate": round(false_positive_rate, 3),
            "confidence_impact": "low" if avg_flakiness > 0.5 else "medium"
        }
    
    def _generate_overall_recommendations(self, by_type: Dict[str, int]) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        if not by_type:
            return ["No flaky tests detected"]
            
        # Find most common type
        most_common = max(by_type.items(), key=lambda x: x[1])
        
        recommendations.append(f"Focus on fixing {most_common[0]} flakiness ({most_common[1]} tests)")
        recommendations.append("Implement test retry mechanism with analysis")
        recommendations.append("Add flakiness detection to CI pipeline")
        
        if by_type.get('timing', 0) > 2:
            recommendations.append("Review and standardize timeout values")
        if by_type.get('order', 0) > 2:
            recommendations.append("Randomize test execution order")
        if by_type.get('resource', 0) > 2:
            recommendations.append("Implement resource isolation")
            
        return recommendations[:5]