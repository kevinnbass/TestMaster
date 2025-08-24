"""
Adaptive Testing Framework
Self-improving test patterns that learn from execution results and feedback.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import statistics
import math
import random


class AdaptationStrategy(Enum):
    """Strategies for adapting test patterns"""
    PERFORMANCE_DRIVEN = "performance_driven"
    FAILURE_DRIVEN = "failure_driven"
    COVERAGE_DRIVEN = "coverage_driven"
    QUALITY_DRIVEN = "quality_driven"
    HYBRID = "hybrid"


class TestOutcome(Enum):
    """Possible test outcomes"""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"
    TIMEOUT = "timeout"


@dataclass
class TestExecution:
    """Record of a test execution"""
    test_id: str
    execution_time: datetime
    outcome: TestOutcome
    execution_duration: float  # seconds
    error_message: Optional[str] = None
    coverage_data: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    memory_usage: float = 0.0
    
    @property
    def success_score(self) -> float:
        """Calculate success score (0.0 = worst, 1.0 = best)"""
        base_scores = {
            TestOutcome.PASS: 1.0,
            TestOutcome.SKIP: 0.7,
            TestOutcome.FAIL: 0.3,
            TestOutcome.ERROR: 0.1,
            TestOutcome.TIMEOUT: 0.0
        }
        
        score = base_scores[self.outcome]
        
        # Adjust for performance
        if self.execution_duration > 10.0:  # Slow test penalty
            score *= 0.9
        elif self.execution_duration < 0.1:  # Fast test bonus
            score *= 1.1
        
        return min(1.0, score)


@dataclass
class TestPattern:
    """Adaptive test pattern that learns and evolves"""
    pattern_id: str
    name: str
    template_code: str
    category: str
    complexity_level: int
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    avg_coverage_improvement: float = 0.0
    adaptation_count: int = 0
    last_adapted: Optional[datetime] = None
    execution_history: List[TestExecution] = field(default_factory=list)
    adaptation_rules: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.5
    
    def add_execution(self, execution: TestExecution) -> None:
        """Add execution result and update metrics"""
        self.execution_history.append(execution)
        
        # Keep only recent executions (last 100)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update pattern metrics based on execution history"""
        if not self.execution_history:
            return
        
        # Calculate success rate
        successes = sum(1 for exec in self.execution_history if exec.outcome == TestOutcome.PASS)
        self.success_rate = successes / len(self.execution_history)
        
        # Calculate average execution time
        times = [exec.execution_duration for exec in self.execution_history]
        self.avg_execution_time = statistics.mean(times)
        
        # Calculate coverage improvement
        if self.execution_history:
            coverage_scores = [
                sum(exec.coverage_data.values()) / len(exec.coverage_data.values()) 
                if exec.coverage_data else 0.0
                for exec in self.execution_history
            ]
            self.avg_coverage_improvement = statistics.mean(coverage_scores) if coverage_scores else 0.0
        
        # Update confidence based on execution count and stability
        execution_count = len(self.execution_history)
        stability = 1.0 - statistics.stdev([exec.success_score for exec in self.execution_history[-20:]]) if len(self.execution_history) >= 20 else 0.5
        self.confidence_score = min(1.0, (execution_count / 50.0) * stability)
    
    @property
    def adaptation_score(self) -> float:
        """Calculate how much this pattern needs adaptation"""
        # Higher score = needs more adaptation
        score = 0.0
        
        # Poor success rate increases adaptation need
        if self.success_rate < 0.7:
            score += (0.7 - self.success_rate) * 2.0
        
        # Slow execution increases adaptation need
        if self.avg_execution_time > 5.0:
            score += min(1.0, self.avg_execution_time / 10.0)
        
        # Low coverage increases adaptation need
        if self.avg_coverage_improvement < 0.5:
            score += (0.5 - self.avg_coverage_improvement)
        
        # Recent failures increase adaptation need
        if len(self.execution_history) >= 5:
            recent_failures = sum(1 for exec in self.execution_history[-5:] if exec.outcome != TestOutcome.PASS)
            score += recent_failures * 0.3
        
        return min(2.0, score)  # Cap at 2.0


@dataclass
class AdaptationRule:
    """Rule for adapting test patterns"""
    rule_id: str
    name: str
    condition: str  # Python expression
    action: str     # Adaptation action
    priority: int
    success_count: int = 0
    application_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of this rule"""
        if self.application_count == 0:
            return 0.0
        return self.success_count / self.application_count
    
    @property
    def confidence(self) -> float:
        """Calculate confidence in this rule"""
        if self.application_count < 5:
            return 0.3
        return min(1.0, self.success_rate * (self.application_count / 20.0))


class PatternEvolutionEngine:
    """Engine for evolving test patterns based on feedback"""
    
    def __init__(self):
        self.evolution_strategies = {
            'timeout_optimization': self._optimize_timeout,
            'assertion_enhancement': self._enhance_assertions,
            'setup_refinement': self._refine_setup,
            'error_handling_improvement': self._improve_error_handling,
            'performance_tuning': self._tune_performance,
            'parametrization_expansion': self._expand_parametrization
        }
    
    def evolve_pattern(self, pattern: TestPattern, strategy: str = 'hybrid') -> Optional[TestPattern]:
        """Evolve a test pattern based on its execution history"""
        if not pattern.execution_history or len(pattern.execution_history) < 3:
            return None  # Need sufficient history
        
        # Analyze execution history to determine best evolution strategy
        issues = self._analyze_pattern_issues(pattern)
        
        if not issues:
            return None  # No issues to fix
        
        # Apply most relevant evolution strategy
        primary_issue = max(issues.items(), key=lambda x: x[1])[0]
        
        if primary_issue in self.evolution_strategies:
            evolved_pattern = self.evolution_strategies[primary_issue](pattern)
            if evolved_pattern:
                evolved_pattern.adaptation_count += 1
                evolved_pattern.last_adapted = datetime.now()
                return evolved_pattern
        
        return None
    
    def _analyze_pattern_issues(self, pattern: TestPattern) -> Dict[str, float]:
        """Analyze pattern execution history to identify issues"""
        issues = {}
        
        # Analyze timeouts
        timeout_count = sum(1 for exec in pattern.execution_history if exec.outcome == TestOutcome.TIMEOUT)
        if timeout_count > 0:
            issues['timeout_optimization'] = timeout_count / len(pattern.execution_history)
        
        # Analyze failures
        failure_count = sum(1 for exec in pattern.execution_history if exec.outcome == TestOutcome.FAIL)
        if failure_count > len(pattern.execution_history) * 0.2:  # >20% failure rate
            issues['assertion_enhancement'] = failure_count / len(pattern.execution_history)
        
        # Analyze errors
        error_count = sum(1 for exec in pattern.execution_history if exec.outcome == TestOutcome.ERROR)
        if error_count > 0:
            issues['error_handling_improvement'] = error_count / len(pattern.execution_history)
        
        # Analyze performance
        slow_executions = sum(1 for exec in pattern.execution_history if exec.execution_duration > 10.0)
        if slow_executions > 0:
            issues['performance_tuning'] = slow_executions / len(pattern.execution_history)
        
        # Analyze setup issues (errors in first 10% of execution time)
        setup_errors = sum(1 for exec in pattern.execution_history 
                          if exec.outcome == TestOutcome.ERROR and exec.execution_duration < 1.0)
        if setup_errors > 0:
            issues['setup_refinement'] = setup_errors / len(pattern.execution_history)
        
        return issues
    
    def _optimize_timeout(self, pattern: TestPattern) -> Optional[TestPattern]:
        """Optimize timeout settings based on execution history"""
        execution_times = [exec.execution_duration for exec in pattern.execution_history 
                          if exec.outcome != TestOutcome.TIMEOUT]
        
        if not execution_times:
            return None
        
        # Set timeout to 95th percentile + buffer
        p95_time = sorted(execution_times)[int(len(execution_times) * 0.95)]
        new_timeout = p95_time * 2.0  # 100% buffer
        
        # Modify template code
        new_code = pattern.template_code
        if 'timeout=' in new_code:
            # Replace existing timeout
            import re
            new_code = re.sub(r'timeout=\d+\.?\d*', f'timeout={new_timeout:.1f}', new_code)
        else:
            # Add timeout decorator
            new_code = f'@pytest.mark.timeout({new_timeout:.1f})\n{new_code}'
        
        # Create evolved pattern
        evolved = TestPattern(
            pattern_id=f"{pattern.pattern_id}_timeout_optimized",
            name=f"{pattern.name} (Timeout Optimized)",
            template_code=new_code,
            category=pattern.category,
            complexity_level=pattern.complexity_level,
            adaptation_count=pattern.adaptation_count
        )
        
        return evolved
    
    def _enhance_assertions(self, pattern: TestPattern) -> Optional[TestPattern]:
        """Enhance assertions based on failure patterns"""
        # Analyze common failure patterns
        failures = [exec for exec in pattern.execution_history if exec.outcome == TestOutcome.FAIL]
        
        if not failures:
            return None
        
        # Add more robust assertions
        enhanced_code = pattern.template_code
        
        # Add type checking
        if 'assert' in enhanced_code and 'isinstance' not in enhanced_code:
            enhanced_code = enhanced_code.replace(
                'assert result',
                'assert result is not None\n    assert isinstance(result, (str, int, float, bool, list, dict))'
            )
        
        # Add boundary checks
        if 'assert' in enhanced_code and 'len(' not in enhanced_code:
            enhanced_code = enhanced_code.replace(
                'assert result',
                'assert result is not None\n    if hasattr(result, "__len__"):\n        assert len(result) >= 0'
            )
        
        # Create evolved pattern
        evolved = TestPattern(
            pattern_id=f"{pattern.pattern_id}_assertions_enhanced",
            name=f"{pattern.name} (Enhanced Assertions)",
            template_code=enhanced_code,
            category=pattern.category,
            complexity_level=pattern.complexity_level + 1,
            adaptation_count=pattern.adaptation_count
        )
        
        return evolved
    
    def _refine_setup(self, pattern: TestPattern) -> Optional[TestPattern]:
        """Refine test setup based on early errors"""
        # Add more robust setup
        enhanced_code = pattern.template_code
        
        # Add setup validation
        if 'def test_' in enhanced_code:
            setup_validation = '''
    # Adaptive setup validation
    import sys
    import os
    
    # Ensure test environment is ready
    assert sys.version_info >= (3, 6), "Python 3.6+ required"
    
    '''
            enhanced_code = enhanced_code.replace('def test_', f'{setup_validation}def test_')
        
        # Create evolved pattern
        evolved = TestPattern(
            pattern_id=f"{pattern.pattern_id}_setup_refined",
            name=f"{pattern.name} (Refined Setup)",
            template_code=enhanced_code,
            category=pattern.category,
            complexity_level=pattern.complexity_level + 1,
            adaptation_count=pattern.adaptation_count
        )
        
        return evolved
    
    def _improve_error_handling(self, pattern: TestPattern) -> Optional[TestPattern]:
        """Improve error handling based on error patterns"""
        enhanced_code = pattern.template_code
        
        # Wrap test body in try-except for better error handling
        if 'try:' not in enhanced_code:
            # Find the test function body and wrap it
            lines = enhanced_code.split('\n')
            new_lines = []
            in_test_function = False
            indent_level = 0
            
            for line in lines:
                if line.strip().startswith('def test_'):
                    new_lines.append(line)
                    in_test_function = True
                    indent_level = len(line) - len(line.lstrip())
                elif in_test_function and line.strip() and not line.startswith(' ' * (indent_level + 4)):
                    # End of function
                    in_test_function = False
                    new_lines.append(line)
                elif in_test_function and line.strip():
                    # Inside test function - add try-except wrapper
                    if 'try:' not in '\n'.join(new_lines[-5:]):  # Don't add if already exists
                        new_lines.append(' ' * (indent_level + 4) + 'try:')
                        new_lines.append(' ' * (indent_level + 8) + line.strip())
                        new_lines.append(' ' * (indent_level + 4) + 'except Exception as e:')
                        new_lines.append(' ' * (indent_level + 8) + 'pytest.fail(f"Test failed with error: {e}")')
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            enhanced_code = '\n'.join(new_lines)
        
        # Create evolved pattern
        evolved = TestPattern(
            pattern_id=f"{pattern.pattern_id}_error_handling_improved",
            name=f"{pattern.name} (Improved Error Handling)",
            template_code=enhanced_code,
            category=pattern.category,
            complexity_level=pattern.complexity_level + 1,
            adaptation_count=pattern.adaptation_count
        )
        
        return evolved
    
    def _tune_performance(self, pattern: TestPattern) -> Optional[TestPattern]:
        """Tune performance based on execution time patterns"""
        # Add performance optimizations
        enhanced_code = pattern.template_code
        
        # Add caching for expensive operations
        if 'def test_' in enhanced_code and '@lru_cache' not in enhanced_code:
            enhanced_code = 'from functools import lru_cache\n\n' + enhanced_code
            
            # Add caching to helper functions if any
            enhanced_code = enhanced_code.replace(
                'def helper_',
                '@lru_cache(maxsize=32)\ndef helper_'
            )
        
        # Add performance monitoring
        perf_wrapper = '''
    import time
    start_time = time.time()
    
    '''
        
        perf_end = '''
    
    end_time = time.time()
    execution_time = end_time - start_time
    assert execution_time < 30.0, f"Test too slow: {execution_time:.2f}s"
    '''
        
        enhanced_code = enhanced_code.replace(
            'def test_',
            f'{perf_wrapper}def test_'
        ) + perf_end
        
        # Create evolved pattern
        evolved = TestPattern(
            pattern_id=f"{pattern.pattern_id}_performance_tuned",
            name=f"{pattern.name} (Performance Tuned)",
            template_code=enhanced_code,
            category=pattern.category,
            complexity_level=pattern.complexity_level,
            adaptation_count=pattern.adaptation_count
        )
        
        return evolved
    
    def _expand_parametrization(self, pattern: TestPattern) -> Optional[TestPattern]:
        """Expand parametrization based on success patterns"""
        if '@pytest.mark.parametrize' not in pattern.template_code:
            return None  # Only expand existing parametrized tests
        
        # Add more test cases based on successful patterns
        enhanced_code = pattern.template_code
        
        # Find parametrize decorator and expand test cases
        import re
        param_match = re.search(r'@pytest\.mark\.parametrize\(["\']([^"\']+)["\'],\s*\[([^\]]+)\]', enhanced_code)
        
        if param_match:
            param_names = param_match.group(1)
            current_cases = param_match.group(2)
            
            # Add edge cases
            if 'str' in param_names.lower():
                additional_cases = ', ("", "empty"), ("   ", "whitespace"), ("très_long_string_" * 10, "long")'
            elif 'int' in param_names.lower():
                additional_cases = ', (0, "zero"), (-1, "negative"), (sys.maxsize, "max_int")'
            else:
                additional_cases = ', (None, "none_case")'
            
            enhanced_code = enhanced_code.replace(
                f'[{current_cases}]',
                f'[{current_cases}{additional_cases}]'
            )
        
        # Create evolved pattern
        evolved = TestPattern(
            pattern_id=f"{pattern.pattern_id}_parametrization_expanded",
            name=f"{pattern.name} (Expanded Parametrization)",
            template_code=enhanced_code,
            category=pattern.category,
            complexity_level=pattern.complexity_level + 1,
            adaptation_count=pattern.adaptation_count
        )
        
        return evolved


class AdaptiveFeedbackCollector:
    """Collects and analyzes feedback from test executions"""
    
    def __init__(self):
        self.feedback_history = deque(maxlen=1000)  # Keep last 1000 feedback entries
        self.pattern_performance = defaultdict(list)
        self.adaptation_effectiveness = defaultdict(list)
    
    def collect_execution_feedback(self, pattern_id: str, execution: TestExecution) -> None:
        """Collect feedback from a test execution"""
        feedback_entry = {
            'timestamp': datetime.now(),
            'pattern_id': pattern_id,
            'execution': execution,
            'environmental_factors': self._capture_environment()
        }
        
        self.feedback_history.append(feedback_entry)
        self.pattern_performance[pattern_id].append(execution.success_score)
        
        # Keep only recent performance data
        if len(self.pattern_performance[pattern_id]) > 50:
            self.pattern_performance[pattern_id] = self.pattern_performance[pattern_id][-50:]
    
    def collect_adaptation_feedback(self, original_pattern_id: str, evolved_pattern_id: str, 
                                  improvement_score: float) -> None:
        """Collect feedback on pattern adaptation effectiveness"""
        self.adaptation_effectiveness[original_pattern_id].append({
            'evolved_pattern_id': evolved_pattern_id,
            'improvement_score': improvement_score,
            'timestamp': datetime.now()
        })
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environmental factors that might affect test performance"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_load': self._get_system_load(),
            'memory_available': self._get_memory_info(),
            'time_of_day': datetime.now().hour
        }
    
    def _get_system_load(self) -> float:
        """Get system load (simplified)"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def _get_memory_info(self) -> float:
        """Get available memory percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def analyze_pattern_trends(self, pattern_id: str) -> Dict[str, Any]:
        """Analyze performance trends for a specific pattern"""
        if pattern_id not in self.pattern_performance:
            return {}
        
        scores = self.pattern_performance[pattern_id]
        
        if len(scores) < 5:
            return {'status': 'insufficient_data'}
        
        # Calculate trend
        recent_scores = scores[-10:]  # Last 10 executions
        older_scores = scores[-20:-10] if len(scores) >= 20 else scores[:-10]
        
        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores) if older_scores else recent_avg
        
        trend = 'improving' if recent_avg > older_avg + 0.05 else \
               'declining' if recent_avg < older_avg - 0.05 else 'stable'
        
        return {
            'trend': trend,
            'recent_average': recent_avg,
            'overall_average': statistics.mean(scores),
            'stability': 1.0 - statistics.stdev(recent_scores),
            'execution_count': len(scores),
            'recommendation': self._generate_recommendation(trend, recent_avg, scores)
        }
    
    def _generate_recommendation(self, trend: str, recent_avg: float, scores: List[float]) -> str:
        """Generate recommendation based on pattern analysis"""
        if trend == 'declining' and recent_avg < 0.7:
            return "Pattern needs immediate adaptation - performance declining"
        elif trend == 'stable' and recent_avg > 0.9:
            return "Pattern performing well - consider as template for others"
        elif trend == 'improving':
            return "Pattern improving - monitor and document successful adaptations"
        elif recent_avg < 0.5:
            return "Pattern consistently underperforming - consider retirement"
        else:
            return "Pattern performance acceptable - continue monitoring"


class AdaptiveTestingFramework:
    """Main framework for adaptive testing with self-improving patterns"""
    
    def __init__(self, adaptation_strategy: AdaptationStrategy = AdaptationStrategy.HYBRID):
        self.patterns: Dict[str, TestPattern] = {}
        self.evolution_engine = PatternEvolutionEngine()
        self.feedback_collector = AdaptiveFeedbackCollector()
        self.adaptation_strategy = adaptation_strategy
        self.adaptation_rules: List[AdaptationRule] = []
        self.performance_history = defaultdict(list)
        
        # Initialize with basic adaptation rules
        self._initialize_adaptation_rules()
    
    def _initialize_adaptation_rules(self):
        """Initialize default adaptation rules"""
        default_rules = [
            AdaptationRule(
                rule_id="timeout_rule",
                name="Timeout Optimization Rule",
                condition="pattern.avg_execution_time > 10.0 and timeout_failures > 2",
                action="optimize_timeout",
                priority=1
            ),
            AdaptationRule(
                rule_id="assertion_rule", 
                name="Assertion Enhancement Rule",
                condition="pattern.success_rate < 0.7 and failure_count > 3",
                action="enhance_assertions",
                priority=2
            ),
            AdaptationRule(
                rule_id="performance_rule",
                name="Performance Tuning Rule", 
                condition="pattern.avg_execution_time > 5.0 and slow_executions > 5",
                action="tune_performance",
                priority=3
            )
        ]
        
        self.adaptation_rules.extend(default_rules)
    
    def register_pattern(self, pattern: TestPattern) -> None:
        """Register a new test pattern for adaptation"""
        self.patterns[pattern.pattern_id] = pattern
        print(f"Registered adaptive pattern: {pattern.name}")
    
    def execute_pattern(self, pattern_id: str, test_function: Callable[[], Any]) -> TestExecution:
        """Execute a test pattern and collect feedback"""
        if pattern_id not in self.patterns:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        pattern = self.patterns[pattern_id]
        start_time = time.time()
        
        try:
            # Execute the test
            test_function()
            outcome = TestOutcome.PASS
            error_message = None
        except AssertionError as e:
            outcome = TestOutcome.FAIL
            error_message = str(e)
        except TimeoutError:
            outcome = TestOutcome.TIMEOUT
            error_message = "Test timed out"
        except Exception as e:
            outcome = TestOutcome.ERROR
            error_message = str(e)
        
        execution_duration = time.time() - start_time
        
        # Create execution record
        execution = TestExecution(
            test_id=f"{pattern_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            execution_time=datetime.now(),
            outcome=outcome,
            execution_duration=execution_duration,
            error_message=error_message,
            performance_metrics={'cpu_usage': 0.5, 'memory_usage': 0.3}  # Placeholder
        )
        
        # Add to pattern history
        pattern.add_execution(execution)
        
        # Collect feedback
        self.feedback_collector.collect_execution_feedback(pattern_id, execution)
        
        return execution
    
    def adapt_patterns(self, adaptation_threshold: float = 1.0) -> List[TestPattern]:
        """Adapt patterns that need improvement"""
        adapted_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            if pattern.adaptation_score >= adaptation_threshold:
                print(f"Adapting pattern {pattern.name} (score: {pattern.adaptation_score:.2f})")
                
                # Evolve the pattern
                evolved_pattern = self.evolution_engine.evolve_pattern(pattern)
                
                if evolved_pattern:
                    # Register evolved pattern
                    self.register_pattern(evolved_pattern)
                    adapted_patterns.append(evolved_pattern)
                    
                    # Calculate improvement score (simplified)
                    improvement_score = self._calculate_improvement_score(pattern, evolved_pattern)
                    
                    # Collect adaptation feedback
                    self.feedback_collector.collect_adaptation_feedback(
                        pattern_id, evolved_pattern.pattern_id, improvement_score
                    )
                    
                    print(f"Created evolved pattern: {evolved_pattern.name}")
        
        return adapted_patterns
    
    def _calculate_improvement_score(self, original: TestPattern, evolved: TestPattern) -> float:
        """Calculate improvement score between original and evolved pattern"""
        # This is a placeholder - in practice would compare actual performance
        base_score = 0.1 * evolved.adaptation_count  # Small improvement per adaptation
        
        # Bonus for addressing specific issues
        if 'timeout' in evolved.pattern_id.lower():
            base_score += 0.3
        elif 'assertion' in evolved.pattern_id.lower():
            base_score += 0.4
        elif 'performance' in evolved.pattern_id.lower():
            base_score += 0.5
        
        return min(1.0, base_score)
    
    def get_pattern_recommendations(self, pattern_id: str) -> Dict[str, Any]:
        """Get recommendations for improving a specific pattern"""
        if pattern_id not in self.patterns:
            return {'error': 'Pattern not found'}
        
        pattern = self.patterns[pattern_id]
        trends = self.feedback_collector.analyze_pattern_trends(pattern_id)
        
        recommendations = {
            'pattern_analysis': {
                'success_rate': pattern.success_rate,
                'avg_execution_time': pattern.avg_execution_time,
                'confidence_score': pattern.confidence_score,
                'adaptation_score': pattern.adaptation_score
            },
            'trends': trends,
            'suggested_adaptations': []
        }
        
        # Add specific recommendations based on analysis
        if pattern.adaptation_score > 1.0:
            recommendations['suggested_adaptations'].append('Pattern needs adaptation')
        
        if pattern.success_rate < 0.8:
            recommendations['suggested_adaptations'].append('Improve assertion robustness')
        
        if pattern.avg_execution_time > 10.0:
            recommendations['suggested_adaptations'].append('Optimize for performance')
        
        return recommendations
    
    def export_adaptation_report(self, output_path: str) -> None:
        """Export comprehensive adaptation report"""
        report_data = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'total_patterns': len(self.patterns),
                'adaptation_strategy': self.adaptation_strategy.value,
                'total_adaptations': sum(p.adaptation_count for p in self.patterns.values())
            },
            'pattern_summary': [
                {
                    'pattern_id': pattern.pattern_id,
                    'name': pattern.name,
                    'success_rate': pattern.success_rate,
                    'avg_execution_time': pattern.avg_execution_time,
                    'confidence_score': pattern.confidence_score,
                    'adaptation_score': pattern.adaptation_score,
                    'adaptation_count': pattern.adaptation_count,
                    'execution_count': len(pattern.execution_history)
                }
                for pattern in self.patterns.values()
            ],
            'performance_trends': {
                pattern_id: self.feedback_collector.analyze_pattern_trends(pattern_id)
                for pattern_id in self.patterns.keys()
            },
            'adaptation_rules': [
                {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'success_rate': rule.success_rate,
                    'confidence': rule.confidence,
                    'application_count': rule.application_count
                }
                for rule in self.adaptation_rules
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Adaptation report exported to: {output_path}")
    
    def cleanup_obsolete_patterns(self, min_confidence: float = 0.3) -> List[str]:
        """Remove patterns with persistently low performance"""
        obsolete_patterns = []
        
        for pattern_id, pattern in list(self.patterns.items()):
            if (pattern.confidence_score < min_confidence and 
                len(pattern.execution_history) >= 20 and
                pattern.success_rate < 0.4):
                
                obsolete_patterns.append(pattern_id)
                del self.patterns[pattern_id]
                print(f"Removed obsolete pattern: {pattern.name}")
        
        return obsolete_patterns


# Testing framework
class AdaptiveTestingFrameworkTests:
    """Testing framework for adaptive testing system"""
    
    def test_pattern_registration(self) -> bool:
        """Test pattern registration and retrieval"""
        try:
            framework = AdaptiveTestingFramework()
            
            test_pattern = TestPattern(
                pattern_id="test_pattern_001",
                name="Test Pattern",
                template_code="def test_example(): assert True",
                category="unit",
                complexity_level=1
            )
            
            framework.register_pattern(test_pattern)
            
            assert len(framework.patterns) == 1
            assert "test_pattern_001" in framework.patterns
            
            return True
        except Exception as e:
            print(f"Pattern registration test failed: {e}")
            return False
    
    def test_pattern_execution(self) -> bool:
        """Test pattern execution and feedback collection"""
        try:
            framework = AdaptiveTestingFramework()
            
            test_pattern = TestPattern(
                pattern_id="test_pattern_002", 
                name="Execution Test Pattern",
                template_code="def test_execution(): assert True",
                category="unit",
                complexity_level=1
            )
            
            framework.register_pattern(test_pattern)
            
            # Execute pattern
            def sample_test():
                assert 2 + 2 == 4
            
            execution = framework.execute_pattern("test_pattern_002", sample_test)
            
            assert execution.outcome == TestOutcome.PASS
            assert len(test_pattern.execution_history) == 1
            
            return True
        except Exception as e:
            print(f"Pattern execution test failed: {e}")
            return False
    
    def test_pattern_adaptation(self) -> bool:
        """Test pattern adaptation mechanism"""
        try:
            framework = AdaptiveTestingFramework()
            
            # Create a pattern that needs adaptation
            failing_pattern = TestPattern(
                pattern_id="failing_pattern",
                name="Failing Pattern",
                template_code="def test_failing(): assert False",  # Always fails
                category="unit",
                complexity_level=1,
                success_rate=0.2,  # Low success rate
                adaptation_count=0
            )
            
            # Add some failing executions
            for i in range(5):
                execution = TestExecution(
                    test_id=f"fail_{i}",
                    execution_time=datetime.now(),
                    outcome=TestOutcome.FAIL,
                    execution_duration=1.0,
                    error_message="Assertion failed"
                )
                failing_pattern.add_execution(execution)
            
            framework.register_pattern(failing_pattern)
            
            # Trigger adaptation
            adapted_patterns = framework.adapt_patterns(adaptation_threshold=0.5)
            
            # Should create adapted pattern
            assert len(adapted_patterns) > 0
            assert any('assertions_enhanced' in p.pattern_id for p in adapted_patterns)
            
            return True
        except Exception as e:
            print(f"Pattern adaptation test failed: {e}")
            return False
    
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all adaptive testing framework tests"""
        tests = [
            'test_pattern_registration',
            'test_pattern_execution', 
            'test_pattern_adaptation'
        ]
        
        results = {}
        for test_name in tests:
            try:
                result = getattr(self, test_name)()
                results[test_name] = result
                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: FAILED - {e}")
        
        return results


# Main execution
if __name__ == "__main__":
    print("🧠 Adaptive Testing Framework")
    
    # Run tests
    test_framework = AdaptiveTestingFrameworkTests()
    results = test_framework.run_comprehensive_tests()
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All adaptive testing framework tests passed!")
        
        # Demonstrate adaptive testing
        print("\n🚀 Running adaptive testing demonstration...")
        
        framework = AdaptiveTestingFramework()
        
        # Create sample patterns
        patterns = [
            TestPattern(
                pattern_id="unit_test_pattern",
                name="Basic Unit Test",
                template_code="def test_unit(): assert result is not None",
                category="unit",
                complexity_level=1
            ),
            TestPattern(
                pattern_id="integration_test_pattern", 
                name="Integration Test",
                template_code="def test_integration(): assert system.connect()",
                category="integration", 
                complexity_level=3
            )
        ]
        
        # Register patterns
        for pattern in patterns:
            framework.register_pattern(pattern)
        
        # Simulate some test executions
        def mock_passing_test():
            time.sleep(0.1)  # Simulate test execution
            assert True
        
        def mock_failing_test():
            time.sleep(0.2)
            assert False, "Mock failure"
        
        # Execute patterns multiple times
        for i in range(10):
            # Most tests pass, some fail
            test_func = mock_passing_test if i < 7 else mock_failing_test
            
            for pattern_id in framework.patterns.keys():
                try:
                    framework.execute_pattern(pattern_id, test_func)
                except:
                    pass  # Expected for failing tests
        
        # Perform adaptation
        adapted = framework.adapt_patterns(adaptation_threshold=0.8)
        
        # Generate report
        output_path = f"adaptive_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        framework.export_adaptation_report(output_path)
        
        print(f"\n📈 Adaptive Testing Demonstration Complete:")
        print(f"  Original patterns: {len(patterns)}")
        print(f"  Adapted patterns: {len(adapted)}")
        print(f"  Total patterns: {len(framework.patterns)}")
        print(f"  Report exported: {output_path}")
        
        # Show pattern recommendations
        for pattern_id in list(framework.patterns.keys())[:2]:  # Show first 2
            recommendations = framework.get_pattern_recommendations(pattern_id)
            print(f"\n📋 Recommendations for {pattern_id}:")
            print(f"  Success rate: {recommendations['pattern_analysis']['success_rate']:.2f}")
            print(f"  Adaptation score: {recommendations['pattern_analysis']['adaptation_score']:.2f}")
            print(f"  Trend: {recommendations['trends'].get('trend', 'unknown')}")
            
    else:
        print("❌ Some tests failed. Check the output above.")