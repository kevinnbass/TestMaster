"""
Real-time Failure Detection System

Inspired by PraisonAI's performance monitoring patterns:
- Function decoration for automatic failure tracking
- Error categorization and statistics collection 
- Real-time failure notifications
- Root cause analysis
"""

import functools
import traceback
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import subprocess
import sys

from core.layer_manager import requires_layer


class FailureType(Enum):
    """Types of test failures."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error" 
    ASSERTION_ERROR = "assertion_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT_ERROR = "timeout_error"
    ENVIRONMENT_ERROR = "environment_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FailureInstance:
    """Single instance of a test failure."""
    test_path: str
    failure_type: FailureType
    error_message: str
    traceback_text: str
    timestamp: datetime
    duration_ms: float = 0.0
    environment_info: Dict[str, Any] = field(default_factory=dict)
    related_files: List[str] = field(default_factory=list)


@dataclass
class FailureReport:
    """Comprehensive failure analysis report."""
    test_path: str
    total_failures: int
    recent_failures: int
    failure_rate: float
    avg_failure_duration: float
    most_common_failure: FailureType
    failure_history: List[FailureInstance]
    is_flaky: bool
    suggested_fixes: List[str]
    last_success: Optional[datetime] = None
    

class FailureDetector:
    """
    Real-time test failure detection and analysis.
    
    Features:
    - Automatic failure categorization
    - Statistical failure tracking
    - Flaky test detection
    - Root cause analysis
    - Performance monitoring
    """
    
    @requires_layer("layer1_test_foundation", "failure_detection")
    def __init__(self, max_history: int = 1000):
        """
        Initialize failure detector.
        
        Args:
            max_history: Maximum number of failure instances to keep
        """
        self.max_history = max_history
        
        # Failure tracking (PraisonAI pattern)
        self._failure_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_calls': 0,
            'failure_count': 0,
            'recent_failures': deque(maxlen=100),
            'recent_times': deque(maxlen=100),
            'last_success': None,
            'last_failure': None
        })
        
        # Failure history
        self._failure_history: List[FailureInstance] = []
        
        # Pattern recognition
        self._failure_patterns: Dict[str, int] = defaultdict(int)
        
        # Environment tracking
        self._environment_baseline = self._capture_environment()
    
    def monitor_test_execution(self, test_identifier: str = None):
        """
        Decorator to monitor test execution for failures.
        
        Args:
            test_identifier: Optional identifier for the test
        """
        def decorator(test_func: Callable) -> Callable:
            @functools.wraps(test_func)
            def wrapper(*args, **kwargs):
                test_name = test_identifier or test_func.__name__
                start_time = time.time()
                
                try:
                    # Execute test
                    result = test_func(*args, **kwargs)
                    
                    # Record success
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_success(test_name, duration_ms)
                    
                    return result
                    
                except Exception as e:
                    # Record failure
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_failure(test_name, e, duration_ms)
                    raise
                    
            return wrapper
        return decorator
    
    def run_test_with_monitoring(self, test_command: List[str], 
                                test_path: str) -> FailureReport:
        """
        Run a test command and monitor for failures.
        
        Args:
            test_command: Command to run the test
            test_path: Path to the test file
            
        Returns:
            Failure report with analysis
        """
        print(f"🔍 Monitoring test execution: {test_path}")
        
        start_time = time.time()
        
        try:
            # Run test with timeout
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path.cwd()
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                # Test passed
                self._record_success(test_path, duration_ms)
                print(f"✅ Test passed: {test_path}")
            else:
                # Test failed
                error_output = result.stderr or result.stdout
                self._record_test_failure(test_path, error_output, duration_ms)
                print(f"❌ Test failed: {test_path}")
            
        except subprocess.TimeoutExpired:
            # Timeout failure
            duration_ms = (time.time() - start_time) * 1000
            self._record_timeout_failure(test_path, duration_ms)
            print(f"⏰ Test timed out: {test_path}")
        
        except Exception as e:
            # Execution failure
            duration_ms = (time.time() - start_time) * 1000
            self._record_execution_failure(test_path, e, duration_ms)
            print(f"💥 Test execution failed: {test_path}")
        
        # Generate report
        return self.generate_failure_report(test_path)
    
    def _record_success(self, test_name: str, duration_ms: float):
        """Record successful test execution."""
        stats = self._failure_stats[test_name]
        stats['total_calls'] += 1
        stats['recent_times'].append(duration_ms)
        stats['last_success'] = datetime.now()
    
    def _record_failure(self, test_name: str, exception: Exception, duration_ms: float):
        """Record test failure from exception."""
        # Categorize failure
        failure_type = self._categorize_exception(exception)
        
        # Create failure instance
        failure = FailureInstance(
            test_path=test_name,
            failure_type=failure_type,
            error_message=str(exception),
            traceback_text=traceback.format_exc(),
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            environment_info=self._capture_environment()
        )
        
        self._store_failure(failure)
    
    def _record_test_failure(self, test_path: str, error_output: str, duration_ms: float):
        """Record test failure from command output."""
        # Parse error output to categorize
        failure_type = self._categorize_error_output(error_output)
        
        failure = FailureInstance(
            test_path=test_path,
            failure_type=failure_type,
            error_message=self._extract_error_message(error_output),
            traceback_text=error_output,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            environment_info=self._capture_environment()
        )
        
        self._store_failure(failure)
    
    def _record_timeout_failure(self, test_path: str, duration_ms: float):
        """Record timeout failure."""
        failure = FailureInstance(
            test_path=test_path,
            failure_type=FailureType.TIMEOUT_ERROR,
            error_message="Test execution timed out",
            traceback_text="Test exceeded maximum execution time",
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            environment_info=self._capture_environment()
        )
        
        self._store_failure(failure)
    
    def _record_execution_failure(self, test_path: str, exception: Exception, duration_ms: float):
        """Record test execution failure."""
        failure = FailureInstance(
            test_path=test_path,
            failure_type=FailureType.RUNTIME_ERROR,
            error_message=f"Test execution failed: {str(exception)}",
            traceback_text=traceback.format_exc(),
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            environment_info=self._capture_environment()
        )
        
        self._store_failure(failure)
    
    def _store_failure(self, failure: FailureInstance):
        """Store failure instance and update statistics."""
        # Add to history
        self._failure_history.append(failure)
        
        # Maintain max history size
        if len(self._failure_history) > self.max_history:
            self._failure_history.pop(0)
        
        # Update statistics
        stats = self._failure_stats[failure.test_path]
        stats['total_calls'] += 1
        stats['failure_count'] += 1
        stats['recent_failures'].append(failure.timestamp)
        stats['last_failure'] = failure.timestamp
        
        # Update failure patterns
        pattern_key = f"{failure.failure_type.value}:{self._extract_pattern(failure.error_message)}"
        self._failure_patterns[pattern_key] += 1
    
    def _categorize_exception(self, exception: Exception) -> FailureType:
        """Categorize exception into failure type."""
        if isinstance(exception, SyntaxError):
            return FailureType.SYNTAX_ERROR
        elif isinstance(exception, ImportError):
            return FailureType.IMPORT_ERROR
        elif isinstance(exception, AssertionError):
            return FailureType.ASSERTION_ERROR
        elif isinstance(exception, TimeoutError):
            return FailureType.TIMEOUT_ERROR
        else:
            return FailureType.RUNTIME_ERROR
    
    def _categorize_error_output(self, error_output: str) -> FailureType:
        """Categorize error from test output."""
        error_lower = error_output.lower()
        
        if "syntaxerror" in error_lower:
            return FailureType.SYNTAX_ERROR
        elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
            return FailureType.IMPORT_ERROR
        elif "assertionerror" in error_lower or "assert" in error_lower:
            return FailureType.ASSERTION_ERROR
        elif "timeout" in error_lower:
            return FailureType.TIMEOUT_ERROR
        elif "environment" in error_lower or "permission" in error_lower:
            return FailureType.ENVIRONMENT_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    def _extract_error_message(self, error_output: str) -> str:
        """Extract concise error message from output."""
        lines = error_output.strip().split('\\n')
        
        # Look for common error patterns
        for line in lines:
            if any(keyword in line.lower() for keyword in 
                  ['error:', 'failed:', 'exception:', 'assert']):
                return line.strip()
        
        # Return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return "Unknown error"
    
    def _extract_pattern(self, error_message: str) -> str:
        """Extract error pattern for recognition."""
        # Simple pattern extraction - could be enhanced
        if "line" in error_message.lower():
            return "line_specific"
        elif "file" in error_message.lower():
            return "file_specific"
        elif "function" in error_message.lower():
            return "function_specific"
        else:
            return "general"
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture current environment information."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_failure_report(self, test_path: str) -> FailureReport:
        """Generate comprehensive failure report for a test."""
        stats = self._failure_stats[test_path]
        
        # Get failure history for this test
        test_failures = [
            f for f in self._failure_history 
            if f.test_path == test_path
        ]
        
        # Calculate metrics
        total_failures = len(test_failures)
        recent_failures = len([
            f for f in test_failures 
            if f.timestamp > datetime.now() - timedelta(hours=24)
        ])
        
        total_calls = stats['total_calls']
        failure_rate = (stats['failure_count'] / max(total_calls, 1)) * 100
        
        avg_duration = 0.0
        if test_failures:
            avg_duration = sum(f.duration_ms for f in test_failures) / len(test_failures)
        
        # Find most common failure type
        failure_types = [f.failure_type for f in test_failures]
        most_common = max(set(failure_types), key=failure_types.count) if failure_types else FailureType.UNKNOWN_ERROR
        
        # Detect flaky tests (alternating pass/fail)
        is_flaky = self._detect_flaky_test(test_path)
        
        # Generate fix suggestions
        suggestions = self._generate_fix_suggestions(test_failures)
        
        return FailureReport(
            test_path=test_path,
            total_failures=total_failures,
            recent_failures=recent_failures,
            failure_rate=failure_rate,
            avg_failure_duration=avg_duration,
            most_common_failure=most_common,
            failure_history=test_failures[-10:],  # Last 10 failures
            is_flaky=is_flaky,
            suggested_fixes=suggestions,
            last_success=stats['last_success']
        )
    
    def _detect_flaky_test(self, test_path: str) -> bool:
        """Detect if a test is flaky (inconsistent results)."""
        stats = self._failure_stats[test_path]
        
        # Simple flaky detection: has both recent successes and failures
        has_recent_success = stats['last_success'] and (
            datetime.now() - stats['last_success'] < timedelta(days=7)
        )
        has_recent_failure = stats['last_failure'] and (
            datetime.now() - stats['last_failure'] < timedelta(days=7)  
        )
        
        return has_recent_success and has_recent_failure
    
    def _generate_fix_suggestions(self, failures: List[FailureInstance]) -> List[str]:
        """Generate fix suggestions based on failure patterns."""
        suggestions = []
        
        if not failures:
            return suggestions
        
        # Analyze failure types
        failure_types = [f.failure_type for f in failures]
        most_common = max(set(failure_types), key=failure_types.count)
        
        if most_common == FailureType.IMPORT_ERROR:
            suggestions.append("Check import statements and module paths")
            suggestions.append("Verify all dependencies are installed")
        elif most_common == FailureType.ASSERTION_ERROR:
            suggestions.append("Review test assertions and expected values")
            suggestions.append("Check if test data or mocks need updating")
        elif most_common == FailureType.SYNTAX_ERROR:
            suggestions.append("Fix syntax errors in test file")
            suggestions.append("Check Python version compatibility")
        elif most_common == FailureType.TIMEOUT_ERROR:
            suggestions.append("Optimize test performance or increase timeout")
            suggestions.append("Check for infinite loops or blocking operations")
        
        # Pattern-based suggestions
        error_messages = [f.error_message for f in failures]
        common_keywords = set()
        for msg in error_messages:
            common_keywords.update(msg.lower().split())
        
        if "permission" in common_keywords:
            suggestions.append("Check file/directory permissions")
        if "connection" in common_keywords:
            suggestions.append("Verify network connectivity or service availability")
        if "memory" in common_keywords:
            suggestions.append("Check memory usage and optimize test data")
        
        return suggestions
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get overall failure statistics."""
        total_tests = len(self._failure_stats)
        total_failures = sum(stats['failure_count'] for stats in self._failure_stats.values())
        total_calls = sum(stats['total_calls'] for stats in self._failure_stats.values())
        
        # Failure type distribution
        failure_type_counts = defaultdict(int)
        for failure in self._failure_history:
            failure_type_counts[failure.failure_type.value] += 1
        
        # Most problematic tests
        problematic_tests = sorted(
            self._failure_stats.items(),
            key=lambda x: x[1]['failure_count'],
            reverse=True
        )[:10]
        
        return {
            "total_tests_monitored": total_tests,
            "total_failures": total_failures,
            "total_test_runs": total_calls,
            "overall_failure_rate": (total_failures / max(total_calls, 1)) * 100,
            "failure_type_distribution": dict(failure_type_counts),
            "most_problematic_tests": [
                {"test": test, "failures": stats['failure_count']}
                for test, stats in problematic_tests
            ],
            "most_common_patterns": dict(sorted(
                self._failure_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }
    
    def clear_history(self, older_than_days: int = 30):
        """Clear old failure history."""
        cutoff = datetime.now() - timedelta(days=older_than_days)
        
        # Clear old failures
        self._failure_history = [
            f for f in self._failure_history 
            if f.timestamp > cutoff
        ]
        
        # Clear old stats
        for stats in self._failure_stats.values():
            stats['recent_failures'] = deque([
                ts for ts in stats['recent_failures']
                if ts > cutoff
            ], maxlen=100)
        
        print(f"🧹 Cleared failure history older than {older_than_days} days")