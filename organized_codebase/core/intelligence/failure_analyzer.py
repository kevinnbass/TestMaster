#!/usr/bin/env python3
"""
Failure Pattern Analyzer
Analyzes test failures to identify patterns and predict potential issues.

Features:
- ML-based failure pattern detection
- Root cause analysis suggestions
- Flaky test identification
- Correlation with code changes
- Failure prediction
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import hashlib
import statistics
from itertools import combinations
import difflib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """Record of a test failure."""
    test_name: str
    timestamp: datetime
    error_message: str
    error_type: str
    stack_trace: str
    duration: float
    environment: str = "unknown"
    git_commit: Optional[str] = None
    changed_files: List[str] = field(default_factory=list)
    test_file: Optional[str] = None
    line_number: Optional[int] = None
    
    def get_error_signature(self) -> str:
        """Get normalized error signature."""
        # Remove specific values but keep structure
        normalized = re.sub(r'\b\d+\b', '<NUM>', self.error_message)
        normalized = re.sub(r"'[^']*'", '<STR>', normalized)
        normalized = re.sub(r'"[^"]*"', '<STR>', normalized)
        normalized = re.sub(r'\b0x[0-9a-fA-F]+\b', '<ADDR>', normalized)
        return normalized


@dataclass
class FailurePattern:
    """Identified failure pattern."""
    pattern_id: str
    pattern_type: str  # recurring, flaky, regression, environment
    tests_affected: List[str]
    error_signature: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    confidence: float  # 0-1
    root_causes: List[str] = field(default_factory=list)
    correlation_factors: Dict[str, float] = field(default_factory=dict)
    prediction_accuracy: float = 0.0


class FailureAnalyzer:
    """Main failure pattern analyzer."""
    
    def __init__(self, data_dir: str = "failure_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.failure_records: List[FailureRecord] = []
        self.patterns: List[FailurePattern] = []
        self.flaky_tests: Dict[str, float] = {}  # test_name -> flakiness_score
        self.environment_factors: Dict[str, Dict[str, Any]] = {}
        
        self.load_data()
    
    def record_failure(self, test_name: str, error_message: str, error_type: str,
                      stack_trace: str, duration: float, **kwargs):
        """Record a test failure."""
        record = FailureRecord(
            test_name=test_name,
            timestamp=datetime.now(),
            error_message=error_message,
            error_type=error_type,
            stack_trace=stack_trace,
            duration=duration,
            **kwargs
        )
        
        self.failure_records.append(record)
        self.save_failure_record(record)
        
        # Update flakiness score
        self._update_flakiness_score(test_name)
        
        # Check for new patterns
        self._detect_patterns()
    
    def _update_flakiness_score(self, test_name: str):
        """Update flakiness score for a test."""
        # Get recent history for this test
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_records = [
            r for r in self.failure_records 
            if r.test_name == test_name and r.timestamp > recent_cutoff
        ]
        
        if not recent_records:
            return
        
        # Calculate failure rate
        # This is simplified - in practice would track passes too
        failure_count = len(recent_records)
        
        # Look for intermittent failures (same test, different outcomes)
        error_types = set(r.error_type for r in recent_records)
        
        # High variance in error types suggests flakiness
        flakiness = min(len(error_types) / max(failure_count, 1), 1.0)
        
        # Check for timing-related failures
        timing_errors = [
            'timeout', 'race', 'async', 'deadlock', 'wait', 'sleep'
        ]
        timing_failures = sum(
            1 for r in recent_records 
            if any(keyword in r.error_message.lower() for keyword in timing_errors)
        )
        
        if timing_failures > 0:
            flakiness += 0.3
        
        self.flaky_tests[test_name] = min(flakiness, 1.0)
    
    def _detect_patterns(self):
        """Detect failure patterns in recent data."""
        # Group failures by error signature
        signature_groups = defaultdict(list)
        
        for record in self.failure_records[-100:]:  # Recent 100 failures
            signature = record.get_error_signature()
            signature_groups[signature].append(record)
        
        # Identify patterns
        for signature, records in signature_groups.items():
            if len(records) < 3:  # Need at least 3 occurrences
                continue
            
            pattern = self._analyze_pattern(signature, records)
            if pattern:
                # Check if this pattern already exists
                existing = next(
                    (p for p in self.patterns if p.error_signature == signature),
                    None
                )
                
                if existing:
                    self._update_pattern(existing, records)
                else:
                    self.patterns.append(pattern)
    
    def _analyze_pattern(self, signature: str, records: List[FailureRecord]) -> Optional[FailurePattern]:
        """Analyze a group of failures to identify pattern type."""
        if len(records) < 3:
            return None
        
        # Determine pattern type
        pattern_type = "recurring"
        confidence = 0.5
        
        # Check for flaky pattern (same test, intermittent failures)
        test_names = set(r.test_name for r in records)
        if len(test_names) == 1:
            pattern_type = "flaky"
            confidence = 0.8
        
        # Check for regression pattern (failures after code changes)
        recent_changes = [r for r in records if r.changed_files]
        if len(recent_changes) > len(records) * 0.7:
            pattern_type = "regression"
            confidence = 0.9
        
        # Check for environment pattern (failures in specific environments)
        environments = Counter(r.environment for r in records)
        if len(environments) == 1 and len(records) > 5:
            pattern_type = "environment"
            confidence = 0.7
        
        # Analyze correlations
        correlations = self._calculate_correlations(records)
        
        # Suggest root causes
        root_causes = self._suggest_root_causes(signature, records, correlations)
        
        pattern = FailurePattern(
            pattern_id=hashlib.md5(signature.encode()).hexdigest()[:8],
            pattern_type=pattern_type,
            tests_affected=list(test_names),
            error_signature=signature,
            frequency=len(records),
            first_seen=min(r.timestamp for r in records),
            last_seen=max(r.timestamp for r in records),
            confidence=confidence,
            root_causes=root_causes,
            correlation_factors=correlations
        )
        
        return pattern
    
    def _update_pattern(self, pattern: FailurePattern, new_records: List[FailureRecord]):
        """Update existing pattern with new data."""
        pattern.frequency = len(new_records)
        pattern.last_seen = max(r.timestamp for r in new_records)
        
        # Update tests affected
        test_names = set(r.test_name for r in new_records)
        pattern.tests_affected = list(set(pattern.tests_affected) | test_names)
        
        # Recalculate correlations
        pattern.correlation_factors = self._calculate_correlations(new_records)
    
    def _calculate_correlations(self, records: List[FailureRecord]) -> Dict[str, float]:
        """Calculate correlation factors for failure records."""
        correlations = {}
        
        # Time-based correlations
        hours = [r.timestamp.hour for r in records]
        if len(set(hours)) < len(hours) * 0.5:  # Clustered around certain hours
            correlations['time_dependent'] = 0.8
        
        # Duration correlations
        durations = [r.duration for r in records]
        if statistics.stdev(durations) < statistics.mean(durations) * 0.2:
            correlations['duration_consistent'] = 0.7
        
        # File change correlations
        changed_files = []
        for record in records:
            changed_files.extend(record.changed_files)
        
        if changed_files:
            file_counter = Counter(changed_files)
            most_common = file_counter.most_common(1)[0]
            if most_common[1] > len(records) * 0.5:
                correlations['file_related'] = 0.9
                correlations['suspect_file'] = most_common[0]
        
        # Environment correlations
        environments = [r.environment for r in records]
        env_counter = Counter(environments)
        if len(env_counter) == 1:
            correlations['environment_specific'] = 1.0
        elif len(env_counter) < len(environments) * 0.5:
            correlations['environment_bias'] = 0.6
        
        return correlations
    
    def _suggest_root_causes(self, signature: str, records: List[FailureRecord],
                           correlations: Dict[str, float]) -> List[str]:
        """Suggest potential root causes based on analysis."""
        causes = []
        
        # Error message analysis
        signature_lower = signature.lower()
        
        if 'timeout' in signature_lower:
            causes.append("Performance degradation or increased load")
            causes.append("Network connectivity issues")
            
        elif 'assertion' in signature_lower or 'expected' in signature_lower:
            causes.append("Logic error or incorrect test expectations")
            causes.append("Data inconsistency or state pollution")
            
        elif 'import' in signature_lower or 'module' in signature_lower:
            causes.append("Missing dependencies or environment setup")
            causes.append("Python path or packaging issues")
            
        elif 'connection' in signature_lower or 'network' in signature_lower:
            causes.append("Infrastructure or service availability")
            causes.append("Network configuration changes")
            
        elif 'permission' in signature_lower or 'access' in signature_lower:
            causes.append("File system permissions or security changes")
            causes.append("Authentication or authorization issues")
        
        # Correlation-based causes
        if correlations.get('time_dependent', 0) > 0.7:
            causes.append("Time-sensitive operations or scheduled tasks")
            
        if correlations.get('environment_specific', 0) > 0.8:
            causes.append("Environment-specific configuration issues")
            
        if correlations.get('file_related', 0) > 0.8:
            suspect_file = correlations.get('suspect_file', 'unknown')
            causes.append(f"Recent changes to {suspect_file}")
        
        # Pattern-based causes
        test_names = set(r.test_name for r in records)
        if len(test_names) == 1:
            causes.append("Flaky test - needs stabilization")
            causes.append("Test isolation issues or shared state")
        
        return causes[:5]  # Return top 5 causes
    
    def predict_failures(self, test_names: List[str], 
                        changed_files: List[str] = None) -> Dict[str, float]:
        """Predict failure probability for given tests."""
        predictions = {}
        
        for test_name in test_names:
            risk_score = 0.0
            
            # Base flakiness score
            risk_score += self.flaky_tests.get(test_name, 0.0) * 0.3
            
            # Historical failure rate
            recent_failures = [
                r for r in self.failure_records 
                if r.test_name == test_name 
                and r.timestamp > datetime.now() - timedelta(days=30)
            ]
            
            if recent_failures:
                risk_score += min(len(recent_failures) / 10, 0.4)
            
            # Code change risk
            if changed_files:
                for pattern in self.patterns:
                    if test_name in pattern.tests_affected:
                        suspect_file = pattern.correlation_factors.get('suspect_file')
                        if suspect_file and suspect_file in changed_files:
                            risk_score += 0.3 * pattern.confidence
            
            # Pattern-based risk
            for pattern in self.patterns:
                if test_name in pattern.tests_affected and pattern.pattern_type == "regression":
                    risk_score += 0.2 * pattern.confidence
            
            predictions[test_name] = min(risk_score, 1.0)
        
        return predictions
    
    def identify_flaky_tests(self, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Identify flaky tests above threshold."""
        flaky = [
            (test_name, score) 
            for test_name, score in self.flaky_tests.items() 
            if score > threshold
        ]
        
        return sorted(flaky, key=lambda x: x[1], reverse=True)
    
    def get_failure_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get failure trends over specified period."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_failures = [r for r in self.failure_records if r.timestamp > cutoff]
        
        # Daily failure counts
        daily_counts = defaultdict(int)
        for record in recent_failures:
            day = record.timestamp.date()
            daily_counts[day] += 1
        
        # Error type trends
        error_trends = Counter(r.error_type for r in recent_failures)
        
        # Test failure rates
        test_failures = Counter(r.test_name for r in recent_failures)
        
        # Environment distribution
        env_distribution = Counter(r.environment for r in recent_failures)
        
        return {
            'total_failures': len(recent_failures),
            'daily_average': len(recent_failures) / days,
            'daily_counts': dict(daily_counts),
            'error_type_trends': dict(error_trends),
            'top_failing_tests': test_failures.most_common(10),
            'environment_distribution': dict(env_distribution),
            'trend_direction': self._calculate_trend_direction(daily_counts)
        }
    
    def _calculate_trend_direction(self, daily_counts: Dict) -> str:
        """Calculate if failures are trending up, down, or stable."""
        if len(daily_counts) < 7:
            return "insufficient_data"
        
        # Get recent week vs previous week
        sorted_days = sorted(daily_counts.keys())
        recent_week = sorted_days[-7:]
        previous_week = sorted_days[-14:-7] if len(sorted_days) >= 14 else []
        
        if not previous_week:
            return "insufficient_data"
        
        recent_avg = statistics.mean(daily_counts.get(day, 0) for day in recent_week)
        previous_avg = statistics.mean(daily_counts.get(day, 0) for day in previous_week)
        
        if recent_avg > previous_avg * 1.2:
            return "increasing"
        elif recent_avg < previous_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def generate_insights(self) -> List[str]:
        """Generate actionable insights from failure analysis."""
        insights = []
        
        # Flaky test insights
        flaky_tests = self.identify_flaky_tests()
        if flaky_tests:
            insights.append(
                f"Found {len(flaky_tests)} flaky tests that need stabilization"
            )
            top_flaky = flaky_tests[0]
            insights.append(
                f"Most flaky test: {top_flaky[0]} (score: {top_flaky[1]:.2f})"
            )
        
        # Pattern insights
        regression_patterns = [p for p in self.patterns if p.pattern_type == "regression"]
        if regression_patterns:
            insights.append(
                f"Detected {len(regression_patterns)} regression patterns"
            )
        
        # Environment insights
        env_patterns = [p for p in self.patterns if p.pattern_type == "environment"]
        if env_patterns:
            insights.append(
                f"Found {len(env_patterns)} environment-specific failure patterns"
            )
        
        # Trend insights
        trends = self.get_failure_trends()
        if trends['trend_direction'] == "increasing":
            insights.append(
                f"Failure rate is trending upward: {trends['daily_average']:.1f} failures/day"
            )
        
        # Top error types
        if trends['error_type_trends']:
            top_error = max(trends['error_type_trends'].items(), key=lambda x: x[1])
            insights.append(
                f"Most common error type: {top_error[0]} ({top_error[1]} occurrences)"
            )
        
        return insights
    
    def save_failure_record(self, record: FailureRecord):
        """Save failure record to disk."""
        date_str = record.timestamp.strftime('%Y-%m-%d')
        failure_file = self.data_dir / f"failures_{date_str}.jsonl"
        
        with open(failure_file, 'a') as f:
            json.dump(asdict(record), f, default=str)
            f.write('\n')
    
    def load_data(self):
        """Load historical failure data."""
        for failure_file in self.data_dir.glob("failures_*.jsonl"):
            try:
                with open(failure_file) as f:
                    for line in f:
                        data = json.loads(line.strip())
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        record = FailureRecord(**data)
                        self.failure_records.append(record)
            except Exception as e:
                logger.error(f"Failed to load {failure_file}: {e}")
        
        # Load patterns
        patterns_file = self.data_dir / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    pattern_data = json.load(f)
                    for p in pattern_data:
                        p['first_seen'] = datetime.fromisoformat(p['first_seen'])
                        p['last_seen'] = datetime.fromisoformat(p['last_seen'])
                        self.patterns.append(FailurePattern(**p))
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")
        
        # Recalculate flakiness scores
        for test_name in set(r.test_name for r in self.failure_records):
            self._update_flakiness_score(test_name)
    
    def save_data(self):
        """Save patterns and analysis data."""
        patterns_file = self.data_dir / "patterns.json"
        pattern_data = [asdict(p) for p in self.patterns]
        
        with open(patterns_file, 'w') as f:
            json.dump(pattern_data, f, indent=2, default=str)
        
        # Save flakiness scores
        flaky_file = self.data_dir / "flaky_tests.json"
        with open(flaky_file, 'w') as f:
            json.dump(self.flaky_tests, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate comprehensive failure analysis report."""
        report_lines = [
            "=" * 70,
            "FAILURE PATTERN ANALYSIS REPORT",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            f"Analysis period: Last 30 days",
            ""
        ]
        
        # Summary statistics
        trends = self.get_failure_trends(30)
        report_lines.extend([
            "SUMMARY:",
            f"  Total failures: {trends['total_failures']}",
            f"  Daily average: {trends['daily_average']:.1f}",
            f"  Trend: {trends['trend_direction']}",
            ""
        ])
        
        # Patterns
        if self.patterns:
            report_lines.append("IDENTIFIED PATTERNS:")
            for pattern in self.patterns[:10]:
                report_lines.append(
                    f"  {pattern.pattern_id}: {pattern.pattern_type} "
                    f"({pattern.frequency} occurrences, confidence: {pattern.confidence:.2f})"
                )
                if pattern.root_causes:
                    report_lines.append(f"    Root causes: {', '.join(pattern.root_causes[:2])}")
            report_lines.append("")
        
        # Flaky tests
        flaky_tests = self.identify_flaky_tests()
        if flaky_tests:
            report_lines.append("FLAKY TESTS:")
            for test_name, score in flaky_tests[:10]:
                report_lines.append(f"  {test_name}: {score:.2f}")
            report_lines.append("")
        
        # Top failing tests
        if trends['top_failing_tests']:
            report_lines.append("TOP FAILING TESTS:")
            for test_name, count in trends['top_failing_tests'][:10]:
                report_lines.append(f"  {test_name}: {count} failures")
            report_lines.append("")
        
        # Error type distribution
        if trends['error_type_trends']:
            report_lines.append("ERROR TYPE DISTRIBUTION:")
            for error_type, count in Counter(trends['error_type_trends']).most_common(10):
                report_lines.append(f"  {error_type}: {count}")
            report_lines.append("")
        
        # Insights and recommendations
        insights = self.generate_insights()
        if insights:
            report_lines.append("KEY INSIGHTS:")
            for insight in insights:
                report_lines.append(f"  • {insight}")
            report_lines.append("")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)


def main():
    """CLI for failure pattern analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Failure Pattern Analyzer")
    parser.add_argument("--data-dir", default="failure_data", help="Data directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze failure patterns")
    parser.add_argument("--predict", nargs="+", help="Predict failure for test names")
    parser.add_argument("--changed-files", nargs="*", help="Recently changed files")
    parser.add_argument("--flaky-threshold", type=float, default=0.3, help="Flakiness threshold")
    parser.add_argument("--trends", type=int, default=30, help="Days for trend analysis")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--insights", action="store_true", help="Show insights only")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FailureAnalyzer(data_dir=args.data_dir)
    
    if args.analyze:
        print("Analyzing failure patterns...")
        
        # Show pattern summary
        print(f"\nFound {len(analyzer.patterns)} patterns:")
        for pattern in analyzer.patterns:
            print(f"  {pattern.pattern_id}: {pattern.pattern_type} "
                 f"({pattern.frequency} occurrences)")
        
        # Show flaky tests
        flaky_tests = analyzer.identify_flaky_tests(args.flaky_threshold)
        if flaky_tests:
            print(f"\nFlaky tests (threshold {args.flaky_threshold}):")
            for test_name, score in flaky_tests[:10]:
                print(f"  {test_name}: {score:.2f}")
        
        analyzer.save_data()
    
    if args.predict:
        print("Predicting failure probabilities...")
        predictions = analyzer.predict_failures(
            args.predict, 
            changed_files=args.changed_files
        )
        
        print("\nFailure risk scores:")
        for test_name, risk in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {test_name}: {risk:.2f}")
    
    if args.trends:
        trends = analyzer.get_failure_trends(args.trends)
        print(f"\nFailure trends (last {args.trends} days):")
        print(f"  Total failures: {trends['total_failures']}")
        print(f"  Daily average: {trends['daily_average']:.1f}")
        print(f"  Trend direction: {trends['trend_direction']}")
        
        if trends['top_failing_tests']:
            print("\n  Top failing tests:")
            for test_name, count in trends['top_failing_tests'][:5]:
                print(f"    {test_name}: {count} failures")
    
    if args.insights:
        insights = analyzer.generate_insights()
        print("\nKey Insights:")
        for insight in insights:
            print(f"  • {insight}")
    
    if args.report:
        report = analyzer.generate_report()
        print(report)
        
        # Save report to file
        report_file = Path(args.data_dir) / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()