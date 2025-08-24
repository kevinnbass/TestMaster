#!/usr/bin/env python3
"""
Simple test for our implementations without unicode issues.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_implementations():
    """Test our key implementations."""
    results = []
    
    # Test 1: Realtime Metrics Collector
    try:
        from realtime_metrics_collector import RealtimeMetricsCollector
        collector = RealtimeMetricsCollector()
        collector.start_collection()
        collector.stop_collection()
        results.append(("Realtime Metrics Collector", "PASS"))
    except Exception as e:
        results.append(("Realtime Metrics Collector", f"FAIL: {str(e)[:50]}"))
    
    # Test 2: Performance Profiler
    try:
        from performance_profiler import PerformanceProfiler
        profiler = PerformanceProfiler()
        session = profiler.start_profiling()
        profiler.stop_profiling(session)
        results.append(("Performance Profiler", "PASS"))
    except Exception as e:
        results.append(("Performance Profiler", f"FAIL: {str(e)[:50]}"))
    
    # Test 3: Live Code Quality Monitor
    try:
        from live_code_quality_monitor import LiveCodeQualityMonitor
        monitor = LiveCodeQualityMonitor()
        result = monitor.analyze_file_quality('test_backend_health.py')
        if result:
            results.append(("Live Code Quality Monitor", f"PASS (score: {result.overall_score:.1f})"))
        else:
            results.append(("Live Code Quality Monitor", "FAIL: No result"))
    except Exception as e:
        results.append(("Live Code Quality Monitor", f"FAIL: {str(e)[:50]}"))
    
    # Test 4: Enhanced Incremental AST Engine
    try:
        from enhanced_incremental_ast_engine import EnhancedIncrementalASTEngine
        engine = EnhancedIncrementalASTEngine()
        # Just test initialization
        results.append(("Enhanced Incremental AST Engine", "PASS"))
    except Exception as e:
        results.append(("Enhanced Incremental AST Engine", f"FAIL: {str(e)[:50]}"))
    
    # Test 5: Risk-Based Test Targeter
    try:
        from risk_based_test_targeter import RiskBasedTestTargeter
        targeter = RiskBasedTestTargeter()
        profile = targeter.analyze_risk('test_backend_health.py', include_history=False)
        if profile:
            results.append(("Risk-Based Test Targeter", f"PASS (risk: {profile.risk_level.value})"))
        else:
            results.append(("Risk-Based Test Targeter", "FAIL: No profile"))
    except Exception as e:
        results.append(("Risk-Based Test Targeter", f"FAIL: {str(e)[:50]}"))
    
    # Test 6: Test Complexity Prioritizer
    try:
        from test_complexity_prioritizer import TestComplexityPrioritizer
        prioritizer = TestComplexityPrioritizer()
        # Just test initialization
        results.append(("Test Complexity Prioritizer", "PASS"))
    except Exception as e:
        results.append(("Test Complexity Prioritizer", f"FAIL: {str(e)[:50]}"))
    
    # Test 7: Test Dependency Orderer
    try:
        from test_dependency_orderer import TestDependencyOrderer
        orderer = TestDependencyOrderer()
        # Just test initialization
        results.append(("Test Dependency Orderer", "PASS"))
    except Exception as e:
        results.append(("Test Dependency Orderer", f"FAIL: {str(e)[:50]}"))
    
    # Test 8: Documentation CLI
    try:
        from documentation_cli import DocumentationCLI
        cli = DocumentationCLI()
        # Just test initialization
        results.append(("Documentation CLI", "PASS"))
    except Exception as e:
        results.append(("Documentation CLI", f"FAIL: {str(e)[:50]}"))
    
    return results

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING NEW IMPLEMENTATIONS")
    print("=" * 60)
    print()
    
    results = test_implementations()
    
    passed = 0
    failed = 0
    
    for name, status in results:
        print(f"{name:35} {status}")
        if status.startswith("PASS"):
            passed += 1
        else:
            failed += 1
    
    print()
    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())