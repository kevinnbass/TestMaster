#!/usr/bin/env python3
"""
Simple test script for our new implementations.
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_realtime_metrics_collector():
    """Test the realtime metrics collector."""
    try:
        from realtime_metrics_collector import RealtimeMetricsCollector
        collector = RealtimeMetricsCollector(collection_interval_ms=100)
        collector.start_collection()
        
        # Add some test metrics
        collector.record_metric('test.metric', 42.0)
        collector.record_metric('test.metric', 43.0)
        collector.record_metric('test.metric', 44.0)
        
        time.sleep(0.5)
        collector.stop_collection()
        
        # Check metrics
        metrics = collector.get_metrics('test.metric')
        if metrics and len(metrics) >= 3:
            print("✅ Realtime Metrics Collector: PASSED")
            return True
        else:
            print("❌ Realtime Metrics Collector: FAILED - No metrics collected")
            return False
    except Exception as e:
        print(f"❌ Realtime Metrics Collector: FAILED - {e}")
        return False

def test_performance_profiler():
    """Test the performance profiler."""
    try:
        from performance_profiler import PerformanceProfiler
        profiler = PerformanceProfiler()
        
        # Start profiling
        session_id = profiler.start_profiling()
        
        # Do some work
        def test_function():
            result = 0
            for i in range(1000):
                result += i * i
            return result
        
        test_function()
        
        # Stop profiling
        profile_data = profiler.stop_profiling(session_id)
        
        if profile_data and profile_data.get('duration', 0) > 0:
            print("✅ Performance Profiler: PASSED")
            return True
        else:
            print("❌ Performance Profiler: FAILED - No profile data")
            return False
    except Exception as e:
        print(f"❌ Performance Profiler: FAILED - {e}")
        return False

def test_live_code_quality_monitor():
    """Test the live code quality monitor."""
    try:
        from live_code_quality_monitor import LiveCodeQualityMonitor
        monitor = LiveCodeQualityMonitor()
        
        # Analyze this file
        result = monitor.analyze_file_quality(__file__)
        
        if result and result.overall_score >= 0:
            print(f"✅ Live Code Quality Monitor: PASSED (score: {result.overall_score:.1f})")
            return True
        else:
            print("❌ Live Code Quality Monitor: FAILED - No analysis result")
            return False
    except Exception as e:
        print(f"❌ Live Code Quality Monitor: FAILED - {e}")
        return False

def test_enhanced_incremental_ast_engine():
    """Test the enhanced incremental AST engine."""
    try:
        from enhanced_incremental_ast_engine import EnhancedIncrementalASTEngine
        engine = EnhancedIncrementalASTEngine()
        
        # Test code
        test_code = '''
def hello(name):
    return f"Hello, {name}!"

def goodbye(name):
    return f"Goodbye, {name}!"
'''
        
        # Create a temp file
        temp_file = Path('temp_test.py')
        temp_file.write_text(test_code)
        
        try:
            # Analyze
            result = engine.analyze_incremental(str(temp_file))
            
            if result and result.semantic_changes:
                print("✅ Enhanced Incremental AST Engine: PASSED")
                return True
            else:
                print("❌ Enhanced Incremental AST Engine: FAILED - No semantic changes detected")
                return False
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
                
    except Exception as e:
        print(f"❌ Enhanced Incremental AST Engine: FAILED - {e}")
        return False

def test_risk_based_test_targeter():
    """Test the risk-based test targeter."""
    try:
        from risk_based_test_targeter import RiskBasedTestTargeter
        targeter = RiskBasedTestTargeter()
        
        # Analyze risk for this file
        profile = targeter.analyze_risk(__file__, include_history=False)
        
        if profile and profile.risk_score >= 0:
            print(f"✅ Risk-Based Test Targeter: PASSED (risk: {profile.risk_level.value})")
            return True
        else:
            print("❌ Risk-Based Test Targeter: FAILED - No risk profile")
            return False
    except Exception as e:
        print(f"❌ Risk-Based Test Targeter: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING OUR NEW IMPLEMENTATIONS")
    print("=" * 60)
    
    tests = [
        test_realtime_metrics_collector,
        test_performance_profiler,
        test_live_code_quality_monitor,
        test_enhanced_incremental_ast_engine,
        test_risk_based_test_targeter
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️ {failed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())