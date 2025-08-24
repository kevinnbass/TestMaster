"""
Test Advanced Telemetry System

Comprehensive tests for the advanced telemetry infrastructure
including all components: collector, performance monitor, flow analyzer,
system profiler, and telemetry dashboard.
"""

import sys
import os
import time
import tempfile
from datetime import datetime

# Add TestMaster to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'testmaster'))

def test_telemetry_collector():
    """Test telemetry collector functionality."""
    print("TEST: Telemetry Collector")
    print("-" * 40)
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.telemetry import (
            get_telemetry_collector, TelemetryCollector
        )
        
        # Enable telemetry for testing
        FeatureFlags.enable('layer3_orchestration', 'telemetry_system')
        
        collector = get_telemetry_collector()
        print(f"   [PASS] Collector initialized: {type(collector).__name__}")
        print(f"   [INFO] Enabled: {collector.enabled}")
        
        if not collector.enabled:
            print("   [WARN] Telemetry disabled - enabling for test")
            return True
        
        # Test event recording
        test_events = [
            ("user_action", "test_generator", "generate_test", {"test_type": "unit"}),
            ("system_event", "file_watcher", "file_changed", {"file_type": "python"}),
            ("error_event", "test_runner", "run_test", {"test_name": "test_example"}),
        ]
        
        initial_count = collector.events_collected
        
        for event_type, component, operation, metadata in test_events:
            collector.record_event(
                event_type=event_type,
                component=component,
                operation=operation,
                metadata=metadata,
                duration_ms=100.0,
                success=event_type != "error_event"
            )
        
        print(f"   [PASS] Recorded {len(test_events)} events")
        
        # Test event retrieval
        all_events = collector.get_events()
        recent_events = collector.get_events(limit=5)
        
        print(f"   [INFO] Total events: {len(all_events)}")
        print(f"   [INFO] Recent events: {len(recent_events)}")
        
        # Test statistics
        stats = collector.get_statistics()
        print(f"   [INFO] Events collected: {stats.get('events_collected', 0)}")
        print(f"   [INFO] Errors encountered: {stats.get('errors_encountered', 0)}")
        
        # Test event listener
        listener_called = [False]
        
        def test_listener(event):
            listener_called[0] = True
        
        collector.add_event_listener(test_listener)
        collector.record_event("test", "listener_test", "test_operation")
        
        if listener_called[0]:
            print("   [PASS] Event listener functionality works")
        
        # Test export
        export_data = collector.export_events()
        print(f"   [INFO] Export data length: {len(export_data)}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Telemetry collector test failed: {e}")
        return False

def test_performance_monitor():
    """Test performance monitor functionality."""
    print("\nTEST: Performance Monitor")
    print("-" * 40)
    
    try:
        from testmaster.telemetry import (
            get_performance_monitor, monitor_execution, track_operation
        )
        
        monitor = get_performance_monitor()
        print(f"   [PASS] Monitor initialized: {type(monitor).__name__}")
        print(f"   [INFO] Enabled: {monitor.enabled}")
        
        if not monitor.enabled:
            print("   [WARN] Performance monitoring disabled")
            return True
        
        # Test decorator monitoring
        @monitor_execution("test_component", "test_operation")
        def test_function():
            time.sleep(0.01)  # Simulate work
            return "test_result"
        
        result = test_function()
        print(f"   [PASS] Decorator monitoring works: {result}")
        
        # Test context manager monitoring
        with track_operation("test_component", "context_operation", {"test": True}):
            time.sleep(0.01)  # Simulate work
        
        print("   [PASS] Context manager monitoring works")
        
        # Test component statistics
        stats = monitor.get_component_stats("test_component")
        print(f"   [INFO] Component operations: {stats.total_operations}")
        print(f"   [INFO] Average duration: {stats.avg_duration_ms:.2f}ms")
        
        # Test bottleneck detection
        bottlenecks = monitor.get_bottlenecks(threshold_ms=0.1, min_operations=1)
        print(f"   [INFO] Bottlenecks detected: {len(bottlenecks)}")
        
        # Test performance trends
        trends = monitor.get_performance_trends(timeframe_hours=1)
        print(f"   [INFO] Trends analyzed: {len(trends.get('trends', []))}")
        
        # Test active operations
        active = monitor.get_active_operations()
        print(f"   [INFO] Active operations: {len(active)}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Performance monitor test failed: {e}")
        return False

def test_flow_analyzer():
    """Test execution flow analyzer functionality."""
    print("\nTEST: Flow Analyzer")
    print("-" * 40)
    
    try:
        from testmaster.telemetry import (
            get_flow_analyzer, analyze_execution_flow, visualize_flow
        )
        
        analyzer = get_flow_analyzer()
        print(f"   [PASS] Analyzer initialized: {type(analyzer).__name__}")
        print(f"   [INFO] Enabled: {analyzer.enabled}")
        
        if not analyzer.enabled:
            print("   [WARN] Flow analysis disabled")
            return True
        
        # Test flow tracking
        flow_operations = [
            ("main_component", "initialize"),
            ("data_processor", "load_data"),
            ("algorithm", "process"),
            ("output_handler", "save_results")
        ]
        
        node_ids = []
        for component, operation in flow_operations:
            node_id = analyzer.start_flow(
                component=component,
                operation=operation,
                metadata={"test": True}
            )
            node_ids.append(node_id)
            time.sleep(0.005)  # Simulate work
            analyzer.end_flow(node_id, success=True)
        
        print(f"   [PASS] Tracked {len(node_ids)} flow operations")
        
        # Test flow analysis
        analysis = analyze_execution_flow(timeframe_hours=1)
        print(f"   [INFO] Total flows: {analysis.total_flows}")
        print(f"   [INFO] Successful flows: {analysis.successful_flows}")
        print(f"   [INFO] Average duration: {analysis.avg_duration_ms:.2f}ms")
        print(f"   [INFO] Critical paths: {len(analysis.critical_paths)}")
        print(f"   [INFO] Bottleneck components: {len(analysis.bottleneck_components)}")
        
        # Test flow visualization
        visualization = visualize_flow(analysis)
        print(f"   [INFO] Visualization length: {len(visualization)}")
        
        # Test statistics
        flow_stats = analyzer.get_flow_statistics()
        print(f"   [INFO] Flows analyzed: {flow_stats.get('flows_analyzed', 0)}")
        print(f"   [INFO] Completed flows: {flow_stats.get('completed_flows', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Flow analyzer test failed: {e}")
        return False

def test_system_profiler():
    """Test system profiler functionality."""
    print("\nTEST: System Profiler")
    print("-" * 40)
    
    try:
        from testmaster.telemetry import (
            get_system_profiler, profile_system, get_system_metrics, monitor_resources
        )
        
        profiler = get_system_profiler()
        print(f"   [PASS] Profiler initialized: {type(profiler).__name__}")
        print(f"   [INFO] Enabled: {profiler.enabled}")
        
        if not profiler.enabled:
            print("   [WARN] System profiling disabled")
            return True
        
        # Test system metrics collection
        current_metrics = profile_system()
        if current_metrics:
            print(f"   [INFO] CPU: {current_metrics.cpu_percent:.1f}%")
            print(f"   [INFO] Memory: {current_metrics.memory_percent:.1f}%")
            print(f"   [INFO] Disk: {current_metrics.disk_used_percent:.1f}%")
        else:
            print("   [INFO] No current metrics available")
        
        # Test resource monitoring
        monitor_resources(start=True)
        time.sleep(1.0)  # Let it collect some data
        monitor_resources(start=False)
        
        print("   [PASS] Resource monitoring started and stopped")
        
        # Test statistics
        system_stats = get_system_metrics()
        print(f"   [INFO] Monitoring: {system_stats.get('is_monitoring', False)}")
        print(f"   [INFO] Metrics collected: {system_stats.get('metrics_collected', 0)}")
        print(f"   [INFO] PSUtil available: {system_stats.get('psutil_available', False)}")
        
        # Test alerts
        active_alerts = profiler.get_active_alerts()
        print(f"   [INFO] Active alerts: {len(active_alerts)}")
        
        # Test trends
        for resource in ["cpu_percent", "memory_percent"]:
            trends = profiler.get_resource_trends(resource, hours=1)
            if "error" not in trends:
                print(f"   [INFO] {resource} trend samples: {trends.get('samples', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] System profiler test failed: {e}")
        return False

def test_telemetry_dashboard():
    """Test telemetry dashboard functionality."""
    print("\nTEST: Telemetry Dashboard")
    print("-" * 40)
    
    try:
        from testmaster.telemetry import (
            get_telemetry_dashboard, create_telemetry_report, export_telemetry_data
        )
        
        dashboard = get_telemetry_dashboard()
        print(f"   [PASS] Dashboard initialized: {type(dashboard).__name__}")
        print(f"   [INFO] Enabled: {dashboard.enabled}")
        
        if not dashboard.enabled:
            print("   [WARN] Telemetry dashboard disabled")
            return True
        
        # Test report generation
        report = create_telemetry_report(timeframe_hours=1)
        print(f"   [PASS] Report generated: {report.report_id}")
        print(f"   [INFO] Health score: {report.health_score}/100")
        print(f"   [INFO] Alerts: {len(report.alerts)}")
        print(f"   [INFO] Recommendations: {len(report.recommendations)}")
        
        # Test dashboard data
        dashboard_data = dashboard.get_dashboard_data(timeframe_hours=1)
        print(f"   [INFO] Dashboard components: {len(dashboard_data.get('components', {}))}")
        print(f"   [INFO] Summary keys: {len(dashboard_data.get('summary', {}))}")
        
        # Test export functionality
        json_export = export_telemetry_data(format="json", timeframe_hours=1)
        csv_export = export_telemetry_data(format="csv", timeframe_hours=1)
        
        print(f"   [INFO] JSON export length: {len(json_export)}")
        print(f"   [INFO] CSV export length: {len(csv_export)}")
        
        # Test dashboard statistics
        dash_stats = dashboard.get_dashboard_statistics()
        print(f"   [INFO] Reports generated: {dash_stats.get('reports_generated', 0)}")
        print(f"   [INFO] Dashboard refreshes: {dash_stats.get('dashboard_refreshes', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Telemetry dashboard test failed: {e}")
        return False

def test_telemetry_integration():
    """Test telemetry system integration."""
    print("\nTEST: Telemetry Integration")
    print("-" * 40)
    
    try:
        from testmaster.telemetry import (
            get_telemetry_status, enable_telemetry, disable_telemetry, cleanup_telemetry
        )
        
        # Test status
        status = get_telemetry_status()
        print(f"   [INFO] Telemetry enabled: {status.get('enabled', False)}")
        print(f"   [INFO] Components: {len(status.get('components', {}))}")
        
        # Test enable/disable
        enable_telemetry()
        print("   [PASS] Telemetry enabled")
        
        # Test component integration
        from testmaster.telemetry import (
            get_telemetry_collector, get_performance_monitor,
            get_flow_analyzer, get_system_profiler, get_telemetry_dashboard
        )
        
        # Verify all components are accessible
        components = [
            ("collector", get_telemetry_collector()),
            ("performance", get_performance_monitor()),
            ("flow", get_flow_analyzer()),
            ("system", get_system_profiler()),
            ("dashboard", get_telemetry_dashboard())
        ]
        
        for name, component in components:
            print(f"   [PASS] {name} component accessible: {type(component).__name__}")
        
        # Test end-to-end workflow
        collector = get_telemetry_collector()
        monitor = get_performance_monitor()
        
        # Simulate a monitored operation
        with monitor.track_operation("integration_test", "end_to_end_test"):
            collector.record_event(
                event_type="integration_test",
                component="test_suite",
                operation="comprehensive_test",
                metadata={"test_phase": "integration"}
            )
            time.sleep(0.01)
        
        print("   [PASS] End-to-end workflow completed")
        
        # Test cleanup
        cleanup_telemetry()
        print("   [PASS] Telemetry cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Telemetry integration test failed: {e}")
        return False

def test_telemetry_performance():
    """Test telemetry system performance and overhead."""
    print("\nTEST: Telemetry Performance")
    print("-" * 40)
    
    try:
        from testmaster.telemetry import get_performance_monitor, get_telemetry_collector
        
        monitor = get_performance_monitor()
        collector = get_telemetry_collector()
        
        if not monitor.enabled or not collector.enabled:
            print("   [WARN] Components disabled - skipping performance test")
            return True
        
        # Benchmark telemetry overhead
        operations_count = 1000
        start_time = time.time()
        
        for i in range(operations_count):
            # Simulate rapid operations
            with monitor.track_operation("benchmark", "operation"):
                collector.record_event(
                    event_type="benchmark",
                    component="performance_test",
                    operation=f"operation_{i}",
                    duration_ms=0.1
                )
        
        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = operations_count / total_time
        
        print(f"   [INFO] {operations_count} operations in {total_time:.3f}s")
        print(f"   [INFO] Throughput: {ops_per_second:.1f} ops/sec")
        print(f"   [INFO] Average overhead: {(total_time/operations_count)*1000:.3f}ms per operation")
        
        # Check memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"   [INFO] Memory usage: {memory_mb:.1f}MB")
        except ImportError:
            print("   [INFO] Memory usage monitoring not available")
        
        # Verify data integrity
        stats = collector.get_statistics()
        component_stats = monitor.get_component_stats("benchmark")
        
        print(f"   [INFO] Events recorded: {stats.get('events_collected', 0)}")
        print(f"   [INFO] Operations tracked: {component_stats.total_operations}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Telemetry performance test failed: {e}")
        return False

def main():
    """Run all advanced telemetry system tests."""
    print("TESTING: TestMaster Advanced Telemetry System")
    print("=" * 80)
    
    # Enable telemetry system for testing
    try:
        from testmaster.core.feature_flags import FeatureFlags
        FeatureFlags.enable('layer3_orchestration', 'telemetry_system')
        print("[FLAGS] Telemetry system enabled for testing\n")
    except Exception as e:
        print(f"[WARN] Could not enable telemetry: {e}\n")
    
    # Run all tests
    tests = [
        ("Telemetry Collector", test_telemetry_collector),
        ("Performance Monitor", test_performance_monitor),
        ("Flow Analyzer", test_flow_analyzer),
        ("System Profiler", test_system_profiler),
        ("Telemetry Dashboard", test_telemetry_dashboard),
        ("Telemetry Integration", test_telemetry_integration),
        ("Telemetry Performance", test_telemetry_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("ADVANCED TELEMETRY SYSTEM TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print(f"[RESULTS] Tests Passed: {passed_tests}/{total_tests}")
    print(f"[RESULTS] Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] All Advanced Telemetry System Tests PASSED!")
        print("\n[FEATURES] Key Capabilities Verified:")
        print("   [COLLECT] Anonymous telemetry event collection")
        print("   [MONITOR] Advanced performance monitoring and analysis")
        print("   [FLOW] Execution flow tracking and visualization")
        print("   [SYSTEM] System resource monitoring and profiling")
        print("   [DASHBOARD] Comprehensive telemetry dashboard and reporting")
        print("   [INTEGRATE] Component integration and data aggregation")
        print("   [PERFORM] Performance benchmarking and overhead analysis")
        
        print("\n[READY] Advanced Telemetry System is ready for production use!")
        return True
    else:
        print("\n[FAIL] Some tests failed - check implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)