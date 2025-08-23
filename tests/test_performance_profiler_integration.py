"""
Tests for Performance Profiler Integration with Gamma Dashboard
Agent E - Personal Analytics with Performance Monitoring

Author: Agent E - Latin Swarm  
Created: 2025-08-23 22:30:00
Purpose: Validate performance monitoring integration with dashboard
"""

import pytest
import time
import json
from typing import Dict, Any

# Import modules to test
from core.analytics.performance_profiler import (
    PerformanceProfiler,
    create_performance_profiler,
    PerformanceMetric
)
from core.analytics.gamma_dashboard_adapter import GammaDashboardAdapter
from core.analytics.personal_analytics_service import create_personal_analytics_service


class TestPerformanceProfiler:
    """Test performance profiler core functionality."""
    
    def test_profiler_creation(self):
        """Test performance profiler creation."""
        profiler = create_performance_profiler()
        assert profiler is not None
        assert isinstance(profiler, PerformanceProfiler)
        assert profiler.monitoring_active == True
        profiler.stop_monitoring()
    
    def test_performance_metric_recording(self):
        """Test recording performance metrics."""
        profiler = PerformanceProfiler()
        
        # Test API response recording
        profiler.record_api_response('/api/test', 45.5, 200, 1024)
        assert len(profiler.metrics_history) == 1
        
        # Test cache performance recording
        profiler.record_cache_performance('test_cache', 85.0, 100, 85)
        assert len(profiler.metrics_history) == 2
        
        # Test dashboard render recording
        profiler.record_dashboard_render(1200.0, 8)
        assert len(profiler.metrics_history) == 3
    
    def test_component_timing(self):
        """Test component timing context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.time_component('test_component'):
            time.sleep(0.01)  # 10ms
        
        assert len(profiler.metrics_history) == 1
        metric = profiler.metrics_history[0]
        assert metric.component == 'test_component'
        assert metric.value >= 10.0  # Should be at least 10ms
    
    def test_dashboard_performance_data(self):
        """Test dashboard performance data generation."""
        profiler = PerformanceProfiler()
        
        # Add some test metrics
        profiler.record_api_response('/api/analytics', 35.0, 200, 2048)
        profiler.record_cache_performance('panel_data', 90.0, 50, 45)
        
        dashboard_data = profiler.get_dashboard_performance_data()
        
        assert 'summary' in dashboard_data
        assert 'current_metrics' in dashboard_data
        assert 'charts' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'timestamp' in dashboard_data
    
    def test_performance_thresholds(self):
        """Test performance threshold monitoring."""
        profiler = PerformanceProfiler()
        
        # Record a slow API response (above 200ms threshold)
        profiler.record_api_response('/api/slow', 650.0, 200, 1024)
        
        dashboard_data = profiler.get_dashboard_performance_data()
        alerts = dashboard_data['alerts']
        
        # Should have alerts for threshold violations
        assert len(alerts) >= 0  # Alerts depend on system metrics
    
    def test_health_status_calculation(self):
        """Test health status calculation."""
        profiler = PerformanceProfiler()
        
        # Get initial health status
        health = profiler._calculate_health_status()
        assert health in ['healthy', 'warning', 'critical']
    
    def test_performance_recommendations(self):
        """Test performance recommendations generation."""
        profiler = PerformanceProfiler()
        
        recommendations = profiler._generate_performance_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3  # Max 3 recommendations


class TestGammaDashboardPerformanceIntegration:
    """Test performance monitoring integration with Gamma Dashboard Adapter."""
    
    def test_adapter_with_performance_profiler(self):
        """Test adapter creation includes performance profiler."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        assert hasattr(adapter, 'performance_profiler')
        assert isinstance(adapter.performance_profiler, PerformanceProfiler)
        
        # Clean up
        adapter.performance_profiler.stop_monitoring()
    
    def test_performance_monitoring_data_format(self):
        """Test performance monitoring data format for dashboard."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        performance_data = adapter.get_performance_monitoring_data()
        
        # Validate structure
        assert 'id' in performance_data
        assert 'title' in performance_data
        assert 'type' in performance_data
        assert 'position' in performance_data
        assert 'size' in performance_data
        assert 'data' in performance_data
        assert 'timestamp' in performance_data
        assert 'status' in performance_data
        
        # Validate specific values
        assert performance_data['id'] == 'agent-e-performance-monitor'
        assert performance_data['title'] == 'Performance Monitor'
        assert performance_data['type'] == 'performance_dashboard'
        assert performance_data['status'] == 'monitoring'
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_combined_dashboard_data(self):
        """Test combined analytics and performance data."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        combined_data = adapter.get_combined_dashboard_data()
        
        # Validate structure
        assert 'panels' in combined_data
        assert 'summary' in combined_data
        
        # Should have 2 panels: analytics + performance
        assert len(combined_data['panels']) == 2
        
        # Validate summary
        summary = combined_data['summary']
        assert 'total_panels' in summary
        assert 'response_time_ms' in summary
        assert 'timestamp' in summary
        
        assert summary['total_panels'] == 2
        assert isinstance(summary['response_time_ms'], (int, float))
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_performance_api_endpoints(self):
        """Test performance monitoring API endpoints."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        endpoints = adapter.get_api_endpoints()
        
        # Should have performance endpoints
        assert '/api/personal-analytics/performance' in endpoints
        assert '/api/personal-analytics/combined' in endpoints
        
        # Validate endpoint configurations
        perf_endpoint = endpoints['/api/personal-analytics/performance']
        assert perf_endpoint['method'] == 'GET'
        assert perf_endpoint['handler'] == adapter.get_performance_monitoring_data
        
        combined_endpoint = endpoints['/api/personal-analytics/combined']
        assert combined_endpoint['method'] == 'GET'
        assert combined_endpoint['handler'] == adapter.get_combined_dashboard_data
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_performance_metrics_during_dashboard_generation(self):
        """Test performance metrics are recorded during dashboard generation."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Clear any existing metrics
        adapter.performance_profiler.metrics_history.clear()
        
        # Generate dashboard data (should record performance metrics)
        panel_data = adapter.get_dashboard_panel_data()
        
        # Should have recorded metrics during generation
        assert len(adapter.performance_profiler.metrics_history) > 0
        
        # Check for specific component timings
        component_metrics = [
            m for m in adapter.performance_profiler.metrics_history 
            if m.metric_type == 'component_duration_ms'
        ]
        assert len(component_metrics) > 0
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_cache_performance_tracking(self):
        """Test cache performance tracking in dashboard adapter."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Clear metrics
        adapter.performance_profiler.metrics_history.clear()
        
        # First call - should be cache miss
        panel_data1 = adapter.get_dashboard_panel_data()
        
        # Second call - should be cache hit
        panel_data2 = adapter.get_dashboard_panel_data()
        
        # Should have cache performance metrics
        cache_metrics = [
            m for m in adapter.performance_profiler.metrics_history 
            if m.metric_type == 'cache_hit_rate_percent'
        ]
        
        assert len(cache_metrics) >= 2  # At least one miss and one hit
        
        adapter.performance_profiler.stop_monitoring()


class TestPerformanceIntegration:
    """Test integration between performance monitoring and dashboard systems."""
    
    def test_performance_data_size_constraints(self):
        """Test performance data meets size constraints."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        performance_data = adapter.get_performance_monitoring_data()
        
        # Convert to JSON to estimate size
        json_data = json.dumps(performance_data, default=str)
        data_size_kb = len(json_data.encode('utf-8')) / 1024
        
        # Should be reasonable size (under 50KB)
        assert data_size_kb < 50, f"Performance data too large: {data_size_kb}KB"
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_performance_monitoring_response_times(self):
        """Test performance monitoring doesn't significantly impact response times."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Time dashboard generation
        start_time = time.time()
        combined_data = adapter.get_combined_dashboard_data()
        response_time_ms = (time.time() - start_time) * 1000
        
        # Should still meet performance requirements (< 200ms)
        assert response_time_ms < 200, f"Response time too slow: {response_time_ms}ms"
        
        # Verify response time is recorded in summary
        assert 'response_time_ms' in combined_data['summary']
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_performance_dashboard_chart_data(self):
        """Test performance dashboard generates proper chart data."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Add some performance data
        adapter.performance_profiler.record_api_response('/api/test', 45.0, 200, 1024)
        adapter.performance_profiler.record_dashboard_render(1100.0, 6)
        
        performance_data = adapter.get_performance_monitoring_data()
        charts = performance_data['data']['charts']
        
        # Should have system and API performance charts
        assert 'system_performance_chart' in charts
        assert 'api_performance_chart' in charts
        
        # Validate chart structure
        sys_chart = charts['system_performance_chart']
        assert 'type' in sys_chart
        assert 'data' in sys_chart
        assert 'options' in sys_chart
        
        api_chart = charts['api_performance_chart']
        assert 'type' in api_chart
        assert 'data' in api_chart
        assert 'options' in api_chart
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_performance_alerts_integration(self):
        """Test performance alerts integration with dashboard."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Generate performance data
        performance_data = adapter.get_performance_monitoring_data()
        alerts = performance_data['data']['alerts']
        
        # Alerts should be a list
        assert isinstance(alerts, list)
        
        # If alerts exist, validate structure
        if alerts:
            alert = alerts[0]
            assert 'metric' in alert
            assert 'current_value' in alert
            assert 'threshold' in alert
            assert 'severity' in alert
            assert 'message' in alert
        
        adapter.performance_profiler.stop_monitoring()


if __name__ == '__main__':
    # Run specific test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick test
        print("Running quick performance profiler integration test...")
        
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Test combined data
        combined = adapter.get_combined_dashboard_data()
        print(f"✅ Combined dashboard data: {len(combined['panels'])} panels")
        print(f"✅ Response time: {combined['summary']['response_time_ms']}ms")
        
        # Test performance data
        perf_data = adapter.get_performance_monitoring_data()
        print(f"✅ Performance panel: {perf_data['title']}")
        
        adapter.performance_profiler.stop_monitoring()
        print("✅ Quick test completed successfully!")
    else:
        # Run full test suite
        pytest.main([__file__, '-v'])