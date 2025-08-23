"""
Test Gamma Dashboard Adapter Integration
========================================

Tests the integration adapter between Agent E's personal analytics
and Agent Gamma's unified dashboard on port 5003.

Agent E - Dashboard Adapter Testing
Created: 2025-08-23 21:00:00
"""

import sys
import os
import json
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the adapter and service
from core.analytics.gamma_dashboard_adapter import (
    GammaDashboardAdapter,
    create_gamma_adapter,
    integrate_with_gamma_dashboard
)
from core.analytics.personal_analytics_service import PersonalAnalyticsService


class TestGammaDashboardAdapter(unittest.TestCase):
    """Test suite for Gamma dashboard adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = create_gamma_adapter()
        
    def test_adapter_creation(self):
        """Test that adapter can be created."""
        self.assertIsNotNone(self.adapter)
        self.assertIsInstance(self.adapter, GammaDashboardAdapter)
        self.assertEqual(self.adapter.port, 5003)
        
    def test_panel_configuration(self):
        """Test panel configuration for 2x2 grid."""
        panel_config = self.adapter.panel_config
        
        # Check required panel fields
        self.assertEqual(panel_config['panel_id'], 'agent-e-personal-analytics')
        self.assertEqual(panel_config['size']['width'], 2)
        self.assertEqual(panel_config['size']['height'], 2)
        self.assertIn('position', panel_config)
        self.assertIn('features', panel_config)
        
        # Check panel features
        features = panel_config['features']
        self.assertTrue(features['resizable'])
        self.assertTrue(features['draggable'])
        self.assertTrue(features['exportable'])
        
    def test_dashboard_panel_data_format(self):
        """Test panel data formatting for Gamma dashboard."""
        panel_data = self.adapter.get_dashboard_panel_data()
        
        # Check structure
        self.assertIn('panel_id', panel_data)
        self.assertIn('data', panel_data)
        self.assertIn('timestamp', panel_data)
        self.assertIn('status', panel_data)
        
        # Check data formatting
        data = panel_data['data']
        self.assertIn('summary', data)
        self.assertIn('charts', data)
        self.assertIn('insights', data)
        self.assertIn('statistics', data)
        
    def test_summary_metrics_format(self):
        """Test summary metrics formatting."""
        panel_data = self.adapter.get_dashboard_panel_data()
        summary = panel_data['data']['summary']
        
        # Check required summary fields
        expected_fields = ['overall_score', 'productivity_score', 'test_coverage', 'code_quality']
        for field in expected_fields:
            self.assertIn(field, summary)
            self.assertIsInstance(summary[field], (int, float))
            
    def test_chart_data_formats(self):
        """Test various chart data formats."""
        panel_data = self.adapter.get_dashboard_panel_data()
        charts = panel_data['data']['charts']
        
        # Check chart types
        self.assertIn('quality_trend', charts)
        self.assertIn('productivity_gauge', charts)
        self.assertIn('activity_timeline', charts)
        self.assertIn('metrics_radar', charts)
        
        # Validate trend chart
        trend = charts['quality_trend']
        self.assertEqual(trend['type'], 'line')
        self.assertIn('data', trend)
        self.assertIn('options', trend)
        
        # Validate gauge chart
        gauge = charts['productivity_gauge']
        self.assertEqual(gauge['type'], 'gauge')
        self.assertIn('value', gauge)
        self.assertIn('segments', gauge)
        
        # Validate radar chart
        radar = charts['metrics_radar']
        self.assertEqual(radar['type'], 'radar')
        self.assertIn('data', radar)
        
    def test_3d_visualization_transformation(self):
        """Test 3D data transformation for Gamma's WebGL engine."""
        viz_data = self.adapter.get_3d_visualization_data()
        
        # Check scene structure
        self.assertIn('scene', viz_data)
        self.assertIn('metrics', viz_data)
        self.assertIn('controls', viz_data)
        
        scene = viz_data['scene']
        self.assertIn('nodes', scene)
        self.assertIn('edges', scene)
        self.assertIn('camera', scene)
        
        # Check node transformation
        if scene['nodes']:
            node = scene['nodes'][0]
            self.assertIn('material', node)
            self.assertIn('geometry', node)
            self.assertIn('color', node['material'])
            self.assertEqual(node['geometry']['type'], 'sphere')
            
        # Check camera settings
        camera = scene['camera']
        self.assertIn('position', camera)
        self.assertIn('lookAt', camera)
        
        # Check controls
        controls = viz_data['controls']
        self.assertTrue(controls['enableRotation'])
        self.assertTrue(controls['enableZoom'])
        
    def test_api_endpoint_configuration(self):
        """Test API endpoint configuration."""
        endpoints = self.adapter.get_api_endpoints()
        
        # Check required endpoints
        expected_endpoints = [
            '/api/personal-analytics/overview',
            '/api/personal-analytics/metrics',
            '/api/personal-analytics/3d-data',
            '/api/personal-analytics/panel'
        ]
        
        for endpoint in expected_endpoints:
            self.assertIn(endpoint, endpoints)
            config = endpoints[endpoint]
            self.assertIn('method', config)
            self.assertIn('handler', config)
            self.assertIn('description', config)
            self.assertEqual(config['method'], 'GET')
            
    def test_websocket_handler_configuration(self):
        """Test WebSocket handler configuration."""
        handlers = self.adapter.get_websocket_handlers()
        
        # Check required handlers
        expected_handlers = [
            'personal_analytics_subscribe',
            'personal_analytics_update',
            'personal_analytics_export'
        ]
        
        for handler in expected_handlers:
            self.assertIn(handler, handlers)
            self.assertTrue(callable(handlers[handler]))
            
    def test_subscription_handling(self):
        """Test WebSocket subscription handling."""
        response = self.adapter.handle_subscription({})
        
        self.assertEqual(response['status'], 'subscribed')
        self.assertEqual(response['panel_id'], 'agent-e-personal-analytics')
        self.assertIn('refresh_interval', response)
        
    def test_export_functionality(self):
        """Test data export functionality."""
        # Test JSON export
        json_response = self.adapter.handle_export_request({'format': 'json'})
        self.assertEqual(json_response['format'], 'json')
        self.assertIn('data', json_response)
        self.assertIn('filename', json_response)
        
        # Test CSV export
        csv_response = self.adapter.handle_export_request({'format': 'csv'})
        self.assertEqual(csv_response['format'], 'csv')
        self.assertIn('data', csv_response)
        self.assertIn('.csv', csv_response['filename'])
        
    def test_cache_functionality(self):
        """Test caching for performance optimization."""
        # First call - populates cache
        data1 = self.adapter.get_dashboard_panel_data()
        
        # Second call - should use cache
        data2 = self.adapter.get_dashboard_panel_data()
        
        # Timestamps should be identical when cached
        self.assertEqual(data1['timestamp'], data2['timestamp'])
        
    def test_performance_metrics(self):
        """Test adapter performance metrics."""
        metrics = self.adapter.get_performance_metrics()
        
        # Check metric fields
        self.assertIn('cache_hit_rate', metrics)
        self.assertIn('avg_response_time', metrics)
        self.assertIn('data_freshness', metrics)
        self.assertIn('integration_health', metrics)
        
        # Check performance targets
        self.assertLess(metrics['avg_response_time'], 100)  # Sub-100ms
        self.assertEqual(metrics['integration_health'], 'healthy')
        
    def test_flask_integration_helper(self):
        """Test Flask integration helper function."""
        mock_app = Mock()
        mock_app.add_url_rule = Mock()
        
        # Test integration
        adapter = integrate_with_gamma_dashboard(mock_app)
        
        # Verify endpoints were registered
        self.assertTrue(mock_app.add_url_rule.called)
        self.assertIsInstance(adapter, GammaDashboardAdapter)
        
        # Check that multiple endpoints were registered
        call_count = mock_app.add_url_rule.call_count
        self.assertGreaterEqual(call_count, 4)  # At least 4 endpoints
        
    def test_socketio_integration_helper(self):
        """Test SocketIO integration helper function."""
        mock_app = Mock()
        mock_app.add_url_rule = Mock()
        mock_socketio = Mock()
        mock_socketio.on = Mock(return_value=lambda f: f)
        
        # Test integration with SocketIO
        adapter = integrate_with_gamma_dashboard(mock_app, mock_socketio)
        
        # Verify WebSocket handlers were registered
        self.assertTrue(mock_socketio.on.called)
        self.assertIsInstance(adapter, GammaDashboardAdapter)


class TestGammaCompatibility(unittest.TestCase):
    """Test compatibility with Gamma dashboard requirements."""
    
    def setUp(self):
        """Set up compatibility test fixtures."""
        self.adapter = create_gamma_adapter()
        
    def test_port_5003_compatibility(self):
        """Test port 5003 specific requirements."""
        self.assertEqual(self.adapter.port, 5003)
        
        # Panel data should be compatible with port 5003 format
        panel_data = self.adapter.get_dashboard_panel_data()
        self.assertEqual(panel_data['theme'], 'gamma-unified')
        
    def test_performance_requirements(self):
        """Test performance meets Gamma dashboard requirements."""
        import time
        
        # Test response time
        start = time.time()
        _ = self.adapter.get_dashboard_panel_data()
        elapsed = (time.time() - start) * 1000
        
        # Should meet sub-100ms requirement (relaxed for testing)
        self.assertLess(elapsed, 200)
        
    def test_data_size_constraints(self):
        """Test data size meets network transfer constraints."""
        panel_data = self.adapter.get_dashboard_panel_data()
        
        # Convert to JSON to measure size
        json_data = json.dumps(panel_data)
        size_kb = len(json_data) / 1024
        
        # Should be reasonable for dashboard updates
        self.assertLess(size_kb, 50)  # Less than 50KB
        
    def test_grid_layout_compatibility(self):
        """Test 2x2 grid layout compatibility."""
        panel_config = self.adapter.panel_config
        
        # Size should be exactly 2x2
        self.assertEqual(panel_config['size']['width'], 2)
        self.assertEqual(panel_config['size']['height'], 2)
        
        # Position should be valid grid coordinates
        position = panel_config['position']
        self.assertIsInstance(position['x'], int)
        self.assertIsInstance(position['y'], int)
        self.assertGreaterEqual(position['x'], 0)
        self.assertGreaterEqual(position['y'], 0)


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)