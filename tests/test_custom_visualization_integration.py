"""
Tests for Custom Visualization Builder Integration with Gamma Dashboard
Agent E - Personal Analytics with Custom Chart Creation

Author: Agent E - Latin Swarm  
Created: 2025-08-23 23:30:00
Purpose: Validate custom visualization builder integration with dashboard
"""

import pytest
import json
import uuid
from typing import Dict, List, Any

# Import modules to test
from core.analytics.custom_visualization_builder import (
    CustomVisualizationBuilder,
    create_custom_visualization_builder,
    ChartType,
    DataSource,
    ChartConfiguration,
    VisualizationPanel
)
from core.analytics.gamma_dashboard_adapter import GammaDashboardAdapter
from core.analytics.personal_analytics_service import create_personal_analytics_service


class TestCustomVisualizationBuilder:
    """Test custom visualization builder core functionality."""
    
    def test_builder_creation(self):
        """Test custom visualization builder creation."""
        builder = create_custom_visualization_builder()
        assert builder is not None
        assert isinstance(builder, CustomVisualizationBuilder)
        assert len(builder.templates) > 0  # Should have default templates
    
    def test_default_templates_loaded(self):
        """Test default templates are properly loaded."""
        builder = CustomVisualizationBuilder()
        
        expected_templates = [
            "productivity_trend", "code_quality_radar", 
            "performance_gauge", "commit_activity_heatmap"
        ]
        
        for template_name in expected_templates:
            assert template_name in builder.templates
            template = builder.templates[template_name]
            assert "title" in template
            assert "chart_type" in template
            assert "data_source" in template
            assert "data_fields" in template
    
    def test_chart_creation(self):
        """Test custom chart creation."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart(
            title="Test Chart",
            chart_type=ChartType.LINE,
            data_source=DataSource.PERSONAL_ANALYTICS,
            data_fields=["commits", "lines_added"]
        )
        
        assert chart_id in builder.charts
        chart = builder.charts[chart_id]
        assert chart.title == "Test Chart"
        assert chart.chart_type == ChartType.LINE
        assert chart.data_source == DataSource.PERSONAL_ANALYTICS
        assert chart.data_fields == ["commits", "lines_added"]
    
    def test_chart_from_template(self):
        """Test creating chart from template."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart_from_template(
            "productivity_trend",
            "My Productivity"
        )
        
        assert chart_id in builder.charts
        chart = builder.charts[chart_id]
        assert chart.title == "My Productivity"
        assert chart.chart_type == ChartType.LINE
    
    def test_chart_data_generation(self):
        """Test chart data generation."""
        builder = CustomVisualizationBuilder()
        
        # Create a test chart
        chart_id = builder.create_chart(
            title="Test Quality Chart",
            chart_type=ChartType.BAR,
            data_source=DataSource.QUALITY_METRICS,
            data_fields=["maintainability", "complexity"]
        )
        
        # Sample data
        analytics_data = {
            "quality_metrics": {
                "maintainability": 85,
                "complexity": 65
            }
        }
        
        chart_config = builder.generate_chart_data(chart_id, analytics_data)
        
        # Validate chart configuration
        assert "type" in chart_config
        assert "data" in chart_config
        assert "options" in chart_config
        assert chart_config["type"] == "bar"
        
        # Validate data structure
        data = chart_config["data"]
        assert "labels" in data
        assert "datasets" in data
        assert len(data["datasets"]) > 0
    
    def test_panel_creation(self):
        """Test visualization panel creation."""
        builder = CustomVisualizationBuilder()
        
        # Create some charts first
        chart1 = builder.create_chart(
            "Chart 1", ChartType.LINE, DataSource.PERSONAL_ANALYTICS, ["commits"]
        )
        chart2 = builder.create_chart(
            "Chart 2", ChartType.BAR, DataSource.QUALITY_METRICS, ["coverage"]
        )
        
        # Create panel
        panel_id = builder.create_panel(
            title="Test Panel",
            description="Test panel with multiple charts",
            chart_ids=[chart1, chart2],
            layout={"type": "grid", "columns": 2}
        )
        
        assert panel_id in builder.panels
        panel = builder.panels[panel_id]
        assert panel.title == "Test Panel"
        assert len(panel.charts) == 2
        assert chart1 in panel.charts
        assert chart2 in panel.charts
    
    def test_chart_update(self):
        """Test chart configuration updates."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart(
            "Original Title", ChartType.LINE, DataSource.PERSONAL_ANALYTICS, ["commits"]
        )
        
        # Update chart
        success = builder.update_chart(
            chart_id,
            title="Updated Title",
            data_fields=["commits", "lines_added"]
        )
        
        assert success
        chart = builder.charts[chart_id]
        assert chart.title == "Updated Title"
        assert chart.data_fields == ["commits", "lines_added"]
    
    def test_chart_deletion(self):
        """Test chart deletion."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart(
            "Delete Me", ChartType.PIE, DataSource.PERSONAL_ANALYTICS, ["commits"]
        )
        
        # Verify chart exists
        assert chart_id in builder.charts
        
        # Delete chart
        success = builder.delete_chart(chart_id)
        
        assert success
        assert chart_id not in builder.charts
    
    def test_export_import_configuration(self):
        """Test configuration export and import."""
        builder = CustomVisualizationBuilder()
        
        # Create test data
        chart_id = builder.create_chart(
            "Export Test", ChartType.RADAR, DataSource.QUALITY_METRICS, ["coverage"]
        )
        
        # Export configuration
        config = builder.export_configuration()
        
        assert "charts" in config
        assert "panels" in config
        assert "exported_at" in config
        assert chart_id in config["charts"]
        
        # Create new builder and import
        new_builder = CustomVisualizationBuilder()
        success = new_builder.import_configuration(config)
        
        assert success
        assert chart_id in new_builder.charts
        assert new_builder.charts[chart_id].title == "Export Test"


class TestGammaDashboardCustomVisualizationIntegration:
    """Test custom visualization integration with Gamma Dashboard Adapter."""
    
    def test_adapter_with_visualization_builder(self):
        """Test adapter creation includes visualization builder."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        assert hasattr(adapter, 'visualization_builder')
        assert isinstance(adapter.visualization_builder, CustomVisualizationBuilder)
        assert hasattr(adapter, 'default_chart_ids')
        assert len(adapter.default_chart_ids) > 0
        
        # Clean up
        adapter.performance_profiler.stop_monitoring()
    
    def test_default_charts_initialization(self):
        """Test default charts are initialized on adapter creation."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        expected_charts = ['productivity_trend', 'code_quality_radar', 'performance_gauge']
        
        for chart_name in expected_charts:
            assert chart_name in adapter.default_chart_ids
            chart_id = adapter.default_chart_ids[chart_name]
            assert chart_id in adapter.visualization_builder.charts
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_custom_visualization_data_format(self):
        """Test custom visualization data format for dashboard."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        custom_viz_data = adapter.get_custom_visualization_data()
        
        # Validate structure
        assert 'id' in custom_viz_data
        assert 'title' in custom_viz_data
        assert 'type' in custom_viz_data
        assert 'position' in custom_viz_data
        assert 'size' in custom_viz_data
        assert 'data' in custom_viz_data
        assert 'timestamp' in custom_viz_data
        assert 'status' in custom_viz_data
        
        # Validate specific values
        assert custom_viz_data['id'] == 'agent-e-custom-visualizations'
        assert custom_viz_data['title'] == 'Custom Visualizations'
        assert custom_viz_data['type'] == 'custom_visualization_panel'
        assert custom_viz_data['status'] == 'active'
        
        # Validate data section
        data = custom_viz_data['data']
        assert 'charts' in data
        assert 'summary' in data
        assert len(data['charts']) > 0
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_custom_chart_creation_via_adapter(self):
        """Test creating custom charts via adapter interface."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        chart_id = adapter.create_custom_chart(
            title="Test Adapter Chart",
            chart_type="line",
            data_source="personal_analytics",
            data_fields=["commits_per_day", "lines_per_day"]
        )
        
        assert chart_id in adapter.visualization_builder.charts
        chart = adapter.visualization_builder.charts[chart_id]
        assert chart.title == "Test Adapter Chart"
        assert chart.chart_type == ChartType.LINE
        assert chart.data_source == DataSource.PERSONAL_ANALYTICS
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_visualization_templates_endpoint(self):
        """Test visualization templates API endpoint."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        templates = adapter.get_visualization_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        
        # Validate template structure
        for template in templates:
            assert 'name' in template
            assert 'title' in template
            assert 'chart_type' in template
            assert 'data_source' in template
            assert 'description' in template
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_custom_charts_list_endpoint(self):
        """Test custom charts list API endpoint."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Create a custom chart
        adapter.create_custom_chart(
            "Test List Chart", "bar", "quality_metrics", ["coverage", "complexity"]
        )
        
        charts_list = adapter.get_custom_charts_list()
        
        assert isinstance(charts_list, list)
        assert len(charts_list) > 0
        
        # Validate chart list structure
        for chart_info in charts_list:
            assert 'id' in chart_info
            assert 'title' in chart_info
            assert 'chart_type' in chart_info
            assert 'data_source' in chart_info
            assert 'created_at' in chart_info
            assert 'updated_at' in chart_info
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_custom_visualization_api_endpoints(self):
        """Test custom visualization API endpoints configuration."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        endpoints = adapter.get_api_endpoints()
        
        # Should have custom visualization endpoints
        expected_endpoints = [
            '/api/personal-analytics/custom-visualizations',
            '/api/personal-analytics/visualization-templates',
            '/api/personal-analytics/custom-charts'
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in endpoints
            assert 'method' in endpoints[endpoint]
            assert 'handler' in endpoints[endpoint]
            assert 'description' in endpoints[endpoint]
            assert endpoints[endpoint]['method'] == 'GET'
        
        adapter.performance_profiler.stop_monitoring()


class TestCustomVisualizationChartGeneration:
    """Test custom visualization chart data generation and formatting."""
    
    def test_line_chart_data_formatting(self):
        """Test line chart data formatting."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart(
            "Line Chart Test", ChartType.LINE, DataSource.PERSONAL_ANALYTICS,
            ["commits_per_day", "lines_per_day"]
        )
        
        test_data = {
            "commits_per_day": 5,
            "lines_per_day": 150
        }
        
        chart_config = builder.generate_chart_data(chart_id, test_data)
        
        assert chart_config['type'] == 'line'
        assert 'data' in chart_config
        assert 'labels' in chart_config['data']
        assert 'datasets' in chart_config['data']
        assert len(chart_config['data']['datasets']) == 1
        assert chart_config['data']['datasets'][0]['data'] == [5, 150]
    
    def test_pie_chart_data_formatting(self):
        """Test pie chart data formatting."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart(
            "Pie Chart Test", ChartType.PIE, DataSource.QUALITY_METRICS,
            ["test_coverage", "code_quality"]
        )
        
        test_data = {
            "quality_metrics": {
                "test_coverage": 85,
                "code_quality": 90
            }
        }
        
        chart_config = builder.generate_chart_data(chart_id, test_data)
        
        assert chart_config['type'] == 'pie'
        assert 'data' in chart_config
        assert 'labels' in chart_config['data']
        assert 'datasets' in chart_config['data']
        assert len(chart_config['data']['datasets'][0]['data']) == 2
        assert chart_config['data']['datasets'][0]['data'] == [85, 90]
    
    def test_radar_chart_data_formatting(self):
        """Test radar chart data formatting."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart(
            "Radar Chart Test", ChartType.RADAR, DataSource.QUALITY_METRICS,
            ["maintainability", "complexity", "coverage"]
        )
        
        test_data = {
            "quality_metrics": {
                "maintainability": 80,
                "complexity": 65,
                "coverage": 75
            }
        }
        
        chart_config = builder.generate_chart_data(chart_id, test_data)
        
        assert chart_config['type'] == 'radar'
        assert 'data' in chart_config
        assert 'labels' in chart_config['data']
        assert 'datasets' in chart_config['data']
        assert chart_config['data']['datasets'][0]['data'] == [80, 65, 75]
    
    def test_chart_styling_application(self):
        """Test custom chart styling application."""
        builder = CustomVisualizationBuilder()
        
        chart_id = builder.create_chart(
            "Styled Chart", ChartType.LINE, DataSource.PERSONAL_ANALYTICS,
            ["commits"],
            styling={
                "colors": ["#FF6384"],
                "line_width": 3
            }
        )
        
        test_data = {"commits": 10}
        chart_config = builder.generate_chart_data(chart_id, test_data)
        
        # Check if styling was applied
        dataset = chart_config['data']['datasets'][0]
        assert dataset['borderColor'] == "#FF6384"
        assert dataset['borderWidth'] == 3
    
    def test_chart_options_merging(self):
        """Test custom chart options merging."""
        builder = CustomVisualizationBuilder()
        
        custom_options = {
            "plugins": {
                "legend": {
                    "position": "bottom"
                }
            },
            "scales": {
                "y": {
                    "max": 200
                }
            }
        }
        
        chart_id = builder.create_chart(
            "Options Chart", ChartType.BAR, DataSource.PERSONAL_ANALYTICS,
            ["lines_added"],
            options=custom_options
        )
        
        test_data = {"lines_added": 120}
        chart_config = builder.generate_chart_data(chart_id, test_data)
        
        # Check if options were merged correctly
        assert chart_config['options']['plugins']['legend']['position'] == 'bottom'
        assert chart_config['options']['scales']['y']['max'] == 200
        # Should still have default responsive option
        assert chart_config['options']['responsive'] == True


class TestCustomVisualizationIntegration:
    """Test integration between custom visualization and dashboard systems."""
    
    def test_custom_visualization_data_size_constraints(self):
        """Test custom visualization data meets size constraints."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        custom_viz_data = adapter.get_custom_visualization_data()
        
        # Convert to JSON to estimate size
        json_data = json.dumps(custom_viz_data, default=str)
        data_size_kb = len(json_data.encode('utf-8')) / 1024
        
        # Should be reasonable size (under 100KB for visualization data)
        assert data_size_kb < 100, f"Custom visualization data too large: {data_size_kb}KB"
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_custom_visualization_response_times(self):
        """Test custom visualization doesn't significantly impact response times."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        import time
        
        # Time visualization generation
        start_time = time.time()
        custom_viz_data = adapter.get_custom_visualization_data()
        response_time_ms = (time.time() - start_time) * 1000
        
        # Should still meet performance requirements (< 300ms for complex visualizations)
        assert response_time_ms < 300, f"Custom visualization too slow: {response_time_ms}ms"
        
        # Verify response time is recorded in summary
        assert 'response_time_ms' in custom_viz_data['data']['summary']
        
        adapter.performance_profiler.stop_monitoring()
    
    def test_custom_visualization_with_all_data_sources(self):
        """Test custom visualizations can access all data sources."""
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Test that custom visualization can generate with all data
        custom_viz_data = adapter.get_custom_visualization_data()
        
        # Should have generated charts
        charts = custom_viz_data['data']['charts']
        assert len(charts) > 0
        
        # Each chart should have proper structure
        for chart in charts:
            assert 'type' in chart
            assert 'data' in chart
            assert 'options' in chart
            assert 'id' in chart
            assert 'name' in chart
        
        adapter.performance_profiler.stop_monitoring()


if __name__ == '__main__':
    # Run specific test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick test
        print("Running quick custom visualization integration test...")
        
        service = create_personal_analytics_service()
        adapter = GammaDashboardAdapter(service)
        
        # Test custom visualization data
        custom_viz = adapter.get_custom_visualization_data()
        print(f"✅ Custom visualization panel: {custom_viz['title']}")
        print(f"✅ Charts generated: {len(custom_viz['data']['charts'])}")
        
        # Test templates
        templates = adapter.get_visualization_templates()
        print(f"✅ Templates available: {len(templates)}")
        
        # Test chart creation
        chart_id = adapter.create_custom_chart(
            "Quick Test Chart", "line", "personal_analytics", ["commits"]
        )
        print(f"✅ Custom chart created: {chart_id[:8]}...")
        
        adapter.performance_profiler.stop_monitoring()
        print("✅ Quick test completed successfully!")
    else:
        # Run full test suite
        pytest.main([__file__, '-v'])