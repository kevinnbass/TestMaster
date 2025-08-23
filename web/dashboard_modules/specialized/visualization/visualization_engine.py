#!/usr/bin/env python3
"""
STEELCLAD Phase 5: Visualization Engine - Extracted from Performance Analytics Dashboard
========================================================================================

Interactive visualization engine using Plotly for comprehensive performance analytics
with real-time charts, ML predictions, and system integration dashboards.

Author: Agent Z (STEELCLAD Protocol)
Extracted from: performance_analytics_dashboard.py (181 lines)
"""

import logging
import numpy as np
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any

class VisualizationEngine:
    """Creates interactive visualizations using Plotly"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('VisualizationEngine')
    
    def create_performance_overview_chart(self, metrics_data: Dict[str, Any]) -> str:
        """Create overview chart showing key performance metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Response Time', 'Cache Hit Ratio'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Sample data - in production would use real metrics
        timestamps = [datetime.now() - timedelta(minutes=x) for x in range(60, 0, -1)]
        
        # CPU Usage
        cpu_data = np.random.normal(70, 10, 60)  # Would be real data
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_data, name='CPU %', line=dict(color='red')),
            row=1, col=1
        )
        
        # Memory Usage
        memory_data = np.random.normal(75, 8, 60)
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_data, name='Memory %', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Response Time
        response_data = np.random.normal(85, 15, 60)
        fig.add_trace(
            go.Scatter(x=timestamps, y=response_data, name='Response Time (ms)', line=dict(color='green')),
            row=2, col=1
        )
        
        # Cache Hit Ratio
        cache_data = np.random.normal(0.85, 0.05, 60)
        fig.add_trace(
            go.Scatter(x=timestamps, y=cache_data, name='Cache Hit Ratio', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Performance Overview - Last Hour',
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def create_ml_predictions_chart(self, predictions: List[Dict]) -> str:
        """Create chart showing ML predictions"""
        if not predictions:
            # Empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No ML predictions available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        else:
            fig = go.Figure()
            
            for pred in predictions:
                fig.add_trace(go.Bar(
                    x=[pred['metric_name']],
                    y=[pred['current_value']],
                    name='Current',
                    marker_color='blue',
                    opacity=0.7
                ))
                
                fig.add_trace(go.Bar(
                    x=[pred['metric_name']],
                    y=[pred['predicted_value']],
                    name='Predicted',
                    marker_color='red',
                    opacity=0.7
                ))
        
        fig.update_layout(
            title='ML Performance Predictions',
            xaxis_title='Metrics',
            yaxis_title='Values',
            template='plotly_white',
            height=400
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def create_distributed_scaling_chart(self, scaling_data: Dict) -> str:
        """Create chart showing distributed scaling status"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Instance Count', 'Instance Health'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Instance count over time (sample data)
        timestamps = [datetime.now() - timedelta(minutes=x) for x in range(30, 0, -1)]
        instance_counts = np.random.randint(2, 6, 30)
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, 
                y=instance_counts, 
                name='Instances', 
                line=dict(color='purple'),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Instance health status
        health_labels = ['Healthy', 'Degraded', 'Unhealthy']
        health_counts = [
            scaling_data.get('healthy_instances', 0),
            scaling_data.get('total_instances', 0) - scaling_data.get('healthy_instances', 0),
            0  # Unhealthy count
        ]
        
        fig.add_trace(
            go.Bar(
                x=health_labels,
                y=health_counts,
                name='Instance Health',
                marker_color=['green', 'yellow', 'red']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Distributed Scaling Status',
            height=400,
            template='plotly_white'
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def create_alpha_integration_chart(self, alpha_data: Dict) -> str:
        """Create chart showing Alpha system integration status"""
        fig = go.Figure()
        
        # ML optimization score gauge
        if 'ml_optimization_score' in alpha_data:
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=alpha_data['ml_optimization_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Alpha ML Optimization Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
        
        fig.update_layout(
            title='Alpha Integration Status',
            height=400,
            template='plotly_white'
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)

def create_visualization_engine(config) -> VisualizationEngine:
    """
    Factory function to create a configured visualization engine instance.
    
    Args:
        config: Dashboard configuration object
        
    Returns:
        Configured VisualizationEngine instance
    """
    return VisualizationEngine(config)

# Export key components
__all__ = ['VisualizationEngine', 'create_visualization_engine']
"""
Visualization Engine Module
============================

Visualization functionality extracted from performance_analytics_dashboard.py
for STEELCLAD modularization (Agent Y supporting Agent Z)

Handles:
- Interactive Plotly visualizations
- Real-time chart generation
- Performance trend visualization
- Multi-system data correlation charts
"""

import logging
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

# Import configuration - inline definition for standalone operation
class DashboardConfig:
    """Dashboard configuration for standalone operation"""
    def __init__(self):
        self.max_data_points = 1000
        self.enable_predictions = True
        self.enable_alpha_integration = True
        self.performance_thresholds = {
            'response_time_ms': {'good': 50, 'warning': 100, 'critical': 200},
            'cpu_usage_percent': {'good': 70, 'warning': 80, 'critical': 90},
            'memory_usage_percent': {'good': 75, 'warning': 85, 'critical': 95},
            'cache_hit_ratio': {'good': 0.9, 'warning': 0.8, 'critical': 0.7},
            'error_rate': {'good': 0.01, 'warning': 0.05, 'critical': 0.1}
        }


class VisualizationEngine:
    """
    Creates interactive visualizations using Plotly
    
    Generates real-time charts and graphs for performance analytics dashboard.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger('VisualizationEngine')
    
    def create_performance_overview_chart(self, metrics_data: Dict[str, Any]) -> str:
        """Create comprehensive performance overview chart"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('System Performance', 'Cache Efficiency', 
                              'ML Predictions', 'Test Results'),
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                       [{"secondary_y": True}, {"secondary_y": True}]]
            )
            
            # System performance metrics
            perf_data = metrics_data.get('performance_monitoring', {})
            if perf_data:
                x_data = list(range(len(perf_data)))
                y_data = [data.get('value', 0) for data in perf_data.values()]
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data, name='Performance Score'),
                    row=1, col=1
                )
            
            # Cache metrics
            cache_data = metrics_data.get('caching_system', {})
            if cache_data:
                fig.add_trace(
                    go.Bar(x=['Hit Ratio', 'Memory Util'], 
                          y=[cache_data.get('hit_ratio', 0), 
                             cache_data.get('memory_utilization', 0)],
                          name='Cache Metrics'),
                    row=1, col=2
                )
            
            # ML predictions
            ml_data = metrics_data.get('ml_optimizer', {})
            if ml_data and 'predictions' in ml_data:
                predictions = ml_data['predictions'][:5]  # Top 5
                pred_names = [p['metric_name'] for p in predictions]
                pred_values = [p['predicted_value'] for p in predictions]
                fig.add_trace(
                    go.Scatter(x=pred_names, y=pred_values, 
                             mode='markers+lines', name='Predictions'),
                    row=2, col=1
                )
            
            # Test results
            alpha_data = metrics_data.get('alpha_monitoring', {})
            if alpha_data:
                test_labels = ['Passed', 'Failed']
                test_values = [alpha_data.get('passed_tests', 0), 
                              alpha_data.get('failed_tests', 0)]
                fig.add_trace(
                    go.Pie(labels=test_labels, values=test_values, name='Test Results'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="Performance Analytics Overview",
                height=600,
                showlegend=True,
                template='plotly_dark'
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Failed to create performance overview chart: {e}")
            return "<div>Error creating chart</div>"
    
    def create_real_time_metrics_chart(self, history_data: Dict[str, List]) -> str:
        """Create real-time metrics chart"""
        try:
            fig = go.Figure()
            
            colors = ['#00ff88', '#00ccff', '#ff6b6b', '#ffd93d', '#6bcf7f']
            color_idx = 0
            
            for metric_name, data_points in history_data.items():
                if data_points and len(data_points) > 1:
                    timestamps = [dp.get('timestamp', i) for i, dp in enumerate(data_points)]
                    values = [dp.get('value', 0) if isinstance(dp, dict) else dp 
                             for dp in data_points]
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines+markers',
                        name=metric_name.replace('_', ' ').title(),
                        line=dict(color=colors[color_idx % len(colors)]),
                        marker=dict(size=4)
                    ))
                    color_idx += 1
            
            fig.update_layout(
                title="Real-Time Performance Metrics",
                xaxis_title="Time",
                yaxis_title="Value",
                template='plotly_dark',
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Failed to create real-time metrics chart: {e}")
            return "<div>Error creating real-time chart</div>"
    
    def create_correlation_heatmap(self, correlation_data: Dict[str, float]) -> str:
        """Create correlation heatmap"""
        try:
            if not correlation_data:
                return "<div>No correlation data available</div>"
            
            # Parse correlation pairs
            metrics = set()
            for pair in correlation_data.keys():
                metric1, metric2 = pair.split('_vs_')
                metrics.add(metric1)
                metrics.add(metric2)
            
            metrics = sorted(list(metrics))
            correlation_matrix = [[0 for _ in metrics] for _ in metrics]
            
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics):
                    if i == j:
                        correlation_matrix[i][j] = 1.0
                    else:
                        pair_key1 = f"{metric1}_vs_{metric2}"
                        pair_key2 = f"{metric2}_vs_{metric1}"
                        if pair_key1 in correlation_data:
                            correlation_matrix[i][j] = correlation_data[pair_key1]
                        elif pair_key2 in correlation_data:
                            correlation_matrix[i][j] = correlation_data[pair_key2]
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=metrics,
                y=metrics,
                colorscale='RdBu',
                zmid=0,
                text=[[f'{val:.2f}' for val in row] for row in correlation_matrix],
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Metrics Correlation Analysis",
                template='plotly_dark',
                height=500,
                margin=dict(l=100, r=40, t=60, b=100)
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Failed to create correlation heatmap: {e}")
            return "<div>Error creating correlation chart</div>"
    
    def create_predictions_chart(self, predictions_data: List[Dict[str, Any]]) -> str:
        """Create predictions visualization chart"""
        try:
            if not predictions_data:
                return "<div>No predictions available</div>"
            
            fig = go.Figure()
            
            for i, prediction in enumerate(predictions_data[:10]):  # Limit to 10
                metric_name = prediction.get('metric_name', f'Metric {i+1}')
                current_val = prediction.get('current_value', 0)
                predicted_val = prediction.get('predicted_value', 0)
                confidence = prediction.get('confidence', 0)
                
                # Current vs predicted values
                fig.add_trace(go.Bar(
                    x=[metric_name],
                    y=[current_val],
                    name='Current',
                    marker_color='#00ff88',
                    opacity=0.7
                ))
                
                fig.add_trace(go.Bar(
                    x=[metric_name],
                    y=[predicted_val],
                    name='Predicted',
                    marker_color='#ff6b6b',
                    opacity=0.7
                ))
                
                # Add confidence as annotation
                fig.add_annotation(
                    x=metric_name,
                    y=max(current_val, predicted_val) * 1.1,
                    text=f"Confidence: {confidence:.1%}",
                    showarrow=False,
                    font=dict(size=10)
                )
            
            fig.update_layout(
                title="ML Predictions vs Current Values",
                xaxis_title="Metrics",
                yaxis_title="Value",
                template='plotly_dark',
                height=400,
                barmode='group'
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Failed to create predictions chart: {e}")
            return "<div>Error creating predictions chart</div>"
    
    def create_system_health_gauge(self, health_score: float, health_status: str) -> str:
        """Create system health gauge chart"""
        try:
            color_map = {
                'excellent': 'green',
                'good': 'lightgreen', 
                'fair': 'yellow',
                'poor': 'orange',
                'critical': 'red'
            }
            
            color = color_map.get(health_status, 'gray')
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"System Health: {health_status.title()}"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=300,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Failed to create system health gauge: {e}")
            return "<div>Error creating health gauge</div>"
    
    def create_performance_trends_chart(self, trends_data: Dict[str, Any]) -> str:
        """Create performance trends analysis chart"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Performance Trends', 'Resource Utilization'),
                vertical_spacing=0.1
            )
            
            # Performance trends
            if 'performance_history' in trends_data:
                history = trends_data['performance_history']
                if history:
                    x_data = list(range(len(history)))
                    y_data = [h.get('overall_score', 0) for h in history]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='lines+markers',
                            name='Performance Score',
                            line=dict(color='#00ff88', width=2)
                        ),
                        row=1, col=1
                    )
            
            # Resource utilization
            if 'resource_history' in trends_data:
                resource_history = trends_data['resource_history']
                if resource_history:
                    x_data = list(range(len(resource_history)))
                    cpu_data = [r.get('cpu_usage', 0) for r in resource_history]
                    memory_data = [r.get('memory_usage', 0) for r in resource_history]
                    
                    fig.add_trace(
                        go.Scatter(x=x_data, y=cpu_data, name='CPU Usage',
                                 line=dict(color='#ff6b6b')),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=x_data, y=memory_data, name='Memory Usage',
                                 line=dict(color='#ffd93d')),
                        row=2, col=1
                    )
            
            fig.update_layout(
                title="Performance Trends Analysis",
                template='plotly_dark',
                height=500,
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Failed to create performance trends chart: {e}")
            return "<div>Error creating trends chart</div>"
    
    def generate_dashboard_html(self, all_charts: Dict[str, str]) -> str:
        """Generate complete dashboard HTML with all charts"""
        try:
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Performance Analytics Dashboard</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1e1e1e; color: #fff; }}
                    .chart-container {{ margin-bottom: 30px; background: #2d2d2d; padding: 20px; border-radius: 10px; }}
                    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                    .full-width {{ grid-column: span 2; }}
                    h1 {{ text-align: center; color: #00ff88; margin-bottom: 30px; }}
                    h2 {{ color: #00ccff; margin-top: 0; }}
                </style>
            </head>
            <body>
                <h1>🚀 Performance Analytics Dashboard</h1>
                
                <div class="chart-container full-width">
                    <h2>System Health Overview</h2>
                    {health_gauge}
                </div>
                
                <div class="chart-grid">
                    <div class="chart-container">
                        <h2>Performance Overview</h2>
                        {overview_chart}
                    </div>
                    <div class="chart-container">
                        <h2>Real-Time Metrics</h2>
                        {realtime_chart}
                    </div>
                </div>
                
                <div class="chart-grid">
                    <div class="chart-container">
                        <h2>ML Predictions</h2>
                        {predictions_chart}
                    </div>
                    <div class="chart-container">
                        <h2>Correlation Analysis</h2>
                        {correlation_chart}
                    </div>
                </div>
                
                <div class="chart-container full-width">
                    <h2>Performance Trends</h2>
                    {trends_chart}
                </div>
                
                <script>
                    // Auto-refresh every 30 seconds
                    setTimeout(() => location.reload(), 30000);
                </script>
            </body>
            </html>
            """
            
            return html_template.format(
                health_gauge=all_charts.get('health_gauge', ''),
                overview_chart=all_charts.get('overview', ''),
                realtime_chart=all_charts.get('realtime', ''),
                predictions_chart=all_charts.get('predictions', ''),
                correlation_chart=all_charts.get('correlation', ''),
                trends_chart=all_charts.get('trends', '')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard HTML: {e}")
            return "<html><body><h1>Dashboard Generation Error</h1></body></html>"