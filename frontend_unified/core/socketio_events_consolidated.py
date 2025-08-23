#!/usr/bin/env python3
"""
STEELCLAD MODULE: SocketIO Events Consolidated
==============================================

SocketIO event handlers extracted from unified_dashboard_modular.py
Original: 1,154 lines → SocketIO Events Module: ~114 lines

Complete functionality extraction with zero regression.

Author: Agent X (STEELCLAD Anti-Regression Modularization)
"""

from datetime import datetime
from flask_socketio import emit
from flask import request


def register_socketio_events(socketio, dashboard_instance):
    """Register all SocketIO event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print(f"Client connected: {request.sid}")
        emit('connected', {
            'message': 'Connected to Modular Dashboard',
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print(f"Client disconnected: {request.sid}")
    
    @socketio.on('request_analysis')
    def handle_analysis_request(data):
        """Handle real-time analysis requests."""
        agent_data = data.get('agent_data', {})
        analysis = dashboard_instance.contextual_engine.analyze_multi_agent_context(agent_data)
        
        emit('analysis_result', {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
    
    # Hour 8: Advanced Real-time Data Streaming Events
    @socketio.on('subscribe_live_data')
    def handle_live_data_subscription(data):
        """Handle subscription to real-time data streams."""
        stream_types = data.get('streams', [])
        client_id = request.sid
        
        print(f"Client {client_id} subscribed to streams: {stream_types}")
        
        # Start streaming data for subscribed types
        for stream_type in stream_types:
            if stream_type == 'performance_metrics':
                dashboard_instance.helpers._start_performance_stream(client_id)
            elif stream_type == 'visualization_updates':
                dashboard_instance.helpers._start_visualization_stream(client_id)
            elif stream_type == 'predictive_analytics':
                dashboard_instance.helpers._start_predictive_stream(client_id)
        
        emit('subscription_confirmed', {
            'streams': stream_types,
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_chart_data')
    def handle_chart_data_request(data):
        """Handle requests for chart data with real-time updates."""
        chart_type = data.get('chart_type', 'line')
        data_range = data.get('range', '1h')
        filters = data.get('filters', {})
        
        # Generate chart data based on request
        chart_data = dashboard_instance.helpers._generate_chart_data(chart_type, data_range, filters)
        
        emit('chart_data_update', {
            'chart_type': chart_type,
            'data': chart_data,
            'metadata': {
                'range': data_range,
                'filters': filters,
                'update_frequency': dashboard_instance.helpers._get_update_frequency(chart_type)
            },
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_predictive_analysis')
    def handle_predictive_analysis_request(data):
        """Handle requests for predictive analytics visualization."""
        analysis_type = data.get('type', 'trend_forecast')
        historical_data = data.get('historical_data', {})
        forecast_horizon = data.get('horizon', 24)  # hours
        
        # Generate predictive analysis
        prediction_result = dashboard_instance.helpers._generate_predictive_analysis(
            analysis_type, historical_data, forecast_horizon
        )
        
        emit('predictive_analysis_result', {
            'analysis_type': analysis_type,
            'predictions': prediction_result,
            'confidence_intervals': prediction_result.get('confidence', {}),
            'recommendation': prediction_result.get('recommendation', ''),
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('update_chart_config')
    def handle_chart_config_update(data):
        """Handle dynamic chart configuration updates."""
        chart_id = data.get('chart_id')
        new_config = data.get('config', {})
        user_context = data.get('user_context', {})
        
        # Apply intelligent configuration updates
        optimized_config = dashboard_instance.visualization_engine.create_interactive_chart_config(
            new_config.get('type', 'intelligent_line_chart'),
            new_config.get('data', {}),
            user_context,
            new_config.get('enhancements', [])
        )
        
        emit('chart_config_updated', {
            'chart_id': chart_id,
            'config': optimized_config,
            'optimizations_applied': optimized_config.get('optimizations', []),
            'timestamp': datetime.now().isoformat()
        })