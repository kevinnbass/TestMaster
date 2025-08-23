#!/usr/bin/env python3
"""
STEELCLAD MODULE: SocketIO Event Handlers
=========================================

WebSocket event handlers extracted from unified_dashboard_modular.py
Original: 3,977 lines → SocketIO Module: ~200 lines

Provides complete WebSocket functionality with FULL event handling
extracted from the original setup_socketio_events method.

Author: Agent X (STEELCLAD Anti-Regression Modularization)
"""

from flask_socketio import emit
from flask import request
from datetime import datetime


def register_socketio_events(socketio, dashboard_instance):
    """
    Register ALL SocketIO events from unified_dashboard_modular.py setup_socketio_events method.
    
    STEELCLAD PROTOCOL: This function contains the COMPLETE functionality
    of the original setup_socketio_events method with zero regression.
    """
    
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
                dashboard_instance.helpers.start_performance_stream(client_id)
            elif stream_type == 'visualization_updates':
                dashboard_instance.helpers.start_visualization_stream(client_id)
            elif stream_type == 'predictive_analytics':
                dashboard_instance.helpers.start_predictive_stream(client_id)
        
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
        chart_data = dashboard_instance.helpers.generate_chart_data(chart_type, data_range, filters)
        
        emit('chart_data_update', {
            'chart_type': chart_type,
            'data': chart_data,
            'metadata': {
                'range': data_range,
                'filters': filters,
                'update_frequency': dashboard_instance.helpers.get_update_frequency(chart_type)
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
        prediction_result = dashboard_instance.helpers.generate_predictive_analysis(
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
        
        # Update chart configuration
        if chart_id:
            dashboard_instance.chart_engine.update_chart_config(chart_id, new_config)
            
            emit('chart_config_updated', {
                'chart_id': chart_id,
                'config': new_config,
                'timestamp': datetime.now().isoformat()
            })
    
    @socketio.on('request_3d_data')
    def handle_3d_data_request(data):
        """Handle requests for 3D visualization data."""
        project_structure = dashboard_instance.helpers.generate_3d_project_structure()
        
        emit('3d_data_response', {
            'data': project_structure,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_live_performance')
    def handle_live_performance_request(data):
        """Handle requests for live performance monitoring."""
        metrics_type = data.get('type', 'all')
        
        # Get current performance metrics
        performance_data = {
            'cpu_usage': dashboard_instance.performance_monitor.get_cpu_usage(),
            'memory_usage': dashboard_instance.performance_monitor.get_memory_usage(),
            'network_io': dashboard_instance.performance_monitor.get_network_stats(),
            'response_times': dashboard_instance.performance_monitor.get_response_times()
        }
        
        emit('live_performance_data', {
            'metrics': performance_data,
            'type': metrics_type,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_security_update')
    def handle_security_update_request(data):
        """Handle requests for security status updates."""
        security_data = {
            'real_time_threats': dashboard_instance.helpers.generate_real_time_security_metrics(),
            'threat_correlations': dashboard_instance.helpers.generate_threat_correlations(),
            'predictive_analytics': dashboard_instance.helpers.generate_predictive_security_analytics()
        }
        
        emit('security_update', {
            'security_data': security_data,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_contextual_analysis')
    def handle_contextual_analysis_request(data):
        """Handle requests for contextual intelligence analysis."""
        user_context = data.get('user_context', {})
        system_data = data.get('system_data', {})
        
        # Generate contextual analysis
        analysis = dashboard_instance.enhanced_contextual.analyze_multi_agent_context(system_data)
        proactive_insights = dashboard_instance.enhanced_contextual.generate_proactive_insights(
            system_data, user_context
        )
        
        emit('contextual_analysis_result', {
            'analysis': analysis,
            'proactive_insights': proactive_insights,
            'user_context': user_context,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_ai_visualization')
    def handle_ai_visualization_request(data):
        """Handle requests for AI-powered visualization recommendations."""
        data_characteristics = data.get('data_characteristics', {})
        user_context = data.get('user_context', {})
        
        recommendations = dashboard_instance.ai_visualization.select_optimal_visualization(
            data_characteristics, user_context
        )
        
        emit('ai_visualization_recommendations', {
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_multi_service_data')
    def handle_multi_service_data_request(data):
        """Handle requests for multi-service backend aggregated data."""
        service_filter = data.get('services', None)
        
        # Get aggregated data from all backend services
        aggregated_data = dashboard_instance.service_aggregator.get_aggregated_data()
        service_health = dashboard_instance.service_aggregator.check_all_services_health()
        
        emit('multi_service_data', {
            'aggregated_data': aggregated_data,
            'service_health': service_health,
            'requested_services': service_filter,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_budget_status')
    def handle_budget_status_request(data):
        """Handle requests for API budget and usage status."""
        budget_data = {
            'usage_statistics': dashboard_instance.api_usage_tracker.get_usage_statistics(),
            'persistent_stats': dashboard_instance.persistent_api_tracker.get_usage_stats(24),
            'agent_budgets': dashboard_instance.persistent_api_tracker.get_agent_budgets()
        }
        
        emit('budget_status_update', {
            'budget_data': budget_data,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('request_filter_presets')
    def handle_filter_presets_request(data):
        """Handle requests for filter presets and configurations."""
        filter_type = data.get('type', 'all')
        
        presets = dashboard_instance.filter_ui.get_filter_presets()
        
        emit('filter_presets_response', {
            'presets': presets,
            'type': filter_type,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('export_chart_request')
    def handle_chart_export_request(data):
        """Handle chart export requests via WebSocket."""
        chart_id = data.get('chart_id')
        export_format = data.get('format', 'png')
        
        try:
            export_data = dashboard_instance.chart_engine.export_chart(chart_id, export_format)
            
            emit('chart_export_complete', {
                'chart_id': chart_id,
                'format': export_format,
                'export_data': export_data,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            emit('chart_export_error', {
                'chart_id': chart_id,
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            })
    
    print("✅ STEELCLAD: All SocketIO events registered successfully")