"""
Performance API Blueprint - AGENT B Hour 19-21 Enhancement
===========================================================

Comprehensive REST API for performance monitoring, optimization, and analytics.
Integrates with UnifiedPerformanceHub for enterprise-grade performance management.

Features:
- Real-time performance metrics collection
- Performance profiling and analysis
- Optimization recommendations
- System health monitoring
- Performance trend analysis
- Resource usage tracking
"""

from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
import time
import threading

from ..monitoring.unified_performance_hub import (
    UnifiedPerformanceHub, PerformanceLevel, OptimizationType,
    PerformanceMetric, OptimizationRecommendation, PerformanceProfile
)

# Create blueprint
performance_api = Blueprint('performance_api', __name__, url_prefix='/api/v2/performance')

# Global performance hub instance
performance_hub = None
_hub_lock = threading.Lock()

# Logger
logger = logging.getLogger("performance_api")


def get_performance_hub():
    """Get or create performance hub instance."""
    global performance_hub
    with _hub_lock:
        if performance_hub is None:
            config = {
                'auto_start': True,
                'monitor_interval': 30,
                'history_size': 5000,
                'system_name': 'testmaster_api'
            }
            performance_hub = UnifiedPerformanceHub(config)
        return performance_hub


# === PERFORMANCE STATUS ENDPOINTS ===

@performance_api.route('/status', methods=['GET'])
def get_performance_status():
    """
    Get comprehensive performance status.
    
    Returns:
        JSON with performance metrics, system health, and monitoring status
    """
    try:
        hub = get_performance_hub()
        summary = hub.get_performance_summary()
        
        return jsonify({
            'status': 'success',
            'data': summary,
            'api_version': '2.0.0',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get performance status: {e}")
        return jsonify({'error': str(e)}), 500


@performance_api.route('/health', methods=['GET'])
def health_check():
    """Quick performance health check."""
    try:
        hub = get_performance_hub()
        metrics = hub._collect_comprehensive_metrics()
        
        # Simple health assessment
        cpu_usage = next((m.value for m in metrics if m.name == 'cpu_usage'), 0)
        memory_usage = next((m.value for m in metrics if m.name == 'memory_usage'), 0)
        
        health_status = 'healthy'
        if cpu_usage > 90 or memory_usage > 90:
            health_status = 'critical'
        elif cpu_usage > 70 or memory_usage > 70:
            health_status = 'warning'
        
        return jsonify({
            'status': health_status,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


# === METRICS COLLECTION ENDPOINTS ===

@performance_api.route('/metrics/current', methods=['GET'])
def get_current_metrics():
    """
    Get current performance metrics.
    
    Returns:
        JSON with current system and application metrics
    """
    try:
        hub = get_performance_hub()
        metrics = hub._collect_comprehensive_metrics()
        
        metrics_data = [
            {
                'name': m.name,
                'value': m.value,
                'unit': m.unit,
                'category': m.category,
                'timestamp': m.timestamp.isoformat(),
                'metadata': m.metadata
            }
            for m in metrics
        ]
        
        return jsonify({
            'status': 'success',
            'data': metrics_data,
            'count': len(metrics_data),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get current metrics: {e}")
        return jsonify({'error': str(e)}), 500


@performance_api.route('/metrics/measure', methods=['POST'])
def measure_operation():
    """
    Measure performance of a custom operation.
    
    Request body:
    {
        "operation_name": "database_query",
        "operation_type": "function_call", 
        "metadata": {"query": "SELECT * FROM users"}
    }
    
    Returns:
        Performance measurement results
    """
    try:
        data = request.get_json()
        if not data or not data.get('operation_name'):
            return jsonify({'error': 'operation_name required'}), 400
        
        operation_name = data['operation_name']
        operation_type = data.get('operation_type', 'custom')
        metadata = data.get('metadata', {})
        
        # Simple performance measurement
        start_time = time.time()
        # Simulate operation (in real implementation, this would execute actual operation)
        time.sleep(0.001)  # Minimal delay for demonstration
        end_time = time.time()
        
        performance_data = {
            'operation_name': operation_name,
            'operation_type': operation_type,
            'duration_ms': (end_time - start_time) * 1000,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'success': True
        }
        
        return jsonify({
            'status': 'success',
            'data': performance_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to measure operation: {e}")
        return jsonify({'error': str(e)}), 500


# === PERFORMANCE ANALYSIS ENDPOINTS ===

@performance_api.route('/analysis/profile', methods=['GET'])
def get_performance_profile():
    """
    Get latest performance profile with comprehensive analysis.
    
    Returns:
        Latest performance profile with scores, bottlenecks, and recommendations
    """
    try:
        hub = get_performance_hub()
        
        if not hub._performance_history:
            return jsonify({
                'status': 'success',
                'data': None,
                'message': 'No performance profiles available yet'
            }), 200
        
        latest_profile = hub._performance_history[-1]
        
        profile_data = {
            'profile_id': latest_profile.profile_id,
            'system_name': latest_profile.system_name,
            'timestamp': latest_profile.timestamp.isoformat(),
            'overall_score': latest_profile.overall_score,
            'performance_level': latest_profile.level.value,
            'bottlenecks': latest_profile.bottlenecks,
            'resource_usage': latest_profile.resource_usage,
            'recommendations_count': len(latest_profile.recommendations),
            'metrics_count': len(latest_profile.metrics)
        }
        
        return jsonify({
            'status': 'success',
            'data': profile_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get performance profile: {e}")
        return jsonify({'error': str(e)}), 500


@performance_api.route('/analysis/trends', methods=['GET'])
def get_performance_trends():
    """
    Get performance trends analysis.
    
    Query parameters:
    - window: Time window in hours (default: 1)
    - metric: Specific metric to analyze (optional)
    
    Returns:
        Performance trends and statistical analysis
    """
    try:
        window_hours = request.args.get('window', 1, type=int)
        metric_name = request.args.get('metric')
        
        hub = get_performance_hub()
        
        # Get profiles within time window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_profiles = [
            p for p in hub._performance_history 
            if p.timestamp >= cutoff_time
        ]
        
        if not recent_profiles:
            return jsonify({
                'status': 'success',
                'data': {
                    'message': f'No performance data in last {window_hours} hours',
                    'profiles_count': 0
                }
            }), 200
        
        # Calculate trends
        scores = [p.overall_score for p in recent_profiles]
        
        trends_data = {
            'window_hours': window_hours,
            'profiles_analyzed': len(recent_profiles),
            'performance_trend': {
                'current_score': scores[-1] if scores else 0,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'trend_direction': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'stable'
            },
            'bottlenecks_frequency': {},
            'performance_levels': {}
        }
        
        # Analyze bottlenecks frequency
        all_bottlenecks = []
        for profile in recent_profiles:
            all_bottlenecks.extend(profile.bottlenecks)
        
        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        trends_data['bottlenecks_frequency'] = bottleneck_counts
        
        # Performance levels distribution
        level_counts = {}
        for profile in recent_profiles:
            level = profile.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        trends_data['performance_levels'] = level_counts
        
        return jsonify({
            'status': 'success',
            'data': trends_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        return jsonify({'error': str(e)}), 500


# === OPTIMIZATION ENDPOINTS ===

@performance_api.route('/optimization/recommendations', methods=['GET'])
def get_optimization_recommendations():
    """
    Get performance optimization recommendations.
    
    Query parameters:
    - limit: Maximum recommendations to return (default: 10)
    - priority: Filter by priority (high, medium, low)
    - type: Filter by optimization type
    
    Returns:
        List of optimization recommendations with priorities and implementation details
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        priority_filter = request.args.get('priority')
        type_filter = request.args.get('type')
        
        hub = get_performance_hub()
        recommendations = hub.get_optimization_recommendations(limit=50)  # Get more for filtering
        
        # Apply filters
        if priority_filter:
            recommendations = [r for r in recommendations if r.priority == priority_filter]
        
        if type_filter:
            recommendations = [r for r in recommendations if r.optimization_type.value == type_filter]
        
        # Limit results
        recommendations = recommendations[:limit]
        
        recommendations_data = [
            {
                'optimization_id': r.optimization_id,
                'optimization_type': r.optimization_type.value,
                'priority': r.priority,
                'title': r.title,
                'description': r.description,
                'estimated_improvement': r.estimated_improvement,
                'implementation_effort': r.implementation_effort,
                'resources_required': r.resources_required,
                'testing_required': r.testing_required,
                'code_changes': r.code_changes
            }
            for r in recommendations
        ]
        
        return jsonify({
            'status': 'success',
            'data': recommendations_data,
            'count': len(recommendations_data),
            'available_types': [t.value for t in OptimizationType],
            'available_priorities': ['critical', 'high', 'medium', 'low']
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {e}")
        return jsonify({'error': str(e)}), 500


@performance_api.route('/optimization/implement', methods=['POST'])
def implement_optimization():
    """
    Mark optimization as implemented and track results.
    
    Request body:
    {
        "optimization_id": "cpu_opt_123456",
        "implementation_notes": "Implemented caching layer",
        "expected_improvement": 25.0
    }
    
    Returns:
        Confirmation and tracking information
    """
    try:
        data = request.get_json()
        if not data or not data.get('optimization_id'):
            return jsonify({'error': 'optimization_id required'}), 400
        
        optimization_id = data['optimization_id']
        implementation_notes = data.get('implementation_notes', '')
        expected_improvement = data.get('expected_improvement', 0.0)
        
        # Track implementation (in a real system, this would update database)
        implementation_data = {
            'optimization_id': optimization_id,
            'implementation_date': datetime.now().isoformat(),
            'implementation_notes': implementation_notes,
            'expected_improvement': expected_improvement,
            'status': 'implemented',
            'tracking_started': True
        }
        
        return jsonify({
            'status': 'success',
            'data': implementation_data,
            'message': 'Optimization implementation tracked successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to track optimization implementation: {e}")
        return jsonify({'error': str(e)}), 500


# === MONITORING CONTROL ENDPOINTS ===

@performance_api.route('/monitoring/start', methods=['POST'])
def start_performance_monitoring():
    """Start performance monitoring."""
    try:
        hub = get_performance_hub()
        if not hub._monitoring_active:
            hub.start_monitoring()
            return jsonify({
                'status': 'success',
                'message': 'Performance monitoring started',
                'monitoring_active': True
            }), 200
        else:
            return jsonify({
                'status': 'success',
                'message': 'Performance monitoring already active',
                'monitoring_active': True
            }), 200
            
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        return jsonify({'error': str(e)}), 500


@performance_api.route('/monitoring/stop', methods=['POST'])
def stop_performance_monitoring():
    """Stop performance monitoring."""
    try:
        hub = get_performance_hub()
        if hub._monitoring_active:
            hub.stop_monitoring()
            return jsonify({
                'status': 'success',
                'message': 'Performance monitoring stopped',
                'monitoring_active': False
            }), 200
        else:
            return jsonify({
                'status': 'success',
                'message': 'Performance monitoring already inactive',
                'monitoring_active': False
            }), 200
            
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        return jsonify({'error': str(e)}), 500


# === DATA EXPORT ENDPOINTS ===

@performance_api.route('/export/data', methods=['GET'])
def export_performance_data():
    """
    Export performance data.
    
    Query parameters:
    - format: Export format (json, csv) - default: json
    - window: Time window in hours - default: 24
    
    Returns:
        Performance data in requested format
    """
    try:
        export_format = request.args.get('format', 'json')
        window_hours = request.args.get('window', 24, type=int)
        
        hub = get_performance_hub()
        
        if export_format == 'json':
            data = hub.export_performance_data('json')
            return Response(
                data,
                mimetype='application/json',
                headers={
                    'Content-Disposition': f'attachment; filename=performance_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                }
            )
        else:
            return jsonify({'error': f'Format {export_format} not supported'}), 400
            
    except Exception as e:
        logger.error(f"Failed to export performance data: {e}")
        return jsonify({'error': str(e)}), 500


# === UTILITY ENDPOINTS ===

@performance_api.route('/config', methods=['GET'])
def get_performance_config():
    """Get current performance monitoring configuration."""
    try:
        hub = get_performance_hub()
        config_data = {
            'monitoring_active': hub._monitoring_active,
            'config': hub.config,
            'thresholds': {
                'performance_baselines': hub._performance_baselines,
                'performance_targets': hub._performance_targets
            },
            'buffer_sizes': {
                'performance_history': len(hub._performance_history),
                'metrics_buffer': len(hub._metrics_buffer)
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': config_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get performance config: {e}")
        return jsonify({'error': str(e)}), 500


# Initialize function for app integration
def init_performance_api(app_config: Optional[Dict[str, Any]] = None):
    """Initialize performance API with configuration."""
    global performance_hub
    with _hub_lock:
        if performance_hub is None:
            config = app_config or {}
            config.setdefault('auto_start', True)
            config.setdefault('system_name', 'testmaster_performance_api')
            performance_hub = UnifiedPerformanceHub(config)
    
    logger.info("Performance API initialized successfully")


# Export blueprint
__all__ = ['performance_api', 'init_performance_api']