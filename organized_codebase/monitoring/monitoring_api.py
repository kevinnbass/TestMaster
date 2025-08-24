"""
TestMaster Monitoring API Blueprint - AGENT B ENHANCED
======================================================

REST API for monitoring and quality assurance capabilities.
Provides comprehensive monitoring, alerting, and performance tracking.

AGENT B Enhancement: Hour 10-12
- Real-time monitoring endpoints
- Quality assurance integration
- Performance metrics collection
- Alert management system
"""

from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
import time
from collections import deque

# Import monitoring components
try:
    from ..monitoring.agent_qa_modular import AgentQualityAssurance
    from ..monitoring.qa_base import QualityLevel, AlertType
    from ..monitoring.qa_monitor import QualityMonitor
    from ..monitoring.qa_scorer import QualityScorer
    from ..monitoring.enterprise_performance_monitor import EnterprisePerformanceMonitor
    
    # Try legacy imports with fallbacks
    pattern_detector = None
    try:
        from ..monitoring.pattern_detector import PatternDetector
    except ImportError:
        PatternDetector = None
        
except ImportError as e:
    logger.warning(f"Could not import monitoring components: {e}")
    AgentQualityAssurance = None
    QualityLevel = None
    AlertType = None
    QualityMonitor = None
    QualityScorer = None
    PatternDetector = None
    EnterprisePerformanceMonitor = None

# Create blueprint
monitoring_api = Blueprint('monitoring_api', __name__, url_prefix='/api/v2/monitoring')

# Component instances - initialize with enhanced system
qa_system = AgentQualityAssurance("api_agent") if AgentQualityAssurance else None
pattern_detector = PatternDetector() if PatternDetector else None
qa_monitor = QualityMonitor() if QualityMonitor else None
qa_scorer = QualityScorer() if QualityScorer else None
performance_monitor = EnterprisePerformanceMonitor() if EnterprisePerformanceMonitor else None

# Real-time metrics buffer
metrics_buffer = deque(maxlen=10000)
alerts_buffer = deque(maxlen=1000)

# Logger
logger = logging.getLogger("monitoring_api")


def init_monitoring_api(app_config: Optional[Dict[str, Any]] = None):
    """Initialize monitoring API with configuration."""
    global qa_system, pattern_detector, qa_monitor, qa_scorer, performance_monitor
    
    config = app_config or {}
    qa_system = AgentQualityAssurance("monitoring_api", config)
    pattern_detector = PatternDetector(config)
    qa_monitor = QAMonitor(config)
    qa_scorer = QAScorer(config)
    performance_monitor = EnterprisePerformanceMonitor(config)
    
    # Start monitoring if configured
    if config.get('auto_start_monitoring', False):
        qa_monitor.start_monitoring()
    
    logger.info("Monitoring API initialized with all components")


# === MONITORING STATUS ENDPOINTS ===

@monitoring_api.route('/status', methods=['GET'])
def get_monitoring_status():
    """
    Get comprehensive monitoring system status.
    
    Returns:
        JSON with monitoring status, active monitors, and statistics
    """
    try:
        status = {
            'api_version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'qa_system': 'active' if qa_system else 'inactive',
                'pattern_detector': 'active' if pattern_detector else 'inactive',
                'qa_monitor': 'active' if qa_monitor else 'inactive',
                'qa_scorer': 'active' if qa_scorer else 'inactive',
                'performance_monitor': 'active' if performance_monitor else 'inactive'
            },
            'metrics': {
                'buffer_size': len(metrics_buffer),
                'alerts_count': len(alerts_buffer),
                'monitoring_uptime': time.time()  # Would track actual uptime
            }
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/health', methods=['GET'])
def health_check():
    """Quick health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200


# === REAL-TIME MONITORING ENDPOINTS ===

@monitoring_api.route('/metrics/collect', methods=['POST'])
def collect_metrics():
    """
    Collect real-time metrics.
    
    Request body:
    {
        "metric_name": "cpu_usage",
        "value": 75.5,
        "unit": "percentage",
        "tags": {"host": "server1"},
        "timestamp": "2024-01-01T00:00:00"
    }
    
    Returns:
        Confirmation of metric collection
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('metric_name') or 'value' not in data:
            return jsonify({'error': 'metric_name and value required'}), 400
        
        # Create metric entry
        metric = {
            'metric_name': data['metric_name'],
            'value': data['value'],
            'unit': data.get('unit', 'count'),
            'tags': data.get('tags', {}),
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        }
        
        # Add to buffer
        metrics_buffer.append(metric)
        
        # Check for alert conditions
        _check_alert_conditions(metric)
        
        return jsonify({'status': 'collected', 'metric': metric}), 200
        
    except Exception as e:
        logger.error(f"Metric collection failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/metrics/batch', methods=['POST'])
def collect_metrics_batch():
    """
    Collect multiple metrics in batch.
    
    Request body:
    {
        "metrics": [
            {"metric_name": "cpu", "value": 75},
            {"metric_name": "memory", "value": 60}
        ]
    }
    
    Returns:
        Batch collection summary
    """
    try:
        data = request.get_json()
        metrics = data.get('metrics', [])
        
        collected = 0
        for metric_data in metrics:
            if metric_data.get('metric_name') and 'value' in metric_data:
                metric = {
                    'metric_name': metric_data['metric_name'],
                    'value': metric_data['value'],
                    'unit': metric_data.get('unit', 'count'),
                    'tags': metric_data.get('tags', {}),
                    'timestamp': metric_data.get('timestamp', datetime.now().isoformat())
                }
                metrics_buffer.append(metric)
                collected += 1
        
        return jsonify({
            'status': 'collected',
            'metrics_collected': collected,
            'metrics_total': len(metrics)
        }), 200
        
    except Exception as e:
        logger.error(f"Batch metric collection failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/metrics/query', methods=['GET'])
def query_metrics():
    """
    Query collected metrics.
    
    Query parameters:
    - metric_name: Filter by metric name
    - start_time: Start time for query
    - end_time: End time for query
    - limit: Maximum results (default 100)
    
    Returns:
        Matching metrics
    """
    try:
        metric_name = request.args.get('metric_name')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = int(request.args.get('limit', 100))
        
        # Filter metrics
        results = []
        for metric in metrics_buffer:
            # Apply filters
            if metric_name and metric['metric_name'] != metric_name:
                continue
            if start_time and metric['timestamp'] < start_time:
                continue
            if end_time and metric['timestamp'] > end_time:
                continue
            
            results.append(metric)
            if len(results) >= limit:
                break
        
        return jsonify({
            'metrics': results,
            'count': len(results),
            'query': {
                'metric_name': metric_name,
                'start_time': start_time,
                'end_time': end_time
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Metric query failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/metrics/stream', methods=['GET'])
def stream_metrics():
    """
    Stream real-time metrics using Server-Sent Events.
    
    Returns:
        SSE stream of metrics
    """
    def generate():
        """Generate SSE events."""
        last_index = 0
        while True:
            # Get new metrics since last index
            current_metrics = list(metrics_buffer)
            if len(current_metrics) > last_index:
                new_metrics = current_metrics[last_index:]
                for metric in new_metrics:
                    yield f"data: {json.dumps(metric)}\n\n"
                last_index = len(current_metrics)
            time.sleep(1)  # Check every second
    
    return Response(generate(), mimetype="text/event-stream")


# === QUALITY ASSURANCE ENDPOINTS ===

@monitoring_api.route('/quality/assess', methods=['POST'])
def assess_quality():
    """
    Assess quality of agent output.
    
    Request body:
    {
        "agent_id": "agent1",
        "output": {...},
        "expected": {...},
        "metadata": {...}
    }
    
    Returns:
        Quality assessment report
    """
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'unknown')
        output = data.get('output', {})
        expected = data.get('expected')
        metadata = data.get('metadata', {})
        
        # Assess quality
        report = qa_system.assess_quality(
            agent_output=output,
            expected_output=expected,
            metadata=metadata
        )
        
        # Convert to JSON format
        report_data = {
            'agent_id': agent_id,
            'overall_score': report.overall_score.score if hasattr(report, 'overall_score') else 0,
            'breakdown': {
                'functionality': report.overall_score.breakdown.get('functionality', 0) if hasattr(report, 'overall_score') else 0,
                'performance': report.overall_score.breakdown.get('performance', 0) if hasattr(report, 'overall_score') else 0,
                'reliability': report.overall_score.breakdown.get('reliability', 0) if hasattr(report, 'overall_score') else 0
            },
            'quality_level': 'high' if report.overall_score.score > 80 else 'medium' if report.overall_score.score > 60 else 'low',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(report_data), 200
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/quality/validate', methods=['POST'])
def validate_output():
    """
    Validate agent output against rules.
    
    Request body:
    {
        "agent_id": "agent1",
        "output": {...},
        "validation_rules": [...]
    }
    
    Returns:
        Validation results
    """
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'unknown')
        output = data.get('output', {})
        rules = data.get('validation_rules', [])
        
        # Perform validation
        validation_result = validate_agent_output(output)
        
        return jsonify({
            'agent_id': agent_id,
            'valid': validation_result.get('valid', False) if validation_result else False,
            'issues': validation_result.get('issues', []) if validation_result else [],
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/quality/score', methods=['POST'])
def calculate_quality_score():
    """
    Calculate quality score for agent performance.
    
    Request body:
    {
        "metrics": {...},
        "weights": {...}
    }
    
    Returns:
        Quality score with breakdown
    """
    try:
        data = request.get_json()
        metrics = data.get('metrics', {})
        weights = data.get('weights', {})
        
        # Calculate score
        score = qa_scorer.calculate_score(metrics, weights)
        
        return jsonify({
            'score': score,
            'grade': 'A' if score > 90 else 'B' if score > 80 else 'C' if score > 70 else 'D' if score > 60 else 'F',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Score calculation failed: {e}")
        return jsonify({'error': str(e)}), 500


# === PATTERN DETECTION ENDPOINTS ===

@monitoring_api.route('/patterns/detect', methods=['POST'])
def detect_patterns():
    """
    Detect patterns in monitoring data.
    
    Request body:
    {
        "data": [...],
        "pattern_types": ["anomaly", "trend", "seasonal"],
        "sensitivity": 0.95
    }
    
    Returns:
        Detected patterns
    """
    try:
        data = request.get_json()
        monitoring_data = data.get('data', [])
        pattern_types = data.get('pattern_types', ['anomaly'])
        sensitivity = data.get('sensitivity', 0.95)
        
        patterns = []
        
        # Detect requested pattern types
        if 'anomaly' in pattern_types:
            anomalies = pattern_detector.detect_anomalies(monitoring_data, sensitivity)
            patterns.extend([{'type': 'anomaly', 'data': a} for a in anomalies])
        
        if 'trend' in pattern_types:
            trends = pattern_detector.detect_trends(monitoring_data)
            patterns.extend([{'type': 'trend', 'data': t} for t in trends])
        
        return jsonify({
            'patterns': patterns,
            'pattern_count': len(patterns),
            'sensitivity': sensitivity
        }), 200
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")
        return jsonify({'error': str(e)}), 500


# === ALERT MANAGEMENT ENDPOINTS ===

@monitoring_api.route('/alerts/create', methods=['POST'])
def create_alert():
    """
    Create a monitoring alert.
    
    Request body:
    {
        "alert_type": "threshold|anomaly|quality",
        "severity": "low|medium|high|critical",
        "title": "High CPU Usage",
        "description": "CPU usage exceeded 90%",
        "source": "server1"
    }
    
    Returns:
        Created alert details
    """
    try:
        data = request.get_json()
        
        alert = {
            'alert_id': f"alert_{int(time.time())}",
            'alert_type': data.get('alert_type', 'threshold'),
            'severity': data.get('severity', 'medium'),
            'title': data.get('title', 'Monitoring Alert'),
            'description': data.get('description', ''),
            'source': data.get('source', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        
        # Add to alerts buffer
        alerts_buffer.append(alert)
        
        # Trigger alert actions based on severity
        if alert['severity'] == 'critical':
            logger.critical(f"Critical alert: {alert['title']}")
        
        return jsonify(alert), 201
        
    except Exception as e:
        logger.error(f"Alert creation failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/alerts/list', methods=['GET'])
def list_alerts():
    """
    List monitoring alerts.
    
    Query parameters:
    - severity: Filter by severity
    - resolved: Filter by resolved status
    - limit: Maximum results
    
    Returns:
        List of alerts
    """
    try:
        severity = request.args.get('severity')
        resolved = request.args.get('resolved')
        limit = int(request.args.get('limit', 100))
        
        # Filter alerts
        results = []
        for alert in alerts_buffer:
            # Apply filters
            if severity and alert.get('severity') != severity:
                continue
            if resolved is not None:
                alert_resolved = alert.get('resolved', False)
                if resolved.lower() == 'true' and not alert_resolved:
                    continue
                if resolved.lower() == 'false' and alert_resolved:
                    continue
            
            results.append(alert)
            if len(results) >= limit:
                break
        
        return jsonify({
            'alerts': results,
            'count': len(results),
            'active_count': sum(1 for a in results if not a.get('resolved', False))
        }), 200
        
    except Exception as e:
        logger.error(f"Alert listing failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """
    Resolve a monitoring alert.
    
    Args:
        alert_id: ID of alert to resolve
        
    Request body:
    {
        "resolution_notes": "Issue resolved by restarting service"
    }
    
    Returns:
        Updated alert
    """
    try:
        data = request.get_json() or {}
        resolution_notes = data.get('resolution_notes', '')
        
        # Find and update alert
        for alert in alerts_buffer:
            if alert.get('alert_id') == alert_id:
                alert['resolved'] = True
                alert['resolved_at'] = datetime.now().isoformat()
                alert['resolution_notes'] = resolution_notes
                return jsonify(alert), 200
        
        return jsonify({'error': 'Alert not found'}), 404
        
    except Exception as e:
        logger.error(f"Alert resolution failed: {e}")
        return jsonify({'error': str(e)}), 500


# === PERFORMANCE MONITORING ENDPOINTS ===

@monitoring_api.route('/performance/analyze', methods=['POST'])
def analyze_performance():
    """
    Analyze performance metrics.
    
    Request body:
    {
        "time_window": 3600,
        "metrics": ["latency", "throughput", "error_rate"]
    }
    
    Returns:
        Performance analysis results
    """
    try:
        data = request.get_json()
        time_window = data.get('time_window', 3600)  # Default 1 hour
        metric_names = data.get('metrics', ['latency'])
        
        # Perform analysis
        analysis = performance_monitor.analyze_performance(
            time_window=timedelta(seconds=time_window),
            metrics=metric_names
        )
        
        return jsonify(analysis), 200
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@monitoring_api.route('/performance/trends', methods=['GET'])
def get_performance_trends():
    """
    Get performance trends over time.
    
    Query parameters:
    - metric: Metric name
    - period: Time period (hour|day|week)
    
    Returns:
        Performance trend data
    """
    try:
        metric = request.args.get('metric', 'latency')
        period = request.args.get('period', 'hour')
        
        # Calculate time window
        if period == 'hour':
            time_window = timedelta(hours=1)
        elif period == 'day':
            time_window = timedelta(days=1)
        elif period == 'week':
            time_window = timedelta(weeks=1)
        else:
            time_window = timedelta(hours=1)
        
        # Get trends
        trends = performance_monitor.get_trends(metric, time_window)
        
        return jsonify({
            'metric': metric,
            'period': period,
            'trends': trends
        }), 200
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


# === MONITORING DASHBOARD ENDPOINTS ===

@monitoring_api.route('/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    """
    Get monitoring dashboard summary.
    
    Returns:
        Complete dashboard data
    """
    try:
        # Calculate summary metrics
        active_alerts = sum(1 for a in alerts_buffer if not a.get('resolved', False))
        recent_metrics = list(metrics_buffer)[-100:]  # Last 100 metrics
        
        # Calculate averages
        avg_values = {}
        metric_groups = {}
        for metric in recent_metrics:
            name = metric['metric_name']
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric['value'])
        
        for name, values in metric_groups.items():
            avg_values[name] = sum(values) / len(values) if values else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'alerts': {
                'active': active_alerts,
                'total': len(alerts_buffer),
                'critical': sum(1 for a in alerts_buffer if a.get('severity') == 'critical' and not a.get('resolved', False))
            },
            'metrics': {
                'buffer_size': len(metrics_buffer),
                'recent_count': len(recent_metrics),
                'averages': avg_values
            },
            'system_health': 'healthy' if active_alerts < 5 else 'warning' if active_alerts < 10 else 'critical'
        }
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Dashboard summary failed: {e}")
        return jsonify({'error': str(e)}), 500


# === HELPER FUNCTIONS ===

def _check_alert_conditions(metric: Dict[str, Any]):
    """Check if metric triggers alert conditions."""
    # Example alert conditions
    if metric['metric_name'] == 'cpu_usage' and metric['value'] > 90:
        alert = {
            'alert_id': f"alert_{int(time.time())}",
            'alert_type': 'threshold',
            'severity': 'high',
            'title': 'High CPU Usage',
            'description': f"CPU usage at {metric['value']}%",
            'source': metric.get('tags', {}).get('host', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        alerts_buffer.append(alert)
    
    elif metric['metric_name'] == 'error_rate' and metric['value'] > 5:
        alert = {
            'alert_id': f"alert_{int(time.time())}",
            'alert_type': 'threshold',
            'severity': 'critical',
            'title': 'High Error Rate',
            'description': f"Error rate at {metric['value']}%",
            'source': metric.get('tags', {}).get('service', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        alerts_buffer.append(alert)


# Export blueprint and initialization function
__all__ = ['monitoring_api', 'init_monitoring_api']