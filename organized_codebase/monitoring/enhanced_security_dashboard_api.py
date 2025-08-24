
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

#!/usr/bin/env python3
"""
Enhanced Security Dashboard API
================================
Advanced security dashboard API with real-time monitoring, unified scanning,
and comprehensive security intelligence integration.
"""

from flask import Blueprint, jsonify, request, Response
from flask_cors import CORS
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import logging
from datetime import datetime, timedelta

# Import security components
from unified_security_scanner import UnifiedSecurityScanner, ScanConfiguration
from enhanced_realtime_security_monitor import EnhancedRealtimeSecurityMonitor
from enhanced_security_intelligence_agent import EnhancedSecurityIntelligenceAgent
from live_code_quality_monitor import LiveCodeQualityMonitor
from realtime_metrics_collector import RealtimeMetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask blueprint
enhanced_security_bp = Blueprint('enhanced_security', __name__)
CORS(enhanced_security_bp)

# Global instances
security_monitor = None
unified_scanner = None
intelligence_agent = None
quality_monitor = None
metrics_collector = None

# Real-time data streams
active_streams = {}
scan_queue = asyncio.Queue()
event_subscribers = []

def initialize_security_systems():
    """Initialize all security systems."""
    global security_monitor, unified_scanner, intelligence_agent, quality_monitor, metrics_collector
    
    config = {
        'enable_real_time_monitoring': True,
        'enable_classical_analysis': True,
        'enable_correlation_analysis': True,
        'risk_threshold': 70.0
    }
    
    security_monitor = EnhancedRealtimeSecurityMonitor(config)
    unified_scanner = UnifiedSecurityScanner(ScanConfiguration(target_paths=['.']))
    intelligence_agent = EnhancedSecurityIntelligenceAgent(config)
    quality_monitor = LiveCodeQualityMonitor()
    metrics_collector = RealtimeMetricsCollector(collection_interval_ms=100)
    
    # Start monitoring systems
    security_monitor.start()
    quality_monitor.start_monitoring(['.'])
    metrics_collector.start_collection()
    
    logger.info("Security systems initialized")

# Initialize on import
initialize_security_systems()

@enhanced_security_bp.route('/api/security/status', methods=['GET'])
def get_security_status():
    """Get overall security status."""
    try:
        # Get risk dashboard
        risk_dashboard = security_monitor.get_risk_dashboard()
        
        # Get current metrics
        current_metrics = metrics_collector.get_current_metrics()
        
        # Get quality statistics
        quality_stats = quality_monitor.get_quality_statistics()
        
        # Get scanner statistics
        scanner_stats = unified_scanner.get_statistics()
        
        status = {
            'timestamp': time.time(),
            'overall_status': {
                'risk_score': risk_dashboard['overall_metrics']['average_risk_score'],
                'active_alerts': risk_dashboard['overall_metrics']['total_active_alerts'],
                'high_risk_files': risk_dashboard['overall_metrics']['high_risk_files'],
                'monitoring_active': risk_dashboard['overall_metrics']['monitoring_systems_active']
            },
            'risk_distribution': risk_dashboard['risk_distribution'],
            'metrics_summary': {
                'collection_rate': current_metrics['performance']['collections_per_second'],
                'error_rate': current_metrics['performance']['error_rate'],
                'system_metrics_count': current_metrics['system_metrics']['total_count'],
                'application_metrics_count': current_metrics['application_metrics']['total_count']
            },
            'quality_summary': quality_stats,
            'scanner_summary': {
                'total_scans': scanner_stats['total_scans'],
                'success_rate': scanner_stats['success_rate'],
                'avg_scan_time': scanner_stats['avg_scan_time']
            }
        }
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/scan', methods=['POST'])
def trigger_security_scan():
    """Trigger unified security scan."""
    try:
        data = request.json or {}
        target_path = data.get('target_path')
        scan_options = data.get('options', {})
        
        # Configure scan
        config = ScanConfiguration(
            target_paths=[target_path] if target_path else ['.'],
            enable_real_time_monitoring=scan_options.get('realtime', True),
            enable_classical_analysis=scan_options.get('classical', True),
            enable_quality_correlation=scan_options.get('quality', True),
            enable_performance_profiling=scan_options.get('performance', True),
            enable_compliance_checking=scan_options.get('compliance', True),
            enable_intelligent_testing=scan_options.get('testing', True),
            generate_report=scan_options.get('report', True)
        )
        
        # Update scanner configuration
        unified_scanner.config = config
        
        # Perform scan asynchronously
        scan_result = unified_scanner.scan(target_path)
        
        # Convert to dict for JSON serialization
        result_dict = {
            'scan_id': scan_result.scan_id,
            'timestamp': scan_result.timestamp,
            'target': scan_result.target_path,
            'duration': scan_result.scan_duration,
            'overall_risk_score': scan_result.overall_risk_score,
            'confidence_score': scan_result.confidence_score,
            'findings_summary': {
                'vulnerabilities': len(scan_result.vulnerabilities),
                'compliance_violations': len(scan_result.compliance_violations),
                'quality_issues': len(scan_result.quality_issues),
                'performance_concerns': len(scan_result.performance_concerns)
            },
            'risk_distribution': scan_result.risk_distribution,
            'remediation_plan': scan_result.remediation_plan,
            'scan_metrics': scan_result.scan_metrics
        }
        
        return jsonify(result_dict)
    
    except Exception as e:
        logger.error(f"Error triggering scan: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/alerts', methods=['GET'])
def get_security_alerts():
    """Get security alerts with filtering."""
    try:
        # Get query parameters
        threat_level = request.args.get('threat_level')
        limit = int(request.args.get('limit', 50))
        hours = int(request.args.get('hours', 24))
        
        # Get alerts from monitor
        alerts = security_monitor.get_security_alerts()
        
        # Filter by threat level if specified
        if threat_level:
            alerts = [a for a in alerts if a.get('threat_level') == threat_level]
        
        # Filter by time window
        cutoff_time = time.time() - (hours * 3600)
        alerts = [a for a in alerts if a.get('timestamp', 0) >= cutoff_time]
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Limit results
        alerts = alerts[:limit]
        
        return jsonify({
            'alerts': alerts,
            'total': len(alerts),
            'time_window_hours': hours,
            'filters': {
                'threat_level': threat_level,
                'limit': limit
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/trends', methods=['GET'])
def get_security_trends():
    """Get security trends and analytics."""
    try:
        hours = int(request.args.get('hours', 24))
        time_window = hours * 3600
        
        # Get trends from monitor
        trends = security_monitor.get_security_trends(time_window)
        
        # Convert trends to dict format
        trends_dict = {}
        for trend_name, trend_data in trends.items():
            trends_dict[trend_name] = {
                'metric_name': trend_data.metric_name,
                'current_value': trend_data.current_value,
                'trend_direction': trend_data.trend_direction,
                'change_rate': trend_data.change_rate,
                'confidence': trend_data.confidence,
                'time_window': trend_data.time_window,
                'data_points': len(trend_data.historical_values)
            }
        
        return jsonify({
            'trends': trends_dict,
            'time_window_hours': hours,
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/intelligence', methods=['POST'])
def get_security_intelligence():
    """Get security intelligence for a target."""
    try:
        data = request.json or {}
        target_path = data.get('target_path', '.')
        
        # Perform intelligence analysis
        analysis = intelligence_agent.analyze_security_with_classical_context(
            target_path, data.get('context', {})
        )
        
        # Extract key insights
        enhanced = analysis.get('enhanced_analysis', {})
        insights = {
            'target': target_path,
            'timestamp': enhanced.get('analysis_timestamp', time.time()),
            'risk_assessment': enhanced.get('integrated_risk_assessment', {}),
            'correlation_insights': enhanced.get('correlation_insights', {}),
            'enhanced_findings_count': len(enhanced.get('enhanced_findings', [])),
            'test_suite': None
        }
        
        # Add test suite summary if available
        if enhanced.get('intelligent_test_suite'):
            suite = enhanced['intelligent_test_suite']
            insights['test_suite'] = {
                'test_cases': len(suite.get('test_cases', [])),
                'execution_priority': suite.get('execution_priority', 0),
                'expected_coverage': suite.get('expected_coverage', 0),
                'estimated_time': suite.get('estimated_execution_time', 0)
            }
        
        return jsonify(insights)
    
    except Exception as e:
        logger.error(f"Error getting intelligence: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/realtime/stream', methods=['GET'])
def stream_security_events():
    """Stream real-time security events using Server-Sent Events."""
    def generate():
        """Generate SSE stream."""
        stream_id = f"stream_{int(time.time())}"
        active_streams[stream_id] = True
        
        try:
            while active_streams.get(stream_id, False):
                # Get latest security events
                events = security_monitor.security_events
                
                if events:
                    # Get most recent event
                    latest_event = events[-1]
                    
                    # Format as SSE
                    event_data = {
                        'event_id': latest_event.event_id,
                        'event_type': latest_event.event_type,
                        'severity': latest_event.severity,
                        'timestamp': latest_event.timestamp,
                        'description': latest_event.description,
                        'metrics': latest_event.metrics
                    }
                    
                    yield f"data: {json.dumps(event_data)}\n\n"
                
                # Also stream metrics
                current_metrics = metrics_collector.get_current_metrics()
                metrics_event = {
                    'event_type': 'metrics_update',
                    'timestamp': time.time(),
                    'metrics': current_metrics['performance']
                }
                
                yield f"data: {json.dumps(metrics_event)}\n\n"
                
                # Sleep briefly
                time.sleep(1)
                
        except GeneratorExit:
            # Client disconnected
            active_streams.pop(stream_id, None)
            logger.info(f"Stream {stream_id} closed")
    
    return Response(generate(), mimetype='text/event-stream')

@enhanced_security_bp.route('/api/security/quality/correlation', methods=['GET'])
def get_quality_security_correlation():
    """Get quality-security correlation analysis."""
    try:
        target_path = request.args.get('target_path', '.')
        
        # Get quality snapshot
        quality_snapshot = quality_monitor.analyze_file_quality(target_path)
        
        # Get security alerts for the file
        security_alerts = security_monitor.alert_manager.get_alerts_by_file(target_path)
        
        # Perform correlation
        correlation = security_monitor.metrics_integrator.correlate_quality_security(
            target_path, quality_snapshot, security_alerts
        )
        
        return jsonify({
            'target': target_path,
            'quality_score': correlation['quality_score'],
            'security_alert_count': correlation['security_alert_count'],
            'correlations': correlation['correlations'],
            'timestamp': correlation['timestamp']
        })
    
    except Exception as e:
        logger.error(f"Error getting correlation: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/risk/profiles', methods=['GET'])
def get_risk_profiles():
    """Get risk profiles for monitored files."""
    try:
        limit = int(request.args.get('limit', 10))
        sort_by = request.args.get('sort_by', 'overall_risk_score')
        
        # Get all risk profiles
        risk_profiles = list(security_monitor.risk_profiles.values())
        
        # Sort by specified field
        if sort_by == 'overall_risk_score':
            risk_profiles.sort(key=lambda x: x.overall_risk_score, reverse=True)
        elif sort_by == 'vulnerability_score':
            risk_profiles.sort(key=lambda x: x.vulnerability_score, reverse=True)
        elif sort_by == 'quality_score':
            risk_profiles.sort(key=lambda x: x.quality_score)
        elif sort_by == 'complexity_score':
            risk_profiles.sort(key=lambda x: x.complexity_score)
        
        # Limit results
        risk_profiles = risk_profiles[:limit]
        
        # Convert to dict
        profiles_dict = []
        for profile in risk_profiles:
            profiles_dict.append({
                'file_path': profile.file_path,
                'overall_risk_score': profile.overall_risk_score,
                'vulnerability_score': profile.vulnerability_score,
                'quality_score': profile.quality_score,
                'performance_score': profile.performance_score,
                'complexity_score': profile.complexity_score,
                'risk_factors': profile.risk_factors,
                'mitigation_suggestions': profile.mitigation_suggestions,
                'last_updated': profile.last_updated
            })
        
        return jsonify({
            'risk_profiles': profiles_dict,
            'total': len(profiles_dict),
            'sort_by': sort_by
        })
    
    except Exception as e:
        logger.error(f"Error getting risk profiles: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    """Get comprehensive dashboard summary."""
    try:
        # Get risk dashboard
        risk_dashboard = security_monitor.get_risk_dashboard()
        
        # Get quality statistics
        quality_stats = quality_monitor.get_quality_statistics()
        
        # Get scanner statistics
        scanner_stats = unified_scanner.get_statistics()
        
        # Get current metrics
        current_metrics = metrics_collector.get_current_metrics()
        
        # Get recent scan history
        recent_scans = list(unified_scanner.scan_history)[-5:]
        scan_history = []
        for scan in recent_scans:
            scan_history.append({
                'scan_id': scan.scan_id,
                'timestamp': scan.timestamp,
                'target': scan.target_path,
                'risk_score': scan.overall_risk_score,
                'findings': scan.scan_metrics.get('total_findings', 0)
            })
        
        summary = {
            'timestamp': time.time(),
            'risk_overview': {
                'average_risk': risk_dashboard['overall_metrics']['average_risk_score'],
                'high_risk_files': risk_dashboard['overall_metrics']['high_risk_files'],
                'active_alerts': risk_dashboard['overall_metrics']['total_active_alerts'],
                'risk_distribution': risk_dashboard['risk_distribution']
            },
            'quality_overview': {
                'average_score': quality_stats.get('average_quality_score', 0),
                'files_monitored': quality_stats.get('files_monitored', 0),
                'quality_distribution': quality_stats.get('quality_distribution', {})
            },
            'scanning_overview': {
                'total_scans': scanner_stats['total_scans'],
                'success_rate': scanner_stats['success_rate'],
                'avg_scan_time': scanner_stats['avg_scan_time'],
                'total_vulnerabilities': scanner_stats['total_vulnerabilities']
            },
            'monitoring_status': {
                'security_monitoring': risk_dashboard['overall_metrics']['monitoring_systems_active']['security'],
                'quality_monitoring': risk_dashboard['overall_metrics']['monitoring_systems_active']['quality'],
                'metrics_collection': risk_dashboard['overall_metrics']['monitoring_systems_active']['metrics'],
                'collection_rate': current_metrics['performance']['collections_per_second']
            },
            'recent_scans': scan_history,
            'correlation_insights': risk_dashboard.get('correlation_insights', []),
            'high_risk_events': [
                {
                    'event_id': event['event_id'],
                    'event_type': event['event_type'],
                    'severity': event['severity'],
                    'timestamp': event['timestamp']
                }
                for event in risk_dashboard.get('high_risk_events', [])[:5]
            ]
        }
        
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/reports/generate', methods=['POST'])
def generate_security_report():
    """Generate comprehensive security report."""
    try:
        data = request.json or {}
        report_type = data.get('type', 'comprehensive')
        format_type = data.get('format', 'json')
        time_range = data.get('time_range', 24)  # hours
        
        # Gather data for report
        report_data = {
            'generated_at': time.time(),
            'report_type': report_type,
            'time_range_hours': time_range
        }
        
        if report_type == 'comprehensive':
            # Get all available data
            report_data['risk_dashboard'] = security_monitor.get_risk_dashboard()
            report_data['security_trends'] = security_monitor.get_security_trends(time_range * 3600)
            report_data['quality_statistics'] = quality_monitor.get_quality_statistics()
            report_data['scanner_statistics'] = unified_scanner.get_statistics()
            report_data['recent_scans'] = [
                {
                    'scan_id': scan.scan_id,
                    'timestamp': scan.timestamp,
                    'risk_score': scan.overall_risk_score,
                    'findings': scan.scan_metrics.get('total_findings', 0)
                }
                for scan in list(unified_scanner.scan_history)[-10:]
            ]
        
        elif report_type == 'vulnerabilities':
            # Focus on vulnerabilities
            alerts = security_monitor.get_security_alerts()
            report_data['total_vulnerabilities'] = len(alerts)
            report_data['critical_vulnerabilities'] = len([a for a in alerts if a.get('threat_level') == 'critical'])
            report_data['vulnerability_distribution'] = {}
            for alert in alerts:
                alert_type = alert.get('alert_type', 'unknown')
                report_data['vulnerability_distribution'][alert_type] = report_data['vulnerability_distribution'].get(alert_type, 0) + 1
        
        elif report_type == 'compliance':
            # Focus on compliance
            # This would integrate with compliance checking
            report_data['compliance_status'] = 'Requires dedicated compliance scan'
        
        # Format report based on requested format
        if format_type == 'json':
            return jsonify(report_data)
        
        elif format_type == 'html':
            # Generate HTML report
            html = f"""
            <html>
            <head><title>Security Report</title></head>
            <body>
                <h1>Security Report</h1>
                <p>Generated: {datetime.fromtimestamp(report_data['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Type: {report_data['report_type']}</p>
                <p>Time Range: {report_data['time_range_hours']} hours</p>
                <pre>{json.dumps(report_data, indent=2, default=str)}</pre>
            </body>
            </html>
            """
            return Response(html, mimetype='text/html')
        
        elif format_type == 'markdown':
            # Generate Markdown report
            md = f"""# Security Report
Generated: {datetime.fromtimestamp(report_data['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}
Type: {report_data['report_type']}
Time Range: {report_data['time_range_hours']} hours

## Data
```json
{json.dumps(report_data, indent=2, default=str)}
```
"""
            return Response(md, mimetype='text/markdown')
        
        return jsonify(report_data)
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/config', methods=['GET', 'PUT'])
def manage_security_config():
    """Get or update security configuration."""
    try:
        if request.method == 'GET':
            # Get current configuration
            config = {
                'security_monitor': {
                    'enable_correlation_analysis': security_monitor.enable_correlation_analysis,
                    'enable_advanced_detection': security_monitor.enable_advanced_detection,
                    'risk_score_threshold': security_monitor.risk_score_threshold
                },
                'quality_monitor': {
                    'analysis_interval': quality_monitor.analysis_interval,
                    'enable_trends': quality_monitor.enable_trends,
                    'quality_thresholds': quality_monitor.quality_thresholds
                },
                'metrics_collector': {
                    'collection_interval': metrics_collector.collection_interval * 1000,  # Convert to ms
                    'running': metrics_collector.running
                },
                'scanner': {
                    'parallel_workers': unified_scanner.config.parallel_workers,
                    'cache_results': unified_scanner.config.cache_results,
                    'scan_timeout': unified_scanner.config.scan_timeout
                }
            }
            return jsonify(config)
        
        elif request.method == 'PUT':
            # Update configuration
            data = request.json or {}
            
            # Update security monitor config
            if 'security_monitor' in data:
                sm_config = data['security_monitor']
                if 'enable_correlation_analysis' in sm_config:
                    security_monitor.enable_correlation_analysis = sm_config['enable_correlation_analysis']
                if 'enable_advanced_detection' in sm_config:
                    security_monitor.enable_advanced_detection = sm_config['enable_advanced_detection']
                if 'risk_score_threshold' in sm_config:
                    security_monitor.risk_score_threshold = sm_config['risk_score_threshold']
            
            # Update quality monitor config
            if 'quality_monitor' in data:
                qm_config = data['quality_monitor']
                if 'analysis_interval' in qm_config:
                    quality_monitor.analysis_interval = qm_config['analysis_interval']
                if 'enable_trends' in qm_config:
                    quality_monitor.enable_trends = qm_config['enable_trends']
                if 'quality_thresholds' in qm_config:
                    quality_monitor.quality_thresholds.update(qm_config['quality_thresholds'])
            
            # Update metrics collector config
            if 'metrics_collector' in data:
                mc_config = data['metrics_collector']
                if 'collection_interval' in mc_config:
                    # Would need to restart collector with new interval
                    pass
            
            # Update scanner config
            if 'scanner' in data:
                sc_config = data['scanner']
                if 'parallel_workers' in sc_config:
                    unified_scanner.config.parallel_workers = sc_config['parallel_workers']
                if 'cache_results' in sc_config:
                    unified_scanner.config.cache_results = sc_config['cache_results']
                if 'scan_timeout' in sc_config:
                    unified_scanner.config.scan_timeout = sc_config['scan_timeout']
            
            return jsonify({'status': 'Configuration updated successfully'})
    
    except Exception as e:
        logger.error(f"Error managing config: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_security_bp.route('/api/security/health', methods=['GET'])
def get_security_health():
    """Get health status of security systems."""
    try:
        health = {
            'timestamp': time.time(),
            'systems': {
                'security_monitor': {
                    'running': security_monitor.running,
                    'active_alerts': len(security_monitor.alert_manager.get_active_alerts()),
                    'risk_profiles': len(security_monitor.risk_profiles),
                    'status': 'healthy' if security_monitor.running else 'stopped'
                },
                'quality_monitor': {
                    'running': quality_monitor.monitoring,
                    'files_monitored': len(quality_monitor.file_watch_list),
                    'snapshots': len(quality_monitor.quality_snapshots),
                    'status': 'healthy' if quality_monitor.monitoring else 'stopped'
                },
                'metrics_collector': {
                    'running': metrics_collector.running,
                    'collections': metrics_collector.collection_stats['collections'],
                    'errors': metrics_collector.collection_stats['errors'],
                    'status': 'healthy' if metrics_collector.running else 'stopped'
                },
                'unified_scanner': {
                    'total_scans': unified_scanner.scan_statistics['total_scans'],
                    'success_rate': unified_scanner.scan_statistics.get('success_rate', 0),
                    'cache_size': len(unified_scanner.scan_cache),
                    'status': 'healthy'
                }
            },
            'overall_health': 'healthy' if all(
                s.get('status') == 'healthy' 
                for s in health['systems'].values()
            ) else 'degraded'
        }
        
        return jsonify(health)
    
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        return jsonify({
            'timestamp': time.time(),
            'overall_health': 'error',
            'error': str(e)
        }), 500

# Error handler
@enhanced_security_bp.errorhandler(Exception)
def handle_error(error):
    """Handle API errors."""
    logger.error(f"API error: {error}")
    return jsonify({
        'error': str(error),
        'timestamp': time.time()
    }), 500

# Export blueprint
__all__ = ['enhanced_security_bp']