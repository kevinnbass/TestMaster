"""
Orchestration API Blueprint - AGENT B Hour 25-36 Enhancement
============================================================

Ultimate REST API for testing and monitoring orchestration.
Integrates AdvancedTestingOrchestrator and AdvancedMonitoringCoordinator
for complete infrastructure automation and coordination.

Features:
- Advanced test orchestration and scheduling
- AI-driven monitoring coordination
- Cross-system correlation and insights
- Predictive analytics and optimization
- Comprehensive orchestration management
- Real-time coordination status and control
"""

from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
import uuid

from ..testing.advanced_testing_orchestrator import (
    AdvancedTestingOrchestrator, TestStrategy, TestPriority, TestEnvironment,
    TestPlan, TestExecution, TestInsight
)
from ..monitoring.advanced_monitoring_coordinator import (
    AdvancedMonitoringCoordinator, MonitoringStrategy, AlertSeverity, MonitoringScope,
    MonitoringPlan, CorrelatedAlert, MonitoringInsight, SystemHealthProfile
)

# Create blueprint
orchestration_api = Blueprint('orchestration_api', __name__, url_prefix='/api/v2/orchestration')

# Global orchestration instances
testing_orchestrator = None
monitoring_coordinator = None

# Logger
logger = logging.getLogger("orchestration_api")


def get_testing_orchestrator():
    """Get or create testing orchestrator instance."""
    global testing_orchestrator
    if testing_orchestrator is None:
        config = {
            'auto_start_orchestration': True,
            'max_concurrent_executions': 6,
            'orchestration_interval': 15,
            'system_name': 'testmaster_orchestration'
        }
        testing_orchestrator = AdvancedTestingOrchestrator(config)
    return testing_orchestrator


def get_monitoring_coordinator():
    """Get or create monitoring coordinator instance."""
    global monitoring_coordinator
    if monitoring_coordinator is None:
        config = {
            'auto_start_coordination': True,
            'coordination_interval': 20,
            'system_name': 'testmaster_coordination'
        }
        monitoring_coordinator = AdvancedMonitoringCoordinator(config)
    return monitoring_coordinator


# === ORCHESTRATION STATUS ENDPOINTS ===

@orchestration_api.route('/status', methods=['GET'])
def get_orchestration_status():
    """
    Get comprehensive orchestration status.
    
    Returns:
        JSON with testing and monitoring orchestration status
    """
    try:
        testing_orch = get_testing_orchestrator()
        monitoring_coord = get_monitoring_coordinator()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'api_version': '2.0.0',
            'testing_orchestration': testing_orch.get_orchestration_status(),
            'monitoring_coordination': monitoring_coord.get_coordination_status(),
            'overall_health': {
                'testing_active': testing_orch._orchestration_active,
                'monitoring_active': monitoring_coord._coordination_active,
                'total_active_systems': 2
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': status
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get orchestration status: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/health', methods=['GET'])
def orchestration_health_check():
    """Quick orchestration health check."""
    try:
        testing_orch = get_testing_orchestrator()
        monitoring_coord = get_monitoring_coordinator()
        
        testing_active = testing_orch._orchestration_active
        monitoring_active = monitoring_coord._coordination_active
        
        if testing_active and monitoring_active:
            health_status = 'healthy'
        elif testing_active or monitoring_active:
            health_status = 'partial'
        else:
            health_status = 'inactive'
        
        return jsonify({
            'status': health_status,
            'testing_orchestration': 'active' if testing_active else 'inactive',
            'monitoring_coordination': 'active' if monitoring_active else 'inactive',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Orchestration health check failed: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


# === TESTING ORCHESTRATION ENDPOINTS ===

@orchestration_api.route('/testing/plans', methods=['POST'])
def create_intelligent_test_plan():
    """
    Create AI-optimized test plan.
    
    Request body:
    {
        "name": "Critical System Tests",
        "priority": "high",
        "environment": "system",
        "complexity": "high",
        "performance_critical": true,
        "security_critical": false,
        "urgency": "high",
        "include_unit": true,
        "include_integration": true,
        "dependencies": ["plan_123"]
    }
    
    Returns:
        Created test plan with AI optimizations
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        testing_orch = get_testing_orchestrator()
        test_plan = testing_orch.create_intelligent_test_plan(data)
        
        plan_data = {
            'plan_id': test_plan.plan_id,
            'name': test_plan.name,
            'strategy': test_plan.strategy.value,
            'priority': test_plan.priority.value,
            'environment': test_plan.environment.value,
            'test_suites': test_plan.test_suites,
            'estimated_duration': test_plan.estimated_duration,
            'resource_requirements': test_plan.resource_requirements,
            'dependencies': test_plan.dependencies,
            'success_criteria': test_plan.success_criteria,
            'created_at': test_plan.created_at.isoformat(),
            'ai_optimized': test_plan.metadata.get('ai_optimized', False)
        }
        
        return jsonify({
            'status': 'success',
            'data': plan_data
        }), 201
        
    except Exception as e:
        logger.error(f"Failed to create test plan: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/testing/plans/<plan_id>/schedule', methods=['POST'])
def schedule_test_execution():
    """
    Schedule test execution for a plan.
    
    Request body:
    {
        "scheduled_time": "2024-01-01T12:00:00Z" (optional)
    }
    
    Returns:
        Execution ID and scheduling confirmation
    """
    try:
        plan_id = request.view_args['plan_id']
        data = request.get_json() or {}
        
        scheduled_time = None
        if data.get('scheduled_time'):
            scheduled_time = datetime.fromisoformat(data['scheduled_time'].replace('Z', '+00:00'))
        
        testing_orch = get_testing_orchestrator()
        execution_id = testing_orch.schedule_test_execution(plan_id, scheduled_time)
        
        return jsonify({
            'status': 'success',
            'data': {
                'execution_id': execution_id,
                'plan_id': plan_id,
                'scheduled_time': scheduled_time.isoformat() if scheduled_time else None,
                'status': 'scheduled'
            }
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to schedule test execution: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/testing/executions/<execution_id>', methods=['GET'])
def get_test_execution_status():
    """Get test execution status and results."""
    try:
        execution_id = request.view_args['execution_id']
        testing_orch = get_testing_orchestrator()
        
        if execution_id not in testing_orch._test_executions:
            return jsonify({'error': 'Execution not found'}), 404
        
        execution = testing_orch._test_executions[execution_id]
        
        execution_data = {
            'execution_id': execution.execution_id,
            'plan_id': execution.plan_id,
            'status': execution.status,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'duration': execution.duration,
            'results': execution.results,
            'performance_metrics': execution.performance_metrics,
            'error_details': execution.error_details
        }
        
        return jsonify({
            'status': 'success',
            'data': execution_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get execution status: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/testing/insights', methods=['GET'])
def get_testing_insights():
    """
    Get AI-generated testing insights.
    
    Query parameters:
    - category: Filter by category (optimization, failure_prediction, resource_efficiency)
    - limit: Maximum insights to return (default: 20)
    
    Returns:
        List of testing insights with AI recommendations
    """
    try:
        category = request.args.get('category')
        limit = request.args.get('limit', 20, type=int)
        
        testing_orch = get_testing_orchestrator()
        insights = testing_orch.get_test_insights(category=category, limit=limit)
        
        insights_data = [
            {
                'insight_id': insight.insight_id,
                'category': insight.category,
                'confidence': insight.confidence,
                'title': insight.title,
                'description': insight.description,
                'impact_assessment': insight.impact_assessment,
                'recommended_actions': insight.recommended_actions,
                'data_sources': insight.data_sources,
                'timestamp': insight.timestamp.isoformat()
            }
            for insight in insights
        ]
        
        return jsonify({
            'status': 'success',
            'data': insights_data,
            'count': len(insights_data),
            'filters_applied': {
                'category': category,
                'limit': limit
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get testing insights: {e}")
        return jsonify({'error': str(e)}), 500


# === MONITORING COORDINATION ENDPOINTS ===

@orchestration_api.route('/monitoring/plans', methods=['POST'])
def create_monitoring_plan():
    """
    Create intelligent monitoring plan.
    
    Request body:
    {
        "name": "System Monitoring Plan",
        "strategy": "adaptive",
        "scope": "system",
        "targets": ["cpu", "memory", "disk", "network"],
        "interval": 30,
        "retention_days": 30,
        "thresholds": {"cpu_usage": 80, "memory_usage": 75},
        "correlation_rules": []
    }
    
    Returns:
        Created monitoring plan
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        monitoring_coord = get_monitoring_coordinator()
        monitoring_plan = monitoring_coord.create_monitoring_plan(data)
        
        plan_data = {
            'plan_id': monitoring_plan.plan_id,
            'name': monitoring_plan.name,
            'strategy': monitoring_plan.strategy.value,
            'scope': monitoring_plan.scope.value,
            'monitoring_targets': monitoring_plan.monitoring_targets,
            'collection_interval': monitoring_plan.collection_interval,
            'retention_period': monitoring_plan.retention_period.total_seconds(),
            'alert_thresholds': monitoring_plan.alert_thresholds,
            'correlation_rules': monitoring_plan.correlation_rules,
            'created_at': monitoring_plan.created_at.isoformat(),
            'active': monitoring_plan.active
        }
        
        return jsonify({
            'status': 'success',
            'data': plan_data
        }), 201
        
    except Exception as e:
        logger.error(f"Failed to create monitoring plan: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/monitoring/health', methods=['GET'])
def get_system_health():
    """
    Get comprehensive system health profile.
    
    Returns:
        Latest system health analysis with predictive indicators
    """
    try:
        monitoring_coord = get_monitoring_coordinator()
        
        if not monitoring_coord._health_profiles:
            return jsonify({
                'status': 'success',
                'data': {
                    'message': 'No health profiles available yet',
                    'profiles_count': 0
                }
            }), 200
        
        latest_profile = monitoring_coord._health_profiles[-1]
        
        health_data = {
            'profile_id': latest_profile.profile_id,
            'system_name': latest_profile.system_name,
            'timestamp': latest_profile.timestamp.isoformat(),
            'overall_health_score': latest_profile.overall_health_score,
            'dimension_scores': latest_profile.dimension_scores,
            'active_alerts_count': len(latest_profile.active_alerts),
            'trend_analysis': latest_profile.trend_analysis,
            'predictive_indicators': latest_profile.predictive_indicators,
            'resource_efficiency': latest_profile.resource_efficiency,
            'availability_metrics': latest_profile.availability_metrics
        }
        
        return jsonify({
            'status': 'success',
            'data': health_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/monitoring/alerts/correlated', methods=['GET'])
def get_correlated_alerts():
    """
    Get correlated alerts with AI analysis.
    
    Query parameters:
    - severity: Filter by severity (info, warning, error, critical, emergency)
    - limit: Maximum alerts to return (default: 20)
    - resolved: Include resolved alerts (default: false)
    
    Returns:
        List of correlated alerts with context and recommendations
    """
    try:
        severity_filter = request.args.get('severity')
        limit = request.args.get('limit', 20, type=int)
        include_resolved = request.args.get('resolved', 'false').lower() == 'true'
        
        monitoring_coord = get_monitoring_coordinator()
        correlated_alerts = list(monitoring_coord._correlated_alerts.values())
        
        # Apply filters
        if not include_resolved:
            correlated_alerts = [a for a in correlated_alerts if a.resolved_at is None]
        
        if severity_filter:
            try:
                severity_enum = AlertSeverity(severity_filter)
                correlated_alerts = [a for a in correlated_alerts if a.severity == severity_enum]
            except ValueError:
                return jsonify({'error': f'Invalid severity: {severity_filter}'}), 400
        
        # Sort by severity and creation time
        severity_priority = {
            AlertSeverity.EMERGENCY: 5,
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.ERROR: 3,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 1
        }
        
        correlated_alerts.sort(
            key=lambda a: (severity_priority.get(a.severity, 0), a.created_at),
            reverse=True
        )
        
        # Limit results
        correlated_alerts = correlated_alerts[:limit]
        
        alerts_data = [
            {
                'alert_id': alert.alert_id,
                'correlation_id': alert.correlation_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'affected_systems': alert.affected_systems,
                'root_cause_analysis': alert.root_cause_analysis,
                'impact_assessment': alert.impact_assessment,
                'recommended_actions': alert.recommended_actions,
                'confidence_score': alert.confidence_score,
                'created_at': alert.created_at.isoformat(),
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                'escalation_path': alert.escalation_path
            }
            for alert in correlated_alerts
        ]
        
        return jsonify({
            'status': 'success',
            'data': alerts_data,
            'count': len(alerts_data),
            'filters_applied': {
                'severity': severity_filter,
                'include_resolved': include_resolved,
                'limit': limit
            },
            'available_severities': [s.value for s in AlertSeverity]
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get correlated alerts: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/monitoring/insights', methods=['GET'])
def get_monitoring_insights():
    """
    Get AI-generated monitoring insights.
    
    Query parameters:
    - category: Filter by category (anomaly, trend, optimization, prediction)
    - limit: Maximum insights to return (default: 20)
    
    Returns:
        List of monitoring insights with AI recommendations
    """
    try:
        category = request.args.get('category')
        limit = request.args.get('limit', 20, type=int)
        
        monitoring_coord = get_monitoring_coordinator()
        insights = monitoring_coord.get_monitoring_insights(category=category, limit=limit)
        
        insights_data = [
            {
                'insight_id': insight.insight_id,
                'category': insight.category,
                'confidence': insight.confidence,
                'title': insight.title,
                'description': insight.description,
                'data_sources': insight.data_sources,
                'time_window': insight.time_window.total_seconds(),
                'predicted_impact': insight.predicted_impact,
                'recommended_adjustments': insight.recommended_adjustments,
                'timestamp': insight.timestamp.isoformat()
            }
            for insight in insights
        ]
        
        return jsonify({
            'status': 'success',
            'data': insights_data,
            'count': len(insights_data),
            'filters_applied': {
                'category': category,
                'limit': limit
            },
            'available_categories': ['anomaly', 'trend', 'optimization', 'prediction']
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get monitoring insights: {e}")
        return jsonify({'error': str(e)}), 500


# === CROSS-SYSTEM COORDINATION ENDPOINTS ===

@orchestration_api.route('/coordination/insights', methods=['GET'])
def get_cross_system_insights():
    """
    Get cross-system insights from both testing and monitoring.
    
    Returns:
        Combined insights with correlation analysis
    """
    try:
        testing_orch = get_testing_orchestrator()
        monitoring_coord = get_monitoring_coordinator()
        
        # Get insights from both systems
        testing_insights = testing_orch.get_test_insights(limit=10)
        monitoring_insights = monitoring_coord.get_monitoring_insights(limit=10)
        
        # Combine and analyze correlations
        combined_insights = {
            'testing_insights': [
                {
                    'insight_id': insight.insight_id,
                    'category': insight.category,
                    'confidence': insight.confidence,
                    'title': insight.title,
                    'description': insight.description,
                    'source': 'testing_orchestration'
                }
                for insight in testing_insights
            ],
            'monitoring_insights': [
                {
                    'insight_id': insight.insight_id,
                    'category': insight.category,
                    'confidence': insight.confidence,
                    'title': insight.title,
                    'description': insight.description,
                    'source': 'monitoring_coordination'
                }
                for insight in monitoring_insights
            ],
            'correlation_analysis': {
                'high_confidence_insights': len([
                    i for i in testing_insights + monitoring_insights
                    if i.confidence > 0.8
                ]),
                'cross_system_patterns': [],  # Would be populated by correlation analysis
                'recommended_actions': [
                    "Review high-confidence insights first",
                    "Correlate testing failures with monitoring anomalies",
                    "Consider system-wide optimization opportunities"
                ]
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': combined_insights,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get cross-system insights: {e}")
        return jsonify({'error': str(e)}), 500


# === CONTROL ENDPOINTS ===

@orchestration_api.route('/control/start', methods=['POST'])
def start_orchestration():
    """Start both testing and monitoring orchestration."""
    try:
        testing_orch = get_testing_orchestrator()
        monitoring_coord = get_monitoring_coordinator()
        
        testing_started = False
        monitoring_started = False
        
        if not testing_orch._orchestration_active:
            testing_orch.start_orchestration()
            testing_started = True
        
        if not monitoring_coord._coordination_active:
            monitoring_coord.start_coordination()
            monitoring_started = True
        
        return jsonify({
            'status': 'success',
            'message': 'Orchestration systems started',
            'testing_orchestration': {
                'active': testing_orch._orchestration_active,
                'started_now': testing_started
            },
            'monitoring_coordination': {
                'active': monitoring_coord._coordination_active,
                'started_now': monitoring_started
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to start orchestration: {e}")
        return jsonify({'error': str(e)}), 500


@orchestration_api.route('/control/stop', methods=['POST'])
def stop_orchestration():
    """Stop both testing and monitoring orchestration."""
    try:
        testing_orch = get_testing_orchestrator()
        monitoring_coord = get_monitoring_coordinator()
        
        testing_stopped = False
        monitoring_stopped = False
        
        if testing_orch._orchestration_active:
            testing_orch.stop_orchestration()
            testing_stopped = True
        
        if monitoring_coord._coordination_active:
            monitoring_coord.stop_coordination()
            monitoring_stopped = True
        
        return jsonify({
            'status': 'success',
            'message': 'Orchestration systems stopped',
            'testing_orchestration': {
                'active': testing_orch._orchestration_active,
                'stopped_now': testing_stopped
            },
            'monitoring_coordination': {
                'active': monitoring_coord._coordination_active,
                'stopped_now': monitoring_stopped
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to stop orchestration: {e}")
        return jsonify({'error': str(e)}), 500


# === DATA EXPORT ENDPOINTS ===

@orchestration_api.route('/export/comprehensive', methods=['GET'])
def export_comprehensive_data():
    """
    Export comprehensive orchestration data.
    
    Query parameters:
    - format: Export format (json) - default: json
    - include_testing: Include testing data (default: true)
    - include_monitoring: Include monitoring data (default: true)
    
    Returns:
        Comprehensive orchestration data export
    """
    try:
        export_format = request.args.get('format', 'json')
        include_testing = request.args.get('include_testing', 'true').lower() == 'true'
        include_monitoring = request.args.get('include_monitoring', 'true').lower() == 'true'
        
        data = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'export_format': export_format,
                'includes': {
                    'testing': include_testing,
                    'monitoring': include_monitoring
                }
            }
        }
        
        if include_testing:
            testing_orch = get_testing_orchestrator()
            testing_data = testing_orch.export_orchestration_data('dict')
            data['testing_orchestration'] = testing_data
        
        if include_monitoring:
            monitoring_coord = get_monitoring_coordinator()
            monitoring_data = monitoring_coord.export_coordination_data('dict')
            data['monitoring_coordination'] = monitoring_data
        
        if export_format == 'json':
            return Response(
                json.dumps(data, indent=2, default=str),
                mimetype='application/json',
                headers={
                    'Content-Disposition': f'attachment; filename=orchestration_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                }
            )
        else:
            return jsonify({'error': f'Format {export_format} not supported'}), 400
            
    except Exception as e:
        logger.error(f"Failed to export orchestration data: {e}")
        return jsonify({'error': str(e)}), 500


# Initialize function for app integration
def init_orchestration_api(app_config: Optional[Dict[str, Any]] = None):
    """Initialize orchestration API with configuration."""
    global testing_orchestrator, monitoring_coordinator
    
    if testing_orchestrator is None:
        config = app_config or {}
        config.setdefault('auto_start_orchestration', True)
        config.setdefault('system_name', 'testmaster_orchestration_api')
        testing_orchestrator = AdvancedTestingOrchestrator(config)
    
    if monitoring_coordinator is None:
        config = app_config or {}
        config.setdefault('auto_start_coordination', True)
        config.setdefault('system_name', 'testmaster_coordination_api')
        monitoring_coordinator = AdvancedMonitoringCoordinator(config)
    
    logger.info("Orchestration API initialized successfully")


# Export blueprint
__all__ = ['orchestration_api', 'init_orchestration_api']