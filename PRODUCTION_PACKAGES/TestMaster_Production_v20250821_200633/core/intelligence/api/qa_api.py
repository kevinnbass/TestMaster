"""
Quality Assurance API Blueprint - AGENT B Hour 22-24 Enhancement  
=================================================================

Comprehensive REST API for quality assurance, validation, and quality management.
Integrates with UnifiedQAFramework for enterprise-grade quality assessment.

Features:
- Comprehensive quality assessment and profiling
- Quality dimension analysis and scoring
- AI-powered quality insights and recommendations
- Quality trend analysis and prediction
- Validation and benchmarking management
- Quality reporting and export capabilities
"""

from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
import uuid

from ..monitoring.unified_qa_framework import (
    UnifiedQAFramework, QualityDimension, QualityRisk, QualityInsight,
    QualityTrendAnalysis, QualityProfile
)

# Create blueprint
qa_api = Blueprint('qa_api', __name__, url_prefix='/api/v2/quality')

# Global QA framework instance
qa_framework = None

# Logger
logger = logging.getLogger("qa_api")


def get_qa_framework():
    """Get or create QA framework instance."""
    global qa_framework
    if qa_framework is None:
        config = {
            'auto_start_qa': True,
            'qa_monitoring_interval': 30,
            'system_name': 'testmaster_qa_api'
        }
        qa_framework = UnifiedQAFramework(config)
    return qa_framework


# === QUALITY STATUS ENDPOINTS ===

@qa_api.route('/status', methods=['GET'])
def get_quality_status():
    """
    Get comprehensive quality status.
    
    Returns:
        JSON with quality metrics, scores, and system health
    """
    try:
        framework = get_qa_framework()
        summary = framework.get_quality_summary()
        
        return jsonify({
            'status': 'success',
            'data': summary,
            'api_version': '2.0.0',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get quality status: {e}")
        return jsonify({'error': str(e)}), 500


@qa_api.route('/health', methods=['GET'])
def quality_health_check():
    """Quick quality health check."""
    try:
        framework = get_qa_framework()
        summary = framework.get_quality_summary()
        
        current_quality = summary.get('current_quality', {})
        overall_score = current_quality.get('overall_score', 0)
        
        if overall_score >= 90:
            health_status = 'excellent'
        elif overall_score >= 80:
            health_status = 'good'
        elif overall_score >= 70:
            health_status = 'satisfactory'
        elif overall_score >= 60:
            health_status = 'poor'
        else:
            health_status = 'critical'
        
        return jsonify({
            'status': health_status,
            'overall_score': overall_score,
            'quality_level': current_quality.get('quality_level', 'unknown'),
            'actionable_insights': summary.get('actionable_insights', 0),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Quality health check failed: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


# === QUALITY ASSESSMENT ENDPOINTS ===

@qa_api.route('/assess', methods=['POST'])
def assess_system_quality():
    """
    Assess quality of a system or component.
    
    Request body:
    {
        "system_name": "user_authentication",
        "system_data": {"endpoints": 5, "tests": 20, "coverage": 85},
        "assessment_type": "comprehensive"
    }
    
    Returns:
        Comprehensive quality assessment profile
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        system_name = data.get('system_name', 'unknown_system')
        system_data = data.get('system_data', {})
        assessment_type = data.get('assessment_type', 'standard')
        
        framework = get_qa_framework()
        
        # Perform quality assessment
        quality_profile = framework.assess_system_quality(system_data)
        
        if not quality_profile:
            return jsonify({
                'status': 'error',
                'message': 'Failed to generate quality profile'
            }), 500
        
        # Convert profile to JSON-serializable format
        profile_data = {
            'profile_id': quality_profile.profile_id,
            'system_name': system_name,
            'timestamp': quality_profile.timestamp.isoformat(),
            'overall_quality_score': quality_profile.overall_quality_score,
            'quality_level': quality_profile.quality_level.value,
            'dimension_scores': {
                dim.value: score for dim, score in quality_profile.dimension_scores.items()
            },
            'risk_assessment': {
                risk.value: count for risk, count in quality_profile.risk_assessment.items()
            },
            'insights': [
                {
                    'insight_id': insight.insight_id,
                    'dimension': insight.dimension.value,
                    'risk_level': insight.risk_level.value,
                    'title': insight.title,
                    'description': insight.description,
                    'recommended_actions': insight.recommended_actions,
                    'confidence_score': insight.confidence_score
                }
                for insight in quality_profile.insights
            ],
            'validation_results_count': len(quality_profile.validation_results),
            'benchmark_results_count': len(quality_profile.benchmark_results)
        }
        
        # Add trend analysis if available
        if quality_profile.trend_analysis:
            profile_data['trend_analysis'] = {
                'overall_trend': quality_profile.trend_analysis.overall_trend,
                'trend_strength': quality_profile.trend_analysis.trend_strength,
                'predictive_alerts': quality_profile.trend_analysis.predictive_alerts
            }
        
        return jsonify({
            'status': 'success',
            'data': profile_data,
            'assessment_type': assessment_type
        }), 200
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return jsonify({'error': str(e)}), 500


@qa_api.route('/dimensions', methods=['GET'])
def get_quality_dimensions():
    """
    Get available quality dimensions and their standards.
    
    Returns:
        List of quality dimensions with standards and descriptions
    """
    try:
        framework = get_qa_framework()
        
        dimensions_data = []
        for dimension in QualityDimension:
            standards = framework._quality_standards.get(dimension, {})
            
            dimension_info = {
                'dimension': dimension.value,
                'name': dimension.value.replace('_', ' ').title(),
                'description': _get_dimension_description(dimension),
                'standards': standards,
                'importance_weight': _get_dimension_weight(dimension)
            }
            dimensions_data.append(dimension_info)
        
        return jsonify({
            'status': 'success',
            'data': {
                'dimensions': dimensions_data,
                'total_dimensions': len(dimensions_data)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get quality dimensions: {e}")
        return jsonify({'error': str(e)}), 500


def _get_dimension_description(dimension: QualityDimension) -> str:
    """Get description for quality dimension."""
    descriptions = {
        QualityDimension.CORRECTNESS: "Accuracy and precision of system behavior and outputs",
        QualityDimension.COMPLETENESS: "Coverage and thoroughness of features and functionality", 
        QualityDimension.CONSISTENCY: "Uniformity and coherence across system components",
        QualityDimension.RELIABILITY: "System stability, availability, and fault tolerance",
        QualityDimension.MAINTAINABILITY: "Ease of modification, extension, and debugging",
        QualityDimension.TESTABILITY: "Ease of testing and validation of system behavior",
        QualityDimension.PERFORMANCE: "Efficiency, speed, and resource utilization",
        QualityDimension.SECURITY: "Protection against threats and vulnerabilities",
        QualityDimension.USABILITY: "Ease of use and user experience quality"
    }
    return descriptions.get(dimension, "Quality dimension")


def _get_dimension_weight(dimension: QualityDimension) -> float:
    """Get importance weight for quality dimension."""
    weights = {
        QualityDimension.CORRECTNESS: 0.25,
        QualityDimension.RELIABILITY: 0.20,
        QualityDimension.PERFORMANCE: 0.15,
        QualityDimension.SECURITY: 0.15,
        QualityDimension.MAINTAINABILITY: 0.10,
        QualityDimension.COMPLETENESS: 0.05,
        QualityDimension.CONSISTENCY: 0.05,
        QualityDimension.TESTABILITY: 0.03,
        QualityDimension.USABILITY: 0.02
    }
    return weights.get(dimension, 0.01)


# === QUALITY INSIGHTS ENDPOINTS ===

@qa_api.route('/insights', methods=['GET'])
def get_quality_insights():
    """
    Get quality insights and recommendations.
    
    Query parameters:
    - risk_level: Filter by risk level (critical, high, medium, low, negligible)
    - dimension: Filter by quality dimension
    - limit: Maximum insights to return (default: 20)
    
    Returns:
        List of quality insights with recommendations
    """
    try:
        risk_level_param = request.args.get('risk_level')
        dimension_param = request.args.get('dimension') 
        limit = request.args.get('limit', 20, type=int)
        
        # Parse filters
        risk_filter = None
        if risk_level_param:
            try:
                risk_filter = QualityRisk(risk_level_param)
            except ValueError:
                return jsonify({'error': f'Invalid risk level: {risk_level_param}'}), 400
        
        dimension_filter = None
        if dimension_param:
            try:
                dimension_filter = QualityDimension(dimension_param)
            except ValueError:
                return jsonify({'error': f'Invalid dimension: {dimension_param}'}), 400
        
        framework = get_qa_framework()
        insights = framework.get_quality_insights(
            risk_filter=risk_filter,
            dimension_filter=dimension_filter,
            limit=limit
        )
        
        insights_data = [
            {
                'insight_id': insight.insight_id,
                'dimension': insight.dimension.value,
                'risk_level': insight.risk_level.value,
                'title': insight.title,
                'description': insight.description,
                'impact_assessment': insight.impact_assessment,
                'recommended_actions': insight.recommended_actions,
                'confidence_score': insight.confidence_score,
                'data_points': insight.data_points,
                'timestamp': insight.timestamp.isoformat()
            }
            for insight in insights
        ]
        
        return jsonify({
            'status': 'success',
            'data': insights_data,
            'count': len(insights_data),
            'filters_applied': {
                'risk_level': risk_level_param,
                'dimension': dimension_param,
                'limit': limit
            },
            'available_filters': {
                'risk_levels': [risk.value for risk in QualityRisk],
                'dimensions': [dim.value for dim in QualityDimension]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get quality insights: {e}")
        return jsonify({'error': str(e)}), 500


@qa_api.route('/insights/critical', methods=['GET'])
def get_critical_insights():
    """Get critical quality insights requiring immediate attention."""
    try:
        framework = get_qa_framework()
        critical_insights = framework.get_quality_insights(
            risk_filter=QualityRisk.CRITICAL,
            limit=10
        )
        
        high_insights = framework.get_quality_insights(
            risk_filter=QualityRisk.HIGH,
            limit=15
        )
        
        insights_data = []
        for insights_list, priority in [(critical_insights, 'critical'), (high_insights, 'high')]:
            for insight in insights_list:
                insight_data = {
                    'insight_id': insight.insight_id,
                    'priority': priority,
                    'dimension': insight.dimension.value,
                    'title': insight.title,
                    'description': insight.description,
                    'impact_assessment': insight.impact_assessment,
                    'recommended_actions': insight.recommended_actions[:3],  # Top 3 actions
                    'confidence_score': insight.confidence_score
                }
                insights_data.append(insight_data)
        
        return jsonify({
            'status': 'success',
            'data': insights_data,
            'critical_count': len(critical_insights),
            'high_priority_count': len(high_insights),
            'total_urgent_issues': len(insights_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get critical insights: {e}")
        return jsonify({'error': str(e)}), 500


# === QUALITY TRENDS ENDPOINTS ===

@qa_api.route('/trends', methods=['GET'])
def get_quality_trends():
    """
    Get quality trends analysis.
    
    Query parameters:
    - window: Time window in hours for trend analysis (default: 24)
    - dimension: Specific dimension to analyze (optional)
    
    Returns:
        Quality trend analysis with predictions
    """
    try:
        window_hours = request.args.get('window', 24, type=int)
        dimension_param = request.args.get('dimension')
        
        framework = get_qa_framework()
        
        # Get recent profiles for trend analysis
        if not framework._quality_profiles:
            return jsonify({
                'status': 'success',
                'data': {
                    'message': 'No quality data available for trend analysis',
                    'profiles_count': 0
                }
            }), 200
        
        # Get the latest trend analysis from the most recent profile
        latest_profile = framework._quality_profiles[-1]
        trend_analysis = latest_profile.trend_analysis
        
        if not trend_analysis:
            return jsonify({
                'status': 'success',
                'data': {
                    'message': 'Trend analysis not available - insufficient data points',
                    'profiles_count': len(framework._quality_profiles)
                }
            }), 200
        
        trend_data = {
            'analysis_id': trend_analysis.analysis_id,
            'time_window_hours': window_hours,
            'overall_trend': trend_analysis.overall_trend,
            'trend_strength': trend_analysis.trend_strength,
            'dimension_trends': {
                dim.value: trend for dim, trend in trend_analysis.dimension_trends.items()
            },
            'risk_factors': trend_analysis.risk_factors,
            'predictive_alerts': trend_analysis.predictive_alerts,
            'confidence_interval': {
                'lower': trend_analysis.confidence_interval[0],
                'upper': trend_analysis.confidence_interval[1]
            },
            'profiles_analyzed': len(framework._quality_profiles)
        }
        
        # Add dimension-specific analysis if requested
        if dimension_param:
            try:
                dimension = QualityDimension(dimension_param)
                recent_profiles = list(framework._quality_profiles)[-10:]
                
                dimension_scores = [
                    p.dimension_scores.get(dimension, 75.0) 
                    for p in recent_profiles
                ]
                
                trend_data['dimension_analysis'] = {
                    'dimension': dimension.value,
                    'current_score': dimension_scores[-1] if dimension_scores else 0,
                    'average_score': sum(dimension_scores) / len(dimension_scores) if dimension_scores else 0,
                    'score_range': {
                        'min': min(dimension_scores) if dimension_scores else 0,
                        'max': max(dimension_scores) if dimension_scores else 0
                    },
                    'trend': trend_analysis.dimension_trends.get(dimension, 'stable')
                }
                
            except ValueError:
                return jsonify({'error': f'Invalid dimension: {dimension_param}'}), 400
        
        return jsonify({
            'status': 'success',
            'data': trend_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get quality trends: {e}")
        return jsonify({'error': str(e)}), 500


# === VALIDATION AND BENCHMARKING ENDPOINTS ===

@qa_api.route('/validate', methods=['POST'])
def validate_quality():
    """
    Validate quality of system or component data.
    
    Request body:
    {
        "data": {"key": "value", ...},
        "validation_rules": ["null_check", "format_consistency"],
        "strict_mode": false
    }
    
    Returns:
        Validation results with issues and recommendations
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Request body required'}), 400
        
        data_to_validate = request_data.get('data', {})
        requested_rules = request_data.get('validation_rules', [])
        strict_mode = request_data.get('strict_mode', False)
        
        framework = get_qa_framework()
        
        # Perform validation using framework
        validation_results = []
        
        # Use framework's validation rules or requested specific rules
        rules_to_execute = framework._validation_rules
        if requested_rules:
            rules_to_execute = [
                rule for rule in framework._validation_rules 
                if rule.rule_id in requested_rules
            ]
        
        for rule in rules_to_execute:
            try:
                result = framework._execute_validation_rule(rule, data_to_validate)
                validation_results.append(result)
            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                if strict_mode:
                    return jsonify({'error': f'Validation failed: {str(e)}'}), 500
        
        # Aggregate results
        total_validations = len(validation_results)
        passed_validations = sum(1 for r in validation_results if r.is_valid)
        overall_success = passed_validations == total_validations
        
        all_issues = []
        for result in validation_results:
            all_issues.extend(result.issues)
        
        response_data = {
            'overall_success': overall_success,
            'success_rate': (passed_validations / total_validations * 100) if total_validations > 0 else 100,
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': total_validations - passed_validations,
            'issues': [
                {
                    'issue_id': issue.issue_id,
                    'validation_type': issue.validation_type.value,
                    'severity': issue.severity,
                    'message': issue.message,
                    'location': issue.location,
                    'suggestion': issue.suggestion
                }
                for issue in all_issues
            ],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'data': response_data
        }), 200
        
    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        return jsonify({'error': str(e)}), 500


@qa_api.route('/benchmark', methods=['POST'])
def benchmark_quality():
    """
    Benchmark quality performance of operations.
    
    Request body:
    {
        "operation_name": "data_processing",
        "operation_data": {"items": 1000},
        "benchmark_type": "performance"
    }
    
    Returns:
        Benchmarking results with performance metrics
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        operation_name = data.get('operation_name', 'unknown_operation')
        operation_data = data.get('operation_data', {})
        benchmark_type = data.get('benchmark_type', 'performance')
        
        framework = get_qa_framework()
        
        # Perform benchmarking
        def test_operation():
            return framework._simulate_system_operation(operation_data)
        
        benchmark_result = framework.agent_qa.benchmark_performance(test_operation)
        
        result_data = {
            'operation_name': operation_name,
            'benchmark_type': benchmark_type,
            'benchmark_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {
                'value': benchmark_result.value,
                'baseline': benchmark_result.baseline,
                'threshold': benchmark_result.threshold,
                'passed': benchmark_result.passed,
                'improvement': benchmark_result.improvement
            },
            'metadata': benchmark_result.metadata
        }
        
        return jsonify({
            'status': 'success',
            'data': result_data
        }), 200
        
    except Exception as e:
        logger.error(f"Quality benchmarking failed: {e}")
        return jsonify({'error': str(e)}), 500


# === REPORTING ENDPOINTS ===

@qa_api.route('/report', methods=['GET'])
def generate_quality_report():
    """
    Generate comprehensive quality report.
    
    Query parameters:
    - format: Report format (json, summary) - default: json
    - include_details: Include detailed analysis (default: true)
    
    Returns:
        Comprehensive quality report
    """
    try:
        report_format = request.args.get('format', 'json')
        include_details = request.args.get('include_details', 'true').lower() == 'true'
        
        framework = get_qa_framework()
        
        if report_format == 'json':
            report_data = framework.export_quality_report('json')
            
            if include_details:
                return Response(
                    report_data,
                    mimetype='application/json',
                    headers={
                        'Content-Disposition': f'attachment; filename=quality_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    }
                )
            else:
                # Return summary only
                full_data = json.loads(report_data)
                summary_data = {
                    'report_metadata': full_data['report_metadata'],
                    'quality_summary': full_data['quality_summary'],
                    'insights_count': len(full_data.get('quality_insights', [])),
                    'profiles_count': len(full_data.get('recent_profiles', []))
                }
                return jsonify({
                    'status': 'success',
                    'data': summary_data
                }), 200
        else:
            return jsonify({'error': f'Format {report_format} not supported'}), 400
        
    except Exception as e:
        logger.error(f"Failed to generate quality report: {e}")
        return jsonify({'error': str(e)}), 500


# === MONITORING CONTROL ENDPOINTS ===

@qa_api.route('/monitoring/start', methods=['POST'])
def start_quality_monitoring():
    """Start quality monitoring."""
    try:
        framework = get_qa_framework()
        if not framework._monitoring_active:
            framework.start_quality_monitoring()
            return jsonify({
                'status': 'success',
                'message': 'Quality monitoring started',
                'monitoring_active': True
            }), 200
        else:
            return jsonify({
                'status': 'success',
                'message': 'Quality monitoring already active',
                'monitoring_active': True
            }), 200
            
    except Exception as e:
        logger.error(f"Failed to start quality monitoring: {e}")
        return jsonify({'error': str(e)}), 500


@qa_api.route('/monitoring/stop', methods=['POST'])
def stop_quality_monitoring():
    """Stop quality monitoring."""
    try:
        framework = get_qa_framework()
        if framework._monitoring_active:
            framework.stop_quality_monitoring()
            return jsonify({
                'status': 'success',
                'message': 'Quality monitoring stopped',
                'monitoring_active': False
            }), 200
        else:
            return jsonify({
                'status': 'success',
                'message': 'Quality monitoring already inactive',
                'monitoring_active': False
            }), 200
            
    except Exception as e:
        logger.error(f"Failed to stop quality monitoring: {e}")
        return jsonify({'error': str(e)}), 500


# === UTILITY ENDPOINTS ===

@qa_api.route('/config', methods=['GET'])
def get_quality_config():
    """Get quality framework configuration."""
    try:
        framework = get_qa_framework()
        
        config_data = {
            'monitoring_active': framework._monitoring_active,
            'framework_config': framework.config,
            'quality_standards': {
                dim.value: standards for dim, standards in framework._quality_standards.items()
            },
            'validation_rules_count': len(framework._validation_rules),
            'data_storage': {
                'quality_profiles': len(framework._quality_profiles),
                'quality_insights': len(framework._quality_insights),
                'validation_history': len(framework._validation_history),
                'benchmark_history': len(framework._benchmark_history)
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': config_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get quality config: {e}")
        return jsonify({'error': str(e)}), 500


# Initialize function for app integration
def init_qa_api(app_config: Optional[Dict[str, Any]] = None):
    """Initialize QA API with configuration."""
    global qa_framework
    if qa_framework is None:
        config = app_config or {}
        config.setdefault('auto_start_qa', True)
        config.setdefault('system_name', 'testmaster_qa_api')
        qa_framework = UnifiedQAFramework(config)
    
    logger.info("Quality Assurance API initialized successfully")


# Export blueprint
__all__ = ['qa_api', 'init_qa_api']