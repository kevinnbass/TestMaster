"""
TestMaster Intelligence Hub REST API Endpoints
==============================================

Comprehensive REST API exposure for all intelligence hub capabilities.
Provides unified access to analytics, testing, and integration features.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

# Import intelligence hub components
from .. import IntelligenceHub
from ..analytics import ConsolidatedAnalyticsHub
from ..testing import ConsolidatedTestingHub
from ..integration import ConsolidatedIntegrationHub
from ..base import UnifiedAnalysisType

# Create blueprint
intelligence_api = Blueprint('intelligence_api', __name__, url_prefix='/api/intelligence')

# Import additional API blueprints - AGENT B ENHANCED
from .testing_api import testing_api, init_testing_api
from .monitoring_api import monitoring_api, init_monitoring_api

# Initialize components (would be done in app initialization in production)
intelligence_hub = None
analytics_hub = None
testing_hub = None
integration_hub = None

# Logger
logger = logging.getLogger(__name__)


def init_intelligence_api(app_config: Optional[Dict[str, Any]] = None):
    """Initialize intelligence API with configuration."""
    global intelligence_hub, analytics_hub, testing_hub, integration_hub
    
    config = app_config or {}
    intelligence_hub = IntelligenceHub(config)
    analytics_hub = ConsolidatedAnalyticsHub(config)
    testing_hub = ConsolidatedTestingHub(config)
    integration_hub = ConsolidatedIntegrationHub(config)
    
    # Initialize additional API blueprints - AGENT B ENHANCEMENT
    init_testing_api(config)
    init_monitoring_api(config)
    
    logger.info("Intelligence API initialized with all hubs and enhanced testing/monitoring APIs")


# === MAIN INTELLIGENCE ENDPOINTS ===

@intelligence_api.route('/status', methods=['GET'])
def get_intelligence_status():
    """Get overall intelligence hub status."""
    try:
        status = {
            'hub_status': 'active',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'analytics': analytics_hub.get_analytics_intelligence() if analytics_hub else {},
                'testing': testing_hub.get_testing_intelligence() if testing_hub else {},
                'integration': integration_hub.get_integration_intelligence() if integration_hub else {}
            },
            'api_version': '2.0.0',
            'architecture': 'modularized'
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Failed to get intelligence status: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/analyze', methods=['POST'])
def analyze():
    """
    Unified analysis endpoint for all intelligence capabilities.
    
    Request body:
    {
        "analysis_type": "comprehensive|statistical|ml|testing|integration",
        "data": {...},
        "options": {...}
    }
    """
    try:
        data = request.get_json()
        analysis_type = data.get('analysis_type', 'comprehensive')
        analysis_data = data.get('data', {})
        options = data.get('options', {})
        
        # Route to appropriate hub based on analysis type
        if analysis_type in ['testing', 'test']:
            result = testing_hub.execute_unified_test_analysis(
                analysis_data.get('test_results', []),
                analysis_type
            )
        elif analysis_type == 'integration':
            result = integration_hub.execute_unified_integration_analysis(
                analysis_data.get('systems'),
                analysis_type
            )
        else:
            # Default to analytics hub
            result = analytics_hub.analyze_metrics(
                analysis_data.get('metrics', []),
                UnifiedAnalysisType.COMPREHENSIVE if analysis_type == 'comprehensive' else UnifiedAnalysisType.STATISTICAL
            )
        
        # Convert result to JSON-serializable format
        response = {
            'analysis_id': result.analysis_id,
            'timestamp': result.timestamp.isoformat(),
            'type': result.analysis_type.value if hasattr(result.analysis_type, 'value') else str(result.analysis_type),
            'results': result.results,
            'confidence_score': result.confidence_score,
            'quality_metrics': result.quality_metrics
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


# === ANALYTICS ENDPOINTS ===

@intelligence_api.route('/analytics/analyze', methods=['POST'])
def analyze_metrics():
    """
    Analyze metrics with advanced analytics capabilities.
    
    Request body:
    {
        "metrics": [...],
        "analysis_type": "statistical|ml|predictive|anomaly",
        "enhanced_features": true
    }
    """
    try:
        data = request.get_json()
        metrics = data.get('metrics', [])
        analysis_type = data.get('analysis_type', 'statistical')
        enhanced = data.get('enhanced_features', True)
        
        # Map string to enum
        type_map = {
            'statistical': UnifiedAnalysisType.STATISTICAL,
            'ml': UnifiedAnalysisType.ML,
            'predictive': UnifiedAnalysisType.PREDICTIVE,
            'anomaly': UnifiedAnalysisType.ANOMALY
        }
        
        analysis = analytics_hub.analyze_metrics(
            metrics,
            type_map.get(analysis_type, UnifiedAnalysisType.STATISTICAL),
            enhanced_features=enhanced
        )
        
        return jsonify({
            'analysis_id': analysis.analysis_id,
            'results': analysis.results,
            'confidence_score': analysis.confidence_score
        }), 200
        
    except Exception as e:
        logger.error(f"Metrics analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/analytics/correlations', methods=['POST'])
def find_correlations():
    """Find correlations in metrics data."""
    try:
        data = request.get_json()
        metrics = data.get('metrics', [])
        threshold = data.get('threshold', 0.7)
        
        correlations = analytics_hub.find_correlations(metrics, threshold)
        
        return jsonify({'correlations': correlations}), 200
        
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/analytics/predict', methods=['POST'])
def predict_trends():
    """Predict future trends based on historical data."""
    try:
        data = request.get_json()
        historical_data = data.get('historical_data', [])
        prediction_horizon = data.get('horizon', 24)
        
        predictions = analytics_hub.predict_trends(
            historical_data,
            prediction_horizon
        )
        
        return jsonify({'predictions': predictions}), 200
        
    except Exception as e:
        logger.error(f"Trend prediction failed: {e}")
        return jsonify({'error': str(e)}), 500


# === TESTING ENDPOINTS ===

@intelligence_api.route('/testing/coverage', methods=['POST'])
def analyze_coverage():
    """
    Analyze test coverage with statistical insights.
    
    Request body:
    {
        "test_results": [...],
        "statistical_analysis": true,
        "confidence_level": 0.95
    }
    """
    try:
        data = request.get_json()
        test_results = data.get('test_results', [])
        statistical = data.get('statistical_analysis', True)
        confidence = data.get('confidence_level', 0.95)
        
        analysis = testing_hub.analyze_coverage(
            test_results,
            statistical_analysis=statistical,
            confidence_level=confidence
        )
        
        report = testing_hub.generate_coverage_report(analysis)
        
        return jsonify(report), 200
        
    except Exception as e:
        logger.error(f"Coverage analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/testing/optimize', methods=['POST'])
def optimize_tests():
    """
    Optimize test suite using ML.
    
    Request body:
    {
        "test_results": [...],
        "strategy": "comprehensive|latency|throughput|reliability"
    }
    """
    try:
        data = request.get_json()
        test_results = data.get('test_results', [])
        strategy = data.get('strategy', 'comprehensive')
        
        optimization = testing_hub.optimize_test_suite(
            test_results,
            optimization_strategy=strategy
        )
        
        return jsonify(optimization), 200
        
    except Exception as e:
        logger.error(f"Test optimization failed: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/testing/predict-failures', methods=['POST'])
def predict_failures():
    """Predict test failure probabilities."""
    try:
        data = request.get_json()
        test_ids = data.get('test_identifiers', [])
        historical_data = data.get('historical_data')
        
        predictions = testing_hub.predict_test_failures(
            test_ids,
            historical_data
        )
        
        return jsonify({'predictions': predictions}), 200
        
    except Exception as e:
        logger.error(f"Failure prediction failed: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/testing/generate', methods=['POST'])
def generate_tests():
    """Generate integration tests."""
    try:
        data = request.get_json()
        components = data.get('system_components', [])
        api_endpoints = data.get('api_endpoints')
        complexity = data.get('complexity_level', 'medium')
        
        tests = testing_hub.generate_integration_tests(
            components,
            api_endpoints,
            complexity
        )
        
        # Convert tests to JSON-serializable format
        test_list = []
        for test in tests:
            test_dict = {
                'test_id': test.test_id,
                'test_name': test.test_name,
                'test_type': test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type),
                'complexity_level': getattr(test, 'complexity_level', 'medium'),
                'estimated_execution_time': getattr(test, 'estimated_execution_time', 0)
            }
            test_list.append(test_dict)
        
        return jsonify({'generated_tests': test_list}), 200
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return jsonify({'error': str(e)}), 500


# === INTEGRATION ENDPOINTS ===

@intelligence_api.route('/integration/systems/analyze', methods=['POST'])
def analyze_systems():
    """
    Analyze cross-system performance.
    
    Request body:
    {
        "systems": ["system1", "system2"],
        "time_window_hours": 24,
        "include_correlations": true
    }
    """
    try:
        data = request.get_json()
        systems = data.get('systems', [])
        time_window = data.get('time_window_hours', 24)
        correlations = data.get('include_correlations', True)
        
        analysis = integration_hub.analyze_cross_system_performance(
            systems,
            time_window,
            correlations
        )
        
        # Convert to JSON-serializable format
        response = {
            'analysis_id': analysis.analysis_id,
            'timestamp': analysis.timestamp.isoformat(),
            'systems_analyzed': analysis.systems_analyzed,
            'health_scores': analysis.system_health_scores,
            'correlations': analysis.system_correlations,
            'bottlenecks': analysis.bottleneck_analysis,
            'optimization_opportunities': analysis.optimization_opportunities
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"System analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/integration/endpoints', methods=['GET'])
def get_endpoints():
    """Get all registered integration endpoints."""
    try:
        endpoints = integration_hub.endpoint_manager.get_all_endpoints()
        
        endpoint_list = []
        for ep_id, endpoint in endpoints.items():
            endpoint_list.append({
                'endpoint_id': ep_id,
                'name': endpoint.name,
                'url': endpoint.url,
                'status': endpoint.status.value,
                'type': endpoint.integration_type.value,
                'availability': endpoint.availability_percentage
            })
        
        return jsonify({'endpoints': endpoint_list}), 200
        
    except Exception as e:
        logger.error(f"Failed to get endpoints: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/integration/endpoints/<endpoint_id>/health', methods=['GET'])
def get_endpoint_health(endpoint_id):
    """Get health status for a specific endpoint."""
    try:
        health = integration_hub.get_endpoint_health(endpoint_id)
        return jsonify(health), 200
        
    except Exception as e:
        logger.error(f"Failed to get endpoint health: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/integration/events/publish', methods=['POST'])
def publish_event():
    """
    Publish integration event.
    
    Request body:
    {
        "event_id": "...",
        "source_system": "...",
        "event_type": "...",
        "payload": {...}
    }
    """
    try:
        data = request.get_json()
        
        # Create event from request data
        from ..integration.base import IntegrationEvent
        event = IntegrationEvent(
            event_id=data.get('event_id', str(datetime.now().timestamp())),
            timestamp=datetime.now(),
            source_system=data.get('source_system', 'api'),
            target_system=data.get('target_system'),
            event_type=data.get('event_type', 'api_event'),
            payload=data.get('payload', {})
        )
        
        success = integration_hub.publish_integration_event(event)
        
        return jsonify({'success': success, 'event_id': event.event_id}), 200
        
    except Exception as e:
        logger.error(f"Failed to publish event: {e}")
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/integration/performance', methods=['GET'])
def get_performance():
    """Get integration performance metrics."""
    try:
        endpoint_id = request.args.get('endpoint_id')
        time_window = int(request.args.get('time_window_hours', 1))
        
        metrics = integration_hub.get_integration_performance_metrics(
            endpoint_id,
            time_window
        )
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return jsonify({'error': str(e)}), 500


# === REAL-TIME MONITORING ENDPOINTS ===

@intelligence_api.route('/monitoring/realtime', methods=['GET'])
def get_realtime_metrics():
    """Get real-time metrics stream."""
    try:
        # Get current metrics from all hubs
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'analytics': {
                'active_analyses': len(getattr(analytics_hub, '_analysis_cache', {})),
                'metrics_processed': getattr(analytics_hub, '_metrics_processed', 0)
            },
            'testing': {
                'active_tests': len(getattr(testing_hub.execution_engine, '_active_executions', set())),
                'execution_stats': testing_hub.execution_engine.get_execution_statistics()
            },
            'integration': {
                'connected_endpoints': len(integration_hub.endpoint_manager.get_connected_endpoints()),
                'event_queue_size': len(getattr(integration_hub.event_processor, '_event_queue', [])),
                'performance_grade': integration_hub.performance_monitor._calculate_performance_grade()
            }
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Failed to get real-time metrics: {e}")
        return jsonify({'error': str(e)}), 500


# === BATCH OPERATIONS ===

@intelligence_api.route('/batch/analyze', methods=['POST'])
def batch_analyze():
    """
    Perform batch analysis across multiple domains.
    
    Request body:
    {
        "analyses": [
            {"type": "analytics", "data": {...}},
            {"type": "testing", "data": {...}},
            {"type": "integration", "data": {...}}
        ]
    }
    """
    try:
        data = request.get_json()
        analyses = data.get('analyses', [])
        
        results = []
        for analysis_request in analyses:
            analysis_type = analysis_request.get('type')
            analysis_data = analysis_request.get('data', {})
            
            try:
                if analysis_type == 'analytics':
                    result = analytics_hub.analyze_metrics(
                        analysis_data.get('metrics', [])
                    )
                elif analysis_type == 'testing':
                    result = testing_hub.execute_unified_test_analysis(
                        analysis_data.get('test_results', [])
                    )
                elif analysis_type == 'integration':
                    result = integration_hub.execute_unified_integration_analysis(
                        analysis_data.get('systems')
                    )
                else:
                    result = {'error': f'Unknown analysis type: {analysis_type}'}
                
                results.append({
                    'type': analysis_type,
                    'success': 'error' not in result,
                    'result': result
                })
                
            except Exception as e:
                results.append({
                    'type': analysis_type,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({'batch_results': results}), 200
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


# === HEALTH CHECK ===

@intelligence_api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'analytics': analytics_hub is not None,
                'testing': testing_hub is not None,
                'integration': integration_hub is not None,
                'api': True
            }
        }
        
        # Check if all components are healthy
        if all(health['components'].values()):
            return jsonify(health), 200
        else:
            health['status'] = 'degraded'
            return jsonify(health), 503
            
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503


# Export blueprint
__all__ = ['intelligence_api', 'init_intelligence_api']