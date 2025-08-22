"""
Comprehensive Monitoring API
============================

Advanced monitoring endpoints that provide complete visibility into
all analytics processes with real-time updates and detailed insights.

Author: TestMaster Team
"""

import logging
import json
import time
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ComprehensiveMonitoringAPI:
    """Comprehensive monitoring API with enhanced analytics visibility."""
    
    def __init__(self, aggregator=None, delivery_guarantee=None, realtime_tracker=None, observability_metrics=None):
        """
        Initialize comprehensive monitoring API.
        
        Args:
            aggregator: Analytics aggregator instance
            delivery_guarantee: Delivery guarantee system
            realtime_tracker: Real-time analytics tracker
            observability_metrics: System observability metrics
        """
        self.aggregator = aggregator
        self.delivery_guarantee = delivery_guarantee
        self.realtime_tracker = realtime_tracker
        self.observability_metrics = observability_metrics
        
        self.blueprint = Blueprint('comprehensive_monitoring', __name__, url_prefix='/api/comprehensive')
        self._setup_routes()
        
        logger.info("Comprehensive Monitoring API initialized")
    
    def _setup_routes(self):
        """Setup comprehensive monitoring routes."""
        
        @self.blueprint.route('/overview', methods=['GET'])
        def system_overview():
            """Get comprehensive system overview."""
            try:
                overview = {
                    'timestamp': datetime.now().isoformat(),
                    'system_status': 'operational',
                    'components': {},
                    'metrics': {},
                    'alerts': [],
                    'performance': {}
                }
                
                # Analytics aggregator status
                if self.aggregator:
                    try:
                        robustness = self.aggregator.get_robustness_monitoring()
                        overview['components']['analytics_aggregator'] = {
                            'status': 'healthy' if robustness['overall_health_score'] > 70 else 'degraded',
                            'health_score': robustness['overall_health_score'],
                            'features': len(robustness.get('robustness_features', {})),
                            'last_check': robustness['timestamp']
                        }
                    except Exception as e:
                        overview['components']['analytics_aggregator'] = {
                            'status': 'error',
                            'error': str(e)
                        }
                
                # Delivery guarantee status
                if self.delivery_guarantee:
                    try:
                        guarantee_stats = self.delivery_guarantee.get_guarantee_statistics()
                        stats = guarantee_stats['statistics']
                        overview['components']['delivery_guarantee'] = {
                            'status': 'healthy' if stats['delivery_success_rate'] > 95 else 'degraded',
                            'success_rate': stats['delivery_success_rate'],
                            'pending_deliveries': stats['current_pending'],
                            'total_deliveries': stats['total_submissions']
                        }
                    except Exception as e:
                        overview['components']['delivery_guarantee'] = {
                            'status': 'error',
                            'error': str(e)
                        }
                
                # Real-time tracker status
                if self.realtime_tracker:
                    try:
                        tracker_summary = self.realtime_tracker.get_tracking_summary()
                        stats = tracker_summary['statistics']
                        overview['components']['realtime_tracker'] = {
                            'status': 'healthy',
                            'events_per_second': stats['events_per_second'],
                            'active_analytics': stats['active_analytics_count'],
                            'websocket_clients': tracker_summary['websocket_clients']
                        }
                    except Exception as e:
                        overview['components']['realtime_tracker'] = {
                            'status': 'error',
                            'error': str(e)
                        }
                
                # Observability metrics status
                if self.observability_metrics:
                    try:
                        health_score = self.observability_metrics.get_health_score()
                        overview['components']['observability_metrics'] = {
                            'status': 'healthy' if health_score['overall_score'] > 70 else 'degraded',
                            'overall_score': health_score['overall_score'],
                            'component_scores': health_score['component_scores']
                        }
                    except Exception as e:
                        overview['components']['observability_metrics'] = {
                            'status': 'error',
                            'error': str(e)
                        }
                
                # Calculate overall system health
                component_healths = []
                for comp_name, comp_data in overview['components'].items():
                    if comp_data.get('status') == 'healthy':
                        component_healths.append(100)
                    elif comp_data.get('status') == 'degraded':
                        component_healths.append(70)
                    else:
                        component_healths.append(0)
                
                if component_healths:
                    overall_health = sum(component_healths) / len(component_healths)
                    if overall_health >= 90:
                        overview['system_status'] = 'excellent'
                    elif overall_health >= 70:
                        overview['system_status'] = 'good'
                    elif overall_health >= 50:
                        overview['system_status'] = 'degraded'
                    else:
                        overview['system_status'] = 'critical'
                
                return jsonify({
                    'status': 'success',
                    'overview': overview
                }), 200
                
            except Exception as e:
                logger.error(f"System overview failed: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.blueprint.route('/analytics-flow', methods=['GET'])
        def analytics_flow_status():
            """Get detailed analytics flow status."""
            try:
                flow_data = {
                    'timestamp': datetime.now().isoformat(),
                    'flow_summary': {},
                    'active_analytics': {},
                    'delivery_status': {},
                    'performance_metrics': {}
                }
                
                # Get flow monitor data
                if self.aggregator and hasattr(self.aggregator, 'flow_monitor'):
                    flow_summary = self.aggregator.flow_monitor.get_flow_summary()
                    flow_data['flow_summary'] = flow_summary
                
                # Get real-time tracking data
                if self.realtime_tracker:
                    tracking_summary = self.realtime_tracker.get_tracking_summary()
                    flow_data['active_analytics'] = tracking_summary['active_analytics']
                    flow_data['performance_metrics'] = tracking_summary['statistics']
                
                # Get delivery guarantee data
                if self.delivery_guarantee:
                    guarantee_stats = self.delivery_guarantee.get_guarantee_statistics()
                    flow_data['delivery_status'] = guarantee_stats
                
                return jsonify({
                    'status': 'success',
                    'flow_data': flow_data
                }), 200
                
            except Exception as e:
                logger.error(f"Analytics flow status failed: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.blueprint.route('/analytics/<analytics_id>/journey', methods=['GET'])
        def analytics_journey(analytics_id):
            """Get complete journey of specific analytics."""
            try:
                journey_data = {
                    'analytics_id': analytics_id,
                    'found': False,
                    'journey': None,
                    'delivery_status': None,
                    'tracking_events': []
                }
                
                # Get journey from real-time tracker
                if self.realtime_tracker:
                    journey = self.realtime_tracker.get_analytics_journey(analytics_id)
                    if journey:
                        journey_data['found'] = True
                        journey_data['journey'] = journey
                        journey_data['tracking_events'] = journey.get('events', [])
                
                # Get delivery status
                if self.delivery_guarantee:
                    delivery_status = self.delivery_guarantee.get_delivery_status(analytics_id)
                    if delivery_status:
                        journey_data['delivery_status'] = delivery_status
                        if not journey_data['found']:
                            journey_data['found'] = True
                
                # Search in flow monitor if not found
                if not journey_data['found'] and self.aggregator and hasattr(self.aggregator, 'flow_monitor'):
                    # Try to find in recent transactions
                    flow_summary = self.aggregator.flow_monitor.get_flow_summary()
                    recent_transactions = flow_summary.get('recent_transactions', [])
                    
                    for transaction in recent_transactions:
                        if transaction.get('data', {}).get('analytics_id') == analytics_id:
                            journey_data['found'] = True
                            journey_data['journey'] = {
                                'analytics_id': analytics_id,
                                'status': 'completed',
                                'flow_data': transaction
                            }
                            break
                
                return jsonify({
                    'status': 'success',
                    'journey_data': journey_data
                }), 200
                
            except Exception as e:
                logger.error(f"Analytics journey lookup failed: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.blueprint.route('/components/<component>/performance', methods=['GET'])
        def component_performance(component):
            """Get detailed performance metrics for specific component."""
            try:
                performance_data = {
                    'component': component,
                    'found': False,
                    'metrics': {},
                    'recent_activity': [],
                    'health_status': 'unknown'
                }
                
                # Get from real-time tracker
                if self.realtime_tracker:
                    perf_data = self.realtime_tracker.get_component_performance(component)
                    if perf_data:
                        performance_data['found'] = True
                        performance_data['metrics'] = perf_data
                        performance_data['recent_activity'] = perf_data.get('recent_events', [])
                
                # Get from observability metrics
                if self.observability_metrics:
                    metric_summary = self.observability_metrics.get_metric_summary()
                    component_metrics = {}
                    
                    # Look for metrics related to this component
                    for metric_name, metric_data in metric_summary.get('metrics', {}).items():
                        if component.lower() in metric_name.lower():
                            component_metrics[metric_name] = metric_data
                    
                    if component_metrics:
                        performance_data['found'] = True
                        performance_data['observability_metrics'] = component_metrics
                
                # Determine health status
                if performance_data['found']:
                    metrics = performance_data.get('metrics', {})
                    error_count = metrics.get('error_count', 0)
                    events_processed = metrics.get('events_processed', 0)
                    
                    if events_processed > 0:
                        error_rate = (error_count / events_processed) * 100
                        if error_rate < 1:
                            performance_data['health_status'] = 'healthy'
                        elif error_rate < 5:
                            performance_data['health_status'] = 'degraded'
                        else:
                            performance_data['health_status'] = 'unhealthy'
                    else:
                        performance_data['health_status'] = 'inactive'
                
                return jsonify({
                    'status': 'success',
                    'performance_data': performance_data
                }), 200
                
            except Exception as e:
                logger.error(f"Component performance lookup failed: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.blueprint.route('/delivery-test', methods=['POST'])
        def comprehensive_delivery_test():
            """Perform comprehensive delivery test across all systems."""
            try:
                test_data = {
                    'test_id': f"comprehensive_test_{int(time.time())}",
                    'timestamp': datetime.now().isoformat(),
                    'test_type': 'comprehensive_delivery',
                    'test_data': {
                        'value': 42,
                        'message': 'Comprehensive delivery test'
                    }
                }
                
                results = {
                    'test_id': test_data['test_id'],
                    'timestamp': test_data['timestamp'],
                    'tests_performed': {},
                    'overall_success': False,
                    'success_count': 0,
                    'total_tests': 0
                }
                
                # Test analytics aggregator
                if self.aggregator:
                    try:
                        # Submit through aggregator
                        analytics_result = self.aggregator.aggregate_analytics(test_data)
                        results['tests_performed']['analytics_aggregator'] = {
                            'success': True,
                            'result': 'Analytics processed successfully',
                            'response_time': analytics_result.get('response_time_ms', 0)
                        }
                        results['success_count'] += 1
                    except Exception as e:
                        results['tests_performed']['analytics_aggregator'] = {
                            'success': False,
                            'error': str(e)
                        }
                    results['total_tests'] += 1
                
                # Test delivery guarantee
                if self.delivery_guarantee:
                    try:
                        from dashboard.dashboard_core.analytics_delivery_guarantee import DeliveryPriority
                        delivery_id = self.delivery_guarantee.submit_analytics(
                            test_data,
                            DeliveryPriority.HIGH
                        )
                        results['tests_performed']['delivery_guarantee'] = {
                            'success': True,
                            'delivery_id': delivery_id,
                            'message': 'Submitted for guaranteed delivery'
                        }
                        results['success_count'] += 1
                    except Exception as e:
                        results['tests_performed']['delivery_guarantee'] = {
                            'success': False,
                            'error': str(e)
                        }
                    results['total_tests'] += 1
                
                # Test real-time tracker
                if self.realtime_tracker:
                    try:
                        from dashboard.dashboard_core.realtime_analytics_tracker import TrackingEvent, TrackingPriority
                        tracking_id = self.realtime_tracker.track_event(
                            test_data['test_id'],
                            TrackingEvent.CREATED,
                            'comprehensive_test',
                            data_size=len(json.dumps(test_data)),
                            priority=TrackingPriority.HIGH,
                            metadata={'test_type': 'comprehensive'}
                        )
                        results['tests_performed']['realtime_tracker'] = {
                            'success': True,
                            'tracking_id': tracking_id,
                            'message': 'Event tracked successfully'
                        }
                        results['success_count'] += 1
                    except Exception as e:
                        results['tests_performed']['realtime_tracker'] = {
                            'success': False,
                            'error': str(e)
                        }
                    results['total_tests'] += 1
                
                # Test observability metrics
                if self.observability_metrics:
                    try:
                        from dashboard.dashboard_core.system_observability_metrics import MetricSeverity
                        self.observability_metrics.record_metric(
                            'comprehensive_test_metric',
                            42.0,
                            labels={'test_id': test_data['test_id']},
                            severity=MetricSeverity.INFO
                        )
                        results['tests_performed']['observability_metrics'] = {
                            'success': True,
                            'message': 'Metric recorded successfully'
                        }
                        results['success_count'] += 1
                    except Exception as e:
                        results['tests_performed']['observability_metrics'] = {
                            'success': False,
                            'error': str(e)
                        }
                    results['total_tests'] += 1
                
                # Calculate overall success
                if results['total_tests'] > 0:
                    success_rate = (results['success_count'] / results['total_tests']) * 100
                    results['overall_success'] = success_rate >= 75  # At least 75% success
                    results['success_rate'] = success_rate
                
                return jsonify({
                    'status': 'success',
                    'test_results': results
                }), 200
                
            except Exception as e:
                logger.error(f"Comprehensive delivery test failed: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.blueprint.route('/health-dashboard', methods=['GET'])
        def health_dashboard():
            """Get data for comprehensive health dashboard."""
            try:
                dashboard_data = {
                    'timestamp': datetime.now().isoformat(),
                    'overall_health': {
                        'score': 0,
                        'status': 'unknown',
                        'components': {}
                    },
                    'key_metrics': {},
                    'alerts': [],
                    'performance_trends': [],
                    'recent_activity': []
                }
                
                component_scores = []
                
                # Analytics aggregator health
                if self.aggregator:
                    try:
                        robustness = self.aggregator.get_robustness_monitoring()
                        score = robustness['overall_health_score']
                        component_scores.append(score)
                        
                        dashboard_data['overall_health']['components']['analytics'] = {
                            'score': score,
                            'status': 'healthy' if score > 70 else 'degraded',
                            'features': robustness.get('robustness_features', {})
                        }
                        
                        # Add key metrics
                        dashboard_data['key_metrics']['analytics_health_score'] = score
                        
                    except Exception as e:
                        logger.error(f"Failed to get analytics health: {e}")
                
                # Delivery guarantee health
                if self.delivery_guarantee:
                    try:
                        stats = self.delivery_guarantee.get_guarantee_statistics()
                        success_rate = stats['statistics']['delivery_success_rate']
                        component_scores.append(success_rate)
                        
                        dashboard_data['overall_health']['components']['delivery'] = {
                            'score': success_rate,
                            'status': 'healthy' if success_rate > 95 else 'degraded',
                            'pending': stats['statistics']['current_pending']
                        }
                        
                        dashboard_data['key_metrics']['delivery_success_rate'] = success_rate
                        dashboard_data['key_metrics']['pending_deliveries'] = stats['statistics']['current_pending']
                        
                    except Exception as e:
                        logger.error(f"Failed to get delivery health: {e}")
                
                # Real-time tracking health
                if self.realtime_tracker:
                    try:
                        summary = self.realtime_tracker.get_tracking_summary()
                        events_per_sec = summary['statistics']['events_per_second']
                        
                        # Calculate health based on activity
                        tracking_score = min(100, events_per_sec * 10)  # Up to 100 if 10+ events/sec
                        component_scores.append(tracking_score)
                        
                        dashboard_data['overall_health']['components']['tracking'] = {
                            'score': tracking_score,
                            'status': 'healthy' if tracking_score > 50 else 'low_activity',
                            'events_per_second': events_per_sec,
                            'active_analytics': summary['statistics']['active_analytics_count']
                        }
                        
                        dashboard_data['key_metrics']['events_per_second'] = events_per_sec
                        dashboard_data['key_metrics']['active_analytics'] = summary['statistics']['active_analytics_count']
                        
                        # Add recent activity
                        dashboard_data['recent_activity'] = summary.get('recent_events', [])[-10:]
                        
                    except Exception as e:
                        logger.error(f"Failed to get tracking health: {e}")
                
                # System observability health
                if self.observability_metrics:
                    try:
                        health_score = self.observability_metrics.get_health_score()
                        score = health_score['overall_score']
                        component_scores.append(score)
                        
                        dashboard_data['overall_health']['components']['system'] = {
                            'score': score,
                            'status': 'healthy' if score > 70 else 'degraded',
                            'component_scores': health_score['component_scores']
                        }
                        
                        dashboard_data['key_metrics']['system_health_score'] = score
                        
                        # Add performance trends
                        performance_data = self.observability_metrics.get_metric_data(
                            'cpu_usage_percent',
                            time_window=timedelta(minutes=30)
                        )
                        dashboard_data['performance_trends'] = performance_data[-20:]  # Last 20 points
                        
                    except Exception as e:
                        logger.error(f"Failed to get system health: {e}")
                
                # Calculate overall health
                if component_scores:
                    overall_score = sum(component_scores) / len(component_scores)
                    dashboard_data['overall_health']['score'] = round(overall_score, 1)
                    
                    if overall_score >= 90:
                        dashboard_data['overall_health']['status'] = 'excellent'
                    elif overall_score >= 75:
                        dashboard_data['overall_health']['status'] = 'good'
                    elif overall_score >= 50:
                        dashboard_data['overall_health']['status'] = 'degraded'
                    else:
                        dashboard_data['overall_health']['status'] = 'critical'
                
                return jsonify({
                    'status': 'success',
                    'dashboard_data': dashboard_data
                }), 200
                
            except Exception as e:
                logger.error(f"Health dashboard failed: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.blueprint.route('/alerts', methods=['GET'])
        def system_alerts():
            """Get current system alerts and warnings."""
            try:
                alerts = []
                current_time = datetime.now()
                
                # Check analytics aggregator for issues
                if self.aggregator:
                    try:
                        robustness = self.aggregator.get_robustness_monitoring()
                        score = robustness['overall_health_score']
                        
                        if score < 50:
                            alerts.append({
                                'severity': 'critical',
                                'component': 'analytics_aggregator',
                                'message': f'Analytics health score critically low: {score}%',
                                'timestamp': current_time.isoformat(),
                                'action': 'Check robustness features and component health'
                            })
                        elif score < 75:
                            alerts.append({
                                'severity': 'warning',
                                'component': 'analytics_aggregator',
                                'message': f'Analytics health score below optimal: {score}%',
                                'timestamp': current_time.isoformat(),
                                'action': 'Review component performance'
                            })
                    except Exception as e:
                        alerts.append({
                            'severity': 'error',
                            'component': 'analytics_aggregator',
                            'message': f'Failed to check analytics health: {str(e)}',
                            'timestamp': current_time.isoformat(),
                            'action': 'Check analytics aggregator status'
                        })
                
                # Check delivery guarantee for issues
                if self.delivery_guarantee:
                    try:
                        stats = self.delivery_guarantee.get_guarantee_statistics()
                        success_rate = stats['statistics']['delivery_success_rate']
                        pending = stats['statistics']['current_pending']
                        
                        if success_rate < 90:
                            alerts.append({
                                'severity': 'critical',
                                'component': 'delivery_guarantee',
                                'message': f'Delivery success rate critically low: {success_rate:.1f}%',
                                'timestamp': current_time.isoformat(),
                                'action': 'Check delivery handlers and retry configuration'
                            })
                        
                        if pending > 100:
                            alerts.append({
                                'severity': 'warning',
                                'component': 'delivery_guarantee',
                                'message': f'High number of pending deliveries: {pending}',
                                'timestamp': current_time.isoformat(),
                                'action': 'Check delivery processing performance'
                            })
                    except Exception as e:
                        alerts.append({
                            'severity': 'error',
                            'component': 'delivery_guarantee',
                            'message': f'Failed to check delivery status: {str(e)}',
                            'timestamp': current_time.isoformat(),
                            'action': 'Check delivery guarantee system'
                        })
                
                return jsonify({
                    'status': 'success',
                    'alerts': alerts,
                    'alert_count': len(alerts),
                    'critical_count': len([a for a in alerts if a['severity'] == 'critical']),
                    'warning_count': len([a for a in alerts if a['severity'] == 'warning']),
                    'timestamp': current_time.isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"System alerts check failed: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500

# Global comprehensive monitoring instance
comprehensive_monitoring = None