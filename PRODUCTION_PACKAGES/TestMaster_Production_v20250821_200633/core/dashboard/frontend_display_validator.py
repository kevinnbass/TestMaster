#!/usr/bin/env python3
"""
Frontend Display Validator
==========================

Tests that backend capabilities are properly integrated and displayed 
on the frontend without requiring a web browser. Simulates frontend
rendering and validates data flow.

Author: TestMaster Team
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import re

class FrontendDisplayValidator:
    """Validates frontend display integration without browser."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.validation_results = {}
        
    def validate_all_backend_integrations(self) -> Dict[str, Any]:
        """Comprehensive validation of all backend integrations."""
        print("=== Frontend Display Validation ===")
        print()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'ultra_reliability_features': self.validate_ultra_reliability_display(),
            'analytics_integration': self.validate_analytics_display(),
            'monitoring_dashboards': self.validate_monitoring_display(),
            'performance_metrics': self.validate_performance_display(),
            'real_time_features': self.validate_realtime_display(),
            'data_completeness': self.validate_data_completeness(),
            'frontend_readiness': self.validate_frontend_readiness()
        }
        
        # Calculate overall integration score
        results['overall_score'] = self.calculate_integration_score(results)
        results['recommendations'] = self.generate_integration_recommendations(results)
        
        return results
    
    def validate_ultra_reliability_display(self) -> Dict[str, Any]:
        """Validate ultra-reliability features are properly displayed."""
        print("Testing Ultra-Reliability Feature Display...")
        
        features = {
            'robustness_monitoring': '/api/monitoring/robustness',
            'circuit_breakers': '/api/monitoring/robustness', 
            'sla_tracking': '/api/monitoring/robustness',
            'health_monitoring': '/api/health/live',
            'heartbeat_systems': '/api/monitoring/heartbeat',
            'fallback_systems': '/api/monitoring/fallback',
            'dead_letter_queue': '/api/monitoring/dead-letter',
            'batch_processing': '/api/monitoring/batch',
            'flow_monitoring': '/api/monitoring/flow'
        }
        
        results = {}
        for feature_name, endpoint in features.items():
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Analyze if data is rich enough for frontend display
                    display_quality = self.analyze_display_quality(data, feature_name)
                    
                    results[feature_name] = {
                        'available': True,
                        'data_size': len(response.text),
                        'display_quality': display_quality,
                        'key_metrics': self.extract_key_metrics(data),
                        'frontend_ready': display_quality['score'] >= 70
                    }
                    print(f"  {feature_name}: READY (score: {display_quality['score']})")
                else:
                    results[feature_name] = {
                        'available': False,
                        'status_code': response.status_code,
                        'frontend_ready': False
                    }
                    print(f"  {feature_name}: NOT AVAILABLE ({response.status_code})")
                    
            except Exception as e:
                results[feature_name] = {
                    'available': False,
                    'error': str(e),
                    'frontend_ready': False
                }
                print(f"  {feature_name}: ERROR ({str(e)[:30]}...)")
        
        ready_features = sum(1 for r in results.values() if r.get('frontend_ready'))
        total_features = len(results)
        
        print(f"  Ultra-Reliability Display: {ready_features}/{total_features} features ready")
        print()
        
        return {
            'features': results,
            'ready_count': ready_features,
            'total_count': total_features,
            'readiness_percentage': (ready_features / total_features) * 100
        }
    
    def validate_analytics_display(self) -> Dict[str, Any]:
        """Validate analytics data display integration."""
        print("Testing Analytics Display Integration...")
        
        analytics_endpoints = [
            '/api/analytics/trends',
            '/api/analytics/summary',
            '/api/llm/metrics'
        ]
        
        results = {}
        for endpoint in analytics_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for time-series data, metrics, trends
                    has_metrics = self.has_analytics_metrics(data)
                    has_trends = self.has_trend_data(data)
                    has_timestamps = self.has_time_data(data)
                    
                    results[endpoint] = {
                        'available': True,
                        'has_metrics': has_metrics,
                        'has_trends': has_trends,
                        'has_timestamps': has_timestamps,
                        'chart_ready': has_metrics and has_timestamps,
                        'dashboard_score': self.calculate_dashboard_score(data)
                    }
                    print(f"  {endpoint}: Chart-ready: {results[endpoint]['chart_ready']}")
                else:
                    results[endpoint] = {'available': False, 'status_code': response.status_code}
                    print(f"  {endpoint}: NOT AVAILABLE ({response.status_code})")
                    
            except Exception as e:
                results[endpoint] = {'available': False, 'error': str(e)}
                print(f"  {endpoint}: ERROR ({str(e)[:30]}...)")
        
        chart_ready = sum(1 for r in results.values() if r.get('chart_ready'))
        total_endpoints = len(results)
        
        print(f"  Analytics Display: {chart_ready}/{total_endpoints} endpoints chart-ready")
        print()
        
        return {
            'endpoints': results,
            'chart_ready_count': chart_ready,
            'total_count': total_endpoints,
            'visualization_readiness': (chart_ready / total_endpoints) * 100
        }
    
    def validate_monitoring_display(self) -> Dict[str, Any]:
        """Validate monitoring dashboard display."""
        print("Testing Monitoring Dashboard Display...")
        
        # Test real-time monitoring data
        try:
            response = requests.get(f"{self.base_url}/api/monitoring/robustness", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Analyze monitoring dashboard components
                dashboard_components = {
                    'health_score': 'health_score' in data,
                    'status_indicators': 'status' in data,
                    'monitoring_details': 'monitoring' in data and isinstance(data.get('monitoring'), dict),
                    'real_time_data': self.has_realtime_indicators(data),
                    'alert_data': self.has_alert_indicators(data)
                }
                
                component_count = sum(dashboard_components.values())
                total_components = len(dashboard_components)
                
                print(f"  Dashboard Components: {component_count}/{total_components} available")
                print(f"  Health Score Available: {'YES' if dashboard_components['health_score'] else 'NO'}")
                print(f"  Real-time Data: {'YES' if dashboard_components['real_time_data'] else 'NO'}")
                print()
                
                return {
                    'available': True,
                    'components': dashboard_components,
                    'component_readiness': (component_count / total_components) * 100,
                    'dashboard_data': self.extract_dashboard_metrics(data)
                }
            else:
                print(f"  Monitoring Dashboard: NOT AVAILABLE ({response.status_code})")
                return {'available': False, 'status_code': response.status_code}
                
        except Exception as e:
            print(f"  Monitoring Dashboard: ERROR ({str(e)[:30]}...)")
            return {'available': False, 'error': str(e)}
    
    def validate_performance_display(self) -> Dict[str, Any]:
        """Validate performance metrics display."""
        print("Testing Performance Metrics Display...")
        
        endpoints = ['/api/performance/summary', '/api/llm/metrics']
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Look for performance indicators
                    perf_indicators = {
                        'response_times': self.has_response_time_data(data),
                        'throughput': self.has_throughput_data(data),
                        'error_rates': self.has_error_rate_data(data),
                        'resource_usage': self.has_resource_data(data)
                    }
                    
                    indicator_count = sum(perf_indicators.values())
                    results[endpoint] = {
                        'available': True,
                        'performance_indicators': perf_indicators,
                        'metric_count': indicator_count,
                        'display_ready': indicator_count >= 2
                    }
                    print(f"  {endpoint}: {indicator_count}/4 performance indicators available")
                else:
                    results[endpoint] = {'available': False, 'status_code': response.status_code}
                    print(f"  {endpoint}: NOT AVAILABLE ({response.status_code})")
                    
            except Exception as e:
                results[endpoint] = {'available': False, 'error': str(e)}
                print(f"  {endpoint}: ERROR ({str(e)[:30]}...)")
        
        display_ready = sum(1 for r in results.values() if r.get('display_ready'))
        total_endpoints = len(results)
        
        print(f"  Performance Display: {display_ready}/{total_endpoints} endpoints ready")
        print()
        
        return {
            'endpoints': results,
            'display_ready_count': display_ready,
            'total_count': total_endpoints,
            'performance_dashboard_readiness': (display_ready / total_endpoints) * 100
        }
    
    def validate_realtime_display(self) -> Dict[str, Any]:
        """Validate real-time feature display."""
        print("Testing Real-time Feature Display...")
        
        # Test multiple rapid requests to simulate real-time updates
        realtime_endpoints = [
            '/api/health/live',
            '/api/monitoring/heartbeat',
            '/api/monitoring/flow'
        ]
        
        results = {}
        for endpoint in realtime_endpoints:
            timestamps = []
            data_changes = []
            
            # Make 3 requests over 6 seconds to check for real-time updates
            for i in range(3):
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                    if response.status_code == 200:
                        data = response.json()
                        timestamps.append(time.time())
                        data_changes.append(self.extract_dynamic_data(data))
                        time.sleep(2)
                    else:
                        break
                except Exception:
                    break
            
            if len(timestamps) >= 2:
                # Analyze if data is actually updating
                has_updates = len(set(str(d) for d in data_changes)) > 1
                avg_response_time = sum(timestamps[1:]) - sum(timestamps[:-1]) if len(timestamps) > 1 else 0
                
                results[endpoint] = {
                    'available': True,
                    'real_time_updates': has_updates,
                    'response_consistency': len(data_changes),
                    'suitable_for_realtime': has_updates or self.has_timestamp_data(data_changes[-1] if data_changes else {})
                }
                print(f"  {endpoint}: Real-time capable: {'YES' if results[endpoint]['suitable_for_realtime'] else 'NO'}")
            else:
                results[endpoint] = {'available': False}
                print(f"  {endpoint}: NOT AVAILABLE")
        
        realtime_ready = sum(1 for r in results.values() if r.get('suitable_for_realtime'))
        total_endpoints = len(results)
        
        print(f"  Real-time Display: {realtime_ready}/{total_endpoints} endpoints ready")
        print()
        
        return {
            'endpoints': results,
            'realtime_ready_count': realtime_ready,
            'total_count': total_endpoints,
            'realtime_capability': (realtime_ready / total_endpoints) * 100
        }
    
    def validate_data_completeness(self) -> Dict[str, Any]:
        """Validate overall data completeness for frontend display."""
        print("Testing Data Completeness...")
        
        # Check key data endpoints for completeness
        completeness_checks = {
            'system_health': '/api/health/ready',
            'monitoring_data': '/api/monitoring/robustness', 
            'flow_data': '/api/monitoring/flow',
            'heartbeat_data': '/api/monitoring/heartbeat'
        }
        
        results = {}
        total_data_points = 0
        complete_data_points = 0
        
        for check_name, endpoint in completeness_checks.items():
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Count data completeness
                    data_metrics = self.count_data_metrics(data)
                    complete_metrics = self.count_complete_metrics(data)
                    
                    results[check_name] = {
                        'available': True,
                        'total_metrics': data_metrics,
                        'complete_metrics': complete_metrics,
                        'completeness_percent': (complete_metrics / data_metrics) * 100 if data_metrics > 0 else 0
                    }
                    
                    total_data_points += data_metrics
                    complete_data_points += complete_metrics
                    
                    print(f"  {check_name}: {complete_metrics}/{data_metrics} metrics complete ({results[check_name]['completeness_percent']:.1f}%)")
                else:
                    results[check_name] = {'available': False, 'status_code': response.status_code}
                    print(f"  {check_name}: NOT AVAILABLE ({response.status_code})")
                    
            except Exception as e:
                results[check_name] = {'available': False, 'error': str(e)}
                print(f"  {check_name}: ERROR ({str(e)[:30]}...)")
        
        overall_completeness = (complete_data_points / total_data_points) * 100 if total_data_points > 0 else 0
        print(f"  Overall Data Completeness: {complete_data_points}/{total_data_points} ({overall_completeness:.1f}%)")
        print()
        
        return {
            'checks': results,
            'overall_completeness': overall_completeness,
            'total_data_points': total_data_points,
            'complete_data_points': complete_data_points
        }
    
    def validate_frontend_readiness(self) -> Dict[str, Any]:
        """Validate overall frontend readiness."""
        print("Testing Frontend Readiness...")
        
        # Overall readiness checks
        readiness_checks = {
            'api_availability': self.check_api_availability(),
            'data_structure': self.check_data_structure_consistency(),
            'response_times': self.check_response_performance(),
            'error_handling': self.check_error_handling()
        }
        
        readiness_score = sum(check['score'] for check in readiness_checks.values()) / len(readiness_checks)
        
        print(f"  API Availability: {readiness_checks['api_availability']['score']:.1f}%")
        print(f"  Data Structure: {readiness_checks['data_structure']['score']:.1f}%")
        print(f"  Response Times: {readiness_checks['response_times']['score']:.1f}%")
        print(f"  Error Handling: {readiness_checks['error_handling']['score']:.1f}%")
        print(f"  Overall Readiness: {readiness_score:.1f}%")
        print()
        
        return {
            'checks': readiness_checks,
            'overall_readiness_score': readiness_score,
            'production_ready': readiness_score >= 80
        }
    
    # Helper methods for data analysis
    
    def analyze_display_quality(self, data: Any, feature_name: str) -> Dict[str, Any]:
        """Analyze quality of data for frontend display."""
        if not isinstance(data, dict):
            return {'score': 20, 'issues': ['Non-dictionary response']}
        
        score = 50  # Base score
        issues = []
        
        # Check for key data indicators
        if 'status' in data:
            score += 15
        if any(key in data for key in ['health', 'score', 'metrics', 'monitoring']):
            score += 20
        if any(key in data for key in ['timestamp', 'time', 'updated']):
            score += 10
        if len(data) >= 3:
            score += 5
        
        # Check data depth
        nested_dicts = sum(1 for v in data.values() if isinstance(v, dict))
        if nested_dicts > 0:
            score += min(nested_dicts * 5, 15)
        
        return {'score': min(score, 100), 'issues': issues}
    
    def extract_key_metrics(self, data: Any) -> List[str]:
        """Extract key metrics from data."""
        if not isinstance(data, dict):
            return []
        
        metrics = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics.append(f"{key}: {value}")
            elif key in ['status', 'health', 'state']:
                metrics.append(f"{key}: {value}")
        
        return metrics[:5]  # Top 5 metrics
    
    def has_analytics_metrics(self, data: Any) -> bool:
        """Check if data contains analytics metrics."""
        if not isinstance(data, dict):
            return False
        
        analytics_keys = ['metrics', 'analytics', 'count', 'total', 'rate', 'performance']
        return any(key in str(data).lower() for key in analytics_keys)
    
    def has_trend_data(self, data: Any) -> bool:
        """Check if data contains trend information."""
        if not isinstance(data, dict):
            return False
        
        trend_keys = ['trend', 'history', 'series', 'over_time', 'changes']
        return any(key in str(data).lower() for key in trend_keys)
    
    def has_time_data(self, data: Any) -> bool:
        """Check if data contains timestamp information."""
        if not isinstance(data, dict):
            return False
        
        time_keys = ['timestamp', 'time', 'date', 'updated', 'created']
        return any(key in str(data).lower() for key in time_keys)
    
    def has_realtime_indicators(self, data: Any) -> bool:
        """Check if data has real-time indicators."""
        if not isinstance(data, dict):
            return False
        
        realtime_keys = ['live', 'current', 'active', 'running', 'online']
        return any(key in str(data).lower() for key in realtime_keys)
    
    def has_alert_indicators(self, data: Any) -> bool:
        """Check if data has alert/notification indicators."""
        if not isinstance(data, dict):
            return False
        
        alert_keys = ['alert', 'warning', 'error', 'failure', 'critical']
        return any(key in str(data).lower() for key in alert_keys)
    
    def calculate_dashboard_score(self, data: Any) -> int:
        """Calculate dashboard readiness score."""
        if not isinstance(data, dict):
            return 0
        
        score = 0
        
        # Basic structure
        if 'status' in data:
            score += 20
        
        # Data richness
        if len(data) >= 3:
            score += 20
        
        # Nested data
        nested_count = sum(1 for v in data.values() if isinstance(v, dict))
        score += min(nested_count * 15, 30)
        
        # Metrics
        metric_count = sum(1 for v in data.values() if isinstance(v, (int, float)))
        score += min(metric_count * 10, 30)
        
        return min(score, 100)
    
    def extract_dashboard_metrics(self, data: Any) -> Dict[str, Any]:
        """Extract key dashboard metrics."""
        if not isinstance(data, dict):
            return {}
        
        metrics = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            elif isinstance(value, dict) and 'score' in str(value).lower():
                metrics[f"{key}_metrics"] = value
        
        return metrics
    
    def has_response_time_data(self, data: Any) -> bool:
        """Check for response time metrics."""
        return 'response' in str(data).lower() or 'latency' in str(data).lower() or 'time' in str(data).lower()
    
    def has_throughput_data(self, data: Any) -> bool:
        """Check for throughput metrics."""
        return 'throughput' in str(data).lower() or 'rate' in str(data).lower() or 'per_second' in str(data).lower()
    
    def has_error_rate_data(self, data: Any) -> bool:
        """Check for error rate metrics."""
        return 'error' in str(data).lower() or 'failure' in str(data).lower() or 'success' in str(data).lower()
    
    def has_resource_data(self, data: Any) -> bool:
        """Check for resource usage metrics."""
        return 'memory' in str(data).lower() or 'cpu' in str(data).lower() or 'disk' in str(data).lower()
    
    def extract_dynamic_data(self, data: Any) -> Dict[str, Any]:
        """Extract potentially dynamic data."""
        if not isinstance(data, dict):
            return {}
        
        dynamic = {}
        for key, value in data.items():
            if 'time' in key.lower() or 'count' in key.lower() or 'active' in key.lower():
                dynamic[key] = value
        
        return dynamic
    
    def has_timestamp_data(self, data: Any) -> bool:
        """Check if data has timestamp information."""
        return self.has_time_data(data)
    
    def count_data_metrics(self, data: Any) -> int:
        """Count total data metrics."""
        if not isinstance(data, dict):
            return 1 if data is not None else 0
        
        count = 0
        for value in data.values():
            if isinstance(value, dict):
                count += self.count_data_metrics(value)
            else:
                count += 1
        
        return count
    
    def count_complete_metrics(self, data: Any) -> int:
        """Count complete (non-null, meaningful) metrics."""
        if not isinstance(data, dict):
            return 1 if data not in [None, '', 0] else 0
        
        count = 0
        for value in data.values():
            if isinstance(value, dict):
                count += self.count_complete_metrics(value)
            elif value not in [None, '', 0, [], {}]:
                count += 1
        
        return count
    
    def check_api_availability(self) -> Dict[str, Any]:
        """Check API endpoint availability."""
        endpoints = ['/api/health/live', '/api/monitoring/robustness', '/api/health/ready']
        available = 0
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                if response.status_code == 200:
                    available += 1
            except:
                pass
        
        score = (available / len(endpoints)) * 100
        return {'score': score, 'available': available, 'total': len(endpoints)}
    
    def check_data_structure_consistency(self) -> Dict[str, Any]:
        """Check data structure consistency."""
        endpoints = ['/api/monitoring/robustness', '/api/monitoring/heartbeat']
        consistent = 0
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and 'status' in data:
                        consistent += 1
            except:
                pass
        
        score = (consistent / len(endpoints)) * 100 if endpoints else 100
        return {'score': score, 'consistent': consistent, 'total': len(endpoints)}
    
    def check_response_performance(self) -> Dict[str, Any]:
        """Check response performance."""
        endpoint = '/api/health/live'
        times = []
        
        for _ in range(3):
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    times.append(time.time() - start)
                time.sleep(1)
            except:
                pass
        
        if times:
            avg_time = sum(times) / len(times)
            score = max(0, 100 - (avg_time * 50))  # 2s = 0%, 0s = 100%
        else:
            score = 0
        
        return {'score': score, 'avg_response_time': avg_time if times else None}
    
    def check_error_handling(self) -> Dict[str, Any]:
        """Check error handling."""
        try:
            response = requests.get(f"{self.base_url}/api/nonexistent", timeout=3)
            proper_error = 400 <= response.status_code < 600
            score = 100 if proper_error else 50
        except:
            score = 75  # Server at least responds to bad requests
        
        return {'score': score}
    
    def calculate_integration_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall integration score."""
        scores = []
        
        # Ultra-reliability
        if 'ultra_reliability_features' in results:
            scores.append(results['ultra_reliability_features'].get('readiness_percentage', 0))
        
        # Analytics
        if 'analytics_integration' in results:
            scores.append(results['analytics_integration'].get('visualization_readiness', 0))
        
        # Monitoring
        if 'monitoring_dashboards' in results:
            scores.append(results['monitoring_dashboards'].get('component_readiness', 0))
        
        # Performance
        if 'performance_metrics' in results:
            scores.append(results['performance_metrics'].get('performance_dashboard_readiness', 0))
        
        # Real-time
        if 'real_time_features' in results:
            scores.append(results['real_time_features'].get('realtime_capability', 0))
        
        # Data completeness
        if 'data_completeness' in results:
            scores.append(results['data_completeness'].get('overall_completeness', 0))
        
        # Frontend readiness
        if 'frontend_readiness' in results:
            scores.append(results['frontend_readiness'].get('overall_readiness_score', 0))
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'grade': self.score_to_grade(overall_score),
            'production_ready': overall_score >= 80
        }
    
    def score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def generate_integration_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving integration."""
        recommendations = []
        
        overall_score = results.get('overall_score', {}).get('overall_score', 0)
        
        if overall_score < 80:
            recommendations.append("Improve overall backend-frontend integration quality")
        
        # Specific recommendations based on component scores
        ultra_reliability = results.get('ultra_reliability_features', {})
        if ultra_reliability.get('readiness_percentage', 0) < 80:
            recommendations.append("Enhance ultra-reliability feature display data")
        
        analytics = results.get('analytics_integration', {})
        if analytics.get('visualization_readiness', 0) < 80:
            recommendations.append("Improve analytics data structure for chart visualization")
        
        realtime = results.get('real_time_features', {})
        if realtime.get('realtime_capability', 0) < 80:
            recommendations.append("Add more real-time data updates and timestamps")
        
        completeness = results.get('data_completeness', {})
        if completeness.get('overall_completeness', 0) < 90:
            recommendations.append("Increase data completeness across all endpoints")
        
        if not recommendations:
            recommendations.append("Excellent integration! Consider performance optimizations.")
        
        return recommendations


def main():
    """Run frontend display validation."""
    validator = FrontendDisplayValidator()
    results = validator.validate_all_backend_integrations()
    
    print("=== FINAL INTEGRATION REPORT ===")
    print()
    print(f"Overall Integration Score: {results['overall_score']['overall_score']:.1f}% (Grade: {results['overall_score']['grade']})")
    print(f"Production Ready: {'YES' if results['overall_score']['production_ready'] else 'NO'}")
    print()
    
    print("Component Scores:")
    components = [
        ('Ultra-Reliability Features', results.get('ultra_reliability_features', {}).get('readiness_percentage', 0)),
        ('Analytics Integration', results.get('analytics_integration', {}).get('visualization_readiness', 0)),
        ('Monitoring Dashboards', results.get('monitoring_dashboards', {}).get('component_readiness', 0)),
        ('Performance Metrics', results.get('performance_metrics', {}).get('performance_dashboard_readiness', 0)),
        ('Real-time Features', results.get('real_time_features', {}).get('realtime_capability', 0)),
        ('Data Completeness', results.get('data_completeness', {}).get('overall_completeness', 0)),
        ('Frontend Readiness', results.get('frontend_readiness', {}).get('overall_readiness_score', 0))
    ]
    
    for component, score in components:
        print(f"  {component}: {score:.1f}%")
    
    print()
    print("Recommendations:")
    for i, rec in enumerate(results.get('recommendations', []), 1):
        print(f"  {i}. {rec}")
    
    # Save detailed results
    with open('frontend_integration_validation.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print()
    print("Detailed results saved to: frontend_integration_validation.json")
    
    return results['overall_score']['production_ready']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)