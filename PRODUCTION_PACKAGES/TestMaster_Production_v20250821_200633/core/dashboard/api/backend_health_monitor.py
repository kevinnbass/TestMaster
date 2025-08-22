
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

"""
Backend Health Monitor & Robustness Enhancement
===============================================

Comprehensive backend monitoring, validation, and auto-recovery system.
Ensures all features work properly and are accessible to the frontend.

Author: TestMaster Team
"""

import requests
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from flask import Blueprint, jsonify, request
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class EndpointHealth:
    """Health status of an API endpoint."""
    endpoint: str
    name: str
    method: str
    status: str  # 'healthy', 'degraded', 'down'
    last_check: datetime
    response_time: float
    error_count: int
    success_count: int
    last_error: Optional[str] = None
    frontend_compatible: bool = True
    has_data: bool = False
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0
        
    @property
    def availability(self) -> str:
        """Get availability status."""
        if self.success_rate >= 95:
            return "excellent"
        elif self.success_rate >= 80:
            return "good"
        elif self.success_rate >= 60:
            return "degraded"
        else:
            return "poor"

class BackendHealthMonitor:
    """
    Comprehensive backend health monitoring and validation system.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.endpoints: Dict[str, EndpointHealth] = {}
        self.monitoring_active = False
        self.check_interval = 30  # seconds
        self.logger = logging.getLogger('BackendHealthMonitor')
        
        # Circuit breaker pattern
        self.circuit_breakers = {}
        self.retry_counts = defaultdict(int)
        self.max_retries = 3
        self.circuit_break_threshold = 5  # failures before circuit opens
        
        # Metrics
        self.global_metrics = {
            'total_endpoints': 0,
            'healthy_endpoints': 0,
            'degraded_endpoints': 0,
            'down_endpoints': 0,
            'average_response_time': 0,
            'last_full_check': None,
            'uptime_percentage': 100.0
        }
        
        # Issues tracking
        self.known_issues = []
        self.auto_fixes_applied = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize endpoint registry
        self._register_core_endpoints()
        
        self.logger.info("Backend Health Monitor initialized")
        
    def _register_core_endpoints(self):
        """Register all core endpoints for monitoring."""
        core_endpoints = [
            # Core System
            ('/api/health', 'GET', 'Core Health Check'),
            ('/api/config', 'GET', 'Configuration'),
            ('/api/debug/routes', 'GET', 'Route Debug'),
            
            # Crew Orchestration
            ('/api/crew/agents', 'GET', 'Crew Agents'),
            ('/api/crew/crews', 'GET', 'Crew List'),
            ('/api/crew/swarm-types', 'GET', 'Crew Swarm Types'),
            
            # Swarm Orchestration
            ('/api/swarm/agents', 'GET', 'Swarm Agents'),
            ('/api/swarm/swarms', 'GET', 'Swarm List'),
            ('/api/swarm/architectures', 'GET', 'Swarm Architectures'),
            
            # Observability
            ('/api/observability/health', 'GET', 'Observability Health'),
            ('/api/observability/metrics', 'GET', 'Observability Metrics'),
            
            # Production API
            ('/api/production/health', 'GET', 'Production Health'),
            ('/api/production/metrics', 'GET', 'Production Metrics'),
            ('/api/production/streams/active', 'GET', 'Active Streams'),
            
            # Enhanced Telemetry
            ('/api/telemetry/health', 'GET', 'Telemetry Health'),
            ('/api/telemetry/metrics', 'GET', 'Telemetry Metrics'),
            ('/api/telemetry/spans', 'GET', 'Telemetry Spans'),
            ('/api/telemetry/custom-metrics', 'GET', 'Custom Metrics'),
            ('/api/telemetry/events', 'GET', 'Telemetry Events'),
            
            # Analytics & Performance
            ('/api/analytics/health', 'GET', 'Analytics Health'),
            ('/api/analytics/metrics', 'GET', 'Analytics Metrics'),
            ('/api/performance/health', 'GET', 'Performance Health'),
            ('/api/performance/metrics', 'GET', 'Performance Metrics'),
        ]
        
        for endpoint, method, name in core_endpoints:
            self.endpoints[endpoint] = EndpointHealth(
                endpoint=endpoint,
                name=name,
                method=method,
                status='unknown',
                last_check=datetime.now(),
                response_time=0,
                error_count=0,
                success_count=0
            )
            
        self.global_metrics['total_endpoints'] = len(self.endpoints)
        
    def _is_circuit_open(self, endpoint_path: str) -> bool:
        """Check if circuit breaker is open for endpoint."""
        return self.circuit_breakers.get(endpoint_path, False)
        
    def _open_circuit(self, endpoint_path: str):
        """Open circuit breaker for endpoint."""
        self.circuit_breakers[endpoint_path] = True
        self.logger.warning(f"Circuit breaker opened for {endpoint_path}")
        
    def _close_circuit(self, endpoint_path: str):
        """Close circuit breaker for endpoint."""
        if endpoint_path in self.circuit_breakers:
            del self.circuit_breakers[endpoint_path]
        self.retry_counts[endpoint_path] = 0
        
    def check_endpoint(self, endpoint_path: str) -> EndpointHealth:
        """Check health of a specific endpoint."""
        if endpoint_path not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_path} not registered")
            
        endpoint = self.endpoints[endpoint_path]
        
        # Check circuit breaker
        if self._is_circuit_open(endpoint_path):
            endpoint.status = 'down'
            endpoint.last_error = "Circuit breaker open - too many failures"
            endpoint.last_check = datetime.now()
            return endpoint
            
        start_time = time.time()
        
        # Optimized timeout based on endpoint type (targeting <1s average)
        timeout_map = {
            '/api/health': 1,
            '/api/config': 2,
            '/api/debug/routes': 3,
            '/api/production/health': 5,  # Reduced from 15s
            '/api/telemetry/spans': 4,    # Reduced from 12s
            '/api/analytics/metrics': 3,  # Reduced from 10s
            '/api/performance/health': 2,
            '/api/observability/health': 2,
            '/api/crew/agents': 2,
            '/api/swarm/agents': 2
        }
        timeout = timeout_map.get(endpoint_path, 3)  # Reduced default from 8s to 3s
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint_path}",
                timeout=timeout,
                headers={'Accept': 'application/json'}
            )
            
            response_time = time.time() - start_time
            endpoint.response_time = response_time
            endpoint.last_check = datetime.now()
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Check frontend compatibility
                    endpoint.frontend_compatible = self._validate_frontend_compatibility(data)
                    
                    # Check if endpoint has useful data
                    endpoint.has_data = self._check_has_data(data)
                    
                    # Determine status based on response
                    if isinstance(data, dict) and data.get('status') in ['success', 'healthy']:
                        endpoint.status = 'healthy'
                        endpoint.success_count += 1
                        endpoint.last_error = None
                        # Close circuit breaker on success
                        self._close_circuit(endpoint_path)
                    else:
                        endpoint.status = 'degraded'
                        endpoint.error_count += 1
                        endpoint.last_error = f"Unexpected response format: {data.get('status', 'unknown')}"
                        self._handle_failure(endpoint_path)
                        
                except json.JSONDecodeError:
                    endpoint.status = 'degraded'
                    endpoint.error_count += 1
                    endpoint.last_error = "Invalid JSON response"
                    endpoint.frontend_compatible = False
                    self._handle_failure(endpoint_path)
                    
            elif response.status_code == 404:
                endpoint.status = 'down'
                endpoint.error_count += 1
                endpoint.last_error = "Endpoint not found (404)"
                endpoint.frontend_compatible = False
                self._handle_failure(endpoint_path)
                
            else:
                endpoint.status = 'degraded'
                endpoint.error_count += 1
                endpoint.last_error = f"HTTP {response.status_code}"
                self._handle_failure(endpoint_path)
                
        except requests.exceptions.Timeout:
            endpoint.status = 'degraded'
            endpoint.error_count += 1
            endpoint.last_error = f"Request timeout after {timeout}s"
            endpoint.frontend_compatible = False
            endpoint.response_time = timeout
            endpoint.last_check = datetime.now()
            self._handle_failure(endpoint_path)
        except requests.exceptions.ConnectionError:
            endpoint.status = 'down'
            endpoint.error_count += 1
            endpoint.last_error = "Connection failed - service may be down"
            endpoint.frontend_compatible = False
            endpoint.response_time = time.time() - start_time
            endpoint.last_check = datetime.now()
            self._handle_failure(endpoint_path)
        except requests.exceptions.RequestException as e:
            endpoint.status = 'degraded'
            endpoint.error_count += 1
            endpoint.last_error = f"Request error: {str(e)[:100]}"
            endpoint.frontend_compatible = False
            endpoint.response_time = time.time() - start_time
            endpoint.last_check = datetime.now()
            self._handle_failure(endpoint_path)
            
        return endpoint
        
    def _handle_failure(self, endpoint_path: str):
        """Handle endpoint failure with circuit breaker logic."""
        self.retry_counts[endpoint_path] += 1
        
        if self.retry_counts[endpoint_path] >= self.circuit_break_threshold:
            self._open_circuit(endpoint_path)
            self.logger.warning(f"Circuit breaker opened for {endpoint_path} after {self.retry_counts[endpoint_path]} failures")
        
    def _validate_frontend_compatibility(self, data: Any) -> bool:
        """Validate if response is compatible with frontend expectations."""
        if not isinstance(data, dict):
            return False
            
        # Check for required fields
        if 'status' not in data:
            return False
            
        # Status should be meaningful
        valid_statuses = ['success', 'healthy', 'error', 'degraded']
        if data.get('status') not in valid_statuses:
            return False
            
        return True
        
    def _check_has_data(self, data: Dict[str, Any]) -> bool:
        """Check if endpoint response contains useful data."""
        if not isinstance(data, dict):
            return False
            
        # Look for common data fields
        data_fields = ['data', 'metrics', 'agents', 'crews', 'swarms', 'spans', 'events']
        
        for field in data_fields:
            if field in data and data[field]:
                return True
                
        return False
        
    def check_all_endpoints(self, max_workers: int = 5) -> Dict[str, Any]:
        """Check all registered endpoints concurrently and return summary."""
        results = {
            'healthy': [],
            'degraded': [],
            'down': [],
            'frontend_issues': [],
            'missing_data': []
        }
        
        # Check endpoints concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all endpoint checks
            future_to_endpoint = {
                executor.submit(self.check_endpoint, endpoint_path): endpoint_path
                for endpoint_path in self.endpoints
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_endpoint):
                endpoint_path = future_to_endpoint[future]
                try:
                    endpoint = future.result(timeout=20)  # 20 second timeout for each check
                    
                    if endpoint.status == 'healthy':
                        results['healthy'].append(endpoint)
                    elif endpoint.status == 'degraded':
                        results['degraded'].append(endpoint)
                    else:
                        results['down'].append(endpoint)
                        
                    if not endpoint.frontend_compatible:
                        results['frontend_issues'].append(endpoint)
                        
                    if not endpoint.has_data and endpoint.status == 'healthy':
                        results['missing_data'].append(endpoint)
                        
                except Exception as e:
                    self.logger.error(f"Failed to check endpoint {endpoint_path}: {e}")
                    # Create a failed endpoint entry
                    failed_endpoint = self.endpoints[endpoint_path]
                    failed_endpoint.status = 'down'
                    failed_endpoint.last_error = f"Check failed: {str(e)[:100]}"
                    failed_endpoint.last_check = datetime.now()
                    results['down'].append(failed_endpoint)
                
        # Update global metrics
        with self.lock:
            total_endpoints = len(self.endpoints)
            healthy_count = len(results['healthy'])
            
            self.global_metrics.update({
                'healthy_endpoints': healthy_count,
                'degraded_endpoints': len(results['degraded']),
                'down_endpoints': len(results['down']),
                'last_full_check': datetime.now().isoformat(),
                'uptime_percentage': (healthy_count / total_endpoints * 100) if total_endpoints > 0 else 0
            })
            
            # Calculate average response time from recent checks
            recent_times = [ep.response_time for ep in self.endpoints.values() if ep.response_time > 0]
            if recent_times:
                self.global_metrics['average_response_time'] = sum(recent_times) / len(recent_times)
            
        return results
        
    def get_endpoint_status(self, endpoint_path: str) -> Dict[str, Any]:
        """Get detailed status of a specific endpoint."""
        if endpoint_path not in self.endpoints:
            return {'error': 'Endpoint not found'}
            
        endpoint = self.endpoints[endpoint_path]
        return {
            'endpoint': endpoint.endpoint,
            'name': endpoint.name,
            'status': endpoint.status,
            'availability': endpoint.availability,
            'success_rate': endpoint.success_rate,
            'response_time': endpoint.response_time,
            'last_check': endpoint.last_check.isoformat(),
            'frontend_compatible': endpoint.frontend_compatible,
            'has_data': endpoint.has_data,
            'last_error': endpoint.last_error
        }
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall backend health summary."""
        return {
            'global_metrics': self.global_metrics,
            'monitoring_active': self.monitoring_active,
            'total_issues': len(self.known_issues),
            'auto_fixes_applied': len(self.auto_fixes_applied),
            'last_check': datetime.now().isoformat()
        }
        
    def identify_issues(self) -> List[Dict[str, Any]]:
        """Identify and categorize backend issues."""
        issues = []
        
        for endpoint in self.endpoints.values():
            if endpoint.status == 'down':
                issues.append({
                    'type': 'endpoint_down',
                    'severity': 'high',
                    'endpoint': endpoint.endpoint,
                    'name': endpoint.name,
                    'error': endpoint.last_error,
                    'recommendation': 'Check endpoint implementation and routing'
                })
                
            elif endpoint.status == 'degraded':
                issues.append({
                    'type': 'endpoint_degraded',
                    'severity': 'medium',
                    'endpoint': endpoint.endpoint,
                    'name': endpoint.name,
                    'error': endpoint.last_error,
                    'recommendation': 'Review endpoint response format'
                })
                
            elif not endpoint.frontend_compatible:
                issues.append({
                    'type': 'frontend_compatibility',
                    'severity': 'medium',
                    'endpoint': endpoint.endpoint,
                    'name': endpoint.name,
                    'recommendation': 'Ensure response includes status field and proper structure'
                })
                
            elif not endpoint.has_data and endpoint.status == 'healthy':
                issues.append({
                    'type': 'missing_data',
                    'severity': 'low',
                    'endpoint': endpoint.endpoint,
                    'name': endpoint.name,
                    'recommendation': 'Endpoint is healthy but returning no data - may need data population'
                })
                
        return issues
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    results = self.check_all_endpoints()
                    issues = self.identify_issues()
                    
                    if issues:
                        self.known_issues = issues
                        self.logger.warning(f"Found {len(issues)} backend issues")
                        
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    time.sleep(5)  # Short delay before retry
                    
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        self.logger.info("Backend health monitoring started")
        
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        self.logger.info("Backend health monitoring stopped")


# Global health monitor instance
health_monitor = BackendHealthMonitor()

# Flask Blueprint for health monitoring API
health_monitor_bp = Blueprint('health_monitor', __name__)

@health_monitor_bp.route('/status', methods=['GET'])
def get_backend_status():
    """Get comprehensive backend status."""
    try:
        results = health_monitor.check_all_endpoints()
        issues = health_monitor.identify_issues()
        summary = health_monitor.get_health_summary()
        
        return jsonify({
            'status': 'success',
            'summary': summary,
            'endpoints': {
                'healthy': len(results['healthy']),
                'degraded': len(results['degraded']),
                'down': len(results['down']),
                'frontend_issues': len(results['frontend_issues']),
                'missing_data': len(results['missing_data'])
            },
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_monitor_bp.route('/endpoints', methods=['GET'])
def list_all_endpoints():
    """List all monitored endpoints with their status."""
    try:
        endpoints_status = {}
        
        for endpoint_path in health_monitor.endpoints:
            endpoints_status[endpoint_path] = health_monitor.get_endpoint_status(endpoint_path)
            
        return jsonify({
            'status': 'success',
            'endpoints': endpoints_status,
            'total': len(endpoints_status)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_monitor_bp.route('/endpoints/<path:endpoint_path>', methods=['GET'])
def get_endpoint_detail(endpoint_path):
    """Get detailed status of a specific endpoint."""
    try:
        # Add leading slash if not present
        if not endpoint_path.startswith('/'):
            endpoint_path = '/' + endpoint_path
            
        endpoint_status = health_monitor.get_endpoint_status(endpoint_path)
        
        if 'error' in endpoint_status:
            return jsonify({
                'status': 'error',
                'error': endpoint_status['error']
            }), 404
            
        return jsonify({
            'status': 'success',
            'endpoint': endpoint_status
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_monitor_bp.route('/check', methods=['POST'])
def run_health_check():
    """Run immediate health check on all endpoints."""
    try:
        results = health_monitor.check_all_endpoints()
        issues = health_monitor.identify_issues()
        
        return jsonify({
            'status': 'success',
            'results': {
                'healthy': [{'endpoint': ep.endpoint, 'name': ep.name} for ep in results['healthy']],
                'degraded': [{'endpoint': ep.endpoint, 'name': ep.name, 'error': ep.last_error} for ep in results['degraded']],
                'down': [{'endpoint': ep.endpoint, 'name': ep.name, 'error': ep.last_error} for ep in results['down']]
            },
            'issues': issues,
            'summary': health_monitor.get_health_summary()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_monitor_bp.route('/monitoring/start', methods=['POST'])
def start_monitoring():
    """Start continuous monitoring."""
    try:
        health_monitor.start_monitoring()
        return jsonify({
            'status': 'success',
            'message': 'Health monitoring started'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_monitor_bp.route('/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop continuous monitoring."""
    try:
        health_monitor.stop_monitoring()
        return jsonify({
            'status': 'success',
            'message': 'Health monitoring stopped'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_monitor_bp.route('/issues', methods=['GET'])
def get_current_issues():
    """Get current backend issues."""
    try:
        issues = health_monitor.identify_issues()
        
        return jsonify({
            'status': 'success',
            'issues': issues,
            'total_issues': len(issues),
            'by_severity': {
                'high': len([i for i in issues if i['severity'] == 'high']),
                'medium': len([i for i in issues if i['severity'] == 'medium']),
                'low': len([i for i in issues if i['severity'] == 'low'])
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_monitor_bp.route('/health', methods=['GET'])
def health_monitor_health():
    """Health check for the health monitor itself."""
    return jsonify({
        'status': 'healthy',
        'service': 'Backend Health Monitor',
        'monitoring_active': health_monitor.monitoring_active,
        'endpoints_registered': len(health_monitor.endpoints),
        'timestamp': datetime.now().isoformat()
    })

# Export key components
__all__ = [
    'BackendHealthMonitor',
    'EndpointHealth',
    'health_monitor',
    'health_monitor_bp'
]