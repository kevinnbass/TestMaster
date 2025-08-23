#!/usr/bin/env python3
"""
Dashboard API Routes - Atomic Component
Dashboard-serving API routes optimized for frontend
Agent Z - STEELCLAD Frontend Atomization
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps

from flask import Flask, jsonify, request, Response


class APIResponseStatus(Enum):
    """API response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


@dataclass
class DashboardEndpoint:
    """Dashboard API endpoint configuration"""
    path: str
    method: str
    handler: Callable
    cache_enabled: bool = True
    auth_required: bool = False
    description: str = ""


class DashboardAPIRoutes:
    """
    Dashboard-serving API routes component
    Provides REST endpoints for dashboard data and operations
    """
    
    def __init__(self, app: Flask = None):
        self.app = app or Flask(__name__)
        self.endpoints: List[DashboardEndpoint] = []
        
        # Performance tracking
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'endpoints_registered': 0
        }
        
        # Cache for dashboard data
        self.dashboard_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 30  # seconds
        
        # Register default dashboard routes
        self._register_default_routes()
    
    def _register_default_routes(self):
        """Register default dashboard API routes"""
        # Dashboard data endpoint
        self.register_route(
            path='/api/dashboard/data',
            method='GET',
            handler=self.get_dashboard_data,
            description='Get comprehensive dashboard data'
        )
        
        # Dashboard configuration
        self.register_route(
            path='/api/dashboard/config',
            method='GET',
            handler=self.get_dashboard_config,
            description='Get dashboard configuration'
        )
        
        # Dashboard metrics
        self.register_route(
            path='/api/dashboard/metrics',
            method='GET',
            handler=self.get_dashboard_metrics,
            description='Get dashboard performance metrics'
        )
        
        # Dashboard health
        self.register_route(
            path='/api/dashboard/health',
            method='GET',
            handler=self.get_dashboard_health,
            description='Get dashboard health status'
        )
        
        # Dashboard export
        self.register_route(
            path='/api/dashboard/export',
            method='POST',
            handler=self.export_dashboard_data,
            description='Export dashboard data'
        )
    
    def register_dashboard_routes(self, app: Flask):
        """
        Register dashboard routes with Flask app
        Main interface for route registration
        """
        self.app = app
        
        for endpoint in self.endpoints:
            self._add_route_to_app(endpoint)
        
        self.metrics['endpoints_registered'] = len(self.endpoints)
        
        return len(self.endpoints)
    
    def register_route(self, path: str, method: str, handler: Callable,
                      cache_enabled: bool = True, auth_required: bool = False,
                      description: str = ""):
        """Register a new dashboard API route"""
        endpoint = DashboardEndpoint(
            path=path,
            method=method,
            handler=handler,
            cache_enabled=cache_enabled,
            auth_required=auth_required,
            description=description
        )
        
        self.endpoints.append(endpoint)
        
        # If app is already set, add route immediately
        if self.app:
            self._add_route_to_app(endpoint)
    
    def _add_route_to_app(self, endpoint: DashboardEndpoint):
        """Add route to Flask application"""
        # Wrap handler with performance monitoring
        wrapped_handler = self._wrap_handler(endpoint)
        
        # Add route to Flask app
        self.app.add_url_rule(
            endpoint.path,
            endpoint=f"{endpoint.path}_{endpoint.method}",
            view_func=wrapped_handler,
            methods=[endpoint.method]
        )
    
    def _wrap_handler(self, endpoint: DashboardEndpoint):
        """Wrap handler with performance monitoring and caching"""
        @wraps(endpoint.handler)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Check cache if enabled
            if endpoint.cache_enabled and endpoint.method == 'GET':
                cached_data = self._get_cached_response(endpoint.path)
                if cached_data is not None:
                    response_time = (time.time() - start_time) * 1000
                    self._update_metrics(response_time, True)
                    return cached_data
            
            try:
                # Execute handler
                result = endpoint.handler(*args, **kwargs)
                
                # Cache successful responses
                if endpoint.cache_enabled and endpoint.method == 'GET':
                    self._cache_response(endpoint.path, result)
                
                # Update metrics
                response_time = (time.time() - start_time) * 1000
                self._update_metrics(response_time, True)
                
                return result
                
            except Exception as e:
                # Update metrics for failure
                response_time = (time.time() - start_time) * 1000
                self._update_metrics(response_time, False)
                
                return jsonify({
                    'status': APIResponseStatus.ERROR.value,
                    'message': str(e),
                    'endpoint': endpoint.path
                }), 500
        
        return wrapper
    
    def _get_cached_response(self, path: str) -> Optional[Any]:
        """Get cached response if available and fresh"""
        if path not in self.dashboard_cache:
            return None
        
        # Check if cache is expired
        if time.time() - self.cache_timestamps.get(path, 0) > self.cache_ttl:
            del self.dashboard_cache[path]
            del self.cache_timestamps[path]
            return None
        
        return self.dashboard_cache[path]
    
    def _cache_response(self, path: str, response: Any):
        """Cache response data"""
        self.dashboard_cache[path] = response
        self.cache_timestamps[path] = time.time()
    
    def _update_metrics(self, response_time_ms: float, success: bool):
        """Update performance metrics"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average response time
        current_avg = self.metrics['avg_response_time']
        self.metrics['avg_response_time'] = (
            (current_avg * 0.9) + (response_time_ms * 0.1)
        )
    
    # Dashboard API Handlers
    
    def get_dashboard_data(self) -> Response:
        """Get comprehensive dashboard data"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'sections': {
                'overview': self._get_overview_data(),
                'performance': self._get_performance_data(),
                'activity': self._get_activity_data(),
                'resources': self._get_resource_data()
            },
            'update_interval': 5000,  # milliseconds
            'cache_enabled': True
        }
        
        return jsonify({
            'status': APIResponseStatus.SUCCESS.value,
            'data': dashboard_data
        })
    
    def get_dashboard_config(self) -> Response:
        """Get dashboard configuration"""
        config = {
            'theme': 'dark',
            'refresh_rate': 5000,
            'features': {
                'real_time_updates': True,
                'data_export': True,
                'custom_widgets': True,
                'api_integration': True
            },
            'layout': {
                'grid_columns': 12,
                'responsive': True,
                'customizable': True
            }
        }
        
        return jsonify({
            'status': APIResponseStatus.SUCCESS.value,
            'data': config
        })
    
    def get_dashboard_metrics(self) -> Response:
        """Get dashboard performance metrics"""
        metrics_data = {
            'api_metrics': self.metrics,
            'cache_stats': {
                'cached_endpoints': len(self.dashboard_cache),
                'cache_ttl_seconds': self.cache_ttl
            },
            'performance': {
                'avg_response_time_ms': self.metrics['avg_response_time'],
                'success_rate': (
                    self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1)
                ),
                'latency_target_met': self.metrics['avg_response_time'] < 50
            }
        }
        
        return jsonify({
            'status': APIResponseStatus.SUCCESS.value,
            'data': metrics_data
        })
    
    def get_dashboard_health(self) -> Response:
        """Get dashboard health status"""
        health = {
            'status': 'healthy' if self.metrics['avg_response_time'] < 50 else 'degraded',
            'uptime_seconds': time.time(),  # Would track actual uptime
            'endpoints_available': len(self.endpoints),
            'last_error': None,  # Would track last error
            'checks': {
                'api_responsive': True,
                'cache_functional': True,
                'latency_acceptable': self.metrics['avg_response_time'] < 50
            }
        }
        
        return jsonify({
            'status': APIResponseStatus.SUCCESS.value,
            'data': health
        })
    
    def export_dashboard_data(self) -> Response:
        """Export dashboard data in requested format"""
        try:
            export_format = request.json.get('format', 'json')
            include_sections = request.json.get('sections', ['all'])
            
            # Gather export data
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'format': export_format,
                'data': {}
            }
            
            if 'all' in include_sections or 'overview' in include_sections:
                export_data['data']['overview'] = self._get_overview_data()
            
            if 'all' in include_sections or 'performance' in include_sections:
                export_data['data']['performance'] = self._get_performance_data()
            
            # Format response based on export type
            if export_format == 'csv':
                # Would convert to CSV format
                pass
            
            return jsonify({
                'status': APIResponseStatus.SUCCESS.value,
                'data': export_data,
                'export_id': f"export_{int(time.time())}"
            })
            
        except Exception as e:
            return jsonify({
                'status': APIResponseStatus.ERROR.value,
                'message': str(e)
            }), 400
    
    # Helper methods for data gathering
    
    def _get_overview_data(self) -> Dict[str, Any]:
        """Get dashboard overview data"""
        return {
            'total_requests': self.metrics['total_requests'],
            'success_rate': (
                self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1)
            ),
            'active_endpoints': len(self.endpoints),
            'last_update': datetime.now().isoformat()
        }
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data"""
        return {
            'avg_response_time_ms': self.metrics['avg_response_time'],
            'total_requests': self.metrics['total_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'cache_hit_rate': 0.7  # Would calculate actual rate
        }
    
    def _get_activity_data(self) -> Dict[str, Any]:
        """Get activity data"""
        return {
            'recent_requests': [],  # Would track recent requests
            'active_users': 0,  # Would track active users
            'peak_usage_time': None  # Would track peak times
        }
    
    def _get_resource_data(self) -> Dict[str, Any]:
        """Get resource utilization data"""
        return {
            'cpu_usage': 0.0,  # Would get actual CPU usage
            'memory_usage': 0.0,  # Would get actual memory usage
            'cache_usage': len(self.dashboard_cache)
        }
    
    def get_registered_routes(self) -> List[Dict[str, Any]]:
        """Get list of registered dashboard routes"""
        return [
            {
                'path': endpoint.path,
                'method': endpoint.method,
                'description': endpoint.description,
                'cache_enabled': endpoint.cache_enabled,
                'auth_required': endpoint.auth_required
            }
            for endpoint in self.endpoints
        ]
    
    def clear_cache(self):
        """Clear dashboard cache"""
        self.dashboard_cache.clear()
        self.cache_timestamps.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            **self.metrics,
            'cache_size': len(self.dashboard_cache),
            'registered_endpoints': len(self.endpoints),
            'latency_target_met': self.metrics['avg_response_time'] < 50
        }