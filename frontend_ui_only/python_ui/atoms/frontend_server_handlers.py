#!/usr/bin/env python3
"""
Frontend Server Handlers - Atomic Component
Frontend request handling infrastructure
Agent Z - STEELCLAD Frontend Atomization
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from flask import request, jsonify, Response


class HandlerType(Enum):
    """Types of request handlers"""
    API = "api"
    WEBSOCKET = "websocket"
    STATIC = "static"
    DASHBOARD = "dashboard"
    METRICS = "metrics"


@dataclass
class RequestContext:
    """Request context information"""
    path: str
    method: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    body: Any
    timestamp: datetime
    client_ip: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'path': self.path,
            'method': self.method,
            'headers': self.headers,
            'params': self.params,
            'timestamp': self.timestamp.isoformat(),
            'client_ip': self.client_ip
        }


class FrontendServerHandlers:
    """
    Frontend request handling component
    Manages request processing for dashboard frontends
    """
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        self.error_handlers: Dict[int, Callable] = {}
        
        # Request tracking
        self.request_history = []
        self.max_history = 1000
        
        # Handler metrics
        self.handler_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'handlers_registered': 0,
            'middleware_count': 0
        }
        
        # Response cache
        self.response_cache: Dict[str, Any] = {}
        self.cache_ttl = 60  # seconds
        
        # Setup default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default request handlers"""
        # Default error handlers
        self.error_handlers[404] = self._handle_404
        self.error_handlers[500] = self._handle_500
        self.error_handlers[400] = self._handle_400
    
    def register_handler(self, path: str, handler: Callable, 
                        handler_type: HandlerType = HandlerType.API):
        """
        Register a request handler
        Main interface for adding frontend handlers
        """
        handler_key = f"{handler_type.value}:{path}"
        self.handlers[handler_key] = handler
        self.handler_metrics['handlers_registered'] = len(self.handlers)
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to request processing pipeline"""
        self.middleware.append(middleware)
        self.handler_metrics['middleware_count'] = len(self.middleware)
    
    def handle_frontend_request(self, path: str, method: str = "GET",
                              handler_type: HandlerType = HandlerType.API) -> Response:
        """Process frontend request through handlers"""
        start_time = time.time()
        
        try:
            # Create request context
            context = self._create_request_context(path, method)
            
            # Apply middleware
            for mw in self.middleware:
                context = mw(context)
                if context is None:
                    return self._error_response(403, "Middleware rejected request")
            
            # Check cache for GET requests
            if method == "GET":
                cached = self._get_cached_response(path)
                if cached is not None:
                    return cached
            
            # Find and execute handler
            handler_key = f"{handler_type.value}:{path}"
            
            if handler_key in self.handlers:
                handler = self.handlers[handler_key]
                response = handler(context)
            else:
                # Try pattern matching
                response = self._find_pattern_handler(path, handler_type, context)
            
            # Cache successful GET responses
            if method == "GET" and response.status_code == 200:
                self._cache_response(path, response)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time, True)
            
            # Add to history
            self._add_to_history(context, response.status_code)
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time, False)
            
            return self._error_response(500, str(e))
    
    def _create_request_context(self, path: str, method: str) -> RequestContext:
        """Create request context from Flask request"""
        return RequestContext(
            path=path,
            method=method,
            headers=dict(request.headers) if request else {},
            params=request.args.to_dict() if request and request.args else {},
            body=request.get_json() if request and request.is_json else None,
            timestamp=datetime.now(),
            client_ip=request.remote_addr if request else "127.0.0.1"
        )
    
    def _find_pattern_handler(self, path: str, handler_type: HandlerType, 
                            context: RequestContext) -> Response:
        """Find handler using pattern matching"""
        # Check for wildcard handlers
        for handler_key, handler in self.handlers.items():
            if handler_key.startswith(f"{handler_type.value}:"):
                pattern = handler_key.split(":", 1)[1]
                
                # Simple wildcard matching
                if "*" in pattern:
                    pattern_parts = pattern.split("*")
                    if all(part in path for part in pattern_parts if part):
                        return handler(context)
        
        # No handler found
        return self._error_response(404, "Handler not found")
    
    def _get_cached_response(self, path: str) -> Optional[Response]:
        """Get cached response if available"""
        if path not in self.response_cache:
            return None
        
        cached = self.response_cache[path]
        if time.time() - cached['timestamp'] > self.cache_ttl:
            del self.response_cache[path]
            return None
        
        return cached['response']
    
    def _cache_response(self, path: str, response: Response):
        """Cache response for future requests"""
        self.response_cache[path] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update handler metrics"""
        self.handler_metrics['total_requests'] += 1
        
        if success:
            self.handler_metrics['successful_requests'] += 1
        else:
            self.handler_metrics['failed_requests'] += 1
        
        # Update average processing time
        current_avg = self.handler_metrics['avg_processing_time']
        total = self.handler_metrics['total_requests']
        
        self.handler_metrics['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def _add_to_history(self, context: RequestContext, status_code: int):
        """Add request to history"""
        self.request_history.append({
            'context': context.to_dict(),
            'status_code': status_code,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]
    
    def _error_response(self, status_code: int, message: str) -> Response:
        """Generate error response"""
        if status_code in self.error_handlers:
            return self.error_handlers[status_code](message)
        
        return jsonify({
            'error': True,
            'status_code': status_code,
            'message': message
        }), status_code
    
    def _handle_404(self, message: str) -> Response:
        """Handle 404 errors"""
        return jsonify({
            'error': 'Not Found',
            'message': message,
            'status_code': 404
        }), 404
    
    def _handle_400(self, message: str) -> Response:
        """Handle 400 errors"""
        return jsonify({
            'error': 'Bad Request',
            'message': message,
            'status_code': 400
        }), 400
    
    def _handle_500(self, message: str) -> Response:
        """Handle 500 errors"""
        return jsonify({
            'error': 'Internal Server Error',
            'message': message,
            'status_code': 500
        }), 500
    
    def register_error_handler(self, status_code: int, handler: Callable):
        """Register custom error handler"""
        self.error_handlers[status_code] = handler
    
    def create_api_handler(self, endpoint: str) -> Callable:
        """Create API endpoint handler"""
        def handler(context: RequestContext) -> Response:
            # Process API request
            data = {
                'endpoint': endpoint,
                'method': context.method,
                'params': context.params,
                'timestamp': context.timestamp.isoformat()
            }
            
            return jsonify(data)
        
        return handler
    
    def create_dashboard_handler(self, dashboard_name: str) -> Callable:
        """Create dashboard page handler"""
        def handler(context: RequestContext) -> Response:
            # Return dashboard HTML
            html = f"""
            <html>
            <head><title>{dashboard_name}</title></head>
            <body>
                <h1>{dashboard_name}</h1>
                <p>Path: {context.path}</p>
                <p>Time: {context.timestamp}</p>
            </body>
            </html>
            """
            
            return Response(html, mimetype='text/html')
        
        return handler
    
    def create_metrics_handler(self) -> Callable:
        """Create metrics endpoint handler"""
        def handler(context: RequestContext) -> Response:
            metrics = {
                'handler_metrics': self.handler_metrics,
                'cache_size': len(self.response_cache),
                'history_size': len(self.request_history),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(metrics)
        
        return handler
    
    def get_request_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request history"""
        return self.request_history[-limit:]
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        success_rate = 0.0
        if self.handler_metrics['total_requests'] > 0:
            success_rate = (
                self.handler_metrics['successful_requests'] / 
                self.handler_metrics['total_requests']
            )
        
        return {
            'total_handlers': len(self.handlers),
            'middleware_count': len(self.middleware),
            'error_handlers': len(self.error_handlers),
            'success_rate': success_rate,
            'cache_entries': len(self.response_cache),
            'avg_processing_time_ms': self.handler_metrics['avg_processing_time']
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            **self.handler_metrics,
            'cache_size': len(self.response_cache),
            'history_size': len(self.request_history),
            'error_handlers': len(self.error_handlers),
            'latency_target_met': self.handler_metrics['avg_processing_time'] < 50
        }