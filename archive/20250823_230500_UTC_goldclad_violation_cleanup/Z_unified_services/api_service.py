#!/usr/bin/env python3
"""
Unified API Service Module - Agent Z Phase 2
Consolidated API endpoints and service management

Provides unified REST API endpoints for:  
- Architecture health and service status
- Multi-agent coordination and handoff management
- API cost tracking and budget monitoring
- Cross-agent synthesis and pattern insights
- Performance metrics and system monitoring
- WebSocket service integration
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from functools import wraps
from flask import Flask, jsonify, request, Response
from collections import defaultdict

from .coordination_service import get_coordination_service, AgentStatus, CoordinationType
from .websocket_service import get_websocket_service

logger = logging.getLogger(__name__)


class UnifiedAPIService:
    """
    Unified API service providing consolidated REST endpoints
    for all dashboard and coordination functionality.
    """
    
    def __init__(self, app: Optional[Flask] = None):
        self.app = app or Flask(__name__)
        self.coordination_service = get_coordination_service()
        self.websocket_service = get_websocket_service()
        
        # API rate limiting and caching
        self.request_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 30  # seconds
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_window = 100
        
        # Setup all API routes
        self.setup_routes()
        
        logger.info("Unified API Service initialized")
    
    def setup_routes(self):
        """Setup all unified API endpoints"""
        
        # Health and status endpoints
        @self.app.route('/api/health')
        @self._with_caching(ttl=10)
        @self._with_rate_limiting()
        def health_check():
            """System health check endpoint"""
            return jsonify({
                'service': 'UnifiedAPIService',
                'status': 'healthy',
                'websocket_service': self.websocket_service.health_check(),
                'coordination_service': {
                    'active_agents': len(self.coordination_service.agents),
                    'message_queue_size': len(self.coordination_service.message_queue),
                    'status': 'healthy'
                },
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/service-status')
        @self._with_caching(ttl=30)
        @self._with_rate_limiting()
        def service_status():
            """Comprehensive service status"""
            return jsonify({
                'websocket_metrics': self.websocket_service.get_performance_metrics(),
                'coordination_status': self.coordination_service.get_swarm_status(),
                'api_service_stats': self._get_api_stats(),
                'timestamp': datetime.now().isoformat()
            })
        
        # Multi-agent coordination endpoints
        @self.app.route('/api/agents')
        @self._with_caching(ttl=15)
        @self._with_rate_limiting()
        def get_all_agents():
            """Get all registered agents"""
            swarm_status = self.coordination_service.get_swarm_status()
            return jsonify({
                'agents': swarm_status.get('agent_details', {}),
                'total_count': swarm_status.get('total_agents', 0),
                'active_count': swarm_status.get('active_agents', 0),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/agents/<agent_id>', methods=['GET'])
        @self._with_rate_limiting()
        def get_agent_details(agent_id):
            """Get specific agent details"""
            if agent_id not in self.coordination_service.agents:
                return jsonify({'error': 'Agent not found'}), 404
            
            agent = self.coordination_service.agents[agent_id]
            return jsonify({
                'agent_id': agent_id,
                'details': agent.__dict__,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/agents/<agent_id>/status', methods=['PUT'])
        @self._with_rate_limiting()
        def update_agent_status(agent_id):
            """Update agent status"""
            data = request.get_json() or {}
            
            try:
                status = AgentStatus(data.get('status', 'active'))
                success = self.coordination_service.update_agent_status(
                    agent_id=agent_id,
                    status=status,
                    phase=data.get('phase'),
                    progress=data.get('progress'),
                    tasks=data.get('tasks')
                )
                
                if success:
                    return jsonify({'message': 'Agent status updated successfully'})
                else:
                    return jsonify({'error': 'Failed to update agent status'}), 500
                    
            except ValueError as e:
                return jsonify({'error': f'Invalid status value: {e}'}), 400
        
        @self.app.route('/api/coordination/message', methods=['POST'])
        @self._with_rate_limiting()
        def send_coordination_message():
            """Send coordination message between agents"""
            data = request.get_json() or {}
            
            try:
                coord_type = CoordinationType(data.get('type', 'status_update'))
                message_id = self.coordination_service.send_coordination_message(
                    sender=data.get('sender', 'unknown'),
                    targets=data.get('targets', []),
                    coord_type=coord_type,
                    payload=data.get('payload', {}),
                    priority=data.get('priority', 'normal'),
                    requires_response=data.get('requires_response', False)
                )
                
                return jsonify({
                    'message_id': message_id,
                    'status': 'sent' if message_id else 'failed'
                })
                
            except ValueError as e:
                return jsonify({'error': f'Invalid coordination type: {e}'}), 400
        
        @self.app.route('/api/handoff/request', methods=['POST'])
        @self._with_rate_limiting()
        def request_handoff():
            """Request agent handoff"""
            data = request.get_json() or {}
            
            handoff_id = self.coordination_service.request_agent_handoff(
                from_agent=data.get('from_agent'),
                to_agent=data.get('to_agent'),
                handoff_data=data.get('handoff_data', {})
            )
            
            return jsonify({
                'handoff_id': handoff_id,
                'status': 'requested' if handoff_id else 'failed'
            })
        
        @self.app.route('/api/handoff/<handoff_id>/complete', methods=['POST'])  
        @self._with_rate_limiting()
        def complete_handoff(handoff_id):
            """Complete agent handoff"""
            data = request.get_json() or {}
            success = data.get('success', True)
            
            result = self.coordination_service.complete_agent_handoff(handoff_id, success)
            return jsonify({
                'handoff_id': handoff_id,
                'completed': result,
                'success': success
            })
        
        # WebSocket and real-time endpoints
        @self.app.route('/api/websocket/metrics')
        @self._with_caching(ttl=5)
        @self._with_rate_limiting()
        def websocket_metrics():
            """Get WebSocket performance metrics"""
            return jsonify(self.websocket_service.get_performance_metrics())
        
        @self.app.route('/api/websocket/broadcast', methods=['POST'])
        @self._with_rate_limiting()  
        def broadcast_message():
            """Broadcast message via WebSocket"""
            data = request.get_json() or {}
            message_type = data.get('type', 'generic')
            payload = data.get('payload', {})
            
            try:
                if message_type == 'architecture_health':
                    self.websocket_service.broadcast_architecture_health(payload)
                elif message_type == 'agent_status':
                    self.websocket_service.broadcast_agent_status(
                        data.get('agent_id'), payload
                    )
                elif message_type == 'cost_update':
                    self.websocket_service.broadcast_cost_update(
                        payload.get('provider'), payload.get('model'),
                        payload.get('cost', 0), payload.get('tokens', 0)
                    )
                elif message_type == 'synthesis_insight':
                    self.websocket_service.broadcast_synthesis_insight(
                        data.get('synthesis_id'), payload  
                    )
                elif message_type == 'coordination':
                    self.websocket_service.broadcast_coordination_message(
                        data.get('coord_type'), payload
                    )
                else:
                    return jsonify({'error': 'Unknown message type'}), 400
                
                return jsonify({'status': 'broadcast_sent'})
                
            except Exception as e:
                logger.error(f"Failed to broadcast message: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Cost tracking endpoints (integrated from gamma_alpha functionality)
        @self.app.route('/api/cost/track', methods=['POST'])
        @self._with_rate_limiting()
        def track_api_cost():
            """Track API usage and cost"""
            data = request.get_json() or {}
            
            # Track via WebSocket service API metrics
            cost_data = self.websocket_service.stream.api_metrics.track_api_call(
                provider=data.get('provider', 'unknown'),
                model=data.get('model', 'unknown'),
                input_tokens=data.get('input_tokens', 0),
                output_tokens=data.get('output_tokens', 0)
            )
            
            # Broadcast update
            self.websocket_service.broadcast_cost_update(
                cost_data['provider'], cost_data['model'],
                cost_data['cost'], data.get('input_tokens', 0) + data.get('output_tokens', 0)
            )
            
            return jsonify(cost_data)
        
        @self.app.route('/api/cost/summary')
        @self._with_caching(ttl=60) 
        @self._with_rate_limiting()
        def cost_summary():
            """Get API cost summary"""
            api_metrics = self.websocket_service.stream.api_metrics
            
            return jsonify({
                'api_calls': dict(api_metrics.api_calls),
                'api_costs': api_metrics.api_costs,
                'model_usage': dict(api_metrics.model_usage),
                'daily_budget': api_metrics.daily_budget,
                'budget_alerts': api_metrics.budget_alerts,
                'timestamp': datetime.now().isoformat()
            })
    
    def _with_caching(self, ttl: int = 30):
        """Decorator for endpoint caching"""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                cache_key = f"{request.endpoint}:{request.query_string.decode()}"
                
                # Check cache
                if cache_key in self.request_cache:
                    cached_data, cached_time = self.request_cache[cache_key]
                    if (datetime.now() - cached_time).total_seconds() < ttl:
                        return cached_data
                
                # Execute and cache
                result = f(*args, **kwargs)
                self.request_cache[cache_key] = (result, datetime.now())
                
                # Clean old cache entries
                self._clean_cache()
                
                return result
            return wrapper
        return decorator
    
    def _with_rate_limiting(self):
        """Decorator for rate limiting"""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                client_ip = request.remote_addr or 'unknown'
                current_time = datetime.now()
                
                # Clean old requests
                self.rate_limits[client_ip] = [
                    req_time for req_time in self.rate_limits[client_ip]
                    if (current_time - req_time).total_seconds() < self.rate_limit_window
                ]
                
                # Check rate limit
                if len(self.rate_limits[client_ip]) >= self.max_requests_per_window:
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                
                # Record request
                self.rate_limits[client_ip].append(current_time)
                
                return f(*args, **kwargs)
            return wrapper
        return decorator
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (data, cached_time) in self.request_cache.items()
            if (current_time - cached_time).total_seconds() > self.cache_ttl * 2
        ]
        
        for key in expired_keys:
            del self.request_cache[key]
    
    def _get_api_stats(self) -> Dict[str, Any]:
        """Get API service statistics"""
        total_requests = sum(len(requests) for requests in self.rate_limits.values())
        
        return {
            'total_requests': total_requests,
            'active_clients': len(self.rate_limits),
            'cache_size': len(self.request_cache),
            'rate_limit_window': self.rate_limit_window,
            'max_requests_per_window': self.max_requests_per_window
        }


# Global service instance
_api_service: Optional[UnifiedAPIService] = None


def get_api_service(app: Optional[Flask] = None) -> UnifiedAPIService:
    """Get global API service instance"""
    global _api_service
    if _api_service is None:
        _api_service = UnifiedAPIService(app)
    return _api_service