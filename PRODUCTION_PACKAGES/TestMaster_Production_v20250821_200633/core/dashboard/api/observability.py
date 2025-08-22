
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
TestMaster Observability Integration
====================================

Enhanced monitoring and telemetry integration using AgentOps patterns.
Provides comprehensive tracking for crew/swarm orchestration and test execution.

Author: TestMaster Team
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from flask import Blueprint, jsonify, request
from functools import wraps

# Core observability components
class TestMasterObservability:
    """
    Core observability system for TestMaster with AgentOps-inspired patterns.
    Provides session tracking, performance monitoring, and telemetry collection.
    """
    
    def __init__(self):
        self.sessions = {}
        self.events = []
        self.metrics = {
            'total_sessions': 0,
            'total_events': 0,
            'total_execution_time': 0,
            'error_count': 0,
            'success_count': 0
        }
        self.active_traces = {}
        self.performance_data = {}
        
        # Initialize logger
        self.logger = logging.getLogger('TestMasterObservability')
        
    def start_session(self, session_name: str = "testmaster_session", 
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new observability session for tracking operations.
        
        Args:
            session_name: Name for the session
            tags: Optional tags for categorization
            metadata: Additional metadata for the session
            
        Returns:
            session_id: Unique identifier for the session
        """
        session_id = f"session_{int(time.time() * 1000)}_{session_name}"
        
        session_data = {
            'id': session_id,
            'name': session_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'tags': tags or [],
            'metadata': metadata or {},
            'events': [],
            'status': 'active',
            'metrics': {
                'duration': 0,
                'event_count': 0,
                'error_count': 0,
                'success_count': 0
            }
        }
        
        self.sessions[session_id] = session_data
        self.metrics['total_sessions'] += 1
        
        self.logger.info(f"Started observability session: {session_id}")
        return session_id
        
    def end_session(self, session_id: str, status: str = 'success') -> Dict[str, Any]:
        """
        End an observability session and calculate final metrics.
        
        Args:
            session_id: The session to end
            status: Final status ('success', 'error', 'cancelled')
            
        Returns:
            Session summary data
        """
        if session_id not in self.sessions:
            self.logger.warning(f"Session {session_id} not found")
            return {}
            
        session = self.sessions[session_id]
        session['end_time'] = datetime.now().isoformat()
        session['status'] = status
        
        # Calculate duration
        start_time = datetime.fromisoformat(session['start_time'])
        end_time = datetime.fromisoformat(session['end_time'])
        duration = (end_time - start_time).total_seconds()
        session['metrics']['duration'] = duration
        
        # Update global metrics
        self.metrics['total_execution_time'] += duration
        if status == 'success':
            self.metrics['success_count'] += 1
        else:
            self.metrics['error_count'] += 1
            
        self.logger.info(f"Ended session {session_id} with status {status}, duration: {duration:.2f}s")
        return session
        
    def track_event(self, session_id: str, event_type: str, 
                   event_data: Dict[str, Any], 
                   duration: Optional[float] = None) -> str:
        """
        Track an event within a session.
        
        Args:
            session_id: Session to track the event in
            event_type: Type of event ('crew_execution', 'swarm_orchestration', etc.)
            event_data: Event-specific data
            duration: Optional execution duration
            
        Returns:
            event_id: Unique identifier for the event
        """
        event_id = f"event_{int(time.time() * 1000000)}"
        
        event = {
            'id': event_id,
            'session_id': session_id,
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': event_data,
            'duration': duration,
            'status': 'completed'
        }
        
        # Add to session if it exists
        if session_id in self.sessions:
            self.sessions[session_id]['events'].append(event)
            self.sessions[session_id]['metrics']['event_count'] += 1
            
        # Add to global events
        self.events.append(event)
        self.metrics['total_events'] += 1
        
        self.logger.debug(f"Tracked event {event_id} of type {event_type}")
        return event_id
        
    def track_agent_performance(self, agent_id: str, agent_type: str,
                              operation: str, duration: float,
                              success: bool, metadata: Optional[Dict] = None) -> None:
        """
        Track agent performance metrics.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent ('crew_agent', 'swarm_agent', etc.)
            operation: Operation performed
            duration: Time taken for operation
            success: Whether operation was successful
            metadata: Additional performance data
        """
        if agent_id not in self.performance_data:
            self.performance_data[agent_id] = {
                'agent_type': agent_type,
                'operations': [],
                'total_operations': 0,
                'total_duration': 0,
                'success_count': 0,
                'error_count': 0,
                'average_duration': 0
            }
            
        perf_data = self.performance_data[agent_id]
        
        operation_data = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'success': success,
            'metadata': metadata or {}
        }
        
        perf_data['operations'].append(operation_data)
        perf_data['total_operations'] += 1
        perf_data['total_duration'] += duration
        
        if success:
            perf_data['success_count'] += 1
        else:
            perf_data['error_count'] += 1
            
        # Calculate average duration
        perf_data['average_duration'] = perf_data['total_duration'] / perf_data['total_operations']
        
        self.logger.debug(f"Tracked performance for agent {agent_id}: {operation} ({duration:.3f}s)")
        
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a session."""
        if session_id not in self.sessions:
            return {}
            
        session = self.sessions[session_id].copy()
        
        # Add derived metrics
        session['metrics']['success_rate'] = 0
        if session['metrics']['event_count'] > 0:
            session['metrics']['success_rate'] = (
                session['metrics']['success_count'] / session['metrics']['event_count'] * 100
            )
            
        return session
        
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get system-wide observability metrics."""
        total_ops = self.metrics['success_count'] + self.metrics['error_count']
        success_rate = 0
        if total_ops > 0:
            success_rate = (self.metrics['success_count'] / total_ops) * 100
            
        return {
            **self.metrics,
            'success_rate': success_rate,
            'average_session_duration': (
                self.metrics['total_execution_time'] / max(self.metrics['total_sessions'], 1)
            ),
            'active_sessions': len([s for s in self.sessions.values() if s['status'] == 'active']),
            'agent_count': len(self.performance_data)
        }
        
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed performance analytics for all agents."""
        analytics = {
            'total_agents': len(self.performance_data),
            'top_performers': [],
            'performance_trends': {},
            'agent_breakdown': {}
        }
        
        # Calculate top performers by success rate and average duration
        agent_performances = []
        for agent_id, data in self.performance_data.items():
            if data['total_operations'] > 0:
                success_rate = (data['success_count'] / data['total_operations']) * 100
                agent_performances.append({
                    'agent_id': agent_id,
                    'agent_type': data['agent_type'],
                    'success_rate': success_rate,
                    'average_duration': data['average_duration'],
                    'total_operations': data['total_operations']
                })
                
        # Sort by success rate, then by average duration
        analytics['top_performers'] = sorted(
            agent_performances, 
            key=lambda x: (x['success_rate'], -x['average_duration']),
            reverse=True
        )[:10]
        
        # Agent type breakdown
        type_breakdown = {}
        for data in self.performance_data.values():
            agent_type = data['agent_type']
            if agent_type not in type_breakdown:
                type_breakdown[agent_type] = {
                    'count': 0,
                    'total_operations': 0,
                    'total_duration': 0,
                    'success_count': 0
                }
            
            type_breakdown[agent_type]['count'] += 1
            type_breakdown[agent_type]['total_operations'] += data['total_operations']
            type_breakdown[agent_type]['total_duration'] += data['total_duration']
            type_breakdown[agent_type]['success_count'] += data['success_count']
            
        analytics['agent_breakdown'] = type_breakdown
        
        return analytics


# Decorator for automatic performance tracking
def track_performance(operation_name: str = None, agent_type: str = "default"):
    """
    Decorator to automatically track function performance.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        agent_type: Type of agent performing the operation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get observability instance from global context or create one
            if not hasattr(wrapper, '_observability'):
                wrapper._observability = TestMasterObservability()
                
            op_name = operation_name or func.__name__
            agent_id = f"{agent_type}_{func.__name__}"
            
            start_time = time.time()
            success = False
            error = None
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Track performance
                wrapper._observability.track_agent_performance(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    operation=op_name,
                    duration=duration,
                    success=success,
                    metadata={'error': error} if error else None
                )
                
        return wrapper
    return decorator


# Global observability instance
observability = TestMasterObservability()

# Flask Blueprint for observability API
observability_bp = Blueprint('observability', __name__)

@observability_bp.route('/sessions', methods=['POST'])
def start_observability_session():
    """Start a new observability session."""
    try:
        data = request.get_json() or {}
        session_name = data.get('name', 'testmaster_session')
        tags = data.get('tags', [])
        metadata = data.get('metadata', {})
        
        session_id = observability.start_session(session_name, tags, metadata)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': 'Observability session started'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@observability_bp.route('/sessions/<session_id>', methods=['DELETE'])
def end_observability_session(session_id):
    """End an observability session."""
    try:
        data = request.get_json() or {}
        status = data.get('status', 'success')
        
        session_summary = observability.end_session(session_id, status)
        
        return jsonify({
            'status': 'success',
            'session_summary': session_summary,
            'message': 'Session ended successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@observability_bp.route('/sessions/<session_id>/events', methods=['POST'])
def track_session_event(session_id):
    """Track an event in a session."""
    try:
        data = request.get_json() or {}
        event_type = data.get('type', 'unknown')
        event_data = data.get('data', {})
        duration = data.get('duration')
        
        event_id = observability.track_event(session_id, event_type, event_data, duration)
        
        return jsonify({
            'status': 'success',
            'event_id': event_id,
            'message': 'Event tracked successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@observability_bp.route('/sessions/<session_id>/summary', methods=['GET'])
def get_session_summary(session_id):
    """Get session summary and metrics."""
    try:
        summary = observability.get_session_summary(session_id)
        
        if not summary:
            return jsonify({
                'status': 'error',
                'error': 'Session not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'session_summary': summary
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@observability_bp.route('/metrics', methods=['GET'])
def get_global_metrics():
    """Get global observability metrics."""
    try:
        metrics = observability.get_global_metrics()
        
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@observability_bp.route('/performance', methods=['GET'])
def get_performance_analytics():
    """Get detailed performance analytics."""
    try:
        analytics = observability.get_performance_analytics()
        
        return jsonify({
            'status': 'success',
            'analytics': analytics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@observability_bp.route('/agents/<agent_id>/performance', methods=['POST'])
def track_agent_performance_endpoint(agent_id):
    """Track performance metrics for a specific agent."""
    try:
        data = request.get_json() or {}
        agent_type = data.get('agent_type', 'unknown')
        operation = data.get('operation', 'unknown')
        duration = data.get('duration', 0)
        success = data.get('success', True)
        metadata = data.get('metadata', {})
        
        observability.track_agent_performance(
            agent_id, agent_type, operation, duration, success, metadata
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Performance data tracked'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@observability_bp.route('/health', methods=['GET'])
def observability_health_check():
    """Health check for observability system."""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len([s for s in observability.sessions.values() if s['status'] == 'active']),
            'total_events': len(observability.events),
            'tracked_agents': len(observability.performance_data)
        }
        
        return jsonify(health_data)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Integration helpers
def integrate_with_crew_orchestration():
    """Helper to integrate observability with crew orchestration."""
    # This would be called from crew_orchestration.py to add tracking
    pass

def integrate_with_swarm_orchestration():
    """Helper to integrate observability with swarm orchestration."""
    # This would be called from swarm_orchestration.py to add tracking
    pass

# Export key components
__all__ = [
    'TestMasterObservability',
    'observability',
    'observability_bp',
    'track_performance',
    'integrate_with_crew_orchestration',
    'integrate_with_swarm_orchestration'
]