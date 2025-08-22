
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
Real-Time Health Dashboard with WebSocket Updates
===============================================

Live streaming dashboard for all robustness features with instant updates.

Author: TestMaster Team
"""

import logging
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Set, Optional
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import asyncio

logger = logging.getLogger(__name__)

class WebSocketHealthDashboard:
    """
    Real-time health dashboard with WebSocket streaming.
    """
    
    def __init__(self, app: Flask, aggregator=None):
        """
        Initialize WebSocket dashboard.
        
        Args:
            app: Flask application
            aggregator: Analytics aggregator instance
        """
        self.app = app
        self.aggregator = aggregator
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=True,
            engineio_logger=True
        )
        
        # Active connections
        self.active_connections: Set[str] = set()
        self.rooms: Dict[str, Set[str]] = {
            'health': set(),
            'analytics': set(),
            'robustness': set(),
            'monitoring': set()
        }
        
        # Update configuration
        self.update_interval = 2.0  # seconds
        self.detailed_interval = 5.0  # seconds
        self.broadcast_active = True
        
        # Cached data for diff detection
        self.last_health_data = {}
        self.last_analytics_data = {}
        self.last_robustness_data = {}
        
        # Statistics
        self.stats = {
            'connections_total': 0,
            'connections_active': 0,
            'messages_sent': 0,
            'updates_pushed': 0,
            'errors': 0
        }
        
        # Setup event handlers
        self._setup_socketio_handlers()
        
        # Start broadcasting thread
        self.broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            daemon=True
        )
        self.broadcast_thread.start()
        
        logger.info("WebSocket Health Dashboard initialized")
    
    def _setup_socketio_handlers(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_id = request.sid
            self.active_connections.add(client_id)
            self.stats['connections_total'] += 1
            self.stats['connections_active'] = len(self.active_connections)
            
            logger.info(f"Client connected: {client_id}")
            
            # Send initial data
            self._send_initial_data(client_id)
            
            # Send connection acknowledgment
            emit('connected', {
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'server_status': 'healthy',
                'available_rooms': list(self.rooms.keys())
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            self.active_connections.discard(client_id)
            self.stats['connections_active'] = len(self.active_connections)
            
            # Remove from all rooms
            for room_name, room_clients in self.rooms.items():
                room_clients.discard(client_id)
            
            logger.info(f"Client disconnected: {client_id}")
        
        @self.socketio.on('join_room')
        def handle_join_room(data):
            """Handle room join request."""
            client_id = request.sid
            room_name = data.get('room')
            
            if room_name in self.rooms:
                join_room(room_name)
                self.rooms[room_name].add(client_id)
                
                emit('room_joined', {
                    'room': room_name,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Send room-specific initial data
                self._send_room_data(client_id, room_name)
                
                logger.debug(f"Client {client_id} joined room: {room_name}")
            else:
                emit('error', {
                    'message': f'Room {room_name} does not exist',
                    'available_rooms': list(self.rooms.keys())
                })
        
        @self.socketio.on('leave_room')
        def handle_leave_room(data):
            """Handle room leave request."""
            client_id = request.sid
            room_name = data.get('room')
            
            if room_name in self.rooms:
                leave_room(room_name)
                self.rooms[room_name].discard(client_id)
                
                emit('room_left', {
                    'room': room_name,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.debug(f"Client {client_id} left room: {room_name}")
        
        @self.socketio.on('get_stats')
        def handle_get_stats():
            """Send dashboard statistics."""
            emit('dashboard_stats', {
                'stats': self.stats,
                'room_counts': {
                    room: len(clients) for room, clients in self.rooms.items()
                },
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('set_update_rate')
        def handle_set_update_rate(data):
            """Set update rate for client."""
            rate = data.get('rate', 2.0)
            if 0.5 <= rate <= 10.0:
                # Store per-client rate if needed
                emit('update_rate_set', {
                    'rate': rate,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                emit('error', {
                    'message': 'Update rate must be between 0.5 and 10.0 seconds'
                })
    
    def _send_initial_data(self, client_id: str):
        """Send initial data to newly connected client."""
        try:
            # Health data
            if self.aggregator:
                health_data = self._get_health_data()
                self.socketio.emit('health_update', health_data, room=client_id)
                
                analytics_data = self._get_analytics_data()
                self.socketio.emit('analytics_update', analytics_data, room=client_id)
                
                robustness_data = self._get_robustness_data()
                self.socketio.emit('robustness_update', robustness_data, room=client_id)
            
            # Dashboard status
            self.socketio.emit('dashboard_status', {
                'status': 'active',
                'features_enabled': {
                    'heartbeat_monitoring': True,
                    'fallback_system': True,
                    'dead_letter_queue': True,
                    'batch_processing': True,
                    'flow_monitoring': True,
                    'compression': True,
                    'recovery_orchestrator': True
                },
                'timestamp': datetime.now().isoformat()
            }, room=client_id)
            
        except Exception as e:
            logger.error(f"Failed to send initial data: {e}")
            self.stats['errors'] += 1
    
    def _send_room_data(self, client_id: str, room_name: str):
        """Send room-specific data to client."""
        try:
            if room_name == 'health' and self.aggregator:
                health_data = self._get_detailed_health_data()
                self.socketio.emit('detailed_health', health_data, room=client_id)
            
            elif room_name == 'analytics' and self.aggregator:
                analytics_data = self._get_detailed_analytics_data()
                self.socketio.emit('detailed_analytics', analytics_data, room=client_id)
            
            elif room_name == 'robustness' and self.aggregator:
                robustness_data = self._get_detailed_robustness_data()
                self.socketio.emit('detailed_robustness', robustness_data, room=client_id)
            
            elif room_name == 'monitoring' and self.aggregator:
                monitoring_data = self._get_monitoring_data()
                self.socketio.emit('monitoring_data', monitoring_data, room=client_id)
        
        except Exception as e:
            logger.error(f"Failed to send room data: {e}")
            self.stats['errors'] += 1
    
    def _broadcast_loop(self):
        """Background loop for broadcasting updates."""
        while self.broadcast_active:
            try:
                if self.active_connections and self.aggregator:
                    # Regular updates
                    self._broadcast_health_updates()
                    self._broadcast_analytics_updates()
                    self._broadcast_robustness_updates()
                    
                    # Update statistics
                    self.stats['updates_pushed'] += 1
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
                self.stats['errors'] += 1
                time.sleep(5)  # Back off on error
    
    def _broadcast_health_updates(self):
        """Broadcast health status updates."""
        try:
            current_data = self._get_health_data()
            
            # Check for changes
            if self._data_changed(current_data, self.last_health_data):
                self.socketio.emit('health_update', current_data, room='health')
                self.last_health_data = current_data
                self.stats['messages_sent'] += 1
        
        except Exception as e:
            logger.error(f"Health broadcast failed: {e}")
    
    def _broadcast_analytics_updates(self):
        """Broadcast analytics updates."""
        try:
            current_data = self._get_analytics_data()
            
            # Check for changes
            if self._data_changed(current_data, self.last_analytics_data):
                self.socketio.emit('analytics_update', current_data, room='analytics')
                self.last_analytics_data = current_data
                self.stats['messages_sent'] += 1
        
        except Exception as e:
            logger.error(f"Analytics broadcast failed: {e}")
    
    def _broadcast_robustness_updates(self):
        """Broadcast robustness updates."""
        try:
            current_data = self._get_robustness_data()
            
            # Check for changes
            if self._data_changed(current_data, self.last_robustness_data):
                self.socketio.emit('robustness_update', current_data, room='robustness')
                self.last_robustness_data = current_data
                self.stats['messages_sent'] += 1
        
        except Exception as e:
            logger.error(f"Robustness broadcast failed: {e}")
    
    def _get_health_data(self) -> Dict[str, Any]:
        """Get basic health data."""
        if not self.aggregator:
            return {'status': 'no_aggregator', 'timestamp': datetime.now().isoformat()}
        
        try:
            # Heartbeat status
            heartbeat_status = self.aggregator.heartbeat_monitor.get_connection_status()
            
            # Overall health score
            robustness = self.aggregator.get_robustness_monitoring()
            health_score = robustness.get('overall_health_score', 0)
            
            return {
                'overall_health': heartbeat_status.get('overall_health', 'unknown'),
                'health_score': health_score,
                'endpoints': heartbeat_status.get('endpoints', {}),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get health data: {e}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_analytics_data(self) -> Dict[str, Any]:
        """Get basic analytics data."""
        if not self.aggregator:
            return {'status': 'no_aggregator', 'timestamp': datetime.now().isoformat()}
        
        try:
            # Flow monitoring
            flow_summary = self.aggregator.flow_monitor.get_flow_summary()
            
            # Batch processing
            batch_status = self.aggregator.batch_processor.get_status()
            
            return {
                'active_transactions': flow_summary.get('active_transactions', 0),
                'completed_transactions': flow_summary.get('completed_transactions', 0),
                'failed_transactions': flow_summary.get('failed_transactions', 0),
                'pending_items': batch_status.get('pending_items', 0),
                'queued_batches': batch_status.get('queued_batches', 0),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get analytics data: {e}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_robustness_data(self) -> Dict[str, Any]:
        """Get basic robustness data."""
        if not self.aggregator:
            return {'status': 'no_aggregator', 'timestamp': datetime.now().isoformat()}
        
        try:
            # Dead letter queue
            dlq_stats = self.aggregator.dead_letter_queue.get_statistics()
            
            # Fallback system
            fallback_status = self.aggregator.fallback_system.get_fallback_status()
            
            # Compression stats
            compression_stats = self.aggregator.compressor.get_compression_stats()
            
            return {
                'dead_letter_size': dlq_stats.get('queue_size', 0),
                'fallback_level': fallback_status.get('current_level', 'unknown'),
                'compression_efficiency': compression_stats.get('compression_efficiency', 0),
                'recovery_status': 'active',  # Would integrate with recovery orchestrator
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get robustness data: {e}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_detailed_health_data(self) -> Dict[str, Any]:
        """Get detailed health data."""
        basic = self._get_health_data()
        
        if not self.aggregator:
            return basic
        
        try:
            # Add detailed metrics
            robustness = self.aggregator.get_robustness_monitoring()
            
            basic.update({
                'detailed_features': robustness.get('robustness_features', {}),
                'performance_metrics': robustness.get('performance_metrics', {}),
                'system_info': robustness.get('system_info', {})
            })
            
            return basic
        
        except Exception as e:
            logger.error(f"Failed to get detailed health data: {e}")
            return basic
    
    def _get_detailed_analytics_data(self) -> Dict[str, Any]:
        """Get detailed analytics data."""
        basic = self._get_analytics_data()
        
        if not self.aggregator:
            return basic
        
        try:
            # Add flow success rates
            flow_summary = self.aggregator.flow_monitor.get_flow_summary()
            
            basic.update({
                'success_rates': flow_summary.get('success_rates', {}),
                'stage_performance': flow_summary.get('stage_performance', {}),
                'transaction_history': flow_summary.get('recent_transactions', [])[-10:]  # Last 10
            })
            
            return basic
        
        except Exception as e:
            logger.error(f"Failed to get detailed analytics data: {e}")
            return basic
    
    def _get_detailed_robustness_data(self) -> Dict[str, Any]:
        """Get detailed robustness data."""
        basic = self._get_robustness_data()
        
        if not self.aggregator:
            return basic
        
        try:
            # Add comprehensive robustness monitoring
            robustness = self.aggregator.get_robustness_monitoring()
            
            basic.update({
                'all_features': robustness.get('robustness_features', {}),
                'health_checks': robustness.get('health_checks', {}),
                'recovery_actions': robustness.get('recovery_actions', [])
            })
            
            return basic
        
        except Exception as e:
            logger.error(f"Failed to get detailed robustness data: {e}")
            return basic
    
    def _get_monitoring_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data."""
        try:
            if not self.aggregator:
                return {'status': 'no_aggregator', 'timestamp': datetime.now().isoformat()}
            
            # Get all monitoring data
            return {
                'robustness': self.aggregator.get_robustness_monitoring(),
                'timestamp': datetime.now().isoformat(),
                'dashboard_stats': self.stats
            }
        
        except Exception as e:
            logger.error(f"Failed to get monitoring data: {e}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _data_changed(self, current: Dict, previous: Dict) -> bool:
        """Check if data has changed significantly."""
        if not previous:
            return True
        
        # Check key fields for changes
        key_fields = ['overall_health', 'health_score', 'active_transactions', 
                     'dead_letter_size', 'fallback_level', 'pending_items']
        
        for field in key_fields:
            if current.get(field) != previous.get(field):
                return True
        
        return False
    
    def send_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Send alert to all connected clients."""
        try:
            alert_data = {
                'type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            }
            
            self.socketio.emit('alert', alert_data)
            self.stats['messages_sent'] += 1
            
            logger.info(f"Alert sent: {alert_type} - {message}")
        
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            self.stats['errors'] += 1
    
    def broadcast_event(self, event_name: str, data: Dict[str, Any], room: Optional[str] = None):
        """Broadcast custom event."""
        try:
            event_data = {
                **data,
                'timestamp': datetime.now().isoformat()
            }
            
            if room and room in self.rooms:
                self.socketio.emit(event_name, event_data, room=room)
            else:
                self.socketio.emit(event_name, event_data)
            
            self.stats['messages_sent'] += 1
            
        except Exception as e:
            logger.error(f"Failed to broadcast event {event_name}: {e}")
            self.stats['errors'] += 1
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            'stats': self.stats,
            'active_connections': len(self.active_connections),
            'room_counts': {
                room: len(clients) for room, clients in self.rooms.items()
            },
            'broadcast_active': self.broadcast_active,
            'update_interval': self.update_interval
        }
    
    def shutdown(self):
        """Shutdown dashboard."""
        self.broadcast_active = False
        
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=5)
        
        logger.info(f"WebSocket Health Dashboard shutdown - Stats: {self.stats}")

# Global dashboard instance (will be initialized by server)
websocket_dashboard = None