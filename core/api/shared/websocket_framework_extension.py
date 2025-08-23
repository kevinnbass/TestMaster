"""
WebSocket Framework Extension - Agent E Consolidation
====================================================

Unified WebSocket application framework extending SharedFlaskFramework
to eliminate WebSocket duplication and standardize real-time communication.

Created: 2025-08-22 21:15:00
Author: Agent E (Latin Swarm)
Protocol: GOLDCLAD Anti-Duplication + IRONCLAD Consolidation
"""

from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from typing import Dict, Any, Optional, Set, List, Callable, Union
import logging
from datetime import datetime
import threading
import time
import json
import asyncio
from functools import wraps
from .shared_flask_framework import (
    BaseFlaskApp, SecurityMiddlewareComponent, 
    AsyncRouteManager, BlueprintRegistrationManager
)


class WebSocketSecurityFramework:
    """
    WebSocket Security Framework extending Agent D Security Integration
    
    Standardizes security patterns across all WebSocket implementations
    """
    
    def __init__(self, security_component: SecurityMiddlewareComponent):
        self.security_component = security_component
        self.authenticated_sessions: Dict[str, Dict[str, Any]] = {}
        self.security_enabled = security_component.is_enabled
        self.logger = logging.getLogger(__name__)
    
    def authenticate_websocket_connection(self, session_id: str, auth_data: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Authenticate WebSocket connection using Agent D security"""
        if not self.security_enabled:
            return True, {'status': 'security_disabled'}
        
        try:
            # Simulate HTTP request data for Agent D validation
            request_data = {
                'ip_address': request.environ.get('REMOTE_ADDR', 'unknown'),
                'endpoint': '/websocket/auth',
                'method': 'WS_AUTH',
                'user_agent': request.headers.get('User-Agent', ''),
                'body': auth_data,
                'query_params': {},
                'headers': dict(request.headers),
                'session_id': session_id
            }
            
            # Use Agent D security validation
            valid, security_data = self.security_component._api_security.validate_request(request_data)
            
            if valid:
                self.authenticated_sessions[session_id] = {
                    'authenticated': True,
                    'timestamp': datetime.now().isoformat(),
                    'security_data': security_data,
                    'auth_data': auth_data
                }
                return True, {'status': 'authenticated', 'session_id': session_id}
            else:
                return False, {'status': 'authentication_failed', 'reason': security_data}
                
        except Exception as e:
            self.logger.error(f"WebSocket authentication failed: {e}")
            return False, {'status': 'authentication_error', 'error': str(e)}
    
    def is_session_authenticated(self, session_id: str) -> bool:
        """Check if WebSocket session is authenticated"""
        if not self.security_enabled:
            return True
        return session_id in self.authenticated_sessions
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get authentication info for session"""
        return self.authenticated_sessions.get(session_id)
    
    def deauthenticate_session(self, session_id: str):
        """Remove session authentication"""
        self.authenticated_sessions.pop(session_id, None)


class WebSocketConnectionManager:
    """
    Advanced WebSocket connection management
    
    Consolidates connection patterns from websocket_dashboard.py
    """
    
    def __init__(self):
        self.active_connections: Set[str] = set()
        self.rooms: Dict[str, Set[str]] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            'connections_total': 0,
            'connections_active': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'rooms_active': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def add_connection(self, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Add new WebSocket connection"""
        self.active_connections.add(session_id)
        self.stats['connections_total'] += 1
        self.stats['connections_active'] = len(self.active_connections)
        
        self.connection_metadata[session_id] = {
            'connected_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'rooms': set(),
            'metadata': metadata or {}
        }
        
        self.logger.info(f"WebSocket connection added: {session_id}")
    
    def remove_connection(self, session_id: str):
        """Remove WebSocket connection"""
        self.active_connections.discard(session_id)
        self.stats['connections_active'] = len(self.active_connections)
        
        # Remove from all rooms
        if session_id in self.connection_metadata:
            for room_name in self.connection_metadata[session_id]['rooms']:
                if room_name in self.rooms:
                    self.rooms[room_name].discard(session_id)
                    if not self.rooms[room_name]:
                        del self.rooms[room_name]
        
        self.connection_metadata.pop(session_id, None)
        self._update_room_stats()
        
        self.logger.info(f"WebSocket connection removed: {session_id}")
    
    def join_room(self, session_id: str, room_name: str) -> bool:
        """Add connection to room"""
        if session_id not in self.active_connections:
            return False
        
        if room_name not in self.rooms:
            self.rooms[room_name] = set()
        
        self.rooms[room_name].add(session_id)
        
        if session_id in self.connection_metadata:
            self.connection_metadata[session_id]['rooms'].add(room_name)
        
        self._update_room_stats()
        return True
    
    def leave_room(self, session_id: str, room_name: str) -> bool:
        """Remove connection from room"""
        if room_name in self.rooms:
            self.rooms[room_name].discard(session_id)
            if not self.rooms[room_name]:
                del self.rooms[room_name]
        
        if session_id in self.connection_metadata:
            self.connection_metadata[session_id]['rooms'].discard(room_name)
        
        self._update_room_stats()
        return True
    
    def get_room_connections(self, room_name: str) -> Set[str]:
        """Get all connections in room"""
        return self.rooms.get(room_name, set()).copy()
    
    def update_activity(self, session_id: str):
        """Update last activity timestamp"""
        if session_id in self.connection_metadata:
            self.connection_metadata[session_id]['last_activity'] = datetime.now().isoformat()
    
    def _update_room_stats(self):
        """Update room statistics"""
        self.stats['rooms_active'] = len(self.rooms)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        return {
            **self.stats,
            'room_details': {
                room: len(connections) for room, connections in self.rooms.items()
            },
            'timestamp': datetime.now().isoformat()
        }


class WebSocketMessageRouter:
    """
    Message routing and broadcasting system
    
    Consolidates message patterns from WebSocket implementations
    """
    
    def __init__(self, socketio: SocketIO, connection_manager: WebSocketConnectionManager):
        self.socketio = socketio
        self.connection_manager = connection_manager
        self.message_handlers: Dict[str, Callable] = {}
        self.broadcast_filters: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
    
    def register_broadcast_filter(self, room_name: str, filter_func: Callable):
        """Register filter function for room broadcasts"""
        self.broadcast_filters[room_name] = filter_func
    
    def route_message(self, session_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Route message to appropriate handler"""
        try:
            if message_type in self.message_handlers:
                self.connection_manager.update_activity(session_id)
                self.connection_manager.stats['messages_received'] += 1
                
                handler = self.message_handlers[message_type]
                return handler(session_id, data)
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Message routing failed: {e}")
            self.connection_manager.stats['errors'] += 1
            return False
    
    def broadcast_to_room(self, room_name: str, event: str, data: Dict[str, Any], 
                         filter_data: bool = True) -> int:
        """Broadcast message to room with optional filtering"""
        try:
            # Apply filter if configured
            if filter_data and room_name in self.broadcast_filters:
                filter_func = self.broadcast_filters[room_name]
                data = filter_func(data)
            
            # Get room connections
            connections = self.connection_manager.get_room_connections(room_name)
            if not connections:
                return 0
            
            # Broadcast message
            self.socketio.emit(event, data, room=room_name)
            self.connection_manager.stats['messages_sent'] += len(connections)
            
            return len(connections)
            
        except Exception as e:
            self.logger.error(f"Room broadcast failed: {e}")
            self.connection_manager.stats['errors'] += 1
            return 0
    
    def send_to_session(self, session_id: str, event: str, data: Dict[str, Any]) -> bool:
        """Send message to specific session"""
        try:
            if session_id in self.connection_manager.active_connections:
                self.socketio.emit(event, data, room=session_id)
                self.connection_manager.stats['messages_sent'] += 1
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Session message failed: {e}")
            self.connection_manager.stats['errors'] += 1
            return False


class WebSocketBroadcastManager:
    """
    Automated broadcasting system for real-time updates
    
    Consolidates broadcast patterns from websocket_dashboard.py
    """
    
    def __init__(self, message_router: WebSocketMessageRouter):
        self.message_router = message_router
        self.broadcast_threads: Dict[str, threading.Thread] = {}
        self.broadcast_configs: Dict[str, Dict[str, Any]] = {}
        self.active_broadcasts: Set[str] = set()
        self.logger = logging.getLogger(__name__)
    
    def register_broadcast(self, broadcast_id: str, config: Dict[str, Any]):
        """Register automated broadcast"""
        self.broadcast_configs[broadcast_id] = {
            'interval': config.get('interval', 2.0),
            'room': config.get('room', 'default'),
            'event': config.get('event', 'update'),
            'data_source': config.get('data_source'),  # Callable that returns data
            'filter_changes': config.get('filter_changes', True),
            'last_data': None
        }
    
    def start_broadcast(self, broadcast_id: str) -> bool:
        """Start automated broadcast"""
        if broadcast_id not in self.broadcast_configs:
            return False
        
        if broadcast_id in self.active_broadcasts:
            return True
        
        config = self.broadcast_configs[broadcast_id]
        thread = threading.Thread(
            target=self._broadcast_loop,
            args=(broadcast_id, config),
            daemon=True
        )
        
        self.broadcast_threads[broadcast_id] = thread
        self.active_broadcasts.add(broadcast_id)
        thread.start()
        
        self.logger.info(f"Started broadcast: {broadcast_id}")
        return True
    
    def stop_broadcast(self, broadcast_id: str):
        """Stop automated broadcast"""
        self.active_broadcasts.discard(broadcast_id)
        self.logger.info(f"Stopped broadcast: {broadcast_id}")
    
    def _broadcast_loop(self, broadcast_id: str, config: Dict[str, Any]):
        """Background broadcast loop"""
        while broadcast_id in self.active_broadcasts:
            try:
                # Get data from source
                data_source = config['data_source']
                if data_source and callable(data_source):
                    current_data = data_source()
                    
                    # Check for changes if filtering enabled
                    should_broadcast = True
                    if config['filter_changes']:
                        if current_data == config['last_data']:
                            should_broadcast = False
                        else:
                            config['last_data'] = current_data
                    
                    # Broadcast if needed
                    if should_broadcast and current_data:
                        self.message_router.broadcast_to_room(
                            config['room'],
                            config['event'], 
                            current_data
                        )
                
                time.sleep(config['interval'])
                
            except Exception as e:
                self.logger.error(f"Broadcast loop error for {broadcast_id}: {e}")
                time.sleep(5)  # Back off on error


class BaseWebSocketApp(BaseFlaskApp):
    """
    Base WebSocket Application Factory extending BaseFlaskApp
    
    Eliminates WebSocket duplication patterns across implementations
    """
    
    def __init__(self, app_name: str = __name__, socketio_config: Optional[Dict[str, Any]] = None, **config):
        # Initialize base Flask app
        super().__init__(app_name, **config)
        
        # WebSocket-specific components
        self.websocket_security = WebSocketSecurityFramework(self.security)
        self.connection_manager = WebSocketConnectionManager()
        
        # Initialize SocketIO
        socketio_defaults = {
            'cors_allowed_origins': "*",
            'async_mode': 'threading',
            'logger': True,
            'engineio_logger': False
        }
        socketio_config = {**socketio_defaults, **(socketio_config or {})}
        
        self.socketio = SocketIO(self.app, **socketio_config)
        
        # Initialize message routing and broadcasting
        self.message_router = WebSocketMessageRouter(self.socketio, self.connection_manager)
        self.broadcast_manager = WebSocketBroadcastManager(self.message_router)
        
        # Setup standard WebSocket handlers
        self._setup_websocket_handlers()
        
        self.logger.info("BaseWebSocketApp initialized")
    
    def _setup_websocket_handlers(self):
        """Setup standard WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            """Handle client connection with optional authentication"""
            session_id = request.sid
            
            # Agent D security integration
            if self.websocket_security.security_enabled and auth:
                authenticated, auth_result = self.websocket_security.authenticate_websocket_connection(
                    session_id, auth
                )
                if not authenticated:
                    emit('auth_failed', auth_result)
                    disconnect()
                    return
            
            # Add connection
            self.connection_manager.add_connection(session_id, {
                'user_agent': request.headers.get('User-Agent', ''),
                'remote_addr': request.environ.get('REMOTE_ADDR', 'unknown')
            })
            
            # Send connection acknowledgment
            emit('connected', {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'server_status': 'healthy',
                'security_enabled': self.websocket_security.security_enabled
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            session_id = request.sid
            
            # Clean up authentication
            self.websocket_security.deauthenticate_session(session_id)
            
            # Remove connection
            self.connection_manager.remove_connection(session_id)
        
        @self.socketio.on('join_room')
        def handle_join_room(data):
            """Handle room join requests"""
            session_id = request.sid
            room_name = data.get('room')
            
            if not room_name:
                emit('error', {'message': 'Room name required'})
                return
            
            # Check authentication if required
            if self.websocket_security.security_enabled:
                if not self.websocket_security.is_session_authenticated(session_id):
                    emit('error', {'message': 'Authentication required'})
                    return
            
            # Join room
            if self.connection_manager.join_room(session_id, room_name):
                join_room(room_name)
                emit('room_joined', {
                    'room': room_name,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                emit('error', {'message': 'Failed to join room'})
        
        @self.socketio.on('leave_room')
        def handle_leave_room(data):
            """Handle room leave requests"""
            session_id = request.sid
            room_name = data.get('room')
            
            if room_name:
                self.connection_manager.leave_room(session_id, room_name)
                leave_room(room_name)
                emit('room_left', {
                    'room': room_name,
                    'timestamp': datetime.now().isoformat()
                })
        
        @self.socketio.on('get_stats')
        def handle_get_stats():
            """Send connection statistics"""
            stats = self.connection_manager.get_connection_stats()
            emit('stats', stats)
        
        @self.socketio.on('message')
        def handle_message(data):
            """Handle generic messages with routing"""
            session_id = request.sid
            message_type = data.get('type', 'generic')
            message_data = data.get('data', {})
            
            self.message_router.route_message(session_id, message_type, message_data)
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register custom message handler"""
        self.message_router.register_message_handler(message_type, handler)
    
    def register_broadcast(self, broadcast_id: str, config: Dict[str, Any]):
        """Register automated broadcast"""
        self.broadcast_manager.register_broadcast(broadcast_id, config)
    
    def start_broadcast(self, broadcast_id: str) -> bool:
        """Start automated broadcast"""
        return self.broadcast_manager.start_broadcast(broadcast_id)
    
    def stop_broadcast(self, broadcast_id: str):
        """Stop automated broadcast"""
        self.broadcast_manager.stop_broadcast(broadcast_id)
    
    def get_socketio(self) -> SocketIO:
        """Get SocketIO instance"""
        return self.socketio
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False, **options):
        """Run WebSocket application"""
        self.logger.info(f"Starting WebSocket application on {host}:{port}")
        return self.socketio.run(self.app, host=host, port=port, debug=debug, **options)


def create_websocket_app(app_name: str, 
                        config: Optional[Dict[str, Any]] = None,
                        socketio_config: Optional[Dict[str, Any]] = None) -> tuple[Flask, SocketIO]:
    """
    WebSocket app factory function
    
    Replaces duplicate WebSocket app creation patterns
    """
    websocket_app = BaseWebSocketApp(app_name, socketio_config, **(config or {}))
    return websocket_app.get_app(), websocket_app.get_socketio()


def create_dashboard_websocket_app(app_name: str = 'dashboard_websocket',
                                  config: Optional[Dict[str, Any]] = None) -> tuple[Flask, SocketIO]:
    """
    Specialized factory for dashboard WebSocket applications
    
    Targets websocket_dashboard.py patterns
    """
    # Dashboard-specific SocketIO configuration
    dashboard_socketio_config = {
        'cors_allowed_origins': "*",
        'async_mode': 'threading', 
        'logger': True,
        'engineio_logger': True,
        'ping_timeout': 10,
        'ping_interval': 5
    }
    
    websocket_app = BaseWebSocketApp(
        app_name, 
        dashboard_socketio_config, 
        **(config or {})
    )
    
    # Add dashboard-specific rooms
    dashboard_rooms = ['health', 'analytics', 'robustness', 'monitoring']
    for room in dashboard_rooms:
        websocket_app.connection_manager.rooms[room] = set()
    
    return websocket_app.get_app(), websocket_app.get_socketio()


# Export WebSocket framework components
__all__ = [
    'BaseWebSocketApp',
    'WebSocketSecurityFramework',
    'WebSocketConnectionManager', 
    'WebSocketMessageRouter',
    'WebSocketBroadcastManager',
    'create_websocket_app',
    'create_dashboard_websocket_app'
]