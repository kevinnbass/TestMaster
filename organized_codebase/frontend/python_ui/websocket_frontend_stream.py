#!/usr/bin/env python3
"""
WebSocket Frontend Stream - Atomic Component
Handles WebSocket → Frontend streaming with <50ms latency
Agent Z - STEELCLAD Frontend Atomization
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Set, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = object


class MessageType(Enum):
    """WebSocket message types for frontend streaming"""
    ARCHITECTURE_HEALTH = "architecture_health"
    SERVICE_STATUS = "service_status"
    HEARTBEAT = "heartbeat"
    SYSTEM_ALERT = "system_alert"


@dataclass
class WebSocketMessage:
    """WebSocket message structure for frontend"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: str
    client_id: Optional[str] = None
    sequence: int = 0
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            'type': self.type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'client_id': self.client_id,
            'sequence': self.sequence
        })


class WebSocketFrontendStream:
    """
    WebSocket → Frontend streaming component
    Optimized for <50ms latency dashboard updates
    """
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.message_sequence = 0
        
        # Performance tracking
        self.stream_metrics = {
            'messages_sent': 0,
            'clients_connected': 0,
            'avg_response_time': 0.0,
            'start_time': datetime.now()
        }
    
    async def register_client(self, websocket: WebSocketServerProtocol, path: str):
        """Register new WebSocket client for frontend streaming"""
        if not WEBSOCKETS_AVAILABLE:
            return
        
        client_id = f"client_{len(self.clients)}_{int(time.time())}"
        
        try:
            self.clients.add(websocket)
            self.client_metadata[client_id] = {
                'connected_at': datetime.now().isoformat(),
                'path': path,
                'websocket': websocket
            }
            
            self.stream_metrics['clients_connected'] += 1
            
            # Send initial dashboard data
            await self._send_initial_dashboard_data(websocket, client_id)
            
            # Handle client messages
            await self._handle_client_messages(websocket, client_id)
            
        finally:
            await self._unregister_client(websocket, client_id)
    
    async def _send_initial_dashboard_data(self, websocket: WebSocketServerProtocol, client_id: str):
        """Send initial data to newly connected dashboard client"""
        try:
            initial_message = WebSocketMessage(
                type=MessageType.SYSTEM_ALERT,
                data={'status': 'connected', 'message': 'Dashboard connected successfully'},
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(initial_message.to_json())
            
        except Exception:
            pass
    
    async def _handle_client_messages(self, websocket: WebSocketServerProtocol, client_id: str):
        """Handle incoming messages from dashboard client"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_dashboard_request(client_id, data)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
    
    async def _process_dashboard_request(self, client_id: str, data: Dict[str, Any]):
        """Process dashboard request from frontend"""
        message_type = data.get('type')
        
        if message_type == 'request_update':
            await self.send_dashboard_update(client_id, data.get('update_type', 'full'))
        elif message_type == 'heartbeat':
            await self._send_heartbeat(client_id)
    
    async def _unregister_client(self, websocket: WebSocketServerProtocol, client_id: str):
        """Unregister dashboard client"""
        self.clients.discard(websocket)
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
    
    async def stream_to_dashboard(self, data: Dict[str, Any], priority: str = "normal"):
        """
        Stream data to dashboard with <50ms latency
        Main interface for WebSocket → Frontend streaming
        """
        start_time = time.time()
        
        message = WebSocketMessage(
            type=MessageType.SERVICE_STATUS,
            data=data,
            timestamp=datetime.now().isoformat(),
            sequence=self._get_next_sequence()
        )
        
        await self._broadcast_to_dashboard(message, priority)
        
        # Track performance
        response_time = time.time() - start_time
        self.stream_metrics['avg_response_time'] = (
            (self.stream_metrics['avg_response_time'] * 0.9) + (response_time * 0.1)
        )
    
    async def _broadcast_to_dashboard(self, message: WebSocketMessage, priority: str):
        """Broadcast message to all connected dashboard clients"""
        if not self.clients:
            return
        
        message_json = message.to_json()
        
        # High priority messages bypass queue
        if priority == "high":
            send_tasks = [self._safe_send(client, message_json) for client in list(self.clients)]
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
        else:
            # Normal priority with batching
            for client in list(self.clients):
                await self._safe_send(client, message_json)
        
        self.stream_metrics['messages_sent'] += len(self.clients)
    
    async def _safe_send(self, client: WebSocketServerProtocol, message_json: str):
        """Safely send message to client"""
        try:
            await client.send(message_json)
            return True
        except Exception:
            self.clients.discard(client)
            return False
    
    async def send_dashboard_update(self, client_id: str, update_type: str):
        """Send specific update to dashboard client"""
        client_data = self.client_metadata.get(client_id)
        if not client_data:
            return
        
        websocket = client_data['websocket']
        
        update_message = WebSocketMessage(
            type=MessageType.SERVICE_STATUS,
            data={'update_type': update_type, 'timestamp': datetime.now().isoformat()},
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            sequence=self._get_next_sequence()
        )
        
        await self._safe_send(websocket, update_message.to_json())
    
    async def _send_heartbeat(self, client_id: str):
        """Send heartbeat to specific dashboard client"""
        client_data = self.client_metadata.get(client_id)
        if not client_data:
            return
        
        websocket = client_data['websocket']
        
        heartbeat = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={'server_time': datetime.now().isoformat()},
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            sequence=self._get_next_sequence()
        )
        
        await self._safe_send(websocket, heartbeat.to_json())
    
    def _get_next_sequence(self) -> int:
        """Get next message sequence number"""
        self.message_sequence += 1
        return self.message_sequence
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get WebSocket frontend streaming metrics"""
        uptime = (datetime.now() - self.stream_metrics['start_time']).total_seconds()
        
        return {
            'running': self.running,
            'clients_connected': len(self.clients),
            'messages_sent': self.stream_metrics['messages_sent'],
            'avg_response_time_ms': round(self.stream_metrics['avg_response_time'] * 1000, 2),
            'uptime_seconds': uptime,
            'latency_target_met': self.stream_metrics['avg_response_time'] * 1000 < 50
        }