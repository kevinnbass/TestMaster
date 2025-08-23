#!/usr/bin/env python3
"""
WebSocket Architecture Stream - Agent A Hour 5
Real-time architecture monitoring via WebSocket

Provides real-time streaming of architecture health metrics and updates
to the dashboard using WebSocket connections.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Set, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Import architecture components
from core.architecture.architecture_integration import get_architecture_framework
from core.services.service_registry import get_service_registry

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    # Fallback for when websockets library not available
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = object


class MessageType(Enum):
    """WebSocket message types"""
    ARCHITECTURE_HEALTH = "architecture_health"
    SERVICE_STATUS = "service_status"
    LAYER_COMPLIANCE = "layer_compliance"
    DEPENDENCY_HEALTH = "dependency_health"
    INTEGRATION_STATUS = "integration_status"
    SYSTEM_ALERT = "system_alert"
    HEARTBEAT = "heartbeat"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
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


class ArchitectureWebSocketStream:
    """
    WebSocket stream for real-time architecture monitoring
    
    Streams architecture health metrics, service status updates,
    and system alerts to connected dashboard clients.
    """
    
    def __init__(self, port: int = 8765, update_interval: int = 5):
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.update_interval = update_interval
        
        # Architecture components
        self.framework = get_architecture_framework()
        self.service_registry = get_service_registry()
        
        # WebSocket management
        self.clients: Set[WebSocketServerProtocol] = set()
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.message_sequence = 0
        
        # Optimized stream configuration
        self.stream_config = {
            'architecture_health': True,
            'service_status': True,
            'layer_compliance': True,
            'dependency_health': True,
            'integration_status': True,
            'heartbeat_interval': 15,  # Reduced for better responsiveness
            'enable_compression': True,
            'enable_batching': True,
            'max_message_size': 8192,
            'connection_timeout': 60
        }
        
        # Message queues for different priorities with connection pooling
        self.high_priority_queue: List[WebSocketMessage] = []
        self.normal_priority_queue: List[WebSocketMessage] = []
        self.batch_queue: List[WebSocketMessage] = []
        self.max_batch_size = 10
        self.batch_timeout = 2.0  # seconds
        self.last_batch_time = time.time()
        
        # Enhanced performance metrics
        self.stream_metrics = {
            'messages_sent': 0,
            'clients_connected': 0,
            'clients_disconnected': 0,
            'errors': 0,
            'start_time': datetime.now(),
            'avg_response_time': 0.0,
            'peak_concurrent_clients': 0,
            'messages_batched': 0,
            'compression_ratio': 0.0
        }
        
        self.logger.info(f"Architecture WebSocket Stream initialized on port {port}")
    
    async def register_client(self, websocket: WebSocketServerProtocol, path: str):
        """Register new WebSocket client"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.error("WebSockets library not available")
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
            
            self.logger.info(f"Client registered: {client_id} ({len(self.clients)} total)")
            
            # Send initial data
            await self._send_initial_data(websocket, client_id)
            
            # Handle client messages
            await self._handle_client_messages(websocket, client_id)
            
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
            self.stream_metrics['errors'] += 1
        finally:
            await self._unregister_client(websocket, client_id)
    
    async def _send_initial_data(self, websocket: WebSocketServerProtocol, client_id: str):
        """Send initial data to newly connected client"""
        try:
            # Send current architecture health
            health_data = self.framework.get_architecture_metrics()
            health_message = WebSocketMessage(
                type=MessageType.ARCHITECTURE_HEALTH,
                data=health_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(health_message.to_json())
            
            # Send service registry status
            service_data = self.service_registry.get_registration_report()
            service_message = WebSocketMessage(
                type=MessageType.SERVICE_STATUS,
                data=service_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(service_message.to_json())
            
            self.logger.debug(f"Initial data sent to {client_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send initial data to {client_id}: {e}")
    
    async def _handle_client_messages(self, websocket: WebSocketServerProtocol, client_id: str):
        """Handle incoming messages from client"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_client_message(client_id, data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON from {client_id}: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in message handler for {client_id}: {e}")
    
    async def _process_client_message(self, client_id: str, data: Dict[str, Any]):
        """Process message from client"""
        message_type = data.get('type')
        
        if message_type == 'configure_stream':
            # Update stream configuration for client
            config = data.get('config', {})
            self.stream_config.update(config)
            self.logger.info(f"Stream configuration updated by {client_id}")
            
        elif message_type == 'request_update':
            # Send immediate update
            await self._send_architecture_update(client_id)
            
        elif message_type == 'heartbeat':
            # Respond to heartbeat
            await self._send_heartbeat(client_id)
    
    async def _unregister_client(self, websocket: WebSocketServerProtocol, client_id: str):
        """Unregister WebSocket client"""
        try:
            self.clients.discard(websocket)
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]
            
            self.stream_metrics['clients_disconnected'] += 1
            
            self.logger.info(f"Client unregistered: {client_id} ({len(self.clients)} remaining)")
            
        except Exception as e:
            self.logger.error(f"Error unregistering client {client_id}: {e}")
    
    async def start_streaming(self):
        """Start WebSocket streaming service"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.error("Cannot start WebSocket stream - websockets library not available")
            return
        
        self.running = True
        self.logger.info(f"Starting WebSocket architecture stream on port {self.port}")
        
        # Start background update task
        update_task = asyncio.create_task(self._background_updates())
        
        try:
            # Start WebSocket server
            async with websockets.serve(self.register_client, "localhost", self.port):
                self.logger.info("WebSocket server started")
                await update_task
        
        except Exception as e:
            self.logger.error(f"WebSocket server error: {e}")
            self.running = False
            update_task.cancel()
    
    async def _background_updates(self):
        """Background task for periodic updates"""
        last_heartbeat = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Send regular architecture updates
                if self.clients and self.stream_config.get('architecture_health', True):
                    await self._broadcast_architecture_update()
                
                # Send heartbeat if needed
                if (current_time - last_heartbeat) >= self.stream_config.get('heartbeat_interval', 30):
                    await self._broadcast_heartbeat()
                    last_heartbeat = current_time
                
                # Process message queues
                await self._process_message_queues()
                
                # Wait for next update cycle
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in background updates: {e}")
                self.stream_metrics['errors'] += 1
    
    async def _broadcast_architecture_update(self):
        """Broadcast architecture health update to all clients"""
        try:
            health_data = self.framework.get_architecture_metrics()
            
            message = WebSocketMessage(
                type=MessageType.ARCHITECTURE_HEALTH,
                data=health_data,
                timestamp=datetime.now().isoformat(),
                sequence=self._get_next_sequence()
            )
            
            await self._broadcast_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast architecture update: {e}")
    
    async def _send_architecture_update(self, client_id: str):
        """Send architecture update to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            health_data = self.framework.get_architecture_metrics()
            
            message = WebSocketMessage(
                type=MessageType.ARCHITECTURE_HEALTH,
                data=health_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send update to {client_id}: {e}")
    
    async def _broadcast_heartbeat(self):
        """Broadcast heartbeat to all clients"""
        message = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={'server_time': datetime.now().isoformat()},
            timestamp=datetime.now().isoformat(),
            sequence=self._get_next_sequence()
        )
        
        await self._broadcast_message(message)
    
    async def _send_heartbeat(self, client_id: str):
        """Send heartbeat to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            
            message = WebSocketMessage(
                type=MessageType.HEARTBEAT,
                data={'server_time': datetime.now().isoformat()},
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat to {client_id}: {e}")
    
    async def _broadcast_message(self, message: WebSocketMessage):
        """Broadcast message to all connected clients with connection pooling"""
        if not self.clients:
            return
        
        start_time = time.time()
        message_json = message.to_json()
        
        # Apply compression if enabled and message is large
        if (self.stream_config.get('enable_compression', True) and 
            len(message_json) > 1024):
            try:
                import gzip
                compressed = gzip.compress(message_json.encode())
                if len(compressed) < len(message_json):
                    message_json = compressed.decode('latin-1')
                    self.stream_metrics['compression_ratio'] = len(compressed) / len(message_json)
            except Exception:
                pass  # Use uncompressed if compression fails
        
        # Use asyncio.gather for concurrent message sending
        send_tasks = []
        active_clients = list(self.clients.copy())
        
        for client in active_clients:
            send_tasks.append(self._safe_send_message(client, message_json))
        
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            
            # Track performance metrics
            response_time = time.time() - start_time
            self.stream_metrics['avg_response_time'] = (
                (self.stream_metrics['avg_response_time'] * 0.9) + (response_time * 0.1)
            )
            self.stream_metrics['peak_concurrent_clients'] = max(
                self.stream_metrics['peak_concurrent_clients'], len(active_clients)
            )
            
            # Remove failed clients
            failed_clients = [
                client for client, result in zip(active_clients, results)
                if isinstance(result, Exception)
            ]
            
            for client in failed_clients:
                self.clients.discard(client)
    
    async def _safe_send_message(self, client: WebSocketServerProtocol, message_json: str):
        """Safely send message to client with error handling"""
        try:
            await client.send(message_json)
            self.stream_metrics['messages_sent'] += 1
            return True
        except Exception as e:
            self.logger.warning(f"Failed to send message to client: {e}")
            raise e
    
    async def _process_message_queues(self):
        """Process high and normal priority message queues with batch optimization"""
        current_time = time.time()
        
        # Always process high priority messages immediately
        while self.high_priority_queue:
            message = self.high_priority_queue.pop(0)
            await self._broadcast_message(message)
        
        # Process normal priority messages with batching
        if self.stream_config.get('enable_batching', True):
            # Add normal priority messages to batch queue
            while self.normal_priority_queue:
                message = self.normal_priority_queue.pop(0)
                self.batch_queue.append(message)
            
            # Process batch if conditions are met
            should_process_batch = (
                len(self.batch_queue) >= self.max_batch_size or
                (self.batch_queue and 
                 (current_time - self.last_batch_time) >= self.batch_timeout)
            )
            
            if should_process_batch:
                await self._process_batch_messages()
                self.last_batch_time = current_time
        else:
            # Process normal messages individually if batching disabled
            while self.normal_priority_queue:
                message = self.normal_priority_queue.pop(0)
                await self._broadcast_message(message)
    
    async def _process_batch_messages(self):
        """Process batched messages for improved performance"""
        if not self.batch_queue:
            return
        
        try:
            # Create batch message
            batch_data = {
                'batch_id': f"batch_{int(time.time())}",
                'message_count': len(self.batch_queue),
                'messages': [
                    {
                        'type': msg.type.value,
                        'data': msg.data,
                        'timestamp': msg.timestamp,
                        'sequence': msg.sequence
                    }
                    for msg in self.batch_queue
                ]
            }
            
            batch_message = WebSocketMessage(
                type=MessageType.SYSTEM_ALERT,  # Use system alert for batch messages
                data={
                    'alert_type': 'batch_update',
                    'batch': batch_data
                },
                timestamp=datetime.now().isoformat(),
                sequence=self._get_next_sequence()
            )
            
            await self._broadcast_message(batch_message)
            
            self.stream_metrics['messages_batched'] += len(self.batch_queue)
            self.batch_queue.clear()
            
        except Exception as e:
            self.logger.error(f"Error processing batch messages: {e}")
            # Fallback to individual processing
            for message in self.batch_queue:
                await self._broadcast_message(message)
            self.batch_queue.clear()
    
    def _get_next_sequence(self) -> int:
        """Get next message sequence number"""
        self.message_sequence += 1
        return self.message_sequence
    
    def queue_alert(self, alert_type: str, message: str, priority: str = "normal"):
        """Queue system alert for broadcast"""
        alert_message = WebSocketMessage(
            type=MessageType.SYSTEM_ALERT,
            data={
                'alert_type': alert_type,
                'message': message,
                'priority': priority
            },
            timestamp=datetime.now().isoformat(),
            sequence=self._get_next_sequence()
        )
        
        if priority == "high":
            self.high_priority_queue.append(alert_message)
        else:
            self.normal_priority_queue.append(alert_message)
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get enhanced WebSocket stream performance metrics"""
        uptime = (datetime.now() - self.stream_metrics['start_time']).total_seconds()
        messages_per_second = self.stream_metrics['messages_sent'] / max(uptime, 1)
        
        return {
            'running': self.running,
            'clients_connected': len(self.clients),
            'total_connected': self.stream_metrics['clients_connected'],
            'total_disconnected': self.stream_metrics['clients_disconnected'],
            'messages_sent': self.stream_metrics['messages_sent'],
            'messages_batched': self.stream_metrics['messages_batched'],
            'messages_per_second': round(messages_per_second, 2),
            'avg_response_time_ms': round(self.stream_metrics['avg_response_time'] * 1000, 2),
            'peak_concurrent_clients': self.stream_metrics['peak_concurrent_clients'],
            'compression_ratio': round(self.stream_metrics['compression_ratio'], 3),
            'errors': self.stream_metrics['errors'],
            'uptime_seconds': uptime,
            'websockets_available': WEBSOCKETS_AVAILABLE,
            'port': self.port,
            'update_interval': self.update_interval,
            'batch_queue_size': len(self.batch_queue),
            'high_priority_queue_size': len(self.high_priority_queue),
            'normal_priority_queue_size': len(self.normal_priority_queue),
            'optimizations': {
                'compression_enabled': self.stream_config.get('enable_compression', False),
                'batching_enabled': self.stream_config.get('enable_batching', False),
                'max_batch_size': self.max_batch_size,
                'batch_timeout_seconds': self.batch_timeout,
                'heartbeat_interval': self.stream_config.get('heartbeat_interval', 30)
            }
        }
    
    def stop_streaming(self):
        """Stop WebSocket streaming service"""
        self.running = False
        self.logger.info("WebSocket streaming stopped")


# Global stream instance
_websocket_stream: Optional[ArchitectureWebSocketStream] = None


def get_websocket_stream(port: int = 8765) -> ArchitectureWebSocketStream:
    """Get global WebSocket stream instance"""
    global _websocket_stream
    if _websocket_stream is None:
        _websocket_stream = ArchitectureWebSocketStream(port=port)
    return _websocket_stream


async def start_architecture_stream(port: int = 8765) -> None:
    """Start architecture WebSocket streaming service"""
    stream = get_websocket_stream(port)
    await stream.start_streaming()