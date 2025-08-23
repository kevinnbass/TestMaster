"""
WebSocket Security Stream Module
Extracted from advanced_security_dashboard.py for Agent X's Epsilon base integration
< 200 lines per STEELCLAD protocol

Provides real-time security monitoring via WebSocket connections with:
- Client connection management
- Real-time analytics streaming
- Security event broadcasting
- Connection pooling and monitoring
"""

import asyncio
import websockets
import uuid
import logging
from typing import Set, Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketSecurityStream:
    """Pluggable WebSocket security streaming module for dashboard integration"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.websocket_server = None
        self.connected_clients: Set[Tuple[str, websockets.WebSocketServerProtocol]] = set()
        self.stream_active = False
        
    async def initialize_security_stream(self):
        """Initialize WebSocket server for security monitoring"""
        if self.stream_active:
            logger.warning("Security stream already active")
            return False
            
        try:
            self.websocket_server = await websockets.serve(
                self._handle_security_client,
                "localhost",
                self.port,
                max_size=1024*1024,  # 1MB max message size
                ping_interval=30,
                ping_timeout=10
            )
            
            self.stream_active = True
            logger.info(f"Security WebSocket stream started on port {self.port}")
            
            # Start monitoring loops
            asyncio.create_task(self._security_analytics_loop())
            asyncio.create_task(self._client_health_monitor())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start security stream: {e}")
            self.stream_active = False
            return False
    
    async def _handle_security_client(self, websocket, path):
        """Handle individual WebSocket client connections for security data"""
        client_id = str(uuid.uuid4())
        self.connected_clients.add((client_id, websocket))
        
        logger.info(f"Security client connected: {client_id}")
        
        try:
            # Send initial security status
            await self._send_security_welcome(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                await self._process_security_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Security client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling security client {client_id}: {e}")
        finally:
            # Clean up client connection
            self.connected_clients = {
                (cid, ws) for cid, ws in self.connected_clients 
                if cid != client_id
            }
    
    async def _send_security_welcome(self, websocket):
        """Send initial security status to new client"""
        welcome_data = {
            "type": "security_welcome",
            "timestamp": datetime.now().isoformat(),
            "connected_clients": len(self.connected_clients),
            "stream_status": "active"
        }
        await websocket.send(str(welcome_data))
    
    async def _process_security_message(self, client_id: str, message: str):
        """Process incoming security messages from clients"""
        try:
            # Handle security commands and data
            logger.debug(f"Security message from {client_id}: {message[:100]}")
            
            # Broadcast security alerts to all clients
            if "alert" in message.lower():
                await self.broadcast_security_event({
                    "type": "security_alert",
                    "source": client_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": message
                })
                
        except Exception as e:
            logger.error(f"Error processing security message: {e}")
    
    async def broadcast_security_event(self, event_data: Dict[str, Any]):
        """Broadcast security events to all connected clients"""
        if not self.connected_clients:
            return
            
        message = str(event_data)
        disconnected_clients = set()
        
        for client_id, websocket in self.connected_clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add((client_id, websocket))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.add((client_id, websocket))
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def _security_analytics_loop(self):
        """Real-time security analytics processing loop"""
        while self.stream_active:
            try:
                # Generate security metrics
                security_metrics = {
                    "type": "security_metrics",
                    "timestamp": datetime.now().isoformat(),
                    "active_clients": len(self.connected_clients),
                    "threat_level": "low",  # Placeholder - integrate with actual threat detection
                    "events_processed": 0   # Placeholder - integrate with event counter
                }
                
                await self.broadcast_security_event(security_metrics)
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in security analytics loop: {e}")
                await asyncio.sleep(10)
    
    async def _client_health_monitor(self):
        """Monitor client connection health"""
        while self.stream_active:
            try:
                # Remove stale connections
                active_clients = set()
                for client_id, websocket in self.connected_clients:
                    if websocket.open:
                        active_clients.add((client_id, websocket))
                
                self.connected_clients = active_clients
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in client health monitor: {e}")
                await asyncio.sleep(30)
    
    async def stop_security_stream(self):
        """Stop the security WebSocket stream"""
        self.stream_active = False
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            
        self.connected_clients.clear()
        logger.info("Security WebSocket stream stopped")
    
    def get_client_count(self) -> int:
        """Get count of active security clients"""
        return len(self.connected_clients)
    
    def is_active(self) -> bool:
        """Check if security stream is active"""
        return self.stream_active

# Plugin interface for Agent X integration
def create_security_stream_plugin(config: Dict[str, Any] = None):
    """Factory function to create WebSocket security stream plugin"""
    port = config.get('port', 8765) if config else 8765
    return WebSocketSecurityStream(port)