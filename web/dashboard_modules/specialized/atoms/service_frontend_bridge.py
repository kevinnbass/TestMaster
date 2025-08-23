#!/usr/bin/env python3
"""
Service Frontend Bridge - Atomic Component
Service → Frontend communication bridge
Agent Z - STEELCLAD Frontend Atomization
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque


class BridgeStatus(Enum):
    """Bridge connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    SYNCING = "syncing"


class DataDirection(Enum):
    """Data flow direction"""
    SERVICE_TO_FRONTEND = "service_to_frontend"
    FRONTEND_TO_SERVICE = "frontend_to_service"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class BridgeMessage:
    """Message structure for bridge communication"""
    message_id: str
    direction: DataDirection
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    requires_ack: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'message_id': self.message_id,
            'direction': self.direction.value,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'requires_ack': self.requires_ack
        }


class ServiceFrontendBridge:
    """
    Service → Frontend bridge component
    Manages communication between backend services and frontend dashboards
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.status = BridgeStatus.DISCONNECTED
        self.max_queue_size = max_queue_size
        
        # Message queues
        self.outbound_queue = deque(maxlen=max_queue_size)  # Service → Frontend
        self.inbound_queue = deque(maxlen=max_queue_size)   # Frontend → Service
        self.priority_queue = deque(maxlen=100)             # High priority messages
        
        # Connection management
        self.connected_frontends: Set[str] = set()
        self.service_endpoints: Dict[str, Callable] = {}
        
        # Message tracking
        self.pending_acks: Dict[str, BridgeMessage] = {}
        self.message_history = deque(maxlen=1000)
        
        # Performance metrics
        self.bridge_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'avg_latency_ms': 0.0,
            'connection_errors': 0,
            'successful_syncs': 0
        }
        
        # Sync state
        self.last_sync_time = None
        self.sync_in_progress = False
        self.sync_interval = 30  # seconds
    
    async def connect_to_frontend(self, frontend_id: str) -> bool:
        """
        Connect bridge to frontend dashboard
        Main interface for establishing bridge connection
        """
        if self.status == BridgeStatus.CONNECTED:
            self.connected_frontends.add(frontend_id)
            return True
        
        try:
            self.status = BridgeStatus.CONNECTING
            
            # Simulate connection establishment
            await asyncio.sleep(0.01)
            
            self.connected_frontends.add(frontend_id)
            self.status = BridgeStatus.CONNECTED
            
            # Send initial sync
            await self._initial_sync(frontend_id)
            
            return True
            
        except Exception:
            self.status = BridgeStatus.ERROR
            self.bridge_metrics['connection_errors'] += 1
            return False
    
    def disconnect_frontend(self, frontend_id: str):
        """Disconnect frontend from bridge"""
        self.connected_frontends.discard(frontend_id)
        
        if not self.connected_frontends:
            self.status = BridgeStatus.DISCONNECTED
    
    def register_service_endpoint(self, name: str, handler: Callable):
        """Register service endpoint handler"""
        self.service_endpoints[name] = handler
    
    async def send_to_frontend(self, data: Dict[str, Any], 
                              priority: int = 1,
                              requires_ack: bool = False) -> str:
        """Send data from service to frontend"""
        message = BridgeMessage(
            message_id=f"msg_{int(time.time() * 1000)}",
            direction=DataDirection.SERVICE_TO_FRONTEND,
            payload=data,
            timestamp=datetime.now(),
            priority=priority,
            requires_ack=requires_ack
        )
        
        # Add to appropriate queue
        if priority > 5:
            self.priority_queue.append(message)
        else:
            self.outbound_queue.append(message)
        
        # Track if acknowledgment needed
        if requires_ack:
            self.pending_acks[message.message_id] = message
        
        # Process immediately if connected
        if self.status == BridgeStatus.CONNECTED:
            await self._process_outbound_message(message)
        
        self.message_history.append(message)
        self.bridge_metrics['messages_sent'] += 1
        
        return message.message_id
    
    async def receive_from_frontend(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Receive data from frontend to service"""
        message = BridgeMessage(
            message_id=f"msg_{int(time.time() * 1000)}",
            direction=DataDirection.FRONTEND_TO_SERVICE,
            payload=data,
            timestamp=datetime.now()
        )
        
        self.inbound_queue.append(message)
        self.message_history.append(message)
        self.bridge_metrics['messages_received'] += 1
        
        # Process through service endpoints
        response = await self._process_inbound_message(message)
        
        return response
    
    async def _process_outbound_message(self, message: BridgeMessage):
        """Process outbound message to frontend"""
        start_time = time.time()
        
        try:
            # Send to all connected frontends
            for frontend_id in self.connected_frontends:
                # Simulate sending (would be actual WebSocket/HTTP call)
                await asyncio.sleep(0.001)
            
            # Update latency metrics
            latency = (time.time() - start_time) * 1000
            self.bridge_metrics['avg_latency_ms'] = (
                (self.bridge_metrics['avg_latency_ms'] * 0.9) + (latency * 0.1)
            )
            
        except Exception:
            self.bridge_metrics['messages_dropped'] += 1
    
    async def _process_inbound_message(self, message: BridgeMessage) -> Dict[str, Any]:
        """Process inbound message from frontend"""
        payload = message.payload
        endpoint = payload.get('endpoint', 'default')
        
        if endpoint in self.service_endpoints:
            handler = self.service_endpoints[endpoint]
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(payload)
                else:
                    result = handler(payload)
                
                return {
                    'success': True,
                    'result': result,
                    'message_id': message.message_id
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message_id': message.message_id
                }
        
        return {
            'success': False,
            'error': 'Unknown endpoint',
            'message_id': message.message_id
        }
    
    async def sync_with_frontend(self) -> bool:
        """Synchronize state between service and frontend"""
        if self.sync_in_progress:
            return False
        
        self.sync_in_progress = True
        self.status = BridgeStatus.SYNCING
        
        try:
            # Gather sync data
            sync_data = {
                'timestamp': datetime.now().isoformat(),
                'service_state': self._get_service_state(),
                'pending_messages': len(self.outbound_queue),
                'connected_frontends': list(self.connected_frontends)
            }
            
            # Send sync to frontend
            await self.send_to_frontend(
                {'type': 'sync', 'data': sync_data},
                priority=10
            )
            
            self.last_sync_time = datetime.now()
            self.bridge_metrics['successful_syncs'] += 1
            
            return True
            
        finally:
            self.sync_in_progress = False
            self.status = BridgeStatus.CONNECTED
    
    async def _initial_sync(self, frontend_id: str):
        """Perform initial sync when frontend connects"""
        initial_data = {
            'type': 'initial_sync',
            'frontend_id': frontend_id,
            'bridge_status': self.status.value,
            'available_endpoints': list(self.service_endpoints.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        await self.send_to_frontend(initial_data, priority=10)
    
    def _get_service_state(self) -> Dict[str, Any]:
        """Get current service state for sync"""
        return {
            'status': self.status.value,
            'connected_frontends': len(self.connected_frontends),
            'registered_endpoints': len(self.service_endpoints),
            'queue_sizes': {
                'outbound': len(self.outbound_queue),
                'inbound': len(self.inbound_queue),
                'priority': len(self.priority_queue)
            }
        }
    
    async def process_queues(self):
        """Process message queues"""
        # Process priority queue first
        while self.priority_queue:
            message = self.priority_queue.popleft()
            await self._process_outbound_message(message)
        
        # Process normal outbound queue
        batch_size = min(10, len(self.outbound_queue))
        for _ in range(batch_size):
            if self.outbound_queue:
                message = self.outbound_queue.popleft()
                await self._process_outbound_message(message)
    
    def acknowledge_message(self, message_id: str):
        """Acknowledge receipt of message"""
        if message_id in self.pending_acks:
            del self.pending_acks[message_id]
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status and metrics"""
        return {
            'status': self.status.value,
            'connected_frontends': list(self.connected_frontends),
            'metrics': self.bridge_metrics.copy(),
            'queue_status': {
                'outbound_pending': len(self.outbound_queue),
                'inbound_pending': len(self.inbound_queue),
                'priority_pending': len(self.priority_queue),
                'awaiting_acks': len(self.pending_acks)
            },
            'last_sync': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'latency_target_met': self.bridge_metrics['avg_latency_ms'] < 50
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform bridge health check"""
        health = {
            'healthy': self.status == BridgeStatus.CONNECTED,
            'status': self.status.value,
            'latency_ok': self.bridge_metrics['avg_latency_ms'] < 50,
            'queues_ok': (
                len(self.outbound_queue) < self.max_queue_size * 0.8 and
                len(self.inbound_queue) < self.max_queue_size * 0.8
            ),
            'error_rate': (
                self.bridge_metrics['messages_dropped'] / 
                max(self.bridge_metrics['messages_sent'], 1)
            )
        }
        
        health['overall'] = (
            health['healthy'] and 
            health['latency_ok'] and 
            health['queues_ok'] and
            health['error_rate'] < 0.05
        )
        
        return health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bridge metrics"""
        return {
            **self.bridge_metrics,
            'status': self.status.value,
            'connected_frontends': len(self.connected_frontends),
            'registered_endpoints': len(self.service_endpoints),
            'message_history_size': len(self.message_history),
            'latency_target_met': self.bridge_metrics['avg_latency_ms'] < 50
        }