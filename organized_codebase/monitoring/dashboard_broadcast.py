#!/usr/bin/env python3
"""
Dashboard Broadcast - Atomic Component
Handles dashboard data broadcasting with optimization
Agent Z - STEELCLAD Frontend Atomization
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class BroadcastMessage:
    """Broadcast message structure"""
    channel: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: str = "normal"
    ttl: int = 60  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'channel': self.channel,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'ttl': self.ttl
        }


class DashboardBroadcast:
    """
    Dashboard data broadcasting component
    Optimized for multi-channel dashboard updates
    """
    
    def __init__(self, max_batch_size: int = 10):
        self.channels: Dict[str, Set[str]] = {}  # channel -> subscriber IDs
        self.max_batch_size = max_batch_size
        
        # Message queues
        self.broadcast_queue = deque(maxlen=1000)
        self.batch_buffer: List[BroadcastMessage] = []
        
        # Performance tracking
        self.metrics = {
            'messages_broadcast': 0,
            'batches_sent': 0,
            'subscribers_total': 0,
            'avg_batch_size': 0.0,
            'avg_broadcast_time': 0.0
        }
        
        # Channel-specific data cache
        self.channel_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
    
    def subscribe_to_channel(self, channel: str, subscriber_id: str):
        """Subscribe to a broadcast channel"""
        if channel not in self.channels:
            self.channels[channel] = set()
        
        self.channels[channel].add(subscriber_id)
        self.metrics['subscribers_total'] = sum(len(subs) for subs in self.channels.values())
    
    def unsubscribe_from_channel(self, channel: str, subscriber_id: str):
        """Unsubscribe from a broadcast channel"""
        if channel in self.channels:
            self.channels[channel].discard(subscriber_id)
            
            # Remove empty channels
            if not self.channels[channel]:
                del self.channels[channel]
            
            self.metrics['subscribers_total'] = sum(len(subs) for subs in self.channels.values())
    
    async def broadcast_to_dashboard(self, channel: str, data: Dict[str, Any], 
                                    priority: str = "normal", ttl: int = 60) -> Dict[str, Any]:
        """
        Broadcast data to dashboard channel
        Main interface for dashboard broadcasting
        """
        start_time = time.time()
        
        message = BroadcastMessage(
            channel=channel,
            data=data,
            timestamp=datetime.now(),
            priority=priority,
            ttl=ttl
        )
        
        # Add to queue
        self.broadcast_queue.append(message)
        
        # Update cache
        self._update_channel_cache(channel, data)
        
        # Process based on priority
        if priority == "high":
            result = await self._immediate_broadcast(message)
        else:
            result = await self._batched_broadcast(message)
        
        # Track metrics
        broadcast_time = time.time() - start_time
        self.metrics['avg_broadcast_time'] = (
            (self.metrics['avg_broadcast_time'] * 0.9) + (broadcast_time * 0.1)
        )
        self.metrics['messages_broadcast'] += 1
        
        return result
    
    async def _immediate_broadcast(self, message: BroadcastMessage) -> Dict[str, Any]:
        """Immediately broadcast high-priority message"""
        subscribers = self.channels.get(message.channel, set())
        
        if not subscribers:
            return {'broadcast': False, 'reason': 'no_subscribers'}
        
        # Simulate broadcast to subscribers
        broadcast_tasks = []
        for subscriber_id in subscribers:
            broadcast_tasks.append(self._send_to_subscriber(subscriber_id, message))
        
        if broadcast_tasks:
            results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            
            return {
                'broadcast': True,
                'channel': message.channel,
                'subscribers_reached': successful,
                'total_subscribers': len(subscribers)
            }
        
        return {'broadcast': False, 'reason': 'broadcast_failed'}
    
    async def _batched_broadcast(self, message: BroadcastMessage) -> Dict[str, Any]:
        """Add message to batch for optimized broadcasting"""
        self.batch_buffer.append(message)
        
        # Process batch if full
        if len(self.batch_buffer) >= self.max_batch_size:
            return await self._process_batch()
        
        return {
            'broadcast': 'batched',
            'batch_size': len(self.batch_buffer),
            'max_batch_size': self.max_batch_size
        }
    
    async def _process_batch(self) -> Dict[str, Any]:
        """Process batched messages"""
        if not self.batch_buffer:
            return {'broadcast': False, 'reason': 'empty_batch'}
        
        batch_size = len(self.batch_buffer)
        
        # Group messages by channel
        channel_messages: Dict[str, List[BroadcastMessage]] = {}
        for message in self.batch_buffer:
            if message.channel not in channel_messages:
                channel_messages[message.channel] = []
            channel_messages[message.channel].append(message)
        
        # Broadcast to each channel
        total_sent = 0
        for channel, messages in channel_messages.items():
            subscribers = self.channels.get(channel, set())
            
            if subscribers:
                # Combine messages for batch send
                batch_data = {
                    'batch': True,
                    'messages': [msg.to_dict() for msg in messages],
                    'count': len(messages)
                }
                
                for subscriber_id in subscribers:
                    await self._send_batch_to_subscriber(subscriber_id, channel, batch_data)
                    total_sent += 1
        
        # Clear batch buffer
        self.batch_buffer.clear()
        
        # Update metrics
        self.metrics['batches_sent'] += 1
        self.metrics['avg_batch_size'] = (
            (self.metrics['avg_batch_size'] * 0.9) + (batch_size * 0.1)
        )
        
        return {
            'broadcast': True,
            'batch_size': batch_size,
            'channels': len(channel_messages),
            'total_sent': total_sent
        }
    
    async def _send_to_subscriber(self, subscriber_id: str, 
                                 message: BroadcastMessage) -> bool:
        """Send message to individual subscriber"""
        # Simulate network send with minimal delay
        await asyncio.sleep(0.001)
        return True
    
    async def _send_batch_to_subscriber(self, subscriber_id: str, 
                                       channel: str, batch_data: Dict[str, Any]) -> bool:
        """Send batch to subscriber"""
        # Simulate batch send
        await asyncio.sleep(0.002)
        return True
    
    def _update_channel_cache(self, channel: str, data: Dict[str, Any]):
        """Update channel cache with latest data"""
        self.channel_cache[channel] = data
        self.cache_timestamps[channel] = time.time()
    
    def get_channel_data(self, channel: str, max_age: int = 30) -> Optional[Dict[str, Any]]:
        """Get cached channel data if fresh"""
        if channel not in self.channel_cache:
            return None
        
        age = time.time() - self.cache_timestamps.get(channel, 0)
        if age <= max_age:
            return self.channel_cache[channel]
        
        return None
    
    async def broadcast_system_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast system-wide update to all channels"""
        system_message = {
            'type': 'system_update',
            'update_type': update_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast to all channels
        broadcast_tasks = []
        for channel in self.channels.keys():
            broadcast_tasks.append(
                self.broadcast_to_dashboard(channel, system_message, priority="high")
            )
        
        if broadcast_tasks:
            await asyncio.gather(*broadcast_tasks, return_exceptions=True)
    
    def get_channel_info(self, channel: str) -> Dict[str, Any]:
        """Get information about a specific channel"""
        return {
            'channel': channel,
            'subscribers': len(self.channels.get(channel, set())),
            'has_cache': channel in self.channel_cache,
            'cache_age': (
                time.time() - self.cache_timestamps.get(channel, 0)
                if channel in self.cache_timestamps else None
            )
        }
    
    def get_all_channels(self) -> List[str]:
        """Get list of all active channels"""
        return list(self.channels.keys())
    
    async def flush_batch(self):
        """Force flush of batch buffer"""
        if self.batch_buffer:
            return await self._process_batch()
        return {'broadcast': False, 'reason': 'no_pending_messages'}
    
    def clear_expired_messages(self):
        """Clear expired messages from queue"""
        current_time = datetime.now()
        initial_size = len(self.broadcast_queue)
        
        # Filter out expired messages
        self.broadcast_queue = deque(
            [msg for msg in self.broadcast_queue 
             if (current_time - msg.timestamp).total_seconds() < msg.ttl],
            maxlen=1000
        )
        
        return initial_size - len(self.broadcast_queue)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get broadcast metrics"""
        return {
            **self.metrics,
            'active_channels': len(self.channels),
            'queue_size': len(self.broadcast_queue),
            'batch_buffer_size': len(self.batch_buffer),
            'cache_channels': len(self.channel_cache),
            'latency_target_met': self.metrics['avg_broadcast_time'] * 1000 < 50
        }