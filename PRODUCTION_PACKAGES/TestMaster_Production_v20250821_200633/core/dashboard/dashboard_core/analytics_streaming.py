"""
Analytics Streaming Engine
=========================

Real-time analytics streaming with WebSocket support for live dashboard updates.
Provides continuous data flow to ensure analytics reach the dashboard instantly.

Author: TestMaster Team
"""

import asyncio
import json
import logging
import threading
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from collections import defaultdict, deque
import queue

logger = logging.getLogger(__name__)

class AnalyticsStreamManager:
    """
    Manages real-time analytics streaming to connected clients.
    """
    
    def __init__(self, analytics_aggregator=None, stream_interval: float = 1.0):
        """
        Initialize the analytics stream manager.
        
        Args:
            analytics_aggregator: The analytics aggregator instance
            stream_interval: Streaming interval in seconds
        """
        self.analytics_aggregator = analytics_aggregator
        self.stream_interval = stream_interval
        
        # Client management
        self.connected_clients = set()
        self.client_subscriptions = defaultdict(set)  # client_id -> set of metric types
        
        # Streaming state
        self.streaming_active = False
        self.stream_thread = None
        self.message_queue = queue.Queue(maxsize=1000)
        
        # Performance tracking
        self.stream_stats = {
            'messages_sent': 0,
            'clients_connected': 0,
            'errors_encountered': 0,
            'bytes_transmitted': 0,
            'start_time': datetime.now()
        }
        
        # Data transformation pipeline
        self.data_transformers = []
        self.stream_filters = []
        
        # Rate limiting
        self.rate_limits = {
            'max_messages_per_second': 100,
            'max_clients': 50,
            'max_message_size': 1024 * 1024  # 1MB
        }
        
        logger.info("Analytics Stream Manager initialized")
    
    def start_streaming(self):
        """Start the analytics streaming service."""
        if self.streaming_active:
            logger.warning("Analytics streaming already active")
            return
        
        self.streaming_active = True
        self.stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.stream_thread.start()
        
        logger.info("Analytics streaming started")
    
    def stop_streaming(self):
        """Stop the analytics streaming service."""
        self.streaming_active = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        
        # Notify all clients of shutdown
        shutdown_message = {
            'type': 'system',
            'event': 'streaming_stopped',
            'timestamp': datetime.now().isoformat()
        }
        self._broadcast_message(shutdown_message)
        
        logger.info("Analytics streaming stopped")
    
    def add_client(self, client_id: str, websocket=None, subscriptions: List[str] = None):
        """
        Add a client to the streaming service.
        
        Args:
            client_id: Unique client identifier
            websocket: WebSocket connection (if available)
            subscriptions: List of metric types to subscribe to
        """
        if len(self.connected_clients) >= self.rate_limits['max_clients']:
            logger.warning(f"Maximum clients ({self.rate_limits['max_clients']}) reached")
            return False
        
        client_info = {
            'id': client_id,
            'websocket': websocket,
            'connected_at': datetime.now(),
            'messages_sent': 0,
            'last_activity': datetime.now()
        }
        
        self.connected_clients.add(client_id)
        
        # Set up subscriptions
        if subscriptions:
            self.client_subscriptions[client_id].update(subscriptions)
        else:
            # Default subscriptions for all metric types
            self.client_subscriptions[client_id].update([
                'system_metrics', 'test_analytics', 'workflow_analytics',
                'agent_activity', 'security_insights', 'performance_trends'
            ])
        
        self.stream_stats['clients_connected'] = len(self.connected_clients)
        
        # Send welcome message
        welcome_message = {
            'type': 'system',
            'event': 'client_connected',
            'client_id': client_id,
            'subscriptions': list(self.client_subscriptions[client_id]),
            'timestamp': datetime.now().isoformat()
        }
        self._send_to_client(client_id, welcome_message)
        
        logger.info(f"Client {client_id} connected to analytics stream")
        return True
    
    def remove_client(self, client_id: str):
        """Remove a client from the streaming service."""
        if client_id in self.connected_clients:
            self.connected_clients.discard(client_id)
            self.client_subscriptions.pop(client_id, None)
            self.stream_stats['clients_connected'] = len(self.connected_clients)
            logger.info(f"Client {client_id} disconnected from analytics stream")
    
    def subscribe_client(self, client_id: str, metric_types: List[str]):
        """Subscribe a client to specific metric types."""
        if client_id in self.connected_clients:
            self.client_subscriptions[client_id].update(metric_types)
            
            # Send confirmation
            confirmation = {
                'type': 'system',
                'event': 'subscription_updated',
                'client_id': client_id,
                'subscriptions': list(self.client_subscriptions[client_id]),
                'timestamp': datetime.now().isoformat()
            }
            self._send_to_client(client_id, confirmation)
    
    def stream_analytics_update(self, analytics_data: Dict[str, Any], metric_type: str = 'comprehensive'):
        """
        Stream an analytics update to subscribed clients.
        
        Args:
            analytics_data: The analytics data to stream
            metric_type: Type of metrics being streamed
        """
        if not self.streaming_active or not self.connected_clients:
            return
        
        # Apply data transformations
        transformed_data = self._transform_data(analytics_data, metric_type)
        
        # Create stream message
        stream_message = {
            'type': 'analytics_update',
            'metric_type': metric_type,
            'data': transformed_data,
            'timestamp': datetime.now().isoformat(),
            'client_count': len(self.connected_clients)
        }
        
        # Queue message for streaming
        try:
            self.message_queue.put_nowait(stream_message)
        except queue.Full:
            logger.warning("Analytics stream message queue full, dropping message")
            self.stream_stats['errors_encountered'] += 1
    
    def add_data_transformer(self, transformer: Callable[[Dict, str], Dict]):
        """Add a data transformation function to the pipeline."""
        self.data_transformers.append(transformer)
    
    def add_stream_filter(self, filter_func: Callable[[Dict, str], bool]):
        """Add a stream filter function."""
        self.stream_filters.append(filter_func)
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        uptime = (datetime.now() - self.stream_stats['start_time']).total_seconds()
        
        return {
            'streaming_active': self.streaming_active,
            'connected_clients': len(self.connected_clients),
            'messages_sent': self.stream_stats['messages_sent'],
            'errors_encountered': self.stream_stats['errors_encountered'],
            'bytes_transmitted': self.stream_stats['bytes_transmitted'],
            'uptime_seconds': uptime,
            'messages_per_second': self.stream_stats['messages_sent'] / max(uptime, 1),
            'queue_size': self.message_queue.qsize(),
            'client_subscriptions': dict(self.client_subscriptions)
        }
    
    def _streaming_loop(self):
        """Main streaming loop."""
        last_analytics_time = 0
        message_count = 0
        
        while self.streaming_active:
            try:
                current_time = time.time()
                
                # Rate limiting check
                if message_count >= self.rate_limits['max_messages_per_second']:
                    time.sleep(1.0)
                    message_count = 0
                    continue
                
                # Check for queued messages
                try:
                    message = self.message_queue.get_nowait()
                    self._broadcast_message(message)
                    message_count += 1
                    continue
                except queue.Empty:
                    pass
                
                # Periodic analytics streaming
                if current_time - last_analytics_time >= self.stream_interval:
                    if self.analytics_aggregator and self.connected_clients:
                        try:
                            # Get lightweight analytics for streaming
                            analytics = self._get_streaming_analytics()
                            if analytics:
                                self.stream_analytics_update(analytics, 'real_time')
                                last_analytics_time = current_time
                        except Exception as e:
                            logger.error(f"Error getting streaming analytics: {e}")
                            self.stream_stats['errors_encountered'] += 1
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in analytics streaming loop: {e}")
                self.stream_stats['errors_encountered'] += 1
                time.sleep(1.0)
    
    def _get_streaming_analytics(self) -> Dict[str, Any]:
        """Get lightweight analytics optimized for streaming."""
        try:
            # Get only essential metrics for real-time streaming
            if hasattr(self.analytics_aggregator, '_get_system_metrics'):
                system_metrics = self.analytics_aggregator._get_system_metrics()
            else:
                system_metrics = {}
            
            if hasattr(self.analytics_aggregator, 'agent_activity'):
                agent_activity = dict(self.analytics_aggregator.agent_activity)
            else:
                agent_activity = {}
            
            # Lightweight analytics payload
            streaming_data = {
                'system_health': 'healthy' if system_metrics.get('cpu', {}).get('usage_percent', 0) < 80 else 'warning',
                'cpu_usage': system_metrics.get('cpu', {}).get('usage_percent', 0),
                'memory_usage': system_metrics.get('memory', {}).get('percent', 0),
                'disk_usage': system_metrics.get('disk', {}).get('percent', 0),
                'agent_calls': sum(agent_activity.values()),
                'active_agents': len(agent_activity),
                'timestamp': datetime.now().isoformat()
            }
            
            return streaming_data
            
        except Exception as e:
            logger.error(f"Error creating streaming analytics: {e}")
            return {}
    
    def _transform_data(self, data: Dict[str, Any], metric_type: str) -> Dict[str, Any]:
        """Apply data transformations to the analytics data."""
        transformed = data.copy()
        
        for transformer in self.data_transformers:
            try:
                transformed = transformer(transformed, metric_type)
            except Exception as e:
                logger.error(f"Error in data transformer: {e}")
        
        return transformed
    
    def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.connected_clients:
            return
        
        message_str = json.dumps(message)
        message_size = len(message_str.encode('utf-8'))
        
        # Check message size limit
        if message_size > self.rate_limits['max_message_size']:
            logger.warning(f"Message size ({message_size}) exceeds limit, truncating")
            # Truncate the message data
            if 'data' in message:
                message['data'] = {'error': 'Message too large, data truncated'}
                message_str = json.dumps(message)
        
        # Send to subscribed clients
        for client_id in list(self.connected_clients):
            if self._client_subscribed_to_message(client_id, message):
                self._send_to_client(client_id, message, message_str)
    
    def _client_subscribed_to_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Check if a client is subscribed to receive a message."""
        if client_id not in self.client_subscriptions:
            return True  # Default to sending all messages
        
        message_type = message.get('metric_type', message.get('type', 'system'))
        client_subs = self.client_subscriptions[client_id]
        
        # System messages are always sent
        if message_type == 'system':
            return True
        
        # Check specific subscriptions
        return message_type in client_subs or 'all' in client_subs
    
    def _send_to_client(self, client_id: str, message: Dict[str, Any], message_str: str = None):
        """Send a message to a specific client."""
        try:
            if message_str is None:
                message_str = json.dumps(message)
            
            # Update statistics
            self.stream_stats['messages_sent'] += 1
            self.stream_stats['bytes_transmitted'] += len(message_str.encode('utf-8'))
            
            # In a real implementation, this would send via WebSocket
            # For now, we'll just log successful streaming
            logger.debug(f"Streamed message to client {client_id}: {message.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
            self.stream_stats['errors_encountered'] += 1
            # Remove problematic client
            self.remove_client(client_id)


class AnalyticsStreamIntegrator:
    """
    Integrates the streaming system with the analytics aggregator.
    """
    
    def __init__(self, analytics_aggregator, stream_manager: AnalyticsStreamManager):
        """
        Initialize the stream integrator.
        
        Args:
            analytics_aggregator: The analytics aggregator instance
            stream_manager: The stream manager instance
        """
        self.analytics_aggregator = analytics_aggregator
        self.stream_manager = stream_manager
        
        # Hook into analytics aggregator methods
        self._setup_streaming_hooks()
        
        logger.info("Analytics Stream Integrator initialized")
    
    def _setup_streaming_hooks(self):
        """Set up hooks to automatically stream analytics updates."""
        if hasattr(self.analytics_aggregator, 'get_comprehensive_analytics'):
            original_method = self.analytics_aggregator.get_comprehensive_analytics
            
            def streaming_wrapper(*args, **kwargs):
                analytics = original_method(*args, **kwargs)
                # Stream the analytics data
                self.stream_manager.stream_analytics_update(analytics, 'comprehensive')
                return analytics
            
            self.analytics_aggregator.get_comprehensive_analytics = streaming_wrapper
        
        # Hook other key methods for real-time streaming
        self._hook_method('update_test_metrics', 'test_analytics')
        self._hook_method('update_workflow_metrics', 'workflow_analytics')
        self._hook_method('record_agent_activity', 'agent_activity')
        self._hook_method('update_bridge_status', 'bridge_status')
    
    def _hook_method(self, method_name: str, metric_type: str):
        """Hook a specific method for streaming."""
        if hasattr(self.analytics_aggregator, method_name):
            original_method = getattr(self.analytics_aggregator, method_name)
            
            def streaming_wrapper(*args, **kwargs):
                result = original_method(*args, **kwargs)
                # Trigger a lightweight update stream
                if hasattr(self.analytics_aggregator, f'_get_{metric_type}'):
                    try:
                        metric_data = getattr(self.analytics_aggregator, f'_get_{metric_type}')()
                        self.stream_manager.stream_analytics_update({metric_type: metric_data}, metric_type)
                    except Exception as e:
                        logger.debug(f"Could not stream {metric_type} update: {e}")
                return result
            
            setattr(self.analytics_aggregator, method_name, streaming_wrapper)


# Data transformation functions
def compress_analytics_for_streaming(data: Dict[str, Any], metric_type: str) -> Dict[str, Any]:
    """Compress analytics data for efficient streaming."""
    if metric_type == 'real_time':
        # Keep only essential fields for real-time streaming
        essential_fields = ['system_health', 'cpu_usage', 'memory_usage', 'disk_usage', 
                          'agent_calls', 'active_agents', 'timestamp']
        return {k: v for k, v in data.items() if k in essential_fields}
    
    return data

def add_streaming_metadata(data: Dict[str, Any], metric_type: str) -> Dict[str, Any]:
    """Add streaming-specific metadata to analytics data."""
    enhanced = data.copy()
    enhanced['_streaming'] = {
        'compressed': len(str(data)) > 1000,
        'metric_type': metric_type,
        'stream_time': datetime.now().isoformat(),
        'size_bytes': len(str(data).encode('utf-8'))
    }
    return enhanced