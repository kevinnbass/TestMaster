"""
Real-Time Analytics Flow Tracker
=================================

Provides real-time tracking and monitoring of all analytics flowing
through the system with live dashboards and instant notifications.

Author: TestMaster Team
"""

import logging
import time
import threading
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
import queue

logger = logging.getLogger(__name__)

class TrackingEvent(Enum):
    """Analytics tracking event types."""
    CREATED = "created"
    PROCESSING = "processing"
    TRANSFORMED = "transformed"
    VALIDATED = "validated"
    CACHED = "cached"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRIED = "retried"

class TrackingPriority(Enum):
    """Tracking priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class AnalyticsTrackingEntry:
    """Real-time analytics tracking entry."""
    tracking_id: str
    analytics_id: str
    event: TrackingEvent
    timestamp: datetime
    priority: TrackingPriority
    component: str
    data_size: int
    processing_time: float
    metadata: Dict[str, Any]
    error_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'tracking_id': self.tracking_id,
            'analytics_id': self.analytics_id,
            'event': self.event.value,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'component': self.component,
            'data_size': self.data_size,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'error_info': self.error_info
        }

class RealTimeAnalyticsTracker:
    """
    Real-time analytics flow tracking and monitoring system.
    """
    
    def __init__(self,
                 aggregator=None,
                 max_entries: int = 10000,
                 websocket_port: int = 8765):
        """
        Initialize real-time analytics tracker.
        
        Args:
            aggregator: Analytics aggregator instance
            max_entries: Maximum tracking entries to keep
            websocket_port: WebSocket server port for real-time updates
        """
        self.aggregator = aggregator
        self.max_entries = max_entries
        self.websocket_port = websocket_port
        
        # Tracking storage
        self.tracking_entries: deque = deque(maxlen=max_entries)
        self.active_analytics: Dict[str, Dict[str, Any]] = {}
        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'events_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'data_volume': 0,
            'error_count': 0,
            'last_activity': None
        })
        
        # Real-time subscriptions
        self.websocket_clients: Set = set()
        self.event_filters: Dict[str, Set[TrackingEvent]] = {}
        self.component_filters: Dict[str, Set[str]] = {}
        
        # Event queue for real-time processing
        self.event_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'events_per_second': 0.0,
            'active_analytics_count': 0,
            'completed_analytics_count': 0,
            'failed_analytics_count': 0,
            'average_flow_time': 0.0,
            'peak_concurrent_analytics': 0,
            'data_throughput_bytes_sec': 0.0
        }
        
        # Performance tracking
        self.performance_window = deque(maxlen=100)
        self.last_stats_update = time.time()
        
        # Background threads
        self.tracking_active = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.stats_thread = threading.Thread(
            target=self._stats_loop,
            daemon=True
        )
        self.websocket_thread = threading.Thread(
            target=self._websocket_server,
            daemon=True
        )
        
        # Start threads
        self.processing_thread.start()
        self.stats_thread.start()
        self.websocket_thread.start()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Real-Time Analytics Tracker initialized on port {websocket_port}")
    
    def track_event(self,
                   analytics_id: str,
                   event: TrackingEvent,
                   component: str,
                   data_size: int = 0,
                   processing_time: float = 0.0,
                   priority: TrackingPriority = TrackingPriority.NORMAL,
                   metadata: Optional[Dict[str, Any]] = None,
                   error_info: Optional[str] = None) -> str:
        """
        Track an analytics event.
        
        Args:
            analytics_id: Unique analytics identifier
            event: Type of event
            component: Component generating the event
            data_size: Size of data in bytes
            processing_time: Processing time in seconds
            priority: Event priority
            metadata: Additional metadata
            error_info: Error information if applicable
            
        Returns:
            Tracking ID
        """
        try:
            tracking_id = f"track_{int(time.time() * 1000000)}"
            
            entry = AnalyticsTrackingEntry(
                tracking_id=tracking_id,
                analytics_id=analytics_id,
                event=event,
                timestamp=datetime.now(),
                priority=priority,
                component=component,
                data_size=data_size,
                processing_time=processing_time,
                metadata=metadata or {},
                error_info=error_info
            )
            
            # Add to event queue for processing
            try:
                self.event_queue.put_nowait(entry)
            except queue.Full:
                logger.warning("Event queue full, dropping tracking event")
            
            return tracking_id
            
        except Exception as e:
            logger.error(f"Failed to track event: {e}")
            return ""
    
    def _processing_loop(self):
        """Background event processing loop."""
        while self.tracking_active:
            try:
                # Get events from queue
                try:
                    entry = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the tracking entry
                self._process_tracking_entry(entry)
                
                # Mark queue task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(1)
    
    def _process_tracking_entry(self, entry: AnalyticsTrackingEntry):
        """Process a single tracking entry."""
        with self.lock:
            # Add to tracking entries
            self.tracking_entries.append(entry)
            
            # Update active analytics tracking
            analytics_id = entry.analytics_id
            
            if analytics_id not in self.active_analytics:
                self.active_analytics[analytics_id] = {
                    'start_time': entry.timestamp,
                    'events': [],
                    'current_component': entry.component,
                    'total_processing_time': 0.0,
                    'data_size': 0,
                    'status': 'active'
                }
            
            # Update analytics info
            analytics_info = self.active_analytics[analytics_id]
            analytics_info['events'].append(entry.to_dict())
            analytics_info['current_component'] = entry.component
            analytics_info['total_processing_time'] += entry.processing_time
            analytics_info['data_size'] = max(analytics_info['data_size'], entry.data_size)
            analytics_info['last_event'] = entry.event.value
            analytics_info['last_update'] = entry.timestamp.isoformat()
            
            # Check if analytics completed or failed
            if entry.event in [TrackingEvent.DELIVERED, TrackingEvent.FAILED]:
                analytics_info['status'] = 'completed' if entry.event == TrackingEvent.DELIVERED else 'failed'
                analytics_info['end_time'] = entry.timestamp
                
                # Calculate flow time
                flow_time = (entry.timestamp - analytics_info['start_time']).total_seconds()
                analytics_info['flow_time'] = flow_time
                
                # Update stats
                if entry.event == TrackingEvent.DELIVERED:
                    self.stats['completed_analytics_count'] += 1
                else:
                    self.stats['failed_analytics_count'] += 1
                
                # Move to completed (remove from active after delay)
                self._schedule_analytics_cleanup(analytics_id)
            
            # Update component metrics
            self._update_component_metrics(entry)
            
            # Update general stats
            self.stats['total_events'] += 1
            self.stats['active_analytics_count'] = len(self.active_analytics)
            
            # Track peak concurrent analytics
            current_active = len([a for a in self.active_analytics.values() if a['status'] == 'active'])
            self.stats['peak_concurrent_analytics'] = max(
                self.stats['peak_concurrent_analytics'],
                current_active
            )
            
            # Broadcast to WebSocket clients
            self._broadcast_event(entry)
    
    def _update_component_metrics(self, entry: AnalyticsTrackingEntry):
        """Update metrics for a component."""
        component = entry.component
        metrics = self.component_metrics[component]
        
        metrics['events_processed'] += 1
        metrics['total_processing_time'] += entry.processing_time
        metrics['data_volume'] += entry.data_size
        metrics['last_activity'] = entry.timestamp.isoformat()
        
        if entry.error_info:
            metrics['error_count'] += 1
        
        # Calculate average processing time
        if metrics['events_processed'] > 0:
            metrics['average_processing_time'] = (
                metrics['total_processing_time'] / metrics['events_processed']
            )
    
    def _schedule_analytics_cleanup(self, analytics_id: str):
        """Schedule cleanup of completed analytics."""
        def cleanup():
            time.sleep(60)  # Keep completed analytics for 1 minute
            with self.lock:
                self.active_analytics.pop(analytics_id, None)
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _stats_loop(self):
        """Background statistics calculation loop."""
        while self.tracking_active:
            try:
                time.sleep(5)  # Update stats every 5 seconds
                
                current_time = time.time()
                time_diff = current_time - self.last_stats_update
                
                with self.lock:
                    # Calculate events per second
                    recent_events = [
                        e for e in self.tracking_entries
                        if (datetime.now() - e.timestamp).total_seconds() <= time_diff
                    ]
                    
                    self.stats['events_per_second'] = len(recent_events) / time_diff if time_diff > 0 else 0
                    
                    # Calculate data throughput
                    data_volume = sum(e.data_size for e in recent_events)
                    self.stats['data_throughput_bytes_sec'] = data_volume / time_diff if time_diff > 0 else 0
                    
                    # Calculate average flow time
                    completed_analytics = [
                        a for a in self.active_analytics.values()
                        if a['status'] == 'completed' and 'flow_time' in a
                    ]
                    
                    if completed_analytics:
                        flow_times = [a['flow_time'] for a in completed_analytics]
                        self.stats['average_flow_time'] = sum(flow_times) / len(flow_times)
                    
                    # Store performance data
                    self.performance_window.append({
                        'timestamp': current_time,
                        'events_per_second': self.stats['events_per_second'],
                        'data_throughput': self.stats['data_throughput_bytes_sec'],
                        'active_analytics': self.stats['active_analytics_count']
                    })
                
                self.last_stats_update = current_time
                
            except Exception as e:
                logger.error(f"Stats loop error: {e}")
    
    def _websocket_server(self):
        """WebSocket server for real-time updates."""
        try:
            import asyncio
            import websockets
            
            async def handle_client(websocket, path):
                """Handle WebSocket client connection."""
                self.websocket_clients.add(websocket)
                logger.info(f"WebSocket client connected: {websocket.remote_address}")
                
                try:
                    # Send initial state
                    await self._send_initial_state(websocket)
                    
                    # Handle incoming messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(websocket, data)
                        except json.JSONDecodeError:
                            await websocket.send(json.dumps({
                                'error': 'Invalid JSON format'
                            }))
                        except Exception as e:
                            logger.error(f"WebSocket message handling error: {e}")
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
                except Exception as e:
                    logger.error(f"WebSocket client error: {e}")
                finally:
                    self.websocket_clients.discard(websocket)
            
            # Start WebSocket server
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server = websockets.serve(
                handle_client,
                "localhost",
                self.websocket_port
            )
            
            logger.info(f"WebSocket server starting on port {self.websocket_port}")
            loop.run_until_complete(start_server)
            loop.run_forever()
            
        except ImportError:
            logger.warning("WebSocket support not available (websockets package not installed)")
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    async def _send_initial_state(self, websocket):
        """Send initial state to new WebSocket client."""
        try:
            with self.lock:
                initial_data = {
                    'type': 'initial_state',
                    'data': {
                        'statistics': self.stats,
                        'active_analytics': dict(self.active_analytics),
                        'component_metrics': dict(self.component_metrics),
                        'recent_events': [
                            entry.to_dict() for entry in list(self.tracking_entries)[-50:]
                        ]
                    }
                }
                
                await websocket.send(json.dumps(initial_data, default=str))
                
        except Exception as e:
            logger.error(f"Failed to send initial state: {e}")
    
    async def _handle_websocket_message(self, websocket, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        try:
            message_type = data.get('type')
            
            if message_type == 'subscribe_events':
                # Set event filters for this client
                events = data.get('events', [])
                self.event_filters[websocket] = {TrackingEvent(e) for e in events if e}
                
                await websocket.send(json.dumps({
                    'type': 'subscription_confirmed',
                    'events': events
                }))
                
            elif message_type == 'subscribe_components':
                # Set component filters for this client
                components = data.get('components', [])
                self.component_filters[websocket] = set(components)
                
                await websocket.send(json.dumps({
                    'type': 'subscription_confirmed',
                    'components': components
                }))
                
            elif message_type == 'get_analytics_details':
                # Get detailed analytics information
                analytics_id = data.get('analytics_id')
                if analytics_id in self.active_analytics:
                    await websocket.send(json.dumps({
                        'type': 'analytics_details',
                        'analytics_id': analytics_id,
                        'data': self.active_analytics[analytics_id]
                    }, default=str))
                
            elif message_type == 'get_component_details':
                # Get detailed component metrics
                component = data.get('component')
                if component in self.component_metrics:
                    await websocket.send(json.dumps({
                        'type': 'component_details',
                        'component': component,
                        'data': dict(self.component_metrics[component])
                    }, default=str))
                
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
    
    def _broadcast_event(self, entry: AnalyticsTrackingEntry):
        """Broadcast tracking event to WebSocket clients."""
        if not self.websocket_clients:
            return
        
        try:
            message = {
                'type': 'tracking_event',
                'data': entry.to_dict()
            }
            
            # Send to all connected clients (with filters)
            for websocket in list(self.websocket_clients):
                try:
                    # Check event filters
                    if websocket in self.event_filters:
                        if entry.event not in self.event_filters[websocket]:
                            continue
                    
                    # Check component filters
                    if websocket in self.component_filters:
                        if entry.component not in self.component_filters[websocket]:
                            continue
                    
                    # Send message (non-blocking)
                    asyncio.run_coroutine_threadsafe(
                        websocket.send(json.dumps(message, default=str)),
                        websocket.loop if hasattr(websocket, 'loop') else None
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to broadcast to WebSocket client: {e}")
                    # Remove problematic client
                    self.websocket_clients.discard(websocket)
                    
        except Exception as e:
            logger.error(f"Event broadcast error: {e}")
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get comprehensive tracking summary."""
        with self.lock:
            return {
                'statistics': dict(self.stats),
                'active_analytics': {
                    'count': len(self.active_analytics),
                    'details': dict(self.active_analytics)
                },
                'component_metrics': dict(self.component_metrics),
                'performance_data': list(self.performance_window),
                'websocket_clients': len(self.websocket_clients),
                'recent_events': [
                    entry.to_dict() for entry in list(self.tracking_entries)[-100:]
                ],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_analytics_journey(self, analytics_id: str) -> Optional[Dict[str, Any]]:
        """Get complete journey of specific analytics."""
        with self.lock:
            # Check active analytics
            if analytics_id in self.active_analytics:
                return self.active_analytics[analytics_id]
            
            # Search in tracking entries
            events = [
                entry.to_dict() for entry in self.tracking_entries
                if entry.analytics_id == analytics_id
            ]
            
            if events:
                return {
                    'analytics_id': analytics_id,
                    'events': events,
                    'event_count': len(events),
                    'status': 'historical'
                }
            
            return None
    
    def get_component_performance(self, component: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for specific component."""
        with self.lock:
            if component in self.component_metrics:
                metrics = dict(self.component_metrics[component])
                
                # Add recent events for this component
                recent_events = [
                    entry.to_dict() for entry in self.tracking_entries
                    if entry.component == component and 
                    (datetime.now() - entry.timestamp).total_seconds() <= 300  # Last 5 minutes
                ]
                
                metrics['recent_events'] = recent_events
                metrics['recent_event_count'] = len(recent_events)
                
                return metrics
            
            return None
    
    def shutdown(self):
        """Shutdown real-time analytics tracker."""
        self.tracking_active = False
        
        # Close WebSocket connections
        for websocket in list(self.websocket_clients):
            try:
                asyncio.run_coroutine_threadsafe(
                    websocket.close(),
                    websocket.loop if hasattr(websocket, 'loop') else None
                )
            except:
                pass
        
        # Wait for threads to complete
        for thread in [self.processing_thread, self.stats_thread, self.websocket_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Real-Time Analytics Tracker shutdown - Stats: {self.stats}")

# Global tracker instance
realtime_tracker = None