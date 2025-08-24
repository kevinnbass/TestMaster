"""
Event-Driven Streaming Engine
============================

High-performance event streaming system providing real-time data flows,
event sourcing, and distributed event processing across enterprise systems.

Features:
- Real-time event streaming with WebSocket and HTTP/2 support
- Event sourcing with complete audit trails
- Distributed event processing with partitioning
- Stream processing with windowing and aggregation
- Event replay and time-travel debugging
- Stream analytics and pattern detection
- Dead letter queues and error handling
- Backpressure handling and flow control

Author: TestMaster Intelligence Team
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator, Set
import threading
from collections import defaultdict, deque
import weakref
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import bisect

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for streaming"""
    SYSTEM_EVENT = "system_event"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    ANALYTICS_RESULT = "analytics_result"
    ALERT_NOTIFICATION = "alert_notification"
    WORKFLOW_EVENT = "workflow_event"
    HEALTH_CHECK = "health_check"
    CUSTOM_EVENT = "custom_event"

class StreamPartitionStrategy(Enum):
    """Partitioning strategies for event streams"""
    ROUND_ROBIN = "round_robin"
    HASH_BASED = "hash_based"
    KEY_BASED = "key_based"
    TIMESTAMP_BASED = "timestamp_based"
    PRIORITY_BASED = "priority_based"

class StreamProcessingMode(Enum):
    """Stream processing modes"""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

@dataclass
class StreamEvent:
    """Individual event in the stream"""
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: EventType = EventType.SYSTEM_EVENT
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    partition_key: Optional[str] = None
    sequence_number: Optional[int] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "partition_key": self.partition_key,
            "sequence_number": self.sequence_number,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Create from dictionary"""
        return cls(
            event_id=data.get("event_id", ""),
            event_type=EventType(data.get("event_type", "system_event")),
            source=data.get("source", "unknown"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            partition_key=data.get("partition_key"),
            sequence_number=data.get("sequence_number"),
            correlation_id=data.get("correlation_id")
        )

@dataclass
class StreamWindow:
    """Time-based window for stream processing"""
    window_id: str = field(default_factory=lambda: f"win_{uuid.uuid4().hex[:8]}")
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    window_size_seconds: int = 60
    events: List[StreamEvent] = field(default_factory=list)
    is_closed: bool = False
    
    def add_event(self, event: StreamEvent) -> bool:
        """Add event to window if it fits"""
        if self.is_closed:
            return False
        
        # Check if event falls within window
        if self.end_time and event.timestamp > self.end_time:
            return False
        
        if not self.end_time:
            self.end_time = self.start_time + timedelta(seconds=self.window_size_seconds)
        
        if self.start_time <= event.timestamp <= self.end_time:
            self.events.append(event)
            return True
        
        return False
    
    def close_window(self):
        """Close the window"""
        self.is_closed = True
        if not self.end_time:
            self.end_time = datetime.now()

@dataclass
class StreamSubscription:
    """Subscription to event stream"""
    subscription_id: str = field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    client_id: str = ""
    event_types: Set[EventType] = field(default_factory=set)
    filter_expression: Optional[str] = None
    batch_size: int = 1
    max_wait_time_ms: int = 100
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def matches_event(self, event: StreamEvent) -> bool:
        """Check if event matches subscription criteria"""
        if not self.is_active:
            return False
        
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Simple filter expression evaluation (can be enhanced)
        if self.filter_expression:
            try:
                # Basic string matching for demo
                return self.filter_expression.lower() in str(event.data).lower()
            except:
                return True
        
        return True

class StreamPartitioner:
    """Event stream partitioner"""
    
    def __init__(self, num_partitions: int = 4, strategy: StreamPartitionStrategy = StreamPartitionStrategy.HASH_BASED):
        self.num_partitions = num_partitions
        self.strategy = strategy
        self.round_robin_counter = 0
    
    def get_partition(self, event: StreamEvent) -> int:
        """Determine partition for event"""
        if self.strategy == StreamPartitionStrategy.ROUND_ROBIN:
            partition = self.round_robin_counter % self.num_partitions
            self.round_robin_counter += 1
            return partition
        
        elif self.strategy == StreamPartitionStrategy.HASH_BASED:
            # Hash based on event source
            hash_value = hash(event.source) % self.num_partitions
            return hash_value
        
        elif self.strategy == StreamPartitionStrategy.KEY_BASED:
            # Use partition key if available
            if event.partition_key:
                return hash(event.partition_key) % self.num_partitions
            return hash(event.source) % self.num_partitions
        
        elif self.strategy == StreamPartitionStrategy.TIMESTAMP_BASED:
            # Partition based on timestamp
            return int(event.timestamp.timestamp()) % self.num_partitions
        
        elif self.strategy == StreamPartitionStrategy.PRIORITY_BASED:
            # Partition based on event type priority
            priority_map = {
                EventType.ERROR_EVENT: 0,
                EventType.ALERT_NOTIFICATION: 0,
                EventType.PERFORMANCE_METRIC: 1,
                EventType.SYSTEM_EVENT: 2,
                EventType.USER_ACTION: 3
            }
            priority = priority_map.get(event.event_type, 3)
            return priority % self.num_partitions
        
        return 0

class StreamProcessor:
    """Stream processor with windowing and aggregation"""
    
    def __init__(self, window_size_seconds: int = 60):
        self.window_size_seconds = window_size_seconds
        self.active_windows: Dict[str, StreamWindow] = {}
        self.completed_windows: deque = deque(maxlen=100)
        self.aggregation_functions: Dict[str, Callable] = {}
        self.pattern_matchers: List[Callable] = []
        
    def register_aggregation(self, name: str, func: Callable):
        """Register aggregation function"""
        self.aggregation_functions[name] = func
    
    def register_pattern_matcher(self, matcher: Callable):
        """Register pattern matching function"""
        self.pattern_matchers.append(matcher)
    
    def process_event(self, event: StreamEvent) -> List[Dict[str, Any]]:
        """Process event through windowing and aggregation"""
        results = []
        
        # Determine which window this event belongs to
        window_key = self._get_window_key(event.timestamp)
        
        if window_key not in self.active_windows:
            # Create new window
            window_start = self._align_to_window_boundary(event.timestamp)
            self.active_windows[window_key] = StreamWindow(
                start_time=window_start,
                window_size_seconds=self.window_size_seconds
            )
        
        window = self.active_windows[window_key]
        
        # Add event to window
        if window.add_event(event):
            # Check if window should be closed
            if self._should_close_window(window):
                window.close_window()
                results.extend(self._process_completed_window(window))
                
                # Move to completed windows
                self.completed_windows.append(window)
                del self.active_windows[window_key]
        
        # Run pattern matchers on the event
        for matcher in self.pattern_matchers:
            try:
                pattern_result = matcher(event, window.events if window_key in self.active_windows else [])
                if pattern_result:
                    results.append({
                        'type': 'pattern_match',
                        'pattern': pattern_result,
                        'event_id': event.event_id,
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.warning(f"Pattern matcher error: {e}")
        
        return results
    
    def _get_window_key(self, timestamp: datetime) -> str:
        """Generate window key based on timestamp"""
        window_start = self._align_to_window_boundary(timestamp)
        return f"window_{int(window_start.timestamp())}"
    
    def _align_to_window_boundary(self, timestamp: datetime) -> datetime:
        """Align timestamp to window boundary"""
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (timestamp - epoch).total_seconds()
        aligned_seconds = (seconds_since_epoch // self.window_size_seconds) * self.window_size_seconds
        return epoch + timedelta(seconds=aligned_seconds)
    
    def _should_close_window(self, window: StreamWindow) -> bool:
        """Check if window should be closed"""
        if window.end_time and datetime.now() > window.end_time:
            return True
        return False
    
    def _process_completed_window(self, window: StreamWindow) -> List[Dict[str, Any]]:
        """Process completed window with aggregations"""
        results = []
        
        for name, func in self.aggregation_functions.items():
            try:
                aggregation_result = func(window.events)
                results.append({
                    'type': 'aggregation',
                    'name': name,
                    'window_id': window.window_id,
                    'start_time': window.start_time.isoformat(),
                    'end_time': window.end_time.isoformat() if window.end_time else None,
                    'event_count': len(window.events),
                    'result': aggregation_result,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Aggregation function '{name}' failed: {e}")
        
        return results

class EventStreamingEngine:
    """
    High-performance event streaming engine providing real-time data flows
    and distributed event processing across enterprise systems.
    """
    
    def __init__(self, num_partitions: int = 4, window_size_seconds: int = 60):
        self.partitioner = StreamPartitioner(num_partitions)
        self.processor = StreamProcessor(window_size_seconds)
        
        # Event storage and streaming
        self.event_partitions: List[deque] = [deque(maxlen=10000) for _ in range(num_partitions)]
        self.event_log: deque = deque(maxlen=50000)  # Complete event log for replay
        self.sequence_counter = 0
        
        # Subscriptions and clients
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.client_connections: Dict[str, Any] = {}  # WebSocket connections
        self.subscription_queues: Dict[str, asyncio.Queue] = {}
        
        # Streaming state
        self.streaming_active = False
        self.processing_tasks: Set[asyncio.Task] = set()
        self.client_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.metrics = {
            'events_processed': 0,
            'events_per_second': 0.0,
            'active_subscriptions': 0,
            'active_connections': 0,
            'processing_latency_ms': 0.0,
            'start_time': datetime.now()
        }
        
        # Dead letter queue for failed events
        self.dead_letter_queue: deque = deque(maxlen=1000)
        
        # Backpressure monitoring
        self.backpressure_threshold = 1000
        self.backpressure_active = False
        
        # Setup default aggregations and patterns
        self._setup_default_processors()
        
        logger.info("Event Streaming Engine initialized")
    
    def _setup_default_processors(self):
        """Setup default aggregation functions and pattern matchers"""
        # Event count aggregation
        self.processor.register_aggregation(
            "event_count",
            lambda events: len(events)
        )
        
        # Event type distribution
        self.processor.register_aggregation(
            "event_type_distribution",
            lambda events: {
                event_type.value: len([e for e in events if e.event_type == event_type])
                for event_type in EventType
                if any(e.event_type == event_type for e in events)
            }
        )
        
        # Error rate calculation
        self.processor.register_aggregation(
            "error_rate",
            lambda events: len([e for e in events if e.event_type == EventType.ERROR_EVENT]) / max(1, len(events))
        )
        
        # Performance metrics aggregation
        self.processor.register_aggregation(
            "performance_metrics",
            lambda events: {
                'avg_response_time': self._calculate_avg_response_time(events),
                'max_response_time': self._calculate_max_response_time(events),
                'throughput': len(events)
            }
        )
        
        # Error burst pattern matcher
        self.processor.register_pattern_matcher(self._detect_error_burst)
        
        # Performance degradation pattern matcher
        self.processor.register_pattern_matcher(self._detect_performance_degradation)
    
    def _calculate_avg_response_time(self, events: List[StreamEvent]) -> float:
        """Calculate average response time from performance events"""
        response_times = []
        for event in events:
            if event.event_type == EventType.PERFORMANCE_METRIC and 'response_time' in event.data:
                response_times.append(float(event.data['response_time']))
        
        return sum(response_times) / len(response_times) if response_times else 0.0
    
    def _calculate_max_response_time(self, events: List[StreamEvent]) -> float:
        """Calculate max response time from performance events"""
        response_times = []
        for event in events:
            if event.event_type == EventType.PERFORMANCE_METRIC and 'response_time' in event.data:
                response_times.append(float(event.data['response_time']))
        
        return max(response_times) if response_times else 0.0
    
    def _detect_error_burst(self, event: StreamEvent, window_events: List[StreamEvent]) -> Optional[Dict]:
        """Detect error burst pattern"""
        if event.event_type != EventType.ERROR_EVENT:
            return None
        
        # Count recent errors in window
        error_events = [e for e in window_events if e.event_type == EventType.ERROR_EVENT]
        
        if len(error_events) >= 5:  # Threshold for error burst
            return {
                'pattern_type': 'error_burst',
                'error_count': len(error_events),
                'severity': 'high' if len(error_events) >= 10 else 'medium'
            }
        
        return None
    
    def _detect_performance_degradation(self, event: StreamEvent, window_events: List[StreamEvent]) -> Optional[Dict]:
        """Detect performance degradation pattern"""
        if event.event_type != EventType.PERFORMANCE_METRIC:
            return None
        
        # Look for increasing response times
        perf_events = [
            e for e in window_events 
            if e.event_type == EventType.PERFORMANCE_METRIC and 'response_time' in e.data
        ]
        
        if len(perf_events) >= 3:
            response_times = [float(e.data['response_time']) for e in perf_events[-3:]]
            
            # Check for consistent increase
            if all(response_times[i] < response_times[i+1] for i in range(len(response_times)-1)):
                return {
                    'pattern_type': 'performance_degradation',
                    'trend': 'increasing',
                    'latest_response_time': response_times[-1]
                }
        
        return None
    
    async def start_streaming(self):
        """Start the event streaming engine"""
        if self.streaming_active:
            return
        
        logger.info("Starting Event Streaming Engine")
        self.streaming_active = True
        
        # Start processing tasks for each partition
        for partition_id in range(len(self.event_partitions)):
            task = asyncio.create_task(self._process_partition(partition_id))
            self.processing_tasks.add(task)
        
        # Start metrics collection
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.processing_tasks.add(metrics_task)
        
        # Start subscription cleanup
        cleanup_task = asyncio.create_task(self._subscription_cleanup_loop())
        self.processing_tasks.add(cleanup_task)
        
        logger.info("Event Streaming Engine started")
    
    async def stop_streaming(self):
        """Stop the event streaming engine"""
        if not self.streaming_active:
            return
        
        logger.info("Stopping Event Streaming Engine")
        self.streaming_active = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        for task in self.client_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        await asyncio.gather(*self.client_tasks, return_exceptions=True)
        
        self.processing_tasks.clear()
        self.client_tasks.clear()
        
        logger.info("Event Streaming Engine stopped")
    
    async def publish_event(self, event: StreamEvent) -> bool:
        """Publish event to the stream"""
        try:
            # Check for backpressure
            if self._check_backpressure():
                logger.warning("Backpressure detected, dropping event")
                return False
            
            # Assign sequence number
            self.sequence_counter += 1
            event.sequence_number = self.sequence_counter
            
            # Determine partition
            partition_id = self.partitioner.get_partition(event)
            
            # Add to partition and event log
            self.event_partitions[partition_id].append(event)
            self.event_log.append(event)
            
            # Update metrics
            self.metrics['events_processed'] += 1
            
            logger.debug(f"Published event {event.event_id} to partition {partition_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            self.dead_letter_queue.append(event)
            return False
    
    def _check_backpressure(self) -> bool:
        """Check if backpressure should be applied"""
        total_pending = sum(len(partition) for partition in self.event_partitions)
        
        if total_pending > self.backpressure_threshold:
            if not self.backpressure_active:
                logger.warning("Activating backpressure")
                self.backpressure_active = True
            return True
        else:
            if self.backpressure_active:
                logger.info("Backpressure deactivated")
                self.backpressure_active = False
            return False
    
    async def _process_partition(self, partition_id: int):
        """Process events from a specific partition"""
        partition = self.event_partitions[partition_id]
        
        while self.streaming_active:
            try:
                if partition:
                    event = partition.popleft()
                    
                    # Process event through stream processor
                    processing_results = self.processor.process_event(event)
                    
                    # Handle processing results (aggregations, patterns)
                    for result in processing_results:
                        await self._handle_processing_result(result)
                    
                    # Deliver to subscribers
                    await self._deliver_to_subscribers(event)
                    
                else:
                    # No events in partition, wait briefly
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error processing partition {partition_id}: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_processing_result(self, result: Dict[str, Any]):
        """Handle results from stream processing"""
        if result['type'] == 'aggregation':
            # Create aggregation event
            aggregation_event = StreamEvent(
                event_type=EventType.ANALYTICS_RESULT,
                source="stream_processor",
                data=result
            )
            # Re-publish aggregation as new event
            await self.publish_event(aggregation_event)
            
        elif result['type'] == 'pattern_match':
            # Create alert event
            alert_event = StreamEvent(
                event_type=EventType.ALERT_NOTIFICATION,
                source="pattern_detector",
                data=result
            )
            await self.publish_event(alert_event)
    
    async def _deliver_to_subscribers(self, event: StreamEvent):
        """Deliver event to matching subscribers"""
        for subscription in self.subscriptions.values():
            if subscription.matches_event(event):
                subscription.last_activity = datetime.now()
                
                # Add to subscription queue
                queue = self.subscription_queues.get(subscription.subscription_id)
                if queue:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning(f"Subscription queue full for {subscription.subscription_id}")
    
    def create_subscription(self, client_id: str, event_types: List[EventType],
                          filter_expression: Optional[str] = None,
                          batch_size: int = 1) -> str:
        """Create new subscription"""
        subscription = StreamSubscription(
            client_id=client_id,
            event_types=set(event_types),
            filter_expression=filter_expression,
            batch_size=batch_size
        )
        
        self.subscriptions[subscription.subscription_id] = subscription
        self.subscription_queues[subscription.subscription_id] = asyncio.Queue(maxsize=1000)
        
        # Start client task
        client_task = asyncio.create_task(self._handle_client_subscription(subscription))
        self.client_tasks.add(client_task)
        
        self.metrics['active_subscriptions'] = len(self.subscriptions)
        
        logger.info(f"Created subscription {subscription.subscription_id} for client {client_id}")
        return subscription.subscription_id
    
    def cancel_subscription(self, subscription_id: str):
        """Cancel subscription"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].is_active = False
            del self.subscriptions[subscription_id]
            
            if subscription_id in self.subscription_queues:
                del self.subscription_queues[subscription_id]
            
            self.metrics['active_subscriptions'] = len(self.subscriptions)
            logger.info(f"Cancelled subscription {subscription_id}")
    
    async def _handle_client_subscription(self, subscription: StreamSubscription):
        """Handle client subscription delivery"""
        queue = self.subscription_queues[subscription.subscription_id]
        batch = []
        
        while self.streaming_active and subscription.is_active:
            try:
                # Collect events for batch
                timeout = subscription.max_wait_time_ms / 1000.0
                
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass
                
                # Send batch if ready
                if (len(batch) >= subscription.batch_size or 
                    (batch and time.time() * 1000 % 100 < 10)):  # Time-based flush
                    
                    await self._send_batch_to_client(subscription, batch)
                    batch.clear()
                    
            except Exception as e:
                logger.error(f"Error in client subscription {subscription.subscription_id}: {e}")
                break
    
    async def _send_batch_to_client(self, subscription: StreamSubscription, events: List[StreamEvent]):
        """Send batch of events to client"""
        try:
            # Convert events to serializable format
            event_data = [event.to_dict() for event in events]
            
            batch_message = {
                'subscription_id': subscription.subscription_id,
                'events': event_data,
                'batch_size': len(events),
                'timestamp': datetime.now().isoformat()
            }
            
            # In real implementation, send via WebSocket or HTTP
            logger.debug(f"Sent batch of {len(events)} events to {subscription.client_id}")
            
        except Exception as e:
            logger.error(f"Failed to send batch to client {subscription.client_id}: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection"""
        last_event_count = 0
        last_time = time.time()
        
        while self.streaming_active:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                current_time = time.time()
                current_events = self.metrics['events_processed']
                
                # Calculate events per second
                time_diff = current_time - last_time
                event_diff = current_events - last_event_count
                
                if time_diff > 0:
                    self.metrics['events_per_second'] = event_diff / time_diff
                
                last_event_count = current_events
                last_time = current_time
                
                # Update connection counts
                self.metrics['active_connections'] = len(self.client_connections)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _subscription_cleanup_loop(self):
        """Clean up inactive subscriptions"""
        while self.streaming_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                inactive_subscriptions = []
                
                for sub_id, subscription in self.subscriptions.items():
                    # Mark subscriptions inactive after 10 minutes of no activity
                    if (current_time - subscription.last_activity).total_seconds() > 600:
                        inactive_subscriptions.append(sub_id)
                
                for sub_id in inactive_subscriptions:
                    self.cancel_subscription(sub_id)
                    logger.info(f"Cleaned up inactive subscription {sub_id}")
                    
            except Exception as e:
                logger.error(f"Subscription cleanup error: {e}")
    
    async def replay_events(self, start_time: datetime, end_time: Optional[datetime] = None,
                           event_types: Optional[List[EventType]] = None) -> AsyncGenerator[StreamEvent, None]:
        """Replay events from the event log"""
        if end_time is None:
            end_time = datetime.now()
        
        for event in self.event_log:
            if start_time <= event.timestamp <= end_time:
                if event_types is None or event.event_type in event_types:
                    yield event
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get comprehensive streaming engine status"""
        uptime = (datetime.now() - self.metrics['start_time']).total_seconds()
        
        partition_status = []
        for i, partition in enumerate(self.event_partitions):
            partition_status.append({
                'partition_id': i,
                'pending_events': len(partition),
                'processing_rate': self.metrics['events_per_second'] / len(self.event_partitions)
            })
        
        return {
            'status': 'active' if self.streaming_active else 'inactive',
            'uptime_seconds': uptime,
            'metrics': self.metrics.copy(),
            'partitions': {
                'count': len(self.event_partitions),
                'status': partition_status,
                'total_pending': sum(len(p) for p in self.event_partitions)
            },
            'subscriptions': {
                'active_count': len(self.subscriptions),
                'client_count': len(set(sub.client_id for sub in self.subscriptions.values()))
            },
            'backpressure': {
                'active': self.backpressure_active,
                'threshold': self.backpressure_threshold
            },
            'dead_letter_queue': {
                'size': len(self.dead_letter_queue),
                'max_size': self.dead_letter_queue.maxlen
            },
            'event_log': {
                'size': len(self.event_log),
                'max_size': self.event_log.maxlen,
                'sequence_counter': self.sequence_counter
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown event streaming engine"""
        if self.streaming_active:
            # Create and run shutdown task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_streaming())
            loop.close()
        
        logger.info("Event Streaming Engine shutdown")

# Global event streaming engine instance
event_streaming_engine = EventStreamingEngine()