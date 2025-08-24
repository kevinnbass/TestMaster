#!/usr/bin/env python3
"""
Dashboard Metrics Stream - Atomic Component  
Metrics streaming to dashboards with real-time updates
Agent Z - STEELCLAD Frontend Atomization
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum


class StreamType(Enum):
    """Types of metric streams"""
    CONTINUOUS = "continuous"
    INTERVAL = "interval"
    ON_DEMAND = "on_demand"
    BATCH = "batch"


class MetricCategory(Enum):
    """Categories of metrics"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class MetricStream:
    """Metric stream configuration"""
    stream_id: str
    stream_type: StreamType
    metric_category: MetricCategory
    interval_seconds: int = 5
    batch_size: int = 10
    filters: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'stream_id': self.stream_id,
            'type': self.stream_type.value,
            'category': self.metric_category.value,
            'interval': self.interval_seconds,
            'batch_size': self.batch_size,
            'filters': self.filters or {}
        }


class DashboardMetricsStream:
    """
    Dashboard metrics streaming component
    Streams real-time metrics to dashboard frontends
    """
    
    def __init__(self, max_subscribers: int = 100):
        self.max_subscribers = max_subscribers
        self.active_streams: Dict[str, MetricStream] = {}
        self.subscribers: Dict[str, Set[Callable]] = {}
        
        # Metric buffers
        self.metric_buffer = deque(maxlen=1000)
        self.batch_buffer: Dict[str, List[Dict[str, Any]]] = {}
        
        # Stream control
        self.streaming_active = False
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.stream_metrics = {
            'metrics_streamed': 0,
            'batches_sent': 0,
            'active_streams': 0,
            'total_subscribers': 0,
            'avg_stream_latency': 0.0,
            'errors': 0
        }
        
        # Metric cache for on-demand streams
        self.metric_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 30  # seconds
    
    def create_stream(self, stream_id: str, stream_type: StreamType,
                     category: MetricCategory, **kwargs) -> bool:
        """Create a new metric stream"""
        if stream_id in self.active_streams:
            return False
        
        stream = MetricStream(
            stream_id=stream_id,
            stream_type=stream_type,
            metric_category=category,
            interval_seconds=kwargs.get('interval', 5),
            batch_size=kwargs.get('batch_size', 10),
            filters=kwargs.get('filters', {})
        )
        
        self.active_streams[stream_id] = stream
        self.subscribers[stream_id] = set()
        self.stream_metrics['active_streams'] = len(self.active_streams)
        
        # Start stream if continuous or interval
        if stream_type in [StreamType.CONTINUOUS, StreamType.INTERVAL]:
            asyncio.create_task(self._start_stream(stream_id))
        
        return True
    
    def subscribe_to_stream(self, stream_id: str, callback: Callable) -> bool:
        """Subscribe to a metric stream"""
        if stream_id not in self.active_streams:
            return False
        
        if len(self.subscribers[stream_id]) >= self.max_subscribers:
            return False
        
        self.subscribers[stream_id].add(callback)
        self._update_subscriber_count()
        
        return True
    
    def unsubscribe_from_stream(self, stream_id: str, callback: Callable) -> bool:
        """Unsubscribe from a metric stream"""
        if stream_id not in self.subscribers:
            return False
        
        self.subscribers[stream_id].discard(callback)
        self._update_subscriber_count()
        
        return True
    
    async def stream_metrics_to_dashboard(self, metrics: Dict[str, Any],
                                         category: MetricCategory = MetricCategory.SYSTEM):
        """
        Stream metrics to dashboard subscribers
        Main interface for metric streaming
        """
        start_time = time.time()
        
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().isoformat()
        
        # Add to buffer
        self.metric_buffer.append(metrics)
        
        # Stream to relevant subscribers
        for stream_id, stream in self.active_streams.items():
            if stream.metric_category == category:
                await self._send_to_stream(stream_id, metrics)
        
        # Update performance metrics
        stream_latency = time.time() - start_time
        self.stream_metrics['metrics_streamed'] += 1
        self.stream_metrics['avg_stream_latency'] = (
            (self.stream_metrics['avg_stream_latency'] * 0.9) + (stream_latency * 0.1)
        )
    
    async def _start_stream(self, stream_id: str):
        """Start a continuous or interval stream"""
        stream = self.active_streams[stream_id]
        
        if stream.stream_type == StreamType.CONTINUOUS:
            await self._continuous_stream(stream_id)
        elif stream.stream_type == StreamType.INTERVAL:
            await self._interval_stream(stream_id)
    
    async def _continuous_stream(self, stream_id: str):
        """Handle continuous streaming"""
        stream = self.active_streams[stream_id]
        
        while stream_id in self.active_streams:
            try:
                # Get latest metrics from buffer
                if self.metric_buffer:
                    latest_metrics = list(self.metric_buffer)[-10:]  # Last 10 metrics
                    
                    for metrics in latest_metrics:
                        if self._apply_filters(metrics, stream.filters):
                            await self._broadcast_to_subscribers(stream_id, metrics)
                
                await asyncio.sleep(0.1)  # Small delay for continuous streaming
                
            except Exception:
                self.stream_metrics['errors'] += 1
                await asyncio.sleep(1)
    
    async def _interval_stream(self, stream_id: str):
        """Handle interval-based streaming"""
        stream = self.active_streams[stream_id]
        
        while stream_id in self.active_streams:
            try:
                # Collect metrics for interval
                interval_metrics = self._collect_interval_metrics(stream)
                
                if interval_metrics:
                    await self._broadcast_to_subscribers(stream_id, interval_metrics)
                
                await asyncio.sleep(stream.interval_seconds)
                
            except Exception:
                self.stream_metrics['errors'] += 1
                await asyncio.sleep(stream.interval_seconds)
    
    async def _send_to_stream(self, stream_id: str, metrics: Dict[str, Any]):
        """Send metrics to a specific stream"""
        stream = self.active_streams.get(stream_id)
        if not stream:
            return
        
        if stream.stream_type == StreamType.BATCH:
            # Add to batch buffer
            if stream_id not in self.batch_buffer:
                self.batch_buffer[stream_id] = []
            
            self.batch_buffer[stream_id].append(metrics)
            
            # Send batch if full
            if len(self.batch_buffer[stream_id]) >= stream.batch_size:
                await self._send_batch(stream_id)
                
        elif stream.stream_type == StreamType.ON_DEMAND:
            # Cache for on-demand retrieval
            self._cache_metrics(stream_id, metrics)
            
        else:
            # Direct streaming (handled by stream tasks)
            pass
    
    async def _send_batch(self, stream_id: str):
        """Send batched metrics"""
        if stream_id not in self.batch_buffer:
            return
        
        batch = self.batch_buffer[stream_id]
        if not batch:
            return
        
        batch_data = {
            'batch': True,
            'metrics': batch,
            'count': len(batch),
            'timestamp': datetime.now().isoformat()
        }
        
        await self._broadcast_to_subscribers(stream_id, batch_data)
        
        # Clear batch buffer
        self.batch_buffer[stream_id] = []
        self.stream_metrics['batches_sent'] += 1
    
    async def _broadcast_to_subscribers(self, stream_id: str, data: Dict[str, Any]):
        """Broadcast data to stream subscribers"""
        subscribers = self.subscribers.get(stream_id, set())
        
        for callback in list(subscribers):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception:
                # Remove problematic subscriber
                self.subscribers[stream_id].discard(callback)
                self._update_subscriber_count()
    
    def _collect_interval_metrics(self, stream: MetricStream) -> Dict[str, Any]:
        """Collect metrics for interval streaming"""
        # Get recent metrics from buffer
        recent_metrics = list(self.metric_buffer)[-stream.batch_size:]
        
        if not recent_metrics:
            return {}
        
        # Apply filters
        filtered_metrics = [
            m for m in recent_metrics
            if self._apply_filters(m, stream.filters)
        ]
        
        if not filtered_metrics:
            return {}
        
        return {
            'interval': stream.interval_seconds,
            'metrics': filtered_metrics,
            'count': len(filtered_metrics),
            'timestamp': datetime.now().isoformat()
        }
    
    def _apply_filters(self, metrics: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to metrics"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key not in metrics:
                return False
            
            if isinstance(value, dict):
                # Range filter
                if 'min' in value and metrics[key] < value['min']:
                    return False
                if 'max' in value and metrics[key] > value['max']:
                    return False
            else:
                # Exact match
                if metrics[key] != value:
                    return False
        
        return True
    
    def _cache_metrics(self, stream_id: str, metrics: Dict[str, Any]):
        """Cache metrics for on-demand retrieval"""
        self.metric_cache[stream_id] = metrics
        self.cache_timestamps[stream_id] = time.time()
    
    def get_on_demand_metrics(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get on-demand metrics from cache"""
        if stream_id not in self.metric_cache:
            return None
        
        # Check if cache is expired
        if time.time() - self.cache_timestamps.get(stream_id, 0) > self.cache_ttl:
            del self.metric_cache[stream_id]
            del self.cache_timestamps[stream_id]
            return None
        
        return self.metric_cache[stream_id]
    
    async def flush_stream(self, stream_id: str):
        """Flush pending data for a stream"""
        stream = self.active_streams.get(stream_id)
        if not stream:
            return
        
        if stream.stream_type == StreamType.BATCH and stream_id in self.batch_buffer:
            await self._send_batch(stream_id)
    
    def stop_stream(self, stream_id: str) -> bool:
        """Stop a metric stream"""
        if stream_id not in self.active_streams:
            return False
        
        # Cancel stream task if exists
        if stream_id in self.stream_tasks:
            self.stream_tasks[stream_id].cancel()
            del self.stream_tasks[stream_id]
        
        # Clean up
        del self.active_streams[stream_id]
        del self.subscribers[stream_id]
        
        if stream_id in self.batch_buffer:
            del self.batch_buffer[stream_id]
        
        if stream_id in self.metric_cache:
            del self.metric_cache[stream_id]
            del self.cache_timestamps[stream_id]
        
        self.stream_metrics['active_streams'] = len(self.active_streams)
        
        return True
    
    def _update_subscriber_count(self):
        """Update total subscriber count"""
        total = sum(len(subs) for subs in self.subscribers.values())
        self.stream_metrics['total_subscribers'] = total
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a stream"""
        stream = self.active_streams.get(stream_id)
        if not stream:
            return None
        
        return {
            **stream.to_dict(),
            'subscribers': len(self.subscribers.get(stream_id, set())),
            'cached_metrics': stream_id in self.metric_cache,
            'batch_pending': len(self.batch_buffer.get(stream_id, []))
        }
    
    def get_all_streams(self) -> List[Dict[str, Any]]:
        """Get information about all streams"""
        return [
            self.get_stream_info(stream_id)
            for stream_id in self.active_streams.keys()
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            **self.stream_metrics,
            'buffer_size': len(self.metric_buffer),
            'cache_size': len(self.metric_cache),
            'max_subscribers': self.max_subscribers,
            'latency_target_met': self.stream_metrics['avg_stream_latency'] * 1000 < 50
        }