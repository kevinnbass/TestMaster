#!/usr/bin/env python3
"""
Real-time Dashboard Updates - Atomic Component
Handles real-time data updates to dashboard
Agent Z - STEELCLAD Frontend Atomization
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum


class UpdateType(Enum):
    """Types of dashboard updates"""
    METRICS = "metrics"
    STATUS = "status"
    ALERTS = "alerts"
    PERFORMANCE = "performance"
    AGENTS = "agents"
    COST = "cost"


@dataclass
class DashboardUpdate:
    """Dashboard update data structure"""
    update_type: UpdateType
    data: Dict[str, Any]
    timestamp: datetime
    priority: str = "normal"
    target_components: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        return {
            'type': self.update_type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'target_components': self.target_components or []
        }


class RealtimeDashboardUpdates:
    """
    Real-time dashboard data updates component
    Manages and broadcasts dashboard updates with <50ms latency
    """
    
    def __init__(self, max_history: int = 1000):
        self.update_history = deque(maxlen=max_history)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.update_cache: Dict[str, Any] = {}
        self.cache_ttl = 30  # seconds
        
        # Performance metrics
        self.metrics = {
            'updates_sent': 0,
            'updates_cached': 0,
            'cache_hits': 0,
            'avg_broadcast_time': 0.0,
            'subscribers_count': 0
        }
        
        # Update queues by priority
        self.high_priority_queue: List[DashboardUpdate] = []
        self.normal_priority_queue: List[DashboardUpdate] = []
        
        # Real-time data stores
        self.current_metrics: Dict[str, Any] = {}
        self.agent_status: Dict[str, Any] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        self.performance_data: Dict[str, float] = {}
    
    def broadcast_dashboard_update(self, data: Dict[str, Any], 
                                  update_type: UpdateType = UpdateType.METRICS,
                                  priority: str = "normal",
                                  target_components: List[str] = None):
        """
        Broadcast real-time update to dashboard
        Main interface for dashboard data updates
        """
        start_time = time.time()
        
        update = DashboardUpdate(
            update_type=update_type,
            data=data,
            timestamp=datetime.now(),
            priority=priority,
            target_components=target_components
        )
        
        # Add to appropriate queue
        if priority == "high":
            self.high_priority_queue.append(update)
        else:
            self.normal_priority_queue.append(update)
        
        # Update history
        self.update_history.append(update)
        
        # Notify subscribers
        self._notify_subscribers(update)
        
        # Update cache
        self._update_cache(update)
        
        # Track metrics
        broadcast_time = time.time() - start_time
        self.metrics['avg_broadcast_time'] = (
            (self.metrics['avg_broadcast_time'] * 0.9) + (broadcast_time * 0.1)
        )
        self.metrics['updates_sent'] += 1
        
        return update.to_dict()
    
    def subscribe_to_updates(self, subscriber_id: str, callback: Callable):
        """Subscribe to dashboard updates"""
        if subscriber_id not in self.subscribers:
            self.subscribers[subscriber_id] = []
        
        self.subscribers[subscriber_id].append(callback)
        self.metrics['subscribers_count'] = len(self.subscribers)
    
    def unsubscribe_from_updates(self, subscriber_id: str):
        """Unsubscribe from dashboard updates"""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            self.metrics['subscribers_count'] = len(self.subscribers)
    
    def _notify_subscribers(self, update: DashboardUpdate):
        """Notify all subscribers of new update"""
        for subscriber_callbacks in self.subscribers.values():
            for callback in subscriber_callbacks:
                try:
                    callback(update.to_dict())
                except Exception:
                    pass
    
    def _update_cache(self, update: DashboardUpdate):
        """Update cache with latest data"""
        cache_key = f"{update.update_type.value}:{update.timestamp.timestamp()}"
        self.update_cache[cache_key] = {
            'data': update.data,
            'timestamp': update.timestamp,
            'expires': time.time() + self.cache_ttl
        }
        self.metrics['updates_cached'] += 1
        
        # Clean expired cache entries
        self._clean_cache()
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.update_cache.items()
            if value['expires'] < current_time
        ]
        
        for key in expired_keys:
            del self.update_cache[key]
    
    def get_cached_update(self, update_type: UpdateType, 
                          max_age_seconds: int = 30) -> Optional[Dict[str, Any]]:
        """Get cached update if available and fresh"""
        current_time = time.time()
        min_timestamp = current_time - max_age_seconds
        
        # Search cache for matching update
        for key, value in self.update_cache.items():
            if (key.startswith(f"{update_type.value}:") and 
                value['timestamp'].timestamp() >= min_timestamp):
                self.metrics['cache_hits'] += 1
                return value['data']
        
        return None
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update dashboard metrics"""
        self.current_metrics.update(metrics)
        self.broadcast_dashboard_update(
            data=self.current_metrics,
            update_type=UpdateType.METRICS,
            priority="normal"
        )
    
    def update_agent_status(self, agent_id: str, status: Dict[str, Any]):
        """Update agent status on dashboard"""
        self.agent_status[agent_id] = {
            **status,
            'last_update': datetime.now().isoformat()
        }
        
        self.broadcast_dashboard_update(
            data={'agent_id': agent_id, 'status': self.agent_status[agent_id]},
            update_type=UpdateType.AGENTS,
            priority="normal"
        )
    
    def send_alert(self, alert: Dict[str, Any], priority: str = "high"):
        """Send alert to dashboard"""
        alert['timestamp'] = datetime.now().isoformat()
        self.active_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]
        
        self.broadcast_dashboard_update(
            data=alert,
            update_type=UpdateType.ALERTS,
            priority=priority
        )
    
    def update_performance(self, performance: Dict[str, float]):
        """Update performance metrics on dashboard"""
        self.performance_data.update(performance)
        
        # Calculate performance score
        latency_ok = performance.get('response_time_ms', 100) < 50
        throughput_ok = performance.get('throughput_rps', 0) > 10
        
        self.performance_data['health_score'] = (
            (50 if latency_ok else 0) + (50 if throughput_ok else 0)
        )
        
        self.broadcast_dashboard_update(
            data=self.performance_data,
            update_type=UpdateType.PERFORMANCE,
            priority="normal"
        )
    
    async def process_update_queues(self):
        """Process update queues asynchronously"""
        # Process high priority first
        while self.high_priority_queue:
            update = self.high_priority_queue.pop(0)
            await self._async_broadcast(update)
        
        # Then normal priority
        while self.normal_priority_queue:
            update = self.normal_priority_queue.pop(0)
            await self._async_broadcast(update)
    
    async def _async_broadcast(self, update: DashboardUpdate):
        """Asynchronously broadcast update"""
        # This would integrate with WebSocket streaming
        await asyncio.sleep(0.001)  # Simulate network delay
        return update.to_dict()
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current dashboard state"""
        return {
            'metrics': self.current_metrics,
            'agents': self.agent_status,
            'alerts': self.active_alerts[-10:],  # Last 10 alerts
            'performance': self.performance_data,
            'update_history_count': len(self.update_history),
            'last_update': (
                self.update_history[-1].timestamp.isoformat() 
                if self.update_history else None
            )
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get real-time update metrics"""
        return {
            **self.metrics,
            'high_priority_pending': len(self.high_priority_queue),
            'normal_priority_pending': len(self.normal_priority_queue),
            'cache_size': len(self.update_cache),
            'active_alerts': len(self.active_alerts),
            'latency_target_met': self.metrics['avg_broadcast_time'] * 1000 < 50
        }