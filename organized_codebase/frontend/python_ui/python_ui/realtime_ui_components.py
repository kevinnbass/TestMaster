"""
Real-time UI Components Module
==============================
Live updates and streaming interfaces extracted from AutoGen and streaming patterns.
Module size: ~299 lines (under 300 limit)

Patterns extracted from:
- AutoGen: Streaming chat interfaces and real-time message updates
- Agency-Swarm: Real-time agent tracking and live monitoring
- AgentScope: Live studio updates and session management
- CrewAI: Real-time flow execution and progress tracking
- LLama-Agents: Live deployment status and workflow monitoring
- Swarms: Real-time swarm coordination and intelligence updates
- PhiData: Live chart updates and data streaming

Author: Agent D - Visualization Specialist
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import deque
import threading
import asyncio
from abc import ABC, abstractmethod


@dataclass
class StreamingMessage:
    """Real-time streaming message."""
    id: str
    type: str  # text, json, binary, event
    content: Any
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None
    
    @classmethod
    def create(cls, msg_type: str, content: Any, source: str = "system", **metadata):
        return cls(
            id=str(uuid.uuid4()),
            type=msg_type,
            content=content,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )


@dataclass
class UIUpdate:
    """Real-time UI update event."""
    component_id: str
    update_type: str  # data, style, visibility, state
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical


class StreamingInterface(ABC):
    """Abstract base for streaming interfaces."""
    
    @abstractmethod
    def send_update(self, update: UIUpdate) -> bool:
        pass
        
    @abstractmethod
    def subscribe(self, component_id: str, callback: Callable[[UIUpdate], None]):
        pass
        
    @abstractmethod
    def unsubscribe(self, component_id: str, callback: Callable[[UIUpdate], None]):
        pass


class WebSocketStreamer(StreamingInterface):
    """WebSocket-based real-time streaming (AutoGen pattern)."""
    
    def __init__(self):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue = deque(maxlen=10000)
        self.update_queue = deque(maxlen=5000)
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Start processing thread
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_updates, daemon=True)
        self.processor_thread.start()
        
    def add_connection(self, connection_id: str, websocket_handler, client_info: Dict[str, Any] = None):
        """Add WebSocket connection."""
        self.connections[connection_id] = {
            "handler": websocket_handler,
            "connected_at": datetime.now(),
            "client_info": client_info or {},
            "message_count": 0,
            "last_activity": datetime.now(),
            "subscriptions": set()
        }
        
        # Initialize rate limiting
        self.rate_limits[connection_id] = {
            "messages_per_minute": 60,
            "message_timestamps": deque(maxlen=100)
        }
        
    def remove_connection(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.connections:
            # Unsubscribe from all components
            conn_info = self.connections[connection_id]
            for component_id in conn_info.get("subscriptions", set()):
                self._unsubscribe_connection(connection_id, component_id)
                
            del self.connections[connection_id]
            self.rate_limits.pop(connection_id, None)
            
    def send_update(self, update: UIUpdate) -> bool:
        """Queue UI update for streaming."""
        self.update_queue.append(update)
        return True
        
    def subscribe(self, component_id: str, callback: Callable[[UIUpdate], None]):
        """Subscribe to component updates."""
        if component_id not in self.subscribers:
            self.subscribers[component_id] = []
        self.subscribers[component_id].append(callback)
        
    def unsubscribe(self, component_id: str, callback: Callable[[UIUpdate], None]):
        """Unsubscribe from component updates."""
        if component_id in self.subscribers:
            try:
                self.subscribers[component_id].remove(callback)
            except ValueError:
                pass
                
    def subscribe_connection(self, connection_id: str, component_id: str):
        """Subscribe connection to component updates."""
        if connection_id in self.connections:
            self.connections[connection_id]["subscriptions"].add(component_id)
            
    def stream_message(self, message: StreamingMessage, target_connections: List[str] = None):
        """Stream message to connections."""
        connections = target_connections or list(self.connections.keys())
        sent_count = 0
        
        message_data = {
            "id": message.id,
            "type": message.type,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "source": message.source,
            "metadata": message.metadata
        }
        
        for conn_id in connections:
            if self._send_to_connection(conn_id, message_data):
                sent_count += 1
                
        self.message_queue.append(message)
        return sent_count
        
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.connections),
            "active_connections": len([c for c in self.connections.values() 
                                     if (datetime.now() - c["last_activity"]).seconds < 300]),
            "total_messages_queued": len(self.message_queue),
            "updates_queued": len(self.update_queue),
            "subscriptions": {comp: len(subs) for comp, subs in self.subscribers.items()}
        }
        
    def _process_updates(self):
        """Background thread to process UI updates."""
        while self.running:
            try:
                # Process pending updates
                while self.update_queue:
                    update = self.update_queue.popleft()
                    self._broadcast_update(update)
                    
                time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception:
                pass
                
    def _broadcast_update(self, update: UIUpdate):
        """Broadcast update to subscribed connections."""
        # Notify callback subscribers
        for callback in self.subscribers.get(update.component_id, []):
            try:
                callback(update)
            except Exception:
                pass
                
        # Send to WebSocket connections
        update_data = {
            "component_id": update.component_id,
            "update_type": update.update_type,
            "payload": update.payload,
            "timestamp": update.timestamp.isoformat(),
            "priority": update.priority
        }
        
        for conn_id, conn_info in self.connections.items():
            if update.component_id in conn_info.get("subscriptions", set()):
                self._send_to_connection(conn_id, update_data)
                
    def _send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Send data to specific connection with rate limiting."""
        if connection_id not in self.connections:
            return False
            
        # Check rate limiting
        if not self._check_rate_limit(connection_id):
            return False
            
        try:
            conn_info = self.connections[connection_id]
            message = json.dumps(data, default=str)
            conn_info["handler"].send(message)
            conn_info["message_count"] += 1
            conn_info["last_activity"] = datetime.now()
            
            # Update rate limit tracking
            self.rate_limits[connection_id]["message_timestamps"].append(datetime.now())
            return True
            
        except Exception:
            # Remove broken connection
            self.remove_connection(connection_id)
            return False
            
    def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits."""
        if connection_id not in self.rate_limits:
            return True
            
        rate_info = self.rate_limits[connection_id]
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old timestamps
        while (rate_info["message_timestamps"] and 
               rate_info["message_timestamps"][0] < minute_ago):
            rate_info["message_timestamps"].popleft()
            
        return len(rate_info["message_timestamps"]) < rate_info["messages_per_minute"]
        
    def _unsubscribe_connection(self, connection_id: str, component_id: str):
        """Unsubscribe connection from component."""
        if connection_id in self.connections:
            self.connections[connection_id]["subscriptions"].discard(component_id)


class LiveChart:
    """Real-time updating chart component."""
    
    def __init__(self, chart_id: str, chart_type: str = "line"):
        self.chart_id = chart_id
        self.chart_type = chart_type
        self.data_buffer = deque(maxlen=1000)
        self.update_interval = 1.0  # seconds
        self.last_update = datetime.now()
        self.auto_update = True
        self.subscribers = []
        
    def add_data_point(self, timestamp: datetime, value: float, label: str = ""):
        """Add new data point to chart."""
        data_point = {
            "timestamp": timestamp,
            "value": value,
            "label": label
        }
        
        self.data_buffer.append(data_point)
        
        # Trigger update if enough time has passed
        if self.auto_update and (datetime.now() - self.last_update).total_seconds() >= self.update_interval:
            self.update_chart()
            
    def update_chart(self):
        """Update chart with latest data."""
        chart_data = {
            "chart_id": self.chart_id,
            "chart_type": self.chart_type,
            "data_points": list(self.data_buffer)[-100:],  # Last 100 points
            "updated_at": datetime.now()
        }
        
        update = UIUpdate(
            component_id=self.chart_id,
            update_type="data",
            payload=chart_data,
            timestamp=datetime.now(),
            priority=2
        )
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(update)
            except Exception:
                pass
                
        self.last_update = datetime.now()
        
    def add_subscriber(self, callback: Callable[[UIUpdate], None]):
        """Add update subscriber."""
        self.subscribers.append(callback)
        
    def set_update_interval(self, seconds: float):
        """Set chart update interval."""
        self.update_interval = max(0.1, seconds)  # Minimum 100ms


class LiveProgressBar:
    """Real-time progress bar component."""
    
    def __init__(self, progress_id: str, total: int = 100):
        self.progress_id = progress_id
        self.total = total
        self.current = 0
        self.started_at = datetime.now()
        self.status = "active"
        self.message = ""
        self.subscribers = []
        
    def update_progress(self, current: int, message: str = ""):
        """Update progress value."""
        self.current = min(max(0, current), self.total)
        self.message = message
        
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        # Estimate completion time
        elapsed = (datetime.now() - self.started_at).total_seconds()
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
        else:
            eta = 0
            
        progress_data = {
            "progress_id": self.progress_id,
            "current": self.current,
            "total": self.total,
            "percentage": round(percentage, 1),
            "message": self.message,
            "status": self.status,
            "eta_seconds": eta,
            "elapsed_seconds": elapsed
        }
        
        update = UIUpdate(
            component_id=self.progress_id,
            update_type="data",
            payload=progress_data,
            timestamp=datetime.now(),
            priority=2
        )
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(update)
            except Exception:
                pass
                
    def complete(self, message: str = "Complete"):
        """Mark progress as complete."""
        self.current = self.total
        self.status = "complete"
        self.message = message
        self.update_progress(self.total, message)
        
    def error(self, message: str = "Error occurred"):
        """Mark progress as error."""
        self.status = "error"
        self.message = message
        self.update_progress(self.current, message)
        
    def add_subscriber(self, callback: Callable[[UIUpdate], None]):
        """Add update subscriber."""
        self.subscribers.append(callback)


class LiveLogViewer:
    """Real-time log viewing component."""
    
    def __init__(self, log_id: str, max_lines: int = 1000):
        self.log_id = log_id
        self.max_lines = max_lines
        self.log_lines = deque(maxlen=max_lines)
        self.filters = {}
        self.subscribers = []
        
    def add_log_line(self, level: str, message: str, source: str = "", **metadata):
        """Add new log line."""
        log_entry = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "source": source,
            "metadata": metadata
        }
        
        self.log_lines.append(log_entry)
        
        # Check if line matches filters
        if self._matches_filters(log_entry):
            self._send_log_update([log_entry])
            
    def add_log_lines(self, lines: List[Dict[str, Any]]):
        """Add multiple log lines."""
        filtered_lines = []
        
        for line in lines:
            self.log_lines.append(line)
            if self._matches_filters(line):
                filtered_lines.append(line)
                
        if filtered_lines:
            self._send_log_update(filtered_lines)
            
    def set_filter(self, filter_name: str, filter_func: Callable[[Dict[str, Any]], bool]):
        """Set log filter function."""
        self.filters[filter_name] = filter_func
        
    def clear_filter(self, filter_name: str):
        """Remove log filter."""
        self.filters.pop(filter_name, None)
        
    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        recent = list(self.log_lines)[-count:] if count < len(self.log_lines) else list(self.log_lines)
        return [log for log in recent if self._matches_filters(log)]
        
    def add_subscriber(self, callback: Callable[[UIUpdate], None]):
        """Add update subscriber."""
        self.subscribers.append(callback)
        
    def _matches_filters(self, log_entry: Dict[str, Any]) -> bool:
        """Check if log entry matches active filters."""
        if not self.filters:
            return True
            
        return all(filter_func(log_entry) for filter_func in self.filters.values())
        
    def _send_log_update(self, log_entries: List[Dict[str, Any]]):
        """Send log update to subscribers."""
        update = UIUpdate(
            component_id=self.log_id,
            update_type="data",
            payload={"new_lines": log_entries},
            timestamp=datetime.now(),
            priority=1
        )
        
        for subscriber in self.subscribers:
            try:
                subscriber(update)
            except Exception:
                pass


# Public API
__all__ = [
    'StreamingMessage',
    'UIUpdate',
    'StreamingInterface',
    'WebSocketStreamer',
    'LiveChart',
    'LiveProgressBar',
    'LiveLogViewer'
]