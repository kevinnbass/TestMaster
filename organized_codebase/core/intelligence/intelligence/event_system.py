"""
Event-Driven Architecture System

Lightweight event system for real-time communication between intelligence components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Event data structure."""
    event_id: str
    event_type: str
    source: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "metadata": self.metadata
        }


@dataclass
class EventHandler:
    """Event handler configuration."""
    handler_id: str
    event_types: Set[str]
    handler_func: Callable
    is_async: bool = True
    priority: int = 0  # Higher priority handlers run first


class EventSystem:
    """
    Lightweight event-driven system for intelligence components.
    
    Features:
    - Asynchronous event processing
    - Priority-based event handling
    - Event filtering and routing
    - Real-time event streaming
    - Event history and replay
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.handlers: Dict[str, EventHandler] = {}
        self.event_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in EventPriority
        }
        self.event_history: List[Event] = []
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "handler_count": 0
        }
    
    async def start(self):
        """Start the event system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processor_task = asyncio.create_task(self._process_events())
        self.logger.info("Event system started")
    
    async def stop(self):
        """Stop the event system."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Event system stopped")
    
    def register_handler(self, 
                        handler_id: str,
                        event_types: Set[str],
                        handler_func: Callable,
                        priority: int = 0):
        """Register an event handler."""
        is_async = asyncio.iscoroutinefunction(handler_func)
        
        handler = EventHandler(
            handler_id=handler_id,
            event_types=event_types,
            handler_func=handler_func,
            is_async=is_async,
            priority=priority
        )
        
        self.handlers[handler_id] = handler
        self.metrics["handler_count"] = len(self.handlers)
        
        self.logger.info(f"Registered event handler: {handler_id}")
    
    def unregister_handler(self, handler_id: str):
        """Unregister an event handler."""
        if handler_id in self.handlers:
            del self.handlers[handler_id]
            self.metrics["handler_count"] = len(self.handlers)
            self.logger.info(f"Unregistered event handler: {handler_id}")
    
    async def emit(self, 
                   event_type: str,
                   source: str,
                   data: Any,
                   priority: EventPriority = EventPriority.NORMAL,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Emit an event."""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            data=data,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add to appropriate queue
        await self.event_queues[priority].put(event)
        
        # Store in history (limited)
        self.event_history.append(event)
        if len(self.event_history) > 10000:
            self.event_history = self.event_history[-5000:]
        
        return event.event_id
    
    async def _process_events(self):
        """Process events from queues."""
        while self.is_running:
            try:
                # Process events by priority
                for priority in [EventPriority.CRITICAL, EventPriority.HIGH, 
                               EventPriority.NORMAL, EventPriority.LOW]:
                    queue = self.event_queues[priority]
                    
                    try:
                        # Non-blocking get with timeout
                        event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        await self._handle_event(event)
                    except asyncio.TimeoutError:
                        continue
                
                # Brief pause if no events
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in event processing: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_event(self, event: Event):
        """Handle a single event."""
        try:
            # Find matching handlers
            matching_handlers = []
            for handler in self.handlers.values():
                if event.event_type in handler.event_types or "*" in handler.event_types:
                    matching_handlers.append(handler)
            
            # Sort by priority
            matching_handlers.sort(key=lambda h: h.priority, reverse=True)
            
            # Execute handlers
            for handler in matching_handlers:
                try:
                    if handler.is_async:
                        await handler.handler_func(event)
                    else:
                        handler.handler_func(event)
                except Exception as e:
                    self.logger.error(f"Handler {handler.handler_id} failed: {e}")
                    self.metrics["events_failed"] += 1
            
            self.metrics["events_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to handle event {event.event_id}: {e}")
            self.metrics["events_failed"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            "is_running": self.is_running,
            "metrics": self.metrics.copy(),
            "queue_sizes": {
                priority.name: queue.qsize() 
                for priority, queue in self.event_queues.items()
            },
            "total_handlers": len(self.handlers),
            "event_history_size": len(self.event_history)
        }


# Global event system instance
_global_event_system: Optional[EventSystem] = None


def get_event_system() -> EventSystem:
    """Get the global event system instance."""
    global _global_event_system
    if _global_event_system is None:
        _global_event_system = EventSystem()
    return _global_event_system


async def emit_event(event_type: str, 
                    source: str, 
                    data: Any,
                    priority: EventPriority = EventPriority.NORMAL) -> str:
    """Convenience function to emit an event."""
    event_system = get_event_system()
    return await event_system.emit(event_type, source, data, priority)