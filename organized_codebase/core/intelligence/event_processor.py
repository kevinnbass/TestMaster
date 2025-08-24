"""
TestMaster Event Processing Component
=====================================

Extracted from consolidated integration hub for better modularization.
Handles event-driven integration with intelligent routing and processing.

Original location: core/intelligence/integration/__init__.py (lines ~1000-1300)
"""

from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import logging
import uuid

from ..base import IntegrationEvent


class EventProcessor:
    """
    Manages event-driven integration with intelligent routing and processing.
    
    Features:
    - Event publishing and subscription
    - Intelligent routing based on rules
    - Real-time and batch processing
    - Event correlation tracking
    - Callback management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("event_processor")
        
        # Event storage
        self._event_queue: List[IntegrationEvent] = []
        self._processed_events: Dict[str, IntegrationEvent] = {}
        
        # Event processing
        self._event_processors: Dict[str, Dict[str, Any]] = {}
        self._routing_rules: Dict[str, List[Callable]] = {}
        self._correlation_tracker: Dict[str, List[str]] = {}
        
        # Real-time systems
        self._real_time_systems: Set[str] = set()
        
        # Threading
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.get('event_workers', 4)
        )
        self._processing_lock = threading.Lock()
        self._batch_processor_thread = None
        self._stop_processing = threading.Event()
        
        # Statistics
        self._event_statistics = {
            'total_published': 0,
            'total_processed': 0,
            'total_failed': 0,
            'average_processing_time': 0.0
        }
        
        # Start batch processor if configured
        if self.config.get('enable_batch_processing', True):
            self._start_batch_processor()
    
    def publish_event(self, 
                     event: IntegrationEvent,
                     routing_rules: Optional[List[str]] = None) -> bool:
        """
        Publish event to integration event stream with intelligent routing.
        
        Args:
            event: Event to publish
            routing_rules: Optional routing rules for the event
            
        Returns:
            True if event published successfully
        """
        try:
            # Set routing rules
            if routing_rules:
                event.routing_rules = routing_rules
            
            # Add to event queue
            with self._processing_lock:
                self._event_queue.append(event)
                self._event_statistics['total_published'] += 1
            
            # Track correlation if present
            if event.correlation_id:
                if event.correlation_id not in self._correlation_tracker:
                    self._correlation_tracker[event.correlation_id] = []
                self._correlation_tracker[event.correlation_id].append(event.event_id)
            
            # Process event immediately for real-time systems
            if event.source_system in self._real_time_systems:
                self._executor.submit(self._process_event_immediately, event)
            else:
                # Event will be processed by batch processor
                self.logger.debug(f"Event {event.event_id} queued for batch processing")
            
            self.logger.debug(f"Event published: {event.event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False
    
    def subscribe_to_events(self, 
                          event_types: List[str],
                          callback: Callable[[IntegrationEvent], None],
                          filter_rules: Optional[Dict[str, Any]] = None) -> str:
        """
        Subscribe to integration events with filtering and callback processing.
        
        Args:
            event_types: List of event types to subscribe to
            callback: Callback function for event processing
            filter_rules: Optional filtering rules
            
        Returns:
            Subscription ID
        """
        try:
            subscription_id = str(uuid.uuid4())
            
            # Register event processor
            for event_type in event_types:
                if event_type not in self._event_processors:
                    self._event_processors[event_type] = {}
                
                processor_config = {
                    'callback': callback,
                    'filter_rules': filter_rules or {},
                    'subscription_id': subscription_id,
                    'created_at': datetime.now()
                }
                
                self._event_processors[event_type][subscription_id] = processor_config
            
            self.logger.info(f"Event subscription created: {subscription_id} for {event_types}")
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"Failed to create event subscription: {e}")
            return ""
    
    def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID of subscription to remove
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            removed = False
            for event_type in list(self._event_processors.keys()):
                if subscription_id in self._event_processors[event_type]:
                    del self._event_processors[event_type][subscription_id]
                    removed = True
            
            if removed:
                self.logger.info(f"Event subscription removed: {subscription_id}")
            
            return removed
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe: {e}")
            return False
    
    def register_real_time_system(self, system_name: str):
        """Register a system for real-time event processing."""
        self._real_time_systems.add(system_name)
        self.logger.info(f"System {system_name} registered for real-time processing")
    
    def get_event_by_id(self, event_id: str) -> Optional[IntegrationEvent]:
        """Get event by ID."""
        # Check processed events
        if event_id in self._processed_events:
            return self._processed_events[event_id]
        
        # Check queue
        for event in self._event_queue:
            if event.event_id == event_id:
                return event
        
        return None
    
    def get_correlated_events(self, correlation_id: str) -> List[IntegrationEvent]:
        """Get all events with the same correlation ID."""
        correlated_events = []
        
        event_ids = self._correlation_tracker.get(correlation_id, [])
        for event_id in event_ids:
            event = self.get_event_by_id(event_id)
            if event:
                correlated_events.append(event)
        
        return correlated_events
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get comprehensive event processing statistics."""
        stats = self._event_statistics.copy()
        
        # Add queue information
        stats['queue_size'] = len(self._event_queue)
        stats['processed_events'] = len(self._processed_events)
        stats['active_subscriptions'] = sum(
            len(processors) for processors in self._event_processors.values()
        )
        stats['real_time_systems'] = list(self._real_time_systems)
        
        # Calculate success rate
        total_completed = stats['total_processed'] + stats['total_failed']
        if total_completed > 0:
            stats['success_rate'] = stats['total_processed'] / total_completed * 100
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def process_pending_events(self, max_events: int = 100) -> int:
        """
        Process pending events from the queue.
        
        Args:
            max_events: Maximum number of events to process
            
        Returns:
            Number of events processed
        """
        processed_count = 0
        
        with self._processing_lock:
            events_to_process = self._event_queue[:max_events]
            self._event_queue = self._event_queue[max_events:]
        
        for event in events_to_process:
            if self._process_event(event):
                processed_count += 1
        
        return processed_count
    
    # === Private Methods ===
    
    def _process_event_immediately(self, event: IntegrationEvent) -> bool:
        """Process event immediately for real-time systems."""
        try:
            start_time = time.time()
            
            # Update event status
            event.status = "processing"
            
            # Apply routing rules
            target_processors = self._apply_routing_rules(event)
            
            # Process with each applicable processor
            success_count = 0
            for processor_config in target_processors:
                try:
                    callback = processor_config['callback']
                    callback(event)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Event processor failed for {event.event_id}: {e}")
                    event.error_message = str(e)
            
            # Update event status
            if success_count > 0:
                event.status = "completed"
                event.processing_time = time.time() - start_time
                self._event_statistics['total_processed'] += 1
            else:
                event.status = "failed"
                self._event_statistics['total_failed'] += 1
            
            # Store processed event
            self._processed_events[event.event_id] = event
            
            # Update average processing time
            self._update_processing_time_average(event.processing_time)
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to process event {event.event_id}: {e}")
            event.status = "failed"
            event.error_message = str(e)
            self._event_statistics['total_failed'] += 1
            return False
    
    def _process_event(self, event: IntegrationEvent) -> bool:
        """Process a single event."""
        return self._process_event_immediately(event)
    
    def _apply_routing_rules(self, event: IntegrationEvent) -> List[Dict[str, Any]]:
        """Apply routing rules to determine event processors."""
        applicable_processors = []
        
        # Check event type processors
        event_processors = self._event_processors.get(event.event_type, {})
        
        for processor_config in event_processors.values():
            filter_rules = processor_config.get('filter_rules', {})
            
            # Apply filters
            if self._event_matches_filters(event, filter_rules):
                applicable_processors.append(processor_config)
        
        # Apply custom routing rules
        if event.routing_rules:
            for rule in event.routing_rules:
                if rule in self._routing_rules:
                    for router in self._routing_rules[rule]:
                        try:
                            if router(event):
                                # Find processor for this route
                                # (simplified - in real implementation would be more sophisticated)
                                pass
                        except Exception as e:
                            self.logger.error(f"Routing rule failed: {e}")
        
        return applicable_processors
    
    def _event_matches_filters(self, event: IntegrationEvent, filter_rules: Dict[str, Any]) -> bool:
        """Check if event matches filter rules."""
        if not filter_rules:
            return True
        
        for field, expected_value in filter_rules.items():
            if hasattr(event, field):
                actual_value = getattr(event, field)
                
                # Support different filter operations
                if isinstance(expected_value, dict):
                    # Complex filter with operators
                    if 'equals' in expected_value:
                        if actual_value != expected_value['equals']:
                            return False
                    if 'contains' in expected_value:
                        if expected_value['contains'] not in str(actual_value):
                            return False
                    if 'in' in expected_value:
                        if actual_value not in expected_value['in']:
                            return False
                else:
                    # Simple equality check
                    if actual_value != expected_value:
                        return False
        
        return True
    
    def _update_processing_time_average(self, processing_time: float):
        """Update average processing time statistic."""
        current_avg = self._event_statistics['average_processing_time']
        total_processed = self._event_statistics['total_processed']
        
        if total_processed == 1:
            self._event_statistics['average_processing_time'] = processing_time
        else:
            # Calculate new average
            new_avg = ((current_avg * (total_processed - 1)) + processing_time) / total_processed
            self._event_statistics['average_processing_time'] = new_avg
    
    def _start_batch_processor(self):
        """Start background batch processor thread."""
        def batch_processor_loop():
            while not self._stop_processing.is_set():
                try:
                    # Process pending events
                    if self._event_queue:
                        self.process_pending_events(max_events=50)
                    
                    # Sleep for batch interval
                    time.sleep(self.config.get('batch_interval', 5.0))
                    
                except Exception as e:
                    self.logger.error(f"Batch processor error: {e}")
        
        self._batch_processor_thread = threading.Thread(
            target=batch_processor_loop,
            daemon=True
        )
        self._batch_processor_thread.start()
        self.logger.info("Batch event processor started")
    
    def add_routing_rule(self, rule_name: str, rule_function: Callable[[IntegrationEvent], bool]):
        """Add a custom routing rule."""
        if rule_name not in self._routing_rules:
            self._routing_rules[rule_name] = []
        self._routing_rules[rule_name].append(rule_function)
    
    def clear_processed_events(self, older_than_hours: int = 24):
        """Clear old processed events to free memory."""
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        
        events_to_remove = []
        for event_id, event in self._processed_events.items():
            if event.timestamp.timestamp() < cutoff_time:
                events_to_remove.append(event_id)
        
        for event_id in events_to_remove:
            del self._processed_events[event_id]
        
        self.logger.info(f"Cleared {len(events_to_remove)} old processed events")
    
    def shutdown(self):
        """Gracefully shutdown the event processor."""
        self.logger.info("Shutting down event processor...")
        
        # Stop batch processor
        if self._batch_processor_thread:
            self._stop_processing.set()
            self._batch_processor_thread.join(timeout=5)
        
        # Process remaining events
        remaining = self.process_pending_events(max_events=1000)
        if remaining > 0:
            self.logger.info(f"Processed {remaining} remaining events during shutdown")
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Clear data structures
        self._event_queue.clear()
        self._processed_events.clear()
        self._event_processors.clear()
        
        self.logger.info("Event processor shutdown complete")
    
    def get_all_subscribers(self) -> List[Dict[str, Any]]:
        """Get all event subscribers."""
        subscribers = []
        for event_type, handlers in self._event_subscriptions.items():
            for handler in handlers:
                subscribers.append({
                    'event_type': event_type,
                    'handler': handler.__name__ if hasattr(handler, '__name__') else str(handler),
                    'priority': getattr(handler, 'priority', 0)
                })
        return subscribers
    
    def clear_all_subscriptions(self):
        """Clear all event subscriptions."""
        self._event_subscriptions.clear()
        self.logger.info("All event subscriptions cleared")


# Public API exports
__all__ = ['EventProcessor']