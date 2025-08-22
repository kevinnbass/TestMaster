"""
Analytics Event Queue with Guaranteed Delivery
==============================================

Provides a persistent event queue with guaranteed delivery for analytics data.
Ensures no analytics are lost even during system failures.

Author: TestMaster Team
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import queue

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class AnalyticsEvent:
    """Represents an analytics event."""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    priority: EventPriority
    status: EventStatus
    created_at: datetime
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class AnalyticsEventQueue:
    """
    Persistent event queue with guaranteed delivery.
    """
    
    def __init__(self, 
                 db_path: str = "analytics_events.db",
                 max_retries: int = 5,
                 batch_size: int = 100,
                 delivery_timeout: float = 30.0):
        """
        Initialize the event queue.
        
        Args:
            db_path: Path to SQLite database for persistence
            max_retries: Maximum retry attempts per event
            batch_size: Number of events to process in batch
            delivery_timeout: Timeout for delivery attempts
        """
        self.db_path = db_path
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.delivery_timeout = delivery_timeout
        
        # In-memory queues by priority
        self.priority_queues = {
            EventPriority.CRITICAL: deque(),
            EventPriority.HIGH: deque(),
            EventPriority.NORMAL: deque(),
            EventPriority.LOW: deque()
        }
        
        # Processing state
        self.processing_events = {}
        self.failed_events = deque(maxlen=1000)
        
        # Delivery handlers
        self.delivery_handlers = {}
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'delivered_events': 0,
            'failed_events': 0,
            'retried_events': 0,
            'current_queue_size': 0,
            'average_delivery_time': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.processing_active = False
        self.processing_threads = []
        
        # Initialize database
        self._init_database()
        
        # Load pending events from database
        self._load_pending_events()
        
        logger.info("Analytics Event Queue initialized")
    
    def _init_database(self):
        """Initialize SQLite database for event persistence."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    data BLOB NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    last_attempt TEXT,
                    delivered_at TEXT,
                    error_message TEXT,
                    metadata BLOB
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_status 
                ON analytics_events(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_priority 
                ON analytics_events(priority)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_created 
                ON analytics_events(created_at)
            """)
            
            conn.commit()
    
    def _load_pending_events(self):
        """Load pending events from database on startup."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load undelivered events
                cursor.execute("""
                    SELECT event_id, event_type, data, priority, status,
                           created_at, attempts, last_attempt, error_message, metadata
                    FROM analytics_events
                    WHERE status IN (?, ?, ?)
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 10000
                """, (EventStatus.PENDING.value, EventStatus.PROCESSING.value, 
                     EventStatus.RETRYING.value))
                
                rows = cursor.fetchall()
                loaded_count = 0
                
                for row in rows:
                    event = AnalyticsEvent(
                        event_id=row[0],
                        event_type=row[1],
                        data=SafePickleHandler.safe_load(row[2]),
                        priority=EventPriority(row[3]),
                        status=EventStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        attempts=row[6],
                        last_attempt=datetime.fromisoformat(row[7]) if row[7] else None,
                        error_message=row[8],
                        metadata=SafePickleHandler.safe_load(row[9]) if row[9] else {}
                    )
                    
                    # Reset processing events to pending
                    if event.status == EventStatus.PROCESSING:
                        event.status = EventStatus.PENDING
                    
                    # Add to appropriate queue
                    self.priority_queues[event.priority].append(event)
                    loaded_count += 1
                
                logger.info(f"Loaded {loaded_count} pending events from database")
                
        except Exception as e:
            logger.error(f"Failed to load pending events: {e}")
    
    def enqueue(self, 
                event_type: str,
                data: Dict[str, Any],
                priority: EventPriority = EventPriority.NORMAL,
                metadata: Dict[str, Any] = None) -> str:
        """
        Add an event to the queue.
        
        Args:
            event_type: Type of analytics event
            data: Event data
            priority: Event priority
            metadata: Optional metadata
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AnalyticsEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            priority=priority,
            status=EventStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        with self.lock:
            # Add to memory queue
            self.priority_queues[priority].append(event)
            
            # Persist to database
            self._persist_event(event)
            
            # Update statistics
            self.stats['total_events'] += 1
            self.stats['current_queue_size'] = self._get_total_queue_size()
        
        logger.debug(f"Enqueued event {event_id} with priority {priority.name}")
        return event_id
    
    def _persist_event(self, event: AnalyticsEvent):
        """Persist event to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO analytics_events
                    (event_id, event_type, data, priority, status, created_at,
                     attempts, last_attempt, delivered_at, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type,
                    pickle.dumps(event.data),
                    event.priority.value,
                    event.status.value,
                    event.created_at.isoformat(),
                    event.attempts,
                    event.last_attempt.isoformat() if event.last_attempt else None,
                    event.delivered_at.isoformat() if event.delivered_at else None,
                    event.error_message,
                    pickle.dumps(event.metadata) if event.metadata else None
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to persist event {event.event_id}: {e}")
    
    def register_handler(self, event_type: str, handler: Callable):
        """
        Register a delivery handler for an event type.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        self.delivery_handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    def start_processing(self, num_workers: int = 3):
        """Start event processing workers."""
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._processing_worker,
                name=f"EventWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.processing_threads.append(worker)
        
        # Start retry worker
        retry_worker = threading.Thread(
            target=self._retry_worker,
            name="RetryWorker",
            daemon=True
        )
        retry_worker.start()
        self.processing_threads.append(retry_worker)
        
        # Start cleanup worker
        cleanup_worker = threading.Thread(
            target=self._cleanup_worker,
            name="CleanupWorker",
            daemon=True
        )
        cleanup_worker.start()
        self.processing_threads.append(cleanup_worker)
        
        logger.info(f"Started event processing with {num_workers} workers")
    
    def stop_processing(self):
        """Stop event processing."""
        self.processing_active = False
        
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.processing_threads.clear()
        logger.info("Event processing stopped")
    
    def _processing_worker(self):
        """Main processing worker thread."""
        while self.processing_active:
            try:
                # Get next event from highest priority queue
                event = self._get_next_event()
                
                if event:
                    self._process_event(event)
                else:
                    time.sleep(0.1)  # No events, brief pause
                    
            except Exception as e:
                logger.error(f"Processing worker error: {e}")
    
    def _get_next_event(self) -> Optional[AnalyticsEvent]:
        """Get next event from priority queues."""
        with self.lock:
            # Check queues in priority order
            for priority in [EventPriority.CRITICAL, EventPriority.HIGH,
                           EventPriority.NORMAL, EventPriority.LOW]:
                queue = self.priority_queues[priority]
                if queue:
                    event = queue.popleft()
                    event.status = EventStatus.PROCESSING
                    self.processing_events[event.event_id] = event
                    self._persist_event(event)
                    return event
        
        return None
    
    def _process_event(self, event: AnalyticsEvent):
        """Process a single event."""
        start_time = time.time()
        
        try:
            # Update attempt info
            event.attempts += 1
            event.last_attempt = datetime.now()
            
            # Get handler for event type
            handler = self.delivery_handlers.get(event.event_type)
            
            if not handler:
                # No handler, use default delivery
                handler = self._default_delivery_handler
            
            # Attempt delivery with timeout
            success = self._deliver_with_timeout(handler, event)
            
            if success:
                # Mark as delivered
                event.status = EventStatus.DELIVERED
                event.delivered_at = datetime.now()
                
                # Update statistics
                delivery_time = time.time() - start_time
                self._update_delivery_stats(delivery_time)
                
                logger.debug(f"Successfully delivered event {event.event_id}")
            else:
                # Delivery failed
                self._handle_delivery_failure(event)
                
        except Exception as e:
            event.error_message = str(e)
            self._handle_delivery_failure(event)
            
        finally:
            # Remove from processing
            with self.lock:
                self.processing_events.pop(event.event_id, None)
            
            # Update persistence
            self._persist_event(event)
    
    def _deliver_with_timeout(self, handler: Callable, event: AnalyticsEvent) -> bool:
        """Deliver event with timeout."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler, event.data)
            
            try:
                result = future.result(timeout=self.delivery_timeout)
                return result is not False  # Allow None or True as success
            except concurrent.futures.TimeoutError:
                logger.warning(f"Delivery timeout for event {event.event_id}")
                event.error_message = "Delivery timeout"
                return False
            except Exception as e:
                logger.error(f"Delivery error for event {event.event_id}: {e}")
                event.error_message = str(e)
                return False
    
    def _default_delivery_handler(self, data: Dict[str, Any]) -> bool:
        """Default delivery handler - just logs the data."""
        logger.info(f"Default delivery for analytics: {data.get('type', 'unknown')}")
        return True
    
    def _handle_delivery_failure(self, event: AnalyticsEvent):
        """Handle failed delivery."""
        with self.lock:
            if event.attempts >= self.max_retries:
                # Max retries exceeded
                event.status = EventStatus.FAILED
                self.failed_events.append(event)
                self.stats['failed_events'] += 1
                logger.error(f"Event {event.event_id} failed after {event.attempts} attempts")
            else:
                # Queue for retry
                event.status = EventStatus.RETRYING
                retry_delay = min(2 ** event.attempts, 300)  # Exponential backoff, max 5 min
                
                # Add back to queue with delay
                threading.Timer(retry_delay, self._requeue_event, args=[event]).start()
                self.stats['retried_events'] += 1
                logger.warning(f"Event {event.event_id} queued for retry (attempt {event.attempts})")
    
    def _requeue_event(self, event: AnalyticsEvent):
        """Requeue event for processing."""
        with self.lock:
            event.status = EventStatus.PENDING
            self.priority_queues[event.priority].append(event)
    
    def _retry_worker(self):
        """Worker thread for handling retries."""
        while self.processing_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Check for stuck processing events
                current_time = datetime.now()
                
                with self.lock:
                    stuck_events = []
                    for event_id, event in self.processing_events.items():
                        if event.last_attempt:
                            processing_time = (current_time - event.last_attempt).total_seconds()
                            if processing_time > self.delivery_timeout * 2:
                                stuck_events.append(event)
                    
                    # Requeue stuck events
                    for event in stuck_events:
                        logger.warning(f"Requeuing stuck event {event.event_id}")
                        self.processing_events.pop(event.event_id, None)
                        event.status = EventStatus.PENDING
                        self.priority_queues[event.priority].append(event)
                
            except Exception as e:
                logger.error(f"Retry worker error: {e}")
    
    def _cleanup_worker(self):
        """Worker thread for cleaning up old delivered events."""
        while self.processing_active:
            try:
                time.sleep(3600)  # Clean up every hour
                
                # Remove old delivered events from database
                cutoff_date = datetime.now() - timedelta(days=7)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        DELETE FROM analytics_events
                        WHERE status = ? AND delivered_at < ?
                    """, (EventStatus.DELIVERED.value, cutoff_date.isoformat()))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old delivered events")
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    def _update_delivery_stats(self, delivery_time: float):
        """Update delivery statistics."""
        with self.lock:
            self.stats['delivered_events'] += 1
            
            # Update average delivery time
            total_deliveries = self.stats['delivered_events']
            current_avg = self.stats['average_delivery_time']
            
            # Incremental average calculation
            self.stats['average_delivery_time'] = (
                (current_avg * (total_deliveries - 1) + delivery_time) / total_deliveries
            )
            
            self.stats['current_queue_size'] = self._get_total_queue_size()
    
    def _get_total_queue_size(self) -> int:
        """Get total number of events in all queues."""
        return sum(len(q) for q in self.priority_queues.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            stats = self.stats.copy()
            
            # Add queue sizes by priority
            stats['queue_sizes'] = {
                priority.name: len(queue)
                for priority, queue in self.priority_queues.items()
            }
            
            # Add processing info
            stats['processing_count'] = len(self.processing_events)
            stats['failed_count'] = len(self.failed_events)
            
            return stats
    
    def get_failed_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent failed events."""
        with self.lock:
            failed = []
            for event in list(self.failed_events)[-limit:]:
                failed.append({
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'priority': event.priority.name,
                    'attempts': event.attempts,
                    'created_at': event.created_at.isoformat(),
                    'error_message': event.error_message
                })
            
            return failed
    
    def retry_failed_event(self, event_id: str) -> bool:
        """Manually retry a failed event."""
        with self.lock:
            # Find event in failed queue
            for event in self.failed_events:
                if event.event_id == event_id:
                    # Reset and requeue
                    event.status = EventStatus.PENDING
                    event.attempts = 0
                    event.error_message = None
                    
                    self.priority_queues[event.priority].append(event)
                    self._persist_event(event)
                    
                    logger.info(f"Manually retrying event {event_id}")
                    return True
        
        return False
    
    def shutdown(self):
        """Shutdown the event queue."""
        self.stop_processing()
        
        # Ensure all events are persisted
        with self.lock:
            for queue in self.priority_queues.values():
                for event in queue:
                    self._persist_event(event)
        
        logger.info("Analytics Event Queue shutdown")