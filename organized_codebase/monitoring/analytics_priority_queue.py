#!/usr/bin/env python3
"""
Analytics Priority Queuing with Express Lanes
=============================================

Provides ultra-reliability through intelligent priority-based queuing,
express lanes for critical analytics, dynamic load balancing, and QoS guarantees.

Author: TestMaster Team
"""

import logging
import threading
import time
import sqlite3
import os
import json
import heapq
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import statistics


class QueuePriority(Enum):
    """Queue priority levels."""
    EMERGENCY = 0       # Emergency/critical analytics
    EXPRESS = 1         # Express lane for high priority
    HIGH = 2           # High priority
    NORMAL = 3         # Normal priority  
    LOW = 4            # Low priority
    BULK = 5           # Bulk processing


class QueueType(Enum):
    """Types of priority queues."""
    EXPRESS_LANE = "express_lane"           # High-speed processing
    NORMAL_LANE = "normal_lane"             # Standard processing
    BULK_LANE = "bulk_lane"                 # Batch processing
    OVERFLOW_LANE = "overflow_lane"         # Overflow handling


class ProcessingStatus(Enum):
    """Analytics processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    THROTTLED = "throttled"


@dataclass
class QueuedAnalytics:
    """Queued analytics item."""
    analytics_id: str
    priority: QueuePriority
    queue_type: QueueType
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    queued_at: datetime
    expires_at: Optional[datetime]
    processing_estimate_ms: float
    retry_count: int = 0
    last_error: Optional[str] = None
    
    def __lt__(self, other):
        """For priority queue comparison."""
        # Lower priority value = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # Earlier queued items have higher priority within same priority level
        return self.queued_at < other.queued_at


@dataclass
class QueueMetrics:
    """Queue performance metrics."""
    queue_type: QueueType
    current_size: int
    max_size: int
    throughput_per_second: float
    avg_wait_time_ms: float
    avg_processing_time_ms: float
    success_rate: float
    error_rate: float
    last_updated: datetime


@dataclass
class ProcessingLane:
    """Processing lane configuration."""
    lane_id: str
    queue_type: QueueType
    max_concurrent: int
    max_queue_size: int
    timeout_seconds: float
    rate_limit_per_second: int
    priority_threshold: QueuePriority
    processing_workers: int = field(default=2)


class AnalyticsPriorityQueue:
    """Priority-based analytics queue with express lanes."""
    
    def __init__(
        self,
        db_path: str = "data/priority_queue.db",
        max_queue_size: int = 10000,
        processing_workers: int = 8
    ):
        """Initialize the priority queue system."""
        self.db_path = db_path
        self.max_queue_size = max_queue_size
        self.processing_workers = processing_workers
        
        # Queue management
        self.queues: Dict[QueueType, List[QueuedAnalytics]] = {
            QueueType.EXPRESS_LANE: [],
            QueueType.NORMAL_LANE: [],
            QueueType.BULK_LANE: [],
            QueueType.OVERFLOW_LANE: []
        }
        
        # Processing lanes
        self.lanes: Dict[QueueType, ProcessingLane] = {}
        self.active_processors: Dict[str, Dict[str, Any]] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.executor = ThreadPoolExecutor(max_workers=processing_workers)
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Metrics and monitoring
        self.metrics: Dict[QueueType, QueueMetrics] = {}
        self.processing_history: List[Dict[str, Any]] = []
        
        # Rate limiting
        self.rate_limiters: Dict[QueueType, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_failed': 0,
            'total_expired': 0,
            'avg_queue_wait_ms': 0.0,
            'throughput_per_second': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        self._setup_processing_lanes()
        self._initialize_metrics()
        
        self.logger.info("Analytics Priority Queue system initialized")
    
    def _initialize_database(self):
        """Initialize the priority queue database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Queued analytics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS queued_analytics (
                        analytics_id TEXT PRIMARY KEY,
                        priority INTEGER NOT NULL,
                        queue_type TEXT NOT NULL,
                        data TEXT NOT NULL,
                        metadata TEXT,
                        queued_at TEXT NOT NULL,
                        expires_at TEXT,
                        processing_estimate_ms REAL NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        last_error TEXT,
                        status TEXT DEFAULT 'queued'
                    )
                """)
                
                # Processing history table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analytics_id TEXT NOT NULL,
                        queue_type TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        queued_at TEXT NOT NULL,
                        processing_started_at TEXT,
                        processing_completed_at TEXT,
                        wait_time_ms REAL,
                        processing_time_ms REAL,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        worker_id TEXT
                    )
                """)
                
                # Queue metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS queue_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        queue_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        current_size INTEGER NOT NULL,
                        throughput_per_second REAL NOT NULL,
                        avg_wait_time_ms REAL NOT NULL,
                        avg_processing_time_ms REAL NOT NULL,
                        success_rate REAL NOT NULL,
                        error_rate REAL NOT NULL
                    )
                """)
                
                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_queued_priority ON queued_analytics(priority)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_queued_type ON queued_analytics(queue_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON processing_history(processing_completed_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON queue_metrics(timestamp)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _setup_processing_lanes(self):
        """Setup processing lanes with different configurations."""
        self.lanes = {
            QueueType.EXPRESS_LANE: ProcessingLane(
                lane_id="express",
                queue_type=QueueType.EXPRESS_LANE,
                max_concurrent=4,
                max_queue_size=100,
                timeout_seconds=10.0,
                rate_limit_per_second=50,
                priority_threshold=QueuePriority.HIGH,
                processing_workers=3
            ),
            QueueType.NORMAL_LANE: ProcessingLane(
                lane_id="normal",
                queue_type=QueueType.NORMAL_LANE,
                max_concurrent=6,
                max_queue_size=5000,
                timeout_seconds=30.0,
                rate_limit_per_second=25,
                priority_threshold=QueuePriority.NORMAL,
                processing_workers=4
            ),
            QueueType.BULK_LANE: ProcessingLane(
                lane_id="bulk",
                queue_type=QueueType.BULK_LANE,
                max_concurrent=8,
                max_queue_size=8000,
                timeout_seconds=60.0,
                rate_limit_per_second=10,
                priority_threshold=QueuePriority.LOW,
                processing_workers=2
            ),
            QueueType.OVERFLOW_LANE: ProcessingLane(
                lane_id="overflow",
                queue_type=QueueType.OVERFLOW_LANE,
                max_concurrent=2,
                max_queue_size=1000,
                timeout_seconds=120.0,
                rate_limit_per_second=5,
                priority_threshold=QueuePriority.BULK,
                processing_workers=1
            )
        }
    
    def _initialize_metrics(self):
        """Initialize queue metrics."""
        for queue_type in QueueType:
            self.metrics[queue_type] = QueueMetrics(
                queue_type=queue_type,
                current_size=0,
                max_size=self.lanes[queue_type].max_queue_size,
                throughput_per_second=0.0,
                avg_wait_time_ms=0.0,
                avg_processing_time_ms=0.0,
                success_rate=100.0,
                error_rate=0.0,
                last_updated=datetime.now()
            )
            
            self.rate_limiters[queue_type] = {
                'tokens': self.lanes[queue_type].rate_limit_per_second,
                'last_refill': time.time(),
                'max_tokens': self.lanes[queue_type].rate_limit_per_second
            }
    
    def enqueue_analytics(
        self,
        analytics_id: str,
        data: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL,
        metadata: Dict[str, Any] = None,
        expiration_minutes: int = 60,
        processing_estimate_ms: float = 100.0
    ) -> bool:
        """Enqueue analytics for processing."""
        try:
            # Determine appropriate queue type based on priority
            queue_type = self._determine_queue_type(priority, len(data))
            
            # Check queue capacity
            if not self._check_queue_capacity(queue_type):
                # Try overflow lane
                if queue_type != QueueType.OVERFLOW_LANE:
                    queue_type = QueueType.OVERFLOW_LANE
                    if not self._check_queue_capacity(queue_type):
                        self.logger.warning(f"All queues full, dropping analytics {analytics_id}")
                        return False
            
            # Create queued analytics item
            now = datetime.now()
            expires_at = now + timedelta(minutes=expiration_minutes) if expiration_minutes > 0 else None
            
            queued_item = QueuedAnalytics(
                analytics_id=analytics_id,
                priority=priority,
                queue_type=queue_type,
                data=data,
                metadata=metadata or {},
                queued_at=now,
                expires_at=expires_at,
                processing_estimate_ms=processing_estimate_ms
            )
            
            # Add to appropriate queue
            with self.condition:
                heapq.heappush(self.queues[queue_type], queued_item)
                self.stats['total_queued'] += 1
                self.metrics[queue_type].current_size = len(self.queues[queue_type])
                self.condition.notify_all()
            
            # Save to database
            self._save_queued_item_to_db(queued_item)
            
            self.logger.info(f"Analytics queued: {analytics_id} in {queue_type.value} (priority: {priority.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enqueue analytics {analytics_id}: {e}")
            return False
    
    def _determine_queue_type(self, priority: QueuePriority, data_size: int) -> QueueType:
        """Determine appropriate queue type based on priority and data size."""
        if priority in [QueuePriority.EMERGENCY, QueuePriority.EXPRESS]:
            return QueueType.EXPRESS_LANE
        elif priority == QueuePriority.HIGH:
            return QueueType.EXPRESS_LANE if data_size < 1000 else QueueType.NORMAL_LANE
        elif priority == QueuePriority.NORMAL:
            return QueueType.NORMAL_LANE
        elif priority == QueuePriority.LOW:
            return QueueType.BULK_LANE if data_size > 500 else QueueType.NORMAL_LANE
        else:  # BULK
            return QueueType.BULK_LANE
    
    def _check_queue_capacity(self, queue_type: QueueType) -> bool:
        """Check if queue has capacity."""
        current_size = len(self.queues[queue_type])
        max_size = self.lanes[queue_type].max_queue_size
        return current_size < max_size
    
    def _check_rate_limit(self, queue_type: QueueType) -> bool:
        """Check rate limiting for queue type."""
        now = time.time()
        limiter = self.rate_limiters[queue_type]
        
        # Refill tokens based on time elapsed
        time_elapsed = now - limiter['last_refill']
        tokens_to_add = time_elapsed * self.lanes[queue_type].rate_limit_per_second
        limiter['tokens'] = min(limiter['max_tokens'], limiter['tokens'] + tokens_to_add)
        limiter['last_refill'] = now
        
        # Check if we have tokens available
        if limiter['tokens'] >= 1.0:
            limiter['tokens'] -= 1.0
            return True
        return False
    
    def _save_queued_item_to_db(self, item: QueuedAnalytics):
        """Save queued item to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO queued_analytics 
                    (analytics_id, priority, queue_type, data, metadata,
                     queued_at, expires_at, processing_estimate_ms, retry_count, last_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.analytics_id,
                    item.priority.value,
                    item.queue_type.value,
                    json.dumps(item.data),
                    json.dumps(item.metadata),
                    item.queued_at.isoformat(),
                    item.expires_at.isoformat() if item.expires_at else None,
                    item.processing_estimate_ms,
                    item.retry_count,
                    item.last_error
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save queued item to database: {e}")
    
    def start_processing(self):
        """Start the queue processing workers."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads for each lane
        for queue_type, lane in self.lanes.items():
            for i in range(lane.processing_workers):
                worker_id = f"{lane.lane_id}_worker_{i}"
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(queue_type, worker_id),
                    daemon=True
                )
                thread.start()
                self.worker_threads.append(thread)
        
        # Start metrics monitoring
        monitor_thread = threading.Thread(target=self._metrics_monitor, daemon=True)
        monitor_thread.start()
        self.worker_threads.append(monitor_thread)
        
        self.logger.info(f"Started {len(self.worker_threads)} processing workers")
    
    def _worker_loop(self, queue_type: QueueType, worker_id: str):
        """Worker loop for processing analytics from a specific queue."""
        lane = self.lanes[queue_type]
        
        while self.running:
            try:
                # Wait for items in queue
                with self.condition:
                    while self.running and not self.queues[queue_type]:
                        self.condition.wait(timeout=1.0)
                    
                    if not self.running:
                        break
                    
                    # Check rate limiting
                    if not self._check_rate_limit(queue_type):
                        time.sleep(0.1)  # Brief delay if rate limited
                        continue
                    
                    # Get next item
                    if self.queues[queue_type]:
                        item = heapq.heappop(self.queues[queue_type])
                        self.metrics[queue_type].current_size = len(self.queues[queue_type])
                    else:
                        continue
                
                # Process the analytics item
                self._process_analytics_item(item, worker_id)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(1.0)
    
    def _process_analytics_item(self, item: QueuedAnalytics, worker_id: str):
        """Process individual analytics item."""
        processing_start = time.time()
        processing_started_at = datetime.now()
        
        # Calculate wait time
        wait_time_ms = (processing_started_at - item.queued_at).total_seconds() * 1000
        
        # Track active processing
        with self.lock:
            self.active_processors[item.analytics_id] = {
                'worker_id': worker_id,
                'queue_type': item.queue_type.value,
                'started_at': processing_started_at,
                'item': item
            }
        
        try:
            # Check if item has expired
            if item.expires_at and datetime.now() > item.expires_at:
                self._record_processing_result(
                    item, worker_id, wait_time_ms, 0.0,
                    ProcessingStatus.EXPIRED, "Item expired before processing"
                )
                return
            
            # Simulate analytics processing (replace with actual processing logic)
            result = self._simulate_analytics_processing(item)
            
            processing_time_ms = (time.time() - processing_start) * 1000
            
            if result['success']:
                self._record_processing_result(
                    item, worker_id, wait_time_ms, processing_time_ms,
                    ProcessingStatus.COMPLETED, None
                )
                self.stats['total_processed'] += 1
                
            else:
                # Handle failure - maybe retry
                if item.retry_count < 3:  # Max 3 retries
                    item.retry_count += 1
                    item.last_error = result['error']
                    item.queued_at = datetime.now()  # Reset queue time for retry
                    
                    # Re-queue for retry (with lower priority)
                    retry_priority = QueuePriority(min(QueuePriority.BULK.value, item.priority.value + 1))
                    retry_queue_type = self._determine_queue_type(retry_priority, len(item.data))
                    
                    with self.condition:
                        heapq.heappush(self.queues[retry_queue_type], item)
                        self.condition.notify()
                    
                    self.logger.warning(f"Retrying analytics {item.analytics_id} (attempt {item.retry_count})")
                    
                else:
                    self._record_processing_result(
                        item, worker_id, wait_time_ms, processing_time_ms,
                        ProcessingStatus.FAILED, result['error']
                    )
                    self.stats['total_failed'] += 1
                
        except Exception as e:
            processing_time_ms = (time.time() - processing_start) * 1000
            self._record_processing_result(
                item, worker_id, wait_time_ms, processing_time_ms,
                ProcessingStatus.FAILED, str(e)
            )
            self.stats['total_failed'] += 1
            
        finally:
            # Remove from active processors
            with self.lock:
                self.active_processors.pop(item.analytics_id, None)
    
    def _simulate_analytics_processing(self, item: QueuedAnalytics) -> Dict[str, Any]:
        """Simulate analytics processing (replace with actual processing logic)."""
        # Simulate processing time based on estimate
        processing_time = item.processing_estimate_ms / 1000.0
        time.sleep(min(processing_time, 2.0))  # Cap simulation time
        
        # Simulate 95% success rate
        success = (hash(item.analytics_id) % 100) < 95
        
        if success:
            return {
                'success': True,
                'result': f"Processed analytics {item.analytics_id}",
                'metrics': {
                    'items_processed': 1,
                    'processing_time_ms': processing_time * 1000
                }
            }
        else:
            return {
                'success': False,
                'error': f"Simulated processing failure for {item.analytics_id}"
            }
    
    def _record_processing_result(
        self,
        item: QueuedAnalytics,
        worker_id: str,
        wait_time_ms: float,
        processing_time_ms: float,
        status: ProcessingStatus,
        error_message: Optional[str]
    ):
        """Record processing result."""
        try:
            # Add to processing history
            result = {
                'analytics_id': item.analytics_id,
                'queue_type': item.queue_type.value,
                'priority': item.priority.value,
                'queued_at': item.queued_at,
                'processing_started_at': datetime.now(),
                'processing_completed_at': datetime.now(),
                'wait_time_ms': wait_time_ms,
                'processing_time_ms': processing_time_ms,
                'status': status.value,
                'error_message': error_message,
                'worker_id': worker_id
            }
            
            self.processing_history.append(result)
            
            # Keep only recent history (last 1000 items)
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-1000:]
            
            # Save to database
            self._save_processing_result_to_db(result)
            
            self.logger.debug(f"Processed {item.analytics_id}: {status.value} (wait: {wait_time_ms:.1f}ms, process: {processing_time_ms:.1f}ms)")
            
        except Exception as e:
            self.logger.error(f"Failed to record processing result: {e}")
    
    def _save_processing_result_to_db(self, result: Dict[str, Any]):
        """Save processing result to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO processing_history 
                    (analytics_id, queue_type, priority, queued_at,
                     processing_started_at, processing_completed_at,
                     wait_time_ms, processing_time_ms, status, error_message, worker_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['analytics_id'],
                    result['queue_type'],
                    result['priority'],
                    result['queued_at'].isoformat(),
                    result['processing_started_at'].isoformat(),
                    result['processing_completed_at'].isoformat(),
                    result['wait_time_ms'],
                    result['processing_time_ms'],
                    result['status'],
                    result['error_message'],
                    result['worker_id']
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save processing result to database: {e}")
    
    def _metrics_monitor(self):
        """Background metrics monitoring."""
        while self.running:
            try:
                self._update_queue_metrics()
                self._cleanup_expired_items()
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics monitoring error: {e}")
                time.sleep(10)
    
    def _update_queue_metrics(self):
        """Update queue performance metrics."""
        now = datetime.now()
        recent_cutoff = now - timedelta(minutes=5)
        
        for queue_type in QueueType:
            # Get recent processing results for this queue
            recent_results = [
                r for r in self.processing_history
                if (r['queue_type'] == queue_type.value and
                    r['processing_completed_at'] >= recent_cutoff)
            ]
            
            if recent_results:
                # Calculate metrics
                wait_times = [r['wait_time_ms'] for r in recent_results]
                processing_times = [r['processing_time_ms'] for r in recent_results]
                successful = [r for r in recent_results if r['status'] == 'completed']
                
                throughput = len(recent_results) / 300.0  # Per second over 5 minutes
                avg_wait_time = statistics.mean(wait_times) if wait_times else 0.0
                avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
                success_rate = (len(successful) / len(recent_results)) * 100 if recent_results else 100.0
                error_rate = 100.0 - success_rate
                
                # Update metrics
                self.metrics[queue_type].throughput_per_second = throughput
                self.metrics[queue_type].avg_wait_time_ms = avg_wait_time
                self.metrics[queue_type].avg_processing_time_ms = avg_processing_time
                self.metrics[queue_type].success_rate = success_rate
                self.metrics[queue_type].error_rate = error_rate
                self.metrics[queue_type].last_updated = now
                
                # Save metrics to database
                self._save_metrics_to_db(self.metrics[queue_type])
    
    def _save_metrics_to_db(self, metrics: QueueMetrics):
        """Save queue metrics to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO queue_metrics 
                    (queue_type, timestamp, current_size, throughput_per_second,
                     avg_wait_time_ms, avg_processing_time_ms, success_rate, error_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.queue_type.value,
                    metrics.last_updated.isoformat(),
                    metrics.current_size,
                    metrics.throughput_per_second,
                    metrics.avg_wait_time_ms,
                    metrics.avg_processing_time_ms,
                    metrics.success_rate,
                    metrics.error_rate
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save metrics to database: {e}")
    
    def _cleanup_expired_items(self):
        """Remove expired items from queues."""
        now = datetime.now()
        
        with self.condition:
            for queue_type in QueueType:
                queue = self.queues[queue_type]
                expired_items = []
                
                # Find expired items
                for item in queue:
                    if item.expires_at and now > item.expires_at:
                        expired_items.append(item)
                
                # Remove expired items
                for item in expired_items:
                    try:
                        queue.remove(item)
                        self.stats['total_expired'] += 1
                        
                        self._record_processing_result(
                            item, "cleanup", 0.0, 0.0,
                            ProcessingStatus.EXPIRED, "Expired in queue"
                        )
                        
                    except ValueError:
                        pass  # Item already removed
                
                # Re-heapify if items were removed
                if expired_items:
                    heapq.heapify(queue)
                    self.metrics[queue_type].current_size = len(queue)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status."""
        with self.lock:
            queue_sizes = {qt.value: len(self.queues[qt]) for qt in QueueType}
            active_processing = len(self.active_processors)
            
            # Calculate recent performance
            recent_results = [
                r for r in self.processing_history[-100:]  # Last 100 results
                if r['status'] == 'completed'
            ]
            
            if recent_results:
                avg_wait_time = statistics.mean([r['wait_time_ms'] for r in recent_results])
                avg_processing_time = statistics.mean([r['processing_time_ms'] for r in recent_results])
                throughput = len(recent_results) / 60.0  # Approximate per second
            else:
                avg_wait_time = avg_processing_time = throughput = 0.0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'queue_sizes': queue_sizes,
                'total_queued_items': sum(queue_sizes.values()),
                'active_processing': active_processing,
                'processing_workers': len(self.worker_threads),
                'metrics': {
                    qt.value: asdict(self.metrics[qt]) for qt in QueueType
                },
                'performance': {
                    'avg_wait_time_ms': avg_wait_time,
                    'avg_processing_time_ms': avg_processing_time,
                    'throughput_per_second': throughput
                },
                'statistics': self.stats.copy(),
                'system_health': self._calculate_system_health()
            }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health."""
        total_items = sum(len(queue) for queue in self.queues.values())
        total_capacity = sum(lane.max_queue_size for lane in self.lanes.values())
        
        utilization = (total_items / total_capacity) * 100 if total_capacity > 0 else 0
        
        # Calculate health score
        if utilization < 50:
            health_status = "healthy"
            health_score = 100 - utilization
        elif utilization < 75:
            health_status = "moderate"
            health_score = 100 - (utilization * 1.5)
        elif utilization < 90:
            health_status = "stressed"
            health_score = 100 - (utilization * 2)
        else:
            health_status = "critical"
            health_score = max(0, 100 - (utilization * 3))
        
        return {
            'status': health_status,
            'score': health_score,
            'utilization_percent': utilization,
            'bottlenecks': self._identify_bottlenecks()
        }
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify processing bottlenecks."""
        bottlenecks = []
        
        for queue_type, queue in self.queues.items():
            lane = self.lanes[queue_type]
            utilization = len(queue) / lane.max_queue_size
            
            if utilization > 0.8:
                bottlenecks.append(f"{queue_type.value}_queue_near_capacity")
            
            if self.metrics[queue_type].avg_wait_time_ms > 5000:  # 5 seconds
                bottlenecks.append(f"{queue_type.value}_high_wait_times")
            
            if self.metrics[queue_type].error_rate > 10:  # 10% error rate
                bottlenecks.append(f"{queue_type.value}_high_error_rate")
        
        return bottlenecks
    
    def stop_processing(self):
        """Stop all queue processing."""
        self.running = False
        
        with self.condition:
            self.condition.notify_all()
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Queue processing stopped")
    
    def shutdown(self):
        """Shutdown the priority queue system."""
        self.stop_processing()


# Global instance for easy access
priority_queue = None

def get_priority_queue() -> AnalyticsPriorityQueue:
    """Get the global priority queue instance."""
    global priority_queue
    if priority_queue is None:
        priority_queue = AnalyticsPriorityQueue()
    return priority_queue


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    queue_system = AnalyticsPriorityQueue()
    queue_system.start_processing()
    
    try:
        # Simulate analytics queuing with different priorities
        for i in range(50):
            priority = QueuePriority.EMERGENCY if i % 20 == 0 else \
                      QueuePriority.EXPRESS if i % 10 == 0 else \
                      QueuePriority.HIGH if i % 5 == 0 else \
                      QueuePriority.NORMAL
            
            analytics_data = {
                'test_id': f'test_{i}',
                'timestamp': datetime.now().isoformat(),
                'data_size': len(f'analytics_data_{i}'),
                'priority_level': priority.value
            }
            
            queue_system.enqueue_analytics(
                analytics_id=f'analytics_{i}',
                data=analytics_data,
                priority=priority,
                metadata={'source': 'test_generator'},
                expiration_minutes=30,
                processing_estimate_ms=100.0 + (i % 500)
            )
            
            time.sleep(0.1)
        
        # Wait for processing
        time.sleep(10)
        
        # Get queue status
        status = queue_system.get_queue_status()
        print(json.dumps(status, indent=2, default=str))
        
    except KeyboardInterrupt:
        print("Stopping priority queue system...")
    
    finally:
        queue_system.shutdown()