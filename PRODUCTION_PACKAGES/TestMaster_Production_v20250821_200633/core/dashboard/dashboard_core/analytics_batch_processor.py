"""
Analytics Batch Processing System
==================================

Intelligent batching with automatic flush, size/time thresholds,
and priority-based processing for optimal throughput.

Author: TestMaster Team
"""

import logging
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class BatchPriority(Enum):
    """Batch processing priority."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class FlushReason(Enum):
    """Reasons for batch flush."""
    SIZE_THRESHOLD = "size_threshold"
    TIME_THRESHOLD = "time_threshold"
    PRIORITY_TRIGGER = "priority_trigger"
    MANUAL_FLUSH = "manual_flush"
    SHUTDOWN = "shutdown"
    ERROR_RECOVERY = "error_recovery"

@dataclass
class AnalyticsBatch:
    """Represents a batch of analytics."""
    batch_id: str
    created_at: datetime
    items: List[Dict[str, Any]]
    priority: BatchPriority
    size_bytes: int
    item_count: int
    metadata: Dict[str, Any]

class AnalyticsBatchProcessor:
    """
    Intelligent batch processing for analytics delivery.
    """
    
    def __init__(self,
                 batch_size: int = 100,
                 batch_bytes: int = 1048576,  # 1MB
                 flush_interval: float = 5.0,
                 processor_func: Optional[Callable] = None):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Maximum items per batch
            batch_bytes: Maximum bytes per batch
            flush_interval: Seconds between automatic flushes
            processor_func: Function to process batches
        """
        self.batch_size = batch_size
        self.batch_bytes = batch_bytes
        self.flush_interval = flush_interval
        self.processor_func = processor_func
        
        # Batch queues by priority
        self.batches = {
            BatchPriority.CRITICAL: deque(),
            BatchPriority.HIGH: deque(),
            BatchPriority.NORMAL: deque(),
            BatchPriority.LOW: deque()
        }
        
        # Current accumulating batch
        self.current_batch = {
            priority: [] for priority in BatchPriority
        }
        self.current_batch_size = {
            priority: 0 for priority in BatchPriority
        }
        
        # Deduplication
        self.seen_hashes = set()
        self.dedup_window = 1000  # Keep last 1000 hashes
        
        # Statistics
        self.stats = {
            'total_items': 0,
            'total_batches': 0,
            'duplicates_filtered': 0,
            'items_processed': 0,
            'batches_flushed': 0,
            'flush_reasons': defaultdict(int),
            'avg_batch_size': 0,
            'avg_batch_bytes': 0
        }
        
        # Flush strategies
        self.flush_strategies = {
            'aggressive': {'size': 50, 'bytes': 524288, 'interval': 2.0},
            'balanced': {'size': 100, 'bytes': 1048576, 'interval': 5.0},
            'conservative': {'size': 200, 'bytes': 2097152, 'interval': 10.0}
        }
        self.current_strategy = 'balanced'
        
        # Processing thread
        self.processing_active = True
        self.flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True
        )
        self.process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        
        # Start threads
        self.flush_thread.start()
        self.process_thread.start()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        self.flush_lock = threading.Lock()
        
        # Last flush times
        self.last_flush = {
            priority: datetime.now() for priority in BatchPriority
        }
        
        logger.info("Analytics Batch Processor initialized")
    
    def add_item(self,
                item: Dict[str, Any],
                priority: BatchPriority = BatchPriority.NORMAL) -> bool:
        """
        Add item to batch.
        
        Args:
            item: Analytics item to batch
            priority: Processing priority
            
        Returns:
            True if added, False if duplicate
        """
        with self.lock:
            # Deduplication check
            item_hash = self._calculate_hash(item)
            if item_hash in self.seen_hashes:
                self.stats['duplicates_filtered'] += 1
                return False
            
            # Add to dedup set
            self.seen_hashes.add(item_hash)
            if len(self.seen_hashes) > self.dedup_window:
                # Remove oldest (approximately)
                self.seen_hashes.pop()
            
            # Calculate item size
            item_size = len(json.dumps(item, default=str))
            
            # Add to current batch
            self.current_batch[priority].append(item)
            self.current_batch_size[priority] += item_size
            
            # Update statistics
            self.stats['total_items'] += 1
            
            # Check if should flush
            should_flush = self._should_flush(priority)
            
            if should_flush:
                self._flush_batch(priority, FlushReason.SIZE_THRESHOLD)
            
            # Critical items trigger immediate flush
            if priority == BatchPriority.CRITICAL and len(self.current_batch[priority]) > 0:
                self._flush_batch(priority, FlushReason.PRIORITY_TRIGGER)
            
            return True
    
    def add_batch(self,
                 items: List[Dict[str, Any]],
                 priority: BatchPriority = BatchPriority.NORMAL):
        """Add multiple items as batch."""
        added = 0
        for item in items:
            if self.add_item(item, priority):
                added += 1
        
        return added
    
    def flush_all(self, reason: FlushReason = FlushReason.MANUAL_FLUSH):
        """Flush all pending batches."""
        with self.lock:
            for priority in BatchPriority:
                if self.current_batch[priority]:
                    self._flush_batch(priority, reason)
    
    def _should_flush(self, priority: BatchPriority) -> bool:
        """Determine if batch should be flushed."""
        # Get current strategy thresholds
        strategy = self.flush_strategies[self.current_strategy]
        
        # Size threshold
        if len(self.current_batch[priority]) >= strategy['size']:
            return True
        
        # Bytes threshold
        if self.current_batch_size[priority] >= strategy['bytes']:
            return True
        
        # Time threshold
        time_since_flush = (datetime.now() - self.last_flush[priority]).total_seconds()
        if time_since_flush >= strategy['interval'] and self.current_batch[priority]:
            return True
        
        return False
    
    def _flush_batch(self, priority: BatchPriority, reason: FlushReason):
        """Flush current batch to processing queue."""
        if not self.current_batch[priority]:
            return
        
        batch_id = f"batch_{int(time.time() * 1000000)}"
        
        # Create batch
        batch = AnalyticsBatch(
            batch_id=batch_id,
            created_at=datetime.now(),
            items=self.current_batch[priority].copy(),
            priority=priority,
            size_bytes=self.current_batch_size[priority],
            item_count=len(self.current_batch[priority]),
            metadata={
                'flush_reason': reason.value,
                'strategy': self.current_strategy
            }
        )
        
        # Add to processing queue
        self.batches[priority].append(batch)
        
        # Clear current batch
        self.current_batch[priority] = []
        self.current_batch_size[priority] = 0
        self.last_flush[priority] = datetime.now()
        
        # Update statistics
        self.stats['total_batches'] += 1
        self.stats['batches_flushed'] += 1
        self.stats['flush_reasons'][reason.value] += 1
        
        # Update averages
        self._update_averages(batch)
        
        logger.debug(
            f"Flushed batch {batch_id}: {batch.item_count} items, "
            f"{batch.size_bytes} bytes, priority: {priority.name}, "
            f"reason: {reason.value}"
        )
    
    def _flush_loop(self):
        """Background flush loop."""
        while self.processing_active:
            try:
                time.sleep(1)  # Check every second
                
                with self.lock:
                    # Check each priority for time-based flush
                    for priority in BatchPriority:
                        if self._should_flush(priority):
                            self._flush_batch(priority, FlushReason.TIME_THRESHOLD)
                
            except Exception as e:
                logger.error(f"Flush loop error: {e}")
    
    def _process_loop(self):
        """Background batch processing loop."""
        while self.processing_active:
            try:
                # Process batches in priority order
                batch_processed = False
                
                for priority in BatchPriority:
                    if self.batches[priority]:
                        batch = self.batches[priority].popleft()
                        self._process_batch(batch)
                        batch_processed = True
                        break  # Process one batch at a time
                
                if not batch_processed:
                    time.sleep(0.1)  # No batches, wait briefly
                
            except Exception as e:
                logger.error(f"Process loop error: {e}")
    
    def _process_batch(self, batch: AnalyticsBatch):
        """Process a batch of analytics."""
        try:
            if self.processor_func:
                # Use provided processor
                result = self.processor_func(batch.items)
                
                # Update statistics
                self.stats['items_processed'] += batch.item_count
                
                logger.info(
                    f"Processed batch {batch.batch_id}: "
                    f"{batch.item_count} items"
                )
            else:
                # Default processing (log only)
                logger.info(
                    f"Batch ready for processing: {batch.batch_id} "
                    f"({batch.item_count} items)"
                )
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Re-queue for retry if critical
            if batch.priority == BatchPriority.CRITICAL:
                with self.lock:
                    self.batches[BatchPriority.HIGH].append(batch)
    
    def _calculate_hash(self, item: Dict[str, Any]) -> str:
        """Calculate item hash for deduplication."""
        try:
            # Create deterministic string representation
            key_parts = []
            for k in sorted(['id', 'timestamp', 'type', 'source']):
                if k in item:
                    key_parts.append(f"{k}:{item[k]}")
            
            if not key_parts:
                # Fallback to full item hash
                key_parts = [json.dumps(item, sort_keys=True, default=str)]
            
            key_str = '|'.join(key_parts)
            return hashlib.md5(key_str.encode()).hexdigest()[:16]
        
        except:
            # If hashing fails, don't deduplicate
            return str(time.time())
    
    def _update_averages(self, batch: AnalyticsBatch):
        """Update running averages."""
        total_batches = self.stats['total_batches']
        
        if total_batches > 0:
            # Update average batch size
            current_avg_size = self.stats['avg_batch_size']
            self.stats['avg_batch_size'] = (
                (current_avg_size * (total_batches - 1) + batch.item_count) / 
                total_batches
            )
            
            # Update average batch bytes
            current_avg_bytes = self.stats['avg_batch_bytes']
            self.stats['avg_batch_bytes'] = (
                (current_avg_bytes * (total_batches - 1) + batch.size_bytes) / 
                total_batches
            )
    
    def adjust_strategy(self, strategy: str):
        """
        Adjust batching strategy.
        
        Args:
            strategy: One of 'aggressive', 'balanced', 'conservative'
        """
        if strategy in self.flush_strategies:
            self.current_strategy = strategy
            
            # Update thresholds
            config = self.flush_strategies[strategy]
            self.batch_size = config['size']
            self.batch_bytes = config['bytes']
            self.flush_interval = config['interval']
            
            logger.info(f"Switched to {strategy} batching strategy")
    
    def auto_adjust_strategy(self):
        """Automatically adjust strategy based on performance."""
        with self.lock:
            # Check processing rate
            if self.stats['total_items'] > 0:
                processing_rate = self.stats['items_processed'] / self.stats['total_items']
                
                # Adjust based on backlog
                total_pending = sum(
                    len(self.batches[p]) for p in BatchPriority
                )
                
                if total_pending > 50:
                    # Large backlog - use aggressive strategy
                    self.adjust_strategy('aggressive')
                elif total_pending < 10 and processing_rate > 0.95:
                    # Low backlog, high success - use conservative
                    self.adjust_strategy('conservative')
                else:
                    # Normal conditions
                    self.adjust_strategy('balanced')
    
    def get_status(self) -> Dict[str, Any]:
        """Get batch processor status."""
        with self.lock:
            pending_items = sum(
                len(self.current_batch[p]) for p in BatchPriority
            )
            
            queued_batches = sum(
                len(self.batches[p]) for p in BatchPriority
            )
            
            return {
                'pending_items': pending_items,
                'queued_batches': queued_batches,
                'current_strategy': self.current_strategy,
                'statistics': dict(self.stats),
                'flush_reasons': dict(self.stats['flush_reasons']),
                'batch_thresholds': {
                    'size': self.batch_size,
                    'bytes': self.batch_bytes,
                    'interval': self.flush_interval
                },
                'priority_distribution': {
                    p.name: len(self.current_batch[p]) 
                    for p in BatchPriority
                }
            }
    
    def set_processor(self, processor_func: Callable):
        """Set batch processor function."""
        self.processor_func = processor_func
    
    def wait_for_completion(self, timeout: float = 30) -> bool:
        """
        Wait for all batches to be processed.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if completed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                # Check if everything is processed
                pending = sum(len(self.current_batch[p]) for p in BatchPriority)
                queued = sum(len(self.batches[p]) for p in BatchPriority)
                
                if pending == 0 and queued == 0:
                    return True
            
            time.sleep(0.1)
        
        return False
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown batch processor.
        
        Args:
            wait: Wait for pending batches to complete
        """
        # Flush all pending
        self.flush_all(FlushReason.SHUTDOWN)
        
        if wait:
            self.wait_for_completion(timeout=10)
        
        # Stop threads
        self.processing_active = False
        
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5)
        
        logger.info(f"Batch Processor shutdown - Stats: {self.stats}")

# Global batch processor instance
batch_processor = AnalyticsBatchProcessor()