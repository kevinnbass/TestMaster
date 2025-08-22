"""
Analytics Dead Letter Queue System
===================================

Handles permanently failed analytics with retry exhaustion, providing
recovery mechanisms and analysis of failure patterns.

Author: TestMaster Team
"""

import logging
import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle

logger = logging.getLogger(__name__)

class FailureReason(Enum):
    """Reasons for dead letter queue entry."""
    MAX_RETRIES = "max_retries"
    VALIDATION_ERROR = "validation_error"
    SERIALIZATION_ERROR = "serialization_error"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER = "circuit_breaker"
    CORRUPT_DATA = "corrupt_data"
    ENDPOINT_GONE = "endpoint_gone"
    UNKNOWN = "unknown"

@dataclass
class DeadLetterEntry:
    """Entry in dead letter queue."""
    entry_id: str
    timestamp: datetime
    analytics_data: Any
    failure_reason: FailureReason
    error_details: str
    retry_count: int
    last_retry: datetime
    endpoint: str
    checksum: str
    can_retry: bool
    metadata: Dict[str, Any]

class AnalyticsDeadLetterQueue:
    """
    Manages failed analytics that cannot be delivered.
    """
    
    def __init__(self,
                 db_path: str = "dead_letter_queue.db",
                 max_entries: int = 10000,
                 retention_days: int = 30):
        """
        Initialize dead letter queue.
        
        Args:
            db_path: Database path for persistence
            max_entries: Maximum entries to retain
            retention_days: Days to retain entries
        """
        self.db_path = db_path
        self.max_entries = max_entries
        self.retention_days = retention_days
        
        # In-memory cache
        self.queue = deque(maxlen=max_entries)
        self.entry_index = {}
        
        # Failure analysis
        self.failure_patterns = defaultdict(int)
        self.endpoint_failures = defaultdict(list)
        
        # Recovery strategies
        self.recovery_strategies = {
            FailureReason.VALIDATION_ERROR: self._recover_validation,
            FailureReason.SERIALIZATION_ERROR: self._recover_serialization,
            FailureReason.TIMEOUT: self._recover_timeout,
            FailureReason.CORRUPT_DATA: self._recover_corrupt
        }
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'recovered_entries': 0,
            'purged_entries': 0,
            'reprocessed_entries': 0,
            'failure_by_reason': defaultdict(int)
        }
        
        # Setup persistence
        self._setup_database()
        self._load_from_database()
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Analytics Dead Letter Queue initialized")
    
    def add_entry(self,
                 analytics_data: Any,
                 failure_reason: FailureReason,
                 error_details: str,
                 retry_count: int = 0,
                 endpoint: str = "unknown",
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add failed analytics to dead letter queue.
        
        Args:
            analytics_data: Failed analytics data
            failure_reason: Reason for failure
            error_details: Error details
            retry_count: Number of retries attempted
            endpoint: Target endpoint
            metadata: Additional metadata
            
        Returns:
            Entry ID
        """
        entry_id = f"dlq_{int(time.time() * 1000000)}"
        
        with self.lock:
            # Calculate checksum
            checksum = self._calculate_checksum(analytics_data)
            
            # Check for duplicates
            if checksum in self.entry_index:
                existing = self.entry_index[checksum]
                logger.debug(f"Duplicate entry detected: {existing}")
                return existing
            
            # Determine if can retry
            can_retry = self._can_retry(failure_reason, retry_count)
            
            # Create entry
            entry = DeadLetterEntry(
                entry_id=entry_id,
                timestamp=datetime.now(),
                analytics_data=analytics_data,
                failure_reason=failure_reason,
                error_details=error_details,
                retry_count=retry_count,
                last_retry=datetime.now(),
                endpoint=endpoint,
                checksum=checksum,
                can_retry=can_retry,
                metadata=metadata or {}
            )
            
            # Add to queue
            self.queue.append(entry)
            self.entry_index[checksum] = entry_id
            
            # Update patterns
            self.failure_patterns[failure_reason] += 1
            self.endpoint_failures[endpoint].append({
                'timestamp': datetime.now(),
                'reason': failure_reason
            })
            
            # Update statistics
            self.stats['total_entries'] += 1
            self.stats['failure_by_reason'][failure_reason.value] += 1
            
            # Persist to database
            self._persist_entry(entry)
            
            # Log for monitoring
            logger.warning(
                f"Added to dead letter queue: {entry_id} "
                f"(reason: {failure_reason.value}, endpoint: {endpoint})"
            )
            
            # Attempt immediate recovery if possible
            if can_retry and failure_reason in self.recovery_strategies:
                self._attempt_recovery(entry)
            
            return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[DeadLetterEntry]:
        """Get entry by ID."""
        with self.lock:
            for entry in self.queue:
                if entry.entry_id == entry_id:
                    return entry
            return None
    
    def reprocess_entry(self, entry_id: str, processor_func) -> bool:
        """
        Attempt to reprocess an entry.
        
        Args:
            entry_id: Entry ID to reprocess
            processor_func: Function to process the data
            
        Returns:
            Success status
        """
        with self.lock:
            entry = self.get_entry(entry_id)
            if not entry:
                return False
            
            try:
                # Attempt reprocessing
                result = processor_func(entry.analytics_data)
                
                # If successful, remove from queue
                self.queue.remove(entry)
                del self.entry_index[entry.checksum]
                
                # Update database
                self._remove_from_database(entry_id)
                
                # Update statistics
                self.stats['reprocessed_entries'] += 1
                
                logger.info(f"Successfully reprocessed entry: {entry_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reprocess {entry_id}: {e}")
                
                # Update retry count
                entry.retry_count += 1
                entry.last_retry = datetime.now()
                
                # Check if should keep retrying
                if entry.retry_count > 10:
                    entry.can_retry = False
                
                return False
    
    def bulk_reprocess(self, 
                       processor_func,
                       filter_func=None,
                       max_entries: int = 100) -> Dict[str, Any]:
        """
        Bulk reprocess entries.
        
        Args:
            processor_func: Processing function
            filter_func: Optional filter for entries
            max_entries: Maximum entries to process
            
        Returns:
            Processing results
        """
        results = {
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'skipped': 0
        }
        
        with self.lock:
            entries_to_process = []
            
            for entry in list(self.queue)[:max_entries]:
                if filter_func and not filter_func(entry):
                    results['skipped'] += 1
                    continue
                    
                if entry.can_retry:
                    entries_to_process.append(entry)
            
            for entry in entries_to_process:
                results['processed'] += 1
                
                if self.reprocess_entry(entry.entry_id, processor_func):
                    results['succeeded'] += 1
                else:
                    results['failed'] += 1
        
        return results
    
    def _can_retry(self, reason: FailureReason, retry_count: int) -> bool:
        """Determine if entry can be retried."""
        # Some failures should not be retried
        non_retryable = [
            FailureReason.VALIDATION_ERROR,
            FailureReason.CORRUPT_DATA,
            FailureReason.ENDPOINT_GONE
        ]
        
        if reason in non_retryable:
            return False
        
        # Check retry count
        if retry_count > 10:
            return False
        
        return True
    
    def _attempt_recovery(self, entry: DeadLetterEntry):
        """Attempt automatic recovery."""
        strategy = self.recovery_strategies.get(entry.failure_reason)
        
        if strategy:
            try:
                recovered_data = strategy(entry)
                if recovered_data:
                    entry.analytics_data = recovered_data
                    entry.metadata['recovered'] = True
                    self.stats['recovered_entries'] += 1
                    logger.info(f"Recovered entry {entry.entry_id}")
            except Exception as e:
                logger.error(f"Recovery failed for {entry.entry_id}: {e}")
    
    def _recover_validation(self, entry: DeadLetterEntry) -> Optional[Any]:
        """Recover from validation errors."""
        try:
            data = entry.analytics_data
            
            # Remove invalid fields
            if isinstance(data, dict):
                cleaned = {}
                for key, value in data.items():
                    if value is not None and value != 'null':
                        cleaned[key] = value
                
                # Add required fields
                if 'timestamp' not in cleaned:
                    cleaned['timestamp'] = datetime.now().isoformat()
                
                return cleaned
        except:
            return None
    
    def _recover_serialization(self, entry: DeadLetterEntry) -> Optional[Any]:
        """Recover from serialization errors."""
        try:
            # Convert to serializable format
            return json.loads(json.dumps(entry.analytics_data, default=str))
        except:
            return None
    
    def _recover_timeout(self, entry: DeadLetterEntry) -> Optional[Any]:
        """Recover from timeout errors."""
        # Timeout recovery might involve reducing data size
        try:
            if isinstance(entry.analytics_data, dict):
                # Keep only essential fields
                essential = ['timestamp', 'status', 'metrics', 'error']
                return {k: v for k, v in entry.analytics_data.items() 
                       if k in essential}
        except:
            return None
    
    def _recover_corrupt(self, entry: DeadLetterEntry) -> Optional[Any]:
        """Recover from corrupt data."""
        # Attempt to reconstruct minimal valid data
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'recovered_from_corruption',
            'original_checksum': entry.checksum,
            'recovery_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate data checksum."""
        try:
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True, default=str)
            else:
                data_str = str(data)
            
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        except:
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def _setup_database(self):
        """Setup SQLite database."""
        try:
            self.db = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    analytics_data BLOB,
                    failure_reason TEXT,
                    error_details TEXT,
                    retry_count INTEGER,
                    last_retry TEXT,
                    endpoint TEXT,
                    checksum TEXT,
                    can_retry INTEGER,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON dead_letter_entries(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checksum 
                ON dead_letter_entries(checksum)
            """)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            self.db = None
    
    def _persist_entry(self, entry: DeadLetterEntry):
        """Persist entry to database."""
        if not self.db:
            return
        
        try:
            cursor = self.db.cursor()
            
            # Serialize analytics data
            data_blob = pickle.dumps(entry.analytics_data)
            
            cursor.execute("""
                INSERT OR REPLACE INTO dead_letter_entries
                (entry_id, timestamp, analytics_data, failure_reason,
                 error_details, retry_count, last_retry, endpoint,
                 checksum, can_retry, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.timestamp.isoformat(),
                data_blob,
                entry.failure_reason.value,
                entry.error_details,
                entry.retry_count,
                entry.last_retry.isoformat(),
                entry.endpoint,
                entry.checksum,
                1 if entry.can_retry else 0,
                json.dumps(entry.metadata)
            ))
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to persist entry: {e}")
    
    def _load_from_database(self):
        """Load entries from database on startup."""
        if not self.db:
            return
        
        try:
            cursor = self.db.cursor()
            
            # Load recent entries
            cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
            
            cursor.execute("""
                SELECT * FROM dead_letter_entries
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff, self.max_entries))
            
            rows = cursor.fetchall()
            
            for row in rows:
                try:
                    entry = DeadLetterEntry(
                        entry_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        analytics_data=SafePickleHandler.safe_load(row[2]),
                        failure_reason=FailureReason(row[3]),
                        error_details=row[4],
                        retry_count=row[5],
                        last_retry=datetime.fromisoformat(row[6]),
                        endpoint=row[7],
                        checksum=row[8],
                        can_retry=bool(row[9]),
                        metadata=json.loads(row[10])
                    )
                    
                    self.queue.append(entry)
                    self.entry_index[entry.checksum] = entry.entry_id
                    
                except Exception as e:
                    logger.error(f"Failed to load entry: {e}")
            
            logger.info(f"Loaded {len(rows)} entries from database")
            
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
    
    def _remove_from_database(self, entry_id: str):
        """Remove entry from database."""
        if not self.db:
            return
        
        try:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM dead_letter_entries WHERE entry_id = ?", (entry_id,))
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to remove entry: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring and cleanup."""
        while self.monitoring_active:
            try:
                time.sleep(300)  # Every 5 minutes
                
                with self.lock:
                    # Clean old entries
                    self._cleanup_old_entries()
                    
                    # Analyze patterns
                    self._analyze_failure_patterns()
                    
                    # Attempt batch recovery
                    self._batch_recovery_attempt()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _cleanup_old_entries(self):
        """Remove old entries."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        removed = 0
        
        for entry in list(self.queue):
            if entry.timestamp < cutoff:
                self.queue.remove(entry)
                del self.entry_index[entry.checksum]
                self._remove_from_database(entry.entry_id)
                removed += 1
        
        if removed > 0:
            self.stats['purged_entries'] += removed
            logger.info(f"Purged {removed} old entries")
    
    def _analyze_failure_patterns(self):
        """Analyze failure patterns for insights."""
        if not self.failure_patterns:
            return
        
        total = sum(self.failure_patterns.values())
        
        for reason, count in self.failure_patterns.items():
            percentage = (count / total) * 100
            
            if percentage > 30:
                logger.warning(
                    f"High failure rate for {reason.value}: "
                    f"{count} ({percentage:.1f}%)"
                )
    
    def _batch_recovery_attempt(self):
        """Attempt batch recovery of retryable entries."""
        retryable = [e for e in self.queue if e.can_retry]
        
        if retryable:
            # Group by failure reason
            by_reason = defaultdict(list)
            for entry in retryable:
                by_reason[entry.failure_reason].append(entry)
            
            # Attempt recovery by group
            for reason, entries in by_reason.items():
                if reason in self.recovery_strategies:
                    logger.info(f"Attempting batch recovery for {reason.value}: {len(entries)} entries")
                    
                    for entry in entries[:10]:  # Limit batch size
                        self._attempt_recovery(entry)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dead letter queue statistics."""
        with self.lock:
            # Calculate failure rates
            failure_rates = {}
            total = self.stats['total_entries']
            
            if total > 0:
                for reason, count in self.stats['failure_by_reason'].items():
                    failure_rates[reason] = (count / total) * 100
            
            return {
                'queue_size': len(self.queue),
                'total_entries': self.stats['total_entries'],
                'recovered_entries': self.stats['recovered_entries'],
                'purged_entries': self.stats['purged_entries'],
                'reprocessed_entries': self.stats['reprocessed_entries'],
                'failure_by_reason': dict(self.stats['failure_by_reason']),
                'failure_rates_percent': failure_rates,
                'retryable_entries': sum(1 for e in self.queue if e.can_retry),
                'oldest_entry': min(
                    (e.timestamp for e in self.queue),
                    default=None
                ),
                'most_common_failure': max(
                    self.failure_patterns.items(),
                    key=lambda x: x[1],
                    default=(None, 0)
                )[0].value if self.failure_patterns else None
            }
    
    def export_entries(self, 
                      format: str = 'json',
                      filter_func=None) -> str:
        """Export dead letter entries."""
        with self.lock:
            entries = []
            
            for entry in self.queue:
                if filter_func and not filter_func(entry):
                    continue
                
                entries.append({
                    'entry_id': entry.entry_id,
                    'timestamp': entry.timestamp.isoformat(),
                    'failure_reason': entry.failure_reason.value,
                    'error_details': entry.error_details,
                    'retry_count': entry.retry_count,
                    'endpoint': entry.endpoint,
                    'can_retry': entry.can_retry
                })
            
            if format == 'json':
                return json.dumps(entries, indent=2)
            else:
                # CSV format
                lines = ['entry_id,timestamp,reason,retries,endpoint,can_retry']
                for e in entries:
                    lines.append(
                        f"{e['entry_id']},{e['timestamp']},{e['failure_reason']},"
                        f"{e['retry_count']},{e['endpoint']},{e['can_retry']}"
                    )
                return '\n'.join(lines)
    
    def shutdown(self):
        """Shutdown dead letter queue."""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        if self.db:
            self.db.close()
        
        logger.info(f"Dead Letter Queue shutdown - Stats: {self.stats}")

# Global dead letter queue instance
dead_letter_queue = AnalyticsDeadLetterQueue()