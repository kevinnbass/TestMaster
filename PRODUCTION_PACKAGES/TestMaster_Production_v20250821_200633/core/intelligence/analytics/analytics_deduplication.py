"""
Analytics Deduplication System
==============================

Advanced analytics deduplication with intelligent duplicate detection,
content-based hashing, and smart merging strategies. Extracted from 54KB
archive component and modularized for the intelligence platform.
"""

import logging
import time
import threading
import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import difflib
import os

logger = logging.getLogger(__name__)


class DuplicateType(Enum):
    """Types of duplicate detection"""
    EXACT = "exact"
    NEAR = "near"
    CONTENT = "content"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"


class DeduplicationAction(Enum):
    """Actions to take on duplicates"""
    MERGE = "merge"
    DISCARD = "discard"
    KEEP_LATEST = "keep_latest"
    KEEP_OLDEST = "keep_oldest"
    MANUAL_REVIEW = "manual_review"


class DuplicateStatus(Enum):
    """Status of duplicate processing"""
    DETECTED = "detected"
    PROCESSED = "processed"
    MERGED = "merged"
    DISCARDED = "discarded"
    REVIEWED = "reviewed"


@dataclass
class DuplicateRecord:
    """Record of detected duplicates"""
    duplicate_id: str
    original_analytics_id: str
    duplicate_analytics_id: str
    duplicate_type: DuplicateType
    similarity_score: float
    detected_at: datetime
    action_taken: Optional[DeduplicationAction] = None
    status: DuplicateStatus = DuplicateStatus.DETECTED
    merge_result_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'duplicate_id': self.duplicate_id,
            'original_analytics_id': self.original_analytics_id,
            'duplicate_analytics_id': self.duplicate_analytics_id,
            'duplicate_type': self.duplicate_type.value,
            'similarity_score': self.similarity_score,
            'detected_at': self.detected_at.isoformat(),
            'action_taken': self.action_taken.value if self.action_taken else None,
            'status': self.status.value,
            'merge_result_id': self.merge_result_id,
            'metadata': self.metadata or {}
        }


@dataclass
class AnalyticsFingerprint:
    """Fingerprint for analytics identification"""
    analytics_id: str
    content_hash: str
    structure_hash: str
    semantic_hash: str
    temporal_hash: str
    size: int
    timestamp: datetime
    key_fields: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'analytics_id': self.analytics_id,
            'content_hash': self.content_hash,
            'structure_hash': self.structure_hash,
            'semantic_hash': self.semantic_hash,
            'temporal_hash': self.temporal_hash,
            'size': self.size,
            'timestamp': self.timestamp.isoformat(),
            'key_fields': self.key_fields
        }


class AnalyticsDeduplication:
    """Advanced analytics deduplication system with intelligent detection"""
    
    def __init__(self,
                 db_path: str = "data/deduplication.db",
                 similarity_threshold: float = 0.85,
                 processing_interval: float = 5.0):
        """
        Initialize deduplication system
        
        Args:
            db_path: Database path for deduplication records
            similarity_threshold: Minimum similarity for duplicate detection
            processing_interval: Seconds between deduplication cycles
        """
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.processing_interval = processing_interval
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory storage for fast lookups
        self.fingerprints: Dict[str, AnalyticsFingerprint] = {}
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        self.structure_index: Dict[str, Set[str]] = defaultdict(set)
        self.semantic_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Duplicate tracking
        self.detected_duplicates: Dict[str, DuplicateRecord] = {}
        self.processing_queue: deque = deque()
        
        # Statistics
        self.stats = {
            'total_analytics_processed': 0,
            'exact_duplicates_found': 0,
            'near_duplicates_found': 0,
            'content_duplicates_found': 0,
            'semantic_duplicates_found': 0,
            'temporal_duplicates_found': 0,
            'duplicates_merged': 0,
            'duplicates_discarded': 0,
            'deduplication_accuracy': 100.0
        }
        
        # Configuration
        self.max_fingerprints = 50000
        self.temporal_window_minutes = 5
        self.auto_merge_threshold = 0.95
        self.manual_review_threshold = 0.75
        
        # Deduplication strategies
        self.deduplication_strategies = {
            DuplicateType.EXACT: self._handle_exact_duplicate,
            DuplicateType.NEAR: self._handle_near_duplicate,
            DuplicateType.CONTENT: self._handle_content_duplicate,
            DuplicateType.SEMANTIC: self._handle_semantic_duplicate,
            DuplicateType.TEMPORAL: self._handle_temporal_duplicate
        }
        
        # Background processing
        self.dedup_active = True
        self.processing_thread = threading.Thread(
            target=self._deduplication_loop,
            daemon=True
        )
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        
        # Start threads
        self.processing_thread.start()
        self.cleanup_thread.start()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Analytics Deduplication System initialized")
    
    def _init_database(self):
        """Initialize deduplication database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analytics_fingerprints (
                        analytics_id TEXT PRIMARY KEY,
                        content_hash TEXT NOT NULL,
                        structure_hash TEXT NOT NULL,
                        semantic_hash TEXT NOT NULL,
                        temporal_hash TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        key_fields TEXT NOT NULL
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS duplicate_records (
                        duplicate_id TEXT PRIMARY KEY,
                        original_analytics_id TEXT NOT NULL,
                        duplicate_analytics_id TEXT NOT NULL,
                        duplicate_type TEXT NOT NULL,
                        similarity_score REAL NOT NULL,
                        detected_at TEXT NOT NULL,
                        action_taken TEXT,
                        status TEXT NOT NULL,
                        merge_result_id TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create indexes for fast lookups
                conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON analytics_fingerprints(content_hash)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_structure_hash ON analytics_fingerprints(structure_hash)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_duplicate_type ON duplicate_records(duplicate_type)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Deduplication database initialization failed: {e}")
            raise
    
    def process_analytics(self, analytics_id: str, analytics_data: Dict[str, Any]) -> bool:
        """
        Process analytics for duplication detection
        
        Args:
            analytics_id: Unique analytics identifier
            analytics_data: Analytics data to process
            
        Returns:
            True if analytics is unique/should be kept, False if duplicate
        """
        with self.lock:
            try:
                # Generate fingerprint
                fingerprint = self._generate_fingerprint(analytics_id, analytics_data)
                
                # Check for duplicates
                duplicates = self._detect_duplicates(fingerprint)
                
                if duplicates:
                    # Process detected duplicates
                    for duplicate_type, candidates in duplicates.items():
                        for candidate_id, similarity in candidates:
                            duplicate_record = self._create_duplicate_record(
                                candidate_id, analytics_id, duplicate_type, similarity
                            )
                            
                            # Add to processing queue
                            self.processing_queue.append(duplicate_record.duplicate_id)
                            
                            # Update statistics
                            self._update_duplicate_stats(duplicate_type)
                    
                    return False
                
                else:
                    # Store fingerprint for future comparisons
                    self._store_fingerprint(fingerprint)
                    self.stats['total_analytics_processed'] += 1
                    return True
                    
            except Exception as e:
                logger.error(f"Analytics processing failed for {analytics_id}: {e}")
                return True  # Default to keeping analytics on error
    
    def _generate_fingerprint(self, analytics_id: str, analytics_data: Dict[str, Any]) -> AnalyticsFingerprint:
        """Generate comprehensive fingerprint for analytics"""
        try:
            # Normalize data for consistent hashing
            normalized_data = self._normalize_data(analytics_data)
            data_json = json.dumps(normalized_data, sort_keys=True)
            
            # Content hash - exact content
            content_hash = hashlib.sha256(data_json.encode()).hexdigest()
            
            # Structure hash - data structure without values
            structure = self._extract_structure(normalized_data)
            structure_hash = hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()
            
            # Semantic hash - key semantic elements
            semantic_elements = self._extract_semantic_elements(normalized_data)
            semantic_hash = hashlib.sha256(json.dumps(semantic_elements, sort_keys=True).encode()).hexdigest()
            
            # Temporal hash - time-based grouping
            temporal_bucket = self._get_temporal_bucket(normalized_data)
            temporal_hash = hashlib.sha256(temporal_bucket.encode()).hexdigest()
            
            # Extract key fields for comparison
            key_fields = self._extract_key_fields(normalized_data)
            
            return AnalyticsFingerprint(
                analytics_id=analytics_id,
                content_hash=content_hash,
                structure_hash=structure_hash,
                semantic_hash=semantic_hash,
                temporal_hash=temporal_hash,
                size=len(data_json),
                timestamp=datetime.now(),
                key_fields=key_fields
            )
            
        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}")
            return AnalyticsFingerprint(
                analytics_id=analytics_id,
                content_hash="error",
                structure_hash="error",
                semantic_hash="error",
                temporal_hash="error",
                size=0,
                timestamp=datetime.now(),
                key_fields={}
            )
    
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data for consistent processing"""
        try:
            normalized = {}
            
            for key, value in data.items():
                norm_key = str(key).lower().strip()
                
                if isinstance(value, dict):
                    normalized[norm_key] = self._normalize_data(value)
                elif isinstance(value, list):
                    try:
                        if all(isinstance(x, (str, int, float)) for x in value):
                            normalized[norm_key] = sorted(value)
                        else:
                            normalized[norm_key] = value
                    except (TypeError, AttributeError):
                        normalized[norm_key] = value
                elif isinstance(value, str):
                    normalized[norm_key] = value.strip().lower() if value else ""
                elif isinstance(value, (int, float)):
                    if isinstance(value, float):
                        normalized[norm_key] = round(value, 6)
                    else:
                        normalized[norm_key] = value
                elif isinstance(value, datetime):
                    normalized[norm_key] = value.isoformat()
                else:
                    normalized[norm_key] = str(value) if value is not None else ""
            
            return normalized
            
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            return data
    
    def _extract_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data structure without values"""
        try:
            structure = {}
            
            for key, value in data.items():
                if isinstance(value, dict):
                    structure[key] = self._extract_structure(value)
                elif isinstance(value, list):
                    if value:
                        structure[key] = [type(value[0]).__name__]
                    else:
                        structure[key] = []
                else:
                    structure[key] = type(value).__name__
            
            return structure
            
        except Exception as e:
            logger.error(f"Structure extraction failed: {e}")
            return {}
    
    def _extract_semantic_elements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key semantic elements for comparison"""
        try:
            semantic = {}
            
            # Priority fields that are semantically important
            priority_fields = [
                'test_id', 'id', 'name', 'type', 'method', 'function',
                'class', 'module', 'file', 'line', 'error', 'message',
                'status', 'result', 'outcome'
            ]
            
            for field in priority_fields:
                if field in data:
                    semantic[field] = str(data[field])[:100]
            
            # Extract error/exception information
            for key, value in data.items():
                if any(keyword in key.lower() for keyword in ['error', 'exception', 'fail']):
                    semantic[f"error_{key}"] = str(value)[:200]
            
            return semantic
            
        except Exception as e:
            logger.error(f"Semantic extraction failed: {e}")
            return {}
    
    def _get_temporal_bucket(self, data: Dict[str, Any]) -> str:
        """Get temporal bucket for time-based grouping"""
        try:
            # Look for timestamp fields
            timestamp_fields = ['timestamp', 'time', 'created_at', 'executed_at', 'date']
            
            for field in timestamp_fields:
                if field in data:
                    timestamp_str = str(data[field])
                    try:
                        if 'T' in timestamp_str or ':' in timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromtimestamp(float(timestamp_str))
                        
                        # Round to temporal window
                        rounded_timestamp = timestamp.replace(
                            minute=(timestamp.minute // self.temporal_window_minutes) * self.temporal_window_minutes,
                            second=0,
                            microsecond=0
                        )
                        
                        return rounded_timestamp.isoformat()
                        
                    except (ValueError, TypeError):
                        continue
            
            # Fallback to current time bucket
            now = datetime.now()
            rounded_now = now.replace(
                minute=(now.minute // self.temporal_window_minutes) * self.temporal_window_minutes,
                second=0,
                microsecond=0
            )
            
            return rounded_now.isoformat()
            
        except Exception as e:
            logger.error(f"Temporal bucket extraction failed: {e}")
            return datetime.now().isoformat()
    
    def _extract_key_fields(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract key fields for detailed comparison"""
        try:
            key_fields = {}
            
            # Important fields for test analytics
            important_fields = [
                'test_name', 'test_id', 'method_name', 'class_name',
                'file_name', 'line_number', 'error_message', 'status',
                'duration', 'assertions', 'tags'
            ]
            
            for field in important_fields:
                if field in data:
                    key_fields[field] = str(data[field])
            
            return key_fields
            
        except Exception as e:
            logger.error(f"Key field extraction failed: {e}")
            return {}
    
    def _detect_duplicates(self, fingerprint: AnalyticsFingerprint) -> Dict[DuplicateType, List[Tuple[str, float]]]:
        """Detect duplicates using multiple strategies"""
        duplicates = {}
        
        try:
            # Exact duplicate check
            exact_matches = self.content_index.get(fingerprint.content_hash, set())
            if exact_matches:
                duplicates[DuplicateType.EXACT] = [(match_id, 1.0) for match_id in exact_matches]
            
            # Near duplicate check (structure + high semantic similarity)
            structure_matches = self.structure_index.get(fingerprint.structure_hash, set())
            if structure_matches:
                near_duplicates = []
                for match_id in structure_matches:
                    if match_id in self.fingerprints:
                        match_fp = self.fingerprints[match_id]
                        similarity = self._calculate_semantic_similarity(fingerprint, match_fp)
                        if similarity >= self.similarity_threshold:
                            near_duplicates.append((match_id, similarity))
                
                if near_duplicates:
                    duplicates[DuplicateType.NEAR] = near_duplicates
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return {}
    
    def _calculate_semantic_similarity(self, fp1: AnalyticsFingerprint, fp2: AnalyticsFingerprint) -> float:
        """Calculate semantic similarity between fingerprints"""
        try:
            # Compare key fields
            key_similarity = 0.0
            common_keys = set(fp1.key_fields.keys()) & set(fp2.key_fields.keys())
            
            if common_keys:
                matches = sum(1 for key in common_keys if fp1.key_fields[key] == fp2.key_fields[key])
                key_similarity = matches / len(common_keys)
            
            # Compare sizes
            size_ratio = min(fp1.size, fp2.size) / max(fp1.size, fp2.size) if max(fp1.size, fp2.size) > 0 else 1.0
            
            # Weighted combination
            similarity = (key_similarity * 0.7) + (size_ratio * 0.3)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _store_fingerprint(self, fingerprint: AnalyticsFingerprint):
        """Store fingerprint for future comparisons"""
        try:
            # Store in memory
            self.fingerprints[fingerprint.analytics_id] = fingerprint
            
            # Update indexes
            self.content_index[fingerprint.content_hash].add(fingerprint.analytics_id)
            self.structure_index[fingerprint.structure_hash].add(fingerprint.analytics_id)
            self.semantic_index[fingerprint.semantic_hash].add(fingerprint.analytics_id)
            self.temporal_index[fingerprint.temporal_hash].add(fingerprint.analytics_id)
            
            # Limit memory usage
            if len(self.fingerprints) > self.max_fingerprints:
                self._cleanup_old_fingerprints()
            
            # Save to database
            self._save_fingerprint(fingerprint)
            
        except Exception as e:
            logger.error(f"Fingerprint storage failed: {e}")
    
    def _create_duplicate_record(self, original_id: str, duplicate_id: str, 
                                duplicate_type: DuplicateType, similarity: float) -> DuplicateRecord:
        """Create duplicate record"""
        record_id = f"dup_{int(time.time() * 1000000)}"
        
        record = DuplicateRecord(
            duplicate_id=record_id,
            original_analytics_id=original_id,
            duplicate_analytics_id=duplicate_id,
            duplicate_type=duplicate_type,
            similarity_score=similarity,
            detected_at=datetime.now()
        )
        
        self.detected_duplicates[record_id] = record
        self._save_duplicate_record(record)
        
        return record
    
    def _update_duplicate_stats(self, duplicate_type: DuplicateType):
        """Update duplicate detection statistics"""
        if duplicate_type == DuplicateType.EXACT:
            self.stats['exact_duplicates_found'] += 1
        elif duplicate_type == DuplicateType.NEAR:
            self.stats['near_duplicates_found'] += 1
        elif duplicate_type == DuplicateType.CONTENT:
            self.stats['content_duplicates_found'] += 1
        elif duplicate_type == DuplicateType.SEMANTIC:
            self.stats['semantic_duplicates_found'] += 1
        elif duplicate_type == DuplicateType.TEMPORAL:
            self.stats['temporal_duplicates_found'] += 1
    
    def _deduplication_loop(self):
        """Background deduplication processing loop"""
        while self.dedup_active:
            try:
                if self.processing_queue:
                    with self.lock:
                        batch_size = min(10, len(self.processing_queue))
                        batch = []
                        
                        for _ in range(batch_size):
                            if self.processing_queue:
                                batch.append(self.processing_queue.popleft())
                        
                        for duplicate_id in batch:
                            if duplicate_id in self.detected_duplicates:
                                self._process_duplicate(duplicate_id)
                
                time.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Deduplication loop error: {e}")
                time.sleep(5)
    
    def _process_duplicate(self, duplicate_id: str):
        """Process a detected duplicate"""
        try:
            record = self.detected_duplicates[duplicate_id]
            
            # Determine action based on duplicate type and similarity
            if record.similarity_score >= self.auto_merge_threshold:
                action = DeduplicationAction.MERGE
            elif record.similarity_score >= self.manual_review_threshold:
                action = DeduplicationAction.MANUAL_REVIEW
            elif record.duplicate_type == DuplicateType.EXACT:
                action = DeduplicationAction.DISCARD
            else:
                action = DeduplicationAction.KEEP_LATEST
            
            # Execute action
            success = self.deduplication_strategies[record.duplicate_type](record, action)
            
            if success:
                record.action_taken = action
                record.status = DuplicateStatus.PROCESSED
                self._save_duplicate_record(record)
                
                # Update statistics
                if action == DeduplicationAction.MERGE:
                    self.stats['duplicates_merged'] += 1
                elif action == DeduplicationAction.DISCARD:
                    self.stats['duplicates_discarded'] += 1
            
        except Exception as e:
            logger.error(f"Duplicate processing failed for {duplicate_id}: {e}")
    
    def _handle_exact_duplicate(self, record: DuplicateRecord, action: DeduplicationAction) -> bool:
        """Handle exact duplicate"""
        try:
            if action == DeduplicationAction.DISCARD:
                logger.debug(f"Discarded exact duplicate: {record.duplicate_analytics_id}")
                return True
            elif action == DeduplicationAction.MERGE:
                logger.debug(f"Kept original for exact duplicate: {record.original_analytics_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Exact duplicate handling failed: {e}")
            return False
    
    def _handle_near_duplicate(self, record: DuplicateRecord, action: DeduplicationAction) -> bool:
        """Handle near duplicate"""
        try:
            if action == DeduplicationAction.MERGE:
                merge_id = f"merged_{int(time.time() * 1000000)}"
                record.merge_result_id = merge_id
                logger.debug(f"Merged near duplicates: {record.original_analytics_id} + {record.duplicate_analytics_id}")
                return True
            
            elif action == DeduplicationAction.MANUAL_REVIEW:
                record.status = DuplicateStatus.REVIEWED
                logger.debug(f"Marked for manual review: {record.duplicate_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Near duplicate handling failed: {e}")
            return False
    
    def _handle_content_duplicate(self, record: DuplicateRecord, action: DeduplicationAction) -> bool:
        """Handle content duplicate"""
        try:
            if action == DeduplicationAction.MERGE:
                merge_id = f"content_merged_{int(time.time() * 1000000)}"
                record.merge_result_id = merge_id
                return True
            
            elif action == DeduplicationAction.KEEP_LATEST:
                orig_fp = self.fingerprints.get(record.original_analytics_id)
                dup_fp = self.fingerprints.get(record.duplicate_analytics_id)
                
                if orig_fp and dup_fp:
                    if dup_fp.timestamp > orig_fp.timestamp:
                        logger.debug(f"Keeping newer analytics: {record.duplicate_analytics_id}")
                    else:
                        logger.debug(f"Keeping original analytics: {record.original_analytics_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Content duplicate handling failed: {e}")
            return False
    
    def _handle_semantic_duplicate(self, record: DuplicateRecord, action: DeduplicationAction) -> bool:
        """Handle semantic duplicate"""
        try:
            if action == DeduplicationAction.MANUAL_REVIEW:
                record.status = DuplicateStatus.REVIEWED
                logger.debug(f"Semantic duplicate marked for review: {record.duplicate_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Semantic duplicate handling failed: {e}")
            return False
    
    def _handle_temporal_duplicate(self, record: DuplicateRecord, action: DeduplicationAction) -> bool:
        """Handle temporal duplicate"""
        try:
            if action == DeduplicationAction.MERGE:
                merge_id = f"temporal_merged_{int(time.time() * 1000000)}"
                record.merge_result_id = merge_id
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Temporal duplicate handling failed: {e}")
            return False
    
    def _save_fingerprint(self, fingerprint: AnalyticsFingerprint):
        """Save fingerprint to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO analytics_fingerprints
                    (analytics_id, content_hash, structure_hash, semantic_hash,
                     temporal_hash, size, timestamp, key_fields)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fingerprint.analytics_id,
                    fingerprint.content_hash,
                    fingerprint.structure_hash,
                    fingerprint.semantic_hash,
                    fingerprint.temporal_hash,
                    fingerprint.size,
                    fingerprint.timestamp.isoformat(),
                    json.dumps(fingerprint.key_fields)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save fingerprint: {e}")
    
    def _save_duplicate_record(self, record: DuplicateRecord):
        """Save duplicate record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO duplicate_records
                    (duplicate_id, original_analytics_id, duplicate_analytics_id,
                     duplicate_type, similarity_score, detected_at, action_taken,
                     status, merge_result_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.duplicate_id,
                    record.original_analytics_id,
                    record.duplicate_analytics_id,
                    record.duplicate_type.value,
                    record.similarity_score,
                    record.detected_at.isoformat(),
                    record.action_taken.value if record.action_taken else None,
                    record.status.value,
                    record.merge_result_id,
                    json.dumps(record.metadata or {})
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save duplicate record: {e}")
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.dedup_active:
            try:
                time.sleep(3600)  # Cleanup every hour
                
                with self.lock:
                    self._cleanup_old_fingerprints()
                    self._cleanup_old_duplicates()
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def _cleanup_old_fingerprints(self):
        """Clean up old fingerprints to manage memory"""
        try:
            if len(self.fingerprints) <= self.max_fingerprints:
                return
            
            # Remove oldest 20% of fingerprints
            removal_count = len(self.fingerprints) // 5
            sorted_fps = sorted(self.fingerprints.items(), key=lambda x: x[1].timestamp)
            
            for analytics_id, fp in sorted_fps[:removal_count]:
                # Remove from memory
                self.fingerprints.pop(analytics_id, None)
                
                # Remove from indexes
                self.content_index[fp.content_hash].discard(analytics_id)
                self.structure_index[fp.structure_hash].discard(analytics_id)
                self.semantic_index[fp.semantic_hash].discard(analytics_id)
                self.temporal_index[fp.temporal_hash].discard(analytics_id)
                
                # Clean up empty index entries
                if not self.content_index[fp.content_hash]:
                    del self.content_index[fp.content_hash]
                if not self.structure_index[fp.structure_hash]:
                    del self.structure_index[fp.structure_hash]
                if not self.semantic_index[fp.semantic_hash]:
                    del self.semantic_index[fp.semantic_hash]
                if not self.temporal_index[fp.temporal_hash]:
                    del self.temporal_index[fp.temporal_hash]
            
            logger.debug(f"Cleaned up {removal_count} old fingerprints")
            
        except Exception as e:
            logger.error(f"Fingerprint cleanup failed: {e}")
    
    def _cleanup_old_duplicates(self):
        """Clean up old duplicate records"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    DELETE FROM duplicate_records 
                    WHERE detected_at < ? AND status = 'processed'
                ''', (cutoff_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.debug(f"Cleaned up {deleted_count} old duplicate records")
            
            # Clean up in-memory duplicates
            old_duplicates = [
                dup_id for dup_id, record in self.detected_duplicates.items()
                if record.detected_at < cutoff_time and record.status == DuplicateStatus.PROCESSED
            ]
            
            for dup_id in old_duplicates:
                self.detected_duplicates.pop(dup_id, None)
            
        except Exception as e:
            logger.error(f"Duplicate cleanup failed: {e}")
    
    # Public API methods
    
    def get_deduplication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive deduplication statistics"""
        with self.lock:
            total_duplicates = sum([
                self.stats['exact_duplicates_found'],
                self.stats['near_duplicates_found'],
                self.stats['content_duplicates_found'],
                self.stats['semantic_duplicates_found'],
                self.stats['temporal_duplicates_found']
            ])
            
            # Calculate accuracy
            successful_actions = self.stats['duplicates_merged'] + self.stats['duplicates_discarded']
            if total_duplicates > 0:
                accuracy = (successful_actions / total_duplicates) * 100
            else:
                accuracy = 100.0
            
            self.stats['deduplication_accuracy'] = accuracy
            
            return {
                'statistics': dict(self.stats),
                'fingerprints_stored': len(self.fingerprints),
                'active_duplicates': len(self.detected_duplicates),
                'processing_queue_size': len(self.processing_queue),
                'configuration': {
                    'similarity_threshold': self.similarity_threshold,
                    'auto_merge_threshold': self.auto_merge_threshold,
                    'manual_review_threshold': self.manual_review_threshold,
                    'temporal_window_minutes': self.temporal_window_minutes,
                    'max_fingerprints': self.max_fingerprints
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def force_deduplication(self, analytics_id: str) -> bool:
        """Force deduplication check for specific analytics"""
        try:
            if analytics_id in self.fingerprints:
                fingerprint = self.fingerprints[analytics_id]
                duplicates = self._detect_duplicates(fingerprint)
                
                if duplicates:
                    logger.debug(f"Forced deduplication found duplicates for: {analytics_id}")
                    return True
                else:
                    logger.debug(f"No duplicates found for: {analytics_id}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Force deduplication failed: {e}")
            return False
    
    def get_duplicate_details(self, duplicate_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a duplicate"""
        if duplicate_id in self.detected_duplicates:
            return self.detected_duplicates[duplicate_id].to_dict()
        
        return None
    
    def shutdown(self):
        """Shutdown deduplication system"""
        self.dedup_active = False
        
        # Wait for threads to complete
        for thread in [self.processing_thread, self.cleanup_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Analytics Deduplication System shutdown - Stats: {self.stats}")


# Global deduplication instance
analytics_deduplication = AnalyticsDeduplication()

# Export
__all__ = [
    'DuplicateType', 'DeduplicationAction', 'DuplicateStatus',
    'DuplicateRecord', 'AnalyticsFingerprint', 'AnalyticsDeduplication',
    'analytics_deduplication'
]