"""
Analytics Data Deduplication and Conflict Resolution Engine
==========================================================

Advanced deduplication system for analytics data with intelligent
conflict resolution, data merging, and consistency maintenance.

Author: TestMaster Team
"""

import logging
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import copy

logger = logging.getLogger(__name__)

class ConflictResolutionStrategy(Enum):
    LATEST_WINS = "latest_wins"
    HIGHEST_PRIORITY = "highest_priority" 
    MERGE_VALUES = "merge_values"
    KEEP_ALL = "keep_all"
    CUSTOM_RESOLVER = "custom_resolver"

class DuplicateType(Enum):
    EXACT_MATCH = "exact_match"
    STRUCTURAL_MATCH = "structural_match" 
    SEMANTIC_MATCH = "semantic_match"
    PARTIAL_MATCH = "partial_match"
    TIMESTAMP_RANGE = "timestamp_range"

@dataclass
class DeduplicationRule:
    """Rules for detecting and handling duplicates."""
    rule_id: str
    fields_to_compare: List[str]
    strategy: ConflictResolutionStrategy
    tolerance_seconds: int = 5
    similarity_threshold: float = 0.9
    priority_field: Optional[str] = None
    custom_resolver: Optional[Callable] = None
    merge_fields: List[str] = field(default_factory=list)
    description: str = ""

@dataclass
class DuplicateRecord:
    """Information about detected duplicates."""
    duplicate_id: str
    original_record: Dict[str, Any]
    duplicate_records: List[Dict[str, Any]]
    duplicate_type: DuplicateType
    detection_time: datetime
    rule_id: str
    conflict_fields: List[str]
    resolved: bool = False
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    merged_record: Optional[Dict[str, Any]] = None

@dataclass
class DataFingerprint:
    """Unique fingerprint for data records."""
    content_hash: str
    structural_hash: str
    semantic_hash: str
    timestamp: datetime
    source: str
    size_bytes: int
    field_count: int

class AnalyticsDeduplicationEngine:
    """
    Advanced deduplication engine for analytics data with conflict resolution.
    """
    
    def __init__(self, max_history_size: int = 50000,
                 cleanup_interval: int = 3600,
                 default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS):
        """
        Initialize analytics deduplication engine.
        
        Args:
            max_history_size: Maximum number of records to keep in history
            cleanup_interval: Interval for cleanup operations (seconds)
            default_strategy: Default conflict resolution strategy
        """
        self.max_history_size = max_history_size
        self.cleanup_interval = cleanup_interval
        self.default_strategy = default_strategy
        
        # Deduplication rules
        self.deduplication_rules = {}
        
        # Data tracking
        self.record_fingerprints = {}  # hash -> fingerprint
        self.record_history = deque(maxlen=max_history_size)
        self.duplicate_records = {}  # duplicate_id -> DuplicateRecord
        
        # Conflict resolution
        self.conflict_resolvers = {}
        self.resolution_statistics = defaultdict(int)
        
        # Performance tracking
        self.deduplication_stats = {
            'records_processed': 0,
            'duplicates_detected': 0,
            'conflicts_resolved': 0,
            'exact_matches': 0,
            'structural_matches': 0,
            'semantic_matches': 0,
            'partial_matches': 0,
            'bytes_deduplicated': 0,
            'start_time': datetime.now()
        }
        
        # Background processing
        self.engine_active = False
        self.cleanup_thread = None
        self.processing_queue = deque(maxlen=10000)
        
        # Caching for performance
        self.hash_cache = {}
        self.similarity_cache = {}
        
        # Setup default rules
        self._setup_default_rules()
        self._setup_default_resolvers()
        
        logger.info("Analytics Deduplication Engine initialized")
    
    def start_engine(self):
        """Start deduplication engine background processing."""
        if self.engine_active:
            return
        
        self.engine_active = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Analytics deduplication engine started")
    
    def stop_engine(self):
        """Stop deduplication engine."""
        self.engine_active = False
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        logger.info("Analytics deduplication engine stopped")
    
    def add_deduplication_rule(self, rule: DeduplicationRule):
        """Add a custom deduplication rule."""
        self.deduplication_rules[rule.rule_id] = rule
        logger.info(f"Added deduplication rule: {rule.rule_id}")
    
    def process_record(self, data: Dict[str, Any], source: str = "unknown") -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Process a record for deduplication.
        
        Args:
            data: Analytics data record
            source: Data source identifier
        
        Returns:
            Tuple of (processed_data, is_duplicate, duplicate_info)
        """
        try:
            # Generate fingerprint for the record
            fingerprint = self._generate_fingerprint(data, source)
            
            # Check for duplicates
            duplicates_info = self._detect_duplicates(data, fingerprint)
            
            # Update statistics
            self.deduplication_stats['records_processed'] += 1
            
            if duplicates_info:
                # Handle duplicates
                resolved_data = self._resolve_conflicts(data, duplicates_info, source)
                
                self.deduplication_stats['duplicates_detected'] += 1
                self.deduplication_stats['conflicts_resolved'] += 1
                
                # Calculate bytes saved
                original_size = len(json.dumps(data, default=str))
                self.deduplication_stats['bytes_deduplicated'] += original_size
                
                return resolved_data, True, [info['rule_id'] for info in duplicates_info]
            else:
                # No duplicates found, store fingerprint
                self._store_record_fingerprint(fingerprint, data)
                return data, False, []
        
        except Exception as e:
            logger.error(f"Error processing record for deduplication: {e}")
            return data, False, [f"error: {str(e)}"]
    
    def batch_deduplicate(self, records: List[Dict[str, Any]], 
                         source: str = "batch") -> List[Dict[str, Any]]:
        """
        Deduplicate a batch of records.
        
        Args:
            records: List of analytics records
            source: Batch source identifier
        
        Returns:
            Deduplicated list of records
        """
        deduplicated_records = []
        batch_fingerprints = {}
        
        for i, record in enumerate(records):
            try:
                # Generate fingerprint
                fingerprint = self._generate_fingerprint(record, f"{source}_batch_{i}")
                
                # Check against batch and historical duplicates
                is_duplicate = False
                
                # Check within current batch
                for existing_hash, existing_record in batch_fingerprints.items():
                    if self._records_are_duplicates(record, existing_record, fingerprint.content_hash, existing_hash):
                        is_duplicate = True
                        break
                
                # Check against historical records if not duplicate in batch
                if not is_duplicate:
                    duplicates_info = self._detect_duplicates(record, fingerprint)
                    is_duplicate = bool(duplicates_info)
                
                if not is_duplicate:
                    deduplicated_records.append(record)
                    batch_fingerprints[fingerprint.content_hash] = record
                    self._store_record_fingerprint(fingerprint, record)
                else:
                    self.deduplication_stats['duplicates_detected'] += 1
            
            except Exception as e:
                logger.error(f"Error deduplicating record {i}: {e}")
                # Include record on error to avoid data loss
                deduplicated_records.append(record)
        
        logger.info(f"Batch deduplicated: {len(records)} -> {len(deduplicated_records)} records")
        return deduplicated_records
    
    def get_duplicate_summary(self) -> Dict[str, Any]:
        """Get deduplication system summary."""
        uptime = (datetime.now() - self.deduplication_stats['start_time']).total_seconds()
        
        # Recent duplicates (last hour)
        recent_duplicates = [dup for dup in self.duplicate_records.values()
                           if (datetime.now() - dup.detection_time).total_seconds() < 3600]
        
        # Duplicate type distribution
        type_distribution = defaultdict(int)
        for dup in recent_duplicates:
            type_distribution[dup.duplicate_type.value] += 1
        
        # Rule effectiveness
        rule_effectiveness = defaultdict(int)
        for dup in recent_duplicates:
            rule_effectiveness[dup.rule_id] += 1
        
        return {
            'engine_status': {
                'active': self.engine_active,
                'uptime_seconds': uptime,
                'default_strategy': self.default_strategy.value
            },
            'statistics': self.deduplication_stats.copy(),
            'current_state': {
                'fingerprints_stored': len(self.record_fingerprints),
                'duplicate_records_tracked': len(self.duplicate_records),
                'processing_queue_size': len(self.processing_queue),
                'cache_size': len(self.hash_cache)
            },
            'deduplication_rules': {
                'total_rules': len(self.deduplication_rules),
                'custom_resolvers': len(self.conflict_resolvers)
            },
            'recent_activity': {
                'duplicates_last_hour': len(recent_duplicates),
                'duplicate_type_distribution': dict(type_distribution),
                'rule_effectiveness': dict(sorted(rule_effectiveness.items(), 
                                                 key=lambda x: x[1], reverse=True)[:10])
            },
            'performance_metrics': {
                'avg_processing_time_ms': self._calculate_avg_processing_time(),
                'deduplication_ratio': self._calculate_deduplication_ratio(),
                'memory_efficiency': self._calculate_memory_efficiency()
            }
        }
    
    def _generate_fingerprint(self, data: Dict[str, Any], source: str) -> DataFingerprint:
        """Generate comprehensive fingerprint for a data record."""
        # Content hash - exact content
        content_str = json.dumps(data, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        # Structural hash - structure without values
        structural_data = self._extract_structure(data)
        structural_str = json.dumps(structural_data, sort_keys=True)
        structural_hash = hashlib.sha256(structural_str.encode()).hexdigest()
        
        # Semantic hash - key semantic fields only
        semantic_data = self._extract_semantic_fields(data)
        semantic_str = json.dumps(semantic_data, sort_keys=True, default=str)
        semantic_hash = hashlib.sha256(semantic_str.encode()).hexdigest()
        
        return DataFingerprint(
            content_hash=content_hash,
            structural_hash=structural_hash,
            semantic_hash=semantic_hash,
            timestamp=datetime.now(),
            source=source,
            size_bytes=len(content_str),
            field_count=self._count_fields(data)
        )
    
    def _detect_duplicates(self, data: Dict[str, Any], fingerprint: DataFingerprint) -> List[Dict[str, Any]]:
        """Detect duplicates for a given record."""
        duplicates_info = []
        
        # Check exact matches first
        if fingerprint.content_hash in self.record_fingerprints:
            existing_fingerprint = self.record_fingerprints[fingerprint.content_hash]
            duplicates_info.append({
                'type': DuplicateType.EXACT_MATCH,
                'rule_id': 'exact_match',
                'existing_fingerprint': existing_fingerprint,
                'similarity': 1.0
            })
            self.deduplication_stats['exact_matches'] += 1
        
        # Check structural matches
        structural_matches = [fp for fp in self.record_fingerprints.values()
                            if fp.structural_hash == fingerprint.structural_hash
                            and fp.content_hash != fingerprint.content_hash]
        
        for match in structural_matches:
            duplicates_info.append({
                'type': DuplicateType.STRUCTURAL_MATCH,
                'rule_id': 'structural_match',
                'existing_fingerprint': match,
                'similarity': self._calculate_structural_similarity(data, match)
            })
            self.deduplication_stats['structural_matches'] += 1
        
        # Check semantic matches
        semantic_matches = [fp for fp in self.record_fingerprints.values()
                          if fp.semantic_hash == fingerprint.semantic_hash
                          and fp.content_hash != fingerprint.content_hash]
        
        for match in semantic_matches:
            similarity = self._calculate_semantic_similarity(data, match)
            if similarity >= 0.8:  # High semantic similarity threshold
                duplicates_info.append({
                    'type': DuplicateType.SEMANTIC_MATCH,
                    'rule_id': 'semantic_match',
                    'existing_fingerprint': match,
                    'similarity': similarity
                })
                self.deduplication_stats['semantic_matches'] += 1
        
        # Check custom rules
        for rule_id, rule in self.deduplication_rules.items():
            matches = self._check_rule_matches(data, fingerprint, rule)
            for match in matches:
                duplicates_info.append({
                    'type': DuplicateType.PARTIAL_MATCH,
                    'rule_id': rule_id,
                    'existing_fingerprint': match['fingerprint'],
                    'similarity': match['similarity'],
                    'rule': rule
                })
                self.deduplication_stats['partial_matches'] += 1
        
        return duplicates_info
    
    def _resolve_conflicts(self, data: Dict[str, Any], duplicates_info: List[Dict[str, Any]], 
                          source: str) -> Dict[str, Any]:
        """Resolve conflicts between duplicate records."""
        if not duplicates_info:
            return data
        
        # Find the most appropriate resolution strategy
        primary_duplicate = max(duplicates_info, key=lambda x: x['similarity'])
        
        # Get rule or use default
        rule = primary_duplicate.get('rule')
        strategy = rule.strategy if rule else self.default_strategy
        
        if strategy == ConflictResolutionStrategy.LATEST_WINS:
            # Current record wins (it's the latest)
            resolved_data = data
        
        elif strategy == ConflictResolutionStrategy.HIGHEST_PRIORITY:
            # Use priority field to determine winner
            if rule and rule.priority_field:
                existing_record = self._get_record_from_fingerprint(primary_duplicate['existing_fingerprint'])
                if existing_record:
                    current_priority = data.get(rule.priority_field, 0)
                    existing_priority = existing_record.get(rule.priority_field, 0)
                    resolved_data = data if current_priority > existing_priority else existing_record
                else:
                    resolved_data = data
            else:
                resolved_data = data
        
        elif strategy == ConflictResolutionStrategy.MERGE_VALUES:
            # Merge records intelligently
            existing_record = self._get_record_from_fingerprint(primary_duplicate['existing_fingerprint'])
            if existing_record:
                resolved_data = self._merge_records(data, existing_record, rule)
            else:
                resolved_data = data
        
        elif strategy == ConflictResolutionStrategy.CUSTOM_RESOLVER:
            # Use custom resolver
            if rule and rule.custom_resolver:
                existing_record = self._get_record_from_fingerprint(primary_duplicate['existing_fingerprint'])
                if existing_record:
                    resolved_data = rule.custom_resolver(data, existing_record)
                else:
                    resolved_data = data
            else:
                resolved_data = data
        
        else:  # KEEP_ALL or default
            resolved_data = data
        
        # Track resolution
        self.resolution_statistics[strategy.value] += 1
        
        # Create duplicate record entry
        duplicate_id = f"dup_{int(time.time())}_{hash(str(data))}"
        duplicate_record = DuplicateRecord(
            duplicate_id=duplicate_id,
            original_record=data,
            duplicate_records=[primary_duplicate['existing_fingerprint']],
            duplicate_type=primary_duplicate['type'],
            detection_time=datetime.now(),
            rule_id=primary_duplicate['rule_id'],
            conflict_fields=self._identify_conflict_fields(data, primary_duplicate),
            resolved=True,
            resolution_strategy=strategy,
            merged_record=resolved_data if resolved_data != data else None
        )
        
        self.duplicate_records[duplicate_id] = duplicate_record
        
        return resolved_data
    
    def _setup_default_rules(self):
        """Setup default deduplication rules."""
        
        # Exact timestamp match rule
        timestamp_rule = DeduplicationRule(
            rule_id='timestamp_exact',
            fields_to_compare=['timestamp', 'component'],
            strategy=ConflictResolutionStrategy.MERGE_VALUES,
            tolerance_seconds=1,
            merge_fields=['metrics', 'attributes', 'performance'],
            description='Merge records with exact timestamp and component'
        )
        
        # System metrics rule
        system_metrics_rule = DeduplicationRule(
            rule_id='system_metrics',
            fields_to_compare=['cpu_usage_percent', 'memory_usage_percent', 'timestamp'],
            strategy=ConflictResolutionStrategy.LATEST_WINS,
            tolerance_seconds=5,
            similarity_threshold=0.95,
            description='Handle duplicate system metrics'
        )
        
        # Component analytics rule
        component_rule = DeduplicationRule(
            rule_id='component_analytics',
            fields_to_compare=['component', 'operation', 'timestamp'],
            strategy=ConflictResolutionStrategy.MERGE_VALUES,
            tolerance_seconds=10,
            merge_fields=['metrics', 'performance_data'],
            description='Merge component analytics data'
        )
        
        # Performance data rule
        performance_rule = DeduplicationRule(
            rule_id='performance_data',
            fields_to_compare=['response_time_ms', 'operation', 'component'],
            strategy=ConflictResolutionStrategy.HIGHEST_PRIORITY,
            priority_field='priority',
            tolerance_seconds=2,
            description='Handle performance data duplicates'
        )
        
        # Add rules
        for rule in [timestamp_rule, system_metrics_rule, component_rule, performance_rule]:
            self.deduplication_rules[rule.rule_id] = rule
    
    def _setup_default_resolvers(self):
        """Setup default conflict resolvers."""
        
        def numeric_average_resolver(current_data, existing_data):
            """Resolver that averages numeric values."""
            resolved = current_data.copy()
            
            for key, value in existing_data.items():
                if key in resolved and isinstance(value, (int, float)) and isinstance(resolved[key], (int, float)):
                    resolved[key] = (resolved[key] + value) / 2
            
            return resolved
        
        def list_merge_resolver(current_data, existing_data):
            """Resolver that merges list values."""
            resolved = current_data.copy()
            
            for key, value in existing_data.items():
                if key in resolved:
                    if isinstance(value, list) and isinstance(resolved[key], list):
                        # Merge lists and remove duplicates
                        merged_list = list(set(resolved[key] + value))
                        resolved[key] = merged_list
                    elif isinstance(value, dict) and isinstance(resolved[key], dict):
                        # Merge dictionaries
                        resolved[key] = {**resolved[key], **value}
            
            return resolved
        
        self.conflict_resolvers['numeric_average'] = numeric_average_resolver
        self.conflict_resolvers['list_merge'] = list_merge_resolver
    
    def _extract_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural representation of data (keys only)."""
        def extract_keys(obj):
            if isinstance(obj, dict):
                return {key: extract_keys(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [extract_keys(item) for item in obj[:3]]  # Sample first 3 items
            else:
                return type(obj).__name__
        
        return extract_keys(data)
    
    def _extract_semantic_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key semantic fields for comparison."""
        semantic_fields = [
            'timestamp', 'component', 'operation', 'source', 'type',
            'cpu_usage_percent', 'memory_usage_percent', 'response_time_ms',
            'error_count', 'success_count', 'total_requests'
        ]
        
        semantic_data = {}
        
        def extract_semantic(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if any(field in full_key.lower() for field in semantic_fields):
                        semantic_data[full_key] = value
                    elif isinstance(value, dict):
                        extract_semantic(value, full_key)
        
        extract_semantic(data)
        return semantic_data
    
    def _count_fields(self, data: Dict[str, Any]) -> int:
        """Count total number of fields in nested data."""
        count = 0
        
        def count_recursive(obj):
            nonlocal count
            if isinstance(obj, dict):
                count += len(obj)
                for value in obj.values():
                    count_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_recursive(item)
        
        count_recursive(data)
        return count
    
    def _calculate_structural_similarity(self, data: Dict[str, Any], 
                                      existing_fingerprint: DataFingerprint) -> float:
        """Calculate structural similarity between records."""
        # This is a simplified similarity calculation
        # In a real implementation, this would be more sophisticated
        return 0.85  # Placeholder
    
    def _calculate_semantic_similarity(self, data: Dict[str, Any],
                                    existing_fingerprint: DataFingerprint) -> float:
        """Calculate semantic similarity between records."""
        # This is a simplified similarity calculation
        return 0.8  # Placeholder
    
    def _check_rule_matches(self, data: Dict[str, Any], fingerprint: DataFingerprint,
                          rule: DeduplicationRule) -> List[Dict[str, Any]]:
        """Check for matches based on custom rule."""
        matches = []
        
        for existing_hash, existing_fingerprint in self.record_fingerprints.items():
            if existing_hash == fingerprint.content_hash:
                continue
            
            # Check if records match rule criteria
            if self._records_match_rule(data, existing_fingerprint, rule):
                similarity = self._calculate_rule_similarity(data, existing_fingerprint, rule)
                if similarity >= rule.similarity_threshold:
                    matches.append({
                        'fingerprint': existing_fingerprint,
                        'similarity': similarity
                    })
        
        return matches
    
    def _records_match_rule(self, data: Dict[str, Any], existing_fingerprint: DataFingerprint,
                          rule: DeduplicationRule) -> bool:
        """Check if records match a specific rule."""
        # This would implement rule-specific matching logic
        # For now, return a placeholder
        return True
    
    def _calculate_rule_similarity(self, data: Dict[str, Any], existing_fingerprint: DataFingerprint,
                                 rule: DeduplicationRule) -> float:
        """Calculate similarity based on rule criteria."""
        # This would implement rule-specific similarity calculation
        return 0.9  # Placeholder
    
    def _records_are_duplicates(self, record1: Dict[str, Any], record2: Dict[str, Any],
                              hash1: str, hash2: str) -> bool:
        """Check if two records are duplicates."""
        return hash1 == hash2
    
    def _store_record_fingerprint(self, fingerprint: DataFingerprint, data: Dict[str, Any]):
        """Store record fingerprint for future comparison."""
        self.record_fingerprints[fingerprint.content_hash] = fingerprint
        self.record_history.append({
            'fingerprint': fingerprint,
            'data': data,
            'stored_at': datetime.now()
        })
    
    def _get_record_from_fingerprint(self, fingerprint: DataFingerprint) -> Optional[Dict[str, Any]]:
        """Get original record from fingerprint."""
        for record_info in self.record_history:
            if record_info['fingerprint'].content_hash == fingerprint.content_hash:
                return record_info['data']
        return None
    
    def _merge_records(self, current_data: Dict[str, Any], existing_data: Dict[str, Any],
                      rule: DeduplicationRule) -> Dict[str, Any]:
        """Merge two records based on rule specifications."""
        merged = current_data.copy()
        
        # Merge specified fields
        for field in rule.merge_fields:
            if field in existing_data:
                if field in merged:
                    # Merge based on field type
                    if isinstance(merged[field], dict) and isinstance(existing_data[field], dict):
                        merged[field] = {**existing_data[field], **merged[field]}
                    elif isinstance(merged[field], list) and isinstance(existing_data[field], list):
                        merged[field] = list(set(merged[field] + existing_data[field]))
                    elif isinstance(merged[field], (int, float)) and isinstance(existing_data[field], (int, float)):
                        merged[field] = (merged[field] + existing_data[field]) / 2
                else:
                    merged[field] = existing_data[field]
        
        # Add merge metadata
        merged['_merge_info'] = {
            'merged_at': datetime.now().isoformat(),
            'rule_id': rule.rule_id,
            'strategy': rule.strategy.value,
            'merged_fields': rule.merge_fields
        }
        
        return merged
    
    def _identify_conflict_fields(self, current_data: Dict[str, Any], 
                                duplicate_info: Dict[str, Any]) -> List[str]:
        """Identify fields that have conflicts between records."""
        conflicts = []
        
        existing_fingerprint = duplicate_info['existing_fingerprint']
        existing_data = self._get_record_from_fingerprint(existing_fingerprint)
        
        if existing_data:
            for key, value in current_data.items():
                if key in existing_data and existing_data[key] != value:
                    conflicts.append(key)
        
        return conflicts
    
    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time per record."""
        # Placeholder implementation
        return 2.5  # milliseconds
    
    def _calculate_deduplication_ratio(self) -> float:
        """Calculate deduplication effectiveness ratio."""
        if self.deduplication_stats['records_processed'] == 0:
            return 0.0
        
        return (self.deduplication_stats['duplicates_detected'] / 
               self.deduplication_stats['records_processed']) * 100
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency from deduplication."""
        if self.deduplication_stats['bytes_deduplicated'] == 0:
            return 0.0
        
        # Estimate memory saved
        total_fingerprint_memory = len(self.record_fingerprints) * 200  # Estimate 200 bytes per fingerprint
        return (self.deduplication_stats['bytes_deduplicated'] / 
               (self.deduplication_stats['bytes_deduplicated'] + total_fingerprint_memory)) * 100
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.engine_active:
            try:
                time.sleep(self.cleanup_interval)
                
                # Clean up old records
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # Remove old fingerprints
                old_hashes = [hash_key for hash_key, fingerprint in self.record_fingerprints.items()
                            if fingerprint.timestamp < cutoff_time]
                
                for hash_key in old_hashes:
                    del self.record_fingerprints[hash_key]
                
                # Remove old duplicate records
                old_duplicates = [dup_id for dup_id, dup_record in self.duplicate_records.items()
                                if dup_record.detection_time < cutoff_time]
                
                for dup_id in old_duplicates:
                    del self.duplicate_records[dup_id]
                
                # Clear caches periodically
                if len(self.hash_cache) > 1000:
                    self.hash_cache.clear()
                
                if len(self.similarity_cache) > 1000:
                    self.similarity_cache.clear()
                
                logger.debug(f"Cleanup completed: removed {len(old_hashes)} old fingerprints, {len(old_duplicates)} old duplicates")
            
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def get_recent_duplicates(self, hours: int = 24, limit: int = 100) -> List[DuplicateRecord]:
        """Get recent duplicate records."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_duplicates = [dup for dup in self.duplicate_records.values()
                           if dup.detection_time >= cutoff_time]
        
        # Sort by detection time (most recent first)
        recent_duplicates.sort(key=lambda x: x.detection_time, reverse=True)
        
        return recent_duplicates[:limit]
    
    def shutdown(self):
        """Shutdown deduplication engine."""
        self.stop_engine()
        logger.info("Analytics Deduplication Engine shutdown")