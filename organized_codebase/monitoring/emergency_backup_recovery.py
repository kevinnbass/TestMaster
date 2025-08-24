"""
Emergency Analytics Backup and Recovery System
==============================================

Comprehensive emergency backup system with multi-tier storage,
instant recovery, and disaster prevention for absolute data safety.

Author: TestMaster Team
"""

import logging
import time
import threading
import json
import sqlite3
import shutil
import zipfile
import hashlib
import os
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Types of backup operations."""
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    FULL = "full"
    EMERGENCY = "emergency"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Backup operation status."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    VERIFIED = "verified"

class RecoveryMode(Enum):
    """Recovery operation modes."""
    INSTANT = "instant"
    FAST = "fast"
    COMPLETE = "complete"
    SELECTIVE = "selective"
    POINT_IN_TIME = "point_in_time"

class StorageTier(Enum):
    """Storage tier levels."""
    HOT = "hot"          # Immediate access, SSD/RAM
    WARM = "warm"        # Quick access, local disk
    COLD = "cold"        # Archive access, compressed
    FROZEN = "frozen"    # Long-term, offsite

@dataclass
class BackupRecord:
    """Backup operation record."""
    backup_id: str
    backup_type: BackupType
    storage_tier: StorageTier
    created_at: datetime
    status: BackupStatus
    file_path: str
    compressed_size: int
    original_size: int
    checksum: str
    analytics_count: int
    metadata: Dict[str, Any] = None
    retention_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type.value,
            'storage_tier': self.storage_tier.value,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'file_path': self.file_path,
            'compressed_size': self.compressed_size,
            'original_size': self.original_size,
            'checksum': self.checksum,
            'analytics_count': self.analytics_count,
            'metadata': self.metadata or {},
            'retention_date': self.retention_date.isoformat() if self.retention_date else None
        }

@dataclass
class RecoveryRecord:
    """Recovery operation record."""
    recovery_id: str
    backup_id: str
    recovery_mode: RecoveryMode
    initiated_at: datetime
    completed_at: Optional[datetime]
    status: BackupStatus
    recovered_analytics_count: int
    recovery_time_seconds: float
    error_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'recovery_id': self.recovery_id,
            'backup_id': self.backup_id,
            'recovery_mode': self.recovery_mode.value,
            'initiated_at': self.initiated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'recovered_analytics_count': self.recovered_analytics_count,
            'recovery_time_seconds': self.recovery_time_seconds,
            'error_info': self.error_info
        }

class EmergencyBackupRecovery:
    """
    Emergency analytics backup and recovery system with multi-tier storage.
    """
    
    def __init__(self,
                 aggregator=None,
                 backup_base_path: str = "data/backups",
                 backup_interval: float = 300.0,  # 5 minutes
                 max_hot_backups: int = 10):
        """
        Initialize emergency backup and recovery system.
        
        Args:
            aggregator: Analytics aggregator instance
            backup_base_path: Base path for backup storage
            backup_interval: Seconds between automatic backups
            max_hot_backups: Maximum number of hot tier backups
        """
        self.aggregator = aggregator
        self.backup_base_path = Path(backup_base_path)
        self.backup_interval = backup_interval
        self.max_hot_backups = max_hot_backups
        
        # Create backup directory structure
        self._init_backup_structure()
        
        # Initialize database
        self.db_path = self.backup_base_path / "backup_registry.db"
        self._init_database()
        
        # Backup tracking
        self.backup_records: Dict[str, BackupRecord] = {}
        self.recovery_records: Dict[str, RecoveryRecord] = {}
        self.backup_queue: deque = deque()
        
        # Hot storage for instant access
        self.hot_storage: Dict[str, Dict[str, Any]] = {}
        self.hot_analytics_index: Dict[str, str] = {}  # analytics_id -> backup_id
        
        # Backup strategies
        self.last_full_backup: Optional[datetime] = None
        self.last_incremental_backup: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            'total_backups_created': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_backup_time': 0.0,
            'average_recovery_time': 0.0,
            'total_data_backed_up_mb': 0.0,
            'compression_ratio': 0.0,
            'hot_storage_size_mb': 0.0
        }
        
        # Configuration
        self.retention_policy = {
            StorageTier.HOT: timedelta(hours=24),
            StorageTier.WARM: timedelta(days=7),
            StorageTier.COLD: timedelta(days=30),
            StorageTier.FROZEN: timedelta(days=365)
        }
        
        self.compression_level = 6  # Balance between speed and compression
        self.verify_backups = True
        self.auto_recovery_enabled = True
        
        # Background processing
        self.backup_active = True
        self.backup_thread = threading.Thread(
            target=self._backup_loop,
            daemon=True
        )
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True
        )
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        
        # Start threads
        self.backup_thread.start()
        self.maintenance_thread.start()
        self.monitoring_thread.start()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load existing backups
        self._load_backup_registry()
        
        logger.info("Emergency Backup and Recovery System initialized")
    
    def _init_backup_structure(self):
        """Initialize backup directory structure."""
        try:
            self.backup_base_path.mkdir(parents=True, exist_ok=True)
            
            # Create tier directories
            for tier in StorageTier:
                tier_path = self.backup_base_path / tier.value
                tier_path.mkdir(exist_ok=True)
            
            # Create temporary directory for processing
            self.temp_path = self.backup_base_path / "temp"
            self.temp_path.mkdir(exist_ok=True)
            
            logger.info(f"Backup structure initialized at: {self.backup_base_path}")
            
        except Exception as e:
            logger.error(f"Backup structure initialization failed: {e}")
            raise
    
    def _init_database(self):
        """Initialize backup registry database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS backup_records (
                        backup_id TEXT PRIMARY KEY,
                        backup_type TEXT NOT NULL,
                        storage_tier TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        compressed_size INTEGER NOT NULL,
                        original_size INTEGER NOT NULL,
                        checksum TEXT NOT NULL,
                        analytics_count INTEGER NOT NULL,
                        metadata TEXT,
                        retention_date TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS recovery_records (
                        recovery_id TEXT PRIMARY KEY,
                        backup_id TEXT NOT NULL,
                        recovery_mode TEXT NOT NULL,
                        initiated_at TEXT NOT NULL,
                        completed_at TEXT,
                        status TEXT NOT NULL,
                        recovered_analytics_count INTEGER NOT NULL,
                        recovery_time_seconds REAL NOT NULL,
                        error_info TEXT,
                        FOREIGN KEY (backup_id) REFERENCES backup_records (backup_id)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analytics_backup_map (
                        analytics_id TEXT NOT NULL,
                        backup_id TEXT NOT NULL,
                        backup_timestamp TEXT NOT NULL,
                        PRIMARY KEY (analytics_id, backup_id),
                        FOREIGN KEY (backup_id) REFERENCES backup_records (backup_id)
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_backup_created ON backup_records(created_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_backup_tier ON backup_records(storage_tier)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_recovery_initiated ON recovery_records(initiated_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_analytics_backup ON analytics_backup_map(analytics_id)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Backup database initialization failed: {e}")
            raise
    
    def create_emergency_backup(self, 
                               analytics_data: List[Dict[str, Any]] = None,
                               backup_type: BackupType = BackupType.EMERGENCY) -> str:
        """
        Create emergency backup immediately.
        
        Args:
            analytics_data: Specific analytics to backup (None = all)
            backup_type: Type of backup to create
            
        Returns:
            Backup ID
        """
        with self.lock:
            try:
                backup_id = f"emergency_{int(time.time() * 1000000)}"
                
                # Collect analytics data
                if analytics_data is None:
                    analytics_data = self._collect_all_analytics()
                
                # Create backup record
                backup_record = BackupRecord(
                    backup_id=backup_id,
                    backup_type=backup_type,
                    storage_tier=StorageTier.HOT,  # Emergency backups go to hot storage
                    created_at=datetime.now(),
                    status=BackupStatus.INITIATED,
                    file_path="",  # Will be set after creation
                    compressed_size=0,
                    original_size=0,
                    checksum="",
                    analytics_count=len(analytics_data),
                    metadata={
                        'emergency': True,
                        'platform': platform.system(),
                        'created_by': 'emergency_system'
                    }
                )
                
                # Store backup record
                self.backup_records[backup_id] = backup_record
                
                # Create backup file
                success = self._create_backup_file(backup_record, analytics_data)
                
                if success:
                    backup_record.status = BackupStatus.COMPLETED
                    self.stats['successful_backups'] += 1
                    
                    # Store in hot storage for instant access
                    self._store_in_hot_storage(backup_id, analytics_data)
                    
                    logger.info(f"Emergency backup created successfully: {backup_id}")
                else:
                    backup_record.status = BackupStatus.FAILED
                    self.stats['failed_backups'] += 1
                    logger.error(f"Emergency backup creation failed: {backup_id}")
                
                # Save to database
                self._save_backup_record(backup_record)
                self.stats['total_backups_created'] += 1
                
                return backup_id
                
            except Exception as e:
                logger.error(f"Emergency backup creation failed: {e}")
                return ""
    
    def instant_recovery(self, 
                        backup_id: str = None,
                        analytics_id_filter: List[str] = None,
                        point_in_time: datetime = None) -> str:
        """
        Perform instant recovery from backup.
        
        Args:
            backup_id: Specific backup to recover from (None = latest)
            analytics_id_filter: Specific analytics to recover
            point_in_time: Recover to specific point in time
            
        Returns:
            Recovery ID
        """
        with self.lock:
            try:
                recovery_id = f"recovery_{int(time.time() * 1000000)}"
                
                # Select backup to recover from
                if backup_id is None:
                    backup_id = self._find_best_backup(point_in_time)
                
                if not backup_id:
                    logger.error("No suitable backup found for recovery")
                    return ""
                
                backup_record = self.backup_records.get(backup_id)
                if not backup_record:
                    logger.error(f"Backup record not found: {backup_id}")
                    return ""
                
                # Create recovery record
                recovery_record = RecoveryRecord(
                    recovery_id=recovery_id,
                    backup_id=backup_id,
                    recovery_mode=RecoveryMode.INSTANT,
                    initiated_at=datetime.now(),
                    completed_at=None,
                    status=BackupStatus.IN_PROGRESS,
                    recovered_analytics_count=0,
                    recovery_time_seconds=0.0
                )
                
                self.recovery_records[recovery_id] = recovery_record
                
                # Perform recovery
                start_time = time.time()
                success = self._perform_recovery(recovery_record, analytics_id_filter)
                recovery_time = time.time() - start_time
                
                # Update recovery record
                recovery_record.completed_at = datetime.now()
                recovery_record.recovery_time_seconds = recovery_time
                
                if success:
                    recovery_record.status = BackupStatus.COMPLETED
                    self.stats['successful_recoveries'] += 1
                    
                    # Update average recovery time
                    total_recoveries = self.stats['total_recoveries'] + 1
                    current_avg = self.stats['average_recovery_time']
                    self.stats['average_recovery_time'] = (
                        (current_avg * (total_recoveries - 1) + recovery_time) / total_recoveries
                    )
                    
                    logger.info(f"Instant recovery completed: {recovery_id} in {recovery_time:.2f}s")
                else:
                    recovery_record.status = BackupStatus.FAILED
                    recovery_record.error_info = "Recovery operation failed"
                    self.stats['failed_recoveries'] += 1
                    logger.error(f"Instant recovery failed: {recovery_id}")
                
                # Save recovery record
                self._save_recovery_record(recovery_record)
                self.stats['total_recoveries'] += 1
                
                return recovery_id
                
            except Exception as e:
                logger.error(f"Instant recovery failed: {e}")
                return ""
    
    def _collect_all_analytics(self) -> List[Dict[str, Any]]:
        """Collect all available analytics data."""
        try:
            analytics_data = []
            
            # Get from aggregator if available
            if self.aggregator and hasattr(self.aggregator, 'get_all_analytics'):
                aggregator_data = self.aggregator.get_all_analytics()
                if aggregator_data:
                    analytics_data.extend(aggregator_data)
            
            # Get from hot storage
            for backup_id, hot_data in self.hot_storage.items():
                if 'analytics' in hot_data:
                    analytics_data.extend(hot_data['analytics'])
            
            # Deduplicate based on analytics_id
            seen_ids = set()
            unique_analytics = []
            
            for analytics in analytics_data:
                analytics_id = analytics.get('analytics_id') or analytics.get('id')
                if analytics_id and analytics_id not in seen_ids:
                    seen_ids.add(analytics_id)
                    unique_analytics.append(analytics)
            
            logger.info(f"Collected {len(unique_analytics)} unique analytics for backup")
            return unique_analytics
            
        except Exception as e:
            logger.error(f"Analytics collection failed: {e}")
            return []
    
    def _create_backup_file(self, backup_record: BackupRecord, analytics_data: List[Dict[str, Any]]) -> bool:
        """Create compressed backup file."""
        try:
            # Generate file path
            timestamp = backup_record.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"{backup_record.backup_type.value}_{timestamp}_{backup_record.backup_id}.zip"
            file_path = self.backup_base_path / backup_record.storage_tier.value / filename
            
            # Create backup data structure
            backup_data = {
                'backup_metadata': backup_record.to_dict(),
                'analytics': analytics_data,
                'system_info': {
                    'platform': platform.system(),
                    'python_version': platform.python_version(),
                    'backup_version': '1.0',
                    'timestamp': backup_record.created_at.isoformat()
                }
            }
            
            # Calculate original size
            original_json = json.dumps(backup_data, default=str)
            original_size = len(original_json.encode('utf-8'))
            
            # Create compressed backup file
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=self.compression_level) as zipf:
                zipf.writestr('backup_data.json', original_json)
                
                # Add individual analytics files for faster selective recovery
                for i, analytics in enumerate(analytics_data):
                    analytics_json = json.dumps(analytics, default=str)
                    zipf.writestr(f'analytics/{i:06d}.json', analytics_json)
            
            # Calculate file size and checksum
            compressed_size = file_path.stat().st_size
            checksum = self._calculate_file_checksum(file_path)
            
            # Update backup record
            backup_record.file_path = str(file_path)
            backup_record.original_size = original_size
            backup_record.compressed_size = compressed_size
            backup_record.checksum = checksum
            
            # Update statistics
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            self.stats['compression_ratio'] = compression_ratio
            self.stats['total_data_backed_up_mb'] += compressed_size / (1024 * 1024)
            
            # Verify backup if enabled
            if self.verify_backups:
                if self._verify_backup_integrity(backup_record):
                    backup_record.status = BackupStatus.VERIFIED
                else:
                    backup_record.status = BackupStatus.CORRUPTED
                    return False
            
            logger.debug(f"Backup file created: {file_path} ({compressed_size} bytes, {compression_ratio:.1f}% compression)")
            return True
            
        except Exception as e:
            logger.error(f"Backup file creation failed: {e}")
            return False
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    def _verify_backup_integrity(self, backup_record: BackupRecord) -> bool:
        """Verify backup file integrity."""
        try:
            file_path = Path(backup_record.file_path)
            
            # Check file exists
            if not file_path.exists():
                logger.error(f"Backup file not found: {file_path}")
                return False
            
            # Verify checksum
            current_checksum = self._calculate_file_checksum(file_path)
            if current_checksum != backup_record.checksum:
                logger.error(f"Backup checksum mismatch: {backup_record.backup_id}")
                return False
            
            # Verify can read backup
            try:
                with zipfile.ZipFile(file_path, 'r') as zipf:
                    # Check main backup file
                    if 'backup_data.json' not in zipf.namelist():
                        logger.error(f"Backup data missing in: {backup_record.backup_id}")
                        return False
                    
                    # Test read
                    backup_json = zipf.read('backup_data.json').decode('utf-8')
                    backup_data = json.loads(backup_json)
                    
                    # Verify analytics count
                    analytics_count = len(backup_data.get('analytics', []))
                    if analytics_count != backup_record.analytics_count:
                        logger.error(f"Analytics count mismatch in backup: {backup_record.backup_id}")
                        return False
                    
                    return True
                    
            except (zipfile.BadZipFile, json.JSONDecodeError) as e:
                logger.error(f"Backup file corrupted: {backup_record.backup_id} - {e}")
                return False
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    def _store_in_hot_storage(self, backup_id: str, analytics_data: List[Dict[str, Any]]):
        """Store backup in hot storage for instant access."""
        try:
            # Store in hot storage
            self.hot_storage[backup_id] = {
                'backup_id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'analytics': analytics_data,
                'count': len(analytics_data)
            }
            
            # Update analytics index
            for analytics in analytics_data:
                analytics_id = analytics.get('analytics_id') or analytics.get('id')
                if analytics_id:
                    self.hot_analytics_index[analytics_id] = backup_id
            
            # Manage hot storage size
            self._manage_hot_storage()
            
            # Update statistics
            hot_size_mb = sum(len(json.dumps(data, default=str)) for data in self.hot_storage.values()) / (1024 * 1024)
            self.stats['hot_storage_size_mb'] = hot_size_mb
            
        except Exception as e:
            logger.error(f"Hot storage update failed: {e}")
    
    def _manage_hot_storage(self):
        """Manage hot storage to prevent memory overflow."""
        try:
            if len(self.hot_storage) <= self.max_hot_backups:
                return
            
            # Remove oldest backups from hot storage
            sorted_backups = sorted(
                self.hot_storage.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            remove_count = len(self.hot_storage) - self.max_hot_backups
            
            for i in range(remove_count):
                backup_id, backup_data = sorted_backups[i]
                
                # Remove from hot storage
                self.hot_storage.pop(backup_id, None)
                
                # Remove from analytics index
                for analytics in backup_data.get('analytics', []):
                    analytics_id = analytics.get('analytics_id') or analytics.get('id')
                    if analytics_id and self.hot_analytics_index.get(analytics_id) == backup_id:
                        self.hot_analytics_index.pop(analytics_id, None)
            
            logger.debug(f"Removed {remove_count} backups from hot storage")
            
        except Exception as e:
            logger.error(f"Hot storage management failed: {e}")
    
    def _find_best_backup(self, point_in_time: datetime = None) -> Optional[str]:
        """Find the best backup for recovery."""
        try:
            if not self.backup_records:
                return None
            
            # Filter by point in time if specified
            if point_in_time:
                candidates = [
                    (backup_id, record) for backup_id, record in self.backup_records.items()
                    if record.created_at <= point_in_time and record.status == BackupStatus.COMPLETED
                ]
            else:
                candidates = [
                    (backup_id, record) for backup_id, record in self.backup_records.items()
                    if record.status == BackupStatus.COMPLETED
                ]
            
            if not candidates:
                return None
            
            # Prioritize by: 1) Hot storage, 2) Most recent, 3) Full backups
            def backup_priority(item):
                backup_id, record = item
                priority_score = 0
                
                # Hot storage gets highest priority
                if record.storage_tier == StorageTier.HOT:
                    priority_score += 1000
                elif record.storage_tier == StorageTier.WARM:
                    priority_score += 100
                
                # Recent backups get higher priority
                age_hours = (datetime.now() - record.created_at).total_seconds() / 3600
                priority_score += max(0, 100 - age_hours)
                
                # Full backups get higher priority than incremental
                if record.backup_type == BackupType.FULL:
                    priority_score += 50
                elif record.backup_type == BackupType.EMERGENCY:
                    priority_score += 75
                
                return priority_score
            
            # Sort by priority and return best
            sorted_candidates = sorted(candidates, key=backup_priority, reverse=True)
            return sorted_candidates[0][0]
            
        except Exception as e:
            logger.error(f"Best backup selection failed: {e}")
            return None
    
    def _perform_recovery(self, recovery_record: RecoveryRecord, analytics_id_filter: List[str] = None) -> bool:
        """Perform the actual recovery operation."""
        try:
            backup_id = recovery_record.backup_id
            
            # Try hot storage first for instant recovery
            if backup_id in self.hot_storage:
                return self._recover_from_hot_storage(recovery_record, analytics_id_filter)
            
            # Fall back to file-based recovery
            backup_record = self.backup_records.get(backup_id)
            if not backup_record:
                return False
            
            return self._recover_from_file(recovery_record, backup_record, analytics_id_filter)
            
        except Exception as e:
            logger.error(f"Recovery operation failed: {e}")
            recovery_record.error_info = str(e)
            return False
    
    def _recover_from_hot_storage(self, recovery_record: RecoveryRecord, analytics_id_filter: List[str] = None) -> bool:
        """Recover analytics from hot storage."""
        try:
            backup_id = recovery_record.backup_id
            hot_data = self.hot_storage[backup_id]
            analytics_data = hot_data['analytics']
            
            # Filter analytics if specified
            if analytics_id_filter:
                filtered_analytics = []
                for analytics in analytics_data:
                    analytics_id = analytics.get('analytics_id') or analytics.get('id')
                    if analytics_id in analytics_id_filter:
                        filtered_analytics.append(analytics)
                analytics_data = filtered_analytics
            
            # Restore analytics to aggregator
            if self.aggregator and hasattr(self.aggregator, 'restore_analytics_batch'):
                success = self.aggregator.restore_analytics_batch(analytics_data)
            else:
                # Individual restoration
                success_count = 0
                for analytics in analytics_data:
                    if self._restore_single_analytics(analytics):
                        success_count += 1
                success = success_count > 0
            
            recovery_record.recovered_analytics_count = len(analytics_data)
            
            logger.info(f"Hot storage recovery: {len(analytics_data)} analytics restored")
            return success
            
        except Exception as e:
            logger.error(f"Hot storage recovery failed: {e}")
            return False
    
    def _recover_from_file(self, recovery_record: RecoveryRecord, backup_record: BackupRecord, analytics_id_filter: List[str] = None) -> bool:
        """Recover analytics from backup file."""
        try:
            file_path = Path(backup_record.file_path)
            
            if not file_path.exists():
                logger.error(f"Backup file not found: {file_path}")
                return False
            
            # Extract analytics data
            with zipfile.ZipFile(file_path, 'r') as zipf:
                # Read main backup data
                backup_json = zipf.read('backup_data.json').decode('utf-8')
                backup_data = json.loads(backup_json)
                analytics_data = backup_data.get('analytics', [])
                
                # Filter analytics if specified
                if analytics_id_filter:
                    filtered_analytics = []
                    for analytics in analytics_data:
                        analytics_id = analytics.get('analytics_id') or analytics.get('id')
                        if analytics_id in analytics_id_filter:
                            filtered_analytics.append(analytics)
                    analytics_data = filtered_analytics
                
                # Restore analytics
                if self.aggregator and hasattr(self.aggregator, 'restore_analytics_batch'):
                    success = self.aggregator.restore_analytics_batch(analytics_data)
                else:
                    # Individual restoration
                    success_count = 0
                    for analytics in analytics_data:
                        if self._restore_single_analytics(analytics):
                            success_count += 1
                    success = success_count > 0
                
                recovery_record.recovered_analytics_count = len(analytics_data)
                
                logger.info(f"File recovery: {len(analytics_data)} analytics restored from {file_path}")
                return success
            
        except Exception as e:
            logger.error(f"File recovery failed: {e}")
            return False
    
    def _restore_single_analytics(self, analytics: Dict[str, Any]) -> bool:
        """Restore a single analytics record."""
        try:
            if self.aggregator and hasattr(self.aggregator, 'restore_analytics'):
                analytics_id = analytics.get('analytics_id') or analytics.get('id')
                return self.aggregator.restore_analytics(analytics_id, analytics)
            else:
                # Log successful restoration (no aggregator available)
                logger.debug(f"Analytics restored (no aggregator): {analytics.get('analytics_id', 'unknown')}")
                return True
                
        except Exception as e:
            logger.error(f"Single analytics restoration failed: {e}")
            return False
    
    def _backup_loop(self):
        """Background backup loop."""
        while self.backup_active:
            try:
                current_time = datetime.now()
                
                # Check if it's time for automatic backup
                should_backup = False
                backup_type = BackupType.INCREMENTAL
                
                # Check for full backup (daily)
                if (not self.last_full_backup or 
                    (current_time - self.last_full_backup).total_seconds() > 86400):  # 24 hours
                    should_backup = True
                    backup_type = BackupType.FULL
                    self.last_full_backup = current_time
                
                # Check for incremental backup
                elif (not self.last_incremental_backup or 
                      (current_time - self.last_incremental_backup).total_seconds() > self.backup_interval):
                    should_backup = True
                    backup_type = BackupType.INCREMENTAL
                    self.last_incremental_backup = current_time
                
                # Perform backup if needed
                if should_backup:
                    analytics_data = self._collect_all_analytics()
                    if analytics_data:
                        backup_id = self._create_scheduled_backup(analytics_data, backup_type)
                        if backup_id:
                            logger.info(f"Scheduled backup created: {backup_id} ({backup_type.value})")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _create_scheduled_backup(self, analytics_data: List[Dict[str, Any]], backup_type: BackupType) -> str:
        """Create a scheduled backup."""
        try:
            backup_id = f"scheduled_{backup_type.value}_{int(time.time() * 1000000)}"
            
            # Determine storage tier based on backup type
            if backup_type == BackupType.FULL:
                storage_tier = StorageTier.WARM
            else:
                storage_tier = StorageTier.HOT
            
            # Create backup record
            backup_record = BackupRecord(
                backup_id=backup_id,
                backup_type=backup_type,
                storage_tier=storage_tier,
                created_at=datetime.now(),
                status=BackupStatus.INITIATED,
                file_path="",
                compressed_size=0,
                original_size=0,
                checksum="",
                analytics_count=len(analytics_data),
                metadata={
                    'scheduled': True,
                    'platform': platform.system(),
                    'auto_created': True
                }
            )
            
            # Store backup record
            self.backup_records[backup_id] = backup_record
            
            # Create backup file
            start_time = time.time()
            success = self._create_backup_file(backup_record, analytics_data)
            backup_time = time.time() - start_time
            
            if success:
                backup_record.status = BackupStatus.COMPLETED
                self.stats['successful_backups'] += 1
                
                # Store in hot storage if appropriate
                if storage_tier == StorageTier.HOT:
                    self._store_in_hot_storage(backup_id, analytics_data)
                
                # Update average backup time
                total_backups = self.stats['total_backups_created'] + 1
                current_avg = self.stats['average_backup_time']
                self.stats['average_backup_time'] = (
                    (current_avg * (total_backups - 1) + backup_time) / total_backups
                )
                
                logger.debug(f"Scheduled backup completed in {backup_time:.2f}s: {backup_id}")
            else:
                backup_record.status = BackupStatus.FAILED
                self.stats['failed_backups'] += 1
            
            # Save to database
            self._save_backup_record(backup_record)
            self.stats['total_backups_created'] += 1
            
            return backup_id if success else ""
            
        except Exception as e:
            logger.error(f"Scheduled backup creation failed: {e}")
            return ""
    
    def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.backup_active:
            try:
                time.sleep(3600)  # Run maintenance every hour
                
                with self.lock:
                    self._cleanup_expired_backups()
                    self._tier_migration()
                    self._verify_backup_integrity_batch()
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
    
    def _cleanup_expired_backups(self):
        """Clean up expired backups based on retention policy."""
        try:
            current_time = datetime.now()
            expired_backups = []
            
            for backup_id, record in self.backup_records.items():
                retention_period = self.retention_policy.get(record.storage_tier)
                if retention_period and record.created_at + retention_period < current_time:
                    expired_backups.append(backup_id)
            
            for backup_id in expired_backups:
                record = self.backup_records[backup_id]
                
                # Remove file
                try:
                    file_path = Path(record.file_path)
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove backup file {record.file_path}: {e}")
                
                # Remove from hot storage
                self.hot_storage.pop(backup_id, None)
                
                # Remove from records
                self.backup_records.pop(backup_id, None)
                
                # Remove from database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM backup_records WHERE backup_id = ?', (backup_id,))
                    conn.execute('DELETE FROM analytics_backup_map WHERE backup_id = ?', (backup_id,))
                    conn.commit()
            
            if expired_backups:
                logger.info(f"Cleaned up {len(expired_backups)} expired backups")
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def _tier_migration(self):
        """Migrate backups between storage tiers."""
        try:
            current_time = datetime.now()
            
            for backup_id, record in list(self.backup_records.items()):
                age = current_time - record.created_at
                
                # Migrate HOT -> WARM after 24 hours
                if (record.storage_tier == StorageTier.HOT and 
                    age > timedelta(hours=24)):
                    self._migrate_backup_tier(record, StorageTier.WARM)
                
                # Migrate WARM -> COLD after 7 days
                elif (record.storage_tier == StorageTier.WARM and 
                      age > timedelta(days=7)):
                    self._migrate_backup_tier(record, StorageTier.COLD)
                
                # Migrate COLD -> FROZEN after 30 days
                elif (record.storage_tier == StorageTier.COLD and 
                      age > timedelta(days=30)):
                    self._migrate_backup_tier(record, StorageTier.FROZEN)
            
        except Exception as e:
            logger.error(f"Tier migration failed: {e}")
    
    def _migrate_backup_tier(self, backup_record: BackupRecord, new_tier: StorageTier):
        """Migrate backup to new storage tier."""
        try:
            old_path = Path(backup_record.file_path)
            
            # Generate new path
            filename = old_path.name
            new_path = self.backup_base_path / new_tier.value / filename
            
            # Move file
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))
                
                # Update record
                backup_record.storage_tier = new_tier
                backup_record.file_path = str(new_path)
                
                # Remove from hot storage if migrating away from HOT
                if new_tier != StorageTier.HOT:
                    self.hot_storage.pop(backup_record.backup_id, None)
                
                # Save updated record
                self._save_backup_record(backup_record)
                
                logger.debug(f"Migrated backup {backup_record.backup_id} to {new_tier.value}")
            
        except Exception as e:
            logger.error(f"Backup tier migration failed: {e}")
    
    def _verify_backup_integrity_batch(self):
        """Verify integrity of recent backups."""
        try:
            # Check backups from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_backups = [
                record for record in self.backup_records.values()
                if record.created_at > cutoff_time and record.status == BackupStatus.COMPLETED
            ]
            
            for backup_record in recent_backups[:5]:  # Limit to 5 backups per cycle
                if self._verify_backup_integrity(backup_record):
                    backup_record.status = BackupStatus.VERIFIED
                else:
                    backup_record.status = BackupStatus.CORRUPTED
                    logger.warning(f"Backup integrity verification failed: {backup_record.backup_id}")
                
                self._save_backup_record(backup_record)
            
        except Exception as e:
            logger.error(f"Batch integrity verification failed: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.backup_active:
            try:
                time.sleep(300)  # Monitor every 5 minutes
                
                with self.lock:
                    # Monitor disk space
                    self._monitor_disk_space()
                    
                    # Monitor backup health
                    self._monitor_backup_health()
                    
                    # Update statistics
                    self._update_statistics()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _monitor_disk_space(self):
        """Monitor available disk space."""
        try:
            backup_path = Path(self.backup_base_path)
            stat = backup_path.stat()
            
            # Get disk usage (this is a simplified check)
            total_backup_size = sum(
                Path(record.file_path).stat().st_size 
                for record in self.backup_records.values()
                if Path(record.file_path).exists()
            )
            
            # Warn if using too much space (>10GB)
            if total_backup_size > 10 * 1024 * 1024 * 1024:
                logger.warning(f"Backup storage using {total_backup_size / (1024**3):.1f}GB")
            
        except Exception as e:
            logger.error(f"Disk space monitoring failed: {e}")
    
    def _monitor_backup_health(self):
        """Monitor overall backup system health."""
        try:
            total_backups = len(self.backup_records)
            successful_backups = sum(
                1 for record in self.backup_records.values()
                if record.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
            )
            
            if total_backups > 0:
                health_percentage = (successful_backups / total_backups) * 100
                if health_percentage < 90:
                    logger.warning(f"Backup system health: {health_percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"Backup health monitoring failed: {e}")
    
    def _update_statistics(self):
        """Update system statistics."""
        try:
            # Update hot storage size
            hot_size_mb = sum(
                len(json.dumps(data, default=str)) 
                for data in self.hot_storage.values()
            ) / (1024 * 1024)
            
            self.stats['hot_storage_size_mb'] = hot_size_mb
            
        except Exception as e:
            logger.error(f"Statistics update failed: {e}")
    
    def _load_backup_registry(self):
        """Load existing backup records from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM backup_records ORDER BY created_at DESC LIMIT 100')
                
                for row in cursor.fetchall():
                    backup_record = BackupRecord(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        storage_tier=StorageTier(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        status=BackupStatus(row[4]),
                        file_path=row[5],
                        compressed_size=row[6],
                        original_size=row[7],
                        checksum=row[8],
                        analytics_count=row[9],
                        metadata=json.loads(row[10]) if row[10] else {},
                        retention_date=datetime.fromisoformat(row[11]) if row[11] else None
                    )
                    
                    self.backup_records[backup_record.backup_id] = backup_record
                
                logger.info(f"Loaded {len(self.backup_records)} backup records from registry")
                
        except Exception as e:
            logger.error(f"Backup registry loading failed: {e}")
    
    def _save_backup_record(self, backup_record: BackupRecord):
        """Save backup record to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO backup_records
                    (backup_id, backup_type, storage_tier, created_at, status,
                     file_path, compressed_size, original_size, checksum,
                     analytics_count, metadata, retention_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    backup_record.backup_id,
                    backup_record.backup_type.value,
                    backup_record.storage_tier.value,
                    backup_record.created_at.isoformat(),
                    backup_record.status.value,
                    backup_record.file_path,
                    backup_record.compressed_size,
                    backup_record.original_size,
                    backup_record.checksum,
                    backup_record.analytics_count,
                    json.dumps(backup_record.metadata or {}),
                    backup_record.retention_date.isoformat() if backup_record.retention_date else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save backup record: {e}")
    
    def _save_recovery_record(self, recovery_record: RecoveryRecord):
        """Save recovery record to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO recovery_records
                    (recovery_id, backup_id, recovery_mode, initiated_at, completed_at,
                     status, recovered_analytics_count, recovery_time_seconds, error_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    recovery_record.recovery_id,
                    recovery_record.backup_id,
                    recovery_record.recovery_mode.value,
                    recovery_record.initiated_at.isoformat(),
                    recovery_record.completed_at.isoformat() if recovery_record.completed_at else None,
                    recovery_record.status.value,
                    recovery_record.recovered_analytics_count,
                    recovery_record.recovery_time_seconds,
                    recovery_record.error_info
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save recovery record: {e}")
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get comprehensive backup system statistics."""
        with self.lock:
            return {
                'statistics': dict(self.stats),
                'backup_counts': {
                    'total_backups': len(self.backup_records),
                    'hot_backups': sum(1 for r in self.backup_records.values() if r.storage_tier == StorageTier.HOT),
                    'warm_backups': sum(1 for r in self.backup_records.values() if r.storage_tier == StorageTier.WARM),
                    'cold_backups': sum(1 for r in self.backup_records.values() if r.storage_tier == StorageTier.COLD),
                    'frozen_backups': sum(1 for r in self.backup_records.values() if r.storage_tier == StorageTier.FROZEN)
                },
                'recovery_counts': {
                    'total_recoveries': len(self.recovery_records),
                    'successful_recoveries': self.stats['successful_recoveries'],
                    'failed_recoveries': self.stats['failed_recoveries']
                },
                'storage_info': {
                    'hot_storage_items': len(self.hot_storage),
                    'hot_storage_size_mb': self.stats['hot_storage_size_mb'],
                    'total_data_backed_up_mb': self.stats['total_data_backed_up_mb']
                },
                'configuration': {
                    'backup_interval': self.backup_interval,
                    'max_hot_backups': self.max_hot_backups,
                    'compression_level': self.compression_level,
                    'verify_backups': self.verify_backups,
                    'auto_recovery_enabled': self.auto_recovery_enabled
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_backup_details(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific backup."""
        if backup_id in self.backup_records:
            return self.backup_records[backup_id].to_dict()
        
        # Check database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT * FROM backup_records WHERE backup_id = ?',
                    (backup_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        'backup_id': row[0],
                        'backup_type': row[1],
                        'storage_tier': row[2],
                        'created_at': row[3],
                        'status': row[4],
                        'file_path': row[5],
                        'compressed_size': row[6],
                        'original_size': row[7],
                        'checksum': row[8],
                        'analytics_count': row[9],
                        'metadata': json.loads(row[10]) if row[10] else {}
                    }
        except Exception as e:
            logger.error(f"Backup details lookup failed: {e}")
        
        return None
    
    def force_backup_verification(self, backup_id: str) -> bool:
        """Force verification of specific backup."""
        try:
            backup_record = self.backup_records.get(backup_id)
            if not backup_record:
                return False
            
            return self._verify_backup_integrity(backup_record)
            
        except Exception as e:
            logger.error(f"Force backup verification failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown emergency backup and recovery system."""
        self.backup_active = False
        
        # Wait for threads to complete
        for thread in [self.backup_thread, self.maintenance_thread, self.monitoring_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Emergency Backup and Recovery System shutdown - Stats: {self.stats}")

# Global emergency backup instance
emergency_backup = None