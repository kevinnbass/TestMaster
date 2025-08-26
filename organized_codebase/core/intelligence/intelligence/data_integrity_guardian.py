"""
Archive Derived Data Integrity Guardian Security Module
Extracted from TestMaster archive integrity systems for comprehensive data protection
Enhanced for multi-layer integrity verification and tamper detection
"""

import uuid
import time
import json
import hashlib
import hmac
import logging
import threading
import sqlite3
import zlib
import base64
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .error_handler import SecurityError, security_error_handler


class IntegrityStatus(Enum):
    """Data integrity verification status"""
    VERIFIED = "verified"
    CORRUPTED = "corrupted"
    TAMPERED = "tampered"
    MISSING = "missing"
    RECOVERED = "recovered"
    QUARANTINED = "quarantined"


class ChecksumAlgorithm(Enum):
    """Cryptographic checksum algorithms"""
    MD5 = "md5"
    SHA256 = "sha256"
    SHA512 = "sha512"
    CRC32 = "crc32"
    MULTI_LAYER = "multi_layer"


class IntegrityViolationType(Enum):
    """Types of integrity violations"""
    CHECKSUM_MISMATCH = "checksum_mismatch"
    TAMPER_DETECTION = "tamper_detection"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_CORRUPTION = "data_corruption"
    REPLAY_ATTACK = "replay_attack"


@dataclass
class IntegrityRecord:
    """Comprehensive data integrity record"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_id: str = ""
    data_type: str = ""
    original_checksum: str = ""
    verification_checksum: str = ""
    algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    status: IntegrityStatus = IntegrityStatus.VERIFIED
    created_at: datetime = field(default_factory=datetime.utcnow)
    verified_at: Optional[datetime] = None
    error_message: Optional[str] = None
    violation_type: Optional[IntegrityViolationType] = None
    recovery_attempted: bool = False
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for storage"""
        return {
            'record_id': self.record_id,
            'data_id': self.data_id,
            'data_type': self.data_type,
            'original_checksum': self.original_checksum,
            'verification_checksum': self.verification_checksum,
            'algorithm': self.algorithm.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'error_message': self.error_message,
            'violation_type': self.violation_type.value if self.violation_type else None,
            'recovery_attempted': self.recovery_attempted,
            'access_count': self.access_count,
            'last_access': self.last_access.isoformat() if self.last_access else None,
            'metadata': self.metadata
        }


@dataclass
class DataSnapshot:
    """Secure data snapshot with integrity protection"""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_id: str = ""
    compressed_data: bytes = b""
    checksum: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    encryption_key: Optional[str] = None
    size: int = 0
    compression_ratio: float = 0.0
    
    def calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of data"""
        return hashlib.sha256(data).hexdigest()


class IntegrityViolationDetector:
    """Advanced tamper and integrity violation detection"""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Checksum manipulation attempts
            r'checksum\s*=\s*["\'][a-f0-9]{32,128}["\']',
            # Hash collision attempts
            r'collision\s*attack',
            # Data modification patterns
            r'modify\s*integrity',
            # Replay attack indicators
            r'replay\s*attack'
        ]
        
        self.violation_thresholds = {
            IntegrityViolationType.CHECKSUM_MISMATCH: 3,
            IntegrityViolationType.TAMPER_DETECTION: 2,
            IntegrityViolationType.UNAUTHORIZED_ACCESS: 5,
            IntegrityViolationType.DATA_CORRUPTION: 1,
            IntegrityViolationType.REPLAY_ATTACK: 2
        }
        
        self.logger = logging.getLogger(__name__)
    
    def detect_tampering(self, original_data: bytes, current_data: bytes, 
                        metadata: Dict[str, Any]) -> Tuple[bool, Optional[IntegrityViolationType]]:
        """Detect potential tampering in data"""
        try:
            # Basic checksum comparison
            if hashlib.sha256(original_data).hexdigest() != hashlib.sha256(current_data).hexdigest():
                return True, IntegrityViolationType.CHECKSUM_MISMATCH
            
            # Size difference check
            if len(original_data) != len(current_data):
                return True, IntegrityViolationType.DATA_CORRUPTION
            
            # Timestamp analysis for replay attacks
            if 'timestamp' in metadata:
                try:
                    data_timestamp = datetime.fromisoformat(metadata['timestamp'])
                    if (datetime.utcnow() - data_timestamp).total_seconds() > 3600:  # 1 hour threshold
                        return True, IntegrityViolationType.REPLAY_ATTACK
                except (ValueError, TypeError):
                    pass
            
            # Pattern-based tamper detection
            combined_data = original_data + current_data
            data_str = combined_data.decode('utf-8', errors='ignore')
            
            import re
            for pattern in self.suspicious_patterns:
                if re.search(pattern, data_str, re.IGNORECASE):
                    return True, IntegrityViolationType.TAMPER_DETECTION
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Tamper detection failed: {e}")
            return True, IntegrityViolationType.DATA_CORRUPTION
    
    def analyze_access_patterns(self, access_history: List[Tuple[datetime, str]]) -> bool:
        """Analyze access patterns for suspicious behavior"""
        if len(access_history) < 2:
            return False
        
        # Check for rapid successive accesses (potential attack)
        recent_accesses = []
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        
        for access_time, accessor in access_history:
            if access_time > cutoff:
                recent_accesses.append((access_time, accessor))
        
        # More than 10 accesses in 5 minutes from same accessor
        if len(recent_accesses) > 10:
            accessor_counts = {}
            for _, accessor in recent_accesses:
                accessor_counts[accessor] = accessor_counts.get(accessor, 0) + 1
            
            if max(accessor_counts.values()) > 10:
                return True
        
        return False


class SecureDataStorage:
    """Secure data storage with compression and encryption"""
    
    def __init__(self, storage_path: str = "secure_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.compression_level = 6  # zlib compression level
        self.logger = logging.getLogger(__name__)
    
    def store_data(self, data_id: str, data: bytes, encrypt: bool = True) -> DataSnapshot:
        """Store data securely with compression and optional encryption"""
        try:
            # Compress data
            compressed_data = zlib.compress(data, self.compression_level)
            compression_ratio = len(compressed_data) / len(data) if len(data) > 0 else 0.0
            
            # Calculate checksum
            checksum = hashlib.sha256(data).hexdigest()
            
            # Create snapshot
            snapshot = DataSnapshot(
                data_id=data_id,
                compressed_data=compressed_data,
                checksum=checksum,
                size=len(data),
                compression_ratio=compression_ratio
            )
            
            # Encrypt if requested
            if encrypt:
                encryption_key = base64.b64encode(uuid.uuid4().bytes).decode()
                snapshot.encryption_key = encryption_key
                # In production, use proper encryption like Fernet
                # For now, just XOR with key (placeholder)
                key_bytes = encryption_key.encode()[:32]
                encrypted_data = bytes(a ^ key_bytes[i % len(key_bytes)] 
                                     for i, a in enumerate(compressed_data))
                snapshot.compressed_data = encrypted_data
            
            # Store to file
            snapshot_file = self.storage_path / f"{snapshot.snapshot_id}.dat"
            with open(snapshot_file, 'wb') as f:
                f.write(snapshot.compressed_data)
            
            # Store metadata
            metadata_file = self.storage_path / f"{snapshot.snapshot_id}.meta"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'data_id': snapshot.data_id,
                    'checksum': snapshot.checksum,
                    'timestamp': snapshot.timestamp.isoformat(),
                    'size': snapshot.size,
                    'compression_ratio': snapshot.compression_ratio,
                    'encrypted': encrypt
                }, f)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Data storage failed: {e}")
            raise SecurityError(f"Failed to store data securely: {str(e)}", "DATA_STORE_001")
    
    def retrieve_data(self, snapshot: DataSnapshot) -> bytes:
        """Retrieve and decompress data from snapshot"""
        try:
            snapshot_file = self.storage_path / f"{snapshot.snapshot_id}.dat"
            
            if not snapshot_file.exists():
                raise SecurityError("Snapshot file not found", "DATA_RETRIEVE_001")
            
            with open(snapshot_file, 'rb') as f:
                compressed_data = f.read()
            
            # Decrypt if necessary
            if snapshot.encryption_key:
                key_bytes = snapshot.encryption_key.encode()[:32]
                decrypted_data = bytes(a ^ key_bytes[i % len(key_bytes)] 
                                     for i, a in enumerate(compressed_data))
                compressed_data = decrypted_data
            
            # Decompress
            original_data = zlib.decompress(compressed_data)
            
            # Verify checksum
            if hashlib.sha256(original_data).hexdigest() != snapshot.checksum:
                raise SecurityError("Data checksum verification failed", "DATA_RETRIEVE_002")
            
            return original_data
            
        except Exception as e:
            self.logger.error(f"Data retrieval failed: {e}")
            raise SecurityError(f"Failed to retrieve data: {str(e)}", "DATA_RETRIEVE_003")


class DataIntegrityGuardian:
    """Comprehensive data integrity guardian with multi-layer security"""
    
    def __init__(self, db_path: str = "data/integrity_guardian.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.violation_detector = IntegrityViolationDetector()
        self.secure_storage = SecureDataStorage()
        
        # Initialize database
        self._init_database()
        
        # In-memory tracking
        self.integrity_records: Dict[str, IntegrityRecord] = {}
        self.data_snapshots: Dict[str, List[DataSnapshot]] = {}
        self.access_history: Dict[str, List[Tuple[datetime, str]]] = {}
        self.quarantine_zone: Set[str] = set()
        
        # Statistics
        self.stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'integrity_violations': 0,
            'tamper_attempts': 0,
            'data_recoveries': 0,
            'quarantined_items': 0,
            'verification_success_rate': 100.0
        }
        
        # Configuration
        self.max_snapshots_per_data = 5
        self.verification_interval = 60  # seconds
        self.auto_quarantine = True
        self.enable_recovery = True
        
        # Background monitoring
        self.guardian_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        
        # Thread safety
        self.guardian_lock = threading.RLock()
        
        # Start background threads
        self.monitor_thread.start()
        self.cleanup_thread.start()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Data Integrity Guardian initialized")
    
    def _init_database(self):
        """Initialize integrity database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS integrity_records (
                        record_id TEXT PRIMARY KEY,
                        data_id TEXT NOT NULL,
                        data_type TEXT,
                        original_checksum TEXT NOT NULL,
                        verification_checksum TEXT,
                        algorithm TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        verified_at TEXT,
                        error_message TEXT,
                        violation_type TEXT,
                        recovery_attempted BOOLEAN DEFAULT 0,
                        access_count INTEGER DEFAULT 0,
                        last_access TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS access_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_id TEXT NOT NULL,
                        accessor_id TEXT NOT NULL,
                        access_time TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        success BOOLEAN DEFAULT 1
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_integrity_data_id 
                    ON integrity_records(data_id)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_access_log_data_id 
                    ON access_log(data_id)
                ''')
                
        except Exception as e:
            raise SecurityError(f"Database initialization failed: {str(e)}", "DB_INIT_001")
    
    def protect_data(self, data_id: str, data: bytes, data_type: str = "binary",
                    accessor_id: str = "system") -> bool:
        """Protect data with integrity verification and secure storage"""
        try:
            with self.guardian_lock:
                # Calculate checksums
                checksums = self._calculate_checksums(data)
                
                # Create integrity record
                record = IntegrityRecord(
                    data_id=data_id,
                    data_type=data_type,
                    original_checksum=checksums['sha256'],
                    algorithm=ChecksumAlgorithm.MULTI_LAYER,
                    status=IntegrityStatus.VERIFIED
                )
                
                # Store data snapshot
                snapshot = self.secure_storage.store_data(data_id, data, encrypt=True)
                
                if data_id not in self.data_snapshots:
                    self.data_snapshots[data_id] = []
                
                self.data_snapshots[data_id].append(snapshot)
                
                # Limit snapshots per data item
                if len(self.data_snapshots[data_id]) > self.max_snapshots_per_data:
                    self.data_snapshots[data_id] = self.data_snapshots[data_id][-self.max_snapshots_per_data:]
                
                # Store integrity record
                self.integrity_records[data_id] = record
                self._persist_integrity_record(record)
                
                # Log access
                self._log_access(data_id, accessor_id, "protect")
                
                self.logger.info(f"Data protected: {data_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Data protection failed: {str(e)}", "DATA_PROTECT_001")
            security_error_handler.handle_error(error)
            return False
    
    def verify_data_integrity(self, data_id: str, current_data: bytes, 
                            accessor_id: str = "system") -> Tuple[bool, IntegrityStatus]:
        """Verify data integrity against stored checksums"""
        try:
            with self.guardian_lock:
                self.stats['total_verifications'] += 1
                
                # Check if data is quarantined
                if data_id in self.quarantine_zone:
                    return False, IntegrityStatus.QUARANTINED
                
                # Get integrity record
                if data_id not in self.integrity_records:
                    return False, IntegrityStatus.MISSING
                
                record = self.integrity_records[data_id]
                
                # Calculate current checksum
                current_checksum = hashlib.sha256(current_data).hexdigest()
                record.verification_checksum = current_checksum
                record.verified_at = datetime.utcnow()
                record.access_count += 1
                record.last_access = datetime.utcnow()
                
                # Verify integrity
                if current_checksum == record.original_checksum:
                    record.status = IntegrityStatus.VERIFIED
                    self.stats['successful_verifications'] += 1
                    
                    # Log successful access
                    self._log_access(data_id, accessor_id, "verify", success=True)
                    
                    # Update statistics
                    self.stats['verification_success_rate'] = (
                        self.stats['successful_verifications'] / self.stats['total_verifications'] * 100
                    )
                    
                    return True, IntegrityStatus.VERIFIED
                else:
                    # Integrity violation detected
                    self.stats['integrity_violations'] += 1
                    record.status = IntegrityStatus.CORRUPTED
                    record.error_message = "Checksum mismatch detected"
                    
                    # Detect tampering
                    if data_id in self.data_snapshots and self.data_snapshots[data_id]:
                        latest_snapshot = self.data_snapshots[data_id][-1]
                        original_data = self.secure_storage.retrieve_data(latest_snapshot)
                        
                        is_tampered, violation_type = self.violation_detector.detect_tampering(
                            original_data, current_data, record.metadata
                        )
                        
                        if is_tampered:
                            record.violation_type = violation_type
                            record.status = IntegrityStatus.TAMPERED
                            self.stats['tamper_attempts'] += 1
                            
                            # Auto-quarantine if enabled
                            if self.auto_quarantine:
                                self.quarantine_zone.add(data_id)
                                self.stats['quarantined_items'] += 1
                                self.logger.warning(f"Data quarantined due to tampering: {data_id}")
                    
                    # Log failed access
                    self._log_access(data_id, accessor_id, "verify", success=False)
                    
                    # Update record
                    self._persist_integrity_record(record)
                    
                    self.logger.warning(f"Integrity violation detected: {data_id}")
                    return False, record.status
                
        except Exception as e:
            error = SecurityError(f"Integrity verification failed: {str(e)}", "VERIFY_001")
            security_error_handler.handle_error(error)
            return False, IntegrityStatus.CORRUPTED
    
    def recover_data(self, data_id: str, accessor_id: str = "system") -> Optional[bytes]:
        """Attempt to recover data from secure snapshots"""
        try:
            with self.guardian_lock:
                if not self.enable_recovery:
                    return None
                
                if data_id not in self.data_snapshots or not self.data_snapshots[data_id]:
                    return None
                
                # Try to recover from latest snapshot
                for snapshot in reversed(self.data_snapshots[data_id]):
                    try:
                        recovered_data = self.secure_storage.retrieve_data(snapshot)
                        
                        # Verify recovered data integrity
                        if hashlib.sha256(recovered_data).hexdigest() == snapshot.checksum:
                            # Update integrity record
                            if data_id in self.integrity_records:
                                record = self.integrity_records[data_id]
                                record.status = IntegrityStatus.RECOVERED
                                record.recovery_attempted = True
                                record.error_message = "Data recovered from snapshot"
                                self._persist_integrity_record(record)
                            
                            # Remove from quarantine
                            if data_id in self.quarantine_zone:
                                self.quarantine_zone.remove(data_id)
                                self.stats['quarantined_items'] -= 1
                            
                            self.stats['data_recoveries'] += 1
                            self._log_access(data_id, accessor_id, "recover")
                            
                            self.logger.info(f"Data recovered successfully: {data_id}")
                            return recovered_data
                            
                    except Exception as e:
                        self.logger.warning(f"Snapshot recovery failed: {e}")
                        continue
                
                return None
                
        except Exception as e:
            error = SecurityError(f"Data recovery failed: {str(e)}", "RECOVER_001")
            security_error_handler.handle_error(error)
            return None
    
    def get_integrity_stats(self) -> Dict[str, Any]:
        """Get comprehensive integrity statistics"""
        with self.guardian_lock:
            return {
                **self.stats,
                'protected_data_items': len(self.integrity_records),
                'total_snapshots': sum(len(snapshots) for snapshots in self.data_snapshots.values()),
                'quarantined_items_list': list(self.quarantine_zone),
                'recent_violations': self._get_recent_violations()
            }
    
    def _calculate_checksums(self, data: bytes) -> Dict[str, str]:
        """Calculate multiple checksums for enhanced security"""
        checksums = {
            'sha256': hashlib.sha256(data).hexdigest(),
            'sha512': hashlib.sha512(data).hexdigest(),
            'md5': hashlib.md5(data).hexdigest(),
            'crc32': hex(zlib.crc32(data) & 0xffffffff)
        }
        
        # Multi-layer checksum (combination)
        combined = ''.join(checksums.values())
        checksums['multi_layer'] = hashlib.sha256(combined.encode()).hexdigest()
        
        return checksums
    
    def _persist_integrity_record(self, record: IntegrityRecord):
        """Persist integrity record to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO integrity_records 
                    (record_id, data_id, data_type, original_checksum, verification_checksum,
                     algorithm, status, created_at, verified_at, error_message, violation_type,
                     recovery_attempted, access_count, last_access, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.record_id, record.data_id, record.data_type,
                    record.original_checksum, record.verification_checksum,
                    record.algorithm.value, record.status.value,
                    record.created_at.isoformat(),
                    record.verified_at.isoformat() if record.verified_at else None,
                    record.error_message,
                    record.violation_type.value if record.violation_type else None,
                    record.recovery_attempted, record.access_count,
                    record.last_access.isoformat() if record.last_access else None,
                    json.dumps(record.metadata)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to persist integrity record: {e}")
    
    def _log_access(self, data_id: str, accessor_id: str, operation: str, success: bool = True):
        """Log data access for audit trail"""
        try:
            # Update in-memory access history
            if data_id not in self.access_history:
                self.access_history[data_id] = []
            
            self.access_history[data_id].append((datetime.utcnow(), accessor_id))
            
            # Keep only recent access history (last 100 accesses)
            if len(self.access_history[data_id]) > 100:
                self.access_history[data_id] = self.access_history[data_id][-100:]
            
            # Persist to database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO access_log (data_id, accessor_id, access_time, operation, success)
                    VALUES (?, ?, ?, ?, ?)
                ''', (data_id, accessor_id, datetime.utcnow().isoformat(), operation, success))
                
        except Exception as e:
            self.logger.error(f"Failed to log access: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.guardian_active:
            try:
                time.sleep(self.verification_interval)
                
                # Periodic integrity checks
                with self.guardian_lock:
                    for data_id, record in list(self.integrity_records.items()):
                        # Check access patterns for suspicious behavior
                        if data_id in self.access_history:
                            if self.violation_detector.analyze_access_patterns(self.access_history[data_id]):
                                self.logger.warning(f"Suspicious access pattern detected: {data_id}")
                                if self.auto_quarantine:
                                    self.quarantine_zone.add(data_id)
                                    self.stats['quarantined_items'] += 1
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.guardian_active:
            try:
                time.sleep(3600)  # Run every hour
                
                with self.guardian_lock:
                    # Clean old access logs
                    cutoff = datetime.utcnow() - timedelta(days=7)
                    
                    with sqlite3.connect(str(self.db_path)) as conn:
                        conn.execute('''
                            DELETE FROM access_log 
                            WHERE access_time < ?
                        ''', (cutoff.isoformat(),))
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def _get_recent_violations(self) -> List[Dict[str, Any]]:
        """Get recent integrity violations"""
        violations = []
        
        for record in self.integrity_records.values():
            if (record.status in [IntegrityStatus.CORRUPTED, IntegrityStatus.TAMPERED] and
                record.verified_at and 
                (datetime.utcnow() - record.verified_at).total_seconds() < 3600):  # Last hour
                
                violations.append({
                    'data_id': record.data_id,
                    'status': record.status.value,
                    'violation_type': record.violation_type.value if record.violation_type else None,
                    'error_message': record.error_message,
                    'verified_at': record.verified_at.isoformat()
                })
        
        return violations
    
    def shutdown(self):
        """Shutdown integrity guardian"""
        self.guardian_active = False
        self.logger.info("Data Integrity Guardian shutdown")


# Global data integrity guardian
data_integrity_guardian = DataIntegrityGuardian()


def protect_sensitive_data(data_id: str, data: bytes, data_type: str = "binary") -> bool:
    """Convenience function to protect sensitive data"""
    return data_integrity_guardian.protect_data(data_id, data, data_type)


def verify_data_integrity(data_id: str, current_data: bytes) -> bool:
    """Convenience function to verify data integrity"""
    is_valid, status = data_integrity_guardian.verify_data_integrity(data_id, current_data)
    return is_valid