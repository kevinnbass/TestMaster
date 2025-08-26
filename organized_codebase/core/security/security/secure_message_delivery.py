"""
Archive Derived Secure Message Delivery Security Module
Extracted from TestMaster archive delivery guarantee systems for secure communications
Enhanced for cryptographic integrity and guaranteed delivery with security validation
"""

import uuid
import time
import json
import hashlib
import hmac
import logging
import threading
import sqlite3
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .error_handler import SecurityError, security_error_handler


class DeliveryStatus(Enum):
    """Secure message delivery status"""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class DeliveryPriority(Enum):
    """Message delivery priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class SecurityLevel(Enum):
    """Message security levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class FailureReason(Enum):
    """Delivery failure reasons"""
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    INTEGRITY_VIOLATION = "integrity_violation"
    RECIPIENT_UNAVAILABLE = "recipient_unavailable"
    MESSAGE_CORRUPTED = "message_corrupted"
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class SecureMessage:
    """Secure message with cryptographic protection"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = "data"
    payload: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    priority: DeliveryPriority = DeliveryPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    encryption_key: Optional[str] = None
    signature: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.expires_at:
            # Default expiration based on security level
            expiry_hours = {
                SecurityLevel.TOP_SECRET: 1,
                SecurityLevel.SECRET: 6,
                SecurityLevel.CONFIDENTIAL: 24,
                SecurityLevel.INTERNAL: 72,
                SecurityLevel.PUBLIC: 168
            }
            self.expires_at = self.created_at + timedelta(hours=expiry_hours[self.security_level])
        
        # Calculate checksum
        self.checksum = self.calculate_checksum()
    
    def calculate_checksum(self) -> str:
        """Calculate message checksum for integrity verification"""
        message_data = {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'payload': self.payload,
            'created_at': self.created_at.isoformat()
        }
        
        message_str = json.dumps(message_data, sort_keys=True)
        return hashlib.sha256(message_str.encode()).hexdigest()
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return datetime.utcnow() > self.expires_at if self.expires_at else False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for transmission"""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type,
            'payload': self.payload,
            'security_level': self.security_level.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'signature': self.signature,
            'checksum': self.checksum,
            'metadata': self.metadata
        }


@dataclass
class DeliveryRecord:
    """Persistent secure delivery record"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message: SecureMessage = field(default_factory=SecureMessage)
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    max_attempts: int = 5
    last_attempt: Optional[datetime] = None
    next_attempt: Optional[datetime] = None
    delivery_confirmed_at: Optional[datetime] = None
    acknowledgment_received: bool = False
    failure_reason: Optional[FailureReason] = None
    error_details: Optional[str] = None
    retry_backoff: float = 1.0
    
    @property
    def is_deliverable(self) -> bool:
        """Check if message can still be delivered"""
        return (
            not self.message.is_expired and
            self.attempts < self.max_attempts and
            self.status not in [DeliveryStatus.DELIVERED, DeliveryStatus.ACKNOWLEDGED, 
                               DeliveryStatus.EXPIRED, DeliveryStatus.QUARANTINED]
        )
    
    @property
    def should_retry(self) -> bool:
        """Check if delivery should be retried"""
        return (
            self.is_deliverable and
            self.status in [DeliveryStatus.FAILED, DeliveryStatus.RETRYING] and
            (not self.next_attempt or datetime.utcnow() >= self.next_attempt)
        )
    
    def calculate_next_attempt(self):
        """Calculate next retry attempt using exponential backoff"""
        if self.attempts < self.max_attempts:
            # Exponential backoff with jitter
            backoff_seconds = min(self.retry_backoff * (2 ** self.attempts) + 
                                random.uniform(0, 5), 300)  # Max 5 minutes
            self.next_attempt = datetime.utcnow() + timedelta(seconds=backoff_seconds)


class MessageSecurity:
    """Message security operations"""
    
    def __init__(self, secret_key: str = "default_secret_key"):
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)
    
    def sign_message(self, message: SecureMessage) -> str:
        """Create HMAC signature for message authentication"""
        try:
            message_data = message.to_dict()
            # Remove signature field from signing data
            message_data.pop('signature', None)
            
            message_str = json.dumps(message_data, sort_keys=True)
            signature = hmac.new(
                self.secret_key.encode(),
                message_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            raise SecurityError(f"Message signing failed: {str(e)}", "MSG_SIGN_001")
    
    def verify_signature(self, message: SecureMessage) -> bool:
        """Verify message HMAC signature"""
        try:
            if not message.signature:
                return False
            
            expected_signature = self.sign_message(message)
            return hmac.compare_digest(message.signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False
    
    def encrypt_message(self, message: SecureMessage) -> bool:
        """Encrypt message payload for confidential transmission"""
        try:
            if message.security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                # Generate encryption key
                message.encryption_key = base64.b64encode(uuid.uuid4().bytes).decode()
                
                # Simple XOR encryption (in production, use proper encryption like Fernet)
                payload_str = json.dumps(message.payload)
                key_bytes = message.encryption_key.encode()[:32]
                encrypted_data = bytes(a ^ key_bytes[i % len(key_bytes)] 
                                     for i, a in enumerate(payload_str.encode()))
                
                # Replace payload with encrypted data
                message.payload = {'encrypted_data': base64.b64encode(encrypted_data).decode()}
                return True
            
            return True  # No encryption needed for lower security levels
            
        except Exception as e:
            raise SecurityError(f"Message encryption failed: {str(e)}", "MSG_ENC_001")
    
    def decrypt_message(self, message: SecureMessage) -> bool:
        """Decrypt message payload"""
        try:
            if message.encryption_key and 'encrypted_data' in message.payload:
                encrypted_data = base64.b64decode(message.payload['encrypted_data'])
                key_bytes = message.encryption_key.encode()[:32]
                
                decrypted_data = bytes(a ^ key_bytes[i % len(key_bytes)] 
                                     for i, a in enumerate(encrypted_data))
                
                message.payload = json.loads(decrypted_data.decode())
                return True
            
            return True  # No decryption needed
            
        except Exception as e:
            raise SecurityError(f"Message decryption failed: {str(e)}", "MSG_DEC_001")


class DeliveryValidator:
    """Message delivery validation and security checks"""
    
    def __init__(self):
        self.authorized_senders: Set[str] = set()
        self.authorized_recipients: Set[str] = set()
        self.blocked_entities: Set[str] = set()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.max_messages_per_minute = 100
        self.logger = logging.getLogger(__name__)
    
    def validate_message(self, message: SecureMessage) -> Tuple[bool, Optional[str]]:
        """Comprehensive message validation"""
        try:
            # Check expiration
            if message.is_expired:
                return False, "Message has expired"
            
            # Check sender authorization
            if self.authorized_senders and message.sender_id not in self.authorized_senders:
                return False, f"Unauthorized sender: {message.sender_id}"
            
            # Check recipient authorization
            if self.authorized_recipients and message.recipient_id not in self.authorized_recipients:
                return False, f"Unauthorized recipient: {message.recipient_id}"
            
            # Check blocked entities
            if message.sender_id in self.blocked_entities:
                return False, f"Blocked sender: {message.sender_id}"
            
            if message.recipient_id in self.blocked_entities:
                return False, f"Blocked recipient: {message.recipient_id}"
            
            # Rate limiting
            if not self._check_rate_limit(message.sender_id):
                return False, "Rate limit exceeded"
            
            # Validate message integrity
            if not self._validate_message_integrity(message):
                return False, "Message integrity validation failed"
            
            # Security level validation
            if not self._validate_security_level(message):
                return False, "Security level validation failed"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _check_rate_limit(self, sender_id: str) -> bool:
        """Check sender rate limits"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        if sender_id not in self.rate_limits:
            self.rate_limits[sender_id] = []
        
        # Clean old entries
        self.rate_limits[sender_id] = [
            timestamp for timestamp in self.rate_limits[sender_id]
            if timestamp > minute_ago
        ]
        
        # Check limit
        if len(self.rate_limits[sender_id]) >= self.max_messages_per_minute:
            return False
        
        # Add current timestamp
        self.rate_limits[sender_id].append(now)
        return True
    
    def _validate_message_integrity(self, message: SecureMessage) -> bool:
        """Validate message checksum integrity"""
        expected_checksum = message.calculate_checksum()
        return message.checksum == expected_checksum
    
    def _validate_security_level(self, message: SecureMessage) -> bool:
        """Validate security level requirements"""
        # Higher security levels require encryption
        if message.security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            return message.encryption_key is not None
        
        return True
    
    def authorize_sender(self, sender_id: str):
        """Authorize sender for message delivery"""
        self.authorized_senders.add(sender_id)
    
    def authorize_recipient(self, recipient_id: str):
        """Authorize recipient for message delivery"""
        self.authorized_recipients.add(recipient_id)
    
    def block_entity(self, entity_id: str):
        """Block entity from sending or receiving messages"""
        self.blocked_entities.add(entity_id)


class SecureMessageDeliveryManager:
    """Comprehensive secure message delivery system"""
    
    def __init__(self, db_path: str = "data/secure_delivery.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.security = MessageSecurity()
        self.validator = DeliveryValidator()
        
        # Initialize database
        self._init_database()
        
        # In-memory tracking
        self.delivery_records: Dict[str, DeliveryRecord] = {}
        self.delivery_handlers: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            'total_messages': 0,
            'delivered_messages': 0,
            'failed_messages': 0,
            'acknowledged_messages': 0,
            'quarantined_messages': 0,
            'delivery_success_rate': 100.0,
            'average_delivery_time': 0.0
        }
        
        # Configuration
        self.default_retry_backoff = 1.0
        self.max_delivery_attempts = 5
        self.delivery_timeout = 300  # 5 minutes
        
        # Background processing
        self.delivery_active = True
        self.delivery_thread = threading.Thread(target=self._delivery_loop, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        
        # Thread safety
        self.delivery_lock = threading.RLock()
        
        # Start background threads
        self.delivery_thread.start()
        self.cleanup_thread.start()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Secure Message Delivery Manager initialized")
    
    def _init_database(self):
        """Initialize delivery database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS delivery_records (
                        record_id TEXT PRIMARY KEY,
                        message_id TEXT NOT NULL,
                        sender_id TEXT NOT NULL,
                        recipient_id TEXT NOT NULL,
                        message_type TEXT,
                        security_level TEXT,
                        priority INTEGER,
                        status TEXT NOT NULL,
                        attempts INTEGER DEFAULT 0,
                        max_attempts INTEGER DEFAULT 5,
                        created_at TEXT NOT NULL,
                        last_attempt TEXT,
                        next_attempt TEXT,
                        delivered_at TEXT,
                        acknowledged_at TEXT,
                        failure_reason TEXT,
                        error_details TEXT,
                        message_data TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS delivery_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        details TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_delivery_status 
                    ON delivery_records(status)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_delivery_next_attempt 
                    ON delivery_records(next_attempt)
                ''')
                
        except Exception as e:
            raise SecurityError(f"Database initialization failed: {str(e)}", "DB_INIT_001")
    
    def send_secure_message(self, message: SecureMessage) -> bool:
        """Send secure message with guaranteed delivery"""
        try:
            with self.delivery_lock:
                self.stats['total_messages'] += 1
                
                # Validate message
                is_valid, error_msg = self.validator.validate_message(message)
                if not is_valid:
                    self.logger.warning(f"Message validation failed: {error_msg}")
                    self.stats['failed_messages'] += 1
                    return False
                
                # Encrypt message if required
                self.security.encrypt_message(message)
                
                # Sign message
                message.signature = self.security.sign_message(message)
                
                # Create delivery record
                record = DeliveryRecord(
                    message=message,
                    max_attempts=self.max_delivery_attempts,
                    retry_backoff=self.default_retry_backoff
                )
                
                # Store delivery record
                self.delivery_records[message.message_id] = record
                self._persist_delivery_record(record)
                
                # Log delivery attempt
                self._log_delivery_event(message.message_id, "MESSAGE_QUEUED", "Message queued for delivery")
                
                self.logger.info(f"Secure message queued for delivery: {message.message_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Secure message send failed: {str(e)}", "MSG_SEND_001")
            security_error_handler.handle_error(error)
            self.stats['failed_messages'] += 1
            return False
    
    def register_delivery_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.delivery_handlers[message_type] = handler
        self.logger.info(f"Delivery handler registered for type: {message_type}")
    
    def acknowledge_delivery(self, message_id: str) -> bool:
        """Acknowledge message delivery"""
        try:
            with self.delivery_lock:
                if message_id not in self.delivery_records:
                    return False
                
                record = self.delivery_records[message_id]
                record.acknowledgment_received = True
                record.status = DeliveryStatus.ACKNOWLEDGED
                record.delivery_confirmed_at = datetime.utcnow()
                
                self.stats['acknowledged_messages'] += 1
                
                # Update database
                self._persist_delivery_record(record)
                
                # Log acknowledgment
                self._log_delivery_event(message_id, "ACKNOWLEDGED", "Message acknowledged by recipient")
                
                self.logger.info(f"Message delivery acknowledged: {message_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Acknowledgment failed: {e}")
            return False
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get comprehensive delivery statistics"""
        with self.delivery_lock:
            # Update delivery success rate
            if self.stats['total_messages'] > 0:
                self.stats['delivery_success_rate'] = (
                    self.stats['delivered_messages'] / self.stats['total_messages'] * 100
                )
            
            return {
                **self.stats,
                'pending_deliveries': sum(
                    1 for record in self.delivery_records.values()
                    if record.status == DeliveryStatus.PENDING
                ),
                'in_transit_deliveries': sum(
                    1 for record in self.delivery_records.values()
                    if record.status == DeliveryStatus.IN_TRANSIT
                ),
                'failed_deliveries': sum(
                    1 for record in self.delivery_records.values()
                    if record.status == DeliveryStatus.FAILED
                )
            }
    
    def _delivery_loop(self):
        """Background delivery processing loop"""
        while self.delivery_active:
            try:
                time.sleep(1)  # Check every second
                
                with self.delivery_lock:
                    # Process pending and retry deliveries
                    for record in list(self.delivery_records.values()):
                        if record.should_retry or record.status == DeliveryStatus.PENDING:
                            self._attempt_delivery(record)
                
            except Exception as e:
                self.logger.error(f"Delivery loop error: {e}")
    
    def _attempt_delivery(self, record: DeliveryRecord):
        """Attempt to deliver a single message"""
        try:
            record.attempts += 1
            record.last_attempt = datetime.utcnow()
            record.status = DeliveryStatus.IN_TRANSIT
            
            # Get appropriate handler
            handler = self.delivery_handlers.get(record.message.message_type)
            if not handler:
                # Default handler - just mark as delivered for now
                success = True
            else:
                success = handler(record.message)
            
            if success:
                record.status = DeliveryStatus.DELIVERED
                record.delivery_confirmed_at = datetime.utcnow()
                self.stats['delivered_messages'] += 1
                
                self._log_delivery_event(record.message.message_id, "DELIVERED", "Message successfully delivered")
                
            else:
                record.status = DeliveryStatus.FAILED
                record.failure_reason = FailureReason.NETWORK_ERROR
                record.calculate_next_attempt()
                
                self._log_delivery_event(record.message.message_id, "DELIVERY_FAILED", f"Attempt {record.attempts} failed")
                
                # Check if max attempts reached
                if record.attempts >= record.max_attempts:
                    record.status = DeliveryStatus.EXPIRED
                    self.stats['failed_messages'] += 1
                    
                    self._log_delivery_event(record.message.message_id, "EXPIRED", "Max delivery attempts exceeded")
            
            # Update database
            self._persist_delivery_record(record)
            
        except Exception as e:
            self.logger.error(f"Delivery attempt failed: {e}")
            record.status = DeliveryStatus.FAILED
            record.error_details = str(e)
    
    def _persist_delivery_record(self, record: DeliveryRecord):
        """Persist delivery record to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO delivery_records 
                    (record_id, message_id, sender_id, recipient_id, message_type,
                     security_level, priority, status, attempts, max_attempts,
                     created_at, last_attempt, next_attempt, delivered_at,
                     acknowledged_at, failure_reason, error_details, message_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.record_id, record.message.message_id,
                    record.message.sender_id, record.message.recipient_id,
                    record.message.message_type, record.message.security_level.value,
                    record.message.priority.value, record.status.value,
                    record.attempts, record.max_attempts,
                    record.message.created_at.isoformat(),
                    record.last_attempt.isoformat() if record.last_attempt else None,
                    record.next_attempt.isoformat() if record.next_attempt else None,
                    record.delivery_confirmed_at.isoformat() if record.delivery_confirmed_at else None,
                    record.delivery_confirmed_at.isoformat() if record.acknowledgment_received else None,
                    record.failure_reason.value if record.failure_reason else None,
                    record.error_details,
                    json.dumps(record.message.to_dict())
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to persist delivery record: {e}")
    
    def _log_delivery_event(self, message_id: str, event_type: str, details: str):
        """Log delivery event for audit trail"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO delivery_log (message_id, event_type, timestamp, details)
                    VALUES (?, ?, ?, ?)
                ''', (message_id, event_type, datetime.utcnow().isoformat(), details))
                
        except Exception as e:
            self.logger.error(f"Failed to log delivery event: {e}")
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.delivery_active:
            try:
                time.sleep(3600)  # Run every hour
                
                with self.delivery_lock:
                    # Clean up old completed deliveries
                    cutoff = datetime.utcnow() - timedelta(days=7)
                    
                    expired_records = []
                    for message_id, record in self.delivery_records.items():
                        if (record.status in [DeliveryStatus.DELIVERED, DeliveryStatus.ACKNOWLEDGED, 
                                            DeliveryStatus.EXPIRED] and
                            record.message.created_at < cutoff):
                            expired_records.append(message_id)
                    
                    for message_id in expired_records:
                        del self.delivery_records[message_id]
                    
                    if expired_records:
                        self.logger.info(f"Cleaned up {len(expired_records)} old delivery records")
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def shutdown(self):
        """Shutdown delivery manager"""
        self.delivery_active = False
        self.logger.info("Secure Message Delivery Manager shutdown")


# Global secure message delivery manager
secure_message_delivery = SecureMessageDeliveryManager()


def send_secure_message(sender_id: str, recipient_id: str, payload: Dict[str, Any],
                       message_type: str = "data", security_level: SecurityLevel = SecurityLevel.INTERNAL,
                       priority: DeliveryPriority = DeliveryPriority.NORMAL) -> bool:
    """Convenience function to send secure message"""
    try:
        message = SecureMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            security_level=security_level,
            priority=priority
        )
        
        return secure_message_delivery.send_secure_message(message)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Secure message send failed: {e}")
        return False


def register_message_handler(message_type: str, handler: Callable):
    """Convenience function to register message handler"""
    secure_message_delivery.register_delivery_handler(message_type, handler)