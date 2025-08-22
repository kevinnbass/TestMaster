#!/usr/bin/env python3
"""
Analytics Delivery Confirmation with Receipt Tracking
====================================================

Provides ultra-reliability through end-to-end delivery confirmation,
receipt tracking, audit trails, and guaranteed delivery mechanisms.

Author: TestMaster Team
"""

import logging
import threading
import time
import sqlite3
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import hmac
import base64


class DeliveryStatus(Enum):
    """Analytics delivery status."""
    PENDING = "pending"           # Delivery initiated
    IN_TRANSIT = "in_transit"     # Being processed
    DELIVERED = "delivered"       # Successfully delivered
    CONFIRMED = "confirmed"       # Receipt confirmed
    FAILED = "failed"             # Delivery failed
    EXPIRED = "expired"           # Delivery expired
    RETRYING = "retrying"         # Being retried


class ReceiptType(Enum):
    """Types of delivery receipts."""
    AUTOMATIC = "automatic"       # System-generated receipt
    MANUAL = "manual"             # User-confirmed receipt
    CALLBACK = "callback"         # Callback-based confirmation
    WEBHOOK = "webhook"           # Webhook notification
    HEARTBEAT = "heartbeat"       # Periodic heartbeat


class DeliveryPriority(Enum):
    """Delivery priority levels."""
    CRITICAL = "critical"         # Immediate delivery required
    HIGH = "high"                 # High priority delivery
    NORMAL = "normal"             # Normal priority
    LOW = "low"                   # Low priority
    BULK = "bulk"                 # Bulk/batch delivery


@dataclass
class DeliveryReceipt:
    """Delivery receipt record."""
    receipt_id: str
    analytics_id: str
    delivery_id: str
    receipt_type: ReceiptType
    timestamp: datetime
    signature: str
    metadata: Dict[str, Any]
    verification_status: str
    processing_time_ms: float


@dataclass
class DeliveryAttempt:
    """Individual delivery attempt."""
    attempt_id: str
    delivery_id: str
    attempt_number: int
    timestamp: datetime
    status: DeliveryStatus
    response_time_ms: float
    error_message: Optional[str]
    retry_reason: Optional[str]


@dataclass
class AnalyticsDelivery:
    """Analytics delivery tracking record."""
    delivery_id: str
    analytics_id: str
    destination: str
    priority: DeliveryPriority
    status: DeliveryStatus
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    attempts: List[DeliveryAttempt]
    receipts: List[DeliveryReceipt]
    metadata: Dict[str, Any]
    checksum: str
    signature: str


class AnalyticsReceiptTracker:
    """Analytics delivery confirmation and receipt tracking system."""
    
    def __init__(
        self,
        db_path: str = "data/receipt_tracking.db",
        secret_key: str = None,
        confirmation_timeout: int = 300,  # 5 minutes
        max_retry_attempts: int = 5
    ):
        """Initialize the receipt tracker."""
        self.db_path = db_path
        self.secret_key = secret_key or self._generate_secret_key()
        self.confirmation_timeout = confirmation_timeout
        self.max_retry_attempts = max_retry_attempts
        
        # Tracking storage
        self.active_deliveries: Dict[str, AnalyticsDelivery] = {}
        self.pending_confirmations: Dict[str, Dict[str, Any]] = {}
        self.delivery_callbacks: Dict[str, Callable] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=6)
        self.running = False
        self.monitor_thread = None
        
        # Statistics
        self.stats = {
            'total_deliveries': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'confirmed_deliveries': 0,
            'expired_deliveries': 0,
            'retry_attempts': 0,
            'avg_confirmation_time_ms': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        
        self.logger.info("Analytics Receipt Tracker initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate a secret key for signatures."""
        return base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def _initialize_database(self):
        """Initialize the receipt tracking database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Deliveries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS deliveries (
                        delivery_id TEXT PRIMARY KEY,
                        analytics_id TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        expires_at TEXT,
                        metadata TEXT,
                        checksum TEXT NOT NULL,
                        signature TEXT NOT NULL
                    )
                """)
                
                # Delivery attempts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS delivery_attempts (
                        attempt_id TEXT PRIMARY KEY,
                        delivery_id TEXT NOT NULL,
                        attempt_number INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        response_time_ms REAL NOT NULL,
                        error_message TEXT,
                        retry_reason TEXT,
                        FOREIGN KEY (delivery_id) REFERENCES deliveries (delivery_id)
                    )
                """)
                
                # Receipts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS receipts (
                        receipt_id TEXT PRIMARY KEY,
                        analytics_id TEXT NOT NULL,
                        delivery_id TEXT NOT NULL,
                        receipt_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        signature TEXT NOT NULL,
                        metadata TEXT,
                        verification_status TEXT NOT NULL,
                        processing_time_ms REAL NOT NULL,
                        FOREIGN KEY (delivery_id) REFERENCES deliveries (delivery_id)
                    )
                """)
                
                # Audit trail table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_trail (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        delivery_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        details TEXT,
                        user_id TEXT,
                        ip_address TEXT
                    )
                """)
                
                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_status ON deliveries(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_created ON deliveries(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_delivery ON delivery_attempts(delivery_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_delivery ON receipts(delivery_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_delivery ON audit_trail(delivery_id)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def initiate_delivery(
        self,
        analytics_id: str,
        destination: str,
        analytics_data: Dict[str, Any],
        priority: DeliveryPriority = DeliveryPriority.NORMAL,
        expiration_minutes: int = 60,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Initiate analytics delivery with tracking."""
        delivery_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(minutes=expiration_minutes) if expiration_minutes > 0 else None
        
        # Calculate checksum for data integrity
        checksum = self._calculate_checksum(analytics_data)
        
        # Create delivery record
        delivery = AnalyticsDelivery(
            delivery_id=delivery_id,
            analytics_id=analytics_id,
            destination=destination,
            priority=priority,
            status=DeliveryStatus.PENDING,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            attempts=[],
            receipts=[],
            metadata=metadata or {},
            checksum=checksum,
            signature=self._generate_signature(delivery_id, analytics_id, checksum)
        )
        
        # Store in memory and database
        with self.lock:
            self.active_deliveries[delivery_id] = delivery
            self.stats['total_deliveries'] += 1
        
        self._save_delivery_to_db(delivery)
        self._log_audit_trail(delivery_id, "delivery_initiated", {
            'analytics_id': analytics_id,
            'destination': destination,
            'priority': priority.value
        })
        
        # Schedule delivery processing
        self.executor.submit(self._process_delivery, delivery_id, analytics_data)
        
        self.logger.info(f"Delivery initiated: {delivery_id} for analytics {analytics_id}")
        return delivery_id
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity."""
        data_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data_json.encode()).hexdigest()
    
    def _generate_signature(self, delivery_id: str, analytics_id: str, checksum: str) -> str:
        """Generate delivery signature for verification."""
        message = f"{delivery_id}:{analytics_id}:{checksum}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _verify_signature(self, delivery_id: str, analytics_id: str, checksum: str, signature: str) -> bool:
        """Verify delivery signature."""
        expected_signature = self._generate_signature(delivery_id, analytics_id, checksum)
        return hmac.compare_digest(signature, expected_signature)
    
    def _save_delivery_to_db(self, delivery: AnalyticsDelivery):
        """Save delivery to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO deliveries 
                    (delivery_id, analytics_id, destination, priority, status,
                     created_at, updated_at, expires_at, metadata, checksum, signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    delivery.delivery_id,
                    delivery.analytics_id,
                    delivery.destination,
                    delivery.priority.value,
                    delivery.status.value,
                    delivery.created_at.isoformat(),
                    delivery.updated_at.isoformat(),
                    delivery.expires_at.isoformat() if delivery.expires_at else None,
                    json.dumps(delivery.metadata),
                    delivery.checksum,
                    delivery.signature
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save delivery to database: {e}")
    
    def _process_delivery(self, delivery_id: str, analytics_data: Dict[str, Any]):
        """Process analytics delivery."""
        try:
            with self.lock:
                delivery = self.active_deliveries.get(delivery_id)
                if not delivery:
                    return
                
                delivery.status = DeliveryStatus.IN_TRANSIT
                delivery.updated_at = datetime.now()
            
            self._save_delivery_to_db(delivery)
            self._log_audit_trail(delivery_id, "processing_started")
            
            # Simulate delivery processing
            attempt = self._create_delivery_attempt(delivery_id, 1)
            
            try:
                # Simulate actual delivery (replace with real delivery logic)
                processing_start = time.time()
                success = self._simulate_delivery(delivery.destination, analytics_data)
                processing_time = (time.time() - processing_start) * 1000
                
                if success:
                    # Update attempt as successful
                    attempt.status = DeliveryStatus.DELIVERED
                    attempt.response_time_ms = processing_time
                    
                    # Update delivery status
                    with self.lock:
                        delivery.status = DeliveryStatus.DELIVERED
                        delivery.updated_at = datetime.now()
                        delivery.attempts.append(attempt)
                    
                    self._save_delivery_to_db(delivery)
                    self._save_attempt_to_db(attempt)
                    self._log_audit_trail(delivery_id, "delivery_successful", {
                        'processing_time_ms': processing_time
                    })
                    
                    # Generate automatic receipt
                    self._generate_automatic_receipt(delivery_id, processing_time)
                    
                    # Schedule confirmation timeout
                    self.executor.submit(self._schedule_confirmation_timeout, delivery_id)
                    
                else:
                    # Delivery failed, schedule retry
                    attempt.status = DeliveryStatus.FAILED
                    attempt.error_message = "Delivery simulation failed"
                    
                    self._handle_delivery_failure(delivery, attempt, "Simulated failure")
                
            except Exception as e:
                attempt.status = DeliveryStatus.FAILED
                attempt.error_message = str(e)
                self._handle_delivery_failure(delivery, attempt, str(e))
                
        except Exception as e:
            self.logger.error(f"Delivery processing failed for {delivery_id}: {e}")
    
    def _simulate_delivery(self, destination: str, analytics_data: Dict[str, Any]) -> bool:
        """Simulate analytics delivery (replace with actual delivery logic)."""
        # Simulate network delay
        time.sleep(0.1 + (hash(destination) % 100) / 1000)
        
        # Simulate 90% success rate
        return (hash(str(analytics_data)) % 10) < 9
    
    def _create_delivery_attempt(self, delivery_id: str, attempt_number: int) -> DeliveryAttempt:
        """Create a new delivery attempt record."""
        return DeliveryAttempt(
            attempt_id=str(uuid.uuid4()),
            delivery_id=delivery_id,
            attempt_number=attempt_number,
            timestamp=datetime.now(),
            status=DeliveryStatus.PENDING,
            response_time_ms=0.0,
            error_message=None,
            retry_reason=None
        )
    
    def _handle_delivery_failure(
        self,
        delivery: AnalyticsDelivery,
        attempt: DeliveryAttempt,
        error_message: str
    ):
        """Handle delivery failure and schedule retry if needed."""
        with self.lock:
            delivery.attempts.append(attempt)
            delivery.updated_at = datetime.now()
            
            # Check if should retry
            if len(delivery.attempts) < self.max_retry_attempts:
                delivery.status = DeliveryStatus.RETRYING
                self.stats['retry_attempts'] += 1
                
                # Schedule retry with exponential backoff
                retry_delay = 2 ** len(delivery.attempts)  # 2, 4, 8, 16 seconds
                self.executor.submit(self._schedule_retry, delivery.delivery_id, retry_delay)
                
                self._log_audit_trail(delivery.delivery_id, "retry_scheduled", {
                    'attempt_number': len(delivery.attempts),
                    'retry_delay_seconds': retry_delay,
                    'error': error_message
                })
                
            else:
                delivery.status = DeliveryStatus.FAILED
                self.stats['failed_deliveries'] += 1
                
                self._log_audit_trail(delivery.delivery_id, "delivery_failed", {
                    'final_error': error_message,
                    'total_attempts': len(delivery.attempts)
                })
        
        self._save_delivery_to_db(delivery)
        self._save_attempt_to_db(attempt)
    
    def _schedule_retry(self, delivery_id: str, delay_seconds: int):
        """Schedule delivery retry."""
        time.sleep(delay_seconds)
        
        with self.lock:
            delivery = self.active_deliveries.get(delivery_id)
            if not delivery or delivery.status != DeliveryStatus.RETRYING:
                return
        
        # Load analytics data for retry (in real implementation, this would be stored)
        analytics_data = {'retry': True, 'delivery_id': delivery_id}  # Placeholder
        
        self.logger.info(f"Retrying delivery {delivery_id}, attempt {len(delivery.attempts) + 1}")
        self._process_delivery(delivery_id, analytics_data)
    
    def _generate_automatic_receipt(self, delivery_id: str, processing_time_ms: float):
        """Generate automatic delivery receipt."""
        with self.lock:
            delivery = self.active_deliveries.get(delivery_id)
            if not delivery:
                return
        
        receipt = DeliveryReceipt(
            receipt_id=str(uuid.uuid4()),
            analytics_id=delivery.analytics_id,
            delivery_id=delivery_id,
            receipt_type=ReceiptType.AUTOMATIC,
            timestamp=datetime.now(),
            signature=self._generate_receipt_signature(delivery_id, delivery.analytics_id),
            metadata={
                'auto_generated': True,
                'delivery_time_ms': processing_time_ms
            },
            verification_status='verified',
            processing_time_ms=processing_time_ms
        )
        
        with self.lock:
            delivery.receipts.append(receipt)
        
        self._save_receipt_to_db(receipt)
        self._log_audit_trail(delivery_id, "receipt_generated", {
            'receipt_id': receipt.receipt_id,
            'receipt_type': receipt.receipt_type.value
        })
        
        self.logger.info(f"Automatic receipt generated for delivery {delivery_id}")
    
    def _generate_receipt_signature(self, delivery_id: str, analytics_id: str) -> str:
        """Generate receipt signature."""
        timestamp = datetime.now().isoformat()
        message = f"{delivery_id}:{analytics_id}:{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def confirm_delivery(
        self,
        delivery_id: str,
        receipt_type: ReceiptType = ReceiptType.MANUAL,
        confirmation_data: Dict[str, Any] = None
    ) -> bool:
        """Confirm delivery receipt."""
        try:
            with self.lock:
                delivery = self.active_deliveries.get(delivery_id)
                if not delivery:
                    self.logger.warning(f"Delivery not found for confirmation: {delivery_id}")
                    return False
                
                if delivery.status not in [DeliveryStatus.DELIVERED, DeliveryStatus.CONFIRMED]:
                    self.logger.warning(f"Cannot confirm delivery in status {delivery.status.value}")
                    return False
            
            # Create confirmation receipt
            receipt = DeliveryReceipt(
                receipt_id=str(uuid.uuid4()),
                analytics_id=delivery.analytics_id,
                delivery_id=delivery_id,
                receipt_type=receipt_type,
                timestamp=datetime.now(),
                signature=self._generate_receipt_signature(delivery_id, delivery.analytics_id),
                metadata=confirmation_data or {},
                verification_status='verified',
                processing_time_ms=(datetime.now() - delivery.created_at).total_seconds() * 1000
            )
            
            with self.lock:
                delivery.status = DeliveryStatus.CONFIRMED
                delivery.updated_at = datetime.now()
                delivery.receipts.append(receipt)
                self.stats['confirmed_deliveries'] += 1
                self.stats['successful_deliveries'] += 1
            
            self._save_delivery_to_db(delivery)
            self._save_receipt_to_db(receipt)
            self._log_audit_trail(delivery_id, "delivery_confirmed", {
                'receipt_id': receipt.receipt_id,
                'receipt_type': receipt_type.value
            })
            
            self.logger.info(f"Delivery confirmed: {delivery_id} with {receipt_type.value} receipt")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to confirm delivery {delivery_id}: {e}")
            return False
    
    def _schedule_confirmation_timeout(self, delivery_id: str):
        """Schedule confirmation timeout check."""
        time.sleep(self.confirmation_timeout)
        
        with self.lock:
            delivery = self.active_deliveries.get(delivery_id)
            if not delivery or delivery.status == DeliveryStatus.CONFIRMED:
                return
            
            # Check if delivery has expired
            if delivery.expires_at and datetime.now() > delivery.expires_at:
                delivery.status = DeliveryStatus.EXPIRED
                delivery.updated_at = datetime.now()
                self.stats['expired_deliveries'] += 1
                
                self._save_delivery_to_db(delivery)
                self._log_audit_trail(delivery_id, "delivery_expired")
                
                self.logger.warning(f"Delivery expired without confirmation: {delivery_id}")
    
    def _save_attempt_to_db(self, attempt: DeliveryAttempt):
        """Save delivery attempt to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO delivery_attempts 
                    (attempt_id, delivery_id, attempt_number, timestamp, status,
                     response_time_ms, error_message, retry_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    attempt.attempt_id,
                    attempt.delivery_id,
                    attempt.attempt_number,
                    attempt.timestamp.isoformat(),
                    attempt.status.value,
                    attempt.response_time_ms,
                    attempt.error_message,
                    attempt.retry_reason
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save attempt to database: {e}")
    
    def _save_receipt_to_db(self, receipt: DeliveryReceipt):
        """Save receipt to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO receipts 
                    (receipt_id, analytics_id, delivery_id, receipt_type, timestamp,
                     signature, metadata, verification_status, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    receipt.receipt_id,
                    receipt.analytics_id,
                    receipt.delivery_id,
                    receipt.receipt_type.value,
                    receipt.timestamp.isoformat(),
                    receipt.signature,
                    json.dumps(receipt.metadata),
                    receipt.verification_status,
                    receipt.processing_time_ms
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save receipt to database: {e}")
    
    def _log_audit_trail(
        self,
        delivery_id: str,
        action: str,
        details: Dict[str, Any] = None,
        user_id: str = None,
        ip_address: str = None
    ):
        """Log audit trail entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_trail 
                    (delivery_id, action, timestamp, details, user_id, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    delivery_id,
                    action,
                    datetime.now().isoformat(),
                    json.dumps(details) if details else None,
                    user_id,
                    ip_address
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to log audit trail: {e}")
    
    def get_delivery_status(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        """Get delivery status and tracking information."""
        with self.lock:
            delivery = self.active_deliveries.get(delivery_id)
            if not delivery:
                return None
            
            return {
                'delivery_id': delivery.delivery_id,
                'analytics_id': delivery.analytics_id,
                'destination': delivery.destination,
                'priority': delivery.priority.value,
                'status': delivery.status.value,
                'created_at': delivery.created_at.isoformat(),
                'updated_at': delivery.updated_at.isoformat(),
                'expires_at': delivery.expires_at.isoformat() if delivery.expires_at else None,
                'attempts_count': len(delivery.attempts),
                'receipts_count': len(delivery.receipts),
                'metadata': delivery.metadata,
                'signature_verified': self._verify_signature(
                    delivery.delivery_id,
                    delivery.analytics_id,
                    delivery.checksum,
                    delivery.signature
                )
            }
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get comprehensive tracking summary."""
        with self.lock:
            active_count = len(self.active_deliveries)
            status_counts = {}
            
            for delivery in self.active_deliveries.values():
                status = delivery.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'timestamp': datetime.now().isoformat(),
                'statistics': self.stats.copy(),
                'active_deliveries': active_count,
                'status_breakdown': status_counts,
                'system_health': {
                    'confirmation_rate': (self.stats['confirmed_deliveries'] / max(1, self.stats['total_deliveries'])) * 100,
                    'success_rate': (self.stats['successful_deliveries'] / max(1, self.stats['total_deliveries'])) * 100,
                    'failure_rate': (self.stats['failed_deliveries'] / max(1, self.stats['total_deliveries'])) * 100
                }
            }
    
    def start_monitoring(self):
        """Start receipt tracking monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_deliveries, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Receipt tracking monitoring started")
    
    def _monitor_deliveries(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Check for expired deliveries
                self._check_expired_deliveries()
                
                # Clean up old completed deliveries
                self._cleanup_old_deliveries()
                
                # Update statistics
                self._update_statistics()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Delivery monitoring error: {e}")
                time.sleep(30)
    
    def _check_expired_deliveries(self):
        """Check for expired deliveries."""
        now = datetime.now()
        
        with self.lock:
            expired_deliveries = []
            for delivery in self.active_deliveries.values():
                if (delivery.expires_at and now > delivery.expires_at and
                    delivery.status not in [DeliveryStatus.CONFIRMED, DeliveryStatus.FAILED, DeliveryStatus.EXPIRED]):
                    expired_deliveries.append(delivery.delivery_id)
            
            for delivery_id in expired_deliveries:
                delivery = self.active_deliveries[delivery_id]
                delivery.status = DeliveryStatus.EXPIRED
                delivery.updated_at = now
                self.stats['expired_deliveries'] += 1
                
                self._save_delivery_to_db(delivery)
                self._log_audit_trail(delivery_id, "delivery_expired")
    
    def _cleanup_old_deliveries(self):
        """Clean up old completed deliveries."""
        cutoff = datetime.now() - timedelta(hours=24)
        
        with self.lock:
            to_remove = []
            for delivery_id, delivery in self.active_deliveries.items():
                if (delivery.status in [DeliveryStatus.CONFIRMED, DeliveryStatus.FAILED, DeliveryStatus.EXPIRED] and
                    delivery.updated_at < cutoff):
                    to_remove.append(delivery_id)
            
            for delivery_id in to_remove:
                del self.active_deliveries[delivery_id]
    
    def _update_statistics(self):
        """Update tracking statistics."""
        # Calculate average confirmation time
        confirmed_times = []
        
        with self.lock:
            for delivery in self.active_deliveries.values():
                if delivery.status == DeliveryStatus.CONFIRMED and delivery.receipts:
                    confirmation_time = (delivery.receipts[-1].timestamp - delivery.created_at).total_seconds() * 1000
                    confirmed_times.append(confirmation_time)
        
        if confirmed_times:
            self.stats['avg_confirmation_time_ms'] = sum(confirmed_times) / len(confirmed_times)
    
    def stop_monitoring(self):
        """Stop receipt tracking monitoring."""
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Receipt tracking monitoring stopped")
    
    def shutdown(self):
        """Shutdown the receipt tracker."""
        self.stop_monitoring()


# Global instance for easy access
receipt_tracker = None

def get_receipt_tracker() -> AnalyticsReceiptTracker:
    """Get the global receipt tracker instance."""
    global receipt_tracker
    if receipt_tracker is None:
        receipt_tracker = AnalyticsReceiptTracker()
    return receipt_tracker


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    tracker = AnalyticsReceiptTracker()
    tracker.start_monitoring()
    
    try:
        # Simulate analytics deliveries
        for i in range(10):
            analytics_data = {
                'test_id': f'test_{i}',
                'timestamp': datetime.now().isoformat(),
                'data': f'Analytics data {i}'
            }
            
            delivery_id = tracker.initiate_delivery(
                analytics_id=f'analytics_{i}',
                destination=f'dashboard_endpoint_{i % 3}',
                analytics_data=analytics_data,
                priority=DeliveryPriority.NORMAL,
                expiration_minutes=30
            )
            
            # Simulate some confirmations
            if i % 3 == 0:
                time.sleep(2)  # Wait a bit
                tracker.confirm_delivery(delivery_id, ReceiptType.MANUAL, {
                    'confirmed_by': 'test_user',
                    'confirmation_method': 'dashboard'
                })
            
            time.sleep(0.5)
        
        # Wait a bit for processing
        time.sleep(5)
        
        # Get tracking summary
        summary = tracker.get_tracking_summary()
        print(json.dumps(summary, indent=2, default=str))
        
    except KeyboardInterrupt:
        print("Stopping receipt tracker...")
    
    finally:
        tracker.shutdown()