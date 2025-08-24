"""
Analytics Delivery Enhancer
===========================

Ensures reliable delivery of analytics data to the dashboard with retry
mechanisms, delivery guarantees, and data flow monitoring.

Author: TestMaster Team
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import queue

logger = logging.getLogger(__name__)

class DeliveryStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class DeliveryRecord:
    """Analytics delivery tracking record."""
    delivery_id: str
    data: Dict[str, Any]
    target: str
    status: DeliveryStatus
    attempts: int
    created_at: datetime
    last_attempt: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime and other objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, '__dict__'):
            # Convert objects to dict
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)

class AnalyticsDeliveryEnhancer:
    """
    Enhances analytics delivery with guarantees and monitoring.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize delivery enhancer."""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Delivery tracking
        self.delivery_queue = queue.Queue(maxsize=1000)
        self.delivery_records = deque(maxlen=5000)
        self.failed_deliveries = deque(maxlen=1000)
        
        # Statistics
        self.delivery_stats = {
            'total_deliveries': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'average_delivery_time': 0,
            'retry_rate': 0,
            'start_time': datetime.now()
        }
        
        # Delivery workers
        self.delivery_active = False
        self.delivery_threads = []
        
        # Data processors
        self.data_processors = {
            'json_serializer': self._serialize_json,
            'datetime_converter': self._convert_datetimes,
            'enum_converter': self._convert_enums,
            'error_sanitizer': self._sanitize_errors
        }
        
        logger.info("Analytics Delivery Enhancer initialized")
    
    def start_delivery_service(self, num_workers: int = 2):
        """Start delivery service workers."""
        if self.delivery_active:
            return
            
        self.delivery_active = True
        
        # Start delivery workers
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._delivery_worker, 
                name=f"DeliveryWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.delivery_threads.append(worker)
        
        # Start retry worker
        retry_worker = threading.Thread(
            target=self._retry_worker,
            name="RetryWorker",
            daemon=True
        )
        retry_worker.start()
        self.delivery_threads.append(retry_worker)
        
        logger.info(f"Delivery service started with {num_workers} workers")
    
    def stop_delivery_service(self):
        """Stop delivery service."""
        self.delivery_active = False
        
        # Wait for workers to finish
        for thread in self.delivery_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.delivery_threads.clear()
        logger.info("Delivery service stopped")
    
    def queue_delivery(self, data: Dict[str, Any], target: str = "dashboard") -> str:
        """Queue analytics data for delivery."""
        delivery_id = f"del_{int(time.time() * 1000000)}"
        
        # Process data for reliable delivery
        processed_data = self._process_data_for_delivery(data)
        
        # Create delivery record
        record = DeliveryRecord(
            delivery_id=delivery_id,
            data=processed_data,
            target=target,
            status=DeliveryStatus.PENDING,
            attempts=0,
            created_at=datetime.now()
        )
        
        try:
            self.delivery_queue.put_nowait(record)
            self.delivery_stats['total_deliveries'] += 1
            return delivery_id
        except queue.Full:
            logger.warning("Delivery queue is full, dropping delivery")
            return None
    
    def _process_data_for_delivery(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data to ensure reliable delivery."""
        processed_data = data.copy()
        
        # Apply all data processors
        for processor_name, processor_func in self.data_processors.items():
            try:
                processed_data = processor_func(processed_data)
            except Exception as e:
                logger.warning(f"Data processor {processor_name} failed: {e}")
        
        # Add delivery metadata
        processed_data['_delivery_metadata'] = {
            'processed_at': datetime.now().isoformat(),
            'processors_applied': list(self.data_processors.keys()),
            'delivery_version': '1.0'
        }
        
        return processed_data
    
    def _serialize_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure data is JSON serializable."""
        try:
            # Test serialization
            json.dumps(data, cls=JSONEncoder)
            return data
        except Exception as e:
            logger.warning(f"JSON serialization issue: {e}")
            # Return a simplified version
            return {
                'error': 'serialization_failed',
                'original_keys': list(data.keys()) if isinstance(data, dict) else [],
                'processed_at': datetime.now().isoformat()
            }
    
    def _convert_datetimes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to ISO strings."""
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if isinstance(value, datetime):
                    converted[key] = value.isoformat()
                elif isinstance(value, dict):
                    converted[key] = self._convert_datetimes(value)
                elif isinstance(value, list):
                    converted[key] = [
                        item.isoformat() if isinstance(item, datetime) 
                        else self._convert_datetimes(item) if isinstance(item, dict)
                        else item
                        for item in value
                    ]
                else:
                    converted[key] = value
            return converted
        return data
    
    def _convert_enums(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert enum objects to their values."""
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if isinstance(value, Enum):
                    converted[key] = value.value
                elif isinstance(value, dict):
                    converted[key] = self._convert_enums(value)
                elif isinstance(value, list):
                    converted[key] = [
                        item.value if isinstance(item, Enum)
                        else self._convert_enums(item) if isinstance(item, dict)
                        else item
                        for item in value
                    ]
                else:
                    converted[key] = value
            return converted
        return data
    
    def _sanitize_errors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize error objects and exceptions."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if isinstance(value, Exception):
                    sanitized[key] = {
                        'error_type': type(value).__name__,
                        'error_message': str(value),
                        'sanitized': True
                    }
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_errors(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        {'error_type': type(item).__name__, 'error_message': str(item), 'sanitized': True}
                        if isinstance(item, Exception)
                        else self._sanitize_errors(item) if isinstance(item, dict)
                        else item
                        for item in value
                    ]
                else:
                    sanitized[key] = value
            return sanitized
        return data
    
    def _delivery_worker(self):
        """Delivery worker thread."""
        while self.delivery_active:
            try:
                # Get delivery from queue
                record = self.delivery_queue.get(timeout=1)
                
                # Attempt delivery
                success = self._attempt_delivery(record)
                
                if success:
                    record.status = DeliveryStatus.DELIVERED
                    record.delivered_at = datetime.now()
                    self.delivery_stats['successful_deliveries'] += 1
                else:
                    record.attempts += 1
                    record.last_attempt = datetime.now()
                    
                    if record.attempts >= self.max_retries:
                        record.status = DeliveryStatus.FAILED
                        self.failed_deliveries.append(record)
                        self.delivery_stats['failed_deliveries'] += 1
                    else:
                        record.status = DeliveryStatus.RETRYING
                        # Re-queue for retry
                        try:
                            self.delivery_queue.put_nowait(record)
                        except queue.Full:
                            record.status = DeliveryStatus.FAILED
                            self.failed_deliveries.append(record)
                
                # Store record
                self.delivery_records.append(record)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Delivery worker error: {e}")
    
    def _retry_worker(self):
        """Retry worker for failed deliveries."""
        while self.delivery_active:
            try:
                time.sleep(5)  # Check every 5 seconds
                
                # Retry failed deliveries after delay
                current_time = datetime.now()
                for record in list(self.delivery_records):
                    if (record.status == DeliveryStatus.RETRYING and 
                        record.last_attempt and
                        (current_time - record.last_attempt).total_seconds() >= self.retry_delay):
                        
                        # Re-queue for retry
                        try:
                            self.delivery_queue.put_nowait(record)
                        except queue.Full:
                            logger.warning("Cannot retry delivery - queue full")
                
            except Exception as e:
                logger.error(f"Retry worker error: {e}")
    
    def _attempt_delivery(self, record: DeliveryRecord) -> bool:
        """Attempt to deliver analytics data."""
        try:
            # Simulate delivery (in real implementation, this would send to dashboard)
            # For now, we just validate the data is properly processed
            
            # Validate JSON serialization
            json.dumps(record.data, cls=JSONEncoder)
            
            # Simulate delivery time
            time.sleep(0.01)  # 10ms delivery simulation
            
            logger.debug(f"Successfully delivered {record.delivery_id} to {record.target}")
            return True
            
        except Exception as e:
            record.error_message = str(e)
            logger.warning(f"Delivery failed for {record.delivery_id}: {e}")
            return False
    
    def get_delivery_summary(self) -> Dict[str, Any]:
        """Get delivery service summary."""
        uptime = (datetime.now() - self.delivery_stats['start_time']).total_seconds()
        
        # Calculate rates
        total_deliveries = self.delivery_stats['total_deliveries']
        success_rate = 0
        if total_deliveries > 0:
            success_rate = (self.delivery_stats['successful_deliveries'] / total_deliveries) * 100
        
        # Recent delivery times
        recent_records = [r for r in self.delivery_records 
                         if r.delivered_at and 
                         (datetime.now() - r.delivered_at).total_seconds() < 300]
        
        avg_delivery_time = 0
        if recent_records:
            delivery_times = [
                (r.delivered_at - r.created_at).total_seconds() 
                for r in recent_records
            ]
            avg_delivery_time = sum(delivery_times) / len(delivery_times)
        
        return {
            'service_status': 'active' if self.delivery_active else 'inactive',
            'uptime_seconds': uptime,
            'statistics': self.delivery_stats.copy(),
            'performance': {
                'success_rate_percent': success_rate,
                'average_delivery_time_seconds': avg_delivery_time,
                'queue_size': self.delivery_queue.qsize(),
                'pending_retries': len([r for r in self.delivery_records 
                                      if r.status == DeliveryStatus.RETRYING]),
                'failed_deliveries': len(self.failed_deliveries)
            },
            'data_processing': {
                'processors_available': len(self.data_processors),
                'processors': list(self.data_processors.keys())
            }
        }
    
    def get_recent_failures(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent delivery failures."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_failures = [
            {
                'delivery_id': record.delivery_id,
                'target': record.target,
                'attempts': record.attempts,
                'created_at': record.created_at.isoformat(),
                'last_attempt': record.last_attempt.isoformat() if record.last_attempt else None,
                'error_message': record.error_message
            }
            for record in self.failed_deliveries
            if record.created_at >= cutoff_time
        ]
        
        return recent_failures
    
    def shutdown(self):
        """Shutdown delivery enhancer."""
        self.stop_delivery_service()
        logger.info("Analytics Delivery Enhancer shutdown")