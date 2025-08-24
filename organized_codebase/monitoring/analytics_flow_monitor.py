"""
Analytics Flow Monitor
======================

Comprehensive monitoring and logging system for analytics data flow.
Tracks every stage of analytics processing from collection to delivery.

Author: TestMaster Team
"""

import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import hashlib

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class FlowStage(Enum):
    """Analytics flow stages."""
    COLLECTION = "collection"
    VALIDATION = "validation"
    PROCESSING = "processing"
    AGGREGATION = "aggregation"
    OPTIMIZATION = "optimization"
    DELIVERY = "delivery"
    STORAGE = "storage"
    COMPLETE = "complete"

class FlowStatus(Enum):
    """Flow status."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"
    RETRY = "retry"

@dataclass
class FlowEvent:
    """Represents an event in the analytics flow."""
    event_id: str
    stage: FlowStage
    status: FlowStatus
    timestamp: datetime
    duration_ms: float
    data_size_bytes: int
    message: str
    metadata: Dict[str, Any]
    error_details: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class FlowTransaction:
    """Represents a complete analytics flow transaction."""
    transaction_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration_ms: float
    stages_completed: List[FlowStage]
    stages_failed: List[FlowStage]
    events: List[FlowEvent]
    data_checksum: str
    final_status: FlowStatus
    retry_count: int = 0

class AnalyticsFlowMonitor:
    """
    Monitors and logs the complete analytics data flow.
    """
    
    def __init__(self, 
                 log_to_file: bool = True,
                 log_file_path: str = "analytics_flow.log",
                 max_transactions: int = 1000):
        """
        Initialize flow monitor.
        
        Args:
            log_to_file: Enable file logging
            log_file_path: Path to log file
            max_transactions: Maximum transactions to keep in memory
        """
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.max_transactions = max_transactions
        
        # Transaction tracking
        self.active_transactions = {}
        self.completed_transactions = deque(maxlen=max_transactions)
        self.failed_transactions = deque(maxlen=100)
        
        # Flow statistics
        self.flow_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'warning': 0,
            'error': 0,
            'timeout': 0,
            'retry': 0,
            'avg_duration_ms': 0,
            'total_bytes': 0
        })
        
        # Stage timings
        self.stage_timings = defaultdict(list)
        
        # Alerts and thresholds
        self.alert_thresholds = {
            'max_duration_ms': 5000,  # 5 seconds
            'max_retries': 3,
            'error_rate_percent': 10,
            'warning_rate_percent': 20
        }
        
        self.alerts = deque(maxlen=100)
        
        # File logger setup
        if self.log_to_file:
            self._setup_file_logger()
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Analytics Flow Monitor initialized")
    
    def _setup_file_logger(self):
        """Setup file logger for detailed flow logging."""
        self.file_logger = logging.getLogger(f"{__name__}.flow")
        self.file_logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | FLOW | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.file_logger.addHandler(file_handler)
    
    def start_transaction(self, transaction_id: Optional[str] = None) -> str:
        """
        Start a new analytics flow transaction.
        
        Args:
            transaction_id: Optional transaction ID
            
        Returns:
            Transaction ID
        """
        if not transaction_id:
            transaction_id = f"txn_{int(time.time() * 1000000)}"
        
        with self.lock:
            transaction = FlowTransaction(
                transaction_id=transaction_id,
                start_time=datetime.now(),
                end_time=None,
                total_duration_ms=0,
                stages_completed=[],
                stages_failed=[],
                events=[],
                data_checksum="",
                final_status=FlowStatus.SUCCESS
            )
            
            self.active_transactions[transaction_id] = transaction
            
            self._log_event(
                f"TRANSACTION_START | ID: {transaction_id} | Time: {transaction.start_time.isoformat()}"
            )
        
        return transaction_id
    
    def record_stage(self,
                    transaction_id: str,
                    stage: FlowStage,
                    status: FlowStatus,
                    data: Any = None,
                    message: str = "",
                    metadata: Optional[Dict[str, Any]] = None,
                    error: Optional[Exception] = None) -> FlowEvent:
        """
        Record a stage in the analytics flow.
        
        Args:
            transaction_id: Transaction ID
            stage: Flow stage
            status: Stage status
            data: Optional data being processed
            message: Stage message
            metadata: Additional metadata
            error: Optional exception
            
        Returns:
            Flow event
        """
        start_time = time.time()
        
        with self.lock:
            # Get transaction
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                logger.warning(f"Transaction {transaction_id} not found")
                # Create new transaction
                transaction_id = self.start_transaction(transaction_id)
                transaction = self.active_transactions[transaction_id]
            
            # Calculate data size
            data_size = self._calculate_data_size(data)
            
            # Create event
            event = FlowEvent(
                event_id=f"evt_{int(time.time() * 1000000)}",
                stage=stage,
                status=status,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                data_size_bytes=data_size,
                message=message,
                metadata=metadata or {},
                error_details=str(error) if error else None,
                stack_trace=traceback.format_exc() if error else None
            )
            
            # Add event to transaction
            transaction.events.append(event)
            
            # Update transaction status
            if status == FlowStatus.SUCCESS:
                if stage not in transaction.stages_completed:
                    transaction.stages_completed.append(stage)
            elif status in [FlowStatus.ERROR, FlowStatus.TIMEOUT]:
                if stage not in transaction.stages_failed:
                    transaction.stages_failed.append(stage)
                transaction.final_status = status
            elif status == FlowStatus.WARNING and transaction.final_status == FlowStatus.SUCCESS:
                transaction.final_status = FlowStatus.WARNING
            
            # Update statistics
            self._update_statistics(stage, status, event.duration_ms, data_size)
            
            # Log event
            self._log_stage_event(transaction_id, event)
            
            # Check thresholds
            self._check_thresholds(transaction, event)
            
            return event
    
    def complete_transaction(self, 
                           transaction_id: str,
                           status: Optional[FlowStatus] = None) -> FlowTransaction:
        """
        Complete an analytics flow transaction.
        
        Args:
            transaction_id: Transaction ID
            status: Optional final status
            
        Returns:
            Completed transaction
        """
        with self.lock:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                logger.warning(f"Transaction {transaction_id} not found for completion")
                return None
            
            # Complete transaction
            transaction.end_time = datetime.now()
            transaction.total_duration_ms = (
                (transaction.end_time - transaction.start_time).total_seconds() * 1000
            )
            
            if status:
                transaction.final_status = status
            
            # Calculate data checksum
            if transaction.events:
                data_str = json.dumps([e.metadata for e in transaction.events], sort_keys=True)
                transaction.data_checksum = hashlib.md5(data_str.encode()).hexdigest()
            
            # Move to completed or failed
            if transaction.final_status in [FlowStatus.ERROR, FlowStatus.TIMEOUT]:
                self.failed_transactions.append(transaction)
            else:
                self.completed_transactions.append(transaction)
            
            # Remove from active
            del self.active_transactions[transaction_id]
            
            # Log completion
            self._log_event(
                f"TRANSACTION_COMPLETE | ID: {transaction_id} | "
                f"Status: {transaction.final_status.value} | "
                f"Duration: {transaction.total_duration_ms:.2f}ms | "
                f"Stages: {len(transaction.stages_completed)}/{len(FlowStage)}"
            )
            
            return transaction
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate size of data."""
        try:
            if data is None:
                return 0
            elif isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, dict):
                return len(json.dumps(data))
            else:
                return len(str(data))
        except:
            return 0
    
    def _update_statistics(self, stage: FlowStage, status: FlowStatus, 
                          duration_ms: float, data_size: int):
        """Update flow statistics."""
        stats = self.flow_stats[stage.value]
        stats['total'] += 1
        stats[status.value] = stats.get(status.value, 0) + 1
        stats['total_bytes'] += data_size
        
        # Update average duration
        total = stats['total']
        current_avg = stats['avg_duration_ms']
        stats['avg_duration_ms'] = ((current_avg * (total - 1)) + duration_ms) / total
        
        # Track stage timings
        self.stage_timings[stage.value].append(duration_ms)
        if len(self.stage_timings[stage.value]) > 1000:
            self.stage_timings[stage.value] = self.stage_timings[stage.value][-1000:]
    
    def _log_event(self, message: str):
        """Log event to console and file."""
        logger.info(message)
        if self.log_to_file and hasattr(self, 'file_logger'):
            self.file_logger.info(message)
    
    def _log_stage_event(self, transaction_id: str, event: FlowEvent):
        """Log stage event with details."""
        log_level = logging.INFO
        if event.status == FlowStatus.ERROR:
            log_level = logging.ERROR
        elif event.status == FlowStatus.WARNING:
            log_level = logging.WARNING
        
        message = (
            f"STAGE | TXN: {transaction_id} | "
            f"Stage: {event.stage.value} | "
            f"Status: {event.status.value} | "
            f"Duration: {event.duration_ms:.2f}ms | "
            f"Size: {event.data_size_bytes} bytes | "
            f"Message: {event.message}"
        )
        
        logger.log(log_level, message)
        
        if self.log_to_file and hasattr(self, 'file_logger'):
            self.file_logger.log(log_level, message)
            
            if event.error_details:
                self.file_logger.error(f"ERROR_DETAILS | {event.error_details}")
            if event.stack_trace:
                self.file_logger.error(f"STACK_TRACE | {event.stack_trace}")
    
    def _check_thresholds(self, transaction: FlowTransaction, event: FlowEvent):
        """Check thresholds and generate alerts."""
        alerts = []
        
        # Check duration threshold
        if event.duration_ms > self.alert_thresholds['max_duration_ms']:
            alerts.append({
                'type': 'SLOW_STAGE',
                'transaction_id': transaction.transaction_id,
                'stage': event.stage.value,
                'duration_ms': event.duration_ms,
                'threshold_ms': self.alert_thresholds['max_duration_ms']
            })
        
        # Check retry count
        if transaction.retry_count > self.alert_thresholds['max_retries']:
            alerts.append({
                'type': 'EXCESSIVE_RETRIES',
                'transaction_id': transaction.transaction_id,
                'retry_count': transaction.retry_count,
                'threshold': self.alert_thresholds['max_retries']
            })
        
        # Log alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"ALERT | {alert['type']} | {json.dumps(alert)}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                time.sleep(60)  # Check every minute
                
                with self.lock:
                    # Check for stuck transactions
                    current_time = datetime.now()
                    stuck_transactions = []
                    
                    for txn_id, transaction in self.active_transactions.items():
                        age = (current_time - transaction.start_time).total_seconds()
                        if age > 300:  # 5 minutes
                            stuck_transactions.append(txn_id)
                    
                    # Handle stuck transactions
                    for txn_id in stuck_transactions:
                        logger.error(f"STUCK_TRANSACTION | ID: {txn_id} | Forcing completion")
                        self.complete_transaction(txn_id, FlowStatus.TIMEOUT)
                    
                    # Calculate error rates
                    self._calculate_error_rates()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _calculate_error_rates(self):
        """Calculate and check error rates."""
        for stage, stats in self.flow_stats.items():
            total = stats['total']
            if total > 0:
                error_rate = (stats.get('error', 0) / total) * 100
                warning_rate = (stats.get('warning', 0) / total) * 100
                
                if error_rate > self.alert_thresholds['error_rate_percent']:
                    logger.warning(
                        f"HIGH_ERROR_RATE | Stage: {stage} | Rate: {error_rate:.1f}% | "
                        f"Threshold: {self.alert_thresholds['error_rate_percent']}%"
                    )
                
                if warning_rate > self.alert_thresholds['warning_rate_percent']:
                    logger.warning(
                        f"HIGH_WARNING_RATE | Stage: {stage} | Rate: {warning_rate:.1f}% | "
                        f"Threshold: {self.alert_thresholds['warning_rate_percent']}%"
                    )
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get flow monitoring summary."""
        with self.lock:
            # Calculate success rates
            success_rates = {}
            for stage, stats in self.flow_stats.items():
                total = stats['total']
                if total > 0:
                    success_rates[stage] = (stats.get('success', 0) / total) * 100
            
            return {
                'active_transactions': len(self.active_transactions),
                'completed_transactions': len(self.completed_transactions),
                'failed_transactions': len(self.failed_transactions),
                'flow_statistics': dict(self.flow_stats),
                'success_rates': success_rates,
                'recent_alerts': list(self.alerts)[-10:],
                'thresholds': self.alert_thresholds
            }
    
    def get_transaction_details(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed transaction information."""
        with self.lock:
            # Check active transactions
            if transaction_id in self.active_transactions:
                transaction = self.active_transactions[transaction_id]
            else:
                # Check completed transactions
                transaction = None
                for txn in self.completed_transactions:
                    if txn.transaction_id == transaction_id:
                        transaction = txn
                        break
                
                if not transaction:
                    for txn in self.failed_transactions:
                        if txn.transaction_id == transaction_id:
                            transaction = txn
                            break
            
            if transaction:
                return {
                    'transaction_id': transaction.transaction_id,
                    'start_time': transaction.start_time.isoformat(),
                    'end_time': transaction.end_time.isoformat() if transaction.end_time else None,
                    'duration_ms': transaction.total_duration_ms,
                    'status': transaction.final_status.value,
                    'stages_completed': [s.value for s in transaction.stages_completed],
                    'stages_failed': [s.value for s in transaction.stages_failed],
                    'event_count': len(transaction.events),
                    'events': [
                        {
                            'stage': e.stage.value,
                            'status': e.status.value,
                            'timestamp': e.timestamp.isoformat(),
                            'duration_ms': e.duration_ms,
                            'message': e.message
                        }
                        for e in transaction.events
                    ],
                    'data_checksum': transaction.data_checksum,
                    'retry_count': transaction.retry_count
                }
            
            return None
    
    def export_flow_logs(self, format: str = 'json') -> str:
        """Export flow logs."""
        with self.lock:
            data = {
                'export_time': datetime.now().isoformat(),
                'summary': self.get_flow_summary(),
                'completed_transactions': [
                    self.get_transaction_details(t.transaction_id)
                    for t in list(self.completed_transactions)[-100:]
                ],
                'failed_transactions': [
                    self.get_transaction_details(t.transaction_id)
                    for t in list(self.failed_transactions)
                ]
            }
            
            if format == 'json':
                return json.dumps(data, indent=2)
            else:
                # Simple text format
                lines = [
                    f"Analytics Flow Logs - {data['export_time']}",
                    "=" * 60,
                    f"Active Transactions: {data['summary']['active_transactions']}",
                    f"Completed: {data['summary']['completed_transactions']}",
                    f"Failed: {data['summary']['failed_transactions']}",
                    "",
                    "Success Rates by Stage:",
                    "-" * 30
                ]
                
                for stage, rate in data['summary']['success_rates'].items():
                    lines.append(f"  {stage}: {rate:.1f}%")
                
                return "\n".join(lines)
    
    def shutdown(self):
        """Shutdown flow monitor."""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Complete all active transactions
        with self.lock:
            for txn_id in list(self.active_transactions.keys()):
                self.complete_transaction(txn_id, FlowStatus.TIMEOUT)
        
        logger.info("Analytics Flow Monitor shutdown")

# Global flow monitor instance
flow_monitor = AnalyticsFlowMonitor()