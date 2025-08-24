"""
Analytics Delivery Verification Loop
=====================================

Comprehensive verification system that continuously tests analytics delivery
to ensure 100% reliability and immediate detection of any delivery failures.

Author: TestMaster Team
"""

import logging
import time
import threading
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple, Set
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import requests
import asyncio

logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    """Verification status types."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

class DeliveryMethod(Enum):
    """Delivery method types."""
    DIRECT = "direct"
    BATCH = "batch"
    HEARTBEAT = "heartbeat"
    FALLBACK = "fallback"
    EMERGENCY = "emergency"

@dataclass
class VerificationTest:
    """Represents a delivery verification test."""
    test_id: str
    test_type: str
    payload: Dict[str, Any]
    delivery_method: DeliveryMethod
    created_at: datetime
    expected_delivery_time: datetime
    status: VerificationStatus
    delivery_confirmed: bool = False
    confirmation_time: Optional[datetime] = None
    attempts: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class AnalyticsDeliveryVerifier:
    """
    Continuous verification system for analytics delivery.
    """
    
    def __init__(self,
                 aggregator=None,
                 dashboard_url: str = "http://localhost:5000",
                 verification_interval: float = 10.0,
                 max_pending_tests: int = 100):
        """
        Initialize delivery verifier.
        
        Args:
            aggregator: Analytics aggregator instance
            dashboard_url: Dashboard URL for verification
            verification_interval: Seconds between verification cycles
            max_pending_tests: Maximum pending tests to track
        """
        self.aggregator = aggregator
        self.dashboard_url = dashboard_url
        self.verification_interval = verification_interval
        self.max_pending_tests = max_pending_tests
        
        # Test tracking
        self.pending_tests: Dict[str, VerificationTest] = {}
        self.completed_tests: deque = deque(maxlen=1000)  # Keep last 1000
        self.test_history: deque = deque(maxlen=10000)   # Detailed history
        
        # Verification endpoints
        self.verification_endpoints = {
            'health': f"{dashboard_url}/api/health/live",
            'analytics': f"{dashboard_url}/api/analytics/metrics",
            'monitoring': f"{dashboard_url}/api/monitoring/robustness",
            'heartbeat': f"{dashboard_url}/api/monitoring/heartbeat",
            'flow': f"{dashboard_url}/api/monitoring/flow"
        }
        
        # Test patterns for different scenarios
        self.test_patterns = {
            'basic_delivery': {
                'type': 'basic_test',
                'payload_size': 'small',
                'priority': 'normal',
                'expected_delivery_ms': 1000
            },
            'high_priority': {
                'type': 'priority_test',
                'payload_size': 'medium',
                'priority': 'high',
                'expected_delivery_ms': 500
            },
            'large_payload': {
                'type': 'stress_test',
                'payload_size': 'large',
                'priority': 'normal',
                'expected_delivery_ms': 3000
            },
            'batch_test': {
                'type': 'batch_test',
                'payload_size': 'small',
                'priority': 'low',
                'expected_delivery_ms': 5000
            },
            'failover_test': {
                'type': 'failover_test',
                'payload_size': 'medium',
                'priority': 'critical',
                'expected_delivery_ms': 2000
            }
        }
        
        # Statistics
        self.stats = {
            'total_tests': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'timeout_deliveries': 0,
            'avg_delivery_time': 0.0,
            'delivery_success_rate': 100.0,
            'tests_per_method': defaultdict(int),
            'failures_per_method': defaultdict(int),
            'last_successful_test': None,
            'last_failed_test': None,
            'verification_cycles': 0,
            'alerts_triggered': 0
        }
        
        # Configuration
        self.timeout_threshold = 30.0  # seconds
        self.retry_attempts = 3
        self.failure_threshold = 0.05  # 5% failure rate triggers alert
        self.critical_failure_threshold = 0.10  # 10% failure rate is critical
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Verification thread
        self.verification_active = True
        self.verification_thread = threading.Thread(
            target=self._verification_loop,
            daemon=True
        )
        self.verification_thread.start()
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Analytics Delivery Verifier initialized")
    
    def add_alert_handler(self, handler: Callable[[str, str, str], None]):
        """
        Add alert handler.
        
        Args:
            handler: Function(alert_type, message, severity)
        """
        self.alert_handlers.append(handler)
    
    def create_verification_test(self,
                                test_type: str = "basic_delivery",
                                delivery_method: DeliveryMethod = DeliveryMethod.DIRECT,
                                custom_payload: Optional[Dict] = None) -> str:
        """
        Create a new verification test.
        
        Args:
            test_type: Type of test to perform
            delivery_method: How to deliver the test payload
            custom_payload: Custom payload data
            
        Returns:
            Test ID
        """
        with self.lock:
            test_id = f"verify_{int(time.time() * 1000000)}"
            
            # Get test pattern
            pattern = self.test_patterns.get(test_type, self.test_patterns['basic_delivery'])
            
            # Create test payload
            if custom_payload:
                payload = custom_payload
            else:
                payload = self._generate_test_payload(pattern)
            
            # Add verification markers
            payload.update({
                'verification_test': True,
                'test_id': test_id,
                'test_type': test_type,
                'created_at': datetime.now().isoformat(),
                'verification_token': self._generate_verification_token(test_id)
            })
            
            # Calculate expected delivery time
            expected_ms = pattern['expected_delivery_ms']
            expected_delivery = datetime.now() + timedelta(milliseconds=expected_ms)
            
            # Create test
            test = VerificationTest(
                test_id=test_id,
                test_type=test_type,
                payload=payload,
                delivery_method=delivery_method,
                created_at=datetime.now(),
                expected_delivery_time=expected_delivery,
                status=VerificationStatus.PENDING
            )
            
            # Store test
            self.pending_tests[test_id] = test
            
            # Update statistics
            self.stats['total_tests'] += 1
            self.stats['tests_per_method'][delivery_method.value] += 1
            
            logger.debug(f"Created verification test {test_id} ({test_type})")
            
            return test_id
    
    def send_verification_test(self, test_id: str) -> bool:
        """
        Send verification test using specified delivery method.
        
        Args:
            test_id: Test ID to send
            
        Returns:
            Success status
        """
        with self.lock:
            if test_id not in self.pending_tests:
                logger.error(f"Test {test_id} not found")
                return False
            
            test = self.pending_tests[test_id]
            test.attempts += 1
            
            try:
                success = False
                
                if test.delivery_method == DeliveryMethod.DIRECT:
                    success = self._send_direct_test(test)
                elif test.delivery_method == DeliveryMethod.BATCH:
                    success = self._send_batch_test(test)
                elif test.delivery_method == DeliveryMethod.HEARTBEAT:
                    success = self._send_heartbeat_test(test)
                elif test.delivery_method == DeliveryMethod.FALLBACK:
                    success = self._send_fallback_test(test)
                elif test.delivery_method == DeliveryMethod.EMERGENCY:
                    success = self._send_emergency_test(test)
                
                if success:
                    test.status = VerificationStatus.PENDING
                    logger.info(f"Sent verification test {test_id} via {test.delivery_method.value}")
                else:
                    test.status = VerificationStatus.FAILED
                    test.errors.append(f"Failed to send via {test.delivery_method.value}")
                    
                return success
                
            except Exception as e:
                test.status = VerificationStatus.FAILED
                test.errors.append(str(e))
                logger.error(f"Error sending test {test_id}: {e}")
                return False
    
    def _send_direct_test(self, test: VerificationTest) -> bool:
        """Send test directly through analytics aggregator."""
        if not self.aggregator:
            return False
        
        try:
            # Use flow monitor to track delivery
            if hasattr(self.aggregator, 'flow_monitor'):
                from ..core.analytics_flow_monitor import FlowStage, FlowStatus
                
                transaction_id = self.aggregator.flow_monitor.start_transaction()
                
                # Record test delivery
                self.aggregator.flow_monitor.record_stage(
                    transaction_id,
                    FlowStage.COLLECTION,
                    FlowStatus.SUCCESS,
                    data=test.payload,
                    message=f"Verification test {test.test_id}"
                )
                
                # Complete transaction
                self.aggregator.flow_monitor.complete_transaction(transaction_id)
                
                # Store transaction ID for verification
                test.payload['transaction_id'] = transaction_id
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Direct test delivery failed: {e}")
            return False
    
    def _send_batch_test(self, test: VerificationTest) -> bool:
        """Send test through batch processor."""
        if not self.aggregator or not hasattr(self.aggregator, 'batch_processor'):
            return False
        
        try:
            from ..core.analytics_batch_processor import BatchPriority
            
            # Determine priority
            priority_map = {
                'critical': BatchPriority.CRITICAL,
                'high': BatchPriority.HIGH,
                'normal': BatchPriority.NORMAL,
                'low': BatchPriority.LOW
            }
            
            pattern = self.test_patterns.get(test.test_type, {})
            priority_str = pattern.get('priority', 'normal')
            priority = priority_map.get(priority_str, BatchPriority.NORMAL)
            
            # Add to batch
            added = self.aggregator.batch_processor.add_item(test.payload, priority)
            
            return added
            
        except Exception as e:
            logger.error(f"Batch test delivery failed: {e}")
            return False
    
    def _send_heartbeat_test(self, test: VerificationTest) -> bool:
        """Send test through heartbeat monitor."""
        if not self.aggregator or not hasattr(self.aggregator, 'heartbeat_monitor'):
            return False
        
        try:
            # Send via heartbeat analytics delivery
            delivery_id = self.aggregator.heartbeat_monitor.send_analytics(
                test.payload,
                endpoint='main_dashboard',
                strategy='direct',
                priority=8
            )
            
            test.payload['delivery_id'] = delivery_id
            return True
            
        except Exception as e:
            logger.error(f"Heartbeat test delivery failed: {e}")
            return False
    
    def _send_fallback_test(self, test: VerificationTest) -> bool:
        """Send test through fallback system."""
        if not self.aggregator or not hasattr(self.aggregator, 'fallback_system'):
            return False
        
        try:
            # Use fallback system for delivery
            success = self.aggregator.fallback_system.deliver_analytics(
                test.payload,
                force_fallback=True
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Fallback test delivery failed: {e}")
            return False
    
    def _send_emergency_test(self, test: VerificationTest) -> bool:
        """Send test through emergency delivery path."""
        try:
            # Direct HTTP POST to dashboard
            response = requests.post(
                f"{self.dashboard_url}/api/analytics/emergency",
                json=test.payload,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Emergency test delivery failed: {e}")
            return False
    
    def check_delivery_confirmation(self, test_id: str) -> bool:
        """
        Check if test delivery was confirmed.
        
        Args:
            test_id: Test ID to check
            
        Returns:
            True if confirmed, False otherwise
        """
        if test_id not in self.pending_tests:
            return False
        
        test = self.pending_tests[test_id]
        
        try:
            # Check various endpoints for our test data
            confirmed = (
                self._check_analytics_endpoint(test) or
                self._check_monitoring_endpoint(test) or
                self._check_flow_endpoint(test) or
                self._check_heartbeat_endpoint(test)
            )
            
            if confirmed and not test.delivery_confirmed:
                test.delivery_confirmed = True
                test.confirmation_time = datetime.now()
                test.status = VerificationStatus.SUCCESS
                
                # Calculate delivery time
                delivery_time = (test.confirmation_time - test.created_at).total_seconds()
                self._update_delivery_stats(delivery_time, True)
                
                logger.info(f"Delivery confirmed for test {test_id} in {delivery_time:.2f}s")
            
            return confirmed
            
        except Exception as e:
            logger.error(f"Confirmation check failed for {test_id}: {e}")
            return False
    
    def _check_analytics_endpoint(self, test: VerificationTest) -> bool:
        """Check analytics endpoint for test data."""
        try:
            response = requests.get(
                self.verification_endpoints['analytics'],
                timeout=5
            )
            
            if response.status_code == 200:
                # Check if our test token appears in response
                content = response.text
                verification_token = test.payload.get('verification_token', '')
                
                return verification_token in content
            
            return False
            
        except Exception:
            return False
    
    def _check_monitoring_endpoint(self, test: VerificationTest) -> bool:
        """Check monitoring endpoint for test data."""
        try:
            response = requests.get(
                self.verification_endpoints['monitoring'],
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check monitoring data for our test
                monitoring = data.get('monitoring', {})
                
                # Look for test in various sections
                for section in monitoring.values():
                    if isinstance(section, dict) and self._contains_test_token(section, test):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _check_flow_endpoint(self, test: VerificationTest) -> bool:
        """Check flow monitoring for test transaction."""
        try:
            response = requests.get(
                self.verification_endpoints['flow'],
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                flow_data = data.get('flow_monitor', {})
                
                # Check for our transaction ID
                transaction_id = test.payload.get('transaction_id')
                if transaction_id:
                    transactions = flow_data.get('recent_transactions', [])
                    for txn in transactions:
                        if txn.get('transaction_id') == transaction_id:
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _check_heartbeat_endpoint(self, test: VerificationTest) -> bool:
        """Check heartbeat monitoring for test delivery."""
        try:
            response = requests.get(
                self.verification_endpoints['heartbeat'],
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                heartbeat_data = data.get('heartbeat', {})
                
                # Check for our delivery ID
                delivery_id = test.payload.get('delivery_id')
                if delivery_id:
                    deliveries = heartbeat_data.get('recent_deliveries', [])
                    for delivery in deliveries:
                        if delivery.get('delivery_id') == delivery_id:
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _contains_test_token(self, data: Any, test: VerificationTest) -> bool:
        """Check if data contains our test verification token."""
        verification_token = test.payload.get('verification_token', '')
        
        if isinstance(data, dict):
            for value in data.values():
                if self._contains_test_token(value, test):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_test_token(item, test):
                    return True
        elif isinstance(data, str):
            return verification_token in data
        
        return False
    
    def _generate_test_payload(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test payload based on pattern."""
        base_payload = {
            'timestamp': datetime.now().isoformat(),
            'source': 'delivery_verifier',
            'type': 'verification_test'
        }
        
        # Add payload size variations
        payload_size = pattern.get('payload_size', 'small')
        
        if payload_size == 'small':
            base_payload['data'] = {'test_value': 42}
        elif payload_size == 'medium':
            base_payload['data'] = {
                'test_values': list(range(100)),
                'test_string': 'x' * 1000
            }
        elif payload_size == 'large':
            base_payload['data'] = {
                'large_array': list(range(10000)),
                'large_string': 'x' * 50000
            }
        
        return base_payload
    
    def _generate_verification_token(self, test_id: str) -> str:
        """Generate unique verification token."""
        unique_data = f"{test_id}_{datetime.now().isoformat()}_{uuid.uuid4()}"
        return hashlib.md5(unique_data.encode()).hexdigest()
    
    def _update_delivery_stats(self, delivery_time: float, success: bool):
        """Update delivery statistics."""
        if success:
            self.stats['successful_deliveries'] += 1
            self.stats['last_successful_test'] = datetime.now().isoformat()
            
            # Update average delivery time
            total_successful = self.stats['successful_deliveries']
            current_avg = self.stats['avg_delivery_time']
            
            self.stats['avg_delivery_time'] = (
                (current_avg * (total_successful - 1) + delivery_time) / 
                total_successful
            )
        else:
            self.stats['failed_deliveries'] += 1
            self.stats['last_failed_test'] = datetime.now().isoformat()
        
        # Update success rate
        total_tests = self.stats['successful_deliveries'] + self.stats['failed_deliveries']
        if total_tests > 0:
            self.stats['delivery_success_rate'] = (
                self.stats['successful_deliveries'] / total_tests * 100
            )
    
    def _verification_loop(self):
        """Background verification loop."""
        while self.verification_active:
            try:
                with self.lock:
                    self.stats['verification_cycles'] += 1
                    
                    # Create periodic verification tests
                    if len(self.pending_tests) < self.max_pending_tests:
                        self._create_periodic_tests()
                    
                    # Check pending tests
                    self._check_pending_tests()
                    
                    # Handle timeouts and retries
                    self._handle_timeouts_and_retries()
                    
                    # Check for failure patterns
                    self._check_failure_patterns()
                
                time.sleep(self.verification_interval)
                
            except Exception as e:
                logger.error(f"Verification loop error: {e}")
                time.sleep(5)
    
    def _create_periodic_tests(self):
        """Create periodic verification tests."""
        current_time = datetime.now()
        
        # Create different types of tests
        test_types = ['basic_delivery', 'high_priority', 'batch_test']
        delivery_methods = [DeliveryMethod.DIRECT, DeliveryMethod.BATCH, DeliveryMethod.HEARTBEAT]
        
        for i, (test_type, method) in enumerate(zip(test_types, delivery_methods)):
            test_id = self.create_verification_test(test_type, method)
            self.send_verification_test(test_id)
    
    def _check_pending_tests(self):
        """Check all pending tests for confirmation."""
        completed_tests = []
        
        for test_id, test in self.pending_tests.items():
            if test.status == VerificationStatus.PENDING:
                if self.check_delivery_confirmation(test_id):
                    completed_tests.append(test_id)
        
        # Move completed tests
        for test_id in completed_tests:
            test = self.pending_tests.pop(test_id)
            self.completed_tests.append(test)
            self.test_history.append(test)
    
    def _handle_timeouts_and_retries(self):
        """Handle test timeouts and retries."""
        current_time = datetime.now()
        timed_out_tests = []
        
        for test_id, test in self.pending_tests.items():
            if test.status == VerificationStatus.PENDING:
                # Check for timeout
                if current_time > test.expected_delivery_time + timedelta(seconds=self.timeout_threshold):
                    if test.attempts < self.retry_attempts:
                        # Retry with different delivery method
                        test.status = VerificationStatus.RETRYING
                        test.delivery_method = self._get_fallback_delivery_method(test.delivery_method)
                        test.expected_delivery_time = current_time + timedelta(seconds=30)
                        self.send_verification_test(test_id)
                    else:
                        # Mark as timeout
                        test.status = VerificationStatus.TIMEOUT
                        test.errors.append("Delivery timeout exceeded")
                        timed_out_tests.append(test_id)
                        
                        self.stats['timeout_deliveries'] += 1
                        self._update_delivery_stats(0, False)
        
        # Move timed out tests
        for test_id in timed_out_tests:
            test = self.pending_tests.pop(test_id)
            self.completed_tests.append(test)
            self.test_history.append(test)
    
    def _get_fallback_delivery_method(self, current_method: DeliveryMethod) -> DeliveryMethod:
        """Get fallback delivery method."""
        fallback_chain = {
            DeliveryMethod.DIRECT: DeliveryMethod.BATCH,
            DeliveryMethod.BATCH: DeliveryMethod.HEARTBEAT,
            DeliveryMethod.HEARTBEAT: DeliveryMethod.FALLBACK,
            DeliveryMethod.FALLBACK: DeliveryMethod.EMERGENCY,
            DeliveryMethod.EMERGENCY: DeliveryMethod.DIRECT
        }
        
        return fallback_chain.get(current_method, DeliveryMethod.EMERGENCY)
    
    def _check_failure_patterns(self):
        """Check for concerning failure patterns."""
        success_rate = self.stats['delivery_success_rate']
        
        # Check for critical failure rate
        if success_rate < (100 - self.critical_failure_threshold * 100):
            self._trigger_alert(
                'critical_failure_rate',
                f"Critical delivery failure rate: {100 - success_rate:.1f}%",
                'critical'
            )
        elif success_rate < (100 - self.failure_threshold * 100):
            self._trigger_alert(
                'high_failure_rate',
                f"High delivery failure rate: {100 - success_rate:.1f}%",
                'warning'
            )
        
        # Check for consecutive failures
        recent_tests = list(self.completed_tests)[-10:]  # Last 10 tests
        consecutive_failures = 0
        
        for test in reversed(recent_tests):
            if test.status in [VerificationStatus.FAILED, VerificationStatus.TIMEOUT]:
                consecutive_failures += 1
            else:
                break
        
        if consecutive_failures >= 5:
            self._trigger_alert(
                'consecutive_failures',
                f"5+ consecutive delivery failures detected",
                'critical'
            )
    
    def _trigger_alert(self, alert_type: str, message: str, severity: str):
        """Trigger alert for delivery issues."""
        self.stats['alerts_triggered'] += 1
        
        for handler in self.alert_handlers:
            try:
                handler(alert_type, message, severity)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"DELIVERY ALERT [{severity.upper()}]: {message}")
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.verification_active:
            try:
                time.sleep(300)  # Cleanup every 5 minutes
                
                with self.lock:
                    # Clean very old pending tests (should not happen)
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    old_tests = [
                        test_id for test_id, test in self.pending_tests.items()
                        if test.created_at < cutoff_time
                    ]
                    
                    for test_id in old_tests:
                        logger.warning(f"Cleaning up old pending test: {test_id}")
                        test = self.pending_tests.pop(test_id)
                        test.status = VerificationStatus.FAILED
                        test.errors.append("Cleanup timeout")
                        self.completed_tests.append(test)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def get_verification_status(self) -> Dict[str, Any]:
        """Get comprehensive verification status."""
        with self.lock:
            return {
                'statistics': dict(self.stats),
                'pending_tests': len(self.pending_tests),
                'completed_tests': len(self.completed_tests),
                'recent_tests': [
                    {
                        'test_id': test.test_id,
                        'test_type': test.test_type,
                        'status': test.status.value,
                        'delivery_method': test.delivery_method.value,
                        'created_at': test.created_at.isoformat(),
                        'delivery_time': (
                            (test.confirmation_time - test.created_at).total_seconds()
                            if test.confirmation_time else None
                        ),
                        'attempts': test.attempts,
                        'errors': test.errors
                    }
                    for test in list(self.completed_tests)[-20:]  # Last 20 tests
                ],
                'verification_active': self.verification_active,
                'endpoints': self.verification_endpoints
            }
    
    def force_verification_test(self,
                               test_type: str = "basic_delivery",
                               delivery_method: str = "direct") -> str:
        """
        Force a verification test for manual testing.
        
        Args:
            test_type: Type of test
            delivery_method: Delivery method string
            
        Returns:
            Test ID
        """
        method_map = {
            'direct': DeliveryMethod.DIRECT,
            'batch': DeliveryMethod.BATCH,
            'heartbeat': DeliveryMethod.HEARTBEAT,
            'fallback': DeliveryMethod.FALLBACK,
            'emergency': DeliveryMethod.EMERGENCY
        }
        
        method = method_map.get(delivery_method, DeliveryMethod.DIRECT)
        
        test_id = self.create_verification_test(test_type, method)
        self.send_verification_test(test_id)
        
        return test_id
    
    def shutdown(self):
        """Shutdown verification system."""
        self.verification_active = False
        
        if self.verification_thread and self.verification_thread.is_alive():
            self.verification_thread.join(timeout=10)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        logger.info(f"Analytics Delivery Verifier shutdown - Stats: {self.stats}")

# Global verifier instance
delivery_verifier = None