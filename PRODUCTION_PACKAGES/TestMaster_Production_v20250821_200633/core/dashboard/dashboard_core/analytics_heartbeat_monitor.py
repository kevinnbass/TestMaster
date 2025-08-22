"""
Dashboard Connection Heartbeat Monitor
======================================

Monitors dashboard connectivity and ensures analytics delivery with
heartbeat checks, connection pooling, and automatic reconnection.

Author: TestMaster Team
"""

import logging
import time
import threading
import socket
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """Connection status states."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class DeliveryStatus(Enum):
    """Analytics delivery status."""
    DELIVERED = "delivered"
    PENDING = "pending"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class HeartbeatPulse:
    """Represents a heartbeat pulse."""
    pulse_id: str
    timestamp: datetime
    endpoint: str
    response_time_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None

@dataclass
class DeliveryConfirmation:
    """Analytics delivery confirmation."""
    confirmation_id: str
    analytics_id: str
    timestamp: datetime
    endpoint: str
    status: DeliveryStatus
    attempts: int
    response_time_ms: float
    payload_size_bytes: int
    error: Optional[str] = None

class DashboardHeartbeatMonitor:
    """
    Monitors dashboard connectivity and ensures analytics delivery.
    """
    
    def __init__(self,
                 heartbeat_interval: int = 30,
                 timeout: int = 5,
                 max_failures: int = 3):
        """
        Initialize heartbeat monitor.
        
        Args:
            heartbeat_interval: Seconds between heartbeats
            timeout: Request timeout in seconds
            max_failures: Failures before marking disconnected
        """
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.max_failures = max_failures
        
        # Endpoint tracking
        self.endpoints = {}
        self.endpoint_status = {}
        self.endpoint_failures = defaultdict(int)
        
        # Heartbeat history
        self.heartbeat_history = deque(maxlen=1000)
        self.last_heartbeat = {}
        
        # Delivery tracking
        self.delivery_queue = deque(maxlen=10000)
        self.delivery_confirmations = deque(maxlen=1000)
        self.pending_deliveries = {}
        
        # Connection pooling
        self.connection_pool = {}
        self.pool_size = 5
        
        # Statistics
        self.stats = {
            'total_heartbeats': 0,
            'successful_heartbeats': 0,
            'failed_heartbeats': 0,
            'total_deliveries': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'reconnections': 0,
            'average_response_time_ms': 0
        }
        
        # Callbacks
        self.connection_callbacks = {
            'on_connect': [],
            'on_disconnect': [],
            'on_reconnect': [],
            'on_degraded': []
        }
        
        # Delivery strategies
        self.delivery_strategies = {
            'direct': self._deliver_direct,
            'batched': self._deliver_batched,
            'compressed': self._deliver_compressed,
            'chunked': self._deliver_chunked
        }
        
        # Health checks
        self.health_checks = {
            'ping': self._check_ping,
            'http': self._check_http,
            'websocket': self._check_websocket
        }
        
        # Monitoring threads
        self.monitoring_active = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.delivery_thread = threading.Thread(
            target=self._delivery_loop,
            daemon=True
        )
        
        # Start monitoring
        self.heartbeat_thread.start()
        self.delivery_thread.start()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Dashboard Heartbeat Monitor initialized")
    
    def register_endpoint(self,
                         name: str,
                         url: str,
                         check_type: str = 'http',
                         critical: bool = True):
        """
        Register a dashboard endpoint for monitoring.
        
        Args:
            name: Endpoint name
            url: Endpoint URL
            check_type: Type of health check (ping, http, websocket)
            critical: Whether endpoint is critical
        """
        with self.lock:
            self.endpoints[name] = {
                'url': url,
                'check_type': check_type,
                'critical': critical,
                'registered_at': datetime.now()
            }
            
            self.endpoint_status[name] = ConnectionStatus.UNKNOWN
            self.endpoint_failures[name] = 0
            
            # Initialize connection pool for endpoint
            self._init_connection_pool(name, url)
            
            logger.info(f"Registered endpoint: {name} ({url})")
    
    def send_analytics(self,
                      analytics_data: Dict[str, Any],
                      endpoint: str = 'main_dashboard',
                      strategy: str = 'direct',
                      priority: int = 5) -> str:
        """
        Send analytics to dashboard with delivery confirmation.
        
        Args:
            analytics_data: Analytics data to send
            endpoint: Target endpoint name
            strategy: Delivery strategy
            priority: Delivery priority (1-10)
            
        Returns:
            Delivery ID for tracking
        """
        delivery_id = f"del_{int(time.time() * 1000000)}"
        
        with self.lock:
            # Add to delivery queue
            delivery_item = {
                'id': delivery_id,
                'data': analytics_data,
                'endpoint': endpoint,
                'strategy': strategy,
                'priority': priority,
                'timestamp': datetime.now(),
                'attempts': 0
            }
            
            # Priority queue insertion
            if priority > 7:
                self.delivery_queue.appendleft(delivery_item)
            else:
                self.delivery_queue.append(delivery_item)
            
            # Track as pending
            self.pending_deliveries[delivery_id] = delivery_item
            
            logger.debug(f"Queued analytics delivery: {delivery_id}")
        
        return delivery_id
    
    def get_delivery_status(self, delivery_id: str) -> Optional[DeliveryConfirmation]:
        """
        Get status of analytics delivery.
        
        Args:
            delivery_id: Delivery ID to check
            
        Returns:
            Delivery confirmation or None
        """
        with self.lock:
            # Check confirmations
            for confirmation in self.delivery_confirmations:
                if confirmation.analytics_id == delivery_id:
                    return confirmation
            
            # Check if still pending
            if delivery_id in self.pending_deliveries:
                return DeliveryConfirmation(
                    confirmation_id=f"conf_{int(time.time() * 1000000)}",
                    analytics_id=delivery_id,
                    timestamp=datetime.now(),
                    endpoint=self.pending_deliveries[delivery_id]['endpoint'],
                    status=DeliveryStatus.PENDING,
                    attempts=self.pending_deliveries[delivery_id]['attempts'],
                    response_time_ms=0,
                    payload_size_bytes=len(json.dumps(
                        self.pending_deliveries[delivery_id]['data']
                    )),
                    error=None
                )
        
        return None
    
    def _heartbeat_loop(self):
        """Background heartbeat monitoring loop."""
        while self.monitoring_active:
            try:
                time.sleep(self.heartbeat_interval)
                
                with self.lock:
                    for endpoint_name, endpoint_info in self.endpoints.items():
                        self._send_heartbeat(endpoint_name, endpoint_info)
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    def _send_heartbeat(self, endpoint_name: str, endpoint_info: Dict[str, Any]):
        """Send heartbeat to endpoint."""
        pulse_id = f"pulse_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        check_type = endpoint_info['check_type']
        check_func = self.health_checks.get(check_type, self._check_http)
        
        try:
            # Perform health check
            success, status_code, error = check_func(endpoint_info['url'])
            response_time = (time.time() - start_time) * 1000
            
            # Create heartbeat pulse
            pulse = HeartbeatPulse(
                pulse_id=pulse_id,
                timestamp=datetime.now(),
                endpoint=endpoint_name,
                response_time_ms=response_time,
                status_code=status_code,
                success=success,
                error=error
            )
            
            # Update statistics
            self.stats['total_heartbeats'] += 1
            if success:
                self.stats['successful_heartbeats'] += 1
                self.endpoint_failures[endpoint_name] = 0
                
                # Update status if needed
                if self.endpoint_status[endpoint_name] != ConnectionStatus.CONNECTED:
                    self._handle_connection_change(
                        endpoint_name,
                        ConnectionStatus.CONNECTED
                    )
            else:
                self.stats['failed_heartbeats'] += 1
                self.endpoint_failures[endpoint_name] += 1
                
                # Check failure threshold
                if self.endpoint_failures[endpoint_name] >= self.max_failures:
                    if self.endpoint_status[endpoint_name] == ConnectionStatus.CONNECTED:
                        self._handle_connection_change(
                            endpoint_name,
                            ConnectionStatus.DISCONNECTED
                        )
            
            # Store pulse
            self.heartbeat_history.append(pulse)
            self.last_heartbeat[endpoint_name] = pulse
            
            # Update average response time
            self._update_response_time_average(response_time)
            
        except Exception as e:
            logger.error(f"Heartbeat failed for {endpoint_name}: {e}")
            self.endpoint_failures[endpoint_name] += 1
    
    def _check_http(self, url: str) -> tuple:
        """HTTP health check."""
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={'X-Heartbeat': 'true'}
            )
            return (
                response.status_code == 200,
                response.status_code,
                None
            )
        except requests.RequestException as e:
            return (False, 0, str(e))
    
    def _check_ping(self, url: str) -> tuple:
        """Ping health check."""
        try:
            parsed = urlparse(url)
            host = parsed.hostname or 'localhost'
            
            # Simple TCP connection test
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((host, parsed.port or 80))
            sock.close()
            
            success = result == 0
            return (success, 200 if success else 503, None)
        except Exception as e:
            return (False, 0, str(e))
    
    def _check_websocket(self, url: str) -> tuple:
        """WebSocket health check."""
        # Simplified - would use websocket library in production
        return self._check_http(url.replace('ws://', 'http://').replace('wss://', 'https://'))
    
    def _delivery_loop(self):
        """Background delivery processing loop."""
        while self.monitoring_active:
            try:
                time.sleep(1)  # Process deliveries every second
                
                with self.lock:
                    if self.delivery_queue:
                        # Process up to 10 deliveries
                        for _ in range(min(10, len(self.delivery_queue))):
                            delivery = self.delivery_queue.popleft()
                            self._process_delivery(delivery)
                
            except Exception as e:
                logger.error(f"Delivery loop error: {e}")
    
    def _process_delivery(self, delivery: Dict[str, Any]):
        """Process analytics delivery."""
        endpoint_name = delivery['endpoint']
        
        # Check endpoint status
        if self.endpoint_status.get(endpoint_name) == ConnectionStatus.DISCONNECTED:
            # Re-queue if disconnected
            if delivery['attempts'] < 5:
                delivery['attempts'] += 1
                self.delivery_queue.append(delivery)
            else:
                self._mark_delivery_failed(delivery, "Endpoint disconnected")
            return
        
        # Get delivery strategy
        strategy = self.delivery_strategies.get(
            delivery['strategy'],
            self._deliver_direct
        )
        
        # Attempt delivery
        start_time = time.time()
        success, error = strategy(delivery)
        response_time = (time.time() - start_time) * 1000
        
        # Create confirmation
        confirmation = DeliveryConfirmation(
            confirmation_id=f"conf_{int(time.time() * 1000000)}",
            analytics_id=delivery['id'],
            timestamp=datetime.now(),
            endpoint=endpoint_name,
            status=DeliveryStatus.DELIVERED if success else DeliveryStatus.FAILED,
            attempts=delivery['attempts'] + 1,
            response_time_ms=response_time,
            payload_size_bytes=len(json.dumps(delivery['data'])),
            error=error
        )
        
        # Update statistics
        self.stats['total_deliveries'] += 1
        if success:
            self.stats['successful_deliveries'] += 1
            # Remove from pending
            self.pending_deliveries.pop(delivery['id'], None)
        else:
            self.stats['failed_deliveries'] += 1
            # Retry if needed
            if delivery['attempts'] < 3:
                delivery['attempts'] += 1
                self.delivery_queue.append(delivery)
                confirmation.status = DeliveryStatus.RETRYING
        
        # Store confirmation
        self.delivery_confirmations.append(confirmation)
    
    def _deliver_direct(self, delivery: Dict[str, Any]) -> tuple:
        """Direct delivery strategy."""
        try:
            endpoint_info = self.endpoints.get(delivery['endpoint'], {})
            url = endpoint_info.get('url', 'http://localhost:5000/api/analytics')
            
            response = requests.post(
                url,
                json=delivery['data'],
                timeout=self.timeout,
                headers={'X-Analytics-ID': delivery['id']}
            )
            
            return (response.status_code == 200, None)
        except Exception as e:
            return (False, str(e))
    
    def _deliver_batched(self, delivery: Dict[str, Any]) -> tuple:
        """Batched delivery strategy."""
        # Collect multiple deliveries and send as batch
        batch = [delivery['data']]
        
        # Collect more if available
        with self.lock:
            for _ in range(min(9, len(self.delivery_queue))):
                if self.delivery_queue and self.delivery_queue[0]['endpoint'] == delivery['endpoint']:
                    batch.append(self.delivery_queue.popleft()['data'])
        
        try:
            endpoint_info = self.endpoints.get(delivery['endpoint'], {})
            url = endpoint_info.get('url', 'http://localhost:5000/api/analytics')
            
            response = requests.post(
                f"{url}/batch",
                json={'batch': batch},
                timeout=self.timeout
            )
            
            return (response.status_code == 200, None)
        except Exception as e:
            return (False, str(e))
    
    def _deliver_compressed(self, delivery: Dict[str, Any]) -> tuple:
        """Compressed delivery strategy."""
        import zlib
        import base64
        
        try:
            # Compress data
            json_str = json.dumps(delivery['data'])
            compressed = zlib.compress(json_str.encode())
            encoded = base64.b64encode(compressed).decode()
            
            endpoint_info = self.endpoints.get(delivery['endpoint'], {})
            url = endpoint_info.get('url', 'http://localhost:5000/api/analytics')
            
            response = requests.post(
                url,
                json={'compressed': encoded},
                timeout=self.timeout,
                headers={'Content-Encoding': 'gzip'}
            )
            
            return (response.status_code == 200, None)
        except Exception as e:
            return (False, str(e))
    
    def _deliver_chunked(self, delivery: Dict[str, Any]) -> tuple:
        """Chunked delivery for large payloads."""
        try:
            json_str = json.dumps(delivery['data'])
            chunk_size = 10000  # 10KB chunks
            
            chunks = [
                json_str[i:i+chunk_size]
                for i in range(0, len(json_str), chunk_size)
            ]
            
            endpoint_info = self.endpoints.get(delivery['endpoint'], {})
            url = endpoint_info.get('url', 'http://localhost:5000/api/analytics')
            
            # Send chunks
            for i, chunk in enumerate(chunks):
                response = requests.post(
                    f"{url}/chunk",
                    json={
                        'chunk_id': f"{delivery['id']}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'data': chunk
                    },
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    return (False, f"Chunk {i} failed")
            
            return (True, None)
        except Exception as e:
            return (False, str(e))
    
    def _mark_delivery_failed(self, delivery: Dict[str, Any], error: str):
        """Mark delivery as failed."""
        confirmation = DeliveryConfirmation(
            confirmation_id=f"conf_{int(time.time() * 1000000)}",
            analytics_id=delivery['id'],
            timestamp=datetime.now(),
            endpoint=delivery['endpoint'],
            status=DeliveryStatus.FAILED,
            attempts=delivery['attempts'],
            response_time_ms=0,
            payload_size_bytes=len(json.dumps(delivery['data'])),
            error=error
        )
        
        self.delivery_confirmations.append(confirmation)
        self.pending_deliveries.pop(delivery['id'], None)
        self.stats['failed_deliveries'] += 1
    
    def _handle_connection_change(self, endpoint: str, new_status: ConnectionStatus):
        """Handle connection status change."""
        old_status = self.endpoint_status[endpoint]
        self.endpoint_status[endpoint] = new_status
        
        logger.info(f"Endpoint {endpoint}: {old_status.value} -> {new_status.value}")
        
        # Trigger callbacks
        if new_status == ConnectionStatus.CONNECTED:
            callbacks = self.connection_callbacks['on_connect']
        elif new_status == ConnectionStatus.DISCONNECTED:
            callbacks = self.connection_callbacks['on_disconnect']
        elif new_status == ConnectionStatus.RECONNECTING:
            callbacks = self.connection_callbacks['on_reconnect']
            self.stats['reconnections'] += 1
        else:
            callbacks = self.connection_callbacks['on_degraded']
        
        for callback in callbacks:
            try:
                callback(endpoint, old_status, new_status)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _init_connection_pool(self, name: str, url: str):
        """Initialize connection pool for endpoint."""
        self.connection_pool[name] = {
            'url': url,
            'connections': [],
            'active': 0,
            'max_size': self.pool_size
        }
    
    def _update_response_time_average(self, response_time: float):
        """Update average response time."""
        total = self.stats['successful_heartbeats']
        if total > 0:
            current_avg = self.stats['average_response_time_ms']
            self.stats['average_response_time_ms'] = (
                (current_avg * (total - 1) + response_time) / total
            )
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register callback for connection events.
        
        Args:
            event: Event type (on_connect, on_disconnect, etc.)
            callback: Callback function
        """
        if event in self.connection_callbacks:
            self.connection_callbacks[event].append(callback)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        with self.lock:
            endpoints_status = {}
            for name in self.endpoints:
                heartbeat_data = None
                if name in self.last_heartbeat:
                    hb = self.last_heartbeat[name]
                    heartbeat_data = {
                        'pulse_id': hb.pulse_id,
                        'timestamp': hb.timestamp.isoformat() if hb.timestamp else None,
                        'endpoint': hb.endpoint,
                        'response_time_ms': hb.response_time_ms,
                        'status_code': hb.status_code,
                        'success': hb.success,
                        'error': hb.error
                    }
                
                endpoints_status[name] = {
                    'status': self.endpoint_status[name].value,
                    'failures': self.endpoint_failures[name],
                    'last_heartbeat': heartbeat_data
                }
            
            return {
                'endpoints': endpoints_status,
                'statistics': self.stats,
                'pending_deliveries': len(self.pending_deliveries),
                'delivery_queue_size': len(self.delivery_queue),
                'overall_health': self._calculate_overall_health()
            }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health."""
        if not self.endpoints:
            return "unknown"
        
        connected = sum(
            1 for status in self.endpoint_status.values()
            if status == ConnectionStatus.CONNECTED
        )
        
        critical_down = any(
            self.endpoints[name]['critical'] and 
            self.endpoint_status[name] != ConnectionStatus.CONNECTED
            for name in self.endpoints
        )
        
        if critical_down:
            return "critical"
        elif connected == len(self.endpoints):
            return "healthy"
        elif connected > 0:
            return "degraded"
        else:
            return "down"
    
    def shutdown(self):
        """Shutdown heartbeat monitor."""
        self.monitoring_active = False
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        if self.delivery_thread and self.delivery_thread.is_alive():
            self.delivery_thread.join(timeout=5)
        
        logger.info(f"Heartbeat Monitor shutdown - Stats: {self.stats}")

# Global heartbeat monitor instance
heartbeat_monitor = DashboardHeartbeatMonitor()