"""
Analytics Dashboard Connectivity and Data Flow Monitor
=====================================================

Comprehensive monitoring system for dashboard connectivity, data flow,
real-time updates, and end-to-end data delivery verification.

Author: TestMaster Team
"""

import logging
import time
import threading
import requests
import websocket
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import socket
# import ping3  # Optional dependency
import psutil

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    UNSTABLE = "unstable"
    RECONNECTING = "reconnecting"

class DataFlowStatus(Enum):
    FLOWING = "flowing"
    BLOCKED = "blocked"
    DELAYED = "delayed"
    CORRUPTED = "corrupted"
    PARTIAL = "partial"

class MonitoringLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    REAL_TIME = "real_time"

@dataclass
class ConnectionMetrics:
    """Connection quality metrics."""
    latency_ms: float
    bandwidth_mbps: float
    packet_loss_percent: float
    jitter_ms: float
    uptime_percent: float
    last_disconnection: Optional[datetime]
    reconnection_count: int

@dataclass
class DataFlowMetrics:
    """Data flow quality metrics."""
    throughput_records_per_second: float
    data_freshness_seconds: float
    delivery_success_rate: float
    corruption_rate: float
    duplicate_rate: float
    out_of_order_rate: float
    average_payload_size_kb: float

@dataclass
class DashboardEndpoint:
    """Dashboard endpoint configuration."""
    endpoint_id: str
    url: str
    endpoint_type: str  # http, websocket, sse
    expected_update_interval: float
    timeout_seconds: float
    retry_attempts: int
    health_check_path: Optional[str] = None
    auth_headers: Dict[str, str] = None

@dataclass
class ConnectivityEvent:
    """Connectivity event record."""
    event_id: str
    timestamp: datetime
    endpoint_id: str
    event_type: str  # connection_lost, connection_restored, degraded, timeout
    severity: str  # info, warning, error, critical
    description: str
    metrics: Dict[str, Any]
    duration_seconds: Optional[float] = None

class AnalyticsConnectivityMonitor:
    """
    Comprehensive dashboard connectivity and data flow monitoring system.
    """
    
    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
                 check_interval: float = 10.0,
                 data_flow_timeout: float = 60.0):
        """
        Initialize analytics connectivity monitor.
        
        Args:
            monitoring_level: Level of monitoring detail
            check_interval: Interval between connectivity checks
            data_flow_timeout: Timeout for data flow verification
        """
        self.monitoring_level = monitoring_level
        self.check_interval = check_interval
        self.data_flow_timeout = data_flow_timeout
        
        # Dashboard endpoints
        self.dashboard_endpoints = {}
        self.connection_status = {}
        self.connection_metrics = {}
        
        # Data flow tracking
        self.data_flow_status = {}
        self.data_flow_metrics = {}
        self.pending_data_confirmations = {}
        
        # Event tracking
        self.connectivity_events = deque(maxlen=10000)
        self.performance_history = defaultdict(deque)
        
        # Real-time monitoring
        self.websocket_connections = {}
        self.sse_connections = {}
        
        # Monitoring statistics
        self.monitor_stats = {
            'total_checks_performed': 0,
            'connection_failures': 0,
            'data_flow_failures': 0,
            'reconnection_attempts': 0,
            'successful_reconnections': 0,
            'average_latency_ms': 0,
            'start_time': datetime.now()
        }
        
        # Background monitoring
        self.monitor_active = False
        self.connectivity_thread = None
        self.data_flow_thread = None
        self.websocket_thread = None
        
        # Alerting system
        self.alert_handlers = []
        self.alert_thresholds = {
            'latency_ms': 1000,
            'packet_loss_percent': 5.0,
            'uptime_percent': 95.0,
            'data_freshness_seconds': 120,
            'delivery_success_rate': 95.0
        }
        
        # Data verification
        self.data_checksums = {}
        self.expected_data_patterns = {}
        
        logger.info(f"Analytics Connectivity Monitor initialized: {monitoring_level.value}")
    
    def start_monitoring(self):
        """Start connectivity and data flow monitoring."""
        if self.monitor_active:
            return
        
        self.monitor_active = True
        
        # Start monitoring threads
        self.connectivity_thread = threading.Thread(target=self._connectivity_monitoring_loop, daemon=True)
        self.data_flow_thread = threading.Thread(target=self._data_flow_monitoring_loop, daemon=True)
        
        if self.monitoring_level in [MonitoringLevel.COMPREHENSIVE, MonitoringLevel.REAL_TIME]:
            self.websocket_thread = threading.Thread(target=self._websocket_monitoring_loop, daemon=True)
            self.websocket_thread.start()
        
        self.connectivity_thread.start()
        self.data_flow_thread.start()
        
        logger.info("Analytics connectivity monitoring started")
    
    def stop_monitoring(self):
        """Stop connectivity monitoring."""
        self.monitor_active = False
        
        # Close all connections
        for ws in self.websocket_connections.values():
            try:
                ws.close()
            except:
                pass
        
        # Wait for threads to finish
        for thread in [self.connectivity_thread, self.data_flow_thread, self.websocket_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("Analytics connectivity monitoring stopped")
    
    def register_dashboard_endpoint(self, endpoint: DashboardEndpoint):
        """Register a dashboard endpoint for monitoring."""
        self.dashboard_endpoints[endpoint.endpoint_id] = endpoint
        self.connection_status[endpoint.endpoint_id] = ConnectionStatus.DISCONNECTED
        self.data_flow_status[endpoint.endpoint_id] = DataFlowStatus.BLOCKED
        
        # Initialize metrics
        self.connection_metrics[endpoint.endpoint_id] = ConnectionMetrics(
            latency_ms=0,
            bandwidth_mbps=0,
            packet_loss_percent=0,
            jitter_ms=0,
            uptime_percent=100,
            last_disconnection=None,
            reconnection_count=0
        )
        
        self.data_flow_metrics[endpoint.endpoint_id] = DataFlowMetrics(
            throughput_records_per_second=0,
            data_freshness_seconds=0,
            delivery_success_rate=100,
            corruption_rate=0,
            duplicate_rate=0,
            out_of_order_rate=0,
            average_payload_size_kb=0
        )
        
        logger.info(f"Registered dashboard endpoint: {endpoint.endpoint_id}")
    
    def verify_data_delivery(self, data: Dict[str, Any], endpoint_id: str,
                           expected_arrival_time: Optional[datetime] = None) -> str:
        """
        Verify that data reaches the dashboard endpoint.
        
        Args:
            data: Data to track
            endpoint_id: Target endpoint
            expected_arrival_time: When data should arrive
        
        Returns:
            Tracking ID for verification
        """
        tracking_id = self._generate_tracking_id(data, endpoint_id)
        
        # Create verification record
        verification_record = {
            'tracking_id': tracking_id,
            'data_checksum': self._calculate_data_checksum(data),
            'endpoint_id': endpoint_id,
            'sent_time': datetime.now(),
            'expected_arrival_time': expected_arrival_time or (datetime.now() + timedelta(seconds=30)),
            'confirmed': False,
            'arrival_time': None,
            'corruption_detected': False
        }
        
        self.pending_data_confirmations[tracking_id] = verification_record
        
        # Add tracking metadata to data
        tracked_data = data.copy()
        tracked_data['_tracking_id'] = tracking_id
        tracked_data['_sent_time'] = verification_record['sent_time'].isoformat()
        tracked_data['_checksum'] = verification_record['data_checksum']
        
        return tracking_id
    
    def confirm_data_arrival(self, tracking_id: str, received_data: Dict[str, Any]) -> bool:
        """
        Confirm that data arrived at dashboard endpoint.
        
        Args:
            tracking_id: Data tracking ID
            received_data: Data received at endpoint
        
        Returns:
            True if data matches and is confirmed
        """
        if tracking_id not in self.pending_data_confirmations:
            return False
        
        verification_record = self.pending_data_confirmations[tracking_id]
        
        # Check data integrity
        received_checksum = self._calculate_data_checksum(received_data)
        expected_checksum = verification_record['data_checksum']
        
        corruption_detected = received_checksum != expected_checksum
        
        # Update verification record
        verification_record['confirmed'] = True
        verification_record['arrival_time'] = datetime.now()
        verification_record['corruption_detected'] = corruption_detected
        
        # Update metrics
        endpoint_id = verification_record['endpoint_id']
        flow_metrics = self.data_flow_metrics[endpoint_id]
        
        # Calculate delivery time
        delivery_time = (verification_record['arrival_time'] - verification_record['sent_time']).total_seconds()
        flow_metrics.data_freshness_seconds = delivery_time
        
        # Update delivery success rate
        total_confirmations = len([r for r in self.pending_data_confirmations.values() if r['confirmed']])
        successful_deliveries = len([r for r in self.pending_data_confirmations.values() 
                                   if r['confirmed'] and not r['corruption_detected']])
        
        if total_confirmations > 0:
            flow_metrics.delivery_success_rate = (successful_deliveries / total_confirmations) * 100
            flow_metrics.corruption_rate = (
                len([r for r in self.pending_data_confirmations.values() 
                    if r['confirmed'] and r['corruption_detected']]) / total_confirmations
            ) * 100
        
        # Update data flow status
        if corruption_detected:
            self.data_flow_status[endpoint_id] = DataFlowStatus.CORRUPTED
        elif delivery_time < self.data_flow_timeout:
            self.data_flow_status[endpoint_id] = DataFlowStatus.FLOWING
        else:
            self.data_flow_status[endpoint_id] = DataFlowStatus.DELAYED
        
        logger.debug(f"Data delivery confirmed for {tracking_id}: corruption={corruption_detected}")
        return not corruption_detected
    
    def check_endpoint_connectivity(self, endpoint_id: str) -> Tuple[ConnectionStatus, ConnectionMetrics]:
        """
        Check connectivity to a specific endpoint.
        
        Args:
            endpoint_id: Endpoint to check
        
        Returns:
            Tuple of (connection_status, metrics)
        """
        if endpoint_id not in self.dashboard_endpoints:
            return ConnectionStatus.DISCONNECTED, None
        
        endpoint = self.dashboard_endpoints[endpoint_id]
        metrics = self.connection_metrics[endpoint_id]
        
        try:
            start_time = time.time()
            
            if endpoint.endpoint_type == "http":
                status, latency = self._check_http_endpoint(endpoint)
            elif endpoint.endpoint_type == "websocket":
                status, latency = self._check_websocket_endpoint(endpoint)
            elif endpoint.endpoint_type == "sse":
                status, latency = self._check_sse_endpoint(endpoint)
            else:
                status, latency = ConnectionStatus.DISCONNECTED, 0
            
            # Update metrics
            metrics.latency_ms = latency
            
            # Calculate bandwidth (simplified)
            if status == ConnectionStatus.CONNECTED:
                # Estimate bandwidth with a small test request
                bandwidth = self._estimate_bandwidth(endpoint)
                metrics.bandwidth_mbps = bandwidth
            
            # Update connection status
            self.connection_status[endpoint_id] = status
            
            # Record event if status changed
            self._record_connectivity_event(endpoint_id, status, metrics)
            
            self.monitor_stats['total_checks_performed'] += 1
            
            return status, metrics
        
        except Exception as e:
            logger.error(f"Error checking connectivity for {endpoint_id}: {e}")
            self.connection_status[endpoint_id] = ConnectionStatus.DISCONNECTED
            self.monitor_stats['connection_failures'] += 1
            return ConnectionStatus.DISCONNECTED, metrics
    
    def get_connectivity_summary(self) -> Dict[str, Any]:
        """Get comprehensive connectivity monitoring summary."""
        uptime = (datetime.now() - self.monitor_stats['start_time']).total_seconds()
        
        # Calculate overall system health
        total_endpoints = len(self.dashboard_endpoints)
        connected_endpoints = len([s for s in self.connection_status.values() 
                                 if s == ConnectionStatus.CONNECTED])
        
        system_health = (connected_endpoints / max(1, total_endpoints)) * 100
        
        # Recent events
        recent_events = [e for e in self.connectivity_events
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        # Connection status summary
        status_summary = defaultdict(int)
        for status in self.connection_status.values():
            status_summary[status.value] += 1
        
        # Data flow status summary
        flow_summary = defaultdict(int)
        for status in self.data_flow_status.values():
            flow_summary[status.value] += 1
        
        # Average metrics
        avg_latency = 0
        avg_delivery_rate = 0
        if self.connection_metrics:
            avg_latency = sum(m.latency_ms for m in self.connection_metrics.values()) / len(self.connection_metrics)
        if self.data_flow_metrics:
            avg_delivery_rate = sum(m.delivery_success_rate for m in self.data_flow_metrics.values()) / len(self.data_flow_metrics)
        
        return {
            'monitoring_status': {
                'active': self.monitor_active,
                'monitoring_level': self.monitoring_level.value,
                'uptime_seconds': uptime,
                'system_health_percent': system_health
            },
            'statistics': self.monitor_stats.copy(),
            'endpoint_status': {
                'total_endpoints': total_endpoints,
                'connected_endpoints': connected_endpoints,
                'connection_status_distribution': dict(status_summary),
                'data_flow_status_distribution': dict(flow_summary)
            },
            'performance_metrics': {
                'average_latency_ms': avg_latency,
                'average_delivery_success_rate': avg_delivery_rate,
                'pending_confirmations': len(self.pending_data_confirmations),
                'recent_events': len(recent_events)
            },
            'real_time_connections': {
                'active_websockets': len(self.websocket_connections),
                'active_sse_connections': len(self.sse_connections)
            },
            'alert_status': {
                'alert_handlers_configured': len(self.alert_handlers),
                'alert_thresholds': self.alert_thresholds.copy()
            }
        }
    
    def add_alert_handler(self, handler: Callable[[ConnectivityEvent], None]):
        """Add alert handler for connectivity events."""
        self.alert_handlers.append(handler)
        logger.info("Added connectivity alert handler")
    
    def _connectivity_monitoring_loop(self):
        """Background connectivity monitoring loop."""
        while self.monitor_active:
            try:
                for endpoint_id in self.dashboard_endpoints:
                    status, metrics = self.check_endpoint_connectivity(endpoint_id)
                    
                    # Check for alert conditions
                    self._check_alert_conditions(endpoint_id, status, metrics)
                
                time.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Connectivity monitoring loop error: {e}")
    
    def _data_flow_monitoring_loop(self):
        """Background data flow monitoring loop."""
        while self.monitor_active:
            try:
                current_time = datetime.now()
                
                # Check for overdue data confirmations
                for tracking_id, verification_record in list(self.pending_data_confirmations.items()):
                    if (not verification_record['confirmed'] and 
                        current_time > verification_record['expected_arrival_time']):
                        
                        # Mark as failed delivery
                        endpoint_id = verification_record['endpoint_id']
                        self.data_flow_status[endpoint_id] = DataFlowStatus.BLOCKED
                        
                        # Create event
                        event = ConnectivityEvent(
                            event_id=f"data_timeout_{int(time.time())}",
                            timestamp=current_time,
                            endpoint_id=endpoint_id,
                            event_type="data_timeout",
                            severity="warning",
                            description=f"Data delivery timeout for tracking ID {tracking_id}",
                            metrics={'tracking_id': tracking_id}
                        )
                        self.connectivity_events.append(event)
                        
                        # Remove from pending
                        del self.pending_data_confirmations[tracking_id]
                        
                        self.monitor_stats['data_flow_failures'] += 1
                
                # Update data flow metrics
                self._update_data_flow_metrics()
                
                time.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Data flow monitoring loop error: {e}")
    
    def _websocket_monitoring_loop(self):
        """Background WebSocket monitoring loop."""
        while self.monitor_active:
            try:
                for endpoint_id, endpoint in self.dashboard_endpoints.items():
                    if endpoint.endpoint_type == "websocket":
                        self._maintain_websocket_connection(endpoint_id, endpoint)
                
                time.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"WebSocket monitoring loop error: {e}")
    
    def _check_http_endpoint(self, endpoint: DashboardEndpoint) -> Tuple[ConnectionStatus, float]:
        """Check HTTP endpoint connectivity."""
        start_time = time.time()
        
        try:
            # Use health check path if available
            url = endpoint.url
            if endpoint.health_check_path:
                url = url.rstrip('/') + '/' + endpoint.health_check_path.lstrip('/')
            
            headers = endpoint.auth_headers or {}
            response = requests.get(url, headers=headers, timeout=endpoint.timeout_seconds)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return ConnectionStatus.CONNECTED, latency_ms
            elif response.status_code in [503, 502, 504]:
                return ConnectionStatus.DEGRADED, latency_ms
            else:
                return ConnectionStatus.UNSTABLE, latency_ms
        
        except requests.exceptions.Timeout:
            return ConnectionStatus.UNSTABLE, (time.time() - start_time) * 1000
        except requests.exceptions.ConnectionError:
            return ConnectionStatus.DISCONNECTED, (time.time() - start_time) * 1000
        except Exception as e:
            logger.error(f"HTTP check error: {e}")
            return ConnectionStatus.DISCONNECTED, (time.time() - start_time) * 1000
    
    def _check_websocket_endpoint(self, endpoint: DashboardEndpoint) -> Tuple[ConnectionStatus, float]:
        """Check WebSocket endpoint connectivity."""
        start_time = time.time()
        
        try:
            # Create a test WebSocket connection
            ws = websocket.create_connection(endpoint.url, timeout=endpoint.timeout_seconds)
            latency_ms = (time.time() - start_time) * 1000
            
            # Send ping
            ws.ping()
            ws.close()
            
            return ConnectionStatus.CONNECTED, latency_ms
        
        except websocket.WebSocketTimeoutException:
            return ConnectionStatus.UNSTABLE, (time.time() - start_time) * 1000
        except Exception as e:
            logger.debug(f"WebSocket check error: {e}")
            return ConnectionStatus.DISCONNECTED, (time.time() - start_time) * 1000
    
    def _check_sse_endpoint(self, endpoint: DashboardEndpoint) -> Tuple[ConnectionStatus, float]:
        """Check Server-Sent Events endpoint connectivity."""
        start_time = time.time()
        
        try:
            headers = endpoint.auth_headers or {}
            headers['Accept'] = 'text/event-stream'
            
            response = requests.get(endpoint.url, headers=headers, 
                                  timeout=endpoint.timeout_seconds, stream=True)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Try to read first event
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith('data:'):
                        response.close()
                        return ConnectionStatus.CONNECTED, latency_ms
                    elif line.strip():  # Any non-empty line indicates activity
                        response.close()
                        return ConnectionStatus.CONNECTED, latency_ms
                
                response.close()
                return ConnectionStatus.DEGRADED, latency_ms
            else:
                return ConnectionStatus.DISCONNECTED, latency_ms
        
        except Exception as e:
            logger.debug(f"SSE check error: {e}")
            return ConnectionStatus.DISCONNECTED, (time.time() - start_time) * 1000
    
    def _estimate_bandwidth(self, endpoint: DashboardEndpoint) -> float:
        """Estimate bandwidth to endpoint."""
        try:
            # Send a small test request and measure throughput
            test_data = {'test': 'bandwidth_test', 'data': 'x' * 1024}  # 1KB test
            
            start_time = time.time()
            response = requests.post(endpoint.url, json=test_data, timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                # Calculate bandwidth (very rough estimate)
                data_size_mb = len(json.dumps(test_data).encode()) / (1024 * 1024)
                bandwidth_mbps = (data_size_mb / duration) if duration > 0 else 0
                return bandwidth_mbps
        
        except:
            pass
        
        return 0.0
    
    def _maintain_websocket_connection(self, endpoint_id: str, endpoint: DashboardEndpoint):
        """Maintain persistent WebSocket connection for real-time monitoring."""
        if endpoint_id in self.websocket_connections:
            ws = self.websocket_connections[endpoint_id]
            try:
                # Check if connection is still alive
                ws.ping()
                return
            except:
                # Connection is dead, remove it
                del self.websocket_connections[endpoint_id]
        
        # Create new connection
        try:
            def on_message(ws, message):
                # Handle incoming messages for data verification
                try:
                    data = json.loads(message)
                    if '_tracking_id' in data:
                        self.confirm_data_arrival(data['_tracking_id'], data)
                except:
                    pass
            
            def on_error(ws, error):
                logger.warning(f"WebSocket error for {endpoint_id}: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket closed for {endpoint_id}")
                if endpoint_id in self.websocket_connections:
                    del self.websocket_connections[endpoint_id]
            
            ws = websocket.WebSocketApp(
                endpoint.url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.websocket_connections[endpoint_id] = ws
            
            # Start connection in separate thread
            threading.Thread(target=ws.run_forever, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error creating WebSocket connection for {endpoint_id}: {e}")
    
    def _record_connectivity_event(self, endpoint_id: str, status: ConnectionStatus, 
                                 metrics: ConnectionMetrics):
        """Record connectivity event if status changed."""
        previous_status = getattr(self, '_previous_status', {}).get(endpoint_id)
        
        if not hasattr(self, '_previous_status'):
            self._previous_status = {}
        
        if previous_status != status:
            event = ConnectivityEvent(
                event_id=f"conn_{endpoint_id}_{int(time.time())}",
                timestamp=datetime.now(),
                endpoint_id=endpoint_id,
                event_type=f"status_change_{status.value}",
                severity="info" if status == ConnectionStatus.CONNECTED else "warning",
                description=f"Connection status changed to {status.value}",
                metrics={
                    'latency_ms': metrics.latency_ms,
                    'previous_status': previous_status.value if previous_status else None,
                    'new_status': status.value
                }
            )
            
            self.connectivity_events.append(event)
            self._trigger_alerts(event)
            
            self._previous_status[endpoint_id] = status
    
    def _check_alert_conditions(self, endpoint_id: str, status: ConnectionStatus, 
                               metrics: ConnectionMetrics):
        """Check if alert conditions are met."""
        alerts_triggered = []
        
        # Check latency threshold
        if metrics.latency_ms > self.alert_thresholds['latency_ms']:
            alerts_triggered.append(f"High latency: {metrics.latency_ms:.1f}ms")
        
        # Check connection status
        if status == ConnectionStatus.DISCONNECTED:
            alerts_triggered.append("Connection lost")
        elif status == ConnectionStatus.DEGRADED:
            alerts_triggered.append("Connection degraded")
        
        # Check data flow
        if endpoint_id in self.data_flow_metrics:
            flow_metrics = self.data_flow_metrics[endpoint_id]
            
            if flow_metrics.delivery_success_rate < self.alert_thresholds['delivery_success_rate']:
                alerts_triggered.append(f"Low delivery success rate: {flow_metrics.delivery_success_rate:.1f}%")
            
            if flow_metrics.data_freshness_seconds > self.alert_thresholds['data_freshness_seconds']:
                alerts_triggered.append(f"Stale data: {flow_metrics.data_freshness_seconds:.1f}s")
        
        # Trigger alerts
        for alert_message in alerts_triggered:
            event = ConnectivityEvent(
                event_id=f"alert_{endpoint_id}_{int(time.time())}",
                timestamp=datetime.now(),
                endpoint_id=endpoint_id,
                event_type="alert",
                severity="warning",
                description=alert_message,
                metrics={'latency_ms': metrics.latency_ms}
            )
            
            self._trigger_alerts(event)
    
    def _trigger_alerts(self, event: ConnectivityEvent):
        """Trigger alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _update_data_flow_metrics(self):
        """Update data flow metrics based on recent confirmations."""
        for endpoint_id in self.dashboard_endpoints:
            if endpoint_id not in self.data_flow_metrics:
                continue
            
            flow_metrics = self.data_flow_metrics[endpoint_id]
            
            # Get recent confirmations for this endpoint
            recent_confirmations = [
                r for r in self.pending_data_confirmations.values()
                if (r['endpoint_id'] == endpoint_id and 
                    r['confirmed'] and
                    (datetime.now() - r['arrival_time']).total_seconds() < 300)  # Last 5 minutes
            ]
            
            if recent_confirmations:
                # Calculate throughput
                time_window = 300  # 5 minutes
                flow_metrics.throughput_records_per_second = len(recent_confirmations) / time_window
                
                # Calculate average payload size
                if hasattr(self, '_payload_sizes'):
                    recent_sizes = [size for size in self._payload_sizes if size > 0]
                    if recent_sizes:
                        flow_metrics.average_payload_size_kb = sum(recent_sizes) / len(recent_sizes) / 1024
    
    def _generate_tracking_id(self, data: Dict[str, Any], endpoint_id: str) -> str:
        """Generate unique tracking ID for data."""
        content = json.dumps(data, sort_keys=True, default=str)
        timestamp = str(int(time.time() * 1000000))
        combined = f"{endpoint_id}:{timestamp}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _calculate_data_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity verification."""
        # Remove tracking metadata for checksum calculation
        clean_data = {k: v for k, v in data.items() 
                     if not k.startswith('_tracking') and not k.startswith('_sent') and k != '_checksum'}
        content = json.dumps(clean_data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_recent_events(self, hours: int = 24, endpoint_id: Optional[str] = None,
                         severity: Optional[str] = None) -> List[ConnectivityEvent]:
        """Get recent connectivity events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [e for e in self.connectivity_events if e.timestamp >= cutoff_time]
        
        if endpoint_id:
            recent_events = [e for e in recent_events if e.endpoint_id == endpoint_id]
        
        if severity:
            recent_events = [e for e in recent_events if e.severity == severity]
        
        # Sort by timestamp (most recent first)
        recent_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return recent_events
    
    def get_endpoint_performance_history(self, endpoint_id: str, hours: int = 24) -> Dict[str, List[float]]:
        """Get performance history for an endpoint."""
        history = self.performance_history.get(endpoint_id, deque())
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_history = [h for h in history if h['timestamp'] >= cutoff_time]
        
        return {
            'timestamps': [h['timestamp'].isoformat() for h in recent_history],
            'latency_ms': [h['latency_ms'] for h in recent_history],
            'success_rate': [h['success_rate'] for h in recent_history]
        }
    
    def shutdown(self):
        """Shutdown connectivity monitor."""
        self.stop_monitoring()
        logger.info("Analytics Connectivity Monitor shutdown")