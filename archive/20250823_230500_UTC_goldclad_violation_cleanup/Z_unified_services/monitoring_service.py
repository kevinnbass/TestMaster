#!/usr/bin/env python3
"""
Unified Monitoring Service Module - Agent Z Phase 2
Real-time performance monitoring and health tracking

Provides comprehensive monitoring capabilities including:
- Performance metrics collection and analysis
- System health monitoring and alerting  
- Service availability tracking
- <50ms latency monitoring and optimization
- Resource utilization monitoring
- Error tracking and analysis
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


class AlertLevel(Enum):
    """Monitoring alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    response_time_ms: float
    throughput_requests_per_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_connections: int
    error_rate: float
    queue_size: int


@dataclass
class HealthCheck:
    """Health check result"""
    service_name: str
    status: HealthStatus
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class MonitoringAlert:
    """Monitoring alert"""
    alert_id: str
    level: AlertLevel
    service: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class UnifiedMonitoringService:
    """
    Unified monitoring service providing comprehensive real-time
    monitoring, health checks, and performance tracking.
    """
    
    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1min intervals
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.service_monitors: Dict[str, Callable[[], HealthCheck]] = {}
        
        # Alert management
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_handlers: List[Callable[[MonitoringAlert], None]] = []
        
        # Performance thresholds
        self.thresholds = {
            'max_response_time_ms': 50.0,  # <50ms latency target
            'max_error_rate': 0.05,  # 5% error rate
            'max_cpu_usage': 80.0,  # 80% CPU usage
            'max_memory_usage_mb': 1024.0,  # 1GB memory
            'min_throughput_rps': 10.0,  # 10 requests per second minimum
            'max_queue_size': 100  # Maximum queue size
        }
        
        # Statistics
        self.monitoring_stats = {
            'checks_performed': 0,
            'alerts_generated': 0,
            'performance_violations': 0,
            'start_time': datetime.now()
        }
        
        # Setup default service monitors
        self._setup_default_monitors()
        
        logger.info("Unified Monitoring Service initialized")
    
    def start(self):
        """Start the monitoring service"""
        if self.running:
            logger.warning("Monitoring service is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Unified Monitoring Service started")
    
    def stop(self):
        """Stop the monitoring service"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Unified Monitoring Service stopped")
    
    def register_service_monitor(self, service_name: str, 
                                monitor_func: Callable[[], HealthCheck]):
        """Register a service health monitor"""
        self.service_monitors[service_name] = monitor_func
        logger.info(f"Registered monitor for service: {service_name}")
    
    def add_alert_handler(self, handler: Callable[[MonitoringAlert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def record_performance_metrics(self, response_time_ms: float, 
                                 throughput_rps: float,
                                 active_connections: int,
                                 error_rate: float = 0.0,
                                 queue_size: int = 0):
        """Record performance metrics"""
        try:
            # Get system metrics (simplified - in real implementation would use psutil)
            cpu_usage = 0.0  # Would get actual CPU usage
            memory_usage = 0.0  # Would get actual memory usage
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                throughput_requests_per_sec=throughput_rps,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                active_connections=active_connections,
                error_rate=error_rate,
                queue_size=queue_size
            )
            
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Check performance thresholds
            self._check_performance_thresholds(metrics)
            
        except Exception as e:
            logger.error(f"Failed to record performance metrics: {e}")
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            overall_status = HealthStatus.HEALTHY
            services_status = {}
            
            for service_name, health_check in self.health_checks.items():
                services_status[service_name] = asdict(health_check)
                
                # Determine overall status
                if health_check.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif health_check.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.WARNING
            
            return {
                'overall_status': overall_status.value,
                'services': services_status,
                'active_alerts': len(self.active_alerts),
                'last_check': datetime.now().isoformat(),
                'performance_summary': self._get_performance_summary()
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'No metrics available for requested period'}
            
            # Calculate statistics
            response_times = [m.response_time_ms for m in recent_metrics]
            throughputs = [m.throughput_requests_per_sec for m in recent_metrics]
            error_rates = [m.error_rate for m in recent_metrics]
            
            return {
                'metrics_count': len(recent_metrics),
                'time_period_hours': hours,
                'response_time': {
                    'avg_ms': sum(response_times) / len(response_times),
                    'min_ms': min(response_times),
                    'max_ms': max(response_times),
                    'target_ms': self.thresholds['max_response_time_ms'],
                    'within_target': sum(1 for rt in response_times if rt <= self.thresholds['max_response_time_ms']) / len(response_times) * 100
                },
                'throughput': {
                    'avg_rps': sum(throughputs) / len(throughputs),
                    'min_rps': min(throughputs),
                    'max_rps': max(throughputs)
                },
                'error_rate': {
                    'avg_percent': sum(error_rates) / len(error_rates) * 100,
                    'max_percent': max(error_rates) * 100
                },
                'performance_score': self._calculate_performance_score(recent_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {'error': str(e)}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active monitoring alerts"""
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.now()
                
                # Move to history and remove from active
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert {alert_id} resolved")
                return True
            else:
                logger.warning(f"Alert {alert_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    def _setup_default_monitors(self):
        """Setup default service monitors"""
        # WebSocket service monitor
        def websocket_monitor() -> HealthCheck:
            try:
                from .websocket_service import get_websocket_service
                ws_service = get_websocket_service()
                
                start_time = time.time()
                health_data = ws_service.health_check()
                response_time = (time.time() - start_time) * 1000
                
                status = HealthStatus.HEALTHY if health_data['status'] == 'healthy' else HealthStatus.ERROR
                
                return HealthCheck(
                    service_name='WebSocketService',
                    status=status,
                    response_time_ms=response_time,
                    last_check=datetime.now(),
                    details=health_data
                )
                
            except Exception as e:
                return HealthCheck(
                    service_name='WebSocketService',
                    status=HealthStatus.ERROR,
                    response_time_ms=0,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
        
        # Coordination service monitor
        def coordination_monitor() -> HealthCheck:
            try:
                from .coordination_service import get_coordination_service
                coord_service = get_coordination_service()
                
                start_time = time.time()
                swarm_status = coord_service.get_swarm_status()
                response_time = (time.time() - start_time) * 1000
                
                # Determine health based on active agents and errors
                active_ratio = swarm_status.get('active_agents', 0) / max(swarm_status.get('total_agents', 1), 1)
                if active_ratio >= 0.8:
                    status = HealthStatus.HEALTHY
                elif active_ratio >= 0.5:
                    status = HealthStatus.WARNING
                else:
                    status = HealthStatus.CRITICAL
                
                return HealthCheck(
                    service_name='CoordinationService',
                    status=status,
                    response_time_ms=response_time,
                    last_check=datetime.now(),
                    details=swarm_status
                )
                
            except Exception as e:
                return HealthCheck(
                    service_name='CoordinationService', 
                    status=HealthStatus.ERROR,
                    response_time_ms=0,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
        
        # Register default monitors
        self.register_service_monitor('WebSocketService', websocket_monitor)
        self.register_service_monitor('CoordinationService', coordination_monitor)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Perform health checks
                for service_name, monitor_func in self.service_monitors.items():
                    health_check = monitor_func()
                    self.health_checks[service_name] = health_check
                    self.monitoring_stats['checks_performed'] += 1
                    
                    # Generate alerts for unhealthy services
                    if health_check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.ERROR]:
                        self._generate_health_alert(health_check)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance metrics exceed thresholds"""
        violations = []
        
        if metrics.response_time_ms > self.thresholds['max_response_time_ms']:
            violations.append(f"Response time {metrics.response_time_ms}ms exceeds {self.thresholds['max_response_time_ms']}ms")
        
        if metrics.error_rate > self.thresholds['max_error_rate']:
            violations.append(f"Error rate {metrics.error_rate*100:.1f}% exceeds {self.thresholds['max_error_rate']*100:.1f}%")
        
        if metrics.throughput_requests_per_sec < self.thresholds['min_throughput_rps']:
            violations.append(f"Throughput {metrics.throughput_requests_per_sec} rps below minimum {self.thresholds['min_throughput_rps']} rps")
        
        if metrics.queue_size > self.thresholds['max_queue_size']:
            violations.append(f"Queue size {metrics.queue_size} exceeds maximum {self.thresholds['max_queue_size']}")
        
        # Generate performance alerts
        for violation in violations:
            self._generate_performance_alert(violation, metrics)
            self.monitoring_stats['performance_violations'] += 1
    
    def _generate_health_alert(self, health_check: HealthCheck):
        """Generate alert for unhealthy service"""
        alert_id = f"health_{health_check.service_name}_{int(datetime.now().timestamp())}"
        
        level_map = {
            HealthStatus.WARNING: AlertLevel.WARNING,
            HealthStatus.CRITICAL: AlertLevel.CRITICAL,
            HealthStatus.ERROR: AlertLevel.EMERGENCY
        }
        
        alert = MonitoringAlert(
            alert_id=alert_id,
            level=level_map.get(health_check.status, AlertLevel.INFO),
            service=health_check.service_name,
            message=f"Service health check failed: {health_check.error_message or 'Status: ' + health_check.status.value}",
            details=asdict(health_check),
            timestamp=datetime.now()
        )
        
        self._process_alert(alert)
    
    def _generate_performance_alert(self, violation: str, metrics: PerformanceMetrics):
        """Generate alert for performance threshold violation"""
        alert_id = f"perf_{int(datetime.now().timestamp())}"
        
        alert = MonitoringAlert(
            alert_id=alert_id,
            level=AlertLevel.WARNING if 'Response time' in violation else AlertLevel.CRITICAL,
            service='PerformanceMonitoring',
            message=f"Performance threshold violation: {violation}",
            details=asdict(metrics),
            timestamp=datetime.now()
        )
        
        self._process_alert(alert)
    
    def _process_alert(self, alert: MonitoringAlert):
        """Process and handle new alert"""
        self.active_alerts[alert.alert_id] = alert
        self.monitoring_stats['alerts_generated'] += 1
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(f"Alert generated: {alert.message}")
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from current metrics"""
        if not self.current_metrics:
            return {'status': 'no_metrics'}
        
        metrics = self.current_metrics
        return {
            'response_time_ms': metrics.response_time_ms,
            'throughput_rps': metrics.throughput_requests_per_sec,
            'active_connections': metrics.active_connections,
            'error_rate_percent': metrics.error_rate * 100,
            'queue_size': metrics.queue_size,
            'within_latency_target': metrics.response_time_ms <= self.thresholds['max_response_time_ms']
        }
    
    def _calculate_performance_score(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics_list:
            return 0.0
        
        score = 100.0
        
        # Response time score (40% weight)
        avg_response_time = sum(m.response_time_ms for m in metrics_list) / len(metrics_list)
        if avg_response_time <= self.thresholds['max_response_time_ms']:
            response_score = 40.0
        else:
            response_score = max(0, 40.0 - (avg_response_time - self.thresholds['max_response_time_ms']) / 10)
        
        # Error rate score (30% weight)
        avg_error_rate = sum(m.error_rate for m in metrics_list) / len(metrics_list)
        if avg_error_rate <= self.thresholds['max_error_rate']:
            error_score = 30.0
        else:
            error_score = max(0, 30.0 - (avg_error_rate - self.thresholds['max_error_rate']) * 600)
        
        # Throughput score (30% weight)
        avg_throughput = sum(m.throughput_requests_per_sec for m in metrics_list) / len(metrics_list)
        if avg_throughput >= self.thresholds['min_throughput_rps']:
            throughput_score = 30.0
        else:
            throughput_score = max(0, 30.0 * (avg_throughput / self.thresholds['min_throughput_rps']))
        
        return round(response_score + error_score + throughput_score, 1)


# Global service instance
_monitoring_service: Optional[UnifiedMonitoringService] = None


def get_monitoring_service() -> UnifiedMonitoringService:
    """Get global monitoring service instance"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = UnifiedMonitoringService()
    return _monitoring_service