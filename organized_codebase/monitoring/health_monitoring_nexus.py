"""
Health Monitoring Nexus - Archive-Derived Reliability System
===========================================================

Comprehensive health monitoring system with predictive analytics,
automated recovery, and intelligent alerting mechanisms.

Author: Agent C Security Framework
Created: 2025-08-21
"""

import logging
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import os
import statistics

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"

class MetricType(Enum):
    """Health metric types."""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    AVAILABILITY = "availability"
    LATENCY = "latency"
    SUCCESS_RATE = "success_rate"
    QUEUE_SIZE = "queue_size"
    CONNECTION_COUNT = "connection_count"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class HealthMetric:
    """Individual health metric record."""
    metric_id: str
    component_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_violated: bool = False
    anomaly_score: float = 0.0

@dataclass
class HealthCheck:
    """Health check definition."""
    check_id: str
    name: str
    component: str
    check_function: Callable[[], Tuple[bool, Dict[str, Any]]]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    enabled: bool = True
    consecutive_failures_threshold: int = 3
    recovery_threshold: int = 2
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    component: str
    metric_type: MetricType
    condition: str  # e.g., "value > 90", "value < 0.95"
    severity: AlertSeverity
    enabled: bool = True
    cooldown_seconds: int = 300
    last_triggered: Optional[datetime] = None
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class HealthAlert:
    """Health alert record."""
    alert_id: str
    rule_id: str
    component: str
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)

class HealthMonitoringNexus:
    """
    Comprehensive health monitoring system with predictive capabilities.
    """
    
    def __init__(self, db_path: str = "data/health_monitoring.db"):
        self.db_path = db_path
        
        # Initialize database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
        # Core monitoring data structures
        self.components: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, HealthAlert] = {}
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_history: Dict[str, List[HealthMetric]] = defaultdict(list)
        
        # Thresholds and baselines
        self.component_thresholds: Dict[str, Dict[MetricType, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        self.baseline_metrics: Dict[str, Dict[MetricType, float]] = defaultdict(lambda: defaultdict(float))
        
        # Predictive analytics
        self.anomaly_detectors: Dict[str, Any] = {}
        self.trend_analyzers: Dict[str, Any] = {}
        self.failure_predictors: Dict[str, Any] = {}
        
        # Statistics and analytics
        self.monitoring_stats = {
            'total_checks_performed': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'alerts_triggered': 0,
            'alerts_resolved': 0,
            'components_monitored': 0,
            'average_response_time': 0.0,
            'system_uptime_percentage': 100.0,
            'anomalies_detected': 0,
            'predictions_made': 0,
            'false_positives': 0,
            'true_positives': 0
        }
        
        # Configuration
        self.check_interval = 15  # Default check interval in seconds
        self.metric_retention_days = 30
        self.anomaly_detection_enabled = True
        self.predictive_analysis_enabled = True
        self.auto_recovery_enabled = True
        
        # Background processing
        self.monitoring_active = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.metrics_collector_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self.anomaly_detector_thread = threading.Thread(target=self._anomaly_detection_loop, daemon=True)
        self.alert_processor_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        
        # Start monitoring threads
        self.health_check_thread.start()
        self.metrics_collector_thread.start()
        
        if self.anomaly_detection_enabled:
            self.anomaly_detector_thread.start()
        
        self.alert_processor_thread.start()
        self.cleanup_thread.start()
        
        # Thread safety
        self.monitoring_lock = threading.RLock()
        
        # Notification system
        self.notification_handlers: Dict[str, Callable] = {}
        self.escalation_rules: List[Dict[str, Any]] = []
        
        logger.info("Health Monitoring Nexus initialized with predictive analytics")
    
    def _init_database(self):
        """Initialize health monitoring database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Components table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS components (
                        component_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        last_check TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Health metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS health_metrics (
                        metric_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        unit TEXT,
                        tags TEXT,
                        threshold_violated INTEGER DEFAULT 0,
                        anomaly_score REAL DEFAULT 0.0
                    )
                ''')
                
                # Health checks table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS health_checks (
                        check_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        component TEXT NOT NULL,
                        interval_seconds INTEGER DEFAULT 30,
                        timeout_seconds INTEGER DEFAULT 10,
                        enabled INTEGER DEFAULT 1,
                        consecutive_failures INTEGER DEFAULT 0,
                        consecutive_successes INTEGER DEFAULT 0,
                        last_check TEXT,
                        configuration TEXT
                    )
                ''')
                
                # Alerts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS health_alerts (
                        alert_id TEXT PRIMARY KEY,
                        rule_id TEXT NOT NULL,
                        component TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        severity INTEGER NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        resolved INTEGER DEFAULT 0,
                        resolved_at TEXT,
                        context TEXT
                    )
                ''')
                
                # Alert rules table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alert_rules (
                        rule_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        component TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        condition_text TEXT NOT NULL,
                        severity INTEGER NOT NULL,
                        enabled INTEGER DEFAULT 1,
                        cooldown_seconds INTEGER DEFAULT 300,
                        last_triggered TEXT,
                        configuration TEXT
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON health_metrics(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_component ON health_metrics(component_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON health_alerts(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_component ON health_alerts(component)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Health monitoring database initialization failed: {e}")
            raise
    
    def register_component(self, component_id: str, name: str, component_type: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a component for health monitoring."""
        try:
            with self.monitoring_lock:
                self.components[component_id] = {
                    'name': name,
                    'type': component_type,
                    'status': HealthStatus.UNKNOWN,
                    'last_check': None,
                    'metadata': metadata or {}
                }
                
                # Save to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO components
                        (component_id, name, type, status, last_check, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        component_id,
                        name,
                        component_type,
                        HealthStatus.UNKNOWN.value,
                        None,
                        json.dumps(metadata or {})
                    ))
                    conn.commit()
                
                self.monitoring_stats['components_monitored'] = len(self.components)
                logger.info(f"Registered component: {component_id} ({component_type})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            return False
    
    def add_health_check(self, check: HealthCheck) -> bool:
        """Add a health check for a component."""
        try:
            with self.monitoring_lock:
                self.health_checks[check.check_id] = check
                
                # Save to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO health_checks
                        (check_id, name, component, interval_seconds, timeout_seconds, 
                         enabled, consecutive_failures, consecutive_successes, last_check, configuration)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        check.check_id,
                        check.name,
                        check.component,
                        check.interval_seconds,
                        check.timeout_seconds,
                        1 if check.enabled else 0,
                        check.consecutive_failures,
                        check.consecutive_successes,
                        check.last_check.isoformat() if check.last_check else None,
                        json.dumps({
                            'threshold': check.consecutive_failures_threshold,
                            'recovery': check.recovery_threshold
                        })
                    ))
                    conn.commit()
                
                logger.info(f"Added health check: {check.check_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add health check {check.check_id}: {e}")
            return False
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add an alert rule for monitoring metrics."""
        try:
            with self.monitoring_lock:
                self.alert_rules[rule.rule_id] = rule
                
                # Save to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO alert_rules
                        (rule_id, name, component, metric_type, condition_text, 
                         severity, enabled, cooldown_seconds, last_triggered, configuration)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rule.rule_id,
                        rule.name,
                        rule.component,
                        rule.metric_type.value,
                        rule.condition,
                        rule.severity.value,
                        1 if rule.enabled else 0,
                        rule.cooldown_seconds,
                        rule.last_triggered.isoformat() if rule.last_triggered else None,
                        json.dumps({'notification_channels': rule.notification_channels})
                    ))
                    conn.commit()
                
                logger.info(f"Added alert rule: {rule.rule_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add alert rule {rule.rule_id}: {e}")
            return False
    
    def record_metric(self, component_name: str, metric_type: MetricType, 
                     value: float, unit: str = "", tags: Dict[str, str] = None) -> bool:
        """Record a health metric for a component."""
        try:
            metric = HealthMetric(
                metric_id=f"{component_name}_{metric_type.value}_{int(time.time() * 1000000)}",
                component_name=component_name,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                unit=unit,
                tags=tags or {},
                threshold_violated=False,
                anomaly_score=0.0
            )
            
            with self.monitoring_lock:
                # Store in memory
                metric_key = f"{component_name}_{metric_type.value}"
                self.metrics[metric_key].append(metric)
                
                # Check for threshold violations
                self._check_metric_thresholds(metric)
                
                # Anomaly detection
                if self.anomaly_detection_enabled:
                    anomaly_score = self._detect_anomaly(metric)
                    metric.anomaly_score = anomaly_score
                    
                    if anomaly_score > 0.8:  # High anomaly threshold
                        self.monitoring_stats['anomalies_detected'] += 1
                
                # Save to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO health_metrics
                        (metric_id, component_name, metric_type, value, timestamp, 
                         unit, tags, threshold_violated, anomaly_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metric.metric_id,
                        metric.component_name,
                        metric.metric_type.value,
                        metric.value,
                        metric.timestamp.isoformat(),
                        metric.unit,
                        json.dumps(metric.tags),
                        1 if metric.threshold_violated else 0,
                        metric.anomaly_score
                    ))
                    conn.commit()
                
                # Trigger alert processing
                self._process_metric_alerts(metric)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to record metric {component_name}_{metric_type.value}: {e}")
            return False
    
    def _health_check_loop(self):
        """Main health check processing loop."""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                with self.monitoring_lock:
                    for check_id, check in self.health_checks.items():
                        if not check.enabled:
                            continue
                        
                        # Check if it's time for this health check
                        if (not check.last_check or 
                            (current_time - check.last_check).total_seconds() >= check.interval_seconds):
                            
                            self._perform_health_check(check)
                
                time.sleep(5)  # Check every 5 seconds for due health checks
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(10)
    
    def _perform_health_check(self, check: HealthCheck):
        """Perform an individual health check."""
        try:
            check_start = time.time()
            check.last_check = datetime.now()
            
            self.monitoring_stats['total_checks_performed'] += 1
            
            # Execute the health check function with timeout
            try:
                success, check_data = self._execute_with_timeout(
                    check.check_function, 
                    check.timeout_seconds
                )
                
                check_duration = time.time() - check_start
                
                if success:
                    check.consecutive_failures = 0
                    check.consecutive_successes += 1
                    self.monitoring_stats['successful_checks'] += 1
                    
                    # Update component status
                    if check.component in self.components:
                        if check.consecutive_successes >= check.recovery_threshold:
                            self.components[check.component]['status'] = HealthStatus.HEALTHY
                        self.components[check.component]['last_check'] = datetime.now()
                    
                    # Record response time metric
                    self.record_metric(
                        component_name=check.component,
                        metric_type=MetricType.RESPONSE_TIME,
                        value=check_duration * 1000,  # Convert to milliseconds
                        unit="ms",
                        tags={'check_id': check.check_id}
                    )
                    
                else:
                    check.consecutive_successes = 0
                    check.consecutive_failures += 1
                    self.monitoring_stats['failed_checks'] += 1
                    
                    # Update component status based on failure severity
                    if check.component in self.components:
                        if check.consecutive_failures >= check.consecutive_failures_threshold:
                            self.components[check.component]['status'] = HealthStatus.CRITICAL
                        elif check.consecutive_failures >= check.consecutive_failures_threshold // 2:
                            self.components[check.component]['status'] = HealthStatus.WARNING
                        else:
                            self.components[check.component]['status'] = HealthStatus.DEGRADED
                    
                    # Generate alert for check failure
                    self._generate_health_check_alert(check, check_data)
                
                # Update average response time
                total_response_time = (
                    self.monitoring_stats['average_response_time'] * 
                    (self.monitoring_stats['total_checks_performed'] - 1) + 
                    check_duration
                ) / self.monitoring_stats['total_checks_performed']
                self.monitoring_stats['average_response_time'] = total_response_time
                
            except TimeoutError:
                check.consecutive_failures += 1
                check.consecutive_successes = 0
                self.monitoring_stats['failed_checks'] += 1
                
                if check.component in self.components:
                    self.components[check.component]['status'] = HealthStatus.CRITICAL
                
                self._generate_health_check_alert(check, {'error': 'Health check timeout'})
                
        except Exception as e:
            logger.error(f"Health check execution failed for {check.check_id}: {e}")
            check.consecutive_failures += 1
            check.consecutive_successes = 0
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health monitoring report."""
        with self.monitoring_lock:
            # Calculate system health summary
            component_statuses = [comp['status'] for comp in self.components.values()]
            healthy_count = sum(1 for status in component_statuses if status == HealthStatus.HEALTHY)
            total_components = len(component_statuses)
            
            system_health = HealthStatus.HEALTHY
            if total_components > 0:
                health_percentage = (healthy_count / total_components) * 100
                if health_percentage < 50:
                    system_health = HealthStatus.CRITICAL
                elif health_percentage < 75:
                    system_health = HealthStatus.WARNING
                elif health_percentage < 90:
                    system_health = HealthStatus.DEGRADED
            
            # Get recent alerts
            recent_alerts = sorted(
                [alert for alert in self.active_alerts.values()],
                key=lambda x: x.timestamp,
                reverse=True
            )[:10]
            
            # Calculate uptime
            total_checks = self.monitoring_stats['total_checks_performed']
            successful_checks = self.monitoring_stats['successful_checks']
            uptime_percentage = (successful_checks / max(1, total_checks)) * 100
            
            return {
                'system_health': {
                    'overall_status': system_health.value,
                    'health_percentage': (healthy_count / max(1, total_components)) * 100,
                    'uptime_percentage': uptime_percentage,
                    'components_monitored': total_components,
                    'healthy_components': healthy_count,
                    'degraded_components': sum(1 for s in component_statuses if s == HealthStatus.DEGRADED),
                    'warning_components': sum(1 for s in component_statuses if s == HealthStatus.WARNING),
                    'critical_components': sum(1 for s in component_statuses if s == HealthStatus.CRITICAL),
                    'failed_components': sum(1 for s in component_statuses if s == HealthStatus.FAILED)
                },
                'components': {
                    comp_id: {
                        'name': comp['name'],
                        'type': comp['type'],
                        'status': comp['status'].value if hasattr(comp['status'], 'value') else comp['status'],
                        'last_check': comp['last_check'].isoformat() if comp['last_check'] else None,
                        'metadata': comp['metadata']
                    }
                    for comp_id, comp in self.components.items()
                },
                'active_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'component': alert.component,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'resolved': alert.resolved
                    }
                    for alert in recent_alerts
                ],
                'monitoring_statistics': self.monitoring_stats.copy(),
                'health_checks': {
                    check_id: {
                        'name': check.name,
                        'component': check.component,
                        'enabled': check.enabled,
                        'interval_seconds': check.interval_seconds,
                        'consecutive_failures': check.consecutive_failures,
                        'consecutive_successes': check.consecutive_successes,
                        'last_check': check.last_check.isoformat() if check.last_check else None
                    }
                    for check_id, check in self.health_checks.items()
                },
                'alert_rules': {
                    rule_id: {
                        'name': rule.name,
                        'component': rule.component,
                        'metric_type': rule.metric_type.value,
                        'condition': rule.condition,
                        'severity': rule.severity.value,
                        'enabled': rule.enabled,
                        'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                    }
                    for rule_id, rule in self.alert_rules.items()
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Shutdown health monitoring nexus."""
        self.monitoring_active = False
        
        # Wait for threads to complete
        for thread in [self.health_check_thread, self.metrics_collector_thread,
                      self.anomaly_detector_thread, self.alert_processor_thread,
                      self.cleanup_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Health Monitoring Nexus shutdown - Final Stats: {self.monitoring_stats}")

# Global health monitoring nexus instance
health_monitoring_nexus = None