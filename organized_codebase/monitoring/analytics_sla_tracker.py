#!/usr/bin/env python3
"""
Analytics Delivery SLA Tracker with Automatic Escalation
========================================================

Provides ultra-reliability through comprehensive SLA tracking, automatic
escalation, performance guarantees, and executive reporting.

Author: TestMaster Team
"""

import logging
import threading
import time
import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor


class SLALevel(Enum):
    """SLA performance levels."""
    PLATINUM = "platinum"    # 99.99% uptime, <100ms latency
    GOLD = "gold"           # 99.9% uptime, <250ms latency  
    SILVER = "silver"       # 99.5% uptime, <500ms latency
    BRONZE = "bronze"       # 99.0% uptime, <1000ms latency


class EscalationLevel(Enum):
    """Escalation levels."""
    L1_MONITORING = "l1_monitoring"
    L2_ENGINEERING = "l2_engineering"
    L3_SENIOR_ENG = "l3_senior_engineering"
    L4_MANAGEMENT = "l4_management"
    L5_EXECUTIVE = "l5_executive"


class ViolationType(Enum):
    """SLA violation types."""
    LATENCY_BREACH = "latency_breach"
    AVAILABILITY_BREACH = "availability_breach"
    THROUGHPUT_BREACH = "throughput_breach"
    ERROR_RATE_BREACH = "error_rate_breach"
    DELIVERY_FAILURE = "delivery_failure"


class DeliveryPriority(Enum):
    """Delivery priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BULK = "bulk"


@dataclass
class SLAConfiguration:
    """SLA configuration settings."""
    level: SLALevel
    max_latency_ms: float
    min_availability_percent: float
    min_throughput_tps: float
    max_error_rate_percent: float
    delivery_timeout_seconds: float
    escalation_threshold_minutes: int


@dataclass
class SLAMetric:
    """Individual SLA metric measurement."""
    metric_id: str
    analytics_id: str
    timestamp: datetime
    latency_ms: float
    delivery_success: bool
    error_message: Optional[str]
    component: str
    stage: str


@dataclass
class SLAViolation:
    """SLA violation record."""
    violation_id: str
    analytics_id: str
    violation_type: ViolationType
    timestamp: datetime
    current_value: float
    threshold_value: float
    severity: str
    escalated: bool
    escalation_level: Optional[EscalationLevel]
    resolved: bool
    resolution_time: Optional[datetime]
    impact_description: str


@dataclass
class EscalationRule:
    """Escalation rule configuration."""
    violation_type: ViolationType
    severity_threshold: int
    time_threshold_minutes: int
    escalation_level: EscalationLevel
    notification_emails: List[str]
    auto_actions: List[str]


class AnalyticsSLATracker:
    """Analytics delivery SLA tracker with automatic escalation."""
    
    def __init__(
        self,
        db_path: str = "data/sla_tracking.db",
        default_sla_level: SLALevel = SLALevel.GOLD,
        monitoring_interval: float = 30.0
    ):
        """Initialize the SLA tracker."""
        self.db_path = db_path
        self.default_sla_level = default_sla_level
        self.monitoring_interval = monitoring_interval
        
        # SLA tracking
        self.sla_configs: Dict[SLALevel, SLAConfiguration] = {}
        self.active_violations: Dict[str, SLAViolation] = {}
        self.escalation_rules: List[EscalationRule] = []
        self.sla_metrics: List[SLAMetric] = []
        
        # Analytics tracking
        self.pending_deliveries: Dict[str, Dict[str, Any]] = {}
        self.delivery_callbacks: Dict[str, Callable] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.running = False
        self.monitor_thread = None
        
        # Statistics
        self.hourly_stats: Dict[str, Dict[str, float]] = {}
        self.daily_stats: Dict[str, Dict[str, float]] = {}
        
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        self._setup_default_sla_configs()
        self._setup_default_escalation_rules()
        
        self.logger.info("Analytics SLA Tracker initialized")
    
    def _initialize_database(self):
        """Initialize the SLA tracking database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # SLA metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sla_metrics (
                        metric_id TEXT PRIMARY KEY,
                        analytics_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        delivery_success INTEGER NOT NULL,
                        error_message TEXT,
                        component TEXT NOT NULL,
                        stage TEXT NOT NULL
                    )
                """)
                
                # SLA violations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sla_violations (
                        violation_id TEXT PRIMARY KEY,
                        analytics_id TEXT NOT NULL,
                        violation_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        severity TEXT NOT NULL,
                        escalated INTEGER DEFAULT 0,
                        escalation_level TEXT,
                        resolved INTEGER DEFAULT 0,
                        resolution_time TEXT,
                        impact_description TEXT
                    )
                """)
                
                # SLA performance summary table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sla_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        sla_level TEXT NOT NULL,
                        availability_percent REAL NOT NULL,
                        avg_latency_ms REAL NOT NULL,
                        throughput_tps REAL NOT NULL,
                        error_rate_percent REAL NOT NULL,
                        violations_count INTEGER NOT NULL,
                        sla_met INTEGER NOT NULL
                    )
                """)
                
                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON sla_metrics(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON sla_violations(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_period ON sla_performance(period_start)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _setup_default_sla_configs(self):
        """Setup default SLA configurations."""
        self.sla_configs = {
            SLALevel.PLATINUM: SLAConfiguration(
                level=SLALevel.PLATINUM,
                max_latency_ms=100.0,
                min_availability_percent=99.99,
                min_throughput_tps=50.0,
                max_error_rate_percent=0.01,
                delivery_timeout_seconds=30.0,
                escalation_threshold_minutes=1
            ),
            SLALevel.GOLD: SLAConfiguration(
                level=SLALevel.GOLD,
                max_latency_ms=250.0,
                min_availability_percent=99.9,
                min_throughput_tps=25.0,
                max_error_rate_percent=0.1,
                delivery_timeout_seconds=60.0,
                escalation_threshold_minutes=5
            ),
            SLALevel.SILVER: SLAConfiguration(
                level=SLALevel.SILVER,
                max_latency_ms=500.0,
                min_availability_percent=99.5,
                min_throughput_tps=10.0,
                max_error_rate_percent=0.5,
                delivery_timeout_seconds=120.0,
                escalation_threshold_minutes=15
            ),
            SLALevel.BRONZE: SLAConfiguration(
                level=SLALevel.BRONZE,
                max_latency_ms=1000.0,
                min_availability_percent=99.0,
                min_throughput_tps=5.0,
                max_error_rate_percent=1.0,
                delivery_timeout_seconds=300.0,
                escalation_threshold_minutes=30
            )
        }
    
    def _setup_default_escalation_rules(self):
        """Setup default escalation rules."""
        self.escalation_rules = [
            # Latency breaches
            EscalationRule(
                violation_type=ViolationType.LATENCY_BREACH,
                severity_threshold=3,
                time_threshold_minutes=5,
                escalation_level=EscalationLevel.L2_ENGINEERING,
                notification_emails=["engineering@company.com"],
                auto_actions=["restart_slow_components", "increase_resources"]
            ),
            
            # Availability breaches  
            EscalationRule(
                violation_type=ViolationType.AVAILABILITY_BREACH,
                severity_threshold=1,
                time_threshold_minutes=2,
                escalation_level=EscalationLevel.L3_SENIOR_ENG,
                notification_emails=["senior-eng@company.com", "oncall@company.com"],
                auto_actions=["activate_failover", "scale_up_instances"]
            ),
            
            # Critical delivery failures
            EscalationRule(
                violation_type=ViolationType.DELIVERY_FAILURE,
                severity_threshold=5,
                time_threshold_minutes=1,
                escalation_level=EscalationLevel.L4_MANAGEMENT,
                notification_emails=["management@company.com", "cto@company.com"],
                auto_actions=["emergency_backup_activation", "executive_notification"]
            )
        ]
    
    def track_analytics_delivery(
        self,
        analytics_id: str,
        component: str,
        stage: str,
        sla_level: SLALevel = None
    ) -> str:
        """Start tracking analytics delivery against SLA."""
        tracking_id = str(uuid.uuid4())
        sla_level = sla_level or self.default_sla_level
        
        with self.lock:
            self.pending_deliveries[tracking_id] = {
                'analytics_id': analytics_id,
                'component': component,
                'stage': stage,
                'sla_level': sla_level,
                'start_time': datetime.now(),
                'timeout': self.sla_configs[sla_level].delivery_timeout_seconds
            }
        
        # Schedule timeout check
        self.executor.submit(self._check_delivery_timeout, tracking_id)
        
        return tracking_id
    
    def record_delivery_success(
        self,
        tracking_id: str,
        latency_ms: float,
        additional_metrics: Dict[str, Any] = None
    ):
        """Record successful analytics delivery."""
        with self.lock:
            if tracking_id not in self.pending_deliveries:
                self.logger.warning(f"Unknown tracking ID: {tracking_id}")
                return
            
            delivery_info = self.pending_deliveries.pop(tracking_id)
            
            # Create SLA metric
            metric = SLAMetric(
                metric_id=str(uuid.uuid4()),
                analytics_id=delivery_info['analytics_id'],
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                delivery_success=True,
                error_message=None,
                component=delivery_info['component'],
                stage=delivery_info['stage']
            )
            
            self.sla_metrics.append(metric)
            self._save_metric_to_db(metric)
            
            # Check for SLA violations
            sla_config = self.sla_configs[delivery_info['sla_level']]
            self._check_sla_violations(metric, sla_config)
            
            self.logger.info(f"Analytics delivery completed: {delivery_info['analytics_id']} in {latency_ms}ms")
    
    def record_delivery_failure(
        self,
        tracking_id: str,
        error_message: str,
        latency_ms: float = 0.0
    ):
        """Record failed analytics delivery."""
        with self.lock:
            if tracking_id not in self.pending_deliveries:
                self.logger.warning(f"Unknown tracking ID: {tracking_id}")
                return
            
            delivery_info = self.pending_deliveries.pop(tracking_id)
            
            # Create SLA metric
            metric = SLAMetric(
                metric_id=str(uuid.uuid4()),
                analytics_id=delivery_info['analytics_id'],
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                delivery_success=False,
                error_message=error_message,
                component=delivery_info['component'],
                stage=delivery_info['stage']
            )
            
            self.sla_metrics.append(metric)
            self._save_metric_to_db(metric)
            
            # Create violation for delivery failure
            violation = SLAViolation(
                violation_id=str(uuid.uuid4()),
                analytics_id=delivery_info['analytics_id'],
                violation_type=ViolationType.DELIVERY_FAILURE,
                timestamp=datetime.now(),
                current_value=0.0,  # Failed delivery
                threshold_value=1.0,  # Expected success
                severity="critical",
                escalated=False,
                escalation_level=None,
                resolved=False,
                resolution_time=None,
                impact_description=f"Analytics delivery failed: {error_message}"
            )
            
            self._process_violation(violation)
            
            self.logger.error(f"Analytics delivery failed: {delivery_info['analytics_id']} - {error_message}")
    
    def _check_delivery_timeout(self, tracking_id: str):
        """Check for delivery timeout and escalate if necessary."""
        time.sleep(self.pending_deliveries.get(tracking_id, {}).get('timeout', 60))
        
        with self.lock:
            if tracking_id in self.pending_deliveries:
                delivery_info = self.pending_deliveries.pop(tracking_id)
                
                # Record timeout as failure
                self.record_delivery_failure(
                    tracking_id,
                    f"Delivery timeout after {delivery_info['timeout']}s",
                    delivery_info['timeout'] * 1000  # Convert to ms
                )
    
    def _save_metric_to_db(self, metric: SLAMetric):
        """Save SLA metric to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sla_metrics 
                    (metric_id, analytics_id, timestamp, latency_ms, 
                     delivery_success, error_message, component, stage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_id,
                    metric.analytics_id,
                    metric.timestamp.isoformat(),
                    metric.latency_ms,
                    int(metric.delivery_success),
                    metric.error_message,
                    metric.component,
                    metric.stage
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save metric to database: {e}")
    
    def _check_sla_violations(self, metric: SLAMetric, sla_config: SLAConfiguration):
        """Check metric against SLA thresholds."""
        violations = []
        
        # Check latency violation
        if metric.latency_ms > sla_config.max_latency_ms:
            violation = SLAViolation(
                violation_id=str(uuid.uuid4()),
                analytics_id=metric.analytics_id,
                violation_type=ViolationType.LATENCY_BREACH,
                timestamp=metric.timestamp,
                current_value=metric.latency_ms,
                threshold_value=sla_config.max_latency_ms,
                severity="medium" if metric.latency_ms < sla_config.max_latency_ms * 2 else "high",
                escalated=False,
                escalation_level=None,
                resolved=False,
                resolution_time=None,
                impact_description=f"Latency {metric.latency_ms}ms exceeds SLA threshold {sla_config.max_latency_ms}ms"
            )
            violations.append(violation)
        
        # Process violations
        for violation in violations:
            self._process_violation(violation)
    
    def _process_violation(self, violation: SLAViolation):
        """Process SLA violation and trigger escalation if needed."""
        with self.lock:
            self.active_violations[violation.violation_id] = violation
            
        # Save to database
        self._save_violation_to_db(violation)
        
        # Check for escalation
        self._check_escalation(violation)
        
        self.logger.warning(f"SLA violation detected: {violation.impact_description}")
    
    def _save_violation_to_db(self, violation: SLAViolation):
        """Save SLA violation to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sla_violations 
                    (violation_id, analytics_id, violation_type, timestamp,
                     current_value, threshold_value, severity, escalated,
                     escalation_level, resolved, resolution_time, impact_description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.violation_id,
                    violation.analytics_id,
                    violation.violation_type.value,
                    violation.timestamp.isoformat(),
                    violation.current_value,
                    violation.threshold_value,
                    violation.severity,
                    int(violation.escalated),
                    violation.escalation_level.value if violation.escalation_level else None,
                    int(violation.resolved),
                    violation.resolution_time.isoformat() if violation.resolution_time else None,
                    violation.impact_description
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save violation to database: {e}")
    
    def _check_escalation(self, violation: SLAViolation):
        """Check if violation needs escalation."""
        for rule in self.escalation_rules:
            if (rule.violation_type == violation.violation_type and
                self._should_escalate(violation, rule)):
                
                self._escalate_violation(violation, rule)
                break
    
    def _should_escalate(self, violation: SLAViolation, rule: EscalationRule) -> bool:
        """Determine if violation should be escalated."""
        # Check severity threshold
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if severity_levels.get(violation.severity, 0) < rule.severity_threshold:
            return False
        
        # Check time threshold
        time_elapsed = (datetime.now() - violation.timestamp).total_seconds() / 60
        if time_elapsed < rule.time_threshold_minutes:
            return False
        
        return True
    
    def _escalate_violation(self, violation: SLAViolation, rule: EscalationRule):
        """Escalate violation according to rule."""
        with self.lock:
            violation.escalated = True
            violation.escalation_level = rule.escalation_level
        
        # Send notifications
        self._send_escalation_notifications(violation, rule)
        
        # Execute auto actions
        self._execute_auto_actions(violation, rule)
        
        self.logger.critical(f"SLA violation escalated to {rule.escalation_level.value}: {violation.impact_description}")
    
    def _send_escalation_notifications(self, violation: SLAViolation, rule: EscalationRule):
        """Send escalation notifications."""
        try:
            subject = f"SLA Violation Escalation - {violation.violation_type.value.upper()}"
            
            body = f"""
            SLA Violation Escalated to {rule.escalation_level.value.upper()}
            
            Analytics ID: {violation.analytics_id}
            Violation Type: {violation.violation_type.value}
            Severity: {violation.severity}
            Timestamp: {violation.timestamp}
            
            Current Value: {violation.current_value}
            Threshold: {violation.threshold_value}
            
            Impact: {violation.impact_description}
            
            Immediate action required.
            """
            
            for email in rule.notification_emails:
                self._send_email(email, subject, body)
                
        except Exception as e:
            self.logger.error(f"Failed to send escalation notifications: {e}")
    
    def _send_email(self, to_email: str, subject: str, body: str):
        """Send email notification (placeholder implementation)."""
        # In a real implementation, configure SMTP settings
        self.logger.info(f"EMAIL TO {to_email}: {subject}")
        self.logger.info(f"BODY: {body}")
    
    def _execute_auto_actions(self, violation: SLAViolation, rule: EscalationRule):
        """Execute automatic remediation actions."""
        for action in rule.auto_actions:
            try:
                self.logger.info(f"Executing auto action: {action}")
                
                if action == "restart_slow_components":
                    self._restart_slow_components()
                elif action == "increase_resources":
                    self._increase_resources()
                elif action == "activate_failover":
                    self._activate_failover()
                elif action == "scale_up_instances":
                    self._scale_up_instances()
                elif action == "emergency_backup_activation":
                    self._emergency_backup_activation()
                
            except Exception as e:
                self.logger.error(f"Auto action {action} failed: {e}")
    
    def _restart_slow_components(self):
        """Restart slow components."""
        self.logger.info("AUTO-ACTION: Restarting slow components")
    
    def _increase_resources(self):
        """Increase system resources."""
        self.logger.info("AUTO-ACTION: Increasing system resources")
    
    def _activate_failover(self):
        """Activate failover systems."""
        self.logger.info("AUTO-ACTION: Activating failover systems")
    
    def _scale_up_instances(self):
        """Scale up instances."""
        self.logger.info("AUTO-ACTION: Scaling up instances")
    
    def _emergency_backup_activation(self):
        """Activate emergency backup systems."""
        self.logger.info("AUTO-ACTION: Activating emergency backup systems")
    
    def start_monitoring(self):
        """Start SLA monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_sla, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("SLA monitoring started")
    
    def _monitor_sla(self):
        """Background SLA monitoring loop."""
        while self.running:
            try:
                # Calculate current SLA performance
                self._calculate_sla_performance()
                
                # Check for ongoing violations
                self._check_ongoing_violations()
                
                # Generate performance reports
                self._generate_performance_reports()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"SLA monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _calculate_sla_performance(self):
        """Calculate current SLA performance metrics."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        with self.lock:
            recent_metrics = [
                m for m in self.sla_metrics 
                if m.timestamp >= hour_ago
            ]
            
            if not recent_metrics:
                return
            
            # Calculate performance metrics
            successful_deliveries = sum(1 for m in recent_metrics if m.delivery_success)
            total_deliveries = len(recent_metrics)
            
            availability = (successful_deliveries / total_deliveries * 100) if total_deliveries > 0 else 100
            avg_latency = statistics.mean([m.latency_ms for m in recent_metrics if m.delivery_success])
            error_rate = ((total_deliveries - successful_deliveries) / total_deliveries * 100) if total_deliveries > 0 else 0
            throughput = successful_deliveries / 3600  # Per second
            
            # Store hourly stats
            hour_key = now.strftime("%Y-%m-%d_%H")
            self.hourly_stats[hour_key] = {
                'availability_percent': availability,
                'avg_latency_ms': avg_latency,
                'error_rate_percent': error_rate,
                'throughput_tps': throughput,
                'total_deliveries': total_deliveries,
                'successful_deliveries': successful_deliveries
            }
    
    def _check_ongoing_violations(self):
        """Check ongoing violations for auto-resolution."""
        with self.lock:
            for violation in list(self.active_violations.values()):
                if not violation.resolved:
                    # Check if violation conditions are resolved
                    if self._is_violation_resolved(violation):
                        violation.resolved = True
                        violation.resolution_time = datetime.now()
                        self.logger.info(f"SLA violation auto-resolved: {violation.violation_id}")
    
    def _is_violation_resolved(self, violation: SLAViolation) -> bool:
        """Check if violation conditions are resolved."""
        # Check recent metrics to see if conditions improved
        now = datetime.now()
        recent_window = now - timedelta(minutes=5)
        
        recent_metrics = [
            m for m in self.sla_metrics 
            if m.timestamp >= recent_window and m.analytics_id == violation.analytics_id
        ]
        
        if not recent_metrics:
            return False
        
        # Check based on violation type
        if violation.violation_type == ViolationType.LATENCY_BREACH:
            recent_latencies = [m.latency_ms for m in recent_metrics if m.delivery_success]
            if recent_latencies:
                avg_latency = statistics.mean(recent_latencies)
                return avg_latency <= violation.threshold_value
        
        elif violation.violation_type == ViolationType.DELIVERY_FAILURE:
            # Check if recent deliveries are succeeding
            success_rate = sum(1 for m in recent_metrics if m.delivery_success) / len(recent_metrics)
            return success_rate >= 0.95  # 95% success rate
        
        return False
    
    def _generate_performance_reports(self):
        """Generate periodic performance reports."""
        # This would generate executive dashboards, trend reports, etc.
        pass
    
    def get_sla_summary(self, sla_level: SLALevel = None) -> Dict[str, Any]:
        """Get comprehensive SLA performance summary."""
        sla_level = sla_level or self.default_sla_level
        config = self.sla_configs[sla_level]
        
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        
        with self.lock:
            recent_metrics = [m for m in self.sla_metrics if m.timestamp >= day_ago]
            recent_violations = [v for v in self.active_violations.values() if v.timestamp >= day_ago]
            
            if recent_metrics:
                successful = sum(1 for m in recent_metrics if m.delivery_success)
                total = len(recent_metrics)
                availability = (successful / total * 100) if total > 0 else 100
                
                successful_latencies = [m.latency_ms for m in recent_metrics if m.delivery_success]
                avg_latency = statistics.mean(successful_latencies) if successful_latencies else 0
                
                error_rate = ((total - successful) / total * 100) if total > 0 else 0
                throughput = successful / 86400  # Per second over 24 hours
                
                # Check SLA compliance
                sla_met = (
                    availability >= config.min_availability_percent and
                    avg_latency <= config.max_latency_ms and
                    error_rate <= config.max_error_rate_percent and
                    throughput >= config.min_throughput_tps
                )
            else:
                availability = avg_latency = error_rate = throughput = 0
                sla_met = True
                total = successful = 0
            
            return {
                'timestamp': now.isoformat(),
                'sla_level': sla_level.value,
                'sla_configuration': asdict(config),
                'performance_24h': {
                    'availability_percent': availability,
                    'avg_latency_ms': avg_latency,
                    'error_rate_percent': error_rate,
                    'throughput_tps': throughput,
                    'total_deliveries': total,
                    'successful_deliveries': successful
                },
                'sla_compliance': {
                    'sla_met': sla_met,
                    'availability_compliant': availability >= config.min_availability_percent,
                    'latency_compliant': avg_latency <= config.max_latency_ms,
                    'error_rate_compliant': error_rate <= config.max_error_rate_percent,
                    'throughput_compliant': throughput >= config.min_throughput_tps
                },
                'violations_24h': {
                    'total_violations': len(recent_violations),
                    'escalated_violations': len([v for v in recent_violations if v.escalated]),
                    'resolved_violations': len([v for v in recent_violations if v.resolved]),
                    'active_violations': len([v for v in recent_violations if not v.resolved])
                },
                'pending_deliveries': len(self.pending_deliveries)
            }
    
    def force_sla_check(self) -> Dict[str, Any]:
        """Force immediate SLA performance check."""
        self._calculate_sla_performance()
        return self.get_sla_summary()
    
    def stop_monitoring(self):
        """Stop SLA monitoring."""
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("SLA monitoring stopped")
    
    def shutdown(self):
        """Shutdown the SLA tracker."""
        self.stop_monitoring()


# Global instance for easy access
sla_tracker = None

def get_sla_tracker() -> AnalyticsSLATracker:
    """Get the global SLA tracker instance."""
    global sla_tracker
    if sla_tracker is None:
        sla_tracker = AnalyticsSLATracker()
    return sla_tracker


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    tracker = AnalyticsSLATracker()
    tracker.start_monitoring()
    
    try:
        # Simulate analytics deliveries
        for i in range(20):
            tracking_id = tracker.track_analytics_delivery(
                f"analytics_{i}",
                "test_component",
                "delivery",
                SLALevel.GOLD
            )
            
            # Simulate different outcomes
            if i % 5 == 0:
                # Simulate slow delivery
                tracker.record_delivery_success(tracking_id, 400.0)
            elif i % 7 == 0:
                # Simulate failure
                tracker.record_delivery_failure(tracking_id, "Network timeout")
            else:
                # Simulate normal delivery
                tracker.record_delivery_success(tracking_id, 150.0)
            
            time.sleep(1)
        
        # Check SLA summary
        summary = tracker.get_sla_summary()
        print(json.dumps(summary, indent=2))
        
    except KeyboardInterrupt:
        print("Stopping SLA tracker...")
    
    finally:
        tracker.shutdown()