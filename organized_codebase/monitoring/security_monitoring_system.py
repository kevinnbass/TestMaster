"""
Archive Derived Security Monitoring System Module
Extracted from TestMaster archive watchdog systems for comprehensive security monitoring
Enhanced for threat detection, anomaly monitoring, and automated incident response
"""

import uuid
import time
import json
import logging
import threading
import subprocess
import psutil
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from .error_handler import SecurityError, security_error_handler


class MonitoringLevel(Enum):
    """Security monitoring levels"""
    PASSIVE = "passive"
    ACTIVE = "active"
    AGGRESSIVE = "aggressive"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentType(Enum):
    """Security incident types"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_ACTIVITY = "malicious_activity"
    DATA_BREACH = "data_breach"
    SERVICE_DISRUPTION = "service_disruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    CONFIGURATION_CHANGE = "configuration_change"
    FAILED_AUTHENTICATION = "failed_authentication"


class ResponseAction(Enum):
    """Automated response actions"""
    ALERT_ONLY = "alert_only"
    LOG_INCIDENT = "log_incident"
    BLOCK_IP = "block_ip"
    DISABLE_ACCOUNT = "disable_account"
    QUARANTINE_SYSTEM = "quarantine_system"
    RESTART_SERVICE = "restart_service"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    ESCALATE_TO_ADMIN = "escalate_to_admin"


class MonitoringStatus(Enum):
    """Monitoring component status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"


@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    incident_type: IncidentType = IncidentType.ANOMALOUS_BEHAVIOR
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    source_ip: Optional[str] = None
    target_system: str = ""
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    status: str = "open"
    response_actions: List[ResponseAction] = field(default_factory=list)
    assignee: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary"""
        return {
            'incident_id': self.incident_id,
            'incident_type': self.incident_type.value,
            'threat_level': self.threat_level.value,
            'source_ip': self.source_ip,
            'target_system': self.target_system,
            'description': self.description,
            'evidence': self.evidence,
            'detected_at': self.detected_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'status': self.status,
            'response_actions': [action.value for action in self.response_actions],
            'assignee': self.assignee,
            'metadata': self.metadata
        }


@dataclass
class MonitoringRule:
    """Security monitoring rule definition"""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_name: str = ""
    description: str = ""
    monitor_function: Callable[[], Tuple[bool, Dict[str, Any]]] = None
    check_interval: int = 60  # seconds
    failure_threshold: int = 3
    monitoring_level: MonitoringLevel = MonitoringLevel.ACTIVE
    enabled: bool = True
    response_actions: List[ResponseAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.rule_name:
            self.rule_name = f"rule_{self.rule_id[:8]}"


@dataclass
class SystemMetrics:
    """System security metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_connections: int = 0
    active_processes: int = 0
    failed_logins: int = 0
    security_events: int = 0
    threat_detections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_connections': self.network_connections,
            'active_processes': self.active_processes,
            'failed_logins': self.failed_logins,
            'security_events': self.security_events,
            'threat_detections': self.threat_detections
        }


class AnomalyDetector:
    """ML-based anomaly detection for security monitoring"""
    
    def __init__(self, lookback_minutes: int = 60):
        self.lookback_minutes = lookback_minutes
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.baselines: Dict[str, float] = {}
        self.thresholds: Dict[str, Tuple[float, float]] = {}  # (lower, upper)
        self.logger = logging.getLogger(__name__)
        
        # Initialize default thresholds
        self.thresholds = {
            'cpu_usage': (0.0, 85.0),
            'memory_usage': (0.0, 90.0),
            'disk_usage': (0.0, 95.0),
            'network_connections': (0, 1000),
            'active_processes': (0, 500),
            'failed_logins': (0, 10),
            'security_events': (0, 50),
            'threat_detections': (0, 5)
        }
    
    def add_metrics(self, metrics: SystemMetrics):
        """Add metrics to history for analysis"""
        self.metrics_history.append(metrics)
        self._update_baselines()
    
    def detect_anomalies(self, metrics: SystemMetrics) -> List[Tuple[str, float, str]]:
        """Detect anomalies in system metrics"""
        anomalies = []
        
        try:
            metrics_dict = metrics.to_dict()
            
            for metric_name, value in metrics_dict.items():
                if metric_name == 'timestamp':
                    continue
                
                if not isinstance(value, (int, float)):
                    continue
                
                # Check against thresholds
                if metric_name in self.thresholds:
                    lower, upper = self.thresholds[metric_name]
                    
                    if value < lower:
                        anomalies.append((metric_name, value, f"Below threshold: {lower}"))
                    elif value > upper:
                        anomalies.append((metric_name, value, f"Above threshold: {upper}"))
                
                # Check against baseline (if available)
                if metric_name in self.baselines and len(self.metrics_history) > 10:
                    baseline = self.baselines[metric_name]
                    deviation = abs(value - baseline) / baseline if baseline > 0 else 0
                    
                    if deviation > 0.5:  # 50% deviation from baseline
                        anomalies.append((
                            metric_name, value, 
                            f"Significant deviation from baseline: {baseline:.2f}"
                        ))
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _update_baselines(self):
        """Update metric baselines based on historical data"""
        if len(self.metrics_history) < 10:
            return
        
        # Calculate moving averages for baselines
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        
        try:
            metric_sums = defaultdict(float)
            metric_counts = defaultdict(int)
            
            for metrics in recent_metrics:
                metrics_dict = metrics.to_dict()
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)) and key != 'timestamp':
                        metric_sums[key] += value
                        metric_counts[key] += 1
            
            # Update baselines
            for metric_name, total in metric_sums.items():
                if metric_counts[metric_name] > 0:
                    self.baselines[metric_name] = total / metric_counts[metric_name]
                    
        except Exception as e:
            self.logger.error(f"Baseline update failed: {e}")


class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.suspicious_patterns = {
            'sql_injection': [
                r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
                r"(\bdrop\b\s+table)|(\bdelete\b\s+from)",
                r"1\s*=\s*1|1\s*'\s*=\s*'1"
            ],
            'xss_attempt': [
                r"<script[^>]*>.*?</script>",
                r"javascript\s*:",
                r"on\w+\s*=\s*[\"'][^\"']*[\"']"
            ],
            'brute_force': [
                r"multiple\s+failed\s+login",
                r"password\s+attempt",
                r"authentication\s+failure"
            ],
            'privilege_escalation': [
                r"sudo\s+su|su\s+-",
                r"chmod\s+777",
                r"setuid\s+root"
            ]
        }
        
        self.ip_tracking: Dict[str, List[datetime]] = defaultdict(list)
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def analyze_log_entry(self, log_entry: str, source_ip: Optional[str] = None) -> List[Tuple[str, str]]:
        """Analyze log entry for threats"""
        threats = []
        
        try:
            import re
            
            # Check for known threat patterns
            for threat_type, patterns in self.suspicious_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, log_entry, re.IGNORECASE):
                        threats.append((threat_type, pattern))
            
            # Track IP behavior
            if source_ip:
                self.ip_tracking[source_ip].append(datetime.utcnow())
                
                # Clean old entries (keep last hour)
                cutoff = datetime.utcnow() - timedelta(hours=1)
                self.ip_tracking[source_ip] = [
                    timestamp for timestamp in self.ip_tracking[source_ip]
                    if timestamp > cutoff
                ]
                
                # Check for suspicious frequency
                if len(self.ip_tracking[source_ip]) > 100:  # More than 100 requests per hour
                    threats.append(('high_frequency_access', f'IP {source_ip} excessive requests'))
            
            return threats
            
        except Exception as e:
            self.logger.error(f"Threat analysis failed: {e}")
            return []
    
    def analyze_network_traffic(self, traffic_data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Analyze network traffic for threats"""
        threats = []
        
        try:
            # Check for port scanning
            if 'connection_attempts' in traffic_data:
                attempts = traffic_data['connection_attempts']
                if attempts > 50:  # More than 50 connection attempts
                    threats.append(('port_scanning', f'{attempts} connection attempts detected'))
            
            # Check for DDoS patterns
            if 'requests_per_second' in traffic_data:
                rps = traffic_data['requests_per_second']
                if rps > 1000:  # More than 1000 requests per second
                    threats.append(('ddos_attempt', f'{rps} requests per second detected'))
            
            # Check for data exfiltration
            if 'outbound_data_mb' in traffic_data:
                data_mb = traffic_data['outbound_data_mb']
                if data_mb > 1000:  # More than 1GB outbound
                    threats.append(('data_exfiltration', f'{data_mb}MB outbound data detected'))
            
            return threats
            
        except Exception as e:
            self.logger.error(f"Network traffic analysis failed: {e}")
            return []


class SecurityMonitoringSystem:
    """Comprehensive security monitoring and incident response system"""
    
    def __init__(self):
        # Core components
        self.anomaly_detector = AnomalyDetector()
        self.threat_detector = ThreatDetector()
        
        # Data storage
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.metrics_history: deque = deque(maxlen=10000)
        
        # Monitoring state
        self.monitoring_active = True
        self.monitoring_status = MonitoringStatus.HEALTHY
        
        # Statistics
        self.stats = {
            'total_incidents': 0,
            'critical_incidents': 0,
            'resolved_incidents': 0,
            'false_positives': 0,
            'response_actions_taken': 0,
            'uptime_percentage': 100.0,
            'mean_time_to_detection': 0.0,
            'mean_time_to_response': 0.0
        }
        
        # Configuration
        self.metrics_collection_interval = 30  # seconds
        self.incident_retention_days = 30
        self.auto_response_enabled = True
        self.escalation_threshold = ThreatLevel.HIGH
        
        # Background threads
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        
        # Thread safety
        self.monitor_lock = threading.RLock()
        
        # Event handlers
        self.incident_handlers: Dict[IncidentType, List[Callable]] = defaultdict(list)
        
        # Start background processing
        self.monitor_thread.start()
        self.metrics_thread.start()
        self.cleanup_thread.start()
        
        # Initialize default monitoring rules
        self._setup_default_rules()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Security Monitoring System initialized")
    
    def add_monitoring_rule(self, rule: MonitoringRule) -> bool:
        """Add new monitoring rule"""
        try:
            with self.monitor_lock:
                self.monitoring_rules[rule.rule_id] = rule
                self.logger.info(f"Added monitoring rule: {rule.rule_name}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Failed to add monitoring rule: {str(e)}", "MON_RULE_001")
            security_error_handler.handle_error(error)
            return False
    
    def create_incident(self, incident_type: IncidentType, threat_level: ThreatLevel,
                       description: str, evidence: Dict[str, Any] = None,
                       source_ip: str = None, target_system: str = "") -> str:
        """Create new security incident"""
        try:
            with self.monitor_lock:
                incident = SecurityIncident(
                    incident_type=incident_type,
                    threat_level=threat_level,
                    description=description,
                    evidence=evidence or {},
                    source_ip=source_ip,
                    target_system=target_system
                )
                
                self.active_incidents[incident.incident_id] = incident
                self.stats['total_incidents'] += 1
                
                if threat_level == ThreatLevel.CRITICAL:
                    self.stats['critical_incidents'] += 1
                
                # Execute automated response
                if self.auto_response_enabled:
                    self._execute_automated_response(incident)
                
                # Trigger incident handlers
                self._trigger_incident_handlers(incident)
                
                self.logger.warning(f"Security incident created: {incident.incident_id} - {description}")
                return incident.incident_id
                
        except Exception as e:
            error = SecurityError(f"Failed to create incident: {str(e)}", "INCIDENT_001")
            security_error_handler.handle_error(error)
            return ""
    
    def resolve_incident(self, incident_id: str, resolution_notes: str = "") -> bool:
        """Resolve security incident"""
        try:
            with self.monitor_lock:
                if incident_id not in self.active_incidents:
                    return False
                
                incident = self.active_incidents[incident_id]
                incident.resolved_at = datetime.utcnow()
                incident.status = "resolved"
                incident.metadata['resolution_notes'] = resolution_notes
                
                self.stats['resolved_incidents'] += 1
                
                # Calculate resolution time
                resolution_time = (incident.resolved_at - incident.detected_at).total_seconds()
                self.stats['mean_time_to_response'] = (
                    (self.stats['mean_time_to_response'] * (self.stats['resolved_incidents'] - 1) + 
                     resolution_time) / self.stats['resolved_incidents']
                )
                
                self.logger.info(f"Security incident resolved: {incident_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Failed to resolve incident: {str(e)}", "RESOLVE_001")
            security_error_handler.handle_error(error)
            return False
    
    def register_incident_handler(self, incident_type: IncidentType, handler: Callable):
        """Register handler for specific incident type"""
        self.incident_handlers[incident_type].append(handler)
        self.logger.info(f"Handler registered for incident type: {incident_type.value}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            metrics = SystemMetrics(
                cpu_usage=psutil.cpu_percent(interval=1),
                memory_usage=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage('/').percent,
                network_connections=len(psutil.net_connections()),
                active_processes=len(psutil.pids()),
                # These would be collected from logs/security systems
                failed_logins=self._get_failed_login_count(),
                security_events=self._get_security_event_count(),
                threat_detections=self._get_threat_detection_count()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()  # Return empty metrics
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        with self.monitor_lock:
            return {
                **self.stats,
                'active_incidents': len(self.active_incidents),
                'monitoring_rules': len(self.monitoring_rules),
                'enabled_rules': sum(1 for rule in self.monitoring_rules.values() if rule.enabled),
                'system_status': self.monitoring_status.value,
                'recent_anomalies': len([
                    incident for incident in self.active_incidents.values()
                    if incident.incident_type == IncidentType.ANOMALOUS_BEHAVIOR and
                    (datetime.utcnow() - incident.detected_at).total_seconds() < 3600
                ])
            }
    
    def _setup_default_rules(self):
        """Setup default monitoring rules"""
        # CPU usage monitoring
        cpu_rule = MonitoringRule(
            rule_name="High CPU Usage",
            description="Monitor for high CPU usage",
            monitor_function=lambda: self._check_cpu_usage(),
            check_interval=60,
            failure_threshold=3,
            response_actions=[ResponseAction.ALERT_ONLY]
        )
        self.add_monitoring_rule(cpu_rule)
        
        # Memory usage monitoring
        memory_rule = MonitoringRule(
            rule_name="High Memory Usage",
            description="Monitor for high memory usage",
            monitor_function=lambda: self._check_memory_usage(),
            check_interval=60,
            failure_threshold=3,
            response_actions=[ResponseAction.ALERT_ONLY]
        )
        self.add_monitoring_rule(memory_rule)
        
        # Failed login monitoring
        login_rule = MonitoringRule(
            rule_name="Failed Login Attempts",
            description="Monitor for excessive failed login attempts",
            monitor_function=lambda: self._check_failed_logins(),
            check_interval=300,  # Check every 5 minutes
            failure_threshold=1,
            response_actions=[ResponseAction.LOG_INCIDENT, ResponseAction.ALERT_ONLY]
        )
        self.add_monitoring_rule(login_rule)
    
    def _check_cpu_usage(self) -> Tuple[bool, Dict[str, Any]]:
        """Check CPU usage levels"""
        cpu_percent = psutil.cpu_percent(interval=1)
        is_healthy = cpu_percent < 85.0
        return is_healthy, {'cpu_usage': cpu_percent, 'threshold': 85.0}
    
    def _check_memory_usage(self) -> Tuple[bool, Dict[str, Any]]:
        """Check memory usage levels"""
        memory = psutil.virtual_memory()
        is_healthy = memory.percent < 90.0
        return is_healthy, {
            'memory_usage': memory.percent,
            'available_gb': memory.available / (1024**3),
            'threshold': 90.0
        }
    
    def _check_failed_logins(self) -> Tuple[bool, Dict[str, Any]]:
        """Check for excessive failed login attempts"""
        failed_count = self._get_failed_login_count()
        is_healthy = failed_count < 10
        return is_healthy, {'failed_logins': failed_count, 'threshold': 10}
    
    def _get_failed_login_count(self) -> int:
        """Get recent failed login count (placeholder implementation)"""
        # In a real implementation, this would query security logs
        return 0
    
    def _get_security_event_count(self) -> int:
        """Get recent security event count (placeholder implementation)"""
        # In a real implementation, this would query security event logs
        return 0
    
    def _get_threat_detection_count(self) -> int:
        """Get recent threat detection count (placeholder implementation)"""
        # In a real implementation, this would query threat detection systems
        return 0
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                with self.monitor_lock:
                    # Execute monitoring rules
                    for rule_id, rule in self.monitoring_rules.items():
                        if not rule.enabled:
                            continue
                        
                        try:
                            is_healthy, evidence = rule.monitor_function()
                            
                            if not is_healthy:
                                # Create incident for failed check
                                self.create_incident(
                                    incident_type=IncidentType.ANOMALOUS_BEHAVIOR,
                                    threat_level=ThreatLevel.MEDIUM,
                                    description=f"Monitoring rule failed: {rule.rule_name}",
                                    evidence=evidence,
                                    target_system=rule.rule_name
                                )
                                
                        except Exception as e:
                            self.logger.error(f"Monitoring rule execution failed: {rule.rule_name} - {e}")
                
                time.sleep(60)  # Check rules every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self.get_system_metrics()
                
                # Add to history
                self.metrics_history.append(metrics)
                self.anomaly_detector.add_metrics(metrics)
                
                # Check for anomalies
                anomalies = self.anomaly_detector.detect_anomalies(metrics)
                
                for metric_name, value, description in anomalies:
                    self.create_incident(
                        incident_type=IncidentType.ANOMALOUS_BEHAVIOR,
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Anomaly detected in {metric_name}: {description}",
                        evidence={'metric': metric_name, 'value': value, 'description': description}
                    )
                
                time.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(30)
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.monitoring_active:
            try:
                time.sleep(3600)  # Run every hour
                
                with self.monitor_lock:
                    # Clean up old resolved incidents
                    cutoff = datetime.utcnow() - timedelta(days=self.incident_retention_days)
                    
                    expired_incidents = []
                    for incident_id, incident in self.active_incidents.items():
                        if (incident.status == "resolved" and
                            incident.resolved_at and
                            incident.resolved_at < cutoff):
                            expired_incidents.append(incident_id)
                    
                    for incident_id in expired_incidents:
                        del self.active_incidents[incident_id]
                    
                    if expired_incidents:
                        self.logger.info(f"Cleaned up {len(expired_incidents)} old incidents")
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def _execute_automated_response(self, incident: SecurityIncident):
        """Execute automated response actions for incident"""
        try:
            # Get appropriate rule's response actions
            response_actions = []
            
            # Default responses based on threat level
            if incident.threat_level == ThreatLevel.CRITICAL:
                response_actions = [ResponseAction.ESCALATE_TO_ADMIN, ResponseAction.LOG_INCIDENT]
            elif incident.threat_level == ThreatLevel.HIGH:
                response_actions = [ResponseAction.LOG_INCIDENT, ResponseAction.ALERT_ONLY]
            else:
                response_actions = [ResponseAction.LOG_INCIDENT]
            
            for action in response_actions:
                self._execute_response_action(action, incident)
                self.stats['response_actions_taken'] += 1
            
            incident.response_actions = response_actions
            
        except Exception as e:
            self.logger.error(f"Automated response execution failed: {e}")
    
    def _execute_response_action(self, action: ResponseAction, incident: SecurityIncident):
        """Execute individual response action"""
        try:
            if action == ResponseAction.ALERT_ONLY:
                self.logger.warning(f"SECURITY ALERT: {incident.description}")
            
            elif action == ResponseAction.LOG_INCIDENT:
                self.logger.info(f"SECURITY INCIDENT LOGGED: {incident.to_dict()}")
            
            elif action == ResponseAction.BLOCK_IP:
                if incident.source_ip:
                    self.logger.warning(f"BLOCKING IP: {incident.source_ip}")
                    # In production, integrate with firewall/IPS
            
            elif action == ResponseAction.ESCALATE_TO_ADMIN:
                self.logger.critical(f"ESCALATING TO ADMIN: {incident.description}")
                # In production, send email/SMS to administrators
            
            # Add more response actions as needed
            
        except Exception as e:
            self.logger.error(f"Response action execution failed: {action.value} - {e}")
    
    def _trigger_incident_handlers(self, incident: SecurityIncident):
        """Trigger registered incident handlers"""
        try:
            handlers = self.incident_handlers.get(incident.incident_type, [])
            
            for handler in handlers:
                try:
                    handler(incident)
                except Exception as e:
                    self.logger.error(f"Incident handler failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Handler triggering failed: {e}")
    
    def shutdown(self):
        """Shutdown monitoring system"""
        self.monitoring_active = False
        self.logger.info("Security Monitoring System shutdown")


# Global security monitoring system
security_monitoring_system = SecurityMonitoringSystem()


def create_security_incident(incident_type: IncidentType, threat_level: ThreatLevel,
                           description: str, evidence: Dict[str, Any] = None) -> str:
    """Convenience function to create security incident"""
    return security_monitoring_system.create_incident(
        incident_type, threat_level, description, evidence
    )


def register_incident_handler(incident_type: IncidentType, handler: Callable):
    """Convenience function to register incident handler"""
    security_monitoring_system.register_incident_handler(incident_type, handler)