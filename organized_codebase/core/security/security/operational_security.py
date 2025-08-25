"""
AgentOps Derived Operational Security Module
Extracted from AgentOps monitoring and operational patterns
Enhanced for comprehensive operational security monitoring and management
"""

import os
import time
import psutil
import hashlib
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread
from .error_handler import SecurityError, security_error_handler


class OperationalSecurityLevel(Enum):
    """Operational security levels"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityMetric:
    """Security metric data structure"""
    name: str
    value: Union[int, float, str]
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_above_warning(self) -> bool:
        """Check if metric is above warning threshold"""
        if self.threshold_warning and isinstance(self.value, (int, float)):
            return self.value >= self.threshold_warning
        return False
    
    @property
    def is_above_critical(self) -> bool:
        """Check if metric is above critical threshold"""
        if self.threshold_critical and isinstance(self.value, (int, float)):
            return self.value >= self.threshold_critical
        return False


@dataclass
class SecurityAlert:
    """Security alert data structure"""
    alert_id: str
    alert_type: str
    message: str
    severity: ThreatLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_component: str = "operational_security"
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def acknowledge(self, user: str = "system"):
        """Acknowledge the alert"""
        self.acknowledged = True
        self.metadata['acknowledged_by'] = user
        self.metadata['acknowledged_at'] = datetime.utcnow().isoformat()
    
    def resolve(self, user: str = "system", resolution: str = ""):
        """Resolve the alert"""
        self.resolved = True
        self.metadata['resolved_by'] = user
        self.metadata['resolved_at'] = datetime.utcnow().isoformat()
        self.metadata['resolution'] = resolution


class SystemMonitor:
    """System resource and security monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring_enabled = True
        
    def get_system_metrics(self) -> Dict[str, SecurityMetric]:
        """Get current system security metrics"""
        try:
            metrics = {}
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['cpu_usage'] = SecurityMetric(
                name='cpu_usage',
                value=cpu_percent,
                unit='percent',
                threshold_warning=80.0,
                threshold_critical=95.0
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics['memory_usage'] = SecurityMetric(
                name='memory_usage',
                value=memory.percent,
                unit='percent',
                threshold_warning=80.0,
                threshold_critical=95.0,
                metadata={'available_gb': round(memory.available / (1024**3), 2)}
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics['disk_usage'] = SecurityMetric(
                name='disk_usage',
                value=disk_percent,
                unit='percent',
                threshold_warning=85.0,
                threshold_critical=95.0,
                metadata={'free_gb': round(disk.free / (1024**3), 2)}
            )
            
            # Network connections
            connections = psutil.net_connections()
            established_count = len([c for c in connections if c.status == 'ESTABLISHED'])
            metrics['network_connections'] = SecurityMetric(
                name='network_connections',
                value=established_count,
                unit='count',
                threshold_warning=100,
                threshold_critical=500,
                metadata={'total_connections': len(connections)}
            )
            
            # Process count
            process_count = len(psutil.pids())
            metrics['process_count'] = SecurityMetric(
                name='process_count',
                value=process_count,
                unit='count',
                threshold_warning=300,
                threshold_critical=500
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_security_processes(self) -> List[Dict[str, Any]]:
        """Get information about running security-relevant processes"""
        try:
            security_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    
                    # Check for security-relevant processes
                    security_keywords = [
                        'antivirus', 'firewall', 'security', 'defender', 'scanner',
                        'monitor', 'agent', 'guardian', 'protect', 'vault'
                    ]
                    
                    if any(keyword in proc_name for keyword in security_keywords):
                        security_processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cpu_percent': proc_info['cpu_percent'],
                            'memory_percent': proc_info['memory_percent']
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return security_processes
            
        except Exception as e:
            self.logger.error(f"Error getting security processes: {e}")
            return []


class ThreatDetector:
    """Threat detection and analysis engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Suspicious patterns to monitor
        self.suspicious_patterns = [
            r'\.exe\s+.*\s+--hidden',
            r'powershell.*-encodedcommand',
            r'cmd.*\/c.*del.*\*',
            r'wget.*http.*\/tmp\/',
            r'curl.*-o.*\/tmp\/',
            r'python.*-c.*import.*os',
        ]
        
        # Network threat indicators
        self.threat_ips = set()  # Would be populated from threat feeds
        self.suspicious_ports = {22, 23, 135, 139, 445, 1433, 3389, 5432}
    
    def analyze_network_activity(self) -> List[SecurityAlert]:
        """Analyze network activity for threats"""
        alerts = []
        
        try:
            connections = psutil.net_connections()
            
            for conn in connections:
                if conn.raddr:
                    remote_ip = conn.raddr.ip
                    remote_port = conn.raddr.port
                    
                    # Check against threat IPs
                    if remote_ip in self.threat_ips:
                        alert = SecurityAlert(
                            alert_id=hashlib.sha256(f"threat_ip_{remote_ip}_{time.time()}".encode()).hexdigest()[:16],
                            alert_type="threat_ip_connection",
                            message=f"Connection to known threat IP: {remote_ip}:{remote_port}",
                            severity=ThreatLevel.HIGH,
                            metadata={
                                'remote_ip': remote_ip,
                                'remote_port': remote_port,
                                'local_port': conn.laddr.port if conn.laddr else None
                            }
                        )
                        alerts.append(alert)
                    
                    # Check for suspicious port usage
                    if remote_port in self.suspicious_ports:
                        alert = SecurityAlert(
                            alert_id=hashlib.sha256(f"suspicious_port_{remote_port}_{time.time()}".encode()).hexdigest()[:16],
                            alert_type="suspicious_port_connection",
                            message=f"Connection to suspicious port: {remote_ip}:{remote_port}",
                            severity=ThreatLevel.MEDIUM,
                            metadata={
                                'remote_ip': remote_ip,
                                'remote_port': remote_port,
                                'connection_status': conn.status
                            }
                        )
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error analyzing network activity: {e}")
            return []
    
    def analyze_process_activity(self) -> List[SecurityAlert]:
        """Analyze process activity for threats"""
        alerts = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    proc_info = proc.info
                    
                    # Check command line for suspicious patterns
                    cmdline = ' '.join(proc_info.get('cmdline', []))
                    
                    import re
                    for pattern in self.suspicious_patterns:
                        if re.search(pattern, cmdline, re.IGNORECASE):
                            alert = SecurityAlert(
                                alert_id=hashlib.sha256(f"suspicious_process_{proc_info['pid']}_{time.time()}".encode()).hexdigest()[:16],
                                alert_type="suspicious_process_activity",
                                message=f"Suspicious process activity detected: {proc_info['name']}",
                                severity=ThreatLevel.HIGH,
                                metadata={
                                    'pid': proc_info['pid'],
                                    'process_name': proc_info['name'],
                                    'command_line': cmdline,
                                    'pattern_matched': pattern
                                }
                            )
                            alerts.append(alert)
                            break
                    
                    # Check for recently created processes
                    if proc_info.get('create_time'):
                        age_seconds = time.time() - proc_info['create_time']
                        if age_seconds < 60:  # Process created in last minute
                            process_name = proc_info['name'].lower()
                            suspicious_names = ['powershell', 'cmd', 'bash', 'sh', 'wget', 'curl']
                            
                            if any(name in process_name for name in suspicious_names):
                                alert = SecurityAlert(
                                    alert_id=hashlib.sha256(f"new_process_{proc_info['pid']}_{time.time()}".encode()).hexdigest()[:16],
                                    alert_type="recently_created_process",
                                    message=f"Recently created potentially suspicious process: {proc_info['name']}",
                                    severity=ThreatLevel.MEDIUM,
                                    metadata={
                                        'pid': proc_info['pid'],
                                        'process_name': proc_info['name'],
                                        'age_seconds': age_seconds
                                    }
                                )
                                alerts.append(alert)
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error analyzing process activity: {e}")
            return []


class OperationalSecurityManager:
    """Central operational security management system"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.threat_detector = ThreatDetector()
        
        self.security_level = OperationalSecurityLevel.NORMAL
        self.metrics_history: List[SecurityMetric] = []
        self.active_alerts: List[SecurityAlert] = []
        self.resolved_alerts: List[SecurityAlert] = []
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
        self.metrics_lock = Lock()
        
        self.logger = logging.getLogger(__name__)
        
        # Security thresholds
        self.alert_thresholds = {
            'max_unresolved_alerts': 10,
            'critical_alerts_threshold': 3,
            'metric_violation_threshold': 5
        }
        
        # Alert handlers
        self.alert_handlers: List[Callable[[SecurityAlert], None]] = []
    
    def start_monitoring(self):
        """Start continuous operational security monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Operational security monitoring started")
    
    def stop_monitoring(self):
        """Stop operational security monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Operational security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self.system_monitor.get_system_metrics()
                
                with self.metrics_lock:
                    self.metrics_history.extend(metrics.values())
                    
                    # Keep history manageable
                    if len(self.metrics_history) > 10000:
                        self.metrics_history = self.metrics_history[-5000:]
                
                # Check for metric violations
                self._check_metric_violations(metrics)
                
                # Run threat detection
                network_alerts = self.threat_detector.analyze_network_activity()
                process_alerts = self.threat_detector.analyze_process_activity()
                
                # Process new alerts
                new_alerts = network_alerts + process_alerts
                for alert in new_alerts:
                    self._process_alert(alert)
                
                # Update security level
                self._update_security_level()
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_metric_violations(self, metrics: Dict[str, SecurityMetric]):
        """Check metrics for security violations"""
        try:
            for metric in metrics.values():
                if metric.is_above_critical:
                    alert = SecurityAlert(
                        alert_id=hashlib.sha256(f"metric_critical_{metric.name}_{time.time()}".encode()).hexdigest()[:16],
                        alert_type="metric_critical_violation",
                        message=f"Critical threshold exceeded for {metric.name}: {metric.value}{metric.unit}",
                        severity=ThreatLevel.CRITICAL,
                        metadata={
                            'metric_name': metric.name,
                            'metric_value': metric.value,
                            'threshold_critical': metric.threshold_critical,
                            'metadata': metric.metadata
                        }
                    )
                    self._process_alert(alert)
                
                elif metric.is_above_warning:
                    alert = SecurityAlert(
                        alert_id=hashlib.sha256(f"metric_warning_{metric.name}_{time.time()}".encode()).hexdigest()[:16],
                        alert_type="metric_warning_violation", 
                        message=f"Warning threshold exceeded for {metric.name}: {metric.value}{metric.unit}",
                        severity=ThreatLevel.MEDIUM,
                        metadata={
                            'metric_name': metric.name,
                            'metric_value': metric.value,
                            'threshold_warning': metric.threshold_warning,
                            'metadata': metric.metadata
                        }
                    )
                    self._process_alert(alert)
                    
        except Exception as e:
            self.logger.error(f"Error checking metric violations: {e}")
    
    def _process_alert(self, alert: SecurityAlert):
        """Process a new security alert"""
        try:
            # Check if we already have this alert (avoid duplicates)
            existing_alert = next(
                (a for a in self.active_alerts 
                 if a.alert_type == alert.alert_type and not a.resolved),
                None
            )
            
            if existing_alert:
                # Update existing alert metadata
                existing_alert.metadata['last_seen'] = datetime.utcnow().isoformat()
                existing_alert.metadata['occurrence_count'] = existing_alert.metadata.get('occurrence_count', 1) + 1
            else:
                # New alert
                self.active_alerts.append(alert)
                
                # Trigger alert handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        self.logger.error(f"Alert handler error: {e}")
                
                # Log based on severity
                if alert.severity == ThreatLevel.CRITICAL:
                    self.logger.critical(f"CRITICAL ALERT: {alert.message}")
                elif alert.severity == ThreatLevel.HIGH:
                    self.logger.error(f"HIGH ALERT: {alert.message}")
                elif alert.severity == ThreatLevel.MEDIUM:
                    self.logger.warning(f"MEDIUM ALERT: {alert.message}")
                else:
                    self.logger.info(f"LOW ALERT: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Error processing alert: {e}")
    
    def _update_security_level(self):
        """Update operational security level based on current conditions"""
        try:
            active_critical = len([a for a in self.active_alerts 
                                 if not a.resolved and a.severity == ThreatLevel.CRITICAL])
            active_high = len([a for a in self.active_alerts
                              if not a.resolved and a.severity == ThreatLevel.HIGH])
            
            if active_critical >= 3:
                new_level = OperationalSecurityLevel.CRITICAL
            elif active_critical >= 1 or active_high >= 5:
                new_level = OperationalSecurityLevel.HIGH
            elif active_high >= 2:
                new_level = OperationalSecurityLevel.ELEVATED
            else:
                new_level = OperationalSecurityLevel.NORMAL
            
            if new_level != self.security_level:
                old_level = self.security_level
                self.security_level = new_level
                
                self.logger.warning(f"Security level changed: {old_level.value} -> {new_level.value}")
                
                # Create security level change alert
                level_alert = SecurityAlert(
                    alert_id=hashlib.sha256(f"security_level_change_{time.time()}".encode()).hexdigest()[:16],
                    alert_type="security_level_change",
                    message=f"Operational security level changed to {new_level.value}",
                    severity=ThreatLevel.HIGH if new_level.value in ['high', 'critical'] else ThreatLevel.MEDIUM,
                    metadata={
                        'old_level': old_level.value,
                        'new_level': new_level.value,
                        'active_critical_alerts': active_critical,
                        'active_high_alerts': active_high
                    }
                )
                self._process_alert(level_alert)
                
        except Exception as e:
            self.logger.error(f"Error updating security level: {e}")
    
    def add_alert_handler(self, handler: Callable[[SecurityAlert], None]):
        """Add alert handler for notifications"""
        self.alert_handlers.append(handler)
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge a security alert"""
        try:
            alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
            if alert:
                alert.acknowledge(user)
                self.logger.info(f"Alert acknowledged: {alert_id} by {user}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, user: str = "system", resolution: str = "") -> bool:
        """Resolve a security alert"""
        try:
            alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
            if alert:
                alert.resolve(user, resolution)
                self.resolved_alerts.append(alert)
                self.active_alerts.remove(alert)
                
                self.logger.info(f"Alert resolved: {alert_id} by {user}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False
    
    def get_operational_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive operational security summary"""
        try:
            # Current metrics
            current_metrics = self.system_monitor.get_system_metrics()
            
            # Alert statistics
            alert_counts = {
                'active_alerts': len(self.active_alerts),
                'resolved_alerts': len(self.resolved_alerts),
                'unacknowledged_alerts': len([a for a in self.active_alerts if not a.acknowledged]),
                'critical_alerts': len([a for a in self.active_alerts if a.severity == ThreatLevel.CRITICAL]),
                'high_alerts': len([a for a in self.active_alerts if a.severity == ThreatLevel.HIGH])
            }
            
            # Security processes
            security_processes = self.system_monitor.get_security_processes()
            
            # Recent metrics (last hour)
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            with self.metrics_lock:
                recent_metrics = [m for m in self.metrics_history if m.timestamp > hour_ago]
            
            return {
                'operational_security_level': self.security_level.value,
                'monitoring_active': self.monitoring_active,
                'current_metrics': {name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'is_warning': metric.is_above_warning,
                    'is_critical': metric.is_above_critical
                } for name, metric in current_metrics.items()},
                'alert_statistics': alert_counts,
                'security_processes_count': len(security_processes),
                'metrics_collected_1h': len(recent_metrics),
                'total_metrics_history': len(self.metrics_history),
                'alert_handlers_registered': len(self.alert_handlers)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating operational security summary: {e}")
            return {'error': str(e)}


# Global operational security manager
operational_security_manager = OperationalSecurityManager()


def start_operational_monitoring():
    """Convenience function to start operational monitoring"""
    operational_security_manager.start_monitoring()


def stop_operational_monitoring():
    """Convenience function to stop operational monitoring"""
    operational_security_manager.stop_monitoring()


def get_security_status() -> Dict[str, Any]:
    """Convenience function to get security status"""
    return operational_security_manager.get_operational_security_summary()


def add_security_alert_handler(handler: Callable[[SecurityAlert], None]):
    """Convenience function to add alert handler"""
    operational_security_manager.add_alert_handler(handler)