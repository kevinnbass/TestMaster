"""
Security Monitoring Dashboard Backend

Real-time security metrics collection and monitoring backend.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from threading import Thread, Lock
from collections import deque
import logging

from .vulnerability_scanner import VulnerabilityScanner
from .compliance_checker import ComplianceChecker
from .threat_modeler import ThreatModeler
from .dependency_scanner import DependencyScanner
from .crypto_analyzer import CryptoAnalyzer
from .audit_logger import AuditLogger, EventType, EventSeverity

logger = logging.getLogger(__name__)


@dataclass
class SecurityMetric:
    """Represents a security metric data point."""
    timestamp: str
    metric_name: str
    value: float
    severity: str
    metadata: Dict[str, Any]
    

@dataclass
class SecurityAlert:
    """Represents a security alert."""
    id: str
    timestamp: str
    severity: str
    type: str
    title: str
    description: str
    affected_components: List[str]
    remediation: str
    acknowledged: bool = False
    

class SecurityDashboard:
    """
    Real-time security monitoring dashboard backend.
    Collects metrics, tracks vulnerabilities, and manages alerts.
    """
    
    def __init__(self, scan_interval: int = 300):  # 5 minutes
        """
        Initialize security dashboard.
        
        Args:
            scan_interval: Scan interval in seconds
        """
        self.scan_interval = scan_interval
        self.running = False
        self.metrics_lock = Lock()
        
        # Initialize scanners
        self.vuln_scanner = VulnerabilityScanner()
        self.compliance_checker = ComplianceChecker()
        self.threat_modeler = ThreatModeler()
        self.dep_scanner = DependencyScanner()
        self.crypto_analyzer = CryptoAnalyzer()
        self.audit_logger = AuditLogger("security_audit.log")
        
        # Metrics storage (last 24 hours)
        self.metrics_history = deque(maxlen=288)  # 24h * 60min / 5min
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)
        
        # Security trends
        self.security_trends = {
            'vulnerability_count': [],
            'compliance_score': [],
            'threat_level': [],
            'dependency_risks': []
        }
        
        logger.info(f"Security Dashboard initialized (scan interval: {scan_interval}s)")
        
    def start_monitoring(self, project_path: str) -> None:
        """
        Start real-time security monitoring.
        
        Args:
            project_path: Path to monitor
        """
        if self.running:
            return
            
        self.running = True
        self.project_path = project_path
        
        # Start monitoring thread
        monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        logger.info("Security monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        self.running = False
        logger.info("Security monitoring stopped")
        
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current security status.
        
        Returns:
            Current security status
        """
        with self.metrics_lock:
            latest_metrics = {}
            for metric in reversed(self.metrics_history):
                if metric.metric_name not in latest_metrics:
                    latest_metrics[metric.metric_name] = metric
                    
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.running,
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts if a.severity == 'critical']),
            'latest_metrics': {name: metric.value for name, metric in latest_metrics.items()},
            'trend_summary': self._calculate_trend_summary(),
            'risk_level': self._calculate_overall_risk_level()
        }
        
    def get_metrics_history(self, hours: int = 24) -> List[SecurityMetric]:
        """
        Get metrics history.
        
        Args:
            hours: Hours of history to return
            
        Returns:
            Metrics history
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.metrics_lock:
            return [
                metric for metric in self.metrics_history
                if datetime.fromisoformat(metric.timestamp) > cutoff_time
            ]
            
    def get_active_alerts(self) -> List[SecurityAlert]:
        """Get active security alerts."""
        return [a for a in self.active_alerts if not a.acknowledged]
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a security alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if alert was found and acknowledged
        """
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.audit_logger.log_event(
                    event_type=EventType.SECURITY_SCAN,
                    severity=EventSeverity.INFO,
                    source="security_dashboard",
                    actor="system",
                    resource="alert",
                    action="acknowledge",
                    result="success",
                    details={"alert_id": alert_id}
                )
                return True
        return False
        
    def run_immediate_scan(self, scan_type: str = "full") -> Dict[str, Any]:
        """
        Run immediate security scan.
        
        Args:
            scan_type: Type of scan (vulnerability, compliance, threat, dependency, crypto, full)
            
        Returns:
            Scan results
        """
        scan_start = time.time()
        results = {}
        
        if scan_type in ["vulnerability", "full"]:
            vuln_results = self.vuln_scanner.scan_directory(self.project_path)
            results['vulnerabilities'] = len(sum(vuln_results.values(), []))
            
        if scan_type in ["compliance", "full"]:
            comp_issues = self.compliance_checker.check_owasp_compliance(self.project_path)
            results['compliance_issues'] = len(comp_issues)
            
        if scan_type in ["dependency", "full"]:
            dep_results = self.dep_scanner.scan_python_dependencies(self.project_path)
            results['dependency_vulnerabilities'] = sum(len(d.vulnerabilities) for d in dep_results)
            
        if scan_type in ["crypto", "full"]:
            crypto_files = list(Path(self.project_path).rglob("*.py"))[:10]  # Sample
            crypto_issues = []
            for file_path in crypto_files:
                crypto_issues.extend(self.crypto_analyzer.analyze_file(str(file_path)))
            results['crypto_issues'] = len(crypto_issues)
            
        scan_duration = time.time() - scan_start
        results['scan_duration'] = scan_duration
        results['scan_type'] = scan_type
        results['timestamp'] = datetime.now().isoformat()
        
        return results
        
    def get_security_trends(self, days: int = 7) -> Dict[str, List]:
        """
        Get security trends over time.
        
        Args:
            days: Number of days of trends
            
        Returns:
            Trend data
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        trends = {}
        for trend_name, trend_data in self.security_trends.items():
            filtered_data = [
                point for point in trend_data
                if datetime.fromisoformat(point['timestamp']) > cutoff_time
            ]
            trends[trend_name] = filtered_data
            
        return trends
        
    # Private methods
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_metrics()
                self._check_for_alerts()
                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
                
    def _collect_metrics(self) -> None:
        """Collect security metrics."""
        timestamp = datetime.now().isoformat()
        
        # Collect vulnerability metrics
        vuln_scan = self.vuln_scanner.scan_directory(self.project_path, recursive=False)
        vuln_count = len(sum(vuln_scan.values(), []))
        
        self._add_metric("vulnerability_count", vuln_count, "medium", timestamp)
        
        # Collect compliance metrics
        comp_report = self.compliance_checker.generate_compliance_report(["OWASP"])
        compliance_score = comp_report.compliance_score
        
        self._add_metric("compliance_score", compliance_score, "low", timestamp)
        
        # Update trends
        self.security_trends['vulnerability_count'].append({
            'timestamp': timestamp,
            'value': vuln_count
        })
        self.security_trends['compliance_score'].append({
            'timestamp': timestamp,
            'value': compliance_score
        })
        
        # Trim trends to last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        for trend_data in self.security_trends.values():
            while (trend_data and 
                   datetime.fromisoformat(trend_data[0]['timestamp']) < cutoff):
                trend_data.pop(0)
                
    def _add_metric(self, name: str, value: float, severity: str, timestamp: str) -> None:
        """Add a metric to history."""
        metric = SecurityMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            severity=severity,
            metadata={}
        )
        
        with self.metrics_lock:
            self.metrics_history.append(metric)
            
    def _check_for_alerts(self) -> None:
        """Check for new security alerts."""
        # Check vulnerability threshold
        recent_vulns = [
            m for m in self.metrics_history 
            if m.metric_name == "vulnerability_count" and
            datetime.fromisoformat(m.timestamp) > datetime.now() - timedelta(minutes=10)
        ]
        
        if recent_vulns and recent_vulns[-1].value > 5:
            self._create_alert(
                severity="high",
                alert_type="vulnerability",
                title="High Vulnerability Count",
                description=f"Found {recent_vulns[-1].value} vulnerabilities",
                affected_components=["codebase"],
                remediation="Review and fix identified vulnerabilities"
            )
            
        # Check compliance score
        recent_compliance = [
            m for m in self.metrics_history
            if m.metric_name == "compliance_score" and
            datetime.fromisoformat(m.timestamp) > datetime.now() - timedelta(minutes=10)
        ]
        
        if recent_compliance and recent_compliance[-1].value < 70:
            self._create_alert(
                severity="medium",
                alert_type="compliance",
                title="Low Compliance Score",
                description=f"Compliance score: {recent_compliance[-1].value}%",
                affected_components=["compliance"],
                remediation="Address compliance issues identified"
            )
            
    def _create_alert(self, severity: str, alert_type: str, title: str, 
                     description: str, affected_components: List[str], remediation: str) -> None:
        """Create a new security alert."""
        alert_id = f"{alert_type}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing = any(
            a.type == alert_type and a.title == title and not a.acknowledged
            for a in self.active_alerts
        )
        
        if existing:
            return
            
        alert = SecurityAlert(
            id=alert_id,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            type=alert_type,
            title=title,
            description=description,
            affected_components=affected_components,
            remediation=remediation
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Log the alert
        self.audit_logger.log_event(
            event_type=EventType.SECURITY_SCAN,
            severity=EventSeverity.HIGH if severity == "critical" else EventSeverity.MEDIUM,
            source="security_dashboard",
            actor="system",
            resource="security_monitoring",
            action="alert_created",
            result="success",
            details={"alert_id": alert_id, "alert_type": alert_type}
        )
        
    def _calculate_trend_summary(self) -> Dict[str, str]:
        """Calculate trend summaries."""
        summary = {}
        
        for trend_name, trend_data in self.security_trends.items():
            if len(trend_data) >= 2:
                recent = trend_data[-1]['value']
                previous = trend_data[-2]['value']
                
                if recent > previous:
                    summary[trend_name] = "increasing"
                elif recent < previous:
                    summary[trend_name] = "decreasing"
                else:
                    summary[trend_name] = "stable"
            else:
                summary[trend_name] = "insufficient_data"
                
        return summary
        
    def _calculate_overall_risk_level(self) -> str:
        """Calculate overall risk level."""
        critical_alerts = len([a for a in self.active_alerts if a.severity == "critical"])
        high_alerts = len([a for a in self.active_alerts if a.severity == "high"])
        
        if critical_alerts > 0:
            return "critical"
        elif high_alerts > 2:
            return "high"
        elif high_alerts > 0:
            return "medium"
        else:
            return "low"