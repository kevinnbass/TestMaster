"""
Quality Monitor for TestMaster Agent QA

Continuous monitoring of agent quality and performance.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

from core.feature_flags import FeatureFlags

class AlertType(Enum):
    """Types of quality alerts."""
    QUALITY_DEGRADATION = "quality_degradation"
    PERFORMANCE_ISSUE = "performance_issue"
    ERROR_SPIKE = "error_spike"
    THRESHOLD_BREACH = "threshold_breach"
    TREND_ANOMALY = "trend_anomaly"

@dataclass
class QualityThreshold:
    """Quality monitoring threshold."""
    name: str
    metric: str
    value: float
    operator: str  # "gt", "lt", "eq"
    alert_type: AlertType
    severity: str  # "low", "medium", "high", "critical"

@dataclass
class QualityAlert:
    """Quality alert."""
    alert_id: str
    agent_id: str
    alert_type: AlertType
    severity: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool = False

class QualityMonitor:
    """Quality monitor for continuous agent assessment."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer1_test_foundation', 'agent_qa')
        self.lock = threading.RLock()
        self.monitoring = False
        self.monitor_thread = None
        self.monitoring_interval = 30.0  # seconds
        
        self.alerts: List[QualityAlert] = []
        self.thresholds: List[QualityThreshold] = []
        self.agent_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        
        if not self.enabled:
            return
        
        self._setup_default_thresholds()
        
        print("Quality monitor initialized")
        print(f"   Monitoring interval: {self.monitoring_interval}s")
        print(f"   Default thresholds: {len(self.thresholds)}")
    
    def _setup_default_thresholds(self):
        """Setup default quality thresholds."""
        self.thresholds = [
            QualityThreshold(
                name="low_quality_score",
                metric="overall_score",
                value=0.6,
                operator="lt",
                alert_type=AlertType.QUALITY_DEGRADATION,
                severity="high"
            ),
            QualityThreshold(
                name="poor_response_time",
                metric="response_time",
                value=500.0,
                operator="gt",
                alert_type=AlertType.PERFORMANCE_ISSUE,
                severity="medium"
            ),
            QualityThreshold(
                name="high_error_rate",
                metric="error_rate",
                value=0.05,
                operator="gt",
                alert_type=AlertType.ERROR_SPIKE,
                severity="high"
            ),
            QualityThreshold(
                name="memory_usage_high",
                metric="memory_usage",
                value=200.0,
                operator="gt",
                alert_type=AlertType.THRESHOLD_BREACH,
                severity="medium"
            )
        ]
    
    def start_monitoring(self):
        """Start quality monitoring."""
        if not self.enabled or self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print("Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop quality monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        print("Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_all_agents()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Brief pause before retrying
    
    def _check_all_agents(self):
        """Check quality for all monitored agents."""
        with self.lock:
            for agent_id in self.agent_metrics.keys():
                self._check_agent_quality(agent_id)
    
    def _check_agent_quality(self, agent_id: str):
        """Check quality for a specific agent."""
        agent_data = self.agent_metrics.get(agent_id, {})
        
        for threshold in self.thresholds:
            metric_values = agent_data.get(threshold.metric, [])
            if not metric_values:
                continue
            
            current_value = metric_values[-1]  # Latest value
            
            # Check threshold breach
            if self._check_threshold_breach(current_value, threshold):
                alert = self._create_alert(agent_id, threshold, current_value)
                self._raise_alert(alert)
            
            # Check for trend anomalies
            if len(metric_values) >= 5:
                if self._detect_trend_anomaly(metric_values, threshold):
                    alert = self._create_trend_alert(agent_id, threshold, metric_values)
                    self._raise_alert(alert)
    
    def _check_threshold_breach(self, value: float, threshold: QualityThreshold) -> bool:
        """Check if value breaches threshold."""
        if threshold.operator == "gt":
            return value > threshold.value
        elif threshold.operator == "lt":
            return value < threshold.value
        elif threshold.operator == "eq":
            return abs(value - threshold.value) < 0.01
        return False
    
    def _detect_trend_anomaly(self, values: List[float], threshold: QualityThreshold) -> bool:
        """Detect trend anomalies in metric values."""
        if len(values) < 5:
            return False
        
        # Simple trend detection - check if recent values show concerning trend
        recent_values = values[-3:]
        older_values = values[-5:-3]
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        # Check for significant degradation
        if threshold.metric in ["overall_score", "accuracy"]:
            # Higher is better - check for decline
            decline_threshold = 0.1  # 10% decline
            return (older_avg - recent_avg) / older_avg > decline_threshold
        else:
            # Lower is better - check for increase
            increase_threshold = 0.2  # 20% increase
            return (recent_avg - older_avg) / older_avg > increase_threshold
    
    def _create_alert(self, agent_id: str, threshold: QualityThreshold, current_value: float) -> QualityAlert:
        """Create quality alert."""
        alert_id = f"alert_{int(time.time())}_{agent_id}_{threshold.name}"
        
        message = f"Agent {agent_id}: {threshold.metric} {threshold.operator} {threshold.value} (current: {current_value:.3f})"
        
        return QualityAlert(
            alert_id=alert_id,
            agent_id=agent_id,
            alert_type=threshold.alert_type,
            severity=threshold.severity,
            message=message,
            metric_name=threshold.metric,
            current_value=current_value,
            threshold_value=threshold.value,
            timestamp=datetime.now()
        )
    
    def _create_trend_alert(self, agent_id: str, threshold: QualityThreshold, values: List[float]) -> QualityAlert:
        """Create trend anomaly alert."""
        alert_id = f"trend_{int(time.time())}_{agent_id}_{threshold.metric}"
        
        recent_avg = sum(values[-3:]) / 3
        message = f"Agent {agent_id}: Trend anomaly detected in {threshold.metric} (recent avg: {recent_avg:.3f})"
        
        return QualityAlert(
            alert_id=alert_id,
            agent_id=agent_id,
            alert_type=AlertType.TREND_ANOMALY,
            severity="medium",
            message=message,
            metric_name=threshold.metric,
            current_value=recent_avg,
            threshold_value=threshold.value,
            timestamp=datetime.now()
        )
    
    def _raise_alert(self, alert: QualityAlert):
        """Raise quality alert."""
        # Check for duplicate alerts (avoid spam)
        recent_alerts = [a for a in self.alerts if 
                        a.agent_id == alert.agent_id and 
                        a.metric_name == alert.metric_name and
                        (datetime.now() - a.timestamp).seconds < 300]  # 5 minutes
        
        if recent_alerts:
            return  # Skip duplicate alert
        
        with self.lock:
            self.alerts.append(alert)
        
        print(f"[{alert.severity.upper()}] Quality Alert: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def record_metric(self, agent_id: str, metric_name: str, value: float):
        """Record metric value for an agent."""
        with self.lock:
            if agent_id not in self.agent_metrics:
                self.agent_metrics[agent_id] = {}
            
            if metric_name not in self.agent_metrics[agent_id]:
                self.agent_metrics[agent_id][metric_name] = []
            
            # Keep only recent values (last 100 measurements)
            metrics_list = self.agent_metrics[agent_id][metric_name]
            metrics_list.append(value)
            if len(metrics_list) > 100:
                metrics_list.pop(0)
    
    def add_threshold(self, threshold: QualityThreshold):
        """Add custom quality threshold."""
        self.thresholds.append(threshold)
        print(f"Added quality threshold: {threshold.name}")
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add callback for quality alerts."""
        self.alert_callbacks.append(callback)
    
    def get_alerts(self, agent_id: str = None, severity: str = None, since: datetime = None) -> List[QualityAlert]:
        """Get quality alerts with optional filtering."""
        alerts = self.alerts.copy()
        
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    print(f"Alert acknowledged: {alert_id}")
                    break
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        with self.lock:
            total_alerts = len(self.alerts)
            unacknowledged_alerts = len([a for a in self.alerts if not a.acknowledged])
            critical_alerts = len([a for a in self.alerts if a.severity == "critical"])
            
            return {
                "monitoring": self.monitoring,
                "monitored_agents": len(self.agent_metrics),
                "total_alerts": total_alerts,
                "unacknowledged_alerts": unacknowledged_alerts,
                "critical_alerts": critical_alerts,
                "thresholds": len(self.thresholds),
                "monitoring_interval": self.monitoring_interval
            }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status for a specific agent."""
        agent_alerts = self.get_alerts(agent_id=agent_id)
        recent_alerts = [a for a in agent_alerts if (datetime.now() - a.timestamp).days < 1]
        
        metrics = self.agent_metrics.get(agent_id, {})
        latest_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                latest_metrics[metric_name] = values[-1]
        
        return {
            "agent_id": agent_id,
            "total_alerts": len(agent_alerts),
            "recent_alerts": len(recent_alerts),
            "latest_metrics": latest_metrics,
            "metrics_count": sum(len(values) for values in metrics.values())
        }
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.lock:
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_date]
        
        print(f"Cleared alerts older than {days} days")

def get_quality_monitor() -> QualityMonitor:
    """Get quality monitor instance."""
    return QualityMonitor()