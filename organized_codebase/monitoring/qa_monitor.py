"""
Quality Monitor Component
=========================

Monitors agent quality metrics and generates alerts.
Part of modularized agent_qa system.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import deque, defaultdict
import statistics
import uuid

from .qa_base import (
    QualityThreshold, QualityAlert, QualityMetric,
    AlertType, ScoreCategory, QualityLevel
)


class QualityMonitor:
    """Monitors agent quality metrics and generates alerts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality monitor."""
        self.config = config or {}
        
        # Alert management
        self._thresholds: List[QualityThreshold] = []
        self._active_alerts: Dict[str, QualityAlert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._alert_callbacks: List[Callable] = []
        
        # Metrics tracking
        self._metrics_buffer: deque = deque(maxlen=10000)
        self._metrics_by_category: Dict[ScoreCategory, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = self.config.get('monitor_interval', 60)
        
        # Statistical tracking
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._trend_window = self.config.get('trend_window', 3600)  # 1 hour
        
        self._initialize_default_thresholds()
    
    def _initialize_default_thresholds(self):
        """Set up default quality thresholds."""
        defaults = [
            QualityThreshold(
                name="Low Quality Score",
                metric="quality_score",
                value=60.0,
                operator="lt",
                alert_type=AlertType.QUALITY_DEGRADATION,
                severity="high"
            ),
            QualityThreshold(
                name="High Error Rate",
                metric="error_rate",
                value=0.1,
                operator="gt",
                alert_type=AlertType.ERROR_SPIKE,
                severity="critical"
            ),
            QualityThreshold(
                name="Slow Response Time",
                metric="response_time",
                value=5000,
                operator="gt",
                alert_type=AlertType.PERFORMANCE_ISSUE,
                severity="medium"
            )
        ]
        self._thresholds.extend(defaults)
    
    def add_threshold(self, threshold: QualityThreshold):
        """Add a monitoring threshold."""
        self._thresholds.append(threshold)
    
    def record_metric(self, metric: QualityMetric):
        """Record a quality metric."""
        self._metrics_buffer.append(metric)
        self._metrics_by_category[metric.category].append(metric)
        
        # Check thresholds
        self._check_thresholds(metric)
    
    def _check_thresholds(self, metric: QualityMetric):
        """Check if metric breaches any thresholds."""
        for threshold in self._thresholds:
            if threshold.metric == metric.name:
                if self._evaluate_threshold(metric.value, threshold):
                    self._generate_alert(threshold, metric)
    
    def _evaluate_threshold(self, value: float, threshold: QualityThreshold) -> bool:
        """Evaluate if value breaches threshold."""
        if threshold.operator == "gt":
            return value > threshold.value
        elif threshold.operator == "lt":
            return value < threshold.value
        elif threshold.operator == "eq":
            return value == threshold.value
        return False
    
    def _generate_alert(self, threshold: QualityThreshold, metric: QualityMetric):
        """Generate a quality alert."""
        alert = QualityAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=threshold.alert_type,
            severity=threshold.severity,
            message=f"{threshold.name}: {metric.name} = {metric.value:.2f}",
            timestamp=datetime.now(),
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=threshold.value,
            context={'category': metric.category.value}
        )
        
        self._active_alerts[alert.alert_id] = alert
        self._alert_history.append(alert)
        
        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass  # Ignore callback errors
    
    def register_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self._active_alerts[alert_id]
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self._analyze_trends()
                self._check_anomalies()
                time.sleep(self._monitor_interval)
            except Exception:
                pass  # Continue monitoring despite errors
    
    def _analyze_trends(self):
        """Analyze metric trends."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self._trend_window)
        
        # Analyze each category
        for category, metrics in self._metrics_by_category.items():
            recent_metrics = [
                m for m in metrics 
                if m.timestamp >= window_start
            ]
            
            if len(recent_metrics) >= 10:
                values = [m.value for m in recent_metrics]
                trend = self._calculate_trend(values)
                
                if abs(trend) > 0.2:  # 20% change
                    self._generate_trend_alert(category, trend)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend coefficient."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _check_anomalies(self):
        """Check for anomalous patterns."""
        # Implement anomaly detection logic
        pass
    
    def _generate_trend_alert(self, category: ScoreCategory, trend: float):
        """Generate alert for trend anomaly."""
        direction = "increasing" if trend > 0 else "decreasing"
        alert = QualityAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=AlertType.TREND_ANOMALY,
            severity="medium",
            message=f"{category.value} metrics {direction} by {abs(trend)*100:.1f}%",
            timestamp=datetime.now(),
            metric_name=f"{category.value}_trend",
            current_value=trend,
            threshold_value=0.2,
            context={'category': category.value}
        )
        
        self._active_alerts[alert.alert_id] = alert
        self._alert_history.append(alert)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        summary = {}
        
        for category, metrics in self._metrics_by_category.items():
            if metrics:
                values = [m.value for m in metrics]
                summary[category.value] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
        
        return summary


# Export
__all__ = ['QualityMonitor']