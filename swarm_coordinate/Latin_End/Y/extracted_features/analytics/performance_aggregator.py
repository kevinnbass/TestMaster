"""
Multi-System Performance Aggregation Module
Extracted from performance_analytics_dashboard.py for Agent X's Epsilon base integration
< 200 lines per STEELCLAD protocol

Provides comprehensive performance metric aggregation:
- Multi-source metric collection
- Real-time data aggregation
- Performance threshold monitoring
- System health assessment
"""

import asyncio
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class MetricPoint:
    """Individual metric data point"""
    value: float
    timestamp: datetime
    unit: str
    source: str = "system"

@dataclass
class SystemThresholds:
    """Performance thresholds for health assessment"""
    good: float
    warning: float
    critical: float

class PerformanceAggregator:
    """Multi-system performance metric aggregation engine"""
    
    def __init__(self, max_data_points: int = 100):
        self.max_data_points = max_data_points
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_data_points))
        self.system_sources = set()
        self._lock = threading.RLock()
        
        # Default performance thresholds
        self.thresholds = {
            'cpu_usage_percent': SystemThresholds(good=70, warning=85, critical=95),
            'memory_usage_percent': SystemThresholds(good=75, warning=85, critical=95),
            'response_time_ms': SystemThresholds(good=100, warning=250, critical=500),
            'cache_hit_ratio': SystemThresholds(good=0.9, warning=0.8, critical=0.7),
            'error_rate': SystemThresholds(good=0.01, warning=0.05, critical=0.1),
            'disk_usage_percent': SystemThresholds(good=80, warning=90, critical=95),
            'network_latency_ms': SystemThresholds(good=10, warning=50, critical=100)
        }
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system performance metrics"""
        all_metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'performance_data': {},
            'system_health': HealthStatus.UNKNOWN.value,
            'sources': list(self.system_sources),
            'alerts': []
        }
        
        with self._lock:
            # Aggregate latest metrics from each source
            performance_data = {}
            health_scores = []
            alerts = []
            
            for metric_name, history in self.metrics_history.items():
                if history:
                    latest = history[-1]
                    performance_data[metric_name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp.isoformat(),
                        'unit': latest.unit,
                        'source': latest.source
                    }
                    
                    # Assess health and generate alerts
                    health_status, alert = self._assess_metric_health(metric_name, latest.value)
                    if health_status != HealthStatus.UNKNOWN:
                        health_scores.append(self._health_to_score(health_status))
                    if alert:
                        alerts.append(alert)
            
            all_metrics['performance_data'] = performance_data
            all_metrics['alerts'] = alerts
            
            # Calculate overall system health
            if health_scores:
                avg_health_score = sum(health_scores) / len(health_scores)
                all_metrics['system_health'] = self._score_to_health(avg_health_score).value
            
        return all_metrics
    
    def add_metric_data(self, metric_name: str, value: float, unit: str = "", source: str = "system"):
        """Add new metric data point"""
        with self._lock:
            metric_point = MetricPoint(
                value=value,
                timestamp=datetime.now(timezone.utc),
                unit=unit,
                source=source
            )
            
            self.metrics_history[metric_name].append(metric_point)
            self.system_sources.add(source)
    
    def add_batch_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """Add multiple metrics in batch"""
        for metric_name, metric_data in metrics.items():
            self.add_metric_data(
                metric_name=metric_name,
                value=metric_data.get('value', 0),
                unit=metric_data.get('unit', ''),
                source=metric_data.get('source', 'system')
            )
    
    def _assess_metric_health(self, metric_name: str, value: float) -> tuple[HealthStatus, Optional[Dict[str, Any]]]:
        """Assess health status of a specific metric"""
        if metric_name not in self.thresholds:
            return HealthStatus.UNKNOWN, None
        
        threshold = self.thresholds[metric_name]
        alert = None
        
        # Determine health status based on metric type
        if metric_name in ['cache_hit_ratio']:  # Higher is better
            if value >= threshold.good:
                status = HealthStatus.GOOD
            elif value >= threshold.warning:
                status = HealthStatus.WARNING
                alert = self._create_alert(metric_name, value, "warning", f"Cache hit ratio below optimal: {value:.2%}")
            else:
                status = HealthStatus.CRITICAL
                alert = self._create_alert(metric_name, value, "critical", f"Cache hit ratio critically low: {value:.2%}")
        
        elif metric_name in ['error_rate']:  # Lower is better
            if value <= threshold.good:
                status = HealthStatus.GOOD
            elif value <= threshold.warning:
                status = HealthStatus.WARNING
                alert = self._create_alert(metric_name, value, "warning", f"Error rate elevated: {value:.2%}")
            else:
                status = HealthStatus.CRITICAL
                alert = self._create_alert(metric_name, value, "critical", f"Error rate critically high: {value:.2%}")
        
        else:  # Standard metrics (lower is better)
            if value <= threshold.good:
                status = HealthStatus.GOOD
            elif value <= threshold.warning:
                status = HealthStatus.WARNING
                alert = self._create_alert(metric_name, value, "warning", f"{metric_name} approaching limits: {value}")
            elif value <= threshold.critical:
                status = HealthStatus.CRITICAL
                alert = self._create_alert(metric_name, value, "critical", f"{metric_name} in critical range: {value}")
            else:
                status = HealthStatus.CRITICAL
                alert = self._create_alert(metric_name, value, "critical", f"{metric_name} exceeded critical threshold: {value}")
        
        return status, alert
    
    def _create_alert(self, metric_name: str, value: float, severity: str, message: str) -> Dict[str, Any]:
        """Create alert dictionary"""
        return {
            'metric': metric_name,
            'value': value,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _health_to_score(self, health: HealthStatus) -> float:
        """Convert health status to numeric score"""
        score_map = {
            HealthStatus.EXCELLENT: 1.0,
            HealthStatus.GOOD: 0.8,
            HealthStatus.WARNING: 0.6,
            HealthStatus.CRITICAL: 0.3,
            HealthStatus.UNKNOWN: 0.5
        }
        return score_map.get(health, 0.5)
    
    def _score_to_health(self, score: float) -> HealthStatus:
        """Convert numeric score to health status"""
        if score >= 0.9:
            return HealthStatus.EXCELLENT
        elif score >= 0.75:
            return HealthStatus.GOOD
        elif score >= 0.5:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL
    
    def get_metric_trends(self, metric_name: str, lookback_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for specific metric"""
        with self._lock:
            if metric_name not in self.metrics_history:
                return {"status": "not_found", "message": f"Metric {metric_name} not found"}
            
            history = self.metrics_history[metric_name]
            if len(history) < 2:
                return {"status": "insufficient_data", "message": "Need at least 2 data points"}
            
            # Filter by time window
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            recent_data = [point for point in history if point.timestamp >= cutoff_time]
            
            if len(recent_data) < 2:
                recent_data = list(history)[-10:]  # Use last 10 points as fallback
            
            values = [point.value for point in recent_data]
            
            return {
                "status": "success",
                "metric": metric_name,
                "current_value": values[-1],
                "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable",
                "change": values[-1] - values[0],
                "data_points": len(values),
                "min_value": min(values),
                "max_value": max(values),
                "avg_value": sum(values) / len(values)
            }
    
    def update_thresholds(self, metric_name: str, thresholds: SystemThresholds):
        """Update performance thresholds for a metric"""
        self.thresholds[metric_name] = thresholds
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get summary of aggregation status"""
        with self._lock:
            return {
                "metrics_tracked": len(self.metrics_history),
                "data_sources": len(self.system_sources),
                "total_data_points": sum(len(history) for history in self.metrics_history.values()),
                "oldest_data": min([history[0].timestamp for history in self.metrics_history.values() if history], default=None),
                "newest_data": max([history[-1].timestamp for history in self.metrics_history.values() if history], default=None)
            }

# Plugin interface for Agent X integration
def create_performance_aggregator_plugin(config: Dict[str, Any] = None):
    """Factory function to create performance aggregator plugin"""
    max_data_points = config.get('max_data_points', 100) if config else 100
    aggregator = PerformanceAggregator(max_data_points)
    
    # Apply custom thresholds if provided
    if config and 'thresholds' in config:
        for metric_name, threshold_data in config['thresholds'].items():
            thresholds = SystemThresholds(**threshold_data)
            aggregator.update_thresholds(metric_name, thresholds)
    
    return aggregator