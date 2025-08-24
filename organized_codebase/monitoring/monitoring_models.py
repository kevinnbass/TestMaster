"""
Monitoring Data Models
=====================

Core data structures for performance monitoring system.
Extracted from realtime_performance_monitoring.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import statistics
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricCategory(Enum):
    """Performance metric categories."""
    SYSTEM_RESOURCE = "system_resource"
    APPLICATION_PERFORMANCE = "application_performance"
    NETWORK_PERFORMANCE = "network_performance"
    USER_EXPERIENCE = "user_experience"
    BUSINESS_METRIC = "business_metric"
    CUSTOM = "custom"


class MonitoringMode(Enum):
    """Monitoring operation modes."""
    PASSIVE = "passive"  # Collect metrics only
    ACTIVE = "active"    # Collect metrics and trigger actions
    PREDICTIVE = "predictive"  # Include predictive monitoring
    ADAPTIVE = "adaptive"  # Adaptive monitoring based on conditions


class SystemType(Enum):
    """System types for monitoring."""
    INTELLIGENCE = "intelligence"
    TESTING = "testing"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    ORCHESTRATION = "orchestration"
    CUSTOM = "custom"


@dataclass
class PerformanceMetric:
    """Performance metric definition."""
    metric_id: str
    name: str
    system: SystemType
    category: MetricCategory
    
    # Metric configuration
    unit: str = ""
    description: str = ""
    collection_interval_seconds: int = 30
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    baseline_value: Optional[float] = None
    
    # Data storage
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Statistical properties
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_deviation: Optional[float] = None
    
    # Trend analysis
    trend_direction: Optional[str] = None
    trend_strength: float = 0.0
    anomaly_score: float = 0.0
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    enabled: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add new metric value."""
        if not timestamp:
            timestamp = datetime.now()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
        self.last_updated = timestamp
        
        # Update statistics
        self._update_statistics()
        
        # Check for anomalies
        self._detect_anomalies()
    
    def _update_statistics(self):
        """Update statistical properties."""
        if not self.values:
            return
        
        values_list = list(self.values)
        
        self.min_value = min(values_list)
        self.max_value = max(values_list)
        self.mean_value = statistics.mean(values_list)
        
        if len(values_list) > 1:
            self.std_deviation = statistics.stdev(values_list)
            self._analyze_trend()
    
    def _analyze_trend(self):
        """Analyze trend in metric values."""
        if len(self.values) < 10:
            return
        
        values_list = list(self.values)[-20:]  # Last 20 values
        x_values = list(range(len(values_list)))
        
        # Simple linear regression for trend
        n = len(values_list)
        sum_x = sum(x_values)
        sum_y = sum(values_list)
        sum_xy = sum(x_values[i] * values_list[i] for i in range(n))
        sum_xx = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        
        self.trend_strength = abs(slope)
        
        if abs(slope) < 0.001:
            self.trend_direction = "stable"
        elif slope > 0:
            self.trend_direction = "increasing"
        else:
            self.trend_direction = "decreasing"
    
    def _detect_anomalies(self):
        """Detect anomalies in metric values."""
        if len(self.values) < 20:
            return
        
        values_list = list(self.values)
        current_value = values_list[-1]
        
        # Z-score based anomaly detection
        if self.mean_value and self.std_deviation and self.std_deviation > 0:
            z_score = abs(current_value - self.mean_value) / self.std_deviation
            self.anomaly_score = z_score
        else:
            self.anomaly_score = 0.0
    
    def is_threshold_breached(self) -> Optional[AlertSeverity]:
        """Check if current value breaches thresholds."""
        if not self.values:
            return None
        
        current_value = self.values[-1]
        
        if self.critical_threshold is not None and current_value >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        elif self.warning_threshold is not None and current_value >= self.warning_threshold:
            return AlertSeverity.WARNING
        
        return None
    
    def get_recent_values(self, count: int = 10) -> List[Tuple[datetime, float]]:
        """Get recent metric values with timestamps."""
        recent_count = min(count, len(self.values))
        return list(zip(
            list(self.timestamps)[-recent_count:],
            list(self.values)[-recent_count:]
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "system": self.system.value,
            "category": self.category.value,
            "unit": self.unit,
            "current_value": self.values[-1] if self.values else None,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_deviation": self.std_deviation,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "anomaly_score": self.anomaly_score,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "last_updated": self.last_updated.isoformat(),
            "enabled": self.enabled
        }


@dataclass
class PerformanceAlert:
    """Performance alert."""
    # Alert details (required fields first)
    metric_id: str
    system: SystemType
    severity: AlertSeverity
    title: str
    description: str
    current_value: float
    
    # Optional fields with defaults
    alert_id: str = field(default_factory=lambda: f"alert_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.now)
    threshold_value: Optional[float] = None
    baseline_value: Optional[float] = None
    
    # State management
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    # Actions
    auto_actions_triggered: List[str] = field(default_factory=list)
    escalation_level: int = 0
    
    def acknowledge(self):
        """Acknowledge the alert."""
        self.acknowledged = True
    
    def resolve(self):
        """Resolve the alert."""
        self.resolved = True
        self.resolution_time = datetime.now()
    
    def escalate(self):
        """Escalate the alert."""
        self.escalation_level += 1
    
    def get_duration(self) -> float:
        """Get alert duration in seconds."""
        end_time = self.resolution_time or datetime.now()
        return (end_time - self.timestamp).total_seconds()


@dataclass
class SystemHealthSnapshot:
    """System health snapshot."""
    # Required fields first
    system: SystemType
    
    # Optional fields with defaults
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Overall health
    health_score: float = 100.0  # 0-100
    status: str = "healthy"  # healthy, degraded, critical, offline
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_usage: float = 0.0
    
    # Performance metrics
    response_time_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    error_rate_percent: float = 0.0
    
    # Detailed metrics
    active_connections: int = 0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    
    # Issues
    active_alerts: int = 0
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def calculate_health_score(self):
        """Calculate overall health score."""
        # Base score
        score = 100.0
        
        # Resource penalties
        if self.cpu_usage > 80:
            score -= min((self.cpu_usage - 80) * 2, 30)
        if self.memory_usage > 85:
            score -= min((self.memory_usage - 85) * 3, 30)
        if self.disk_usage > 90:
            score -= min((self.disk_usage - 90) * 4, 20)
        
        # Performance penalties
        if self.response_time_ms > 1000:
            score -= min((self.response_time_ms - 1000) / 100, 20)
        if self.error_rate_percent > 1:
            score -= min(self.error_rate_percent * 10, 25)
        
        # Alert penalties
        score -= min(self.active_alerts * 5, 20)
        score -= len(self.critical_issues) * 10
        
        self.health_score = max(score, 0.0)
        
        # Update status
        if self.health_score >= 90:
            self.status = "healthy"
        elif self.health_score >= 70:
            self.status = "degraded"
        elif self.health_score >= 30:
            self.status = "critical"
        else:
            self.status = "offline"


__all__ = [
    'AlertSeverity',
    'MetricCategory',
    'MonitoringMode',
    'SystemType',
    'PerformanceMetric',
    'PerformanceAlert',
    'SystemHealthSnapshot'
]