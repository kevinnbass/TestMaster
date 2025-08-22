"""
Advanced Anomaly Detection System
==================================

Sophisticated anomaly detection with ML-powered analysis and real-time alerting.
Integrated from archive advanced analytics components.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import statistics
import json
import threading
import time

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies detected"""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    PATTERN_DEVIATION = "pattern_deviation"
    THRESHOLD_BREACH = "threshold_breach"
    MISSING_DATA = "missing_data"
    CORRELATION_BREAK = "correlation_break"

class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    anomaly_id: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    detected_at: datetime
    value: float
    expected_value: float
    confidence: float
    description: str
    context: Dict[str, Any]

class AdvancedAnomalyDetector:
    """Advanced anomaly detection with multiple algorithms"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.z_threshold = self.config.get('z_threshold', 2.5)
        self.iqr_factor = self.config.get('iqr_factor', 1.5)
        self.window_size = self.config.get('window_size', 50)
        self.min_samples = self.config.get('min_samples', 10)
        
        # Data storage
        self.metric_windows = defaultdict(lambda: deque(maxlen=self.window_size))
        self.baseline_stats = defaultdict(dict)
        self.detected_anomalies = []
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start real-time anomaly monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Advanced anomaly detection started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Advanced anomaly detection stopped")
    
    def add_metric_value(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add new metric value for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            data_point = {'value': value, 'timestamp': timestamp}
            self.metric_windows[metric_name].append(data_point)
            
            # Update baseline statistics
            self._update_baseline_stats(metric_name)
            
            # Check for anomalies
            anomalies = self._detect_anomalies(metric_name, value, timestamp)
            self.detected_anomalies.extend(anomalies)
            
            return anomalies
    
    def _update_baseline_stats(self, metric_name: str):
        """Update baseline statistics for a metric"""
        window = self.metric_windows[metric_name]
        if len(window) < self.min_samples:
            return
        
        values = [point['value'] for point in window]
        
        self.baseline_stats[metric_name] = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'updated_at': datetime.now()
        }
    
    def _detect_anomalies(self, metric_name: str, value: float, timestamp: datetime) -> List[Anomaly]:
        """Detect anomalies using multiple methods"""
        anomalies = []
        
        if metric_name not in self.baseline_stats:
            return anomalies
        
        stats = self.baseline_stats[metric_name]
        
        # Z-Score anomaly detection
        z_anomaly = self._detect_z_score_anomaly(metric_name, value, timestamp, stats)
        if z_anomaly:
            anomalies.append(z_anomaly)
        
        # IQR anomaly detection
        iqr_anomaly = self._detect_iqr_anomaly(metric_name, value, timestamp, stats)
        if iqr_anomaly:
            anomalies.append(iqr_anomaly)
        
        # Trend change detection
        trend_anomaly = self._detect_trend_change(metric_name, value, timestamp)
        if trend_anomaly:
            anomalies.append(trend_anomaly)
        
        return anomalies
    
    def _detect_z_score_anomaly(self, metric_name: str, value: float, timestamp: datetime, stats: Dict) -> Optional[Anomaly]:
        """Detect anomalies using Z-score method"""
        if stats['std'] == 0:
            return None
        
        z_score = abs(value - stats['mean']) / stats['std']
        
        if z_score > self.z_threshold:
            severity = AnomalySeverity.CRITICAL if z_score > 3 else AnomalySeverity.WARNING
            anomaly_type = AnomalyType.SPIKE if value > stats['mean'] else AnomalyType.DROP
            
            return Anomaly(
                anomaly_id=f"z_{metric_name}_{int(timestamp.timestamp())}",
                metric_name=metric_name,
                anomaly_type=anomaly_type,
                severity=severity,
                detected_at=timestamp,
                value=value,
                expected_value=stats['mean'],
                confidence=min(z_score / 3, 1.0),
                description=f"Z-score anomaly: {z_score:.2f} (threshold: {self.z_threshold})",
                context={'z_score': z_score, 'method': 'z_score'}
            )
        
        return None
    
    def _detect_iqr_anomaly(self, metric_name: str, value: float, timestamp: datetime, stats: Dict) -> Optional[Anomaly]:
        """Detect anomalies using IQR method"""
        iqr = stats['q3'] - stats['q1']
        lower_bound = stats['q1'] - self.iqr_factor * iqr
        upper_bound = stats['q3'] + self.iqr_factor * iqr
        
        if value < lower_bound or value > upper_bound:
            severity = AnomalySeverity.WARNING
            anomaly_type = AnomalyType.SPIKE if value > upper_bound else AnomalyType.DROP
            
            distance = max(lower_bound - value, value - upper_bound, 0)
            confidence = min(distance / (iqr + 1), 1.0)
            
            return Anomaly(
                anomaly_id=f"iqr_{metric_name}_{int(timestamp.timestamp())}",
                metric_name=metric_name,
                anomaly_type=anomaly_type,
                severity=severity,
                detected_at=timestamp,
                value=value,
                expected_value=stats['median'],
                confidence=confidence,
                description=f"IQR outlier: value {value} outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                context={'iqr': iqr, 'bounds': [lower_bound, upper_bound], 'method': 'iqr'}
            )
        
        return None
    
    def _detect_trend_change(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect sudden trend changes"""
        window = self.metric_windows[metric_name]
        if len(window) < 20:  # Need enough data for trend analysis
            return None
        
        # Calculate recent trend vs historical trend
        recent_values = [p['value'] for p in list(window)[-10:]]
        historical_values = [p['value'] for p in list(window)[-20:-10]]
        
        if len(recent_values) < 5 or len(historical_values) < 5:
            return None
        
        recent_slope = self._calculate_slope(recent_values)
        historical_slope = self._calculate_slope(historical_values)
        
        # Detect significant trend change
        if abs(recent_slope - historical_slope) > 2.0:  # Threshold for trend change
            return Anomaly(
                anomaly_id=f"trend_{metric_name}_{int(timestamp.timestamp())}",
                metric_name=metric_name,
                anomaly_type=AnomalyType.TREND_CHANGE,
                severity=AnomalySeverity.INFO,
                detected_at=timestamp,
                value=value,
                expected_value=historical_values[-1],
                confidence=min(abs(recent_slope - historical_slope) / 5.0, 1.0),
                description=f"Trend change detected: slope changed from {historical_slope:.2f} to {recent_slope:.2f}",
                context={'recent_slope': recent_slope, 'historical_slope': historical_slope, 'method': 'trend'}
            )
        
        return None
    
    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of values using linear regression"""
        if len(values) < 2:
            return 0
        
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Anomaly]:
        """Get anomalies detected in the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.detected_anomalies if a.detected_at >= cutoff]
    
    def get_anomalies_by_metric(self, metric_name: str) -> List[Anomaly]:
        """Get all anomalies for a specific metric"""
        return [a for a in self.detected_anomalies if a.metric_name == metric_name]
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection"""
        recent_anomalies = self.get_recent_anomalies(24)
        
        return {
            'total_anomalies': len(self.detected_anomalies),
            'recent_anomalies': len(recent_anomalies),
            'anomalies_by_type': self._count_by_type(recent_anomalies),
            'anomalies_by_severity': self._count_by_severity(recent_anomalies),
            'metrics_monitored': list(self.metric_windows.keys()),
            'detection_methods': ['z_score', 'iqr', 'trend_change']
        }
    
    def _count_by_type(self, anomalies: List[Anomaly]) -> Dict[str, int]:
        """Count anomalies by type"""
        counts = defaultdict(int)
        for anomaly in anomalies:
            counts[anomaly.anomaly_type.value] += 1
        return dict(counts)
    
    def _count_by_severity(self, anomalies: List[Anomaly]) -> Dict[str, int]:
        """Count anomalies by severity"""
        counts = defaultdict(int)
        for anomaly in anomalies:
            counts[anomaly.severity.value] += 1
        return dict(counts)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Clean old anomalies (keep last 1000)
                if len(self.detected_anomalies) > 1000:
                    self.detected_anomalies = self.detected_anomalies[-1000:]
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in anomaly monitoring loop: {e}")
                time.sleep(30)

# Export
__all__ = ['AdvancedAnomalyDetector', 'Anomaly', 'AnomalyType', 'AnomalySeverity']