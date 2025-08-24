"""
Analytics Anomaly Detection System
===================================

Detects anomalies in analytics data using statistical methods and machine learning.
Provides real-time alerts for unusual patterns and outliers.

Author: TestMaster Team
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
    """Types of anomalies."""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    PATTERN_DEVIATION = "pattern_deviation"
    THRESHOLD_BREACH = "threshold_breach"
    MISSING_DATA = "missing_data"
    CORRELATION_BREAK = "correlation_break"

class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    detected_at: datetime
    value: float
    expected_value: float
    deviation: float
    confidence: float
    description: str
    context: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AnalyticsAnomalyDetector:
    """
    Advanced anomaly detection for analytics data.
    """
    
    def __init__(self,
                 window_size: int = 100,
                 sensitivity: float = 2.0,
                 min_data_points: int = 10):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Size of sliding window for analysis
            sensitivity: Sensitivity factor (lower = more sensitive)
            min_data_points: Minimum data points before detection starts
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_data_points = min_data_points
        
        # Metric history
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.metric_timestamps = defaultdict(lambda: deque(maxlen=window_size))
        
        # Statistical models
        self.metric_stats = defaultdict(dict)
        self.baseline_models = defaultdict(dict)
        
        # Anomaly tracking
        self.detected_anomalies = deque(maxlen=1000)
        self.active_anomalies = {}
        
        # Thresholds and rules
        self.static_thresholds = {}
        self.dynamic_thresholds = defaultdict(dict)
        self.correlation_pairs = []
        
        # Detection methods
        self.detection_methods = {
            'zscore': self._detect_zscore_anomaly,
            'iqr': self._detect_iqr_anomaly,
            'isolation_forest': self._detect_isolation_forest_anomaly,
            'trend': self._detect_trend_anomaly,
            'missing': self._detect_missing_data,
            'correlation': self._detect_correlation_anomaly
        }
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'detection_time_ms': 0,
            'metrics_monitored': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Analytics Anomaly Detector initialized")
    
    def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        Add a data point for a metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            # Add to history
            self.metric_history[metric_name].append(value)
            self.metric_timestamps[metric_name].append(timestamp)
            
            # Update statistics
            self._update_statistics(metric_name)
            
            # Check for anomalies if enough data
            if len(self.metric_history[metric_name]) >= self.min_data_points:
                anomalies = self._detect_anomalies(metric_name, value, timestamp)
                
                for anomaly in anomalies:
                    self._handle_anomaly(anomaly)
    
    def _update_statistics(self, metric_name: str):
        """Update statistical models for a metric."""
        values = list(self.metric_history[metric_name])
        
        if len(values) < 2:
            return
        
        # Calculate basic statistics
        self.metric_stats[metric_name] = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'count': len(values)
        }
        
        # Update baseline model
        self._update_baseline_model(metric_name, values)
    
    def _update_baseline_model(self, metric_name: str, values: List[float]):
        """Update baseline model for expected values."""
        # Simple exponential smoothing for baseline
        alpha = 0.3  # Smoothing factor
        
        if metric_name not in self.baseline_models:
            self.baseline_models[metric_name] = {
                'level': values[0],
                'trend': 0,
                'seasonality': []
            }
        
        model = self.baseline_models[metric_name]
        
        # Update level (exponential smoothing)
        for value in values[-10:]:  # Use last 10 values
            model['level'] = alpha * value + (1 - alpha) * model['level']
        
        # Detect trend
        if len(values) >= 20:
            recent = values[-10:]
            older = values[-20:-10]
            model['trend'] = (statistics.mean(recent) - statistics.mean(older)) / 10
        
        # Detect seasonality (simplified)
        if len(values) >= 24:  # At least 24 data points
            model['seasonality'] = self._detect_seasonality(values)
    
    def _detect_seasonality(self, values: List[float]) -> List[float]:
        """Detect seasonal patterns in data."""
        # Simplified seasonality detection
        # In production, use more sophisticated methods like FFT
        
        season_length = min(24, len(values) // 2)  # Assume 24-hour seasonality
        
        if len(values) < season_length * 2:
            return []
        
        seasonal_pattern = []
        for i in range(season_length):
            points = [values[j] for j in range(i, len(values), season_length)]
            seasonal_pattern.append(statistics.mean(points))
        
        return seasonal_pattern
    
    def _detect_anomalies(self, metric_name: str, value: float, timestamp: datetime) -> List[Anomaly]:
        """Detect anomalies for a metric value."""
        anomalies = []
        
        # Run multiple detection methods
        for method_name, method_func in self.detection_methods.items():
            anomaly = method_func(metric_name, value, timestamp)
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_zscore_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect anomalies using Z-score method."""
        stats = self.metric_stats.get(metric_name, {})
        
        if not stats or stats['stdev'] == 0:
            return None
        
        # Calculate Z-score
        zscore = abs((value - stats['mean']) / stats['stdev'])
        
        if zscore > self.sensitivity:
            anomaly_type = AnomalyType.SPIKE if value > stats['mean'] else AnomalyType.DROP
            
            return Anomaly(
                anomaly_id=f"zscore_{metric_name}_{timestamp.timestamp()}",
                metric_name=metric_name,
                anomaly_type=anomaly_type,
                severity=self._calculate_severity(zscore, 2, 3, 4),
                detected_at=timestamp,
                value=value,
                expected_value=stats['mean'],
                deviation=zscore,
                confidence=min(0.99, zscore / 5),  # Higher Z-score = higher confidence
                description=f"{anomaly_type.value} detected: Z-score = {zscore:.2f}",
                context={'method': 'zscore', 'stats': stats}
            )
        
        return None
    
    def _detect_iqr_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect anomalies using Interquartile Range method."""
        stats = self.metric_stats.get(metric_name, {})
        
        if not stats or 'iqr' not in stats:
            return None
        
        # IQR boundaries
        lower_bound = stats['q1'] - 1.5 * stats['iqr']
        upper_bound = stats['q3'] + 1.5 * stats['iqr']
        
        if value < lower_bound or value > upper_bound:
            anomaly_type = AnomalyType.SPIKE if value > upper_bound else AnomalyType.DROP
            expected = stats['q1'] if value < lower_bound else stats['q3']
            
            return Anomaly(
                anomaly_id=f"iqr_{metric_name}_{timestamp.timestamp()}",
                metric_name=metric_name,
                anomaly_type=anomaly_type,
                severity=AnomalySeverity.WARNING,
                detected_at=timestamp,
                value=value,
                expected_value=expected,
                deviation=abs(value - expected),
                confidence=0.75,
                description=f"IQR outlier: value outside [{lower_bound:.2f}, {upper_bound:.2f}]",
                context={'method': 'iqr', 'bounds': (lower_bound, upper_bound)}
            )
        
        return None
    
    def _detect_isolation_forest_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect anomalies using Isolation Forest algorithm (simplified)."""
        values = list(self.metric_history[metric_name])
        
        if len(values) < 20:
            return None
        
        # Simplified isolation forest scoring
        # In production, use sklearn's IsolationForest
        
        # Calculate isolation score based on distance from other points
        distances = [abs(value - v) for v in values[:-1]]  # Exclude current value
        avg_distance = statistics.mean(distances)
        min_distance = min(distances)
        
        # Anomaly score: points far from others have higher scores
        isolation_score = min_distance / (avg_distance + 1e-10)
        
        if isolation_score > 2.0:  # Threshold for isolation
            return Anomaly(
                anomaly_id=f"isolation_{metric_name}_{timestamp.timestamp()}",
                metric_name=metric_name,
                anomaly_type=AnomalyType.PATTERN_DEVIATION,
                severity=AnomalySeverity.INFO,
                detected_at=timestamp,
                value=value,
                expected_value=statistics.median(values),
                deviation=isolation_score,
                confidence=min(0.9, isolation_score / 3),
                description=f"Isolated point detected: score = {isolation_score:.2f}",
                context={'method': 'isolation_forest', 'score': isolation_score}
            )
        
        return None
    
    def _detect_trend_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect trend changes."""
        values = list(self.metric_history[metric_name])
        
        if len(values) < 20:
            return None
        
        # Calculate trend for last 10 vs previous 10 points
        recent_trend = self._calculate_trend(values[-10:])
        previous_trend = self._calculate_trend(values[-20:-10])
        
        trend_change = abs(recent_trend - previous_trend)
        
        # Significant trend change
        if trend_change > 0.5:  # Threshold
            return Anomaly(
                anomaly_id=f"trend_{metric_name}_{timestamp.timestamp()}",
                metric_name=metric_name,
                anomaly_type=AnomalyType.TREND_CHANGE,
                severity=AnomalySeverity.WARNING,
                detected_at=timestamp,
                value=value,
                expected_value=values[-2] + previous_trend if len(values) > 1 else value,
                deviation=trend_change,
                confidence=0.7,
                description=f"Trend change detected: {previous_trend:.2f} -> {recent_trend:.2f}",
                context={
                    'method': 'trend',
                    'previous_trend': previous_trend,
                    'recent_trend': recent_trend
                }
            )
        
        return None
    
    def _detect_missing_data(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect missing data patterns."""
        timestamps = list(self.metric_timestamps[metric_name])
        
        if len(timestamps) < 2:
            return None
        
        # Check time gaps
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                     for i in range(1, len(timestamps))]
        
        if time_diffs:
            avg_interval = statistics.mean(time_diffs)
            last_interval = (timestamp - timestamps[-2]).total_seconds() if len(timestamps) > 1 else 0
            
            # Detect large gaps
            if last_interval > avg_interval * 3:
                return Anomaly(
                    anomaly_id=f"missing_{metric_name}_{timestamp.timestamp()}",
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.MISSING_DATA,
                    severity=AnomalySeverity.INFO,
                    detected_at=timestamp,
                    value=value,
                    expected_value=value,
                    deviation=last_interval - avg_interval,
                    confidence=0.8,
                    description=f"Data gap detected: {last_interval:.0f}s (expected {avg_interval:.0f}s)",
                    context={
                        'method': 'missing_data',
                        'gap_seconds': last_interval,
                        'expected_interval': avg_interval
                    }
                )
        
        return None
    
    def _detect_correlation_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect correlation breaks between related metrics."""
        # Check if this metric is part of any correlation pairs
        for pair in self.correlation_pairs:
            if metric_name in pair:
                other_metric = pair[1] if pair[0] == metric_name else pair[0]
                
                if other_metric in self.metric_history:
                    correlation = self._calculate_correlation(metric_name, other_metric)
                    
                    if correlation is not None and abs(correlation) < 0.5:  # Weak correlation
                        return Anomaly(
                            anomaly_id=f"correlation_{metric_name}_{timestamp.timestamp()}",
                            metric_name=metric_name,
                            anomaly_type=AnomalyType.CORRELATION_BREAK,
                            severity=AnomalySeverity.WARNING,
                            detected_at=timestamp,
                            value=value,
                            expected_value=value,
                            deviation=1 - abs(correlation),
                            confidence=0.6,
                            description=f"Correlation break with {other_metric}: {correlation:.2f}",
                            context={
                                'method': 'correlation',
                                'paired_metric': other_metric,
                                'correlation': correlation
                            }
                        )
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_correlation(self, metric1: str, metric2: str) -> Optional[float]:
        """Calculate correlation between two metrics."""
        values1 = list(self.metric_history[metric1])
        values2 = list(self.metric_history[metric2])
        
        # Align lengths
        min_len = min(len(values1), len(values2))
        if min_len < 10:
            return None
        
        values1 = values1[-min_len:]
        values2 = values2[-min_len:]
        
        # Calculate Pearson correlation
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        
        numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(min_len))
        denominator1 = sum((v - mean1) ** 2 for v in values1) ** 0.5
        denominator2 = sum((v - mean2) ** 2 for v in values2) ** 0.5
        
        if denominator1 == 0 or denominator2 == 0:
            return None
        
        correlation = numerator / (denominator1 * denominator2)
        return correlation
    
    def _calculate_severity(self, value: float, info_threshold: float, 
                          warning_threshold: float, critical_threshold: float) -> AnomalySeverity:
        """Calculate anomaly severity based on thresholds."""
        if value >= critical_threshold:
            return AnomalySeverity.CRITICAL
        elif value >= warning_threshold:
            return AnomalySeverity.WARNING
        else:
            return AnomalySeverity.INFO
    
    def _handle_anomaly(self, anomaly: Anomaly):
        """Handle detected anomaly."""
        with self.lock:
            # Add to detected anomalies
            self.detected_anomalies.append(anomaly)
            
            # Track active anomalies
            key = f"{anomaly.metric_name}_{anomaly.anomaly_type.value}"
            self.active_anomalies[key] = anomaly
            
            # Update statistics
            self.detection_stats['total_detections'] += 1
            
            # Log based on severity
            if anomaly.severity == AnomalySeverity.CRITICAL:
                logger.error(f"CRITICAL ANOMALY: {anomaly.description}")
            elif anomaly.severity == AnomalySeverity.WARNING:
                logger.warning(f"Anomaly detected: {anomaly.description}")
            else:
                logger.info(f"Anomaly info: {anomaly.description}")
    
    def set_threshold(self, metric_name: str, min_value: Optional[float] = None, 
                     max_value: Optional[float] = None):
        """Set static thresholds for a metric."""
        self.static_thresholds[metric_name] = {
            'min': min_value,
            'max': max_value
        }
    
    def add_correlation_pair(self, metric1: str, metric2: str):
        """Add a pair of metrics that should be correlated."""
        self.correlation_pairs.append((metric1, metric2))
    
    def get_anomalies(self, 
                      metric_name: Optional[str] = None,
                      anomaly_type: Optional[AnomalyType] = None,
                      severity: Optional[AnomalySeverity] = None,
                      hours: int = 24) -> List[Dict[str, Any]]:
        """Get detected anomalies with filters."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            anomalies = []
            
            for anomaly in self.detected_anomalies:
                # Apply filters
                if anomaly.detected_at < cutoff_time:
                    continue
                    
                if metric_name and anomaly.metric_name != metric_name:
                    continue
                    
                if anomaly_type and anomaly.anomaly_type != anomaly_type:
                    continue
                    
                if severity and anomaly.severity != severity:
                    continue
                
                # Convert to dict
                anomalies.append({
                    'anomaly_id': anomaly.anomaly_id,
                    'metric_name': anomaly.metric_name,
                    'type': anomaly.anomaly_type.value,
                    'severity': anomaly.severity.value,
                    'detected_at': anomaly.detected_at.isoformat(),
                    'value': anomaly.value,
                    'expected_value': anomaly.expected_value,
                    'deviation': anomaly.deviation,
                    'confidence': anomaly.confidence,
                    'description': anomaly.description,
                    'resolved': anomaly.resolved
                })
            
            return anomalies
    
    def resolve_anomaly(self, anomaly_id: str):
        """Mark an anomaly as resolved."""
        with self.lock:
            for anomaly in self.detected_anomalies:
                if anomaly.anomaly_id == anomaly_id:
                    anomaly.resolved = True
                    anomaly.resolved_at = datetime.now()
                    
                    # Remove from active anomalies
                    key = f"{anomaly.metric_name}_{anomaly.anomaly_type.value}"
                    self.active_anomalies.pop(key, None)
                    
                    logger.info(f"Anomaly resolved: {anomaly_id}")
                    return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        with self.lock:
            return {
                'detection_stats': self.detection_stats.copy(),
                'active_anomalies': len(self.active_anomalies),
                'total_anomalies': len(self.detected_anomalies),
                'metrics_monitored': len(self.metric_history),
                'detection_methods': list(self.detection_methods.keys()),
                'sensitivity': self.sensitivity,
                'window_size': self.window_size
            }
    
    def export_anomalies(self, format: str = 'json') -> str:
        """Export anomalies in specified format."""
        anomalies = self.get_anomalies(hours=24*7)  # Last week
        
        if format == 'json':
            return json.dumps(anomalies, indent=2)
        elif format == 'csv':
            # Simple CSV export
            if not anomalies:
                return "anomaly_id,metric_name,type,severity,detected_at,value,expected_value,deviation,confidence,description\n"
            
            csv_lines = ["anomaly_id,metric_name,type,severity,detected_at,value,expected_value,deviation,confidence,description"]
            
            for anomaly in anomalies:
                csv_lines.append(
                    f"{anomaly['anomaly_id']},{anomaly['metric_name']},{anomaly['type']},"
                    f"{anomaly['severity']},{anomaly['detected_at']},{anomaly['value']},"
                    f"{anomaly['expected_value']},{anomaly['deviation']},{anomaly['confidence']},"
                    f"\"{anomaly['description']}\""
                )
            
            return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def shutdown(self):
        """Shutdown anomaly detector."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Analytics Anomaly Detector shutdown")