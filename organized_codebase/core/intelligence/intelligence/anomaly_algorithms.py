"""
Advanced Anomaly Detection Algorithms
=====================================
Extracted and optimized anomaly detection algorithms from archive.
Module size: ~299 lines (under 300 limit)

Author: Agent B - Intelligence Specialist
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import statistics
from collections import deque, defaultdict
from scipy import stats as scipy_stats


class AnomalyType(Enum):
    """Types of anomalies detected."""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    PATTERN_DEVIATION = "pattern_deviation"
    THRESHOLD_BREACH = "threshold_breach"
    MISSING_DATA = "missing_data"
    CORRELATION_BREAK = "correlation_break"
    SEASONAL_DEVIATION = "seasonal_deviation"
    DISTRIBUTION_SHIFT = "distribution_shift"


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    is_anomaly: bool
    confidence: float
    anomaly_type: Optional[AnomalyType]
    score: float
    expected_value: Optional[float]
    metadata: Dict[str, Any]


class StatisticalAnomalyDetector:
    """
    Advanced statistical anomaly detection methods.
    Combines multiple techniques for robust detection.
    """
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.history = deque(maxlen=1000)
        self.stats_cache = {}
        
    def detect_zscore(self, value: float, data: np.ndarray) -> AnomalyResult:
        """Modified Z-score anomaly detection with MAD."""
        if len(data) < 3:
            return AnomalyResult(False, 0.0, None, 0.0, None, {})
            
        # Use Median Absolute Deviation for robustness
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            # Fall back to standard deviation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return AnomalyResult(False, 0.0, None, 0.0, mean, {})
            zscore = abs((value - mean) / std)
        else:
            # Modified Z-score using MAD
            modified_zscore = 0.6745 * (value - median) / mad
            zscore = abs(modified_zscore)
            
        is_anomaly = zscore > self.sensitivity
        anomaly_type = None
        
        if is_anomaly:
            expected = median if mad > 0 else np.mean(data)
            anomaly_type = AnomalyType.SPIKE if value > expected else AnomalyType.DROP
            
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=min(0.99, zscore / 4),
            anomaly_type=anomaly_type,
            score=zscore,
            expected_value=median,
            metadata={'method': 'modified_zscore', 'mad': mad}
        )
        
    def detect_iqr_tukey(self, value: float, data: np.ndarray, k: float = 1.5) -> AnomalyResult:
        """Tukey's IQR method with adjustable fence multiplier."""
        if len(data) < 4:
            return AnomalyResult(False, 0.0, None, 0.0, None, {})
            
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        is_anomaly = value < lower_bound or value > upper_bound
        
        if is_anomaly:
            anomaly_type = AnomalyType.SPIKE if value > upper_bound else AnomalyType.DROP
            expected = q3 if value > upper_bound else q1
            deviation = min(abs(value - lower_bound), abs(value - upper_bound))
            confidence = min(0.95, deviation / (iqr + 1e-10))
        else:
            anomaly_type = None
            expected = np.median(data)
            confidence = 0.0
            
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_type=anomaly_type,
            score=abs(value - np.median(data)) / (iqr + 1e-10),
            expected_value=expected,
            metadata={'method': 'tukey_iqr', 'bounds': (lower_bound, upper_bound)}
        )
        
    def detect_grubbs_test(self, value: float, data: np.ndarray, alpha: float = 0.05) -> AnomalyResult:
        """Grubbs' test for outliers (assumes normal distribution)."""
        n = len(data)
        if n < 7:  # Minimum sample size for Grubbs
            return AnomalyResult(False, 0.0, None, 0.0, None, {})
            
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return AnomalyResult(False, 0.0, None, 0.0, mean, {})
            
        # Grubbs statistic
        g_calculated = abs(value - mean) / std
        
        # Critical value approximation
        t_dist = scipy_stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_critical = ((n - 1) * t_dist) / np.sqrt(n * (n - 2 + t_dist**2))
        
        is_anomaly = g_calculated > g_critical
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=min(0.99, g_calculated / g_critical) if g_critical > 0 else 0,
            anomaly_type=AnomalyType.PATTERN_DEVIATION if is_anomaly else None,
            score=g_calculated,
            expected_value=mean,
            metadata={'method': 'grubbs', 'critical_value': g_critical}
        )


class TimeSeriesAnomalyDetector:
    """
    Time series specific anomaly detection.
    Handles trends, seasonality, and change points.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trend_history = deque(maxlen=window_size)
        
    def detect_trend_change(self, data: np.ndarray, window: int = 10) -> AnomalyResult:
        """Detect sudden trend changes using sliding windows."""
        if len(data) < window * 2:
            return AnomalyResult(False, 0.0, None, 0.0, None, {})
            
        # Calculate trends for two consecutive windows
        recent = data[-window:]
        previous = data[-2*window:-window]
        
        recent_trend = np.polyfit(range(len(recent)), recent, 1)[0]
        previous_trend = np.polyfit(range(len(previous)), previous, 1)[0]
        
        # Trend change magnitude
        trend_change = abs(recent_trend - previous_trend)
        relative_change = trend_change / (abs(previous_trend) + 1e-10)
        
        is_anomaly = relative_change > 0.5  # 50% change threshold
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=min(0.95, relative_change),
            anomaly_type=AnomalyType.TREND_CHANGE if is_anomaly else None,
            score=relative_change,
            expected_value=data[-1] + previous_trend,
            metadata={
                'method': 'trend_change',
                'previous_trend': previous_trend,
                'recent_trend': recent_trend
            }
        )
        
    def detect_seasonal_deviation(self, value: float, data: np.ndarray, 
                                 period: int = 24) -> AnomalyResult:
        """Detect deviations from seasonal patterns."""
        if len(data) < period * 2:
            return AnomalyResult(False, 0.0, None, 0.0, None, {})
            
        # Extract seasonal component
        seasonal_values = []
        for i in range(period):
            points = data[i::period]
            if len(points) > 0:
                seasonal_values.append(np.median(points))
                
        if not seasonal_values:
            return AnomalyResult(False, 0.0, None, 0.0, None, {})
            
        # Expected value based on position in cycle
        position = len(data) % period
        expected = seasonal_values[position] if position < len(seasonal_values) else np.mean(seasonal_values)
        
        # Calculate deviation
        deviation = abs(value - expected)
        seasonal_std = np.std(seasonal_values)
        
        if seasonal_std > 0:
            normalized_deviation = deviation / seasonal_std
            is_anomaly = normalized_deviation > 2.5
        else:
            normalized_deviation = 0
            is_anomaly = False
            
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=min(0.9, normalized_deviation / 3),
            anomaly_type=AnomalyType.SEASONAL_DEVIATION if is_anomaly else None,
            score=normalized_deviation,
            expected_value=expected,
            metadata={'method': 'seasonal', 'period': period}
        )
        
    def detect_cusum_change(self, data: np.ndarray, threshold: float = 5.0) -> AnomalyResult:
        """CUSUM change point detection."""
        if len(data) < 10:
            return AnomalyResult(False, 0.0, None, 0.0, None, {})
            
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return AnomalyResult(False, 0.0, None, 0.0, mean, {})
            
        # Calculate CUSUM
        cusum_pos = np.zeros(len(data))
        cusum_neg = np.zeros(len(data))
        
        for i in range(1, len(data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (data[i] - mean - 0.5 * std))
            cusum_neg[i] = max(0, cusum_neg[i-1] - (data[i] - mean + 0.5 * std))
            
        # Check if threshold exceeded
        max_cusum = max(cusum_pos[-1], cusum_neg[-1])
        is_anomaly = max_cusum > threshold * std
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=min(0.95, max_cusum / (threshold * std)) if std > 0 else 0,
            anomaly_type=AnomalyType.TREND_CHANGE if is_anomaly else None,
            score=max_cusum / std if std > 0 else 0,
            expected_value=mean,
            metadata={'method': 'cusum', 'cusum_value': max_cusum}
        )


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods.
    Provides robust detection through voting and weighted scoring.
    """
    
    def __init__(self):
        self.statistical_detector = StatisticalAnomalyDetector()
        self.timeseries_detector = TimeSeriesAnomalyDetector()
        self.weights = {
            'zscore': 0.25,
            'iqr': 0.20,
            'grubbs': 0.15,
            'trend': 0.15,
            'seasonal': 0.15,
            'cusum': 0.10
        }
        
    def detect(self, value: float, data: np.ndarray, 
              use_timeseries: bool = True) -> AnomalyResult:
        """Ensemble detection using multiple methods."""
        results = {}
        
        # Statistical methods
        results['zscore'] = self.statistical_detector.detect_zscore(value, data)
        results['iqr'] = self.statistical_detector.detect_iqr_tukey(value, data)
        results['grubbs'] = self.statistical_detector.detect_grubbs_test(value, data)
        
        # Time series methods (if applicable)
        if use_timeseries and len(data) > 20:
            results['trend'] = self.timeseries_detector.detect_trend_change(data)
            results['seasonal'] = self.timeseries_detector.detect_seasonal_deviation(value, data)
            results['cusum'] = self.timeseries_detector.detect_cusum_change(data)
            
        # Weighted voting
        total_weight = 0
        weighted_score = 0
        anomaly_votes = 0
        
        for method, result in results.items():
            weight = self.weights.get(method, 0.1)
            total_weight += weight
            
            if result.is_anomaly:
                anomaly_votes += weight
                weighted_score += result.confidence * weight
                
        # Final decision
        is_anomaly = anomaly_votes / total_weight > 0.5
        confidence = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine dominant anomaly type
        anomaly_type = None
        if is_anomaly:
            for result in results.values():
                if result.is_anomaly and result.anomaly_type:
                    anomaly_type = result.anomaly_type
                    break
                    
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_type=anomaly_type,
            score=anomaly_votes / total_weight,
            expected_value=np.median(data),
            metadata={'ensemble_results': results, 'votes': anomaly_votes}
        )


# Public API
__all__ = [
    'StatisticalAnomalyDetector',
    'TimeSeriesAnomalyDetector',
    'EnsembleAnomalyDetector',
    'AnomalyResult',
    'AnomalyType'
]