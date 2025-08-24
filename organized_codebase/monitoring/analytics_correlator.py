"""
Analytics Correlator
====================

Advanced correlation and anomaly detection for analytics data.
Identifies patterns, relationships, and anomalies across different metrics.

Author: TestMaster Team
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import statistics
import math

logger = logging.getLogger(__name__)

class AnalyticsCorrelator:
    """
    Correlates analytics data and detects anomalies across multiple metrics.
    """
    
    def __init__(self, history_window: int = 1000, correlation_threshold: float = 0.7):
        """
        Initialize the analytics correlator.
        
        Args:
            history_window: Number of data points to keep for correlation analysis
            correlation_threshold: Minimum correlation coefficient to flag relationships
        """
        self.history_window = history_window
        self.correlation_threshold = correlation_threshold
        
        # Data storage for correlation analysis
        self.metric_history = defaultdict(lambda: deque(maxlen=history_window))
        self.correlation_matrix = {}
        self.anomaly_patterns = []
        
        # Anomaly detection parameters
        self.anomaly_threshold = 2.5  # Standard deviations
        self.pattern_memory = deque(maxlen=100)
        
        logger.info("Analytics Correlator initialized")
    
    def add_metrics_sample(self, metrics: Dict[str, Any]):
        """
        Add a metrics sample for correlation analysis.
        
        Args:
            metrics: Dictionary of metric values with timestamp
        """
        timestamp = metrics.get('timestamp', datetime.now().isoformat())
        
        # Extract numeric metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                self.metric_history[key].append({
                    'timestamp': timestamp,
                    'value': value
                })
    
    def calculate_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between all metric pairs.
        
        Returns:
            Dictionary of correlation coefficients between metrics
        """
        correlations = {}
        metric_names = list(self.metric_history.keys())
        
        for i, metric1 in enumerate(metric_names):
            correlations[metric1] = {}
            
            for j, metric2 in enumerate(metric_names):
                if i == j:
                    correlations[metric1][metric2] = 1.0
                elif metric2 in correlations and metric1 in correlations[metric2]:
                    # Use previously calculated correlation (symmetric)
                    correlations[metric1][metric2] = correlations[metric2][metric1]
                else:
                    correlation = self._calculate_pearson_correlation(metric1, metric2)
                    correlations[metric1][metric2] = correlation
        
        self.correlation_matrix = correlations
        return correlations
    
    def detect_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in current metrics based on historical patterns.
        
        Args:
            current_metrics: Current metric values
            
        Returns:
            List of detected anomalies with details
        """
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            if not isinstance(current_value, (int, float)) or math.isnan(current_value):
                continue
            
            if metric_name not in self.metric_history:
                continue
            
            # Get historical values
            historical_values = [point['value'] for point in self.metric_history[metric_name]]
            
            if len(historical_values) < 10:  # Need sufficient data
                continue
            
            # Statistical anomaly detection
            mean = statistics.mean(historical_values)
            stdev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
            
            if stdev > 0:
                z_score = abs(current_value - mean) / stdev
                
                if z_score > self.anomaly_threshold:
                    anomaly = {
                        'metric': metric_name,
                        'current_value': current_value,
                        'expected_value': mean,
                        'z_score': z_score,
                        'severity': self._calculate_anomaly_severity(z_score),
                        'type': 'statistical',
                        'timestamp': datetime.now().isoformat()
                    }
                    anomalies.append(anomaly)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(metric_name, current_value)
            anomalies.extend(pattern_anomalies)
        
        # Correlation-based anomaly detection
        correlation_anomalies = self._detect_correlation_anomalies(current_metrics)
        anomalies.extend(correlation_anomalies)
        
        # Store anomaly patterns for learning
        if anomalies:
            self.pattern_memory.append({
                'timestamp': datetime.now().isoformat(),
                'anomalies': anomalies,
                'metrics': current_metrics.copy()
            })
        
        return anomalies
    
    def identify_metric_relationships(self) -> List[Dict[str, Any]]:
        """
        Identify significant relationships between metrics.
        
        Returns:
            List of metric relationships with correlation strength
        """
        if not self.correlation_matrix:
            self.calculate_correlations()
        
        relationships = []
        
        for metric1, correlations in self.correlation_matrix.items():
            for metric2, correlation in correlations.items():
                if metric1 != metric2 and abs(correlation) >= self.correlation_threshold:
                    relationships.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation,
                        'strength': self._classify_correlation_strength(correlation),
                        'type': 'positive' if correlation > 0 else 'negative'
                    })
        
        # Sort by correlation strength
        relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return relationships
    
    def predict_metric_values(self, metric_name: str, horizon_minutes: int = 30) -> Dict[str, Any]:
        """
        Predict future metric values based on historical patterns.
        
        Args:
            metric_name: Name of the metric to predict
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Prediction with confidence intervals
        """
        if metric_name not in self.metric_history:
            return {}
        
        historical_values = [point['value'] for point in self.metric_history[metric_name]]
        
        if len(historical_values) < 20:  # Need sufficient data
            return {'error': 'Insufficient historical data for prediction'}
        
        # Simple trend-based prediction
        recent_values = historical_values[-20:]  # Last 20 points
        
        # Calculate trend
        x = list(range(len(recent_values)))
        y = recent_values
        
        # Linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future value
        future_x = len(recent_values) + horizon_minutes / 5  # Assuming 5-minute intervals
        predicted_value = slope * future_x + intercept
        
        # Calculate confidence interval
        residuals = [y[i] - (slope * x[i] + intercept) for i in range(n)]
        mse = sum(r ** 2 for r in residuals) / n
        std_error = math.sqrt(mse)
        
        confidence_95 = 1.96 * std_error
        
        return {
            'predicted_value': predicted_value,
            'confidence_interval': {
                'lower': predicted_value - confidence_95,
                'upper': predicted_value + confidence_95
            },
            'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
            'confidence_score': max(0, min(100, 100 - abs(std_error / statistics.mean(y) * 100)))
        }
    
    def get_correlation_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive correlation insights.
        
        Returns:
            Insights about metric correlations and patterns
        """
        relationships = self.identify_metric_relationships()
        
        insights = {
            'strong_correlations': [r for r in relationships if abs(r['correlation']) > 0.8],
            'moderate_correlations': [r for r in relationships if 0.5 <= abs(r['correlation']) <= 0.8],
            'negative_correlations': [r for r in relationships if r['correlation'] < -0.5],
            'total_relationships': len(relationships),
            'metrics_analyzed': len(self.metric_history),
            'data_points_per_metric': {
                name: len(history) for name, history in self.metric_history.items()
            }
        }
        
        # Add insights
        if insights['strong_correlations']:
            insights['insights'] = [
                f"Strong correlation detected between {r['metric1']} and {r['metric2']} (r={r['correlation']:.3f})"
                for r in insights['strong_correlations'][:3]
            ]
        
        if insights['negative_correlations']:
            insights['insights'] = insights.get('insights', []) + [
                f"Negative correlation: {r['metric1']} decreases as {r['metric2']} increases (r={r['correlation']:.3f})"
                for r in insights['negative_correlations'][:2]
            ]
        
        return insights
    
    def _calculate_pearson_correlation(self, metric1: str, metric2: str) -> float:
        """Calculate Pearson correlation coefficient between two metrics."""
        if metric1 not in self.metric_history or metric2 not in self.metric_history:
            return 0.0
        
        # Align data points by timestamp (approximate)
        data1 = self.metric_history[metric1]
        data2 = self.metric_history[metric2]
        
        if len(data1) < 2 or len(data2) < 2:
            return 0.0
        
        # Simple alignment by index (assuming similar collection intervals)
        min_length = min(len(data1), len(data2))
        values1 = [data1[i]['value'] for i in range(min_length)]
        values2 = [data2[i]['value'] for i in range(min_length)]
        
        if len(values1) < 2:
            return 0.0
        
        try:
            # Calculate Pearson correlation
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)
            
            numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(len(values1)))
            
            sum_sq1 = sum((values1[i] - mean1) ** 2 for i in range(len(values1)))
            sum_sq2 = sum((values2[i] - mean2) ** 2 for i in range(len(values2)))
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
            
        except Exception as e:
            logger.debug(f"Error calculating correlation between {metric1} and {metric2}: {e}")
            return 0.0
    
    def _detect_pattern_anomalies(self, metric_name: str, current_value: float) -> List[Dict[str, Any]]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        if len(self.metric_history[metric_name]) < 10:
            return anomalies
        
        historical_values = [point['value'] for point in self.metric_history[metric_name]]
        
        # Detect sudden spikes (value much higher than recent average)
        recent_avg = statistics.mean(historical_values[-5:])
        if current_value > recent_avg * 3:  # 3x spike
            anomalies.append({
                'metric': metric_name,
                'current_value': current_value,
                'expected_value': recent_avg,
                'type': 'spike',
                'severity': 'high',
                'message': f"Sudden spike detected: {current_value:.2f} vs recent avg {recent_avg:.2f}"
            })
        
        # Detect sudden drops
        if current_value < recent_avg * 0.3:  # 70% drop
            anomalies.append({
                'metric': metric_name,
                'current_value': current_value,
                'expected_value': recent_avg,
                'type': 'drop',
                'severity': 'medium',
                'message': f"Sudden drop detected: {current_value:.2f} vs recent avg {recent_avg:.2f}"
            })
        
        return anomalies
    
    def _detect_correlation_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies based on correlation violations."""
        anomalies = []
        
        if not self.correlation_matrix:
            return anomalies
        
        for metric1, correlations in self.correlation_matrix.items():
            if metric1 not in current_metrics:
                continue
                
            for metric2, correlation in correlations.items():
                if metric2 not in current_metrics or abs(correlation) < self.correlation_threshold:
                    continue
                
                value1 = current_metrics[metric1]
                value2 = current_metrics[metric2]
                
                if not isinstance(value1, (int, float)) or not isinstance(value2, (int, float)):
                    continue
                
                # Check if current values violate expected correlation
                historical1 = [p['value'] for p in self.metric_history[metric1]]
                historical2 = [p['value'] for p in self.metric_history[metric2]]
                
                if len(historical1) < 10 or len(historical2) < 10:
                    continue
                
                # Normalize values to detect correlation violations
                mean1, std1 = statistics.mean(historical1), statistics.stdev(historical1)
                mean2, std2 = statistics.mean(historical2), statistics.stdev(historical2)
                
                if std1 == 0 or std2 == 0:
                    continue
                
                norm1 = (value1 - mean1) / std1
                norm2 = (value2 - mean2) / std2
                
                # For positive correlation, normalized values should have same sign
                # For negative correlation, normalized values should have opposite signs
                if correlation > 0.7 and norm1 * norm2 < -1:  # Strong positive correlation violated
                    anomalies.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation,
                        'type': 'correlation_violation',
                        'severity': 'medium',
                        'message': f"Correlation violation: {metric1} and {metric2} expected to move together"
                    })
                elif correlation < -0.7 and norm1 * norm2 > 1:  # Strong negative correlation violated
                    anomalies.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation,
                        'type': 'correlation_violation',
                        'severity': 'medium',
                        'message': f"Correlation violation: {metric1} and {metric2} expected to move oppositely"
                    })
        
        return anomalies
    
    def _calculate_anomaly_severity(self, z_score: float) -> str:
        """Calculate anomaly severity based on z-score."""
        if z_score > 4:
            return 'critical'
        elif z_score > 3:
            return 'high'
        elif z_score > 2.5:
            return 'medium'
        else:
            return 'low'
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr > 0.9:
            return 'very_strong'
        elif abs_corr > 0.7:
            return 'strong'
        elif abs_corr > 0.5:
            return 'moderate'
        elif abs_corr > 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def clear_history(self):
        """Clear correlation history."""
        self.metric_history.clear()
        self.correlation_matrix.clear()
        self.pattern_memory.clear()
        logger.info("Correlation history cleared")