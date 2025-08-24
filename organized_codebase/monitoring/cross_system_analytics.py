"""
Cross-System Analytics and Metrics Correlation
==============================================

Advanced analytics engine that correlates metrics across all unified systems,
provides predictive insights, and enables intelligent decision-making.

Integrates with:
- Unified Observability for metrics collection
- Workflow Engine for execution analytics
- Cross-System APIs for system health data
- Unified Dashboard for analytics visualization

Author: TestMaster Phase 1B Integration System
"""

import asyncio
import json
import logging
import numpy as np
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
import threading
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import cross-system APIs
from .cross_system_apis import SystemType, cross_system_coordinator


# ============================================================================
# ANALYTICS TYPES
# ============================================================================

class MetricType(Enum):
    """Types of metrics"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    BUSINESS = "business"
    QUALITY = "quality"
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


class CorrelationType(Enum):
    """Types of correlations"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    STRONG = "strong"
    WEAK = "weak"
    INVERSE = "inverse"


class TrendDirection(Enum):
    """Trend directions"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    CYCLIC = "cyclic"


@dataclass
class MetricDataPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: float
    system: SystemType
    metric_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "system": self.system.value,
            "metric_name": self.metric_name,
            "metadata": self.metadata
        }


@dataclass
class MetricSeries:
    """Time series of metric data"""
    metric_id: str
    system: SystemType
    metric_name: str
    metric_type: MetricType
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Statistical properties
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_deviation: Optional[float] = None
    
    # Trend analysis
    trend_direction: Optional[TrendDirection] = None
    trend_strength: float = 0.0
    
    # Metadata
    unit: str = ""
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    
    def add_data_point(self, point: MetricDataPoint):
        """Add data point to series"""
        self.data_points.append(point)
        self._update_statistics()
    
    def _update_statistics(self):
        """Update statistical properties"""
        if not self.data_points:
            return
        
        values = [p.value for p in self.data_points]
        
        self.min_value = min(values)
        self.max_value = max(values)
        self.mean_value = statistics.mean(values)
        
        if len(values) > 1:
            self.std_deviation = statistics.stdev(values)
            self._analyze_trend()
    
    def _analyze_trend(self):
        """Analyze trend in the data"""
        if len(self.data_points) < 10:
            return
        
        values = [p.value for p in self.data_points]
        x = list(range(len(values)))
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        self.trend_strength = abs(r_value)
        
        if abs(slope) < 0.001:  # Very small slope
            self.trend_direction = TrendDirection.STABLE
        elif slope > 0:
            self.trend_direction = TrendDirection.INCREASING
        else:
            self.trend_direction = TrendDirection.DECREASING
        
        # Check for volatility
        if self.std_deviation and self.mean_value:
            cv = self.std_deviation / self.mean_value  # Coefficient of variation
            if cv > 0.5:  # High volatility
                self.trend_direction = TrendDirection.VOLATILE
    
    def get_recent_values(self, count: int = 10) -> List[float]:
        """Get recent values"""
        recent_points = list(self.data_points)[-count:]
        return [p.value for p in recent_points]
    
    def get_values_in_timeframe(self, start_time: datetime, end_time: datetime) -> List[MetricDataPoint]:
        """Get values within timeframe"""
        return [p for p in self.data_points 
                if start_time <= p.timestamp <= end_time]


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    metric1_id: str
    metric2_id: str
    correlation_coefficient: float
    correlation_type: CorrelationType
    significance: float
    sample_size: int
    
    # Analysis metadata
    timeframe: Tuple[datetime, datetime]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if correlation is statistically significant"""
        return self.significance < threshold
    
    def is_strong(self, threshold: float = 0.7) -> bool:
        """Check if correlation is strong"""
        return abs(self.correlation_coefficient) >= threshold


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    metric_id: str
    timestamp: datetime
    value: float
    expected_value: float
    deviation_score: float
    anomaly_type: str  # "outlier", "trend_break", "pattern_change"
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Prediction result"""
    metric_id: str
    prediction_horizon: timedelta
    predicted_values: List[Tuple[datetime, float]]
    confidence_interval: List[Tuple[float, float]]
    model_accuracy: float
    prediction_timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# CROSS-SYSTEM ANALYTICS ENGINE
# ============================================================================

class CrossSystemAnalyticsEngine:
    """
    Advanced analytics engine for cross-system metrics correlation,
    anomaly detection, and predictive analysis.
    """
    
    def __init__(self, max_metrics: int = 10000):
        self.logger = logging.getLogger("cross_system_analytics")
        
        # Metric storage
        self.metric_series: Dict[str, MetricSeries] = {}
        self.max_metrics = max_metrics
        
        # Correlation analysis
        self.correlations: Dict[Tuple[str, str], CorrelationResult] = {}
        self.correlation_threshold = 0.3
        
        # Anomaly detection
        self.anomalies: List[AnomalyDetection] = []
        self.anomaly_thresholds = {
            "z_score": 3.0,
            "iqr_multiplier": 1.5
        }
        
        # Predictive models
        self.predictions: Dict[str, PredictionResult] = {}
        
        # Analytics configuration
        self.analysis_config = {
            "correlation_window_hours": 24,
            "anomaly_detection_window_hours": 6,
            "prediction_horizon_hours": 4,
            "min_samples_for_analysis": 20
        }
        
        # Background tasks
        self.analytics_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Performance tracking
        self.analytics_stats = {
            "total_metrics": 0,
            "correlations_found": 0,
            "anomalies_detected": 0,
            "predictions_made": 0,
            "analysis_cycles": 0
        }
        
        self.logger.info("Cross-system analytics engine initialized")
    
    async def start_analytics(self):
        """Start the analytics engine"""
        if self.is_running:
            return
        
        self.logger.info("Starting cross-system analytics engine")
        self.is_running = True
        
        # Start background analytics task
        self.analytics_task = asyncio.create_task(self._analytics_loop())
        
        self.logger.info("Cross-system analytics engine started")
    
    async def stop_analytics(self):
        """Stop the analytics engine"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping cross-system analytics engine")
        self.is_running = False
        
        if self.analytics_task:
            self.analytics_task.cancel()
            try:
                await self.analytics_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Cross-system analytics engine stopped")
    
    async def _analytics_loop(self):
        """Main analytics processing loop"""
        while self.is_running:
            try:
                # Collect metrics from all systems
                await self._collect_system_metrics()
                
                # Perform correlation analysis
                await self._analyze_correlations()
                
                # Detect anomalies
                await self._detect_anomalies()
                
                # Generate predictions
                await self._generate_predictions()
                
                # Update statistics
                self.analytics_stats["analysis_cycles"] += 1
                
                # Sleep between analysis cycles
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                self.logger.error(f"Analytics loop error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Collect metrics from all unified systems"""
        try:
            # Collect from each system type
            for system_type in SystemType:
                await self._collect_metrics_from_system(system_type)
                
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    async def _collect_metrics_from_system(self, system_type: SystemType):
        """Collect metrics from specific system"""
        try:
            # Request metrics from system
            response = await cross_system_coordinator.execute_cross_system_operation(
                operation="get_metrics",
                target_system=system_type,
                parameters={"include_detailed": True}
            )
            
            if response.success and response.result:
                metrics_data = response.result
                await self._process_system_metrics(system_type, metrics_data)
                
        except Exception as e:
            self.logger.debug(f"Could not collect metrics from {system_type.value}: {e}")
    
    async def _process_system_metrics(self, system_type: SystemType, metrics_data: Dict[str, Any]):
        """Process metrics data from a system"""
        try:
            current_time = datetime.now()
            
            # Process different metric types
            for metric_name, metric_value in metrics_data.items():
                if isinstance(metric_value, (int, float)):
                    await self._add_metric_point(
                        system_type, metric_name, float(metric_value), current_time
                    )
                elif isinstance(metric_value, dict) and "value" in metric_value:
                    await self._add_metric_point(
                        system_type, metric_name, float(metric_value["value"]), current_time,
                        metadata=metric_value.get("metadata", {})
                    )
                    
        except Exception as e:
            self.logger.error(f"Error processing metrics from {system_type.value}: {e}")
    
    async def _add_metric_point(self, system: SystemType, metric_name: str, value: float,
                              timestamp: datetime, metadata: Dict[str, Any] = None):
        """Add metric data point"""
        metric_id = f"{system.value}.{metric_name}"
        
        # Create metric series if not exists
        if metric_id not in self.metric_series:
            if len(self.metric_series) >= self.max_metrics:
                # Remove oldest metric series
                oldest_id = min(self.metric_series.keys(), 
                              key=lambda k: self.metric_series[k].data_points[0].timestamp 
                              if self.metric_series[k].data_points else datetime.min)
                del self.metric_series[oldest_id]
            
            metric_type = self._classify_metric_type(metric_name)
            
            self.metric_series[metric_id] = MetricSeries(
                metric_id=metric_id,
                system=system,
                metric_name=metric_name,
                metric_type=metric_type,
                description=f"{metric_name} from {system.value}",
                tags={system.value, metric_type.value}
            )
            
            self.analytics_stats["total_metrics"] += 1
        
        # Add data point
        data_point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            system=system,
            metric_name=metric_name,
            metadata=metadata or {}
        )
        
        self.metric_series[metric_id].add_data_point(data_point)
    
    def _classify_metric_type(self, metric_name: str) -> MetricType:
        """Classify metric type based on name"""
        name_lower = metric_name.lower()
        
        if any(word in name_lower for word in ["cpu", "memory", "disk", "network"]):
            return MetricType.RESOURCE
        elif any(word in name_lower for word in ["latency", "response_time", "duration"]):
            return MetricType.LATENCY
        elif any(word in name_lower for word in ["error", "fail", "exception"]):
            return MetricType.ERROR_RATE
        elif any(word in name_lower for word in ["throughput", "requests", "transactions"]):
            return MetricType.THROUGHPUT
        elif any(word in name_lower for word in ["availability", "uptime"]):
            return MetricType.AVAILABILITY
        elif any(word in name_lower for word in ["quality", "score", "rating"]):
            return MetricType.QUALITY
        else:
            return MetricType.PERFORMANCE
    
    async def _analyze_correlations(self):
        """Analyze correlations between metrics"""
        try:
            # Get metrics with sufficient data
            eligible_metrics = [
                series for series in self.metric_series.values()
                if len(series.data_points) >= self.analysis_config["min_samples_for_analysis"]
            ]
            
            if len(eligible_metrics) < 2:
                return
            
            # Analyze pairwise correlations
            correlation_count = 0
            for i, series1 in enumerate(eligible_metrics):
                for series2 in eligible_metrics[i+1:]:
                    correlation = await self._calculate_correlation(series1, series2)
                    
                    if correlation and abs(correlation.correlation_coefficient) >= self.correlation_threshold:
                        correlation_key = (series1.metric_id, series2.metric_id)
                        self.correlations[correlation_key] = correlation
                        
                        correlation_count += 1
                        if correlation_count >= 100:  # Limit correlations per cycle
                            break
                
                if correlation_count >= 100:
                    break
            
            if correlation_count > 0:
                self.analytics_stats["correlations_found"] += correlation_count
                self.logger.info(f"Found {correlation_count} significant correlations")
                
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
    
    async def _calculate_correlation(self, series1: MetricSeries, series2: MetricSeries) -> Optional[CorrelationResult]:
        """Calculate correlation between two metric series"""
        try:
            # Get overlapping time window
            window_hours = self.analysis_config["correlation_window_hours"]
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=window_hours)
            
            # Get values in timeframe
            values1_raw = series1.get_values_in_timeframe(start_time, end_time)
            values2_raw = series2.get_values_in_timeframe(start_time, end_time)
            
            if len(values1_raw) < 10 or len(values2_raw) < 10:
                return None
            
            # Align timestamps and get synchronized values
            values1, values2 = self._align_time_series(values1_raw, values2_raw)
            
            if len(values1) < 10:
                return None
            
            # Calculate Pearson correlation
            correlation_coeff, p_value = stats.pearsonr(values1, values2)
            
            # Determine correlation type
            if correlation_coeff > 0.7:
                corr_type = CorrelationType.STRONG
            elif correlation_coeff < -0.7:
                corr_type = CorrelationType.STRONG
            elif correlation_coeff > 0.3:
                corr_type = CorrelationType.POSITIVE
            elif correlation_coeff < -0.3:
                corr_type = CorrelationType.NEGATIVE
            else:
                corr_type = CorrelationType.WEAK
            
            return CorrelationResult(
                metric1_id=series1.metric_id,
                metric2_id=series2.metric_id,
                correlation_coefficient=correlation_coeff,
                correlation_type=corr_type,
                significance=p_value,
                sample_size=len(values1),
                timeframe=(start_time, end_time)
            )
            
        except Exception as e:
            self.logger.debug(f"Correlation calculation failed: {e}")
            return None
    
    def _align_time_series(self, series1: List[MetricDataPoint], 
                          series2: List[MetricDataPoint]) -> Tuple[List[float], List[float]]:
        """Align two time series by timestamp"""
        # Create timestamp maps
        map1 = {int(p.timestamp.timestamp()): p.value for p in series1}
        map2 = {int(p.timestamp.timestamp()): p.value for p in series2}
        
        # Find common timestamps
        common_timestamps = set(map1.keys()) & set(map2.keys())
        
        if len(common_timestamps) < 10:
            # Try approximate alignment (within 60 seconds)
            aligned_values1 = []
            aligned_values2 = []
            
            for ts1, val1 in map1.items():
                closest_ts2 = min(map2.keys(), key=lambda x: abs(x - ts1), default=None)
                if closest_ts2 and abs(closest_ts2 - ts1) <= 60:  # Within 60 seconds
                    aligned_values1.append(val1)
                    aligned_values2.append(map2[closest_ts2])
            
            return aligned_values1, aligned_values2
        
        # Use exact timestamp matches
        values1 = [map1[ts] for ts in sorted(common_timestamps)]
        values2 = [map2[ts] for ts in sorted(common_timestamps)]
        
        return values1, values2
    
    async def _detect_anomalies(self):
        """Detect anomalies in metric series"""
        try:
            anomaly_count = 0
            
            for series in self.metric_series.values():
                if len(series.data_points) < self.analysis_config["min_samples_for_analysis"]:
                    continue
                
                anomalies = await self._detect_series_anomalies(series)
                self.anomalies.extend(anomalies)
                anomaly_count += len(anomalies)
                
                # Limit anomalies processed per cycle
                if anomaly_count >= 50:
                    break
            
            # Clean up old anomalies (keep last 1000)
            if len(self.anomalies) > 1000:
                self.anomalies = self.anomalies[-1000:]
            
            if anomaly_count > 0:
                self.analytics_stats["anomalies_detected"] += anomaly_count
                self.logger.info(f"Detected {anomaly_count} anomalies")
                
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
    
    async def _detect_series_anomalies(self, series: MetricSeries) -> List[AnomalyDetection]:
        """Detect anomalies in a metric series"""
        anomalies = []
        
        try:
            # Get recent data for analysis
            window_hours = self.analysis_config["anomaly_detection_window_hours"]
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=window_hours)
            
            recent_points = series.get_values_in_timeframe(start_time, end_time)
            
            if len(recent_points) < 10:
                return anomalies
            
            values = [p.value for p in recent_points]
            
            # Z-score based anomaly detection
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val > 0:
                z_threshold = self.anomaly_thresholds["z_score"]
                
                for point in recent_points[-10:]:  # Check last 10 points
                    z_score = abs(point.value - mean_val) / std_val
                    
                    if z_score > z_threshold:
                        anomaly = AnomalyDetection(
                            metric_id=series.metric_id,
                            timestamp=point.timestamp,
                            value=point.value,
                            expected_value=mean_val,
                            deviation_score=z_score,
                            anomaly_type="outlier",
                            confidence=min(z_score / z_threshold, 1.0),
                            context={
                                "z_score": z_score,
                                "threshold": z_threshold,
                                "mean": mean_val,
                                "std": std_val
                            }
                        )
                        anomalies.append(anomaly)
            
            # IQR based anomaly detection
            if len(values) >= 20:
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                if iqr > 0:
                    multiplier = self.anomaly_thresholds["iqr_multiplier"]
                    lower_bound = q1 - multiplier * iqr
                    upper_bound = q3 + multiplier * iqr
                    
                    for point in recent_points[-5:]:  # Check last 5 points
                        if point.value < lower_bound or point.value > upper_bound:
                            deviation = min(abs(point.value - lower_bound), 
                                          abs(point.value - upper_bound))
                            
                            anomaly = AnomalyDetection(
                                metric_id=series.metric_id,
                                timestamp=point.timestamp,
                                value=point.value,
                                expected_value=(q1 + q3) / 2,
                                deviation_score=deviation / iqr if iqr > 0 else 0,
                                anomaly_type="outlier",
                                confidence=0.8,
                                context={
                                    "method": "iqr",
                                    "q1": q1,
                                    "q3": q3,
                                    "iqr": iqr,
                                    "bounds": [lower_bound, upper_bound]
                                }
                            )
                            anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.debug(f"Anomaly detection failed for {series.metric_id}: {e}")
        
        return anomalies
    
    async def _generate_predictions(self):
        """Generate predictions for metric series"""
        try:
            prediction_count = 0
            
            for series in self.metric_series.values():
                if len(series.data_points) < 50:  # Need more data for predictions
                    continue
                
                prediction = await self._predict_series(series)
                if prediction:
                    self.predictions[series.metric_id] = prediction
                    prediction_count += 1
                
                # Limit predictions per cycle
                if prediction_count >= 20:
                    break
            
            if prediction_count > 0:
                self.analytics_stats["predictions_made"] += prediction_count
                self.logger.info(f"Generated {prediction_count} predictions")
                
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
    
    async def _predict_series(self, series: MetricSeries) -> Optional[PredictionResult]:
        """Generate prediction for a metric series"""
        try:
            # Get recent values for prediction
            recent_values = series.get_recent_values(50)
            
            if len(recent_values) < 20:
                return None
            
            # Simple linear regression prediction
            x = np.array(range(len(recent_values)))
            y = np.array(recent_values)
            
            # Fit linear model
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Generate predictions
            horizon_hours = self.analysis_config["prediction_horizon_hours"]
            prediction_horizon = timedelta(hours=horizon_hours)
            
            last_timestamp = series.data_points[-1].timestamp
            predicted_values = []
            confidence_intervals = []
            
            # Predict for next N points
            prediction_points = min(horizon_hours * 4, 24)  # 15-minute intervals
            
            for i in range(1, prediction_points + 1):
                future_timestamp = last_timestamp + timedelta(minutes=15 * i)
                future_x = len(recent_values) + i
                
                predicted_value = slope * future_x + intercept
                
                # Simple confidence interval based on standard error
                confidence_margin = std_err * 2  # ~95% confidence
                
                predicted_values.append((future_timestamp, predicted_value))
                confidence_intervals.append((
                    predicted_value - confidence_margin,
                    predicted_value + confidence_margin
                ))
            
            # Calculate model accuracy (R-squared)
            model_accuracy = r_value ** 2
            
            return PredictionResult(
                metric_id=series.metric_id,
                prediction_horizon=prediction_horizon,
                predicted_values=predicted_values,
                confidence_interval=confidence_intervals,
                model_accuracy=model_accuracy
            )
            
        except Exception as e:
            self.logger.debug(f"Prediction failed for {series.metric_id}: {e}")
            return None
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_metric_series(self, metric_id: str) -> Optional[MetricSeries]:
        """Get metric series by ID"""
        return self.metric_series.get(metric_id)
    
    def get_system_metrics(self, system: SystemType) -> List[MetricSeries]:
        """Get all metrics for a system"""
        return [series for series in self.metric_series.values() 
                if series.system == system]
    
    def get_correlations(self, metric_id: Optional[str] = None, 
                        min_strength: float = 0.3) -> List[CorrelationResult]:
        """Get correlations, optionally filtered by metric and strength"""
        correlations = []
        
        for (metric1, metric2), correlation in self.correlations.items():
            if metric_id and metric_id not in (metric1, metric2):
                continue
            
            if abs(correlation.correlation_coefficient) >= min_strength:
                correlations.append(correlation)
        
        return sorted(correlations, 
                     key=lambda c: abs(c.correlation_coefficient), 
                     reverse=True)
    
    def get_recent_anomalies(self, hours: int = 24, 
                           metric_id: Optional[str] = None) -> List[AnomalyDetection]:
        """Get recent anomalies"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_anomalies = [
            anomaly for anomaly in self.anomalies
            if anomaly.timestamp >= cutoff_time
        ]
        
        if metric_id:
            recent_anomalies = [a for a in recent_anomalies if a.metric_id == metric_id]
        
        return sorted(recent_anomalies, key=lambda a: a.timestamp, reverse=True)
    
    def get_predictions(self, metric_id: Optional[str] = None) -> List[PredictionResult]:
        """Get predictions"""
        if metric_id:
            prediction = self.predictions.get(metric_id)
            return [prediction] if prediction else []
        
        return list(self.predictions.values())
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        # System health overview
        system_health = {}
        for system in SystemType:
            metrics = self.get_system_metrics(system)
            if metrics:
                avg_values = {}
                for metric in metrics:
                    if metric.data_points:
                        recent_values = metric.get_recent_values(5)
                        if recent_values:
                            avg_values[metric.metric_name] = statistics.mean(recent_values)
                
                system_health[system.value] = {
                    "metric_count": len(metrics),
                    "recent_averages": avg_values
                }
        
        # Top correlations
        top_correlations = self.get_correlations(min_strength=0.5)[:10]
        
        # Recent anomalies
        recent_anomalies = self.get_recent_anomalies(hours=6)
        
        # Active predictions
        active_predictions = len(self.predictions)
        
        return {
            "analytics_statistics": self.analytics_stats.copy(),
            "system_health": system_health,
            "total_metrics": len(self.metric_series),
            "top_correlations": [
                {
                    "metric1": corr.metric1_id,
                    "metric2": corr.metric2_id,
                    "coefficient": corr.correlation_coefficient,
                    "type": corr.correlation_type.value
                }
                for corr in top_correlations
            ],
            "recent_anomalies_count": len(recent_anomalies),
            "critical_anomalies": [
                {
                    "metric": anomaly.metric_id,
                    "timestamp": anomaly.timestamp.isoformat(),
                    "value": anomaly.value,
                    "expected": anomaly.expected_value,
                    "confidence": anomaly.confidence
                }
                for anomaly in recent_anomalies
                if anomaly.confidence > 0.8
            ][:5],
            "active_predictions": active_predictions,
            "engine_status": "running" if self.is_running else "stopped"
        }
    
    def get_metric_insights(self, metric_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a specific metric"""
        series = self.get_metric_series(metric_id)
        if not series:
            return {"error": "Metric not found"}
        
        # Basic statistics
        insights = {
            "metric_id": metric_id,
            "system": series.system.value,
            "metric_type": series.metric_type.value,
            "data_points": len(series.data_points),
            "statistics": {
                "min": series.min_value,
                "max": series.max_value,
                "mean": series.mean_value,
                "std_dev": series.std_deviation
            },
            "trend": {
                "direction": series.trend_direction.value if series.trend_direction else None,
                "strength": series.trend_strength
            }
        }
        
        # Correlations involving this metric
        correlations = self.get_correlations(metric_id=metric_id, min_strength=0.3)
        insights["correlations"] = [
            {
                "related_metric": corr.metric2_id if corr.metric1_id == metric_id else corr.metric1_id,
                "coefficient": corr.correlation_coefficient,
                "type": corr.correlation_type.value
            }
            for corr in correlations[:5]
        ]
        
        # Recent anomalies
        anomalies = self.get_recent_anomalies(hours=24, metric_id=metric_id)
        insights["recent_anomalies"] = len(anomalies)
        insights["anomaly_details"] = [
            {
                "timestamp": anomaly.timestamp.isoformat(),
                "value": anomaly.value,
                "expected": anomaly.expected_value,
                "deviation": anomaly.deviation_score,
                "type": anomaly.anomaly_type
            }
            for anomaly in anomalies[:3]
        ]
        
        # Predictions
        predictions = self.get_predictions(metric_id=metric_id)
        if predictions:
            prediction = predictions[0]
            insights["prediction"] = {
                "horizon_hours": prediction.prediction_horizon.total_seconds() / 3600,
                "model_accuracy": prediction.model_accuracy,
                "next_values": prediction.predicted_values[:5]  # Next 5 predictions
            }
        
        return insights


# ============================================================================
# GLOBAL ANALYTICS ENGINE INSTANCE
# ============================================================================

# Global instance for cross-system analytics
cross_system_analytics = CrossSystemAnalyticsEngine()

# Export for external use
__all__ = [
    'MetricType',
    'CorrelationType',
    'TrendDirection',
    'MetricDataPoint',
    'MetricSeries',
    'CorrelationResult',
    'AnomalyDetection',
    'PredictionResult',
    'CrossSystemAnalyticsEngine',
    'cross_system_analytics'
]