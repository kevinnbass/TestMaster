"""
Telemetry ML Collector (Part 1/3) - TestMaster Advanced ML
Core ML-driven telemetry collection with intelligent sampling and analysis
Extracted from analytics_telemetry.py (680 lines) → 3 coordinated ML modules
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class TelemetryLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TraceType(Enum):
    REQUEST = "request"
    OPERATION = "operation"
    COMPONENT = "component"
    SYSTEM = "system"
    USER = "user"


@dataclass
class MLTelemetryEvent:
    """ML-enhanced telemetry event with intelligent attributes"""
    
    event_id: str
    timestamp: datetime
    level: TelemetryLevel
    component: str
    operation: str
    message: str
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # ML Enhancement Fields
    anomaly_score: Optional[float] = None
    pattern_cluster: Optional[int] = None
    prediction_confidence: Optional[float] = None
    ml_insights: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLSpan:
    """ML-enhanced distributed tracing span"""
    
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    component: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "active"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    
    # ML Enhancement Fields
    performance_score: Optional[float] = None
    bottleneck_probability: Optional[float] = None
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class MLMetricPoint:
    """ML-enhanced metric point with predictive capabilities"""
    
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    metric_type: str = "gauge"
    
    # ML Enhancement Fields
    trend_direction: Optional[str] = None  # increasing, decreasing, stable
    anomaly_detected: bool = False
    predicted_next_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class AdvancedTelemetryMLCollector:
    """
    ML-enhanced telemetry collector with intelligent sampling and analysis
    Part 1/3 of the complete telemetry system
    """
    
    def __init__(self,
                 service_name: str = "analytics_ml_system",
                 export_interval: int = 30,
                 max_events: int = 10000,
                 enable_ml_analysis: bool = True,
                 sampling_rate: float = 1.0):
        """Initialize ML-enhanced telemetry collector"""
        
        self.service_name = service_name
        self.export_interval = export_interval
        self.max_events = max_events
        self.enable_ml_analysis = enable_ml_analysis
        self.sampling_rate = sampling_rate
        
        # ML Models for Telemetry Intelligence
        self.anomaly_detector: Optional[IsolationForest] = None
        self.pattern_clusterer: Optional[KMeans] = None
        self.performance_classifier: Optional[RandomForestClassifier] = None
        self.trend_analyzer: Optional[LogisticRegression] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.ml_feature_history: deque = deque(maxlen=1000)
        
        # Telemetry Storage
        self.events: deque = deque(maxlen=max_events)
        self.active_spans: Dict[str, MLSpan] = {}
        self.completed_spans: deque = deque(maxlen=max_events)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # ML Analysis Results
        self.ml_insights: Dict[str, Any] = {}
        self.anomaly_patterns: List[Dict[str, Any]] = []
        self.performance_predictions: Dict[str, Dict[str, Any]] = {}
        
        # System State
        self.telemetry_active = True
        self.current_context = type('Context', (), {})()
        
        # Statistics and Monitoring
        self.telemetry_stats = {
            'start_time': datetime.now(),
            'events_collected': 0,
            'spans_created': 0,
            'spans_completed': 0,
            'metrics_recorded': 0,
            'ml_analyses_performed': 0,
            'anomalies_detected': 0,
            'predictions_made': 0
        }
        
        # Synchronization
        self.collection_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        # Initialize ML components
        if enable_ml_analysis:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_analysis_loop())
        
        self.logger = logging.getLogger(__name__)
        asyncio.create_task(self._collection_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for telemetry intelligence"""
        
        try:
            # Anomaly Detection for unusual telemetry patterns
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Pattern Clustering for telemetry event grouping
            self.pattern_clusterer = KMeans(
                n_clusters=8,
                random_state=42,
                n_init=10
            )
            
            # Performance Classification for span analysis
            self.performance_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Trend Analysis for metric predictions
            self.trend_analyzer = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"ML model initialization failed: {e}")
            self.enable_ml_analysis = False
    
    async def collect_event(self,
                          level: TelemetryLevel,
                          component: str,
                          operation: str,
                          message: str,
                          duration_ms: Optional[float] = None,
                          attributes: Optional[Dict[str, Any]] = None,
                          metrics: Optional[Dict[str, float]] = None,
                          error: Optional[str] = None) -> Optional[str]:
        """Collect telemetry event with ML enhancement"""
        
        # Intelligent sampling based on ML analysis
        if not await self._ml_should_sample(level, component, operation):
            return None
        
        event_id = str(uuid.uuid4())
        
        # Get current trace context
        trace_id = getattr(self.current_context, 'trace_id', None)
        span_id = getattr(self.current_context, 'span_id', None)
        
        event = MLTelemetryEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            level=level,
            component=component,
            operation=operation,
            message=message,
            duration_ms=duration_ms,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {},
            metrics=metrics or {},
            error=error
        )
        
        # ML Enhancement
        if self.enable_ml_analysis:
            await self._enhance_event_with_ml(event)
        
        with self.collection_lock:
            self.events.append(event)
            self.telemetry_stats['events_collected'] += 1
        
        # Add to active span if exists
        if span_id and span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.events.append({
                'timestamp': event.timestamp.isoformat(),
                'level': level.value,
                'message': message,
                'attributes': attributes or {},
                'ml_insights': event.ml_insights
            })
        
        return event_id
    
    async def start_span(self,
                        operation_name: str,
                        component: str,
                        parent_span_id: Optional[str] = None,
                        attributes: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start ML-enhanced distributed tracing span"""
        
        span_id = str(uuid.uuid4())
        
        # Get or create trace ID
        trace_id = getattr(self.current_context, 'trace_id', None)
        if not trace_id:
            trace_id = str(uuid.uuid4())
            self.current_context.trace_id = trace_id
        
        # Use current span as parent if not specified
        if not parent_span_id:
            parent_span_id = getattr(self.current_context, 'span_id', None)
        
        span = MLSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            component=component,
            start_time=datetime.now(),
            attributes=attributes or {}
        )
        
        with self.collection_lock:
            self.active_spans[span_id] = span
            self.current_context.span_id = span_id
            self.telemetry_stats['spans_created'] += 1
        
        return span_id
    
    async def end_span(self,
                      span_id: str,
                      status: str = "ok",
                      attributes: Optional[Dict[str, Any]] = None):
        """End span with ML performance analysis"""
        
        if span_id not in self.active_spans:
            return
        
        with self.collection_lock:
            span = self.active_spans.pop(span_id)
            span.end_time = datetime.now()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
            span.status = status
            
            if attributes:
                span.attributes.update(attributes)
            
            # ML Enhancement for span analysis
            if self.enable_ml_analysis:
                await self._enhance_span_with_ml(span)
            
            self.completed_spans.append(span)
            self.telemetry_stats['spans_completed'] += 1
        
        # Update context to parent span
        if span.parent_span_id and span.parent_span_id in self.active_spans:
            self.current_context.span_id = span.parent_span_id
        else:
            if hasattr(self.current_context, 'span_id'):
                delattr(self.current_context, 'span_id')
    
    async def record_metric(self,
                           name: str,
                           value: float,
                           labels: Optional[Dict[str, str]] = None,
                           unit: str = "",
                           metric_type: str = "gauge") -> None:
        """Record ML-enhanced metric point"""
        
        metric_point = MLMetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit,
            metric_type=metric_type
        )
        
        # ML Enhancement for metric analysis
        if self.enable_ml_analysis:
            await self._enhance_metric_with_ml(metric_point, name)
        
        with self.collection_lock:
            self.metrics[name].append(metric_point)
            self.telemetry_stats['metrics_recorded'] += 1
    
    async def _ml_should_sample(self, level: TelemetryLevel, component: str, operation: str) -> bool:
        """ML-driven intelligent sampling decision"""
        
        if not self.enable_ml_analysis:
            return np.random.random() < self.sampling_rate
        
        # Higher sampling for errors and critical events
        if level in [TelemetryLevel.ERROR, TelemetryLevel.CRITICAL]:
            return True
        
        # ML-based adaptive sampling based on historical patterns
        feature_vector = self._extract_sampling_features(level, component, operation)
        
        if len(self.ml_feature_history) > 50:
            # Use ML model to predict sampling importance
            try:
                importance_score = await self._predict_event_importance(feature_vector)
                adaptive_rate = max(0.1, min(1.0, self.sampling_rate * (1 + importance_score)))
                return np.random.random() < adaptive_rate
            except:
                pass
        
        return np.random.random() < self.sampling_rate
    
    def _extract_sampling_features(self, level: TelemetryLevel, component: str, operation: str) -> np.ndarray:
        """Extract features for ML sampling decision"""
        
        features = [
            # Level encoding
            ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(level.value),
            # Component hash (simplified)
            hash(component) % 1000 / 1000.0,
            # Operation hash (simplified)
            hash(operation) % 1000 / 1000.0,
            # Time of day factor
            datetime.now().hour / 24.0,
            # Recent event frequency
            len([e for e in list(self.events)[-100:] if e.component == component]) / 100.0
        ]
        
        return np.array(features)
    
    async def _enhance_event_with_ml(self, event: MLTelemetryEvent):
        """Enhance event with ML analysis"""
        
        if not self.enable_ml_analysis or len(self.events) < 10:
            return
        
        try:
            with self.ml_lock:
                # Extract features for anomaly detection
                event_features = self._extract_event_features(event)
                
                if self.anomaly_detector and len(self.ml_feature_history) > 50:
                    # Anomaly detection
                    anomaly_score = self.anomaly_detector.decision_function([event_features])[0]
                    event.anomaly_score = float(anomaly_score)
                    
                    if anomaly_score < -0.5:  # Threshold for anomaly
                        event.ml_insights['anomaly_detected'] = True
                        self.telemetry_stats['anomalies_detected'] += 1
                
                # Pattern clustering
                if self.pattern_clusterer and len(self.ml_feature_history) > 20:
                    cluster = self.pattern_clusterer.predict([event_features])[0]
                    event.pattern_cluster = int(cluster)
                    event.ml_insights['pattern_cluster'] = cluster
                
                # Add features to history
                self.ml_feature_history.append(event_features)
                self.telemetry_stats['ml_analyses_performed'] += 1
                
        except Exception as e:
            self.logger.error(f"ML event enhancement failed: {e}")
    
    def _extract_event_features(self, event: MLTelemetryEvent) -> np.ndarray:
        """Extract ML features from telemetry event"""
        
        features = [
            # Basic event properties
            ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(event.level.value),
            hash(event.component) % 1000 / 1000.0,
            hash(event.operation) % 1000 / 1000.0,
            event.duration_ms or 0.0,
            len(event.message),
            len(event.attributes),
            
            # Temporal features
            event.timestamp.hour,
            event.timestamp.minute / 60.0,
            event.timestamp.weekday(),
            
            # Context features
            1.0 if event.error else 0.0,
            len(event.metrics) if event.metrics else 0.0
        ]
        
        return np.array(features, dtype=np.float64)
    
    async def _collection_loop(self):
        """Background collection and maintenance loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Run every 10 seconds
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Update ML models if enough data
                if self.enable_ml_analysis and len(self.ml_feature_history) > 100:
                    await self._update_ml_models()
                
            except Exception as e:
                self.logger.error(f"Collection loop error: {e}")
    
    async def _ml_analysis_loop(self):
        """Background ML analysis and insight generation"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute
                
                if len(self.events) > 50:
                    await self._generate_ml_insights()
                    await self._detect_anomaly_patterns()
                
            except Exception as e:
                self.logger.error(f"ML analysis loop error: {e}")
    
    def get_ml_telemetry_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML telemetry summary"""
        
        uptime = (datetime.now() - self.telemetry_stats['start_time']).total_seconds()
        
        # Recent activity analysis
        recent_events = [e for e in list(self.events)
                        if (datetime.now() - e.timestamp).total_seconds() < 300]
        
        anomaly_events = [e for e in recent_events if e.anomaly_score and e.anomaly_score < -0.5]
        
        return {
            'service_name': self.service_name,
            'collection_status': {
                'active': self.telemetry_active,
                'uptime_seconds': uptime,
                'ml_analysis_enabled': self.enable_ml_analysis
            },
            'statistics': self.telemetry_stats.copy(),
            'ml_insights': {
                'total_insights': len(self.ml_insights),
                'anomaly_patterns': len(self.anomaly_patterns),
                'recent_anomalies': len(anomaly_events),
                'prediction_accuracy': self._calculate_prediction_accuracy()
            },
            'current_state': {
                'events_in_buffer': len(self.events),
                'active_spans': len(self.active_spans),
                'metrics_tracked': len(self.metrics),
                'ml_feature_history_size': len(self.ml_feature_history)
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of telemetry collector"""
        
        self.logger.info("Shutting down ML telemetry collector...")
        self.shutdown_event.set()
        self.telemetry_active = False
        await asyncio.sleep(1)
        self.logger.info("ML telemetry collector shutdown complete")