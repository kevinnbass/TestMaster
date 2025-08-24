"""
Telemetry Observability Engine (Part 2/3) - TestMaster Advanced ML
Advanced observability and monitoring with ML-driven insights
Extracted from analytics_telemetry.py (680 lines) → 3 coordinated ML modules
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import json
import os
import psutil

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

from .telemetry_ml_collector import MLTelemetryEvent, MLSpan, MLMetricPoint, TelemetryLevel


@dataclass
class ObservabilityAlert:
    """ML-generated observability alert"""
    
    alert_id: str
    severity: str  # low, medium, high, critical
    title: str
    description: str
    component: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    prediction_horizon_minutes: int = 0
    confidence: float = 0.0
    suggested_actions: List[str] = field(default_factory=list)
    ml_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceInsight:
    """ML-driven performance insight"""
    
    insight_id: str
    category: str  # bottleneck, trend, anomaly, prediction
    title: str
    description: str
    impact_score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    components_affected: List[str]
    metrics_involved: List[str]
    timestamp: datetime
    recommendations: List[str] = field(default_factory=list)
    data_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthScore:
    """Comprehensive system health assessment"""
    
    overall_score: float  # 0.0 - 1.0
    component_scores: Dict[str, float]
    metric_scores: Dict[str, float]
    trend_analysis: Dict[str, str]  # improving, stable, degrading
    risk_factors: List[str]
    timestamp: datetime
    ml_predictions: Dict[str, float] = field(default_factory=dict)


class AdvancedObservabilityEngine:
    """
    ML-driven observability and monitoring engine
    Part 2/3 of the complete telemetry system
    """
    
    def __init__(self,
                 telemetry_collector,
                 analysis_interval: int = 60,
                 alert_cooldown: int = 300,
                 enable_predictions: bool = True):
        """Initialize observability engine"""
        
        self.telemetry_collector = telemetry_collector
        self.analysis_interval = analysis_interval
        self.alert_cooldown = alert_cooldown
        self.enable_predictions = enable_predictions
        
        # ML Models for Observability
        self.performance_predictor: Optional[RandomForestRegressor] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.trend_analyzer: Optional[LinearRegression] = None
        self.bottleneck_detector: Optional[DBSCAN] = None
        
        # Feature Processing
        self.feature_scaler = StandardScaler()
        self.historical_features: deque = deque(maxlen=1000)
        
        # Observability State
        self.active_alerts: Dict[str, ObservabilityAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.performance_insights: deque = deque(maxlen=500)
        self.health_history: deque = deque(maxlen=100)
        
        # Alert Management
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {
            'cpu_usage': {'warning': 70.0, 'critical': 85.0},
            'memory_usage': {'warning': 75.0, 'critical': 90.0},
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'response_time': {'warning': 1000.0, 'critical': 2000.0}
        }
        
        # System Monitoring
        self.system_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.component_health: Dict[str, float] = {}
        
        # ML Insights
        self.ml_insights_cache: Dict[str, Any] = {}
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.observability_stats = {
            'alerts_generated': 0,
            'insights_created': 0,
            'predictions_made': 0,
            'health_assessments': 0,
            'analysis_cycles': 0
        }
        
        # Synchronization
        self.analysis_lock = RLock()
        self.alert_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and start background tasks
        self._initialize_ml_models()
        asyncio.create_task(self._observability_loop())
        asyncio.create_task(self._system_monitoring_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for observability"""
        
        try:
            # Performance prediction model
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            
            # Anomaly detection for unusual patterns
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Trend analysis for metric predictions
            self.trend_analyzer = Ridge(alpha=0.1)
            
            # Bottleneck detection clustering
            self.bottleneck_detector = DBSCAN(eps=0.3, min_samples=5)
            
            self.logger.info("Observability ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Observability ML model initialization failed: {e}")
    
    async def _observability_loop(self):
        """Main observability analysis loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.analysis_interval)
                
                # Perform comprehensive analysis
                await self._analyze_system_health()
                await self._detect_performance_issues()
                await self._generate_ml_insights()
                await self._update_predictions()
                
                # Generate alerts if needed
                await self._evaluate_alert_conditions()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                self.observability_stats['analysis_cycles'] += 1
                
            except Exception as e:
                self.logger.error(f"Observability loop error: {e}")
                await asyncio.sleep(5)
    
    async def _system_monitoring_loop(self):
        """System resource monitoring loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Store metrics
                self.system_metrics['cpu_usage'].append(cpu_percent)
                self.system_metrics['memory_usage'].append(memory.percent)
                self.system_metrics['disk_usage'].append(disk.percent)
                
                # Network I/O if available
                try:
                    net_io = psutil.net_io_counters()
                    self.system_metrics['network_bytes_sent'].append(net_io.bytes_sent)
                    self.system_metrics['network_bytes_recv'].append(net_io.bytes_recv)
                except:
                    pass
                
                # Process telemetry metrics
                await self._process_telemetry_metrics()
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _analyze_system_health(self):
        """Comprehensive system health analysis with ML"""
        
        try:
            with self.analysis_lock:
                # Collect recent events and spans
                recent_events = list(self.telemetry_collector.events)[-100:]
                recent_spans = list(self.telemetry_collector.completed_spans)[-50:]
                
                # Calculate component health scores
                component_scores = await self._calculate_component_health(recent_events, recent_spans)
                
                # Calculate metric health scores  
                metric_scores = await self._calculate_metric_health()
                
                # Analyze trends
                trend_analysis = await self._analyze_health_trends()
                
                # Identify risk factors
                risk_factors = await self._identify_risk_factors(recent_events)
                
                # ML-driven predictions
                ml_predictions = {}
                if self.enable_predictions and len(self.historical_features) > 20:
                    ml_predictions = await self._predict_health_metrics()
                
                # Calculate overall health score
                overall_score = self._calculate_overall_health_score(
                    component_scores, metric_scores, trend_analysis
                )
                
                # Create health assessment
                health_score = SystemHealthScore(
                    overall_score=overall_score,
                    component_scores=component_scores,
                    metric_scores=metric_scores,
                    trend_analysis=trend_analysis,
                    risk_factors=risk_factors,
                    timestamp=datetime.now(),
                    ml_predictions=ml_predictions
                )
                
                self.health_history.append(health_score)
                self.observability_stats['health_assessments'] += 1
                
        except Exception as e:
            self.logger.error(f"Health analysis error: {e}")
    
    async def _calculate_component_health(self, events: List[MLTelemetryEvent], 
                                        spans: List[MLSpan]) -> Dict[str, float]:
        """Calculate health scores for each component"""
        
        component_scores = {}
        
        # Group events by component
        component_events = defaultdict(list)
        for event in events:
            component_events[event.component].append(event)
        
        # Group spans by component
        component_spans = defaultdict(list)
        for span in spans:
            component_spans[span.component].append(span)
        
        # Calculate scores for each component
        for component in set(list(component_events.keys()) + list(component_spans.keys())):
            score = 1.0  # Start with perfect score
            
            # Error rate impact
            component_event_list = component_events.get(component, [])
            if component_event_list:
                error_events = [e for e in component_event_list 
                               if e.level in [TelemetryLevel.ERROR, TelemetryLevel.CRITICAL]]
                error_rate = len(error_events) / len(component_event_list)
                score *= (1.0 - error_rate)
            
            # Performance impact from spans
            component_span_list = component_spans.get(component, [])
            if component_span_list:
                # Average performance score from spans
                perf_scores = [s.performance_score for s in component_span_list 
                              if s.performance_score is not None]
                if perf_scores:
                    avg_perf = sum(perf_scores) / len(perf_scores)
                    score *= avg_perf
            
            # ML anomaly impact
            anomaly_events = [e for e in component_event_list 
                             if e.anomaly_score and e.anomaly_score < -0.5]
            if anomaly_events:
                anomaly_penalty = min(0.3, len(anomaly_events) * 0.05)
                score *= (1.0 - anomaly_penalty)
            
            component_scores[component] = max(0.0, min(1.0, score))
        
        return component_scores
    
    async def _calculate_metric_health(self) -> Dict[str, float]:
        """Calculate health scores for metrics"""
        
        metric_scores = {}
        
        for metric_name, metric_points in self.telemetry_collector.metrics.items():
            if not metric_points:
                continue
            
            recent_points = list(metric_points)[-20:]  # Last 20 points
            values = [p.value for p in recent_points]
            
            if len(values) < 3:
                metric_scores[metric_name] = 1.0
                continue
            
            score = 1.0
            
            # Volatility check
            volatility = np.std(values) / (np.mean(values) + 1e-6)
            if volatility > 0.5:
                score *= 0.8  # High volatility penalty
            
            # Anomaly detection
            anomaly_points = [p for p in recent_points if p.anomaly_detected]
            if anomaly_points:
                anomaly_penalty = min(0.4, len(anomaly_points) * 0.1)
                score *= (1.0 - anomaly_penalty)
            
            # Trend analysis
            if len(values) >= 5:
                trend_score = await self._analyze_metric_trend(values)
                score *= trend_score
            
            metric_scores[metric_name] = max(0.0, min(1.0, score))
        
        return metric_scores
    
    async def _detect_performance_issues(self):
        """ML-driven performance issue detection"""
        
        try:
            # Analyze recent spans for performance issues
            recent_spans = list(self.telemetry_collector.completed_spans)[-100:]
            
            if len(recent_spans) < 10:
                return
            
            # Extract performance features
            performance_features = []
            for span in recent_spans:
                if span.duration_ms is not None:
                    features = [
                        span.duration_ms,
                        len(span.events),
                        len(span.attributes),
                        1.0 if span.status == 'error' else 0.0,
                        hash(span.component) % 100,
                        hash(span.operation_name) % 100
                    ]
                    performance_features.append(features)
            
            if len(performance_features) < 10:
                return
            
            # ML-based bottleneck detection
            if self.bottleneck_detector:
                try:
                    clusters = self.bottleneck_detector.fit_predict(performance_features)
                    
                    # Identify problematic clusters
                    cluster_performance = defaultdict(list)
                    for i, cluster in enumerate(clusters):
                        if cluster != -1:  # Not noise
                            cluster_performance[cluster].append(recent_spans[i])
                    
                    # Find slow clusters
                    for cluster_id, cluster_spans in cluster_performance.items():
                        avg_duration = sum(s.duration_ms for s in cluster_spans) / len(cluster_spans)
                        
                        if avg_duration > 1000:  # Slow cluster (>1 second)
                            await self._create_performance_insight(
                                "bottleneck",
                                f"Performance bottleneck detected in cluster {cluster_id}",
                                f"Average duration: {avg_duration:.1f}ms",
                                0.8,
                                [s.component for s in cluster_spans],
                                ["duration", "performance"]
                            )
                
                except Exception as e:
                    self.logger.error(f"Bottleneck detection error: {e}")
            
        except Exception as e:
            self.logger.error(f"Performance issue detection error: {e}")
    
    async def _create_performance_insight(self, category: str, title: str, 
                                        description: str, impact_score: float,
                                        components: List[str], metrics: List[str]):
        """Create a performance insight"""
        
        insight = PerformanceInsight(
            insight_id=f"insight_{int(time.time() * 1000)}",
            category=category,
            title=title,
            description=description,
            impact_score=impact_score,
            confidence=0.8,  # Default confidence
            components_affected=list(set(components)),
            metrics_involved=metrics,
            timestamp=datetime.now(),
            recommendations=await self._generate_recommendations(category, components)
        )
        
        self.performance_insights.append(insight)
        self.observability_stats['insights_created'] += 1
        
        self.logger.info(f"Performance insight created: {title}")
    
    async def _generate_recommendations(self, category: str, components: List[str]) -> List[str]:
        """Generate ML-driven recommendations"""
        
        recommendations = []
        
        if category == "bottleneck":
            recommendations.extend([
                "Consider implementing caching for frequently accessed operations",
                "Review database query performance and add appropriate indexes",
                "Analyze component interaction patterns for optimization opportunities",
                "Implement asynchronous processing where possible"
            ])
        elif category == "anomaly":
            recommendations.extend([
                "Investigate recent configuration changes",
                "Check for resource constraints or external dependencies",
                "Review error logs for underlying issues",
                "Consider implementing circuit breaker patterns"
            ])
        elif category == "trend":
            recommendations.extend([
                "Monitor resource usage trends closely",
                "Plan for capacity scaling if degradation continues",
                "Review performance baselines and adjust thresholds",
                "Implement proactive alerting for trend changes"
            ])
        
        # Component-specific recommendations
        for component in components:
            if 'database' in component.lower():
                recommendations.append("Optimize database connection pooling")
            elif 'api' in component.lower():
                recommendations.append("Implement API rate limiting and caching")
            elif 'cache' in component.lower():
                recommendations.append("Review cache hit rates and eviction policies")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def _evaluate_alert_conditions(self):
        """Evaluate conditions for generating alerts"""
        
        try:
            current_time = datetime.now()
            
            # System metric alerts
            for metric_name, threshold_config in self.alert_thresholds.items():
                if metric_name in self.system_metrics:
                    recent_values = list(self.system_metrics[metric_name])[-5:]
                    
                    if recent_values:
                        current_value = recent_values[-1]
                        avg_value = sum(recent_values) / len(recent_values)
                        
                        # Check critical threshold
                        if avg_value >= threshold_config['critical']:
                            await self._generate_alert(
                                "critical",
                                f"Critical {metric_name} level",
                                f"{metric_name} is at {avg_value:.1f}% (critical threshold: {threshold_config['critical']}%)",
                                "system",
                                metric_name,
                                avg_value,
                                threshold_config['critical']
                            )
                        
                        # Check warning threshold
                        elif avg_value >= threshold_config['warning']:
                            await self._generate_alert(
                                "warning",
                                f"High {metric_name} level",
                                f"{metric_name} is at {avg_value:.1f}% (warning threshold: {threshold_config['warning']}%)",
                                "system",
                                metric_name,
                                avg_value,
                                threshold_config['warning']
                            )
            
            # ML-driven predictive alerts
            if self.enable_predictions:
                await self._evaluate_predictive_alerts()
            
        except Exception as e:
            self.logger.error(f"Alert evaluation error: {e}")
    
    async def _generate_alert(self, severity: str, title: str, description: str,
                            component: str, metric_name: str, current_value: float,
                            threshold: float, prediction_horizon: int = 0):
        """Generate observability alert"""
        
        alert_key = f"{component}_{metric_name}_{severity}"
        
        # Check cooldown
        if alert_key in self.alert_cooldowns:
            if datetime.now() - self.alert_cooldowns[alert_key] < timedelta(seconds=self.alert_cooldown):
                return
        
        alert = ObservabilityAlert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            severity=severity,
            title=title,
            description=description,
            component=component,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now(),
            prediction_horizon_minutes=prediction_horizon,
            confidence=0.9,
            suggested_actions=await self._generate_alert_actions(severity, metric_name)
        )
        
        with self.alert_lock:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self.alert_cooldowns[alert_key] = datetime.now()
        
        self.observability_stats['alerts_generated'] += 1
        self.logger.warning(f"Alert generated: {title}")
    
    def get_observability_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive observability dashboard data"""
        
        current_health = self.health_history[-1] if self.health_history else None
        
        # Recent performance insights
        recent_insights = list(self.performance_insights)[-10:]
        
        # Active alerts by severity
        alerts_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            alerts_by_severity[alert.severity] += 1
        
        # System metrics summary
        system_summary = {}
        for metric_name, values in self.system_metrics.items():
            if values:
                recent_values = list(values)[-10:]
                system_summary[metric_name] = {
                    'current': recent_values[-1],
                    'average': sum(recent_values) / len(recent_values),
                    'trend': 'stable'  # Simplified
                }
        
        return {
            'system_health': {
                'overall_score': current_health.overall_score if current_health else 0.5,
                'component_scores': current_health.component_scores if current_health else {},
                'trend_analysis': current_health.trend_analysis if current_health else {}
            },
            'alerts': {
                'active_count': len(self.active_alerts),
                'by_severity': dict(alerts_by_severity),
                'recent_alerts': [
                    {
                        'severity': a.severity,
                        'title': a.title,
                        'timestamp': a.timestamp.isoformat()
                    }
                    for a in list(self.alert_history)[-5:]
                ]
            },
            'performance_insights': [
                {
                    'category': i.category,
                    'title': i.title,
                    'impact_score': i.impact_score,
                    'timestamp': i.timestamp.isoformat()
                }
                for i in recent_insights
            ],
            'system_metrics': system_summary,
            'statistics': self.observability_stats.copy()
        }
    
    async def shutdown(self):
        """Graceful shutdown of observability engine"""
        
        self.logger.info("Shutting down observability engine...")
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Observability engine shutdown complete")