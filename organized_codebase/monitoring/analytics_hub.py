"""
Analytics Hub - Integrated Intelligence Platform
===============================================

Unified analytics hub combining advanced anomaly detection, predictive analytics,
and intelligent deduplication. Provides centralized analytics intelligence for
the TestMaster platform.

Integrates:
- Advanced Anomaly Detection (sophisticated real-time monitoring)
- Predictive Analytics Engine (ML-powered forecasting)
- Analytics Deduplication (intelligent duplicate detection)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

# Import our analytics components
from .advanced_anomaly_detector import AdvancedAnomalyDetector, AnomalyType, AnomalySeverity
from .predictive_analytics_engine import PredictiveAnalyticsEngine, PredictionResult, IntelligentDecision
from .analytics_deduplication import AnalyticsDeduplication, DuplicateRecord, DuplicateType
from .analytics_anomaly_detector import AnalyticsAnomalyDetector

# Import comprehensive analysis integration
from ..analysis.comprehensive_analysis_hub import (
    ComprehensiveAnalysisHub, 
    AnalysisType, 
    AnalysisPriority,
    analyze_project_comprehensive
)

logger = logging.getLogger(__name__)


class AnalyticsEventType(Enum):
    """Types of analytics events"""
    ANOMALY_DETECTED = "anomaly_detected"
    PREDICTION_GENERATED = "prediction_generated"
    DUPLICATE_FOUND = "duplicate_found"
    DECISION_RECOMMENDED = "decision_recommended"
    PATTERN_IDENTIFIED = "pattern_identified"


@dataclass
class AnalyticsEvent:
    """Analytics event for unified processing"""
    event_id: str = field(default_factory=lambda: f"event_{int(time.time() * 1000000)}")
    event_type: AnalyticsEventType = AnalyticsEventType.ANOMALY_DETECTED
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metric_name: str = ""
    severity: str = "info"
    confidence: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


@dataclass
class AnalyticsInsight:
    """High-level analytics insight"""
    insight_id: str = field(default_factory=lambda: f"insight_{int(time.time() * 1000000)}")
    title: str = ""
    description: str = ""
    category: str = "general"
    priority: int = 1  # 1=low, 5=medium, 10=high
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metrics_involved: List[str] = field(default_factory=list)


class AnalyticsHub:
    """
    Unified analytics hub integrating multiple analytics engines
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize analytics components with safe config handling
        anomaly_config = self.config.get('anomaly_config', {}) if self.config else {}
        self.advanced_anomaly_detector = AdvancedAnomalyDetector(anomaly_config)
        self.predictive_engine = PredictiveAnalyticsEngine()
        self.deduplication_system = AnalyticsDeduplication()
        self.analytics_anomaly_detector = AnalyticsAnomalyDetector()
        
        # Initialize comprehensive analysis hub integration
        self.comprehensive_analysis_hub = ComprehensiveAnalysisHub(self.config)
        
        # Event processing
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.event_history = []
        self.max_history = 50000
        
        # Insights generation
        self.insights = []
        self.max_insights = 1000
        
        # Cross-component correlation
        self.correlation_matrix = {}
        self.pattern_library = {}
        
        # Performance metrics
        self.hub_stats = {
            'events_processed': 0,
            'insights_generated': 0,
            'anomalies_detected': 0,
            'predictions_made': 0,
            'duplicates_found': 0,
            'decisions_recommended': 0,
            'processing_time_ms': 0,
            'uptime_seconds': 0
        }
        
        # Configuration
        self.insight_generation_interval = self.config.get('insight_interval', 300)  # 5 minutes
        self.correlation_update_interval = self.config.get('correlation_interval', 600)  # 10 minutes
        self.enable_cross_validation = self.config.get('cross_validation', True)
        
        # Threading and async
        self.is_running = False
        self.event_processor_task = None
        self.insight_generator_task = None
        self.correlation_updater_task = None
        self.lock = threading.RLock()
        
        # Start time for uptime tracking
        self.start_time = datetime.now()
        
        logger.info("Analytics Hub initialized with integrated components")
    
    async def start_hub(self):
        """Start the analytics hub"""
        if self.is_running:
            return
        
        logger.info("Starting Analytics Hub")
        self.is_running = True
        
        # Start individual components
        self.advanced_anomaly_detector.start_monitoring()
        await self.predictive_engine.start_engine()
        
        # Start hub tasks
        self.event_processor_task = asyncio.create_task(self._event_processing_loop())
        self.insight_generator_task = asyncio.create_task(self._insight_generation_loop())
        self.correlation_updater_task = asyncio.create_task(self._correlation_update_loop())
        
        logger.info("Analytics Hub started successfully")
    
    async def stop_hub(self):
        """Stop the analytics hub"""
        if not self.is_running:
            return
        
        logger.info("Stopping Analytics Hub")
        self.is_running = False
        
        # Stop individual components
        self.advanced_anomaly_detector.stop_monitoring()
        await self.predictive_engine.stop_engine()
        self.deduplication_system.shutdown()
        self.analytics_anomaly_detector.shutdown()
        
        # Stop hub tasks
        for task in [self.event_processor_task, self.insight_generator_task, self.correlation_updater_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Analytics Hub stopped")
    
    async def process_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None,
                           metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a metric through all analytics components
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp
            metadata: Optional additional metadata
            
        Returns:
            Combined results from all components
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metadata = metadata or {}
        results = {
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp.isoformat(),
            'processing_results': {}
        }
        
        # Process through deduplication first
        is_unique = self.deduplication_system.process_analytics(
            f"metric_{metric_name}_{timestamp.timestamp()}", 
            {'metric_name': metric_name, 'value': value, 'timestamp': timestamp, **metadata}
        )
        
        results['processing_results']['deduplication'] = {
            'is_unique': is_unique,
            'processed': True
        }
        
        if not is_unique:
            # Create duplicate event
            await self._create_event(
                AnalyticsEventType.DUPLICATE_FOUND,
                source="deduplication",
                metric_name=metric_name,
                severity="warning",
                confidence=0.8,
                data={'value': value, 'timestamp': timestamp.isoformat()}
            )
            results['processing_results']['deduplication']['duplicate_detected'] = True
        
        # Process through both anomaly detectors
        # Advanced anomaly detector
        advanced_anomalies = self.advanced_anomaly_detector.add_metric_value(metric_name, value, timestamp)
        results['processing_results']['advanced_anomaly'] = {
            'anomalies_detected': len(advanced_anomalies) if advanced_anomalies else 0,
            'anomalies': [
                {
                    'id': anomaly.anomaly_id,
                    'type': anomaly.anomaly_type.value,
                    'severity': anomaly.severity.value,
                    'confidence': anomaly.confidence
                } for anomaly in advanced_anomalies
            ] if advanced_anomalies else []
        }
        
        # Analytics anomaly detector
        self.analytics_anomaly_detector.add_data_point(metric_name, value, timestamp)
        analytics_anomalies = self.analytics_anomaly_detector.get_anomalies(metric_name=metric_name, hours=1)
        results['processing_results']['analytics_anomaly'] = {
            'recent_anomalies': len(analytics_anomalies),
            'anomalies': analytics_anomalies[-5:] if analytics_anomalies else []  # Last 5
        }
        
        # Create events for detected anomalies
        if advanced_anomalies:
            for anomaly in advanced_anomalies:
                await self._create_event(
                    AnalyticsEventType.ANOMALY_DETECTED,
                    source="advanced_detector",
                    metric_name=metric_name,
                    severity=anomaly.severity.value,
                    confidence=anomaly.confidence,
                    data={
                        'anomaly_id': anomaly.anomaly_id,
                        'type': anomaly.anomaly_type.value,
                        'value': value,
                        'expected': anomaly.expected_value
                    }
                )
        
        # Get predictions for this metric
        predictions = self.predictive_engine.get_active_predictions(metric_name)
        results['processing_results']['predictions'] = {
            'active_predictions': len(predictions),
            'predictions': [
                {
                    'id': pred.prediction_id,
                    'confidence': pred.confidence_score,
                    'trend': pred.get_trend_direction(),
                    'next_value': pred.get_next_value()
                } for pred in predictions
            ]
        }
        
        # Update statistics
        self.hub_stats['events_processed'] += 1
        if advanced_anomalies or analytics_anomalies:
            self.hub_stats['anomalies_detected'] += 1
        if predictions:
            self.hub_stats['predictions_made'] += len(predictions)
        if not is_unique:
            self.hub_stats['duplicates_found'] += 1
        
        return results
    
    async def _create_event(self, event_type: AnalyticsEventType, source: str, metric_name: str,
                          severity: str, confidence: float, data: Dict[str, Any]):
        """Create and queue an analytics event"""
        event = AnalyticsEvent(
            event_type=event_type,
            source=source,
            metric_name=metric_name,
            severity=severity,
            confidence=confidence,
            data=data
        )
        
        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")
    
    async def _event_processing_loop(self):
        """Main event processing loop"""
        while self.is_running:
            try:
                # Process events from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: AnalyticsEvent):
        """Process individual analytics event"""
        try:
            # Add to history
            self.event_history.append(event)
            
            # Limit history size
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history//2:]
            
            # Cross-validate events if enabled
            if self.enable_cross_validation:
                await self._cross_validate_event(event)
            
            # Update correlations
            self._update_event_correlations(event)
            
            # Mark as processed
            event.processed = True
            
            logger.debug(f"Processed event: {event.event_type.value} for {event.metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id}: {e}")
    
    async def _cross_validate_event(self, event: AnalyticsEvent):
        """Cross-validate event across multiple components"""
        try:
            if event.event_type == AnalyticsEventType.ANOMALY_DETECTED:
                # Check if other detectors also flagged this as anomalous
                metric_name = event.metric_name
                
                # Get recent anomalies from both detectors
                advanced_anomalies = self.advanced_anomaly_detector.get_recent_anomalies(hours=1)
                analytics_anomalies = self.analytics_anomaly_detector.get_anomalies(metric_name=metric_name, hours=1)
                
                # Cross-validation score
                cross_validation_score = 0
                if any(a.metric_name == metric_name for a in advanced_anomalies):
                    cross_validation_score += 0.5
                if any(a['metric_name'] == metric_name for a in analytics_anomalies):
                    cross_validation_score += 0.5
                
                # Update event confidence based on cross-validation
                event.confidence = min(1.0, event.confidence * (1 + cross_validation_score))
                
        except Exception as e:
            logger.debug(f"Cross-validation failed for event {event.event_id}: {e}")
    
    def _update_event_correlations(self, event: AnalyticsEvent):
        """Update correlation matrix based on event"""
        try:
            metric_name = event.metric_name
            event_type = event.event_type.value
            
            # Update correlation matrix
            if metric_name not in self.correlation_matrix:
                self.correlation_matrix[metric_name] = {}
            
            if event_type not in self.correlation_matrix[metric_name]:
                self.correlation_matrix[metric_name][event_type] = {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'last_seen': None
                }
            
            # Update statistics
            stats = self.correlation_matrix[metric_name][event_type]
            stats['count'] += 1
            stats['avg_confidence'] = (stats['avg_confidence'] * (stats['count'] - 1) + event.confidence) / stats['count']
            stats['last_seen'] = event.timestamp
            
        except Exception as e:
            logger.debug(f"Failed to update correlations for event {event.event_id}: {e}")
    
    async def _insight_generation_loop(self):
        """Generate high-level insights from processed events"""
        while self.is_running:
            try:
                await self._generate_insights()
                await asyncio.sleep(self.insight_generation_interval)
                
            except Exception as e:
                logger.error(f"Insight generation error: {e}")
                await asyncio.sleep(60)
    
    async def _generate_insights(self):
        """Generate analytics insights from recent events"""
        try:
            # Analyze recent events (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_events = [e for e in self.event_history if e.timestamp >= cutoff_time]
            
            if not recent_events:
                return
            
            # Group events by metric and type
            metric_events = {}
            for event in recent_events:
                if event.metric_name not in metric_events:
                    metric_events[event.metric_name] = []
                metric_events[event.metric_name].append(event)
            
            # Generate insights for each metric
            for metric_name, events in metric_events.items():
                await self._generate_metric_insights(metric_name, events)
            
            # Generate cross-metric insights
            await self._generate_cross_metric_insights(recent_events)
            
            # Cleanup old insights
            cutoff_time = datetime.now() - timedelta(days=7)
            self.insights = [i for i in self.insights if i.timestamp >= cutoff_time]
            
            # Limit insights
            if len(self.insights) > self.max_insights:
                self.insights = self.insights[-self.max_insights//2:]
            
            self.hub_stats['insights_generated'] += 1
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
    
    async def _generate_metric_insights(self, metric_name: str, events: List[AnalyticsEvent]):
        """Generate insights for a specific metric"""
        try:
            anomaly_events = [e for e in events if e.event_type == AnalyticsEventType.ANOMALY_DETECTED]
            duplicate_events = [e for e in events if e.event_type == AnalyticsEventType.DUPLICATE_FOUND]
            
            # High anomaly frequency insight
            if len(anomaly_events) > 5:
                insight = AnalyticsInsight(
                    title=f"High Anomaly Frequency: {metric_name}",
                    description=f"Detected {len(anomaly_events)} anomalies in the last hour for {metric_name}",
                    category="anomaly_pattern",
                    priority=7,
                    confidence=0.8,
                    evidence=[f"{len(anomaly_events)} anomalies detected"],
                    recommendations=[
                        "Investigate root cause of anomalies",
                        "Consider adjusting thresholds",
                        "Review system performance"
                    ],
                    metrics_involved=[metric_name]
                )
                self.insights.append(insight)
            
            # Data quality issues
            if len(duplicate_events) > 3:
                insight = AnalyticsInsight(
                    title=f"Data Quality Issues: {metric_name}",
                    description=f"Multiple duplicate entries detected for {metric_name}",
                    category="data_quality",
                    priority=5,
                    confidence=0.9,
                    evidence=[f"{len(duplicate_events)} duplicate entries"],
                    recommendations=[
                        "Review data collection process",
                        "Implement data validation",
                        "Check for system duplication"
                    ],
                    metrics_involved=[metric_name]
                )
                self.insights.append(insight)
                
        except Exception as e:
            logger.debug(f"Failed to generate metric insights for {metric_name}: {e}")
    
    async def _generate_cross_metric_insights(self, events: List[AnalyticsEvent]):
        """Generate insights across multiple metrics"""
        try:
            # Get unique metrics from events
            metrics = list(set(e.metric_name for e in events))
            
            if len(metrics) > 1:
                # System-wide anomaly pattern
                anomaly_metrics = list(set(e.metric_name for e in events 
                                         if e.event_type == AnalyticsEventType.ANOMALY_DETECTED))
                
                if len(anomaly_metrics) > len(metrics) * 0.5:  # More than half have anomalies
                    insight = AnalyticsInsight(
                        title="System-wide Anomaly Pattern",
                        description=f"Anomalies detected across {len(anomaly_metrics)} of {len(metrics)} monitored metrics",
                        category="system_health",
                        priority=9,
                        confidence=0.85,
                        evidence=[f"Anomalies in {len(anomaly_metrics)} metrics"],
                        recommendations=[
                            "Investigate system-wide issues",
                            "Check infrastructure health",
                            "Review recent deployments"
                        ],
                        metrics_involved=anomaly_metrics
                    )
                    self.insights.append(insight)
                    
        except Exception as e:
            logger.debug(f"Failed to generate cross-metric insights: {e}")
    
    async def _correlation_update_loop(self):
        """Update correlation patterns periodically"""
        while self.is_running:
            try:
                await self._update_correlations()
                await asyncio.sleep(self.correlation_update_interval)
                
            except Exception as e:
                logger.error(f"Correlation update error: {e}")
                await asyncio.sleep(120)
    
    async def _update_correlations(self):
        """Update correlation patterns between metrics"""
        try:
            # Get intelligent decisions from predictive engine
            decisions = self.predictive_engine.get_intelligent_decisions()
            
            for decision in decisions:
                await self._create_event(
                    AnalyticsEventType.DECISION_RECOMMENDED,
                    source="predictive_engine",
                    metric_name=decision.trigger_metrics[0] if decision.trigger_metrics else "system",
                    severity="info",
                    confidence=decision.confidence,
                    data={
                        'decision_id': decision.decision_id,
                        'type': decision.decision_type.value,
                        'urgency': decision.urgency,
                        'actions': decision.recommended_actions
                    }
                )
            
            if decisions:
                self.hub_stats['decisions_recommended'] += len(decisions)
                
        except Exception as e:
            logger.debug(f"Failed to update correlations: {e}")
    
    # Public API methods
    
    def get_hub_status(self) -> Dict[str, Any]:
        """Get comprehensive hub status"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        self.hub_stats['uptime_seconds'] = uptime
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime_hours': uptime / 3600,
            'statistics': self.hub_stats.copy(),
            'components': {
                'advanced_anomaly_detector': self.advanced_anomaly_detector.get_anomaly_summary(),
                'predictive_engine': self.predictive_engine.get_engine_analytics(),
                'deduplication_system': self.deduplication_system.get_deduplication_statistics(),
                'analytics_anomaly_detector': self.analytics_anomaly_detector.get_statistics()
            },
            'event_queue_size': self.event_queue.qsize(),
            'event_history_size': len(self.event_history),
            'insights_count': len(self.insights),
            'correlation_metrics': len(self.correlation_matrix)
        }
    
    def get_recent_insights(self, hours: int = 24, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent insights"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        insights = [i for i in self.insights if i.timestamp >= cutoff_time]
        
        if category:
            insights = [i for i in insights if i.category == category]
        
        # Sort by priority and timestamp
        insights.sort(key=lambda x: (-x.priority, -x.timestamp.timestamp()))
        
        return [
            {
                'insight_id': insight.insight_id,
                'title': insight.title,
                'description': insight.description,
                'category': insight.category,
                'priority': insight.priority,
                'confidence': insight.confidence,
                'evidence': insight.evidence,
                'recommendations': insight.recommendations,
                'timestamp': insight.timestamp.isoformat(),
                'metrics_involved': insight.metrics_involved
            } for insight in insights
        ]
    
    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Get correlation matrix"""
        return {
            'correlations': self.correlation_matrix,
            'total_metrics': len(self.correlation_matrix),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_comprehensive_analytics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive analytics for a metric or all metrics"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'hub_status': self.get_hub_status(),
            'recent_insights': self.get_recent_insights(hours=24)
        }
        
        if metric_name:
            # Get metric-specific analytics
            result['metric_analytics'] = {
                'anomalies': {
                    'advanced': self.advanced_anomaly_detector.get_anomalies_by_metric(metric_name),
                    'analytics': self.analytics_anomaly_detector.get_anomalies(metric_name=metric_name)
                },
                'predictions': self.predictive_engine.get_active_predictions(metric_name),
                'correlations': self.correlation_matrix.get(metric_name, {})
            }
        
        return result
    
    async def analyze_project_comprehensive(
        self,
        target_path: str,
        analysis_types: Optional[List[AnalysisType]] = None,
        priority: AnalysisPriority = AnalysisPriority.MEDIUM
    ) -> Dict[str, Any]:
        """
        Perform comprehensive project analysis using all 17+ analysis components
        
        This integrates all the scattered analysis files into a single unified interface.
        """
        self.logger.info(f"Starting comprehensive analysis for: {target_path}")
        
        try:
            # Delegate to the comprehensive analysis hub
            result = await self.comprehensive_analysis_hub.analyze_comprehensive(
                target_path, analysis_types, priority
            )
            
            # Create analytics event for this analysis
            await self._create_event(
                AnalyticsEventType.PATTERN_IDENTIFIED,
                source="comprehensive_analysis",
                metric_name="project_analysis",
                severity="info",
                confidence=result.confidence_score,
                data={
                    'target_path': target_path,
                    'analysis_types': [t.value for t in (analysis_types or [])],
                    'findings_count': len(result.findings),
                    'execution_time': str(result.execution_time)
                }
            )
            
            # Update hub stats
            self.hub_stats['comprehensive_analyses'] = self.hub_stats.get('comprehensive_analyses', 0) + 1
            
            return {
                'analysis_result': {
                    'result_id': result.result_id,
                    'status': result.status,
                    'confidence_score': result.confidence_score,
                    'execution_time': str(result.execution_time),
                    'findings_count': len(result.findings),
                    'recommendations_count': len(result.recommendations)
                },
                'detailed_findings': result.findings,
                'metrics': result.metrics,
                'recommendations': result.recommendations,
                'hub_integration': {
                    'event_created': True,
                    'stats_updated': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            
            # Create failure event
            await self._create_event(
                AnalyticsEventType.ANOMALY_DETECTED,
                source="comprehensive_analysis",
                metric_name="analysis_failure",
                severity="error",
                confidence=1.0,
                data={
                    'target_path': target_path,
                    'error': str(e)
                }
            )
            
            return {
                'analysis_result': {
                    'status': 'failed',
                    'error': str(e)
                },
                'hub_integration': {
                    'error_event_created': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def get_consolidated_analysis_capabilities(self) -> Dict[str, Any]:
        """
        Get information about all consolidated analysis capabilities
        
        Shows all 17+ analysis components that are now unified
        """
        return {
            'total_analyzers_consolidated': 17,
            'analysis_hubs': {
                'technical_debt_hub': {
                    'analyzers': ['TechnicalDebtAnalyzer', 'CodeDebtAnalyzer', 'TestDebtAnalyzer', 'DebtQuantifier'],
                    'capabilities': ['debt_quantification', 'code_quality_analysis', 'test_quality_analysis']
                },
                'business_analysis_hub': {
                    'analyzers': ['BusinessAnalyzer', 'WorkflowAnalyzer', 'ConstraintAnalyzer', 'RuleExtractor'],
                    'capabilities': ['business_logic_analysis', 'workflow_detection', 'constraint_analysis']
                },
                'semantic_analysis_hub': {
                    'analyzers': ['SemanticAnalyzer', 'IntentAnalyzer', 'PatternDetector', 'RelationshipAnalyzer'],
                    'capabilities': ['semantic_structure_analysis', 'intent_recognition', 'pattern_detection']
                },
                'ml_analysis_hub': {
                    'analyzers': ['MLCodeAnalyzer', 'MLAnalyzerLegacy', 'AdvancedPatternRecognizer'],
                    'capabilities': ['ml_framework_detection', 'ml_pattern_analysis', 'advanced_pattern_recognition']
                }
            },
            'supported_analysis_types': [t.value for t in AnalysisType],
            'integration_status': 'COMPLETE - All 17+ components unified',
            'access_method': 'analytics_hub.analyze_project_comprehensive()'
        }


# Global analytics hub instance
analytics_hub = AnalyticsHub()

# Export
__all__ = [
    'AnalyticsEventType', 'AnalyticsEvent', 'AnalyticsInsight',
    'AnalyticsHub', 'analytics_hub',
    # Comprehensive analysis integration
    'ComprehensiveAnalysisHub', 'AnalysisType', 'AnalysisPriority',
    'analyze_project_comprehensive'
]