"""
Unified Quality Assurance System - Agent C Phase 2 Enhancement
Complete Intelligence-Amplified QA Framework
Hours 104-105: System Integration and Intelligence Synthesis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from .data_models import (
    QualityReport, QualityScore, Alert, AlertType, ScoreCategory,
    QualityLevel, BenchmarkResult, ValidationResult, BenchmarkType, ValidationType
)
from .quality_monitor import AdaptiveQualityMonitor, AdaptiveQualityConfig
from .scoring_system import IntelligentScoringSystem, ScoringWeight


@dataclass
class QASystemConfig:
    """Configuration for unified QA system."""
    enable_adaptive_monitoring: bool = True
    enable_intelligent_scoring: bool = True
    enable_predictive_analysis: bool = True
    enable_auto_remediation: bool = True
    monitoring_interval: int = 60
    alert_aggregation_window: int = 300
    learning_rate: float = 0.1
    performance_cache_size: int = 1000


class UnifiedQASystem:
    """
    Unified Quality Assurance System with Intelligence Amplification
    Agent C Phase 2 Enhancement: Hours 104-105
    
    Complete QA framework combining adaptive monitoring, intelligent scoring,
    predictive analysis, and automated remediation capabilities.
    """
    
    def __init__(self, config: Optional[QASystemConfig] = None):
        """Initialize unified QA system."""
        self.config = config or QASystemConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.adaptive_monitor = None
        self.intelligent_scorer = None
        
        # System state
        self.active_targets = set()
        self.qa_reports_history = {}
        self.alert_aggregator = {}
        self.performance_metrics = {}
        self.remediation_actions = {}
        
        # Intelligence features
        self.pattern_detector = {}
        self.trend_analyzer = {}
        self.anomaly_detector = {}
        self.recommendation_engine = {}
        
        # Integration hooks
        self.external_integrations = {}
        self.notification_handlers = []
        self.custom_validators = {}
        
    async def initialize(self) -> bool:
        """Initialize the unified QA system."""
        try:
            self.logger.info("Initializing Unified QA System...")
            
            # Initialize adaptive monitoring
            if self.config.enable_adaptive_monitoring:
                monitor_config = AdaptiveQualityConfig(
                    learning_rate=self.config.learning_rate,
                    auto_adjust_thresholds=True,
                    predictive_monitoring=self.config.enable_predictive_analysis
                )
                self.adaptive_monitor = AdaptiveQualityMonitor(monitor_config)
                self.adaptive_monitor.register_alert_callback(self._handle_quality_alert)
                self.logger.info("Adaptive monitoring initialized")
            
            # Initialize intelligent scoring
            if self.config.enable_intelligent_scoring:
                self.intelligent_scorer = IntelligentScoringSystem()
                self.logger.info("Intelligent scoring system initialized")
            
            # Initialize intelligence features
            await self._initialize_intelligence_features()
            
            # Load previous state
            await self._load_system_state()
            
            self.logger.info("Unified QA System initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Unified QA System: {e}")
            return False
    
    async def _initialize_intelligence_features(self):
        """Initialize AI-powered intelligence features."""
        # Pattern detection for quality trends
        self.pattern_detector = {
            'degradation_patterns': [],
            'improvement_patterns': [],
            'cyclic_patterns': [],
            'anomaly_patterns': []
        }
        
        # Trend analysis for predictive insights
        self.trend_analyzer = {
            'short_term_trends': {},  # 1-hour trends
            'medium_term_trends': {},  # 24-hour trends
            'long_term_trends': {}     # 7-day trends
        }
        
        # Anomaly detection thresholds
        self.anomaly_detector = {
            'statistical_thresholds': {},
            'ml_models': {},
            'pattern_based_detection': {}
        }
        
        # Recommendation engine
        self.recommendation_engine = {
            'improvement_suggestions': {},
            'optimization_recommendations': {},
            'preventive_actions': {},
            'best_practices': {}
        }
    
    async def register_target(self, target_id: str, 
                            target_config: Optional[Dict[str, Any]] = None) -> bool:
        """Register a target for quality monitoring."""
        try:
            self.active_targets.add(target_id)
            
            # Initialize target-specific tracking
            self.qa_reports_history[target_id] = []
            self.performance_metrics[target_id] = {}
            self.alert_aggregator[target_id] = []
            
            # Configure adaptive monitoring
            if self.adaptive_monitor:
                self.adaptive_monitor.start_monitoring([target_id])
            
            # Initialize target-specific intelligence
            await self._initialize_target_intelligence(target_id, target_config)
            
            self.logger.info(f"Target {target_id} registered for QA monitoring")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register target {target_id}: {e}")
            return False
    
    async def _initialize_target_intelligence(self, target_id: str, 
                                           config: Optional[Dict[str, Any]]):
        """Initialize intelligence features for specific target."""
        # Initialize trend tracking
        for trend_type in self.trend_analyzer:
            self.trend_analyzer[trend_type][target_id] = {
                'data_points': [],
                'trend_direction': 'stable',
                'trend_strength': 0.0,
                'last_analysis': None
            }
        
        # Initialize anomaly detection
        self.anomaly_detector['statistical_thresholds'][target_id] = {
            'mean': 0.0,
            'std': 1.0,
            'outlier_threshold': 2.0
        }
        
        # Initialize recommendations
        self.recommendation_engine['improvement_suggestions'][target_id] = []
        self.recommendation_engine['optimization_recommendations'][target_id] = []
    
    async def run_comprehensive_qa_assessment(self, target_id: str,
                                            measurements: Optional[Dict[str, Any]] = None) -> QualityReport:
        """Run comprehensive QA assessment for target."""
        try:
            if target_id not in self.active_targets:
                raise ValueError(f"Target {target_id} not registered")
            
            # Gather measurements
            if measurements is None:
                measurements = await self._gather_measurements(target_id)
            
            # Intelligent scoring
            category_scores = {}
            if self.intelligent_scorer:
                for category in ScoreCategory:
                    score = self.intelligent_scorer.calculate_comprehensive_score(
                        measurements, context=target_id
                    )
                    category_scores[category] = score
            
            # Generate quality report
            quality_report = await self._generate_comprehensive_report(
                target_id, measurements, category_scores
            )
            
            # Record in history
            self.qa_reports_history[target_id].append(quality_report)
            
            # Run intelligence analysis
            await self._run_intelligence_analysis(target_id, quality_report)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(target_id, quality_report)
            quality_report.recommendations = recommendations
            
            # Auto-remediation if enabled
            if self.config.enable_auto_remediation:
                await self._attempt_auto_remediation(target_id, quality_report)
            
            # Save system state
            await self._save_system_state()
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"QA assessment failed for {target_id}: {e}")
            raise
    
    async def _gather_measurements(self, target_id: str) -> Dict[str, Any]:
        """Gather measurements from various sources."""
        measurements = {}
        
        # Performance metrics
        if target_id in self.performance_metrics:
            measurements.update(self.performance_metrics[target_id])
        
        # Custom measurements from integrations
        for integration_name, integration in self.external_integrations.items():
            try:
                if hasattr(integration, 'get_measurements'):
                    integration_data = await integration.get_measurements(target_id)
                    measurements.update(integration_data)
            except Exception as e:
                self.logger.warning(f"Failed to get measurements from {integration_name}: {e}")
        
        # Default measurements if none available
        if not measurements:
            measurements = {
                'timestamp': datetime.now().isoformat(),
                'target_id': target_id,
                'measurement_source': 'unified_qa_system'
            }
        
        return measurements
    
    async def _generate_comprehensive_report(self, target_id: str,
                                           measurements: Dict[str, Any],
                                           category_scores: Dict[ScoreCategory, QualityScore]) -> QualityReport:
        """Generate comprehensive quality report."""
        # Calculate overall score
        if category_scores:
            overall_score = sum(score.score for score in category_scores.values()) / len(category_scores)
        else:
            overall_score = 50.0  # Default neutral score
        
        # Generate alerts
        alerts = await self._generate_alerts(target_id, overall_score, category_scores)
        
        # Generate benchmarks
        benchmarks = await self._generate_benchmarks(target_id, measurements)
        
        # Generate validations
        validations = await self._generate_validations(target_id, measurements)
        
        return QualityReport(
            overall_score=overall_score,
            category_scores=category_scores,
            alerts=alerts,
            benchmarks=benchmarks,
            validations=validations,
            recommendations=[],  # Will be filled later
            metadata={
                'target_id': target_id,
                'assessment_timestamp': datetime.now().isoformat(),
                'measurement_count': len(measurements),
                'intelligence_enhanced': True,
                'system_version': '2.0'
            }
        )
    
    async def _generate_alerts(self, target_id: str, overall_score: float,
                             category_scores: Dict[ScoreCategory, QualityScore]) -> List[Alert]:
        """Generate intelligent alerts based on scores and patterns."""
        alerts = []
        
        # Overall score alert
        if overall_score < 60:
            alerts.append(Alert(
                type=AlertType.QUALITY_DEGRADATION,
                severity="high" if overall_score < 40 else "medium",
                message=f"Overall quality score {overall_score:.1f} below acceptable threshold",
                current_value=overall_score,
                threshold=60.0
            ))
        
        # Category-specific alerts
        for category, score in category_scores.items():
            if score.score < 50:
                alerts.append(Alert(
                    type=AlertType.THRESHOLD_BREACH,
                    severity="high" if score.score < 30 else "medium",
                    message=f"{category.value} score {score.score:.1f} critically low",
                    category=category,
                    current_value=score.score,
                    threshold=50.0
                ))
        
        # Pattern-based alerts
        pattern_alerts = await self._detect_pattern_alerts(target_id)
        alerts.extend(pattern_alerts)
        
        return alerts
    
    async def _detect_pattern_alerts(self, target_id: str) -> List[Alert]:
        """Detect pattern-based alerts using intelligence."""
        alerts = []
        
        if target_id not in self.qa_reports_history:
            return alerts
        
        history = self.qa_reports_history[target_id]
        if len(history) < 3:
            return alerts
        
        # Analyze recent trend
        recent_scores = [report.overall_score for report in history[-5:]]
        
        # Declining trend detection
        if len(recent_scores) >= 3:
            trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            if trend_slope < -5:  # Declining by more than 5 points per measurement
                alerts.append(Alert(
                    type=AlertType.TREND_ANOMALY,
                    severity="warning",
                    message=f"Declining quality trend detected (slope: {trend_slope:.2f})",
                    trend=f"declining at {trend_slope:.2f} per assessment"
                ))
        
        return alerts
    
    async def _generate_benchmarks(self, target_id: str, 
                                 measurements: Dict[str, Any]) -> List[BenchmarkResult]:
        """Generate benchmark results."""
        benchmarks = []
        
        # Performance benchmarks
        if 'response_time_ms' in measurements:
            benchmarks.append(BenchmarkResult(
                type=BenchmarkType.RESPONSE_TIME,
                value=measurements['response_time_ms'],
                unit='ms',
                baseline=100.0,  # 100ms baseline
                metadata={'target_id': target_id}
            ))
        
        return benchmarks
    
    async def _generate_validations(self, target_id: str,
                                  measurements: Dict[str, Any]) -> List[ValidationResult]:
        """Generate validation results.""" 
        validations = []
        
        # Custom validations
        if target_id in self.custom_validators:
            for validator_name, validator_func in self.custom_validators[target_id].items():
                try:
                    result = await validator_func(measurements)
                    validations.append(ValidationResult(
                        type=ValidationType.CUSTOM,
                        passed=result.get('passed', False),
                        message=result.get('message', f'{validator_name} validation'),
                        score=result.get('score'),
                        details=result.get('details', {})
                    ))
                except Exception as e:
                    self.logger.error(f"Custom validator {validator_name} failed: {e}")
        
        return validations
    
    async def _run_intelligence_analysis(self, target_id: str, quality_report: QualityReport):
        """Run intelligence analysis on quality report."""
        # Update trend analysis
        await self._update_trend_analysis(target_id, quality_report)
        
        # Update anomaly detection
        await self._update_anomaly_detection(target_id, quality_report)
        
        # Update pattern recognition
        await self._update_pattern_recognition(target_id, quality_report)
    
    async def _update_trend_analysis(self, target_id: str, quality_report: QualityReport):
        """Update trend analysis for target."""
        timestamp = datetime.now()
        
        for trend_type, timeframe in [('short_term_trends', 1), ('medium_term_trends', 24), ('long_term_trends', 168)]:
            trend_data = self.trend_analyzer[trend_type][target_id]
            
            # Add current data point
            trend_data['data_points'].append({
                'timestamp': timestamp,
                'score': quality_report.overall_score
            })
            
            # Keep only data within timeframe (hours)
            cutoff_time = timestamp - timedelta(hours=timeframe)
            trend_data['data_points'] = [
                dp for dp in trend_data['data_points'] 
                if dp['timestamp'] > cutoff_time
            ]
            
            # Analyze trend
            if len(trend_data['data_points']) >= 2:
                scores = [dp['score'] for dp in trend_data['data_points']]
                if len(scores) > 1:
                    recent_change = scores[-1] - scores[0]
                    if recent_change > 5:
                        trend_data['trend_direction'] = 'improving'
                    elif recent_change < -5:
                        trend_data['trend_direction'] = 'declining'
                    else:
                        trend_data['trend_direction'] = 'stable'
                    
                    trend_data['trend_strength'] = abs(recent_change) / len(scores)
                    trend_data['last_analysis'] = timestamp
    
    async def _generate_recommendations(self, target_id: str, 
                                      quality_report: QualityReport) -> List[str]:
        """Generate intelligent recommendations."""
        recommendations = []
        
        # Score-based recommendations
        if quality_report.overall_score < 70:
            recommendations.append("Consider implementing additional quality checks")
        
        for category, score in quality_report.category_scores.items():
            if score.score < 60:
                recommendations.append(f"Focus on improving {category.value} metrics")
        
        # Pattern-based recommendations
        if target_id in self.trend_analyzer['short_term_trends']:
            trend_data = self.trend_analyzer['short_term_trends'][target_id]
            if trend_data['trend_direction'] == 'declining':
                recommendations.append("Investigate recent changes causing quality decline")
        
        # Alert-based recommendations
        if quality_report.alerts:
            high_severity_alerts = [a for a in quality_report.alerts if a.severity == "high"]
            if high_severity_alerts:
                recommendations.append("Address high-severity alerts immediately")
        
        return recommendations
    
    async def _attempt_auto_remediation(self, target_id: str, quality_report: QualityReport):
        """Attempt automatic remediation for known issues."""
        for alert in quality_report.alerts:
            remediation_key = f"{target_id}_{alert.type.value}"
            
            if remediation_key in self.remediation_actions:
                remediation_func = self.remediation_actions[remediation_key]
                try:
                    await remediation_func(target_id, alert, quality_report)
                    self.logger.info(f"Auto-remediation applied for {alert.type.value} on {target_id}")
                except Exception as e:
                    self.logger.error(f"Auto-remediation failed for {remediation_key}: {e}")
    
    async def _handle_quality_alert(self, target_id: str, alert: Alert):
        """Handle alerts from adaptive monitor."""
        # Aggregate alerts
        if target_id not in self.alert_aggregator:
            self.alert_aggregator[target_id] = []
        
        self.alert_aggregator[target_id].append({
            'alert': alert,
            'timestamp': datetime.now()
        })
        
        # Notify handlers
        for handler in self.notification_handlers:
            try:
                await handler(target_id, alert)
            except Exception as e:
                self.logger.error(f"Notification handler failed: {e}")
    
    def register_integration(self, name: str, integration: Any):
        """Register external integration."""
        self.external_integrations[name] = integration
        self.logger.info(f"Registered integration: {name}")
    
    def register_notification_handler(self, handler: Callable):
        """Register notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def register_custom_validator(self, target_id: str, validator_name: str, 
                                validator_func: Callable):
        """Register custom validator for target."""
        if target_id not in self.custom_validators:
            self.custom_validators[target_id] = {}
        self.custom_validators[target_id][validator_name] = validator_func
    
    def register_remediation_action(self, target_id: str, alert_type: AlertType,
                                  remediation_func: Callable):
        """Register auto-remediation action."""
        key = f"{target_id}_{alert_type.value}"
        self.remediation_actions[key] = remediation_func
    
    async def get_system_insights(self) -> Dict[str, Any]:
        """Get comprehensive system insights."""
        insights = {
            'system_status': {
                'active_targets': len(self.active_targets),
                'total_assessments': sum(len(history) for history in self.qa_reports_history.values()),
                'active_integrations': len(self.external_integrations),
                'custom_validators': sum(len(validators) for validators in self.custom_validators.values())
            },
            'performance_summary': {},
            'trend_analysis': {},
            'intelligence_metrics': {}
        }
        
        # Performance summary
        for target_id in self.active_targets:
            if target_id in self.qa_reports_history and self.qa_reports_history[target_id]:
                recent_report = self.qa_reports_history[target_id][-1]
                insights['performance_summary'][target_id] = {
                    'current_score': recent_report.overall_score,
                    'quality_level': recent_report.quality_level.value,
                    'alert_count': len(recent_report.alerts),
                    'last_assessment': recent_report.timestamp.isoformat()
                }
        
        # Intelligence metrics
        if self.adaptive_monitor:
            for target_id in self.active_targets:
                adaptive_insights = self.adaptive_monitor.get_adaptive_insights(target_id)
                insights['intelligence_metrics'][target_id] = adaptive_insights
        
        return insights
    
    async def _load_system_state(self):
        """Load system state from disk."""
        state_file = Path(__file__).parent / "qa_system_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore what we can
                self.performance_metrics = state.get('performance_metrics', {})
                # Note: Some state like trend analysis would need more complex restoration
                
                self.logger.info("System state loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load system state: {e}")
    
    async def _save_system_state(self):
        """Save system state to disk."""
        state_file = Path(__file__).parent / "qa_system_state.json"
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'active_targets': list(self.active_targets),
                'performance_metrics': self.performance_metrics,
                'config': {
                    'enable_adaptive_monitoring': self.config.enable_adaptive_monitoring,
                    'enable_intelligent_scoring': self.config.enable_intelligent_scoring,
                    'enable_predictive_analysis': self.config.enable_predictive_analysis
                }
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save system state: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the QA system."""
        self.logger.info("Shutting down Unified QA System...")
        
        # Stop adaptive monitoring
        if self.adaptive_monitor:
            self.adaptive_monitor.stop_monitoring()
        
        # Save final state
        await self._save_system_state()
        
        self.logger.info("Unified QA System shutdown complete")


async def create_unified_qa_system(config: Optional[QASystemConfig] = None) -> UnifiedQASystem:
    """Factory function to create and initialize unified QA system."""
    system = UnifiedQASystem(config)
    
    if await system.initialize():
        return system
    else:
        raise RuntimeError("Failed to initialize Unified QA System")