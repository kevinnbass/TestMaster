"""
Modularized Agent Quality Assurance System
==========================================

Main interface for agent quality assurance, using modular components.
This replaces the original 1749-line agent_qa.py with a clean, modular design.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import uuid

from .qa_base import (
    QualityMetric, QualityAlert, QualityReport, QualityThreshold, QualityScore,
    ValidationResult, ValidationIssue, BenchmarkResult, ScoreCategory, ScoreBreakdown,
    PerformanceMetric, ValidationRule, AlertType
)
from .qa_monitor import QualityMonitor
from .qa_scorer import QualityScorer


class AgentQualityAssurance:
    """
    Unified agent quality assurance system.
    
    Coordinates monitoring, scoring, validation, and benchmarking
    through modular components.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the quality assurance system."""
        self.agent_id = agent_id
        self.config = config or {}
        
        # Initialize components
        self.monitor = QualityMonitor(config)
        self.scorer = QualityScorer(config)
        
        # Advanced features from archive integration
        self._validation_rules = []
        self._benchmark_baselines = {}
        self._alerts = []
        self._thresholds = []
        
        # Advanced monitoring from archive
        self._agent_metrics = {}
        self._scoring_history = {}
        self._benchmark_history = {}
        self._validation_history = {}
        self._quality_standards = self._setup_quality_standards()
        self._score_weights = self._setup_default_weights()
        
        # Report generation
        self._reports: List[QualityReport] = []
        
        # Start monitoring if configured
        if self.config.get('auto_start_monitoring', False):
            self.monitor.start_monitoring()
    
    def assess_quality(self, 
                       agent_output: Any,
                       expected_output: Optional[Any] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> QualityReport:
        """
        Perform comprehensive quality assessment.
        
        Args:
            agent_output: The output to assess
            expected_output: Expected output for comparison
            metadata: Additional context
            
        Returns:
            Comprehensive quality report
        """
        metrics = []
        
        # Extract metrics from output
        if isinstance(agent_output, dict):
            # Extract performance metrics
            if 'response_time' in agent_output:
                metrics.append(QualityMetric(
                    name='response_time',
                    value=agent_output['response_time'],
                    timestamp=datetime.now(),
                    category=ScoreCategory.PERFORMANCE
                ))
            
            # Extract reliability metrics
            if 'success' in agent_output:
                metrics.append(QualityMetric(
                    name='success_rate',
                    value=100.0 if agent_output['success'] else 0.0,
                    timestamp=datetime.now(),
                    category=ScoreCategory.RELIABILITY
                ))
        
        # Add functionality metrics
        if expected_output is not None:
            accuracy = self._calculate_accuracy(agent_output, expected_output)
            metrics.append(QualityMetric(
                name='accuracy',
                value=accuracy,
                timestamp=datetime.now(),
                category=ScoreCategory.FUNCTIONALITY
            ))
        
        # Record metrics for monitoring
        for metric in metrics:
            self.monitor.record_metric(metric)
        
        # Calculate quality score
        quality_score = self.scorer.calculate_score(metrics)
        
        # Perform validation (simplified for now)
        validation_results = []
        
        # Perform benchmarking (simplified for now)
        benchmark_results = []
        
        # Get active alerts
        alerts = self.monitor.get_active_alerts()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_score, validation_results, benchmark_results
        )
        
        # Create report
        report = QualityReport(
            report_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            quality_score=quality_score,
            benchmark_results=benchmark_results,
            validation_results=validation_results,
            alerts=alerts,
            recommendations=recommendations,
            metadata=metadata or {}
        )
        
        self._reports.append(report)
        return report
    
    def _calculate_accuracy(self, actual: Any, expected: Any) -> float:
        """Calculate accuracy between actual and expected output."""
        if actual == expected:
            return 100.0
        
        # Simple similarity for strings
        if isinstance(actual, str) and isinstance(expected, str):
            # Simplified - in reality would use more sophisticated comparison
            common = sum(1 for a, e in zip(actual, expected) if a == e)
            return (common / max(len(actual), len(expected))) * 100
        
        return 0.0
    
    def _generate_recommendations(self,
                                 quality_score,
                                 validation_results: List,
                                 benchmark_results: List) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Score-based recommendations
        if quality_score.total_score < 60:
            recommendations.append("Critical: Quality score below acceptable threshold")
            
            # Category-specific recommendations
            for breakdown in quality_score.breakdown:
                if breakdown.score < 50:
                    recommendations.append(
                        f"Improve {breakdown.category.value}: score {breakdown.score:.1f}"
                    )
        
        # Alert-based recommendations
        alerts = self.monitor.get_active_alerts()
        if len(alerts) > 5:
            recommendations.append("Multiple quality alerts active - investigate immediately")
        
        # Trend-based recommendations
        trends = self.scorer.get_score_trends()
        if trends.get('trend_direction') == 'degrading':
            recommendations.append("Quality trend is degrading - preventive action needed")
        
        return recommendations
    
    def _setup_quality_standards(self) -> Dict[str, Dict[str, float]]:
        """Setup quality standards from archive."""
        return {
            'response_time': {'excellent': 100, 'good': 200, 'poor': 1000},
            'accuracy': {'excellent': 95, 'good': 85, 'poor': 70},
            'reliability': {'excellent': 99, 'good': 95, 'poor': 90}
        }
    
    def _setup_default_weights(self) -> Dict[ScoreCategory, float]:
        """Setup default score weights."""
        return {
            ScoreCategory.FUNCTIONALITY: 0.3,
            ScoreCategory.RELIABILITY: 0.25,
            ScoreCategory.PERFORMANCE: 0.2,
            ScoreCategory.SECURITY: 0.15,
            ScoreCategory.MAINTAINABILITY: 0.1
        }
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add custom validation rule."""
        if self.agent_id not in self._validation_rules:
            self._validation_rules[self.agent_id] = []
        self._validation_rules[self.agent_id].append(rule)
    
    def validate_agent_output(self, agent_output: Any) -> ValidationResult:
        """Comprehensive validation of agent output."""
        from .qa_base import ValidationType
        issues = []
        
        # Basic validation checks
        if agent_output is None:
            issues.append(ValidationIssue(
                issue_id=str(uuid.uuid4()),
                validation_type=ValidationType.CONTENT,
                severity="error",
                message="Agent output is null",
                suggestion="Ensure agent produces valid output"
            ))
        
        # Format validation for dict outputs
        if isinstance(agent_output, dict):
            if 'error' in agent_output:
                issues.append(ValidationIssue(
                    issue_id=str(uuid.uuid4()),
                    validation_type=ValidationType.FORMAT,
                    severity="warning",
                    message=f"Error in output: {agent_output['error']}",
                    suggestion="Address the reported error"
                ))
        
        # Create validation result
        result = ValidationResult(
            is_valid=len([i for i in issues if i.severity == 'error']) == 0,
            validation_type=ValidationType.CONTENT,
            issues=issues,
            timestamp=datetime.now()
        )
        
        # Store validation history
        if self.agent_id not in self._validation_history:
            self._validation_history[self.agent_id] = []
        self._validation_history[self.agent_id].append(result)
        
        return result
    
    def benchmark_performance(self, operation: Callable, *args, **kwargs) -> BenchmarkResult:
        """Benchmark agent operation performance."""
        from .qa_base import BenchmarkType
        start_time = time.time()
        
        try:
            # Execute operation and measure
            result = operation(*args, **kwargs)
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms
            
            # Create response time benchmark result
            baseline = self._benchmark_baselines.get('response_time', 500)
            threshold = baseline * 1.2
            passed = response_time <= threshold
            improvement = ((baseline - response_time) / baseline * 100) if baseline > 0 else 0
            
            benchmark_result = BenchmarkResult(
                benchmark_type=BenchmarkType.RESPONSE_TIME,
                value=response_time,
                baseline=baseline,
                threshold=threshold,
                passed=passed,
                improvement=improvement,
                timestamp=datetime.now(),
                metadata={'operation': str(operation.__name__) if hasattr(operation, '__name__') else 'unknown'}
            )
            
            # Store benchmark history
            if self.agent_id not in self._benchmark_history:
                self._benchmark_history[self.agent_id] = []
            self._benchmark_history[self.agent_id].append(benchmark_result)
            
            return benchmark_result
            
        except Exception as e:
            # Handle benchmark failure
            return BenchmarkResult(
                benchmark_type=BenchmarkType.RESPONSE_TIME,
                value=0.0,
                baseline=500.0,
                threshold=600.0,
                passed=False,
                improvement=-100.0,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics summary."""
        latest_score = self.scorer.get_latest_score()
        return {
            'agent_id': self.agent_id,
            'latest_score': latest_score,
            'average_score': self.scorer.get_average_score(),
            'active_alerts': len(self.monitor.get_active_alerts()),
            'metrics_summary': self.monitor.get_metrics_summary(),
            'score_trends': self.scorer.get_score_trends(),
            'reports_generated': len(self._reports),
            'validation_history_count': len(self._validation_history.get(self.agent_id, [])),
            'benchmark_history_count': len(self._benchmark_history.get(self.agent_id, [])),
            'total_alerts': len(self._alerts),
            'quality_grade': latest_score.grade if hasattr(latest_score, 'grade') else 'N/A'
        }
    
    def start_monitoring(self):
        """Start continuous quality monitoring."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop continuous quality monitoring."""
        self.monitor.stop_monitoring()
    
    def get_latest_report(self) -> Optional[QualityReport]:
        """Get the most recent quality report."""
        return self._reports[-1] if self._reports else None
    
    def get_reports(self, limit: int = 10) -> List[QualityReport]:
        """Get recent quality reports."""
        return self._reports[-limit:]
    
    def export_metrics(self, format: str = 'json') -> Union[str, Dict]:
        """Export quality metrics in specified format."""
        metrics = self.get_quality_metrics()
        
        if format == 'json':
            import json
            return json.dumps(metrics, default=str, indent=2)
        else:
            return metrics
    
    def reset_metrics(self):
        """Reset all quality metrics and history."""
        self.monitor = QualityMonitor(self.config)
        self.scorer = QualityScorer(self.config)
        self._reports = []
    
    def configure_threshold(self, metric: str, value: float, operator: str = 'gt'):
        """Configure a monitoring threshold."""
        from .qa_base import QualityThreshold, AlertType
        
        threshold = QualityThreshold(
            name=f"Custom threshold for {metric}",
            metric=metric,
            value=value,
            operator=operator,
            alert_type=AlertType.THRESHOLD_BREACH,
            severity="medium"
        )
        self.monitor.add_threshold(threshold)
    
    def get_validation_history(self, limit: int = 10) -> List[ValidationResult]:
        """Get recent validation history."""
        history = self._validation_history.get(self.agent_id, [])
        return history[-limit:] if history else []
    
    def get_benchmark_history(self, limit: int = 10) -> List[BenchmarkResult]:
        """Get recent benchmark history."""
        history = self._benchmark_history.get(self.agent_id, [])
        return history[-limit:] if history else []
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Get comprehensive quality trends analysis."""
        scores = self._scoring_history.get(self.agent_id, [])
        validations = self._validation_history.get(self.agent_id, [])
        benchmarks = self._benchmark_history.get(self.agent_id, [])
        
        trends = {
            'score_trend': 'stable',
            'validation_trend': 'stable',
            'benchmark_trend': 'stable',
            'overall_direction': 'stable'
        }
        
        # Analyze score trends
        if len(scores) >= 3:
            recent_scores = [s.total_score for s in scores[-3:]]
            if recent_scores[-1] > recent_scores[0]:
                trends['score_trend'] = 'improving'
            elif recent_scores[-1] < recent_scores[0]:
                trends['score_trend'] = 'degrading'
        
        # Analyze validation trends (using is_valid boolean)
        if len(validations) >= 3:
            recent_validations = [1.0 if v.is_valid else 0.0 for v in validations[-3:]]
            if recent_validations[-1] > recent_validations[0]:
                trends['validation_trend'] = 'improving'
            elif recent_validations[-1] < recent_validations[0]:
                trends['validation_trend'] = 'degrading'
        
        # Analyze benchmark trends (using improvement percentage)
        if len(benchmarks) >= 3:
            recent_benchmarks = [b.improvement for b in benchmarks[-3:]]
            if recent_benchmarks[-1] > recent_benchmarks[0]:
                trends['benchmark_trend'] = 'improving'
            elif recent_benchmarks[-1] < recent_benchmarks[0]:
                trends['benchmark_trend'] = 'degrading'
        
        # Overall direction
        improving_count = sum(1 for t in [trends['score_trend'], trends['validation_trend'], trends['benchmark_trend']] if t == 'improving')
        degrading_count = sum(1 for t in [trends['score_trend'], trends['validation_trend'], trends['benchmark_trend']] if t == 'degrading')
        
        if improving_count > degrading_count:
            trends['overall_direction'] = 'improving'
        elif degrading_count > improving_count:
            trends['overall_direction'] = 'degrading'
        
        return trends
    
    def __repr__(self) -> str:
        """String representation."""
        score = self.scorer.get_latest_score()
        if hasattr(score, 'total_score'):
            score_val = f"{score.total_score:.1f}"
        elif hasattr(score, 'overall_score'):
            score_val = f"{score.overall_score:.1f}"
        else:
            score_val = "N/A"
        return f"AgentQualityAssurance(agent_id='{self.agent_id}', latest_score={score_val})"


# Export main class
__all__ = ['AgentQualityAssurance']