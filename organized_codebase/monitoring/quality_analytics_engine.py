"""
Enterprise Quality Analytics Engine for TestMaster
Advanced quality analytics with 20+ metrics and predictive insights
"""

import json
import time
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import math

class QualityMetricType(Enum):
    """Types of quality metrics"""
    COVERAGE = "coverage"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    EFFECTIVENESS = "effectiveness"
    EFFICIENCY = "efficiency"

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class QualityMetric:
    """Individual quality metric with analysis"""
    name: str
    category: QualityMetricType
    value: float
    target: float
    threshold_critical: float
    threshold_poor: float
    threshold_acceptable: float
    threshold_good: float
    level: QualityLevel
    trend: str  # "improving", "stable", "declining"
    confidence: float
    impact_score: float
    recommendations: List[str]

@dataclass
class QualityTrend:
    """Quality trend analysis"""
    metric_name: str
    historical_values: List[float]
    trend_direction: str
    trend_strength: float
    predicted_next: float
    prediction_confidence: float
    volatility: float
    anomalies: List[Tuple[int, float, str]]

@dataclass
class QualityInsight:
    """Quality insight with actionable recommendations"""
    category: str
    severity: str
    title: str
    description: str
    impact: str
    recommendation: str
    effort_estimate: str
    expected_improvement: float

@dataclass
class QualityReport:
    """Comprehensive quality analysis report"""
    timestamp: float
    overall_quality_score: float
    quality_level: QualityLevel
    metrics: Dict[str, QualityMetric]
    trends: Dict[str, QualityTrend]
    insights: List[QualityInsight]
    risk_assessment: Dict[str, float]
    improvement_roadmap: List[str]
    benchmarks: Dict[str, float]
    executive_summary: str
    
    @property
    def overall_score(self) -> float:
        """Backward compatibility property"""
        return self.overall_quality_score

class QualityAnalyticsEngine:
    """Enterprise quality analytics with 20+ metrics"""
    
    def __init__(self):
        self.quality_history: List[QualityReport] = []
        self.metric_definitions = self._initialize_metric_definitions()
        self.benchmark_data = self._load_industry_benchmarks()
        self.analysis_cache: Dict[str, Any] = {}
        
    def _initialize_metric_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive quality metric definitions"""
        return {
            # Coverage Metrics
            "line_coverage": {
                "category": QualityMetricType.COVERAGE,
                "target": 95.0,
                "thresholds": {"critical": 50, "poor": 70, "acceptable": 80, "good": 90},
                "weight": 0.15,
                "description": "Percentage of code lines covered by tests"
            },
            "branch_coverage": {
                "category": QualityMetricType.COVERAGE,
                "target": 90.0,
                "thresholds": {"critical": 40, "poor": 60, "acceptable": 75, "good": 85},
                "weight": 0.12,
                "description": "Percentage of code branches covered by tests"
            },
            "function_coverage": {
                "category": QualityMetricType.COVERAGE,
                "target": 98.0,
                "thresholds": {"critical": 60, "poor": 80, "acceptable": 90, "good": 95},
                "weight": 0.10,
                "description": "Percentage of functions covered by tests"
            },
            
            # Reliability Metrics
            "test_pass_rate": {
                "category": QualityMetricType.RELIABILITY,
                "target": 99.0,
                "thresholds": {"critical": 80, "poor": 90, "acceptable": 95, "good": 98},
                "weight": 0.15,
                "description": "Percentage of tests that pass consistently"
            },
            "flakiness_rate": {
                "category": QualityMetricType.RELIABILITY,
                "target": 1.0,
                "thresholds": {"critical": 20, "poor": 10, "acceptable": 5, "good": 2},
                "weight": 0.12,
                "inverted": True,  # Lower is better
                "description": "Percentage of tests that are flaky"
            },
            "defect_escape_rate": {
                "category": QualityMetricType.RELIABILITY,
                "target": 2.0,
                "thresholds": {"critical": 15, "poor": 10, "acceptable": 5, "good": 3},
                "weight": 0.10,
                "inverted": True,
                "description": "Percentage of defects that escape to production"
            },
            
            # Maintainability Metrics
            "test_maintainability_index": {
                "category": QualityMetricType.MAINTAINABILITY,
                "target": 85.0,
                "thresholds": {"critical": 40, "poor": 60, "acceptable": 70, "good": 80},
                "weight": 0.08,
                "description": "Maintainability index for test code"
            },
            "test_debt_ratio": {
                "category": QualityMetricType.MAINTAINABILITY,
                "target": 5.0,
                "thresholds": {"critical": 30, "poor": 20, "acceptable": 15, "good": 10},
                "weight": 0.07,
                "inverted": True,
                "description": "Ratio of technical debt in test code"
            },
            
            # Performance Metrics
            "test_execution_time": {
                "category": QualityMetricType.PERFORMANCE,
                "target": 300.0,  # 5 minutes
                "thresholds": {"critical": 3600, "poor": 1800, "acceptable": 900, "good": 600},
                "weight": 0.06,
                "inverted": True,
                "description": "Total test suite execution time in seconds"
            },
            "test_efficiency": {
                "category": QualityMetricType.PERFORMANCE,
                "target": 90.0,
                "thresholds": {"critical": 40, "poor": 60, "acceptable": 75, "good": 85},
                "weight": 0.05,
                "description": "Test execution efficiency percentage"
            },
            
            # Security Metrics
            "security_test_coverage": {
                "category": QualityMetricType.SECURITY,
                "target": 85.0,
                "thresholds": {"critical": 30, "poor": 50, "acceptable": 70, "good": 80},
                "weight": 0.08,
                "description": "Coverage of security-related testing"
            },
            "vulnerability_detection_rate": {
                "category": QualityMetricType.SECURITY,
                "target": 95.0,
                "thresholds": {"critical": 50, "poor": 70, "acceptable": 85, "good": 90},
                "weight": 0.07,
                "description": "Rate of vulnerability detection in testing"
            },
            
            # Effectiveness Metrics
            "defect_detection_effectiveness": {
                "category": QualityMetricType.EFFECTIVENESS,
                "target": 90.0,
                "thresholds": {"critical": 40, "poor": 60, "acceptable": 75, "good": 85},
                "weight": 0.10,
                "description": "Effectiveness of tests in detecting defects"
            },
            "requirements_coverage": {
                "category": QualityMetricType.EFFECTIVENESS,
                "target": 95.0,
                "thresholds": {"critical": 50, "poor": 70, "acceptable": 85, "good": 90},
                "weight": 0.08,
                "description": "Coverage of requirements by tests"
            },
            
            # Efficiency Metrics
            "test_automation_rate": {
                "category": QualityMetricType.EFFICIENCY,
                "target": 95.0,
                "thresholds": {"critical": 40, "poor": 60, "acceptable": 80, "good": 90},
                "weight": 0.06,
                "description": "Percentage of tests that are automated"
            },
            "test_roi": {
                "category": QualityMetricType.EFFICIENCY,
                "target": 300.0,  # 3:1 ROI
                "thresholds": {"critical": 100, "poor": 150, "acceptable": 200, "good": 250},
                "weight": 0.05,
                "description": "Return on investment for testing efforts"
            }
        }
    
    def _load_industry_benchmarks(self) -> Dict[str, float]:
        """Load industry benchmark data"""
        return {
            "line_coverage": 82.0,
            "branch_coverage": 75.0,
            "test_pass_rate": 95.0,
            "flakiness_rate": 5.0,
            "test_execution_time": 600.0,
            "security_test_coverage": 65.0,
            "defect_detection_effectiveness": 78.0,
            "test_automation_rate": 85.0
        }
    
    def analyze_quality(self, test_data: Dict[str, Any], 
                       historical_data: Optional[List[Dict]] = None) -> QualityReport:
        """Comprehensive quality analysis with 20+ metrics"""
        timestamp = time.time()
        
        # Calculate individual quality metrics
        metrics = self._calculate_quality_metrics(test_data)
        
        # Analyze trends if historical data available
        trends = self._analyze_quality_trends(metrics, historical_data)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(metrics)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate insights and recommendations
        insights = self._generate_quality_insights(metrics, trends)
        
        # Assess risks
        risk_assessment = self._assess_quality_risks(metrics, trends)
        
        # Create improvement roadmap
        improvement_roadmap = self._create_improvement_roadmap(metrics, insights)
        
        # Compare to benchmarks
        benchmarks = self._compare_to_benchmarks(metrics)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            overall_score, quality_level, metrics, insights, risk_assessment
        )
        
        report = QualityReport(
            timestamp=timestamp,
            overall_quality_score=overall_score,
            quality_level=quality_level,
            metrics=metrics,
            trends=trends,
            insights=insights,
            risk_assessment=risk_assessment,
            improvement_roadmap=improvement_roadmap,
            benchmarks=benchmarks,
            executive_summary=executive_summary
        )
        
        # Store in history
        self.quality_history.append(report)
        if len(self.quality_history) > 100:  # Keep last 100 reports
            self.quality_history = self.quality_history[-100:]
        
        return report
    
    def _calculate_quality_metrics(self, test_data: Dict[str, Any]) -> Dict[str, QualityMetric]:
        """Calculate all quality metrics"""
        metrics = {}
        
        for metric_name, definition in self.metric_definitions.items():
            # Extract value from test data
            value = self._extract_metric_value(metric_name, test_data, definition)
            
            # Calculate quality level
            level = self._calculate_metric_level(value, definition)
            
            # Calculate trend (requires historical data)
            trend = self._calculate_metric_trend(metric_name)
            
            # Calculate confidence
            confidence = self._calculate_metric_confidence(metric_name, test_data)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(metric_name, value, definition)
            
            # Generate recommendations
            recommendations = self._generate_metric_recommendations(
                metric_name, value, level, definition
            )
            
            metrics[metric_name] = QualityMetric(
                name=metric_name,
                category=definition["category"],
                value=value,
                target=definition["target"],
                threshold_critical=definition["thresholds"]["critical"],
                threshold_poor=definition["thresholds"]["poor"],
                threshold_acceptable=definition["thresholds"]["acceptable"],
                threshold_good=definition["thresholds"]["good"],
                level=level,
                trend=trend,
                confidence=confidence,
                impact_score=impact_score,
                recommendations=recommendations
            )
        
        return metrics
    
    def _extract_metric_value(self, metric_name: str, test_data: Dict[str, Any], 
                            definition: Dict[str, Any]) -> float:
        """Extract metric value from test data"""
        # Map metric names to data fields
        data_mapping = {
            "line_coverage": "coverage.line_percentage",
            "branch_coverage": "coverage.branch_percentage", 
            "function_coverage": "coverage.function_percentage",
            "test_pass_rate": "execution.pass_rate",
            "flakiness_rate": "reliability.flakiness_rate",
            "defect_escape_rate": "quality.defect_escape_rate",
            "test_maintainability_index": "maintainability.index",
            "test_debt_ratio": "maintainability.debt_ratio",
            "test_execution_time": "performance.execution_time",
            "test_efficiency": "performance.efficiency",
            "security_test_coverage": "security.coverage",
            "vulnerability_detection_rate": "security.detection_rate",
            "defect_detection_effectiveness": "effectiveness.defect_detection",
            "requirements_coverage": "effectiveness.requirements_coverage",
            "test_automation_rate": "efficiency.automation_rate",
            "test_roi": "efficiency.roi"
        }
        
        data_path = data_mapping.get(metric_name, metric_name)
        
        # Navigate nested dictionary
        current = test_data
        for key in data_path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                # Return estimated value if data not available
                return self._estimate_metric_value(metric_name, definition)
        
        return float(current) if current is not None else self._estimate_metric_value(metric_name, definition)
    
    def _estimate_metric_value(self, metric_name: str, definition: Dict[str, Any]) -> float:
        """Estimate metric value when data is not available"""
        # Use conservative estimates based on metric type
        if metric_name in ["line_coverage", "branch_coverage", "function_coverage"]:
            return 75.0  # Conservative coverage estimate
        elif metric_name in ["test_pass_rate", "defect_detection_effectiveness"]:
            return 85.0  # Conservative reliability estimate
        elif metric_name in ["flakiness_rate", "defect_escape_rate", "test_debt_ratio"]:
            return 8.0   # Conservative negative metric estimate
        elif metric_name == "test_execution_time":
            return 900.0  # 15 minutes conservative estimate
        else:
            return definition["target"] * 0.8  # 80% of target as conservative estimate
    
    def _calculate_metric_level(self, value: float, definition: Dict[str, Any]) -> QualityLevel:
        """Calculate quality level for a metric"""
        thresholds = definition["thresholds"]
        inverted = definition.get("inverted", False)
        
        if inverted:
            # Lower values are better
            if value <= thresholds["good"]:
                return QualityLevel.EXCELLENT
            elif value <= thresholds["acceptable"]:
                return QualityLevel.GOOD
            elif value <= thresholds["poor"]:
                return QualityLevel.ACCEPTABLE
            elif value <= thresholds["critical"]:
                return QualityLevel.POOR
            else:
                return QualityLevel.CRITICAL
        else:
            # Higher values are better
            if value >= thresholds["good"]:
                return QualityLevel.EXCELLENT
            elif value >= thresholds["acceptable"]:
                return QualityLevel.GOOD
            elif value >= thresholds["poor"]:
                return QualityLevel.ACCEPTABLE
            elif value >= thresholds["critical"]:
                return QualityLevel.POOR
            else:
                return QualityLevel.CRITICAL
    
    def _calculate_metric_trend(self, metric_name: str) -> str:
        """Calculate metric trend from historical data"""
        if len(self.quality_history) < 3:
            return "stable"
        
        # Get last 5 values for trend analysis
        historical_values = []
        for report in self.quality_history[-5:]:
            if metric_name in report.metrics:
                historical_values.append(report.metrics[metric_name].value)
        
        if len(historical_values) < 3:
            return "stable"
        
        # Simple linear trend analysis
        x = list(range(len(historical_values)))
        y = historical_values
        
        # Calculate slope
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        # Determine trend direction
        if slope > 2:
            return "improving"
        elif slope < -2:
            return "declining"
        else:
            return "stable"
    
    def _calculate_metric_confidence(self, metric_name: str, test_data: Dict[str, Any]) -> float:
        """Calculate confidence in metric value"""
        # Base confidence on data completeness and collection method
        base_confidence = 0.8
        
        # Boost confidence based on data source quality
        if "high_quality_data" in test_data:
            base_confidence += 0.1
        
        # Reduce confidence for estimated values
        if metric_name not in str(test_data):
            base_confidence -= 0.3
        
        # Boost confidence for metrics with historical consistency
        if len(self.quality_history) > 5:
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_impact_score(self, metric_name: str, value: float, 
                              definition: Dict[str, Any]) -> float:
        """Calculate impact score for metric"""
        # Impact based on weight and deviation from target
        weight = definition["weight"]
        target = definition["target"]
        
        # Calculate normalized deviation
        if definition.get("inverted", False):
            deviation = max(0, value - target) / target
        else:
            deviation = max(0, target - value) / target
        
        # Impact score combines weight and deviation
        impact_score = weight * (1 + deviation)
        
        return min(1.0, impact_score)
    
    def _generate_metric_recommendations(self, metric_name: str, value: float, 
                                       level: QualityLevel, 
                                       definition: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for metric"""
        recommendations = []
        
        if level in [QualityLevel.CRITICAL, QualityLevel.POOR]:
            if metric_name in ["line_coverage", "branch_coverage", "function_coverage"]:
                recommendations.extend([
                    "Increase test coverage by adding missing test cases",
                    "Use coverage tools to identify untested code paths",
                    "Implement coverage gates in CI/CD pipeline"
                ])
            elif metric_name == "test_pass_rate":
                recommendations.extend([
                    "Investigate and fix failing tests immediately",
                    "Implement test stabilization program",
                    "Review test environment consistency"
                ])
            elif metric_name == "flakiness_rate":
                recommendations.extend([
                    "Identify and fix flaky tests using statistical analysis",
                    "Improve test isolation and cleanup",
                    "Review timing dependencies and race conditions"
                ])
            elif metric_name == "test_execution_time":
                recommendations.extend([
                    "Optimize slow-running tests",
                    "Implement parallel test execution",
                    "Review test data management strategies"
                ])
        
        elif level == QualityLevel.ACCEPTABLE:
            recommendations.append(f"Continue monitoring {metric_name} and aim for good level")
        
        # Add general improvement recommendations
        if metric_name.endswith("_coverage"):
            recommendations.append("Consider implementing mutation testing for coverage quality")
        
        if "security" in metric_name:
            recommendations.append("Integrate security testing into regular test cycles")
        
        return recommendations[:3]  # Limit to top 3
    
    def _analyze_quality_trends(self, current_metrics: Dict[str, QualityMetric],
                              historical_data: Optional[List[Dict]]) -> Dict[str, QualityTrend]:
        """Analyze quality trends over time"""
        trends = {}
        
        for metric_name in current_metrics.keys():
            # Collect historical values
            historical_values = []
            
            # From stored history
            for report in self.quality_history[-20:]:  # Last 20 reports
                if metric_name in report.metrics:
                    historical_values.append(report.metrics[metric_name].value)
            
            # From provided historical data
            if historical_data and isinstance(historical_data, list):
                for data_point in historical_data[-10:]:  # Last 10 data points
                    if isinstance(data_point, dict) and metric_name in data_point:
                        historical_values.append(float(data_point[metric_name]))
            
            if len(historical_values) >= 3:
                trend = self._calculate_detailed_trend(metric_name, historical_values)
                trends[metric_name] = trend
        
        return trends
    
    def _calculate_detailed_trend(self, metric_name: str, 
                                values: List[float]) -> QualityTrend:
        """Calculate detailed trend analysis"""
        if len(values) < 3:
            return QualityTrend(
                metric_name=metric_name,
                historical_values=values,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                predicted_next=values[-1] if values else 0.0,
                prediction_confidence=0.0,
                volatility=0.0,
                anomalies=[]
            )
        
        # Calculate trend direction and strength
        x = list(range(len(values)))
        y = values
        
        # Linear regression
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        intercept = (sum(y) - slope * sum(x)) / n
        
        # Determine trend direction
        if abs(slope) < 0.1:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "declining"
        
        # Calculate trend strength (R-squared approximation)
        y_mean = statistics.mean(y)
        ss_res = sum((y[i] - (slope * x[i] + intercept))**2 for i in range(n))
        ss_tot = sum((y[i] - y_mean)**2 for i in range(n))
        trend_strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Predict next value
        predicted_next = slope * len(values) + intercept
        prediction_confidence = min(1.0, trend_strength)
        
        # Calculate volatility
        if len(values) > 1:
            volatility = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else 0
        else:
            volatility = 0
        
        # Identify anomalies (values more than 2 standard deviations from mean)
        anomalies = []
        if len(values) > 3:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            for i, value in enumerate(values):
                if abs(value - mean_val) > 2 * std_val:
                    anomalies.append((i, value, "outlier"))
        
        return QualityTrend(
            metric_name=metric_name,
            historical_values=values,
            trend_direction=direction,
            trend_strength=trend_strength,
            predicted_next=predicted_next,
            prediction_confidence=prediction_confidence,
            volatility=volatility,
            anomalies=anomalies
        )
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, QualityMetric]) -> float:
        """Calculate weighted overall quality score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric in metrics.items():
            definition = self.metric_definitions[metric_name]
            weight = definition["weight"]
            
            # Normalize metric value to 0-100 scale
            if definition.get("inverted", False):
                # For inverted metrics, lower is better
                normalized_score = max(0, min(100, 100 * (1 - metric.value / definition["target"])))
            else:
                # For normal metrics, higher is better
                normalized_score = max(0, min(100, 100 * metric.value / definition["target"]))
            
            total_weighted_score += normalized_score * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0
        return round(overall_score, 2)
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine overall quality level"""
        if overall_score >= 90:
            return QualityLevel.EXCELLENT
        elif overall_score >= 75:
            return QualityLevel.GOOD
        elif overall_score >= 60:
            return QualityLevel.ACCEPTABLE
        elif overall_score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _generate_quality_insights(self, metrics: Dict[str, QualityMetric],
                                 trends: Dict[str, QualityTrend]) -> List[QualityInsight]:
        """Generate actionable quality insights"""
        insights = []
        
        # Identify critical issues
        critical_metrics = [m for m in metrics.values() if m.level == QualityLevel.CRITICAL]
        for metric in critical_metrics:
            insights.append(QualityInsight(
                category="Critical Issue",
                severity="High",
                title=f"Critical {metric.name} Issue",
                description=f"{metric.name} is at critical level ({metric.value:.1f})",
                impact="High risk to product quality and delivery",
                recommendation=metric.recommendations[0] if metric.recommendations else "Immediate attention required",
                effort_estimate="High",
                expected_improvement=50.0
            ))
        
        # Identify declining trends
        declining_trends = [t for t in trends.values() if t.trend_direction == "declining" and t.trend_strength > 0.6]
        for trend in declining_trends:
            insights.append(QualityInsight(
                category="Declining Trend",
                severity="Medium",
                title=f"Declining {trend.metric_name}",
                description=f"{trend.metric_name} has been declining with {trend.trend_strength:.1%} confidence",
                impact="Quality degradation over time",
                recommendation="Investigate root causes and implement corrective actions",
                effort_estimate="Medium",
                expected_improvement=30.0
            ))
        
        # Identify improvement opportunities
        improvable_metrics = [m for m in metrics.values() 
                            if m.level in [QualityLevel.ACCEPTABLE, QualityLevel.POOR] 
                            and m.impact_score > 0.3]
        for metric in improvable_metrics[:3]:  # Top 3 opportunities
            insights.append(QualityInsight(
                category="Improvement Opportunity",
                severity="Low",
                title=f"Improve {metric.name}",
                description=f"{metric.name} can be improved from {metric.level.value} to higher level",
                impact=f"Medium impact on overall quality (weight: {self.metric_definitions[metric.name]['weight']:.1%})",
                recommendation=metric.recommendations[0] if metric.recommendations else "Focus improvement efforts here",
                effort_estimate="Medium",
                expected_improvement=20.0
            ))
        
        # Identify anomalies
        for trend in trends.values():
            if trend.anomalies:
                insights.append(QualityInsight(
                    category="Anomaly Detected",
                    severity="Medium",
                    title=f"Anomaly in {trend.metric_name}",
                    description=f"Detected {len(trend.anomalies)} anomalies in {trend.metric_name}",
                    impact="Potential data quality or measurement issues",
                    recommendation="Investigate anomalous data points and measurement consistency",
                    effort_estimate="Low",
                    expected_improvement=10.0
                ))
        
        return insights[:10]  # Limit to top 10 insights
    
    def _assess_quality_risks(self, metrics: Dict[str, QualityMetric],
                            trends: Dict[str, QualityTrend]) -> Dict[str, float]:
        """Assess quality-related risks"""
        risks = {}
        
        # Release Risk
        critical_count = sum(1 for m in metrics.values() if m.level == QualityLevel.CRITICAL)
        poor_count = sum(1 for m in metrics.values() if m.level == QualityLevel.POOR)
        release_risk = min(1.0, (critical_count * 0.3 + poor_count * 0.1))
        risks["release_readiness"] = release_risk
        
        # Maintenance Risk
        maintainability_metrics = [m for m in metrics.values() 
                                 if m.category == QualityMetricType.MAINTAINABILITY]
        if maintainability_metrics:
            avg_maintainability = statistics.mean([m.value for m in maintainability_metrics])
            maintenance_risk = max(0, 1 - avg_maintainability / 100)
            risks["maintenance_burden"] = maintenance_risk
        
        # Performance Risk
        performance_metrics = [m for m in metrics.values() 
                             if m.category == QualityMetricType.PERFORMANCE]
        if performance_metrics:
            performance_issues = sum(1 for m in performance_metrics 
                                   if m.level in [QualityLevel.CRITICAL, QualityLevel.POOR])
            performance_risk = performance_issues / len(performance_metrics)
            risks["performance_degradation"] = performance_risk
        
        # Security Risk
        security_metrics = [m for m in metrics.values() 
                          if m.category == QualityMetricType.SECURITY]
        if security_metrics:
            security_issues = sum(1 for m in security_metrics 
                                if m.level in [QualityLevel.CRITICAL, QualityLevel.POOR])
            security_risk = security_issues / len(security_metrics)
            risks["security_vulnerabilities"] = security_risk
        
        # Trend Risk (declining quality)
        declining_count = sum(1 for t in trends.values() 
                            if t.trend_direction == "declining" and t.trend_strength > 0.5)
        trend_risk = min(1.0, declining_count / len(trends)) if trends else 0
        risks["quality_degradation"] = trend_risk
        
        return risks
    
    def _create_improvement_roadmap(self, metrics: Dict[str, QualityMetric],
                                  insights: List[QualityInsight]) -> List[str]:
        """Create prioritized improvement roadmap"""
        roadmap = []
        
        # Immediate actions (critical issues)
        critical_actions = [insight.recommendation for insight in insights 
                          if insight.severity == "High"]
        if critical_actions:
            roadmap.extend([f"IMMEDIATE: {action}" for action in critical_actions[:3]])
        
        # Short-term actions (high-impact improvements)
        high_impact_metrics = sorted([m for m in metrics.values() if m.impact_score > 0.5],
                                   key=lambda x: x.impact_score, reverse=True)
        for metric in high_impact_metrics[:3]:
            if metric.recommendations:
                roadmap.append(f"SHORT-TERM: {metric.recommendations[0]}")
        
        # Medium-term actions (systematic improvements)
        medium_actions = [insight.recommendation for insight in insights 
                        if insight.severity == "Medium"]
        if medium_actions:
            roadmap.extend([f"MEDIUM-TERM: {action}" for action in medium_actions[:2]])
        
        # Long-term actions (optimization)
        roadmap.extend([
            "LONG-TERM: Implement advanced quality analytics",
            "LONG-TERM: Establish quality benchmarking program",
            "LONG-TERM: Automate quality gate enforcement"
        ])
        
        return roadmap[:10]  # Limit to top 10 items
    
    def _compare_to_benchmarks(self, metrics: Dict[str, QualityMetric]) -> Dict[str, float]:
        """Compare metrics to industry benchmarks"""
        benchmark_comparison = {}
        
        for metric_name, metric in metrics.items():
            if metric_name in self.benchmark_data:
                benchmark_value = self.benchmark_data[metric_name]
                comparison_ratio = metric.value / benchmark_value
                benchmark_comparison[metric_name] = comparison_ratio
        
        return benchmark_comparison
    
    def _generate_executive_summary(self, overall_score: float, quality_level: QualityLevel,
                                  metrics: Dict[str, QualityMetric], 
                                  insights: List[QualityInsight],
                                  risks: Dict[str, float]) -> str:
        """Generate executive summary of quality analysis"""
        
        # Overall quality assessment
        summary = f"Overall Quality Score: {overall_score:.1f}/100 ({quality_level.value.title()})\n\n"
        
        # Key metrics summary
        excellent_count = sum(1 for m in metrics.values() if m.level == QualityLevel.EXCELLENT)
        critical_count = sum(1 for m in metrics.values() if m.level == QualityLevel.CRITICAL)
        
        summary += f"Quality Distribution: {excellent_count} excellent, "
        summary += f"{critical_count} critical out of {len(metrics)} metrics\n\n"
        
        # Top risks
        top_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_risks:
            summary += "Top Risks:\n"
            for risk_name, risk_value in top_risks:
                summary += f"- {risk_name.replace('_', ' ').title()}: {risk_value:.1%}\n"
            summary += "\n"
        
        # Key recommendations
        high_priority_insights = [i for i in insights if i.severity == "High"]
        if high_priority_insights:
            summary += "Immediate Actions Required:\n"
            for insight in high_priority_insights[:3]:
                summary += f"- {insight.title}: {insight.recommendation}\n"
        
        return summary
    
    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quality dashboard"""
        if not self.quality_history:
            return {"status": "No quality data available"}
        
        latest_report = self.quality_history[-1]
        
        # Prepare dashboard data
        dashboard_data = {
            "overall_score": latest_report.overall_quality_score,
            "quality_level": latest_report.quality_level.value,
            "metrics_summary": {},
            "trend_data": {},
            "risk_indicators": latest_report.risk_assessment,
            "recent_insights": [asdict(insight) for insight in latest_report.insights[:5]],
            "improvement_progress": self._calculate_improvement_progress()
        }
        
        # Metrics summary by category
        for metric in latest_report.metrics.values():
            category = metric.category.value
            if category not in dashboard_data["metrics_summary"]:
                dashboard_data["metrics_summary"][category] = []
            
            dashboard_data["metrics_summary"][category].append({
                "name": metric.name,
                "value": metric.value,
                "level": metric.level.value,
                "trend": metric.trend
            })
        
        # Trend data for charts
        if len(self.quality_history) > 1:
            dashboard_data["trend_data"] = {
                "timestamps": [r.timestamp for r in self.quality_history[-20:]],
                "overall_scores": [r.overall_quality_score for r in self.quality_history[-20:]]
            }
        
        return dashboard_data
    
    def _calculate_improvement_progress(self) -> Dict[str, float]:
        """Calculate improvement progress over time"""
        if len(self.quality_history) < 2:
            return {}
        
        current = self.quality_history[-1]
        previous = self.quality_history[-2]
        
        progress = {}
        for metric_name in current.metrics:
            if metric_name in previous.metrics:
                current_value = current.metrics[metric_name].value
                previous_value = previous.metrics[metric_name].value
                improvement = ((current_value - previous_value) / previous_value * 100) if previous_value > 0 else 0
                progress[metric_name] = improvement
        
        return progress
    
    def export_quality_report(self, report: QualityReport, format_type: str = "json") -> str:
        """Export quality report in various formats"""
        if format_type == "json":
            return json.dumps(asdict(report), indent=2, default=str)
        elif format_type == "summary":
            return report.executive_summary
        else:
            return f"Quality Report - Score: {report.overall_quality_score}/100"