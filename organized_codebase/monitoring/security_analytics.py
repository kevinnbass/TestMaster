"""
Security Analytics Engine

Advanced security trend analysis and risk assessment algorithms.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityTrend:
    """Represents a security trend."""
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable
    confidence: float
    rate_of_change: float
    prediction: float
    

@dataclass
class RiskAssessment:
    """Risk assessment result."""
    risk_score: float  # 0-100
    risk_level: str    # low, medium, high, critical
    contributing_factors: List[str]
    recommendations: List[str]
    confidence: float
    

class SecurityAnalytics:
    """
    Advanced security analytics with trend analysis and risk assessment.
    Provides predictive security insights and threat intelligence integration.
    """
    
    def __init__(self):
        """Initialize security analytics engine."""
        self.trend_history = defaultdict(list)
        self.risk_factors = {}
        self.threat_indicators = []
        logger.info("Security Analytics Engine initialized")
        
    def analyze_security_trends(self, 
                               metrics_data: List[Dict],
                               window_days: int = 7) -> List[SecurityTrend]:
        """
        Analyze security trends from metrics data.
        
        Args:
            metrics_data: Historical metrics data
            window_days: Analysis window in days
            
        Returns:
            List of security trends
        """
        trends = []
        
        # Group metrics by name
        grouped_metrics = defaultdict(list)
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        for metric in metrics_data:
            metric_date = datetime.fromisoformat(metric['timestamp'])
            if metric_date >= cutoff_date:
                grouped_metrics[metric['metric_name']].append({
                    'timestamp': metric_date,
                    'value': metric['value']
                })
                
        # Analyze each metric
        for metric_name, data_points in grouped_metrics.items():
            if len(data_points) >= 3:  # Need minimum data points
                trend = self._calculate_trend(metric_name, data_points)
                trends.append(trend)
                
        return trends
        
    def assess_overall_risk(self, 
                           vulnerability_count: int,
                           compliance_score: float,
                           threat_indicators: List[str],
                           dependency_risks: int) -> RiskAssessment:
        """
        Assess overall security risk.
        
        Args:
            vulnerability_count: Number of vulnerabilities
            compliance_score: Compliance score (0-100)
            threat_indicators: List of threat indicators
            dependency_risks: Number of dependency risks
            
        Returns:
            Risk assessment
        """
        # Calculate base risk score
        vuln_risk = min(vulnerability_count * 5, 40)  # Max 40 points
        compliance_risk = max(0, (100 - compliance_score) * 0.3)  # Max 30 points
        threat_risk = len(threat_indicators) * 10  # Max varies
        dependency_risk = min(dependency_risks * 3, 20)  # Max 20 points
        
        total_risk = vuln_risk + compliance_risk + threat_risk + dependency_risk
        risk_score = min(100, total_risk)
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "critical"
        elif risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 30:
            risk_level = "medium"
        else:
            risk_level = "low"
            
        # Contributing factors
        factors = []
        if vuln_risk > 20:
            factors.append(f"High vulnerability count ({vulnerability_count})")
        if compliance_risk > 15:
            factors.append(f"Low compliance score ({compliance_score:.1f}%)")
        if threat_risk > 10:
            factors.append(f"Active threat indicators ({len(threat_indicators)})")
        if dependency_risk > 10:
            factors.append(f"Dependency vulnerabilities ({dependency_risks})")
            
        # Recommendations
        recommendations = self._generate_risk_recommendations(
            risk_level, vulnerability_count, compliance_score, 
            len(threat_indicators), dependency_risks
        )
        
        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            contributing_factors=factors,
            recommendations=recommendations,
            confidence=0.85  # Base confidence
        )
        
    def detect_anomalies(self, 
                        metrics_data: List[Dict],
                        sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect security anomalies using statistical analysis.
        
        Args:
            metrics_data: Metrics data
            sensitivity: Anomaly detection sensitivity
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Group by metric name
        grouped_metrics = defaultdict(list)
        for metric in metrics_data:
            grouped_metrics[metric['metric_name']].append(metric['value'])
            
        for metric_name, values in grouped_metrics.items():
            if len(values) >= 10:  # Need sufficient data
                anomaly_scores = self._calculate_anomaly_scores(values, sensitivity)
                
                for i, score in enumerate(anomaly_scores):
                    if score > sensitivity:
                        anomalies.append({
                            'metric_name': metric_name,
                            'value': values[i],
                            'anomaly_score': score,
                            'severity': 'high' if score > 3.0 else 'medium',
                            'timestamp': metrics_data[i].get('timestamp', '')
                        })
                        
        return anomalies
        
    def predict_security_metrics(self, 
                                metrics_data: List[Dict],
                                forecast_days: int = 7) -> Dict[str, List[float]]:
        """
        Predict future security metrics.
        
        Args:
            metrics_data: Historical metrics
            forecast_days: Days to forecast
            
        Returns:
            Predicted metrics
        """
        predictions = {}
        
        # Group by metric name
        grouped_metrics = defaultdict(list)
        for metric in metrics_data:
            grouped_metrics[metric['metric_name']].append(metric['value'])
            
        for metric_name, values in grouped_metrics.items():
            if len(values) >= 5:
                forecast = self._simple_forecast(values, forecast_days)
                predictions[metric_name] = forecast
                
        return predictions
        
    def correlate_security_events(self, 
                                 events: List[Dict],
                                 time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Find correlations between security events.
        
        Args:
            events: Security events
            time_window_hours: Correlation time window
            
        Returns:
            Correlated event patterns
        """
        correlations = []
        time_window = timedelta(hours=time_window_hours)
        
        # Group events by time windows
        event_windows = []
        current_window = []
        
        events_sorted = sorted(events, key=lambda x: x.get('timestamp', ''))
        
        for event in events_sorted:
            if not current_window:
                current_window.append(event)
            else:
                last_time = datetime.fromisoformat(current_window[-1]['timestamp'])
                event_time = datetime.fromisoformat(event['timestamp'])
                
                if event_time - last_time <= time_window:
                    current_window.append(event)
                else:
                    if len(current_window) > 1:
                        event_windows.append(current_window)
                    current_window = [event]
                    
        # Analyze correlations within windows
        for window in event_windows:
            if len(window) >= 2:
                correlation = self._analyze_event_correlation(window)
                if correlation:
                    correlations.append(correlation)
                    
        return correlations
        
    def generate_threat_intelligence_report(self, 
                                          recent_events: List[Dict],
                                          trends: List[SecurityTrend],
                                          risk_assessment: RiskAssessment) -> str:
        """
        Generate threat intelligence report.
        
        Args:
            recent_events: Recent security events
            trends: Security trends
            risk_assessment: Risk assessment
            
        Returns:
            Threat intelligence report
        """
        report = [
            "# Security Threat Intelligence Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- Overall Risk Level: **{risk_assessment.risk_level.upper()}**",
            f"- Risk Score: {risk_assessment.risk_score:.1f}/100",
            f"- Active Threats: {len(recent_events)}",
            f"- Trending Metrics: {len(trends)}",
            "",
            "## Risk Assessment",
            ""
        ]
        
        for factor in risk_assessment.contributing_factors:
            report.append(f"- {factor}")
            
        report.extend([
            "",
            "## Security Trends",
            ""
        ])
        
        for trend in trends:
            direction_emoji = "📈" if trend.trend_direction == "increasing" else "📉" if trend.trend_direction == "decreasing" else "➡️"
            report.append(f"- {direction_emoji} **{trend.metric_name}**: {trend.trend_direction} (confidence: {trend.confidence:.1f})")
            
        report.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        for rec in risk_assessment.recommendations:
            report.append(f"- {rec}")
            
        return "\n".join(report)
        
    # Private methods
    def _calculate_trend(self, metric_name: str, data_points: List[Dict]) -> SecurityTrend:
        """Calculate trend for a metric."""
        values = [dp['value'] for dp in sorted(data_points, key=lambda x: x['timestamp'])]
        
        if len(values) < 2:
            return SecurityTrend(metric_name, "stable", 0.5, 0, values[-1])
            
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0
            
        # Determine trend direction
        if abs(slope) < 0.1:
            direction = "stable"
            confidence = 0.8
        elif slope > 0:
            direction = "increasing"
            confidence = min(0.95, 0.5 + abs(slope) * 0.1)
        else:
            direction = "decreasing"
            confidence = min(0.95, 0.5 + abs(slope) * 0.1)
            
        # Predict next value
        prediction = values[-1] + slope
        
        return SecurityTrend(
            metric_name=metric_name,
            trend_direction=direction,
            confidence=confidence,
            rate_of_change=slope,
            prediction=max(0, prediction)
        )
        
    def _calculate_anomaly_scores(self, values: List[float], sensitivity: float) -> List[float]:
        """Calculate anomaly scores using z-score method."""
        if len(values) < 3:
            return [0] * len(values)
            
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return [0] * len(values)
            
        z_scores = [abs((value - mean) / std) for value in values]
        return z_scores
        
    def _simple_forecast(self, values: List[float], days: int) -> List[float]:
        """Simple forecasting using linear trend."""
        if len(values) < 2:
            return [values[-1]] * days
            
        # Calculate trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Generate forecast
        forecast = []
        last_value = values[-1]
        
        for i in range(1, days + 1):
            predicted = last_value + (slope * i)
            forecast.append(max(0, predicted))
            
        return forecast
        
    def _analyze_event_correlation(self, events: List[Dict]) -> Optional[Dict[str, Any]]:
        """Analyze correlation within event window."""
        event_types = [event.get('type', 'unknown') for event in events]
        
        # Check for suspicious patterns
        if len(set(event_types)) > 1:  # Multiple event types
            return {
                'pattern_type': 'multi_event_sequence',
                'events': len(events),
                'event_types': list(set(event_types)),
                'time_span': self._calculate_time_span(events),
                'severity': 'high' if len(events) > 5 else 'medium'
            }
            
        return None
        
    def _calculate_time_span(self, events: List[Dict]) -> str:
        """Calculate time span of events."""
        timestamps = [datetime.fromisoformat(event['timestamp']) for event in events]
        time_span = max(timestamps) - min(timestamps)
        return str(time_span)
        
    def _generate_risk_recommendations(self, 
                                     risk_level: str,
                                     vuln_count: int,
                                     compliance_score: float,
                                     threat_count: int,
                                     dep_risks: int) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []
        
        if risk_level == "critical":
            recommendations.append("URGENT: Implement immediate security measures")
            
        if vuln_count > 10:
            recommendations.append("Priority: Address high-severity vulnerabilities")
            
        if compliance_score < 70:
            recommendations.append("Improve compliance posture immediately")
            
        if threat_count > 3:
            recommendations.append("Investigate and respond to active threats")
            
        if dep_risks > 5:
            recommendations.append("Update vulnerable dependencies")
            
        # General recommendations
        recommendations.extend([
            "Implement continuous security monitoring",
            "Regular security training for development team",
            "Establish incident response procedures"
        ])
        
        return recommendations