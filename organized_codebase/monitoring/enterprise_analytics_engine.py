#!/usr/bin/env python3
"""
Enterprise Analytics Engine
Agent B Hours 110-120: Enterprise Integration & Advanced Analytics

Advanced analytics engine with machine learning capabilities and business intelligence features.
"""

import json
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import statistics

@dataclass
class AnalyticsInsight:
    """Analytics insight data structure"""
    id: str
    type: str
    category: str  # 'performance', 'capacity', 'optimization', 'anomaly'
    title: str
    description: str
    impact: str  # 'high', 'medium', 'low'
    confidence: float  # 0-1
    recommendation: str
    metrics: Dict[str, float]
    timestamp: datetime

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    trend_strength: float  # 0-1
    prediction_30_days: float
    prediction_confidence: float
    seasonal_patterns: List[Dict[str, Any]]
    anomalies_detected: List[Dict[str, Any]]

@dataclass
class PerformanceBaseline:
    """Performance baseline data"""
    metric_name: str
    baseline_value: float
    upper_threshold: float
    lower_threshold: float
    last_updated: datetime
    sample_count: int
    confidence_interval: Tuple[float, float]

class EnterpriseAnalyticsEngine:
    """Advanced analytics engine with ML-like capabilities for enterprise monitoring"""
    
    def __init__(self, data_file: str = "analytics_data.json"):
        self.data_file = Path(data_file)
        self.metrics_history = deque(maxlen=10000)  # Store up to 10,000 data points
        self.insights_history = []
        self.performance_baselines = {}
        self.trend_models = {}
        
        # Analytics configuration
        self.config = {
            'anomaly_sensitivity': 0.8,  # 0-1, higher = more sensitive
            'trend_window_hours': 168,   # 7 days for trend analysis
            'baseline_window_hours': 720, # 30 days for baseline calculation
            'prediction_confidence_threshold': 0.7,
            'seasonal_analysis_enabled': True
        }
        
        # Load existing data
        self.load_analytics_data()
        
        # Initialize analytics modules
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.capacity_planner = CapacityPlanner()
        self.performance_analyzer = PerformanceAnalyzer()
        
        print("[OK] Enterprise Analytics Engine initialized")
        print(f"[OK] Loaded {len(self.metrics_history)} historical data points")
    
    def load_analytics_data(self):
        """Load historical analytics data"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load metrics history
                for metric_data in data.get('metrics_history', []):
                    metric_data['timestamp'] = datetime.fromisoformat(metric_data['timestamp'])
                    self.metrics_history.append(metric_data)
                
                # Load performance baselines
                for baseline_name, baseline_data in data.get('baselines', {}).items():
                    baseline_data['last_updated'] = datetime.fromisoformat(baseline_data['last_updated'])
                    self.performance_baselines[baseline_name] = PerformanceBaseline(**baseline_data)
                
                # Load insights history
                for insight_data in data.get('insights', []):
                    insight_data['timestamp'] = datetime.fromisoformat(insight_data['timestamp'])
                    self.insights_history.append(AnalyticsInsight(**insight_data))
                    
            except Exception as e:
                print(f"[WARNING] Failed to load analytics data: {e}")
    
    def save_analytics_data(self):
        """Save analytics data to file"""
        try:
            # Convert to serializable format
            serializable_data = {
                'metrics_history': [],
                'baselines': {},
                'insights': []
            }
            
            # Save recent metrics history (last 1000 points)
            recent_metrics = list(self.metrics_history)[-1000:]
            for metric in recent_metrics:
                metric_dict = dict(metric)
                metric_dict['timestamp'] = metric_dict['timestamp'].isoformat()
                serializable_data['metrics_history'].append(metric_dict)
            
            # Save baselines
            for name, baseline in self.performance_baselines.items():
                baseline_dict = asdict(baseline)
                baseline_dict['last_updated'] = baseline_dict['last_updated'].isoformat()
                serializable_data['baselines'][name] = baseline_dict
            
            # Save recent insights (last 100)
            recent_insights = self.insights_history[-100:]
            for insight in recent_insights:
                insight_dict = asdict(insight)
                insight_dict['timestamp'] = insight_dict['timestamp'].isoformat()
                serializable_data['insights'].append(insight_dict)
            
            with open(self.data_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
        except Exception as e:
            print(f"[ERROR] Failed to save analytics data: {e}")
    
    def ingest_metrics(self, metrics: Dict[str, Any]):
        """Ingest new metrics for analysis"""
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now()
        elif isinstance(metrics['timestamp'], str):
            metrics['timestamp'] = datetime.fromisoformat(metrics['timestamp'])
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Update performance baselines
        self._update_performance_baselines(metrics)
        
        # Trigger real-time analysis
        new_insights = self._perform_real_time_analysis(metrics)
        
        # Save data periodically
        if len(self.metrics_history) % 50 == 0:  # Save every 50 metrics
            self.save_analytics_data()
        
        return new_insights
    
    def _update_performance_baselines(self, metrics: Dict[str, Any]):
        """Update performance baselines with new metrics"""
        # Extract key metrics for baseline tracking
        key_metrics = {
            'cpu_percent': metrics.get('system', {}).get('cpu_percent', 0),
            'memory_percent': metrics.get('system', {}).get('memory_percent', 0),
            'disk_percent': metrics.get('system', {}).get('disk_percent', 0),
            'database_size_mb': metrics.get('totals', {}).get('database_size_mb', 0),
            'query_count': metrics.get('totals', {}).get('query_count', 0)
        }
        
        current_time = datetime.now()
        
        for metric_name, value in key_metrics.items():
            if value == 0:  # Skip zero values
                continue
                
            if metric_name not in self.performance_baselines:
                # Create new baseline
                self.performance_baselines[metric_name] = PerformanceBaseline(
                    metric_name=metric_name,
                    baseline_value=value,
                    upper_threshold=value * 1.2,
                    lower_threshold=value * 0.8,
                    last_updated=current_time,
                    sample_count=1,
                    confidence_interval=(value * 0.9, value * 1.1)
                )
            else:
                # Update existing baseline
                baseline = self.performance_baselines[metric_name]
                baseline.sample_count += 1
                
                # Exponential moving average for baseline
                alpha = 0.1  # Smoothing factor
                baseline.baseline_value = (alpha * value) + ((1 - alpha) * baseline.baseline_value)
                
                # Update thresholds based on variance
                recent_values = [m.get('system', {}).get(metric_name.replace('_mb', '_percent'), 
                                                          m.get('totals', {}).get(metric_name, 0)) 
                                for m in list(self.metrics_history)[-100:]]  # Last 100 values
                recent_values = [v for v in recent_values if v > 0]
                
                if len(recent_values) >= 10:
                    std_dev = statistics.stdev(recent_values)
                    baseline.upper_threshold = baseline.baseline_value + (2 * std_dev)
                    baseline.lower_threshold = max(0, baseline.baseline_value - (2 * std_dev))
                
                baseline.last_updated = current_time
    
    def _perform_real_time_analysis(self, metrics: Dict[str, Any]) -> List[AnalyticsInsight]:
        """Perform real-time analysis on new metrics"""
        insights = []
        
        # Anomaly detection
        anomalies = self.anomaly_detector.detect_anomalies(metrics, self.performance_baselines)
        for anomaly in anomalies:
            insight = AnalyticsInsight(
                id=f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(insights)}",
                type="anomaly",
                category="performance",
                title=f"Anomaly Detected: {anomaly['metric']}",
                description=f"Value {anomaly['value']:.2f} is {anomaly['deviation']:.1f}x higher than baseline {anomaly['baseline']:.2f}",
                impact=anomaly['severity'],
                confidence=anomaly['confidence'],
                recommendation=anomaly['recommendation'],
                metrics={anomaly['metric']: anomaly['value']},
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # Performance degradation detection
        performance_issues = self.performance_analyzer.analyze_performance_degradation(list(self.metrics_history)[-50:])
        for issue in performance_issues:
            insight = AnalyticsInsight(
                id=f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(insights)}",
                type="performance_degradation",
                category="performance",
                title=f"Performance Degradation: {issue['area']}",
                description=issue['description'],
                impact=issue['impact'],
                confidence=issue['confidence'],
                recommendation=issue['recommendation'],
                metrics=issue['metrics'],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # Store insights
        self.insights_history.extend(insights)
        
        return insights
    
    def generate_comprehensive_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        if len(self.metrics_history) < 10:
            return {"message": "Insufficient data for comprehensive analytics"}
        
        # Trend analysis
        trend_analysis = self.analyze_trends()
        
        # Capacity planning
        capacity_forecast = self.capacity_planner.generate_capacity_forecast(list(self.metrics_history))
        
        # Performance analysis
        performance_summary = self.performance_analyzer.generate_performance_summary(list(self.metrics_history))
        
        # Anomaly summary
        recent_anomalies = [insight for insight in self.insights_history[-100:] 
                          if insight.type == "anomaly" and 
                          insight.timestamp > datetime.now() - timedelta(hours=24)]
        
        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities()
        
        # Business intelligence insights
        business_insights = self._generate_business_insights()
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "data_points_analyzed": len(self.metrics_history),
            "analysis_period_hours": self._get_analysis_period_hours(),
            "trend_analysis": trend_analysis,
            "capacity_forecast": capacity_forecast,
            "performance_summary": performance_summary,
            "anomaly_summary": {
                "total_anomalies_24h": len(recent_anomalies),
                "anomaly_types": list(set(a.category for a in recent_anomalies)),
                "highest_impact_anomalies": [asdict(a) for a in recent_anomalies if a.impact == "high"]
            },
            "optimization_opportunities": optimization_opportunities,
            "business_insights": business_insights,
            "system_health_score": self._calculate_enterprise_health_score(),
            "recommendations": self._generate_enterprise_recommendations()
        }
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends across all metrics"""
        if len(self.metrics_history) < 20:
            return {"message": "Insufficient data for trend analysis"}
        
        trends = {}
        
        # Key metrics to analyze
        metric_extractors = {
            'cpu_utilization': lambda m: m.get('system', {}).get('cpu_percent', 0),
            'memory_utilization': lambda m: m.get('system', {}).get('memory_percent', 0),
            'database_growth': lambda m: m.get('totals', {}).get('database_size_mb', 0),
            'query_performance': lambda m: m.get('totals', {}).get('query_count', 0)
        }
        
        for metric_name, extractor in metric_extractors.items():
            values = [extractor(m) for m in list(self.metrics_history)[-200:]]  # Last 200 points
            values = [v for v in values if v > 0]  # Remove zero values
            
            if len(values) < 10:
                continue
            
            trend_analysis = self.trend_analyzer.analyze_metric_trend(metric_name, values)
            trends[metric_name] = trend_analysis
        
        return trends
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities based on analytics"""
        opportunities = []
        
        if len(self.metrics_history) < 50:
            return opportunities
        
        recent_metrics = list(self.metrics_history)[-50:]
        
        # CPU optimization opportunities
        cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in recent_metrics]
        cpu_avg = statistics.mean([v for v in cpu_values if v > 0])
        
        if cpu_avg > 70:
            opportunities.append({
                "type": "cpu_optimization",
                "priority": "high" if cpu_avg > 85 else "medium",
                "description": f"Average CPU utilization is {cpu_avg:.1f}% over recent period",
                "recommendation": "Consider CPU optimization or scaling",
                "potential_savings": f"{max(0, cpu_avg - 50):.1f}% CPU reduction possible",
                "estimated_impact": "high"
            })
        
        # Memory optimization opportunities
        memory_values = [m.get('system', {}).get('memory_percent', 0) for m in recent_metrics]
        memory_avg = statistics.mean([v for v in memory_values if v > 0])
        
        if memory_avg > 80:
            opportunities.append({
                "type": "memory_optimization",
                "priority": "high" if memory_avg > 90 else "medium",
                "description": f"Average memory utilization is {memory_avg:.1f}% over recent period",
                "recommendation": "Consider memory optimization or expansion",
                "potential_savings": f"{max(0, memory_avg - 70):.1f}% memory reduction possible",
                "estimated_impact": "high"
            })
        
        # Database growth optimization
        db_sizes = [m.get('totals', {}).get('database_size_mb', 0) for m in recent_metrics]
        if db_sizes and max(db_sizes) > 100:
            opportunities.append({
                "type": "database_optimization",
                "priority": "medium",
                "description": f"Database size has grown to {max(db_sizes):.1f}MB",
                "recommendation": "Consider data archiving or compression",
                "potential_savings": "20-40% size reduction possible",
                "estimated_impact": "medium"
            })
        
        return opportunities
    
    def _generate_business_insights(self) -> List[Dict[str, Any]]:
        """Generate business intelligence insights"""
        insights = []
        
        if len(self.metrics_history) < 100:
            return insights
        
        # Cost analysis insights
        recent_metrics = list(self.metrics_history)[-168:]  # Last week
        
        # Resource utilization efficiency
        cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in recent_metrics]
        cpu_efficiency = statistics.mean([v for v in cpu_values if v > 0])
        
        if cpu_efficiency < 30:
            insights.append({
                "type": "resource_efficiency",
                "category": "cost_optimization",
                "title": "Low Resource Utilization",
                "description": f"Average CPU utilization is only {cpu_efficiency:.1f}%",
                "business_impact": "Potential over-provisioning of resources",
                "cost_implications": "Consider downsizing to reduce costs",
                "confidence": 0.8
            })
        elif cpu_efficiency > 85:
            insights.append({
                "type": "resource_constraint",
                "category": "performance_risk",
                "title": "High Resource Utilization",
                "description": f"Average CPU utilization is {cpu_efficiency:.1f}%",
                "business_impact": "Risk of performance degradation during peak loads",
                "cost_implications": "Consider scaling up to maintain performance SLA",
                "confidence": 0.9
            })
        
        # Growth rate analysis
        db_sizes = [m.get('totals', {}).get('database_size_mb', 0) for m in recent_metrics]
        if len(db_sizes) > 50 and max(db_sizes) > min(db_sizes):
            growth_rate = (max(db_sizes) - min(db_sizes)) / len(db_sizes) * 24  # MB per day
            if growth_rate > 10:  # Growing more than 10MB per day
                insights.append({
                    "type": "capacity_planning",
                    "category": "business_planning",
                    "title": "Rapid Data Growth",
                    "description": f"Database growing at {growth_rate:.1f}MB per day",
                    "business_impact": "Need for capacity planning and budget allocation",
                    "cost_implications": f"Estimated storage needs: {growth_rate * 365:.0f}MB annually",
                    "confidence": 0.7
                })
        
        return insights
    
    def _calculate_enterprise_health_score(self) -> int:
        """Calculate enterprise-grade system health score"""
        if not self.metrics_history:
            return 50
        
        score = 100
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 data points
        
        # Performance factors
        cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in recent_metrics]
        memory_values = [m.get('system', {}).get('memory_percent', 0) for m in recent_metrics]
        
        if cpu_values:
            avg_cpu = statistics.mean(cpu_values)
            if avg_cpu > 90:
                score -= 30
            elif avg_cpu > 80:
                score -= 20
            elif avg_cpu > 70:
                score -= 10
        
        if memory_values:
            avg_memory = statistics.mean(memory_values)
            if avg_memory > 95:
                score -= 25
            elif avg_memory > 85:
                score -= 15
            elif avg_memory > 75:
                score -= 8
        
        # Anomaly factors
        recent_anomalies = [i for i in self.insights_history[-50:] 
                          if i.type == "anomaly" and 
                          i.timestamp > datetime.now() - timedelta(hours=6)]
        
        score -= len(recent_anomalies) * 5  # -5 points per anomaly
        
        # High-impact anomalies
        critical_anomalies = [a for a in recent_anomalies if a.impact == "high"]
        score -= len(critical_anomalies) * 10  # Additional -10 points for critical anomalies
        
        # Trend factors
        if len(self.metrics_history) > 100:
            # Check for degrading trends
            recent_cpu = cpu_values[-10:] if len(cpu_values) >= 10 else cpu_values
            if len(recent_cpu) > 5 and recent_cpu[-1] > recent_cpu[0] * 1.2:
                score -= 15  # CPU trending upward significantly
        
        return max(0, min(100, score))
    
    def _generate_enterprise_recommendations(self) -> List[str]:
        """Generate enterprise-grade recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return ["Start collecting metrics to generate recommendations"]
        
        # Analyze recent performance
        recent_metrics = list(self.metrics_history)[-50:]
        
        # CPU recommendations
        cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in recent_metrics]
        if cpu_values:
            avg_cpu = statistics.mean(cpu_values)
            if avg_cpu > 80:
                recommendations.append(f"CPU utilization averaging {avg_cpu:.1f}% - consider performance optimization or scaling")
            elif avg_cpu < 20:
                recommendations.append(f"CPU utilization only {avg_cpu:.1f}% - consider rightsizing to reduce costs")
        
        # Memory recommendations
        memory_values = [m.get('system', {}).get('memory_percent', 0) for m in recent_metrics]
        if memory_values:
            avg_memory = statistics.mean(memory_values)
            if avg_memory > 85:
                recommendations.append(f"Memory utilization {avg_memory:.1f}% - risk of performance issues, consider memory optimization")
        
        # Anomaly-based recommendations
        recent_anomalies = [i for i in self.insights_history[-20:] if i.type == "anomaly"]
        if len(recent_anomalies) > 5:
            recommendations.append(f"Detected {len(recent_anomalies)} anomalies recently - investigate system stability")
        
        # Business recommendations
        if len(self.metrics_history) > 200:
            health_score = self._calculate_enterprise_health_score()
            if health_score < 70:
                recommendations.append(f"System health score is {health_score}/100 - comprehensive system review recommended")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters - continue monitoring")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _get_analysis_period_hours(self) -> float:
        """Get the analysis period in hours"""
        if len(self.metrics_history) < 2:
            return 0
        
        first_timestamp = self.metrics_history[0]['timestamp']
        last_timestamp = self.metrics_history[-1]['timestamp']
        
        return (last_timestamp - first_timestamp).total_seconds() / 3600

class AnomalyDetector:
    """Anomaly detection using statistical methods"""
    
    def detect_anomalies(self, metrics: Dict[str, Any], baselines: Dict[str, PerformanceBaseline]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics"""
        anomalies = []
        
        # Check system metrics
        system_metrics = metrics.get('system', {})
        
        for metric_name in ['cpu_percent', 'memory_percent', 'disk_percent']:
            if metric_name in system_metrics:
                value = system_metrics[metric_name]
                baseline_key = metric_name
                
                if baseline_key in baselines and value > 0:
                    baseline = baselines[baseline_key]
                    
                    # Check for anomalies
                    if value > baseline.upper_threshold:
                        deviation = value / baseline.baseline_value
                        severity = "high" if deviation > 2.0 else "medium"
                        confidence = min(0.95, deviation / 3.0)
                        
                        anomalies.append({
                            'metric': metric_name,
                            'value': value,
                            'baseline': baseline.baseline_value,
                            'threshold': baseline.upper_threshold,
                            'deviation': deviation,
                            'severity': severity,
                            'confidence': confidence,
                            'recommendation': f"Investigate {metric_name} spike - value {deviation:.1f}x higher than baseline"
                        })
        
        return anomalies

class TrendAnalyzer:
    """Trend analysis using statistical methods"""
    
    def analyze_metric_trend(self, metric_name: str, values: List[float]) -> Dict[str, Any]:
        """Analyze trend for a specific metric"""
        if len(values) < 10:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate basic trend
        x = list(range(len(values)))
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Determine trend direction and strength
        trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        trend_strength = min(1.0, abs(slope) / (statistics.mean(values) / len(values)))
        
        # Predict 30 days ahead (assuming measurements are hourly)
        prediction_30_days = intercept + slope * (len(values) + 30 * 24)
        prediction_confidence = max(0.3, min(0.9, 1.0 - trend_strength))
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "slope": slope,
            "current_value": values[-1],
            "prediction_30_days": max(0, prediction_30_days),
            "prediction_confidence": prediction_confidence,
            "data_points": len(values)
        }

class CapacityPlanner:
    """Capacity planning with forecasting"""
    
    def generate_capacity_forecast(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate capacity forecast"""
        if len(metrics_history) < 50:
            return {"error": "Insufficient data for capacity planning"}
        
        forecasts = {}
        
        # CPU capacity forecast
        cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in metrics_history[-200:]]
        cpu_values = [v for v in cpu_values if v > 0]
        
        if len(cpu_values) > 20:
            cpu_trend = self._calculate_simple_trend(cpu_values)
            forecasts['cpu'] = {
                "current_avg": statistics.mean(cpu_values[-20:]),
                "30_day_forecast": max(0, min(100, cpu_values[-1] + cpu_trend * 30 * 24)),
                "capacity_exhaustion_days": self._calculate_capacity_exhaustion(cpu_values, 90),
                "recommendation": self._get_capacity_recommendation("CPU", cpu_values[-1], cpu_trend)
            }
        
        # Memory capacity forecast
        memory_values = [m.get('system', {}).get('memory_percent', 0) for m in metrics_history[-200:]]
        memory_values = [v for v in memory_values if v > 0]
        
        if len(memory_values) > 20:
            memory_trend = self._calculate_simple_trend(memory_values)
            forecasts['memory'] = {
                "current_avg": statistics.mean(memory_values[-20:]),
                "30_day_forecast": max(0, min(100, memory_values[-1] + memory_trend * 30 * 24)),
                "capacity_exhaustion_days": self._calculate_capacity_exhaustion(memory_values, 95),
                "recommendation": self._get_capacity_recommendation("Memory", memory_values[-1], memory_trend)
            }
        
        # Database growth forecast
        db_sizes = [m.get('totals', {}).get('database_size_mb', 0) for m in metrics_history[-200:]]
        db_sizes = [v for v in db_sizes if v > 0]
        
        if len(db_sizes) > 20:
            db_trend = self._calculate_simple_trend(db_sizes)
            forecasts['database'] = {
                "current_size_mb": db_sizes[-1],
                "30_day_forecast_mb": max(0, db_sizes[-1] + db_trend * 30 * 24),
                "monthly_growth_mb": db_trend * 30 * 24,
                "recommendation": self._get_storage_recommendation(db_sizes[-1], db_trend)
            }
        
        return forecasts
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend"""
        if len(values) < 2:
            return 0
        
        # Use last 20% of values for trend calculation
        recent_count = max(10, len(values) // 5)
        recent_values = values[-recent_count:]
        
        return (recent_values[-1] - recent_values[0]) / len(recent_values)
    
    def _calculate_capacity_exhaustion(self, values: List[float], threshold: float) -> Optional[int]:
        """Calculate days until capacity threshold is reached"""
        if len(values) < 10:
            return None
        
        trend = self._calculate_simple_trend(values)
        if trend <= 0:
            return None  # Not trending upward
        
        current_value = values[-1]
        days_to_threshold = (threshold - current_value) / (trend * 24)  # Assuming hourly measurements
        
        return max(0, int(days_to_threshold)) if days_to_threshold > 0 else 0
    
    def _get_capacity_recommendation(self, resource: str, current_value: float, trend: float) -> str:
        """Get capacity recommendation"""
        if trend > 0.5:  # Significant upward trend
            if current_value > 80:
                return f"{resource} usage high ({current_value:.1f}%) with upward trend - immediate scaling recommended"
            else:
                return f"{resource} trending upward - monitor closely and plan for scaling"
        elif trend < -0.5:  # Significant downward trend
            return f"{resource} usage decreasing - potential cost optimization opportunity"
        else:
            return f"{resource} usage stable at {current_value:.1f}% - no immediate action needed"
    
    def _get_storage_recommendation(self, current_size: float, trend: float) -> str:
        """Get storage recommendation"""
        if trend > 10:  # Growing more than 10MB per day
            return f"Rapid database growth ({trend * 30:.0f}MB/month) - plan for storage expansion"
        elif trend > 1:
            return f"Moderate database growth - monitor storage capacity"
        else:
            return f"Database size stable - no storage concerns"

class PerformanceAnalyzer:
    """Performance analysis and optimization"""
    
    def analyze_performance_degradation(self, recent_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze for performance degradation patterns"""
        issues = []
        
        if len(recent_metrics) < 20:
            return issues
        
        # CPU performance degradation
        cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in recent_metrics]
        if len(cpu_values) >= 20:
            early_cpu = statistics.mean(cpu_values[:10])
            late_cpu = statistics.mean(cpu_values[-10:])
            
            if late_cpu > early_cpu * 1.5 and late_cpu > 60:  # 50% increase and above 60%
                issues.append({
                    'area': 'CPU Performance',
                    'description': f"CPU usage increased from {early_cpu:.1f}% to {late_cpu:.1f}%",
                    'impact': 'high' if late_cpu > 85 else 'medium',
                    'confidence': 0.8,
                    'recommendation': 'Investigate processes causing CPU spike',
                    'metrics': {'early_cpu': early_cpu, 'current_cpu': late_cpu}
                })
        
        # Memory performance degradation
        memory_values = [m.get('system', {}).get('memory_percent', 0) for m in recent_metrics]
        if len(memory_values) >= 20:
            early_memory = statistics.mean(memory_values[:10])
            late_memory = statistics.mean(memory_values[-10:])
            
            if late_memory > early_memory * 1.3 and late_memory > 70:  # 30% increase and above 70%
                issues.append({
                    'area': 'Memory Performance',
                    'description': f"Memory usage increased from {early_memory:.1f}% to {late_memory:.1f}%",
                    'impact': 'high' if late_memory > 90 else 'medium',
                    'confidence': 0.8,
                    'recommendation': 'Check for memory leaks or increase memory allocation',
                    'metrics': {'early_memory': early_memory, 'current_memory': late_memory}
                })
        
        return issues
    
    def generate_performance_summary(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary"""
        if len(metrics_history) < 10:
            return {"error": "Insufficient data for performance analysis"}
        
        # Recent performance (last 50 data points)
        recent_metrics = metrics_history[-50:]
        
        # CPU performance
        cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in recent_metrics]
        cpu_values = [v for v in cpu_values if v > 0]
        
        cpu_summary = {
            "average": statistics.mean(cpu_values) if cpu_values else 0,
            "peak": max(cpu_values) if cpu_values else 0,
            "minimum": min(cpu_values) if cpu_values else 0,
            "volatility": statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
        }
        
        # Memory performance
        memory_values = [m.get('system', {}).get('memory_percent', 0) for m in recent_metrics]
        memory_values = [v for v in memory_values if v > 0]
        
        memory_summary = {
            "average": statistics.mean(memory_values) if memory_values else 0,
            "peak": max(memory_values) if memory_values else 0,
            "minimum": min(memory_values) if memory_values else 0,
            "volatility": statistics.stdev(memory_values) if len(memory_values) > 1 else 0
        }
        
        # Overall performance score
        performance_score = self._calculate_performance_score(cpu_summary, memory_summary)
        
        return {
            "analysis_period": "Recent 50 measurements",
            "cpu_performance": cpu_summary,
            "memory_performance": memory_summary,
            "overall_performance_score": performance_score,
            "performance_grade": self._get_performance_grade(performance_score)
        }
    
    def _calculate_performance_score(self, cpu_summary: Dict, memory_summary: Dict) -> int:
        """Calculate overall performance score"""
        score = 100
        
        # CPU factors
        cpu_avg = cpu_summary.get('average', 0)
        if cpu_avg > 90:
            score -= 40
        elif cpu_avg > 80:
            score -= 30
        elif cpu_avg > 70:
            score -= 20
        elif cpu_avg > 60:
            score -= 10
        
        # Memory factors
        memory_avg = memory_summary.get('average', 0)
        if memory_avg > 95:
            score -= 35
        elif memory_avg > 85:
            score -= 25
        elif memory_avg > 75:
            score -= 15
        elif memory_avg > 65:
            score -= 8
        
        # Volatility penalty
        cpu_volatility = cpu_summary.get('volatility', 0)
        memory_volatility = memory_summary.get('volatility', 0)
        
        score -= min(15, (cpu_volatility + memory_volatility) / 2)
        
        return max(0, min(100, int(score)))
    
    def _get_performance_grade(self, score: int) -> str:
        """Get performance grade based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Critical"

def main():
    """Main function for testing analytics engine"""
    engine = EnterpriseAnalyticsEngine()
    
    print("[OK] Enterprise Analytics Engine ready for testing")
    
    # Simulate some metrics data
    test_metrics = []
    base_time = datetime.now() - timedelta(hours=48)
    
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i * 30)
        
        # Simulate realistic metrics with some trends and anomalies
        cpu_base = 40 + (i * 0.1) + (10 * np.sin(i * 0.1))  # Trending up with oscillation
        memory_base = 60 + (i * 0.05) + (5 * np.sin(i * 0.15))  # Slight upward trend
        
        # Add some anomalies
        if i == 30 or i == 75:
            cpu_base += 50  # CPU spikes
        if i == 60:
            memory_base += 30  # Memory spike
        
        metric = {
            'timestamp': timestamp,
            'system': {
                'cpu_percent': max(0, min(100, cpu_base)),
                'memory_percent': max(0, min(100, memory_base)),
                'disk_percent': 45 + (i * 0.02)
            },
            'totals': {
                'database_size_mb': 50 + (i * 0.5),
                'query_count': 10 + int(i * 0.1)
            }
        }
        
        # Ingest metric and collect insights
        insights = engine.ingest_metrics(metric)
        if insights:
            print(f"[INSIGHT] Generated {len(insights)} insights at data point {i}")
    
    # Generate comprehensive report
    print("\n[OK] Generating comprehensive analytics report...")
    report = engine.generate_comprehensive_analytics_report()
    
    print("\n" + "="*60)
    print("ENTERPRISE ANALYTICS REPORT")
    print("="*60)
    
    print(f"\nData Points Analyzed: {report['data_points_analyzed']}")
    print(f"Analysis Period: {report['analysis_period_hours']:.1f} hours")
    print(f"System Health Score: {report['system_health_score']}/100")
    
    # Trend analysis
    if 'trend_analysis' in report:
        print("\nTREND ANALYSIS:")
        for metric, trend in report['trend_analysis'].items():
            if 'error' not in trend:
                print(f"  {metric}: {trend['trend_direction']} (strength: {trend['trend_strength']:.2f})")
                print(f"    30-day forecast: {trend.get('prediction_30_days', 'N/A')}")
    
    # Optimization opportunities
    if report['optimization_opportunities']:
        print(f"\nOPTIMIZATION OPPORTUNITIES ({len(report['optimization_opportunities'])}):")
        for opp in report['optimization_opportunities']:
            print(f"  [{opp['priority'].upper()}] {opp['type']}: {opp['description']}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    print(f"\n[OK] Analytics engine test completed successfully!")

if __name__ == "__main__":
    main()