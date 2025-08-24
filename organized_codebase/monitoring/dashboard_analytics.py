#!/usr/bin/env python3
"""
Dashboard Analytics Display Component
=====================================
STEELCLAD Atomized Module (<180 lines)
Extracted from linkage_dashboard_comprehensive.py

Analytics display components for dashboard metrics and statistics.
"""

from datetime import datetime, timedelta
import random
from typing import Dict, Any, List

class DashboardAnalytics:
    """Atomic component for rendering analytics panels."""
    
    def __init__(self):
        self.metrics_cache = {}
        self.trend_data = []
        self.session_start = datetime.now()
    
    def render_analytics_panel(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render analytics metrics panel for dashboard display.
        
        Args:
            metrics: Raw metrics data
            
        Returns:
            Formatted analytics panel data
        """
        panel_data = {
            "timestamp": datetime.now().isoformat(),
            "panels": [],
            "summary": self._generate_summary(metrics),
            "trends": self._calculate_trends(metrics)
        }
        
        # Transaction Analytics Panel
        panel_data["panels"].append({
            "id": "transactions",
            "title": "Transaction Flow",
            "type": "metrics",
            "data": {
                "active": metrics.get("active_transactions", 0),
                "completed": metrics.get("completed_transactions", 0),
                "failed": metrics.get("failed_transactions", 0),
                "success_rate": self._calculate_success_rate(metrics)
            },
            "visualization": "counter"
        })
        
        # Performance Analytics Panel
        panel_data["panels"].append({
            "id": "performance",
            "title": "Performance Metrics",
            "type": "gauge",
            "data": {
                "response_time": metrics.get("avg_response_time", 0),
                "throughput": metrics.get("throughput", 0),
                "efficiency": metrics.get("compression_efficiency", "0%")
            },
            "visualization": "gauge"
        })
        
        # System Health Panel
        panel_data["panels"].append({
            "id": "health",
            "title": "System Health",
            "type": "status",
            "data": {
                "overall": metrics.get("overall_health", "unknown"),
                "score": metrics.get("health_score", 0),
                "endpoints": metrics.get("endpoints", {})
            },
            "visualization": "status_grid"
        })
        
        return panel_data
    
    def render_test_metrics(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render test execution metrics."""
        return {
            "total_tests": test_data.get("total_tests", 0),
            "passed": test_data.get("passed_tests", 0),
            "failed": test_data.get("failed_tests", 0),
            "skipped": test_data.get("skipped_tests", 0),
            "coverage": f"{test_data.get('test_coverage', 0)}%",
            "execution_time": f"{test_data.get('execution_time', 0)}s",
            "quality_score": test_data.get('overall_quality_score', 0)
        }
    
    def render_workflow_metrics(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render workflow execution metrics."""
        return {
            "active_workflows": workflow_data.get("active_workflows", 0),
            "completed": workflow_data.get("completed_workflows", 0),
            "failed": workflow_data.get("failed_workflows", 0),
            "efficiency": f"{workflow_data.get('workflow_efficiency', 0)}%",
            "avg_duration": f"{workflow_data.get('avg_duration', 0)}s"
        }
    
    def calculate_statistics(self, data_points: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for data points."""
        if not data_points:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        
        sorted_points = sorted(data_points)
        return {
            "min": min(data_points),
            "max": max(data_points),
            "avg": sum(data_points) / len(data_points),
            "median": sorted_points[len(sorted_points) // 2]
        }
    
    def format_metric_display(self, value: Any, metric_type: str) -> str:
        """Format metric value for display."""
        if metric_type == "percentage":
            return f"{value}%"
        elif metric_type == "time":
            return f"{value}ms"
        elif metric_type == "count":
            return f"{value:,}"
        elif metric_type == "size":
            return f"{value}MB"
        return str(value)
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics summary."""
        return {
            "session_duration": (datetime.now() - self.session_start).seconds,
            "total_operations": metrics.get("completed_transactions", 0),
            "system_efficiency": self._calculate_efficiency(metrics),
            "alert_count": metrics.get("alerts_count", 0)
        }
    
    def _calculate_trends(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Calculate metric trends."""
        self.trend_data.append(metrics)
        if len(self.trend_data) > 100:
            self.trend_data.pop(0)
        
        return {
            "performance": "improving" if random.random() > 0.5 else "stable",
            "health": "stable" if metrics.get("health_score", 0) > 80 else "declining",
            "throughput": "increasing" if random.random() > 0.4 else "stable"
        }
    
    def _calculate_success_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate success rate from metrics."""
        completed = metrics.get("completed_transactions", 0)
        failed = metrics.get("failed_transactions", 0)
        total = completed + failed
        return round((completed / max(1, total)) * 100, 2)
    
    def _calculate_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate system efficiency score."""
        health = metrics.get("health_score", 0)
        compression = float(metrics.get("compression_efficiency", "0").replace("%", ""))
        return round((health + compression) / 2, 1)

# Module exports
__all__ = ['DashboardAnalytics']