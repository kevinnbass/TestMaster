"""
Telemetry Dashboard for TestMaster

Comprehensive telemetry dashboard and reporting system
that aggregates data from all telemetry components.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import uuid

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from .telemetry_collector import get_telemetry_collector
from .performance_monitor import get_performance_monitor
from .flow_analyzer import get_flow_analyzer
from .system_profiler import get_system_profiler

@dataclass
class TelemetryReport:
    """Comprehensive telemetry report."""
    report_id: str
    generated_at: datetime
    timeframe_hours: int
    summary: Dict[str, Any]
    telemetry_data: Dict[str, Any]
    performance_data: Dict[str, Any]
    flow_analysis: Dict[str, Any]
    system_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    health_score: int

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    auto_refresh_interval: int = 300  # 5 minutes
    metrics_retention_hours: int = 168  # 1 week
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate_percent": 5.0,
        "avg_response_time_ms": 5000.0,
        "cpu_percent": 80.0,
        "memory_percent": 85.0
    })
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])

class TelemetryDashboard:
    """
    Comprehensive telemetry dashboard for TestMaster.
    
    Aggregates and presents data from:
    - Telemetry collector (events and usage)
    - Performance monitor (execution metrics)
    - Flow analyzer (execution patterns)
    - System profiler (resource usage)
    """
    
    def __init__(self, config: DashboardConfig = None):
        """
        Initialize telemetry dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system')
        
        if not self.enabled:
            return
        
        self.config = config or DashboardConfig()
        
        # Component references
        self.telemetry_collector = get_telemetry_collector()
        self.performance_monitor = get_performance_monitor()
        self.flow_analyzer = get_flow_analyzer()
        self.system_profiler = get_system_profiler()
        
        # Data storage
        self.reports: List[TelemetryReport] = []
        self.dashboard_cache: Dict[str, Any] = {}
        self.last_refresh: Optional[datetime] = None
        
        # Threading
        self.lock = threading.RLock()
        self.refresh_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Integration with shared state
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        # Statistics
        self.reports_generated = 0
        self.dashboard_refreshes = 0
        
        # Start auto-refresh
        self._start_refresh_thread()
        
        print("Telemetry dashboard initialized")
        print(f"   Auto-refresh: {self.config.auto_refresh_interval}s")
        print(f"   Retention: {self.config.metrics_retention_hours}h")
    
    def generate_report(self, timeframe_hours: int = 24) -> TelemetryReport:
        """
        Generate comprehensive telemetry report.
        
        Args:
            timeframe_hours: Timeframe for analysis in hours
            
        Returns:
            Complete telemetry report
        """
        if not self.enabled:
            return TelemetryReport(
                report_id="disabled",
                generated_at=datetime.now(),
                timeframe_hours=timeframe_hours,
                summary={},
                telemetry_data={},
                performance_data={},
                flow_analysis={},
                system_metrics={},
                alerts=[],
                recommendations=[],
                health_score=0
            )
        
        report_id = str(uuid.uuid4())
        generated_at = datetime.now()
        
        # Collect data from all components
        telemetry_data = self._collect_telemetry_data(timeframe_hours)
        performance_data = self._collect_performance_data(timeframe_hours)
        flow_analysis = self._collect_flow_analysis(timeframe_hours)
        system_metrics = self._collect_system_metrics(timeframe_hours)
        
        # Generate alerts
        alerts = self._generate_alerts(telemetry_data, performance_data, flow_analysis, system_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(telemetry_data, performance_data, flow_analysis, system_metrics)
        
        # Calculate health score
        health_score = self._calculate_health_score(telemetry_data, performance_data, flow_analysis, system_metrics)
        
        # Create summary
        summary = self._create_summary(telemetry_data, performance_data, flow_analysis, system_metrics)
        
        report = TelemetryReport(
            report_id=report_id,
            generated_at=generated_at,
            timeframe_hours=timeframe_hours,
            summary=summary,
            telemetry_data=telemetry_data,
            performance_data=performance_data,
            flow_analysis=flow_analysis,
            system_metrics=system_metrics,
            alerts=alerts,
            recommendations=recommendations,
            health_score=health_score
        )
        
        # Store report
        with self.lock:
            self.reports.append(report)
            self.reports_generated += 1
            
            # Keep only recent reports
            if len(self.reports) > 100:
                self.reports = self.reports[-50:]
        
        # Update shared state
        if self.shared_state:
            self.shared_state.set("telemetry_reports_generated", self.reports_generated)
            self.shared_state.set("telemetry_health_score", health_score)
        
        return report
    
    def _collect_telemetry_data(self, timeframe_hours: int) -> Dict[str, Any]:
        """Collect data from telemetry collector."""
        try:
            stats = self.telemetry_collector.get_statistics()
            
            # Get recent events
            cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
            all_events = self.telemetry_collector.get_events()
            recent_events = [
                event for event in all_events
                if event.timestamp >= cutoff_time
            ]
            
            # Analyze events
            event_types = {}
            components = {}
            success_count = 0
            
            for event in recent_events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
                components[event.component] = components.get(event.component, 0) + 1
                if event.success:
                    success_count += 1
            
            return {
                "enabled": stats.get("enabled", False),
                "events_collected": stats.get("events_collected", 0),
                "recent_events": len(recent_events),
                "event_types": event_types,
                "components": components,
                "success_rate": (success_count / len(recent_events) * 100) if recent_events else 100,
                "collection_timeframe": timeframe_hours
            }
        except Exception as e:
            return {"error": str(e), "enabled": False}
    
    def _collect_performance_data(self, timeframe_hours: int) -> Dict[str, Any]:
        """Collect data from performance monitor."""
        try:
            # Get component statistics
            component_stats = self.performance_monitor.get_component_stats()
            
            # Get bottlenecks
            bottlenecks = self.performance_monitor.get_bottlenecks()
            
            # Get trends
            trends = self.performance_monitor.get_performance_trends(timeframe_hours=timeframe_hours)
            
            # Get active operations
            active_operations = self.performance_monitor.get_active_operations()
            
            # Summarize component performance
            total_operations = sum(stats.total_operations for stats in component_stats.values())
            avg_duration = (
                sum(stats.avg_duration_ms * stats.total_operations for stats in component_stats.values()) / 
                max(total_operations, 1)
            )
            error_rate = (
                sum(stats.failed_operations for stats in component_stats.values()) / 
                max(total_operations, 1) * 100
            )
            
            return {
                "enabled": self.performance_monitor.enabled,
                "total_operations": total_operations,
                "avg_duration_ms": round(avg_duration, 2),
                "error_rate": round(error_rate, 2),
                "component_count": len(component_stats),
                "bottlenecks_detected": len(bottlenecks),
                "active_operations": len(active_operations),
                "component_stats": {name: asdict(stats) for name, stats in component_stats.items()},
                "bottlenecks": bottlenecks,
                "trends": trends
            }
        except Exception as e:
            return {"error": str(e), "enabled": False}
    
    def _collect_flow_analysis(self, timeframe_hours: int) -> Dict[str, Any]:
        """Collect data from flow analyzer."""
        try:
            # Get flow statistics
            flow_stats = self.flow_analyzer.get_flow_statistics()
            
            # Perform flow analysis
            analysis = self.flow_analyzer.analyze_flows(timeframe_hours)
            
            return {
                "enabled": flow_stats.get("enabled", False),
                "flows_analyzed": flow_stats.get("flows_analyzed", 0),
                "active_flows": flow_stats.get("active_flows", 0),
                "completed_flows": flow_stats.get("completed_flows", 0),
                "analysis": asdict(analysis) if analysis else {},
                "bottlenecks_detected": len(analysis.bottleneck_components) if analysis else 0
            }
        except Exception as e:
            return {"error": str(e), "enabled": False}
    
    def _collect_system_metrics(self, timeframe_hours: int) -> Dict[str, Any]:
        """Collect data from system profiler."""
        try:
            # Get profiler statistics
            profiler_stats = self.system_profiler.get_profiler_statistics()
            
            # Get current metrics
            current_metrics = self.system_profiler.get_current_metrics()
            
            # Get active alerts
            active_alerts = self.system_profiler.get_active_alerts()
            
            # Get resource trends
            trends = {}
            for resource in ["cpu_percent", "memory_percent", "disk_used_percent"]:
                trends[resource] = self.system_profiler.get_resource_trends(resource, timeframe_hours)
            
            return {
                "enabled": profiler_stats.get("enabled", False),
                "is_monitoring": profiler_stats.get("is_monitoring", False),
                "metrics_collected": profiler_stats.get("metrics_collected", 0),
                "active_alerts": len(active_alerts),
                "current_metrics": asdict(current_metrics) if current_metrics else {},
                "trends": trends,
                "alerts": [asdict(alert) for alert in active_alerts]
            }
        except Exception as e:
            return {"error": str(e), "enabled": False}
    
    def _generate_alerts(self, telemetry_data: Dict[str, Any], performance_data: Dict[str, Any],
                        flow_analysis: Dict[str, Any], system_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on collected data."""
        alerts = []
        
        # Performance alerts
        if performance_data.get("enabled", False):
            error_rate = performance_data.get("error_rate", 0)
            if error_rate > self.config.alert_thresholds["error_rate_percent"]:
                alerts.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": f"High error rate: {error_rate:.1f}%",
                    "threshold": self.config.alert_thresholds["error_rate_percent"]
                })
            
            avg_duration = performance_data.get("avg_duration_ms", 0)
            if avg_duration > self.config.alert_thresholds["avg_response_time_ms"]:
                alerts.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": f"High average response time: {avg_duration:.1f}ms",
                    "threshold": self.config.alert_thresholds["avg_response_time_ms"]
                })
        
        # System resource alerts
        if system_metrics.get("enabled", False) and system_metrics.get("current_metrics"):
            current = system_metrics["current_metrics"]
            
            cpu_percent = current.get("cpu_percent", 0)
            if cpu_percent > self.config.alert_thresholds["cpu_percent"]:
                alerts.append({
                    "type": "system",
                    "severity": "warning",
                    "message": f"High CPU usage: {cpu_percent:.1f}%",
                    "threshold": self.config.alert_thresholds["cpu_percent"]
                })
            
            memory_percent = current.get("memory_percent", 0)
            if memory_percent > self.config.alert_thresholds["memory_percent"]:
                alerts.append({
                    "type": "system",
                    "severity": "warning",
                    "message": f"High memory usage: {memory_percent:.1f}%",
                    "threshold": self.config.alert_thresholds["memory_percent"]
                })
        
        # Flow analysis alerts
        if flow_analysis.get("enabled", False) and flow_analysis.get("analysis"):
            analysis = flow_analysis["analysis"]
            if analysis.get("failed_flows", 0) > 0:
                failed_flows = analysis["failed_flows"]
                total_flows = analysis.get("total_flows", 1)
                failure_rate = (failed_flows / total_flows) * 100
                
                if failure_rate > 10:  # 10% failure rate threshold
                    alerts.append({
                        "type": "flow",
                        "severity": "warning",
                        "message": f"High flow failure rate: {failure_rate:.1f}%",
                        "threshold": 10.0
                    })
        
        return alerts
    
    def _generate_recommendations(self, telemetry_data: Dict[str, Any], performance_data: Dict[str, Any],
                                flow_analysis: Dict[str, Any], system_metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        if performance_data.get("enabled", False):
            if performance_data.get("bottlenecks_detected", 0) > 0:
                recommendations.append("Investigate and optimize detected performance bottlenecks")
            
            if performance_data.get("error_rate", 0) > 5:
                recommendations.append("Review error patterns to improve system reliability")
            
            if performance_data.get("avg_duration_ms", 0) > 3000:
                recommendations.append("Consider optimizing long-running operations")
        
        # Flow analysis recommendations
        if flow_analysis.get("enabled", False) and flow_analysis.get("analysis"):
            analysis = flow_analysis["analysis"]
            if analysis.get("recommendations"):
                recommendations.extend(analysis["recommendations"])
        
        # System recommendations
        if system_metrics.get("enabled", False):
            if system_metrics.get("active_alerts", 0) > 0:
                recommendations.append("Address active system resource alerts")
            
            current = system_metrics.get("current_metrics", {})
            if current.get("memory_percent", 0) > 75:
                recommendations.append("Consider increasing available memory or optimizing memory usage")
        
        # Telemetry recommendations
        if telemetry_data.get("enabled", False):
            if telemetry_data.get("success_rate", 100) < 95:
                recommendations.append("Investigate telemetry collection issues")
        
        if not recommendations:
            recommendations.append("System is performing well - no critical recommendations")
        
        return recommendations
    
    def _calculate_health_score(self, telemetry_data: Dict[str, Any], performance_data: Dict[str, Any],
                              flow_analysis: Dict[str, Any], system_metrics: Dict[str, Any]) -> int:
        """Calculate overall system health score (0-100)."""
        score = 100
        
        # Performance penalties
        if performance_data.get("enabled", False):
            error_rate = performance_data.get("error_rate", 0)
            score -= min(error_rate * 2, 20)  # Max 20 points for errors
            
            avg_duration = performance_data.get("avg_duration_ms", 0)
            if avg_duration > 1000:
                score -= min((avg_duration - 1000) / 100, 15)  # Max 15 points for slow response
        
        # System resource penalties
        if system_metrics.get("enabled", False) and system_metrics.get("current_metrics"):
            current = system_metrics["current_metrics"]
            
            cpu_percent = current.get("cpu_percent", 0)
            if cpu_percent > 80:
                score -= (cpu_percent - 80) * 0.5
            
            memory_percent = current.get("memory_percent", 0)
            if memory_percent > 85:
                score -= (memory_percent - 85) * 0.5
        
        # Flow analysis penalties
        if flow_analysis.get("enabled", False) and flow_analysis.get("analysis"):
            analysis = flow_analysis["analysis"]
            if analysis.get("total_flows", 0) > 0:
                failure_rate = (analysis.get("failed_flows", 0) / analysis["total_flows"]) * 100
                score -= min(failure_rate * 2, 15)  # Max 15 points for flow failures
        
        # Telemetry penalties
        if telemetry_data.get("enabled", False):
            success_rate = telemetry_data.get("success_rate", 100)
            score -= (100 - success_rate) * 0.1
        
        return max(0, min(100, int(score)))
    
    def _create_summary(self, telemetry_data: Dict[str, Any], performance_data: Dict[str, Any],
                       flow_analysis: Dict[str, Any], system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of telemetry data."""
        return {
            "telemetry": {
                "enabled": telemetry_data.get("enabled", False),
                "events_collected": telemetry_data.get("events_collected", 0),
                "success_rate": telemetry_data.get("success_rate", 100)
            },
            "performance": {
                "enabled": performance_data.get("enabled", False),
                "total_operations": performance_data.get("total_operations", 0),
                "avg_duration_ms": performance_data.get("avg_duration_ms", 0),
                "error_rate": performance_data.get("error_rate", 0)
            },
            "flows": {
                "enabled": flow_analysis.get("enabled", False),
                "flows_analyzed": flow_analysis.get("flows_analyzed", 0),
                "bottlenecks_detected": flow_analysis.get("bottlenecks_detected", 0)
            },
            "system": {
                "enabled": system_metrics.get("enabled", False),
                "monitoring": system_metrics.get("is_monitoring", False),
                "active_alerts": system_metrics.get("active_alerts", 0)
            }
        }
    
    def get_dashboard_data(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        if not self.enabled:
            return {"enabled": False}
        
        # Check cache freshness
        if (self.last_refresh and 
            (datetime.now() - self.last_refresh).seconds < self.config.auto_refresh_interval):
            return self.dashboard_cache
        
        # Generate fresh data
        report = self.generate_report(timeframe_hours)
        
        dashboard_data = {
            "enabled": True,
            "generated_at": report.generated_at.isoformat(),
            "timeframe_hours": timeframe_hours,
            "health_score": report.health_score,
            "summary": report.summary,
            "alerts": report.alerts,
            "recommendations": report.recommendations,
            "components": {
                "telemetry": report.telemetry_data,
                "performance": report.performance_data,
                "flows": report.flow_analysis,
                "system": report.system_metrics
            }
        }
        
        # Update cache
        with self.lock:
            self.dashboard_cache = dashboard_data
            self.last_refresh = datetime.now()
            self.dashboard_refreshes += 1
        
        return dashboard_data
    
    def export_report(self, report: TelemetryReport, format: str = "json") -> str:
        """Export telemetry report in specified format."""
        if not self.enabled:
            return "{}" if format == "json" else ""
        
        if format == "json":
            # Convert to serializable format
            report_dict = asdict(report)
            report_dict['generated_at'] = report.generated_at.isoformat()
            
            return json.dumps(report_dict, indent=2, default=str)
        
        elif format == "csv":
            # Create CSV summary
            lines = [
                "metric,value",
                f"report_id,{report.report_id}",
                f"generated_at,{report.generated_at.isoformat()}",
                f"health_score,{report.health_score}",
                f"alerts_count,{len(report.alerts)}",
                f"recommendations_count,{len(report.recommendations)}"
            ]
            
            # Add summary data
            for key, value in report.summary.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        lines.append(f"{key}_{subkey},{subvalue}")
                else:
                    lines.append(f"{key},{value}")
            
            return "\n".join(lines)
        
        return str(asdict(report))
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "reports_generated": self.reports_generated,
            "dashboard_refreshes": self.dashboard_refreshes,
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "auto_refresh_interval": self.config.auto_refresh_interval,
            "cached_data_available": bool(self.dashboard_cache)
        }
    
    def _start_refresh_thread(self):
        """Start background refresh thread."""
        if not self.enabled:
            return
        
        def refresh_worker():
            while not self.shutdown_event.is_set():
                try:
                    if self.shutdown_event.wait(timeout=self.config.auto_refresh_interval):
                        break
                    
                    # Refresh dashboard data
                    self.get_dashboard_data()
                    
                except Exception:
                    # Handle errors silently
                    pass
        
        self.refresh_thread = threading.Thread(target=refresh_worker, daemon=True)
        self.refresh_thread.start()
    
    def shutdown(self):
        """Shutdown telemetry dashboard."""
        if not self.enabled:
            return
        
        self.shutdown_event.set()
        
        if self.refresh_thread and self.refresh_thread.is_alive():
            self.refresh_thread.join(timeout=1.0)
        
        print(f"Telemetry dashboard shutdown - generated {self.reports_generated} reports")

# Global instance
_telemetry_dashboard: Optional[TelemetryDashboard] = None

def get_telemetry_dashboard() -> TelemetryDashboard:
    """Get the global telemetry dashboard instance."""
    global _telemetry_dashboard
    if _telemetry_dashboard is None:
        _telemetry_dashboard = TelemetryDashboard()
    return _telemetry_dashboard

# Convenience functions
def create_telemetry_report(timeframe_hours: int = 24) -> TelemetryReport:
    """Create a comprehensive telemetry report."""
    dashboard = get_telemetry_dashboard()
    return dashboard.generate_report(timeframe_hours)

def export_telemetry_data(format: str = "json", timeframe_hours: int = 24) -> str:
    """Export telemetry data in specified format."""
    dashboard = get_telemetry_dashboard()
    report = dashboard.generate_report(timeframe_hours)
    return dashboard.export_report(report, format)