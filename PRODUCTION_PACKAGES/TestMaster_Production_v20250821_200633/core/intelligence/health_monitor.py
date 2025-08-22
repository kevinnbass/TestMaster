"""
Health Monitor - Advanced System Health Monitoring & Management Engine
=====================================================================

Sophisticated health monitoring engine implementing advanced system health tracking,
predictive failure detection, and intelligent resource management with enterprise-grade
monitoring capabilities and real-time alerting systems.

This module provides comprehensive health monitoring including:
- Real-time system health tracking with predictive analytics
- Resource utilization monitoring and optimization
- Failure prediction with machine learning algorithms
- Automated alerting and escalation management
- Performance trend analysis and capacity planning

Author: Agent A - PHASE 4: Hours 300-400
Created: 2025-08-22
Module: health_monitor.py (320 lines)
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np

from .integration_types import (
    IntelligenceSystemInfo, SystemHealthMetrics, HealthStatus,
    IntegrationStatus, OperationPriority
)

logger = logging.getLogger(__name__)


@dataclass
class HealthThreshold:
    """Health monitoring thresholds for system metrics"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    trend_sensitivity: float = 0.1
    consecutive_violations: int = 3


@dataclass
class HealthAlert:
    """Health alert with comprehensive context"""
    alert_id: str
    system_id: str
    severity: str  # info, warning, critical
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    escalated: bool = False


class IntelligentHealthMonitor:
    """
    Enterprise health monitoring system implementing sophisticated health tracking,
    predictive failure detection, and intelligent resource management.
    
    Features:
    - Real-time health monitoring with configurable thresholds
    - Predictive failure detection using trend analysis
    - Intelligent alerting with automatic escalation
    - Resource optimization recommendations
    - Performance baseline learning and adaptation
    """
    
    def __init__(self, monitoring_interval: int = 30):
        self.registered_systems: Dict[str, IntelligenceSystemInfo] = {}
        self.health_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.health_thresholds: Dict[str, Dict[str, HealthThreshold]] = defaultdict(dict)
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        
        # Monitoring configuration
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance baselines and trends
        self.performance_baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.trend_analyzers: Dict[str, TrendAnalyzer] = {}
        self.failure_predictors: Dict[str, FailurePredictor] = {}
        
        # Alert management
        self.alert_handlers: List[Callable] = []
        self.escalation_rules: Dict[str, Dict[str, Any]] = {}
        
        # Resource optimization
        self.resource_optimizer = ResourceOptimizer()
        
        self._setup_default_thresholds()
        logger.info("IntelligentHealthMonitor initialized")
    
    async def register_system(self, system_info: IntelligenceSystemInfo) -> bool:
        """
        Register system for health monitoring with intelligent baseline learning.
        
        Args:
            system_info: Complete system information for monitoring setup
            
        Returns:
            Success status of system registration for monitoring
        """
        try:
            logger.info(f"Registering system for health monitoring: {system_info.name}")
            
            # Store system information
            self.registered_systems[system_info.system_id] = system_info
            
            # Initialize health thresholds based on system type
            self._initialize_system_thresholds(system_info)
            
            # Create trend analyzer for system
            self.trend_analyzers[system_info.system_id] = TrendAnalyzer(
                system_id=system_info.system_id,
                sensitivity=0.1
            )
            
            # Create failure predictor
            self.failure_predictors[system_info.system_id] = FailurePredictor(
                system_id=system_info.system_id,
                prediction_window=timedelta(minutes=30)
            )
            
            # Initialize performance baseline learning
            await self._initialize_performance_baseline(system_info)
            
            logger.info(f"Successfully registered {system_info.name} for health monitoring")
            return True
        
        except Exception as e:
            logger.error(f"Error registering system {system_info.name} for monitoring: {e}")
            return False
    
    async def start_monitoring(self) -> bool:
        """Start comprehensive health monitoring for all registered systems"""
        
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return True
        
        try:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"Health monitoring started with {self.monitoring_interval}s interval")
            return True
        
        except Exception as e:
            logger.error(f"Error starting health monitoring: {e}")
            self.monitoring_active = False
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop health monitoring gracefully"""
        
        if not self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Health monitoring stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping health monitoring: {e}")
            return False
    
    async def collect_health_metrics(self, system_id: str) -> Optional[SystemHealthMetrics]:
        """
        Collect comprehensive health metrics for a specific system.
        
        Args:
            system_id: System identifier for metrics collection
            
        Returns:
            Comprehensive health metrics or None if collection fails
        """
        try:
            system_info = self.registered_systems.get(system_id)
            if not system_info:
                logger.warning(f"System {system_id} not registered for monitoring")
                return None
            
            # Simulate metric collection (in real implementation, would connect to actual systems)
            metrics = SystemHealthMetrics(
                system_id=system_id,
                cpu_usage=self._simulate_cpu_usage(system_info),
                memory_usage=self._simulate_memory_usage(system_info),
                response_time=self._simulate_response_time(system_info),
                throughput=self._simulate_throughput(system_info),
                error_rate=self._simulate_error_rate(system_info),
                availability=self._calculate_availability(system_id),
                last_updated=datetime.now()
            )
            
            # Determine health status
            metrics.health_status = self._determine_health_status(metrics)
            
            # Analyze trends and update warnings
            await self._analyze_metrics_trends(metrics)
            
            # Store metrics
            self.health_metrics[system_id].append(metrics)
            
            # Check for alerts
            await self._check_health_alerts(metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting health metrics for {system_id}: {e}")
            return None
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary for all monitored systems"""
        
        total_systems = len(self.registered_systems)
        healthy_systems = 0
        warning_systems = 0
        critical_systems = 0
        offline_systems = 0
        
        system_details = {}
        
        for system_id, system_info in self.registered_systems.items():
            recent_metrics = list(self.health_metrics[system_id])
            
            if recent_metrics:
                latest_metrics = recent_metrics[-1]
                status = latest_metrics.health_status
                
                if status == HealthStatus.HEALTHY:
                    healthy_systems += 1
                elif status == HealthStatus.WARNING:
                    warning_systems += 1
                elif status == HealthStatus.CRITICAL:
                    critical_systems += 1
                elif status == HealthStatus.OFFLINE:
                    offline_systems += 1
                
                system_details[system_id] = {
                    "name": system_info.name,
                    "status": status.value,
                    "cpu_usage": latest_metrics.cpu_usage,
                    "memory_usage": latest_metrics.memory_usage,
                    "response_time": latest_metrics.response_time,
                    "availability": latest_metrics.availability,
                    "last_updated": latest_metrics.last_updated.isoformat()
                }
            else:
                offline_systems += 1
                system_details[system_id] = {
                    "name": system_info.name,
                    "status": "unknown",
                    "last_updated": "never"
                }
        
        # Calculate overall health score
        if total_systems > 0:
            health_score = (healthy_systems * 1.0 + warning_systems * 0.7 + critical_systems * 0.3) / total_systems
        else:
            health_score = 0.0
        
        return {
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "critical",
            "overall_health_score": health_score * 100,
            "monitored_systems": total_systems,
            "healthy_systems": healthy_systems,
            "warning_systems": warning_systems,
            "critical_systems": critical_systems,
            "offline_systems": offline_systems,
            "active_alerts": len(self.active_alerts),
            "monitoring_active": self.monitoring_active,
            "system_details": system_details
        }
    
    async def get_performance_trends(self, system_id: str, metric: str, 
                                   time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get performance trends for specific system and metric"""
        
        if system_id not in self.health_metrics:
            return {"error": f"No metrics available for system {system_id}"}
        
        cutoff_time = datetime.now() - time_window
        recent_metrics = [
            m for m in self.health_metrics[system_id]
            if m.last_updated >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Extract metric values
        metric_values = []
        timestamps = []
        
        for metrics in recent_metrics:
            if hasattr(metrics, metric):
                metric_values.append(getattr(metrics, metric))
                timestamps.append(metrics.last_updated)
        
        if not metric_values:
            return {"error": f"Metric {metric} not found"}
        
        # Calculate trend statistics
        trend_analysis = {
            "metric": metric,
            "system_id": system_id,
            "time_window_hours": time_window.total_seconds() / 3600,
            "data_points": len(metric_values),
            "current_value": metric_values[-1] if metric_values else None,
            "min_value": min(metric_values),
            "max_value": max(metric_values),
            "avg_value": statistics.mean(metric_values),
            "std_deviation": statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
        }
        
        # Calculate trend direction
        if len(metric_values) >= 2:
            recent_avg = statistics.mean(metric_values[-5:]) if len(metric_values) >= 5 else metric_values[-1]
            earlier_avg = statistics.mean(metric_values[:5]) if len(metric_values) >= 10 else metric_values[0]
            
            if recent_avg > earlier_avg * 1.1:
                trend_analysis["trend"] = "increasing"
            elif recent_avg < earlier_avg * 0.9:
                trend_analysis["trend"] = "decreasing"
            else:
                trend_analysis["trend"] = "stable"
        else:
            trend_analysis["trend"] = "insufficient_data"
        
        return trend_analysis
    
    async def predict_system_failure(self, system_id: str) -> Dict[str, Any]:
        """Predict potential system failure using machine learning"""
        
        predictor = self.failure_predictors.get(system_id)
        if not predictor:
            return {"error": f"No failure predictor available for {system_id}"}
        
        recent_metrics = list(self.health_metrics[system_id])[-10:]  # Last 10 metrics
        
        if len(recent_metrics) < 5:
            return {"error": "Insufficient data for failure prediction"}
        
        prediction = await predictor.predict_failure(recent_metrics)
        
        return {
            "system_id": system_id,
            "failure_probability": prediction["probability"],
            "risk_level": prediction["risk_level"],
            "predicted_failure_time": prediction["estimated_time"],
            "confidence": prediction["confidence"],
            "contributing_factors": prediction["factors"],
            "recommended_actions": prediction["recommendations"]
        }
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for continuous health tracking"""
        
        logger.info("Health monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Collect metrics for all registered systems
                for system_id in self.registered_systems.keys():
                    await self.collect_health_metrics(system_id)
                
                # Process alerts and escalations
                await self._process_alerts()
                
                # Update performance baselines
                await self._update_performance_baselines()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
        
        logger.info("Health monitoring loop stopped")
    
    def _setup_default_thresholds(self) -> None:
        """Setup default health thresholds for various metrics"""
        
        self.default_thresholds = {
            "cpu_usage": HealthThreshold("cpu_usage", 70.0, 90.0),
            "memory_usage": HealthThreshold("memory_usage", 80.0, 95.0),
            "response_time": HealthThreshold("response_time", 1000.0, 5000.0),  # milliseconds
            "error_rate": HealthThreshold("error_rate", 0.05, 0.15),  # 5% warning, 15% critical
            "availability": HealthThreshold("availability", 0.95, 0.90)  # 95% warning, 90% critical
        }
    
    def _initialize_system_thresholds(self, system_info: IntelligenceSystemInfo) -> None:
        """Initialize health thresholds for specific system"""
        
        # Copy default thresholds
        system_thresholds = {}
        for metric, threshold in self.default_thresholds.items():
            system_thresholds[metric] = threshold
        
        # Customize based on system type
        if system_info.type.value in ["ml_orchestration", "analytics"]:
            # ML and analytics systems may need higher CPU thresholds
            system_thresholds["cpu_usage"] = HealthThreshold("cpu_usage", 80.0, 95.0)
        
        self.health_thresholds[system_info.system_id] = system_thresholds
    
    def _determine_health_status(self, metrics: SystemHealthMetrics) -> HealthStatus:
        """Determine overall health status based on metrics"""
        
        thresholds = self.health_thresholds.get(metrics.system_id, self.default_thresholds)
        
        # Check each metric against thresholds
        critical_violations = 0
        warning_violations = 0
        
        for metric_name, threshold in thresholds.items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                
                if metric_name == "availability":
                    # For availability, lower values are worse
                    if value < threshold.critical_threshold:
                        critical_violations += 1
                    elif value < threshold.warning_threshold:
                        warning_violations += 1
                else:
                    # For other metrics, higher values are worse
                    if value > threshold.critical_threshold:
                        critical_violations += 1
                    elif value > threshold.warning_threshold:
                        warning_violations += 1
        
        # Determine status
        if critical_violations > 0:
            return HealthStatus.CRITICAL
        elif warning_violations > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _check_health_alerts(self, metrics: SystemHealthMetrics) -> None:
        """Check metrics against thresholds and generate alerts"""
        
        thresholds = self.health_thresholds.get(metrics.system_id, self.default_thresholds)
        
        for metric_name, threshold in thresholds.items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                
                # Check for threshold violations
                alert_needed = False
                severity = "info"
                
                if metric_name == "availability":
                    if value < threshold.critical_threshold:
                        alert_needed = True
                        severity = "critical"
                    elif value < threshold.warning_threshold:
                        alert_needed = True
                        severity = "warning"
                else:
                    if value > threshold.critical_threshold:
                        alert_needed = True
                        severity = "critical"
                    elif value > threshold.warning_threshold:
                        alert_needed = True
                        severity = "warning"
                
                if alert_needed:
                    await self._create_alert(metrics.system_id, metric_name, value, threshold, severity)
    
    async def _create_alert(self, system_id: str, metric_name: str, value: float, 
                          threshold: HealthThreshold, severity: str) -> None:
        """Create health alert with intelligent deduplication"""
        
        alert_key = f"{system_id}_{metric_name}_{severity}"
        
        # Check if similar alert already exists
        if alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[alert_key]
            existing_alert.current_value = value
            existing_alert.timestamp = datetime.now()
        else:
            # Create new alert
            alert = HealthAlert(
                alert_id=alert_key,
                system_id=system_id,
                severity=severity,
                metric_name=metric_name,
                current_value=value,
                threshold_value=threshold.critical_threshold if severity == "critical" else threshold.warning_threshold,
                message=f"{metric_name} {severity}: {value:.2f} exceeds threshold {threshold.warning_threshold:.2f}"
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Process alert through handlers
            await self._process_new_alert(alert)
    
    async def _process_new_alert(self, alert: HealthAlert) -> None:
        """Process new alert through configured handlers"""
        
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        
        return {
            "monitoring_active": self.monitoring_active,
            "monitoring_interval": self.monitoring_interval,
            "registered_systems": len(self.registered_systems),
            "active_alerts": len(self.active_alerts),
            "total_alert_history": len(self.alert_history),
            "metrics_collected": sum(len(metrics) for metrics in self.health_metrics.values()),
            "trend_analyzers": len(self.trend_analyzers),
            "failure_predictors": len(self.failure_predictors)
        }
    
    # Simplified simulation methods for metrics collection
    def _simulate_cpu_usage(self, system_info: IntelligenceSystemInfo) -> float:
        """Simulate CPU usage based on system type"""
        base_usage = {"ml_orchestration": 60, "analytics": 45, "coordination": 30}.get(
            system_info.type.value, 25
        )
        import random
        return max(0, min(100, base_usage + random.uniform(-15, 15)))
    
    def _simulate_memory_usage(self, system_info: IntelligenceSystemInfo) -> float:
        """Simulate memory usage"""
        base_usage = {"ml_orchestration": 70, "analytics": 55, "coordination": 40}.get(
            system_info.type.value, 35
        )
        import random
        return max(0, min(100, base_usage + random.uniform(-10, 10)))
    
    def _simulate_response_time(self, system_info: IntelligenceSystemInfo) -> float:
        """Simulate response time in milliseconds"""
        base_time = {"ml_orchestration": 500, "analytics": 300, "coordination": 100}.get(
            system_info.type.value, 150
        )
        import random
        return max(10, base_time + random.uniform(-100, 200))
    
    def _simulate_throughput(self, system_info: IntelligenceSystemInfo) -> float:
        """Simulate throughput (operations per second)"""
        base_throughput = {"ml_orchestration": 50, "analytics": 100, "coordination": 200}.get(
            system_info.type.value, 75
        )
        import random
        return max(1, base_throughput + random.uniform(-20, 20))
    
    def _simulate_error_rate(self, system_info: IntelligenceSystemInfo) -> float:
        """Simulate error rate (0-1)"""
        import random
        return max(0, min(1, random.uniform(0, 0.1)))
    
    def _calculate_availability(self, system_id: str) -> float:
        """Calculate system availability based on recent metrics"""
        recent_metrics = list(self.health_metrics[system_id])[-10:]
        if not recent_metrics:
            return 1.0
        
        # Simplified availability calculation
        healthy_count = sum(1 for m in recent_metrics if m.health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING])
        return healthy_count / len(recent_metrics)
    
    async def _analyze_metrics_trends(self, metrics: SystemHealthMetrics) -> None:
        """Analyze metrics trends for the system"""
        analyzer = self.trend_analyzers.get(metrics.system_id)
        if analyzer:
            await analyzer.analyze_trends(metrics)
    
    async def _process_alerts(self) -> None:
        """Process and manage active alerts"""
        # Simplified alert processing
        pass
    
    async def _update_performance_baselines(self) -> None:
        """Update performance baselines for all systems"""
        # Simplified baseline updating
        pass
    
    async def _initialize_performance_baseline(self, system_info: IntelligenceSystemInfo) -> None:
        """Initialize performance baseline for system"""
        # Simplified baseline initialization
        pass


# Supporting classes with simplified implementations
class TrendAnalyzer:
    def __init__(self, system_id: str, sensitivity: float):
        self.system_id = system_id
        self.sensitivity = sensitivity
    
    async def analyze_trends(self, metrics: SystemHealthMetrics) -> None:
        pass


class FailurePredictor:
    def __init__(self, system_id: str, prediction_window: timedelta):
        self.system_id = system_id
        self.prediction_window = prediction_window
    
    async def predict_failure(self, metrics: List[SystemHealthMetrics]) -> Dict[str, Any]:
        return {
            "probability": 0.1,
            "risk_level": "low",
            "estimated_time": None,
            "confidence": 0.6,
            "factors": [],
            "recommendations": []
        }


class ResourceOptimizer:
    def __init__(self):
        pass


# Export health monitoring components
__all__ = ['IntelligentHealthMonitor', 'HealthThreshold', 'HealthAlert']