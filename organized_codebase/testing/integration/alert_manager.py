"""
Alert Manager Module
===================

Handles alert generation, evaluation, and management.
Extracted from realtime_performance_monitoring.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set

from .monitoring_models import (
    PerformanceAlert, PerformanceMetric, AlertSeverity,
    SystemType, MetricCategory
)


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self):
        self.logger = logging.getLogger("alert_manager")
        
        # Alert storage
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_history: deque = deque(maxlen=10000)
        
        # Alert configuration
        self.alert_config = {
            "evaluation_interval_seconds": 30,
            "auto_resolve_enabled": True,
            "escalation_enabled": True,
            "escalation_timeout_minutes": 30,
            "max_active_alerts": 1000,
            "alert_grouping_enabled": True,
            "alert_deduplication_enabled": True
        }
        
        # Alert handlers by severity
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        
        # Alert rules
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Alert statistics
        self.alert_stats = {
            "total_alerts_generated": 0,
            "alerts_by_severity": defaultdict(int),
            "alerts_by_system": defaultdict(int),
            "avg_resolution_time_seconds": 0.0,
            "auto_resolved_count": 0,
            "escalated_count": 0
        }
        
        # Alert suppression
        self.suppressed_alerts: Set[str] = set()
        self.alert_cooldown: Dict[str, datetime] = {}
    
    def register_alert_handler(self, severity: AlertSeverity, handler: Callable):
        """Register handler for specific alert severity."""
        self.alert_handlers[severity].append(handler)
        self.logger.info(f"Registered alert handler for {severity.value}")
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add custom alert rule."""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.get('name', 'unnamed')}")
    
    async def evaluate_metrics(self, metrics: Dict[str, PerformanceMetric]):
        """Evaluate metrics and generate alerts."""
        new_alerts = []
        
        for metric_id, metric in metrics.items():
            # Check threshold breaches
            severity = metric.is_threshold_breached()
            if severity:
                alert = self._create_alert(metric, severity)
                if alert and not self._is_duplicate_alert(alert):
                    new_alerts.append(alert)
            
            # Check custom rules
            for rule in self.alert_rules:
                if self._evaluate_rule(rule, metric):
                    alert = self._create_alert_from_rule(rule, metric)
                    if alert and not self._is_duplicate_alert(alert):
                        new_alerts.append(alert)
        
        # Process new alerts
        for alert in new_alerts:
            await self._process_alert(alert)
    
    def _create_alert(self, metric: PerformanceMetric, severity: AlertSeverity) -> Optional[PerformanceAlert]:
        """Create alert from metric threshold breach."""
        if not metric.values:
            return None
        
        # Check cooldown
        if self._is_in_cooldown(metric.metric_id):
            return None
        
        current_value = metric.values[-1]
        threshold_value = (
            metric.critical_threshold if severity == AlertSeverity.CRITICAL
            else metric.warning_threshold
        )
        
        alert = PerformanceAlert(
            metric_id=metric.metric_id,
            system=metric.system,
            severity=severity,
            title=f"{metric.name} {severity.value.upper()} Alert",
            description=f"{metric.name} has reached {current_value:.2f}{metric.unit} "
                       f"(threshold: {threshold_value:.2f}{metric.unit})",
            current_value=current_value,
            threshold_value=threshold_value,
            baseline_value=metric.baseline_value
        )
        
        return alert
    
    def _create_alert_from_rule(self, rule: Dict[str, Any], metric: PerformanceMetric) -> Optional[PerformanceAlert]:
        """Create alert from custom rule."""
        if not metric.values:
            return None
        
        alert = PerformanceAlert(
            metric_id=metric.metric_id,
            system=metric.system,
            severity=rule.get("severity", AlertSeverity.WARNING),
            title=rule.get("title", f"Custom Alert for {metric.name}"),
            description=rule.get("description", f"Rule '{rule.get('name')}' triggered"),
            current_value=metric.values[-1]
        )
        
        return alert
    
    def _evaluate_rule(self, rule: Dict[str, Any], metric: PerformanceMetric) -> bool:
        """Evaluate custom alert rule."""
        if not metric.values:
            return False
        
        # Check if rule applies to this metric
        if "metric_pattern" in rule:
            if rule["metric_pattern"] not in metric.metric_id:
                return False
        
        # Evaluate conditions
        condition = rule.get("condition", {})
        current_value = metric.values[-1]
        
        if "min_value" in condition and current_value < condition["min_value"]:
            return False
        if "max_value" in condition and current_value > condition["max_value"]:
            return False
        if "anomaly_score" in condition and metric.anomaly_score < condition["anomaly_score"]:
            return False
        
        return True
    
    def _is_duplicate_alert(self, alert: PerformanceAlert) -> bool:
        """Check if alert is duplicate of existing active alert."""
        if not self.alert_config["alert_deduplication_enabled"]:
            return False
        
        for active_alert in self.active_alerts:
            if (active_alert.metric_id == alert.metric_id and
                active_alert.severity == alert.severity and
                not active_alert.resolved):
                return True
        
        return False
    
    def _is_in_cooldown(self, metric_id: str) -> bool:
        """Check if metric is in alert cooldown period."""
        if metric_id in self.alert_cooldown:
            if datetime.now() < self.alert_cooldown[metric_id]:
                return True
            else:
                del self.alert_cooldown[metric_id]
        return False
    
    async def _process_alert(self, alert: PerformanceAlert):
        """Process new alert."""
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Update statistics
        self.alert_stats["total_alerts_generated"] += 1
        self.alert_stats["alerts_by_severity"][alert.severity.value] += 1
        self.alert_stats["alerts_by_system"][alert.system.value] += 1
        
        # Set cooldown
        cooldown_minutes = 5 if alert.severity == AlertSeverity.WARNING else 15
        self.alert_cooldown[alert.metric_id] = datetime.now() + timedelta(minutes=cooldown_minutes)
        
        # Trigger handlers
        await self._trigger_alert_handlers(alert)
        
        self.logger.info(f"Alert generated: {alert.alert_id} - {alert.title}")
    
    async def _trigger_alert_handlers(self, alert: PerformanceAlert):
        """Trigger registered alert handlers."""
        handlers = self.alert_handlers.get(alert.severity, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge()
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolve()
                
                # Update resolution statistics
                duration = alert.get_duration()
                prev_avg = self.alert_stats["avg_resolution_time_seconds"]
                total_resolved = sum(1 for a in self.alert_history if a.resolved) + 1
                self.alert_stats["avg_resolution_time_seconds"] = \
                    (prev_avg * (total_resolved - 1) + duration) / total_resolved
                
                # Move to history
                self.alert_history.append(alert)
                self.active_alerts.remove(alert)
                
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[PerformanceAlert]:
        """Get active alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self.active_alerts if a.severity == severity]
        return self.active_alerts.copy()
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            **self.alert_stats,
            "active_alerts_count": len(self.active_alerts),
            "suppressed_alerts_count": len(self.suppressed_alerts),
            "alerts_in_cooldown": len(self.alert_cooldown)
        }


__all__ = ['AlertManager']