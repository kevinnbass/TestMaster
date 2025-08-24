"""
Event Monitoring Module
======================

Focused module for event monitoring, alerts, and multi-modal analysis.
Extracted from unified_monitor.py for better modularization.

Author: TestMaster Phase 1C Consolidation
"""

import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

class MonitoringMode(Enum):
    """Monitoring operation modes"""
    PASSIVE = "passive"
    INTERACTIVE = "interactive"
    PROACTIVE = "proactive"
    CONVERSATIONAL = "conversational"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MonitoringEvent:
    """Represents a monitoring event"""
    id: str = field(default_factory=lambda: f"event_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    source: str = ""
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class MonitoringAgent(ABC):
    """Base class for monitoring agents - RESTORED from enhanced_monitor.py"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.agent_id = f"monitor_{uuid.uuid4().hex[:12]}"
        self.active = False
        self.events: List[MonitoringEvent] = []
        self.logger = logging.getLogger(f'MonitoringAgent.{name}')
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> List[MonitoringEvent]:
        """Analyze monitoring data and generate events"""
        pass
    
    @abstractmethod
    async def respond_to_query(self, query: str, context: Dict[str, Any]) -> str:
        """Respond to monitoring queries"""
        pass
    
    async def start_monitoring(self):
        """Start the monitoring agent"""
        self.active = True
        self.logger.info(f"Monitoring agent {self.name} started")
    
    async def stop_monitoring(self):
        """Stop the monitoring agent"""
        self.active = False
        self.logger.info(f"Monitoring agent {self.name} stopped")

class MultiModalMonitor:
    """Multi-modal monitoring with conversational interface"""
    
    def __init__(self, mode: MonitoringMode = MonitoringMode.INTERACTIVE):
        self.mode = mode
        self.events: List[MonitoringEvent] = []
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_thresholds = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 10,
            AlertLevel.ERROR: 5,
            AlertLevel.CRITICAL: 1
        }
        self.conversational_history = []

    def emit_event(self, event: MonitoringEvent):
        """Emit a monitoring event"""
        self.events.append(event)
        
        # Trigger handlers
        for handler in self.event_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Event handler error: {e}")
        
        # Handle alerts based on level
        if event.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            self._handle_alert(event)

    def _handle_alert(self, event: MonitoringEvent):
        """Handle critical alerts"""
        alert_data = {
            "event_id": event.id,
            "level": event.level.value,
            "message": event.message,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "data": event.data
        }
        
        # Log alert
        logging.error(f"ALERT [{event.level.value}]: {event.message}")
        
        # Store for conversational interface
        self.conversational_history.append({
            "type": "alert",
            "data": alert_data,
            "timestamp": datetime.now().isoformat()
        })

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type].append(handler)

    def get_recent_events(self, count: int = 100, 
                         level: Optional[AlertLevel] = None) -> List[MonitoringEvent]:
        """Get recent events, optionally filtered by level"""
        events = self.events[-count:]
        
        if level:
            events = [e for e in events if e.level == level]
        
        return events

    def generate_insights(self) -> Dict[str, Any]:
        """Generate monitoring insights"""
        recent_events = self.get_recent_events(1000)
        
        # Event statistics
        event_stats = defaultdict(int)
        level_stats = defaultdict(int)
        source_stats = defaultdict(int)
        
        for event in recent_events:
            event_stats[event.event_type] += 1
            level_stats[event.level.value] += 1
            source_stats[event.source] += 1
        
        # Performance trends
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        recent_critical = [e for e in recent_events 
                          if e.level == AlertLevel.CRITICAL and e.timestamp >= hour_ago]
        
        insights = {
            "timestamp": now.isoformat(),
            "total_events": len(recent_events),
            "event_types": dict(event_stats),
            "alert_levels": dict(level_stats),
            "top_sources": dict(sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            "critical_alerts_last_hour": len(recent_critical),
            "recommendations": self._generate_recommendations(recent_events)
        }
        
        return insights

    def _generate_recommendations(self, events: List[MonitoringEvent]) -> List[str]:
        """Generate recommendations based on events"""
        recommendations = []
        
        # Check for patterns
        error_count = sum(1 for e in events if e.level == AlertLevel.ERROR)
        critical_count = sum(1 for e in events if e.level == AlertLevel.CRITICAL)
        
        if critical_count > 5:
            recommendations.append("High number of critical alerts detected - investigate system stability")
        
        if error_count > 20:
            recommendations.append("Elevated error rate - review error patterns and root causes")
        
        # Check source concentration
        source_counts = defaultdict(int)
        for event in events:
            source_counts[event.source] += 1
        
        max_source = max(source_counts.values()) if source_counts else 0
        if max_source > len(events) * 0.8:
            recommendations.append("Single source generating majority of events - investigate bottleneck")
        
        return recommendations

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
    
    def add_alert_rule(self, rule_id: str, event_type: str, conditions: Dict[str, Any]):
        """Add an alert rule"""
        self.alert_rules[rule_id] = {
            "event_type": event_type,
            "conditions": conditions,
            "created_at": datetime.now()
        }
    
    def add_notification_channel(self, channel: Callable):
        """Add a notification channel"""
        self.notification_channels.append(channel)
    
    def process_event(self, event: MonitoringEvent):
        """Process event against alert rules"""
        for rule_id, rule in self.alert_rules.items():
            if self._evaluate_rule(event, rule):
                self._trigger_alert(rule_id, event, rule)
    
    def _evaluate_rule(self, event: MonitoringEvent, rule: Dict[str, Any]) -> bool:
        """Evaluate if event matches alert rule"""
        if event.event_type != rule["event_type"]:
            return False
        
        conditions = rule["conditions"]
        
        # Check level condition
        if "min_level" in conditions:
            min_level = AlertLevel(conditions["min_level"])
            level_order = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
            if level_order.index(event.level) < level_order.index(min_level):
                return False
        
        # Check source condition
        if "source" in conditions and event.source != conditions["source"]:
            return False
        
        return True
    
    def _trigger_alert(self, rule_id: str, event: MonitoringEvent, rule: Dict[str, Any]):
        """Trigger alert for matched rule"""
        alert = {
            "rule_id": rule_id,
            "event_id": event.id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Alert triggered: {event.message}",
            "level": event.level.value,
            "source": event.source
        }
        
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logging.error(f"Notification channel failed: {e}")

class EventMonitor:
    """High-level event monitoring interface"""
    
    def __init__(self, mode: MonitoringMode = MonitoringMode.INTERACTIVE):
        self.multimodal_monitor = MultiModalMonitor(mode)
        self.alert_manager = AlertManager()
        
        # Connect alert manager to monitor
        self.multimodal_monitor.register_handler("*", self.alert_manager.process_event)
    
    def emit_event(self, event_type: str, source: str, level: AlertLevel,
                   message: str, data: Optional[Dict[str, Any]] = None):
        """Emit a monitoring event"""
        event = MonitoringEvent(
            event_type=event_type,
            source=source,
            level=level,
            message=message,
            data=data or {}
        )
        
        self.multimodal_monitor.emit_event(event)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health summary"""
        insights = self.multimodal_monitor.generate_insights()
        return {
            "status": "healthy" if insights["critical_alerts_last_hour"] < 5 else "warning",
            "insights": insights,
            "recent_alerts": len([a for a in self.alert_manager.alert_history 
                                if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=1)])
        }

# Export key components
__all__ = [
    'MonitoringMode',
    'AlertLevel',
    'MonitoringEvent',
    'MultiModalMonitor',
    'AlertManager',
    'EventMonitor'
]