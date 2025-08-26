"""
Alert and Notification System

Comprehensive alert management with multiple notification channels,
priority-based routing, and intelligent alert aggregation.

Features:
- Multi-level alert system (info, warning, error, critical)
- Multiple notification channels (dashboard, file, console)
- Alert aggregation and deduplication
- Priority-based alert routing and escalation
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from core.layer_manager import requires_layer


class AlertLevel(IntEnum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5


class AlertChannel(Enum):
    """Alert notification channels."""
    DASHBOARD = "dashboard"
    CONSOLE = "console"
    FILE = "file"
    EMAIL = "email"
    WEBHOOK = "webhook"


class AlertCategory(Enum):
    """Categories of alerts."""
    SYSTEM = "system"
    TEST_FAILURE = "test_failure"
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    IDLE_MODULE = "idle_module"
    QUEUE = "queue"
    COMMUNICATION = "communication"


@dataclass
class Alert:
    """A system alert."""
    alert_id: str
    level: AlertLevel
    category: AlertCategory
    title: str
    message: str
    timestamp: datetime
    
    # Alert metadata
    source: str = "TestMaster"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Alert lifecycle
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    # Alert routing
    channels: List[AlertChannel] = field(default_factory=list)
    escalation_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AlertRule:
    """Rule for alert generation and routing."""
    rule_id: str
    pattern: str  # Pattern to match against alert content
    target_level: AlertLevel
    target_channels: List[AlertChannel]
    escalation_minutes: Optional[int] = None
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertStatistics:
    """Alert system statistics."""
    total_alerts: int
    alerts_by_level: Dict[str, int]
    alerts_by_category: Dict[str, int]
    acknowledged_alerts: int
    resolved_alerts: int
    active_alerts: int
    avg_resolution_time_minutes: float
    last_updated: datetime = field(default_factory=datetime.now)


class AlertSystem:
    """
    Comprehensive alert and notification system.
    
    Manages alert generation, routing, escalation, and resolution
    with support for multiple notification channels.
    """
    
    @requires_layer("layer2_monitoring", "alert_system")
    def __init__(self, alert_dir: str = ".testmaster_alerts",
                 max_alert_history: int = 1000,
                 default_channels: List[AlertChannel] = None):
        """
        Initialize alert system.
        
        Args:
            alert_dir: Directory for alert persistence
            max_alert_history: Maximum alerts to keep in history
            default_channels: Default notification channels
        """
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(exist_ok=True)
        
        self.max_alert_history = max_alert_history
        self.default_channels = default_channels or [AlertChannel.DASHBOARD, AlertChannel.CONSOLE]
        
        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=max_alert_history)
        
        # Alert rules and routing
        self._alert_rules: Dict[str, AlertRule] = {}
        self._channel_handlers: Dict[AlertChannel, Callable] = {}
        
        # Alert aggregation and deduplication
        self._alert_counts: Dict[str, int] = defaultdict(int)
        self._last_alert_time: Dict[str, datetime] = {}
        self._suppressed_alerts: Set[str] = set()
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'level_counts': {level.name: 0 for level in AlertLevel},
            'category_counts': {cat.value: 0 for cat in AlertCategory},
            'acknowledged_count': 0,
            'resolved_count': 0,
            'resolution_times': deque(maxlen=100)
        }
        
        # Alert processing
        self._is_running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._alert_queue: deque = deque()
        
        # Callbacks
        self.on_alert_created: Optional[Callable[[Alert], None]] = None
        self.on_alert_acknowledged: Optional[Callable[[Alert], None]] = None
        self.on_alert_resolved: Optional[Callable[[Alert], None]] = None
        
        # Setup default channel handlers
        self._setup_default_handlers()
        
        print(f"🚨 Alert system initialized")
        print(f"   📁 Alert directory: {self.alert_dir}")
        print(f"   📢 Default channels: {', '.join(ch.value for ch in self.default_channels)}")
    
    def start(self):
        """Start alert processing."""
        if self._is_running:
            print("⚠️ Alert system is already running")
            return
        
        print("🚀 Starting alert system...")
        
        # Load existing alerts
        self._load_persisted_alerts()
        
        # Start processing thread
        self._is_running = True
        self._processing_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self._processing_thread.start()
        
        print("✅ Alert system started")
    
    def stop(self):
        """Stop alert processing."""
        if not self._is_running:
            return
        
        print("🛑 Stopping alert system...")
        
        self._is_running = False
        
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        
        # Persist active alerts
        self._persist_alerts()
        
        print("✅ Alert system stopped")
    
    def create_alert(self, level: AlertLevel, category: AlertCategory,
                    title: str, message: str,
                    source: str = "TestMaster",
                    tags: List[str] = None,
                    metadata: Dict[str, Any] = None,
                    channels: List[AlertChannel] = None) -> str:
        """
        Create a new alert.
        
        Args:
            level: Alert severity level
            category: Alert category
            title: Alert title
            message: Alert message
            source: Alert source
            tags: Alert tags
            metadata: Additional metadata
            channels: Override default channels
            
        Returns:
            Alert ID
        """
        # Generate alert ID
        alert_id = self._generate_alert_id()
        
        # Check for deduplication
        dedup_key = f"{category.value}:{title}:{message}"
        if self._should_suppress_alert(dedup_key):
            return alert_id  # Return ID but don't create duplicate
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            level=level,
            category=category,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            tags=tags or [],
            metadata=metadata or {},
            channels=channels or self.default_channels.copy()
        )
        
        # Apply alert rules
        self._apply_alert_rules(alert)
        
        # Store alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # Update statistics
        self._stats['total_alerts'] += 1
        self._stats['level_counts'][level.name] += 1
        self._stats['category_counts'][category.value] += 1
        
        # Track for deduplication
        self._alert_counts[dedup_key] += 1
        self._last_alert_time[dedup_key] = alert.timestamp
        
        # Queue for processing
        self._alert_queue.append(alert)
        
        # Call callback
        if self.on_alert_created:
            try:
                self.on_alert_created(alert)
            except Exception as e:
                print(f"⚠️ Error in alert created callback: {e}")
        
        print(f"🚨 Alert created: {level.name} - {title}")
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "System") -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if acknowledgment was successful
        """
        if alert_id not in self._active_alerts:
            return False
        
        alert = self._active_alerts[alert_id]
        
        if alert.acknowledged:
            return True  # Already acknowledged
        
        # Update alert
        alert.acknowledged = True
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        
        # Update statistics
        self._stats['acknowledged_count'] += 1
        
        # Call callback
        if self.on_alert_acknowledged:
            try:
                self.on_alert_acknowledged(alert)
            except Exception as e:
                print(f"⚠️ Error in alert acknowledged callback: {e}")
        
        print(f"✅ Alert acknowledged: {alert_id}")
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "System") -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of alert to resolve
            resolved_by: Who resolved the alert
            
        Returns:
            True if resolution was successful
        """
        if alert_id not in self._active_alerts:
            return False
        
        alert = self._active_alerts[alert_id]
        
        if alert.resolved:
            return True  # Already resolved
        
        # Update alert
        alert.resolved = True
        alert.resolved_at = datetime.now()
        alert.resolved_by = resolved_by
        
        # Calculate resolution time
        if alert.acknowledged_at:
            resolution_time = (alert.resolved_at - alert.acknowledged_at).total_seconds() / 60
        else:
            resolution_time = (alert.resolved_at - alert.timestamp).total_seconds() / 60
        
        self._stats['resolution_times'].append(resolution_time)
        self._stats['resolved_count'] += 1
        
        # Remove from active alerts
        del self._active_alerts[alert_id]
        
        # Call callback
        if self.on_alert_resolved:
            try:
                self.on_alert_resolved(alert)
            except Exception as e:
                print(f"⚠️ Error in alert resolved callback: {e}")
        
        print(f"✅ Alert resolved: {alert_id}")
        return True
    
    def add_alert_rule(self, rule_id: str, pattern: str, target_level: AlertLevel,
                      target_channels: List[AlertChannel],
                      escalation_minutes: int = None,
                      conditions: Dict[str, Any] = None) -> bool:
        """
        Add an alert routing rule.
        
        Args:
            rule_id: Unique rule identifier
            pattern: Pattern to match against alerts
            target_level: Target alert level
            target_channels: Target notification channels
            escalation_minutes: Minutes before escalation
            conditions: Additional conditions
            
        Returns:
            True if rule was added
        """
        rule = AlertRule(
            rule_id=rule_id,
            pattern=pattern,
            target_level=target_level,
            target_channels=target_channels,
            escalation_minutes=escalation_minutes,
            conditions=conditions or {}
        )
        
        self._alert_rules[rule_id] = rule
        print(f"📋 Added alert rule: {rule_id}")
        return True
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            print(f"🗑️ Removed alert rule: {rule_id}")
            return True
        return False
    
    def register_channel_handler(self, channel: AlertChannel, handler: Callable[[Alert], bool]):
        """
        Register a handler for a notification channel.
        
        Args:
            channel: Notification channel
            handler: Handler function that returns True on success
        """
        self._channel_handlers[channel] = handler
        print(f"📢 Registered handler for channel: {channel.value}")
    
    def _setup_default_handlers(self):
        """Setup default notification channel handlers."""
        # Console handler
        def console_handler(alert: Alert) -> bool:
            level_icon = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🆘"
            }.get(alert.level, "📢")
            
            print(f"{level_icon} [{alert.level.name}] {alert.title}: {alert.message}")
            return True
        
        # File handler
        def file_handler(alert: Alert) -> bool:
            try:
                alert_file = self.alert_dir / f"alert_{alert.alert_id}.json"
                alert_data = asdict(alert)
                
                # Convert datetime objects to ISO format
                alert_data['timestamp'] = alert.timestamp.isoformat()
                if alert.acknowledged_at:
                    alert_data['acknowledged_at'] = alert.acknowledged_at.isoformat()
                if alert.resolved_at:
                    alert_data['resolved_at'] = alert.resolved_at.isoformat()
                if alert.escalation_time:
                    alert_data['escalation_time'] = alert.escalation_time.isoformat()
                
                # Convert enums to strings
                alert_data['level'] = alert.level.name
                alert_data['category'] = alert.category.value
                alert_data['channels'] = [ch.value for ch in alert.channels]
                
                with open(alert_file, 'w') as f:
                    json.dump(alert_data, f, indent=2, default=str)
                
                return True
            except Exception as e:
                print(f"⚠️ Error writing alert to file: {e}")
                return False
        
        # Dashboard handler (placeholder - actual implementation depends on dashboard)
        def dashboard_handler(alert: Alert) -> bool:
            # This would integrate with the dashboard WebSocket system
            # For now, just return success
            return True
        
        # Register default handlers
        self._channel_handlers[AlertChannel.CONSOLE] = console_handler
        self._channel_handlers[AlertChannel.FILE] = file_handler
        self._channel_handlers[AlertChannel.DASHBOARD] = dashboard_handler
    
    def _process_alerts(self):
        """Process alert queue."""
        while self._is_running:
            try:
                # Process queued alerts
                while self._alert_queue and self._is_running:
                    alert = self._alert_queue.popleft()
                    self._send_alert_to_channels(alert)
                
                # Check for escalations
                self._check_escalations()
                
                # Cleanup old alerts
                self._cleanup_old_alerts()
                
                # Wait before next cycle
                time.sleep(5)
                
            except Exception as e:
                print(f"⚠️ Error processing alerts: {e}")
                time.sleep(10)
    
    def _send_alert_to_channels(self, alert: Alert):
        """Send alert to configured channels."""
        for channel in alert.channels:
            if channel in self._channel_handlers:
                try:
                    success = self._channel_handlers[channel](alert)
                    if not success and alert.retry_count < alert.max_retries:
                        # Retry failed sends
                        alert.retry_count += 1
                        self._alert_queue.append(alert)
                except Exception as e:
                    print(f"⚠️ Error sending alert to {channel.value}: {e}")
    
    def _apply_alert_rules(self, alert: Alert):
        """Apply alert rules to modify alert routing."""
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue
            
            # Simple pattern matching (could be enhanced with regex)
            if rule.pattern.lower() in alert.message.lower() or rule.pattern.lower() in alert.title.lower():
                # Apply rule modifications
                if rule.target_level > alert.level:
                    alert.level = rule.target_level
                
                # Add additional channels
                for channel in rule.target_channels:
                    if channel not in alert.channels:
                        alert.channels.append(channel)
                
                # Set escalation time
                if rule.escalation_minutes:
                    alert.escalation_time = alert.timestamp + timedelta(minutes=rule.escalation_minutes)
    
    def _should_suppress_alert(self, dedup_key: str) -> bool:
        """Check if alert should be suppressed due to deduplication."""
        # Suppress if same alert occurred recently
        if dedup_key in self._last_alert_time:
            last_time = self._last_alert_time[dedup_key]
            if datetime.now() - last_time < timedelta(minutes=5):  # 5-minute suppression window
                return True
        
        # Suppress if too many of same alert
        if self._alert_counts[dedup_key] > 10:  # Max 10 of same alert
            return True
        
        return False
    
    def _check_escalations(self):
        """Check for alerts that need escalation."""
        current_time = datetime.now()
        
        for alert in self._active_alerts.values():
            if (alert.escalation_time and 
                current_time > alert.escalation_time and 
                not alert.acknowledged):
                
                # Escalate alert
                self._escalate_alert(alert)
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an alert to higher priority."""
        # Increase alert level
        if alert.level < AlertLevel.EMERGENCY:
            alert.level = AlertLevel(alert.level + 1)
        
        # Add emergency channels
        if AlertChannel.EMAIL not in alert.channels:
            alert.channels.append(AlertChannel.EMAIL)
        
        # Re-queue for sending
        self._alert_queue.append(alert)
        
        print(f"🔺 Alert escalated: {alert.alert_id} to {alert.level.name}")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff = datetime.now() - timedelta(hours=24)
        
        # Remove old resolved alerts from active list
        resolved_ids = [
            alert_id for alert_id, alert in self._active_alerts.items()
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff
        ]
        
        for alert_id in resolved_ids:
            del self._active_alerts[alert_id]
        
        if resolved_ids:
            print(f"🧹 Cleaned up {len(resolved_ids)} old resolved alerts")
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        return f"alert_{int(time.time() * 1000)}_{hash(datetime.now()) % 10000}"
    
    def _load_persisted_alerts(self):
        """Load persisted alerts from disk."""
        try:
            for alert_file in self.alert_dir.glob("alert_*.json"):
                try:
                    with open(alert_file, 'r') as f:
                        alert_data = json.load(f)
                    
                    # Convert back to Alert object
                    alert = self._dict_to_alert(alert_data)
                    
                    # Only load unresolved alerts
                    if not alert.resolved:
                        self._active_alerts[alert.alert_id] = alert
                        
                except Exception as e:
                    print(f"⚠️ Error loading alert {alert_file}: {e}")
            
            print(f"📁 Loaded {len(self._active_alerts)} persisted alerts")
            
        except Exception as e:
            print(f"⚠️ Error loading persisted alerts: {e}")
    
    def _persist_alerts(self):
        """Persist active alerts to disk."""
        try:
            for alert in self._active_alerts.values():
                alert_file = self.alert_dir / f"alert_{alert.alert_id}.json"
                alert_data = asdict(alert)
                
                # Convert datetime objects and enums
                alert_data['timestamp'] = alert.timestamp.isoformat()
                if alert.acknowledged_at:
                    alert_data['acknowledged_at'] = alert.acknowledged_at.isoformat()
                if alert.resolved_at:
                    alert_data['resolved_at'] = alert.resolved_at.isoformat()
                if alert.escalation_time:
                    alert_data['escalation_time'] = alert.escalation_time.isoformat()
                
                alert_data['level'] = alert.level.name
                alert_data['category'] = alert.category.value
                alert_data['channels'] = [ch.value for ch in alert.channels]
                
                with open(alert_file, 'w') as f:
                    json.dump(alert_data, f, indent=2, default=str)
            
            print(f"💾 Persisted {len(self._active_alerts)} active alerts")
            
        except Exception as e:
            print(f"⚠️ Error persisting alerts: {e}")
    
    def _dict_to_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Convert dictionary back to Alert object."""
        # Parse datetime fields
        timestamp = datetime.fromisoformat(alert_data['timestamp'])
        acknowledged_at = datetime.fromisoformat(alert_data['acknowledged_at']) if alert_data.get('acknowledged_at') else None
        resolved_at = datetime.fromisoformat(alert_data['resolved_at']) if alert_data.get('resolved_at') else None
        escalation_time = datetime.fromisoformat(alert_data['escalation_time']) if alert_data.get('escalation_time') else None
        
        # Parse enums
        level = AlertLevel[alert_data['level']]
        category = AlertCategory(alert_data['category'])
        channels = [AlertChannel(ch) for ch in alert_data.get('channels', [])]
        
        return Alert(
            alert_id=alert_data['alert_id'],
            level=level,
            category=category,
            title=alert_data['title'],
            message=alert_data['message'],
            timestamp=timestamp,
            source=alert_data.get('source', 'TestMaster'),
            tags=alert_data.get('tags', []),
            metadata=alert_data.get('metadata', {}),
            acknowledged=alert_data.get('acknowledged', False),
            acknowledged_at=acknowledged_at,
            acknowledged_by=alert_data.get('acknowledged_by'),
            resolved=alert_data.get('resolved', False),
            resolved_at=resolved_at,
            resolved_by=alert_data.get('resolved_by'),
            channels=channels,
            escalation_time=escalation_time,
            retry_count=alert_data.get('retry_count', 0),
            max_retries=alert_data.get('max_retries', 3)
        )
    
    def get_active_alerts(self, level: AlertLevel = None,
                         category: AlertCategory = None) -> List[Alert]:
        """
        Get active alerts, optionally filtered.
        
        Args:
            level: Filter by alert level
            category: Filter by alert category
            
        Returns:
            List of active alerts
        """
        alerts = list(self._active_alerts.values())
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if category:
            alerts = [alert for alert in alerts if alert.category == category]
        
        # Sort by level (highest first) then by timestamp
        alerts.sort(key=lambda a: (-a.level, a.timestamp))
        
        return alerts
    
    def get_alert_statistics(self) -> AlertStatistics:
        """Get alert system statistics."""
        # Calculate average resolution time
        avg_resolution_time = 0.0
        if self._stats['resolution_times']:
            avg_resolution_time = statistics.mean(self._stats['resolution_times'])
        
        active_count = len(self._active_alerts)
        
        return AlertStatistics(
            total_alerts=self._stats['total_alerts'],
            alerts_by_level=dict(self._stats['level_counts']),
            alerts_by_category=dict(self._stats['category_counts']),
            acknowledged_alerts=self._stats['acknowledged_count'],
            resolved_alerts=self._stats['resolved_count'],
            active_alerts=active_count,
            avg_resolution_time_minutes=avg_resolution_time
        )
    
    def clear_resolved_alerts(self, older_than_hours: int = 24):
        """Clear resolved alerts older than specified hours."""
        cutoff = datetime.now() - timedelta(hours=older_than_hours)
        
        # Remove from history
        original_count = len(self._alert_history)
        self._alert_history = deque([
            alert for alert in self._alert_history
            if not (alert.resolved and alert.resolved_at and alert.resolved_at < cutoff)
        ], maxlen=self.max_alert_history)
        
        cleared_count = original_count - len(self._alert_history)
        
        # Remove alert files
        for alert_file in self.alert_dir.glob("alert_*.json"):
            try:
                with open(alert_file, 'r') as f:
                    alert_data = json.load(f)
                
                if (alert_data.get('resolved') and 
                    alert_data.get('resolved_at')):
                    resolved_at = datetime.fromisoformat(alert_data['resolved_at'])
                    if resolved_at < cutoff:
                        alert_file.unlink()
                        
            except Exception as e:
                print(f"⚠️ Error cleaning alert file {alert_file}: {e}")
        
        print(f"🧹 Cleared {cleared_count} old resolved alerts")


# Convenience functions for common alerts

def create_test_failure_alert(test_name: str, error_message: str, 
                            alert_system: AlertSystem) -> str:
    """Create a test failure alert."""
    return alert_system.create_alert(
        level=AlertLevel.ERROR,
        category=AlertCategory.TEST_FAILURE,
        title=f"Test Failed: {test_name}",
        message=error_message,
        tags=["test_failure", "automated"],
        metadata={"test_name": test_name}
    )


def create_idle_module_alert(module_path: str, idle_hours: float,
                           alert_system: AlertSystem) -> str:
    """Create an idle module alert."""
    return alert_system.create_alert(
        level=AlertLevel.WARNING,
        category=AlertCategory.IDLE_MODULE,
        title=f"Module Idle: {Path(module_path).name}",
        message=f"Module has been idle for {idle_hours:.1f} hours",
        tags=["idle_module", "automated"],
        metadata={"module_path": module_path, "idle_hours": idle_hours}
    )


def create_coverage_alert(coverage_percentage: float, threshold: float,
                        alert_system: AlertSystem) -> str:
    """Create a coverage threshold alert."""
    level = AlertLevel.WARNING if coverage_percentage < threshold else AlertLevel.INFO
    
    return alert_system.create_alert(
        level=level,
        category=AlertCategory.COVERAGE,
        title=f"Coverage Alert: {coverage_percentage:.1f}%",
        message=f"Coverage {coverage_percentage:.1f}% is below threshold {threshold}%",
        tags=["coverage", "automated"],
        metadata={"coverage": coverage_percentage, "threshold": threshold}
    )


# Convenience function for alert system setup
def setup_alert_system(channels: List[AlertChannel] = None) -> AlertSystem:
    """Setup alert system with default configuration."""
    default_channels = channels or [AlertChannel.DASHBOARD, AlertChannel.CONSOLE, AlertChannel.FILE]
    return AlertSystem(default_channels=default_channels)