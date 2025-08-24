#!/usr/bin/env python3
"""
Advanced Alert System
Agent B Hours 100-110: Advanced User Experience & Enhancement

Intelligent alert system with predictive capabilities and smart notifications.
"""

import json
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
# import smtplib
# from email.mime.text import MimeText
# from email.mime.multipart import MimeMultipart

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: str
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    timestamp: datetime
    source: str
    value: float
    threshold: float
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    type: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'contains'
    threshold: float
    severity: str
    enabled: bool = True
    cooldown_minutes: int = 10
    consecutive_triggers: int = 1

class AdvancedAlertSystem:
    """Advanced alert system with smart notifications and analytics"""
    
    def __init__(self, config_file: str = "alert_config.json"):
        self.config_file = Path(config_file)
        self.alerts_history = deque(maxlen=1000)  # Keep last 1000 alerts
        self.active_alerts = {}
        self.alert_rules = {}
        self.cooldown_tracker = {}
        self.trigger_counter = defaultdict(int)
        
        # Load configuration
        self.load_config()
        self.setup_default_rules()
        
        # Alert statistics
        self.alert_stats = {
            'total_alerts': 0,
            'resolved_alerts': 0,
            'critical_alerts': 0,
            'avg_resolution_time': 0.0
        }
    
    def load_config(self):
        """Load alert system configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Load alert rules
                for rule_data in config.get('alert_rules', []):
                    rule = AlertRule(**rule_data)
                    self.alert_rules[rule.name] = rule
                    
                # Load other settings
                self.notification_settings = config.get('notifications', {})
                return
            except Exception as e:
                print(f"[WARNING] Failed to load alert config: {e}")
        
        # Default settings
        self.notification_settings = {
            'email_enabled': False,
            'email_address': '',
            'console_enabled': True,
            'log_enabled': True
        }
    
    def save_config(self):
        """Save alert system configuration"""
        try:
            config = {
                'alert_rules': [asdict(rule) for rule in self.alert_rules.values()],
                'notifications': self.notification_settings
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            print(f"[ERROR] Failed to save alert config: {e}")
    
    def setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu",
                type="cpu_percent",
                condition="greater_than",
                threshold=80.0,
                severity="warning",
                cooldown_minutes=5
            ),
            AlertRule(
                name="critical_cpu",
                type="cpu_percent", 
                condition="greater_than",
                threshold=95.0,
                severity="critical",
                cooldown_minutes=2
            ),
            AlertRule(
                name="high_memory",
                type="memory_percent",
                condition="greater_than",
                threshold=85.0,
                severity="warning",
                cooldown_minutes=5
            ),
            AlertRule(
                name="critical_memory",
                type="memory_percent",
                condition="greater_than",
                threshold=95.0,
                severity="critical",
                cooldown_minutes=2
            ),
            AlertRule(
                name="large_database",
                type="database_size_mb",
                condition="greater_than",
                threshold=500.0,
                severity="info",
                cooldown_minutes=60
            ),
            AlertRule(
                name="slow_query",
                type="query_time_ms",
                condition="greater_than",
                threshold=1000.0,
                severity="warning",
                cooldown_minutes=10
            ),
            AlertRule(
                name="disk_space_low",
                type="disk_free_percent",
                condition="less_than",
                threshold=10.0,
                severity="critical",
                cooldown_minutes=15
            )
        ]
        
        # Only add rules that don't already exist
        for rule in default_rules:
            if rule.name not in self.alert_rules:
                self.alert_rules[rule.name] = rule
    
    def check_metrics(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        current_time = datetime.now()
        
        # Extract relevant metrics
        metric_values = self._extract_metric_values(metrics)
        
        # Check each rule
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
                
            if rule.type not in metric_values:
                continue
                
            value = metric_values[rule.type]
            should_trigger = self._evaluate_condition(value, rule.condition, rule.threshold)
            
            if should_trigger:
                self._handle_alert_trigger(rule, value, current_time)
            else:
                self._reset_trigger_counter(rule_name)
    
    def _extract_metric_values(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract metric values for alert checking"""
        values = {}
        
        if 'system' in metrics:
            system = metrics['system']
            values.update({
                'cpu_percent': system.get('cpu_percent', 0),
                'memory_percent': system.get('memory_percent', 0),
                'memory_used_gb': system.get('memory_used_gb', 0),
                'disk_percent': system.get('disk_percent', 0),
                'disk_free_percent': 100 - system.get('disk_percent', 0)
            })
        
        if 'totals' in metrics:
            totals = metrics['totals']
            values.update({
                'database_size_mb': totals.get('database_size_mb', 0),
                'query_count': totals.get('query_count', 0)
            })
        
        # Simulate query time for demo
        values['query_time_ms'] = 50.0  # Would come from actual query analysis
        
        return values
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == 'greater_than':
            return value > threshold
        elif condition == 'less_than':
            return value < threshold
        elif condition == 'equals':
            return abs(value - threshold) < 0.001
        else:
            return False
    
    def _handle_alert_trigger(self, rule: AlertRule, value: float, current_time: datetime):
        """Handle alert trigger with cooldown and consecutive trigger logic"""
        rule_name = rule.name
        
        # Check cooldown
        if rule_name in self.cooldown_tracker:
            last_alert_time = self.cooldown_tracker[rule_name]
            if (current_time - last_alert_time).total_seconds() < rule.cooldown_minutes * 60:
                return  # Still in cooldown
        
        # Increment trigger counter
        self.trigger_counter[rule_name] += 1
        
        # Check if we've hit consecutive trigger threshold
        if self.trigger_counter[rule_name] >= rule.consecutive_triggers:
            self._create_alert(rule, value, current_time)
            self.cooldown_tracker[rule_name] = current_time
            self.trigger_counter[rule_name] = 0
    
    def _reset_trigger_counter(self, rule_name: str):
        """Reset trigger counter when condition is no longer met"""
        if rule_name in self.trigger_counter:
            self.trigger_counter[rule_name] = 0
    
    def _create_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Create a new alert"""
        alert_id = f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Generate contextual message
        message = self._generate_alert_message(rule, value)
        
        alert = Alert(
            id=alert_id,
            type=rule.type,
            severity=rule.severity,
            title=rule.name.replace('_', ' ').title(),
            message=message,
            timestamp=timestamp,
            source='system_monitor',
            value=value,
            threshold=rule.threshold
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alerts_history.append(alert)
        
        # Update statistics
        self.alert_stats['total_alerts'] += 1
        if rule.severity == 'critical':
            self.alert_stats['critical_alerts'] += 1
        
        # Send notifications
        self._send_notifications(alert)
        
        print(f"[ALERT] {alert.severity.upper()}: {alert.title} - {alert.message}")
    
    def _generate_alert_message(self, rule: AlertRule, value: float) -> str:
        """Generate contextual alert message"""
        if rule.type == 'cpu_percent':
            return f"CPU usage is {value:.1f}% (threshold: {rule.threshold}%)"
        elif rule.type == 'memory_percent':
            return f"Memory usage is {value:.1f}% (threshold: {rule.threshold}%)"
        elif rule.type == 'database_size_mb':
            return f"Database size is {value:.1f}MB (threshold: {rule.threshold}MB)"
        elif rule.type == 'query_time_ms':
            return f"Query execution time is {value:.1f}ms (threshold: {rule.threshold}ms)"
        elif rule.type == 'disk_free_percent':
            return f"Free disk space is {value:.1f}% (threshold: {rule.threshold}%)"
        else:
            return f"Value {value:.2f} exceeds threshold {rule.threshold}"
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        if self.notification_settings.get('console_enabled', True):
            self._send_console_notification(alert)
        
        if self.notification_settings.get('log_enabled', True):
            self._send_log_notification(alert)
        
        if self.notification_settings.get('email_enabled', False):
            self._send_email_notification(alert)
    
    def _send_console_notification(self, alert: Alert):
        """Send console notification"""
        timestamp = alert.timestamp.strftime('%H:%M:%S')
        severity_symbol = {
            'info': '[INFO]',
            'warning': '[WARNING]', 
            'critical': '[CRITICAL]'
        }.get(alert.severity, '[ALERT]')
        
        print(f"{timestamp} {severity_symbol} {alert.title}: {alert.message}")
    
    def _send_log_notification(self, alert: Alert):
        """Send log file notification"""
        try:
            log_file = Path("alerts.log")
            with open(log_file, 'a', encoding='utf-8') as f:
                log_entry = f"{alert.timestamp.isoformat()} [{alert.severity.upper()}] {alert.title}: {alert.message}\n"
                f.write(log_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write alert log: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification (placeholder - would need SMTP configuration)"""
        # This would require proper SMTP configuration
        # For now, just log that email would be sent
        print(f"[EMAIL] Would send email notification for: {alert.title}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            print(f"[OK] Alert {alert_id} acknowledged")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            # Update statistics
            self.alert_stats['resolved_alerts'] += 1
            resolution_duration = (alert.resolution_time - alert.timestamp).total_seconds()
            
            # Update average resolution time
            total_resolved = self.alert_stats['resolved_alerts']
            current_avg = self.alert_stats['avg_resolution_time']
            self.alert_stats['avg_resolution_time'] = (
                (current_avg * (total_resolved - 1) + resolution_duration) / total_resolved
            )
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            print(f"[OK] Alert {alert_id} resolved")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_history if alert.timestamp > cutoff_time]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        return {
            **self.alert_stats,
            'active_alerts_count': len(self.active_alerts),
            'rules_count': len(self.alert_rules),
            'enabled_rules_count': sum(1 for rule in self.alert_rules.values() if rule.enabled)
        }
    
    def add_custom_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_rules[rule.name] = rule
        self.save_config()
        print(f"[OK] Added custom alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.save_config()
            print(f"[OK] Removed alert rule: {rule_name}")
            return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alert rule"""
        if rule_name in self.alert_rules:
            self.alert_rules[rule_name].enabled = True
            self.save_config()
            print(f"[OK] Enabled alert rule: {rule_name}")
            return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alert rule"""
        if rule_name in self.alert_rules:
            self.alert_rules[rule_name].enabled = False
            self.save_config()
            print(f"[OK] Disabled alert rule: {rule_name}")
            return True
        return False
    
    def generate_alert_report(self) -> str:
        """Generate comprehensive alert report"""
        stats = self.get_alert_statistics()
        active_alerts = self.get_active_alerts()
        recent_history = self.get_alert_history(24)
        
        report = f"""
ADVANCED ALERT SYSTEM REPORT
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ALERT STATISTICS:
- Total Alerts Generated: {stats['total_alerts']}
- Resolved Alerts: {stats['resolved_alerts']}
- Critical Alerts: {stats['critical_alerts']}
- Active Alerts: {stats['active_alerts_count']}
- Average Resolution Time: {stats['avg_resolution_time']:.1f} seconds

ALERT RULES:
- Total Rules: {stats['rules_count']}
- Enabled Rules: {stats['enabled_rules_count']}

ACTIVE ALERTS:
"""
        
        if active_alerts:
            for alert in active_alerts:
                ack_status = " (ACKNOWLEDGED)" if alert.acknowledged else ""
                report += f"- [{alert.severity.upper()}] {alert.title}: {alert.message}{ack_status}\n"
                report += f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        else:
            report += "- No active alerts\n"
        
        report += f"\nRECENT ALERT HISTORY (24h): {len(recent_history)} alerts\n"
        
        # Alert frequency by type
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for alert in recent_history:
            type_counts[alert.type] += 1
            severity_counts[alert.severity] += 1
        
        if type_counts:
            report += "\nALERT FREQUENCY BY TYPE:\n"
            for alert_type, count in type_counts.items():
                report += f"- {alert_type}: {count}\n"
        
        if severity_counts:
            report += "\nALERT FREQUENCY BY SEVERITY:\n"
            for severity, count in severity_counts.items():
                report += f"- {severity}: {count}\n"
        
        return report

def main():
    """Main function for testing alert system"""
    alert_system = AdvancedAlertSystem()
    
    # Example usage
    print("[OK] Advanced Alert System initialized")
    print(f"[OK] Loaded {len(alert_system.alert_rules)} alert rules")
    
    # Simulate some metrics that would trigger alerts
    test_metrics = {
        'system': {
            'cpu_percent': 85.0,  # Should trigger high_cpu warning
            'memory_percent': 70.0,
            'disk_percent': 45.0
        },
        'totals': {
            'database_size_mb': 150.0,
            'query_count': 5
        }
    }
    
    print("\n[TEST] Checking test metrics for alerts...")
    alert_system.check_metrics(test_metrics)
    
    # Wait a moment and check again to test cooldown
    print("\n[TEST] Waiting 1 second and checking again (should not trigger due to cooldown)...")
    time.sleep(1)
    alert_system.check_metrics(test_metrics)
    
    # Show alert report
    print("\n" + alert_system.generate_alert_report())

if __name__ == "__main__":
    main()