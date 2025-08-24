#!/usr/bin/env python3
"""
Budget Control & Alert System - Hours 10-15
Multi-tier budget management with smart alerting and automatic request blocking

Author: Agent Alpha
Created: 2025-08-23 21:05:00
Version: 1.0.0
"""

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BudgetPeriod(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    PROJECT = "project"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class BudgetConfiguration:
    """Budget configuration with hierarchy support"""
    budget_id: str
    name: str
    period: BudgetPeriod
    limit_usd: float
    parent_budget_id: Optional[str]
    alert_thresholds: List[float]  # [0.5, 0.75, 0.90, 0.95]
    hard_stop_enabled: bool
    rollover_enabled: bool
    created_at: datetime
    last_updated: datetime
    active: bool

@dataclass
class BudgetAlert:
    """Budget alert with escalating notifications"""
    alert_id: str
    budget_id: str
    alert_level: AlertLevel
    threshold_percent: float
    current_usage_usd: float
    budget_limit_usd: float
    message: str
    notification_channels: List[str]  # email, desktop, webhook
    acknowledged: bool
    created_at: datetime
    acknowledged_at: Optional[datetime]

@dataclass
class BudgetUsage:
    """Real-time budget usage tracking"""
    usage_id: str
    budget_id: str
    period_start: datetime
    period_end: datetime
    current_usage_usd: float
    projected_usage_usd: float
    remaining_budget_usd: float
    usage_percent: float
    last_updated: datetime

class BudgetControlAlertSystem:
    """Advanced budget control system with multi-tier management and smart alerting"""
    
    def __init__(self, db_path: str = "budget_control.db"):
        self.db_path = db_path
        self.budget_configs: Dict[str, BudgetConfiguration] = {}
        self.active_alerts: Dict[str, BudgetAlert] = {}
        self.budget_usage: Dict[str, BudgetUsage] = {}
        self.usage_history: deque = deque(maxlen=10000)
        self.budget_lock = threading.RLock()
        
        # Circuit breaker for request blocking
        self.circuit_breaker_active: Dict[str, bool] = {}
        self.blocked_request_count: Dict[str, int] = defaultdict(int)
        
        # Alert configuration
        self.default_thresholds = [0.5, 0.75, 0.90, 0.95]
        self.notification_cooldown = 300  # 5 minutes between same alerts
        self.last_notification_time: Dict[str, datetime] = {}
        
        # Initialize system
        self._init_database()
        self._create_default_budgets()
        
        # Background monitoring
        self.monitoring_active = False
        
    def _init_database(self):
        """Initialize SQLite database for budget control system"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_configurations (
                    budget_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    period TEXT NOT NULL,
                    limit_usd REAL NOT NULL,
                    parent_budget_id TEXT,
                    alert_thresholds TEXT NOT NULL,
                    hard_stop_enabled BOOLEAN DEFAULT TRUE,
                    rollover_enabled BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (parent_budget_id) REFERENCES budget_configurations(budget_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_alerts (
                    alert_id TEXT PRIMARY KEY,
                    budget_id TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    threshold_percent REAL NOT NULL,
                    current_usage_usd REAL NOT NULL,
                    budget_limit_usd REAL NOT NULL,
                    message TEXT NOT NULL,
                    notification_channels TEXT NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    FOREIGN KEY (budget_id) REFERENCES budget_configurations(budget_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_usage (
                    usage_id TEXT PRIMARY KEY,
                    budget_id TEXT NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    current_usage_usd REAL NOT NULL,
                    projected_usage_usd REAL NOT NULL,
                    remaining_budget_usd REAL NOT NULL,
                    usage_percent REAL NOT NULL,
                    last_updated TIMESTAMP,
                    FOREIGN KEY (budget_id) REFERENCES budget_configurations(budget_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_requests_log (
                    request_id TEXT PRIMARY KEY,
                    budget_id TEXT,
                    provider TEXT,
                    model TEXT,
                    tokens_used INTEGER,
                    cost_usd REAL,
                    timestamp TIMESTAMP,
                    blocked BOOLEAN DEFAULT FALSE,
                    block_reason TEXT,
                    FOREIGN KEY (budget_id) REFERENCES budget_configurations(budget_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_optimizations (
                    optimization_id TEXT PRIMARY KEY,
                    budget_id TEXT,
                    optimization_type TEXT,
                    suggestion TEXT,
                    potential_savings_usd REAL,
                    confidence_score REAL,
                    created_at TIMESTAMP,
                    applied BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (budget_id) REFERENCES budget_configurations(budget_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_budget_usage_budget_id ON budget_usage(budget_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_timestamp ON api_requests_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_budget_alerts_budget_id ON budget_alerts(budget_id)")
            
            conn.commit()
            
    def _create_default_budgets(self):
        """Create default budget hierarchy"""
        
        # Monthly master budget
        monthly_budget = BudgetConfiguration(
            budget_id="budget_monthly_master",
            name="Monthly API Budget",
            period=BudgetPeriod.MONTHLY,
            limit_usd=50.0,
            parent_budget_id=None,
            alert_thresholds=self.default_thresholds,
            hard_stop_enabled=True,
            rollover_enabled=True,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            active=True
        )
        
        # Daily budget (child of monthly)
        daily_budget = BudgetConfiguration(
            budget_id="budget_daily_limit",
            name="Daily API Budget",
            period=BudgetPeriod.DAILY,
            limit_usd=5.0,
            parent_budget_id="budget_monthly_master",
            alert_thresholds=self.default_thresholds,
            hard_stop_enabled=True,
            rollover_enabled=False,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            active=True
        )
        
        # Hourly budget (child of daily)
        hourly_budget = BudgetConfiguration(
            budget_id="budget_hourly_limit",
            name="Hourly API Budget",
            period=BudgetPeriod.HOURLY,
            limit_usd=0.5,
            parent_budget_id="budget_daily_limit",
            alert_thresholds=[0.75, 0.90, 0.95],
            hard_stop_enabled=True,
            rollover_enabled=False,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            active=True
        )
        
        # Project-specific budget
        project_budget = BudgetConfiguration(
            budget_id="budget_project_testmaster",
            name="TestMaster Project Budget",
            period=BudgetPeriod.PROJECT,
            limit_usd=200.0,
            parent_budget_id=None,
            alert_thresholds=self.default_thresholds,
            hard_stop_enabled=False,
            rollover_enabled=True,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            active=True
        )
        
        budgets = [monthly_budget, daily_budget, hourly_budget, project_budget]
        for budget in budgets:
            self.budget_configs[budget.budget_id] = budget
            self._save_budget_config(budget)
            
    def _save_budget_config(self, config: BudgetConfiguration):
        """Save budget configuration to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO budget_configurations
                (budget_id, name, period, limit_usd, parent_budget_id, alert_thresholds,
                 hard_stop_enabled, rollover_enabled, created_at, last_updated, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.budget_id, config.name, config.period.value, config.limit_usd,
                config.parent_budget_id, json.dumps(config.alert_thresholds),
                config.hard_stop_enabled, config.rollover_enabled,
                config.created_at, config.last_updated, config.active
            ))
            conn.commit()
            
    def create_budget(self, name: str, period: BudgetPeriod, limit_usd: float,
                     parent_budget_id: Optional[str] = None,
                     alert_thresholds: Optional[List[float]] = None,
                     hard_stop_enabled: bool = True,
                     rollover_enabled: bool = False) -> str:
        """Create new budget configuration"""
        
        budget_id = f"budget_{uuid.uuid4().hex[:8]}"
        
        if alert_thresholds is None:
            alert_thresholds = self.default_thresholds.copy()
            
        config = BudgetConfiguration(
            budget_id=budget_id,
            name=name,
            period=period,
            limit_usd=limit_usd,
            parent_budget_id=parent_budget_id,
            alert_thresholds=alert_thresholds,
            hard_stop_enabled=hard_stop_enabled,
            rollover_enabled=rollover_enabled,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            active=True
        )
        
        with self.budget_lock:
            self.budget_configs[budget_id] = config
            self._save_budget_config(config)
            
        logger.info(f"Created budget: {name} ({budget_id}) - ${limit_usd} {period.value}")
        return budget_id
        
    def update_budget_usage(self, budget_id: str, cost_usd: float, 
                           provider: str = "unknown", model: str = "unknown",
                           tokens_used: int = 0):
        """Update budget usage and check thresholds"""
        
        if budget_id not in self.budget_configs:
            logger.warning(f"Budget {budget_id} not found")
            return
            
        config = self.budget_configs[budget_id]
        
        # Check if request should be blocked
        if self._should_block_request(budget_id):
            self._log_blocked_request(budget_id, provider, model, tokens_used, cost_usd)
            raise BudgetExceededException(f"Budget {config.name} has exceeded hard limit")
            
        # Calculate period boundaries
        period_start, period_end = self._get_period_boundaries(config.period)
        
        # Get or create usage record
        usage = self._get_or_create_usage_record(budget_id, period_start, period_end)
        
        # Update usage
        with self.budget_lock:
            usage.current_usage_usd += cost_usd
            usage.remaining_budget_usd = config.limit_usd - usage.current_usage_usd
            usage.usage_percent = (usage.current_usage_usd / config.limit_usd) * 100
            usage.projected_usage_usd = self._calculate_projected_usage(usage, period_end)
            usage.last_updated = datetime.now()
            
            # Store updated usage
            self.budget_usage[budget_id] = usage
            self._save_budget_usage(usage)
            
            # Log API request
            self._log_api_request(budget_id, provider, model, tokens_used, cost_usd)
            
        # Check alert thresholds
        self._check_alert_thresholds(config, usage)
        
        # Update parent budgets recursively
        if config.parent_budget_id:
            self.update_budget_usage(config.parent_budget_id, cost_usd, provider, model, tokens_used)
            
    def _should_block_request(self, budget_id: str) -> bool:
        """Check if request should be blocked based on budget status"""
        
        if budget_id not in self.budget_configs:
            return False
            
        config = self.budget_configs[budget_id]
        
        if not config.hard_stop_enabled:
            return False
            
        # Check circuit breaker status
        if self.circuit_breaker_active.get(budget_id, False):
            return True
            
        # Check current usage
        if budget_id in self.budget_usage:
            usage = self.budget_usage[budget_id]
            if usage.usage_percent >= 100.0:
                self.circuit_breaker_active[budget_id] = True
                return True
                
        return False
        
    def _get_period_boundaries(self, period: BudgetPeriod) -> Tuple[datetime, datetime]:
        """Get period start and end boundaries"""
        now = datetime.now()
        
        if period == BudgetPeriod.HOURLY:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            # Start of week (Monday)
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        else:  # PROJECT
            # Project budgets don't have fixed periods
            start = datetime.min
            end = datetime.max
            
        return start, end
        
    def _get_or_create_usage_record(self, budget_id: str, period_start: datetime, 
                                   period_end: datetime) -> BudgetUsage:
        """Get existing usage record or create new one"""
        
        # Check if we have current usage record
        if budget_id in self.budget_usage:
            usage = self.budget_usage[budget_id]
            if usage.period_start <= datetime.now() < usage.period_end:
                return usage
                
        # Create new usage record
        usage_id = f"usage_{uuid.uuid4().hex[:8]}"
        config = self.budget_configs[budget_id]
        
        usage = BudgetUsage(
            usage_id=usage_id,
            budget_id=budget_id,
            period_start=period_start,
            period_end=period_end,
            current_usage_usd=0.0,
            projected_usage_usd=0.0,
            remaining_budget_usd=config.limit_usd,
            usage_percent=0.0,
            last_updated=datetime.now()
        )
        
        return usage
        
    def _calculate_projected_usage(self, usage: BudgetUsage, period_end: datetime) -> float:
        """Calculate projected usage for remainder of period"""
        
        now = datetime.now()
        if now >= period_end:
            return usage.current_usage_usd
            
        # Calculate time elapsed and remaining
        total_duration = (period_end - usage.period_start).total_seconds()
        elapsed_duration = (now - usage.period_start).total_seconds()
        
        if elapsed_duration <= 0:
            return 0.0
            
        # Calculate usage rate and project to end of period
        usage_rate = usage.current_usage_usd / elapsed_duration
        projected = usage_rate * total_duration
        
        return max(usage.current_usage_usd, projected)
        
    def _save_budget_usage(self, usage: BudgetUsage):
        """Save budget usage to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO budget_usage
                (usage_id, budget_id, period_start, period_end, current_usage_usd,
                 projected_usage_usd, remaining_budget_usd, usage_percent, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                usage.usage_id, usage.budget_id, usage.period_start, usage.period_end,
                usage.current_usage_usd, usage.projected_usage_usd, usage.remaining_budget_usd,
                usage.usage_percent, usage.last_updated
            ))
            conn.commit()
            
    def _log_api_request(self, budget_id: str, provider: str, model: str,
                        tokens_used: int, cost_usd: float):
        """Log API request to database"""
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_requests_log
                (request_id, budget_id, provider, model, tokens_used, cost_usd, timestamp, blocked, block_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id, budget_id, provider, model, tokens_used, cost_usd,
                datetime.now(), False, None
            ))
            conn.commit()
            
    def _log_blocked_request(self, budget_id: str, provider: str, model: str,
                           tokens_used: int, cost_usd: float):
        """Log blocked API request"""
        request_id = f"req_blocked_{uuid.uuid4().hex[:8]}"
        config = self.budget_configs[budget_id]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_requests_log
                (request_id, budget_id, provider, model, tokens_used, cost_usd, 
                 timestamp, blocked, block_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id, budget_id, provider, model, tokens_used, cost_usd,
                datetime.now(), True, f"Budget {config.name} exceeded hard limit"
            ))
            conn.commit()
            
        self.blocked_request_count[budget_id] += 1
        logger.warning(f"BLOCKED REQUEST: Budget {config.name} exceeded - Provider: {provider}, Cost: ${cost_usd}")
        
    def _check_alert_thresholds(self, config: BudgetConfiguration, usage: BudgetUsage):
        """Check if usage has crossed alert thresholds"""
        
        for threshold in config.alert_thresholds:
            if usage.usage_percent >= (threshold * 100) and not self._alert_recently_sent(config.budget_id, threshold):
                self._create_budget_alert(config, usage, threshold)
                
    def _alert_recently_sent(self, budget_id: str, threshold: float) -> bool:
        """Check if alert for this threshold was recently sent"""
        alert_key = f"{budget_id}_{threshold}"
        
        if alert_key not in self.last_notification_time:
            return False
            
        time_since_last = datetime.now() - self.last_notification_time[alert_key]
        return time_since_last.total_seconds() < self.notification_cooldown
        
    def _create_budget_alert(self, config: BudgetConfiguration, usage: BudgetUsage, threshold: float):
        """Create and send budget alert"""
        
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        # Determine alert level based on threshold
        if threshold >= 0.95:
            alert_level = AlertLevel.EMERGENCY
        elif threshold >= 0.90:
            alert_level = AlertLevel.CRITICAL
        elif threshold >= 0.75:
            alert_level = AlertLevel.WARNING
        else:
            alert_level = AlertLevel.INFO
            
        # Create alert message
        message = self._generate_alert_message(config, usage, threshold, alert_level)
        
        alert = BudgetAlert(
            alert_id=alert_id,
            budget_id=config.budget_id,
            alert_level=alert_level,
            threshold_percent=threshold * 100,
            current_usage_usd=usage.current_usage_usd,
            budget_limit_usd=config.limit_usd,
            message=message,
            notification_channels=["desktop", "log"],  # Add email/webhook as needed
            acknowledged=False,
            created_at=datetime.now(),
            acknowledged_at=None
        )
        
        with self.budget_lock:
            self.active_alerts[alert_id] = alert
            self._save_budget_alert(alert)
            
        # Send notifications
        self._send_alert_notifications(alert)
        
        # Update last notification time
        alert_key = f"{config.budget_id}_{threshold}"
        self.last_notification_time[alert_key] = datetime.now()
        
        logger.warning(f"BUDGET ALERT: {config.name} - {alert_level.value.upper()} - {message}")
        
    def _generate_alert_message(self, config: BudgetConfiguration, usage: BudgetUsage,
                               threshold: float, alert_level: AlertLevel) -> str:
        """Generate detailed alert message"""
        
        return (f"{alert_level.value.upper()}: Budget '{config.name}' has reached "
               f"{usage.usage_percent:.1f}% usage (${usage.current_usage_usd:.2f} of "
               f"${config.limit_usd:.2f}). Remaining: ${usage.remaining_budget_usd:.2f}. "
               f"Projected end-of-period usage: ${usage.projected_usage_usd:.2f}")
        
    def _save_budget_alert(self, alert: BudgetAlert):
        """Save budget alert to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO budget_alerts
                (alert_id, budget_id, alert_level, threshold_percent, current_usage_usd,
                 budget_limit_usd, message, notification_channels, acknowledged, 
                 created_at, acknowledged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.budget_id, alert.alert_level.value,
                alert.threshold_percent, alert.current_usage_usd, alert.budget_limit_usd,
                alert.message, json.dumps(alert.notification_channels),
                alert.acknowledged, alert.created_at, alert.acknowledged_at
            ))
            conn.commit()
            
    def _send_alert_notifications(self, alert: BudgetAlert):
        """Send alert notifications through configured channels"""
        
        for channel in alert.notification_channels:
            try:
                if channel == "desktop":
                    self._send_desktop_notification(alert)
                elif channel == "email":
                    self._send_email_notification(alert)
                elif channel == "webhook":
                    self._send_webhook_notification(alert)
                elif channel == "log":
                    logger.warning(f"BUDGET ALERT: {alert.message}")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
                
    def _send_desktop_notification(self, alert: BudgetAlert):
        """Send desktop notification (placeholder - requires platform-specific implementation)"""
        # This would integrate with Windows notifications, macOS notifications, or Linux desktop notifications
        print(f"DESKTOP NOTIFICATION: {alert.alert_level.value.upper()}")
        print(f"Budget Alert: {alert.message}")
        
    def _send_email_notification(self, alert: BudgetAlert):
        """Send email notification (requires SMTP configuration)"""
        # Placeholder for email notification - would require SMTP settings
        pass
        
    def _send_webhook_notification(self, alert: BudgetAlert):
        """Send webhook notification (placeholder - requires webhook URL configuration)"""
        # Placeholder for webhook notification
        pass
        
    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status for all configured budgets"""
        
        status = {}
        
        with self.budget_lock:
            for budget_id, config in self.budget_configs.items():
                if not config.active:
                    continue
                    
                usage = self.budget_usage.get(budget_id)
                active_alerts = [alert for alert in self.active_alerts.values() 
                               if alert.budget_id == budget_id and not alert.acknowledged]
                blocked_requests = self.blocked_request_count.get(budget_id, 0)
                circuit_breaker_active = self.circuit_breaker_active.get(budget_id, False)
                
                status[budget_id] = {
                    "config": asdict(config),
                    "usage": asdict(usage) if usage else None,
                    "active_alerts": len(active_alerts),
                    "blocked_requests": blocked_requests,
                    "circuit_breaker_active": circuit_breaker_active,
                    "health_status": self._get_budget_health_status(config, usage)
                }
                
        return status
        
    def _get_budget_health_status(self, config: BudgetConfiguration, 
                                 usage: Optional[BudgetUsage]) -> str:
        """Get budget health status"""
        
        if not usage:
            return "healthy"
            
        if usage.usage_percent >= 100:
            return "exceeded"
        elif usage.usage_percent >= 95:
            return "critical"
        elif usage.usage_percent >= 75:
            return "warning"
        elif usage.usage_percent >= 50:
            return "caution"
        else:
            return "healthy"
            
    def generate_budget_optimization_suggestions(self, budget_id: str) -> List[Dict[str, Any]]:
        """Generate ML-based budget optimization suggestions"""
        
        if budget_id not in self.budget_configs:
            return []
            
        suggestions = []
        usage = self.budget_usage.get(budget_id)
        
        if not usage:
            return suggestions
            
        # Analyze usage patterns and generate suggestions
        with sqlite3.connect(self.db_path) as conn:
            # Get recent request history
            cursor = conn.execute("""
                SELECT provider, model, AVG(cost_usd) as avg_cost, COUNT(*) as request_count
                FROM api_requests_log 
                WHERE budget_id = ? AND timestamp > datetime('now', '-7 days')
                GROUP BY provider, model
                ORDER BY request_count DESC
            """, (budget_id,))
            
            request_patterns = cursor.fetchall()
            
        # Generate optimization suggestions based on patterns
        total_requests = sum(row[3] for row in request_patterns)
        
        for provider, model, avg_cost, request_count in request_patterns:
            if request_count > total_requests * 0.1 and avg_cost > 0.01:  # Significant usage
                # Suggest model optimization
                suggestion = {
                    "type": "model_optimization",
                    "description": f"Consider using a more cost-effective model for {provider}/{model}",
                    "current_avg_cost": avg_cost,
                    "request_count": request_count,
                    "potential_savings": avg_cost * request_count * 0.3,  # Assume 30% savings
                    "confidence": 0.7
                }
                suggestions.append(suggestion)
                
        # Suggest batch processing if high frequency
        if total_requests > 100:
            suggestions.append({
                "type": "batch_processing",
                "description": "Consider batching requests to reduce per-request overhead",
                "potential_savings": usage.current_usage_usd * 0.15,  # 15% savings
                "confidence": 0.8
            })
            
        # Suggest caching if repetitive patterns detected
        suggestions.append({
            "type": "response_caching",
            "description": "Implement response caching to reduce duplicate requests",
            "potential_savings": usage.current_usage_usd * 0.25,  # 25% savings
            "confidence": 0.6
        })
        
        return suggestions
        
    def reset_circuit_breaker(self, budget_id: str) -> bool:
        """Reset circuit breaker for budget (manual override)"""
        
        if budget_id not in self.budget_configs:
            return False
            
        with self.budget_lock:
            self.circuit_breaker_active[budget_id] = False
            logger.info(f"Circuit breaker reset for budget {budget_id}")
            
        return True
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge budget alert"""
        
        if alert_id not in self.active_alerts:
            return False
            
        with self.budget_lock:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_at = datetime.now()
            
            # Update in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE budget_alerts 
                    SET acknowledged = ?, acknowledged_at = ?
                    WHERE alert_id = ?
                """, (True, alert.acknowledged_at, alert_id))
                conn.commit()
                
        logger.info(f"Alert {alert_id} acknowledged")
        return True


class BudgetExceededException(Exception):
    """Exception raised when budget limit is exceeded"""
    pass


# Global budget control system instance
budget_system = None

def get_budget_system() -> BudgetControlAlertSystem:
    """Get global budget control system instance"""
    global budget_system
    if budget_system is None:
        budget_system = BudgetControlAlertSystem()
    return budget_system

def check_budget_and_record_usage(budget_id: str, cost_usd: float, 
                                 provider: str = "unknown", model: str = "unknown",
                                 tokens_used: int = 0):
    """Check budget limits and record API usage"""
    system = get_budget_system()
    system.update_budget_usage(budget_id, cost_usd, provider, model, tokens_used)

def get_budget_status():
    """Get current budget status for all budgets"""
    system = get_budget_system()
    return system.get_budget_status()

def create_budget(name: str, period: str, limit_usd: float, **kwargs) -> str:
    """Create new budget configuration"""
    system = get_budget_system()
    period_enum = BudgetPeriod(period)
    return system.create_budget(name, period_enum, limit_usd, **kwargs)

if __name__ == "__main__":
    # Test the budget control system
    print("Budget Control & Alert System Testing...")
    
    # Get system instance
    system = get_budget_system()
    
    # Test budget usage
    try:
        # Simulate some API usage
        check_budget_and_record_usage("budget_hourly_limit", 0.25, "openai", "gpt-4", 1000)
        print("✅ Recorded $0.25 usage")
        
        check_budget_and_record_usage("budget_hourly_limit", 0.15, "anthropic", "claude", 800)
        print("✅ Recorded $0.15 usage")
        
        # This should trigger alerts
        check_budget_and_record_usage("budget_hourly_limit", 0.12, "openai", "gpt-4", 600)
        print("✅ Recorded $0.12 usage - should trigger alerts")
        
    except BudgetExceededException as e:
        print(f"❌ Budget exceeded: {e}")
        
    # Get status
    status = get_budget_status()
    for budget_id, info in status.items():
        config = info['config']
        usage = info.get('usage')
        if usage:
            print(f"Budget {config['name']}: ${usage['current_usage_usd']:.2f} / ${config['limit_usd']:.2f} ({usage['usage_percent']:.1f}%)")
        else:
            print(f"Budget {config['name']}: No usage recorded")