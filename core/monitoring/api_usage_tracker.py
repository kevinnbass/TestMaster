#!/usr/bin/env python3
"""
API Usage Tracker - Critical Cost Control System
=================================================

CRITICAL: This module tracks ALL AI/LLM API calls to prevent cost overruns.
Must be integrated with dashboards BEFORE any AI-powered analysis tools are used.

Agent A - Emergency Implementation for Cost Control
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import warnings

# Cost configuration for different models (prices per 1K tokens)
API_COSTS = {
    # OpenAI Models
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    
    # Anthropic Models
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-2.1": {"input": 0.008, "output": 0.024},
    "claude-2": {"input": 0.008, "output": 0.024},
    
    # Google Models
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-pro-vision": {"input": 0.00025, "output": 0.0005},
    "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
    "gemini-1.5-flash": {"input": 0.00035, "output": 0.00105},
    "gemini-2.5-pro": {"input": 0.0035, "output": 0.0105},  # Same as 1.5-pro pricing
    
    # Cohere Models
    "command": {"input": 0.0015, "output": 0.0015},
    "command-light": {"input": 0.00015, "output": 0.00015},
    
    # Default fallback
    "unknown": {"input": 0.01, "output": 0.01}
}


class APICallType(Enum):
    """Types of API calls being made"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    TESTING = "testing"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    OTHER = "other"


class CostWarningLevel(Enum):
    """Enhanced cost warning levels with precise thresholds"""
    SAFE = "safe"           # Under 50% of budget
    WARNING = "warning"     # 50-75% of budget
    CRITICAL = "critical"   # 75-90% of budget
    DANGER = "danger"       # 90-95% of budget
    EXTREME = "extreme"     # 95-100% of budget
    EXCEEDED = "exceeded"   # Over budget


@dataclass
class APICall:
    """Record of a single API call"""
    call_id: str
    timestamp: datetime
    model: str
    call_type: APICallType
    purpose: str
    component: str  # Which component/module made the call
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    actual_cost: Optional[float] = None
    response_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # IRONCLAD EXTRACTION: Added endpoint tracking from File A
    endpoint: Optional[str] = None
    # IRONCLAD EXTRACTION: Added agent tracking from File A  
    agent: Optional[str] = None  # alpha, beta, gamma
    # IRONCLAD EXTRACTION: Added request/response size tracking from File A
    request_size: int = 0
    response_size: int = 0


@dataclass
class APIUsageStats:
    """Statistics for API usage"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost: float = 0.0
    total_actual_cost: float = 0.0
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    calls_by_type: Dict[str, int] = field(default_factory=dict)
    calls_by_component: Dict[str, int] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_type: Dict[str, float] = field(default_factory=dict)
    hourly_calls: Dict[str, int] = field(default_factory=dict)
    hourly_cost: Dict[str, float] = field(default_factory=dict)
    # IRONCLAD EXTRACTION: Added agent breakdown analytics from File A
    calls_by_agent: Dict[str, int] = field(default_factory=dict)
    cost_by_agent: Dict[str, float] = field(default_factory=dict)
    # IRONCLAD EXTRACTION: Added endpoint analytics from File A
    calls_by_endpoint: Dict[str, int] = field(default_factory=dict)
    cost_by_endpoint: Dict[str, float] = field(default_factory=dict)
    # IRONCLAD EXTRACTION: Added session tracking from File A
    session_calls: int = 0
    session_cost: float = 0.0


@dataclass
class NotificationConfig:
    """Notification system configuration"""
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    dashboard_alerts: bool = True
    alert_thresholds: List[float] = field(default_factory=lambda: [0.50, 0.75, 0.90, 0.95])
    alert_cooldown_minutes: int = 15  # Minimum time between duplicate alerts

@dataclass
class CostBudget:
    """Enhanced budget configuration for API costs"""
    daily_limit: float = 10.0  # $10 default daily limit
    hourly_limit: float = 2.0   # $2 default hourly limit
    per_call_limit: float = 0.5  # $0.50 per call limit
    total_limit: float = 100.0   # $100 total limit
    warning_threshold: float = 0.75  # Warn at 75% of limits
    auto_stop: bool = True  # Automatically stop when limit reached
    # HOUR 3 ENHANCEMENT: Advanced threshold controls
    enable_predictive_alerts: bool = True  # Alert before hitting limits
    grace_period_minutes: int = 60  # Grace period after hitting limits
    admin_override_enabled: bool = True  # Allow admin overrides


class APIUsageTracker:
    """
    CRITICAL: Central API usage tracking system
    Must be used for ALL AI/LLM API calls to track costs
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single tracker instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the API usage tracker"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        
        # Database for persistent tracking
        self.db_path = Path("state_data/api_usage.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # In-memory tracking
        self.current_session_calls: List[APICall] = []
        self.budget = CostBudget()
        self.stats = APIUsageStats()
        
        # IRONCLAD EXTRACTION: Added deque-based storage from File A for memory efficiency
        from collections import deque
        self.recent_calls = deque(maxlen=10000)  # Store last 10k calls efficiently
        
        # HOUR 3 ENHANCEMENT: Advanced notification system
        self.notification_config = NotificationConfig()
        self.alert_history = deque(maxlen=1000)  # Track alert history
        self.admin_overrides = deque(maxlen=100)  # Track admin overrides
        
        # HOUR 4 ENHANCEMENT: AI Integration System
        self.ai_enabled = True
        self.ai_engine = None
        self.cost_predictions = deque(maxlen=500)  # Store cost predictions
        self.usage_patterns = deque(maxlen=1000)   # Store usage patterns for AI
        self.ai_insights = deque(maxlen=200)       # Store AI-generated insights
        self._initialize_ai_engine()
        
        # Load existing stats
        self._load_stats()
        
        # Dashboard integration
        self.dashboard_callbacks = []
        
        self.logger.warning("=" * 60)
        self.logger.warning("API USAGE TRACKER INITIALIZED - COST MONITORING ACTIVE")
        self.logger.warning(f"Daily Budget: ${self.budget.daily_limit}")
        self.logger.warning(f"Current Usage: ${self.stats.total_estimated_cost:.2f}")
        self.logger.warning("=" * 60)
    
    def _init_database(self):
        """Initialize the database for persistent tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists and get column info
        cursor.execute("PRAGMA table_info(api_calls)")
        columns = cursor.fetchall()
        
        if not columns:
            # Create new table with all columns
            cursor.execute("""
                CREATE TABLE api_calls (
                    call_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    model TEXT,
                    call_type TEXT,
                    purpose TEXT,
                    component TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    estimated_cost REAL,
                    actual_cost REAL,
                    response_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    metadata TEXT,
                    endpoint TEXT,
                    agent TEXT,
                    request_size INTEGER,
                    response_size INTEGER
                )
            """)
        else:
            # IRONCLAD EXTRACTION: Handle database migration for existing tables
            existing_cols = {col[1] for col in columns}
            required_cols = ['endpoint', 'agent', 'request_size', 'response_size']
            
            for col in required_cols:
                if col not in existing_cols:
                    if col in ['endpoint', 'agent']:
                        cursor.execute(f"ALTER TABLE api_calls ADD COLUMN {col} TEXT")
                    else:
                        cursor.execute(f"ALTER TABLE api_calls ADD COLUMN {col} INTEGER DEFAULT 0")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_budgets (
                date DATE PRIMARY KEY,
                spent REAL,
                call_count INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def track_api_call(self,
                      model: str,
                      call_type: APICallType,
                      purpose: str,
                      component: str,
                      input_tokens: int,
                      output_tokens: int,
                      endpoint: Optional[str] = None,
                      agent: Optional[str] = None,
                      **kwargs) -> Tuple[bool, str, float]:
        """
        Track an API call and check if it's within budget
        
        Returns:
            Tuple of (allowed, message, estimated_cost)
        """
        # Calculate estimated cost
        model_costs = API_COSTS.get(model.lower(), API_COSTS["unknown"])
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        estimated_cost = input_cost + output_cost
        
        # HOUR 3 ENHANCEMENT: Advanced pre-execution budget checks
        # Check for active admin overrides first
        active_overrides = self.check_admin_overrides()
        
        # Check budgets
        warning_level, budget_message = self._check_budgets(estimated_cost)
        
        # Enhanced blocking logic with admin override support
        if warning_level == CostWarningLevel.EXCEEDED and self.budget.auto_stop and active_overrides == 0:
            self.logger.error(f"API CALL BLOCKED: Budget exceeded! {budget_message}")
            return False, f"BLOCKED: {budget_message} (Use admin override if critical)", estimated_cost
        
        # Predictive blocking for extreme usage
        if (warning_level == CostWarningLevel.EXTREME and 
            self.budget.enable_predictive_alerts and 
            active_overrides == 0 and
            estimated_cost > self.budget.per_call_limit * 2):  # Large calls at extreme usage
            self.logger.error(f"API CALL BLOCKED: Predictive limit reached at 95% budget with large call")
            return False, f"BLOCKED: Large call ({estimated_cost:.4f}) at 95% budget limit", estimated_cost
        
        # Create API call record
        api_call = APICall(
            call_id=f"{datetime.now().isoformat()}_{component}_{model}",
            timestamp=datetime.now(),
            model=model,
            call_type=call_type,
            purpose=purpose,
            component=component,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            # IRONCLAD EXTRACTION: Added endpoint and agent from File A
            endpoint=endpoint,
            agent=agent,
            **kwargs
        )
        
        # Track the call
        self._record_call(api_call)
        
        # Notify dashboards
        self._notify_dashboards(api_call)
        
        # HOUR 3 ENHANCEMENT: Advanced alert system with notifications
        if warning_level in [CostWarningLevel.WARNING, CostWarningLevel.CRITICAL, CostWarningLevel.DANGER, CostWarningLevel.EXTREME]:
            self.logger.warning(f"COST ALERT ({warning_level.value}): {budget_message}")
            self._send_budget_alert(warning_level, budget_message, estimated_cost)
        
        return True, budget_message, estimated_cost
    
    def pre_check_api_call(self, 
                          model: str,
                          estimated_input_tokens: int,
                          estimated_output_tokens: int) -> Tuple[bool, str, float]:
        """
        Pre-check if an API call would be within budget
        WITHOUT actually tracking it
        """
        model_costs = API_COSTS.get(model.lower(), API_COSTS["unknown"])
        input_cost = (estimated_input_tokens / 1000) * model_costs["input"]
        output_cost = (estimated_output_tokens / 1000) * model_costs["output"]
        estimated_cost = input_cost + output_cost
        
        warning_level, message = self._check_budgets(estimated_cost)
        
        allowed = warning_level != CostWarningLevel.EXCEEDED or not self.budget.auto_stop
        
        return allowed, message, estimated_cost
    
    def _check_budgets(self, additional_cost: float) -> Tuple[CostWarningLevel, str]:
        """Check if adding cost would exceed budgets"""
        messages = []
        highest_level = CostWarningLevel.SAFE
        
        # Check per-call limit
        if additional_cost > self.budget.per_call_limit:
            messages.append(f"Single call cost ${additional_cost:.2f} exceeds limit ${self.budget.per_call_limit}")
            highest_level = CostWarningLevel.EXCEEDED
        
        # Check hourly limit
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        hourly_spent = self.stats.hourly_cost.get(current_hour, 0.0)
        if hourly_spent + additional_cost > self.budget.hourly_limit:
            messages.append(f"Hourly budget would exceed: ${hourly_spent + additional_cost:.2f} > ${self.budget.hourly_limit}")
            if highest_level != CostWarningLevel.EXCEEDED:
                highest_level = CostWarningLevel.CRITICAL
        
        # Check daily limit
        today = datetime.now().strftime("%Y-%m-%d")
        daily_spent = self._get_daily_spent(today)
        if daily_spent + additional_cost > self.budget.daily_limit:
            messages.append(f"Daily budget would exceed: ${daily_spent + additional_cost:.2f} > ${self.budget.daily_limit}")
            highest_level = CostWarningLevel.EXCEEDED
        
        # Check total limit
        if self.stats.total_estimated_cost + additional_cost > self.budget.total_limit:
            messages.append(f"Total budget would exceed: ${self.stats.total_estimated_cost + additional_cost:.2f} > ${self.budget.total_limit}")
            highest_level = CostWarningLevel.EXCEEDED
        
        # HOUR 3 ENHANCEMENT: Calculate warning level with precise thresholds (50%, 75%, 90%, 95%)
        if highest_level == CostWarningLevel.SAFE:
            daily_percentage = (daily_spent + additional_cost) / self.budget.daily_limit
            if daily_percentage >= 1.0:
                highest_level = CostWarningLevel.EXCEEDED
            elif daily_percentage >= 0.95:
                highest_level = CostWarningLevel.EXTREME
            elif daily_percentage >= 0.90:
                highest_level = CostWarningLevel.DANGER
            elif daily_percentage >= 0.75:
                highest_level = CostWarningLevel.CRITICAL
            elif daily_percentage >= 0.50:
                highest_level = CostWarningLevel.WARNING
        
        message = " | ".join(messages) if messages else f"Within budget (${additional_cost:.4f})"
        return highest_level, message
    
    def _record_call(self, api_call: APICall):
        """Record an API call in database and memory"""
        # Update in-memory stats
        self.current_session_calls.append(api_call)
        self.recent_calls.append(api_call)  # IRONCLAD EXTRACTION: Add to deque storage
        self.stats.total_calls += 1
        
        if api_call.success:
            self.stats.successful_calls += 1
        else:
            self.stats.failed_calls += 1
        
        self.stats.total_input_tokens += api_call.input_tokens
        self.stats.total_output_tokens += api_call.output_tokens
        self.stats.total_estimated_cost += api_call.estimated_cost
        
        # IRONCLAD EXTRACTION: Update session tracking from File A
        self.stats.session_calls += 1
        self.stats.session_cost += api_call.estimated_cost
        
        # Update categorized stats
        model_key = api_call.model
        type_key = api_call.call_type.value
        
        self.stats.calls_by_model[model_key] = self.stats.calls_by_model.get(model_key, 0) + 1
        self.stats.calls_by_type[type_key] = self.stats.calls_by_type.get(type_key, 0) + 1
        self.stats.calls_by_component[api_call.component] = self.stats.calls_by_component.get(api_call.component, 0) + 1
        
        self.stats.cost_by_model[model_key] = self.stats.cost_by_model.get(model_key, 0.0) + api_call.estimated_cost
        self.stats.cost_by_type[type_key] = self.stats.cost_by_type.get(type_key, 0.0) + api_call.estimated_cost
        
        # IRONCLAD EXTRACTION: Added agent and endpoint tracking from File A
        if api_call.agent:
            self.stats.calls_by_agent[api_call.agent] = self.stats.calls_by_agent.get(api_call.agent, 0) + 1
            self.stats.cost_by_agent[api_call.agent] = self.stats.cost_by_agent.get(api_call.agent, 0.0) + api_call.estimated_cost
            
        if api_call.endpoint:
            self.stats.calls_by_endpoint[api_call.endpoint] = self.stats.calls_by_endpoint.get(api_call.endpoint, 0) + 1
            self.stats.cost_by_endpoint[api_call.endpoint] = self.stats.cost_by_endpoint.get(api_call.endpoint, 0.0) + api_call.estimated_cost
        
        # Update hourly stats
        hour_key = api_call.timestamp.strftime("%Y-%m-%d %H:00")
        self.stats.hourly_calls[hour_key] = self.stats.hourly_calls.get(hour_key, 0) + 1
        self.stats.hourly_cost[hour_key] = self.stats.hourly_cost.get(hour_key, 0.0) + api_call.estimated_cost
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO api_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            api_call.call_id,
            api_call.timestamp,
            api_call.model,
            api_call.call_type.value,
            api_call.purpose,
            api_call.component,
            api_call.input_tokens,
            api_call.output_tokens,
            api_call.estimated_cost,
            api_call.actual_cost,
            api_call.response_time,
            api_call.success,
            api_call.error_message,
            json.dumps(api_call.metadata),
            # IRONCLAD EXTRACTION: Added new fields from File A
            api_call.endpoint,
            api_call.agent,
            api_call.request_size,
            api_call.response_size
        ))
        
        # Update daily budget
        today = api_call.timestamp.strftime("%Y-%m-%d")
        cursor.execute("""
            INSERT INTO daily_budgets (date, spent, call_count)
            VALUES (?, ?, 1)
            ON CONFLICT(date) DO UPDATE SET
                spent = spent + ?,
                call_count = call_count + 1
        """, (today, api_call.estimated_cost, api_call.estimated_cost))
        
        conn.commit()
        conn.close()
    
    def _get_daily_spent(self, date: str) -> float:
        """Get amount spent on a specific date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT spent FROM daily_budgets WHERE date = ?", (date,))
        result = cursor.fetchone()
        
        conn.close()
        
        return result[0] if result else 0.0
    
    def _load_stats(self):
        """Load statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_calls,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(estimated_cost) as total_estimated_cost,
                SUM(actual_cost) as total_actual_cost
            FROM api_calls
        """)
        
        result = cursor.fetchone()
        if result and result[0]:
            self.stats.total_calls = result[0]
            self.stats.successful_calls = result[1] or 0
            self.stats.failed_calls = result[2] or 0
            self.stats.total_input_tokens = result[3] or 0
            self.stats.total_output_tokens = result[4] or 0
            self.stats.total_estimated_cost = result[5] or 0.0
            self.stats.total_actual_cost = result[6] or 0.0
        
        conn.close()
    
    def _notify_dashboards(self, api_call: APICall):
        """Notify registered dashboards of new API call"""
        for callback in self.dashboard_callbacks:
            try:
                callback(api_call)
            except Exception as e:
                self.logger.error(f"Dashboard notification failed: {e}")
    
    def register_dashboard_callback(self, callback):
        """Register a dashboard callback for real-time updates"""
        self.dashboard_callbacks.append(callback)
    
    # HOUR 3 ENHANCEMENT: Advanced Alert and Override System
    def _send_budget_alert(self, warning_level: CostWarningLevel, message: str, cost: float):
        """Send budget alert through configured notification channels"""
        # Check cooldown to prevent spam
        current_time = datetime.now()
        alert_key = f"{warning_level.value}_{hash(message) % 10000}"
        
        # Check if we sent this alert recently
        for alert in self.alert_history:
            if (alert.get('key') == alert_key and 
                current_time - datetime.fromisoformat(alert.get('timestamp', '1970-01-01')) < 
                timedelta(minutes=self.notification_config.alert_cooldown_minutes)):
                return  # Skip duplicate alert
        
        alert_data = {
            'timestamp': current_time.isoformat(),
            'key': alert_key,
            'level': warning_level.value,
            'message': message,
            'cost': cost,
            'budget_status': self._get_budget_status()
        }
        
        self.alert_history.append(alert_data)
        
        # Send email notification
        if self.notification_config.email_enabled:
            self._send_email_alert(alert_data)
        
        # Send webhook notification  
        if self.notification_config.webhook_enabled:
            self._send_webhook_alert(alert_data)
            
        # Dashboard notification (always enabled)
        self._send_dashboard_alert(alert_data)
    
    def _send_email_alert(self, alert_data: Dict):
        """Send email alert (placeholder for email integration)"""
        # This would integrate with email service (SMTP, SendGrid, etc.)
        self.logger.info(f"EMAIL ALERT: {alert_data['level']} - {alert_data['message']}")
        print(f"📧 EMAIL ALERT [{alert_data['level'].upper()}]: {alert_data['message']}")
        
    def _send_webhook_alert(self, alert_data: Dict):
        """Send webhook alert (placeholder for webhook integration)"""
        # This would send HTTP POST to configured webhook URL
        self.logger.info(f"WEBHOOK ALERT: {alert_data['level']} - {alert_data['message']}")
        print(f"🔗 WEBHOOK ALERT [{alert_data['level'].upper()}]: {alert_data['message']}")
    
    def _send_dashboard_alert(self, alert_data: Dict):
        """Send real-time dashboard alert"""
        alert_message = {
            'type': 'budget_alert',
            'severity': alert_data['level'],
            'message': alert_data['message'],
            'cost': alert_data['cost'],
            'timestamp': alert_data['timestamp']
        }
        
        # Notify all dashboard callbacks
        for callback in self.dashboard_callbacks:
            try:
                callback(alert_message)
            except Exception as e:
                self.logger.error(f"Dashboard alert callback failed: {e}")
    
    def admin_override_budget(self, reason: str, override_minutes: int = 60, admin_id: str = "system"):
        """Allow admin to temporarily override budget limits"""
        if not self.budget.admin_override_enabled:
            return False, "Admin overrides are disabled"
            
        current_time = datetime.now()
        override_data = {
            'timestamp': current_time.isoformat(),
            'admin_id': admin_id,
            'reason': reason,
            'duration_minutes': override_minutes,
            'expires_at': (current_time + timedelta(minutes=override_minutes)).isoformat()
        }
        
        self.admin_overrides.append(override_data)
        
        # Temporarily disable auto_stop
        self.budget.auto_stop = False
        
        self.logger.warning(f"ADMIN OVERRIDE: {admin_id} - {reason} (expires in {override_minutes} minutes)")
        
        # Send override notification
        self._send_dashboard_alert({
            'type': 'admin_override',
            'severity': 'warning',
            'message': f"Admin override active: {reason}",
            'admin': admin_id,
            'expires': override_data['expires_at'],
            'timestamp': current_time.isoformat()
        })
        
        return True, f"Override active for {override_minutes} minutes"
    
    def check_admin_overrides(self):
        """Check and expire admin overrides"""
        current_time = datetime.now()
        active_overrides = []
        
        for override in self.admin_overrides:
            expires_at = datetime.fromisoformat(override['expires_at'])
            if current_time < expires_at:
                active_overrides.append(override)
            else:
                # Override expired
                self.logger.info(f"Admin override expired: {override['admin_id']}")
        
        # Update override list
        self.admin_overrides.clear()
        self.admin_overrides.extend(active_overrides)
        
        # Re-enable auto_stop if no active overrides
        if not active_overrides and not self.budget.auto_stop:
            self.budget.auto_stop = True
            self.logger.info("Auto-stop re-enabled - no active admin overrides")
        
        return len(active_overrides)
    
    # HOUR 4 ENHANCEMENT: AI Intelligence Integration
    def _initialize_ai_engine(self):
        """Initialize AI engine for cost prediction and analysis"""
        try:
            # Import AI engine dynamically to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            
            from ai_intelligence_engine import AIIntelligenceEngine
            self.ai_engine = AIIntelligenceEngine()
            self.logger.info("AI Intelligence Engine integrated successfully")
        except Exception as e:
            self.logger.warning(f"AI Engine initialization failed: {e}")
            self.ai_enabled = False
    
    def predict_cost_trend(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """AI-powered cost prediction for future usage"""
        if not self.ai_enabled or not self.ai_engine:
            return {"error": "AI engine not available"}
        
        try:
            # Prepare historical data for AI analysis
            recent_calls = list(self.recent_calls)[-100:]  # Last 100 calls
            if len(recent_calls) < 10:
                return {"error": "Insufficient historical data for prediction"}
            
            # Extract features for prediction
            features = []
            for call in recent_calls:
                call_features = {
                    'hour': datetime.fromisoformat(call.timestamp).hour,
                    'cost': call.estimated_cost,
                    'tokens': call.input_tokens + call.output_tokens,
                    'model_type': 1 if 'gpt' in call.model.lower() else 0,
                    'agent': hash(call.agent or 'unknown') % 100,
                    'endpoint': hash(call.endpoint or 'unknown') % 100
                }
                features.append(call_features)
            
            # Use AI engine to analyze patterns and make predictions
            ai_metrics = {
                'cost_history': [f['cost'] for f in features],
                'hourly_distribution': [f['hour'] for f in features],
                'token_usage': [f['tokens'] for f in features],
                'model_distribution': [f['model_type'] for f in features]
            }
            
            # Process through AI engine
            ai_result = self.ai_engine.process_metrics(ai_metrics)
            
            # Generate cost predictions based on AI analysis
            current_hourly_rate = sum(f['cost'] for f in features[-10:]) / max(1, len(features[-10:]))
            
            predictions = []
            for hour in range(hours_ahead):
                # AI-enhanced prediction with pattern recognition
                base_prediction = current_hourly_rate
                
                # Apply AI insights for refinement
                if ai_result and ai_result.get('predictions'):
                    ai_confidence = ai_result.get('confidence', 0.5)
                    ai_modifier = 1.0 + (ai_confidence - 0.5) * 0.5  # -0.25 to +0.25 modifier
                    base_prediction *= ai_modifier
                
                # Add time-based patterns (higher usage during business hours)
                current_time = datetime.now() + timedelta(hours=hour)
                if 9 <= current_time.hour <= 17:  # Business hours
                    base_prediction *= 1.3
                elif 22 <= current_time.hour or current_time.hour <= 6:  # Night hours
                    base_prediction *= 0.6
                
                predictions.append({
                    'hour': current_time.strftime('%Y-%m-%d %H:00'),
                    'predicted_cost': base_prediction,
                    'confidence': ai_result.get('confidence', 0.7) if ai_result else 0.7
                })
            
            total_predicted = sum(p['predicted_cost'] for p in predictions)
            
            prediction_result = {
                'predictions': predictions,
                'total_predicted_cost': total_predicted,
                'current_budget_remaining': max(0, self.budget.daily_limit - self._get_daily_spent(datetime.now().strftime('%Y-%m-%d'))),
                'risk_assessment': self._assess_budget_risk(total_predicted),
                'ai_insights': ai_result.get('insights', []) if ai_result else [],
                'generated_at': datetime.now().isoformat()
            }
            
            # Store prediction for historical analysis
            self.cost_predictions.append(prediction_result)
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Cost prediction failed: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _assess_budget_risk(self, predicted_cost: float) -> Dict[str, Any]:
        """Assess budget risk based on predicted costs"""
        current_spent = self._get_daily_spent(datetime.now().strftime('%Y-%m-%d'))
        total_projected = current_spent + predicted_cost
        
        risk_percentage = (total_projected / self.budget.daily_limit) * 100
        
        if risk_percentage < 50:
            risk_level = "LOW"
        elif risk_percentage < 75:
            risk_level = "MODERATE"
        elif risk_percentage < 90:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'risk_level': risk_level,
            'risk_percentage': risk_percentage,
            'projected_total': total_projected,
            'budget_limit': self.budget.daily_limit,
            'recommendation': self._get_risk_recommendation(risk_level, risk_percentage)
        }
    
    def _get_risk_recommendation(self, risk_level: str, percentage: float) -> str:
        """Generate risk-based recommendations"""
        recommendations = {
            "LOW": "Continue normal operations. Budget usage is well within limits.",
            "MODERATE": "Monitor usage closely. Consider optimizing expensive operations.",
            "HIGH": "Implement cost controls. Review and optimize high-cost API calls.",
            "CRITICAL": "Immediate action required. Consider enabling predictive blocking or admin override."
        }
        
        base_rec = recommendations.get(risk_level, "Monitor usage carefully.")
        
        if percentage > 95:
            base_rec += " Budget limit will likely be exceeded today."
        
        return base_rec
    
    def analyze_usage_patterns(self) -> Dict[str, Any]:
        """AI-powered usage pattern analysis"""
        if not self.ai_enabled or not self.ai_engine:
            return {"error": "AI engine not available"}
        
        try:
            recent_calls = list(self.recent_calls)[-200:]  # Analyze last 200 calls
            
            if len(recent_calls) < 20:
                return {"error": "Insufficient data for pattern analysis"}
            
            # Extract usage patterns
            patterns = {
                'hourly_distribution': {},
                'model_preferences': {},
                'agent_usage': {},
                'endpoint_patterns': {},
                'cost_efficiency': []
            }
            
            for call in recent_calls:
                # Hourly distribution
                hour = datetime.fromisoformat(call.timestamp).hour
                patterns['hourly_distribution'][hour] = patterns['hourly_distribution'].get(hour, 0) + 1
                
                # Model preferences
                patterns['model_preferences'][call.model] = patterns['model_preferences'].get(call.model, 0) + 1
                
                # Agent usage
                if call.agent:
                    patterns['agent_usage'][call.agent] = patterns['agent_usage'].get(call.agent, 0) + 1
                
                # Endpoint patterns
                if call.endpoint:
                    patterns['endpoint_patterns'][call.endpoint] = patterns['endpoint_patterns'].get(call.endpoint, 0) + 1
                
                # Cost efficiency (cost per token)
                total_tokens = call.input_tokens + call.output_tokens
                if total_tokens > 0:
                    efficiency = call.estimated_cost / total_tokens * 1000  # Cost per 1K tokens
                    patterns['cost_efficiency'].append({
                        'model': call.model,
                        'efficiency': efficiency,
                        'timestamp': call.timestamp
                    })
            
            # Use AI engine to analyze patterns
            ai_analysis = self.ai_engine.process_metrics({
                'usage_patterns': patterns,
                'total_calls': len(recent_calls),
                'total_cost': sum(call.estimated_cost for call in recent_calls)
            })
            
            # Generate insights
            insights = []
            
            # Most active hours
            if patterns['hourly_distribution']:
                peak_hour = max(patterns['hourly_distribution'], key=patterns['hourly_distribution'].get)
                insights.append(f"Peak usage hour: {peak_hour}:00 with {patterns['hourly_distribution'][peak_hour]} calls")
            
            # Most used models
            if patterns['model_preferences']:
                top_model = max(patterns['model_preferences'], key=patterns['model_preferences'].get)
                insights.append(f"Most used model: {top_model} ({patterns['model_preferences'][top_model]} calls)")
            
            # Cost efficiency analysis
            if patterns['cost_efficiency']:
                avg_efficiency = sum(ce['efficiency'] for ce in patterns['cost_efficiency']) / len(patterns['cost_efficiency'])
                insights.append(f"Average cost efficiency: ${avg_efficiency:.4f} per 1K tokens")
                
                # Find most efficient model
                model_efficiency = {}
                for ce in patterns['cost_efficiency']:
                    model = ce['model']
                    if model not in model_efficiency:
                        model_efficiency[model] = []
                    model_efficiency[model].append(ce['efficiency'])
                
                for model, efficiencies in model_efficiency.items():
                    avg_eff = sum(efficiencies) / len(efficiencies)
                    model_efficiency[model] = avg_eff
                
                if model_efficiency:
                    most_efficient = min(model_efficiency, key=model_efficiency.get)
                    insights.append(f"Most cost-efficient model: {most_efficient} (${model_efficiency[most_efficient]:.4f}/1K tokens)")
            
            analysis_result = {
                'patterns': patterns,
                'insights': insights,
                'ai_analysis': ai_analysis,
                'recommendations': self._generate_optimization_recommendations(patterns),
                'generated_at': datetime.now().isoformat()
            }
            
            # Store analysis for historical tracking
            self.usage_patterns.append(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_optimization_recommendations(self, patterns: Dict) -> List[str]:
        """Generate optimization recommendations based on usage patterns"""
        recommendations = []
        
        # Model optimization recommendations
        if patterns.get('cost_efficiency'):
            model_costs = {}
            for ce in patterns['cost_efficiency']:
                model = ce['model']
                if model not in model_costs:
                    model_costs[model] = []
                model_costs[model].append(ce['efficiency'])
            
            for model, costs in model_costs.items():
                avg_cost = sum(costs) / len(costs)
                if avg_cost > 0.02:  # High cost per 1K tokens
                    recommendations.append(f"Consider switching from {model} to a more cost-effective alternative for non-critical tasks")
        
        # Usage time optimization
        if patterns.get('hourly_distribution'):
            night_usage = sum(count for hour, count in patterns['hourly_distribution'].items() if 22 <= hour or hour <= 6)
            total_usage = sum(patterns['hourly_distribution'].values())
            
            if night_usage / total_usage > 0.3:  # More than 30% night usage
                recommendations.append("High night-time usage detected. Consider batching non-urgent operations during off-peak hours for potential cost savings")
        
        # Agent usage balance
        if patterns.get('agent_usage') and len(patterns['agent_usage']) > 1:
            usage_values = list(patterns['agent_usage'].values())
            max_usage = max(usage_values)
            min_usage = min(usage_values)
            
            if max_usage / min_usage > 3:  # Highly imbalanced usage
                recommendations.append("Imbalanced agent usage detected. Consider load balancing API calls across agents")
        
        return recommendations
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current usage statistics for dashboards"""
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "total_tokens": self.stats.total_input_tokens + self.stats.total_output_tokens,
            "total_cost": self.stats.total_estimated_cost,
            "calls_by_model": dict(self.stats.calls_by_model),
            "calls_by_type": dict(self.stats.calls_by_type),
            "calls_by_component": dict(self.stats.calls_by_component),
            "cost_by_model": dict(self.stats.cost_by_model),
            "cost_by_type": dict(self.stats.cost_by_type),
            "hourly_calls": dict(self.stats.hourly_calls),
            "hourly_cost": dict(self.stats.hourly_cost),
            # IRONCLAD EXTRACTION: Added agent and endpoint breakdown from File A
            "calls_by_agent": dict(self.stats.calls_by_agent),
            "cost_by_agent": dict(self.stats.cost_by_agent),
            "calls_by_endpoint": dict(self.stats.calls_by_endpoint),
            "cost_by_endpoint": dict(self.stats.cost_by_endpoint),
            "session_calls": self.stats.session_calls,
            "session_cost": self.stats.session_cost,
            "budget_status": self._get_budget_status(),
            "recent_calls": [asdict(call) for call in self.current_session_calls[-10:]],
            # IRONCLAD EXTRACTION: Added top endpoints from File A
            "top_endpoints": self.get_top_endpoints(),
            # HOUR 3 ENHANCEMENT: Added alert and override information
            "recent_alerts": list(self.alert_history)[-10:],
            "active_overrides": [o for o in self.admin_overrides if datetime.now() < datetime.fromisoformat(o['expires_at'])],
            "notification_config": {
                "email_enabled": self.notification_config.email_enabled,
                "webhook_enabled": self.notification_config.webhook_enabled,
                "dashboard_alerts": self.notification_config.dashboard_alerts,
                "alert_thresholds": self.notification_config.alert_thresholds
            },
            # HOUR 4 ENHANCEMENT: Added AI intelligence data
            "ai_enabled": self.ai_enabled,
            "ai_predictions": list(self.cost_predictions)[-5:] if self.cost_predictions else [],
            "ai_insights": list(self.ai_insights)[-5:] if self.ai_insights else [],
            "usage_patterns": list(self.usage_patterns)[-3:] if self.usage_patterns else []
        }
    
    def get_top_endpoints(self) -> List[Dict[str, Any]]:
        """IRONCLAD EXTRACTION: Get top endpoints by usage from File A"""
        # Sort endpoints by cost
        sorted_endpoints = sorted(
            self.stats.cost_by_endpoint.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                'endpoint': endpoint,
                'calls': self.stats.calls_by_endpoint.get(endpoint, 0),
                'cost': cost,
                'avg_cost': cost / self.stats.calls_by_endpoint.get(endpoint, 1)
            }
            for endpoint, cost in sorted_endpoints[:10]
        ]
    
    def _get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        
        daily_spent = self._get_daily_spent(today)
        hourly_spent = self.stats.hourly_cost.get(current_hour, 0.0)
        
        return {
            "daily": {
                "spent": daily_spent,
                "limit": self.budget.daily_limit,
                "percentage": (daily_spent / self.budget.daily_limit * 100) if self.budget.daily_limit > 0 else 0
            },
            "hourly": {
                "spent": hourly_spent,
                "limit": self.budget.hourly_limit,
                "percentage": (hourly_spent / self.budget.hourly_limit * 100) if self.budget.hourly_limit > 0 else 0
            },
            "total": {
                "spent": self.stats.total_estimated_cost,
                "limit": self.budget.total_limit,
                "percentage": (self.stats.total_estimated_cost / self.budget.total_limit * 100) if self.budget.total_limit > 0 else 0
            }
        }
    
    def set_budget(self, 
                  daily_limit: Optional[float] = None,
                  hourly_limit: Optional[float] = None,
                  per_call_limit: Optional[float] = None,
                  total_limit: Optional[float] = None,
                  auto_stop: Optional[bool] = None):
        """Update budget configuration"""
        if daily_limit is not None:
            self.budget.daily_limit = daily_limit
        if hourly_limit is not None:
            self.budget.hourly_limit = hourly_limit
        if per_call_limit is not None:
            self.budget.per_call_limit = per_call_limit
        if total_limit is not None:
            self.budget.total_limit = total_limit
        if auto_stop is not None:
            self.budget.auto_stop = auto_stop
        
        self.logger.info(f"Budget updated: Daily=${self.budget.daily_limit}, Hourly=${self.budget.hourly_limit}")
    
    def export_usage_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export comprehensive usage report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": asdict(self.stats),
            "budget": asdict(self.budget),
            "budget_status": self._get_budget_status(),
            "top_expensive_calls": self._get_top_expensive_calls(10),
            "recommendations": self._generate_cost_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _get_top_expensive_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most expensive API calls"""
        sorted_calls = sorted(self.current_session_calls, key=lambda x: x.estimated_cost, reverse=True)
        return [asdict(call) for call in sorted_calls[:limit]]
    
    def _generate_cost_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Check for expensive models
        if "gpt-4" in self.stats.cost_by_model and self.stats.cost_by_model["gpt-4"] > self.stats.total_estimated_cost * 0.5:
            recommendations.append("Consider using GPT-3.5-Turbo for non-critical tasks to reduce costs")
        
        if "claude-3-opus" in self.stats.cost_by_model:
            recommendations.append("Consider using Claude-3-Haiku for simpler tasks instead of Opus")
        
        # Check for high-frequency components
        top_component = max(self.stats.calls_by_component.items(), key=lambda x: x[1])[0] if self.stats.calls_by_component else None
        if top_component:
            recommendations.append(f"Component '{top_component}' has the most API calls - consider caching or batching")
        
        # Check token usage
        avg_tokens = (self.stats.total_input_tokens + self.stats.total_output_tokens) / max(1, self.stats.total_calls)
        if avg_tokens > 2000:
            recommendations.append("High average token usage detected - consider prompt optimization")
        
        return recommendations


# Global tracker instance
_tracker = None

def get_api_tracker() -> APIUsageTracker:
    """Get the global API usage tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = APIUsageTracker()
    return _tracker


def track_api_call(model: str,
                  call_type: str,
                  purpose: str,
                  component: str,
                  input_tokens: int,
                  output_tokens: int,
                  endpoint: str = None,
                  agent: str = None,
                  **kwargs) -> Tuple[bool, str, float]:
    """
    Convenience function to track an API call
    
    MUST be called before ANY AI/LLM API call
    """
    tracker = get_api_tracker()
    
    # Convert string to enum if needed
    if isinstance(call_type, str):
        try:
            call_type = APICallType[call_type.upper()]
        except:
            call_type = APICallType.OTHER
    
    return tracker.track_api_call(model, call_type, purpose, component, input_tokens, output_tokens, endpoint=endpoint, agent=agent, **kwargs)


def pre_check_cost(model: str, estimated_input_tokens: int, estimated_output_tokens: int) -> Tuple[bool, str, float]:
    """Pre-check if an API call would be within budget"""
    tracker = get_api_tracker()
    return tracker.pre_check_api_call(model, estimated_input_tokens, estimated_output_tokens)


def get_usage_stats() -> Dict[str, Any]:
    """Get current usage statistics"""
    tracker = get_api_tracker()
    return tracker.get_current_stats()


def set_api_budget(daily_limit: float = None, hourly_limit: float = None, auto_stop: bool = None):
    """Set API usage budget"""
    tracker = get_api_tracker()
    tracker.set_budget(daily_limit=daily_limit, hourly_limit=hourly_limit, auto_stop=auto_stop)


# IRONCLAD EXTRACTION: Added global convenience functions from File A
def check_budget_before_call(estimated_tokens: int, model: str) -> Dict[str, Any]:
    """Check if we can afford an operation before making the call"""
    tracker = get_api_tracker()
    return tracker.pre_check_api_call(model, estimated_tokens, estimated_tokens // 2)

def get_usage_dashboard_data() -> Dict[str, Any]:
    """Get data for dashboard display"""
    tracker = get_api_tracker()
    return tracker.get_current_stats()

def get_remaining_budget() -> Dict[str, Any]:
    """Get remaining budget information"""
    tracker = get_api_tracker()
    budget_status = tracker._get_budget_status()
    
    return {
        'daily_remaining': max(0, tracker.budget.daily_limit - budget_status['daily']['spent']),
        'hourly_remaining': max(0, tracker.budget.hourly_limit - budget_status['hourly']['spent']),
        'daily_percentage_used': budget_status['daily']['percentage'],
        'hourly_percentage_used': budget_status['hourly']['percentage']
    }

# HOUR 3 ENHANCEMENT: Global functions for advanced features
def configure_notifications(email_enabled: bool = False, 
                          email_recipients: List[str] = None,
                          webhook_enabled: bool = False,
                          webhook_url: str = None):
    """Configure notification settings"""
    tracker = get_api_tracker()
    tracker.notification_config.email_enabled = email_enabled
    tracker.notification_config.webhook_enabled = webhook_enabled
    
    if email_recipients:
        tracker.notification_config.email_recipients = email_recipients
    if webhook_url:
        tracker.notification_config.webhook_url = webhook_url
    
    return {"status": "notifications_configured", "config": tracker.notification_config.__dict__}

def admin_override(reason: str, duration_minutes: int = 60, admin_id: str = "system"):
    """Create admin override for budget limits"""
    tracker = get_api_tracker()
    return tracker.admin_override_budget(reason, duration_minutes, admin_id)

def get_alert_history(limit: int = 20) -> List[Dict]:
    """Get recent alert history"""
    tracker = get_api_tracker()
    return list(tracker.alert_history)[-limit:]

def get_active_overrides() -> List[Dict]:
    """Get currently active admin overrides"""
    tracker = get_api_tracker()
    current_time = datetime.now()
    return [o for o in tracker.admin_overrides if current_time < datetime.fromisoformat(o['expires_at'])]


if __name__ == "__main__":
    # HOUR 3 ENHANCEMENT: Advanced Alert System Testing
    print("API USAGE TRACKER - ENHANCED ALERT SYSTEM TEST")
    print("=" * 60)
    
    # Set conservative budget for testing alert thresholds
    set_api_budget(daily_limit=2.0, hourly_limit=1.0, auto_stop=True)
    
    # Configure notifications for testing
    configure_notifications(
        email_enabled=True,
        email_recipients=["admin@testmaster.ai"],
        webhook_enabled=True,
        webhook_url="https://hooks.testmaster.ai/budget-alerts"
    )
    
    print("BUDGET CONFIGURATION:")
    print(f"   Daily Limit: $2.00")
    print(f"   Hourly Limit: $1.00") 
    print(f"   Alert Thresholds: 50%, 75%, 90%, 95%")
    print()
    
    # Test progression through alert levels
    test_calls = [
        {"tokens": 500, "desc": "Normal call (should be safe)"},
        {"tokens": 2000, "desc": "Moderate call (may trigger 50% warning)"},  
        {"tokens": 3000, "desc": "Large call (may trigger 75% critical)"},
        {"tokens": 2000, "desc": "Push to 90% danger level"},
        {"tokens": 1000, "desc": "Approach 95% extreme level"},
        {"tokens": 500, "desc": "This should be blocked at 100%"}
    ]
    
    for i, call_info in enumerate(test_calls, 1):
        print(f"TEST CALL {i}: {call_info['desc']}")
        
        allowed, message, cost = track_api_call(
            model="gpt-3.5-turbo",
            call_type="testing",
            purpose=f"Test call {i} - Alert system verification",
            component="alert_test_module",
            input_tokens=call_info['tokens'],
            output_tokens=call_info['tokens'] // 3,
            agent="alpha",
            endpoint="/test/alert-system"
        )
        
        print(f"   Result: {'ALLOWED' if allowed else 'BLOCKED'}")
        print(f"   Cost: ${cost:.4f}")
        print(f"   Message: {message}")
        
        if not allowed:
            print(f"   TESTING ADMIN OVERRIDE...")
            success, override_msg = admin_override(
                reason="Testing alert system - critical test continuation needed",
                duration_minutes=10,
                admin_id="test_admin"
            )
            print(f"   Override: {'SUCCESS' if success else 'FAILED'} - {override_msg}")
            
            if success:
                # Retry the call with override active
                allowed_override, message_override, cost_override = track_api_call(
                    model="gpt-3.5-turbo",
                    call_type="testing",
                    purpose=f"Test call {i} - With admin override",
                    component="alert_test_module",
                    input_tokens=call_info['tokens'],
                    output_tokens=call_info['tokens'] // 3,
                    agent="alpha",
                    endpoint="/test/alert-system-override"
                )
                print(f"   Override Call: {'ALLOWED' if allowed_override else 'STILL BLOCKED'}")
        
        print()
    
    # Display comprehensive stats
    stats = get_usage_stats()
    print("COMPREHENSIVE STATISTICS:")
    print(f"   Total Calls: {stats['total_calls']}")
    print(f"   Total Cost: ${stats['total_cost']:.4f}")
    print(f"   Session Calls: {stats['session_calls']}")
    print(f"   Session Cost: ${stats['session_cost']:.4f}")
    
    budget_status = stats['budget_status']
    print(f"\nBUDGET STATUS:")
    print(f"   Daily: ${budget_status['daily']['spent']:.4f} / ${budget_status['daily']['limit']:.2f} ({budget_status['daily']['percentage']:.1f}%)")
    print(f"   Hourly: ${budget_status['hourly']['spent']:.4f} / ${budget_status['hourly']['limit']:.2f} ({budget_status['hourly']['percentage']:.1f}%)")
    
    # Show alert history
    alerts = get_alert_history(5)
    if alerts:
        print(f"\nRECENT ALERTS ({len(alerts)}):")
        for alert in alerts[-3:]:  # Show last 3 alerts
            print(f"   [{alert['level'].upper()}] {alert['message']} (${alert['cost']:.4f})")
    
    # Show active overrides
    active_overrides = get_active_overrides()
    if active_overrides:
        print(f"\nACTIVE OVERRIDES ({len(active_overrides)}):")
        for override in active_overrides:
            print(f"   Admin: {override['admin_id']} - {override['reason']}")
            print(f"   Expires: {override['expires_at']}")
    
    print("\n" + "=" * 60)
    print("ENHANCED API USAGE TRACKER WITH ALERT SYSTEM READY!")
    print("   - Multi-level thresholds (50%, 75%, 90%, 95%)")
    print("   - Email & webhook notifications") 
    print("   - Admin override system with audit trails")
    print("   - Predictive blocking for large calls")
    print("   - Real-time dashboard integration")
    print("=" * 60)