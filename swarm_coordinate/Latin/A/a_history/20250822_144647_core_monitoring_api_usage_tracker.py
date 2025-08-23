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
    """Cost warning levels"""
    SAFE = "safe"           # Under 25% of budget
    CAUTION = "caution"     # 25-50% of budget
    WARNING = "warning"     # 50-75% of budget
    CRITICAL = "critical"   # 75-90% of budget
    DANGER = "danger"       # 90-95% of budget
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


@dataclass
class CostBudget:
    """Budget configuration for API costs"""
    daily_limit: float = 10.0  # $10 default daily limit
    hourly_limit: float = 2.0   # $2 default hourly limit
    per_call_limit: float = 0.5  # $0.50 per call limit
    total_limit: float = 100.0   # $100 total limit
    warning_threshold: float = 0.75  # Warn at 75% of limits
    auto_stop: bool = True  # Automatically stop when limit reached


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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
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
                metadata TEXT
            )
        """)
        
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
        
        # Check budgets
        warning_level, budget_message = self._check_budgets(estimated_cost)
        
        if warning_level == CostWarningLevel.EXCEEDED and self.budget.auto_stop:
            self.logger.error(f"API CALL BLOCKED: Budget exceeded! {budget_message}")
            return False, f"BLOCKED: {budget_message}", estimated_cost
        
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
            **kwargs
        )
        
        # Track the call
        self._record_call(api_call)
        
        # Notify dashboards
        self._notify_dashboards(api_call)
        
        # Log warning if needed
        if warning_level in [CostWarningLevel.WARNING, CostWarningLevel.CRITICAL, CostWarningLevel.DANGER]:
            self.logger.warning(f"COST WARNING ({warning_level.value}): {budget_message}")
        
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
        
        # Calculate warning level based on percentage
        if highest_level == CostWarningLevel.SAFE:
            daily_percentage = (daily_spent + additional_cost) / self.budget.daily_limit
            if daily_percentage > 0.9:
                highest_level = CostWarningLevel.DANGER
            elif daily_percentage > 0.75:
                highest_level = CostWarningLevel.CRITICAL
            elif daily_percentage > 0.5:
                highest_level = CostWarningLevel.WARNING
            elif daily_percentage > 0.25:
                highest_level = CostWarningLevel.CAUTION
        
        message = " | ".join(messages) if messages else f"Within budget (${additional_cost:.4f})"
        return highest_level, message
    
    def _record_call(self, api_call: APICall):
        """Record an API call in database and memory"""
        # Update in-memory stats
        self.current_session_calls.append(api_call)
        self.stats.total_calls += 1
        
        if api_call.success:
            self.stats.successful_calls += 1
        else:
            self.stats.failed_calls += 1
        
        self.stats.total_input_tokens += api_call.input_tokens
        self.stats.total_output_tokens += api_call.output_tokens
        self.stats.total_estimated_cost += api_call.estimated_cost
        
        # Update categorized stats
        model_key = api_call.model
        type_key = api_call.call_type.value
        
        self.stats.calls_by_model[model_key] = self.stats.calls_by_model.get(model_key, 0) + 1
        self.stats.calls_by_type[type_key] = self.stats.calls_by_type.get(type_key, 0) + 1
        self.stats.calls_by_component[api_call.component] = self.stats.calls_by_component.get(api_call.component, 0) + 1
        
        self.stats.cost_by_model[model_key] = self.stats.cost_by_model.get(model_key, 0.0) + api_call.estimated_cost
        self.stats.cost_by_type[type_key] = self.stats.cost_by_type.get(type_key, 0.0) + api_call.estimated_cost
        
        # Update hourly stats
        hour_key = api_call.timestamp.strftime("%Y-%m-%d %H:00")
        self.stats.hourly_calls[hour_key] = self.stats.hourly_calls.get(hour_key, 0) + 1
        self.stats.hourly_cost[hour_key] = self.stats.hourly_cost.get(hour_key, 0.0) + api_call.estimated_cost
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO api_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(api_call.metadata)
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
            "budget_status": self._get_budget_status(),
            "recent_calls": [asdict(call) for call in self.current_session_calls[-10:]]
        }
    
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
    
    return tracker.track_api_call(model, call_type, purpose, component, input_tokens, output_tokens, **kwargs)


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


if __name__ == "__main__":
    # Test the tracker
    print("API Usage Tracker Test")
    print("=" * 60)
    
    # Set conservative budget for testing
    set_api_budget(daily_limit=5.0, hourly_limit=1.0, auto_stop=True)
    
    # Test tracking a call
    allowed, message, cost = track_api_call(
        model="gpt-3.5-turbo",
        call_type="analysis",
        purpose="Test code analysis",
        component="test_module",
        input_tokens=500,
        output_tokens=200
    )
    
    print(f"Call allowed: {allowed}")
    print(f"Message: {message}")
    print(f"Estimated cost: ${cost:.4f}")
    
    # Get stats
    stats = get_usage_stats()
    print("\nCurrent Stats:")
    print(json.dumps(stats, indent=2, default=str))
    
    print("\n" + "=" * 60)
    print("CRITICAL: This tracker MUST be integrated with all AI tools!")
    print("=" * 60)