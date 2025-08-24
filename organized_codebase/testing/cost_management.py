"""
Cost Management Module
====================

Focused module for LLM cost tracking and management.
Extracted from unified_monitor.py for better modularization.

Author: TestMaster Phase 1C Consolidation
"""

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class LLMCall:
    """Represents an LLM API call with cost tracking"""
    call_id: str
    model: str
    provider: str
    timestamp: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CostTracker:
    """Advanced cost tracking for LLM usage in testing"""
    
    def __init__(self):
        self.session_costs: Dict[str, float] = defaultdict(float)
        self.model_pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }
        self.daily_limits = {
            "total_cost": 100.0,
            "tokens": 1000000,
            "calls": 10000
        }
        self.current_usage = {
            "total_cost": 0.0,
            "tokens": 0,
            "calls": 0,
            "reset_time": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        }
        self.lock = threading.Lock()

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for LLM call"""
        if model not in self.model_pricing:
            return 0.0
        
        pricing = self.model_pricing[model]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def track_llm_call(self, session_id: str, call: LLMCall):
        """Track an LLM call for cost monitoring"""
        with self.lock:
            # Reset daily usage if needed
            if datetime.now().date() > self.current_usage["reset_time"].date():
                self._reset_daily_usage()
            
            # Update usage
            self.current_usage["total_cost"] += call.cost
            self.current_usage["tokens"] += call.total_tokens
            self.current_usage["calls"] += 1
            self.session_costs[session_id] += call.cost

    def _reset_daily_usage(self):
        """Reset daily usage counters"""
        self.current_usage.update({
            "total_cost": 0.0,
            "tokens": 0,
            "calls": 0,
            "reset_time": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        })

    def check_limits(self) -> Dict[str, Any]:
        """Check if usage is approaching limits"""
        warnings = []
        
        for limit_type, limit_value in self.daily_limits.items():
            current = self.current_usage[limit_type]
            percentage = (current / limit_value) * 100
            
            if percentage >= 90:
                warnings.append({
                    "type": limit_type,
                    "current": current,
                    "limit": limit_value,
                    "percentage": percentage,
                    "level": "CRITICAL"
                })
            elif percentage >= 75:
                warnings.append({
                    "type": limit_type,
                    "current": current,
                    "limit": limit_value,
                    "percentage": percentage,
                    "level": "WARNING"
                })
        
        return {
            "within_limits": len(warnings) == 0,
            "warnings": warnings,
            "usage": self.current_usage.copy()
        }

    def get_session_cost_breakdown(self, session_id: str) -> Dict[str, Any]:
        """Get detailed cost breakdown for a session"""
        return {
            "total_cost": self.session_costs.get(session_id, 0.0),
            "daily_usage": self.current_usage["total_cost"],
            "daily_limit": self.daily_limits["total_cost"],
            "remaining_budget": self.daily_limits["total_cost"] - self.current_usage["total_cost"]
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get overall cost summary"""
        return {
            "daily_usage": self.current_usage.copy(),
            "daily_limits": self.daily_limits.copy(),
            "total_sessions": len(self.session_costs),
            "average_session_cost": (
                sum(self.session_costs.values()) / len(self.session_costs)
                if self.session_costs else 0.0
            )
        }

class CostManager:
    """High-level cost management interface"""
    
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.llm_calls: Dict[str, LLMCall] = {}
    
    def track_llm_call(self, session_id: str, model: str, provider: str,
                      prompt_tokens: int, completion_tokens: int,
                      latency: float, success: bool, error: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track an LLM API call"""
        call_id = f"llm_{uuid.uuid4().hex[:8]}"
        cost = self.cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
        
        llm_call = LLMCall(
            call_id=call_id,
            model=model,
            provider=provider,
            timestamp=time.time(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency=latency,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        self.cost_tracker.track_llm_call(session_id, llm_call)
        self.llm_calls[call_id] = llm_call
        
        return call_id
    
    def get_session_costs(self, session_id: str) -> Dict[str, Any]:
        """Get cost information for a session"""
        return self.cost_tracker.get_session_cost_breakdown(session_id)
    
    def check_budget_status(self) -> Dict[str, Any]:
        """Check current budget status"""
        return self.cost_tracker.check_limits()

# Export key components
__all__ = [
    'LLMCall',
    'CostTracker',
    'CostManager'
]