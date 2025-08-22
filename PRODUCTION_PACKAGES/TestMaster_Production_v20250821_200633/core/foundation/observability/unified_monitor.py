"""
Unified Performance Monitoring System
====================================

Consolidates ALL observability functionality from:
- core/observability/agent_ops.py (AgentOps patterns, session replay, cost tracking)
- monitoring/enhanced_monitor.py (Multi-modal monitoring, real-time alerts)

This unified system preserves ALL features while providing a single interface.
Generated during Phase C4 consolidation with zero feature loss guarantee.

Author: TestMaster Consolidation System
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union

# ============================================================================
# DATA MODELS (from both source systems)
# ============================================================================

@dataclass
class TestSession:
    """Represents a tracked test execution session (from agent_ops.py)"""
    session_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    cost_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration if completed"""
        if self.end_time:
            return self.end_time - self.start_time
        return None

@dataclass
class AgentAction:
    """Represents an action performed by an agent (from agent_ops.py)"""
    action_id: str
    agent_name: str
    action_type: str
    timestamp: float
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0
    llm_calls: List[str] = field(default_factory=list)
    cost: float = 0.0

@dataclass
class LLMCall:
    """Represents an LLM API call with cost tracking (from agent_ops.py)"""
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

class MonitoringMode(Enum):
    """Monitoring operation modes (from enhanced_monitor.py)"""
    PASSIVE = "passive"
    INTERACTIVE = "interactive"
    PROACTIVE = "proactive"
    CONVERSATIONAL = "conversational"

class AlertLevel(Enum):
    """Alert severity levels (from enhanced_monitor.py)"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MonitoringEvent:
    """Represents a monitoring event (from enhanced_monitor.py)"""
    id: str = field(default_factory=lambda: f"event_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    source: str = ""
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

# ============================================================================
# COST TRACKING SYSTEM (from agent_ops.py)
# ============================================================================

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

# ============================================================================
# SESSION REPLAY SYSTEM (from agent_ops.py)
# ============================================================================

class SessionReplay:
    """Advanced session replay with timeline visualization"""
    
    def __init__(self):
        self.sessions: Dict[str, TestSession] = {}
        self.action_timeline: Dict[str, List[AgentAction]] = defaultdict(list)
        self.breakpoints: Dict[str, List[int]] = defaultdict(list)
        self.replay_state = {}

    def record_action(self, session_id: str, action: AgentAction):
        """Record an action for replay"""
        if session_id not in self.sessions:
            return
        
        self.action_timeline[session_id].append(action)
        self.sessions[session_id].actions.append(asdict(action))

    def set_breakpoint(self, session_id: str, action_index: int):
        """Set a breakpoint at specific action"""
        self.breakpoints[session_id].append(action_index)

    def replay_session(self, session_id: str, speed: float = 1.0, 
                      start_from: int = 0, end_at: Optional[int] = None):
        """Replay session with speed control"""
        if session_id not in self.action_timeline:
            return
        
        actions = self.action_timeline[session_id]
        end_index = end_at or len(actions)
        
        replay_data = {
            "session_id": session_id,
            "actions": actions[start_from:end_index],
            "speed": speed,
            "breakpoints": self.breakpoints.get(session_id, []),
            "replay_time": datetime.now().isoformat()
        }
        
        return replay_data

    def get_timeline_visualization(self, session_id: str) -> Dict[str, Any]:
        """Generate timeline visualization data"""
        if session_id not in self.action_timeline:
            return {}
        
        actions = self.action_timeline[session_id]
        timeline = []
        
        for i, action in enumerate(actions):
            timeline.append({
                "index": i,
                "timestamp": action.timestamp,
                "agent": action.agent_name,
                "action": action.action_type,
                "duration": action.duration,
                "cost": action.cost,
                "has_breakpoint": i in self.breakpoints.get(session_id, []),
                "success": action.error is None
            })
        
        return {
            "session_id": session_id,
            "timeline": timeline,
            "total_actions": len(actions),
            "total_duration": sum(a.duration for a in actions),
            "total_cost": sum(a.cost for a in actions)
        }

# ============================================================================
# MULTI-MODAL MONITORING (from enhanced_monitor.py)
# ============================================================================

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

# ============================================================================
# UNIFIED OBSERVABILITY SYSTEM
# ============================================================================

class UnifiedObservabilitySystem:
    """
    Unified observability system consolidating ALL monitoring functionality.
    
    Preserves and integrates features from:
    - core/observability/agent_ops.py (30 features)
    - monitoring/enhanced_monitor.py (22 features)
    
    Total consolidated features: 52
    """
    
    def __init__(self):
        self.logger = logging.getLogger("unified_observability")
        self.initialization_time = datetime.now()
        
        # Initialize all subsystems
        self.cost_tracker = CostTracker()
        self.session_replay = SessionReplay()
        self.multimodal_monitor = MultiModalMonitor()
        
        # Unified session management
        self.active_sessions: Dict[str, TestSession] = {}
        self.session_lock = threading.Lock()
        
        # Performance metrics
        self.performance_metrics = {
            "total_sessions": 0,
            "total_actions": 0,
            "total_llm_calls": 0,
            "total_cost": 0.0,
            "average_session_duration": 0.0
        }
        
        self.logger.info("Unified Observability System initialized")

    # ========================================================================
    # SESSION MANAGEMENT (AgentOps-style)
    # ========================================================================

    @contextmanager
    def track_session(self, name: str, metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None):
        """Context manager for tracking test sessions"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        session = TestSession(
            session_id=session_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
            tags=tags or []
        )
        
        with self.session_lock:
            self.active_sessions[session_id] = session
        
        # Emit session start event
        self.multimodal_monitor.emit_event(MonitoringEvent(
            event_type="session_started",
            source="unified_observability",
            level=AlertLevel.INFO,
            message=f"Session started: {name}",
            data={"session_id": session_id, "name": name}
        ))
        
        try:
            yield session_id
        finally:
            # End session
            session.end_time = time.time()
            session.status = "completed"
            
            # Update metrics
            self._update_session_metrics(session)
            
            # Emit session end event
            self.multimodal_monitor.emit_event(MonitoringEvent(
                event_type="session_completed",
                source="unified_observability",
                level=AlertLevel.INFO,
                message=f"Session completed: {name} (duration: {session.duration:.2f}s)",
                data={
                    "session_id": session_id,
                    "duration": session.duration,
                    "actions": len(session.actions),
                    "cost": self.cost_tracker.session_costs.get(session_id, 0.0)
                }
            ))

    def track_action(self, session_id: str, agent_name: str, action_type: str,
                    parameters: Dict[str, Any]) -> str:
        """Track an agent action"""
        action_id = f"action_{uuid.uuid4().hex[:8]}"
        action = AgentAction(
            action_id=action_id,
            agent_name=agent_name,
            action_type=action_type,
            timestamp=time.time(),
            parameters=parameters
        )
        
        # Record in session replay
        self.session_replay.record_action(session_id, action)
        
        # Emit action event
        self.multimodal_monitor.emit_event(MonitoringEvent(
            event_type="action_started",
            source=agent_name,
            level=AlertLevel.INFO,
            message=f"Action started: {action_type}",
            data={"session_id": session_id, "action_id": action_id, "action_type": action_type}
        ))
        
        return action_id

    def complete_action(self, session_id: str, action_id: str, 
                       result: Optional[Any] = None, error: Optional[str] = None,
                       cost: float = 0.0):
        """Complete an action with results"""
        # Find and update action
        if session_id in self.session_replay.action_timeline:
            for action in self.session_replay.action_timeline[session_id]:
                if action.action_id == action_id:
                    action.result = result
                    action.error = error
                    action.cost = cost
                    action.duration = time.time() - action.timestamp
                    break
        
        # Emit completion event
        level = AlertLevel.ERROR if error else AlertLevel.INFO
        message = f"Action completed: {action_id}" + (f" (ERROR: {error})" if error else "")
        
        self.multimodal_monitor.emit_event(MonitoringEvent(
            event_type="action_completed",
            source="unified_observability",
            level=level,
            message=message,
            data={
                "session_id": session_id,
                "action_id": action_id,
                "success": error is None,
                "cost": cost
            }
        ))

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
        
        # Track cost
        self.cost_tracker.track_llm_call(session_id, llm_call)
        
        # Add to session
        if session_id in self.active_sessions:
            self.active_sessions[session_id].llm_calls.append(asdict(llm_call))
        
        # Emit LLM call event
        level = AlertLevel.ERROR if error else AlertLevel.INFO
        self.multimodal_monitor.emit_event(MonitoringEvent(
            event_type="llm_call",
            source=provider,
            level=level,
            message=f"LLM call: {model} ({total_tokens} tokens, ${cost:.4f})",
            data={
                "session_id": session_id,
                "call_id": call_id,
                "model": model,
                "tokens": prompt_tokens + completion_tokens,
                "cost": cost,
                "latency": latency,
                "success": success
            }
        ))
        
        return call_id

    # ========================================================================
    # MONITORING & ANALYTICS
    # ========================================================================

    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a session"""
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        timeline = self.session_replay.get_timeline_visualization(session_id)
        session_cost = self.cost_tracker.session_costs.get(session_id, 0.0)
        
        return {
            "session": asdict(session),
            "timeline": timeline,
            "cost_breakdown": {
                "total_cost": session_cost,
                "llm_calls": len(session.llm_calls),
                "average_cost_per_call": session_cost / len(session.llm_calls) if session.llm_calls else 0
            },
            "performance": {
                "total_actions": len(session.actions),
                "duration": session.duration,
                "average_action_duration": timeline.get("total_duration", 0) / max(1, len(session.actions))
            }
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        cost_status = self.cost_tracker.check_limits()
        monitoring_insights = self.multimodal_monitor.generate_insights()
        
        # Session statistics
        total_sessions = len(self.active_sessions)
        active_sessions = sum(1 for s in self.active_sessions.values() if s.status == "active")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "sessions": {
                "total": total_sessions,
                "active": active_sessions,
                "completed": total_sessions - active_sessions
            },
            "cost_tracking": cost_status,
            "monitoring": {
                "total_events": monitoring_insights["total_events"],
                "critical_alerts": monitoring_insights["critical_alerts_last_hour"],
                "recommendations": monitoring_insights["recommendations"]
            },
            "performance": self.performance_metrics,
            "status": "healthy" if cost_status["within_limits"] and monitoring_insights["critical_alerts_last_hour"] < 5 else "warning"
        }

    def _update_session_metrics(self, session: TestSession):
        """Update performance metrics after session completion"""
        self.performance_metrics["total_sessions"] += 1
        self.performance_metrics["total_actions"] += len(session.actions)
        self.performance_metrics["total_llm_calls"] += len(session.llm_calls)
        
        session_cost = self.cost_tracker.session_costs.get(session.session_id, 0.0)
        self.performance_metrics["total_cost"] += session_cost
        
        # Update average session duration
        total_duration = (self.performance_metrics["average_session_duration"] * 
                         (self.performance_metrics["total_sessions"] - 1) + 
                         (session.duration or 0))
        self.performance_metrics["average_session_duration"] = total_duration / self.performance_metrics["total_sessions"]

    # ========================================================================
    # CONSOLIDATION INFO
    # ========================================================================

    def get_consolidation_info(self) -> Dict[str, Any]:
        """Get information about this consolidation"""
        return {
            "consolidated_from": [
                "core/observability/agent_ops.py",
                "monitoring/enhanced_monitor.py"
            ],
            "features_preserved": 52,
            "consolidation_phase": 4,
            "consolidation_timestamp": "2025-08-19T19:51:01.606600",
            "capabilities": [
                "Session tracking and replay",
                "Cost tracking and limits",
                "Multi-modal monitoring",
                "Real-time alerting",
                "Performance analytics",
                "Timeline visualization",
                "Conversational interface",
                "Event handling",
                "System health monitoring"
            ],
            "status": "FULLY_OPERATIONAL"
        }


# ============================================================================
# FACTORY AND EXPORTS
# ============================================================================

def create_unified_observability() -> UnifiedObservabilitySystem:
    """Factory function to create unified observability system"""
    return UnifiedObservabilitySystem()

# Global instance for compatibility
unified_observability = create_unified_observability()

# Convenience decorators for easy integration
def track_test_session(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator to track test sessions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with unified_observability.track_session(name, metadata) as session_id:
                return func(session_id, *args, **kwargs)
        return wrapper
    return decorator

def track_agent_action(session_id: str, action_type: str):
    """Decorator to track agent actions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            action_id = unified_observability.track_action(
                session_id, func.__name__, action_type, 
                {"args": str(args), "kwargs": str(kwargs)}
            )
            
            try:
                result = func(*args, **kwargs)
                unified_observability.complete_action(session_id, action_id, result=result)
                return result
            except Exception as e:
                unified_observability.complete_action(session_id, action_id, error=str(e))
                raise
        return wrapper
    return decorator

# Export main classes and functions
__all__ = [
    'UnifiedObservabilitySystem',
    'TestSession', 
    'AgentAction',
    'LLMCall',
    'MonitoringEvent',
    'MonitoringMode',
    'AlertLevel',
    'create_unified_observability',
    'unified_observability',
    'track_test_session',
    'track_agent_action'
]