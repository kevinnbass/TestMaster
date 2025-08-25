"""
TestMaster AgentOps Observability Integration
=============================================

Comprehensive observability for test execution with session replay,
cost tracking, and performance analytics using AgentOps patterns.

Author: TestMaster Team
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
import uuid

@dataclass
class TestSession:
    """Represents a tracked test execution session"""
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
    """Represents an action performed by an agent"""
    action_id: str
    agent_name: str
    action_type: str
    parameters: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    session_id: Optional[str] = None
    parent_action_id: Optional[str] = None

@dataclass
class LLMCall:
    """Represents a call to an LLM service"""
    call_id: str
    session_id: str
    action_id: Optional[str]
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    cost: float
    timestamp: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CostTracker:
    """Tracks costs for different LLM providers and models"""
    
    def __init__(self):
        # Cost per token in USD (approximate)
        self.pricing = {
            "openai": {
                "gpt-4": {"input": 0.03/1000, "output": 0.06/1000},
                "gpt-4-turbo": {"input": 0.01/1000, "output": 0.03/1000},
                "gpt-3.5-turbo": {"input": 0.0015/1000, "output": 0.002/1000},
            },
            "anthropic": {
                "claude-3-opus": {"input": 0.015/1000, "output": 0.075/1000},
                "claude-3-sonnet": {"input": 0.003/1000, "output": 0.015/1000},
                "claude-3-haiku": {"input": 0.00025/1000, "output": 0.00125/1000},
            },
            "google": {
                "gemini-pro": {"input": 0.0005/1000, "output": 0.0015/1000},
                "gemini-pro-vision": {"input": 0.0005/1000, "output": 0.0015/1000},
            }
        }
        self.session_costs = defaultdict(float)
        
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for an LLM call"""
        # Extract provider from model name
        provider = "openai"  # default
        if "claude" in model.lower():
            provider = "anthropic"
        elif "gemini" in model.lower():
            provider = "google"
        
        # Get pricing for provider and model
        provider_pricing = self.pricing.get(provider, {})
        model_pricing = provider_pricing.get(model, provider_pricing.get(list(provider_pricing.keys())[0] if provider_pricing else "gpt-3.5-turbo", {"input": 0.002/1000, "output": 0.002/1000}))
        
        input_cost = input_tokens * model_pricing["input"]
        output_cost = output_tokens * model_pricing["output"]
        
        return input_cost + output_cost
    
    def track_llm_call(self, session_id: str, llm_call: LLMCall):
        """Track cost for an LLM call"""
        cost = self.calculate_cost(llm_call.model, llm_call.prompt_tokens, llm_call.completion_tokens)
        llm_call.cost = cost
        self.session_costs[session_id] += cost
    
    def get_session_costs(self, session_id: str) -> Dict[str, Any]:
        """Get cost breakdown for a session"""
        return {
            "session_id": session_id,
            "total_cost": self.session_costs[session_id],
            "cost_usd": f"${self.session_costs[session_id]:.4f}"
        }

class TestMasterObservability:
    """
    Central observability system for TestMaster using AgentOps patterns.
    
    Features:
    - Session tracking with hierarchical action trees
    - LLM call monitoring and cost tracking
    - Performance metrics and execution replay
    - Error tracking and debugging
    - Cost optimization insights
    """
    
    def __init__(self):
        self.sessions: Dict[str, TestSession] = {}
        self.actions: Dict[str, AgentAction] = {}
        self.llm_calls: Dict[str, LLMCall] = {}
        self.cost_tracker = CostTracker()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.action_timings = defaultdict(list)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logging.info("TestMaster Observability System initialized")
    
    def start_test_session(self, session_name: str, metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """Start a new test session"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        session = TestSession(
            session_id=session_id,
            name=session_name,
            start_time=time.time(),
            metadata=metadata or {},
            tags=tags or []
        )
        
        with self._lock:
            self.sessions[session_id] = session
        
        self._emit_event("session_started", {
            "session_id": session_id,
            "name": session_name,
            "metadata": metadata,
            "tags": tags
        })
        
        logging.info(f"Started test session: {session_name} ({session_id})")
        return session_id
    
    def end_test_session(self, session_id: str, status: str = "completed") -> Dict[str, Any]:
        """End a test session and generate summary"""
        with self._lock:
            if session_id not in self.sessions:
                return {"error": "Session not found"}
            
            session = self.sessions[session_id]
            session.end_time = time.time()
            session.status = status
        
        # Generate session summary
        summary = self.generate_session_replay(session_id)
        
        self._emit_event("session_ended", {
            "session_id": session_id,
            "status": status,
            "duration": session.duration,
            "summary": summary
        })
        
        logging.info(f"Ended test session {session_id} with status: {status}")
        return summary
    
    def track_agent_action(self, session_id: str, agent_name: str, action_type: str, 
                          parameters: Dict[str, Any], parent_action_id: str = None) -> str:
        """Start tracking an agent action"""
        action_id = f"action_{uuid.uuid4().hex[:12]}"
        
        action = AgentAction(
            action_id=action_id,
            agent_name=agent_name,
            action_type=action_type,
            parameters=parameters,
            start_time=time.time(),
            session_id=session_id,
            parent_action_id=parent_action_id
        )
        
        with self._lock:
            self.actions[action_id] = action
            if session_id in self.sessions:
                self.sessions[session_id].actions.append({
                    "action_id": action_id,
                    "agent_name": agent_name,
                    "action_type": action_type,
                    "start_time": action.start_time,
                    "parent_action_id": parent_action_id
                })
        
        self._emit_event("action_started", {
            "action_id": action_id,
            "session_id": session_id,
            "agent_name": agent_name,
            "action_type": action_type
        })
        
        return action_id
    
    def complete_agent_action(self, action_id: str, result: Dict[str, Any] = None, 
                            error: str = None, llm_calls: List[Dict[str, Any]] = None):
        """Complete an agent action"""
        with self._lock:
            if action_id not in self.actions:
                return
            
            action = self.actions[action_id]
            action.end_time = time.time()
            action.result = result
            action.error = error
            action.llm_calls = llm_calls or []
        
        # Track performance
        execution_time = action.end_time - action.start_time
        self.action_timings[action.action_type].append(execution_time)
        
        self._emit_event("action_completed", {
            "action_id": action_id,
            "success": error is None,
            "execution_time": execution_time,
            "result": result,
            "error": error
        })
    
    def track_llm_call(self, session_id: str, model: str, provider: str, 
                      prompt_tokens: int, completion_tokens: int, latency: float,
                      success: bool, error: str = None, metadata: Dict[str, Any] = None) -> str:
        """Track an LLM call"""
        call_id = f"llm_{uuid.uuid4().hex[:12]}"
        
        llm_call = LLMCall(
            call_id=call_id,
            session_id=session_id,
            action_id=None,  # Can be set later
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency=latency,
            cost=0.0,  # Will be calculated
            timestamp=time.time(),
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        # Calculate cost
        self.cost_tracker.track_llm_call(session_id, llm_call)
        
        with self._lock:
            self.llm_calls[call_id] = llm_call
            if session_id in self.sessions:
                self.sessions[session_id].llm_calls.append({
                    "call_id": call_id,
                    "model": model,
                    "tokens": llm_call.total_tokens,
                    "cost": llm_call.cost,
                    "latency": latency,
                    "success": success,
                    "timestamp": llm_call.timestamp
                })
        
        self._emit_event("llm_call", {
            "call_id": call_id,
            "session_id": session_id,
            "model": model,
            "tokens": llm_call.total_tokens,
            "cost": llm_call.cost,
            "success": success
        })
        
        return call_id
    
    def generate_session_replay(self, session_id: str) -> Dict[str, Any]:
        """Generate a comprehensive session replay"""
        with self._lock:
            if session_id not in self.sessions:
                return {"error": "Session not found"}
            
            session = self.sessions[session_id]
            session_actions = [a for a in self.actions.values() if a.session_id == session_id]
            session_llm_calls = [c for c in self.llm_calls.values() if c.session_id == session_id]
        
        # Generate timeline
        timeline = self._generate_session_timeline(session, session_actions, session_llm_calls)
        
        # Calculate performance metrics
        performance = self._calculate_session_performance(session)
        
        # Generate analytics
        analytics = self._generate_session_analytics(session)
        
        return {
            "session_id": session_id,
            "name": session.name,
            "status": session.status,
            "duration": session.duration,
            "timeline": timeline,
            "performance": performance,
            "analytics": analytics,
            "costs": self.cost_tracker.get_session_costs(session_id),
            "metadata": session.metadata,
            "tags": session.tags
        }
    
    def _generate_session_timeline(self, session: TestSession, actions: List[AgentAction], llm_calls: List[LLMCall]) -> List[Dict[str, Any]]:
        """Generate chronological timeline of session events"""
        events = []
        
        # Add session start
        events.append({
            "timestamp": session.start_time,
            "type": "session_start",
            "name": session.name,
            "data": {"metadata": session.metadata}
        })
        
        # Add actions
        for action in actions:
            events.append({
                "timestamp": action.start_time,
                "type": "action_start",
                "agent": action.agent_name,
                "action": action.action_type,
                "data": {"parameters": action.parameters}
            })
            
            if action.end_time:
                events.append({
                    "timestamp": action.end_time,
                    "type": "action_end",
                    "agent": action.agent_name,
                    "action": action.action_type,
                    "data": {
                        "duration": action.end_time - action.start_time,
                        "success": action.error is None,
                        "result": action.result,
                        "error": action.error
                    }
                })
        
        # Add LLM calls
        for call in llm_calls:
            events.append({
                "timestamp": call.timestamp,
                "type": "llm_call",
                "model": call.model,
                "data": {
                    "tokens": call.total_tokens,
                    "cost": call.cost,
                    "latency": call.latency,
                    "success": call.success
                }
            })
        
        # Add session end
        if session.end_time:
            events.append({
                "timestamp": session.end_time,
                "type": "session_end",
                "status": session.status,
                "data": {"duration": session.duration}
            })
        
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])
        return events
    
    def _calculate_session_performance(self, session: TestSession) -> Dict[str, Any]:
        """Calculate performance metrics for session"""
        session_actions = [a for a in self.actions.values() if a.session_id == session.session_id]
        session_llm_calls = [c for c in self.llm_calls.values() if c.session_id == session.session_id]
        
        # Action performance
        completed_actions = [a for a in session_actions if a.end_time]
        failed_actions = [a for a in completed_actions if a.error]
        
        action_durations = [a.end_time - a.start_time for a in completed_actions]
        avg_action_duration = sum(action_durations) / len(action_durations) if action_durations else 0
        
        # LLM performance
        total_tokens = sum(c.total_tokens for c in session_llm_calls)
        total_llm_cost = sum(c.cost for c in session_llm_calls)
        avg_llm_latency = sum(c.latency for c in session_llm_calls) / len(session_llm_calls) if session_llm_calls else 0
        
        return {
            "actions": {
                "total": len(session_actions),
                "completed": len(completed_actions),
                "failed": len(failed_actions),
                "success_rate": (len(completed_actions) - len(failed_actions)) / len(completed_actions) if completed_actions else 0,
                "avg_duration": avg_action_duration
            },
            "llm": {
                "total_calls": len(session_llm_calls),
                "total_tokens": total_tokens,
                "total_cost": total_llm_cost,
                "avg_latency": avg_llm_latency
            },
            "efficiency_score": self._calculate_efficiency_score(session)
        }
    
    def _generate_session_analytics(self, session: TestSession) -> Dict[str, Any]:
        """Generate analytics and insights for session"""
        return {
            "session_type": self._classify_session_type(session),
            "bottlenecks": self._identify_bottlenecks(session),
            "optimization_suggestions": self._generate_optimization_suggestions(session)
        }
    
    def _calculate_efficiency_score(self, session: TestSession) -> float:
        """Calculate overall efficiency score (0-100)"""
        # Simplified efficiency calculation
        session_actions = [a for a in self.actions.values() if a.session_id == session.session_id]
        completed_actions = [a for a in session_actions if a.end_time and not a.error]
        
        if not session_actions:
            return 0.0
        
        success_rate = len(completed_actions) / len(session_actions)
        return success_rate * 100
    
    def _classify_session_type(self, session: TestSession) -> str:
        """Classify the type of test session"""
        # Simple classification based on actions
        action_types = [a.action_type for a in self.actions.values() if a.session_id == session.session_id]
        
        if "test_generation" in action_types:
            return "test_generation"
        elif "test_execution" in action_types:
            return "test_execution"
        elif "analysis" in action_types:
            return "analysis"
        else:
            return "general"
    
    def _identify_bottlenecks(self, session: TestSession) -> List[str]:
        """Identify performance bottlenecks in session"""
        bottlenecks = []
        
        session_actions = [a for a in self.actions.values() if a.session_id == session.session_id and a.end_time]
        if session_actions:
            # Find slow actions (>95th percentile)
            durations = [a.end_time - a.start_time for a in session_actions]
            if durations:
                p95 = sorted(durations)[int(len(durations) * 0.95)]
                slow_actions = [a for a in session_actions if (a.end_time - a.start_time) > p95]
                
                for action in slow_actions:
                    bottlenecks.append(f"Slow {action.action_type} action (agent: {action.agent_name})")
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, session: TestSession) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Analyze LLM usage
        session_llm_calls = [c for c in self.llm_calls.values() if c.session_id == session.session_id]
        total_cost = sum(c.cost for c in session_llm_calls)
        
        if total_cost > 1.0:  # $1 threshold
            suggestions.append("Consider using smaller models for simple tasks to reduce costs")
        
        # Analyze action patterns
        session_actions = [a for a in self.actions.values() if a.session_id == session.session_id]
        failed_actions = [a for a in session_actions if a.error]
        
        if len(failed_actions) > len(session_actions) * 0.2:  # >20% failure rate
            suggestions.append("High failure rate detected - review error handling and retry logic")
        
        return suggestions
    
    def get_observability_status(self) -> Dict[str, Any]:
        """Get current observability system status"""
        with self._lock:
            active_sessions = [s for s in self.sessions.values() if s.status == "active"]
            
            return {
                "total_sessions": len(self.sessions),
                "active_sessions": len(active_sessions),
                "total_actions": len(self.actions),
                "total_llm_calls": len(self.llm_calls),
                "total_cost": sum(self.cost_tracker.session_costs.values()),
                "event_handlers": {event: len(handlers) for event, handlers in self.event_handlers.items()}
            }
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        for handler in self.event_handlers[event_type]:
            try:
                handler(event_type, data)
            except Exception as e:
                logging.error(f"Error in event handler for {event_type}: {e}")

# Decorator for automatic action tracking
def track_test_execution(session_name: str, observability_system: TestMasterObservability):
    """Decorator to automatically track test execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_id = observability_system.start_test_session(session_name)
            action_id = observability_system.track_agent_action(
                session_id, 
                func.__module__, 
                func.__name__, 
                {"args": str(args), "kwargs": str(kwargs)}
            )
            
            try:
                result = func(*args, **kwargs)
                observability_system.complete_agent_action(action_id, {"result": str(result)})
                observability_system.end_test_session(session_id, "completed")
                return result
            except Exception as e:
                observability_system.complete_agent_action(action_id, error=str(e))
                observability_system.end_test_session(session_id, "failed")
                raise
        
        return wrapper
    return decorator

# Global instance
global_observability = TestMasterObservability()