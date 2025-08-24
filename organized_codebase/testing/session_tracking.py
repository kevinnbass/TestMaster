"""
Session Tracking Module
=====================

Focused module for test session tracking and replay functionality.
Extracted from unified_monitor.py for better modularization.

Author: TestMaster Phase 1C Consolidation
"""

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional

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
    timestamp: float
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0
    llm_calls: List[str] = field(default_factory=list)
    cost: float = 0.0

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

class SessionTracker:
    """Manages test session lifecycle and tracking"""
    
    def __init__(self):
        self.active_sessions: Dict[str, TestSession] = {}
        self.session_replay = SessionReplay()
        
    def create_session(self, name: str, metadata: Optional[Dict[str, Any]] = None,
                      tags: Optional[List[str]] = None) -> str:
        """Create a new test session"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        session = TestSession(
            session_id=session_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
            tags=tags or []
        )
        
        self.active_sessions[session_id] = session
        self.session_replay.sessions[session_id] = session
        
        return session_id
    
    def end_session(self, session_id: str, status: str = "completed") -> Optional[TestSession]:
        """End a test session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        session.end_time = time.time()
        session.status = status
        
        return session
    
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
        
        self.session_replay.record_action(session_id, action)
        return action_id

# Export key components
__all__ = [
    'TestSession',
    'AgentAction', 
    'SessionReplay',
    'SessionTracker'
]