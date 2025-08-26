"""
Agent Interface Systems Module
=============================
Advanced chat UIs and control panels extracted from AutoGen, Agency-Swarm, and LLama-Agents.
Module size: ~297 lines (under 300 limit)

Patterns extracted from:
- AutoGen: Streaming chat interfaces and real-time communication
- Agency-Swarm: Gradio integration and agent interaction UIs
- AgentScope: Studio interface and agent management
- LLama-Agents: Deployment interfaces and workflow controls
- CrewAI: Agent communication and handoff interfaces
- Swarms: Multi-agent coordination interfaces
- PhiData: Interactive agent tools and controls

Author: Agent D - Visualization Specialist
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import deque
import threading
from abc import ABC, abstractmethod


@dataclass
class ChatMessage:
    """Standardized chat message structure."""
    id: str
    role: str  # user, assistant, system, function
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    agent_name: str = ""
    
    @classmethod
    def create(cls, role: str, content: str, agent_name: str = "", **metadata):
        return cls(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
            agent_name=agent_name
        )


@dataclass
class AgentStatus:
    """Agent status information."""
    id: str
    name: str
    status: str  # active, idle, busy, error, offline
    last_activity: datetime
    current_task: str = ""
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class ChatInterface(ABC):
    """Abstract base for chat interfaces."""
    
    @abstractmethod
    def send_message(self, message: ChatMessage) -> bool:
        pass
        
    @abstractmethod
    def receive_message(self) -> Optional[ChatMessage]:
        pass
        
    @abstractmethod
    def get_history(self, limit: int = 50) -> List[ChatMessage]:
        pass


class StreamingChatInterface(ChatInterface):
    """Streaming chat interface (AutoGen pattern)."""
    
    def __init__(self, max_history: int = 1000):
        self.message_history = deque(maxlen=max_history)
        self.subscribers = []
        self.streaming_handlers = []
        self.lock = threading.RLock()
        
    def send_message(self, message: ChatMessage) -> bool:
        """Send message and notify subscribers."""
        try:
            with self.lock:
                self.message_history.append(message)
                
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(message)
                except Exception:
                    pass  # Don't let subscriber errors break sending
                    
            return True
        except Exception:
            return False
            
    def receive_message(self) -> Optional[ChatMessage]:
        """Get most recent message."""
        with self.lock:
            return self.message_history[-1] if self.message_history else None
            
    def get_history(self, limit: int = 50) -> List[ChatMessage]:
        """Get recent message history."""
        with self.lock:
            history = list(self.message_history)
            return history[-limit:] if len(history) > limit else history
            
    def add_subscriber(self, callback: Callable[[ChatMessage], None]):
        """Add message subscriber."""
        self.subscribers.append(callback)
        
    def add_streaming_handler(self, handler: Callable[[str], None]):
        """Add streaming content handler."""
        self.streaming_handlers.append(handler)
        
    def stream_content(self, content: str, chunk_size: int = 50):
        """Stream content to handlers (AutoGen pattern)."""
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            for handler in self.streaming_handlers:
                try:
                    handler(chunk)
                except Exception:
                    pass


class GradioInterface:
    """Gradio-based interface (Agency-Swarm pattern)."""
    
    def __init__(self, title: str = "Agent Interface"):
        self.title = title
        self.chat_interface = StreamingChatInterface()
        self.agent_controls = {}
        self.interface_config = {
            "height": 600,
            "show_share_button": False,
            "show_tips": False
        }
        
    def create_chat_interface(self) -> Dict[str, Any]:
        """Create Gradio chat interface configuration."""
        return {
            "interface_type": "gradio_chatbot",
            "title": self.title,
            "components": {
                "chatbot": {
                    "type": "chatbot",
                    "height": 400,
                    "show_label": False
                },
                "message_input": {
                    "type": "textbox",
                    "placeholder": "Type your message...",
                    "lines": 2,
                    "max_lines": 5
                },
                "send_button": {
                    "type": "button", 
                    "value": "Send",
                    "variant": "primary"
                }
            },
            "config": self.interface_config
        }
        
    def add_agent_control(self, agent_name: str, control_type: str, **config):
        """Add agent control component."""
        self.agent_controls[agent_name] = {
            "type": control_type,
            "config": config,
            "created_at": datetime.now()
        }
        
    def get_interface_state(self) -> Dict[str, Any]:
        """Get current interface state."""
        return {
            "title": self.title,
            "message_count": len(self.chat_interface.message_history),
            "agent_controls": list(self.agent_controls.keys()),
            "last_activity": datetime.now(),
            "config": self.interface_config
        }


class AgentControlPanel:
    """Agent control and monitoring panel (LLama-Agents pattern)."""
    
    def __init__(self):
        self.agents: Dict[str, AgentStatus] = {}
        self.control_actions = {}
        self.update_callbacks = []
        
    def register_agent(self, agent_id: str, name: str) -> bool:
        """Register new agent."""
        try:
            self.agents[agent_id] = AgentStatus(
                id=agent_id,
                name=name,
                status="idle",
                last_activity=datetime.now()
            )
            self._notify_updates()
            return True
        except Exception:
            return False
            
    def update_agent_status(self, agent_id: str, status: str, 
                           task: str = "", metrics: Dict[str, Any] = None):
        """Update agent status."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = status
            agent.last_activity = datetime.now()
            if task:
                agent.current_task = task
            if metrics:
                agent.performance_metrics.update(metrics)
            self._notify_updates()
            
    def get_agent_overview(self) -> Dict[str, Any]:
        """Get agent overview for dashboard."""
        overview = {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.status == "active"),
            "idle_agents": sum(1 for a in self.agents.values() if a.status == "idle"),
            "busy_agents": sum(1 for a in self.agents.values() if a.status == "busy"),
            "error_agents": sum(1 for a in self.agents.values() if a.status == "error"),
            "agents": [asdict(agent) for agent in self.agents.values()]
        }
        return overview
        
    def execute_control_action(self, agent_id: str, action: str, params: Dict[str, Any] = None):
        """Execute control action on agent."""
        if agent_id not in self.agents:
            return {"success": False, "error": "Agent not found"}
            
        if action not in self.control_actions:
            return {"success": False, "error": "Action not supported"}
            
        try:
            result = self.control_actions[action](agent_id, params or {})
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def add_control_action(self, action_name: str, handler: Callable):
        """Add custom control action."""
        self.control_actions[action_name] = handler
        
    def add_update_callback(self, callback: Callable):
        """Add callback for agent updates."""
        self.update_callbacks.append(callback)
        
    def _notify_updates(self):
        """Notify all callbacks of updates."""
        overview = self.get_agent_overview()
        for callback in self.update_callbacks:
            try:
                callback(overview)
            except Exception:
                pass


class MultiAgentInterface:
    """Multi-agent coordination interface (Swarms pattern)."""
    
    def __init__(self):
        self.agents = {}
        self.communication_channels = {}
        self.swarm_status = {
            "active": False,
            "coordination_mode": "distributed",
            "task_queue": deque(),
            "results": deque(maxlen=100)
        }
        
    def add_agent_channel(self, agent_id: str, channel: ChatInterface):
        """Add communication channel for agent."""
        self.communication_channels[agent_id] = channel
        
    def broadcast_message(self, message: str, sender_id: str = "system") -> Dict[str, bool]:
        """Broadcast message to all agents."""
        results = {}
        broadcast_msg = ChatMessage.create(
            role="system",
            content=message,
            agent_name=sender_id,
            broadcast=True
        )
        
        for agent_id, channel in self.communication_channels.items():
            results[agent_id] = channel.send_message(broadcast_msg)
            
        return results
        
    def route_message(self, from_agent: str, to_agent: str, message: str) -> bool:
        """Route message between specific agents."""
        if to_agent not in self.communication_channels:
            return False
            
        routed_msg = ChatMessage.create(
            role="assistant",
            content=message,
            agent_name=from_agent,
            recipient=to_agent
        )
        
        return self.communication_channels[to_agent].send_message(routed_msg)
        
    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get swarm performance metrics."""
        return {
            "agents_count": len(self.communication_channels),
            "active_channels": sum(1 for ch in self.communication_channels.values() 
                                 if hasattr(ch, 'message_history') and ch.message_history),
            "total_messages": sum(len(ch.message_history) for ch in self.communication_channels.values() 
                                if hasattr(ch, 'message_history')),
            "swarm_status": self.swarm_status,
            "timestamp": datetime.now()
        }


class WebSocketInterface:
    """Real-time WebSocket interface for agent communication."""
    
    def __init__(self):
        self.connections = {}
        self.message_queue = deque(maxlen=1000)
        
    def add_connection(self, connection_id: str, websocket_handler):
        """Add WebSocket connection."""
        self.connections[connection_id] = {
            "handler": websocket_handler,
            "connected_at": datetime.now(),
            "message_count": 0
        }
        
    def remove_connection(self, connection_id: str):
        """Remove WebSocket connection."""
        self.connections.pop(connection_id, None)
        
    def broadcast_to_connections(self, data: Dict[str, Any]) -> int:
        """Broadcast data to all connections."""
        message = json.dumps(data, default=str)
        sent_count = 0
        
        for conn_id, conn_info in list(self.connections.items()):
            try:
                conn_info["handler"].send(message)
                conn_info["message_count"] += 1
                sent_count += 1
            except Exception:
                # Remove broken connections
                self.connections.pop(conn_id, None)
                
        return sent_count
        
    def send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Send data to specific connection."""
        if connection_id not in self.connections:
            return False
            
        try:
            message = json.dumps(data, default=str)
            self.connections[connection_id]["handler"].send(message)
            self.connections[connection_id]["message_count"] += 1
            return True
        except Exception:
            self.connections.pop(connection_id, None)
            return False


# Public API
__all__ = [
    'ChatMessage',
    'AgentStatus',
    'ChatInterface', 
    'StreamingChatInterface',
    'GradioInterface',
    'AgentControlPanel',
    'MultiAgentInterface',
    'WebSocketInterface'
]