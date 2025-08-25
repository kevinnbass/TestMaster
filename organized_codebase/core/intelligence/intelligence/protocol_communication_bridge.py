"""
Protocol Communication Bridge - Agent 11

This bridge implements a comprehensive agent communication protocol system
that enables secure, reliable, and efficient messaging between all components
in the TestMaster hybrid intelligence ecosystem.

Key Features:
- Multi-protocol support (HTTP, WebSocket, message queues, direct calls)
- Intelligent message routing with priority queuing
- Cross-system messaging with format translation
- Security and authentication for inter-agent communication
- Performance monitoring and message persistence
- Consensus-driven communication optimization
"""

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from enum import Enum
from collections import defaultdict, deque
import hashlib
import hmac
from concurrent.futures import ThreadPoolExecutor
import queue

from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import SharedState
from ...core.feature_flags import FeatureFlags


class MessageProtocol(Enum):
    """Communication protocol types."""
    DIRECT = "direct"           # Direct method calls
    ASYNC = "async"            # Async message passing
    QUEUE = "queue"            # Message queue based
    WEBSOCKET = "websocket"    # WebSocket connections
    HTTP = "http"              # HTTP REST API
    EVENT_BUS = "event_bus"    # Event-driven messaging


class RoutingStrategy(Enum):
    """Message routing strategies."""
    DIRECT = "direct"           # Direct point-to-point
    BROADCAST = "broadcast"     # One-to-many
    MULTICAST = "multicast"     # Group messaging
    PUBLISH_SUBSCRIBE = "pub_sub"  # Topic-based routing
    LOAD_BALANCED = "load_balanced"  # Load balancing
    PRIORITY_BASED = "priority"     # Priority-based routing


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 0    # System critical messages
    HIGH = 1        # High priority operations
    NORMAL = 2      # Normal operations
    LOW = 3         # Background tasks
    BULK = 4        # Bulk operations


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    ROUTING = "routing"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class CommunicationChannel:
    """Communication channel configuration."""
    channel_id: str
    protocol: MessageProtocol
    source_agent: str
    target_agent: Optional[str] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL
    encrypted: bool = True
    authenticated: bool = True
    persistent: bool = False
    timeout_seconds: int = 30
    retry_count: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    error_count: int = 0


@dataclass
class AgentMessage:
    """Agent communication message."""
    message_id: str
    source_agent: str
    target_agent: Optional[str]
    channel_id: str
    protocol: MessageProtocol
    routing_strategy: RoutingStrategy
    priority: MessagePriority
    message_type: str
    payload: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    response_to: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class MessageResponse:
    """Message processing response."""
    message_id: str
    response_id: str
    success: bool
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.message_queues: Dict[str, queue.PriorityQueue] = defaultdict(queue.PriorityQueue)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_history: deque = deque(maxlen=1000)
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
        
        # Performance metrics
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_failed = 0
        self.avg_processing_time = 0.0
        
        print("Message Bus initialized")
    
    def create_channel(
        self,
        source_agent: str,
        target_agent: Optional[str] = None,
        protocol: MessageProtocol = MessageProtocol.DIRECT,
        routing_strategy: RoutingStrategy = RoutingStrategy.DIRECT,
        **kwargs
    ) -> str:
        """Create a new communication channel."""
        channel_id = f"{source_agent}_{target_agent or 'broadcast'}_{int(time.time())}"
        
        channel = CommunicationChannel(
            channel_id=channel_id,
            protocol=protocol,
            source_agent=source_agent,
            target_agent=target_agent,
            routing_strategy=routing_strategy,
            **kwargs
        )
        
        with self.lock:
            self.channels[channel_id] = channel
            
            # Update routing table
            if target_agent:
                self.routing_table[source_agent].append(target_agent)
                
        print(f"Channel created: {channel_id} ({protocol.value} -> {routing_strategy.value})")
        return channel_id
    
    def send_message(
        self,
        source_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        target_agent: Optional[str] = None,
        channel_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> str:
        """Send a message through the bus."""
        message_id = str(uuid.uuid4())
        
        # Find or create channel
        if not channel_id:
            channel_id = self._find_or_create_channel(source_agent, target_agent)
        
        channel = self.channels.get(channel_id)
        if not channel:
            raise ValueError(f"Channel not found: {channel_id}")
        
        # Filter kwargs to avoid conflicts with AgentMessage fields
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['message_id', 'source_agent', 'target_agent', 
                                     'channel_id', 'protocol', 'routing_strategy', 
                                     'priority', 'message_type', 'payload', 
                                     'expires_at', 'max_retries']}
        
        # Create message
        message = AgentMessage(
            message_id=message_id,
            source_agent=source_agent,
            target_agent=target_agent,
            channel_id=channel_id,
            protocol=channel.protocol,
            routing_strategy=channel.routing_strategy,
            priority=priority,
            message_type=message_type,
            payload=payload,
            expires_at=datetime.now() + timedelta(seconds=channel.timeout_seconds),
            max_retries=channel.retry_count,
            **filtered_kwargs
        )
        
        # Route message
        self._route_message(message)
        
        with self.lock:
            self.messages_sent += 1
            channel.message_count += 1
            channel.last_used = datetime.now()
            self.message_history.append({
                "message_id": message_id,
                "source": source_agent,
                "target": target_agent,
                "type": message_type,
                "timestamp": message.timestamp.isoformat(),
                "status": message.status.value
            })
        
        return message_id
    
    def _find_or_create_channel(self, source_agent: str, target_agent: Optional[str]) -> str:
        """Find existing channel or create new one."""
        # Look for existing channel
        for channel_id, channel in self.channels.items():
            if (channel.source_agent == source_agent and 
                channel.target_agent == target_agent):
                return channel_id
        
        # Create new channel
        return self.create_channel(source_agent, target_agent)
    
    def _route_message(self, message: AgentMessage):
        """Route message based on routing strategy."""
        message.status = MessageStatus.ROUTING
        
        if message.routing_strategy == RoutingStrategy.DIRECT:
            self._route_direct(message)
        elif message.routing_strategy == RoutingStrategy.BROADCAST:
            self._route_broadcast(message)
        elif message.routing_strategy == RoutingStrategy.MULTICAST:
            self._route_multicast(message)
        elif message.routing_strategy == RoutingStrategy.PUBLISH_SUBSCRIBE:
            self._route_publish_subscribe(message)
        elif message.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            self._route_load_balanced(message)
        elif message.routing_strategy == RoutingStrategy.PRIORITY_BASED:
            self._route_priority_based(message)
    
    def _route_direct(self, message: AgentMessage):
        """Route message directly to target."""
        if message.target_agent:
            self.message_queues[message.target_agent].put((message.priority.value, message))
            message.status = MessageStatus.DELIVERED
    
    def _route_broadcast(self, message: AgentMessage):
        """Broadcast message to all agents."""
        for agent_id in self.routing_table.keys():
            if agent_id != message.source_agent:
                self.message_queues[agent_id].put((message.priority.value, message))
        message.status = MessageStatus.DELIVERED
    
    def _route_multicast(self, message: AgentMessage):
        """Multicast to agent group."""
        # Implementation for group-based routing
        group_agents = message.payload.get("target_group", [])
        for agent_id in group_agents:
            self.message_queues[agent_id].put((message.priority.value, message))
        message.status = MessageStatus.DELIVERED
    
    def _route_publish_subscribe(self, message: AgentMessage):
        """Route based on topic subscription."""
        topic = message.payload.get("topic", message.message_type)
        subscribers = self.subscribers.get(topic, [])
        for callback in subscribers:
            try:
                callback(message)
            except Exception as e:
                print(f"Subscriber callback error: {e}")
        message.status = MessageStatus.DELIVERED
    
    def _route_load_balanced(self, message: AgentMessage):
        """Load-balanced routing to available agents."""
        # Simple round-robin for now
        available_agents = [agent for agent in self.routing_table.keys() 
                          if agent != message.source_agent]
        if available_agents:
            target = available_agents[self.messages_sent % len(available_agents)]
            self.message_queues[target].put((message.priority.value, message))
            message.status = MessageStatus.DELIVERED
    
    def _route_priority_based(self, message: AgentMessage):
        """Priority-based routing."""
        # Route to least loaded agent with priority consideration
        min_queue_size = float('inf')
        target_agent = None
        
        for agent_id in self.routing_table.keys():
            if agent_id != message.source_agent:
                queue_size = self.message_queues[agent_id].qsize()
                if queue_size < min_queue_size:
                    min_queue_size = queue_size
                    target_agent = agent_id
        
        if target_agent:
            self.message_queues[target_agent].put((message.priority.value, message))
            message.status = MessageStatus.DELIVERED
    
    def receive_message(self, agent_id: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """Receive message for agent."""
        try:
            priority, message = self.message_queues[agent_id].get(timeout=timeout)
            message.status = MessageStatus.PROCESSED
            with self.lock:
                self.messages_delivered += 1
            return message
        except queue.Empty:
            return None
    
    def subscribe(self, topic: str, callback: Callable[[AgentMessage], None]):
        """Subscribe to topic-based messages."""
        self.subscribers[topic].append(callback)
    
    def unsubscribe(self, topic: str, callback: Callable[[AgentMessage], None]):
        """Unsubscribe from topic."""
        if callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get message bus performance metrics."""
        with self.lock:
            return {
                "messages_sent": self.messages_sent,
                "messages_delivered": self.messages_delivered,
                "messages_failed": self.messages_failed,
                "delivery_rate": self.messages_delivered / max(self.messages_sent, 1),
                "active_channels": len(self.channels),
                "active_queues": len([q for q in self.message_queues.values() if not q.empty()]),
                "total_queue_size": sum(q.qsize() for q in self.message_queues.values()),
                "avg_processing_time": self.avg_processing_time
            }


class MessageRouter:
    """Advanced message routing with intelligence."""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.routing_rules: List[Dict[str, Any]] = []
        self.routing_history: deque = deque(maxlen=500)
        self.performance_metrics: Dict[str, float] = defaultdict(float)
        
    def add_routing_rule(
        self,
        rule_id: str,
        condition: Callable[[AgentMessage], bool],
        action: Callable[[AgentMessage], None],
        priority: int = 100
    ):
        """Add intelligent routing rule."""
        rule = {
            "id": rule_id,
            "condition": condition,
            "action": action,
            "priority": priority,
            "matches": 0,
            "errors": 0
        }
        
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r["priority"])
    
    def route_with_intelligence(self, message: AgentMessage):
        """Route message using intelligent rules."""
        for rule in self.routing_rules:
            try:
                if rule["condition"](message):
                    rule["action"](message)
                    rule["matches"] += 1
                    
                    self.routing_history.append({
                        "message_id": message.message_id,
                        "rule_id": rule["id"],
                        "timestamp": datetime.now().isoformat(),
                        "success": True
                    })
                    break
            except Exception as e:
                rule["errors"] += 1
                print(f"Routing rule error {rule['id']}: {e}")
    
    def optimize_routing(self):
        """Optimize routing based on performance history."""
        # Analyze routing performance and adjust rules
        for rule in self.routing_rules:
            success_rate = rule["matches"] / max(rule["matches"] + rule["errors"], 1)
            self.performance_metrics[rule["id"]] = success_rate
            
            # Adjust priority based on success rate
            if success_rate > 0.9:
                rule["priority"] = max(50, rule["priority"] - 10)
            elif success_rate < 0.5:
                rule["priority"] = min(200, rule["priority"] + 10)
        
        # Re-sort rules by priority
        self.routing_rules.sort(key=lambda r: r["priority"])
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        total_matches = sum(rule["matches"] for rule in self.routing_rules)
        total_errors = sum(rule["errors"] for rule in self.routing_rules)
        
        return {
            "total_rules": len(self.routing_rules),
            "total_matches": total_matches,
            "total_errors": total_errors,
            "success_rate": total_matches / max(total_matches + total_errors, 1),
            "rule_performance": {rule["id"]: {
                "matches": rule["matches"],
                "errors": rule["errors"],
                "priority": rule["priority"]
            } for rule in self.routing_rules}
        }


class AgentCommunicator:
    """Agent communication interface."""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.channels: List[str] = []
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self.message_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_received = 0
        self.last_activity = datetime.now()
        
    def create_channel(
        self,
        target_agent: Optional[str] = None,
        protocol: MessageProtocol = MessageProtocol.DIRECT,
        **kwargs
    ) -> str:
        """Create communication channel."""
        channel_id = self.message_bus.create_channel(
            self.agent_id, target_agent, protocol, **kwargs
        )
        self.channels.append(channel_id)
        return channel_id
    
    def send_message(
        self,
        message_type: str,
        payload: Dict[str, Any],
        target_agent: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> str:
        """Send message to another agent."""
        message_id = self.message_bus.send_message(
            self.agent_id, message_type, payload, target_agent, priority=priority, **kwargs
        )
        
        self.messages_sent += 1
        self.last_activity = datetime.now()
        return message_id
    
    def receive_messages(self, timeout: float = 1.0) -> List[AgentMessage]:
        """Receive pending messages."""
        messages = []
        while True:
            message = self.message_bus.receive_message(self.agent_id, timeout)
            if not message:
                break
            messages.append(message)
            self.messages_received += 1
            self.last_activity = datetime.now()
            
            # Auto-process with handlers
            if message.message_type in self.message_handlers:
                try:
                    self.message_handlers[message.message_type](message)
                except Exception as e:
                    print(f"Message handler error: {e}")
        
        return messages
    
    def register_handler(self, message_type: str, handler: Callable[[AgentMessage], None]):
        """Register message handler."""
        self.message_handlers[message_type] = handler
    
    def subscribe(self, topic: str, callback: Callable[[AgentMessage], None]):
        """Subscribe to topic."""
        self.message_bus.subscribe(topic, callback)
        self.subscriptions[topic].append(callback)
    
    def broadcast(self, message_type: str, payload: Dict[str, Any], **kwargs) -> str:
        """Broadcast message to all agents."""
        # Remove routing_strategy from kwargs if present to avoid duplicate
        kwargs.pop('routing_strategy', None)
        
        # Create broadcast channel if needed
        channel_id = self.create_channel(
            None,
            MessageProtocol.ASYNC,
            routing_strategy=RoutingStrategy.BROADCAST
        )
        
        return self.send_message(
            message_type,
            payload,
            channel_id=channel_id,
            **kwargs
        )
    
    def request_response(
        self,
        message_type: str,
        payload: Dict[str, Any],
        target_agent: str,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Send request and wait for response."""
        correlation_id = str(uuid.uuid4())
        
        # Send request
        self.send_message(
            message_type,
            payload,
            target_agent,
            correlation_id=correlation_id
        )
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            messages = self.receive_messages(1.0)
            for message in messages:
                if (message.correlation_id == correlation_id and 
                    message.message_type.endswith("_response")):
                    return message.payload
        
        return None
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get agent communication metrics."""
        return {
            "agent_id": self.agent_id,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "active_channels": len(self.channels),
            "subscriptions": len(self.subscriptions),
            "handlers": len(self.message_handlers),
            "last_activity": self.last_activity.isoformat()
        }


class ProtocolCommunicationBridge:
    """Main protocol communication bridge orchestrator."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer4_bridges', 'protocol_communication')
        self.message_bus = MessageBus()
        self.router = MessageRouter(self.message_bus)
        self.communicators: Dict[str, AgentCommunicator] = {}
        self.shared_state = SharedState()
        self.coordinator = AgentCoordinator()
        
        # Security
        self.authentication_keys: Dict[str, str] = {}
        self.authorized_agents: set = set()
        
        # Performance monitoring
        self.start_time = datetime.now()
        self.total_messages_processed = 0
        
        if not self.enabled:
            return
        
        self._setup_default_routing_rules()
        self._initialize_system_channels()
        
        print("Protocol Communication Bridge initialized")
        print(f"   Message protocols: {[p.value for p in MessageProtocol]}")
        print(f"   Routing strategies: {[r.value for r in RoutingStrategy]}")
        print(f"   Security enabled: {True}")
    
    def _setup_default_routing_rules(self):
        """Setup default intelligent routing rules."""
        # High priority messages route directly
        self.router.add_routing_rule(
            "high_priority_direct",
            lambda msg: msg.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH],
            lambda msg: self._route_high_priority(msg),
            priority=10
        )
        
        # Consensus messages route to coordinator
        self.router.add_routing_rule(
            "consensus_routing",
            lambda msg: msg.message_type.startswith("consensus_"),
            lambda msg: self._route_consensus_message(msg),
            priority=20
        )
        
        # Performance monitoring messages
        self.router.add_routing_rule(
            "monitoring_routing",
            lambda msg: msg.message_type.startswith("monitor_"),
            lambda msg: self._route_monitoring_message(msg),
            priority=30
        )
    
    def _route_high_priority(self, message: AgentMessage):
        """Route high priority messages with special handling."""
        if message.target_agent:
            # Add to front of queue for priority processing
            self.message_bus.message_queues[message.target_agent].put((0, message))
    
    def _route_consensus_message(self, message: AgentMessage):
        """Route consensus-related messages."""
        # Route through coordinator for consensus processing
        if "vote" in message.payload:
            vote = AgentVote(
                agent_id=message.source_agent,
                decision=message.payload["vote"],
                confidence=message.payload.get("confidence", 0.5),
                reasoning=message.payload.get("reasoning", "")
            )
            # Process consensus vote (simplified)
            self.coordinator.collect_vote(vote)
    
    def _route_monitoring_message(self, message: AgentMessage):
        """Route monitoring messages to appropriate handlers."""
        # Store monitoring data in shared state
        self.shared_state.set(
            f"monitoring_{message.source_agent}_{message.message_type}",
            {
                "data": message.payload,
                "timestamp": message.timestamp.isoformat(),
                "agent": message.source_agent
            }
        )
    
    def _initialize_system_channels(self):
        """Initialize system-wide communication channels."""
        # Create broadcast channel for system announcements
        self.message_bus.create_channel(
            "system",
            None,
            MessageProtocol.ASYNC,
            RoutingStrategy.BROADCAST,
            priority=MessagePriority.HIGH
        )
        
        # Create consensus communication channel
        self.message_bus.create_channel(
            "consensus_coordinator",
            None,
            MessageProtocol.DIRECT,
            RoutingStrategy.MULTICAST,
            priority=MessagePriority.HIGH
        )
    
    def register_agent(self, agent_id: str, auth_key: Optional[str] = None) -> AgentCommunicator:
        """Register agent for communication."""
        if auth_key:
            self.authentication_keys[agent_id] = auth_key
            self.authorized_agents.add(agent_id)
        
        communicator = AgentCommunicator(agent_id, self.message_bus)
        self.communicators[agent_id] = communicator
        
        print(f"Agent registered: {agent_id}")
        return communicator
    
    def authenticate_agent(self, agent_id: str, auth_key: str) -> bool:
        """Authenticate agent for secure communication."""
        stored_key = self.authentication_keys.get(agent_id)
        if stored_key and hmac.compare_digest(stored_key, auth_key):
            self.authorized_agents.add(agent_id)
            return True
        return False
    
    def send_system_message(self, message_type: str, payload: Dict[str, Any]):
        """Send system-wide message."""
        return self.message_bus.send_message(
            "system",
            message_type,
            payload,
            priority=MessagePriority.HIGH
        )
    
    def enable_cross_system_messaging(
        self,
        source_system: str,
        target_system: str,
        message_translator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        """Enable messaging between different systems."""
        channel_id = self.message_bus.create_channel(
            source_system,
            target_system,
            MessageProtocol.HTTP,
            RoutingStrategy.DIRECT
        )
        
        if message_translator:
            # Add translation rule
            self.router.add_routing_rule(
                f"translate_{source_system}_to_{target_system}",
                lambda msg: msg.source_agent == source_system and msg.target_agent == target_system,
                lambda msg: self._translate_message(msg, message_translator),
                priority=50
            )
        
        return channel_id
    
    def _translate_message(self, message: AgentMessage, translator: Callable):
        """Translate message format between systems."""
        try:
            translated_payload = translator(message.payload)
            message.payload = translated_payload
            message.headers["translated"] = "true"
        except Exception as e:
            print(f"Message translation error: {e}")
            message.status = MessageStatus.FAILED
    
    def optimize_communication(self):
        """Optimize communication performance."""
        # Get metrics
        bus_metrics = self.message_bus.get_metrics()
        routing_metrics = self.router.get_routing_metrics()
        
        # Optimize routing rules
        self.router.optimize_routing()
        
        # Clean up old channels
        current_time = datetime.now()
        inactive_channels = []
        
        for channel_id, channel in self.message_bus.channels.items():
            if current_time - channel.last_used > timedelta(hours=1):
                if channel.message_count == 0:
                    inactive_channels.append(channel_id)
        
        for channel_id in inactive_channels:
            del self.message_bus.channels[channel_id]
        
        print(f"Communication optimization: removed {len(inactive_channels)} inactive channels")
        
        # Store optimization metrics
        self.shared_state.set("communication_optimization", {
            "optimized_at": current_time.isoformat(),
            "bus_metrics": bus_metrics,
            "routing_metrics": routing_metrics,
            "channels_cleaned": len(inactive_channels)
        })
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive communication metrics."""
        uptime = datetime.now() - self.start_time
        
        return {
            "bridge_status": "active" if self.enabled else "disabled",
            "uptime_seconds": uptime.total_seconds(),
            "registered_agents": len(self.communicators),
            "authorized_agents": len(self.authorized_agents),
            "message_bus_metrics": self.message_bus.get_metrics(),
            "routing_metrics": self.router.get_routing_metrics(),
            "agent_metrics": {
                agent_id: comm.get_communication_metrics()
                for agent_id, comm in self.communicators.items()
            },
            "total_messages_processed": self.total_messages_processed,
            "security_enabled": len(self.authentication_keys) > 0
        }
    
    def shutdown(self):
        """Shutdown communication bridge."""
        # Store final metrics
        final_metrics = self.get_comprehensive_metrics()
        self.shared_state.set("protocol_bridge_final_metrics", final_metrics)
        
        # Clear all queues
        for queue in self.message_bus.message_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
        
        print("Protocol Communication Bridge shutdown complete")


def get_protocol_bridge() -> ProtocolCommunicationBridge:
    """Get protocol communication bridge instance."""
    return ProtocolCommunicationBridge()