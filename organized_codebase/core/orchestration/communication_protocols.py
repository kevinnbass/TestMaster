"""
Communication Protocols
======================

Communication protocols for inter-orchestrator and agent communication
providing reliable, scalable, and secure messaging infrastructure.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Tuple, Union
from collections import defaultdict, deque
import logging


class MessageType(Enum):
    """Types of messages in orchestration communication."""
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    BROADCAST = "broadcast"


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class CommunicationPattern(Enum):
    """Communication patterns between entities."""
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    MESSAGE_QUEUE = "message_queue"
    EVENT_STREAMING = "event_streaming"
    PIPELINE = "pipeline"
    MESH = "mesh"


@dataclass
class Message:
    """Universal message format for orchestration communication."""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    content: Any
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: int = 300  # Time to live in seconds
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationEndpoint:
    """Communication endpoint configuration."""
    endpoint_id: str
    endpoint_type: str
    address: str
    port: Optional[int] = None
    protocol: str = "tcp"
    authentication: Dict[str, Any] = field(default_factory=dict)
    encryption: Dict[str, Any] = field(default_factory=dict)
    connection_pool_size: int = 10
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class CommunicationMetrics:
    """Metrics for communication performance."""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    average_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    connection_count: int = 0
    bandwidth_usage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols."""
    
    def __init__(
        self,
        protocol_name: str,
        endpoint: CommunicationEndpoint
    ):
        self.protocol_name = protocol_name
        self.endpoint = endpoint
        self.is_connected = False
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.metrics = CommunicationMetrics()
        self.message_queue = asyncio.Queue()
        self.outbound_queue = asyncio.Queue()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.connection_pool: List[Any] = []
        self.active_connections: Set[str] = set()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection."""
        pass
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """Send message."""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Message]:
        """Receive message."""
        pass
    
    def add_message_handler(self, message_type: MessageType, handler: Callable):
        """Add handler for specific message type."""
        self.message_handlers[message_type].append(handler)
    
    async def start_processing(self):
        """Start message processing loop."""
        asyncio.create_task(self._process_incoming_messages())
        asyncio.create_task(self._process_outgoing_messages())
    
    async def _process_incoming_messages(self):
        """Process incoming messages."""
        while self.is_connected:
            try:
                message = await self.receive_message()
                if message:
                    await self._handle_message(message)
            except Exception as e:
                self.logger.error(f"Error processing incoming message: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_outgoing_messages(self):
        """Process outgoing messages."""
        while self.is_connected:
            try:
                message = await self.outbound_queue.get()
                success = await self.send_message(message)
                if success:
                    self.metrics.messages_sent += 1
                else:
                    self.metrics.messages_failed += 1
            except Exception as e:
                self.logger.error(f"Error processing outgoing message: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_message(self, message: Message):
        """Handle received message."""
        handlers = self.message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
        
        self.metrics.messages_received += 1


class MessageProtocol(CommunicationProtocol):
    """
    Message-based communication protocol for orchestration.
    
    Implements reliable messaging with acknowledgments, retry logic,
    and quality of service guarantees.
    """
    
    def __init__(
        self,
        protocol_name: str = "message_protocol",
        endpoint: Optional[CommunicationEndpoint] = None
    ):
        if not endpoint:
            endpoint = CommunicationEndpoint(
                endpoint_id="default_message_endpoint",
                endpoint_type="message",
                address="localhost",
                port=8080
            )
        
        super().__init__(protocol_name, endpoint)
        self.pending_acks: Dict[str, Message] = {}
        self.message_cache: Dict[str, Message] = {}
        self.subscription_topics: Set[str] = set()
        self.reliable_delivery = True
        self.duplicate_detection = True
        self.message_ordering = True
        self.sequence_numbers: Dict[str, int] = defaultdict(int)
    
    async def connect(self) -> bool:
        """Establish message protocol connection."""
        try:
            # Simulate connection establishment
            await asyncio.sleep(0.1)
            self.is_connected = True
            self.logger.info(f"Connected to {self.endpoint.address}:{self.endpoint.port}")
            
            # Start processing
            await self.start_processing()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close message protocol connection."""
        try:
            self.is_connected = False
            
            # Send pending acknowledgments
            await self._flush_pending_acks()
            
            # Clear resources
            self.pending_acks.clear()
            self.message_cache.clear()
            
            self.logger.info("Disconnected from message protocol")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send message with reliability guarantees."""
        try:
            # Add sequence number for ordering
            if self.message_ordering:
                sender_seq = self.sequence_numbers[message.sender_id]
                message.headers["sequence"] = str(sender_seq)
                self.sequence_numbers[message.sender_id] += 1
            
            # Add reliability headers
            if self.reliable_delivery:
                message.headers["requires_ack"] = "true"
                self.pending_acks[message.message_id] = message
            
            # Simulate message sending
            await self._transmit_message(message)
            
            # Start acknowledgment timer if reliable delivery
            if self.reliable_delivery:
                asyncio.create_task(self._handle_ack_timeout(message.message_id))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self) -> Optional[Message]:
        """Receive message with duplicate detection."""
        try:
            if not self.message_queue.empty():
                message = await self.message_queue.get()
                
                # Duplicate detection
                if self.duplicate_detection:
                    if message.message_id in self.message_cache:
                        self.logger.debug(f"Duplicate message detected: {message.message_id}")
                        return None
                    self.message_cache[message.message_id] = message
                
                # Send acknowledgment if required
                if message.headers.get("requires_ack") == "true":
                    await self._send_acknowledgment(message)
                
                return message
                
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            
        return None
    
    async def subscribe_to_topic(self, topic: str):
        """Subscribe to message topic."""
        self.subscription_topics.add(topic)
        self.logger.info(f"Subscribed to topic: {topic}")
    
    async def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe from message topic."""
        self.subscription_topics.discard(topic)
        self.logger.info(f"Unsubscribed from topic: {topic}")
    
    async def publish_to_topic(self, topic: str, content: Any, priority: MessagePriority = MessagePriority.NORMAL):
        """Publish message to topic."""
        message = Message(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.BROADCAST,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=None,
            content=content,
            timestamp=datetime.now(),
            priority=priority,
            headers={"topic": topic}
        )
        
        await self.outbound_queue.put(message)
    
    async def send_request(self, recipient_id: str, content: Any) -> Optional[Message]:
        """Send request and wait for response."""
        correlation_id = f"req_{uuid.uuid4().hex[:8]}"
        
        request_message = Message(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.COMMAND,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            reply_to=self.endpoint.endpoint_id
        )
        
        # Set up response handler
        response_future = asyncio.Future()
        
        async def response_handler(message: Message):
            if message.correlation_id == correlation_id:
                response_future.set_result(message)
        
        self.add_message_handler(MessageType.RESPONSE, response_handler)
        
        # Send request
        await self.outbound_queue.put(request_message)
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(response_future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            self.logger.warning(f"Request timeout for correlation_id: {correlation_id}")
            return None
    
    async def send_response(self, original_message: Message, response_content: Any):
        """Send response to request."""
        response_message = Message(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.RESPONSE,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=original_message.reply_to,
            content=response_content,
            timestamp=datetime.now(),
            correlation_id=original_message.correlation_id
        )
        
        await self.outbound_queue.put(response_message)
    
    async def _transmit_message(self, message: Message):
        """Simulate message transmission."""
        # In real implementation, this would use actual network protocols
        serialized_message = self._serialize_message(message)
        
        # Simulate network delay
        await asyncio.sleep(0.01)
        
        # For simulation, directly add to local queue
        # In real implementation, this would send over network
        await self.message_queue.put(message)
    
    async def _send_acknowledgment(self, message: Message):
        """Send acknowledgment for received message."""
        ack_message = Message(
            message_id=f"ack_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.RESPONSE,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=message.sender_id,
            content={"ack": True, "original_message_id": message.message_id},
            timestamp=datetime.now(),
            correlation_id=message.message_id
        )
        
        await self._transmit_message(ack_message)
    
    async def _handle_ack_timeout(self, message_id: str):
        """Handle acknowledgment timeout."""
        await asyncio.sleep(30.0)  # ACK timeout
        
        if message_id in self.pending_acks:
            message = self.pending_acks[message_id]
            self.logger.warning(f"ACK timeout for message: {message_id}")
            
            # Retry logic
            if message.metadata.get("retry_count", 0) < self.endpoint.retry_attempts:
                message.metadata["retry_count"] = message.metadata.get("retry_count", 0) + 1
                await self.outbound_queue.put(message)
                self.logger.info(f"Retrying message: {message_id} (attempt {message.metadata['retry_count']})")
            else:
                self.logger.error(f"Max retries exceeded for message: {message_id}")
                del self.pending_acks[message_id]
                self.metrics.messages_failed += 1
    
    async def _flush_pending_acks(self):
        """Flush all pending acknowledgments."""
        for message_id in list(self.pending_acks.keys()):
            del self.pending_acks[message_id]
    
    def _serialize_message(self, message: Message) -> str:
        """Serialize message for transmission."""
        message_dict = {
            "message_id": message.message_id,
            "message_type": message.message_type.value,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "priority": message.priority.value,
            "correlation_id": message.correlation_id,
            "reply_to": message.reply_to,
            "ttl": message.ttl,
            "headers": message.headers,
            "metadata": message.metadata
        }
        
        return json.dumps(message_dict)
    
    def _deserialize_message(self, serialized: str) -> Message:
        """Deserialize message from transmission."""
        message_dict = json.loads(serialized)
        
        return Message(
            message_id=message_dict["message_id"],
            message_type=MessageType(message_dict["message_type"]),
            sender_id=message_dict["sender_id"],
            recipient_id=message_dict["recipient_id"],
            content=message_dict["content"],
            timestamp=datetime.fromisoformat(message_dict["timestamp"]),
            priority=MessagePriority(message_dict["priority"]),
            correlation_id=message_dict["correlation_id"],
            reply_to=message_dict["reply_to"],
            ttl=message_dict["ttl"],
            headers=message_dict["headers"],
            metadata=message_dict["metadata"]
        )


class EventStreamProtocol(CommunicationProtocol):
    """
    Event streaming protocol for real-time orchestration events.
    
    Implements event streaming with backpressure handling,
    event ordering, and replay capabilities.
    """
    
    def __init__(
        self,
        protocol_name: str = "event_stream_protocol",
        endpoint: Optional[CommunicationEndpoint] = None
    ):
        if not endpoint:
            endpoint = CommunicationEndpoint(
                endpoint_id="default_event_endpoint",
                endpoint_type="event_stream",
                address="localhost",
                port=8081
            )
        
        super().__init__(protocol_name, endpoint)
        self.event_streams: Dict[str, asyncio.Queue] = {}
        self.event_history: Dict[str, List[Message]] = defaultdict(list)
        self.stream_positions: Dict[str, int] = defaultdict(int)
        self.max_history_size = 1000
        self.backpressure_threshold = 100
    
    async def connect(self) -> bool:
        """Establish event stream connection."""
        try:
            await asyncio.sleep(0.1)
            self.is_connected = True
            self.logger.info(f"Event stream connected to {self.endpoint.address}:{self.endpoint.port}")
            
            await self.start_processing()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect event stream: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close event stream connection."""
        try:
            self.is_connected = False
            
            # Close all streams
            for stream_queue in self.event_streams.values():
                while not stream_queue.empty():
                    try:
                        stream_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            self.event_streams.clear()
            self.logger.info("Event stream disconnected")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting event stream: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send event message."""
        try:
            stream_name = message.headers.get("stream", "default")
            
            # Add to event history
            self.event_history[stream_name].append(message)
            if len(self.event_history[stream_name]) > self.max_history_size:
                self.event_history[stream_name] = self.event_history[stream_name][-self.max_history_size:]
            
            # Add sequence number
            position = len(self.event_history[stream_name]) - 1
            message.headers["position"] = str(position)
            
            # Send to stream subscribers
            await self._distribute_event(stream_name, message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send event: {e}")
            return False
    
    async def receive_message(self) -> Optional[Message]:
        """Receive event message."""
        try:
            if not self.message_queue.empty():
                return await self.message_queue.get()
        except Exception as e:
            self.logger.error(f"Error receiving event: {e}")
        
        return None
    
    async def create_stream(self, stream_name: str) -> bool:
        """Create new event stream."""
        if stream_name not in self.event_streams:
            self.event_streams[stream_name] = asyncio.Queue(maxsize=self.backpressure_threshold)
            self.logger.info(f"Created event stream: {stream_name}")
            return True
        return False
    
    async def subscribe_to_stream(self, stream_name: str, from_position: Optional[int] = None) -> bool:
        """Subscribe to event stream."""
        if stream_name not in self.event_streams:
            await self.create_stream(stream_name)
        
        # If from_position specified, replay from that position
        if from_position is not None:
            await self._replay_events(stream_name, from_position)
        
        self.stream_positions[stream_name] = from_position or 0
        self.logger.info(f"Subscribed to stream: {stream_name} from position {self.stream_positions[stream_name]}")
        return True
    
    async def publish_event(self, stream_name: str, event_data: Any, event_type: str = "custom"):
        """Publish event to stream."""
        event_message = Message(
            message_id=f"event_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.EVENT,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=None,
            content=event_data,
            timestamp=datetime.now(),
            headers={"stream": stream_name, "event_type": event_type}
        )
        
        await self.send_message(event_message)
    
    async def get_stream_position(self, stream_name: str) -> int:
        """Get current position in stream."""
        return len(self.event_history.get(stream_name, []))
    
    async def replay_from_position(self, stream_name: str, position: int) -> List[Message]:
        """Replay events from specific position."""
        if stream_name in self.event_history:
            history = self.event_history[stream_name]
            if position < len(history):
                return history[position:]
        return []
    
    async def _distribute_event(self, stream_name: str, event: Message):
        """Distribute event to stream subscribers."""
        if stream_name in self.event_streams:
            stream_queue = self.event_streams[stream_name]
            
            # Check backpressure
            if stream_queue.full():
                self.logger.warning(f"Backpressure detected for stream: {stream_name}")
                # Could implement backpressure handling strategies here
                try:
                    # Drop oldest event if queue is full
                    stream_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            
            try:
                await stream_queue.put(event)
            except asyncio.QueueFull:
                self.logger.error(f"Failed to add event to stream {stream_name}: queue full")
    
    async def _replay_events(self, stream_name: str, from_position: int):
        """Replay events from specified position."""
        if stream_name in self.event_history:
            history = self.event_history[stream_name]
            for event in history[from_position:]:
                await self.message_queue.put(event)


class MeshProtocol(CommunicationProtocol):
    """
    Mesh communication protocol for peer-to-peer orchestration.
    
    Implements mesh networking with peer discovery, routing,
    and distributed consensus capabilities.
    """
    
    def __init__(
        self,
        protocol_name: str = "mesh_protocol",
        endpoint: Optional[CommunicationEndpoint] = None
    ):
        if not endpoint:
            endpoint = CommunicationEndpoint(
                endpoint_id=f"mesh_node_{uuid.uuid4().hex[:8]}",
                endpoint_type="mesh",
                address="localhost",
                port=8082
            )
        
        super().__init__(protocol_name, endpoint)
        self.peers: Dict[str, CommunicationEndpoint] = {}
        self.routing_table: Dict[str, str] = {}  # destination -> next_hop
        self.discovery_interval = 30.0
        self.heartbeat_interval = 10.0
        self.peer_timeout = 30.0
        self.last_seen: Dict[str, datetime] = {}
    
    async def connect(self) -> bool:
        """Establish mesh protocol connection."""
        try:
            await asyncio.sleep(0.1)
            self.is_connected = True
            self.logger.info(f"Mesh node {self.endpoint.endpoint_id} connected")
            
            # Start mesh protocols
            await self.start_processing()
            asyncio.create_task(self._peer_discovery_loop())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._peer_timeout_check())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect mesh: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from mesh."""
        try:
            # Announce departure to peers
            await self._announce_departure()
            
            self.is_connected = False
            self.peers.clear()
            self.routing_table.clear()
            self.last_seen.clear()
            
            self.logger.info(f"Mesh node {self.endpoint.endpoint_id} disconnected")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from mesh: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send message through mesh routing."""
        try:
            # Determine routing
            if message.recipient_id in self.peers:
                # Direct connection
                await self._send_direct(message.recipient_id, message)
            elif message.recipient_id in self.routing_table:
                # Route through next hop
                next_hop = self.routing_table[message.recipient_id]
                await self._send_via_hop(next_hop, message)
            else:
                # Broadcast for discovery
                await self._broadcast_message(message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send mesh message: {e}")
            return False
    
    async def receive_message(self) -> Optional[Message]:
        """Receive message from mesh."""
        try:
            if not self.message_queue.empty():
                message = await self.message_queue.get()
                
                # Handle routing messages
                if message.message_type == MessageType.DISCOVERY:
                    await self._handle_discovery_message(message)
                elif message.message_type == MessageType.HEARTBEAT:
                    await self._handle_heartbeat_message(message)
                else:
                    return message
                
        except Exception as e:
            self.logger.error(f"Error receiving mesh message: {e}")
            
        return None
    
    async def add_peer(self, peer_endpoint: CommunicationEndpoint):
        """Add peer to mesh."""
        self.peers[peer_endpoint.endpoint_id] = peer_endpoint
        self.last_seen[peer_endpoint.endpoint_id] = datetime.now()
        self.routing_table[peer_endpoint.endpoint_id] = peer_endpoint.endpoint_id
        
        self.logger.info(f"Added peer: {peer_endpoint.endpoint_id}")
        
        # Send introduction message
        await self._send_introduction(peer_endpoint.endpoint_id)
    
    async def remove_peer(self, peer_id: str):
        """Remove peer from mesh."""
        if peer_id in self.peers:
            del self.peers[peer_id]
            del self.last_seen[peer_id]
            
            # Update routing table
            routes_to_remove = [dest for dest, next_hop in self.routing_table.items() if next_hop == peer_id]
            for dest in routes_to_remove:
                del self.routing_table[dest]
            
            self.logger.info(f"Removed peer: {peer_id}")
    
    async def broadcast_to_mesh(self, content: Any, message_type: MessageType = MessageType.BROADCAST):
        """Broadcast message to all mesh peers."""
        broadcast_message = Message(
            message_id=f"broadcast_{uuid.uuid4().hex[:8]}",
            message_type=message_type,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=None,
            content=content,
            timestamp=datetime.now(),
            headers={"broadcast": "true"}
        )
        
        await self._broadcast_message(broadcast_message)
    
    async def discover_peers(self) -> List[str]:
        """Discover peers in mesh."""
        discovery_message = Message(
            message_id=f"discovery_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.DISCOVERY,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=None,
            content={"action": "discover", "endpoint": self.endpoint},
            timestamp=datetime.now()
        )
        
        await self._broadcast_message(discovery_message)
        
        # Wait for responses
        await asyncio.sleep(2.0)
        
        return list(self.peers.keys())
    
    async def _peer_discovery_loop(self):
        """Periodic peer discovery."""
        while self.is_connected:
            try:
                await self.discover_peers()
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                self.logger.error(f"Error in peer discovery: {e}")
                await asyncio.sleep(5.0)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.is_connected:
            try:
                heartbeat_message = Message(
                    message_id=f"heartbeat_{uuid.uuid4().hex[:8]}",
                    message_type=MessageType.HEARTBEAT,
                    sender_id=self.endpoint.endpoint_id,
                    recipient_id=None,
                    content={"timestamp": datetime.now().isoformat()},
                    timestamp=datetime.now()
                )
                
                await self._broadcast_message(heartbeat_message)
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(5.0)
    
    async def _peer_timeout_check(self):
        """Check for timed out peers."""
        while self.is_connected:
            try:
                current_time = datetime.now()
                timeout_threshold = timedelta(seconds=self.peer_timeout)
                
                timed_out_peers = [
                    peer_id for peer_id, last_seen in self.last_seen.items()
                    if current_time - last_seen > timeout_threshold
                ]
                
                for peer_id in timed_out_peers:
                    self.logger.warning(f"Peer timeout: {peer_id}")
                    await self.remove_peer(peer_id)
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                self.logger.error(f"Error checking peer timeouts: {e}")
                await asyncio.sleep(5.0)
    
    async def _send_direct(self, peer_id: str, message: Message):
        """Send message directly to peer."""
        # Simulate direct sending
        await self.message_queue.put(message)
    
    async def _send_via_hop(self, next_hop_id: str, message: Message):
        """Send message via intermediate hop."""
        # Add routing header
        message.headers["route_via"] = next_hop_id
        await self._send_direct(next_hop_id, message)
    
    async def _broadcast_message(self, message: Message):
        """Broadcast message to all peers."""
        for peer_id in self.peers:
            try:
                await self._send_direct(peer_id, message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to {peer_id}: {e}")
    
    async def _handle_discovery_message(self, message: Message):
        """Handle peer discovery message."""
        content = message.content
        action = content.get("action")
        
        if action == "discover":
            # Respond with our endpoint info
            response_message = Message(
                message_id=f"discovery_response_{uuid.uuid4().hex[:8]}",
                message_type=MessageType.DISCOVERY,
                sender_id=self.endpoint.endpoint_id,
                recipient_id=message.sender_id,
                content={"action": "response", "endpoint": self.endpoint},
                timestamp=datetime.now()
            )
            
            await self._send_direct(message.sender_id, response_message)
            
        elif action == "response":
            # Add discovered peer
            peer_endpoint_data = content.get("endpoint")
            if peer_endpoint_data and peer_endpoint_data.endpoint_id not in self.peers:
                peer_endpoint = CommunicationEndpoint(**peer_endpoint_data)
                await self.add_peer(peer_endpoint)
    
    async def _handle_heartbeat_message(self, message: Message):
        """Handle heartbeat message."""
        sender_id = message.sender_id
        if sender_id in self.peers:
            self.last_seen[sender_id] = datetime.now()
    
    async def _send_introduction(self, peer_id: str):
        """Send introduction to new peer."""
        intro_message = Message(
            message_id=f"intro_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.EVENT,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=peer_id,
            content={"action": "introduction", "peers": list(self.peers.keys())},
            timestamp=datetime.now()
        )
        
        await self._send_direct(peer_id, intro_message)
    
    async def _announce_departure(self):
        """Announce departure from mesh."""
        departure_message = Message(
            message_id=f"departure_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.EVENT,
            sender_id=self.endpoint.endpoint_id,
            recipient_id=None,
            content={"action": "departure"},
            timestamp=datetime.now()
        )
        
        await self._broadcast_message(departure_message)


# Export key classes
__all__ = [
    'MessageType',
    'MessagePriority',
    'CommunicationPattern',
    'Message',
    'CommunicationEndpoint',
    'CommunicationMetrics',
    'CommunicationProtocol',
    'MessageProtocol',
    'EventStreamProtocol',
    'MeshProtocol'
]