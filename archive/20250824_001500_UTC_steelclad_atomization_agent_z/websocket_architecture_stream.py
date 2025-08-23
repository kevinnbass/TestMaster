#!/usr/bin/env python3
"""
WebSocket Architecture Stream - Agent A Hour 5
Real-time architecture monitoring via WebSocket

Provides real-time streaming of architecture health metrics and updates
to the dashboard using WebSocket connections.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Set, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from functools import wraps
from enum import Enum

# Import architecture components  
from core.architecture.architecture_integration import get_architecture_framework
from core.services.service_registry import get_service_registry

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    # Fallback for when websockets library not available
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = object


class MessageType(Enum):
    """WebSocket message types"""
    ARCHITECTURE_HEALTH = "architecture_health"
    SERVICE_STATUS = "service_status"
    LAYER_COMPLIANCE = "layer_compliance"
    DEPENDENCY_HEALTH = "dependency_health"
    INTEGRATION_STATUS = "integration_status"
    SYSTEM_ALERT = "system_alert"
    HEARTBEAT = "heartbeat"
    # Added from gamma_alpha_collaboration_dashboard
    COST_UPDATE = "cost_update"
    BUDGET_ALERT = "budget_alert"
    # Added from unified_greek_dashboard  
    SWARM_STATUS = "swarm_status_update"
    AGENTS_UPDATE = "agents_update"
    COORDINATION_MESSAGE = "coordination_message"
    # Added from unified_cross_agent_dashboard
    AGENT_SYNTHESIS = "agent_synthesis"
    PATTERN_INSIGHT = "pattern_insight"
    METRICS_UPDATE = "metrics_update"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: str
    client_id: Optional[str] = None
    sequence: int = 0
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            'type': self.type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'client_id': self.client_id,
            'sequence': self.sequence
        })


@dataclass  
class APIUsageMetrics:
    """API usage and cost tracking metrics"""
    api_calls: Dict[str, int]
    api_costs: Dict[str, float] 
    model_usage: Dict[str, int]
    daily_budget: float
    budget_alerts: List[Dict[str, Any]]
    
    def track_api_call(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Track API call with cost calculation"""
        cost_estimates = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }
        
        cost = 0.0
        if model in cost_estimates:
            rates = cost_estimates[model]
            cost = (input_tokens * rates["input"] / 1000) + (output_tokens * rates["output"] / 1000)
        
        # Update tracking
        self.api_calls[f"{provider}:{model}"] += 1
        self.model_usage[model] += input_tokens + output_tokens
        
        today = datetime.now().date().isoformat()
        if today not in self.api_costs:
            self.api_costs[today] = 0.0
        self.api_costs[today] += cost
        
        return {
            "provider": provider,
            "model": model, 
            "cost": cost,
            "timestamp": datetime.now().isoformat()
        }


class AgentStatus(Enum):
    """Agent status enumeration for coordination"""
    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing" 
    ERROR = "error"
    OFFLINE = "offline"


class CoordinationType(Enum):
    """Coordination message types"""
    STATUS_UPDATE = "status_update"
    HANDOFF_REQUEST = "handoff_request"
    HANDOFF_COMPLETE = "handoff_complete"
    RESOURCE_REQUEST = "resource_request"
    PRIORITY_ALERT = "priority_alert"
    SYNC_REQUEST = "sync_request"


@dataclass
class AgentInfo:
    """Agent information tracking"""
    agent_id: str
    agent_type: str
    status: AgentStatus
    last_heartbeat: datetime
    current_phase: str
    progress_percentage: float
    active_tasks: List[str]
    performance_metrics: Dict[str, Any]
    coordination_subscriptions: List[str]


@dataclass
class CoordinationMessage:
    """Inter-agent coordination message"""
    message_id: str
    sender_agent: str
    target_agents: List[str]
    coordination_type: CoordinationType
    payload: Dict[str, Any]
    timestamp: datetime
    priority: str
    requires_response: bool


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure for monitoring"""
    timestamp: datetime
    response_time_ms: float
    throughput_requests_per_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_connections: int
    error_rate: float
    queue_size: int


@dataclass
class MonitoringAlert:
    """Monitoring alert for system health"""
    alert_id: str
    level: str
    service: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ArchitectureWebSocketStream:
    """
    WebSocket stream for real-time architecture monitoring
    
    Streams architecture health metrics, service status updates,
    and system alerts to connected dashboard clients.
    """
    
    def __init__(self, port: int = 8765, update_interval: int = 5):
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.update_interval = update_interval
        
        # Architecture components
        self.framework = get_architecture_framework()
        self.service_registry = get_service_registry()
        
        # WebSocket management
        self.clients: Set[WebSocketServerProtocol] = set()
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.message_sequence = 0
        
        # Multi-agent coordination (from unified_greek_dashboard)
        self.agent_status_cache: Dict[str, Dict[str, Any]] = {}
        self.swarm_coordination_data: Dict[str, Any] = {}
        self.coordination_subscriptions: Dict[str, List[str]] = {}
        
        # API usage tracking (from gamma_alpha_collaboration_dashboard)
        self.api_metrics = APIUsageMetrics(
            api_calls=defaultdict(int),
            api_costs={},
            model_usage=defaultdict(int),
            daily_budget=100.0,
            budget_alerts=[]
        )
        
        # Cross-agent synthesis tracking (from unified_cross_agent_dashboard)
        self.synthesis_processes: Dict[str, Dict[str, Any]] = {}
        self.pattern_insights: Dict[str, Dict[str, Any]] = {}
        self.agent_synthesis_cache = deque(maxlen=1000)
        
        # Multi-agent coordination service integration
        self.agents: Dict[str, AgentInfo] = {}
        self.coordination_history = deque(maxlen=1000)
        self.active_handoffs: Dict[str, Dict[str, Any]] = {}
        self.message_queue = deque(maxlen=500)
        self.coord_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.coordination_stats = {
            'messages_processed': 0,
            'handoffs_completed': 0,
            'sync_operations': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Performance monitoring integration
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1min intervals
        self.current_performance_metrics: Optional[PerformanceMetrics] = None
        self.monitoring_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history = deque(maxlen=1000)
        self.performance_thresholds = {
            'max_response_time_ms': 50.0,
            'max_error_rate': 0.05,
            'max_cpu_usage': 80.0,
            'max_memory_usage_mb': 1024.0,
            'min_throughput_rps': 10.0,
            'max_queue_size': 100
        }
        
        # API service integration
        self.request_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 30
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.rate_limit_window = 60
        self.max_requests_per_window = 100
        
        # Optimized stream configuration
        self.stream_config = {
            'architecture_health': True,
            'service_status': True,
            'layer_compliance': True,
            'dependency_health': True,
            'integration_status': True,
            'heartbeat_interval': 15,  # Reduced for better responsiveness
            'enable_compression': True,
            'enable_batching': True,
            'max_message_size': 8192,
            'connection_timeout': 60
        }
        
        # Message queues for different priorities with connection pooling
        self.high_priority_queue: List[WebSocketMessage] = []
        self.normal_priority_queue: List[WebSocketMessage] = []
        self.batch_queue: List[WebSocketMessage] = []
        self.max_batch_size = 10
        self.batch_timeout = 2.0  # seconds
        self.last_batch_time = time.time()
        
        # Enhanced performance metrics
        self.stream_metrics = {
            'messages_sent': 0,
            'clients_connected': 0,
            'clients_disconnected': 0,
            'errors': 0,
            'start_time': datetime.now(),
            'avg_response_time': 0.0,
            'peak_concurrent_clients': 0,
            'messages_batched': 0,
            'compression_ratio': 0.0
        }
        
        self.logger.info(f"Architecture WebSocket Stream initialized on port {port}")
    
    async def register_client(self, websocket: WebSocketServerProtocol, path: str):
        """Register new WebSocket client"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.error("WebSockets library not available")
            return
        
        client_id = f"client_{len(self.clients)}_{int(time.time())}"
        
        try:
            self.clients.add(websocket)
            self.client_metadata[client_id] = {
                'connected_at': datetime.now().isoformat(),
                'path': path,
                'websocket': websocket
            }
            
            self.stream_metrics['clients_connected'] += 1
            
            self.logger.info(f"Client registered: {client_id} ({len(self.clients)} total)")
            
            # Send initial data
            await self._send_initial_data(websocket, client_id)
            
            # Handle client messages
            await self._handle_client_messages(websocket, client_id)
            
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
            self.stream_metrics['errors'] += 1
        finally:
            await self._unregister_client(websocket, client_id)
    
    async def _send_initial_data(self, websocket: WebSocketServerProtocol, client_id: str):
        """Send initial data to newly connected client"""
        try:
            # Send current architecture health
            health_data = self.framework.get_architecture_metrics()
            health_message = WebSocketMessage(
                type=MessageType.ARCHITECTURE_HEALTH,
                data=health_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(health_message.to_json())
            
            # Send service registry status
            service_data = self.service_registry.get_registration_report()
            service_message = WebSocketMessage(
                type=MessageType.SERVICE_STATUS,
                data=service_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(service_message.to_json())
            
            self.logger.debug(f"Initial data sent to {client_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send initial data to {client_id}: {e}")
    
    async def _handle_client_messages(self, websocket: WebSocketServerProtocol, client_id: str):
        """Handle incoming messages from client"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_client_message(client_id, data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON from {client_id}: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in message handler for {client_id}: {e}")
    
    async def _process_client_message(self, client_id: str, data: Dict[str, Any]):
        """Process message from client"""
        message_type = data.get('type')
        
        if message_type == 'configure_stream':
            # Update stream configuration for client
            config = data.get('config', {})
            self.stream_config.update(config)
            self.logger.info(f"Stream configuration updated by {client_id}")
            
        elif message_type == 'request_update':
            # Send immediate update
            await self._send_architecture_update(client_id)
            
        elif message_type == 'heartbeat':
            # Respond to heartbeat
            await self._send_heartbeat(client_id)
            
        # Multi-agent coordination handlers (from unified_greek_dashboard)
        elif message_type == 'subscribe_agent_updates':
            agent_types = data.get('agent_types', [])
            self.coordination_subscriptions[client_id] = agent_types
            await self._send_confirmation(client_id, 'subscription_confirmed', {'agent_types': agent_types})
            
        elif message_type == 'request_swarm_status':
            await self._send_swarm_status(client_id)
            
        # API cost tracking handlers (from gamma_alpha_collaboration_dashboard)  
        elif message_type == 'request_cost_update':
            await self._send_cost_summary(client_id)
            
        elif message_type == 'track_api_call':
            call_data = self.api_metrics.track_api_call(
                data.get('provider', 'unknown'),
                data.get('model', 'unknown'), 
                data.get('input_tokens', 0),
                data.get('output_tokens', 0)
            )
            await self._broadcast_cost_update(call_data)
            
        # Cross-agent synthesis handlers (from unified_cross_agent_dashboard)
        elif message_type == 'request_synthesis_status':
            await self._send_synthesis_status(client_id)
    
    async def _unregister_client(self, websocket: WebSocketServerProtocol, client_id: str):
        """Unregister WebSocket client"""
        try:
            self.clients.discard(websocket)
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]
            
            self.stream_metrics['clients_disconnected'] += 1
            
            self.logger.info(f"Client unregistered: {client_id} ({len(self.clients)} remaining)")
            
        except Exception as e:
            self.logger.error(f"Error unregistering client {client_id}: {e}")
    
    async def start_streaming(self):
        """Start WebSocket streaming service"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.error("Cannot start WebSocket stream - websockets library not available")
            return
        
        self.running = True
        self.logger.info(f"Starting WebSocket architecture stream on port {self.port}")
        
        # Start background update task
        update_task = asyncio.create_task(self._background_updates())
        
        try:
            # Start WebSocket server
            async with websockets.serve(self.register_client, "localhost", self.port):
                self.logger.info("WebSocket server started")
                await update_task
        
        except Exception as e:
            self.logger.error(f"WebSocket server error: {e}")
            self.running = False
            update_task.cancel()
    
    async def _background_updates(self):
        """Background task for periodic updates"""
        last_heartbeat = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Send regular architecture updates
                if self.clients and self.stream_config.get('architecture_health', True):
                    await self._broadcast_architecture_update()
                
                # Send heartbeat if needed
                if (current_time - last_heartbeat) >= self.stream_config.get('heartbeat_interval', 30):
                    await self._broadcast_heartbeat()
                    last_heartbeat = current_time
                
                # Process message queues
                await self._process_message_queues()
                
                # Wait for next update cycle
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in background updates: {e}")
                self.stream_metrics['errors'] += 1
    
    async def _broadcast_architecture_update(self):
        """Broadcast architecture health update to all clients"""
        try:
            health_data = self.framework.get_architecture_metrics()
            
            message = WebSocketMessage(
                type=MessageType.ARCHITECTURE_HEALTH,
                data=health_data,
                timestamp=datetime.now().isoformat(),
                sequence=self._get_next_sequence()
            )
            
            await self._broadcast_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast architecture update: {e}")
    
    async def _send_architecture_update(self, client_id: str):
        """Send architecture update to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            health_data = self.framework.get_architecture_metrics()
            
            message = WebSocketMessage(
                type=MessageType.ARCHITECTURE_HEALTH,
                data=health_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send update to {client_id}: {e}")
    
    async def _broadcast_heartbeat(self):
        """Broadcast heartbeat to all clients"""
        message = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={'server_time': datetime.now().isoformat()},
            timestamp=datetime.now().isoformat(),
            sequence=self._get_next_sequence()
        )
        
        await self._broadcast_message(message)
    
    async def _send_heartbeat(self, client_id: str):
        """Send heartbeat to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            
            message = WebSocketMessage(
                type=MessageType.HEARTBEAT,
                data={'server_time': datetime.now().isoformat()},
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat to {client_id}: {e}")
    
    async def _send_confirmation(self, client_id: str, message_type: str, data: Dict[str, Any]):
        """Send confirmation message to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            message = WebSocketMessage(
                type=MessageType.SYSTEM_ALERT,
                data={'alert_type': message_type, 'data': data},
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send confirmation to {client_id}: {e}")
    
    async def _send_swarm_status(self, client_id: str):
        """Send swarm coordination status to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            status_data = {
                'agents': self.agent_status_cache,
                'coordination_data': self.swarm_coordination_data,
                'timestamp': datetime.now().isoformat()
            }
            
            message = WebSocketMessage(
                type=MessageType.SWARM_STATUS,
                data=status_data,
                timestamp=datetime.now().isoformat(), 
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send swarm status to {client_id}: {e}")
    
    async def _send_cost_summary(self, client_id: str):
        """Send API cost summary to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            cost_data = {
                'api_calls': dict(self.api_metrics.api_calls),
                'api_costs': self.api_metrics.api_costs,
                'model_usage': dict(self.api_metrics.model_usage),
                'daily_budget': self.api_metrics.daily_budget,
                'budget_alerts': self.api_metrics.budget_alerts
            }
            
            message = WebSocketMessage(
                type=MessageType.COST_UPDATE,
                data=cost_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send cost summary to {client_id}: {e}")
    
    async def _send_synthesis_status(self, client_id: str):
        """Send cross-agent synthesis status to specific client"""
        try:
            client_data = self.client_metadata.get(client_id)
            if not client_data:
                return
            
            websocket = client_data['websocket']
            synthesis_data = {
                'active_processes': self.synthesis_processes,
                'pattern_insights': self.pattern_insights,
                'recent_synthesis': list(self.agent_synthesis_cache)[-10:]
            }
            
            message = WebSocketMessage(
                type=MessageType.AGENT_SYNTHESIS,
                data=synthesis_data,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                sequence=self._get_next_sequence()
            )
            
            await websocket.send(message.to_json())
            
        except Exception as e:
            self.logger.error(f"Failed to send synthesis status to {client_id}: {e}")
    
    async def _broadcast_cost_update(self, call_data: Dict[str, Any]):
        """Broadcast cost update to all subscribed clients"""
        message = WebSocketMessage(
            type=MessageType.COST_UPDATE,
            data=call_data,
            timestamp=datetime.now().isoformat(),
            sequence=self._get_next_sequence()
        )
        
        await self._broadcast_message(message)
    
    # Multi-Agent Coordination Methods
    def register_agent(self, agent_id: str, agent_type: str, 
                      initial_status: AgentStatus = AgentStatus.ACTIVE) -> bool:
        """Register new agent with coordination service"""
        try:
            self.agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                status=initial_status,
                last_heartbeat=datetime.now(),
                current_phase="initialization",
                progress_percentage=0.0,
                active_tasks=[],
                performance_metrics={},
                coordination_subscriptions=[]
            )
            
            self.logger.info(f"Agent {agent_id} ({agent_type}) registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {e}")
            self.coordination_stats['errors'] += 1
            return False
    
    def update_agent_status(self, agent_id: str, status: AgentStatus, 
                           phase: Optional[str] = None,
                           progress: Optional[float] = None,
                           tasks: Optional[List[str]] = None) -> bool:
        """Update agent status and broadcast changes"""
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Attempted to update unregistered agent: {agent_id}")
                return False
            
            agent = self.agents[agent_id]
            agent.status = status
            agent.last_heartbeat = datetime.now()
            
            if phase is not None:
                agent.current_phase = phase
            if progress is not None:
                agent.progress_percentage = progress
            if tasks is not None:
                agent.active_tasks = tasks
            
            # Broadcast update via WebSocket
            self.queue_alert("agent_status_update", json.dumps({
                'agent_id': agent_id,
                'status': status.value,
                'phase': agent.current_phase,
                'progress': agent.progress_percentage
            }), priority="normal")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent {agent_id} status: {e}")
            self.coordination_stats['errors'] += 1
            return False
    
    def send_coordination_message(self, sender: str, targets: List[str],
                                 coord_type: CoordinationType, payload: Dict[str, Any],
                                 priority: str = "normal") -> str:
        """Send coordination message between agents"""
        try:
            message_id = f"coord_{int(datetime.now().timestamp())}_{sender}"
            
            message = CoordinationMessage(
                message_id=message_id,
                sender_agent=sender,
                target_agents=targets,
                coordination_type=coord_type,
                payload=payload,
                timestamp=datetime.now(),
                priority=priority,
                requires_response=False
            )
            
            self.message_queue.append(message)
            self.coordination_history.append(message)
            self.coordination_stats['messages_processed'] += 1
            
            # Broadcast coordination message
            self.queue_alert("coordination_message", json.dumps({
                'message_id': message_id,
                'sender': sender,
                'targets': targets,
                'type': coord_type.value,
                'payload': payload
            }), priority)
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Failed to send coordination message: {e}")
            self.coordination_stats['errors'] += 1
            return ""
    
    def request_agent_handoff(self, from_agent: str, to_agent: str, 
                             handoff_data: Dict[str, Any]) -> str:
        """Request agent handoff with data transfer"""
        try:
            handoff_id = f"handoff_{int(datetime.now().timestamp())}_{from_agent}_{to_agent}"
            
            self.active_handoffs[handoff_id] = {
                'from_agent': from_agent,
                'to_agent': to_agent,
                'handoff_data': handoff_data,
                'status': 'requested',
                'timestamp': datetime.now(),
                'completion_time': None
            }
            
            self.send_coordination_message(
                sender=from_agent,
                targets=[to_agent],
                coord_type=CoordinationType.HANDOFF_REQUEST,
                payload={'handoff_id': handoff_id, 'data': handoff_data},
                priority="high"
            )
            
            return handoff_id
            
        except Exception as e:
            self.logger.error(f"Failed to request handoff: {e}")
            return ""
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        try:
            active_agents = [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]
            
            return {
                'total_agents': len(self.agents),
                'active_agents': len(active_agents),
                'agent_details': {aid: asdict(agent) for aid, agent in self.agents.items()},
                'active_handoffs': len(self.active_handoffs),
                'message_queue_size': len(self.message_queue),
                'coordination_stats': self.coordination_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get swarm status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    # Performance Monitoring Methods
    def record_performance_metrics(self, response_time_ms: float, 
                                 throughput_rps: float,
                                 active_connections: int,
                                 error_rate: float = 0.0,
                                 queue_size: int = 0):
        """Record performance metrics"""
        try:
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                throughput_requests_per_sec=throughput_rps,
                cpu_usage_percent=0.0,  # Would get actual CPU usage
                memory_usage_mb=0.0,  # Would get actual memory usage
                active_connections=active_connections,
                error_rate=error_rate,
                queue_size=queue_size
            )
            
            self.current_performance_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Check performance thresholds
            self._check_performance_thresholds(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to record performance metrics: {e}")
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance metrics exceed thresholds"""
        violations = []
        
        if metrics.response_time_ms > self.performance_thresholds['max_response_time_ms']:
            violations.append(f"Response time {metrics.response_time_ms}ms exceeds {self.performance_thresholds['max_response_time_ms']}ms")
        
        if metrics.error_rate > self.performance_thresholds['max_error_rate']:
            violations.append(f"Error rate {metrics.error_rate*100:.1f}% exceeds {self.performance_thresholds['max_error_rate']*100:.1f}%")
        
        # Generate performance alerts
        for violation in violations:
            alert_id = f"perf_{int(datetime.now().timestamp())}"
            alert = MonitoringAlert(
                alert_id=alert_id,
                level="warning",
                service="PerformanceMonitoring",
                message=f"Performance threshold violation: {violation}",
                details=asdict(metrics),
                timestamp=datetime.now()
            )
            
            self.monitoring_alerts[alert_id] = alert
            self.queue_alert("performance_alert", json.dumps({
                'alert_id': alert_id,
                'message': violation,
                'metrics': asdict(metrics)
            }), priority="high")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring service status"""
        return {
            'active_alerts': len(self.monitoring_alerts),
            'metrics_collected': len(self.metrics_history),
            'current_performance': asdict(self.current_performance_metrics) if self.current_performance_metrics else None,
            'performance_thresholds': self.performance_thresholds,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _broadcast_message(self, message: WebSocketMessage):
        """Broadcast message to all connected clients with connection pooling"""
        if not self.clients:
            return
        
        start_time = time.time()
        message_json = message.to_json()
        
        # Apply compression if enabled and message is large
        if (self.stream_config.get('enable_compression', True) and 
            len(message_json) > 1024):
            try:
                import gzip
                compressed = gzip.compress(message_json.encode())
                if len(compressed) < len(message_json):
                    message_json = compressed.decode('latin-1')
                    self.stream_metrics['compression_ratio'] = len(compressed) / len(message_json)
            except Exception:
                pass  # Use uncompressed if compression fails
        
        # Use asyncio.gather for concurrent message sending
        send_tasks = []
        active_clients = list(self.clients.copy())
        
        for client in active_clients:
            send_tasks.append(self._safe_send_message(client, message_json))
        
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            
            # Track performance metrics
            response_time = time.time() - start_time
            self.stream_metrics['avg_response_time'] = (
                (self.stream_metrics['avg_response_time'] * 0.9) + (response_time * 0.1)
            )
            self.stream_metrics['peak_concurrent_clients'] = max(
                self.stream_metrics['peak_concurrent_clients'], len(active_clients)
            )
            
            # Remove failed clients
            failed_clients = [
                client for client, result in zip(active_clients, results)
                if isinstance(result, Exception)
            ]
            
            for client in failed_clients:
                self.clients.discard(client)
    
    async def _safe_send_message(self, client: WebSocketServerProtocol, message_json: str):
        """Safely send message to client with error handling"""
        try:
            await client.send(message_json)
            self.stream_metrics['messages_sent'] += 1
            return True
        except Exception as e:
            self.logger.warning(f"Failed to send message to client: {e}")
            raise e
    
    async def _process_message_queues(self):
        """Process high and normal priority message queues with batch optimization"""
        current_time = time.time()
        
        # Always process high priority messages immediately
        while self.high_priority_queue:
            message = self.high_priority_queue.pop(0)
            await self._broadcast_message(message)
        
        # Process normal priority messages with batching
        if self.stream_config.get('enable_batching', True):
            # Add normal priority messages to batch queue
            while self.normal_priority_queue:
                message = self.normal_priority_queue.pop(0)
                self.batch_queue.append(message)
            
            # Process batch if conditions are met
            should_process_batch = (
                len(self.batch_queue) >= self.max_batch_size or
                (self.batch_queue and 
                 (current_time - self.last_batch_time) >= self.batch_timeout)
            )
            
            if should_process_batch:
                await self._process_batch_messages()
                self.last_batch_time = current_time
        else:
            # Process normal messages individually if batching disabled
            while self.normal_priority_queue:
                message = self.normal_priority_queue.pop(0)
                await self._broadcast_message(message)
    
    async def _process_batch_messages(self):
        """Process batched messages for improved performance"""
        if not self.batch_queue:
            return
        
        try:
            # Create batch message
            batch_data = {
                'batch_id': f"batch_{int(time.time())}",
                'message_count': len(self.batch_queue),
                'messages': [
                    {
                        'type': msg.type.value,
                        'data': msg.data,
                        'timestamp': msg.timestamp,
                        'sequence': msg.sequence
                    }
                    for msg in self.batch_queue
                ]
            }
            
            batch_message = WebSocketMessage(
                type=MessageType.SYSTEM_ALERT,  # Use system alert for batch messages
                data={
                    'alert_type': 'batch_update',
                    'batch': batch_data
                },
                timestamp=datetime.now().isoformat(),
                sequence=self._get_next_sequence()
            )
            
            await self._broadcast_message(batch_message)
            
            self.stream_metrics['messages_batched'] += len(self.batch_queue)
            self.batch_queue.clear()
            
        except Exception as e:
            self.logger.error(f"Error processing batch messages: {e}")
            # Fallback to individual processing
            for message in self.batch_queue:
                await self._broadcast_message(message)
            self.batch_queue.clear()
    
    def _get_next_sequence(self) -> int:
        """Get next message sequence number"""
        self.message_sequence += 1
        return self.message_sequence
    
    def queue_alert(self, alert_type: str, message: str, priority: str = "normal"):
        """Queue system alert for broadcast"""
        alert_message = WebSocketMessage(
            type=MessageType.SYSTEM_ALERT,
            data={
                'alert_type': alert_type,
                'message': message,
                'priority': priority
            },
            timestamp=datetime.now().isoformat(),
            sequence=self._get_next_sequence()
        )
        
        if priority == "high":
            self.high_priority_queue.append(alert_message)
        else:
            self.normal_priority_queue.append(alert_message)
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get enhanced WebSocket stream performance metrics"""
        uptime = (datetime.now() - self.stream_metrics['start_time']).total_seconds()
        messages_per_second = self.stream_metrics['messages_sent'] / max(uptime, 1)
        
        return {
            'running': self.running,
            'clients_connected': len(self.clients),
            'total_connected': self.stream_metrics['clients_connected'],
            'total_disconnected': self.stream_metrics['clients_disconnected'],
            'messages_sent': self.stream_metrics['messages_sent'],
            'messages_batched': self.stream_metrics['messages_batched'],
            'messages_per_second': round(messages_per_second, 2),
            'avg_response_time_ms': round(self.stream_metrics['avg_response_time'] * 1000, 2),
            'peak_concurrent_clients': self.stream_metrics['peak_concurrent_clients'],
            'compression_ratio': round(self.stream_metrics['compression_ratio'], 3),
            'errors': self.stream_metrics['errors'],
            'uptime_seconds': uptime,
            'websockets_available': WEBSOCKETS_AVAILABLE,
            'port': self.port,
            'update_interval': self.update_interval,
            'batch_queue_size': len(self.batch_queue),
            'high_priority_queue_size': len(self.high_priority_queue),
            'normal_priority_queue_size': len(self.normal_priority_queue),
            # Multi-agent coordination metrics
            'coordination': {
                'active_agents': len(self.agent_status_cache),
                'coordination_subscriptions': len(self.coordination_subscriptions),
                'synthesis_processes': len(self.synthesis_processes),
                'pattern_insights': len(self.pattern_insights)
            },
            # API cost tracking metrics  
            'api_usage': {
                'total_calls': sum(self.api_metrics.api_calls.values()),
                'total_cost': sum(self.api_metrics.api_costs.values()),
                'daily_budget': self.api_metrics.daily_budget,
                'active_alerts': len(self.api_metrics.budget_alerts)
            },
            # Multi-agent coordination metrics
            'coordination': {
                'registered_agents': len(self.agents),
                'active_agents': len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
                'coordination_subscriptions': len(self.coordination_subscriptions),
                'messages_processed': self.coordination_stats['messages_processed'],
                'active_handoffs': len(self.active_handoffs)
            },
            # Performance monitoring metrics
            'monitoring': {
                'active_alerts': len(self.monitoring_alerts),
                'metrics_collected': len(self.metrics_history),
                'performance_score': self._calculate_performance_score() if self.current_performance_metrics else 0.0,
                'latency_within_target': self.current_performance_metrics.response_time_ms <= 50.0 if self.current_performance_metrics else False
            },
            'optimizations': {
                'compression_enabled': self.stream_config.get('enable_compression', False),
                'batching_enabled': self.stream_config.get('enable_batching', False),
                'max_batch_size': self.max_batch_size,
                'batch_timeout_seconds': self.batch_timeout,
                'heartbeat_interval': self.stream_config.get('heartbeat_interval', 30)
            }
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100) based on current metrics"""
        if not self.current_performance_metrics:
            return 0.0
        
        metrics = self.current_performance_metrics
        score = 100.0
        
        # Response time score (50% weight)
        if metrics.response_time_ms <= 50.0:
            response_score = 50.0
        else:
            response_score = max(0, 50.0 - (metrics.response_time_ms - 50.0) / 2)
        
        # Error rate score (30% weight) 
        if metrics.error_rate <= 0.05:
            error_score = 30.0
        else:
            error_score = max(0, 30.0 - (metrics.error_rate - 0.05) * 600)
        
        # Throughput score (20% weight)
        if metrics.throughput_requests_per_sec >= 10.0:
            throughput_score = 20.0
        else:
            throughput_score = max(0, 20.0 * (metrics.throughput_requests_per_sec / 10.0))
        
        return round(response_score + error_score + throughput_score, 1)
    
    # API Service Integration Methods
    def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health status for API endpoints"""
        return {
            'websocket_service': {
                'running': self.running,
                'clients_connected': len(self.clients),
                'port': self.port,
                'status': 'healthy' if self.running else 'stopped'
            },
            'coordination_service': self.get_swarm_status(),
            'monitoring_service': self.get_monitoring_status(),
            'api_usage': {
                'cache_size': len(self.request_cache),
                'rate_limited_clients': len(self.rate_limits),
                'total_requests': sum(len(requests) for requests in self.rate_limits.values())
            },
            'overall_health': self._determine_overall_health(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_overall_health(self) -> str:
        """Determine overall system health"""
        if not self.running:
            return 'stopped'
        
        # Check critical metrics
        latency_good = (not self.current_performance_metrics or 
                       self.current_performance_metrics.response_time_ms <= 50.0)
        low_errors = self.coordination_stats['errors'] < 5
        few_alerts = len(self.monitoring_alerts) < 3
        
        if latency_good and low_errors and few_alerts:
            return 'healthy'
        elif latency_good or (low_errors and few_alerts):
            return 'warning'
        else:
            return 'critical'
    
    def stop_streaming(self):
        """Stop WebSocket streaming service"""
        self.running = False
        self.logger.info("WebSocket streaming stopped")


# Global stream instance
_websocket_stream: Optional[ArchitectureWebSocketStream] = None


def get_websocket_stream(port: int = 8765) -> ArchitectureWebSocketStream:
    """Get global WebSocket stream instance"""
    global _websocket_stream
    if _websocket_stream is None:
        _websocket_stream = ArchitectureWebSocketStream(port=port)
    return _websocket_stream


async def start_architecture_stream(port: int = 8765) -> None:
    """Start architecture WebSocket streaming service"""
    stream = get_websocket_stream(port)
    await stream.start_streaming()