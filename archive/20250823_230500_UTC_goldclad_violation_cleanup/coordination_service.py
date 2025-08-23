#!/usr/bin/env python3
"""
Multi-Agent Coordination Service Module - Agent Z Phase 2  
Unified coordination service for Latin_End swarm management

Provides comprehensive multi-agent coordination including:
- Agent status monitoring and synchronization
- Cross-agent message routing and coordination
- Swarm health tracking and reporting
- Agent handoff protocol management
- Real-time coordination event broadcasting
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status enumeration"""
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
    agent_type: str  # X, Y, Z, etc.
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


class MultiAgentCoordinationService:
    """
    Multi-agent coordination service providing unified coordination
    capabilities for the Latin_End swarm dashboard system.
    """
    
    def __init__(self):
        # Agent registry and status tracking
        self.agents: Dict[str, AgentInfo] = {}
        self.coordination_history = deque(maxlen=1000)
        self.active_handoffs: Dict[str, Dict[str, Any]] = {}
        
        # Message routing and subscriptions
        self.message_queue = deque(maxlen=500)
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.coordination_stats = {
            'messages_processed': 0,
            'handoffs_completed': 0,
            'sync_operations': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Swarm health monitoring
        self.swarm_health_cache = {}
        self.performance_thresholds = {
            'max_response_time': 5.0,  # seconds
            'min_heartbeat_interval': 30.0,  # seconds  
            'max_error_rate': 0.05  # 5%
        }
        
        logger.info("Multi-Agent Coordination Service initialized")
    
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
            
            logger.info(f"Agent {agent_id} ({agent_type}) registered successfully")
            self._broadcast_agent_update(agent_id, "registration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            self.coordination_stats['errors'] += 1
            return False
    
    def update_agent_status(self, agent_id: str, status: AgentStatus, 
                           phase: Optional[str] = None,
                           progress: Optional[float] = None,
                           tasks: Optional[List[str]] = None) -> bool:
        """Update agent status and broadcast changes"""
        try:
            if agent_id not in self.agents:
                logger.warning(f"Attempted to update unregistered agent: {agent_id}")
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
            
            self._broadcast_agent_update(agent_id, "status_update")
            logger.debug(f"Agent {agent_id} status updated: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id} status: {e}")
            self.coordination_stats['errors'] += 1
            return False
    
    def send_coordination_message(self, sender: str, targets: List[str],
                                 coord_type: CoordinationType, payload: Dict[str, Any],
                                 priority: str = "normal", 
                                 requires_response: bool = False) -> str:
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
                requires_response=requires_response
            )
            
            self.message_queue.append(message)
            self.coordination_history.append(message)
            self.coordination_stats['messages_processed'] += 1
            
            # Notify subscribed agents
            for target in targets:
                if target in self.subscriptions:
                    self._notify_subscribers(target, message)
            
            logger.info(f"Coordination message sent: {sender} → {targets} ({coord_type})")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send coordination message: {e}")
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
            
            # Send coordination message for handoff
            self.send_coordination_message(
                sender=from_agent,
                targets=[to_agent],
                coord_type=CoordinationType.HANDOFF_REQUEST,
                payload={'handoff_id': handoff_id, 'data': handoff_data},
                priority="high",
                requires_response=True
            )
            
            logger.info(f"Agent handoff requested: {from_agent} → {to_agent}")
            return handoff_id
            
        except Exception as e:
            logger.error(f"Failed to request handoff: {e}")
            self.coordination_stats['errors'] += 1
            return ""
    
    def complete_agent_handoff(self, handoff_id: str, success: bool = True) -> bool:
        """Complete agent handoff process"""
        try:
            if handoff_id not in self.active_handoffs:
                logger.warning(f"Attempted to complete unknown handoff: {handoff_id}")
                return False
            
            handoff = self.active_handoffs[handoff_id]
            handoff['status'] = 'completed' if success else 'failed'
            handoff['completion_time'] = datetime.now()
            
            if success:
                self.coordination_stats['handoffs_completed'] += 1
            
            # Notify involved agents
            self.send_coordination_message(
                sender="coordination_service",
                targets=[handoff['from_agent'], handoff['to_agent']],
                coord_type=CoordinationType.HANDOFF_COMPLETE,
                payload={'handoff_id': handoff_id, 'success': success},
                priority="high"
            )
            
            logger.info(f"Agent handoff completed: {handoff_id} ({'success' if success else 'failed'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete handoff {handoff_id}: {e}")
            self.coordination_stats['errors'] += 1
            return False
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        try:
            active_agents = [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]
            
            status = {
                'total_agents': len(self.agents),
                'active_agents': len(active_agents),
                'agent_details': {aid: asdict(agent) for aid, agent in self.agents.items()},
                'active_handoffs': len(self.active_handoffs),
                'message_queue_size': len(self.message_queue),
                'coordination_stats': self.coordination_stats,
                'swarm_health': self._calculate_swarm_health(),
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get swarm status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _broadcast_agent_update(self, agent_id: str, update_type: str):
        """Broadcast agent update to WebSocket service"""
        try:
            # Import here to avoid circular imports
            from .websocket_service import get_websocket_service
            
            ws_service = get_websocket_service()
            agent_data = asdict(self.agents[agent_id]) if agent_id in self.agents else {}
            ws_service.broadcast_agent_status(agent_id, {
                'update_type': update_type,
                'agent_data': agent_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"Failed to broadcast agent update: {e}")
    
    def _notify_subscribers(self, target_agent: str, message: CoordinationMessage):
        """Notify subscribers of coordination message"""
        # Implementation for notifying subscribed agents
        pass
    
    def _calculate_swarm_health(self) -> Dict[str, Any]:
        """Calculate overall swarm health metrics"""
        try:
            if not self.agents:
                return {'status': 'no_agents', 'score': 0.0}
            
            active_count = sum(1 for a in self.agents.values() if a.status == AgentStatus.ACTIVE)
            error_count = sum(1 for a in self.agents.values() if a.status == AgentStatus.ERROR)
            
            health_score = (active_count / len(self.agents)) * 0.7
            if error_count == 0:
                health_score += 0.3
            else:
                health_score -= (error_count / len(self.agents)) * 0.3
            
            return {
                'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy',
                'score': round(health_score, 2),
                'active_ratio': active_count / len(self.agents),
                'error_ratio': error_count / len(self.agents)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate swarm health: {e}")
            return {'status': 'error', 'score': 0.0}


# Global service instance
_coordination_service: Optional[MultiAgentCoordinationService] = None


def get_coordination_service() -> MultiAgentCoordinationService:
    """Get global coordination service instance"""
    global _coordination_service
    if _coordination_service is None:
        _coordination_service = MultiAgentCoordinationService()
    return _coordination_service