"""
Swarms Derived Distributed Agent Registry Security Module
Extracted from Swarms agent registry patterns for secure distributed agent management
Enhanced for thread-safe operations and Byzantine fault tolerance
"""

import uuid
import time
import json
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from .error_handler import SecurityError, security_error_handler


class AgentStatus(Enum):
    """Agent status in distributed registry"""
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    COMPROMISED = "compromised"


class RegistryEventType(Enum):
    """Registry event types for audit trail"""
    AGENT_REGISTERED = "agent_registered"
    AGENT_ACTIVATED = "agent_activated"
    AGENT_DEACTIVATED = "agent_deactivated"
    AGENT_SUSPENDED = "agent_suspended"
    SECURITY_VIOLATION = "security_violation"
    HEARTBEAT_FAILED = "heartbeat_failed"


@dataclass
class DistributedAgent:
    """Secure agent identity for distributed systems"""
    agent_id: str
    agent_name: str
    agent_type: str
    host_address: str
    port: int
    public_key: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.REGISTERED
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: Optional[datetime] = None
    trust_score: float = 1.0
    security_violations: int = 0
    
    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())
        
        # Validate required fields
        if not self.agent_name or len(self.agent_name) < 3:
            raise SecurityError("Agent name must be at least 3 characters", "AGENT_REG_001")
        
        if not self.host_address:
            raise SecurityError("Agent host address is required", "AGENT_REG_002")
        
        if not 1 <= self.port <= 65535:
            raise SecurityError("Agent port must be between 1 and 65535", "AGENT_REG_003")
    
    @property
    def is_healthy(self) -> bool:
        """Check if agent is healthy based on heartbeat"""
        if not self.last_heartbeat:
            return False
        
        # Agent is considered unhealthy after 5 minutes without heartbeat
        threshold = datetime.utcnow() - timedelta(minutes=5)
        return self.last_heartbeat > threshold
    
    @property
    def endpoint(self) -> str:
        """Get agent network endpoint"""
        return f"{self.host_address}:{self.port}"
    
    def calculate_agent_hash(self) -> str:
        """Calculate unique hash for agent identity verification"""
        identity_data = {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'host_address': self.host_address,
            'port': self.port,
            'public_key': self.public_key
        }
        
        identity_str = json.dumps(identity_data, sort_keys=True)
        return hashlib.sha256(identity_str.encode()).hexdigest()


@dataclass
class RegistryEvent:
    """Registry event for audit and monitoring"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: RegistryEventType = RegistryEventType.AGENT_REGISTERED
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"


class DistributedAgentRegistry:
    """Thread-safe distributed agent registry with security controls"""
    
    def __init__(self, max_agents: int = 1000):
        self.max_agents = max_agents
        self.agents: Dict[str, DistributedAgent] = {}
        self.agent_hashes: Dict[str, str] = {}  # agent_id -> hash mapping
        self.host_agents: Dict[str, Set[str]] = {}  # host -> agent_ids mapping
        self.events: List[RegistryEvent] = []
        
        # Thread safety
        self.registry_lock = threading.RLock()
        self.event_lock = threading.Lock()
        
        # Security settings
        self.max_agents_per_host = 10
        self.min_trust_score = 0.3
        self.heartbeat_interval = 60  # seconds
        
        self.logger = logging.getLogger(__name__)
        
        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self.start_monitoring()
    
    def register_agent(self, agent: DistributedAgent) -> bool:
        """Register new agent with security validation"""
        try:
            with self.registry_lock:
                # Check registry capacity
                if len(self.agents) >= self.max_agents:
                    raise SecurityError(f"Registry at maximum capacity: {self.max_agents}", "REG_CAP_001")
                
                # Check if agent already exists
                if agent.agent_id in self.agents:
                    existing = self.agents[agent.agent_id]
                    if existing.status != AgentStatus.INACTIVE:
                        raise SecurityError(f"Agent {agent.agent_id} already registered", "REG_DUP_001")
                
                # Check host capacity
                host_key = f"{agent.host_address}:{agent.port}"
                if host_key not in self.host_agents:
                    self.host_agents[host_key] = set()
                
                if len(self.host_agents[host_key]) >= self.max_agents_per_host:
                    raise SecurityError(f"Too many agents from host {host_key}", "REG_HOST_001")
                
                # Validate agent identity hash
                agent_hash = agent.calculate_agent_hash()
                for existing_id, existing_hash in self.agent_hashes.items():
                    if existing_hash == agent_hash and existing_id != agent.agent_id:
                        raise SecurityError("Agent identity hash collision detected", "REG_HASH_001")
                
                # Register the agent
                agent.status = AgentStatus.REGISTERED
                agent.last_heartbeat = datetime.utcnow()
                
                self.agents[agent.agent_id] = agent
                self.agent_hashes[agent.agent_id] = agent_hash
                self.host_agents[host_key].add(agent.agent_id)
                
                # Log registration event
                self._log_event(RegistryEvent(
                    event_type=RegistryEventType.AGENT_REGISTERED,
                    agent_id=agent.agent_id,
                    details={'endpoint': agent.endpoint, 'type': agent.agent_type}
                ))
                
                self.logger.info(f"Agent registered: {agent.agent_id} at {agent.endpoint}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Agent registration failed: {str(e)}", "REG_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def activate_agent(self, agent_id: str) -> bool:
        """Activate registered agent with security checks"""
        try:
            with self.registry_lock:
                if agent_id not in self.agents:
                    raise SecurityError(f"Agent {agent_id} not found", "REG_ACT_001")
                
                agent = self.agents[agent_id]
                
                # Check trust score
                if agent.trust_score < self.min_trust_score:
                    raise SecurityError(f"Agent {agent_id} trust score too low: {agent.trust_score}", "REG_TRUST_001")
                
                # Check security violations
                if agent.security_violations >= 3:
                    raise SecurityError(f"Agent {agent_id} has too many security violations", "REG_VIOL_001")
                
                agent.status = AgentStatus.ACTIVE
                agent.last_heartbeat = datetime.utcnow()
                
                self._log_event(RegistryEvent(
                    event_type=RegistryEventType.AGENT_ACTIVATED,
                    agent_id=agent_id,
                    details={'trust_score': agent.trust_score}
                ))
                
                self.logger.info(f"Agent activated: {agent_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Agent activation failed: {str(e)}", "REG_ACT_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def heartbeat(self, agent_id: str, health_data: Optional[Dict[str, Any]] = None) -> bool:
        """Process agent heartbeat with health validation"""
        try:
            with self.registry_lock:
                if agent_id not in self.agents:
                    return False
                
                agent = self.agents[agent_id]
                agent.last_heartbeat = datetime.utcnow()
                
                # Process health data if provided
                if health_data:
                    # Validate health metrics
                    cpu_usage = health_data.get('cpu_usage', 0)
                    memory_usage = health_data.get('memory_usage', 0)
                    
                    # Check for resource exhaustion (security concern)
                    if cpu_usage > 95 or memory_usage > 95:
                        self.logger.warning(f"Agent {agent_id} showing resource exhaustion")
                        agent.trust_score = max(0.1, agent.trust_score - 0.1)
                    
                    # Update metadata with health data
                    agent.metadata['health'] = health_data
                    agent.metadata['last_health_check'] = datetime.utcnow().isoformat()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Heartbeat processing failed for {agent_id}: {e}")
            return False
    
    def suspend_agent(self, agent_id: str, reason: str) -> bool:
        """Suspend agent for security reasons"""
        try:
            with self.registry_lock:
                if agent_id not in self.agents:
                    return False
                
                agent = self.agents[agent_id]
                agent.status = AgentStatus.SUSPENDED
                agent.security_violations += 1
                agent.trust_score = max(0.0, agent.trust_score - 0.2)
                
                self._log_event(RegistryEvent(
                    event_type=RegistryEventType.AGENT_SUSPENDED,
                    agent_id=agent_id,
                    details={'reason': reason, 'violations': agent.security_violations},
                    severity="warning"
                ))
                
                self.logger.warning(f"Agent suspended: {agent_id}, reason: {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"Agent suspension failed: {e}")
            return False
    
    def get_active_agents(self) -> List[DistributedAgent]:
        """Get list of active, healthy agents"""
        with self.registry_lock:
            return [
                agent for agent in self.agents.values()
                if agent.status == AgentStatus.ACTIVE and agent.is_healthy
            ]
    
    def get_agent_by_endpoint(self, host: str, port: int) -> Optional[DistributedAgent]:
        """Find agent by network endpoint"""
        with self.registry_lock:
            for agent in self.agents.values():
                if agent.host_address == host and agent.port == port:
                    return agent
            return None
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics for monitoring"""
        with self.registry_lock:
            status_counts = {}
            for status in AgentStatus:
                status_counts[status.value] = sum(
                    1 for agent in self.agents.values() 
                    if agent.status == status
                )
            
            healthy_agents = sum(
                1 for agent in self.agents.values() 
                if agent.is_healthy and agent.status == AgentStatus.ACTIVE
            )
            
            return {
                'total_agents': len(self.agents),
                'active_agents': status_counts.get('active', 0),
                'healthy_agents': healthy_agents,
                'status_breakdown': status_counts,
                'total_hosts': len(self.host_agents),
                'average_trust_score': sum(a.trust_score for a in self.agents.values()) / max(1, len(self.agents))
            }
    
    def start_monitoring(self):
        """Start background monitoring for agent health"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_agents, daemon=True)
            self._monitor_thread.start()
            self.logger.info("Agent monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Agent monitoring stopped")
    
    def _monitor_agents(self):
        """Background monitoring for unhealthy agents"""
        while self._monitoring_active:
            try:
                with self.registry_lock:
                    current_time = datetime.utcnow()
                    
                    for agent in list(self.agents.values()):
                        if agent.status == AgentStatus.ACTIVE and not agent.is_healthy:
                            # Mark as inactive and log event
                            agent.status = AgentStatus.INACTIVE
                            
                            self._log_event(RegistryEvent(
                                event_type=RegistryEventType.HEARTBEAT_FAILED,
                                agent_id=agent.agent_id,
                                details={'last_heartbeat': agent.last_heartbeat.isoformat() if agent.last_heartbeat else None},
                                severity="warning"
                            ))
                            
                            self.logger.warning(f"Agent {agent.agent_id} marked inactive due to failed heartbeat")
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _log_event(self, event: RegistryEvent):
        """Log registry event"""
        with self.event_lock:
            self.events.append(event)
            
            # Keep only recent events (last 10000)
            if len(self.events) > 10000:
                self.events = self.events[-10000:]


# Global distributed agent registry
distributed_agent_registry = DistributedAgentRegistry()


def register_distributed_agent(agent_name: str, agent_type: str, host_address: str, 
                              port: int, capabilities: Optional[List[str]] = None) -> Optional[str]:
    """Convenience function to register a distributed agent"""
    try:
        agent = DistributedAgent(
            agent_id=str(uuid.uuid4()),
            agent_name=agent_name,
            agent_type=agent_type,
            host_address=host_address,
            port=port,
            capabilities=capabilities or []
        )
        
        if distributed_agent_registry.register_agent(agent):
            return agent.agent_id
        return None
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent registration failed: {e}")
        return None


def send_agent_heartbeat(agent_id: str, health_data: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to send agent heartbeat"""
    return distributed_agent_registry.heartbeat(agent_id, health_data)