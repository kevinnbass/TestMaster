"""
Agent Coordination Protocols
Advanced protocols for multi-agent ML system coordination
"""Core Module - Split from agent_coordination_protocols.py"""


import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from collections import defaultdict, deque
from pathlib import Path
import uuid
from enum import Enum
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import socket
import ssl
import jwt


class ProtocolType(Enum):
    """Types of coordination protocols"""
    CONSENSUS = "consensus"
    LEADER_ELECTION = "leader_election"
    LOAD_BALANCING = "load_balancing"
    FAILOVER = "failover"
    SYNCHRONIZATION = "synchronization"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    SECURITY = "security"

class ConsensusAlgorithm(Enum):
    """Consensus algorithms for distributed coordination"""
    RAFT = "raft"
    PBFT = "pbft"  # Practical Byzantine Fault Tolerance
    PAXOS = "paxos"
    DPOS = "dpos"  # Delegated Proof of Stake

class SecurityLevel(Enum):
    """Security levels for agent communication"""
    NONE = "none"
    BASIC = "basic"
    ENCRYPTED = "encrypted"
    AUTHENTICATED = "authenticated"
    MUTUAL_TLS = "mutual_tls"

class AgentRole(Enum):
    """Agent roles in coordination protocols"""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    OBSERVER = "observer"
    COORDINATOR = "coordinator"

@dataclass
class CoordinationMessage:
    """Standardized coordination message format"""
    message_id: str
    protocol_type: ProtocolType
    source_agent_id: str
    target_agent_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    sequence_number: int
    priority: int
    ttl_seconds: int
    signature: Optional[str] = None
    encryption_key: Optional[str] = None

@dataclass
class AgentState:
    """Current state of an agent in coordination protocols"""
    agent_id: str
    agent_type: str
    role: AgentRole
    status: str  # active, inactive, failed, recovering
    last_heartbeat: datetime
    current_term: int  # For consensus protocols
    voted_for: Optional[str]  # For leader election
    log_index: int  # For state synchronization
    capabilities: Set[str]
    load_metrics: Dict[str, float]
    security_credentials: Dict[str, Any]
    protocol_participation: Set[ProtocolType]

@dataclass
class ConsensusProposal:
    """Proposal for distributed consensus"""
    proposal_id: str
    proposer_id: str
    term: int
    proposal_type: str
    proposal_data: Dict[str, Any]
    timestamp: datetime
    required_votes: int
    received_votes: Set[str]
    status: str  # proposed, accepted, rejected, committed

@dataclass
class LoadBalancingRule:
    """Load balancing rule definition"""
    rule_id: str
    name: str
    algorithm: str  # round_robin, weighted, least_connections, response_time
    target_agents: List[str]
    weights: Dict[str, float]
    health_check_enabled: bool
    failover_enabled: bool
    max_connections_per_agent: int
    conditions: Dict[str, Any]

class AgentCoordinationProtocols:
    """
    Agent Coordination Protocols Manager
    
    Provides comprehensive coordination protocols for multi-agent ML systems
    with security, fault tolerance, and performance optimization.
    """
    
    def __init__(self, agent_id: str, agent_type: str, config_path: str = "coordination_config.json"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config_path = config_path
        
        # Protocol state
        self.current_role = AgentRole.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log_index = 0
        self.leader_id = None
        self.cluster_members = {}
        self.active_protocols = set()
        
        # Message handling
        self.message_queue = deque(maxlen=10000)
        self.pending_proposals = {}
        self.message_handlers = {}
        self.sequence_counter = 0
        
        # Load balancing
        self.load_balancing_rules = {}
        self.agent_loads = {}
        self.routing_table = {}
        
        # Security
        self.security_keys = {}
        self.authentication_tokens = {}
        self.trusted_agents = set()
        
        # Configuration
        self.protocol_config = {
            "consensus": {
                "algorithm": ConsensusAlgorithm.RAFT,
                "election_timeout_ms": 5000,
                "heartbeat_interval_ms": 1000,
                "max_log_entries": 10000,
                "commit_batch_size": 100
            },
            "security": {
                "level": SecurityLevel.AUTHENTICATED,
                "token_expiry_hours": 24,
                "encryption_algorithm": "AES256",
                "signature_algorithm": "HMAC-SHA256",
                "mutual_auth_required": True
            },
            "load_balancing": {
                "default_algorithm": "weighted",
                "health_check_interval": 30,
                "failover_threshold": 3,
                "load_update_interval": 10,
                "max_retries": 3
            },
            "network": {
                "max_message_size": 1048576,  # 1MB
                "connection_timeout": 30,
                "retry_backoff_ms": 1000,
                "max_connections": 100
            },
            "monitoring": {
                "metrics_collection_interval": 30,
                "performance_tracking": True,
                "anomaly_detection": True,
                "protocol_optimization": True
            }
        }
        
        self.logger = logging.getLogger(__name__)
        self.coordination_active = True
        
        # Initialize protocols
        self._initialize_security()
        self._initialize_message_handlers()
        self._start_protocol_threads()
    
    def _initialize_security(self):
        """Initialize security components"""
        
        # Generate encryption key
        password = self.agent_id.encode()
        salt = b'agent_coordination_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.security_keys['encryption'] = Fernet(key)
        
        # Generate signing key
        self.security_keys['signing'] = hashlib.sha256(
            f"{self.agent_id}_signing_key".encode()
        ).hexdigest()
        
        # Generate authentication token
        self.authentication_tokens[self.agent_id] = self._generate_auth_token()
        
        self.logger.info("Security components initialized")
    
    def _generate_auth_token(self) -> str:
        """Generate JWT authentication token"""
        
        payload = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'role': self.current_role.value,
            'issued_at': datetime.now().timestamp(),
            'expires_at': (datetime.now() + timedelta(hours=24)).timestamp(),
            'capabilities': list(self.get_agent_capabilities())
        }
        
        return jwt.encode(payload, self.security_keys['signing'], algorithm='HS256')
    
    def _initialize_message_handlers(self):
        """Initialize message handlers for different protocol types"""
        
        self.message_handlers = {
            ProtocolType.CONSENSUS: {
                "vote_request": self._handle_vote_request,
                "vote_response": self._handle_vote_response,
                "append_entries": self._handle_append_entries,
                "append_response": self._handle_append_response
            },
            ProtocolType.LEADER_ELECTION: {
                "election_start": self._handle_election_start,
                "candidate_announcement": self._handle_candidate_announcement,
                "leader_announcement": self._handle_leader_announcement
            },
            ProtocolType.LOAD_BALANCING: {
                "load_update": self._handle_load_update,
                "routing_update": self._handle_routing_update,
                "health_check": self._handle_health_check
            },
            ProtocolType.FAILOVER: {
                "failover_start": self._handle_failover_start,
                "failover_complete": self._handle_failover_complete,
                "recovery_notification": self._handle_recovery_notification
            },
            ProtocolType.SYNCHRONIZATION: {
                "state_sync_request": self._handle_state_sync_request,
                "state_sync_response": self._handle_state_sync_response,
                "checkpoint_sync": self._handle_checkpoint_sync
            },
            ProtocolType.HEARTBEAT: {
                "heartbeat": self._handle_heartbeat,
                "heartbeat_response": self._handle_heartbeat_response
            },
            ProtocolType.DISCOVERY: {
                "agent_discovery": self._handle_agent_discovery,
                "capability_announcement": self._handle_capability_announcement,
                "topology_update": self._handle_topology_update
            },
            ProtocolType.SECURITY: {
                "auth_request": self._handle_auth_request,
                "auth_response": self._handle_auth_response,
                "key_exchange": self._handle_key_exchange
            }
        }
    
    def _start_protocol_threads(self):
        """Start background protocol threads"""
        
        # Message processing thread
        message_thread = threading.Thread(target=self._message_processing_loop, daemon=True)
        message_thread.start()
        
        # Consensus protocol thread
        consensus_thread = threading.Thread(target=self._consensus_protocol_loop, daemon=True)
        consensus_thread.start()
        
        # Heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        
        # Load monitoring thread
        load_thread = threading.Thread(target=self._load_monitoring_loop, daemon=True)
        load_thread.start()
        
        # Security monitoring thread
        security_thread = threading.Thread(target=self._security_monitoring_loop, daemon=True)
        security_thread.start()
        
        # Protocol optimization thread
        optimization_thread = threading.Thread(target=self._protocol_optimization_loop, daemon=True)
        optimization_thread.start()
    
    def send_coordination_message(self, target_agent_id: str, protocol_type: ProtocolType,
                                 message_type: str, payload: Dict[str, Any],
                                 priority: int = 5, ttl_seconds: int = 300) -> str:
        """Send a coordination message to another agent"""
        
        message_id = str(uuid.uuid4())
        self.sequence_counter += 1
        
        message = CoordinationMessage(
            message_id=message_id,
            protocol_type=protocol_type,
            source_agent_id=self.agent_id,
            target_agent_id=target_agent_id,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            sequence_number=self.sequence_counter,
            priority=priority,
            ttl_seconds=ttl_seconds
        )
        
        # Add security if required
        if self.protocol_config["security"]["level"] != SecurityLevel.NONE:
            message.signature = self._sign_message(message)
            
            if self.protocol_config["security"]["level"] in [SecurityLevel.ENCRYPTED, SecurityLevel.MUTUAL_TLS]:
                message.payload = self._encrypt_payload(message.payload)
                message.encryption_key = os.getenv('KEY')
        
        # Add to message queue for processing
        self.message_queue.append(message)
        
        self.logger.debug(f"Sent {protocol_type.value} message: {message_type} to {target_agent_id}")
        
        return message_id
    
    def start_leader_election(self) -> bool:
        """Start leader election process"""
        
        if self.current_role == AgentRole.LEADER:
            self.logger.warning("Already the leader, cannot start election")
            return False
        
        # Transition to candidate
        self.current_role = AgentRole.CANDIDATE
        self.current_term += 1
        self.voted_for = self.agent_id
        
        # Send vote requests to all cluster members
        vote_request_payload = {
            "term": self.current_term,
            "candidate_id": self.agent_id,
            "last_log_index": self.log_index,
            "last_log_term": self.current_term
        }
        
        votes_received = 1  # Vote for self
        required_votes = len(self.cluster_members) // 2 + 1
        
        for member_id in self.cluster_members:
            if member_id != self.agent_id:
                self.send_coordination_message(
                    target_agent_id=member_id,
                    protocol_type=ProtocolType.CONSENSUS,
                    message_type="vote_request",
                    payload=vote_request_payload,
                    priority=9
                )
        
        self.logger.info(f"Started leader election for term {self.current_term}")
        
        return True
    
    def propose_consensus_change(self, proposal_type: str, proposal_data: Dict[str, Any]) -> str:
        """Propose a change through consensus protocol"""
        
        if self.current_role != AgentRole.LEADER:
            self.logger.error("Only leader can propose consensus changes")
            return None
        
        proposal_id = str(uuid.uuid4())
        required_votes = len(self.cluster_members) // 2 + 1
        
        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            proposer_id=self.agent_id,
            term=self.current_term,
            proposal_type=proposal_type,
            proposal_data=proposal_data,
            timestamp=datetime.now(),
            required_votes=required_votes,
            received_votes={self.agent_id},  # Leader votes for own proposal
            status="proposed"
        )
        
        self.pending_proposals[proposal_id] = proposal
        
        # Send proposal to all followers
        proposal_payload = {
            "proposal_id": proposal_id,
            "term": self.current_term,
            "proposal_type": proposal_type,
            "proposal_data": proposal_data,
            "leader_id": self.agent_id
        }
        
        for member_id in self.cluster_members:
            if member_id != self.agent_id:
                self.send_coordination_message(
                    target_agent_id=member_id,
                    protocol_type=ProtocolType.CONSENSUS,
                    message_type="append_entries",
                    payload=proposal_payload,
                    priority=8
                )
        
        self.logger.info(f"Proposed consensus change: {proposal_type} ({proposal_id})")
        
        return proposal_id
    
    def create_load_balancing_rule(self, name: str, algorithm: str, 
                                  target_agents: List[str], weights: Dict[str, float] = None,
                                  conditions: Dict[str, Any] = None) -> str:
        """Create a new load balancing rule"""
        
        rule_id = str(uuid.uuid4())
        
        rule = LoadBalancingRule(
            rule_id=rule_id,
            name=name,
            algorithm=algorithm,
            target_agents=target_agents,
            weights=weights or {},
            health_check_enabled=True,
            failover_enabled=True,
            max_connections_per_agent=100,
            conditions=conditions or {}
        )
        
        self.load_balancing_rules[rule_id] = rule
        
        # Update routing table
        self._update_routing_table(rule)
        
        self.logger.info(f"Created load balancing rule: {name} ({rule_id})")
        
        return rule_id
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: Set[str],
                      endpoint: str, authentication_token: str = None) -> bool:
        """Register a new agent in the coordination cluster"""
        
        # Verify authentication if required
        if (self.protocol_config["security"]["level"] != SecurityLevel.NONE and
            not self._verify_auth_token(authentication_token)):
            self.logger.warning(f"Failed to authenticate agent: {agent_id}")
            return False
        
        # Create agent state
        agent_state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            role=AgentRole.FOLLOWER,
            status="active",
            last_heartbeat=datetime.now(),
            current_term=0,
            voted_for=None,
            log_index=0,
            capabilities=capabilities,
            load_metrics={},
            security_credentials={"token": authentication_token} if authentication_token else {},
            protocol_participation=set()
        )
        
        self.cluster_members[agent_id] = agent_state
        self.trusted_agents.add(agent_id)
        
        # Send discovery announcement to all members
        discovery_payload = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": list(capabilities),
            "endpoint": endpoint,
            "timestamp": datetime.now().isoformat()
        }
        
        for member_id in self.cluster_members:
            if member_id != agent_id:
                self.send_coordination_message(
                    target_agent_id=member_id,
                    protocol_type=ProtocolType.DISCOVERY,
                    message_type="agent_discovery",
                    payload=discovery_payload
                )
        
        self.logger.info(f"Registered agent: {agent_id} ({agent_type})")
        
        return True
    
    def get_agent_for_task(self, task_requirements: Dict[str, Any]) -> Optional[str]:
        """Get best agent for a task based on load balancing rules"""
        
        # Find matching load balancing rules
        matching_rules = []
        for rule in self.load_balancing_rules.values():
            if self._rule_matches_requirements(rule, task_requirements):
                matching_rules.append(rule)
        
        if not matching_rules:
            # Use default round-robin selection
            active_agents = [aid for aid, state in self.cluster_members.items() 
                           if state.status == "active"]
            if not active_agents:
                return None
            
            return active_agents[self.sequence_counter % len(active_agents)]
        
        # Use the first matching rule
        rule = matching_rules[0]
        return self._select_agent_by_algorithm(rule, task_requirements)
    
    def _select_agent_by_algorithm(self, rule: LoadBalancingRule, 
                                  requirements: Dict[str, Any]) -> Optional[str]:
        """Select agent based on load balancing algorithm"""
        
        eligible_agents = [aid for aid in rule.target_agents 
                          if aid in self.cluster_members and 
                          self.cluster_members[aid].status == "active"]
        
        if not eligible_agents:
            return None
        
        if rule.algorithm == "round_robin":
            return eligible_agents[self.sequence_counter % len(eligible_agents)]
        
        elif rule.algorithm == "weighted":
            # Select based on weights
            total_weight = sum(rule.weights.get(aid, 1.0) for aid in eligible_agents)
            if total_weight == 0:
                return eligible_agents[0]
            
            import random
            r = random.random() * total_weight
            cumulative_weight = 0
            
            for agent_id in eligible_agents:
                cumulative_weight += rule.weights.get(agent_id, 1.0)
                if r <= cumulative_weight:
                    return agent_id
            
            return eligible_agents[-1]
        
        elif rule.algorithm == "least_connections":
            # Select agent with least current load
            min_load = float('inf')
            best_agent = None
            
            for agent_id in eligible_agents:
                load = self.agent_loads.get(agent_id, {}).get("active_connections", 0)
                if load < min_load:
                    min_load = load
                    best_agent = agent_id
            
            return best_agent
        
        elif rule.algorithm == "response_time":
            # Select agent with best response time
            best_response_time = float('inf')
            best_agent = None
            
            for agent_id in eligible_agents:
                response_time = self.agent_loads.get(agent_id, {}).get("avg_response_time", 1000)
                if response_time < best_response_time:
                    best_response_time = response_time
                    best_agent = agent_id
            
            return best_agent
        
        return eligible_agents[0]  # Fallback
    
    def _message_processing_loop(self):
        """Main message processing loop"""
        while self.coordination_active:
            try:
                if self.message_queue:
                    message = self.message_queue.popleft()
                    self._process_coordination_message(message)
                else:
                    time.sleep(0.01)  # Brief pause when no messages
                    
            except Exception as e:
                self.logger.error(f"Error in message processing: {e}")
                time.sleep(0.1)
    
    def _process_coordination_message(self, message: CoordinationMessage):
        """Process an individual coordination message"""
        
        # Check message TTL
        age = (datetime.now() - message.timestamp).total_seconds()
        if age > message.ttl_seconds:
            self.logger.warning(f"Dropping expired message: {message.message_id}")
            return
        
        # Verify security if required
        if self.protocol_config["security"]["level"] != SecurityLevel.NONE:
            if not self._verify_message_signature(message):
                self.logger.warning(f"Message signature verification failed: {message.message_id}")
                return
            
            if message.encryption_key:
                message.payload = self._decrypt_payload(message.payload)
        
        # Route to appropriate handler
        protocol_handlers = self.message_handlers.get(message.protocol_type, {})
        handler = protocol_handlers.get(message.message_type)
        
        if handler:
            try:
                handler(message)
            except Exception as e:
                self.logger.error(f"Message handler error: {e}")
        else:
            self.logger.warning(f"No handler for {message.protocol_type.value}:{message.message_type}")
    
    def _consensus_protocol_loop(self):
        """Consensus protocol management loop"""
        while self.coordination_active:
            try:
                if self.current_role == AgentRole.LEADER:
                    self._send_heartbeats()
                    self._process_pending_proposals()
                elif self.current_role == AgentRole.CANDIDATE:
                    self._check_election_timeout()
                elif self.current_role == AgentRole.FOLLOWER:
                    self._check_leader_timeout()
                
                time.sleep(0.5)  # Run twice per second
