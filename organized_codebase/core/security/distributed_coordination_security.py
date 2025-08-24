"""
Llama-Agents Derived Distributed Coordination Security
Extracted from Llama-Agents distributed agent coordination patterns and security mechanisms
Enhanced for secure multi-agent coordination and consensus protocols
"""

import logging
import uuid
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
from .error_handler import SecurityError, ValidationError, security_error_handler


class AgentRole(Enum):
    """Agent roles in distributed system based on Llama-Agents patterns"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    OBSERVER = "observer"
    VALIDATOR = "validator"
    LEADER = "leader"
    FOLLOWER = "follower"


class CoordinationProtocol(Enum):
    """Coordination protocols for distributed agents"""
    CONSENSUS = "consensus"
    LEADER_ELECTION = "leader_election"
    DISTRIBUTED_LOCK = "distributed_lock"
    MESSAGE_PASSING = "message_passing"
    STATE_SYNC = "state_sync"


class SecurityLevel(Enum):
    """Security levels for coordination messages"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"


@dataclass
class AgentIdentity:
    """Secure agent identity for distributed coordination"""
    agent_id: str
    agent_name: str
    role: AgentRole
    public_key: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    security_clearance: SecurityLevel = SecurityLevel.AUTHENTICATED
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    trust_score: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_online(self) -> bool:
        """Check if agent is considered online"""
        return (datetime.utcnow() - self.last_heartbeat).total_seconds() < 60
    
    @property
    def is_trusted(self) -> bool:
        """Check if agent meets trust threshold"""
        return self.trust_score >= 70.0


@dataclass
class CoordinationMessage:
    """Secure coordination message between agents"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: str
    payload: Dict[str, Any]
    security_level: SecurityLevel = SecurityLevel.AUTHENTICATED
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: Optional[str] = None
    encryption_key_id: Optional[str] = None
    ttl_seconds: int = 300  # 5 minutes default TTL
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl_seconds
    
    @property
    def is_broadcast(self) -> bool:
        """Check if message is broadcast"""
        return self.receiver_id is None


@dataclass
class ConsensusProposal:
    """Consensus proposal for distributed decision making"""
    proposal_id: str
    proposer_id: str
    proposal_data: Dict[str, Any]
    required_votes: int
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    abstentions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    @property
    def total_votes(self) -> int:
        """Get total votes cast"""
        return len(self.votes_for) + len(self.votes_against) + len(self.abstentions)
    
    @property
    def is_passed(self) -> bool:
        """Check if proposal has passed"""
        return len(self.votes_for) >= self.required_votes
    
    @property
    def is_expired(self) -> bool:
        """Check if proposal has expired"""
        return self.expires_at and datetime.utcnow() > self.expires_at


class MessageCrypto:
    """Cryptographic operations for secure messaging"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.message_signatures: Dict[str, str] = {}
    
    def sign_message(self, message: CoordinationMessage, private_key: str = None) -> str:
        """Sign coordination message"""
        try:
            # Create message hash for signing
            message_data = {
                'message_id': message.message_id,
                'sender_id': message.sender_id,
                'receiver_id': message.receiver_id,
                'message_type': message.message_type,
                'payload': message.payload,
                'timestamp': message.timestamp.isoformat()
            }
            
            message_json = json.dumps(message_data, sort_keys=True)
            message_hash = hashlib.sha256(message_json.encode()).hexdigest()
            
            # Simple signature (in production, use proper cryptographic signing)
            signature_data = f"{private_key or 'default_key'}:{message_hash}"
            signature = hashlib.sha256(signature_data.encode()).hexdigest()
            
            # Store signature for verification
            self.message_signatures[message.message_id] = signature
            
            return signature
            
        except Exception as e:
            raise SecurityError(f"Message signing failed: {str(e)}", "MESSAGE_SIGN_001")
    
    def verify_signature(self, message: CoordinationMessage, public_key: str = None) -> bool:
        """Verify message signature"""
        try:
            if not message.signature:
                return False
            
            if message.message_id not in self.message_signatures:
                return False
            
            stored_signature = self.message_signatures[message.message_id]
            return message.signature == stored_signature
            
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False
    
    def encrypt_payload(self, payload: Dict[str, Any], encryption_key: str) -> str:
        """Encrypt message payload"""
        try:
            payload_json = json.dumps(payload)
            
            # Simple XOR encryption for demonstration
            # In production, use proper encryption like AES-GCM
            key_bytes = encryption_key.encode()
            payload_bytes = payload_json.encode()
            
            encrypted_bytes = bytearray()
            for i, byte in enumerate(payload_bytes):
                key_byte = key_bytes[i % len(key_bytes)]
                encrypted_bytes.append(byte ^ key_byte)
            
            # Convert to hex string for storage
            return encrypted_bytes.hex()
            
        except Exception as e:
            raise SecurityError(f"Payload encryption failed: {str(e)}", "ENCRYPT_001")
    
    def decrypt_payload(self, encrypted_payload: str, encryption_key: str) -> Dict[str, Any]:
        """Decrypt message payload"""
        try:
            # Convert from hex string
            encrypted_bytes = bytes.fromhex(encrypted_payload)
            
            # Simple XOR decryption
            key_bytes = encryption_key.encode()
            decrypted_bytes = bytearray()
            
            for i, byte in enumerate(encrypted_bytes):
                key_byte = key_bytes[i % len(key_bytes)]
                decrypted_bytes.append(byte ^ key_byte)
            
            payload_json = decrypted_bytes.decode()
            return json.loads(payload_json)
            
        except Exception as e:
            raise SecurityError(f"Payload decryption failed: {str(e)}", "DECRYPT_001")


class ConsensusManager:
    """Consensus protocol manager for distributed coordination"""
    
    def __init__(self, required_majority: float = 0.6):
        self.logger = logging.getLogger(__name__)
        self.required_majority = required_majority
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.completed_proposals: List[ConsensusProposal] = []
        self.max_completed_history = 1000
    
    def create_proposal(self, proposer_id: str, proposal_data: Dict[str, Any],
                       participant_count: int, timeout_minutes: int = 10) -> str:
        """Create new consensus proposal"""
        try:
            proposal_id = str(uuid.uuid4())
            required_votes = max(1, int(participant_count * self.required_majority))
            
            expires_at = datetime.utcnow() + timedelta(minutes=timeout_minutes)
            
            proposal = ConsensusProposal(
                proposal_id=proposal_id,
                proposer_id=proposer_id,
                proposal_data=proposal_data,
                required_votes=required_votes,
                expires_at=expires_at
            )
            
            self.active_proposals[proposal_id] = proposal
            
            self.logger.info(f"Created consensus proposal: {proposal_id} by {proposer_id}")
            return proposal_id
            
        except Exception as e:
            error = SecurityError(f"Failed to create consensus proposal: {str(e)}", "CONSENSUS_CREATE_001")
            security_error_handler.handle_error(error)
            raise error
    
    def cast_vote(self, proposal_id: str, voter_id: str, vote: str, 
                  voter_identity: AgentIdentity = None) -> bool:
        """Cast vote on consensus proposal"""
        try:
            if proposal_id not in self.active_proposals:
                raise ValidationError(f"Proposal not found: {proposal_id}")
            
            proposal = self.active_proposals[proposal_id]
            
            # Check if proposal has expired
            if proposal.is_expired:
                raise ValidationError(f"Proposal has expired: {proposal_id}")
            
            # Validate voter if identity provided
            if voter_identity and not voter_identity.is_trusted:
                raise ValidationError(f"Voter not trusted: {voter_id}")
            
            # Remove any previous votes by this voter
            proposal.votes_for.discard(voter_id)
            proposal.votes_against.discard(voter_id)
            proposal.abstentions.discard(voter_id)
            
            # Cast new vote
            if vote.lower() == "for":
                proposal.votes_for.add(voter_id)
            elif vote.lower() == "against":
                proposal.votes_against.add(voter_id)
            else:
                proposal.abstentions.add(voter_id)
            
            self.logger.info(f"Vote cast on proposal {proposal_id}: {voter_id} voted {vote}")
            
            # Check if proposal is decided
            if proposal.is_passed or proposal.is_expired:
                self._complete_proposal(proposal_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cast vote: {e}")
            return False
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of consensus proposal"""
        try:
            if proposal_id in self.active_proposals:
                proposal = self.active_proposals[proposal_id]
                status = "active"
            else:
                # Check completed proposals
                proposal = next((p for p in self.completed_proposals if p.proposal_id == proposal_id), None)
                if not proposal:
                    return None
                status = "completed"
            
            return {
                'proposal_id': proposal.proposal_id,
                'proposer_id': proposal.proposer_id,
                'status': status,
                'votes_for': len(proposal.votes_for),
                'votes_against': len(proposal.votes_against),
                'abstentions': len(proposal.abstentions),
                'total_votes': proposal.total_votes,
                'required_votes': proposal.required_votes,
                'is_passed': proposal.is_passed,
                'is_expired': proposal.is_expired,
                'created_at': proposal.created_at.isoformat(),
                'expires_at': proposal.expires_at.isoformat() if proposal.expires_at else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting proposal status: {e}")
            return None
    
    def _complete_proposal(self, proposal_id: str):
        """Move proposal from active to completed"""
        if proposal_id in self.active_proposals:
            proposal = self.active_proposals.pop(proposal_id)
            self.completed_proposals.append(proposal)
            
            # Limit completed proposals history
            if len(self.completed_proposals) > self.max_completed_history:
                self.completed_proposals = self.completed_proposals[-self.max_completed_history // 2:]
            
            result = "PASSED" if proposal.is_passed else "FAILED/EXPIRED"
            self.logger.info(f"Consensus proposal completed: {proposal_id} - {result}")


class LeaderElection:
    """Leader election protocol for distributed agents"""
    
    def __init__(self, election_timeout: int = 30):
        self.logger = logging.getLogger(__name__)
        self.election_timeout = election_timeout
        self.current_leader: Optional[str] = None
        self.election_in_progress = False
        self.election_votes: Dict[str, str] = {}  # voter_id -> candidate_id
        self.election_start_time: Optional[datetime] = None
        self.leadership_history: List[Dict[str, Any]] = []
    
    def start_election(self, candidates: List[AgentIdentity]) -> str:
        """Start leader election process"""
        try:
            if self.election_in_progress:
                raise ValidationError("Election already in progress")
            
            if not candidates:
                raise ValidationError("No candidates for leader election")
            
            # Filter trusted candidates
            trusted_candidates = [c for c in candidates if c.is_trusted and c.is_online]
            
            if not trusted_candidates:
                raise ValidationError("No trusted candidates available")
            
            election_id = str(uuid.uuid4())
            self.election_in_progress = True
            self.election_start_time = datetime.utcnow()
            self.election_votes.clear()
            
            self.logger.info(f"Started leader election: {election_id} with {len(trusted_candidates)} candidates")
            
            return election_id
            
        except Exception as e:
            error = SecurityError(f"Failed to start leader election: {str(e)}", "ELECTION_START_001")
            security_error_handler.handle_error(error)
            raise error
    
    def cast_election_vote(self, voter_id: str, candidate_id: str, 
                          voter_identity: AgentIdentity = None) -> bool:
        """Cast vote in leader election"""
        try:
            if not self.election_in_progress:
                raise ValidationError("No election in progress")
            
            # Check election timeout
            if self.election_start_time:
                elapsed = (datetime.utcnow() - self.election_start_time).total_seconds()
                if elapsed > self.election_timeout:
                    self._complete_election()
                    raise ValidationError("Election has timed out")
            
            # Validate voter
            if voter_identity and not voter_identity.is_trusted:
                raise ValidationError(f"Voter not trusted: {voter_id}")
            
            # Cast vote
            self.election_votes[voter_id] = candidate_id
            
            self.logger.info(f"Election vote cast: {voter_id} voted for {candidate_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cast election vote: {e}")
            return False
    
    def complete_election(self) -> Optional[str]:
        """Complete election and determine winner"""
        try:
            if not self.election_in_progress:
                return self.current_leader
            
            return self._complete_election()
            
        except Exception as e:
            self.logger.error(f"Failed to complete election: {e}")
            return None
    
    def _complete_election(self) -> Optional[str]:
        """Internal method to complete election"""
        self.election_in_progress = False
        
        if not self.election_votes:
            self.logger.warning("Election completed with no votes")
            return self.current_leader
        
        # Count votes
        vote_counts = defaultdict(int)
        for candidate_id in self.election_votes.values():
            vote_counts[candidate_id] += 1
        
        # Find winner (candidate with most votes)
        winner = max(vote_counts.items(), key=lambda x: x[1])
        new_leader = winner[0]
        vote_count = winner[1]
        
        # Record leadership change
        leadership_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'previous_leader': self.current_leader,
            'new_leader': new_leader,
            'vote_count': vote_count,
            'total_voters': len(self.election_votes)
        }
        
        self.leadership_history.append(leadership_record)
        
        # Update current leader
        old_leader = self.current_leader
        self.current_leader = new_leader
        
        self.logger.info(f"Leader election completed: {new_leader} elected with {vote_count} votes (was: {old_leader})")
        
        # Clear election state
        self.election_votes.clear()
        self.election_start_time = None
        
        return new_leader


class DistributedCoordinationSecurityManager:
    """Comprehensive distributed coordination security system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.registered_agents: Dict[str, AgentIdentity] = {}
        self.message_history: List[CoordinationMessage] = []
        self.crypto = MessageCrypto()
        self.consensus_manager = ConsensusManager()
        self.leader_election = LeaderElection()
        self.coordination_events: List[Dict[str, Any]] = []
        self.max_history = 10000
        self._lock = threading.RLock()
    
    def register_agent(self, agent_identity: AgentIdentity) -> bool:
        """Register agent for distributed coordination"""
        try:
            with self._lock:
                if not agent_identity.agent_id:
                    raise ValidationError("Agent ID cannot be empty")
                
                # Update existing agent or add new one
                self.registered_agents[agent_identity.agent_id] = agent_identity
                
                self._record_coordination_event(
                    "agent_registered", agent_identity.agent_id,
                    {"role": agent_identity.role.value, "trust_score": agent_identity.trust_score}
                )
                
                self.logger.info(f"Registered agent: {agent_identity.agent_id} ({agent_identity.role.value})")
                return True
                
        except Exception as e:
            error = SecurityError(f"Failed to register agent: {str(e)}", "AGENT_REG_001")
            security_error_handler.handle_error(error)
            return False
    
    def send_coordination_message(self, message: CoordinationMessage, 
                                 sender_identity: AgentIdentity = None) -> bool:
        """Send secure coordination message"""
        try:
            with self._lock:
                # Validate sender
                if sender_identity and sender_identity.agent_id != message.sender_id:
                    raise ValidationError("Sender identity mismatch")
                
                if message.sender_id not in self.registered_agents:
                    raise ValidationError(f"Sender not registered: {message.sender_id}")
                
                sender_agent = self.registered_agents[message.sender_id]
                if not sender_agent.is_trusted:
                    raise ValidationError(f"Sender not trusted: {message.sender_id}")
                
                # Validate receiver if not broadcast
                if not message.is_broadcast:
                    if message.receiver_id not in self.registered_agents:
                        raise ValidationError(f"Receiver not registered: {message.receiver_id}")
                
                # Sign message if required
                if message.security_level in [SecurityLevel.SIGNED, SecurityLevel.ENCRYPTED]:
                    signature = self.crypto.sign_message(message, sender_agent.public_key)
                    message.signature = signature
                
                # Encrypt payload if required
                if message.security_level == SecurityLevel.ENCRYPTED:
                    if not message.encryption_key_id:
                        message.encryption_key_id = f"key_{message.sender_id}_{int(time.time())}"
                    
                    # In production, use proper key management
                    encryption_key = f"enc_key_{message.sender_id}"
                    encrypted_payload = self.crypto.encrypt_payload(message.payload, encryption_key)
                    message.payload = {"encrypted_data": encrypted_payload}
                
                # Store message
                self.message_history.append(message)
                
                # Limit message history
                if len(self.message_history) > self.max_history:
                    self.message_history = self.message_history[-self.max_history // 2:]
                
                self._record_coordination_event(
                    "message_sent", message.sender_id,
                    {
                        "message_id": message.message_id,
                        "message_type": message.message_type,
                        "receiver_id": message.receiver_id,
                        "security_level": message.security_level.value
                    }
                )
                
                self.logger.info(f"Coordination message sent: {message.message_id} from {message.sender_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to send coordination message: {e}")
            return False
    
    def receive_coordination_message(self, message_id: str, receiver_id: str = None) -> Optional[CoordinationMessage]:
        """Receive and validate coordination message"""
        try:
            with self._lock:
                # Find message
                message = next((m for m in self.message_history if m.message_id == message_id), None)
                if not message:
                    return None
                
                # Check if message is for this receiver or is broadcast
                if not message.is_broadcast and message.receiver_id != receiver_id:
                    return None
                
                # Check expiration
                if message.is_expired:
                    self.logger.warning(f"Message expired: {message_id}")
                    return None
                
                # Verify signature if present
                if message.signature:
                    sender_agent = self.registered_agents.get(message.sender_id)
                    if not sender_agent or not self.crypto.verify_signature(message, sender_agent.public_key):
                        self.logger.warning(f"Message signature verification failed: {message_id}")
                        return None
                
                # Decrypt payload if needed
                if message.security_level == SecurityLevel.ENCRYPTED and "encrypted_data" in message.payload:
                    encryption_key = f"enc_key_{message.sender_id}"
                    try:
                        decrypted_payload = self.crypto.decrypt_payload(
                            message.payload["encrypted_data"], encryption_key
                        )
                        message.payload = decrypted_payload
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt message payload: {e}")
                        return None
                
                self._record_coordination_event(
                    "message_received", receiver_id or "broadcast",
                    {
                        "message_id": message_id,
                        "sender_id": message.sender_id,
                        "message_type": message.message_type
                    }
                )
                
                return message
                
        except Exception as e:
            self.logger.error(f"Failed to receive coordination message: {e}")
            return None
    
    def update_agent_heartbeat(self, agent_id: str, trust_score_delta: float = 0) -> bool:
        """Update agent heartbeat and trust score"""
        try:
            with self._lock:
                if agent_id not in self.registered_agents:
                    return False
                
                agent = self.registered_agents[agent_id]
                agent.last_heartbeat = datetime.utcnow()
                
                # Update trust score
                if trust_score_delta != 0:
                    agent.trust_score = max(0.0, min(100.0, agent.trust_score + trust_score_delta))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update agent heartbeat: {e}")
            return False
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination system status"""
        try:
            with self._lock:
                # Agent statistics
                total_agents = len(self.registered_agents)
                online_agents = sum(1 for agent in self.registered_agents.values() if agent.is_online)
                trusted_agents = sum(1 for agent in self.registered_agents.values() if agent.is_trusted)
                
                # Role distribution
                role_distribution = defaultdict(int)
                for agent in self.registered_agents.values():
                    role_distribution[agent.role.value] += 1
                
                # Message statistics
                recent_messages = len([m for m in self.message_history 
                                     if (datetime.utcnow() - m.timestamp).total_seconds() < 3600])
                
                # Security level distribution
                security_levels = defaultdict(int)
                for message in self.message_history[-100:]:
                    security_levels[message.security_level.value] += 1
                
                # Consensus statistics
                active_proposals = len(self.consensus_manager.active_proposals)
                completed_proposals = len(self.consensus_manager.completed_proposals)
                
                return {
                    'agent_stats': {
                        'total_agents': total_agents,
                        'online_agents': online_agents,
                        'trusted_agents': trusted_agents,
                        'online_rate_pct': (online_agents / max(total_agents, 1)) * 100,
                        'trust_rate_pct': (trusted_agents / max(total_agents, 1)) * 100,
                        'role_distribution': dict(role_distribution)
                    },
                    'message_stats': {
                        'total_messages': len(self.message_history),
                        'recent_messages_1h': recent_messages,
                        'security_level_distribution': dict(security_levels)
                    },
                    'consensus_stats': {
                        'active_proposals': active_proposals,
                        'completed_proposals': completed_proposals
                    },
                    'leadership': {
                        'current_leader': self.leader_election.current_leader,
                        'election_in_progress': self.leader_election.election_in_progress
                    },
                    'overall_health_score': self._calculate_coordination_health_score()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating coordination status: {e}")
            return {'error': str(e)}
    
    def _calculate_coordination_health_score(self) -> float:
        """Calculate overall coordination system health score"""
        try:
            score = 0.0
            
            if not self.registered_agents:
                return 0.0
            
            # Agent health (40 points)
            total_agents = len(self.registered_agents)
            online_agents = sum(1 for agent in self.registered_agents.values() if agent.is_online)
            trusted_agents = sum(1 for agent in self.registered_agents.values() if agent.is_trusted)
            
            score += (online_agents / total_agents) * 20
            score += (trusted_agents / total_agents) * 20
            
            # Message security (30 points)
            recent_messages = [m for m in self.message_history[-100:]]
            if recent_messages:
                secure_messages = sum(1 for m in recent_messages 
                                    if m.security_level in [SecurityLevel.ENCRYPTED, SecurityLevel.SIGNED])
                score += (secure_messages / len(recent_messages)) * 30
            else:
                score += 15  # No recent messages is neutral
            
            # Leadership stability (20 points)
            if self.leader_election.current_leader:
                score += 15
                if not self.leader_election.election_in_progress:
                    score += 5
            
            # System activity (10 points)
            recent_events = len([e for e in self.coordination_events[-100:]])
            if recent_events > 0:
                score += min(10, recent_events / 10)
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating coordination health score: {e}")
            return 0.0
    
    def _record_coordination_event(self, event_type: str, agent_id: str, context: Dict[str, Any]):
        """Record coordination system event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'agent_id': agent_id,
            'context': context
        }
        
        self.coordination_events.append(event)
        
        # Limit event history
        if len(self.coordination_events) > self.max_history:
            self.coordination_events = self.coordination_events[-self.max_history // 2:]


# Global distributed coordination security manager
distributed_coordination_security = DistributedCoordinationSecurityManager()


# Convenience functions
def register_secure_agent(agent_id: str, agent_name: str, role: AgentRole) -> bool:
    """Convenience function to register secure agent"""
    identity = AgentIdentity(
        agent_id=agent_id,
        agent_name=agent_name,
        role=role,
        security_clearance=SecurityLevel.AUTHENTICATED
    )
    return distributed_coordination_security.register_agent(identity)


def send_secure_message(sender_id: str, message_type: str, payload: Dict[str, Any],
                       receiver_id: str = None, security_level: SecurityLevel = SecurityLevel.AUTHENTICATED) -> bool:
    """Convenience function to send secure coordination message"""
    message = CoordinationMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=message_type,
        payload=payload,
        security_level=security_level
    )
    return distributed_coordination_security.send_coordination_message(message)