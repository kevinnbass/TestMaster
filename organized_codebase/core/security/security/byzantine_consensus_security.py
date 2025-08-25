"""
Swarms Derived Byzantine Fault Tolerant Consensus Security Module
Extracted from Swarms majority voting patterns for secure distributed decision-making
Enhanced for Byzantine fault tolerance and Sybil attack resistance
"""

import uuid
import time
import json
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from collections import Counter
from .error_handler import SecurityError, security_error_handler


class ConsensusAlgorithm(Enum):
    """Supported consensus algorithms"""
    MAJORITY_VOTING = "majority_voting"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    WEIGHTED_CONSENSUS = "weighted_consensus"
    PROOF_OF_STAKE = "proof_of_stake"


class VoteStatus(Enum):
    """Vote processing status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ThreatType(Enum):
    """Security threat types in consensus"""
    SYBIL_ATTACK = "sybil_attack"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    COORDINATION_ATTACK = "coordination_attack"
    ECLIPSE_ATTACK = "eclipse_attack"
    MAJORITY_ATTACK = "majority_attack"


@dataclass
class ConsensusAgent:
    """Secure agent participant in consensus protocol"""
    agent_id: str
    agent_type: str
    trust_score: float = 1.0
    stake_weight: float = 1.0
    vote_history: List[str] = field(default_factory=list)
    reputation_score: float = 1.0
    last_active: datetime = field(default_factory=datetime.utcnow)
    security_violations: int = 0
    public_key: Optional[str] = None
    
    def __post_init__(self):
        # Validate trust and reputation scores
        if not 0.0 <= self.trust_score <= 1.0:
            raise SecurityError("Trust score must be between 0.0 and 1.0", "CONS_AGENT_001")
        
        if not 0.0 <= self.reputation_score <= 1.0:
            raise SecurityError("Reputation score must be between 0.0 and 1.0", "CONS_AGENT_002")
    
    @property
    def effective_weight(self) -> float:
        """Calculate effective voting weight based on trust and stake"""
        base_weight = self.stake_weight * self.trust_score * self.reputation_score
        
        # Penalize agents with security violations
        violation_penalty = max(0.1, 1.0 - (self.security_violations * 0.1))
        
        return base_weight * violation_penalty
    
    @property
    def is_trustworthy(self) -> bool:
        """Check if agent is trustworthy for consensus participation"""
        return (
            self.trust_score >= 0.5 and
            self.reputation_score >= 0.3 and
            self.security_violations < 3
        )


@dataclass
class ConsensusProposal:
    """Secure consensus proposal with metadata"""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str = ""
    proposer_id: str = ""
    proposed_at: datetime = field(default_factory=datetime.utcnow)
    deadline: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=10))
    minimum_participants: int = 3
    required_consensus: float = 0.67  # 67% consensus required
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.5 <= self.required_consensus <= 1.0:
            raise SecurityError("Required consensus must be between 50% and 100%", "CONS_PROP_001")
        
        if self.minimum_participants < 1:
            raise SecurityError("Minimum participants must be at least 1", "CONS_PROP_002")
    
    @property
    def is_expired(self) -> bool:
        """Check if proposal has expired"""
        return datetime.utcnow() > self.deadline
    
    def calculate_proposal_hash(self) -> str:
        """Calculate cryptographic hash of proposal for integrity verification"""
        proposal_data = {
            'proposal_id': self.proposal_id,
            'task_description': self.task_description,
            'proposer_id': self.proposer_id,
            'proposed_at': self.proposed_at.isoformat(),
            'required_consensus': self.required_consensus
        }
        
        proposal_str = json.dumps(proposal_data, sort_keys=True)
        return hashlib.sha256(proposal_str.encode()).hexdigest()


@dataclass
class SecureVote:
    """Secure vote with cryptographic integrity"""
    vote_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposal_id: str = ""
    voter_id: str = ""
    vote_content: Any = None
    vote_weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: Optional[str] = None
    
    def calculate_vote_hash(self) -> str:
        """Calculate cryptographic hash of vote for integrity verification"""
        vote_data = {
            'vote_id': self.vote_id,
            'proposal_id': self.proposal_id,
            'voter_id': self.voter_id,
            'vote_content': json.dumps(self.vote_content) if self.vote_content is not None else None,
            'timestamp': self.timestamp.isoformat()
        }
        
        vote_str = json.dumps(vote_data, sort_keys=True)
        return hashlib.sha256(vote_str.encode()).hexdigest()


@dataclass
class ConsensusResult:
    """Results of consensus process with security analysis"""
    proposal_id: str
    consensus_reached: bool
    final_decision: Any
    participant_count: int
    consensus_percentage: float
    execution_time: float
    security_threats_detected: List[ThreatType] = field(default_factory=list)
    minority_opinions: List[Any] = field(default_factory=list)
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for logging"""
        return {
            'proposal_id': self.proposal_id,
            'consensus_reached': self.consensus_reached,
            'final_decision': self.final_decision,
            'participant_count': self.participant_count,
            'consensus_percentage': self.consensus_percentage,
            'execution_time': self.execution_time,
            'security_threats': [threat.value for threat in self.security_threats_detected],
            'confidence_score': self.confidence_score
        }


class ThreatDetector:
    """Security threat detection for consensus protocols"""
    
    def __init__(self):
        self.suspicious_patterns: List[Callable] = [
            self._detect_sybil_attack,
            self._detect_byzantine_behavior,
            self._detect_coordination_attack,
            self._detect_majority_attack
        ]
        self.logger = logging.getLogger(__name__)
    
    def analyze_consensus_security(self, votes: List[SecureVote], 
                                 agents: Dict[str, ConsensusAgent]) -> List[ThreatType]:
        """Analyze consensus process for security threats"""
        detected_threats = []
        
        for pattern_detector in self.suspicious_patterns:
            try:
                threats = pattern_detector(votes, agents)
                detected_threats.extend(threats)
            except Exception as e:
                self.logger.error(f"Threat detection failed: {e}")
        
        return list(set(detected_threats))  # Remove duplicates
    
    def _detect_sybil_attack(self, votes: List[SecureVote], 
                           agents: Dict[str, ConsensusAgent]) -> List[ThreatType]:
        """Detect potential Sybil attacks through voting pattern analysis"""
        threats = []
        
        # Group votes by content similarity
        vote_groups = {}
        for vote in votes:
            vote_str = json.dumps(vote.vote_content) if vote.vote_content else "None"
            vote_hash = hashlib.md5(vote_str.encode()).hexdigest()[:8]
            
            if vote_hash not in vote_groups:
                vote_groups[vote_hash] = []
            vote_groups[vote_hash].append(vote)
        
        # Check for suspicious identical voting patterns
        for vote_group in vote_groups.values():
            if len(vote_group) >= 3:  # 3 or more identical votes
                # Check if voters have similar characteristics (potential Sybil nodes)
                voter_ids = [vote.voter_id for vote in vote_group]
                voter_trust_scores = [agents.get(voter_id, ConsensusAgent("", "")).trust_score for voter_id in voter_ids]
                
                # If multiple new/low-trust agents vote identically, flag as potential Sybil attack
                if sum(1 for score in voter_trust_scores if score < 0.3) >= 2:
                    threats.append(ThreatType.SYBIL_ATTACK)
                    break
        
        return threats
    
    def _detect_byzantine_behavior(self, votes: List[SecureVote], 
                                 agents: Dict[str, ConsensusAgent]) -> List[ThreatType]:
        """Detect Byzantine behavior through inconsistent voting patterns"""
        threats = []
        
        # Analyze vote timing patterns for Byzantine behavior
        vote_times = [vote.timestamp for vote in votes]
        if len(vote_times) >= 3:
            # Check for suspicious simultaneous voting (coordination)
            time_deltas = []
            sorted_times = sorted(vote_times)
            
            for i in range(1, len(sorted_times)):
                delta = (sorted_times[i] - sorted_times[i-1]).total_seconds()
                time_deltas.append(delta)
            
            # If many votes occur within very short time windows, flag as suspicious
            simultaneous_votes = sum(1 for delta in time_deltas if delta < 1.0)  # Within 1 second
            if simultaneous_votes >= 3:
                threats.append(ThreatType.BYZANTINE_BEHAVIOR)
        
        return threats
    
    def _detect_coordination_attack(self, votes: List[SecureVote], 
                                  agents: Dict[str, ConsensusAgent]) -> List[ThreatType]:
        """Detect coordinated attacks through voting correlation"""
        threats = []
        
        # Check for perfect vote correlation among low-reputation agents
        low_rep_voters = [
            vote for vote in votes 
            if agents.get(vote.voter_id, ConsensusAgent("", "")).reputation_score < 0.5
        ]
        
        if len(low_rep_voters) >= 3:
            # Check if low-reputation agents all vote the same way
            vote_contents = [json.dumps(vote.vote_content) for vote in low_rep_voters]
            if len(set(vote_contents)) == 1:  # All identical votes
                threats.append(ThreatType.COORDINATION_ATTACK)
        
        return threats
    
    def _detect_majority_attack(self, votes: List[SecureVote], 
                              agents: Dict[str, ConsensusAgent]) -> List[ThreatType]:
        """Detect potential majority attacks"""
        threats = []
        
        if len(votes) >= 5:
            # Check if a large percentage of votes come from new/untrusted agents
            new_agent_votes = sum(
                1 for vote in votes 
                if agents.get(vote.voter_id, ConsensusAgent("", "")).trust_score < 0.6
            )
            
            new_agent_percentage = new_agent_votes / len(votes)
            if new_agent_percentage > 0.6:  # More than 60% from new/untrusted agents
                threats.append(ThreatType.MAJORITY_ATTACK)
        
        return threats


class ByzantineConsensusSecurityManager:
    """Secure Byzantine fault-tolerant consensus manager"""
    
    def __init__(self, max_workers: int = 8):
        self.agents: Dict[str, ConsensusAgent] = {}
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.votes: Dict[str, List[SecureVote]] = {}  # proposal_id -> votes
        self.results: Dict[str, ConsensusResult] = {}
        
        # Security components
        self.threat_detector = ThreatDetector()
        self.consensus_lock = threading.RLock()
        
        # Performance settings
        self.max_workers = max_workers
        self.vote_timeout = 300  # 5 minutes
        
        self.logger = logging.getLogger(__name__)
    
    def register_consensus_agent(self, agent: ConsensusAgent) -> bool:
        """Register agent for consensus participation"""
        try:
            with self.consensus_lock:
                if agent.agent_id in self.agents:
                    existing = self.agents[agent.agent_id]
                    # Allow re-registration with higher trust score
                    if agent.trust_score <= existing.trust_score:
                        return False
                
                self.agents[agent.agent_id] = agent
                self.logger.info(f"Consensus agent registered: {agent.agent_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Consensus agent registration failed: {str(e)}", "CONS_REG_001")
            security_error_handler.handle_error(error)
            return False
    
    def submit_proposal(self, proposal: ConsensusProposal) -> bool:
        """Submit proposal for consensus decision"""
        try:
            with self.consensus_lock:
                if proposal.proposer_id not in self.agents:
                    raise SecurityError("Proposer not registered", "CONS_SUBMIT_001")
                
                if not self.agents[proposal.proposer_id].is_trustworthy:
                    raise SecurityError("Proposer not trustworthy", "CONS_SUBMIT_002")
                
                self.proposals[proposal.proposal_id] = proposal
                self.votes[proposal.proposal_id] = []
                
                self.logger.info(f"Consensus proposal submitted: {proposal.proposal_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Proposal submission failed: {str(e)}", "CONS_SUBMIT_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def cast_vote(self, vote: SecureVote) -> bool:
        """Cast secure vote for proposal"""
        try:
            with self.consensus_lock:
                # Validate voter
                if vote.voter_id not in self.agents:
                    raise SecurityError("Voter not registered", "CONS_VOTE_001")
                
                agent = self.agents[vote.voter_id]
                if not agent.is_trustworthy:
                    raise SecurityError("Voter not trustworthy", "CONS_VOTE_002")
                
                # Validate proposal
                if vote.proposal_id not in self.proposals:
                    raise SecurityError("Proposal not found", "CONS_VOTE_003")
                
                proposal = self.proposals[vote.proposal_id]
                if proposal.is_expired:
                    raise SecurityError("Proposal has expired", "CONS_VOTE_004")
                
                # Check for duplicate votes
                existing_votes = self.votes[vote.proposal_id]
                if any(v.voter_id == vote.voter_id for v in existing_votes):
                    raise SecurityError("Duplicate vote detected", "CONS_VOTE_005")
                
                # Set vote weight based on agent's effective weight
                vote.vote_weight = agent.effective_weight
                
                # Add vote to proposal
                self.votes[vote.proposal_id].append(vote)
                
                # Update agent activity
                agent.last_active = datetime.utcnow()
                agent.vote_history.append(vote.proposal_id)
                
                self.logger.info(f"Vote cast: {vote.vote_id} for proposal {vote.proposal_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Vote casting failed: {str(e)}", "CONS_VOTE_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def execute_consensus(self, proposal_id: str, 
                         algorithm: ConsensusAlgorithm = ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT) -> ConsensusResult:
        """Execute consensus process with security analysis"""
        start_time = time.time()
        
        try:
            with self.consensus_lock:
                if proposal_id not in self.proposals:
                    raise SecurityError("Proposal not found", "CONS_EXEC_001")
                
                proposal = self.proposals[proposal_id]
                votes = self.votes.get(proposal_id, [])
                
                # Check minimum participation
                if len(votes) < proposal.minimum_participants:
                    return ConsensusResult(
                        proposal_id=proposal_id,
                        consensus_reached=False,
                        final_decision=None,
                        participant_count=len(votes),
                        consensus_percentage=0.0,
                        execution_time=time.time() - start_time
                    )
                
                # Security analysis
                threats = self.threat_detector.analyze_consensus_security(votes, self.agents)
                
                # Execute consensus algorithm
                if algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
                    result = self._byzantine_consensus(proposal, votes)
                elif algorithm == ConsensusAlgorithm.WEIGHTED_CONSENSUS:
                    result = self._weighted_consensus(proposal, votes)
                else:
                    result = self._majority_voting_consensus(proposal, votes)
                
                result.security_threats_detected = threats
                result.execution_time = time.time() - start_time
                
                # Store result
                self.results[proposal_id] = result
                
                self.logger.info(f"Consensus executed for {proposal_id}: {result.consensus_reached}")
                return result
                
        except Exception as e:
            error = SecurityError(f"Consensus execution failed: {str(e)}", "CONS_EXEC_FAIL_001")
            security_error_handler.handle_error(error)
            
            return ConsensusResult(
                proposal_id=proposal_id,
                consensus_reached=False,
                final_decision=None,
                participant_count=0,
                consensus_percentage=0.0,
                execution_time=time.time() - start_time
            )
    
    def _byzantine_consensus(self, proposal: ConsensusProposal, votes: List[SecureVote]) -> ConsensusResult:
        """Execute Byzantine fault-tolerant consensus"""
        # Filter out votes from compromised agents
        trusted_votes = [
            vote for vote in votes 
            if self.agents[vote.voter_id].is_trustworthy
        ]
        
        # Weight votes by trust and reputation
        weighted_votes = {}
        total_weight = 0
        
        for vote in trusted_votes:
            vote_content_str = json.dumps(vote.vote_content) if vote.vote_content else "None"
            
            if vote_content_str not in weighted_votes:
                weighted_votes[vote_content_str] = 0
            
            weighted_votes[vote_content_str] += vote.vote_weight
            total_weight += vote.vote_weight
        
        if not weighted_votes:
            return ConsensusResult(
                proposal_id=proposal.proposal_id,
                consensus_reached=False,
                final_decision=None,
                participant_count=len(votes),
                consensus_percentage=0.0
            )
        
        # Find consensus
        sorted_votes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
        winning_vote = sorted_votes[0]
        winning_percentage = winning_vote[1] / total_weight if total_weight > 0 else 0
        
        consensus_reached = winning_percentage >= proposal.required_consensus
        final_decision = json.loads(winning_vote[0]) if winning_vote[0] != "None" else None
        
        # Collect minority opinions
        minority_opinions = []
        for vote_content, weight in sorted_votes[1:3]:  # Top 2 minority opinions
            if vote_content != "None":
                minority_opinions.append(json.loads(vote_content))
        
        return ConsensusResult(
            proposal_id=proposal.proposal_id,
            consensus_reached=consensus_reached,
            final_decision=final_decision,
            participant_count=len(votes),
            consensus_percentage=winning_percentage,
            minority_opinions=minority_opinions,
            confidence_score=winning_percentage * 0.8 + (len(trusted_votes) / max(1, len(votes))) * 0.2
        )
    
    def _weighted_consensus(self, proposal: ConsensusProposal, votes: List[SecureVote]) -> ConsensusResult:
        """Execute weighted consensus based on agent stakes"""
        vote_weights = {}
        total_weight = 0
        
        for vote in votes:
            vote_content_str = json.dumps(vote.vote_content) if vote.vote_content else "None"
            
            if vote_content_str not in vote_weights:
                vote_weights[vote_content_str] = 0
            
            vote_weights[vote_content_str] += vote.vote_weight
            total_weight += vote.vote_weight
        
        if not vote_weights:
            return ConsensusResult(
                proposal_id=proposal.proposal_id,
                consensus_reached=False,
                final_decision=None,
                participant_count=len(votes),
                consensus_percentage=0.0
            )
        
        # Find weighted consensus
        sorted_votes = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
        winning_vote = sorted_votes[0]
        winning_percentage = winning_vote[1] / total_weight if total_weight > 0 else 0
        
        consensus_reached = winning_percentage >= proposal.required_consensus
        final_decision = json.loads(winning_vote[0]) if winning_vote[0] != "None" else None
        
        return ConsensusResult(
            proposal_id=proposal.proposal_id,
            consensus_reached=consensus_reached,
            final_decision=final_decision,
            participant_count=len(votes),
            consensus_percentage=winning_percentage,
            confidence_score=winning_percentage
        )
    
    def _majority_voting_consensus(self, proposal: ConsensusProposal, votes: List[SecureVote]) -> ConsensusResult:
        """Execute simple majority voting consensus"""
        vote_counts = Counter()
        
        for vote in votes:
            vote_content_str = json.dumps(vote.vote_content) if vote.vote_content else "None"
            vote_counts[vote_content_str] += 1
        
        if not vote_counts:
            return ConsensusResult(
                proposal_id=proposal.proposal_id,
                consensus_reached=False,
                final_decision=None,
                participant_count=len(votes),
                consensus_percentage=0.0
            )
        
        # Find majority
        winning_vote = vote_counts.most_common(1)[0]
        winning_percentage = winning_vote[1] / len(votes) if len(votes) > 0 else 0
        
        consensus_reached = winning_percentage >= proposal.required_consensus
        final_decision = json.loads(winning_vote[0]) if winning_vote[0] != "None" else None
        
        return ConsensusResult(
            proposal_id=proposal.proposal_id,
            consensus_reached=consensus_reached,
            final_decision=final_decision,
            participant_count=len(votes),
            consensus_percentage=winning_percentage,
            confidence_score=winning_percentage
        )
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus system statistics"""
        with self.consensus_lock:
            active_proposals = sum(1 for p in self.proposals.values() if not p.is_expired)
            completed_consensus = len(self.results)
            
            threat_counts = Counter()
            for result in self.results.values():
                for threat in result.security_threats_detected:
                    threat_counts[threat.value] += 1
            
            return {
                'registered_agents': len(self.agents),
                'trustworthy_agents': sum(1 for a in self.agents.values() if a.is_trustworthy),
                'active_proposals': active_proposals,
                'completed_consensus': completed_consensus,
                'security_threats_detected': dict(threat_counts),
                'average_agent_trust': sum(a.trust_score for a in self.agents.values()) / max(1, len(self.agents))
            }


# Global Byzantine consensus manager
byzantine_consensus_security = ByzantineConsensusSecurityManager()


def create_consensus_proposal(task_description: str, proposer_id: str, 
                            required_consensus: float = 0.67) -> Optional[str]:
    """Convenience function to create consensus proposal"""
    try:
        proposal = ConsensusProposal(
            task_description=task_description,
            proposer_id=proposer_id,
            required_consensus=required_consensus
        )
        
        if byzantine_consensus_security.submit_proposal(proposal):
            return proposal.proposal_id
        return None
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Proposal creation failed: {e}")
        return None


def participate_in_consensus(proposal_id: str, voter_id: str, vote_content: Any) -> bool:
    """Convenience function to participate in consensus"""
    try:
        vote = SecureVote(
            proposal_id=proposal_id,
            voter_id=voter_id,
            vote_content=vote_content
        )
        
        return byzantine_consensus_security.cast_vote(vote)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Consensus participation failed: {e}")
        return False