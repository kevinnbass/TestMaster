"""
Consensus Engine for Multi-Agent Decision Making

Implements various consensus algorithms for coordinating decisions across
multiple test generation agents, plan evaluators, and optimization components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
import statistics
import math
from datetime import datetime


class VotingMethod(Enum):
    """Different voting methods for consensus."""
    MAJORITY = "majority"              # Simple majority wins
    WEIGHTED = "weighted"              # Weighted by agent confidence/expertise
    RANKED_CHOICE = "ranked_choice"    # Ranked choice voting
    BORDA_COUNT = "borda_count"        # Borda count system
    APPROVAL = "approval"              # Approval voting
    CONSENSUS_THRESHOLD = "threshold"   # Require minimum threshold agreement


class ConsensusStrategy(Enum):
    """Strategies for reaching consensus."""
    FIRST_PAST_POST = "first_past_post"    # First to reach majority
    ITERATIVE_REFINEMENT = "iterative"     # Multiple rounds of voting
    WEIGHTED_AVERAGE = "weighted_avg"       # Weighted average of scores
    MEDIAN_AGGREGATION = "median"           # Median-based aggregation
    BYZANTINE_FAULT_TOLERANT = "bft"        # Byzantine fault tolerant


@dataclass 
class AgentVote:
    """Represents a vote from an agent."""
    agent_id: str
    choice: Any
    confidence: float = 1.0
    weight: float = 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """Result of consensus process."""
    decision: Any
    confidence: float
    support_ratio: float
    participating_agents: List[str]
    votes: List[AgentVote]
    strategy_used: ConsensusStrategy
    voting_method: VotingMethod
    rounds: int = 1
    convergence_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusEngine:
    """Main consensus engine for multi-agent coordination."""
    
    def __init__(self, 
                 default_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE,
                 default_voting: VotingMethod = VotingMethod.WEIGHTED,
                 min_participants: int = 2,
                 consensus_threshold: float = 0.6):
        
        self.default_strategy = default_strategy
        self.default_voting = default_voting
        self.min_participants = min_participants
        self.consensus_threshold = consensus_threshold
        
        # Consensus history for learning
        self.consensus_history: List[ConsensusResult] = []
        
        print("Consensus Engine initialized")
        print(f"   Default strategy: {default_strategy.value}")
        print(f"   Default voting: {default_voting.value}")
        print(f"   Min participants: {min_participants}")
        print(f"   Consensus threshold: {consensus_threshold}")
    
    def reach_consensus(self, 
                       votes: List[AgentVote],
                       strategy: Optional[ConsensusStrategy] = None,
                       voting_method: Optional[VotingMethod] = None,
                       max_rounds: int = 3) -> ConsensusResult:
        """Reach consensus from a set of agent votes."""
        
        strategy = strategy or self.default_strategy
        voting_method = voting_method or self.default_voting
        
        if len(votes) < self.min_participants:
            raise ValueError(f"Insufficient participants: {len(votes)} < {self.min_participants}")
        
        start_time = datetime.now()
        
        print(f"\nReaching consensus with {len(votes)} votes")
        print(f"Strategy: {strategy.value}, Voting: {voting_method.value}")
        
        # Execute consensus strategy
        if strategy == ConsensusStrategy.FIRST_PAST_POST:
            result = self._first_past_post_consensus(votes, voting_method)
        elif strategy == ConsensusStrategy.ITERATIVE_REFINEMENT:
            result = self._iterative_consensus(votes, voting_method, max_rounds)
        elif strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
            result = self._weighted_average_consensus(votes)
        elif strategy == ConsensusStrategy.MEDIAN_AGGREGATION:
            result = self._median_consensus(votes)
        elif strategy == ConsensusStrategy.BYZANTINE_FAULT_TOLERANT:
            result = self._byzantine_fault_tolerant_consensus(votes, voting_method)
        else:
            # Default to weighted average
            result = self._weighted_average_consensus(votes)
        
        # Calculate convergence time
        result.convergence_time = (datetime.now() - start_time).total_seconds()
        
        # Store in history
        self.consensus_history.append(result)
        
        print(f"Consensus reached: {result.decision}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Support: {result.support_ratio:.1%}")
        print(f"  Time: {result.convergence_time:.3f}s")
        
        return result
    
    def _first_past_post_consensus(self, votes: List[AgentVote], voting_method: VotingMethod) -> ConsensusResult:
        """First-past-the-post consensus (winner takes all)."""
        
        if voting_method == VotingMethod.WEIGHTED:
            # Weight votes by confidence and agent weight
            choice_weights = {}
            for vote in votes:
                choice = str(vote.choice)  # Convert to string for comparison
                weight = vote.confidence * vote.weight
                choice_weights[choice] = choice_weights.get(choice, 0) + weight
            
            # Find winner
            winner = max(choice_weights, key=choice_weights.get)
            total_weight = sum(choice_weights.values())
            support_ratio = choice_weights[winner] / total_weight
            
        else:
            # Simple majority
            choice_counts = {}
            for vote in votes:
                choice = str(vote.choice)
                choice_counts[choice] = choice_counts.get(choice, 0) + 1
            
            winner = max(choice_counts, key=choice_counts.get)
            support_ratio = choice_counts[winner] / len(votes)
        
        # Calculate confidence based on support ratio and vote confidence
        winning_votes = [v for v in votes if str(v.choice) == winner]
        avg_confidence = statistics.mean(v.confidence for v in winning_votes)
        final_confidence = support_ratio * avg_confidence
        
        return ConsensusResult(
            decision=self._parse_choice(winner),
            confidence=final_confidence,
            support_ratio=support_ratio,
            participating_agents=[v.agent_id for v in votes],
            votes=votes,
            strategy_used=ConsensusStrategy.FIRST_PAST_POST,
            voting_method=voting_method
        )
    
    def _weighted_average_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Weighted average consensus for numeric choices."""
        
        # Check if all choices are numeric
        numeric_votes = []
        for vote in votes:
            try:
                numeric_value = float(vote.choice)
                numeric_votes.append((numeric_value, vote.confidence * vote.weight))
            except (ValueError, TypeError):
                # Fall back to majority voting for non-numeric
                return self._first_past_post_consensus(votes, VotingMethod.WEIGHTED)
        
        # Calculate weighted average
        total_weight = sum(weight for _, weight in numeric_votes)
        if total_weight == 0:
            weighted_avg = statistics.mean(value for value, _ in numeric_votes)
        else:
            weighted_avg = sum(value * weight for value, weight in numeric_votes) / total_weight
        
        # Calculate confidence based on agreement
        values = [value for value, _ in numeric_votes]
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        max_std = max(values) - min(values) if len(values) > 1 else 1
        agreement = 1 - (std_dev / max(max_std, 0.1))  # Avoid division by zero
        
        avg_confidence = statistics.mean(v.confidence for v in votes)
        final_confidence = agreement * avg_confidence
        
        return ConsensusResult(
            decision=weighted_avg,
            confidence=final_confidence,
            support_ratio=1.0,  # All votes contribute
            participating_agents=[v.agent_id for v in votes],
            votes=votes,
            strategy_used=ConsensusStrategy.WEIGHTED_AVERAGE,
            voting_method=VotingMethod.WEIGHTED
        )
    
    def _median_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Median-based consensus for robust aggregation."""
        
        # Extract numeric values
        numeric_votes = []
        for vote in votes:
            try:
                numeric_value = float(vote.choice)
                numeric_votes.append(numeric_value)
            except (ValueError, TypeError):
                # Fall back to majority voting
                return self._first_past_post_consensus(votes, VotingMethod.MAJORITY)
        
        # Calculate median
        median_value = statistics.median(numeric_votes)
        
        # Calculate confidence based on how close votes are to median
        distances = [abs(value - median_value) for value in numeric_votes]
        max_distance = max(distances) if distances else 1
        closeness = 1 - (statistics.mean(distances) / max(max_distance, 0.1))
        
        avg_confidence = statistics.mean(v.confidence for v in votes)
        final_confidence = closeness * avg_confidence
        
        return ConsensusResult(
            decision=median_value,
            confidence=final_confidence,
            support_ratio=1.0,
            participating_agents=[v.agent_id for v in votes],
            votes=votes,
            strategy_used=ConsensusStrategy.MEDIAN_AGGREGATION,
            voting_method=VotingMethod.MAJORITY
        )
    
    def _iterative_consensus(self, votes: List[AgentVote], 
                           voting_method: VotingMethod, max_rounds: int) -> ConsensusResult:
        """Iterative consensus with multiple rounds."""
        
        current_votes = votes.copy()
        rounds = 0
        
        for round_num in range(max_rounds):
            rounds += 1
            
            # Get consensus for this round
            round_result = self._first_past_post_consensus(current_votes, voting_method)
            
            # Check if we have sufficient consensus
            if round_result.support_ratio >= self.consensus_threshold:
                round_result.rounds = rounds
                round_result.strategy_used = ConsensusStrategy.ITERATIVE_REFINEMENT
                return round_result
            
            # If not final round, simulate agent feedback/adjustment
            if round_num < max_rounds - 1:
                current_votes = self._simulate_agent_adjustment(current_votes, round_result)
        
        # Return best result after max rounds
        final_result = self._first_past_post_consensus(current_votes, voting_method)
        final_result.rounds = rounds
        final_result.strategy_used = ConsensusStrategy.ITERATIVE_REFINEMENT
        return final_result
    
    def _byzantine_fault_tolerant_consensus(self, votes: List[AgentVote], 
                                          voting_method: VotingMethod) -> ConsensusResult:
        """Byzantine fault tolerant consensus."""
        
        # Remove potential outliers (simple BFT approach)
        if len(votes) < 4:
            # Need at least 4 agents for BFT with 1 fault tolerance
            return self._first_past_post_consensus(votes, voting_method)
        
        # Remove bottom 25% by confidence (potential Byzantine agents)
        sorted_votes = sorted(votes, key=lambda v: v.confidence, reverse=True)
        reliable_votes = sorted_votes[:int(len(sorted_votes) * 0.75)]
        
        # Get consensus from reliable votes
        result = self._first_past_post_consensus(reliable_votes, voting_method)
        result.strategy_used = ConsensusStrategy.BYZANTINE_FAULT_TOLERANT
        result.metadata['excluded_agents'] = [v.agent_id for v in votes if v not in reliable_votes]
        
        return result
    
    def _simulate_agent_adjustment(self, votes: List[AgentVote], 
                                 round_result: ConsensusResult) -> List[AgentVote]:
        """Simulate agents adjusting their votes based on round result."""
        
        adjusted_votes = []
        
        for vote in votes:
            # Agents might adjust confidence based on how their vote aligns with consensus
            choice_match = str(vote.choice) == str(round_result.decision)
            
            if choice_match:
                # Increase confidence if vote matched consensus
                new_confidence = min(1.0, vote.confidence * 1.1)
            else:
                # Decrease confidence if vote didn't match
                new_confidence = max(0.1, vote.confidence * 0.9)
            
            adjusted_vote = AgentVote(
                agent_id=vote.agent_id,
                choice=vote.choice,
                confidence=new_confidence,
                weight=vote.weight,
                reasoning=vote.reasoning,
                metadata=vote.metadata
            )
            
            adjusted_votes.append(adjusted_vote)
        
        return adjusted_votes
    
    def _parse_choice(self, choice_str: str) -> Any:
        """Parse choice string back to appropriate type."""
        try:
            # Try to parse as float
            return float(choice_str)
        except ValueError:
            try:
                # Try to parse as int
                return int(choice_str)
            except ValueError:
                # Return as string
                return choice_str
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get statistics about consensus performance."""
        if not self.consensus_history:
            return {"message": "No consensus history available"}
        
        confidences = [r.confidence for r in self.consensus_history]
        support_ratios = [r.support_ratio for r in self.consensus_history]
        convergence_times = [r.convergence_time for r in self.consensus_history]
        
        return {
            "total_consensus_reached": len(self.consensus_history),
            "average_confidence": statistics.mean(confidences),
            "average_support_ratio": statistics.mean(support_ratios),
            "average_convergence_time": statistics.mean(convergence_times),
            "strategy_usage": self._get_strategy_usage_stats(),
            "most_recent_consensus": {
                "decision": self.consensus_history[-1].decision,
                "confidence": self.consensus_history[-1].confidence,
                "strategy": self.consensus_history[-1].strategy_used.value
            }
        }
    
    def _get_strategy_usage_stats(self) -> Dict[str, int]:
        """Get statistics on strategy usage."""
        strategy_counts = {}
        for result in self.consensus_history:
            strategy = result.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        return strategy_counts


def test_consensus_engine():
    """Test the consensus engine with sample votes."""
    print("\n" + "="*60)
    print("Testing Consensus Engine")
    print("="*60)
    
    # Create consensus engine
    engine = ConsensusEngine(
        default_strategy=ConsensusStrategy.WEIGHTED_AVERAGE,
        consensus_threshold=0.6
    )
    
    # Test with numeric votes (test scores)
    numeric_votes = [
        AgentVote("agent_1", 0.85, confidence=0.9, weight=1.0, reasoning="High coverage achieved"),
        AgentVote("agent_2", 0.78, confidence=0.8, weight=1.0, reasoning="Good but some edge cases missed"),
        AgentVote("agent_3", 0.82, confidence=0.85, weight=1.2, reasoning="Quality generator, high confidence"),
        AgentVote("agent_4", 0.75, confidence=0.7, weight=0.8, reasoning="Basic coverage only")
    ]
    
    print("\n1. Testing Weighted Average Consensus (Test Scores)")
    result = engine.reach_consensus(numeric_votes, ConsensusStrategy.WEIGHTED_AVERAGE)
    print(f"   Final Score: {result.decision:.3f}")
    
    # Test with categorical votes (strategy selection)
    categorical_votes = [
        AgentVote("agent_1", "comprehensive", confidence=0.9, weight=1.0),
        AgentVote("agent_2", "basic", confidence=0.6, weight=1.0),
        AgentVote("agent_3", "comprehensive", confidence=0.8, weight=1.2),
        AgentVote("agent_4", "security_focused", confidence=0.7, weight=0.9),
        AgentVote("agent_5", "comprehensive", confidence=0.85, weight=1.1)
    ]
    
    print("\n2. Testing Majority Consensus (Strategy Selection)")
    result = engine.reach_consensus(categorical_votes, ConsensusStrategy.FIRST_PAST_POST)
    print(f"   Chosen Strategy: {result.decision}")
    
    # Test iterative consensus
    print("\n3. Testing Iterative Consensus")
    result = engine.reach_consensus(
        categorical_votes, 
        ConsensusStrategy.ITERATIVE_REFINEMENT,
        max_rounds=3
    )
    print(f"   Final Strategy: {result.decision}")
    print(f"   Rounds: {result.rounds}")
    
    # Get stats
    stats = engine.get_consensus_stats()
    print(f"\n4. Consensus Statistics:")
    print(f"   Total consensus reached: {stats['total_consensus_reached']}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")
    print(f"   Average support ratio: {stats['average_support_ratio']:.1%}")
    
    print("\n✅ Consensus Engine test completed successfully!")
    return True


if __name__ == "__main__":
    test_consensus_engine()