"""
Dynamic Agent Handoff System

Inspired by OpenAI Swarm's dynamic handoff patterns.
Provides intelligent routing of messages to appropriate agents.

Features:
- Intelligent agent capability matching
- Context-aware routing decisions  
- Handoff history and learning
- Fallback mechanisms
- Performance tracking
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from core.context_manager import get_context_manager
from core.monitoring_decorators import monitor_performance


class AgentType(Enum):
    """Available agent types for dynamic handoff."""
    CLAUDE_CODE = "claude_code"
    CLAUDE_ARTIFACTS = "claude_artifacts"
    CLAUDE_SONNET = "claude_sonnet"
    SPECIALIZED_TESTER = "specialized_tester"
    CODE_REVIEWER = "code_reviewer"
    DOCUMENTATION_AGENT = "documentation_agent"
    PERFORMANCE_ANALYZER = "performance_analyzer"


@dataclass
class HandoffDecision:
    """Result of a handoff routing decision."""
    target_agent: AgentType
    confidence: float  # 0-100
    reasoning: str
    fallback_agents: List[AgentType] = field(default_factory=list)
    context_preserved: bool = False
    estimated_response_time: Optional[float] = None


@dataclass
class AgentCapability:
    """Capability definition for an agent."""
    agent_type: AgentType
    strengths: List[str]
    complexity_range: tuple  # (min, max) on 1-10 scale
    file_types: List[str]
    preferred_tasks: List[str]
    availability_score: float = 1.0  # 0-1 scale
    response_time_avg: float = 60.0  # seconds


class DynamicHandoffSystem:
    """
    Dynamic agent handoff system for intelligent message routing.
    
    Uses OpenAI Swarm-style handoff patterns with:
    - Capability-based agent matching
    - Context preservation during handoffs
    - Learning from handoff outcomes
    - Intelligent fallback mechanisms
    """
    
    def __init__(self):
        """Initialize dynamic handoff system."""
        self.enabled = FeatureFlags.is_enabled('layer2_monitoring', 'dynamic_handoff')
        
        if not self.enabled:
            return
            
        # Initialize core components
        self.agent_router = AgentRouter()
        self.handoff_history = []
        self.performance_tracker = HandoffPerformanceTracker()
        
        # Configuration
        config = FeatureFlags.get_config('layer2_monitoring', 'dynamic_handoff')
        self.preserve_context = config.get('preserve_context', True)
        self.auto_fallback = config.get('auto_fallback', True)
        self.max_handoff_attempts = config.get('max_attempts', 3)
        self.learning_enabled = config.get('learning', True)
        
        # Initialize state management
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
            self._load_handoff_history()
        else:
            self.shared_state = None
        
        if FeatureFlags.is_enabled('layer1_test_foundation', 'context_preservation'):
            self.context_manager = get_context_manager()
        else:
            self.context_manager = None
        
        print("Dynamic agent handoff system initialized")
        print(f"   Context preservation: {'enabled' if self.preserve_context else 'disabled'}")
        print(f"   Auto fallback: {'enabled' if self.auto_fallback else 'disabled'}")
        print(f"   Learning: {'enabled' if self.learning_enabled else 'disabled'}")
    
    @monitor_performance(name="handoff_decision")
    def determine_handoff(self, message_analysis: Dict[str, Any]) -> HandoffDecision:
        """
        Determine the best agent for handling a message.
        
        Args:
            message_analysis: Analysis of the message characteristics
            
        Returns:
            HandoffDecision with target agent and reasoning
        """
        if not self.enabled:
            return HandoffDecision(
                target_agent=AgentType.CLAUDE_CODE,
                confidence=100.0,
                reasoning="Dynamic handoff disabled"
            )
        
        # Get routing decision from agent router
        decision = self.agent_router.route_message(message_analysis)
        
        # Apply learning adjustments if enabled
        if self.learning_enabled:
            decision = self._apply_learning_adjustments(decision, message_analysis)
        
        # Record decision
        self._record_handoff_decision(decision, message_analysis)
        
        return decision
    
    @monitor_performance(name="handoff_execution")
    def execute_handoff(self, decision: HandoffDecision, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a handoff to the target agent.
        
        Args:
            decision: Handoff decision
            message_data: Original message data
            
        Returns:
            Handoff execution result
        """
        if not self.enabled:
            return {'success': False, 'reason': 'Dynamic handoff disabled'}
        
        try:
            # Preserve context if enabled
            context_id = None
            if self.preserve_context and self.context_manager:
                context = self._create_handoff_context(decision, message_data)
                preserved_context = self.context_manager.preserve(context)
                context_id = preserved_context['_preservation']['context_id']
            
            # Create handoff package
            handoff_package = self._create_handoff_package(decision, message_data, context_id)
            
            # Execute handoff
            result = self._perform_handoff(decision.target_agent, handoff_package)
            
            # Record outcome
            self._record_handoff_outcome(decision, result)
            
            # Update performance tracking
            self.performance_tracker.record_handoff(
                decision.target_agent,
                result.get('success', False),
                result.get('response_time', 0)
            )
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'fallback_attempted': False
            }
            
            # Try fallback if enabled
            if self.auto_fallback and decision.fallback_agents:
                fallback_result = self._attempt_fallback(decision, message_data)
                error_result['fallback_attempted'] = True
                error_result['fallback_result'] = fallback_result
                
                if fallback_result.get('success'):
                    return fallback_result
            
            return error_result
    
    def _create_handoff_context(self, decision: HandoffDecision, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create context for handoff preservation."""
        return {
            'handoff_type': 'dynamic_routing',
            'target_agent': decision.target_agent.value,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'original_message_id': message_data.get('message_id'),
            'original_message_type': message_data.get('message_type'),
            'handoff_timestamp': datetime.now().isoformat(),
            'fallback_agents': [agent.value for agent in decision.fallback_agents],
            'estimated_response_time': decision.estimated_response_time
        }
    
    def _create_handoff_package(self, decision: HandoffDecision, message_data: Dict[str, Any], 
                               context_id: Optional[str]) -> Dict[str, Any]:
        """Create handoff package for target agent."""
        return {
            'handoff_metadata': {
                'target_agent': decision.target_agent.value,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'context_id': context_id,
                'handoff_timestamp': datetime.now().isoformat(),
                'source_system': 'testmaster_dynamic_handoff'
            },
            'original_message': message_data,
            'agent_instructions': self._generate_agent_instructions(decision),
            'expected_deliverables': self._define_expected_deliverables(decision, message_data)
        }
    
    def _generate_agent_instructions(self, decision: HandoffDecision) -> Dict[str, Any]:
        """Generate specific instructions for the target agent."""
        base_instructions = {
            'task_priority': 'normal',
            'response_format': 'structured',
            'include_reasoning': True
        }
        
        # Agent-specific instructions
        agent_specific = {
            AgentType.SPECIALIZED_TESTER: {
                'focus_areas': ['edge_cases', 'integration_patterns', 'performance_impacts'],
                'test_types': ['unit', 'integration', 'performance'],
                'coverage_target': 90
            },
            AgentType.CODE_REVIEWER: {
                'review_aspects': ['security', 'performance', 'maintainability', 'best_practices'],
                'severity_levels': ['critical', 'major', 'minor', 'suggestion'],
                'include_recommendations': True
            },
            AgentType.PERFORMANCE_ANALYZER: {
                'analysis_areas': ['cpu_usage', 'memory_consumption', 'io_patterns'],
                'benchmarking': True,
                'optimization_suggestions': True
            },
            AgentType.DOCUMENTATION_AGENT: {
                'doc_types': ['api_docs', 'user_guides', 'technical_specs'],
                'format': 'markdown',
                'include_examples': True
            }
        }
        
        if decision.target_agent in agent_specific:
            base_instructions.update(agent_specific[decision.target_agent])
        
        return base_instructions
    
    def _define_expected_deliverables(self, decision: HandoffDecision, message_data: Dict[str, Any]) -> List[str]:
        """Define what deliverables are expected from the agent."""
        message_type = message_data.get('message_type', '')
        
        deliverable_map = {
            'breaking_tests': [
                'Root cause analysis',
                'Fix recommendations',
                'Test repair suggestions',
                'Prevention strategies'
            ],
            'coverage_gaps': [
                'Test case specifications', 
                'Implementation examples',
                'Coverage improvement plan',
                'Priority recommendations'
            ],
            'performance_issues': [
                'Performance analysis report',
                'Bottleneck identification',
                'Optimization recommendations',
                'Implementation guidance'
            ]
        }
        
        return deliverable_map.get(message_type, ['Analysis', 'Recommendations', 'Action plan'])
    
    def _perform_handoff(self, target_agent: AgentType, handoff_package: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual handoff to target agent."""
        # This would integrate with actual agent communication systems
        # For now, return a simulated result
        
        handoff_timestamp = datetime.now()
        
        # Simulate handoff based on agent capabilities
        success_probability = self.agent_router.get_agent_capability(target_agent).availability_score
        
        if success_probability > 0.7:  # Successful handoff
            return {
                'success': True,
                'target_agent': target_agent.value,
                'handoff_id': f"handoff_{int(handoff_timestamp.timestamp())}",
                'estimated_completion': handoff_timestamp.isoformat(),
                'response_time': 1.5,  # Simulated
                'agent_acknowledgment': True
            }
        else:  # Failed handoff
            return {
                'success': False,
                'target_agent': target_agent.value,
                'error': 'Agent unavailable or overloaded',
                'response_time': 0.5
            }
    
    def _attempt_fallback(self, decision: HandoffDecision, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt fallback to alternative agents."""
        for fallback_agent in decision.fallback_agents:
            try:
                fallback_decision = HandoffDecision(
                    target_agent=fallback_agent,
                    confidence=decision.confidence * 0.8,  # Reduced confidence
                    reasoning=f"Fallback from {decision.target_agent.value}",
                    fallback_agents=[]
                )
                
                context_id = None
                if self.preserve_context and self.context_manager:
                    context = self._create_handoff_context(fallback_decision, message_data)
                    preserved_context = self.context_manager.preserve(context)
                    context_id = preserved_context['_preservation']['context_id']
                
                handoff_package = self._create_handoff_package(fallback_decision, message_data, context_id)
                result = self._perform_handoff(fallback_agent, handoff_package)
                
                if result.get('success'):
                    result['fallback_from'] = decision.target_agent.value
                    return result
                    
            except Exception as e:
                print(f"Fallback to {fallback_agent.value} failed: {e}")
                continue
        
        # All fallbacks failed
        return {
            'success': False,
            'error': 'All fallback agents failed',
            'attempted_agents': [agent.value for agent in decision.fallback_agents]
        }
    
    def _apply_learning_adjustments(self, decision: HandoffDecision, 
                                   message_analysis: Dict[str, Any]) -> HandoffDecision:
        """Apply learning-based adjustments to routing decision."""
        if not self.learning_enabled or not self.shared_state:
            return decision
        
        # Get historical performance for this agent and message type
        agent_performance = self.performance_tracker.get_agent_performance(decision.target_agent)
        
        if agent_performance:
            # Adjust confidence based on historical success rate
            historical_success = agent_performance.get('success_rate', 0.5)
            adjusted_confidence = decision.confidence * (0.5 + historical_success * 0.5)
            
            decision.confidence = min(100.0, max(0.0, adjusted_confidence))
        
        return decision
    
    def _record_handoff_decision(self, decision: HandoffDecision, message_analysis: Dict[str, Any]):
        """Record handoff decision for analysis."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'target_agent': decision.target_agent.value,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'message_analysis': message_analysis,
            'fallback_agents': [agent.value for agent in decision.fallback_agents]
        }
        
        self.handoff_history.append(record)
        
        # Keep only recent history
        if len(self.handoff_history) > 1000:
            self.handoff_history = self.handoff_history[-1000:]
        
        # Store in shared state if enabled
        if self.shared_state:
            self.shared_state.append('handoff_decisions', record)
    
    def _record_handoff_outcome(self, decision: HandoffDecision, result: Dict[str, Any]):
        """Record the outcome of a handoff."""
        outcome = {
            'timestamp': datetime.now().isoformat(),
            'target_agent': decision.target_agent.value,
            'success': result.get('success', False),
            'response_time': result.get('response_time', 0),
            'error': result.get('error'),
            'fallback_used': 'fallback_from' in result
        }
        
        if self.shared_state:
            self.shared_state.append('handoff_outcomes', outcome)
    
    def _load_handoff_history(self):
        """Load handoff history from shared state."""
        if self.shared_state:
            history = self.shared_state.get('handoff_decisions', [])
            self.handoff_history = history[-1000:]  # Keep recent history
    
    def get_handoff_statistics(self) -> Dict[str, Any]:
        """Get handoff system statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        total_handoffs = len(self.handoff_history)
        if total_handoffs == 0:
            return {'enabled': True, 'total_handoffs': 0}
        
        # Agent distribution
        agent_counts = {}
        for record in self.handoff_history:
            agent = record['target_agent']
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Average confidence
        avg_confidence = sum(record['confidence'] for record in self.handoff_history) / total_handoffs
        
        # Performance metrics
        performance_stats = self.performance_tracker.get_overall_stats()
        
        return {
            'enabled': True,
            'total_handoffs': total_handoffs,
            'agent_distribution': agent_counts,
            'average_confidence': avg_confidence,
            'performance': performance_stats,
            'learning_enabled': self.learning_enabled,
            'context_preservation': self.preserve_context
        }


class AgentRouter:
    """Routes messages to appropriate agents based on capabilities."""
    
    def __init__(self):
        """Initialize agent router with default capabilities."""
        self.agent_capabilities = self._setup_default_capabilities()
        self.routing_history = []
    
    def _setup_default_capabilities(self) -> Dict[AgentType, AgentCapability]:
        """Setup default agent capabilities."""
        return {
            AgentType.CLAUDE_CODE: AgentCapability(
                agent_type=AgentType.CLAUDE_CODE,
                strengths=['code_generation', 'debugging', 'refactoring', 'general_coding'],
                complexity_range=(1, 10),
                file_types=['.py', '.js', '.ts', '.java', '.cpp', '.c'],
                preferred_tasks=['test_generation', 'bug_fixing', 'code_review'],
                availability_score=0.95,
                response_time_avg=30.0
            ),
            AgentType.SPECIALIZED_TESTER: AgentCapability(
                agent_type=AgentType.SPECIALIZED_TESTER,
                strengths=['test_design', 'edge_cases', 'integration_testing', 'performance_testing'],
                complexity_range=(3, 10),
                file_types=['.py', '.js', '.ts'],
                preferred_tasks=['complex_testing', 'integration_tests', 'performance_tests'],
                availability_score=0.8,
                response_time_avg=45.0
            ),
            AgentType.CODE_REVIEWER: AgentCapability(
                agent_type=AgentType.CODE_REVIEWER,
                strengths=['code_quality', 'security_analysis', 'best_practices', 'architecture'],
                complexity_range=(2, 10),
                file_types=['.py', '.js', '.ts', '.java', '.cpp'],
                preferred_tasks=['code_review', 'security_audit', 'architecture_analysis'],
                availability_score=0.9,
                response_time_avg=60.0
            ),
            AgentType.PERFORMANCE_ANALYZER: AgentCapability(
                agent_type=AgentType.PERFORMANCE_ANALYZER,
                strengths=['performance_optimization', 'profiling', 'benchmarking', 'scalability'],
                complexity_range=(4, 10),
                file_types=['.py', '.js', '.ts', '.java', '.cpp'],
                preferred_tasks=['performance_analysis', 'optimization', 'profiling'],
                availability_score=0.7,
                response_time_avg=90.0
            )
        }
    
    def route_message(self, message_analysis: Dict[str, Any]) -> HandoffDecision:
        """Route message to best-fit agent."""
        scores = {}
        
        for agent_type, capability in self.agent_capabilities.items():
            score = self._calculate_agent_score(capability, message_analysis)
            scores[agent_type] = score
        
        # Sort agents by score
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_agent = sorted_agents[0][0]
        best_score = sorted_agents[0][1]
        
        # Determine fallback agents
        fallback_agents = [agent for agent, score in sorted_agents[1:3] if score > 0.3]
        
        # Calculate confidence based on score and availability
        base_confidence = min(100.0, best_score * 20)  # Scale to 0-100
        availability = self.agent_capabilities[best_agent].availability_score
        final_confidence = base_confidence * availability
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_agent, message_analysis, best_score)
        
        decision = HandoffDecision(
            target_agent=best_agent,
            confidence=final_confidence,
            reasoning=reasoning,
            fallback_agents=fallback_agents,
            estimated_response_time=self.agent_capabilities[best_agent].response_time_avg
        )
        
        # Record routing decision
        self.routing_history.append({
            'timestamp': datetime.now().isoformat(),
            'message_analysis': message_analysis,
            'scores': {agent.value: score for agent, score in scores.items()},
            'selected_agent': best_agent.value,
            'confidence': final_confidence
        })
        
        return decision
    
    def _calculate_agent_score(self, capability: AgentCapability, analysis: Dict[str, Any]) -> float:
        """Calculate how well an agent matches message requirements."""
        score = 0.0
        
        # Task type matching
        task_type = analysis.get('task_type', '')
        if task_type in capability.preferred_tasks:
            score += 2.0
        
        # Domain expertise
        domain = analysis.get('domain', '')
        domain_matches = sum(1 for strength in capability.strengths if domain in strength)
        score += domain_matches * 1.5
        
        # Complexity range matching
        complexity = analysis.get('complexity', 5)
        if capability.complexity_range[0] <= complexity <= capability.complexity_range[1]:
            score += 1.0
        
        # File type support
        file_types = analysis.get('file_types', [])
        type_matches = sum(1 for ft in file_types if ft in capability.file_types)
        score += type_matches * 0.5
        
        # Priority and urgency adjustments
        priority = analysis.get('priority', 2)
        if priority >= 4:  # High priority
            score += 0.5
        
        # Availability factor
        score *= capability.availability_score
        
        return score
    
    def _generate_reasoning(self, agent: AgentType, analysis: Dict[str, Any], score: float) -> str:
        """Generate human-readable reasoning for agent selection."""
        capability = self.agent_capabilities[agent]
        task_type = analysis.get('task_type', 'general')
        complexity = analysis.get('complexity', 5)
        
        if task_type in capability.preferred_tasks:
            return f"Agent specializes in {task_type} (complexity: {complexity}/10)"
        elif score > 3.0:
            return f"High capability match for {analysis.get('domain', 'general')} domain"
        elif score > 2.0:
            return f"Good fit based on file types and complexity"
        else:
            return f"Basic compatibility with task requirements"
    
    def get_agent_capability(self, agent_type: AgentType) -> AgentCapability:
        """Get capability information for an agent."""
        return self.agent_capabilities.get(agent_type)


class HandoffPerformanceTracker:
    """Tracks performance metrics for handoff agents."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.agent_metrics = {}
    
    def record_handoff(self, agent: AgentType, success: bool, response_time: float):
        """Record a handoff performance metric."""
        if agent not in self.agent_metrics:
            self.agent_metrics[agent] = {
                'total_handoffs': 0,
                'successful_handoffs': 0,
                'total_response_time': 0.0,
                'min_response_time': float('inf'),
                'max_response_time': 0.0
            }
        
        metrics = self.agent_metrics[agent]
        metrics['total_handoffs'] += 1
        
        if success:
            metrics['successful_handoffs'] += 1
        
        if response_time > 0:
            metrics['total_response_time'] += response_time
            metrics['min_response_time'] = min(metrics['min_response_time'], response_time)
            metrics['max_response_time'] = max(metrics['max_response_time'], response_time)
    
    def get_agent_performance(self, agent: AgentType) -> Dict[str, Any]:
        """Get performance metrics for a specific agent."""
        if agent not in self.agent_metrics:
            return None
        
        metrics = self.agent_metrics[agent]
        
        success_rate = metrics['successful_handoffs'] / max(1, metrics['total_handoffs'])
        avg_response_time = metrics['total_response_time'] / max(1, metrics['total_handoffs'])
        
        return {
            'success_rate': success_rate,
            'average_response_time': avg_response_time,
            'min_response_time': metrics['min_response_time'],
            'max_response_time': metrics['max_response_time'],
            'total_handoffs': metrics['total_handoffs']
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.agent_metrics:
            return {}
        
        total_handoffs = sum(m['total_handoffs'] for m in self.agent_metrics.values())
        total_successful = sum(m['successful_handoffs'] for m in self.agent_metrics.values())
        
        overall_success_rate = total_successful / max(1, total_handoffs)
        
        # Best performing agent
        best_agent = None
        best_success_rate = 0
        
        for agent, metrics in self.agent_metrics.items():
            agent_success_rate = metrics['successful_handoffs'] / max(1, metrics['total_handoffs'])
            if agent_success_rate > best_success_rate:
                best_success_rate = agent_success_rate
                best_agent = agent
        
        return {
            'total_handoffs': total_handoffs,
            'overall_success_rate': overall_success_rate,
            'best_performing_agent': best_agent.value if best_agent else None,
            'agents_tracked': len(self.agent_metrics)
        }


# Global instance
_dynamic_handoff_system = None


def get_dynamic_handoff_system() -> DynamicHandoffSystem:
    """Get the global dynamic handoff system instance."""
    global _dynamic_handoff_system
    if _dynamic_handoff_system is None:
        _dynamic_handoff_system = DynamicHandoffSystem()
    return _dynamic_handoff_system