"""
Hybrid Patterns
==============

Hybrid orchestration patterns that combine multiple orchestration approaches
for maximum flexibility and optimal execution strategies.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Tuple, Union
from collections import defaultdict
import json


class HybridMode(Enum):
    """Hybrid orchestration modes."""
    SEQUENTIAL_PARALLEL = "sequential_parallel"
    WORKFLOW_SWARM = "workflow_swarm"
    INTELLIGENCE_WORKFLOW = "intelligence_workflow"
    ADAPTIVE_MULTI = "adaptive_multi"
    CONTEXTUAL_SWITCHING = "contextual_switching"
    DYNAMIC_COMPOSITION = "dynamic_composition"


class SwitchingStrategy(Enum):
    """Strategy for switching between orchestration modes."""
    PERFORMANCE_BASED = "performance_based"
    WORKLOAD_BASED = "workload_based"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"
    MANUAL = "manual"
    INTELLIGENT = "intelligent"


class CompositionStrategy(Enum):
    """Strategy for composing multiple orchestration patterns."""
    LAYERED = "layered"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    MESH = "mesh"
    DYNAMIC = "dynamic"


@dataclass
class PatternMetrics:
    """Metrics for orchestration pattern performance."""
    execution_time: float = 0.0
    resource_utilization: float = 0.0
    success_rate: float = 1.0
    throughput: float = 0.0
    latency: float = 0.0
    cost: float = 0.0
    scalability_score: float = 1.0
    adaptability_score: float = 1.0


@dataclass
class SwitchingDecision:
    """Decision for switching between patterns."""
    from_pattern: str
    to_pattern: str
    reason: str
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]
    expected_improvement: float


@dataclass
class CompositionPlan:
    """Plan for composing multiple patterns."""
    primary_pattern: str
    secondary_patterns: List[str]
    composition_strategy: CompositionStrategy
    coordination_rules: Dict[str, Any]
    resource_distribution: Dict[str, float]
    execution_order: List[str]


class HybridPattern(ABC):
    """Abstract base class for hybrid orchestration patterns."""
    
    def __init__(
        self,
        pattern_name: str,
        hybrid_mode: HybridMode = HybridMode.ADAPTIVE_MULTI
    ):
        self.pattern_name = pattern_name
        self.hybrid_mode = hybrid_mode
        self.available_patterns: Dict[str, Any] = {}
        self.active_patterns: Dict[str, Any] = {}
        self.pattern_metrics: Dict[str, PatternMetrics] = {}
        self.switching_strategy = SwitchingStrategy.ADAPTIVE
        self.composition_strategy = CompositionStrategy.DYNAMIC
        self.switching_history: List[SwitchingDecision] = []
        self.composition_history: List[CompositionPlan] = []
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    @abstractmethod
    async def execute_hybrid(self, task: Any) -> Dict[str, Any]:
        """Execute task using hybrid orchestration."""
        pass
    
    @abstractmethod
    async def select_patterns(self, context: Dict[str, Any]) -> List[str]:
        """Select appropriate patterns for given context."""
        pass
    
    @abstractmethod
    async def compose_patterns(self, selected_patterns: List[str]) -> CompositionPlan:
        """Compose selected patterns into execution plan."""
        pass
    
    def register_pattern(self, pattern_name: str, pattern_instance: Any):
        """Register orchestration pattern."""
        self.available_patterns[pattern_name] = pattern_instance
        self.pattern_metrics[pattern_name] = PatternMetrics()
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit event to handlers."""
        for handler in self.event_handlers[event_type]:
            try:
                handler(event_type, event_data)
            except Exception:
                pass


class MultiModePattern(HybridPattern):
    """
    Multi-mode hybrid pattern that dynamically switches between
    orchestration approaches based on context and performance.
    
    Supports workflow, swarm, intelligence, and custom patterns
    with intelligent switching and composition strategies.
    """
    
    def __init__(self, multimode_name: str = "multimode_hybrid"):
        super().__init__(multimode_name, HybridMode.ADAPTIVE_MULTI)
        self.decision_engine: Optional[Any] = None
        self.performance_monitor: Optional[Any] = None
        self.context_analyzer: Optional[Any] = None
        self.switching_threshold = 0.2
        self.composition_cache: Dict[str, CompositionPlan] = {}
    
    async def execute_hybrid(self, task: Any) -> Dict[str, Any]:
        """Execute task using multi-mode hybrid orchestration."""
        try:
            # Analyze execution context
            context = await self._analyze_context(task)
            
            # Select appropriate patterns
            selected_patterns = await self.select_patterns(context)
            
            # Compose execution plan
            composition_plan = await self.compose_patterns(selected_patterns)
            
            # Execute with hybrid approach
            result = await self._execute_composition(task, composition_plan, context)
            
            # Monitor and potentially switch patterns
            await self._monitor_and_adapt(result, composition_plan, context)
            
            return result
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "fallback_executed": await self._execute_fallback(task)
            }
    
    async def select_patterns(self, context: Dict[str, Any]) -> List[str]:
        """Select appropriate patterns based on context analysis."""
        pattern_scores = {}
        
        # Score each available pattern
        for pattern_name, pattern_instance in self.available_patterns.items():
            score = await self._score_pattern(pattern_name, context)
            pattern_scores[pattern_name] = score
        
        # Select top patterns
        sorted_patterns = sorted(
            pattern_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select primary pattern and potential secondary patterns
        selected = [sorted_patterns[0][0]]  # Primary pattern
        
        # Add secondary patterns if they provide value
        for pattern_name, score in sorted_patterns[1:]:
            if score > 0.7 and len(selected) < 3:  # Max 3 patterns
                selected.append(pattern_name)
        
        self._emit_event("patterns_selected", {
            "selected_patterns": selected,
            "pattern_scores": pattern_scores
        })
        
        return selected
    
    async def compose_patterns(self, selected_patterns: List[str]) -> CompositionPlan:
        """Compose selected patterns into execution plan."""
        if len(selected_patterns) == 1:
            # Single pattern execution
            return CompositionPlan(
                primary_pattern=selected_patterns[0],
                secondary_patterns=[],
                composition_strategy=CompositionStrategy.LAYERED,
                coordination_rules={},
                resource_distribution={selected_patterns[0]: 1.0},
                execution_order=selected_patterns
            )
        
        # Multi-pattern composition
        primary_pattern = selected_patterns[0]
        secondary_patterns = selected_patterns[1:]
        
        # Determine composition strategy
        composition_strategy = await self._determine_composition_strategy(
            selected_patterns
        )
        
        # Calculate resource distribution
        resource_distribution = await self._calculate_resource_distribution(
            selected_patterns, composition_strategy
        )
        
        # Define coordination rules
        coordination_rules = await self._define_coordination_rules(
            selected_patterns, composition_strategy
        )
        
        # Determine execution order
        execution_order = await self._determine_execution_order(
            selected_patterns, composition_strategy
        )
        
        composition_plan = CompositionPlan(
            primary_pattern=primary_pattern,
            secondary_patterns=secondary_patterns,
            composition_strategy=composition_strategy,
            coordination_rules=coordination_rules,
            resource_distribution=resource_distribution,
            execution_order=execution_order
        )
        
        # Cache the composition
        context_hash = self._hash_context(selected_patterns)
        self.composition_cache[context_hash] = composition_plan
        
        self.composition_history.append(composition_plan)
        
        return composition_plan
    
    async def _analyze_context(self, task: Any) -> Dict[str, Any]:
        """Analyze execution context for pattern selection."""
        context = {
            "task_type": getattr(task, 'task_type', 'unknown'),
            "task_size": getattr(task, 'size', 1),
            "task_complexity": getattr(task, 'complexity', 0.5),
            "resource_requirements": getattr(task, 'resource_requirements', {}),
            "performance_requirements": getattr(task, 'performance_requirements', {}),
            "constraints": getattr(task, 'constraints', {}),
            "available_resources": await self._get_available_resources(),
            "historical_performance": await self._get_historical_performance(task),
            "current_load": await self._get_current_load()
        }
        
        return context
    
    async def _score_pattern(self, pattern_name: str, context: Dict[str, Any]) -> float:
        """Score pattern suitability for given context."""
        score = 0.0
        pattern_metrics = self.pattern_metrics[pattern_name]
        
        # Base score from historical performance
        score += pattern_metrics.success_rate * 0.3
        
        # Task type affinity
        task_type = context.get("task_type", "unknown")
        if pattern_name == "workflow" and task_type in ["sequential", "dag", "pipeline"]:
            score += 0.25
        elif pattern_name == "swarm" and task_type in ["distributed", "parallel", "collaborative"]:
            score += 0.25
        elif pattern_name == "intelligence" and task_type in ["adaptive", "learning", "optimization"]:
            score += 0.25
        
        # Resource efficiency
        resource_reqs = context.get("resource_requirements", {})
        available_resources = context.get("available_resources", {})
        
        if self._check_resource_compatibility(resource_reqs, available_resources):
            score += pattern_metrics.resource_utilization * 0.2
        
        # Performance characteristics
        perf_reqs = context.get("performance_requirements", {})
        if perf_reqs.get("low_latency", False) and pattern_metrics.latency < 1.0:
            score += 0.15
        if perf_reqs.get("high_throughput", False) and pattern_metrics.throughput > 0.8:
            score += 0.15
        
        # Scalability requirements
        task_size = context.get("task_size", 1)
        if task_size > 10 and pattern_metrics.scalability_score > 0.8:
            score += 0.1
        
        return min(1.0, score)
    
    async def _determine_composition_strategy(self, patterns: List[str]) -> CompositionStrategy:
        """Determine how to compose multiple patterns."""
        if len(patterns) <= 1:
            return CompositionStrategy.LAYERED
        
        # Analyze pattern compatibility
        primary = patterns[0]
        
        if "workflow" in patterns and "swarm" in patterns:
            return CompositionStrategy.HIERARCHICAL
        elif "intelligence" in patterns:
            return CompositionStrategy.LAYERED
        elif len(patterns) == 2:
            return CompositionStrategy.PARALLEL
        else:
            return CompositionStrategy.DYNAMIC
    
    async def _calculate_resource_distribution(
        self,
        patterns: List[str],
        strategy: CompositionStrategy
    ) -> Dict[str, float]:
        """Calculate resource distribution among patterns."""
        distribution = {}
        
        if strategy == CompositionStrategy.PARALLEL:
            # Equal distribution for parallel execution
            share = 1.0 / len(patterns)
            for pattern in patterns:
                distribution[pattern] = share
        elif strategy == CompositionStrategy.HIERARCHICAL:
            # Primary pattern gets most resources
            distribution[patterns[0]] = 0.7
            remaining = 0.3 / (len(patterns) - 1) if len(patterns) > 1 else 0
            for pattern in patterns[1:]:
                distribution[pattern] = remaining
        else:
            # Dynamic or layered - primary pattern gets priority
            distribution[patterns[0]] = 0.8
            for pattern in patterns[1:]:
                distribution[pattern] = 0.2 / (len(patterns) - 1) if len(patterns) > 1 else 0
        
        return distribution
    
    async def _define_coordination_rules(
        self,
        patterns: List[str],
        strategy: CompositionStrategy
    ) -> Dict[str, Any]:
        """Define coordination rules between patterns."""
        rules = {
            "communication_protocol": "async_messaging",
            "conflict_resolution": "primary_priority",
            "synchronization_points": [],
            "data_sharing": "controlled",
            "error_propagation": "contained"
        }
        
        if strategy == CompositionStrategy.HIERARCHICAL:
            rules.update({
                "primary_control": patterns[0],
                "delegation_rules": {
                    pattern: f"delegate_to_{pattern}"
                    for pattern in patterns[1:]
                }
            })
        elif strategy == CompositionStrategy.PARALLEL:
            rules.update({
                "synchronization_points": ["start", "end"],
                "result_aggregation": "merge_all"
            })
        
        return rules
    
    async def _determine_execution_order(
        self,
        patterns: List[str],
        strategy: CompositionStrategy
    ) -> List[str]:
        """Determine execution order for patterns."""
        if strategy == CompositionStrategy.PARALLEL:
            return patterns  # All execute simultaneously
        elif strategy == CompositionStrategy.HIERARCHICAL:
            return [patterns[0]]  # Primary controls others
        else:
            # Sequential or layered execution
            return patterns
    
    async def _execute_composition(
        self,
        task: Any,
        composition_plan: CompositionPlan,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the composed orchestration plan."""
        strategy = composition_plan.composition_strategy
        
        if strategy == CompositionStrategy.PARALLEL:
            return await self._execute_parallel_composition(task, composition_plan, context)
        elif strategy == CompositionStrategy.HIERARCHICAL:
            return await self._execute_hierarchical_composition(task, composition_plan, context)
        else:
            return await self._execute_layered_composition(task, composition_plan, context)
    
    async def _execute_parallel_composition(
        self,
        task: Any,
        composition_plan: CompositionPlan,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute patterns in parallel."""
        execution_tasks = []
        
        for pattern_name in composition_plan.execution_order:
            pattern_instance = self.available_patterns[pattern_name]
            resource_allocation = composition_plan.resource_distribution[pattern_name]
            
            # Create execution task
            exec_task = asyncio.create_task(
                self._execute_pattern_with_resources(
                    pattern_instance, task, resource_allocation, context
                )
            )
            execution_tasks.append((pattern_name, exec_task))
        
        # Wait for all patterns to complete
        results = {}
        for pattern_name, exec_task in execution_tasks:
            try:
                result = await exec_task
                results[pattern_name] = result
            except Exception as e:
                results[pattern_name] = {"status": "failed", "error": str(e)}
        
        # Aggregate results
        final_result = await self._aggregate_parallel_results(results)
        
        return final_result
    
    async def _execute_hierarchical_composition(
        self,
        task: Any,
        composition_plan: CompositionPlan,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute patterns in hierarchical manner."""
        primary_pattern = composition_plan.primary_pattern
        primary_instance = self.available_patterns[primary_pattern]
        
        # Enhance primary pattern with secondary patterns
        enhanced_context = context.copy()
        enhanced_context["secondary_patterns"] = {
            pattern_name: self.available_patterns[pattern_name]
            for pattern_name in composition_plan.secondary_patterns
        }
        enhanced_context["coordination_rules"] = composition_plan.coordination_rules
        
        # Execute primary pattern with enhanced capabilities
        result = await self._execute_pattern_with_resources(
            primary_instance, task,
            composition_plan.resource_distribution[primary_pattern],
            enhanced_context
        )
        
        return result
    
    async def _execute_layered_composition(
        self,
        task: Any,
        composition_plan: CompositionPlan,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute patterns in layered sequence."""
        current_task = task
        results = {}
        
        for pattern_name in composition_plan.execution_order:
            pattern_instance = self.available_patterns[pattern_name]
            resource_allocation = composition_plan.resource_distribution[pattern_name]
            
            # Execute pattern
            result = await self._execute_pattern_with_resources(
                pattern_instance, current_task, resource_allocation, context
            )
            
            results[pattern_name] = result
            
            # Use result as input for next layer if successful
            if result.get("status") == "completed":
                current_task = result.get("result", current_task)
            else:
                # Layer failed, decide whether to continue
                if not await self._should_continue_after_failure(pattern_name, result):
                    break
        
        return {
            "status": "completed",
            "final_result": current_task,
            "layer_results": results
        }
    
    async def _execute_pattern_with_resources(
        self,
        pattern_instance: Any,
        task: Any,
        resource_allocation: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute pattern with allocated resources."""
        # Set resource constraints based on allocation
        constrained_context = context.copy()
        constrained_context["resource_allocation"] = resource_allocation
        
        # Execute the pattern
        if hasattr(pattern_instance, 'execute_hybrid'):
            return await pattern_instance.execute_hybrid(task)
        elif hasattr(pattern_instance, 'execute_swarm'):
            return await pattern_instance.execute_swarm(task)
        elif hasattr(pattern_instance, 'execute'):
            return await pattern_instance.execute(task)
        else:
            # Fallback execution
            return {"status": "completed", "result": f"Executed {pattern_instance}"}
    
    async def _aggregate_parallel_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from parallel pattern execution."""
        successful_results = [
            result for result in results.values()
            if result.get("status") == "completed"
        ]
        
        if not successful_results:
            return {
                "status": "failed",
                "error": "All parallel patterns failed",
                "individual_results": results
            }
        
        # Simple aggregation strategy
        aggregated_result = {
            "status": "completed",
            "aggregated_data": [r.get("result") for r in successful_results],
            "individual_results": results,
            "success_count": len(successful_results),
            "total_count": len(results)
        }
        
        return aggregated_result
    
    async def _monitor_and_adapt(
        self,
        result: Dict[str, Any],
        composition_plan: CompositionPlan,
        context: Dict[str, Any]
    ):
        """Monitor execution and adapt patterns if needed."""
        # Update pattern metrics
        for pattern_name in composition_plan.execution_order:
            await self._update_pattern_metrics(pattern_name, result, context)
        
        # Check if switching is needed
        if await self._should_switch_patterns(result, composition_plan, context):
            switching_decision = await self._make_switching_decision(
                result, composition_plan, context
            )
            await self._execute_pattern_switch(switching_decision)
    
    async def _should_switch_patterns(
        self,
        result: Dict[str, Any],
        composition_plan: CompositionPlan,
        context: Dict[str, Any]
    ) -> bool:
        """Determine if pattern switching is needed."""
        # Check performance degradation
        if result.get("status") == "failed":
            return True
        
        # Check performance metrics
        primary_pattern = composition_plan.primary_pattern
        current_metrics = self.pattern_metrics[primary_pattern]
        
        if current_metrics.success_rate < 0.8:
            return True
        
        if current_metrics.performance_score < self.switching_threshold:
            return True
        
        return False
    
    async def _make_switching_decision(
        self,
        result: Dict[str, Any],
        current_plan: CompositionPlan,
        context: Dict[str, Any]
    ) -> SwitchingDecision:
        """Make decision about pattern switching."""
        current_pattern = current_plan.primary_pattern
        
        # Find best alternative pattern
        alternative_patterns = await self.select_patterns(context)
        best_alternative = None
        
        for pattern_name in alternative_patterns:
            if pattern_name != current_pattern:
                best_alternative = pattern_name
                break
        
        if not best_alternative:
            best_alternative = "fallback"
        
        switching_decision = SwitchingDecision(
            from_pattern=current_pattern,
            to_pattern=best_alternative,
            reason="performance_degradation",
            confidence=0.7,
            timestamp=datetime.now(),
            context=context,
            expected_improvement=0.2
        )
        
        return switching_decision
    
    async def _execute_pattern_switch(self, switching_decision: SwitchingDecision):
        """Execute pattern switching."""
        self.switching_history.append(switching_decision)
        
        # Update active patterns
        if switching_decision.from_pattern in self.active_patterns:
            del self.active_patterns[switching_decision.from_pattern]
        
        if switching_decision.to_pattern in self.available_patterns:
            self.active_patterns[switching_decision.to_pattern] = (
                self.available_patterns[switching_decision.to_pattern]
            )
        
        self._emit_event("pattern_switched", {
            "switching_decision": switching_decision
        })
    
    async def _execute_fallback(self, task: Any) -> Dict[str, Any]:
        """Execute fallback strategy when hybrid execution fails."""
        # Simple fallback to first available pattern
        if self.available_patterns:
            fallback_pattern = list(self.available_patterns.values())[0]
            try:
                if hasattr(fallback_pattern, 'execute'):
                    return await fallback_pattern.execute(task)
                else:
                    return {"status": "completed", "result": "fallback_executed"}
            except Exception:
                return {"status": "failed", "error": "fallback_failed"}
        
        return {"status": "failed", "error": "no_fallback_available"}
    
    # Helper methods
    async def _get_available_resources(self) -> Dict[str, float]:
        """Get currently available resources."""
        return {"cpu": 1.0, "memory": 1.0, "network": 1.0}
    
    async def _get_historical_performance(self, task: Any) -> Dict[str, float]:
        """Get historical performance for similar tasks."""
        return {"average_time": 1.0, "success_rate": 0.9}
    
    async def _get_current_load(self) -> float:
        """Get current system load."""
        return 0.5
    
    def _check_resource_compatibility(
        self,
        requirements: Dict[str, Any],
        available: Dict[str, Any]
    ) -> bool:
        """Check if resource requirements can be satisfied."""
        for resource, required_amount in requirements.items():
            if available.get(resource, 0) < required_amount:
                return False
        return True
    
    def _hash_context(self, patterns: List[str]) -> str:
        """Create hash of pattern combination for caching."""
        return "_".join(sorted(patterns))
    
    async def _update_pattern_metrics(
        self,
        pattern_name: str,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Update metrics for specific pattern."""
        metrics = self.pattern_metrics[pattern_name]
        
        # Update success rate
        if result.get("status") == "completed":
            metrics.success_rate = 0.9 * metrics.success_rate + 0.1 * 1.0
        else:
            metrics.success_rate = 0.9 * metrics.success_rate + 0.1 * 0.0
        
        # Update other metrics based on result
        execution_time = result.get("execution_time", 1.0)
        metrics.execution_time = 0.9 * metrics.execution_time + 0.1 * execution_time
    
    async def _should_continue_after_failure(self, pattern_name: str, result: Dict[str, Any]) -> bool:
        """Determine if execution should continue after pattern failure."""
        # Simple strategy: continue unless critical failure
        return result.get("error_type") != "critical"


# Export key classes
__all__ = [
    'HybridMode',
    'SwitchingStrategy',
    'CompositionStrategy',
    'PatternMetrics',
    'SwitchingDecision',
    'CompositionPlan',
    'HybridPattern',
    'MultiModePattern'
]