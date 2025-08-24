"""
Bottleneck Detection & Resolution Agent

Intelligent bottleneck detection and automated resolution system that analyzes
workflow performance patterns, identifies bottlenecks using advanced algorithms,
and implements adaptive resolution strategies with consensus-driven decision making.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from enum import Enum
from datetime import datetime, timedelta
import statistics
import json
from collections import defaultdict, deque
import math

from ..hierarchical_planning import (
    HierarchicalTestPlanner, 
    PlanningNode, 
    TestPlanGenerator, 
    TestPlanEvaluator,
    EvaluationCriteria,
    get_best_planner
)
from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import get_shared_state, cache_test_result, get_cached_test_result
from ...flow_optimizer.dependency_resolver import get_dependency_resolver, DependencyGraph, TaskNode
from ...telemetry.flow_analyzer import get_flow_analyzer, FlowAnalysis
from ...flow_optimizer.flow_analyzer import get_flow_analyzer as get_legacy_flow_analyzer


class BottleneckType(Enum):
    """Types of bottlenecks that can be detected."""
    DEPENDENCY_CHAIN = "dependency_chain"
    RESOURCE_CONTENTION = "resource_contention"
    SERIALIZATION_POINT = "serialization_point"
    HOT_SPOT = "hot_spot"
    QUEUE_BUILDUP = "queue_buildup"
    SYNCHRONIZATION_POINT = "synchronization_point"
    DATA_FLOW = "data_flow"
    CRITICAL_PATH = "critical_path"


class ResolutionStrategy(Enum):
    """Strategies for resolving bottlenecks."""
    PARALLEL_EXECUTION = "parallel_execution"
    RESOURCE_SCALING = "resource_scaling"
    DEPENDENCY_OPTIMIZATION = "dependency_optimization"
    QUEUE_OPTIMIZATION = "queue_optimization"
    CACHING_STRATEGY = "caching_strategy"
    LOAD_BALANCING = "load_balancing"
    ASYNC_PROCESSING = "async_processing"
    BATCHING_OPTIMIZATION = "batching_optimization"


class BottleneckSeverity(Enum):
    """Severity levels for bottlenecks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BottleneckDetection:
    """Detected bottleneck information."""
    bottleneck_id: str
    bottleneck_type: BottleneckType
    severity: BottleneckSeverity
    component: str
    description: str
    impact_score: float  # 0-1 scale
    frequency: float     # How often this bottleneck occurs
    location: str
    affected_workflows: List[str]
    performance_degradation: float  # Percentage impact
    detection_confidence: float     # 0-1 scale
    detected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolutionAction:
    """Action to resolve a bottleneck."""
    action_id: str
    strategy: ResolutionStrategy
    description: str
    parameters: Dict[str, Any]
    estimated_improvement: float  # Expected performance improvement percentage
    implementation_complexity: float  # 0-1 scale
    resource_requirements: Dict[str, Any]
    success_probability: float   # 0-1 scale
    side_effects: List[str]
    execution_time_estimate: float  # Minutes


@dataclass
class BottleneckResolution:
    """Resolution plan for a bottleneck."""
    resolution_id: str
    bottleneck_id: str
    primary_action: ResolutionAction
    alternative_actions: List[ResolutionAction]
    resolution_priority: float
    estimated_total_improvement: float
    risk_assessment: Dict[str, Any]
    implementation_plan: List[str]
    success_metrics: List[str]
    created_at: datetime
    executed: bool = False
    execution_results: Optional[Dict[str, Any]] = None


class BottleneckDetector:
    """Core bottleneck detection engine."""
    
    def __init__(self):
        self.flow_analyzer = get_flow_analyzer()
        self.legacy_flow_analyzer = get_legacy_flow_analyzer()
        self.dependency_resolver = get_dependency_resolver()
        self.shared_state = get_shared_state()
        
        # Detection parameters
        self.impact_threshold = 0.1      # Minimum impact to consider as bottleneck
        self.frequency_threshold = 0.3   # Minimum frequency to flag as bottleneck
        self.confidence_threshold = 0.7  # Minimum confidence for detection
        
        # Analysis window
        self.analysis_window_hours = 4
        self.min_samples = 10
        
        print("Bottleneck Detector initialized")
        print(f"   Analysis window: {self.analysis_window_hours} hours")
        print(f"   Impact threshold: {self.impact_threshold}")
    
    def detect_bottlenecks(self, workflow_id: str = None) -> List[BottleneckDetection]:
        """Detect bottlenecks across workflows or specific workflow."""
        
        all_bottlenecks = []
        
        # Get flow analysis data
        flow_analysis = self._get_flow_analysis_data()
        
        # Get dependency graph data
        dependency_data = self._get_dependency_data(workflow_id)
        
        # Detect different types of bottlenecks
        dependency_bottlenecks = self._detect_dependency_bottlenecks(dependency_data, workflow_id)
        resource_bottlenecks = self._detect_resource_bottlenecks(flow_analysis, workflow_id)
        serialization_bottlenecks = self._detect_serialization_bottlenecks(flow_analysis, workflow_id)
        hotspot_bottlenecks = self._detect_hotspot_bottlenecks(flow_analysis, workflow_id)
        queue_bottlenecks = self._detect_queue_bottlenecks(flow_analysis, workflow_id)
        
        all_bottlenecks.extend(dependency_bottlenecks)
        all_bottlenecks.extend(resource_bottlenecks)
        all_bottlenecks.extend(serialization_bottlenecks)
        all_bottlenecks.extend(hotspot_bottlenecks)
        all_bottlenecks.extend(queue_bottlenecks)
        
        # Filter by confidence and impact
        filtered_bottlenecks = [
            b for b in all_bottlenecks 
            if b.detection_confidence >= self.confidence_threshold 
            and b.impact_score >= self.impact_threshold
        ]
        
        # Sort by impact score
        filtered_bottlenecks.sort(key=lambda x: x.impact_score, reverse=True)
        
        print(f"Detected {len(filtered_bottlenecks)} bottlenecks (from {len(all_bottlenecks)} candidates)")
        
        return filtered_bottlenecks
    
    def _get_flow_analysis_data(self) -> Optional[FlowAnalysis]:
        """Get flow analysis data."""
        try:
            return self.flow_analyzer.analyze_flows(timeframe_hours=self.analysis_window_hours)
        except Exception as e:
            print(f"Flow analysis failed: {e}")
            return None
    
    def _get_dependency_data(self, workflow_id: str = None) -> Optional[DependencyGraph]:
        """Get dependency graph data."""
        try:
            # Create sample tasks for analysis
            sample_tasks = self._get_sample_tasks(workflow_id)
            return self.dependency_resolver.resolve_dependencies(sample_tasks)
        except Exception as e:
            print(f"Dependency analysis failed: {e}")
            return None
    
    def _get_sample_tasks(self, workflow_id: str = None) -> List[Dict[str, Any]]:
        """Get sample tasks for dependency analysis."""
        # In a real implementation, this would come from actual workflow data
        return [
            {
                'id': 'test_generation',
                'dependencies': [],
                'estimated_duration': 200.0,
                'resource_requirements': {'cpu': 0.5, 'memory': '1GB'}
            },
            {
                'id': 'test_execution',
                'dependencies': ['test_generation'],
                'estimated_duration': 500.0,
                'resource_requirements': {'cpu': 0.8, 'memory': '2GB'}
            },
            {
                'id': 'result_validation',
                'dependencies': ['test_execution'],
                'estimated_duration': 100.0,
                'resource_requirements': {'cpu': 0.3, 'memory': '512MB'}
            },
            {
                'id': 'report_generation',
                'dependencies': ['result_validation'],
                'estimated_duration': 150.0,
                'resource_requirements': {'cpu': 0.4, 'memory': '1GB'}
            }
        ]
    
    def _detect_dependency_bottlenecks(self, dependency_data: Optional[DependencyGraph], 
                                     workflow_id: str = None) -> List[BottleneckDetection]:
        """Detect dependency chain bottlenecks."""
        bottlenecks = []
        
        if not dependency_data or not dependency_data.critical_path:
            return bottlenecks
        
        # Analyze critical path length
        critical_path_length = len(dependency_data.critical_path)
        total_tasks = len(dependency_data.nodes)
        
        if critical_path_length > total_tasks * 0.6:  # More than 60% of tasks in critical path
            impact_score = min(1.0, critical_path_length / total_tasks)
            
            bottlenecks.append(BottleneckDetection(
                bottleneck_id=f"dep_chain_{workflow_id or 'global'}",
                bottleneck_type=BottleneckType.DEPENDENCY_CHAIN,
                severity=self._calculate_severity(impact_score),
                component="dependency_chain",
                description=f"Long dependency chain detected: {critical_path_length} tasks in critical path",
                impact_score=impact_score,
                frequency=1.0,  # Always present if detected
                location="workflow_dependencies",
                affected_workflows=[workflow_id] if workflow_id else ["all"],
                performance_degradation=impact_score * 50,  # Estimated percentage
                detection_confidence=0.9,
                detected_at=datetime.now(),
                metadata={
                    'critical_path_length': critical_path_length,
                    'total_tasks': total_tasks,
                    'critical_path_ratio': critical_path_length / total_tasks
                }
            ))
        
        # Detect serialization points (tasks with many dependents)
        node_map = {node.task_id: node for node in dependency_data.nodes}
        for node in dependency_data.nodes:
            dependent_count = len(node.dependents)
            if dependent_count > 3:  # Task with many dependents
                impact_score = min(1.0, dependent_count / total_tasks)
                
                bottlenecks.append(BottleneckDetection(
                    bottleneck_id=f"serial_point_{node.task_id}",
                    bottleneck_type=BottleneckType.SERIALIZATION_POINT,
                    severity=self._calculate_severity(impact_score),
                    component=node.task_id,
                    description=f"Serialization point: {node.task_id} has {dependent_count} dependents",
                    impact_score=impact_score,
                    frequency=0.8,
                    location=node.task_id,
                    affected_workflows=[workflow_id] if workflow_id else ["all"],
                    performance_degradation=impact_score * 30,
                    detection_confidence=0.8,
                    detected_at=datetime.now(),
                    metadata={
                        'dependent_count': dependent_count,
                        'dependents': node.dependents
                    }
                ))
        
        return bottlenecks
    
    def _detect_resource_bottlenecks(self, flow_analysis: Optional[FlowAnalysis], 
                                   workflow_id: str = None) -> List[BottleneckDetection]:
        """Detect resource contention bottlenecks."""
        bottlenecks = []
        
        if not flow_analysis:
            return bottlenecks
        
        # Analyze resource utilization from flow analysis
        parallelism_analysis = flow_analysis.parallelism_analysis
        
        # Check for low parallelism efficiency
        parallelism_efficiency = parallelism_analysis.get('parallelism_efficiency', 1.0)
        if parallelism_efficiency < 0.5:
            impact_score = 1.0 - parallelism_efficiency
            
            bottlenecks.append(BottleneckDetection(
                bottleneck_id=f"resource_contention_{workflow_id or 'global'}",
                bottleneck_type=BottleneckType.RESOURCE_CONTENTION,
                severity=self._calculate_severity(impact_score),
                component="resource_pool",
                description=f"Low parallelism efficiency: {parallelism_efficiency:.1%}",
                impact_score=impact_score,
                frequency=0.7,
                location="resource_management",
                affected_workflows=[workflow_id] if workflow_id else ["all"],
                performance_degradation=impact_score * 40,
                detection_confidence=0.8,
                detected_at=datetime.now(),
                metadata={
                    'parallelism_efficiency': parallelism_efficiency,
                    'max_concurrent_threads': parallelism_analysis.get('max_concurrent_threads', 1),
                    'avg_concurrent_threads': parallelism_analysis.get('avg_concurrent_threads', 1)
                }
            ))
        
        return bottlenecks
    
    def _detect_serialization_bottlenecks(self, flow_analysis: Optional[FlowAnalysis], 
                                        workflow_id: str = None) -> List[BottleneckDetection]:
        """Detect serialization point bottlenecks."""
        bottlenecks = []
        
        if not flow_analysis or not flow_analysis.critical_paths:
            return bottlenecks
        
        # Analyze critical paths for serialization points
        for path in flow_analysis.critical_paths:
            if path.total_duration_ms > 2000:  # Paths longer than 2 seconds
                impact_score = min(1.0, path.total_duration_ms / 10000)  # Normalize to 10 seconds
                
                bottlenecks.append(BottleneckDetection(
                    bottleneck_id=f"slow_path_{path.path_id}",
                    bottleneck_type=BottleneckType.CRITICAL_PATH,
                    severity=self._calculate_severity(impact_score),
                    component="execution_path",
                    description=f"Slow critical path: {path.total_duration_ms:.1f}ms",
                    impact_score=impact_score,
                    frequency=0.6,
                    location="critical_path",
                    affected_workflows=[workflow_id] if workflow_id else ["all"],
                    performance_degradation=impact_score * 35,
                    detection_confidence=0.7,
                    detected_at=datetime.now(),
                    metadata={
                        'path_duration_ms': path.total_duration_ms,
                        'node_count': len(path.nodes),
                        'bottleneck_nodes': path.bottleneck_nodes
                    }
                ))
        
        return bottlenecks
    
    def _detect_hotspot_bottlenecks(self, flow_analysis: Optional[FlowAnalysis], 
                                  workflow_id: str = None) -> List[BottleneckDetection]:
        """Detect performance hotspots."""
        bottlenecks = []
        
        if not flow_analysis or not flow_analysis.bottleneck_components:
            return bottlenecks
        
        # Analyze identified bottleneck components
        for component in flow_analysis.bottleneck_components:
            # Calculate impact based on component frequency in analysis
            impact_score = 0.8  # High impact for identified bottlenecks
            
            bottlenecks.append(BottleneckDetection(
                bottleneck_id=f"hotspot_{component}",
                bottleneck_type=BottleneckType.HOT_SPOT,
                severity=self._calculate_severity(impact_score),
                component=component,
                description=f"Performance hotspot detected in {component}",
                impact_score=impact_score,
                frequency=0.9,
                location=component,
                affected_workflows=[workflow_id] if workflow_id else ["all"],
                performance_degradation=impact_score * 45,
                detection_confidence=0.9,
                detected_at=datetime.now(),
                metadata={
                    'component': component,
                    'analysis_source': 'flow_analyzer'
                }
            ))
        
        return bottlenecks
    
    def _detect_queue_bottlenecks(self, flow_analysis: Optional[FlowAnalysis], 
                                workflow_id: str = None) -> List[BottleneckDetection]:
        """Detect queue buildup bottlenecks."""
        bottlenecks = []
        
        # Simulate queue analysis from shared state
        queue_metrics = self._get_queue_metrics()
        
        for queue_name, metrics in queue_metrics.items():
            avg_depth = metrics.get('avg_depth', 0)
            max_depth = metrics.get('max_depth', 0)
            
            if avg_depth > 10 or max_depth > 20:
                impact_score = min(1.0, avg_depth / 50)
                
                bottlenecks.append(BottleneckDetection(
                    bottleneck_id=f"queue_buildup_{queue_name}",
                    bottleneck_type=BottleneckType.QUEUE_BUILDUP,
                    severity=self._calculate_severity(impact_score),
                    component=queue_name,
                    description=f"Queue buildup in {queue_name}: avg {avg_depth}, max {max_depth}",
                    impact_score=impact_score,
                    frequency=0.6,
                    location=queue_name,
                    affected_workflows=[workflow_id] if workflow_id else ["all"],
                    performance_degradation=impact_score * 25,
                    detection_confidence=0.8,
                    detected_at=datetime.now(),
                    metadata={
                        'avg_queue_depth': avg_depth,
                        'max_queue_depth': max_depth,
                        'queue_name': queue_name
                    }
                ))
        
        return bottlenecks
    
    def _get_queue_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get queue metrics from shared state or telemetry."""
        # Simulate queue metrics
        return {
            'test_generation_queue': {
                'avg_depth': 5,
                'max_depth': 15,
                'throughput': 2.5
            },
            'execution_queue': {
                'avg_depth': 12,
                'max_depth': 25,
                'throughput': 1.8
            }
        }
    
    def _calculate_severity(self, impact_score: float) -> BottleneckSeverity:
        """Calculate bottleneck severity based on impact score."""
        if impact_score >= 0.8:
            return BottleneckSeverity.CRITICAL
        elif impact_score >= 0.6:
            return BottleneckSeverity.HIGH
        elif impact_score >= 0.3:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW


class BottleneckResolver:
    """Generates resolution strategies for detected bottlenecks."""
    
    def __init__(self):
        self.resolution_strategies = self._initialize_resolution_strategies()
        self.shared_state = get_shared_state()
        
        print("Bottleneck Resolver initialized")
        print(f"   Resolution strategies: {len(self.resolution_strategies)}")
    
    def _initialize_resolution_strategies(self) -> Dict[BottleneckType, List[ResolutionStrategy]]:
        """Initialize resolution strategies for each bottleneck type."""
        return {
            BottleneckType.DEPENDENCY_CHAIN: [
                ResolutionStrategy.PARALLEL_EXECUTION,
                ResolutionStrategy.DEPENDENCY_OPTIMIZATION,
                ResolutionStrategy.ASYNC_PROCESSING
            ],
            BottleneckType.RESOURCE_CONTENTION: [
                ResolutionStrategy.RESOURCE_SCALING,
                ResolutionStrategy.LOAD_BALANCING,
                ResolutionStrategy.QUEUE_OPTIMIZATION
            ],
            BottleneckType.SERIALIZATION_POINT: [
                ResolutionStrategy.PARALLEL_EXECUTION,
                ResolutionStrategy.ASYNC_PROCESSING,
                ResolutionStrategy.BATCHING_OPTIMIZATION
            ],
            BottleneckType.HOT_SPOT: [
                ResolutionStrategy.CACHING_STRATEGY,
                ResolutionStrategy.RESOURCE_SCALING,
                ResolutionStrategy.LOAD_BALANCING
            ],
            BottleneckType.QUEUE_BUILDUP: [
                ResolutionStrategy.QUEUE_OPTIMIZATION,
                ResolutionStrategy.PARALLEL_EXECUTION,
                ResolutionStrategy.RESOURCE_SCALING
            ],
            BottleneckType.CRITICAL_PATH: [
                ResolutionStrategy.PARALLEL_EXECUTION,
                ResolutionStrategy.DEPENDENCY_OPTIMIZATION,
                ResolutionStrategy.CACHING_STRATEGY
            ]
        }
    
    def generate_resolution_plan(self, bottleneck: BottleneckDetection) -> BottleneckResolution:
        """Generate a comprehensive resolution plan for a bottleneck."""
        
        # Get applicable strategies
        strategies = self.resolution_strategies.get(bottleneck.bottleneck_type, [])
        
        # Generate resolution actions for each strategy
        actions = []
        for strategy in strategies:
            action = self._create_resolution_action(strategy, bottleneck)
            if action:
                actions.append(action)
        
        # Select primary action (highest success probability * improvement)
        if actions:
            actions.sort(key=lambda x: x.success_probability * x.estimated_improvement, reverse=True)
            primary_action = actions[0]
            alternative_actions = actions[1:]
        else:
            # Create fallback action
            primary_action = self._create_fallback_action(bottleneck)
            alternative_actions = []
        
        # Calculate resolution priority
        resolution_priority = self._calculate_resolution_priority(bottleneck, primary_action)
        
        # Estimate total improvement
        total_improvement = primary_action.estimated_improvement
        if alternative_actions:
            # Add diminishing returns from alternatives
            for i, action in enumerate(alternative_actions[:2]):  # Consider top 2 alternatives
                total_improvement += action.estimated_improvement * (0.5 ** (i + 1))
        
        # Generate implementation plan
        implementation_plan = self._generate_implementation_plan(primary_action, bottleneck)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(bottleneck, primary_action)
        
        # Risk assessment
        risk_assessment = self._assess_risks(primary_action, bottleneck)
        
        return BottleneckResolution(
            resolution_id=f"resolution_{bottleneck.bottleneck_id}_{int(datetime.now().timestamp())}",
            bottleneck_id=bottleneck.bottleneck_id,
            primary_action=primary_action,
            alternative_actions=alternative_actions,
            resolution_priority=resolution_priority,
            estimated_total_improvement=min(100.0, total_improvement),
            risk_assessment=risk_assessment,
            implementation_plan=implementation_plan,
            success_metrics=success_metrics,
            created_at=datetime.now()
        )
    
    def _create_resolution_action(self, strategy: ResolutionStrategy, 
                                bottleneck: BottleneckDetection) -> Optional[ResolutionAction]:
        """Create a resolution action for a specific strategy."""
        
        action_id = f"action_{strategy.value}_{bottleneck.bottleneck_id}"
        
        if strategy == ResolutionStrategy.PARALLEL_EXECUTION:
            return ResolutionAction(
                action_id=action_id,
                strategy=strategy,
                description=f"Implement parallel execution for {bottleneck.component}",
                parameters={
                    'max_workers': min(8, max(2, int(bottleneck.impact_score * 8))),
                    'chunk_size': 'auto',
                    'execution_mode': 'thread_pool'
                },
                estimated_improvement=bottleneck.impact_score * 60,
                implementation_complexity=0.6,
                resource_requirements={'cpu_cores': 2, 'memory_mb': 1024},
                success_probability=0.8,
                side_effects=['Increased memory usage', 'Coordination overhead'],
                execution_time_estimate=30.0
            )
        
        elif strategy == ResolutionStrategy.RESOURCE_SCALING:
            return ResolutionAction(
                action_id=action_id,
                strategy=strategy,
                description=f"Scale resources for {bottleneck.component}",
                parameters={
                    'scale_factor': min(3.0, 1.0 + bottleneck.impact_score * 2),
                    'resource_type': 'compute',
                    'scaling_mode': 'horizontal'
                },
                estimated_improvement=bottleneck.impact_score * 50,
                implementation_complexity=0.4,
                resource_requirements={'budget_impact': 'medium'},
                success_probability=0.9,
                side_effects=['Increased costs', 'Resource coordination'],
                execution_time_estimate=15.0
            )
        
        elif strategy == ResolutionStrategy.CACHING_STRATEGY:
            return ResolutionAction(
                action_id=action_id,
                strategy=strategy,
                description=f"Implement caching for {bottleneck.component}",
                parameters={
                    'cache_type': 'intelligent_cache',
                    'cache_size': '1GB',
                    'ttl_seconds': 3600,
                    'cache_hit_ratio_target': 0.8
                },
                estimated_improvement=bottleneck.impact_score * 70,
                implementation_complexity=0.5,
                resource_requirements={'memory_mb': 1024, 'storage_mb': 2048},
                success_probability=0.85,
                side_effects=['Memory overhead', 'Cache invalidation complexity'],
                execution_time_estimate=45.0
            )
        
        elif strategy == ResolutionStrategy.QUEUE_OPTIMIZATION:
            return ResolutionAction(
                action_id=action_id,
                strategy=strategy,
                description=f"Optimize queue processing for {bottleneck.component}",
                parameters={
                    'queue_type': 'priority_queue',
                    'batch_size': min(50, max(5, int(bottleneck.impact_score * 50))),
                    'processing_mode': 'batch_parallel'
                },
                estimated_improvement=bottleneck.impact_score * 40,
                implementation_complexity=0.6,
                resource_requirements={'cpu_cores': 1, 'memory_mb': 512},
                success_probability=0.75,
                side_effects=['Batching delays', 'Memory usage for batching'],
                execution_time_estimate=25.0
            )
        
        elif strategy == ResolutionStrategy.DEPENDENCY_OPTIMIZATION:
            return ResolutionAction(
                action_id=action_id,
                strategy=strategy,
                description=f"Optimize dependencies for {bottleneck.component}",
                parameters={
                    'optimization_type': 'redundancy_removal',
                    'parallel_branches': True,
                    'lazy_loading': True
                },
                estimated_improvement=bottleneck.impact_score * 45,
                implementation_complexity=0.7,
                resource_requirements={'development_time': 'high'},
                success_probability=0.7,
                side_effects=['Code complexity', 'Testing overhead'],
                execution_time_estimate=60.0
            )
        
        else:
            # Generic action for other strategies
            return ResolutionAction(
                action_id=action_id,
                strategy=strategy,
                description=f"Apply {strategy.value} to {bottleneck.component}",
                parameters={'generic': True},
                estimated_improvement=bottleneck.impact_score * 30,
                implementation_complexity=0.5,
                resource_requirements={},
                success_probability=0.6,
                side_effects=['Various implementation-specific effects'],
                execution_time_estimate=30.0
            )
    
    def _create_fallback_action(self, bottleneck: BottleneckDetection) -> ResolutionAction:
        """Create a fallback resolution action."""
        return ResolutionAction(
            action_id=f"fallback_{bottleneck.bottleneck_id}",
            strategy=ResolutionStrategy.RESOURCE_SCALING,
            description=f"Generic optimization for {bottleneck.component}",
            parameters={'optimization_level': 'basic'},
            estimated_improvement=bottleneck.impact_score * 20,
            implementation_complexity=0.3,
            resource_requirements={},
            success_probability=0.5,
            side_effects=['Minimal impact'],
            execution_time_estimate=20.0
        )
    
    def _calculate_resolution_priority(self, bottleneck: BottleneckDetection, 
                                     action: ResolutionAction) -> float:
        """Calculate resolution priority (0-1)."""
        # Factors: impact, severity, success probability, implementation complexity
        impact_factor = bottleneck.impact_score
        severity_factor = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}[bottleneck.severity.value]
        success_factor = action.success_probability
        complexity_factor = 1.0 - action.implementation_complexity
        
        priority = (impact_factor * 0.3 + severity_factor * 0.3 + 
                   success_factor * 0.2 + complexity_factor * 0.2)
        
        return round(priority, 3)
    
    def _generate_implementation_plan(self, action: ResolutionAction, 
                                    bottleneck: BottleneckDetection) -> List[str]:
        """Generate step-by-step implementation plan."""
        plan = [
            f"1. Analyze current state of {bottleneck.component}",
            f"2. Prepare resources for {action.strategy.value} implementation",
            f"3. Implement {action.description}",
            f"4. Configure parameters: {action.parameters}",
            f"5. Test implementation in controlled environment",
            f"6. Monitor performance metrics",
            f"7. Deploy to production with rollback plan",
            f"8. Validate success metrics and optimization results"
        ]
        
        return plan
    
    def _define_success_metrics(self, bottleneck: BottleneckDetection, 
                              action: ResolutionAction) -> List[str]:
        """Define success metrics for resolution."""
        base_metrics = [
            f"Reduce {bottleneck.bottleneck_type.value} impact by {action.estimated_improvement:.1f}%",
            f"Improve overall performance score by at least {action.estimated_improvement * 0.5:.1f}%",
            f"Maintain system stability with no critical errors"
        ]
        
        # Add specific metrics based on bottleneck type
        if bottleneck.bottleneck_type == BottleneckType.QUEUE_BUILDUP:
            base_metrics.append("Reduce average queue depth by 50%")
        elif bottleneck.bottleneck_type == BottleneckType.RESOURCE_CONTENTION:
            base_metrics.append("Increase parallelism efficiency to >70%")
        elif bottleneck.bottleneck_type == BottleneckType.DEPENDENCY_CHAIN:
            base_metrics.append("Reduce critical path length by 30%")
        
        return base_metrics
    
    def _assess_risks(self, action: ResolutionAction, bottleneck: BottleneckDetection) -> Dict[str, Any]:
        """Assess risks associated with the resolution action."""
        
        # Calculate overall risk score
        complexity_risk = action.implementation_complexity
        failure_risk = 1.0 - action.success_probability
        side_effect_risk = len(action.side_effects) * 0.1
        
        overall_risk = (complexity_risk + failure_risk + side_effect_risk) / 3.0
        
        # Risk categories
        risk_level = "low"
        if overall_risk > 0.7:
            risk_level = "high"
        elif overall_risk > 0.4:
            risk_level = "medium"
        
        return {
            'overall_risk_score': round(overall_risk, 3),
            'risk_level': risk_level,
            'complexity_risk': complexity_risk,
            'failure_risk': failure_risk,
            'side_effect_risk': side_effect_risk,
            'mitigation_strategies': [
                'Implement gradual rollout',
                'Maintain rollback capability',
                'Monitor key metrics during implementation',
                'Have contingency plans for each identified side effect'
            ],
            'rollback_plan': f"Revert {action.strategy.value} configuration and restore previous state"
        }


class BottleneckDetectionResolutionAgent:
    """Main bottleneck detection and resolution agent."""
    
    def __init__(self, coordinator: AgentCoordinator = None):
        self.coordinator = coordinator
        self.shared_state = get_shared_state()
        
        # Initialize components
        self.detector = BottleneckDetector()
        self.resolver = BottleneckResolver()
        
        # Agent state
        self.detected_bottlenecks: Dict[str, BottleneckDetection] = {}
        self.resolution_plans: Dict[str, BottleneckResolution] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.bottlenecks_detected = 0
        self.resolutions_created = 0
        self.resolutions_executed = 0
        self.success_rate = 0.0
        
        # Register with coordinator if provided
        if self.coordinator:
            self.coordinator.register_agent(
                "bottleneck_detection_resolution_agent",
                AgentRole.PERFORMANCE_OPTIMIZER,
                weight=1.2,
                specialization=["bottleneck_detection", "performance_resolution", "workflow_optimization"]
            )
        
        print("Bottleneck Detection & Resolution Agent initialized")
        print("   Components: detector, resolver")
    
    def analyze_and_resolve_bottlenecks(self, workflow_id: str = None) -> Dict[str, Any]:
        """Analyze bottlenecks and generate resolution plans."""
        
        print(f"\nBottleneck Analysis: {workflow_id or 'All Workflows'}")
        
        # Detect bottlenecks
        bottlenecks = self.detector.detect_bottlenecks(workflow_id)
        
        # Store detected bottlenecks
        for bottleneck in bottlenecks:
            self.detected_bottlenecks[bottleneck.bottleneck_id] = bottleneck
        
        # Generate resolution plans
        resolution_plans = []
        for bottleneck in bottlenecks:
            resolution_plan = self.resolver.generate_resolution_plan(bottleneck)
            self.resolution_plans[resolution_plan.resolution_id] = resolution_plan
            resolution_plans.append(resolution_plan)
        
        # Sort by priority
        resolution_plans.sort(key=lambda x: x.resolution_priority, reverse=True)
        
        # Coordinate with other agents for consensus if needed
        consensus_results = []
        if self.coordinator and bottlenecks:
            for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
                consensus_result = self._coordinate_resolution_consensus(bottleneck, workflow_id)
                if consensus_result:
                    consensus_results.append(consensus_result)
        
        # Update statistics
        self.bottlenecks_detected += len(bottlenecks)
        self.resolutions_created += len(resolution_plans)
        
        # Store results in shared state
        self._store_analysis_results(workflow_id, bottlenecks, resolution_plans)
        
        result = {
            'workflow_id': workflow_id,
            'bottlenecks_detected': len(bottlenecks),
            'bottlenecks': [self._bottleneck_to_dict(b) for b in bottlenecks],
            'resolution_plans': [self._resolution_to_dict(r) for r in resolution_plans],
            'consensus_results': consensus_results,
            'recommendations': self._generate_recommendations(bottlenecks, resolution_plans),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        print(f"   Bottlenecks detected: {len(bottlenecks)}")
        print(f"   Resolution plans created: {len(resolution_plans)}")
        
        return result
    
    def execute_resolution(self, resolution_id: str) -> Dict[str, Any]:
        """Execute a resolution plan."""
        
        if resolution_id not in self.resolution_plans:
            return {'success': False, 'error': 'Resolution plan not found'}
        
        resolution = self.resolution_plans[resolution_id]
        action = resolution.primary_action
        
        print(f"Executing resolution: {action.description}")
        
        # Simulate execution (in real implementation, this would actually execute the action)
        execution_start = datetime.now()
        
        # Simulate execution time
        time.sleep(min(2.0, action.execution_time_estimate / 30.0))  # Scaled down for testing
        
        # Simulate execution results
        success_probability = action.success_probability
        success = success_probability > 0.5  # Simplified success determination
        
        execution_results = {
            'success': success,
            'execution_time_minutes': action.execution_time_estimate,
            'actual_improvement': action.estimated_improvement * (0.8 if success else 0.2),
            'side_effects_observed': action.side_effects[:2] if success else action.side_effects,
            'metrics_improved': resolution.success_metrics[:3] if success else [],
            'execution_start': execution_start.isoformat(),
            'execution_end': datetime.now().isoformat()
        }
        
        # Update resolution
        resolution.executed = True
        resolution.execution_results = execution_results
        
        # Update statistics
        self.resolutions_executed += 1
        if success:
            self.success_rate = (self.success_rate * (self.resolutions_executed - 1) + 1.0) / self.resolutions_executed
        else:
            self.success_rate = (self.success_rate * (self.resolutions_executed - 1) + 0.0) / self.resolutions_executed
        
        # Store execution history
        self.execution_history.append({
            'resolution_id': resolution_id,
            'action_strategy': action.strategy.value,
            'success': success,
            'improvement': execution_results['actual_improvement'],
            'execution_time': execution_start
        })
        
        print(f"   Execution {'successful' if success else 'failed'}")
        print(f"   Actual improvement: {execution_results['actual_improvement']:.1f}%")
        
        return execution_results
    
    def _coordinate_resolution_consensus(self, bottleneck: BottleneckDetection, 
                                       workflow_id: str = None) -> Optional[Dict[str, Any]]:
        """Coordinate with other agents for resolution consensus."""
        
        try:
            # Create coordination task
            task_id = self.coordinator.create_coordination_task(
                description=f"Resolve {bottleneck.bottleneck_type.value} in {bottleneck.component}",
                required_roles={AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.QUALITY_ASSESSOR},
                context={
                    'bottleneck_type': bottleneck.bottleneck_type.value,
                    'severity': bottleneck.severity.value,
                    'impact_score': bottleneck.impact_score,
                    'component': bottleneck.component
                }
            )
            
            # Submit resolution assessment vote
            self.coordinator.submit_vote(
                task_id=task_id,
                agent_id="bottleneck_detection_resolution_agent",
                choice=bottleneck.impact_score,
                confidence=bottleneck.detection_confidence,
                reasoning=f"Detected {bottleneck.bottleneck_type.value} with {bottleneck.impact_score:.1%} impact"
            )
            
            print(f"   Consensus requested for {bottleneck.component} bottleneck")
            
            # Wait for consensus (in real implementation, this would be event-driven)
            time.sleep(1)
            
            result = self.coordinator.get_coordination_result(task_id)
            return result.to_dict() if result else None
            
        except Exception as e:
            print(f"Resolution consensus coordination failed: {e}")
            return None
    
    def _store_analysis_results(self, workflow_id: str, bottlenecks: List[BottleneckDetection], 
                              resolution_plans: List[BottleneckResolution]):
        """Store analysis results in shared state."""
        
        analysis_data = {
            'bottlenecks_count': len(bottlenecks),
            'resolution_plans_count': len(resolution_plans),
            'highest_impact': max([b.impact_score for b in bottlenecks]) if bottlenecks else 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Store in shared state
        key = f"bottleneck_analysis_{workflow_id or 'global'}"
        self.shared_state.set(key, analysis_data)
        
        # Cache results
        cache_test_result(key, analysis_data, len(bottlenecks) * 10)
    
    def _bottleneck_to_dict(self, bottleneck: BottleneckDetection) -> Dict[str, Any]:
        """Convert bottleneck to dictionary."""
        return {
            'bottleneck_id': bottleneck.bottleneck_id,
            'type': bottleneck.bottleneck_type.value,
            'severity': bottleneck.severity.value,
            'component': bottleneck.component,
            'description': bottleneck.description,
            'impact_score': bottleneck.impact_score,
            'performance_degradation': bottleneck.performance_degradation,
            'confidence': bottleneck.detection_confidence,
            'detected_at': bottleneck.detected_at.isoformat()
        }
    
    def _resolution_to_dict(self, resolution: BottleneckResolution) -> Dict[str, Any]:
        """Convert resolution to dictionary."""
        return {
            'resolution_id': resolution.resolution_id,
            'bottleneck_id': resolution.bottleneck_id,
            'primary_strategy': resolution.primary_action.strategy.value,
            'description': resolution.primary_action.description,
            'estimated_improvement': resolution.estimated_total_improvement,
            'priority': resolution.resolution_priority,
            'complexity': resolution.primary_action.implementation_complexity,
            'success_probability': resolution.primary_action.success_probability,
            'executed': resolution.executed
        }
    
    def _generate_recommendations(self, bottlenecks: List[BottleneckDetection], 
                                resolution_plans: List[BottleneckResolution]) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        if not bottlenecks:
            recommendations.append("No significant bottlenecks detected - system performance is optimal")
            return recommendations
        
        # High-impact bottlenecks
        critical_bottlenecks = [b for b in bottlenecks if b.severity == BottleneckSeverity.CRITICAL]
        if critical_bottlenecks:
            recommendations.append(f"Address {len(critical_bottlenecks)} critical bottlenecks immediately")
        
        # Most common bottleneck types
        type_counts = defaultdict(int)
        for bottleneck in bottlenecks:
            type_counts[bottleneck.bottleneck_type] += 1
        
        if type_counts:
            most_common = max(type_counts, key=type_counts.get)
            recommendations.append(f"Focus on {most_common.value} optimization - appears in {type_counts[most_common]} components")
        
        # High-priority resolutions
        high_priority_resolutions = [r for r in resolution_plans if r.resolution_priority > 0.7]
        if high_priority_resolutions:
            recommendations.append(f"Execute {len(high_priority_resolutions)} high-priority resolution plans")
        
        # Resource optimization
        resource_bottlenecks = [b for b in bottlenecks if b.bottleneck_type == BottleneckType.RESOURCE_CONTENTION]
        if resource_bottlenecks:
            recommendations.append("Consider resource scaling or load balancing improvements")
        
        return recommendations
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics."""
        
        active_bottlenecks = len([b for b in self.detected_bottlenecks.values() 
                                if b.severity in [BottleneckSeverity.HIGH, BottleneckSeverity.CRITICAL]])
        
        return {
            'bottlenecks_detected': self.bottlenecks_detected,
            'active_bottlenecks': active_bottlenecks,
            'resolutions_created': self.resolutions_created,
            'resolutions_executed': self.resolutions_executed,
            'success_rate': round(self.success_rate, 3),
            'recent_executions': len(self.execution_history),
            'detector_thresholds': {
                'impact_threshold': self.detector.impact_threshold,
                'confidence_threshold': self.detector.confidence_threshold
            }
        }


def test_bottleneck_detection_resolution():
    """Test the bottleneck detection and resolution agent."""
    print("\n" + "="*60)
    print("Testing Bottleneck Detection & Resolution Agent")
    print("="*60)
    
    # Create agent
    agent = BottleneckDetectionResolutionAgent()
    
    # Test bottleneck analysis
    print("\n1. Testing bottleneck analysis...")
    analysis_result = agent.analyze_and_resolve_bottlenecks("test_workflow")
    
    print(f"   Bottlenecks detected: {analysis_result['bottlenecks_detected']}")
    print(f"   Resolution plans: {len(analysis_result['resolution_plans'])}")
    
    # Show detected bottlenecks
    if analysis_result['bottlenecks']:
        print("\n2. Detected bottlenecks:")
        for i, bottleneck in enumerate(analysis_result['bottlenecks'][:3], 1):
            print(f"   {i}. {bottleneck['type']} in {bottleneck['component']} ({bottleneck['severity']})")
            print(f"      Impact: {bottleneck['impact_score']:.1%}, Confidence: {bottleneck['confidence']:.1%}")
    
    # Show resolution plans
    if analysis_result['resolution_plans']:
        print("\n3. Resolution plans:")
        for i, plan in enumerate(analysis_result['resolution_plans'][:3], 1):
            print(f"   {i}. {plan['primary_strategy']} - {plan['description']}")
            print(f"      Improvement: {plan['estimated_improvement']:.1f}%, Priority: {plan['priority']:.3f}")
    
    # Test resolution execution
    if analysis_result['resolution_plans']:
        print("\n4. Testing resolution execution...")
        resolution_id = analysis_result['resolution_plans'][0]['resolution_id']
        execution_result = agent.execute_resolution(resolution_id)
        
        print(f"   Execution success: {execution_result['success']}")
        print(f"   Actual improvement: {execution_result['actual_improvement']:.1f}%")
    
    # Test agent status
    print("\n5. Agent status:")
    status = agent.get_agent_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\nBottleneck Detection & Resolution Agent test completed successfully!")
    return True


if __name__ == "__main__":
    test_bottleneck_detection_resolution()