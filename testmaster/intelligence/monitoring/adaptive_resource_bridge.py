"""
Adaptive Resource Management Bridge for TestMaster Deep Integration

This module bridges the Adaptive Resource Management Agent with the existing
flow optimizer resource infrastructure, creating a unified resource management
system that combines traditional optimization with AI-driven intelligence.

Deep Integration Features:
- Seamless connection between AI agent and existing resource optimizer
- Shared resource pool management with intelligent prediction
- Consensus-driven scaling decisions combined with algorithmic optimization
- Unified cost optimization across both systems
- Cross-module communication through shared state management
"""

import threading
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .adaptive_resource_management_agent import (
    AdaptiveResourceManagementAgent,
    ResourceStrategy,
    ScalingDecision,
    ResourceMetrics,
    AdaptiveConfiguration,
    ScalingDirection,
    ScalingTrigger
)
from ...flow_optimizer.resource_optimizer import (
    ResourceOptimizer,
    ResourceType,
    OptimizationPolicy,
    ResourcePool,
    ResourceAllocation,
    ResourceRequirement
)
from ...core.feature_flags import FeatureFlags
from ...core.shared_state import SharedState


class BridgeMode(Enum):
    """Bridge operation modes."""
    AGENT_ONLY = "agent_only"
    OPTIMIZER_ONLY = "optimizer_only"
    HYBRID = "hybrid"
    CONSENSUS = "consensus"


@dataclass
class BridgeConfiguration:
    """Configuration for resource management bridge."""
    mode: BridgeMode = BridgeMode.HYBRID
    agent_weight: float = 0.6
    optimizer_weight: float = 0.4
    consensus_threshold: float = 0.7
    sync_interval: int = 30
    enable_cross_optimization: bool = True
    enable_predictive_scaling: bool = True


@dataclass
class UnifiedResourceDecision:
    """Unified resource management decision from bridge."""
    allocation: ResourceAllocation
    scaling_decision: ScalingDecision
    consensus_score: float
    decision_method: str
    optimization_metrics: Dict[str, float]
    timestamp: float


class AdaptiveResourceBridge:
    """Bridge between AI agent and traditional resource optimizer."""
    
    def __init__(self, config: BridgeConfiguration = None):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'adaptive_resource_bridge')
        self.config = config or BridgeConfiguration()
        self.lock = threading.RLock()
        
        # Initialize components
        self.agent = AdaptiveResourceManagementAgent()
        self.optimizer = ResourceOptimizer()
        self.shared_state = SharedState()
        
        # Bridge state
        self.unified_decisions: Dict[str, UnifiedResourceDecision] = {}
        self.resource_pools_sync: Dict[str, float] = {}
        self.prediction_cache: Dict[str, Any] = {}
        self.consensus_history: List[Dict[str, Any]] = []
        
        # Synchronization
        self.sync_thread = None
        self.running = False
        
        if not self.enabled:
            return
        
        self._initialize_resource_sync()
        self._start_sync_thread()
        
        print("Adaptive Resource Bridge initialized")
        print(f"   Bridge mode: {self.config.mode.value}")
        print(f"   Agent weight: {self.config.agent_weight}")
        print(f"   Optimizer weight: {self.config.optimizer_weight}")
        print(f"   Cross-optimization: {self.config.enable_cross_optimization}")
    
    def _initialize_resource_sync(self):
        """Initialize synchronization between agent and optimizer resource pools."""
        # Sync existing optimizer pools to agent
        optimizer_pools = self.optimizer.get_resource_utilization()
        for pool_id, pool_data in optimizer_pools.items():
            self.resource_pools_sync[pool_id] = time.time()
            
            # Share pool information with agent
            self.shared_state.set(f"resource_pool_{pool_id}", {
                "type": pool_data["resource_type"],
                "utilization": pool_data["utilization_ratio"],
                "capacity": pool_data["total_capacity"],
                "available": pool_data["available_capacity"],
                "source": "optimizer"
            })
        
        print(f"Synchronized {len(optimizer_pools)} resource pools")
    
    def _start_sync_thread(self):
        """Start background synchronization thread."""
        if self.sync_thread and self.sync_thread.is_alive():
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
    
    def _sync_worker(self):
        """Background worker for continuous synchronization."""
        while self.running:
            try:
                self._perform_sync()
                time.sleep(self.config.sync_interval)
            except Exception as e:
                print(f"Bridge sync error: {e}")
                time.sleep(5)
    
    def _perform_sync(self):
        """Perform synchronization between agent and optimizer."""
        with self.lock:
            # Sync resource utilization data
            optimizer_utilization = self.optimizer.get_resource_utilization()
            agent_utilization = self.agent.get_current_utilization()
            
            # Update shared state with latest utilization
            self.shared_state.set("unified_resource_utilization", {
                "optimizer": optimizer_utilization,
                "agent": agent_utilization,
                "sync_timestamp": time.time()
            })
            
            # Sync predictions if enabled
            if self.config.enable_predictive_scaling:
                self._sync_predictions()
            
            # Cross-optimize if enabled
            if self.config.enable_cross_optimization:
                self._perform_cross_optimization(optimizer_utilization, agent_utilization)
    
    def _sync_predictions(self):
        """Synchronize predictive data between systems."""
        # Get agent predictions
        agent_predictions = self.agent.get_resource_predictions()
        
        # Share predictions with optimizer for future allocations
        for resource_type, prediction in agent_predictions.items():
            self.shared_state.set(f"prediction_{resource_type}", {
                "predicted_demand": prediction.predicted_demand,
                "confidence": prediction.confidence,
                "time_horizon": prediction.time_horizon,
                "source": "agent_predictor"
            })
    
    def _perform_cross_optimization(self, optimizer_util: Dict[str, Any], agent_util: Dict[str, Any]):
        """Perform cross-optimization between systems."""
        # Identify optimization opportunities
        for resource_type in ["cpu", "memory", "network", "storage"]:
            opt_pools = [p for p in optimizer_util.values() if p.get("resource_type") == resource_type]
            agent_metrics = agent_util.get(resource_type, {})
            
            if opt_pools and agent_metrics:
                # Suggest optimizer pool adjustments based on agent insights
                avg_utilization = sum(p["utilization_ratio"] for p in opt_pools) / len(opt_pools)
                agent_prediction = agent_metrics.get("predicted_utilization", avg_utilization)
                
                if abs(agent_prediction - avg_utilization) > 0.2:  # Significant difference
                    self._suggest_pool_optimization(resource_type, opt_pools, agent_prediction)
    
    def _suggest_pool_optimization(self, resource_type: str, pools: List[Dict], predicted_utilization: float):
        """Suggest pool capacity optimizations."""
        for pool_data in pools:
            pool_id = f"{resource_type}_pool_1"  # Assuming standard naming
            current_capacity = pool_data["total_capacity"]
            
            # Calculate recommended capacity based on prediction
            if predicted_utilization > 0.8:  # High utilization predicted
                recommended_capacity = current_capacity * 1.3
            elif predicted_utilization < 0.3:  # Low utilization predicted
                recommended_capacity = current_capacity * 0.8
            else:
                continue  # No adjustment needed
            
            # Apply optimization suggestion
            if pool_id in self.optimizer.resource_pools:
                self.optimizer.resource_pools[pool_id].total_capacity = recommended_capacity
                print(f"Cross-optimization: {pool_id} capacity adjusted to {recommended_capacity:.1f}")
    
    def make_unified_decision(
        self,
        workflow_id: str,
        resource_request: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> UnifiedResourceDecision:
        """
        Make unified resource management decision using both agent and optimizer.
        
        Args:
            workflow_id: Workflow identifier
            resource_request: Resource requirements and constraints
            context: Additional context for decision making
            
        Returns:
            Unified resource management decision
        """
        if not self.enabled:
            # Fallback to optimizer only
            allocation = self.optimizer.optimize_allocation(
                workflow_id,
                resource_request.get("tasks", []),
                resource_request.get("available_resources", {}),
                resource_request.get("constraints", {})
            )
            return UnifiedResourceDecision(
                allocation=allocation,
                scaling_decision=ScalingDecision(
                    decision_id="fallback_001",
                    resource_type=ResourceType.CPU,
                    direction=ScalingDirection.MAINTAIN,
                    trigger=ScalingTrigger.MANUAL_REQUEST,
                    current_capacity=100.0,
                    target_capacity=100.0,
                    scaling_factor=1.0,
                    confidence=0.5,
                    reasoning="Bridge disabled - optimizer fallback",
                    estimated_cost_impact=0.0,
                    estimated_performance_impact=0.0,
                    execution_priority=0.5
                ),
                consensus_score=1.0,
                decision_method="optimizer_fallback",
                optimization_metrics={},
                timestamp=time.time()
            )
        
        context = context or {}
        start_time = time.time()
        
        # Get decisions from both systems
        agent_decision = self._get_agent_decision(workflow_id, resource_request, context)
        optimizer_decision = self._get_optimizer_decision(workflow_id, resource_request, context)
        
        # Combine decisions based on mode
        unified_decision = self._combine_decisions(
            workflow_id, agent_decision, optimizer_decision, context
        )
        
        # Store decision
        with self.lock:
            self.unified_decisions[workflow_id] = unified_decision
        
        # Update consensus history
        self._update_consensus_history(unified_decision, time.time() - start_time)
        
        print(f"Unified decision for {workflow_id}: {unified_decision.decision_method}, "
              f"consensus: {unified_decision.consensus_score:.3f}")
        
        return unified_decision
    
    def _get_agent_decision(self, workflow_id: str, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Get decision from adaptive resource management agent."""
        # Convert request to agent format
        agent_context = {
            "workflow_id": workflow_id,
            "resource_requirements": request.get("constraints", {}),
            "current_utilization": self.agent.get_current_utilization(),
            **context
        }
        
        # Get agent analysis and scaling decision
        scaling_decision = self.agent.make_scaling_decision(agent_context)
        predictions = self.agent.get_resource_predictions()
        
        return {
            "scaling_decision": scaling_decision,
            "predictions": predictions,
            "utilization": self.agent.get_current_utilization(),
            "confidence": scaling_decision.confidence
        }
    
    def _get_optimizer_decision(self, workflow_id: str, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Get decision from traditional resource optimizer."""
        allocation = self.optimizer.optimize_allocation(
            workflow_id,
            request.get("tasks", []),
            request.get("available_resources", {}),
            request.get("constraints", {})
        )
        
        utilization = self.optimizer.get_resource_utilization()
        
        return {
            "allocation": allocation,
            "utilization": utilization,
            "efficiency": allocation.efficiency_score,
            "cost": allocation.total_cost
        }
    
    def _combine_decisions(
        self,
        workflow_id: str,
        agent_decision: Dict[str, Any],
        optimizer_decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> UnifiedResourceDecision:
        """Combine agent and optimizer decisions based on bridge mode."""
        
        if self.config.mode == BridgeMode.AGENT_ONLY:
            return self._create_agent_only_decision(workflow_id, agent_decision)
        elif self.config.mode == BridgeMode.OPTIMIZER_ONLY:
            return self._create_optimizer_only_decision(workflow_id, optimizer_decision)
        elif self.config.mode == BridgeMode.CONSENSUS:
            return self._create_consensus_decision(workflow_id, agent_decision, optimizer_decision)
        else:  # HYBRID
            return self._create_hybrid_decision(workflow_id, agent_decision, optimizer_decision)
    
    def _create_hybrid_decision(
        self,
        workflow_id: str,
        agent_decision: Dict[str, Any],
        optimizer_decision: Dict[str, Any]
    ) -> UnifiedResourceDecision:
        """Create hybrid decision by weighted combination."""
        allocation = optimizer_decision["allocation"]
        scaling_decision = agent_decision["scaling_decision"]
        
        # Calculate weighted metrics
        agent_weight = self.config.agent_weight
        optimizer_weight = self.config.optimizer_weight
        
        combined_confidence = (
            agent_decision["confidence"] * agent_weight +
            (allocation.efficiency_score if allocation.efficiency_score > 0 else 0.5) * optimizer_weight
        )
        
        optimization_metrics = {
            "agent_confidence": agent_decision["confidence"],
            "optimizer_efficiency": allocation.efficiency_score,
            "optimizer_cost": allocation.total_cost,
            "combined_confidence": combined_confidence,
            "agent_weight": agent_weight,
            "optimizer_weight": optimizer_weight
        }
        
        return UnifiedResourceDecision(
            allocation=allocation,
            scaling_decision=scaling_decision,
            consensus_score=combined_confidence,
            decision_method="hybrid_weighted",
            optimization_metrics=optimization_metrics,
            timestamp=time.time()
        )
    
    def _create_consensus_decision(
        self,
        workflow_id: str,
        agent_decision: Dict[str, Any],
        optimizer_decision: Dict[str, Any]
    ) -> UnifiedResourceDecision:
        """Create consensus decision by agreement analysis."""
        allocation = optimizer_decision["allocation"]
        scaling_decision = agent_decision["scaling_decision"]
        
        # Analyze agreement between decisions
        agent_confidence = agent_decision["confidence"]
        optimizer_efficiency = allocation.efficiency_score
        
        # Check for agreement indicators
        agreement_score = 0.0
        
        # Compare resource utilization trends
        if agent_decision.get("utilization") and optimizer_decision.get("utilization"):
            # Simple agreement check based on utilization patterns
            agreement_score += 0.4
        
        # Compare efficiency/confidence alignment
        confidence_diff = abs(agent_confidence - optimizer_efficiency)
        if confidence_diff < 0.2:
            agreement_score += 0.4
        elif confidence_diff < 0.4:
            agreement_score += 0.2
        
        # Cost vs prediction alignment
        if allocation.total_cost < 100.0 and agent_confidence > 0.7:  # Low cost, high confidence
            agreement_score += 0.2
        
        consensus_reached = agreement_score >= self.config.consensus_threshold
        
        optimization_metrics = {
            "agreement_score": agreement_score,
            "consensus_reached": consensus_reached,
            "confidence_diff": confidence_diff,
            "agent_confidence": agent_confidence,
            "optimizer_efficiency": optimizer_efficiency
        }
        
        return UnifiedResourceDecision(
            allocation=allocation,
            scaling_decision=scaling_decision,
            consensus_score=agreement_score,
            decision_method="consensus" if consensus_reached else "consensus_fallback",
            optimization_metrics=optimization_metrics,
            timestamp=time.time()
        )
    
    def _create_agent_only_decision(self, workflow_id: str, agent_decision: Dict[str, Any]) -> UnifiedResourceDecision:
        """Create decision using only agent intelligence."""
        scaling_decision = agent_decision["scaling_decision"]
        
        # Create minimal allocation based on agent decision
        allocation = ResourceAllocation(
            workflow_id=workflow_id,
            allocations={"agent_managed": {"capacity": scaling_decision.target_capacity}},
            status="agent_managed",
            efficiency_score=agent_decision["confidence"]
        )
        
        return UnifiedResourceDecision(
            allocation=allocation,
            scaling_decision=scaling_decision,
            consensus_score=agent_decision["confidence"],
            decision_method="agent_only",
            optimization_metrics={"agent_confidence": agent_decision["confidence"]},
            timestamp=time.time()
        )
    
    def _create_optimizer_only_decision(self, workflow_id: str, optimizer_decision: Dict[str, Any]) -> UnifiedResourceDecision:
        """Create decision using only traditional optimizer."""
        allocation = optimizer_decision["allocation"]
        
        # Create minimal scaling decision
        scaling_decision = ScalingDecision(
            decision_id="optimizer_001",
            resource_type=ResourceType.CPU,
            direction=ScalingDirection.MAINTAIN,
            trigger=ScalingTrigger.MANUAL_REQUEST,
            current_capacity=100.0,
            target_capacity=100.0,
            scaling_factor=1.0,
            confidence=allocation.efficiency_score,
            reasoning="Traditional optimizer decision",
            estimated_cost_impact=0.0,
            estimated_performance_impact=0.0,
            execution_priority=allocation.efficiency_score
        )
        
        return UnifiedResourceDecision(
            allocation=allocation,
            scaling_decision=scaling_decision,
            consensus_score=allocation.efficiency_score,
            decision_method="optimizer_only",
            optimization_metrics={
                "efficiency": allocation.efficiency_score,
                "cost": allocation.total_cost
            },
            timestamp=time.time()
        )
    
    def _update_consensus_history(self, decision: UnifiedResourceDecision, processing_time: float):
        """Update consensus decision history for learning."""
        history_entry = {
            "timestamp": decision.timestamp,
            "method": decision.decision_method,
            "consensus_score": decision.consensus_score,
            "processing_time": processing_time,
            "metrics": decision.optimization_metrics
        }
        
        with self.lock:
            self.consensus_history.append(history_entry)
            # Keep last 100 decisions
            if len(self.consensus_history) > 100:
                self.consensus_history.pop(0)
    
    def get_bridge_metrics(self) -> Dict[str, Any]:
        """Get comprehensive bridge performance metrics."""
        with self.lock:
            recent_decisions = self.consensus_history[-20:] if self.consensus_history else []
        
        if not recent_decisions:
            return {"status": "no_decisions"}
        
        # Calculate metrics
        avg_consensus = sum(d["consensus_score"] for d in recent_decisions) / len(recent_decisions)
        avg_processing_time = sum(d["processing_time"] for d in recent_decisions) / len(recent_decisions)
        
        method_counts = {}
        for decision in recent_decisions:
            method = decision["method"]
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "total_decisions": len(self.consensus_history),
            "recent_decisions": len(recent_decisions),
            "avg_consensus_score": avg_consensus,
            "avg_processing_time": avg_processing_time,
            "decision_methods": method_counts,
            "sync_status": "active" if self.running else "inactive",
            "unified_pools": len(self.resource_pools_sync),
            "bridge_mode": self.config.mode.value
        }
    
    def optimize_bridge_configuration(self, performance_data: Dict[str, Any] = None):
        """Optimize bridge configuration based on performance history."""
        if not self.consensus_history:
            return
        
        with self.lock:
            recent_decisions = self.consensus_history[-50:]
        
        # Analyze decision effectiveness
        method_performance = {}
        for decision in recent_decisions:
            method = decision["method"]
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(decision["consensus_score"])
        
        # Find best performing method
        best_method = None
        best_score = 0.0
        
        for method, scores in method_performance.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_method = method
        
        # Adjust configuration based on performance
        if best_method == "agent_only" and best_score > 0.8:
            self.config.agent_weight = min(0.8, self.config.agent_weight + 0.1)
            self.config.optimizer_weight = 1.0 - self.config.agent_weight
        elif best_method == "optimizer_only" and best_score > 0.8:
            self.config.optimizer_weight = min(0.8, self.config.optimizer_weight + 0.1)
            self.config.agent_weight = 1.0 - self.config.optimizer_weight
        
        print(f"Bridge configuration optimized: agent={self.config.agent_weight:.2f}, "
              f"optimizer={self.config.optimizer_weight:.2f}")
    
    def shutdown(self):
        """Shutdown bridge and cleanup resources."""
        self.running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        
        # Store final state
        self.shared_state.set("bridge_final_metrics", self.get_bridge_metrics())
        
        print("Adaptive Resource Bridge shutdown complete")


def get_adaptive_resource_bridge(config: BridgeConfiguration = None) -> AdaptiveResourceBridge:
    """Get adaptive resource bridge instance."""
    return AdaptiveResourceBridge(config)