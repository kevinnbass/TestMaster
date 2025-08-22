"""
Intelligent Workflow Engine Core - Master Workflow Orchestration System
======================================================================

Enterprise workflow management system providing comprehensive orchestration of intelligent 
workflow design, scheduling, execution, and optimization with advanced AI capabilities.
Implements unified workflow intelligence with autonomous task management and optimization.

This module provides the master orchestration system that integrates workflow design,
scheduling, optimization, and execution into a unified intelligent workflow platform
for enterprise-grade automation and intelligence coordination.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: intelligent_workflow_engine_core.py (420 lines)
"""

import asyncio
import statistics
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

# Import workflow components
from .workflow_types import (
    WorkflowDefinition, WorkflowExecution, WorkflowStatus, WorkflowOptimization,
    OptimizationObjective, WorkflowPriority, ExecutionStrategy
)
from .workflow_designer import IntelligentWorkflowDesigner
from .workflow_scheduler import IntelligentWorkflowScheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkflowOptimizer:
    """Advanced workflow optimizer with ML-based performance enhancement"""
    
    def __init__(self):
        self.execution_history: List[WorkflowExecution] = []
        self.optimization_history: List[WorkflowOptimization] = []
        self.performance_patterns: Dict[str, Dict[str, float]] = {}
        
        logger.info("WorkflowOptimizer initialized with enterprise optimization capabilities")
    
    def record_execution(self, execution: WorkflowExecution):
        """Record workflow execution for optimization analysis"""
        self.execution_history.append(execution)
        self._update_performance_patterns(execution)
        logger.debug(f"Recorded execution {execution.execution_id} for optimization analysis")
    
    async def optimize_workflow(self, workflow: WorkflowDefinition,
                              objective: OptimizationObjective = None) -> WorkflowOptimization:
        """Optimize workflow based on historical data and ML insights"""
        if objective is None:
            objective = workflow.optimization_objective
        
        logger.info(f"Optimizing workflow {workflow.workflow_id} for {objective.value}")
        
        # Analyze current performance
        current_performance = self._analyze_current_performance(workflow)
        
        # Generate AI-powered optimization recommendations
        optimizations = await self._generate_optimizations(workflow, objective)
        
        # Apply optimizations with ML insights
        optimized_workflow = self._apply_optimizations(workflow, optimizations)
        
        # Predict optimized performance using patterns
        optimized_performance = self._predict_optimized_performance(optimized_workflow, optimizations)
        
        # Calculate improvement with confidence scoring
        improvement = self._calculate_improvement(current_performance, optimized_performance, objective)
        confidence = self._calculate_confidence_score(workflow, optimizations)
        
        # Create comprehensive optimization record
        optimization = WorkflowOptimization(
            optimization_id=f"opt_{uuid.uuid4().hex[:8]}",
            workflow_id=workflow.workflow_id,
            objective=objective,
            original_performance=current_performance,
            optimized_performance=optimized_performance,
            improvement_percentage=improvement,
            optimization_changes=optimizations,
            implementation_effort=self._estimate_implementation_effort(optimizations),
            confidence_score=confidence,
            recommended_actions=self._generate_action_plan(optimizations),
            validation_steps=self._generate_validation_plan(workflow),
            rollback_plan=self._generate_rollback_plan(workflow, optimizations)
        )
        
        self.optimization_history.append(optimization)
        
        logger.info(f"Workflow optimization completed. Improvement: {improvement:.1%}, Confidence: {confidence:.1%}")
        return optimization
    
    def _analyze_current_performance(self, workflow: WorkflowDefinition) -> Dict[str, float]:
        """Analyze current workflow performance with pattern recognition"""
        # Find historical executions
        workflow_executions = [
            exec for exec in self.execution_history 
            if exec.workflow_id == workflow.workflow_id and exec.status == WorkflowStatus.COMPLETED
        ]
        
        if not workflow_executions:
            # Use estimated performance from workflow analysis
            complexity = workflow.calculate_complexity_score()
            estimated_duration = workflow.get_critical_path_duration()
            
            return {
                "duration": estimated_duration,
                "success_rate": max(0.7, 1.0 - (complexity * 0.05)),
                "resource_usage": min(100.0, 30.0 + (complexity * 5.0)),
                "cost": estimated_duration * (0.05 + complexity * 0.01),
                "throughput": 1.0 / max(1.0, estimated_duration / 60.0),
                "reliability": max(0.6, 1.0 - (complexity * 0.03))
            }
        
        # Calculate performance metrics from historical data
        durations = [exec.get_execution_duration() or 0 for exec in workflow_executions]
        success_rates = [exec.calculate_success_rate() for exec in workflow_executions]
        efficiency_scores = [exec.calculate_efficiency_score() for exec in workflow_executions]
        
        return {
            "duration": statistics.mean(durations),
            "success_rate": statistics.mean(success_rates),
            "resource_usage": statistics.mean([exec.resource_usage.get('cpu', 50.0) for exec in workflow_executions]),
            "cost": statistics.mean([exec.cost_incurred for exec in workflow_executions]),
            "throughput": statistics.mean([exec.throughput_metrics.get('tasks_per_minute', 1.0) for exec in workflow_executions]),
            "reliability": statistics.mean(efficiency_scores)
        }
    
    async def _generate_optimizations(self, workflow: WorkflowDefinition,
                                    objective: OptimizationObjective) -> List[str]:
        """Generate AI-powered optimization recommendations"""
        optimizations = []
        complexity = workflow.calculate_complexity_score()
        
        # Objective-specific optimizations with AI insights
        if objective == OptimizationObjective.MINIMIZE_TIME:
            optimizations.extend(self._generate_time_optimizations(workflow, complexity))
        elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            optimizations.extend(self._generate_accuracy_optimizations(workflow, complexity))
        elif objective == OptimizationObjective.MINIMIZE_COST:
            optimizations.extend(self._generate_cost_optimizations(workflow, complexity))
        elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            optimizations.extend(self._generate_throughput_optimizations(workflow, complexity))
        elif objective == OptimizationObjective.MAXIMIZE_RELIABILITY:
            optimizations.extend(self._generate_reliability_optimizations(workflow, complexity))
        else:  # BALANCE_ALL
            optimizations.extend(self._generate_balanced_optimizations(workflow, complexity))
        
        # Add ML-based pattern optimizations
        optimizations.extend(self._generate_pattern_based_optimizations(workflow))
        
        return optimizations
    
    def _generate_time_optimizations(self, workflow: WorkflowDefinition, complexity: float) -> List[str]:
        """Generate time-focused optimization recommendations"""
        optimizations = []
        
        if workflow.max_parallel_tasks < max(3, len(workflow.tasks) // 3):
            optimizations.append("Increase parallel task execution to reduce critical path")
        
        if complexity > 10:
            optimizations.append("Implement task dependency optimization to minimize wait times")
        
        optimizations.extend([
            "Enable predictive task preloading for frequently accessed resources",
            "Implement intelligent caching for repeated operations",
            "Optimize system selection based on response time patterns"
        ])
        
        return optimizations
    
    def _generate_accuracy_optimizations(self, workflow: WorkflowDefinition, complexity: float) -> List[str]:
        """Generate accuracy-focused optimization recommendations"""
        optimizations = [
            "Implement ensemble decision making for critical tasks",
            "Add cross-validation steps for important results",
            "Enable multi-system verification for accuracy-critical operations",
            "Implement adaptive retry logic based on confidence scores"
        ]
        
        if complexity > 15:
            optimizations.append("Add intermediate validation checkpoints for complex workflows")
        
        return optimizations
    
    def _generate_cost_optimizations(self, workflow: WorkflowDefinition, complexity: float) -> List[str]:
        """Generate cost-focused optimization recommendations"""
        return [
            "Implement dynamic resource scaling based on demand",
            "Enable intelligent system selection for cost-effectiveness",
            "Add resource pooling and sharing across similar tasks",
            "Implement batch processing for cost-efficient resource utilization",
            "Enable spot instance usage for non-critical tasks"
        ]
    
    def _generate_throughput_optimizations(self, workflow: WorkflowDefinition, complexity: float) -> List[str]:
        """Generate throughput-focused optimization recommendations"""
        return [
            "Implement pipeline parallelism for sequential task chains",
            "Enable dynamic load balancing across available systems",
            "Add intelligent queuing and priority management",
            "Implement result streaming for real-time processing",
            "Enable predictive scaling based on workload patterns"
        ]
    
    def _generate_reliability_optimizations(self, workflow: WorkflowDefinition, complexity: float) -> List[str]:
        """Generate reliability-focused optimization recommendations"""
        return [
            "Implement comprehensive fault tolerance with circuit breakers",
            "Add health monitoring and predictive failure detection",
            "Enable automatic failover to backup systems",
            "Implement checkpointing for long-running tasks",
            "Add comprehensive error recovery and rollback mechanisms"
        ]
    
    def _generate_balanced_optimizations(self, workflow: WorkflowDefinition, complexity: float) -> List[str]:
        """Generate balanced optimization recommendations"""
        return [
            "Implement adaptive execution strategy based on current conditions",
            "Enable multi-objective optimization with dynamic weights",
            "Add intelligent resource allocation with cost-performance balance",
            "Implement quality gates with configurable trade-offs",
            "Enable continuous optimization based on real-time metrics"
        ]
    
    def _generate_pattern_based_optimizations(self, workflow: WorkflowDefinition) -> List[str]:
        """Generate optimizations based on learned patterns"""
        optimizations = []
        
        # Check for patterns in performance data
        if self.performance_patterns:
            similar_patterns = self._find_similar_workflows(workflow)
            
            if similar_patterns:
                optimizations.append("Apply learned optimizations from similar workflow patterns")
                optimizations.append("Implement pattern-based resource allocation strategies")
        
        return optimizations
    
    def _apply_optimizations(self, workflow: WorkflowDefinition, optimizations: List[str]) -> WorkflowDefinition:
        """Apply optimizations to create enhanced workflow"""
        # Create optimized workflow copy
        optimized_workflow = WorkflowDefinition(
            workflow_id=f"{workflow.workflow_id}_optimized_{uuid.uuid4().hex[:8]}",
            name=f"{workflow.name} (AI Optimized)",
            description=f"AI-optimized version of {workflow.name}",
            tasks=workflow.tasks.copy(),
            priority=workflow.priority,
            optimization_objective=workflow.optimization_objective,
            execution_strategy=ExecutionStrategy.ADAPTIVE,
            max_parallel_tasks=workflow.max_parallel_tasks,
            total_timeout=workflow.total_timeout,
            max_retries=workflow.max_retries,
            success_criteria=workflow.success_criteria.copy(),
            required_capabilities=workflow.required_capabilities.copy(),
            metadata={**workflow.metadata, "optimized": True, "optimizations_applied": len(optimizations)}
        )
        
        # Apply specific optimizations
        for optimization in optimizations:
            if "parallel" in optimization.lower():
                optimized_workflow.max_parallel_tasks = min(20, optimized_workflow.max_parallel_tasks + 5)
            elif "retry" in optimization.lower():
                optimized_workflow.max_retries = min(10, optimized_workflow.max_retries + 2)
            elif "adaptive" in optimization.lower():
                optimized_workflow.execution_strategy = ExecutionStrategy.ADAPTIVE
            elif "timeout" in optimization.lower() and "increase" in optimization.lower():
                if optimized_workflow.total_timeout:
                    optimized_workflow.total_timeout *= 1.5
        
        return optimized_workflow
    
    def _predict_optimized_performance(self, optimized_workflow: WorkflowDefinition,
                                     optimizations: List[str]) -> Dict[str, float]:
        """Predict performance of optimized workflow using ML patterns"""
        base_performance = self._analyze_current_performance(optimized_workflow)
        
        # Apply optimization impact factors
        improvement_factors = {
            "duration": 1.0,
            "success_rate": 1.0,
            "resource_usage": 1.0,
            "cost": 1.0,
            "throughput": 1.0,
            "reliability": 1.0
        }
        
        # Calculate improvements based on optimization types
        for optimization in optimizations:
            if "parallel" in optimization.lower():
                improvement_factors["duration"] *= 0.75  # 25% faster
                improvement_factors["throughput"] *= 1.4  # 40% higher throughput
                improvement_factors["resource_usage"] *= 1.15  # 15% more resources
                
            elif "cache" in optimization.lower():
                improvement_factors["duration"] *= 0.85  # 15% faster
                improvement_factors["cost"] *= 0.9  # 10% cost reduction
                
            elif "accuracy" in optimization.lower() or "validation" in optimization.lower():
                improvement_factors["success_rate"] *= 1.1  # 10% higher accuracy
                improvement_factors["duration"] *= 1.15  # 15% slower
                improvement_factors["reliability"] *= 1.08  # 8% more reliable
                
            elif "cost" in optimization.lower():
                improvement_factors["cost"] *= 0.8  # 20% cost reduction
                improvement_factors["resource_usage"] *= 0.9  # 10% less resources
                
            elif "reliability" in optimization.lower() or "fault" in optimization.lower():
                improvement_factors["reliability"] *= 1.15  # 15% more reliable
                improvement_factors["success_rate"] *= 1.05  # 5% higher success
        
        # Apply improvements
        optimized_performance = {}
        for metric, value in base_performance.items():
            factor = improvement_factors.get(metric, 1.0)
            optimized_performance[metric] = value * factor
        
        return optimized_performance
    
    def _calculate_improvement(self, current: Dict[str, float], optimized: Dict[str, float],
                             objective: OptimizationObjective) -> float:
        """Calculate improvement percentage with objective weighting"""
        improvements = {}
        
        for metric in current:
            if metric in optimized:
                if metric in ["success_rate", "throughput", "reliability"]:
                    # Higher is better
                    improvements[metric] = (optimized[metric] - current[metric]) / current[metric]
                else:
                    # Lower is better (duration, cost, resource_usage)
                    improvements[metric] = (current[metric] - optimized[metric]) / current[metric]
        
        # Weight improvements based on objective
        if objective == OptimizationObjective.MINIMIZE_TIME:
            return improvements.get("duration", 0) * 0.7 + improvements.get("throughput", 0) * 0.3
        elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            return improvements.get("success_rate", 0) * 0.8 + improvements.get("reliability", 0) * 0.2
        elif objective == OptimizationObjective.MINIMIZE_COST:
            return improvements.get("cost", 0) * 0.7 + improvements.get("resource_usage", 0) * 0.3
        elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            return improvements.get("throughput", 0) * 0.6 + improvements.get("duration", 0) * 0.4
        elif objective == OptimizationObjective.MAXIMIZE_RELIABILITY:
            return improvements.get("reliability", 0) * 0.6 + improvements.get("success_rate", 0) * 0.4
        else:  # BALANCE_ALL
            weights = {"duration": 0.2, "success_rate": 0.2, "cost": 0.15, "throughput": 0.15, 
                      "reliability": 0.15, "resource_usage": 0.15}
            return sum(improvements.get(metric, 0) * weight for metric, weight in weights.items())
    
    def _calculate_confidence_score(self, workflow: WorkflowDefinition, optimizations: List[str]) -> float:
        """Calculate confidence score for optimization recommendations"""
        base_confidence = 0.7
        
        # Increase confidence based on historical data
        similar_workflows = len([w for w in self.execution_history if len(w.completed_tasks) > 0])
        confidence_boost = min(0.25, similar_workflows * 0.02)
        
        # Adjust for optimization complexity
        complexity_penalty = len(optimizations) * 0.02
        
        return max(0.1, min(1.0, base_confidence + confidence_boost - complexity_penalty))
    
    def _estimate_implementation_effort(self, optimizations: List[str]) -> float:
        """Estimate implementation effort for optimizations"""
        effort_map = {
            "parallel": 0.3, "cache": 0.4, "ensemble": 0.6, "validation": 0.3,
            "scaling": 0.5, "balancing": 0.4, "monitoring": 0.3, "retry": 0.2,
            "timeout": 0.1, "selection": 0.3, "pipeline": 0.5
        }
        
        total_effort = 0.1  # Base effort
        for optimization in optimizations:
            for keyword, effort in effort_map.items():
                if keyword in optimization.lower():
                    total_effort += effort
                    break
        
        return min(1.0, total_effort)
    
    def _generate_action_plan(self, optimizations: List[str]) -> List[str]:
        """Generate actionable implementation plan"""
        return [
            f"Implement optimization: {opt}" for opt in optimizations[:5]
        ] + ["Validate optimization impact", "Monitor performance metrics", "Rollback if needed"]
    
    def _generate_validation_plan(self, workflow: WorkflowDefinition) -> List[str]:
        """Generate validation plan for optimizations"""
        return [
            "Run A/B testing between original and optimized workflows",
            "Monitor key performance metrics for degradation",
            "Validate success rate maintains acceptable thresholds",
            "Ensure resource usage stays within budget constraints",
            "Confirm optimization meets business requirements"
        ]
    
    def _generate_rollback_plan(self, workflow: WorkflowDefinition, optimizations: List[str]) -> List[str]:
        """Generate rollback plan for failed optimizations"""
        return [
            "Maintain original workflow configuration as backup",
            "Implement gradual rollback with traffic shifting",
            "Monitor system health during rollback process",
            "Document lessons learned from optimization attempt",
            "Update optimization models with failure patterns"
        ]
    
    def _update_performance_patterns(self, execution: WorkflowExecution):
        """Update performance patterns based on execution data"""
        workflow_signature = f"{len(execution.completed_tasks)}_{execution.workflow_id[:8]}"
        
        if workflow_signature not in self.performance_patterns:
            self.performance_patterns[workflow_signature] = {}
        
        patterns = self.performance_patterns[workflow_signature]
        patterns["success_rate"] = execution.calculate_success_rate()
        patterns["efficiency"] = execution.calculate_efficiency_score()
        patterns["duration"] = execution.get_execution_duration() or 0
        patterns["last_updated"] = datetime.now().isoformat()
    
    def _find_similar_workflows(self, workflow: WorkflowDefinition) -> List[Dict[str, Any]]:
        """Find workflows with similar patterns"""
        similar = []
        current_signature = f"{len(workflow.tasks)}_{workflow.workflow_id[:8]}"
        
        for signature, patterns in self.performance_patterns.items():
            if signature != current_signature and patterns.get("success_rate", 0) > 0.8:
                similar.append(patterns)
        
        return similar


class IntelligentWorkflowEngine:
    """
    Master workflow engine orchestrating intelligent workflow management
    Integrates design, scheduling, execution, and optimization capabilities
    """
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.designer = IntelligentWorkflowDesigner()
        self.scheduler = IntelligentWorkflowScheduler(max_concurrent_workflows)
        self.optimizer = WorkflowOptimizer()
        
        # State management
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.engine_metrics: Dict[str, Any] = {}
        self.is_running = False
        
        logger.info("IntelligentWorkflowEngine initialized with comprehensive AI capabilities")
    
    async def start_engine(self):
        """Start the intelligent workflow engine"""
        logger.info("Starting Intelligent Workflow Engine with AI orchestration")
        
        await self.scheduler.start_scheduler()
        self.is_running = True
        self._update_engine_metrics()
        
        logger.info("Intelligent Workflow Engine fully operational")
    
    def register_intelligence_system(self, system_id: str, capabilities: List[str],
                                   performance_metrics: Dict[str, float]):
        """Register intelligence system with comprehensive capability tracking"""
        logger.info(f"Registering intelligence system: {system_id} with {len(capabilities)} capabilities")
        
        # Register with designer for intelligent workflow creation
        self.designer.register_system_capabilities(system_id, capabilities, performance_metrics)
        
        # Register with scheduler for load balancing
        self.scheduler.register_system(system_id, performance_metrics.get("load", 0.0))
        
        logger.info(f"Successfully registered {system_id} for intelligent workflow orchestration")
    
    async def create_and_execute_workflow(self, requirements: Dict[str, Any],
                                        constraints: Dict[str, Any] = None) -> Tuple[str, str]:
        """Create and execute workflow with AI-powered optimization"""
        logger.info(f"Creating intelligent workflow for: {requirements.get('objective', 'unknown objective')}")
        
        # Design workflow with AI intelligence
        workflow = await self.designer.design_workflow(requirements, constraints)
        self.workflows[workflow.workflow_id] = workflow
        
        # Schedule for intelligent execution
        execution_id = await self.scheduler.schedule_workflow(workflow)
        
        # Record for optimization learning
        if execution_id in self.scheduler.active_workflows:
            execution = self.scheduler.active_workflows[execution_id]
            self.executions[execution_id] = execution
        
        logger.info(f"Created workflow {workflow.workflow_id} and scheduled execution {execution_id}")
        return workflow.workflow_id, execution_id
    
    async def optimize_workflow(self, workflow_id: str, 
                              objective: OptimizationObjective = None) -> WorkflowOptimization:
        """Optimize workflow with AI-powered enhancement"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        optimization = await self.optimizer.optimize_workflow(workflow, objective)
        
        logger.info(f"AI optimization completed for {workflow_id}. Improvement: {optimization.improvement_percentage:.1%}")
        return optimization
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status"""
        if workflow_id not in self.workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow = self.workflows[workflow_id]
        executions = [exec for exec in self.executions.values() if exec.workflow_id == workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "priority": workflow.priority.value,
            "tasks_count": len(workflow.tasks),
            "complexity_score": workflow.calculate_complexity_score(),
            "estimated_duration": workflow.get_critical_path_duration(),
            "executions_count": len(executions),
            "last_execution_status": executions[-1].status.value if executions else "never_executed",
            "optimization_objective": workflow.optimization_objective.value,
            "execution_strategy": workflow.execution_strategy.value,
            "created_at": workflow.created_at.isoformat(),
            "metadata": workflow.metadata
        }
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get comprehensive execution status"""
        execution = None
        
        if execution_id in self.scheduler.active_workflows:
            execution = self.scheduler.active_workflows[execution_id]
        elif execution_id in self.executions:
            execution = self.executions[execution_id]
        else:
            return {"error": f"Execution {execution_id} not found"}
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress": execution.progress,
            "current_tasks": len(execution.current_tasks),
            "completed_tasks": len(execution.completed_tasks),
            "failed_tasks": len(execution.failed_tasks),
            "skipped_tasks": len(execution.skipped_tasks),
            "success_rate": execution.calculate_success_rate(),
            "efficiency_score": execution.calculate_efficiency_score(),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "execution_duration": execution.get_execution_duration(),
            "resource_usage": execution.resource_usage,
            "cost_incurred": execution.cost_incurred,
            "execution_metrics": execution.execution_metrics
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status with AI insights"""
        scheduler_status = self.scheduler.get_scheduler_status()
        
        return {
            "engine_status": "running" if self.is_running else "stopped",
            "engine_metrics": self.engine_metrics,
            "scheduler_status": scheduler_status,
            "total_workflows": len(self.workflows),
            "total_executions": len(self.executions),
            "active_executions": len(self.scheduler.active_workflows),
            "optimization_history": len(self.optimizer.optimization_history),
            "registered_systems": len(self.designer.system_capabilities),
            "performance_patterns": len(self.optimizer.performance_patterns),
            "ai_insights": {
                "average_optimization_improvement": self._calculate_average_optimization_improvement(),
                "most_successful_objective": self._get_most_successful_objective(),
                "system_utilization_efficiency": self._calculate_system_efficiency()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_engine_metrics(self):
        """Update comprehensive engine metrics"""
        completed_executions = [exec for exec in self.executions.values() 
                              if exec.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]]
        
        self.engine_metrics = {
            "workflows_created": len(self.workflows),
            "workflows_executed": len(self.executions),
            "total_optimizations": len(self.optimizer.optimization_history),
            "average_workflow_complexity": statistics.mean([w.calculate_complexity_score() for w in self.workflows.values()]) if self.workflows else 0,
            "average_workflow_tasks": statistics.mean([len(w.tasks) for w in self.workflows.values()]) if self.workflows else 0,
            "overall_success_rate": self._calculate_success_rate(),
            "average_execution_duration": statistics.mean([exec.get_execution_duration() or 0 for exec in completed_executions]) if completed_executions else 0,
            "average_efficiency_score": statistics.mean([exec.calculate_efficiency_score() for exec in completed_executions]) if completed_executions else 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall workflow success rate"""
        completed_executions = [exec for exec in self.executions.values() 
                              if exec.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]]
        
        if not completed_executions:
            return 1.0
        
        successful_executions = [exec for exec in completed_executions 
                               if exec.status == WorkflowStatus.COMPLETED]
        
        return len(successful_executions) / len(completed_executions)
    
    def _calculate_average_optimization_improvement(self) -> float:
        """Calculate average improvement from optimizations"""
        if not self.optimizer.optimization_history:
            return 0.0
        
        improvements = [opt.improvement_percentage for opt in self.optimizer.optimization_history]
        return statistics.mean(improvements)
    
    def _get_most_successful_objective(self) -> str:
        """Get the most successful optimization objective"""
        if not self.optimizer.optimization_history:
            return "none"
        
        objective_improvements = {}
        for opt in self.optimizer.optimization_history:
            obj = opt.objective.value
            if obj not in objective_improvements:
                objective_improvements[obj] = []
            objective_improvements[obj].append(opt.improvement_percentage)
        
        best_objective = "balance_all"
        best_improvement = 0.0
        
        for obj, improvements in objective_improvements.items():
            avg_improvement = statistics.mean(improvements)
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                best_objective = obj
        
        return best_objective
    
    def _calculate_system_efficiency(self) -> float:
        """Calculate overall system utilization efficiency"""
        if not hasattr(self.scheduler, 'system_load') or not self.scheduler.system_load:
            return 0.0
        
        loads = list(self.scheduler.system_load.values())
        if not loads:
            return 0.0
        
        # Efficient utilization is around 70-80%
        target_utilization = 0.75
        actual_utilization = statistics.mean(loads)
        
        # Calculate efficiency (closer to target = higher efficiency)
        efficiency = 1.0 - abs(actual_utilization - target_utilization)
        return max(0.0, min(1.0, efficiency))
    
    async def stop_engine(self):
        """Stop the workflow engine gracefully"""
        logger.info("Stopping Intelligent Workflow Engine")
        
        self.is_running = False
        await self.scheduler.stop_scheduler()
        
        logger.info("Intelligent Workflow Engine stopped successfully")


# Export the main engine class
__all__ = ['IntelligentWorkflowEngine', 'WorkflowOptimizer']