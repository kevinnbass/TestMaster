"""
Workflow Designer - Automated Workflow Design System

This module implements the WorkflowDesigner component that designs workflows
based on requirements and available system capabilities, with intelligent
task creation and dependency management.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import logging
import uuid
from typing import Dict, List, Any

import networkx as nx

from .workflow_types import (
    WorkflowDefinition, WorkflowTask, WorkflowPriority,
    OptimizationObjective, TaskStatus
)

logger = logging.getLogger(__name__)


class WorkflowDesigner:
    """Designs workflows based on requirements and available systems"""
    
    def __init__(self):
        self.design_templates: Dict[str, Dict[str, Any]] = {}
        self.system_capabilities: Dict[str, List[str]] = {}
        self.performance_models: Dict[str, Dict[str, float]] = {}
        
        logger.info("WorkflowDesigner initialized")
    
    def register_system_capabilities(self, system_id: str, capabilities: List[str],
                                   performance_metrics: Dict[str, float]):
        """Register system capabilities for workflow design"""
        self.system_capabilities[system_id] = capabilities
        self.performance_models[system_id] = performance_metrics
        logger.info(f"Registered capabilities for {system_id}: {len(capabilities)} capabilities")
    
    async def design_workflow(self, requirements: Dict[str, Any],
                            constraints: Dict[str, Any] = None) -> WorkflowDefinition:
        """Design a workflow based on requirements"""
        logger.info(f"Designing workflow for: {requirements.get('objective', 'unknown')}")
        
        if constraints is None:
            constraints = {}
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        # Analyze requirements
        required_capabilities = self._analyze_requirements(requirements)
        
        # Select systems and create tasks
        tasks = await self._create_tasks(required_capabilities, requirements, constraints)
        
        # Build execution graph
        execution_graph = self._build_execution_graph(tasks, requirements)
        
        # Determine optimization objective
        optimization_objective = self._determine_optimization_objective(requirements)
        
        # Set workflow properties
        priority = self._determine_priority(requirements)
        max_parallel_tasks = constraints.get("max_parallel_tasks", 5)
        total_timeout = constraints.get("total_timeout", 3600.0)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=requirements.get("name", f"Workflow_{workflow_id}"),
            description=requirements.get("description", "Auto-generated workflow"),
            tasks=tasks,
            execution_graph=execution_graph,
            priority=priority,
            optimization_objective=optimization_objective,
            max_parallel_tasks=max_parallel_tasks,
            total_timeout=total_timeout,
            retry_policy=self._create_retry_policy(requirements),
            success_criteria=self._define_success_criteria(requirements),
            metadata=requirements.get("metadata", {})
        )
        
        logger.info(f"Designed workflow {workflow_id} with {len(tasks)} tasks")
        return workflow
    
    def _analyze_requirements(self, requirements: Dict[str, Any]) -> List[str]:
        """Analyze requirements to determine needed capabilities"""
        required_capabilities = []
        
        objective = requirements.get("objective", "").lower()
        
        # Map objectives to capabilities
        if "analyze" in objective or "analysis" in objective:
            required_capabilities.extend(["data_analysis", "statistical_analysis"])
        
        if "predict" in objective or "forecast" in objective:
            required_capabilities.extend(["prediction", "forecasting", "modeling"])
        
        if "optimize" in objective or "improvement" in objective:
            required_capabilities.extend(["optimization", "tuning"])
        
        if "pattern" in objective or "detect" in objective:
            required_capabilities.extend(["pattern_detection", "anomaly_detection"])
        
        if "recommend" in objective or "suggest" in objective:
            required_capabilities.extend(["recommendation", "decision_support"])
        
        # Add explicit capability requirements
        if "required_capabilities" in requirements:
            required_capabilities.extend(requirements["required_capabilities"])
        
        # Remove duplicates and return
        return list(set(required_capabilities))
    
    async def _create_tasks(self, required_capabilities: List[str],
                          requirements: Dict[str, Any],
                          constraints: Dict[str, Any]) -> List[WorkflowTask]:
        """Create tasks based on required capabilities"""
        tasks = []
        task_counter = 1
        
        for capability in required_capabilities:
            # Find systems that provide this capability
            suitable_systems = []
            for system_id, capabilities in self.system_capabilities.items():
                if capability in capabilities:
                    suitable_systems.append(system_id)
            
            if suitable_systems:
                # Select best system for this capability
                best_system = self._select_best_system(suitable_systems, capability, constraints)
                
                # Create task
                task = WorkflowTask(
                    task_id=f"task_{task_counter:03d}",
                    name=f"{capability}_task",
                    task_type=capability,
                    target_system=best_system,
                    parameters=self._create_task_parameters(capability, requirements),
                    estimated_duration=self._estimate_task_duration(best_system, capability),
                    timeout=constraints.get("task_timeout", 300.0),
                    max_retries=constraints.get("max_retries", 3),
                    priority=self._calculate_task_priority(capability, requirements)
                )
                
                tasks.append(task)
                task_counter += 1
        
        # Add dependencies between tasks
        tasks = self._add_task_dependencies(tasks, requirements)
        
        return tasks
    
    def _select_best_system(self, suitable_systems: List[str], capability: str,
                          constraints: Dict[str, Any]) -> str:
        """Select the best system for a capability"""
        if len(suitable_systems) == 1:
            return suitable_systems[0]
        
        # Score systems based on performance metrics
        best_system = suitable_systems[0]
        best_score = 0.0
        
        for system_id in suitable_systems:
            score = self._calculate_system_score(system_id, capability, constraints)
            if score > best_score:
                best_score = score
                best_system = system_id
        
        return best_system
    
    def _calculate_system_score(self, system_id: str, capability: str,
                              constraints: Dict[str, Any]) -> float:
        """Calculate score for a system for a specific capability"""
        if system_id not in self.performance_models:
            return 0.5  # Default score
        
        performance = self.performance_models[system_id]
        score = 0.0
        
        # Factor in response time (lower is better)
        response_time = performance.get("response_time", 1000)
        score += max(0, (2000 - response_time) / 2000) * 0.3
        
        # Factor in availability (higher is better)
        availability = performance.get("availability", 0.95)
        score += availability * 0.3
        
        # Factor in throughput (higher is better)
        throughput = performance.get("throughput", 100)
        score += min(throughput / 1000, 1.0) * 0.2
        
        # Factor in error rate (lower is better)
        error_rate = performance.get("error_rate", 0.05)
        score += max(0, (0.1 - error_rate) / 0.1) * 0.2
        
        # Apply constraint penalties
        if constraints.get("max_response_time") and response_time > constraints["max_response_time"]:
            score *= 0.5
        
        if constraints.get("min_availability") and availability < constraints["min_availability"]:
            score *= 0.3
        
        return score
    
    def _create_task_parameters(self, capability: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create parameters for a task based on capability and requirements"""
        base_parameters = {
            "capability": capability,
            "timeout": 300,
            "priority": "medium"
        }
        
        # Add capability-specific parameters
        if capability in ["data_analysis", "statistical_analysis"]:
            base_parameters.update({
                "analysis_type": requirements.get("analysis_type", "comprehensive"),
                "include_visualization": requirements.get("include_visualization", False)
            })
        
        elif capability in ["prediction", "forecasting"]:
            base_parameters.update({
                "prediction_horizon": requirements.get("prediction_horizon", "short_term"),
                "confidence_level": requirements.get("confidence_level", 0.95)
            })
        
        elif capability in ["optimization", "tuning"]:
            base_parameters.update({
                "optimization_target": requirements.get("optimization_target", "performance"),
                "max_iterations": requirements.get("max_iterations", 100)
            })
        
        # Add common parameters from requirements
        if "input_data" in requirements:
            base_parameters["input_data"] = requirements["input_data"]
        
        if "output_format" in requirements:
            base_parameters["output_format"] = requirements["output_format"]
        
        return base_parameters
    
    def _estimate_task_duration(self, system_id: str, capability: str) -> float:
        """Estimate duration for a task"""
        if system_id not in self.performance_models:
            return 60.0  # Default 1 minute
        
        performance = self.performance_models[system_id]
        base_time = performance.get("response_time", 1000) / 1000  # Convert to seconds
        
        # Adjust based on capability complexity
        complexity_multipliers = {
            "data_analysis": 2.0,
            "prediction": 3.0,
            "optimization": 4.0,
            "pattern_detection": 1.5,
            "recommendation": 1.0
        }
        
        multiplier = complexity_multipliers.get(capability, 2.0)
        return base_time * multiplier
    
    def _calculate_task_priority(self, capability: str, requirements: Dict[str, Any]) -> int:
        """Calculate priority for a task"""
        base_priority = 5  # Medium priority
        
        # Adjust based on capability importance
        critical_capabilities = ["optimization", "decision_support", "prediction"]
        if capability in critical_capabilities:
            base_priority = 2  # High priority
        
        # Adjust based on requirements
        if requirements.get("priority") == "critical":
            base_priority = 1
        elif requirements.get("priority") == "high":
            base_priority = 2
        elif requirements.get("priority") == "low":
            base_priority = 4
        
        return base_priority
    
    def _add_task_dependencies(self, tasks: List[WorkflowTask],
                             requirements: Dict[str, Any]) -> List[WorkflowTask]:
        """Add dependencies between tasks based on logical flow"""
        
        # Create dependency mapping based on task types
        dependency_rules = {
            "data_analysis": [],  # Usually first
            "pattern_detection": ["data_analysis"],
            "prediction": ["data_analysis", "pattern_detection"],
            "optimization": ["data_analysis", "prediction"],
            "recommendation": ["optimization", "prediction"]
        }
        
        # Build task lookup
        task_lookup = {task.task_type: task for task in tasks}
        
        # Apply dependency rules
        for task in tasks:
            required_predecessors = dependency_rules.get(task.task_type, [])
            for predecessor_type in required_predecessors:
                if predecessor_type in task_lookup:
                    predecessor_task = task_lookup[predecessor_type]
                    if predecessor_task.task_id not in task.dependencies:
                        task.dependencies.append(predecessor_task.task_id)
        
        return tasks
    
    def _build_execution_graph(self, tasks: List[WorkflowTask],
                             requirements: Dict[str, Any]) -> nx.DiGraph:
        """Build execution graph from tasks and dependencies"""
        graph = nx.DiGraph()
        
        # Add nodes for tasks
        for task in tasks:
            graph.add_node(task.task_id, **{
                "name": task.name,
                "task_type": task.task_type,
                "target_system": task.target_system,
                "estimated_duration": task.estimated_duration,
                "priority": task.priority
            })
        
        # Add edges for dependencies
        for task in tasks:
            for dependency in task.dependencies:
                graph.add_edge(dependency, task.task_id, relationship="dependency")
        
        # Add start and end nodes
        graph.add_node("start", type="control")
        graph.add_node("end", type="control")
        
        # Connect start to tasks with no dependencies
        for task in tasks:
            if not task.dependencies:
                graph.add_edge("start", task.task_id, relationship="start")
        
        # Connect tasks with no successors to end
        for task in tasks:
            has_successors = any(task.task_id in other_task.dependencies for other_task in tasks)
            if not has_successors:
                graph.add_edge(task.task_id, "end", relationship="end")
        
        return graph
    
    def _determine_optimization_objective(self, requirements: Dict[str, Any]) -> OptimizationObjective:
        """Determine optimization objective from requirements"""
        explicit_objective = requirements.get("optimization_objective")
        if explicit_objective:
            return OptimizationObjective(explicit_objective)
        
        # Infer from requirements
        if requirements.get("time_critical", False):
            return OptimizationObjective.MINIMIZE_TIME
        elif requirements.get("accuracy_critical", False):
            return OptimizationObjective.MAXIMIZE_ACCURACY
        elif requirements.get("cost_sensitive", False):
            return OptimizationObjective.MINIMIZE_COST
        elif requirements.get("high_throughput", False):
            return OptimizationObjective.MAXIMIZE_THROUGHPUT
        else:
            return OptimizationObjective.BALANCE_ALL
    
    def _determine_priority(self, requirements: Dict[str, Any]) -> WorkflowPriority:
        """Determine workflow priority from requirements"""
        priority_str = requirements.get("priority", "medium").lower()
        
        priority_mapping = {
            "critical": WorkflowPriority.CRITICAL,
            "high": WorkflowPriority.HIGH,
            "medium": WorkflowPriority.MEDIUM,
            "low": WorkflowPriority.LOW,
            "background": WorkflowPriority.BACKGROUND
        }
        
        return priority_mapping.get(priority_str, WorkflowPriority.MEDIUM)
    
    def _create_retry_policy(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create retry policy based on requirements"""
        return {
            "max_retries": requirements.get("max_retries", 3),
            "retry_delay": requirements.get("retry_delay", 5.0),
            "exponential_backoff": requirements.get("exponential_backoff", True),
            "retry_on_timeout": requirements.get("retry_on_timeout", True),
            "retry_on_system_error": requirements.get("retry_on_system_error", True)
        }
    
    def _define_success_criteria(self, requirements: Dict[str, Any]) -> List[str]:
        """Define success criteria for the workflow"""
        criteria = [
            "All tasks completed successfully",
            "No critical errors occurred",
            "Execution time within acceptable limits"
        ]
        
        # Add specific criteria from requirements
        if "success_criteria" in requirements:
            criteria.extend(requirements["success_criteria"])
        
        if requirements.get("accuracy_threshold"):
            criteria.append(f"Results accuracy >= {requirements['accuracy_threshold']}")
        
        if requirements.get("max_execution_time"):
            criteria.append(f"Execution time <= {requirements['max_execution_time']} seconds")
        
        return criteria