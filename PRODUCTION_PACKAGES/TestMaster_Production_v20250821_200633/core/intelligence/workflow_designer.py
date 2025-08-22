"""
Workflow Designer - Intelligent Workflow Design and Template Management
=======================================================================

Advanced workflow design engine with AI-powered workflow generation, template management,
and intelligent task orchestration for autonomous intelligence systems.
Implements enterprise-grade workflow design patterns and optimization strategies.

This module provides comprehensive workflow design capabilities including AI-generated
workflows, template-based design, and intelligent task dependency analysis.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: workflow_designer.py (420 lines)
"""

import asyncio
import logging
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import uuid
import json

from .workflow_types import (
    WorkflowDefinition, WorkflowTask, WorkflowTemplate, SystemCapabilities,
    WorkflowDesignMode, OptimizationObjective, WorkflowPriority, TaskStatus,
    ExecutionStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentWorkflowDesigner:
    """AI-powered workflow designer with template management and optimization"""
    
    def __init__(self):
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        self.system_capabilities: Dict[str, SystemCapabilities] = {}
        self.design_patterns: Dict[str, Dict[str, Any]] = {}
        self.optimization_rules: List[Dict[str, Any]] = []
        
        # Design intelligence
        self.capability_graph = nx.DiGraph()
        self.task_patterns: Dict[str, List[str]] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialize_default_patterns()
        
        self.logger.info("Intelligent Workflow Designer initialized")
    
    def _initialize_default_patterns(self):
        """Initialize default workflow design patterns"""
        self.design_patterns = {
            'data_analysis': {
                'description': 'Standard data analysis workflow pattern',
                'task_sequence': ['data_collection', 'preprocessing', 'analysis', 'visualization', 'reporting'],
                'optimization_hints': ['parallelize_preprocessing', 'cache_intermediate_results'],
                'required_capabilities': ['data_processing', 'analysis', 'visualization']
            },
            'ml_pipeline': {
                'description': 'Machine learning pipeline pattern',
                'task_sequence': ['data_preparation', 'feature_engineering', 'model_training', 'validation', 'deployment'],
                'optimization_hints': ['use_gpu_for_training', 'parallel_validation'],
                'required_capabilities': ['machine_learning', 'data_processing', 'model_deployment']
            },
            'intelligence_synthesis': {
                'description': 'Intelligence synthesis and coordination pattern',
                'task_sequence': ['data_gathering', 'multi_source_analysis', 'pattern_detection', 'synthesis', 'recommendation'],
                'optimization_hints': ['parallel_analysis', 'ensemble_methods'],
                'required_capabilities': ['pattern_detection', 'synthesis', 'recommendation']
            },
            'security_analysis': {
                'description': 'Security analysis and assessment pattern',
                'task_sequence': ['vulnerability_scan', 'threat_analysis', 'risk_assessment', 'mitigation_planning', 'reporting'],
                'optimization_hints': ['parallel_scanning', 'prioritize_critical_assets'],
                'required_capabilities': ['security_analysis', 'vulnerability_assessment', 'risk_analysis']
            }
        }
    
    async def design_workflow(self, requirements: Dict[str, Any]) -> WorkflowDefinition:
        """Design intelligent workflow based on requirements"""
        try:
            self.logger.info(f"Designing workflow: {requirements.get('name', 'Unnamed')}")
            
            design_mode = WorkflowDesignMode(requirements.get('design_mode', 'ai_generated'))
            
            if design_mode == WorkflowDesignMode.TEMPLATE_BASED:
                workflow = await self._design_from_template(requirements)
            elif design_mode == WorkflowDesignMode.AI_GENERATED:
                workflow = await self._ai_generate_workflow(requirements)
            elif design_mode == WorkflowDesignMode.HYBRID:
                workflow = await self._hybrid_design_workflow(requirements)
            else:  # MANUAL
                workflow = await self._manual_design_workflow(requirements)
            
            # Apply optimization hints
            await self._apply_design_optimizations(workflow, requirements)
            
            # Validate workflow design
            validation_result = await self._validate_workflow_design(workflow)
            if not validation_result['valid']:
                self.logger.warning(f"Workflow validation issues: {validation_result['issues']}")
            
            self.logger.info(f"Workflow designed successfully: {workflow.workflow_id}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Error designing workflow: {e}")
            raise
    
    async def _ai_generate_workflow(self, requirements: Dict[str, Any]) -> WorkflowDefinition:
        """AI-powered workflow generation based on requirements"""
        try:
            # Analyze requirements to determine workflow pattern
            required_capabilities = requirements.get('required_capabilities', [])
            objective = requirements.get('objective', '')
            
            # Select best matching pattern
            pattern = self._select_best_pattern(required_capabilities, objective)
            
            # Generate tasks based on pattern and requirements
            tasks = await self._generate_intelligent_tasks(requirements, pattern)
            
            # Optimize task dependencies
            tasks = await self._optimize_task_dependencies(tasks, requirements)
            
            # Create workflow definition
            workflow = WorkflowDefinition(
                workflow_id=f"ai_workflow_{uuid.uuid4().hex[:8]}",
                name=requirements.get('name', 'AI Generated Workflow'),
                description=requirements.get('description', f'AI-generated workflow for {objective}'),
                tasks=tasks,
                priority=WorkflowPriority(requirements.get('priority', 'medium').upper()),
                optimization_objective=OptimizationObjective(requirements.get('optimization_objective', 'balance_all')),
                design_mode=WorkflowDesignMode.AI_GENERATED,
                execution_strategy=ExecutionStrategy(requirements.get('execution_strategy', 'adaptive')),
                max_parallel_tasks=requirements.get('max_parallel_tasks', 5),
                total_timeout=requirements.get('total_timeout'),
                success_criteria=requirements.get('success_criteria', []),
                required_capabilities=required_capabilities,
                metadata=requirements.get('metadata', {})
            )
            
            return workflow
            
        except Exception as e:
            self.logger.error(f"AI workflow generation failed: {e}")
            raise
    
    def _select_best_pattern(self, required_capabilities: List[str], objective: str) -> Dict[str, Any]:
        """Select the best workflow pattern based on requirements"""
        best_pattern = None
        best_score = 0.0
        
        for pattern_name, pattern in self.design_patterns.items():
            score = self._calculate_pattern_match_score(pattern, required_capabilities, objective)
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        # If no good match, create a generic pattern
        if best_pattern is None or best_score < 0.3:
            best_pattern = {
                'description': 'Generic workflow pattern',
                'task_sequence': ['preparation', 'execution', 'analysis', 'completion'],
                'optimization_hints': ['parallel_execution'],
                'required_capabilities': required_capabilities
            }
        
        return best_pattern
    
    def _calculate_pattern_match_score(self, pattern: Dict[str, Any], 
                                     required_capabilities: List[str], objective: str) -> float:
        """Calculate how well a pattern matches the requirements"""
        score = 0.0
        
        # Capability match score
        pattern_capabilities = pattern.get('required_capabilities', [])
        if required_capabilities:
            capability_overlap = len(set(required_capabilities) & set(pattern_capabilities))
            capability_score = capability_overlap / len(required_capabilities)
            score += capability_score * 0.6
        
        # Objective match score (simplified text matching)
        objective_keywords = objective.lower().split()
        pattern_description = pattern.get('description', '').lower()
        
        keyword_matches = sum(1 for keyword in objective_keywords if keyword in pattern_description)
        if objective_keywords:
            objective_score = keyword_matches / len(objective_keywords)
            score += objective_score * 0.4
        
        return score
    
    async def _generate_intelligent_tasks(self, requirements: Dict[str, Any], 
                                        pattern: Dict[str, Any]) -> List[WorkflowTask]:
        """Generate intelligent tasks based on pattern and requirements"""
        tasks = []
        task_sequence = pattern.get('task_sequence', ['generic_task'])
        
        for i, task_type in enumerate(task_sequence):
            # Determine target system for task
            target_system = await self._select_optimal_system(task_type, requirements)
            
            # Generate task parameters
            task_params = self._generate_task_parameters(task_type, requirements)
            
            # Estimate task duration
            estimated_duration = self._estimate_task_duration(task_type, task_params)
            
            # Create task
            task = WorkflowTask(
                task_id=f"task_{i+1}_{task_type}",
                name=self._generate_task_name(task_type),
                task_type=task_type,
                target_system=target_system,
                parameters=task_params,
                estimated_duration=estimated_duration,
                timeout=estimated_duration * 3,  # Allow 3x estimated time
                success_criteria=self._generate_task_success_criteria(task_type),
                resource_requirements=self._estimate_resource_requirements(task_type)
            )
            
            tasks.append(task)
        
        return tasks
    
    async def _select_optimal_system(self, task_type: str, requirements: Dict[str, Any]) -> str:
        """Select optimal system for task execution"""
        objective = OptimizationObjective(requirements.get('optimization_objective', 'balance_all'))
        
        best_system = None
        best_score = 0.0
        
        for system_id, capabilities in self.system_capabilities.items():
            if self._system_can_handle_task_type(capabilities, task_type):
                score = capabilities.calculate_suitability_score_for_type(task_type, objective)
                if score > best_score:
                    best_score = score
                    best_system = system_id
        
        # Default to 'intelligence' if no specific system found
        return best_system or 'intelligence'
    
    def _system_can_handle_task_type(self, capabilities: SystemCapabilities, task_type: str) -> bool:
        """Check if system can handle specific task type"""
        # Simplified capability matching
        capability_mapping = {
            'data_collection': ['data_processing', 'extraction'],
            'preprocessing': ['data_processing', 'transformation'],
            'analysis': ['analysis', 'pattern_detection'],
            'visualization': ['visualization', 'reporting'],
            'machine_learning': ['ml', 'training', 'prediction'],
            'security_analysis': ['security', 'vulnerability_assessment']
        }
        
        required_caps = capability_mapping.get(task_type, [task_type])
        return any(cap in capabilities.available_capabilities for cap in required_caps)
    
    def _generate_task_parameters(self, task_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent task parameters"""
        base_params = {
            'task_type': task_type,
            'created_at': datetime.now().isoformat(),
            'auto_generated': True
        }
        
        # Add task-specific parameters
        if task_type == 'data_collection':
            base_params.update({
                'data_sources': requirements.get('data_sources', ['default']),
                'collection_method': 'intelligent',
                'quality_threshold': 0.8
            })
        elif task_type == 'analysis':
            base_params.update({
                'analysis_methods': ['pattern_detection', 'statistical_analysis'],
                'confidence_threshold': 0.85,
                'depth': 'comprehensive'
            })
        elif task_type == 'machine_learning':
            base_params.update({
                'model_types': ['ensemble'],
                'validation_method': 'cross_validation',
                'optimization_metric': 'f1_score'
            })
        
        # Merge with user-provided parameters
        user_params = requirements.get('task_parameters', {}).get(task_type, {})
        base_params.update(user_params)
        
        return base_params
    
    def _estimate_task_duration(self, task_type: str, parameters: Dict[str, Any]) -> float:
        """Estimate task duration based on type and parameters"""
        # Base duration estimates (in seconds)
        base_durations = {
            'data_collection': 30.0,
            'preprocessing': 20.0,
            'analysis': 45.0,
            'visualization': 15.0,
            'machine_learning': 120.0,
            'security_analysis': 60.0,
            'reporting': 25.0,
            'validation': 30.0,
            'synthesis': 40.0,
            'optimization': 50.0
        }
        
        base_duration = base_durations.get(task_type, 30.0)
        
        # Adjust based on parameters
        complexity_factor = 1.0
        if 'depth' in parameters:
            depth_factors = {'simple': 0.7, 'standard': 1.0, 'comprehensive': 1.5, 'deep': 2.0}
            complexity_factor *= depth_factors.get(parameters['depth'], 1.0)
        
        if 'data_size' in parameters:
            size_factors = {'small': 0.8, 'medium': 1.0, 'large': 1.5, 'xlarge': 2.5}
            complexity_factor *= size_factors.get(parameters['data_size'], 1.0)
        
        return base_duration * complexity_factor
    
    def _generate_task_name(self, task_type: str) -> str:
        """Generate human-readable task name"""
        name_mapping = {
            'data_collection': 'Data Collection',
            'preprocessing': 'Data Preprocessing',
            'analysis': 'Intelligent Analysis',
            'visualization': 'Data Visualization',
            'machine_learning': 'ML Model Training',
            'security_analysis': 'Security Assessment',
            'reporting': 'Report Generation',
            'validation': 'Result Validation',
            'synthesis': 'Intelligence Synthesis',
            'optimization': 'Performance Optimization'
        }
        
        return name_mapping.get(task_type, task_type.replace('_', ' ').title())
    
    def _generate_task_success_criteria(self, task_type: str) -> List[str]:
        """Generate success criteria for task type"""
        criteria_mapping = {
            'data_collection': ['Data quality > 80%', 'Collection completed without errors'],
            'analysis': ['Analysis confidence > 85%', 'Results validated'],
            'machine_learning': ['Model accuracy > 90%', 'Validation successful'],
            'security_analysis': ['Security scan completed', 'Vulnerabilities identified'],
            'reporting': ['Report generated successfully', 'All sections included']
        }
        
        return criteria_mapping.get(task_type, ['Task completed successfully'])
    
    def _estimate_resource_requirements(self, task_type: str) -> Dict[str, float]:
        """Estimate resource requirements for task type"""
        requirements_mapping = {
            'data_collection': {'cpu': 2.0, 'memory': 1.0, 'network': 0.5},
            'preprocessing': {'cpu': 3.0, 'memory': 2.0, 'storage': 1.0},
            'analysis': {'cpu': 4.0, 'memory': 3.0, 'compute': 2.0},
            'machine_learning': {'cpu': 8.0, 'memory': 4.0, 'gpu': 2.0},
            'visualization': {'cpu': 2.0, 'memory': 1.5, 'graphics': 1.0},
            'security_analysis': {'cpu': 3.0, 'memory': 2.0, 'network': 1.0}
        }
        
        return requirements_mapping.get(task_type, {'cpu': 2.0, 'memory': 1.0})
    
    async def _optimize_task_dependencies(self, tasks: List[WorkflowTask], 
                                        requirements: Dict[str, Any]) -> List[WorkflowTask]:
        """Optimize task dependencies for better execution"""
        if len(tasks) <= 1:
            return tasks
        
        # Create dependency graph based on logical task order
        dependency_rules = {
            'data_collection': [],
            'preprocessing': ['data_collection'],
            'analysis': ['preprocessing'],
            'visualization': ['analysis'],
            'machine_learning': ['preprocessing'],
            'validation': ['analysis', 'machine_learning'],
            'synthesis': ['analysis', 'validation'],
            'reporting': ['synthesis', 'visualization']
        }
        
        # Apply dependencies
        task_map = {task.task_type: task for task in tasks}
        
        for task in tasks:
            task_type = task.task_type
            if task_type in dependency_rules:
                for dep_type in dependency_rules[task_type]:
                    if dep_type in task_map:
                        dep_task = task_map[dep_type]
                        if dep_task.task_id not in task.dependencies:
                            task.dependencies.append(dep_task.task_id)
        
        # Optimize for parallelism where possible
        max_parallel = requirements.get('max_parallel_tasks', 5)
        if max_parallel > 1:
            await self._optimize_for_parallelism(tasks, max_parallel)
        
        return tasks
    
    async def _optimize_for_parallelism(self, tasks: List[WorkflowTask], max_parallel: int):
        """Optimize task dependencies for maximum parallelism"""
        # Remove unnecessary sequential dependencies where tasks can run in parallel
        parallel_task_types = ['visualization', 'validation', 'reporting']
        
        for task in tasks:
            if task.task_type in parallel_task_types:
                # Remove some dependencies to allow parallel execution
                if len(task.dependencies) > 1:
                    # Keep only the most critical dependency
                    critical_deps = [dep for dep in task.dependencies 
                                   if any(t.task_id == dep and t.task_type in ['analysis', 'preprocessing'] 
                                         for t in tasks)]
                    if critical_deps:
                        task.dependencies = critical_deps[:1]
    
    async def _design_from_template(self, requirements: Dict[str, Any]) -> WorkflowDefinition:
        """Design workflow from existing template"""
        template_id = requirements.get('template_id')
        if not template_id or template_id not in self.workflow_templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self.workflow_templates[template_id]
        
        # Instantiate workflow from template
        workflow = template.instantiate_workflow(requirements.get('parameters', {}))
        
        # Update template usage statistics
        template.usage_count += 1
        
        return workflow
    
    async def _hybrid_design_workflow(self, requirements: Dict[str, Any]) -> WorkflowDefinition:
        """Design workflow using hybrid approach (template + AI enhancements)"""
        # Start with template if specified
        if 'template_id' in requirements:
            workflow = await self._design_from_template(requirements)
        else:
            workflow = await self._ai_generate_workflow(requirements)
        
        # Apply AI enhancements
        enhanced_tasks = await self._enhance_tasks_with_ai(workflow.tasks, requirements)
        workflow.tasks = enhanced_tasks
        
        return workflow
    
    async def _manual_design_workflow(self, requirements: Dict[str, Any]) -> WorkflowDefinition:
        """Design workflow from manual specifications"""
        tasks = []
        task_specs = requirements.get('tasks', [])
        
        for i, spec in enumerate(task_specs):
            task = WorkflowTask(
                task_id=spec.get('task_id', f"manual_task_{i+1}"),
                name=spec.get('name', f"Manual Task {i+1}"),
                task_type=spec.get('task_type', 'generic'),
                target_system=spec.get('target_system', 'default'),
                parameters=spec.get('parameters', {}),
                dependencies=spec.get('dependencies', []),
                estimated_duration=spec.get('estimated_duration', 30.0)
            )
            tasks.append(task)
        
        workflow = WorkflowDefinition(
            workflow_id=f"manual_workflow_{uuid.uuid4().hex[:8]}",
            name=requirements.get('name', 'Manual Workflow'),
            description=requirements.get('description', 'Manually designed workflow'),
            tasks=tasks,
            design_mode=WorkflowDesignMode.MANUAL
        )
        
        return workflow
    
    async def _apply_design_optimizations(self, workflow: WorkflowDefinition, 
                                        requirements: Dict[str, Any]):
        """Apply design-time optimizations to workflow"""
        objective = workflow.optimization_objective
        
        # Apply objective-specific optimizations
        if objective == OptimizationObjective.MINIMIZE_TIME:
            await self._optimize_for_speed(workflow)
        elif objective == OptimizationObjective.MINIMIZE_COST:
            await self._optimize_for_cost(workflow)
        elif objective == OptimizationObjective.MAXIMIZE_RELIABILITY:
            await self._optimize_for_reliability(workflow)
        
        # Apply general optimizations
        await self._apply_general_optimizations(workflow)
    
    async def _optimize_for_speed(self, workflow: WorkflowDefinition):
        """Optimize workflow for execution speed"""
        # Increase parallelism
        workflow.max_parallel_tasks = min(10, len(workflow.tasks))
        
        # Reduce task timeouts
        for task in workflow.tasks:
            if task.timeout:
                task.timeout = task.timeout * 0.8
        
        # Select fastest systems
        for task in workflow.tasks:
            if task.target_system == 'default':
                task.target_system = await self._select_fastest_system(task.task_type)
    
    async def _optimize_for_cost(self, workflow: WorkflowDefinition):
        """Optimize workflow for cost efficiency"""
        # Select cost-effective systems
        for task in workflow.tasks:
            if task.target_system == 'default':
                task.target_system = await self._select_cheapest_system(task.task_type)
        
        # Reduce resource requirements
        for task in workflow.tasks:
            for resource, amount in task.resource_requirements.items():
                task.resource_requirements[resource] = amount * 0.8
    
    async def _optimize_for_reliability(self, workflow: WorkflowDefinition):
        """Optimize workflow for reliability"""
        # Increase retry attempts
        for task in workflow.tasks:
            task.max_retries = min(5, task.max_retries + 2)
        
        # Select most reliable systems
        for task in workflow.tasks:
            if task.target_system == 'default':
                task.target_system = await self._select_most_reliable_system(task.task_type)
    
    async def _apply_general_optimizations(self, workflow: WorkflowDefinition):
        """Apply general workflow optimizations"""
        # Remove redundant dependencies
        await self._remove_redundant_dependencies(workflow)
        
        # Optimize task ordering
        await self._optimize_task_ordering(workflow)
    
    async def _validate_workflow_design(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Validate workflow design for common issues"""
        issues = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies(workflow):
            issues.append("Circular dependencies detected")
        
        # Check for orphaned tasks
        orphaned_tasks = self._find_orphaned_tasks(workflow)
        if orphaned_tasks:
            issues.append(f"Orphaned tasks found: {orphaned_tasks}")
        
        # Check resource requirements
        total_resources = self._calculate_total_resources(workflow)
        if any(amount > 100 for amount in total_resources.values()):
            issues.append("High resource requirements detected")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': []
        }
    
    def _has_circular_dependencies(self, workflow: WorkflowDefinition) -> bool:
        """Check for circular dependencies in workflow"""
        # Create dependency graph
        graph = nx.DiGraph()
        
        for task in workflow.tasks:
            graph.add_node(task.task_id)
            for dep in task.dependencies:
                graph.add_edge(dep, task.task_id)
        
        return not nx.is_directed_acyclic_graph(graph)
    
    def _find_orphaned_tasks(self, workflow: WorkflowDefinition) -> List[str]:
        """Find tasks that are not connected to the workflow"""
        # This is a simplified check - could be more sophisticated
        all_task_ids = {task.task_id for task in workflow.tasks}
        referenced_ids = set()
        
        for task in workflow.tasks:
            referenced_ids.update(task.dependencies)
        
        # Tasks that are referenced but don't exist
        return list(referenced_ids - all_task_ids)
    
    def _calculate_total_resources(self, workflow: WorkflowDefinition) -> Dict[str, float]:
        """Calculate total resource requirements"""
        total_resources = defaultdict(float)
        
        for task in workflow.tasks:
            for resource, amount in task.resource_requirements.items():
                total_resources[resource] += amount
        
        return dict(total_resources)
    
    # Utility methods for system selection
    async def _select_fastest_system(self, task_type: str) -> str:
        """Select fastest system for task type"""
        # Simplified implementation
        return 'high_performance'
    
    async def _select_cheapest_system(self, task_type: str) -> str:
        """Select most cost-effective system"""
        return 'cost_effective'
    
    async def _select_most_reliable_system(self, task_type: str) -> str:
        """Select most reliable system"""
        return 'reliable'
    
    async def _enhance_tasks_with_ai(self, tasks: List[WorkflowTask], 
                                   requirements: Dict[str, Any]) -> List[WorkflowTask]:
        """Enhance tasks with AI optimizations"""
        # Placeholder for AI enhancements
        return tasks
    
    async def _remove_redundant_dependencies(self, workflow: WorkflowDefinition):
        """Remove redundant task dependencies"""
        # Simplified implementation
        pass
    
    async def _optimize_task_ordering(self, workflow: WorkflowDefinition):
        """Optimize task execution order"""
        # Simplified implementation
        pass
    
    # Public API methods
    def register_system_capabilities(self, system_id: str, capabilities: SystemCapabilities):
        """Register system capabilities for workflow design"""
        self.system_capabilities[system_id] = capabilities
        self.logger.info(f"Registered system capabilities: {system_id}")
    
    def add_workflow_template(self, template: WorkflowTemplate):
        """Add workflow template to designer"""
        self.workflow_templates[template.template_id] = template
        self.logger.info(f"Added workflow template: {template.template_id}")
    
    def get_design_recommendations(self, requirements: Dict[str, Any]) -> List[str]:
        """Get design recommendations for requirements"""
        recommendations = []
        
        # Analyze requirements and provide suggestions
        if 'optimization_objective' not in requirements:
            recommendations.append("Consider specifying optimization objective for better performance")
        
        if 'max_parallel_tasks' not in requirements:
            recommendations.append("Specify parallel task limit for optimal resource usage")
        
        required_caps = requirements.get('required_capabilities', [])
        if len(required_caps) > 5:
            recommendations.append("Consider breaking down into smaller workflows")
        
        return recommendations


# Export main classes
__all__ = [
    'IntelligentWorkflowDesigner'
]