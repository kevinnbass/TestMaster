"""
Workflow Definition Framework
============================

YAML-based workflow system enabling complex processes that span across
all unified systems with intelligent coordination and state management.

Integrates with:
- Cross-System APIs for system communication
- Unified State Manager for workflow state persistence
- ML Router for intelligent step execution
- Unified Dashboard for workflow visualization

Author: TestMaster Phase 1B Integration System
"""

import asyncio
import json
import logging
import time
import uuid
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Set
from concurrent.futures import ThreadPoolExecutor

# Import cross-system integration
from .cross_system_apis import (
    SystemType, IntegrationEventType, SystemMessage, CrossSystemRequest, 
    cross_system_coordinator
)


# ============================================================================
# WORKFLOW DEFINITION TYPES
# ============================================================================

class WorkflowStepType(Enum):
    """Types of workflow steps"""
    SYSTEM_OPERATION = "system_operation"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP = "loop"
    DELAY = "delay"
    HUMAN_APPROVAL = "human_approval"
    DATA_TRANSFORM = "data_transform"
    EXTERNAL_API = "external_api"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Individual step status"""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class WorkflowVariable:
    """Variable definition for workflows"""
    name: str
    type: str  # string, integer, float, boolean, object, array
    default_value: Any = None
    required: bool = False
    description: str = ""
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    name: str
    type: WorkflowStepType
    target_system: Optional[SystemType] = None
    operation: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    
    # Execution state
    status: StepStatus = StepStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "type": self.type.value,
            "target_system": self.target_system.value if self.target_system else None,
            "operation": self.operation,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "conditions": self.conditions,
            "retry_config": self.retry_config,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": self.result,
            "error_message": self.error_message,
            "retry_count": self.retry_count
        }


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str = "1.0.0"
    variables: List[WorkflowVariable] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution configuration
    max_parallel_steps: int = 10
    default_timeout_seconds: int = 3600
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_variable(self, name: str) -> Optional[WorkflowVariable]:
        """Get variable by name"""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def validate(self) -> List[str]:
        """Validate workflow definition"""
        errors = []
        
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
        
        # Check dependencies
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id} depends on non-existent step {dep}")
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected")
        
        return errors
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using topological sort"""
        # Build adjacency list
        graph = {step.step_id: step.depends_on for step in self.steps}
        
        # Kahn's algorithm for cycle detection
        in_degree = {step_id: 0 for step_id in graph}
        for step_id in graph:
            for dep in graph[step_id]:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        processed = 0
        
        while queue:
            current = queue.pop(0)
            processed += 1
            
            for step_id in graph:
                if current in graph[step_id]:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
        
        return processed != len(self.steps)


@dataclass 
class WorkflowExecution:
    """Runtime workflow execution instance"""
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    workflow_definition: WorkflowDefinition = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    
    # Execution state
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    running_steps: Set[str] = field(default_factory=set)
    
    # Performance metrics
    total_execution_time: float = 0.0
    step_timings: Dict[str, float] = field(default_factory=dict)
    
    def get_progress_percentage(self) -> float:
        """Calculate workflow progress percentage"""
        if not self.workflow_definition or not self.workflow_definition.steps:
            return 0.0
        
        total_steps = len(self.workflow_definition.steps)
        completed = len(self.completed_steps)
        
        return (completed / total_steps) * 100.0
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_definition.workflow_id if self.workflow_definition else None,
            "status": self.status.value,
            "progress_percentage": self.get_progress_percentage(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_execution_time": self.total_execution_time,
            "completed_steps": len(self.completed_steps),
            "failed_steps": len(self.failed_steps),
            "running_steps": len(self.running_steps),
            "total_steps": len(self.workflow_definition.steps) if self.workflow_definition else 0
        }


# ============================================================================
# YAML WORKFLOW PARSER
# ============================================================================

class WorkflowYAMLParser:
    """Parse YAML workflow definitions"""
    
    def __init__(self):
        self.logger = logging.getLogger("workflow_yaml_parser")
    
    def parse_workflow_file(self, file_path: Union[str, Path]) -> WorkflowDefinition:
        """Parse workflow from YAML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
            
            return self.parse_workflow_dict(yaml_content)
            
        except Exception as e:
            self.logger.error(f"Failed to parse workflow file {file_path}: {e}")
            raise
    
    def parse_workflow_dict(self, yaml_dict: Dict[str, Any]) -> WorkflowDefinition:
        """Parse workflow from dictionary"""
        try:
            # Extract basic workflow info
            workflow_id = yaml_dict.get("workflow_id", f"workflow_{uuid.uuid4().hex[:8]}")
            name = yaml_dict.get("name", "Unnamed Workflow")
            description = yaml_dict.get("description", "")
            version = yaml_dict.get("version", "1.0.0")
            
            # Parse variables
            variables = []
            for var_dict in yaml_dict.get("variables", []):
                variable = WorkflowVariable(
                    name=var_dict["name"],
                    type=var_dict.get("type", "string"),
                    default_value=var_dict.get("default"),
                    required=var_dict.get("required", False),
                    description=var_dict.get("description", ""),
                    validation_rules=var_dict.get("validation", {})
                )
                variables.append(variable)
            
            # Parse steps
            steps = []
            for step_dict in yaml_dict.get("steps", []):
                step = self._parse_step(step_dict)
                steps.append(step)
            
            # Parse execution configuration
            config = yaml_dict.get("execution_config", {})
            
            workflow = WorkflowDefinition(
                workflow_id=workflow_id,
                name=name,
                description=description,
                version=version,
                variables=variables,
                steps=steps,
                metadata=yaml_dict.get("metadata", {}),
                max_parallel_steps=config.get("max_parallel_steps", 10),
                default_timeout_seconds=config.get("default_timeout_seconds", 3600),
                retry_policy=config.get("retry_policy", {})
            )
            
            # Validate workflow
            errors = workflow.validate()
            if errors:
                raise ValueError(f"Workflow validation failed: {'; '.join(errors)}")
            
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to parse workflow dictionary: {e}")
            raise
    
    def _parse_step(self, step_dict: Dict[str, Any]) -> WorkflowStep:
        """Parse individual workflow step"""
        step_id = step_dict["step_id"]
        name = step_dict.get("name", step_id)
        step_type = WorkflowStepType(step_dict["type"])
        
        # Parse target system
        target_system = None
        if "target_system" in step_dict:
            target_system = SystemType(step_dict["target_system"])
        
        # Parse retry configuration
        retry_config = step_dict.get("retry", {})
        if "max_attempts" not in retry_config:
            retry_config["max_attempts"] = 3
        if "delay_seconds" not in retry_config:
            retry_config["delay_seconds"] = 5
        
        step = WorkflowStep(
            step_id=step_id,
            name=name,
            type=step_type,
            target_system=target_system,
            operation=step_dict.get("operation"),
            parameters=step_dict.get("parameters", {}),
            depends_on=step_dict.get("depends_on", []),
            conditions=step_dict.get("conditions", {}),
            retry_config=retry_config,
            timeout_seconds=step_dict.get("timeout_seconds", 300)
        )
        
        return step
    
    def export_workflow_to_yaml(self, workflow: WorkflowDefinition) -> str:
        """Export workflow definition to YAML string"""
        workflow_dict = {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "metadata": workflow.metadata
        }
        
        # Export variables
        if workflow.variables:
            workflow_dict["variables"] = [
                {
                    "name": var.name,
                    "type": var.type,
                    "default": var.default_value,
                    "required": var.required,
                    "description": var.description,
                    "validation": var.validation_rules
                }
                for var in workflow.variables
            ]
        
        # Export steps
        workflow_dict["steps"] = [
            {
                "step_id": step.step_id,
                "name": step.name,
                "type": step.type.value,
                "target_system": step.target_system.value if step.target_system else None,
                "operation": step.operation,
                "parameters": step.parameters,
                "depends_on": step.depends_on,
                "conditions": step.conditions,
                "retry": step.retry_config,
                "timeout_seconds": step.timeout_seconds
            }
            for step in workflow.steps
        ]
        
        # Export execution configuration
        workflow_dict["execution_config"] = {
            "max_parallel_steps": workflow.max_parallel_steps,
            "default_timeout_seconds": workflow.default_timeout_seconds,
            "retry_policy": workflow.retry_policy
        }
        
        return yaml.dump(workflow_dict, default_flow_style=False, sort_keys=False)


# ============================================================================
# WORKFLOW TEMPLATE LIBRARY
# ============================================================================

class WorkflowTemplateLibrary:
    """Library of pre-built workflow templates"""
    
    def __init__(self):
        self.logger = logging.getLogger("workflow_template_library")
        self.templates: Dict[str, str] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize built-in workflow templates"""
        
        # System Health Check Workflow
        self.templates["system_health_check"] = """
workflow_id: system_health_check
name: System Health Check
description: Comprehensive health check across all unified systems
version: 1.0.0

variables:
  - name: alert_threshold
    type: float
    default: 0.8
    description: Alert threshold for system performance

steps:
  - step_id: check_observability
    name: Check Observability System
    type: system_operation
    target_system: observability
    operation: health_check
    timeout_seconds: 30

  - step_id: check_state_config
    name: Check State & Config System
    type: system_operation
    target_system: state_config
    operation: health_check
    timeout_seconds: 30

  - step_id: check_orchestration
    name: Check Orchestration System
    type: system_operation
    target_system: orchestration
    operation: health_check
    timeout_seconds: 30

  - step_id: check_ui_dashboard
    name: Check UI Dashboard System
    type: system_operation
    target_system: ui_dashboard
    operation: health_check
    timeout_seconds: 30

  - step_id: generate_health_report
    name: Generate Health Report
    type: data_transform
    depends_on:
      - check_observability
      - check_state_config
      - check_orchestration
      - check_ui_dashboard
    parameters:
      report_format: json
      include_recommendations: true

execution_config:
  max_parallel_steps: 4
  default_timeout_seconds: 300
"""

        # Cross-System Data Pipeline
        self.templates["cross_system_pipeline"] = """
workflow_id: cross_system_pipeline
name: Cross-System Data Pipeline
description: ETL pipeline that processes data across all systems
version: 1.0.0

variables:
  - name: data_source
    type: string
    required: true
    description: Source of data to process
  - name: batch_size
    type: integer
    default: 1000
    description: Processing batch size

steps:
  - step_id: extract_data
    name: Extract Data
    type: system_operation
    target_system: observability
    operation: get_analytics
    parameters:
      source: "{{data_source}}"
      limit: "{{batch_size}}"

  - step_id: transform_data
    name: Transform Data
    type: data_transform
    depends_on:
      - extract_data
    parameters:
      transformations:
        - normalize_timestamps
        - aggregate_metrics
        - enrich_metadata

  - step_id: save_to_state
    name: Save to State Manager
    type: system_operation
    target_system: state_config
    operation: save_state
    depends_on:
      - transform_data
    parameters:
      state_key: "processed_data"
      data: "{{transform_data.result}}"

  - step_id: update_dashboard
    name: Update Dashboard
    type: system_operation
    target_system: ui_dashboard
    operation: update_widget
    depends_on:
      - save_to_state
    parameters:
      widget_id: "data_pipeline_status"
      data: "{{save_to_state.result}}"

  - step_id: trigger_orchestration
    name: Trigger Follow-up Orchestration
    type: system_operation
    target_system: orchestration
    operation: start_workflow
    depends_on:
      - update_dashboard
    parameters:
      workflow_type: "data_analysis"
      input_data: "{{save_to_state.result}}"

execution_config:
  max_parallel_steps: 2
  default_timeout_seconds: 1800
"""

        # Automated Scaling Workflow
        self.templates["automated_scaling"] = """
workflow_id: automated_scaling
name: Automated System Scaling
description: Automatically scale systems based on load and performance metrics
version: 1.0.0

variables:
  - name: cpu_threshold
    type: float
    default: 0.75
    description: CPU threshold for scaling trigger
  - name: memory_threshold
    type: float
    default: 0.80
    description: Memory threshold for scaling trigger

steps:
  - step_id: monitor_performance
    name: Monitor System Performance
    type: system_operation
    target_system: observability
    operation: get_metrics
    parameters:
      metrics:
        - cpu_usage
        - memory_usage
        - request_latency
      time_window: 300

  - step_id: evaluate_scaling_need
    name: Evaluate Scaling Need
    type: conditional
    depends_on:
      - monitor_performance
    conditions:
      cpu_high: "{{monitor_performance.result.cpu_usage > cpu_threshold}}"
      memory_high: "{{monitor_performance.result.memory_usage > memory_threshold}}"
      latency_high: "{{monitor_performance.result.request_latency > 1000}}"

  - step_id: scale_up_resources
    name: Scale Up Resources
    type: system_operation
    target_system: orchestration
    operation: scale_resources
    depends_on:
      - evaluate_scaling_need
    conditions:
      execute_if: "{{evaluate_scaling_need.cpu_high or evaluate_scaling_need.memory_high}}"
    parameters:
      action: scale_up
      factor: 1.5

  - step_id: update_monitoring
    name: Update Monitoring Configuration
    type: system_operation
    target_system: observability
    operation: update_config
    depends_on:
      - scale_up_resources
    parameters:
      alert_thresholds:
        cpu: "{{cpu_threshold}}"
        memory: "{{memory_threshold}}"

  - step_id: wait_for_stabilization
    name: Wait for System Stabilization
    type: delay
    depends_on:
      - update_monitoring
    parameters:
      seconds: 300

  - step_id: verify_scaling_success
    name: Verify Scaling Success
    type: system_operation
    target_system: observability
    operation: get_metrics
    depends_on:
      - wait_for_stabilization
    parameters:
      metrics:
        - cpu_usage
        - memory_usage
        - request_latency
      time_window: 120

execution_config:
  max_parallel_steps: 1
  default_timeout_seconds: 1200
"""

    def get_template(self, template_name: str) -> Optional[str]:
        """Get workflow template by name"""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def add_template(self, name: str, yaml_content: str):
        """Add custom template to library"""
        self.templates[name] = yaml_content
        self.logger.info(f"Added workflow template: {name}")
    
    def create_workflow_from_template(self, template_name: str, 
                                    variables: Dict[str, Any] = None) -> WorkflowDefinition:
        """Create workflow from template with variable substitution"""
        template_yaml = self.get_template(template_name)
        if not template_yaml:
            raise ValueError(f"Template {template_name} not found")
        
        # Simple variable substitution
        if variables:
            for var_name, var_value in variables.items():
                placeholder = f"{{{{{var_name}}}}}"
                template_yaml = template_yaml.replace(placeholder, str(var_value))
        
        # Parse workflow
        parser = WorkflowYAMLParser()
        workflow_dict = yaml.safe_load(template_yaml)
        return parser.parse_workflow_dict(workflow_dict)


# ============================================================================
# GLOBAL WORKFLOW FRAMEWORK INSTANCE
# ============================================================================

# Global instances for workflow framework
workflow_parser = WorkflowYAMLParser()
workflow_templates = WorkflowTemplateLibrary()

# Export for external use
__all__ = [
    'WorkflowStepType',
    'WorkflowStatus', 
    'StepStatus',
    'WorkflowVariable',
    'WorkflowStep',
    'WorkflowDefinition',
    'WorkflowExecution',
    'WorkflowYAMLParser',
    'WorkflowTemplateLibrary',
    'workflow_parser',
    'workflow_templates'
]