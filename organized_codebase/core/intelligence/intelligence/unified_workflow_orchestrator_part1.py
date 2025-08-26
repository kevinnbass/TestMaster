"""
Unified ML Workflow Orchestrator
Enterprise workflow orchestration for cross-agent ML operations
"""Core Module - Split from unified_workflow_orchestrator.py"""


import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from collections import defaultdict, deque
from pathlib import Path
import uuid
from enum import Enum
import yaml
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


class WorkflowType(Enum):
    """Types of ML workflows"""
    ML_TRAINING = "ml_training"
    DATA_PIPELINE = "data_pipeline"
    INFERENCE_PIPELINE = "inference_pipeline"
    MONITORING_WORKFLOW = "monitoring_workflow"
    INTEGRATION_WORKFLOW = "integration_workflow"
    TESTING_WORKFLOW = "testing_workflow"
    DEPLOYMENT_WORKFLOW = "deployment_workflow"

class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class TaskPriority(Enum):
    """Task execution priorities"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class WorkflowResource:
    """Resource requirements for workflow execution"""
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    storage_gb: float
    network_bandwidth_mbps: int
    estimated_duration_minutes: int
    resource_tags: List[str] = field(default_factory=list)

@dataclass
class WorkflowStep:
    """Individual step within a workflow"""
    step_id: str
    workflow_id: str
    name: str
    description: str
    agent_type: str
    operation_type: str
    parameters: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    dependencies: Set[str]
    resource_requirements: WorkflowResource
    priority: TaskPriority
    retry_policy: Dict[str, Any]
    timeout_seconds: int
    status: str = "pending"
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    execution_duration: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

@dataclass
class WorkflowTemplate:
    """Reusable workflow template"""
    template_id: str
    name: str
    description: str
    version: str
    workflow_type: WorkflowType
    execution_mode: ExecutionMode
    steps: List[Dict[str, Any]]
    default_parameters: Dict[str, Any]
    resource_profile: WorkflowResource
    tags: List[str]
    created_by: str
    created_time: datetime
    usage_count: int = 0

@dataclass
class MLWorkflowInstance:
    """Complete ML workflow instance"""
    workflow_id: str
    template_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    execution_mode: ExecutionMode
    steps: List[WorkflowStep]
    global_context: Dict[str, Any]
    parameters: Dict[str, Any]
    created_by: str
    created_time: datetime
    scheduled_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    status: str = "created"
    progress_percentage: float = 0.0
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    estimated_completion: Optional[datetime] = None
    actual_resources_used: Optional[WorkflowResource] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution tracking"""
    execution_id: str
    workflow_id: str
    execution_node: str
    start_time: datetime
    status: str
    current_step: Optional[str] = None
    progress_details: Dict[str, Any] = field(default_factory=dict)
    resource_allocation: Optional[Dict[str, Any]] = None
    performance_data: Dict[str, Any] = field(default_factory=dict)

class UnifiedWorkflowOrchestrator:
    """
    Unified ML Workflow Orchestrator
    
    Provides enterprise-grade orchestration for complex ML workflows
    across multiple agents with intelligent optimization and monitoring.
    """
    
    def __init__(self, orchestrator_id: str = None, config_path: str = "workflow_config.json"):
        self.orchestrator_id = orchestrator_id or f"orchestrator_{uuid.uuid4().hex[:8]}"
        self.config_path = config_path
        
        # Workflow storage
        self.workflow_templates = {}
        self.active_workflows = {}
        self.workflow_history = deque(maxlen=1000)
        self.execution_queue = deque()
        self.running_executions = {}
        
        # Orchestration configuration
        self.orchestration_config = {
            "execution_engine": {
                "max_concurrent_workflows": 20,
                "max_parallel_steps": 10,
                "default_timeout_seconds": 3600,
                "retry_attempts": 3,
                "retry_delay_seconds": 30,
                "resource_optimization": True,
                "adaptive_scheduling": True
            },
            "resource_management": {
                "cpu_allocation_strategy": "fair_share",
                "memory_allocation_strategy": "demand_based",
                "gpu_allocation_strategy": "exclusive",
                "storage_allocation_strategy": "shared",
                "resource_monitoring_interval": 30
            },
            "optimization": {
                "workflow_optimization": True,
                "step_parallelization": True,
                "resource_prediction": True,
                "performance_learning": True,
                "cost_optimization": True,
                "completion_time_prediction": True
            },
            "persistence": {
                "database_path": "workflows.db",
                "checkpoint_interval": 300,
                "backup_interval": 3600,
                "retention_days": 30
            },
            "monitoring": {
                "real_time_monitoring": True,
                "performance_tracking": True,
                "anomaly_detection": True,
                "predictive_alerting": True,
                "dashboard_updates": True
            }
        }
        
        self.logger = logging.getLogger(__name__)
        self.orchestration_active = True
        self.resource_manager = None
        self.performance_optimizer = None
        
        # Initialize components
        self._initialize_database()
        self._initialize_templates()
        self._initialize_resource_manager()
        self._start_orchestration_threads()
    
    def _initialize_database(self):
        """Initialize workflow persistence database"""
        
        db_path = self.orchestration_config["persistence"]["database_path"]
        self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
        self.db_lock = threading.Lock()
        
        # Create tables
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            # Workflow templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_templates (
                    template_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    version TEXT,
                    workflow_type TEXT,
                    execution_mode TEXT,
                    template_data TEXT,
                    created_time TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            ''')
            
            # Workflow instances table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_instances (
                    workflow_id TEXT PRIMARY KEY,
                    template_id TEXT,
                    name TEXT,
                    status TEXT,
                    created_time TIMESTAMP,
                    start_time TIMESTAMP,
                    completion_time TIMESTAMP,
                    progress_percentage REAL,
                    workflow_data TEXT
                )
            ''')
            
            # Workflow executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    execution_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    execution_node TEXT,
                    start_time TIMESTAMP,
                    status TEXT,
                    execution_data TEXT
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_metrics (
                    metric_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    step_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP
                )
            ''')
            
            self.db_connection.commit()
    
    def _initialize_templates(self):
        """Initialize predefined workflow templates"""
        
        # ML Model Training Template
        self.create_workflow_template(
            name="ML Model Training Pipeline",
            description="Complete ML model training with data preprocessing and validation",
            workflow_type=WorkflowType.ML_TRAINING,
            execution_mode=ExecutionMode.HYBRID,
            steps=[
                {
                    "name": "Data Validation",
                    "agent_type": "ml_intelligence",
                    "operation_type": "data_validation",
                    "parameters": {"validation_rules": ["completeness", "consistency", "accuracy"]},
                    "dependencies": [],
                    "timeout_seconds": 300
                },
                {
                    "name": "Feature Engineering",
                    "agent_type": "ml_intelligence", 
                    "operation_type": "feature_engineering",
                    "parameters": {"feature_selection": "auto", "scaling": "standard"},
                    "dependencies": ["Data Validation"],
                    "timeout_seconds": 600
                },
                {
                    "name": "Model Training",
                    "agent_type": "ml_intelligence",
                    "operation_type": "model_training",
                    "parameters": {"algorithm": "auto", "cross_validation": True},
                    "dependencies": ["Feature Engineering"],
                    "timeout_seconds": 3600
                },
                {
                    "name": "Model Validation",
                    "agent_type": "testing",
                    "operation_type": "model_validation",
                    "parameters": {"validation_split": 0.2, "metrics": ["accuracy", "precision", "recall"]},
                    "dependencies": ["Model Training"],
                    "timeout_seconds": 900
                },
                {
                    "name": "Model Deployment",
                    "agent_type": "deployment",
                    "operation_type": "model_deployment",
                    "parameters": {"deployment_strategy": "canary", "rollback_enabled": True},
                    "dependencies": ["Model Validation"],
                    "timeout_seconds": 1200
                }
            ],
            default_parameters={
                "dataset_path": "/data/training",
                "model_output_path": "/models/output",
                "performance_threshold": 0.85
            },
            tags=["ml", "training", "production"]
        )
        
        # Real-time Inference Pipeline Template
        self.create_workflow_template(
            name="Real-time ML Inference Pipeline",
            description="High-throughput ML inference with monitoring and optimization",
            workflow_type=WorkflowType.INFERENCE_PIPELINE,
            execution_mode=ExecutionMode.PARALLEL,
            steps=[
                {
                    "name": "Input Validation",
                    "agent_type": "ml_intelligence",
                    "operation_type": "input_validation",
                    "parameters": {"schema_validation": True, "data_cleansing": True},
                    "dependencies": [],
                    "timeout_seconds": 100
                },
                {
                    "name": "Feature Extraction",
                    "agent_type": "ml_intelligence",
                    "operation_type": "feature_extraction",
                    "parameters": {"feature_cache": True, "parallel_processing": True},
                    "dependencies": ["Input Validation"],
                    "timeout_seconds": 200
                },
                {
                    "name": "Model Inference",
                    "agent_type": "ml_intelligence",
                    "operation_type": "model_inference",
                    "parameters": {"batch_size": 32, "gpu_acceleration": True},
                    "dependencies": ["Feature Extraction"],
                    "timeout_seconds": 500
                },
                {
                    "name": "Result Processing",
                    "agent_type": "ml_intelligence",
                    "operation_type": "result_processing",
                    "parameters": {"confidence_threshold": 0.7, "result_formatting": "json"},
                    "dependencies": ["Model Inference"],
                    "timeout_seconds": 100
                },
                {
                    "name": "Performance Monitoring",
                    "agent_type": "monitoring",
                    "operation_type": "performance_monitoring",
                    "parameters": {"latency_tracking": True, "accuracy_monitoring": True},
                    "dependencies": ["Result Processing"],
                    "timeout_seconds": 50
                }
            ],
            default_parameters={
                "model_endpoint": "/api/v1/ml/inference",
                "max_latency_ms": 1000,
                "min_accuracy": 0.9
            },
            tags=["inference", "real-time", "production"]
        )
        
        # System Integration Testing Template
        self.create_workflow_template(
            name="ML System Integration Testing",
            description="Comprehensive testing workflow for ML system components",
            workflow_type=WorkflowType.TESTING_WORKFLOW,
            execution_mode=ExecutionMode.SEQUENTIAL,
            steps=[
                {
                    "name": "Unit Test Execution",
                    "agent_type": "testing",
                    "operation_type": "unit_testing",
                    "parameters": {"coverage_threshold": 0.8, "parallel_execution": True},
                    "dependencies": [],
                    "timeout_seconds": 1800
                },
                {
                    "name": "Integration Test Execution",
                    "agent_type": "integration",
                    "operation_type": "integration_testing",
                    "parameters": {"test_environments": ["staging", "pre-prod"]},
                    "dependencies": ["Unit Test Execution"],
                    "timeout_seconds": 2400
                },
                {
                    "name": "Performance Testing",
                    "agent_type": "testing",
                    "operation_type": "performance_testing",
                    "parameters": {"load_profile": "standard", "duration_minutes": 30},
                    "dependencies": ["Integration Test Execution"],
                    "timeout_seconds": 3600
                },
                {
                    "name": "ML Model Accuracy Testing",
                    "agent_type": "ml_intelligence",
                    "operation_type": "accuracy_testing",
                    "parameters": {"test_dataset": "/data/validation", "accuracy_threshold": 0.85},
                    "dependencies": ["Performance Testing"],
                    "timeout_seconds": 1200
                },
                {
                    "name": "Test Report Generation",
                    "agent_type": "testing",
                    "operation_type": "report_generation",
                    "parameters": {"format": "html", "include_visualizations": True},
                    "dependencies": ["ML Model Accuracy Testing"],
                    "timeout_seconds": 300
                }
            ],
            default_parameters={
                "test_suite_path": "/tests",
                "report_output_path": "/reports",
                "notification_enabled": True
            },
            tags=["testing", "integration", "quality"]
        )
    
    def _initialize_resource_manager(self):
        """Initialize resource management system"""
        
        self.available_resources = {
            "cpu_cores": 64,
            "memory_gb": 256.0,
            "gpu_count": 8,
            "storage_gb": 10000.0,
            "network_bandwidth_mbps": 10000
        }
        
        self.allocated_resources = {
            "cpu_cores": 0,
            "memory_gb": 0.0,
            "gpu_count": 0,
            "storage_gb": 0.0,
            "network_bandwidth_mbps": 0
        }
        
        self.resource_reservations = {}
    
    def _start_orchestration_threads(self):
        """Start background orchestration threads"""
        
        # Workflow execution thread
        execution_thread = threading.Thread(target=self._workflow_execution_loop, daemon=True)
        execution_thread.start()
        
        # Resource monitoring thread
        resource_thread = threading.Thread(target=self._resource_monitoring_loop, daemon=True)
        resource_thread.start()
        
        # Performance optimization thread
        optimization_thread = threading.Thread(target=self._performance_optimization_loop, daemon=True)
        optimization_thread.start()
        
        # Checkpoint thread
        checkpoint_thread = threading.Thread(target=self._checkpoint_loop, daemon=True)
        checkpoint_thread.start()
        
        # Monitoring thread
        monitoring_thread = threading.Thread(target=self._workflow_monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def create_workflow_template(self, name: str, description: str, 
                                workflow_type: WorkflowType, execution_mode: ExecutionMode,
                                steps: List[Dict[str, Any]], default_parameters: Dict[str, Any],
                                tags: List[str] = None) -> str:
        """Create a new workflow template"""
        
        template_id = str(uuid.uuid4())
        
        # Calculate resource profile
        resource_profile = self._calculate_template_resource_profile(steps)
        
        template = WorkflowTemplate(
            template_id=template_id,
            name=name,
            description=description,
            version="1.0.0",
            workflow_type=workflow_type,
            execution_mode=execution_mode,
            steps=steps,
            default_parameters=default_parameters,
            resource_profile=resource_profile,
            tags=tags or [],
            created_by=self.orchestrator_id,
            created_time=datetime.now()
        )
        
        # Store template
        self.workflow_templates[template_id] = template
        
        # Persist to database
        self._persist_template(template)
        
        self.logger.info(f"Created workflow template: {name} ({template_id})")
        
        return template_id
    
    def create_workflow_from_template(self, template_id: str, name: str = None,
                                     parameters: Dict[str, Any] = None,
                                     scheduled_time: datetime = None) -> str:
        """Create a workflow instance from a template"""
        
        if template_id not in self.workflow_templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self.workflow_templates[template_id]
        workflow_id = str(uuid.uuid4())
        
        # Merge parameters
        merged_parameters = template.default_parameters.copy()
        if parameters:
            merged_parameters.update(parameters)
        
        # Create workflow steps
        workflow_steps = []
        for i, step_def in enumerate(template.steps):
            step_id = f"{workflow_id}_step_{i}"
            
            # Create resource requirements
            resource_req = WorkflowResource(
                cpu_cores=step_def.get("cpu_cores", 2),
                memory_gb=step_def.get("memory_gb", 4.0),
                gpu_count=step_def.get("gpu_count", 0),
                storage_gb=step_def.get("storage_gb", 10.0),
                network_bandwidth_mbps=step_def.get("network_bandwidth", 100),
                estimated_duration_minutes=step_def.get("estimated_duration", 30),
                resource_tags=step_def.get("resource_tags", [])
            )
            
            step = WorkflowStep(
                step_id=step_id,
                workflow_id=workflow_id,
                name=step_def["name"],
                description=step_def.get("description", ""),
                agent_type=step_def["agent_type"],
                operation_type=step_def["operation_type"],
                parameters=step_def.get("parameters", {}),
                inputs=step_def.get("inputs", {}),
                outputs=step_def.get("outputs", {}),
                dependencies=set(step_def.get("dependencies", [])),
                resource_requirements=resource_req,
                priority=TaskPriority(step_def.get("priority", 3)),
                retry_policy=step_def.get("retry_policy", {"max_attempts": 3, "delay_seconds": 30}),
                timeout_seconds=step_def.get("timeout_seconds", 3600)
            )
            
            workflow_steps.append(step)
        
        # Create workflow instance
        workflow = MLWorkflowInstance(
            workflow_id=workflow_id,
            template_id=template_id,
            name=name or f"{template.name} - {workflow_id[:8]}",
            description=template.description,
            workflow_type=template.workflow_type,
            execution_mode=template.execution_mode,
            steps=workflow_steps,
            global_context={},
            parameters=merged_parameters,
            created_by=self.orchestrator_id,
            created_time=datetime.now(),
            scheduled_time=scheduled_time,
            total_steps=len(workflow_steps)
        )
        
        # Store workflow
        self.active_workflows[workflow_id] = workflow
        
        # Persist to database
        self._persist_workflow(workflow)
        
        # Update template usage count
        template.usage_count += 1
        
        self.logger.info(f"Created workflow from template: {workflow.name} ({workflow_id})")
        
        return workflow_id
    
    def schedule_workflow(self, workflow_id: str, scheduled_time: datetime = None) -> bool:
        """Schedule a workflow for execution"""
        
        if workflow_id not in self.active_workflows:
            self.logger.error(f"Workflow not found: {workflow_id}")
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow.status not in ["created", "scheduled"]:
            self.logger.error(f"Workflow cannot be scheduled - current status: {workflow.status}")
            return False
        
        # Set scheduled time
        workflow.scheduled_time = scheduled_time or datetime.now()
        workflow.status = "scheduled"
        
        # Add to execution queue
        self.execution_queue.append(workflow_id)
        
        self.logger.info(f"Scheduled workflow: {workflow.name} for {workflow.scheduled_time}")
        
        return True
    
    def execute_workflow(self, workflow_id: str) -> bool:
        """Execute a workflow immediately"""
        
        if workflow_id not in self.active_workflows:
            self.logger.error(f"Workflow not found: {workflow_id}")
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        # Check resource availability
        if not self._check_resource_availability(workflow):
            self.logger.warning(f"Insufficient resources for workflow: {workflow_id}")
            return False
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            execution_node=self.orchestrator_id,
            start_time=datetime.now(),
            status="running"
        )
        
        # Reserve resources
        self._reserve_workflow_resources(workflow)
        
        # Update workflow status
        workflow.status = "running"
        workflow.start_time = datetime.now()
        
        # Store execution
        self.running_executions[execution_id] = execution
        
        # Start workflow execution in background
        execution_thread = threading.Thread(
            target=self._execute_workflow_steps,
            args=(workflow, execution),
            daemon=True
        )
