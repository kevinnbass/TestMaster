"""
Intelligent Workflow Engine - Main Orchestration Engine

This module implements the main IntelligentWorkflowEngine that coordinates
workflow design, execution, and optimization through its component subsystems.

Enhanced Hours 30-40: Integration with OrchestratorBase for unified processing

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
Enhanced: 2025-01-22 (Hours 30-40)
"""

import asyncio
import logging
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple

from .workflow_designer import WorkflowDesigner
from .workflow_scheduler import WorkflowScheduler
from .workflow_optimizer import WorkflowOptimizer
from .workflow_types import (
    WorkflowDefinition, WorkflowExecution, WorkflowOptimization,
    WorkflowStatus, OptimizationObjective
)

# Enhanced Hours 30-40: Orchestration base integration
try:
    from ..foundations.abstractions.orchestrator_base import (
        OrchestratorBase, OrchestratorType, OrchestratorCapabilities, ExecutionStrategy
    )
    ORCHESTRATION_BASE_AVAILABLE = True
except ImportError:
    ORCHESTRATION_BASE_AVAILABLE = False
    logger.warning("OrchestratorBase not available for integration")

logger = logging.getLogger(__name__)


class IntelligentWorkflowEngine:
    """
    Main workflow engine that coordinates design, execution, and optimization
    Enhanced Hours 30-40: Optional OrchestratorBase integration for unified processing
    """
    
    def __init__(self, max_concurrent_workflows: int = 10, enable_orchestration_base: bool = True):
        # Core workflow components
        self.designer = WorkflowDesigner()
        self.scheduler = WorkflowScheduler(max_concurrent_workflows)
        self.optimizer = WorkflowOptimizer()
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.engine_metrics: Dict[str, Any] = {}
        
        # Enhanced Hours 30-40: Orchestration base integration
        self.orchestration_base = None
        self.orchestration_enabled = False
        
        if enable_orchestration_base and ORCHESTRATION_BASE_AVAILABLE:
            self._initialize_orchestration_base()
        
        logger.info("IntelligentWorkflowEngine initialized with comprehensive automation capabilities")
    
    async def start_engine(self):
        """Start the workflow engine"""
        logger.info("Starting Intelligent Workflow Engine")
        await self.scheduler.start_scheduler()
        self._update_engine_metrics()
    
    def register_intelligence_system(self, system_id: str, capabilities: List[str],
                                   performance_metrics: Dict[str, float]):
        """Register an intelligence system with the workflow engine"""
        logger.info(f"Registering intelligence system: {system_id}")
        
        # Register with designer
        self.designer.register_system_capabilities(system_id, capabilities, performance_metrics)
        
        # Register with scheduler
        self.scheduler.register_system(system_id, performance_metrics.get("load", 0.0))
        
        logger.info(f"Successfully registered {system_id} with {len(capabilities)} capabilities")
    
    async def create_and_execute_workflow(self, requirements: Dict[str, Any],
                                        constraints: Dict[str, Any] = None) -> Tuple[str, str]:
        """Create and execute a workflow based on requirements"""
        logger.info(f"Creating and executing workflow for: {requirements.get('objective', 'unknown')}")
        
        # Design workflow
        workflow = await self.designer.design_workflow(requirements, constraints)
        self.workflows[workflow.workflow_id] = workflow
        
        # Schedule for execution
        execution_id = await self.scheduler.schedule_workflow(workflow)
        
        logger.info(f"Created workflow {workflow.workflow_id} and scheduled execution {execution_id}")
        return workflow.workflow_id, execution_id
    
    async def optimize_workflow(self, workflow_id: str, 
                              objective: OptimizationObjective = None) -> WorkflowOptimization:
        """Optimize an existing workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        optimization = await self.optimizer.optimize_workflow(workflow, objective)
        
        logger.info(f"Optimized workflow {workflow_id}. Improvement: {optimization.improvement_percentage:.1%}")
        return optimization
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow"""
        if workflow_id not in self.workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow = self.workflows[workflow_id]
        
        # Find executions of this workflow
        executions = [exec for exec in self.executions.values() if exec.workflow_id == workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.priority.value,
            "tasks_count": len(workflow.tasks),
            "executions": len(executions),
            "last_execution": executions[-1].status.value if executions else "never_executed",
            "optimization_objective": workflow.optimization_objective.value,
            "created_at": workflow.created_at.isoformat()
        }
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of a workflow execution"""
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
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "execution_metrics": execution.execution_metrics
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the workflow engine"""
        scheduler_status = self.scheduler.get_scheduler_status()
        
        return {
            "engine_metrics": self.engine_metrics,
            "scheduler_status": scheduler_status,
            "total_workflows": len(self.workflows),
            "total_executions": len(self.executions),
            "optimization_history": len(self.optimizer.optimization_history),
            "registered_systems": len(self.designer.system_capabilities),
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_engine_metrics(self):
        """Update engine performance metrics"""
        self.engine_metrics = {
            "workflows_created": len(self.workflows),
            "workflows_executed": len(self.executions),
            "total_optimizations": len(self.optimizer.optimization_history),
            "average_workflow_tasks": statistics.mean([len(w.tasks) for w in self.workflows.values()]) if self.workflows else 0,
            "success_rate": self._calculate_success_rate(),
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
    
    def stop_engine(self):
        """Stop the workflow engine"""
        logger.info("Stopping Intelligent Workflow Engine")
        self.scheduler.stop_scheduler()
    
    # ========================================================================
    # ENHANCED HOURS 30-40: ORCHESTRATION BASE INTEGRATION
    # ========================================================================
    
    def _initialize_orchestration_base(self):
        """Initialize orchestration base integration for unified processing"""
        if not ORCHESTRATION_BASE_AVAILABLE:
            return
        
        try:
            # Create embedded orchestrator for workflow management
            class WorkflowOrchestrator(OrchestratorBase):
                """Workflow-specialized orchestrator"""
                
                def __init__(self, workflow_engine):
                    super().__init__(
                        orchestrator_type=OrchestratorType.WORKFLOW,
                        name="WorkflowOrchestrator"
                    )
                    self.workflow_engine = workflow_engine
                    
                    # Set enhanced workflow capabilities
                    self.capabilities.supports_workflow_design = True
                    self.capabilities.supports_workflow_optimization = True
                    self.capabilities.supports_adaptive_execution = True
                    self.capabilities.supports_intelligent_routing = True
                    self.capabilities.workflow_patterns.update({
                        'automated_design', 'adaptive_scheduling', 'performance_optimization'
                    })
                
                async def execute_task(self, task: Any) -> Any:
                    """Execute workflow task through engine"""
                    if isinstance(task, dict) and task.get('type') == 'workflow_execution':
                        workflow_id = task.get('workflow_id')
                        if workflow_id in self.workflow_engine.workflows:
                            execution = await self.workflow_engine.execute_workflow(workflow_id)
                            return {"status": "completed", "execution_id": execution.execution_id}
                    
                    return {"status": "completed", "result": task}
                
                async def design_workflow(self, requirements: Dict[str, Any]) -> Optional[Any]:
                    """Design workflow using engine designer"""
                    return await self.workflow_engine.design_workflow(requirements)
                
                async def optimize_workflow(self, workflow: Any, performance_data: Dict[str, Any]) -> Optional[Any]:
                    """Optimize workflow using engine optimizer"""
                    if isinstance(workflow, str) and workflow in self.workflow_engine.workflows:
                        optimization = await self.workflow_engine.optimize_workflow(workflow, performance_data)
                        return optimization
                    return None
                
                def get_supported_capabilities(self) -> OrchestratorCapabilities:
                    """Get workflow orchestrator capabilities"""
                    return self.capabilities
            
            # Initialize the embedded orchestrator
            self.orchestration_base = WorkflowOrchestrator(self)
            self.orchestration_enabled = True
            
            logger.info("Workflow engine enhanced with orchestration base integration")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestration base integration: {e}")
            self.orchestration_enabled = False
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get orchestration integration status"""
        if not self.orchestration_enabled or not self.orchestration_base:
            return {"orchestration_enabled": False}
        
        return {
            "orchestration_enabled": True,
            "orchestrator_name": self.orchestration_base.name,
            "orchestrator_type": self.orchestration_base.orchestrator_type.value,
            "capabilities": {
                "workflow_design": self.orchestration_base.capabilities.supports_workflow_design,
                "workflow_optimization": self.orchestration_base.capabilities.supports_workflow_optimization,
                "adaptive_execution": self.orchestration_base.capabilities.supports_adaptive_execution,
                "intelligent_routing": self.orchestration_base.capabilities.supports_intelligent_routing
            },
            "workflow_patterns": list(self.orchestration_base.capabilities.workflow_patterns),
            "orchestrator_status": self.orchestration_base.get_status_summary() if hasattr(self.orchestration_base, 'get_status_summary') else {}
        }
    
    async def execute_with_orchestration(self, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow through orchestration base if available"""
        if not self.orchestration_enabled:
            # Fall back to direct execution
            execution = await self.execute_workflow(workflow_id)
            return {"orchestrated": False, "execution_id": execution.execution_id}
        
        # Execute through orchestration base
        task = {
            "type": "workflow_execution",
            "workflow_id": workflow_id
        }
        
        result = await self.orchestration_base.execute_task(task)
        result["orchestrated"] = True
        return result