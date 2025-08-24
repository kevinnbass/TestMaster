"""
Testing Team
============

Integrated team management for multi-agent testing workflows.
Combines roles, supervisor, and workflow coordination.

Author: TestMaster Team
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

from ..roles import (
    TestArchitect, TestEngineer, QualityAssuranceAgent,
    TestExecutor, TestCoordinator
)
from ..supervisor import TestingSupervisor, SupervisorMode
from core.observability import global_observability

class TeamRole(Enum):
    """Predefined team roles"""
    ARCHITECT = "architect"
    ENGINEER = "engineer" 
    QA_AGENT = "qa_agent"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"

@dataclass
class TeamConfiguration:
    """Configuration for testing team"""
    roles: List[TeamRole] = field(default_factory=list)
    supervisor_mode: SupervisorMode = SupervisorMode.GUIDED
    workflow_type: str = "standard"
    max_parallel_tasks: int = 3
    quality_threshold: float = 80.0
    timeout_minutes: int = 30

@dataclass 
class TeamWorkflow:
    """Defines a team workflow"""
    name: str
    phases: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)

class TestingTeam:
    """
    Integrated testing team that combines multiple agent roles
    with hierarchical supervision for complete testing workflows.
    """
    
    def __init__(self, configuration: TeamConfiguration):
        self.config = configuration
        self.team_id = f"team_{uuid.uuid4().hex[:12]}"
        
        # Team components
        self.roles: Dict[str, Any] = {}
        self.supervisor: Optional[TestingSupervisor] = None
        self.current_workflow: Optional[TeamWorkflow] = None
        
        # State management
        self.active = False
        self.session_id = None
        
        # Performance tracking
        self.team_metrics = {
            "workflows_completed": 0,
            "total_execution_time": 0.0,
            "success_rate": 0.0,
            "quality_score": 0.0,
            "collaboration_score": 0.0
        }
        
        # Logging
        self.logger = logging.getLogger(f'TestingTeam.{self.team_id}')
        
        # Initialize team
        self._initialize_team()
    
    def _initialize_team(self):
        """Initialize team roles and supervisor"""
        # Create role instances based on configuration
        for role_type in self.config.roles:
            if role_type == TeamRole.ARCHITECT:
                self.roles["architect"] = TestArchitect()
            elif role_type == TeamRole.ENGINEER:
                self.roles["engineer"] = TestEngineer()
            elif role_type == TeamRole.QA_AGENT:
                self.roles["qa_agent"] = QualityAssuranceAgent()
            elif role_type == TeamRole.EXECUTOR:
                self.roles["executor"] = TestExecutor()
            elif role_type == TeamRole.COORDINATOR:
                self.roles["coordinator"] = TestCoordinator()
        
        # Set up collaborations between roles
        self._setup_collaborations()
        
        # Create supervisor
        self.supervisor = TestingSupervisor(mode=self.config.supervisor_mode)
        
        self.logger.info(f"Initialized team with {len(self.roles)} roles and {self.config.supervisor_mode.value} supervision")
    
    def _setup_collaborations(self):
        """Set up collaboration relationships between roles"""
        role_instances = list(self.roles.values())
        
        # Connect all roles to each other for collaboration
        for role in role_instances:
            for other_role in role_instances:
                if role != other_role:
                    role.add_collaborator(other_role)
    
    async def start_team(self, session_id: Optional[str] = None):
        """Start the testing team"""
        self.active = True
        self.session_id = session_id
        
        # Start all roles
        for role_name, role in self.roles.items():
            await role.start_role(session_id)
            self.logger.info(f"Started role: {role_name}")
        
        # Start supervisor
        if self.supervisor:
            await self.supervisor.start_supervision(
                self.roles,
                session_id=session_id
            )
        
        if session_id:
            global_observability.track_agent_action(
                session_id,
                "TestingTeam",
                "team_started",
                {
                    "team_id": self.team_id,
                    "roles": list(self.roles.keys()),
                    "supervisor_mode": self.config.supervisor_mode.value
                }
            )
        
        self.logger.info(f"Testing team {self.team_id} started successfully")
    
    async def stop_team(self):
        """Stop the testing team"""
        self.active = False
        
        # Stop supervisor first
        if self.supervisor:
            final_metrics = await self.supervisor.stop_supervision()
            self.team_metrics.update(final_metrics.get("supervision_metrics", {}))
        
        # Stop all roles
        for role_name, role in self.roles.items():
            await role.stop_role()
            self.logger.info(f"Stopped role: {role_name}")
        
        if self.session_id:
            global_observability.track_agent_action(
                self.session_id,
                "TestingTeam",
                "team_stopped",
                {
                    "team_id": self.team_id,
                    "final_metrics": self.team_metrics
                }
            )
        
        self.logger.info(f"Testing team {self.team_id} stopped")
    
    async def execute_workflow(self, workflow: TeamWorkflow, target_path: str) -> Dict[str, Any]:
        """Execute a complete testing workflow"""
        if not self.active:
            raise RuntimeError("Team must be started before executing workflows")
        
        self.current_workflow = workflow
        workflow_start = datetime.now()
        
        workflow_result = {
            "workflow_name": workflow.name,
            "team_id": self.team_id,
            "target_path": target_path,
            "status": "unknown",
            "phases_completed": [],
            "phases_failed": [],
            "total_time": 0.0,
            "results": {},
            "metrics": {}
        }
        
        try:
            self.logger.info(f"Starting workflow: {workflow.name}")
            
            # Execute workflow phases
            for phase in workflow.phases:
                phase_result = await self._execute_phase(phase, target_path)
                
                if phase_result["status"] == "success":
                    workflow_result["phases_completed"].append(phase["name"])
                else:
                    workflow_result["phases_failed"].append(phase["name"])
                
                workflow_result["results"][phase["name"]] = phase_result
            
            # Determine overall status
            if not workflow_result["phases_failed"]:
                workflow_result["status"] = "success"
            elif workflow_result["phases_completed"]:
                workflow_result["status"] = "partial_success"
            else:
                workflow_result["status"] = "failed"
            
            # Calculate metrics
            workflow_result["total_time"] = (datetime.now() - workflow_start).total_seconds()
            workflow_result["metrics"] = await self._calculate_workflow_metrics(workflow_result)
            
            # Update team metrics
            self._update_team_metrics(workflow_result)
            
        except Exception as e:
            workflow_result["status"] = "error"
            workflow_result["error"] = str(e)
            self.logger.error(f"Workflow execution failed: {e}")
        
        return workflow_result
    
    async def _execute_phase(self, phase: Dict[str, Any], target_path: str) -> Dict[str, Any]:
        """Execute a single workflow phase"""
        phase_name = phase["name"]
        role_name = phase.get("role", "architect")
        action_type = phase.get("action", "analyze")
        
        self.logger.info(f"Executing phase: {phase_name} with role: {role_name}")
        
        phase_result = {
            "phase": phase_name,
            "role": role_name,
            "action": action_type,
            "status": "unknown",
            "result": None,
            "execution_time": 0.0
        }
        
        try:
            if role_name not in self.roles:
                raise ValueError(f"Role {role_name} not available in team")
            
            role = self.roles[role_name]
            
            # Import action type
            from ..roles.base_role import TestActionType, TestAction
            
            # Create action
            action = TestAction(
                role=role_name,
                action_type=TestActionType(action_type),
                description=phase.get("description", f"Execute {action_type} for {phase_name}"),
                parameters={
                    "target_path": target_path,
                    "phase_config": phase,
                    **phase.get("parameters", {})
                }
            )
            
            # Execute action
            completed_action = await role.perform_action(action)
            
            phase_result["status"] = completed_action.status
            phase_result["result"] = completed_action.result
            phase_result["execution_time"] = completed_action.duration
            
            if completed_action.error:
                phase_result["error"] = completed_action.error
            
        except Exception as e:
            phase_result["status"] = "failed"
            phase_result["error"] = str(e)
            self.logger.error(f"Phase {phase_name} failed: {e}")
        
        return phase_result
    
    async def _calculate_workflow_metrics(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for completed workflow"""
        total_phases = len(workflow_result["phases_completed"]) + len(workflow_result["phases_failed"])
        success_rate = len(workflow_result["phases_completed"]) / total_phases * 100 if total_phases > 0 else 0
        
        # Get role performance metrics
        role_metrics = {}
        for role_name, role in self.roles.items():
            role_metrics[role_name] = role.get_status()["performance_metrics"]
        
        return {
            "phase_success_rate": success_rate,
            "total_phases": total_phases,
            "execution_efficiency": success_rate / workflow_result["total_time"] if workflow_result["total_time"] > 0 else 0,
            "role_performance": role_metrics
        }
    
    def _update_team_metrics(self, workflow_result: Dict[str, Any]):
        """Update team-level performance metrics"""
        self.team_metrics["workflows_completed"] += 1
        self.team_metrics["total_execution_time"] += workflow_result["total_time"]
        
        # Update success rate
        if workflow_result["status"] == "success":
            current_successes = self.team_metrics["workflows_completed"] * self.team_metrics["success_rate"] / 100
            new_successes = current_successes + 1
            self.team_metrics["success_rate"] = new_successes / self.team_metrics["workflows_completed"] * 100
        
        # Update quality score based on phase success rate
        phase_success = workflow_result["metrics"]["phase_success_rate"]
        current_quality = self.team_metrics["quality_score"]
        workflows = self.team_metrics["workflows_completed"]
        
        self.team_metrics["quality_score"] = (
            (current_quality * (workflows - 1) + phase_success) / workflows
        )
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get comprehensive team status"""
        role_statuses = {
            role_name: role.get_status()
            for role_name, role in self.roles.items()
        }
        
        supervisor_status = None
        if self.supervisor:
            supervisor_status = self.supervisor.get_supervision_status()
        
        return {
            "team_id": self.team_id,
            "active": self.active,
            "configuration": {
                "roles": [role.value for role in self.config.roles],
                "supervisor_mode": self.config.supervisor_mode.value,
                "workflow_type": self.config.workflow_type
            },
            "current_workflow": self.current_workflow.name if self.current_workflow else None,
            "team_metrics": self.team_metrics,
            "role_statuses": role_statuses,
            "supervisor_status": supervisor_status
        }
    
    @classmethod
    def create_standard_team(cls) -> 'TestingTeam':
        """Create a standard testing team with all roles"""
        config = TeamConfiguration(
            roles=[
                TeamRole.ARCHITECT,
                TeamRole.ENGINEER,
                TeamRole.QA_AGENT,
                TeamRole.EXECUTOR,
                TeamRole.COORDINATOR
            ],
            supervisor_mode=SupervisorMode.GUIDED,
            workflow_type="comprehensive"
        )
        return cls(config)
    
    @classmethod
    def create_minimal_team(cls) -> 'TestingTeam':
        """Create a minimal testing team for simple projects"""
        config = TeamConfiguration(
            roles=[
                TeamRole.ARCHITECT,
                TeamRole.ENGINEER,
                TeamRole.EXECUTOR
            ],
            supervisor_mode=SupervisorMode.AUTONOMOUS,
            workflow_type="minimal"
        )
        return cls(config)

# Predefined workflows
STANDARD_TESTING_WORKFLOW = TeamWorkflow(
    name="Standard Testing Workflow",
    phases=[
        {
            "name": "requirements_analysis",
            "role": "architect",
            "action": "analyze",
            "description": "Analyze testing requirements and create strategy"
        },
        {
            "name": "test_design",
            "role": "architect", 
            "action": "design",
            "description": "Design comprehensive test architecture"
        },
        {
            "name": "test_implementation",
            "role": "engineer",
            "action": "implement",
            "description": "Implement tests based on design"
        },
        {
            "name": "test_execution",
            "role": "executor",
            "action": "execute",
            "description": "Execute tests and collect results"
        },
        {
            "name": "quality_review",
            "role": "qa_agent",
            "action": "review",
            "description": "Review test quality and coverage"
        }
    ],
    success_criteria={
        "min_coverage": 80.0,
        "min_quality_score": 75.0,
        "max_execution_time": 1800  # 30 minutes
    }
)

# Export components
__all__ = [
    'TestingTeam',
    'TeamConfiguration', 
    'TeamRole',
    'TeamWorkflow',
    'STANDARD_TESTING_WORKFLOW'
]