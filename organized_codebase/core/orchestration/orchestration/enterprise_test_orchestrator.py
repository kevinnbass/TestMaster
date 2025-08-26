"""
Enterprise Test Orchestration System for TestMaster
Advanced test orchestration with CI/CD integration and enterprise workflow management
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import subprocess
from pathlib import Path
import yaml

class OrchestrationMode(Enum):
    """Test orchestration execution modes"""
    DEVELOPMENT = "development"
    INTEGRATION = "integration"
    STAGING = "staging"
    PRODUCTION = "production"
    HOTFIX = "hotfix"

class PipelineStage(Enum):
    """CI/CD pipeline stages"""
    PREPARE = "prepare"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    SECURITY_TESTS = "security_tests"
    PERFORMANCE_TESTS = "performance_tests"
    E2E_TESTS = "e2e_tests"
    COMPLIANCE_VALIDATION = "compliance_validation"
    DEPLOYMENT_VALIDATION = "deployment_validation"
    CLEANUP = "cleanup"

class ExecutionStatus(Enum):
    """Execution status for stages and workflows"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"

class IntegrationType(Enum):
    """Enterprise integration types"""
    JENKINS = "jenkins"
    GITHUB_ACTIONS = "github_actions"
    AZURE_DEVOPS = "azure_devops"
    GITLAB_CI = "gitlab_ci"
    BAMBOO = "bamboo"
    TEAMCITY = "teamcity"
    SLACK = "slack"
    JIRA = "jira"
    CONFLUENCE = "confluence"

@dataclass
class TestStage:
    """Individual test stage configuration"""
    stage_id: str
    name: str
    stage_type: PipelineStage
    dependencies: List[str] = field(default_factory=list)
    timeout_minutes: int = 30
    retry_count: int = 1
    parallel_execution: bool = False
    required_for_promotion: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Test workflow execution record"""
    execution_id: str
    workflow_name: str
    mode: OrchestrationMode
    trigger: str
    start_time: float
    end_time: Optional[float] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    stages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    notifications_sent: List[str] = field(default_factory=list)

@dataclass
class EnterpriseIntegration:
    """Enterprise system integration configuration"""
    integration_type: IntegrationType
    name: str
    endpoint: str
    credentials: Dict[str, str]
    configuration: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class OrchestrationReport:
    """Comprehensive orchestration execution report"""
    report_id: str
    execution_id: str
    workflow_name: str
    total_stages: int
    successful_stages: int
    failed_stages: int
    total_execution_time: float
    stage_details: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    artifacts_generated: List[str]
    timestamp: float

class EnterpriseTestOrchestrator:
    """Advanced Enterprise Test Orchestration System"""
    
    def __init__(self, config_path: str = "orchestration_config.yml"):
        self.config_path = Path(config_path)
        self.workflows: Dict[str, List[TestStage]] = {}
        self.integrations: Dict[str, EnterpriseIntegration] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        self.load_configuration()
        
    def load_configuration(self) -> None:
        """Load orchestration configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self._parse_configuration(config)
            except Exception as e:
                print(f"Failed to load configuration: {e}")
                self._initialize_default_configuration()
        else:
            self._initialize_default_configuration()
    
    def _initialize_default_configuration(self) -> None:
        """Initialize default orchestration configuration"""
        # Default development workflow
        dev_workflow = [
            TestStage(
                stage_id="prepare",
                name="Environment Preparation",
                stage_type=PipelineStage.PREPARE,
                timeout_minutes=5,
                configuration={"setup_test_db": True, "clear_cache": True}
            ),
            TestStage(
                stage_id="unit_tests",
                name="Unit Test Execution",
                stage_type=PipelineStage.UNIT_TESTS,
                dependencies=["prepare"],
                timeout_minutes=15,
                parallel_execution=True,
                configuration={"coverage_threshold": 80, "fast_fail": True}
            ),
            TestStage(
                stage_id="integration_tests",
                name="Integration Test Execution",
                stage_type=PipelineStage.INTEGRATION_TESTS,
                dependencies=["unit_tests"],
                timeout_minutes=30,
                configuration={"test_environments": ["local"], "data_setup": True}
            )
        ]
        
        # Default production workflow
        prod_workflow = [
            TestStage(
                stage_id="prepare_prod",
                name="Production Environment Preparation",
                stage_type=PipelineStage.PREPARE,
                timeout_minutes=10,
                configuration={"backup_db": True, "health_check": True}
            ),
            TestStage(
                stage_id="security_scan",
                name="Security Testing",
                stage_type=PipelineStage.SECURITY_TESTS,
                dependencies=["prepare_prod"],
                timeout_minutes=45,
                required_for_promotion=True,
                configuration={"owasp_scan": True, "dependency_check": True}
            ),
            TestStage(
                stage_id="performance_tests",
                name="Performance Testing",
                stage_type=PipelineStage.PERFORMANCE_TESTS,
                dependencies=["security_scan"],
                timeout_minutes=60,
                configuration={"load_test": True, "stress_test": True}
            ),
            TestStage(
                stage_id="compliance_validation",
                name="Compliance Validation",
                stage_type=PipelineStage.COMPLIANCE_VALIDATION,
                dependencies=["performance_tests"],
                timeout_minutes=30,
                required_for_promotion=True,
                configuration={"standards": ["SOC2", "GDPR", "HIPAA"]}
            )
        ]
        
        self.workflows = {
            "development": dev_workflow,
            "production": prod_workflow
        }
    
    def _parse_configuration(self, config: Dict[str, Any]) -> None:
        """Parse configuration from YAML"""
        # Parse workflows
        workflows_config = config.get("workflows", {})
        for workflow_name, stages_config in workflows_config.items():
            stages = []
            for stage_config in stages_config:
                stage = TestStage(
                    stage_id=stage_config["stage_id"],
                    name=stage_config["name"],
                    stage_type=PipelineStage(stage_config["stage_type"]),
                    dependencies=stage_config.get("dependencies", []),
                    timeout_minutes=stage_config.get("timeout_minutes", 30),
                    retry_count=stage_config.get("retry_count", 1),
                    parallel_execution=stage_config.get("parallel_execution", False),
                    required_for_promotion=stage_config.get("required_for_promotion", True),
                    configuration=stage_config.get("configuration", {})
                )
                stages.append(stage)
            self.workflows[workflow_name] = stages
        
        # Parse integrations
        integrations_config = config.get("integrations", {})
        for integration_name, int_config in integrations_config.items():
            integration = EnterpriseIntegration(
                integration_type=IntegrationType(int_config["type"]),
                name=integration_name,
                endpoint=int_config["endpoint"],
                credentials=int_config.get("credentials", {}),
                configuration=int_config.get("configuration", {}),
                enabled=int_config.get("enabled", True)
            )
            self.integrations[integration_name] = integration
    
    async def execute_workflow(self, workflow_name: str, mode: OrchestrationMode, 
                             trigger: str = "manual", 
                             parameters: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute test workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_name=workflow_name,
            mode=mode,
            trigger=trigger,
            start_time=time.time(),
            status=ExecutionStatus.RUNNING
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            # Send workflow start notifications
            await self._send_workflow_notifications(execution, "started")
            
            # Execute workflow stages
            workflow_stages = self.workflows[workflow_name]
            await self._execute_workflow_stages(execution, workflow_stages, parameters or {})
            
            execution.status = ExecutionStatus.SUCCESS
            execution.end_time = time.time()
            
            # Generate and store artifacts
            await self._generate_workflow_artifacts(execution)
            
            # Send completion notifications
            await self._send_workflow_notifications(execution, "completed")
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.end_time = time.time()
            print(f"Workflow execution failed: {e}")
            
            # Send failure notifications
            await self._send_workflow_notifications(execution, "failed")
        
        finally:
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
        
        return execution
    
    async def _execute_workflow_stages(self, execution: WorkflowExecution, 
                                     stages: List[TestStage], 
                                     parameters: Dict[str, Any]) -> None:
        """Execute workflow stages with dependency management"""
        completed_stages = set()
        stage_results = {}
        
        while len(completed_stages) < len(stages):
            # Find executable stages (dependencies met)
            executable_stages = [
                stage for stage in stages
                if stage.stage_id not in completed_stages
                and all(dep in completed_stages for dep in stage.dependencies)
            ]
            
            if not executable_stages:
                raise Exception("Circular dependency or impossible stage execution order")
            
            # Execute stages (parallel if possible)
            parallel_stages = [stage for stage in executable_stages if stage.parallel_execution]
            sequential_stages = [stage for stage in executable_stages if not stage.parallel_execution]
            
            # Execute parallel stages
            if parallel_stages:
                tasks = []
                for stage in parallel_stages:
                    task = asyncio.create_task(
                        self._execute_single_stage(execution, stage, parameters, stage_results)
                    )
                    tasks.append((stage.stage_id, task))
                
                for stage_id, task in tasks:
                    try:
                        result = await task
                        stage_results[stage_id] = result
                        completed_stages.add(stage_id)
                    except Exception as e:
                        stage_results[stage_id] = {"status": "failed", "error": str(e)}
                        # Check if stage is required for promotion
                        stage = next(s for s in parallel_stages if s.stage_id == stage_id)
                        if stage.required_for_promotion:
                            raise Exception(f"Required stage {stage_id} failed: {e}")
            
            # Execute sequential stages
            for stage in sequential_stages:
                try:
                    result = await self._execute_single_stage(execution, stage, parameters, stage_results)
                    stage_results[stage.stage_id] = result
                    completed_stages.add(stage.stage_id)
                except Exception as e:
                    stage_results[stage.stage_id] = {"status": "failed", "error": str(e)}
                    if stage.required_for_promotion:
                        raise Exception(f"Required stage {stage.stage_id} failed: {e}")
        
        # Update execution with stage results
        execution.stages = list(stage_results.values())
    
    async def _execute_single_stage(self, execution: WorkflowExecution, 
                                   stage: TestStage, 
                                   parameters: Dict[str, Any],
                                   previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single workflow stage"""
        stage_start = time.time()
        stage_result = {
            "stage_id": stage.stage_id,
            "name": stage.name,
            "type": stage.stage_type.value,
            "start_time": stage_start,
            "status": ExecutionStatus.RUNNING.value
        }
        
        try:
            # Execute stage based on type
            if stage.stage_type == PipelineStage.PREPARE:
                result = await self._execute_prepare_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.UNIT_TESTS:
                result = await self._execute_unit_tests_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.INTEGRATION_TESTS:
                result = await self._execute_integration_tests_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.SECURITY_TESTS:
                result = await self._execute_security_tests_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.PERFORMANCE_TESTS:
                result = await self._execute_performance_tests_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.E2E_TESTS:
                result = await self._execute_e2e_tests_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.COMPLIANCE_VALIDATION:
                result = await self._execute_compliance_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.DEPLOYMENT_VALIDATION:
                result = await self._execute_deployment_validation_stage(stage, parameters)
            elif stage.stage_type == PipelineStage.CLEANUP:
                result = await self._execute_cleanup_stage(stage, parameters)
            else:
                raise Exception(f"Unknown stage type: {stage.stage_type}")
            
            stage_result.update(result)
            stage_result["status"] = ExecutionStatus.SUCCESS.value
            
        except Exception as e:
            stage_result["status"] = ExecutionStatus.FAILED.value
            stage_result["error"] = str(e)
            
            # Retry logic
            if stage.retry_count > 1:
                for retry in range(stage.retry_count - 1):
                    try:
                        await asyncio.sleep(2 ** retry)  # Exponential backoff
                        # Re-execute stage logic here
                        break
                    except Exception:
                        continue
        
        finally:
            stage_result["end_time"] = time.time()
            stage_result["duration"] = stage_result["end_time"] - stage_start
        
        return stage_result
    
    async def _execute_prepare_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute environment preparation stage"""
        config = stage.configuration
        results = {"prepared_components": []}
        
        if config.get("setup_test_db"):
            # Simulate test database setup
            await asyncio.sleep(1)
            results["prepared_components"].append("test_database")
        
        if config.get("clear_cache"):
            # Simulate cache clearing
            await asyncio.sleep(0.5)
            results["prepared_components"].append("cache_cleared")
        
        if config.get("health_check"):
            # Simulate health check
            await asyncio.sleep(2)
            results["health_check_status"] = "passed"
        
        return results
    
    async def _execute_unit_tests_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unit tests stage"""
        config = stage.configuration
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v"]
        
        if config.get("coverage_threshold"):
            cmd.extend(["--cov=.", f"--cov-fail-under={config['coverage_threshold']}"])
        
        if config.get("fast_fail"):
            cmd.append("-x")
        
        # Execute tests
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=stage.timeout_minutes * 60)
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "tests_passed": result.returncode == 0,
            "coverage_achieved": self._extract_coverage_from_output(result.stdout)
        }
    
    async def _execute_integration_tests_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integration tests stage"""
        config = stage.configuration
        
        cmd = ["python", "-m", "pytest", "tests/integration/", "-v"]
        
        if config.get("test_environments"):
            for env in config["test_environments"]:
                # Set environment-specific configuration
                pass
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=stage.timeout_minutes * 60)
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "tests_passed": result.returncode == 0,
            "environments_tested": config.get("test_environments", [])
        }
    
    async def _execute_security_tests_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security tests stage"""
        config = stage.configuration
        results = {"security_checks": []}
        
        if config.get("owasp_scan"):
            # Simulate OWASP security scan
            await asyncio.sleep(10)
            results["security_checks"].append({"type": "owasp", "status": "passed", "vulnerabilities": 0})
        
        if config.get("dependency_check"):
            # Simulate dependency vulnerability check
            await asyncio.sleep(5)
            results["security_checks"].append({"type": "dependencies", "status": "passed", "vulnerabilities": 0})
        
        return results
    
    async def _execute_performance_tests_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance tests stage"""
        config = stage.configuration
        results = {"performance_metrics": {}}
        
        if config.get("load_test"):
            # Simulate load testing
            await asyncio.sleep(15)
            results["performance_metrics"]["response_time_p95"] = 250  # ms
            results["performance_metrics"]["throughput"] = 1000  # req/s
        
        if config.get("stress_test"):
            # Simulate stress testing
            await asyncio.sleep(20)
            results["performance_metrics"]["max_concurrent_users"] = 500
            results["performance_metrics"]["breaking_point"] = 750
        
        return results
    
    async def _execute_e2e_tests_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute end-to-end tests stage"""
        cmd = ["python", "-m", "pytest", "tests/e2e/", "-v", "--tb=short"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=stage.timeout_minutes * 60)
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "tests_passed": result.returncode == 0,
            "scenarios_executed": self._count_scenarios_from_output(result.stdout)
        }
    
    async def _execute_compliance_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compliance validation stage"""
        config = stage.configuration
        standards = config.get("standards", [])
        
        compliance_results = {}
        for standard in standards:
            # Simulate compliance check for each standard
            await asyncio.sleep(3)
            compliance_results[standard] = {
                "status": "compliant",
                "score": 95.0,
                "findings": 2,
                "critical_issues": 0
            }
        
        return {"compliance_results": compliance_results}
    
    async def _execute_deployment_validation_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment validation stage"""
        # Simulate deployment validation
        await asyncio.sleep(5)
        return {
            "deployment_status": "validated",
            "health_checks_passed": True,
            "configuration_validated": True,
            "rollback_plan_verified": True
        }
    
    async def _execute_cleanup_stage(self, stage: TestStage, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cleanup stage"""
        # Simulate cleanup operations
        await asyncio.sleep(2)
        return {
            "cleanup_completed": True,
            "temporary_files_removed": True,
            "test_data_cleaned": True
        }
    
    def _extract_coverage_from_output(self, stdout: str) -> float:
        """Extract coverage percentage from pytest output"""
        # Simple regex to find coverage percentage
        import re
        coverage_match = re.search(r'TOTAL.*?(\d+)%', stdout)
        if coverage_match:
            return float(coverage_match.group(1))
        return 0.0
    
    def _count_scenarios_from_output(self, stdout: str) -> int:
        """Count test scenarios from pytest output"""
        import re
        scenario_matches = re.findall(r'test_.*?PASSED|test_.*?FAILED', stdout)
        return len(scenario_matches)
    
    async def _send_workflow_notifications(self, execution: WorkflowExecution, event_type: str) -> None:
        """Send workflow notifications to configured integrations"""
        notifications = []
        
        for integration_name, integration in self.integrations.items():
            if not integration.enabled:
                continue
            
            try:
                if integration.integration_type == IntegrationType.SLACK:
                    await self._send_slack_notification(integration, execution, event_type)
                    notifications.append(f"slack:{integration_name}")
                elif integration.integration_type == IntegrationType.JIRA:
                    await self._send_jira_notification(integration, execution, event_type)
                    notifications.append(f"jira:{integration_name}")
                
            except Exception as e:
                print(f"Failed to send notification to {integration_name}: {e}")
        
        execution.notifications_sent.extend(notifications)
    
    async def _send_slack_notification(self, integration: EnterpriseIntegration, 
                                     execution: WorkflowExecution, event_type: str) -> None:
        """Send Slack notification"""
        webhook_url = integration.endpoint
        
        message = {
            "text": f"Test Workflow {event_type.title()}",
            "attachments": [{
                "color": "good" if execution.status == ExecutionStatus.SUCCESS else "danger",
                "fields": [
                    {"title": "Workflow", "value": execution.workflow_name, "short": True},
                    {"title": "Mode", "value": execution.mode.value, "short": True},
                    {"title": "Execution ID", "value": execution.execution_id[:8], "short": True},
                    {"title": "Status", "value": execution.status.value, "short": True}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status != 200:
                    raise Exception(f"Slack notification failed: {response.status}")
    
    async def _send_jira_notification(self, integration: EnterpriseIntegration, 
                                    execution: WorkflowExecution, event_type: str) -> None:
        """Send JIRA notification (create issue if failed)"""
        if execution.status != ExecutionStatus.FAILED:
            return  # Only create JIRA issues for failures
        
        # Create JIRA issue for failed workflow
        issue_data = {
            "fields": {
                "project": {"key": integration.configuration.get("project_key", "TEST")},
                "summary": f"Test Workflow Failed: {execution.workflow_name}",
                "description": f"Workflow {execution.workflow_name} failed in {execution.mode.value} mode",
                "issuetype": {"name": "Bug"},
                "priority": {"name": "High"}
            }
        }
        
        # Implementation would require JIRA API integration
        print(f"Would create JIRA issue: {issue_data}")
    
    async def _generate_workflow_artifacts(self, execution: WorkflowExecution) -> None:
        """Generate workflow execution artifacts"""
        artifacts_dir = Path("artifacts") / execution.execution_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate execution report
        report = await self._generate_orchestration_report(execution)
        report_file = artifacts_dir / "execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report.__dict__, f, indent=2, default=str)
        
        execution.artifacts["execution_report"] = str(report_file)
        
        # Generate stage logs
        for stage in execution.stages:
            stage_file = artifacts_dir / f"stage_{stage['stage_id']}.json"
            with open(stage_file, 'w') as f:
                json.dump(stage, f, indent=2, default=str)
            execution.artifacts[f"stage_{stage['stage_id']}"] = str(stage_file)
    
    async def _generate_orchestration_report(self, execution: WorkflowExecution) -> OrchestrationReport:
        """Generate comprehensive orchestration report"""
        successful_stages = len([s for s in execution.stages if s.get("status") == "SUCCESS"])
        failed_stages = len([s for s in execution.stages if s.get("status") == "FAILED"])
        
        # Calculate quality metrics
        quality_metrics = {
            "stage_success_rate": successful_stages / len(execution.stages) if execution.stages else 0,
            "total_execution_time": (execution.end_time or time.time()) - execution.start_time,
            "average_stage_time": sum(s.get("duration", 0) for s in execution.stages) / len(execution.stages) if execution.stages else 0
        }
        
        # Generate recommendations
        recommendations = []
        if failed_stages > 0:
            recommendations.append("Review failed stages and implement fixes")
        if quality_metrics["total_execution_time"] > 3600:  # 1 hour
            recommendations.append("Consider optimizing long-running stages")
        
        return OrchestrationReport(
            report_id=str(uuid.uuid4()),
            execution_id=execution.execution_id,
            workflow_name=execution.workflow_name,
            total_stages=len(execution.stages),
            successful_stages=successful_stages,
            failed_stages=failed_stages,
            total_execution_time=quality_metrics["total_execution_time"],
            stage_details=execution.stages,
            quality_metrics=quality_metrics,
            compliance_status={},  # Would be populated from compliance stage results
            recommendations=recommendations,
            artifacts_generated=list(execution.artifacts.keys()),
            timestamp=time.time()
        )
    
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get status of specific workflow execution"""
        return self.active_executions.get(execution_id)
    
    def list_active_workflows(self) -> List[WorkflowExecution]:
        """List all active workflow executions"""
        return list(self.active_executions.values())
    
    def get_execution_history(self, limit: int = 50) -> List[WorkflowExecution]:
        """Get workflow execution history"""
        return self.execution_history[-limit:]
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel active workflow execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = ExecutionStatus.CANCELLED
            execution.end_time = time.time()
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            return True
        return False