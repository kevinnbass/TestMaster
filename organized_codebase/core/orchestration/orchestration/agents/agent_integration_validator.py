#!/usr/bin/env python3
"""
Agent Integration Validator
===========================

Comprehensive validation system for cross-agent integration testing.
Tests communication, coordination, and resource sharing between all agents.

Features:
- Agent A (Intelligence) integration testing
- Agent B (Testing/Monitoring) integration testing  
- Agent C (Security/Coordination) integration testing
- Agent D (Documentation/Validation) integration testing
- Agent E (Infrastructure) self-validation
- Cross-agent workflow validation
- Performance and reliability testing

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from cross_agent_integration_framework import (
    CrossAgentCoordinator, AgentType, AgentCapabilities,
    MessageType, AgentStatus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestResult:
    """Result of integration test."""
    test_name: str
    agents_involved: List[str]
    status: str  # PASS, FAIL, SKIP
    execution_time: float
    details: Dict[str, Any]
    error: Optional[str] = None


class AgentIntegrationValidator:
    """
    Comprehensive validator for cross-agent integration.
    
    Tests all aspects of multi-agent coordination and communication.
    """
    
    def __init__(self):
        self.coordinator = CrossAgentCoordinator()
        self.test_results: List[IntegrationTestResult] = []
        self.start_time = None
        self.end_time = None
    
    async def run_comprehensive_integration_tests(self) -> Dict[str, Any]:
        """Run complete cross-agent integration test suite."""
        self.start_time = datetime.now()
        logger.info("Starting comprehensive cross-agent integration tests")
        
        # Start coordinator
        await self.coordinator.start_coordinator()
        
        try:
            # Test suites
            test_suites = [
                ("Agent Registration", self.test_agent_registration),
                ("Agent Discovery", self.test_agent_discovery),
                ("Cross-Agent Communication", self.test_cross_agent_communication),
                ("Agent A Intelligence Integration", self.test_agent_a_integration),
                ("Agent B Testing Integration", self.test_agent_b_integration),
                ("Agent C Security Integration", self.test_agent_c_integration),
                ("Agent D Documentation Integration", self.test_agent_d_integration),
                ("Agent E Infrastructure Integration", self.test_agent_e_integration),
                ("Multi-Agent Workflows", self.test_multi_agent_workflows),
                ("Shared State Management", self.test_shared_state_management),
                ("Capability Request System", self.test_capability_requests),
                ("Integration Performance", self.test_integration_performance)
            ]
            
            for suite_name, test_func in test_suites:
                logger.info(f"Running {suite_name} tests...")
                try:
                    await test_func()
                except Exception as e:
                    logger.error(f"Error in {suite_name} tests: {e}")
                    self.test_results.append(IntegrationTestResult(
                        test_name=suite_name,
                        agents_involved=["coordinator"],
                        status="FAIL",
                        execution_time=0.0,
                        details={"error": str(e)},
                        error=str(e)
                    ))
            
        finally:
            await self.coordinator.stop_coordinator()
        
        self.end_time = datetime.now()
        return self.generate_integration_report()
    
    async def test_agent_registration(self):
        """Test agent registration and capability registration."""
        start_time = time.time()
        
        try:
            # Register all agents
            agents_registered = []
            
            agent_configs = [
                (AgentType.INTELLIGENCE, AgentCapabilities.get_intelligence_capabilities()),
                (AgentType.TESTING, AgentCapabilities.get_testing_capabilities()),
                (AgentType.SECURITY, AgentCapabilities.get_security_capabilities()),
                (AgentType.DOCUMENTATION, AgentCapabilities.get_documentation_capabilities()),
                (AgentType.INFRASTRUCTURE, AgentCapabilities.get_infrastructure_capabilities())
            ]
            
            for agent_type, capabilities in agent_configs:
                agent_id = await self.coordinator.register_agent(
                    agent_type=agent_type,
                    capabilities=capabilities,
                    metadata={"test_mode": True}
                )
                agents_registered.append(agent_type.value)
                
                # Verify registration
                status = self.coordinator.get_agent_status(agent_type)
                assert status == AgentStatus.ACTIVE, f"Agent {agent_type.value} not active"
            
            # Verify capabilities are registered
            available_capabilities = self.coordinator.get_available_capabilities()
            expected_capability_count = sum(len(caps) for _, caps in agent_configs)
            actual_capability_count = len(available_capabilities)
            
            assert actual_capability_count == expected_capability_count, \
                f"Expected {expected_capability_count} capabilities, got {actual_capability_count}"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent Registration",
                agents_involved=agents_registered,
                status="PASS",
                execution_time=execution_time,
                details={
                    "agents_registered": len(agents_registered),
                    "capabilities_registered": actual_capability_count,
                    "registration_successful": True
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent Registration",
                agents_involved=["coordinator"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_agent_discovery(self):
        """Test agent discovery and presence detection."""
        start_time = time.time()
        
        try:
            # Test discovery broadcasts
            broadcast_count = await self.coordinator.broadcast_to_agents(
                sender=AgentType.INFRASTRUCTURE,
                message_type=MessageType.DISCOVERY,
                payload={"action": "ping", "test": True}
            )
            
            # Verify all agents received discovery
            expected_agents = 4  # All except sender
            assert broadcast_count == expected_agents, \
                f"Expected {expected_agents} discovery messages, sent {broadcast_count}"
            
            # Test agent status queries
            active_agents = []
            for agent_type in AgentType:
                status = self.coordinator.get_agent_status(agent_type)
                if status == AgentStatus.ACTIVE:
                    active_agents.append(agent_type.value)
            
            assert len(active_agents) == 5, f"Expected 5 active agents, found {len(active_agents)}"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent Discovery",
                agents_involved=active_agents,
                status="PASS",
                execution_time=execution_time,
                details={
                    "discovery_broadcasts": broadcast_count,
                    "active_agents": len(active_agents),
                    "discovery_successful": True
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent Discovery",
                agents_involved=["coordinator"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_cross_agent_communication(self):
        """Test direct agent-to-agent communication."""
        start_time = time.time()
        
        try:
            # Test message sending between agents
            message_id = await self.coordinator.send_agent_message(
                sender=AgentType.INFRASTRUCTURE,
                recipient=AgentType.INTELLIGENCE,
                message_type=MessageType.REQUEST,
                payload={"test_request": "ping", "data": "test_data"},
                requires_response=True
            )
            
            assert message_id is not None, "Message sending failed"
            
            # Test broadcast communication
            broadcast_count = await self.coordinator.broadcast_to_agents(
                sender=AgentType.INFRASTRUCTURE,
                message_type=MessageType.NOTIFICATION,
                payload={"test_broadcast": "system_status", "status": "operational"}
            )
            
            assert broadcast_count > 0, "Broadcast failed"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Cross-Agent Communication",
                agents_involved=["agent_e", "agent_a", "all_agents"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "direct_message_sent": True,
                    "message_id": message_id,
                    "broadcast_recipients": broadcast_count,
                    "communication_successful": True
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Cross-Agent Communication",
                agents_involved=["coordinator"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_agent_a_integration(self):
        """Test Agent A (Intelligence) integration capabilities."""
        start_time = time.time()
        
        try:
            # Test intelligence capabilities
            capabilities_tested = []
            
            # Test technical debt analysis capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="technical_debt_analysis",
                request_data={"codebase_path": "test_project", "analysis_depth": "comprehensive"}
            )
            
            if result:
                capabilities_tested.append("technical_debt_analysis")
            
            # Test ML code analysis capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="ml_code_analysis",
                request_data={"code_files": ["ml_model.py"], "optimization_target": "performance"}
            )
            
            if result:
                capabilities_tested.append("ml_code_analysis")
            
            # Test predictive analytics capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="predictive_analytics",
                request_data={"metrics_data": {"performance": [1, 2, 3, 4, 5]}}
            )
            
            if result:
                capabilities_tested.append("predictive_analytics")
            
            assert len(capabilities_tested) >= 2, "Insufficient Agent A capabilities tested"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent A Intelligence Integration",
                agents_involved=["agent_a", "agent_e"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "capabilities_tested": capabilities_tested,
                    "intelligence_integration": "SUCCESSFUL",
                    "capability_count": len(capabilities_tested)
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent A Intelligence Integration",
                agents_involved=["agent_a"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_agent_b_integration(self):
        """Test Agent B (Testing/Monitoring) integration capabilities."""
        start_time = time.time()
        
        try:
            capabilities_tested = []
            
            # Test test generation capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="test_generation",
                request_data={"source_files": ["module.py"], "test_framework": "pytest"}
            )
            
            if result:
                capabilities_tested.append("test_generation")
            
            # Test coverage analysis capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="coverage_analysis",
                request_data={"test_suite": "tests/", "target_coverage": 90}
            )
            
            if result:
                capabilities_tested.append("coverage_analysis")
            
            # Test performance monitoring capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="performance_monitoring",
                request_data={"metrics_config": {"interval": 60, "alerts": True}}
            )
            
            if result:
                capabilities_tested.append("performance_monitoring")
            
            assert len(capabilities_tested) >= 2, "Insufficient Agent B capabilities tested"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent B Testing Integration",
                agents_involved=["agent_b", "agent_e"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "capabilities_tested": capabilities_tested,
                    "testing_integration": "SUCCESSFUL",
                    "capability_count": len(capabilities_tested)
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent B Testing Integration",
                agents_involved=["agent_b"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_agent_c_integration(self):
        """Test Agent C (Security/Coordination) integration capabilities."""
        start_time = time.time()
        
        try:
            capabilities_tested = []
            
            # Test security audit capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="security_audit",
                request_data={"audit_scope": "full", "compliance_standards": ["PCI", "SOX"]}
            )
            
            if result:
                capabilities_tested.append("security_audit")
            
            # Test coordination management capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="coordination_management",
                request_data={"workflow_type": "multi_agent", "coordination_pattern": "orchestrated"}
            )
            
            if result:
                capabilities_tested.append("coordination_management")
            
            # Test infrastructure validation capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="infrastructure_validation",
                request_data={"validation_scope": "security", "compliance_check": True}
            )
            
            if result:
                capabilities_tested.append("infrastructure_validation")
            
            assert len(capabilities_tested) >= 2, "Insufficient Agent C capabilities tested"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent C Security Integration",
                agents_involved=["agent_c", "agent_e"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "capabilities_tested": capabilities_tested,
                    "security_integration": "SUCCESSFUL",
                    "capability_count": len(capabilities_tested)
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent C Security Integration",
                agents_involved=["agent_c"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_agent_d_integration(self):
        """Test Agent D (Documentation/Validation) integration capabilities."""
        start_time = time.time()
        
        try:
            capabilities_tested = []
            
            # Test documentation generation capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="documentation_generation",
                request_data={"source_code": "modules/", "format": "markdown", "include_api": True}
            )
            
            if result:
                capabilities_tested.append("documentation_generation")
            
            # Test API validation capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="api_validation",
                request_data={"api_specs": "openapi.yaml", "validation_level": "strict"}
            )
            
            if result:
                capabilities_tested.append("api_validation")
            
            # Test knowledge management capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="knowledge_management",
                request_data={"knowledge_base": "docs/", "index_type": "full_text"}
            )
            
            if result:
                capabilities_tested.append("knowledge_management")
            
            assert len(capabilities_tested) >= 2, "Insufficient Agent D capabilities tested"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent D Documentation Integration",
                agents_involved=["agent_d", "agent_e"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "capabilities_tested": capabilities_tested,
                    "documentation_integration": "SUCCESSFUL",
                    "capability_count": len(capabilities_tested)
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent D Documentation Integration",
                agents_involved=["agent_d"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_agent_e_integration(self):
        """Test Agent E (Infrastructure) self-integration capabilities."""
        start_time = time.time()
        
        try:
            capabilities_tested = []
            
            # Test architecture optimization capability
            result = await self.coordinator.request_capability(
                requester=AgentType.INTELLIGENCE,  # Different requester
                capability_name="architecture_optimization",
                request_data={"architecture_type": "microservices", "optimization_goals": ["performance", "scalability"]}
            )
            
            if result:
                capabilities_tested.append("architecture_optimization")
            
            # Test modularization capability
            result = await self.coordinator.request_capability(
                requester=AgentType.TESTING,
                capability_name="modularization",
                request_data={"large_files": ["monolith.py"], "target_size": 300}
            )
            
            if result:
                capabilities_tested.append("modularization")
            
            # Test consolidation capability
            result = await self.coordinator.request_capability(
                requester=AgentType.SECURITY,
                capability_name="consolidation",
                request_data={"systems_to_consolidate": ["auth", "logging"], "consolidation_strategy": "merge"}
            )
            
            if result:
                capabilities_tested.append("consolidation")
            
            assert len(capabilities_tested) >= 2, "Insufficient Agent E capabilities tested"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent E Infrastructure Integration",
                agents_involved=["agent_e", "agent_a", "agent_b", "agent_c"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "capabilities_tested": capabilities_tested,
                    "infrastructure_integration": "SUCCESSFUL",
                    "capability_count": len(capabilities_tested)
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Agent E Infrastructure Integration",
                agents_involved=["agent_e"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_multi_agent_workflows(self):
        """Test complex multi-agent workflows."""
        start_time = time.time()
        
        try:
            workflow_steps = []
            
            # Step 1: Agent A analyzes technical debt
            result1 = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="technical_debt_analysis",
                request_data={"project": "testmaster"}
            )
            if result1:
                workflow_steps.append("debt_analysis")
            
            # Step 2: Agent B generates tests based on analysis
            result2 = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="test_generation",
                request_data={"based_on_debt_analysis": True}
            )
            if result2:
                workflow_steps.append("test_generation")
            
            # Step 3: Agent C validates security of generated tests
            result3 = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="security_audit",
                request_data={"audit_generated_tests": True}
            )
            if result3:
                workflow_steps.append("security_validation")
            
            # Step 4: Agent D documents the workflow
            result4 = await self.coordinator.request_capability(
                requester=AgentType.INFRASTRUCTURE,
                capability_name="documentation_generation",
                request_data={"document_workflow": True}
            )
            if result4:
                workflow_steps.append("documentation")
            
            assert len(workflow_steps) >= 3, "Multi-agent workflow incomplete"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Multi-Agent Workflows",
                agents_involved=["agent_a", "agent_b", "agent_c", "agent_d", "agent_e"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "workflow_steps_completed": workflow_steps,
                    "multi_agent_coordination": "SUCCESSFUL",
                    "workflow_length": len(workflow_steps)
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Multi-Agent Workflows",
                agents_involved=["all_agents"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_shared_state_management(self):
        """Test shared state management across agents."""
        start_time = time.time()
        
        try:
            # Test setting shared state
            test_data = {
                "project_status": "active",
                "current_phase": "integration_testing",
                "metrics": {"agents": 5, "capabilities": 15}
            }
            
            for key, value in test_data.items():
                self.coordinator.set_shared_state(key, value)
            
            # Test getting shared state
            retrieved_data = {}
            for key in test_data.keys():
                retrieved_data[key] = self.coordinator.get_shared_state(key)
            
            # Verify data integrity
            assert retrieved_data == test_data, "Shared state data integrity failed"
            
            # Test state persistence across operations
            self.coordinator.set_shared_state("agent_coordination", {"status": "operational"})
            coordination_state = self.coordinator.get_shared_state("agent_coordination")
            
            assert coordination_state is not None, "Shared state persistence failed"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Shared State Management",
                agents_involved=["coordinator", "all_agents"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "state_items_tested": len(test_data),
                    "data_integrity": "VERIFIED",
                    "persistence": "CONFIRMED"
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Shared State Management",
                agents_involved=["coordinator"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_capability_requests(self):
        """Test capability request system performance and reliability."""
        start_time = time.time()
        
        try:
            # Test multiple capability requests
            requests_made = 0
            successful_requests = 0
            
            test_requests = [
                ("technical_debt_analysis", {"quick_scan": True}),
                ("test_generation", {"module": "test_module"}),
                ("security_audit", {"scope": "limited"}),
                ("documentation_generation", {"format": "brief"}),
                ("architecture_optimization", {"focus": "performance"})
            ]
            
            for capability, data in test_requests:
                requests_made += 1
                result = await self.coordinator.request_capability(
                    requester=AgentType.INFRASTRUCTURE,
                    capability_name=capability,
                    request_data=data
                )
                if result:
                    successful_requests += 1
            
            success_rate = (successful_requests / requests_made) * 100
            assert success_rate >= 80, f"Capability request success rate too low: {success_rate}%"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Capability Request System",
                agents_involved=["all_agents"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "requests_made": requests_made,
                    "successful_requests": successful_requests,
                    "success_rate": success_rate,
                    "capability_system": "OPERATIONAL"
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Capability Request System",
                agents_involved=["coordinator"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    async def test_integration_performance(self):
        """Test integration performance and scalability."""
        start_time = time.time()
        
        try:
            metrics = self.coordinator.get_integration_metrics()
            
            # Performance assertions
            assert metrics["registered_agents"] == 5, "Not all agents registered"
            assert metrics["available_capabilities"] >= 15, "Insufficient capabilities available"
            assert metrics["message_metrics"]["messages_sent"] > 0, "No messages sent"
            
            # Test rapid capability requests
            rapid_test_start = time.time()
            rapid_requests = 10
            rapid_successful = 0
            
            for i in range(rapid_requests):
                result = await self.coordinator.request_capability(
                    requester=AgentType.INFRASTRUCTURE,
                    capability_name="test_generation",
                    request_data={"rapid_test": i}
                )
                if result:
                    rapid_successful += 1
            
            rapid_test_time = time.time() - rapid_test_start
            avg_request_time = rapid_test_time / rapid_requests
            
            assert avg_request_time < 0.1, f"Request time too slow: {avg_request_time}s"
            
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Integration Performance",
                agents_involved=["all_agents"],
                status="PASS",
                execution_time=execution_time,
                details={
                    "performance_metrics": metrics,
                    "rapid_requests": rapid_requests,
                    "rapid_successful": rapid_successful,
                    "avg_request_time": avg_request_time,
                    "performance": "ACCEPTABLE"
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Integration Performance",
                agents_involved=["coordinator"],
                status="FAIL",
                execution_time=execution_time,
                details={"error": str(e)},
                error=str(e)
            ))
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == "PASS")
        failed_tests = sum(1 for r in self.test_results if r.status == "FAIL")
        
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        report = {
            "integration_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_execution_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "agents_involved": r.agents_involved,
                    "status": r.status,
                    "execution_time": r.execution_time,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.test_results
            ],
            "integration_status": "VALIDATED" if failed_tests == 0 else "ISSUES_DETECTED",
            "cross_agent_readiness": "PRODUCTION_READY" if passed_tests >= total_tests * 0.9 else "NEEDS_IMPROVEMENT"
        }
        
        return report


async def main():
    """Run comprehensive cross-agent integration validation."""
    print("Cross-Agent Integration Validator")
    print("=" * 50)
    
    validator = AgentIntegrationValidator()
    report = await validator.run_comprehensive_integration_tests()
    
    print("\nCROSS-AGENT INTEGRATION REPORT")
    print("=" * 50)
    
    summary = report["integration_summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Execution Time: {summary['total_execution_time']:.2f}s")
    print(f"Integration Status: {report['integration_status']}")
    print(f"Cross-Agent Readiness: {report['cross_agent_readiness']}")
    
    print("\nTEST DETAILS")
    print("-" * 30)
    for result in report["test_results"]:
        status_icon = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
        agents = ", ".join(result["agents_involved"][:2])  # Show first 2 agents
        if len(result["agents_involved"]) > 2:
            agents += f" (+{len(result['agents_involved'])-2} more)"
        print(f"{status_icon} {result['test_name']}: {result['status']} ({result['execution_time']:.2f}s)")
        print(f"    Agents: {agents}")
        if result["error"]:
            print(f"    Error: {result['error'][:100]}...")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(__file__).parent / f"cross_agent_integration_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())