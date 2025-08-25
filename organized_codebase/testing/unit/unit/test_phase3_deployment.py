"""
Phase 3 Deployment Test
========================

Validates enterprise deployment architecture components.
"""

import sys
import asyncio
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_enterprise_deployment():
    """Test enterprise deployment system"""
    print("Testing Enterprise Deployment...")
    
    try:
        from deployment import (
            EnterpriseTestDeployment,
            ServiceConfig,
            ServiceType,
            DeploymentMode,
            DeploymentStatus
        )
        
        # Create deployment
        deployment = EnterpriseTestDeployment(DeploymentMode.STAGING)
        print(f"[OK] Created deployment: {deployment.deployment_id}")
        
        # Deploy a test service
        test_service = ServiceConfig(
            service_type=ServiceType.TEST_EXECUTOR,
            name="Test Executor Service",
            replicas=2
        )
        service_id = deployment.deploy_service(test_service)
        print(f"[OK] Deployed service: {service_id}")
        
        # Check deployment status
        status = deployment.get_deployment_status()
        print(f"[OK] Deployment status: {status['status']}, Services: {len(status['services'])}")
        
        # Perform health check
        health = await deployment.perform_health_check()
        print(f"[OK] Health check completed: {len(health['services'])} services checked")
        
        # Test scaling
        success = await deployment.scale_service(service_id, 3)
        print(f"[OK] Scaled service: {success}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Enterprise deployment test failed: {e}")
        return False

async def test_service_registry():
    """Test service registry system"""
    print("\nTesting Service Registry...")
    
    try:
        from deployment import (
            ServiceRegistry,
            ServiceDescriptor,
            ServiceEndpoint,
            ServiceHealth
        )
        
        # Create registry
        registry = ServiceRegistry()
        print(f"[OK] Created registry: {registry.registry_id}")
        
        # Register a service
        service = ServiceDescriptor(
            name="Test Service",
            service_type="test_executor",
            endpoints=[
                ServiceEndpoint(host="localhost", port=8080)
            ],
            capabilities=["test_execution", "test_analysis"],
            tags={"environment", "test"}
        )
        service_id = registry.register_service(service)
        print(f"[OK] Registered service: {service_id}")
        
        # Discover services
        discovered = registry.discover_services(service_type="test_executor")
        print(f"[OK] Discovered {len(discovered)} services")
        
        # Update health
        registry.update_health(service_id, ServiceHealth.HEALTHY)
        print(f"[OK] Updated service health")
        
        # Get registry status
        status = registry.get_registry_status()
        print(f"[OK] Registry status: {status['total_services']} services, {status['healthy_services']} healthy")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Service registry test failed: {e}")
        return False

async def test_swarm_orchestrator():
    """Test swarm orchestration system"""
    print("\nTesting Swarm Orchestrator...")
    
    try:
        from deployment import (
            SwarmOrchestrator,
            SwarmConfig,
            SwarmTask,
            SwarmAgent
        )
        
        # Create orchestrator
        config = SwarmConfig(
            name="TestSwarm",
            min_agents=2,
            max_agents=10
        )
        orchestrator = SwarmOrchestrator(config)
        print(f"[OK] Created orchestrator: {orchestrator.orchestrator_id}")
        
        # Add agents
        agent1 = SwarmAgent(
            name="TestAgent1",
            capabilities={"test_execution", "test_analysis"}
        )
        agent_id = orchestrator.add_agent(agent1)
        print(f"[OK] Added agent: {agent_id}")
        
        # Submit task
        task = SwarmTask(
            task_type="test_execution",
            priority=5,
            payload={"test": "data"}
        )
        task_id = orchestrator.submit_task(task)
        print(f"[OK] Submitted task: {task_id}")
        
        # Assign task
        assigned_to = await orchestrator.assign_task(task_id)
        print(f"[OK] Task assigned to: {assigned_to}")
        
        # Complete task
        success = await orchestrator.complete_task(task_id, {"result": "success"})
        print(f"[OK] Task completed: {success}")
        
        # Get swarm status
        status = orchestrator.get_swarm_status()
        print(f"[OK] Swarm status: {status['agents']['total']} agents, {status['tasks']['completed']} tasks completed")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Swarm orchestrator test failed: {e}")
        return False

async def main():
    """Main test runner"""
    print("Phase 3 Enterprise Deployment Validation")
    print("=" * 50)
    
    # Test components
    deployment_success = await test_enterprise_deployment()
    registry_success = await test_service_registry()
    swarm_success = await test_swarm_orchestrator()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Enterprise Deployment: {'PASS' if deployment_success else 'FAIL'}")
    print(f"Service Registry: {'PASS' if registry_success else 'FAIL'}")
    print(f"Swarm Orchestrator: {'PASS' if swarm_success else 'FAIL'}")
    
    overall_success = deployment_success and registry_success and swarm_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\nPhase 3 deployment components are working correctly!")
    else:
        print("\nSome Phase 3 components need attention.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)