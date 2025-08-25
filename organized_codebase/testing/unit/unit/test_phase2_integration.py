"""
Phase 2 Integration Test
========================

Validation test for Phase 2 multi-agent testing framework and enhanced monitoring.
Tests role-based testing, hierarchical supervision, and conversational monitoring.

Author: TestMaster Team
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

def test_phase2_imports():
    """Test that all Phase 2 components can be imported"""
    print("Testing Phase 2 Component Imports...")
    
    try:
        from agents.roles import TestArchitect, TestEngineer, QualityAssuranceAgent
        print("[OK] Multi-agent roles imported")
    except Exception as e:
        print(f"[FAIL] Roles import failed: {e}")
        return False
    
    try:
        from agents.supervisor import TestingSupervisor, SupervisorMode
        print("[OK] Supervisor components imported")
    except Exception as e:
        print(f"[FAIL] Supervisor import failed: {e}")
        return False
    
    try:
        from agents.team import TestingTeam, TeamConfiguration, STANDARD_TESTING_WORKFLOW
        print("[OK] Team management imported")
    except Exception as e:
        print(f"[FAIL] Team management import failed: {e}")
        return False
    
    try:
        from monitoring import EnhancedTestMonitor, PerformanceMonitoringAgent
        print("[OK] Enhanced monitoring imported")
    except Exception as e:
        print(f"[FAIL] Enhanced monitoring import failed: {e}")
        return False
    
    return True

def test_role_functionality():
    """Test basic role functionality"""
    print("\nTesting Role Functionality...")
    
    try:
        from agents.roles import TestArchitect, TestActionType
        from agents.roles.base_role import TestAction
        
        # Create test architect
        architect = TestArchitect()
        
        # Test action creation
        action = TestAction(
            role="TestArchitect",
            action_type=TestActionType.ANALYZE,
            description="Test analysis action",
            parameters={"target_path": "."}
        )
        
        print(f"[OK] Role created: {architect.name}")
        print(f"[OK] Action created: {action.action_type.value}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Role functionality test failed: {e}")
        return False

def test_supervisor_functionality():
    """Test supervisor functionality"""
    print("\nTesting Supervisor Functionality...")
    
    try:
        from agents.supervisor import TestingSupervisor, SupervisorMode
        
        # Create supervisor
        supervisor = TestingSupervisor(mode=SupervisorMode.GUIDED)
        
        # Test status
        status = supervisor.get_supervision_status()
        
        print(f"[OK] Supervisor created: {supervisor.supervisor_id}")
        print(f"[OK] Mode: {supervisor.mode.value}")
        print(f"[OK] Status obtained: {status['active']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Supervisor functionality test failed: {e}")
        return False

def test_team_functionality():
    """Test team management functionality"""
    print("\nTesting Team Functionality...")
    
    try:
        from agents.team import TestingTeam, TeamConfiguration, TeamRole
        
        # Create standard team
        team = TestingTeam.create_standard_team()
        
        # Test status
        status = team.get_team_status()
        
        print(f"[OK] Team created: {team.team_id}")
        print(f"[OK] Roles count: {len(team.roles)}")
        print(f"[OK] Configuration: {status['configuration']['supervisor_mode']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Team functionality test failed: {e}")
        return False

def test_monitoring_functionality():
    """Test enhanced monitoring functionality"""
    print("\nTesting Enhanced Monitoring Functionality...")
    
    try:
        from monitoring import EnhancedTestMonitor, MonitoringMode, PerformanceMonitoringAgent
        
        # Create monitor
        monitor = EnhancedTestMonitor(mode=MonitoringMode.INTERACTIVE)
        
        # Add monitoring agent
        performance_agent = PerformanceMonitoringAgent()
        monitor.add_monitoring_agent(performance_agent)
        
        # Test status
        status = monitor.get_monitoring_status()
        
        print(f"[OK] Monitor created: {monitor.monitor_id}")
        print(f"[OK] Mode: {monitor.mode.value}")
        print(f"[OK] Agents: {len(monitor.monitoring_agents)}")
        print(f"[OK] Status obtained: uptime {status['uptime']:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Enhanced monitoring functionality test failed: {e}")
        return False

async def test_integration_workflow():
    """Test complete Phase 2 integration workflow"""
    print("\nTesting Phase 2 Integration Workflow...")
    
    try:
        from agents.team import TestingTeam, STANDARD_TESTING_WORKFLOW
        from monitoring import EnhancedTestMonitor, MonitoringMode
        
        # Create team and monitor
        team = TestingTeam.create_minimal_team()  # Use minimal for faster testing
        monitor = EnhancedTestMonitor(mode=MonitoringMode.PASSIVE)
        
        # Start components
        await team.start_team()
        await monitor.start_monitoring()
        
        print(f"[OK] Team started: {team.team_id}")
        print(f"[OK] Monitor started: {monitor.monitor_id}")
        
        # Test conversational interface
        response = await monitor.process_conversation("What's the system status?")
        print(f"[OK] Conversational response: {len(response)} chars")
        
        # Stop components
        await team.stop_team()
        await monitor.stop_monitoring()
        
        print(f"[OK] Components stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Integration workflow test failed: {e}")
        return False

def run_phase2_validation():
    """Run complete Phase 2 validation"""
    print("Phase 2 Integration Validation")
    print("=" * 40)
    
    start_time = time.time()
    
    # Test results
    results = {
        "imports": test_phase2_imports(),
        "roles": test_role_functionality(),
        "supervisor": test_supervisor_functionality(),
        "team": test_team_functionality(),
        "monitoring": test_monitoring_functionality()
    }
    
    # Test integration workflow
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results["integration"] = loop.run_until_complete(test_integration_workflow())
    except Exception as e:
        print(f"[FAIL] Integration workflow: {e}")
        results["integration"] = False
    finally:
        loop.close()
    
    execution_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 40)
    print("Phase 2 Integration Test Summary")
    print("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    if passed == total:
        print("\nPhase 2 Integration: SUCCESSFUL")
        print("All advanced coordination components are working correctly!")
        return True
    else:
        print(f"\nPhase 2 Integration: PARTIAL ({passed}/{total} components)")
        print("Some components need attention.")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    success = run_phase2_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)