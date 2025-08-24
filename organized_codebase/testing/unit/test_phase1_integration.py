"""
Phase 1 Integration Test
========================

Validation test for Phase 1 multi-agent integration components.
Tests Agent Orchestration, AgentOps Observability, and Type-Safe Tools.

Author: TestMaster Team
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports and basic functionality
def test_orchestration_components():
    """Test Agent Orchestration Framework components"""
    print("🔧 Testing Agent Orchestration Framework...")
    
    try:
        from core.orchestration import (
            TestOrchestrationEngine, TestAgent, OrchestrationTask,
            AgentStatus, OrchestrationMode, orchestration_engine
        )
        
        # Test agent creation
        test_agent = TestAgent(
            agent_id="test_agent_001",
            name="Test Agent",
            role="tester",
            capabilities=["test_execution", "validation"]
        )
        
        # Test orchestration engine
        engine = TestOrchestrationEngine()
        agent_id = engine.register_agent(test_agent)
        
        # Test status
        status = engine.get_orchestration_status()
        
        print(f"✅ Agent Orchestration Framework: OK")
        print(f"   - Registered agent: {agent_id}")
        print(f"   - System health: {status['system_health']}")
        print(f"   - Active agents: {status['metrics']['active_agents']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent Orchestration Framework: FAILED - {e}")
        return False

def test_observability_components():
    """Test AgentOps Observability Integration components"""
    print("\n📊 Testing AgentOps Observability Integration...")
    
    try:
        from core.observability import (
            TestMasterObservability, TestSession, AgentAction,
            LLMCall, CostTracker, global_observability
        )
        
        # Test session management
        obs = TestMasterObservability()
        session_id = obs.start_test_session(
            "Phase 1 Test Session",
            metadata={"test": "phase1_validation"}
        )
        
        # Test action tracking
        action_id = obs.track_agent_action(
            session_id,
            "test_agent", 
            "validation_test",
            {"component": "observability"}
        )
        
        obs.complete_agent_action(action_id, result="success")
        
        # Test session completion
        session = obs.end_test_session(session_id, "completed")
        
        # Test cost tracking
        cost_tracker = CostTracker()
        test_cost = cost_tracker.calculate_cost("gpt-4", 100, 50)
        
        print(f"✅ AgentOps Observability Integration: OK")
        print(f"   - Session created: {session_id}")
        print(f"   - Session duration: {session.duration:.2f}s")
        print(f"   - Cost calculation test: ${test_cost:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ AgentOps Observability Integration: FAILED - {e}")
        return False

def test_type_safe_tools():
    """Test Type-Safe Tool Integration components"""
    print("\n🛠️ Testing Type-Safe Tool Integration...")
    
    try:
        from core.tools import (
            TypeSafeTool, ToolRegistry, ToolInput, ToolOutput,
            ToolMetadata, ToolCategory, ToolStatus, ValidationLevel,
            global_tool_registry
        )
        
        # Test tool metadata
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool for Phase 1 validation",
            category=ToolCategory.UTILITY
        )
        
        # Test tool registry
        available_tools = global_tool_registry.get_available_tools()
        
        # Test validation levels
        validation_levels = [ValidationLevel.STRICT, ValidationLevel.MODERATE, ValidationLevel.LENIENT]
        
        print(f"✅ Type-Safe Tool Integration: OK")
        print(f"   - Available tools: {len(available_tools)}")
        print(f"   - Tool categories: {len(list(ToolCategory))}")
        print(f"   - Validation levels: {len(validation_levels)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Type-Safe Tool Integration: FAILED - {e}")
        return False

def test_concrete_tools():
    """Test concrete tool implementations"""
    print("\n⚙️ Testing Concrete Tool Implementations...")
    
    try:
        from core.tools.concrete_tools import (
            PytestExecutionTool, CoverageAnalysisTool, CodeQualityAnalysisTool
        )
        
        print(f"✅ Concrete Tool Implementations: OK")
        print(f"   - PytestExecutionTool: Available")
        print(f"   - CoverageAnalysisTool: Available") 
        print(f"   - CodeQualityAnalysisTool: Available")
        
        return True
        
    except Exception as e:
        print(f"❌ Concrete Tool Implementations: FAILED - {e}")
        return False

async def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n🔄 Testing Integration Workflow...")
    
    try:
        from core.orchestration import orchestration_engine, TestAgent, OrchestrationMode
        from core.observability import global_observability
        from core.tools import global_tool_registry
        
        # Create test agent
        agent = TestAgent(
            agent_id="integration_test_agent",
            name="Integration Test Agent",
            role="integration_tester",
            capabilities=["orchestration", "observability", "tool_execution"]
        )
        
        agent_id = orchestration_engine.register_agent(agent)
        
        # Start observability session
        session_id = global_observability.start_test_session(
            "Integration Workflow Test",
            metadata={"test_type": "integration", "phase": "1"}
        )
        
        # Test simple workflow execution
        session_config = {
            "session_id": "test_workflow_001",
            "name": "Phase 1 Integration Test",
            "workflow": {
                "tasks": []  # Empty for basic test
            }
        }
        
        result = await orchestration_engine.execute_orchestration_session(
            session_config, 
            OrchestrationMode.WORKFLOW
        )
        
        # Complete observability session
        global_observability.end_test_session(session_id, "completed")
        
        print(f"✅ Integration Workflow: OK")
        print(f"   - Agent registration: Success")
        print(f"   - Session execution: {result['status']}")
        print(f"   - Workflow mode: {result['mode']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration Workflow: FAILED - {e}")
        return False

def run_phase1_validation():
    """Run complete Phase 1 validation"""
    print("Starting Phase 1 Integration Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test results
    results = {
        "orchestration": test_orchestration_components(),
        "observability": test_observability_components(), 
        "type_safe_tools": test_type_safe_tools(),
        "concrete_tools": test_concrete_tools()
    }
    
    # Test integration workflow
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results["integration_workflow"] = loop.run_until_complete(test_integration_workflow())
    except Exception as e:
        print(f"❌ Integration Workflow: FAILED - {e}")
        results["integration_workflow"] = False
    finally:
        loop.close()
    
    execution_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Phase 1 Integration Test Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️ Execution time: {execution_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 Phase 1 Integration: SUCCESSFUL")
        print("All core components are working correctly!")
        return True
    else:
        print(f"\n⚠️ Phase 1 Integration: PARTIAL ({passed}/{total} components)")
        print("Some components need attention before proceeding to Phase 2.")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    success = run_phase1_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)