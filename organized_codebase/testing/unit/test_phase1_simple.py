"""
Simple Phase 1 Integration Test
===============================

Basic validation test for Phase 1 multi-agent integration components.
"""

import sys
import os
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all Phase 1 components can be imported"""
    print("Testing Phase 1 Component Imports...")
    
    try:
        from core.orchestration import TestOrchestrationEngine, TestAgent
        print("[OK] Orchestration components imported")
    except Exception as e:
        print(f"[FAIL] Orchestration import failed: {e}")
        return False
    
    try:
        from core.observability import TestMasterObservability, TestSession
        print("[OK] Observability components imported")
    except Exception as e:
        print(f"[FAIL] Observability import failed: {e}")
        return False
    
    try:
        from core.tools import TypeSafeTool, ToolRegistry
        print("[OK] Type-safe tools imported")
    except Exception as e:
        print(f"[FAIL] Tools import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of Phase 1 components"""
    print("\nTesting Basic Functionality...")
    
    try:
        # Test orchestration
        from core.orchestration import TestOrchestrationEngine, TestAgent
        engine = TestOrchestrationEngine()
        agent = TestAgent("test", "Test Agent", "tester", ["test"])
        agent_id = engine.register_agent(agent)
        print(f"[OK] Agent registered: {agent_id}")
        
        # Test observability
        from core.observability import TestMasterObservability
        obs = TestMasterObservability()
        session_id = obs.start_test_session("Test Session")
        obs.end_test_session(session_id, "completed")
        print(f"[OK] Session created: {session_id}")
        
        # Test tools
        from core.tools import global_tool_registry
        tools = global_tool_registry.get_available_tools()
        print(f"[OK] Tools available: {len(tools)}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("Phase 1 Integration Validation")
    print("=" * 40)
    
    # Test imports
    import_success = test_imports()
    
    # Test functionality if imports succeeded
    func_success = False
    if import_success:
        func_success = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"Functionality: {'PASS' if func_success else 'FAIL'}")
    
    overall_success = import_success and func_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\nPhase 1 components are working correctly!")
    else:
        print("\nSome Phase 1 components need attention.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)