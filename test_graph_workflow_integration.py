#!/usr/bin/env python3
"""
Test Graph-Based Workflow Management Integration

Tests the integration of the graph-based workflow system with TestMaster
monitoring components including TestMonitor, FileWatcher integration,
and workflow orchestration capabilities.
"""

import sys
import tempfile
import time
import asyncio
from pathlib import Path

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent))

def test_workflow_graph_basic():
    """Test basic workflow graph functionality."""
    print("Testing basic workflow graph functionality...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.workflow_graph import WorkflowGraph, NodeType
        
        # Enable graph workflows
        FeatureFlags.enable('layer2_monitoring', 'graph_workflows')
        
        # Create workflow
        workflow = WorkflowGraph("test_workflow")
        
        # Test basic workflow creation
        assert workflow.enabled == True, "Workflow should be enabled"
        assert workflow.name == "test_workflow", "Workflow should have correct name"
        assert "START" in workflow.nodes, "Should have START node"
        assert "END" in workflow.nodes, "Should have END node"
        
        # Add test nodes
        def test_action(context):
            return {"action_executed": True}
        
        def test_condition(context):
            return "success" if context.data.get("action_executed") else "fail"
        
        workflow.add_node("test_action", NodeType.ACTION, handler=test_action)
        workflow.add_node("test_condition", NodeType.CONDITION, condition=test_condition)
        workflow.add_node("success_node", NodeType.ACTION, 
                         handler=lambda ctx: {"result": "success"})
        workflow.add_node("fail_node", NodeType.ACTION, 
                         handler=lambda ctx: {"result": "failed"})
        
        # Add edges
        workflow.add_edge("START", "test_action")
        workflow.add_edge("test_action", "test_condition")
        workflow.add_conditional_edge("test_condition", test_condition, {
            "success": "success_node",
            "fail": "fail_node"
        })
        workflow.add_edge("success_node", "END")
        workflow.add_edge("fail_node", "END")
        
        # Test workflow execution
        result = workflow.invoke({"initial_data": "test"})
        
        assert result["success"] == True, "Workflow should complete successfully"
        assert result["final_state"] == "COMPLETED", "Should reach completed state"
        assert "START" in result["execution_path"], "Should include START in path"
        assert "END" in result["execution_path"], "Should include END in path"
        assert result["context"]["action_executed"] == True, "Action should execute"
        assert result["context"]["result"] == "success", "Should take success path"
        
        print("   PASS: Basic workflow graph functionality working")
        return True
        
    except Exception as e:
        print(f"   FAIL: Basic workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_graph_parallel():
    """Test parallel workflow execution."""
    print("Testing parallel workflow execution...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.workflow_graph import WorkflowGraph, NodeType
        
        # Enable graph workflows
        FeatureFlags.enable('layer2_monitoring', 'graph_workflows')
        
        # Create workflow with parallel processing
        workflow = WorkflowGraph("parallel_workflow", max_parallel_branches=3)
        
        # Add parallel handlers
        def branch_1(context):
            return {"branch_1": "executed"}
        
        def branch_2(context):
            return {"branch_2": "executed"}
        
        def branch_3(context):
            return {"branch_3": "executed"}
        
        # Add nodes
        workflow.add_node("prepare", NodeType.ACTION, 
                         handler=lambda ctx: {"prepared": True})
        workflow.add_node("parallel_process", NodeType.PARALLEL, 
                         parallel_branches=["branch_1", "branch_2", "branch_3"])
        workflow.add_node("branch_1", NodeType.ACTION, handler=branch_1)
        workflow.add_node("branch_2", NodeType.ACTION, handler=branch_2)
        workflow.add_node("branch_3", NodeType.ACTION, handler=branch_3)
        workflow.add_node("merge", NodeType.ACTION,
                         handler=lambda ctx: {"merged": True})
        
        # Add edges
        workflow.add_edge("START", "prepare")
        workflow.add_edge("prepare", "parallel_process")
        workflow.add_edge("parallel_process", "merge")
        workflow.add_edge("merge", "END")
        
        # Execute workflow
        result = workflow.invoke({"test": "parallel"})
        
        assert result["success"] == True, "Parallel workflow should succeed"
        assert result["context"]["prepared"] == True, "Preparation should complete"
        assert result["context"]["merged"] == True, "Merge should complete"
        assert len(result["parallel_results"]) == 3, "Should have 3 parallel results"
        assert "branch_1" in result["parallel_results"], "Should include branch 1 result"
        assert "branch_2" in result["parallel_results"], "Should include branch 2 result"
        assert "branch_3" in result["parallel_results"], "Should include branch 3 result"
        
        print("   PASS: Parallel workflow execution working")
        return True
        
    except Exception as e:
        print(f"   FAIL: Parallel workflow test failed: {e}")
        return False


def test_test_monitor_integration():
    """Test TestMonitor integration with workflow graphs."""
    print("Testing TestMonitor integration with workflow graphs...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.layer_manager import get_layer_manager
        from testmaster.monitoring.test_monitor import TestMonitor
        
        # Enable required features
        layer_manager = get_layer_manager()
        layer_manager.enable_layer('layer2_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'test_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'test_scheduling')
        
        FeatureFlags.enable('layer2_monitoring', 'graph_workflows')
        FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        
        # Create test monitor
        monitor = TestMonitor(watch_paths=["."], polling_interval=0.1)
        
        # Verify workflow integration
        assert monitor.workflow_graph is not None, "Should have workflow graph"
        assert monitor.workflow_graph.enabled == True, "Workflow should be enabled"
        
        # Test workflow execution through monitor
        context_data = {
            "file_path": "test_module.py",
            "change_type": "modified",
            "timestamp": time.time()
        }
        
        result = monitor.workflow_graph.invoke(context_data)
        
        assert result["success"] == True, "Monitor workflow should succeed"
        assert "context" in result, "Should include context data"
        
        # Test statistics
        stats = monitor.get_monitoring_statistics()
        assert "workflow_graph" in stats, "Should include workflow statistics"
        assert stats["workflow_graph"]["enabled"] == True, "Workflow should be enabled in stats"
        
        print("   PASS: TestMonitor integration working")
        return True
        
    except Exception as e:
        print(f"   FAIL: TestMonitor integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_watcher_integration():
    """Test FileWatcher integration with enhanced features."""
    print("Testing FileWatcher integration with enhanced features...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.layer_manager import get_layer_manager
        from testmaster.monitoring.file_watcher import FileWatcher, WatcherCallbacks
        
        # Enable required features
        layer_manager = get_layer_manager()
        layer_manager.enable_layer('layer2_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'file_monitoring')
        
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file watcher
            callbacks = WatcherCallbacks()
            events_received = []
            
            def test_callback(event):
                events_received.append(event)
            
            callbacks.on_file_created = test_callback
            callbacks.on_file_modified = test_callback
            
            watcher = FileWatcher(watch_paths=[temp_dir], callbacks=callbacks)
            
            # Verify enhanced features
            assert watcher.shared_state is not None, "Should have shared state integration"
            assert watcher.tracking_manager is not None, "Should have tracking manager integration"
            
            print("   PASS: FileWatcher integration working")
            return True
            
    except Exception as e:
        print(f"   FAIL: FileWatcher integration test failed: {e}")
        return False


def test_workflow_error_handling():
    """Test workflow error handling and recovery."""
    print("Testing workflow error handling...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.workflow_graph import WorkflowGraph, NodeType
        
        # Enable graph workflows
        FeatureFlags.enable('layer2_monitoring', 'graph_workflows')
        
        # Create workflow with error-prone node
        workflow = WorkflowGraph("error_test_workflow")
        
        def failing_action(context):
            raise Exception("Intentional test failure")
        
        def recovery_action(context):
            return {"recovered": True}
        
        workflow.add_node("failing_node", NodeType.ACTION, handler=failing_action)
        workflow.add_node("recovery_node", NodeType.ACTION, handler=recovery_action)
        
        workflow.add_edge("START", "failing_node")
        workflow.add_edge("failing_node", "END")
        
        # Execute workflow and expect failure
        result = workflow.invoke({"test": "error"})
        
        assert result["success"] == False, "Workflow should fail"
        assert "error" in result, "Should include error message"
        assert result["final_state"] == "FAILED", "Should be in failed state"
        
        print("   PASS: Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Error handling test failed: {e}")
        return False


def test_workflow_async_execution():
    """Test asynchronous workflow execution."""
    print("Testing asynchronous workflow execution...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.workflow_graph import WorkflowGraph, NodeType
        
        # Enable graph workflows
        FeatureFlags.enable('layer2_monitoring', 'graph_workflows')
        
        # Create simple workflow
        workflow = WorkflowGraph("async_workflow")
        
        def async_action(context):
            time.sleep(0.01)  # Small delay to simulate async work
            return {"async_completed": True}
        
        workflow.add_node("async_task", NodeType.ACTION, handler=async_action)
        workflow.add_edge("START", "async_task")
        workflow.add_edge("async_task", "END")
        
        # Test async execution
        async def run_async_test():
            result = await workflow.ainvoke({"async": True})
            return result
        
        # Run async test
        result = asyncio.run(run_async_test())
        
        assert result["success"] == True, "Async workflow should succeed"
        assert result["context"]["async_completed"] == True, "Async action should complete"
        
        print("   PASS: Async execution working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Async execution test failed: {e}")
        return False


def test_workflow_disabled_fallback():
    """Test workflow fallback when disabled."""
    print("Testing workflow disabled fallback...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.workflow_graph import WorkflowGraph
        
        # Disable graph workflows
        FeatureFlags.disable('layer2_monitoring', 'graph_workflows')
        
        # Create workflow (should be disabled)
        workflow = WorkflowGraph("disabled_workflow")
        
        assert workflow.enabled == False, "Workflow should be disabled"
        
        # Test fallback execution
        result = workflow.invoke({"test": "fallback"})
        
        assert "fallback" in result, "Should indicate fallback mode"
        assert result["final_state"] == "COMPLETED_LINEAR", "Should use linear completion"
        
        print("   PASS: Disabled fallback working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Disabled fallback test failed: {e}")
        return False


def run_graph_workflow_tests():
    """Run all graph workflow integration tests."""
    print("Starting Graph-Based Workflow Management Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Workflow Functionality", test_workflow_graph_basic),
        ("Parallel Workflow Execution", test_workflow_graph_parallel),
        ("TestMonitor Integration", test_test_monitor_integration),
        ("FileWatcher Integration", test_file_watcher_integration),
        ("Error Handling", test_workflow_error_handling),
        ("Async Execution", test_workflow_async_execution),
        ("Disabled Fallback", test_workflow_disabled_fallback)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\\nRunning: {test_name}")
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                failed += 1
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[CRASH] {test_name}: {e}")
            failed += 1
    
    print("\\n" + "=" * 60)
    print("GRAPH WORKFLOW TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\\nGraph-Based Workflow Management Integration Complete!")
        return True
    else:
        print(f"\\nWARNING: {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_graph_workflow_tests()
    sys.exit(0 if success else 1)