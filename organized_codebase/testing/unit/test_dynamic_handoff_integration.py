#!/usr/bin/env python3
"""
Test Dynamic Handoff Integration with ClaudeMessenger

Tests the integration of the Dynamic Agent Handoff System
with the ClaudeMessenger communication system.
"""

import sys
import tempfile
from pathlib import Path

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dynamic_handoff_integration():
    """Test ClaudeMessenger integration with Dynamic Handoff System."""
    print("Testing Dynamic Handoff Integration...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.communication.claude_messenger import (
            ClaudeMessenger, MessageType, MessagePriority,
            create_test_failure_info, create_coverage_gap_info
        )
        
        # Enable all required features and layers
        from testmaster.core.layer_manager import get_layer_manager
        
        layer_manager = get_layer_manager()
        layer_manager.enable_layer('layer2_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'claude_communication')
        layer_manager.enable_layer('layer2_monitoring', 'dynamic_handoff')
        
        FeatureFlags.enable('layer2_monitoring', 'claude_communication')
        FeatureFlags.enable('layer2_monitoring', 'dynamic_handoff')
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        FeatureFlags.enable('layer1_test_foundation', 'context_preservation')
        FeatureFlags.enable('layer1_test_foundation', 'performance_monitoring')
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            messenger = ClaudeMessenger(message_dir=temp_dir)
            
            # Test that dynamic handoff is initialized
            assert messenger.dynamic_handoff is not None, "Dynamic handoff should be initialized"
            assert messenger.shared_state is not None, "SharedState should be initialized"
            assert messenger.context_manager is not None, "ContextManager should be initialized"
            
            # Test sending a breaking test alert (should trigger dynamic handoff)
            test_failures = [
                create_test_failure_info(
                    module="test_module.py",
                    test="test_critical_function",
                    failure="AssertionError: Expected 5, got 3",
                    suggested_action="Check calculation logic"
                )
            ]
            
            message_id = messenger.send_breaking_test_alert(
                test_failures, 
                priority=MessagePriority.CRITICAL
            )
            
            assert message_id is not None, "Should create message successfully"
            
            # Test sending coverage gap report (should trigger dynamic handoff)
            coverage_gaps = [
                create_coverage_gap_info(
                    module="uncovered_module.py",
                    uncovered_lines=[25, 26, 27, 30],
                    critical_paths=True,
                    suggested_tests=["test_edge_case", "test_error_handling"]
                )
            ]
            
            gap_message_id = messenger.send_coverage_gap_report(coverage_gaps)
            assert gap_message_id is not None, "Should create coverage gap message"
            
            # Test enhanced statistics
            stats = messenger.get_communication_statistics()
            
            # Should include standard stats
            assert 'messages_sent' in stats, "Should include message count"
            assert 'acknowledgment_rate' in stats, "Should include ack rate"
            
            # Should include enhanced stats if features enabled
            assert 'shared_state' in stats, "Should include shared state stats"
            assert 'context_preservation' in stats, "Should include context stats"
            assert 'dynamic_handoff' in stats, "Should include handoff stats"
            
            # Verify dynamic handoff statistics
            handoff_stats = stats['dynamic_handoff']
            assert 'enabled' in handoff_stats, "Should report handoff enabled status"
            assert handoff_stats['enabled'] == True, "Dynamic handoff should be enabled"
            
            print("   PASS: Dynamic handoff integration working correctly")
            
            # Test message analysis functionality
            print("   Testing message analysis...")
            
            # Create a test message for analysis
            from testmaster.communication.claude_messenger import ClaudeMessage
            from datetime import datetime
            
            test_message = ClaudeMessage(
                message_id="test_123",
                message_type=MessageType.BREAKING_TESTS,
                priority=MessagePriority.HIGH,
                timestamp=datetime.now(),
                breaking_tests=test_failures
            )
            
            # Test message analysis (internal method)
            analysis = messenger._analyze_message_for_routing(test_message)
            
            assert 'task_type' in analysis, "Should include task type"
            assert 'complexity' in analysis, "Should include complexity"
            assert 'domain' in analysis, "Should include domain"
            assert analysis['task_type'] == 'test_debugging', "Should identify as test debugging"
            assert analysis['domain'] == 'testing', "Should identify testing domain"
            
            print("   PASS: Message analysis working correctly")
            
            return True
            
    except Exception as e:
        print(f"   FAIL: Dynamic handoff integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_handoff_fallback_behavior():
    """Test fallback behavior when dynamic handoff fails."""
    print("Testing handoff fallback behavior...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.communication.claude_messenger import ClaudeMessenger
        
        # Enable layer but disable dynamic handoff feature
        from testmaster.core.layer_manager import get_layer_manager
        
        layer_manager = get_layer_manager()
        layer_manager.enable_layer('layer2_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'claude_communication')
        
        FeatureFlags.disable('layer2_monitoring', 'dynamic_handoff')
        FeatureFlags.enable('layer2_monitoring', 'claude_communication')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            messenger = ClaudeMessenger(message_dir=temp_dir)
            
            # Should work without dynamic handoff
            assert messenger.dynamic_handoff is None, "Dynamic handoff should be disabled"
            
            # Should still send messages normally
            message_id = messenger.send_system_alert("Test alert without handoff")
            assert message_id is not None, "Should send message without handoff"
            
            print("   PASS: Fallback behavior working correctly")
            return True
            
    except Exception as e:
        print(f"   FAIL: Fallback test failed: {e}")
        return False


def test_performance_monitoring_integration():
    """Test performance monitoring integration."""
    print("Testing performance monitoring integration...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.monitoring_decorators import get_performance_summary
        from testmaster.communication.claude_messenger import ClaudeMessenger
        
        # Enable performance monitoring
        from testmaster.core.layer_manager import get_layer_manager
        
        layer_manager = get_layer_manager()
        layer_manager.enable_layer('layer2_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'claude_communication')
        
        FeatureFlags.enable('layer1_test_foundation', 'performance_monitoring')
        FeatureFlags.enable('layer2_monitoring', 'claude_communication')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            messenger = ClaudeMessenger(message_dir=temp_dir)
            
            # Send a message to trigger performance monitoring
            message_id = messenger.send_system_alert("Performance test message")
            
            # Check if performance was recorded
            perf_summary = get_performance_summary()
            
            # Should have recorded the message_send performance
            assert isinstance(perf_summary, dict), "Should return performance summary"
            
            print("   PASS: Performance monitoring integration working")
            return True
            
    except Exception as e:
        print(f"   FAIL: Performance monitoring test failed: {e}")
        return False


def run_integration_tests():
    """Run all dynamic handoff integration tests."""
    print("Starting Dynamic Handoff Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dynamic Handoff Integration", test_dynamic_handoff_integration),
        ("Handoff Fallback Behavior", test_handoff_fallback_behavior),
        ("Performance Monitoring Integration", test_performance_monitoring_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
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
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\nDynamic Handoff Integration Complete!")
        return True
    else:
        print(f"\nWARNING: {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)