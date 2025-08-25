#!/usr/bin/env python3
"""
Comprehensive Test Suite for TestMaster Enhancements

Tests all implemented features:
- FeatureFlags system
- SharedState management
- Performance monitoring
- Context preservation
- Advanced configuration
- Handoff tools
"""

import sys
import os
import tempfile
import time
import json
from pathlib import Path

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent))

def test_feature_flags():
    """Test FeatureFlags system."""
    print("Testing FeatureFlags system...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        
        # Initialize feature flags
        FeatureFlags.initialize()
        
        # Test enabling/disabling features
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        assert FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'), "Feature should be enabled"
        
        FeatureFlags.disable('layer1_test_foundation', 'shared_state')
        assert not FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'), "Feature should be disabled"
        
        # Test getting all features
        features = FeatureFlags.get_all_features()
        assert isinstance(features, dict), "Should return dictionary"
        
        print("   PASS: FeatureFlags system working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: FeatureFlags test failed: {e}")
        return False


def test_shared_state():
    """Test SharedState management."""
    print("Testing SharedState management...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.shared_state import SharedState, get_shared_state
        
        # Enable shared state
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        
        # Get shared state instance
        shared_state = get_shared_state()
        
        # Test basic operations
        shared_state.set('test_key', 'test_value')
        value = shared_state.get('test_key')
        assert value == 'test_value', f"Expected 'test_value', got {value}"
        
        # Test increment
        shared_state.set('counter', 5)
        new_value = shared_state.increment('counter', 3)
        assert new_value == 8, f"Expected 8, got {new_value}"
        
        # Test append
        shared_state.set('list', [1, 2])
        new_list = shared_state.append('list', 3)
        assert new_list == [1, 2, 3], f"Expected [1, 2, 3], got {new_list}"
        
        # Test statistics
        stats = shared_state.get_stats()
        assert 'reads' in stats, "Stats should contain 'reads'"
        assert 'writes' in stats, "Stats should contain 'writes'"
        
        print("   PASS: SharedState management working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: SharedState test failed: {e}")
        return False


def test_performance_monitoring():
    """Test Performance monitoring decorators."""
    print("Testing Performance monitoring...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.monitoring_decorators import monitor_performance, get_performance_summary
        
        # Enable performance monitoring
        FeatureFlags.enable('layer1_test_foundation', 'performance_monitoring')
        
        # Test decorator
        @monitor_performance(name="test_function")
        def test_func():
            time.sleep(0.01)  # Small delay
            return "test_result"
        
        # Call the function
        result = test_func()
        assert result == "test_result", "Function should return correct value"
        
        # Check performance summary
        summary = get_performance_summary()
        assert isinstance(summary, dict), "Should return performance summary dictionary"
        
        print("   PASS: Performance monitoring working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Performance monitoring test failed: {e}")
        return False


def test_context_preservation():
    """Test Context preservation system."""
    print("Testing Context preservation...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.context_manager import ContextManager, get_context_manager
        
        # Enable context preservation
        FeatureFlags.enable('layer1_test_foundation', 'context_preservation')
        
        # Get context manager
        context_manager = get_context_manager()
        
        # Test context preservation
        test_context = {
            'module_path': '/test/module.py',
            'generation_phase': 'testing',
            'metadata': {'test': 'data'}
        }
        
        preserved_context = context_manager.preserve(test_context, 'test_ctx')
        assert '_preservation' in preserved_context, "Should contain preservation metadata"
        
        # Test context retrieval
        retrieved_context = context_manager.retrieve('test_ctx')
        assert retrieved_context is not None, "Should retrieve context"
        assert retrieved_context['module_path'] == '/test/module.py', "Should preserve data correctly"
        
        # Test context injection
        source_code = 'def test_function():\n    pass'
        injected_code = context_manager.inject_context(source_code, test_context)
        assert 'Context Information:' in injected_code, "Should inject context header"
        
        # Test statistics
        stats = context_manager.get_context_statistics()
        assert 'total_contexts' in stats, "Should provide statistics"
        
        print("   PASS: Context preservation working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Context preservation test failed: {e}")
        return False


def test_advanced_configuration():
    """Test Advanced configuration management."""
    print("Testing Advanced configuration...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.layer_manager import LayerManager, get_layer_manager
        
        # Enable advanced config
        FeatureFlags.enable('layer1_test_foundation', 'advanced_config')
        
        # Get layer manager
        layer_manager = get_layer_manager()
        
        # Test configuration status
        status = layer_manager.get_status()
        assert isinstance(status, dict), "Should return status dictionary"
        
        # Test enabling/disabling layers
        layer_manager.enable_layer('layer2_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'file_monitoring')
        assert layer_manager.is_enabled('layer2_monitoring', 'file_monitoring'), "Feature should be enabled"
        
        # Test configuration validation
        issues = layer_manager.validate_config()
        assert isinstance(issues, list), "Should return list of issues"
        
        print("   PASS: Advanced configuration working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Advanced configuration test failed: {e}")
        return False


def test_handoff_manager():
    """Test HandoffManager with advanced tools."""
    print("Testing HandoffManager...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.orchestrator.handoff_manager import HandoffManager, HandoffType
        
        # Enable handoff tools
        FeatureFlags.enable('layer2_monitoring', 'handoff_tools')
        
        # Create temporary directory for handoffs
        with tempfile.TemporaryDirectory() as temp_dir:
            handoff_manager = HandoffManager(temp_dir)
            
            # Test creating a handoff
            handoff_id = handoff_manager.create_handoff(
                handoff_type=HandoffType.INVESTIGATION_REQUEST,
                title="Test Handoff",
                description="Testing handoff creation",
                primary_target="test_module.py",
                requested_action="Test the handoff system"
            )
            
            assert handoff_id is not None, "Should create handoff successfully"
            
            # Test retrieving handoff
            handoff = handoff_manager.get_handoff_by_id(handoff_id)
            assert handoff is not None, "Should retrieve handoff"
            assert handoff.title == "Test Handoff", "Should preserve handoff data"
            
            # Test statistics
            stats = handoff_manager.get_handoff_statistics()
            assert 'total_handoffs' in stats, "Should provide statistics"
            assert stats['total_handoffs'] >= 1, "Should count created handoffs"
            
        print("   PASS: HandoffManager working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: HandoffManager test failed: {e}")
        return False


def test_base_generator_integration():
    """Test BaseGenerator integration with enhancements."""
    print("Testing BaseGenerator integration...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.generators.base import BaseGenerator, GenerationConfig, ModuleAnalysis
        
        # Enable all enhancements
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        FeatureFlags.enable('layer1_test_foundation', 'context_preservation')
        FeatureFlags.enable('layer1_test_foundation', 'performance_monitoring')
        
        # Create a test generator
        class TestGenerator(BaseGenerator):
            def analyze_module(self, module_path, context=None):
                return ModuleAnalysis(
                    purpose="Test module",
                    classes=[],
                    functions=[]
                )
            
            def generate_test_code(self, module_path, analysis, context=None):
                return "# Generated test code"
        
        generator = TestGenerator()
        
        # Test that enhancements are initialized
        assert generator.shared_state is not None, "SharedState should be initialized"
        assert generator.context_manager is not None, "ContextManager should be initialized"
        
        # Test statistics include enhancement data
        stats = generator.get_stats()
        assert 'shared_state' in stats, "Stats should include shared state info"
        assert 'context_preservation' in stats, "Stats should include context info"
        
        print("   PASS: BaseGenerator integration working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: BaseGenerator integration test failed: {e}")
        return False


def test_feature_toggles():
    """Test that features can be toggled on/off."""
    print("Testing feature toggle functionality...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.shared_state import get_shared_state
        from testmaster.core.context_manager import get_context_manager
        
        # Test with features disabled
        FeatureFlags.disable('layer1_test_foundation', 'shared_state')
        FeatureFlags.disable('layer1_test_foundation', 'context_preservation')
        
        # Performance monitoring decorator should not execute when disabled
        from testmaster.core.monitoring_decorators import monitor_performance
        
        @monitor_performance(name="disabled_test")
        def test_func_disabled():
            return "result"
        
        result = test_func_disabled()
        assert result == "result", "Function should still work when monitoring disabled"
        
        # Test enabling features
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        FeatureFlags.enable('layer1_test_foundation', 'context_preservation')
        
        # Should work when enabled
        shared_state = get_shared_state()
        shared_state.set('toggle_test', 'enabled')
        value = shared_state.get('toggle_test')
        assert value == 'enabled', "Should work when feature is enabled"
        
        print("   PASS: Feature toggle functionality working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Feature toggle test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Starting TestMaster Enhancement Test Suite")
    print("=" * 60)
    
    tests = [
        ("FeatureFlags System", test_feature_flags),
        ("SharedState Management", test_shared_state),
        ("Performance Monitoring", test_performance_monitoring),
        ("Context Preservation", test_context_preservation),
        ("Advanced Configuration", test_advanced_configuration),
        ("HandoffManager", test_handoff_manager),
        ("BaseGenerator Integration", test_base_generator_integration),
        ("Feature Toggle Functionality", test_feature_toggles)
    ]
    
    passed = 0
    failed = 0
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            success = test_func()
            if success:
                passed += 1
                results.append((test_name, "PASS"))
            else:
                failed += 1
                results.append((test_name, "FAIL"))
        except Exception as e:
            print(f"   CRASH: Test crashed: {e}")
            failed += 1
            results.append((test_name, "CRASH"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status_symbol = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"{status_symbol} {test_name}: {result}")
    
    print(f"\nOVERALL RESULTS:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\nALL TESTS PASSED! Implementation is ready for production.")
        return True
    else:
        print(f"\nWARNING: {failed} test(s) failed. Please review and fix before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)