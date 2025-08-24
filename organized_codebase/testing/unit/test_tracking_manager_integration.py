#!/usr/bin/env python3
"""
Test Comprehensive Tracking Manager Integration

Tests the integration of the Tracking Manager with TestMaster components
including test generators, monitoring, and various operations.
"""

import sys
import tempfile
import time
from pathlib import Path

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent))

def test_tracking_manager_basic():
    """Test basic tracking manager functionality."""
    print("Testing Tracking Manager basic functionality...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.tracking_manager import TrackingManager, get_tracking_manager
        
        # Enable tracking
        FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as db_file:
            tracking_manager = TrackingManager(db_path=db_file.name)
            
            # Test chain operations
            chain_id = tracking_manager.start_chain(
                chain_name="test_chain",
                inputs={"test": "data"}
            )
            
            assert chain_id is not None, "Should create chain successfully"
            assert chain_id.startswith("chain_"), "Chain ID should have correct format"
            
            # Test operation tracking
            tracking_manager.track_operation(
                run_id="test_operation_1",
                component="test_component",
                operation="test_operation",
                inputs={"input_key": "input_value"},
                outputs={"output_key": "output_value"},
                parent_run_id=chain_id,
                success=True,
                duration_ms=100.5
            )
            
            # Test chain completion
            tracking_manager.end_chain(
                chain_id=chain_id,
                outputs={"result": "success"},
                success=True
            )
            
            # Test statistics
            stats = tracking_manager.get_tracking_statistics()
            assert stats['enabled'] == True, "Should report as enabled"
            assert 'last_24h' in stats, "Should include 24-hour statistics"
            
            print("   PASS: Basic tracking manager functionality working")
            return True
            
    except Exception as e:
        print(f"   FAIL: Basic tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracking_manager_generator_integration():
    """Test tracking manager integration with BaseGenerator."""
    print("Testing Tracking Manager integration with BaseGenerator...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.layer_manager import get_layer_manager
        from testmaster.generators.base import BaseGenerator, ModuleAnalysis
        
        # Enable required layers and features
        layer_manager = get_layer_manager()
        layer_manager.enable_layer('layer2_monitoring')
        layer_manager.enable_layer('layer2_monitoring', 'tracking_manager')
        
        FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        FeatureFlags.enable('layer1_test_foundation', 'context_preservation')
        
        # Create a test generator
        class TestTrackingGenerator(BaseGenerator):
            def analyze_module(self, module_path, context=None):
                return ModuleAnalysis(
                    purpose="Test module for tracking",
                    classes=[{"name": "TestClass", "methods": ["test_method"]}],
                    functions=[{"name": "test_function", "args": []}]
                )
            
            def generate_test_code(self, module_path, analysis, context=None):
                return "# Generated test code with tracking\\ndef test_example():\\n    assert True"
        
        generator = TestTrackingGenerator()
        
        # Verify tracking manager is initialized
        assert generator.tracking_manager is not None, "Tracking manager should be initialized"
        
        # Test that tracking is enabled
        assert generator.tracking_manager.enabled == True, "Tracking should be enabled"
        
        # Test getting active chains (should be empty initially)
        active_chains = generator.tracking_manager.get_active_chains()
        assert isinstance(active_chains, list), "Should return list of active chains"
        
        # Test statistics include tracking
        stats = generator.get_stats()
        assert 'tracking' in stats, "Stats should include tracking information"
        assert stats['tracking']['enabled'] == True, "Tracking should be reported as enabled"
        
        print("   PASS: Tracking manager integration with BaseGenerator working")
        return True
        
    except Exception as e:
        print(f"   FAIL: Generator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracking_manager_chain_depth():
    """Test tracking manager chain depth limits."""
    print("Testing Tracking Manager chain depth limits...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.tracking_manager import TrackingManager
        
        # Enable tracking with specific chain depth
        FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as db_file:
            tracking_manager = TrackingManager(db_path=db_file.name)
            
            # Test nested chains up to max depth
            chain_ids = []
            parent_id = None
            
            # Create chains up to max depth
            for i in range(tracking_manager.max_chain_depth):
                chain_id = tracking_manager.start_chain(
                    chain_name=f"nested_chain_{i}",
                    inputs={"depth": i},
                    parent_chain_id=parent_id
                )
                chain_ids.append(chain_id)
                parent_id = chain_id
            
            # Test active chains
            active_chains = tracking_manager.get_active_chains()
            assert len(active_chains) == tracking_manager.max_chain_depth, \
                f"Should have {tracking_manager.max_chain_depth} active chains"
            
            # End all chains
            for chain_id in reversed(chain_ids):
                tracking_manager.end_chain(chain_id, success=True)
            
            # Verify all chains ended
            active_chains = tracking_manager.get_active_chains()
            assert len(active_chains) == 0, "Should have no active chains after ending all"
            
            print("   PASS: Chain depth limits working correctly")
            return True
            
    except Exception as e:
        print(f"   FAIL: Chain depth test failed: {e}")
        return False


def test_tracking_manager_decorator():
    """Test tracking manager operation decorator."""
    print("Testing Tracking Manager operation decorator...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.tracking_manager import track_operation, get_tracking_manager
        
        # Enable tracking
        FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
        
        # Test the decorator
        @track_operation("test_component", "test_operation")
        def test_function(x, y):
            time.sleep(0.01)  # Small delay for timing
            return x + y
        
        # Execute decorated function
        result = test_function(2, 3)
        assert result == 5, "Function should work correctly with decorator"
        
        # Test decorator with disabled tracking
        FeatureFlags.disable('layer2_monitoring', 'tracking_manager')
        
        @track_operation("test_component", "disabled_test")
        def test_function_disabled(x):
            return x * 2
        
        result = test_function_disabled(5)
        assert result == 10, "Function should work when tracking disabled"
        
        print("   PASS: Operation decorator working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Decorator test failed: {e}")
        return False


def test_tracking_manager_error_handling():
    """Test tracking manager error handling."""
    print("Testing Tracking Manager error handling...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.tracking_manager import TrackingManager
        
        # Enable tracking
        FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as db_file:
            tracking_manager = TrackingManager(db_path=db_file.name)
            
            # Test ending non-existent chain (should handle gracefully)
            tracking_manager.end_chain("non_existent_chain", success=True)
            
            # Test tracking operation with error
            chain_id = tracking_manager.start_chain("error_test_chain")
            
            tracking_manager.track_operation(
                run_id="error_operation",
                component="test_component",
                operation="failing_operation",
                parent_run_id=chain_id,
                success=False,
                error="Test error message"
            )
            
            # End chain with error
            tracking_manager.end_chain(
                chain_id=chain_id,
                success=False,
                error="Chain failed due to operation error"
            )
            
            # Verify statistics include error counts
            stats = tracking_manager.get_tracking_statistics()
            assert 'last_24h' in stats, "Should include 24-hour statistics"
            
            print("   PASS: Error handling working correctly")
            return True
            
    except Exception as e:
        print(f"   FAIL: Error handling test failed: {e}")
        return False


def test_tracking_manager_disabled():
    """Test tracking manager when disabled."""
    print("Testing Tracking Manager when disabled...")
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        from testmaster.core.tracking_manager import TrackingManager, get_tracking_manager
        
        # Disable tracking
        FeatureFlags.disable('layer2_monitoring', 'tracking_manager')
        
        tracking_manager = TrackingManager()
        
        # Should report as disabled
        assert tracking_manager.enabled == False, "Should be disabled"
        
        # Operations should be no-ops
        chain_id = tracking_manager.start_chain("disabled_chain")
        assert chain_id.startswith("disabled_"), "Should return disabled chain ID"
        
        tracking_manager.track_operation(
            run_id="disabled_op",
            component="test",
            operation="test"
        )
        
        tracking_manager.end_chain(chain_id, success=True)
        
        # Statistics should show disabled
        stats = tracking_manager.get_tracking_statistics()
        assert stats['enabled'] == False, "Should report as disabled"
        
        print("   PASS: Disabled state working correctly")
        return True
        
    except Exception as e:
        print(f"   FAIL: Disabled test failed: {e}")
        return False


def run_tracking_integration_tests():
    """Run all tracking manager integration tests."""
    print("Starting Tracking Manager Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_tracking_manager_basic),
        ("Generator Integration", test_tracking_manager_generator_integration),
        ("Chain Depth Limits", test_tracking_manager_chain_depth),
        ("Operation Decorator", test_tracking_manager_decorator),
        ("Error Handling", test_tracking_manager_error_handling),
        ("Disabled State", test_tracking_manager_disabled)
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
    print("TRACKING INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\\nComprehensive Tracking Manager Integration Complete!")
        return True
    else:
        print(f"\\nWARNING: {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_tracking_integration_tests()
    sys.exit(0 if success else 1)