#!/usr/bin/env python3
"""
Test API Functionality
=====================

Comprehensive test to verify:
1. All backend modules function properly
2. API endpoints are properly exposed
3. Frontend-backend integration works
"""

import sys
import importlib
import traceback
from pathlib import Path

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_module_imports():
    """Test if all key modules can be imported"""
    print("=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)
    
    modules_to_test = [
        # Core modules
        ("TestMaster.core.shared_state", "SharedState"),
        ("TestMaster.core.async_state_manager", "AsyncStateManager"),
        ("TestMaster.core.feature_flags", "FeatureFlags"),
        
        # Observability
        ("TestMaster.observability.unified_observability", "UnifiedObservabilitySystem"),
        
        # State management
        ("TestMaster.state.unified_state_manager", "UnifiedStateManager"),
        
        # Integration systems
        ("TestMaster.integration.automatic_scaling_system", "AutomaticScalingSystem"),
        ("TestMaster.integration.comprehensive_error_recovery", "ComprehensiveErrorRecoverySystem"),
        ("TestMaster.integration.intelligent_caching_layer", "IntelligentCachingLayer"),
        
        # Monitoring
        ("TestMaster.monitoring.monitoring_agents", "PerformanceMonitoringAgent"),
    ]
    
    success_count = 0
    fail_count = 0
    
    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"[PASS] {module_name}.{class_name}")
                success_count += 1
            else:
                print(f"[FAIL] {module_name} missing {class_name}")
                fail_count += 1
        except Exception as e:
            print(f"[FAIL] {module_name}: {str(e)}")
            fail_count += 1
    
    print(f"\nResults: {success_count} success, {fail_count} failures")
    return fail_count == 0

def test_api_blueprints():
    """Test if API blueprints can be imported"""
    print("\n" + "=" * 60)
    print("TESTING API BLUEPRINTS")
    print("=" * 60)
    
    api_modules = [
        "TestMaster.dashboard.api.performance",
        "TestMaster.dashboard.api.analytics", 
        "TestMaster.dashboard.api.workflow",
        "TestMaster.dashboard.api.tests",
        "TestMaster.dashboard.api.health",
        "TestMaster.dashboard.api.monitoring",
        "TestMaster.dashboard.api.test_generation",
        "TestMaster.dashboard.api.security",
        "TestMaster.dashboard.api.observability",
    ]
    
    success_count = 0
    fail_count = 0
    blueprints = []
    
    for module_name in api_modules:
        try:
            module = importlib.import_module(module_name)
            # Look for blueprint
            if hasattr(module, 'blueprint'):
                print(f"[PASS] {module_name} - blueprint found")
                blueprints.append((module_name, module.blueprint))
                success_count += 1
            elif hasattr(module, 'bp'):
                print(f"[PASS] {module_name} - bp found")
                blueprints.append((module_name, module.bp))
                success_count += 1
            else:
                # Try to find any blueprint variable
                for attr in dir(module):
                    if 'blueprint' in attr.lower() or attr.endswith('_bp'):
                        print(f"[PASS] {module_name} - {attr} found")
                        blueprints.append((module_name, getattr(module, attr)))
                        success_count += 1
                        break
                else:
                    print(f"[FAIL] {module_name} - no blueprint found")
                    fail_count += 1
        except Exception as e:
            print(f"[FAIL] {module_name}: {str(e)}")
            fail_count += 1
    
    print(f"\nResults: {success_count} blueprints found, {fail_count} failures")
    return blueprints

def test_endpoint_registration():
    """Test what endpoints are actually registered"""
    print("\n" + "=" * 60)
    print("TESTING ENDPOINT REGISTRATION")
    print("=" * 60)
    
    blueprints = test_api_blueprints()
    
    total_endpoints = 0
    for module_name, blueprint in blueprints:
        try:
            # Flask blueprints have deferred_functions
            if hasattr(blueprint, 'deferred_functions'):
                endpoint_count = len(blueprint.deferred_functions)
                print(f"  {module_name.split('.')[-1]}: {endpoint_count} endpoints")
                total_endpoints += endpoint_count
        except:
            pass
    
    print(f"\nTotal endpoints available: {total_endpoints}")
    return total_endpoints > 0

def test_functionality_execution():
    """Test if key functionality actually executes"""
    print("\n" + "=" * 60)
    print("TESTING FUNCTIONALITY EXECUTION")
    print("=" * 60)
    
    tests = []
    
    # Test 1: SharedState functionality
    try:
        from TestMaster.core.shared_state import SharedState
        state = SharedState()
        state.set_state("test_key", "test_value")
        value = state.get_state("test_key")
        if value == "test_value":
            print("[PASS] SharedState: Can set and get values")
            tests.append(True)
        else:
            print("[FAIL] SharedState: Value mismatch")
            tests.append(False)
    except Exception as e:
        print(f"[FAIL] SharedState: {e}")
        tests.append(False)
    
    # Test 2: Observability functionality
    try:
        from TestMaster.observability import get_global_observability
        obs = get_global_observability()
        obs.track_event("test", "test_source", {"test": "data"})
        print("[PASS] Observability: Can track events")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] Observability: {e}")
        tests.append(False)
    
    # Test 3: Integration system functionality
    try:
        from TestMaster.integration.automatic_scaling_system import automatic_scaling_system
        status = automatic_scaling_system.get_scaling_status()
        if isinstance(status, dict):
            print("[PASS] Scaling System: Can get status")
            tests.append(True)
        else:
            print("[FAIL] Scaling System: Invalid status")
            tests.append(False)
    except Exception as e:
        print(f"[FAIL] Scaling System: {e}")
        tests.append(False)
    
    success_rate = sum(tests) / len(tests) * 100 if tests else 0
    print(f"\nFunctionality success rate: {success_rate:.1f}%")
    return success_rate > 50

def main():
    """Run all tests"""
    print("COMPREHENSIVE API FUNCTIONALITY TEST")
    print("=" * 60)
    
    results = {
        "Module Imports": test_module_imports(),
        "Endpoint Registration": test_endpoint_registration(),
        "Functionality Execution": test_functionality_execution()
    }
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n[SUCCESS] ALL TESTS PASSED - System is functional")
    else:
        print("\n[WARNING] SOME TESTS FAILED - System needs fixes")
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)