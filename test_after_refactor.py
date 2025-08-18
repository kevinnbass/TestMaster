"""
Test suite to verify that refactoring worked correctly.
Tests both old names (backward compatibility) and new names.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_new_names_work():
    """Test that new names work correctly."""
    print("\n=== Testing New Names ===")
    try:
        # Import with new names
        from testmaster.intelligence import (
            UniversalHierarchicalTestGenerator,
            HierarchicalPlanningConfig,
            HierarchicalPlanningResult
        )
        
        # Create config with new names
        config = HierarchicalPlanningConfig(
            planning_strategy="best_first",
            max_planning_depth=5,
            target_coverage=85.0
        )
        
        # Create generator
        generator = UniversalHierarchicalTestGenerator(config)
        
        print("+ New names import and work correctly")
        return True
    except Exception as e:
        print(f"X New names test FAILED: {e}")
        traceback.print_exc()
        return False


def test_old_names_still_work():
    """Test that old names still work (backward compatibility)."""
    print("\n=== Testing Backward Compatibility ===")
    try:
        # Import with old names
        from testmaster.intelligence import (
            UniversalToTTestGenerator,
            ToTGenerationConfig,
            ToTGenerationResult
        )
        
        # Create config with old names
        config = ToTGenerationConfig(
            reasoning_strategy="monte_carlo",
            max_reasoning_depth=4,
            target_coverage=80.0
        )
        
        # Create generator
        generator = UniversalToTTestGenerator(config)
        
        print("+ Old names still work (backward compatibility)")
        return True
    except Exception as e:
        print(f"X Old names test FAILED: {e}")
        traceback.print_exc()
        return False


def test_orchestrator_with_new_names():
    """Test that orchestrator works with new names."""
    print("\n=== Testing Orchestrator with New Names ===")
    try:
        from testmaster.orchestration import (
            UniversalTestOrchestrator,
            OrchestrationConfig,
            OrchestrationMode
        )
        
        # Test with new parameter name
        config = OrchestrationConfig(
            mode=OrchestrationMode.INTELLIGENT,
            enable_hierarchical_planning=True,  # New name
            max_workers=1
        )
        
        # Orchestrator should initialize
        orchestrator = UniversalTestOrchestrator(config)
        
        # Check that HTP generator was created
        assert orchestrator.htp_generator is not None, "HTP generator not initialized"
        
        print("+ Orchestrator works with new parameter names")
        return True
    except Exception as e:
        print(f"X Orchestrator test FAILED: {e}")
        traceback.print_exc()
        return False


def test_orchestrator_backward_compat():
    """Test that orchestrator still accepts old parameter names."""
    print("\n=== Testing Orchestrator Backward Compatibility ===")
    try:
        from testmaster.orchestration import (
            UniversalTestOrchestrator,
            OrchestrationConfig,
            OrchestrationMode
        )
        
        # Test with OLD parameter name
        config = OrchestrationConfig(
            mode=OrchestrationMode.INTELLIGENT,
            enable_tot_reasoning=True,  # Old name
            max_workers=1
        )
        
        # Should map to new name internally
        assert config.enable_hierarchical_planning == True, "Old name not mapped to new"
        
        # Orchestrator should initialize
        orchestrator = UniversalTestOrchestrator(config)
        
        # Check that HTP generator was created
        assert orchestrator.htp_generator is not None, "HTP generator not initialized"
        
        print("+ Orchestrator accepts old parameter names")
        return True
    except Exception as e:
        print(f"X Orchestrator backward compat FAILED: {e}")
        traceback.print_exc()
        return False


def test_main_module_updated():
    """Test that main module uses new names."""
    print("\n=== Testing Main Module Updates ===")
    try:
        from testmaster.main import create_orchestration_config
        
        # Create mock args
        class Args:
            mode = 'intelligent'
            target = './test'
            output = None
            enable_intelligence = True
            enable_optimization = True
            enable_llm = False
            enable_security = False
            enable_compliance = False
            enable_security_tests = False
            compliance_standards = []
            auto_detect_frameworks = True
            frameworks = []
            output_formats = None
            include_docs = True
            include_metrics = True
            parallel = True
            workers = 2
            timeout = 600
            min_quality = 0.8
            min_coverage = 0.85
            enable_self_healing = True
        
        # Create config
        config = create_orchestration_config(Args())
        
        # Should use new parameter name
        assert hasattr(config, 'enable_hierarchical_planning'), "Config missing new parameter"
        assert config.enable_hierarchical_planning == True, "New parameter not set correctly"
        
        print("+ Main module uses new parameter names")
        return True
    except Exception as e:
        print(f"X Main module test FAILED: {e}")
        traceback.print_exc()
        return False


def test_aliases_are_same():
    """Test that old and new names refer to the same classes."""
    print("\n=== Testing Aliases ===")
    try:
        from testmaster.intelligence import (
            UniversalToTTestGenerator,
            UniversalHierarchicalTestGenerator,
            ToTGenerationConfig,
            HierarchicalPlanningConfig,
            ToTGenerationResult,
            HierarchicalPlanningResult
        )
        
        # They should be the exact same class
        assert UniversalToTTestGenerator is UniversalHierarchicalTestGenerator, "Generators not aliased"
        assert ToTGenerationConfig is HierarchicalPlanningConfig, "Configs not aliased"
        assert ToTGenerationResult is HierarchicalPlanningResult, "Results not aliased"
        
        print("+ Old and new names are properly aliased")
        return True
    except Exception as e:
        print(f"X Alias test FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all post-refactor tests."""
    print("=" * 70)
    print("POST-REFACTOR TEST SUITE")
    print("Verifying that refactoring was successful")
    print("=" * 70)
    
    tests = [
        test_new_names_work,
        test_old_names_still_work,
        test_orchestrator_with_new_names,
        test_orchestrator_backward_compat,
        test_main_module_updated,
        test_aliases_are_same
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"X {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("\nSUCCESS: Refactoring completed successfully!")
        print("\nWorking features:")
        print("- New names (HierarchicalPlanning) work correctly")
        print("- Old names (ToT) still work for backward compatibility")
        print("- Orchestrator accepts both old and new parameter names")
        print("- Main module updated to use new names")
        print("- Proper aliasing in place")
    else:
        print("\nFAILED: Some tests failed - review refactoring")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)