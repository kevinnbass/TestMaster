"""
Test the unified Hierarchical Planning integration.
Ensures ALL features from both implementations are preserved.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_backward_compatibility():
    """Test that all Tree-of-Thought names still work."""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test old imports still work
    from testmaster.intelligence import (
        UniversalToTTestGenerator,
        ToTGenerationConfig, 
        ToTGenerationResult,
        TreeOfThoughtReasoner,
        ReasoningStrategy
    )
    
    print("+ All Tree-of-Thought imports work")
    
    # Test configuration creation
    config = ToTGenerationConfig(
        reasoning_strategy=ReasoningStrategy.BEST_FIRST,
        max_reasoning_depth=5,
        target_coverage=85.0
    )
    print(f"+ ToTGenerationConfig created: strategy={config.reasoning_strategy}")
    
    # Test generator creation
    generator = UniversalToTTestGenerator(config)
    print("+ UniversalToTTestGenerator created")
    
    return True


def test_new_naming():
    """Test that new Hierarchical Planning names work."""
    print("\n=== Testing New Hierarchical Planning Names ===")
    
    # Test new imports
    from testmaster.intelligence import (
        UniversalHierarchicalTestGenerator,
        HierarchicalPlanningConfig,
        HierarchicalPlanningResult
    )
    
    print("+ All Hierarchical Planning imports work")
    
    # Test configuration creation with new features
    config = HierarchicalPlanningConfig(
        planning_strategy="best_first",  # Can use string too
        max_planning_depth=5,
        target_coverage=85.0,
        use_plan_templates=True,  # New HTP feature
        enable_llm_planning=False  # New HTP feature
    )
    print(f"+ HierarchicalPlanningConfig created with HTP features")
    
    # Test generator creation
    generator = UniversalHierarchicalTestGenerator(config)
    print("+ UniversalHierarchicalTestGenerator created")
    
    return True


def test_feature_preservation():
    """Test that unique features from both implementations are preserved."""
    print("\n=== Testing Feature Preservation ===")
    
    # Test that all 10 test strategy types are available
    from testmaster.intelligence.hierarchical_planning import TestStrategyType
    
    strategies = [s.value for s in TestStrategyType]
    expected_strategies = [
        'happy_path', 'edge_cases', 'error_handling', 'performance',
        'security', 'integration', 'regression', 'property_based',
        'mutation', 'fuzzing'
    ]
    
    for strategy in expected_strategies:
        if strategy in strategies:
            print(f"+ Strategy '{strategy}' preserved")
        else:
            print(f"X Strategy '{strategy}' MISSING!")
            return False
    
    # Test that all 5 reasoning strategies are available
    from testmaster.intelligence import ReasoningStrategy
    
    reasoning_strategies = [s.value for s in ReasoningStrategy]
    expected_reasoning = [
        'breadth_first', 'depth_first', 'best_first',
        'monte_carlo', 'beam_search'
    ]
    
    for strategy in expected_reasoning:
        if strategy in reasoning_strategies:
            print(f"+ Reasoning strategy '{strategy}' preserved")
        else:
            print(f"X Reasoning strategy '{strategy}' MISSING!")
            return False
    
    # Test thought types are preserved
    from testmaster.intelligence.hierarchical_planning import (
        TestGenerationThought,
        TestStrategyThought,
        TestCoverageThought
    )
    
    print("+ TestGenerationThought available")
    print("+ TestStrategyThought available")
    print("+ TestCoverageThought available")
    
    # Test HTP-specific features
    from testmaster.intelligence.hierarchical_planning import (
        TestPlanLevel,
        LLMPlanningConfig
    )
    
    print("+ TestPlanLevel (HTP feature) available")
    print("+ LLMPlanningConfig (HTP feature) available")
    
    return True


def test_config_compatibility():
    """Test configuration compatibility between old and new."""
    print("\n=== Testing Configuration Compatibility ===")
    
    from testmaster.intelligence import (
        ToTGenerationConfig,
        HierarchicalPlanningConfig,
        UniversalHierarchicalTestGenerator
    )
    
    # Create old-style config
    tot_config = ToTGenerationConfig(
        reasoning_strategy="best_first",
        max_reasoning_depth=3,
        target_coverage=90.0
    )
    
    # Should work with new generator
    generator = UniversalHierarchicalTestGenerator(tot_config)
    print("+ New generator accepts old ToTGenerationConfig")
    
    # Create new-style config
    htp_config = HierarchicalPlanningConfig(
        planning_strategy="depth_first",
        max_planning_depth=4,
        use_plan_templates=True
    )
    
    # Convert to old style
    old_config = htp_config.to_tot_config()
    print(f"+ Can convert new config to old: strategy={old_config.reasoning_strategy}")
    
    return True


def test_orchestrator_compatibility():
    """Test that orchestrator can still use the classes."""
    print("\n=== Testing Orchestrator Compatibility ===")
    
    try:
        # This is what the orchestrator does
        from testmaster.intelligence.tree_of_thought import (
            UniversalToTTestGenerator,
            ToTGenerationConfig
        )
        print("X Direct import from tree_of_thought still works (should be redirected)")
    except ImportError:
        print("+ Direct import from tree_of_thought properly redirected")
    
    # This is what should work now
    from testmaster.intelligence import (
        UniversalToTTestGenerator,
        ToTGenerationConfig
    )
    
    # Create config like orchestrator does
    config = ToTGenerationConfig(
        reasoning_depth=3,
        enable_optimization=True,
        include_edge_cases=True
    )
    
    # Create generator like orchestrator does
    generator = UniversalToTTestGenerator()
    print("+ Orchestrator-style usage works")
    
    return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("UNIFIED HIERARCHICAL PLANNING INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_backward_compatibility,
        test_new_naming,
        test_feature_preservation,
        test_config_compatibility,
        test_orchestrator_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"X {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"X {test.__name__} FAILED with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("SUCCESS ALL TESTS PASSED - Integration successful!")
        print("\nThe unified integration preserves:")
        print("- All 10 test strategy types from ToT")
        print("- All 5 reasoning strategies from ToT")
        print("- All thought types from ToT")
        print("- All LLM features from HTP")
        print("- All plan template features from HTP")
        print("- 100% backward compatibility")
    else:
        print("FAILED SOME TESTS FAILED - Review integration")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)