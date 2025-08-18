"""
Comprehensive test suite to run BEFORE refactoring Tree-of-Thought names.
This ensures we don't break anything during the renaming process.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_orchestrator_functionality():
    """Test that the orchestrator works with current names."""
    print("\n=== Testing Orchestrator ===")
    try:
        from testmaster.orchestration.universal_orchestrator import (
            UniversalTestOrchestrator,
            OrchestrationConfig,
            OrchestrationMode
        )
        
        # Create config with ToT reasoning
        config = OrchestrationConfig(
            mode=OrchestrationMode.INTELLIGENT,
            enable_tot_reasoning=True,
            max_workers=1
        )
        
        # The orchestrator should initialize without errors
        orchestrator = UniversalTestOrchestrator(config)
        
        # Check that ToT generator was created
        assert orchestrator.tot_generator is not None, "ToT generator not initialized"
        
        print("+ Orchestrator initialization: PASS")
        return True
    except Exception as e:
        print(f"X Orchestrator test FAILED: {e}")
        traceback.print_exc()
        return False


def test_main_module():
    """Test main.py functionality."""
    print("\n=== Testing Main Module ===")
    try:
        from testmaster.main import (
            create_orchestration_config,
            parse_arguments
        )
        
        # Test that we can create a config
        class Args:
            mode = 'intelligent'
            target = './test'
            output = './output'
            enable_tot = True
            enable_optimization = True
            enable_security = False
            enable_compliance = False
            compliance_standards = []
            frameworks = []
            output_formats = ['python']
            max_workers = 2
            min_quality = 0.8
            min_coverage = 0.85
            no_docs = False
            no_metrics = False
        
        config = create_orchestration_config(Args())
        assert config.enable_tot_reasoning == True, "ToT reasoning not enabled"
        
        print("+ Main module config creation: PASS")
        return True
    except Exception as e:
        print(f"X Main module test FAILED: {e}")
        traceback.print_exc()
        return False


def test_intelligence_imports():
    """Test that all intelligence imports work."""
    print("\n=== Testing Intelligence Imports ===")
    try:
        # Test old names (backward compatibility)
        from testmaster.intelligence import (
            UniversalToTTestGenerator,
            ToTGenerationConfig,
            ToTGenerationResult,
            TreeOfThoughtReasoner,
            ReasoningStrategy
        )
        
        # Test new names
        from testmaster.intelligence import (
            UniversalHierarchicalTestGenerator,
            HierarchicalPlanningConfig,
            HierarchicalPlanningResult
        )
        
        # Verify they're aliases
        assert UniversalToTTestGenerator == UniversalHierarchicalTestGenerator, "Not properly aliased"
        
        print("+ Intelligence imports: PASS")
        return True
    except Exception as e:
        print(f"X Intelligence imports FAILED: {e}")
        traceback.print_exc()
        return False


def test_config_creation():
    """Test configuration creation with various parameters."""
    print("\n=== Testing Configuration Creation ===")
    try:
        from testmaster.intelligence import (
            ToTGenerationConfig,
            HierarchicalPlanningConfig,
            ReasoningStrategy
        )
        
        # Test with old-style config
        old_config = ToTGenerationConfig(
            reasoning_strategy=ReasoningStrategy.BEST_FIRST,
            max_reasoning_depth=5,
            target_coverage=85.0
        )
        assert old_config.planning_strategy == ReasoningStrategy.BEST_FIRST
        
        # Test with new-style config
        new_config = HierarchicalPlanningConfig(
            planning_strategy=ReasoningStrategy.MONTE_CARLO,
            max_planning_depth=3,
            use_plan_templates=True
        )
        assert new_config.planning_strategy == ReasoningStrategy.MONTE_CARLO
        
        # Test config conversion
        tot_config = new_config.to_tot_config()
        assert tot_config.reasoning_strategy == ReasoningStrategy.MONTE_CARLO
        
        print("+ Configuration creation: PASS")
        return True
    except Exception as e:
        print(f"X Configuration test FAILED: {e}")
        traceback.print_exc()
        return False


def test_tot_generator():
    """Test ToT generator creation and basic functionality."""
    print("\n=== Testing ToT Generator ===")
    try:
        from testmaster.intelligence import (
            UniversalToTTestGenerator,
            ToTGenerationConfig,
            ReasoningStrategy
        )
        
        config = ToTGenerationConfig(
            reasoning_strategy=ReasoningStrategy.BEAM_SEARCH,
            beam_width=3,
            max_reasoning_depth=4
        )
        
        generator = UniversalToTTestGenerator(config)
        
        # Verify internal ToT generator exists
        assert generator._tot_generator is not None, "Internal ToT generator missing"
        
        print("+ ToT generator creation: PASS")
        return True
    except Exception as e:
        print(f"X ToT generator test FAILED: {e}")
        traceback.print_exc()
        return False


def test_reasoning_components():
    """Test core reasoning components."""
    print("\n=== Testing Reasoning Components ===")
    try:
        from testmaster.intelligence.hierarchical_planning import (
            ThoughtNode,
            ThoughtTree,
            TreeOfThoughtReasoner,
            SimpleThoughtGenerator,
            SimpleThoughtEvaluator,
            ReasoningStrategy,
            EvaluationCriteria
        )
        
        # Create thought node
        node = ThoughtNode(
            id="test",
            content={"test": "data"},
            depth=0
        )
        
        # Create thought tree
        tree = ThoughtTree(root=node)
        
        # Create reasoner
        generator = SimpleThoughtGenerator()
        evaluator = SimpleThoughtEvaluator()
        reasoner = TreeOfThoughtReasoner(
            thought_generator=generator,
            thought_evaluator=evaluator,
            strategy=ReasoningStrategy.BEST_FIRST
        )
        
        # Add criterion
        criterion = EvaluationCriteria(
            name="test",
            weight=1.0
        )
        reasoner.add_criterion(criterion)
        
        print("+ Reasoning components: PASS")
        return True
    except Exception as e:
        print(f"X Reasoning components test FAILED: {e}")
        traceback.print_exc()
        return False


def test_test_strategies():
    """Test that all test strategies are available."""
    print("\n=== Testing Test Strategies ===")
    try:
        from testmaster.intelligence.hierarchical_planning import TestStrategyType
        
        strategies = [s.value for s in TestStrategyType]
        expected = [
            'happy_path', 'edge_cases', 'error_handling', 'performance',
            'security', 'integration', 'regression', 'property_based',
            'mutation', 'fuzzing'
        ]
        
        for strategy in expected:
            assert strategy in strategies, f"Missing strategy: {strategy}"
        
        print(f"+ All {len(expected)} test strategies available: PASS")
        return True
    except Exception as e:
        print(f"X Test strategies test FAILED: {e}")
        traceback.print_exc()
        return False


def test_orchestration_modes():
    """Test all orchestration modes."""
    print("\n=== Testing Orchestration Modes ===")
    try:
        from testmaster.orchestration import OrchestrationMode
        
        modes = [
            OrchestrationMode.STANDARD,
            OrchestrationMode.INTELLIGENT,
            OrchestrationMode.SECURITY_FOCUSED,
            OrchestrationMode.COMPLIANCE,
            OrchestrationMode.COMPREHENSIVE,
            OrchestrationMode.RAPID,
            OrchestrationMode.ENTERPRISE
        ]
        
        for mode in modes:
            assert mode.value is not None, f"Mode {mode} has no value"
        
        print(f"+ All {len(modes)} orchestration modes available: PASS")
        return True
    except Exception as e:
        print(f"X Orchestration modes test FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all pre-refactor tests."""
    print("=" * 70)
    print("PRE-REFACTOR TEST SUITE")
    print("Running comprehensive tests BEFORE Tree-of-Thought renaming")
    print("=" * 70)
    
    tests = [
        test_orchestrator_functionality,
        test_main_module,
        test_intelligence_imports,
        test_config_creation,
        test_tot_generator,
        test_reasoning_components,
        test_test_strategies,
        test_orchestration_modes
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
        print("\n✅ ALL TESTS PASSED - Safe to proceed with refactoring")
        print("\nCurrent working features:")
        print("- Orchestrator with ToT reasoning")
        print("- Main module configuration")
        print("- Both old and new import names")
        print("- Config creation and conversion")
        print("- All reasoning components")
        print("- All test strategies")
        print("- All orchestration modes")
    else:
        print("\n❌ SOME TESTS FAILED - Fix issues before refactoring")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)