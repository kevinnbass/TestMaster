"""
Test the Tree-of-Thought Universal Test Generation System
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from testmaster.core.ast_abstraction import UniversalASTAbstractor as UniversalASTBuilder
from testmaster.intelligence import (
    UniversalToTTestGenerator,
    ToTGenerationConfig,
    ReasoningStrategy
)
from testmaster.core.framework_abstraction import FrameworkAdapterRegistry


def test_tot_generation():
    """Test Tree-of-Thought test generation."""
    print("\n" + "="*80)
    print("Testing Tree-of-Thought Universal Test Generation")
    print("="*80)
    
    # Build AST for TestMaster project
    print("\n1. Building Universal AST...")
    ast_builder = UniversalASTBuilder()
    project_path = Path(__file__).parent / "testmaster"
    universal_ast = ast_builder.create_project_ast(str(project_path))
    
    print(f"   [OK] AST built: {universal_ast.total_files} files, {universal_ast.total_functions} functions")
    
    # Configure ToT generation
    print("\n2. Configuring Tree-of-Thought reasoning...")
    config = ToTGenerationConfig(
        reasoning_strategy=ReasoningStrategy.BEAM_SEARCH,
        max_reasoning_depth=4,
        max_iterations=30,
        beam_width=3,
        target_coverage=80.0,
        prioritize_complex=True,
        prioritize_security=True
    )
    
    print(f"   [OK] Configuration set: {config.reasoning_strategy.value} strategy")
    
    # Generate tests using ToT
    print("\n3. Generating tests with Tree-of-Thought reasoning...")
    tot_generator = UniversalToTTestGenerator(config)
    result = tot_generator.generate_tests(universal_ast)
    
    print(f"   [OK] Tests generated: {result.test_suite.count_tests()} tests")
    
    # Convert to pytest format
    print("\n4. Converting to pytest format...")
    adapter = FrameworkAdapterRegistry.get_adapter("pytest")
    if adapter:
        pytest_code = adapter.convert_test_suite(result.test_suite)
        
        # Save to file
        output_path = Path("test_tot_output.py")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pytest_code)
        
        print(f"   [OK] Pytest code generated and saved to {output_path}")
        
        # Show first 50 lines
        lines = pytest_code.split('\n')[:50]
        print("\n   Preview (first 50 lines):")
        print("   " + "-"*60)
        for line in lines:
            print(f"   {line}")
        total_lines = len(pytest_code.split('\n'))
        if total_lines > 50:
            print(f"   ... ({total_lines - 50} more lines)")
    
    # Visualize thought tree
    print("\n5. Thought Tree Visualization:")
    print("   " + "-"*60)
    tree_viz = result.thought_tree.visualize(max_depth=3)
    for line in tree_viz.split('\n'):
        print(f"   {line}")
    
    # Show insights
    print("\n6. Generation Insights:")
    print("   " + "-"*60)
    for insight in result.key_insights:
        print(f"   * {insight}")
    
    print("\n7. Recommendations:")
    print("   " + "-"*60)
    for rec in result.recommended_improvements:
        print(f"   * {rec}")
    
    print("\n" + "="*80)
    print("[SUCCESS] Tree-of-Thought Test Generation Complete!")
    print("="*80)
    
    return result


def test_different_strategies():
    """Test different reasoning strategies."""
    print("\n" + "="*80)
    print("Testing Different Reasoning Strategies")
    print("="*80)
    
    # Build AST
    ast_builder = UniversalASTBuilder()
    project_path = Path(__file__).parent / "testmaster" / "core" / "language_detection"
    universal_ast = ast_builder.create_project_ast(str(project_path))
    
    strategies = [
        ReasoningStrategy.BREADTH_FIRST,
        ReasoningStrategy.DEPTH_FIRST,
        ReasoningStrategy.BEST_FIRST
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy...")
        
        config = ToTGenerationConfig(
            reasoning_strategy=strategy,
            max_reasoning_depth=3,
            max_iterations=20,
            beam_width=2
        )
        
        generator = UniversalToTTestGenerator(config)
        result = generator.generate_tests(universal_ast)
        
        results[strategy.value] = {
            'tests': result.test_suite.count_tests(),
            'assertions': result.test_suite.count_assertions(),
            'thoughts': result.total_thoughts_generated,
            'time': result.reasoning_time,
            'quality': result.test_quality_score
        }
    
    # Compare results
    print("\n" + "="*80)
    print("Strategy Comparison:")
    print("="*80)
    print(f"{'Strategy':<20} {'Tests':<10} {'Assertions':<12} {'Thoughts':<10} {'Time(s)':<10} {'Quality':<10}")
    print("-"*80)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<20} {metrics['tests']:<10} {metrics['assertions']:<12} "
              f"{metrics['thoughts']:<10} {metrics['time']:<10.2f} {metrics['quality']:<10.1f}")
    
    print("="*80)


if __name__ == "__main__":
    # Test basic ToT generation
    result = test_tot_generation()
    
    # Test different strategies
    print("\n" * 3)
    test_different_strategies()
    
    print("\n[SUCCESS] All Tree-of-Thought tests completed successfully!")