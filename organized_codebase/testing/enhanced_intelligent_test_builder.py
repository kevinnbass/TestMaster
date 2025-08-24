#!/usr/bin/env python3
"""
Enhanced Intelligent Test Builder - Integrates classical analysis with test generation.
Uses complexity, dependency, and architecture analysis to create more intelligent tests.
"""

import os
import sys
import json
import time
import ast
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import existing test builder
from intelligent_test_builder import IntelligentTestBuilder

# Import classical analysis components
from testmaster.intelligence.classical.complexity_analyzer import ComplexityAnalyzer
from testmaster.intelligence.classical.dependency_analyzer import DependencyAnalyzer
from testmaster.intelligence.classical.architecture_analyzer import ArchitectureAnalyzer

# Import our new test components
from test_complexity_prioritizer import TestComplexityPrioritizer
from test_dependency_orderer import TestDependencyOrderer
from risk_based_test_targeter import RiskBasedTestTargeter


@dataclass
class EnhancedTestContext:
    """Enhanced context for test generation with classical analysis."""
    module_path: str
    module_name: str
    complexity_analysis: Dict[str, Any] = field(default_factory=dict)
    dependency_analysis: Dict[str, Any] = field(default_factory=dict)
    architecture_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_profile: Dict[str, Any] = field(default_factory=dict)
    test_priority: float = 0.5
    recommended_test_types: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    critical_paths: List[str] = field(default_factory=list)


@dataclass
class TestGenerationPlan:
    """Plan for generating tests based on analysis."""
    test_file_path: str
    module_path: str
    test_types: List[str]
    focus_areas: List[str]
    estimated_tests: int
    priority: float
    complexity_level: str
    dependencies_to_mock: List[str]
    dependencies_to_integrate: List[str]


class EnhancedIntelligentTestBuilder(IntelligentTestBuilder):
    """Enhanced test builder that uses classical analysis for smarter test generation."""
    
    def __init__(self, model: str = None, api_key: Optional[str] = None, project_root: str = '.'):
        """Initialize enhanced test builder with classical analyzers."""
        super().__init__(model=model, api_key=api_key)
        
        self.project_root = Path(project_root).resolve()
        
        # Initialize classical analyzers
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.architecture_analyzer = ArchitectureAnalyzer()
        
        # Initialize test optimization components
        self.test_prioritizer = TestComplexityPrioritizer(project_root)
        self.test_orderer = TestDependencyOrderer(project_root)
        self.risk_targeter = RiskBasedTestTargeter(project_root)
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, EnhancedTestContext] = {}
        
        # Configuration
        self.config = {
            'max_complexity_for_unit_tests': 10,
            'min_complexity_for_integration_tests': 15,
            'high_dependency_threshold': 5,
            'critical_module_patterns': ['auth', 'security', 'payment', 'core', 'api'],
            'max_parallel_analysis': 4
        }
        
        # Threading
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config['max_parallel_analysis'])
        
        print(f"Initialized Enhanced Intelligent Test Builder with classical analysis")
    
    def analyze_module_enhanced(self, module_path: Path) -> EnhancedTestContext:
        """Analyze module with both AI and classical analysis."""
        module_path_str = str(module_path)
        
        # Check cache
        if module_path_str in self._analysis_cache:
            return self._analysis_cache[module_path_str]
        
        # Create context
        context = EnhancedTestContext(
            module_path=module_path_str,
            module_name=module_path.stem
        )
        
        # Run analyses in parallel
        with self._executor as executor:
            # Submit analysis tasks
            futures = {
                'complexity': executor.submit(self.complexity_analyzer.analyze, module_path_str),
                'dependency': executor.submit(self.dependency_analyzer.analyze, module_path_str),
                'architecture': executor.submit(self.architecture_analyzer.analyze, module_path_str),
                'risk': executor.submit(self.risk_targeter.analyze_risk, module_path_str),
                'ai': executor.submit(super().analyze_module, module_path)
            }
            
            # Collect results
            try:
                context.complexity_analysis = futures['complexity'].result(timeout=30)
            except Exception as e:
                print(f"Complexity analysis failed: {e}")
                context.complexity_analysis = {}
            
            try:
                context.dependency_analysis = futures['dependency'].result(timeout=30)
            except Exception as e:
                print(f"Dependency analysis failed: {e}")
                context.dependency_analysis = {}
            
            try:
                context.architecture_analysis = futures['architecture'].result(timeout=30)
            except Exception as e:
                print(f"Architecture analysis failed: {e}")
                context.architecture_analysis = {}
            
            try:
                risk_result = futures['risk'].result(timeout=30)
                context.risk_profile = risk_result.to_dict() if hasattr(risk_result, 'to_dict') else {}
            except Exception as e:
                print(f"Risk analysis failed: {e}")
                context.risk_profile = {}
            
            try:
                ai_analysis = futures['ai'].result(timeout=60)
                # Merge AI analysis into context
                if ai_analysis:
                    context.edge_cases = ai_analysis.get('edge_cases', [])
                    context.critical_paths = ai_analysis.get('business_logic', '').split('\n')[:5]
            except Exception as e:
                print(f"AI analysis failed: {e}")
        
        # Determine test priority based on analyses
        context.test_priority = self._calculate_test_priority(context)
        
        # Recommend test types based on analysis
        context.recommended_test_types = self._recommend_test_types(context)
        
        # Cache result
        with self._lock:
            self._analysis_cache[module_path_str] = context
        
        return context
    
    def _calculate_test_priority(self, context: EnhancedTestContext) -> float:
        """Calculate test priority based on multiple factors."""
        priority = 0.5  # Base priority
        
        # Factor 1: Complexity
        complexity = context.complexity_analysis.get('total_complexity', 0)
        if complexity > 20:
            priority += 0.2
        elif complexity > 10:
            priority += 0.1
        
        # Factor 2: Dependencies
        dependencies = len(context.dependency_analysis.get('dependencies', []))
        if dependencies > 10:
            priority += 0.15
        elif dependencies > 5:
            priority += 0.1
        
        # Factor 3: Risk level
        risk_score = context.risk_profile.get('risk_score', 0)
        priority += risk_score * 0.2
        
        # Factor 4: Critical module check
        module_name = context.module_name.lower()
        if any(pattern in module_name for pattern in self.config['critical_module_patterns']):
            priority += 0.15
        
        return min(1.0, priority)
    
    def _recommend_test_types(self, context: EnhancedTestContext) -> List[str]:
        """Recommend test types based on module characteristics."""
        test_types = ['unit']  # Always include unit tests
        
        complexity = context.complexity_analysis.get('total_complexity', 0)
        dependencies = len(context.dependency_analysis.get('dependencies', []))
        risk_level = context.risk_profile.get('risk_level', 'low')
        
        # Add integration tests for complex modules
        if complexity >= self.config['min_complexity_for_integration_tests']:
            test_types.append('integration')
        
        # Add dependency tests for highly coupled modules
        if dependencies >= self.config['high_dependency_threshold']:
            test_types.append('dependency')
        
        # Add security tests for high-risk modules
        if risk_level in ['high', 'critical']:
            test_types.append('security')
        
        # Add performance tests for complex algorithms
        if complexity > 30:
            test_types.append('performance')
        
        # Add edge case tests if many were identified
        if len(context.edge_cases) > 5:
            test_types.append('edge_cases')
        
        return test_types
    
    def create_test_generation_plan(self, module_path: Path) -> TestGenerationPlan:
        """Create a detailed plan for test generation."""
        # Get enhanced analysis
        context = self.analyze_module_enhanced(module_path)
        
        # Determine test file path
        test_dir = self.project_root / 'tests'
        test_file_path = test_dir / f"test_{module_path.stem}.py"
        
        # Identify focus areas based on complexity
        focus_areas = []
        if context.complexity_analysis:
            complex_functions = context.complexity_analysis.get('complex_functions', [])
            focus_areas.extend([f['name'] for f in complex_functions[:5]])
        
        # Determine what to mock vs integrate
        dependencies = context.dependency_analysis.get('dependencies', [])
        external_deps = []
        internal_deps = []
        
        for dep in dependencies:
            if isinstance(dep, dict):
                module = dep.get('module', dep.get('name', ''))
            else:
                module = str(dep)
            
            if self._is_external_dependency(module):
                external_deps.append(module)
            else:
                internal_deps.append(module)
        
        # Estimate number of tests needed
        estimated_tests = self._estimate_test_count(context)
        
        # Determine complexity level
        complexity = context.complexity_analysis.get('total_complexity', 0)
        if complexity < 10:
            complexity_level = 'simple'
        elif complexity < 25:
            complexity_level = 'moderate'
        else:
            complexity_level = 'complex'
        
        return TestGenerationPlan(
            test_file_path=str(test_file_path),
            module_path=str(module_path),
            test_types=context.recommended_test_types,
            focus_areas=focus_areas,
            estimated_tests=estimated_tests,
            priority=context.test_priority,
            complexity_level=complexity_level,
            dependencies_to_mock=external_deps[:5],  # Limit to top 5
            dependencies_to_integrate=internal_deps[:3]  # Limit to top 3
        )
    
    def generate_enhanced_tests(self, module_path: Path, output_path: Optional[Path] = None) -> str:
        """Generate enhanced tests using classical analysis insights."""
        # Create test generation plan
        plan = self.create_test_generation_plan(module_path)
        
        # Get enhanced context
        context = self.analyze_module_enhanced(module_path)
        
        # Read module content
        content = module_path.read_text(encoding='utf-8')
        
        # Create enhanced prompt with classical analysis insights
        prompt = self._create_enhanced_prompt(content, context, plan)
        
        # Generate tests using AI with enhanced context
        try:
            response = self.provider(prompt, temperature=0.2, max_output_tokens=4000)
            
            # Extract test code
            if isinstance(response, list):
                test_code = response[0] if response else ""
            else:
                test_code = str(response)
            
            # Clean up response
            test_code = self._clean_test_code(test_code)
            
            # Add metadata comments
            test_code = self._add_test_metadata(test_code, context, plan)
            
            # Save if output path provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(test_code, encoding='utf-8')
                print(f"Enhanced tests saved to {output_path}")
            
            return test_code
            
        except Exception as e:
            print(f"Error generating enhanced tests: {e}")
            return self._generate_fallback_tests(module_path, context, plan)
    
    def _create_enhanced_prompt(self, content: str, context: EnhancedTestContext, plan: TestGenerationPlan) -> str:
        """Create an enhanced prompt with classical analysis insights."""
        prompt = f"""Generate comprehensive Python tests for this module using classical analysis insights.

MODULE: {context.module_name}
COMPLEXITY: {plan.complexity_level} (score: {context.complexity_analysis.get('total_complexity', 0)})
PRIORITY: {plan.priority:.2f}
RISK LEVEL: {context.risk_profile.get('risk_level', 'unknown')}

RECOMMENDED TEST TYPES: {', '.join(plan.test_types)}
FOCUS AREAS: {', '.join(plan.focus_areas) if plan.focus_areas else 'General coverage'}

KEY INSIGHTS FROM ANALYSIS:
1. Complexity Analysis:
   - Cyclomatic Complexity: {context.complexity_analysis.get('total_complexity', 0)}
   - Cognitive Complexity: {context.complexity_analysis.get('cognitive_complexity', 0)}
   - Maintainability Index: {context.complexity_analysis.get('maintainability_index', 100):.1f}

2. Dependency Analysis:
   - Total Dependencies: {len(context.dependency_analysis.get('dependencies', []))}
   - External to Mock: {', '.join(plan.dependencies_to_mock[:3]) if plan.dependencies_to_mock else 'None'}
   - Internal to Integrate: {', '.join(plan.dependencies_to_integrate[:3]) if plan.dependencies_to_integrate else 'None'}

3. Risk Factors:
"""
        
        # Add risk factors
        risk_factors = context.risk_profile.get('risk_factors', [])
        for i, factor in enumerate(risk_factors[:5], 1):
            if isinstance(factor, dict):
                prompt += f"   - {factor.get('description', 'Risk factor')}\n"
        
        prompt += f"""
4. Edge Cases to Test:
"""
        for i, edge_case in enumerate(context.edge_cases[:5], 1):
            prompt += f"   - {edge_case}\n"
        
        prompt += f"""
5. Critical Paths:
"""
        for i, path in enumerate(context.critical_paths[:5], 1):
            prompt += f"   - {path}\n"
        
        prompt += f"""

MODULE CODE:
{content[:6000]}  # Limit for prompt size

REQUIREMENTS:
1. Generate tests for all recommended test types: {', '.join(plan.test_types)}
2. Focus especially on: {', '.join(plan.focus_areas[:3]) if plan.focus_areas else 'core functionality'}
3. Include at least {plan.estimated_tests} test cases
4. Mock these external dependencies: {', '.join(plan.dependencies_to_mock[:3]) if plan.dependencies_to_mock else 'as needed'}
5. Use real imports for internal dependencies: {', '.join(plan.dependencies_to_integrate[:3]) if plan.dependencies_to_integrate else 'all internal'}
6. Test all identified edge cases
7. Include tests for error scenarios
8. Add performance tests if complexity > 30 (current: {context.complexity_analysis.get('total_complexity', 0)})

Generate complete, runnable test code with proper imports, fixtures, and assertions.
Use pytest framework with clear test names and docstrings."""
        
        return prompt
    
    def _clean_test_code(self, test_code: str) -> str:
        """Clean up generated test code."""
        # Remove markdown code blocks
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0]
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0]
        
        return test_code.strip()
    
    def _add_test_metadata(self, test_code: str, context: EnhancedTestContext, plan: TestGenerationPlan) -> str:
        """Add metadata comments to test code."""
        metadata = f"""#!/usr/bin/env python3
\"\"\"
Enhanced tests for {context.module_name}
Generated with classical analysis insights

Module Complexity: {plan.complexity_level}
Test Priority: {plan.priority:.2f}
Risk Level: {context.risk_profile.get('risk_level', 'unknown')}
Test Types: {', '.join(plan.test_types)}

Classical Analysis Metrics:
- Cyclomatic Complexity: {context.complexity_analysis.get('total_complexity', 0)}
- Dependencies: {len(context.dependency_analysis.get('dependencies', []))}
- Risk Score: {context.risk_profile.get('risk_score', 0):.2f}
\"\"\"

"""
        return metadata + test_code
    
    def _generate_fallback_tests(self, module_path: Path, context: EnhancedTestContext, plan: TestGenerationPlan) -> str:
        """Generate basic fallback tests if AI generation fails."""
        return f"""#!/usr/bin/env python3
\"\"\"
Fallback tests for {context.module_name}
\"\"\"

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import module to test
from {module_path.stem} import *

class Test{context.module_name.title().replace('_', '')}:
    \"\"\"Test suite for {context.module_name}\"\"\"
    
    def test_module_imports(self):
        \"\"\"Test that module imports successfully.\"\"\"
        assert True  # Module imported without errors
    
    # TODO: Add tests based on analysis:
    # - Complexity: {context.complexity_analysis.get('total_complexity', 0)}
    # - Dependencies: {len(context.dependency_analysis.get('dependencies', []))}
    # - Risk Level: {context.risk_profile.get('risk_level', 'unknown')}
    # - Recommended test types: {', '.join(plan.test_types)}
"""
    
    def _estimate_test_count(self, context: EnhancedTestContext) -> int:
        """Estimate number of tests needed based on complexity."""
        base_tests = 5
        
        # Add tests based on complexity
        complexity = context.complexity_analysis.get('total_complexity', 0)
        complexity_tests = min(10, complexity // 5)
        
        # Add tests for edge cases
        edge_case_tests = len(context.edge_cases)
        
        # Add tests based on risk
        risk_score = context.risk_profile.get('risk_score', 0)
        risk_tests = int(risk_score * 10)
        
        return base_tests + complexity_tests + edge_case_tests + risk_tests
    
    def _is_external_dependency(self, module: str) -> bool:
        """Check if a dependency is external."""
        external_indicators = [
            'unittest', 'pytest', 'mock', 'numpy', 'pandas', 'requests',
            'flask', 'django', 'sqlalchemy', 'asyncio'
        ]
        return any(ind in module.lower() for ind in external_indicators)
    
    def batch_generate_tests(self, 
                            module_paths: List[Path],
                            prioritize: bool = True,
                            max_parallel: int = 4) -> Dict[str, str]:
        """Generate tests for multiple modules with prioritization."""
        results = {}
        
        # Prioritize if requested
        if prioritize:
            # Analyze all modules first
            contexts = {}
            for path in module_paths:
                contexts[str(path)] = self.analyze_module_enhanced(path)
            
            # Sort by priority
            module_paths = sorted(
                module_paths,
                key=lambda p: contexts[str(p)].test_priority,
                reverse=True
            )
            
            print(f"Prioritized {len(module_paths)} modules for test generation")
        
        # Generate tests
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for path in module_paths:
                output_path = self.project_root / 'tests' / f"test_{path.stem}.py"
                future = executor.submit(self.generate_enhanced_tests, path, output_path)
                futures[str(path)] = future
            
            for path_str, future in futures.items():
                try:
                    test_code = future.result(timeout=120)
                    results[path_str] = test_code
                    print(f"Generated tests for {Path(path_str).name}")
                except Exception as e:
                    print(f"Failed to generate tests for {Path(path_str).name}: {e}")
                    results[path_str] = None
        
        return results


def main():
    """Main function to demonstrate enhanced test generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced tests with classical analysis')
    parser.add_argument('--module', type=str, help='Module to generate tests for')
    parser.add_argument('--directory', type=str, help='Directory of modules to test')
    parser.add_argument('--output', type=str, help='Output directory for tests')
    parser.add_argument('--prioritize', action='store_true', help='Prioritize test generation')
    parser.add_argument('--max-parallel', type=int, default=4, help='Max parallel generations')
    
    args = parser.parse_args()
    
    # Initialize enhanced test builder
    builder = EnhancedIntelligentTestBuilder()
    
    if args.module:
        # Single module
        module_path = Path(args.module)
        if not module_path.exists():
            print(f"Module not found: {module_path}")
            return 1
        
        print(f"\nAnalyzing {module_path.name} with classical analysis...")
        context = builder.analyze_module_enhanced(module_path)
        
        print(f"\nAnalysis Results:")
        print(f"  Complexity: {context.complexity_analysis.get('total_complexity', 0)}")
        print(f"  Dependencies: {len(context.dependency_analysis.get('dependencies', []))}")
        print(f"  Risk Level: {context.risk_profile.get('risk_level', 'unknown')}")
        print(f"  Test Priority: {context.test_priority:.2f}")
        print(f"  Recommended Tests: {', '.join(context.recommended_test_types)}")
        
        print(f"\nCreating test generation plan...")
        plan = builder.create_test_generation_plan(module_path)
        
        print(f"\nTest Generation Plan:")
        print(f"  Output: {plan.test_file_path}")
        print(f"  Test Types: {', '.join(plan.test_types)}")
        print(f"  Focus Areas: {', '.join(plan.focus_areas[:3]) if plan.focus_areas else 'General'}")
        print(f"  Estimated Tests: {plan.estimated_tests}")
        print(f"  Complexity Level: {plan.complexity_level}")
        
        print(f"\nGenerating enhanced tests...")
        output_path = Path(args.output) / f"test_{module_path.stem}.py" if args.output else None
        test_code = builder.generate_enhanced_tests(module_path, output_path)
        
        if test_code:
            print(f"\nTests generated successfully!")
            if not output_path:
                print(f"\nGenerated test preview:")
                print(test_code[:500])
        
    elif args.directory:
        # Batch generation
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return 1
        
        # Find Python modules
        modules = list(directory.glob('*.py'))
        print(f"\nFound {len(modules)} modules to test")
        
        # Generate tests
        results = builder.batch_generate_tests(
            modules,
            prioritize=args.prioritize,
            max_parallel=args.max_parallel
        )
        
        # Summary
        successful = sum(1 for r in results.values() if r)
        print(f"\nGeneration complete: {successful}/{len(modules)} successful")
    
    else:
        print("Please specify --module or --directory")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())