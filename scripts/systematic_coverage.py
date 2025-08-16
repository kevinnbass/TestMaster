#!/usr/bin/env python3
"""
Systematic Module Coverage
==========================

Analyze and generate tests for each module systematically.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ModuleCoverage:
    """Coverage information for a module."""
    module_path: str
    covered_lines: Set[int]
    missing_lines: Set[int]
    coverage_percent: float
    functions: List[str]
    classes: List[str]
    
class SystematicCoverageAnalyzer:
    """Analyze and improve coverage systematically."""
    
    def __init__(self):
        self.src_dir = Path("src_new")
        self.test_dir = Path("tests_new")
        self.coverage_data = {}
        self.module_analysis = {}
        
    def analyze_module_structure(self, module_path: Path) -> Dict:
        """Analyze a Python module's structure."""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            structure = {
                'classes': [],
                'functions': [],
                'methods': defaultdict(list),
                'imports': [],
                'lines': len(open(module_path, 'r').readlines())
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure['classes'].append(node.name)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            structure['methods'][node.name].append(item.name)
                elif isinstance(node, ast.FunctionDef):
                    if node.col_offset == 0:  # Top-level function
                        structure['functions'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        structure['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        structure['imports'].append(node.module)
            
            return structure
        except Exception as e:
            print(f"Error analyzing {module_path}: {e}")
            return {}
    
    def get_current_coverage(self) -> Dict[str, ModuleCoverage]:
        """Get current coverage data for all modules."""
        print("Analyzing current coverage...")
        
        # Run coverage
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', 
             '--cov=src_new', '--cov-report=json',
             '--tb=no', '-q', '--disable-warnings',
             '--timeout=60'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        coverage_modules = {}
        
        # Load coverage data
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                data = json.load(f)
                
            for file_path, file_data in data.get('files', {}).items():
                if 'src_new' in file_path:
                    module_name = Path(file_path).stem
                    executed = set(file_data.get('executed_lines', []))
                    missing = set(file_data.get('missing_lines', []))
                    total = len(executed) + len(missing)
                    percent = (len(executed) / total * 100) if total > 0 else 0
                    
                    # Get module structure
                    structure = self.analyze_module_structure(Path(file_path))
                    
                    coverage_modules[module_name] = ModuleCoverage(
                        module_path=file_path,
                        covered_lines=executed,
                        missing_lines=missing,
                        coverage_percent=percent,
                        functions=structure.get('functions', []),
                        classes=structure.get('classes', [])
                    )
        
        return coverage_modules
    
    def prioritize_modules(self, coverage_data: Dict[str, ModuleCoverage]) -> List[Tuple[str, ModuleCoverage]]:
        """Prioritize modules for testing based on importance and coverage."""
        # Priority criteria:
        # 1. Core modules (application, domain, container, bootstrap)
        # 2. Low coverage percentage
        # 3. High number of uncovered lines
        
        priorities = []
        
        for module_name, coverage in coverage_data.items():
            priority_score = 0
            
            # Core modules get highest priority
            if module_name in ['application', 'domain', 'container', 'bootstrap']:
                priority_score += 1000
            elif module_name in ['interfaces', 'infrastructure']:
                priority_score += 500
            
            # Low coverage increases priority
            priority_score += (100 - coverage.coverage_percent) * 10
            
            # Many uncovered lines increases priority
            priority_score += len(coverage.missing_lines)
            
            priorities.append((priority_score, module_name, coverage))
        
        # Sort by priority (highest first)
        priorities.sort(reverse=True)
        
        return [(name, cov) for _, name, cov in priorities]
    
    def generate_test_template(self, module_name: str, coverage: ModuleCoverage) -> str:
        """Generate a test template for a module."""
        module_path = Path(coverage.module_path)
        structure = self.analyze_module_structure(module_path)
        
        # Generate test code
        test_code = f'''#!/usr/bin/env python3
"""
Comprehensive tests for {module_name} module.
Generated to achieve 100% coverage.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add src_new to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))

# Import the module to test
'''
        
        # Add imports based on module location
        rel_path = Path(coverage.module_path).relative_to(self.src_dir)
        import_path = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
        test_code += f"from {import_path} import *\n\n"
        
        # Generate test classes for each class in module
        for class_name in structure.get('classes', []):
            test_code += f'''
class Test{class_name}:
    """Test {class_name} class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dependency = Mock()
'''
            
            # Generate test methods for each method
            for method_name in structure.get('methods', {}).get(class_name, []):
                if method_name.startswith('_'):
                    continue  # Skip private methods for now
                    
                test_code += f'''
    def test_{method_name}(self):
        """Test {class_name}.{method_name} method."""
        # TODO: Implement test for {method_name}
        instance = {class_name}()
        # Add assertions here
        assert instance is not None
'''
                
                # Add async test if method is async
                if 'async' in method_name.lower():
                    test_code += f'''
    @pytest.mark.asyncio
    async def test_{method_name}_async(self):
        """Test async {class_name}.{method_name} method."""
        instance = {class_name}()
        # Add async assertions here
        assert instance is not None
'''
        
        # Generate tests for standalone functions
        for func_name in structure.get('functions', []):
            if func_name.startswith('_'):
                continue
                
            test_code += f'''
def test_{func_name}():
    """Test {func_name} function."""
    # TODO: Implement test for {func_name}
    # Add test implementation here
    pass
'''
        
        # Add coverage-focused tests for missing lines
        if coverage.missing_lines:
            test_code += '''
class TestCoverageGaps:
    """Tests specifically targeting uncovered lines."""
'''
            
            # Group missing lines into ranges
            missing_ranges = self._group_lines_into_ranges(coverage.missing_lines)
            
            for i, (start, end) in enumerate(missing_ranges[:10]):  # Limit to 10 ranges
                test_code += f'''
    def test_lines_{start}_to_{end}(self):
        """Test lines {start}-{end} for coverage."""
        # Target specific uncovered code paths
        # TODO: Add specific test for these lines
        pass
'''
        
        return test_code
    
    def _group_lines_into_ranges(self, lines: Set[int]) -> List[Tuple[int, int]]:
        """Group line numbers into ranges."""
        if not lines:
            return []
        
        sorted_lines = sorted(lines)
        ranges = []
        start = sorted_lines[0]
        end = sorted_lines[0]
        
        for line in sorted_lines[1:]:
            if line == end + 1:
                end = line
            else:
                ranges.append((start, end))
                start = end = line
        
        ranges.append((start, end))
        return ranges
    
    def generate_module_tests(self):
        """Generate tests for all modules systematically."""
        print("=" * 70)
        print("SYSTEMATIC MODULE COVERAGE GENERATION")
        print("=" * 70)
        
        # Get current coverage
        coverage_data = self.get_current_coverage()
        
        if not coverage_data:
            print("No coverage data available. Running basic analysis...")
            # Analyze all Python files
            for py_file in self.src_dir.rglob("*.py"):
                if '__pycache__' in str(py_file):
                    continue
                module_name = py_file.stem
                structure = self.analyze_module_structure(py_file)
                coverage_data[module_name] = ModuleCoverage(
                    module_path=str(py_file),
                    covered_lines=set(),
                    missing_lines=set(range(1, structure.get('lines', 100))),
                    coverage_percent=0,
                    functions=structure.get('functions', []),
                    classes=structure.get('classes', [])
                )
        
        # Prioritize modules
        prioritized = self.prioritize_modules(coverage_data)
        
        print(f"\nFound {len(prioritized)} modules to test")
        print("\nTop 10 Priority Modules:")
        for module_name, coverage in prioritized[:10]:
            print(f"  {module_name}: {coverage.coverage_percent:.1f}% coverage, "
                  f"{len(coverage.missing_lines)} lines missing")
        
        # Generate tests for top priority modules
        generated_count = 0
        for module_name, coverage in prioritized[:20]:  # Generate for top 20 modules
            if coverage.coverage_percent >= 95:
                continue  # Skip well-covered modules
            
            test_file = self.test_dir / f"test_{module_name}_systematic.py"
            
            # Don't overwrite existing systematic tests
            if test_file.exists():
                continue
            
            print(f"\nGenerating test for {module_name}...")
            test_code = self.generate_test_template(module_name, coverage)
            
            test_file.write_text(test_code, encoding='utf-8')
            generated_count += 1
            print(f"  Created: {test_file.name}")
        
        print(f"\n{generated_count} test files generated")
        return generated_count
    
    def run_and_measure_improvement(self):
        """Run tests and measure coverage improvement."""
        print("\n" + "=" * 70)
        print("MEASURING COVERAGE IMPROVEMENT")
        print("=" * 70)
        
        # Run coverage again
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', 
             '--cov=src_new', '--cov-report=term',
             '--tb=no', '-q', '--disable-warnings',
             '--timeout=120'],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # Extract coverage percentage from output
        for line in result.stdout.split('\n'):
            if 'TOTAL' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        coverage = float(parts[-1].rstrip('%'))
                        print(f"\nCurrent Coverage: {coverage}%")
                        return coverage
                    except:
                        pass
        
        return 0


def main():
    """Run systematic coverage generation."""
    analyzer = SystematicCoverageAnalyzer()
    
    # Generate tests
    generated = analyzer.generate_module_tests()
    
    if generated > 0:
        # Measure improvement
        coverage = analyzer.run_and_measure_improvement()
        
        print("\n" + "=" * 70)
        print("SYSTEMATIC COVERAGE COMPLETE")
        print("=" * 70)
        print(f"\nGenerated {generated} new test files")
        print(f"Current coverage: {coverage}%")
        print(f"Target: 100%")
        print(f"Gap remaining: {100 - coverage}%")
    else:
        print("\nNo new tests generated - checking existing coverage...")
        coverage = analyzer.run_and_measure_improvement()
        print(f"Current coverage: {coverage}%")
    
    print("\nNext steps:")
    print("1. Review and enhance generated test templates")
    print("2. Run AI-powered test generation for remaining gaps")
    print("3. Focus on branch coverage for complex functions")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())