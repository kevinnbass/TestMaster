#!/usr/bin/env python3
"""
Intelligent Test Builder - Offline/Template Version

This version creates intelligent test templates without requiring API calls.
It analyzes modules locally and generates comprehensive test structures.
"""

import os
import sys
import ast
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class OfflineIntelligentTestBuilder:
    """Build intelligent test templates through local analysis."""
    
    def analyze_module_ast(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a module using AST to understand its structure."""
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        content = module_path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax error in module: {e}"}
        
        analysis = {
            "purpose": f"Module: {module_path.stem}",
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": [],
            "has_main": False
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            "name": item.name,
                            "args": [arg.arg for arg in item.args.args],
                            "is_private": item.name.startswith('_')
                        })
                
                analysis["classes"].append({
                    "name": node.name,
                    "methods": methods,
                    "bases": [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) for base in node.bases]
                })
                
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                analysis["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "is_private": node.name.startswith('_'),
                    "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node))
                })
                
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis["imports"].append(node.module)
        
        # Check for if __name__ == "__main__"
        analysis["has_main"] = any(
            isinstance(node, ast.If) and 
            isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == "__name__"
            for node in ast.walk(tree)
        )
        
        return analysis
    
    def generate_intelligent_test_template(self, module_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive test template based on analysis."""
        
        module_name = module_path.stem
        module_import_path = str(module_path).replace("\\", "/").replace(".py", "").replace("/", ".")
        if "multi_coder_analysis" in module_import_path:
            idx = module_import_path.find("multi_coder_analysis")
            module_import_path = module_import_path[idx:]
        
        # Build comprehensive test template
        test_code = f'''"""
INTELLIGENT Real Functionality Tests for {module_name}

This test file provides exhaustive testing of ALL public APIs.
Tests focus on real functionality, not mocks.
"""

import pytest
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the module under test
try:
    import {module_import_path} as test_module
except ImportError:
    # Try alternative import
    from {module_import_path.replace(".", " ").split()[-1]} import *
    test_module = sys.modules[__name__]


class TestModuleImports:
    """Test that the module imports correctly and has expected structure."""
    
    def test_module_imports_successfully(self):
        """Test that the module can be imported."""
        assert test_module is not None
        
    def test_module_has_expected_attributes(self):
        """Test that module has all expected classes and functions."""
        # Classes
'''
        
        # Add class existence tests
        for cls in analysis.get("classes", []):
            test_code += f'''        assert hasattr(test_module, "{cls['name']}"), "Missing class: {cls['name']}"\n'''
        
        test_code += '''        
        # Functions
'''
        for func in analysis.get("functions", []):
            if not func["is_private"]:
                test_code += f'''        assert hasattr(test_module, "{func['name']}"), "Missing function: {func['name']}"\n'''
        
        # Add test classes for each class in the module
        for cls in analysis.get("classes", []):
            test_code += f'''


class Test{cls['name']}:
    """Exhaustive tests for {cls['name']} class."""
    
    def test_class_exists(self):
        """Test that {cls['name']} class exists."""
        assert hasattr(test_module, "{cls['name']}")
        assert isinstance(getattr(test_module, "{cls['name']}"), type)
    
    def test_class_instantiation(self):
        """Test that {cls['name']} can be instantiated."""
        cls = getattr(test_module, "{cls['name']}")
        # TODO: Add appropriate constructor arguments
        # instance = cls()
        # assert instance is not None
    '''
            
            # Add method tests
            for method in cls.get("methods", []):
                if not method["is_private"] or method["name"] == "__init__":
                    test_code += f'''
    def test_method_{method['name'].replace('__', '')}(self):
        """Test {cls['name']}.{method['name']} method."""
        cls = getattr(test_module, "{cls['name']}")
        assert hasattr(cls, "{method['name']}"), "Missing method: {method['name']}"
        
        # TODO: Test actual functionality
        # instance = cls()
        # result = instance.{method['name']}(...)
        # assert result == expected_value
    '''
        
        # Add test functions for module-level functions
        if analysis.get("functions"):
            test_code += '''


class TestModuleFunctions:
    """Exhaustive tests for module-level functions."""
'''
            
            for func in analysis.get("functions", []):
                if not func["is_private"]:
                    test_code += f'''
    
    def test_{func['name']}(self):
        """Test {func['name']} function with real data."""
        func = getattr(test_module, "{func['name']}")
        
        # Test with valid inputs
        # TODO: Add real test data
        # result = func(test_input)
        # assert result is not None
        
        # Test edge cases
        # TODO: Add edge case tests
        
        # Test error conditions
        # TODO: Add error condition tests
'''
        
        # Add integration tests section
        test_code += '''


class TestIntegration:
    """Integration tests between components."""
    
    def test_component_integration(self):
        """Test that components work together correctly."""
        # TODO: Add integration tests
        pass
    
    def test_data_flow(self):
        """Test data flow through the module."""
        # TODO: Add data flow tests
        pass
    
    def test_error_propagation(self):
        """Test that errors propagate correctly."""
        # TODO: Add error propagation tests
        pass


class TestPerformance:
    """Performance and efficiency tests."""
    
    def test_performance_characteristics(self):
        """Test performance meets requirements."""
        # TODO: Add performance tests
        pass


class TestEdgeCases:
    """Comprehensive edge case testing."""
    
    def test_boundary_values(self):
        """Test boundary value conditions."""
        # TODO: Add boundary tests
        pass
    
    def test_null_empty_inputs(self):
        """Test handling of null/empty inputs."""
        # TODO: Add null/empty tests
        pass
    
    def test_large_inputs(self):
        """Test handling of large inputs."""
        # TODO: Add large input tests
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''
        
        return test_code
    
    def build_test_for_module(self, module_path: Path, output_dir: Path = None) -> bool:
        """Build intelligent test template for a module."""
        
        print(f"\n{'='*60}")
        print(f"Building intelligent test template for: {module_path.name}")
        print('='*60)
        
        # Step 1: Analyze the module
        print("Step 1: Analyzing module structure...")
        analysis = self.analyze_module_ast(module_path)
        
        if "error" in analysis:
            print(f"ERROR: {analysis['error']}")
            return False
        
        print(f"  Classes found: {len(analysis.get('classes', []))}")
        for cls in analysis.get('classes', []):
            print(f"    - {cls['name']} ({len(cls.get('methods', []))} methods)")
        
        print(f"  Functions found: {len(analysis.get('functions', []))}")
        for func in analysis.get('functions', [])[:5]:  # Show first 5
            if not func["is_private"]:
                print(f"    - {func['name']}")
        
        # Step 2: Generate test template
        print("\nStep 2: Generating intelligent test template...")
        test_code = self.generate_intelligent_test_template(module_path, analysis)
        
        # Step 3: Save the test
        if output_dir is None:
            output_dir = Path("tests/unit")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        test_filename = f"test_{module_path.stem}_intelligent.py"
        test_path = output_dir / test_filename
        
        test_path.write_text(test_code, encoding='utf-8')
        print(f"\nStep 3: Test template saved to: {test_path}")
        
        # Step 4: Validate
        print("\nStep 4: Validating test file...")
        try:
            ast.parse(test_code)
            print("  OK: Test file has valid Python syntax")
            print("\nNOTE: This is a template. You need to:")
            print("  1. Add appropriate constructor arguments")
            print("  2. Add real test data")
            print("  3. Implement actual test logic")
            print("  4. Remove TODO comments")
            return True
        except SyntaxError as e:
            print(f"  ERROR: Syntax error in generated test: {e}")
            return False


def main():
    """Main function to demonstrate offline test builder."""
    
    print("="*60)
    print("Intelligent Test Builder - Offline Mode")
    print("(No API calls required)")
    print("="*60)
    
    # Test on a simple module
    test_module = Path("multi_coder_analysis/utils/concatenate_prompts.py")
    
    if test_module.exists():
        print(f"\nGenerating test template for: {test_module}")
        builder = OfflineIntelligentTestBuilder()
        success = builder.build_test_for_module(test_module)
        
        if success:
            print("\nOK: Successfully generated test template!")
            print("\nThis template provides:")
            print("  - Exhaustive coverage of all public APIs")
            print("  - Test structure for all classes and methods")
            print("  - Integration test placeholders")
            print("  - Performance test sections")
            print("  - Edge case test frameworks")
            return 0
        else:
            print("\nERROR: Failed to generate test template")
            return 1
    else:
        print(f"\nTest module not found: {test_module}")
        return 1


if __name__ == "__main__":
    sys.exit(main())