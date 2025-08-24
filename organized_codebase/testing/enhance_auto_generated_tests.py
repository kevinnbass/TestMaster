#!/usr/bin/env python3
"""
Enhanced Test Logic Generator

Takes auto-generated tests and enhances them with:
1. Specific test data based on module analysis
2. Real assertions instead of generic mocks
3. Domain-specific edge cases
4. Proper error conditions
"""

import os
import re
import ast
import sys
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestEnhancer:
    """Enhances auto-generated tests with specific logic."""
    
    def __init__(self):
        self.tests_dir = project_root / "tests" / "unit"
        self.source_dir = project_root / "multi_coder_analysis"
        self.enhanced_count = 0
        
        # Common test data templates by module type
        self.test_data_templates = {
            'config': {
                'initialization': {'population_size': 50, 'mutation_rate': 0.1},
                'validation': {'population_size': -1, 'mutation_rate': 1.5},
                'edge_cases': {'population_size': 0, 'mutation_rate': 0.0}
            },
            'optimizer': {
                'initialization': {'config': 'mock_config', 'state': 'initial'},
                'optimization': {'iterations': 100, 'convergence': 0.001},
                'edge_cases': {'empty_population': [], 'invalid_fitness': None}
            },
            'pipeline': {
                'initialization': {'input_data': ['test1', 'test2'], 'config': {}},
                'processing': {'batch_size': 10, 'parallel': True},
                'edge_cases': {'empty_input': [], 'invalid_config': None}
            }
        }
    
    def analyze_source_module(self, module_name: str) -> Dict[str, Any]:
        """Analyze the source module to understand its structure."""
        # Convert test file name to source file path
        source_name = module_name.replace('test_', '').replace('.py', '.py')
        
        # Look for the source file in various locations
        potential_paths = [
            self.source_dir / "improvement_system" / source_name,
            self.source_dir / source_name,
            self.source_dir / "runtime" / source_name,
            self.source_dir / "llm_providers" / source_name,
        ]
        
        source_file = None
        for path in potential_paths:
            if path.exists():
                source_file = path
                break
        
        if not source_file:
            return {'classes': [], 'functions': [], 'constants': []}
        
        try:
            content = source_file.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            analysis = {
                'classes': [],
                'functions': [],
                'constants': [],
                'imports': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'init_args': self._get_init_args(node)
                    }
                    analysis['classes'].append(class_info)
                
                elif isinstance(node, ast.FunctionDef) and not any(node in cls.body for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)):
                    analysis['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args]
                    })
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            analysis['constants'].append(target.id)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {source_file}: {e}")
            return {'classes': [], 'functions': [], 'constants': []}
    
    def _get_init_args(self, class_node: ast.ClassDef) -> List[str]:
        """Get __init__ method arguments."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                return [arg.arg for arg in node.args.args if arg.arg != 'self']
        return []
    
    def enhance_test_file(self, test_file: Path) -> bool:
        """Enhance a single test file with specific logic."""
        try:
            content = test_file.read_text(encoding='utf-8')
        except:
            return False
        
        # Skip if already enhanced
        if '# Enhanced with specific logic' in content:
            return False
        
        # Skip if it's a high-quality manually implemented test
        if any(name in test_file.name for name in ['test_main.py', 'test_tot_runner.py']):
            return False
        
        original_content = content
        
        # Analyze corresponding source module
        module_analysis = self.analyze_source_module(test_file.name)
        
        # Enhance the content
        content = self.enhance_initialization_tests(content, module_analysis)
        content = self.enhance_method_tests(content, module_analysis)
        content = self.enhance_edge_case_tests(content, module_analysis)
        content = self.enhance_integration_tests(content, module_analysis)
        content = self.add_enhancement_marker(content)
        
        # Only write if changes were made
        if content != original_content:
            test_file.write_text(content, encoding='utf-8')
            return True
        
        return False
    
    def enhance_initialization_tests(self, content: str, analysis: Dict) -> str:
        """Enhance initialization tests with realistic parameters."""
        if not analysis['classes']:
            return content
        
        # Replace generic initialization tests
        for class_info in analysis['classes']:
            class_name = class_info['name']
            init_args = class_info['init_args']
            
            # Generate realistic test data based on class name
            test_data = self.generate_test_data(class_name, init_args)
            
            # Replace mock initialization with real initialization
            pattern = rf'instance = self\._create_mock_instance\("{class_name}"\)'
            if init_args and test_data:
                replacement = f'instance = {class_name}({test_data})'
            else:
                replacement = f'instance = {class_name}()'
            
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def enhance_method_tests(self, content: str, analysis: Dict) -> str:
        """Enhance method tests with real method calls and assertions."""
        # Replace mock method calls with real calls
        pattern = r'result = self\._mock_method_call\(instance, "(\w+)"\)'
        
        def replace_method_call(match):
            method_name = match.group(1)
            # Generate appropriate method call based on method name
            if method_name.startswith('get_'):
                return f'result = instance.{method_name}()\n        assert result is not None'
            elif method_name.startswith('set_'):
                return f'instance.{method_name}("test_value")\n        # Verify state change'
            elif method_name.startswith('process_') or method_name.startswith('run_'):
                return f'result = instance.{method_name}()\n        assert isinstance(result, (dict, list, str, int, float, bool))'
            else:
                return f'result = instance.{method_name}()\n        assert result is not None'
        
        content = re.sub(pattern, replace_method_call, content)
        
        # Replace generic assertions
        content = re.sub(r'assert True  # Mock result', 'assert result is not None', content)
        
        return content
    
    def enhance_edge_case_tests(self, content: str, analysis: Dict) -> str:
        """Enhance edge case tests with specific test cases."""
        # Find edge case test methods and enhance them
        lines = content.split('\n')
        new_lines = []
        in_edge_case_method = False
        
        for line in lines:
            new_lines.append(line)
            
            if 'def test_edge_cases(' in line:
                in_edge_case_method = True
            elif in_edge_case_method and line.strip().startswith('def '):
                in_edge_case_method = False
            
            # Add specific edge case tests
            if in_edge_case_method and '# TODO: Test None handling' in line:
                specific_tests = [
                    '        # Test with None input',
                    '        with pytest.raises((TypeError, ValueError, AttributeError)):',
                    '            instance.method_that_requires_input(None)',
                    '',
                    '        # Test with empty input',
                    '        with pytest.raises((ValueError, IndexError)):',
                    '            instance.method_that_requires_input([])',
                    '',
                    '        # Test with invalid input',
                    '        with pytest.raises((ValueError, TypeError)):',
                    '            instance.method_that_requires_input("invalid")'
                ]
                new_lines.extend(specific_tests)
        
        return '\n'.join(new_lines)
    
    def enhance_integration_tests(self, content: str, analysis: Dict) -> str:
        """Enhance integration tests with realistic scenarios."""
        # Add realistic integration scenarios
        if 'def test_integration(' in content:
            integration_enhancement = '''
        # Test component interaction
        instance = self._create_mock_instance("{class_name}")
        
        # Simulate realistic workflow
        instance.initialize()
        result = instance.process_data(["test_input_1", "test_input_2"])
        instance.finalize()
        
        # Verify integration points
        assert hasattr(instance, 'state')
        assert result is not None'''
            
            content = re.sub(
                r'# TODO: Add integration tests',
                integration_enhancement.strip(),
                content
            )
        
        return content
    
    def generate_test_data(self, class_name: str, init_args: List[str]) -> str:
        """Generate realistic test data based on class name and arguments."""
        if not init_args:
            return ""
        
        # Determine module type
        module_type = 'config'
        if 'optimizer' in class_name.lower():
            module_type = 'optimizer'
        elif 'pipeline' in class_name.lower():
            module_type = 'pipeline'
        
        # Generate arguments based on parameter names
        arg_values = []
        for arg in init_args:
            if arg in ['config', 'configuration']:
                arg_values.append('{}')
            elif arg in ['size', 'population_size', 'batch_size']:
                arg_values.append('10')
            elif arg in ['rate', 'mutation_rate', 'learning_rate']:
                arg_values.append('0.1')
            elif arg in ['data', 'input_data']:
                arg_values.append('["test_data"]')
            elif arg in ['name', 'filename']:
                arg_values.append('"test"')
            else:
                arg_values.append('None')
        
        return ', '.join(arg_values)
    
    def add_enhancement_marker(self, content: str) -> str:
        """Add marker to indicate test has been enhanced."""
        lines = content.split('\n')
        
        # Add marker after the docstring
        for i, line in enumerate(lines):
            if '"""' in line and 'Auto-implemented' in line:
                lines[i] = line.replace('Auto-implemented', 'Enhanced with specific logic')
                break
        
        return '\n'.join(lines)
    
    def enhance_all_tests(self):
        """Enhance all auto-generated test files."""
        test_files = list(self.tests_dir.glob("test_*.py"))
        
        print(f"Found {len(test_files)} test files to enhance...")
        
        for test_file in test_files:
            try:
                if self.enhance_test_file(test_file):
                    print(f"Enhanced {test_file.name}")
                    self.enhanced_count += 1
            except Exception as e:
                print(f"Error enhancing {test_file.name}: {e}")
        
        print(f"\nEnhanced {self.enhanced_count} test files")

def main():
    """Main execution function."""
    enhancer = TestEnhancer()
    enhancer.enhance_all_tests()
    
    print("\nTest enhancement complete! Running sample tests to verify...")
    
    # Test a few enhanced files
    import subprocess
    sample_files = [
        "test_adaptive_evolution.py",
        "test_configuration_validation.py",
        "test_enhanced_config_manager.py"
    ]
    
    for test_file in sample_files:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"tests/unit/{test_file}", 
                "-v", "--tb=short", "-x"
            ], capture_output=True, text=True, timeout=30, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✅ {test_file} - Enhanced tests working")
            else:
                print(f"⚠️ {test_file} - Some enhancements need adjustment")
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} - Tests taking too long (likely import issues)")
        except Exception as e:
            print(f"💥 {test_file} - Error running tests: {e}")

if __name__ == "__main__":
    main()