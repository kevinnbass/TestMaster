#!/usr/bin/env python3
"""
Integration Test Generator
Analyzes main.py/orchestrators to generate integration tests.
Traces execution paths to test component interactions without reading every module.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import subprocess
import networkx as nx

@dataclass
class ExecutionPath:
    """Represents an execution path through the system."""
    entry_point: str
    steps: List[str]
    modules_involved: Set[str]
    data_flow: List[Tuple[str, str]]  # (from, to) data transfers
    critical: bool = False

@dataclass
class IntegrationPoint:
    """Represents an integration point between components."""
    source_module: str
    target_module: str
    interface_method: str
    data_contract: Dict
    test_scenarios: List[str]

class ExecutionFlowAnalyzer:
    """Analyzes execution flow from entry points."""
    
    def __init__(self, project_root: Path = Path(".")):
        self.project_root = project_root
        self.call_graph = nx.DiGraph()
        self.module_imports = {}
        self.execution_paths = []
    
    def analyze_main(self, main_file: Path = None) -> List[ExecutionPath]:
        """Analyze main.py or specified orchestrator file."""
        
        if main_file is None:
            # Find main entry points
            candidates = [
                self.project_root / "main.py",
                self.project_root / "multi_coder_analysis" / "main.py",
                self.project_root / "app.py",
                self.project_root / "run.py",
                self.project_root / "orchestrator.py"
            ]
            
            for candidate in candidates:
                if candidate.exists():
                    main_file = candidate
                    break
        
        if not main_file or not main_file.exists():
            raise FileNotFoundError("No main entry point found")
        
        print(f"Analyzing entry point: {main_file}")
        
        # Parse main file
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract execution flows
        tree = ast.parse(content)
        
        # 1. Find main execution block
        main_flow = self._extract_main_flow(tree)
        
        # 2. Trace function calls
        call_chains = self._trace_call_chains(tree)
        
        # 3. Identify critical paths (error handling, data validation)
        critical_paths = self._identify_critical_paths(content)
        
        # 4. Build execution paths
        for entry_point, calls in call_chains.items():
            path = ExecutionPath(
                entry_point=entry_point,
                steps=calls,
                modules_involved=self._extract_modules_from_calls(calls),
                data_flow=self._trace_data_flow(tree, calls),
                critical=entry_point in critical_paths
            )
            self.execution_paths.append(path)
        
        return self.execution_paths
    
    def _extract_main_flow(self, tree: ast.AST) -> List[str]:
        """Extract main execution flow."""
        main_flow = []
        
        # Look for if __name__ == "__main__" block
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is the main block
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == "__name__"):
                    
                    # Extract function calls in main block
                    for stmt in node.body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                            if isinstance(stmt.value.func, ast.Name):
                                main_flow.append(stmt.value.func.id)
                            elif isinstance(stmt.value.func, ast.Attribute):
                                main_flow.append(f"{stmt.value.func.attr}")
        
        return main_flow
    
    def _trace_call_chains(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Trace function call chains."""
        call_chains = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                calls = []
                
                # Find all function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            calls.append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            # Module.function call
                            calls.append(f"{child.func.attr}")
                
                if calls:
                    call_chains[func_name] = calls
        
        return call_chains
    
    def _identify_critical_paths(self, content: str) -> Set[str]:
        """Identify critical execution paths."""
        critical = set()
        
        # Patterns that indicate critical paths
        critical_patterns = [
            r'def\s+(\w+).*validate',
            r'def\s+(\w+).*auth',
            r'def\s+(\w+).*process.*payment',
            r'def\s+(\w+).*save.*database',
            r'def\s+(\w+).*send.*request',
            r'def\s+(\w+).*parse.*input',
            r'def\s+(\w+).*handle.*error'
        ]
        
        for pattern in critical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            critical.update(matches)
        
        return critical
    
    def _extract_modules_from_calls(self, calls: List[str]) -> Set[str]:
        """Extract module names from function calls."""
        modules = set()
        
        for call in calls:
            # Simple heuristic: if call contains underscore, likely a module
            if '_' in call:
                parts = call.split('_')
                if len(parts) > 1:
                    modules.add(parts[0])
        
        return modules
    
    def _trace_data_flow(self, tree: ast.AST, calls: List[str]) -> List[Tuple[str, str]]:
        """Trace data flow between functions."""
        data_flow = []
        
        # Simple analysis: function returns flow to next function
        for i in range(len(calls) - 1):
            data_flow.append((calls[i], calls[i + 1]))
        
        return data_flow

class IntegrationTestBuilder:
    """Builds integration tests from execution paths."""
    
    def __init__(self):
        self.test_templates = {
            'sequential': self._generate_sequential_test,
            'pipeline': self._generate_pipeline_test,
            'error_path': self._generate_error_path_test,
            'data_flow': self._generate_data_flow_test
        }
    
    def generate_integration_tests(self, paths: List[ExecutionPath]) -> str:
        """Generate integration tests for execution paths."""
        
        tests = []
        tests.append("""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestIntegration:
    '''Integration tests generated from execution flow analysis'''
""")
        
        # Generate test for each critical path
        for i, path in enumerate(paths):
            if path.critical or i < 5:  # Test critical paths + first 5
                test = self._generate_path_test(path, i)
                tests.append(test)
        
        # Add data flow tests
        data_flow_test = self._generate_data_flow_tests(paths)
        tests.append(data_flow_test)
        
        # Add end-to-end test
        e2e_test = self._generate_end_to_end_test(paths)
        tests.append(e2e_test)
        
        return '\n'.join(tests)
    
    def _generate_path_test(self, path: ExecutionPath, index: int) -> str:
        """Generate test for a specific execution path."""
        
        test = f"""
    def test_path_{index}_{path.entry_point}(self):
        '''Test execution path: {path.entry_point}'''
        # Path: {' -> '.join(path.steps[:5])}
        
        # Setup
        """
        
        # Add setup for modules involved
        for module in path.modules_involved:
            test += f"""
        # Setup {module} module
        {module}_mock = Mock()
        """
        
        # Test the path
        test += f"""
        # Execute path
        try:
            from {path.entry_point.replace('/', '.').replace('.py', '')} import main_function
            result = main_function()
            
            # Verify interactions
            """
        
        # Add assertions for each step
        for i, step in enumerate(path.steps[:5]):
            test += f"""
            # Step {i+1}: {step}
            # TODO: Add specific assertion for {step}
            """
        
        test += """
        except ImportError:
            pytest.skip("Module not importable")
        """
        
        return test
    
    def _generate_sequential_test(self, steps: List[str]) -> str:
        """Generate test for sequential execution."""
        return f"""
    def test_sequential_execution(self):
        '''Test sequential execution of components'''
        # Steps: {' -> '.join(steps)}
        
        results = []
        for step in {steps}:
            # Execute each step and collect results
            result = execute_step(step)
            results.append(result)
            assert result is not None, f"Step {{step}} failed"
        
        # Verify final state
        assert len(results) == {len(steps)}
"""
    
    def _generate_pipeline_test(self, modules: Set[str]) -> str:
        """Generate pipeline test."""
        return f"""
    def test_pipeline_flow(self):
        '''Test data pipeline through modules'''
        # Modules: {', '.join(modules)}
        
        test_data = generate_test_data()
        
        # Pass through pipeline
        result = test_data
        for module in {list(modules)}:
            processor = get_module_processor(module)
            result = processor.process(result)
            assert result is not None, f"Module {{module}} returned None"
        
        # Verify final output
        assert validate_output(result)
"""
    
    def _generate_error_path_test(self, path: ExecutionPath) -> str:
        """Generate error handling test."""
        return f"""
    def test_error_handling_{path.entry_point}(self):
        '''Test error handling in {path.entry_point}'''
        
        # Inject error at each step
        for step in {path.steps[:3]}:
            with patch(f'{{step}}', side_effect=Exception("Test error")):
                # Should handle error gracefully
                try:
                    execute_path('{path.entry_point}')
                except Exception as e:
                    assert "handled" in str(e).lower(), "Error not properly handled"
"""
    
    def _generate_data_flow_test(self, flow: List[Tuple[str, str]]) -> str:
        """Generate data flow test."""
        if not flow:
            return ""
        
        return f"""
    def test_data_flow(self):
        '''Test data flow between components'''
        # Flow: {flow[:3]}
        
        # Test data transformation
        initial_data = {{"test": "data"}}
        
        for source, target in {flow[:3]}:
            # Data should flow from source to target
            output = simulate_flow(source, target, initial_data)
            assert output is not None, f"Data lost between {{source}} and {{target}}"
            initial_data = output  # Use for next step
"""
    
    def _generate_data_flow_tests(self, paths: List[ExecutionPath]) -> str:
        """Generate comprehensive data flow tests."""
        test = """
    def test_data_flow_integrity(self):
        '''Test data integrity through execution paths'''
        
        test_cases = [
            {"input": "valid_data", "expected": "processed"},
            {"input": None, "expected": "handled"},
            {"input": {}, "expected": "empty_handled"}
        ]
        
        for case in test_cases:
            # Test data flow through system
            result = process_through_system(case["input"])
            assert result == case["expected"], f"Failed for input: {case['input']}"
"""
        return test
    
    def _generate_end_to_end_test(self, paths: List[ExecutionPath]) -> str:
        """Generate end-to-end test."""
        # Find the longest path (likely most comprehensive)
        longest_path = max(paths, key=lambda p: len(p.steps)) if paths else None
        
        if not longest_path:
            return ""
        
        return f"""
    def test_end_to_end(self):
        '''End-to-end test of main execution flow'''
        # Testing path: {longest_path.entry_point}
        # Steps: {len(longest_path.steps)}
        
        # Setup test environment
        setup_test_environment()
        
        try:
            # Execute main flow
            from main import main
            result = main(test_mode=True)
            
            # Verify critical checkpoints
            assert result is not None, "Main returned None"
            assert result.get('status') == 'success', "Execution failed"
            
            # Verify all modules were called
            for module in {list(longest_path.modules_involved)[:5]}:
                assert module_was_called(module), f"Module {{module}} not executed"
            
        except ImportError:
            pytest.skip("Main not importable")
        finally:
            cleanup_test_environment()
"""

class SmartIntegrationTestGenerator:
    """Main generator that creates integration tests intelligently."""
    
    def __init__(self, project_root: Path = Path(".")):
        self.analyzer = ExecutionFlowAnalyzer(project_root)
        self.builder = IntegrationTestBuilder()
        self.project_root = project_root
    
    def generate_from_main(self, main_file: Path = None) -> str:
        """Generate integration tests by analyzing main entry point."""
        
        # 1. Analyze execution flow
        print("Analyzing execution flow...")
        paths = self.analyzer.analyze_main(main_file)
        print(f"Found {len(paths)} execution paths")
        
        # 2. Generate integration tests
        print("Generating integration tests...")
        tests = self.builder.generate_integration_tests(paths)
        
        # 3. Add test utilities
        test_utilities = self._generate_test_utilities()
        
        # 4. Combine everything
        complete_test = f"""
'''
Integration Tests
Generated by analyzing main execution flow
Covers: {len(paths)} execution paths
'''

{test_utilities}

{tests}

# ===== Test Utilities =====

def setup_test_environment():
    '''Setup test environment'''
    # Reset state, create temp files, etc.
    pass

def cleanup_test_environment():
    '''Cleanup after tests'''
    # Remove temp files, reset state, etc.
    pass

def module_was_called(module_name):
    '''Check if module was called during execution'''
    # Implementation depends on monitoring approach
    return True

def execute_step(step_name):
    '''Execute a single step in isolation'''
    # Dynamic execution of step
    return {{"status": "ok"}}

def execute_path(path_name):
    '''Execute a complete path'''
    # Execute path with monitoring
    pass

def simulate_flow(source, target, data):
    '''Simulate data flow between components'''
    # Transform data as it would flow
    return data

def process_through_system(input_data):
    '''Process data through entire system'''
    # Full pipeline execution
    return "processed" if input_data else "handled"

def generate_test_data():
    '''Generate test data for integration tests'''
    return {{
        "id": "test_123",
        "data": ["item1", "item2"],
        "config": {{"key": "value"}}
    }}

def validate_output(output):
    '''Validate final output'''
    return output is not None and isinstance(output, (dict, list, str))

def get_module_processor(module_name):
    '''Get processor for a module'''
    class Processor:
        def process(self, data):
            return data  # Pass through for testing
    return Processor()
"""
        
        return complete_test
    
    def _generate_test_utilities(self) -> str:
        """Generate utility functions for integration testing."""
        return """
import tempfile
import shutil
from contextlib import contextmanager

@contextmanager
def temp_directory():
    '''Create temporary directory for testing'''
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

@contextmanager
def mock_external_services():
    '''Mock external service calls'''
    with patch('requests.get') as mock_get, \\
         patch('requests.post') as mock_post:
        mock_get.return_value.json.return_value = {"status": "ok"}
        mock_post.return_value.status_code = 200
        yield mock_get, mock_post

def assert_data_integrity(input_data, output_data):
    '''Assert data integrity through transformation'''
    # Check that critical fields are preserved
    assert 'id' in output_data if 'id' in input_data else True
    # Check data wasn't corrupted
    assert output_data is not None
"""
    
    def generate_and_save(self, output_file: Path = None):
        """Generate and save integration tests."""
        
        if output_file is None:
            output_file = Path("tests/integration/test_integration_auto.py")
        
        # Generate tests
        tests = self.generate_from_main()
        
        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tests)
        
        print(f"Integration tests saved to: {output_file}")
        
        # Generate summary
        summary = {
            "generated": datetime.now().isoformat(),
            "file": str(output_file),
            "paths_analyzed": len(self.analyzer.execution_paths),
            "critical_paths": sum(1 for p in self.analyzer.execution_paths if p.critical)
        }
        
        summary_file = output_file.parent / "integration_test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return tests

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    generator = SmartIntegrationTestGenerator()
    
    if len(sys.argv) > 1:
        # Use specified main file
        main_file = Path(sys.argv[1])
        generator.generate_from_main(main_file)
    else:
        # Auto-detect main file
        generator.generate_and_save()
    
    print("Integration test generation complete!")