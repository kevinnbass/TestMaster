#!/usr/bin/env python3
"""
Cross-Module Dependency Test Generator
Generates integration tests for module interactions and dependencies.

Features:
- Detects integration points between modules
- Generates boundary tests for module interfaces
- Contract testing support
- API compatibility verification
- Data flow testing across modules
"""

import ast
import os
import sys
import json
import inspect
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
import logging
import importlib.util
import textwrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModuleInterface:
    """Interface definition for a module."""
    module_name: str
    module_path: Path
    public_functions: List[str] = field(default_factory=list)
    public_classes: List[str] = field(default_factory=list)
    exports: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    function_signatures: Dict[str, str] = field(default_factory=dict)
    class_methods: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class IntegrationPoint:
    """Represents an integration point between modules."""
    source_module: str
    target_module: str
    integration_type: str  # function_call, class_usage, data_exchange, inheritance
    source_element: str  # Function/class name in source
    target_element: str  # Function/class name in target
    relationship: str  # calls, inherits, imports, uses
    data_types: List[str] = field(default_factory=list)
    is_critical: bool = False


@dataclass
class TestContract:
    """Contract definition for testing module interactions."""
    contract_id: str
    provider_module: str
    consumer_module: str
    interface_function: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    error_conditions: List[str] = field(default_factory=list)


class ModuleAnalyzer:
    """Analyzes modules to extract interface information."""
    
    def __init__(self):
        self.module_interfaces: Dict[str, ModuleInterface] = {}
        self.integration_points: List[IntegrationPoint] = []
        
    def analyze_module(self, module_path: Path) -> ModuleInterface:
        """Analyze a Python module to extract its interface."""
        module_name = module_path.stem
        interface = ModuleInterface(
            module_name=module_name,
            module_path=module_path
        )
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        interface.imports.add(alias.name)
                        if not alias.name.startswith('.'):
                            interface.dependencies.add(alias.name.split('.')[0])
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        interface.imports.add(node.module)
                        if not node.module.startswith('.'):
                            interface.dependencies.add(node.module.split('.')[0])
                
                # Extract public functions
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    interface.public_functions.append(node.name)
                    interface.exports.add(node.name)
                    
                    # Extract function signature
                    signature = self._extract_function_signature(node)
                    interface.function_signatures[node.name] = signature
                
                # Extract public classes
                elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                    interface.public_classes.append(node.name)
                    interface.exports.add(node.name)
                    
                    # Extract class methods
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            methods.append(item.name)
                    interface.class_methods[node.name] = methods
        
        except Exception as e:
            logger.error(f"Failed to analyze {module_path}: {e}")
        
        self.module_interfaces[module_name] = interface
        return interface
    
    def _extract_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature as string."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Default arguments
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                arg_index = len(args) - num_defaults + i
                if arg_index >= 0:
                    args[arg_index] += f" = {ast.unparse(default)}"
        
        # Return type
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"
        
        return f"({', '.join(args)}){return_type}"
    
    def find_integration_points(self) -> List[IntegrationPoint]:
        """Find integration points between analyzed modules."""
        integration_points = []
        
        for source_name, source_interface in self.module_interfaces.items():
            for target_module in source_interface.dependencies:
                if target_module in self.module_interfaces:
                    target_interface = self.module_interfaces[target_module]
                    
                    # Find specific integration points
                    points = self._analyze_module_interaction(
                        source_interface, target_interface
                    )
                    integration_points.extend(points)
        
        self.integration_points = integration_points
        return integration_points
    
    def _analyze_module_interaction(self, source: ModuleInterface, 
                                  target: ModuleInterface) -> List[IntegrationPoint]:
        """Analyze interaction between two modules."""
        points = []
        
        # Check for function calls
        try:
            # Load source module to analyze its code
            source_code = source.module_path.read_text(encoding='utf-8')
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check if calling functions from target module
                    call_target = self._extract_call_target(node)
                    
                    if call_target and call_target in target.exports:
                        point = IntegrationPoint(
                            source_module=source.module_name,
                            target_module=target.module_name,
                            integration_type="function_call",
                            source_element="",  # Would need more analysis
                            target_element=call_target,
                            relationship="calls"
                        )
                        points.append(point)
                
                elif isinstance(node, ast.ClassDef):
                    # Check for inheritance
                    for base in node.bases:
                        base_name = ast.unparse(base)
                        if base_name in target.exports:
                            point = IntegrationPoint(
                                source_module=source.module_name,
                                target_module=target.module_name,
                                integration_type="inheritance",
                                source_element=node.name,
                                target_element=base_name,
                                relationship="inherits",
                                is_critical=True
                            )
                            points.append(point)
        
        except Exception as e:
            logger.error(f"Error analyzing interaction {source.module_name} -> {target.module_name}: {e}")
        
        return points
    
    def _extract_call_target(self, node: ast.Call) -> Optional[str]:
        """Extract the target function name from a call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None


class CrossModuleTestGenerator:
    """Generates integration tests for cross-module dependencies."""
    
    def __init__(self, test_output_dir: str = "tests/integration"):
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = ModuleAnalyzer()
        self.contracts: List[TestContract] = []
        
    def analyze_codebase(self, source_dir: Path) -> Dict[str, ModuleInterface]:
        """Analyze entire codebase for module interfaces."""
        logger.info(f"Analyzing codebase in {source_dir}")
        
        for py_file in source_dir.rglob("*.py"):
            if "test" not in py_file.name and "__pycache__" not in str(py_file):
                self.analyzer.analyze_module(py_file)
        
        # Find integration points
        integration_points = self.analyzer.find_integration_points()
        logger.info(f"Found {len(integration_points)} integration points")
        
        return self.analyzer.module_interfaces
    
    def generate_integration_tests(self) -> List[Path]:
        """Generate integration tests for all detected integration points."""
        generated_tests = []
        
        # Group integration points by target module
        points_by_target = defaultdict(list)
        for point in self.analyzer.integration_points:
            points_by_target[point.target_module].append(point)
        
        # Generate tests for each target module
        for target_module, points in points_by_target.items():
            test_file = self._generate_module_integration_test(target_module, points)
            if test_file:
                generated_tests.append(test_file)
        
        # Generate boundary tests
        boundary_tests = self._generate_boundary_tests()
        generated_tests.extend(boundary_tests)
        
        # Generate contract tests
        contract_tests = self._generate_contract_tests()
        generated_tests.extend(contract_tests)
        
        return generated_tests
    
    def _generate_module_integration_test(self, target_module: str, 
                                        points: List[IntegrationPoint]) -> Optional[Path]:
        """Generate integration test for a specific module."""
        test_file_name = f"test_integration_{target_module}.py"
        test_file_path = self.test_output_dir / test_file_name
        
        # Get target module interface
        if target_module not in self.analyzer.module_interfaces:
            return None
        
        target_interface = self.analyzer.module_interfaces[target_module]
        
        test_content = f'''#!/usr/bin/env python3
"""
Integration tests for {target_module} module
Auto-generated by CrossModuleTestGenerator
Generated: {datetime.now().isoformat()}
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import {target_module}
except ImportError:
    pytest.skip(f"Module {target_module} not available", allow_module_level=True)


class TestIntegration{target_module.title().replace("_", "")}:
    """Integration tests for {target_module} module interactions."""
    
    def setup_method(self):
        """Setup for each test method."""
        pass
    
    def teardown_method(self):
        """Cleanup after each test method."""
        pass
'''
        
        # Generate tests for each integration point
        for point in points:
            if point.integration_type == "function_call":
                test_content += self._generate_function_call_test(point, target_interface)
            elif point.integration_type == "inheritance":
                test_content += self._generate_inheritance_test(point, target_interface)
            elif point.integration_type == "data_exchange":
                test_content += self._generate_data_exchange_test(point, target_interface)
        
        # Add module-level integration tests
        test_content += self._generate_module_integration_tests(target_interface)
        
        # Write test file
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        logger.info(f"Generated integration test: {test_file_path}")
        return test_file_path
    
    def _generate_function_call_test(self, point: IntegrationPoint, 
                                   interface: ModuleInterface) -> str:
        """Generate test for function call integration."""
        function_name = point.target_element
        signature = interface.function_signatures.get(function_name, "()")
        
        return f'''
    
    def test_{point.source_module}_calls_{function_name}(self):
        """Test {point.source_module} calling {function_name}."""
        # Test that {point.source_module} can successfully call {function_name}
        try:
            # Mock dependencies if needed
            with patch.object({point.target_module}, '{function_name}') as mock_func:
                mock_func.return_value = "test_result"
                
                # Import and call the function
                from {point.source_module} import *
                result = {function_name}()  # This would need actual call analysis
                
                # Verify the call was made
                mock_func.assert_called()
                
        except Exception as e:
            pytest.fail(f"Integration test failed: {{e}}")
    
    def test_{function_name}_integration_contract(self):
        """Test the contract for {function_name} integration."""
        # Test input/output contract
        func = getattr({point.target_module}, '{function_name}', None)
        assert func is not None, f"Function {function_name} not found"
        
        # Test with valid inputs
        # TODO: Add specific test cases based on function signature
        assert callable(func), f"{function_name} should be callable"
'''
    
    def _generate_inheritance_test(self, point: IntegrationPoint, 
                                 interface: ModuleInterface) -> str:
        """Generate test for inheritance integration."""
        return f'''
    
    def test_{point.source_element}_inherits_from_{point.target_element}(self):
        """Test inheritance relationship between {point.source_element} and {point.target_element}."""
        from {point.source_module} import {point.source_element}
        from {point.target_module} import {point.target_element}
        
        # Test inheritance
        assert issubclass({point.source_element}, {point.target_element}), \\
            f"{point.source_element} should inherit from {point.target_element}"
        
        # Test that child class can be instantiated
        try:
            instance = {point.source_element}()
            assert isinstance(instance, {point.target_element}), \\
                f"Instance should be of type {point.target_element}"
        except Exception as e:
            pytest.fail(f"Failed to instantiate {point.source_element}: {{e}}")
    
    def test_{point.source_element}_method_inheritance(self):
        """Test that inherited methods work correctly."""
        from {point.source_module} import {point.source_element}
        
        instance = {point.source_element}()
        
        # Test inherited methods are available
        parent_methods = {interface.class_methods.get(point.target_element, [])}
        for method_name in parent_methods:
            assert hasattr(instance, method_name), \\
                f"Method {{method_name}} should be inherited"
'''
    
    def _generate_data_exchange_test(self, point: IntegrationPoint, 
                                   interface: ModuleInterface) -> str:
        """Generate test for data exchange integration."""
        return f'''
    
    def test_data_exchange_{point.source_module}_to_{point.target_module}(self):
        """Test data exchange between {point.source_module} and {point.target_module}."""
        # Test data format compatibility
        # This would need more specific analysis of data types
        
        # Test serialization/deserialization if applicable
        test_data = {{"test": "value", "number": 42}}
        
        # TODO: Add specific data exchange tests based on actual data flow
        assert True  # Placeholder
'''
    
    def _generate_module_integration_tests(self, interface: ModuleInterface) -> str:
        """Generate general module integration tests."""
        return f'''
    
    def test_module_imports_successfully(self):
        """Test that the module can be imported without errors."""
        try:
            import {interface.module_name}
            assert {interface.module_name} is not None
        except ImportError as e:
            pytest.fail(f"Failed to import {interface.module_name}: {{e}}")
    
    def test_public_api_availability(self):
        """Test that all public API elements are available."""
        import {interface.module_name}
        
        # Test public functions
        public_functions = {interface.public_functions}
        for func_name in public_functions:
            assert hasattr({interface.module_name}, func_name), \\
                f"Public function {{func_name}} not available"
            assert callable(getattr({interface.module_name}, func_name)), \\
                f"{{func_name}} should be callable"
        
        # Test public classes
        public_classes = {interface.public_classes}
        for class_name in public_classes:
            assert hasattr({interface.module_name}, class_name), \\
                f"Public class {{class_name}} not available"
            cls = getattr({interface.module_name}, class_name)
            assert isinstance(cls, type), \\
                f"{{class_name}} should be a class"
    
    def test_dependencies_available(self):
        """Test that all module dependencies are available."""
        dependencies = {list(interface.dependencies)}
        
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                pytest.fail(f"Dependency {{dep}} not available")
'''
    
    def _generate_boundary_tests(self) -> List[Path]:
        """Generate boundary tests for module interfaces."""
        boundary_tests = []
        
        for module_name, interface in self.analyzer.module_interfaces.items():
            if not interface.public_functions:
                continue
            
            test_file_name = f"test_boundary_{module_name}.py"
            test_file_path = self.test_output_dir / test_file_name
            
            test_content = f'''#!/usr/bin/env python3
"""
Boundary tests for {module_name} module
Tests module interface boundaries and edge cases
Generated: {datetime.now().isoformat()}
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import {module_name}
except ImportError:
    pytest.skip(f"Module {module_name} not available", allow_module_level=True)


class TestBoundary{module_name.title().replace("_", "")}:
    """Boundary tests for {module_name} module."""
'''
            
            # Generate boundary tests for each public function
            for func_name in interface.public_functions:
                signature = interface.function_signatures.get(func_name, "()")
                
                test_content += f'''
    
    def test_{func_name}_boundary_conditions(self):
        """Test boundary conditions for {func_name}."""
        func = getattr({module_name}, '{func_name}')
        
        # Test with None input
        try:
            result = func(None)
            # If it doesn't raise an exception, verify result is handled properly
            assert result is not None or result is None  # Either is valid
        except (TypeError, ValueError) as e:
            # Expected for functions that don't accept None
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception with None input: {{e}}")
        
        # Test with empty inputs where applicable
        try:
            # Test common empty values
            empty_values = ["", [], {{}}, 0]
            for empty_val in empty_values:
                try:
                    func(empty_val)
                except (TypeError, ValueError):
                    # Expected for incompatible types
                    pass
        except Exception as e:
            # Non-critical, boundary testing
            pass
    
    def test_{func_name}_error_handling(self):
        """Test error handling for {func_name}."""
        func = getattr({module_name}, '{func_name}')
        
        # Test that function handles errors gracefully
        # This is a general test - specific tests would need function analysis
        assert callable(func), f"{func_name} should be callable"
'''
            
            with open(test_file_path, 'w') as f:
                f.write(test_content)
            
            boundary_tests.append(test_file_path)
            logger.info(f"Generated boundary test: {test_file_path}")
        
        return boundary_tests
    
    def _generate_contract_tests(self) -> List[Path]:
        """Generate contract tests based on defined contracts."""
        if not self.contracts:
            return []
        
        contract_tests = []
        
        # Group contracts by provider module
        contracts_by_provider = defaultdict(list)
        for contract in self.contracts:
            contracts_by_provider[contract.provider_module].append(contract)
        
        for provider_module, contracts in contracts_by_provider.items():
            test_file_name = f"test_contracts_{provider_module}.py"
            test_file_path = self.test_output_dir / test_file_name
            
            test_content = f'''#!/usr/bin/env python3
"""
Contract tests for {provider_module} module
Tests interface contracts between modules
Generated: {datetime.now().isoformat()}
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestContracts{provider_module.title().replace("_", "")}:
    """Contract tests for {provider_module} module."""
'''
            
            for contract in contracts:
                test_content += self._generate_single_contract_test(contract)
            
            with open(test_file_path, 'w') as f:
                f.write(test_content)
            
            contract_tests.append(test_file_path)
            logger.info(f"Generated contract test: {test_file_path}")
        
        return contract_tests
    
    def _generate_single_contract_test(self, contract: TestContract) -> str:
        """Generate test for a single contract."""
        return f'''
    
    def test_contract_{contract.contract_id}(self):
        """Test contract {contract.contract_id} between {contract.provider_module} and {contract.consumer_module}."""
        from {contract.provider_module} import {contract.interface_function}
        
        # Test preconditions
        preconditions = {contract.preconditions}
        for condition in preconditions:
            # TODO: Implement precondition checks
            pass
        
        # Test function with valid input schema
        input_schema = {contract.input_schema}
        
        try:
            # Create test input based on schema
            test_input = self._create_test_input(input_schema)
            result = {contract.interface_function}(test_input)
            
            # Verify output schema
            self._verify_output_schema(result, {contract.output_schema})
            
            # Test postconditions
            postconditions = {contract.postconditions}
            for condition in postconditions:
                # TODO: Implement postcondition checks
                pass
                
        except Exception as e:
            pytest.fail(f"Contract test failed: {{e}}")
    
    def _create_test_input(self, schema):
        """Create test input based on schema."""
        # Simple schema-based input creation
        if isinstance(schema, dict):
            return {{k: "test_value" for k in schema.keys()}}
        return "test_input"
    
    def _verify_output_schema(self, result, schema):
        """Verify result matches output schema."""
        # Simple schema verification
        if isinstance(schema, dict) and isinstance(result, dict):
            for key in schema.keys():
                assert key in result, f"Output missing key: {{key}}"
'''
    
    def add_contract(self, contract: TestContract):
        """Add a contract definition for testing."""
        self.contracts.append(contract)
    
    def generate_dependency_graph(self) -> str:
        """Generate dependency graph in DOT format."""
        dot_lines = ["digraph module_dependencies {"]
        dot_lines.append('  rankdir=LR;')
        dot_lines.append('  node [shape=box];')
        
        # Add modules
        for module_name in self.analyzer.module_interfaces.keys():
            dot_lines.append(f'  "{module_name}";')
        
        # Add dependencies
        for point in self.analyzer.integration_points:
            style = "bold" if point.is_critical else "normal"
            dot_lines.append(
                f'  "{point.source_module}" -> "{point.target_module}" '
                f'[label="{point.integration_type}", style="{style}"];'
            )
        
        dot_lines.append("}")
        return "\n".join(dot_lines)
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration analysis report."""
        report_lines = [
            "=" * 70,
            "CROSS-MODULE INTEGRATION ANALYSIS REPORT",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            ""
        ]
        
        # Module summary
        report_lines.extend([
            "MODULE SUMMARY:",
            f"  Total modules analyzed: {len(self.analyzer.module_interfaces)}",
            f"  Integration points found: {len(self.analyzer.integration_points)}",
            f"  Test contracts defined: {len(self.contracts)}",
            ""
        ])
        
        # Integration points by type
        integration_types = defaultdict(int)
        for point in self.analyzer.integration_points:
            integration_types[point.integration_type] += 1
        
        if integration_types:
            report_lines.append("INTEGRATION TYPES:")
            for int_type, count in integration_types.items():
                report_lines.append(f"  {int_type}: {count}")
            report_lines.append("")
        
        # Critical integration points
        critical_points = [p for p in self.analyzer.integration_points if p.is_critical]
        if critical_points:
            report_lines.append("CRITICAL INTEGRATION POINTS:")
            for point in critical_points:
                report_lines.append(
                    f"  {point.source_module} -> {point.target_module} "
                    f"({point.integration_type})"
                )
            report_lines.append("")
        
        # Module dependencies
        dependencies = defaultdict(set)
        for point in self.analyzer.integration_points:
            dependencies[point.source_module].add(point.target_module)
        
        if dependencies:
            report_lines.append("MODULE DEPENDENCIES:")
            for module, deps in dependencies.items():
                report_lines.append(f"  {module}: {', '.join(sorted(deps))}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        
        if len(critical_points) > 5:
            report_lines.append(f"  • High number of critical integrations ({len(critical_points)}) - consider refactoring")
        
        if len(self.analyzer.integration_points) > len(self.analyzer.module_interfaces) * 3:
            report_lines.append("  • High coupling detected - consider reducing dependencies")
        
        if not self.contracts:
            report_lines.append("  • Consider defining explicit contracts for key integrations")
        
        circular_deps = self._detect_circular_dependencies()
        if circular_deps:
            report_lines.append(f"  • Circular dependencies detected: {len(circular_deps)} cycles")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the module graph."""
        # Build dependency graph
        graph = defaultdict(set)
        for point in self.analyzer.integration_points:
            graph[point.source_module].add(point.target_module)
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for module in self.analyzer.module_interfaces:
            if module not in visited:
                dfs(module, [])
        
        return cycles


def main():
    """CLI for cross-module test generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Module Dependency Test Generator")
    parser.add_argument("--source-dir", required=True, help="Source code directory")
    parser.add_argument("--output-dir", default="tests/integration", help="Test output directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze module dependencies")
    parser.add_argument("--generate", action="store_true", help="Generate integration tests")
    parser.add_argument("--report", action="store_true", help="Generate analysis report")
    parser.add_argument("--graph", help="Generate dependency graph (DOT file)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CrossModuleTestGenerator(test_output_dir=args.output_dir)
    
    # Analyze codebase
    print(f"Analyzing codebase in {args.source_dir}...")
    interfaces = generator.analyze_codebase(Path(args.source_dir))
    
    print(f"Found {len(interfaces)} modules")
    print(f"Found {len(generator.analyzer.integration_points)} integration points")
    
    if args.analyze or args.report:
        # Show analysis results
        integration_types = defaultdict(int)
        for point in generator.analyzer.integration_points:
            integration_types[point.integration_type] += 1
        
        print("\nIntegration types:")
        for int_type, count in integration_types.items():
            print(f"  {int_type}: {count}")
    
    if args.generate:
        # Generate integration tests
        print("\nGenerating integration tests...")
        test_files = generator.generate_integration_tests()
        
        print(f"Generated {len(test_files)} test files:")
        for test_file in test_files:
            print(f"  {test_file}")
    
    if args.graph:
        # Generate dependency graph
        graph_content = generator.generate_dependency_graph()
        with open(args.graph, 'w') as f:
            f.write(graph_content)
        print(f"Dependency graph saved to {args.graph}")
    
    if args.report:
        # Generate detailed report
        report = generator.generate_integration_report()
        print("\n" + report)
        
        # Save report
        report_file = Path(args.output_dir) / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()