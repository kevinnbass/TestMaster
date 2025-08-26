#!/usr/bin/env python3
"""
Agent C - Function Call Graph Builder
Hours 4-6: Comprehensive function call graph construction and analysis.

Features:
- Complete function call graph generation
- Direct and indirect call tracking
- Recursive pattern detection
- Dead code identification
- Performance bottleneck analysis
"""

import ast
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Information about a function call."""
    caller_function: str
    caller_module: str
    called_function: str
    called_module: Optional[str]
    call_type: str  # 'direct', 'method', 'attribute', 'dynamic'
    line_number: int
    is_recursive: bool = False
    is_conditional: bool = False
    call_depth: int = 0


@dataclass
class FunctionDefinition:
    """Information about a function definition."""
    name: str
    module: str
    file_path: str
    line_number: int
    is_async: bool = False
    is_method: bool = False
    is_static: bool = False
    is_class_method: bool = False
    is_property: bool = False
    parameters: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    complexity_score: int = 0
    calls_made: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)


@dataclass
class CallGraphMetrics:
    """Metrics about the call graph."""
    total_functions: int = 0
    total_calls: int = 0
    recursive_calls: int = 0
    dead_code_functions: int = 0
    max_call_depth: int = 0
    average_calls_per_function: float = 0.0
    most_called_functions: List[Tuple[str, int]] = field(default_factory=list)
    most_calling_functions: List[Tuple[str, int]] = field(default_factory=list)
    circular_call_chains: List[List[str]] = field(default_factory=list)


class FunctionCallGraphBuilder:
    """
    Build comprehensive function call graphs for the codebase.
    """
    
    def __init__(self, root_path: Path = Path(".")):
        self.root_path = root_path.resolve()
        self.functions: Dict[str, FunctionDefinition] = {}
        self.calls: List[FunctionCall] = []
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.metrics = CallGraphMetrics()
        self.scan_timestamp = datetime.now()
        
        # Tracking for analysis
        self.module_functions: Dict[str, Set[str]] = defaultdict(set)
        self.recursive_patterns: Dict[str, List[str]] = {}
        self.dead_code: Set[str] = set()
        self.call_chains: List[List[str]] = []
        
    def build_call_graph(self) -> Dict[str, Any]:
        """
        Build the complete function call graph.
        """
        start_time = time.time()
        logger.info(f"Starting function call graph construction for {self.root_path}")
        
        # Phase 1: Discover and analyze all functions
        python_files = self._discover_python_files()
        logger.info(f"Analyzing functions in {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                self._analyze_file_functions(file_path)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Phase 2: Build call relationships
        logger.info("Building function call relationships...")
        for file_path in python_files:
            try:
                self._analyze_function_calls(file_path)
            except Exception as e:
                logger.error(f"Error analyzing calls in {file_path}: {e}")
        
        # Phase 3: Detect patterns and compute metrics
        self._detect_recursive_patterns()
        self._identify_dead_code()
        self._detect_circular_call_chains()
        self._compute_metrics()
        
        duration = time.time() - start_time
        logger.info(f"Call graph construction completed in {duration:.2f} seconds")
        
        return self._generate_comprehensive_report()
    
    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files in the codebase."""
        python_files = []
        
        for py_file in self.root_path.rglob("*.py"):
            # Skip common non-source directories
            if any(exclude in str(py_file) for exclude in [
                '__pycache__', '.git', '.venv', 'venv', 'env',
                'node_modules', '.pytest_cache', '.coverage'
            ]):
                continue
                
            python_files.append(py_file)
        
        return sorted(python_files)
    
    def _analyze_file_functions(self, file_path: Path):
        """Analyze all function definitions in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
                
            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping")
                return
            
            module_name = self._calculate_module_name(file_path)
            
            # Extract function definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_def = self._extract_function_definition(node, module_name, file_path, source_code)
                    if func_def:
                        full_name = f"{module_name}.{func_def.name}"
                        self.functions[full_name] = func_def
                        self.module_functions[module_name].add(func_def.name)
                        
        except Exception as e:
            logger.error(f"Error analyzing functions in {file_path}: {e}")
    
    def _extract_function_definition(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                                   module_name: str, file_path: Path, source_code: str) -> Optional[FunctionDefinition]:
        """Extract function definition information from AST node."""
        try:
            func_def = FunctionDefinition(
                name=node.name,
                module=module_name,
                file_path=str(file_path),
                line_number=node.lineno,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                docstring=ast.get_docstring(node),
                complexity_score=self._calculate_function_complexity(node)
            )
            
            # Extract parameters
            for arg in node.args.args:
                func_def.parameters.append(arg.arg)
            
            # Extract decorators
            for decorator in node.decorator_list:
                decorator_name = self._extract_decorator_name(decorator)
                func_def.decorators.append(decorator_name)
                
                # Check for special decorator types
                if decorator_name in ['staticmethod']:
                    func_def.is_static = True
                elif decorator_name in ['classmethod']:
                    func_def.is_class_method = True
                elif decorator_name in ['property']:
                    func_def.is_property = True
            
            # Check if it's a method (has 'self' or 'cls' parameter)
            if func_def.parameters:
                if func_def.parameters[0] in ['self', 'cls']:
                    func_def.is_method = True
            
            return func_def
            
        except Exception as e:
            logger.error(f"Error extracting function definition for {node.name}: {e}")
            return None
    
    def _extract_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._extract_decorator_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._extract_decorator_name(decorator.func)
        else:
            return "unknown"
    
    def _calculate_function_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _analyze_function_calls(self, file_path: Path):
        """Analyze function calls within a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
                
            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                return
            
            module_name = self._calculate_module_name(file_path)
            current_function = None
            
            # Track current function context
            class CallVisitor(ast.NodeVisitor):
                def __init__(self, builder):
                    self.builder = builder
                    self.current_function = None
                    self.current_module = module_name
                    
                def visit_FunctionDef(self, node):
                    old_function = self.current_function
                    self.current_function = f"{self.current_module}.{node.name}"
                    self.generic_visit(node)
                    self.current_function = old_function
                    
                def visit_AsyncFunctionDef(self, node):
                    self.visit_FunctionDef(node)
                    
                def visit_Call(self, node):
                    if self.current_function:
                        call_info = self.builder._extract_call_information(
                            node, self.current_function, self.current_module, source_code
                        )
                        if call_info:
                            self.builder.calls.append(call_info)
                            
                            # Update call graph
                            self.builder.call_graph[self.current_function].add(call_info.called_function)
                            self.builder.reverse_call_graph[call_info.called_function].add(self.current_function)
                    
                    self.generic_visit(node)
            
            visitor = CallVisitor(self)
            visitor.visit(tree)
                        
        except Exception as e:
            logger.error(f"Error analyzing calls in {file_path}: {e}")
    
    def _extract_call_information(self, call_node: ast.Call, current_function: str, 
                                current_module: str, source_code: str) -> Optional[FunctionCall]:
        """Extract information about a function call."""
        try:
            called_function = self._resolve_call_target(call_node)
            if not called_function:
                return None
            
            call_type = self._determine_call_type(call_node)
            
            # Check if it's a recursive call
            is_recursive = called_function == current_function or current_function.endswith(f".{called_function}")
            
            # Check if call is conditional
            is_conditional = self._is_conditional_call(call_node, source_code)
            
            return FunctionCall(
                caller_function=current_function,
                caller_module=current_module,
                called_function=called_function,
                called_module=self._resolve_called_module(called_function),
                call_type=call_type,
                line_number=getattr(call_node, 'lineno', 0),
                is_recursive=is_recursive,
                is_conditional=is_conditional
            )
            
        except Exception as e:
            logger.error(f"Error extracting call information: {e}")
            return None
    
    def _resolve_call_target(self, call_node: ast.Call) -> Optional[str]:
        """Resolve the target function of a call."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return f"{self._resolve_attribute_chain(call_node.func.value)}.{call_node.func.attr}"
        else:
            return None
    
    def _resolve_attribute_chain(self, node: ast.AST) -> str:
        """Resolve a chain of attributes (e.g., obj.method.submethod)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._resolve_attribute_chain(node.value)}.{node.attr}"
        else:
            return "unknown"
    
    def _determine_call_type(self, call_node: ast.Call) -> str:
        """Determine the type of function call."""
        if isinstance(call_node.func, ast.Name):
            return 'direct'
        elif isinstance(call_node.func, ast.Attribute):
            return 'method'
        else:
            return 'dynamic'
    
    def _resolve_called_module(self, called_function: str) -> Optional[str]:
        """Resolve the module of a called function."""
        if '.' in called_function:
            parts = called_function.split('.')
            if len(parts) >= 2:
                return '.'.join(parts[:-1])
        return None
    
    def _is_conditional_call(self, call_node: ast.Call, source_code: str) -> bool:
        """Check if a call is within a conditional block."""
        # Simple heuristic: check if call is indented more than base level
        lines = source_code.splitlines()
        if hasattr(call_node, 'lineno') and call_node.lineno <= len(lines):
            line = lines[call_node.lineno - 1]
            return line.lstrip() != line  # Indented = conditional
        return False
    
    def _calculate_module_name(self, file_path: Path) -> str:
        """Calculate the module name from file path."""
        relative_path = file_path.relative_to(self.root_path)
        parts = list(relative_path.parts)
        
        # Remove .py extension
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        # Handle __init__.py files
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts) if parts else '__main__'
    
    def _detect_recursive_patterns(self):
        """Detect recursive function call patterns."""
        for call in self.calls:
            if call.is_recursive:
                caller = call.caller_function
                if caller not in self.recursive_patterns:
                    self.recursive_patterns[caller] = []
                
                # Trace the recursive call chain
                chain = self._trace_recursive_chain(caller, set())
                if chain:
                    self.recursive_patterns[caller] = chain
    
    def _trace_recursive_chain(self, function: str, visited: Set[str]) -> List[str]:
        """Trace a recursive call chain."""
        if function in visited:
            return [function]  # Found the recursion point
        
        visited.add(function)
        
        for called_func in self.call_graph.get(function, set()):
            if called_func == function:  # Direct recursion
                return [function, called_func]
            
            chain = self._trace_recursive_chain(called_func, visited.copy())
            if chain and function in chain:
                return [function] + chain
        
        return []
    
    def _identify_dead_code(self):
        """Identify functions that are never called (potential dead code)."""
        all_functions = set(self.functions.keys())
        called_functions = set()
        
        for call in self.calls:
            called_functions.add(call.called_function)
            # Also add qualified names
            if '.' not in call.called_function:
                for func_name in all_functions:
                    if func_name.endswith(f".{call.called_function}"):
                        called_functions.add(func_name)
        
        # Functions that are never called
        self.dead_code = all_functions - called_functions
        
        # Remove entry points and special methods from dead code
        entry_points = {'__main__.main', 'main', '__init__'}
        special_methods = {f for f in self.dead_code if f.split('.')[-1].startswith('__')}
        
        self.dead_code = self.dead_code - entry_points - special_methods
    
    def _detect_circular_call_chains(self):
        """Detect circular call chains using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(function, path):
            if function in rec_stack:
                # Found a cycle
                cycle_start = path.index(function)
                cycle = path[cycle_start:] + [function]
                self.call_chains.append(cycle)
                return
            
            if function in visited:
                return
            
            visited.add(function)
            rec_stack.add(function)
            
            for called_func in self.call_graph.get(function, set()):
                dfs(called_func, path + [called_func])
            
            rec_stack.remove(function)
        
        for function in self.functions:
            if function not in visited:
                dfs(function, [function])
    
    def _compute_metrics(self):
        """Compute comprehensive metrics about the call graph."""
        self.metrics.total_functions = len(self.functions)
        self.metrics.total_calls = len(self.calls)
        self.metrics.recursive_calls = sum(1 for call in self.calls if call.is_recursive)
        self.metrics.dead_code_functions = len(self.dead_code)
        self.metrics.circular_call_chains = self.call_chains
        
        # Calculate call statistics
        call_counts = defaultdict(int)
        caller_counts = defaultdict(int)
        
        for call in self.calls:
            call_counts[call.called_function] += 1
            caller_counts[call.caller_function] += 1
        
        if self.functions:
            self.metrics.average_calls_per_function = len(self.calls) / len(self.functions)
        
        # Most called functions
        self.metrics.most_called_functions = sorted(
            call_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Most calling functions
        self.metrics.most_calling_functions = sorted(
            caller_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Calculate max call depth (simplified)
        self.metrics.max_call_depth = max(
            [len(chain) for chain in self.call_chains] + [0]
        )
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive call graph analysis report."""
        return {
            'scan_metadata': {
                'timestamp': self.scan_timestamp.isoformat(),
                'root_path': str(self.root_path),
                'analysis_type': 'function_call_graph'
            },
            'metrics': asdict(self.metrics),
            'functions': {
                name: asdict(func_def) for name, func_def in self.functions.items()
            },
            'calls': [asdict(call) for call in self.calls],
            'call_graph': {
                func: list(calls) for func, calls in self.call_graph.items()
            },
            'reverse_call_graph': {
                func: list(callers) for func, callers in self.reverse_call_graph.items()
            },
            'recursive_patterns': self.recursive_patterns,
            'dead_code': list(self.dead_code),
            'circular_call_chains': self.call_chains,
            'analysis_insights': self._generate_insights()
        }
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate analytical insights from the call graph."""
        insights = {
            'code_quality': {},
            'performance': {},
            'architecture': {},
            'maintenance': {}
        }
        
        # Code quality insights
        if self.dead_code:
            insights['code_quality']['dead_code_percentage'] = (
                len(self.dead_code) / len(self.functions) * 100
            )
        
        # Performance insights
        if self.metrics.most_called_functions:
            insights['performance']['hotspots'] = self.metrics.most_called_functions[:5]
        
        # Architecture insights
        if self.call_chains:
            insights['architecture']['circular_dependencies'] = len(self.call_chains)
        
        # Maintenance insights
        complex_functions = [
            name for name, func in self.functions.items() 
            if func.complexity_score > 10
        ]
        insights['maintenance']['high_complexity_functions'] = len(complex_functions)
        
        return insights
    
    def save_report(self, output_path: Path) -> None:
        """Save the call graph analysis report."""
        report = self._generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Call graph analysis report saved to {output_path}")


def main():
    """Main entry point for the function call graph builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent C - Function Call Graph Builder")
    parser.add_argument("--root", default=".", help="Root directory to analyze")
    parser.add_argument("--output", default="function_call_graph_hour4.json", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create builder and run analysis
    builder = FunctionCallGraphBuilder(Path(args.root))
    
    print("Agent C - Function Call Graph Construction (Hours 4-6)")
    print(f"Analyzing: {builder.root_path}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Build call graph
    report = builder.build_call_graph()
    
    # Save report
    builder.save_report(Path(args.output))
    
    # Print summary
    metrics = report['metrics']
    print(f"\nFunction Call Graph Analysis Results:")
    print(f"   Total Functions: {metrics['total_functions']}")
    print(f"   Total Function Calls: {metrics['total_calls']}")
    print(f"   Recursive Calls: {metrics['recursive_calls']}")
    print(f"   Dead Code Functions: {metrics['dead_code_functions']}")
    print(f"   Circular Call Chains: {len(metrics['circular_call_chains'])}")
    print(f"   Average Calls per Function: {metrics['average_calls_per_function']:.2f}")
    print(f"   Max Call Depth: {metrics['max_call_depth']}")
    
    if metrics['most_called_functions']:
        print(f"\nMost Called Functions:")
        for func, count in metrics['most_called_functions'][:5]:
            print(f"   {func}: {count} calls")
    
    if report['dead_code']:
        print(f"\nDead Code Detected:")
        for func in list(report['dead_code'])[:5]:
            print(f"   {func}")
    
    print(f"\nCall graph analysis complete! Report saved to {args.output}")


if __name__ == "__main__":
    main()