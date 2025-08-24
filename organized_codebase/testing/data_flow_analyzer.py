#!/usr/bin/env python3
"""
Agent C - Data Flow Analyzer
Hours 10-12: Comprehensive data flow and variable tracking analysis.

Features:
- Variable lifecycle tracking (creation to destruction)
- Data transformation pipeline mapping
- Parameter flow analysis across functions
- Return value propagation tracking
- Global state mutation detection
- Data dependency graph construction
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataFlow:
    """Represents a data flow between components."""
    source: str  # Function/method that produces data
    target: str  # Function/method that consumes data
    data_type: str  # Type of data flowing
    flow_type: str  # 'parameter', 'return', 'global', 'attribute'
    variable_name: Optional[str] = None
    line_number: int = 0
    is_mutable: bool = False
    transformations: List[str] = field(default_factory=list)


@dataclass
class VariableLifecycle:
    """Tracks the lifecycle of a variable."""
    name: str
    scope: str  # Module, class, or function scope
    creation_line: int
    destruction_line: Optional[int] = None
    data_type: Optional[str] = None
    is_global: bool = False
    is_mutable: bool = False
    assignments: List[int] = field(default_factory=list)
    reads: List[int] = field(default_factory=list)
    mutations: List[int] = field(default_factory=list)
    passed_to_functions: List[str] = field(default_factory=list)


@dataclass
class DataTransformation:
    """Represents a data transformation operation."""
    input_vars: List[str]
    output_var: str
    operation_type: str  # 'map', 'filter', 'reduce', 'transform', 'aggregate'
    function: str
    line_number: int
    complexity: int = 1


@dataclass
class GlobalStateMutation:
    """Tracks global state mutations."""
    variable_name: str
    mutating_function: str
    mutation_type: str  # 'assignment', 'append', 'update', 'delete'
    line_number: int
    impact_scope: List[str] = field(default_factory=list)  # Functions affected


class DataFlowAnalyzer:
    """
    Comprehensive data flow analysis across the codebase.
    """
    
    def __init__(self, root_path: Path = Path(".")):
        self.root_path = root_path.resolve()
        self.data_flows: List[DataFlow] = []
        self.variable_lifecycles: Dict[str, VariableLifecycle] = {}
        self.transformations: List[DataTransformation] = []
        self.global_mutations: List[GlobalStateMutation] = []
        self.scan_timestamp = datetime.now()
        
        # Analysis tracking
        self.function_parameters: Dict[str, List[str]] = defaultdict(list)
        self.function_returns: Dict[str, List[str]] = defaultdict(list)
        self.data_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.transformation_chains: List[List[str]] = []
        self.critical_paths: List[List[str]] = []
        
        # Statistics
        self.stats = {
            'total_variables': 0,
            'global_variables': 0,
            'mutable_state': 0,
            'data_flows': 0,
            'transformations': 0,
            'global_mutations': 0,
            'scan_duration': 0.0
        }
    
    def analyze_data_flow(self) -> Dict[str, Any]:
        """
        Perform comprehensive data flow analysis.
        """
        start_time = time.time()
        logger.info(f"Starting data flow analysis for {self.root_path}")
        
        # Phase 1: Discover and analyze all Python files
        python_files = self._discover_python_files()
        logger.info(f"Analyzing data flow in {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                self._analyze_file_data_flow(file_path)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Phase 2: Build data dependency graph
        self._build_data_dependencies()
        
        # Phase 3: Identify transformation chains
        self._identify_transformation_chains()
        
        # Phase 4: Detect critical data paths
        self._detect_critical_paths()
        
        # Phase 5: Analyze global state mutations
        self._analyze_global_state_impact()
        
        # Phase 6: Calculate statistics
        self.stats['scan_duration'] = time.time() - start_time
        self._calculate_statistics()
        
        logger.info(f"Data flow analysis completed in {self.stats['scan_duration']:.2f} seconds")
        
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
    
    def _analyze_file_data_flow(self, file_path: Path):
        """Analyze data flow within a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
                
            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping")
                return
            
            module_name = self._calculate_module_name(file_path)
            
            # Create a visitor to analyze data flow
            visitor = DataFlowVisitor(self, module_name, source_code)
            visitor.visit(tree)
            
        except Exception as e:
            logger.error(f"Error analyzing data flow in {file_path}: {e}")
    
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
    
    def _build_data_dependencies(self):
        """Build data dependency relationships."""
        for flow in self.data_flows:
            if flow.source and flow.target:
                self.data_dependencies[flow.target].add(flow.source)
    
    def _identify_transformation_chains(self):
        """Identify chains of data transformations."""
        # Build transformation graph
        trans_graph = defaultdict(list)
        
        for trans in self.transformations:
            for input_var in trans.input_vars:
                trans_graph[input_var].append(trans.output_var)
        
        # Find transformation chains using DFS
        visited = set()
        
        def find_chain(var, chain):
            if var in visited:
                return
            visited.add(var)
            
            if var in trans_graph:
                for next_var in trans_graph[var]:
                    new_chain = chain + [next_var]
                    if len(new_chain) > 2:  # Only track significant chains
                        self.transformation_chains.append(new_chain)
                    find_chain(next_var, new_chain)
        
        for var in trans_graph:
            find_chain(var, [var])
    
    def _detect_critical_paths(self):
        """Detect critical data flow paths."""
        # Identify paths with high complexity or many transformations
        for chain in self.transformation_chains:
            complexity = sum(
                trans.complexity for trans in self.transformations
                if trans.output_var in chain
            )
            if complexity > 5 or len(chain) > 5:
                self.critical_paths.append(chain)
    
    def _analyze_global_state_impact(self):
        """Analyze the impact of global state mutations."""
        for mutation in self.global_mutations:
            # Find all functions that read this global variable
            affected_functions = []
            
            for var_name, lifecycle in self.variable_lifecycles.items():
                if var_name == mutation.variable_name and lifecycle.reads:
                    for func in lifecycle.passed_to_functions:
                        if func != mutation.mutating_function:
                            affected_functions.append(func)
            
            mutation.impact_scope = affected_functions
    
    def _calculate_statistics(self):
        """Calculate comprehensive statistics."""
        self.stats['total_variables'] = len(self.variable_lifecycles)
        self.stats['global_variables'] = sum(
            1 for v in self.variable_lifecycles.values() if v.is_global
        )
        self.stats['mutable_state'] = sum(
            1 for v in self.variable_lifecycles.values() if v.is_mutable
        )
        self.stats['data_flows'] = len(self.data_flows)
        self.stats['transformations'] = len(self.transformations)
        self.stats['global_mutations'] = len(self.global_mutations)
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive data flow analysis report."""
        return {
            'scan_metadata': {
                'timestamp': self.scan_timestamp.isoformat(),
                'root_path': str(self.root_path),
                'analysis_type': 'data_flow_analysis'
            },
            'statistics': self.stats,
            'data_flows': [asdict(flow) for flow in self.data_flows[:1000]],  # Limit for report size
            'variable_lifecycles': {
                name: asdict(lifecycle) 
                for name, lifecycle in list(self.variable_lifecycles.items())[:500]
            },
            'transformations': [asdict(trans) for trans in self.transformations[:500]],
            'global_mutations': [asdict(mut) for mut in self.global_mutations],
            'data_dependencies': {
                k: list(v) for k, v in list(self.data_dependencies.items())[:500]
            },
            'transformation_chains': self.transformation_chains[:100],
            'critical_paths': self.critical_paths,
            'insights': self._generate_insights()
        }
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate analytical insights from data flow analysis."""
        insights = {
            'data_quality': {},
            'complexity': {},
            'risks': {},
            'recommendations': []
        }
        
        # Data quality insights
        if self.stats['global_variables'] > 0:
            insights['data_quality']['global_state_percentage'] = (
                self.stats['global_variables'] / self.stats['total_variables'] * 100
                if self.stats['total_variables'] > 0 else 0
            )
        
        # Complexity insights
        if self.transformation_chains:
            insights['complexity']['longest_chain'] = max(
                len(chain) for chain in self.transformation_chains
            )
            insights['complexity']['critical_paths'] = len(self.critical_paths)
        
        # Risk insights
        if self.global_mutations:
            insights['risks']['global_mutations'] = len(self.global_mutations)
            insights['risks']['functions_affected'] = len(set(
                func for mut in self.global_mutations 
                for func in mut.impact_scope
            ))
        
        # Recommendations
        if self.stats['global_mutations'] > 10:
            insights['recommendations'].append(
                "High number of global mutations detected. Consider refactoring to reduce global state."
            )
        
        if self.critical_paths:
            insights['recommendations'].append(
                f"Found {len(self.critical_paths)} critical data paths. Review for optimization opportunities."
            )
        
        return insights
    
    def save_report(self, output_path: Path) -> None:
        """Save the data flow analysis report."""
        report = self._generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Data flow analysis report saved to {output_path}")


class DataFlowVisitor(ast.NodeVisitor):
    """AST visitor for analyzing data flow patterns."""
    
    def __init__(self, analyzer: DataFlowAnalyzer, module_name: str, source_code: str):
        self.analyzer = analyzer
        self.module_name = module_name
        self.source_code = source_code
        self.current_function = None
        self.current_class = None
        self.scope_stack = [module_name]
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        old_function = self.current_function
        self.current_function = f"{self.module_name}.{node.name}"
        self.scope_stack.append(node.name)
        
        # Analyze parameters
        for arg in node.args.args:
            param_name = arg.arg
            if param_name != 'self':
                self.analyzer.function_parameters[self.current_function].append(param_name)
                
                # Track parameter flow
                flow = DataFlow(
                    source="caller",
                    target=self.current_function,
                    data_type=self._get_type_annotation(arg),
                    flow_type='parameter',
                    variable_name=param_name,
                    line_number=node.lineno
                )
                self.analyzer.data_flows.append(flow)
        
        # Visit function body
        self.generic_visit(node)
        
        self.scope_stack.pop()
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = f"{self.module_name}.{node.name}"
        self.scope_stack.append(node.name)
        
        self.generic_visit(node)
        
        self.scope_stack.pop()
        self.current_class = old_class
    
    def visit_Assign(self, node):
        """Visit assignment statement."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                scope = '.'.join(self.scope_stack)
                
                # Create or update variable lifecycle
                lifecycle_key = f"{scope}.{var_name}"
                if lifecycle_key not in self.analyzer.variable_lifecycles:
                    self.analyzer.variable_lifecycles[lifecycle_key] = VariableLifecycle(
                        name=var_name,
                        scope=scope,
                        creation_line=node.lineno,
                        is_global='global' in scope or self.current_function is None
                    )
                
                lifecycle = self.analyzer.variable_lifecycles[lifecycle_key]
                lifecycle.assignments.append(node.lineno)
                
                # Check if it's a transformation
                if isinstance(node.value, ast.Call):
                    self._analyze_transformation(node, var_name)
        
        self.generic_visit(node)
    
    def visit_Global(self, node):
        """Visit global statement."""
        for name in node.names:
            if self.current_function:
                # Mark as global mutation
                mutation = GlobalStateMutation(
                    variable_name=name,
                    mutating_function=self.current_function,
                    mutation_type='assignment',
                    line_number=node.lineno
                )
                self.analyzer.global_mutations.append(mutation)
        
        self.generic_visit(node)
    
    def visit_Return(self, node):
        """Visit return statement."""
        if self.current_function and node.value:
            # Track return value flow
            flow = DataFlow(
                source=self.current_function,
                target="caller",
                data_type=self._infer_type(node.value),
                flow_type='return',
                line_number=node.lineno
            )
            self.analyzer.data_flows.append(flow)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function call."""
        if self.current_function:
            called_func = self._get_called_function_name(node)
            
            # Track data flow through function calls
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Name):
                    flow = DataFlow(
                        source=self.current_function,
                        target=called_func,
                        data_type=self._infer_type(arg),
                        flow_type='parameter',
                        variable_name=arg.id if isinstance(arg, ast.Name) else None,
                        line_number=node.lineno
                    )
                    self.analyzer.data_flows.append(flow)
        
        self.generic_visit(node)
    
    def _analyze_transformation(self, node: ast.Assign, output_var: str):
        """Analyze a data transformation."""
        if isinstance(node.value, ast.Call):
            func_name = self._get_called_function_name(node.value)
            input_vars = []
            
            # Extract input variables
            for arg in node.value.args:
                if isinstance(arg, ast.Name):
                    input_vars.append(arg.id)
            
            if input_vars:
                trans = DataTransformation(
                    input_vars=input_vars,
                    output_var=output_var,
                    operation_type=self._infer_operation_type(func_name),
                    function=self.current_function or self.module_name,
                    line_number=node.lineno,
                    complexity=len(input_vars)
                )
                self.analyzer.transformations.append(trans)
    
    def _get_called_function_name(self, call_node: ast.Call) -> str:
        """Extract the name of the called function."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return f"{self._get_called_function_name(call_node.func.value)}.{call_node.func.attr}"
        else:
            return "unknown"
    
    def _get_type_annotation(self, arg: ast.arg) -> str:
        """Get type annotation from argument."""
        if arg.annotation:
            return ast.unparse(arg.annotation)
        return "Any"
    
    def _infer_type(self, node: ast.AST) -> str:
        """Infer the type of an AST node."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Name):
            return "variable"
        else:
            return "unknown"
    
    def _infer_operation_type(self, func_name: str) -> str:
        """Infer the type of operation from function name."""
        if 'map' in func_name.lower():
            return 'map'
        elif 'filter' in func_name.lower():
            return 'filter'
        elif 'reduce' in func_name.lower():
            return 'reduce'
        elif 'aggregate' in func_name.lower():
            return 'aggregate'
        else:
            return 'transform'


def main():
    """Main entry point for the data flow analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent C - Data Flow Analyzer")
    parser.add_argument("--root", default=".", help="Root directory to analyze")
    parser.add_argument("--output", default="data_flow_hour10.json", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create analyzer and run analysis
    analyzer = DataFlowAnalyzer(Path(args.root))
    
    print("Agent C - Data Flow Analysis (Hours 10-12)")
    print(f"Analyzing: {analyzer.root_path}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Analyze data flow
    report = analyzer.analyze_data_flow()
    
    # Save report
    analyzer.save_report(Path(args.output))
    
    # Print summary
    stats = report['statistics']
    print(f"\nData Flow Analysis Results:")
    print(f"   Total Variables: {stats['total_variables']}")
    print(f"   Global Variables: {stats['global_variables']}")
    print(f"   Mutable State: {stats['mutable_state']}")
    print(f"   Data Flows: {stats['data_flows']}")
    print(f"   Transformations: {stats['transformations']}")
    print(f"   Global Mutations: {stats['global_mutations']}")
    print(f"   Scan Duration: {stats['scan_duration']:.2f} seconds")
    
    if report['critical_paths']:
        print(f"\nCritical Data Paths: {len(report['critical_paths'])}")
    
    if report['insights']['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['insights']['recommendations']:
            print(f"   - {rec}")
    
    print(f"\nData flow analysis complete! Report saved to {args.output}")


if __name__ == "__main__":
    main()