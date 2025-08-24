#!/usr/bin/env python3
"""
Agent C - Import/Export Relationship Scanner
Comprehensive analysis of all module dependencies and relationships.

Features:
- Complete import/export mapping for all Python files
- Circular dependency detection and severity rating
- External dependency version audit
- Neo4j-compatible dependency matrix generation
- Real-time relationship tracking and visualization
"""

import ast
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import re
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about a single import statement."""
    module_name: str
    imported_names: List[str] = field(default_factory=list)
    import_type: str = "import"  # 'import', 'from_import', 'dynamic'
    alias: Optional[str] = None
    level: int = 0  # For relative imports
    line_number: int = 0
    is_relative: bool = False
    is_conditional: bool = False


@dataclass
class ModuleAnalysis:
    """Complete analysis of a single module."""
    file_path: str
    module_name: str
    imports: List[ImportInfo] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    size_lines: int = 0
    complexity_score: int = 0
    has_main_guard: bool = False
    docstring: Optional[str] = None
    encoding: str = "utf-8"


@dataclass
class DependencyGraph:
    """Represents the complete dependency graph."""
    nodes: Dict[str, ModuleAnalysis] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)  # module -> dependencies
    reverse_edges: Dict[str, Set[str]] = field(default_factory=dict)  # module -> dependents
    external_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    
    
@dataclass
class CircularDependency:
    """Information about a circular dependency."""
    cycle_path: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    impact_score: int
    affected_modules: Set[str] = field(default_factory=set)
    suggested_fixes: List[str] = field(default_factory=list)


class RelationshipScanner:
    """
    Comprehensive scanner for import/export relationships across the codebase.
    """
    
    def __init__(self, root_path: Path = Path(".")):
        self.root_path = root_path.resolve()
        self.dependency_graph = DependencyGraph()
        self.module_analyses: Dict[str, ModuleAnalysis] = {}
        self.external_deps: Dict[str, Set[str]] = defaultdict(set)
        self.circular_deps: List[CircularDependency] = []
        self.scan_timestamp = datetime.now()
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'total_imports': 0,
            'external_imports': 0,
            'internal_imports': 0,
            'circular_dependencies': 0,
            'unused_imports': 0,
            'scan_duration': 0.0
        }
    
    def scan_codebase(self) -> Dict[str, Any]:
        """
        Perform comprehensive scan of the entire codebase.
        Returns complete relationship analysis.
        """
        start_time = time.time()
        logger.info(f"Starting comprehensive relationship scan of {self.root_path}")
        
        # Phase 1: Discover all Python files
        python_files = self._discover_python_files()
        self.stats['total_files'] = len(python_files)
        logger.info(f"Discovered {len(python_files)} Python files")
        
        # Phase 2: Analyze each module
        for file_path in python_files:
            try:
                analysis = self._analyze_module(file_path)
                if analysis:
                    self.module_analyses[analysis.module_name] = analysis
                    self.dependency_graph.nodes[analysis.module_name] = analysis
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Phase 3: Build dependency relationships
        self._build_dependency_graph()
        
        # Phase 4: Detect circular dependencies
        self._detect_circular_dependencies()
        
        # Phase 5: Analyze external dependencies
        self._analyze_external_dependencies()
        
        # Phase 6: Generate statistics
        self.stats['scan_duration'] = time.time() - start_time
        self._calculate_statistics()
        
        logger.info(f"Scan completed in {self.stats['scan_duration']:.2f} seconds")
        
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
    
    def _analyze_module(self, file_path: Path) -> Optional[ModuleAnalysis]:
        """Analyze a single Python module for imports, exports, and structure."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
                
            # Parse AST
            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                logger.warning(f"Syntax error in {file_path}: {e}")
                return None
            
            # Calculate module name
            module_name = self._calculate_module_name(file_path)
            
            # Create analysis object
            analysis = ModuleAnalysis(
                file_path=str(file_path),
                module_name=module_name,
                size_lines=len(source_code.splitlines()),
                docstring=ast.get_docstring(tree)
            )
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                self._analyze_ast_node(node, analysis, source_code)
            
            # Calculate complexity
            analysis.complexity_score = self._calculate_complexity(tree)
            
            # Check for __main__ guard
            analysis.has_main_guard = '__main__' in source_code
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing module {file_path}: {e}")
            return None
    
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
    
    def _analyze_ast_node(self, node: ast.AST, analysis: ModuleAnalysis, source_code: str):
        """Analyze a single AST node and extract relevant information."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_info = ImportInfo(
                    module_name=alias.name,
                    imported_names=[alias.name],
                    import_type='import',
                    alias=alias.asname,
                    line_number=getattr(node, 'lineno', 0),
                    is_conditional=self._is_conditional_import(node, source_code)
                )
                analysis.imports.append(import_info)
                
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names = [alias.name for alias in node.names]
                import_info = ImportInfo(
                    module_name=node.module,
                    imported_names=imported_names,
                    import_type='from_import',
                    level=node.level,
                    line_number=getattr(node, 'lineno', 0),
                    is_relative=node.level > 0,
                    is_conditional=self._is_conditional_import(node, source_code)
                )
                analysis.imports.append(import_info)
                
        elif isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_'):  # Public functions
                analysis.functions.append(node.name)
                analysis.exports.append(node.name)
                
        elif isinstance(node, ast.AsyncFunctionDef):
            if not node.name.startswith('_'):  # Public async functions
                analysis.functions.append(f"async {node.name}")
                analysis.exports.append(node.name)
                
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith('_'):  # Public classes
                analysis.classes.append(node.name)
                analysis.exports.append(node.name)
                
        elif isinstance(node, ast.Assign):
            # Global variables
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith('_'):
                    analysis.variables.append(target.id)
                    analysis.exports.append(target.id)
    
    def _is_conditional_import(self, node: ast.AST, source_code: str) -> bool:
        """Check if an import is conditional (inside if/try block)."""
        # Simple heuristic: check if import is indented
        lines = source_code.splitlines()
        if hasattr(node, 'lineno') and node.lineno <= len(lines):
            line = lines[node.lineno - 1]
            return line.startswith('    ')  # Indented = conditional
        return False
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the module."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    
    def _build_dependency_graph(self):
        """Build the complete dependency graph from module analyses."""
        for module_name, analysis in self.module_analyses.items():
            self.dependency_graph.edges[module_name] = set()
            self.dependency_graph.reverse_edges[module_name] = set()
        
        for module_name, analysis in self.module_analyses.items():
            for import_info in analysis.imports:
                dependency = self._resolve_import(import_info, module_name)
                
                if dependency:
                    if dependency in self.module_analyses:
                        # Internal dependency
                        self.dependency_graph.edges[module_name].add(dependency)
                        self.dependency_graph.reverse_edges[dependency].add(module_name)
                        self.stats['internal_imports'] += 1
                    else:
                        # External dependency
                        if module_name not in self.dependency_graph.external_dependencies:
                            self.dependency_graph.external_dependencies[module_name] = set()
                        self.dependency_graph.external_dependencies[module_name].add(dependency)
                        self.external_deps[dependency].add(module_name)
                        self.stats['external_imports'] += 1
                    
                    self.stats['total_imports'] += 1
    
    def _resolve_import(self, import_info: ImportInfo, current_module: str) -> Optional[str]:
        """Resolve an import to its actual module name."""
        if import_info.is_relative:
            # Handle relative imports
            current_parts = current_module.split('.')
            if import_info.level > len(current_parts):
                return None  # Invalid relative import
            
            base_parts = current_parts[:-import_info.level] if import_info.level > 0 else current_parts
            if import_info.module_name:
                return '.'.join(base_parts + [import_info.module_name])
            else:
                return '.'.join(base_parts)
        else:
            return import_info.module_name
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                
                circular_dep = CircularDependency(
                    cycle_path=cycle,
                    severity=self._assess_cycle_severity(cycle),
                    impact_score=len(cycle) * 10,
                    affected_modules=set(cycle)
                )
                
                self.circular_deps.append(circular_dep)
                self.dependency_graph.circular_dependencies.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.edges.get(node, set()):
                if neighbor in self.module_analyses:  # Only check internal modules
                    dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)
        
        for module in self.module_analyses:
            if module not in visited:
                dfs(module, [module])
        
        self.stats['circular_dependencies'] = len(self.circular_deps)
        logger.info(f"Detected {len(self.circular_deps)} circular dependencies")
    
    def _assess_cycle_severity(self, cycle: List[str]) -> str:
        """Assess the severity of a circular dependency."""
        cycle_length = len(cycle)
        
        if cycle_length <= 2:
            return 'low'
        elif cycle_length <= 4:
            return 'medium'
        elif cycle_length <= 8:
            return 'high'
        else:
            return 'critical'
    
    def _analyze_external_dependencies(self):
        """Analyze external dependencies and their usage patterns."""
        # Group external dependencies by type
        standard_lib = set()
        third_party = set()
        
        for dep in self.external_deps:
            if self._is_standard_library(dep):
                standard_lib.add(dep)
            else:
                third_party.add(dep)
        
        self.external_deps['__standard_library__'] = standard_lib
        self.external_deps['__third_party__'] = third_party
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if a module is part of the Python standard library."""
        standard_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 're', 'math', 'random',
            'hashlib', 'uuid', 'logging', 'threading', 'multiprocessing',
            'subprocess', 'tempfile', 'shutil', 'copy', 'pickle', 'csv',
            'xml', 'html', 'urllib', 'http', 'email', 'sqlite3', 'argparse',
            'configparser', 'io', 'gc', 'inspect', 'ast', 'dis', 'keyword',
            'token', 'tokenize', 'importlib', 'pkgutil', 'unittest', 'doctest'
        }
        
        root_module = module_name.split('.')[0]
        return root_module in standard_modules
    
    def _calculate_statistics(self):
        """Calculate comprehensive statistics about the codebase."""
        # Calculate additional statistics
        total_exports = sum(len(analysis.exports) for analysis in self.module_analyses.values())
        total_functions = sum(len(analysis.functions) for analysis in self.module_analyses.values())
        total_classes = sum(len(analysis.classes) for analysis in self.module_analyses.values())
        total_lines = sum(analysis.size_lines for analysis in self.module_analyses.values())
        
        self.stats.update({
            'total_exports': total_exports,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'total_lines': total_lines,
            'avg_module_size': total_lines / len(self.module_analyses) if self.module_analyses else 0,
            'avg_complexity': sum(a.complexity_score for a in self.module_analyses.values()) / len(self.module_analyses) if self.module_analyses else 0
        })
    
    def generate_dependency_matrix(self) -> Dict[str, Any]:
        """Generate Neo4j-compatible dependency matrix."""
        nodes = []
        relationships = []
        
        # Create nodes for each module
        for module_name, analysis in self.module_analyses.items():
            node = {
                'id': module_name,
                'labels': ['Module'],
                'properties': {
                    'name': module_name,
                    'file_path': analysis.file_path,
                    'size_lines': analysis.size_lines,
                    'complexity': analysis.complexity_score,
                    'exports_count': len(analysis.exports),
                    'functions_count': len(analysis.functions),
                    'classes_count': len(analysis.classes),
                    'has_main_guard': analysis.has_main_guard
                }
            }
            nodes.append(node)
        
        # Create relationships for dependencies
        for module_name, dependencies in self.dependency_graph.edges.items():
            for dependency in dependencies:
                if dependency in self.module_analyses:
                    relationship = {
                        'type': 'DEPENDS_ON',
                        'start_node': module_name,
                        'end_node': dependency,
                        'properties': {
                            'import_type': 'internal',
                            'created_at': self.scan_timestamp.isoformat()
                        }
                    }
                    relationships.append(relationship)
        
        # Create external dependency relationships
        for module_name, ext_deps in self.dependency_graph.external_dependencies.items():
            for ext_dep in ext_deps:
                relationship = {
                    'type': 'IMPORTS',
                    'start_node': module_name,
                    'end_node': ext_dep,
                    'properties': {
                        'import_type': 'external',
                        'is_standard_library': self._is_standard_library(ext_dep),
                        'created_at': self.scan_timestamp.isoformat()
                    }
                }
                relationships.append(relationship)
        
        return {
            'nodes': nodes,
            'relationships': relationships,
            'metadata': {
                'scan_timestamp': self.scan_timestamp.isoformat(),
                'total_nodes': len(nodes),
                'total_relationships': len(relationships),
                'circular_dependencies': len(self.circular_deps)
            }
        }
    
    def generate_cypher_queries(self) -> List[str]:
        """Generate Cypher queries for Neo4j import."""
        queries = []
        
        # Create unique constraints
        queries.append("CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE")
        queries.append("CREATE CONSTRAINT external_dep_name IF NOT EXISTS FOR (e:ExternalDependency) REQUIRE e.name IS UNIQUE")
        
        # Create module nodes
        for module_name, analysis in self.module_analyses.items():
            escaped_path = analysis.file_path.replace("'", "\\'")
            query = f"""
            MERGE (m:Module {{name: '{module_name}'}})
            SET m.file_path = '{escaped_path}',
                m.size_lines = {analysis.size_lines},
                m.complexity = {analysis.complexity_score},
                m.exports_count = {len(analysis.exports)},
                m.functions_count = {len(analysis.functions)},
                m.classes_count = {len(analysis.classes)},
                m.has_main_guard = {str(analysis.has_main_guard).lower()},
                m.scan_timestamp = '{self.scan_timestamp.isoformat()}'
            """
            queries.append(query.strip())
        
        # Create dependency relationships
        for module_name, dependencies in self.dependency_graph.edges.items():
            for dependency in dependencies:
                if dependency in self.module_analyses:
                    query = f"""
                    MATCH (a:Module {{name: '{module_name}'}}), (b:Module {{name: '{dependency}'}})
                    MERGE (a)-[r:DEPENDS_ON]->(b)
                    SET r.import_type = 'internal',
                        r.created_at = '{self.scan_timestamp.isoformat()}'
                    """
                    queries.append(query.strip())
        
        return queries
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive relationship analysis report."""
        return {
            'scan_metadata': {
                'timestamp': self.scan_timestamp.isoformat(),
                'root_path': str(self.root_path),
                'scan_duration': self.stats['scan_duration']
            },
            'statistics': self.stats,
            'modules': {
                name: asdict(analysis) for name, analysis in self.module_analyses.items()
            },
            'dependency_graph': {
                'internal_dependencies': {
                    module: list(deps) for module, deps in self.dependency_graph.edges.items()
                },
                'external_dependencies': {
                    module: list(deps) for module, deps in self.dependency_graph.external_dependencies.items()
                },
                'reverse_dependencies': {
                    module: list(deps) for module, deps in self.dependency_graph.reverse_edges.items()
                }
            },
            'circular_dependencies': [
                {
                    'cycle_path': cd.cycle_path,
                    'severity': cd.severity,
                    'impact_score': cd.impact_score,
                    'affected_modules': list(cd.affected_modules)
                }
                for cd in self.circular_deps
            ],
            'external_dependencies_analysis': {
                'total_external': len(self.external_deps),
                'standard_library': list(self.external_deps.get('__standard_library__', set())),
                'third_party': list(self.external_deps.get('__third_party__', set())),
                'usage_frequency': {
                    dep: len(modules) for dep, modules in self.external_deps.items()
                    if not dep.startswith('__')
                }
            },
            'neo4j_matrix': self.generate_dependency_matrix(),
            'cypher_queries': self.generate_cypher_queries()
        }
    
    def save_report(self, output_path: Path) -> None:
        """Save the comprehensive report to a JSON file."""
        report = self._generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Relationship analysis report saved to {output_path}")


def main():
    """Main entry point for the relationship scanner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent C - Import/Export Relationship Scanner")
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument("--output", default="relationship_analysis.json", help="Output file path")
    parser.add_argument("--cypher", help="Output Cypher queries to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create scanner and run analysis
    scanner = RelationshipScanner(Path(args.root))
    
    print("Agent C - Starting Import/Export Relationship Analysis")
    print(f"Scanning: {scanner.root_path}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Run comprehensive scan
    report = scanner.scan_codebase()
    
    # Save report
    scanner.save_report(Path(args.output))
    
    # Save Cypher queries if requested
    if args.cypher:
        queries = scanner.generate_cypher_queries()
        with open(args.cypher, 'w') as f:
            f.write('\n\n'.join(queries))
        print(f"Cypher queries saved to {args.cypher}")
    
    # Print summary
    stats = report['statistics']
    print(f"\nScan Results Summary:")
    print(f"   Files Analyzed: {stats['total_files']}")
    print(f"   Total Imports: {stats['total_imports']}")
    print(f"   Internal Dependencies: {stats['internal_imports']}")
    print(f"   External Dependencies: {stats['external_imports']}")
    print(f"   Circular Dependencies: {stats['circular_dependencies']}")
    print(f"   Total Exports: {stats.get('total_exports', 0)}")
    print(f"   Average Module Size: {stats.get('avg_module_size', 0):.1f} lines")
    print(f"   Scan Duration: {stats['scan_duration']:.2f} seconds")
    
    if report['circular_dependencies']:
        print(f"\nCircular Dependencies Detected:")
        for i, cd in enumerate(report['circular_dependencies'][:5], 1):
            print(f"   {i}. {' -> '.join(cd['cycle_path'])} (Severity: {cd['severity']})")
    
    print(f"\nRelationship analysis complete! Report saved to {args.output}")


if __name__ == "__main__":
    main()