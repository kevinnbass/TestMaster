from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
#!/usr/bin/env python3
"""
Unified Code Analysis Master
============================

Consolidates ALL code analysis functionality into one powerful tool.

Consolidated scripts:
- analyze_components.py
- dependency_analyzer.py
- api_documenter.py
... and related analysis functionality

Author: Agent E - Infrastructure Consolidation
"""

import os
import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import importlib.util
import inspect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of code analysis."""
    STRUCTURE = "structure"  # Code structure analysis
    DEPENDENCIES = "dependencies"  # Dependency analysis
    COMPLEXITY = "complexity"  # Complexity metrics
    QUALITY = "quality"  # Code quality analysis
    API = "api"  # API documentation
    SECURITY = "security"  # Security analysis
    PERFORMANCE = "performance"  # Performance analysis


@dataclass
class CodeMetrics:
    """Code metrics for a module."""
    lines_of_code: int = 0
    lines_of_comments: int = 0
    lines_of_docstrings: int = 0
    cyclomatic_complexity: int = 0
    maintainability_index: float = 0.0
    
    # Structural metrics
    num_classes: int = 0
    num_functions: int = 0
    num_methods: int = 0
    max_depth: int = 0
    
    # Quality metrics
    docstring_coverage: float = 0.0
    type_hint_coverage: float = 0.0
    test_coverage: float = 0.0


@dataclass
class DependencyInfo:
    """Dependency information."""
    imports: List[str] = field(default_factory=list)
    external_deps: List[str] = field(default_factory=list)
    internal_deps: List[str] = field(default_factory=list)
    circular_deps: List[str] = field(default_factory=list)
    unused_imports: List[str] = field(default_factory=list)


@dataclass
class APIInfo:
    """API information for a module."""
    module_name: str
    docstring: Optional[str] = None
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis report for a module."""
    file_path: Path
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    dependencies: DependencyInfo = field(default_factory=DependencyInfo)
    api_info: Optional[APIInfo] = None
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class CodeAnalysisMaster:
    """
    Master code analysis system consolidating all analysis capabilities.
    """
    
    def __init__(self, analysis_types: Optional[Set[AnalysisType]] = None):
        self.analysis_types = analysis_types or {
            AnalysisType.STRUCTURE,
            AnalysisType.DEPENDENCIES,
            AnalysisType.COMPLEXITY,
            AnalysisType.QUALITY,
            AnalysisType.API
        }
        
        self.reports: Dict[Path, AnalysisReport] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Statistics
        self.stats = {
            "files_analyzed": 0,
            "total_lines": 0,
            "total_classes": 0,
            "total_functions": 0,
            "avg_complexity": 0.0,
            "issues_found": 0
        }
    
    def analyze(self, target: str) -> Dict[Path, AnalysisReport]:
        """Analyze code for target."""
        logger.info(f"Starting code analysis for: {target}")
        
        # Find target files
        target_files = self._find_target_files(target)
        if not target_files:
            logger.error(f"No Python files found for: {target}")
            return {}
        
        logger.info(f"Analyzing {len(target_files)} files")
        
        # Analyze each file
        for file_path in target_files:
            report = self._analyze_file(file_path)
            self.reports[file_path] = report
        
        # Build dependency graph
        if AnalysisType.DEPENDENCIES in self.analysis_types:
            self._build_dependency_graph()
            self._detect_circular_dependencies()
        
        # Update statistics
        self._update_statistics()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Print summary
        self._print_summary()
        
        return self.reports
    
    def _find_target_files(self, target: str) -> List[Path]:
        """Find Python files to analyze."""
        target_path = Path(target)
        
        if target_path.is_file() and target_path.suffix == ".py":
            return [target_path]
        elif target_path.is_dir():
            return list(target_path.rglob("*.py"))
        else:
            return list(Path(".").rglob(target))
    
    def _analyze_file(self, file_path: Path) -> AnalysisReport:
        """Analyze a single file."""
        logger.debug(f"Analyzing: {file_path}")
        
        report = AnalysisReport(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Perform different types of analysis
            if AnalysisType.STRUCTURE in self.analysis_types:
                self._analyze_structure(tree, source, report)
            
            if AnalysisType.DEPENDENCIES in self.analysis_types:
                self._analyze_dependencies(tree, report)
            
            if AnalysisType.COMPLEXITY in self.analysis_types:
                self._analyze_complexity(tree, report)
            
            if AnalysisType.QUALITY in self.analysis_types:
                self._analyze_quality(tree, source, report)
            
            if AnalysisType.API in self.analysis_types:
                self._analyze_api(tree, file_path, report)
            
            if AnalysisType.SECURITY in self.analysis_types:
                self._analyze_security(tree, report)
            
            if AnalysisType.PERFORMANCE in self.analysis_types:
                self._analyze_performance(tree, report)
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            report.issues.append({
                "type": "parse_error",
                "message": str(e)
            })
        
        return report
    
    def _analyze_structure(self, tree: ast.AST, source: str, report: AnalysisReport):
        """Analyze code structure."""
        lines = source.splitlines()
        
        # Count lines
        report.metrics.lines_of_code = len(lines)
        
        # Count comments and docstrings
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        report.metrics.lines_of_comments = comment_lines
        
        # Count classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                report.metrics.num_classes += 1
                
                # Count methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        report.metrics.num_methods += 1
            
            elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    report.metrics.num_functions += 1
        
        # Calculate depth
        report.metrics.max_depth = self._calculate_max_depth(tree)
    
    def _analyze_dependencies(self, tree: ast.AST, report: AnalysisReport):
        """Analyze dependencies."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    report.dependencies.imports.append(module_name)
                    
                    # Classify as internal or external
                    if self._is_external_module(module_name):
                        report.dependencies.external_deps.append(module_name)
                    else:
                        report.dependencies.internal_deps.append(module_name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module
                    report.dependencies.imports.append(module_name)
                    
                    if self._is_external_module(module_name):
                        report.dependencies.external_deps.append(module_name)
                    else:
                        report.dependencies.internal_deps.append(module_name)
    
    def _analyze_complexity(self, tree: ast.AST, report: AnalysisReport):
        """Analyze code complexity."""
        # Calculate cyclomatic complexity
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        report.metrics.cyclomatic_complexity = complexity
        
        # Calculate maintainability index (simplified)
        # MI = 171 - 5.2 * ln(V) - 0.23 * C - 16.2 * ln(L)
        # Where V = Halstead Volume, C = Cyclomatic Complexity, L = Lines of Code
        
        if report.metrics.lines_of_code > 0:
            import math
            L = report.metrics.lines_of_code
            C = complexity
            # Simplified calculation
            MI = max(0, min(100, 171 - 0.23 * C - 16.2 * math.log(L)))
            report.metrics.maintainability_index = MI
    
    def _analyze_quality(self, tree: ast.AST, source: str, report: AnalysisReport):
        """Analyze code quality."""
        total_functions = 0
        functions_with_docstrings = 0
        functions_with_type_hints = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                
                # Check for docstring
                if ast.get_docstring(node):
                    functions_with_docstrings += 1
                
                # Check for type hints
                if node.returns or any(arg.annotation for arg in node.args.args):
                    functions_with_type_hints += 1
        
        if total_functions > 0:
            report.metrics.docstring_coverage = (functions_with_docstrings / total_functions) * 100
            report.metrics.type_hint_coverage = (functions_with_type_hints / total_functions) * 100
        
        # Check for common quality issues
        if report.metrics.cyclomatic_complexity > 10:
            report.issues.append({
                "type": "high_complexity",
                "message": f"High cyclomatic complexity: {report.metrics.cyclomatic_complexity}"
            })
        
        if report.metrics.docstring_coverage < 80:
            report.suggestions.append("Improve docstring coverage (currently {:.1f}%)".format(
                report.metrics.docstring_coverage
            ))
    
    def _analyze_api(self, tree: ast.AST, file_path: Path, report: AnalysisReport):
        """Analyze API and generate documentation."""
        module_name = file_path.stem
        
        api_info = APIInfo(
            module_name=module_name,
            docstring=ast.get_docstring(tree)
        )
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "attributes": []
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            "name": item.name,
                            "docstring": ast.get_docstring(item),
                            "args": [arg.arg for arg in item.args.args],
                            "returns": ast.unparse(item.returns) if item.returns else None
                        }
                        class_info["methods"].append(method_info)
                
                api_info.classes.append(class_info)
            
            elif isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "args": [arg.arg for arg in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns else None
                }
                api_info.functions.append(func_info)
            
            elif isinstance(node, ast.Assign):
                # Constants (uppercase names)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        api_info.constants.append({
                            "name": target.id,
                            "value": ast.unparse(node.value) if node.value else None
                        })
        
        # Check for __all__
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            api_info.exports = [
                                ast.literal_SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval(elt) for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
        
        report.api_info = api_info
    
    def _analyze_security(self, tree: ast.AST, report: AnalysisReport):
        """Analyze security issues."""
        # Check for common security issues
        for node in ast.walk(tree):
            # Check for exec/eval usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval']:
                        report.issues.append({
                            "type": "security",
                            "severity": "high",
                            "message": f"Use of {node.func.id}() is a security risk"
                        })
            
            # Check for hardcoded secrets (simplified)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id.lower()
                        if any(secret in name for secret in ['password', 'secret', 'key', 'token']):
                            if isinstance(node.value, ast.Constant):
                                report.issues.append({
                                    "type": "security",
                                    "severity": "critical",
                                    "message": f"Possible hardcoded secret: {target.id}"
                                })
    
    def _analyze_performance(self, tree: ast.AST, report: AnalysisReport):
        """Analyze performance issues."""
        # Check for common performance issues
        for node in ast.walk(tree):
            # Check for inefficient string concatenation in loops
            if isinstance(node, ast.For):
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.AugAssign):
                        if isinstance(inner_node.op, ast.Add):
                            if isinstance(inner_node.target, ast.Name):
                                # Simplified check for string concatenation
                                report.suggestions.append(
                                    "Consider using list.append() and ''.join() for string concatenation in loops"
                                )
                                break
    
    def _is_external_module(self, module_name: str) -> bool:
        """Check if module is external (from pip)."""
        # Simplified check - assumes modules without dots are external
        # unless they're built-in
        import sys
        
        if module_name in sys.builtin_module_names:
            return False
        
        if '.' not in module_name:
            # Try to find in site-packages
            try:
                spec = importlib.util.find_spec(module_name)
                if spec and spec.origin:
                    return 'site-packages' in spec.origin
            except:
                pass
        
        return False
    
    def _calculate_max_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.FunctionDef, ast.ClassDef)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)
    
    def _build_dependency_graph(self):
        """Build dependency graph from all analyzed files."""
        for file_path, report in self.reports.items():
            module_name = file_path.stem
            
            if module_name not in self.dependency_graph:
                self.dependency_graph[module_name] = set()
            
            for dep in report.dependencies.internal_deps:
                self.dependency_graph[module_name].add(dep)
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies."""
        def find_cycle(graph, start, visited, rec_stack, path):
            visited.add(start)
            rec_stack.add(start)
            path.append(start)
            
            for neighbor in graph.get(start, []):
                if neighbor not in visited:
                    cycle = find_cycle(graph, neighbor, visited, rec_stack, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            path.pop()
            rec_stack.remove(start)
            return None
        
        visited = set()
        
        for module in self.dependency_graph:
            if module not in visited:
                rec_stack = set()
                path = []
                cycle = find_cycle(self.dependency_graph, module, visited, rec_stack, path)
                
                if cycle:
                    # Add circular dependency to affected reports
                    for file_path, report in self.reports.items():
                        if file_path.stem in cycle:
                            report.dependencies.circular_deps = cycle
                            report.issues.append({
                                "type": "circular_dependency",
                                "message": f"Circular dependency detected: {' -> '.join(cycle)}"
                            })
    
    def _update_statistics(self):
        """Update analysis statistics."""
        self.stats["files_analyzed"] = len(self.reports)
        
        for report in self.reports.values():
            self.stats["total_lines"] += report.metrics.lines_of_code
            self.stats["total_classes"] += report.metrics.num_classes
            self.stats["total_functions"] += report.metrics.num_functions
            self.stats["issues_found"] += len(report.issues)
        
        if self.reports:
            total_complexity = sum(r.metrics.cyclomatic_complexity for r in self.reports.values())
            self.stats["avg_complexity"] = total_complexity / len(self.reports)
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis."""
        for file_path, report in self.reports.items():
            # Complexity recommendations
            if report.metrics.cyclomatic_complexity > 20:
                report.suggestions.append("Consider refactoring to reduce complexity")
            
            # Size recommendations
            if report.metrics.lines_of_code > 500:
                report.suggestions.append("Consider splitting this module into smaller files")
            
            # Dependency recommendations
            if len(report.dependencies.external_deps) > 20:
                report.suggestions.append("High number of external dependencies - consider reducing")
            
            # Quality recommendations
            if report.metrics.maintainability_index < 50:
                report.suggestions.append("Low maintainability index - refactoring recommended")
    
    def _print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("CODE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Files analyzed: {self.stats['files_analyzed']}")
        print(f"Total lines of code: {self.stats['total_lines']:,}")
        print(f"Total classes: {self.stats['total_classes']}")
        print(f"Total functions: {self.stats['total_functions']}")
        print(f"Average complexity: {self.stats['avg_complexity']:.1f}")
        print(f"Issues found: {self.stats['issues_found']}")
        
        # Show top issues
        all_issues = []
        for file_path, report in self.reports.items():
            for issue in report.issues:
                all_issues.append((file_path, issue))
        
        if all_issues:
            print("\nTop Issues:")
            for file_path, issue in all_issues[:5]:
                print(f"  - {file_path.name}: {issue['message']}")
        
        print("="*60)
    
    def generate_documentation(self, output_dir: Path):
        """Generate API documentation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path, report in self.reports.items():
            if report.api_info:
                doc_file = output_dir / f"{report.api_info.module_name}.md"
                
                with open(doc_file, 'w') as f:
                    f.write(f"# {report.api_info.module_name}\n\n")
                    
                    if report.api_info.docstring:
                        f.write(f"{report.api_info.docstring}\n\n")
                    
                    if report.api_info.classes:
                        f.write("## Classes\n\n")
                        for cls in report.api_info.classes:
                            f.write(f"### {cls['name']}\n\n")
                            if cls['docstring']:
                                f.write(f"{cls['docstring']}\n\n")
                            
                            if cls['methods']:
                                f.write("#### Methods\n\n")
                                for method in cls['methods']:
                                    f.write(f"- **{method['name']}**")
                                    if method['args']:
                                        f.write(f"({', '.join(method['args'])})")
                                    f.write("\n")
                    
                    if report.api_info.functions:
                        f.write("## Functions\n\n")
                        for func in report.api_info.functions:
                            f.write(f"### {func['name']}\n\n")
                            if func['docstring']:
                                f.write(f"{func['docstring']}\n\n")
                
                logger.info(f"Generated documentation: {doc_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Code Analysis Master")
    parser.add_argument("target", help="Target file, directory, or pattern")
    parser.add_argument("--types", nargs="+",
                       choices=[t.value for t in AnalysisType],
                       help="Analysis types to perform")
    parser.add_argument("--generate-docs", action="store_true",
                       help="Generate API documentation")
    parser.add_argument("--output", default="analysis_output",
                       help="Output directory for documentation")
    
    args = parser.parse_args()
    
    analysis_types = {AnalysisType(t) for t in args.types} if args.types else None
    
    analyzer = CodeAnalysisMaster(analysis_types)
    reports = analyzer.analyze(args.target)
    
    if args.generate_docs:
        analyzer.generate_documentation(Path(args.output))


if __name__ == "__main__":
    main()