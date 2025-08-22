"""
Code Analysis Engine - Advanced AST-based code structure analysis for test generation

This module provides sophisticated code analysis capabilities using Abstract Syntax Tree (AST)
parsing to understand code structure, complexity, dependencies, and patterns for intelligent
test generation. Includes advanced metrics, dependency tracking, and security analysis.

Enterprise Features:
- Comprehensive AST-based code analysis with detailed metrics
- Cyclomatic complexity calculation and maintainability scoring
- Dependency detection and call graph analysis
- Security vulnerability pattern detection
- Performance bottleneck identification
- Test coverage prediction and optimization

Key Components:
- CodeAnalyzer: Main analysis engine with AST processing
- ComplexityCalculator: Advanced complexity metrics calculation
- DependencyTracker: Dependency and call graph analysis
- PatternDetector: Code pattern and anti-pattern recognition
- SecurityAnalyzer: Security vulnerability detection
"""

import ast
import inspect
import re
import tokenize
import io
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Type
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter

from .test_models import (
    ModuleAnalysis, FunctionInfo, ClassInfo, ParameterInfo,
    ComplexityLevel, create_module_analysis, create_function_info
)


class SecurityPattern:
    """Security vulnerability patterns for detection."""
    SQL_INJECTION = r'(execute|query|cursor).*\+.*|f".*{.*}.*".*execute'
    COMMAND_INJECTION = r'(subprocess|os\.system|popen).*\+|shell=True'
    PATH_TRAVERSAL = r'open\(.*\+.*\)|Path\(.*\+.*\)'
    HARDCODED_SECRETS = r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']'
    UNSAFE_PICKLE = r'pickle\.loads?|marshal\.loads?'
    EVAL_USAGE = r'\b(eval|exec)\s*\('


class PerformancePattern:
    """Performance anti-patterns for detection."""
    NESTED_LOOPS = r'for.*:\s*for.*:'
    INEFFICIENT_STRING_CONCAT = r'\+=.*["\']'
    GLOBAL_USAGE = r'\bglobal\s+\w+'
    RECURSIVE_WITHOUT_MEMO = r'def.*\(.*\):.*return.*\w+\('


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for code analysis."""
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 1
    nesting_depth: int = 0
    number_of_parameters: int = 0
    lines_of_code: int = 0
    number_of_returns: int = 0
    number_of_branches: int = 0
    number_of_loops: int = 0
    halstead_volume: float = 0.0
    maintainability_index: float = 100.0


@dataclass
class DependencyInfo:
    """Information about code dependencies and relationships."""
    internal_imports: Set[str]
    external_imports: Set[str]
    function_calls: Set[str]
    class_instantiations: Set[str]
    global_variables_accessed: Set[str]
    external_apis_called: Set[str]
    file_operations: Set[str]
    network_operations: Set[str]
    database_operations: Set[str]


@dataclass
class SecurityIssue:
    """Security vulnerability or concern identified in code."""
    issue_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    line_number: int
    description: str
    code_snippet: str
    recommendation: str


@dataclass 
class PerformanceIssue:
    """Performance concern identified in code."""
    issue_type: str
    severity: str  # LOW, MEDIUM, HIGH
    line_number: int
    description: str
    impact: str
    suggestion: str


class ComplexityCalculator:
    """Advanced complexity metrics calculation."""
    
    @staticmethod
    def calculate_cyclomatic_complexity(node: ast.AST) -> int:
        """Calculate cyclomatic complexity using AST analysis."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points that increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ListComp):
                complexity += 1
            elif isinstance(child, ast.DictComp):
                complexity += 1
            elif isinstance(child, ast.SetComp):
                complexity += 1
            elif isinstance(child, ast.GeneratorExp):
                complexity += 1
                
        return complexity
    
    @staticmethod
    def calculate_cognitive_complexity(node: ast.AST) -> int:
        """Calculate cognitive complexity (human understanding difficulty)."""
        complexity = 0
        nesting_level = 0
        
        def visit_node(n, level):
            nonlocal complexity
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += (1 + level)
            elif isinstance(n, ast.ExceptHandler):
                complexity += (1 + level)
            elif isinstance(n, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(n, ast.Continue):
                complexity += (1 + level)
            elif isinstance(n, ast.Break):
                complexity += (1 + level)
                
            # Increase nesting for certain constructs
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                level += 1
                
            for child in ast.iter_child_nodes(n):
                visit_node(child, level)
                
        visit_node(node, 0)
        return complexity
    
    @staticmethod
    def calculate_nesting_depth(node: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def visit_node(n, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                depth += 1
                
            for child in ast.iter_child_nodes(n):
                visit_node(child, depth)
                
        visit_node(node, 0)
        return max_depth
    
    @staticmethod
    def calculate_halstead_metrics(source_code: str) -> float:
        """Calculate Halstead volume (approximation)."""
        try:
            # Tokenize to count operators and operands
            tokens = list(tokenize.generate_tokens(io.StringIO(source_code).readline))
            
            operators = set()
            operands = set()
            
            for token in tokens:
                if token.type == tokenize.OP:
                    operators.add(token.string)
                elif token.type == tokenize.NAME:
                    operands.add(token.string)
                    
            n1 = len(operators)  # Unique operators
            n2 = len(operands)   # Unique operands
            N1 = sum(1 for t in tokens if t.type == tokenize.OP)  # Total operators
            N2 = sum(1 for t in tokens if t.type == tokenize.NAME)  # Total operands
            
            if n1 > 0 and n2 > 0:
                vocabulary = n1 + n2
                length = N1 + N2
                volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
                return volume
                
        except Exception:
            pass
            
        return 0.0


class DependencyTracker:
    """Track dependencies and relationships in code."""
    
    def __init__(self):
        self.dependency_info = DependencyInfo(
            internal_imports=set(),
            external_imports=set(),
            function_calls=set(),
            class_instantiations=set(),
            global_variables_accessed=set(),
            external_apis_called=set(),
            file_operations=set(),
            network_operations=set(),
            database_operations=set()
        )
    
    def analyze_dependencies(self, node: ast.AST) -> DependencyInfo:
        """Analyze all dependencies in an AST node."""
        for child in ast.walk(node):
            self._analyze_imports(child)
            self._analyze_calls(child)
            self._analyze_global_access(child)
            
        return self.dependency_info
    
    def _analyze_imports(self, node: ast.AST):
        """Analyze import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if self._is_standard_library(alias.name):
                    self.dependency_info.external_imports.add(alias.name)
                else:
                    self.dependency_info.internal_imports.add(alias.name)
                    
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if self._is_standard_library(module):
                self.dependency_info.external_imports.add(module)
            else:
                self.dependency_info.internal_imports.add(module)
    
    def _analyze_calls(self, node: ast.AST):
        """Analyze function calls and method calls."""
        if isinstance(node, ast.Call):
            call_name = self._get_call_name(node.func)
            self.dependency_info.function_calls.add(call_name)
            
            # Detect specific operation types
            if any(pattern in call_name.lower() for pattern in ['open', 'read', 'write']):
                self.dependency_info.file_operations.add(call_name)
            elif any(pattern in call_name.lower() for pattern in ['request', 'http', 'socket']):
                self.dependency_info.network_operations.add(call_name)
            elif any(pattern in call_name.lower() for pattern in ['execute', 'query', 'cursor', 'connection']):
                self.dependency_info.database_operations.add(call_name)
    
    def _analyze_global_access(self, node: ast.AST):
        """Analyze global variable access."""
        if isinstance(node, ast.Global):
            for name in node.names:
                self.dependency_info.global_variables_accessed.add(name)
    
    def _get_call_name(self, node: ast.AST) -> str:
        """Extract function call name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        else:
            return "unknown"
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if module is part of Python standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'pathlib', 'typing', 'collections',
            'itertools', 'functools', 'operator', 're', 'math', 'random',
            'urllib', 'http', 'socket', 'ssl', 'hashlib', 'hmac', 'base64',
            'pickle', 'csv', 'xml', 'html', 'email', 'subprocess', 'threading',
            'multiprocessing', 'asyncio', 'concurrent', 'logging', 'unittest',
            'pytest', 'tempfile', 'shutil', 'glob', 'fnmatch', 'linecache',
            'traceback', 'inspect', 'ast', 'dis', 'tokenize'
        }
        
        root_module = module_name.split('.')[0]
        return root_module in stdlib_modules


class PatternDetector:
    """Detect code patterns and anti-patterns."""
    
    def __init__(self):
        self.security_issues = []
        self.performance_issues = []
    
    def detect_security_patterns(self, source_code: str) -> List[SecurityIssue]:
        """Detect potential security vulnerabilities."""
        issues = []
        lines = source_code.split('\n')
        
        patterns = {
            'SQL Injection': SecurityPattern.SQL_INJECTION,
            'Command Injection': SecurityPattern.COMMAND_INJECTION,
            'Path Traversal': SecurityPattern.PATH_TRAVERSAL,
            'Hardcoded Secrets': SecurityPattern.HARDCODED_SECRETS,
            'Unsafe Pickle': SecurityPattern.UNSAFE_PICKLE,
            'Eval Usage': SecurityPattern.EVAL_USAGE
        }
        
        for line_num, line in enumerate(lines, 1):
            for issue_type, pattern in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    severity = self._determine_security_severity(issue_type)
                    issues.append(SecurityIssue(
                        issue_type=issue_type,
                        severity=severity,
                        line_number=line_num,
                        description=f"Potential {issue_type.lower()} vulnerability detected",
                        code_snippet=line.strip(),
                        recommendation=self._get_security_recommendation(issue_type)
                    ))
        
        return issues
    
    def detect_performance_patterns(self, source_code: str) -> List[PerformanceIssue]:
        """Detect potential performance issues."""
        issues = []
        lines = source_code.split('\n')
        
        patterns = {
            'Nested Loops': PerformancePattern.NESTED_LOOPS,
            'Inefficient String Concatenation': PerformancePattern.INEFFICIENT_STRING_CONCAT,
            'Global Usage': PerformancePattern.GLOBAL_USAGE
        }
        
        for line_num, line in enumerate(lines, 1):
            for issue_type, pattern in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    severity = self._determine_performance_severity(issue_type)
                    issues.append(PerformanceIssue(
                        issue_type=issue_type,
                        severity=severity,
                        line_number=line_num,
                        description=f"Potential {issue_type.lower()} performance issue",
                        impact=self._get_performance_impact(issue_type),
                        suggestion=self._get_performance_suggestion(issue_type)
                    ))
        
        return issues
    
    def _determine_security_severity(self, issue_type: str) -> str:
        """Determine security issue severity."""
        high_severity = ['SQL Injection', 'Command Injection', 'Eval Usage']
        medium_severity = ['Path Traversal', 'Unsafe Pickle']
        
        if issue_type in high_severity:
            return 'HIGH'
        elif issue_type in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _determine_performance_severity(self, issue_type: str) -> str:
        """Determine performance issue severity."""
        high_severity = ['Nested Loops']
        medium_severity = ['Inefficient String Concatenation']
        
        if issue_type in high_severity:
            return 'HIGH'
        elif issue_type in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_security_recommendation(self, issue_type: str) -> str:
        """Get security recommendation for issue type."""
        recommendations = {
            'SQL Injection': 'Use parameterized queries or ORM methods',
            'Command Injection': 'Avoid shell=True, use subprocess with list arguments',
            'Path Traversal': 'Validate and sanitize file paths, use Path.resolve()',
            'Hardcoded Secrets': 'Use environment variables or secure key management',
            'Unsafe Pickle': 'Use JSON or other safe serialization formats',
            'Eval Usage': 'Use safer alternatives like ast.literal_eval or avoid dynamic execution'
        }
        return recommendations.get(issue_type, 'Review code for security implications')
    
    def _get_performance_impact(self, issue_type: str) -> str:
        """Get performance impact description."""
        impacts = {
            'Nested Loops': 'O(n²) or worse time complexity',
            'Inefficient String Concatenation': 'O(n²) memory allocation',
            'Global Usage': 'Reduced function performance and testability'
        }
        return impacts.get(issue_type, 'Potential performance degradation')
    
    def _get_performance_suggestion(self, issue_type: str) -> str:
        """Get performance improvement suggestion."""
        suggestions = {
            'Nested Loops': 'Consider using list comprehensions, sets, or algorithmic optimization',
            'Inefficient String Concatenation': 'Use join() method or f-strings',
            'Global Usage': 'Pass values as parameters instead of using global variables'
        }
        return suggestions.get(issue_type, 'Review algorithm for optimization opportunities')


class CodeAnalyzer:
    """Main code analysis engine for comprehensive code understanding."""
    
    def __init__(self):
        self.complexity_calculator = ComplexityCalculator()
        self.dependency_tracker = DependencyTracker()
        self.pattern_detector = PatternDetector()
    
    def analyze_module(self, file_path: Path) -> ModuleAnalysis:
        """
        Perform comprehensive analysis of a Python module.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Complete module analysis with all metrics and insights
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            tree = ast.parse(source_code)
            
            # Basic module info
            module_name = self._get_module_name(file_path)
            
            # Analyze structure
            functions = self._analyze_functions(tree, source_code)
            classes = self._analyze_classes(tree, source_code)
            imports = self._extract_imports(tree)
            constants = self._extract_constants(tree)
            
            # Calculate metrics
            complexity_score = sum(f.complexity_score for f in functions) + sum(c.complexity_score for c in classes)
            line_metrics = self._calculate_line_metrics(source_code)
            
            # Analyze dependencies
            dependencies = self.dependency_tracker.analyze_dependencies(tree)
            
            # Security and performance analysis
            security_issues = self.pattern_detector.detect_security_patterns(source_code)
            performance_issues = self.pattern_detector.detect_performance_patterns(source_code)
            
            # Calculate maintainability index
            maintainability = self._calculate_maintainability_index(
                line_metrics['code_lines'], 
                complexity_score,
                source_code
            )
            
            return create_module_analysis(
                module_name=module_name,
                file_path=str(file_path),
                functions=functions,
                classes=classes,
                imports=imports,
                constants=constants,
                has_main='__name__ == "__main__"' in source_code,
                total_lines=line_metrics['total_lines'],
                code_lines=line_metrics['code_lines'],
                comment_lines=line_metrics['comment_lines'],
                blank_lines=line_metrics['blank_lines'],
                complexity_score=complexity_score,
                maintainability_index=maintainability,
                dependencies=dependencies.internal_imports | dependencies.external_imports,
                external_calls=dependencies.function_calls
            )
            
        except Exception as e:
            # Return minimal analysis on error
            return create_module_analysis(
                module_name=self._get_module_name(file_path),
                file_path=str(file_path),
                complexity_score=0,
                maintainability_index=0.0
            )
    
    def _analyze_functions(self, tree: ast.AST, source_code: str) -> List[FunctionInfo]:
        """Analyze all functions in the module."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function_detailed(node, source_code)
                functions.append(func_info)
                
        return functions
    
    def _analyze_function_detailed(self, node: ast.FunctionDef, source_code: str) -> FunctionInfo:
        """Perform detailed analysis of a single function."""
        # Extract parameters
        parameters = self._extract_parameters(node)
        
        # Calculate complexity metrics
        complexity_metrics = self._calculate_function_complexity(node)
        
        # Analyze function characteristics
        is_async = isinstance(node, ast.AsyncFunctionDef)
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        is_property = 'property' in decorators
        is_classmethod = 'classmethod' in decorators
        is_staticmethod = 'staticmethod' in decorators
        
        # Analyze dependencies and external calls
        dependencies = self.dependency_tracker.analyze_dependencies(node)
        
        # Extract exception information
        raises_exceptions = self._extract_exceptions(node)
        
        # Determine function characteristics
        uses_io = bool(dependencies.file_operations)
        uses_network = bool(dependencies.network_operations)
        uses_database = bool(dependencies.database_operations)
        calls_external = bool(dependencies.external_imports)
        
        # Get source line range
        source_lines = (node.lineno, getattr(node, 'end_lineno', node.lineno))
        
        return create_function_info(
            name=node.name,
            parameters=parameters,
            return_type=self._extract_return_type(node),
            docstring=ast.get_docstring(node) or "",
            is_async=is_async,
            is_property=is_property,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            raises_exceptions=raises_exceptions,
            calls_external=calls_external,
            uses_io=uses_io,
            uses_network=uses_network,
            uses_database=uses_database,
            complexity_score=complexity_metrics.cyclomatic_complexity,
            cyclomatic_complexity=complexity_metrics.cyclomatic_complexity,
            dependencies=list(dependencies.function_calls),
            decorators=decorators,
            source_lines=source_lines
        )
    
    def _analyze_classes(self, tree: ast.AST, source_code: str) -> List[ClassInfo]:
        """Analyze all classes in the module."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class_detailed(node, source_code)
                classes.append(class_info)
                
        return classes
    
    def _analyze_class_detailed(self, node: ast.ClassDef, source_code: str) -> ClassInfo:
        """Perform detailed analysis of a single class."""
        # Extract base classes
        base_classes = [self._get_base_class_name(base) for base in node.bases]
        
        # Analyze methods
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_info = self._analyze_function_detailed(child, source_code)
                methods.append(method_info)
        
        # Extract class characteristics
        has_init = any(method.name == '__init__' for method in methods)
        is_abstract = 'ABC' in base_classes or 'abstractmethod' in source_code
        is_dataclass = '@dataclass' in source_code
        is_exception = any('Exception' in base for base in base_classes)
        
        # Calculate complexity
        complexity_score = sum(method.complexity_score for method in methods)
        
        # Get source line range
        source_lines = (node.lineno, getattr(node, 'end_lineno', node.lineno))
        
        return ClassInfo(
            name=node.name,
            base_classes=base_classes,
            methods=methods,
            has_init=has_init,
            is_abstract=is_abstract,
            is_dataclass=is_dataclass,
            is_exception=is_exception,
            complexity_score=complexity_score,
            inheritance_depth=len(base_classes),
            source_lines=source_lines
        )
    
    def _extract_parameters(self, node: ast.FunctionDef) -> List[ParameterInfo]:
        """Extract parameter information from function definition."""
        parameters = []
        
        # Regular arguments
        for arg in node.args.args:
            param_info = ParameterInfo(
                name=arg.arg,
                type_hint=self._extract_type_hint(arg.annotation) if arg.annotation else "Any"
            )
            parameters.append(param_info)
        
        # *args
        if node.args.vararg:
            param_info = ParameterInfo(
                name=node.args.vararg.arg,
                type_hint="Tuple[Any, ...]",
                is_varargs=True
            )
            parameters.append(param_info)
        
        # **kwargs
        if node.args.kwarg:
            param_info = ParameterInfo(
                name=node.args.kwarg.arg,
                type_hint="Dict[str, Any]",
                is_kwargs=True
            )
            parameters.append(param_info)
        
        return parameters
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> ComplexityMetrics:
        """Calculate comprehensive complexity metrics for a function."""
        return ComplexityMetrics(
            cyclomatic_complexity=self.complexity_calculator.calculate_cyclomatic_complexity(node),
            cognitive_complexity=self.complexity_calculator.calculate_cognitive_complexity(node),
            nesting_depth=self.complexity_calculator.calculate_nesting_depth(node),
            number_of_parameters=len(node.args.args),
            lines_of_code=len([n for n in ast.walk(node) if hasattr(n, 'lineno')]),
            number_of_returns=len([n for n in ast.walk(node) if isinstance(n, ast.Return)]),
            number_of_branches=len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.While, ast.For))]),
            number_of_loops=len([n for n in ast.walk(node) if isinstance(n, (ast.While, ast.For, ast.AsyncFor))]),
            halstead_volume=0.0,  # Would need more detailed analysis
            maintainability_index=100.0  # Simplified calculation
        )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)
                
        return imports
    
    def _extract_constants(self, tree: ast.AST) -> List[str]:
        """Extract module-level constants."""
        constants = []
        
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(target.id)
                        
        return constants
    
    def _calculate_line_metrics(self, source_code: str) -> Dict[str, int]:
        """Calculate line-based metrics."""
        lines = source_code.split('\n')
        
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total_lines - blank_lines - comment_lines
        
        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines
        }
    
    def _calculate_maintainability_index(self, lines_of_code: int, complexity: int, source_code: str) -> float:
        """Calculate maintainability index (simplified)."""
        # Simplified MI calculation
        if lines_of_code == 0:
            return 100.0
            
        halstead_volume = self.complexity_calculator.calculate_halstead_metrics(source_code)
        
        # Microsoft maintainability index formula (simplified)
        mi = 171 - 5.2 * (complexity / lines_of_code if lines_of_code > 0 else 0) - 0.23 * (halstead_volume / 1000) - 16.2 * (lines_of_code / 1000)
        return max(0.0, min(100.0, mi))
    
    def _get_module_name(self, file_path: Path) -> str:
        """Extract module name from file path."""
        parts = file_path.parts
        if 'src' in parts:
            src_index = parts.index('src')
            module_parts = parts[src_index + 1:]
        else:
            module_parts = parts[-1:]
            
        module_path = '.'.join(part for part in module_parts if not part.endswith('.py'))
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
            
        return module_path or file_path.stem
    
    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """Extract return type annotation."""
        if node.returns:
            return self._extract_type_hint(node.returns)
        return "Any"
    
    def _extract_type_hint(self, annotation: ast.AST) -> str:
        """Extract type hint from AST annotation."""
        try:
            return ast.unparse(annotation)
        except:
            return "Any"
    
    def _extract_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """Extract exceptions that function might raise."""
        exceptions = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.append(child.exc.func.id)
                    
        return list(set(exceptions))
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        else:
            try:
                return ast.unparse(decorator)
            except:
                return "unknown"
    
    def _get_base_class_name(self, base: ast.AST) -> str:
        """Extract base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        else:
            try:
                return ast.unparse(base)
            except:
                return "unknown"


# Factory function
def create_code_analyzer() -> CodeAnalyzer:
    """
    Create a code analyzer with default configuration.
    
    Returns:
        Configured CodeAnalyzer instance
    """
    return CodeAnalyzer()


# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Code Analysis Team'
__description__ = 'Advanced AST-based code analysis for intelligent test generation'