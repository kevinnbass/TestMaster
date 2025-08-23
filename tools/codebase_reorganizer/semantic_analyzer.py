#!/usr/bin/env python3
"""
Semantic Code Analyzer
======================

Analyzes Python code to understand its semantic meaning, purpose, and relationships.
Uses AST analysis, keyword extraction, and pattern recognition to categorize code
and determine its role in the system.

This intelligence module provides deep understanding of code semantics for
the intelligent reorganization system.
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import keyword


@dataclass
class SemanticAnalysis:
    """Semantic analysis results for a code file"""
    file_path: Path
    primary_purpose: str
    secondary_purposes: List[str]
    functionality_score: float
    complexity_score: float
    domain_keywords: List[str]
    technical_keywords: List[str]
    relationships: Dict[str, List[str]]
    imports_analysis: Dict[str, Any]
    class_hierarchy: List[Dict[str, Any]]
    function_purposes: List[Dict[str, Any]]
    code_patterns: List[str]
    semantic_confidence: float
    categorization_hints: List[str]


class SemanticAnalyzer:
    """
    Advanced semantic analyzer that understands code meaning and relationships.
    Used by the intelligence-enhanced reorganizer to make informed decisions.
    """

    def __init__(self) -> None:
        """Initialize the semantic analyzer"""
        self._load_semantic_knowledge()

    def _load_semantic_knowledge(self) -> None:
        """Load semantic knowledge base for code categorization"""
        self.domain_patterns = {
            'data_processing': {
                'keywords': ['process', 'transform', 'clean', 'validate', 'parse', 'extract', 'load', 'save', 'csv', 'json', 'xml', 'database', 'sql'],
                'imports': ['pandas', 'numpy', 'csv', 'json', 'sqlite3', 'sqlalchemy', 'requests', 'beautifulsoup']
            },
            'machine_learning': {
                'keywords': ['train', 'predict', 'model', 'algorithm', 'neural', 'deep', 'learning', 'classification', 'regression', 'cluster'],
                'imports': ['sklearn', 'tensorflow', 'torch', 'keras', 'xgboost', 'lightgbm', 'transformers']
            },
            'web_development': {
                'keywords': ['route', 'endpoint', 'api', 'request', 'response', 'server', 'client', 'http', 'flask', 'django', 'fastapi'],
                'imports': ['flask', 'django', 'fastapi', 'requests', 'httpx', 'aiohttp', 'uvicorn']
            },
            'testing': {
                'keywords': ['test', 'assert', 'mock', 'fixture', 'pytest', 'unittest', 'coverage', 'spec'],
                'imports': ['pytest', 'unittest', 'mock', 'coverage', 'hypothesis']
            },
            'utilities': {
                'keywords': ['helper', 'util', 'tool', 'common', 'shared', 'config', 'settings', 'logging', 'cache'],
                'imports': ['logging', 'configparser', 'json', 'os', 'sys', 'pathlib']
            },
            'security': {
                'keywords': ['auth', 'encrypt', 'decrypt', 'hash', 'token', 'password', 'security', 'auth', 'permission'],
                'imports': ['cryptography', 'hashlib', 'secrets', 'jwt', 'bcrypt', 'passlib']
            },
            'data_analysis': {
                'keywords': ['analyze', 'statistics', 'plot', 'chart', 'graph', 'visualization', 'report', 'dashboard'],
                'imports': ['matplotlib', 'seaborn', 'plotly', 'pandas', 'numpy', 'scipy', 'statsmodels']
            },
            'automation': {
                'keywords': ['automation', 'script', 'batch', 'job', 'task', 'schedule', 'cron', 'workflow'],
                'imports': ['schedule', 'apscheduler', 'celery', 'airflow', 'subprocess', 'os', 'shutil']
            }
        }

        self.technical_patterns = {
            'asynchronous': ['async', 'await', 'asyncio', 'coroutine', 'concurrent', 'threading'],
            'object_oriented': ['class', 'method', 'property', 'inheritance', 'polymorphism'],
            'functional': ['lambda', 'map', 'filter', 'reduce', 'decorator', 'generator'],
            'data_structures': ['list', 'dict', 'set', 'tuple', 'queue', 'stack', 'tree', 'graph'],
            'error_handling': ['try', 'except', 'raise', 'logging', 'debug', 'traceback'],
            'performance': ['cache', 'memoization', 'optimization', 'profiling', 'benchmark'],
            'io_operations': ['file', 'read', 'write', 'open', 'close', 'stream', 'socket'],
            'networking': ['http', 'api', 'client', 'server', 'request', 'response', 'websocket']
        }

    def analyze_semantics(self, content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis of code content.

        Args:
            content: The code content to analyze
            file_path: Optional path to the file for context

        Returns:
            Dictionary containing semantic analysis results
        """
        try:
            # Parse the AST
            tree = ast.parse(content)

            # Extract semantic information
            imports_analysis = self._analyze_imports(tree)
            class_analysis = self._analyze_classes(tree)
            function_analysis = self._analyze_functions(tree, content)
            keyword_analysis = self._extract_semantic_keywords(content)

            # Determine primary purpose
            primary_purpose, secondary_purposes = self._determine_purpose(
                imports_analysis, class_analysis, function_analysis, keyword_analysis
            )

            # Calculate scores
            functionality_score = self._calculate_functionality_score(
                class_analysis, function_analysis, keyword_analysis
            )
            complexity_score = self._calculate_complexity_score(tree, content)

            # Extract relationships
            relationships = self._extract_relationships(tree, imports_analysis)

            # Generate categorization hints
            categorization_hints = self._generate_categorization_hints(
                primary_purpose, secondary_purposes, imports_analysis, keyword_analysis
            )

            # Calculate semantic confidence
            semantic_confidence = self._calculate_semantic_confidence(
                functionality_score, complexity_score, len(keyword_analysis['domain_keywords'])
            )

            # Create comprehensive result
            result = {
                'primary_purpose': primary_purpose,
                'secondary_purposes': secondary_purposes,
                'functionality_score': functionality_score,
                'complexity_score': complexity_score,
                'domain_keywords': keyword_analysis['domain_keywords'],
                'technical_keywords': keyword_analysis['technical_keywords'],
                'relationships': relationships,
                'imports_analysis': imports_analysis,
                'class_hierarchy': class_analysis,
                'function_purposes': function_analysis,
                'code_patterns': self._identify_code_patterns(tree, content),
                'semantic_confidence': semantic_confidence,
                'categorization_hints': categorization_hints,
                'file_path': str(file_path) if file_path else 'unknown'
            }

            return result

        except SyntaxError as e:
            return self._fallback_semantic_analysis(content, file_path, e)
        except Exception as e:
            return {
                'error': f'Semantic analysis failed: {e}',
                'primary_purpose': 'unknown',
                'secondary_purposes': [],
                'functionality_score': 0.0,
                'complexity_score': 0.0,
                'domain_keywords': [],
                'technical_keywords': [],
                'relationships': {},
                'imports_analysis': {},
                'class_hierarchy': [],
                'function_purposes': [],
                'code_patterns': [],
                'semantic_confidence': 0.0,
                'categorization_hints': [],
                'file_path': str(file_path) if file_path else 'unknown'
            }

    def _analyze_imports(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze import statements for semantic meaning"""
        standard_library_imports = set()
        third_party_imports = set()
        local_imports = set()
        import_patterns: Dict[str, int] = defaultdict(int)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name.split('.')[0]
                    if self._is_standard_library(import_name):
                        standard_library_imports.add(import_name)
                    else:
                        third_party_imports.add(import_name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_name = node.module.split('.')[0]
                    if self._is_standard_library(import_name):
                        standard_library_imports.add(import_name)
                    else:
                        third_party_imports.add(import_name)
                else:
                    # Relative import
                    local_imports.add('relative_import')

        return {
            'standard_library': list(standard_library_imports),
            'third_party': list(third_party_imports),
            'local': list(local_imports),
            'total_imports': len(standard_library_imports) + len(third_party_imports) + len(local_imports),
            'import_diversity': len(set(standard_library_imports) | set(third_party_imports))
        }

    def _is_standard_library(self, module_name: str) -> bool:
        """Check if a module is part of Python's standard library"""
        standard_lib_modules = {
            'os', 'sys', 'json', 're', 'ast', 'pathlib', 'collections', 'itertools',
            'functools', 'typing', 'dataclasses', 'enum', 'datetime', 'time',
            'random', 'hashlib', 'uuid', 'logging', 'configparser', 'argparse',
            'subprocess', 'threading', 'multiprocessing', 'queue', 'contextlib',
            'tempfile', 'shutil', 'glob', 'fnmatch', 'linecache', 'pickle',
            'copyreg', 'copy', 'pprint', 'reprlib', 'enum', 'numbers', 'math',
            'cmath', 'decimal', 'fractions', 'random', 'statistics', 'datetime',
            'calendar', 'time', 'zoneinfo', 'locale', 'gettext', 'argparse',
            'getopt', 'getpass', 'curses', 'platform', 'errno', 'ctypes',
            'msvcrt', 'winreg', 'winsound', 'posix', 'pwd', 'spwd', 'grp',
            'crypt', 'termios', 'tty', 'pty', 'fcntl', 'pipes', 'signal',
            'socket', 'ssl', 'select', 'selectors', 'asyncio', 'queue',
            'sched', '_thread', 'dummy_thread', 'contextvars', 'concurrent'
        }
        return module_name in standard_lib_modules

    def _analyze_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze class definitions for semantic meaning"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'methods': [],
                    'properties': [],
                    # Convert base classes (replacing complex comprehension with explicit loop)
                    base_classes = []
                    for base in node.bases:
                        if hasattr(base, 'id'):
                            base_classes.append(base.id)
                        else:
                            base_classes.append(str(base))
                    'base_classes': base_classes,
                    'line_number': node.lineno,
                    'method_count': 0,
                    'property_count': 0
                }

                # Analyze methods and properties
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if not item.name.startswith('_'):
                            class_info['methods'].append(item.name)
                        class_info['method_count'] += 1
                    elif isinstance(item, ast.Assign) and len(item.targets) == 1:
                        target = item.targets[0]
                        if isinstance(target, ast.Name) and not target.id.startswith('_'):
                            class_info['properties'].append(target.id)
                            class_info['property_count'] += 1

                classes.append(class_info)

        return classes

    def _analyze_functions(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Analyze function definitions for semantic meaning"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method_in_class(tree, node):
                # Get function content
                lines = content.split('\n')
                start_line = node.lineno - 1
                end_line = self._find_function_end(lines, start_line)

                function_content = '\n'.join(lines[start_line:end_line + 1])

                # Analyze function purpose
                purpose = self._analyze_function_purpose(node.name, function_content, node)

                functions.append({
                    'name': node.name,
                    'purpose': purpose,
                    'line_number': node.lineno,
                    'parameter_count': len(node.args.args),
                    'has_return': node.returns is not None,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'complexity': self._estimate_function_complexity(node)
                })

        return functions

    def _is_method_in_class(self, tree: ast.AST, func_node: ast.FunctionDef) -> bool:
        """Check if a function is actually a method inside a class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in ast.walk(node):
                    if item == func_node:
                        return True
        return False

    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a function"""
        indent_level = None
        end_line = start_line

        for i in range(start_line, len(lines)):
            line = lines[i].rstrip()
            if not line:
                continue

            current_indent = len(line) - len(line.lstrip())

            if indent_level is None:
                indent_level = current_indent
            elif current_indent <= indent_level and line.lstrip():
                break

            end_line = i

        return end_line

    def _analyze_function_purpose(self, name: str, content: str, node: ast.FunctionDef) -> str:
        """Analyze the purpose of a function based on its name and content"""
        name_lower = name.lower()

        # Check function name patterns
        if 'get' in name_lower or 'fetch' in name_lower or 'retrieve' in name_lower:
            return 'data_retrieval'
        elif 'set' in name_lower or 'update' in name_lower or 'modify' in name_lower:
            return 'data_modification'
        elif 'process' in name_lower or 'handle' in name_lower or 'manage' in name_lower:
            return 'data_processing'
        elif 'validate' in name_lower or 'check' in name_lower or 'verify' in name_lower:
            return 'validation'
        elif 'parse' in name_lower or 'extract' in name_lower or 'analyze' in name_lower:
            return 'data_analysis'
        elif 'train' in name_lower or 'predict' in name_lower or 'classify' in name_lower:
            return 'machine_learning'
        elif 'test' in name_lower or 'assert' in name_lower:
            return 'testing'
        elif 'log' in name_lower or 'print' in name_lower or 'display' in name_lower:
            return 'logging_output'
        elif 'init' in name_lower or 'setup' in name_lower or 'initialize' in name_lower:
            return 'initialization'
        elif 'clean' in name_lower or 'sanitize' in name_lower or 'normalize' in name_lower:
            return 'data_cleaning'
        elif 'save' in name_lower or 'write' in name_lower or 'store' in name_lower:
            return 'data_persistence'
        elif 'load' in name_lower or 'read' in name_lower or 'import' in name_lower:
            return 'data_loading'
        else:
            return 'utility'

    def _extract_semantic_keywords(self, content: str) -> Dict[str, List[str]]:
        """Extract semantic keywords from content"""
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content.lower())

        # Remove Python keywords (replacing complex comprehension with explicit loop)
        python_keywords = set(keyword.kwlist)
        filtered_words = []
        for word in words:
            if word not in python_keywords and len(word) > 2:
                filtered_words.append(word)

        # Categorize keywords
        domain_keywords = []
        technical_keywords = []

        for word in filtered_words:
            # Check if it's a domain-specific keyword
            for domain, patterns in self.domain_patterns.items():
                if word in patterns['keywords']:
                    domain_keywords.append(word)
                    break

            # Check if it's a technical keyword
            for tech_area, patterns in self.technical_patterns.items():
                if word in patterns:
                    technical_keywords.append(word)
                    break

        # Remove duplicates and sort by frequency (replacing complex comprehension with explicit loop)
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

        # Sort unique domain keywords by frequency
        unique_domain_keywords = list(set(domain_keywords))
        unique_domain_keywords.sort(key=lambda x: word_counts.get(x, 0), reverse=True)
        domain_keywords = unique_domain_keywords
        # Sort unique technical keywords by frequency
        unique_technical_keywords = list(set(technical_keywords))
        unique_technical_keywords.sort(key=lambda x: word_counts.get(x, 0), reverse=True)
        technical_keywords = unique_technical_keywords

        return {
            'domain_keywords': domain_keywords[:10],  # Top 10
            'technical_keywords': technical_keywords[:10]  # Top 10
        }

    def _determine_purpose(self, imports_analysis: Dict[str, Any],
                          class_analysis: List[Dict[str, Any]],
                          function_analysis: List[Dict[str, Any]],
                          keyword_analysis: Dict[str, List[str]]) -> Tuple[str, List[str]]:
        """Determine the primary and secondary purposes of the code"""

        # Score each domain based on imports, keywords, and functions
        domain_scores: Dict[str, float] = defaultdict(float)

        # Import-based scoring
        for import_name in imports_analysis['third_party']:
            for domain, patterns in self.domain_patterns.items():
                if import_name in patterns['imports']:
                    domain_scores[domain] += 2.0

        # Keyword-based scoring
        for keyword in keyword_analysis['domain_keywords']:
            for domain, patterns in self.domain_patterns.items():
                if keyword in patterns['keywords']:
                    domain_scores[domain] += 1.0

        # Function-based scoring
        for func in function_analysis:
            purpose = func['purpose']
            # Map function purposes to domains
            purpose_to_domain = {
                'data_processing': 'data_processing',
                'machine_learning': 'machine_learning',
                'data_analysis': 'data_analysis',
                'validation': 'data_processing',
                'data_cleaning': 'data_processing',
                'data_persistence': 'data_processing',
                'data_loading': 'data_processing',
                'data_retrieval': 'data_processing',
                'data_modification': 'data_processing',
                'testing': 'testing',
                'utility': 'utilities'
            }

            domain = purpose_to_domain.get(purpose, 'utilities')
            domain_scores[domain] += 0.5

        # Determine primary and secondary purposes
        if not domain_scores:
            return 'utilities', []

        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_purpose = sorted_domains[0][0]
        # Find secondary purposes (replacing complex comprehension with explicit loop)
        secondary_purposes = []
        if len(sorted_domains) > 1:
            primary_score = sorted_domains[0][1]
            threshold = primary_score * 0.3
            for domain, score in sorted_domains[1:4]:
                if score > threshold:
                    secondary_purposes.append(domain)

        return primary_purpose, secondary_purposes

    def _calculate_functionality_score(self, class_analysis: List[Dict[str, Any]],
                                     function_analysis: List[Dict[str, Any]],
                                     keyword_analysis: Dict[str, List[str]]) -> float:
        """Calculate functionality score based on classes, functions, and keywords"""

        # Base score from function count
        function_score = min(len(function_analysis) / 10.0, 1.0)

        # Score from class count and complexity
        class_score = 0.0
        for cls in class_analysis:
            method_score = min(cls['method_count'] / 5.0, 1.0)
            property_score = min(cls['property_count'] / 3.0, 0.5)
            class_score += (method_score + property_score) / 2
        class_score = min(class_score, 1.0)

        # Score from keyword diversity
        keyword_score = min((len(keyword_analysis['domain_keywords']) +
                           len(keyword_analysis['technical_keywords'])) / 10.0, 1.0)

        # Combine scores
        return (function_score * 0.4 + class_score * 0.4 + keyword_score * 0.2)

    def _calculate_complexity_score(self, tree: ast.AST, content: str) -> float:
        """Calculate complexity score based on AST structure and content"""

        complexity_indicators = 0
        total_lines = len(content.split('\n'))

        # Count nested structures
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity_indicators += 1
            elif isinstance(node, ast.FunctionDef):
                # Complex functions (many parameters, long functions)
                if len(node.args.args) > 5:
                    complexity_indicators += 1
                if hasattr(node, 'body') and len(node.body) > 20:
                    complexity_indicators += 1
            elif isinstance(node, ast.ClassDef):
                # Complex classes (many methods)
                method_count = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
                if method_count > 10:
                    complexity_indicators += 1

        # Normalize complexity score
        if total_lines == 0:
            return 0.0

        complexity_ratio = complexity_indicators / max(total_lines / 10, 1)
        return min(complexity_ratio, 1.0)

    def _extract_relationships(self, tree: ast.AST, imports_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract relationships between modules and components"""
        relationships = defaultdict(list)

        # Extract import relationships
        for import_name in imports_analysis['third_party']:
            relationships['external_dependencies'].append(import_name)

        for import_name in imports_analysis['local']:
            relationships['internal_dependencies'].append(import_name)

        # Extract class inheritance relationships
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.bases:
                for base in node.bases:
                    if hasattr(base, 'id'):
                        relationships['inheritance'].append(f"{node.name} -> {base.id}")

        return dict(relationships)

    def _identify_code_patterns(self, tree: ast.AST, content: str) -> List[str]:
        """Identify common code patterns and architectural patterns"""
        patterns = []

        # Check for common patterns
        if 'async def' in content:
            patterns.append('asynchronous_programming')

        if 'with ' in content and 'open(' in content:
            patterns.append('context_management')

        if 'try:' in content and 'except:' in content:
            patterns.append('error_handling')

        if 'class ' in content and '__init__' in content:
            patterns.append('object_oriented')

        if 'lambda' in content or 'map(' in content or 'filter(' in content:
            patterns.append('functional_programming')

        if 'if __name__ == "__main__":' in content:
            patterns.append('main_module')

        # Check for design patterns
        if self._has_singleton_pattern(tree):
            patterns.append('singleton_pattern')

        if self._has_factory_pattern(tree):
            patterns.append('factory_pattern')

        if self._has_observer_pattern(tree):
            patterns.append('observer_pattern')

        return patterns

    def _has_singleton_pattern(self, tree: ast.AST) -> bool:
        """Check if code uses singleton pattern"""
        has_class = False
        has_instance_variable = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_class = True
                # Check for class-level instance variable
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and 'instance' in target.id.lower():
                                has_instance_variable = True

        return has_class and has_instance_variable

    def _has_factory_pattern(self, tree: ast.AST) -> bool:
        """Check if code uses factory pattern"""
        factory_indicators = ['create', 'factory', 'build', 'make']

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name.lower()
                if any(indicator in func_name for indicator in factory_indicators):
                    return True

        return False

    def _has_observer_pattern(self, tree: ast.AST) -> bool:
        """Check if code uses observer pattern"""
        observer_indicators = ['subscribe', 'notify', 'observer', 'listener', 'callback']

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name.lower()
                if any(indicator in func_name for indicator in observer_indicators):
                    return True

        return False

    def _generate_categorization_hints(self, primary_purpose: str,
                                     secondary_purposes: List[str],
                                     imports_analysis: Dict[str, Any],
                                     keyword_analysis: Dict[str, List[str]]) -> List[str]:
        """Generate hints for code categorization"""
        hints = []

        # Primary purpose hint
        hints.append(f"Primary purpose: {primary_purpose}")

        # Secondary purposes
        if secondary_purposes:
            hints.append(f"Secondary purposes: {', '.join(secondary_purposes)}")

        # Import-based hints
        if imports_analysis['third_party']:
            hints.append(f"Uses external libraries: {', '.join(imports_analysis['third_party'][:3])}")

        # Keyword-based hints
        if keyword_analysis['domain_keywords']:
            hints.append(f"Domain keywords: {', '.join(keyword_analysis['domain_keywords'][:3])}")

        # Technical hints
        if keyword_analysis['technical_keywords']:
            hints.append(f"Technical approach: {', '.join(keyword_analysis['technical_keywords'][:3])}")

        return hints

    def _calculate_semantic_confidence(self, functionality_score: float,
                                     complexity_score: float,
                                     keyword_count: int) -> float:
        """Calculate confidence in semantic analysis"""
        # Base confidence from functionality score
        confidence = functionality_score * 0.5

        # Add complexity factor (complex code is easier to analyze)
        confidence += complexity_score * 0.2

        # Add keyword diversity factor
        keyword_factor = min(keyword_count / 10.0, 1.0)
        confidence += keyword_factor * 0.3

        return min(confidence, 1.0)

    def _estimate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate function complexity"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and len(child.values) > 2:
                complexity += len(child.values) - 2

        return complexity

    def _fallback_semantic_analysis(self, content: str, file_path: Optional[Path],
                                  error: SyntaxError) -> Dict[str, Any]:
        """Fallback semantic analysis for files with syntax errors"""
        # Extract what we can from the content despite syntax errors
        keywords = self._extract_semantic_keywords(content)

        return {
            'primary_purpose': 'unknown',
            'secondary_purposes': [],
            'functionality_score': 0.0,
            'complexity_score': 0.0,
            'domain_keywords': keywords['domain_keywords'],
            'technical_keywords': keywords['technical_keywords'],
            'relationships': {},
            'imports_analysis': {'total_imports': 0, 'import_diversity': 0},
            'class_hierarchy': [],
            'function_purposes': [],
            'code_patterns': ['syntax_error'],
            'semantic_confidence': 0.0,
            'categorization_hints': [f'Contains syntax error: {error}'],
            'file_path': str(file_path) if file_path else 'unknown'
        }


# Module-level functions for easy integration
def analyze_semantics(content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Module-level function for semantic analysis"""
    analyzer = SemanticAnalyzer()
    return analyzer.analyze_semantics(content, file_path)


def extract_semantic_keywords(content: str) -> Dict[str, List[str]]:
    """Extract semantic keywords from content"""
    analyzer = SemanticAnalyzer()
    return analyzer._extract_semantic_keywords(content)


def determine_code_purpose(content: str) -> Tuple[str, List[str]]:
    """Determine the primary purpose of code"""
    analyzer = SemanticAnalyzer()

    # Parse and analyze
    tree = ast.parse(content)
    imports_analysis = analyzer._analyze_imports(tree)
    class_analysis = analyzer._analyze_classes(tree)
    function_analysis = analyzer._analyze_functions(tree, content)
    keyword_analysis = analyzer._extract_semantic_keywords(content)

    return analyzer._determine_purpose(
        imports_analysis, class_analysis, function_analysis, keyword_analysis
    )


if __name__ == "__main__":
    # Example usage
    sample_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

def analyze_results(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {"accuracy": model.score(X_test, y_test)}
"""

    analyzer = SemanticAnalyzer()
    result = analyzer.analyze_semantics(sample_code, Path("sample.py"))

    print("Semantic Analysis Results:")
    print(f"Primary Purpose: {result['primary_purpose']}")
    print(f"Secondary Purposes: {result['secondary_purposes']}")
    print(f"Functionality Score: {result['functionality_score']:.2f}")
    print(f"Semantic Confidence: {result['semantic_confidence']:.2f}")
    print(f"Domain Keywords: {result['domain_keywords']}")
    print(f"Technical Keywords: {result['technical_keywords']}")
    print(f"Categorization Hints: {result['categorization_hints']}")
