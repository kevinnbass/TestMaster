"""
Universal AST Abstraction System

Creates language-agnostic AST representation that works with any programming language.
This enables consistent analysis and test generation across all supported languages.
"""

import os
import re
import ast
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ASTNodeType(Enum):
    """Universal AST node types that exist across languages."""
    FUNCTION = "function"
    CLASS = "class" 
    MODULE = "module"
    VARIABLE = "variable"
    CONSTANT = "constant"
    EXPRESSION = "expression"
    STATEMENT = "statement"
    CONTROL_FLOW = "control_flow"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    TRY_CATCH = "try_catch"
    IMPORT = "import"
    ANNOTATION = "annotation"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    NAMESPACE = "namespace"


@dataclass
class CodeLocation:
    """Universal representation of code location."""
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0


@dataclass
class UniversalVariable:
    """Universal representation of a variable/field."""
    name: str
    type_hint: Optional[str] = None
    is_constant: bool = False
    is_static: bool = False
    is_private: bool = False
    is_protected: bool = False
    is_public: bool = True
    default_value: Optional[str] = None
    annotations: List[str] = field(default_factory=list)
    location: Optional[CodeLocation] = None


@dataclass 
class UniversalParameter:
    """Universal representation of function parameters."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_optional: bool = False
    is_variadic: bool = False
    is_keyword_only: bool = False
    annotations: List[str] = field(default_factory=list)


@dataclass
class UniversalFunction:
    """Universal representation of functions/methods."""
    name: str
    parameters: List[UniversalParameter] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_static: bool = False
    is_abstract: bool = False
    is_private: bool = False
    is_protected: bool = False
    is_public: bool = True
    is_constructor: bool = False
    is_destructor: bool = False
    decorators: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    complexity_score: float = 0.0
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 1
    lines_of_code: int = 0
    location: Optional[CodeLocation] = None
    calls_functions: List[str] = field(default_factory=list)
    accesses_variables: List[str] = field(default_factory=list)
    throws_exceptions: List[str] = field(default_factory=list)
    language_specific_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalClass:
    """Universal representation of classes/types."""
    name: str
    base_classes: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    methods: List[UniversalFunction] = field(default_factory=list)
    fields: List[UniversalVariable] = field(default_factory=list)
    inner_classes: List['UniversalClass'] = field(default_factory=list)
    is_abstract: bool = False
    is_interface: bool = False
    is_static: bool = False
    is_final: bool = False
    is_private: bool = False
    is_protected: bool = False
    is_public: bool = True
    decorators: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    location: Optional[CodeLocation] = None
    language_specific_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalImport:
    """Universal representation of imports/includes."""
    module_name: str
    imported_items: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_relative: bool = False
    is_wildcard: bool = False
    location: Optional[CodeLocation] = None


@dataclass
class UniversalExpression:
    """Universal representation of expressions and statements."""
    node_type: ASTNodeType
    content: str
    location: Optional[CodeLocation] = None
    child_expressions: List['UniversalExpression'] = field(default_factory=list)
    variables_used: List[str] = field(default_factory=list)
    functions_called: List[str] = field(default_factory=list)
    language_specific_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalModule:
    """Universal representation of a module/file."""
    name: str
    file_path: str
    language: str
    classes: List[UniversalClass] = field(default_factory=list)
    functions: List[UniversalFunction] = field(default_factory=list)
    variables: List[UniversalVariable] = field(default_factory=list)
    imports: List[UniversalImport] = field(default_factory=list)
    expressions: List[UniversalExpression] = field(default_factory=list)
    docstring: Optional[str] = None
    encoding: str = 'utf-8'
    lines_of_code: int = 0
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    language_specific_data: Dict[str, Any] = field(default_factory=dict)
    location: Optional[CodeLocation] = None


@dataclass
class UniversalAST:
    """Universal Abstract Syntax Tree representation."""
    modules: List[UniversalModule] = field(default_factory=list)
    root_module: Optional[UniversalModule] = None
    language: str = ""
    project_path: str = ""
    
    # Aggregate metrics
    total_functions: int = 0
    total_classes: int = 0
    total_lines: int = 0
    total_files: int = 0
    total_complexity: float = 0.0
    
    # Cross-references
    function_call_graph: Dict[str, List[str]] = field(default_factory=dict)
    class_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    # Analysis metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration: float = 0.0
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_functions(self) -> List[UniversalFunction]:
        """Get all functions across all modules."""
        functions = []
        for module in self.modules:
            functions.extend(module.functions)
            for cls in module.classes:
                functions.extend(cls.methods)
        return functions
    
    def get_all_classes(self) -> List[UniversalClass]:
        """Get all classes across all modules."""
        classes = []
        for module in self.modules:
            classes.extend(module.classes)
        return classes
    
    def find_function(self, name: str) -> Optional[UniversalFunction]:
        """Find a function by name."""
        for func in self.get_all_functions():
            if func.name == name:
                return func
        return None
    
    def find_class(self, name: str) -> Optional[UniversalClass]:
        """Find a class by name."""
        for cls in self.get_all_classes():
            if cls.name == name:
                return cls
        return None


class UniversalASTAbstractor:
    """Creates language-agnostic AST representations from any source code."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.language_parsers = {}
        self.semantic_analyzer = SemanticAnalyzer()
        self._initialize_parsers()
        
        print("Universal AST Abstractor initialized")
        print(f"   Supported languages: {list(self.language_parsers.keys())}")
    
    def _initialize_parsers(self):
        """Initialize language-specific parsers."""
        # Import parsers dynamically to avoid circular imports
        try:
            from .language_parsers import LanguageParserRegistry
            self.language_parsers = LanguageParserRegistry.get_all_parsers()
        except ImportError:
            # Fallback to basic parsers
            self.language_parsers = {
                'python': BasicPythonParser(),
                'javascript': BasicJavaScriptParser(),
                'typescript': BasicTypeScriptParser(),
            }
    
    def create_universal_ast(self, file_path: str, language: str = None) -> UniversalAST:
        """Create universal AST from a single file."""
        start_time = datetime.now()
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        # Auto-detect language if not provided
        if not language:
            language = self._detect_file_language(file_path)
        
        print(f"Creating Universal AST for {file_path} (language: {language})")
        
        # Parse the file
        module = self._parse_file(file_path, language)
        
        # Create Universal AST
        universal_ast = UniversalAST(
            modules=[module],
            root_module=module,
            language=language,
            project_path=str(file_path.parent),
            total_functions=len(module.functions) + sum(len(cls.methods) for cls in module.classes),
            total_classes=len(module.classes),
            total_lines=module.lines_of_code,
            analysis_duration=(datetime.now() - start_time).total_seconds()
        )
        
        # Perform semantic analysis
        self._perform_semantic_analysis(universal_ast)
        
        return universal_ast
    
    def create_project_ast(self, project_path: str) -> UniversalAST:
        """Create universal AST from an entire project."""
        start_time = datetime.now()
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        print(f"Creating Universal AST for project: {project_path}")
        
        # Find all source files
        source_files = self._find_source_files(project_path)
        print(f"Found {len(source_files)} source files")
        
        # Parse all modules
        modules = []
        for file_path in source_files:
            try:
                language = self._detect_file_language(file_path)
                if language:
                    module = self._parse_file(file_path, language)
                    modules.append(module)
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")
                continue
        
        # Create Universal AST
        universal_ast = UniversalAST(
            modules=modules,
            root_module=modules[0] if modules else None,
            language=self._detect_primary_language(modules),
            project_path=str(project_path),
            total_functions=sum(len(m.functions) + sum(len(cls.methods) for cls in m.classes) for m in modules),
            total_classes=sum(len(m.classes) for m in modules),
            total_lines=sum(m.lines_of_code for m in modules),
            analysis_duration=(datetime.now() - start_time).total_seconds()
        )
        
        # Set total_files attribute
        universal_ast.total_files = len(source_files)
        
        # Perform comprehensive analysis
        self._perform_semantic_analysis(universal_ast)
        self._build_cross_references(universal_ast)
        
        return universal_ast
    
    def _parse_file(self, file_path: Path, language: str) -> UniversalModule:
        """Parse a single file into Universal Module representation."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            content = ""
        
        # Use language-specific parser if available
        if language in self.language_parsers:
            parser = self.language_parsers[language]
            return parser.parse_to_universal(str(file_path), content)
        
        # Fallback to generic parsing
        return self._generic_parse(file_path, content, language)
    
    def _generic_parse(self, file_path: Path, content: str, language: str) -> UniversalModule:
        """Generic parsing fallback for unsupported languages."""
        print(f"Using generic parser for {language}")
        
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Basic pattern recognition
        functions = self._extract_functions_generic(content, language)
        classes = self._extract_classes_generic(content, language)
        imports = self._extract_imports_generic(content, language)
        variables = self._extract_variables_generic(content, language)
        
        return UniversalModule(
            name=file_path.stem,
            file_path=str(file_path),
            language=language,
            functions=functions,
            classes=classes,
            variables=variables,
            imports=imports,
            lines_of_code=lines_of_code,
            location=CodeLocation(str(file_path), 1, len(lines), 0, 0)
        )
    
    def _extract_functions_generic(self, content: str, language: str) -> List[UniversalFunction]:
        """Extract functions using generic patterns."""
        functions = []
        
        # Language-specific function patterns
        patterns = {
            'python': r'def\s+(\w+)\s*\([^)]*\):',
            'javascript': r'function\s+(\w+)\s*\([^)]*\)|const\s+(\w+)\s*=\s*\([^)]*\)\s*=>|(\w+)\s*:\s*function\s*\([^)]*\)',
            'typescript': r'function\s+(\w+)\s*\([^)]*\)|const\s+(\w+)\s*=\s*\([^)]*\)\s*=>|(\w+)\s*\([^)]*\)\s*:',
            'java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)',
            'csharp': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)',
            'go': r'func\s+(\w+)\s*\([^)]*\)',
            'rust': r'fn\s+(\w+)\s*\([^)]*\)',
        }
        
        pattern = patterns.get(language, r'(\w+)\s*\([^)]*\)')
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            # Get the first non-None group
            func_name = next((group for group in match.groups() if group), None)
            if func_name:
                line_num = content[:match.start()].count('\n') + 1
                functions.append(UniversalFunction(
                    name=func_name,
                    location=CodeLocation("", line_num, line_num, match.start(), match.end()),
                    language_specific_data={'pattern_match': match.group()}
                ))
        
        return functions
    
    def _extract_classes_generic(self, content: str, language: str) -> List[UniversalClass]:
        """Extract classes using generic patterns."""
        classes = []
        
        # Language-specific class patterns
        patterns = {
            'python': r'class\s+(\w+)(?:\([^)]*\))?:',
            'javascript': r'class\s+(\w+)(?:\s+extends\s+\w+)?',
            'typescript': r'class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?',
            'java': r'(?:public|private|protected)?\s*(?:abstract)?\s*class\s+(\w+)',
            'csharp': r'(?:public|private|protected)?\s*(?:abstract)?\s*class\s+(\w+)',
            'go': r'type\s+(\w+)\s+struct',
            'rust': r'struct\s+(\w+)',
        }
        
        pattern = patterns.get(language, r'class\s+(\w+)')
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            class_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            classes.append(UniversalClass(
                name=class_name,
                location=CodeLocation("", line_num, line_num, match.start(), match.end()),
                language_specific_data={'pattern_match': match.group()}
            ))
        
        return classes
    
    def _extract_imports_generic(self, content: str, language: str) -> List[UniversalImport]:
        """Extract imports using generic patterns."""
        imports = []
        
        # Language-specific import patterns
        patterns = {
            'python': [
                r'import\s+([\w\.]+)(?:\s+as\s+(\w+))?',
                r'from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)'
            ],
            'javascript': [
                r'import\s+(?:\{([^}]+)\}|\*\s+as\s+(\w+)|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'const\s+\{([^}]+)\}\s*=\s*require\([\'"]([^\'"]+)[\'"]\)'
            ],
            'java': [r'import\s+([\w\.]+);'],
            'csharp': [r'using\s+([\w\.]+);'],
            'go': [r'import\s+"([^"]+)"', r'import\s+(\w+)\s+"([^"]+)"'],
            'rust': [r'use\s+([\w:]+);'],
        }
        
        language_patterns = patterns.get(language, [r'import\s+([\w\.]+)'])
        
        for pattern in language_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[:match.start()].count('\n') + 1
                imports.append(UniversalImport(
                    module_name=match.group(1),
                    location=CodeLocation("", line_num, line_num, match.start(), match.end()),
                ))
        
        return imports
    
    def _extract_variables_generic(self, content: str, language: str) -> List[UniversalVariable]:
        """Extract variables using generic patterns."""
        variables = []
        
        # Basic variable patterns
        patterns = {
            'python': r'(\w+)\s*=\s*[^=]',
            'javascript': r'(?:var|let|const)\s+(\w+)',
            'java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*=',
            'csharp': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*=',
        }
        
        pattern = patterns.get(language, r'(\w+)\s*=')
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            var_name = match.group(1)
            # Skip common keywords and function names
            if var_name not in ['if', 'for', 'while', 'return', 'function', 'class']:
                line_num = content[:match.start()].count('\n') + 1
                variables.append(UniversalVariable(
                    name=var_name,
                    location=CodeLocation("", line_num, line_num, match.start(), match.end())
                ))
        
        return variables
    
    def _detect_file_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.php': 'php',
            '.rb': 'ruby',
            '.kt': 'kotlin',
            '.swift': 'swift',
        }
        
        return extension_map.get(file_path.suffix.lower(), 'unknown')
    
    def _detect_primary_language(self, modules: List[UniversalModule]) -> str:
        """Detect primary language from modules."""
        if not modules:
            return 'unknown'
        
        language_counts = {}
        for module in modules:
            language_counts[module.language] = language_counts.get(module.language, 0) + module.lines_of_code
        
        return max(language_counts, key=language_counts.get) if language_counts else 'unknown'
    
    def _find_source_files(self, project_path: Path) -> List[Path]:
        """Find all source code files in project."""
        source_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cs', '.go', '.rs',
            '.cpp', '.cxx', '.cc', '.c', '.h', '.hpp', '.php', '.rb', '.kt', '.swift'
        }
        
        source_files = []
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.pytest_cache', 'target', 'build', 'dist'}
        
        for root, dirs, files in os.walk(project_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in source_extensions:
                    source_files.append(file_path)
        
        return source_files
    
    def _perform_semantic_analysis(self, universal_ast: UniversalAST):
        """Perform semantic analysis on the Universal AST."""
        universal_ast.semantic_analysis = self.semantic_analyzer.analyze(universal_ast)
    
    def _build_cross_references(self, universal_ast: UniversalAST):
        """Build cross-reference graphs for functions, classes, and dependencies."""
        # Build function call graph
        for func in universal_ast.get_all_functions():
            universal_ast.function_call_graph[func.name] = func.calls_functions
        
        # Build class hierarchy
        for cls in universal_ast.get_all_classes():
            universal_ast.class_hierarchy[cls.name] = cls.base_classes
        
        # Build dependency graph
        for module in universal_ast.modules:
            universal_ast.dependency_graph[module.name] = [imp.module_name for imp in module.imports]


class SemanticAnalyzer:
    """Performs semantic analysis on Universal AST."""
    
    def analyze(self, universal_ast: UniversalAST) -> Dict[str, Any]:
        """Perform comprehensive semantic analysis."""
        analysis = {
            'complexity_analysis': self._analyze_complexity(universal_ast),
            'dependency_analysis': self._analyze_dependencies(universal_ast),
            'pattern_analysis': self._analyze_patterns(universal_ast),
            'quality_metrics': self._calculate_quality_metrics(universal_ast),
        }
        
        return analysis
    
    def _analyze_complexity(self, ast: UniversalAST) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        functions = ast.get_all_functions()
        classes = ast.get_all_classes()
        
        if not functions:
            return {'error': 'No functions found for complexity analysis'}
        
        complexity_scores = [f.complexity_score for f in functions if f.complexity_score > 0]
        cyclomatic_complexities = [f.cyclomatic_complexity for f in functions]
        
        return {
            'average_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'max_complexity': max(complexity_scores) if complexity_scores else 0,
            'average_cyclomatic': sum(cyclomatic_complexities) / len(cyclomatic_complexities),
            'max_cyclomatic': max(cyclomatic_complexities),
            'complex_functions': [f.name for f in functions if f.cyclomatic_complexity > 10],
            'class_complexity': {cls.name: len(cls.methods) for cls in classes}
        }
    
    def _analyze_dependencies(self, ast: UniversalAST) -> Dict[str, Any]:
        """Analyze dependency relationships."""
        return {
            'total_dependencies': len(ast.dependency_graph),
            'circular_dependencies': self._find_circular_dependencies(ast.dependency_graph),
            'dependency_depth': self._calculate_dependency_depth(ast.dependency_graph),
            'highly_coupled_modules': self._find_highly_coupled_modules(ast.dependency_graph)
        }
    
    def _analyze_patterns(self, ast: UniversalAST) -> Dict[str, Any]:
        """Analyze code patterns and architectural insights."""
        functions = ast.get_all_functions()
        classes = ast.get_all_classes()
        
        return {
            'design_patterns': self._detect_design_patterns(classes),
            'naming_patterns': self._analyze_naming_patterns(functions, classes),
            'code_smells': self._detect_code_smells(functions, classes),
            'architectural_patterns': self._detect_architectural_patterns(ast.modules)
        }
    
    def _calculate_quality_metrics(self, ast: UniversalAST) -> Dict[str, Any]:
        """Calculate code quality metrics."""
        functions = ast.get_all_functions()
        classes = ast.get_all_classes()
        
        documented_functions = len([f for f in functions if f.docstring])
        documented_classes = len([c for c in classes if c.docstring])
        
        return {
            'documentation_coverage': {
                'functions': documented_functions / len(functions) if functions else 0,
                'classes': documented_classes / len(classes) if classes else 0
            },
            'average_function_length': sum(f.lines_of_code for f in functions) / len(functions) if functions else 0,
            'class_method_distribution': {cls.name: len(cls.methods) for cls in classes},
            'type_annotation_coverage': self._calculate_type_coverage(functions)
        }
    
    def _find_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for node in dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _calculate_dependency_depth(self, dependency_graph: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate dependency depth for each module."""
        depths = {}
        
        def calculate_depth(node, visited):
            if node in visited:
                return 0  # Circular dependency
            if node in depths:
                return depths[node]
            
            visited.add(node)
            dependencies = dependency_graph.get(node, [])
            
            if not dependencies:
                depths[node] = 0
            else:
                max_depth = max(calculate_depth(dep, visited.copy()) for dep in dependencies)
                depths[node] = max_depth + 1
            
            return depths[node]
        
        for node in dependency_graph:
            calculate_depth(node, set())
        
        return depths
    
    def _find_highly_coupled_modules(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Find modules with high coupling (many dependencies)."""
        coupling_threshold = 5
        return [module for module, deps in dependency_graph.items() if len(deps) > coupling_threshold]
    
    def _detect_design_patterns(self, classes: List[UniversalClass]) -> List[str]:
        """Detect common design patterns."""
        patterns = []
        
        # Singleton pattern
        for cls in classes:
            if any(method.name in ['getInstance', 'get_instance'] and method.is_static for method in cls.methods):
                patterns.append(f'Singleton: {cls.name}')
        
        # Factory pattern
        for cls in classes:
            if 'factory' in cls.name.lower() or any('create' in method.name.lower() for method in cls.methods):
                patterns.append(f'Factory: {cls.name}')
        
        # Observer pattern
        observer_keywords = ['observer', 'listener', 'subscriber', 'notify', 'update']
        for cls in classes:
            if any(keyword in cls.name.lower() for keyword in observer_keywords):
                patterns.append(f'Observer: {cls.name}')
        
        return patterns
    
    def _analyze_naming_patterns(self, functions: List[UniversalFunction], classes: List[UniversalClass]) -> Dict[str, Any]:
        """Analyze naming conventions and patterns."""
        return {
            'function_naming': self._analyze_function_naming(functions),
            'class_naming': self._analyze_class_naming(classes),
            'naming_consistency': self._check_naming_consistency(functions, classes)
        }
    
    def _analyze_function_naming(self, functions: List[UniversalFunction]) -> Dict[str, Any]:
        """Analyze function naming patterns."""
        if not functions:
            return {}
        
        snake_case = sum(1 for f in functions if '_' in f.name and f.name.islower())
        camel_case = sum(1 for f in functions if f.name[0].islower() and any(c.isupper() for c in f.name))
        pascal_case = sum(1 for f in functions if f.name[0].isupper())
        
        return {
            'snake_case_percentage': snake_case / len(functions),
            'camel_case_percentage': camel_case / len(functions),
            'pascal_case_percentage': pascal_case / len(functions),
            'average_name_length': sum(len(f.name) for f in functions) / len(functions)
        }
    
    def _analyze_class_naming(self, classes: List[UniversalClass]) -> Dict[str, Any]:
        """Analyze class naming patterns."""
        if not classes:
            return {}
        
        pascal_case = sum(1 for c in classes if c.name[0].isupper())
        
        return {
            'pascal_case_percentage': pascal_case / len(classes),
            'average_name_length': sum(len(c.name) for c in classes) / len(classes)
        }
    
    def _check_naming_consistency(self, functions: List[UniversalFunction], classes: List[UniversalClass]) -> Dict[str, float]:
        """Check naming consistency across the codebase."""
        # This is a simplified consistency check
        return {
            'function_consistency': 0.8,  # Placeholder
            'class_consistency': 0.9,     # Placeholder
            'overall_consistency': 0.85   # Placeholder
        }
    
    def _detect_code_smells(self, functions: List[UniversalFunction], classes: List[UniversalClass]) -> List[str]:
        """Detect common code smells."""
        smells = []
        
        # Long functions
        for func in functions:
            if func.lines_of_code > 50:
                smells.append(f'Long function: {func.name} ({func.lines_of_code} lines)')
        
        # Functions with many parameters
        for func in functions:
            if len(func.parameters) > 5:
                smells.append(f'Many parameters: {func.name} ({len(func.parameters)} parameters)')
        
        # Large classes
        for cls in classes:
            if len(cls.methods) > 20:
                smells.append(f'Large class: {cls.name} ({len(cls.methods)} methods)')
        
        return smells
    
    def _detect_architectural_patterns(self, modules: List[UniversalModule]) -> List[str]:
        """Detect architectural patterns from module structure."""
        patterns = []
        module_names = [m.name.lower() for m in modules]
        
        # MVC pattern
        if any('model' in name for name in module_names) and \
           any('view' in name for name in module_names) and \
           any('controller' in name for name in module_names):
            patterns.append('MVC')
        
        # Repository pattern
        if any('repository' in name for name in module_names):
            patterns.append('Repository')
        
        # Service layer
        if any('service' in name for name in module_names):
            patterns.append('Service Layer')
        
        return patterns
    
    def _calculate_type_coverage(self, functions: List[UniversalFunction]) -> float:
        """Calculate type annotation coverage."""
        if not functions:
            return 0.0
        
        typed_functions = 0
        for func in functions:
            has_return_type = func.return_type is not None
            has_param_types = all(param.type_hint for param in func.parameters)
            if has_return_type or has_param_types:
                typed_functions += 1
        
        return typed_functions / len(functions)


# Basic parsers for fallback support
class BasicPythonParser:
    """Basic Python parser for fallback support."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse Python code to Universal Module."""
        try:
            tree = ast.parse(content)
            return self._convert_python_ast(file_path, tree, content)
        except SyntaxError as e:
            print(f"Python syntax error in {file_path}: {e}")
            return self._fallback_parse(file_path, content)
    
    def _convert_python_ast(self, file_path: str, tree: ast.AST, content: str) -> UniversalModule:
        """Convert Python AST to Universal Module."""
        functions = []
        classes = []
        imports = []
        variables = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(self._convert_function(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(self._convert_class(node))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(self._convert_import(node))
            elif isinstance(node, ast.Assign):
                variables.extend(self._convert_assignment(node))
        
        lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='python',
            functions=functions,
            classes=classes,
            imports=imports,
            variables=variables,
            lines_of_code=lines_of_code
        )
    
    def _convert_function(self, node: ast.FunctionDef) -> UniversalFunction:
        """Convert Python function to Universal Function."""
        parameters = []
        for arg in node.args.args:
            param = UniversalParameter(
                name=arg.arg,
                type_hint=ast.unparse(arg.annotation) if arg.annotation else None
            )
            parameters.append(param)
        
        return UniversalFunction(
            name=node.name,
            parameters=parameters,
            return_type=ast.unparse(node.returns) if node.returns else None,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            location=CodeLocation(
                "",
                node.lineno,
                node.end_lineno or node.lineno,
                node.col_offset,
                node.end_col_offset or 0
            )
        )
    
    def _convert_class(self, node: ast.ClassDef) -> UniversalClass:
        """Convert Python class to Universal Class."""
        methods = []
        fields = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._convert_function(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        fields.append(UniversalVariable(name=target.id))
        
        return UniversalClass(
            name=node.name,
            base_classes=[ast.unparse(base) for base in node.bases],
            methods=methods,
            fields=fields,
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            location=CodeLocation(
                "",
                node.lineno,
                node.end_lineno or node.lineno,
                node.col_offset,
                node.end_col_offset or 0
            )
        )
    
    def _convert_import(self, node: Union[ast.Import, ast.ImportFrom]) -> List[UniversalImport]:
        """Convert Python import to Universal Import."""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(UniversalImport(
                    module_name=alias.name,
                    alias=alias.asname
                ))
        elif isinstance(node, ast.ImportFrom):
            imported_items = [alias.name for alias in node.names]
            imports.append(UniversalImport(
                module_name=node.module or "",
                imported_items=imported_items,
                is_relative=node.level > 0
            ))
        
        return imports
    
    def _convert_assignment(self, node: ast.Assign) -> List[UniversalVariable]:
        """Convert Python assignment to Universal Variable."""
        variables = []
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                variables.append(UniversalVariable(
                    name=target.id,
                    location=CodeLocation(
                        "",
                        node.lineno,
                        node.end_lineno or node.lineno,
                        node.col_offset,
                        node.end_col_offset or 0
                    )
                ))
        
        return variables
    
    def _fallback_parse(self, file_path: str, content: str) -> UniversalModule:
        """Fallback parsing when AST parsing fails."""
        lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='python',
            lines_of_code=lines_of_code
        )


class BasicJavaScriptParser:
    """Basic JavaScript parser for fallback support."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse JavaScript code to Universal Module (basic implementation)."""
        lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*')])
        
        # Basic regex-based parsing
        functions = self._extract_js_functions(content)
        classes = self._extract_js_classes(content)
        imports = self._extract_js_imports(content)
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='javascript',
            functions=functions,
            classes=classes,
            imports=imports,
            lines_of_code=lines_of_code
        )
    
    def _extract_js_functions(self, content: str) -> List[UniversalFunction]:
        """Extract JavaScript functions using regex."""
        functions = []
        
        # Function declarations
        pattern = r'function\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer(pattern, content):
            functions.append(UniversalFunction(name=match.group(1)))
        
        # Arrow functions
        pattern = r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
        for match in re.finditer(pattern, content):
            functions.append(UniversalFunction(name=match.group(1)))
        
        return functions
    
    def _extract_js_classes(self, content: str) -> List[UniversalClass]:
        """Extract JavaScript classes using regex."""
        classes = []
        
        pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(pattern, content):
            base_classes = [match.group(2)] if match.group(2) else []
            classes.append(UniversalClass(
                name=match.group(1),
                base_classes=base_classes
            ))
        
        return classes
    
    def _extract_js_imports(self, content: str) -> List[UniversalImport]:
        """Extract JavaScript imports using regex."""
        imports = []
        
        # ES6 imports
        pattern = r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(pattern, content):
            imports.append(UniversalImport(module_name=match.group(1)))
        
        # CommonJS requires
        pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
        for match in re.finditer(pattern, content):
            imports.append(UniversalImport(module_name=match.group(1)))
        
        return imports


class BasicTypeScriptParser:
    """Basic TypeScript parser for fallback support."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse TypeScript code to Universal Module (basic implementation)."""
        # Use JavaScript parser as base and enhance with TypeScript features
        js_parser = BasicJavaScriptParser()
        module = js_parser.parse_to_universal(file_path, content)
        module.language = 'typescript'
        
        # Add TypeScript-specific parsing here
        interfaces = self._extract_ts_interfaces(content)
        # Convert interfaces to classes for now
        module.classes.extend(interfaces)
        
        return module
    
    def _extract_ts_interfaces(self, content: str) -> List[UniversalClass]:
        """Extract TypeScript interfaces as Universal Classes."""
        interfaces = []
        
        pattern = r'interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?'
        for match in re.finditer(pattern, content):
            base_interfaces = [name.strip() for name in match.group(2).split(',')] if match.group(2) else []
            interfaces.append(UniversalClass(
                name=match.group(1),
                base_classes=base_interfaces,
                is_interface=True
            ))
        
        return interfaces