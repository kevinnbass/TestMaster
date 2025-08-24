"""
AST-Based Code Understanding Module
Extracted from enhanced_intelligence_linkage.py for Agent X's Epsilon base integration
< 200 lines per STEELCLAD protocol

Provides deep code analysis using Abstract Syntax Tree parsing:
- Structural code analysis
- Conceptual element extraction
- Class and function mapping
- Architectural pattern detection
"""

import ast
import re
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ClassInfo:
    """Class information extracted from AST"""
    name: str
    methods: List[str] = field(default_factory=list)
    inheritance: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    line_number: int = 0

@dataclass  
class FunctionInfo:
    """Function information extracted from AST"""
    name: str
    args_count: int = 0
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    line_number: int = 0
    is_method: bool = False

@dataclass
class CodeStructure:
    """Complete code structure analysis"""
    classes: List[ClassInfo]
    functions: List[FunctionInfo] 
    imports: List[str]
    complexity_indicators: List[str]
    architectural_patterns: List[str]
    total_lines: int
    analysis_timestamp: str

class ASTCodeUnderstanding:
    """AST-based code analysis and understanding module"""
    
    def __init__(self):
        self.architectural_patterns = {
            "mvc": ["model", "view", "controller"],
            "singleton": ["singleton", "__new__", "_instance"],
            "factory": ["factory", "create", "builder"],
            "observer": ["observer", "notify", "subscribe"],
            "decorator": ["decorator", "wrapper", "@"],
            "strategy": ["strategy", "algorithm", "execute"],
            "repository": ["repository", "save", "find", "query"],
            "service": ["service", "business", "logic"]
        }
        
        self.complexity_indicators = [
            "nested_loops", "deep_inheritance", "long_methods", 
            "high_coupling", "low_cohesion", "complex_conditions"
        ]
        
    def analyze_code_structure(self, content: str, filename: str = None) -> CodeStructure:
        """Analyze code structure using AST parsing"""
        classes = []
        functions = []
        imports = []
        complexity_indicators = []
        architectural_patterns = []
        
        try:
            # Parse content with AST
            tree = ast.parse(content)
            
            # Extract structural elements
            classes = self._extract_classes(tree)
            functions = self._extract_functions(tree)
            imports = self._extract_imports(tree)
            
            # Analyze complexity
            complexity_indicators = self._analyze_complexity(tree, content)
            
            # Detect architectural patterns
            architectural_patterns = self._detect_patterns(content, classes, functions)
            
        except SyntaxError as e:
            logger.warning(f"AST parsing failed for {filename}, falling back to regex: {e}")
            # Fallback to regex-based analysis
            classes, functions, imports = self._fallback_analysis(content)
        
        return CodeStructure(
            classes=classes,
            functions=functions,
            imports=imports,
            complexity_indicators=complexity_indicators,
            architectural_patterns=architectural_patterns,
            total_lines=len(content.splitlines()),
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _extract_classes(self, tree: ast.AST) -> List[ClassInfo]:
        """Extract class information from AST"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract method names
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                # Extract inheritance
                inheritance = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        inheritance.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        inheritance.append(f"{base.value.id}.{base.attr}")
                
                # Extract decorators
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                
                classes.append(ClassInfo(
                    name=node.name,
                    methods=methods,
                    inheritance=inheritance,
                    decorators=decorators,
                    line_number=getattr(node, 'lineno', 0)
                ))
        
        return classes
    
    def _extract_functions(self, tree: ast.AST) -> List[FunctionInfo]:
        """Extract function information from AST"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract decorators
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                
                functions.append(FunctionInfo(
                    name=node.name,
                    args_count=len(node.args.args),
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    decorators=decorators,
                    line_number=getattr(node, 'lineno', 0),
                    is_method=self._is_method(node)
                ))
        
        return functions
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    # Also add specific imports
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports
    
    def _is_method(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is a method (inside a class)"""
        # Simple heuristic: if first argument is 'self' or 'cls'
        if func_node.args.args:
            first_arg = func_node.args.args[0].arg
            return first_arg in ['self', 'cls']
        return False
    
    def _analyze_complexity(self, tree: ast.AST, content: str) -> List[str]:
        """Analyze code complexity indicators"""
        indicators = []
        
        # Count nested loops
        nested_loop_count = self._count_nested_structures(tree, (ast.For, ast.While))
        if nested_loop_count > 2:
            indicators.append(f"nested_loops_{nested_loop_count}")
        
        # Check for long methods (> 50 lines)
        long_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    length = node.end_lineno - node.lineno
                    if length > 50:
                        long_methods.append(node.name)
        
        if long_methods:
            indicators.append(f"long_methods_{len(long_methods)}")
        
        # Check for complex conditions
        complex_conditions = self._count_complex_conditions(tree)
        if complex_conditions > 5:
            indicators.append(f"complex_conditions_{complex_conditions}")
        
        # Check for high cyclomatic complexity
        complexity_score = self._calculate_cyclomatic_complexity(content)
        if complexity_score > 15:
            indicators.append(f"high_complexity_{complexity_score}")
        
        return indicators
    
    def _count_nested_structures(self, tree: ast.AST, node_types: tuple) -> int:
        """Count nested structures of given types"""
        max_depth = 0
        
        def count_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, node_types):
                    count_depth(child, current_depth + 1)
                else:
                    count_depth(child, current_depth)
        
        count_depth(tree)
        return max_depth
    
    def _count_complex_conditions(self, tree: ast.AST) -> int:
        """Count complex conditional statements"""
        complex_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for complex boolean expressions
                if self._is_complex_condition(node.test):
                    complex_count += 1
        
        return complex_count
    
    def _is_complex_condition(self, node: ast.expr) -> bool:
        """Check if condition is complex (multiple boolean operators)"""
        bool_op_count = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.And, ast.Or)):
                bool_op_count += 1
        
        return bool_op_count > 2
    
    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity using keyword counting"""
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'and', 'or', 'try', 'except']
        complexity = 1  # Base complexity
        
        content_lower = content.lower()
        for keyword in complexity_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', content_lower))
        
        return complexity
    
    def _detect_patterns(self, content: str, classes: List[ClassInfo], functions: List[FunctionInfo]) -> List[str]:
        """Detect architectural patterns in code"""
        detected_patterns = []
        content_lower = content.lower()
        
        for pattern_name, indicators in self.architectural_patterns.items():
            matches = sum(1 for indicator in indicators if indicator in content_lower)
            
            # Additional pattern-specific checks
            if pattern_name == "mvc" and len(classes) > 2:
                matches += 1
            elif pattern_name == "singleton" and any("__new__" in f.name for f in functions):
                matches += 2
            elif pattern_name == "decorator" and any(f.decorators for f in functions):
                matches += 2
            
            if matches >= 2:
                detected_patterns.append(f"{pattern_name}_pattern")
        
        return detected_patterns
    
    def _fallback_analysis(self, content: str) -> tuple:
        """Fallback regex-based analysis when AST parsing fails"""
        # Extract class names
        class_matches = re.findall(r'class\s+(\w+)', content)
        classes = [ClassInfo(name=name) for name in class_matches]
        
        # Extract function names
        func_matches = re.findall(r'def\s+(\w+)', content)
        functions = [FunctionInfo(name=name) for name in func_matches]
        
        # Extract imports
        import_matches = re.findall(r'(?:from\s+(\S+)\s+)?import\s+([^\n]+)', content)
        imports = []
        for match in import_matches:
            if match[0]:  # from X import Y
                imports.append(match[0])
            imports.extend([imp.strip() for imp in match[1].split(',')])
        
        return classes, functions, imports
    
    def get_analysis_summary(self, structure: CodeStructure) -> Dict[str, Any]:
        """Generate analysis summary"""
        return {
            "file_stats": {
                "total_lines": structure.total_lines,
                "classes": len(structure.classes),
                "functions": len(structure.functions),
                "imports": len(structure.imports)
            },
            "complexity_analysis": {
                "indicators": structure.complexity_indicators,
                "complexity_score": len(structure.complexity_indicators)
            },
            "architectural_patterns": structure.architectural_patterns,
            "analysis_quality": "high" if structure.classes or structure.functions else "basic",
            "timestamp": structure.analysis_timestamp
        }

# Plugin interface for Agent X integration
def create_ast_analyzer_plugin(config: Dict[str, Any] = None):
    """Factory function to create AST code understanding plugin"""
    return ASTCodeUnderstanding()