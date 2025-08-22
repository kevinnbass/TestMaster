"""
AST Abstraction Module
======================
Provides AST analysis and abstraction capabilities.
"""

import ast
from typing import Any, Dict, List

class UniversalAST:
    """Universal AST representation for cross-language analysis."""
    
    def __init__(self, source_code: str = "", language: str = "python"):
        self.source_code = source_code
        self.language = language
        self.tree = None
        if source_code:
            self.parse()
    
    def parse(self):
        """Parse the source code into AST."""
        if self.language == "python":
            self.tree = ast.parse(self.source_code)
        return self.tree
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze the AST."""
        if not self.tree:
            return {}
        return {
            'type': 'Module',
            'language': self.language,
            'nodes': len(list(ast.walk(self.tree)))
        }

class UniversalFunction:
    """Universal function representation for cross-language analysis."""
    
    def __init__(self, name: str = "", params: List[str] = None, body: str = ""):
        self.name = name
        self.params = params or []
        self.body = body
        self.language = "python"
    
    def to_ast(self):
        """Convert to AST representation."""
        func_str = f"def {self.name}({', '.join(self.params)}):\n    {self.body or 'pass'}"
        return ast.parse(func_str)
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze the function."""
        return {
            'name': self.name,
            'params': self.params,
            'param_count': len(self.params),
            'has_body': bool(self.body)
        }

class UniversalClass:
    """Universal class representation for cross-language analysis."""
    
    def __init__(self, name: str = "", methods: List[str] = None, attributes: List[str] = None):
        self.name = name
        self.methods = methods or []
        self.attributes = attributes or []
        self.language = "python"
    
    def to_ast(self):
        """Convert to AST representation."""
        class_str = f"class {self.name}:\n    pass"
        return ast.parse(class_str)
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze the class."""
        return {
            'name': self.name,
            'methods': self.methods,
            'method_count': len(self.methods),
            'attributes': self.attributes,
            'attribute_count': len(self.attributes)
        }

class ASTAbstraction:
    """AST abstraction for code analysis."""
    
    def __init__(self):
        self.trees = {}
    
    def parse(self, code: str) -> ast.AST:
        """Parse code into AST."""
        return ast.parse(code)
    
    def analyze(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze an AST tree."""
        return {
            'type': type(tree).__name__,
            'children': len(list(ast.iter_child_nodes(tree)))
        }

# Global instance
_ast_abstraction = ASTAbstraction()

def get_ast_abstraction():
    """Get the global AST abstraction instance."""
    return _ast_abstraction
