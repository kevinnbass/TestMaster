"""
Universal AST Abstraction for TestMaster

Provides language-agnostic AST representation and analysis.
"""

from .universal_ast import (
    UniversalASTAbstractor, 
    UniversalAST,
    UniversalFunction,
    UniversalClass,
    UniversalModule,
    UniversalVariable,
    UniversalExpression,
    ASTNodeType,
    SemanticAnalyzer
)

from .language_parsers import (
    LanguageParserRegistry,
    BaseLanguageParser,
    PythonASTParser,
    JavaScriptASTParser,
    TypeScriptASTParser,
    JavaASTParser,
    CSharpASTParser,
    GoASTParser,
    RustASTParser
)

__all__ = [
    'UniversalASTAbstractor',
    'UniversalAST',
    'UniversalFunction',
    'UniversalClass', 
    'UniversalModule',
    'UniversalVariable',
    'UniversalExpression',
    'ASTNodeType',
    'SemanticAnalyzer',
    'LanguageParserRegistry',
    'BaseLanguageParser',
    'PythonASTParser',
    'JavaScriptASTParser', 
    'TypeScriptASTParser',
    'JavaASTParser',
    'CSharpASTParser',
    'GoASTParser',
    'RustASTParser'
]