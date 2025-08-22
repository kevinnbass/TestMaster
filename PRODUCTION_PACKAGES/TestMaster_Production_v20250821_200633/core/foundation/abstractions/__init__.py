"""
Core Abstractions Package
========================

Core abstraction layers providing universal interfaces for AST manipulation,
framework abstraction, and language detection across the TestMaster ecosystem.

Author: Agent E - Infrastructure Consolidation
"""

from .ast_abstraction import UniversalAST
from .framework_abstraction import UniversalTestSuite, TestMetadata
from .language_detection import UniversalLanguageDetector, CodebaseProfile

__all__ = [
    'UniversalAST',
    'UniversalTestSuite',
    'TestMetadata', 
    'UniversalLanguageDetector',
    'CodebaseProfile'
]