"""
Universal Language Detection for TestMaster

Codebase-agnostic language detection and analysis system.
"""

from .universal_detector import (
    UniversalLanguageDetector, CodebaseProfile, LanguageInfo,
    FrameworkInfo, BuildSystemInfo, DependencyInfo, FileInfo
)

__all__ = [
    'UniversalLanguageDetector',
    'CodebaseProfile', 
    'LanguageInfo',
    'FrameworkInfo',
    'BuildSystemInfo',
    'DependencyInfo',
    'FileInfo'
]