"""
TestMaster Mapping Module

Bidirectional test-module mapping system inspired by Agency-Swarm's 
hierarchical thread mapping patterns.

Provides:
- Test ↔ Module relationship tracking
- Integration test → multiple module mappings
- Real-time mapping updates on file changes
- Dependency graph construction
"""

from .test_mapper import TestMapper, TestModuleMapping
from .dependency_tracker import DependencyTracker, ModuleDependency
from .mapping_cache import MappingCache

__all__ = [
    "TestMapper",
    "TestModuleMapping", 
    "DependencyTracker",
    "ModuleDependency",
    "MappingCache"
]