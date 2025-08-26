"""
Base Configuration Classes
=========================

Core configuration abstractions providing the foundation for
hierarchical configuration management.

Author: Agent E - Infrastructure Consolidation
"""

from .config_base import ConfigurationBase
from .validation import ConfigurationValidator
from .serialization import ConfigurationSerializer

__all__ = [
    'ConfigurationBase',
    'ConfigurationValidator',
    'ConfigurationSerializer'
]