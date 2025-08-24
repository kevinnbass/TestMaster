"""
Neo4J_Dominator Module - Import Redirect
==========================================

This module was split into smaller components for maintainability.
All original functionality is preserved through the split modules.

Original module archived: archive/oversized_modules_*/
Split modules location: neo4j_dominator_modules/
"""

# Import all components from split modules to maintain backward compatibility
from .neo4j_dominator_modules import *

# Preserve original module's public interface
__all__ = []
try:
    # Import any __all__ definitions from split modules
    from .neo4j_dominator_modules import __all__ as split_all
    __all__.extend(split_all)
except ImportError:
    pass

# Add common module metadata
__version__ = "2.0.0-modularized"
__split_timestamp__ = "2025-08-21T04:20:18.426090"
__original_lines__ = "1000+ (see archive for original)"
