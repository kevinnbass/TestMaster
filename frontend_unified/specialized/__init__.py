"""
STEELCLAD MODULE PACKAGE: Specialized Dashboard Components
==========================================================

Modularized dashboard components extracted from large monolithic files
using STEELCLAD Anti-Regression Modularization Protocol.

Modules:
- enhanced_linkage_dashboard_modular: Core coordination module
- linkage_analysis: Analysis engine and AST parsing
- data_generator: Live data simulation system  
- api_routes: Flask route handlers and API endpoints

Author: Agent X (STEELCLAD Modularization)
"""

from .enhanced_linkage_dashboard_modular import EnhancedLinkageDashboard
from .linkage_analysis import quick_linkage_analysis, analyze_file_quick, get_codebase_statistics
from .data_generator import LiveDataGenerator
from .api_routes import register_routes

__all__ = [
    'EnhancedLinkageDashboard',
    'quick_linkage_analysis', 
    'analyze_file_quick',
    'get_codebase_statistics',
    'LiveDataGenerator',
    'register_routes'
]