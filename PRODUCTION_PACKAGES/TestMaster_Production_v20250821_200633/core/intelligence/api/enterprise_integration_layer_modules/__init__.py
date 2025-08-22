"""
Enterprise_Integration_Layer Module Split
================================

This module was split from the original enterprise_integration_layer.py to maintain
modules under 1000 lines while preserving all functionality.
"""

# Import all components to maintain backward compatibility
from .enterprise_integration_layer_core import *
from .enterprise_integration_layer_processing import *
