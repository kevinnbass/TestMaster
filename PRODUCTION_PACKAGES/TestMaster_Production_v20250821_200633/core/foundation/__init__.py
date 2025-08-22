"""
Core Foundation Layer
====================

Foundation layer providing core abstractions, shared utilities, and 
infrastructure components for the entire TestMaster system.

Author: Agent E - Infrastructure Consolidation
"""

# Core abstractions
from .abstractions.ast_abstraction import UniversalAST
from .abstractions.framework_abstraction import UniversalTestSuite, TestMetadata
from .abstractions.language_detection import UniversalLanguageDetector, CodebaseProfile

# Shared utilities
from .shared.shared_state import SharedState
from .shared.context_manager import ContextManager
from .shared.feature_flags import FeatureFlags

# Context management
from .context.tracking_manager import TrackingManager

# Observability
try:
    from .observability.unified_monitor import UnifiedObservabilitySystem as UnifiedMonitor
except ImportError:
    # Fallback class if import fails
    class UnifiedMonitor:
        def __init__(self):
            self.active = False

__all__ = [
    # Core abstractions
    'UniversalAST',
    'UniversalTestSuite', 
    'TestMetadata',
    'UniversalLanguageDetector',
    'CodebaseProfile',
    
    # Shared utilities
    'SharedState',
    'ContextManager', 
    'FeatureFlags',
    
    # Context management
    'TrackingManager',
    
    # Observability
    'UnifiedMonitor'
]