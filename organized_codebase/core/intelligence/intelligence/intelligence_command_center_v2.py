"""
Intelligence Command Center V2 - Facade
========================================

Backward-compatible facade for the modularized Intelligence Command Center.
This file provides the same interface as the original monolithic version
while delegating to the new modular components.

Agent D - Hour 6-8: Intelligence Command Center Modularization Complete
"""

from command_center import (
    IntelligenceCommandCenter,
    FrameworkType,
    OrchestrationPriority,
    OrchestrationStatus,
    ResourceType,
    FrameworkCapability,
    OrchestrationTask,
    ResourceAllocation,
    FrameworkHealthStatus,
    FrameworkController,
    AnalyticsFrameworkController,
    MLFrameworkController
)

# Re-export all classes for backward compatibility
__all__ = [
    'IntelligenceCommandCenter',
    'FrameworkType',
    'OrchestrationPriority',
    'OrchestrationStatus', 
    'ResourceType',
    'FrameworkCapability',
    'OrchestrationTask',
    'ResourceAllocation',
    'FrameworkHealthStatus',
    'FrameworkController',
    'AnalyticsFrameworkController',
    'MLFrameworkController'
]


# Factory function for easy instantiation
def create_intelligence_command_center(config=None):
    """Create and return a configured Intelligence Command Center"""
    return IntelligenceCommandCenter(config)


# Legacy class alias for backward compatibility
IntelligenceOrchestrator = IntelligenceCommandCenter