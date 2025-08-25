"""
Business Logic Analyzer V2 - Facade
====================================

Backward-compatible facade for the modularized Business Logic Analyzer.
This file provides the same interface as the original monolithic version
while delegating to the new modular components.

Agent D - Hour 8-10: Business Logic Analyzer Modularization Complete
"""

from business_logic import (
    BusinessLogicAnalyzer,
    BusinessRule,
    DomainEntity,
    BusinessProcess,
    create_business_logic_analyzer
)

# Re-export all classes for backward compatibility
__all__ = [
    'BusinessLogicAnalyzer',
    'BusinessRule',
    'DomainEntity',
    'BusinessProcess',
    'create_business_logic_analyzer'
]

# Legacy alias for backward compatibility
BusinessAnalyzer = BusinessLogicAnalyzer