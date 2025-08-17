"""
TestMaster Overview Module - Layer 3

Codebase intelligence and functional structure mapping
for comprehensive understanding and analysis.

Provides:
- Functional structure mapping and module relationships
- Coverage intelligence with critical path identification
- Regression tracking and predictive failure detection
- API surface tracking and business logic identification
"""

from .structure_mapper import StructureMapper, ModuleRelationship, FunctionalMap
from .coverage_intelligence import CoverageIntelligence, CriticalPath, CoverageGap
from .regression_tracker import RegressionTracker, RegressionPattern, FailurePrediction

__all__ = [
    "StructureMapper",
    "ModuleRelationship", 
    "FunctionalMap",
    "CoverageIntelligence",
    "CriticalPath",
    "CoverageGap",
    "RegressionTracker",
    "RegressionPattern",
    "FailurePrediction"
]