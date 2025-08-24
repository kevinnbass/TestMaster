"""
Shared utilities for semantic_analysis modules

Contains common data structures, constants, and helper functions
used across all semantic_analysis submodules.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# TODO: Extract common data structures from original semantic_analysis.py
# TODO: Extract common constants and patterns
# TODO: Extract common helper functions

@dataclass
class AnalysisIssue:
    """Common issue structure for all analyzers"""
    issue_type: str
    severity: str
    location: str
    description: str
    recommendation: str
    impact: str


# Common patterns and constants will be extracted here
COMMON_PATTERNS = {
    # TODO: Extract from original module
}

def extract_common_patterns(tree: ast.AST, content: str) -> List[Dict[str, Any]]:
    """Extract common patterns from AST and content"""
    # TODO: Implement common pattern extraction
    return []

def calculate_complexity_score(node: ast.AST) -> int:
    """Calculate complexity score for a node"""
    # TODO: Implement common complexity calculation
    return 0
