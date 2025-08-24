#!/usr/bin/env python3
"""
Compliance Audit Data Classes
============================

Data structures for compliance auditing results.
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class ComplianceIssue:
    """Represents a compliance issue"""
    file_path: str
    line_number: int
    rule_number: int
    rule_description: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    suggestion: str


@dataclass
class ComplianceReport:
    """Complete compliance report"""
    total_files_analyzed: int = 0
    total_functions_analyzed: int = 0
    total_lines_analyzed: int = 0
    issues_found: List[ComplianceIssue] = field(default_factory=list)
    compliance_score: float = 0.0
    rule_compliance: Dict[int, bool] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)

