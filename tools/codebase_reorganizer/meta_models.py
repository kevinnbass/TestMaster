#!/usr/bin/env python3
"""
Meta-Reorganizer Data Models
============================

Data structures and models for the intelligence-driven reorganizer system.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class IntelligenceAnalysis:
    """Analysis results from your intelligence modules"""
    file_path: Path
    semantic_analysis: Dict[str, Any]  # From your semantic analyzers
    relationship_analysis: Dict[str, Any]  # From your relationship analyzers
    pattern_analysis: Dict[str, Any]  # From your pattern detectors
    ml_analysis: Dict[str, Any]  # From your ML analyzers
    dependency_analysis: Dict[str, Any]  # From your dependency resolvers
    confidence_score: float
    recommended_category: str
    reasoning: List[str]


@dataclass
class ModuleRelationship:
    """Relationships between modules discovered by your intelligence"""
    source_module: Path
    target_module: Path
    relationship_type: str  # 'imports', 'shares_patterns', 'semantic_similarity', etc.
    strength: float
    evidence: List[str]
