#!/usr/bin/env python3
"""
Relationship Data Classes
=========================

Data structures for relationship analysis results.
"""

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ModuleRelationship:
    """Relationship between two modules"""
    source_module: Path
    target_module: Path
    relationship_type: str  # 'imports', 'calls', 'shares_data', 'extends'
    strength: float
    evidence: List[str]
    bidirectional: bool


@dataclass
class ClassRelationship:
    """Relationship between two classes"""
    source_class: str
    target_class: str
    relationship_type: str  # 'inherits', 'composes', 'uses', 'associates'
    strength: float
    evidence: List[str]


@dataclass
class FunctionCallGraph:
    """Graph of function calls within and between modules"""
    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # source, target, weight
    clusters: List[List[str]]


@dataclass
class CouplingMetrics:
    """Metrics related to coupling between components"""
    overall_coupling: float
    import_coupling: float
    functional_coupling: float
    data_coupling: float
    inheritance_coupling: float
    highly_coupled_components: List[str]
    loosely_coupled_components: List[str]


@dataclass
class RelationshipCluster:
    """A cluster of related components"""
    cluster_id: str
    components: List[str]
    cohesion_score: float
    coupling_score: float
    recommended_action: str


@dataclass
class RelationshipAnalysis:
    """Complete relationship analysis result"""
    import_relationships: List[ModuleRelationship]
    class_relationships: List[ClassRelationship]
    function_call_graph: FunctionCallGraph
    coupling_metrics: CouplingMetrics
    clusters: List[RelationshipCluster]
    recommendations: List[str]
