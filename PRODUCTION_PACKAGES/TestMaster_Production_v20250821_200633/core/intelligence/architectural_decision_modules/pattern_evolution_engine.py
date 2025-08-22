"""
Design Pattern Evolution Engine
==============================

Modularized from architectural_decision_engine.py for better maintainability.
Analyzes and recommends design pattern evolution using graph-based analysis.

Author: Agent E - Infrastructure Consolidation
"""

import networkx as nx
import logging
from datetime import timedelta
from typing import Dict, List, Optional, Any

from .data_models import (
    ArchitecturalPattern, PatternEvolution
)

logger = logging.getLogger(__name__)


class DesignPatternEvolutionEngine:
    """Analyzes and recommends design pattern evolution"""
    
    def __init__(self):
        self.pattern_relationships = self._build_pattern_graph()
        self.evolution_paths = self._define_evolution_paths()
    
    def _build_pattern_graph(self) -> nx.DiGraph:
        """Build graph of pattern relationships and evolution paths"""
        graph = nx.DiGraph()
        
        # Add patterns as nodes
        for pattern in ArchitecturalPattern:
            graph.add_node(pattern)
        
        # Add evolution relationships
        evolution_edges = [
            (ArchitecturalPattern.MONOLITH, ArchitecturalPattern.MODULAR_MONOLITH),
            (ArchitecturalPattern.MODULAR_MONOLITH, ArchitecturalPattern.MICROSERVICES),
            (ArchitecturalPattern.LAYERED, ArchitecturalPattern.HEXAGONAL),
            (ArchitecturalPattern.MICROSERVICES, ArchitecturalPattern.SERVICE_MESH),
            (ArchitecturalPattern.MICROSERVICES, ArchitecturalPattern.SERVERLESS),
            (ArchitecturalPattern.MONOLITH, ArchitecturalPattern.STRANGLER_FIG),
            (ArchitecturalPattern.LAYERED, ArchitecturalPattern.EVENT_DRIVEN),
            (ArchitecturalPattern.EVENT_DRIVEN, ArchitecturalPattern.CQRS),
        ]
        
        for source, target in evolution_edges:
            graph.add_edge(source, target, weight=1.0)
        
        return graph
    
    def _define_evolution_paths(self) -> Dict[ArchitecturalPattern, List[List[ArchitecturalPattern]]]:
        """Define possible evolution paths for each pattern"""
        paths = {}
        
        for pattern in ArchitecturalPattern:
            paths[pattern] = []
            
            # Find all paths from current pattern to other patterns
            for target_pattern in ArchitecturalPattern:
                if pattern != target_pattern:
                    try:
                        path = nx.shortest_path(self.pattern_relationships, pattern, target_pattern)
                        if len(path) <= 4:  # Reasonable evolution path length
                            paths[pattern].append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        return paths
    
    def analyze_pattern_evolution(self, current_patterns: List[ArchitecturalPattern], 
                                 target_requirements: Dict[str, Any]) -> List[PatternEvolution]:
        """Analyze how patterns should evolve to meet requirements"""
        evolutions = []
        
        for current_pattern in current_patterns:
            # Find best target pattern based on requirements
            target_pattern = self._select_target_pattern(current_pattern, target_requirements)
            
            if target_pattern and target_pattern != current_pattern:
                # Find evolution path
                try:
                    evolution_path = nx.shortest_path(self.pattern_relationships, 
                                                    current_pattern, target_pattern)
                    
                    evolution = PatternEvolution(
                        pattern=current_pattern,
                        current_state={"pattern": current_pattern.value},
                        evolution_path=[p.value for p in evolution_path],
                        target_state={"pattern": target_pattern.value},
                        migration_steps=self._generate_migration_steps(evolution_path),
                        risk_factors=self._assess_evolution_risks(evolution_path),
                        success_criteria=self._define_success_criteria(current_pattern, target_pattern),
                        timeline=self._estimate_evolution_timeline(evolution_path)
                    )
                    
                    evolutions.append(evolution)
                
                except nx.NetworkXNoPath:
                    logger.warning(f"No evolution path found from {current_pattern} to {target_pattern}")
        
        return evolutions
    
    def _select_target_pattern(self, current_pattern: ArchitecturalPattern, 
                              requirements: Dict[str, Any]) -> Optional[ArchitecturalPattern]:
        """Select best target pattern based on requirements"""
        # Simplified pattern selection logic
        if requirements.get('scalability_required', False):
            if current_pattern == ArchitecturalPattern.MONOLITH:
                return ArchitecturalPattern.MICROSERVICES
            elif current_pattern == ArchitecturalPattern.LAYERED:
                return ArchitecturalPattern.EVENT_DRIVEN
        
        if requirements.get('maintainability_focus', False):
            if current_pattern == ArchitecturalPattern.MONOLITH:
                return ArchitecturalPattern.MODULAR_MONOLITH
            elif current_pattern == ArchitecturalPattern.LAYERED:
                return ArchitecturalPattern.HEXAGONAL
        
        if requirements.get('event_driven_needed', False):
            if current_pattern in [ArchitecturalPattern.LAYERED, ArchitecturalPattern.MONOLITH]:
                return ArchitecturalPattern.EVENT_DRIVEN
        
        return None
    
    def _generate_migration_steps(self, evolution_path: List[ArchitecturalPattern]) -> List[str]:
        """Generate migration steps for pattern evolution"""
        steps = []
        
        for i in range(len(evolution_path) - 1):
            current = evolution_path[i]
            next_pattern = evolution_path[i + 1]
            
            step = f"Migrate from {current.value} to {next_pattern.value}"
            steps.append(step)
            
            # Add specific migration guidance
            if current == ArchitecturalPattern.MONOLITH and next_pattern == ArchitecturalPattern.MODULAR_MONOLITH:
                steps.extend([
                    "Identify bounded contexts within monolith",
                    "Extract modules with clear interfaces",
                    "Implement internal API boundaries"
                ])
            elif current == ArchitecturalPattern.MODULAR_MONOLITH and next_pattern == ArchitecturalPattern.MICROSERVICES:
                steps.extend([
                    "Extract modules as separate services",
                    "Implement service discovery",
                    "Set up inter-service communication"
                ])
        
        return steps
    
    def _assess_evolution_risks(self, evolution_path: List[ArchitecturalPattern]) -> List[str]:
        """Assess risks in pattern evolution"""
        risks = []
        
        for i in range(len(evolution_path) - 1):
            current = evolution_path[i]
            next_pattern = evolution_path[i + 1]
            
            if current == ArchitecturalPattern.MONOLITH and next_pattern == ArchitecturalPattern.MICROSERVICES:
                risks.extend([
                    "Distributed system complexity",
                    "Network latency and reliability",
                    "Data consistency challenges"
                ])
            elif next_pattern == ArchitecturalPattern.EVENT_DRIVEN:
                risks.extend([
                    "Event ordering complexity",
                    "Eventual consistency challenges",
                    "Debugging distributed events"
                ])
        
        return risks
    
    def _define_success_criteria(self, current: ArchitecturalPattern, 
                                target: ArchitecturalPattern) -> List[str]:
        """Define success criteria for pattern evolution"""
        criteria = [
            "Zero downtime during migration",
            "No data loss during transition",
            "Performance metrics maintained or improved"
        ]
        
        if target == ArchitecturalPattern.MICROSERVICES:
            criteria.extend([
                "Services can be deployed independently",
                "Service boundaries are clearly defined",
                "Inter-service communication is reliable"
            ])
        elif target == ArchitecturalPattern.EVENT_DRIVEN:
            criteria.extend([
                "Events are processed reliably",
                "Event ordering is preserved where needed",
                "System remains responsive under event load"
            ])
        
        return criteria
    
    def _estimate_evolution_timeline(self, evolution_path: List[ArchitecturalPattern]) -> timedelta:
        """Estimate timeline for pattern evolution"""
        base_time = timedelta(weeks=2)  # Base migration time
        
        # Add time based on complexity of evolution
        complexity_multiplier = len(evolution_path) - 1
        
        # Add time for specific patterns
        for pattern in evolution_path:
            if pattern == ArchitecturalPattern.MICROSERVICES:
                base_time += timedelta(weeks=4)
            elif pattern == ArchitecturalPattern.EVENT_DRIVEN:
                base_time += timedelta(weeks=3)
            elif pattern == ArchitecturalPattern.CQRS:
                base_time += timedelta(weeks=2)
        
        return base_time * complexity_multiplier


__all__ = ['DesignPatternEvolutionEngine']