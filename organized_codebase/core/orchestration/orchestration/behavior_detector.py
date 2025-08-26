"""
Emergent Behavior Detection Module

This module detects and analyzes emergent behaviors across systems including
adaptive, self-organizing, optimization, and coordination behaviors.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Any

from .semantic_types import (
    SemanticConcept, SemanticRelationship, EmergentBehavior
)


class EmergentBehaviorDetector:
    """Detects emergent behaviors in semantic systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detection_stats = {
            "adaptive_behaviors": 0,
            "self_organizing_behaviors": 0,
            "optimization_behaviors": 0,
            "coordination_behaviors": 0,
            "total_behaviors": 0
        }
    
    async def detect_emergent_behaviors(self, 
                                       concepts: Dict[str, SemanticConcept],
                                       relationships: Dict[str, SemanticRelationship],
                                       pattern_history: Dict[str, deque]) -> List[EmergentBehavior]:
        """Detect all types of emergent behaviors"""
        behaviors = []
        
        try:
            # Detect different behavior types
            adaptive = await self._detect_adaptive_behaviors(concepts, relationships, pattern_history)
            behaviors.extend(adaptive)
            
            self_organizing = await self._detect_self_organizing_behaviors(concepts, relationships)
            behaviors.extend(self_organizing)
            
            optimization = await self._detect_optimization_behaviors(concepts, pattern_history)
            behaviors.extend(optimization)
            
            coordination = await self._detect_coordination_behaviors(concepts, relationships)
            behaviors.extend(coordination)
            
            # Validate and assess behaviors
            validated_behaviors = self._validate_behaviors(behaviors)
            
            self.detection_stats["total_behaviors"] = len(validated_behaviors)
            return validated_behaviors
            
        except Exception as e:
            self.logger.error(f"Emergent behavior detection failed: {e}")
            return []
    
    async def _detect_adaptive_behaviors(self, 
                                        concepts: Dict[str, SemanticConcept],
                                        relationships: Dict[str, SemanticRelationship],
                                        pattern_history: Dict[str, deque]) -> List[EmergentBehavior]:
        """Detect adaptive behaviors in the system"""
        behaviors = []
        
        try:
            # Look for concepts that change over time
            evolving_concepts = [
                c for c in concepts.values()
                if len(c.evolution_history) > 3
            ]
            
            if len(evolving_concepts) > 5:
                behaviors.append(EmergentBehavior(
                    behavior_id=f"adaptive_{int(datetime.now().timestamp())}",
                    behavior_name="System Adaptation",
                    behavior_type="adaptive",
                    complexity_score=min(1.0, len(evolving_concepts) / 10),
                    emergence_conditions=["concept_evolution", "pattern_learning"],
                    participating_systems=list(set(
                        system for c in evolving_concepts
                        for system in c.system_manifestations.keys()
                    )),
                    observable_effects=["concept_refinement", "accuracy_improvement"],
                    predictive_indicators=["evolution_rate", "confidence_growth"],
                    stability_assessment="evolving"
                ))
            
            self.detection_stats["adaptive_behaviors"] += len(behaviors)
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Adaptive behavior detection failed: {e}")
            return []
    
    async def _detect_self_organizing_behaviors(self,
                                               concepts: Dict[str, SemanticConcept],
                                               relationships: Dict[str, SemanticRelationship]) -> List[EmergentBehavior]:
        """Detect self-organizing behaviors"""
        behaviors = []
        
        try:
            # Look for clusters of highly connected concepts
            concept_connectivity = self._calculate_concept_connectivity(concepts, relationships)
            
            # Find highly connected clusters
            clusters = self._find_concept_clusters(concept_connectivity)
            
            if clusters:
                for cluster_id, cluster_concepts in enumerate(clusters):
                    if len(cluster_concepts) > 3:
                        behaviors.append(EmergentBehavior(
                            behavior_id=f"self_org_{cluster_id}_{int(datetime.now().timestamp())}",
                            behavior_name=f"Self-Organizing Cluster {cluster_id}",
                            behavior_type="self_organizing",
                            complexity_score=min(1.0, len(cluster_concepts) / 5),
                            emergence_conditions=["high_connectivity", "concept_clustering"],
                            participating_systems=list(set(
                                system for c_id in cluster_concepts
                                for system in concepts[c_id].system_manifestations.keys()
                                if c_id in concepts
                            )),
                            observable_effects=["cluster_formation", "pattern_emergence"],
                            predictive_indicators=["connectivity_growth", "cluster_stability"],
                            stability_assessment="stable"
                        ))
            
            self.detection_stats["self_organizing_behaviors"] += len(behaviors)
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Self-organizing behavior detection failed: {e}")
            return []
    
    async def _detect_optimization_behaviors(self,
                                           concepts: Dict[str, SemanticConcept],
                                           pattern_history: Dict[str, deque]) -> List[EmergentBehavior]:
        """Detect optimization behaviors"""
        behaviors = []
        
        try:
            # Look for improving performance patterns
            performance_concepts = [
                c for c in concepts.values()
                if "performance" in c.semantic_properties or
                "optimization" in c.semantic_properties
            ]
            
            if performance_concepts:
                # Check for improvement trends
                avg_confidence = sum(c.confidence for c in performance_concepts) / len(performance_concepts)
                
                if avg_confidence > 0.7:
                    behaviors.append(EmergentBehavior(
                        behavior_id=f"optimization_{int(datetime.now().timestamp())}",
                        behavior_name="Performance Optimization",
                        behavior_type="optimization",
                        complexity_score=avg_confidence,
                        emergence_conditions=["performance_monitoring", "feedback_loops"],
                        participating_systems=list(set(
                            system for c in performance_concepts
                            for system in c.system_manifestations.keys()
                        )),
                        observable_effects=["performance_improvement", "efficiency_gains"],
                        predictive_indicators=["optimization_rate", "resource_efficiency"],
                        stability_assessment="stable" if avg_confidence > 0.8 else "evolving"
                    ))
            
            self.detection_stats["optimization_behaviors"] += len(behaviors)
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Optimization behavior detection failed: {e}")
            return []
    
    async def _detect_coordination_behaviors(self,
                                           concepts: Dict[str, SemanticConcept],
                                           relationships: Dict[str, SemanticRelationship]) -> List[EmergentBehavior]:
        """Detect coordination behaviors between systems"""
        behaviors = []
        
        try:
            # Find cross-system relationships
            cross_system_relationships = [
                r for r in relationships.values()
                if len(r.systems_involved) > 1
            ]
            
            if len(cross_system_relationships) > 10:
                # Group by system pairs
                system_pairs = {}
                for rel in cross_system_relationships:
                    pair = tuple(sorted(rel.systems_involved[:2]))
                    if pair not in system_pairs:
                        system_pairs[pair] = []
                    system_pairs[pair].append(rel)
                
                # Create coordination behaviors for active pairs
                for (sys1, sys2), rels in system_pairs.items():
                    if len(rels) > 5:
                        avg_strength = sum(r.strength for r in rels) / len(rels)
                        
                        behaviors.append(EmergentBehavior(
                            behavior_id=f"coordination_{sys1}_{sys2}_{int(datetime.now().timestamp())}",
                            behavior_name=f"Coordination: {sys1}-{sys2}",
                            behavior_type="coordination",
                            complexity_score=min(1.0, len(rels) / 10 * avg_strength),
                            emergence_conditions=["cross_system_communication", "shared_goals"],
                            participating_systems=[sys1, sys2],
                            observable_effects=["synchronized_actions", "information_sharing"],
                            predictive_indicators=["message_frequency", "coordination_strength"],
                            stability_assessment="stable" if avg_strength > 0.7 else "evolving"
                        ))
            
            self.detection_stats["coordination_behaviors"] += len(behaviors)
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Coordination behavior detection failed: {e}")
            return []
    
    def _calculate_concept_connectivity(self, 
                                       concepts: Dict[str, SemanticConcept],
                                       relationships: Dict[str, SemanticRelationship]) -> Dict[str, int]:
        """Calculate connectivity score for each concept"""
        connectivity = {}
        
        for concept_id in concepts:
            connections = 0
            for rel in relationships.values():
                if concept_id in [rel.source_concept, rel.target_concept]:
                    connections += 1
            connectivity[concept_id] = connections
        
        return connectivity
    
    def _find_concept_clusters(self, connectivity: Dict[str, int]) -> List[List[str]]:
        """Find clusters of highly connected concepts"""
        clusters = []
        
        # Simple clustering based on connectivity threshold
        high_connectivity_concepts = [
            c_id for c_id, conn in connectivity.items()
            if conn > 3
        ]
        
        # Group connected concepts (simplified clustering)
        if high_connectivity_concepts:
            # For now, treat all highly connected concepts as one cluster
            # In production, use proper clustering algorithms
            clusters.append(high_connectivity_concepts)
        
        return clusters
    
    def _validate_behaviors(self, behaviors: List[EmergentBehavior]) -> List[EmergentBehavior]:
        """Validate and filter detected behaviors"""
        validated = []
        
        for behavior in behaviors:
            # Apply validation criteria
            if self._is_valid_behavior(behavior):
                # Assess stability
                behavior = self._assess_behavior_stability(behavior)
                validated.append(behavior)
        
        return validated
    
    def _is_valid_behavior(self, behavior: EmergentBehavior) -> bool:
        """Check if behavior meets validity criteria"""
        # Must have minimum complexity
        if behavior.complexity_score < 0.3:
            return False
        
        # Must involve at least one system
        if not behavior.participating_systems:
            return False
        
        # Must have observable effects
        if not behavior.observable_effects:
            return False
        
        return True
    
    def _assess_behavior_stability(self, behavior: EmergentBehavior) -> EmergentBehavior:
        """Assess and update behavior stability"""
        # Simple stability assessment based on complexity
        if behavior.complexity_score > 0.8:
            behavior.stability_assessment = "stable"
        elif behavior.complexity_score > 0.5:
            behavior.stability_assessment = "evolving"
        else:
            behavior.stability_assessment = "unstable"
        
        return behavior
    
    def get_detection_stats(self) -> Dict[str, int]:
        """Get behavior detection statistics"""
        return self.detection_stats.copy()