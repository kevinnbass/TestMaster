"""
Semantic Relationship Discovery Module

This module discovers and analyzes relationships between semantic concepts
including causal, correlation, containment, and transformation relationships.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Optional

from .semantic_types import SemanticConcept, SemanticRelationship


class RelationshipDiscoverer:
    """Discovers relationships between semantic concepts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.discovery_stats = {
            "causal_relationships": 0,
            "correlation_relationships": 0,
            "containment_relationships": 0,
            "transformation_relationships": 0,
            "total_relationships": 0
        }
    
    async def discover_causal_relationships(self, concepts: List[SemanticConcept]) -> List[SemanticRelationship]:
        """Discover causal relationships between concepts"""
        relationships = []
        
        try:
            for i, concept_a in enumerate(concepts):
                for concept_b in concepts[i+1:]:
                    # Check if there's temporal evidence of causation
                    causal_strength = await self._analyze_causal_evidence(concept_a, concept_b)
                    
                    if causal_strength > 0.5:
                        relationships.append(SemanticRelationship(
                            relationship_id=f"causal_{concept_a.concept_id}_{concept_b.concept_id}",
                            source_concept=concept_a.concept_id,
                            target_concept=concept_b.concept_id,
                            relationship_type="causes",
                            strength=causal_strength,
                            directionality="source_to_target",
                            evidence=[{"type": "temporal_sequence", "strength": causal_strength}],
                            systems_involved=list(set(
                                list(concept_a.system_manifestations.keys()) + 
                                list(concept_b.system_manifestations.keys())
                            ))
                        ))
            
            self.discovery_stats["causal_relationships"] += len(relationships)
            return relationships
            
        except Exception as e:
            self.logger.error(f"Causal relationship discovery failed: {e}")
            return []
    
    async def discover_correlation_relationships(self, concepts: List[SemanticConcept]) -> List[SemanticRelationship]:
        """Discover correlation relationships between concepts"""
        relationships = []
        
        try:
            for i, concept_a in enumerate(concepts):
                for concept_b in concepts[i+1:]:
                    correlation_strength = self._calculate_correlation_strength(concept_a, concept_b)
                    
                    if correlation_strength > 0.5:
                        relationships.append(SemanticRelationship(
                            relationship_id=f"correlation_{concept_a.concept_id}_{concept_b.concept_id}",
                            source_concept=concept_a.concept_id,
                            target_concept=concept_b.concept_id,
                            relationship_type="correlates",
                            strength=correlation_strength,
                            directionality="bidirectional",
                            evidence=[{"type": "co_occurrence", "strength": correlation_strength}],
                            systems_involved=list(set(
                                list(concept_a.system_manifestations.keys()) + 
                                list(concept_b.system_manifestations.keys())
                            ))
                        ))
            
            self.discovery_stats["correlation_relationships"] += len(relationships)
            return relationships
            
        except Exception as e:
            self.logger.error(f"Correlation relationship discovery failed: {e}")
            return []
    
    async def discover_containment_relationships(self, concepts: List[SemanticConcept]) -> List[SemanticRelationship]:
        """Discover containment relationships (concept A contains concept B)"""
        relationships = []
        
        try:
            for concept_a in concepts:
                for concept_b in concepts:
                    if concept_a.concept_id != concept_b.concept_id:
                        containment_strength = self._analyze_containment(concept_a, concept_b)
                        
                        if containment_strength > 0.6:
                            relationships.append(SemanticRelationship(
                                relationship_id=f"containment_{concept_a.concept_id}_{concept_b.concept_id}",
                                source_concept=concept_a.concept_id,
                                target_concept=concept_b.concept_id,
                                relationship_type="contains",
                                strength=containment_strength,
                                directionality="source_to_target",
                                evidence=[{"type": "abstraction_hierarchy", "strength": containment_strength}],
                                systems_involved=list(set(
                                    list(concept_a.system_manifestations.keys()) + 
                                    list(concept_b.system_manifestations.keys())
                                ))
                            ))
            
            self.discovery_stats["containment_relationships"] += len(relationships)
            return relationships
            
        except Exception as e:
            self.logger.error(f"Containment relationship discovery failed: {e}")
            return []
    
    async def discover_transformation_relationships(self, concepts: List[SemanticConcept]) -> List[SemanticRelationship]:
        """Discover transformation relationships (concept A transforms into concept B)"""
        relationships = []
        
        try:
            for i, concept_a in enumerate(concepts):
                for concept_b in concepts[i+1:]:
                    transformation_strength = await self._analyze_transformation(concept_a, concept_b)
                    
                    if transformation_strength > 0.5:
                        # Determine direction based on temporal order and abstraction
                        if (concept_a.discovered_at < concept_b.discovered_at or 
                            concept_a.abstraction_level < concept_b.abstraction_level):
                            source, target = concept_a, concept_b
                        else:
                            source, target = concept_b, concept_a
                        
                        relationships.append(SemanticRelationship(
                            relationship_id=f"transform_{source.concept_id}_{target.concept_id}",
                            source_concept=source.concept_id,
                            target_concept=target.concept_id,
                            relationship_type="transforms",
                            strength=transformation_strength,
                            directionality="source_to_target",
                            evidence=[{"type": "state_transition", "strength": transformation_strength}],
                            systems_involved=list(set(
                                list(source.system_manifestations.keys()) + 
                                list(target.system_manifestations.keys())
                            ))
                        ))
            
            self.discovery_stats["transformation_relationships"] += len(relationships)
            return relationships
            
        except Exception as e:
            self.logger.error(f"Transformation relationship discovery failed: {e}")
            return []
    
    async def _analyze_causal_evidence(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Analyze evidence for causal relationship between concepts"""
        try:
            # Check for shared systems
            shared_systems = set(concept_a.system_manifestations.keys()) & set(concept_b.system_manifestations.keys())
            if not shared_systems:
                return 0.0
            
            # Check temporal relationship
            if concept_a.discovered_at < concept_b.discovered_at:
                time_diff = (concept_b.discovered_at - concept_a.discovered_at).total_seconds()
                if time_diff < 3600:  # Within an hour
                    causal_strength = 0.7
                    
                    # Boost if concepts are from related domains
                    domain_a = concept_a.semantic_properties.get("domain", "")
                    domain_b = concept_b.semantic_properties.get("domain", "")
                    if domain_a and domain_b and domain_a == domain_b:
                        causal_strength += 0.2
                    
                    return min(1.0, causal_strength)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_correlation_strength(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Calculate correlation strength between two concepts"""
        try:
            correlation_factors = []
            
            # Confidence correlation
            conf_similarity = 1.0 - abs(concept_a.confidence - concept_b.confidence)
            correlation_factors.append(conf_similarity)
            
            # System overlap
            systems_a = set(concept_a.system_manifestations.keys())
            systems_b = set(concept_b.system_manifestations.keys())
            system_overlap = len(systems_a & systems_b) / max(len(systems_a | systems_b), 1)
            correlation_factors.append(system_overlap)
            
            # Temporal proximity
            time_diff = abs((concept_a.discovered_at - concept_b.discovered_at).total_seconds())
            temporal_proximity = max(0, 1.0 - time_diff / 3600)  # Normalize to 1 hour
            correlation_factors.append(temporal_proximity)
            
            return statistics.mean(correlation_factors)
            
        except Exception:
            return 0.0
    
    def _analyze_containment(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Analyze if concept A contains concept B"""
        try:
            # Higher abstraction level suggests containment of lower level
            if concept_a.abstraction_level > concept_b.abstraction_level:
                level_diff = concept_a.abstraction_level - concept_b.abstraction_level
                abstraction_strength = min(1.0, level_diff / 3.0)
                
                # Check if A's systems encompass B's systems
                systems_a = set(concept_a.system_manifestations.keys())
                systems_b = set(concept_b.system_manifestations.keys())
                
                if systems_b.issubset(systems_a):
                    system_containment = 1.0
                else:
                    system_containment = len(systems_a & systems_b) / max(len(systems_b), 1)
                
                return (abstraction_strength * 0.7 + system_containment * 0.3)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _analyze_transformation(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Analyze if concept A can transform into concept B"""
        try:
            # Check for similar base properties but different states
            domain_match = (
                concept_a.semantic_properties.get("domain") == 
                concept_b.semantic_properties.get("domain")
            )
            
            if not domain_match:
                return 0.0
            
            # Check for evolution indicators
            type_evolution = (
                concept_a.concept_type in ["entity", "pattern"] and
                concept_b.concept_type in ["behavior", "relationship"]
            )
            
            # Check abstraction evolution
            abstraction_evolution = abs(concept_a.abstraction_level - concept_b.abstraction_level) == 1
            
            # Calculate transformation strength
            if type_evolution or abstraction_evolution:
                base_strength = 0.6
                
                # Boost for temporal proximity
                time_diff = abs((concept_a.discovered_at - concept_b.discovered_at).total_seconds())
                if time_diff < 1800:  # Within 30 minutes
                    base_strength += 0.2
                
                # Boost for confidence similarity
                conf_similarity = 1.0 - abs(concept_a.confidence - concept_b.confidence)
                base_strength += conf_similarity * 0.2
                
                return min(1.0, base_strength)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def validate_relationships(self, relationships: List[SemanticRelationship]) -> List[SemanticRelationship]:
        """Validate and strengthen discovered relationships"""
        validated = []
        
        for relationship in relationships:
            # Apply validation criteria
            if self._validate_relationship(relationship):
                # Potentially strengthen relationship based on additional evidence
                strengthened = await self._strengthen_relationship(relationship)
                validated.append(strengthened)
        
        self.discovery_stats["total_relationships"] = len(validated)
        return validated
    
    def _validate_relationship(self, relationship: SemanticRelationship) -> bool:
        """Validate a relationship meets quality criteria"""
        # Check minimum strength
        if relationship.strength < 0.3:
            return False
        
        # Check evidence quality
        if not relationship.evidence:
            return False
        
        # Check system involvement
        if not relationship.systems_involved:
            return False
        
        return True
    
    async def _strengthen_relationship(self, relationship: SemanticRelationship) -> SemanticRelationship:
        """Strengthen relationship with additional evidence"""
        # Could add additional analysis here to boost confidence
        # For now, return as-is
        return relationship
    
    def get_discovery_stats(self) -> Dict[str, int]:
        """Get relationship discovery statistics"""
        return self.discovery_stats.copy()