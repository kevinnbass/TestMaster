"""
Concept Unification Module

This module handles the unification and merging of similar semantic concepts
across different systems to create a unified knowledge representation.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Optional

from .semantic_types import SemanticConcept


class ConceptUnifier:
    """Unifies and merges similar semantic concepts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.unification_stats = {
            "concepts_unified": 0,
            "concepts_merged": 0,
            "similarity_checks": 0,
            "groups_created": 0
        }
    
    async def unify_concepts(self, new_concepts: List[SemanticConcept],
                            existing_concepts: Dict[str, SemanticConcept]) -> List[SemanticConcept]:
        """Unify new concepts with existing knowledge base"""
        unified = []
        
        try:
            # Group similar concepts
            concept_groups = self._group_concepts_by_similarity(
                new_concepts + list(existing_concepts.values())
            )
            
            # Create unified concepts from groups
            for group in concept_groups:
                if len(group) > 1:
                    unified_concept = await self._create_unified_concept(group)
                    if unified_concept:
                        unified.append(unified_concept)
                        self.unification_stats["concepts_unified"] += 1
                else:
                    # Single concept, no unification needed
                    unified.append(group[0])
            
            self.unification_stats["groups_created"] = len(concept_groups)
            return unified
            
        except Exception as e:
            self.logger.error(f"Concept unification failed: {e}")
            return new_concepts
    
    def _group_concepts_by_similarity(self, concepts: List[SemanticConcept]) -> List[List[SemanticConcept]]:
        """Group concepts by semantic similarity"""
        groups = []
        used_concepts = set()
        
        for concept in concepts:
            if concept.concept_id in used_concepts:
                continue
            
            similar_group = [concept]
            used_concepts.add(concept.concept_id)
            
            # Find similar concepts
            for other_concept in concepts:
                if (other_concept.concept_id != concept.concept_id and 
                    other_concept.concept_id not in used_concepts):
                    
                    similarity = self._calculate_concept_similarity(concept, other_concept)
                    self.unification_stats["similarity_checks"] += 1
                    
                    if similarity > 0.7:  # High similarity threshold
                        similar_group.append(other_concept)
                        used_concepts.add(other_concept.concept_id)
            
            groups.append(similar_group)
        
        return groups
    
    def _calculate_concept_similarity(self, concept_a: SemanticConcept, 
                                    concept_b: SemanticConcept) -> float:
        """Calculate similarity between two concepts"""
        try:
            similarity_factors = []
            
            # Type similarity
            if concept_a.concept_type == concept_b.concept_type:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.3)
            
            # Name similarity (simple string similarity)
            name_a_tokens = set(concept_a.concept_name.lower().split())
            name_b_tokens = set(concept_b.concept_name.lower().split())
            name_similarity = len(name_a_tokens & name_b_tokens) / max(
                len(name_a_tokens | name_b_tokens), 1
            )
            similarity_factors.append(name_similarity)
            
            # Abstraction level similarity
            abs_diff = abs(concept_a.abstraction_level - concept_b.abstraction_level)
            abs_similarity = max(0, 1.0 - abs_diff / 5.0)
            similarity_factors.append(abs_similarity)
            
            # Domain similarity (from semantic properties)
            domain_a = concept_a.semantic_properties.get("domain", "")
            domain_b = concept_b.semantic_properties.get("domain", "")
            domain_similarity = (
                1.0 if domain_a == domain_b 
                else 0.5 if domain_a and domain_b 
                else 0.0
            )
            similarity_factors.append(domain_similarity)
            
            # System manifestation overlap
            systems_a = set(concept_a.system_manifestations.keys())
            systems_b = set(concept_b.system_manifestations.keys())
            if systems_a and systems_b:
                system_overlap = len(systems_a & systems_b) / len(systems_a | systems_b)
                similarity_factors.append(system_overlap)
            
            return statistics.mean(similarity_factors)
            
        except Exception:
            return 0.0
    
    async def _create_unified_concept(self, concept_group: List[SemanticConcept]) -> Optional[SemanticConcept]:
        """Create unified concept from a group of similar concepts"""
        try:
            if not concept_group:
                return None
            
            # Aggregate properties
            avg_confidence = statistics.mean([c.confidence for c in concept_group])
            
            # Find most common type
            type_counts = {}
            for c in concept_group:
                type_counts[c.concept_type] = type_counts.get(c.concept_type, 0) + 1
            most_common_type = max(type_counts, key=type_counts.get)
            
            # Average abstraction level
            avg_abstraction = int(statistics.mean([c.abstraction_level for c in concept_group]))
            
            # Merge system manifestations
            unified_manifestations = {}
            for concept in concept_group:
                for system, manifest in concept.system_manifestations.items():
                    if system not in unified_manifestations:
                        unified_manifestations[system] = manifest
                    else:
                        # Merge manifestations for same system
                        if isinstance(manifest, dict) and isinstance(unified_manifestations[system], dict):
                            unified_manifestations[system].update(manifest)
            
            # Merge semantic properties
            unified_properties = {}
            for concept in concept_group:
                unified_properties.update(concept.semantic_properties)
            unified_properties["unified"] = True
            unified_properties["source_concepts"] = [c.concept_id for c in concept_group]
            unified_properties["unification_time"] = datetime.now().isoformat()
            
            # Merge related concepts
            unified_related = []
            for concept in concept_group:
                unified_related.extend(concept.related_concepts)
            unified_related = list(set(unified_related))
            
            # Create evolution history entry
            evolution_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "unified",
                "source_concepts": len(concept_group),
                "confidence_range": [
                    min(c.confidence for c in concept_group),
                    max(c.confidence for c in concept_group)
                ]
            }
            
            return SemanticConcept(
                concept_id=f"unified_{int(datetime.now().timestamp())}_{len(concept_group)}",
                concept_name=self._generate_unified_name(concept_group),
                concept_type=most_common_type,
                confidence=avg_confidence,
                abstraction_level=avg_abstraction,
                system_manifestations=unified_manifestations,
                semantic_properties=unified_properties,
                related_concepts=unified_related,
                evolution_history=[evolution_entry]
            )
            
        except Exception as e:
            self.logger.error(f"Unified concept creation failed: {e}")
            return None
    
    def _generate_unified_name(self, concept_group: List[SemanticConcept]) -> str:
        """Generate name for unified concept"""
        # Extract common tokens from names
        all_tokens = []
        for concept in concept_group:
            tokens = concept.concept_name.split(":")
            if tokens:
                all_tokens.append(tokens[0].strip())
        
        # Find most common token
        if all_tokens:
            token_counts = {}
            for token in all_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            most_common = max(token_counts, key=token_counts.get)
            return f"Unified: {most_common}"
        
        return f"Unified Concept ({len(concept_group)} sources)"
    
    def find_existing_concept(self, new_concept: SemanticConcept,
                            existing_concepts: Dict[str, SemanticConcept]) -> Optional[SemanticConcept]:
        """Find existing similar concept in knowledge base"""
        for existing_concept in existing_concepts.values():
            similarity = self._calculate_concept_similarity(existing_concept, new_concept)
            if similarity > 0.8:  # Very high similarity
                return existing_concept
        return None
    
    async def merge_concepts(self, existing: SemanticConcept, new: SemanticConcept):
        """Merge new concept information with existing concept"""
        try:
            # Update confidence with weighted average
            weight = 0.3  # Weight for new concept
            existing.confidence = (existing.confidence * (1 - weight) + new.confidence * weight)
            
            # Merge system manifestations
            for system, manifest in new.system_manifestations.items():
                if system not in existing.system_manifestations:
                    existing.system_manifestations[system] = manifest
                elif isinstance(manifest, dict) and isinstance(existing.system_manifestations[system], dict):
                    existing.system_manifestations[system].update(manifest)
            
            # Merge semantic properties
            existing.semantic_properties.update(new.semantic_properties)
            
            # Add unique related concepts
            new_related = set(existing.related_concepts)
            new_related.update(new.related_concepts)
            existing.related_concepts = list(new_related)
            
            # Add to evolution history
            existing.evolution_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "merged",
                "source_concept": new.concept_id,
                "confidence_update": new.confidence,
                "properties_added": len(new.semantic_properties)
            })
            
            self.unification_stats["concepts_merged"] += 1
            
        except Exception as e:
            self.logger.error(f"Concept merge failed: {e}")
    
    def get_unification_stats(self) -> Dict[str, int]:
        """Get unification statistics"""
        return self.unification_stats.copy()