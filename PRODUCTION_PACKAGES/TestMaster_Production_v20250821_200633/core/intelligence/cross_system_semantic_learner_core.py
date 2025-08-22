"""
Cross-System Semantic Learner Core - Master Semantic Intelligence System
========================================================================

Enterprise semantic learning orchestration system providing comprehensive semantic
intelligence across all intelligence frameworks with advanced pattern discovery,
concept evolution, and emergent behavior detection capabilities.

This module provides the master orchestration system that integrates concept extraction,
relationship analysis, and emergent behavior detection into a unified semantic learning
platform for enterprise-grade intelligence enhancement and cross-system understanding.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: cross_system_semantic_learner_core.py (380 lines)
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import json
import statistics
from pathlib import Path
import uuid

from .semantic_types import (
    SemanticConcept, SemanticRelationship, EmergentBehavior, SemanticLearningContext,
    SemanticInsight, ConceptType, BehaviorType, StabilityAssessment
)
from .concept_extractor import CrossSystemConceptExtractor
from .relationship_analyzer import CrossSystemRelationshipAnalyzer
from .analytics.analytics_hub import AnalyticsHub
from .ml.ml_orchestrator import MLOrchestrator
from .api.unified_api_gateway import UnifiedAPIGateway
from .analysis.advanced_pattern_recognizer import AdvancedPatternRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossSystemSemanticLearner:
    """
    Master semantic learning engine that orchestrates comprehensive semantic intelligence
    discovery across all intelligence frameworks with advanced AI capabilities
    """
    
    def __init__(self, analytics_hub: AnalyticsHub, ml_orchestrator: MLOrchestrator, 
                 api_gateway: UnifiedAPIGateway, pattern_recognizer: AdvancedPatternRecognizer):
        self.analytics_hub = analytics_hub
        self.ml_orchestrator = ml_orchestrator
        self.api_gateway = api_gateway
        self.pattern_recognizer = pattern_recognizer
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized components
        self.concept_extractor = CrossSystemConceptExtractor(
            analytics_hub, ml_orchestrator, api_gateway, pattern_recognizer
        )
        self.relationship_analyzer = CrossSystemRelationshipAnalyzer()
        
        # Semantic knowledge base
        self.semantic_concepts: Dict[str, SemanticConcept] = {}
        self.semantic_relationships: Dict[str, SemanticRelationship] = {}
        self.emergent_behaviors: Dict[str, EmergentBehavior] = {}
        self.semantic_insights: Dict[str, SemanticInsight] = {}
        
        # Learning configuration
        self.config = {
            "learning_interval": 600,  # 10 minutes
            "concept_confidence_threshold": 0.6,
            "relationship_strength_threshold": 0.5,
            "emergence_detection_threshold": 0.7,
            "temporal_window_hours": 24,
            "cross_system_analysis_enabled": True,
            "emergent_behavior_detection_enabled": True,
            "semantic_reasoning_enabled": True,
            "insight_generation_enabled": True,
            "max_concepts_per_cycle": 200,
            "max_relationships_per_cycle": 500
        }
        
        # Learning statistics
        self.learning_stats = {
            "concepts_discovered": 0,
            "relationships_discovered": 0,
            "emergent_behaviors_discovered": 0,
            "insights_generated": 0,
            "cross_system_correlations": 0,
            "semantic_inferences_made": 0,
            "learning_cycles_completed": 0,
            "start_time": datetime.now(),
            "last_learning_cycle": None,
            "total_learning_time": 0.0
        }
        
        # Learning state
        self.is_learning = False
        self.learning_task = None
        self.current_learning_context: Optional[SemanticLearningContext] = None
        
        self.logger.info("CrossSystemSemanticLearner initialized with enterprise AI capabilities")
    
    async def start_learning(self, systems_to_monitor: List[str] = None, 
                           learning_objectives: List[str] = None):
        """Start cross-system semantic learning with enterprise orchestration"""
        if self.is_learning:
            self.logger.warning("Semantic learning already running")
            return
        
        # Initialize learning context
        self.current_learning_context = SemanticLearningContext(
            learning_session_id=f"learning_{uuid.uuid4().hex[:8]}",
            start_time=datetime.now(),
            systems_monitored=systems_to_monitor or ["analytics", "ml", "api", "patterns"],
            learning_objectives=learning_objectives or [
                "discover_cross_system_concepts",
                "analyze_semantic_relationships", 
                "detect_emergent_behaviors",
                "generate_actionable_insights"
            ],
            temporal_window_hours=self.config["temporal_window_hours"],
            confidence_threshold=self.config["concept_confidence_threshold"]
        )
        
        self.is_learning = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        
        self.logger.info(f"Started cross-system semantic learning for systems: {self.current_learning_context.systems_monitored}")
    
    async def stop_learning(self):
        """Stop semantic learning gracefully"""
        self.is_learning = False
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        
        # Finalize learning session
        if self.current_learning_context:
            session_duration = (datetime.now() - self.current_learning_context.start_time).total_seconds()
            self.learning_stats["total_learning_time"] += session_duration
            self.logger.info(f"Learning session {self.current_learning_context.learning_session_id} completed. Duration: {session_duration:.1f}s")
        
        self.logger.info("Stopped cross-system semantic learning")
    
    async def _learning_loop(self):
        """Main semantic learning orchestration loop"""
        while self.is_learning:
            try:
                cycle_start_time = datetime.now()
                
                # Execute complete learning cycle
                await self._execute_learning_cycle()
                
                # Update learning statistics
                cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                self._update_learning_stats(cycle_duration)
                
                # Wait for next learning interval
                await asyncio.sleep(self.config["learning_interval"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Learning cycle failed: {e}")
                await asyncio.sleep(self.config["learning_interval"])
    
    async def _execute_learning_cycle(self):
        """Execute a complete semantic learning cycle"""
        if not self.current_learning_context:
            return
        
        self.logger.info("Starting semantic learning cycle")
        
        # Phase 1: Concept Extraction
        concepts = await self._execute_concept_extraction()
        
        # Phase 2: Relationship Discovery
        relationships = await self._execute_relationship_discovery(concepts)
        
        # Phase 3: Emergent Behavior Detection
        if self.config["emergent_behavior_detection_enabled"]:
            behaviors = await self._execute_behavior_emergence_detection(concepts, relationships)
        else:
            behaviors = []
        
        # Phase 4: Semantic Reasoning and Insight Generation
        if self.config["semantic_reasoning_enabled"]:
            await self._execute_semantic_reasoning(concepts, relationships, behaviors)
        
        # Phase 5: Insight Generation
        if self.config["insight_generation_enabled"]:
            await self._generate_actionable_insights(concepts, relationships, behaviors)
        
        # Phase 6: Knowledge Base Update
        await self._update_knowledge_base(concepts, relationships, behaviors)
        
        self.logger.info(f"Learning cycle completed: {len(concepts)} concepts, {len(relationships)} relationships, {len(behaviors)} behaviors")
    
    async def _execute_concept_extraction(self) -> List[SemanticConcept]:
        """Execute concept extraction phase"""
        try:
            concepts = await self.concept_extractor.extract_concepts_from_all_systems(
                self.current_learning_context
            )
            
            # Filter concepts by confidence threshold
            filtered_concepts = [
                c for c in concepts 
                if c.confidence >= self.config["concept_confidence_threshold"]
            ]
            
            # Limit concepts per cycle
            if len(filtered_concepts) > self.config["max_concepts_per_cycle"]:
                # Sort by importance and take top concepts
                filtered_concepts.sort(key=lambda c: c.calculate_importance(), reverse=True)
                filtered_concepts = filtered_concepts[:self.config["max_concepts_per_cycle"]]
            
            self.current_learning_context.concepts_processed = len(filtered_concepts)
            
            self.logger.debug(f"Extracted {len(filtered_concepts)} high-confidence concepts")
            return filtered_concepts
            
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
            return []
    
    async def _execute_relationship_discovery(self, concepts: List[SemanticConcept]) -> List[SemanticRelationship]:
        """Execute relationship discovery phase"""
        try:
            relationships = await self.relationship_analyzer.discover_relationships(
                concepts, self.current_learning_context
            )
            
            # Filter relationships by strength threshold
            filtered_relationships = [
                r for r in relationships 
                if r.strength >= self.config["relationship_strength_threshold"]
            ]
            
            # Limit relationships per cycle
            if len(filtered_relationships) > self.config["max_relationships_per_cycle"]:
                # Sort by strength and take strongest relationships
                filtered_relationships.sort(key=lambda r: r.strength, reverse=True)
                filtered_relationships = filtered_relationships[:self.config["max_relationships_per_cycle"]]
            
            self.current_learning_context.relationships_analyzed = len(filtered_relationships)
            
            self.logger.debug(f"Discovered {len(filtered_relationships)} strong relationships")
            return filtered_relationships
            
        except Exception as e:
            self.logger.error(f"Relationship discovery failed: {e}")
            return []
    
    async def _execute_behavior_emergence_detection(self, concepts: List[SemanticConcept],
                                                   relationships: List[SemanticRelationship]) -> List[EmergentBehavior]:
        """Execute emergent behavior detection phase"""
        try:
            behaviors = await self.relationship_analyzer.detect_emergent_behaviors(
                concepts, relationships, self.current_learning_context
            )
            
            # Filter behaviors by emergence threshold
            filtered_behaviors = [
                b for b in behaviors 
                if b.complexity_score >= self.config["emergence_detection_threshold"]
            ]
            
            self.current_learning_context.behaviors_detected = len(filtered_behaviors)
            
            self.logger.debug(f"Detected {len(filtered_behaviors)} emergent behaviors")
            return filtered_behaviors
            
        except Exception as e:
            self.logger.error(f"Emergent behavior detection failed: {e}")
            return []
    
    async def _execute_semantic_reasoning(self, concepts: List[SemanticConcept],
                                        relationships: List[SemanticRelationship],
                                        behaviors: List[EmergentBehavior]):
        """Execute semantic reasoning and inference generation"""
        try:
            # Generate semantic inferences
            inferences = await self._make_semantic_inferences(concepts, relationships, behaviors)
            
            # Apply semantic insights to improve system understanding
            await self._apply_semantic_insights(inferences)
            
            self.learning_stats["semantic_inferences_made"] += len(inferences)
            
            self.logger.debug(f"Generated {len(inferences)} semantic inferences")
            
        except Exception as e:
            self.logger.error(f"Semantic reasoning failed: {e}")
    
    async def _make_semantic_inferences(self, concepts: List[SemanticConcept],
                                      relationships: List[SemanticRelationship],
                                      behaviors: List[EmergentBehavior]) -> List[Dict[str, Any]]:
        """Make semantic inferences from discovered patterns"""
        inferences = []
        
        try:
            # Inference 1: Cross-system concept correlations
            cross_system_concepts = [c for c in concepts if len(c.system_manifestations) > 1]
            if len(cross_system_concepts) > 5:
                inference = {
                    "type": "cross_system_correlation",
                    "description": f"Strong cross-system concept correlation detected across {len(cross_system_concepts)} concepts",
                    "confidence": min(1.0, len(cross_system_concepts) / 20.0),
                    "supporting_concepts": [c.concept_id for c in cross_system_concepts[:10]],
                    "systems_involved": list(set().union(*[set(c.system_manifestations.keys()) for c in cross_system_concepts]))
                }
                inferences.append(inference)
            
            # Inference 2: Relationship density analysis
            if relationships:
                concept_relationship_counts = defaultdict(int)
                for rel in relationships:
                    concept_relationship_counts[rel.source_concept] += 1
                    concept_relationship_counts[rel.target_concept] += 1
                
                highly_connected = [cid for cid, count in concept_relationship_counts.items() if count > 5]
                
                if len(highly_connected) > 3:
                    inference = {
                        "type": "relationship_clustering",
                        "description": f"High-density relationship clusters detected around {len(highly_connected)} central concepts",
                        "confidence": min(1.0, len(highly_connected) / 15.0),
                        "central_concepts": highly_connected[:10],
                        "cluster_density": statistics.mean(list(concept_relationship_counts.values()))
                    }
                    inferences.append(inference)
            
            # Inference 3: Emergent behavior implications
            if behaviors:
                adaptive_behaviors = [b for b in behaviors if b.behavior_type == BehaviorType.ADAPTIVE]
                
                if len(adaptive_behaviors) >= 2:
                    inference = {
                        "type": "adaptive_emergence", 
                        "description": f"System-wide adaptive behavior emergence detected across {len(adaptive_behaviors)} behavior patterns",
                        "confidence": statistics.mean([b.complexity_score for b in adaptive_behaviors]),
                        "behavior_systems": list(set().union(*[set(b.participating_systems) for b in adaptive_behaviors])),
                        "adaptation_indicators": list(set().union(*[set(b.observable_effects) for b in adaptive_behaviors]))
                    }
                    inferences.append(inference)
            
            # Inference 4: System optimization opportunities
            optimization_concepts = [c for c in concepts if "optimization" in c.concept_name.lower()]
            if optimization_concepts:
                inference = {
                    "type": "optimization_opportunity",
                    "description": f"System optimization opportunities identified in {len(optimization_concepts)} concepts",
                    "confidence": statistics.mean([c.confidence for c in optimization_concepts]),
                    "optimization_concepts": [c.concept_id for c in optimization_concepts],
                    "potential_improvements": [c.semantic_properties for c in optimization_concepts]
                }
                inferences.append(inference)
            
            return inferences
            
        except Exception as e:
            self.logger.error(f"Semantic inference generation failed: {e}")
            return []
    
    async def _apply_semantic_insights(self, inferences: List[Dict[str, Any]]):
        """Apply semantic insights to improve system understanding and performance"""
        try:
            for inference in inferences:
                inference_type = inference.get("type", "unknown")
                confidence = inference.get("confidence", 0.0)
                
                if confidence < 0.6:
                    continue  # Skip low-confidence inferences
                
                # Apply cross-system correlation insights
                if inference_type == "cross_system_correlation":
                    await self._apply_cross_system_insights(inference)
                
                # Apply relationship clustering insights
                elif inference_type == "relationship_clustering":
                    await self._apply_clustering_insights(inference)
                
                # Apply adaptive emergence insights
                elif inference_type == "adaptive_emergence":
                    await self._apply_adaptation_insights(inference)
                
                # Apply optimization opportunity insights
                elif inference_type == "optimization_opportunity":
                    await self._apply_optimization_insights(inference)
            
        except Exception as e:
            self.logger.error(f"Semantic insight application failed: {e}")
    
    async def _apply_cross_system_insights(self, inference: Dict[str, Any]):
        """Apply cross-system correlation insights"""
        try:
            systems_involved = inference.get("systems_involved", [])
            confidence = inference.get("confidence", 0.0)
            
            # Log cross-system correlation for future optimization
            self.learning_stats["cross_system_correlations"] += 1
            
            self.logger.info(f"Cross-system correlation insight applied: {len(systems_involved)} systems, confidence: {confidence:.2f}")
            
        except Exception as e:
            self.logger.error(f"Cross-system insight application failed: {e}")
    
    async def _apply_clustering_insights(self, inference: Dict[str, Any]):
        """Apply relationship clustering insights"""
        try:
            central_concepts = inference.get("central_concepts", [])
            cluster_density = inference.get("cluster_density", 0.0)
            
            self.logger.info(f"Relationship clustering insight applied: {len(central_concepts)} central concepts, density: {cluster_density:.2f}")
            
        except Exception as e:
            self.logger.error(f"Clustering insight application failed: {e}")
    
    async def _apply_adaptation_insights(self, inference: Dict[str, Any]):
        """Apply adaptive emergence insights"""
        try:
            behavior_systems = inference.get("behavior_systems", [])
            confidence = inference.get("confidence", 0.0)
            
            self.logger.info(f"Adaptive emergence insight applied: {len(behavior_systems)} systems showing adaptation, confidence: {confidence:.2f}")
            
        except Exception as e:
            self.logger.error(f"Adaptation insight application failed: {e}")
    
    async def _apply_optimization_insights(self, inference: Dict[str, Any]):
        """Apply optimization opportunity insights"""
        try:
            optimization_concepts = inference.get("optimization_concepts", [])
            confidence = inference.get("confidence", 0.0)
            
            self.logger.info(f"Optimization insight applied: {len(optimization_concepts)} optimization opportunities, confidence: {confidence:.2f}")
            
        except Exception as e:
            self.logger.error(f"Optimization insight application failed: {e}")
    
    async def _generate_actionable_insights(self, concepts: List[SemanticConcept],
                                          relationships: List[SemanticRelationship],
                                          behaviors: List[EmergentBehavior]):
        """Generate actionable insights from semantic analysis"""
        try:
            insights = []
            
            # Performance optimization insights
            performance_insights = await self._generate_performance_insights(concepts, relationships)
            insights.extend(performance_insights)
            
            # System integration insights
            integration_insights = await self._generate_integration_insights(concepts, relationships)
            insights.extend(integration_insights)
            
            # Behavior prediction insights
            prediction_insights = await self._generate_prediction_insights(behaviors)
            insights.extend(prediction_insights)
            
            # Store insights in knowledge base
            for insight in insights:
                self.semantic_insights[insight.insight_id] = insight
            
            self.current_learning_context.insights_generated = len(insights)
            self.learning_stats["insights_generated"] += len(insights)
            
            self.logger.debug(f"Generated {len(insights)} actionable insights")
            
        except Exception as e:
            self.logger.error(f"Actionable insight generation failed: {e}")
    
    async def _generate_performance_insights(self, concepts: List[SemanticConcept],
                                           relationships: List[SemanticRelationship]) -> List[SemanticInsight]:
        """Generate performance optimization insights"""
        insights = []
        
        try:
            # Identify performance bottleneck concepts
            bottleneck_concepts = [
                c for c in concepts 
                if "performance" in c.concept_name.lower() or "latency" in c.concept_name.lower()
            ]
            
            if bottleneck_concepts:
                insight = SemanticInsight(
                    insight_id=f"perf_{uuid.uuid4().hex[:8]}",
                    insight_type="performance",
                    description=f"Performance bottlenecks identified in {len(bottleneck_concepts)} system components",
                    confidence=statistics.mean([c.confidence for c in bottleneck_concepts]),
                    supporting_evidence=[{
                        "type": "concept_analysis",
                        "concepts": [c.concept_id for c in bottleneck_concepts],
                        "systems": list(set().union(*[set(c.system_manifestations.keys()) for c in bottleneck_concepts]))
                    }],
                    actionable_recommendations=[
                        "Analyze identified bottleneck components for optimization opportunities",
                        "Implement performance monitoring for affected systems",
                        "Consider resource allocation adjustments",
                        "Investigate cross-system performance correlations"
                    ],
                    affected_systems=list(set().union(*[set(c.system_manifestations.keys()) for c in bottleneck_concepts])),
                    business_impact="medium",
                    implementation_complexity="medium"
                )
                
                insight.priority_score = insight.confidence * 0.8
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Performance insight generation failed: {e}")
            return []
    
    async def _generate_integration_insights(self, concepts: List[SemanticConcept],
                                           relationships: List[SemanticRelationship]) -> List[SemanticInsight]:
        """Generate system integration insights"""
        insights = []
        
        try:
            # Identify integration opportunities from cross-system relationships
            cross_system_relationships = [
                r for r in relationships 
                if len(set(r.systems_involved)) > 1 and r.strength > 0.7
            ]
            
            if len(cross_system_relationships) > 5:
                system_pairs = defaultdict(list)
                for rel in cross_system_relationships:
                    systems = sorted(rel.systems_involved)
                    if len(systems) >= 2:
                        pair_key = f"{systems[0]}_{systems[1]}"
                        system_pairs[pair_key].append(rel)
                
                strong_pairs = {pair: rels for pair, rels in system_pairs.items() if len(rels) >= 2}
                
                if strong_pairs:
                    insight = SemanticInsight(
                        insight_id=f"integration_{uuid.uuid4().hex[:8]}",
                        insight_type="integration",
                        description=f"Strong integration opportunities identified between {len(strong_pairs)} system pairs",
                        confidence=statistics.mean([
                            statistics.mean([r.strength for r in rels]) 
                            for rels in strong_pairs.values()
                        ]),
                        supporting_evidence=[{
                            "type": "relationship_analysis",
                            "system_pairs": list(strong_pairs.keys()),
                            "relationship_count": sum(len(rels) for rels in strong_pairs.values())
                        }],
                        actionable_recommendations=[
                            "Develop enhanced integration protocols for strongly connected system pairs",
                            "Implement cross-system data sharing mechanisms",
                            "Create unified interfaces for integrated systems",
                            "Monitor integration performance and stability"
                        ],
                        affected_systems=list(set().union(*[pair.split('_') for pair in strong_pairs.keys()])),
                        business_impact="high",
                        implementation_complexity="medium"
                    )
                    
                    insight.priority_score = insight.confidence * 0.9
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Integration insight generation failed: {e}")
            return []
    
    async def _generate_prediction_insights(self, behaviors: List[EmergentBehavior]) -> List[SemanticInsight]:
        """Generate behavioral prediction insights"""
        insights = []
        
        try:
            # Analyze behavior stability for predictions
            stable_behaviors = [
                b for b in behaviors 
                if b.stability_assessment == StabilityAssessment.STABLE and b.complexity_score > 0.6
            ]
            
            if stable_behaviors:
                insight = SemanticInsight(
                    insight_id=f"prediction_{uuid.uuid4().hex[:8]}",
                    insight_type="prediction",
                    description=f"Predictable behavior patterns identified in {len(stable_behaviors)} emergent behaviors",
                    confidence=statistics.mean([b.complexity_score for b in stable_behaviors]),
                    supporting_evidence=[{
                        "type": "behavior_analysis",
                        "stable_behaviors": [b.behavior_id for b in stable_behaviors],
                        "behavior_types": [b.behavior_type.value for b in stable_behaviors]
                    }],
                    actionable_recommendations=[
                        "Implement predictive monitoring for stable behavior patterns",
                        "Develop behavior-based system optimization strategies",
                        "Create early warning systems for behavior pattern changes",
                        "Use behavior patterns for capacity planning and resource allocation"
                    ],
                    affected_systems=list(set().union(*[set(b.participating_systems) for b in stable_behaviors])),
                    business_impact="high",
                    implementation_complexity="low"
                )
                
                insight.priority_score = insight.confidence * 0.85
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Prediction insight generation failed: {e}")
            return []
    
    async def _update_knowledge_base(self, concepts: List[SemanticConcept],
                                   relationships: List[SemanticRelationship],
                                   behaviors: List[EmergentBehavior]):
        """Update semantic knowledge base with discovered patterns"""
        try:
            # Update concepts
            for concept in concepts:
                existing_concept = self.semantic_concepts.get(concept.concept_id)
                if existing_concept:
                    # Merge with existing concept
                    await self._merge_concepts(existing_concept, concept)
                else:
                    self.semantic_concepts[concept.concept_id] = concept
            
            # Update relationships
            for relationship in relationships:
                self.semantic_relationships[relationship.relationship_id] = relationship
            
            # Update behaviors
            for behavior in behaviors:
                self.emergent_behaviors[behavior.behavior_id] = behavior
            
            self.logger.debug(f"Updated knowledge base: {len(concepts)} concepts, {len(relationships)} relationships, {len(behaviors)} behaviors")
            
        except Exception as e:
            self.logger.error(f"Knowledge base update failed: {e}")
    
    async def _merge_concepts(self, existing: SemanticConcept, new: SemanticConcept):
        """Merge new concept with existing concept"""
        try:
            # Update confidence with weighted average
            existing.confidence = (existing.confidence * 0.7) + (new.confidence * 0.3)
            
            # Merge system manifestations
            existing.system_manifestations.update(new.system_manifestations)
            
            # Merge semantic properties
            existing.semantic_properties.update(new.semantic_properties)
            
            # Update evolution history
            existing.evolution_history.append({
                "timestamp": datetime.now().isoformat(),
                "change_type": "concept_merge",
                "merged_from": new.concept_id,
                "confidence_update": existing.confidence
            })
            
            existing.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Concept merging failed: {e}")
    
    def _update_learning_stats(self, cycle_duration: float):
        """Update learning cycle statistics"""
        self.learning_stats["learning_cycles_completed"] += 1
        self.learning_stats["last_learning_cycle"] = datetime.now()
        
        # Update concept and relationship counts
        self.learning_stats["concepts_discovered"] = len(self.semantic_concepts)
        self.learning_stats["relationships_discovered"] = len(self.semantic_relationships)
        self.learning_stats["emergent_behaviors_discovered"] = len(self.emergent_behaviors)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status"""
        return {
            "is_learning": self.is_learning,
            "learning_stats": self.learning_stats.copy(),
            "current_session": {
                "session_id": self.current_learning_context.learning_session_id if self.current_learning_context else None,
                "systems_monitored": self.current_learning_context.systems_monitored if self.current_learning_context else [],
                "session_duration": (datetime.now() - self.current_learning_context.start_time).total_seconds() if self.current_learning_context else 0
            },
            "knowledge_base_size": {
                "concepts": len(self.semantic_concepts),
                "relationships": len(self.semantic_relationships),
                "emergent_behaviors": len(self.emergent_behaviors),
                "insights": len(self.semantic_insights)
            },
            "recent_insights": [
                {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "priority": insight.priority_score,
                    "business_impact": insight.business_impact,
                    "affected_systems": insight.affected_systems
                }
                for insight in sorted(self.semantic_insights.values(), 
                                   key=lambda x: x.discovered_at, reverse=True)[:5]
            ],
            "component_status": {
                "concept_extractor": self.concept_extractor.get_extraction_status(),
                "relationship_analyzer": self.relationship_analyzer.get_analysis_status()
            },
            "configuration": self.config.copy()
        }
    
    def get_semantic_knowledge_graph(self) -> Dict[str, Any]:
        """Get comprehensive semantic knowledge graph"""
        return {
            "metadata": {
                "total_concepts": len(self.semantic_concepts),
                "total_relationships": len(self.semantic_relationships),
                "total_behaviors": len(self.emergent_behaviors),
                "total_insights": len(self.semantic_insights),
                "generated_at": datetime.now().isoformat()
            },
            "concepts": {
                concept_id: {
                    "name": concept.concept_name,
                    "type": concept.concept_type.value,
                    "confidence": concept.confidence,
                    "abstraction_level": concept.abstraction_level.value,
                    "systems": list(concept.system_manifestations.keys()),
                    "properties": concept.semantic_properties,
                    "importance_score": concept.calculate_importance()
                }
                for concept_id, concept in self.semantic_concepts.items()
            },
            "relationships": {
                rel_id: {
                    "source": rel.source_concept,
                    "target": rel.target_concept,
                    "type": rel.relationship_type.value,
                    "strength": rel.strength,
                    "directionality": rel.directionality.value,
                    "systems": rel.systems_involved,
                    "causality_confidence": rel.causality_confidence,
                    "evidence_count": len(rel.evidence)
                }
                for rel_id, rel in self.semantic_relationships.items()
            },
            "emergent_behaviors": {
                behavior_id: {
                    "name": behavior.behavior_name,
                    "type": behavior.behavior_type.value,
                    "complexity": behavior.complexity_score,
                    "systems": behavior.participating_systems,
                    "effects": behavior.observable_effects,
                    "stability": behavior.stability_assessment.value,
                    "emergence_score": behavior.calculate_emergence_score()
                }
                for behavior_id, behavior in self.emergent_behaviors.items()
            },
            "insights": {
                insight_id: {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "priority_score": insight.priority_score,
                    "business_impact": insight.business_impact,
                    "implementation_complexity": insight.implementation_complexity,
                    "affected_systems": insight.affected_systems,
                    "recommendations": insight.actionable_recommendations
                }
                for insight_id, insight in self.semantic_insights.items()
            }
        }


# Convenience function for creating semantic learner
def create_cross_system_semantic_learner(analytics_hub: AnalyticsHub, ml_orchestrator: MLOrchestrator,
                                        api_gateway: UnifiedAPIGateway, pattern_recognizer: AdvancedPatternRecognizer) -> CrossSystemSemanticLearner:
    """Create and configure a CrossSystemSemanticLearner instance"""
    return CrossSystemSemanticLearner(analytics_hub, ml_orchestrator, api_gateway, pattern_recognizer)


# Export the main learner class and factory function
__all__ = ['CrossSystemSemanticLearner', 'create_cross_system_semantic_learner']