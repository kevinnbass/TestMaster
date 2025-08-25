"""
Cross-System Semantic Learning Engine
====================================

Advanced semantic learning system that operates across all intelligence frameworks
(analytics, ML, API) to discover unified patterns, semantic relationships, and
emergent intelligence behaviors.

Author: Agent A Phase 2 - Advanced Pattern Recognition & Semantic Analysis
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import statistics
from pathlib import Path

from .analytics.analytics_hub import AnalyticsHub
from .ml.ml_orchestrator import MLOrchestrator
from .api.unified_api_gateway import UnifiedAPIGateway
from .analysis.advanced_pattern_recognizer import AdvancedPatternRecognizer, AdvancedPattern


@dataclass
class SemanticConcept:
    """Unified semantic concept across systems"""
    concept_id: str
    concept_name: str
    concept_type: str  # "entity", "relationship", "behavior", "pattern"
    confidence: float
    abstraction_level: int  # 1=concrete, 5=highly abstract
    system_manifestations: Dict[str, Any]  # How concept appears in each system
    semantic_properties: Dict[str, Any]
    related_concepts: List[str]
    evolution_history: List[Dict[str, Any]]
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class SemanticRelationship:
    """Relationship between semantic concepts"""
    relationship_id: str
    source_concept: str
    target_concept: str
    relationship_type: str  # "causes", "correlates", "contains", "transforms"
    strength: float  # 0-1
    directionality: str  # "bidirectional", "source_to_target", "target_to_source"
    evidence: List[Dict[str, Any]]
    systems_involved: List[str]
    temporal_aspects: Optional[Dict[str, Any]] = None


@dataclass
class EmergentBehavior:
    """Emergent behaviors discovered across systems"""
    behavior_id: str
    behavior_name: str
    behavior_type: str  # "adaptive", "self_organizing", "optimization", "coordination"
    complexity_score: float
    emergence_conditions: List[str]
    participating_systems: List[str]
    observable_effects: List[str]
    predictive_indicators: List[str]
    stability_assessment: str  # "stable", "unstable", "evolving"


class CrossSystemSemanticLearner:
    """
    Advanced semantic learning engine that discovers unified patterns and concepts
    across all intelligence systems, enabling higher-order understanding and
    emergent intelligence behaviors.
    """
    
    def __init__(self, analytics_hub: AnalyticsHub, ml_orchestrator: MLOrchestrator, 
                 api_gateway: UnifiedAPIGateway, pattern_recognizer: AdvancedPatternRecognizer):
        self.analytics_hub = analytics_hub
        self.ml_orchestrator = ml_orchestrator
        self.api_gateway = api_gateway
        self.pattern_recognizer = pattern_recognizer
        self.logger = logging.getLogger(__name__)
        
        # Semantic knowledge base
        self.semantic_concepts: Dict[str, SemanticConcept] = {}
        self.semantic_relationships: Dict[str, SemanticRelationship] = {}
        self.emergent_behaviors: Dict[str, EmergentBehavior] = {}
        
        # Learning data structures
        self.concept_co_occurrence_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cross_system_event_correlation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.behavioral_pattern_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Semantic processing pipeline
        self.concept_extraction_pipeline = []
        self.relationship_discovery_pipeline = []
        self.behavior_emergence_pipeline = []
        
        # Configuration
        self.config = {
            "learning_interval": 600,  # 10 minutes
            "concept_confidence_threshold": 0.6,
            "relationship_strength_threshold": 0.5,
            "emergence_detection_threshold": 0.7,
            "abstraction_levels": 5,
            "temporal_window_hours": 24,
            "cross_system_analysis_enabled": True,
            "emergent_behavior_detection_enabled": True,
            "semantic_reasoning_enabled": True
        }
        
        # Learning statistics
        self.learning_stats = {
            "concepts_discovered": 0,
            "relationships_discovered": 0,
            "emergent_behaviors_discovered": 0,
            "cross_system_correlations": 0,
            "semantic_inferences_made": 0,
            "learning_cycles_completed": 0,
            "start_time": datetime.now()
        }
        
        # Learning state
        self.is_learning = False
        self.learning_task = None
        
        # Initialize semantic processing pipelines
        self._initialize_processing_pipelines()
        
        self.logger.info("Cross-System Semantic Learner initialized")
    
    def _initialize_processing_pipelines(self):
        """Initialize semantic processing pipelines"""
        # Concept extraction pipeline
        self.concept_extraction_pipeline = [
            self._extract_analytics_concepts,
            self._extract_ml_concepts,
            self._extract_api_concepts,
            self._extract_pattern_concepts,
            self._unify_concepts,
            self._abstract_concepts
        ]
        
        # Relationship discovery pipeline
        self.relationship_discovery_pipeline = [
            self._discover_causal_relationships,
            self._discover_correlation_relationships,
            self._discover_containment_relationships,
            self._discover_transformation_relationships,
            self._validate_relationships,
            self._strengthen_relationships
        ]
        
        # Behavior emergence pipeline
        self.behavior_emergence_pipeline = [
            self._detect_adaptive_behaviors,
            self._detect_self_organizing_behaviors,
            self._detect_optimization_behaviors,
            self._detect_coordination_behaviors,
            self._validate_emergent_behaviors,
            self._predict_behavior_evolution
        ]
    
    async def start_learning(self):
        """Start cross-system semantic learning"""
        if self.is_learning:
            self.logger.warning("Semantic learning already running")
            return
        
        self.is_learning = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        self.logger.info("Started cross-system semantic learning")
    
    async def stop_learning(self):
        """Stop semantic learning"""
        self.is_learning = False
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped cross-system semantic learning")
    
    async def _learning_loop(self):
        """Main semantic learning loop"""
        while self.is_learning:
            try:
                await asyncio.sleep(self.config["learning_interval"])
                
                # Concept extraction phase
                await self._execute_concept_extraction()
                
                # Relationship discovery phase
                await self._execute_relationship_discovery()
                
                # Emergent behavior detection phase
                if self.config["emergent_behavior_detection_enabled"]:
                    await self._execute_behavior_emergence_detection()
                
                # Semantic reasoning phase
                if self.config["semantic_reasoning_enabled"]:
                    await self._execute_semantic_reasoning()
                
                # Update learning statistics
                self.learning_stats["learning_cycles_completed"] += 1
                
            except Exception as e:
                self.logger.error(f"Semantic learning loop error: {e}")
                await asyncio.sleep(300)
    
    async def _execute_concept_extraction(self):
        """Execute concept extraction pipeline"""
        try:
            # Collect data from all systems
            system_data = await self._collect_cross_system_data()
            
            # Execute concept extraction pipeline
            for extraction_func in self.concept_extraction_pipeline:
                concepts = await extraction_func(system_data)
                
                # Store discovered concepts
                for concept in concepts:
                    if concept.confidence >= self.config["concept_confidence_threshold"]:
                        await self._store_semantic_concept(concept)
                        
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
    
    async def _collect_cross_system_data(self) -> Dict[str, Any]:
        """Collect data from all intelligence systems"""
        system_data = {
            "analytics": {},
            "ml": {},
            "api": {},
            "patterns": {},
            "timestamp": datetime.now()
        }
        
        try:
            # Analytics data
            if self.analytics_hub:
                system_data["analytics"] = {
                    "status": self.analytics_hub.get_hub_status(),
                    "insights": self.analytics_hub.get_recent_insights(limit=100),
                    "correlations": self.analytics_hub.get_correlation_matrix(),
                    "events": await self._get_analytics_events()
                }
            
            # ML orchestration data
            if self.ml_orchestrator:
                system_data["ml"] = {
                    "status": self.ml_orchestrator.get_orchestration_status(),
                    "insights": self.ml_orchestrator.get_integration_insights(),
                    "flows": await self._get_ml_flow_data(),
                    "modules": await self._get_ml_module_data()
                }
            
            # API gateway data
            if self.api_gateway:
                system_data["api"] = {
                    "statistics": self.api_gateway.get_gateway_statistics(),
                    "endpoints": self.api_gateway.get_endpoint_documentation(),
                    "patterns": await self._get_api_patterns()
                }
            
            # Pattern recognition data
            if self.pattern_recognizer:
                system_data["patterns"] = {
                    "status": self.pattern_recognizer.get_recognition_status(),
                    "correlations": self.pattern_recognizer.get_pattern_correlations(),
                    "advanced_patterns": await self._get_advanced_patterns()
                }
            
            return system_data
            
        except Exception as e:
            self.logger.error(f"Cross-system data collection failed: {e}")
            return system_data
    
    async def _get_analytics_events(self) -> List[Dict[str, Any]]:
        """Get recent analytics events"""
        try:
            # Extract events from analytics hub
            events = []
            
            # Get prediction events
            if hasattr(self.analytics_hub, 'predictive_engine'):
                predictions = self.analytics_hub.predictive_engine.get_active_predictions()
                for metric, prediction in predictions.items():
                    events.append({
                        "type": "prediction",
                        "metric": metric,
                        "confidence": prediction.confidence,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Get anomaly events
            if hasattr(self.analytics_hub, 'anomaly_detector'):
                anomalies = self.analytics_hub.anomaly_detector.get_recent_anomalies(hours=2)
                for anomaly in anomalies:
                    events.append({
                        "type": "anomaly",
                        "metric": anomaly.metric_name,
                        "severity": anomaly.severity.value,
                        "timestamp": anomaly.detected_at.isoformat()
                    })
            
            return events
            
        except Exception as e:
            self.logger.debug(f"Failed to get analytics events: {e}")
            return []
    
    async def _get_ml_flow_data(self) -> List[Dict[str, Any]]:
        """Get ML orchestration flow data"""
        try:
            flows = []
            
            # Get integration flows from orchestrator
            status = self.ml_orchestrator.get_orchestration_status()
            integration_flows = status.get("integration_flows", {})
            
            for flow_id, flow_info in integration_flows.items():
                flows.append({
                    "flow_id": flow_id,
                    "pattern": flow_info.get("pattern", "unknown"),
                    "enabled": flow_info.get("enabled", False),
                    "source_modules": flow_info.get("source_modules", []),
                    "target_modules": flow_info.get("target_modules", []),
                    "performance": {
                        "message_count": flow_info.get("message_count", 0),
                        "average_latency": flow_info.get("average_latency", 0.0),
                        "error_count": flow_info.get("error_count", 0)
                    }
                })
            
            return flows
            
        except Exception as e:
            self.logger.debug(f"Failed to get ML flow data: {e}")
            return []
    
    async def _get_ml_module_data(self) -> List[Dict[str, Any]]:
        """Get ML module data"""
        try:
            modules = []
            
            status = self.ml_orchestrator.get_orchestration_status()
            module_status = status.get("module_status", {})
            
            for module_name, module_info in module_status.items():
                modules.append({
                    "module_name": module_name,
                    "status": module_info.get("status", "unknown"),
                    "success_rate": module_info.get("success_rate", 0.0),
                    "processing_time": module_info.get("processing_time", 0.0),
                    "resource_usage": module_info.get("resource_usage", {})
                })
            
            return modules
            
        except Exception as e:
            self.logger.debug(f"Failed to get ML module data: {e}")
            return []
    
    async def _get_api_patterns(self) -> List[Dict[str, Any]]:
        """Get API usage patterns"""
        try:
            patterns = []
            
            stats = self.api_gateway.get_gateway_statistics()
            metrics = stats.get("gateway_metrics", {})
            
            # Extract usage patterns
            patterns.append({
                "pattern_type": "usage",
                "total_requests": metrics.get("total_requests", 0),
                "success_rate": metrics.get("successful_responses", 0) / max(metrics.get("total_requests", 1), 1),
                "average_response_time": metrics.get("average_response_time_ms", 0.0),
                "error_rate": metrics.get("error_responses", 0) / max(metrics.get("total_requests", 1), 1)
            })
            
            return patterns
            
        except Exception as e:
            self.logger.debug(f"Failed to get API patterns: {e}")
            return []
    
    async def _get_advanced_patterns(self) -> List[Dict[str, Any]]:
        """Get advanced patterns from pattern recognizer"""
        try:
            status = self.pattern_recognizer.get_recognition_status()
            return status.get("recent_patterns", [])
            
        except Exception as e:
            self.logger.debug(f"Failed to get advanced patterns: {e}")
            return []
    
    async def _extract_analytics_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from analytics system"""
        concepts = []
        
        try:
            analytics_data = system_data.get("analytics", {})
            
            # Extract concepts from insights
            insights = analytics_data.get("insights", [])
            for insight in insights:
                if hasattr(insight, 'category') and hasattr(insight, 'confidence'):
                    concepts.append(SemanticConcept(
                        concept_id=f"analytics_concept_{insight.category}_{int(datetime.now().timestamp())}",
                        concept_name=f"Analytics {insight.category}",
                        concept_type="pattern",
                        confidence=insight.confidence,
                        abstraction_level=2,
                        system_manifestations={"analytics": {"category": insight.category, "priority": getattr(insight, 'priority', 5)}},
                        semantic_properties={"domain": "analytics", "temporal": True},
                        related_concepts=[]
                    ))
            
            # Extract concepts from events
            events = analytics_data.get("events", [])
            event_types = set(event.get("type") for event in events)
            for event_type in event_types:
                type_events = [e for e in events if e.get("type") == event_type]
                avg_confidence = statistics.mean([e.get("confidence", 0.5) for e in type_events if "confidence" in e]) if type_events else 0.5
                
                concepts.append(SemanticConcept(
                    concept_id=f"analytics_event_{event_type}_{int(datetime.now().timestamp())}",
                    concept_name=f"Analytics Event: {event_type}",
                    concept_type="behavior",
                    confidence=avg_confidence,
                    abstraction_level=3,
                    system_manifestations={"analytics": {"event_type": event_type, "frequency": len(type_events)}},
                    semantic_properties={"domain": "analytics", "behavioral": True},
                    related_concepts=[]
                ))
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Analytics concept extraction failed: {e}")
            return []
    
    async def _extract_ml_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from ML system"""
        concepts = []
        
        try:
            ml_data = system_data.get("ml", {})
            
            # Extract concepts from flows
            flows = ml_data.get("flows", [])
            for flow in flows:
                concepts.append(SemanticConcept(
                    concept_id=f"ml_flow_{flow['flow_id']}_{int(datetime.now().timestamp())}",
                    concept_name=f"ML Flow: {flow['pattern']}",
                    concept_type="relationship",
                    confidence=0.8 if flow.get("enabled", False) else 0.4,
                    abstraction_level=2,
                    system_manifestations={
                        "ml": {
                            "pattern": flow.get("pattern"),
                            "source_modules": flow.get("source_modules", []),
                            "target_modules": flow.get("target_modules", [])
                        }
                    },
                    semantic_properties={"domain": "ml_orchestration", "structural": True},
                    related_concepts=[]
                ))
            
            # Extract concepts from modules
            modules = ml_data.get("modules", [])
            module_statuses = set(module.get("status") for module in modules)
            for status in module_statuses:
                status_modules = [m for m in modules if m.get("status") == status]
                avg_success = statistics.mean([m.get("success_rate", 0.0) for m in status_modules])
                
                concepts.append(SemanticConcept(
                    concept_id=f"ml_module_status_{status}_{int(datetime.now().timestamp())}",
                    concept_name=f"ML Module Status: {status}",
                    concept_type="entity",
                    confidence=avg_success,
                    abstraction_level=1,
                    system_manifestations={"ml": {"status": status, "module_count": len(status_modules)}},
                    semantic_properties={"domain": "ml_orchestration", "operational": True},
                    related_concepts=[]
                ))
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"ML concept extraction failed: {e}")
            return []
    
    async def _extract_api_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from API system"""
        concepts = []
        
        try:
            api_data = system_data.get("api", {})
            
            # Extract concepts from usage patterns
            patterns = api_data.get("patterns", [])
            for pattern in patterns:
                if pattern.get("pattern_type") == "usage":
                    success_rate = pattern.get("success_rate", 0.0)
                    response_time = pattern.get("average_response_time", 0.0)
                    
                    # Create performance concept
                    performance_level = "high" if success_rate > 0.9 and response_time < 100 else "medium" if success_rate > 0.7 else "low"
                    
                    concepts.append(SemanticConcept(
                        concept_id=f"api_performance_{performance_level}_{int(datetime.now().timestamp())}",
                        concept_name=f"API Performance: {performance_level}",
                        concept_type="entity",
                        confidence=success_rate,
                        abstraction_level=2,
                        system_manifestations={
                            "api": {
                                "performance_level": performance_level,
                                "success_rate": success_rate,
                                "response_time": response_time
                            }
                        },
                        semantic_properties={"domain": "api_gateway", "performance": True},
                        related_concepts=[]
                    ))
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"API concept extraction failed: {e}")
            return []
    
    async def _extract_pattern_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from pattern recognition system"""
        concepts = []
        
        try:
            pattern_data = system_data.get("patterns", {})
            
            # Extract concepts from advanced patterns
            advanced_patterns = pattern_data.get("advanced_patterns", [])
            for pattern in advanced_patterns:
                concepts.append(SemanticConcept(
                    concept_id=f"pattern_concept_{pattern.get('type')}_{int(datetime.now().timestamp())}",
                    concept_name=f"Pattern: {pattern.get('name', 'Unknown')}",
                    concept_type="pattern",
                    confidence=pattern.get("confidence", 0.5),
                    abstraction_level=4,  # Patterns are more abstract
                    system_manifestations={
                        "patterns": {
                            "pattern_type": pattern.get("type"),
                            "locations": pattern.get("locations", []),
                            "evolution_stage": pattern.get("evolution_stage", "unknown")
                        }
                    },
                    semantic_properties={"domain": "pattern_recognition", "meta": True},
                    related_concepts=[]
                ))
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Pattern concept extraction failed: {e}")
            return []
    
    async def _unify_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Unify similar concepts across systems"""
        concepts = []
        
        try:
            # Find concepts that represent similar semantic meanings across systems
            existing_concepts = list(self.semantic_concepts.values())
            
            # Group by semantic similarity
            concept_groups = self._group_concepts_by_similarity(existing_concepts)
            
            for group in concept_groups:
                if len(group) > 1:  # Multiple similar concepts
                    unified_concept = await self._create_unified_concept(group)
                    if unified_concept:
                        concepts.append(unified_concept)
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Concept unification failed: {e}")
            return []
    
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
                    if similarity > 0.7:  # High similarity threshold
                        similar_group.append(other_concept)
                        used_concepts.add(other_concept.concept_id)
            
            if len(similar_group) > 1:
                groups.append(similar_group)
        
        return groups
    
    def _calculate_concept_similarity(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Calculate similarity between two concepts"""
        try:
            similarity_factors = []
            
            # Type similarity
            if concept_a.concept_type == concept_b.concept_type:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.3)
            
            # Name similarity (simple string similarity)
            name_similarity = len(set(concept_a.concept_name.lower().split()) & set(concept_b.concept_name.lower().split())) / max(len(concept_a.concept_name.split()), len(concept_b.concept_name.split()), 1)
            similarity_factors.append(name_similarity)
            
            # Abstraction level similarity
            abs_diff = abs(concept_a.abstraction_level - concept_b.abstraction_level)
            abs_similarity = max(0, 1.0 - abs_diff / 5.0)
            similarity_factors.append(abs_similarity)
            
            # Domain similarity (from semantic properties)
            domain_a = concept_a.semantic_properties.get("domain", "")
            domain_b = concept_b.semantic_properties.get("domain", "")
            domain_similarity = 1.0 if domain_a == domain_b else 0.5 if domain_a and domain_b else 0.0
            similarity_factors.append(domain_similarity)
            
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
            most_common_type = max(set([c.concept_type for c in concept_group]), key=[c.concept_type for c in concept_group].count)
            avg_abstraction = int(statistics.mean([c.abstraction_level for c in concept_group]))
            
            # Merge system manifestations
            unified_manifestations = {}
            for concept in concept_group:
                unified_manifestations.update(concept.system_manifestations)
            
            # Merge semantic properties
            unified_properties = {}
            for concept in concept_group:
                unified_properties.update(concept.semantic_properties)
            unified_properties["unified"] = True
            unified_properties["source_concepts"] = [c.concept_id for c in concept_group]
            
            return SemanticConcept(
                concept_id=f"unified_concept_{int(datetime.now().timestamp())}",
                concept_name=f"Unified: {concept_group[0].concept_name.split(':')[0]}",
                concept_type=most_common_type,
                confidence=avg_confidence,
                abstraction_level=avg_abstraction,
                system_manifestations=unified_manifestations,
                semantic_properties=unified_properties,
                related_concepts=[]
            )
            
        except Exception as e:
            self.logger.error(f"Unified concept creation failed: {e}")
            return None
    
    async def _abstract_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Create higher-level abstract concepts"""
        concepts = []
        
        try:
            # Create abstract concepts based on patterns across all data
            
            # System Integration Concept
            systems_active = sum(1 for system in ["analytics", "ml", "api", "patterns"] if system_data.get(system))
            if systems_active >= 3:
                concepts.append(SemanticConcept(
                    concept_id=f"system_integration_{int(datetime.now().timestamp())}",
                    concept_name="System Integration",
                    concept_type="entity",
                    confidence=min(1.0, systems_active / 4.0),
                    abstraction_level=5,
                    system_manifestations={
                        "all": {"active_systems": systems_active, "integration_level": "high"}
                    },
                    semantic_properties={"domain": "meta_system", "integration": True, "emergent": True},
                    related_concepts=[]
                ))
            
            # Intelligence Emergence Concept
            if (len(system_data.get("analytics", {}).get("insights", [])) > 5 and 
                len(system_data.get("patterns", {}).get("advanced_patterns", [])) > 3):
                concepts.append(SemanticConcept(
                    concept_id=f"intelligence_emergence_{int(datetime.now().timestamp())}",
                    concept_name="Intelligence Emergence",
                    concept_type="behavior",
                    confidence=0.8,
                    abstraction_level=5,
                    system_manifestations={
                        "all": {"insights_generated": True, "patterns_discovered": True}
                    },
                    semantic_properties={"domain": "meta_intelligence", "emergent": True, "adaptive": True},
                    related_concepts=[]
                ))
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Concept abstraction failed: {e}")
            return []
    
    async def _store_semantic_concept(self, concept: SemanticConcept):
        """Store a semantic concept in the knowledge base"""
        try:
            # Check for existing similar concept
            existing = self._find_existing_concept(concept)
            
            if existing:
                # Merge with existing concept
                await self._merge_concepts(existing, concept)
            else:
                # Store new concept
                self.semantic_concepts[concept.concept_id] = concept
                self.learning_stats["concepts_discovered"] += 1
                
                self.logger.debug(f"Stored semantic concept: {concept.concept_name} (confidence: {concept.confidence:.2f})")
                
        except Exception as e:
            self.logger.error(f"Failed to store semantic concept: {e}")
    
    def _find_existing_concept(self, new_concept: SemanticConcept) -> Optional[SemanticConcept]:
        """Find existing similar concept"""
        for existing_concept in self.semantic_concepts.values():
            similarity = self._calculate_concept_similarity(existing_concept, new_concept)
            if similarity > 0.8:  # Very high similarity
                return existing_concept
        return None
    
    async def _merge_concepts(self, existing: SemanticConcept, new: SemanticConcept):
        """Merge new concept information with existing concept"""
        try:
            # Update confidence with weighted average
            weight = 0.3
            existing.confidence = (existing.confidence * (1 - weight) + new.confidence * weight)
            
            # Merge system manifestations
            existing.system_manifestations.update(new.system_manifestations)
            
            # Merge semantic properties
            existing.semantic_properties.update(new.semantic_properties)
            
            # Add to evolution history
            existing.evolution_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "merged",
                "source_concept": new.concept_id,
                "confidence_update": new.confidence
            })
            
        except Exception as e:
            self.logger.error(f"Concept merge failed: {e}")
    
    async def _execute_relationship_discovery(self):
        """Execute relationship discovery pipeline"""
        try:
            # Execute relationship discovery functions
            for discovery_func in self.relationship_discovery_pipeline:
                relationships = await discovery_func()
                
                # Store discovered relationships
                for relationship in relationships:
                    if relationship.strength >= self.config["relationship_strength_threshold"]:
                        await self._store_semantic_relationship(relationship)
                        
        except Exception as e:
            self.logger.error(f"Relationship discovery failed: {e}")
    
    async def _discover_causal_relationships(self) -> List[SemanticRelationship]:
        """Discover causal relationships between concepts"""
        relationships = []
        
        try:
            # Look for temporal sequences that suggest causation
            concepts = list(self.semantic_concepts.values())
            
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
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Causal relationship discovery failed: {e}")
            return []
    
    async def _analyze_causal_evidence(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Analyze evidence for causal relationship between concepts"""
        try:
            # Simple heuristic: if concept A appears before concept B consistently,
            # and they share system manifestations, there might be causation
            
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
    
    async def _discover_correlation_relationships(self) -> List[SemanticRelationship]:
        """Discover correlation relationships between concepts"""
        relationships = []
        
        try:
            # Use co-occurrence patterns to find correlations
            concepts = list(self.semantic_concepts.values())
            
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
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Correlation relationship discovery failed: {e}")
            return []
    
    def _calculate_correlation_strength(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Calculate correlation strength between two concepts"""
        try:
            # Factors that suggest correlation
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
    
    async def _discover_containment_relationships(self) -> List[SemanticRelationship]:
        """Discover containment relationships (concept A contains concept B)"""
        relationships = []
        
        try:
            concepts = list(self.semantic_concepts.values())
            
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
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Containment relationship discovery failed: {e}")
            return []
    
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
    
    async def _discover_transformation_relationships(self) -> List[SemanticRelationship]:
        """Discover transformation relationships (concept A transforms into concept B)"""
        relationships = []
        
        try:
            # Look for concepts that represent different states of the same entity
            concepts = list(self.semantic_concepts.values())
            
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
                            relationship_id=f"transformation_{source.concept_id}_{target.concept_id}",
                            source_concept=source.concept_id,
                            target_concept=target.concept_id,
                            relationship_type="transforms",
                            strength=transformation_strength,
                            directionality="source_to_target",
                            evidence=[{"type": "state_evolution", "strength": transformation_strength}],
                            systems_involved=list(set(
                                list(source.system_manifestations.keys()) + 
                                list(target.system_manifestations.keys())
                            ))
                        ))
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Transformation relationship discovery failed: {e}")
            return []
    
    async def _analyze_transformation(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Analyze transformation relationship between concepts"""
        try:
            # Look for concepts that share core identity but differ in state
            similarity = self._calculate_concept_similarity(concept_a, concept_b)
            
            if 0.3 < similarity < 0.8:  # Moderate similarity suggests transformation
                # Check for evolution indicators
                evolution_indicators = []
                
                # Different abstraction levels
                if concept_a.abstraction_level != concept_b.abstraction_level:
                    evolution_indicators.append(0.3)
                
                # Different confidence levels
                if abs(concept_a.confidence - concept_b.confidence) > 0.2:
                    evolution_indicators.append(0.2)
                
                # Temporal sequence
                if abs((concept_a.discovered_at - concept_b.discovered_at).total_seconds()) < 7200:  # 2 hours
                    evolution_indicators.append(0.3)
                
                # Same domain but different properties
                domain_a = concept_a.semantic_properties.get("domain", "")
                domain_b = concept_b.semantic_properties.get("domain", "")
                if domain_a == domain_b and domain_a:
                    evolution_indicators.append(0.2)
                
                return min(1.0, sum(evolution_indicators))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _validate_relationships(self) -> List[SemanticRelationship]:
        """Validate discovered relationships"""
        # This is a placeholder for relationship validation logic
        # In a full implementation, this would check relationship consistency,
        # remove conflicting relationships, and strengthen validated ones
        return []
    
    async def _strengthen_relationships(self) -> List[SemanticRelationship]:
        """Strengthen relationships based on continued evidence"""
        # This is a placeholder for relationship strengthening logic
        # In a full implementation, this would increase relationship strength
        # based on continued evidence and usage patterns
        return []
    
    async def _store_semantic_relationship(self, relationship: SemanticRelationship):
        """Store a semantic relationship"""
        try:
            self.semantic_relationships[relationship.relationship_id] = relationship
            self.learning_stats["relationships_discovered"] += 1
            
            self.logger.debug(f"Stored semantic relationship: {relationship.relationship_type} between {relationship.source_concept} and {relationship.target_concept}")
            
        except Exception as e:
            self.logger.error(f"Failed to store semantic relationship: {e}")
    
    async def _execute_behavior_emergence_detection(self):
        """Execute emergent behavior detection pipeline"""
        try:
            # Execute behavior detection functions
            for detection_func in self.behavior_emergence_pipeline:
                behaviors = await detection_func()
                
                # Store discovered behaviors
                for behavior in behaviors:
                    if self._validate_emergent_behavior(behavior):
                        await self._store_emergent_behavior(behavior)
                        
        except Exception as e:
            self.logger.error(f"Emergent behavior detection failed: {e}")
    
    async def _detect_adaptive_behaviors(self) -> List[EmergentBehavior]:
        """Detect adaptive behaviors across systems"""
        behaviors = []
        
        try:
            # Look for patterns of system adaptation
            # Check if systems are adjusting their behavior based on feedback
            
            # Analytics adaptation
            if (self.analytics_hub and 
                hasattr(self.analytics_hub, 'adaptive_prediction_enhancer')):
                behaviors.append(EmergentBehavior(
                    behavior_id=f"analytics_adaptation_{int(datetime.now().timestamp())}",
                    behavior_name="Analytics System Adaptation",
                    behavior_type="adaptive",
                    complexity_score=0.7,
                    emergence_conditions=["prediction_accuracy_tracking", "parameter_tuning"],
                    participating_systems=["analytics"],
                    observable_effects=["improved_prediction_accuracy", "optimized_parameters"],
                    predictive_indicators=["accuracy_trends", "parameter_changes"],
                    stability_assessment="evolving"
                ))
            
            # ML orchestrator adaptation
            if (self.ml_orchestrator and 
                hasattr(self.ml_orchestrator, 'self_optimizing_orchestrator')):
                behaviors.append(EmergentBehavior(
                    behavior_id=f"ml_adaptation_{int(datetime.now().timestamp())}",
                    behavior_name="ML Orchestration Adaptation",
                    behavior_type="adaptive",
                    complexity_score=0.8,
                    emergence_conditions=["performance_monitoring", "flow_optimization"],
                    participating_systems=["ml"],
                    observable_effects=["optimized_flows", "resource_efficiency"],
                    predictive_indicators=["performance_metrics", "optimization_insights"],
                    stability_assessment="stable"
                ))
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Adaptive behavior detection failed: {e}")
            return []
    
    async def _detect_self_organizing_behaviors(self) -> List[EmergentBehavior]:
        """Detect self-organizing behaviors"""
        behaviors = []
        
        try:
            # Look for systems organizing themselves without external control
            
            # Check for pattern self-organization
            if len(self.semantic_concepts) > 10:
                # Calculate organization metrics
                concept_types = [c.concept_type for c in self.semantic_concepts.values()]
                type_distribution = defaultdict(int)
                for ct in concept_types:
                    type_distribution[ct] += 1
                
                # If concepts are organizing into clear categories
                entropy = self._calculate_entropy(list(type_distribution.values()))
                if entropy < 1.5:  # Low entropy suggests organization
                    behaviors.append(EmergentBehavior(
                        behavior_id=f"concept_organization_{int(datetime.now().timestamp())}",
                        behavior_name="Concept Self-Organization",
                        behavior_type="self_organizing",
                        complexity_score=0.6,
                        emergence_conditions=["concept_discovery", "semantic_clustering"],
                        participating_systems=["semantic_learner"],
                        observable_effects=["concept_categorization", "semantic_structure"],
                        predictive_indicators=["concept_distribution", "entropy_metrics"],
                        stability_assessment="stable"
                    ))
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Self-organizing behavior detection failed: {e}")
            return []
    
    def _calculate_entropy(self, distribution: List[int]) -> float:
        """Calculate entropy of a distribution"""
        try:
            total = sum(distribution)
            if total == 0:
                return 0
            
            entropy = 0
            for count in distribution:
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            return entropy
            
        except Exception:
            return 0
    
    async def _detect_optimization_behaviors(self) -> List[EmergentBehavior]:
        """Detect system optimization behaviors"""
        behaviors = []
        
        try:
            # Look for systems optimizing themselves or each other
            
            # Cross-system optimization
            if (len(self.semantic_relationships) > 5 and 
                len(self.cross_system_patterns) > 2):
                behaviors.append(EmergentBehavior(
                    behavior_id=f"cross_optimization_{int(datetime.now().timestamp())}",
                    behavior_name="Cross-System Optimization",
                    behavior_type="optimization",
                    complexity_score=0.9,
                    emergence_conditions=["system_integration", "pattern_correlation"],
                    participating_systems=["analytics", "ml", "api", "patterns"],
                    observable_effects=["system_coordination", "performance_improvement"],
                    predictive_indicators=["relationship_strength", "pattern_emergence"],
                    stability_assessment="evolving"
                ))
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Optimization behavior detection failed: {e}")
            return []
    
    async def _detect_coordination_behaviors(self) -> List[EmergentBehavior]:
        """Detect coordination behaviors between systems"""
        behaviors = []
        
        try:
            # Look for emergent coordination patterns
            
            # Check for temporal coordination
            recent_events = []
            
            # Collect events from all systems (simplified)
            if len(self.semantic_concepts) > 0:
                recent_concepts = [c for c in self.semantic_concepts.values() 
                                 if (datetime.now() - c.discovered_at).total_seconds() < 3600]
                
                if len(recent_concepts) > 3:
                    # Check if concepts are being discovered in coordination
                    systems_involved = set()
                    for concept in recent_concepts:
                        systems_involved.update(concept.system_manifestations.keys())
                    
                    if len(systems_involved) > 2:
                        behaviors.append(EmergentBehavior(
                            behavior_id=f"temporal_coordination_{int(datetime.now().timestamp())}",
                            behavior_name="Temporal System Coordination",
                            behavior_type="coordination",
                            complexity_score=0.7,
                            emergence_conditions=["multi_system_activity", "temporal_synchronization"],
                            participating_systems=list(systems_involved),
                            observable_effects=["synchronized_discovery", "coordinated_learning"],
                            predictive_indicators=["concept_timing", "system_activity"],
                            stability_assessment="stable"
                        ))
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"Coordination behavior detection failed: {e}")
            return []
    
    def _validate_emergent_behavior(self, behavior: EmergentBehavior) -> bool:
        """Validate that a behavior is truly emergent"""
        try:
            # Emergent behaviors should have certain characteristics:
            # 1. Involve multiple systems or components
            # 2. Show complexity beyond individual parts
            # 3. Have observable effects
            
            validation_score = 0
            
            # Multiple systems involved
            if len(behavior.participating_systems) > 1:
                validation_score += 0.4
            
            # Has complexity
            if behavior.complexity_score > 0.5:
                validation_score += 0.3
            
            # Has observable effects
            if len(behavior.observable_effects) > 1:
                validation_score += 0.3
            
            return validation_score >= self.config["emergence_detection_threshold"]
            
        except Exception:
            return False
    
    async def _validate_emergent_behaviors(self) -> List[EmergentBehavior]:
        """Validate discovered emergent behaviors"""
        # Placeholder for behavior validation
        return []
    
    async def _predict_behavior_evolution(self) -> List[EmergentBehavior]:
        """Predict how emergent behaviors will evolve"""
        # Placeholder for behavior evolution prediction
        return []
    
    async def _store_emergent_behavior(self, behavior: EmergentBehavior):
        """Store an emergent behavior"""
        try:
            self.emergent_behaviors[behavior.behavior_id] = behavior
            self.learning_stats["emergent_behaviors_discovered"] += 1
            
            self.logger.info(f"Discovered emergent behavior: {behavior.behavior_name} (complexity: {behavior.complexity_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to store emergent behavior: {e}")
    
    async def _execute_semantic_reasoning(self):
        """Execute semantic reasoning to derive new insights"""
        try:
            # Use discovered concepts and relationships for reasoning
            inferences = await self._make_semantic_inferences()
            
            # Apply reasoned insights to improve system understanding
            await self._apply_semantic_insights(inferences)
            
            self.learning_stats["semantic_inferences_made"] += len(inferences)
            
        except Exception as e:
            self.logger.error(f"Semantic reasoning failed: {e}")
    
    async def _make_semantic_inferences(self) -> List[Dict[str, Any]]:
        """Make inferences from semantic knowledge"""
        inferences = []
        
        try:
            # Inference 1: If A causes B and B causes C, then A may cause C
            for rel1 in self.semantic_relationships.values():
                if rel1.relationship_type == "causes":
                    for rel2 in self.semantic_relationships.values():
                        if (rel2.relationship_type == "causes" and 
                            rel1.target_concept == rel2.source_concept):
                            
                            # Potential transitive causation
                            confidence = rel1.strength * rel2.strength * 0.7  # Reduce confidence for inference
                            
                            if confidence > 0.4:
                                inferences.append({
                                    "type": "transitive_causation",
                                    "source": rel1.source_concept,
                                    "target": rel2.target_concept,
                                    "confidence": confidence,
                                    "evidence": [rel1.relationship_id, rel2.relationship_id]
                                })
            
            # Inference 2: Highly correlated concepts in same domain may share properties
            for rel in self.semantic_relationships.values():
                if rel.relationship_type == "correlates" and rel.strength > 0.8:
                    source_concept = self.semantic_concepts.get(rel.source_concept)
                    target_concept = self.semantic_concepts.get(rel.target_concept)
                    
                    if (source_concept and target_concept and 
                        source_concept.semantic_properties.get("domain") == 
                        target_concept.semantic_properties.get("domain")):
                        
                        inferences.append({
                            "type": "shared_properties",
                            "concepts": [rel.source_concept, rel.target_concept],
                            "confidence": rel.strength,
                            "shared_domain": source_concept.semantic_properties.get("domain")
                        })
            
            return inferences
            
        except Exception as e:
            self.logger.error(f"Semantic inference failed: {e}")
            return []
    
    async def _apply_semantic_insights(self, inferences: List[Dict[str, Any]]):
        """Apply semantic insights to improve system understanding"""
        try:
            for inference in inferences:
                if inference["confidence"] > 0.6:
                    if inference["type"] == "transitive_causation":
                        # Create inferred causal relationship
                        inferred_rel = SemanticRelationship(
                            relationship_id=f"inferred_causal_{inference['source']}_{inference['target']}",
                            source_concept=inference["source"],
                            target_concept=inference["target"],
                            relationship_type="causes",
                            strength=inference["confidence"],
                            directionality="source_to_target",
                            evidence=[{"type": "transitive_inference", "source_relations": inference["evidence"]}],
                            systems_involved=[]  # Will be populated based on concept systems
                        )
                        
                        await self._store_semantic_relationship(inferred_rel)
                    
                    elif inference["type"] == "shared_properties":
                        # Update concepts with shared property insights
                        for concept_id in inference["concepts"]:
                            if concept_id in self.semantic_concepts:
                                concept = self.semantic_concepts[concept_id]
                                concept.semantic_properties["inferred_shared_properties"] = True
                                concept.semantic_properties["correlation_group"] = inference["shared_domain"]
                
        except Exception as e:
            self.logger.error(f"Semantic insight application failed: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status"""
        return {
            "is_learning": self.is_learning,
            "learning_stats": self.learning_stats.copy(),
            "knowledge_base_size": {
                "concepts": len(self.semantic_concepts),
                "relationships": len(self.semantic_relationships),
                "emergent_behaviors": len(self.emergent_behaviors)
            },
            "recent_concepts": [
                {
                    "name": concept.concept_name,
                    "type": concept.concept_type,
                    "confidence": concept.confidence,
                    "abstraction_level": concept.abstraction_level,
                    "systems": list(concept.system_manifestations.keys()),
                    "discovered_at": concept.discovered_at.isoformat()
                }
                for concept in sorted(self.semantic_concepts.values(), 
                                   key=lambda x: x.discovered_at, reverse=True)[:10]
            ],
            "recent_behaviors": [
                {
                    "name": behavior.behavior_name,
                    "type": behavior.behavior_type,
                    "complexity": behavior.complexity_score,
                    "systems": behavior.participating_systems,
                    "stability": behavior.stability_assessment
                }
                for behavior in sorted(self.emergent_behaviors.values(), 
                                    key=lambda x: x.behavior_id, reverse=True)[:5]
            ],
            "configuration": self.config.copy()
        }
    
    def get_semantic_knowledge_graph(self) -> Dict[str, Any]:
        """Get semantic knowledge graph representation"""
        return {
            "concepts": {
                concept_id: {
                    "name": concept.concept_name,
                    "type": concept.concept_type,
                    "confidence": concept.confidence,
                    "abstraction_level": concept.abstraction_level,
                    "systems": list(concept.system_manifestations.keys()),
                    "properties": concept.semantic_properties
                }
                for concept_id, concept in self.semantic_concepts.items()
            },
            "relationships": {
                rel_id: {
                    "source": rel.source_concept,
                    "target": rel.target_concept,
                    "type": rel.relationship_type,
                    "strength": rel.strength,
                    "directionality": rel.directionality,
                    "systems": rel.systems_involved
                }
                for rel_id, rel in self.semantic_relationships.items()
            },
            "emergent_behaviors": {
                behavior_id: {
                    "name": behavior.behavior_name,
                    "type": behavior.behavior_type,
                    "complexity": behavior.complexity_score,
                    "systems": behavior.participating_systems,
                    "effects": behavior.observable_effects,
                    "stability": behavior.stability_assessment
                }
                for behavior_id, behavior in self.emergent_behaviors.items()
            }
        }


# Export
__all__ = ['CrossSystemSemanticLearner', 'SemanticConcept', 'SemanticRelationship', 'EmergentBehavior']