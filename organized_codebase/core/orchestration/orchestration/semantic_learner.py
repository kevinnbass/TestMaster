"""
Cross-System Semantic Learning Engine

Main orchestration module for the semantic learning system that coordinates
concept extraction, relationship discovery, and emergent behavior detection.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional

from .semantic_types import (
    SemanticConcept, SemanticRelationship, EmergentBehavior,
    SemanticLearningConfig, SemanticLearningMetrics
)
from .concept_extractor import ConceptExtractor
from .relationship_discoverer import RelationshipDiscoverer
from .behavior_detector import EmergentBehaviorDetector
from .concept_unifier import ConceptUnifier


class CrossSystemSemanticLearner:
    """
    Advanced semantic learning engine that discovers unified patterns and concepts
    across all intelligence systems.
    """
    
    def __init__(self, analytics_hub=None, ml_orchestrator=None, 
                 api_gateway=None, pattern_recognizer=None,
                 config: Optional[SemanticLearningConfig] = None):
        self.analytics_hub = analytics_hub
        self.ml_orchestrator = ml_orchestrator
        self.api_gateway = api_gateway
        self.pattern_recognizer = pattern_recognizer
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or SemanticLearningConfig()
        
        # Semantic knowledge base
        self.semantic_concepts: Dict[str, SemanticConcept] = {}
        self.semantic_relationships: Dict[str, SemanticRelationship] = {}
        self.emergent_behaviors: Dict[str, EmergentBehavior] = {}
        
        # Learning components
        self.concept_extractor = ConceptExtractor()
        self.relationship_discoverer = RelationshipDiscoverer()
        self.behavior_detector = EmergentBehaviorDetector()
        self.concept_unifier = ConceptUnifier()
        
        # Learning data structures
        self.concept_co_occurrence_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cross_system_event_correlation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.behavioral_pattern_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Learning metrics
        self.metrics = SemanticLearningMetrics()
        
        # Learning state
        self.is_learning = False
        self.learning_task = None
        
        self.logger.info("Cross-System Semantic Learner initialized")
    
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
                cycle_start = datetime.now()
                
                # Collect cross-system data
                system_data = await self._collect_cross_system_data()
                
                # Execute learning phases
                await self._execute_concept_extraction(system_data)
                await self._execute_relationship_discovery()
                
                if self.config.emergent_behavior_detection_enabled:
                    await self._execute_behavior_detection()
                
                if self.config.semantic_reasoning_enabled:
                    await self._execute_semantic_reasoning()
                
                # Update metrics
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                self._update_metrics(cycle_duration)
                
                # Wait for next cycle
                await asyncio.sleep(self.config.learning_interval)
                
            except Exception as e:
                self.logger.error(f"Semantic learning loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
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
                system_data["analytics"] = await self._get_analytics_data()
            
            # ML orchestration data
            if self.ml_orchestrator:
                system_data["ml"] = await self._get_ml_data()
            
            # API gateway data
            if self.api_gateway:
                system_data["api"] = await self._get_api_data()
            
            # Pattern recognition data
            if self.pattern_recognizer:
                system_data["patterns"] = await self._get_pattern_data()
            
            return system_data
            
        except Exception as e:
            self.logger.error(f"Cross-system data collection failed: {e}")
            return system_data
    
    async def _get_analytics_data(self) -> Dict[str, Any]:
        """Get data from analytics hub"""
        data = {}
        try:
            data["status"] = self.analytics_hub.get_hub_status()
            data["insights"] = self.analytics_hub.get_recent_insights(limit=100)
            data["correlations"] = self.analytics_hub.get_correlation_matrix()
            data["events"] = await self._extract_analytics_events()
        except Exception as e:
            self.logger.debug(f"Failed to get analytics data: {e}")
        return data
    
    async def _get_ml_data(self) -> Dict[str, Any]:
        """Get data from ML orchestrator"""
        data = {}
        try:
            data["status"] = self.ml_orchestrator.get_orchestration_status()
            data["insights"] = self.ml_orchestrator.get_integration_insights()
            data["flows"] = await self._extract_ml_flows()
            data["modules"] = await self._extract_ml_modules()
        except Exception as e:
            self.logger.debug(f"Failed to get ML data: {e}")
        return data
    
    async def _get_api_data(self) -> Dict[str, Any]:
        """Get data from API gateway"""
        data = {}
        try:
            data["statistics"] = self.api_gateway.get_gateway_statistics()
            data["endpoints"] = self.api_gateway.get_endpoint_documentation()
            data["patterns"] = await self._extract_api_patterns()
        except Exception as e:
            self.logger.debug(f"Failed to get API data: {e}")
        return data
    
    async def _get_pattern_data(self) -> Dict[str, Any]:
        """Get data from pattern recognizer"""
        data = {}
        try:
            data["status"] = self.pattern_recognizer.get_recognition_status()
            data["correlations"] = self.pattern_recognizer.get_pattern_correlations()
            data["advanced_patterns"] = self.pattern_recognizer.get_recent_patterns()
        except Exception as e:
            self.logger.debug(f"Failed to get pattern data: {e}")
        return data
    
    async def _execute_concept_extraction(self, system_data: Dict[str, Any]):
        """Execute concept extraction phase"""
        try:
            # Extract concepts from each system
            all_concepts = []
            
            analytics_concepts = await self.concept_extractor.extract_analytics_concepts(system_data)
            all_concepts.extend(analytics_concepts)
            
            ml_concepts = await self.concept_extractor.extract_ml_concepts(system_data)
            all_concepts.extend(ml_concepts)
            
            api_concepts = await self.concept_extractor.extract_api_concepts(system_data)
            all_concepts.extend(api_concepts)
            
            pattern_concepts = await self.concept_extractor.extract_pattern_concepts(system_data)
            all_concepts.extend(pattern_concepts)
            
            # Create abstract concepts
            abstract_concepts = await self.concept_extractor.create_abstract_concepts(system_data)
            all_concepts.extend(abstract_concepts)
            
            # Unify similar concepts
            unified_concepts = await self.concept_unifier.unify_concepts(all_concepts, self.semantic_concepts)
            
            # Store concepts
            for concept in unified_concepts:
                if concept.confidence >= self.config.concept_confidence_threshold:
                    await self._store_semantic_concept(concept)
            
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
    
    async def _execute_relationship_discovery(self):
        """Execute relationship discovery phase"""
        try:
            concepts = list(self.semantic_concepts.values())
            
            # Limit concepts for performance
            if len(concepts) > self.config.max_concepts_per_cycle:
                concepts = concepts[:self.config.max_concepts_per_cycle]
            
            # Discover different types of relationships
            all_relationships = []
            
            causal = await self.relationship_discoverer.discover_causal_relationships(concepts)
            all_relationships.extend(causal)
            
            correlation = await self.relationship_discoverer.discover_correlation_relationships(concepts)
            all_relationships.extend(correlation)
            
            containment = await self.relationship_discoverer.discover_containment_relationships(concepts)
            all_relationships.extend(containment)
            
            transformation = await self.relationship_discoverer.discover_transformation_relationships(concepts)
            all_relationships.extend(transformation)
            
            # Validate and store relationships
            validated = await self.relationship_discoverer.validate_relationships(all_relationships)
            
            for relationship in validated:
                if relationship.strength >= self.config.relationship_strength_threshold:
                    await self._store_semantic_relationship(relationship)
            
        except Exception as e:
            self.logger.error(f"Relationship discovery failed: {e}")
    
    async def _execute_behavior_detection(self):
        """Execute emergent behavior detection phase"""
        try:
            # Detect emergent behaviors
            behaviors = await self.behavior_detector.detect_emergent_behaviors(
                self.semantic_concepts,
                self.semantic_relationships,
                self.behavioral_pattern_history
            )
            
            # Store discovered behaviors
            for behavior in behaviors:
                if behavior.complexity_score >= self.config.emergence_detection_threshold:
                    await self._store_emergent_behavior(behavior)
            
        except Exception as e:
            self.logger.error(f"Behavior detection failed: {e}")
    
    async def _execute_semantic_reasoning(self):
        """Execute semantic reasoning phase"""
        try:
            # Perform semantic inference
            inferences = await self._perform_semantic_inference()
            self.metrics.semantic_inferences_made += len(inferences)
            
            # Update co-occurrence matrix
            self._update_co_occurrence_matrix()
            
            # Analyze cross-system correlations
            correlations = self._analyze_cross_system_correlations()
            self.metrics.cross_system_correlations += len(correlations)
            
        except Exception as e:
            self.logger.error(f"Semantic reasoning failed: {e}")
    
    async def _store_semantic_concept(self, concept: SemanticConcept):
        """Store a semantic concept in the knowledge base"""
        # Check for existing similar concept
        existing = self.concept_unifier.find_existing_concept(concept, self.semantic_concepts)
        
        if existing:
            # Merge with existing concept
            await self.concept_unifier.merge_concepts(existing, concept)
        else:
            # Store new concept
            self.semantic_concepts[concept.concept_id] = concept
            self.metrics.concepts_discovered += 1
            self.logger.debug(f"Stored concept: {concept.concept_name}")
    
    async def _store_semantic_relationship(self, relationship: SemanticRelationship):
        """Store a semantic relationship"""
        self.semantic_relationships[relationship.relationship_id] = relationship
        self.metrics.relationships_discovered += 1
        self.logger.debug(f"Stored relationship: {relationship.relationship_type}")
    
    async def _store_emergent_behavior(self, behavior: EmergentBehavior):
        """Store an emergent behavior"""
        self.emergent_behaviors[behavior.behavior_id] = behavior
        self.metrics.emergent_behaviors_discovered += 1
        self.logger.info(f"Discovered emergent behavior: {behavior.behavior_name}")
    
    def _update_metrics(self, cycle_duration: float):
        """Update learning metrics"""
        self.metrics.learning_cycles_completed += 1
        self.metrics.last_learning_cycle = datetime.now()
        
        # Calculate average cycle duration
        if self.metrics.learning_cycles_completed > 0:
            total_duration = self.metrics.average_cycle_duration * (self.metrics.learning_cycles_completed - 1)
            self.metrics.average_cycle_duration = (total_duration + cycle_duration) / self.metrics.learning_cycles_completed
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status"""
        return {
            "is_learning": self.is_learning,
            "metrics": {
                "concepts_discovered": self.metrics.concepts_discovered,
                "relationships_discovered": self.metrics.relationships_discovered,
                "emergent_behaviors_discovered": self.metrics.emergent_behaviors_discovered,
                "learning_cycles_completed": self.metrics.learning_cycles_completed,
                "average_cycle_duration": self.metrics.average_cycle_duration
            },
            "knowledge_base": {
                "total_concepts": len(self.semantic_concepts),
                "total_relationships": len(self.semantic_relationships),
                "total_behaviors": len(self.emergent_behaviors)
            },
            "configuration": {
                "learning_interval": self.config.learning_interval,
                "concept_threshold": self.config.concept_confidence_threshold,
                "relationship_threshold": self.config.relationship_strength_threshold
            }
        }
    
    # Helper methods for data extraction
    async def _extract_analytics_events(self) -> List[Dict[str, Any]]:
        """Extract events from analytics hub"""
        events = []
        # Implementation would extract actual events
        return events
    
    async def _extract_ml_flows(self) -> List[Dict[str, Any]]:
        """Extract ML flow data"""
        flows = []
        # Implementation would extract actual flows
        return flows
    
    async def _extract_ml_modules(self) -> List[Dict[str, Any]]:
        """Extract ML module data"""
        modules = []
        # Implementation would extract actual modules
        return modules
    
    async def _extract_api_patterns(self) -> List[Dict[str, Any]]:
        """Extract API usage patterns"""
        patterns = []
        # Implementation would extract actual patterns
        return patterns
    
    async def _perform_semantic_inference(self) -> List[Dict[str, Any]]:
        """Perform semantic inference on knowledge base"""
        inferences = []
        # Implementation would perform actual inference
        return inferences
    
    def _update_co_occurrence_matrix(self):
        """Update concept co-occurrence matrix"""
        # Implementation would update co-occurrence patterns
        pass
    
    def _analyze_cross_system_correlations(self) -> List[Dict[str, Any]]:
        """Analyze correlations across systems"""
        correlations = []
        # Implementation would analyze actual correlations
        return correlations