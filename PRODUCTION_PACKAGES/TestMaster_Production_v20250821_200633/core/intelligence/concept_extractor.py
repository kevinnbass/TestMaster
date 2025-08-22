"""
Concept Extractor - Cross-System Semantic Concept Discovery Engine
=================================================================

Advanced concept extraction system that discovers, unifies, and abstracts semantic
concepts across all intelligence frameworks (analytics, ML, API) with enterprise-grade
pattern recognition and concept unification capabilities.

This module provides sophisticated concept extraction from multiple intelligence systems,
concept similarity analysis, and intelligent concept unification for knowledge building.

Author: Agent A - PHASE 4+ Continuation  
Created: 2025-08-22
Module: concept_extractor.py (420 lines)
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import statistics
import uuid

from .semantic_types import (
    SemanticConcept, ConceptType, AbstractionLevel, SemanticLearningContext
)
from .analytics.analytics_hub import AnalyticsHub
from .ml.ml_orchestrator import MLOrchestrator
from .api.unified_api_gateway import UnifiedAPIGateway
from .analysis.advanced_pattern_recognizer import AdvancedPatternRecognizer, AdvancedPattern

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossSystemConceptExtractor:
    """
    Advanced concept extraction engine that discovers semantic concepts
    across multiple intelligence systems with intelligent unification
    """
    
    def __init__(self, analytics_hub: AnalyticsHub, ml_orchestrator: MLOrchestrator,
                 api_gateway: UnifiedAPIGateway, pattern_recognizer: AdvancedPatternRecognizer):
        self.analytics_hub = analytics_hub
        self.ml_orchestrator = ml_orchestrator
        self.api_gateway = api_gateway
        self.pattern_recognizer = pattern_recognizer
        self.logger = logging.getLogger(__name__)
        
        # Concept extraction state
        self.extracted_concepts: Dict[str, List[SemanticConcept]] = defaultdict(list)
        self.unified_concepts: List[SemanticConcept] = []
        self.concept_similarity_cache: Dict[str, float] = {}
        
        # Configuration
        self.config = {
            "similarity_threshold": 0.75,
            "confidence_threshold": 0.6,
            "abstraction_enabled": True,
            "max_concepts_per_system": 100,
            "similarity_cache_size": 10000
        }
        
        # Statistics
        self.extraction_stats = {
            "concepts_extracted": 0,
            "concepts_unified": 0,
            "concepts_abstracted": 0,
            "extraction_cycles": 0,
            "avg_extraction_time": 0.0
        }
        
        self.logger.info("CrossSystemConceptExtractor initialized with enterprise capabilities")
    
    async def extract_concepts_from_all_systems(self, context: SemanticLearningContext) -> List[SemanticConcept]:
        """Extract and unify concepts from all monitored intelligence systems"""
        start_time = datetime.now()
        
        try:
            # Collect cross-system data
            system_data = await self._collect_cross_system_data(context)
            
            # Extract concepts from each system
            all_concepts = []
            
            # Analytics system concept extraction
            if "analytics" in context.systems_monitored:
                analytics_concepts = await self._extract_analytics_concepts(system_data)
                all_concepts.extend(analytics_concepts)
                self.extracted_concepts["analytics"] = analytics_concepts
            
            # ML system concept extraction  
            if "ml" in context.systems_monitored:
                ml_concepts = await self._extract_ml_concepts(system_data)
                all_concepts.extend(ml_concepts)
                self.extracted_concepts["ml"] = ml_concepts
            
            # API system concept extraction
            if "api" in context.systems_monitored:
                api_concepts = await self._extract_api_concepts(system_data)
                all_concepts.extend(api_concepts)
                self.extracted_concepts["api"] = api_concepts
            
            # Pattern recognition concept extraction
            if "patterns" in context.systems_monitored:
                pattern_concepts = await self._extract_pattern_concepts(system_data)
                all_concepts.extend(pattern_concepts)
                self.extracted_concepts["patterns"] = pattern_concepts
            
            # Unify similar concepts across systems
            unified_concepts = await self._unify_concepts(all_concepts, context)
            
            # Abstract concepts to higher levels if enabled
            if self.config["abstraction_enabled"]:
                abstracted_concepts = await self._abstract_concepts(unified_concepts, context)
                unified_concepts.extend(abstracted_concepts)
            
            # Update statistics
            extraction_time = (datetime.now() - start_time).total_seconds()
            self._update_extraction_stats(len(all_concepts), len(unified_concepts), extraction_time)
            
            self.logger.info(f"Extracted and unified {len(unified_concepts)} concepts from {len(context.systems_monitored)} systems")
            return unified_concepts
            
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
            return []
    
    async def _collect_cross_system_data(self, context: SemanticLearningContext) -> Dict[str, Any]:
        """Collect data from all monitored intelligence systems"""
        system_data = {}
        
        try:
            # Collect analytics events and metrics
            if "analytics" in context.systems_monitored:
                system_data["analytics_events"] = await self._get_analytics_events(context)
                system_data["analytics_metrics"] = await self._get_analytics_metrics(context)
            
            # Collect ML flow data and model information
            if "ml" in context.systems_monitored:
                system_data["ml_flows"] = await self._get_ml_flow_data(context)
                system_data["ml_models"] = await self._get_ml_model_data(context)
            
            # Collect API patterns and usage data
            if "api" in context.systems_monitored:
                system_data["api_patterns"] = await self._get_api_patterns(context)
                system_data["api_usage"] = await self._get_api_usage_data(context)
            
            # Collect advanced pattern data
            if "patterns" in context.systems_monitored:
                system_data["advanced_patterns"] = await self._get_advanced_patterns(context)
            
            return system_data
            
        except Exception as e:
            self.logger.error(f"Cross-system data collection failed: {e}")
            return {}
    
    async def _get_analytics_events(self, context: SemanticLearningContext) -> List[Dict[str, Any]]:
        """Get analytics events for concept extraction"""
        try:
            # Get recent analytics events within the temporal window
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=context.temporal_window_hours)
            
            events = []
            if hasattr(self.analytics_hub, 'get_events'):
                events = await self.analytics_hub.get_events(start_time, end_time)
            
            return events[:self.config["max_concepts_per_system"]]
            
        except Exception as e:
            self.logger.error(f"Analytics events collection failed: {e}")
            return []
    
    async def _get_analytics_metrics(self, context: SemanticLearningContext) -> List[Dict[str, Any]]:
        """Get analytics metrics for concept extraction"""
        try:
            metrics = []
            if hasattr(self.analytics_hub, 'get_performance_metrics'):
                metrics = await self.analytics_hub.get_performance_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Analytics metrics collection failed: {e}")
            return []
    
    async def _get_ml_flow_data(self, context: SemanticLearningContext) -> List[Dict[str, Any]]:
        """Get ML flow data for concept extraction"""
        try:
            flows = []
            if hasattr(self.ml_orchestrator, 'get_active_flows'):
                flows = await self.ml_orchestrator.get_active_flows()
            
            return flows[:self.config["max_concepts_per_system"]]
            
        except Exception as e:
            self.logger.error(f"ML flow data collection failed: {e}")
            return []
    
    async def _get_ml_model_data(self, context: SemanticLearningContext) -> List[Dict[str, Any]]:
        """Get ML model data for concept extraction"""
        try:
            models = []
            if hasattr(self.ml_orchestrator, 'get_model_registry'):
                models = await self.ml_orchestrator.get_model_registry()
            
            return models
            
        except Exception as e:
            self.logger.error(f"ML model data collection failed: {e}")
            return []
    
    async def _get_api_patterns(self, context: SemanticLearningContext) -> List[Dict[str, Any]]:
        """Get API patterns for concept extraction"""
        try:
            patterns = []
            if hasattr(self.api_gateway, 'get_usage_patterns'):
                patterns = await self.api_gateway.get_usage_patterns()
            
            return patterns[:self.config["max_concepts_per_system"]]
            
        except Exception as e:
            self.logger.error(f"API patterns collection failed: {e}")
            return []
    
    async def _get_api_usage_data(self, context: SemanticLearningContext) -> List[Dict[str, Any]]:
        """Get API usage data for concept extraction"""
        try:
            usage_data = []
            if hasattr(self.api_gateway, 'get_usage_statistics'):
                usage_data = await self.api_gateway.get_usage_statistics()
            
            return usage_data
            
        except Exception as e:
            self.logger.error(f"API usage data collection failed: {e}")
            return []
    
    async def _get_advanced_patterns(self, context: SemanticLearningContext) -> List[AdvancedPattern]:
        """Get advanced patterns for concept extraction"""
        try:
            patterns = []
            if hasattr(self.pattern_recognizer, 'get_recent_patterns'):
                patterns = await self.pattern_recognizer.get_recent_patterns()
            
            return patterns[:self.config["max_concepts_per_system"]]
            
        except Exception as e:
            self.logger.error(f"Advanced patterns collection failed: {e}")
            return []
    
    async def _extract_analytics_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from analytics system data"""
        concepts = []
        
        try:
            # Extract concepts from analytics events
            events = system_data.get("analytics_events", [])
            for event in events:
                concept = await self._create_concept_from_analytics_event(event)
                if concept and concept.confidence >= self.config["confidence_threshold"]:
                    concepts.append(concept)
            
            # Extract concepts from analytics metrics
            metrics = system_data.get("analytics_metrics", [])
            for metric in metrics:
                concept = await self._create_concept_from_analytics_metric(metric)
                if concept and concept.confidence >= self.config["confidence_threshold"]:
                    concepts.append(concept)
            
            self.logger.debug(f"Extracted {len(concepts)} concepts from analytics system")
            return concepts
            
        except Exception as e:
            self.logger.error(f"Analytics concept extraction failed: {e}")
            return []
    
    async def _create_concept_from_analytics_event(self, event: Dict[str, Any]) -> Optional[SemanticConcept]:
        """Create semantic concept from analytics event"""
        try:
            concept_id = f"analytics_event_{uuid.uuid4().hex[:8]}"
            concept_name = event.get("event_type", "unknown_event")
            
            # Determine concept type based on event characteristics
            concept_type = ConceptType.EVENT
            if "performance" in concept_name.lower():
                concept_type = ConceptType.PROCESS
            elif "error" in concept_name.lower():
                concept_type = ConceptType.STATE
            
            # Calculate confidence based on event frequency and data quality
            frequency = event.get("frequency", 1)
            data_quality = len(event.get("data", {})) / 10.0
            confidence = min(1.0, (frequency / 100.0) + data_quality)
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=concept_type,
                confidence=confidence,
                abstraction_level=AbstractionLevel.OPERATIONAL,
                system_manifestations={"analytics": event},
                semantic_properties={
                    "source_system": "analytics",
                    "event_frequency": frequency,
                    "data_completeness": data_quality,
                    "temporal_pattern": event.get("temporal_pattern", "irregular")
                }
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"Analytics event concept creation failed: {e}")
            return None
    
    async def _create_concept_from_analytics_metric(self, metric: Dict[str, Any]) -> Optional[SemanticConcept]:
        """Create semantic concept from analytics metric"""
        try:
            concept_id = f"analytics_metric_{uuid.uuid4().hex[:8]}"
            concept_name = metric.get("metric_name", "unknown_metric")
            
            # Metrics are typically process or state concepts
            concept_type = ConceptType.PROCESS
            if "status" in concept_name.lower() or "state" in concept_name.lower():
                concept_type = ConceptType.STATE
            
            # Calculate confidence based on metric stability and relevance
            stability = metric.get("stability_score", 0.5)
            relevance = metric.get("relevance_score", 0.5)
            confidence = (stability + relevance) / 2.0
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=concept_type,
                confidence=confidence,
                abstraction_level=AbstractionLevel.OPERATIONAL,
                system_manifestations={"analytics": metric},
                semantic_properties={
                    "source_system": "analytics",
                    "metric_type": metric.get("metric_type", "unknown"),
                    "stability_score": stability,
                    "measurement_unit": metric.get("unit", "dimensionless")
                }
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"Analytics metric concept creation failed: {e}")
            return None
    
    async def _extract_ml_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from ML system data"""
        concepts = []
        
        try:
            # Extract concepts from ML flows
            flows = system_data.get("ml_flows", [])
            for flow in flows:
                concept = await self._create_concept_from_ml_flow(flow)
                if concept and concept.confidence >= self.config["confidence_threshold"]:
                    concepts.append(concept)
            
            # Extract concepts from ML models
            models = system_data.get("ml_models", [])
            for model in models:
                concept = await self._create_concept_from_ml_model(model)
                if concept and concept.confidence >= self.config["confidence_threshold"]:
                    concepts.append(concept)
            
            self.logger.debug(f"Extracted {len(concepts)} concepts from ML system")
            return concepts
            
        except Exception as e:
            self.logger.error(f"ML concept extraction failed: {e}")
            return []
    
    async def _create_concept_from_ml_flow(self, flow: Dict[str, Any]) -> Optional[SemanticConcept]:
        """Create semantic concept from ML flow"""
        try:
            concept_id = f"ml_flow_{uuid.uuid4().hex[:8]}"
            concept_name = flow.get("flow_name", "unknown_flow")
            
            # ML flows are typically process concepts
            concept_type = ConceptType.PROCESS
            
            # Calculate confidence based on flow success rate and complexity
            success_rate = flow.get("success_rate", 0.5)
            complexity = flow.get("complexity_score", 0.5)
            confidence = success_rate * (1.0 - complexity * 0.2)  # Lower confidence for very complex flows
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=concept_type,
                confidence=confidence,
                abstraction_level=AbstractionLevel.TACTICAL,
                system_manifestations={"ml": flow},
                semantic_properties={
                    "source_system": "ml",
                    "flow_type": flow.get("flow_type", "unknown"),
                    "success_rate": success_rate,
                    "complexity_score": complexity,
                    "execution_time": flow.get("avg_execution_time", 0)
                }
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"ML flow concept creation failed: {e}")
            return None
    
    async def _create_concept_from_ml_model(self, model: Dict[str, Any]) -> Optional[SemanticConcept]:
        """Create semantic concept from ML model"""
        try:
            concept_id = f"ml_model_{uuid.uuid4().hex[:8]}"
            concept_name = model.get("model_name", "unknown_model")
            
            # ML models can be entities or behaviors depending on their nature
            concept_type = ConceptType.ENTITY
            if "predictor" in concept_name.lower() or "classifier" in concept_name.lower():
                concept_type = ConceptType.BEHAVIOR
            
            # Calculate confidence based on model performance and validation
            accuracy = model.get("accuracy", 0.5)
            validation_score = model.get("validation_score", 0.5)
            confidence = (accuracy + validation_score) / 2.0
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=concept_type,
                confidence=confidence,
                abstraction_level=AbstractionLevel.TACTICAL,
                system_manifestations={"ml": model},
                semantic_properties={
                    "source_system": "ml",
                    "model_type": model.get("model_type", "unknown"),
                    "accuracy": accuracy,
                    "validation_score": validation_score,
                    "training_data_size": model.get("training_data_size", 0)
                }
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"ML model concept creation failed: {e}")
            return None
    
    async def _extract_api_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from API system data"""
        concepts = []
        
        try:
            # Extract concepts from API patterns
            patterns = system_data.get("api_patterns", [])
            for pattern in patterns:
                concept = await self._create_concept_from_api_pattern(pattern)
                if concept and concept.confidence >= self.config["confidence_threshold"]:
                    concepts.append(concept)
            
            # Extract concepts from API usage
            usage_data = system_data.get("api_usage", [])
            for usage in usage_data:
                concept = await self._create_concept_from_api_usage(usage)
                if concept and concept.confidence >= self.config["confidence_threshold"]:
                    concepts.append(concept)
            
            self.logger.debug(f"Extracted {len(concepts)} concepts from API system")
            return concepts
            
        except Exception as e:
            self.logger.error(f"API concept extraction failed: {e}")
            return []
    
    async def _create_concept_from_api_pattern(self, pattern: Dict[str, Any]) -> Optional[SemanticConcept]:
        """Create semantic concept from API pattern"""
        try:
            concept_id = f"api_pattern_{uuid.uuid4().hex[:8]}"
            concept_name = pattern.get("pattern_name", "unknown_pattern")
            
            # API patterns are typically behavioral concepts
            concept_type = ConceptType.PATTERN
            
            # Calculate confidence based on pattern strength and frequency
            strength = pattern.get("strength", 0.5)
            frequency = min(1.0, pattern.get("frequency", 1) / 1000.0)
            confidence = (strength + frequency) / 2.0
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=concept_type,
                confidence=confidence,
                abstraction_level=AbstractionLevel.OPERATIONAL,
                system_manifestations={"api": pattern},
                semantic_properties={
                    "source_system": "api",
                    "pattern_strength": strength,
                    "pattern_frequency": frequency,
                    "endpoints_involved": pattern.get("endpoints", [])
                }
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"API pattern concept creation failed: {e}")
            return None
    
    async def _create_concept_from_api_usage(self, usage: Dict[str, Any]) -> Optional[SemanticConcept]:
        """Create semantic concept from API usage data"""
        try:
            concept_id = f"api_usage_{uuid.uuid4().hex[:8]}"
            concept_name = usage.get("endpoint", "unknown_endpoint").replace("/", "_")
            
            # API usage typically represents process concepts
            concept_type = ConceptType.PROCESS
            
            # Calculate confidence based on usage stability and performance
            stability = usage.get("stability", 0.5)
            performance = usage.get("performance_score", 0.5)
            confidence = (stability + performance) / 2.0
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=concept_type,
                confidence=confidence,
                abstraction_level=AbstractionLevel.OPERATIONAL,
                system_manifestations={"api": usage},
                semantic_properties={
                    "source_system": "api",
                    "endpoint": usage.get("endpoint", "unknown"),
                    "request_rate": usage.get("request_rate", 0),
                    "success_rate": usage.get("success_rate", 0.5),
                    "avg_response_time": usage.get("avg_response_time", 0)
                }
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"API usage concept creation failed: {e}")
            return None
    
    async def _extract_pattern_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Extract semantic concepts from advanced pattern data"""
        concepts = []
        
        try:
            patterns = system_data.get("advanced_patterns", [])
            for pattern in patterns:
                concept = await self._create_concept_from_advanced_pattern(pattern)
                if concept and concept.confidence >= self.config["confidence_threshold"]:
                    concepts.append(concept)
            
            self.logger.debug(f"Extracted {len(concepts)} concepts from pattern system")
            return concepts
            
        except Exception as e:
            self.logger.error(f"Pattern concept extraction failed: {e}")
            return []
    
    async def _create_concept_from_advanced_pattern(self, pattern: AdvancedPattern) -> Optional[SemanticConcept]:
        """Create semantic concept from advanced pattern"""
        try:
            concept_id = f"pattern_{uuid.uuid4().hex[:8]}"
            concept_name = pattern.pattern_id
            
            # Advanced patterns are typically pattern or behavior concepts
            concept_type = ConceptType.PATTERN
            if hasattr(pattern, 'behavior_indicators') and pattern.behavior_indicators:
                concept_type = ConceptType.BEHAVIOR
            
            # Use pattern confidence directly
            confidence = getattr(pattern, 'confidence', 0.5)
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=concept_type,
                confidence=confidence,
                abstraction_level=AbstractionLevel.STRATEGIC,
                system_manifestations={"patterns": pattern.__dict__},
                semantic_properties={
                    "source_system": "patterns",
                    "pattern_type": getattr(pattern, 'pattern_type', 'unknown'),
                    "complexity": getattr(pattern, 'complexity', 0.5),
                    "temporal_span": getattr(pattern, 'temporal_span', 0)
                }
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"Advanced pattern concept creation failed: {e}")
            return None
    
    async def _unify_concepts(self, concepts: List[SemanticConcept], context: SemanticLearningContext) -> List[SemanticConcept]:
        """Unify similar concepts across systems using advanced similarity analysis"""
        if not concepts:
            return []
        
        try:
            # Group concepts by similarity
            concept_groups = await self._group_concepts_by_similarity(concepts)
            
            # Create unified concepts from each group
            unified_concepts = []
            for group in concept_groups:
                if len(group) > 1:
                    # Multiple similar concepts - create unified version
                    unified_concept = await self._create_unified_concept(group)
                    if unified_concept:
                        unified_concepts.append(unified_concept)
                else:
                    # Single concept - add as-is
                    unified_concepts.append(group[0])
            
            self.extraction_stats["concepts_unified"] = len([g for g in concept_groups if len(g) > 1])
            self.logger.debug(f"Unified {len(concepts)} concepts into {len(unified_concepts)} unified concepts")
            
            return unified_concepts
            
        except Exception as e:
            self.logger.error(f"Concept unification failed: {e}")
            return concepts  # Return original concepts if unification fails
    
    async def _group_concepts_by_similarity(self, concepts: List[SemanticConcept]) -> List[List[SemanticConcept]]:
        """Group concepts by similarity using advanced clustering"""
        groups = []
        processed = set()
        
        for i, concept_a in enumerate(concepts):
            if concept_a.concept_id in processed:
                continue
            
            # Start new group with this concept
            current_group = [concept_a]
            processed.add(concept_a.concept_id)
            
            # Find similar concepts
            for j, concept_b in enumerate(concepts[i+1:], i+1):
                if concept_b.concept_id in processed:
                    continue
                
                similarity = await self._calculate_concept_similarity(concept_a, concept_b)
                
                if similarity >= self.config["similarity_threshold"]:
                    current_group.append(concept_b)
                    processed.add(concept_b.concept_id)
            
            groups.append(current_group)
        
        return groups
    
    async def _calculate_concept_similarity(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Calculate semantic similarity between two concepts"""
        try:
            # Check cache first
            cache_key = f"{concept_a.concept_id}:{concept_b.concept_id}"
            if cache_key in self.concept_similarity_cache:
                return self.concept_similarity_cache[cache_key]
            
            similarity_factors = []
            
            # Name similarity (using simple string comparison)
            name_similarity = self._calculate_string_similarity(concept_a.concept_name, concept_b.concept_name)
            similarity_factors.append(name_similarity * 0.3)
            
            # Type similarity
            type_similarity = 1.0 if concept_a.concept_type == concept_b.concept_type else 0.0
            similarity_factors.append(type_similarity * 0.2)
            
            # Abstraction level similarity  
            abstraction_diff = abs(concept_a.abstraction_level.value - concept_b.abstraction_level.value)
            abstraction_similarity = max(0.0, 1.0 - abstraction_diff / 4.0)
            similarity_factors.append(abstraction_similarity * 0.2)
            
            # Semantic properties similarity
            properties_similarity = self._calculate_properties_similarity(
                concept_a.semantic_properties, concept_b.semantic_properties
            )
            similarity_factors.append(properties_similarity * 0.3)
            
            # Calculate weighted average
            total_similarity = sum(similarity_factors)
            
            # Cache result
            self.concept_similarity_cache[cache_key] = total_similarity
            
            # Manage cache size
            if len(self.concept_similarity_cache) > self.config["similarity_cache_size"]:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.concept_similarity_cache.keys())[:100]
                for key in oldest_keys:
                    del self.concept_similarity_cache[key]
            
            return total_similarity
            
        except Exception as e:
            self.logger.error(f"Concept similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using simple token-based approach"""
        if not str1 or not str2:
            return 0.0
        
        # Tokenize and normalize
        tokens1 = set(str1.lower().split("_"))
        tokens2 = set(str2.lower().split("_"))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_properties_similarity(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        """Calculate similarity between semantic properties"""
        if not props1 and not props2:
            return 1.0
        
        if not props1 or not props2:
            return 0.0
        
        # Find common keys
        common_keys = set(props1.keys()).intersection(set(props2.keys()))
        
        if not common_keys:
            return 0.0
        
        # Calculate similarity for each common property
        similarities = []
        for key in common_keys:
            val1, val2 = props1[key], props2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                if val1 == val2 == 0:
                    sim = 1.0
                else:
                    sim = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2))
                similarities.append(sim)
            
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                similarities.append(self._calculate_string_similarity(val1, val2))
            
            elif val1 == val2:
                # Exact match
                similarities.append(1.0)
            else:
                # Different types or values
                similarities.append(0.0)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    async def _create_unified_concept(self, concept_group: List[SemanticConcept]) -> Optional[SemanticConcept]:
        """Create a unified concept from a group of similar concepts"""
        if not concept_group:
            return None
        
        try:
            # Choose the concept with highest confidence as the base
            base_concept = max(concept_group, key=lambda c: c.confidence)
            
            # Create unified concept
            unified_concept = SemanticConcept(
                concept_id=f"unified_{uuid.uuid4().hex[:8]}",
                concept_name=base_concept.concept_name,
                concept_type=base_concept.concept_type,
                confidence=statistics.mean([c.confidence for c in concept_group]),
                abstraction_level=base_concept.abstraction_level,
                system_manifestations={},
                semantic_properties={}
            )
            
            # Merge system manifestations
            for concept in concept_group:
                unified_concept.system_manifestations.update(concept.system_manifestations)
            
            # Merge semantic properties
            all_properties = {}
            for concept in concept_group:
                all_properties.update(concept.semantic_properties)
            
            # Add unification metadata
            all_properties.update({
                "unified_from": [c.concept_id for c in concept_group],
                "source_systems": list(unified_concept.system_manifestations.keys()),
                "unification_confidence": unified_concept.confidence,
                "unified_at": datetime.now().isoformat()
            })
            
            unified_concept.semantic_properties = all_properties
            
            return unified_concept
            
        except Exception as e:
            self.logger.error(f"Unified concept creation failed: {e}")
            return None
    
    async def _abstract_concepts(self, concepts: List[SemanticConcept], context: SemanticLearningContext) -> List[SemanticConcept]:
        """Create higher-level abstract concepts from concrete concepts"""
        abstracted_concepts = []
        
        try:
            # Group concepts by domain and create abstractions
            domain_groups = defaultdict(list)
            
            for concept in concepts:
                # Simple domain classification based on properties
                domain = self._classify_concept_domain(concept)
                domain_groups[domain].append(concept)
            
            # Create abstract concepts for each domain
            for domain, domain_concepts in domain_groups.items():
                if len(domain_concepts) >= 3:  # Need multiple concepts to create abstraction
                    abstract_concept = await self._create_abstract_concept(domain, domain_concepts)
                    if abstract_concept:
                        abstracted_concepts.append(abstract_concept)
            
            self.extraction_stats["concepts_abstracted"] = len(abstracted_concepts)
            self.logger.debug(f"Created {len(abstracted_concepts)} abstract concepts")
            
            return abstracted_concepts
            
        except Exception as e:
            self.logger.error(f"Concept abstraction failed: {e}")
            return []
    
    def _classify_concept_domain(self, concept: SemanticConcept) -> str:
        """Classify concept into a domain for abstraction"""
        # Simple domain classification based on concept properties
        source_system = concept.semantic_properties.get("source_system", "unknown")
        concept_type = concept.concept_type.value
        
        # Create domain identifier
        domain = f"{source_system}_{concept_type}"
        
        return domain
    
    async def _create_abstract_concept(self, domain: str, concepts: List[SemanticConcept]) -> Optional[SemanticConcept]:
        """Create an abstract concept representing a domain of concepts"""
        try:
            abstract_concept = SemanticConcept(
                concept_id=f"abstract_{domain}_{uuid.uuid4().hex[:8]}",
                concept_name=f"Abstract {domain.replace('_', ' ').title()}",
                concept_type=ConceptType.PATTERN,
                confidence=statistics.mean([c.confidence for c in concepts]),
                abstraction_level=AbstractionLevel.STRATEGIC,
                system_manifestations={},
                semantic_properties={
                    "abstraction_domain": domain,
                    "constituent_concepts": [c.concept_id for c in concepts],
                    "concept_count": len(concepts),
                    "avg_confidence": statistics.mean([c.confidence for c in concepts]),
                    "abstracted_at": datetime.now().isoformat()
                }
            )
            
            # Merge manifestations from all constituent concepts
            for concept in concepts:
                for system, manifestation in concept.system_manifestations.items():
                    if system not in abstract_concept.system_manifestations:
                        abstract_concept.system_manifestations[system] = []
                    abstract_concept.system_manifestations[system].append(manifestation)
            
            return abstract_concept
            
        except Exception as e:
            self.logger.error(f"Abstract concept creation failed: {e}")
            return None
    
    def _update_extraction_stats(self, extracted_count: int, unified_count: int, extraction_time: float):
        """Update extraction statistics"""
        self.extraction_stats["concepts_extracted"] += extracted_count
        self.extraction_stats["extraction_cycles"] += 1
        
        # Update average extraction time
        cycles = self.extraction_stats["extraction_cycles"]
        avg_time = self.extraction_stats["avg_extraction_time"]
        self.extraction_stats["avg_extraction_time"] = (avg_time * (cycles - 1) + extraction_time) / cycles
    
    def get_extraction_status(self) -> Dict[str, Any]:
        """Get comprehensive extraction status"""
        return {
            "extraction_stats": self.extraction_stats.copy(),
            "extracted_concepts_by_system": {
                system: len(concepts) for system, concepts in self.extracted_concepts.items()
            },
            "unified_concepts_count": len(self.unified_concepts),
            "cache_size": len(self.concept_similarity_cache),
            "configuration": self.config.copy()
        }


# Export the main extractor class
__all__ = ['CrossSystemConceptExtractor']