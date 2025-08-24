"""
Semantic Concept Extraction Module

This module handles the extraction of semantic concepts from various
intelligence systems including analytics, ML, API, and pattern recognition.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional

from .semantic_types import SemanticConcept


class ConceptExtractor:
    """Extracts semantic concepts from cross-system data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.extraction_stats = {
            "analytics_concepts": 0,
            "ml_concepts": 0,
            "api_concepts": 0,
            "pattern_concepts": 0,
            "unified_concepts": 0,
            "abstract_concepts": 0
        }
    
    async def extract_analytics_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
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
                        system_manifestations={
                            "analytics": {
                                "category": insight.category,
                                "priority": getattr(insight, 'priority', 5)
                            }
                        },
                        semantic_properties={"domain": "analytics", "temporal": True},
                        related_concepts=[],
                        evolution_history=[]
                    ))
            
            # Extract concepts from events
            events = analytics_data.get("events", [])
            event_types = set(event.get("type") for event in events)
            for event_type in event_types:
                type_events = [e for e in events if e.get("type") == event_type]
                avg_confidence = statistics.mean([
                    e.get("confidence", 0.5) for e in type_events if "confidence" in e
                ]) if type_events else 0.5
                
                concepts.append(SemanticConcept(
                    concept_id=f"analytics_event_{event_type}_{int(datetime.now().timestamp())}",
                    concept_name=f"Analytics Event: {event_type}",
                    concept_type="behavior",
                    confidence=avg_confidence,
                    abstraction_level=3,
                    system_manifestations={
                        "analytics": {"event_type": event_type, "frequency": len(type_events)}
                    },
                    semantic_properties={"domain": "analytics", "behavioral": True},
                    related_concepts=[],
                    evolution_history=[]
                ))
            
            self.extraction_stats["analytics_concepts"] += len(concepts)
            return concepts
            
        except Exception as e:
            self.logger.error(f"Analytics concept extraction failed: {e}")
            return []
    
    async def extract_ml_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
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
                    related_concepts=[],
                    evolution_history=[]
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
                    system_manifestations={
                        "ml": {"status": status, "module_count": len(status_modules)}
                    },
                    semantic_properties={"domain": "ml_orchestration", "operational": True},
                    related_concepts=[],
                    evolution_history=[]
                ))
            
            self.extraction_stats["ml_concepts"] += len(concepts)
            return concepts
            
        except Exception as e:
            self.logger.error(f"ML concept extraction failed: {e}")
            return []
    
    async def extract_api_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
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
                    performance_level = (
                        "high" if success_rate > 0.9 and response_time < 100
                        else "medium" if success_rate > 0.7
                        else "low"
                    )
                    
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
                        related_concepts=[],
                        evolution_history=[]
                    ))
            
            self.extraction_stats["api_concepts"] += len(concepts)
            return concepts
            
        except Exception as e:
            self.logger.error(f"API concept extraction failed: {e}")
            return []
    
    async def extract_pattern_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
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
                    related_concepts=[],
                    evolution_history=[]
                ))
            
            self.extraction_stats["pattern_concepts"] += len(concepts)
            return concepts
            
        except Exception as e:
            self.logger.error(f"Pattern concept extraction failed: {e}")
            return []
    
    async def create_abstract_concepts(self, system_data: Dict[str, Any]) -> List[SemanticConcept]:
        """Create higher-level abstract concepts"""
        concepts = []
        
        try:
            # System Integration Concept
            systems_active = sum(
                1 for system in ["analytics", "ml", "api", "patterns"]
                if system_data.get(system)
            )
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
                    semantic_properties={
                        "domain": "meta_system",
                        "integration": True,
                        "emergent": True
                    },
                    related_concepts=[],
                    evolution_history=[]
                ))
            
            # Intelligence Emergence Concept
            has_insights = len(system_data.get("analytics", {}).get("insights", [])) > 5
            has_patterns = len(system_data.get("patterns", {}).get("advanced_patterns", [])) > 3
            
            if has_insights and has_patterns:
                concepts.append(SemanticConcept(
                    concept_id=f"intelligence_emergence_{int(datetime.now().timestamp())}",
                    concept_name="Intelligence Emergence",
                    concept_type="behavior",
                    confidence=0.8,
                    abstraction_level=5,
                    system_manifestations={
                        "all": {"insights_generated": True, "patterns_discovered": True}
                    },
                    semantic_properties={
                        "domain": "meta_intelligence",
                        "emergent": True,
                        "adaptive": True
                    },
                    related_concepts=[],
                    evolution_history=[]
                ))
            
            self.extraction_stats["abstract_concepts"] += len(concepts)
            return concepts
            
        except Exception as e:
            self.logger.error(f"Abstract concept creation failed: {e}")
            return []
    
    def get_extraction_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return self.extraction_stats.copy()