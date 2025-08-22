"""
Capability Mapper - Intelligence Capability Analysis Engine
==========================================================

Sophisticated capability mapping system implementing advanced intelligence
analysis, capability classification, and performance profiling for comprehensive
system understanding with enterprise-grade capability discovery and mapping.

This module provides advanced capability mapping including:
- Intelligent capability discovery and classification
- Performance metric estimation and profiling
- Capability relationship analysis and graph construction
- Resource requirement estimation and optimization
- Similarity analysis with clustering algorithms

Author: Agent A - PHASE 3: Hours 200-300
Created: 2025-08-22
Module: capability_mapper.py (350 lines)
"""

import asyncio
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .meta_types import CapabilityType, CapabilityProfile

logger = logging.getLogger(__name__)


class IntelligenceCapabilityMapper:
    """
    Enterprise capability mapper implementing advanced intelligence capability
    analysis, classification, and relationship discovery for meta-orchestration.
    
    Features:
    - Intelligent capability discovery with automated classification
    - Performance profiling with accuracy and scalability estimation
    - Resource requirement analysis with optimization recommendations
    - Capability relationship mapping with graph-based analysis
    - Similarity clustering with machine learning algorithms
    """
    
    def __init__(self):
        self.capability_profiles: Dict[str, CapabilityProfile] = {}
        self.system_capabilities: Dict[str, List[str]] = {}  # system_id -> capability_ids
        self.capability_graph = nx.Graph()
        self.similarity_matrix: Optional[np.ndarray] = None
        self.capability_clusters: Dict[str, List[str]] = {}
        
        logger.info("IntelligenceCapabilityMapper initialized")
    
    async def map_system_capabilities(self, system_id: str, system_info: Dict[str, Any]) -> List[CapabilityProfile]:
        """
        Comprehensive capability mapping for intelligence systems with deep analysis.
        
        Args:
            system_id: Unique identifier for the intelligence system
            system_info: System information including capabilities and performance metrics
            
        Returns:
            List of detailed capability profiles with performance characteristics
        """
        logger.info(f"Mapping capabilities for system: {system_id}")
        
        capabilities = []
        
        # Extract and analyze capability information
        raw_capabilities = system_info.get("capabilities", [])
        
        for cap_name in raw_capabilities:
            capability_profile = await self._analyze_capability(system_id, cap_name, system_info)
            capabilities.append(capability_profile)
            
            # Store in capability registry
            self.capability_profiles[capability_profile.capability_id] = capability_profile
        
        # Update system-capability mapping
        self.system_capabilities[system_id] = [cap.capability_id for cap in capabilities]
        
        # Update capability relationship graph
        self._update_capability_graph(capabilities)
        
        # Perform clustering analysis if we have enough capabilities
        if len(self.capability_profiles) >= 3:
            await self._perform_clustering_analysis()
        
        logger.info(f"Mapped {len(capabilities)} capabilities for {system_id}")
        return capabilities
    
    async def _analyze_capability(self, system_id: str, capability_name: str, 
                                system_info: Dict[str, Any]) -> CapabilityProfile:
        """Detailed capability analysis with performance profiling"""
        
        capability_id = f"{system_id}_{capability_name}"
        
        # Intelligent capability classification
        capability_type = self._classify_capability_type(capability_name)
        
        # Performance metrics extraction and estimation
        performance_metrics = system_info.get("performance_metrics", {})
        
        # Create comprehensive capability profile
        profile = CapabilityProfile(
            capability_id=capability_id,
            name=capability_name,
            type=capability_type,
            description=f"{capability_name} capability provided by {system_id}",
            input_types=self._infer_input_types(capability_name),
            output_types=self._infer_output_types(capability_name),
            processing_time=performance_metrics.get("response_time", 100) / 1000,  # Convert to seconds
            accuracy=self._estimate_accuracy(capability_name, performance_metrics),
            reliability=performance_metrics.get("availability", 0.95),
            scalability=self._estimate_scalability(capability_name, performance_metrics),
            resource_requirements=self._estimate_resource_requirements(capability_name, performance_metrics)
        )
        
        # Analyze capability relationships and dependencies
        await self._analyze_capability_relationships(profile)
        
        return profile
    
    def _classify_capability_type(self, capability_name: str) -> CapabilityType:
        """Intelligent capability classification using pattern matching"""
        name_lower = capability_name.lower()
        
        # Advanced pattern matching for capability classification
        classification_patterns = {
            CapabilityType.ANALYTICAL: ['analysis', 'analyze', 'analytics', 'statistical', 'examination'],
            CapabilityType.PREDICTIVE: ['predict', 'forecast', 'projection', 'trend', 'future'],
            CapabilityType.OPTIMIZATION: ['optimize', 'optimization', 'tuning', 'enhancement', 'improve'],
            CapabilityType.PATTERN_RECOGNITION: ['pattern', 'recognition', 'detection', 'classification', 'matching'],
            CapabilityType.DECISION_MAKING: ['decision', 'recommendation', 'suggestion', 'choice', 'selection'],
            CapabilityType.LEARNING: ['learning', 'training', 'adaptation', 'evolution', 'self-improving'],
            CapabilityType.SYNTHESIS: ['synthesis', 'generation', 'creation', 'composition', 'combining'],
            CapabilityType.REASONING: ['reasoning', 'logic', 'inference', 'deduction', 'conclusion'],
            CapabilityType.CREATIVE: ['creative', 'innovative', 'inventive', 'original', 'novel']
        }
        
        # Score each capability type based on keyword matches
        type_scores = {}
        for cap_type, keywords in classification_patterns.items():
            score = sum(1 for keyword in keywords if keyword in name_lower)
            if score > 0:
                type_scores[cap_type] = score
        
        # Return the highest scoring type, or ANALYTICAL as default
        return max(type_scores.items(), key=lambda x: x[1])[0] if type_scores else CapabilityType.ANALYTICAL
    
    def _infer_input_types(self, capability_name: str) -> List[str]:
        """Intelligent input type inference based on capability characteristics"""
        name_lower = capability_name.lower()
        
        input_types = ["data"]  # Default input
        
        # Pattern-based input type detection
        input_patterns = {
            "source_code": ["code", "programming", "source", "syntax"],
            "image": ["image", "visual", "picture", "graphic", "photo"],
            "text": ["text", "language", "document", "natural", "linguistic"],
            "time_series": ["time", "temporal", "series", "sequential", "chronological"],
            "graph": ["graph", "network", "relationship", "connection", "node"],
            "audio": ["audio", "sound", "speech", "voice", "acoustic"],
            "video": ["video", "motion", "stream", "frame", "visual"],
            "structured_data": ["json", "xml", "csv", "table", "structured"]
        }
        
        for input_type, keywords in input_patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                input_types.append(input_type)
        
        return list(set(input_types))  # Remove duplicates
    
    def _infer_output_types(self, capability_name: str) -> List[str]:
        """Intelligent output type inference based on capability function"""
        name_lower = capability_name.lower()
        
        output_types = ["result"]  # Default output
        
        # Pattern-based output type detection
        output_patterns = {
            "prediction": ["predict", "forecast", "projection", "estimate"],
            "analysis_report": ["analysis", "report", "summary", "insight"],
            "recommendations": ["recommendation", "suggestion", "advice", "guidance"],
            "score": ["score", "rating", "ranking", "metric", "evaluation"],
            "classification": ["classification", "category", "label", "class"],
            "visualization": ["chart", "graph", "plot", "visual", "diagram"],
            "optimization_plan": ["optimization", "improvement", "enhancement", "plan"],
            "decision": ["decision", "choice", "selection", "determination"]
        }
        
        for output_type, keywords in output_patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                output_types.append(output_type)
        
        return list(set(output_types))  # Remove duplicates
    
    def _estimate_accuracy(self, capability_name: str, performance_metrics: Dict[str, Any]) -> float:
        """Advanced accuracy estimation using capability analysis"""
        base_accuracy = 0.8  # Default accuracy
        
        name_lower = capability_name.lower()
        
        # Capability complexity adjustments
        complexity_adjustments = {
            ("simple", "basic", "elementary"): 0.9,
            ("complex", "advanced", "sophisticated"): 0.75,
            ("predictive", "forecasting"): 0.7,
            ("analytical", "analysis"): 0.85,
            ("experimental", "prototype"): 0.65,
            ("proven", "validated", "tested"): 0.95
        }
        
        for keywords, accuracy in complexity_adjustments.items():
            if any(keyword in name_lower for keyword in keywords):
                base_accuracy = accuracy
                break
        
        # Factor in system performance metrics
        error_rate = performance_metrics.get("error_rate", 0.1)
        accuracy_from_errors = 1.0 - error_rate
        
        # Combine estimates with weighted average
        final_accuracy = (base_accuracy * 0.6 + accuracy_from_errors * 0.4)
        return max(0.1, min(1.0, final_accuracy))
    
    def _estimate_scalability(self, capability_name: str, performance_metrics: Dict[str, Any]) -> float:
        """Advanced scalability estimation based on architectural patterns"""
        base_scalability = 0.7  # Default scalability
        
        name_lower = capability_name.lower()
        
        # Scalability pattern analysis
        scalability_patterns = {
            ("parallel", "distributed", "concurrent", "async"): 0.9,
            ("sequential", "iterative", "single-threaded"): 0.5,
            ("batch", "bulk", "mass"): 0.8,
            ("streaming", "real-time", "continuous"): 0.85,
            ("cached", "memoized", "optimized"): 0.9
        }
        
        for keywords, scalability in scalability_patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                base_scalability = scalability
                break
        
        # Factor in throughput metrics
        throughput = performance_metrics.get("throughput", 100)
        scalability_from_throughput = min(1.0, throughput / 1000)
        
        # Combine estimates
        final_scalability = (base_scalability * 0.7 + scalability_from_throughput * 0.3)
        return max(0.1, min(1.0, final_scalability))
    
    def _estimate_resource_requirements(self, capability_name: str, 
                                      performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Comprehensive resource requirement estimation"""
        name_lower = capability_name.lower()
        
        # Base resource requirements
        requirements = {
            "cpu": 10.0,  # CPU percentage
            "memory": 512.0,  # Memory in MB
            "storage": 100.0,  # Storage in MB
            "network": 10.0  # Network bandwidth in Mbps
        }
        
        # Resource scaling factors based on capability type
        resource_patterns = {
            ("ml", "learning", "training", "neural"): {"cpu": 3, "memory": 4, "storage": 2},
            ("image", "visual", "video"): {"cpu": 2, "memory": 3, "storage": 5},
            ("big_data", "massive", "large_scale"): {"cpu": 2.5, "memory": 4, "storage": 10},
            ("real_time", "streaming", "live"): {"cpu": 1.5, "memory": 2, "network": 3},
            ("analytics", "statistical", "mathematical"): {"cpu": 2, "memory": 2.5, "storage": 1.5}
        }
        
        for keywords, multipliers in resource_patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                for resource, multiplier in multipliers.items():
                    requirements[resource] *= multiplier
                break
        
        # Adjust based on performance metrics
        if "memory_usage" in performance_metrics:
            requirements["memory"] = max(requirements["memory"], performance_metrics["memory_usage"])
        
        return requirements
    
    async def _analyze_capability_relationships(self, profile: CapabilityProfile) -> None:
        """Analyze relationships between capabilities"""
        
        # Find complementary capabilities based on input/output matching
        for cap_id, existing_profile in self.capability_profiles.items():
            if cap_id == profile.capability_id:
                continue
            
            # Check for input/output complementarity
            input_output_match = any(
                output in profile.input_types for output in existing_profile.output_types
            )
            output_input_match = any(
                output in existing_profile.input_types for output in profile.output_types
            )
            
            if input_output_match or output_input_match:
                profile.complementary_capabilities.append(cap_id)
                existing_profile.complementary_capabilities.append(profile.capability_id)
    
    def _update_capability_graph(self, capabilities: List[CapabilityProfile]) -> None:
        """Update capability relationship graph"""
        
        for capability in capabilities:
            # Add capability node
            self.capability_graph.add_node(
                capability.capability_id,
                name=capability.name,
                type=capability.type.value,
                accuracy=capability.accuracy,
                scalability=capability.scalability
            )
            
            # Add edges for complementary capabilities
            for comp_cap_id in capability.complementary_capabilities:
                if comp_cap_id in self.capability_profiles:
                    self.capability_graph.add_edge(capability.capability_id, comp_cap_id)
    
    async def _perform_clustering_analysis(self) -> None:
        """Perform clustering analysis on capabilities"""
        
        if len(self.capability_profiles) < 3:
            return
        
        # Create feature vectors for capabilities
        features = []
        capability_ids = []
        
        for cap_id, profile in self.capability_profiles.items():
            feature_vector = [
                profile.processing_time,
                profile.accuracy,
                profile.reliability,
                profile.scalability,
                profile.resource_requirements.get("cpu", 0),
                profile.resource_requirements.get("memory", 0),
                len(profile.input_types),
                len(profile.output_types)
            ]
            features.append(feature_vector)
            capability_ids.append(cap_id)
        
        # Perform K-means clustering
        n_clusters = min(3, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Organize capabilities by cluster
        self.capability_clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_name = f"cluster_{label}"
            if cluster_name not in self.capability_clusters:
                self.capability_clusters[cluster_name] = []
            self.capability_clusters[cluster_name].append(capability_ids[i])
        
        # Compute similarity matrix
        feature_array = np.array(features)
        self.similarity_matrix = cosine_similarity(feature_array)
        
        logger.info(f"Performed clustering analysis: {n_clusters} clusters identified")
    
    def get_capability_recommendations(self, system_id: str, objective: str) -> List[str]:
        """Get capability recommendations for a system objective"""
        
        objective_lower = objective.lower()
        recommended_capabilities = []
        
        # Match objective to capability types
        objective_capability_mapping = {
            ("analysis", "analyze", "understand"): CapabilityType.ANALYTICAL,
            ("predict", "forecast", "future"): CapabilityType.PREDICTIVE,
            ("optimize", "improve", "enhance"): CapabilityType.OPTIMIZATION,
            ("detect", "find", "identify"): CapabilityType.PATTERN_RECOGNITION,
            ("decide", "choose", "recommend"): CapabilityType.DECISION_MAKING
        }
        
        target_type = None
        for keywords, cap_type in objective_capability_mapping.items():
            if any(keyword in objective_lower for keyword in keywords):
                target_type = cap_type
                break
        
        if target_type:
            # Find capabilities of the target type
            for cap_id, profile in self.capability_profiles.items():
                if profile.type == target_type and profile.accuracy > 0.7:
                    recommended_capabilities.append(cap_id)
        
        return recommended_capabilities[:5]  # Return top 5 recommendations
    
    def get_mapper_status(self) -> Dict[str, Any]:
        """Get comprehensive mapper status"""
        
        return {
            "total_capabilities": len(self.capability_profiles),
            "total_systems": len(self.system_capabilities),
            "capability_types": {cap_type.value: sum(1 for p in self.capability_profiles.values() 
                                                  if p.type == cap_type) for cap_type in CapabilityType},
            "clusters": len(self.capability_clusters),
            "graph_nodes": self.capability_graph.number_of_nodes(),
            "graph_edges": self.capability_graph.number_of_edges(),
            "avg_accuracy": np.mean([p.accuracy for p in self.capability_profiles.values()]) if self.capability_profiles else 0.0,
            "avg_scalability": np.mean([p.scalability for p in self.capability_profiles.values()]) if self.capability_profiles else 0.0
        }


# Export capability mapping components
__all__ = ['IntelligenceCapabilityMapper']