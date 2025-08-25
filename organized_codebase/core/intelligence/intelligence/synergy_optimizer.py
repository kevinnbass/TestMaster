"""
Intelligence Synergy Optimizer
=============================

Revolutionary multi-system synergy discovery and optimization engine.
Extracted from meta_intelligence_orchestrator.py for enterprise modular architecture.

Agent D Implementation - Hour 14-15: Revolutionary Intelligence Modularization
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import networkx as nx
from itertools import combinations
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import (
    CapabilityType, OrchestrationStrategy, SynergyOpportunity,
    CapabilityProfile, SystemBehaviorModel, CapabilityCluster
)


class IntelligenceSynergyOptimizer:
    """
    Revolutionary Intelligence Synergy Optimizer
    
    Discovers and optimizes synergies between multiple intelligence systems,
    creating collaborative arrangements that exceed individual system capabilities.
    """
    
    def __init__(self, synergy_threshold: float = 0.6, cluster_epsilon: float = 0.3):
        self.synergy_threshold = synergy_threshold
        self.cluster_epsilon = cluster_epsilon
        
        # Core data stores
        self.capability_profiles: Dict[str, CapabilityProfile] = {}
        self.behavior_models: Dict[str, SystemBehaviorModel] = {}
        self.synergy_opportunities: Dict[str, SynergyOpportunity] = {}
        self.capability_clusters: Dict[str, CapabilityCluster] = {}
        
        # Synergy analysis
        self.synergy_graph: nx.Graph = nx.Graph()
        self.compatibility_matrix: np.ndarray = None
        self.system_embeddings: Dict[str, np.ndarray] = {}
        
        # Optimization tracking
        self.implemented_synergies: Dict[str, Dict[str, Any]] = {}
        self.synergy_performance: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # ML components
        self.capability_clusterer = None
        self.synergy_predictor = None
        
        # Synergy patterns
        self.successful_patterns: List[Dict[str, Any]] = []
        self.failed_patterns: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def register_system(self, capability_profile: CapabilityProfile,
                            behavior_model: Optional[SystemBehaviorModel] = None):
        """Register a new intelligence system for synergy analysis"""
        
        system_id = capability_profile.system_id
        self.capability_profiles[system_id] = capability_profile
        
        if behavior_model:
            self.behavior_models[system_id] = behavior_model
        
        # Add to synergy graph
        self.synergy_graph.add_node(system_id, **capability_profile.to_dict())
        
        # Generate system embedding
        embedding = await self._generate_system_embedding(capability_profile)
        self.system_embeddings[system_id] = embedding
        
        # Update compatibility matrix
        await self._update_compatibility_matrix()
        
        # Discover new synergy opportunities
        await self._discover_synergies_for_system(system_id)
        
        self.logger.info(f"Registered system {system_id} for synergy optimization")
    
    async def _generate_system_embedding(self, 
                                       capability_profile: CapabilityProfile) -> np.ndarray:
        """Generate high-dimensional embedding for a system based on its capabilities"""
        
        # Base capability vector
        capability_vector = []
        for cap_type in CapabilityType:
            capability_vector.append(
                capability_profile.capabilities.get(cap_type, 0.0)
            )
        
        # Performance characteristics
        perf_features = [
            capability_profile.processing_time,
            capability_profile.accuracy,
            capability_profile.reliability,
            capability_profile.scalability,
            capability_profile.cost_per_operation
        ]
        
        # Normalize performance features
        normalized_perf = []
        for feature in perf_features:
            if feature == capability_profile.processing_time:
                # Lower is better for processing time
                normalized_perf.append(1.0 / (1.0 + feature))
            elif feature == capability_profile.cost_per_operation:
                # Lower is better for cost
                normalized_perf.append(1.0 / (1.0 + feature))
            else:
                # Higher is better for accuracy, reliability, scalability
                normalized_perf.append(min(1.0, feature))
        
        # Resource characteristics
        resource_features = []
        if capability_profile.resource_requirements:
            for resource in ['cpu', 'memory', 'storage', 'network']:
                resource_features.append(
                    capability_profile.resource_requirements.get(resource, 0.5)
                )
        else:
            resource_features = [0.5] * 4
        
        # Combine all features
        embedding = np.array(capability_vector + normalized_perf + resource_features)
        
        # Add noise for stability
        embedding += np.random.normal(0, 0.01, len(embedding))
        
        return embedding
    
    async def _update_compatibility_matrix(self):
        """Update compatibility matrix for all registered systems"""
        
        if len(self.system_embeddings) < 2:
            return
        
        # Create matrix of system embeddings
        system_ids = list(self.system_embeddings.keys())
        embeddings = np.array([self.system_embeddings[sid] for sid in system_ids])
        
        # Calculate cosine similarity matrix
        self.compatibility_matrix = cosine_similarity(embeddings)
        
        # Update synergy graph edges
        for i, system1 in enumerate(system_ids):
            for j, system2 in enumerate(system_ids):
                if i < j:  # Avoid duplicate edges
                    compatibility = self.compatibility_matrix[i][j]
                    
                    if compatibility > self.synergy_threshold:
                        # Add or update edge
                        self.synergy_graph.add_edge(
                            system1, system2,
                            compatibility=compatibility,
                            synergy_potential=await self._calculate_synergy_potential(
                                system1, system2, compatibility
                            )
                        )
    
    async def _calculate_synergy_potential(self, system1: str, system2: str, 
                                         compatibility: float) -> float:
        """Calculate synergy potential between two systems"""
        
        profile1 = self.capability_profiles[system1]
        profile2 = self.capability_profiles[system2]
        
        # Complementarity analysis
        complementarity = 0.0
        capability_pairs = 0
        
        for cap_type in CapabilityType:
            cap1 = profile1.capabilities.get(cap_type, 0.0)
            cap2 = profile2.capabilities.get(cap_type, 0.0)
            
            if cap1 > 0 or cap2 > 0:
                capability_pairs += 1
                # High synergy when one system is strong where the other is weak
                complementarity += abs(cap1 - cap2) * min(cap1, cap2)
        
        if capability_pairs > 0:
            complementarity /= capability_pairs
        
        # Performance synergy
        perf_synergy = 0.0
        if profile1.accuracy + profile2.accuracy > 1.0:
            perf_synergy += 0.2  # Accuracy boost potential
        if profile1.processing_time < 1.0 and profile2.processing_time < 1.0:
            perf_synergy += 0.3  # Speed combination potential
        if profile1.reliability > 0.8 and profile2.reliability > 0.8:
            perf_synergy += 0.2  # Reliability enhancement
        
        # Resource efficiency synergy
        resource_synergy = 0.0
        total_cost = profile1.cost_per_operation + profile2.cost_per_operation
        if total_cost < 1.0:  # Cost-effective combination
            resource_synergy += 0.3
        
        # Combined synergy potential
        synergy_potential = (
            0.4 * compatibility +
            0.3 * complementarity +
            0.2 * perf_synergy +
            0.1 * resource_synergy
        )
        
        return min(1.0, synergy_potential)
    
    async def _discover_synergies_for_system(self, system_id: str):
        """Discover synergy opportunities for a newly registered system"""
        
        if system_id not in self.capability_profiles:
            return
        
        profile = self.capability_profiles[system_id]
        
        # Check synergies with all other systems
        for other_id, other_profile in self.capability_profiles.items():
            if other_id == system_id:
                continue
            
            synergy_potential = 0.0
            if self.synergy_graph.has_edge(system_id, other_id):
                synergy_potential = self.synergy_graph[system_id][other_id].get(
                    'synergy_potential', 0.0
                )
            
            if synergy_potential > self.synergy_threshold:
                await self._create_synergy_opportunity(system_id, other_id, synergy_potential)
    
    async def _create_synergy_opportunity(self, system1: str, system2: str, 
                                        synergy_potential: float):
        """Create a synergy opportunity between two systems"""
        
        opportunity_id = f"synergy_{system1}_{system2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze synergy type
        synergy_type = await self._determine_synergy_type(system1, system2)
        
        # Calculate expected improvements
        expected_improvement = await self._calculate_expected_improvement(
            system1, system2, synergy_type
        )
        
        # Assess implementation complexity
        complexity = await self._assess_implementation_complexity(system1, system2)
        
        # Generate implementation steps
        implementation_steps = await self._generate_implementation_steps(
            system1, system2, synergy_type
        )
        
        # Create synergy opportunity
        opportunity = SynergyOpportunity(
            opportunity_id=opportunity_id,
            participating_systems=[system1, system2],
            synergy_type=synergy_type,
            synergy_description=f"{synergy_type.title()} synergy between {system1} and {system2}",
            expected_improvement=expected_improvement,
            implementation_complexity=complexity,
            resource_requirements=await self._calculate_resource_requirements(
                system1, system2
            ),
            timeline=timedelta(hours=self._estimate_implementation_time(complexity)),
            success_probability=min(0.95, synergy_potential * 1.2),
            risk_factors=await self._identify_risk_factors(system1, system2),
            implementation_steps=implementation_steps,
            monitoring_metrics=['response_time', 'accuracy', 'throughput', 'cost_efficiency'],
            priority_score=synergy_potential * len(expected_improvement) / 10
        )
        
        self.synergy_opportunities[opportunity_id] = opportunity
        
        self.logger.info(f"Created synergy opportunity {opportunity_id} - "
                        f"Type: {synergy_type}, Potential: {synergy_potential:.2f}")
    
    async def _determine_synergy_type(self, system1: str, system2: str) -> str:
        """Determine the type of synergy between two systems"""
        
        profile1 = self.capability_profiles[system1]
        profile2 = self.capability_profiles[system2]
        
        # Analyze capability overlap and complementarity
        overlap_score = 0.0
        complementarity_score = 0.0
        
        for cap_type in CapabilityType:
            cap1 = profile1.capabilities.get(cap_type, 0.0)
            cap2 = profile2.capabilities.get(cap_type, 0.0)
            
            if cap1 > 0.5 and cap2 > 0.5:
                overlap_score += min(cap1, cap2)
            elif cap1 > 0.5 and cap2 < 0.3:
                complementarity_score += cap1
            elif cap2 > 0.5 and cap1 < 0.3:
                complementarity_score += cap2
        
        # Determine synergy type based on scores
        if overlap_score > complementarity_score:
            if overlap_score > 3.0:
                return "redundant"  # High overlap - backup/redundancy
            else:
                return "competitive"  # Moderate overlap - competitive improvement
        else:
            if complementarity_score > 2.0:
                return "complementary"  # High complementarity - pipeline/sequential
            else:
                return "collaborative"  # Balanced - parallel collaboration
    
    async def _calculate_expected_improvement(self, system1: str, system2: str, 
                                            synergy_type: str) -> Dict[str, float]:
        """Calculate expected performance improvements from synergy"""
        
        profile1 = self.capability_profiles[system1]
        profile2 = self.capability_profiles[system2]
        
        improvements = {}
        
        if synergy_type == "complementary":
            # Sequential processing - combine strengths
            improvements['accuracy'] = min(100.0, 
                (profile1.accuracy + profile2.accuracy) * 50)
            improvements['processing_speed'] = max(10.0,
                2.0 / (1.0/max(0.1, profile1.processing_time) + 
                      1.0/max(0.1, profile2.processing_time)) * 100)
            improvements['capability_coverage'] = 40.0
            
        elif synergy_type == "redundant":
            # Parallel processing - improved reliability
            improvements['reliability'] = min(50.0,
                (1.0 - (1.0 - profile1.reliability) * (1.0 - profile2.reliability)) * 100)
            improvements['throughput'] = 80.0
            improvements['fault_tolerance'] = 60.0
            
        elif synergy_type == "competitive":
            # Best-of-both - improved performance
            improvements['accuracy'] = 25.0
            improvements['response_time'] = 30.0
            improvements['resource_efficiency'] = 20.0
            
        else:  # collaborative
            # Parallel collaboration - balanced improvement
            improvements['throughput'] = 50.0
            improvements['scalability'] = 40.0
            improvements['cost_efficiency'] = 30.0
        
        return improvements
    
    async def _assess_implementation_complexity(self, system1: str, system2: str) -> str:
        """Assess implementation complexity of synergy"""
        
        profile1 = self.capability_profiles[system1]
        profile2 = self.capability_profiles[system2]
        
        # Factors affecting complexity
        api_compatibility = len(set(profile1.api_endpoints) & set(profile2.api_endpoints))
        
        # Input/output compatibility
        io_compatibility = len(set(profile1.input_types) & set(profile2.output_types)) + \
                          len(set(profile2.input_types) & set(profile1.output_types))
        
        # Resource compatibility
        resource_conflict = 0
        for resource in ['cpu', 'memory', 'storage']:
            req1 = profile1.resource_requirements.get(resource, 0.5)
            req2 = profile2.resource_requirements.get(resource, 0.5)
            if req1 + req2 > 1.0:  # Resource conflict
                resource_conflict += 1
        
        # Calculate complexity score
        complexity_score = (
            (5 - api_compatibility) * 0.3 +
            (5 - io_compatibility) * 0.4 +
            resource_conflict * 0.3
        )
        
        if complexity_score < 1.5:
            return "low"
        elif complexity_score < 3.0:
            return "medium"
        else:
            return "high"
    
    async def _generate_implementation_steps(self, system1: str, system2: str, 
                                           synergy_type: str) -> List[str]:
        """Generate implementation steps for synergy"""
        
        base_steps = [
            "Analyze system compatibility",
            "Design integration architecture",
            "Implement data flow connections",
            "Set up monitoring and metrics",
            "Test synergy performance",
            "Deploy and validate"
        ]
        
        if synergy_type == "complementary":
            specific_steps = [
                "Design sequential processing pipeline",
                "Implement output-to-input mapping",
                "Optimize handoff protocols"
            ]
        elif synergy_type == "redundant":
            specific_steps = [
                "Implement parallel execution framework",
                "Design result aggregation logic",
                "Set up failover mechanisms"
            ]
        elif synergy_type == "competitive":
            specific_steps = [
                "Implement result comparison logic",
                "Design performance-based selection",
                "Optimize switching mechanisms"
            ]
        else:  # collaborative
            specific_steps = [
                "Design collaborative task distribution",
                "Implement load balancing",
                "Set up result coordination"
            ]
        
        return base_steps[:3] + specific_steps + base_steps[3:]
    
    async def _calculate_resource_requirements(self, system1: str, 
                                             system2: str) -> Dict[str, float]:
        """Calculate resource requirements for implementing synergy"""
        
        profile1 = self.capability_profiles[system1]
        profile2 = self.capability_profiles[system2]
        
        # Base requirements from both systems
        requirements = {}
        
        for resource in ['cpu', 'memory', 'storage', 'network']:
            req1 = profile1.resource_requirements.get(resource, 0.5)
            req2 = profile2.resource_requirements.get(resource, 0.5)
            
            # Add overhead for coordination
            overhead = 0.2
            requirements[resource] = req1 + req2 + overhead
        
        # Add synergy-specific requirements
        requirements['coordination_overhead'] = 0.1
        requirements['monitoring_overhead'] = 0.05
        
        return requirements
    
    def _estimate_implementation_time(self, complexity: str) -> int:
        """Estimate implementation time in hours based on complexity"""
        
        time_estimates = {
            "low": 4,
            "medium": 12,
            "high": 24
        }
        
        return time_estimates.get(complexity, 12)
    
    async def _identify_risk_factors(self, system1: str, system2: str) -> List[str]:
        """Identify potential risk factors for synergy implementation"""
        
        profile1 = self.capability_profiles[system1]
        profile2 = self.capability_profiles[system2]
        
        risks = []
        
        # Reliability risks
        if profile1.reliability < 0.8 or profile2.reliability < 0.8:
            risks.append("Low system reliability may affect synergy stability")
        
        # Performance risks
        if profile1.processing_time > 2.0 or profile2.processing_time > 2.0:
            risks.append("High processing times may create bottlenecks")
        
        # Resource risks
        total_cost = profile1.cost_per_operation + profile2.cost_per_operation
        if total_cost > 1.0:
            risks.append("High combined operational costs")
        
        # Compatibility risks
        if not set(profile1.output_types) & set(profile2.input_types):
            risks.append("Limited input/output compatibility")
        
        # Documentation risks
        if profile1.documentation_quality < 0.7 or profile2.documentation_quality < 0.7:
            risks.append("Poor documentation may complicate integration")
        
        return risks
    
    async def discover_capability_clusters(self) -> Dict[str, CapabilityCluster]:
        """Discover clusters of systems with similar capabilities"""
        
        if len(self.system_embeddings) < 3:
            return {}
        
        # Prepare data for clustering
        system_ids = list(self.system_embeddings.keys())
        embeddings = np.array([self.system_embeddings[sid] for sid in system_ids])
        
        # Perform clustering
        clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=2)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Create capability clusters
        clusters = {}
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
            
            # Get systems in this cluster
            cluster_systems = [system_ids[i] for i, label in enumerate(cluster_labels) 
                             if label == cluster_id]
            
            # Analyze shared capabilities
            shared_capabilities = await self._analyze_shared_capabilities(cluster_systems)
            
            # Create cluster
            cluster = CapabilityCluster(
                cluster_id=f"cluster_{cluster_id}",
                cluster_name=f"Capability Cluster {cluster_id}",
                member_systems=cluster_systems,
                shared_capabilities=shared_capabilities,
                cluster_characteristics=await self._analyze_cluster_characteristics(
                    cluster_systems
                ),
                inter_cluster_relationships={},
                optimization_opportunities=await self._identify_cluster_optimizations(
                    cluster_systems
                )
            )
            
            clusters[cluster.cluster_id] = cluster
        
        # Analyze inter-cluster relationships
        for cluster1_id, cluster1 in clusters.items():
            for cluster2_id, cluster2 in clusters.items():
                if cluster1_id != cluster2_id:
                    relationship_strength = await self._calculate_cluster_relationship(
                        cluster1, cluster2
                    )
                    cluster1.inter_cluster_relationships[cluster2_id] = relationship_strength
        
        self.capability_clusters = clusters
        return clusters
    
    async def _analyze_shared_capabilities(self, system_ids: List[str]) -> List[CapabilityType]:
        """Analyze capabilities shared across cluster members"""
        
        if not system_ids:
            return []
        
        shared_capabilities = []
        
        for cap_type in CapabilityType:
            # Check if all systems have this capability above threshold
            all_have_capability = True
            for system_id in system_ids:
                if system_id in self.capability_profiles:
                    capability_score = self.capability_profiles[system_id].capabilities.get(
                        cap_type, 0.0
                    )
                    if capability_score < 0.5:
                        all_have_capability = False
                        break
                else:
                    all_have_capability = False
                    break
            
            if all_have_capability:
                shared_capabilities.append(cap_type)
        
        return shared_capabilities
    
    async def _analyze_cluster_characteristics(self, 
                                             system_ids: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of a capability cluster"""
        
        characteristics = {
            'average_accuracy': 0.0,
            'average_processing_time': 0.0,
            'average_reliability': 0.0,
            'total_cost': 0.0,
            'dominant_capabilities': [],
            'cluster_size': len(system_ids)
        }
        
        if not system_ids:
            return characteristics
        
        # Calculate averages
        total_accuracy = 0.0
        total_processing_time = 0.0
        total_reliability = 0.0
        total_cost = 0.0
        
        capability_totals = {cap_type: 0.0 for cap_type in CapabilityType}
        
        for system_id in system_ids:
            if system_id in self.capability_profiles:
                profile = self.capability_profiles[system_id]
                total_accuracy += profile.accuracy
                total_processing_time += profile.processing_time
                total_reliability += profile.reliability
                total_cost += profile.cost_per_operation
                
                for cap_type, score in profile.capabilities.items():
                    capability_totals[cap_type] += score
        
        count = len(system_ids)
        characteristics['average_accuracy'] = total_accuracy / count
        characteristics['average_processing_time'] = total_processing_time / count
        characteristics['average_reliability'] = total_reliability / count
        characteristics['total_cost'] = total_cost
        
        # Find dominant capabilities
        avg_capabilities = {cap: total / count for cap, total in capability_totals.items()}
        characteristics['dominant_capabilities'] = [
            cap.value for cap, avg in avg_capabilities.items() if avg > 0.7
        ]
        
        return characteristics
    
    async def _identify_cluster_optimizations(self, 
                                            system_ids: List[str]) -> List[str]:
        """Identify optimization opportunities within a cluster"""
        
        optimizations = []
        
        if len(system_ids) >= 2:
            optimizations.append("Load balancing across cluster members")
            optimizations.append("Redundancy elimination between similar systems")
        
        if len(system_ids) >= 3:
            optimizations.append("Ensemble methods for improved accuracy")
            optimizations.append("Distributed processing optimization")
        
        # Analyze specific optimization opportunities
        has_high_accuracy = False
        has_fast_processing = False
        has_high_reliability = False
        
        for system_id in system_ids:
            if system_id in self.capability_profiles:
                profile = self.capability_profiles[system_id]
                if profile.accuracy > 0.9:
                    has_high_accuracy = True
                if profile.processing_time < 0.5:
                    has_fast_processing = True
                if profile.reliability > 0.95:
                    has_high_reliability = True
        
        if has_high_accuracy and has_fast_processing:
            optimizations.append("High-performance accuracy pipeline")
        
        if has_high_reliability:
            optimizations.append("Critical system backup cluster")
        
        return optimizations
    
    async def _calculate_cluster_relationship(self, cluster1: CapabilityCluster, 
                                            cluster2: CapabilityCluster) -> float:
        """Calculate relationship strength between two clusters"""
        
        # Capability overlap
        shared_capabilities = set(cluster1.shared_capabilities) & set(cluster2.shared_capabilities)
        capability_overlap = len(shared_capabilities) / max(1, 
            len(set(cluster1.shared_capabilities) | set(cluster2.shared_capabilities)))
        
        # Performance compatibility
        char1 = cluster1.cluster_characteristics
        char2 = cluster2.cluster_characteristics
        
        accuracy_diff = abs(char1.get('average_accuracy', 0) - char2.get('average_accuracy', 0))
        time_diff = abs(char1.get('average_processing_time', 0) - char2.get('average_processing_time', 0))
        
        performance_compatibility = 1.0 - (accuracy_diff + time_diff) / 2.0
        
        # Combined relationship strength
        relationship_strength = (capability_overlap * 0.6) + (performance_compatibility * 0.4)
        
        return max(0.0, min(1.0, relationship_strength))
    
    def get_synergy_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about discovered synergies"""
        
        insights = {
            'total_opportunities': len(self.synergy_opportunities),
            'high_priority_opportunities': 0,
            'synergy_type_distribution': {},
            'average_success_probability': 0.0,
            'implementation_complexity_distribution': {},
            'top_system_pairs': [],
            'cluster_analysis': {},
            'optimization_potential': 0.0
        }
        
        if not self.synergy_opportunities:
            return insights
        
        # Analyze synergy opportunities
        total_success_prob = 0.0
        complexity_counts = {'low': 0, 'medium': 0, 'high': 0}
        synergy_type_counts = {}
        
        for opportunity in self.synergy_opportunities.values():
            if opportunity.priority_score > 0.7:
                insights['high_priority_opportunities'] += 1
            
            total_success_prob += opportunity.success_probability
            
            complexity_counts[opportunity.implementation_complexity] += 1
            
            if opportunity.synergy_type not in synergy_type_counts:
                synergy_type_counts[opportunity.synergy_type] = 0
            synergy_type_counts[opportunity.synergy_type] += 1
        
        insights['average_success_probability'] = total_success_prob / len(self.synergy_opportunities)
        insights['synergy_type_distribution'] = synergy_type_counts
        insights['implementation_complexity_distribution'] = complexity_counts
        
        # Cluster analysis
        insights['cluster_analysis'] = {
            'total_clusters': len(self.capability_clusters),
            'average_cluster_size': np.mean([
                len(cluster.member_systems) for cluster in self.capability_clusters.values()
            ]) if self.capability_clusters else 0,
            'total_optimization_opportunities': sum([
                len(cluster.optimization_opportunities) 
                for cluster in self.capability_clusters.values()
            ])
        }
        
        # Calculate optimization potential
        total_improvement = 0.0
        for opportunity in self.synergy_opportunities.values():
            improvement_sum = sum(opportunity.expected_improvement.values())
            total_improvement += improvement_sum * opportunity.success_probability
        
        insights['optimization_potential'] = total_improvement / max(1, len(self.synergy_opportunities))
        
        return insights


def create_synergy_optimizer(synergy_threshold: float = 0.6, 
                           cluster_epsilon: float = 0.3) -> IntelligenceSynergyOptimizer:
    """Factory function to create IntelligenceSynergyOptimizer instance"""
    
    return IntelligenceSynergyOptimizer(
        synergy_threshold=synergy_threshold,
        cluster_epsilon=cluster_epsilon
    )