"""
Pattern Intelligence Microservice Analyzer
==========================================

Microservice boundary analysis and migration strategies with coupling detection.
Extracted from architectural_decision_engine.py for enterprise modular architecture.

Agent D Implementation - Hour 13-14: Revolutionary Intelligence Modularization
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Network analysis imports  
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("NetworkX not available. Microservice analysis will be limited.")

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import MicroserviceBoundary, ArchitecturalPattern


@dataclass
class ServiceBoundaryHeuristic:
    """Heuristic for determining service boundaries"""
    name: str
    description: str
    weight: float
    evaluator: callable


@dataclass
class CouplingAnalysis:
    """Analysis of coupling between components"""
    component_a: str
    component_b: str
    coupling_type: str  # data, temporal, functional, logical
    coupling_strength: float  # 0-1
    coupling_reasons: List[str]
    decoupling_suggestions: List[str]


@dataclass
class MigrationPhase:
    """Phase in microservice migration"""
    phase_number: int
    phase_name: str
    services_to_extract: List[str]
    dependencies: List[str]
    estimated_effort_days: float
    risk_level: str
    success_criteria: List[str]
    rollback_strategy: List[str]


class MicroserviceEvolutionAnalyzer:
    """Advanced microservice boundary analysis and migration planning"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Service boundary heuristics
        self.boundary_heuristics = self._initialize_boundary_heuristics()
        
        # Coupling detection patterns
        self.coupling_patterns = self._initialize_coupling_patterns()
        
        # Migration strategies
        self.migration_strategies = self._initialize_migration_strategies()
    
    def _initialize_boundary_heuristics(self) -> List[ServiceBoundaryHeuristic]:
        """Initialize service boundary detection heuristics"""
        return [
            ServiceBoundaryHeuristic(
                name="Business Capability Cohesion",
                description="Groups components by business capability",
                weight=0.25,
                evaluator=self._evaluate_business_capability_cohesion
            ),
            
            ServiceBoundaryHeuristic(
                name="Data Cohesion",
                description="Groups components by data entity relationships",
                weight=0.20,
                evaluator=self._evaluate_data_cohesion
            ),
            
            ServiceBoundaryHeuristic(
                name="Change Frequency Coupling",
                description="Groups components that change together",
                weight=0.15,
                evaluator=self._evaluate_change_frequency_coupling
            ),
            
            ServiceBoundaryHeuristic(
                name="Team Ownership Alignment",
                description="Aligns services with team boundaries",
                weight=0.10,
                evaluator=self._evaluate_team_ownership_alignment
            ),
            
            ServiceBoundaryHeuristic(
                name="Performance Requirements",
                description="Groups by similar performance characteristics",
                weight=0.10,
                evaluator=self._evaluate_performance_requirements
            ),
            
            ServiceBoundaryHeuristic(
                name="Security Boundary Alignment",
                description="Groups by security and compliance requirements",
                weight=0.08,
                evaluator=self._evaluate_security_boundary_alignment
            ),
            
            ServiceBoundaryHeuristic(
                name="Deployment Independence",
                description="Evaluates potential for independent deployment",
                weight=0.07,
                evaluator=self._evaluate_deployment_independence
            ),
            
            ServiceBoundaryHeuristic(
                name="Technology Affinity",
                description="Groups components using similar technologies",
                weight=0.05,
                evaluator=self._evaluate_technology_affinity
            )
        ]
    
    def _initialize_coupling_patterns(self) -> Dict[str, Dict]:
        """Initialize coupling detection patterns"""
        return {
            'data_coupling': {
                'indicators': ['shared_database', 'shared_data_structure', 'data_synchronization'],
                'severity_multiplier': 1.5,
                'description': 'Components share data structures or databases'
            },
            
            'temporal_coupling': {
                'indicators': ['synchronous_calls', 'blocking_operations', 'sequential_processing'],
                'severity_multiplier': 1.3,
                'description': 'Components must execute in specific temporal order'
            },
            
            'functional_coupling': {
                'indicators': ['shared_functions', 'utility_dependencies', 'common_algorithms'],
                'severity_multiplier': 1.0,
                'description': 'Components share functional implementations'
            },
            
            'logical_coupling': {
                'indicators': ['business_rule_sharing', 'workflow_dependencies', 'process_coupling'],
                'severity_multiplier': 1.2,
                'description': 'Components are logically related in business processes'
            },
            
            'performance_coupling': {
                'indicators': ['shared_resources', 'resource_contention', 'performance_dependencies'],
                'severity_multiplier': 1.1,
                'description': 'Components compete for or share performance-critical resources'
            }
        }
    
    def _initialize_migration_strategies(self) -> Dict[str, Dict]:
        """Initialize migration strategies"""
        return {
            'strangler_fig': {
                'description': 'Gradually replace monolith functionality with microservices',
                'suitable_for': ['large_monoliths', 'high_risk_tolerance_low'],
                'phases': ['identify_boundaries', 'implement_proxy', 'migrate_incrementally', 'retire_old'],
                'risk_level': 'low',
                'timeline_multiplier': 1.5
            },
            
            'database_per_service': {
                'description': 'Extract data layer for each service',
                'suitable_for': ['data_intensive_applications', 'strong_consistency_requirements'],
                'phases': ['analyze_data_dependencies', 'design_data_apis', 'implement_data_sync', 'migrate_data'],
                'risk_level': 'high',
                'timeline_multiplier': 2.0
            },
            
            'event_driven_decomposition': {
                'description': 'Use events to decouple services during migration',
                'suitable_for': ['async_processing', 'event_sourcing_ready'],
                'phases': ['design_events', 'implement_event_bus', 'migrate_to_events', 'optimize_flows'],
                'risk_level': 'medium',
                'timeline_multiplier': 1.3
            },
            
            'api_gateway_facade': {
                'description': 'Use API gateway to manage service evolution',
                'suitable_for': ['client_stability_required', 'gradual_migration'],
                'phases': ['implement_gateway', 'route_traffic', 'migrate_services', 'optimize_routing'],
                'risk_level': 'low',
                'timeline_multiplier': 1.2
            }
        }
    
    def analyze_service_boundaries(self, components: List[Dict], 
                                 component_relationships: List[Dict]) -> List[MicroserviceBoundary]:
        """Analyze and recommend microservice boundaries"""
        try:
            self.logger.info(f"Analyzing service boundaries for {len(components)} components")
            
            # Build component relationship graph
            relationship_graph = self._build_relationship_graph(components, component_relationships)
            
            # Apply boundary heuristics
            boundary_scores = self._apply_boundary_heuristics(components, relationship_graph)
            
            # Detect communities/clusters
            service_clusters = self._detect_service_clusters(relationship_graph, boundary_scores)
            
            # Create service boundary recommendations
            boundaries = []
            for i, cluster in enumerate(service_clusters):
                boundary = self._create_service_boundary(
                    f"service_{i+1}",
                    cluster,
                    components,
                    relationship_graph
                )
                boundaries.append(boundary)
            
            # Evaluate and refine boundaries
            refined_boundaries = self._refine_service_boundaries(boundaries, relationship_graph)
            
            self.logger.info(f"Recommended {len(refined_boundaries)} service boundaries")
            return refined_boundaries
            
        except Exception as e:
            self.logger.error(f"Error analyzing service boundaries: {e}")
            return []
    
    def _build_relationship_graph(self, components: List[Dict], 
                                relationships: List[Dict]) -> 'nx.Graph':
        """Build component relationship graph"""
        if not HAS_NETWORKX:
            return None
        
        graph = nx.Graph()
        
        # Add components as nodes
        for component in components:
            graph.add_node(
                component['name'],
                **{k: v for k, v in component.items() if k != 'name'}
            )
        
        # Add relationships as edges
        for rel in relationships:
            if 'source' in rel and 'target' in rel:
                weight = rel.get('strength', 1.0)
                relationship_type = rel.get('type', 'unknown')
                
                graph.add_edge(
                    rel['source'],
                    rel['target'],
                    weight=weight,
                    relationship_type=relationship_type,
                    **{k: v for k, v in rel.items() if k not in ['source', 'target']}
                )
        
        return graph
    
    def _apply_boundary_heuristics(self, components: List[Dict], 
                                 relationship_graph: 'nx.Graph') -> Dict[str, float]:
        """Apply boundary detection heuristics"""
        boundary_scores = defaultdict(float)
        
        try:
            for heuristic in self.boundary_heuristics:
                self.logger.debug(f"Applying heuristic: {heuristic.name}")
                
                scores = heuristic.evaluator(components, relationship_graph)
                
                # Apply weight and accumulate scores
                for component_pair, score in scores.items():
                    boundary_scores[component_pair] += score * heuristic.weight
            
            return dict(boundary_scores)
            
        except Exception as e:
            self.logger.error(f"Error applying boundary heuristics: {e}")
            return {}
    
    def _evaluate_business_capability_cohesion(self, components: List[Dict], 
                                             graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate business capability cohesion"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    # Extract business capabilities
                    cap_a = set(comp_a.get('business_capabilities', []))
                    cap_b = set(comp_b.get('business_capabilities', []))
                    
                    if cap_a and cap_b:
                        # Calculate Jaccard similarity
                        intersection = len(cap_a & cap_b)
                        union = len(cap_a | cap_b)
                        
                        if union > 0:
                            similarity = intersection / union
                            scores[(comp_a['name'], comp_b['name'])] = similarity
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating business capability cohesion: {e}")
            return {}
    
    def _evaluate_data_cohesion(self, components: List[Dict], 
                              graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate data entity cohesion"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    # Extract data entities
                    entities_a = set(comp_a.get('data_entities', []))
                    entities_b = set(comp_b.get('data_entities', []))
                    
                    if entities_a and entities_b:
                        # Calculate overlap
                        overlap = len(entities_a & entities_b)
                        total = len(entities_a | entities_b)
                        
                        if total > 0:
                            cohesion = overlap / total
                            scores[(comp_a['name'], comp_b['name'])] = cohesion
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating data cohesion: {e}")
            return {}
    
    def _evaluate_change_frequency_coupling(self, components: List[Dict], 
                                          graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate change frequency coupling"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    # Extract change history (if available)
                    changes_a = comp_a.get('change_history', [])
                    changes_b = comp_b.get('change_history', [])
                    
                    if changes_a and changes_b:
                        # Find overlapping change periods
                        common_changes = len(set(changes_a) & set(changes_b))
                        total_changes = len(set(changes_a) | set(changes_b))
                        
                        if total_changes > 0:
                            coupling = common_changes / total_changes
                            scores[(comp_a['name'], comp_b['name'])] = coupling
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating change frequency coupling: {e}")
            return {}
    
    def _evaluate_team_ownership_alignment(self, components: List[Dict], 
                                         graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate team ownership alignment"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    team_a = comp_a.get('owning_team', '')
                    team_b = comp_b.get('owning_team', '')
                    
                    if team_a and team_b:
                        # Same team = high cohesion
                        alignment = 1.0 if team_a == team_b else 0.2
                        scores[(comp_a['name'], comp_b['name'])] = alignment
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating team ownership alignment: {e}")
            return {}
    
    def _evaluate_performance_requirements(self, components: List[Dict], 
                                         graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate performance requirements similarity"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    perf_a = comp_a.get('performance_requirements', {})
                    perf_b = comp_b.get('performance_requirements', {})
                    
                    if perf_a and perf_b:
                        # Compare performance characteristics
                        similarity = self._calculate_performance_similarity(perf_a, perf_b)
                        scores[(comp_a['name'], comp_b['name'])] = similarity
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating performance requirements: {e}")
            return {}
    
    def _evaluate_security_boundary_alignment(self, components: List[Dict], 
                                            graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate security boundary alignment"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    sec_a = set(comp_a.get('security_requirements', []))
                    sec_b = set(comp_b.get('security_requirements', []))
                    
                    if sec_a and sec_b:
                        # Calculate security alignment
                        overlap = len(sec_a & sec_b)
                        total = len(sec_a | sec_b)
                        
                        if total > 0:
                            alignment = overlap / total
                            scores[(comp_a['name'], comp_b['name'])] = alignment
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating security boundary alignment: {e}")
            return {}
    
    def _evaluate_deployment_independence(self, components: List[Dict], 
                                        graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate deployment independence potential"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    # Check deployment dependencies
                    deps_a = set(comp_a.get('deployment_dependencies', []))
                    deps_b = set(comp_b.get('deployment_dependencies', []))
                    
                    # Lower shared dependencies = higher independence
                    shared_deps = len(deps_a & deps_b)
                    total_deps = len(deps_a | deps_b)
                    
                    if total_deps > 0:
                        independence = 1.0 - (shared_deps / total_deps)
                        scores[(comp_a['name'], comp_b['name'])] = independence * 0.5  # Lower weight for separation
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating deployment independence: {e}")
            return {}
    
    def _evaluate_technology_affinity(self, components: List[Dict], 
                                    graph: 'nx.Graph') -> Dict[Tuple[str, str], float]:
        """Evaluate technology stack affinity"""
        scores = {}
        
        try:
            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    tech_a = set(comp_a.get('technologies', []))
                    tech_b = set(comp_b.get('technologies', []))
                    
                    if tech_a and tech_b:
                        # Calculate technology overlap
                        overlap = len(tech_a & tech_b)
                        total = len(tech_a | tech_b)
                        
                        if total > 0:
                            affinity = overlap / total
                            scores[(comp_a['name'], comp_b['name'])] = affinity
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating technology affinity: {e}")
            return {}
    
    def _calculate_performance_similarity(self, perf_a: Dict, perf_b: Dict) -> float:
        """Calculate similarity between performance requirements"""
        try:
            common_metrics = set(perf_a.keys()) & set(perf_b.keys())
            
            if not common_metrics:
                return 0.0
            
            similarities = []
            for metric in common_metrics:
                val_a = perf_a[metric]
                val_b = perf_b[metric]
                
                # Calculate relative similarity
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    if val_a == 0 and val_b == 0:
                        similarity = 1.0
                    elif val_a == 0 or val_b == 0:
                        similarity = 0.0
                    else:
                        ratio = min(val_a, val_b) / max(val_a, val_b)
                        similarity = ratio
                    
                    similarities.append(similarity)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating performance similarity: {e}")
            return 0.0
    
    def _detect_service_clusters(self, graph: 'nx.Graph', 
                               boundary_scores: Dict) -> List[List[str]]:
        """Detect service clusters using community detection"""
        if not HAS_NETWORKX or not graph:
            return []
        
        try:
            # Use Louvain community detection if available
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.louvain_communities(graph, weight='weight')
                return [list(community) for community in communities]
            except ImportError:
                # Fallback to simple clustering
                return self._simple_clustering(graph, boundary_scores)
                
        except Exception as e:
            self.logger.error(f"Error detecting service clusters: {e}")
            return []
    
    def _simple_clustering(self, graph: 'nx.Graph', 
                          boundary_scores: Dict) -> List[List[str]]:
        """Simple clustering algorithm fallback"""
        try:
            clusters = []
            visited = set()
            
            for node in graph.nodes():
                if node not in visited:
                    cluster = [node]
                    visited.add(node)
                    
                    # Find connected nodes with high boundary scores
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            # Check boundary score
                            pair = tuple(sorted([node, neighbor]))
                            score = boundary_scores.get(pair, 0.0)
                            
                            if score > 0.5:  # Threshold for clustering
                                cluster.append(neighbor)
                                visited.add(neighbor)
                    
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in simple clustering: {e}")
            return []
    
    def _create_service_boundary(self, service_name: str, component_cluster: List[str],
                               components: List[Dict], graph: 'nx.Graph') -> MicroserviceBoundary:
        """Create service boundary from component cluster"""
        try:
            # Aggregate business capabilities
            business_capabilities = []
            data_entities = []
            dependencies = []
            technologies = set()
            
            for comp_name in component_cluster:
                comp = next((c for c in components if c['name'] == comp_name), {})
                business_capabilities.extend(comp.get('business_capabilities', []))
                data_entities.extend(comp.get('data_entities', []))
                dependencies.extend(comp.get('dependencies', []))
                technologies.update(comp.get('technologies', []))
            
            # Remove duplicates
            business_capabilities = list(set(business_capabilities))
            data_entities = list(set(data_entities))
            dependencies = list(set(dependencies))
            
            # Calculate coupling and cohesion scores
            coupling_score = self._calculate_cluster_coupling(component_cluster, graph)
            cohesion_score = self._calculate_cluster_cohesion(component_cluster, components)
            complexity_score = self._calculate_cluster_complexity(component_cluster, components)
            
            # Determine if boundary is recommended
            recommended = (coupling_score < 0.5 and 
                          cohesion_score > 0.6 and 
                          complexity_score < 0.7)
            
            # Generate rationale
            rationale = self._generate_boundary_rationale(
                coupling_score, cohesion_score, complexity_score, len(component_cluster)
            )
            
            return MicroserviceBoundary(
                service_name=service_name,
                business_capabilities=business_capabilities,
                data_entities=data_entities,
                dependencies=dependencies,
                communication_patterns=list(technologies),
                coupling_score=coupling_score,
                cohesion_score=cohesion_score,
                complexity_score=complexity_score,
                recommended=recommended,
                rationale=rationale
            )
            
        except Exception as e:
            self.logger.error(f"Error creating service boundary: {e}")
            return MicroserviceBoundary(
                service_name=service_name,
                business_capabilities=[],
                data_entities=[],
                dependencies=[],
                communication_patterns=[],
                coupling_score=0.5,
                cohesion_score=0.5,
                complexity_score=0.5
            )
    
    def _calculate_cluster_coupling(self, cluster: List[str], graph: 'nx.Graph') -> float:
        """Calculate coupling score for component cluster"""
        if not HAS_NETWORKX or not graph:
            return 0.5
        
        try:
            # Count internal vs external edges
            internal_edges = 0
            external_edges = 0
            
            cluster_set = set(cluster)
            
            for node in cluster:
                if node in graph:
                    for neighbor in graph.neighbors(node):
                        if neighbor in cluster_set:
                            internal_edges += 1
                        else:
                            external_edges += 1
            
            total_edges = internal_edges + external_edges
            if total_edges == 0:
                return 0.0
            
            # Lower external/total ratio = lower coupling
            coupling = external_edges / total_edges
            return coupling
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster coupling: {e}")
            return 0.5
    
    def _calculate_cluster_cohesion(self, cluster: List[str], components: List[Dict]) -> float:
        """Calculate cohesion score for component cluster"""
        try:
            if len(cluster) <= 1:
                return 1.0
            
            # Calculate average similarity between components in cluster
            similarities = []
            
            for i, comp_a_name in enumerate(cluster):
                for comp_b_name in cluster[i+1:]:
                    comp_a = next((c for c in components if c['name'] == comp_a_name), {})
                    comp_b = next((c for c in components if c['name'] == comp_b_name), {})
                    
                    # Calculate similarity based on multiple factors
                    cap_similarity = self._calculate_capability_similarity(comp_a, comp_b)
                    data_similarity = self._calculate_data_similarity(comp_a, comp_b)
                    
                    avg_similarity = (cap_similarity + data_similarity) / 2
                    similarities.append(avg_similarity)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster cohesion: {e}")
            return 0.5
    
    def _calculate_cluster_complexity(self, cluster: List[str], components: List[Dict]) -> float:
        """Calculate complexity score for component cluster"""
        try:
            # Factors that increase complexity
            total_dependencies = 0
            total_technologies = set()
            total_capabilities = 0
            
            for comp_name in cluster:
                comp = next((c for c in components if c['name'] == comp_name), {})
                total_dependencies += len(comp.get('dependencies', []))
                total_technologies.update(comp.get('technologies', []))
                total_capabilities += len(comp.get('business_capabilities', []))
            
            # Normalize complexity factors
            dependency_complexity = min(1.0, total_dependencies / 20)  # Assume 20+ deps is high
            technology_complexity = min(1.0, len(total_technologies) / 10)  # 10+ techs is high
            capability_complexity = min(1.0, total_capabilities / 15)  # 15+ capabilities is high
            
            # Average complexity
            complexity = (dependency_complexity + technology_complexity + capability_complexity) / 3
            return complexity
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster complexity: {e}")
            return 0.5
    
    def _calculate_capability_similarity(self, comp_a: Dict, comp_b: Dict) -> float:
        """Calculate business capability similarity"""
        try:
            cap_a = set(comp_a.get('business_capabilities', []))
            cap_b = set(comp_b.get('business_capabilities', []))
            
            if not cap_a and not cap_b:
                return 1.0
            
            if not cap_a or not cap_b:
                return 0.0
            
            intersection = len(cap_a & cap_b)
            union = len(cap_a | cap_b)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_data_similarity(self, comp_a: Dict, comp_b: Dict) -> float:
        """Calculate data entity similarity"""
        try:
            data_a = set(comp_a.get('data_entities', []))
            data_b = set(comp_b.get('data_entities', []))
            
            if not data_a and not data_b:
                return 1.0
            
            if not data_a or not data_b:
                return 0.0
            
            intersection = len(data_a & data_b)
            union = len(data_a | data_b)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_boundary_rationale(self, coupling: float, cohesion: float, 
                                   complexity: float, component_count: int) -> str:
        """Generate rationale for service boundary recommendation"""
        rationale_parts = []
        
        if coupling < 0.3:
            rationale_parts.append("Low coupling with external components")
        elif coupling > 0.7:
            rationale_parts.append("High coupling may indicate boundary issues")
        
        if cohesion > 0.7:
            rationale_parts.append("High internal cohesion")
        elif cohesion < 0.4:
            rationale_parts.append("Low cohesion may indicate diverse responsibilities")
        
        if complexity < 0.4:
            rationale_parts.append("Low complexity suitable for independent service")
        elif complexity > 0.7:
            rationale_parts.append("High complexity may require further decomposition")
        
        if component_count == 1:
            rationale_parts.append("Single component service")
        elif component_count > 10:
            rationale_parts.append("Large service may benefit from further decomposition")
        
        return ". ".join(rationale_parts) if rationale_parts else "Standard service boundary"
    
    def _refine_service_boundaries(self, boundaries: List[MicroserviceBoundary], 
                                 graph: 'nx.Graph') -> List[MicroserviceBoundary]:
        """Refine service boundaries based on analysis"""
        try:
            refined_boundaries = []
            
            for boundary in boundaries:
                # Check if boundary meets quality criteria
                if (boundary.coupling_score < 0.6 and 
                    boundary.cohesion_score > 0.5 and 
                    len(boundary.business_capabilities) > 0):
                    
                    refined_boundaries.append(boundary)
                else:
                    # Log boundary issues for potential improvement
                    self.logger.debug(f"Boundary {boundary.service_name} may need refinement: "
                                    f"coupling={boundary.coupling_score:.2f}, "
                                    f"cohesion={boundary.cohesion_score:.2f}")
                    
                    # Still include but mark as needing review
                    boundary.rationale += ". Requires review and potential refinement."
                    refined_boundaries.append(boundary)
            
            return refined_boundaries
            
        except Exception as e:
            self.logger.error(f"Error refining service boundaries: {e}")
            return boundaries


def create_microservice_analyzer() -> MicroserviceEvolutionAnalyzer:
    """Factory function to create microservice analyzer"""
    return MicroserviceEvolutionAnalyzer()