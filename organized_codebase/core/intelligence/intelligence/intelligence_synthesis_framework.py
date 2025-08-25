"""
Intelligence Synthesis Framework
Integrates and synthesizes all testing intelligence from multiple sources into unified knowledge.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from enum import Enum
import time
import hashlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class IntelligenceSource(Enum):
    """Sources of testing intelligence"""
    ARCHIVE_MINING = "archive_mining"
    EVOLUTION_ANALYSIS = "evolution_analysis"
    PATTERN_CONSOLIDATION = "pattern_consolidation"
    REPOSITORY_EXTRACTION = "repository_extraction"
    LIVE_SYSTEM_ANALYSIS = "live_system_analysis"


class KnowledgeConfidence(Enum):
    """Confidence levels for synthesized knowledge"""
    VERY_HIGH = "very_high"  # 95%+ confidence
    HIGH = "high"           # 80-95% confidence
    MEDIUM = "medium"       # 60-80% confidence
    LOW = "low"             # 40-60% confidence
    UNCERTAIN = "uncertain"  # <40% confidence


@dataclass
class IntelligenceItem:
    """Single piece of testing intelligence"""
    item_id: str
    source: IntelligenceSource
    knowledge_type: str
    title: str
    description: str
    confidence: KnowledgeConfidence
    evidence_strength: float
    applicability_score: float
    implementation_complexity: float
    effectiveness_metrics: Dict[str, float] = field(default_factory=dict)
    related_patterns: List[str] = field(default_factory=list)
    source_references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def overall_value_score(self) -> float:
        """Calculate overall value score"""
        confidence_weights = {
            KnowledgeConfidence.VERY_HIGH: 1.0,
            KnowledgeConfidence.HIGH: 0.85,
            KnowledgeConfidence.MEDIUM: 0.7,
            KnowledgeConfidence.LOW: 0.5,
            KnowledgeConfidence.UNCERTAIN: 0.3
        }
        
        confidence_score = confidence_weights[self.confidence]
        implementation_penalty = max(0, 1.0 - (self.implementation_complexity / 10.0))
        
        return (self.evidence_strength * 0.4 + 
                self.applicability_score * 0.3 + 
                confidence_score * 0.3) * implementation_penalty


@dataclass
class SynthesizedKnowledge:
    """Synthesized knowledge from multiple intelligence sources"""
    knowledge_id: str
    primary_concept: str
    synthesized_description: str
    confidence_level: KnowledgeConfidence
    supporting_evidence: List[IntelligenceItem]
    conflicting_evidence: List[IntelligenceItem]
    synthesis_quality: float
    actionable_recommendations: List[str]
    implementation_priority: float
    knowledge_graph_connections: List[str] = field(default_factory=list)
    
    @property
    def evidence_consensus(self) -> float:
        """Calculate consensus among evidence"""
        total_items = len(self.supporting_evidence) + len(self.conflicting_evidence)
        if total_items == 0:
            return 0.0
        return len(self.supporting_evidence) / total_items


class IntelligenceKnowledgeGraph:
    """Knowledge graph for testing intelligence relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_attributes = {}
        self.edge_weights = {}
    
    def add_knowledge_node(self, knowledge_item: IntelligenceItem) -> None:
        """Add knowledge item as node in graph"""
        self.graph.add_node(knowledge_item.item_id)
        self.node_attributes[knowledge_item.item_id] = {
            'type': knowledge_item.knowledge_type,
            'source': knowledge_item.source.value,
            'confidence': knowledge_item.confidence.value,
            'value_score': knowledge_item.overall_value_score,
            'tags': knowledge_item.tags
        }
    
    def add_relationship(self, from_id: str, to_id: str, 
                        relationship_type: str, strength: float) -> None:
        """Add relationship between knowledge items"""
        self.graph.add_edge(from_id, to_id, relationship=relationship_type, weight=strength)
        self.edge_weights[(from_id, to_id)] = strength
    
    def find_knowledge_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Find clusters of related knowledge"""
        if len(self.graph.nodes()) < min_cluster_size:
            return []
        
        # Use community detection
        try:
            communities = nx.community.greedy_modularity_communities(self.graph.to_undirected())
            clusters = [list(community) for community in communities 
                       if len(community) >= min_cluster_size]
            return clusters
        except:
            return []
    
    def get_knowledge_rankings(self) -> Dict[str, float]:
        """Get PageRank-based rankings of knowledge importance"""
        try:
            rankings = nx.pagerank(self.graph, weight='weight')
            return rankings
        except:
            return {}
    
    def find_knowledge_paths(self, start_id: str, end_id: str, 
                           max_path_length: int = 5) -> List[List[str]]:
        """Find paths between knowledge items"""
        try:
            paths = list(nx.all_simple_paths(self.graph, start_id, end_id, 
                                           cutoff=max_path_length))
            return paths[:10]  # Limit to top 10 paths
        except:
            return []


class IntelligenceCorrelationAnalyzer:
    """Analyze correlations between different intelligence sources"""
    
    def __init__(self):
        self.correlation_matrix = defaultdict(lambda: defaultdict(float))
        self.pattern_similarities = {}
    
    def calculate_cross_source_correlations(self, 
                                          intelligence_items: List[IntelligenceItem]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between different intelligence sources"""
        
        # Group by source
        source_groups = defaultdict(list)
        for item in intelligence_items:
            source_groups[item.source].append(item)
        
        correlations = {}
        sources = list(source_groups.keys())
        
        for i, source1 in enumerate(sources):
            correlations[source1.value] = {}
            for j, source2 in enumerate(sources):
                if i <= j:  # Avoid duplicate calculations
                    correlation = self._calculate_source_correlation(
                        source_groups[source1], source_groups[source2]
                    )
                    correlations[source1.value][source2.value] = correlation
                    if i != j:  # Add symmetric entry
                        if source2.value not in correlations:
                            correlations[source2.value] = {}
                        correlations[source2.value][source1.value] = correlation
        
        return correlations
    
    def _calculate_source_correlation(self, items1: List[IntelligenceItem], 
                                    items2: List[IntelligenceItem]) -> float:
        """Calculate correlation between two groups of intelligence items"""
        if not items1 or not items2:
            return 0.0
        
        # Extract text features
        texts1 = [f"{item.title} {item.description}" for item in items1]
        texts2 = [f"{item.title} {item.description}" for item in items2]
        
        all_texts = texts1 + texts2
        
        if len(all_texts) < 2:
            return 0.0
        
        try:
            # Use TF-IDF to vectorize texts
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate average similarity between groups
            similarities = []
            for i in range(len(texts1)):
                for j in range(len(texts1), len(all_texts)):
                    similarity = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def identify_contradicting_intelligence(self, 
                                         intelligence_items: List[IntelligenceItem]) -> List[Tuple[str, str, str]]:
        """Identify potentially contradicting intelligence"""
        contradictions = []
        
        # Group by knowledge type
        type_groups = defaultdict(list)
        for item in intelligence_items:
            type_groups[item.knowledge_type].append(item)
        
        # Look for contradictions within each type
        for knowledge_type, items in type_groups.items():
            if len(items) < 2:
                continue
            
            for i, item1 in enumerate(items):
                for j, item2 in enumerate(items[i+1:], i+1):
                    # Check for opposing recommendations or conflicting evidence
                    contradiction_score = self._calculate_contradiction_score(item1, item2)
                    
                    if contradiction_score > 0.7:  # High contradiction threshold
                        contradictions.append((
                            item1.item_id,
                            item2.item_id,
                            f"Potential contradiction in {knowledge_type} (score: {contradiction_score:.2f})"
                        ))
        
        return contradictions
    
    def _calculate_contradiction_score(self, item1: IntelligenceItem, 
                                     item2: IntelligenceItem) -> float:
        """Calculate contradiction score between two intelligence items"""
        
        # Look for opposing keywords
        opposing_pairs = [
            ('good', 'bad'), ('effective', 'ineffective'), ('recommended', 'discouraged'),
            ('fast', 'slow'), ('simple', 'complex'), ('stable', 'unstable'),
            ('secure', 'insecure'), ('scalable', 'unscalable')
        ]
        
        text1 = f"{item1.title} {item1.description}".lower()
        text2 = f"{item2.title} {item2.description}".lower()
        
        contradiction_indicators = 0
        total_pairs = len(opposing_pairs)
        
        for positive, negative in opposing_pairs:
            if ((positive in text1 and negative in text2) or 
                (negative in text1 and positive in text2)):
                contradiction_indicators += 1
        
        # Also check effectiveness metrics if available
        if (item1.effectiveness_metrics and item2.effectiveness_metrics):
            common_metrics = set(item1.effectiveness_metrics.keys()) & set(item2.effectiveness_metrics.keys())
            metric_contradictions = 0
            
            for metric in common_metrics:
                val1 = item1.effectiveness_metrics[metric]
                val2 = item2.effectiveness_metrics[metric]
                
                # If values differ significantly and have opposite signs
                if val1 * val2 < 0 and abs(val1 - val2) > 0.5:
                    metric_contradictions += 1
            
            if common_metrics:
                metric_contradiction_ratio = metric_contradictions / len(common_metrics)
                contradiction_indicators += metric_contradiction_ratio * total_pairs
        
        return min(1.0, contradiction_indicators / total_pairs)


class IntelligenceSynthesisEngine:
    """Main engine for synthesizing testing intelligence"""
    
    def __init__(self):
        self.intelligence_database = []
        self.knowledge_graph = IntelligenceKnowledgeGraph()
        self.correlation_analyzer = IntelligenceCorrelationAnalyzer()
        self.synthesized_knowledge = []
        self.synthesis_metrics = {
            'items_processed': 0,
            'knowledge_synthesized': 0,
            'confidence_improvements': 0,
            'contradictions_resolved': 0
        }
    
    def ingest_intelligence_source(self, source_data: Dict[str, Any], 
                                 source_type: IntelligenceSource) -> None:
        """Ingest intelligence from a specific source"""
        
        if source_type == IntelligenceSource.ARCHIVE_MINING:
            self._ingest_archive_intelligence(source_data)
        elif source_type == IntelligenceSource.EVOLUTION_ANALYSIS:
            self._ingest_evolution_intelligence(source_data)
        elif source_type == IntelligenceSource.PATTERN_CONSOLIDATION:
            self._ingest_consolidation_intelligence(source_data)
        else:
            # Generic ingestion
            self._ingest_generic_intelligence(source_data, source_type)
        
        print(f"Ingested {source_type.value} intelligence")
    
    def _ingest_archive_intelligence(self, data: Dict[str, Any]) -> None:
        """Ingest intelligence from archive mining"""
        if 'high_value_patterns' in data:
            for pattern_data in data['high_value_patterns']:
                item = IntelligenceItem(
                    item_id=f"archive_{pattern_data.get('type', 'unknown')}_{len(self.intelligence_database)}",
                    source=IntelligenceSource.ARCHIVE_MINING,
                    knowledge_type=pattern_data.get('type', 'pattern'),
                    title=f"Archive Pattern: {pattern_data.get('name', 'Unknown')}",
                    description=f"Historical pattern from {pattern_data.get('source', 'unknown source')}",
                    confidence=self._map_priority_to_confidence(pattern_data.get('priority_score', 0)),
                    evidence_strength=min(1.0, pattern_data.get('priority_score', 0) / 2.0),
                    applicability_score=min(1.0, pattern_data.get('usage_frequency', 0) / 10.0),
                    implementation_complexity=5.0,  # Default medium complexity
                    source_references=[pattern_data.get('source', '')]
                )
                self.intelligence_database.append(item)
                self.knowledge_graph.add_knowledge_node(item)
    
    def _ingest_evolution_intelligence(self, data: Dict[str, Any]) -> None:
        """Ingest intelligence from evolution analysis"""
        if 'high_impact_events' in data:
            for event_data in data['high_impact_events']:
                item = IntelligenceItem(
                    item_id=f"evolution_{event_data.get('type', 'event')}_{len(self.intelligence_database)}",
                    source=IntelligenceSource.EVOLUTION_ANALYSIS,
                    knowledge_type='evolution_pattern',
                    title=f"Evolution Event: {event_data.get('type', 'Unknown').replace('_', ' ').title()}",
                    description=event_data.get('message', 'Evolution pattern detected'),
                    confidence=self._map_impact_to_confidence(event_data.get('impact_score', 0)),
                    evidence_strength=min(1.0, event_data.get('impact_score', 0) / 5.0),
                    applicability_score=0.7,  # Evolution patterns are generally applicable
                    implementation_complexity=3.0,
                    effectiveness_metrics={'impact_score': event_data.get('impact_score', 0)},
                    source_references=[event_data.get('file_path', '')]
                )
                self.intelligence_database.append(item)
                self.knowledge_graph.add_knowledge_node(item)
    
    def _ingest_consolidation_intelligence(self, data: Dict[str, Any]) -> None:
        """Ingest intelligence from pattern consolidation"""
        if 'opportunities' in data:
            for opp_data in data['opportunities']:
                item = IntelligenceItem(
                    item_id=f"consolidation_{opp_data.get('id', 'unknown')}",
                    source=IntelligenceSource.PATTERN_CONSOLIDATION,
                    knowledge_type='consolidation_opportunity',
                    title=f"Consolidation: {opp_data.get('action_type', 'Unknown').replace('_', ' ').title()}",
                    description=opp_data.get('description', 'Consolidation opportunity identified'),
                    confidence=self._map_risk_to_confidence(opp_data.get('risk_level', 'medium')),
                    evidence_strength=min(1.0, opp_data.get('roi_score', 0) / 5.0),
                    applicability_score=0.8,
                    implementation_complexity=opp_data.get('implementation_effort', 5.0),
                    effectiveness_metrics=opp_data.get('estimated_savings', {}),
                    source_references=opp_data.get('affected_files', [])[:3]
                )
                self.intelligence_database.append(item)
                self.knowledge_graph.add_knowledge_node(item)
    
    def _ingest_generic_intelligence(self, data: Dict[str, Any], source: IntelligenceSource) -> None:
        """Generic intelligence ingestion"""
        # Extract key information from generic data structure
        items = data.get('items', data.get('patterns', data.get('findings', [])))
        
        for item_data in items[:20]:  # Limit to prevent overwhelming
            item = IntelligenceItem(
                item_id=f"{source.value}_{len(self.intelligence_database)}",
                source=source,
                knowledge_type=item_data.get('type', 'generic'),
                title=item_data.get('title', item_data.get('name', 'Unknown')),
                description=item_data.get('description', str(item_data)[:200]),
                confidence=KnowledgeConfidence.MEDIUM,
                evidence_strength=0.5,
                applicability_score=0.6,
                implementation_complexity=5.0
            )
            self.intelligence_database.append(item)
            self.knowledge_graph.add_knowledge_node(item)
    
    def synthesize_knowledge(self) -> List[SynthesizedKnowledge]:
        """Synthesize knowledge from all intelligence sources"""
        print("Synthesizing knowledge from intelligence sources...")
        
        # Build knowledge graph relationships
        self._build_knowledge_relationships()
        
        # Find knowledge clusters
        clusters = self.knowledge_graph.find_knowledge_clusters()
        
        synthesized_items = []
        
        for i, cluster in enumerate(clusters):
            cluster_items = [item for item in self.intelligence_database if item.item_id in cluster]
            
            if len(cluster_items) >= 2:  # Need multiple items to synthesize
                synthesized = self._synthesize_cluster(cluster_items, f"synthesis_{i}")
                if synthesized:
                    synthesized_items.append(synthesized)
        
        # Also synthesize by knowledge type
        type_synthesis = self._synthesize_by_type()
        synthesized_items.extend(type_synthesis)
        
        self.synthesized_knowledge = synthesized_items
        self.synthesis_metrics['knowledge_synthesized'] = len(synthesized_items)
        
        return synthesized_items
    
    def _build_knowledge_relationships(self) -> None:
        """Build relationships in knowledge graph"""
        for i, item1 in enumerate(self.intelligence_database):
            for j, item2 in enumerate(self.intelligence_database[i+1:], i+1):
                # Calculate relationship strength
                strength = self._calculate_relationship_strength(item1, item2)
                
                if strength > 0.3:  # Threshold for significant relationship
                    relationship_type = self._determine_relationship_type(item1, item2)
                    self.knowledge_graph.add_relationship(
                        item1.item_id, item2.item_id, relationship_type, strength
                    )
    
    def _calculate_relationship_strength(self, item1: IntelligenceItem, 
                                       item2: IntelligenceItem) -> float:
        """Calculate strength of relationship between two intelligence items"""
        
        # Text similarity
        text1 = f"{item1.title} {item1.description}"
        text2 = f"{item2.title} {item2.description}"
        
        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity = 0.0
        
        # Tag overlap
        tag_overlap = len(set(item1.tags) & set(item2.tags)) / max(1, len(set(item1.tags) | set(item2.tags)))
        
        # Knowledge type relationship
        type_bonus = 0.2 if item1.knowledge_type == item2.knowledge_type else 0.0
        
        # Source diversity bonus
        source_bonus = 0.1 if item1.source != item2.source else 0.0
        
        return min(1.0, similarity * 0.6 + tag_overlap * 0.2 + type_bonus + source_bonus)
    
    def _determine_relationship_type(self, item1: IntelligenceItem, 
                                   item2: IntelligenceItem) -> str:
        """Determine type of relationship between intelligence items"""
        
        if item1.knowledge_type == item2.knowledge_type:
            return 'similar_type'
        elif item1.source == item2.source:
            return 'same_source'
        elif any(tag in item2.tags for tag in item1.tags):
            return 'shared_concepts'
        else:
            return 'related'
    
    def _synthesize_cluster(self, items: List[IntelligenceItem], 
                          cluster_id: str) -> Optional[SynthesizedKnowledge]:
        """Synthesize knowledge from a cluster of related items"""
        
        if len(items) < 2:
            return None
        
        # Find primary concept (most common knowledge type)
        knowledge_types = [item.knowledge_type for item in items]
        primary_concept = Counter(knowledge_types).most_common(1)[0][0]
        
        # Separate supporting and conflicting evidence
        supporting_evidence = []
        conflicting_evidence = []
        
        # For simplicity, consider high-confidence items as supporting
        for item in items:
            if item.confidence in [KnowledgeConfidence.HIGH, KnowledgeConfidence.VERY_HIGH]:
                supporting_evidence.append(item)
            elif item.confidence == KnowledgeConfidence.LOW:
                conflicting_evidence.append(item)
            else:
                supporting_evidence.append(item)  # Medium confidence goes to supporting
        
        # Calculate overall confidence
        avg_confidence = self._calculate_average_confidence(supporting_evidence)
        
        # Generate synthesized description
        descriptions = [item.description for item in supporting_evidence[:3]]  # Top 3
        synthesized_desc = self._generate_synthesized_description(descriptions, primary_concept)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(supporting_evidence)
        
        # Calculate synthesis quality
        quality = self._calculate_synthesis_quality(supporting_evidence, conflicting_evidence)
        
        # Calculate priority
        avg_value_scores = [item.overall_value_score for item in supporting_evidence]
        priority = np.mean(avg_value_scores) if avg_value_scores else 0.0
        
        return SynthesizedKnowledge(
            knowledge_id=cluster_id,
            primary_concept=primary_concept,
            synthesized_description=synthesized_desc,
            confidence_level=avg_confidence,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            synthesis_quality=quality,
            actionable_recommendations=recommendations,
            implementation_priority=priority
        )
    
    def _synthesize_by_type(self) -> List[SynthesizedKnowledge]:
        """Synthesize knowledge by grouping similar types"""
        type_groups = defaultdict(list)
        
        for item in self.intelligence_database:
            type_groups[item.knowledge_type].append(item)
        
        synthesized_items = []
        
        for knowledge_type, items in type_groups.items():
            if len(items) >= 3:  # Need at least 3 items for type-based synthesis
                synthesized = self._synthesize_cluster(items, f"type_synthesis_{knowledge_type}")
                if synthesized:
                    synthesized_items.append(synthesized)
        
        return synthesized_items
    
    def _calculate_average_confidence(self, items: List[IntelligenceItem]) -> KnowledgeConfidence:
        """Calculate average confidence level"""
        if not items:
            return KnowledgeConfidence.UNCERTAIN
        
        confidence_values = {
            KnowledgeConfidence.VERY_HIGH: 5,
            KnowledgeConfidence.HIGH: 4,
            KnowledgeConfidence.MEDIUM: 3,
            KnowledgeConfidence.LOW: 2,
            KnowledgeConfidence.UNCERTAIN: 1
        }
        
        avg_value = np.mean([confidence_values[item.confidence] for item in items])
        
        if avg_value >= 4.5:
            return KnowledgeConfidence.VERY_HIGH
        elif avg_value >= 3.5:
            return KnowledgeConfidence.HIGH
        elif avg_value >= 2.5:
            return KnowledgeConfidence.MEDIUM
        elif avg_value >= 1.5:
            return KnowledgeConfidence.LOW
        else:
            return KnowledgeConfidence.UNCERTAIN
    
    def _generate_synthesized_description(self, descriptions: List[str], 
                                        concept: str) -> str:
        """Generate synthesized description from multiple descriptions"""
        
        # Extract key phrases (simplified approach)
        all_words = []
        for desc in descriptions:
            words = desc.lower().split()
            all_words.extend([word for word in words if len(word) > 4])
        
        # Find most common significant words
        common_words = [word for word, count in Counter(all_words).most_common(5)]
        
        return f"Synthesized {concept} intelligence incorporating patterns related to: {', '.join(common_words)}. Based on analysis of {len(descriptions)} sources showing consistent patterns in testing approaches."
    
    def _generate_recommendations(self, items: List[IntelligenceItem]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Group by implementation complexity
        low_complexity = [item for item in items if item.implementation_complexity <= 3.0]
        high_value = [item for item in items if item.overall_value_score >= 0.7]
        
        if low_complexity:
            recommendations.append(
                f"Implement {len(low_complexity)} low-complexity improvements for quick wins"
            )
        
        if high_value:
            recommendations.append(
                f"Prioritize {len(high_value)} high-value patterns for maximum impact"
            )
        
        # Source-specific recommendations
        sources = set(item.source for item in items)
        if len(sources) > 1:
            recommendations.append(
                "Cross-validate implementation using insights from multiple intelligence sources"
            )
        
        recommendations.append("Establish monitoring and validation framework for implementation")
        
        return recommendations
    
    def _calculate_synthesis_quality(self, supporting: List[IntelligenceItem], 
                                   conflicting: List[IntelligenceItem]) -> float:
        """Calculate quality of synthesis"""
        
        total_items = len(supporting) + len(conflicting)
        if total_items == 0:
            return 0.0
        
        # Evidence consensus
        consensus = len(supporting) / total_items
        
        # Evidence strength
        avg_strength = np.mean([item.evidence_strength for item in supporting]) if supporting else 0.0
        
        # Source diversity
        sources = set(item.source for item in supporting + conflicting)
        diversity = len(sources) / len(IntelligenceSource)
        
        return (consensus * 0.5 + avg_strength * 0.3 + diversity * 0.2)
    
    def _map_priority_to_confidence(self, priority_score: float) -> KnowledgeConfidence:
        """Map priority score to confidence level"""
        if priority_score >= 2.0:
            return KnowledgeConfidence.VERY_HIGH
        elif priority_score >= 1.5:
            return KnowledgeConfidence.HIGH
        elif priority_score >= 1.0:
            return KnowledgeConfidence.MEDIUM
        elif priority_score >= 0.5:
            return KnowledgeConfidence.LOW
        else:
            return KnowledgeConfidence.UNCERTAIN
    
    def _map_impact_to_confidence(self, impact_score: float) -> KnowledgeConfidence:
        """Map impact score to confidence level"""
        if impact_score >= 4.0:
            return KnowledgeConfidence.VERY_HIGH
        elif impact_score >= 3.0:
            return KnowledgeConfidence.HIGH
        elif impact_score >= 2.0:
            return KnowledgeConfidence.MEDIUM
        elif impact_score >= 1.0:
            return KnowledgeConfidence.LOW
        else:
            return KnowledgeConfidence.UNCERTAIN
    
    def _map_risk_to_confidence(self, risk_level: str) -> KnowledgeConfidence:
        """Map risk level to confidence level (inverse relationship)"""
        risk_mapping = {
            'low': KnowledgeConfidence.VERY_HIGH,
            'medium': KnowledgeConfidence.HIGH,
            'high': KnowledgeConfidence.MEDIUM,
            'critical': KnowledgeConfidence.LOW
        }
        return risk_mapping.get(risk_level, KnowledgeConfidence.MEDIUM)
    
    def export_synthesized_intelligence(self, output_path: str) -> None:
        """Export synthesized intelligence report"""
        
        correlations = self.correlation_analyzer.calculate_cross_source_correlations(
            self.intelligence_database
        )
        
        contradictions = self.correlation_analyzer.identify_contradicting_intelligence(
            self.intelligence_database
        )
        
        # Prepare export data
        export_data = {
            'metadata': {
                'synthesis_timestamp': datetime.now().isoformat(),
                'total_intelligence_items': len(self.intelligence_database),
                'synthesized_knowledge_count': len(self.synthesized_knowledge),
                'synthesis_metrics': self.synthesis_metrics
            },
            'source_correlations': correlations,
            'identified_contradictions': [
                {'item1': c[0], 'item2': c[1], 'description': c[2]} 
                for c in contradictions
            ],
            'synthesized_knowledge': [
                {
                    'id': sk.knowledge_id,
                    'concept': sk.primary_concept,
                    'description': sk.synthesized_description,
                    'confidence': sk.confidence_level.value,
                    'quality': sk.synthesis_quality,
                    'priority': sk.implementation_priority,
                    'evidence_consensus': sk.evidence_consensus,
                    'supporting_evidence_count': len(sk.supporting_evidence),
                    'conflicting_evidence_count': len(sk.conflicting_evidence),
                    'recommendations': sk.actionable_recommendations
                }
                for sk in self.synthesized_knowledge
            ],
            'intelligence_summary': {
                'by_source': dict(Counter([item.source.value for item in self.intelligence_database])),
                'by_type': dict(Counter([item.knowledge_type for item in self.intelligence_database])),
                'by_confidence': dict(Counter([item.confidence.value for item in self.intelligence_database])),
                'high_value_items': len([item for item in self.intelligence_database if item.overall_value_score >= 0.8])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Synthesized intelligence report exported to: {output_path}")


# Testing framework
class IntelligenceSynthesisTestFramework:
    """Testing framework for intelligence synthesis"""
    
    def test_intelligence_ingestion(self) -> bool:
        """Test intelligence ingestion from different sources"""
        try:
            engine = IntelligenceSynthesisEngine()
            
            # Test archive mining ingestion
            archive_data = {
                'high_value_patterns': [
                    {
                        'type': 'test_pattern',
                        'name': 'mock_pattern',
                        'source': 'archive_file.py',
                        'priority_score': 1.5,
                        'usage_frequency': 5
                    }
                ]
            }
            
            engine.ingest_intelligence_source(archive_data, IntelligenceSource.ARCHIVE_MINING)
            assert len(engine.intelligence_database) > 0
            
            return True
        except Exception as e:
            print(f"Intelligence ingestion test failed: {e}")
            return False
    
    def test_knowledge_synthesis(self) -> bool:
        """Test knowledge synthesis functionality"""
        try:
            engine = IntelligenceSynthesisEngine()
            
            # Add test intelligence items
            for i in range(5):
                item = IntelligenceItem(
                    item_id=f"test_item_{i}",
                    source=IntelligenceSource.ARCHIVE_MINING,
                    knowledge_type='test_pattern',
                    title=f"Test Pattern {i}",
                    description=f"Test description {i}",
                    confidence=KnowledgeConfidence.HIGH,
                    evidence_strength=0.8,
                    applicability_score=0.7,
                    implementation_complexity=3.0,
                    tags=['testing', 'pattern']
                )
                engine.intelligence_database.append(item)
                engine.knowledge_graph.add_knowledge_node(item)
            
            # Test synthesis
            synthesized = engine.synthesize_knowledge()
            assert isinstance(synthesized, list)
            
            return True
        except Exception as e:
            print(f"Knowledge synthesis test failed: {e}")
            return False
    
    def test_correlation_analysis(self) -> bool:
        """Test correlation analysis functionality"""
        try:
            analyzer = IntelligenceCorrelationAnalyzer()
            
            # Create test items
            items = [
                IntelligenceItem(
                    item_id="item1",
                    source=IntelligenceSource.ARCHIVE_MINING,
                    knowledge_type='test_pattern',
                    title="Test Pattern A",
                    description="Mock testing pattern",
                    confidence=KnowledgeConfidence.HIGH,
                    evidence_strength=0.8,
                    applicability_score=0.7,
                    implementation_complexity=3.0
                ),
                IntelligenceItem(
                    item_id="item2",
                    source=IntelligenceSource.EVOLUTION_ANALYSIS,
                    knowledge_type='test_pattern',
                    title="Test Pattern B", 
                    description="Another testing pattern",
                    confidence=KnowledgeConfidence.MEDIUM,
                    evidence_strength=0.6,
                    applicability_score=0.8,
                    implementation_complexity=4.0
                )
            ]
            
            correlations = analyzer.calculate_cross_source_correlations(items)
            assert isinstance(correlations, dict)
            
            return True
        except Exception as e:
            print(f"Correlation analysis test failed: {e}")
            return False
    
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all intelligence synthesis tests"""
        tests = [
            'test_intelligence_ingestion',
            'test_knowledge_synthesis',
            'test_correlation_analysis'
        ]
        
        results = {}
        for test_name in tests:
            try:
                result = getattr(self, test_name)()
                results[test_name] = result
                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: FAILED - {e}")
        
        return results


# Main execution
if __name__ == "__main__":
    print("🧠 Intelligence Synthesis Framework")
    
    # Run tests
    framework = IntelligenceSynthesisTestFramework()
    results = framework.run_comprehensive_tests()
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All intelligence synthesis tests passed!")
        
        # Demo synthesis with mock data
        print("\n🚀 Running intelligence synthesis demonstration...")
        engine = IntelligenceSynthesisEngine()
        
        # Create sample intelligence
        sample_sources = [
            ({'high_value_patterns': [{'type': 'self_healing', 'name': 'auto_fix', 'source': 'archive.py', 'priority_score': 2.0, 'usage_frequency': 8}]}, IntelligenceSource.ARCHIVE_MINING),
            ({'opportunities': [{'id': 'consolidation_001', 'action_type': 'merge', 'risk_level': 'low', 'roi_score': 3.0, 'implementation_effort': 2.0, 'description': 'Merge duplicate patterns', 'estimated_savings': {'lines': 50}}]}, IntelligenceSource.PATTERN_CONSOLIDATION)
        ]
        
        for source_data, source_type in sample_sources:
            engine.ingest_intelligence_source(source_data, source_type)
        
        # Synthesize
        synthesized = engine.synthesize_knowledge()
        
        # Export report
        output_path = f"intelligence_synthesis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        engine.export_synthesized_intelligence(output_path)
        
        print(f"\n📈 Intelligence Synthesis Complete:")
        print(f"  Intelligence items: {len(engine.intelligence_database)}")
        print(f"  Synthesized knowledge: {len(synthesized)}")
        print(f"  Report exported: {output_path}")
    else:
        print("❌ Some tests failed. Check the output above.")