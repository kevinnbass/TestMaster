"""
Meta-Intelligence Capability Mapper
===================================

Intelligence capability discovery and classification with NLP and machine learning.
Extracted from meta_intelligence_orchestrator.py for enterprise modular architecture.

Agent D Implementation - Hour 14-15: Revolutionary Intelligence Modularization
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter

# Machine learning imports
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. Capability analysis will be simplified.")

# Network analysis imports
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("NetworkX not available. Capability relationship analysis will be limited.")

from .data_models import (
    CapabilityType, CapabilityProfile, CapabilityCluster, 
    IntelligenceSystemRegistration
)


class IntelligenceCapabilityMapper:
    """Advanced intelligence capability discovery and mapping system"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Capability keywords and patterns
        self.capability_keywords = self._initialize_capability_keywords()
        
        # Performance characteristic patterns
        self.performance_patterns = self._initialize_performance_patterns()
        
        # Capability relationship graph
        self.capability_graph = None
        if HAS_NETWORKX:
            self.capability_graph = nx.Graph()
            self._initialize_capability_graph()
        
        # Text analysis components
        if HAS_SKLEARN:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Discovered systems registry
        self.discovered_systems = {}
        self.capability_clusters = {}
    
    def _initialize_capability_keywords(self) -> Dict[CapabilityType, List[str]]:
        """Initialize capability detection keywords"""
        return {
            CapabilityType.NATURAL_LANGUAGE_PROCESSING: [
                'nlp', 'natural language', 'text processing', 'language model', 'tokenization',
                'sentiment analysis', 'named entity', 'text classification', 'language understanding',
                'text generation', 'translation', 'summarization', 'question answering'
            ],
            
            CapabilityType.COMPUTER_VISION: [
                'computer vision', 'image processing', 'object detection', 'image classification',
                'face recognition', 'ocr', 'image segmentation', 'feature extraction',
                'visual recognition', 'image analysis', 'opencv', 'tensorflow vision'
            ],
            
            CapabilityType.MACHINE_LEARNING: [
                'machine learning', 'ml', 'neural network', 'deep learning', 'tensorflow',
                'pytorch', 'scikit-learn', 'model training', 'supervised learning',
                'unsupervised learning', 'reinforcement learning', 'feature engineering'
            ],
            
            CapabilityType.DATA_ANALYSIS: [
                'data analysis', 'data science', 'statistics', 'pandas', 'numpy',
                'data visualization', 'exploratory data analysis', 'data mining',
                'statistical analysis', 'data processing', 'analytics'
            ],
            
            CapabilityType.PATTERN_RECOGNITION: [
                'pattern recognition', 'anomaly detection', 'clustering', 'classification',
                'pattern matching', 'signal processing', 'time series analysis',
                'trend analysis', 'pattern discovery'
            ],
            
            CapabilityType.DECISION_MAKING: [
                'decision making', 'decision support', 'expert system', 'rule engine',
                'recommendation system', 'choice analysis', 'multi-criteria decision',
                'decision tree', 'game theory'
            ],
            
            CapabilityType.OPTIMIZATION: [
                'optimization', 'linear programming', 'genetic algorithm', 'simulated annealing',
                'constraint optimization', 'multi-objective optimization', 'resource allocation',
                'scheduling', 'operations research'
            ],
            
            CapabilityType.PREDICTION: [
                'prediction', 'forecasting', 'time series', 'predictive modeling',
                'trend prediction', 'future analysis', 'predictive analytics',
                'regression', 'neural network prediction'
            ],
            
            CapabilityType.CLASSIFICATION: [
                'classification', 'categorization', 'image classification', 'text classification',
                'multi-class', 'binary classification', 'ensemble classification',
                'support vector machine', 'random forest'
            ],
            
            CapabilityType.GENERATION: [
                'generation', 'content generation', 'text generation', 'image generation',
                'synthetic data', 'gpt', 'generative model', 'autoencoder',
                'variational autoencoder', 'generative adversarial network'
            ]
        }
    
    def _initialize_performance_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize performance characteristic detection patterns"""
        return {
            'latency_indicators': {
                'low_latency': r'(real.?time|low.?latency|fast|quick|immediate|instant)',
                'high_latency': r'(batch|offline|slow|delayed|background)',
                'variable_latency': r'(variable|adaptive|dynamic|conditional)'
            },
            
            'throughput_indicators': {
                'high_throughput': r'(high.?throughput|bulk|mass|parallel|concurrent)',
                'low_throughput': r'(sequential|single|limited|constrained)',
                'scalable_throughput': r'(scalable|elastic|auto.?scale)'
            },
            
            'accuracy_indicators': {
                'high_accuracy': r'(precise|accurate|exact|perfect|100%)',
                'probabilistic': r'(probabilistic|confidence|uncertainty|approximate)',
                'learning_based': r'(learning|adaptive|improving|evolving)'
            },
            
            'resource_indicators': {
                'cpu_intensive': r'(cpu|processing|computation|algorithm|calculation)',
                'memory_intensive': r'(memory|ram|cache|storage|large.?data)',
                'gpu_intensive': r'(gpu|cuda|opencl|parallel|tensor)',
                'network_intensive': r'(network|api|remote|distributed|cloud)'
            }
        }
    
    def _initialize_capability_graph(self):
        """Initialize capability relationship graph"""
        if not HAS_NETWORKX:
            return
        
        # Add capability nodes
        for capability in CapabilityType:
            self.capability_graph.add_node(capability.value)
        
        # Define capability relationships
        relationships = [
            # Strong relationships
            ('machine_learning', 'pattern_recognition', 0.9),
            ('machine_learning', 'classification', 0.9),
            ('machine_learning', 'prediction', 0.8),
            ('natural_language_processing', 'classification', 0.7),
            ('computer_vision', 'pattern_recognition', 0.8),
            ('computer_vision', 'classification', 0.8),
            ('data_analysis', 'pattern_recognition', 0.7),
            ('optimization', 'decision_making', 0.8),
            ('prediction', 'data_analysis', 0.7),
            ('generation', 'machine_learning', 0.6),
            
            # Medium relationships
            ('natural_language_processing', 'generation', 0.6),
            ('computer_vision', 'generation', 0.5),
            ('decision_making', 'prediction', 0.6),
            ('optimization', 'machine_learning', 0.5),
            ('pattern_recognition', 'classification', 0.7),
            ('data_analysis', 'prediction', 0.8)
        ]
        
        # Add edges
        for source, target, weight in relationships:
            self.capability_graph.add_edge(source, target, weight=weight)
    
    def discover_system_capabilities(self, system_registration: IntelligenceSystemRegistration) -> CapabilityProfile:
        """Discover and map system capabilities from registration information"""
        try:
            self.logger.info(f"Discovering capabilities for system: {system_registration.system_name}")
            
            # Extract text for analysis
            analysis_text = self._prepare_analysis_text(system_registration)
            
            # Detect capabilities using keyword matching
            capability_scores = self._detect_capabilities_keywords(analysis_text)
            
            # Enhance with NLP analysis if available
            if HAS_SKLEARN:
                nlp_scores = self._detect_capabilities_nlp(analysis_text)
                capability_scores = self._merge_capability_scores(capability_scores, nlp_scores)
            
            # Extract performance characteristics
            performance_characteristics = self._extract_performance_characteristics(analysis_text)
            
            # Estimate resource requirements
            resource_requirements = self._estimate_resource_requirements(analysis_text, capability_scores)
            
            # Analyze input/output types
            input_types, output_types = self._analyze_io_types(analysis_text)
            
            # Create capability profile
            profile = CapabilityProfile(
                system_id=system_registration.system_id,
                system_name=system_registration.system_name,
                capabilities=capability_scores,
                performance_characteristics=performance_characteristics,
                resource_requirements=resource_requirements,
                input_types=input_types,
                output_types=output_types,
                processing_time=self._estimate_processing_time(performance_characteristics),
                accuracy=self._estimate_accuracy(capability_scores, analysis_text),
                reliability=self._estimate_reliability(analysis_text),
                scalability=self._estimate_scalability(analysis_text),
                cost_per_operation=self._estimate_cost(resource_requirements),
                api_endpoints=system_registration.api_endpoints,
                documentation_quality=self._assess_documentation_quality(analysis_text)
            )
            
            # Store discovered system
            self.discovered_systems[system_registration.system_id] = profile
            
            self.logger.info(f"Discovered {len(capability_scores)} capabilities for {system_registration.system_name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error discovering system capabilities: {e}")
            return self._create_fallback_profile(system_registration)
    
    def _prepare_analysis_text(self, registration: IntelligenceSystemRegistration) -> str:
        """Prepare text for capability analysis"""
        text_parts = [
            registration.system_name,
            registration.system_type,
            registration.capability_description
        ]
        
        # Add API endpoint information
        for endpoint in registration.api_endpoints:
            text_parts.append(endpoint)
        
        return " ".join(filter(None, text_parts)).lower()
    
    def _detect_capabilities_keywords(self, text: str) -> Dict[CapabilityType, float]:
        """Detect capabilities using keyword matching"""
        capability_scores = {}
        
        try:
            for capability_type, keywords in self.capability_keywords.items():
                score = 0.0
                matches = 0
                
                for keyword in keywords:
                    # Use word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    keyword_matches = len(re.findall(pattern, text))
                    
                    if keyword_matches > 0:
                        matches += 1
                        # Weight by keyword importance (longer keywords = more specific)
                        weight = len(keyword.split()) * 0.1 + 0.1
                        score += keyword_matches * weight
                
                # Normalize score
                if matches > 0:
                    normalized_score = min(1.0, score / len(keywords))
                    capability_scores[capability_type] = normalized_score
                
            return capability_scores
            
        except Exception as e:
            self.logger.error(f"Error in keyword-based capability detection: {e}")
            return {}
    
    def _detect_capabilities_nlp(self, text: str) -> Dict[CapabilityType, float]:
        """Detect capabilities using NLP analysis"""
        if not HAS_SKLEARN:
            return {}
        
        try:
            # Create capability descriptions for similarity matching
            capability_descriptions = {}
            for cap_type, keywords in self.capability_keywords.items():
                capability_descriptions[cap_type] = " ".join(keywords)
            
            # Vectorize texts
            all_texts = [text] + list(capability_descriptions.values())
            vectors = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:])
            
            # Map similarities to capabilities
            nlp_scores = {}
            for i, capability_type in enumerate(capability_descriptions.keys()):
                similarity = similarity_matrix[0][i]
                if similarity > 0.1:  # Threshold for relevance
                    nlp_scores[capability_type] = similarity
            
            return nlp_scores
            
        except Exception as e:
            self.logger.error(f"Error in NLP-based capability detection: {e}")
            return {}
    
    def _merge_capability_scores(self, keyword_scores: Dict[CapabilityType, float],
                               nlp_scores: Dict[CapabilityType, float]) -> Dict[CapabilityType, float]:
        """Merge capability scores from different detection methods"""
        merged_scores = {}
        
        all_capabilities = set(keyword_scores.keys()) | set(nlp_scores.keys())
        
        for capability in all_capabilities:
            keyword_score = keyword_scores.get(capability, 0.0)
            nlp_score = nlp_scores.get(capability, 0.0)
            
            # Weighted combination (keyword matching weighted higher)
            merged_score = (keyword_score * 0.7) + (nlp_score * 0.3)
            
            if merged_score > 0.05:  # Minimum threshold
                merged_scores[capability] = min(1.0, merged_score)
        
        return merged_scores
    
    def _extract_performance_characteristics(self, text: str) -> Dict[str, float]:
        """Extract performance characteristics from text"""
        characteristics = {}
        
        try:
            for category, patterns in self.performance_patterns.items():
                for characteristic, pattern in patterns.items():
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    if matches > 0:
                        # Score based on match frequency
                        score = min(1.0, matches * 0.3)
                        characteristics[f"{category}_{characteristic}"] = score
            
            # Add derived characteristics
            if 'latency_indicators_low_latency' in characteristics:
                characteristics['response_time_score'] = 0.9
            elif 'latency_indicators_high_latency' in characteristics:
                characteristics['response_time_score'] = 0.3
            else:
                characteristics['response_time_score'] = 0.6
            
            if 'throughput_indicators_high_throughput' in characteristics:
                characteristics['throughput_score'] = 0.9
            elif 'throughput_indicators_scalable_throughput' in characteristics:
                characteristics['throughput_score'] = 0.8
            else:
                characteristics['throughput_score'] = 0.5
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error extracting performance characteristics: {e}")
            return {'response_time_score': 0.5, 'throughput_score': 0.5}
    
    def _estimate_resource_requirements(self, text: str, 
                                      capabilities: Dict[CapabilityType, float]) -> Dict[str, float]:
        """Estimate resource requirements based on capabilities and text analysis"""
        requirements = {
            'cpu_cores': 1.0,
            'memory_gb': 1.0,
            'gpu_memory_gb': 0.0,
            'network_bandwidth_mbps': 10.0,
            'storage_gb': 1.0
        }
        
        try:
            # Base requirements on capabilities
            total_capability_score = sum(capabilities.values())
            complexity_multiplier = min(3.0, 1.0 + total_capability_score)
            
            # Adjust based on specific capabilities
            if CapabilityType.MACHINE_LEARNING in capabilities:
                ml_score = capabilities[CapabilityType.MACHINE_LEARNING]
                requirements['cpu_cores'] += ml_score * 2
                requirements['memory_gb'] += ml_score * 4
                requirements['gpu_memory_gb'] += ml_score * 2
            
            if CapabilityType.COMPUTER_VISION in capabilities:
                cv_score = capabilities[CapabilityType.COMPUTER_VISION]
                requirements['gpu_memory_gb'] += cv_score * 4
                requirements['memory_gb'] += cv_score * 2
            
            if CapabilityType.DATA_ANALYSIS in capabilities:
                da_score = capabilities[CapabilityType.DATA_ANALYSIS]
                requirements['memory_gb'] += da_score * 3
                requirements['storage_gb'] += da_score * 5
            
            # Apply complexity multiplier
            for resource in requirements:
                if resource != 'network_bandwidth_mbps':  # Network scales differently
                    requirements[resource] *= complexity_multiplier
            
            # Text-based adjustments
            if 'gpu' in text or 'cuda' in text:
                requirements['gpu_memory_gb'] = max(requirements['gpu_memory_gb'], 2.0)
            
            if 'distributed' in text or 'cluster' in text:
                requirements['network_bandwidth_mbps'] *= 2
                requirements['cpu_cores'] *= 1.5
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"Error estimating resource requirements: {e}")
            return requirements
    
    def _analyze_io_types(self, text: str) -> Tuple[List[str], List[str]]:
        """Analyze input and output types from text description"""
        input_types = []
        output_types = []
        
        try:
            # Common input types
            input_patterns = {
                'text': r'(text|string|document|message|content)',
                'image': r'(image|picture|photo|visual|png|jpg|jpeg)',
                'audio': r'(audio|sound|voice|speech|wav|mp3)',
                'video': r'(video|movie|stream|mp4|avi)',
                'json': r'(json|api|structured|data)',
                'csv': r'(csv|tabular|spreadsheet|table)',
                'xml': r'(xml|markup|structured)',
                'binary': r'(binary|file|upload|attachment)'
            }
            
            # Common output types
            output_patterns = {
                'classification': r'(classif|category|label|tag)',
                'score': r'(score|rating|confidence|probability)',
                'text': r'(text|generate|content|summary)',
                'json': r'(json|structured|api|response)',
                'visualization': r'(chart|graph|plot|visual)',
                'recommendation': r'(recommend|suggest|propose)',
                'prediction': r'(predict|forecast|estimate)',
                'detection': r'(detect|identify|recognize|find)'
            }
            
            # Detect input types
            for input_type, pattern in input_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    input_types.append(input_type)
            
            # Detect output types
            for output_type, pattern in output_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    output_types.append(output_type)
            
            # Default types if none detected
            if not input_types:
                input_types = ['json']
            if not output_types:
                output_types = ['json']
            
            return input_types, output_types
            
        except Exception as e:
            self.logger.error(f"Error analyzing I/O types: {e}")
            return ['json'], ['json']
    
    def _estimate_processing_time(self, performance_characteristics: Dict[str, float]) -> float:
        """Estimate average processing time"""
        base_time = 1.0  # 1 second base
        
        response_score = performance_characteristics.get('response_time_score', 0.5)
        
        # Lower response score = higher processing time
        estimated_time = base_time / max(0.1, response_score)
        
        return min(60.0, estimated_time)  # Cap at 60 seconds
    
    def _estimate_accuracy(self, capabilities: Dict[CapabilityType, float], text: str) -> float:
        """Estimate system accuracy"""
        base_accuracy = 0.8
        
        # Higher capability scores generally indicate more sophisticated systems
        if capabilities:
            avg_capability = sum(capabilities.values()) / len(capabilities)
            accuracy_boost = avg_capability * 0.15
            base_accuracy += accuracy_boost
        
        # Text-based adjustments
        if re.search(r'(accurate|precise|state.of.art|best.in.class)', text, re.IGNORECASE):
            base_accuracy += 0.1
        
        if re.search(r'(approximate|rough|quick)', text, re.IGNORECASE):
            base_accuracy -= 0.1
        
        return min(0.99, max(0.5, base_accuracy))
    
    def _estimate_reliability(self, text: str) -> float:
        """Estimate system reliability"""
        base_reliability = 0.85
        
        # Text-based reliability indicators
        if re.search(r'(reliable|stable|robust|fault.tolerant)', text, re.IGNORECASE):
            base_reliability += 0.1
        
        if re.search(r'(experimental|beta|prototype|unstable)', text, re.IGNORECASE):
            base_reliability -= 0.2
        
        if re.search(r'(enterprise|production|commercial)', text, re.IGNORECASE):
            base_reliability += 0.05
        
        return min(0.99, max(0.3, base_reliability))
    
    def _estimate_scalability(self, text: str) -> float:
        """Estimate system scalability"""
        base_scalability = 0.7
        
        # Scalability indicators
        if re.search(r'(scalable|elastic|auto.scale|distributed)', text, re.IGNORECASE):
            base_scalability += 0.2
        
        if re.search(r'(cloud|kubernetes|docker|microservice)', text, re.IGNORECASE):
            base_scalability += 0.15
        
        if re.search(r'(single|standalone|monolith)', text, re.IGNORECASE):
            base_scalability -= 0.2
        
        return min(1.0, max(0.2, base_scalability))
    
    def _estimate_cost(self, resource_requirements: Dict[str, float]) -> float:
        """Estimate cost per operation based on resource requirements"""
        # Simple cost model based on resource usage
        cpu_cost = resource_requirements.get('cpu_cores', 1.0) * 0.01
        memory_cost = resource_requirements.get('memory_gb', 1.0) * 0.005
        gpu_cost = resource_requirements.get('gpu_memory_gb', 0.0) * 0.05
        
        total_cost = cpu_cost + memory_cost + gpu_cost
        return max(0.001, total_cost)
    
    def _assess_documentation_quality(self, text: str) -> float:
        """Assess quality of system documentation"""
        quality_score = 0.5
        
        # Length indicates detail level
        text_length = len(text)
        if text_length > 500:
            quality_score += 0.2
        elif text_length > 200:
            quality_score += 0.1
        
        # Quality indicators
        if re.search(r'(example|tutorial|guide|documentation)', text, re.IGNORECASE):
            quality_score += 0.15
        
        if re.search(r'(api|endpoint|parameter|response)', text, re.IGNORECASE):
            quality_score += 0.1
        
        if re.search(r'(version|release|update)', text, re.IGNORECASE):
            quality_score += 0.05
        
        return min(1.0, quality_score)
    
    def _create_fallback_profile(self, registration: IntelligenceSystemRegistration) -> CapabilityProfile:
        """Create fallback profile when discovery fails"""
        return CapabilityProfile(
            system_id=registration.system_id,
            system_name=registration.system_name,
            capabilities={CapabilityType.DATA_ANALYSIS: 0.5},  # Default capability
            performance_characteristics={'response_time_score': 0.5, 'throughput_score': 0.5},
            resource_requirements={
                'cpu_cores': 1.0, 'memory_gb': 1.0, 'gpu_memory_gb': 0.0,
                'network_bandwidth_mbps': 10.0, 'storage_gb': 1.0
            },
            input_types=['json'],
            output_types=['json'],
            processing_time=5.0,
            accuracy=0.7,
            reliability=0.8,
            scalability=0.6,
            cost_per_operation=0.01,
            api_endpoints=registration.api_endpoints,
            documentation_quality=0.5
        )
    
    def create_capability_clusters(self, systems: List[CapabilityProfile]) -> List[CapabilityCluster]:
        """Create clusters of systems with similar capabilities"""
        if not HAS_SKLEARN or len(systems) < 2:
            return []
        
        try:
            # Prepare capability vectors
            capability_vectors = []
            system_ids = []
            
            for system in systems:
                vector = []
                for cap_type in CapabilityType:
                    vector.append(system.capabilities.get(cap_type, 0.0))
                capability_vectors.append(vector)
                system_ids.append(system.system_id)
            
            # Determine optimal number of clusters
            n_clusters = min(5, max(2, len(systems) // 3))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(capability_vectors)
            
            # Create cluster objects
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_systems = [system_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if cluster_systems:
                    # Find shared capabilities
                    shared_capabilities = self._find_shared_capabilities(
                        [s for s in systems if s.system_id in cluster_systems]
                    )
                    
                    cluster = CapabilityCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        cluster_name=f"Capability Cluster {cluster_id + 1}",
                        member_systems=cluster_systems,
                        shared_capabilities=shared_capabilities,
                        cluster_characteristics=self._analyze_cluster_characteristics(
                            [s for s in systems if s.system_id in cluster_systems]
                        ),
                        inter_cluster_relationships={},
                        optimization_opportunities=self._identify_cluster_optimizations(
                            [s for s in systems if s.system_id in cluster_systems]
                        )
                    )
                    
                    clusters.append(cluster)
                    self.capability_clusters[cluster.cluster_id] = cluster
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error creating capability clusters: {e}")
            return []
    
    def _find_shared_capabilities(self, systems: List[CapabilityProfile]) -> List[CapabilityType]:
        """Find capabilities shared by systems in cluster"""
        if not systems:
            return []
        
        # Find capabilities present in all systems
        shared_capabilities = []
        for cap_type in CapabilityType:
            if all(cap_type in system.capabilities and system.capabilities[cap_type] > 0.3 
                   for system in systems):
                shared_capabilities.append(cap_type)
        
        return shared_capabilities
    
    def _analyze_cluster_characteristics(self, systems: List[CapabilityProfile]) -> Dict[str, Any]:
        """Analyze characteristics of capability cluster"""
        if not systems:
            return {}
        
        characteristics = {
            'average_processing_time': sum(s.processing_time for s in systems) / len(systems),
            'average_accuracy': sum(s.accuracy for s in systems) / len(systems),
            'average_reliability': sum(s.reliability for s in systems) / len(systems),
            'average_scalability': sum(s.scalability for s in systems) / len(systems),
            'total_cost': sum(s.cost_per_operation for s in systems),
            'dominant_input_types': self._find_dominant_io_types([s.input_types for s in systems]),
            'dominant_output_types': self._find_dominant_io_types([s.output_types for s in systems])
        }
        
        return characteristics
    
    def _find_dominant_io_types(self, io_type_lists: List[List[str]]) -> List[str]:
        """Find dominant I/O types across systems"""
        all_types = []
        for type_list in io_type_lists:
            all_types.extend(type_list)
        
        # Count occurrences
        type_counts = Counter(all_types)
        
        # Return types that appear in majority of systems
        threshold = len(io_type_lists) // 2
        dominant_types = [io_type for io_type, count in type_counts.items() if count > threshold]
        
        return dominant_types[:5]  # Limit to top 5
    
    def _identify_cluster_optimizations(self, systems: List[CapabilityProfile]) -> List[str]:
        """Identify optimization opportunities for cluster"""
        optimizations = []
        
        if len(systems) > 1:
            # Check for redundant capabilities
            capability_overlap = self._calculate_capability_overlap(systems)
            if capability_overlap > 0.7:
                optimizations.append("High capability overlap - consider load balancing")
            
            # Check for complementary capabilities
            if self._has_complementary_capabilities(systems):
                optimizations.append("Complementary capabilities - consider pipeline orchestration")
            
            # Check resource utilization
            total_cost = sum(s.cost_per_operation for s in systems)
            if total_cost > 0.1:  # Threshold for high cost
                optimizations.append("High combined cost - consider resource sharing")
        
        return optimizations
    
    def _calculate_capability_overlap(self, systems: List[CapabilityProfile]) -> float:
        """Calculate capability overlap between systems"""
        if len(systems) < 2:
            return 0.0
        
        overlaps = []
        for i, system_a in enumerate(systems):
            for system_b in systems[i+1:]:
                shared_caps = set(system_a.capabilities.keys()) & set(system_b.capabilities.keys())
                total_caps = set(system_a.capabilities.keys()) | set(system_b.capabilities.keys())
                
                if total_caps:
                    overlap = len(shared_caps) / len(total_caps)
                    overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0
    
    def _has_complementary_capabilities(self, systems: List[CapabilityProfile]) -> bool:
        """Check if systems have complementary capabilities"""
        if not HAS_NETWORKX or not self.capability_graph:
            return False
        
        try:
            # Check if systems have capabilities that are connected in the graph
            system_capabilities = []
            for system in systems:
                system_caps = [cap.value for cap in system.capabilities.keys()]
                system_capabilities.append(system_caps)
            
            # Look for connections between different systems' capabilities
            for caps_a in system_capabilities:
                for caps_b in system_capabilities:
                    if caps_a != caps_b:
                        for cap_a in caps_a:
                            for cap_b in caps_b:
                                if self.capability_graph.has_edge(cap_a, cap_b):
                                    return True
            
            return False
            
        except Exception:
            return False


def create_capability_mapper() -> IntelligenceCapabilityMapper:
    """Factory function to create capability mapper"""
    return IntelligenceCapabilityMapper()