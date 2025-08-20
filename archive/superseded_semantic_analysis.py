"""
Semantic Code Analysis Module
Analyzes code semantics to understand developer intent and purpose
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import difflib
from enum import Enum

from testmaster.analysis.base_analyzer import BaseAnalyzer


class IntentType(Enum):
    """Types of developer intent"""
    DATA_PROCESSING = "data_processing"
    API_ENDPOINT = "api_endpoint"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    PERSISTENCE = "persistence"
    TRANSFORMATION = "transformation"
    CALCULATION = "calculation"
    ORCHESTRATION = "orchestration"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    CONFIGURATION = "configuration"
    TESTING = "testing"
    UTILITY = "utility"
    UNKNOWN = "unknown"


@dataclass
class SemanticIntent:
    """Represents the semantic intent of a code element"""
    element_type: str  # function, class, module
    name: str
    location: str
    primary_intent: IntentType
    secondary_intents: List[IntentType]
    confidence: float
    semantic_signature: str
    context: Dict[str, Any]
    relationships: List[str]


@dataclass
class ConceptualPattern:
    """Represents a conceptual pattern in code"""
    pattern_name: str
    description: str
    occurrences: List[str]
    semantic_role: str
    implementation_quality: float
    alternatives: List[str]


class SemanticAnalyzer(BaseAnalyzer):
    """
    Analyzes code semantics to understand intent and purpose
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Intent indicators
        self.intent_keywords = {
            IntentType.DATA_PROCESSING: [
                "process", "transform", "parse", "extract", "clean", "normalize",
                "aggregate", "filter", "map", "reduce", "sort", "group"
            ],
            IntentType.API_ENDPOINT: [
                "get", "post", "put", "delete", "patch", "route", "endpoint",
                "api", "rest", "graphql", "handler", "controller"
            ],
            IntentType.AUTHENTICATION: [
                "auth", "login", "logout", "token", "session", "credential",
                "permission", "role", "user", "password", "jwt", "oauth"
            ],
            IntentType.VALIDATION: [
                "validate", "check", "verify", "ensure", "assert", "sanitize",
                "is_valid", "has_", "can_", "should_", "must_"
            ],
            IntentType.PERSISTENCE: [
                "save", "load", "store", "retrieve", "database", "db", "query",
                "insert", "update", "delete", "fetch", "persist", "cache"
            ],
            IntentType.TRANSFORMATION: [
                "convert", "transform", "translate", "map", "serialize",
                "deserialize", "encode", "decode", "format", "parse"
            ],
            IntentType.CALCULATION: [
                "calculate", "compute", "sum", "average", "total", "count",
                "measure", "evaluate", "score", "rate", "analyze"
            ],
            IntentType.ORCHESTRATION: [
                "orchestrate", "coordinate", "manage", "control", "execute",
                "run", "start", "stop", "schedule", "dispatch", "delegate"
            ],
            IntentType.ERROR_HANDLING: [
                "error", "exception", "handle", "catch", "raise", "throw",
                "retry", "recover", "fallback", "fail", "rescue"
            ],
            IntentType.LOGGING: [
                "log", "trace", "debug", "info", "warn", "error", "audit",
                "record", "track", "monitor", "report"
            ],
            IntentType.CONFIGURATION: [
                "config", "setting", "option", "parameter", "environment",
                "initialize", "setup", "configure", "register"
            ],
            IntentType.TESTING: [
                "test", "assert", "mock", "stub", "fixture", "setup", "teardown",
                "expect", "should", "verify", "check"
            ],
            IntentType.UTILITY: [
                "util", "helper", "tool", "common", "shared", "generic",
                "utility", "misc", "general", "support"
            ]
        }
        
        # Semantic patterns
        self.semantic_patterns = {
            "factory": {
                "indicators": ["create", "build", "make", "construct", "factory"],
                "role": "object_creation"
            },
            "singleton": {
                "indicators": ["instance", "singleton", "get_instance"],
                "role": "single_instance"
            },
            "observer": {
                "indicators": ["subscribe", "notify", "observe", "listen", "event"],
                "role": "event_handling"
            },
            "strategy": {
                "indicators": ["strategy", "algorithm", "policy", "behavior"],
                "role": "algorithm_selection"
            },
            "adapter": {
                "indicators": ["adapt", "wrapper", "bridge", "translator"],
                "role": "interface_adaptation"
            },
            "decorator": {
                "indicators": ["decorate", "wrap", "enhance", "extend"],
                "role": "behavior_extension"
            },
            "repository": {
                "indicators": ["repository", "dao", "store", "persistence"],
                "role": "data_access"
            },
            "service": {
                "indicators": ["service", "manager", "handler", "processor"],
                "role": "business_logic"
            },
            "controller": {
                "indicators": ["controller", "endpoint", "route", "view"],
                "role": "request_handling"
            },
            "model": {
                "indicators": ["model", "entity", "domain", "dto"],
                "role": "data_representation"
            }
        }
        
        # Conceptual relationships
        self.relationship_types = {
            "uses": "Direct usage/dependency",
            "implements": "Interface implementation",
            "extends": "Inheritance relationship",
            "composes": "Composition relationship",
            "delegates": "Delegation pattern",
            "transforms": "Data transformation",
            "validates": "Validation relationship",
            "orchestrates": "Controls execution",
            "handles": "Error/event handling",
            "configures": "Configuration relationship"
        }
        
        self.semantic_intents = []
        self.conceptual_patterns = []
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis
        """
        results = {
            "intent_recognition": self._recognize_intent(),
            "semantic_signatures": self._extract_semantic_signatures(),
            "conceptual_patterns": self._identify_conceptual_patterns(),
            "semantic_relationships": self._analyze_semantic_relationships(),
            "purpose_classification": self._classify_code_purpose(),
            "naming_semantics": self._analyze_naming_semantics(),
            "behavioral_patterns": self._identify_behavioral_patterns(),
            "domain_concepts": self._extract_domain_concepts(),
            "semantic_clustering": self._perform_semantic_clustering(),
            "intent_consistency": self._check_intent_consistency(),
            "semantic_smells": self._detect_semantic_smells(),
            "documentation_alignment": self._check_documentation_alignment(),
            "semantic_evolution": self._analyze_semantic_evolution(),
            "recommendations": self._generate_semantic_recommendations(),
            "summary": self._generate_semantic_summary()
        }
        
        return results
    
    def _recognize_intent(self) -> Dict[str, Any]:
        """
        Recognize the intent behind code elements
        """
        intent_recognition = {
            "function_intents": [],
            "class_intents": [],
            "module_intents": [],
            "intent_distribution": defaultdict(int),
            "confidence_scores": [],
            "ambiguous_intents": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Analyze module-level intent
                    module_intent = self._analyze_module_intent(tree, file_path)
                    if module_intent:
                        intent_recognition["module_intents"].append(module_intent)
                        self.semantic_intents.append(module_intent)
                    
                    for node in ast.walk(tree):
                        # Analyze function intent
                        if isinstance(node, ast.FunctionDef):
                            func_intent = self._analyze_function_intent(node, file_path)
                            if func_intent:
                                intent_recognition["function_intents"].append(func_intent)
                                self.semantic_intents.append(func_intent)
                                intent_recognition["intent_distribution"][func_intent.primary_intent.value] += 1
                                intent_recognition["confidence_scores"].append(func_intent.confidence)
                                
                                # Track ambiguous intents
                                if func_intent.confidence < 0.6:
                                    intent_recognition["ambiguous_intents"].append({
                                        "element": func_intent.name,
                                        "location": func_intent.location,
                                        "possible_intents": [func_intent.primary_intent.value] + 
                                                          [i.value for i in func_intent.secondary_intents],
                                        "confidence": func_intent.confidence
                                    })
                        
                        # Analyze class intent
                        elif isinstance(node, ast.ClassDef):
                            class_intent = self._analyze_class_intent(node, file_path)
                            if class_intent:
                                intent_recognition["class_intents"].append(class_intent)
                                self.semantic_intents.append(class_intent)
                                
            except Exception as e:
                self.logger.error(f"Error recognizing intent in {file_path}: {e}")
        
        return intent_recognition
    
    def _extract_semantic_signatures(self) -> Dict[str, Any]:
        """
        Extract semantic signatures that describe code behavior
        """
        signatures = {
            "function_signatures": [],
            "class_signatures": [],
            "semantic_fingerprints": {},
            "signature_patterns": defaultdict(list),
            "unique_signatures": set()
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Create semantic signature
                            signature = self._create_semantic_signature(node)
                            signatures["function_signatures"].append({
                                "function": node.name,
                                "location": str(file_path),
                                "signature": signature,
                                "components": self._decompose_signature(signature)
                            })
                            signatures["unique_signatures"].add(signature)
                            
                            # Group by pattern
                            pattern = self._identify_signature_pattern(signature)
                            signatures["signature_patterns"][pattern].append(node.name)
                            
                        elif isinstance(node, ast.ClassDef):
                            # Create class semantic signature
                            class_sig = self._create_class_signature(node)
                            signatures["class_signatures"].append({
                                "class": node.name,
                                "location": str(file_path),
                                "signature": class_sig,
                                "methods": self._get_class_methods(node)
                            })
                            
            except Exception as e:
                self.logger.error(f"Error extracting signatures from {file_path}: {e}")
        
        # Create semantic fingerprints for modules
        signatures["semantic_fingerprints"] = self._create_module_fingerprints(signatures)
        
        return signatures
    
    def _identify_conceptual_patterns(self) -> Dict[str, Any]:
        """
        Identify high-level conceptual patterns in code
        """
        patterns = {
            "design_patterns": [],
            "architectural_patterns": [],
            "domain_patterns": [],
            "idioms": [],
            "pattern_quality": {},
            "pattern_distribution": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Check for design patterns
                    for pattern_name, pattern_info in self.semantic_patterns.items():
                        if self._matches_pattern(tree, pattern_info):
                            pattern = ConceptualPattern(
                                pattern_name=pattern_name,
                                description=f"{pattern_name.title()} pattern",
                                occurrences=[str(file_path)],
                                semantic_role=pattern_info["role"],
                                implementation_quality=self._assess_pattern_quality(tree, pattern_name),
                                alternatives=self._suggest_pattern_alternatives(pattern_name)
                            )
                            patterns["design_patterns"].append(pattern)
                            self.conceptual_patterns.append(pattern)
                            patterns["pattern_distribution"][pattern_name] += 1
                    
                    # Check for architectural patterns
                    arch_patterns = self._identify_architectural_patterns(tree, file_path)
                    patterns["architectural_patterns"].extend(arch_patterns)
                    
                    # Check for domain patterns
                    domain_patterns = self._identify_domain_patterns(tree, file_path)
                    patterns["domain_patterns"].extend(domain_patterns)
                    
                    # Check for Python idioms
                    idioms = self._identify_python_idioms(tree, file_path)
                    patterns["idioms"].extend(idioms)
                    
            except Exception as e:
                self.logger.error(f"Error identifying patterns in {file_path}: {e}")
        
        # Assess pattern quality
        for pattern in patterns["design_patterns"]:
            patterns["pattern_quality"][pattern.pattern_name] = {
                "quality_score": pattern.implementation_quality,
                "occurrences": len(pattern.occurrences),
                "consistency": self._assess_pattern_consistency(pattern)
            }
        
        return patterns
    
    def _analyze_semantic_relationships(self) -> Dict[str, Any]:
        """
        Analyze semantic relationships between code elements
        """
        relationships = {
            "direct_relationships": [],
            "conceptual_relationships": [],
            "dependency_graph": defaultdict(list),
            "relationship_types": defaultdict(int),
            "coupling_analysis": {},
            "cohesion_analysis": {}
        }
        
        # Build relationship graph from semantic intents
        for intent in self.semantic_intents:
            for related in intent.relationships:
                relationships["dependency_graph"][intent.name].append(related)
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Analyze direct relationships
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Find what this function uses/calls
                            called_functions = self._extract_function_calls(node)
                            for called in called_functions:
                                relationships["direct_relationships"].append({
                                    "from": node.name,
                                    "to": called,
                                    "type": "uses",
                                    "location": str(file_path)
                                })
                                relationships["relationship_types"]["uses"] += 1
                        
                        elif isinstance(node, ast.ClassDef):
                            # Analyze inheritance
                            for base in node.bases:
                                if isinstance(base, ast.Name):
                                    relationships["direct_relationships"].append({
                                        "from": node.name,
                                        "to": base.id,
                                        "type": "extends",
                                        "location": str(file_path)
                                    })
                                    relationships["relationship_types"]["extends"] += 1
                    
                    # Identify conceptual relationships
                    conceptual = self._identify_conceptual_relationships(tree, file_path)
                    relationships["conceptual_relationships"].extend(conceptual)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing relationships in {file_path}: {e}")
        
        # Analyze coupling and cohesion
        relationships["coupling_analysis"] = self._analyze_coupling(relationships["dependency_graph"])
        relationships["cohesion_analysis"] = self._analyze_cohesion(self.semantic_intents)
        
        return relationships
    
    def _classify_code_purpose(self) -> Dict[str, Any]:
        """
        Classify the overall purpose of code sections
        """
        purpose_classification = {
            "primary_purposes": [],
            "secondary_purposes": [],
            "purpose_distribution": defaultdict(int),
            "module_purposes": {},
            "functional_areas": []
        }
        
        # Aggregate intents to determine purposes
        for intent in self.semantic_intents:
            if intent.element_type == "module":
                purpose_classification["module_purposes"][intent.name] = {
                    "primary": intent.primary_intent.value,
                    "secondary": [i.value for i in intent.secondary_intents],
                    "confidence": intent.confidence
                }
            
            purpose_classification["purpose_distribution"][intent.primary_intent.value] += 1
        
        # Identify primary purposes (most common)
        sorted_purposes = sorted(
            purpose_classification["purpose_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if sorted_purposes:
            purpose_classification["primary_purposes"] = [p[0] for p in sorted_purposes[:3]]
            purpose_classification["secondary_purposes"] = [p[0] for p in sorted_purposes[3:6]]
        
        # Identify functional areas
        functional_areas = self._identify_functional_areas(self.semantic_intents)
        purpose_classification["functional_areas"] = functional_areas
        
        return purpose_classification
    
    def _analyze_naming_semantics(self) -> Dict[str, Any]:
        """
        Analyze the semantic meaning of names
        """
        naming_semantics = {
            "naming_patterns": [],
            "semantic_consistency": {},
            "naming_conventions": {},
            "semantic_mismatches": [],
            "suggested_renames": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Analyze function name semantics
                            name_analysis = self._analyze_name_semantics(node.name, "function")
                            
                            # Check if name matches intent
                            func_intent = self._find_intent_for_element(node.name)
                            if func_intent:
                                consistency = self._check_name_intent_consistency(
                                    node.name, func_intent.primary_intent
                                )
                                naming_semantics["semantic_consistency"][node.name] = consistency
                                
                                if consistency < 0.6:
                                    naming_semantics["semantic_mismatches"].append({
                                        "name": node.name,
                                        "location": str(file_path),
                                        "expected_pattern": self._get_naming_pattern_for_intent(func_intent.primary_intent),
                                        "actual_pattern": name_analysis["pattern"]
                                    })
                                    
                                    # Suggest better name
                                    suggestion = self._suggest_semantic_name(
                                        node.name, func_intent.primary_intent
                                    )
                                    if suggestion != node.name:
                                        naming_semantics["suggested_renames"].append({
                                            "current": node.name,
                                            "suggested": suggestion,
                                            "reason": "Better semantic alignment"
                                        })
                        
                        elif isinstance(node, ast.ClassDef):
                            # Analyze class name semantics
                            class_analysis = self._analyze_name_semantics(node.name, "class")
                            naming_semantics["naming_patterns"].append(class_analysis)
                            
            except Exception as e:
                self.logger.error(f"Error analyzing naming semantics in {file_path}: {e}")
        
        # Identify naming conventions
        naming_semantics["naming_conventions"] = self._identify_naming_conventions(
            naming_semantics["naming_patterns"]
        )
        
        return naming_semantics
    
    def _identify_behavioral_patterns(self) -> Dict[str, Any]:
        """
        Identify behavioral patterns in code
        """
        behavioral_patterns = {
            "control_flow_patterns": [],
            "data_flow_patterns": [],
            "interaction_patterns": [],
            "state_patterns": [],
            "error_patterns": [],
            "pattern_sequences": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Analyze control flow
                            control_flow = self._analyze_control_flow_pattern(node)
                            if control_flow:
                                behavioral_patterns["control_flow_patterns"].append({
                                    "function": node.name,
                                    "pattern": control_flow,
                                    "location": str(file_path)
                                })
                            
                            # Analyze data flow
                            data_flow = self._analyze_data_flow_pattern(node)
                            if data_flow:
                                behavioral_patterns["data_flow_patterns"].append({
                                    "function": node.name,
                                    "pattern": data_flow,
                                    "location": str(file_path)
                                })
                            
                            # Analyze error handling patterns
                            error_pattern = self._analyze_error_pattern(node)
                            if error_pattern:
                                behavioral_patterns["error_patterns"].append({
                                    "function": node.name,
                                    "pattern": error_pattern,
                                    "location": str(file_path)
                                })
                        
                        elif isinstance(node, ast.ClassDef):
                            # Analyze state patterns
                            state_pattern = self._analyze_state_pattern(node)
                            if state_pattern:
                                behavioral_patterns["state_patterns"].append({
                                    "class": node.name,
                                    "pattern": state_pattern,
                                    "location": str(file_path)
                                })
                            
                            # Analyze interaction patterns
                            interaction = self._analyze_interaction_pattern(node)
                            if interaction:
                                behavioral_patterns["interaction_patterns"].append({
                                    "class": node.name,
                                    "pattern": interaction,
                                    "location": str(file_path)
                                })
                    
            except Exception as e:
                self.logger.error(f"Error identifying behavioral patterns in {file_path}: {e}")
        
        # Identify pattern sequences
        behavioral_patterns["pattern_sequences"] = self._identify_pattern_sequences(
            behavioral_patterns
        )
        
        return behavioral_patterns
    
    def _extract_domain_concepts(self) -> Dict[str, Any]:
        """
        Extract domain-specific concepts from code
        """
        domain_concepts = {
            "entities": [],
            "value_objects": [],
            "services": [],
            "repositories": [],
            "domain_events": [],
            "ubiquitous_language": {},
            "bounded_contexts": []
        }
        
        # Domain-specific keywords to look for
        domain_indicators = {
            "entity": ["entity", "model", "aggregate", "root"],
            "value_object": ["value", "vo", "immutable"],
            "service": ["service", "handler", "processor", "manager"],
            "repository": ["repository", "dao", "store", "persistence"],
            "event": ["event", "occurred", "happened", "raised"]
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Classify domain concept
                            concept_type = self._classify_domain_concept(node, domain_indicators)
                            
                            if concept_type == "entity":
                                domain_concepts["entities"].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "attributes": self._extract_entity_attributes(node),
                                    "behaviors": self._extract_entity_behaviors(node)
                                })
                            elif concept_type == "value_object":
                                domain_concepts["value_objects"].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "immutable": self._is_immutable(node)
                                })
                            elif concept_type == "service":
                                domain_concepts["services"].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "operations": self._extract_service_operations(node)
                                })
                            elif concept_type == "repository":
                                domain_concepts["repositories"].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "entity": self._find_repository_entity(node)
                                })
                            elif concept_type == "event":
                                domain_concepts["domain_events"].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "payload": self._extract_event_payload(node)
                                })
                    
                    # Extract ubiquitous language
                    language_terms = self._extract_ubiquitous_language(tree, file_path)
                    for term, definition in language_terms.items():
                        domain_concepts["ubiquitous_language"][term] = definition
                    
            except Exception as e:
                self.logger.error(f"Error extracting domain concepts from {file_path}: {e}")
        
        # Identify bounded contexts
        domain_concepts["bounded_contexts"] = self._identify_bounded_contexts(domain_concepts)
        
        return domain_concepts
    
    def _perform_semantic_clustering(self) -> Dict[str, Any]:
        """
        Cluster code elements by semantic similarity
        """
        clustering = {
            "semantic_clusters": [],
            "cluster_cohesion": {},
            "outliers": [],
            "suggested_reorganization": []
        }
        
        # Create similarity matrix
        similarity_matrix = self._create_similarity_matrix(self.semantic_intents)
        
        # Perform clustering (simplified - would use real clustering algorithm)
        clusters = self._cluster_by_similarity(similarity_matrix, threshold=0.7)
        
        for cluster_id, members in clusters.items():
            cluster = {
                "id": cluster_id,
                "members": members,
                "dominant_intent": self._find_dominant_intent(members),
                "cohesion_score": self._calculate_cluster_cohesion(members),
                "description": self._describe_cluster(members)
            }
            clustering["semantic_clusters"].append(cluster)
            clustering["cluster_cohesion"][cluster_id] = cluster["cohesion_score"]
        
        # Identify outliers
        clustering["outliers"] = self._identify_semantic_outliers(
            self.semantic_intents, clusters
        )
        
        # Suggest reorganization
        clustering["suggested_reorganization"] = self._suggest_semantic_reorganization(
            clusters, clustering["outliers"]
        )
        
        return clustering
    
    def _check_intent_consistency(self) -> Dict[str, Any]:
        """
        Check consistency of intent across codebase
        """
        consistency = {
            "consistency_score": 0,
            "inconsistent_elements": [],
            "naming_intent_mismatches": [],
            "behavioral_inconsistencies": [],
            "cross_module_consistency": {}
        }
        
        # Check naming vs intent consistency
        for intent in self.semantic_intents:
            expected_pattern = self._get_naming_pattern_for_intent(intent.primary_intent)
            if not self._name_matches_pattern(intent.name, expected_pattern):
                consistency["naming_intent_mismatches"].append({
                    "element": intent.name,
                    "location": intent.location,
                    "expected_intent": intent.primary_intent.value,
                    "naming_suggests": self._infer_intent_from_name(intent.name).value
                })
        
        # Check behavioral consistency
        similar_functions = self._find_similar_functions(self.semantic_intents)
        for group in similar_functions:
            if not self._check_behavioral_consistency(group):
                consistency["behavioral_inconsistencies"].append({
                    "functions": [f.name for f in group],
                    "issue": "Similar functions with different implementations"
                })
        
        # Check cross-module consistency
        modules = defaultdict(list)
        for intent in self.semantic_intents:
            module = self._get_module_from_location(intent.location)
            modules[module].append(intent)
        
        for module, intents in modules.items():
            consistency["cross_module_consistency"][module] = {
                "intent_variety": len(set(i.primary_intent for i in intents)),
                "cohesion": self._calculate_module_cohesion(intents)
            }
        
        # Calculate overall consistency score
        total_elements = len(self.semantic_intents)
        consistent_elements = total_elements - len(consistency["naming_intent_mismatches"])
        consistency["consistency_score"] = (consistent_elements / total_elements * 100) if total_elements > 0 else 0
        
        return consistency
    
    def _detect_semantic_smells(self) -> Dict[str, Any]:
        """
        Detect semantic code smells
        """
        semantic_smells = {
            "misleading_names": [],
            "ambiguous_purpose": [],
            "semantic_duplication": [],
            "inappropriate_abstractions": [],
            "concept_confusion": [],
            "total_smells": 0
        }
        
        for intent in self.semantic_intents:
            # Check for misleading names
            if self._is_misleading_name(intent):
                semantic_smells["misleading_names"].append({
                    "element": intent.name,
                    "location": intent.location,
                    "actual_purpose": intent.primary_intent.value,
                    "name_suggests": self._infer_intent_from_name(intent.name).value
                })
            
            # Check for ambiguous purpose
            if intent.confidence < 0.5 or len(intent.secondary_intents) > 3:
                semantic_smells["ambiguous_purpose"].append({
                    "element": intent.name,
                    "location": intent.location,
                    "possible_purposes": [intent.primary_intent.value] + 
                                       [i.value for i in intent.secondary_intents],
                    "confidence": intent.confidence
                })
        
        # Check for semantic duplication
        duplicates = self._find_semantic_duplicates(self.semantic_intents)
        for dup_group in duplicates:
            semantic_smells["semantic_duplication"].append({
                "elements": [d.name for d in dup_group],
                "locations": [d.location for d in dup_group],
                "shared_purpose": dup_group[0].primary_intent.value
            })
        
        # Check for inappropriate abstractions
        for pattern in self.conceptual_patterns:
            if pattern.implementation_quality < 0.5:
                semantic_smells["inappropriate_abstractions"].append({
                    "pattern": pattern.pattern_name,
                    "locations": pattern.occurrences,
                    "quality": pattern.implementation_quality,
                    "issue": "Poor pattern implementation"
                })
        
        # Calculate total smells
        semantic_smells["total_smells"] = sum(
            len(v) for k, v in semantic_smells.items() if k != "total_smells"
        )
        
        return semantic_smells
    
    def _check_documentation_alignment(self) -> Dict[str, Any]:
        """
        Check if documentation aligns with semantic intent
        """
        alignment = {
            "aligned_elements": [],
            "misaligned_elements": [],
            "missing_documentation": [],
            "alignment_score": 0,
            "recommendations": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            docstring = ast.get_docstring(node)
                            intent = self._find_intent_for_element(node.name)
                            
                            if intent:
                                if docstring:
                                    # Check alignment
                                    alignment_score = self._check_doc_intent_alignment(
                                        docstring, intent
                                    )
                                    
                                    if alignment_score > 0.7:
                                        alignment["aligned_elements"].append({
                                            "element": node.name,
                                            "score": alignment_score
                                        })
                                    else:
                                        alignment["misaligned_elements"].append({
                                            "element": node.name,
                                            "location": str(file_path),
                                            "documented_as": self._extract_doc_purpose(docstring),
                                            "actual_intent": intent.primary_intent.value,
                                            "score": alignment_score
                                        })
                                else:
                                    alignment["missing_documentation"].append({
                                        "element": node.name,
                                        "location": str(file_path),
                                        "intent": intent.primary_intent.value,
                                        "suggested_doc": self._generate_doc_for_intent(intent)
                                    })
                            
            except Exception as e:
                self.logger.error(f"Error checking documentation alignment in {file_path}: {e}")
        
        # Calculate alignment score
        total = len(alignment["aligned_elements"]) + len(alignment["misaligned_elements"]) + len(alignment["missing_documentation"])
        if total > 0:
            alignment["alignment_score"] = (len(alignment["aligned_elements"]) / total) * 100
        
        # Generate recommendations
        if alignment["misaligned_elements"]:
            alignment["recommendations"].append({
                "action": "Update documentation to reflect actual intent",
                "targets": [e["element"] for e in alignment["misaligned_elements"][:5]]
            })
        
        if alignment["missing_documentation"]:
            alignment["recommendations"].append({
                "action": "Add documentation for undocumented elements",
                "targets": [e["element"] for e in alignment["missing_documentation"][:5]]
            })
        
        return alignment
    
    def _analyze_semantic_evolution(self) -> Dict[str, Any]:
        """
        Analyze how semantic intent evolves (would need git history)
        """
        evolution = {
            "intent_stability": {},
            "semantic_drift": [],
            "refactoring_patterns": [],
            "evolution_trends": []
        }
        
        # This would analyze git history to track semantic changes
        # For now, return placeholder analysis
        
        # Analyze current state for potential evolution
        for intent in self.semantic_intents:
            stability = self._assess_intent_stability(intent)
            evolution["intent_stability"][intent.name] = {
                "current_intent": intent.primary_intent.value,
                "stability_score": stability,
                "change_likelihood": 1 - stability
            }
            
            if stability < 0.6:
                evolution["semantic_drift"].append({
                    "element": intent.name,
                    "current_intent": intent.primary_intent.value,
                    "drift_indicators": self._identify_drift_indicators(intent)
                })
        
        return evolution
    
    def _generate_semantic_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations based on semantic analysis
        """
        recommendations = {
            "naming_improvements": [],
            "refactoring_suggestions": [],
            "pattern_recommendations": [],
            "documentation_needs": [],
            "architectural_suggestions": []
        }
        
        # Naming improvements
        for intent in self.semantic_intents:
            if not self._name_reflects_intent(intent):
                recommendations["naming_improvements"].append({
                    "current_name": intent.name,
                    "suggested_name": self._suggest_semantic_name(intent.name, intent.primary_intent),
                    "reason": f"Better reflects {intent.primary_intent.value} intent"
                })
        
        # Refactoring suggestions
        duplicates = self._find_semantic_duplicates(self.semantic_intents)
        for dup_group in duplicates:
            recommendations["refactoring_suggestions"].append({
                "type": "consolidate_duplicates",
                "elements": [d.name for d in dup_group],
                "suggestion": "Merge semantically equivalent functions"
            })
        
        # Pattern recommendations
        for pattern in self.conceptual_patterns:
            if pattern.implementation_quality < 0.7:
                recommendations["pattern_recommendations"].append({
                    "pattern": pattern.pattern_name,
                    "current_quality": pattern.implementation_quality,
                    "improvements": self._suggest_pattern_improvements(pattern),
                    "alternatives": pattern.alternatives
                })
        
        # Documentation needs
        undocumented = [i for i in self.semantic_intents if not i.context.get("has_docstring")]
        for intent in undocumented[:10]:  # Top 10
            recommendations["documentation_needs"].append({
                "element": intent.name,
                "intent": intent.primary_intent.value,
                "suggested_doc": self._generate_doc_for_intent(intent)
            })
        
        # Architectural suggestions
        clustering_result = self._perform_semantic_clustering()
        if clustering_result["suggested_reorganization"]:
            recommendations["architectural_suggestions"] = clustering_result["suggested_reorganization"]
        
        return recommendations
    
    def _generate_semantic_summary(self) -> Dict[str, Any]:
        """
        Generate summary of semantic analysis
        """
        summary = {
            "semantic_health_score": 0,
            "total_elements_analyzed": len(self.semantic_intents),
            "intent_distribution": {},
            "pattern_usage": {},
            "key_findings": [],
            "improvement_areas": [],
            "semantic_metrics": {}
        }
        
        # Calculate intent distribution
        intent_counts = defaultdict(int)
        for intent in self.semantic_intents:
            intent_counts[intent.primary_intent.value] += 1
        summary["intent_distribution"] = dict(intent_counts)
        
        # Pattern usage summary
        pattern_counts = defaultdict(int)
        for pattern in self.conceptual_patterns:
            pattern_counts[pattern.pattern_name] += len(pattern.occurrences)
        summary["pattern_usage"] = dict(pattern_counts)
        
        # Calculate semantic health score
        factors = {
            "intent_clarity": sum(i.confidence for i in self.semantic_intents) / len(self.semantic_intents) if self.semantic_intents else 0,
            "naming_consistency": self._calculate_naming_consistency(),
            "pattern_quality": sum(p.implementation_quality for p in self.conceptual_patterns) / len(self.conceptual_patterns) if self.conceptual_patterns else 0,
            "documentation_alignment": 0.7  # Placeholder
        }
        
        summary["semantic_health_score"] = sum(factors.values()) / len(factors) * 100
        
        # Key findings
        if summary["semantic_health_score"] < 60:
            summary["key_findings"].append("Low semantic clarity - code intent is unclear")
        
        if len(intent_counts) == 1:
            summary["key_findings"].append("Single-purpose codebase")
        elif len(intent_counts) > 10:
            summary["key_findings"].append("Highly diverse codebase with many different intents")
        
        # Improvement areas
        if factors["intent_clarity"] < 0.6:
            summary["improvement_areas"].append("Clarify code intent through better naming and structure")
        
        if factors["naming_consistency"] < 0.7:
            summary["improvement_areas"].append("Improve naming consistency across codebase")
        
        # Semantic metrics
        summary["semantic_metrics"] = {
            "average_intent_confidence": factors["intent_clarity"],
            "semantic_cohesion": self._calculate_overall_cohesion(),
            "semantic_coupling": self._calculate_overall_coupling(),
            "pattern_diversity": len(pattern_counts),
            "concept_clarity": summary["semantic_health_score"] / 100
        }
        
        return summary
    
    # Helper methods
    def _analyze_function_intent(self, node: ast.FunctionDef, file_path: Path) -> SemanticIntent:
        """Analyze the intent of a function"""
        # Analyze function name
        name_intent = self._infer_intent_from_name(node.name)
        
        # Analyze function body
        body_intents = self._analyze_function_body_intent(node)
        
        # Combine to determine primary intent
        all_intents = [name_intent] + body_intents
        intent_scores = defaultdict(float)
        
        for intent in all_intents:
            intent_scores[intent] += 1
        
        # Find primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else IntentType.UNKNOWN
        
        # Find secondary intents
        secondary_intents = [i for i, score in intent_scores.items() if i != primary_intent and score > 0]
        
        # Calculate confidence
        confidence = intent_scores[primary_intent] / sum(intent_scores.values()) if intent_scores else 0
        
        # Create semantic signature
        signature = self._create_semantic_signature(node)
        
        # Extract relationships
        relationships = self._extract_function_relationships(node)
        
        return SemanticIntent(
            element_type="function",
            name=node.name,
            location=f"{file_path}:{node.name}",
            primary_intent=primary_intent,
            secondary_intents=secondary_intents[:3],  # Top 3 secondary intents
            confidence=confidence,
            semantic_signature=signature,
            context={"has_docstring": bool(ast.get_docstring(node))},
            relationships=relationships
        )
    
    def _infer_intent_from_name(self, name: str) -> IntentType:
        """Infer intent from a name"""
        name_lower = name.lower()
        
        for intent_type, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return intent_type
        
        return IntentType.UNKNOWN
    
    def _analyze_function_body_intent(self, node: ast.FunctionDef) -> List[IntentType]:
        """Analyze function body to determine intents"""
        intents = []
        
        for child in ast.walk(node):
            # Check for validation patterns
            if isinstance(child, ast.If):
                if self._is_validation_check(child):
                    intents.append(IntentType.VALIDATION)
            
            # Check for persistence patterns
            elif isinstance(child, ast.Call):
                func_name = self._get_function_name(child)
                if func_name:
                    intent = self._infer_intent_from_name(func_name)
                    if intent != IntentType.UNKNOWN:
                        intents.append(intent)
        
        return intents
    
    def _create_semantic_signature(self, node: ast.FunctionDef) -> str:
        """Create a semantic signature for a function"""
        components = []
        
        # Input types
        if node.args.args:
            components.append(f"inputs:{len(node.args.args)}")
        
        # Output type
        if node.returns:
            components.append("returns:typed")
        else:
            components.append("returns:untyped")
        
        # Key operations
        operations = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                operations.add("call")
            elif isinstance(child, ast.If):
                operations.add("conditional")
            elif isinstance(child, ast.For):
                operations.add("iteration")
        
        components.append(f"ops:{','.join(sorted(operations))}")
        
        return "|".join(components)
    
    def _get_function_name(self, node: ast.Call) -> Optional[str]:
        """Get function name from a call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
    
    def _is_validation_check(self, node: ast.If) -> bool:
        """Check if an if statement is a validation check"""
        # Simple heuristic - check for common validation patterns
        if hasattr(node.test, 'id'):
            return any(keyword in node.test.id.lower() for keyword in ["valid", "check", "verify"])
        return False
    
    def _find_intent_for_element(self, name: str) -> Optional[SemanticIntent]:
        """Find the semantic intent for an element by name"""
        for intent in self.semantic_intents:
            if intent.name == name:
                return intent
        return None
    
    def _calculate_naming_consistency(self) -> float:
        """Calculate overall naming consistency"""
        consistent = 0
        total = 0
        
        for intent in self.semantic_intents:
            if self._name_reflects_intent(intent):
                consistent += 1
            total += 1
        
        return consistent / total if total > 0 else 0
    
    def _name_reflects_intent(self, intent: SemanticIntent) -> bool:
        """Check if name reflects the semantic intent"""
        name_intent = self._infer_intent_from_name(intent.name)
        return name_intent == intent.primary_intent
    
    def _calculate_overall_cohesion(self) -> float:
        """Calculate overall semantic cohesion"""
        # Simplified cohesion calculation
        if not self.semantic_intents:
            return 0
        
        # Group by module
        modules = defaultdict(list)
        for intent in self.semantic_intents:
            module = self._get_module_from_location(intent.location)
            modules[module].append(intent)
        
        # Calculate average module cohesion
        cohesion_scores = []
        for module_intents in modules.values():
            if len(module_intents) > 1:
                # Check how similar the intents are
                primary_intents = [i.primary_intent for i in module_intents]
                unique_intents = len(set(primary_intents))
                cohesion = 1 / unique_intents if unique_intents > 0 else 0
                cohesion_scores.append(cohesion)
        
        return sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0
    
    def _calculate_overall_coupling(self) -> float:
        """Calculate overall semantic coupling"""
        # Simplified coupling calculation
        if not self.semantic_intents:
            return 0
        
        total_relationships = sum(len(i.relationships) for i in self.semantic_intents)
        total_elements = len(self.semantic_intents)
        
        # Lower is better for coupling
        return total_relationships / (total_elements * total_elements) if total_elements > 0 else 0
    
    def _get_module_from_location(self, location: str) -> str:
        """Extract module name from location string"""
        # Simple extraction - would be more sophisticated in practice
        if ":" in location:
            path = location.split(":")[0]
            return Path(path).stem
        return "unknown"