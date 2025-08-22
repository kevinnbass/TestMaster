"""
Semantic Code Analysis Module - Integrated from Archive
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
import logging


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


class SemanticAnalyzer:
    """
    Analyzes code semantics to understand intent and purpose
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
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
        
    def analyze(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis
        """
        self.base_path = Path(path) if path else Path.cwd()
        
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
            "semantic_quality": self._assess_semantic_quality(),
            "summary": self._generate_semantic_summary()
        }
        
        return results
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project"""
        return list(self.base_path.rglob("*.py"))
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file into an AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ast.parse(f.read())
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _recognize_intent(self) -> Dict[str, Any]:
        """Recognize developer intent from code elements"""
        intent_analysis = {
            "recognized_intents": [],
            "intent_distribution": defaultdict(int),
            "confidence_scores": [],
            "ambiguous_intents": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Analyze functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            intent = self._analyze_function_intent(node, file_path)
                            intent_analysis["recognized_intents"].append(intent)
                            intent_analysis["intent_distribution"][intent.primary_intent.value] += 1
                            intent_analysis["confidence_scores"].append(intent.confidence)
                            
                            if intent.confidence < 0.7:
                                intent_analysis["ambiguous_intents"].append({
                                    "name": intent.name,
                                    "location": str(file_path),
                                    "possible_intents": [i.value for i in intent.secondary_intents]
                                })
                                
                        # Analyze classes
                        elif isinstance(node, ast.ClassDef):
                            intent = self._analyze_class_intent(node, file_path)
                            intent_analysis["recognized_intents"].append(intent)
                            intent_analysis["intent_distribution"][intent.primary_intent.value] += 1
                            
            except Exception as e:
                self.logger.error(f"Error recognizing intent in {file_path}: {e}")
                
        return intent_analysis
    
    def _extract_semantic_signatures(self) -> Dict[str, Any]:
        """Extract semantic signatures from code"""
        signatures = {
            "function_signatures": [],
            "class_signatures": [],
            "module_signatures": [],
            "signature_patterns": defaultdict(list)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Extract module signature
                    module_sig = self._create_module_signature(tree, file_path)
                    signatures["module_signatures"].append(module_sig)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_sig = self._create_function_signature(node)
                            signatures["function_signatures"].append({
                                "name": node.name,
                                "signature": func_sig,
                                "location": str(file_path)
                            })
                            signatures["signature_patterns"][func_sig].append(node.name)
                            
                        elif isinstance(node, ast.ClassDef):
                            class_sig = self._create_class_signature(node)
                            signatures["class_signatures"].append({
                                "name": node.name,
                                "signature": class_sig,
                                "location": str(file_path)
                            })
                            
            except Exception as e:
                self.logger.error(f"Error extracting signatures from {file_path}: {e}")
                
        return signatures
    
    def _identify_conceptual_patterns(self) -> Dict[str, Any]:
        """Identify conceptual patterns in code"""
        patterns = {
            "design_patterns": [],
            "architectural_patterns": [],
            "idioms": [],
            "anti_patterns": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Check for design patterns
                    for pattern_name, pattern_info in self.semantic_patterns.items():
                        if self._check_pattern_presence(tree, pattern_info["indicators"]):
                            patterns["design_patterns"].append({
                                "pattern": pattern_name,
                                "location": str(file_path),
                                "role": pattern_info["role"],
                                "confidence": self._calculate_pattern_confidence(tree, pattern_info)
                            })
                            
                    # Check for anti-patterns
                    anti_patterns = self._detect_anti_patterns(tree)
                    if anti_patterns:
                        patterns["anti_patterns"].extend([{
                            "pattern": ap,
                            "location": str(file_path)
                        } for ap in anti_patterns])
                        
            except Exception as e:
                self.logger.error(f"Error identifying patterns in {file_path}: {e}")
                
        return patterns
    
    def _analyze_semantic_relationships(self) -> Dict[str, Any]:
        """Analyze semantic relationships between code elements"""
        relationships = {
            "direct_relationships": [],
            "indirect_relationships": [],
            "dependency_graph": {},
            "coupling_analysis": {}
        }
        
        # Build a map of all code elements
        element_map = {}
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            element_name = node.name
                            element_map[element_name] = {
                                "type": type(node).__name__,
                                "location": str(file_path),
                                "node": node
                            }
                            
            except Exception as e:
                self.logger.error(f"Error mapping elements in {file_path}: {e}")
                
        # Analyze relationships
        for name, info in element_map.items():
            dependencies = self._extract_dependencies(info["node"])
            relationships["dependency_graph"][name] = dependencies
            
            for dep in dependencies:
                if dep in element_map:
                    relationships["direct_relationships"].append({
                        "from": name,
                        "to": dep,
                        "type": "uses"
                    })
                    
        return relationships
    
    def _classify_code_purpose(self) -> Dict[str, Any]:
        """Classify the purpose of code sections"""
        purpose_classification = {
            "business_logic": [],
            "infrastructure": [],
            "utilities": [],
            "data_access": [],
            "presentation": [],
            "integration": []
        }
        
        for file_path in self._get_python_files():
            try:
                # Classify based on file location and content
                file_purpose = self._determine_file_purpose(file_path)
                
                if file_purpose:
                    purpose_classification[file_purpose].append(str(file_path))
                    
            except Exception as e:
                self.logger.error(f"Error classifying {file_path}: {e}")
                
        return purpose_classification
    
    def _analyze_naming_semantics(self) -> Dict[str, Any]:
        """Analyze naming conventions and semantics"""
        naming_analysis = {
            "naming_conventions": defaultdict(list),
            "semantic_coherence": [],
            "naming_violations": [],
            "suggested_improvements": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            name = node.name
                            convention = self._identify_naming_convention(name)
                            naming_analysis["naming_conventions"][convention].append(name)
                            
                            # Check semantic coherence
                            if isinstance(node, ast.FunctionDef):
                                coherence = self._check_function_name_coherence(node)
                                if coherence < 0.7:
                                    naming_analysis["naming_violations"].append({
                                        "name": name,
                                        "issue": "Low semantic coherence",
                                        "suggestion": self._suggest_better_name(node)
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error analyzing naming in {file_path}: {e}")
                
        return naming_analysis
    
    def _identify_behavioral_patterns(self) -> Dict[str, Any]:
        """Identify behavioral patterns in code"""
        behavioral_patterns = {
            "state_machines": [],
            "event_driven": [],
            "pipeline_patterns": [],
            "callback_patterns": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Detect state machine patterns
                    if self._has_state_machine_pattern(tree):
                        behavioral_patterns["state_machines"].append(str(file_path))
                        
                    # Detect event-driven patterns
                    if self._has_event_pattern(tree):
                        behavioral_patterns["event_driven"].append(str(file_path))
                        
                    # Detect pipeline patterns
                    if self._has_pipeline_pattern(tree):
                        behavioral_patterns["pipeline_patterns"].append(str(file_path))
                        
            except Exception as e:
                self.logger.error(f"Error identifying behavioral patterns in {file_path}: {e}")
                
        return behavioral_patterns
    
    def _extract_domain_concepts(self) -> Dict[str, Any]:
        """Extract domain-specific concepts from code"""
        domain_concepts = {
            "entities": [],
            "value_objects": [],
            "services": [],
            "repositories": [],
            "domain_events": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Classify domain element
                            domain_type = self._classify_domain_element(node)
                            if domain_type:
                                domain_concepts[domain_type].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "properties": self._extract_class_properties(node)
                                })
                                
            except Exception as e:
                self.logger.error(f"Error extracting domain concepts from {file_path}: {e}")
                
        return domain_concepts
    
    def _perform_semantic_clustering(self) -> Dict[str, Any]:
        """Cluster code elements based on semantic similarity"""
        clusters = {
            "semantic_clusters": [],
            "cluster_coherence": [],
            "outliers": []
        }
        
        # Collect all code elements with their semantic features
        elements = []
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            features = self._extract_semantic_features(node)
                            elements.append({
                                "name": node.name,
                                "type": type(node).__name__,
                                "features": features,
                                "location": str(file_path)
                            })
                            
            except Exception as e:
                self.logger.error(f"Error collecting elements from {file_path}: {e}")
                
        # Simple clustering based on feature similarity
        if elements:
            clusters["semantic_clusters"] = self._simple_cluster(elements)
            
        return clusters
    
    def _check_intent_consistency(self) -> Dict[str, Any]:
        """Check consistency of intent across codebase"""
        consistency_analysis = {
            "consistent_patterns": [],
            "inconsistencies": [],
            "naming_intent_mismatches": [],
            "structural_inconsistencies": []
        }
        
        # Analyze consistency across similar functions
        function_groups = defaultdict(list)
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Group by similar names
                            base_name = self._extract_base_name(node.name)
                            function_groups[base_name].append({
                                "name": node.name,
                                "location": str(file_path),
                                "structure": self._get_function_structure(node)
                            })
                            
            except Exception as e:
                self.logger.error(f"Error checking consistency in {file_path}: {e}")
                
        # Check for inconsistencies
        for base_name, functions in function_groups.items():
            if len(functions) > 1:
                if not self._are_structures_consistent(functions):
                    consistency_analysis["structural_inconsistencies"].append({
                        "base_name": base_name,
                        "functions": functions
                    })
                    
        return consistency_analysis
    
    def _assess_semantic_quality(self) -> Dict[str, Any]:
        """Assess the semantic quality of code"""
        quality_assessment = {
            "clarity_score": 0.0,
            "consistency_score": 0.0,
            "expressiveness_score": 0.0,
            "overall_quality": 0.0,
            "improvement_suggestions": []
        }
        
        clarity_scores = []
        consistency_scores = []
        expressiveness_scores = []
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Assess clarity
                    clarity = self._assess_code_clarity(tree)
                    clarity_scores.append(clarity)
                    
                    # Assess consistency
                    consistency = self._assess_code_consistency(tree)
                    consistency_scores.append(consistency)
                    
                    # Assess expressiveness
                    expressiveness = self._assess_code_expressiveness(tree)
                    expressiveness_scores.append(expressiveness)
                    
            except Exception as e:
                self.logger.error(f"Error assessing quality of {file_path}: {e}")
                
        # Calculate averages
        if clarity_scores:
            quality_assessment["clarity_score"] = sum(clarity_scores) / len(clarity_scores)
        if consistency_scores:
            quality_assessment["consistency_score"] = sum(consistency_scores) / len(consistency_scores)
        if expressiveness_scores:
            quality_assessment["expressiveness_score"] = sum(expressiveness_scores) / len(expressiveness_scores)
            
        quality_assessment["overall_quality"] = (
            quality_assessment["clarity_score"] +
            quality_assessment["consistency_score"] +
            quality_assessment["expressiveness_score"]
        ) / 3
        
        # Generate improvement suggestions
        if quality_assessment["clarity_score"] < 0.7:
            quality_assessment["improvement_suggestions"].append(
                "Improve code clarity through better naming and structure"
            )
        if quality_assessment["consistency_score"] < 0.7:
            quality_assessment["improvement_suggestions"].append(
                "Increase consistency in coding patterns and conventions"
            )
        if quality_assessment["expressiveness_score"] < 0.7:
            quality_assessment["improvement_suggestions"].append(
                "Make code more expressive and self-documenting"
            )
            
        return quality_assessment
    
    def _generate_semantic_summary(self) -> Dict[str, Any]:
        """Generate a summary of semantic analysis"""
        return {
            "total_intents_recognized": len(self.semantic_intents),
            "dominant_intent_type": self._get_dominant_intent(),
            "pattern_count": len(self.conceptual_patterns),
            "semantic_complexity": self._calculate_semantic_complexity(),
            "recommendations": self._generate_semantic_recommendations()
        }
    
    # Helper methods
    def _analyze_function_intent(self, node: ast.FunctionDef, file_path: Path) -> SemanticIntent:
        """Analyze the intent of a function"""
        primary_intent = IntentType.UNKNOWN
        secondary_intents = []
        confidence = 0.0
        
        # Analyze function name
        name_lower = node.name.lower()
        intent_scores = defaultdict(float)
        
        for intent_type, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    intent_scores[intent_type] += 1.0
                    
        # Analyze function body
        for child in ast.walk(node):
            for intent_type, keywords in self.intent_keywords.items():
                if isinstance(child, ast.Name):
                    if any(keyword in child.id.lower() for keyword in keywords):
                        intent_scores[intent_type] += 0.5
                        
        # Determine primary intent
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            primary_intent = sorted_intents[0][0]
            confidence = min(sorted_intents[0][1] / 5.0, 1.0)  # Normalize confidence
            
            # Get secondary intents
            secondary_intents = [intent for intent, score in sorted_intents[1:3] if score > 0]
            
        return SemanticIntent(
            element_type="function",
            name=node.name,
            location=str(file_path),
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            semantic_signature=self._create_function_signature(node),
            context={},
            relationships=[]
        )
    
    def _analyze_class_intent(self, node: ast.ClassDef, file_path: Path) -> SemanticIntent:
        """Analyze the intent of a class"""
        # Similar to function intent but for classes
        return SemanticIntent(
            element_type="class",
            name=node.name,
            location=str(file_path),
            primary_intent=self._determine_class_intent(node),
            secondary_intents=[],
            confidence=0.8,
            semantic_signature=self._create_class_signature(node),
            context={},
            relationships=[]
        )
    
    def _determine_class_intent(self, node: ast.ClassDef) -> IntentType:
        """Determine the primary intent of a class"""
        name_lower = node.name.lower()
        
        # Check common patterns
        if any(pattern in name_lower for pattern in ["model", "entity", "dto"]):
            return IntentType.PERSISTENCE
        elif any(pattern in name_lower for pattern in ["service", "manager", "handler"]):
            return IntentType.ORCHESTRATION
        elif any(pattern in name_lower for pattern in ["validator", "checker"]):
            return IntentType.VALIDATION
        elif any(pattern in name_lower for pattern in ["controller", "endpoint", "api"]):
            return IntentType.API_ENDPOINT
        elif "test" in name_lower:
            return IntentType.TESTING
        else:
            return IntentType.UNKNOWN
    
    def _create_function_signature(self, node: ast.FunctionDef) -> str:
        """Create a semantic signature for a function"""
        params = [arg.arg for arg in node.args.args]
        return f"func:{node.name}({','.join(params)})"
    
    def _create_class_signature(self, node: ast.ClassDef) -> str:
        """Create a semantic signature for a class"""
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        return f"class:{node.name}[{','.join(methods[:3])}]"
    
    def _create_module_signature(self, tree: ast.AST, file_path: Path) -> Dict[str, Any]:
        """Create a semantic signature for a module"""
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        return {
            "module": str(file_path),
            "classes": classes[:5],
            "functions": functions[:5],
            "signature": f"module:{file_path.stem}[{len(classes)}c,{len(functions)}f]"
        }
    
    def _check_pattern_presence(self, tree: ast.AST, indicators: List[str]) -> bool:
        """Check if pattern indicators are present in AST"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(indicator in code_str.lower() for indicator in indicators)
    
    def _calculate_pattern_confidence(self, tree: ast.AST, pattern_info: Dict) -> float:
        """Calculate confidence for a pattern match"""
        indicators_found = 0
        for indicator in pattern_info["indicators"]:
            if self._check_pattern_presence(tree, [indicator]):
                indicators_found += 1
        return indicators_found / len(pattern_info["indicators"]) if pattern_info["indicators"] else 0.0
    
    def _detect_anti_patterns(self, tree: ast.AST) -> List[str]:
        """Detect anti-patterns in code"""
        anti_patterns = []
        
        # Check for God Class
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        for cls in classes:
            method_count = sum(1 for n in cls.body if isinstance(n, ast.FunctionDef))
            if method_count > 20:
                anti_patterns.append(f"GodClass:{cls.name}")
                
        # Check for Long Method
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        for func in functions:
            if len(func.body) > 50:
                anti_patterns.append(f"LongMethod:{func.name}")
                
        return anti_patterns
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract dependencies from a node"""
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.append(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    dependencies.append(child.value.id)
        return list(set(dependencies))
    
    def _determine_file_purpose(self, file_path: Path) -> Optional[str]:
        """Determine the purpose of a file based on location and content"""
        path_str = str(file_path).lower()
        
        if any(pattern in path_str for pattern in ["model", "entity", "schema"]):
            return "data_access"
        elif any(pattern in path_str for pattern in ["service", "business", "logic"]):
            return "business_logic"
        elif any(pattern in path_str for pattern in ["util", "helper", "common"]):
            return "utilities"
        elif any(pattern in path_str for pattern in ["api", "endpoint", "controller"]):
            return "presentation"
        elif any(pattern in path_str for pattern in ["integration", "adapter", "gateway"]):
            return "integration"
        elif any(pattern in path_str for pattern in ["infra", "config", "setup"]):
            return "infrastructure"
        
        return None
    
    def _identify_naming_convention(self, name: str) -> str:
        """Identify the naming convention used"""
        if "_" in name:
            return "snake_case"
        elif name[0].isupper():
            return "PascalCase"
        elif name[0].islower() and any(c.isupper() for c in name[1:]):
            return "camelCase"
        else:
            return "lowercase"
    
    def _check_function_name_coherence(self, node: ast.FunctionDef) -> float:
        """Check if function name matches its behavior"""
        # Simplified coherence check
        return 0.8  # Placeholder
    
    def _suggest_better_name(self, node: ast.FunctionDef) -> str:
        """Suggest a better name for a function"""
        # Analyze function body to suggest name
        return f"improved_{node.name}"  # Placeholder
    
    def _has_state_machine_pattern(self, tree: ast.AST) -> bool:
        """Check if code has state machine pattern"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(pattern in code_str.lower() for pattern in ["state", "transition", "fsm"])
    
    def _has_event_pattern(self, tree: ast.AST) -> bool:
        """Check if code has event-driven pattern"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(pattern in code_str.lower() for pattern in ["event", "listener", "emit", "subscribe"])
    
    def _has_pipeline_pattern(self, tree: ast.AST) -> bool:
        """Check if code has pipeline pattern"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(pattern in code_str.lower() for pattern in ["pipeline", "chain", "flow", "stream"])
    
    def _classify_domain_element(self, node: ast.ClassDef) -> Optional[str]:
        """Classify a class as a domain element"""
        name_lower = node.name.lower()
        
        if "entity" in name_lower:
            return "entities"
        elif "value" in name_lower:
            return "value_objects"
        elif "service" in name_lower:
            return "services"
        elif "repository" in name_lower:
            return "repositories"
        elif "event" in name_lower:
            return "domain_events"
        
        return None
    
    def _extract_class_properties(self, node: ast.ClassDef) -> List[str]:
        """Extract properties/attributes from a class"""
        properties = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        properties.append(target.id)
        return properties
    
    def _extract_semantic_features(self, node: ast.AST) -> Dict[str, Any]:
        """Extract semantic features from a node"""
        return {
            "type": type(node).__name__,
            "size": len(ast.walk(node)),
            "complexity": self._calculate_complexity(node)
        }
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
        return complexity
    
    def _simple_cluster(self, elements: List[Dict]) -> List[List[Dict]]:
        """Simple clustering of elements"""
        # Group by type for now
        clusters = defaultdict(list)
        for element in elements:
            clusters[element["type"]].append(element)
        return list(clusters.values())
    
    def _extract_base_name(self, name: str) -> str:
        """Extract base name from function name"""
        # Remove common prefixes/suffixes
        for prefix in ["get_", "set_", "is_", "has_", "can_"]:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    
    def _get_function_structure(self, node: ast.FunctionDef) -> Dict:
        """Get structural information about a function"""
        return {
            "params": len(node.args.args),
            "lines": len(node.body),
            "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node))
        }
    
    def _are_structures_consistent(self, functions: List[Dict]) -> bool:
        """Check if function structures are consistent"""
        if not functions:
            return True
        first_structure = functions[0]["structure"]
        return all(f["structure"] == first_structure for f in functions)
    
    def _assess_code_clarity(self, tree: ast.AST) -> float:
        """Assess code clarity"""
        # Check for docstrings, comments, clear naming
        return 0.75  # Placeholder
    
    def _assess_code_consistency(self, tree: ast.AST) -> float:
        """Assess code consistency"""
        return 0.8  # Placeholder
    
    def _assess_code_expressiveness(self, tree: ast.AST) -> float:
        """Assess code expressiveness"""
        return 0.7  # Placeholder
    
    def _get_dominant_intent(self) -> str:
        """Get the most common intent type"""
        if not self.semantic_intents:
            return "unknown"
        intent_counts = defaultdict(int)
        for intent in self.semantic_intents:
            intent_counts[intent.primary_intent.value] += 1
        return max(intent_counts, key=intent_counts.get) if intent_counts else "unknown"
    
    def _calculate_semantic_complexity(self) -> float:
        """Calculate overall semantic complexity"""
        if not self.semantic_intents:
            return 0.0
        return len(set(i.primary_intent for i in self.semantic_intents)) / len(self.semantic_intents)
    
    def _generate_semantic_recommendations(self) -> List[str]:
        """Generate recommendations based on semantic analysis"""
        recommendations = []
        
        if self._calculate_semantic_complexity() > 0.8:
            recommendations.append("Consider reducing semantic complexity by grouping related functionality")
            
        if len(self.conceptual_patterns) < 5:
            recommendations.append("Consider implementing more design patterns for better structure")
            
        return recommendations