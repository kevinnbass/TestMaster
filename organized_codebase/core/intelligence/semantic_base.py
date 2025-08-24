"""
Semantic Analysis Base Data Structures
======================================

Core data structures and configuration for semantic code analysis.
Part of modularized semantic_analyzer system.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
from collections import defaultdict


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "element_type": self.element_type,
            "name": self.name,
            "location": self.location,
            "primary_intent": self.primary_intent.value,
            "secondary_intents": [intent.value for intent in self.secondary_intents],
            "confidence": self.confidence,
            "semantic_signature": self.semantic_signature,
            "context": self.context,
            "relationships": self.relationships
        }


@dataclass
class ConceptualPattern:
    """Represents a conceptual pattern in code"""
    pattern_name: str
    description: str
    occurrences: List[str]
    semantic_role: str
    implementation_quality: float
    alternatives: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "pattern_name": self.pattern_name,
            "description": self.description,
            "occurrences": self.occurrences,
            "semantic_role": self.semantic_role,
            "implementation_quality": self.implementation_quality,
            "alternatives": self.alternatives
        }


@dataclass
class SemanticRelationship:
    """Represents a semantic relationship between code elements"""
    from_element: str
    to_element: str
    relationship_type: str
    confidence: float
    context: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "from_element": self.from_element,
            "to_element": self.to_element,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "context": self.context
        }


class SemanticConfiguration:
    """Configuration for semantic analysis"""
    
    def __init__(self):
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
        
        # Domain purpose patterns
        self.purpose_indicators = {
            "business_logic": ["service", "business", "logic", "domain"],
            "data_access": ["model", "entity", "schema", "repository", "dao"],
            "utilities": ["util", "helper", "common", "tool"],
            "presentation": ["api", "endpoint", "controller", "view"],
            "integration": ["integration", "adapter", "gateway", "client"],
            "infrastructure": ["infra", "config", "setup", "init"]
        }
        
        # Naming convention patterns
        self.naming_patterns = {
            "snake_case": r"^[a-z]+(_[a-z0-9]+)*$",
            "camelCase": r"^[a-z][a-zA-Z0-9]*$",
            "PascalCase": r"^[A-Z][a-zA-Z0-9]*$",
            "UPPER_CASE": r"^[A-Z]+(_[A-Z0-9]+)*$"
        }
        
        # Anti-pattern thresholds
        self.anti_pattern_thresholds = {
            "god_class_methods": 20,
            "long_method_lines": 50,
            "deep_nesting": 5,
            "complex_condition": 4
        }


# Export
__all__ = [
    'IntentType', 'SemanticIntent', 'ConceptualPattern', 
    'SemanticRelationship', 'SemanticConfiguration'
]