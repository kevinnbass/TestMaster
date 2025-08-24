"""
Business Analysis Base Data Structures
======================================

Core data structures and configuration for business rule analysis.
Part of modularized business_analyzer system.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
from collections import defaultdict


class BusinessRuleType(Enum):
    """Types of business rules"""
    VALIDATION = "validation"
    CALCULATION = "calculation"
    AUTHORIZATION = "authorization"
    WORKFLOW = "workflow"
    CONSTRAINT = "business_constraint"
    COMPLIANCE = "compliance"
    SLA = "sla"
    PRICING = "pricing"


class WorkflowPattern(Enum):
    """Workflow execution patterns"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    BRANCH_MERGE = "branch_merge"


class DomainType(Enum):
    """Business domain categories"""
    FINANCIAL = "financial"
    INVENTORY = "inventory"
    CUSTOMER = "customer"
    COMPLIANCE = "compliance"
    SCHEDULING = "scheduling"
    PRICING = "pricing"


@dataclass
class BusinessRule:
    """Represents a business rule extracted from code"""
    rule_type: str
    name: str
    description: str
    location: str
    conditions: List[str]
    actions: List[str]
    constraints: List[str]
    domain_entities: List[str]
    confidence: float
    documentation: str
    priority: str = "medium"
    complexity_score: int = 1
    business_impact: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "rule_type": self.rule_type,
            "name": self.name,
            "description": self.description,
            "location": self.location,
            "conditions": self.conditions,
            "actions": self.actions,
            "constraints": self.constraints,
            "domain_entities": self.domain_entities,
            "confidence": self.confidence,
            "documentation": self.documentation,
            "priority": self.priority,
            "complexity_score": self.complexity_score,
            "business_impact": self.business_impact
        }


@dataclass 
class WorkflowState:
    """Represents a state in a workflow"""
    name: str
    transitions: Dict[str, str]  # event -> next_state
    entry_actions: List[str]
    exit_actions: List[str]
    guards: List[str]
    state_type: str = "normal"  # normal, initial, final
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "transitions": self.transitions,
            "entry_actions": self.entry_actions,
            "exit_actions": self.exit_actions,
            "guards": self.guards,
            "state_type": self.state_type
        }


@dataclass
class DomainEntity:
    """Represents a domain entity"""
    name: str
    attributes: Dict[str, str]
    behaviors: List[str]
    relationships: Dict[str, str]
    invariants: List[str]
    business_rules: List[str]
    entity_type: str = "entity"  # entity, value_object, aggregate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "attributes": self.attributes,
            "behaviors": self.behaviors,
            "relationships": self.relationships,
            "invariants": self.invariants,
            "business_rules": self.business_rules,
            "entity_type": self.entity_type
        }


@dataclass
class BusinessConstraint:
    """Represents a business constraint"""
    name: str
    constraint_type: str  # numeric, temporal, capacity, relationship
    value: Optional[str]
    description: str
    location: str
    enforcement_level: str = "hard"  # hard, soft, advisory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "constraint_type": self.constraint_type,
            "value": self.value,
            "description": self.description,
            "location": self.location,
            "enforcement_level": self.enforcement_level
        }


@dataclass
class BusinessEvent:
    """Represents a business event"""
    name: str
    event_type: str  # domain_event, integration_event, notification
    triggers: List[str]
    handlers: List[str]
    data_schema: Dict[str, str]
    location: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "event_type": self.event_type,
            "triggers": self.triggers,
            "handlers": self.handlers,
            "data_schema": self.data_schema,
            "location": self.location
        }


class BusinessAnalysisConfiguration:
    """Configuration for business rule analysis"""
    
    def __init__(self):
        # Business rule patterns
        self.rule_patterns = {
            "validation": [
                r"validate_\w+", r"check_\w+", r"verify_\w+", 
                r"ensure_\w+", r"assert_\w+", r"is_valid_\w+"
            ],
            "calculation": [
                r"calculate_\w+", r"compute_\w+", r"derive_\w+",
                r"get_\w+_amount", r"apply_\w+", r"process_\w+"
            ],
            "authorization": [
                r"can_\w+", r"has_permission", r"is_authorized",
                r"check_access", r"allow_\w+", r"deny_\w+"
            ],
            "workflow": [
                r"transition_to", r"change_state", r"move_to",
                r"approve_\w+", r"reject_\w+", r"submit_\w+"
            ],
            "business_constraint": [
                r"max_\w+", r"min_\w+", r"limit_\w+",
                r"threshold_\w+", r"quota_\w+", r"capacity_\w+"
            ]
        }
        
        # Domain keywords that indicate business logic
        self.domain_keywords = {
            "financial": ["payment", "invoice", "balance", "transaction", "fee", "tax", "discount"],
            "inventory": ["stock", "quantity", "warehouse", "shipment", "order", "product"],
            "customer": ["user", "client", "account", "profile", "subscription", "member"],
            "compliance": ["audit", "regulation", "policy", "compliance", "approval", "review"],
            "scheduling": ["appointment", "booking", "calendar", "schedule", "availability"],
            "pricing": ["price", "cost", "rate", "tier", "package", "bundle"]
        }
        
        # State machine patterns
        self.state_patterns = {
            "states": ["PENDING", "APPROVED", "REJECTED", "ACTIVE", "INACTIVE", "COMPLETED"],
            "transitions": ["approve", "reject", "submit", "cancel", "complete", "activate"],
            "guards": ["can_transition", "is_allowed", "meets_criteria"]
        }
        
        # Compliance frameworks
        self.compliance_keywords = {
            "gdpr": ["gdpr", "data protection", "privacy", "consent", "right to be forgotten"],
            "pci": ["pci", "card", "payment", "credit card", "cardholder"],
            "hipaa": ["hipaa", "health", "medical", "patient", "phi"],
            "sox": ["sox", "sarbanes", "audit", "financial reporting"],
            "kyc": ["kyc", "know your customer", "identity", "verification"],
            "aml": ["aml", "anti-money", "laundering", "suspicious"]
        }
        
        # SLA keywords
        self.sla_keywords = {
            "response_time": ["response", "latency", "timeout", "deadline"],
            "availability": ["uptime", "availability", "downtime", "maintenance"],
            "throughput": ["throughput", "requests_per", "transactions_per", "capacity"],
            "quality": ["error_rate", "success_rate", "accuracy", "quality"],
            "escalation": ["escalate", "priority", "severity", "critical"]
        }
        
        # Pricing keywords
        self.pricing_keywords = {
            "pricing": ["price", "cost", "rate", "fee"],
            "discount": ["discount", "rebate", "coupon", "promo"],
            "tier": ["tier", "plan", "package", "subscription"],
            "billing": ["bill", "invoice", "payment", "charge"]
        }


# Export
__all__ = [
    'BusinessRuleType', 'WorkflowPattern', 'DomainType',
    'BusinessRule', 'WorkflowState', 'DomainEntity', 
    'BusinessConstraint', 'BusinessEvent',
    'BusinessAnalysisConfiguration'
]