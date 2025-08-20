"""
Business Rule Analysis Module
Extracts business rules, workflows, and domain logic from code
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json

from testmaster.analysis.base_analyzer import BaseAnalyzer


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


@dataclass 
class WorkflowState:
    """Represents a state in a workflow"""
    name: str
    transitions: Dict[str, str]  # event -> next_state
    entry_actions: List[str]
    exit_actions: List[str]
    guards: List[str]


@dataclass
class DomainEntity:
    """Represents a domain entity"""
    name: str
    attributes: Dict[str, str]
    behaviors: List[str]
    relationships: Dict[str, str]
    invariants: List[str]
    business_rules: List[str]


class BusinessRuleAnalyzer(BaseAnalyzer):
    """
    Analyzes code to extract business rules, workflows, and domain logic
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
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
        
        self.business_rules = []
        self.workflows = []
        self.domain_entities = []
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive business rule analysis
        """
        results = {
            "business_rules": self._extract_business_rules(),
            "validation_rules": self._extract_validation_rules(),
            "calculation_rules": self._extract_calculation_rules(),
            "authorization_rules": self._extract_authorization_rules(),
            "workflow_analysis": self._analyze_workflows(),
            "state_machines": self._detect_state_machines(),
            "domain_model": self._extract_domain_model(),
            "business_constraints": self._extract_business_constraints(),
            "decision_logic": self._extract_decision_logic(),
            "business_events": self._extract_business_events(),
            "compliance_rules": self._extract_compliance_rules(),
            "sla_rules": self._extract_sla_rules(),
            "pricing_rules": self._extract_pricing_rules(),
            "rule_dependencies": self._analyze_rule_dependencies(),
            "summary": self._generate_business_summary()
        }
        
        return results
    
    def _extract_business_rules(self) -> Dict[str, Any]:
        """
        Extract general business rules from code
        """
        rules_extracted = {
            "rules": [],
            "rule_categories": defaultdict(list),
            "confidence_levels": defaultdict(int),
            "documentation_coverage": 0,
            "total_rules": 0
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Extract rules from functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            rule = self._analyze_function_for_rules(node, file_path)
                            if rule:
                                rules_extracted["rules"].append(rule)
                                rules_extracted["rule_categories"][rule.rule_type].append(rule)
                                
                        # Extract rules from classes
                        elif isinstance(node, ast.ClassDef):
                            class_rules = self._analyze_class_for_rules(node, file_path)
                            rules_extracted["rules"].extend(class_rules)
                            
                        # Extract rules from conditionals
                        elif isinstance(node, ast.If):
                            conditional_rule = self._extract_conditional_rule(node, file_path)
                            if conditional_rule:
                                rules_extracted["rules"].append(conditional_rule)
                                
            except Exception as e:
                self.logger.error(f"Error extracting business rules from {file_path}: {e}")
                
        # Calculate metrics
        rules_extracted["total_rules"] = len(rules_extracted["rules"])
        
        # Calculate documentation coverage
        documented = sum(1 for rule in rules_extracted["rules"] if rule.documentation)
        if rules_extracted["total_rules"] > 0:
            rules_extracted["documentation_coverage"] = (documented / rules_extracted["total_rules"]) * 100
            
        # Group by confidence level
        for rule in rules_extracted["rules"]:
            if rule.confidence >= 0.8:
                rules_extracted["confidence_levels"]["high"] += 1
            elif rule.confidence >= 0.5:
                rules_extracted["confidence_levels"]["medium"] += 1
            else:
                rules_extracted["confidence_levels"]["low"] += 1
                
        return rules_extracted
    
    def _extract_validation_rules(self) -> Dict[str, Any]:
        """
        Extract validation-specific business rules
        """
        validation_rules = {
            "field_validations": [],
            "cross_field_validations": [],
            "business_validations": [],
            "format_validations": [],
            "range_validations": [],
            "custom_validations": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if it's a validation function
                            if any(pattern in node.name for pattern in ["validate", "check", "verify", "is_valid"]):
                                validation = self._analyze_validation_function(node, file_path)
                                
                                # Categorize validation type
                                if validation:
                                    if "field" in node.name.lower():
                                        validation_rules["field_validations"].append(validation)
                                    elif any(keyword in str(validation).lower() for keyword in ["between", "cross", "multiple"]):
                                        validation_rules["cross_field_validations"].append(validation)
                                    elif any(keyword in str(validation).lower() for keyword in ["business", "rule", "policy"]):
                                        validation_rules["business_validations"].append(validation)
                                    elif any(keyword in str(validation).lower() for keyword in ["format", "pattern", "regex"]):
                                        validation_rules["format_validations"].append(validation)
                                    elif any(keyword in str(validation).lower() for keyword in ["range", "min", "max", "limit"]):
                                        validation_rules["range_validations"].append(validation)
                                    else:
                                        validation_rules["custom_validations"].append(validation)
                                        
            except Exception as e:
                self.logger.error(f"Error extracting validation rules from {file_path}: {e}")
                
        return validation_rules
    
    def _extract_calculation_rules(self) -> Dict[str, Any]:
        """
        Extract calculation and computation business rules
        """
        calculation_rules = {
            "financial_calculations": [],
            "pricing_calculations": [],
            "tax_calculations": [],
            "discount_calculations": [],
            "scoring_calculations": [],
            "aggregation_rules": [],
            "formula_definitions": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for calculation functions
                            if any(pattern in node.name.lower() for pattern in ["calculate", "compute", "derive", "total", "sum"]):
                                calc_rule = self._analyze_calculation_function(node, file_path)
                                
                                if calc_rule:
                                    # Categorize calculation type
                                    if any(keyword in node.name.lower() for keyword in ["price", "cost", "fee"]):
                                        calculation_rules["pricing_calculations"].append(calc_rule)
                                    elif "tax" in node.name.lower():
                                        calculation_rules["tax_calculations"].append(calc_rule)
                                    elif "discount" in node.name.lower() or "rebate" in node.name.lower():
                                        calculation_rules["discount_calculations"].append(calc_rule)
                                    elif "score" in node.name.lower() or "rating" in node.name.lower():
                                        calculation_rules["scoring_calculations"].append(calc_rule)
                                    elif any(keyword in node.name.lower() for keyword in ["sum", "total", "aggregate"]):
                                        calculation_rules["aggregation_rules"].append(calc_rule)
                                    else:
                                        calculation_rules["financial_calculations"].append(calc_rule)
                                        
                        # Look for formula definitions in assignments
                        elif isinstance(node, ast.Assign):
                            formula = self._extract_formula(node, file_path)
                            if formula:
                                calculation_rules["formula_definitions"].append(formula)
                                
            except Exception as e:
                self.logger.error(f"Error extracting calculation rules from {file_path}: {e}")
                
        return calculation_rules
    
    def _extract_authorization_rules(self) -> Dict[str, Any]:
        """
        Extract authorization and access control rules
        """
        auth_rules = {
            "permission_checks": [],
            "role_based_rules": [],
            "attribute_based_rules": [],
            "context_based_rules": [],
            "delegation_rules": [],
            "access_policies": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for authorization functions
                            if any(pattern in node.name.lower() for pattern in ["can_", "has_permission", "is_authorized", "check_access"]):
                                auth_rule = self._analyze_authorization_function(node, file_path)
                                
                                if auth_rule:
                                    # Categorize authorization type
                                    if "role" in str(auth_rule).lower():
                                        auth_rules["role_based_rules"].append(auth_rule)
                                    elif "attribute" in str(auth_rule).lower() or "property" in str(auth_rule).lower():
                                        auth_rules["attribute_based_rules"].append(auth_rule)
                                    elif "context" in str(auth_rule).lower() or "situation" in str(auth_rule).lower():
                                        auth_rules["context_based_rules"].append(auth_rule)
                                    elif "delegate" in str(auth_rule).lower() or "behalf" in str(auth_rule).lower():
                                        auth_rules["delegation_rules"].append(auth_rule)
                                    else:
                                        auth_rules["permission_checks"].append(auth_rule)
                                        
                        # Look for decorators that might indicate authorization
                        elif isinstance(node, ast.FunctionDef):
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Name):
                                    if any(auth_word in decorator.id.lower() for auth_word in ["auth", "permission", "role", "access"]):
                                        auth_rules["access_policies"].append({
                                            "decorator": decorator.id,
                                            "function": node.name,
                                            "location": str(file_path)
                                        })
                                        
            except Exception as e:
                self.logger.error(f"Error extracting authorization rules from {file_path}: {e}")
                
        return auth_rules
    
    def _analyze_workflows(self) -> Dict[str, Any]:
        """
        Analyze workflow patterns in the code
        """
        workflow_analysis = {
            "workflows": [],
            "workflow_steps": [],
            "transitions": [],
            "approval_flows": [],
            "process_flows": [],
            "workflow_patterns": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Look for workflow classes
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if any(pattern in node.name.lower() for pattern in ["workflow", "process", "flow", "pipeline"]):
                                workflow = self._analyze_workflow_class(node, file_path)
                                workflow_analysis["workflows"].append(workflow)
                                
                        # Look for workflow functions
                        elif isinstance(node, ast.FunctionDef):
                            if any(pattern in node.name.lower() for pattern in ["transition", "approve", "reject", "submit", "process"]):
                                step = self._analyze_workflow_step(node, file_path)
                                workflow_analysis["workflow_steps"].append(step)
                                
                                # Categorize workflow type
                                if "approve" in node.name.lower() or "reject" in node.name.lower():
                                    workflow_analysis["approval_flows"].append(step)
                                elif "process" in node.name.lower():
                                    workflow_analysis["process_flows"].append(step)
                                    
                        # Look for state transitions
                        elif isinstance(node, ast.If):
                            transition = self._detect_state_transition(node, file_path)
                            if transition:
                                workflow_analysis["transitions"].append(transition)
                                
            except Exception as e:
                self.logger.error(f"Error analyzing workflows in {file_path}: {e}")
                
        # Identify workflow patterns
        for workflow in workflow_analysis["workflows"]:
            if "sequential" in str(workflow).lower():
                workflow_analysis["workflow_patterns"]["sequential"] += 1
            elif "parallel" in str(workflow).lower():
                workflow_analysis["workflow_patterns"]["parallel"] += 1
            elif "conditional" in str(workflow).lower():
                workflow_analysis["workflow_patterns"]["conditional"] += 1
                
        return workflow_analysis
    
    def _detect_state_machines(self) -> Dict[str, Any]:
        """
        Detect and analyze state machine implementations
        """
        state_machines = {
            "state_machines": [],
            "states": [],
            "transitions": [],
            "guards": [],
            "actions": [],
            "state_patterns": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = self._parse_file(file_path)
                if tree:
                    # Look for state machine classes
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if any(pattern in node.name.lower() for pattern in ["state", "machine", "fsm"]):
                                machine = self._analyze_state_machine(node, file_path)
                                state_machines["state_machines"].append(machine)
                                
                        # Look for state enums or constants
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    if "state" in target.id.lower():
                                        states = self._extract_states(node)
                                        state_machines["states"].extend(states)
                                        
                        # Look for transition methods
                        elif isinstance(node, ast.FunctionDef):
                            if "transition" in node.name.lower():
                                transition = self._analyze_transition(node, file_path)
                                state_machines["transitions"].append(transition)
                            elif "guard" in node.name.lower() or "can_" in node.name.lower():
                                guard = self._analyze_guard(node, file_path)
                                state_machines["guards"].append(guard)
                            elif "on_enter" in node.name.lower() or "on_exit" in node.name.lower():
                                action = self._analyze_state_action(node, file_path)
                                state_machines["actions"].append(action)
                                
                # Detect state patterns
                if re.search(r"STATE_\w+", content):
                    state_machines["state_patterns"]["enum_states"] += 1
                if re.search(r"transition_table", content, re.IGNORECASE):
                    state_machines["state_patterns"]["transition_table"] += 1
                if re.search(r"state_diagram", content, re.IGNORECASE):
                    state_machines["state_patterns"]["documented"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error detecting state machines in {file_path}: {e}")
                
        return state_machines
    
    def _extract_domain_model(self) -> Dict[str, Any]:
        """
        Extract domain model and entities
        """
        domain_model = {
            "entities": [],
            "value_objects": [],
            "aggregates": [],
            "repositories": [],
            "services": [],
            "domain_events": [],
            "relationships": [],
            "invariants": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Identify domain entities
                            entity = self._analyze_domain_entity(node, file_path)
                            if entity:
                                # Categorize entity type
                                if "entity" in node.name.lower():
                                    domain_model["entities"].append(entity)
                                elif "value" in node.name.lower() or "vo" in node.name.lower():
                                    domain_model["value_objects"].append(entity)
                                elif "aggregate" in node.name.lower() or "root" in node.name.lower():
                                    domain_model["aggregates"].append(entity)
                                elif "repository" in node.name.lower():
                                    domain_model["repositories"].append(entity)
                                elif "service" in node.name.lower():
                                    domain_model["services"].append(entity)
                                else:
                                    # Check if it looks like a domain entity
                                    if self._is_domain_entity(node):
                                        domain_model["entities"].append(entity)
                                        
                            # Extract relationships
                            relationships = self._extract_entity_relationships(node, file_path)
                            domain_model["relationships"].extend(relationships)
                            
                            # Extract invariants
                            invariants = self._extract_invariants(node, file_path)
                            domain_model["invariants"].extend(invariants)
                            
                        # Look for domain events
                        elif isinstance(node, ast.FunctionDef):
                            if "event" in node.name.lower() or "publish" in node.name.lower():
                                event = self._extract_domain_event(node, file_path)
                                if event:
                                    domain_model["domain_events"].append(event)
                                    
            except Exception as e:
                self.logger.error(f"Error extracting domain model from {file_path}: {e}")
                
        return domain_model
    
    def _extract_business_constraints(self) -> Dict[str, Any]:
        """
        Extract business constraints and limits
        """
        constraints = {
            "numeric_constraints": [],
            "temporal_constraints": [],
            "capacity_constraints": [],
            "relationship_constraints": [],
            "business_invariants": [],
            "constraint_violations": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for numeric constraints
                numeric_patterns = [
                    (r"MAX_\w+\s*=\s*(\d+)", "maximum"),
                    (r"MIN_\w+\s*=\s*(\d+)", "minimum"),
                    (r"LIMIT_\w+\s*=\s*(\d+)", "limit"),
                    (r"THRESHOLD_\w+\s*=\s*(\d+)", "threshold")
                ]
                
                for pattern, constraint_type in numeric_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        constraints["numeric_constraints"].append({
                            "type": constraint_type,
                            "name": match.group(0).split('=')[0].strip(),
                            "value": match.group(1),
                            "location": str(file_path)
                        })
                        
                # Look for temporal constraints
                temporal_patterns = [
                    r"deadline", r"expiry", r"timeout", r"duration",
                    r"start_date", r"end_date", r"valid_until"
                ]
                
                for pattern in temporal_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        constraints["temporal_constraints"].append({
                            "type": pattern,
                            "location": str(file_path)
                        })
                        
                tree = self._parse_file(file_path)
                if tree:
                    # Look for constraint checking functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if any(pattern in node.name.lower() for pattern in ["check", "validate", "ensure"]):
                                constraint = self._extract_constraint_logic(node, file_path)
                                if constraint:
                                    if "capacity" in str(constraint).lower():
                                        constraints["capacity_constraints"].append(constraint)
                                    elif "relationship" in str(constraint).lower():
                                        constraints["relationship_constraints"].append(constraint)
                                    else:
                                        constraints["business_invariants"].append(constraint)
                                        
            except Exception as e:
                self.logger.error(f"Error extracting constraints from {file_path}: {e}")
                
        return constraints
    
    def _extract_decision_logic(self) -> Dict[str, Any]:
        """
        Extract decision tables and business logic
        """
        decision_logic = {
            "decision_tables": [],
            "decision_trees": [],
            "rule_chains": [],
            "conditional_logic": [],
            "switch_statements": [],
            "lookup_tables": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Look for decision functions
                        if isinstance(node, ast.FunctionDef):
                            if any(pattern in node.name.lower() for pattern in ["decide", "determine", "evaluate", "choose"]):
                                decision = self._analyze_decision_function(node, file_path)
                                decision_logic["decision_trees"].append(decision)
                                
                        # Look for complex if-elif-else chains (decision trees)
                        elif isinstance(node, ast.If):
                            if self._is_complex_decision(node):
                                tree_logic = self._extract_decision_tree(node, file_path)
                                decision_logic["conditional_logic"].append(tree_logic)
                                
                        # Look for dictionary-based decision tables
                        elif isinstance(node, ast.Assign):
                            if isinstance(node.value, ast.Dict):
                                if self._is_decision_table(node):
                                    table = self._extract_decision_table(node, file_path)
                                    decision_logic["decision_tables"].append(table)
                                    
                        # Look for match/case statements (Python 3.10+)
                        elif hasattr(ast, 'Match') and isinstance(node, ast.Match):
                            switch = self._extract_match_statement(node, file_path)
                            decision_logic["switch_statements"].append(switch)
                            
            except Exception as e:
                self.logger.error(f"Error extracting decision logic from {file_path}: {e}")
                
        return decision_logic
    
    def _extract_business_events(self) -> Dict[str, Any]:
        """
        Extract business events and event handlers
        """
        events = {
            "event_definitions": [],
            "event_handlers": [],
            "event_publishers": [],
            "event_subscribers": [],
            "event_flows": [],
            "event_patterns": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Look for event classes
                        if isinstance(node, ast.ClassDef):
                            if "event" in node.name.lower():
                                event_def = self._analyze_event_class(node, file_path)
                                events["event_definitions"].append(event_def)
                                
                        # Look for event handlers
                        elif isinstance(node, ast.FunctionDef):
                            if any(pattern in node.name.lower() for pattern in ["handle", "on_", "process_event"]):
                                handler = self._analyze_event_handler(node, file_path)
                                events["event_handlers"].append(handler)
                            elif "publish" in node.name.lower() or "emit" in node.name.lower():
                                publisher = self._analyze_event_publisher(node, file_path)
                                events["event_publishers"].append(publisher)
                            elif "subscribe" in node.name.lower() or "listen" in node.name.lower():
                                subscriber = self._analyze_event_subscriber(node, file_path)
                                events["event_subscribers"].append(subscriber)
                                
                        # Look for decorator-based event handling
                        if isinstance(node, ast.FunctionDef):
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Name):
                                    if "event" in decorator.id.lower():
                                        events["event_patterns"]["decorator_based"] += 1
                                        
            except Exception as e:
                self.logger.error(f"Error extracting business events from {file_path}: {e}")
                
        # Analyze event flows
        events["event_flows"] = self._analyze_event_flows(events)
        
        return events
    
    def _extract_compliance_rules(self) -> Dict[str, Any]:
        """
        Extract compliance and regulatory rules
        """
        compliance = {
            "regulatory_rules": [],
            "audit_rules": [],
            "data_privacy_rules": [],
            "retention_policies": [],
            "compliance_checks": [],
            "compliance_frameworks": []
        }
        
        compliance_keywords = {
            "gdpr": ["gdpr", "data protection", "privacy", "consent", "right to be forgotten"],
            "pci": ["pci", "card", "payment", "credit card", "cardholder"],
            "hipaa": ["hipaa", "health", "medical", "patient", "phi"],
            "sox": ["sox", "sarbanes", "audit", "financial reporting"],
            "kyc": ["kyc", "know your customer", "identity", "verification"],
            "aml": ["aml", "anti-money", "laundering", "suspicious"]
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # Check for compliance frameworks
                for framework, keywords in compliance_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        compliance["compliance_frameworks"].append({
                            "framework": framework.upper(),
                            "location": str(file_path),
                            "indicators": [kw for kw in keywords if kw in content]
                        })
                        
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for compliance-related functions
                            if any(pattern in node.name.lower() for pattern in ["audit", "compliance", "regulatory"]):
                                rule = self._analyze_compliance_function(node, file_path)
                                
                                if "audit" in node.name.lower():
                                    compliance["audit_rules"].append(rule)
                                elif "privacy" in node.name.lower() or "gdpr" in node.name.lower():
                                    compliance["data_privacy_rules"].append(rule)
                                elif "retention" in node.name.lower():
                                    compliance["retention_policies"].append(rule)
                                else:
                                    compliance["regulatory_rules"].append(rule)
                                    
            except Exception as e:
                self.logger.error(f"Error extracting compliance rules from {file_path}: {e}")
                
        return compliance
    
    def _extract_sla_rules(self) -> Dict[str, Any]:
        """
        Extract Service Level Agreement rules
        """
        sla_rules = {
            "response_time_rules": [],
            "availability_rules": [],
            "throughput_rules": [],
            "quality_rules": [],
            "escalation_rules": [],
            "sla_metrics": []
        }
        
        sla_keywords = {
            "response_time": ["response", "latency", "timeout", "deadline"],
            "availability": ["uptime", "availability", "downtime", "maintenance"],
            "throughput": ["throughput", "requests_per", "transactions_per", "capacity"],
            "quality": ["error_rate", "success_rate", "accuracy", "quality"],
            "escalation": ["escalate", "priority", "severity", "critical"]
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for SLA-related constants
                sla_patterns = [
                    (r"SLA_\w+\s*=\s*([0-9.]+)", "sla_value"),
                    (r"TIMEOUT_\w+\s*=\s*([0-9.]+)", "timeout"),
                    (r"MAX_RESPONSE_TIME\s*=\s*([0-9.]+)", "response_time"),
                    (r"MIN_AVAILABILITY\s*=\s*([0-9.]+)", "availability")
                ]
                
                for pattern, sla_type in sla_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        sla_rules["sla_metrics"].append({
                            "type": sla_type,
                            "name": match.group(0).split('=')[0].strip(),
                            "value": match.group(1),
                            "location": str(file_path)
                        })
                        
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for SLA-related functions
                            for category, keywords in sla_keywords.items():
                                if any(keyword in node.name.lower() for keyword in keywords):
                                    sla_rule = self._analyze_sla_function(node, file_path)
                                    
                                    if category == "response_time":
                                        sla_rules["response_time_rules"].append(sla_rule)
                                    elif category == "availability":
                                        sla_rules["availability_rules"].append(sla_rule)
                                    elif category == "throughput":
                                        sla_rules["throughput_rules"].append(sla_rule)
                                    elif category == "quality":
                                        sla_rules["quality_rules"].append(sla_rule)
                                    elif category == "escalation":
                                        sla_rules["escalation_rules"].append(sla_rule)
                                        
            except Exception as e:
                self.logger.error(f"Error extracting SLA rules from {file_path}: {e}")
                
        return sla_rules
    
    def _extract_pricing_rules(self) -> Dict[str, Any]:
        """
        Extract pricing and billing rules
        """
        pricing_rules = {
            "pricing_models": [],
            "discount_rules": [],
            "tier_definitions": [],
            "billing_cycles": [],
            "pricing_calculations": [],
            "promotion_rules": []
        }
        
        pricing_keywords = {
            "pricing": ["price", "cost", "rate", "fee"],
            "discount": ["discount", "rebate", "coupon", "promo"],
            "tier": ["tier", "plan", "package", "subscription"],
            "billing": ["bill", "invoice", "payment", "charge"]
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Look for pricing functions
                        if isinstance(node, ast.FunctionDef):
                            for category, keywords in pricing_keywords.items():
                                if any(keyword in node.name.lower() for keyword in keywords):
                                    pricing_rule = self._analyze_pricing_function(node, file_path)
                                    
                                    if category == "pricing":
                                        pricing_rules["pricing_calculations"].append(pricing_rule)
                                    elif category == "discount":
                                        pricing_rules["discount_rules"].append(pricing_rule)
                                    elif category == "tier":
                                        pricing_rules["tier_definitions"].append(pricing_rule)
                                    elif category == "billing":
                                        pricing_rules["billing_cycles"].append(pricing_rule)
                                        
                        # Look for pricing models in classes
                        elif isinstance(node, ast.ClassDef):
                            if any(keyword in node.name.lower() for keyword in ["pricing", "price", "billing"]):
                                model = self._analyze_pricing_model(node, file_path)
                                pricing_rules["pricing_models"].append(model)
                                
                        # Look for promotion rules
                        elif isinstance(node, ast.If):
                            if self._is_promotion_rule(node):
                                promo = self._extract_promotion_rule(node, file_path)
                                pricing_rules["promotion_rules"].append(promo)
                                
            except Exception as e:
                self.logger.error(f"Error extracting pricing rules from {file_path}: {e}")
                
        return pricing_rules
    
    def _analyze_rule_dependencies(self) -> Dict[str, Any]:
        """
        Analyze dependencies between business rules
        """
        dependencies = {
            "rule_graph": {},
            "dependency_chains": [],
            "circular_dependencies": [],
            "rule_conflicts": [],
            "rule_hierarchy": {},
            "execution_order": []
        }
        
        # Build dependency graph from extracted rules
        rule_calls = defaultdict(set)
        
        for rule in self.business_rules:
            # Look for references to other rules
            for other_rule in self.business_rules:
                if rule != other_rule:
                    if other_rule.name in str(rule.conditions) or other_rule.name in str(rule.actions):
                        rule_calls[rule.name].add(other_rule.name)
                        
        dependencies["rule_graph"] = dict(rule_calls)
        
        # Find dependency chains
        for rule_name in rule_calls:
            chain = self._find_dependency_chain(rule_name, rule_calls)
            if len(chain) > 2:
                dependencies["dependency_chains"].append(chain)
                
        # Detect circular dependencies
        for rule_name in rule_calls:
            if self._has_circular_dependency(rule_name, rule_calls):
                dependencies["circular_dependencies"].append(rule_name)
                
        # Detect potential conflicts
        for rule1 in self.business_rules:
            for rule2 in self.business_rules:
                if rule1 != rule2:
                    if self._rules_conflict(rule1, rule2):
                        dependencies["rule_conflicts"].append({
                            "rule1": rule1.name,
                            "rule2": rule2.name,
                            "conflict_type": self._get_conflict_type(rule1, rule2)
                        })
                        
        # Determine execution order
        dependencies["execution_order"] = self._topological_sort(rule_calls)
        
        return dependencies
    
    def _generate_business_summary(self) -> Dict[str, Any]:
        """
        Generate summary of business rule analysis
        """
        summary = {
            "total_rules": len(self.business_rules),
            "rule_categories": {},
            "domain_coverage": {},
            "complexity_assessment": {},
            "documentation_quality": {},
            "recommendations": [],
            "business_insights": []
        }
        
        # Count rules by category
        categories = defaultdict(int)
        for rule in self.business_rules:
            categories[rule.rule_type] += 1
        summary["rule_categories"] = dict(categories)
        
        # Assess domain coverage
        domains_covered = set()
        for rule in self.business_rules:
            for domain, keywords in self.domain_keywords.items():
                if any(keyword in rule.name.lower() or keyword in rule.description.lower() for keyword in keywords):
                    domains_covered.add(domain)
                    
        summary["domain_coverage"] = {
            "domains_identified": list(domains_covered),
            "coverage_percentage": (len(domains_covered) / len(self.domain_keywords)) * 100 if self.domain_keywords else 0
        }
        
        # Assess complexity
        high_complexity_rules = [rule for rule in self.business_rules if len(rule.conditions) > 3]
        summary["complexity_assessment"] = {
            "simple_rules": len([r for r in self.business_rules if len(r.conditions) <= 1]),
            "moderate_rules": len([r for r in self.business_rules if 1 < len(r.conditions) <= 3]),
            "complex_rules": len(high_complexity_rules),
            "average_conditions": sum(len(r.conditions) for r in self.business_rules) / len(self.business_rules) if self.business_rules else 0
        }
        
        # Assess documentation
        documented_rules = [rule for rule in self.business_rules if rule.documentation]
        summary["documentation_quality"] = {
            "documented_rules": len(documented_rules),
            "documentation_percentage": (len(documented_rules) / len(self.business_rules)) * 100 if self.business_rules else 0,
            "quality_score": self._calculate_documentation_quality_score(documented_rules)
        }
        
        # Generate recommendations
        if summary["documentation_quality"]["documentation_percentage"] < 50:
            summary["recommendations"].append({
                "priority": "high",
                "description": "Improve business rule documentation",
                "action": "Add clear documentation to business logic functions"
            })
            
        if len(high_complexity_rules) > len(self.business_rules) * 0.3:
            summary["recommendations"].append({
                "priority": "medium",
                "description": "Simplify complex business rules",
                "action": "Break down complex rules into smaller, composable rules"
            })
            
        # Generate business insights
        if domains_covered:
            summary["business_insights"].append({
                "insight": "Primary business domains",
                "domains": list(domains_covered)[:3],
                "description": f"System primarily handles {', '.join(list(domains_covered)[:3])} operations"
            })
            
        if self.workflows:
            summary["business_insights"].append({
                "insight": "Workflow complexity",
                "workflow_count": len(self.workflows),
                "description": f"System manages {len(self.workflows)} distinct business workflows"
            })
            
        return summary
    
    # Helper methods
    def _analyze_function_for_rules(self, node: ast.FunctionDef, file_path: Path) -> Optional[BusinessRule]:
        """Analyze a function to extract business rules"""
        # Check if function name matches business rule patterns
        for rule_type, patterns in self.rule_patterns.items():
            for pattern in patterns:
                if re.search(pattern, node.name, re.IGNORECASE):
                    conditions = self._extract_conditions(node)
                    actions = self._extract_actions(node)
                    
                    return BusinessRule(
                        rule_type=rule_type,
                        name=node.name,
                        description=self._generate_rule_description(node),
                        location=f"{file_path}:{node.name}",
                        conditions=conditions,
                        actions=actions,
                        constraints=self._extract_constraints(node),
                        domain_entities=self._extract_domain_entities(node),
                        confidence=self._calculate_rule_confidence(node),
                        documentation=ast.get_docstring(node) or ""
                    )
        return None
    
    def _extract_conditions(self, node: ast.FunctionDef) -> List[str]:
        """Extract conditions from a function"""
        conditions = []
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                condition_str = self._ast_to_string(child.test)
                conditions.append(condition_str)
        return conditions
    
    def _extract_actions(self, node: ast.FunctionDef) -> List[str]:
        """Extract actions from a function"""
        actions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                if child.value:
                    actions.append(f"return {self._ast_to_string(child.value)}")
            elif isinstance(child, ast.Raise):
                actions.append(f"raise {child.exc.__class__.__name__ if child.exc else 'exception'}")
        return actions
    
    def _extract_constraints(self, node: ast.FunctionDef) -> List[str]:
        """Extract constraints from a function"""
        constraints = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                constraints.append(self._ast_to_string(child.test))
        return constraints
    
    def _extract_domain_entities(self, node: ast.FunctionDef) -> List[str]:
        """Extract domain entities referenced in a function"""
        entities = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                # Check if name matches domain keywords
                for domain, keywords in self.domain_keywords.items():
                    if any(keyword in child.id.lower() for keyword in keywords):
                        entities.add(child.id)
        return list(entities)
    
    def _calculate_rule_confidence(self, node: ast.FunctionDef) -> float:
        """Calculate confidence score for a business rule"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if has docstring
        if ast.get_docstring(node):
            confidence += 0.2
            
        # Increase confidence if has clear conditions
        if any(isinstance(child, ast.If) for child in ast.walk(node)):
            confidence += 0.15
            
        # Increase confidence if has validation/checks
        if any(isinstance(child, ast.Assert) for child in ast.walk(node)):
            confidence += 0.15
            
        return min(confidence, 1.0)
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation"""
        if hasattr(ast, 'unparse'):
            return ast.unparse(node)
        # Fallback for older Python versions
        return str(node)
    
    def _generate_rule_description(self, node: ast.FunctionDef) -> str:
        """Generate description for a business rule"""
        # Try to use docstring first
        docstring = ast.get_docstring(node)
        if docstring:
            return docstring.split('\n')[0]  # First line
            
        # Generate from function name
        description = node.name.replace('_', ' ').title()
        return f"Business rule: {description}"
    
    def _is_domain_entity(self, node: ast.ClassDef) -> bool:
        """Check if a class represents a domain entity"""
        # Check for entity-like attributes
        has_id = False
        has_business_methods = False
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                if child.name == "__init__":
                    # Check for id attribute
                    for arg in child.args.args:
                        if "id" in arg.arg.lower():
                            has_id = True
                elif not child.name.startswith("_"):
                    # Public method that might be business logic
                    has_business_methods = True
                    
        return has_id or has_business_methods
    
    def _find_dependency_chain(self, start: str, graph: Dict[str, Set[str]]) -> List[str]:
        """Find dependency chain starting from a rule"""
        chain = [start]
        current = start
        visited = set()
        
        while current in graph and current not in visited:
            visited.add(current)
            if graph[current]:
                next_rule = list(graph[current])[0]
                if next_rule not in chain:
                    chain.append(next_rule)
                    current = next_rule
                else:
                    break
            else:
                break
                
        return chain
    
    def _has_circular_dependency(self, start: str, graph: Dict[str, Set[str]]) -> bool:
        """Check if there's a circular dependency"""
        visited = set()
        stack = [start]
        
        while stack:
            current = stack.pop()
            if current in visited:
                return True
            visited.add(current)
            
            if current in graph:
                stack.extend(graph[current])
                
        return False
    
    def _rules_conflict(self, rule1: BusinessRule, rule2: BusinessRule) -> bool:
        """Check if two rules conflict"""
        # Check for conflicting conditions
        for cond1 in rule1.conditions:
            for cond2 in rule2.conditions:
                if self._conditions_conflict(cond1, cond2):
                    return True
        return False
    
    def _conditions_conflict(self, cond1: str, cond2: str) -> bool:
        """Check if two conditions conflict"""
        # Simple conflict detection - could be enhanced
        if "not" in cond1 and cond2 in cond1:
            return True
        if "not" in cond2 and cond1 in cond2:
            return True
        return False
    
    def _get_conflict_type(self, rule1: BusinessRule, rule2: BusinessRule) -> str:
        """Determine type of conflict between rules"""
        if rule1.rule_type == rule2.rule_type:
            return "same_type_conflict"
        if set(rule1.domain_entities) & set(rule2.domain_entities):
            return "domain_conflict"
        return "general_conflict"
    
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Perform topological sort on dependency graph"""
        in_degree = defaultdict(int)
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
                
        queue = [node for node in graph if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return result
    
    def _calculate_documentation_quality_score(self, documented_rules: List[BusinessRule]) -> float:
        """Calculate quality score for documentation"""
        if not documented_rules:
            return 0.0
            
        total_score = 0
        for rule in documented_rules:
            doc_length = len(rule.documentation)
            if doc_length > 100:
                total_score += 1.0
            elif doc_length > 50:
                total_score += 0.7
            elif doc_length > 20:
                total_score += 0.4
            else:
                total_score += 0.2
                
        return (total_score / len(documented_rules)) * 100