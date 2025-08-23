"""
Business Rule Analysis Module
Extracts business rules, workflows, and domain logic from code
"""Analysis Module - Split from business_analyzer.py"""


import ast
import re
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json
import logging



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
        """Extract domain model and entities"""
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
        """Extract business constraints and limits"""
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
        """Extract decision tables and business logic"""
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
        """Extract business events and event handlers"""
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
        """Extract compliance and regulatory rules"""
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
        """Extract Service Level Agreement rules"""
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
        """Extract pricing and billing rules"""
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
                        pass
            except Exception as e:
                continue
        
        return pricing_rules