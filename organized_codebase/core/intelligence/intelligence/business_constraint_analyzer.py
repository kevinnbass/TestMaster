"""
Business Constraint Analysis Component
======================================

Analyzes business constraints, compliance rules, SLA requirements, and pricing rules.
Part of modularized business_analyzer system.
"""

import ast
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

from .business_base import (
    BusinessConstraint, BusinessAnalysisConfiguration
)


class BusinessConstraintAnalyzer:
    """Analyzes business constraints and compliance requirements"""
    
    def __init__(self, config: BusinessAnalysisConfiguration):
        self.config = config
    
    def extract_business_constraints(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract business constraints and limits"""
        constraints = {
            "numeric_constraints": [],
            "temporal_constraints": [],
            "capacity_constraints": [],
            "relationship_constraints": [],
            "business_invariants": [],
            "constraint_violations": []
        }
        
        for file_path in python_files:
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
                print(f"Error extracting constraints from {file_path}: {e}")
                
        return constraints
    
    def extract_compliance_rules(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract compliance and regulatory rules"""
        compliance = {
            "regulatory_rules": [],
            "audit_rules": [],
            "data_privacy_rules": [],
            "retention_policies": [],
            "compliance_checks": [],
            "compliance_frameworks": []
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # Check for compliance frameworks
                for framework, keywords in self.config.compliance_keywords.items():
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
                print(f"Error extracting compliance rules from {file_path}: {e}")
                
        return compliance
    
    def extract_sla_rules(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract Service Level Agreement rules"""
        sla_rules = {
            "response_time_rules": [],
            "availability_rules": [],
            "throughput_rules": [],
            "quality_rules": [],
            "escalation_rules": [],
            "sla_metrics": []
        }
        
        for file_path in python_files:
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
                            for category, keywords in self.config.sla_keywords.items():
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
                print(f"Error extracting SLA rules from {file_path}: {e}")
                
        return sla_rules
    
    def extract_pricing_rules(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract pricing and billing rules"""
        pricing_rules = {
            "pricing_models": [],
            "discount_rules": [],
            "tier_definitions": [],
            "billing_cycles": [],
            "pricing_calculations": [],
            "promotion_rules": []
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Look for pricing functions
                        if isinstance(node, ast.FunctionDef):
                            for category, keywords in self.config.pricing_keywords.items():
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
                print(f"Error extracting pricing rules from {file_path}: {e}")
                
        return pricing_rules
    
    def extract_decision_logic(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract decision tables and business logic"""
        decision_logic = {
            "decision_tables": [],
            "decision_trees": [],
            "rule_chains": [],
            "conditional_logic": [],
            "switch_statements": [],
            "lookup_tables": []
        }
        
        for file_path in python_files:
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
                print(f"Error extracting decision logic from {file_path}: {e}")
                
        return decision_logic
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file and return its AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content)
        except Exception:
            return None
    
    # Simplified placeholder methods for complex analysis functions
    def _extract_constraint_logic(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict]:
        """Extract constraint logic"""
        return {"function": node.name, "location": str(file_path), "type": "constraint"}
    
    def _analyze_compliance_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze compliance function"""
        return {"function": node.name, "location": str(file_path), "type": "compliance"}
    
    def _analyze_sla_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze SLA function"""
        return {"function": node.name, "location": str(file_path), "type": "sla"}
    
    def _analyze_pricing_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze pricing function"""
        return {"function": node.name, "location": str(file_path), "type": "pricing"}
    
    def _analyze_pricing_model(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze pricing model"""
        return {"class": node.name, "location": str(file_path), "type": "pricing_model"}
    
    def _is_promotion_rule(self, node: ast.If) -> bool:
        """Check if conditional is promotion rule"""
        condition_str = str(node.test) if hasattr(node, 'test') else ""
        return any(promo_word in condition_str.lower() for promo_word in ["discount", "promo", "coupon", "sale"])
    
    def _extract_promotion_rule(self, node: ast.If, file_path: Path) -> Dict:
        """Extract promotion rule"""
        return {"location": str(file_path), "type": "promotion"}
    
    def _analyze_decision_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze decision function"""
        return {"function": node.name, "location": str(file_path), "type": "decision"}
    
    def _is_complex_decision(self, node: ast.If) -> bool:
        """Check if conditional is complex decision"""
        # Count nested conditions
        nested_count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.If) and child != node:
                nested_count += 1
        return nested_count > 2
    
    def _extract_decision_tree(self, node: ast.If, file_path: Path) -> Dict:
        """Extract decision tree"""
        return {"location": str(file_path), "type": "decision_tree"}
    
    def _is_decision_table(self, node: ast.Assign) -> bool:
        """Check if assignment is decision table"""
        # Check if dictionary has decision-like structure
        if isinstance(node.value, ast.Dict):
            return len(node.value.keys) > 3  # Simple heuristic
        return False
    
    def _extract_decision_table(self, node: ast.Assign, file_path: Path) -> Dict:
        """Extract decision table"""
        return {"location": str(file_path), "type": "decision_table"}
    
    def _extract_match_statement(self, node, file_path: Path) -> Dict:
        """Extract match statement"""
        return {"location": str(file_path), "type": "match_statement"}


# Export
__all__ = ['BusinessConstraintAnalyzer']