"""
Business Rule Analysis Module
Extracts business rules, workflows, and domain logic from code
Core Analysis Module - Split from business_analyzer.py
"""

import ast
import re
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json
import logging


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


class BusinessAnalyzerCore:
    """Core business analysis functionality extracted from original business_analyzer.py"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def extract_pricing_rules(self, file_path: Path, tree: ast.AST) -> Dict[str, List[Dict]]:
        """Extract pricing-related business rules from AST"""
        pricing_rules = {
            "pricing_calculations": [],
            "discount_rules": [],
            "tier_definitions": [],
            "billing_cycles": [],
            "pricing_models": [],
            "promotion_rules": []
        }
        
        pricing_keywords = {
            "pricing": ["price", "cost", "rate", "fee"],
            "discount": ["discount", "coupon", "promo"],
            "tier": ["tier", "level", "grade", "category"],
            "billing": ["billing", "invoice", "payment", "charge"]
        }
        
        try:
            for node in ast.walk(tree):
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
        """Analyze dependencies between business rules"""
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
        
        return dependencies
    
    def _generate_business_summary(self) -> Dict[str, Any]:
        """Generate summary of business rule analysis"""
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
    
    # Helper methods - simplified implementations for standalone operation
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
        return str(type(node).__name__)
    
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
    
    # Simplified placeholder methods for complex analysis functions
    def _analyze_class_for_rules(self, node: ast.ClassDef, file_path: Path) -> List[BusinessRule]:
        """Analyze a class for business rules - simplified implementation"""
        return []
    
    def _extract_conditional_rule(self, node: ast.If, file_path: Path) -> Optional[BusinessRule]:
        """Extract business rule from conditional - simplified implementation"""
        return None
    
    def _analyze_validation_function(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict]:
        """Analyze validation function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_calculation_function(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict]:
        """Analyze calculation function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _extract_formula(self, node: ast.Assign, file_path: Path) -> Optional[Dict]:
        """Extract formula from assignment - simplified implementation"""
        return None
    
    def _analyze_authorization_function(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict]:
        """Analyze authorization function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_workflow_class(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze workflow class - simplified implementation"""
        return {"class": node.name, "location": str(file_path)}
    
    def _analyze_workflow_step(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze workflow step - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _detect_state_transition(self, node: ast.If, file_path: Path) -> Optional[Dict]:
        """Detect state transition - simplified implementation"""
        return None
    
    def _analyze_state_machine(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze state machine - simplified implementation"""
        return {"class": node.name, "location": str(file_path)}
    
    def _extract_states(self, node: ast.Assign) -> List[Dict]:
        """Extract states from assignment - simplified implementation"""
        return []
    
    def _analyze_transition(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze transition function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_guard(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze guard function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_state_action(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze state action - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_domain_entity(self, node: ast.ClassDef, file_path: Path) -> Optional[Dict]:
        """Analyze domain entity - simplified implementation"""
        if self._is_domain_entity(node):
            return {"class": node.name, "location": str(file_path)}
        return None
    
    def _extract_entity_relationships(self, node: ast.ClassDef, file_path: Path) -> List[Dict]:
        """Extract entity relationships - simplified implementation"""
        return []
    
    def _extract_invariants(self, node: ast.ClassDef, file_path: Path) -> List[Dict]:
        """Extract invariants - simplified implementation"""
        return []
    
    def _extract_domain_event(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict]:
        """Extract domain event - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _extract_constraint_logic(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict]:
        """Extract constraint logic - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_decision_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze decision function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _is_complex_decision(self, node: ast.If) -> bool:
        """Check if conditional is complex decision - simplified implementation"""
        return False
    
    def _extract_decision_tree(self, node: ast.If, file_path: Path) -> Dict:
        """Extract decision tree - simplified implementation"""
        return {"location": str(file_path)}
    
    def _is_decision_table(self, node: ast.Assign) -> bool:
        """Check if assignment is decision table - simplified implementation"""
        return False
    
    def _extract_decision_table(self, node: ast.Assign, file_path: Path) -> Dict:
        """Extract decision table - simplified implementation"""
        return {"location": str(file_path)}
    
    def _extract_match_statement(self, node, file_path: Path) -> Dict:
        """Extract match statement - simplified implementation"""
        return {"location": str(file_path)}
    
    def _analyze_event_class(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze event class - simplified implementation"""
        return {"class": node.name, "location": str(file_path)}
    
    def _analyze_event_handler(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze event handler - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_event_publisher(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze event publisher - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_event_subscriber(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze event subscriber - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_event_flows(self, events: Dict) -> List[Dict]:
        """Analyze event flows - simplified implementation"""
        return []
    
    def _analyze_compliance_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze compliance function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_sla_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze SLA function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_pricing_function(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze pricing function - simplified implementation"""
        return {"function": node.name, "location": str(file_path)}
    
    def _analyze_pricing_model(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze pricing model - simplified implementation"""
        return {"class": node.name, "location": str(file_path)}
    
    def _is_promotion_rule(self, node: ast.If) -> bool:
        """Check if conditional is promotion rule - simplified implementation"""
        return False
    
    def _extract_promotion_rule(self, node: ast.If, file_path: Path) -> Dict:
        """Extract promotion rule - simplified implementation"""
        return {"location": str(file_path)}