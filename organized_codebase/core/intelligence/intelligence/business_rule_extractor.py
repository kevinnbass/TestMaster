"""
Business Rule Extraction Component
==================================

Extracts business rules from code using AST analysis and pattern matching.
Part of modularized business_analyzer system.
"""

import ast
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

from .business_base import (
    BusinessRule, BusinessRuleType, BusinessAnalysisConfiguration
)


class BusinessRuleExtractor:
    """Extracts business rules from Python source code"""
    
    def __init__(self, config: BusinessAnalysisConfiguration):
        self.config = config
        self.extracted_rules = []
    
    def extract_business_rules(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract general business rules from code files"""
        rules_extracted = {
            "rules": [],
            "rule_categories": defaultdict(list),
            "confidence_levels": defaultdict(int),
            "documentation_coverage": 0,
            "total_rules": 0
        }
        
        self.extracted_rules = []
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Extract rules from functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            rule = self._analyze_function_for_rules(node, file_path)
                            if rule:
                                self.extracted_rules.append(rule)
                                rules_extracted["rules"].append(rule)
                                rules_extracted["rule_categories"][rule.rule_type].append(rule)
                                
                        # Extract rules from classes
                        elif isinstance(node, ast.ClassDef):
                            class_rules = self._analyze_class_for_rules(node, file_path)
                            self.extracted_rules.extend(class_rules)
                            rules_extracted["rules"].extend(class_rules)
                            
                        # Extract rules from conditionals
                        elif isinstance(node, ast.If):
                            conditional_rule = self._extract_conditional_rule(node, file_path)
                            if conditional_rule:
                                self.extracted_rules.append(conditional_rule)
                                rules_extracted["rules"].append(conditional_rule)
                                
            except Exception as e:
                print(f"Error extracting business rules from {file_path}: {e}")
                
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
    
    def extract_validation_rules(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract validation-specific business rules"""
        validation_rules = {
            "field_validations": [],
            "cross_field_validations": [],
            "business_validations": [],
            "format_validations": [],
            "range_validations": [],
            "custom_validations": []
        }
        
        for file_path in python_files:
            try:
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
                print(f"Error extracting validation rules from {file_path}: {e}")
                
        return validation_rules
    
    def extract_calculation_rules(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract calculation and computation business rules"""
        calculation_rules = {
            "financial_calculations": [],
            "pricing_calculations": [],
            "tax_calculations": [],
            "discount_calculations": [],
            "scoring_calculations": [],
            "aggregation_rules": [],
            "formula_definitions": []
        }
        
        for file_path in python_files:
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
                print(f"Error extracting calculation rules from {file_path}: {e}")
                
        return calculation_rules
    
    def extract_authorization_rules(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract authorization and access control rules"""
        auth_rules = {
            "permission_checks": [],
            "role_based_rules": [],
            "attribute_based_rules": [],
            "context_based_rules": [],
            "delegation_rules": [],
            "access_policies": []
        }
        
        for file_path in python_files:
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
                print(f"Error extracting authorization rules from {file_path}: {e}")
                
        return auth_rules
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file and return its AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content)
        except Exception:
            return None
    
    def _analyze_function_for_rules(self, node: ast.FunctionDef, file_path: Path) -> Optional[BusinessRule]:
        """Analyze a function to extract business rules"""
        # Check if function name matches business rule patterns
        for rule_type, patterns in self.config.rule_patterns.items():
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
                for domain, keywords in self.config.domain_keywords.items():
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
    
    # Placeholder methods for specialized rule analysis
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


# Export
__all__ = ['BusinessRuleExtractor']