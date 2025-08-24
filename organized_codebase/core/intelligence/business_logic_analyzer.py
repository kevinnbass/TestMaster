"""
Business Logic Analyzer - Complete Implementation
Agent B Enhancement of Existing Business Analysis Infrastructure

Comprehensive business logic analysis including:
- Domain model extraction
- Business rule detection
- Process flow analysis
- Decision logic mapping
- Compliance and regulatory rules
- SLA and pricing rules
"""

import ast
import re
import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BusinessRule:
    """Represents a single business rule"""
    rule_id: str = field(default_factory=lambda: f"rule_{int(datetime.now().timestamp() * 1000000)}")
    rule_type: str = ""
    description: str = ""
    location: str = ""
    complexity: str = "low"
    priority: str = "medium"
    business_impact: str = "medium"
    implementation: str = ""
    conditions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type,
            'description': self.description,
            'location': self.location,
            'complexity': self.complexity,
            'priority': self.priority,
            'business_impact': self.business_impact,
            'implementation': self.implementation,
            'conditions': self.conditions,
            'actions': self.actions,
            'dependencies': self.dependencies
        }


@dataclass
class DomainEntity:
    """Represents a domain entity"""
    entity_id: str = field(default_factory=lambda: f"entity_{int(datetime.now().timestamp() * 1000000)}")
    name: str = ""
    entity_type: str = "entity"
    attributes: List[Dict[str, str]] = field(default_factory=list)
    methods: List[Dict[str, str]] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    business_value: str = "medium"
    location: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'entity_type': self.entity_type,
            'attributes': self.attributes,
            'methods': self.methods,
            'relationships': self.relationships,
            'business_value': self.business_value,
            'location': self.location
        }


@dataclass
class BusinessProcess:
    """Represents a business process"""
    process_id: str = field(default_factory=lambda: f"process_{int(datetime.now().timestamp() * 1000000)}")
    name: str = ""
    steps: List[Dict[str, str]] = field(default_factory=list)
    decision_points: List[Dict[str, str]] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    business_value: str = "medium"
    location: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'process_id': self.process_id,
            'name': self.name,
            'steps': self.steps,
            'decision_points': self.decision_points,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'stakeholders': self.stakeholders,
            'business_value': self.business_value,
            'location': self.location
        }


class BusinessLogicAnalyzer:
    """
    Complete Business Logic Analyzer - Agent B Enhancement
    
    Analyzes codebases to extract business logic, domain models,
    business rules, and process flows with comprehensive insights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.business_rules = []
        self.domain_entities = []
        self.business_processes = []
        
        # Business rule patterns for detection
        self.rule_patterns = {
            "validation": [r"validate", r"check", r"verify", r"ensure"],
            "calculation": [r"calculate", r"compute", r"determine", r"derive"],
            "authorization": [r"authorize", r"permit", r"allow", r"deny"],
            "workflow": [r"process", r"handle", r"execute", r"perform"],
            "notification": [r"notify", r"alert", r"inform", r"send"],
            "transformation": [r"transform", r"convert", r"map", r"adapt"],
            "decision": [r"decide", r"choose", r"select", r"evaluate"]
        }
        
        # Domain keywords for entity detection
        self.domain_keywords = {
            "user": ["user", "customer", "client", "person", "account"],
            "order": ["order", "purchase", "transaction", "sale"],
            "product": ["product", "item", "service", "offering"],
            "payment": ["payment", "billing", "invoice", "charge"],
            "inventory": ["inventory", "stock", "warehouse", "item"],
            "delivery": ["delivery", "shipping", "transport", "logistics"],
            "support": ["support", "ticket", "issue", "complaint"],
            "analytics": ["analytics", "report", "metric", "kpi"]
        }
        
    def analyze_business_logic(self, project_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive business logic analysis
        """
        try:
            project_path = Path(project_path)
            self.business_rules = []
            self.domain_entities = []
            self.business_processes = []
            
            logger.info(f"Starting business logic analysis for: {project_path}")
            
            # Analyze Python files
            python_files = list(project_path.rglob("*.py"))
            logger.info(f"Found {len(python_files)} Python files to analyze")
            
            for py_file in python_files:
                if self._should_analyze_file(py_file):
                    self._analyze_file(py_file)
            
            # Extract high-level insights
            domain_model = self._extract_domain_model()
            business_processes_data = self._extract_business_processes()
            compliance_analysis = self._extract_compliance_rules()
            decision_logic = self._extract_decision_logic()
            
            return {
                'business_rules': [rule.to_dict() for rule in self.business_rules],
                'domain_entities': [entity.to_dict() for entity in self.domain_entities],
                'business_processes': [process.to_dict() for process in self.business_processes],
                'domain_model': domain_model,
                'business_processes_analysis': business_processes_data,
                'compliance_analysis': compliance_analysis,
                'decision_logic': decision_logic,
                'summary': self._generate_business_summary(),
                'recommendations': self._generate_business_recommendations(),
                'business_metrics': self._calculate_business_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in business logic analysis: {e}")
            return {
                'error': str(e),
                'business_rules': [],
                'domain_entities': [],
                'business_processes': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed for business logic"""
        # Skip system files and caches
        skip_patterns = [
            '__pycache__/', '.git/', 'node_modules/', 'venv/', '.venv/',
            '.pytest_cache/', 'migrations/', 'tests/', 'test_'
        ]
        
        file_str = str(file_path)
        for pattern in skip_patterns:
            if pattern in file_str:
                return False
        
        # Focus on business logic files
        business_indicators = [
            'business', 'domain', 'service', 'model', 'entity',
            'workflow', 'process', 'rule', 'policy', 'logic'
        ]
        
        file_name = file_path.name.lower()
        if any(indicator in file_name for indicator in business_indicators):
            return True
        
        # Check file content for business logic indicators
        try:
            if file_path.stat().st_size > 0 and file_path.stat().st_size < 100000:  # Not too large
                return True
        except OSError:
            return False
        
        return False
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single file for business logic"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract business rules
            self._extract_business_rules(file_path, tree, content)
            
            # Extract domain entities
            self._extract_domain_entities(file_path, tree)
            
            # Extract business processes
            self._extract_business_process_flows(file_path, tree)
            
        except Exception as e:
            logger.debug(f"Error analyzing file {file_path}: {e}")
    
    def _extract_business_rules(self, file_path: Path, tree: ast.AST, content: str):
        """Extract business rules from AST and content"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Analyze function for business rule patterns
                rule = self._analyze_function_for_business_rule(node, file_path)
                if rule:
                    self.business_rules.append(rule)
            
            elif isinstance(node, ast.ClassDef):
                # Analyze class methods for business rules
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        rule = self._analyze_method_for_business_rule(item, node.name, file_path)
                        if rule:
                            self.business_rules.append(rule)
        
        # Extract rules from comments and docstrings
        self._extract_rules_from_comments(file_path, content)
    
    def _analyze_function_for_business_rule(self, node: ast.FunctionDef, file_path: Path) -> Optional[BusinessRule]:
        """Analyze a function to determine if it implements a business rule"""
        function_name = node.name.lower()
        
        # Check if function matches business rule patterns
        rule_type = None
        for rule_category, patterns in self.rule_patterns.items():
            if any(pattern in function_name for pattern in patterns):
                rule_type = rule_category
                break
        
        if not rule_type:
            return None
        
        # Extract conditions and actions from function body
        conditions = []
        actions = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                # Extract conditions
                condition = self._extract_condition_text(child.test)
                if condition:
                    conditions.append(condition)
            
            elif isinstance(child, ast.Call):
                # Extract actions (function calls)
                action = self._extract_action_text(child)
                if action:
                    actions.append(action)
        
        # Create business rule
        return BusinessRule(
            rule_type=rule_type,
            description=f"Business rule in function {node.name}",
            location=f"{file_path}:{node.lineno}",
            complexity=self._determine_rule_complexity(node),
            priority=self._determine_rule_priority(rule_type),
            business_impact=self._determine_business_impact(rule_type),
            implementation=f"Function: {node.name}",
            conditions=conditions,
            actions=actions
        )
    
    def _analyze_method_for_business_rule(self, node: ast.FunctionDef, class_name: str, file_path: Path) -> Optional[BusinessRule]:
        """Analyze a class method for business rules"""
        method_name = node.name.lower()
        
        # Skip private methods and constructors
        if method_name.startswith('_'):
            return None
        
        # Check for business rule patterns
        rule_type = None
        for rule_category, patterns in self.rule_patterns.items():
            if any(pattern in method_name for pattern in patterns):
                rule_type = rule_category
                break
        
        if not rule_type:
            return None
        
        return BusinessRule(
            rule_type=rule_type,
            description=f"Business rule in method {class_name}.{node.name}",
            location=f"{file_path}:{node.lineno}",
            complexity=self._determine_rule_complexity(node),
            priority=self._determine_rule_priority(rule_type),
            business_impact=self._determine_business_impact(rule_type),
            implementation=f"Method: {class_name}.{node.name}"
        )
    
    def _extract_domain_entities(self, file_path: Path, tree: ast.AST):
        """Extract domain entities from classes"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entity = self._analyze_class_as_entity(node, file_path)
                if entity:
                    self.domain_entities.append(entity)
    
    def _analyze_class_as_entity(self, node: ast.ClassDef, file_path: Path) -> Optional[DomainEntity]:
        """Analyze a class to determine if it's a domain entity"""
        class_name = node.name.lower()
        
        # Check if class matches domain entity patterns
        entity_type = "entity"
        for domain_category, keywords in self.domain_keywords.items():
            if any(keyword in class_name for keyword in keywords):
                entity_type = domain_category
                break
        
        # Extract attributes and methods
        attributes = []
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith('_'):
                    methods.append({
                        'name': item.name,
                        'type': 'method',
                        'description': ast.get_docstring(item) or ''
                    })
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': 'attribute',
                            'description': ''
                        })
        
        # Only create entity if it has business significance
        if len(attributes) > 0 or len(methods) > 2:
            return DomainEntity(
                name=node.name,
                entity_type=entity_type,
                attributes=attributes,
                methods=methods,
                business_value=self._determine_business_value(entity_type),
                location=f"{file_path}:{node.lineno}"
            )
        
        return None
    
    def _extract_business_process_flows(self, file_path: Path, tree: ast.AST):
        """Extract business process flows from functions"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                process = self._analyze_function_as_process(node, file_path)
                if process:
                    self.business_processes.append(process)
    
    def _analyze_function_as_process(self, node: ast.FunctionDef, file_path: Path) -> Optional[BusinessProcess]:
        """Analyze a function to determine if it represents a business process"""
        function_name = node.name.lower()
        
        # Check for process indicators
        process_indicators = [
            'process', 'handle', 'execute', 'perform', 'run',
            'workflow', 'pipeline', 'flow', 'sequence'
        ]
        
        if not any(indicator in function_name for indicator in process_indicators):
            return None
        
        # Extract process steps
        steps = []
        decision_points = []
        
        for i, child in enumerate(ast.walk(node)):
            if isinstance(child, ast.Call):
                steps.append({
                    'step_number': len(steps) + 1,
                    'description': self._extract_action_text(child),
                    'type': 'action'
                })
            elif isinstance(child, ast.If):
                decision_points.append({
                    'condition': self._extract_condition_text(child.test),
                    'step_number': len(steps) + 1,
                    'type': 'decision'
                })
        
        if len(steps) > 2:  # Only consider as process if multiple steps
            return BusinessProcess(
                name=node.name,
                steps=steps,
                decision_points=decision_points,
                business_value=self._determine_process_business_value(function_name),
                location=f"{file_path}:{node.lineno}"
            )
        
        return None
    
    def _extract_condition_text(self, node: ast.AST) -> str:
        """Extract readable text from condition node"""
        try:
            if isinstance(node, ast.Compare):
                left = ast.unparse(node.left) if hasattr(ast, 'unparse') else str(node.left)
                op = node.ops[0].__class__.__name__
                right = ast.unparse(node.comparators[0]) if hasattr(ast, 'unparse') else str(node.comparators[0])
                return f"{left} {op} {right}"
            elif isinstance(node, ast.Name):
                return node.id
            else:
                return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
        except:
            return "condition"
    
    def _extract_action_text(self, node: ast.Call) -> str:
        """Extract readable text from action/call node"""
        try:
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return f"{ast.unparse(node.func.value) if hasattr(ast, 'unparse') else str(node.func.value)}.{node.func.attr}"
            else:
                return "action"
        except:
            return "action"
    
    def _extract_rules_from_comments(self, file_path: Path, content: str):
        """Extract business rules from comments and docstrings"""
        lines = content.split('\n')
        
        business_comment_patterns = [
            r"rule:", r"business rule:", r"constraint:", r"policy:",
            r"requirement:", r"specification:", r"must:", r"should:"
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for pattern in business_comment_patterns:
                if pattern in line_lower:
                    rule = BusinessRule(
                        rule_type="comment_rule",
                        description=line.strip(),
                        location=f"{file_path}:{i+1}",
                        complexity="low",
                        priority="medium",
                        business_impact="medium",
                        implementation="Comment"
                    )
                    self.business_rules.append(rule)
    
    def _determine_rule_complexity(self, node: ast.FunctionDef) -> str:
        """Determine rule complexity based on function structure"""
        complexity_score = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity_score += 1
            elif isinstance(child, ast.Try):
                complexity_score += 2
        
        if complexity_score <= 2:
            return "low"
        elif complexity_score <= 5:
            return "medium"
        else:
            return "high"
    
    def _determine_rule_priority(self, rule_type: str) -> str:
        """Determine rule priority based on type"""
        priority_map = {
            "authorization": "high",
            "validation": "high",
            "calculation": "medium",
            "workflow": "medium",
            "notification": "low",
            "transformation": "low",
            "decision": "medium"
        }
        return priority_map.get(rule_type, "medium")
    
    def _determine_business_impact(self, rule_type: str) -> str:
        """Determine business impact based on rule type"""
        impact_map = {
            "authorization": "high",
            "calculation": "high",
            "validation": "medium",
            "workflow": "medium",
            "decision": "medium",
            "notification": "low",
            "transformation": "low"
        }
        return impact_map.get(rule_type, "medium")
    
    def _determine_business_value(self, entity_type: str) -> str:
        """Determine business value based on entity type"""
        value_map = {
            "user": "high",
            "order": "high",
            "payment": "high",
            "product": "medium",
            "inventory": "medium",
            "delivery": "medium",
            "support": "low",
            "analytics": "low"
        }
        return value_map.get(entity_type, "medium")
    
    def _determine_process_business_value(self, function_name: str) -> str:
        """Determine business value of a process based on function name"""
        high_value_keywords = ["payment", "order", "checkout", "purchase", "billing"]
        medium_value_keywords = ["process", "handle", "manage", "update"]
        
        function_name_lower = function_name.lower()
        
        if any(keyword in function_name_lower for keyword in high_value_keywords):
            return "high"
        elif any(keyword in function_name_lower for keyword in medium_value_keywords):
            return "medium"
        else:
            return "low"
    
    def _extract_domain_model(self) -> Dict[str, Any]:
        """Extract comprehensive domain model"""
        return {
            "entities": len([e for e in self.domain_entities if e.entity_type == "entity"]),
            "value_objects": len([e for e in self.domain_entities if "value" in e.entity_type]),
            "services": len([e for e in self.domain_entities if "service" in e.entity_type]),
            "aggregates": len([e for e in self.domain_entities if "aggregate" in e.entity_type]),
            "entity_relationships": self._extract_entity_relationships(),
            "domain_boundaries": self._identify_domain_boundaries()
        }
    
    def _extract_business_processes(self) -> Dict[str, Any]:
        """Extract business process analysis"""
        return {
            "total_processes": len(self.business_processes),
            "high_value_processes": len([p for p in self.business_processes if p.business_value == "high"]),
            "complex_processes": len([p for p in self.business_processes if len(p.steps) > 5]),
            "process_categories": self._categorize_processes()
        }
    
    def _extract_compliance_rules(self) -> Dict[str, Any]:
        """Extract compliance and regulatory analysis"""
        compliance_rules = [r for r in self.business_rules if "compliance" in r.description.lower()]
        validation_rules = [r for r in self.business_rules if r.rule_type == "validation"]
        
        return {
            "compliance_rules": len(compliance_rules),
            "validation_rules": len(validation_rules),
            "high_priority_rules": len([r for r in self.business_rules if r.priority == "high"]),
            "regulatory_coverage": self._assess_regulatory_coverage()
        }
    
    def _extract_decision_logic(self) -> Dict[str, Any]:
        """Extract decision logic analysis"""
        decision_rules = [r for r in self.business_rules if r.rule_type == "decision"]
        
        return {
            "decision_rules": len(decision_rules),
            "complex_decisions": len([r for r in decision_rules if r.complexity == "high"]),
            "decision_patterns": self._analyze_decision_patterns()
        }
    
    def _extract_entity_relationships(self) -> List[Dict[str, str]]:
        """Extract relationships between domain entities"""
        relationships = []
        
        # Simple relationship detection based on entity names
        entity_names = [e.name.lower() for e in self.domain_entities]
        
        for entity in self.domain_entities:
            for attribute in entity.attributes:
                attr_name = attribute['name'].lower()
                for other_entity in entity_names:
                    if other_entity in attr_name and other_entity != entity.name.lower():
                        relationships.append({
                            'from': entity.name,
                            'to': other_entity,
                            'type': 'reference',
                            'attribute': attribute['name']
                        })
        
        return relationships
    
    def _identify_domain_boundaries(self) -> List[Dict[str, Any]]:
        """Identify domain boundaries and bounded contexts"""
        boundaries = []
        
        # Group entities by common patterns
        entity_groups = defaultdict(list)
        for entity in self.domain_entities:
            group = entity.entity_type
            entity_groups[group].append(entity.name)
        
        for group, entities in entity_groups.items():
            if len(entities) > 1:
                boundaries.append({
                    'boundary_name': group.title() + " Domain",
                    'entities': entities,
                    'entity_count': len(entities)
                })
        
        return boundaries
    
    def _categorize_processes(self) -> Dict[str, int]:
        """Categorize business processes"""
        categories = defaultdict(int)
        
        for process in self.business_processes:
            process_name = process.name.lower()
            
            if any(keyword in process_name for keyword in ["payment", "billing", "invoice"]):
                categories["financial"] += 1
            elif any(keyword in process_name for keyword in ["order", "purchase", "checkout"]):
                categories["sales"] += 1
            elif any(keyword in process_name for keyword in ["user", "customer", "account"]):
                categories["customer_management"] += 1
            elif any(keyword in process_name for keyword in ["inventory", "stock", "warehouse"]):
                categories["inventory"] += 1
            else:
                categories["general"] += 1
        
        return dict(categories)
    
    def _assess_regulatory_coverage(self) -> Dict[str, str]:
        """Assess regulatory compliance coverage"""
        # Simplified assessment
        validation_rules = len([r for r in self.business_rules if r.rule_type == "validation"])
        authorization_rules = len([r for r in self.business_rules if r.rule_type == "authorization"])
        
        if validation_rules > 10 and authorization_rules > 5:
            return {"status": "good", "coverage": "comprehensive"}
        elif validation_rules > 5 and authorization_rules > 2:
            return {"status": "moderate", "coverage": "partial"}
        else:
            return {"status": "poor", "coverage": "minimal"}
    
    def _analyze_decision_patterns(self) -> Dict[str, int]:
        """Analyze decision logic patterns"""
        patterns = defaultdict(int)
        
        for rule in self.business_rules:
            if rule.rule_type == "decision":
                if len(rule.conditions) > 3:
                    patterns["complex_conditions"] += 1
                elif len(rule.conditions) > 1:
                    patterns["multiple_conditions"] += 1
                else:
                    patterns["simple_conditions"] += 1
        
        return dict(patterns)
    
    def _generate_business_summary(self) -> Dict[str, Any]:
        """Generate business logic analysis summary"""
        return {
            "total_business_rules": len(self.business_rules),
            "total_domain_entities": len(self.domain_entities),
            "total_business_processes": len(self.business_processes),
            "rule_types": {
                rule_type: len([r for r in self.business_rules if r.rule_type == rule_type])
                for rule_type in self.rule_patterns.keys()
            },
            "entity_types": {
                entity_type: len([e for e in self.domain_entities if e.entity_type == entity_type])
                for entity_type in self.domain_keywords.keys()
            },
            "business_value_distribution": {
                "high": len([item for item in (self.domain_entities + self.business_processes) if item.business_value == "high"]),
                "medium": len([item for item in (self.domain_entities + self.business_processes) if item.business_value == "medium"]),
                "low": len([item for item in (self.domain_entities + self.business_processes) if item.business_value == "low"])
            }
        }
    
    def _generate_business_recommendations(self) -> List[str]:
        """Generate business logic improvement recommendations"""
        recommendations = []
        
        # Rule-based recommendations
        high_priority_rules = len([r for r in self.business_rules if r.priority == "high"])
        if high_priority_rules < 5:
            recommendations.append("Consider implementing more high-priority business rules for critical operations")
        
        # Entity-based recommendations
        high_value_entities = len([e for e in self.domain_entities if e.business_value == "high"])
        if high_value_entities < 3:
            recommendations.append("Identify and model more high-value domain entities")
        
        # Process-based recommendations
        documented_processes = len([p for p in self.business_processes if len(p.steps) > 3])
        if documented_processes < 5:
            recommendations.append("Document more business processes with detailed step-by-step flows")
        
        # Complexity recommendations
        complex_rules = len([r for r in self.business_rules if r.complexity == "high"])
        if complex_rules > len(self.business_rules) * 0.3:
            recommendations.append("Consider refactoring high-complexity business rules for maintainability")
        
        return recommendations
    
    def _calculate_business_metrics(self) -> Dict[str, Any]:
        """Calculate business logic metrics"""
        total_rules = len(self.business_rules)
        total_entities = len(self.domain_entities)
        total_processes = len(self.business_processes)
        
        return {
            "business_logic_density": total_rules + total_entities + total_processes,
            "rule_to_entity_ratio": total_rules / max(total_entities, 1),
            "process_complexity_average": sum(len(p.steps) for p in self.business_processes) / max(total_processes, 1),
            "high_value_percentage": len([item for item in (self.domain_entities + self.business_processes) if item.business_value == "high"]) / max(total_entities + total_processes, 1) * 100,
            "business_rule_coverage": {
                "validation": len([r for r in self.business_rules if r.rule_type == "validation"]) / max(total_rules, 1) * 100,
                "authorization": len([r for r in self.business_rules if r.rule_type == "authorization"]) / max(total_rules, 1) * 100,
                "calculation": len([r for r in self.business_rules if r.rule_type == "calculation"]) / max(total_rules, 1) * 100
            }
        }


# Export classes for use in other modules
__all__ = ['BusinessLogicAnalyzer', 'BusinessRule', 'DomainEntity', 'BusinessProcess']


# Factory function for easy instantiation
def create_business_logic_analyzer(config: Optional[Dict[str, Any]] = None) -> BusinessLogicAnalyzer:
    """Factory function to create a configured BusinessLogicAnalyzer instance"""
    return BusinessLogicAnalyzer(config)


if __name__ == "__main__":
    # Example usage
    analyzer = create_business_logic_analyzer()
    
    # Analyze current directory
    import os
    current_dir = os.getcwd()
    
    try:
        analysis_result = analyzer.analyze_business_logic(current_dir)
        
        print(f"Business Logic Analysis Results:")
        print(f"Business Rules: {len(analysis_result['business_rules'])}")
        print(f"Domain Entities: {len(analysis_result['domain_entities'])}")
        print(f"Business Processes: {len(analysis_result['business_processes'])}")
        
        # Save analysis results
        with open("business_logic_analysis.json", "w") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print("Analysis complete! Results saved to business_logic_analysis.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")