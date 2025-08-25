"""
Domain Model Analyzer

This module analyzes domain models, entities, value objects, and domain-driven
design patterns in codebases. It identifies domain boundaries, aggregates,
and domain logic distribution.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

from ..base import BaseAnalyzer


class DomainObjectType(Enum):
    """Types of domain objects"""
    ENTITY = "entity"
    VALUE_OBJECT = "value_object"
    AGGREGATE_ROOT = "aggregate_root"
    DOMAIN_SERVICE = "domain_service"
    REPOSITORY = "repository"
    FACTORY = "factory"
    SPECIFICATION = "specification"
    DOMAIN_EVENT = "domain_event"
    POLICY = "policy"
    ENUM = "enum"


class RelationshipType(Enum):
    """Types of relationships between domain objects"""
    COMPOSITION = "composition"
    AGGREGATION = "aggregation"
    ASSOCIATION = "association"
    INHERITANCE = "inheritance"
    DEPENDENCY = "dependency"
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"


@dataclass
class DomainObject:
    """Represents a domain object"""
    name: str
    object_type: DomainObjectType
    properties: List[str]
    methods: List[str]
    invariants: List[str]
    business_rules: List[str]
    file_path: str
    line_number: int
    complexity_score: int
    is_anemic: bool
    has_identity: bool
    is_immutable: bool


@dataclass
class DomainRelationship:
    """Represents a relationship between domain objects"""
    source: str
    target: str
    relationship_type: RelationshipType
    description: str
    cardinality: str
    file_path: str
    line_number: int


@dataclass
class BoundedContext:
    """Represents a bounded context"""
    name: str
    domain_objects: List[str]
    responsibilities: List[str]
    interfaces: List[str]
    dependencies: List[str]
    file_paths: List[str]
    cohesion_score: float
    coupling_score: float


@dataclass
class DomainIssue:
    """Represents an issue in domain design"""
    object_name: str
    issue_type: str
    severity: str
    description: str
    impact: str
    recommendation: str
    file_path: str
    line_number: int


class DomainModelAnalyzer(BaseAnalyzer):
    """Analyzes domain models and DDD patterns"""
    
    def __init__(self):
        super().__init__()
        self.domain_objects: List[DomainObject] = []
        self.relationships: List[DomainRelationship] = []
        self.bounded_contexts: List[BoundedContext] = []
        self.issues: List[DomainIssue] = []
        
        # Domain object patterns
        self.entity_patterns = [
            r"class.*Entity", r"class.*Model", r"class.*Aggregate",
            r"class.*Root", r"Entity$", r"Model$"
        ]
        
        self.value_object_patterns = [
            r"class.*Value", r"class.*VO", r"ValueObject$",
            r"@dataclass.*frozen=True", r"NamedTuple"
        ]
        
        self.service_patterns = [
            r"class.*Service", r"class.*DomainService",
            r"Service$", r"DomainService$"
        ]
        
        self.repository_patterns = [
            r"class.*Repository", r"class.*Repo",
            r"Repository$", r"Repo$"
        ]
        
        # Business domain keywords
        self.business_keywords = {
            "ecommerce": [
                "order", "product", "customer", "payment", "cart",
                "inventory", "shipping", "discount", "coupon"
            ],
            "finance": [
                "account", "transaction", "balance", "transfer",
                "loan", "interest", "payment", "credit", "debit"
            ],
            "hr": [
                "employee", "department", "salary", "position",
                "performance", "leave", "attendance", "payroll"
            ],
            "healthcare": [
                "patient", "doctor", "appointment", "diagnosis",
                "treatment", "prescription", "medical", "record"
            ],
            "education": [
                "student", "course", "grade", "assignment",
                "teacher", "class", "enrollment", "curriculum"
            ]
        }
        
        # Identity patterns
        self.identity_patterns = [
            r"id\s*:", r"\.id", r"primary_key", r"pk",
            r"uuid", r"guid", r"identifier"
        ]
        
        # Invariant patterns
        self.invariant_patterns = [
            r"assert\s+", r"if.*raise", r"validate",
            r"check", r"ensure", r"require"
        ]
        
        # Anti-patterns
        self.domain_antipatterns = {
            "anemic_model": {
                "description": "Domain object with only getters/setters",
                "severity": "high"
            },
            "god_object": {
                "description": "Domain object with too many responsibilities",
                "severity": "high"
            },
            "primitive_obsession": {
                "description": "Using primitives instead of value objects",
                "severity": "medium"
            },
            "feature_envy": {
                "description": "Object accessing data from other objects",
                "severity": "medium"
            },
            "data_class": {
                "description": "Class used only for data storage",
                "severity": "low"
            }
        }
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze domain models and DDD patterns"""
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        # After analyzing all files, identify bounded contexts
        self._identify_bounded_contexts()
        
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for domain patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Analyze classes for domain objects
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    domain_obj = self._analyze_domain_class(node, content, str(file_path))
                    if domain_obj:
                        self.domain_objects.append(domain_obj)
                        
                        # Analyze relationships
                        relationships = self._analyze_relationships(node, content, str(file_path))
                        self.relationships.extend(relationships)
                        
                        # Check for domain issues
                        issues = self._check_domain_issues(domain_obj, node, content)
                        self.issues.extend(issues)
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
    
    def _analyze_domain_class(self, node: ast.ClassDef, content: str,
                             file_path: str) -> Optional[DomainObject]:
        """Analyze a class to determine if it's a domain object"""
        class_name = node.name
        
        # Determine object type
        object_type = self._classify_domain_object(node, content)
        if not object_type:
            return None
        
        # Extract properties and methods
        properties = self._extract_properties(node)
        methods = self._extract_methods(node)
        
        # Analyze domain characteristics
        invariants = self._extract_invariants(node)
        business_rules = self._extract_business_rules(node, content)
        
        # Calculate characteristics
        complexity_score = self._calculate_domain_complexity(node, methods, properties)
        is_anemic = self._is_anemic_model(node, methods, properties)
        has_identity = self._has_identity(node, properties)
        is_immutable = self._is_immutable(node, content)
        
        return DomainObject(
            name=class_name,
            object_type=object_type,
            properties=properties,
            methods=methods,
            invariants=invariants,
            business_rules=business_rules,
            file_path=file_path,
            line_number=node.lineno,
            complexity_score=complexity_score,
            is_anemic=is_anemic,
            has_identity=has_identity,
            is_immutable=is_immutable
        )
    
    def _classify_domain_object(self, node: ast.ClassDef, content: str) -> Optional[DomainObjectType]:
        """Classify the type of domain object"""
        class_name = node.name
        class_content = ast.unparse(node)
        
        # Check explicit patterns
        if any(re.search(pattern, class_name, re.IGNORECASE) 
              for pattern in self.entity_patterns):
            return DomainObjectType.ENTITY
        
        if any(re.search(pattern, class_content, re.IGNORECASE) 
              for pattern in self.value_object_patterns):
            return DomainObjectType.VALUE_OBJECT
        
        if any(re.search(pattern, class_name, re.IGNORECASE) 
              for pattern in self.service_patterns):
            return DomainObjectType.DOMAIN_SERVICE
        
        if any(re.search(pattern, class_name, re.IGNORECASE) 
              for pattern in self.repository_patterns):
            return DomainObjectType.REPOSITORY
        
        # Check for enum
        if any(isinstance(base, ast.Name) and base.id == "Enum" 
              for base in node.bases):
            return DomainObjectType.ENUM
        
        # Check for domain events
        if "event" in class_name.lower() or "Event" in class_name:
            return DomainObjectType.DOMAIN_EVENT
        
        # Check for factory pattern
        if "factory" in class_name.lower() or "Factory" in class_name:
            return DomainObjectType.FACTORY
        
        # Check for specification pattern
        if "spec" in class_name.lower() or "Specification" in class_name:
            return DomainObjectType.SPECIFICATION
        
        # Heuristic classification
        properties = self._extract_properties(node)
        methods = self._extract_methods(node)
        
        # If has identity and behavior, likely an entity
        if self._has_identity(node, properties) and len(methods) > 2:
            return DomainObjectType.ENTITY
        
        # If immutable with few methods, likely value object
        if self._is_immutable(node, content) and len(methods) <= 3:
            return DomainObjectType.VALUE_OBJECT
        
        # If mostly methods with business logic, likely service
        if len(methods) > len(properties) * 2:
            return DomainObjectType.DOMAIN_SERVICE
        
        # Check if contains business domain keywords
        if self._contains_business_logic(node, content):
            return DomainObjectType.ENTITY
        
        return None
    
    def _extract_properties(self, node: ast.ClassDef) -> List[str]:
        """Extract properties from class"""
        properties = []
        
        for item in node.body:
            # Instance variables from __init__
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(target.value, ast.Name) and target.value.id == "self":
                                    properties.append(target.attr)
            
            # Class attributes
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        properties.append(target.id)
            
            # Properties with @property decorator
            elif isinstance(item, ast.FunctionDef):
                if any(isinstance(d, ast.Name) and d.id == "property" 
                      for d in item.decorator_list):
                    properties.append(item.name)
        
        return properties
    
    def _extract_methods(self, node: ast.ClassDef) -> List[str]:
        """Extract methods from class"""
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith('_'):  # Skip private methods
                    methods.append(item.name)
        
        return methods
    
    def _extract_invariants(self, node: ast.ClassDef) -> List[str]:
        """Extract invariants (assertions, validations)"""
        invariants = []
        
        for item in ast.walk(node):
            if isinstance(item, ast.Assert):
                invariants.append(ast.unparse(item.test))
            elif isinstance(item, ast.If):
                # Check for validation patterns
                test_str = ast.unparse(item.test)
                for stmt in item.body:
                    if isinstance(stmt, ast.Raise):
                        invariants.append(f"if {test_str}: raise")
        
        return invariants
    
    def _extract_business_rules(self, node: ast.ClassDef, content: str) -> List[str]:
        """Extract business rules from class"""
        rules = []
        
        # Look for methods that contain business logic
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_content = ast.unparse(item)
                
                # Check for business logic patterns
                business_patterns = [
                    r"calculate", r"validate", r"process", r"apply",
                    r"check", r"ensure", r"verify", r"confirm"
                ]
                
                for pattern in business_patterns:
                    if re.search(pattern, method_content, re.IGNORECASE):
                        rules.append(f"{item.name}: {pattern} business logic")
                        break
        
        return rules
    
    def _calculate_domain_complexity(self, node: ast.ClassDef, methods: List[str],
                                   properties: List[str]) -> int:
        """Calculate domain complexity score"""
        complexity = 0
        
        # Base complexity from size
        complexity += len(methods) * 2
        complexity += len(properties)
        
        # Complexity from control flow
        for item in ast.walk(node):
            if isinstance(item, (ast.If, ast.For, ast.While)):
                complexity += 2
            elif isinstance(item, ast.Try):
                complexity += 3
        
        # Complexity from relationships
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                call_str = ast.unparse(item)
                if "." in call_str:  # Method calls on other objects
                    complexity += 1
        
        return complexity
    
    def _is_anemic_model(self, node: ast.ClassDef, methods: List[str],
                        properties: List[str]) -> bool:
        """Check if model is anemic (only getters/setters)"""
        if not methods:
            return True
        
        # Count getters and setters
        getters_setters = 0
        business_methods = 0
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name.lower()
                
                # Skip special methods
                if method_name.startswith('__'):
                    continue
                
                # Check for getter/setter pattern
                if (method_name.startswith('get_') or 
                    method_name.startswith('set_') or
                    any(isinstance(d, ast.Name) and d.id == "property" 
                        for d in item.decorator_list)):
                    getters_setters += 1
                else:
                    # Check if method contains business logic
                    method_content = ast.unparse(item)
                    if any(keyword in method_content.lower() 
                          for keyword in ["calculate", "validate", "process", "apply"]):
                        business_methods += 1
        
        # Anemic if mostly getters/setters
        total_methods = getters_setters + business_methods
        return total_methods > 0 and (getters_setters / total_methods) > 0.7
    
    def _has_identity(self, node: ast.ClassDef, properties: List[str]) -> bool:
        """Check if object has identity"""
        # Check for explicit identity properties
        for prop in properties:
            if any(pattern in prop.lower() for pattern in ["id", "uuid", "guid"]):
                return True
        
        # Check for identity patterns in code
        node_str = ast.unparse(node)
        return any(re.search(pattern, node_str, re.IGNORECASE) 
                  for pattern in self.identity_patterns)
    
    def _is_immutable(self, node: ast.ClassDef, content: str) -> bool:
        """Check if object is immutable"""
        # Check for frozen dataclass
        if "@dataclass" in content and "frozen=True" in content:
            return True
        
        # Check for NamedTuple
        if any(isinstance(base, ast.Name) and base.id == "NamedTuple" 
              for base in node.bases):
            return True
        
        # Check for lack of setters
        has_setters = False
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name.startswith('set_') or "setter" in ast.unparse(item):
                    has_setters = True
                    break
        
        return not has_setters
    
    def _contains_business_logic(self, node: ast.ClassDef, content: str) -> bool:
        """Check if class contains business logic"""
        # Check for business domain keywords
        content_lower = content.lower()
        for domain, keywords in self.business_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return True
        
        # Check for business method patterns
        business_method_patterns = [
            r"calculate", r"process", r"validate", r"apply",
            r"execute", r"perform", r"handle", r"manage"
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) 
                  for pattern in business_method_patterns)
    
    def _analyze_relationships(self, node: ast.ClassDef, content: str,
                             file_path: str) -> List[DomainRelationship]:
        """Analyze relationships between domain objects"""
        relationships = []
        class_name = node.name
        
        # Analyze inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                relationships.append(DomainRelationship(
                    source=class_name,
                    target=base.id,
                    relationship_type=RelationshipType.INHERITANCE,
                    description=f"{class_name} inherits from {base.id}",
                    cardinality="1:1",
                    file_path=file_path,
                    line_number=node.lineno
                ))
        
        # Analyze composition/aggregation from properties
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        # Look for object assignments
                        value_str = ast.unparse(stmt.value)
                        if "(" in value_str:  # Constructor call
                            target_class = value_str.split("(")[0].strip()
                            if target_class and target_class[0].isupper():
                                relationships.append(DomainRelationship(
                                    source=class_name,
                                    target=target_class,
                                    relationship_type=RelationshipType.COMPOSITION,
                                    description=f"{class_name} composes {target_class}",
                                    cardinality="1:1",
                                    file_path=file_path,
                                    line_number=stmt.lineno
                                ))
        
        # Analyze associations from method calls
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                call_str = ast.unparse(item)
                if "." in call_str:
                    parts = call_str.split(".")
                    if len(parts) >= 2:
                        target = parts[0]
                        if target != "self" and target[0].isupper():
                            relationships.append(DomainRelationship(
                                source=class_name,
                                target=target,
                                relationship_type=RelationshipType.ASSOCIATION,
                                description=f"{class_name} uses {target}",
                                cardinality="1:*",
                                file_path=file_path,
                                line_number=getattr(item, 'lineno', 0)
                            ))
        
        return relationships
    
    def _check_domain_issues(self, domain_obj: DomainObject, node: ast.ClassDef,
                           content: str) -> List[DomainIssue]:
        """Check for domain design issues"""
        issues = []
        
        # Check for anemic model
        if domain_obj.is_anemic:
            issues.append(DomainIssue(
                object_name=domain_obj.name,
                issue_type="anemic_model",
                severity="high",
                description="Domain object has only getters/setters",
                impact="Business logic scattered across services",
                recommendation="Move business logic into domain object",
                file_path=domain_obj.file_path,
                line_number=domain_obj.line_number
            ))
        
        # Check for god object
        if domain_obj.complexity_score > 30:
            issues.append(DomainIssue(
                object_name=domain_obj.name,
                issue_type="god_object",
                severity="high",
                description=f"Domain object has high complexity ({domain_obj.complexity_score})",
                impact="Difficult to understand and maintain",
                recommendation="Split into smaller, focused objects",
                file_path=domain_obj.file_path,
                line_number=domain_obj.line_number
            ))
        
        # Check for missing identity
        if (domain_obj.object_type == DomainObjectType.ENTITY and 
            not domain_obj.has_identity):
            issues.append(DomainIssue(
                object_name=domain_obj.name,
                issue_type="missing_identity",
                severity="medium",
                description="Entity lacks clear identity",
                impact="Cannot distinguish between instances",
                recommendation="Add unique identifier property",
                file_path=domain_obj.file_path,
                line_number=domain_obj.line_number
            ))
        
        # Check for missing invariants
        if (domain_obj.object_type in [DomainObjectType.ENTITY, DomainObjectType.VALUE_OBJECT] and
            not domain_obj.invariants):
            issues.append(DomainIssue(
                object_name=domain_obj.name,
                issue_type="missing_invariants",
                severity="medium",
                description="Domain object lacks invariants",
                impact="Invalid states may be allowed",
                recommendation="Add validation and assertions",
                file_path=domain_obj.file_path,
                line_number=domain_obj.line_number
            ))
        
        # Check for primitive obsession
        primitive_properties = []
        for prop in domain_obj.properties:
            # Heuristic: check for primitive-like names
            if any(primitive in prop.lower() 
                  for primitive in ["str", "int", "float", "bool", "date"]):
                primitive_properties.append(prop)
        
        if len(primitive_properties) > len(domain_obj.properties) * 0.7:
            issues.append(DomainIssue(
                object_name=domain_obj.name,
                issue_type="primitive_obsession",
                severity="medium",
                description="Excessive use of primitive types",
                impact="Loss of type safety and domain semantics",
                recommendation="Create value objects for domain concepts",
                file_path=domain_obj.file_path,
                line_number=domain_obj.line_number
            ))
        
        return issues
    
    def _identify_bounded_contexts(self) -> None:
        """Identify bounded contexts from analyzed domain objects"""
        # Group objects by file path similarity and business domain
        context_groups = {}
        
        for obj in self.domain_objects:
            # Use directory structure as initial grouping
            path_parts = Path(obj.file_path).parts
            if len(path_parts) >= 2:
                context_key = path_parts[-2]  # Parent directory
            else:
                context_key = os.getenv('KEY')
            
            if context_key not in context_groups:
                context_groups[context_key] = []
            context_groups[context_key].append(obj)
        
        # Create bounded contexts
        for context_name, objects in context_groups.items():
            if len(objects) >= 2:  # Only create context if multiple objects
                bounded_context = BoundedContext(
                    name=context_name,
                    domain_objects=[obj.name for obj in objects],
                    responsibilities=self._identify_responsibilities(objects),
                    interfaces=self._identify_interfaces(objects),
                    dependencies=self._identify_dependencies(objects),
                    file_paths=list(set(obj.file_path for obj in objects)),
                    cohesion_score=self._calculate_cohesion(objects),
                    coupling_score=self._calculate_coupling(objects)
                )
                self.bounded_contexts.append(bounded_context)
    
    def _identify_responsibilities(self, objects: List[DomainObject]) -> List[str]:
        """Identify responsibilities of a bounded context"""
        responsibilities = set()
        
        for obj in objects:
            # Extract responsibilities from business rules and methods
            for rule in obj.business_rules:
                responsibilities.add(rule.split(":")[0])
            
            for method in obj.methods:
                if any(keyword in method.lower() 
                      for keyword in ["process", "handle", "manage", "calculate"]):
                    responsibilities.add(f"{obj.name}.{method}")
        
        return list(responsibilities)
    
    def _identify_interfaces(self, objects: List[DomainObject]) -> List[str]:
        """Identify interfaces of a bounded context"""
        interfaces = []
        
        for obj in objects:
            if obj.object_type in [DomainObjectType.DOMAIN_SERVICE, 
                                  DomainObjectType.REPOSITORY]:
                # Public methods are interfaces
                interfaces.extend([f"{obj.name}.{method}" for method in obj.methods])
        
        return interfaces
    
    def _identify_dependencies(self, objects: List[DomainObject]) -> List[str]:
        """Identify dependencies of a bounded context"""
        dependencies = set()
        object_names = {obj.name for obj in objects}
        
        # Find relationships pointing outside the context
        for rel in self.relationships:
            if rel.source in object_names and rel.target not in object_names:
                dependencies.add(rel.target)
        
        return list(dependencies)
    
    def _calculate_cohesion(self, objects: List[DomainObject]) -> float:
        """Calculate cohesion score for bounded context"""
        if len(objects) <= 1:
            return 1.0
        
        # Count internal relationships
        object_names = {obj.name for obj in objects}
        internal_relationships = 0
        total_relationships = 0
        
        for rel in self.relationships:
            if rel.source in object_names:
                total_relationships += 1
                if rel.target in object_names:
                    internal_relationships += 1
        
        if total_relationships == 0:
            return 0.0
        
        return internal_relationships / total_relationships
    
    def _calculate_coupling(self, objects: List[DomainObject]) -> float:
        """Calculate coupling score for bounded context"""
        if len(objects) <= 1:
            return 0.0
        
        object_names = {obj.name for obj in objects}
        external_dependencies = 0
        
        for rel in self.relationships:
            if rel.source in object_names and rel.target not in object_names:
                external_dependencies += 1
        
        # Normalize by number of objects
        return external_dependencies / len(objects)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive domain model analysis report"""
        # Calculate statistics
        total_objects = len(self.domain_objects)
        total_relationships = len(self.relationships)
        total_contexts = len(self.bounded_contexts)
        total_issues = len(self.issues)
        
        # Group by object type
        objects_by_type = {}
        for obj in self.domain_objects:
            obj_type = obj.object_type.value
            objects_by_type[obj_type] = objects_by_type.get(obj_type, 0) + 1
        
        # Calculate complexity statistics
        if self.domain_objects:
            avg_complexity = sum(obj.complexity_score for obj in self.domain_objects) / total_objects
            max_complexity = max(obj.complexity_score for obj in self.domain_objects)
        else:
            avg_complexity = max_complexity = 0
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in self.issues:
            severity = issue.severity
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        return {
            "summary": {
                "total_domain_objects": total_objects,
                "total_relationships": total_relationships,
                "total_bounded_contexts": total_contexts,
                "total_issues": total_issues,
                "average_complexity": round(avg_complexity, 2),
                "max_complexity": max_complexity,
                "anemic_models": sum(1 for obj in self.domain_objects if obj.is_anemic),
                "entities_with_identity": sum(1 for obj in self.domain_objects 
                                            if obj.object_type == DomainObjectType.ENTITY and obj.has_identity),
                "immutable_objects": sum(1 for obj in self.domain_objects if obj.is_immutable),
            },
            "domain_objects": [
                {
                    "name": obj.name,
                    "type": obj.object_type.value,
                    "file": obj.file_path,
                    "line": obj.line_number,
                    "properties_count": len(obj.properties),
                    "methods_count": len(obj.methods),
                    "invariants_count": len(obj.invariants),
                    "business_rules_count": len(obj.business_rules),
                    "complexity_score": obj.complexity_score,
                    "is_anemic": obj.is_anemic,
                    "has_identity": obj.has_identity,
                    "is_immutable": obj.is_immutable,
                    "properties": obj.properties,
                    "methods": obj.methods,
                    "invariants": obj.invariants,
                    "business_rules": obj.business_rules,
                }
                for obj in self.domain_objects
            ],
            "object_types": objects_by_type,
            "relationships": [
                {
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.relationship_type.value,
                    "description": rel.description,
                    "cardinality": rel.cardinality,
                    "file": rel.file_path,
                    "line": rel.line_number,
                }
                for rel in self.relationships
            ],
            "bounded_contexts": [
                {
                    "name": ctx.name,
                    "domain_objects": ctx.domain_objects,
                    "responsibilities": ctx.responsibilities,
                    "interfaces": ctx.interfaces,
                    "dependencies": ctx.dependencies,
                    "file_paths": ctx.file_paths,
                    "cohesion_score": round(ctx.cohesion_score, 3),
                    "coupling_score": round(ctx.coupling_score, 3),
                }
                for ctx in self.bounded_contexts
            ],
            "issues": [
                {
                    "object": issue.object_name,
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description,
                    "impact": issue.impact,
                    "recommendation": issue.recommendation,
                    "file": issue.file_path,
                    "line": issue.line_number,
                }
                for issue in sorted(self.issues, 
                                  key=lambda x: {"critical": 0, "high": 1, 
                                               "medium": 2, "low": 3}[x.severity])
            ],
            "issues_by_severity": issues_by_severity,
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate domain model improvement recommendations"""
        recommendations = []
        
        # Check for anemic models
        anemic_models = [obj for obj in self.domain_objects if obj.is_anemic]
        if anemic_models:
            recommendations.append({
                "category": "Domain Logic",
                "priority": "high",
                "recommendation": "Eliminate anemic domain models",
                "impact": "Improve encapsulation and reduce coupling",
                "affected_objects": [obj.name for obj in anemic_models]
            })
        
        # Check for missing identities
        entities_without_identity = [
            obj for obj in self.domain_objects 
            if obj.object_type == DomainObjectType.ENTITY and not obj.has_identity
        ]
        if entities_without_identity:
            recommendations.append({
                "category": "Entity Design",
                "priority": "medium",
                "recommendation": "Add identity to entities",
                "impact": "Improve entity distinction and persistence",
                "affected_objects": [obj.name for obj in entities_without_identity]
            })
        
        # Check for high coupling contexts
        high_coupling_contexts = [ctx for ctx in self.bounded_contexts if ctx.coupling_score > 2.0]
        if high_coupling_contexts:
            recommendations.append({
                "category": "Architecture",
                "priority": "medium",
                "recommendation": "Reduce coupling between bounded contexts",
                "impact": "Improve modularity and maintainability",
                "affected_contexts": [ctx.name for ctx in high_coupling_contexts]
            })
        
        # Check for low cohesion contexts
        low_cohesion_contexts = [ctx for ctx in self.bounded_contexts if ctx.cohesion_score < 0.5]
        if low_cohesion_contexts:
            recommendations.append({
                "category": "Architecture",
                "priority": "medium",
                "recommendation": "Improve cohesion within bounded contexts",
                "impact": "Better organize related domain objects",
                "affected_contexts": [ctx.name for ctx in low_cohesion_contexts]
            })
        
        return recommendations