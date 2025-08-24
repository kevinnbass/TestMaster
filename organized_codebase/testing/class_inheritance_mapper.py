#!/usr/bin/env python3
"""
Agent C - Class Inheritance Hierarchy Mapper
Hours 7-9: Comprehensive class inheritance and interface analysis.

Features:
- Complete class hierarchy mapping
- Interface implementation tracking
- Mixin usage pattern analysis
- Abstract base class identification
- Multiple inheritance analysis
"""

import ast
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClassDefinition:
    """Information about a class definition."""
    name: str
    module: str
    file_path: str
    line_number: int
    base_classes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False
    is_dataclass: bool = False
    is_enum: bool = False
    is_exception: bool = False
    metaclass: Optional[str] = None
    mixin_patterns: List[str] = field(default_factory=list)


@dataclass
class InheritanceRelationship:
    """Represents an inheritance relationship between classes."""
    child_class: str
    parent_class: str
    inheritance_type: str  # 'direct', 'interface', 'mixin', 'multiple'
    resolution_order: int = 0


@dataclass
class InheritanceHierarchy:
    """Complete inheritance hierarchy for a class tree."""
    root_class: str
    descendant_classes: List[str] = field(default_factory=list)
    depth: int = 0
    total_methods: int = 0
    overridden_methods: Set[str] = field(default_factory=set)
    abstract_methods: Set[str] = field(default_factory=set)


@dataclass
class InterfaceAnalysis:
    """Analysis of interface-like patterns in the codebase."""
    interface_classes: List[str] = field(default_factory=list)
    implementations: Dict[str, List[str]] = field(default_factory=dict)
    protocol_classes: List[str] = field(default_factory=list)
    abc_classes: List[str] = field(default_factory=list)


class ClassInheritanceMapper:
    """
    Map complete class inheritance hierarchies and interface patterns.
    """
    
    def __init__(self, root_path: Path = Path(".")):
        self.root_path = root_path.resolve()
        self.classes: Dict[str, ClassDefinition] = {}
        self.inheritance_relationships: List[InheritanceRelationship] = []
        self.hierarchies: Dict[str, InheritanceHierarchy] = {}
        self.interface_analysis = InterfaceAnalysis()
        self.scan_timestamp = datetime.now()
        
        # Analysis tracking
        self.multiple_inheritance_classes: Set[str] = set()
        self.mixin_classes: Set[str] = set()
        self.diamond_problems: List[List[str]] = []
        self.abstract_hierarchies: Dict[str, List[str]] = {}
        
    def analyze_inheritance_patterns(self) -> Dict[str, Any]:
        """
        Analyze all inheritance patterns in the codebase.
        """
        start_time = time.time()
        logger.info(f"Starting class inheritance analysis for {self.root_path}")
        
        # Phase 1: Discover all classes
        python_files = self._discover_python_files()
        logger.info(f"Analyzing classes in {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                self._analyze_file_classes(file_path)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Phase 2: Build inheritance relationships
        self._build_inheritance_relationships()
        
        # Phase 3: Analyze inheritance patterns
        self._analyze_multiple_inheritance()
        self._detect_mixin_patterns()
        self._identify_diamond_problems()
        self._analyze_abstract_patterns()
        self._analyze_interface_patterns()
        
        # Phase 4: Build complete hierarchies
        self._build_inheritance_hierarchies()
        
        duration = time.time() - start_time
        logger.info(f"Inheritance analysis completed in {duration:.2f} seconds")
        
        return self._generate_comprehensive_report()
    
    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files in the codebase."""
        python_files = []
        
        for py_file in self.root_path.rglob("*.py"):
            # Skip common non-source directories
            if any(exclude in str(py_file) for exclude in [
                '__pycache__', '.git', '.venv', 'venv', 'env',
                'node_modules', '.pytest_cache', '.coverage'
            ]):
                continue
                
            python_files.append(py_file)
        
        return sorted(python_files)
    
    def _analyze_file_classes(self, file_path: Path):
        """Analyze all class definitions in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
                
            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping")
                return
            
            module_name = self._calculate_module_name(file_path)
            
            # Extract class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_def = self._extract_class_definition(node, module_name, file_path)
                    if class_def:
                        full_name = f"{module_name}.{class_def.name}"
                        self.classes[full_name] = class_def
                        
        except Exception as e:
            logger.error(f"Error analyzing classes in {file_path}: {e}")
    
    def _extract_class_definition(self, node: ast.ClassDef, module_name: str, 
                                file_path: Path) -> Optional[ClassDefinition]:
        """Extract class definition information from AST node."""
        try:
            class_def = ClassDefinition(
                name=node.name,
                module=module_name,
                file_path=str(file_path),
                line_number=node.lineno,
                docstring=ast.get_docstring(node)
            )
            
            # Extract base classes
            for base in node.bases:
                base_name = self._extract_base_class_name(base)
                if base_name:
                    class_def.base_classes.append(base_name)
            
            # Extract methods and properties
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    class_def.methods.append(item.name)
                    
                    # Check for property decorators
                    for decorator in item.decorator_list:
                        decorator_name = self._extract_decorator_name(decorator)
                        if decorator_name in ['property', 'cached_property']:
                            class_def.properties.append(item.name)
                            
                elif isinstance(item, ast.AsyncFunctionDef):
                    class_def.methods.append(f"async {item.name}")
                    
                elif isinstance(item, ast.Assign):
                    # Class variables
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            class_def.class_variables.append(target.id)
            
            # Extract decorators
            for decorator in node.decorator_list:
                decorator_name = self._extract_decorator_name(decorator)
                class_def.decorators.append(decorator_name)
                
                # Check for special decorator types
                if decorator_name == 'dataclass':
                    class_def.is_dataclass = True
                elif decorator_name in ['abstractmethod', 'abc.abstractmethod']:
                    class_def.is_abstract = True
            
            # Check for metaclass
            for keyword in getattr(node, 'keywords', []):
                if keyword.arg == 'metaclass':
                    class_def.metaclass = self._extract_base_class_name(keyword.value)
            
            # Analyze class patterns
            self._analyze_class_patterns(class_def)
            
            return class_def
            
        except Exception as e:
            logger.error(f"Error extracting class definition for {node.name}: {e}")
            return None
    
    def _extract_base_class_name(self, base: ast.AST) -> Optional[str]:
        """Extract base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._extract_base_class_name(base.value)}.{base.attr}"
        elif isinstance(base, ast.Subscript):
            # Handle generic types like List[str]
            return self._extract_base_class_name(base.value)
        else:
            return None
    
    def _extract_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._extract_decorator_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._extract_decorator_name(decorator.func)
        else:
            return "unknown"
    
    def _analyze_class_patterns(self, class_def: ClassDefinition):
        """Analyze patterns in the class definition."""
        # Check if it's an exception class
        if any(base in ['Exception', 'BaseException', 'ValueError', 'TypeError'] 
               for base in class_def.base_classes):
            class_def.is_exception = True
        
        # Check if it's an enum
        if any(base in ['Enum', 'IntEnum', 'Flag', 'IntFlag'] 
               for base in class_def.base_classes):
            class_def.is_enum = True
        
        # Check for mixin patterns
        if class_def.name.endswith('Mixin') or 'mixin' in class_def.name.lower():
            class_def.mixin_patterns.append('naming_convention')
        
        # Check for abstract methods
        for method in class_def.methods:
            if 'abstract' in method.lower():
                class_def.is_abstract = True
    
    def _calculate_module_name(self, file_path: Path) -> str:
        """Calculate the module name from file path."""
        relative_path = file_path.relative_to(self.root_path)
        parts = list(relative_path.parts)
        
        # Remove .py extension
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        # Handle __init__.py files
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts) if parts else '__main__'
    
    def _build_inheritance_relationships(self):
        """Build inheritance relationships between classes."""
        for class_name, class_def in self.classes.items():
            for i, base_class in enumerate(class_def.base_classes):
                inheritance_type = self._determine_inheritance_type(class_def, base_class, i)
                
                relationship = InheritanceRelationship(
                    child_class=class_name,
                    parent_class=base_class,
                    inheritance_type=inheritance_type,
                    resolution_order=i
                )
                
                self.inheritance_relationships.append(relationship)
    
    def _determine_inheritance_type(self, class_def: ClassDefinition, 
                                  base_class: str, order: int) -> str:
        """Determine the type of inheritance relationship."""
        if len(class_def.base_classes) > 1:
            if base_class.endswith('Mixin') or 'mixin' in base_class.lower():
                return 'mixin'
            else:
                return 'multiple'
        elif base_class in ['ABC', 'abc.ABC'] or base_class.startswith('Abstract'):
            return 'interface'
        else:
            return 'direct'
    
    def _analyze_multiple_inheritance(self):
        """Analyze multiple inheritance patterns."""
        for class_name, class_def in self.classes.items():
            if len(class_def.base_classes) > 1:
                self.multiple_inheritance_classes.add(class_name)
    
    def _detect_mixin_patterns(self):
        """Detect mixin usage patterns."""
        for class_name, class_def in self.classes.items():
            # Check for mixin naming conventions
            if class_def.name.endswith('Mixin'):
                self.mixin_classes.add(class_name)
            
            # Check for mixin usage in base classes
            for base_class in class_def.base_classes:
                if base_class.endswith('Mixin') or 'mixin' in base_class.lower():
                    if class_name not in class_def.mixin_patterns:
                        class_def.mixin_patterns.append('uses_mixin')
    
    def _identify_diamond_problems(self):
        """Identify potential diamond inheritance problems."""
        # Build a graph of inheritance relationships
        inheritance_graph = defaultdict(set)
        
        for relationship in self.inheritance_relationships:
            inheritance_graph[relationship.child_class].add(relationship.parent_class)
        
        # Look for diamond patterns (simplified check)
        for class_name in self.multiple_inheritance_classes:
            class_def = self.classes.get(class_name)
            if class_def and len(class_def.base_classes) >= 2:
                # Check if any base classes share common ancestors
                common_ancestors = self._find_common_ancestors(
                    class_def.base_classes, inheritance_graph
                )
                if common_ancestors:
                    diamond_path = [class_name] + class_def.base_classes + list(common_ancestors)
                    self.diamond_problems.append(diamond_path)
    
    def _find_common_ancestors(self, base_classes: List[str], 
                              inheritance_graph: Dict[str, Set[str]]) -> Set[str]:
        """Find common ancestors of multiple base classes."""
        if len(base_classes) < 2:
            return set()
        
        # Get all ancestors for each base class
        ancestor_sets = []
        for base_class in base_classes:
            ancestors = self._get_all_ancestors(base_class, inheritance_graph)
            ancestor_sets.append(ancestors)
        
        # Find intersection of all ancestor sets
        common_ancestors = ancestor_sets[0]
        for ancestor_set in ancestor_sets[1:]:
            common_ancestors = common_ancestors.intersection(ancestor_set)
        
        return common_ancestors
    
    def _get_all_ancestors(self, class_name: str, 
                          inheritance_graph: Dict[str, Set[str]]) -> Set[str]:
        """Get all ancestors of a class using BFS."""
        ancestors = set()
        queue = deque([class_name])
        
        while queue:
            current = queue.popleft()
            for parent in inheritance_graph.get(current, set()):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        
        return ancestors
    
    def _analyze_abstract_patterns(self):
        """Analyze abstract base class patterns."""
        for class_name, class_def in self.classes.items():
            if class_def.is_abstract or any(base.startswith('Abstract') 
                                          for base in class_def.base_classes):
                # Find all concrete implementations
                implementations = []
                for other_name, other_def in self.classes.items():
                    if class_name in other_def.base_classes:
                        implementations.append(other_name)
                
                self.abstract_hierarchies[class_name] = implementations
    
    def _analyze_interface_patterns(self):
        """Analyze interface-like patterns."""
        # Identify interface-like classes
        for class_name, class_def in self.classes.items():
            # ABC-based interfaces
            if any(base in ['ABC', 'abc.ABC'] for base in class_def.base_classes):
                self.interface_analysis.abc_classes.append(class_name)
            
            # Protocol classes (duck typing interfaces)
            if any(base == 'Protocol' for base in class_def.base_classes):
                self.interface_analysis.protocol_classes.append(class_name)
            
            # Interface naming convention
            if (class_def.name.startswith('I') and class_def.name[1].isupper()) or \
               class_def.name.endswith('Interface'):
                self.interface_analysis.interface_classes.append(class_name)
        
        # Map implementations to interfaces
        for interface_class in (self.interface_analysis.abc_classes + 
                               self.interface_analysis.protocol_classes + 
                               self.interface_analysis.interface_classes):
            implementations = []
            for class_name, class_def in self.classes.items():
                if interface_class in class_def.base_classes:
                    implementations.append(class_name)
            
            if implementations:
                self.interface_analysis.implementations[interface_class] = implementations
    
    def _build_inheritance_hierarchies(self):
        """Build complete inheritance hierarchies."""
        # Find root classes (classes with no parents in our codebase)
        internal_classes = set(self.classes.keys())
        children = set()
        
        for relationship in self.inheritance_relationships:
            if relationship.parent_class in internal_classes:
                children.add(relationship.child_class)
        
        root_classes = internal_classes - children
        
        # Build hierarchy for each root class
        for root_class in root_classes:
            hierarchy = self._build_hierarchy_tree(root_class)
            if hierarchy.descendant_classes:  # Only include non-trivial hierarchies
                self.hierarchies[root_class] = hierarchy
    
    def _build_hierarchy_tree(self, root_class: str) -> InheritanceHierarchy:
        """Build a complete hierarchy tree starting from a root class."""
        hierarchy = InheritanceHierarchy(root_class=root_class)
        
        # Find all descendants using BFS
        queue = deque([(root_class, 0)])
        visited = set()
        
        while queue:
            current_class, depth = queue.popleft()
            
            if current_class in visited:
                continue
            visited.add(current_class)
            
            # Find direct children
            for relationship in self.inheritance_relationships:
                if (relationship.parent_class == current_class and 
                    relationship.child_class not in visited):
                    hierarchy.descendant_classes.append(relationship.child_class)
                    queue.append((relationship.child_class, depth + 1))
                    hierarchy.depth = max(hierarchy.depth, depth + 1)
        
        # Calculate additional metrics
        hierarchy.total_methods = sum(
            len(self.classes[cls].methods) 
            for cls in [root_class] + hierarchy.descendant_classes
            if cls in self.classes
        )
        
        return hierarchy
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive inheritance analysis report."""
        return {
            'scan_metadata': {
                'timestamp': self.scan_timestamp.isoformat(),
                'root_path': str(self.root_path),
                'analysis_type': 'class_inheritance_hierarchy'
            },
            'statistics': {
                'total_classes': len(self.classes),
                'inheritance_relationships': len(self.inheritance_relationships),
                'multiple_inheritance_classes': len(self.multiple_inheritance_classes),
                'mixin_classes': len(self.mixin_classes),
                'abstract_classes': len(self.abstract_hierarchies),
                'diamond_problems': len(self.diamond_problems),
                'root_hierarchies': len(self.hierarchies)
            },
            'classes': {
                name: asdict(class_def) for name, class_def in self.classes.items()
            },
            'inheritance_relationships': [
                asdict(rel) for rel in self.inheritance_relationships
            ],
            'inheritance_hierarchies': {
                name: asdict(hierarchy) for name, hierarchy in self.hierarchies.items()
            },
            'multiple_inheritance_analysis': {
                'classes': list(self.multiple_inheritance_classes),
                'diamond_problems': self.diamond_problems,
                'mixin_classes': list(self.mixin_classes)
            },
            'abstract_analysis': self.abstract_hierarchies,
            'interface_analysis': asdict(self.interface_analysis),
            'patterns_detected': self._generate_pattern_insights()
        }
    
    def _generate_pattern_insights(self) -> Dict[str, Any]:
        """Generate insights about inheritance patterns."""
        insights = {
            'design_patterns': {},
            'anti_patterns': {},
            'recommendations': []
        }
        
        # Design patterns
        if self.interface_analysis.abc_classes:
            insights['design_patterns']['abstract_factory'] = len(self.interface_analysis.abc_classes)
        
        if self.mixin_classes:
            insights['design_patterns']['mixin_usage'] = len(self.mixin_classes)
        
        # Anti-patterns
        if self.diamond_problems:
            insights['anti_patterns']['diamond_inheritance'] = len(self.diamond_problems)
            insights['recommendations'].append("Review diamond inheritance patterns for potential refactoring")
        
        if len(self.multiple_inheritance_classes) > len(self.classes) * 0.1:
            insights['anti_patterns']['excessive_multiple_inheritance'] = True
            insights['recommendations'].append("Consider composition over multiple inheritance")
        
        return insights
    
    def save_report(self, output_path: Path) -> None:
        """Save the inheritance analysis report."""
        report = self._generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Inheritance analysis report saved to {output_path}")


def main():
    """Main entry point for the class inheritance mapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent C - Class Inheritance Hierarchy Mapper")
    parser.add_argument("--root", default=".", help="Root directory to analyze")
    parser.add_argument("--output", default="class_inheritance_hour7.json", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create mapper and run analysis
    mapper = ClassInheritanceMapper(Path(args.root))
    
    print("Agent C - Class Inheritance Hierarchy Analysis (Hours 7-9)")
    print(f"Analyzing: {mapper.root_path}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Analyze inheritance patterns
    report = mapper.analyze_inheritance_patterns()
    
    # Save report
    mapper.save_report(Path(args.output))
    
    # Print summary
    stats = report['statistics']
    print(f"\nClass Inheritance Analysis Results:")
    print(f"   Total Classes: {stats['total_classes']}")
    print(f"   Inheritance Relationships: {stats['inheritance_relationships']}")
    print(f"   Multiple Inheritance: {stats['multiple_inheritance_classes']}")
    print(f"   Mixin Classes: {stats['mixin_classes']}")
    print(f"   Abstract Classes: {stats['abstract_classes']}")
    print(f"   Diamond Problems: {stats['diamond_problems']}")
    print(f"   Root Hierarchies: {stats['root_hierarchies']}")
    
    if report['interface_analysis']['abc_classes']:
        print(f"\nInterface Classes Found:")
        for interface in report['interface_analysis']['abc_classes'][:5]:
            print(f"   {interface}")
    
    if report['multiple_inheritance_analysis']['diamond_problems']:
        print(f"\nDiamond Inheritance Problems:")
        for problem in report['multiple_inheritance_analysis']['diamond_problems'][:3]:
            print(f"   {' -> '.join(problem)}")
    
    print(f"\nInheritance analysis complete! Report saved to {args.output}")


if __name__ == "__main__":
    main()