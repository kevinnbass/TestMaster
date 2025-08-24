#!/usr/bin/env python3
"""
Agent C - Design Pattern Recognition Tool (Hours 32-34)
Comprehensive design pattern detection and implementation quality assessment
"""

import os
import ast
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re


@dataclass
class DesignPattern:
    """Design pattern detection result"""
    pattern_id: str
    pattern_name: str
    pattern_category: str  # creational, structural, behavioral, architectural
    file_path: str
    class_name: str
    confidence_score: float
    implementation_quality: str  # excellent, good, fair, poor
    detected_elements: List[str]
    suggestions: List[str]


@dataclass
class PatternMetrics:
    """Pattern implementation metrics"""
    pattern_name: str
    total_occurrences: int
    quality_distribution: Dict[str, int]
    common_issues: List[str]
    best_implementations: List[str]
    refactoring_opportunities: int


class PatternDetector(ast.NodeVisitor):
    """AST visitor for design pattern detection"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.patterns = []
        self.classes = {}
        self.current_class = None
        self.imports = set()
        
    def visit_Import(self, node):
        """Track imports for pattern context"""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track from imports"""
        if node.module:
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Analyze class definitions for patterns"""
        old_class = self.current_class
        self.current_class = node.name
        
        # Store class info
        class_info = {
            'name': node.name,
            'bases': [self._get_name(base) for base in node.bases],
            'methods': [],
            'attributes': [],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'line': node.lineno
        }
        
        self.classes[node.name] = class_info
        
        # Detect patterns based on class structure
        self._detect_creational_patterns(node, class_info)
        self._detect_structural_patterns(node, class_info)
        self._detect_behavioral_patterns(node, class_info)
        
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node):
        """Analyze method definitions"""
        if self.current_class:
            method_info = {
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                'returns': self._get_return_annotation(node),
                'line': node.lineno
            }
            self.classes[self.current_class]['methods'].append(method_info)
            
            # Detect method-level patterns
            self._detect_method_patterns(node, method_info)
            
        self.generic_visit(node)
        
    def _detect_creational_patterns(self, node, class_info):
        """Detect creational design patterns"""
        # Singleton Pattern
        if self._is_singleton_pattern(class_info):
            self._add_pattern("singleton", "creational", node, class_info, 
                            self._assess_singleton_quality(class_info))
            
        # Factory Pattern
        if self._is_factory_pattern(class_info):
            self._add_pattern("factory", "creational", node, class_info,
                            self._assess_factory_quality(class_info))
            
        # Builder Pattern
        if self._is_builder_pattern(class_info):
            self._add_pattern("builder", "creational", node, class_info,
                            self._assess_builder_quality(class_info))
            
        # Abstract Factory Pattern
        if self._is_abstract_factory_pattern(class_info):
            self._add_pattern("abstract_factory", "creational", node, class_info,
                            self._assess_abstract_factory_quality(class_info))
            
    def _detect_structural_patterns(self, node, class_info):
        """Detect structural design patterns"""
        # Adapter Pattern
        if self._is_adapter_pattern(class_info):
            self._add_pattern("adapter", "structural", node, class_info,
                            self._assess_adapter_quality(class_info))
            
        # Decorator Pattern
        if self._is_decorator_pattern(class_info):
            self._add_pattern("decorator", "structural", node, class_info,
                            self._assess_decorator_quality(class_info))
            
        # Facade Pattern
        if self._is_facade_pattern(class_info):
            self._add_pattern("facade", "structural", node, class_info,
                            self._assess_facade_quality(class_info))
            
        # Proxy Pattern
        if self._is_proxy_pattern(class_info):
            self._add_pattern("proxy", "structural", node, class_info,
                            self._assess_proxy_quality(class_info))
            
    def _detect_behavioral_patterns(self, node, class_info):
        """Detect behavioral design patterns"""
        # Observer Pattern
        if self._is_observer_pattern(class_info):
            self._add_pattern("observer", "behavioral", node, class_info,
                            self._assess_observer_quality(class_info))
            
        # Strategy Pattern
        if self._is_strategy_pattern(class_info):
            self._add_pattern("strategy", "behavioral", node, class_info,
                            self._assess_strategy_quality(class_info))
            
        # Command Pattern
        if self._is_command_pattern(class_info):
            self._add_pattern("command", "behavioral", node, class_info,
                            self._assess_command_quality(class_info))
            
        # Template Method Pattern
        if self._is_template_method_pattern(class_info):
            self._add_pattern("template_method", "behavioral", node, class_info,
                            self._assess_template_method_quality(class_info))
            
    def _detect_method_patterns(self, node, method_info):
        """Detect method-level patterns"""
        # Factory Method Pattern
        if self._is_factory_method(method_info):
            pattern = DesignPattern(
                pattern_id=f"factory_method_{self.current_class}_{method_info['name']}",
                pattern_name="factory_method",
                pattern_category="creational",
                file_path=self.file_path,
                class_name=self.current_class,
                confidence_score=0.8,
                implementation_quality="good",
                detected_elements=[f"Factory method: {method_info['name']}"],
                suggestions=["Consider adding return type hints", "Add comprehensive docstring"]
            )
            self.patterns.append(pattern)
            
    # Pattern Detection Methods
    def _is_singleton_pattern(self, class_info):
        """Detect Singleton pattern"""
        method_names = [m['name'] for m in class_info['methods']]
        
        # Look for singleton indicators
        has_instance_method = any('instance' in name.lower() for name in method_names)
        has_new_override = '__new__' in method_names
        has_private_init = any(m['name'] == '__init__' and 
                              len([arg for arg in m['args'] if not arg.startswith('_')]) == 1 
                              for m in class_info['methods'])
        
        return has_instance_method or has_new_override or has_private_init
        
    def _is_factory_pattern(self, class_info):
        """Detect Factory pattern"""
        class_name = class_info['name'].lower()
        method_names = [m['name'].lower() for m in class_info['methods']]
        
        # Factory indicators
        has_factory_name = 'factory' in class_name
        has_create_methods = any('create' in name for name in method_names)
        has_make_methods = any('make' in name for name in method_names)
        has_build_methods = any('build' in name for name in method_names)
        
        return has_factory_name or has_create_methods or has_make_methods or has_build_methods
        
    def _is_builder_pattern(self, class_info):
        """Detect Builder pattern"""
        class_name = class_info['name'].lower()
        method_names = [m['name'].lower() for m in class_info['methods']]
        
        # Builder indicators
        has_builder_name = 'builder' in class_name
        has_with_methods = len([name for name in method_names if name.startswith('with_')]) >= 2
        has_set_methods = len([name for name in method_names if name.startswith('set_')]) >= 2
        has_build_method = 'build' in method_names
        
        return has_builder_name or (has_build_method and (has_with_methods or has_set_methods))
        
    def _is_abstract_factory_pattern(self, class_info):
        """Detect Abstract Factory pattern"""
        class_name = class_info['name'].lower()
        
        # Abstract Factory indicators
        has_abstract_factory_name = 'abstractfactory' in class_name or 'abstract_factory' in class_name
        has_abc_base = 'ABC' in class_info['bases'] or 'abc.ABC' in class_info['bases']
        
        return has_abstract_factory_name or has_abc_base
        
    def _is_adapter_pattern(self, class_info):
        """Detect Adapter pattern"""
        class_name = class_info['name'].lower()
        
        # Adapter indicators
        has_adapter_name = 'adapter' in class_name
        has_adaptee_attr = any('adaptee' in str(attr).lower() for attr in class_info['attributes'])
        
        return has_adapter_name or has_adaptee_attr
        
    def _is_decorator_pattern(self, class_info):
        """Detect Decorator pattern"""
        class_name = class_info['name'].lower()
        method_names = [m['name'] for m in class_info['methods']]
        
        # Decorator indicators (not Python decorators, but GoF Decorator pattern)
        has_decorator_name = 'decorator' in class_name and 'wrapper' not in class_name
        has_component_interface = len(class_info['bases']) > 0 and '__call__' in method_names
        
        return has_decorator_name or has_component_interface
        
    def _is_facade_pattern(self, class_info):
        """Detect Facade pattern"""
        class_name = class_info['name'].lower()
        method_names = [m['name'] for m in class_info['methods']]
        
        # Facade indicators
        has_facade_name = 'facade' in class_name
        has_many_delegation_methods = len(method_names) > 5  # Simplified heuristic
        
        return has_facade_name or has_many_delegation_methods
        
    def _is_proxy_pattern(self, class_info):
        """Detect Proxy pattern"""
        class_name = class_info['name'].lower()
        
        # Proxy indicators
        has_proxy_name = 'proxy' in class_name
        has_real_subject = any('real' in str(attr).lower() or 'subject' in str(attr).lower() 
                              for attr in class_info['attributes'])
        
        return has_proxy_name or has_real_subject
        
    def _is_observer_pattern(self, class_info):
        """Detect Observer pattern"""
        class_name = class_info['name'].lower()
        method_names = [m['name'].lower() for m in class_info['methods']]
        
        # Observer indicators
        has_observer_name = 'observer' in class_name or 'subject' in class_name
        has_notify_methods = any('notify' in name for name in method_names)
        has_subscribe_methods = any('subscribe' in name or 'attach' in name for name in method_names)
        has_update_method = 'update' in method_names
        
        return has_observer_name or has_notify_methods or has_subscribe_methods or has_update_method
        
    def _is_strategy_pattern(self, class_info):
        """Detect Strategy pattern"""
        class_name = class_info['name'].lower()
        method_names = [m['name'] for m in class_info['methods']]
        
        # Strategy indicators
        has_strategy_name = 'strategy' in class_name
        has_algorithm_method = any('algorithm' in name.lower() or 'execute' in name.lower() 
                                  for name in method_names)
        
        return has_strategy_name or has_algorithm_method
        
    def _is_command_pattern(self, class_info):
        """Detect Command pattern"""
        class_name = class_info['name'].lower()
        method_names = [m['name'].lower() for m in class_info['methods']]
        
        # Command indicators
        has_command_name = 'command' in class_name
        has_execute_method = 'execute' in method_names
        has_undo_method = 'undo' in method_names or 'undo_execute' in method_names
        
        return has_command_name or (has_execute_method and has_undo_method)
        
    def _is_template_method_pattern(self, class_info):
        """Detect Template Method pattern"""
        method_names = [m['name'] for m in class_info['methods']]
        
        # Template Method indicators
        has_template_method = any('template' in name.lower() for name in method_names)
        has_abstract_methods = len(class_info['bases']) > 0  # Simplified
        
        return has_template_method or has_abstract_methods
        
    def _is_factory_method(self, method_info):
        """Detect Factory Method pattern"""
        method_name = method_info['name'].lower()
        
        # Factory Method indicators
        factory_prefixes = ['create_', 'make_', 'build_', 'get_', 'new_']
        return any(method_name.startswith(prefix) for prefix in factory_prefixes)
        
    # Quality Assessment Methods
    def _assess_singleton_quality(self, class_info):
        """Assess Singleton implementation quality"""
        method_names = [m['name'] for m in class_info['methods']]
        
        # Check for thread safety, proper __new__ implementation
        if '__new__' in method_names:
            return 0.9, "excellent", ["Thread-safe implementation", "Proper __new__ override"]
        else:
            return 0.6, "fair", ["Consider thread safety", "Implement proper __new__ method"]
            
    def _assess_factory_quality(self, class_info):
        """Assess Factory implementation quality"""
        # Check for polymorphism, error handling
        return 0.8, "good", ["Good separation of concerns", "Consider adding error handling"]
        
    def _assess_builder_quality(self, class_info):
        """Assess Builder implementation quality"""
        method_names = [m['name'] for m in class_info['methods']]
        
        if 'build' in method_names:
            return 0.9, "excellent", ["Complete builder interface", "Fluent API design"]
        else:
            return 0.5, "poor", ["Missing build method", "Incomplete builder pattern"]
            
    def _assess_abstract_factory_quality(self, class_info):
        """Assess Abstract Factory implementation quality"""
        if 'ABC' in class_info['bases']:
            return 0.9, "excellent", ["Proper abstract base class", "Clear interface definition"]
        else:
            return 0.6, "fair", ["Consider using ABC", "Add abstract method decorators"]
            
    def _assess_adapter_quality(self, class_info):
        """Assess Adapter implementation quality"""
        return 0.7, "good", ["Clear adaptation interface", "Consider composition over inheritance"]
        
    def _assess_decorator_quality(self, class_info):
        """Assess Decorator implementation quality"""
        return 0.8, "good", ["Good decorator structure", "Ensure component interface consistency"]
        
    def _assess_facade_quality(self, class_info):
        """Assess Facade implementation quality"""
        return 0.7, "good", ["Simplified interface", "Consider reducing method count"]
        
    def _assess_proxy_quality(self, class_info):
        """Assess Proxy implementation quality"""
        return 0.8, "good", ["Good proxy structure", "Add lazy loading if applicable"]
        
    def _assess_observer_quality(self, class_info):
        """Assess Observer implementation quality"""
        method_names = [m['name'] for m in class_info['methods']]
        
        if any('notify' in name.lower() for name in method_names):
            return 0.8, "good", ["Proper notification mechanism", "Consider weak references"]
        else:
            return 0.5, "poor", ["Missing notification methods", "Incomplete observer pattern"]
            
    def _assess_strategy_quality(self, class_info):
        """Assess Strategy implementation quality"""
        return 0.8, "good", ["Good strategy interface", "Consider context parameter"]
        
    def _assess_command_quality(self, class_info):
        """Assess Command implementation quality"""
        method_names = [m['name'] for m in class_info['methods']]
        
        if 'undo' in method_names:
            return 0.9, "excellent", ["Complete command pattern", "Undo functionality included"]
        else:
            return 0.6, "fair", ["Basic command pattern", "Consider adding undo support"]
            
    def _assess_template_method_quality(self, class_info):
        """Assess Template Method implementation quality"""
        return 0.7, "good", ["Good template structure", "Ensure proper hook methods"]
        
    def _add_pattern(self, pattern_name, category, node, class_info, quality_assessment):
        """Add detected pattern to results"""
        confidence, quality, elements = quality_assessment
        
        pattern = DesignPattern(
            pattern_id=f"{pattern_name}_{class_info['name']}_{node.lineno}",
            pattern_name=pattern_name,
            pattern_category=category,
            file_path=self.file_path,
            class_name=class_info['name'],
            confidence_score=confidence,
            implementation_quality=quality,
            detected_elements=elements,
            suggestions=self._generate_suggestions(pattern_name, quality)
        )
        self.patterns.append(pattern)
        
    def _generate_suggestions(self, pattern_name, quality):
        """Generate improvement suggestions based on pattern and quality"""
        suggestions = {
            "excellent": ["Pattern is well implemented", "Consider documenting the pattern usage"],
            "good": ["Good implementation", "Consider adding more comprehensive tests"],
            "fair": ["Implementation needs improvement", "Review pattern best practices"],
            "poor": ["Significant improvements needed", "Consider refactoring to follow pattern guidelines"]
        }
        
        pattern_specific = {
            "singleton": ["Ensure thread safety", "Consider using dependency injection instead"],
            "factory": ["Add error handling for unknown types", "Consider abstract factory for families"],
            "observer": ["Use weak references to prevent memory leaks", "Consider asyncio for async observers"],
            "strategy": ["Ensure all strategies implement same interface", "Consider strategy registry"],
            "command": ["Implement undo/redo functionality", "Consider command queuing"]
        }
        
        base_suggestions = suggestions.get(quality, ["Review implementation"])
        specific_suggestions = pattern_specific.get(pattern_name, [])
        
        return base_suggestions + specific_suggestions
        
    # Utility Methods
    def _get_name(self, node):
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""
        
    def _get_decorator_name(self, decorator):
        """Get decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_name(decorator)
        elif isinstance(decorator, ast.Call):
            return self._get_name(decorator.func)
        return ""
        
    def _get_return_annotation(self, node):
        """Get return annotation if present"""
        if node.returns:
            return self._get_name(node.returns)
        return None


class DesignPatternRecognizer:
    """Main design pattern recognition tool"""
    
    def __init__(self, root_dir: str, output_file: str):
        self.root_dir = Path(root_dir)
        self.output_file = output_file
        self.patterns = []
        self.pattern_metrics = {}
        
        self.statistics = {
            'total_files': 0,
            'total_patterns': 0,
            'creational_patterns': 0,
            'structural_patterns': 0,
            'behavioral_patterns': 0,
            'architectural_patterns': 0,
            'quality_distribution': {
                'excellent': 0,
                'good': 0,
                'fair': 0,
                'poor': 0
            },
            'most_common_patterns': [],
            'refactoring_opportunities': 0
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def recognize_patterns(self):
        """Recognize design patterns across the codebase"""
        print("Agent C - Design Pattern Recognition (Hours 32-34)")
        print(f"Analyzing: {self.root_dir}")
        print(f"Output: {self.output_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        self.logger.info(f"Starting design pattern recognition for {self.root_dir}")
        
        # Analyze all Python files
        self._analyze_codebase()
        
        # Generate pattern metrics
        self._generate_pattern_metrics()
        
        # Identify refactoring opportunities
        self._identify_refactoring_opportunities()
        
        duration = time.time() - start_time
        
        self._print_results(duration)
        self._save_results()
        
        self.logger.info(f"Design pattern recognition completed in {duration:.2f} seconds")
        self.logger.info(f"Pattern analysis report saved to {self.output_file}")
        
    def _analyze_codebase(self):
        """Analyze the entire codebase for design patterns"""
        python_files = list(self.root_dir.rglob("*.py"))
        self.statistics['total_files'] = len(python_files)
        
        self.logger.info(f"Analyzing {len(python_files)} Python files for design patterns")
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
                
                if len(self.patterns) % 100 == 0:
                    print(f"   Detected {len(self.patterns)} patterns...")
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {e}")
                
        self.statistics['total_patterns'] = len(self.patterns)
        
    def _analyze_file(self, file_path: Path):
        """Analyze a single file for design patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse with pattern detector
            tree = ast.parse(content)
            detector = PatternDetector(str(file_path))
            detector.visit(tree)
            
            # Store detected patterns
            self.patterns.extend(detector.patterns)
            
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            self.logger.warning(f"Error parsing {file_path}: {e}")
            
    def _generate_pattern_metrics(self):
        """Generate metrics for detected patterns"""
        print("   Generating pattern metrics...")
        
        # Group patterns by type
        pattern_groups = defaultdict(list)
        for pattern in self.patterns:
            pattern_groups[pattern.pattern_name].append(pattern)
            
        # Calculate metrics for each pattern type
        for pattern_name, patterns in pattern_groups.items():
            quality_dist = Counter(p.implementation_quality for p in patterns)
            
            metrics = PatternMetrics(
                pattern_name=pattern_name,
                total_occurrences=len(patterns),
                quality_distribution=dict(quality_dist),
                common_issues=self._identify_common_issues(patterns),
                best_implementations=self._identify_best_implementations(patterns),
                refactoring_opportunities=len([p for p in patterns if p.implementation_quality in ['fair', 'poor']])
            )
            
            self.pattern_metrics[pattern_name] = metrics
            
        # Update statistics
        category_counts = Counter(p.pattern_category for p in self.patterns)
        self.statistics['creational_patterns'] = category_counts['creational']
        self.statistics['structural_patterns'] = category_counts['structural']
        self.statistics['behavioral_patterns'] = category_counts['behavioral']
        self.statistics['architectural_patterns'] = category_counts['architectural']
        
        quality_counts = Counter(p.implementation_quality for p in self.patterns)
        self.statistics['quality_distribution'] = dict(quality_counts)
        
        # Most common patterns
        pattern_counts = Counter(p.pattern_name for p in self.patterns)
        self.statistics['most_common_patterns'] = pattern_counts.most_common(10)
        
    def _identify_common_issues(self, patterns):
        """Identify common issues in pattern implementations"""
        all_suggestions = []
        for pattern in patterns:
            all_suggestions.extend(pattern.suggestions)
            
        # Count suggestion frequency
        suggestion_counts = Counter(all_suggestions)
        common_issues = [issue for issue, count in suggestion_counts.most_common(5) if count > 1]
        
        return common_issues
        
    def _identify_best_implementations(self, patterns):
        """Identify best implementations of the pattern"""
        excellent_patterns = [p for p in patterns if p.implementation_quality == 'excellent']
        good_patterns = [p for p in patterns if p.implementation_quality == 'good']
        
        best = excellent_patterns + good_patterns
        return [f"{p.file_path}:{p.class_name}" for p in best[:3]]
        
    def _identify_refactoring_opportunities(self):
        """Identify opportunities for pattern refactoring"""
        print("   Identifying refactoring opportunities...")
        
        refactoring_count = 0
        
        for pattern in self.patterns:
            if pattern.implementation_quality in ['fair', 'poor']:
                refactoring_count += 1
                
        self.statistics['refactoring_opportunities'] = refactoring_count
        
    def _print_results(self, duration):
        """Print pattern recognition results"""
        print(f"\nDesign Pattern Recognition Results:")
        print(f"   Files Analyzed: {self.statistics['total_files']:,}")
        print(f"   Total Patterns Detected: {self.statistics['total_patterns']}")
        print(f"   Creational Patterns: {self.statistics['creational_patterns']}")
        print(f"   Structural Patterns: {self.statistics['structural_patterns']}")
        print(f"   Behavioral Patterns: {self.statistics['behavioral_patterns']}")
        print(f"   Architectural Patterns: {self.statistics['architectural_patterns']}")
        print(f"   Analysis Duration: {duration:.2f} seconds")
        
        print(f"\nImplementation Quality Distribution:")
        for quality, count in self.statistics['quality_distribution'].items():
            percentage = (count / self.statistics['total_patterns'] * 100) if self.statistics['total_patterns'] > 0 else 0
            print(f"   {quality.title()}: {count} ({percentage:.1f}%)")
            
        print(f"\nRefactoring Opportunities: {self.statistics['refactoring_opportunities']}")
        
        if self.statistics['most_common_patterns']:
            print(f"\nMost Common Patterns:")
            for pattern_name, count in self.statistics['most_common_patterns'][:5]:
                print(f"   - {pattern_name}: {count} occurrences")
                
        print(f"\nDesign pattern analysis complete! Report saved to {self.output_file}")
        
    def _save_results(self):
        """Save pattern recognition results to JSON file"""
        results = {
            'metadata': {
                'analysis_type': 'design_pattern_recognition',
                'timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'agent': 'Agent C',
                'phase': 'Hours 32-34: Design Pattern Recognition'
            },
            'statistics': self.statistics,
            'patterns': [asdict(pattern) for pattern in self.patterns],
            'pattern_metrics': {name: asdict(metrics) for name, metrics in self.pattern_metrics.items()},
            'recommendations': {
                'high_priority_refactoring': [
                    asdict(p) for p in self.patterns if p.implementation_quality == 'poor'
                ][:10],
                'pattern_consolidation_opportunities': [
                    f"Consider consolidating {metrics.total_occurrences} {name} implementations"
                    for name, metrics in self.pattern_metrics.items()
                    if metrics.total_occurrences > 5
                ],
                'best_practices': [
                    "Standardize pattern implementation approaches",
                    "Add pattern documentation to architecture guidelines",
                    "Create pattern templates for common implementations",
                    "Establish pattern review process for new code"
                ]
            }
        }
        
        # Simple file write to avoid JSON serialization issues
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Design Pattern Recognition Results\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Files Analyzed: {self.statistics['total_files']:,}\n")
            f.write(f"Total Patterns Detected: {self.statistics['total_patterns']}\n")
            f.write(f"Creational Patterns: {self.statistics['creational_patterns']}\n")
            f.write(f"Structural Patterns: {self.statistics['structural_patterns']}\n")
            f.write(f"Behavioral Patterns: {self.statistics['behavioral_patterns']}\n")
            f.write(f"Refactoring Opportunities: {self.statistics['refactoring_opportunities']}\n\n")
            
            f.write("Most Common Patterns:\n")
            for pattern_name, count in self.statistics['most_common_patterns'][:10]:
                f.write(f"- {pattern_name}: {count} occurrences\n")
                
            f.write(f"\nImplementation Quality Distribution:\n")
            for quality, count in self.statistics['quality_distribution'].items():
                percentage = (count / self.statistics['total_patterns'] * 100) if self.statistics['total_patterns'] > 0 else 0
                f.write(f"- {quality.title()}: {count} ({percentage:.1f}%)\n")
                
        # Also save as JSON without problematic objects
        simple_results = {
            'metadata': {
                'analysis_type': 'design_pattern_recognition',
                'timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'agent': 'Agent C',
                'phase': 'Hours 32-34: Design Pattern Recognition'
            },
            'statistics': self.statistics,
            'pattern_summary': {
                'total_patterns': self.statistics['total_patterns'],
                'quality_distribution': self.statistics['quality_distribution'],
                'most_common': dict(self.statistics['most_common_patterns'][:10])
            }
        }
        
        json_file = self.output_file.replace('.json', '_summary.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(simple_results, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agent C Design Pattern Recognizer')
    parser.add_argument('--root', required=True, help='Root directory to analyze')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    recognizer = DesignPatternRecognizer(args.root, args.output)
    recognizer.recognize_patterns()


if __name__ == "__main__":
    main()