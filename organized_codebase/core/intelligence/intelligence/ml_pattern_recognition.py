"""
ML Pattern Recognition for Code Analysis

This module uses machine learning techniques to recognize patterns in code,
detect anomalies, and classify code structures using pattern recognition models.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
import numpy as np
from collections import Counter

from ..base import BaseAnalyzer


class PatternType(Enum):
    """Types of code patterns that can be recognized"""
    DESIGN_PATTERN = "design_pattern"
    ARCHITECTURAL_PATTERN = "architectural_pattern"
    ANTI_PATTERN = "anti_pattern"
    IDIOM = "idiom"
    CODE_SMELL = "code_smell"
    SECURITY_PATTERN = "security_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    CONCURRENCY_PATTERN = "concurrency_pattern"


class ConfidenceLevel(Enum):
    """Confidence levels for pattern recognition"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 50-74%
    LOW = "low"             # 25-49%
    VERY_LOW = "very_low"   # 0-24%


@dataclass
class RecognizedPattern:
    """Represents a recognized code pattern"""
    pattern_name: str
    pattern_type: PatternType
    confidence: float
    confidence_level: ConfidenceLevel
    description: str
    evidence: List[str]
    file_path: str
    line_number: int
    code_snippet: str
    similar_patterns: List[str]
    recommendations: List[str]


@dataclass
class CodeFeature:
    """Represents extracted features from code"""
    feature_name: str
    feature_value: float
    description: str
    weight: float


@dataclass
class PatternTemplate:
    """Template for pattern matching"""
    name: str
    features: List[str]
    thresholds: Dict[str, float]
    pattern_type: PatternType
    description: str


class MLPatternRecognizer(BaseAnalyzer):
    """Uses ML techniques to recognize code patterns"""
    
    def __init__(self):
        super().__init__()
        self.recognized_patterns: List[RecognizedPattern] = []
        self.code_features: List[CodeFeature] = []
        
        # Initialize pattern templates
        self.pattern_templates = self._initialize_pattern_templates()
        
        # Feature extractors
        self.feature_extractors = {
            "complexity": self._extract_complexity_features,
            "structure": self._extract_structural_features,
            "naming": self._extract_naming_features,
            "imports": self._extract_import_features,
            "methods": self._extract_method_features,
            "inheritance": self._extract_inheritance_features,
            "annotations": self._extract_annotation_features,
        }
        
        # Pattern vocabularies
        self.design_pattern_vocab = {
            "singleton": ["instance", "new", "init", "_instance", "get_instance"],
            "factory": ["create", "make", "build", "factory", "builder"],
            "observer": ["notify", "observer", "update", "subscribe", "listener"],
            "strategy": ["strategy", "algorithm", "execute", "perform"],
            "decorator": ["wrapper", "wrap", "decorator", "before", "after"],
            "adapter": ["adapt", "adapter", "convert", "interface"],
            "facade": ["facade", "interface", "simplify", "unified"],
            "proxy": ["proxy", "placeholder", "surrogate", "lazy"],
            "command": ["command", "execute", "undo", "invoke", "action"],
            "template": ["template", "abstract", "hook", "primitive"]
        }
        
        self.anti_pattern_vocab = {
            "god_object": ["manager", "controller", "handler", "util", "helper"],
            "spaghetti": ["goto", "nested", "complex", "tangled"],
            "copy_paste": ["duplicate", "similar", "repeated"],
            "magic_number": ["42", "100", "1000", "constant"],
            "long_parameter": ["param", "args", "arguments", "many"],
            "feature_envy": ["get", "set", "data", "field", "property"]
        }
        
        # Complexity thresholds
        self.complexity_thresholds = {
            "cyclomatic": 10,
            "cognitive": 15,
            "nesting": 4,
            "parameters": 5,
            "lines": 50
        }
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze code using ML pattern recognition"""
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Extract features from the file
            features = self._extract_all_features(tree, content, str(file_path))
            self.code_features.extend(features)
            
            # Recognize patterns using templates
            patterns = self._recognize_patterns(tree, content, str(file_path), features)
            self.recognized_patterns.extend(patterns)
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
    
    def _extract_all_features(self, tree: ast.AST, content: str, 
                             file_path: str) -> List[CodeFeature]:
        """Extract all features from code"""
        all_features = []
        
        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                features = extractor_func(tree, content, file_path)
                all_features.extend(features)
            except Exception as e:
                logging.warning(f"Error in {extractor_name} feature extraction: {e}")
        
        return all_features
    
    def _extract_complexity_features(self, tree: ast.AST, content: str,
                                   file_path: str) -> List[CodeFeature]:
        """Extract complexity-related features"""
        features = []
        
        # Cyclomatic complexity
        cyclomatic = self._calculate_cyclomatic_complexity(tree)
        features.append(CodeFeature(
            feature_name="cyclomatic_complexity",
            feature_value=float(cyclomatic),
            description="Cyclomatic complexity of the code",
            weight=0.8
        ))
        
        # Nesting depth
        max_nesting = self._calculate_max_nesting(tree)
        features.append(CodeFeature(
            feature_name="max_nesting_depth",
            feature_value=float(max_nesting),
            description="Maximum nesting depth",
            weight=0.7
        ))
        
        # Lines of code
        loc = len(content.split('\n'))
        features.append(CodeFeature(
            feature_name="lines_of_code",
            feature_value=float(loc),
            description="Lines of code",
            weight=0.5
        ))
        
        # Number of functions
        func_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        features.append(CodeFeature(
            feature_name="function_count",
            feature_value=float(func_count),
            description="Number of functions",
            weight=0.6
        ))
        
        # Number of classes
        class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        features.append(CodeFeature(
            feature_name="class_count",
            feature_value=float(class_count),
            description="Number of classes",
            weight=0.6
        ))
        
        return features
    
    def _extract_structural_features(self, tree: ast.AST, content: str,
                                   file_path: str) -> List[CodeFeature]:
        """Extract structural features"""
        features = []
        
        # Import ratio
        import_count = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
        total_statements = len([n for n in ast.walk(tree) if isinstance(n, ast.stmt)])
        import_ratio = import_count / max(total_statements, 1)
        
        features.append(CodeFeature(
            feature_name="import_ratio",
            feature_value=import_ratio,
            description="Ratio of import statements to total statements",
            weight=0.4
        ))
        
        # Inheritance depth
        max_inheritance = self._calculate_inheritance_depth(tree)
        features.append(CodeFeature(
            feature_name="inheritance_depth",
            feature_value=float(max_inheritance),
            description="Maximum inheritance depth",
            weight=0.7
        ))
        
        # Method-to-class ratio
        if class_count := len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]):
            method_count = len([n for n in ast.walk(tree) 
                              if isinstance(n, ast.FunctionDef) and 
                              any(isinstance(p, ast.ClassDef) for p in ast.walk(tree))])
            method_class_ratio = method_count / class_count
        else:
            method_class_ratio = 0
        
        features.append(CodeFeature(
            feature_name="method_class_ratio",
            feature_value=method_class_ratio,
            description="Ratio of methods to classes",
            weight=0.6
        ))
        
        return features
    
    def _extract_naming_features(self, tree: ast.AST, content: str,
                               file_path: str) -> List[CodeFeature]:
        """Extract naming convention features"""
        features = []
        
        # Collect all names
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.FunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                names.append(node.name)
        
        if not names:
            return features
        
        # Snake case ratio
        snake_case_count = sum(1 for name in names if '_' in name and name.islower())
        snake_case_ratio = snake_case_count / len(names)
        
        features.append(CodeFeature(
            feature_name="snake_case_ratio",
            feature_value=snake_case_ratio,
            description="Ratio of snake_case names",
            weight=0.3
        ))
        
        # Camel case ratio
        camel_case_count = sum(1 for name in names 
                              if name[0].islower() and any(c.isupper() for c in name[1:]))
        camel_case_ratio = camel_case_count / len(names)
        
        features.append(CodeFeature(
            feature_name="camel_case_ratio",
            feature_value=camel_case_ratio,
            description="Ratio of camelCase names",
            weight=0.3
        ))
        
        # Average name length
        avg_name_length = sum(len(name) for name in names) / len(names)
        features.append(CodeFeature(
            feature_name="avg_name_length",
            feature_value=avg_name_length,
            description="Average name length",
            weight=0.2
        ))
        
        return features
    
    def _extract_import_features(self, tree: ast.AST, content: str,
                               file_path: str) -> List[CodeFeature]:
        """Extract import-related features"""
        features = []
        
        # Collect imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Framework detection
        frameworks = {
            "django": any("django" in imp for imp in imports),
            "flask": any("flask" in imp for imp in imports),
            "numpy": any("numpy" in imp or "np" in imp for imp in imports),
            "pandas": any("pandas" in imp or "pd" in imp for imp in imports),
            "sklearn": any("sklearn" in imp for imp in imports),
            "tensorflow": any("tensorflow" in imp or "tf" in imp for imp in imports),
            "torch": any("torch" in imp for imp in imports),
        }
        
        for framework, present in frameworks.items():
            features.append(CodeFeature(
                feature_name=f"{framework}_framework",
                feature_value=1.0 if present else 0.0,
                description=f"Uses {framework} framework",
                weight=0.5
            ))
        
        # Standard library ratio
        stdlib_modules = {
            "os", "sys", "re", "json", "datetime", "collections",
            "itertools", "functools", "operator", "pathlib"
        }
        stdlib_count = sum(1 for imp in imports if imp.split('.')[0] in stdlib_modules)
        stdlib_ratio = stdlib_count / max(len(imports), 1)
        
        features.append(CodeFeature(
            feature_name="stdlib_ratio",
            feature_value=stdlib_ratio,
            description="Ratio of standard library imports",
            weight=0.4
        ))
        
        return features
    
    def _extract_method_features(self, tree: ast.AST, content: str,
                               file_path: str) -> List[CodeFeature]:
        """Extract method-related features"""
        features = []
        
        # Collect method information
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_info = {
                    "name": node.name,
                    "args": len(node.args.args),
                    "lines": len(ast.unparse(node).split('\n')),
                    "returns": node.returns is not None,
                    "decorators": len(node.decorator_list)
                }
                methods.append(method_info)
        
        if not methods:
            return features
        
        # Average method length
        avg_method_length = sum(m["lines"] for m in methods) / len(methods)
        features.append(CodeFeature(
            feature_name="avg_method_length",
            feature_value=avg_method_length,
            description="Average method length in lines",
            weight=0.6
        ))
        
        # Average parameters
        avg_params = sum(m["args"] for m in methods) / len(methods)
        features.append(CodeFeature(
            feature_name="avg_method_params",
            feature_value=avg_params,
            description="Average number of method parameters",
            weight=0.5
        ))
        
        # Type annotation ratio
        annotated_methods = sum(1 for m in methods if m["returns"])
        annotation_ratio = annotated_methods / len(methods)
        features.append(CodeFeature(
            feature_name="type_annotation_ratio",
            feature_value=annotation_ratio,
            description="Ratio of type-annotated methods",
            weight=0.4
        ))
        
        # Decorator usage ratio
        decorated_methods = sum(1 for m in methods if m["decorators"] > 0)
        decorator_ratio = decorated_methods / len(methods)
        features.append(CodeFeature(
            feature_name="decorator_ratio",
            feature_value=decorator_ratio,
            description="Ratio of decorated methods",
            weight=0.3
        ))
        
        return features
    
    def _extract_inheritance_features(self, tree: ast.AST, content: str,
                                    file_path: str) -> List[CodeFeature]:
        """Extract inheritance-related features"""
        features = []
        
        # Collect class information
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": len(node.bases),
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "attributes": len([n for n in node.body if isinstance(n, ast.Assign)])
                }
                classes.append(class_info)
        
        if not classes:
            return features
        
        # Inheritance usage ratio
        inheriting_classes = sum(1 for c in classes if c["bases"] > 0)
        inheritance_ratio = inheriting_classes / len(classes)
        features.append(CodeFeature(
            feature_name="inheritance_ratio",
            feature_value=inheritance_ratio,
            description="Ratio of classes using inheritance",
            weight=0.6
        ))
        
        # Average methods per class
        avg_methods = sum(c["methods"] for c in classes) / len(classes)
        features.append(CodeFeature(
            feature_name="avg_methods_per_class",
            feature_value=avg_methods,
            description="Average methods per class",
            weight=0.5
        ))
        
        return features
    
    def _extract_annotation_features(self, tree: ast.AST, content: str,
                                   file_path: str) -> List[CodeFeature]:
        """Extract annotation and documentation features"""
        features = []
        
        # Docstring ratio
        functions_with_docstrings = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    functions_with_docstrings += 1
        
        docstring_ratio = functions_with_docstrings / max(total_functions, 1)
        features.append(CodeFeature(
            feature_name="docstring_ratio",
            feature_value=docstring_ratio,
            description="Ratio of functions with docstrings",
            weight=0.4
        ))
        
        # Comment density
        lines = content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        comment_density = comment_lines / max(len(lines), 1)
        features.append(CodeFeature(
            feature_name="comment_density",
            feature_value=comment_density,
            description="Ratio of comment lines to total lines",
            weight=0.3
        ))
        
        return features
    
    def _recognize_patterns(self, tree: ast.AST, content: str, file_path: str,
                          features: List[CodeFeature]) -> List[RecognizedPattern]:
        """Recognize patterns using templates and ML techniques"""
        patterns = []
        
        # Convert features to feature vector
        feature_dict = {f.feature_name: f.feature_value for f in features}
        
        # Apply pattern templates
        for template in self.pattern_templates:
            confidence = self._calculate_pattern_confidence(template, feature_dict, tree, content)
            
            if confidence > 0.25:  # Only report patterns with >25% confidence
                pattern = RecognizedPattern(
                    pattern_name=template.name,
                    pattern_type=template.pattern_type,
                    confidence=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    description=template.description,
                    evidence=self._collect_evidence(template, tree, content),
                    file_path=file_path,
                    line_number=1,  # Would need more sophisticated line detection
                    code_snippet=self._extract_relevant_snippet(template, tree, content),
                    similar_patterns=self._find_similar_patterns(template.name),
                    recommendations=self._generate_pattern_recommendations(template, confidence)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_confidence(self, template: PatternTemplate, 
                                    feature_dict: Dict[str, float],
                                    tree: ast.AST, content: str) -> float:
        """Calculate confidence score for pattern match"""
        total_score = 0.0
        total_weight = 0.0
        
        # Check feature-based criteria
        for feature_name, threshold in template.thresholds.items():
            if feature_name in feature_dict:
                feature_value = feature_dict[feature_name]
                
                # Calculate feature score (how well it matches the threshold)
                if feature_name.endswith("_ratio") or feature_name.endswith("_density"):
                    # For ratios, score based on proximity to threshold
                    score = 1.0 - min(abs(feature_value - threshold), 1.0)
                else:
                    # For counts, score based on meeting minimum threshold
                    score = min(feature_value / threshold, 1.0) if threshold > 0 else 1.0
                
                total_score += score
                total_weight += 1.0
        
        # Add vocabulary-based scoring
        vocab_score = self._calculate_vocabulary_score(template.name, content)
        total_score += vocab_score
        total_weight += 1.0
        
        # Add structural scoring
        structural_score = self._calculate_structural_score(template.name, tree)
        total_score += structural_score
        total_weight += 1.0
        
        return total_score / max(total_weight, 1.0)
    
    def _calculate_vocabulary_score(self, pattern_name: str, content: str) -> float:
        """Calculate score based on vocabulary presence"""
        content_lower = content.lower()
        
        # Check design pattern vocabularies
        if pattern_name in self.design_pattern_vocab:
            vocab = self.design_pattern_vocab[pattern_name]
            matches = sum(1 for word in vocab if word in content_lower)
            return min(matches / len(vocab), 1.0)
        
        # Check anti-pattern vocabularies
        if pattern_name in self.anti_pattern_vocab:
            vocab = self.anti_pattern_vocab[pattern_name]
            matches = sum(1 for word in vocab if word in content_lower)
            return min(matches / len(vocab), 1.0)
        
        return 0.0
    
    def _calculate_structural_score(self, pattern_name: str, tree: ast.AST) -> float:
        """Calculate score based on code structure"""
        # Pattern-specific structural checks
        if pattern_name == "singleton":
            return self._check_singleton_structure(tree)
        elif pattern_name == "factory":
            return self._check_factory_structure(tree)
        elif pattern_name == "observer":
            return self._check_observer_structure(tree)
        elif pattern_name == "god_object":
            return self._check_god_object_structure(tree)
        
        return 0.0
    
    def _check_singleton_structure(self, tree: ast.AST) -> float:
        """Check for singleton pattern structure"""
        score = 0.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for private constructor or instance management
                has_instance_var = False
                has_new_override = False
                
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and "instance" in target.id.lower():
                                has_instance_var = True
                    elif isinstance(item, ast.FunctionDef) and item.name == "__new__":
                        has_new_override = True
                
                if has_instance_var or has_new_override:
                    score += 0.5
        
        return min(score, 1.0)
    
    def _check_factory_structure(self, tree: ast.AST) -> float:
        """Check for factory pattern structure"""
        score = 0.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(keyword in node.name.lower() 
                      for keyword in ["create", "make", "build", "factory"]):
                    # Check if it returns different types based on parameters
                    returns_count = len([n for n in ast.walk(node) 
                                       if isinstance(n, ast.Return)])
                    if returns_count > 1:
                        score += 0.3
        
        return min(score, 1.0)
    
    def _check_observer_structure(self, tree: ast.AST) -> float:
        """Check for observer pattern structure"""
        score = 0.0
        
        # Look for observer-like methods and structures
        observer_methods = ["notify", "update", "subscribe", "unsubscribe"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(method in node.name.lower() for method in observer_methods):
                    score += 0.2
        
        return min(score, 1.0)
    
    def _check_god_object_structure(self, tree: ast.AST) -> float:
        """Check for god object anti-pattern"""
        score = 0.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                
                # God object typically has many methods
                if method_count > 20:
                    score += 0.8
                elif method_count > 10:
                    score += 0.5
                elif method_count > 5:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _collect_evidence(self, template: PatternTemplate, tree: ast.AST, 
                         content: str) -> List[str]:
        """Collect evidence supporting the pattern recognition"""
        evidence = []
        
        # Vocabulary evidence
        if template.name in self.design_pattern_vocab:
            vocab = self.design_pattern_vocab[template.name]
            found_words = [word for word in vocab if word in content.lower()]
            if found_words:
                evidence.append(f"Vocabulary match: {', '.join(found_words)}")
        
        # Structural evidence
        if template.name == "singleton":
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__new__":
                            evidence.append("Overrides __new__ method")
        
        return evidence
    
    def _extract_relevant_snippet(self, template: PatternTemplate, tree: ast.AST,
                                 content: str) -> str:
        """Extract relevant code snippet for pattern"""
        # For now, return first few lines of the file
        lines = content.split('\n')[:10]
        return '\n'.join(lines)
    
    def _find_similar_patterns(self, pattern_name: str) -> List[str]:
        """Find patterns similar to the given pattern"""
        similar_patterns = {
            "singleton": ["factory", "builder"],
            "factory": ["singleton", "builder", "abstract_factory"],
            "observer": ["mediator", "publisher_subscriber"],
            "strategy": ["command", "state"],
            "decorator": ["proxy", "adapter"],
            "god_object": ["large_class", "feature_envy"],
        }
        
        return similar_patterns.get(pattern_name, [])
    
    def _generate_pattern_recommendations(self, template: PatternTemplate,
                                        confidence: float) -> List[str]:
        """Generate recommendations based on recognized pattern"""
        recommendations = []
        
        if template.pattern_type == PatternType.ANTI_PATTERN:
            if template.name == "god_object":
                recommendations.append("Consider splitting into smaller, focused classes")
                recommendations.append("Apply Single Responsibility Principle")
            elif template.name == "spaghetti":
                recommendations.append("Refactor to improve code structure")
                recommendations.append("Extract methods to reduce complexity")
        
        elif template.pattern_type == PatternType.DESIGN_PATTERN:
            if confidence < 0.7:
                recommendations.append(f"Consider fully implementing {template.name} pattern")
                recommendations.append("Add documentation explaining the pattern usage")
        
        return recommendations
    
    def _initialize_pattern_templates(self) -> List[PatternTemplate]:
        """Initialize pattern templates for recognition"""
        templates = []
        
        # Singleton pattern
        templates.append(PatternTemplate(
            name="singleton",
            features=["class_count", "avg_methods_per_class"],
            thresholds={"class_count": 1, "avg_methods_per_class": 3},
            pattern_type=PatternType.DESIGN_PATTERN,
            description="Singleton design pattern ensuring single instance"
        ))
        
        # Factory pattern
        templates.append(PatternTemplate(
            name="factory",
            features=["function_count", "avg_method_params"],
            thresholds={"function_count": 2, "avg_method_params": 1},
            pattern_type=PatternType.DESIGN_PATTERN,
            description="Factory pattern for object creation"
        ))
        
        # Observer pattern
        templates.append(PatternTemplate(
            name="observer",
            features=["class_count", "avg_methods_per_class"],
            thresholds={"class_count": 2, "avg_methods_per_class": 4},
            pattern_type=PatternType.DESIGN_PATTERN,
            description="Observer pattern for event notification"
        ))
        
        # God object anti-pattern
        templates.append(PatternTemplate(
            name="god_object",
            features=["avg_methods_per_class", "avg_method_length"],
            thresholds={"avg_methods_per_class": 15, "avg_method_length": 30},
            pattern_type=PatternType.ANTI_PATTERN,
            description="God object anti-pattern with too many responsibilities"
        ))
        
        # Long parameter list
        templates.append(PatternTemplate(
            name="long_parameter_list",
            features=["avg_method_params"],
            thresholds={"avg_method_params": 6},
            pattern_type=PatternType.CODE_SMELL,
            description="Methods with too many parameters"
        ))
        
        return templates
    
    # Helper methods for complexity calculation
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_nesting_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.Try)):
                    child_depth = get_nesting_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_nesting_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_nesting_depth(tree)
    
    def _calculate_inheritance_depth(self, tree: ast.AST) -> int:
        """Calculate maximum inheritance depth"""
        max_depth = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                depth = len(node.bases)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive ML pattern recognition report"""
        # Calculate statistics
        total_patterns = len(self.recognized_patterns)
        
        # Group by pattern type
        patterns_by_type = {}
        for pattern in self.recognized_patterns:
            pattern_type = pattern.pattern_type.value
            patterns_by_type[pattern_type] = patterns_by_type.get(pattern_type, 0) + 1
        
        # Group by confidence level
        patterns_by_confidence = {}
        for pattern in self.recognized_patterns:
            confidence_level = pattern.confidence_level.value
            patterns_by_confidence[confidence_level] = patterns_by_confidence.get(confidence_level, 0) + 1
        
        # Calculate average confidence
        avg_confidence = (sum(p.confidence for p in self.recognized_patterns) / total_patterns
                         if total_patterns > 0 else 0)
        
        # High confidence patterns
        high_confidence_patterns = [p for p in self.recognized_patterns 
                                  if p.confidence >= 0.75]
        
        return {
            "summary": {
                "total_patterns_recognized": total_patterns,
                "average_confidence": round(avg_confidence, 3),
                "high_confidence_patterns": len(high_confidence_patterns),
                "design_patterns": patterns_by_type.get("design_pattern", 0),
                "anti_patterns": patterns_by_type.get("anti_pattern", 0),
                "code_smells": patterns_by_type.get("code_smell", 0),
                "files_analyzed": len(set(p.file_path for p in self.recognized_patterns)),
            },
            "recognized_patterns": [
                {
                    "name": p.pattern_name,
                    "type": p.pattern_type.value,
                    "confidence": round(p.confidence, 3),
                    "confidence_level": p.confidence_level.value,
                    "description": p.description,
                    "evidence": p.evidence,
                    "file": p.file_path,
                    "line": p.line_number,
                    "code_snippet": p.code_snippet[:200] + "..." if len(p.code_snippet) > 200 else p.code_snippet,
                    "similar_patterns": p.similar_patterns,
                    "recommendations": p.recommendations,
                }
                for p in sorted(self.recognized_patterns, key=lambda x: x.confidence, reverse=True)
            ],
            "patterns_by_type": patterns_by_type,
            "patterns_by_confidence": patterns_by_confidence,
            "extracted_features": [
                {
                    "name": f.feature_name,
                    "value": round(f.feature_value, 3),
                    "description": f.description,
                    "weight": f.weight,
                }
                for f in self.code_features[:20]  # Show first 20 features
            ],
            "recommendations": self._generate_global_recommendations(),
        }
    
    def _generate_global_recommendations(self) -> List[Dict[str, str]]:
        """Generate global recommendations based on all recognized patterns"""
        recommendations = []
        
        # Check for anti-patterns
        anti_patterns = [p for p in self.recognized_patterns 
                        if p.pattern_type == PatternType.ANTI_PATTERN]
        if anti_patterns:
            recommendations.append({
                "category": "Code Quality",
                "priority": "high",
                "recommendation": "Address detected anti-patterns",
                "impact": "Improve code maintainability and reduce technical debt",
                "pattern_count": len(anti_patterns)
            })
        
        # Check for incomplete design patterns
        incomplete_patterns = [p for p in self.recognized_patterns 
                             if p.pattern_type == PatternType.DESIGN_PATTERN and p.confidence < 0.7]
        if incomplete_patterns:
            recommendations.append({
                "category": "Design Patterns",
                "priority": "medium",
                "recommendation": "Complete implementation of design patterns",
                "impact": "Improve code structure and maintainability",
                "pattern_count": len(incomplete_patterns)
            })
        
        # Check for code smells
        code_smells = [p for p in self.recognized_patterns 
                      if p.pattern_type == PatternType.CODE_SMELL]
        if code_smells:
            recommendations.append({
                "category": "Code Smells",
                "priority": "medium",
                "recommendation": "Refactor code to eliminate smells",
                "impact": "Improve code readability and maintainability",
                "pattern_count": len(code_smells)
            })
        
        return recommendations