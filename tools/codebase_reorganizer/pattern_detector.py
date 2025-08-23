#!/usr/bin/env python3
"""
Pattern Detector
================

Detects common code patterns, architectural patterns, and design patterns
in Python code. This intelligence module helps the reorganizer understand
the coding style, architecture, and patterns used in the codebase.

Key capabilities:
- Design pattern detection (Singleton, Factory, Observer, etc.)
- Architectural pattern recognition (MVC, Layered, etc.)
- Code style pattern identification
- Anti-pattern detection
- Pattern confidence scoring
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DetectedPattern:
    """A detected pattern in the code"""
    pattern_type: str
    pattern_name: str
    confidence: float
    location: str  # class name, function name, or 'module'
    evidence: List[str]
    category: str  # 'design', 'architectural', 'coding', 'anti-pattern'
    recommendations: List[str]


@dataclass
class PatternAnalysis:
    """Complete pattern analysis results"""
    patterns: List[DetectedPattern]
    pattern_categories: Dict[str, List[str]]
    confidence_distribution: Dict[str, int]
    architectural_style: str
    coding_style: str
    recommendations: List[str]


class PatternDetector:
    """
    Detects patterns in Python code including design patterns,
    architectural patterns, and coding style patterns.
    """

    def __init__(self) -> None:
        """Initialize the pattern detector"""
        self._load_pattern_definitions()

    def _load_creational_patterns(self) -> Dict[str, Dict]:
        """Load creational design pattern definitions"""
        return {
            'singleton': {
                'indicators': [
                    r'class.*:\s*\n.*_instance\s*=\s*None',
                    r'class.*:\s*\n.*@classmethod\s*\n.*get_instance',
                    r'_instance\s*=\s*None\s*\n.*def.*__new__',
                    r'class.*:\s*\n.*def.*__init__.*raise.*Error'
                ],
                'keywords': ['singleton', 'instance', 'get_instance', 'getInstance'],
                'structure': ['class', 'class_method', 'private_constructor']
            },
            'factory': {
                'indicators': [
                    r'def.*create.*\(.*\).*->',
                    r'def.*make.*\(.*\).*->',
                    r'def.*build.*\(.*\).*->',
                    r'class.*Factory'
                ],
                'keywords': ['factory', 'create', 'make', 'build', 'creator'],
                'structure': ['creator_method', 'factory_class']
            }
        }


    def _load_behavioral_patterns(self) -> Dict[str, Dict]:
        """Load behavioral design pattern definitions"""
        return {
            'observer': {
                'indicators': [
                    r'def.*notify.*\(.*\)',
                    r'def.*subscribe.*\(.*\)',
                    r'def.*unsubscribe.*\(.*\)',
                    r'def.*update.*\(.*\)',
                    r'class.*Observer',
                    r'class.*Subject'
                ],
                'keywords': ['observer', 'subject', 'notify', 'subscribe', 'callback', 'listener'],
                'structure': ['observer_interface', 'subject_interface', 'notify_method']
            },
            'strategy': {
                'indicators': [
                    r'class.*Strategy',
                    r'def.*execute.*\(.*\)',
                    r'def.*algorithm.*\(.*\)'
                ],
                'keywords': ['strategy', 'algorithm', 'context', 'behavior', 'policy'],
                'structure': ['strategy_interface', 'context_class']
            }
        }


    def _load_structural_patterns(self) -> Dict[str, Dict]:
        """Load structural design pattern definitions"""
        return {
            'adapter': {
                'indicators': [
                    r'class.*Adapter',
                    r'def.*adapt.*\(.*\)',
                    r'def.*convert.*\(.*\)',
                    r'class.*Wrapper'
                ],
                'keywords': ['adapter', 'wrapper', 'convert', 'adapt', 'interface'],
                'structure': ['adapter_class', 'adaptee_interface', 'target_interface']
            },
            'decorator': {
                'indicators': [
                    r'class.*Decorator',
                    r'def.*decorate.*\(.*\)',
                    r'class.*Component',
                    r'def.*operation.*\(.*\)'
                ],
                'keywords': ['decorator', 'component', 'wrap', 'enhance', 'extend'],
                'structure': ['decorator_class', 'component_interface']
            }
        }


    def _load_design_patterns(self) -> Dict[str, Dict]:
        """Load all design pattern definitions"""
        patterns = {}

        # Load patterns by category
        patterns.update(self._load_creational_patterns())
        patterns.update(self._load_behavioral_patterns())
        patterns.update(self._load_structural_patterns())

        return patterns


    def _load_enterprise_patterns(self) -> Dict[str, Dict]:
        """Load enterprise architectural pattern definitions"""
        return {
            'layered_architecture': {
                'indicators': [
                    r'class.*Controller',
                    r'class.*Service',
                    r'class.*Repository',
                    r'def.*handle.*request',
                    r'presentation|business|data'
                ],
                'keywords': ['layer', 'presentation', 'business', 'data', 'service', 'controller', 'repository'],
                'structure': ['presentation_layer', 'business_layer', 'data_layer']
            },
            'microservices': {
                'indicators': [
                    r'class.*Service',
                    r'def.*api.*endpoint',
                    r'def.*rest.*endpoint',
                    r'class.*Controller',
                    r'service.*discovery'
                ],
                'keywords': ['microservice', 'api', 'endpoint', 'rest', 'service', 'discovery'],
                'structure': ['service_component', 'api_endpoint', 'data_service']
            }
        }


    def _load_messaging_patterns(self) -> Dict[str, Dict]:
        """Load messaging and event-driven pattern definitions"""
        return {
            'event_driven': {
                'indicators': [
                    r'def.*handle.*event',
                    r'def.*process.*event',
                    r'class.*Event',
                    r'class.*Handler',
                    r'publish.*subscribe'
                ],
                'keywords': ['event', 'handler', 'publish', 'subscribe', 'message', 'queue'],
                'structure': ['event_handler', 'event_publisher', 'message_queue']
            },
            'message_queue': {
                'indicators': [
                    r'queue.*put',
                    r'queue.*get',
                    r'class.*Queue',
                    r'async.*queue',
                    r'concurrent.*queue'
                ],
                'keywords': ['queue', 'message', 'async', 'concurrent', 'producer', 'consumer'],
                'structure': ['message_queue', 'producer', 'consumer']
            }
        }


    def _load_architectural_patterns(self) -> Dict[str, Dict]:
        """Load all architectural pattern definitions"""
        patterns = {}
        patterns.update(self._load_enterprise_patterns())
        patterns.update(self._load_messaging_patterns())
        return patterns


    def _load_basic_coding_patterns(self) -> Dict[str, Dict]:
        """Load basic coding style pattern definitions"""
        return {
            'functional': {
                'indicators': [
                    r'lambda.*:',
                    r'map\(.*\)',
                    r'filter\(.*\)',
                    r'reduce\(.*\)',
                    r'def.*\(.*\).*->.*:'
                ],
                'keywords': ['lambda', 'map', 'filter', 'reduce', 'comprehension']
            },
            'object_oriented': {
                'indicators': [
                    r'class.*:',
                    r'def.*__init__.*\(.*\)',
                    r'self\.',
                    r'class.*\(.*\):',
                    r'instance.*method'
                ],
                'keywords': ['class', 'self', 'instance', 'method', 'inheritance']
            },
            'procedural': {
                'indicators': [
                    r'def.*main.*\(.*\)',
                    r'if.*__name__.*==.*__main__',
                    r'global.*variables'
                ],
                'keywords': ['main', 'procedure', 'global', 'sequential']
            }
        }


    def _load_coding_patterns(self) -> Dict[str, Dict]:
        """Load all coding pattern definitions"""
        return self._load_basic_coding_patterns()


    def _load_architectural_patterns(self) -> Dict[str, Dict]:
        """Load architectural pattern definitions - calls helper functions"""
        return self._load_enterprise_patterns() | self._load_messaging_patterns()


    def _load_coding_patterns(self) -> Dict[str, Dict]:
        """Load coding pattern definitions"""
        return self._load_basic_coding_patterns()


    def _load_anti_patterns(self) -> Dict[str, Dict]:
        """Load anti-pattern definitions"""
        return {
            'god_object': {
                'indicators': [
                    r'class.*:\s*\n(?:.*\n){50,}',  # Very large class
                    r'def.*__init__.*\(.*self.*,\s*(?:\w+,\s*){10,}',  # Many parameters
                ],
                'threshold': 50  # Lines threshold
            },
            'tight_coupling': {
                'indicators': [
                    r'import.*\*',  # Wildcard imports
                    r'from.*import.*,\s*,',  # Many imports from one module
                ]
            },
            'magic_numbers': {
                'indicators': [
                    r'\b\d{3,}\b',  # Large numbers without constants
                ]
            },
            'long_functions': {
                'threshold': 50  # Lines threshold
            }
        }


    def detect_patterns(self, content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Detect patterns in Python code content.

        Args:
            content: The code content to analyze
            file_path: Optional path to the file

        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            # Initialize pattern detection
            all_patterns = []
            pattern_categories = defaultdict(list)

            # Detect different types of patterns
            design_patterns_found = self._detect_design_patterns(content, tree)
            architectural_patterns_found = self._detect_architectural_patterns(content, tree)
            coding_patterns_found = self._detect_coding_patterns(content, tree)
            anti_patterns_found = self._detect_anti_patterns(content, tree)

            # Combine all patterns
            all_patterns.extend(design_patterns_found)
            all_patterns.extend(architectural_patterns_found)
            all_patterns.extend(coding_patterns_found)
            all_patterns.extend(anti_patterns_found)


    def _load_pattern_definitions(self) -> None:
        """Load all pattern definitions and detection rules"""
        self.design_patterns = self._load_design_patterns()
        self.architectural_patterns = self._load_architectural_patterns()
        self.coding_patterns = self._load_coding_patterns()
        self.anti_patterns = self._load_anti_patterns()


    def detect_patterns(self, content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Detect patterns in Python code content.

        Args:
            content: The code content to analyze
            file_path: Optional path to the file

        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            tree = ast.parse(content)

            # Detect different types of patterns
            design_patterns = self._detect_design_patterns(content, tree)
            architectural_patterns = self._detect_architectural_patterns(content, tree)
            coding_patterns = self._detect_coding_patterns(content, tree)
            anti_patterns = self._detect_anti_patterns(content, tree)

            # Combine all patterns
            all_patterns = design_patterns + architectural_patterns + coding_patterns + anti_patterns

            # Analyze pattern distribution
            pattern_categories = self._categorize_patterns(all_patterns)

            # Determine overall architectural and coding style
            architectural_style = self._determine_architectural_style(architectural_patterns)
            coding_style = self._determine_coding_style(coding_patterns)

            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(
                all_patterns, architectural_style, coding_style
            )

            # Calculate confidence distribution
            confidence_distribution = self._calculate_confidence_distribution(all_patterns)

            # Convert patterns to dictionaries (replacing complex comprehension with explicit loop)
            pattern_list = []
            for pattern in all_patterns:
                if hasattr(pattern, '__dict__'):
                    pattern_list.append(pattern.__dict__)
                else:
                    pattern_list.append(pattern)

            # Count high confidence patterns (replacing complex comprehension with explicit loop)
            high_confidence_count = 0
            for p in all_patterns:
                if getattr(p, 'confidence', 0) > 0.8:
                    high_confidence_count += 1

            result = {
                'patterns': pattern_list,
                'pattern_categories': pattern_categories,
                'confidence_distribution': confidence_distribution,
                'architectural_style': architectural_style,
                'coding_style': coding_style,
                'recommendations': recommendations,
                'file_path': str(file_path) if file_path else 'unknown',
                'total_patterns': len(all_patterns),
                'high_confidence_patterns': high_confidence_count
            }

            return result

        except SyntaxError as e:
            return self._fallback_pattern_detection(content, file_path, e)
        except Exception as e:
            return {
                'error': f'Pattern detection failed: {e}',
                'patterns': [],
                'pattern_categories': {},
                'confidence_distribution': {},
                'architectural_style': 'unknown',
                'coding_style': 'unknown',
                'recommendations': ['Unable to analyze patterns due to error'],
                'file_path': str(file_path) if file_path else 'unknown',
                'total_patterns': 0,
                'high_confidence_patterns': 0
            }

    def _detect_design_patterns(self, content: str, tree: ast.AST) -> List[DetectedPattern]:
        """Detect design patterns in the code"""
        patterns = []

        for pattern_name, pattern_def in self.design_patterns.items():
            confidence = 0.0
            evidence = []

            # Check for indicators
            for indicator in pattern_def['indicators']:
                if re.search(indicator, content, re.MULTILINE | re.IGNORECASE):
                    confidence += 0.3
                    evidence.append(f"Found pattern indicator: {indicator}")

            # Check for keywords
            keyword_matches = 0
            for keyword in pattern_def['keywords']:
                if keyword.lower() in content.lower():
                    keyword_matches += 1

            if keyword_matches > 0:
                keyword_confidence = min(keyword_matches * 0.2, 0.5)
                confidence += keyword_confidence
                evidence.append(f"Found {keyword_matches} keyword matches")

            # Check for structural elements
            for structure in pattern_def['structure']:
                if self._has_structure(tree, structure):
                    confidence += 0.2
                    evidence.append(f"Found structural element: {structure}")

            if confidence > 0.4:
                patterns.append(DetectedPattern(
                    pattern_type='design',
                    pattern_name=pattern_name,
                    confidence=min(confidence, 1.0),
                    location=self._find_pattern_location(tree, pattern_name),
                    evidence=evidence,
                    category='design',
                    recommendations=self._get_design_pattern_recommendations(pattern_name)
                ))

        return patterns

    def _detect_architectural_patterns(self, content: str, tree: ast.AST) -> List[DetectedPattern]:
        """Detect architectural patterns"""
        patterns = []

        for pattern_name, pattern_def in self.architectural_patterns.items():
            confidence = 0.0
            evidence = []

            # Check for indicators
            indicator_matches = 0
            for indicator in pattern_def['indicators']:
                if re.search(indicator, content, re.IGNORECASE):
                    indicator_matches += 1

            if indicator_matches > 0:
                confidence = min(indicator_matches * 0.25, 1.0)
                evidence.append(f"Found {indicator_matches} architectural indicators")

            # Check for component patterns
            if 'components' in pattern_def:
                component_matches = 0
                for component in pattern_def['components']:
                    if component.lower() in content.lower():
                        component_matches += 1

                if component_matches > 0:
                    confidence += min(component_matches * 0.2, 0.5)
                    evidence.append(f"Found {component_matches} component matches")

            if confidence > 0.3:
                patterns.append(DetectedPattern(
                    pattern_type='architectural',
                    pattern_name=pattern_name,
                    confidence=confidence,
                    location='module',
                    evidence=evidence,
                    category='architectural',
                    recommendations=self._get_architectural_pattern_recommendations(pattern_name)
                ))

        return patterns

    def _detect_coding_patterns(self, content: str, tree: ast.AST) -> List[DetectedPattern]:
        """Detect coding style patterns"""
        patterns = []

        for pattern_name, pattern_def in self.coding_patterns.items():
            confidence = 0.0
            evidence = []

            # Check for indicators
            indicator_matches = 0
            for indicator in pattern_def['indicators']:
                matches = len(re.findall(indicator, content, re.MULTILINE))
                indicator_matches += matches

            if indicator_matches > 0:
                confidence = min(indicator_matches * 0.1, 1.0)
                evidence.append(f"Found {indicator_matches} pattern indicators")

            # Check for keywords
            keyword_matches = 0
            for keyword in pattern_def['keywords']:
                if keyword.lower() in content.lower():
                    keyword_matches += 1

            if keyword_matches > 0:
                confidence += min(keyword_matches * 0.15, 0.5)
                evidence.append(f"Found {keyword_matches} keyword matches")

            if confidence > 0.2:
                patterns.append(DetectedPattern(
                    pattern_type='coding',
                    pattern_name=pattern_name,
                    confidence=confidence,
                    location='module',
                    evidence=evidence,
                    category='coding',
                    recommendations=self._get_coding_pattern_recommendations(pattern_name)
                ))

        return patterns

    def _detect_anti_patterns(self, content: str, tree: ast.AST) -> List[DetectedPattern]:
        """Detect anti-patterns in the code"""
        patterns = []

        # Check for god object anti-pattern
        if self.anti_patterns['god_object']['threshold']:
            lines = content.split('\n')
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_start = node.lineno - 1
                    class_end = self._find_class_end(lines, class_start)
                    class_size = class_end - class_start + 1

                    if class_size > self.anti_patterns['god_object']['threshold']:
                        patterns.append(DetectedPattern(
                            pattern_type='anti-pattern',
                            pattern_name='god_object',
                            confidence=0.8,
                            location=node.name,
                            evidence=[f"Class has {class_size} lines (threshold: {self.anti_patterns['god_object']['threshold']})"],
                            category='anti-pattern',
                            recommendations=[
                                "Break down large class into smaller, focused classes",
                                "Extract related methods into separate classes",
                                "Consider using composition over inheritance"
                            ]
                        ))

        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'body') and len(node.body) > self.anti_patterns['long_functions']['threshold']:
                    patterns.append(DetectedPattern(
                        pattern_type='anti-pattern',
                        pattern_name='long_function',
                        confidence=0.7,
                        location=node.name,
                        evidence=[f"Function has {len(node.body)} statements (threshold: {self.anti_patterns['long_functions']['threshold']})"],
                        category='anti-pattern',
                        recommendations=[
                            "Break down function into smaller, focused functions",
                            "Extract complex logic into separate methods",
                            "Consider using helper functions"
                        ]
                    ))

        # Check for magic numbers
        magic_numbers = re.findall(r'\b\d{3,}\b', content)
        if len(magic_numbers) > 5:
            patterns.append(DetectedPattern(
                pattern_type='anti-pattern',
                pattern_name='magic_numbers',
                confidence=0.6,
                location='module',
                evidence=[f"Found {len(magic_numbers)} potential magic numbers"],
                category='anti-pattern',
                recommendations=[
                    "Replace magic numbers with named constants",
                    "Use configuration files for configurable values",
                    "Add comments explaining the meaning of numbers"
                ]
            ))

        return patterns

    def _find_pattern_location(self, tree: ast.AST, pattern_name: str) -> str:
        """Find the location where a pattern was detected"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if pattern_name.lower() in node.name.lower():
                    return node.name
            elif isinstance(node, ast.FunctionDef):
                if pattern_name.lower() in node.name.lower():
                    return node.name

        return 'module'

    def _find_class_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a class"""
        indent_level = None
        end_line = start_line

        for i in range(start_line, len(lines)):
            line = lines[i].rstrip()
            if not line:
                continue

            current_indent = len(line) - len(line.lstrip())

            if indent_level is None:
                indent_level = current_indent
            elif current_indent <= indent_level and line.lstrip():
                break

            end_line = i

        return end_line

    def _has_structure(self, tree: ast.AST, structure: str) -> bool:
        """Check if the code has a specific structural element"""
        for node in ast.walk(tree):
            if structure == 'class' and isinstance(node, ast.ClassDef):
                return True
            elif structure == 'class_method' and isinstance(node, ast.FunctionDef):
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                       if hasattr(parent, 'body') and node in parent.body):
                    return True
            elif structure == 'decorator_function' and isinstance(node, ast.FunctionDef):
                if any(isinstance(item, ast.Name) and item.id == node.name
                      for item in ast.walk(tree) if isinstance(item, ast.Name)):
                    return True

        return False

    def _categorize_patterns(self, patterns: List[DetectedPattern]) -> Dict[str, List[str]]:
        """Categorize patterns by type"""
        categories = defaultdict(list)

        for pattern in patterns:
            categories[pattern.category].append(pattern.pattern_name)

        return dict(categories)

    def _determine_architectural_style(self, patterns: List[DetectedPattern]) -> str:
        """Determine the overall architectural style"""
        # Find architectural patterns (replacing complex comprehension with explicit loop)
        architectural_patterns = []
        for p in patterns:
            if p.category == 'architectural':
                architectural_patterns.append(p)

        if not architectural_patterns:
            return 'unknown'

        # Find the most confident architectural pattern
        best_pattern = max(architectural_patterns, key=lambda p: p.confidence)

        if best_pattern.confidence > 0.6:
            return best_pattern.pattern_name
        else:
            return 'mixed'

    def _determine_coding_style(self, patterns: List[DetectedPattern]) -> str:
        """Determine the overall coding style"""
        # Find coding patterns (replacing complex comprehension with explicit loop)
        coding_patterns = []
        for p in patterns:
            if p.category == 'coding':
                coding_patterns.append(p)

        if not coding_patterns:
            return 'unknown'

        # Find the most confident coding pattern
        best_pattern = max(coding_patterns, key=lambda p: p.confidence)

        if best_pattern.confidence > 0.5:
            return best_pattern.pattern_name
        else:
            return 'mixed'

    def _calculate_confidence_distribution(self, patterns: List[DetectedPattern]) -> Dict[str, int]:
        """Calculate confidence distribution of patterns"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}

        for pattern in patterns:
            if pattern.confidence > 0.8:
                distribution['high'] += 1
            elif pattern.confidence > 0.5:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1

        return distribution

    def _generate_pattern_recommendations(self, patterns: List[DetectedPattern],
                                        architectural_style: str,
                                        coding_style: str) -> List[str]:
        """Generate recommendations based on detected patterns"""
        recommendations = []

        # Anti-pattern recommendations (replacing complex comprehension with explicit loop)
        anti_patterns = []
        for p in patterns:
            if p.category == 'anti-pattern':
                anti_patterns.append(p)
        for pattern in anti_patterns:
            recommendations.extend(pattern.recommendations)

        # Architectural recommendations
        if architectural_style == 'unknown':
            recommendations.append("Consider adopting a clear architectural pattern")
        elif architectural_style == 'mixed':
            recommendations.append("Consolidate architectural patterns for consistency")

        # Design pattern recommendations (replacing complex comprehension with explicit loop)
        design_patterns = []
        for p in patterns:
            if p.category == 'design':
                design_patterns.append(p)
        if len(design_patterns) > 3:
            recommendations.append("Many design patterns detected - ensure they are necessary")
        elif len(design_patterns) == 0:
            recommendations.append("Consider using appropriate design patterns where beneficial")

        # Coding style recommendations
        if coding_style == 'mixed':
            recommendations.append("Consider standardizing on a primary coding style")

        if not recommendations:
            recommendations.append("Pattern analysis looks good - maintain current approach")

        return recommendations

    def _get_design_pattern_recommendations(self, pattern_name: str) -> List[str]:
        """Get recommendations for specific design patterns"""
        recommendations = {
            'singleton': [
                "Ensure thread safety if used in concurrent environments",
                "Consider dependency injection as an alternative",
                "Document the singleton behavior clearly"
            ],
            'factory': [
                "Ensure factory methods have clear, consistent interfaces",
                "Consider abstract factory for complex object creation",
                "Document the creation logic clearly"
            ],
            'observer': [
                "Ensure proper cleanup of observer references",
                "Consider weak references to prevent memory leaks",
                "Document the notification flow clearly"
            ]
        }

        return recommendations.get(pattern_name, ["Follow best practices for this design pattern"])

    def _get_architectural_pattern_recommendations(self, pattern_name: str) -> List[str]:
        """Get recommendations for architectural patterns"""
        recommendations = {
            'mvc': [
                "Ensure clear separation between model, view, and controller",
                "Use appropriate data binding mechanisms",
                "Consider MVVM for complex UIs"
            ],
            'layered': [
                "Maintain clear boundaries between layers",
                "Use dependency injection for loose coupling",
                "Consider hexagonal architecture for better testability"
            ],
            'repository': [
                "Ensure consistent interface across all repositories",
                "Use unit of work pattern for transaction management",
                "Consider CQRS for complex query scenarios"
            ]
        }

        return recommendations.get(pattern_name, ["Follow best practices for this architectural pattern"])

    def _get_coding_pattern_recommendations(self, pattern_name: str) -> List[str]:
        """Get recommendations for coding patterns"""
        recommendations = {
            'functional': [
                "Ensure functional code remains readable and maintainable",
                "Consider performance implications of functional constructs",
                "Use type hints for better IDE support"
            ],
            'object_oriented': [
                "Follow SOLID principles for maintainable OO code",
                "Consider composition over inheritance where appropriate",
                "Use appropriate design patterns"
            ],
            'async_await': [
                "Ensure proper error handling in async code",
                "Use appropriate async libraries and patterns",
                "Consider performance implications"
            ]
        }

        return recommendations.get(pattern_name, ["Follow best practices for this coding pattern"])

    def _fallback_pattern_detection(self, content: str, file_path: Optional[Path],
                                  error: SyntaxError) -> Dict[str, Any]:
        """Fallback pattern detection for files with syntax errors"""
        return {
            'patterns': [],
            'pattern_categories': {},
            'confidence_distribution': {},
            'architectural_style': 'unknown',
            'coding_style': 'unknown',
            'recommendations': [f'Contains syntax error: {error}'],
            'file_path': str(file_path) if file_path else 'unknown',
            'total_patterns': 0,
            'high_confidence_patterns': 0
        }


# Module-level functions for easy integration
def detect_patterns(content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Module-level function for pattern detection"""
    detector = PatternDetector()
    return detector.detect_patterns(content, file_path)


def get_pattern_confidence(content: str) -> float:
    """Get overall pattern detection confidence"""
    detector = PatternDetector()
    result = detector.detect_patterns(content)
    return result.get('total_patterns', 0) / 10.0  # Normalize


def identify_architectural_style(content: str) -> str:
    """Identify the architectural style of the code"""
    detector = PatternDetector()
    result = detector.detect_patterns(content)
    return result.get('architectural_style', 'unknown')


if __name__ == "__main__":
    # Example usage
    sample_code = """
import asyncio
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

class UserRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def save(self, user: User) -> bool:
        # Save user to database
        return True

    def find_by_id(self, user_id: int) -> User:
        # Find user by ID
        return User(id=user_id, name="John", email="john@example.com")

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def create_user(self, name: str, email: str) -> User:
        user = User(id=123, name=name, email=email)
        self.repository.save(user)
        return user

    async def get_user_async(self, user_id: int) -> User:
        # Simulate async operation
        await asyncio.sleep(0.1)
        return self.repository.find_by_id(user_id)

# Singleton pattern for configuration
class Config:
    _instance = None

    def __init__(self):
        if Config._instance is not None:
            raise Exception("Config is a singleton class")
        self.settings = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

def main():
    config = Config.get_instance()
    repo = UserRepository(None)
    service = UserService(repo)

    user = service.create_user("Alice", "alice@example.com")
    print(f"Created user: {user}")
"""

    detector = PatternDetector()
    result = detector.detect_patterns(sample_code, Path("sample.py"))

    print("Pattern Detection Results:")
    print(f"Total Patterns: {result['total_patterns']}")
    print(f"Architectural Style: {result['architectural_style']}")
    print(f"Coding Style: {result['coding_style']}")
    print(f"High Confidence Patterns: {result['high_confidence_patterns']}")
    print(f"Pattern Categories: {result['pattern_categories']}")
    print(f"Recommendations: {result['recommendations'][:3]}")
