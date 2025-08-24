#!/usr/bin/env python3
"""
Pattern Core Detector
====================

Core pattern detection functionality.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict

from pattern_data import (
    DetectedPattern, PatternAnalysis, PatternDefinition,
    PatternDetectionResult
)


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
        """Load all coding style pattern definitions"""
        return self._load_basic_coding_patterns()

    def _load_pattern_definitions(self) -> None:
        """Load all pattern definitions"""
        self.design_patterns = self._load_design_patterns()
        self.architectural_patterns = self._load_architectural_patterns()
        self.coding_patterns = self._load_coding_patterns()

        # Combine all patterns
        self.all_patterns = {}
        self.all_patterns.update(self.design_patterns)
        self.all_patterns.update(self.architectural_patterns)
        self.all_patterns.update(self.coding_patterns)

    def detect_patterns(self, content: str, file_path: Optional[Path] = None) -> PatternAnalysis:
        """
        Perform comprehensive pattern analysis of code content.

        Args:
            content: The code content to analyze
            file_path: Path to the file for context

        Returns:
            PatternAnalysis containing all detected patterns and insights
        """
        try:
            # Parse the code into AST
            tree = ast.parse(content)

            # Detect different types of patterns
            detected_patterns = []
            detected_patterns.extend(self._detect_design_patterns(content, tree))
            detected_patterns.extend(self._detect_architectural_patterns(content, tree))
            detected_patterns.extend(self._detect_coding_patterns(content, tree))

            # Analyze pattern distribution and confidence
            pattern_categories = self._categorize_patterns(detected_patterns)
            confidence_distribution = self._analyze_confidence_distribution(detected_patterns)

            # Determine architectural and coding styles
            architectural_style = self._determine_architectural_style(detected_patterns)
            coding_style = self._determine_coding_style(detected_patterns)

            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(detected_patterns, pattern_categories)

            return PatternAnalysis(
                patterns=detected_patterns,
                pattern_categories=pattern_categories,
                confidence_distribution=confidence_distribution,
                architectural_style=architectural_style,
                coding_style=coding_style,
                recommendations=recommendations
            )

        except SyntaxError as e:
            return PatternAnalysis(
                patterns=[],
                pattern_categories={},
                confidence_distribution={},
                architectural_style="unknown",
                coding_style="unknown",
                recommendations=[f"Unable to analyze patterns due to syntax error: {e}"]
            )

    def _detect_design_patterns(self, content: str, tree: ast.AST) -> List[DetectedPattern]:
        """Detect design patterns in the code"""
        detected = []

        for pattern_name, pattern_def in self.design_patterns.items():
            confidence = self._calculate_pattern_confidence(content, pattern_def)

            if confidence > 0.3:  # Minimum confidence threshold
                category = self._get_pattern_category(pattern_name)
                location = self._find_pattern_location(content, pattern_name)
                evidence = self._collect_pattern_evidence(content, pattern_def)
                recommendations = self._get_pattern_recommendations(pattern_name, confidence)

                detected.append(DetectedPattern(
                    pattern_type="design",
                    pattern_name=pattern_name,
                    confidence=confidence,
                    location=location,
                    evidence=evidence,
                    category=category,
                    recommendations=recommendations
                ))

        return detected

    def _detect_architectural_patterns(self, content: str, tree: ast.AST) -> List[DetectedPattern]:
        """Detect architectural patterns in the code"""
        detected = []

        for pattern_name, pattern_def in self.architectural_patterns.items():
            confidence = self._calculate_pattern_confidence(content, pattern_def)

            if confidence > 0.4:  # Higher threshold for architectural patterns
                category = self._get_pattern_category(pattern_name)
                location = self._find_pattern_location(content, pattern_name)
                evidence = self._collect_pattern_evidence(content, pattern_def)
                recommendations = self._get_pattern_recommendations(pattern_name, confidence)

                detected.append(DetectedPattern(
                    pattern_type="architectural",
                    pattern_name=pattern_name,
                    confidence=confidence,
                    location=location,
                    evidence=evidence,
                    category=category,
                    recommendations=recommendations
                ))

        return detected

    def _detect_coding_patterns(self, content: str, tree: ast.AST) -> List[DetectedPattern]:
        """Detect coding style patterns in the code"""
        detected = []

        for pattern_name, pattern_def in self.coding_patterns.items():
            confidence = self._calculate_pattern_confidence(content, pattern_def)

            if confidence > 0.5:  # Highest threshold for coding patterns
                category = "coding"
                location = "module"
                evidence = self._collect_pattern_evidence(content, pattern_def)
                recommendations = self._get_pattern_recommendations(pattern_name, confidence)

                detected.append(DetectedPattern(
                    pattern_type="coding",
                    pattern_name=pattern_name,
                    confidence=confidence,
                    location=location,
                    evidence=evidence,
                    category=category,
                    recommendations=recommendations
                ))

        return detected

    def _calculate_pattern_confidence(self, content: str, pattern_def: Dict) -> float:
        """Calculate confidence score for a pattern detection"""
        confidence = 0.0
        indicators = pattern_def.get('indicators', [])
        keywords = pattern_def.get('keywords', [])

        # Check for indicator patterns
        for indicator in indicators:
            if re.search(indicator, content, re.IGNORECASE | re.MULTILINE):
                confidence += 0.3

        # Check for keywords
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                confidence += 0.2

        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)

        return confidence

    def _get_pattern_category(self, pattern_name: str) -> str:
        """Get the category of a pattern"""
        if pattern_name in self.design_patterns:
            return "design"
        elif pattern_name in self.architectural_patterns:
            return "architectural"
        else:
            return "coding"

    def _find_pattern_location(self, content: str, pattern_name: str) -> str:
        """Find where a pattern is located in the code"""
        lines = content.split('\n')

        # Look for class or function definitions that might implement the pattern
        for i, line in enumerate(lines):
            if 'class' in line.lower() and pattern_name.lower() in line.lower():
                return f"class definition at line {i+1}"
            elif 'def' in line.lower() and any(keyword in line.lower() for keyword in ['create', 'build', 'factory']):
                return f"function definition at line {i+1}"

        return "module"

    def _collect_pattern_evidence(self, content: str, pattern_def: Dict) -> List[str]:
        """Collect evidence for pattern detection"""
        evidence = []
        indicators = pattern_def.get('indicators', [])
        keywords = pattern_def.get('keywords', [])

        for indicator in indicators:
            if re.search(indicator, content, re.IGNORECASE | re.MULTILINE):
                evidence.append(f"Found pattern indicator: {indicator}")

        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                evidence.append(f"Found keyword: {keyword}")

        return evidence[:5]  # Limit to 5 evidence items

    def _get_pattern_recommendations(self, pattern_name: str, confidence: float) -> List[str]:
        """Generate recommendations for detected patterns"""
        recommendations = []

        if confidence > 0.8:
            recommendations.append(f"Strong {pattern_name} pattern detected - well implemented")
        elif confidence > 0.6:
            recommendations.append(f"Moderate {pattern_name} pattern detected - consider strengthening implementation")
        else:
            recommendations.append(f"Weak {pattern_name} pattern detected - may need improvement")

        return recommendations

    def _categorize_patterns(self, patterns: List[DetectedPattern]) -> Dict[str, List[str]]:
        """Categorize patterns by type"""
        categories = defaultdict(list)

        for pattern in patterns:
            categories[pattern.category].append(pattern.pattern_name)

        return dict(categories)

    def _analyze_confidence_distribution(self, patterns: List[DetectedPattern]) -> Dict[str, int]:
        """Analyze the distribution of pattern confidence levels"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}

        for pattern in patterns:
            if pattern.confidence > 0.7:
                distribution['high'] += 1
            elif pattern.confidence > 0.4:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1

        return distribution

    def _determine_architectural_style(self, patterns: List[DetectedPattern]) -> str:
        """Determine the overall architectural style"""
        architectural_patterns = [p for p in patterns if p.pattern_type == 'architectural']

        if not architectural_patterns:
            return "no clear architectural style"

        # Find the highest confidence architectural pattern
        best_pattern = max(architectural_patterns, key=lambda p: p.confidence)

        if best_pattern.confidence > 0.6:
            return best_pattern.pattern_name.replace('_', ' ')
        else:
            return "mixed architectural styles"

    def _determine_coding_style(self, patterns: List[DetectedPattern]) -> str:
        """Determine the overall coding style"""
        coding_patterns = [p for p in patterns if p.pattern_type == 'coding']

        if not coding_patterns:
            return "no clear coding style"

        # Find the highest confidence coding pattern
        best_pattern = max(coding_patterns, key=lambda p: p.confidence)

        if best_pattern.confidence > 0.7:
            return best_pattern.pattern_name.replace('_', ' ')
        else:
            return "mixed coding styles"

    def _generate_pattern_recommendations(self, patterns: List[DetectedPattern],
                                        categories: Dict[str, List[str]]) -> List[str]:
        """Generate overall pattern recommendations"""
        recommendations = []

        # Check for missing common patterns
        common_patterns = ['factory', 'singleton', 'observer', 'strategy']
        detected_names = [p.pattern_name for p in patterns]

        missing_patterns = [p for p in common_patterns if p not in detected_names]
        if missing_patterns:
            recommendations.append(f"Consider implementing these common patterns: {', '.join(missing_patterns)}")

        # Check pattern distribution
        if len(patterns) > 10:
            recommendations.append("Many patterns detected - consider simplifying the design")
        elif len(patterns) < 3:
            recommendations.append("Few patterns detected - consider using more design patterns")

        # Check confidence levels
        low_confidence = [p for p in patterns if p.confidence < 0.5]
        if low_confidence:
            recommendations.append(f"{len(low_confidence)} patterns have low confidence - may need better implementation")

        return recommendations

