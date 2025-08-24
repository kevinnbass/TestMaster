#!/usr/bin/env python3
"""
Meta-Reorganizer Analysis Module
================================

Analysis functionality for the intelligence-driven reorganizer system.
Contains all analysis methods for semantic, relationship, pattern, ML, and dependency analysis.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


class IntelligenceAnalysisEngine:
    """Handles all intelligence-based analysis operations"""

    def __init__(self, intelligence_modules: Dict[str, Path], logger):
        """Initialize the analysis engine with available intelligence modules"""
        self.intelligence_modules = intelligence_modules
        self.logger = logger

    def run_semantic_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use your semantic analyzer to understand the code's meaning"""
        try:
            if "semantic" in self.intelligence_modules:
                semantic_module = self.intelligence_modules["semantic"]

                # Try to import and use your semantic analyzer
                spec = importlib.util.spec_from_file_location("semantic_analyzer", semantic_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try to call your semantic analysis function
                    if hasattr(module, 'analyze_semantics'):
                        return module.analyze_semantics(content)
                    elif hasattr(module, 'semantic_analysis'):
                        return module.semantic_analysis(file_path)

        except Exception as e:
            self.logger.debug(f"Could not use semantic analyzer: {e}")

        # Fallback: basic semantic analysis
        return self._basic_semantic_analysis(content)

    def run_relationship_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use your relationship analyzer to find module connections"""
        try:
            if "relationship" in self.intelligence_modules:
                relationship_module = self.intelligence_modules["relationship"]

                spec = importlib.util.spec_from_file_location("relationship_analyzer", relationship_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, 'analyze_relationships'):
                        return module.analyze_relationships(content, str(file_path))

        except Exception as e:
            self.logger.debug(f"Could not use relationship analyzer: {e}")

        return self._basic_relationship_analysis(file_path, content)

    def run_pattern_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use your pattern detector to identify code patterns"""
        try:
            if "pattern" in self.intelligence_modules:
                pattern_module = self.intelligence_modules["pattern"]

                spec = importlib.util.spec_from_file_location("pattern_detector", pattern_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, 'detect_patterns'):
                        return module.detect_patterns(content)

        except Exception as e:
            self.logger.debug(f"Could not use pattern detector: {e}")

        return self._basic_pattern_analysis(content)

    def run_ml_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use your ML analyzer for code classification"""
        try:
            if "ml" in self.intelligence_modules:
                ml_module = self.intelligence_modules["ml"]

                spec = importlib.util.spec_from_file_location("ml_analyzer", ml_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, 'analyze_code'):
                        return module.analyze_code(content)
                    elif hasattr(module, 'classify_code'):
                        return module.classify_code(content)

        except Exception as e:
            self.logger.debug(f"Could not use ML analyzer: {e}")

        return self._basic_ml_analysis(content)

    def run_dependency_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use your dependency resolver to understand module dependencies"""
        try:
            if "dependency" in self.intelligence_modules:
                dependency_module = self.intelligence_modules["dependency"]

                spec = importlib.util.spec_from_file_location("dependency_resolver", dependency_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, 'resolve_dependencies'):
                        return module.resolve_dependencies(content, str(file_path))

        except Exception as e:
            self.logger.debug(f"Could not use dependency resolver: {e}")

        return self._basic_dependency_analysis(file_path, content)

    def combine_analyses(self, semantic: Dict, relationship: Dict,
                        pattern: Dict, ml: Dict, dependency: Dict,
                        file_path: Path, content: str) -> Tuple[str, float, List[str]]:
        """Combine all analyses to determine the best category"""

        # Initialize scores and reasoning
        category_scores, reasoning = self._init_analysis_scores()

        # Score each analysis type
        self._score_analysis_types(category_scores, reasoning, semantic, relationship,
                                  pattern, ml, dependency)

        # Find the best category
        if category_scores:
            best_category, confidence = self._find_best_category(category_scores, reasoning)
            return best_category, confidence, reasoning

        # Fallback to basic analysis
        return self._fallback_categorization(file_path, content, reasoning)

    def _init_analysis_scores(self) -> Tuple[Dict[str, float], List[str]]:
        """Initialize category scores and reasoning list"""
        # Pre-allocate with safety bound (Rule 3 compliance)
        MAX_REASONING_ITEMS = 10  # Safety bound for reasoning list
        reasoning = [None] * MAX_REASONING_ITEMS
        reasoning_count = 0
        category_scores = {
            'core/intelligence': 0,
            'core/orchestration': 0,
            'core/security': 0,
            'core/foundation': 0,
            'security': 0,
            'testing': 0,
            'monitoring': 0,
            'deployment': 0,
            'documentation': 0,
            'configuration': 0,
            'utilities': 0
        }
        return category_scores, reasoning

    def _score_analysis_types(self, category_scores: Dict[str, float], reasoning: List[str],
                             semantic: Dict, relationship: Dict, pattern: Dict,
                             ml: Dict, dependency: Dict) -> None:
        """Score each analysis type and update category scores"""

        # Semantic analysis scoring
        if semantic.get('category'):
            category_scores[semantic['category']] += semantic.get('confidence', 0.5)
            reasoning.append(f"Semantic analysis suggests: {semantic['category']}")

        # ML analysis scoring
        if ml.get('predicted_category'):
            category_scores[ml['predicted_category']] += ml.get('confidence', 0.4)
            reasoning.append(f"ML analysis predicts: {ml['predicted_category']}")

        # Pattern analysis scoring
        if pattern.get('dominant_pattern'):
            pattern_category = self._map_pattern_to_category(pattern['dominant_pattern'])
            category_scores[pattern_category] += pattern.get('confidence', 0.3)
            reasoning.append(f"Pattern analysis: {pattern['dominant_pattern']} → {pattern_category}")

        # Dependency analysis scoring
        if dependency.get('primary_domain'):
            dep_category = self._map_domain_to_category(dependency['primary_domain'])
            category_scores[dep_category] += dependency.get('confidence', 0.2)
            reasoning.append(f"Dependency analysis: {dependency['primary_domain']} → {dep_category}")

        # Relationship analysis scoring
        if relationship.get('primary_relationship'):
            rel_category = self._map_relationship_to_category(relationship['primary_relationship'])
            category_scores[rel_category] += relationship.get('strength', 0.1)
            reasoning.append(f"Relationship analysis: {relationship['primary_relationship']} → {rel_category}")

    def _find_best_category(self, category_scores: Dict[str, float], reasoning: List[str]) -> Tuple[str, float]:
        """Find the best category and calculate confidence"""
        best_category = max(category_scores.items(), key=lambda x: x[1])
        confidence = min(best_category[1], 1.0)

        # Boost confidence if multiple analyses agree
        agreeing_analyses = sum(1 for score in category_scores.values() if score > 0.3)
        if agreeing_analyses >= 2:
            confidence = min(confidence + 0.2, 1.0)
            reasoning.append(f"Multiple analyses agree ({agreeing_analyses} total)")

        return best_category[0], confidence

    def _map_pattern_to_category(self, pattern: str) -> str:
        """Map detected patterns to categories"""
        pattern_lower = pattern.lower()

        if any(p in pattern_lower for p in ['intelligence', 'ml', 'ai', 'neural', 'predictive']):
            return 'core/intelligence'
        elif any(p in pattern_lower for p in ['orchestrator', 'coordinator', 'workflow', 'agent']):
            return 'core/orchestration'
        elif any(p in pattern_lower for p in ['security', 'auth', 'encrypt', 'vulnerability']):
            return 'core/security'
        elif any(p in pattern_lower for p in ['monitor', 'dashboard', 'metric', 'alert']):
            return 'monitoring'
        elif any(p in pattern_lower for p in ['test', 'spec', 'mock', 'fixture']):
            return 'testing'

        return 'utilities'

    def _map_domain_to_category(self, domain: str) -> str:
        """Map domains to categories"""
        domain_lower = domain.lower()

        if any(d in domain_lower for d in ['intelligence', 'ml', 'ai', 'learning']):
            return 'core/intelligence'
        elif any(d in domain_lower for d in ['orchestration', 'coordination', 'workflow']):
            return 'core/orchestration'
        elif any(d in domain_lower for d in ['security', 'auth', 'encryption']):
            return 'core/security'
        elif any(d in domain_lower for d in ['monitoring', 'observability']):
            return 'monitoring'

        return 'utilities'

    def _map_relationship_to_category(self, relationship: str) -> str:
        """Map relationships to categories"""
        rel_lower = relationship.lower()

        if 'intelligence' in rel_lower or 'ml' in rel_lower:
            return 'core/intelligence'
        elif 'orchestration' in rel_lower or 'workflow' in rel_lower:
            return 'core/orchestration'
        elif 'security' in rel_lower or 'auth' in rel_lower:
            return 'core/security'
        elif 'monitoring' in rel_lower or 'dashboard' in rel_lower:
            return 'monitoring'

        return 'utilities'

    def _basic_semantic_analysis(self, content: str) -> Dict[str, Any]:
        """Basic semantic analysis as fallback"""
        # Simple keyword-based semantic analysis
        intelligence_keywords = ['intelligence', 'ml', 'ai', 'neural', 'predictive', 'learning']
        orchestration_keywords = ['orchestrator', 'coordinator', 'workflow', 'agent', 'scheduler']
        security_keywords = ['security', 'auth', 'encrypt', 'decrypt', 'vulnerability', 'threat']
        monitoring_keywords = ['monitor', 'dashboard', 'metric', 'alert', 'log', 'telemetry']

        content_lower = content.lower()

        scores = {
            'core/intelligence': sum(1 for kw in intelligence_keywords if kw in content_lower),
            'core/orchestration': sum(1 for kw in orchestration_keywords if kw in content_lower),
            'core/security': sum(1 for kw in security_keywords if kw in content_lower),
            'monitoring': sum(1 for kw in monitoring_keywords if kw in content_lower)
        }

        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return {'category': best[0], 'confidence': min(best[1] * 0.1, 0.8)}

        return {'category': 'utilities', 'confidence': 0.1}

    def _basic_relationship_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Basic relationship analysis"""
        # Look for imports and references to other modules
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from ') or line.startswith('import '):
                # Extract module name
                if ' from ' in line:
                    module = line.split(' from ')[1].split()[0]
                else:
                    module = line.split('import ')[1].split()[0]
                imports.append(module)

        if imports:
            return {
                'primary_relationship': 'has_dependencies',
                'strength': min(len(imports) * 0.1, 0.8),
                'imports': imports
            }

        return {'primary_relationship': 'isolated', 'strength': 0.1}

    def _basic_pattern_analysis(self, content: str) -> Dict[str, Any]:
        """Basic pattern analysis"""
        patterns = []

        if 'class ' in content:
            patterns.append('class_definition')
        if 'def ' in content:
            patterns.append('function_definition')
        if '__init__' in content:
            patterns.append('constructor')
        if 'async def' in content:
            patterns.append('async_function')
        if 'import ' in content or 'from ' in content:
            patterns.append('imports_modules')
        if 'try:' in content or 'except:' in content:
            patterns.append('error_handling')
        if 'for ' in content or 'while ' in content:
            patterns.append('loops')
        if 'if ' in content:
            patterns.append('conditional_logic')

        if patterns:
            return {
                'dominant_pattern': max(set(patterns), key=patterns.count),
                'all_patterns': patterns,
                'confidence': min(len(patterns) * 0.05, 0.7)
            }

        return {'dominant_pattern': 'generic_code', 'confidence': 0.1}

    def _basic_ml_analysis(self, content: str) -> Dict[str, Any]:
        """Basic ML-based analysis"""
        # Simple ML-style classification based on keywords
        ml_keywords = ['model', 'train', 'predict', 'accuracy', 'loss', 'epoch', 'tensor', 'neural']
        orchestration_keywords = ['orchestrate', 'coordinate', 'workflow', 'pipeline', 'task']
        security_keywords = ['encrypt', 'decrypt', 'hash', 'token', 'auth', 'secure']

        content_lower = content.lower()

        scores = {
            'core/intelligence': sum(1 for kw in ml_keywords if kw in content_lower),
            'core/orchestration': sum(1 for kw in orchestration_keywords if kw in content_lower),
            'core/security': sum(1 for kw in security_keywords if kw in content_lower)
        }

        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return {
                'predicted_category': best[0],
                'confidence': min(best[1] * 0.15, 0.8),
                'features_used': [kw for kw in eval(f"{best[0].split('/')[-1]}_keywords") if kw in content_lower]
            }

        return {'predicted_category': 'utilities', 'confidence': 0.1}

    def _basic_dependency_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Basic dependency analysis"""
        # Analyze what the module depends on and what depends on it
        domains = []

        if any(ext in content.lower() for ext in ['.py', 'import', 'def ', 'class ']):
            domains.append('python_development')
        if any(ext in content.lower() for ext in ['intelligence', 'ml', 'ai']):
            domains.append('machine_learning')
        if any(ext in content.lower() for ext in ['security', 'auth', 'encrypt']):
            domains.append('security')
        if any(ext in content.lower() for ext in ['monitor', 'log', 'metric']):
            domains.append('monitoring')
        if any(ext in content.lower() for ext in ['orchestrat', 'coordinat', 'workflow']):
            domains.append('orchestration')

        if domains:
            return {
                'primary_domain': max(set(domains), key=domains.count),
                'all_domains': domains,
                'confidence': min(len(domains) * 0.1, 0.7)
            }

        return {'primary_domain': 'general_utility', 'confidence': 0.1}

    def _fallback_categorization(self, file_path: Path, content: str, existing_reasoning: List[str]) -> Tuple[str, float, List[str]]:
        """Fallback categorization when all else fails"""
        reasoning = existing_reasoning + ["Using fallback categorization"]

        # Check file name and path for clues
        file_name = file_path.name.lower()
        file_stem = file_path.stem.lower()

        if 'intelligence' in file_name or 'ml' in file_name or 'ai' in file_name:
            reasoning.append("File name suggests intelligence/ML")
            return 'core/intelligence', 0.6, reasoning
        elif 'orchestrat' in file_name or 'coordinat' in file_name or 'agent' in file_name:
            reasoning.append("File name suggests orchestration")
            return 'core/orchestration', 0.6, reasoning
        elif 'security' in file_name or 'auth' in file_name:
            reasoning.append("File name suggests security")
            return 'core/security', 0.6, reasoning
        elif 'monitor' in file_name or 'dashboard' in file_name:
            reasoning.append("File name suggests monitoring")
            return 'monitoring', 0.6, reasoning
        elif 'test' in file_name:
            reasoning.append("File name suggests testing")
            return 'testing', 0.6, reasoning

        # Default fallback
        reasoning.append("No specific category detected, using utilities")
        return 'utilities', 0.3, reasoning
