"""
Language Bridge - Natural Language Code Translation and Documentation Engine
============================================================================

Advanced natural language bridge for bidirectional code-language translation,
intelligent documentation generation, and context-aware code explanation.
Implements enterprise-grade language processing for code intelligence systems.

This module provides comprehensive natural language processing capabilities for
code understanding, explanation generation, and intelligent documentation creation.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: language_bridge.py (450 lines)
"""

import asyncio
import logging
import ast
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from .predictive_types import (
    LanguageBridgeDirection, DocumentationType, CodeComplexityLevel,
    NaturalLanguageTranslation, DocumentationGeneration
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaturalLanguageBridge:
    """Enterprise natural language bridge for code-language translation"""
    
    def __init__(self):
        self.code_patterns = {
            'conditional': r'if\s+.*:',
            'loop': r'(for|while)\s+.*:',
            'function_def': r'def\s+\w+\s*\(',
            'class_def': r'class\s+\w+\s*[\(:]',
            'assignment': r'\w+\s*=\s*',
            'return': r'return\s+',
            'import': r'(import|from)\s+\w+',
            'async_pattern': r'async\s+(def|with|for)',
            'exception_handling': r'(try|except|finally|raise)\s*:?',
            'decorator': r'@\w+.*'
        }
        
        self.explanation_templates = {
            'function': "This function {name} {purpose}. It takes {parameters} and {return_behavior}.",
            'class': "This class {name} represents {purpose}. It provides {method_count} methods for {functionality}.",
            'conditional': "This conditional statement checks if {condition} and {action}.",
            'loop': "This loop iterates {iteration_desc} and {action}.",
            'assignment': "This assigns {value} to the variable {variable}.",
            'async_function': "This asynchronous function {name} {purpose} using async/await pattern.",
            'decorator': "This decorator {name} modifies the behavior of {target}."
        }
        
        self.technical_glossary = {
            'async': 'asynchronous programming',
            'await': 'asynchronous wait operation',
            'yield': 'generator function',
            'lambda': 'anonymous function expression',
            'decorator': 'function modifier',
            'comprehension': 'compact iteration syntax',
            'context_manager': 'resource management pattern',
            'metaclass': 'class creation controller',
            'closure': 'function with enclosed scope',
            'coroutine': 'cooperative function'
        }
    
    def translate_code_to_language(self, code: str, target_audience: str = "general",
                                 abstraction_level: str = "medium") -> NaturalLanguageTranslation:
        """Translate code to comprehensive natural language explanation"""
        try:
            translation = NaturalLanguageTranslation(
                direction=LanguageBridgeDirection.CODE_TO_LANGUAGE,
                source_code=code,
                target_audience=target_audience,
                abstraction_level=abstraction_level
            )
            
            # Parse code with error handling
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                translation.natural_language = f"Unable to parse code due to syntax error: {str(e)}"
                translation.translation_quality = 0.0
                return translation
            
            # Generate comprehensive explanation
            explanation_parts = []
            
            # Analyze and explain main components
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # Class explanations
            for class_node in classes:
                class_explanation = self._explain_class(class_node, abstraction_level)
                explanation_parts.append(class_explanation)
            
            # Function explanations (exclude methods already counted in classes)
            for func_node in functions:
                if not any(func_node in class_node.body for class_node in classes):
                    func_explanation = self._explain_function(func_node, abstraction_level)
                    explanation_parts.append(func_explanation)
            
            # Generate overall explanation if no specific components
            if not explanation_parts:
                translation.natural_language = self._generate_general_explanation(code, tree)
            else:
                translation.natural_language = " ".join(explanation_parts)
            
            # Extract technical terms and enhance explanation
            translation.technical_terms = self._extract_technical_terms(code)
            translation.translation_quality = self._assess_translation_quality(translation)
            translation.context_understanding = self._understand_code_context(tree, code)
            translation.validation_score = self._validate_translation_accuracy(translation, tree)
            
            # Add audience-specific adaptations
            if target_audience == "beginner":
                translation.natural_language = self._adapt_for_beginners(translation.natural_language)
            elif target_audience == "expert":
                translation.natural_language = self._enhance_for_experts(translation.natural_language, tree)
            
            return translation
            
        except Exception as e:
            logger.error(f"Error translating code to language: {e}")
            return NaturalLanguageTranslation(
                direction=LanguageBridgeDirection.CODE_TO_LANGUAGE,
                source_code=code,
                natural_language=f"Translation error: {str(e)}",
                translation_quality=0.0
            )
    
    def translate_language_to_code(self, natural_description: str,
                                 programming_language: str = "python") -> NaturalLanguageTranslation:
        """Translate natural language description to code structure"""
        try:
            translation = NaturalLanguageTranslation(
                direction=LanguageBridgeDirection.LANGUAGE_TO_CODE,
                natural_language=natural_description
            )
            
            # Analyze description for code patterns
            code_elements = self._extract_code_elements_from_description(natural_description)
            
            # Generate code structure
            generated_code = self._generate_code_from_elements(code_elements, programming_language)
            translation.source_code = generated_code
            
            # Assess generation quality
            translation.translation_quality = self._assess_code_generation_quality(
                natural_description, generated_code
            )
            
            return translation
            
        except Exception as e:
            logger.error(f"Error translating language to code: {e}")
            return NaturalLanguageTranslation(
                direction=LanguageBridgeDirection.LANGUAGE_TO_CODE,
                natural_language=natural_description,
                source_code=f"# Translation error: {str(e)}"
            )
    
    def _explain_function(self, node: ast.FunctionDef, abstraction_level: str) -> str:
        """Generate comprehensive natural language explanation for function"""
        try:
            name = node.name
            
            # Analyze function characteristics
            param_count = len(node.args.args)
            is_async = isinstance(node, ast.AsyncFunctionDef) or any(
                isinstance(child, ast.Await) for child in ast.walk(node)
            )
            has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
            has_decorators = bool(node.decorator_list)
            
            # Determine purpose and behavior
            purpose = self._infer_function_purpose(name, node)
            param_desc = self._describe_parameters(node.args, abstraction_level)
            return_desc = "returns a value" if has_return else "performs an action"
            
            # Build explanation based on abstraction level
            if abstraction_level == "high":
                explanation = f"The function '{name}' {purpose}."
            elif abstraction_level == "low":
                details = []
                if is_async:
                    details.append("is asynchronous")
                if has_decorators:
                    details.append(f"uses {len(node.decorator_list)} decorator(s)")
                if param_count > 0:
                    details.append(f"accepts {param_count} parameter(s)")
                
                detail_str = ", ".join(details) if details else "is a simple function"
                explanation = f"The function '{name}' {detail_str}, {purpose}, and {return_desc}."
            else:  # medium
                async_prefix = "asynchronous " if is_async else ""
                explanation = f"The {async_prefix}function '{name}' {purpose} {param_desc} and {return_desc}."
            
            return explanation
                
        except Exception as e:
            logger.error(f"Error explaining function: {e}")
            return f"Function '{node.name}' performs operations."
    
    def _explain_class(self, node: ast.ClassDef, abstraction_level: str) -> str:
        """Generate comprehensive natural language explanation for class"""
        try:
            name = node.name
            
            # Analyze class characteristics
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            method_count = len(methods)
            has_inheritance = bool(node.bases)
            has_decorators = bool(node.decorator_list)
            
            # Categorize methods
            special_methods = [m for m in methods if m.name.startswith('__') and m.name.endswith('__')]
            public_methods = [m for m in methods if not m.name.startswith('_')]
            private_methods = [m for m in methods if m.name.startswith('_') and not m.name.startswith('__')]
            
            # Determine class purpose
            purpose = self._infer_class_purpose(name, methods)
            
            # Build explanation
            if abstraction_level == "high":
                explanation = f"The class '{name}' {purpose}."
            elif abstraction_level == "low":
                details = []
                if has_inheritance:
                    details.append(f"inherits from {len(node.bases)} base class(es)")
                if has_decorators:
                    details.append(f"uses {len(node.decorator_list)} decorator(s)")
                details.append(f"defines {method_count} methods")
                if special_methods:
                    details.append(f"including {len(special_methods)} special methods")
                
                detail_str = ", ".join(details)
                explanation = f"The class '{name}' {detail_str} and {purpose}."
            else:  # medium
                explanation = f"The class '{name}' {purpose} through {method_count} methods."
            
            return explanation
                
        except Exception as e:
            logger.error(f"Error explaining class: {e}")
            return f"Class '{node.name}' defines functionality."
    
    def _infer_function_purpose(self, name: str, node: ast.FunctionDef) -> str:
        """Infer comprehensive function purpose from name and structure"""
        try:
            name_lower = name.lower()
            
            # Enhanced pattern matching
            purpose_patterns = {
                'get_': "retrieves and returns information",
                'set_': "updates or modifies data",
                'create_': "creates new objects or data structures",
                'make_': "constructs or builds something",
                'delete_': "removes or deletes data",
                'remove_': "eliminates specified items",
                'validate_': "validates or verifies data integrity",
                'check_': "verifies conditions or states",
                'process_': "processes data through transformations",
                'handle_': "manages or responds to events",
                'calculate_': "performs mathematical calculations",
                'compute_': "executes computational operations",
                'parse_': "analyzes and extracts data from input",
                'format_': "converts data to specified format",
                'convert_': "transforms data from one form to another",
                'generate_': "creates or produces output",
                'render_': "displays or visualizes data",
                'load_': "retrieves data from storage",
                'save_': "stores data persistently",
                'init': "initializes the object state",
                '__init__': "initializes the object state"
            }
            
            for pattern, purpose in purpose_patterns.items():
                if name_lower.startswith(pattern):
                    return purpose
            
            # Context-based inference
            if 'test' in name_lower:
                return "tests functionality and validates behavior"
            elif 'setup' in name_lower:
                return "configures initial conditions"
            elif 'cleanup' in name_lower:
                return "performs cleanup operations"
            elif 'main' in name_lower:
                return "serves as the main entry point"
            else:
                return "performs specialized operations"
                
        except Exception as e:
            logger.error(f"Error inferring function purpose: {e}")
            return "performs operations"
    
    def _infer_class_purpose(self, name: str, methods: List[ast.FunctionDef]) -> str:
        """Infer comprehensive class purpose from name and methods"""
        try:
            name_lower = name.lower()
            
            # Enhanced class purpose patterns
            purpose_patterns = {
                'manager': "manages and coordinates system operations",
                'controller': "controls application flow and logic",
                'handler': "handles specific events and requests",
                'processor': "processes data through pipelines",
                'analyzer': "analyzes and evaluates information",
                'builder': "constructs complex objects systematically",
                'factory': "creates instances using factory patterns",
                'adapter': "adapts interfaces between components",
                'decorator': "enhances functionality through decoration",
                'strategy': "implements algorithmic strategies",
                'observer': "monitors and responds to changes",
                'validator': "validates data and business rules",
                'parser': "parses and interprets structured data",
                'renderer': "renders visual representations",
                'client': "provides client-side functionality",
                'server': "implements server-side services",
                'service': "provides specific business services",
                'repository': "manages data access and storage",
                'model': "represents data and business logic",
                'view': "handles user interface presentation",
                'config': "manages configuration settings",
                'util': "provides utility functions",
                'helper': "assists with common operations"
            }
            
            for pattern, purpose in purpose_patterns.items():
                if pattern in name_lower:
                    return purpose
            
            # Method-based inference
            method_names = [m.name.lower() for m in methods]
            
            if any('process' in m for m in method_names):
                return "processes data through defined workflows"
            elif any('manage' in m for m in method_names):
                return "manages resources and operations"
            elif any('handle' in m for m in method_names):
                return "handles various system events"
            elif any('validate' in m for m in method_names):
                return "validates data and system states"
            else:
                return "encapsulates related functionality and behavior"
                    
        except Exception as e:
            logger.error(f"Error inferring class purpose: {e}")
            return "provides functionality"
    
    def _describe_parameters(self, args: ast.arguments, abstraction_level: str) -> str:
        """Describe function parameters based on abstraction level"""
        try:
            param_count = len(args.args)
            
            if param_count == 0:
                return "with no parameters"
            elif param_count == 1 and args.args[0].arg == 'self':
                return "as an instance method"
            elif abstraction_level == "low":
                param_names = [arg.arg for arg in args.args if arg.arg != 'self']
                if param_names:
                    return f"with parameters: {', '.join(param_names)}"
                else:
                    return "with no additional parameters"
            else:
                actual_params = param_count - (1 if args.args and args.args[0].arg == 'self' else 0)
                return f"with {actual_params} parameter{'s' if actual_params != 1 else ''}"
                
        except Exception as e:
            logger.error(f"Error describing parameters: {e}")
            return "with parameters"
    
    def _generate_general_explanation(self, code: str, tree: ast.AST) -> str:
        """Generate comprehensive general explanation for code without specific components"""
        try:
            explanations = []
            
            # Analyze code structure
            imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            assignments = len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)])
            conditionals = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
            loops = len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))])
            try_blocks = len([n for n in ast.walk(tree) if isinstance(n, ast.Try)])
            
            # Build structural description
            if imports > 0:
                explanations.append(f"imports {imports} external modules or libraries")
            
            if assignments > 0:
                explanations.append(f"performs {assignments} variable assignments")
            
            if conditionals > 0:
                explanations.append(f"includes {conditionals} conditional decision points")
            
            if loops > 0:
                explanations.append(f"contains {loops} iteration structures")
            
            if try_blocks > 0:
                explanations.append(f"implements {try_blocks} error handling blocks")
            
            # Identify code patterns
            patterns = []
            if 'async' in code or 'await' in code:
                patterns.append("asynchronous programming")
            if '@' in code:
                patterns.append("decorator usage")
            if 'yield' in code:
                patterns.append("generator patterns")
            if 'with' in code:
                patterns.append("context management")
            
            if patterns:
                explanations.append(f"utilizes {', '.join(patterns)}")
            
            if explanations:
                return f"This code {', '.join(explanations)}."
            else:
                return "This code contains basic programming statements and operations."
                
        except Exception as e:
            logger.error(f"Error generating general explanation: {e}")
            return "This code performs programming operations."
    
    def _extract_technical_terms(self, code: str) -> List[str]:
        """Extract comprehensive technical terms from code"""
        try:
            terms = set()
            
            # Programming keywords and constructs
            python_constructs = {
                'def': 'function definition',
                'class': 'class definition',
                'if': 'conditional statement',
                'for': 'for loop',
                'while': 'while loop',
                'try': 'exception handling',
                'except': 'exception handling',
                'import': 'module import',
                'from': 'selective import',
                'async': 'asynchronous programming',
                'await': 'asynchronous wait',
                'yield': 'generator function',
                'lambda': 'anonymous function',
                'with': 'context manager'
            }
            
            for construct, description in python_constructs.items():
                if construct in code:
                    terms.add(description)
            
            # Advanced patterns
            if '@' in code:
                terms.add('decorator pattern')
            if 'super()' in code:
                terms.add('inheritance')
            if '__' in code:
                terms.add('special methods')
            if 'self.' in code:
                terms.add('instance attributes')
            if 'cls.' in code:
                terms.add('class attributes')
            
            return list(terms)
            
        except Exception as e:
            logger.error(f"Error extracting technical terms: {e}")
            return []
    
    def _assess_translation_quality(self, translation: NaturalLanguageTranslation) -> float:
        """Assess comprehensive quality of code-to-language translation"""
        try:
            quality_factors = []
            
            # Length appropriateness
            word_count = len(translation.natural_language.split())
            if 15 <= word_count <= 150:
                quality_factors.append(1.0)
            elif 10 <= word_count < 15 or 150 < word_count <= 300:
                quality_factors.append(0.8)
            elif 5 <= word_count < 10 or 300 < word_count <= 500:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.4)
            
            # Technical terminology coverage
            tech_term_ratio = len(translation.technical_terms) / max(word_count / 10, 1)
            quality_factors.append(min(1.0, tech_term_ratio))
            
            # Error absence
            error_indicators = ['error', 'unable', 'failed', 'exception occurred']
            has_errors = any(indicator in translation.natural_language.lower() 
                           for indicator in error_indicators)
            quality_factors.append(0.2 if has_errors else 1.0)
            
            # Completeness indicators
            completeness_indicators = ['function', 'class', 'method', 'performs', 'provides']
            has_substance = any(indicator in translation.natural_language.lower() 
                              for indicator in completeness_indicators)
            quality_factors.append(0.9 if has_substance else 0.3)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            logger.error(f"Error assessing translation quality: {e}")
            return 0.5
    
    def _understand_code_context(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Comprehensive code context understanding"""
        try:
            context = {
                'code_type': 'general',
                'complexity_level': CodeComplexityLevel.MEDIUM,
                'programming_paradigm': 'procedural',
                'domain_context': 'general_purpose',
                'architectural_pattern': 'none_detected'
            }
            
            # Determine code type and paradigm
            has_classes = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
            has_functions = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            has_async = 'async' in code or 'await' in code
            
            if has_classes and has_functions:
                context['code_type'] = 'object_oriented'
                context['programming_paradigm'] = 'object_oriented'
            elif has_classes:
                context['code_type'] = 'class_definition'
                context['programming_paradigm'] = 'object_oriented'
            elif has_functions:
                context['code_type'] = 'functional'
                context['programming_paradigm'] = 'functional'
            else:
                context['code_type'] = 'script'
                context['programming_paradigm'] = 'procedural'
            
            if has_async:
                context['programming_paradigm'] += '_async'
            
            # Assess complexity
            total_nodes = len(list(ast.walk(tree)))
            if total_nodes > 200:
                context['complexity_level'] = CodeComplexityLevel.VERY_HIGH
            elif total_nodes > 100:
                context['complexity_level'] = CodeComplexityLevel.HIGH
            elif total_nodes > 50:
                context['complexity_level'] = CodeComplexityLevel.MEDIUM
            elif total_nodes > 20:
                context['complexity_level'] = CodeComplexityLevel.LOW
            else:
                context['complexity_level'] = CodeComplexityLevel.VERY_LOW
            
            # Infer domain context
            code_lower = code.lower()
            domain_patterns = {
                'web_development': ['http', 'request', 'response', 'api', 'server', 'client', 'url'],
                'data_science': ['data', 'analysis', 'pandas', 'numpy', 'dataframe', 'plot'],
                'machine_learning': ['model', 'train', 'predict', 'algorithm', 'neural', 'ml'],
                'testing': ['test', 'assert', 'mock', 'unittest', 'pytest'],
                'database': ['sql', 'query', 'database', 'table', 'select', 'insert'],
                'security': ['hash', 'encrypt', 'decrypt', 'token', 'auth', 'security'],
                'game_development': ['game', 'player', 'scene', 'sprite', 'collision'],
                'system_administration': ['system', 'process', 'file', 'directory', 'path']
            }
            
            for domain, keywords in domain_patterns.items():
                if sum(1 for keyword in keywords if keyword in code_lower) >= 2:
                    context['domain_context'] = domain
                    break
            
            return context
            
        except Exception as e:
            logger.error(f"Error understanding code context: {e}")
            return {'error': str(e)}
    
    def _validate_translation_accuracy(self, translation: NaturalLanguageTranslation, 
                                     tree: ast.AST) -> float:
        """Validate translation accuracy against source code"""
        try:
            # Count actual code elements
            actual_classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            actual_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            
            # Check if translation mentions appropriate elements
            explanation_lower = translation.natural_language.lower()
            mentions_classes = 'class' in explanation_lower
            mentions_functions = 'function' in explanation_lower
            
            accuracy_score = 0.0
            
            # Accuracy based on element detection
            if actual_classes > 0 and mentions_classes:
                accuracy_score += 0.4
            elif actual_classes == 0 and not mentions_classes:
                accuracy_score += 0.2
            
            if actual_functions > 0 and mentions_functions:
                accuracy_score += 0.4
            elif actual_functions == 0 and not mentions_functions:
                accuracy_score += 0.2
            
            # Bonus for appropriate technical terms
            if translation.technical_terms:
                accuracy_score += 0.2
            
            return min(1.0, accuracy_score)
            
        except Exception as e:
            logger.error(f"Error validating translation accuracy: {e}")
            return 0.5