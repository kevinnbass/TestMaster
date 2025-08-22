"""
Natural Language Bridge
======================

Revolutionary natural language code translation and explanation system.
Extracted from predictive_code_intelligence.py for enterprise modular architecture.

Agent D Implementation - Hour 15-16: Predictive Intelligence Modularization
"""

import ast
import re
import numpy as np
from typing import Dict, List, Any
import logging

from .data_models import NaturalLanguageTranslation, LanguageBridgeDirection


class NaturalLanguageBridge:
    """
    Revolutionary Natural Language Bridge
    
    Bridges code and natural language understanding with bidirectional translation,
    context analysis, and intelligent explanation generation.
    """
    
    def __init__(self):
        self.code_patterns = {
            'conditional': r'if\s+.*:',
            'loop': r'(for|while)\s+.*:',
            'function_def': r'def\s+\w+\s*\(',
            'class_def': r'class\s+\w+\s*[\(:]',
            'assignment': r'\w+\s*=\s*',
            'return': r'return\s+',
            'import': r'(import|from)\s+\w+'
        }
        
        self.explanation_templates = {
            'function': "This function {name} {purpose}. It takes {parameters} and returns {return_type}.",
            'class': "This class {name} represents {purpose}. It provides {methods} methods for {functionality}.",
            'conditional': "This conditional statement checks if {condition} and {action}.",
            'loop': "This loop iterates {iteration_desc} and {action}.",
            'assignment': "This assigns {value} to the variable {variable}."
        }
        
        self.technical_vocabulary = {
            'programming': ['function', 'class', 'method', 'variable', 'parameter', 'return'],
            'control_flow': ['condition', 'loop', 'iteration', 'branch', 'sequence'],
            'data_structures': ['list', 'dictionary', 'array', 'object', 'string', 'number'],
            'algorithms': ['sort', 'search', 'filter', 'map', 'reduce', 'transform'],
            'patterns': ['singleton', 'factory', 'decorator', 'observer', 'adapter']
        }
        
        self.audience_vocabularies = {
            'beginner': {
                'function': 'a reusable piece of code that performs a specific task',
                'class': 'a blueprint for creating objects with related properties and behaviors',
                'method': 'a function that belongs to a class',
                'variable': 'a container that stores data',
                'loop': 'a way to repeat code multiple times'
            },
            'technical': {
                'function': 'a callable code unit with defined parameters and return values',
                'class': 'an object-oriented programming construct encapsulating data and behavior',
                'method': 'a class-bound function with access to instance state',
                'variable': 'a named memory location storing typed data',
                'loop': 'an iterative control structure for repeated execution'
            },
            'general': {
                'function': 'a section of code that performs a specific operation',
                'class': 'a way to organize related code together',
                'method': 'a function that operates on class data',
                'variable': 'a named value that can change',
                'loop': 'code that repeats until a condition is met'
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def translate_code_to_language(self, code: str, target_audience: str = "general",
                                 abstraction_level: str = "medium") -> NaturalLanguageTranslation:
        """Translate code to natural language explanation"""
        
        try:
            translation = NaturalLanguageTranslation(
                direction=LanguageBridgeDirection.CODE_TO_LANGUAGE,
                source_code=code,
                target_audience=target_audience,
                abstraction_level=abstraction_level
            )
            
            # Parse code
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                translation.natural_language = f"Unable to parse code due to syntax errors: {str(e)}"
                translation.translation_quality = 0.0
                return translation
            
            # Generate explanation based on abstraction level
            if abstraction_level == "high":
                explanation = self._generate_high_level_explanation(tree, code, target_audience)
            elif abstraction_level == "low":
                explanation = self._generate_detailed_explanation(tree, code, target_audience)
            else:  # medium
                explanation = self._generate_medium_explanation(tree, code, target_audience)
            
            translation.natural_language = explanation
            
            # Extract technical terms
            translation.technical_terms = self._extract_technical_terms(code, target_audience)
            
            # Assess translation quality
            translation.translation_quality = self._assess_translation_quality(translation)
            
            # Generate context understanding
            translation.context_understanding = self._understand_code_context(tree, code)
            
            # Generate alternative explanations
            translation.alternative_explanations = self._generate_alternative_explanations(
                tree, code, target_audience, abstraction_level
            )
            
            return translation
            
        except Exception as e:
            self.logger.error(f"Error translating code to language: {e}")
            return NaturalLanguageTranslation(
                direction=LanguageBridgeDirection.CODE_TO_LANGUAGE,
                source_code=code,
                natural_language=f"Translation error: {str(e)}",
                translation_quality=0.0
            )
    
    def _generate_high_level_explanation(self, tree: ast.AST, code: str, audience: str) -> str:
        """Generate high-level abstracted explanation"""
        
        try:
            # Analyze overall purpose
            purpose = self._infer_code_purpose(tree, code)
            
            # Count major components
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            
            if classes and functions:
                explanation = f"This code implements {purpose} using object-oriented design with {len(classes)} classes and {len(functions)} functions."
            elif classes:
                explanation = f"This code defines {len(classes)} classes to implement {purpose}."
            elif functions:
                explanation = f"This code provides {len(functions)} functions for {purpose}."
            else:
                explanation = f"This code implements {purpose} through a series of operations."
            
            # Add context for audience
            if audience == "beginner":
                explanation += " The code is organized to make it easy to understand and maintain."
            elif audience == "technical":
                explanation += " The architecture follows standard software engineering principles."
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating high-level explanation: {e}")
            return "This code performs various operations."
    
    def _generate_medium_explanation(self, tree: ast.AST, code: str, audience: str) -> str:
        """Generate medium-detail explanation"""
        
        try:
            explanations = []
            
            # Explain major components
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    func_explanation = self._explain_function(node, audience)
                    explanations.append(func_explanation)
                elif isinstance(node, ast.ClassDef):
                    class_explanation = self._explain_class(node, audience)
                    explanations.append(class_explanation)
            
            # If no major components, analyze general structure
            if not explanations:
                explanations.append(self._generate_general_explanation(code, tree, audience))
            
            # Combine explanations
            combined = " ".join(explanations)
            
            # Add workflow explanation if multiple components
            if len(explanations) > 1:
                combined += f" These {len(explanations)} components work together to accomplish the overall task."
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error generating medium explanation: {e}")
            return "This code contains multiple components that work together."
    
    def _generate_detailed_explanation(self, tree: ast.AST, code: str, audience: str) -> str:
        """Generate detailed low-level explanation"""
        
        try:
            explanations = []
            
            # Analyze imports
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            if imports:
                import_explanation = self._explain_imports(imports, audience)
                explanations.append(import_explanation)
            
            # Analyze each major component in detail
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_explanation = self._explain_class_detailed(node, audience)
                    explanations.append(class_explanation)
                elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    func_explanation = self._explain_function_detailed(node, audience)
                    explanations.append(func_explanation)
            
            # Analyze main execution flow
            main_flow = self._analyze_main_flow(tree, audience)
            if main_flow:
                explanations.append(main_flow)
            
            return " ".join(explanations)
            
        except Exception as e:
            self.logger.error(f"Error generating detailed explanation: {e}")
            return "This code performs detailed operations with multiple steps."
    
    def _explain_function(self, node: ast.FunctionDef, audience: str) -> str:
        """Generate natural language explanation for function"""
        
        try:
            name = node.name
            vocabulary = self.audience_vocabularies.get(audience, self.audience_vocabularies['general'])
            
            # Analyze parameters
            param_count = len(node.args.args)
            if param_count == 0:
                param_desc = "no parameters"
            elif param_count == 1:
                param_desc = "one parameter"
            else:
                param_desc = f"{param_count} parameters"
            
            # Analyze return behavior
            has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
            return_desc = "returns a value" if has_return else "performs an action"
            
            # Determine purpose based on name and structure
            purpose = self._infer_function_purpose(name, node)
            
            # Generate explanation based on audience
            if audience == "beginner":
                function_term = vocabulary.get('function', 'function')
                return f"The {function_term} '{name}' {purpose}. It takes {param_desc} and {return_desc}."
            elif audience == "technical":
                return f"Function '{name}' implements {purpose} with {param_desc}, {return_desc}."
            else:  # general
                return f"The function '{name}' {purpose} and {return_desc}."
                
        except Exception as e:
            self.logger.error(f"Error explaining function: {e}")
            return f"Function '{node.name}' performs operations."
    
    def _explain_class(self, node: ast.ClassDef, audience: str) -> str:
        """Generate natural language explanation for class"""
        
        try:
            name = node.name
            vocabulary = self.audience_vocabularies.get(audience, self.audience_vocabularies['general'])
            
            # Count methods
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            method_count = len(methods)
            
            # Analyze inheritance
            inheritance = len(node.bases)
            
            # Determine purpose
            purpose = self._infer_class_purpose(name, methods)
            
            # Generate explanation based on audience
            if audience == "beginner":
                class_term = vocabulary.get('class', 'class')
                return f"The {class_term} '{name}' {purpose}. It has {method_count} methods to handle different tasks."
            elif audience == "technical":
                inheritance_desc = f" with {inheritance} base classes" if inheritance else ""
                return f"Class '{name}' encapsulates {purpose} through {method_count} methods{inheritance_desc}."
            else:  # general
                return f"The class '{name}' {purpose} using {method_count} different methods."
                
        except Exception as e:
            self.logger.error(f"Error explaining class: {e}")
            return f"Class '{node.name}' defines functionality."
    
    def _infer_function_purpose(self, name: str, node: ast.FunctionDef) -> str:
        """Infer function purpose from name and structure"""
        
        try:
            name_lower = name.lower()
            
            # Common naming patterns
            purpose_patterns = {
                'get_': 'retrieves information',
                'set_': 'updates or modifies data',
                'create_': 'creates new objects or data',
                'make_': 'constructs or builds something',
                'delete_': 'removes or deletes data',
                'remove_': 'removes items',
                'add_': 'adds new items',
                'update_': 'modifies existing data',
                'validate_': 'validates or verifies data',
                'check_': 'checks conditions or states',
                'process_': 'processes data or performs operations',
                'calculate_': 'performs calculations',
                'compute_': 'computes values',
                'parse_': 'parses or analyzes text',
                'format_': 'formats data for display',
                'convert_': 'converts data between formats',
                'transform_': 'transforms data',
                'filter_': 'filters data based on criteria',
                'sort_': 'sorts data in order',
                'search_': 'searches for specific items',
                'find_': 'finds and returns items'
            }
            
            for pattern, purpose in purpose_patterns.items():
                if name_lower.startswith(pattern):
                    return purpose
            
            # Special cases
            if name_lower == '__init__' or name_lower == 'init':
                return 'initializes the object'
            elif 'test' in name_lower:
                return 'tests functionality'
            elif 'main' in name_lower:
                return 'coordinates the main program flow'
            else:
                return 'performs specific operations'
                
        except Exception as e:
            self.logger.error(f"Error inferring function purpose: {e}")
            return 'performs operations'
    
    def _infer_class_purpose(self, name: str, methods: List[ast.FunctionDef]) -> str:
        """Infer class purpose from name and methods"""
        
        try:
            name_lower = name.lower()
            
            # Common class naming patterns
            purpose_patterns = {
                'manager': 'manages and coordinates operations',
                'handler': 'handles specific events or requests',
                'processor': 'processes data or requests',
                'controller': 'controls system behavior',
                'analyzer': 'analyzes data or code',
                'analyser': 'analyzes data or code',
                'validator': 'validates data or inputs',
                'builder': 'constructs objects or structures',
                'factory': 'creates instances of objects',
                'adapter': 'adapts interfaces between components',
                'client': 'provides client functionality',
                'server': 'provides server functionality',
                'service': 'provides specific services',
                'helper': 'provides utility functions',
                'util': 'provides utility functions',
                'tool': 'provides tools for specific tasks',
                'engine': 'drives core functionality',
                'generator': 'generates content or data',
                'parser': 'parses text or data',
                'formatter': 'formats data for display',
                'converter': 'converts data between formats',
                'transformer': 'transforms data'
            }
            
            for pattern, purpose in purpose_patterns.items():
                if pattern in name_lower:
                    return purpose
            
            # Analyze method names for additional clues
            method_names = [m.name.lower() for m in methods]
            
            if any('process' in m for m in method_names):
                return 'processes data'
            elif any('manage' in m for m in method_names):
                return 'manages resources or operations'
            elif any('handle' in m for m in method_names):
                return 'handles events or requests'
            elif any('analyze' in m or 'analyse' in m for m in method_names):
                return 'analyzes information'
            else:
                return 'encapsulates related functionality'
                
        except Exception as e:
            self.logger.error(f"Error inferring class purpose: {e}")
            return 'provides functionality'
    
    def _extract_technical_terms(self, code: str, audience: str) -> List[str]:
        """Extract technical terms from code based on audience"""
        
        try:
            terms = set()
            
            # Programming keywords
            python_keywords = ['def', 'class', 'if', 'for', 'while', 'try', 'except', 'import', 'from', 'return']
            for keyword in python_keywords:
                if keyword in code:
                    terms.add(keyword)
            
            # Technical patterns based on audience
            if audience == "beginner":
                # Focus on basic concepts
                if 'def ' in code:
                    terms.add('function definition')
                if 'class ' in code:
                    terms.add('class definition')
                if 'if ' in code:
                    terms.add('conditional statement')
                if 'for ' in code or 'while ' in code:
                    terms.add('loop')
            
            elif audience == "technical":
                # Include advanced concepts
                if 'async' in code:
                    terms.add('asynchronous programming')
                if 'yield' in code:
                    terms.add('generator function')
                if 'lambda' in code:
                    terms.add('lambda expression')
                if '@' in code:
                    terms.add('decorator')
                if 'super()' in code:
                    terms.add('inheritance')
                if '__' in code:
                    terms.add('dunder methods')
            
            else:  # general audience
                # Balanced technical terms
                if 'def ' in code:
                    terms.add('function')
                if 'class ' in code:
                    terms.add('class')
                if any(pattern in code for pattern in ['if ', 'elif ', 'else:']):
                    terms.add('conditional logic')
                if any(pattern in code for pattern in ['for ', 'while ']):
                    terms.add('iteration')
            
            return list(terms)
            
        except Exception as e:
            self.logger.error(f"Error extracting technical terms: {e}")
            return []
    
    def _assess_translation_quality(self, translation: NaturalLanguageTranslation) -> float:
        """Assess quality of code-to-language translation"""
        
        try:
            quality_factors = []
            
            # Length factor (not too short, not too long)
            length = len(translation.natural_language.split())
            if 10 <= length <= 150:
                quality_factors.append(1.0)
            elif 5 <= length < 10 or 150 < length <= 300:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.4)
            
            # Technical term coverage
            if translation.technical_terms:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            # Completeness (contains actual explanation, not just error messages)
            content_lower = translation.natural_language.lower()
            if "error" not in content_lower and "unable" not in content_lower:
                quality_factors.append(0.9)
            else:
                quality_factors.append(0.2)
            
            # Clarity indicators
            clarity_indicators = ['this', 'the', 'it', 'function', 'class', 'method']
            clarity_score = sum(1 for indicator in clarity_indicators 
                              if indicator in content_lower) / len(clarity_indicators)
            quality_factors.append(clarity_score)
            
            # Audience appropriateness
            if translation.target_audience == "beginner":
                if any(word in content_lower for word in ['simple', 'basic', 'easy']):
                    quality_factors.append(0.8)
                else:
                    quality_factors.append(0.6)
            elif translation.target_audience == "technical":
                if any(word in content_lower for word in ['implement', 'encapsulate', 'architecture']):
                    quality_factors.append(0.8)
                else:
                    quality_factors.append(0.6)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            self.logger.error(f"Error assessing translation quality: {e}")
            return 0.5
    
    def _understand_code_context(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Understand context of code for better translation"""
        
        try:
            context = {
                'code_type': 'general',
                'complexity': 'medium',
                'purpose': 'utility',
                'domain': 'general',
                'programming_paradigm': 'procedural'
            }
            
            # Determine code type and paradigm
            has_classes = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
            has_functions = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            has_async = 'async' in code
            
            if has_classes and has_functions:
                context['code_type'] = 'object_oriented'
                context['programming_paradigm'] = 'object_oriented'
            elif has_classes:
                context['code_type'] = 'class_definition'
                context['programming_paradigm'] = 'object_oriented'
            elif has_functions:
                context['code_type'] = 'functional'
                context['programming_paradigm'] = 'functional'
            elif has_async:
                context['code_type'] = 'asynchronous'
                context['programming_paradigm'] = 'asynchronous'
            else:
                context['code_type'] = 'script'
                context['programming_paradigm'] = 'procedural'
            
            # Determine complexity
            total_nodes = len(list(ast.walk(tree)))
            if total_nodes > 100:
                context['complexity'] = 'high'
            elif total_nodes > 30:
                context['complexity'] = 'medium'
            else:
                context['complexity'] = 'low'
            
            # Infer domain
            context['domain'] = self._infer_code_domain(code)
            
            # Infer purpose
            context['purpose'] = self._infer_code_purpose(tree, code)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error understanding code context: {e}")
            return {}
    
    def _infer_code_domain(self, code: str) -> str:
        """Infer the domain/field of the code"""
        
        code_lower = code.lower()
        
        domain_keywords = {
            'web_development': ['http', 'request', 'api', 'server', 'client', 'url', 'html', 'json'],
            'data_science': ['data', 'analysis', 'pandas', 'numpy', 'dataset', 'csv', 'statistics'],
            'machine_learning': ['ml', 'model', 'train', 'predict', 'algorithm', 'neural', 'sklearn'],
            'testing': ['test', 'assert', 'mock', 'unittest', 'pytest', 'coverage'],
            'database': ['database', 'sql', 'query', 'table', 'insert', 'select', 'orm'],
            'security': ['security', 'auth', 'password', 'encrypt', 'hash', 'token'],
            'networking': ['network', 'socket', 'tcp', 'udp', 'connection', 'protocol'],
            'file_processing': ['file', 'path', 'directory', 'read', 'write', 'parse'],
            'mathematics': ['math', 'calculate', 'compute', 'formula', 'equation', 'algorithm'],
            'game_development': ['game', 'player', 'score', 'level', 'graphics', 'animation'],
            'system_administration': ['system', 'process', 'service', 'daemon', 'config', 'admin']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in code_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general'
    
    def _infer_code_purpose(self, tree: ast.AST, code: str) -> str:
        """Infer the overall purpose of the code"""
        
        # Analyze class and function names for purpose clues
        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                names.append(node.name.lower())
        
        combined_names = ' '.join(names)
        
        purpose_patterns = {
            'data processing': ['process', 'parse', 'transform', 'convert', 'filter'],
            'analysis': ['analyze', 'analyse', 'calculate', 'compute', 'evaluate'],
            'management': ['manage', 'control', 'coordinate', 'organize'],
            'utility': ['helper', 'util', 'tool', 'support'],
            'configuration': ['config', 'setting', 'option', 'parameter'],
            'testing': ['test', 'check', 'validate', 'verify'],
            'communication': ['send', 'receive', 'connect', 'request', 'response'],
            'storage': ['save', 'load', 'store', 'retrieve', 'database'],
            'user interface': ['display', 'show', 'render', 'view', 'interface'],
            'automation': ['automate', 'schedule', 'monitor', 'watch']
        }
        
        for purpose, keywords in purpose_patterns.items():
            if any(keyword in combined_names for keyword in keywords):
                return purpose
        
        return 'general utility'
    
    def _generate_alternative_explanations(self, tree: ast.AST, code: str, 
                                         audience: str, abstraction_level: str) -> List[str]:
        """Generate alternative explanations for the same code"""
        
        try:
            alternatives = []
            
            # Generate explanation with different focus
            if abstraction_level != "high":
                alt_translation = self.translate_code_to_language(code, audience, "high")
                alternatives.append(alt_translation.natural_language)
            
            # Generate simplified version for complex explanations
            if abstraction_level == "low":
                simplified = self._generate_simplified_explanation(tree, code)
                alternatives.append(simplified)
            
            # Generate workflow-focused explanation
            workflow_explanation = self._generate_workflow_explanation(tree, code)
            if workflow_explanation:
                alternatives.append(workflow_explanation)
            
            return alternatives[:3]  # Limit to 3 alternatives
            
        except Exception as e:
            self.logger.error(f"Error generating alternative explanations: {e}")
            return []
    
    def _generate_simplified_explanation(self, tree: ast.AST, code: str) -> str:
        """Generate simplified explanation focusing on main actions"""
        
        try:
            actions = []
            
            # Extract main actions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    action = self._extract_main_action(node)
                    if action:
                        actions.append(action)
            
            if actions:
                if len(actions) == 1:
                    return f"This code {actions[0]}."
                else:
                    return f"This code {', '.join(actions[:-1])}, and {actions[-1]}."
            else:
                return "This code performs basic operations."
                
        except Exception as e:
            self.logger.error(f"Error generating simplified explanation: {e}")
            return "This code performs operations."
    
    def _extract_main_action(self, node: ast.FunctionDef) -> str:
        """Extract the main action from a function"""
        
        name = node.name.lower()
        
        if 'get' in name:
            return "gets information"
        elif 'set' in name or 'update' in name:
            return "updates data"
        elif 'create' in name or 'make' in name:
            return "creates something"
        elif 'delete' in name or 'remove' in name:
            return "removes items"
        elif 'process' in name:
            return "processes data"
        elif 'calculate' in name or 'compute' in name:
            return "performs calculations"
        else:
            return None
    
    def _generate_workflow_explanation(self, tree: ast.AST, code: str) -> str:
        """Generate workflow-focused explanation"""
        
        try:
            steps = []
            
            # Identify workflow steps from function calls and main operations
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    step = self._identify_workflow_step(node)
                    if step:
                        steps.append(step)
            
            if len(steps) > 1:
                workflow = "The workflow follows these steps: "
                for i, step in enumerate(steps):
                    if i == 0:
                        workflow += f"first, {step}"
                    elif i == len(steps) - 1:
                        workflow += f", and finally, {step}"
                    else:
                        workflow += f", then {step}"
                workflow += "."
                return workflow
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating workflow explanation: {e}")
            return None
    
    def _identify_workflow_step(self, node: ast.FunctionDef) -> str:
        """Identify workflow step from function"""
        
        name = node.name.lower()
        
        step_patterns = {
            'init': 'initialize the system',
            'setup': 'set up the environment',
            'load': 'load the data',
            'process': 'process the information',
            'validate': 'validate the results',
            'save': 'save the output',
            'cleanup': 'clean up resources',
            'finalize': 'finalize the operation'
        }
        
        for pattern, step in step_patterns.items():
            if pattern in name:
                return step
        
        return None
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is a method (inside a class)"""
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in ast.walk(node):
                    return True
        return False
    
    def _explain_imports(self, imports: List, audience: str) -> str:
        """Explain import statements"""
        
        import_count = len(imports)
        if import_count == 1:
            return "This code imports one external library to extend its capabilities."
        else:
            return f"This code imports {import_count} external libraries to provide additional functionality."
    
    def _explain_function_detailed(self, node: ast.FunctionDef, audience: str) -> str:
        """Generate detailed function explanation"""
        
        basic_explanation = self._explain_function(node, audience)
        
        # Add complexity analysis
        complexity_info = ""
        if hasattr(node, 'body') and len(node.body) > 10:
            complexity_info = " This is a complex function with multiple operations."
        elif any(isinstance(child, (ast.If, ast.For, ast.While)) for child in ast.walk(node)):
            complexity_info = " It includes conditional logic and iteration."
        
        return basic_explanation + complexity_info
    
    def _explain_class_detailed(self, node: ast.ClassDef, audience: str) -> str:
        """Generate detailed class explanation"""
        
        basic_explanation = self._explain_class(node, audience)
        
        # Add attribute analysis
        init_method = None
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                init_method = child
                break
        
        if init_method:
            # Count attributes set in __init__
            attributes = []
            for child in ast.walk(init_method):
                if (isinstance(child, ast.Assign) and
                    any(isinstance(target, ast.Attribute) and 
                        isinstance(target.value, ast.Name) and 
                        target.value.id == 'self' for target in child.targets)):
                    attributes.append("attribute")
            
            if attributes:
                attr_info = f" It initializes {len(attributes)} attributes to store its state."
                basic_explanation += attr_info
        
        return basic_explanation
    
    def _analyze_main_flow(self, tree: ast.AST, audience: str) -> str:
        """Analyze main execution flow"""
        
        # Look for main execution pattern
        has_main_check = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Compare) and
                isinstance(node.left, ast.Name) and
                node.left.id == '__name__' and
                any(isinstance(comp, ast.Constant) and comp.value == '__main__' 
                    for comp in [node.comparators[0]] if node.comparators)):
                has_main_check = True
                break
        
        if has_main_check:
            return "The code includes a main execution block that runs when the script is executed directly."
        
        return None
    
    def _generate_general_explanation(self, code: str, tree: ast.AST, audience: str) -> str:
        """Generate general explanation when no specific components found"""
        
        try:
            explanations = []
            
            # Count different types of statements
            imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            assignments = len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)])
            conditionals = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
            loops = len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))])
            
            if imports > 0:
                explanations.append(f"imports {imports} external libraries")
            
            if assignments > 0:
                explanations.append(f"contains {assignments} variable assignments")
            
            if conditionals > 0:
                explanations.append(f"includes {conditionals} decision points")
            
            if loops > 0:
                explanations.append(f"has {loops} loops for repetition")
            
            if explanations:
                return f"This code {', '.join(explanations)}."
            else:
                return "This code performs basic programming operations."
                
        except Exception as e:
            self.logger.error(f"Error generating general explanation: {e}")
            return "This code contains programming statements."


def create_language_bridge() -> NaturalLanguageBridge:
    """Factory function to create NaturalLanguageBridge instance"""
    
    return NaturalLanguageBridge()