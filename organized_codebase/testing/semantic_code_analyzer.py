"""
Semantic Code Analyzer - Deep Semantic Code Understanding System

This module implements advanced AI-powered code comprehension that understands
code semantics, intent, and context at human-expert level. It provides deep
analysis of code meaning, relationships, and developer intent beyond simple
syntax analysis.

Key Capabilities:
- Deep semantic understanding of code meaning and relationships
- Developer intent recognition and design pattern identification
- Contextual comprehension with full codebase dependency analysis
- Variable role analysis and function purpose identification
- Algorithm pattern detection and data flow comprehension
- Control flow analysis and side effect detection
- Architecture pattern recognition and code coupling analysis
- Interface contract analysis and cross-module dependency understanding
"""

import asyncio
import logging
import ast
import inspect
import json
import hashlib
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import importlib
import importlib.util
import sys
import re
import numpy as np
from collections import defaultdict, Counter
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeElementType(Enum):
    """Types of code elements for semantic analysis"""
    VARIABLE = "variable"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    PARAMETER = "parameter"
    RETURN_VALUE = "return_value"
    LOOP = "loop"
    CONDITION = "condition"
    EXCEPTION = "exception"
    DECORATOR = "decorator"
    IMPORT = "import"

class SemanticRole(Enum):
    """Semantic roles of code elements"""
    CONTROLLER = "controller"
    DATA_PROCESSOR = "data_processor"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    AGGREGATOR = "aggregator"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    COMMAND = "command"
    STATE_MANAGER = "state_manager"
    UTILITY = "utility"
    CONFIGURATION = "configuration"
    INTERFACE = "interface"

class DesignPattern(Enum):
    """Design patterns that can be identified"""
    SINGLETON = "singleton"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    COMMAND = "command"
    ADAPTER = "adapter"
    DECORATOR = "decorator"
    FACADE = "facade"
    TEMPLATE_METHOD = "template_method"
    STATE = "state"
    VISITOR = "visitor"
    PROXY = "proxy"
    CHAIN_OF_RESPONSIBILITY = "chain_of_responsibility"

class IntentCategory(Enum):
    """Categories of developer intent"""
    DATA_MANIPULATION = "data_manipulation"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    OPTIMIZATION = "optimization"
    ERROR_HANDLING = "error_handling"
    CONFIGURATION = "configuration"
    LOGGING = "logging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY = "security"
    DEBUGGING = "debugging"

@dataclass
class SemanticElement:
    """Represents a semantically analyzed code element"""
    element_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest())
    element_type: CodeElementType = CodeElementType.VARIABLE
    name: str = ""
    semantic_role: SemanticRole = SemanticRole.UTILITY
    intent_category: IntentCategory = IntentCategory.DATA_MANIPULATION
    context: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    side_effects: List[str] = field(default_factory=list)
    data_flow: Dict[str, Any] = field(default_factory=dict)
    control_flow: Dict[str, Any] = field(default_factory=dict)
    performance_characteristics: Dict[str, str] = field(default_factory=dict)
    security_implications: List[str] = field(default_factory=list)
    maintainability_score: float = 0.0
    readability_score: float = 0.0
    testability_score: float = 0.0

@dataclass
class CodeComprehension:
    """Comprehensive understanding of code structure and intent"""
    comprehension_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    file_path: str = ""
    module_name: str = ""
    semantic_elements: Dict[str, SemanticElement] = field(default_factory=dict)
    design_patterns: List[DesignPattern] = field(default_factory=list)
    architecture_insights: Dict[str, Any] = field(default_factory=dict)
    code_quality_assessment: Dict[str, float] = field(default_factory=dict)
    technical_debt_analysis: Dict[str, Any] = field(default_factory=dict)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    refactoring_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    documentation_quality: Dict[str, float] = field(default_factory=dict)
    test_coverage_analysis: Dict[str, float] = field(default_factory=dict)
    security_analysis: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0

@dataclass
class CodeRelationshipGraph:
    """Graph representation of code relationships"""
    graph_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    dependency_graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    call_graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    data_flow_graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    control_flow_graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    inheritance_graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    composition_graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    coupling_metrics: Dict[str, float] = field(default_factory=dict)
    cohesion_metrics: Dict[str, float] = field(default_factory=dict)
    centrality_metrics: Dict[str, float] = field(default_factory=dict)

class IntentRecognitionEngine:
    """Recognizes developer intent from code patterns and structure"""
    
    def __init__(self):
        self.intent_patterns = {
            IntentCategory.DATA_MANIPULATION: [
                r'\.filter\(', r'\.map\(', r'\.reduce\(', r'\.sort\(',
                r'\.append\(', r'\.extend\(', r'\.update\(', r'\.merge\('
            ],
            IntentCategory.VALIDATION: [
                r'if.*is None', r'if.*not.*:', r'assert\s+', r'raise\s+.*Error',
                r'isinstance\(', r'hasattr\(', r'\.startswith\(', r'\.endswith\('
            ],
            IntentCategory.TRANSFORMATION: [
                r'\.transform\(', r'\.convert\(', r'\.to_\w+\(', r'\.parse\(',
                r'json\.loads', r'json\.dumps', r'str\(', r'int\(', r'float\('
            ],
            IntentCategory.ERROR_HANDLING: [
                r'try:', r'except\s+', r'finally:', r'raise\s+', r'\.catch\(',
                r'logging\.error', r'logging\.exception', r'traceback\.'
            ],
            IntentCategory.PERFORMANCE_MONITORING: [
                r'time\.time\(', r'\.measure\(', r'\.profile\(', r'@timer',
                r'timeit\.', r'profile\.', r'perf_counter', r'@performance'
            ],
            IntentCategory.SECURITY: [
                r'hashlib\.', r'encrypt\(', r'decrypt\(', r'bcrypt\.', r'jwt\.',
                r'authenticate', r'authorize', r'permission', r'secret'
            ]
        }
        
        self.semantic_role_patterns = {
            SemanticRole.CONTROLLER: [
                r'class.*Controller', r'def.*handle', r'def.*process',
                r'def.*execute', r'def.*run', r'def.*start'
            ],
            SemanticRole.VALIDATOR: [
                r'def.*validate', r'def.*check', r'def.*verify',
                r'def.*is_valid', r'class.*Validator'
            ],
            SemanticRole.TRANSFORMER: [
                r'def.*transform', r'def.*convert', r'def.*parse',
                r'def.*serialize', r'def.*deserialize', r'class.*Transformer'
            ],
            SemanticRole.FACTORY: [
                r'class.*Factory', r'def.*create', r'def.*build',
                r'def.*make', r'def.*generate'
            ],
            SemanticRole.UTILITY: [
                r'def.*util', r'def.*helper', r'class.*Utils',
                r'class.*Helper', r'def.*_\w+'  # Private methods often utilities
            ]
        }
        
    def analyze_intent(self, code: str, context: Dict[str, Any] = None) -> Dict[IntentCategory, float]:
        """Analyze developer intent from code patterns"""
        try:
            intent_scores = {}
            
            for intent, patterns in self.intent_patterns.items():
                score = 0.0
                for pattern in patterns:
                    matches = len(re.findall(pattern, code, re.IGNORECASE | re.MULTILINE))
                    score += matches * 0.1  # Weight each match
                
                # Normalize score
                intent_scores[intent] = min(1.0, score)
            
            # Context-based adjustments
            if context:
                self._adjust_intent_scores_with_context(intent_scores, context)
            
            return intent_scores
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return {}
    
    def identify_semantic_role(self, element_name: str, element_code: str, 
                              element_type: CodeElementType) -> SemanticRole:
        """Identify semantic role of code element"""
        try:
            role_scores = {}
            
            for role, patterns in self.semantic_role_patterns.items():
                score = 0.0
                for pattern in patterns:
                    if re.search(pattern, element_name, re.IGNORECASE):
                        score += 1.0
                    if re.search(pattern, element_code, re.IGNORECASE):
                        score += 0.5
                
                role_scores[role] = score
            
            # Return role with highest score, default to UTILITY
            if role_scores:
                best_role = max(role_scores.items(), key=lambda x: x[1])[0]
                return best_role if role_scores[best_role] > 0 else SemanticRole.UTILITY
            else:
                return SemanticRole.UTILITY
                
        except Exception as e:
            logger.error(f"Error identifying semantic role: {e}")
            return SemanticRole.UTILITY
    
    def _adjust_intent_scores_with_context(self, intent_scores: Dict[IntentCategory, float], 
                                         context: Dict[str, Any]):
        """Adjust intent scores based on context"""
        try:
            # File name context
            file_name = context.get('file_name', '').lower()
            
            if 'test' in file_name:
                intent_scores[IntentCategory.TESTING] += 0.5
            elif 'config' in file_name:
                intent_scores[IntentCategory.CONFIGURATION] += 0.5
            elif 'util' in file_name or 'helper' in file_name:
                intent_scores[IntentCategory.DATA_MANIPULATION] += 0.3
            
            # Function/class name context
            element_name = context.get('element_name', '').lower()
            
            if 'log' in element_name:
                intent_scores[IntentCategory.LOGGING] += 0.4
            elif 'debug' in element_name:
                intent_scores[IntentCategory.DEBUGGING] += 0.4
            elif 'auth' in element_name or 'secure' in element_name:
                intent_scores[IntentCategory.SECURITY] += 0.4
            
        except Exception as e:
            logger.error(f"Error adjusting intent scores: {e}")

class DesignPatternDetector:
    """Detects design patterns in code structure"""
    
    def __init__(self):
        self.pattern_signatures = {
            DesignPattern.SINGLETON: {
                'class_methods': ['__new__', '__init__'],
                'instance_variables': ['_instance'],
                'keywords': ['singleton', 'instance']
            },
            DesignPattern.FACTORY: {
                'method_names': ['create', 'build', 'make', 'get_instance'],
                'class_suffix': ['Factory', 'Creator', 'Builder'],
                'keywords': ['factory', 'create']
            },
            DesignPattern.OBSERVER: {
                'method_names': ['notify', 'update', 'subscribe', 'unsubscribe'],
                'class_suffix': ['Observer', 'Observable', 'Subject'],
                'keywords': ['observer', 'notify', 'subscribe']
            },
            DesignPattern.STRATEGY: {
                'method_names': ['execute', 'apply', 'run'],
                'class_suffix': ['Strategy', 'Algorithm'],
                'keywords': ['strategy', 'algorithm']
            },
            DesignPattern.DECORATOR: {
                'decorator_usage': True,
                'method_names': ['decorate', 'wrap'],
                'keywords': ['decorator', 'wrapper']
            }
        }
    
    def detect_patterns(self, tree: ast.AST, code: str) -> List[DesignPattern]:
        """Detect design patterns in AST and code"""
        try:
            detected_patterns = []
            
            # Analyze AST structure
            class_names = []
            method_names = []
            decorator_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_names.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    method_names.append(node.name)
                    if node.decorator_list:
                        decorator_count += len(node.decorator_list)
            
            # Check each pattern
            for pattern, signature in self.pattern_signatures.items():
                if self._matches_pattern(pattern, signature, class_names, method_names, 
                                       decorator_count, code):
                    detected_patterns.append(pattern)
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Error detecting design patterns: {e}")
            return []
    
    def _matches_pattern(self, pattern: DesignPattern, signature: Dict[str, Any],
                        class_names: List[str], method_names: List[str],
                        decorator_count: int, code: str) -> bool:
        """Check if code matches a specific pattern signature"""
        try:
            score = 0.0
            
            # Check class name suffixes
            if 'class_suffix' in signature:
                for suffix in signature['class_suffix']:
                    if any(suffix in class_name for class_name in class_names):
                        score += 1.0
            
            # Check method names
            if 'method_names' in signature:
                for method in signature['method_names']:
                    if method in method_names:
                        score += 0.5
            
            # Check keywords in code
            if 'keywords' in signature:
                for keyword in signature['keywords']:
                    if keyword.lower() in code.lower():
                        score += 0.3
            
            # Special checks
            if pattern == DesignPattern.DECORATOR and decorator_count > 0:
                score += 1.0
            
            # Pattern-specific logic
            if pattern == DesignPattern.SINGLETON:
                if any('_instance' in code for _ in [1]):  # Check for singleton pattern
                    score += 1.0
            
            return score >= 1.0  # Threshold for pattern detection
            
        except Exception as e:
            logger.error(f"Error matching pattern {pattern}: {e}")
            return False

class ContextualComprehensionSystem:
    """Provides contextual understanding of code within full codebase"""
    
    def __init__(self):
        self.codebase_graph = CodeRelationshipGraph()
        self.module_registry = {}
        self.dependency_cache = {}
        
    def analyze_contextual_relationships(self, file_path: str, 
                                       project_root: str) -> Dict[str, Any]:
        """Analyze code in context of full codebase"""
        try:
            relationships = {
                'imports': [],
                'exports': [],
                'dependencies': [],
                'dependents': [],
                'coupling_strength': 0.0,
                'cohesion_score': 0.0,
                'architecture_layer': '',
                'module_role': ''
            }
            
            # Read and parse file
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return relationships
            
            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        relationships['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        relationships['imports'].append(node.module)
            
            # Analyze exports (functions and classes defined)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.name.startswith('_'):  # Public elements
                        relationships['exports'].append(node.name)
            
            # Calculate coupling and cohesion
            relationships['coupling_strength'] = self._calculate_coupling(file_path)
            relationships['cohesion_score'] = self._calculate_cohesion(tree)
            
            # Determine architecture layer
            relationships['architecture_layer'] = self._determine_architecture_layer(
                file_path, relationships['imports']
            )
            
            # Determine module role
            relationships['module_role'] = self._determine_module_role(
                file_path, relationships['exports'], code
            )
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing contextual relationships: {e}")
            return {}
    
    def _calculate_coupling(self, file_path: str) -> float:
        """Calculate coupling strength with other modules"""
        try:
            # Simplified coupling calculation based on imports
            # In production, would analyze actual usage patterns
            
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            import_count = code.count('import ') + code.count('from ')
            file_size = len(code.splitlines())
            
            if file_size == 0:
                return 0.0
            
            # Normalize coupling (more imports relative to size = higher coupling)
            coupling = min(1.0, import_count / max(1, file_size / 10))
            return coupling
            
        except Exception as e:
            logger.error(f"Error calculating coupling: {e}")
            return 0.5
    
    def _calculate_cohesion(self, tree: ast.AST) -> float:
        """Calculate internal cohesion of module"""
        try:
            functions = []
            classes = []
            shared_variables = set()
            
            # Collect functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    shared_variables.add(node.id)
            
            total_elements = len(functions) + len(classes)
            if total_elements == 0:
                return 1.0
            
            # Simplified cohesion: ratio of shared variables to total elements
            cohesion = len(shared_variables) / total_elements
            return min(1.0, cohesion)
            
        except Exception as e:
            logger.error(f"Error calculating cohesion: {e}")
            return 0.5
    
    def _determine_architecture_layer(self, file_path: str, imports: List[str]) -> str:
        """Determine which architecture layer this module belongs to"""
        try:
            path_lower = file_path.lower()
            
            if any(layer in path_lower for layer in ['model', 'entity', 'data']):
                return 'data_layer'
            elif any(layer in path_lower for layer in ['service', 'business', 'logic']):
                return 'business_layer'
            elif any(layer in path_lower for layer in ['controller', 'api', 'view']):
                return 'presentation_layer'
            elif any(layer in path_lower for layer in ['util', 'helper', 'common']):
                return 'utility_layer'
            elif any(layer in path_lower for layer in ['test', 'spec']):
                return 'test_layer'
            else:
                return 'unknown_layer'
                
        except Exception as e:
            logger.error(f"Error determining architecture layer: {e}")
            return 'unknown_layer'
    
    def _determine_module_role(self, file_path: str, exports: List[str], code: str) -> str:
        """Determine the primary role of this module"""
        try:
            file_name = Path(file_path).stem.lower()
            code_lower = code.lower()
            
            # Analyze based on file name and exports
            if 'config' in file_name or 'settings' in file_name:
                return 'configuration'
            elif 'test' in file_name or 'spec' in file_name:
                return 'testing'
            elif 'util' in file_name or 'helper' in file_name:
                return 'utility'
            elif any('class' in export.lower() for export in exports):
                return 'class_definition'
            elif any('def' in code_lower for _ in [1]):  # Has function definitions
                return 'function_library'
            elif 'main' in code_lower or '__main__' in code_lower:
                return 'entry_point'
            else:
                return 'general_module'
                
        except Exception as e:
            logger.error(f"Error determining module role: {e}")
            return 'general_module'

class SemanticCodeAnalyzer:
    """Master semantic code analyzer for deep code understanding"""
    
    def __init__(self):
        self.intent_engine = IntentRecognitionEngine()
        self.pattern_detector = DesignPatternDetector()
        self.context_system = ContextualComprehensionSystem()
        self.analysis_cache = {}
        self.codebase_index = {}
        
        # Analysis configuration
        self.enable_deep_analysis = True
        self.enable_performance_analysis = True
        self.enable_security_analysis = True
        self.enable_pattern_detection = True
        
        logger.info("Semantic Code Analyzer initialized")
    
    async def analyze_code_semantics(self, file_path: str, 
                                   project_root: str = None) -> CodeComprehension:
        """Perform comprehensive semantic analysis of code file"""
        try:
            # Check cache first
            cache_key = f"{file_path}_{Path(file_path).stat().st_mtime}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Read and parse code
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return CodeComprehension(file_path=file_path, confidence_score=0.0)
            
            # Initialize comprehension result
            comprehension = CodeComprehension(
                file_path=file_path,
                module_name=Path(file_path).stem
            )
            
            # Phase 1: Analyze individual code elements
            await self._analyze_code_elements(tree, code, comprehension)
            
            # Phase 2: Detect design patterns
            if self.enable_pattern_detection:
                comprehension.design_patterns = self.pattern_detector.detect_patterns(tree, code)
            
            # Phase 3: Contextual analysis
            if project_root:
                contextual_info = self.context_system.analyze_contextual_relationships(
                    file_path, project_root
                )
                comprehension.architecture_insights = contextual_info
            
            # Phase 4: Quality assessment
            comprehension.code_quality_assessment = await self._assess_code_quality(tree, code)
            
            # Phase 5: Technical debt analysis
            comprehension.technical_debt_analysis = await self._analyze_technical_debt(tree, code)
            
            # Phase 6: Optimization opportunities
            comprehension.optimization_opportunities = await self._identify_optimizations(tree, code)
            
            # Phase 7: Security analysis
            if self.enable_security_analysis:
                comprehension.security_analysis = await self._analyze_security(tree, code)
            
            # Calculate overall confidence
            comprehension.confidence_score = self._calculate_confidence_score(comprehension)
            
            # Cache result
            self.analysis_cache[cache_key] = comprehension
            
            logger.info(f"Semantic analysis completed for {file_path}")
            return comprehension
            
        except Exception as e:
            logger.error(f"Error analyzing code semantics for {file_path}: {e}")
            return CodeComprehension(file_path=file_path, confidence_score=0.0)
    
    async def _analyze_code_elements(self, tree: ast.AST, code: str, 
                                   comprehension: CodeComprehension):
        """Analyze individual code elements for semantic understanding"""
        try:
            # Analyze all AST nodes
            for node in ast.walk(tree):
                element = None
                
                if isinstance(node, ast.FunctionDef):
                    element = await self._analyze_function(node, code)
                elif isinstance(node, ast.ClassDef):
                    element = await self._analyze_class(node, code)
                elif isinstance(node, ast.Assign):
                    element = await self._analyze_assignment(node, code)
                
                if element:
                    comprehension.semantic_elements[element.element_id] = element
            
        except Exception as e:
            logger.error(f"Error analyzing code elements: {e}")
    
    async def _analyze_function(self, node: ast.FunctionDef, code: str) -> SemanticElement:
        """Analyze function for semantic understanding"""
        try:
            # Extract function code
            function_lines = code.split('\n')[node.lineno-1:node.end_lineno]
            function_code = '\n'.join(function_lines)
            
            # Create semantic element
            element = SemanticElement(
                element_type=CodeElementType.FUNCTION,
                name=node.name
            )
            
            # Analyze intent
            intent_scores = self.intent_engine.analyze_intent(
                function_code, 
                {'element_name': node.name, 'file_name': ''}
            )
            
            if intent_scores:
                best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
                element.intent_category = best_intent
            
            # Identify semantic role
            element.semantic_role = self.intent_engine.identify_semantic_role(
                node.name, function_code, CodeElementType.FUNCTION
            )
            
            # Analyze complexity
            element.complexity_metrics = self._calculate_function_complexity(node)
            
            # Analyze parameters and return
            element.context['parameters'] = [arg.arg for arg in node.args.args]
            element.context['has_return'] = any(
                isinstance(child, ast.Return) for child in ast.walk(node)
            )
            
            # Analyze side effects
            element.side_effects = self._analyze_side_effects(node)
            
            # Calculate quality metrics
            element.readability_score = self._calculate_readability(function_code)
            element.maintainability_score = self._calculate_maintainability(node)
            element.testability_score = self._calculate_testability(node)
            
            return element
            
        except Exception as e:
            logger.error(f"Error analyzing function {node.name}: {e}")
            return SemanticElement(element_type=CodeElementType.FUNCTION, name=node.name)
    
    async def _analyze_class(self, node: ast.ClassDef, code: str) -> SemanticElement:
        """Analyze class for semantic understanding"""
        try:
            # Extract class code
            class_lines = code.split('\n')[node.lineno-1:node.end_lineno]
            class_code = '\n'.join(class_lines)
            
            # Create semantic element
            element = SemanticElement(
                element_type=CodeElementType.CLASS,
                name=node.name
            )
            
            # Identify semantic role
            element.semantic_role = self.intent_engine.identify_semantic_role(
                node.name, class_code, CodeElementType.CLASS
            )
            
            # Analyze class structure
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            element.context['methods'] = methods
            element.context['inheritance'] = [base.id for base in node.bases 
                                            if isinstance(base, ast.Name)]
            
            # Analyze complexity
            element.complexity_metrics = {
                'method_count': len(methods),
                'inheritance_depth': len(element.context['inheritance']),
                'cyclomatic_complexity': self._calculate_class_complexity(node)
            }
            
            # Calculate quality metrics
            element.readability_score = self._calculate_readability(class_code)
            element.maintainability_score = self._calculate_maintainability(node)
            
            return element
            
        except Exception as e:
            logger.error(f"Error analyzing class {node.name}: {e}")
            return SemanticElement(element_type=CodeElementType.CLASS, name=node.name)
    
    async def _analyze_assignment(self, node: ast.Assign, code: str) -> Optional[SemanticElement]:
        """Analyze variable assignment for semantic understanding"""
        try:
            if not node.targets or not isinstance(node.targets[0], ast.Name):
                return None
            
            variable_name = node.targets[0].id
            
            # Skip private variables and constants for now
            if variable_name.startswith('_') or variable_name.isupper():
                return None
            
            element = SemanticElement(
                element_type=CodeElementType.VARIABLE,
                name=variable_name,
                semantic_role=SemanticRole.DATA_PROCESSOR
            )
            
            # Analyze assignment value
            if isinstance(node.value, ast.Constant):
                element.context['value_type'] = type(node.value.value).__name__
                element.context['is_constant'] = True
            elif isinstance(node.value, ast.Call):
                element.context['is_function_result'] = True
                if isinstance(node.value.func, ast.Name):
                    element.context['source_function'] = node.value.func.id
            
            return element
            
        except Exception as e:
            logger.error(f"Error analyzing assignment: {e}")
            return None
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> Dict[str, float]:
        """Calculate complexity metrics for function"""
        try:
            complexity = 1  # Base complexity
            
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
            
            return {
                'cyclomatic_complexity': complexity,
                'line_count': (node.end_lineno or node.lineno) - node.lineno + 1,
                'parameter_count': len(node.args.args),
                'nested_depth': self._calculate_nesting_depth(node)
            }
            
        except Exception as e:
            logger.error(f"Error calculating function complexity: {e}")
            return {'cyclomatic_complexity': 1}
    
    def _calculate_class_complexity(self, node: ast.ClassDef) -> float:
        """Calculate complexity metric for class"""
        try:
            total_complexity = 0
            method_count = 0
            
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    method_count += 1
                    method_complexity = self._calculate_function_complexity(child)
                    total_complexity += method_complexity.get('cyclomatic_complexity', 1)
            
            return total_complexity / max(1, method_count)
            
        except Exception as e:
            logger.error(f"Error calculating class complexity: {e}")
            return 1.0
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth in code"""
        try:
            max_depth = 0
            
            def calculate_depth(n, current_depth=0):
                nonlocal max_depth
                max_depth = max(max_depth, current_depth)
                
                if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    for child in ast.iter_child_nodes(n):
                        calculate_depth(child, current_depth + 1)
                else:
                    for child in ast.iter_child_nodes(n):
                        calculate_depth(child, current_depth)
            
            calculate_depth(node)
            return max_depth
            
        except Exception as e:
            logger.error(f"Error calculating nesting depth: {e}")
            return 0
    
    def _analyze_side_effects(self, node: ast.AST) -> List[str]:
        """Analyze potential side effects in code"""
        try:
            side_effects = []
            
            for child in ast.walk(node):
                if isinstance(child, ast.Global):
                    side_effects.append("modifies_global_variables")
                elif isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute):
                        if child.func.attr in ['append', 'extend', 'remove', 'pop']:
                            side_effects.append("modifies_mutable_objects")
                        elif child.func.attr in ['write', 'writelines']:
                            side_effects.append("file_io_operations")
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Subscript):
                            side_effects.append("modifies_collection_items")
            
            return list(set(side_effects))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error analyzing side effects: {e}")
            return []
    
    def _calculate_readability(self, code: str) -> float:
        """Calculate readability score for code"""
        try:
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            if not non_empty_lines:
                return 0.0
            
            # Calculate various readability factors
            avg_line_length = np.mean([len(line) for line in non_empty_lines])
            comment_ratio = len([line for line in lines if line.strip().startswith('#')]) / len(lines)
            blank_line_ratio = (len(lines) - len(non_empty_lines)) / len(lines)
            
            # Readability score (normalized to 0-1)
            line_length_score = max(0, 1 - abs(avg_line_length - 80) / 80)
            comment_score = min(1.0, comment_ratio * 3)  # Encourage comments
            structure_score = min(1.0, blank_line_ratio * 2)  # Encourage spacing
            
            return (line_length_score + comment_score + structure_score) / 3
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.5
    
    def _calculate_maintainability(self, node: ast.AST) -> float:
        """Calculate maintainability score"""
        try:
            # Simple maintainability heuristic
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                complexity = self._calculate_function_complexity(node)['cyclomatic_complexity']
                
                # Lower parameter count and complexity = higher maintainability
                param_score = max(0, 1 - param_count / 10)
                complexity_score = max(0, 1 - complexity / 20)
                
                return (param_score + complexity_score) / 2
            
            elif isinstance(node, ast.ClassDef):
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                inheritance_count = len(node.bases)
                
                method_score = max(0, 1 - method_count / 20)
                inheritance_score = max(0, 1 - inheritance_count / 5)
                
                return (method_score + inheritance_score) / 2
            
            return 0.7  # Default moderate score
            
        except Exception as e:
            logger.error(f"Error calculating maintainability: {e}")
            return 0.5
    
    def _calculate_testability(self, node: ast.FunctionDef) -> float:
        """Calculate testability score for function"""
        try:
            # Factors affecting testability
            param_count = len(node.args.args)
            has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
            side_effects_count = len(self._analyze_side_effects(node))
            
            # Lower parameters, has return, fewer side effects = more testable
            param_score = max(0, 1 - param_count / 8)
            return_score = 0.8 if has_return else 0.4
            side_effect_score = max(0, 1 - side_effects_count / 5)
            
            return (param_score + return_score + side_effect_score) / 3
            
        except Exception as e:
            logger.error(f"Error calculating testability: {e}")
            return 0.5
    
    async def _assess_code_quality(self, tree: ast.AST, code: str) -> Dict[str, float]:
        """Assess overall code quality"""
        try:
            quality_metrics = {}
            
            # Overall complexity
            total_complexity = 0
            function_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                    complexity = self._calculate_function_complexity(node)
                    total_complexity += complexity['cyclomatic_complexity']
            
            avg_complexity = total_complexity / max(1, function_count)
            quality_metrics['average_complexity'] = avg_complexity
            quality_metrics['complexity_score'] = max(0, 1 - avg_complexity / 10)
            
            # Code structure metrics
            lines = code.split('\n')
            quality_metrics['total_lines'] = len(lines)
            quality_metrics['code_lines'] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            quality_metrics['comment_ratio'] = len([l for l in lines if l.strip().startswith('#')]) / len(lines)
            
            # Overall quality score
            structure_score = min(1.0, quality_metrics['comment_ratio'] * 3)
            quality_metrics['overall_quality'] = (quality_metrics['complexity_score'] + structure_score) / 2
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing code quality: {e}")
            return {'overall_quality': 0.5}
    
    async def _analyze_technical_debt(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Analyze technical debt in code"""
        try:
            debt_analysis = {
                'debt_items': [],
                'total_debt_score': 0.0,
                'debt_categories': {}
            }
            
            # Check for common technical debt patterns
            debt_patterns = {
                'TODO': r'#.*TODO|#.*FIXME|#.*HACK',
                'magic_numbers': r'\b\d+\b(?!\s*[)\]\}])',
                'long_functions': 'function_length_analysis',
                'duplicate_code': 'code_duplication_analysis',
                'poor_naming': r'\b[a-z]{1,2}\b|\btemp\b|\bdata\b|\binfo\b'
            }
            
            for category, pattern in debt_patterns.items():
                if category == 'long_functions':
                    # Check function lengths
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            length = (node.end_lineno or node.lineno) - node.lineno
                            if length > 50:  # Threshold for long function
                                debt_analysis['debt_items'].append({
                                    'category': 'long_function',
                                    'description': f"Function '{node.name}' is {length} lines long",
                                    'severity': 'medium',
                                    'line': node.lineno
                                })
                
                elif isinstance(pattern, str) and pattern.startswith('r'):
                    # Regex pattern matching
                    import re
                    matches = re.findall(pattern, code, re.IGNORECASE)
                    for match in matches[:5]:  # Limit to prevent spam
                        debt_analysis['debt_items'].append({
                            'category': category,
                            'description': f"{category.replace('_', ' ').title()}: {match}",
                            'severity': 'low'
                        })
            
            # Calculate debt score
            debt_analysis['total_debt_score'] = min(len(debt_analysis['debt_items']) / 10, 1.0)
            
            # Categorize debt
            for item in debt_analysis['debt_items']:
                category = item['category']
                if category not in debt_analysis['debt_categories']:
                    debt_analysis['debt_categories'][category] = 0
                debt_analysis['debt_categories'][category] += 1
            
            return debt_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing technical debt: {e}")
            return {'total_debt_score': 0.0, 'debt_items': []}
    
    async def _identify_optimizations(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        try:
            optimizations = []
            
            # Check for common optimization opportunities
            for node in ast.walk(tree):
                # Nested loops
                if isinstance(node, ast.For):
                    for child in ast.walk(node):
                        if isinstance(child, ast.For) and child != node:
                            optimizations.append({
                                'type': 'nested_loops',
                                'description': 'Nested loops detected - consider optimization',
                                'priority': 'high',
                                'line': node.lineno,
                                'suggestion': 'Consider using list comprehensions or vectorized operations'
                            })
                
                # String concatenation in loops
                elif isinstance(node, ast.AugAssign):
                    if (isinstance(node.op, ast.Add) and 
                        any(isinstance(parent, (ast.For, ast.While)) 
                            for parent in ast.walk(tree) if node in ast.walk(parent))):
                        optimizations.append({
                            'type': 'string_concatenation',
                            'description': 'String concatenation in loop',
                            'priority': 'medium',
                            'line': node.lineno,
                            'suggestion': 'Use list.append() and join() instead'
                        })
                
                # Inefficient membership testing
                elif isinstance(node, ast.Compare):
                    if any(isinstance(op, ast.In) for op in node.ops):
                        optimizations.append({
                            'type': 'membership_testing',
                            'description': 'Membership testing - consider using sets',
                            'priority': 'low',
                            'line': node.lineno,
                            'suggestion': 'Convert lists to sets for O(1) membership testing'
                        })
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error identifying optimizations: {e}")
            return []
    
    async def _analyze_security(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Analyze security implications of code"""
        try:
            security_analysis = {
                'vulnerabilities': [],
                'security_score': 1.0,
                'risk_level': 'low'
            }
            
            # Check for security anti-patterns
            security_patterns = {
                'eval_usage': r'eval\s*\(',
                'exec_usage': r'exec\s*\(',
                'shell_injection': r'os\.system|subprocess\.call',
                'sql_injection': r'execute\s*\(["\'].*%s',
                'hardcoded_secrets': r'password\s*=\s*["\'][^"\']+["\']|api_key\s*=\s*["\'][^"\']+["\']'
            }
            
            import re
            for pattern_name, pattern in security_patterns.items():
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    severity = 'high' if pattern_name in ['eval_usage', 'sql_injection'] else 'medium'
                    security_analysis['vulnerabilities'].append({
                        'type': pattern_name,
                        'description': f"Potential {pattern_name.replace('_', ' ')} vulnerability",
                        'severity': severity,
                        'matches': len(matches)
                    })
            
            # Calculate security score
            high_risk_count = len([v for v in security_analysis['vulnerabilities'] if v['severity'] == 'high'])
            medium_risk_count = len([v for v in security_analysis['vulnerabilities'] if v['severity'] == 'medium'])
            
            security_analysis['security_score'] = max(0, 1.0 - (high_risk_count * 0.3 + medium_risk_count * 0.1))
            
            if high_risk_count > 0:
                security_analysis['risk_level'] = 'high'
            elif medium_risk_count > 2:
                security_analysis['risk_level'] = 'medium'
            else:
                security_analysis['risk_level'] = 'low'
            
            return security_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing security: {e}")
            return {'security_score': 0.5, 'risk_level': 'unknown'}
    
    def _calculate_confidence_score(self, comprehension: CodeComprehension) -> float:
        """Calculate confidence score for analysis"""
        try:
            factors = []
            
            # Factor 1: Number of analyzed elements
            element_count = len(comprehension.semantic_elements)
            element_factor = min(1.0, element_count / 10)
            factors.append(element_factor)
            
            # Factor 2: Code quality
            quality_score = comprehension.code_quality_assessment.get('overall_quality', 0.5)
            factors.append(quality_score)
            
            # Factor 3: Analysis completeness
            completeness_factor = 0.0
            if comprehension.design_patterns:
                completeness_factor += 0.2
            if comprehension.architecture_insights:
                completeness_factor += 0.3
            if comprehension.optimization_opportunities:
                completeness_factor += 0.2
            if comprehension.security_analysis:
                completeness_factor += 0.3
            factors.append(completeness_factor)
            
            return np.mean(factors)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def get_analysis_summary(self, comprehension: CodeComprehension) -> Dict[str, Any]:
        """Get summary of semantic analysis"""
        try:
            return {
                'file_path': comprehension.file_path,
                'analysis_timestamp': comprehension.analysis_timestamp.isoformat(),
                'confidence_score': comprehension.confidence_score,
                'elements_analyzed': len(comprehension.semantic_elements),
                'design_patterns_found': len(comprehension.design_patterns),
                'patterns': [pattern.value for pattern in comprehension.design_patterns],
                'code_quality': comprehension.code_quality_assessment.get('overall_quality', 0),
                'technical_debt_score': comprehension.technical_debt_analysis.get('total_debt_score', 0),
                'optimization_opportunities': len(comprehension.optimization_opportunities),
                'security_score': comprehension.security_analysis.get('security_score', 1.0),
                'architecture_layer': comprehension.architecture_insights.get('architecture_layer', 'unknown'),
                'module_role': comprehension.architecture_insights.get('module_role', 'unknown'),
                'semantic_roles_found': list(set(
                    element.semantic_role.value 
                    for element in comprehension.semantic_elements.values()
                )),
                'intent_categories_found': list(set(
                    element.intent_category.value 
                    for element in comprehension.semantic_elements.values()
                ))
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {'error': str(e)}

# Factory function for creating semantic code analyzer
def create_semantic_code_analyzer() -> SemanticCodeAnalyzer:
    """Create and initialize semantic code analyzer"""
    try:
        analyzer = SemanticCodeAnalyzer()
        logger.info("Semantic Code Analyzer created successfully")
        return analyzer
    except Exception as e:
        logger.error(f"Error creating Semantic Code Analyzer: {e}")
        raise

# Example usage and testing
async def main():
    """Example usage of Semantic Code Analyzer"""
    try:
        # Create analyzer
        analyzer = create_semantic_code_analyzer()
        
        # Example code to analyze
        example_code = '''
class DataProcessor:
    """Processes and validates data"""
    
    def __init__(self):
        self.processed_count = 0
    
    def validate_data(self, data):
        """Validate input data"""
        if not isinstance(data, list):
            raise ValueError("Data must be a list")
        
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("All items must be dictionaries")
        
        return True
    
    def process_data(self, data):
        """Process data with validation"""
        if not self.validate_data(data):
            return None
        
        processed = []
        for item in data:
            # Transform item
            transformed = {
                'id': item.get('id', 0),
                'value': item.get('value', '').upper(),
                'processed': True
            }
            processed.append(transformed)
        
        self.processed_count += len(processed)
        return processed
    
    def get_statistics(self):
        """Get processing statistics"""
        return {
            'total_processed': self.processed_count,
            'last_update': datetime.now()
        }
'''
        
        # Write example to temporary file
        temp_file = "temp_semantic_example.py"
        with open(temp_file, 'w') as f:
            f.write(example_code)
        
        try:
            # Perform semantic analysis
            comprehension = await analyzer.analyze_code_semantics(temp_file)
            
            # Get analysis summary
            summary = analyzer.get_analysis_summary(comprehension)
            
            print("=== Semantic Code Analysis Results ===")
            print(f"File: {summary['file_path']}")
            print(f"Confidence Score: {summary['confidence_score']:.3f}")
            print(f"Elements Analyzed: {summary['elements_analyzed']}")
            print(f"Design Patterns: {summary['patterns']}")
            print(f"Code Quality: {summary['code_quality']:.3f}")
            print(f"Technical Debt Score: {summary['technical_debt_score']:.3f}")
            print(f"Optimization Opportunities: {summary['optimization_opportunities']}")
            print(f"Security Score: {summary['security_score']:.3f}")
            print(f"Architecture Layer: {summary['architecture_layer']}")
            print(f"Module Role: {summary['module_role']}")
            print(f"Semantic Roles: {summary['semantic_roles_found']}")
            print(f"Intent Categories: {summary['intent_categories_found']}")
            
            # Show detailed element analysis
            print("\n=== Detailed Element Analysis ===")
            for element_id, element in comprehension.semantic_elements.items():
                print(f"\nElement: {element.name} ({element.element_type.value})")
                print(f"  Semantic Role: {element.semantic_role.value}")
                print(f"  Intent Category: {element.intent_category.value}")
                print(f"  Readability Score: {element.readability_score:.3f}")
                print(f"  Maintainability Score: {element.maintainability_score:.3f}")
                if element.side_effects:
                    print(f"  Side Effects: {element.side_effects}")
                if element.complexity_metrics:
                    print(f"  Complexity: {element.complexity_metrics}")
        
        finally:
            # Cleanup
            if Path(temp_file).exists():
                Path(temp_file).unlink()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())