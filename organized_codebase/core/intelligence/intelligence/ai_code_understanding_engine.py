"""
AI-Powered Code Understanding Engine - Agent B Implementation
Enhanced code understanding using AI/ML models for semantic analysis and intelligent insights

Key Features:
- Transformer-based code analysis
- Natural language explanations
- Intelligent refactoring suggestions
- Code similarity analysis
- Semantic pattern recognition
- Context-aware code understanding
"""

import ast
import re
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeUnderstanding:
    """Represents AI-powered understanding of code"""
    understanding_id: str = field(default_factory=lambda: f"understanding_{int(datetime.now().timestamp() * 1000000)}")
    file_path: str = ""
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    intent_analysis: Dict[str, Any] = field(default_factory=dict)
    natural_language_explanation: str = ""
    complexity_assessment: Dict[str, Any] = field(default_factory=dict)
    refactoring_suggestions: List[Dict[str, str]] = field(default_factory=list)
    similar_code_patterns: List[Dict[str, str]] = field(default_factory=list)
    confidence_score: float = 0.0
    ai_insights: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'understanding_id': self.understanding_id,
            'file_path': self.file_path,
            'semantic_analysis': self.semantic_analysis,
            'intent_analysis': self.intent_analysis,
            'natural_language_explanation': self.natural_language_explanation,
            'complexity_assessment': self.complexity_assessment,
            'refactoring_suggestions': self.refactoring_suggestions,
            'similar_code_patterns': self.similar_code_patterns,
            'confidence_score': self.confidence_score,
            'ai_insights': self.ai_insights,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SemanticPattern:
    """Represents a semantic code pattern"""
    pattern_id: str = field(default_factory=lambda: f"pattern_{int(datetime.now().timestamp() * 1000000)}")
    pattern_type: str = ""
    description: str = ""
    code_snippet: str = ""
    semantic_features: Dict[str, Any] = field(default_factory=dict)
    usage_frequency: int = 0
    complexity_score: float = 0.0
    business_relevance: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'code_snippet': self.code_snippet,
            'semantic_features': self.semantic_features,
            'usage_frequency': self.usage_frequency,
            'complexity_score': self.complexity_score,
            'business_relevance': self.business_relevance
        }


class CodeTransformerModel:
    """
    Transformer-based model for code understanding
    (Placeholder for actual transformer implementation)
    """
    
    def __init__(self):
        self.model_loaded = False
        self.vocabulary = self._build_code_vocabulary()
        self.semantic_embeddings = {}
        
    def _build_code_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary for code tokens"""
        # Common programming constructs and patterns
        code_tokens = [
            # Control structures
            'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
            'with', 'as', 'return', 'yield', 'break', 'continue', 'pass',
            
            # Object-oriented
            'class', 'def', 'self', 'super', 'property', 'staticmethod',
            'classmethod', 'init', 'new', 'del',
            
            # Data structures
            'list', 'dict', 'set', 'tuple', 'string', 'int', 'float', 'bool',
            'append', 'extend', 'remove', 'pop', 'get', 'items', 'keys', 'values',
            
            # Common patterns
            'validate', 'process', 'handle', 'create', 'update', 'delete',
            'save', 'load', 'parse', 'serialize', 'deserialize', 'transform',
            'filter', 'map', 'reduce', 'sort', 'search', 'find',
            
            # Error handling
            'error', 'exception', 'raise', 'assert', 'warning', 'debug',
            'log', 'logger', 'catch', 'handle_error'
        ]
        
        return {token: idx for idx, token in enumerate(code_tokens)}
    
    def extract_semantics(self, code: str) -> Dict[str, Any]:
        """Extract semantic meaning from code"""
        try:
            tree = ast.parse(code)
            
            semantic_features = {
                'tokens': self._tokenize_code(code),
                'ast_patterns': self._extract_ast_patterns(tree),
                'control_flow': self._analyze_control_flow(tree),
                'data_flow': self._analyze_data_flow(tree),
                'function_signatures': self._extract_function_signatures(tree),
                'class_hierarchy': self._extract_class_hierarchy(tree),
                'import_dependencies': self._extract_imports(tree),
                'complexity_metrics': self._calculate_semantic_complexity(tree)
            }
            
            return semantic_features
            
        except Exception as e:
            logger.warning(f"Error in semantic extraction: {e}")
            return {'tokens': [], 'error': str(e)}
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code for semantic analysis"""
        # Simple tokenization - would use proper tokenizer in production
        tokens = []
        
        # Extract identifiers, keywords, and operators
        token_pattern = r'\b\w+\b|[+\-*/=<>!&|]+|[(){}[\].,;:]'
        raw_tokens = re.findall(token_pattern, code)
        
        for token in raw_tokens:
            if token in self.vocabulary:
                tokens.append(token)
            elif token.isidentifier():
                tokens.append('IDENTIFIER')
            elif token.isdigit():
                tokens.append('NUMBER')
            else:
                tokens.append('OPERATOR')
        
        return tokens
    
    def _extract_ast_patterns(self, tree: ast.AST) -> List[str]:
        """Extract AST-level patterns"""
        patterns = []
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            patterns.append(node_type)
            
            # Extract specific patterns
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list:
                    patterns.append('DECORATED_FUNCTION')
                if any(isinstance(arg, ast.arg) and arg.annotation for arg in node.args.args):
                    patterns.append('TYPED_FUNCTION')
            
            elif isinstance(node, ast.ClassDef):
                if node.bases:
                    patterns.append('INHERITED_CLASS')
                if node.decorator_list:
                    patterns.append('DECORATED_CLASS')
            
            elif isinstance(node, ast.Try):
                patterns.append('ERROR_HANDLING')
            
            elif isinstance(node, ast.With):
                patterns.append('CONTEXT_MANAGER')
        
        return patterns
    
    def _analyze_control_flow(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze control flow patterns"""
        flow_patterns = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                flow_patterns['conditional'] += 1
            elif isinstance(node, ast.For):
                flow_patterns['loop'] += 1
            elif isinstance(node, ast.While):
                flow_patterns['while_loop'] += 1
            elif isinstance(node, ast.Try):
                flow_patterns['exception_handling'] += 1
            elif isinstance(node, ast.Return):
                flow_patterns['return_statement'] += 1
            elif isinstance(node, ast.Yield):
                flow_patterns['generator'] += 1
        
        return dict(flow_patterns)
    
    def _analyze_data_flow(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze data flow patterns"""
        data_flow = {
            'variables': [],
            'assignments': [],
            'function_calls': [],
            'attribute_access': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                data_flow['variables'].append(node.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        data_flow['assignments'].append(target.id)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    data_flow['function_calls'].append(node.func.id)
            elif isinstance(node, ast.Attribute):
                data_flow['attribute_access'].append(node.attr)
        
        return data_flow
    
    def _extract_function_signatures(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function signatures and metadata"""
        signatures = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                signature = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'returns': node.returns is not None,
                    'async': isinstance(node, ast.AsyncFunctionDef),
                    'decorators': len(node.decorator_list),
                    'docstring': ast.get_docstring(node) is not None
                }
                signatures.append(signature)
        
        return signatures
    
    def _extract_class_hierarchy(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class hierarchy information"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    'methods': [item.name for item in node.body if isinstance(item, ast.FunctionDef)],
                    'decorators': len(node.decorator_list),
                    'docstring': ast.get_docstring(node) is not None
                }
                classes.append(class_info)
        
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Extract import dependencies"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': node.module or '',
                        'name': alias.name,
                        'alias': alias.asname
                    })
        
        return imports
    
    def _calculate_semantic_complexity(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate semantic complexity metrics"""
        metrics = {
            'ast_depth': self._calculate_ast_depth(tree),
            'cognitive_complexity': self._calculate_cognitive_complexity(tree),
            'halstead_complexity': self._calculate_halstead_complexity(tree),
            'semantic_coupling': self._calculate_semantic_coupling(tree)
        }
        
        return metrics
    
    def _calculate_ast_depth(self, tree: ast.AST) -> int:
        """Calculate maximum AST depth"""
        def depth(node):
            if not hasattr(node, '_fields'):
                return 0
            
            max_child_depth = 0
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            max_child_depth = max(max_child_depth, depth(item))
                elif isinstance(value, ast.AST):
                    max_child_depth = max(max_child_depth, depth(value))
            
            return 1 + max_child_depth
        
        return depth(tree)
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity score"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_halstead_complexity(self, tree: ast.AST) -> float:
        """Calculate Halstead complexity metrics"""
        operators = set()
        operands = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
                operators.add(type(node).__name__)
            elif isinstance(node, ast.Name):
                operands.add(node.id)
            elif isinstance(node, ast.Constant):
                operands.add(str(node.value))
        
        n1 = len(operators)  # Number of distinct operators
        n2 = len(operands)   # Number of distinct operands
        
        if n1 + n2 == 0:
            return 0.0
        
        # Simplified Halstead complexity
        return (n1 + n2) * math.log2(n1 + n2) if n1 + n2 > 1 else 0.0
    
    def _calculate_semantic_coupling(self, tree: ast.AST) -> float:
        """Calculate semantic coupling score"""
        external_references = 0
        total_references = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                total_references += 1
                if isinstance(node.func, ast.Attribute):
                    external_references += 1
            elif isinstance(node, ast.Attribute):
                total_references += 1
                external_references += 1
        
        return external_references / max(total_references, 1)


class UnderstandingModel:
    """
    Model for analyzing code intent and purpose
    """
    
    def __init__(self):
        self.intent_patterns = self._load_intent_patterns()
        self.purpose_classifiers = self._load_purpose_classifiers()
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns"""
        return {
            'data_processing': [
                'process', 'transform', 'convert', 'parse', 'serialize',
                'filter', 'map', 'reduce', 'aggregate', 'normalize'
            ],
            'validation': [
                'validate', 'check', 'verify', 'ensure', 'assert',
                'is_valid', 'has_valid', 'check_format'
            ],
            'business_logic': [
                'calculate', 'compute', 'determine', 'evaluate',
                'apply_rules', 'process_order', 'handle_payment'
            ],
            'data_access': [
                'get', 'fetch', 'retrieve', 'load', 'read',
                'save', 'store', 'persist', 'write', 'update'
            ],
            'communication': [
                'send', 'receive', 'notify', 'publish', 'subscribe',
                'emit', 'listen', 'broadcast', 'transmit'
            ],
            'security': [
                'authenticate', 'authorize', 'encrypt', 'decrypt',
                'hash', 'verify_token', 'check_permission'
            ],
            'error_handling': [
                'handle_error', 'catch', 'recover', 'retry',
                'fallback', 'log_error', 'raise_exception'
            ]
        }
    
    def _load_purpose_classifiers(self) -> Dict[str, List[str]]:
        """Load purpose classification patterns"""
        return {
            'utility': ['util', 'helper', 'common', 'shared', 'tool'],
            'service': ['service', 'manager', 'handler', 'processor'],
            'model': ['model', 'entity', 'domain', 'data'],
            'controller': ['controller', 'view', 'endpoint', 'api'],
            'configuration': ['config', 'settings', 'constants', 'env'],
            'test': ['test', 'spec', 'mock', 'fixture'],
            'infrastructure': ['db', 'database', 'cache', 'queue', 'logger']
        }
    
    def analyze_intent(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code intent and purpose"""
        try:
            tree = ast.parse(code)
            
            intent_analysis = {
                'primary_intent': self._classify_primary_intent(tree, code),
                'secondary_intents': self._identify_secondary_intents(tree, code),
                'purpose_classification': self._classify_purpose(context.get('file_path', ''), tree),
                'abstraction_level': self._determine_abstraction_level(tree),
                'design_patterns': self._identify_design_patterns(tree),
                'architectural_role': self._determine_architectural_role(tree, context)
            }
            
            return intent_analysis
            
        except Exception as e:
            logger.warning(f"Error in intent analysis: {e}")
            return {'error': str(e)}
    
    def _classify_primary_intent(self, tree: ast.AST, code: str) -> str:
        """Classify the primary intent of the code"""
        intent_scores = defaultdict(int)
        
        # Analyze function names
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name.lower()
                for intent, keywords in self.intent_patterns.items():
                    if any(keyword in function_name for keyword in keywords):
                        intent_scores[intent] += 2
        
        # Analyze code content
        code_lower = code.lower()
        for intent, keywords in self.intent_patterns.items():
            for keyword in keywords:
                intent_scores[intent] += code_lower.count(keyword)
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'general_purpose'
    
    def _identify_secondary_intents(self, tree: ast.AST, code: str) -> List[str]:
        """Identify secondary intents in the code"""
        intent_scores = defaultdict(int)
        
        # Similar to primary intent but return multiple
        code_lower = code.lower()
        for intent, keywords in self.intent_patterns.items():
            for keyword in keywords:
                intent_scores[intent] += code_lower.count(keyword)
        
        # Return top 3 intents (excluding primary)
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        return [intent for intent, score in sorted_intents[1:4] if score > 0]
    
    def _classify_purpose(self, file_path: str, tree: ast.AST) -> str:
        """Classify the purpose/role of the code"""
        file_name = Path(file_path).name.lower() if file_path else ''
        
        for purpose, keywords in self.purpose_classifiers.items():
            if any(keyword in file_name for keyword in keywords):
                return purpose
        
        # Analyze code structure
        has_classes = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        
        if has_classes and not has_functions:
            return 'model'
        elif has_functions and not has_classes:
            return 'utility'
        else:
            return 'mixed'
    
    def _determine_abstraction_level(self, tree: ast.AST) -> str:
        """Determine the abstraction level of the code"""
        # Analyze code characteristics
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        import_count = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
        
        # Simple heuristic for abstraction level
        complexity_score = function_count + class_count * 2 + import_count * 0.5
        
        if complexity_score > 20:
            return 'high'
        elif complexity_score > 10:
            return 'medium'
        else:
            return 'low'
    
    def _identify_design_patterns(self, tree: ast.AST) -> List[str]:
        """Identify design patterns in the code"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Singleton pattern
                if any(isinstance(item, ast.FunctionDef) and item.name == '__new__' for item in node.body):
                    patterns.append('singleton')
                
                # Factory pattern
                method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
                if any(name.startswith('create_') for name in method_names):
                    patterns.append('factory')
                
                # Observer pattern
                observer_methods = {'subscribe', 'unsubscribe', 'notify', 'update'}
                if any(name in observer_methods for name in method_names):
                    patterns.append('observer')
        
        return patterns
    
    def _determine_architectural_role(self, tree: ast.AST, context: Dict[str, Any]) -> str:
        """Determine the architectural role of the code"""
        file_path = context.get('file_path', '').lower()
        
        # Analyze file path structure
        if 'model' in file_path or 'entity' in file_path:
            return 'data_layer'
        elif 'service' in file_path or 'business' in file_path:
            return 'business_layer'
        elif 'controller' in file_path or 'api' in file_path or 'view' in file_path:
            return 'presentation_layer'
        elif 'repository' in file_path or 'dao' in file_path:
            return 'data_access_layer'
        else:
            return 'utility_layer'


class ExplanationGenerator:
    """
    Generates natural language explanations of code
    """
    
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates"""
        return {
            'function': "This function '{name}' {purpose} by {implementation}. It takes {params} and returns {returns}.",
            'class': "This class '{name}' represents {concept} with {methods} methods and {attributes} attributes.",
            'process': "This code implements a {process_type} process that {description}.",
            'validation': "This code validates {subject} by checking {conditions}.",
            'calculation': "This code calculates {result} using {method} approach.",
            'data_access': "This code {operation} data {location} using {mechanism}."
        }
    
    def generate_explanation(self, code: str, understanding: Dict[str, Any]) -> str:
        """Generate natural language explanation of code"""
        try:
            tree = ast.parse(code)
            explanations = []
            
            # Explain functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_explanation = self._explain_function(node, understanding)
                    explanations.append(func_explanation)
                elif isinstance(node, ast.ClassDef):
                    class_explanation = self._explain_class(node, understanding)
                    explanations.append(class_explanation)
            
            # Generate overall explanation
            overall_explanation = self._generate_overall_explanation(understanding, explanations)
            
            return overall_explanation
            
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def _explain_function(self, node: ast.FunctionDef, understanding: Dict[str, Any]) -> str:
        """Generate explanation for a function"""
        name = node.name
        params = [arg.arg for arg in node.args.args] if node.args.args else ["no parameters"]
        has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
        
        # Determine purpose from name and intent
        intent = understanding.get('intent_analysis', {}).get('primary_intent', 'general processing')
        
        purpose_map = {
            'validation': 'validates input data',
            'calculation': 'performs calculations',
            'data_processing': 'processes data',
            'data_access': 'accesses data',
            'business_logic': 'implements business rules',
            'communication': 'handles communication',
            'security': 'manages security',
            'error_handling': 'handles errors'
        }
        
        purpose = purpose_map.get(intent, 'performs operations')
        
        explanation = f"The function '{name}' {purpose}. "
        
        if len(params) > 0:
            explanation += f"It accepts {len(params)} parameter(s): {', '.join(params[:3])}{'...' if len(params) > 3 else ''}. "
        
        if has_return:
            explanation += "It returns a result. "
        
        return explanation
    
    def _explain_class(self, node: ast.ClassDef, understanding: Dict[str, Any]) -> str:
        """Generate explanation for a class"""
        name = node.name
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef) and not item.name.startswith('_')]
        
        explanation = f"The class '{name}' "
        
        # Determine class purpose
        if any('model' in name.lower() or 'entity' in name.lower() for _ in [name]):
            explanation += "represents a data model or entity "
        elif any('service' in name.lower() or 'manager' in name.lower() for _ in [name]):
            explanation += "provides services or manages operations "
        elif any('controller' in name.lower() or 'handler' in name.lower() for _ in [name]):
            explanation += "controls or handles requests "
        else:
            explanation += "implements functionality "
        
        if methods:
            explanation += f"with {len(methods)} public method(s): {', '.join(methods[:3])}{'...' if len(methods) > 3 else ''}."
        
        return explanation
    
    def _generate_overall_explanation(self, understanding: Dict[str, Any], component_explanations: List[str]) -> str:
        """Generate overall explanation of the code"""
        intent = understanding.get('intent_analysis', {}).get('primary_intent', 'general_purpose')
        purpose = understanding.get('intent_analysis', {}).get('purpose_classification', 'utility')
        
        overall = f"This code primarily serves a {intent} purpose and functions as a {purpose} component. "
        
        if component_explanations:
            overall += "It includes the following components:\n\n"
            for i, explanation in enumerate(component_explanations, 1):
                overall += f"{i}. {explanation}\n"
        
        complexity = understanding.get('complexity_assessment', {})
        if complexity.get('cognitive_complexity', 0) > 10:
            overall += "\nNote: This code has relatively high complexity and may benefit from refactoring."
        
        return overall


class AICodeUnderstandingEngine:
    """
    AI-Powered Code Understanding Engine - Agent B Implementation
    
    Provides comprehensive AI-driven code analysis with natural language explanations,
    intelligent insights, and refactoring suggestions.
    """
    
    def __init__(self):
        self.transformer_model = CodeTransformerModel()
        self.understanding_model = UnderstandingModel()
        self.explanation_generator = ExplanationGenerator()
        self.context_analyzer = ContextAnalyzer()
        
        # Pattern database for similarity analysis
        self.known_patterns = []
        self.analysis_cache = {}
    
    def understand_code(self, code: str, context: Dict[str, Any]) -> CodeUnderstanding:
        """
        Deep understanding of code using AI models
        """
        try:
            # Generate cache key
            cache_key = hashlib.md5((code + str(context)).encode()).hexdigest()
            
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Extract semantic meaning
            semantic_analysis = self.transformer_model.extract_semantics(code)
            
            # Analyze intent and purpose
            intent_analysis = self.understanding_model.analyze_intent(code, context)
            
            # Generate natural language explanation
            understanding_data = {
                'semantic_analysis': semantic_analysis,
                'intent_analysis': intent_analysis
            }
            natural_explanation = self.explanation_generator.generate_explanation(code, understanding_data)
            
            # Assess complexity
            complexity_assessment = self._assess_complexity(semantic_analysis, intent_analysis)
            
            # Generate refactoring suggestions
            refactoring_suggestions = self._generate_refactoring_suggestions(code, semantic_analysis, complexity_assessment)
            
            # Find similar code patterns
            similar_patterns = self._find_similar_patterns(semantic_analysis)
            
            # Generate AI insights
            ai_insights = self._generate_ai_insights(semantic_analysis, intent_analysis, complexity_assessment)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(semantic_analysis, intent_analysis)
            
            # Create understanding object
            understanding = CodeUnderstanding(
                file_path=context.get('file_path', ''),
                semantic_analysis=semantic_analysis,
                intent_analysis=intent_analysis,
                natural_language_explanation=natural_explanation,
                complexity_assessment=complexity_assessment,
                refactoring_suggestions=refactoring_suggestions,
                similar_code_patterns=similar_patterns,
                confidence_score=confidence_score,
                ai_insights=ai_insights
            )
            
            # Cache result
            self.analysis_cache[cache_key] = understanding
            
            return understanding
            
        except Exception as e:
            logger.error(f"Error in code understanding: {e}")
            return CodeUnderstanding(
                file_path=context.get('file_path', ''),
                natural_language_explanation=f"Unable to analyze code: {str(e)}",
                confidence_score=0.0
            )
    
    def _assess_complexity(self, semantic_analysis: Dict[str, Any], intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess code complexity from multiple dimensions"""
        complexity_metrics = semantic_analysis.get('complexity_metrics', {})
        
        return {
            'cognitive_complexity': complexity_metrics.get('cognitive_complexity', 0),
            'semantic_complexity': complexity_metrics.get('semantic_coupling', 0.0),
            'structural_complexity': complexity_metrics.get('ast_depth', 0),
            'halstead_complexity': complexity_metrics.get('halstead_complexity', 0.0),
            'intent_complexity': self._calculate_intent_complexity(intent_analysis),
            'overall_complexity': self._calculate_overall_complexity(complexity_metrics, intent_analysis)
        }
    
    def _calculate_intent_complexity(self, intent_analysis: Dict[str, Any]) -> float:
        """Calculate complexity based on intent analysis"""
        primary_intent = intent_analysis.get('primary_intent', '')
        secondary_intents = intent_analysis.get('secondary_intents', [])
        
        intent_complexity_map = {
            'business_logic': 0.8,
            'security': 0.7,
            'data_processing': 0.6,
            'validation': 0.5,
            'communication': 0.5,
            'data_access': 0.4,
            'error_handling': 0.3,
            'general_purpose': 0.2
        }
        
        primary_score = intent_complexity_map.get(primary_intent, 0.5)
        secondary_score = sum(intent_complexity_map.get(intent, 0.3) for intent in secondary_intents) * 0.2
        
        return min(primary_score + secondary_score, 1.0)
    
    def _calculate_overall_complexity(self, complexity_metrics: Dict[str, Any], intent_analysis: Dict[str, Any]) -> str:
        """Calculate overall complexity rating"""
        cognitive = complexity_metrics.get('cognitive_complexity', 0)
        semantic = complexity_metrics.get('semantic_coupling', 0.0)
        intent_complexity = self._calculate_intent_complexity(intent_analysis)
        
        overall_score = (cognitive * 0.4 + semantic * 10 * 0.3 + intent_complexity * 10 * 0.3)
        
        if overall_score > 15:
            return 'high'
        elif overall_score > 8:
            return 'medium'
        else:
            return 'low'
    
    def _generate_refactoring_suggestions(self, code: str, semantic_analysis: Dict[str, Any], complexity_assessment: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate intelligent refactoring suggestions"""
        suggestions = []
        
        try:
            tree = ast.parse(code)
            
            # Check for long functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                        length = node.end_lineno - node.lineno
                        if length > 30:
                            suggestions.append({
                                'type': 'function_length',
                                'description': f"Function '{node.name}' is {length} lines long. Consider breaking it into smaller functions.",
                                'priority': 'medium',
                                'impact': 'readability'
                            })
            
            # Check cognitive complexity
            if complexity_assessment.get('cognitive_complexity', 0) > 10:
                suggestions.append({
                    'type': 'cognitive_complexity',
                    'description': 'High cognitive complexity detected. Consider simplifying control flow and reducing nesting.',
                    'priority': 'high',
                    'impact': 'maintainability'
                })
            
            # Check semantic coupling
            if complexity_assessment.get('semantic_complexity', 0) > 0.7:
                suggestions.append({
                    'type': 'coupling',
                    'description': 'High coupling detected. Consider using dependency injection or interfaces.',
                    'priority': 'medium',
                    'impact': 'modularity'
                })
            
            # Check for missing documentation
            functions_without_docstrings = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    if not ast.get_docstring(node):
                        functions_without_docstrings.append(node.name)
            
            if functions_without_docstrings:
                suggestions.append({
                    'type': 'documentation',
                    'description': f"Missing docstrings for functions: {', '.join(functions_without_docstrings[:3])}{'...' if len(functions_without_docstrings) > 3 else ''}",
                    'priority': 'low',
                    'impact': 'documentation'
                })
            
        except Exception as e:
            logger.warning(f"Error generating refactoring suggestions: {e}")
        
        return suggestions
    
    def _find_similar_patterns(self, semantic_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find similar code patterns in the pattern database"""
        similar_patterns = []
        
        current_patterns = semantic_analysis.get('ast_patterns', [])
        tokens = semantic_analysis.get('tokens', [])
        
        # Simple similarity detection (would use more sophisticated methods in production)
        for known_pattern in self.known_patterns:
            similarity_score = self._calculate_pattern_similarity(current_patterns, known_pattern.semantic_features.get('ast_patterns', []))
            
            if similarity_score > 0.7:
                similar_patterns.append({
                    'pattern_id': known_pattern.pattern_id,
                    'description': known_pattern.description,
                    'similarity_score': similarity_score,
                    'pattern_type': known_pattern.pattern_type
                })
        
        return sorted(similar_patterns, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    def _calculate_pattern_similarity(self, patterns1: List[str], patterns2: List[str]) -> float:
        """Calculate similarity between two pattern lists"""
        if not patterns1 or not patterns2:
            return 0.0
        
        set1 = set(patterns1)
        set2 = set(patterns2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_ai_insights(self, semantic_analysis: Dict[str, Any], intent_analysis: Dict[str, Any], complexity_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered insights about the code"""
        insights = []
        
        # Performance insights
        if semantic_analysis.get('control_flow', {}).get('loop', 0) > 3:
            insights.append({
                'type': 'performance',
                'category': 'optimization',
                'description': 'Multiple loops detected. Consider optimizing for better performance.',
                'confidence': 0.7,
                'actionable': True
            })
        
        # Architecture insights
        abstraction_level = intent_analysis.get('abstraction_level', 'medium')
        if abstraction_level == 'low':
            insights.append({
                'type': 'architecture',
                'category': 'design',
                'description': 'Low abstraction level detected. Consider creating higher-level abstractions.',
                'confidence': 0.6,
                'actionable': True
            })
        
        # Security insights
        if intent_analysis.get('primary_intent') == 'security':
            insights.append({
                'type': 'security',
                'category': 'compliance',
                'description': 'Security-related code detected. Ensure proper security review and testing.',
                'confidence': 0.9,
                'actionable': True
            })
        
        # Maintainability insights
        if complexity_assessment.get('overall_complexity') == 'high':
            insights.append({
                'type': 'maintainability',
                'category': 'quality',
                'description': 'High complexity may impact maintainability. Consider refactoring.',
                'confidence': 0.8,
                'actionable': True
            })
        
        return insights
    
    def _calculate_confidence_score(self, semantic_analysis: Dict[str, Any], intent_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.5  # Base score
        
        # Increase confidence based on available data
        if semantic_analysis.get('tokens'):
            score += 0.1
        
        if semantic_analysis.get('ast_patterns'):
            score += 0.1
        
        if intent_analysis.get('primary_intent') != 'general_purpose':
            score += 0.1
        
        if semantic_analysis.get('function_signatures'):
            score += 0.1
        
        if semantic_analysis.get('complexity_metrics'):
            score += 0.1
        
        return min(score, 1.0)
    
    def add_pattern_to_database(self, pattern: SemanticPattern):
        """Add a new pattern to the pattern database"""
        self.known_patterns.append(pattern)
    
    def analyze_codebase_patterns(self, codebase_path: str) -> List[SemanticPattern]:
        """Analyze entire codebase to extract common patterns"""
        patterns = []
        
        try:
            path = Path(codebase_path)
            python_files = list(path.rglob("*.py"))
            
            pattern_frequency = defaultdict(int)
            
            for file_path in python_files[:50]:  # Limit analysis for performance
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    understanding = self.understand_code(code, {'file_path': str(file_path)})
                    ast_patterns = understanding.semantic_analysis.get('ast_patterns', [])
                    
                    for pattern in ast_patterns:
                        pattern_frequency[pattern] += 1
                        
                except Exception as e:
                    logger.debug(f"Error analyzing {file_path}: {e}")
            
            # Create semantic patterns for common patterns
            for pattern_name, frequency in pattern_frequency.items():
                if frequency > 5:  # Only include frequent patterns
                    semantic_pattern = SemanticPattern(
                        pattern_type=pattern_name,
                        description=f"Common pattern: {pattern_name}",
                        usage_frequency=frequency,
                        semantic_features={'ast_patterns': [pattern_name]},
                        business_relevance='medium'
                    )
                    patterns.append(semantic_pattern)
                    self.add_pattern_to_database(semantic_pattern)
            
        except Exception as e:
            logger.error(f"Error analyzing codebase patterns: {e}")
        
        return patterns


class ContextAnalyzer:
    """
    Analyzes code context for better understanding
    """
    
    def __init__(self):
        self.context_cache = {}
    
    def analyze_context(self, file_path: str, surrounding_files: List[str] = None) -> Dict[str, Any]:
        """Analyze context around a code file"""
        context = {
            'file_context': self._analyze_file_context(file_path),
            'directory_context': self._analyze_directory_context(file_path),
            'project_context': self._analyze_project_context(file_path),
            'dependency_context': self._analyze_dependencies(file_path)
        }
        
        return context
    
    def _analyze_file_context(self, file_path: str) -> Dict[str, Any]:
        """Analyze immediate file context"""
        path = Path(file_path)
        
        return {
            'filename': path.name,
            'extension': path.suffix,
            'directory': path.parent.name,
            'size_category': self._categorize_file_size(file_path)
        }
    
    def _analyze_directory_context(self, file_path: str) -> Dict[str, Any]:
        """Analyze directory structure context"""
        path = Path(file_path)
        parent_dir = path.parent
        
        sibling_files = [f.name for f in parent_dir.glob("*.py")] if parent_dir.exists() else []
        
        return {
            'directory_name': parent_dir.name,
            'sibling_files': sibling_files,
            'file_count': len(sibling_files),
            'directory_type': self._classify_directory_type(parent_dir.name)
        }
    
    def _analyze_project_context(self, file_path: str) -> Dict[str, Any]:
        """Analyze broader project context"""
        path = Path(file_path)
        
        # Look for project indicators
        project_root = self._find_project_root(path)
        
        return {
            'project_root': str(project_root) if project_root else None,
            'project_type': self._determine_project_type(project_root) if project_root else 'unknown'
        }
    
    def _analyze_dependencies(self, file_path: str) -> Dict[str, Any]:
        """Analyze file dependencies"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return {
                'total_imports': len(imports),
                'external_libraries': [imp for imp in imports if '.' not in imp and imp not in ['os', 'sys', 're', 'json']],
                'standard_library': [imp for imp in imports if imp in ['os', 'sys', 're', 'json', 'datetime', 'math']],
                'local_imports': [imp for imp in imports if imp.startswith('.')]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _categorize_file_size(self, file_path: str) -> str:
        """Categorize file by size"""
        try:
            size = Path(file_path).stat().st_size
            if size < 1000:
                return 'small'
            elif size < 10000:
                return 'medium'
            else:
                return 'large'
        except:
            return 'unknown'
    
    def _classify_directory_type(self, dir_name: str) -> str:
        """Classify directory type based on name"""
        dir_name_lower = dir_name.lower()
        
        if dir_name_lower in ['models', 'entities', 'domain']:
            return 'data_layer'
        elif dir_name_lower in ['services', 'business', 'logic']:
            return 'business_layer'
        elif dir_name_lower in ['controllers', 'views', 'api', 'handlers']:
            return 'presentation_layer'
        elif dir_name_lower in ['utils', 'helpers', 'common']:
            return 'utility_layer'
        elif dir_name_lower in ['tests', 'test']:
            return 'test_layer'
        else:
            return 'general'
    
    def _find_project_root(self, path: Path) -> Optional[Path]:
        """Find project root directory"""
        current = path.parent if path.is_file() else path
        
        while current.parent != current:  # Stop at filesystem root
            # Look for common project indicators
            if any((current / indicator).exists() for indicator in ['setup.py', 'pyproject.toml', 'requirements.txt', '.git']):
                return current
            current = current.parent
        
        return None
    
    def _determine_project_type(self, project_root: Path) -> str:
        """Determine project type based on project structure"""
        if (project_root / 'setup.py').exists() or (project_root / 'pyproject.toml').exists():
            return 'python_package'
        elif (project_root / 'manage.py').exists():
            return 'django_project'
        elif (project_root / 'app.py').exists() or (project_root / 'main.py').exists():
            return 'web_application'
        elif (project_root / 'Dockerfile').exists():
            return 'containerized_application'
        else:
            return 'general_python_project'


# Export classes
__all__ = [
    'AICodeUnderstandingEngine', 'CodeUnderstanding', 'SemanticPattern',
    'CodeTransformerModel', 'UnderstandingModel', 'ExplanationGenerator', 'ContextAnalyzer'
]


# Factory function for easy instantiation
def create_ai_code_understanding_engine() -> AICodeUnderstandingEngine:
    """Factory function to create a configured AI Code Understanding Engine"""
    return AICodeUnderstandingEngine()


if __name__ == "__main__":
    # Example usage
    engine = create_ai_code_understanding_engine()
    
    # Example code to analyze
    sample_code = '''
def calculate_order_total(items, tax_rate=0.08, discount=0.0):
    """Calculate the total cost of an order including tax and discount."""
    subtotal = sum(item.price * item.quantity for item in items)
    
    if discount > 0:
        subtotal *= (1 - discount)
    
    tax = subtotal * tax_rate
    total = subtotal + tax
    
    return total
'''
    
    try:
        understanding = engine.understand_code(sample_code, {'file_path': 'example.py'})
        
        print("AI Code Understanding Results:")
        print(f"Primary Intent: {understanding.intent_analysis.get('primary_intent', 'Unknown')}")
        print(f"Confidence Score: {understanding.confidence_score:.2f}")
        print(f"\nExplanation:\n{understanding.natural_language_explanation}")
        
        if understanding.refactoring_suggestions:
            print(f"\nRefactoring Suggestions:")
            for suggestion in understanding.refactoring_suggestions:
                print(f"- {suggestion['description']}")
        
        # Save analysis results
        with open("ai_code_understanding.json", "w") as f:
            json.dump(understanding.to_dict(), f, indent=2, ensure_ascii=False)
        
        print("\nAnalysis complete! Results saved to ai_code_understanding.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")