"""
Predictive Intelligence Core - Master Orchestration and Integration Engine
==========================================================================

Master orchestration system for predictive code intelligence with comprehensive
natural language integration, documentation generation, and evolution prediction.
Implements enterprise-grade intelligence coordination and analysis workflows.

This module serves as the main coordination hub for all predictive intelligence
capabilities, integrating evolution prediction, language bridging, and documentation.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: predictive_intelligence_core.py (350 lines)
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import asdict

from .predictive_types import (
    PredictionType, PredictionConfidence, LanguageBridgeDirection, DocumentationType,
    CodePrediction, NaturalLanguageTranslation, DocumentationGeneration,
    CodeEvolutionPattern, MaintenanceBurdenAnalysis, PredictiveMetrics,
    CodeIntelligenceContext
)
from .code_predictor import CodeEvolutionPredictor
from .language_bridge import NaturalLanguageBridge
from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Enterprise documentation generation with intelligent templates"""
    
    def __init__(self):
        self.docstring_templates = {
            'function': '''"""
{summary}

{description}

Args:
{args}

Returns:
{returns}

Raises:
{raises}

Example:
{example}
"""''',
            'class': '''"""
{summary}

{description}

Attributes:
{attributes}

Example:
{example}
"""''',
            'module': '''"""
{summary}

{description}

Classes:
{classes}

Functions:
{functions}
"""'''
        }
        
        self.quality_thresholds = {
            'minimum_length': 50,
            'ideal_length_range': (100, 500),
            'completeness_factors': ['args', 'returns', 'example'],
            'clarity_indicators': ['description', 'summary', 'example']
        }
    
    def generate_documentation(self, code: str, doc_type: DocumentationType,
                             target_element: str = "") -> DocumentationGeneration:
        """Generate comprehensive documentation with quality assessment"""
        try:
            doc = DocumentationGeneration(
                documentation_type=doc_type,
                target_element=target_element
            )
            
            # Generate content based on type
            if doc_type == DocumentationType.MODULE_DOCSTRING:
                doc.generated_content = self._generate_module_docstring(code)
            elif doc_type == DocumentationType.API_DOCUMENTATION:
                doc.generated_content = self._generate_api_documentation(code)
            elif doc_type == DocumentationType.INLINE_COMMENTS:
                doc.generated_content = self._generate_inline_comments(code)
            else:
                doc.generated_content = self._generate_general_documentation(code)
            
            # Assess quality metrics
            doc.content_quality = self._assess_content_quality(doc)
            doc.completeness_score = self._assess_completeness(doc)
            doc.clarity_score = self._assess_clarity(doc)
            doc.technical_accuracy = self._assess_technical_accuracy(doc, code)
            doc.formatting_compliance = self._check_formatting_compliance(doc)
            
            return doc
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return DocumentationGeneration(
                documentation_type=doc_type,
                generated_content=f"Documentation generation error: {str(e)}",
                content_quality=0.0
            )
    
    def _generate_module_docstring(self, code: str) -> str:
        """Generate comprehensive module documentation"""
        try:
            import ast
            tree = ast.parse(code)
            
            # Extract module components
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) 
                        if isinstance(node, ast.FunctionDef) and 
                        not any(node in class_node.body for class_node in ast.walk(tree) 
                               if isinstance(class_node, ast.ClassDef))]
            
            # Generate summary and description
            summary = "Module providing comprehensive functionality"
            if len(classes) > len(functions):
                summary = f"Object-oriented module with {len(classes)} main classes"
            elif len(functions) > len(classes):
                summary = f"Functional module with {len(functions)} utility functions"
            
            description = self._infer_module_purpose(code, classes, functions)
            classes_desc = "\n".join([f"    {cls}: Main class for {cls.lower()} operations" 
                                    for cls in classes[:5]])
            functions_desc = "\n".join([f"    {func}: Utility function for {func.lower()} operations" 
                                      for func in functions[:5]])
            
            return self.docstring_templates['module'].format(
                summary=summary,
                description=description,
                classes=classes_desc or "    None",
                functions=functions_desc or "    None"
            )
            
        except Exception as e:
            logger.error(f"Error generating module docstring: {e}")
            return "Module documentation could not be generated due to parsing errors."
    
    def _generate_api_documentation(self, code: str) -> str:
        """Generate API documentation with endpoint analysis"""
        try:
            # Analyze code for API patterns
            api_patterns = {
                'routes': code.count('@app.route') + code.count('@router.'),
                'endpoints': code.count('def ') if 'app' in code or 'api' in code else 0,
                'methods': len([m for m in ['GET', 'POST', 'PUT', 'DELETE'] if m in code])
            }
            
            if api_patterns['routes'] > 0:
                return f"""
# API Documentation

This module provides a REST API with {api_patterns['routes']} endpoints.

## Endpoints
- HTTP methods supported: {api_patterns['methods']}
- Route handlers: {api_patterns['endpoints']}

## Usage
See individual function documentation for endpoint details.

## Authentication
Refer to authentication middleware for security requirements.
"""
            else:
                return """
# API Documentation

This module provides API functionality.
Refer to individual functions for specific endpoint documentation.
"""
                
        except Exception as e:
            logger.error(f"Error generating API documentation: {e}")
            return "API documentation could not be generated."
    
    def _generate_inline_comments(self, code: str) -> str:
        """Generate suggestions for inline comments"""
        try:
            suggestions = []
            lines = code.split('\n')
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    if 'if ' in stripped and ':' in stripped:
                        suggestions.append(f"Line {i}: Consider adding comment explaining the condition")
                    elif 'for ' in stripped and ':' in stripped:
                        suggestions.append(f"Line {i}: Consider documenting the iteration purpose")
                    elif '=' in stripped and 'def ' not in stripped and 'class ' not in stripped:
                        if len(stripped) > 50:
                            suggestions.append(f"Line {i}: Complex assignment may need explanation")
            
            if suggestions:
                return "Inline Comment Suggestions:\n" + "\n".join(suggestions[:10])
            else:
                return "Code appears well-documented or self-explanatory."
                
        except Exception as e:
            logger.error(f"Error generating inline comments: {e}")
            return "Inline comment analysis could not be completed."
    
    def _generate_general_documentation(self, code: str) -> str:
        """Generate general documentation for unspecified types"""
        try:
            import ast
            tree = ast.parse(code)
            
            # Basic analysis
            total_lines = len(code.split('\n'))
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            
            return f"""
# Code Documentation

## Overview
This code module contains {total_lines} lines with {classes} classes and {functions} functions.

## Structure
- Classes: {classes}
- Functions: {functions}
- Complexity: {'High' if total_lines > 200 else 'Medium' if total_lines > 50 else 'Low'}

## Purpose
{self._infer_module_purpose(code, [], [])}

## Usage
Refer to individual class and function documentation for specific usage instructions.
"""
                
        except Exception as e:
            logger.error(f"Error generating general documentation: {e}")
            return "General documentation could not be generated."
    
    def _infer_module_purpose(self, code: str, classes: List[str], functions: List[str]) -> str:
        """Infer module purpose from code analysis"""
        try:
            code_lower = code.lower()
            
            # Domain-specific patterns
            if any(term in code_lower for term in ['predict', 'forecast', 'analyze']):
                return "Provides predictive analysis and forecasting capabilities."
            elif any(term in code_lower for term in ['process', 'transform', 'convert']):
                return "Handles data processing and transformation operations."
            elif any(term in code_lower for term in ['manage', 'control', 'coordinate']):
                return "Manages system coordination and control operations."
            elif any(term in code_lower for term in ['test', 'validate', 'verify']):
                return "Provides testing and validation functionality."
            elif any(term in code_lower for term in ['api', 'endpoint', 'route']):
                return "Implements API endpoints and web service functionality."
            elif any(term in code_lower for term in ['database', 'query', 'sql']):
                return "Handles database operations and data persistence."
            else:
                return "Provides utility functions and supporting functionality."
                
        except Exception as e:
            logger.error(f"Error inferring module purpose: {e}")
            return "Module purpose could not be automatically determined."
    
    def _assess_content_quality(self, doc: DocumentationGeneration) -> float:
        """Assess overall content quality"""
        try:
            quality_factors = []
            content = doc.generated_content
            
            # Length appropriateness
            length = len(content)
            if self.quality_thresholds['ideal_length_range'][0] <= length <= self.quality_thresholds['ideal_length_range'][1]:
                quality_factors.append(1.0)
            elif length >= self.quality_thresholds['minimum_length']:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.3)
            
            # Structure presence
            if '"""' in content or '#' in content:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.4)
            
            # Content substance
            if any(word in content.lower() for word in ['description', 'purpose', 'usage', 'example']):
                quality_factors.append(0.9)
            else:
                quality_factors.append(0.5)
            
            return sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            return 0.5
    
    def _assess_completeness(self, doc: DocumentationGeneration) -> float:
        """Assess documentation completeness"""
        try:
            completeness_score = 0.0
            content_lower = doc.generated_content.lower()
            
            completeness_factors = {
                'summary': any(word in content_lower for word in ['summary', 'overview', 'purpose']),
                'description': 'description' in content_lower or len(content_lower) > 100,
                'examples': 'example' in content_lower,
                'parameters': any(word in content_lower for word in ['args', 'parameters', 'param']),
                'returns': 'return' in content_lower,
                'usage': 'usage' in content_lower or 'use' in content_lower
            }
            
            for factor, present in completeness_factors.items():
                if present:
                    completeness_score += 1.0 / len(completeness_factors)
            
            return completeness_score
            
        except Exception as e:
            logger.error(f"Error assessing completeness: {e}")
            return 0.0
    
    def _assess_clarity(self, doc: DocumentationGeneration) -> float:
        """Assess documentation clarity"""
        try:
            clarity_factors = []
            content = doc.generated_content
            
            # Sentence structure
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            if len(sentences) > 1:
                clarity_factors.append(0.8)
            else:
                clarity_factors.append(0.4)
            
            # Descriptive language
            descriptive_words = ['provides', 'implements', 'handles', 'manages', 'performs', 'creates']
            if any(word in content.lower() for word in descriptive_words):
                clarity_factors.append(0.9)
            else:
                clarity_factors.append(0.5)
            
            # Specific information
            if any(char.isdigit() for char in content) or 'function' in content.lower():
                clarity_factors.append(0.7)
            else:
                clarity_factors.append(0.5)
            
            return sum(clarity_factors) / len(clarity_factors)
            
        except Exception as e:
            logger.error(f"Error assessing clarity: {e}")
            return 0.5
    
    def _assess_technical_accuracy(self, doc: DocumentationGeneration, code: str) -> float:
        """Assess technical accuracy of documentation"""
        try:
            # Basic accuracy checks
            accuracy_score = 0.5  # Base score
            
            # Check if documentation mentions appropriate elements
            import ast
            try:
                tree = ast.parse(code)
                has_classes = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
                has_functions = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
                
                doc_lower = doc.generated_content.lower()
                mentions_classes = 'class' in doc_lower
                mentions_functions = 'function' in doc_lower
                
                if has_classes and mentions_classes:
                    accuracy_score += 0.2
                if has_functions and mentions_functions:
                    accuracy_score += 0.2
                
                # Bonus for specific technical terms
                technical_terms = ['method', 'attribute', 'parameter', 'return', 'exception']
                if any(term in doc_lower for term in technical_terms):
                    accuracy_score += 0.1
                    
            except:
                pass  # Syntax errors handled gracefully
            
            return min(1.0, accuracy_score)
            
        except Exception as e:
            logger.error(f"Error assessing technical accuracy: {e}")
            return 0.5
    
    def _check_formatting_compliance(self, doc: DocumentationGeneration) -> bool:
        """Check if documentation follows formatting standards"""
        try:
            content = doc.generated_content
            
            # Basic formatting checks
            has_proper_structure = '"""' in content or '#' in content
            has_sections = any(section in content for section in ['Args:', 'Returns:', 'Example:'])
            reasonable_length = len(content) >= self.quality_thresholds['minimum_length']
            
            return has_proper_structure and reasonable_length
            
        except Exception as e:
            logger.error(f"Error checking formatting compliance: {e}")
            return False


class PredictiveCodeIntelligence:
    """Master predictive code intelligence orchestration system"""
    
    def __init__(self):
        self.evolution_predictor = CodeEvolutionPredictor()
        self.language_bridge = NaturalLanguageBridge()
        self.doc_generator = DocumentationGenerator()
        
        # Intelligence coordination
        self.prediction_history = []
        self.analysis_cache = {}
        self.performance_metrics = PredictiveMetrics()
        
        # Configuration
        self.enable_evolution_prediction = True
        self.enable_language_bridge = True
        self.enable_documentation_generation = True
        self.enable_caching = True
        self.prediction_horizon_days = 30
        self.confidence_threshold = 0.5
        
        logger.info("Predictive Code Intelligence master system initialized")
    
    async def analyze_predictive_intelligence(self, file_path: str,
                                            prediction_types: List[PredictionType] = None,
                                            include_documentation: bool = True,
                                            context: CodeIntelligenceContext = None) -> Dict[str, Any]:
        """Comprehensive predictive intelligence analysis with full orchestration"""
        try:
            # Read and validate code
            if not Path(file_path).exists():
                return {'error': f'File not found: {file_path}'}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Create analysis context
            if not context:
                context = self._create_analysis_context(file_path, code)
            
            # Check cache
            cache_key = self._generate_cache_key(file_path, code, prediction_types)
            if self.enable_caching and cache_key in self.analysis_cache:
                logger.info(f"Returning cached analysis for {file_path}")
                return self.analysis_cache[cache_key]
            
            # Initialize comprehensive results
            results = {
                'file_path': file_path,
                'analysis_timestamp': datetime.now(),
                'context': asdict(context),
                'predictions': [],
                'natural_language_explanations': {},
                'generated_documentation': {},
                'evolution_analysis': None,
                'intelligence_metrics': {},
                'recommendations': [],
                'risk_assessment': {}
            }
            
            # Execute predictive analysis pipeline
            if self.enable_evolution_prediction:
                await self._execute_evolution_analysis(code, file_path, prediction_types, results)
            
            if self.enable_language_bridge:
                await self._execute_language_analysis(code, file_path, results)
            
            if self.enable_documentation_generation and include_documentation:
                await self._execute_documentation_generation(code, file_path, results)
            
            # Generate intelligence summary and recommendations
            await self._generate_intelligence_summary(results)
            
            # Update performance metrics
            self._update_performance_metrics(results)
            
            # Cache results
            if self.enable_caching:
                self.analysis_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in predictive intelligence analysis: {e}")
            return {
                'error': str(e),
                'file_path': file_path,
                'analysis_timestamp': datetime.now()
            }
    
    async def _execute_evolution_analysis(self, code: str, file_path: str,
                                        prediction_types: List[PredictionType],
                                        results: Dict[str, Any]) -> None:
        """Execute comprehensive evolution analysis"""
        try:
            # Primary evolution prediction
            evolution_prediction = self.evolution_predictor.predict_code_evolution(code, file_path)
            results['predictions'].append(evolution_prediction)
            
            # Specific prediction types
            if not prediction_types or PredictionType.MAINTENANCE_HOTSPOTS in prediction_types:
                hotspot_prediction = self.evolution_predictor.predict_maintenance_hotspots(code, file_path)
                results['predictions'].append(hotspot_prediction)
            
            if not prediction_types or PredictionType.SECURITY_VULNERABILITIES in prediction_types:
                security_prediction = self.evolution_predictor.predict_security_vulnerabilities(code, file_path)
                results['predictions'].append(security_prediction)
            
            if not prediction_types or PredictionType.PERFORMANCE_DEGRADATION in prediction_types:
                performance_prediction = self.evolution_predictor.predict_performance_degradation(code, file_path)
                results['predictions'].append(performance_prediction)
            
            logger.info(f"Generated {len(results['predictions'])} predictions for {file_path}")
            
        except Exception as e:
            logger.error(f"Error in evolution analysis: {e}")
    
    async def _execute_language_analysis(self, code: str, file_path: str,
                                       results: Dict[str, Any]) -> None:
        """Execute comprehensive natural language analysis"""
        try:
            # Multi-audience explanations
            audiences = ['general', 'technical', 'beginner']
            abstraction_levels = ['high', 'medium', 'low']
            
            for audience in audiences:
                for level in abstraction_levels:
                    translation = self.language_bridge.translate_code_to_language(
                        code, audience, level
                    )
                    
                    key = f"{audience}_{level}"
                    results['natural_language_explanations'][key] = {
                        'explanation': translation.natural_language,
                        'quality': translation.translation_quality,
                        'technical_terms': translation.technical_terms,
                        'context': translation.context_understanding,
                        'validation_score': translation.validation_score
                    }
            
            logger.info(f"Generated {len(results['natural_language_explanations'])} language explanations")
            
        except Exception as e:
            logger.error(f"Error in language analysis: {e}")
    
    async def _execute_documentation_generation(self, code: str, file_path: str,
                                              results: Dict[str, Any]) -> None:
        """Execute comprehensive documentation generation"""
        try:
            # Module documentation
            module_doc = self.doc_generator.generate_documentation(
                code, DocumentationType.MODULE_DOCSTRING
            )
            results['generated_documentation']['module'] = asdict(module_doc)
            
            # API documentation
            api_doc = self.doc_generator.generate_documentation(
                code, DocumentationType.API_DOCUMENTATION
            )
            results['generated_documentation']['api'] = asdict(api_doc)
            
            # Inline comments suggestions
            inline_doc = self.doc_generator.generate_documentation(
                code, DocumentationType.INLINE_COMMENTS
            )
            results['generated_documentation']['inline_suggestions'] = asdict(inline_doc)
            
            logger.info(f"Generated {len(results['generated_documentation'])} documentation types")
            
        except Exception as e:
            logger.error(f"Error in documentation generation: {e}")
    
    async def _generate_intelligence_summary(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive intelligence summary and recommendations"""
        try:
            predictions = results.get('predictions', [])
            
            # Risk assessment
            risk_levels = {
                'high': len([p for p in predictions if p.confidence == PredictionConfidence.VERY_HIGH]),
                'medium': len([p for p in predictions if p.confidence == PredictionConfidence.HIGH]),
                'low': len([p for p in predictions if p.confidence in [PredictionConfidence.MEDIUM, PredictionConfidence.LOW]])
            }
            
            # Generate recommendations
            recommendations = []
            immediate_actions = set()
            
            for prediction in predictions:
                if prediction.confidence in [PredictionConfidence.VERY_HIGH, PredictionConfidence.HIGH]:
                    recommendations.extend(prediction.recommended_actions[:2])
                    if prediction.timeline_estimate in ["Immediate attention required", "1-2 months"]:
                        immediate_actions.update(prediction.recommended_actions[:1])
            
            results['risk_assessment'] = risk_levels
            results['recommendations'] = list(set(recommendations))[:10]
            results['immediate_actions'] = list(immediate_actions)[:5]
            
            # Intelligence metrics
            results['intelligence_metrics'] = {
                'total_predictions': len(predictions),
                'high_confidence_predictions': risk_levels['high'] + risk_levels['medium'],
                'average_probability': sum(p.probability_score for p in predictions) / max(len(predictions), 1),
                'documentation_quality': self._calculate_avg_doc_quality(results),
                'language_explanation_quality': self._calculate_avg_language_quality(results)
            }
            
        except Exception as e:
            logger.error(f"Error generating intelligence summary: {e}")
    
    def _create_analysis_context(self, file_path: str, code: str) -> CodeIntelligenceContext:
        """Create comprehensive analysis context"""
        try:
            return CodeIntelligenceContext(
                project_name=Path(file_path).parent.name,
                codebase_size=len(code),
                programming_languages=['python'],
                development_stage='active',
                analysis_preferences={'detailed_analysis': True}
            )
        except Exception as e:
            logger.error(f"Error creating analysis context: {e}")
            return CodeIntelligenceContext()
    
    def _generate_cache_key(self, file_path: str, code: str, 
                          prediction_types: List[PredictionType]) -> str:
        """Generate cache key for analysis results"""
        try:
            content = f"{file_path}:{hashlib.md5(code.encode()).hexdigest()}"
            if prediction_types:
                content += f":{','.join(pt.value for pt in prediction_types)}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return str(hash(file_path + code))
    
    def _calculate_avg_doc_quality(self, results: Dict[str, Any]) -> float:
        """Calculate average documentation quality"""
        try:
            docs = results.get('generated_documentation', {})
            if not docs:
                return 0.0
            
            quality_scores = []
            for doc_data in docs.values():
                if isinstance(doc_data, dict) and 'content_quality' in doc_data:
                    quality_scores.append(doc_data['content_quality'])
            
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating doc quality: {e}")
            return 0.0
    
    def _calculate_avg_language_quality(self, results: Dict[str, Any]) -> float:
        """Calculate average language explanation quality"""
        try:
            explanations = results.get('natural_language_explanations', {})
            if not explanations:
                return 0.0
            
            quality_scores = [exp.get('quality', 0.0) for exp in explanations.values()]
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating language quality: {e}")
            return 0.0
    
    def _update_performance_metrics(self, results: Dict[str, Any]) -> None:
        """Update system performance metrics"""
        try:
            predictions = results.get('predictions', [])
            if predictions:
                # Calculate metrics
                high_confidence_ratio = len([p for p in predictions 
                                           if p.confidence in [PredictionConfidence.VERY_HIGH, PredictionConfidence.HIGH]]) / len(predictions)
                
                self.performance_metrics.prediction_accuracy = high_confidence_ratio
                self.performance_metrics.model_performance['recent_analysis'] = {
                    'predictions_generated': len(predictions),
                    'high_confidence_ratio': high_confidence_ratio,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")


# Factory function for creating predictive intelligence system
def create_predictive_code_intelligence() -> PredictiveCodeIntelligence:
    """Create and initialize the master predictive code intelligence system"""
    try:
        intelligence = PredictiveCodeIntelligence()
        logger.info("Master Predictive Code Intelligence system created successfully")
        return intelligence
    except Exception as e:
        logger.error(f"Error creating Predictive Code Intelligence: {e}")
        raise


# Export main classes and functions
__all__ = [
    'PredictiveCodeIntelligence',
    'DocumentationGenerator', 
    'create_predictive_code_intelligence'
]