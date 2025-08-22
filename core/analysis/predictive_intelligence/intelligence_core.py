"""
Predictive Intelligence Core
===========================

Master predictive code intelligence coordination system.
Extracted from predictive_code_intelligence.py for enterprise modular architecture.

Agent D Implementation - Hour 16-17: Predictive Intelligence Modularization
"""

import asyncio
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from .data_models import (
    PredictionType, CodePrediction, NaturalLanguageTranslation,
    GeneratedDocumentation, PredictiveMetrics
)
from .evolution_predictor import CodeEvolutionPredictor
from .language_bridge import NaturalLanguageBridge
from .documentation_generator import DocumentationGenerator
from .security_analyzer import SecurityAnalyzer


class PredictiveCodeIntelligence:
    """
    Master Predictive Code Intelligence System
    
    Revolutionary predictive code intelligence that coordinates multiple AI systems
    to provide comprehensive code analysis, prediction, and documentation generation.
    """
    
    def __init__(self):
        # Core components
        self.evolution_predictor = CodeEvolutionPredictor()
        self.language_bridge = NaturalLanguageBridge()
        self.documentation_generator = DocumentationGenerator()
        self.security_analyzer = SecurityAnalyzer()
        
        # Prediction and translation caches
        self.prediction_history: List[CodePrediction] = []
        self.language_translation_cache: Dict[str, Dict[str, Any]] = {}
        self.documentation_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.metrics = PredictiveMetrics()
        self.prediction_accuracy_tracker: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.configuration = {
            'enable_evolution_prediction': True,
            'enable_language_bridge': True,
            'enable_documentation_generation': True,
            'enable_security_analysis': True,
            'enable_caching': True,
            'prediction_horizon_days': 30,
            'cache_ttl_hours': 24,
            'max_cache_size': 1000,
            'quality_threshold': 0.7,
            'batch_processing': True
        }
        
        # Learning and adaptation
        self.learning_enabled = True
        self.adaptation_rate = 0.1
        self.confidence_threshold = 0.8
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Predictive Code Intelligence System initialized")
    
    async def analyze_predictive_intelligence(self, file_path: str,
                                            prediction_types: Optional[List[PredictionType]] = None,
                                            include_documentation: bool = True,
                                            include_translations: bool = True) -> Dict[str, Any]:
        """Comprehensive predictive intelligence analysis of code file"""
        
        try:
            self.logger.info(f"🔍 Starting comprehensive analysis of {file_path}")
            
            # Read and validate code
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if not code.strip():
                return {'error': 'Empty code file'}
            
            # Initialize results structure
            results = {
                'file_path': file_path,
                'analysis_timestamp': datetime.now(),
                'file_stats': self._analyze_file_stats(code),
                'predictions': [],
                'natural_language_explanations': {},
                'generated_documentation': {},
                'security_analysis': {},
                'evolution_analysis': None,
                'quality_metrics': {},
                'recommendations': [],
                'metadata': {
                    'analysis_duration': 0.0,
                    'cache_hits': 0,
                    'components_analyzed': []
                }
            }
            
            start_time = datetime.now()
            
            # Parallel analysis execution for performance
            analysis_tasks = []
            
            # Evolution prediction
            if self.configuration['enable_evolution_prediction']:
                analysis_tasks.append(self._analyze_code_evolution(code, file_path))
                results['metadata']['components_analyzed'].append('evolution_prediction')
            
            # Security analysis
            if self.configuration['enable_security_analysis']:
                analysis_tasks.append(self._analyze_security(code, file_path))
                results['metadata']['components_analyzed'].append('security_analysis')
            
            # Natural language explanations
            if self.configuration['enable_language_bridge'] and include_translations:
                analysis_tasks.append(self._generate_language_explanations(code, file_path))
                results['metadata']['components_analyzed'].append('language_translation')
            
            # Documentation generation
            if self.configuration['enable_documentation_generation'] and include_documentation:
                analysis_tasks.append(self._generate_comprehensive_documentation(code, file_path))
                results['metadata']['components_analyzed'].append('documentation_generation')
            
            # Execute all analyses in parallel
            if analysis_tasks:
                analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(analysis_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Analysis component failed: {result}")
                        continue
                    
                    component = results['metadata']['components_analyzed'][i]
                    if component == 'evolution_prediction':
                        results['evolution_analysis'] = result['evolution_analysis']
                        results['predictions'].extend(result['predictions'])
                    elif component == 'security_analysis':
                        results['security_analysis'] = result
                        results['predictions'].extend(result.get('predictions', []))
                    elif component == 'language_translation':
                        results['natural_language_explanations'] = result
                    elif component == 'documentation_generation':
                        results['generated_documentation'] = result
            
            # Generate specific predictions if requested
            if prediction_types:
                specific_predictions = await self._generate_specific_predictions(
                    code, file_path, prediction_types, results.get('evolution_analysis')
                )
                results['predictions'].extend(specific_predictions)
            
            # Generate comprehensive recommendations
            results['recommendations'] = await self._generate_recommendations(results)
            
            # Calculate quality metrics
            results['quality_metrics'] = await self._calculate_quality_metrics(results)
            
            # Update performance metrics
            analysis_duration = (datetime.now() - start_time).total_seconds()
            results['metadata']['analysis_duration'] = analysis_duration
            await self._update_performance_metrics(results, analysis_duration)
            
            # Cache results if enabled
            if self.configuration['enable_caching']:
                await self._cache_analysis_results(file_path, code, results)
            
            # Learn from analysis if enabled
            if self.learning_enabled:
                await self._learn_from_analysis(results)
            
            self.logger.info(f"✅ Completed analysis of {file_path} in {analysis_duration:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in predictive intelligence analysis: {e}")
            return {
                'error': str(e),
                'file_path': file_path,
                'analysis_timestamp': datetime.now()
            }
    
    async def _analyze_code_evolution(self, code: str, file_path: str) -> Dict[str, Any]:
        """Analyze code evolution and generate predictions"""
        
        try:
            # Perform evolution analysis
            evolution_analysis = self.evolution_predictor.predict_evolution(code, file_path)
            
            # Generate evolution-based predictions
            predictions = await self._generate_evolution_predictions(evolution_analysis, file_path)
            
            return {
                'evolution_analysis': evolution_analysis,
                'predictions': predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error in code evolution analysis: {e}")
            return {'evolution_analysis': None, 'predictions': []}
    
    async def _analyze_security(self, code: str, file_path: str) -> Dict[str, Any]:
        """Perform comprehensive security analysis"""
        
        try:
            # Analyze security vulnerabilities
            security_predictions = await self.security_analyzer.analyze_security_vulnerabilities(
                code, file_path
            )
            
            # Create security risk report
            risk_report = self.security_analyzer.create_security_risk_report(security_predictions)
            
            return {
                'predictions': security_predictions,
                'risk_report': risk_report,
                'vulnerability_count': len(security_predictions),
                'overall_security_score': 1.0 - risk_report['summary']['overall_risk_score']
            }
            
        except Exception as e:
            self.logger.error(f"Error in security analysis: {e}")
            return {'predictions': [], 'risk_report': {}, 'vulnerability_count': 0}
    
    async def _generate_language_explanations(self, code: str, file_path: str) -> Dict[str, Any]:
        """Generate natural language explanations for code"""
        
        try:
            # Check cache first
            cache_key = hashlib.md5(code.encode()).hexdigest()
            if (self.configuration['enable_caching'] and 
                cache_key in self.language_translation_cache):
                self.logger.debug("Cache hit for language explanations")
                return self.language_translation_cache[cache_key]
            
            explanations = {}
            
            # Generate explanations for different audiences and abstraction levels
            audiences = ['beginner', 'general', 'technical']
            abstraction_levels = ['high', 'medium', 'low']
            
            for audience in audiences:
                for level in abstraction_levels:
                    try:
                        translation = self.language_bridge.translate_code_to_language(
                            code, audience, level
                        )
                        
                        key = f"{audience}_{level}"
                        explanations[key] = {
                            'explanation': translation.natural_language,
                            'quality_score': translation.translation_quality,
                            'technical_terms': translation.technical_terms,
                            'context_understanding': translation.context_understanding,
                            'alternative_explanations': translation.alternative_explanations
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to generate {audience}_{level} explanation: {e}")
            
            # Cache results
            if self.configuration['enable_caching']:
                self.language_translation_cache[cache_key] = explanations
                await self._cleanup_cache(self.language_translation_cache)
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Error generating language explanations: {e}")
            return {}
    
    async def _generate_comprehensive_documentation(self, code: str, file_path: str) -> Dict[str, Any]:
        """Generate comprehensive documentation for code"""
        
        try:
            # Check cache first
            cache_key = hashlib.md5(code.encode()).hexdigest()
            if (self.configuration['enable_caching'] and 
                cache_key in self.documentation_cache):
                self.logger.debug("Cache hit for documentation")
                return self.documentation_cache[cache_key]
            
            documentation = {}
            
            # Parse code to identify elements
            try:
                import ast
                tree = ast.parse(code)
            except SyntaxError:
                return {'error': 'Unable to parse code for documentation'}
            
            # Generate documentation for functions
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            for func_name in functions:
                try:
                    doc = self.documentation_generator.generate_documentation(
                        code, self.documentation_generator.DocumentationType.FUNCTION_DOCSTRING, func_name
                    )
                    documentation[f"function_{func_name}"] = {
                        'content': doc.generated_content,
                        'quality_score': doc.documentation_quality,
                        'completeness_score': doc.completeness_score,
                        'clarity_score': doc.clarity_score,
                        'includes_examples': doc.includes_examples,
                        'includes_parameters': doc.includes_parameters,
                        'includes_return_values': doc.includes_return_values
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to generate documentation for function {func_name}: {e}")
            
            # Generate documentation for classes
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            for class_name in classes:
                try:
                    doc = self.documentation_generator.generate_documentation(
                        code, self.documentation_generator.DocumentationType.CLASS_DOCSTRING, class_name
                    )
                    documentation[f"class_{class_name}"] = {
                        'content': doc.generated_content,
                        'quality_score': doc.documentation_quality,
                        'completeness_score': doc.completeness_score,
                        'clarity_score': doc.clarity_score
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to generate documentation for class {class_name}: {e}")
            
            # Generate module documentation
            try:
                module_doc = self.documentation_generator.generate_documentation(
                    code, self.documentation_generator.DocumentationType.MODULE_DOCSTRING
                )
                documentation['module'] = {
                    'content': module_doc.generated_content,
                    'quality_score': module_doc.documentation_quality,
                    'completeness_score': module_doc.completeness_score,
                    'clarity_score': module_doc.clarity_score
                }
            except Exception as e:
                self.logger.warning(f"Failed to generate module documentation: {e}")
            
            # Generate API documentation
            try:
                api_doc = self.documentation_generator.generate_documentation(
                    code, self.documentation_generator.DocumentationType.API_DOCUMENTATION
                )
                documentation['api'] = {
                    'content': api_doc.generated_content,
                    'quality_score': api_doc.documentation_quality
                }
            except Exception as e:
                self.logger.warning(f"Failed to generate API documentation: {e}")
            
            # Cache results
            if self.configuration['enable_caching']:
                self.documentation_cache[cache_key] = documentation
                await self._cleanup_cache(self.documentation_cache)
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive documentation: {e}")
            return {}
    
    async def _generate_evolution_predictions(self, evolution_analysis, file_path: str) -> List[CodePrediction]:
        """Generate predictions from evolution analysis"""
        
        predictions = []
        
        try:
            # Maintenance hotspot predictions
            maintenance_burden = evolution_analysis.maintenance_burden_projection
            overall_burden = maintenance_burden.get('overall_maintenance_burden', 0)
            
            if overall_burden > 0.6:
                prediction = CodePrediction(
                    prediction_type=PredictionType.MAINTENANCE_HOTSPOTS,
                    target_file=file_path,
                    prediction_summary="High maintenance burden predicted",
                    detailed_analysis=f"Analysis indicates {overall_burden:.1%} likelihood of increased maintenance burden. "
                                    f"Key factors: complexity burden ({maintenance_burden.get('complexity_burden', 0):.1%}), "
                                    f"method count pressure ({maintenance_burden.get('method_count_burden', 0):.1%}), "
                                    f"dependency burden ({maintenance_burden.get('dependency_burden', 0):.1%})",
                    confidence=PredictionConfidence.HIGH if overall_burden > 0.8 else PredictionConfidence.MEDIUM,
                    probability_score=overall_burden,
                    timeline_estimate="3-6 months",
                    impact_assessment={
                        'development_velocity': 'decreased',
                        'bug_frequency': 'increased',
                        'code_quality': 'declining',
                        'team_productivity': 'reduced'
                    },
                    recommended_actions=[
                        "Implement proactive refactoring strategy",
                        "Increase automated test coverage",
                        "Add comprehensive monitoring and alerting",
                        "Consider architectural improvements",
                        "Establish code quality gates"
                    ],
                    prevention_strategies=[
                        "Regular technical debt assessment",
                        "Continuous refactoring practices",
                        "Code quality metrics monitoring"
                    ],
                    evidence_factors=[
                        f"Overall maintenance burden: {overall_burden:.1%}",
                        f"Complexity growth rate: {evolution_analysis.growth_patterns.get('complexity_growth_rate', 0):.1%}",
                        f"Method growth rate: {evolution_analysis.growth_patterns.get('method_growth_rate', 0):.1%}"
                    ]
                )
                predictions.append(prediction)
            
            # Performance degradation predictions
            complexity_trends = evolution_analysis.complexity_trends
            complexity_increase = complexity_trends.get('complexity_increase_tendency', 0)
            
            if complexity_increase > 0.5:
                prediction = CodePrediction(
                    prediction_type=PredictionType.PERFORMANCE_DEGRADATION,
                    target_file=file_path,
                    prediction_summary="Performance degradation risk identified",
                    detailed_analysis=f"Growing complexity ({complexity_increase:.1%} tendency) suggests potential performance impact. "
                                    f"Average complexity: {complexity_trends.get('average_complexity', 0):.1f}, "
                                    f"Max complexity: {complexity_trends.get('max_complexity', 0):.1f}",
                    confidence=PredictionConfidence.MEDIUM,
                    probability_score=complexity_increase * 0.7,
                    timeline_estimate="2-4 months",
                    impact_assessment={
                        'response_time': 'increased',
                        'resource_usage': 'increased',
                        'user_experience': 'degraded',
                        'scalability': 'reduced'
                    },
                    recommended_actions=[
                        "Profile performance critical paths",
                        "Implement performance monitoring",
                        "Consider algorithmic optimizations",
                        "Add performance regression tests",
                        "Optimize data structures and algorithms"
                    ],
                    prevention_strategies=[
                        "Regular performance reviews",
                        "Complexity thresholds enforcement",
                        "Performance budgets implementation"
                    ]
                )
                predictions.append(prediction)
            
            # Feature addition predictions
            feature_likelihood = evolution_analysis.feature_addition_likelihood
            for feature_type, likelihood in feature_likelihood.items():
                if likelihood > 0.6:
                    prediction = CodePrediction(
                        prediction_type=PredictionType.FEATURE_ADDITIONS,
                        target_file=file_path,
                        target_element=feature_type,
                        prediction_summary=f"High likelihood of {feature_type} addition",
                        detailed_analysis=f"Analysis suggests {likelihood:.1%} probability of {feature_type} expansion based on current patterns",
                        confidence=PredictionConfidence.HIGH if likelihood > 0.8 else PredictionConfidence.MEDIUM,
                        probability_score=likelihood,
                        timeline_estimate="1-3 months",
                        impact_assessment={
                            'code_complexity': 'increased',
                            'testing_requirements': 'expanded',
                            'documentation_needs': 'increased',
                            'maintenance_overhead': 'increased'
                        },
                        recommended_actions=[
                            f"Plan architecture for {feature_type} expansion",
                            "Design extensible interfaces",
                            "Prepare test infrastructure",
                            "Update documentation templates",
                            "Consider modular design patterns"
                        ],
                        monitoring_indicators=[
                            "Feature requests in issue tracker",
                            "Usage patterns analysis",
                            "Customer feedback trends",
                            "Market demand signals"
                        ]
                    )
                    predictions.append(prediction)
            
            # Refactoring need predictions
            refactoring_pressure = evolution_analysis.refactoring_pressure
            for pressure_type, pressure_level in refactoring_pressure.items():
                if pressure_level > 0.4:
                    prediction = CodePrediction(
                        prediction_type=PredictionType.REFACTORING_NEEDS,
                        target_file=file_path,
                        target_element=pressure_type,
                        prediction_summary=f"Refactoring needed for {pressure_type}",
                        detailed_analysis=f"Pressure level: {pressure_level:.1%} indicates refactoring urgency for {pressure_type}",
                        confidence=PredictionConfidence.HIGH if pressure_level > 0.7 else PredictionConfidence.MEDIUM,
                        probability_score=pressure_level,
                        timeline_estimate="1-2 months" if pressure_level > 0.7 else "3-6 months",
                        impact_assessment={
                            'maintainability': 'improved_after_refactoring',
                            'code_quality': 'improved',
                            'development_velocity': 'temporarily_decreased_then_improved',
                            'bug_reduction': 'significant'
                        },
                        recommended_actions=self._get_refactoring_recommendations(pressure_type),
                        prevention_strategies=[
                            "Regular code reviews",
                            "Automated code quality checks",
                            "Continuous refactoring practices",
                            "Technical debt tracking"
                        ]
                    )
                    predictions.append(prediction)
            
        except Exception as e:
            self.logger.error(f"Error generating evolution predictions: {e}")
        
        return predictions
    
    async def _generate_specific_predictions(self, code: str, file_path: str, 
                                           prediction_types: List[PredictionType],
                                           evolution_analysis) -> List[CodePrediction]:
        """Generate specific types of predictions"""
        
        predictions = []
        
        for prediction_type in prediction_types:
            try:
                if prediction_type == PredictionType.DOCUMENTATION_NEEDS:
                    doc_predictions = await self._predict_documentation_needs(code, file_path)
                    predictions.extend(doc_predictions)
                elif prediction_type == PredictionType.TESTING_REQUIREMENTS:
                    test_predictions = await self._predict_testing_requirements(code, file_path)
                    predictions.extend(test_predictions)
                elif prediction_type == PredictionType.DEPENDENCY_CHANGES:
                    if evolution_analysis:
                        dep_predictions = await self._predict_dependency_changes(
                            evolution_analysis.dependency_evolution, file_path
                        )
                        predictions.extend(dep_predictions)
                
            except Exception as e:
                self.logger.error(f"Error generating {prediction_type} predictions: {e}")
        
        return predictions
    
    async def _predict_documentation_needs(self, code: str, file_path: str) -> List[CodePrediction]:
        """Predict documentation needs"""
        
        predictions = []
        
        try:
            import ast
            tree = ast.parse(code)
            
            # Calculate current documentation coverage
            total_functions = 0
            documented_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    if ast.get_docstring(node):
                        documented_functions += 1
            
            if total_functions > 0:
                coverage = documented_functions / total_functions
                
                if coverage < 0.5:
                    prediction = CodePrediction(
                        prediction_type=PredictionType.DOCUMENTATION_NEEDS,
                        target_file=file_path,
                        prediction_summary="Low documentation coverage detected",
                        detailed_analysis=f"Current documentation coverage: {coverage:.1%}. "
                                        f"{total_functions - documented_functions} functions lack documentation.",
                        confidence=PredictionConfidence.HIGH,
                        probability_score=1.0 - coverage,
                        timeline_estimate="1-2 weeks",
                        impact_assessment={
                            'maintainability': 'poor',
                            'onboarding_difficulty': 'high',
                            'code_understanding': 'difficult'
                        },
                        recommended_actions=[
                            "Add docstrings to undocumented functions",
                            "Create module-level documentation",
                            "Generate API documentation",
                            "Add usage examples"
                        ]
                    )
                    predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting documentation needs: {e}")
        
        return predictions
    
    async def _predict_testing_requirements(self, code: str, file_path: str) -> List[CodePrediction]:
        """Predict testing requirements"""
        
        predictions = []
        
        try:
            # Analyze code complexity and coverage needs
            if 'test' not in file_path.lower():
                # Estimate testing needs for non-test files
                complexity_score = self._estimate_testing_complexity(code)
                
                if complexity_score > 0.6:
                    prediction = CodePrediction(
                        prediction_type=PredictionType.TESTING_REQUIREMENTS,
                        target_file=file_path,
                        prediction_summary="Comprehensive testing needed",
                        detailed_analysis=f"Code complexity score: {complexity_score:.1%} indicates need for extensive testing",
                        confidence=PredictionConfidence.MEDIUM,
                        probability_score=complexity_score,
                        timeline_estimate="2-4 weeks",
                        impact_assessment={
                            'quality_assurance': 'critical',
                            'regression_risk': 'high',
                            'maintenance_confidence': 'low'
                        },
                        recommended_actions=[
                            "Implement unit tests for all functions",
                            "Add integration tests for complex workflows",
                            "Create edge case tests",
                            "Implement performance tests"
                        ]
                    )
                    predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting testing requirements: {e}")
        
        return predictions
    
    async def _predict_dependency_changes(self, dependency_evolution: Dict[str, List[str]], 
                                        file_path: str) -> List[CodePrediction]:
        """Predict dependency changes"""
        
        predictions = []
        
        try:
            # Likely new dependencies
            new_deps = dependency_evolution.get('likely_new_dependencies', [])
            if new_deps:
                prediction = CodePrediction(
                    prediction_type=PredictionType.DEPENDENCY_CHANGES,
                    target_file=file_path,
                    target_element="new_dependencies",
                    prediction_summary=f"Likely to add {len(new_deps)} new dependencies",
                    detailed_analysis=f"Based on code patterns, likely new dependencies: {', '.join(new_deps)}",
                    confidence=PredictionConfidence.MEDIUM,
                    probability_score=0.7,
                    timeline_estimate="1-2 months",
                    impact_assessment={
                        'complexity_increase': 'moderate',
                        'security_review_needed': 'yes',
                        'dependency_management': 'increased'
                    },
                    recommended_actions=[
                        "Review and approve new dependencies",
                        "Assess security implications",
                        "Update dependency management",
                        "Plan for version compatibility"
                    ]
                )
                predictions.append(prediction)
        
        except Exception as e:
            self.logger.error(f"Error predicting dependency changes: {e}")
        
        return predictions
    
    def _analyze_file_stats(self, code: str) -> Dict[str, Any]:
        """Analyze basic file statistics"""
        
        lines = code.split('\n')
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'character_count': len(code),
            'estimated_reading_time_minutes': len(code.split()) / 200  # Average reading speed
        }
    
    def _estimate_testing_complexity(self, code: str) -> float:
        """Estimate testing complexity based on code characteristics"""
        
        try:
            import ast
            tree = ast.parse(code)
            
            complexity_factors = []
            
            # Function count
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            complexity_factors.append(min(len(functions) / 20, 1.0))
            
            # Conditional complexity
            conditionals = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
            complexity_factors.append(min(len(conditionals) / 30, 1.0))
            
            # Loop complexity
            loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
            complexity_factors.append(min(len(loops) / 20, 1.0))
            
            # Exception handling
            try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
            complexity_factors.append(min(len(try_blocks) / 10, 1.0))
            
            return sum(complexity_factors) / len(complexity_factors)
            
        except Exception:
            return 0.5  # Default moderate complexity
    
    def _get_refactoring_recommendations(self, pressure_type: str) -> List[str]:
        """Get specific refactoring recommendations"""
        
        recommendations_map = {
            'method_length_pressure': [
                "Extract smaller methods from long methods",
                "Apply Single Responsibility Principle",
                "Use Extract Method refactoring pattern",
                "Break down complex operations into steps"
            ],
            'complexity_pressure': [
                "Simplify conditional logic",
                "Extract complex expressions into variables",
                "Apply Strategy pattern for complex conditionals",
                "Reduce cyclomatic complexity"
            ],
            'duplication_pressure': [
                "Extract common functionality into shared methods",
                "Create utility classes for reusable code",
                "Apply Template Method pattern",
                "Implement proper inheritance hierarchy"
            ],
            'class_size_pressure': [
                "Split large classes into focused components",
                "Apply Single Responsibility Principle",
                "Extract related functionality into separate classes",
                "Consider composition over inheritance"
            ],
            'nesting_pressure': [
                "Reduce nesting depth using early returns",
                "Extract nested logic into separate methods",
                "Use guard clauses to eliminate nesting",
                "Apply polymorphism to reduce conditional complexity"
            ]
        }
        
        return recommendations_map.get(pressure_type, [
            "Apply general refactoring principles",
            "Improve code structure and organization",
            "Enhance readability and maintainability"
        ])
    
    async def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on analysis results"""
        
        recommendations = []
        
        try:
            # Security recommendations
            security_analysis = results.get('security_analysis', {})
            if security_analysis.get('vulnerability_count', 0) > 0:
                recommendations.extend([
                    "🔒 Address identified security vulnerabilities immediately",
                    "🔍 Implement automated security scanning in CI/CD pipeline",
                    "📚 Provide security training for development team"
                ])
            
            # Code quality recommendations
            predictions = results.get('predictions', [])
            maintenance_predictions = [p for p in predictions 
                                     if p.prediction_type == PredictionType.MAINTENANCE_HOTSPOTS]
            if maintenance_predictions:
                recommendations.extend([
                    "🛠️ Implement proactive refactoring strategy",
                    "📊 Set up code quality monitoring",
                    "🧪 Increase automated test coverage"
                ])
            
            # Documentation recommendations
            doc_predictions = [p for p in predictions 
                             if p.prediction_type == PredictionType.DOCUMENTATION_NEEDS]
            if doc_predictions:
                recommendations.extend([
                    "📝 Improve code documentation coverage",
                    "📖 Generate comprehensive API documentation",
                    "💡 Add usage examples and tutorials"
                ])
            
            # Performance recommendations
            perf_predictions = [p for p in predictions 
                              if p.prediction_type == PredictionType.PERFORMANCE_DEGRADATION]
            if perf_predictions:
                recommendations.extend([
                    "⚡ Profile and optimize performance critical paths",
                    "📈 Implement performance monitoring",
                    "🎯 Set performance budgets and thresholds"
                ])
            
            # General recommendations
            if len(predictions) > 5:
                recommendations.append("🔄 Consider comprehensive code review and refactoring")
            
            if not recommendations:
                recommendations.append("✅ Code appears to be in good shape, continue regular maintenance")
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("⚠️ Unable to generate specific recommendations")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def _calculate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality metrics"""
        
        try:
            metrics = {}
            
            # Security score
            security_analysis = results.get('security_analysis', {})
            metrics['security_score'] = security_analysis.get('overall_security_score', 0.8)
            
            # Prediction confidence
            predictions = results.get('predictions', [])
            if predictions:
                confidence_scores = [self._confidence_to_score(p.confidence) for p in predictions]
                metrics['prediction_confidence'] = sum(confidence_scores) / len(confidence_scores)
            else:
                metrics['prediction_confidence'] = 0.8
            
            # Documentation quality
            documentation = results.get('generated_documentation', {})
            if documentation:
                doc_scores = [doc.get('quality_score', 0.5) for doc in documentation.values() 
                             if isinstance(doc, dict) and 'quality_score' in doc]
                metrics['documentation_quality'] = sum(doc_scores) / len(doc_scores) if doc_scores else 0.5
            else:
                metrics['documentation_quality'] = 0.5
            
            # Translation quality
            translations = results.get('natural_language_explanations', {})
            if translations:
                trans_scores = [trans.get('quality_score', 0.5) for trans in translations.values() 
                               if isinstance(trans, dict) and 'quality_score' in trans]
                metrics['translation_quality'] = sum(trans_scores) / len(trans_scores) if trans_scores else 0.5
            else:
                metrics['translation_quality'] = 0.5
            
            # Overall quality score
            metrics['overall_quality'] = (
                metrics['security_score'] * 0.3 +
                metrics['prediction_confidence'] * 0.3 +
                metrics['documentation_quality'] * 0.2 +
                metrics['translation_quality'] * 0.2
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            return {'overall_quality': 0.5}
    
    def _confidence_to_score(self, confidence: PredictionConfidence) -> float:
        """Convert confidence enum to numeric score"""
        
        confidence_scores = {
            PredictionConfidence.VERY_HIGH: 1.0,
            PredictionConfidence.HIGH: 0.8,
            PredictionConfidence.MEDIUM: 0.6,
            PredictionConfidence.LOW: 0.4,
            PredictionConfidence.SPECULATIVE: 0.2
        }
        
        return confidence_scores.get(confidence, 0.6)
    
    async def _update_performance_metrics(self, results: Dict[str, Any], duration: float):
        """Update system performance metrics"""
        
        try:
            self.metrics.total_predictions += len(results.get('predictions', []))
            
            # Update analysis time tracking
            if not hasattr(self.metrics, 'analysis_times'):
                self.metrics.analysis_times = []
            
            self.metrics.analysis_times.append(duration)
            if len(self.metrics.analysis_times) > 100:
                self.metrics.analysis_times = self.metrics.analysis_times[-100:]
            
            self.metrics.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _cache_analysis_results(self, file_path: str, code: str, results: Dict[str, Any]):
        """Cache analysis results for future use"""
        
        try:
            # Implementation would store results in cache with TTL
            # For now, just log the caching action
            self.logger.debug(f"Caching analysis results for {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error caching analysis results: {e}")
    
    async def _cleanup_cache(self, cache: Dict[str, Any]):
        """Cleanup cache to maintain size limits"""
        
        try:
            max_size = self.configuration['max_cache_size']
            if len(cache) > max_size:
                # Remove oldest entries (simplified approach)
                items_to_remove = len(cache) - max_size
                for _ in range(items_to_remove):
                    cache.pop(next(iter(cache)))
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
    
    async def _learn_from_analysis(self, results: Dict[str, Any]):
        """Learn from analysis results to improve future predictions"""
        
        try:
            # Store analysis for learning
            self.prediction_history.extend(results.get('predictions', []))
            
            # Keep history manageable
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            # Future: Implement ML model training based on historical accuracy
            
        except Exception as e:
            self.logger.error(f"Error in learning from analysis: {e}")
    
    def get_prediction_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of predictive analysis results"""
        
        try:
            predictions = analysis_results.get('predictions', [])
            
            summary = {
                'analysis_overview': {
                    'file_path': analysis_results.get('file_path', ''),
                    'analysis_timestamp': analysis_results.get('analysis_timestamp'),
                    'total_predictions': len(predictions),
                    'analysis_duration': analysis_results.get('metadata', {}).get('analysis_duration', 0),
                    'components_analyzed': analysis_results.get('metadata', {}).get('components_analyzed', [])
                },
                'prediction_breakdown': {
                    'by_type': {},
                    'by_confidence': {},
                    'by_timeline': {}
                },
                'quality_assessment': analysis_results.get('quality_metrics', {}),
                'security_summary': self._summarize_security_analysis(analysis_results.get('security_analysis', {})),
                'high_priority_items': [],
                'recommended_immediate_actions': analysis_results.get('recommendations', [])[:5],
                'predicted_evolution': self._summarize_evolution_analysis(analysis_results.get('evolution_analysis')),
                'documentation_status': self._summarize_documentation(analysis_results.get('generated_documentation', {})),
                'translation_coverage': self._summarize_translations(analysis_results.get('natural_language_explanations', {}))
            }
            
            # Analyze prediction breakdown
            for prediction in predictions:
                # By type
                pred_type = prediction.prediction_type.value
                summary['prediction_breakdown']['by_type'][pred_type] = \
                    summary['prediction_breakdown']['by_type'].get(pred_type, 0) + 1
                
                # By confidence
                confidence = prediction.confidence.value
                summary['prediction_breakdown']['by_confidence'][confidence] = \
                    summary['prediction_breakdown']['by_confidence'].get(confidence, 0) + 1
                
                # By timeline
                timeline = prediction.timeline_estimate
                summary['prediction_breakdown']['by_timeline'][timeline] = \
                    summary['prediction_breakdown']['by_timeline'].get(timeline, 0) + 1
                
                # High priority items
                if (prediction.confidence in [PredictionConfidence.VERY_HIGH, PredictionConfidence.HIGH] and
                    prediction.probability_score > 0.7):
                    summary['high_priority_items'].append({
                        'type': pred_type,
                        'summary': prediction.prediction_summary,
                        'probability': prediction.probability_score,
                        'timeline': timeline,
                        'confidence': confidence
                    })
            
            # Limit high priority items
            summary['high_priority_items'] = summary['high_priority_items'][:10]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating prediction summary: {e}")
            return {'error': str(e)}
    
    def _summarize_security_analysis(self, security_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize security analysis results"""
        
        return {
            'vulnerability_count': security_analysis.get('vulnerability_count', 0),
            'overall_security_score': security_analysis.get('overall_security_score', 0.8),
            'risk_level': self._get_risk_level(security_analysis.get('overall_security_score', 0.8)),
            'critical_issues': security_analysis.get('risk_report', {}).get('summary', {}).get('critical_count', 0)
        }
    
    def _summarize_evolution_analysis(self, evolution_analysis) -> Dict[str, Any]:
        """Summarize evolution analysis results"""
        
        if not evolution_analysis:
            return {'status': 'not_analyzed'}
        
        return {
            'maintenance_burden_score': evolution_analysis.maintenance_burden_projection.get('overall_maintenance_burden', 0),
            'complexity_trend': evolution_analysis.complexity_trends.get('complexity_increase_tendency', 0),
            'predicted_growth_areas': list(evolution_analysis.feature_addition_likelihood.keys()),
            'refactoring_needed': any(v > 0.5 for v in evolution_analysis.refactoring_pressure.values())
        }
    
    def _summarize_documentation(self, documentation: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize documentation analysis"""
        
        if not documentation:
            return {'status': 'not_generated'}
        
        doc_items = [doc for doc in documentation.values() if isinstance(doc, dict)]
        
        return {
            'total_items': len(doc_items),
            'average_quality': sum(doc.get('quality_score', 0) for doc in doc_items) / len(doc_items) if doc_items else 0,
            'coverage_types': list(documentation.keys())
        }
    
    def _summarize_translations(self, translations: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize translation analysis"""
        
        if not translations:
            return {'status': 'not_generated'}
        
        trans_items = [trans for trans in translations.values() if isinstance(trans, dict)]
        
        return {
            'total_translations': len(trans_items),
            'average_quality': sum(trans.get('quality_score', 0) for trans in trans_items) / len(trans_items) if trans_items else 0,
            'audiences_covered': list(set(key.split('_')[0] for key in translations.keys())),
            'abstraction_levels': list(set(key.split('_')[1] for key in translations.keys() if '_' in key))
        }
    
    def _get_risk_level(self, security_score: float) -> str:
        """Convert security score to risk level"""
        
        if security_score >= 0.9:
            return 'low'
        elif security_score >= 0.7:
            return 'medium'
        elif security_score >= 0.5:
            return 'high'
        else:
            return 'critical'


def create_predictive_code_intelligence() -> PredictiveCodeIntelligence:
    """Factory function to create PredictiveCodeIntelligence instance"""
    
    return PredictiveCodeIntelligence()