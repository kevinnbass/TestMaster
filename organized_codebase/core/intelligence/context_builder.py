"""
Analysis Context Builder

This module converts classical analysis results into structured context that can be
used by LLMs to generate high-quality, accurate documentation. It aggregates
insights from multiple analysis dimensions and formats them for optimal LLM consumption.
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

# Import classical analysis components
try:
    from testmaster.analysis.comprehensive_analysis.main_analyzer import CodeAnalysisEngine
    from testmaster.analysis.comprehensive_analysis.base_analyzer import BaseAnalyzer
except ImportError:
    # Fallback for development
    pass

logger = logging.getLogger(__name__)


@dataclass
class ContextualInsight:
    """Represents a single contextual insight extracted from analysis."""
    category: str
    insight_type: str
    description: str
    evidence: List[str]
    confidence: float
    relevance: str  # "high", "medium", "low"
    source_analyzer: str


@dataclass
class FunctionContext:
    """Context information for a specific function."""
    name: str
    docstring: Optional[str]
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    complexity_score: Optional[float]
    security_issues: List[str]
    performance_notes: List[str]
    usage_patterns: List[str]
    related_functions: List[str]
    insights: List[ContextualInsight]


@dataclass
class ClassContext:
    """Context information for a specific class."""
    name: str
    docstring: Optional[str]
    methods: List[str]
    attributes: List[str]
    inheritance: List[str]
    design_patterns: List[str]
    responsibilities: List[str]
    usage_patterns: List[str]
    insights: List[ContextualInsight]


@dataclass
class ModuleContext:
    """Context information for a module."""
    name: str
    path: str
    docstring: Optional[str]
    imports: List[str]
    exports: List[str]
    functions: List[FunctionContext]
    classes: List[ClassContext]
    module_type: str  # "utility", "business_logic", "interface", etc.
    dependencies: List[str]
    architecture_role: str
    insights: List[ContextualInsight]


@dataclass
class ProjectContext:
    """High-level project context."""
    name: str
    description: str
    architecture_style: str
    primary_domain: str
    tech_stack: List[str]
    key_patterns: List[str]
    quality_metrics: Dict[str, float]
    modules: List[ModuleContext]
    insights: List[ContextualInsight]


class AnalysisContextBuilder:
    """
    Builds comprehensive context from classical analysis results for documentation generation.
    
    This class takes raw analysis results and transforms them into structured,
    LLM-friendly context that preserves important insights while filtering
    out noise and irrelevant details.
    """
    
    def __init__(self, analysis_engine: Optional[CodeAnalysisEngine] = None):
        """
        Initialize the context builder.
        
        Args:
            analysis_engine: Optional pre-configured analysis engine
        """
        self.analysis_engine = analysis_engine or CodeAnalysisEngine()
        self.context_cache = {}
        
    def build_module_context(self, file_path: str) -> ModuleContext:
        """
        Build comprehensive context for a single module.
        
        Args:
            file_path: Path to the Python module file
            
        Returns:
            ModuleContext: Structured context for the module
        """
        logger.info(f"Building context for module: {file_path}")
        
        # Check cache first
        cache_key = f"module:{file_path}"
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        # Run comprehensive analysis
        analysis_results = self._run_comprehensive_analysis(file_path)
        
        # Extract module-level information
        module_info = self._extract_module_info(file_path, analysis_results)
        
        # Build function contexts
        functions = self._build_function_contexts(analysis_results)
        
        # Build class contexts
        classes = self._build_class_contexts(analysis_results)
        
        # Extract architectural insights
        insights = self._extract_architectural_insights(analysis_results)
        
        # Determine module type and role
        module_type = self._infer_module_type(analysis_results)
        architecture_role = self._infer_architecture_role(analysis_results)
        
        context = ModuleContext(
            name=module_info["name"],
            path=file_path,
            docstring=module_info.get("docstring"),
            imports=module_info.get("imports", []),
            exports=module_info.get("exports", []),
            functions=functions,
            classes=classes,
            module_type=module_type,
            dependencies=self._extract_dependencies(analysis_results),
            architecture_role=architecture_role,
            insights=insights
        )
        
        # Cache the result
        self.context_cache[cache_key] = context
        
        return context
        
    def build_function_context(self, file_path: str, function_name: str) -> Optional[FunctionContext]:
        """
        Build context for a specific function.
        
        Args:
            file_path: Path to the file containing the function
            function_name: Name of the function
            
        Returns:
            FunctionContext: Context for the function, or None if not found
        """
        module_context = self.build_module_context(file_path)
        
        for func in module_context.functions:
            if func.name == function_name:
                return func
                
        return None
        
    def build_class_context(self, file_path: str, class_name: str) -> Optional[ClassContext]:
        """
        Build context for a specific class.
        
        Args:
            file_path: Path to the file containing the class
            class_name: Name of the class
            
        Returns:
            ClassContext: Context for the class, or None if not found
        """
        module_context = self.build_module_context(file_path)
        
        for cls in module_context.classes:
            if cls.name == class_name:
                return cls
                
        return None
        
    def build_project_context(self, project_path: str) -> ProjectContext:
        """
        Build comprehensive context for an entire project.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            ProjectContext: High-level project context
        """
        logger.info(f"Building project context for: {project_path}")
        
        # Find all Python files
        python_files = self._find_python_files(project_path)
        
        # Build module contexts
        modules = []
        for file_path in python_files:
            try:
                module_context = self.build_module_context(file_path)
                modules.append(module_context)
            except Exception as e:
                logger.warning(f"Failed to build context for {file_path}: {e}")
        
        # Analyze project-level patterns
        project_insights = self._extract_project_insights(modules)
        
        # Infer project characteristics
        tech_stack = self._infer_tech_stack(modules)
        architecture_style = self._infer_architecture_style(modules)
        primary_domain = self._infer_primary_domain(modules)
        key_patterns = self._extract_key_patterns(modules)
        quality_metrics = self._calculate_quality_metrics(modules)
        
        return ProjectContext(
            name=Path(project_path).name,
            description=self._generate_project_description(modules, project_insights),
            architecture_style=architecture_style,
            primary_domain=primary_domain,
            tech_stack=tech_stack,
            key_patterns=key_patterns,
            quality_metrics=quality_metrics,
            modules=modules,
            insights=project_insights
        )
        
    def _run_comprehensive_analysis(self, file_path: str) -> Dict[str, Any]:
        """Run comprehensive analysis on a file and return aggregated results."""
        try:
            # Use the comprehensive analysis engine
            results = self.analysis_engine.analyze_file(file_path)
            return results
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            return self._fallback_analysis(file_path)
            
    def _fallback_analysis(self, file_path: str) -> Dict[str, Any]:
        """Fallback analysis using AST parsing if comprehensive analysis fails."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            tree = ast.parse(source)
            
            return {
                "ast_analysis": {
                    "functions": [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)],
                    "classes": [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)],
                    "imports": [self._extract_import_name(node) for node in ast.walk(tree) 
                               if isinstance(node, (ast.Import, ast.ImportFrom))],
                },
                "basic_metrics": {
                    "lines_of_code": len(source.splitlines()),
                    "functions_count": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                    "classes_count": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                }
            }
        except Exception as e:
            logger.error(f"Fallback analysis failed for {file_path}: {e}")
            return {}
            
    def _extract_module_info(self, file_path: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic module information."""
        module_name = Path(file_path).stem
        
        # Try to extract docstring
        docstring = None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                    isinstance(tree.body[0].value, ast.Constant)):
                    docstring = tree.body[0].value.value
        except:
            pass
            
        return {
            "name": module_name,
            "docstring": docstring,
            "imports": analysis_results.get("ast_analysis", {}).get("imports", []),
            "exports": self._extract_exports(analysis_results),
        }
        
    def _build_function_contexts(self, analysis_results: Dict[str, Any]) -> List[FunctionContext]:
        """Build function contexts from analysis results."""
        functions = []
        
        function_names = analysis_results.get("ast_analysis", {}).get("functions", [])
        
        for func_name in function_names:
            # Extract function-specific insights from various analyzers
            insights = self._extract_function_insights(func_name, analysis_results)
            
            context = FunctionContext(
                name=func_name,
                docstring=None,  # Will be extracted separately
                parameters=[],   # Will be extracted from AST
                return_type=None,
                complexity_score=self._get_function_complexity(func_name, analysis_results),
                security_issues=self._get_function_security_issues(func_name, analysis_results),
                performance_notes=self._get_function_performance_notes(func_name, analysis_results),
                usage_patterns=self._get_function_usage_patterns(func_name, analysis_results),
                related_functions=self._get_related_functions(func_name, analysis_results),
                insights=insights
            )
            
            functions.append(context)
            
        return functions
        
    def _build_class_contexts(self, analysis_results: Dict[str, Any]) -> List[ClassContext]:
        """Build class contexts from analysis results."""
        classes = []
        
        class_names = analysis_results.get("ast_analysis", {}).get("classes", [])
        
        for class_name in class_names:
            insights = self._extract_class_insights(class_name, analysis_results)
            
            context = ClassContext(
                name=class_name,
                docstring=None,
                methods=[],
                attributes=[],
                inheritance=[],
                design_patterns=self._get_class_design_patterns(class_name, analysis_results),
                responsibilities=self._get_class_responsibilities(class_name, analysis_results),
                usage_patterns=self._get_class_usage_patterns(class_name, analysis_results),
                insights=insights
            )
            
            classes.append(context)
            
        return classes
        
    def _extract_architectural_insights(self, analysis_results: Dict[str, Any]) -> List[ContextualInsight]:
        """Extract high-level architectural insights."""
        insights = []
        
        # Extract insights from different analysis dimensions
        if "security_analysis" in analysis_results:
            security_insights = self._extract_security_insights(analysis_results["security_analysis"])
            insights.extend(security_insights)
            
        if "performance_analysis" in analysis_results:
            performance_insights = self._extract_performance_insights(analysis_results["performance_analysis"])
            insights.extend(performance_insights)
            
        if "quality_analysis" in analysis_results:
            quality_insights = self._extract_quality_insights(analysis_results["quality_analysis"])
            insights.extend(quality_insights)
            
        return insights
        
    def _extract_function_insights(self, func_name: str, analysis_results: Dict[str, Any]) -> List[ContextualInsight]:
        """Extract insights specific to a function."""
        insights = []
        
        # Placeholder for function-specific insight extraction
        # This would integrate with specific analyzers
        
        return insights
        
    def _extract_class_insights(self, class_name: str, analysis_results: Dict[str, Any]) -> List[ContextualInsight]:
        """Extract insights specific to a class."""
        insights = []
        
        # Placeholder for class-specific insight extraction
        
        return insights
        
    def _extract_security_insights(self, security_analysis: Dict[str, Any]) -> List[ContextualInsight]:
        """Extract security-related insights."""
        insights = []
        
        # Example security insight extraction
        if "vulnerabilities" in security_analysis:
            for vuln in security_analysis["vulnerabilities"]:
                insight = ContextualInsight(
                    category="security",
                    insight_type="vulnerability",
                    description=vuln.get("description", "Security vulnerability detected"),
                    evidence=[vuln.get("location", "")],
                    confidence=vuln.get("confidence", 0.8),
                    relevance="high",
                    source_analyzer="security_analyzer"
                )
                insights.append(insight)
                
        return insights
        
    def _extract_performance_insights(self, performance_analysis: Dict[str, Any]) -> List[ContextualInsight]:
        """Extract performance-related insights."""
        insights = []
        
        # Example performance insight extraction
        if "complexity_issues" in performance_analysis:
            for issue in performance_analysis["complexity_issues"]:
                insight = ContextualInsight(
                    category="performance",
                    insight_type="complexity",
                    description=issue.get("description", "Performance complexity issue"),
                    evidence=[issue.get("location", "")],
                    confidence=issue.get("confidence", 0.7),
                    relevance="medium",
                    source_analyzer="performance_analyzer"
                )
                insights.append(insight)
                
        return insights
        
    def _extract_quality_insights(self, quality_analysis: Dict[str, Any]) -> List[ContextualInsight]:
        """Extract code quality insights."""
        insights = []
        
        # Example quality insight extraction
        if "quality_issues" in quality_analysis:
            for issue in quality_analysis["quality_issues"]:
                insight = ContextualInsight(
                    category="quality",
                    insight_type="maintainability",
                    description=issue.get("description", "Code quality issue"),
                    evidence=[issue.get("location", "")],
                    confidence=issue.get("confidence", 0.6),
                    relevance="low",
                    source_analyzer="quality_analyzer"
                )
                insights.append(insight)
                
        return insights
        
    # Helper methods (simplified implementations)
    def _infer_module_type(self, analysis_results: Dict[str, Any]) -> str:
        """Infer the type of module based on analysis results."""
        # Simplified inference logic
        return "business_logic"
        
    def _infer_architecture_role(self, analysis_results: Dict[str, Any]) -> str:
        """Infer the architectural role of the module."""
        # Simplified inference logic
        return "core_component"
        
    def _extract_dependencies(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract module dependencies."""
        return analysis_results.get("ast_analysis", {}).get("imports", [])
        
    def _extract_exports(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract module exports."""
        # Simplified export extraction
        return []
        
    def _get_function_complexity(self, func_name: str, analysis_results: Dict[str, Any]) -> Optional[float]:
        """Get complexity score for a function."""
        return None
        
    def _get_function_security_issues(self, func_name: str, analysis_results: Dict[str, Any]) -> List[str]:
        """Get security issues for a function."""
        return []
        
    def _get_function_performance_notes(self, func_name: str, analysis_results: Dict[str, Any]) -> List[str]:
        """Get performance notes for a function."""
        return []
        
    def _get_function_usage_patterns(self, func_name: str, analysis_results: Dict[str, Any]) -> List[str]:
        """Get usage patterns for a function."""
        return []
        
    def _get_related_functions(self, func_name: str, analysis_results: Dict[str, Any]) -> List[str]:
        """Get related functions."""
        return []
        
    def _get_class_design_patterns(self, class_name: str, analysis_results: Dict[str, Any]) -> List[str]:
        """Get design patterns used by a class."""
        return []
        
    def _get_class_responsibilities(self, class_name: str, analysis_results: Dict[str, Any]) -> List[str]:
        """Get responsibilities of a class."""
        return []
        
    def _get_class_usage_patterns(self, class_name: str, analysis_results: Dict[str, Any]) -> List[str]:
        """Get usage patterns for a class."""
        return []
        
    def _find_python_files(self, project_path: str) -> List[str]:
        """Find all Python files in a project."""
        python_files = []
        project_dir = Path(project_path)
        
        for file_path in project_dir.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                python_files.append(str(file_path))
                
        return python_files
        
    def _extract_project_insights(self, modules: List[ModuleContext]) -> List[ContextualInsight]:
        """Extract project-level insights."""
        insights = []
        
        # Aggregate insights from all modules
        for module in modules:
            insights.extend(module.insights)
            
        return insights
        
    def _infer_tech_stack(self, modules: List[ModuleContext]) -> List[str]:
        """Infer technology stack from modules."""
        tech_stack = set()
        
        for module in modules:
            for import_name in module.imports:
                if any(framework in import_name.lower() for framework in 
                      ['django', 'flask', 'fastapi', 'tornado']):
                    tech_stack.add('web_framework')
                elif any(lib in import_name.lower() for lib in 
                        ['numpy', 'pandas', 'sklearn', 'tensorflow', 'pytorch']):
                    tech_stack.add('data_science')
                elif any(lib in import_name.lower() for lib in 
                        ['requests', 'aiohttp', 'urllib']):
                    tech_stack.add('http_client')
                    
        return list(tech_stack)
        
    def _infer_architecture_style(self, modules: List[ModuleContext]) -> str:
        """Infer overall architecture style."""
        # Simplified inference
        return "layered_architecture"
        
    def _infer_primary_domain(self, modules: List[ModuleContext]) -> str:
        """Infer primary business domain."""
        # Simplified inference
        return "software_analysis"
        
    def _extract_key_patterns(self, modules: List[ModuleContext]) -> List[str]:
        """Extract key design patterns used in the project."""
        patterns = set()
        
        for module in modules:
            for cls in module.classes:
                patterns.update(cls.design_patterns)
                
        return list(patterns)
        
    def _calculate_quality_metrics(self, modules: List[ModuleContext]) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        return {
            "modules_count": len(modules),
            "functions_count": sum(len(module.functions) for module in modules),
            "classes_count": sum(len(module.classes) for module in modules),
            "average_complexity": 2.5,  # Placeholder
            "documentation_coverage": 0.65,  # Placeholder
        }
        
    def _generate_project_description(self, modules: List[ModuleContext], 
                                    insights: List[ContextualInsight]) -> str:
        """Generate a high-level project description."""
        return f"Python project with {len(modules)} modules focusing on software analysis and intelligence."
        
    def _extract_import_name(self, node: Union[ast.Import, ast.ImportFrom]) -> str:
        """Extract import name from AST node."""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else ""
        elif isinstance(node, ast.ImportFrom):
            return node.module or ""
        return ""
        
    def format_context_for_llm(self, context: Union[ModuleContext, FunctionContext, ClassContext, ProjectContext]) -> str:
        """
        Format context information in a way that's optimal for LLM consumption.
        
        Args:
            context: The context object to format
            
        Returns:
            str: Formatted context string ready for LLM prompt injection
        """
        if isinstance(context, ProjectContext):
            return self._format_project_context_for_llm(context)
        elif isinstance(context, ModuleContext):
            return self._format_module_context_for_llm(context)
        elif isinstance(context, ClassContext):
            return self._format_class_context_for_llm(context)
        elif isinstance(context, FunctionContext):
            return self._format_function_context_for_llm(context)
        else:
            return str(context)
            
    def _format_project_context_for_llm(self, context: ProjectContext) -> str:
        """Format project context for LLM consumption."""
        formatted = f"""
PROJECT CONTEXT:
Name: {context.name}
Description: {context.description}
Architecture Style: {context.architecture_style}
Primary Domain: {context.primary_domain}
Technology Stack: {', '.join(context.tech_stack)}
Key Patterns: {', '.join(context.key_patterns)}

Quality Metrics:
{json.dumps(context.quality_metrics, indent=2)}

Modules Count: {len(context.modules)}

Key Insights:
{self._format_insights_for_llm(context.insights[:5])}  # Top 5 insights
"""
        return formatted.strip()
        
    def _format_module_context_for_llm(self, context: ModuleContext) -> str:
        """Format module context for LLM consumption."""
        formatted = f"""
MODULE CONTEXT:
Name: {context.name}
Path: {context.path}
Type: {context.module_type}
Architecture Role: {context.architecture_role}

Current Docstring: {context.docstring or "None"}

Dependencies: {', '.join(context.dependencies[:10])}  # Top 10

Functions ({len(context.functions)}):
{', '.join([f.name for f in context.functions[:10]])}  # Top 10

Classes ({len(context.classes)}):
{', '.join([c.name for c in context.classes[:10]])}  # Top 10

Key Insights:
{self._format_insights_for_llm(context.insights[:3])}  # Top 3 insights
"""
        return formatted.strip()
        
    def _format_class_context_for_llm(self, context: ClassContext) -> str:
        """Format class context for LLM consumption."""
        formatted = f"""
CLASS CONTEXT:
Name: {context.name}
Current Docstring: {context.docstring or "None"}

Methods: {', '.join(context.methods)}
Attributes: {', '.join(context.attributes)}
Inheritance: {', '.join(context.inheritance)}
Design Patterns: {', '.join(context.design_patterns)}
Responsibilities: {', '.join(context.responsibilities)}
Usage Patterns: {', '.join(context.usage_patterns)}

Key Insights:
{self._format_insights_for_llm(context.insights)}
"""
        return formatted.strip()
        
    def _format_function_context_for_llm(self, context: FunctionContext) -> str:
        """Format function context for LLM consumption."""
        formatted = f"""
FUNCTION CONTEXT:
Name: {context.name}
Current Docstring: {context.docstring or "None"}

Parameters: {len(context.parameters)} parameters
Return Type: {context.return_type or "Unknown"}
Complexity Score: {context.complexity_score}

Security Issues: {', '.join(context.security_issues) if context.security_issues else "None"}
Performance Notes: {', '.join(context.performance_notes) if context.performance_notes else "None"}
Usage Patterns: {', '.join(context.usage_patterns) if context.usage_patterns else "None"}
Related Functions: {', '.join(context.related_functions) if context.related_functions else "None"}

Key Insights:
{self._format_insights_for_llm(context.insights)}
"""
        return formatted.strip()
        
    def _format_insights_for_llm(self, insights: List[ContextualInsight]) -> str:
        """Format insights for LLM consumption."""
        if not insights:
            return "No specific insights available."
            
        formatted_insights = []
        for insight in insights:
            formatted = f"- {insight.category.upper()}: {insight.description} (Confidence: {insight.confidence:.1f}, Relevance: {insight.relevance})"
            formatted_insights.append(formatted)
            
        return '\n'.join(formatted_insights)