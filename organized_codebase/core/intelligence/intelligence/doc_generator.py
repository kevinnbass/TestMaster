"""
Core Documentation Generation Orchestrator

Main engine that coordinates analysis context building, LLM integration,
and specialized documentation generators to produce high-quality documentation.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

from .context_builder import AnalysisContextBuilder, ModuleContext, FunctionContext, ClassContext, ProjectContext
from .llm_integration import LLMIntegration, LLMProvider, LLMConfig, create_default_llm_integration
from .quality_assessor import DocumentationQualityAssessor

logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    default_style: str = "google"  # "google", "numpy", "sphinx"
    include_examples: bool = True
    include_security_notes: bool = True
    include_performance_notes: bool = True
    include_architecture_insights: bool = True
    max_concurrent_generations: int = 3
    enable_quality_assessment: bool = True
    minimum_quality_score: float = 0.7
    auto_retry_on_low_quality: bool = True
    cache_results: bool = True


@dataclass
class GenerationResult:
    """Result of documentation generation."""
    content: str
    doc_type: str
    target: str  # file path, function name, class name, etc.
    quality_score: float
    generation_time: float
    tokens_used: int
    provider_used: str
    metadata: Dict[str, Any]


class DocumentationEngine:
    """
    Main documentation generation engine that orchestrates the entire process.
    
    This engine:
    1. Builds comprehensive context from classical analysis
    2. Generates documentation using LLMs with sophisticated prompts
    3. Assesses and validates documentation quality
    4. Provides batch processing and caching capabilities
    """
    
    def __init__(self, 
                 config: Optional[DocumentationConfig] = None,
                 llm_integration: Optional[LLMIntegration] = None,
                 analysis_engine: Optional['CodeAnalysisEngine'] = None):
        """
        Initialize the documentation engine.
        
        Args:
            config: Documentation generation configuration
            llm_integration: LLM integration instance (auto-created if None)
            analysis_engine: Classical analysis engine (auto-created if None)
        """
        self.config = config or DocumentationConfig()
        self.llm_integration = llm_integration or create_default_llm_integration()
        self.context_builder = AnalysisContextBuilder(analysis_engine)
        self.quality_assessor = DocumentationQualityAssessor() if self.config.enable_quality_assessment else None
        
        # Results cache
        self.results_cache = {} if self.config.cache_results else None
        
        logger.info("Documentation engine initialized")
        
    async def generate_docstrings(self, 
                                file_path: str,
                                style: Optional[str] = None,
                                target_functions: Optional[List[str]] = None,
                                target_classes: Optional[List[str]] = None) -> Dict[str, GenerationResult]:
        """
        Generate docstrings for functions and classes in a file.
        
        Args:
            file_path: Path to the Python file
            style: Docstring style ("google", "numpy", "sphinx")
            target_functions: Specific functions to document (all if None)
            target_classes: Specific classes to document (all if None)
            
        Returns:
            Dict[str, GenerationResult]: Generated docstrings mapped to function/class names
        """
        logger.info(f"Generating docstrings for {file_path}")
        
        # Build module context
        module_context = self.context_builder.build_module_context(file_path)
        
        # Prepare generation tasks
        tasks = []
        
        # Functions
        for func_context in module_context.functions:
            if target_functions is None or func_context.name in target_functions:
                tasks.append(self._generate_function_docstring(func_context, style))
                
        # Classes
        for class_context in module_context.classes:
            if target_classes is None or class_context.name in target_classes:
                tasks.append(self._generate_class_docstring(class_context, style))
                
        # Execute in parallel with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
                
        results = await asyncio.gather(*[bounded_task(task) for task in tasks], return_exceptions=True)
        
        # Process results
        docstrings = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Docstring generation failed: {result}")
            else:
                docstrings[result.target] = result
                
        logger.info(f"Generated {len(docstrings)} docstrings for {file_path}")
        return docstrings
        
    async def generate_readme(self, 
                            project_path: str,
                            style: Optional[str] = None) -> GenerationResult:
        """
        Generate a comprehensive README.md for a project.
        
        Args:
            project_path: Path to the project directory
            style: Documentation style
            
        Returns:
            GenerationResult: Generated README content and metadata
        """
        logger.info(f"Generating README for project: {project_path}")
        
        # Build project context
        project_context = self.context_builder.build_project_context(project_path)
        
        # Format context for LLM
        context_str = self.context_builder.format_context_for_llm(project_context)
        
        # Create code structure representation
        code_structure = self._build_project_structure_representation(project_context)
        
        # Generate README
        response = await self.llm_integration.generate_documentation(
            doc_type="readme_project",
            context=context_str,
            code=code_structure,
            style=style or self.config.default_style
        )
        
        # Assess quality
        quality_score = 1.0
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_readme_quality(
                response.content, project_context
            )
            
        # Retry if quality is too low
        if (quality_score < self.config.minimum_quality_score and 
            self.config.auto_retry_on_low_quality):
            logger.warning(f"README quality too low ({quality_score:.2f}), retrying...")
            response = await self._retry_generation_with_enhanced_prompt(
                "readme_project", context_str, code_structure, style or self.config.default_style
            )
            if self.quality_assessor:
                quality_score = await self.quality_assessor.assess_readme_quality(
                    response.content, project_context
                )
        
        result = GenerationResult(
            content=response.content,
            doc_type="readme",
            target=project_path,
            quality_score=quality_score,
            generation_time=response.generation_time,
            tokens_used=response.tokens_used,
            provider_used=response.provider.value,
            metadata=response.metadata
        )
        
        logger.info(f"Generated README for {project_path} (quality: {quality_score:.2f})")
        return result
        
    async def generate_api_documentation(self, 
                                       project_path: str,
                                       include_private: bool = False,
                                       style: Optional[str] = None) -> GenerationResult:
        """
        Generate comprehensive API documentation for a project.
        
        Args:
            project_path: Path to the project directory
            include_private: Include private methods and classes
            style: Documentation style
            
        Returns:
            GenerationResult: Generated API documentation
        """
        logger.info(f"Generating API documentation for: {project_path}")
        
        # Build project context
        project_context = self.context_builder.build_project_context(project_path)
        
        # Filter public API
        api_structure = self._extract_public_api_structure(project_context, include_private)
        
        # Format context
        context_str = self.context_builder.format_context_for_llm(project_context)
        
        # Generate API documentation
        response = await self.llm_integration.generate_documentation(
            doc_type="api_documentation",
            context=context_str,
            code=api_structure,
            style=style or self.config.default_style
        )
        
        # Assess quality
        quality_score = 1.0
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_api_doc_quality(
                response.content, project_context
            )
        
        result = GenerationResult(
            content=response.content,
            doc_type="api_documentation",
            target=project_path,
            quality_score=quality_score,
            generation_time=response.generation_time,
            tokens_used=response.tokens_used,
            provider_used=response.provider.value,
            metadata=response.metadata
        )
        
        logger.info(f"Generated API documentation for {project_path} (quality: {quality_score:.2f})")
        return result
        
    async def generate_module_documentation(self, 
                                          file_path: str,
                                          style: Optional[str] = None) -> GenerationResult:
        """
        Generate comprehensive documentation for a single module.
        
        Args:
            file_path: Path to the Python module
            style: Documentation style
            
        Returns:
            GenerationResult: Generated module documentation
        """
        logger.info(f"Generating module documentation for: {file_path}")
        
        # Build module context
        module_context = self.context_builder.build_module_context(file_path)
        
        # Format context
        context_str = self.context_builder.format_context_for_llm(module_context)
        
        # Create module structure representation
        module_structure = self._build_module_structure_representation(module_context)
        
        # Generate documentation
        response = await self.llm_integration.generate_documentation(
            doc_type="module_documentation",
            context=context_str,
            code=module_structure,
            style=style or self.config.default_style
        )
        
        # Assess quality
        quality_score = 1.0
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_module_doc_quality(
                response.content, module_context
            )
        
        result = GenerationResult(
            content=response.content,
            doc_type="module_documentation",
            target=file_path,
            quality_score=quality_score,
            generation_time=response.generation_time,
            tokens_used=response.tokens_used,
            provider_used=response.provider.value,
            metadata=response.metadata
        )
        
        logger.info(f"Generated module documentation for {file_path} (quality: {quality_score:.2f})")
        return result
        
    async def generate_batch(self, 
                           requests: List[Dict[str, Any]]) -> List[GenerationResult]:
        """
        Generate multiple documentation pieces in batch.
        
        Args:
            requests: List of generation requests with type and parameters
            
        Returns:
            List[GenerationResult]: Generated documentation results
        """
        logger.info(f"Processing batch of {len(requests)} documentation requests")
        
        # Create tasks based on request type
        tasks = []
        for req in requests:
            req_type = req.get("type")
            if req_type == "docstrings":
                tasks.append(self.generate_docstrings(**req.get("params", {})))
            elif req_type == "readme":
                tasks.append(self.generate_readme(**req.get("params", {})))
            elif req_type == "api_documentation":
                tasks.append(self.generate_api_documentation(**req.get("params", {})))
            elif req_type == "module_documentation":
                tasks.append(self.generate_module_documentation(**req.get("params", {})))
            else:
                logger.warning(f"Unknown request type: {req_type}")
                
        # Execute batch with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch generation failed: {result}")
            elif isinstance(result, dict):  # docstrings return dict
                valid_results.extend(result.values())
            else:  # single result
                valid_results.append(result)
                
        logger.info(f"Completed batch generation: {len(valid_results)} successful results")
        return valid_results
        
    async def _generate_function_docstring(self, 
                                         func_context: FunctionContext,
                                         style: Optional[str]) -> GenerationResult:
        """Generate docstring for a single function."""
        # Format context
        context_str = self.context_builder.format_context_for_llm(func_context)
        
        # Create function signature representation
        func_code = f"def {func_context.name}({', '.join([p.get('name', '') for p in func_context.parameters])}):"
        
        # Generate docstring
        response = await self.llm_integration.generate_documentation(
            doc_type="docstring_function",
            context=context_str,
            code=func_code,
            style=style or self.config.default_style
        )
        
        # Assess quality
        quality_score = 1.0
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_docstring_quality(
                response.content, func_context
            )
        
        return GenerationResult(
            content=response.content,
            doc_type="function_docstring",
            target=func_context.name,
            quality_score=quality_score,
            generation_time=response.generation_time,
            tokens_used=response.tokens_used,
            provider_used=response.provider.value,
            metadata=response.metadata
        )
        
    async def _generate_class_docstring(self, 
                                      class_context: ClassContext,
                                      style: Optional[str]) -> GenerationResult:
        """Generate docstring for a single class."""
        # Format context
        context_str = self.context_builder.format_context_for_llm(class_context)
        
        # Create class signature representation
        inheritance_str = f"({', '.join(class_context.inheritance)})" if class_context.inheritance else ""
        class_code = f"class {class_context.name}{inheritance_str}:"
        
        # Generate docstring
        response = await self.llm_integration.generate_documentation(
            doc_type="docstring_class",
            context=context_str,
            code=class_code,
            style=style or self.config.default_style
        )
        
        # Assess quality
        quality_score = 1.0
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_docstring_quality(
                response.content, class_context
            )
        
        return GenerationResult(
            content=response.content,
            doc_type="class_docstring",
            target=class_context.name,
            quality_score=quality_score,
            generation_time=response.generation_time,
            tokens_used=response.tokens_used,
            provider_used=response.provider.value,
            metadata=response.metadata
        )
        
    def _build_project_structure_representation(self, project_context: ProjectContext) -> str:
        """Build a text representation of project structure."""
        structure_lines = [
            f"Project: {project_context.name}",
            f"Description: {project_context.description}",
            f"Architecture: {project_context.architecture_style}",
            f"Domain: {project_context.primary_domain}",
            f"Tech Stack: {', '.join(project_context.tech_stack)}",
            "",
            "Modules:"
        ]
        
        for module in project_context.modules:
            structure_lines.append(f"  - {module.name} ({module.module_type})")
            structure_lines.append(f"    Functions: {len(module.functions)}")
            structure_lines.append(f"    Classes: {len(module.classes)}")
            if module.insights:
                key_insights = [insight.description for insight in module.insights[:2]]
                structure_lines.append(f"    Key insights: {'; '.join(key_insights)}")
            structure_lines.append("")
            
        return "\n".join(structure_lines)
        
    def _build_module_structure_representation(self, module_context: ModuleContext) -> str:
        """Build a text representation of module structure."""
        structure_lines = [
            f"Module: {module_context.name}",
            f"Type: {module_context.module_type}",
            f"Role: {module_context.architecture_role}",
            f"Dependencies: {', '.join(module_context.dependencies[:5])}",
            "",
            "Functions:"
        ]
        
        for func in module_context.functions:
            structure_lines.append(f"  - {func.name}")
            if func.complexity_score:
                structure_lines.append(f"    Complexity: {func.complexity_score}")
            if func.security_issues:
                structure_lines.append(f"    Security: {len(func.security_issues)} issues")
                
        structure_lines.append("\nClasses:")
        for cls in module_context.classes:
            structure_lines.append(f"  - {cls.name}")
            structure_lines.append(f"    Methods: {len(cls.methods)}")
            if cls.design_patterns:
                structure_lines.append(f"    Patterns: {', '.join(cls.design_patterns)}")
                
        return "\n".join(structure_lines)
        
    def _extract_public_api_structure(self, project_context: ProjectContext, 
                                    include_private: bool) -> str:
        """Extract public API structure from project context."""
        api_lines = [
            f"Public API for {project_context.name}",
            "=" * 50,
            ""
        ]
        
        for module in project_context.modules:
            # Skip if module doesn't export anything public
            public_functions = [f for f in module.functions if not f.name.startswith('_') or include_private]
            public_classes = [c for c in module.classes if not c.name.startswith('_') or include_private]
            
            if public_functions or public_classes:
                api_lines.append(f"## Module: {module.name}")
                api_lines.append("")
                
                if public_functions:
                    api_lines.append("### Functions")
                    for func in public_functions:
                        params_str = ', '.join([p.get('name', '') for p in func.parameters])
                        api_lines.append(f"- `{func.name}({params_str})`")
                        if func.insights:
                            key_insight = func.insights[0].description if func.insights else ""
                            api_lines.append(f"  {key_insight}")
                    api_lines.append("")
                    
                if public_classes:
                    api_lines.append("### Classes")
                    for cls in public_classes:
                        api_lines.append(f"- `{cls.name}`")
                        public_methods = [m for m in cls.methods if not m.startswith('_') or include_private]
                        if public_methods:
                            api_lines.append(f"  Methods: {', '.join(public_methods[:5])}")
                        if cls.design_patterns:
                            api_lines.append(f"  Patterns: {', '.join(cls.design_patterns)}")
                    api_lines.append("")
                    
        return "\n".join(api_lines)
        
    async def _retry_generation_with_enhanced_prompt(self, doc_type: str, context: str, 
                                                   code: str, style: str):
        """Retry generation with an enhanced prompt for better quality."""
        enhanced_context = f"""
QUALITY IMPROVEMENT REQUEST:
The previous generation did not meet quality standards. Please focus on:
- More detailed and accurate descriptions
- Better organization and structure
- More practical examples and use cases
- Clear, professional language
- Comprehensive coverage of all important aspects

{context}
"""
        
        return await self.llm_integration.generate_documentation(
            doc_type=doc_type,
            context=enhanced_context,
            code=code,
            style=style,
            temperature=0.05  # Lower temperature for more focused output
        )
        
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about documentation generation."""
        return {
            "total_generations": getattr(self, '_total_generations', 0),
            "successful_generations": getattr(self, '_successful_generations', 0),
            "failed_generations": getattr(self, '_failed_generations', 0),
            "average_quality_score": getattr(self, '_average_quality_score', 0.0),
            "total_tokens_used": getattr(self, '_total_tokens_used', 0),
            "cache_hits": len(self.results_cache) if self.results_cache else 0
        }
        
    def clear_cache(self):
        """Clear all caches."""
        if self.results_cache:
            self.results_cache.clear()
        self.context_builder.context_cache.clear()
        self.llm_integration.clear_cache()