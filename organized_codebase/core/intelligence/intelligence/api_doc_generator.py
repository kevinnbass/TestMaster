"""
API Documentation Generator

Comprehensive API documentation generator that creates detailed documentation
for Python APIs based on classical analysis and code structure understanding.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.context_builder import AnalysisContextBuilder, ProjectContext, ModuleContext
from ..core.llm_integration import LLMIntegration
from ..core.quality_assessor import DocumentationQualityAssessor

logger = logging.getLogger(__name__)


@dataclass
class ApiDocConfig:
    """Configuration for API documentation generation."""
    include_private: bool = False
    include_inherited: bool = True
    include_examples: bool = True
    include_source_links: bool = True
    include_type_annotations: bool = True
    generate_index: bool = True
    output_format: str = "markdown"  # "markdown", "rst", "html"
    group_by_module: bool = True
    include_performance_notes: bool = True
    include_security_notes: bool = True


class ApiDocumentationGenerator:
    """
    Comprehensive API documentation generator.
    
    Features:
    - Complete API coverage with public/private filtering
    - Type annotation documentation
    - Parameter and return value documentation
    - Exception documentation from analysis
    - Performance and security insights
    - Cross-reference generation
    - Multiple output formats
    - Source code linking
    """
    
    def __init__(self, 
                 llm_integration: LLMIntegration,
                 context_builder: AnalysisContextBuilder,
                 quality_assessor: Optional[DocumentationQualityAssessor] = None):
        """
        Initialize the API documentation generator.
        
        Args:
            llm_integration: LLM integration for generation
            context_builder: Analysis context builder
            quality_assessor: Quality assessor for validation
        """
        self.llm_integration = llm_integration
        self.context_builder = context_builder
        self.quality_assessor = quality_assessor
        
    async def generate_api_documentation(self, 
                                       project_path: str,
                                       config: Optional[ApiDocConfig] = None) -> Dict[str, str]:
        """
        Generate comprehensive API documentation for a project.
        
        Args:
            project_path: Path to the project directory
            config: API documentation configuration
            
        Returns:
            Dict[str, str]: Generated documentation files mapped to file names
        """
        logger.info(f"Generating API documentation for project: {project_path}")
        
        config = config or ApiDocConfig()
        
        # Build project context
        project_context = self.context_builder.build_project_context(project_path)
        
        # Generate documentation files
        doc_files = {}
        
        # Generate main index file
        if config.generate_index:
            doc_files["index.md"] = await self._generate_api_index(project_context, config)
        
        # Generate module documentation
        if config.group_by_module:
            for module in project_context.modules:
                module_doc = await self._generate_module_api_doc(module, config)
                if module_doc:
                    doc_files[f"{module.name}.md"] = module_doc
        else:
            # Generate unified API documentation
            doc_files["api.md"] = await self._generate_unified_api_doc(project_context, config)
        
        # Quality assessment
        if self.quality_assessor:
            overall_quality = 0.0
            for filename, content in doc_files.items():
                quality_score = await self.quality_assessor.assess_api_doc_quality(
                    content, project_context
                )
                overall_quality += quality_score
                logger.info(f"API doc quality for {filename}: {quality_score:.2f}")
            
            overall_quality /= len(doc_files)
            logger.info(f"Overall API documentation quality: {overall_quality:.2f}")
        
        logger.info(f"Generated {len(doc_files)} API documentation files")
        return doc_files
        
    async def _generate_api_index(self, project_context: ProjectContext, 
                                config: ApiDocConfig) -> str:
        """Generate the main API documentation index."""
        index_parts = [
            f"# {project_context.name} API Documentation",
            "",
            project_context.description or f"API documentation for {project_context.name}.",
            "",
            "## Overview",
            "",
            f"**Architecture**: {project_context.architecture_style.replace('_', ' ').title()}",
            f"**Domain**: {project_context.primary_domain.replace('_', ' ').title()}",
            f"**Modules**: {len(project_context.modules)}",
            "",
        ]
        
        # Add technology stack
        if project_context.tech_stack:
            index_parts.extend([
                "**Technology Stack:**",
                ""
            ])
            for tech in project_context.tech_stack:
                index_parts.append(f"- {tech.replace('_', ' ').title()}")
            index_parts.append("")
        
        # Add modules index
        index_parts.extend([
            "## Modules",
            ""
        ])
        
        for module in project_context.modules:
            # Count public APIs
            public_functions = [f for f in module.functions if not f.name.startswith('_')]
            public_classes = [c for c in module.classes if not c.name.startswith('_')]
            
            if public_functions or public_classes:
                index_parts.append(f"### [{module.name}]({module.name}.md)")
                index_parts.append("")
                index_parts.append(f"*{module.module_type.replace('_', ' ').title()} module*")
                index_parts.append("")
                
                if public_functions:
                    index_parts.append(f"**Functions ({len(public_functions)}):** " + 
                                     ", ".join([f"`{f.name}()`" for f in public_functions[:5]]))
                    if len(public_functions) > 5:
                        index_parts.append("...")
                    index_parts.append("")
                
                if public_classes:
                    index_parts.append(f"**Classes ({len(public_classes)}):** " + 
                                     ", ".join([f"`{c.name}`" for c in public_classes[:5]]))
                    if len(public_classes) > 5:
                        index_parts.append("...")
                    index_parts.append("")
                
                index_parts.append("---")
                index_parts.append("")
        
        # Add quick reference
        index_parts.extend([
            "## Quick Reference",
            "",
            "### Most Common Functions",
            ""
        ])
        
        # Find most commonly used functions (based on name patterns)
        common_functions = []
        for module in project_context.modules:
            for func in module.functions:
                if not func.name.startswith('_') and any(keyword in func.name.lower() 
                    for keyword in ['get', 'set', 'create', 'update', 'delete', 'find', 'search']):
                    common_functions.append((module.name, func.name))
        
        for module_name, func_name in common_functions[:10]:
            index_parts.append(f"- [`{module_name}.{func_name}()`]({module_name}.md#{func_name.lower()})")
        
        index_parts.append("")
        
        return "\n".join(index_parts)
        
    async def _generate_module_api_doc(self, module_context: ModuleContext, 
                                     config: ApiDocConfig) -> str:
        """Generate API documentation for a single module."""
        # Filter public APIs
        functions = [f for f in module_context.functions 
                    if config.include_private or not f.name.startswith('_')]
        classes = [c for c in module_context.classes 
                  if config.include_private or not c.name.startswith('_')]
        
        if not functions and not classes:
            return ""  # No public API to document
        
        logger.info(f"Generating API doc for module {module_context.name}")
        
        # Generate using LLM with structured prompt
        context_str = self.context_builder.format_context_for_llm(module_context)
        
        # Create API structure representation
        api_structure = self._build_api_structure(module_context, config)
        
        response = await self.llm_integration.generate_documentation(
            doc_type="api_documentation",
            context=context_str,
            code=api_structure,
            style="markdown"
        )
        
        # Post-process the generated documentation
        api_doc = self._post_process_api_doc(response.content, module_context, config)
        
        return api_doc
        
    async def _generate_unified_api_doc(self, project_context: ProjectContext, 
                                      config: ApiDocConfig) -> str:
        """Generate unified API documentation for entire project."""
        logger.info("Generating unified API documentation")
        
        # Build comprehensive API structure
        unified_structure = []
        
        for module in project_context.modules:
            module_api = self._build_api_structure(module, config)
            if module_api.strip():
                unified_structure.append(f"## Module: {module.name}")
                unified_structure.append("")
                unified_structure.append(module_api)
                unified_structure.append("")
        
        context_str = self.context_builder.format_context_for_llm(project_context)
        api_structure = "\n".join(unified_structure)
        
        response = await self.llm_integration.generate_documentation(
            doc_type="api_documentation",
            context=context_str,
            code=api_structure,
            style="markdown"
        )
        
        return response.content
        
    def _build_api_structure(self, module_context: ModuleContext, 
                           config: ApiDocConfig) -> str:
        """Build API structure representation for a module."""
        structure_parts = []
        
        # Module header
        structure_parts.extend([
            f"# {module_context.name}",
            "",
            f"**Type**: {module_context.module_type.replace('_', ' ').title()}",
            f"**Role**: {module_context.architecture_role.replace('_', ' ').title()}",
            ""
        ])
        
        # Module docstring
        if module_context.docstring:
            structure_parts.extend([
                module_context.docstring,
                ""
            ])
        
        # Functions section
        functions = [f for f in module_context.functions 
                    if config.include_private or not f.name.startswith('_')]
        
        if functions:
            structure_parts.extend([
                "## Functions",
                ""
            ])
            
            for func in functions:
                structure_parts.append(f"### {func.name}")
                structure_parts.append("")
                
                # Function signature (would be extracted from AST in real implementation)
                params_str = ', '.join([p.get('name', '') for p in func.parameters])
                structure_parts.append(f"```python")
                structure_parts.append(f"def {func.name}({params_str}):")
                structure_parts.append("```")
                structure_parts.append("")
                
                # Current docstring
                if func.docstring:
                    structure_parts.append(func.docstring)
                    structure_parts.append("")
                
                # Analysis insights
                if func.complexity_score:
                    structure_parts.append(f"**Complexity**: {func.complexity_score}")
                    structure_parts.append("")
                
                if func.security_issues:
                    structure_parts.append(f"**Security Notes**: {len(func.security_issues)} issues identified")
                    structure_parts.append("")
                
                if func.performance_notes:
                    structure_parts.append("**Performance Notes**:")
                    for note in func.performance_notes[:2]:
                        structure_parts.append(f"- {note}")
                    structure_parts.append("")
                
                structure_parts.append("---")
                structure_parts.append("")
        
        # Classes section
        classes = [c for c in module_context.classes 
                  if config.include_private or not c.name.startswith('_')]
        
        if classes:
            structure_parts.extend([
                "## Classes",
                ""
            ])
            
            for cls in classes:
                structure_parts.append(f"### {cls.name}")
                structure_parts.append("")
                
                # Class signature
                inheritance_str = f"({', '.join(cls.inheritance)})" if cls.inheritance else ""
                structure_parts.append(f"```python")
                structure_parts.append(f"class {cls.name}{inheritance_str}:")
                structure_parts.append("```")
                structure_parts.append("")
                
                # Current docstring
                if cls.docstring:
                    structure_parts.append(cls.docstring)
                    structure_parts.append("")
                
                # Attributes
                if cls.attributes:
                    structure_parts.append("**Attributes**:")
                    for attr in cls.attributes:
                        structure_parts.append(f"- `{attr}`")
                    structure_parts.append("")
                
                # Methods
                if cls.methods:
                    structure_parts.append("**Methods**:")
                    public_methods = [m for m in cls.methods 
                                    if config.include_private or not m.startswith('_')]
                    for method in public_methods[:10]:  # Limit to first 10
                        structure_parts.append(f"- `{method}()`")
                    structure_parts.append("")
                
                # Design patterns
                if cls.design_patterns:
                    structure_parts.append("**Design Patterns**:")
                    for pattern in cls.design_patterns:
                        structure_parts.append(f"- {pattern.replace('_', ' ').title()}")
                    structure_parts.append("")
                
                structure_parts.append("---")
                structure_parts.append("")
        
        return "\n".join(structure_parts)
        
    def _post_process_api_doc(self, raw_doc: str, module_context: ModuleContext, 
                            config: ApiDocConfig) -> str:
        """Post-process generated API documentation."""
        doc_lines = raw_doc.split('\n')
        processed_lines = []
        
        for line in doc_lines:
            # Add source links if enabled
            if config.include_source_links and line.startswith('### '):
                function_or_class = line[4:].strip()
                source_link = f"[Source]({module_context.path}#{function_or_class.lower()})"
                processed_lines.append(f"{line} {source_link}")
            else:
                processed_lines.append(line)
        
        # Add navigation footer
        processed_lines.extend([
            "",
            "---",
            "",
            "**Navigation**: [API Index](index.md) | [Project Home](../README.md)",
            ""
        ])
        
        return "\n".join(processed_lines)
        
    def save_api_documentation(self, doc_files: Dict[str, str], 
                             output_dir: str) -> List[str]:
        """
        Save generated API documentation files to directory.
        
        Args:
            doc_files: Generated documentation files
            output_dir: Output directory path
            
        Returns:
            List[str]: Paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for filename, content in doc_files.items():
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            saved_files.append(str(file_path))
            logger.info(f"Saved API documentation: {file_path}")
        
        return saved_files
        
    async def generate_function_doc(self, file_path: str, function_name: str) -> str:
        """
        Generate detailed documentation for a specific function.
        
        Args:
            file_path: Path to file containing the function
            function_name: Name of the function
            
        Returns:
            str: Generated function documentation
        """
        function_context = self.context_builder.build_function_context(file_path, function_name)
        if not function_context:
            raise ValueError(f"Function {function_name} not found in {file_path}")
        
        # Extract detailed function information
        func_info = self._extract_function_info(file_path, function_name)
        
        # Build comprehensive context
        context_str = self.context_builder.format_context_for_llm(function_context)
        
        prompt = f"""
Generate comprehensive API documentation for this function:

FUNCTION CONTEXT:
{context_str}

FUNCTION SIGNATURE:
{func_info['signature']}

FUNCTION BODY PREVIEW:
{func_info['body_preview']}

Generate detailed documentation including:
1. Clear description of functionality
2. Complete parameter documentation with types
3. Return value documentation
4. Exception documentation
5. Usage examples
6. Performance characteristics
7. Security considerations
8. Related functions or methods

Format as clean markdown suitable for API documentation.
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type="api_documentation",
            context=prompt,
            code=func_info['signature'],
            style="markdown"
        )
        
        return response.content
        
    async def generate_class_doc(self, file_path: str, class_name: str) -> str:
        """
        Generate detailed documentation for a specific class.
        
        Args:
            file_path: Path to file containing the class
            class_name: Name of the class
            
        Returns:
            str: Generated class documentation
        """
        class_context = self.context_builder.build_class_context(file_path, class_name)
        if not class_context:
            raise ValueError(f"Class {class_name} not found in {file_path}")
        
        # Extract detailed class information
        class_info = self._extract_class_info(file_path, class_name)
        
        # Build comprehensive context
        context_str = self.context_builder.format_context_for_llm(class_context)
        
        prompt = f"""
Generate comprehensive API documentation for this class:

CLASS CONTEXT:
{context_str}

CLASS SIGNATURE:
{class_info['signature']}

CLASS METHODS:
{class_info['methods_preview']}

Generate detailed documentation including:
1. Clear description of class purpose and responsibilities
2. Detailed attribute documentation
3. Constructor documentation
4. Method overview with key methods detailed
5. Usage examples
6. Inheritance relationships
7. Design patterns used
8. Thread safety considerations
9. Performance characteristics

Format as clean markdown suitable for API documentation.
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type="api_documentation",
            context=prompt,
            code=class_info['signature'],
            style="markdown"
        )
        
        return response.content
        
    def _extract_function_info(self, file_path: str, function_name: str) -> Dict[str, str]:
        """Extract detailed function information from source code."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                # Extract signature
                signature = self._build_function_signature(node)
                
                # Extract body preview (first few lines)
                body_lines = content.split('\n')[node.lineno:node.lineno+10]
                body_preview = '\n'.join(body_lines)
                
                return {
                    'signature': signature,
                    'body_preview': body_preview,
                    'lineno': node.lineno,
                    'decorators': [ast.unparse(d) for d in node.decorator_list]
                }
        
        return {'signature': f'def {function_name}():', 'body_preview': '', 'lineno': 0, 'decorators': []}
        
    def _extract_class_info(self, file_path: str, class_name: str) -> Dict[str, str]:
        """Extract detailed class information from source code."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Extract signature
                signature = self._build_class_signature(node)
                
                # Extract methods overview
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                methods_preview = '\n'.join([f"- {method}" for method in methods[:10]])
                
                return {
                    'signature': signature,
                    'methods_preview': methods_preview,
                    'lineno': node.lineno,
                    'decorators': [ast.unparse(d) for d in node.decorator_list],
                    'methods': methods
                }
        
        return {'signature': f'class {class_name}:', 'methods_preview': '', 'lineno': 0, 'decorators': [], 'methods': []}
        
    def _build_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Build function signature from AST node."""
        args = []
        
        # Regular arguments
        for arg in func_node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Handle defaults
        defaults = func_node.args.defaults
        if defaults:
            for i, default in enumerate(defaults):
                arg_index = len(args) - len(defaults) + i
                args[arg_index] += f" = {ast.unparse(default)}"
        
        # Return annotation
        returns = ""
        if func_node.returns:
            returns = f" -> {ast.unparse(func_node.returns)}"
        
        # Async prefix
        async_prefix = "async " if isinstance(func_node, ast.AsyncFunctionDef) else ""
        
        return f"{async_prefix}def {func_node.name}({', '.join(args)}){returns}:"
        
    def _build_class_signature(self, class_node: ast.ClassDef) -> str:
        """Build class signature from AST node."""
        bases = [ast.unparse(base) for base in class_node.bases]
        base_str = f"({', '.join(bases)})" if bases else ""
        
        return f"class {class_node.name}{base_str}:"