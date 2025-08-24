"""
Intelligent Docstring Generator

Specialized generator for high-quality docstrings with support for multiple styles
and integration with classical analysis insights.
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..core.context_builder import AnalysisContextBuilder, FunctionContext, ClassContext
from ..core.llm_integration import LLMIntegration
from ..core.quality_assessor import DocumentationQualityAssessor

logger = logging.getLogger(__name__)


class DocstringGenerator:
    """
    Intelligent docstring generator with style support and context awareness.
    
    Features:
    - Multiple docstring styles (Google, NumPy, Sphinx)
    - Classical analysis integration for accuracy
    - Automatic parameter type inference
    - Exception documentation from security analysis
    - Performance notes from complexity analysis
    - Quality assessment and iterative improvement
    """
    
    def __init__(self, 
                 llm_integration: LLMIntegration,
                 context_builder: AnalysisContextBuilder,
                 quality_assessor: Optional[DocumentationQualityAssessor] = None):
        """
        Initialize the docstring generator.
        
        Args:
            llm_integration: LLM integration for generation
            context_builder: Analysis context builder
            quality_assessor: Quality assessor for validation
        """
        self.llm_integration = llm_integration
        self.context_builder = context_builder
        self.quality_assessor = quality_assessor
        
        # Style templates
        self.style_templates = self._load_style_templates()
        
    def _load_style_templates(self) -> Dict[str, Dict[str, str]]:
        """Load templates for different docstring styles."""
        return {
            "google": {
                "function": '''"""
{summary}

{detailed_description}

Args:
{args}

Returns:
{returns}

Raises:
{raises}

Example:
{example}
"""''',
                "class": '''"""
{summary}

{detailed_description}

Attributes:
{attributes}

Example:
{example}
"""'''
            },
            "numpy": {
                "function": '''"""
{summary}

{detailed_description}

Parameters
----------
{parameters}

Returns
-------
{returns}

Raises
------
{raises}

Examples
--------
{examples}
"""''',
                "class": '''"""
{summary}

{detailed_description}

Attributes
----------
{attributes}

Examples
--------
{examples}
"""'''
            },
            "sphinx": {
                "function": '''"""
{summary}

{detailed_description}

{param_docs}

{return_docs}

{raises_docs}

{example_docs}
"""''',
                "class": '''"""
{summary}

{detailed_description}

{attribute_docs}

{example_docs}
"""'''
            }
        }
    
    async def generate_function_docstring(self, 
                                        file_path: str,
                                        function_name: str,
                                        style: str = "google") -> str:
        """
        Generate a docstring for a specific function.
        
        Args:
            file_path: Path to the file containing the function
            function_name: Name of the function
            style: Docstring style ("google", "numpy", "sphinx")
            
        Returns:
            str: Generated docstring
        """
        logger.info(f"Generating {style} docstring for {function_name} in {file_path}")
        
        # Build function context
        function_context = self.context_builder.build_function_context(file_path, function_name)
        if not function_context:
            raise ValueError(f"Function {function_name} not found in {file_path}")
        
        # Extract function signature from file
        function_signature = self._extract_function_signature(file_path, function_name)
        
        # Format context for LLM
        context_str = self.context_builder.format_context_for_llm(function_context)
        
        # Generate using LLM
        response = await self.llm_integration.generate_documentation(
            doc_type="docstring_function",
            context=context_str,
            code=function_signature,
            style=style
        )
        
        # Post-process and validate
        docstring = self._post_process_docstring(response.content, style, "function")
        
        # Quality assessment
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_docstring_quality(
                docstring, function_context
            )
            logger.info(f"Generated docstring quality: {quality_score:.2f}")
            
            # Regenerate if quality is too low
            if quality_score < 0.7:
                logger.info("Quality too low, regenerating with enhanced prompt...")
                enhanced_context = self._enhance_context_for_retry(context_str, function_context)
                response = await self.llm_integration.generate_documentation(
                    doc_type="docstring_function",
                    context=enhanced_context,
                    code=function_signature,
                    style=style,
                    temperature=0.05
                )
                docstring = self._post_process_docstring(response.content, style, "function")
        
        return docstring
    
    async def generate_class_docstring(self, 
                                     file_path: str,
                                     class_name: str,
                                     style: str = "google") -> str:
        """
        Generate a docstring for a specific class.
        
        Args:
            file_path: Path to the file containing the class
            class_name: Name of the class
            style: Docstring style ("google", "numpy", "sphinx")
            
        Returns:
            str: Generated docstring
        """
        logger.info(f"Generating {style} docstring for class {class_name} in {file_path}")
        
        # Build class context
        class_context = self.context_builder.build_class_context(file_path, class_name)
        if not class_context:
            raise ValueError(f"Class {class_name} not found in {file_path}")
        
        # Extract class signature from file
        class_signature = self._extract_class_signature(file_path, class_name)
        
        # Format context for LLM
        context_str = self.context_builder.format_context_for_llm(class_context)
        
        # Generate using LLM
        response = await self.llm_integration.generate_documentation(
            doc_type="docstring_class",
            context=context_str,
            code=class_signature,
            style=style
        )
        
        # Post-process and validate
        docstring = self._post_process_docstring(response.content, style, "class")
        
        # Quality assessment
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_docstring_quality(
                docstring, class_context
            )
            logger.info(f"Generated docstring quality: {quality_score:.2f}")
            
            # Regenerate if quality is too low
            if quality_score < 0.7:
                logger.info("Quality too low, regenerating with enhanced prompt...")
                enhanced_context = self._enhance_context_for_retry(context_str, class_context)
                response = await self.llm_integration.generate_documentation(
                    doc_type="docstring_class",
                    context=enhanced_context,
                    code=class_signature,
                    style=style,
                    temperature=0.05
                )
                docstring = self._post_process_docstring(response.content, style, "class")
        
        return docstring
    
    async def generate_module_docstrings(self, 
                                       file_path: str,
                                       style: str = "google",
                                       include_private: bool = False) -> Dict[str, str]:
        """
        Generate docstrings for all functions and classes in a module.
        
        Args:
            file_path: Path to the Python module
            style: Docstring style
            include_private: Include private methods/classes
            
        Returns:
            Dict[str, str]: Generated docstrings mapped to function/class names
        """
        logger.info(f"Generating {style} docstrings for entire module: {file_path}")
        
        # Build module context
        module_context = self.context_builder.build_module_context(file_path)
        
        docstrings = {}
        
        # Generate function docstrings
        for func_context in module_context.functions:
            if include_private or not func_context.name.startswith('_'):
                try:
                    docstring = await self.generate_function_docstring(
                        file_path, func_context.name, style
                    )
                    docstrings[func_context.name] = docstring
                except Exception as e:
                    logger.error(f"Failed to generate docstring for {func_context.name}: {e}")
        
        # Generate class docstrings
        for class_context in module_context.classes:
            if include_private or not class_context.name.startswith('_'):
                try:
                    docstring = await self.generate_class_docstring(
                        file_path, class_context.name, style
                    )
                    docstrings[class_context.name] = docstring
                except Exception as e:
                    logger.error(f"Failed to generate docstring for {class_context.name}: {e}")
        
        logger.info(f"Generated {len(docstrings)} docstrings for {file_path}")
        return docstrings
    
    def apply_docstrings_to_file(self, 
                               file_path: str,
                               docstrings: Dict[str, str],
                               backup: bool = True) -> str:
        """
        Apply generated docstrings to a Python file.
        
        Args:
            file_path: Path to the Python file
            docstrings: Generated docstrings mapped to function/class names
            backup: Create backup of original file
            
        Returns:
            str: Updated file content
        """
        logger.info(f"Applying {len(docstrings)} docstrings to {file_path}")
        
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created backup at {backup_path}")
        
        # Parse AST to find function/class locations
        tree = ast.parse(content)
        
        # Apply docstrings in reverse order to maintain line numbers
        lines = content.splitlines()
        
        for node in reversed(list(ast.walk(tree))):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if node.name in docstrings:
                    # Find insertion point
                    insertion_line = node.lineno  # 1-based
                    
                    # Check if docstring already exists
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        # Replace existing docstring
                        end_line = node.body[0].end_lineno
                        lines[insertion_line-1:end_line] = [self._format_docstring_for_insertion(
                            docstrings[node.name], self._get_indentation(lines[insertion_line-1])
                        )]
                    else:
                        # Insert new docstring
                        indentation = self._get_indentation(lines[insertion_line])
                        formatted_docstring = self._format_docstring_for_insertion(
                            docstrings[node.name], indentation + "    "
                        )
                        lines.insert(insertion_line, formatted_docstring)
        
        updated_content = '\n'.join(lines)
        
        # Write updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        logger.info(f"Applied docstrings to {file_path}")
        return updated_content
    
    def _extract_function_signature(self, file_path: str, function_name: str) -> str:
        """Extract function signature from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                # Reconstruct signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)
                
                # Add defaults
                defaults = node.args.defaults
                if defaults:
                    for i, default in enumerate(defaults):
                        arg_index = len(args) - len(defaults) + i
                        args[arg_index] += f" = {ast.unparse(default)}"
                
                # Return type
                returns = ""
                if node.returns:
                    returns = f" -> {ast.unparse(node.returns)}"
                
                async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                return f"{async_prefix}def {function_name}({', '.join(args)}){returns}:"
        
        return f"def {function_name}():"
    
    def _extract_class_signature(self, file_path: str, class_name: str) -> str:
        """Extract class signature from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                bases = [ast.unparse(base) for base in node.bases]
                base_str = f"({', '.join(bases)})" if bases else ""
                return f"class {class_name}{base_str}:"
        
        return f"class {class_name}:"
    
    def _post_process_docstring(self, raw_docstring: str, style: str, doc_type: str) -> str:
        """Post-process generated docstring for consistency and formatting."""
        # Remove any surrounding triple quotes if present
        content = raw_docstring.strip()
        if content.startswith('"""') and content.endswith('"""'):
            content = content[3:-3].strip()
        elif content.startswith("'''") and content.endswith("'''"):
            content = content[3:-3].strip()
        
        # Ensure proper formatting based on style
        if style == "google":
            content = self._format_google_style(content)
        elif style == "numpy":
            content = self._format_numpy_style(content)
        elif style == "sphinx":
            content = self._format_sphinx_style(content)
        
        # Add triple quotes
        return f'"""{content}"""'
    
    def _format_google_style(self, content: str) -> str:
        """Format docstring according to Google style."""
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Ensure section headers are properly formatted
            if re.match(r'^\s*(Args?|Arguments|Parameters|Returns?|Yields?|Raises?|Note|Example|Examples)\s*:?\s*$', line, re.IGNORECASE):
                section_name = line.strip().rstrip(':')
                formatted_lines.append(f"{section_name}:")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_numpy_style(self, content: str) -> str:
        """Format docstring according to NumPy style."""
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Ensure section headers are properly formatted with underlines
            if re.match(r'^\s*(Parameters|Returns?|Yields?|Raises?|Notes?|Examples?)\s*:?\s*$', line, re.IGNORECASE):
                section_name = line.strip().rstrip(':')
                formatted_lines.append(section_name)
                formatted_lines.append('-' * len(section_name))
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_sphinx_style(self, content: str) -> str:
        """Format docstring according to Sphinx style."""
        # Sphinx style formatting would go here
        # For now, return as-is
        return content
    
    def _enhance_context_for_retry(self, original_context: str, 
                                 func_or_class_context) -> str:
        """Enhance context for retry generation."""
        enhancement = """
QUALITY IMPROVEMENT REQUEST:
Please generate a more comprehensive and detailed docstring. Focus on:
1. Clear, detailed description of functionality
2. Complete parameter documentation with types
3. Detailed return value description
4. All possible exceptions
5. Practical usage examples
6. Important notes about behavior, performance, or security

"""
        return enhancement + original_context
    
    def _get_indentation(self, line: str) -> str:
        """Get the indentation of a line."""
        return line[:len(line) - len(line.lstrip())]
    
    def _format_docstring_for_insertion(self, docstring: str, indentation: str) -> str:
        """Format docstring for insertion into code with proper indentation."""
        lines = docstring.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            if i == 0:
                # First line
                formatted_lines.append(f"{indentation}{line}")
            else:
                # Subsequent lines - maintain relative indentation
                if line.strip():
                    formatted_lines.append(f"{indentation}{line}")
                else:
                    formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def validate_docstring_style(self, docstring: str, style: str) -> Tuple[bool, List[str]]:
        """
        Validate if a docstring conforms to the specified style.
        
        Args:
            docstring: The docstring to validate
            style: The expected style ("google", "numpy", "sphinx")
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if style == "google":
            # Check for proper Args: section
            if "args:" not in docstring.lower() and "arguments:" not in docstring.lower():
                if "(" in docstring:  # Likely has parameters
                    issues.append("Missing 'Args:' section")
            
            # Check for proper Returns: section
            if "return" in docstring.lower() and "returns:" not in docstring.lower():
                issues.append("Missing 'Returns:' section")
        
        elif style == "numpy":
            # Check for proper Parameters section with underline
            if not re.search(r'Parameters\s*\n\s*-+', docstring):
                if "(" in docstring:  # Likely has parameters
                    issues.append("Missing 'Parameters' section with underline")
        
        elif style == "sphinx":
            # Check for :param: directives
            if "(" in docstring and not re.search(r':param\s+\w+:', docstring):
                issues.append("Missing ':param:' directives")
        
        return len(issues) == 0, issues