"""
Docstring Templates System

This module provides comprehensive docstring templates for various documentation styles
including Google, NumPy, Sphinx, and custom formats.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat


class DocstringStyle(Enum):
    """Supported docstring styles."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    EPYTEXT = "epytext"
    CUSTOM = "custom"


@dataclass
class DocstringContext:
    """Context information for docstring generation."""
    function_name: str
    parameters: List[Dict[str, str]]
    return_type: Optional[str] = None
    return_description: Optional[str] = None
    raises: List[Dict[str, str]] = None
    examples: List[str] = None
    notes: List[str] = None
    references: List[str] = None
    class_name: Optional[str] = None
    module_name: Optional[str] = None
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    complexity: str = "medium"  # low, medium, high
    
    def __post_init__(self):
        if self.raises is None:
            self.raises = []
        if self.examples is None:
            self.examples = []
        if self.notes is None:
            self.notes = []
        if self.references is None:
            self.references = []


class DocstringTemplateManager:
    """
    Manages docstring templates for different styles and contexts.
    """
    
    def __init__(self):
        """Initialize the docstring template manager."""
        self.templates: Dict[str, Template] = {}
        self._initialize_builtin_templates()
    
    def _initialize_builtin_templates(self):
        """Initialize built-in docstring templates."""
        self._create_google_templates()
        self._create_numpy_templates()
        self._create_sphinx_templates()
        self._create_epytext_templates()
    
    def _create_google_templates(self):
        """Create Google-style docstring templates."""
        
        # Function template
        function_template = Template(
            metadata=TemplateMetadata(
                name="google_function",
                description="Google-style function docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["google", "function"],
                required_variables=["function_name", "brief_description"],
                optional_variables=["parameters", "returns", "raises", "examples", "notes"],
                target_audience="all"
            ),
            content='''"""{{brief_description}}

{{#detailed_description}}
{{detailed_description}}
{{/detailed_description}}

{{#parameters}}
Args:
{{#parameters}}
    {{name}} ({{type}}): {{description}}
{{/parameters}}
{{/parameters}}

{{#returns}}
Returns:
    {{return_type}}: {{return_description}}
{{/returns}}

{{#raises}}
Raises:
{{#raises}}
    {{exception}}: {{description}}
{{/raises}}
{{/raises}}

{{#examples}}
Examples:
{{#examples}}
    {{example}}
{{/examples}}
{{/examples}}

{{#notes}}
Note:
{{#notes}}
    {{note}}
{{/notes}}
{{/notes}}
"""''',
            examples=[
                '''"""Calculate the sum of two numbers.

Args:
    a (int): The first number
    b (int): The second number

Returns:
    int: The sum of a and b

Examples:
    >>> add_numbers(2, 3)
    5
"""'''
            ]
        )
        
        self.templates["google_function"] = function_template
        
        # Class template
        class_template = Template(
            metadata=TemplateMetadata(
                name="google_class",
                description="Google-style class docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["google", "class"],
                required_variables=["class_name", "brief_description"],
                optional_variables=["attributes", "methods", "examples", "notes"],
                target_audience="all"
            ),
            content='''"""{{brief_description}}

{{#detailed_description}}
{{detailed_description}}
{{/detailed_description}}

{{#attributes}}
Attributes:
{{#attributes}}
    {{name}} ({{type}}): {{description}}
{{/attributes}}
{{/attributes}}

{{#examples}}
Examples:
{{#examples}}
    {{example}}
{{/examples}}
{{/examples}}

{{#notes}}
Note:
{{#notes}}
    {{note}}
{{/notes}}
{{/notes}}
"""''',
            examples=[
                '''"""A simple calculator class.

This class provides basic arithmetic operations.

Attributes:
    result (float): The current result of calculations

Examples:
    >>> calc = Calculator()
    >>> calc.add(5, 3)
    8.0
"""'''
            ]
        )
        
        self.templates["google_class"] = class_template
    
    def _create_numpy_templates(self):
        """Create NumPy-style docstring templates."""
        
        # Function template
        function_template = Template(
            metadata=TemplateMetadata(
                name="numpy_function",
                description="NumPy-style function docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["numpy", "function"],
                required_variables=["function_name", "brief_description"],
                optional_variables=["parameters", "returns", "raises", "examples", "notes"],
                target_audience="all"
            ),
            content='''"""{{brief_description}}

{{#detailed_description}}
{{detailed_description}}
{{/detailed_description}}

{{#parameters}}
Parameters
----------
{{#parameters}}
{{name}} : {{type}}
    {{description}}
{{/parameters}}
{{/parameters}}

{{#returns}}
Returns
-------
{{return_type}}
    {{return_description}}
{{/returns}}

{{#raises}}
Raises
------
{{#raises}}
{{exception}}
    {{description}}
{{/raises}}
{{/raises}}

{{#examples}}
Examples
--------
{{#examples}}
{{example}}
{{/examples}}
{{/examples}}

{{#notes}}
Notes
-----
{{#notes}}
{{note}}
{{/notes}}
{{/notes}}
"""''',
            examples=[
                '''"""Calculate the sum of two numbers.

Parameters
----------
a : int
    The first number
b : int
    The second number

Returns
-------
int
    The sum of a and b

Examples
--------
>>> add_numbers(2, 3)
5
"""'''
            ]
        )
        
        self.templates["numpy_function"] = function_template
        
        # Class template
        class_template = Template(
            metadata=TemplateMetadata(
                name="numpy_class",
                description="NumPy-style class docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["numpy", "class"],
                required_variables=["class_name", "brief_description"],
                optional_variables=["attributes", "methods", "examples", "notes"],
                target_audience="all"
            ),
            content='''"""{{brief_description}}

{{#detailed_description}}
{{detailed_description}}
{{/detailed_description}}

{{#attributes}}
Attributes
----------
{{#attributes}}
{{name}} : {{type}}
    {{description}}
{{/attributes}}
{{/attributes}}

{{#examples}}
Examples
--------
{{#examples}}
{{example}}
{{/examples}}
{{/examples}}

{{#notes}}
Notes
-----
{{#notes}}
{{note}}
{{/notes}}
{{/notes}}
"""''',
            examples=[
                '''"""A simple calculator class.

This class provides basic arithmetic operations.

Attributes
----------
result : float
    The current result of calculations

Examples
--------
>>> calc = Calculator()
>>> calc.add(5, 3)
8.0
"""'''
            ]
        )
        
        self.templates["numpy_class"] = class_template
    
    def _create_sphinx_templates(self):
        """Create Sphinx-style docstring templates."""
        
        # Function template
        function_template = Template(
            metadata=TemplateMetadata(
                name="sphinx_function",
                description="Sphinx-style function docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["sphinx", "function"],
                required_variables=["function_name", "brief_description"],
                optional_variables=["parameters", "returns", "raises", "examples", "notes"],
                target_audience="all"
            ),
            content='''"""{{brief_description}}

{{#detailed_description}}
{{detailed_description}}
{{/detailed_description}}

{{#parameters}}
{{#parameters}}
:param {{name}}: {{description}}
:type {{name}}: {{type}}
{{/parameters}}
{{/parameters}}

{{#returns}}
:returns: {{return_description}}
:rtype: {{return_type}}
{{/returns}}

{{#raises}}
{{#raises}}
:raises {{exception}}: {{description}}
{{/raises}}
{{/raises}}

{{#examples}}
.. code-block:: python

{{#examples}}
   {{example}}
{{/examples}}
{{/examples}}

{{#notes}}
.. note::
{{#notes}}
   {{note}}
{{/notes}}
{{/notes}}
"""''',
            examples=[
                '''"""Calculate the sum of two numbers.

:param a: The first number
:type a: int
:param b: The second number  
:type b: int
:returns: The sum of a and b
:rtype: int

.. code-block:: python

   >>> add_numbers(2, 3)
   5
"""'''
            ]
        )
        
        self.templates["sphinx_function"] = function_template
        
        # Class template
        class_template = Template(
            metadata=TemplateMetadata(
                name="sphinx_class",
                description="Sphinx-style class docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["sphinx", "class"],
                required_variables=["class_name", "brief_description"],
                optional_variables=["attributes", "methods", "examples", "notes"],
                target_audience="all"
            ),
            content='''"""{{brief_description}}

{{#detailed_description}}
{{detailed_description}}
{{/detailed_description}}

{{#attributes}}
{{#attributes}}
.. attribute:: {{name}}

   {{description}}

   :type: {{type}}
{{/attributes}}
{{/attributes}}

{{#examples}}
.. code-block:: python

{{#examples}}
   {{example}}
{{/examples}}
{{/examples}}

{{#notes}}
.. note::
{{#notes}}
   {{note}}
{{/notes}}
{{/notes}}
"""''',
            examples=[
                '''"""A simple calculator class.

This class provides basic arithmetic operations.

.. attribute:: result

   The current result of calculations

   :type: float

.. code-block:: python

   >>> calc = Calculator()
   >>> calc.add(5, 3)
   8.0
"""'''
            ]
        )
        
        self.templates["sphinx_class"] = class_template
    
    def _create_epytext_templates(self):
        """Create Epytext-style docstring templates."""
        
        # Function template
        function_template = Template(
            metadata=TemplateMetadata(
                name="epytext_function",
                description="Epytext-style function docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["epytext", "function"],
                required_variables=["function_name", "brief_description"],
                optional_variables=["parameters", "returns", "raises", "examples", "notes"],
                target_audience="all"
            ),
            content='''"""{{brief_description}}

{{#detailed_description}}
{{detailed_description}}
{{/detailed_description}}

{{#parameters}}
{{#parameters}}
@param {{name}}: {{description}}
@type {{name}}: {{type}}
{{/parameters}}
{{/parameters}}

{{#returns}}
@return: {{return_description}}
@rtype: {{return_type}}
{{/returns}}

{{#raises}}
{{#raises}}
@raise {{exception}}: {{description}}
{{/raises}}
{{/raises}}

{{#examples}}
Examples:
{{#examples}}
{{example}}
{{/examples}}
{{/examples}}

{{#notes}}
@note: 
{{#notes}}
{{note}}
{{/notes}}
{{/notes}}
"""''',
            examples=[
                '''"""Calculate the sum of two numbers.

@param a: The first number
@type a: int
@param b: The second number
@type b: int
@return: The sum of a and b
@rtype: int
"""'''
            ]
        )
        
        self.templates["epytext_function"] = function_template
    
    def get_template(self, style: DocstringStyle, element_type: str) -> Optional[Template]:
        """
        Get a docstring template for the specified style and element type.
        
        Args:
            style: The docstring style (Google, NumPy, etc.)
            element_type: The type of element (function, class, method)
            
        Returns:
            Template if found, None otherwise
        """
        template_key = f"{style.value}_{element_type}"
        return self.templates.get(template_key)
    
    def list_templates(self, style: Optional[DocstringStyle] = None) -> List[str]:
        """
        List available docstring templates.
        
        Args:
            style: Optional style filter
            
        Returns:
            List of template names
        """
        if style is None:
            return list(self.templates.keys())
        
        return [key for key in self.templates.keys() if key.startswith(style.value)]
    
    def generate_docstring(
        self,
        context: DocstringContext,
        style: DocstringStyle = DocstringStyle.GOOGLE,
        element_type: str = "function"
    ) -> str:
        """
        Generate a docstring using the specified style and context.
        
        Args:
            context: Context information for the docstring
            style: Docstring style to use
            element_type: Type of element (function, class, method)
            
        Returns:
            Generated docstring
        """
        template = self.get_template(style, element_type)
        if not template:
            # Fallback to Google style
            template = self.get_template(DocstringStyle.GOOGLE, element_type)
            if not template:
                return f'"""{context.function_name or "Function"} - No template available"""'
        
        # Prepare variables for template rendering
        variables = self._prepare_template_variables(context, style)
        
        # Simple template processing (would use proper template engine in production)
        content = template.content
        
        # Replace simple variables
        for key, value in variables.items():
            if isinstance(value, str):
                content = content.replace(f"{{{{{key}}}}}", value)
        
        # Handle list variables (simplified processing)
        content = self._process_list_variables(content, variables, style)
        
        return content
    
    def _prepare_template_variables(
        self,
        context: DocstringContext,
        style: DocstringStyle
    ) -> Dict[str, Any]:
        """Prepare variables for template rendering."""
        variables = {
            "function_name": context.function_name,
            "brief_description": f"Brief description for {context.function_name}",
            "detailed_description": "",
        }
        
        # Parameters
        if context.parameters:
            variables["parameters"] = context.parameters
        
        # Return information
        if context.return_type and context.return_description:
            variables["returns"] = True
            variables["return_type"] = context.return_type
            variables["return_description"] = context.return_description
        
        # Exceptions
        if context.raises:
            variables["raises"] = context.raises
        
        # Examples
        if context.examples:
            variables["examples"] = context.examples
        
        # Notes
        if context.notes:
            variables["notes"] = context.notes
        
        return variables
    
    def _process_list_variables(
        self,
        content: str,
        variables: Dict[str, Any],
        style: DocstringStyle
    ) -> str:
        """Process list variables in templates (simplified implementation)."""
        # This is a simplified implementation
        # In production, would use proper template engine (Jinja2, Mustache, etc.)
        
        # Remove template conditionals that aren't met
        import re
        
        # Remove empty sections
        patterns = [
            r'\{\{#parameters\}\}.*?\{\{/parameters\}\}',
            r'\{\{#returns\}\}.*?\{\{/returns\}\}',
            r'\{\{#raises\}\}.*?\{\{/raises\}\}',
            r'\{\{#examples\}\}.*?\{\{/examples\}\}',
            r'\{\{#notes\}\}.*?\{\{/notes\}\}',
        ]
        
        for pattern in patterns:
            section_name = pattern.split('}')[0].split('#')[1]
            if section_name not in variables:
                content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        # Process parameters
        if 'parameters' in variables:
            param_section = ""
            if style == DocstringStyle.GOOGLE:
                param_section = "Args:\n"
                for param in variables['parameters']:
                    param_section += f"    {param['name']} ({param['type']}): {param['description']}\n"
            elif style == DocstringStyle.NUMPY:
                param_section = "Parameters\n----------\n"
                for param in variables['parameters']:
                    param_section += f"{param['name']} : {param['type']}\n    {param['description']}\n"
            elif style == DocstringStyle.SPHINX:
                for param in variables['parameters']:
                    param_section += f":param {param['name']}: {param['description']}\n"
                    param_section += f":type {param['name']}: {param['type']}\n"
            
            content = re.sub(r'\{\{#parameters\}\}.*?\{\{/parameters\}\}', param_section, content, flags=re.DOTALL)
        
        # Clean up remaining template syntax
        content = re.sub(r'\{\{[^}]+\}\}', '', content)
        
        # Clean up extra whitespace
        lines = content.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if line.strip():
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True
        
        return '\n'.join(cleaned_lines).strip()
    
    def create_custom_template(
        self,
        name: str,
        template_content: str,
        element_type: str,
        description: str = "",
        required_variables: Optional[List[str]] = None,
        optional_variables: Optional[List[str]] = None
    ) -> str:
        """
        Create a custom docstring template.
        
        Args:
            name: Name of the template
            template_content: Template content with placeholders
            element_type: Type of element (function, class, method)
            description: Template description
            required_variables: List of required variables
            optional_variables: List of optional variables
            
        Returns:
            Template key
        """
        if required_variables is None:
            required_variables = []
        if optional_variables is None:
            optional_variables = []
        
        template_key = f"custom_{name}_{element_type}"
        
        template = Template(
            metadata=TemplateMetadata(
                name=name,
                description=description or f"Custom {element_type} docstring template",
                template_type=TemplateType.DOCSTRING,
                format=TemplateFormat.SIMPLE,
                author="User",
                version="1.0.0",
                tags=["custom", element_type],
                required_variables=required_variables,
                optional_variables=optional_variables,
                target_audience="all"
            ),
            content=template_content
        )
        
        self.templates[template_key] = template
        return template_key
    
    def validate_docstring_style(self, docstring: str, style: DocstringStyle) -> Dict[str, Any]:
        """
        Validate if a docstring follows the specified style conventions.
        
        Args:
            docstring: Docstring to validate
            style: Expected docstring style
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'style': style.value,
            'issues': [],
            'suggestions': []
        }
        
        if style == DocstringStyle.GOOGLE:
            results.update(self._validate_google_style(docstring))
        elif style == DocstringStyle.NUMPY:
            results.update(self._validate_numpy_style(docstring))
        elif style == DocstringStyle.SPHINX:
            results.update(self._validate_sphinx_style(docstring))
        
        return results
    
    def _validate_google_style(self, docstring: str) -> Dict[str, Any]:
        """Validate Google-style docstring."""
        issues = []
        suggestions = []
        
        # Check for proper sections
        if 'Args:' not in docstring and 'Arguments:' not in docstring:
            if any(param_pattern in docstring.lower() for param_pattern in ['parameter', 'param', 'arg']):
                suggestions.append("Consider using 'Args:' section for parameters")
        
        if 'Returns:' not in docstring and 'Return:' not in docstring:
            if 'return' in docstring.lower():
                suggestions.append("Consider using 'Returns:' section for return value")
        
        # Check indentation (simplified)
        lines = docstring.split('\n')
        for i, line in enumerate(lines[1:], 2):  # Skip first line
            if line and not line.startswith('    ') and not line.startswith('"""'):
                if any(section in line for section in ['Args:', 'Returns:', 'Raises:', 'Examples:']):
                    continue
                issues.append(f"Line {i}: Improper indentation in docstring")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'valid': len(issues) == 0
        }
    
    def _validate_numpy_style(self, docstring: str) -> Dict[str, Any]:
        """Validate NumPy-style docstring."""
        issues = []
        suggestions = []
        
        # Check for proper section headers
        numpy_sections = ['Parameters', 'Returns', 'Raises', 'Examples', 'Notes']
        for section in numpy_sections:
            if section in docstring:
                # Check for underline
                lines = docstring.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() == section and i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if not next_line.strip().startswith('---'):
                            issues.append(f"Section '{section}' missing proper underline (---)")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'valid': len(issues) == 0
        }
    
    def _validate_sphinx_style(self, docstring: str) -> Dict[str, Any]:
        """Validate Sphinx-style docstring."""
        issues = []
        suggestions = []
        
        # Check for proper field format
        if ':param' not in docstring and 'parameter' in docstring.lower():
            suggestions.append("Consider using ':param name: description' format for parameters")
        
        if ':returns:' not in docstring and ':return:' not in docstring and 'return' in docstring.lower():
            suggestions.append("Consider using ':returns: description' format for return value")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'valid': len(issues) == 0
        }