"""
Template Processing Engine

This module handles the core template processing logic including variable substitution,
conditionals, and list iteration.
"""

import re
from typing import Dict, Any
from .models import ReadmeContext


class TemplateProcessor:
    """Handles template processing and variable substitution."""
    
    @staticmethod
    def prepare_variables(context: ReadmeContext) -> Dict[str, Any]:
        """
        Prepare variables for template rendering.
        
        Args:
            context: README generation context
            
        Returns:
            Dictionary of template variables
        """
        variables = {
            "project_name": context.project_name,
            "description": context.description,
            "author": context.author,
            "PROJECT_NAME": context.project_name.upper().replace("-", "_"),
            "command_name": context.project_name.lower().replace("_", "-"),
        }
        
        # Add optional variables
        if context.license_type:
            variables["license_type"] = context.license_type
        
        if context.version:
            variables["version"] = context.version
        
        if context.python_version:
            variables["python_version"] = context.python_version
        
        if context.demo_link:
            variables["demo_link"] = context.demo_link
        
        if context.documentation_link:
            variables["documentation_link"] = context.documentation_link
        
        # Add lists
        if context.features:
            variables["features"] = context.features
        
        if context.dependencies:
            variables["dependencies"] = context.dependencies
        
        if context.tech_stack:
            variables["tech_stack"] = context.tech_stack
        
        if context.badges:
            variables["badges"] = context.badges
        
        return variables
    
    @staticmethod
    def process_template(content: str, variables: Dict[str, Any]) -> str:
        """
        Process template content with variables.
        
        Args:
            content: Template content
            variables: Variables to substitute
            
        Returns:
            Processed content
        """
        # Replace simple variables
        for key, value in variables.items():
            if isinstance(value, str):
                content = content.replace(f"{{{{{key}}}}}", value)
        
        # Process conditional sections and lists
        content = TemplateLogicHandler.process_logic(content, variables)
        
        return content


class TemplateLogicHandler:
    """Handles template logic processing (conditionals, loops)."""
    
    @staticmethod
    def process_logic(content: str, variables: Dict[str, Any]) -> str:
        """
        Process template conditionals and loops.
        
        Args:
            content: Template content
            variables: Template variables
            
        Returns:
            Processed content
        """
        # Process conditional sections
        content = TemplateLogicHandler._process_conditionals(content, variables)
        
        # Process list iterations
        content = TemplateLogicHandler._process_lists(content, variables)
        
        # Clean up remaining template syntax
        content = re.sub(r'\{\{[^}]+\}\}', '', content)
        
        # Clean up extra whitespace
        return TemplateLogicHandler._cleanup_whitespace(content)
    
    @staticmethod
    def _process_conditionals(content: str, variables: Dict[str, Any]) -> str:
        """
        Process conditional sections.
        {{#variable}} ... {{/variable}} - show if variable exists and is truthy
        {{^variable}} ... {{/variable}} - show if variable doesn't exist or is falsy
        """
        conditional_pattern = r'\{\{([#^])(\w+)\}\}(.*?)\{\{/\2\}\}'
        
        def replace_conditional(match):
            operator = match.group(1)
            var_name = match.group(2)
            content_block = match.group(3)
            
            var_value = variables.get(var_name)
            
            if operator == '#':
                return content_block if var_value else ''
            else:  # operator == '^'
                return content_block if not var_value else ''
        
        return re.sub(conditional_pattern, replace_conditional, content, flags=re.DOTALL)
    
    @staticmethod
    def _process_lists(content: str, variables: Dict[str, Any]) -> str:
        """Process list iterations in templates."""
        list_pattern = r'\{\{#(\w+)\}\}(.*?)\{\{/\1\}\}'
        
        def replace_list(match):
            var_name = match.group(1)
            template_block = match.group(2)
            
            var_value = variables.get(var_name)
            
            if isinstance(var_value, list):
                result = []
                for item in var_value:
                    if isinstance(item, dict):
                        block = template_block
                        for key, value in item.items():
                            block = block.replace(f"{{{{{key}}}}}", str(value))
                        result.append(block)
                    else:
                        block = template_block.replace("{{.}}", str(item))
                        result.append(block)
                return ''.join(result)
            return ''
        
        return re.sub(list_pattern, replace_list, content, flags=re.DOTALL)
    
    @staticmethod
    def _cleanup_whitespace(content: str) -> str:
        """Clean up extra whitespace in content."""
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