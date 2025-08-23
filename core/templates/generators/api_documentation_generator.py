#!/usr/bin/env python3
"""
ApiDocumentationGenerator - Modular Template Generator
Migrated from monolithic template files by Agent A Template Migration System

Generated on: 2025-08-23T00:37:36.647111
Category: api_documentation
Templates migrated: 1
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TemplateContext:
    """Context for template generation"""
    project_name: str = ""
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    additional_vars: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_vars is None:
            self.additional_vars = {}


class ApiDocumentationGenerator:
    """
    Modular template generator for api_documentation templates
    
    Extracted from monolithic files and integrated with architecture framework.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.category = "api_documentation"
        self.templates = {
            "_create_api_templates": {
                "content": "# Function template: _create_api_templates...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'function', 'ast_type': 'FunctionDef'}
            }
        }
        
        self.logger.info(f"{self.__class__.__name__} initialized with {len(self.templates)} templates")
    
    def generate(self, template_name: str, context: TemplateContext) -> str:
        """Generate template with given context"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found in {self.category} generator")
        
        template_data = self.templates[template_name]
        template_content = template_data['content']
        
        try:
            # Simple variable substitution
            result = self._substitute_variables(template_content, context)
            
            self.logger.debug(f"Generated {template_name} template ({len(result)} chars)")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate {template_name}: {e}")
            raise
    
    def list_templates(self) -> List[str]:
        """List available templates"""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        if template_name not in self.templates:
            return {"error": f"Template '{template_name}' not found"}
        
        return self.templates[template_name]
    
    def _substitute_variables(self, template_content: str, context: TemplateContext) -> str:
        """Substitute template variables with context values"""
        result = template_content
        
        # Basic substitutions
        substitutions = {
            '{{project_name}}': context.project_name,
            '{{description}}': context.description,
            '{{author}}': context.author,
            '{{version}}': context.version,
            '{{date}}': datetime.now().strftime('%Y-%m-%d'),
            '{{year}}': str(datetime.now().year)
        }
        
        # Add additional variables
        for key, value in context.additional_vars.items():
            substitutions[f'{{{key}}}'] = str(value)
        
        # Perform substitutions
        for placeholder, value in substitutions.items():
            result = result.replace(placeholder, value)
        
        return result


# Factory function
def create_api_documentation_generator() -> ApiDocumentationGenerator:
    """Create api_documentation generator instance"""
    return ApiDocumentationGenerator()


# Export for service registration
__all__ = ['ApiDocumentationGenerator', 'TemplateContext', 'create_api_documentation_generator']
