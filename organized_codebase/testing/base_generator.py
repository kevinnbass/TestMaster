"""
Base Template Generator

Abstract base class for all template generators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from ..models import Template, TemplateMetadata


class BaseTemplateGenerator(ABC):
    """Abstract base class for template generators."""
    
    def __init__(self):
        """Initialize the generator with empty templates dictionary."""
        self.templates: Dict[str, Template] = {}
        self.initialize_templates()
    
    @abstractmethod
    def initialize_templates(self):
        """Initialize templates for this generator. Must be implemented by subclasses."""
        pass
    
    def get_template(self, name: str) -> Optional[Template]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template if found, None otherwise
        """
        return self.templates.get(name)
    
    def list_templates(self) -> list:
        """
        List available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def add_template(self, name: str, template: Template):
        """
        Add a template to the generator.
        
        Args:
            name: Template name
            template: Template object
        """
        self.templates[name] = template
    
    @staticmethod
    def create_metadata(
        name: str,
        description: str,
        author: str = "TestMaster",
        version: str = "1.0.0",
        tags: list = None,
        required_variables: list = None,
        optional_variables: list = None,
        target_audience: str = "all"
    ) -> TemplateMetadata:
        """
        Helper method to create template metadata.
        
        Args:
            name: Template name
            description: Template description
            author: Template author
            version: Template version
            tags: Template tags
            required_variables: Required template variables
            optional_variables: Optional template variables
            target_audience: Target audience
            
        Returns:
            TemplateMetadata object
        """
        return TemplateMetadata(
            name=name,
            description=description,
            template_type="README",
            format="MARKDOWN",
            author=author,
            version=version,
            tags=tags or [],
            required_variables=required_variables or [],
            optional_variables=optional_variables or [],
            target_audience=target_audience
        )