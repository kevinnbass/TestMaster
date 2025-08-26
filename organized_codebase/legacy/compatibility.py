"""
Backward Compatibility Layer

Provides compatibility with the original readme_templates.py interface
to ensure existing code continues to work.
"""

from .template_manager import TemplateManager
from .models import ReadmeContext
from .enums import ProjectType, TemplateStyle


class ReadmeTemplateManager:
    """
    Backward compatible interface for ReadmeTemplateManager.
    
    This class maintains the same interface as the original ReadmeTemplateManager
    while using the new modular system underneath.
    """
    
    def __init__(self):
        """Initialize with the new modular template manager."""
        self._manager = TemplateManager()
    
    def get_template(self, project_type: ProjectType, style: str = "comprehensive"):
        """
        Get a README template for the specified project type and style.
        
        Args:
            project_type: Type of project
            style: Style of template
            
        Returns:
            Template if found, None otherwise
        """
        try:
            style_enum = TemplateStyle(style)
        except ValueError:
            style_enum = TemplateStyle.COMPREHENSIVE
        
        return self._manager.get_template(project_type, style_enum)
    
    def list_templates(self, project_type=None):
        """
        List available README templates.
        
        Args:
            project_type: Optional project type filter
            
        Returns:
            List of template names
        """
        return self._manager.list_templates(project_type)
    
    def generate_readme(self, context: ReadmeContext, style: str = "comprehensive"):
        """
        Generate a README using the specified context.
        
        Args:
            context: Context information for README generation
            style: Style of README to generate
            
        Returns:
            Generated README content
        """
        try:
            style_enum = TemplateStyle(style)
        except ValueError:
            style_enum = TemplateStyle.COMPREHENSIVE
        
        return self._manager.generate_readme(context, style_enum)


# For complete backward compatibility, also provide the original class
# This allows existing imports to continue working
class ReadmeTemplateManagerLegacy(ReadmeTemplateManager):
    """Complete legacy compatibility class."""
    
    def __init__(self):
        super().__init__()
        # Initialize any legacy attributes that might be expected
        self.templates = {}
        self._populate_legacy_templates()
    
    def _populate_legacy_templates(self):
        """Populate legacy templates dictionary for compatibility."""
        # Map new templates to old template names
        for project_type in self._manager.get_available_project_types():
            templates = self._manager.list_templates(project_type)
            for template_name in templates:
                template = self._manager.get_template(project_type, TemplateStyle.COMPREHENSIVE)
                if template:
                    self.templates[template_name] = template