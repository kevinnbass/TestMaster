"""
Template Manager

Main interface for the template system. Coordinates between generators,
processors, and provides a unified API for template operations.
"""

from typing import Dict, List, Optional
from .enums import ProjectType, TemplateStyle
from .models import ReadmeContext, Template
from .template_engine import TemplateProcessor
from .generators.base_generator import BaseTemplateGenerator
from .generators.generic_generator import GenericTemplateGenerator


class TemplateManager:
    """
    Main template manager that coordinates all template operations.
    """
    
    def __init__(self):
        """Initialize the template manager with all generators."""
        self.generators: Dict[ProjectType, BaseTemplateGenerator] = {}
        self.processor = TemplateProcessor()
        self._initialize_generators()
    
    def _initialize_generators(self):
        """Initialize all template generators."""
        # Initialize generic generator for now
        # Other generators will be added as they are created
        self.generators[ProjectType.GENERIC] = GenericTemplateGenerator()
        
        # Placeholder for other generators
        # self.generators[ProjectType.WEB_APPLICATION] = WebApplicationGenerator()
        # self.generators[ProjectType.API] = ApiGenerator()
        # self.generators[ProjectType.LIBRARY] = LibraryGenerator()
        # self.generators[ProjectType.CLI_TOOL] = CliToolGenerator()
        # self.generators[ProjectType.DATA_SCIENCE] = DataScienceGenerator()
        # self.generators[ProjectType.MACHINE_LEARNING] = MachineLearningGenerator()
    
    def get_template(
        self,
        project_type: ProjectType,
        style: TemplateStyle = TemplateStyle.COMPREHENSIVE
    ) -> Optional[Template]:
        """
        Get a template for the specified project type and style.
        
        Args:
            project_type: Type of project
            style: Style of template
            
        Returns:
            Template if found, None otherwise
        """
        generator = self.generators.get(project_type)
        if not generator:
            # Fallback to generic if specific type not found
            generator = self.generators.get(ProjectType.GENERIC)
        
        if generator:
            template_name = f"{project_type.value}_{style.value}"
            template = generator.get_template(template_name)
            
            # If specific style not found, try without style suffix
            if not template:
                template = generator.get_template(project_type.value)
            
            # Final fallback to generic comprehensive
            if not template and project_type != ProjectType.GENERIC:
                generator = self.generators.get(ProjectType.GENERIC)
                template = generator.get_template(f"generic_{style.value}")
            
            return template
        
        return None
    
    def list_templates(self, project_type: Optional[ProjectType] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            project_type: Optional project type filter
            
        Returns:
            List of template names
        """
        templates = []
        
        if project_type:
            generator = self.generators.get(project_type)
            if generator:
                templates.extend(generator.list_templates())
        else:
            for generator in self.generators.values():
                templates.extend(generator.list_templates())
        
        return templates
    
    def generate_readme(
        self,
        context: ReadmeContext,
        style: TemplateStyle = TemplateStyle.COMPREHENSIVE
    ) -> str:
        """
        Generate a README using the specified context.
        
        Args:
            context: Context information for README generation
            style: Style of README to generate
            
        Returns:
            Generated README content
        """
        template = self.get_template(context.project_type, style)
        
        if not template:
            # Return basic fallback if no template found
            return self._generate_fallback_readme(context)
        
        # Prepare variables
        variables = self.processor.prepare_variables(context)
        
        # Process template
        content = self.processor.process_template(template.content, variables)
        
        return content
    
    def _generate_fallback_readme(self, context: ReadmeContext) -> str:
        """
        Generate a basic fallback README when no template is available.
        
        Args:
            context: README context
            
        Returns:
            Basic README content
        """
        return f"""# {context.project_name}

{context.description}

## Installation

```bash
pip install {context.project_name}
```

## Usage

```python
import {context.project_name}

# Add your usage example here
```

## Author

{context.author}

## License

{context.license_type or 'MIT License'}

---

*No template available for project type: {context.project_type.value}*
"""
    
    def register_generator(self, project_type: ProjectType, generator: BaseTemplateGenerator):
        """
        Register a new template generator.
        
        Args:
            project_type: Project type for this generator
            generator: Generator instance
        """
        self.generators[project_type] = generator
    
    def get_available_project_types(self) -> List[ProjectType]:
        """
        Get list of project types with available generators.
        
        Returns:
            List of ProjectType enums
        """
        return list(self.generators.keys())
    
    def get_available_styles(self) -> List[TemplateStyle]:
        """
        Get list of available template styles.
        
        Returns:
            List of TemplateStyle enums
        """
        return list(TemplateStyle)