"""
Test script for modularized template system.

This script tests the modularized template system to ensure it works correctly.
"""

from .enums import ProjectType, TemplateStyle
from .models import ReadmeContext
from .template_manager import TemplateManager


def test_modularized_templates():
    """Test the modularized template system."""
    print("Testing modularized template system...")
    
    # Initialize template manager
    manager = TemplateManager()
    
    # Test available project types
    project_types = manager.get_available_project_types()
    print(f"Available project types: {[pt.value for pt in project_types]}")
    
    # Test available templates
    templates = manager.list_templates()
    print(f"Available templates: {templates}")
    
    # Create test context
    context = ReadmeContext(
        project_name="TestProject",
        project_type=ProjectType.GENERIC,
        description="A test project for testing the modularized template system",
        author="Agent E",
        license_type="MIT",
        version="1.0.0",
        features=["Feature 1", "Feature 2", "Feature 3"]
    )
    
    # Generate README
    readme = manager.generate_readme(context, TemplateStyle.COMPREHENSIVE)
    
    print("\n" + "="*50)
    print("Generated README:")
    print("="*50)
    print(readme[:500] + "..." if len(readme) > 500 else readme)
    print("="*50)
    
    # Test minimal style
    readme_minimal = manager.generate_readme(context, TemplateStyle.MINIMAL)
    print("\nMinimal README (first 200 chars):")
    print(readme_minimal[:200] + "...")
    
    print("\n✅ Modularization test completed successfully!")
    return True


if __name__ == "__main__":
    test_modularized_templates()