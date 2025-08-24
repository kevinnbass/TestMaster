#!/usr/bin/env python3
"""
Template Migration System - Agent A Hour 5
Extracts and migrates template systems to modular architecture

Migrates monolithic template files to modular generators using
the integrated architecture framework with proper service registration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import importlib.util
import ast

# Import architecture framework
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.architecture_integration import (
    get_architecture_framework,
    ArchitecturalLayer,
    LifetimeScope
)
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.service_registry import (
    get_service_registry,
    ServiceDefinition,
    ServiceType
)


class TemplateCategory(Enum):
    """Categories of templates"""
    README = "readme"
    API_DOCUMENTATION = "api_documentation"
    PROJECT_STRUCTURE = "project_structure"
    TEST_TEMPLATES = "test_templates"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"


@dataclass
class TemplateDefinition:
    """Definition of a template found in monolithic files"""
    name: str
    category: TemplateCategory
    content: str
    variables: List[str]
    source_file: Path
    line_start: int
    line_end: int
    metadata: Dict[str, Any]


@dataclass
class MigrationResult:
    """Result of template migration"""
    template_name: str
    source_file: Path
    target_file: Path
    success: bool
    error_message: Optional[str] = None
    extracted_variables: List[str] = None
    migration_time: datetime = None
    
    def __post_init__(self):
        if self.migration_time is None:
            self.migration_time = datetime.now()


class TemplateMigrationSystem:
    """
    Template Migration System
    
    Extracts templates from monolithic files and creates modular
    generator services using the architecture framework.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Architecture components
        self.framework = get_architecture_framework()
        self.service_registry = get_service_registry()
        
        # Migration tracking
        self.discovered_templates: Dict[str, TemplateDefinition] = {}
        self.migration_results: List[MigrationResult] = []
        
        # Migration configuration
        self.source_files = [
            Path("TestMaster/readme_templates.py"),
            Path("web/unified_gamma_dashboard.py"),
            Path("predictive_code_intelligence.py")
        ]
        
        # Target directory for modular templates
        self.target_directory = Path("core/templates/generators")
        self.target_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Template Migration System initialized")
    
    def discover_templates(self) -> Dict[str, List[TemplateDefinition]]:
        """Discover templates in monolithic files"""
        discovered = {}
        
        for source_file in self.source_files:
            if not source_file.exists():
                self.logger.warning(f"Source file not found: {source_file}")
                continue
            
            try:
                templates = self._analyze_file_for_templates(source_file)
                if templates:
                    discovered[str(source_file)] = templates
                    self.logger.info(f"Found {len(templates)} templates in {source_file}")
                    
                    # Store in discovered templates registry
                    for template in templates:
                        self.discovered_templates[template.name] = template
                        
            except Exception as e:
                self.logger.error(f"Failed to analyze {source_file}: {e}")
        
        total_templates = sum(len(templates) for templates in discovered.values())
        self.logger.info(f"Discovery complete: {total_templates} templates found in {len(discovered)} files")
        
        return discovered
    
    def _analyze_file_for_templates(self, source_file: Path) -> List[TemplateDefinition]:
        """Analyze a source file for template definitions"""
        templates = []
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find template patterns
            tree = ast.parse(content)
            
            # Look for template-related patterns
            for node in ast.walk(tree):
                template_def = self._extract_template_from_node(node, source_file, content)
                if template_def:
                    templates.append(template_def)
            
            # Also look for string literals that look like templates
            templates.extend(self._find_template_strings(content, source_file))
            
        except Exception as e:
            self.logger.error(f"Error analyzing {source_file}: {e}")
        
        return templates
    
    def _extract_template_from_node(self, node: ast.AST, source_file: Path, content: str) -> Optional[TemplateDefinition]:
        """Extract template definition from AST node"""
        # Look for class definitions that might be templates
        if isinstance(node, ast.ClassDef):
            if any(keyword in node.name.lower() for keyword in ['template', 'generator', 'readme']):
                return TemplateDefinition(
                    name=node.name,
                    category=self._categorize_template(node.name),
                    content=f"# Class template: {node.name}",
                    variables=[],
                    source_file=source_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    metadata={'type': 'class', 'ast_type': 'ClassDef'}
                )
        
        # Look for function definitions that generate templates
        elif isinstance(node, ast.FunctionDef):
            if any(keyword in node.name.lower() for keyword in ['template', 'generate', 'create_readme']):
                return TemplateDefinition(
                    name=node.name,
                    category=self._categorize_template(node.name),
                    content=f"# Function template: {node.name}",
                    variables=self._extract_function_parameters(node),
                    source_file=source_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    metadata={'type': 'function', 'ast_type': 'FunctionDef'}
                )
        
        return None
    
    def _find_template_strings(self, content: str, source_file: Path) -> List[TemplateDefinition]:
        """Find template strings in file content"""
        templates = []
        
        lines = content.split('\n')
        current_template = None
        template_lines = []
        
        for i, line in enumerate(lines):
            # Look for multi-line template strings
            if '"""' in line or "'''" in line:
                if current_template is None:
                    # Start of template
                    if any(keyword in line.lower() for keyword in ['template', 'readme', 'documentation']):
                        current_template = {
                            'name': f"template_{i}",
                            'start_line': i + 1,
                            'lines': [line]
                        }
                else:
                    # End of template
                    template_lines.append(line)
                    template_content = '\n'.join(current_template['lines'] + template_lines)
                    
                    templates.append(TemplateDefinition(
                        name=current_template['name'],
                        category=TemplateCategory.README,  # Default category
                        content=template_content,
                        variables=self._extract_template_variables(template_content),
                        source_file=source_file,
                        line_start=current_template['start_line'],
                        line_end=i + 1,
                        metadata={'type': 'string_literal'}
                    ))
                    
                    current_template = None
                    template_lines = []
            
            elif current_template is not None:
                template_lines.append(line)
        
        return templates
    
    def _categorize_template(self, name: str) -> TemplateCategory:
        """Categorize template based on name"""
        name_lower = name.lower()
        
        if 'readme' in name_lower:
            return TemplateCategory.README
        elif 'api' in name_lower or 'documentation' in name_lower:
            return TemplateCategory.API_DOCUMENTATION
        elif 'test' in name_lower:
            return TemplateCategory.TEST_TEMPLATES
        elif 'config' in name_lower:
            return TemplateCategory.CONFIGURATION
        elif 'deploy' in name_lower:
            return TemplateCategory.DEPLOYMENT
        else:
            return TemplateCategory.PROJECT_STRUCTURE
    
    def _extract_function_parameters(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract function parameters as template variables"""
        parameters = []
        for arg in func_node.args.args:
            if arg.arg != 'self':
                parameters.append(arg.arg)
        return parameters
    
    def _extract_template_variables(self, content: str) -> List[str]:
        """Extract template variables from content"""
        variables = []
        
        # Look for common template variable patterns
        import re
        
        # Find {variable} patterns
        brace_vars = re.findall(r'\\{([^}]+)\\}', content)
        variables.extend(brace_vars)
        
        # Find {{variable}} patterns
        double_brace_vars = re.findall(r'\\{\\{([^}]+)\\}\\}', content)
        variables.extend(double_brace_vars)
        
        # Find ${variable} patterns
        dollar_vars = re.findall(r'\\$\\{([^}]+)\\}', content)
        variables.extend(dollar_vars)
        
        return list(set(variables))  # Remove duplicates
    
    def migrate_templates(self) -> List[MigrationResult]:
        """Migrate discovered templates to modular generators"""
        self.logger.info("Starting template migration...")
        
        # Group templates by category
        by_category = {}
        for template in self.discovered_templates.values():
            category = template.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(template)
        
        # Create modular generators for each category
        for category, templates in by_category.items():
            try:
                result = self._create_modular_generator(category, templates)
                self.migration_results.append(result)
            except Exception as e:
                error_result = MigrationResult(
                    template_name=f"{category.value}_generator",
                    source_file=Path("multiple"),
                    target_file=self.target_directory / f"{category.value}_generator.py",
                    success=False,
                    error_message=str(e)
                )
                self.migration_results.append(error_result)
        
        # Register template services
        self._register_template_services()
        
        success_count = sum(1 for r in self.migration_results if r.success)
        total_count = len(self.migration_results)
        
        self.logger.info(f"Template migration complete: {success_count}/{total_count} successful")
        
        return self.migration_results
    
    def _create_modular_generator(self, category: TemplateCategory, templates: List[TemplateDefinition]) -> MigrationResult:
        """Create modular template generator for a category"""
        generator_name = f"{category.value}_generator"
        target_file = self.target_directory / f"{generator_name}.py"
        
        try:
            # Generate modular generator code
            generator_code = self._generate_template_generator_code(category, templates)
            
            # Write to target file
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(generator_code)
            
            self.logger.info(f"Created modular generator: {target_file}")
            
            return MigrationResult(
                template_name=generator_name,
                source_file=Path("multiple"),
                target_file=target_file,
                success=True,
                extracted_variables=list(set(var for t in templates for var in t.variables))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create generator for {category.value}: {e}")
            return MigrationResult(
                template_name=generator_name,
                source_file=Path("multiple"),
                target_file=target_file,
                success=False,
                error_message=str(e)
            )
    
    def _generate_template_generator_code(self, category: TemplateCategory, templates: List[TemplateDefinition]) -> str:
        """Generate code for modular template generator"""
        class_name = f"{category.value.replace('_', ' ').title().replace(' ', '')}Generator"
        
        code = f'''#!/usr/bin/env python3
"""
{class_name} - Modular Template Generator
Migrated from monolithic template files by Agent A Template Migration System

Generated on: {datetime.now().isoformat()}
Category: {category.value}
Templates migrated: {len(templates)}
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
            self.additional_vars = {{}}


class {class_name}:
    """
    Modular template generator for {category.value} templates
    
    Extracted from monolithic files and integrated with architecture framework.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.category = "{category.value}"
        self.templates = {{
{self._generate_template_dict(templates)}
        }}
        
        self.logger.info(f"{{self.__class__.__name__}} initialized with {{len(self.templates)}} templates")
    
    def generate(self, template_name: str, context: TemplateContext) -> str:
        """Generate template with given context"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{{template_name}}' not found in {{self.category}} generator")
        
        template_data = self.templates[template_name]
        template_content = template_data['content']
        
        try:
            # Simple variable substitution
            result = self._substitute_variables(template_content, context)
            
            self.logger.debug(f"Generated {{template_name}} template ({{len(result)}} chars)")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate {{template_name}}: {{e}}")
            raise
    
    def list_templates(self) -> List[str]:
        """List available templates"""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        if template_name not in self.templates:
            return {{"error": f"Template '{{template_name}}' not found"}}
        
        return self.templates[template_name]
    
    def _substitute_variables(self, template_content: str, context: TemplateContext) -> str:
        """Substitute template variables with context values"""
        result = template_content
        
        # Basic substitutions
        substitutions = {{
            '{{{{project_name}}}}': context.project_name,
            '{{{{description}}}}': context.description,
            '{{{{author}}}}': context.author,
            '{{{{version}}}}': context.version,
            '{{{{date}}}}': datetime.now().strftime('%Y-%m-%d'),
            '{{{{year}}}}': str(datetime.now().year)
        }}
        
        # Add additional variables
        for key, value in context.additional_vars.items():
            substitutions[f'{{{{{{key}}}}}}'] = str(value)
        
        # Perform substitutions
        for placeholder, value in substitutions.items():
            result = result.replace(placeholder, value)
        
        return result


# Factory function
def create_{category.value}_generator() -> {class_name}:
    """Create {category.value} generator instance"""
    return {class_name}()


# Export for service registration
__all__ = ['{class_name}', 'TemplateContext', 'create_{category.value}_generator']
'''
        
        return code
    
    def _generate_template_dict(self, templates: List[TemplateDefinition]) -> str:
        """Generate template dictionary code"""
        template_entries = []
        
        for i, template in enumerate(templates):
            # Escape quotes and newlines in content
            content = template.content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            
            entry = f'''            "{template.name}": {{
                "content": "{content[:200]}...",  # Truncated for brevity
                "variables": {template.variables},
                "source_file": "{template.source_file}",
                "metadata": {template.metadata}
            }}'''
            
            template_entries.append(entry)
        
        return ',\n'.join(template_entries)
    
    def _register_template_services(self):
        """Register template generators as services"""
        try:
            # Register each category generator as a service
            for category in TemplateCategory:
                if any(t.category == category for t in self.discovered_templates.values()):
                    service_name = f"{category.value}_generator"
                    
                    self.service_registry.define_service(ServiceDefinition(
                        name=service_name,
                        service_type=ServiceType.TEMPLATE,
                        layer=ArchitecturalLayer.APPLICATION,
                        lifetime=LifetimeScope.SINGLETON,
                        metadata={
                            "category": category.value,
                            "migrated": True,
                            "migration_time": datetime.now().isoformat()
                        }
                    ))
            
            self.logger.info("Template services registered with architecture framework")
            
        except Exception as e:
            self.logger.error(f"Failed to register template services: {e}")
    
    def get_migration_report(self) -> Dict[str, Any]:
        """Get comprehensive migration report"""
        successful = [r for r in self.migration_results if r.success]
        failed = [r for r in self.migration_results if not r.success]
        
        return {
            'templates_discovered': len(self.discovered_templates),
            'migrations_attempted': len(self.migration_results),
            'migrations_successful': len(successful),
            'migrations_failed': len(failed),
            'success_rate': len(successful) / max(len(self.migration_results), 1),
            'categories_processed': len(set(t.category for t in self.discovered_templates.values())),
            'target_directory': str(self.target_directory),
            'migration_results': [
                {
                    'template_name': r.template_name,
                    'success': r.success,
                    'target_file': str(r.target_file),
                    'error_message': r.error_message
                }
                for r in self.migration_results
            ],
            'timestamp': datetime.now().isoformat()
        }


# Global migration system instance
_migration_system: Optional[TemplateMigrationSystem] = None


def get_migration_system() -> TemplateMigrationSystem:
    """Get global template migration system"""
    global _migration_system
    if _migration_system is None:
        _migration_system = TemplateMigrationSystem()
    return _migration_system


def migrate_all_templates() -> Dict[str, Any]:
    """Discover and migrate all templates"""
    system = get_migration_system()
    
    # Discover templates
    discovered = system.discover_templates()
    
    # Migrate templates
    results = system.migrate_templates()
    
    # Get report
    report = system.get_migration_report()
    
    return {
        'discovered_templates': discovered,
        'migration_results': results,
        'report': report
    }