#!/usr/bin/env python3
"""
ReadmeGenerator - Modular Template Generator
Migrated from monolithic template files by Agent A Template Migration System

Generated on: 2025-08-23T00:37:36.645603
Category: readme
Templates migrated: 13
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
    template_type: str = "generic"
    tech_stack: str = ""
    additional_vars: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_vars is None:
            self.additional_vars = {}
        
        # Store all parameters in additional_vars for template access
        self.additional_vars.update({
            'project_name': self.project_name,
            'description': self.description,
            'author': self.author,
            'version': self.version,
            'template_type': self.template_type,
            'tech_stack': self.tech_stack
        })


class ReadmeGenerator:
    """
    Modular template generator for readme templates
    
    Extracted from monolithic files and integrated with architecture framework.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.category = "readme"
        self.templates = {
            "ReadmeContext": {
                "content": "# Class template: ReadmeContext...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'class', 'ast_type': 'ClassDef'}
            },
            "ReadmeTemplateManager": {
                "content": "# Class template: ReadmeTemplateManager...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'class', 'ast_type': 'ClassDef'}
            },
            "generate_readme": {
                "content": "# Function template: generate_readme...",  # Truncated for brevity
                "variables": ['context', 'style'],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'function', 'ast_type': 'FunctionDef'}
            },
            "template_16": {
                "content": "    \"\"\"Types of projects for README templates.\"\"\"\n    WEB_APPLICATION = \"web_application\"\n    API = \"api\"\n    LIBRARY = \"library\"\n    CLI_TOOL = \"cli_tool\"\n    DATA_SCIENCE = \"data...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_78": {
                "content": "        \"\"\"Initialize the README template manager.\"\"\"\n        self.templates: Dict[str, Template] = {}\n        self._initialize_builtin_templates()\n    \n    def _initialize_builtin_templates...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_93": {
                "content": "        \"\"\"Create generic README templates.\"\"\"\n        \n        # Comprehensive template\n        comprehensive_template = Template(\n            metadata=TemplateMetadata(\n                na...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_400": {
                "content": "        \"\"\"Create web application README templates.\"\"\"\n        \n        web_app_template = Template(\n            metadata=TemplateMetadata(\n                name=\"web_application\",\n       ...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_578": {
                "content": "        \"\"\"Create API README templates.\"\"\"\n        \n        api_template = Template(\n            metadata=TemplateMetadata(\n                name=\"api_service\",\n                description...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_800": {
                "content": "        \"\"\"Create library README templates.\"\"\"\n        \n        library_template = Template(\n            metadata=TemplateMetadata(\n                name=\"python_library\",\n                ...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_1049": {
                "content": "        \"\"\"Create CLI tool README templates.\"\"\"\n        \n        cli_template = Template(\n            metadata=TemplateMetadata(\n                name=\"cli_tool\",\n                descripti...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_1320": {
                "content": "        \"\"\"Create data science README templates.\"\"\"\n        \n        data_science_template = Template(\n            metadata=TemplateMetadata(\n                name=\"data_science\",\n        ...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_1622": {
                "content": "        \"\"\"Create machine learning README templates.\"\"\"\n        \n        ml_template = Template(\n            metadata=TemplateMetadata(\n                name=\"machine_learning\",\n          ...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            },
            "template_2085": {
                "content": "        \"\"\"Prepare variables for template rendering.\"\"\"\n        variables = {\n            \"project_name\": context.project_name,\n            \"description\": context.description,\n          ...",  # Truncated for brevity
                "variables": [],
                "source_file": "TestMaster\readme_templates.py",
                "metadata": {'type': 'string_literal'}
            }
        }
        
        self.logger.info(f"{self.__class__.__name__} initialized with {len(self.templates)} templates")
    
    def generate_readme(self, context: TemplateContext) -> str:
        """Generate README content based on template context"""
        template_type = context.template_type.lower()
        
        if template_type == "library" or template_type == "python_library":
            return self._generate_library_readme(context)
        elif template_type == "web_application":
            return self._generate_webapp_readme(context)
        elif template_type == "cli_tool":
            return self._generate_cli_readme(context)
        elif template_type == "api" or template_type == "api_service":
            return self._generate_api_readme(context)
        else:
            return self._generate_generic_readme(context)
    
    def _generate_generic_readme(self, context: TemplateContext) -> str:
        """Generate generic README template"""
        return f"""# {context.project_name}

{context.description}

## Installation

```bash
pip install {context.project_name.lower().replace(' ', '-')}
```

## Usage

Basic usage examples for {context.project_name}.

```python
import {context.project_name.lower().replace(' ', '_')}

# Example usage here
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

{context.author}

## Version

{context.version}
"""
    
    def _generate_library_readme(self, context: TemplateContext) -> str:
        """Generate Python library README template"""
        package_name = context.project_name.lower().replace(' ', '_').replace('-', '_')
        
        return f"""# {context.project_name}

{context.description}

## Installation

Install from PyPI:

```bash
pip install {context.project_name.lower().replace(' ', '-')}
```

## Quick Start

```python
import {package_name}

# Initialize the library
lib = {package_name}.{context.project_name.replace(' ', '')}()

# Use the library
result = lib.main_function()
print(result)
```

## API Reference

### Main Classes

#### `{context.project_name.replace(' ', '')}`

Main class for {context.project_name.lower()} functionality.

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**{context.author}**
- Version: {context.version}

---

*Generated with Agent A Template Generator*
"""
    
    def _generate_webapp_readme(self, context: TemplateContext) -> str:
        """Generate web application README template"""
        tech_stack = context.tech_stack if context.tech_stack else "React, Node.js, PostgreSQL"
        
        return f"""# {context.project_name}

{context.description}

## 🚀 Tech Stack

{tech_stack}

## 📋 Features

- ✅ User authentication and authorization
- ✅ Responsive design for mobile and desktop
- ✅ Real-time updates

## 🛠️ Installation & Setup

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Database (PostgreSQL/MongoDB)

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/{context.author.lower()}/{context.project_name.lower().replace(' ', '-')}.git
cd {context.project_name.lower().replace(' ', '-')}
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm start
```

## Author

**{context.author}**
- Version: {context.version}

---

*Generated with Agent A Template Generator*
"""
    
    def _generate_cli_readme(self, context: TemplateContext) -> str:
        """Generate CLI tool README template"""
        tool_name = context.project_name.lower().replace(' ', '-')
        
        return f"""# {context.project_name}

{context.description}

## 📦 Installation

```bash
pip install {tool_name}
```

## 🚀 Quick Start

```bash
# Basic usage
{tool_name} --help

# Run with default settings
{tool_name} input.txt
```

## 📖 Commands

### Main Commands

#### `{tool_name} process`
Process files with default settings.

## Author

**{context.author}**
- Version: {context.version}

---

*Generated with Agent A Template Generator*
"""
    
    def _generate_api_readme(self, context: TemplateContext) -> str:
        """Generate API service README template"""
        return f"""# {context.project_name} API

{context.description}

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL/MongoDB

### Installation

```bash
git clone https://github.com/{context.author.lower()}/{context.project_name.lower().replace(' ', '-')}.git
cd {context.project_name.lower().replace(' ', '-')}
pip install -r requirements.txt
```

### Run the API

```bash
python app.py
```

## 📚 API Documentation

- **Swagger UI**: http://localhost:8000/docs

## Author

**{context.author}**
- API Version: {context.version}

---

*Generated with Agent A Template Generator*
"""
    
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
def create_readme_generator() -> ReadmeGenerator:
    """Create readme generator instance"""
    return ReadmeGenerator()


# Export for service registration
__all__ = ['ReadmeGenerator', 'TemplateContext', 'create_readme_generator']
