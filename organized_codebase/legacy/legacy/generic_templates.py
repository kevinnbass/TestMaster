"""
Generic README Templates
=======================

Modularized from readme_templates.py for better maintainability.
Contains generic README templates suitable for any project type.

Author: Agent E - Infrastructure Consolidation
"""

from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat
from typing import Dict


class GenericTemplates:
    """Factory for creating generic README templates."""
    
    @staticmethod
    def create_comprehensive_template() -> Template:
        """Create comprehensive generic README template."""
        return Template(
            metadata=TemplateMetadata(
                name="generic_comprehensive",
                description="Comprehensive README template for any project type",
                template_type=TemplateType.README,
                format=TemplateFormat.MARKDOWN,
                author="TestMaster",
                version="1.0.0",
                tags=["generic", "comprehensive"],
                required_variables=["project_name", "description", "author"],
                optional_variables=["version", "license_type", "installation_steps", "usage_examples", "features"],
                target_audience="all"
            ),
            content='''# {{project_name}}

{{#badges}}
{{#badges}}
![{{alt_text}}]({{url}})
{{/badges}}
{{/badges}}

{{description}}

{{#version}}
**Version:** {{version}}
{{/version}}

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
{{#api_endpoints}}
- [API Reference](#api-reference)
{{/api_endpoints}}
- [Contributing](#contributing)
- [License](#license)
{{#changelog_link}}
- [Changelog](#changelog)
{{/changelog_link}}

## Features

{{#features}}
- {{.}}
{{/features}}
{{^features}}
- Feature 1
- Feature 2
- Feature 3
{{/features}}

## Installation

{{#installation_steps}}
{{#installation_steps}}
{{step_number}}. {{description}}
   ```bash
   {{command}}
   ```
{{/installation_steps}}
{{/installation_steps}}

{{^installation_steps}}
### Using pip

```bash
pip install {{project_name}}
```

### From source

```bash
git clone https://github.com/{{author}}/{{project_name}}.git
cd {{project_name}}
pip install -e .
```
{{/installation_steps}}

## Usage

{{#usage_examples}}
{{#usage_examples}}
### {{title}}

{{description}}

```python
{{code}}
```

{{#output}}
Output:
```
{{output}}
```
{{/output}}

{{/usage_examples}}
{{/usage_examples}}

{{^usage_examples}}
### Basic Usage

```python
from {{project_name}} import main

# Your usage example here
result = main()
print(result)
```
{{/usage_examples}}

{{#api_endpoints}}
## API Reference

{{#api_endpoints}}
### {{method}} {{endpoint}}

{{description}}

**Parameters:**
{{#parameters}}
- `{{name}}` ({{type}}): {{description}}
{{/parameters}}

**Response:**
```json
{{response_example}}
```

{{/api_endpoints}}
{{/api_endpoints}}

## Contributing

{{#contributing_guidelines}}
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Unix) or `venv\\Scripts\\activate` (Windows)
4. Install development dependencies: `pip install -e ".[dev]"`
5. Run tests: `pytest`

### Submitting Changes

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes and add tests
3. Run the test suite: `pytest`
4. Commit your changes: `git commit -am "Add new feature"`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request
{{/contributing_guidelines}}

{{^contributing_guidelines}}
Contributions are welcome! Please feel free to submit a Pull Request.
{{/contributing_guidelines}}

## License

{{#license_type}}
This project is licensed under the {{license_type}} License - see the [LICENSE](LICENSE) file for details.
{{/license_type}}

{{^license_type}}
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
{{/license_type}}

## Author

**{{author}}**

{{#documentation_link}}
- Documentation: [{{documentation_link}}]({{documentation_link}})
{{/documentation_link}}
{{#demo_link}}
- Demo: [{{demo_link}}]({{demo_link}})
{{/demo_link}}

{{#changelog_link}}
## Changelog

For a detailed history of changes, see the [CHANGELOG]({{changelog_link}}).
{{/changelog_link}}

---

*Generated with ❤️ by TestMaster Documentation System*
''',
            examples=[
                '''# MyAwesomeProject

A brief description of what this project does and who it's for.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
pip install myawesomeproject
```

## Usage

```python
from myawesomeproject import main

result = main()
print(result)
```

## License

MIT License
'''
            ]
        )
    
    @staticmethod
    def create_minimal_template() -> Template:
        """Create minimal generic README template."""
        return Template(
            metadata=TemplateMetadata(
                name="generic_minimal",
                description="Minimal README template for simple projects",
                template_type=TemplateType.README,
                format=TemplateFormat.MARKDOWN,
                author="TestMaster",
                version="1.0.0",
                tags=["generic", "minimal"],
                required_variables=["project_name", "description"],
                optional_variables=["installation", "usage", "license_type"],
                target_audience="beginner"
            ),
            content='''# {{project_name}}

{{description}}

## Installation

{{#installation}}
{{installation}}
{{/installation}}
{{^installation}}
```bash
pip install {{project_name}}
```
{{/installation}}

## Usage

{{#usage}}
{{usage}}
{{/usage}}
{{^usage}}
```python
import {{project_name}}

# Add usage example here
```
{{/usage}}

## License

{{license_type}}
''',
            examples=[
                '''# Simple Calculator

A basic calculator for arithmetic operations.

## Installation

```bash
pip install simple-calculator
```

## Usage

```python
from calculator import Calculator

calc = Calculator()
result = calc.add(2, 3)
print(result)  # 5
```

## License

MIT
'''
            ]
        )
    
    @staticmethod
    def create_all() -> Dict[str, Template]:
        """Create all generic templates."""
        return {
            "generic_comprehensive": GenericTemplates.create_comprehensive_template(),
            "generic_minimal": GenericTemplates.create_minimal_template()
        }


__all__ = ['GenericTemplates']