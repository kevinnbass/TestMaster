"""
Generic Template Generator

Provides generic README templates suitable for any project type.
"""

from .base_generator import BaseTemplateGenerator
from ..models import Template


class GenericTemplateGenerator(BaseTemplateGenerator):
    """Generator for generic README templates."""
    
    def initialize_templates(self):
        """Initialize generic templates."""
        self._create_comprehensive_template()
        self._create_minimal_template()
    
    def _create_comprehensive_template(self):
        """Create comprehensive generic template."""
        metadata = self.create_metadata(
            name="generic_comprehensive",
            description="Comprehensive README template for any project type",
            tags=["generic", "comprehensive"],
            required_variables=["project_name", "description", "author"],
            optional_variables=["version", "license_type", "installation_steps", "usage_examples", "features"]
        )
        
        content = '''# {{project_name}}

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

## Contributing

{{#contributing_guidelines}}
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.
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

---

*Generated with ❤️ by TestMaster Documentation System*
'''
        
        examples = [
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
        
        template = Template(
            metadata=metadata,
            content=content,
            examples=examples
        )
        
        self.add_template("generic_comprehensive", template)
    
    def _create_minimal_template(self):
        """Create minimal generic template."""
        metadata = self.create_metadata(
            name="generic_minimal",
            description="Minimal README template for simple projects",
            tags=["generic", "minimal"],
            required_variables=["project_name", "description"],
            optional_variables=["installation", "usage", "license_type"],
            target_audience="beginner"
        )
        
        content = '''# {{project_name}}

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
'''
        
        examples = [
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
        
        template = Template(
            metadata=metadata,
            content=content,
            examples=examples
        )
        
        self.add_template("generic_minimal", template)