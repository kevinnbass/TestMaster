"""
Interactive Documentation System

Creates CLI wizard integration with template-based learning and 
scaffolding documentation based on LLama-Agents interactive approach.
"""

import os
import json
import shutil
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of interactive elements."""
    CLI_WIZARD = "cli_wizard"
    TEMPLATE_GENERATOR = "template_generator"
    CODE_SCAFFOLD = "code_scaffold"
    GUIDED_TUTORIAL = "guided_tutorial"
    CONFIGURATION_BUILDER = "configuration_builder"
    PROJECT_INITIALIZER = "project_initializer"


class TemplateType(Enum):
    """Types of project templates."""
    MINIMAL_AGENT = "minimal_agent"
    MULTI_AGENT = "multi_agent"
    ORCHESTRATOR = "orchestrator"
    MICROSERVICE = "microservice"
    FULL_STACK = "full_stack"
    CUSTOM = "custom"


class ScaffoldLevel(Enum):
    """Scaffolding complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PRODUCTION = "production"


@dataclass
class InteractiveElement:
    """Represents an interactive documentation element."""
    name: str
    interaction_type: InteractionType
    description: str
    prompt_sequence: List[Dict[str, Any]] = field(default_factory=list)
    templates: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    help_text: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectTemplate:
    """Project template definition."""
    name: str
    template_type: TemplateType
    scaffold_level: ScaffoldLevel
    description: str
    files: List[Dict[str, Any]] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    post_generation_steps: List[str] = field(default_factory=list)
    documentation: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class WizardStep:
    """Single step in CLI wizard."""
    step_id: str
    prompt: str
    input_type: str  # text, choice, boolean, file_path
    validation: Optional[Callable] = None
    choices: List[str] = field(default_factory=list)
    default_value: Any = None
    help_text: str = ""
    conditional: Optional[str] = None  # Condition for showing this step


@dataclass
class GuidedTutorial:
    """Interactive guided tutorial."""
    name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    validation_commands: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_time: str = "30 minutes"


class InteractiveDocsSystem:
    """
    Interactive documentation system inspired by LLama-Agents
    CLI wizard integration and template-based learning approach.
    """
    
    def __init__(self, docs_dir: str = "interactive-docs"):
        """Initialize interactive docs system."""
        self.docs_dir = Path(docs_dir)
        self.interactive_elements = []
        self.templates = []
        self.tutorials = []
        self.wizard_configs = {}
        self.template_registry = {}
        logger.info(f"Interactive docs system initialized at {docs_dir}")
        
    def create_interactive_element(self,
                                 name: str,
                                 interaction_type: InteractionType,
                                 description: str,
                                 **kwargs) -> InteractiveElement:
        """Create interactive documentation element."""
        element = InteractiveElement(
            name=name,
            interaction_type=interaction_type,
            description=description,
            **kwargs
        )
        
        self.interactive_elements.append(element)
        logger.info(f"Created interactive element: {name} ({interaction_type.value})")
        return element
        
    def create_project_template(self,
                              name: str,
                              template_type: TemplateType,
                              scaffold_level: ScaffoldLevel,
                              description: str,
                              **kwargs) -> ProjectTemplate:
        """Create project template."""
        template = ProjectTemplate(
            name=name,
            template_type=template_type,
            scaffold_level=scaffold_level,
            description=description,
            **kwargs
        )
        
        self.templates.append(template)
        self.template_registry[name] = template
        logger.info(f"Created project template: {name} ({template_type.value})")
        return template
        
    def create_cli_wizard(self, name: str, steps: List[WizardStep]) -> str:
        """Create CLI wizard configuration."""
        wizard_config = {
            "name": name,
            "description": f"Interactive wizard for {name}",
            "steps": []
        }
        
        for step in steps:
            step_config = {
                "id": step.step_id,
                "prompt": step.prompt,
                "type": step.input_type,
                "help": step.help_text,
                "default": step.default_value,
                "conditional": step.conditional
            }
            
            if step.choices:
                step_config["choices"] = step.choices
                
            wizard_config["steps"].append(step_config)
            
        self.wizard_configs[name] = wizard_config
        
        # Generate CLI script
        cli_script = self._generate_cli_wizard_script(wizard_config)
        return cli_script
        
    def create_guided_tutorial(self,
                             name: str,
                             description: str,
                             steps: List[Dict[str, Any]]) -> GuidedTutorial:
        """Create guided interactive tutorial."""
        tutorial = GuidedTutorial(
            name=name,
            description=description,
            steps=steps
        )
        
        self.tutorials.append(tutorial)
        logger.info(f"Created guided tutorial: {name}")
        return tutorial
        
    def generate_template_scaffolding(self, template: ProjectTemplate, variables: Dict[str, Any]) -> Dict[str, str]:
        """Generate project scaffolding from template."""
        scaffolding = {}
        
        # Process files
        for file_config in template.files:
            file_path = file_config["path"]
            file_content = file_config.get("content", "")
            
            # Replace template variables
            for var_name, var_value in variables.items():
                file_path = file_path.replace(f"{{{{{var_name}}}}}", str(var_value))
                file_content = file_content.replace(f"{{{{{var_name}}}}}", str(var_value))
                
            scaffolding[file_path] = file_content
            
        return scaffolding
        
    def generate_wizard_documentation(self) -> str:
        """Generate documentation for CLI wizards."""
        doc = [
            "# Interactive CLI Wizards",
            "",
            "Use interactive wizards to quickly set up projects and configurations.",
            "",
            "## Available Wizards",
            ""
        ]
        
        for element in self.interactive_elements:
            if element.interaction_type == InteractionType.CLI_WIZARD:
                doc.extend([
                    f"### {element.name}",
                    "",
                    element.description,
                    "",
                    "**Usage:**",
                    "```bash",
                    f"llamactl init --wizard {element.name.lower()}",
                    "```",
                    ""
                ])
                
                if element.examples:
                    doc.extend([
                        "**Examples:**",
                        ""
                    ])
                    for example in element.examples:
                        doc.extend([
                            f"- **{example.get('title', 'Example')}:**",
                            f"  ```bash",
                            f"  {example.get('command', '')}",
                            f"  ```",
                            ""
                        ])
                        
        return "\n".join(doc)
        
    def generate_template_documentation(self) -> str:
        """Generate documentation for project templates."""
        doc = [
            "# Project Templates",
            "",
            "Quickly bootstrap projects with pre-configured templates.",
            "",
            "## Template Categories",
            ""
        ]
        
        # Group templates by type
        by_type = {}
        for template in self.templates:
            if template.template_type not in by_type:
                by_type[template.template_type] = []
            by_type[template.template_type].append(template)
            
        for template_type, templates in by_type.items():
            doc.extend([
                f"### {template_type.value.replace('_', ' ').title()}",
                ""
            ])
            
            for template in sorted(templates, key=lambda x: list(ScaffoldLevel).index(x.scaffold_level)):
                doc.extend([
                    f"#### {template.name}",
                    "",
                    f"**Level:** {template.scaffold_level.value.title()}",
                    "",
                    template.description,
                    "",
                    "**Usage:**",
                    "```bash",
                    f"llamactl init --template {template.name}",
                    "```",
                    ""
                ])
                
                if template.dependencies:
                    doc.extend([
                        "**Dependencies:**",
                        ""
                    ])
                    for dep in template.dependencies:
                        doc.append(f"- {dep}")
                    doc.append("")
                    
                if template.post_generation_steps:
                    doc.extend([
                        "**Post-Generation Steps:**",
                        ""
                    ])
                    for i, step in enumerate(template.post_generation_steps, 1):
                        doc.append(f"{i}. {step}")
                    doc.append("")
                    
        return "\n".join(doc)
        
    def generate_tutorial_guide(self, tutorial: GuidedTutorial) -> str:
        """Generate guide for interactive tutorial."""
        guide = [
            f"# Interactive Tutorial: {tutorial.name}",
            "",
            tutorial.description,
            "",
            f"**Estimated Time:** {tutorial.estimated_time}",
            ""
        ]
        
        if tutorial.prerequisites:
            guide.extend([
                "## Prerequisites",
                ""
            ])
            for prereq in tutorial.prerequisites:
                guide.append(f"- {prereq}")
            guide.append("")
            
        guide.extend([
            "## Tutorial Steps",
            ""
        ])
        
        for i, step in enumerate(tutorial.steps, 1):
            guide.extend([
                f"### Step {i}: {step.get('title', 'Untitled Step')}",
                "",
                step.get('description', ''),
                ""
            ])
            
            if step.get('code'):
                guide.extend([
                    "```bash",
                    step['code'],
                    "```",
                    ""
                ])
                
            if step.get('expected_output'):
                guide.extend([
                    "**Expected Output:**",
                    "```",
                    step['expected_output'],
                    "```",
                    ""
                ])
                
            if step.get('validation'):
                guide.extend([
                    f"**Validation:** {step['validation']}",
                    ""
                ])
                
        if tutorial.checkpoints:
            guide.extend([
                "## Checkpoints",
                ""
            ])
            for checkpoint in tutorial.checkpoints:
                guide.append(f"- {checkpoint}")
            guide.append("")
            
        return "\n".join(guide)
        
    def _generate_cli_wizard_script(self, config: Dict[str, Any]) -> str:
        """Generate Python CLI wizard script."""
        script = f'''#!/usr/bin/env python3
"""
Interactive CLI Wizard: {config["name"]}
{config.get("description", "")}
"""

import sys
import json
from pathlib import Path

def prompt_user(prompt, input_type="text", choices=None, default=None, help_text=""):
    """Prompt user for input with validation."""
    if help_text:
        print(f"\\n{help_text}")
    
    prompt_text = prompt
    if default is not None:
        prompt_text += f" [{default}]"
    prompt_text += ": "
    
    while True:
        if input_type == "choice" and choices:
            print("\\nChoices:")
            for i, choice in enumerate(choices, 1):
                print(f"  {i}. {choice}")
            response = input(f"{prompt_text}")
            
            # Handle numeric choice
            try:
                choice_idx = int(response) - 1
                if 0 <= choice_idx < len(choices):
                    return choices[choice_idx]
            except ValueError:
                pass
                
            # Handle text choice
            if response in choices:
                return response
            elif not response and default:
                return default
            else:
                print("Invalid choice. Please try again.")
                continue
                
        elif input_type == "boolean":
            response = input(f"{prompt_text}").lower().strip()
            if response in ["y", "yes", "true", "1"]:
                return True
            elif response in ["n", "no", "false", "0"]:
                return False
            elif not response and default is not None:
                return default
            else:
                print("Please enter y/n or yes/no")
                continue
                
        else:  # text input
            response = input(prompt_text).strip()
            if response:
                return response
            elif default is not None:
                return default
            else:
                print("This field is required. Please enter a value.")
                continue

def main():
    """Main wizard function."""
    print(f"\\n🧙 {config['name']} Wizard")
    print(f"{config.get('description', '')}")
    print("-" * 50)
    
    results = {{}}
    
'''
        
        # Add wizard steps
        for step in config["steps"]:
            script += f'''
    # Step: {step["id"]}
    '''
            
            if step.get("conditional"):
                script += f'''
    if {step["conditional"]}:
'''
                indent = "    "
            else:
                indent = ""
                
            script += f'''{indent}results["{step["id"]}"] = prompt_user(
        "{step["prompt"]}",
        input_type="{step["type"]}",
        choices={step.get("choices", None)},
        default={repr(step.get("default"))},
        help_text="{step.get("help", "")}"
    )
'''
        
        script += '''
    # Display results
    print("\\n📋 Configuration Summary:")
    print("-" * 30)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Save results
    config_file = Path("wizard_config.json")
    with open(config_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n✅ Configuration saved to {config_file}")
    print("\\n🚀 Next steps:")
    print("1. Review the generated configuration")
    print("2. Run the project initialization")
    print("3. Follow the setup guide")

if __name__ == "__main__":
    main()
'''
        
        return script
        
    def create_default_templates(self) -> None:
        """Create default project templates."""
        # Minimal Agent Template
        self.create_project_template(
            "minimal-agent",
            TemplateType.MINIMAL_AGENT,
            ScaffoldLevel.BASIC,
            "Simple single-agent project with minimal configuration",
            files=[
                {
                    "path": "{{project_name}}/main.py",
                    "content": '''"""
{{project_name}} - Minimal Agent Project
"""

from llama_agents import SimpleAgent

def main():
    """Main application entry point."""
    agent = SimpleAgent(
        name="{{agent_name}}",
        description="{{agent_description}}"
    )
    
    # Your agent logic here
    result = agent.process("Hello, world!")
    print(f"Agent response: {result}")

if __name__ == "__main__":
    main()
'''
                },
                {
                    "path": "{{project_name}}/requirements.txt",
                    "content": "llama-agents>=0.1.0\\nllama-index>=0.10.0\\n"
                },
                {
                    "path": "{{project_name}}/README.md",
                    "content": '''# {{project_name}}

{{project_description}}

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the agent:
   ```bash
   python main.py
   ```

## Configuration

Customize your agent behavior in `main.py`.
'''
                }
            ],
            variables={
                "project_name": "my-agent",
                "agent_name": "MyAgent", 
                "agent_description": "A helpful AI assistant",
                "project_description": "My minimal agent project"
            },
            post_generation_steps=[
                "Install dependencies with `pip install -r requirements.txt`",
                "Customize agent behavior in main.py",
                "Test your agent with `python main.py`"
            ]
        )
        
        # Multi-Agent Template
        self.create_project_template(
            "multi-agent",
            TemplateType.MULTI_AGENT,
            ScaffoldLevel.INTERMEDIATE,
            "Multi-agent system with orchestrator pattern",
            files=[
                {
                    "path": "{{project_name}}/agents/__init__.py",
                    "content": ""
                },
                {
                    "path": "{{project_name}}/agents/research_agent.py", 
                    "content": '''from llama_agents import SimpleAgent

class ResearchAgent(SimpleAgent):
    """Agent specialized in research tasks."""
    
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            description="Conducts research and gathers information"
        )
'''
                },
                {
                    "path": "{{project_name}}/orchestrator.py",
                    "content": '''from llama_agents import Orchestrator, Tool
from agents.research_agent import ResearchAgent

class {{project_name}}Orchestrator:
    """Main orchestrator for {{project_name}}."""
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        
        self.orchestrator = Orchestrator(
            tools=[
                Tool.from_agent(self.research_agent, "research")
            ]
        )
    
    async def process(self, query):
        """Process user query."""
        return await self.orchestrator.run(query)
'''
                }
            ],
            variables={
                "project_name": "multi-agent-system"
            },
            dependencies=[
                "llama-agents>=0.1.0",
                "llama-index>=0.10.0",
                "redis>=4.0.0"
            ]
        )
        
    def create_default_tutorials(self) -> None:
        """Create default guided tutorials."""
        # Getting Started Tutorial
        self.create_guided_tutorial(
            "getting-started",
            "Complete beginner's guide to building your first agent",
            [
                {
                    "title": "Environment Setup",
                    "description": "Set up your development environment",
                    "code": "pip install llama-agents llama-index",
                    "validation": "Check that packages are installed correctly"
                },
                {
                    "title": "Create First Agent",
                    "description": "Create a simple agent",
                    "code": '''python -c "
from llama_agents import SimpleAgent
agent = SimpleAgent(name='MyFirstAgent')
print('Agent created successfully!')
"''',
                    "expected_output": "Agent created successfully!"
                },
                {
                    "title": "Test Agent Interaction",
                    "description": "Send a message to your agent",
                    "code": '''python -c "
from llama_agents import SimpleAgent
agent = SimpleAgent(name='TestAgent')
response = agent.process('Hello!')
print(f'Response: {response}')
"''',
                    "validation": "Verify agent responds appropriately"
                }
            ]
        )
        
    def export_interactive_docs(self, output_dir: str) -> None:
        """Export all interactive documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main documentation
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write("# Interactive Documentation\\n\\nChoose your learning path:\\n\\n")
            f.write("- [CLI Wizards](wizards/)\\n")
            f.write("- [Project Templates](templates/)\\n") 
            f.write("- [Guided Tutorials](tutorials/)\\n")
            
        # Wizards documentation
        wizards_dir = output_path / "wizards"
        wizards_dir.mkdir(exist_ok=True)
        
        with open(wizards_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_wizard_documentation())
            
        # Generate wizard scripts
        for name, config in self.wizard_configs.items():
            script_content = self._generate_cli_wizard_script(config)
            with open(wizards_dir / f"{name.lower()}_wizard.py", 'w', encoding='utf-8') as f:
                f.write(script_content)
                
        # Templates documentation
        templates_dir = output_path / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        with open(templates_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_template_documentation())
            
        # Generate template files
        for template in self.templates:
            template_dir = templates_dir / template.name
            template_dir.mkdir(exist_ok=True)
            
            # Template configuration
            config = {
                "name": template.name,
                "type": template.template_type.value,
                "level": template.scaffold_level.value,
                "description": template.description,
                "variables": template.variables,
                "files": template.files,
                "directories": template.directories,
                "dependencies": template.dependencies,
                "post_generation_steps": template.post_generation_steps
            }
            
            with open(template_dir / "template.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
                
        # Tutorials documentation
        tutorials_dir = output_path / "tutorials"
        tutorials_dir.mkdir(exist_ok=True)
        
        for tutorial in self.tutorials:
            with open(tutorials_dir / f"{tutorial.name}.md", 'w', encoding='utf-8') as f:
                f.write(self.generate_tutorial_guide(tutorial))
                
        logger.info(f"Exported interactive docs to {output_dir}")