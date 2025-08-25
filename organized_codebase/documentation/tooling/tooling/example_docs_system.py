"""
Example Documentation System

Creates comprehensive example-driven documentation with code snippets,
interactive demos, and progressive learning paths based on AgentScope patterns.
"""

import os
import re
import ast
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExampleType(Enum):
    """Types of examples for documentation."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    TUTORIAL = "tutorial"
    QUICKSTART = "quickstart"
    COOKBOOK = "cookbook"
    

class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    BASH = "bash"
    JSON = "json"
    YAML = "yaml"


@dataclass
class CodeExample:
    """Represents a code example with metadata."""
    title: str
    description: str
    code: str
    language: CodeLanguage
    example_type: ExampleType
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    difficulty: int = 1  # 1-5 scale
    expected_output: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    related_examples: List[str] = field(default_factory=list)


@dataclass
class ExampleCollection:
    """Collection of related examples."""
    name: str
    description: str
    examples: List[CodeExample] = field(default_factory=list)
    category: str = "general"
    learning_path: List[str] = field(default_factory=list)


class ExampleDocsSystem:
    """
    Example-driven documentation system inspired by AgentScope's
    comprehensive example structure and tutorial approach.
    """
    
    def __init__(self, docs_path: str = "docs/examples"):
        """Initialize example documentation system."""
        self.docs_path = Path(docs_path)
        self.collections = {}
        self.examples = {}
        self.templates = {
            "basic": self._get_basic_template(),
            "tutorial": self._get_tutorial_template(),
            "quickstart": self._get_quickstart_template()
        }
        logger.info(f"Example docs system initialized at {docs_path}")
        
    def create_example(self, 
                      title: str,
                      description: str,
                      code: str,
                      language: CodeLanguage,
                      example_type: ExampleType,
                      **kwargs) -> CodeExample:
        """Create a new code example."""
        example = CodeExample(
            title=title,
            description=description,
            code=code,
            language=language,
            example_type=example_type,
            **kwargs
        )
        
        example_id = self._generate_example_id(title)
        self.examples[example_id] = example
        
        logger.info(f"Created example: {title}")
        return example
        
    def create_collection(self, name: str, description: str, category: str = "general") -> ExampleCollection:
        """Create a new example collection."""
        collection = ExampleCollection(
            name=name,
            description=description,
            category=category
        )
        
        collection_id = self._generate_collection_id(name)
        self.collections[collection_id] = collection
        
        logger.info(f"Created collection: {name}")
        return collection
        
    def add_to_collection(self, collection_id: str, example: CodeExample) -> None:
        """Add example to collection."""
        if collection_id in self.collections:
            self.collections[collection_id].examples.append(example)
            logger.info(f"Added example to collection {collection_id}")
        else:
            logger.error(f"Collection {collection_id} not found")
            
    def generate_quickstart_guide(self, 
                                 framework_name: str,
                                 key_concepts: List[str],
                                 examples: List[CodeExample]) -> str:
        """Generate a quickstart guide with examples."""
        guide = [
            f"# {framework_name} Quickstart Guide",
            "",
            f"Get started with {framework_name} in minutes!",
            "",
            "## Key Concepts",
            ""
        ]
        
        # Add key concepts
        for concept in key_concepts:
            guide.append(f"- **{concept}**")
            
        guide.extend(["", "## Quick Examples", ""])
        
        # Add examples
        for example in examples[:3]:  # Top 3 examples
            guide.extend([
                f"### {example.title}",
                "",
                example.description,
                "",
                f"```{example.language.value}",
                example.code,
                "```",
                ""
            ])
            
            if example.expected_output:
                guide.extend([
                    "**Expected Output:**",
                    "```",
                    example.expected_output,
                    "```",
                    ""
                ])
                
        return "\n".join(guide)
        
    def generate_tutorial_series(self, 
                               series_name: str,
                               examples: List[CodeExample]) -> str:
        """Generate a progressive tutorial series."""
        # Sort examples by difficulty
        sorted_examples = sorted(examples, key=lambda x: x.difficulty)
        
        tutorial = [
            f"# {series_name} Tutorial Series",
            "",
            "A step-by-step guide to mastering the concepts.",
            "",
            "## Tutorial Overview",
            ""
        ]
        
        # Add overview
        for i, example in enumerate(sorted_examples, 1):
            difficulty_stars = "⭐" * example.difficulty
            tutorial.append(f"{i}. **{example.title}** {difficulty_stars}")
            
        tutorial.extend(["", "---", ""])
        
        # Add detailed tutorials
        for i, example in enumerate(sorted_examples, 1):
            tutorial.extend([
                f"## Step {i}: {example.title}",
                "",
                f"**Difficulty:** {'⭐' * example.difficulty}",
                ""
            ])
            
            # Prerequisites
            if example.prerequisites:
                tutorial.extend([
                    "**Prerequisites:**",
                    ""
                ])
                for prereq in example.prerequisites:
                    tutorial.append(f"- {prereq}")
                tutorial.append("")
                
            # Description and code
            tutorial.extend([
                example.description,
                "",
                f"```{example.language.value}",
                example.code,
                "```",
                ""
            ])
            
            # Notes
            if example.notes:
                tutorial.extend([
                    "**Notes:**",
                    ""
                ])
                for note in example.notes:
                    tutorial.append(f"💡 {note}")
                tutorial.append("")
                
            tutorial.extend(["---", ""])
            
        return "\n".join(tutorial)
        
    def generate_cookbook(self, 
                         cookbook_name: str,
                         recipes: List[CodeExample]) -> str:
        """Generate a cookbook with practical recipes."""
        cookbook = [
            f"# {cookbook_name} Cookbook",
            "",
            "Practical recipes for common tasks and patterns.",
            "",
            "## Recipe Index",
            ""
        ]
        
        # Group by category/tags
        categorized = self._group_examples_by_tags(recipes)
        
        # Add index
        for category, examples in categorized.items():
            cookbook.extend([
                f"### {category.title()}",
                ""
            ])
            
            for example in examples:
                cookbook.append(f"- [{example.title}](#{self._to_anchor(example.title)})")
                
            cookbook.append("")
            
        cookbook.extend(["---", ""])
        
        # Add recipes
        for category, examples in categorized.items():
            cookbook.extend([
                f"## {category.title()} Recipes",
                ""
            ])
            
            for example in examples:
                cookbook.extend([
                    f"### {example.title}",
                    "",
                    f"**Use Case:** {example.description}",
                    ""
                ])
                
                # Tags
                if example.tags:
                    tag_list = " | ".join([f"`{tag}`" for tag in example.tags])
                    cookbook.extend([
                        f"**Tags:** {tag_list}",
                        ""
                    ])
                    
                cookbook.extend([
                    f"```{example.language.value}",
                    example.code,
                    "```",
                    ""
                ])
                
                # Related examples
                if example.related_examples:
                    cookbook.extend([
                        "**See Also:**",
                        ""
                    ])
                    for related in example.related_examples:
                        cookbook.append(f"- {related}")
                    cookbook.append("")
                    
                cookbook.extend(["---", ""])
                
        return "\n".join(cookbook)
        
    def extract_examples_from_code(self, file_path: str) -> List[CodeExample]:
        """Extract examples from existing code files."""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse Python AST
            if file_path.endswith('.py'):
                examples.extend(self._extract_python_examples(content, file_path))
            elif file_path.endswith(('.md', '.rst')):
                examples.extend(self._extract_markdown_examples(content, file_path))
                
        except Exception as e:
            logger.error(f"Error extracting examples from {file_path}: {e}")
            
        return examples
        
    def _extract_python_examples(self, content: str, file_path: str) -> List[CodeExample]:
        """Extract examples from Python code."""
        examples = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('example_') or 'example' in node.name.lower():
                        docstring = ast.get_docstring(node)
                        if docstring:
                            code = ast.get_source_segment(content, node)
                            
                            example = CodeExample(
                                title=node.name.replace('_', ' ').title(),
                                description=docstring.split('\n')[0],
                                code=code or '',
                                language=CodeLanguage.PYTHON,
                                example_type=ExampleType.BASIC,
                                tags=[Path(file_path).stem]
                            )
                            examples.append(example)
                            
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
            
        return examples
        
    def _extract_markdown_examples(self, content: str, file_path: str) -> List[CodeExample]:
        """Extract code blocks from markdown files."""
        examples = []
        
        # Find code blocks
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.finditer(code_pattern, content, re.DOTALL)
        
        for i, match in enumerate(matches):
            language_str = match.group(1) or 'python'
            code = match.group(2).strip()
            
            try:
                language = CodeLanguage(language_str.lower())
            except ValueError:
                language = CodeLanguage.PYTHON
                
            # Try to find preceding description
            before_code = content[:match.start()].split('\n')[-5:]
            description = ' '.join(before_code).strip()[:100] + "..."
            
            example = CodeExample(
                title=f"Example {i+1}",
                description=description,
                code=code,
                language=language,
                example_type=ExampleType.BASIC,
                tags=[Path(file_path).stem]
            )
            examples.append(example)
            
        return examples
        
    def _group_examples_by_tags(self, examples: List[CodeExample]) -> Dict[str, List[CodeExample]]:
        """Group examples by their tags."""
        grouped = {}
        
        for example in examples:
            if not example.tags:
                category = "general"
            else:
                category = example.tags[0]
                
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(example)
            
        return grouped
        
    def _generate_example_id(self, title: str) -> str:
        """Generate unique ID for example."""
        return re.sub(r'[^a-zA-Z0-9]', '_', title.lower())
        
    def _generate_collection_id(self, name: str) -> str:
        """Generate unique ID for collection."""
        return re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        
    def _to_anchor(self, text: str) -> str:
        """Convert text to markdown anchor."""
        return re.sub(r'[^a-zA-Z0-9]', '-', text.lower()).strip('-')
        
    def _get_basic_template(self) -> str:
        """Get basic example template."""
        return """# {title}

{description}

## Code

```{language}
{code}
```

{output_section}

{notes_section}
"""

    def _get_tutorial_template(self) -> str:
        """Get tutorial template."""
        return """# Tutorial: {title}

**Difficulty:** {difficulty_stars}

{prerequisites_section}

## Overview

{description}

## Implementation

```{language}
{code}
```

## Explanation

{explanation}

{next_steps}
"""

    def _get_quickstart_template(self) -> str:
        """Get quickstart template."""
        return """# Quick Start: {title}

{description}

## Installation

```bash
pip install {package}
```

## Basic Usage

```{language}
{code}
```

## Next Steps

{next_steps}
"""

    def export_collection(self, collection_id: str, output_path: str) -> None:
        """Export collection to file."""
        if collection_id not in self.collections:
            logger.error(f"Collection {collection_id} not found")
            return
            
        collection = self.collections[collection_id]
        
        # Generate appropriate format based on collection type
        if collection.category == "tutorial":
            content = self.generate_tutorial_series(collection.name, collection.examples)
        elif collection.category == "cookbook":
            content = self.generate_cookbook(collection.name, collection.examples)
        else:
            content = self.generate_quickstart_guide(collection.name, [], collection.examples)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Exported collection {collection_id} to {output_path}")