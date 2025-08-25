"""
Tutorial Generator for Code Examples and Documentation

This module provides comprehensive tutorial generation that analyzes code examples,
test cases, and usage patterns to create step-by-step tutorials and learning materials.
"""

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

from ..core.context_builder import AnalysisContextBuilder
from ..core.llm_integration import LLMIntegration
from ..core.quality_assessor import DocumentationQualityAssessor


@dataclass
class CodeExample:
    """Represents a code example for tutorial use."""
    name: str
    code: str
    file_path: str
    line_range: Tuple[int, int]
    complexity: str  # beginner, intermediate, advanced
    concepts: List[str]
    dependencies: List[str]
    expected_output: Optional[str] = None
    description: str = ""


@dataclass
class TutorialStep:
    """Represents a single step in a tutorial."""
    step_number: int
    title: str
    description: str
    code_examples: List[CodeExample]
    explanation: str
    learning_objectives: List[str]
    prerequisites: List[str]
    exercises: List[str]
    common_mistakes: List[str]
    tips: List[str]


@dataclass
class Tutorial:
    """Complete tutorial representation."""
    title: str
    description: str
    difficulty_level: str  # beginner, intermediate, advanced
    estimated_time: str
    learning_objectives: List[str]
    prerequisites: List[str]
    steps: List[TutorialStep]
    resources: List[str]
    next_steps: List[str]
    tags: List[str]


@dataclass
class TutorialConfig:
    """Configuration for tutorial generation."""
    target_audience: str = "intermediate"  # beginner, intermediate, advanced
    include_exercises: bool = True
    include_troubleshooting: bool = True
    include_best_practices: bool = True
    max_code_length: int = 50  # lines
    format: str = "markdown"  # markdown, restructuredtext, html
    language_style: str = "conversational"  # formal, conversational, technical


class TutorialGenerator:
    """
    Generates comprehensive tutorials from code examples, tests, and usage patterns.
    """
    
    def __init__(self, base_path: Path):
        """
        Initialize the tutorial generator.
        
        Args:
            base_path: Root path of the project to analyze
        """
        self.base_path = Path(base_path)
        self.context_builder = AnalysisContextBuilder(base_path)
        self.llm_integration = LLMIntegration()
        self.quality_assessor = DocumentationQualityAssessor()
        
        # Programming concepts and their indicators
        self.concept_indicators = {
            'classes': ['class ', 'def __init__', 'self.', 'super()', 'inheritance'],
            'functions': ['def ', 'return ', 'lambda ', 'yield'],
            'async': ['async def', 'await ', 'asyncio', 'async with'],
            'error_handling': ['try:', 'except', 'finally:', 'raise', 'Exception'],
            'file_io': ['open(', 'with open', 'read()', 'write()', 'json.'],
            'data_structures': ['list(', 'dict(', 'set(', '[]', '{}', 'tuple('],
            'loops': ['for ', 'while ', 'enumerate', 'range(', 'zip('],
            'conditionals': ['if ', 'elif ', 'else:', 'and ', 'or '],
            'decorators': ['@', 'functools', 'property', 'staticmethod'],
            'context_managers': ['with ', '__enter__', '__exit__'],
            'generators': ['yield', 'generator', 'next('],
            'testing': ['test_', 'assert', 'pytest', 'unittest', 'mock'],
            'web_development': ['flask', 'django', 'requests', 'http', 'api'],
            'data_science': ['pandas', 'numpy', 'matplotlib', 'sklearn'],
            'databases': ['sql', 'sqlite', 'postgresql', 'orm', 'query']
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'beginner': ['print(', 'input(', 'len(', 'str(', 'int(', 'float('],
            'intermediate': ['class ', 'def ', 'import ', 'try:', 'with '],
            'advanced': ['async ', 'metaclass', '__getattr__', 'decorator', 'generator']
        }

    async def generate_tutorial_from_code(
        self,
        topic: str,
        source_files: Optional[List[str]] = None,
        config: Optional[TutorialConfig] = None
    ) -> Tutorial:
        """
        Generate a comprehensive tutorial from code examples.
        
        Args:
            topic: The main topic/concept for the tutorial
            source_files: Optional list of specific files to analyze
            config: Configuration for tutorial generation
            
        Returns:
            Generated tutorial
        """
        if config is None:
            config = TutorialConfig()
        
        # Discover and analyze code examples
        code_examples = await self._discover_code_examples(topic, source_files)
        
        # Organize examples into logical progression
        organized_examples = await self._organize_examples_by_complexity(
            code_examples, config
        )
        
        # Generate tutorial structure
        tutorial_steps = await self._generate_tutorial_steps(
            organized_examples, topic, config
        )
        
        # Create complete tutorial
        tutorial = await self._create_complete_tutorial(
            topic, tutorial_steps, config
        )
        
        # Enhance with LLM-generated content
        enhanced_tutorial = await self._enhance_tutorial_with_llm(
            tutorial, config
        )
        
        return enhanced_tutorial

    async def generate_api_tutorial(
        self,
        module_path: str,
        config: Optional[TutorialConfig] = None
    ) -> Tutorial:
        """
        Generate a tutorial specifically for API usage.
        
        Args:
            module_path: Path to the module/API to document
            config: Configuration for tutorial generation
            
        Returns:
            Generated API tutorial
        """
        if config is None:
            config = TutorialConfig()
        
        # Analyze the module for API patterns
        api_analysis = await self._analyze_api_module(module_path)
        
        # Generate usage examples
        usage_examples = await self._generate_api_usage_examples(api_analysis)
        
        # Create tutorial steps for API usage
        tutorial_steps = await self._create_api_tutorial_steps(
            usage_examples, api_analysis, config
        )
        
        # Create complete tutorial
        tutorial = Tutorial(
            title=f"API Tutorial: {api_analysis['module_name']}",
            description=f"Learn how to use the {api_analysis['module_name']} API",
            difficulty_level=config.target_audience,
            estimated_time="30-45 minutes",
            learning_objectives=[
                f"Understand {api_analysis['module_name']} API structure",
                "Learn common usage patterns",
                "Handle errors and edge cases",
                "Follow best practices"
            ],
            prerequisites=["Basic Python knowledge"],
            steps=tutorial_steps,
            resources=[],
            next_steps=[],
            tags=["api", "tutorial", api_analysis['module_name']]
        )
        
        return await self._enhance_tutorial_with_llm(tutorial, config)

    async def _discover_code_examples(
        self,
        topic: str,
        source_files: Optional[List[str]] = None
    ) -> List[CodeExample]:
        """Discover relevant code examples for the topic."""
        examples = []
        
        if source_files is None:
            # Search all Python files
            source_files = []
            for root, dirs, files in os.walk(self.base_path):
                # Skip hidden and cache directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for file in files:
                    if file.endswith('.py'):
                        source_files.append(str(Path(root) / file))
        
        for file_path in source_files:
            file_examples = await self._extract_examples_from_file(
                file_path, topic
            )
            examples.extend(file_examples)
        
        return examples

    async def _extract_examples_from_file(
        self,
        file_path: str,
        topic: str
    ) -> List[CodeExample]:
        """Extract relevant code examples from a single file."""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            lines = content.split('\n')
            
            # Extract function and class definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Check if this node is relevant to the topic
                    if self._is_relevant_to_topic(node, content, topic):
                        example = await self._create_code_example_from_node(
                            node, lines, file_path, content
                        )
                        if example:
                            examples.append(example)
        
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            pass
        
        return examples

    def _is_relevant_to_topic(self, node: ast.AST, content: str, topic: str) -> bool:
        """Check if an AST node is relevant to the tutorial topic."""
        # Check node name
        if hasattr(node, 'name') and topic.lower() in node.name.lower():
            return True
        
        # Check for topic-related concepts in the code
        start_line = getattr(node, 'lineno', 1) - 1
        end_line = getattr(node, 'end_lineno', start_line + 10)
        
        node_content = '\n'.join(content.split('\n')[start_line:end_line])
        node_content_lower = node_content.lower()
        
        # Check if topic or related concepts appear in the code
        topic_lower = topic.lower()
        if topic_lower in node_content_lower:
            return True
        
        # Check for related concepts
        topic_concepts = self.concept_indicators.get(topic_lower, [])
        for concept in topic_concepts:
            if concept in node_content_lower:
                return True
        
        return False

    async def _create_code_example_from_node(
        self,
        node: ast.AST,
        lines: List[str],
        file_path: str,
        content: str
    ) -> Optional[CodeExample]:
        """Create a code example from an AST node."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'name'):
            return None
        
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line + 20)
        
        # Extract the code
        code_lines = lines[start_line:end_line]
        code = '\n'.join(code_lines)
        
        # Determine complexity
        complexity = self._determine_code_complexity(code)
        
        # Extract concepts
        concepts = self._extract_concepts_from_code(code)
        
        # Extract dependencies
        dependencies = self._extract_dependencies_from_code(code, content)
        
        example = CodeExample(
            name=node.name,
            code=code,
            file_path=file_path,
            line_range=(start_line + 1, end_line),
            complexity=complexity,
            concepts=concepts,
            dependencies=dependencies,
            description=self._extract_docstring_from_node(node)
        )
        
        return example

    def _determine_code_complexity(self, code: str) -> str:
        """Determine the complexity level of code."""
        code_lower = code.lower()
        
        advanced_count = sum(1 for indicator in self.complexity_indicators['advanced']
                           if indicator in code_lower)
        intermediate_count = sum(1 for indicator in self.complexity_indicators['intermediate']
                               if indicator in code_lower)
        beginner_count = sum(1 for indicator in self.complexity_indicators['beginner']
                           if indicator in code_lower)
        
        if advanced_count >= 2:
            return 'advanced'
        elif intermediate_count >= 2:
            return 'intermediate'
        else:
            return 'beginner'

    def _extract_concepts_from_code(self, code: str) -> List[str]:
        """Extract programming concepts from code."""
        concepts = []
        code_lower = code.lower()
        
        for concept, indicators in self.concept_indicators.items():
            if any(indicator in code_lower for indicator in indicators):
                concepts.append(concept)
        
        return concepts

    def _extract_dependencies_from_code(self, code: str, full_content: str) -> List[str]:
        """Extract dependencies/imports from code."""
        dependencies = []
        
        # Extract imports from the code snippet
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        except SyntaxError:
            # Fall back to regex for imports in the full file
            import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$'
            for line in full_content.split('\n'):
                match = re.match(import_pattern, line)
                if match:
                    module = match.group(1) or match.group(2).split(',')[0].strip()
                    if module not in dependencies:
                        dependencies.append(module)
        
        return dependencies

    def _extract_docstring_from_node(self, node: ast.AST) -> str:
        """Extract docstring from an AST node."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and 
                isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                return node.body[0].value.value
        return ""

    async def _organize_examples_by_complexity(
        self,
        examples: List[CodeExample],
        config: TutorialConfig
    ) -> Dict[str, List[CodeExample]]:
        """Organize examples by complexity and concepts."""
        organized = {
            'beginner': [],
            'intermediate': [],
            'advanced': []
        }
        
        for example in examples:
            organized[example.complexity].append(example)
        
        # Sort within each complexity level by concept progression
        for level in organized:
            organized[level] = self._sort_examples_by_concept_progression(
                organized[level]
            )
        
        return organized

    def _sort_examples_by_concept_progression(
        self,
        examples: List[CodeExample]
    ) -> List[CodeExample]:
        """Sort examples by logical concept progression."""
        # Define concept learning order
        concept_order = [
            'functions',
            'data_structures',
            'conditionals',
            'loops',
            'classes',
            'error_handling',
            'file_io',
            'async',
            'decorators',
            'testing',
            'web_development'
        ]
        
        def get_concept_priority(example: CodeExample) -> int:
            for i, concept in enumerate(concept_order):
                if concept in example.concepts:
                    return i
            return len(concept_order)  # Unknown concepts go last
        
        return sorted(examples, key=get_concept_priority)

    async def _generate_tutorial_steps(
        self,
        organized_examples: Dict[str, List[CodeExample]],
        topic: str,
        config: TutorialConfig
    ) -> List[TutorialStep]:
        """Generate tutorial steps from organized examples."""
        steps = []
        step_number = 1
        
        # Start with beginner examples if targeting beginners or intermediate
        if config.target_audience in ['beginner', 'intermediate']:
            for example in organized_examples['beginner'][:3]:  # Limit to 3 examples
                step = await self._create_tutorial_step_from_example(
                    step_number, example, 'beginner', config
                )
                steps.append(step)
                step_number += 1
        
        # Add intermediate examples
        if config.target_audience in ['intermediate', 'advanced']:
            for example in organized_examples['intermediate'][:4]:  # Limit to 4 examples
                step = await self._create_tutorial_step_from_example(
                    step_number, example, 'intermediate', config
                )
                steps.append(step)
                step_number += 1
        
        # Add advanced examples for advanced audience
        if config.target_audience == 'advanced':
            for example in organized_examples['advanced'][:3]:  # Limit to 3 examples
                step = await self._create_tutorial_step_from_example(
                    step_number, example, 'advanced', config
                )
                steps.append(step)
                step_number += 1
        
        return steps

    async def _create_tutorial_step_from_example(
        self,
        step_number: int,
        example: CodeExample,
        level: str,
        config: TutorialConfig
    ) -> TutorialStep:
        """Create a tutorial step from a code example."""
        # Generate appropriate title
        title = f"Step {step_number}: {example.name.replace('_', ' ').title()}"
        
        # Generate learning objectives
        objectives = [
            f"Understand {concept.replace('_', ' ')}" for concept in example.concepts[:3]
        ]
        
        # Generate prerequisites
        prerequisites = []
        if example.dependencies:
            prerequisites.extend([f"Knowledge of {dep}" for dep in example.dependencies[:2]])
        
        # Generate exercises if requested
        exercises = []
        if config.include_exercises:
            exercises = await self._generate_exercises_for_example(example, level)
        
        # Generate common mistakes
        common_mistakes = []
        if config.include_troubleshooting:
            common_mistakes = await self._generate_common_mistakes(example, level)
        
        step = TutorialStep(
            step_number=step_number,
            title=title,
            description=example.description or f"Learn about {example.name}",
            code_examples=[example],
            explanation="",  # Will be filled by LLM
            learning_objectives=objectives,
            prerequisites=prerequisites,
            exercises=exercises,
            common_mistakes=common_mistakes,
            tips=[]  # Will be filled by LLM
        )
        
        return step

    async def _generate_exercises_for_example(
        self,
        example: CodeExample,
        level: str
    ) -> List[str]:
        """Generate practice exercises for a code example."""
        exercises = []
        
        if 'functions' in example.concepts:
            exercises.append("Create a similar function with different parameters")
            exercises.append("Add error handling to the function")
        
        if 'classes' in example.concepts:
            exercises.append("Extend the class with additional methods")
            exercises.append("Create a subclass that inherits from this class")
        
        if 'loops' in example.concepts:
            exercises.append("Modify the loop to handle different data types")
            exercises.append("Optimize the loop for better performance")
        
        return exercises[:3]  # Limit to 3 exercises

    async def _generate_common_mistakes(
        self,
        example: CodeExample,
        level: str
    ) -> List[str]:
        """Generate common mistakes for a code example."""
        mistakes = []
        
        if 'error_handling' not in example.concepts:
            mistakes.append("Forgetting to handle potential exceptions")
        
        if 'functions' in example.concepts and 'return' not in example.code.lower():
            mistakes.append("Not returning values from functions when expected")
        
        if 'loops' in example.concepts:
            mistakes.append("Creating infinite loops or off-by-one errors")
        
        return mistakes

    async def _create_complete_tutorial(
        self,
        topic: str,
        steps: List[TutorialStep],
        config: TutorialConfig
    ) -> Tutorial:
        """Create a complete tutorial from steps."""
        # Estimate time based on complexity and number of steps
        estimated_minutes = len(steps) * 10  # 10 minutes per step
        if config.target_audience == 'beginner':
            estimated_minutes = int(estimated_minutes * 1.5)
        
        estimated_time = f"{estimated_minutes} minutes"
        if estimated_minutes > 60:
            hours = estimated_minutes // 60
            minutes = estimated_minutes % 60
            estimated_time = f"{hours} hour{'s' if hours > 1 else ''}"
            if minutes > 0:
                estimated_time += f" {minutes} minutes"
        
        # Generate overall learning objectives
        all_concepts = set()
        for step in steps:
            for example in step.code_examples:
                all_concepts.update(example.concepts)
        
        learning_objectives = [
            f"Master {concept.replace('_', ' ')}" for concept in list(all_concepts)[:5]
        ]
        
        tutorial = Tutorial(
            title=f"{topic.title()} Tutorial",
            description=f"Comprehensive tutorial covering {topic} concepts and best practices",
            difficulty_level=config.target_audience,
            estimated_time=estimated_time,
            learning_objectives=learning_objectives,
            prerequisites=["Basic Python knowledge"],
            steps=steps,
            resources=[],
            next_steps=[],
            tags=[topic.lower(), "tutorial", config.target_audience]
        )
        
        return tutorial

    async def _enhance_tutorial_with_llm(
        self,
        tutorial: Tutorial,
        config: TutorialConfig
    ) -> Tutorial:
        """Enhance tutorial content using LLM."""
        # Prepare context for LLM
        context = {
            'tutorial': asdict(tutorial),
            'config': asdict(config),
            'target_audience': config.target_audience,
            'language_style': config.language_style
        }
        
        # Enhance each step
        for i, step in enumerate(tutorial.steps):
            enhanced_step = await self._enhance_step_with_llm(step, context)
            tutorial.steps[i] = enhanced_step
        
        # Enhance overall tutorial description
        enhanced_description = await self._enhance_tutorial_description(
            tutorial, context
        )
        tutorial.description = enhanced_description
        
        # Generate resources and next steps
        tutorial.resources = await self._generate_resources(tutorial, context)
        tutorial.next_steps = await self._generate_next_steps(tutorial, context)
        
        return tutorial

    async def _enhance_step_with_llm(
        self,
        step: TutorialStep,
        context: Dict[str, Any]
    ) -> TutorialStep:
        """Enhance a tutorial step using LLM."""
        prompt = f"""
Enhance this tutorial step for a {context['target_audience']} audience:

Step {step.step_number}: {step.title}
Current Description: {step.description}

Code Example:
{step.code_examples[0].code if step.code_examples else 'No code example'}

Please provide:
1. A detailed explanation of the code
2. Key concepts being demonstrated
3. Practical tips for implementation
4. Why this concept is important

Use a {context['language_style']} tone.
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type='tutorial_step',
            context=prompt,
            code='',
            style=context.get('config', {}).get('format', 'markdown')
        )
        
        if response.success:
            # Parse the enhanced content
            content_parts = response.content.split('\n\n')
            if len(content_parts) >= 1:
                step.explanation = content_parts[0]
            if len(content_parts) >= 2:
                step.tips = [tip.strip('- ') for tip in content_parts[1].split('\n') if tip.strip().startswith('-')]
        
        return step

    async def _enhance_tutorial_description(
        self,
        tutorial: Tutorial,
        context: Dict[str, Any]
    ) -> str:
        """Enhance tutorial description using LLM."""
        prompt = f"""
Create an engaging description for this tutorial:

Title: {tutorial.title}
Difficulty: {tutorial.difficulty_level}
Estimated Time: {tutorial.estimated_time}
Number of Steps: {len(tutorial.steps)}

Learning Objectives:
{chr(10).join('- ' + obj for obj in tutorial.learning_objectives)}

Make it appealing to {context['target_audience']} developers and explain what they'll learn.
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type='tutorial_description',
            context=prompt,
            code='',
            style='markdown'
        )
        
        if response.success:
            return response.content
        else:
            return tutorial.description

    async def _generate_resources(
        self,
        tutorial: Tutorial,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate additional resources for the tutorial."""
        resources = [
            "Python Official Documentation",
            "Real Python Tutorials",
            "Python.org Beginner's Guide"
        ]
        
        # Add specific resources based on concepts
        all_concepts = set()
        for step in tutorial.steps:
            for example in step.code_examples:
                all_concepts.update(example.concepts)
        
        if 'web_development' in all_concepts:
            resources.append("Flask Documentation")
            resources.append("Django Tutorial")
        
        if 'testing' in all_concepts:
            resources.append("pytest Documentation")
            resources.append("Python Testing Best Practices")
        
        if 'data_science' in all_concepts:
            resources.append("pandas Documentation")
            resources.append("NumPy Quickstart")
        
        return resources

    async def _generate_next_steps(
        self,
        tutorial: Tutorial,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate next steps for after completing the tutorial."""
        next_steps = []
        
        if tutorial.difficulty_level == 'beginner':
            next_steps = [
                "Try more advanced examples with the concepts you've learned",
                "Build a small project using these techniques",
                "Explore intermediate-level tutorials"
            ]
        elif tutorial.difficulty_level == 'intermediate':
            next_steps = [
                "Apply these concepts to a real-world project",
                "Learn about performance optimization",
                "Explore advanced patterns and best practices"
            ]
        else:  # advanced
            next_steps = [
                "Contribute to open source projects using these techniques",
                "Mentor others learning these concepts",
                "Explore cutting-edge developments in this area"
            ]
        
        return next_steps

    async def _analyze_api_module(self, module_path: str) -> Dict[str, Any]:
        """Analyze a module for API patterns and structure."""
        api_analysis = {
            'module_name': Path(module_path).stem,
            'classes': [],
            'functions': [],
            'public_interface': [],
            'examples': []
        }
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=module_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not node.name.startswith('_'):  # Public class
                        api_analysis['classes'].append(node.name)
                        api_analysis['public_interface'].append(f"class {node.name}")
                
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith('_'):  # Public function
                        api_analysis['functions'].append(node.name)
                        api_analysis['public_interface'].append(f"function {node.name}")
        
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            pass
        
        return api_analysis

    async def _generate_api_usage_examples(
        self,
        api_analysis: Dict[str, Any]
    ) -> List[CodeExample]:
        """Generate usage examples for an API."""
        examples = []
        
        # Generate basic usage examples
        for func_name in api_analysis['functions'][:3]:  # Limit to first 3
            example_code = f"""
# Basic usage of {func_name}
from {api_analysis['module_name']} import {func_name}

result = {func_name}()
print(result)
"""
            
            example = CodeExample(
                name=f"Using {func_name}",
                code=example_code.strip(),
                file_path="example",
                line_range=(1, 5),
                complexity='beginner',
                concepts=['functions'],
                dependencies=[api_analysis['module_name']]
            )
            examples.append(example)
        
        return examples

    async def _create_api_tutorial_steps(
        self,
        usage_examples: List[CodeExample],
        api_analysis: Dict[str, Any],
        config: TutorialConfig
    ) -> List[TutorialStep]:
        """Create tutorial steps specifically for API usage."""
        steps = []
        
        # Step 1: Installation and setup
        setup_step = TutorialStep(
            step_number=1,
            title="Installation and Setup",
            description="Get started with the API",
            code_examples=[],
            explanation="Learn how to install and configure the API",
            learning_objectives=["Install the required packages", "Set up basic configuration"],
            prerequisites=["Python installed"],
            exercises=["Install the package", "Import the module"],
            common_mistakes=["Forgetting to install dependencies"],
            tips=["Use virtual environments"]
        )
        steps.append(setup_step)
        
        # Add usage examples as steps
        for i, example in enumerate(usage_examples, 2):
            step = TutorialStep(
                step_number=i,
                title=example.name,
                description=f"Learn how to use {example.name}",
                code_examples=[example],
                explanation="",  # Will be filled by LLM
                learning_objectives=[f"Master {example.name} usage"],
                prerequisites=["Previous steps completed"],
                exercises=[f"Modify {example.name} parameters", f"Handle {example.name} errors"],
                common_mistakes=[f"Incorrect {example.name} usage"],
                tips=[]
            )
            steps.append(step)
        
        return steps

    def render_tutorial_to_markdown(self, tutorial: Tutorial) -> str:
        """Render a tutorial to Markdown format."""
        md = f"""# {tutorial.title}

{tutorial.description}

**Difficulty Level:** {tutorial.difficulty_level.title()}  
**Estimated Time:** {tutorial.estimated_time}

## Learning Objectives

{chr(10).join('- ' + obj for obj in tutorial.learning_objectives)}

## Prerequisites

{chr(10).join('- ' + prereq for prereq in tutorial.prerequisites)}

---

"""
        
        for step in tutorial.steps:
            md += f"""## {step.title}

{step.description}

"""
            
            if step.explanation:
                md += f"{step.explanation}\n\n"
            
            for example in step.code_examples:
                md += f"""### Code Example

```python
{example.code}
```

"""
            
            if step.tips:
                md += f"""### Tips
{chr(10).join('- ' + tip for tip in step.tips)}

"""
            
            if step.exercises:
                md += f"""### Practice Exercises
{chr(10).join('- ' + exercise for exercise in step.exercises)}

"""
            
            if step.common_mistakes:
                md += f"""### Common Mistakes to Avoid
{chr(10).join('- ' + mistake for mistake in step.common_mistakes)}

"""
            
            md += "---\n\n"
        
        if tutorial.resources:
            md += f"""## Additional Resources

{chr(10).join('- ' + resource for resource in tutorial.resources)}

"""
        
        if tutorial.next_steps:
            md += f"""## Next Steps

{chr(10).join('- ' + step for step in tutorial.next_steps)}

"""
        
        md += f"""---

*Tutorial generated on {datetime.now().strftime('%Y-%m-%d')}*
"""
        
        return md