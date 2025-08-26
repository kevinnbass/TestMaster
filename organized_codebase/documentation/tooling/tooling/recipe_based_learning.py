"""
Recipe-Based Learning System

Creates progressive complexity documentation with tutorial recipes and 
concept-driven organization based on PhiData's cookbook approach.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RecipeType(Enum):
    """Types of recipe documentation."""
    TUTORIAL = "tutorial"
    CONCEPT = "concept" 
    EXAMPLE = "example"
    WORKFLOW = "workflow"
    REFERENCE = "reference"
    TROUBLESHOOTING = "troubleshooting"


class ComplexityLevel(Enum):
    """Recipe complexity levels."""
    BEGINNER = 1
    BASIC = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


class LearningPath(Enum):
    """Different learning paths through recipes."""
    GETTING_STARTED = "getting_started"
    AGENT_CONCEPTS = "agent_concepts"
    REAL_WORLD_EXAMPLES = "real_world_examples"
    ADVANCED_WORKFLOWS = "advanced_workflows"
    TOOL_INTEGRATION = "tool_integration"
    TEAM_COLLABORATION = "team_collaboration"


@dataclass
class Recipe:
    """Represents a documentation recipe."""
    title: str
    recipe_type: RecipeType
    complexity: ComplexityLevel
    learning_path: LearningPath
    description: str
    prerequisites: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    code_example: str = ""
    example_prompts: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    concepts_introduced: List[str] = field(default_factory=list)
    related_recipes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    estimated_time: str = "15 minutes"
    dependencies: List[str] = field(default_factory=list)
    troubleshooting_tips: List[str] = field(default_factory=list)


@dataclass
class LearningSequence:
    """Represents a sequence of recipes for learning."""
    name: str
    path: LearningPath
    description: str
    recipes: List[Recipe] = field(default_factory=list)
    estimated_total_time: str = ""
    completion_criteria: List[str] = field(default_factory=list)
    final_project: Optional[str] = None


@dataclass
class ConceptMap:
    """Maps concepts across recipes."""
    concept_name: str
    introduction_recipe: str
    reinforcement_recipes: List[str] = field(default_factory=list)
    advanced_applications: List[str] = field(default_factory=list)
    difficulty_progression: List[int] = field(default_factory=list)


class RecipeBasedLearning:
    """
    Recipe-based learning system inspired by PhiData's progressive
    cookbook approach with tutorial recipes and concept-driven organization.
    """
    
    def __init__(self, cookbook_dir: str = "cookbook"):
        """Initialize recipe-based learning system."""
        self.cookbook_dir = Path(cookbook_dir)
        self.recipes = []
        self.learning_sequences = []
        self.concept_maps = {}
        self.recipe_templates = self._load_recipe_templates()
        logger.info(f"Recipe-based learning system initialized at {cookbook_dir}")
        
    def create_recipe(self,
                     title: str,
                     recipe_type: RecipeType,
                     complexity: ComplexityLevel,
                     learning_path: LearningPath,
                     description: str,
                     **kwargs) -> Recipe:
        """Create a new recipe."""
        recipe = Recipe(
            title=title,
            recipe_type=recipe_type,
            complexity=complexity,
            learning_path=learning_path,
            description=description,
            **kwargs
        )
        
        self.recipes.append(recipe)
        logger.info(f"Created recipe: {title} ({recipe_type.value}, Level {complexity.value})")
        return recipe
        
    def create_learning_sequence(self,
                               name: str,
                               path: LearningPath,
                               description: str,
                               recipes: List[Recipe]) -> LearningSequence:
        """Create a learning sequence from recipes."""
        # Sort recipes by complexity
        sorted_recipes = sorted(recipes, key=lambda r: r.complexity.value)
        
        # Calculate total time
        total_minutes = 0
        for recipe in sorted_recipes:
            time_str = recipe.estimated_time
            minutes = int(re.search(r'(\d+)', time_str).group(1)) if re.search(r'(\d+)', time_str) else 15
            total_minutes += minutes
            
        hours = total_minutes // 60
        minutes = total_minutes % 60
        estimated_time = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        sequence = LearningSequence(
            name=name,
            path=path,
            description=description,
            recipes=sorted_recipes,
            estimated_total_time=estimated_time
        )
        
        self.learning_sequences.append(sequence)
        logger.info(f"Created learning sequence: {name}")
        return sequence
        
    def generate_recipe_documentation(self, recipe: Recipe) -> str:
        """Generate complete documentation for a recipe."""
        doc = [
            f'"""',
            f"{recipe.title}",
            "",
            recipe.description,
            ""
        ]
        
        # Add capabilities
        if recipe.capabilities:
            doc.extend([
                "This recipe demonstrates:",
            ])
            for capability in recipe.capabilities:
                doc.append(f"- {capability}")
            doc.append("")
            
        # Add prerequisites
        if recipe.prerequisites:
            doc.extend([
                "Prerequisites:",
            ])
            for prereq in recipe.prerequisites:
                doc.append(f"- {prereq}")
            doc.append("")
            
        # Add example usage
        if recipe.example_prompts:
            doc.extend([
                "Example usage:",
            ])
            for prompt in recipe.example_prompts[:2]:  # Show first 2 prompts
                doc.append(f'- "{prompt}"')
            doc.append("")
            
        doc.extend([
            f'Complexity: {"⭐" * recipe.complexity.value}',
            f"Estimated time: {recipe.estimated_time}",
            '"""',
            ""
        ])
        
        # Add dependencies
        if recipe.dependencies:
            doc.extend([
                "# Dependencies",
                "# Install required packages:",
            ])
            for dep in recipe.dependencies:
                doc.append(f"# pip install {dep}")
            doc.append("")
            
        # Add imports and code
        if recipe.code_example:
            doc.extend([
                recipe.code_example,
                ""
            ])
            
        # Add example usage section
        if recipe.example_prompts:
            doc.extend([
                "# Example Usage",
                "if __name__ == '__main__':",
                "    # Test the implementation with different prompts",
                ""
            ])
            
            for i, prompt in enumerate(recipe.example_prompts, 1):
                doc.extend([
                    f"    # Example {i}:",
                    f'    print("Testing: {prompt}")',
                    f"    # result = your_function('{prompt}')",
                    f"    # print(result)",
                    ""
                ])
                
        # Add troubleshooting
        if recipe.troubleshooting_tips:
            doc.extend([
                "# Troubleshooting Tips:",
            ])
            for tip in recipe.troubleshooting_tips:
                doc.append(f"# - {tip}")
                
        return "\n".join(doc)
        
    def generate_cookbook_index(self) -> str:
        """Generate main cookbook index with learning paths."""
        index = [
            "# Cookbook Documentation",
            "",
            "Progressive recipe-based learning for mastering the system.",
            "",
            "## 🚀 Quick Start",
            "",
            "New to the system? Start here:",
            ""
        ]
        
        # Find getting started recipes
        getting_started = [r for r in self.recipes if r.learning_path == LearningPath.GETTING_STARTED]
        getting_started.sort(key=lambda r: r.complexity.value)
        
        for recipe in getting_started[:3]:  # Show first 3
            complexity_stars = "⭐" * recipe.complexity.value
            index.append(f"1. [{recipe.title}](#{self._to_anchor(recipe.title)}) {complexity_stars}")
            
        index.extend([
            "",
            "## 📚 Learning Paths",
            "",
            "Choose your learning journey:",
            ""
        ])
        
        # Group recipes by learning path
        by_path = {}
        for recipe in self.recipes:
            if recipe.learning_path not in by_path:
                by_path[recipe.learning_path] = []
            by_path[recipe.learning_path].append(recipe)
            
        # Generate path sections
        path_descriptions = {
            LearningPath.GETTING_STARTED: "Perfect for beginners - learn the basics step by step",
            LearningPath.AGENT_CONCEPTS: "Deep dive into core concepts and patterns",
            LearningPath.REAL_WORLD_EXAMPLES: "Practical applications and use cases",
            LearningPath.ADVANCED_WORKFLOWS: "Complex orchestration and coordination",
            LearningPath.TOOL_INTEGRATION: "Integrate with external tools and services",
            LearningPath.TEAM_COLLABORATION: "Multi-agent coordination and teamwork"
        }
        
        for path, recipes in by_path.items():
            if not recipes:
                continue
                
            recipes.sort(key=lambda r: r.complexity.value)
            path_name = path.value.replace('_', ' ').title()
            description = path_descriptions.get(path, "")
            
            index.extend([
                f"### {path_name}",
                "",
                description,
                "",
                f"**Recipes ({len(recipes)}):**",
                ""
            ])
            
            for recipe in recipes:
                complexity_stars = "⭐" * recipe.complexity.value
                index.append(f"- [{recipe.title}](#{self._to_anchor(recipe.title)}) {complexity_stars} ({recipe.estimated_time})")
                
            index.append("")
            
        # Add concepts section
        index.extend([
            "## 🧠 Key Concepts",
            "",
            "Core concepts covered across recipes:",
            ""
        ])
        
        # Extract all unique concepts
        all_concepts = set()
        for recipe in self.recipes:
            all_concepts.update(recipe.concepts_introduced)
            
        for concept in sorted(all_concepts):
            recipes_with_concept = [r for r in self.recipes if concept in r.concepts_introduced]
            index.append(f"- **{concept}** ({len(recipes_with_concept)} recipes)")
            
        return "\n".join(index)
        
    def generate_learning_sequence_guide(self, sequence: LearningSequence) -> str:
        """Generate guide for a learning sequence."""
        guide = [
            f"# {sequence.name} Learning Path",
            "",
            sequence.description,
            "",
            f"**Total Time:** {sequence.estimated_total_time}",
            f"**Recipes:** {len(sequence.recipes)}",
            ""
        ]
        
        if sequence.completion_criteria:
            guide.extend([
                "## Completion Criteria",
                ""
            ])
            for criteria in sequence.completion_criteria:
                guide.append(f"- {criteria}")
            guide.append("")
            
        guide.extend([
            "## Learning Progression",
            ""
        ])
        
        for i, recipe in enumerate(sequence.recipes, 1):
            complexity_stars = "⭐" * recipe.complexity.value
            
            guide.extend([
                f"### Step {i}: {recipe.title}",
                "",
                f"**Complexity:** {complexity_stars}",
                f"**Time:** {recipe.estimated_time}",
                f"**Type:** {recipe.recipe_type.value.title()}",
                "",
                recipe.description,
                ""
            ])
            
            if recipe.concepts_introduced:
                guide.extend([
                    "**New Concepts:**",
                ])
                for concept in recipe.concepts_introduced:
                    guide.append(f"- {concept}")
                guide.append("")
                
            if recipe.prerequisites and i == 1:  # Show prereqs for first recipe only
                guide.extend([
                    "**Prerequisites:**",
                ])
                for prereq in recipe.prerequisites:
                    guide.append(f"- {prereq}")
                guide.append("")
                
        if sequence.final_project:
            guide.extend([
                "## Final Project",
                "",
                sequence.final_project,
                ""
            ])
            
        return "\n".join(guide)
        
    def create_concept_progression_map(self, concept: str) -> ConceptMap:
        """Create a concept progression map across recipes."""
        # Find recipes that introduce this concept
        intro_recipes = [r for r in self.recipes if concept in r.concepts_introduced]
        intro_recipe = min(intro_recipes, key=lambda r: r.complexity.value) if intro_recipes else None
        
        # Find recipes that use this concept
        all_concept_recipes = [r for r in self.recipes if concept in r.concepts_introduced or concept.lower() in r.description.lower()]
        
        # Sort by complexity
        all_concept_recipes.sort(key=lambda r: r.complexity.value)
        
        concept_map = ConceptMap(
            concept_name=concept,
            introduction_recipe=intro_recipe.title if intro_recipe else "",
            reinforcement_recipes=[r.title for r in all_concept_recipes[1:4]],  # Next 3 recipes
            advanced_applications=[r.title for r in all_concept_recipes if r.complexity.value >= 4],
            difficulty_progression=[r.complexity.value for r in all_concept_recipes]
        )
        
        self.concept_maps[concept] = concept_map
        return concept_map
        
    def generate_recipe_series(self, base_concept: str, num_recipes: int = 5) -> List[Recipe]:
        """Generate a progressive series of recipes for a concept."""
        recipes = []
        
        for i in range(1, num_recipes + 1):
            complexity = ComplexityLevel(min(i, 5))
            
            recipe_title = f"{base_concept} - Part {i:02d}: {self._get_progression_title(base_concept, i)}"
            
            recipe = Recipe(
                title=recipe_title,
                recipe_type=RecipeType.TUTORIAL if i <= 2 else RecipeType.EXAMPLE,
                complexity=complexity,
                learning_path=LearningPath.GETTING_STARTED if i <= 2 else LearningPath.AGENT_CONCEPTS,
                description=f"Learn {base_concept} - {self._get_progression_description(base_concept, i)}",
                concepts_introduced=[f"{base_concept} Level {i}"],
                prerequisites=[f"Complete Part {i-1:02d}"] if i > 1 else [],
                estimated_time=f"{15 + (i * 5)} minutes",
                example_prompts=[
                    f"Create a simple {base_concept.lower()} example",
                    f"Test {base_concept.lower()} with real data",
                    f"Demonstrate {base_concept.lower()} capabilities"
                ]
            )
            
            recipes.append(recipe)
            
        return recipes
        
    def create_default_cookbook_structure(self) -> None:
        """Create default PhiData-style cookbook structure."""
        # Getting Started Series
        getting_started_recipes = [
            Recipe(
                title="01 - Basic Agent Creation",
                recipe_type=RecipeType.TUTORIAL,
                complexity=ComplexityLevel.BEGINNER,
                learning_path=LearningPath.GETTING_STARTED,
                description="Create your first AI agent with basic capabilities",
                capabilities=[
                    "Create a simple conversational agent",
                    "Configure basic agent parameters",
                    "Test agent responses"
                ],
                example_prompts=[
                    "Hello! What can you help me with?",
                    "Tell me about artificial intelligence",
                    "What's the weather like today?"
                ],
                concepts_introduced=["Agent Creation", "Basic Configuration"],
                dependencies=["phi-agent>=1.0.0"],
                estimated_time="10 minutes",
                code_example="""from phi.agent import Agent

# Create a basic agent
agent = Agent(
    name="Assistant",
    description="A helpful AI assistant",
    instructions="Be helpful and concise"
)

# Test the agent
response = agent.run("Hello! What can you help me with?")
print(response)"""
            ),
            Recipe(
                title="02 - Agent with Tools",
                recipe_type=RecipeType.TUTORIAL,
                complexity=ComplexityLevel.BASIC,
                learning_path=LearningPath.GETTING_STARTED,
                description="Enhance your agent with external tools and capabilities",
                capabilities=[
                    "Add tools to expand agent capabilities",
                    "Configure tool parameters",
                    "Handle tool responses"
                ],
                prerequisites=["Complete 01 - Basic Agent Creation"],
                example_prompts=[
                    "Search for information about Python",
                    "Calculate 25 * 47",
                    "Get current date and time"
                ],
                concepts_introduced=["Tool Integration", "External APIs"],
                dependencies=["phi-agent>=1.0.0", "requests"],
                estimated_time="15 minutes"
            ),
            Recipe(
                title="03 - Memory and Context",
                recipe_type=RecipeType.TUTORIAL,
                complexity=ComplexityLevel.BASIC,
                learning_path=LearningPath.GETTING_STARTED,
                description="Add memory capabilities to maintain conversation context",
                capabilities=[
                    "Enable conversation memory",
                    "Maintain context across interactions",
                    "Configure memory settings"
                ],
                prerequisites=["Complete 02 - Agent with Tools"],
                concepts_introduced=["Conversation Memory", "Context Management"],
                estimated_time="15 minutes"
            )
        ]
        
        # Agent Concepts Series
        concept_recipes = [
            Recipe(
                title="Advanced Agent Personalities",
                recipe_type=RecipeType.CONCEPT,
                complexity=ComplexityLevel.INTERMEDIATE,
                learning_path=LearningPath.AGENT_CONCEPTS,
                description="Create agents with distinct personalities and specialized roles",
                capabilities=[
                    "Define agent personalities",
                    "Create role-specific instructions",
                    "Implement consistent behavior patterns"
                ],
                concepts_introduced=["Agent Personalities", "Role Definition"],
                example_prompts=[
                    "Act as a friendly NYC tour guide",
                    "Be a professional financial advisor",
                    "Respond as an expert chef"
                ],
                estimated_time="20 minutes"
            ),
            Recipe(
                title="Multi-Modal Agent Interactions",
                recipe_type=RecipeType.CONCEPT,
                complexity=ComplexityLevel.ADVANCED,
                learning_path=LearningPath.AGENT_CONCEPTS,
                description="Handle text, images, audio, and video in agent interactions",
                capabilities=[
                    "Process multiple input types",
                    "Generate multi-modal responses",
                    "Cross-modal understanding"
                ],
                concepts_introduced=["Multi-Modal Processing", "Cross-Modal AI"],
                estimated_time="25 minutes"
            )
        ]
        
        # Real-world Examples
        example_recipes = [
            Recipe(
                title="Financial Analysis Assistant",
                recipe_type=RecipeType.EXAMPLE,
                complexity=ComplexityLevel.INTERMEDIATE,
                learning_path=LearningPath.REAL_WORLD_EXAMPLES,
                description="Build an agent that analyzes financial data and provides insights",
                capabilities=[
                    "Process financial data",
                    "Generate market analysis",
                    "Provide investment insights"
                ],
                concepts_introduced=["Financial Analysis", "Data Processing"],
                example_prompts=[
                    "Analyze the performance of AAPL stock",
                    "Compare tech stocks vs utility stocks",
                    "What's the current market sentiment?"
                ],
                estimated_time="30 minutes"
            ),
            Recipe(
                title="Content Creation Team",
                recipe_type=RecipeType.EXAMPLE,
                complexity=ComplexityLevel.ADVANCED,
                learning_path=LearningPath.TEAM_COLLABORATION,
                description="Coordinate multiple agents for comprehensive content creation",
                capabilities=[
                    "Multi-agent coordination",
                    "Role-based collaboration",
                    "Content quality assurance"
                ],
                concepts_introduced=["Team Coordination", "Multi-Agent Systems"],
                estimated_time="45 minutes"
            )
        ]
        
        # Add all recipes
        self.recipes.extend(getting_started_recipes)
        self.recipes.extend(concept_recipes)
        self.recipes.extend(example_recipes)
        
        # Create learning sequences
        self.create_learning_sequence(
            "Getting Started Journey",
            LearningPath.GETTING_STARTED,
            "Master the fundamentals of AI agents from scratch",
            getting_started_recipes
        )
        
        self.create_learning_sequence(
            "Advanced Concepts Track", 
            LearningPath.AGENT_CONCEPTS,
            "Deep dive into advanced agent concepts and capabilities",
            concept_recipes
        )
        
    def _get_progression_title(self, concept: str, step: int) -> str:
        """Get title for progression step."""
        titles = {
            1: "Basics",
            2: "Configuration", 
            3: "Integration",
            4: "Advanced Features",
            5: "Production Usage"
        }
        return titles.get(step, f"Step {step}")
        
    def _get_progression_description(self, concept: str, step: int) -> str:
        """Get description for progression step."""
        descriptions = {
            1: f"fundamental concepts and basic usage",
            2: f"configuration options and customization",
            3: f"integration with other systems",
            4: f"advanced features and optimization",
            5: f"production deployment and best practices"
        }
        return descriptions.get(step, f"advanced {concept.lower()} concepts")
        
    def _to_anchor(self, text: str) -> str:
        """Convert text to markdown anchor."""
        return text.lower().replace(' ', '-').replace('_', '-').replace(':', '').replace('(', '').replace(')', '')
        
    def _load_recipe_templates(self) -> Dict[str, str]:
        """Load recipe documentation templates."""
        return {
            "tutorial": '''"""
{title}

{description}

This recipe demonstrates:
{capabilities}

Prerequisites:
{prerequisites}

Example usage:
{example_prompts}

Complexity: {complexity_stars}
Estimated time: {estimated_time}
"""

{code_example}

# Example Usage
if __name__ == '__main__':
    {usage_example}
''',
            "concept": '''"""
{title} - Concept Deep Dive

{description}

Key concepts covered:
{concepts_introduced}

This recipe shows:
{capabilities}

Complexity: {complexity_stars}
Estimated time: {estimated_time}
"""

{code_example}
''',
            "example": '''"""
Real-World Example: {title}

{description}

Use cases:
{capabilities}

Try these scenarios:
{example_prompts}

Complexity: {complexity_stars}
Estimated time: {estimated_time}
"""

{code_example}
'''
        }
        
    def export_cookbook_docs(self, output_dir: str) -> None:
        """Export complete cookbook documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main cookbook index
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_cookbook_index())
            
        # Create directories for each learning path
        for path in LearningPath:
            path_dir = output_path / path.value
            path_dir.mkdir(exist_ok=True)
            
            # Find recipes for this path
            path_recipes = [r for r in self.recipes if r.learning_path == path]
            if not path_recipes:
                continue
                
            path_recipes.sort(key=lambda r: r.complexity.value)
            
            # Generate individual recipe files
            for i, recipe in enumerate(path_recipes, 1):
                filename = f"{i:02d}_{recipe.title.lower().replace(' ', '_').replace('-', '_')}.py"
                filename = re.sub(r'[^\w_.]', '', filename)  # Clean filename
                
                with open(path_dir / filename, 'w', encoding='utf-8') as f:
                    f.write(self.generate_recipe_documentation(recipe))
                    
        # Generate learning sequence guides
        for sequence in self.learning_sequences:
            sequence_file = f"{sequence.name.lower().replace(' ', '_')}_guide.md"
            with open(output_path / sequence_file, 'w', encoding='utf-8') as f:
                f.write(self.generate_learning_sequence_guide(sequence))
                
        logger.info(f"Exported cookbook docs to {output_dir}")