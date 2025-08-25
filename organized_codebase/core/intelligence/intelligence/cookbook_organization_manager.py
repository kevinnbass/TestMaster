"""
Cookbook Organization Manager

Manages hierarchical recipe organization with personality-driven examples
and consistent voice patterns based on PhiData's cookbook structure.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PersonalityType(Enum):
    """Agent personality types."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EXPERT = "expert"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
    WITTY = "witty"


class VoiceStyle(Enum):
    """Documentation voice styles."""
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    STORYTELLING = "storytelling"
    INSTRUCTIONAL = "instructional"


class CategoryType(Enum):
    """Recipe category types."""
    FUNDAMENTALS = "fundamentals"
    INTEGRATIONS = "integrations"
    WORKFLOWS = "workflows"
    TOOLS = "tools"
    TEAMS = "teams"
    ADVANCED = "advanced"
    TROUBLESHOOTING = "troubleshooting"
    INDUSTRY_SPECIFIC = "industry_specific"


@dataclass
class PersonalityProfile:
    """Defines an agent personality for examples."""
    name: str
    personality_type: PersonalityType
    description: str
    voice_characteristics: List[str] = field(default_factory=list)
    example_phrases: List[str] = field(default_factory=list)
    emoji_style: str = ""
    instruction_template: str = ""
    expertise_areas: List[str] = field(default_factory=list)
    communication_style: str = ""
    typical_responses: List[str] = field(default_factory=list)


@dataclass
class CookbookCategory:
    """Represents a cookbook category with organization rules."""
    name: str
    category_type: CategoryType
    description: str
    voice_style: VoiceStyle
    icon: str = "📚"
    naming_convention: str = ""
    file_patterns: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    progression_order: List[str] = field(default_factory=list)
    metadata_requirements: List[str] = field(default_factory=list)


@dataclass
class RecipeMetadata:
    """Metadata for organizing and categorizing recipes."""
    title: str
    category: str
    difficulty_level: int
    estimated_time: str
    personality_used: Optional[PersonalityProfile] = None
    voice_style: VoiceStyle = VoiceStyle.CONVERSATIONAL
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = ""
    last_updated: str = ""
    tested_versions: List[str] = field(default_factory=list)


@dataclass
class OrganizationHierarchy:
    """Defines the hierarchical organization structure."""
    root_categories: List[CookbookCategory] = field(default_factory=list)
    subcategories: Dict[str, List[CookbookCategory]] = field(default_factory=dict)
    cross_references: Dict[str, List[str]] = field(default_factory=dict)
    navigation_order: List[str] = field(default_factory=list)


class CookbookOrganizationManager:
    """
    Cookbook organization manager inspired by PhiData's hierarchical
    structure with personality-driven examples and consistent voice patterns.
    """
    
    def __init__(self, cookbook_root: str = "cookbook"):
        """Initialize cookbook organization manager."""
        self.cookbook_root = Path(cookbook_root)
        self.personalities = []
        self.categories = []
        self.recipe_metadata = {}
        self.organization_hierarchy = OrganizationHierarchy()
        self.style_guides = {}
        logger.info(f"Cookbook organization manager initialized at {cookbook_root}")
        
    def create_personality_profile(self,
                                 name: str,
                                 personality_type: PersonalityType,
                                 description: str,
                                 **kwargs) -> PersonalityProfile:
        """Create an agent personality profile."""
        profile = PersonalityProfile(
            name=name,
            personality_type=personality_type,
            description=description,
            **kwargs
        )
        
        self.personalities.append(profile)
        logger.info(f"Created personality profile: {name} ({personality_type.value})")
        return profile
        
    def create_cookbook_category(self,
                               name: str,
                               category_type: CategoryType,
                               description: str,
                               voice_style: VoiceStyle,
                               **kwargs) -> CookbookCategory:
        """Create a cookbook category."""
        category = CookbookCategory(
            name=name,
            category_type=category_type,
            description=description,
            voice_style=voice_style,
            **kwargs
        )
        
        self.categories.append(category)
        logger.info(f"Created cookbook category: {name} ({category_type.value})")
        return category
        
    def organize_hierarchy(self, root_categories: List[str], structure: Dict[str, List[str]]) -> None:
        """Organize cookbook into hierarchical structure."""
        # Set root categories
        self.organization_hierarchy.root_categories = [
            cat for cat in self.categories if cat.name in root_categories
        ]
        
        # Set subcategories
        for parent, children in structure.items():
            child_categories = [cat for cat in self.categories if cat.name in children]
            self.organization_hierarchy.subcategories[parent] = child_categories
            
        # Set navigation order
        self.organization_hierarchy.navigation_order = root_categories
        
        logger.info("Organized cookbook hierarchy")
        
    def generate_personality_driven_example(self, personality: PersonalityProfile, scenario: str) -> str:
        """Generate example using specific personality."""
        example = [
            f'"""',
            f'{scenario} - {personality.name} Style',
            "",
            f"This example demonstrates {scenario.lower()} using a {personality.personality_type.value} personality.",
            "",
            f"Personality: {personality.description}",
            ""
        ]
        
        if personality.voice_characteristics:
            example.extend([
                "Voice characteristics:",
            ])
            for char in personality.voice_characteristics:
                example.append(f"- {char}")
            example.append("")
            
        example.extend([
            '"""',
            "",
            "from phi.agent import Agent",
            "",
            f"# Create {personality.name} with distinct personality",
            f'{personality.name.lower()}_agent = Agent(',
            f'    name="{personality.name}",',
            f'    description="{personality.description}",',
        ])
        
        # Add instruction template if available
        if personality.instruction_template:
            example.extend([
                '    instructions="""',
                personality.instruction_template,
                '    """,'
            ])
        elif personality.voice_characteristics:
            example.extend([
                '    instructions="""',
                f'    You are {personality.name}, a {personality.personality_type.value} assistant.',
                f'    {personality.voice_characteristics[0] if personality.voice_characteristics else ""}',
                '    """,'
            ])
            
        example.extend([
            ")",
            "",
            "# Example interactions",
            "if __name__ == '__main__':",
        ])
        
        # Add example phrases/responses
        for i, phrase in enumerate(personality.example_phrases[:3], 1):
            example.extend([
                f'    # Example {i}',
                f'    response = {personality.name.lower()}_agent.run("{phrase}")',
                f'    print(f"{personality.name}: {{response}}")',
                ""
            ])
            
        return "\n".join(example)
        
    def generate_category_index(self, category: CookbookCategory) -> str:
        """Generate index page for cookbook category."""
        index = [
            f"# {category.icon} {category.name}",
            "",
            category.description,
            "",
            f"**Voice Style:** {category.voice_style.value.title()}",
            f"**Category Type:** {category.category_type.value.replace('_', ' ').title()}",
            ""
        ]
        
        if category.prerequisites:
            index.extend([
                "## Prerequisites",
                ""
            ])
            for prereq in category.prerequisites:
                index.append(f"- {prereq}")
            index.append("")
            
        if category.progression_order:
            index.extend([
                "## Learning Progression",
                "",
                "Follow this order for optimal learning:",
                ""
            ])
            for i, recipe in enumerate(category.progression_order, 1):
                index.append(f"{i}. {recipe}")
            index.append("")
            
        # Find recipes in this category
        category_recipes = [
            metadata for metadata in self.recipe_metadata.values()
            if metadata.category == category.name
        ]
        
        if category_recipes:
            index.extend([
                "## Available Recipes",
                ""
            ])
            
            # Group by difficulty
            by_difficulty = {}
            for recipe in category_recipes:
                level = recipe.difficulty_level
                if level not in by_difficulty:
                    by_difficulty[level] = []
                by_difficulty[level].append(recipe)
                
            for level in sorted(by_difficulty.keys()):
                level_name = self._difficulty_name(level)
                index.extend([
                    f"### {level_name} Level",
                    ""
                ])
                
                for recipe in by_difficulty[level]:
                    personality_info = f" (using {recipe.personality_used.name})" if recipe.personality_used else ""
                    index.append(f"- [{recipe.title}]({self._recipe_filename(recipe.title)}){personality_info} - {recipe.estimated_time}")
                    
                index.append("")
                
        return "\n".join(index)
        
    def generate_master_index(self) -> str:
        """Generate master cookbook index."""
        index = [
            "# 📚 Master Cookbook",
            "",
            "Comprehensive collection of recipes organized by category and personality.",
            "",
            "## 🎭 Featured Personalities",
            ""
        ]
        
        # Show personality profiles
        for personality in self.personalities[:6]:  # Show top 6
            index.extend([
                f"### {personality.name} - {personality.personality_type.value.title()}",
                "",
                personality.description,
                "",
                f"**Communication Style:** {personality.communication_style}",
                f"**Expertise:** {', '.join(personality.expertise_areas[:3])}",
                ""
            ])
            
        index.extend([
            "## 📂 Recipe Categories",
            ""
        ])
        
        # Show hierarchy
        for root_category in self.organization_hierarchy.root_categories:
            index.extend([
                f"### {root_category.icon} [{root_category.name}]({root_category.name.lower().replace(' ', '_')}/)",
                "",
                root_category.description,
                ""
            ])
            
            # Show subcategories
            if root_category.name in self.organization_hierarchy.subcategories:
                subcats = self.organization_hierarchy.subcategories[root_category.name]
                for subcat in subcats:
                    index.append(f"- [{subcat.name}]({root_category.name.lower().replace(' ', '_')}/{subcat.name.lower().replace(' ', '_')}/)")
                    
                index.append("")
                
        # Add navigation tips
        index.extend([
            "## 🧭 Navigation Tips",
            "",
            "- **New to the system?** Start with [Fundamentals](fundamentals/)",
            "- **Looking for specific tools?** Browse [Integrations](integrations/)", 
            "- **Want real examples?** Check [Workflows](workflows/)",
            "- **Building teams?** Explore [Teams](teams/)",
            "",
            "## 🎨 Voice Styles",
            ""
        ])
        
        # Document voice styles used
        styles_used = set(cat.voice_style for cat in self.categories)
        for style in styles_used:
            style_categories = [cat.name for cat in self.categories if cat.voice_style == style]
            index.append(f"- **{style.value.title()}**: {', '.join(style_categories)}")
            
        return "\n".join(index)
        
    def create_style_guide(self, voice_style: VoiceStyle) -> str:
        """Create style guide for consistent voice across recipes."""
        guides = {
            VoiceStyle.CONVERSATIONAL: {
                "tone": "Friendly and approachable",
                "structure": "Use questions to engage readers, include personal anecdotes",
                "language": "Simple terms, avoid excessive jargon",
                "examples": ["Let's explore this together", "You might wonder why...", "Here's what I've learned"]
            },
            VoiceStyle.TECHNICAL: {
                "tone": "Precise and authoritative",
                "structure": "Clear steps, detailed explanations, comprehensive coverage",
                "language": "Technical accuracy, proper terminology",
                "examples": ["Implementation requires", "Consider the following parameters", "This approach ensures"]
            },
            VoiceStyle.TUTORIAL: {
                "tone": "Patient and instructional",
                "structure": "Step-by-step progression, frequent checkpoints",
                "language": "Active voice, clear instructions",
                "examples": ["First, we'll", "Next, let's", "Now you can"]
            }
        }
        
        guide_info = guides.get(voice_style, {})
        
        guide = [
            f"# {voice_style.value.title()} Style Guide",
            "",
            "Guidelines for maintaining consistent voice and tone.",
            "",
            f"## Tone",
            "",
            guide_info.get("tone", ""),
            "",
            "## Structure", 
            "",
            guide_info.get("structure", ""),
            "",
            "## Language",
            "",
            guide_info.get("language", ""),
            "",
            "## Example Phrases",
            ""
        ]
        
        for example in guide_info.get("examples", []):
            guide.append(f"- \"{example}\"")
            
        self.style_guides[voice_style] = "\n".join(guide)
        return self.style_guides[voice_style]
        
    def validate_recipe_organization(self, recipe_path: str) -> Dict[str, Any]:
        """Validate recipe follows organization standards."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        
        try:
            with open(recipe_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check file naming convention
            filename = Path(recipe_path).name
            if not self._follows_naming_convention(filename):
                validation_result["warnings"].append("Filename doesn't follow convention")
                
            # Check for personality usage
            if not self._has_personality_definition(content):
                validation_result["suggestions"].append("Consider adding personality-driven example")
                
            # Check documentation structure
            if not content.startswith('"""'):
                validation_result["errors"].append("Missing docstring header")
                validation_result["valid"] = False
                
            # Check for required sections
            required_sections = ["Example usage", "Prerequisites"]
            for section in required_sections:
                if section.lower() not in content.lower():
                    validation_result["warnings"].append(f"Missing recommended section: {section}")
                    
        except Exception as e:
            validation_result["errors"].append(f"Error reading file: {e}")
            validation_result["valid"] = False
            
        return validation_result
        
    def create_default_personalities(self) -> None:
        """Create default personality profiles based on PhiData patterns."""
        # NYC News Reporter
        self.create_personality_profile(
            "NYC News Reporter",
            PersonalityType.ENTHUSIASTIC,
            "Energetic news reporter with NYC flair and insider knowledge",
            voice_characteristics=[
                "Fast-paced and energetic communication",
                "Uses NYC slang and references",
                "Always looking for the story angle",
                "Confident and assertive tone"
            ],
            example_phrases=[
                "What's the latest buzz in the city?",
                "Tell me about the hottest trends",
                "Give me the inside scoop on this topic"
            ],
            emoji_style="🗞️📺🏙️",
            instruction_template="You are a NYC news reporter. Be energetic, use NYC references, and always look for the newsworthy angle. Keep responses lively and informative.",
            expertise_areas=["Current Events", "NYC Culture", "Breaking News"],
            communication_style="Fast-paced, energetic, with NYC personality"
        )
        
        # Thai Cooking Expert
        self.create_personality_profile(
            "Thai Cooking Expert",
            PersonalityType.EXPERT,
            "Master chef specializing in authentic Thai cuisine and techniques",
            voice_characteristics=[
                "Deep knowledge of Thai cooking traditions",
                "Patient and detailed explanations",
                "Focuses on authenticity and technique",
                "Warm and encouraging approach"
            ],
            example_phrases=[
                "Create an authentic Pad Thai recipe",
                "Explain the difference between Thai curry types",
                "What spices are essential for Thai cooking?"
            ],
            emoji_style="🍜🌶️🥥",
            instruction_template="You are a master Thai chef. Share authentic recipes, explain traditional techniques, and emphasize the importance of fresh ingredients and proper preparation methods.",
            expertise_areas=["Thai Cuisine", "Asian Cooking", "Spice Blending"],
            communication_style="Knowledgeable, patient, detail-oriented"
        )
        
        # Elite Research Assistant
        self.create_personality_profile(
            "Elite Research Assistant",
            PersonalityType.ANALYTICAL,
            "PhD-level research specialist with exceptional analytical skills",
            voice_characteristics=[
                "Systematic and thorough approach",
                "Evidence-based reasoning",
                "Academic precision",
                "Objective and analytical"
            ],
            example_phrases=[
                "Conduct comprehensive research on this topic",
                "Analyze the data and provide insights",
                "What are the key findings in recent literature?"
            ],
            emoji_style="🔬📊📚",
            instruction_template="You are an elite research assistant with PhD-level expertise. Provide thorough, evidence-based analysis with proper citations and systematic methodology.",
            expertise_areas=["Research Methods", "Data Analysis", "Academic Writing"],
            communication_style="Academic, precise, systematic"
        )
        
    def create_default_categories(self) -> None:
        """Create default cookbook categories."""
        # Fundamentals
        self.create_cookbook_category(
            "Getting Started",
            CategoryType.FUNDAMENTALS,
            "Essential recipes for beginners to learn the basics",
            VoiceStyle.TUTORIAL,
            icon="🚀",
            naming_convention="01_basic_*, 02_intermediate_*, etc.",
            file_patterns=["*_basic_*.py", "*_starter_*.py"],
            progression_order=[
                "Basic Agent Creation",
                "Agent with Tools", 
                "Memory and Context",
                "Advanced Configuration"
            ]
        )
        
        # Integrations
        self.create_cookbook_category(
            "Tool Integrations",
            CategoryType.INTEGRATIONS,
            "Recipes for integrating with external tools and services",
            VoiceStyle.TECHNICAL,
            icon="🔧",
            naming_convention="tool_name_integration.py",
            file_patterns=["*_integration.py", "*_tool.py"]
        )
        
        # Workflows
        self.create_cookbook_category(
            "Workflows",
            CategoryType.WORKFLOWS, 
            "Complex multi-step processes and orchestration patterns",
            VoiceStyle.CONVERSATIONAL,
            icon="⚡",
            naming_convention="workflow_name_pattern.py",
            file_patterns=["*_workflow.py", "*_process.py"]
        )
        
        # Teams
        self.create_cookbook_category(
            "Team Collaboration",
            CategoryType.TEAMS,
            "Multi-agent coordination and team-based examples",
            VoiceStyle.STORYTELLING,
            icon="👥",
            naming_convention="team_name_collaboration.py",
            file_patterns=["*_team.py", "*_collaboration.py"]
        )
        
    def _follows_naming_convention(self, filename: str) -> bool:
        """Check if filename follows naming convention."""
        # Basic checks for common patterns
        return (filename.endswith('.py') and 
                not filename.startswith('.') and
                '_' in filename)
        
    def _has_personality_definition(self, content: str) -> bool:
        """Check if content includes personality-driven examples."""
        personality_indicators = [
            "personality", "character", "role", "instructions=", "description="
        ]
        return any(indicator in content.lower() for indicator in personality_indicators)
        
    def _difficulty_name(self, level: int) -> str:
        """Convert difficulty level to name."""
        names = {1: "Beginner", 2: "Basic", 3: "Intermediate", 4: "Advanced", 5: "Expert"}
        return names.get(level, f"Level {level}")
        
    def _recipe_filename(self, title: str) -> str:
        """Convert recipe title to filename."""
        return title.lower().replace(' ', '_').replace('-', '_') + '.py'
        
    def export_organized_cookbook(self, output_dir: str) -> None:
        """Export organized cookbook structure."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Master index
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_master_index())
            
        # Create category directories and indexes
        for category in self.categories:
            category_dir = output_path / category.name.lower().replace(' ', '_')
            category_dir.mkdir(exist_ok=True)
            
            # Category index
            with open(category_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(self.generate_category_index(category))
                
        # Personality showcase
        personalities_dir = output_path / "personalities"
        personalities_dir.mkdir(exist_ok=True)
        
        for personality in self.personalities:
            personality_file = f"{personality.name.lower().replace(' ', '_')}_examples.py"
            with open(personalities_dir / personality_file, 'w', encoding='utf-8') as f:
                example_scenario = f"{personality.expertise_areas[0]} Assistant" if personality.expertise_areas else "General Assistant"
                f.write(self.generate_personality_driven_example(personality, example_scenario))
                
        # Style guides
        styles_dir = output_path / "style_guides"
        styles_dir.mkdir(exist_ok=True)
        
        for style in VoiceStyle:
            style_guide = self.create_style_guide(style)
            style_file = f"{style.value}_style_guide.md"
            with open(styles_dir / style_file, 'w', encoding='utf-8') as f:
                f.write(style_guide)
                
        logger.info(f"Exported organized cookbook to {output_dir}")