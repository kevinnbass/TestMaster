"""
Multi-Modal Recipe Engine

Creates cross-modal patterns with structured data outputs and team 
collaboration based on PhiData's multi-modal and team-based approaches.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of input/output modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED_DATA = "structured_data"
    CODE = "code"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


class OutputFormat(Enum):
    """Structured output formats."""
    JSON = "json"
    PYDANTIC = "pydantic"
    DATACLASS = "dataclass"
    YAML = "yaml"
    MARKDOWN = "markdown"
    CSV = "csv"
    XML = "xml"


class TeamRole(Enum):
    """Agent team roles."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"


class CollaborationPattern(Enum):
    """Team collaboration patterns."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    ROUND_ROBIN = "round_robin"
    CONSENSUS = "consensus"


@dataclass
class MultiModalInput:
    """Represents multi-modal input specification."""
    modalities: List[ModalityType] = field(default_factory=list)
    input_formats: Dict[ModalityType, List[str]] = field(default_factory=dict)
    processing_requirements: Dict[ModalityType, str] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)
    examples: Dict[ModalityType, Any] = field(default_factory=dict)


@dataclass
class StructuredOutput:
    """Structured output specification."""
    format_type: OutputFormat
    schema: Dict[str, Any] = field(default_factory=dict)
    validation_schema: Optional[str] = None
    example_output: Any = None
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    post_processing: List[str] = field(default_factory=list)


@dataclass
class TeamAgent:
    """Represents an agent in a team."""
    name: str
    role: TeamRole
    specialization: str
    responsibilities: List[str] = field(default_factory=list)
    input_requirements: List[ModalityType] = field(default_factory=list)
    output_format: Optional[OutputFormat] = None
    success_criteria: List[str] = field(default_factory=list)
    coordination_instructions: str = ""
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TeamCollaboration:
    """Defines team collaboration structure."""
    team_name: str
    pattern: CollaborationPattern
    agents: List[TeamAgent] = field(default_factory=list)
    workflow_steps: List[Dict[str, Any]] = field(default_factory=list)
    coordination_rules: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    final_output: Optional[StructuredOutput] = None


@dataclass
class MultiModalRecipe:
    """Complete multi-modal recipe specification."""
    title: str
    description: str
    input_spec: MultiModalInput
    output_spec: StructuredOutput
    team_spec: Optional[TeamCollaboration] = None
    processing_pipeline: List[str] = field(default_factory=list)
    cross_modal_interactions: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    example_scenarios: List[Dict[str, Any]] = field(default_factory=list)


class MultiModalRecipeEngine:
    """
    Multi-modal recipe engine inspired by PhiData's cross-modal patterns
    and team collaboration approaches with structured data outputs.
    """
    
    def __init__(self, recipes_dir: str = "multimodal_recipes"):
        """Initialize multi-modal recipe engine."""
        self.recipes_dir = Path(recipes_dir)
        self.recipes = []
        self.team_templates = {}
        self.output_schemas = {}
        self.modality_handlers = self._setup_modality_handlers()
        logger.info(f"Multi-modal recipe engine initialized at {recipes_dir}")
        
    def create_multimodal_recipe(self,
                               title: str,
                               description: str,
                               input_modalities: List[ModalityType],
                               output_format: OutputFormat,
                               **kwargs) -> MultiModalRecipe:
        """Create a multi-modal recipe."""
        input_spec = MultiModalInput(
            modalities=input_modalities,
            **kwargs.get('input_config', {})
        )
        
        output_spec = StructuredOutput(
            format_type=output_format,
            **kwargs.get('output_config', {})
        )
        
        recipe = MultiModalRecipe(
            title=title,
            description=description,
            input_spec=input_spec,
            output_spec=output_spec,
            **kwargs
        )
        
        self.recipes.append(recipe)
        logger.info(f"Created multi-modal recipe: {title}")
        return recipe
        
    def create_team_collaboration(self,
                                team_name: str,
                                pattern: CollaborationPattern,
                                agents: List[TeamAgent]) -> TeamCollaboration:
        """Create team collaboration specification."""
        collaboration = TeamCollaboration(
            team_name=team_name,
            pattern=pattern,
            agents=agents
        )
        
        self.team_templates[team_name] = collaboration
        logger.info(f"Created team collaboration: {team_name}")
        return collaboration
        
    def generate_multimodal_recipe_code(self, recipe: MultiModalRecipe) -> str:
        """Generate complete code for multi-modal recipe."""
        code = [
            f'"""',
            f'{recipe.title}',
            "",
            recipe.description,
            "",
            f"Input Modalities: {', '.join([m.value for m in recipe.input_spec.modalities])}",
            f"Output Format: {recipe.output_spec.format_type.value.upper()}",
        ]
        
        if recipe.team_spec:
            code.extend([
                f"Team Pattern: {recipe.team_spec.pattern.value.title()}",
                f"Team Size: {len(recipe.team_spec.agents)} agents"
            ])
            
        code.extend([
            '"""',
            "",
            "from phi.agent import Agent",
            "from phi.tools import Tool",
            "from pydantic import BaseModel",
            "from typing import List, Dict, Any, Optional",
            "import json",
            ""
        ])
        
        # Generate Pydantic models for structured output
        if recipe.output_spec.format_type == OutputFormat.PYDANTIC:
            code.extend(self._generate_pydantic_schema(recipe.output_spec))
            
        # Generate multi-modal input handlers
        for modality in recipe.input_spec.modalities:
            if modality != ModalityType.TEXT:
                code.extend(self._generate_modality_handler(modality))
                
        # Generate team agents if team collaboration is specified
        if recipe.team_spec:
            code.extend(self._generate_team_agents_code(recipe.team_spec))
            
        # Generate main processing function
        code.extend(self._generate_main_processing_function(recipe))
        
        # Generate example usage
        code.extend(self._generate_example_usage(recipe))
        
        return "\n".join(code)
        
    def _generate_pydantic_schema(self, output_spec: StructuredOutput) -> List[str]:
        """Generate Pydantic schema for structured output."""
        schema_code = []
        
        if output_spec.schema:
            schema_name = output_spec.schema.get("name", "OutputModel")
            
            schema_code.extend([
                f"class {schema_name}(BaseModel):",
                '    """Structured output model."""',
                ""
            ])
            
            # Add fields
            for field_name, field_info in output_spec.schema.get("fields", {}).items():
                field_type = field_info.get("type", "str")
                field_desc = field_info.get("description", "")
                default = field_info.get("default", None)
                
                if default is not None:
                    schema_code.append(f"    {field_name}: {field_type} = {repr(default)}  # {field_desc}")
                else:
                    schema_code.append(f"    {field_name}: {field_type}  # {field_desc}")
                    
            schema_code.append("")
            
        return schema_code
        
    def _generate_modality_handler(self, modality: ModalityType) -> List[str]:
        """Generate handler for specific modality."""
        handlers = {
            ModalityType.IMAGE: [
                "def process_image_input(image_path: str) -> str:",
                '    """Process image input for agent consumption."""',
                "    # Image processing logic here",
                '    return f"Processed image from {image_path}"',
                ""
            ],
            ModalityType.AUDIO: [
                "def process_audio_input(audio_path: str) -> str:",
                '    """Process audio input for agent consumption."""',
                "    # Audio processing logic here", 
                '    return f"Processed audio from {audio_path}"',
                ""
            ],
            ModalityType.VIDEO: [
                "def process_video_input(video_path: str) -> str:",
                '    """Process video input for agent consumption."""',
                "    # Video processing logic here",
                '    return f"Processed video from {video_path}"',
                ""
            ]
        }
        
        return handlers.get(modality, [])
        
    def _generate_team_agents_code(self, team_spec: TeamCollaboration) -> List[str]:
        """Generate code for team agents."""
        team_code = [
            f"# {team_spec.team_name} Team Setup",
            f"# Collaboration Pattern: {team_spec.pattern.value.title()}",
            ""
        ]
        
        for agent in team_spec.agents:
            agent_code = [
                f"# {agent.name} - {agent.role.value.title()}",
                f"{agent.name.lower().replace(' ', '_')} = Agent(",
                f'    name="{agent.name}",',
                f'    description="{agent.specialization}",',
                '    instructions="""',
                f'    You are {agent.name}, a {agent.role.value} in the team.',
                f'    Your specialization: {agent.specialization}',
                "",
                "    Responsibilities:",
            ]
            
            for responsibility in agent.responsibilities:
                agent_code.append(f"    - {responsibility}")
                
            if agent.success_criteria:
                agent_code.extend([
                    "",
                    "    Success criteria:",
                ])
                for criteria in agent.success_criteria:
                    agent_code.append(f"    - {criteria}")
                    
            if agent.coordination_instructions:
                agent_code.extend([
                    "",
                    f"    Coordination: {agent.coordination_instructions}",
                ])
                
            agent_code.extend([
                '    """,',
                ")",
                ""
            ])
            
            team_code.extend(agent_code)
            
        return team_code
        
    def _generate_main_processing_function(self, recipe: MultiModalRecipe) -> List[str]:
        """Generate main processing function."""
        func_code = [
            f"def process_{recipe.title.lower().replace(' ', '_')}(",
        ]
        
        # Add parameters based on input modalities
        params = []
        for modality in recipe.input_spec.modalities:
            if modality == ModalityType.TEXT:
                params.append("text_input: str")
            elif modality == ModalityType.IMAGE:
                params.append("image_path: str")
            elif modality == ModalityType.AUDIO:
                params.append("audio_path: str")
            elif modality == ModalityType.VIDEO:
                params.append("video_path: str")
                
        func_code[0] += ", ".join(params) + "):"
        
        func_code.extend([
            f'    """Process {recipe.title.lower()} with multi-modal inputs."""',
            ""
        ])
        
        # Process each modality
        for modality in recipe.input_spec.modalities:
            if modality == ModalityType.IMAGE:
                func_code.append("    image_data = process_image_input(image_path)")
            elif modality == ModalityType.AUDIO:
                func_code.append("    audio_data = process_audio_input(audio_path)")
            elif modality == ModalityType.VIDEO:
                func_code.append("    video_data = process_video_input(video_path)")
                
        func_code.append("")
        
        # Team processing or single agent
        if recipe.team_spec:
            func_code.extend([
                f"    # {recipe.team_spec.pattern.value.title()} team processing",
                "    results = []",
                ""
            ])
            
            for agent in recipe.team_spec.agents:
                agent_var = agent.name.lower().replace(' ', '_')
                func_code.extend([
                    f"    # {agent.name} processing",
                    f"    {agent_var}_result = {agent_var}.run(",
                    "        f\"Process the input: {text_input}\"",
                    "    )",
                    f"    results.append(({agent.name!r}, {agent_var}_result))",
                    ""
                ])
        else:
            func_code.extend([
                "    # Single agent processing",
                "    result = agent.run(text_input)",
                ""
            ])
            
        # Format output
        if recipe.output_spec.format_type == OutputFormat.JSON:
            func_code.extend([
                "    # Format as JSON",
                "    output = {",
                '        "input_modalities": [m.value for m in recipe.input_spec.modalities],',
                '        "results": results if recipe.team_spec else result,',
                '        "timestamp": datetime.now().isoformat()',
                "    }",
                "    return json.dumps(output, indent=2)",
            ])
        else:
            func_code.append("    return results if recipe.team_spec else result")
            
        func_code.append("")
        return func_code
        
    def _generate_example_usage(self, recipe: MultiModalRecipe) -> List[str]:
        """Generate example usage code."""
        usage_code = [
            "# Example Usage",
            'if __name__ == "__main__":',
        ]
        
        if recipe.example_scenarios:
            for i, scenario in enumerate(recipe.example_scenarios, 1):
                usage_code.extend([
                    f"    # Example {i}: {scenario.get('title', f'Scenario {i}')}",
                    f"    print(\"Testing: {scenario.get('description', '')}\")",
                ])
                
                # Add example function call
                if recipe.input_spec.modalities:
                    params = []
                    for modality in recipe.input_spec.modalities:
                        if modality == ModalityType.TEXT:
                            params.append(f'"{scenario.get("text", "Sample text input")}"')
                        elif modality == ModalityType.IMAGE:
                            params.append(f'"{scenario.get("image_path", "sample_image.jpg")}"')
                        elif modality == ModalityType.AUDIO:
                            params.append(f'"{scenario.get("audio_path", "sample_audio.wav")}"')
                        elif modality == ModalityType.VIDEO:
                            params.append(f'"{scenario.get("video_path", "sample_video.mp4")}"')
                            
                    func_name = f"process_{recipe.title.lower().replace(' ', '_')}"
                    usage_code.extend([
                        f"    result = {func_name}({', '.join(params)})",
                        "    print(f\"Result: {result}\")",
                        ""
                    ])
        else:
            # Default example
            usage_code.extend([
                '    print("Running default example...")',
                "    # Add your test cases here",
                ""
            ])
            
        return usage_code
        
    def create_default_multimodal_recipes(self) -> None:
        """Create default multi-modal recipes based on PhiData patterns."""
        # Image Analysis Recipe
        self.create_multimodal_recipe(
            "Image Analysis Assistant",
            "Analyze images and extract structured information with detailed descriptions",
            [ModalityType.IMAGE, ModalityType.TEXT],
            OutputFormat.PYDANTIC,
            input_config={
                "input_formats": {
                    ModalityType.IMAGE: ["jpg", "png", "gif", "webp"],
                    ModalityType.TEXT: ["string"]
                },
                "processing_requirements": {
                    ModalityType.IMAGE: "Computer vision analysis",
                    ModalityType.TEXT: "Natural language understanding"
                }
            },
            output_config={
                "schema": {
                    "name": "ImageAnalysisResult",
                    "fields": {
                        "description": {"type": "str", "description": "Detailed image description"},
                        "objects_detected": {"type": "List[str]", "description": "List of detected objects"},
                        "scene_analysis": {"type": "str", "description": "Scene context and setting"},
                        "colors": {"type": "List[str]", "description": "Dominant colors"},
                        "sentiment": {"type": "str", "description": "Emotional tone of the image"},
                        "confidence_score": {"type": "float", "description": "Analysis confidence (0-1)"}
                    }
                }
            },
            example_scenarios=[
                {
                    "title": "Product Photo Analysis",
                    "description": "Analyze product images for e-commerce listings",
                    "image_path": "product_photo.jpg",
                    "text": "Analyze this product image for an online store listing"
                },
                {
                    "title": "Scene Understanding",
                    "description": "Understand complex scenes and contexts",
                    "image_path": "street_scene.jpg", 
                    "text": "Describe what's happening in this street scene"
                }
            ]
        )
        
        # Team Research Recipe
        research_team = self.create_team_collaboration(
            "Research Analysis Team",
            CollaborationPattern.SEQUENTIAL,
            [
                TeamAgent(
                    "Web Researcher",
                    TeamRole.RESEARCHER,
                    "Information gathering and source validation",
                    responsibilities=[
                        "Search for relevant information online",
                        "Validate source credibility",
                        "Gather diverse perspectives"
                    ],
                    success_criteria=[
                        "Find at least 5 credible sources",
                        "Cover multiple viewpoints",
                        "Provide source citations"
                    ]
                ),
                TeamAgent(
                    "Data Analyst", 
                    TeamRole.ANALYST,
                    "Statistical analysis and pattern recognition",
                    responsibilities=[
                        "Analyze quantitative data",
                        "Identify trends and patterns",
                        "Provide statistical insights"
                    ],
                    success_criteria=[
                        "Identify key trends",
                        "Provide statistical significance",
                        "Create data visualizations"
                    ]
                ),
                TeamAgent(
                    "Report Writer",
                    TeamRole.WRITER,
                    "Synthesis and professional reporting",
                    responsibilities=[
                        "Synthesize research findings",
                        "Create comprehensive report",
                        "Ensure clarity and flow"
                    ],
                    success_criteria=[
                        "Clear executive summary",
                        "Well-structured findings",
                        "Actionable recommendations"
                    ]
                )
            ]
        )
        
        self.create_multimodal_recipe(
            "Comprehensive Research Report",
            "Multi-agent team conducts thorough research and produces structured reports",
            [ModalityType.TEXT, ModalityType.STRUCTURED_DATA],
            OutputFormat.JSON,
            team_spec=research_team,
            processing_pipeline=[
                "Web research phase",
                "Data analysis phase", 
                "Report synthesis phase",
                "Quality review phase"
            ],
            example_scenarios=[
                {
                    "title": "Market Analysis",
                    "description": "Complete market research for tech startup",
                    "text": "Research the market for AI-powered productivity tools"
                },
                {
                    "title": "Competitive Intelligence",
                    "description": "Analyze competitors in specific industry",
                    "text": "Analyze the competitive landscape for electric vehicles"
                }
            ]
        )
        
    def _setup_modality_handlers(self) -> Dict[ModalityType, str]:
        """Setup modality processing handlers."""
        return {
            ModalityType.IMAGE: "PIL, OpenCV, or specialized image processing",
            ModalityType.AUDIO: "librosa, pydub, or speech recognition APIs",
            ModalityType.VIDEO: "opencv-python, moviepy, or video analysis APIs",
            ModalityType.DOCUMENT: "PyPDF2, docx, or document parsing libraries",
            ModalityType.STRUCTURED_DATA: "pandas, json, or data processing libraries"
        }
        
    def generate_team_collaboration_guide(self, collaboration: TeamCollaboration) -> str:
        """Generate guide for team collaboration pattern."""
        guide = [
            f"# {collaboration.team_name} Collaboration Guide",
            "",
            f"**Pattern:** {collaboration.pattern.value.title()}",
            f"**Team Size:** {len(collaboration.agents)} agents",
            "",
            "## Team Members",
            ""
        ]
        
        for agent in collaboration.agents:
            guide.extend([
                f"### {agent.name} - {agent.role.value.title()}",
                "",
                f"**Specialization:** {agent.specialization}",
                "",
                "**Responsibilities:**",
            ])
            
            for responsibility in agent.responsibilities:
                guide.append(f"- {responsibility}")
                
            if agent.success_criteria:
                guide.extend([
                    "",
                    "**Success Criteria:**",
                ])
                for criteria in agent.success_criteria:
                    guide.append(f"- {criteria}")
                    
            guide.append("")
            
        if collaboration.workflow_steps:
            guide.extend([
                "## Workflow Steps",
                ""
            ])
            
            for i, step in enumerate(collaboration.workflow_steps, 1):
                guide.extend([
                    f"### Step {i}: {step.get('title', f'Step {i}')}",
                    "",
                    f"**Agent:** {step.get('agent', 'TBD')}",
                    f"**Action:** {step.get('action', '')}",
                    f"**Expected Output:** {step.get('output', '')}",
                    ""
                ])
                
        return "\n".join(guide)
        
    def export_multimodal_recipes(self, output_dir: str) -> None:
        """Export all multi-modal recipes."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main index
        index = [
            "# Multi-Modal Recipe Collection",
            "",
            "Advanced recipes for cross-modal AI interactions and team collaboration.",
            "",
            "## Recipe Categories",
            ""
        ]
        
        # Group recipes by modality combinations
        by_modalities = {}
        for recipe in self.recipes:
            key = os.getenv('KEY').join(sorted([m.value for m in recipe.input_spec.modalities]))
            if key not in by_modalities:
                by_modalities[key] = []
            by_modalities[key].append(recipe)
            
        for modalities, recipes in by_modalities.items():
            index.extend([
                f"### {modalities.title()} Processing",
                ""
            ])
            
            for recipe in recipes:
                team_info = f" (Team: {len(recipe.team_spec.agents)} agents)" if recipe.team_spec else ""
                index.append(f"- [{recipe.title}]({recipe.title.lower().replace(' ', '_')}.py){team_info}")
                
            index.append("")
            
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(index))
            
        # Generate individual recipe files
        for recipe in self.recipes:
            filename = f"{recipe.title.lower().replace(' ', '_')}.py"
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                f.write(self.generate_multimodal_recipe_code(recipe))
                
        # Generate team collaboration guides
        teams_dir = output_path / "team_guides"
        teams_dir.mkdir(exist_ok=True)
        
        for team_name, collaboration in self.team_templates.items():
            guide_file = f"{team_name.lower().replace(' ', '_')}_guide.md"
            with open(teams_dir / guide_file, 'w', encoding='utf-8') as f:
                f.write(self.generate_team_collaboration_guide(collaboration))
                
        logger.info(f"Exported multi-modal recipes to {output_dir}")