"""
Tutorial System Module
Creates interactive tutorials and learning paths from documentation patterns
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import yaml


class DifficultyLevel(Enum):
    """Tutorial difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class StepType(Enum):
    """Tutorial step types"""
    EXPLANATION = "explanation"
    CODE_EXAMPLE = "code_example"
    INTERACTIVE = "interactive"
    EXERCISE = "exercise"
    QUIZ = "quiz"
    CHECKPOINT = "checkpoint"


@dataclass
class TutorialStep:
    """Represents a single tutorial step"""
    id: str
    title: str
    type: StepType
    content: str
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    interactive_elements: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    estimated_time: int = 5  # minutes
    verification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tutorial:
    """Represents a complete tutorial"""
    id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    category: str
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    estimated_duration: int = 30  # minutes
    steps: List[TutorialStep] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class LearningPath:
    """Represents a structured learning path"""
    id: str
    title: str
    description: str
    target_audience: str
    tutorials: List[str] = field(default_factory=list)  # Tutorial IDs
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration: int = 0  # Total duration in minutes


class TutorialSystem:
    """Creates and manages interactive tutorials"""
    
    def __init__(self):
        self.tutorials = {}
        self.learning_paths = {}
        self.categories = {
            "getting_started": "Getting Started",
            "core_concepts": "Core Concepts", 
            "advanced_features": "Advanced Features",
            "integrations": "Integrations",
            "best_practices": "Best Practices",
            "troubleshooting": "Troubleshooting"
        }
    
    def create_tutorial_from_docs(self, doc_content: str, 
                                 title: str, category: str = "core_concepts",
                                 difficulty: DifficultyLevel = DifficultyLevel.BEGINNER) -> Tutorial:
        """Create tutorial from documentation content"""
        tutorial_id = self.generate_tutorial_id(title)
        
        # Extract sections from documentation
        sections = self.parse_documentation_sections(doc_content)
        
        # Convert sections to tutorial steps
        steps = []
        for i, section in enumerate(sections):
            step = self.create_tutorial_step(
                f"{tutorial_id}_step_{i+1}",
                section["title"],
                section["content"],
                section.get("code_examples", [])
            )
            steps.append(step)
        
        tutorial = Tutorial(
            id=tutorial_id,
            title=title,
            description=self.extract_description(doc_content),
            difficulty=difficulty,
            category=category,
            steps=steps,
            estimated_duration=self.calculate_duration(steps)
        )
        
        # Extract learning objectives
        tutorial.learning_objectives = self.extract_learning_objectives(doc_content)
        
        self.tutorials[tutorial_id] = tutorial
        return tutorial
    
    def create_tutorial_step(self, step_id: str, title: str, 
                           content: str, code_examples: List[Dict] = None) -> TutorialStep:
        """Create a tutorial step from content"""
        step_type = self.determine_step_type(content, code_examples)
        
        step = TutorialStep(
            id=step_id,
            title=title,
            type=step_type,
            content=self.format_step_content(content),
            code_examples=code_examples or []
        )
        
        # Add interactive elements based on type
        if step_type == StepType.CODE_EXAMPLE:
            step.interactive_elements = {
                "playground": True,
                "editable": True,
                "runnable": self.is_runnable_code(code_examples)
            }
        elif step_type == StepType.EXERCISE:
            step.interactive_elements = {
                "task": self.extract_exercise_task(content),
                "hints": self.extract_hints(content),
                "solution": self.extract_solution(content)
            }
        
        step.estimated_time = self.estimate_step_time(content, code_examples)
        
        return step
    
    def determine_step_type(self, content: str, code_examples: List[Dict] = None) -> StepType:
        """Determine the appropriate step type based on content"""
        content_lower = content.lower()
        
        if code_examples and len(code_examples) > 0:
            if any(keyword in content_lower for keyword in ["try", "example", "demo"]):
                return StepType.CODE_EXAMPLE
            elif any(keyword in content_lower for keyword in ["exercise", "practice", "implement"]):
                return StepType.EXERCISE
        
        if any(keyword in content_lower for keyword in ["quiz", "question", "test"]):
            return StepType.QUIZ
        
        if any(keyword in content_lower for keyword in ["checkpoint", "milestone", "summary"]):
            return StepType.CHECKPOINT
        
        if any(keyword in content_lower for keyword in ["interactive", "playground", "sandbox"]):
            return StepType.INTERACTIVE
        
        return StepType.EXPLANATION
    
    def parse_documentation_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse documentation content into sections"""
        sections = []
        current_section = None
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for headers
            if line.startswith('#'):
                if current_section:
                    sections.append(current_section)
                
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('# ').strip()
                current_section = {
                    "title": title,
                    "level": level,
                    "content": "",
                    "code_examples": []
                }
            elif current_section:
                # Check for code blocks
                if line.startswith('```'):
                    # Handle code block
                    code_block = self.extract_code_block(lines, lines.index(line))
                    if code_block:
                        current_section["code_examples"].append(code_block)
                else:
                    current_section["content"] += line + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def extract_code_block(self, lines: List[str], start_index: int) -> Optional[Dict[str, str]]:
        """Extract code block from lines"""
        if start_index >= len(lines):
            return None
        
        start_line = lines[start_index]
        language = start_line.replace('```', '').strip()
        
        code_lines = []
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip().startswith('```'):
                break
            code_lines.append(lines[i])
        
        return {
            "language": language or "text",
            "code": "\n".join(code_lines).strip(),
            "title": f"{language.title()} Example" if language else "Code Example"
        }
    
    def create_interactive_tutorial(self, tutorial_id: str) -> Dict[str, Any]:
        """Create interactive tutorial configuration"""
        if tutorial_id not in self.tutorials:
            return {"error": "Tutorial not found"}
        
        tutorial = self.tutorials[tutorial_id]
        
        interactive_config = {
            "id": tutorial.id,
            "title": tutorial.title,
            "description": tutorial.description,
            "difficulty": tutorial.difficulty.value,
            "category": tutorial.category,
            "estimated_duration": tutorial.estimated_duration,
            "learning_objectives": tutorial.learning_objectives,
            "steps": []
        }
        
        for step in tutorial.steps:
            step_config = {
                "id": step.id,
                "title": step.title,
                "type": step.type.value,
                "content": step.content,
                "estimated_time": step.estimated_time,
                "interactive_elements": step.interactive_elements
            }
            
            # Add code examples with playground support
            if step.code_examples:
                step_config["code_examples"] = []
                for example in step.code_examples:
                    example_config = {
                        "language": example["language"],
                        "code": example["code"],
                        "title": example.get("title", ""),
                        "editable": step.interactive_elements.get("editable", False),
                        "runnable": step.interactive_elements.get("runnable", False)
                    }
                    step_config["code_examples"].append(example_config)
            
            interactive_config["steps"].append(step_config)
        
        return interactive_config
    
    def create_learning_path(self, path_id: str, title: str, description: str,
                           tutorial_ids: List[str], target_audience: str = "developers") -> LearningPath:
        """Create a structured learning path"""
        # Validate tutorial IDs
        valid_tutorials = [tid for tid in tutorial_ids if tid in self.tutorials]
        
        # Calculate total duration
        total_duration = sum(
            self.tutorials[tid].estimated_duration 
            for tid in valid_tutorials
        )
        
        # Create milestones (every 3-4 tutorials)
        milestones = []
        milestone_size = 3
        for i in range(0, len(valid_tutorials), milestone_size):
            milestone_tutorials = valid_tutorials[i:i + milestone_size]
            milestone = {
                "id": f"milestone_{i//milestone_size + 1}",
                "title": f"Milestone {i//milestone_size + 1}",
                "tutorials": milestone_tutorials,
                "description": f"Complete tutorials {i+1}-{min(i+milestone_size, len(valid_tutorials))}"
            }
            milestones.append(milestone)
        
        learning_path = LearningPath(
            id=path_id,
            title=title,
            description=description,
            target_audience=target_audience,
            tutorials=valid_tutorials,
            milestones=milestones,
            estimated_duration=total_duration
        )
        
        self.learning_paths[path_id] = learning_path
        return learning_path
    
    def generate_tutorial_id(self, title: str) -> str:
        """Generate a unique tutorial ID from title"""
        import re
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        return clean_title.lower().replace(' ', '_')
    
    def extract_description(self, content: str) -> str:
        """Extract description from documentation content"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 20:
                return line[:200] + "..." if len(line) > 200 else line
        return "Generated tutorial from documentation"
    
    def extract_learning_objectives(self, content: str) -> List[str]:
        """Extract learning objectives from content"""
        objectives = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ["learn", "understand", "create", "build"]):
                if line.startswith('-') or line.startswith('*'):
                    objectives.append(line.lstrip('-* '))
                elif ':' in line:
                    objectives.append(line.split(':', 1)[1].strip())
        
        return objectives[:5]  # Limit to 5 objectives
    
    def calculate_duration(self, steps: List[TutorialStep]) -> int:
        """Calculate total tutorial duration"""
        return sum(step.estimated_time for step in steps)
    
    def estimate_step_time(self, content: str, code_examples: List[Dict] = None) -> int:
        """Estimate time needed for a tutorial step"""
        base_time = max(2, len(content.split()) // 50)  # ~50 words per minute reading
        
        if code_examples:
            base_time += len(code_examples) * 3  # 3 minutes per code example
        
        return min(base_time, 15)  # Cap at 15 minutes per step
    
    def format_step_content(self, content: str) -> str:
        """Format content for tutorial step"""
        # Clean up content formatting
        lines = [line.strip() for line in content.split('\n')]
        return '\n'.join(line for line in lines if line)
    
    def is_runnable_code(self, code_examples: List[Dict] = None) -> bool:
        """Check if code examples are runnable"""
        if not code_examples:
            return False
        
        runnable_languages = ["python", "javascript", "typescript", "bash", "shell"]
        return any(
            example.get("language", "").lower() in runnable_languages
            for example in code_examples
        )
    
    def extract_exercise_task(self, content: str) -> str:
        """Extract exercise task from content"""
        # Look for task indicators
        task_indicators = ["task:", "exercise:", "try:", "implement:"]
        lines = content.lower().split('\n')
        
        for line in lines:
            for indicator in task_indicators:
                if indicator in line:
                    return line.split(indicator, 1)[1].strip()
        
        return "Complete the following exercise"
    
    def extract_hints(self, content: str) -> List[str]:
        """Extract hints from content"""
        hints = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('hint:'):
                hints.append(line[5:].strip())
            elif line.lower().startswith('tip:'):
                hints.append(line[4:].strip())
        
        return hints
    
    def extract_solution(self, content: str) -> str:
        """Extract solution from content"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.lower().strip().startswith('solution:'):
                # Collect solution content
                solution_lines = []
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith(' '):
                        break
                    solution_lines.append(lines[j])
                return '\n'.join(solution_lines).strip()
        
        return ""
    
    def export_tutorial(self, tutorial_id: str, format: str = "json") -> Optional[str]:
        """Export tutorial in specified format"""
        if tutorial_id not in self.tutorials:
            return None
        
        tutorial = self.tutorials[tutorial_id]
        
        if format.lower() == "json":
            return json.dumps(
                self.create_interactive_tutorial(tutorial_id),
                indent=2
            )
        elif format.lower() in ["yaml", "yml"]:
            return yaml.dump(
                self.create_interactive_tutorial(tutorial_id),
                default_flow_style=False
            )
        
        return None