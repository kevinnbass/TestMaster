"""
Multi-Agent Pattern Documentation

Creates comprehensive documentation for multi-agent system patterns with 
decision matrices and progressive disclosure based on LLama-Agents approach.
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


class AgentPatternType(Enum):
    """Types of multi-agent patterns."""
    WORKFLOW = "workflow"
    ORCHESTRATOR = "orchestrator"
    CUSTOM_PLANNER = "custom_planner"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"


class ComplexityLevel(Enum):
    """Pattern complexity levels."""
    MINIMAL = "minimal"
    SIMPLE = "simple" 
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class UseCase(Enum):
    """Common multi-agent use cases."""
    RESEARCH = "research"
    CONTENT_GENERATION = "content_generation"
    DATA_PROCESSING = "data_processing"
    TASK_AUTOMATION = "task_automation"
    DECISION_MAKING = "decision_making"
    CREATIVE_TASKS = "creative_tasks"


@dataclass
class AgentPattern:
    """Represents a multi-agent pattern."""
    name: str
    pattern_type: AgentPatternType
    complexity: ComplexityLevel
    description: str
    when_to_use: List[str] = field(default_factory=list)
    when_not_to_use: List[str] = field(default_factory=list)
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    use_cases: List[UseCase] = field(default_factory=list)
    code_sketch: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    required_components: List[str] = field(default_factory=list)
    scalability: str = ""
    maintenance: str = ""
    learning_curve: str = ""


@dataclass
class DecisionMatrix:
    """Decision matrix for pattern selection."""
    criteria: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    scores: Dict[str, Dict[str, int]] = field(default_factory=dict)  # pattern -> criteria -> score
    weights: Dict[str, float] = field(default_factory=dict)  # criteria -> weight
    recommendations: Dict[str, str] = field(default_factory=dict)


@dataclass
class PatternComparison:
    """Side-by-side pattern comparison."""
    patterns: List[AgentPattern] = field(default_factory=list)
    comparison_criteria: List[str] = field(default_factory=list)
    use_case_mapping: Dict[UseCase, List[str]] = field(default_factory=dict)


class MultiAgentPatternDocs:
    """
    Multi-agent pattern documentation system inspired by LLama-Agents
    three-tier approach with decision matrices and progressive disclosure.
    """
    
    def __init__(self, docs_dir: str = "multi-agent-patterns"):
        """Initialize multi-agent pattern docs."""
        self.docs_dir = Path(docs_dir)
        self.patterns = []
        self.decision_matrices = []
        self.comparisons = []
        self.pattern_templates = self._load_pattern_templates()
        logger.info(f"Multi-agent pattern docs initialized at {docs_dir}")
        
    def create_agent_pattern(self,
                           name: str,
                           pattern_type: AgentPatternType,
                           complexity: ComplexityLevel,
                           description: str,
                           **kwargs) -> AgentPattern:
        """Create a new agent pattern."""
        pattern = AgentPattern(
            name=name,
            pattern_type=pattern_type,
            complexity=complexity,
            description=description,
            **kwargs
        )
        
        self.patterns.append(pattern)
        logger.info(f"Created agent pattern: {name} ({pattern_type.value})")
        return pattern
        
    def create_decision_matrix(self,
                             criteria: List[str],
                             patterns: List[str],
                             weights: Optional[Dict[str, float]] = None) -> DecisionMatrix:
        """Create decision matrix for pattern selection."""
        matrix = DecisionMatrix(
            criteria=criteria,
            patterns=patterns,
            weights=weights or {criterion: 1.0 for criterion in criteria}
        )
        
        self.decision_matrices.append(matrix)
        logger.info("Created decision matrix for pattern selection")
        return matrix
        
    def add_pattern_scores(self,
                          matrix: DecisionMatrix,
                          pattern_name: str,
                          scores: Dict[str, int]) -> None:
        """Add scores for pattern in decision matrix."""
        if pattern_name not in matrix.patterns:
            matrix.patterns.append(pattern_name)
            
        matrix.scores[pattern_name] = scores
        logger.info(f"Added scores for pattern: {pattern_name}")
        
    def generate_pattern_overview(self) -> str:
        """Generate overview of all patterns."""
        overview = [
            "# Multi-Agent System Patterns",
            "",
            "Choose the right pattern for your multi-agent system based on complexity,",
            "use case, and requirements.",
            "",
            "## Quick Pattern Selection",
            "",
            "| Pattern | Complexity | Best For | When to Use |",
            "|---------|------------|----------|-------------|"
        ]
        
        for pattern in sorted(self.patterns, key=lambda x: list(ComplexityLevel).index(x.complexity)):
            best_for = ", ".join([uc.value.replace("_", " ").title() for uc in pattern.use_cases[:2]])
            when_to_use = pattern.when_to_use[0] if pattern.when_to_use else "General purpose"
            
            overview.append(
                f"| [{pattern.name}](#{self._to_anchor(pattern.name)}) | "
                f"{pattern.complexity.value.title()} | "
                f"{best_for} | "
                f"{when_to_use} |"
            )
            
        overview.extend([
            "",
            "## Pattern Details",
            ""
        ])
        
        # Group patterns by type
        by_type = {}
        for pattern in self.patterns:
            if pattern.pattern_type not in by_type:
                by_type[pattern.pattern_type] = []
            by_type[pattern.pattern_type].append(pattern)
            
        # Generate detailed sections
        for pattern_type, patterns in by_type.items():
            overview.extend([
                f"### {pattern_type.value.title()} Patterns",
                ""
            ])
            
            for pattern in sorted(patterns, key=lambda x: list(ComplexityLevel).index(x.complexity)):
                overview.extend([
                    f"#### {pattern.name}",
                    "",
                    f"**Complexity:** {pattern.complexity.value.title()}",
                    "",
                    pattern.description,
                    ""
                ])
                
                if pattern.advantages:
                    overview.extend([
                        "**Advantages:**",
                        ""
                    ])
                    for advantage in pattern.advantages:
                        overview.append(f"- ✅ {advantage}")
                    overview.append("")
                    
                if pattern.disadvantages:
                    overview.extend([
                        "**Considerations:**",
                        ""
                    ])
                    for disadvantage in pattern.disadvantages:
                        overview.append(f"- ⚠️ {disadvantage}")
                    overview.append("")
                    
        return "\n".join(overview)
        
    def generate_decision_guide(self) -> str:
        """Generate decision guide with matrices."""
        guide = [
            "# Multi-Agent Pattern Decision Guide",
            "",
            "Use this guide to select the right pattern for your use case.",
            "",
            "## Decision Process",
            "",
            "1. **Identify Your Requirements** - What are you trying to achieve?",
            "2. **Assess Complexity** - How complex is your use case?", 
            "3. **Consider Constraints** - What are your technical constraints?",
            "4. **Use Decision Matrix** - Score patterns against your criteria",
            "5. **Start Simple** - Begin with simpler patterns and evolve",
            "",
            "## Use Case Mapping",
            ""
        ]
        
        # Create use case recommendations
        use_case_patterns = {}
        for pattern in self.patterns:
            for use_case in pattern.use_cases:
                if use_case not in use_case_patterns:
                    use_case_patterns[use_case] = []
                use_case_patterns[use_case].append(pattern.name)
                
        for use_case, pattern_names in use_case_patterns.items():
            guide.extend([
                f"### {use_case.value.replace('_', ' ').title()}",
                "",
                f"**Recommended patterns:** {', '.join(pattern_names)}",
                ""
            ])
            
            # Find patterns for this use case
            relevant_patterns = [p for p in self.patterns if use_case in p.use_cases]
            if relevant_patterns:
                best_pattern = min(relevant_patterns, key=lambda x: list(ComplexityLevel).index(x.complexity))
                guide.extend([
                    f"**Start with:** {best_pattern.name} ({best_pattern.complexity.value})",
                    f"**Why:** {best_pattern.description}",
                    ""
                ])
                
        # Add decision matrices
        for matrix in self.decision_matrices:
            guide.extend([
                "## Decision Matrix",
                "",
                "Score each pattern (1-5) against your requirements:",
                "",
                "| Pattern | " + " | ".join(matrix.criteria) + " | Total |",
                "|---------|" + "|".join(["-" * len(c) for c in matrix.criteria]) + "|-------|"
            ])
            
            for pattern_name in matrix.patterns:
                if pattern_name in matrix.scores:
                    scores = matrix.scores[pattern_name]
                    total = sum(scores.get(criterion, 0) * matrix.weights.get(criterion, 1.0) 
                              for criterion in matrix.criteria)
                    row = [pattern_name]
                    for criterion in matrix.criteria:
                        row.append(str(scores.get(criterion, 0)))
                    row.append(f"**{total:.1f}**")
                    guide.append("| " + " | ".join(row) + " |")
                    
            guide.append("")
            
        return "\n".join(guide)
        
    def generate_pattern_documentation(self, pattern: AgentPattern) -> str:
        """Generate comprehensive documentation for single pattern."""
        doc = [
            f"# {pattern.name}",
            "",
            f"**Pattern Type:** {pattern.pattern_type.value.title()}",
            f"**Complexity:** {pattern.complexity.value.title()}",
            "",
            pattern.description,
            "",
            "## When to Use This Pattern",
            ""
        ]
        
        for reason in pattern.when_to_use:
            doc.append(f"✅ {reason}")
            
        if pattern.when_not_to_use:
            doc.extend([
                "",
                "## When NOT to Use This Pattern",
                ""
            ])
            for reason in pattern.when_not_to_use:
                doc.append(f"❌ {reason}")
                
        doc.extend([
            "",
            "## Key Characteristics",
            "",
            f"- **Scalability:** {pattern.scalability or 'Not specified'}",
            f"- **Maintenance:** {pattern.maintenance or 'Not specified'}",
            f"- **Learning Curve:** {pattern.learning_curve or 'Not specified'}",
            ""
        ])
        
        if pattern.required_components:
            doc.extend([
                "## Required Components",
                ""
            ])
            for component in pattern.required_components:
                doc.append(f"- {component}")
            doc.append("")
            
        if pattern.code_sketch:
            doc.extend([
                "## Code Sketch",
                "",
                "Minimal working example:",
                "",
                "```python",
                pattern.code_sketch,
                "```",
                ""
            ])
            
        if pattern.configuration:
            doc.extend([
                "## Configuration",
                "",
                "```yaml",
                yaml.dump(pattern.configuration, default_flow_style=False),
                "```",
                ""
            ])
            
        return "\n".join(doc)
        
    def create_comparative_analysis(self, patterns: List[AgentPattern], criteria: List[str]) -> str:
        """Create side-by-side pattern comparison."""
        comparison = [
            "# Pattern Comparison",
            "",
            "Side-by-side comparison of multi-agent patterns:",
            "",
            "| Aspect | " + " | ".join([p.name for p in patterns]) + " |",
            "|--------|" + "|".join(["-" * len(p.name) for p in patterns]) + "|"
        ]
        
        # Compare different aspects
        aspects = [
            ("Complexity", lambda p: p.complexity.value.title()),
            ("Type", lambda p: p.pattern_type.value.title()),
            ("Best For", lambda p: ", ".join([uc.value.replace("_", " ").title() for uc in p.use_cases[:2]])),
            ("Learning Curve", lambda p: p.learning_curve or "Not specified"),
            ("Scalability", lambda p: p.scalability or "Not specified"),
            ("Maintenance", lambda p: p.maintenance or "Not specified")
        ]
        
        for aspect_name, extractor in aspects:
            row = [aspect_name]
            for pattern in patterns:
                row.append(extractor(pattern))
            comparison.append("| " + " | ".join(row) + " |")
            
        comparison.extend([
            "",
            "## Detailed Comparison",
            ""
        ])
        
        for i, pattern in enumerate(patterns):
            comparison.extend([
                f"### {pattern.name}",
                "",
                pattern.description,
                ""
            ])
            
            if pattern.advantages:
                comparison.extend([
                    "**Pros:**",
                    ""
                ])
                for advantage in pattern.advantages:
                    comparison.append(f"- ✅ {advantage}")
                comparison.append("")
                
            if pattern.disadvantages:
                comparison.extend([
                    "**Cons:**",
                    ""
                ])
                for disadvantage in pattern.disadvantages:
                    comparison.append(f"- ❌ {disadvantage}")
                comparison.append("")
                
        return "\n".join(comparison)
        
    def create_default_patterns(self) -> None:
        """Create default multi-agent patterns based on LLama-Agents."""
        # Workflow Pattern
        self.create_agent_pattern(
            "AgentWorkflow Pattern",
            AgentPatternType.WORKFLOW,
            ComplexityLevel.MINIMAL,
            "Built-in multi-agent behavior with minimal configuration. Agents work together in a predefined workflow.",
            when_to_use=[
                "You want minimal setup complexity",
                "Workflow steps are well-defined",
                "Agents have clear handoff points"
            ],
            when_not_to_use=[
                "Need dynamic agent coordination",
                "Complex decision trees required",
                "Highly interactive agent behavior needed"
            ],
            advantages=[
                "Minimal code required",
                "Built-in error handling",
                "Easy to understand and debug"
            ],
            disadvantages=[
                "Limited flexibility",
                "Hard to customize behavior",
                "Sequential processing only"
            ],
            use_cases=[UseCase.DATA_PROCESSING, UseCase.TASK_AUTOMATION],
            code_sketch="""from llama_agents import AgentWorkflow, SimpleAgent

# Define agents
research_agent = SimpleAgent(role="researcher")
writer_agent = SimpleAgent(role="writer")

# Create workflow
workflow = AgentWorkflow()
workflow.add_step("research", research_agent)
workflow.add_step("write", writer_agent)

# Execute
result = await workflow.run({"topic": "AI trends"})""",
            scalability="Low to Medium",
            maintenance="Low", 
            learning_curve="Beginner-friendly"
        )
        
        # Orchestrator Pattern
        self.create_agent_pattern(
            "Orchestrator Pattern",
            AgentPatternType.ORCHESTRATOR,
            ComplexityLevel.INTERMEDIATE,
            "Central orchestrator coordinates specialist agents as tools. Single point of control with distributed execution.",
            when_to_use=[
                "Need centralized decision making",
                "Agents have specialized capabilities",
                "Complex coordination required"
            ],
            when_not_to_use=[
                "Agents need to interact directly",
                "Highly parallel workflows",
                "Decentralized decision making preferred"
            ],
            advantages=[
                "Clear control flow",
                "Easy to monitor and debug",
                "Flexible agent composition"
            ],
            disadvantages=[
                "Single point of failure",
                "Orchestrator can become complex",
                "Less efficient for simple tasks"
            ],
            use_cases=[UseCase.RESEARCH, UseCase.DECISION_MAKING, UseCase.CONTENT_GENERATION],
            code_sketch="""from llama_agents import Orchestrator, Tool

# Define specialist agents as tools
research_tool = Tool.from_agent(research_agent)
analysis_tool = Tool.from_agent(analysis_agent)
writing_tool = Tool.from_agent(writing_agent)

# Create orchestrator
orchestrator = Orchestrator(
    tools=[research_tool, analysis_tool, writing_tool]
)

# Execute with dynamic tool selection
result = await orchestrator.run("Write a report on market trends")""",
            scalability="Medium to High",
            maintenance="Medium",
            learning_curve="Intermediate"
        )
        
        # Custom Planner Pattern
        self.create_agent_pattern(
            "Custom Planner Pattern",
            AgentPatternType.CUSTOM_PLANNER,
            ComplexityLevel.ADVANCED,
            "Maximum flexibility with custom planning logic. Build sophisticated multi-agent behaviors from scratch.",
            when_to_use=[
                "Need full control over agent interactions",
                "Complex planning logic required",
                "Custom coordination patterns"
            ],
            when_not_to_use=[
                "Simple linear workflows",
                "Limited development resources",
                "Rapid prototyping needed"
            ],
            advantages=[
                "Maximum flexibility",
                "Can implement any pattern",
                "Optimal performance possible"
            ],
            disadvantages=[
                "High development effort",
                "Complex to debug",
                "Requires expert knowledge"
            ],
            use_cases=[UseCase.CREATIVE_TASKS, UseCase.DECISION_MAKING],
            code_sketch="""from llama_agents import CustomPlanner, AgentService

class MyPlanner(CustomPlanner):
    def plan(self, query, context):
        # Custom planning logic
        if "research" in query.lower():
            return self.create_research_plan(query)
        elif "creative" in query.lower():
            return self.create_creative_plan(query)
        
    def create_research_plan(self, query):
        return [
            ("research_agent", {"query": query}),
            ("analysis_agent", {"data": "{{research_result}}"}),
            ("summary_agent", {"analysis": "{{analysis_result}}"})
        ]

planner = MyPlanner()
result = await planner.execute("Research AI market trends")""",
            scalability="High",
            maintenance="High",
            learning_curve="Expert level"
        )
        
    def generate_getting_started_guide(self) -> str:
        """Generate getting started guide with pattern progression."""
        guide = [
            "# Getting Started with Multi-Agent Patterns",
            "",
            "Start simple and evolve your multi-agent system as needed.",
            "",
            "## Learning Path",
            "",
            "### 1. Start with AgentWorkflow (Minimal)",
            "",
            "Perfect for beginners and simple use cases:",
            "",
            "- ✅ Minimal setup required",
            "- ✅ Built-in error handling",
            "- ✅ Easy to understand",
            "",
            "```python",
            "# Your first multi-agent system",
            "workflow = AgentWorkflow()",
            "workflow.add_step('step1', agent1)",
            "workflow.add_step('step2', agent2)",
            "result = await workflow.run(input_data)",
            "```",
            "",
            "### 2. Evolve to Orchestrator (Intermediate)",
            "",
            "When you need more flexibility:",
            "",
            "- ✅ Dynamic agent selection",
            "- ✅ Centralized control",
            "- ✅ Tool-based agent interaction",
            "",
            "```python",
            "# More flexible coordination",
            "orchestrator = Orchestrator(tools=agent_tools)",
            "result = await orchestrator.run(complex_query)",
            "```",
            "",
            "### 3. Custom Planner for Expert Use (Advanced)",
            "",
            "When you need maximum control:",
            "",
            "- ✅ Custom coordination logic",
            "- ✅ Sophisticated planning",
            "- ✅ Optimal performance",
            "",
            "```python",
            "# Full control over agent behavior",
            "class MyPlanner(CustomPlanner):",
            "    def plan(self, query):",
            "        # Your custom logic here",
            "        return dynamic_plan",
            "```",
            "",
            "## Migration Path",
            "",
            "### From Workflow to Orchestrator",
            "",
            "```python",
            "# Before: Workflow",
            "workflow = AgentWorkflow()",
            "workflow.add_step('research', research_agent)",
            "workflow.add_step('write', writing_agent)",
            "",
            "# After: Orchestrator",
            "orchestrator = Orchestrator(tools=[",
            "    Tool.from_agent(research_agent),",
            "    Tool.from_agent(writing_agent)",
            "])",
            "```",
            "",
            "### From Orchestrator to Custom Planner",
            "",
            "```python",
            "# Before: Orchestrator",
            "orchestrator = Orchestrator(tools=tools)",
            "",
            "# After: Custom Planner",
            "class MyPlanner(CustomPlanner):",
            "    def __init__(self):",
            "        self.tools = tools",
            "        ",
            "    def plan(self, query):",
            "        return self.create_dynamic_plan(query)",
            "```",
            ""
        ]
        
        return "\n".join(guide)
        
    def _to_anchor(self, text: str) -> str:
        """Convert text to markdown anchor."""
        return text.lower().replace(" ", "-").replace("_", "-")
        
    def _load_pattern_templates(self) -> Dict[str, str]:
        """Load pattern documentation templates."""
        return {
            "pattern_doc": """# {name}

**Pattern Type:** {pattern_type}
**Complexity:** {complexity}

{description}

## When to Use
{when_to_use}

## Code Sketch
```python
{code_sketch}
```

## Configuration
```yaml
{configuration}
```
""",
            "comparison": """# Pattern Comparison

{comparison_table}

## Recommendations

{recommendations}
"""
        }
        
    def export_pattern_docs(self, output_dir: str) -> None:
        """Export all pattern documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main overview
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_pattern_overview())
            
        # Decision guide
        with open(output_path / "decision-guide.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_decision_guide())
            
        # Getting started
        with open(output_path / "getting-started.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_getting_started_guide())
            
        # Individual pattern docs
        patterns_dir = output_path / "patterns"
        patterns_dir.mkdir(exist_ok=True)
        
        for pattern in self.patterns:
            filename = f"{pattern.name.lower().replace(' ', '-')}.md"
            with open(patterns_dir / filename, 'w', encoding='utf-8') as f:
                f.write(self.generate_pattern_documentation(pattern))
                
        # Comparison docs
        if len(self.patterns) > 1:
            with open(output_path / "pattern-comparison.md", 'w', encoding='utf-8') as f:
                f.write(self.create_comparative_analysis(self.patterns, 
                       ["complexity", "scalability", "use_cases"]))
                
        logger.info(f"Exported pattern docs to {output_dir}")