"""
Workflow Recipe Documentation

Creates complex orchestration patterns with state management and 
event-driven workflows based on PhiData's workflow documentation approach.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of workflow patterns."""
    LINEAR = "linear"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    EVENT_DRIVEN = "event_driven"
    STATE_MACHINE = "state_machine"
    DAG = "dag"  # Directed Acyclic Graph
    RECURSIVE = "recursive"


class StateType(Enum):
    """Workflow state types."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    RETRY = "retry"


class EventType(Enum):
    """Workflow event types."""
    START = "start"
    COMPLETE = "complete"
    ERROR = "error"
    TIMEOUT = "timeout"
    USER_INPUT = "user_input"
    TRIGGER = "trigger"
    STATE_CHANGE = "state_change"
    DATA_READY = "data_ready"


class CachingStrategy(Enum):
    """Result caching strategies."""
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    DATABASE = "database"
    DISTRIBUTED = "distributed"


@dataclass
class WorkflowStep:
    """Represents a single workflow step."""
    step_id: str
    name: str
    description: str
    agent_name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    caching: CachingStrategy = CachingStrategy.MEMORY
    validation_rules: List[str] = field(default_factory=list)
    error_handling: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """Represents workflow state information."""
    workflow_id: str
    current_step: str
    state: StateType
    step_states: Dict[str, StateType] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    cache_hits: int = 0
    total_steps: int = 0


@dataclass
class EventHandler:
    """Event handling configuration."""
    event_type: EventType
    handler_function: str
    conditions: List[str] = field(default_factory=list)
    priority: int = 1
    async_execution: bool = False
    retry_on_failure: bool = True


@dataclass
class WorkflowRecipe:
    """Complete workflow recipe specification."""
    name: str
    description: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep] = field(default_factory=list)
    event_handlers: List[EventHandler] = field(default_factory=list)
    state_management: Dict[str, Any] = field(default_factory=dict)
    caching_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    quality_gates: List[str] = field(default_factory=list)
    rollback_strategy: str = ""
    execution_timeout: int = 3600  # 1 hour default
    max_parallel_steps: int = 5


class WorkflowRecipeDocs:
    """
    Workflow recipe documentation system inspired by PhiData's complex
    orchestration patterns with state management and event-driven workflows.
    """
    
    def __init__(self, workflows_dir: str = "workflow_recipes"):
        """Initialize workflow recipe documentation."""
        self.workflows_dir = Path(workflows_dir)
        self.workflow_recipes = []
        self.state_machines = {}
        self.event_configurations = {}
        self.caching_strategies = {}
        logger.info(f"Workflow recipe docs initialized at {workflows_dir}")
        
    def create_workflow_recipe(self,
                             name: str,
                             description: str,
                             workflow_type: WorkflowType,
                             **kwargs) -> WorkflowRecipe:
        """Create a new workflow recipe."""
        recipe = WorkflowRecipe(
            name=name,
            description=description,
            workflow_type=workflow_type,
            **kwargs
        )
        
        self.workflow_recipes.append(recipe)
        logger.info(f"Created workflow recipe: {name} ({workflow_type.value})")
        return recipe
        
    def add_workflow_step(self,
                         recipe: WorkflowRecipe,
                         step_id: str,
                         name: str,
                         description: str,
                         agent_name: str,
                         **kwargs) -> WorkflowStep:
        """Add step to workflow recipe."""
        step = WorkflowStep(
            step_id=step_id,
            name=name,
            description=description,
            agent_name=agent_name,
            **kwargs
        )
        
        recipe.steps.append(step)
        logger.info(f"Added step {step_id} to workflow {recipe.name}")
        return step
        
    def add_event_handler(self,
                         recipe: WorkflowRecipe,
                         event_type: EventType,
                         handler_function: str,
                         **kwargs) -> EventHandler:
        """Add event handler to workflow."""
        handler = EventHandler(
            event_type=event_type,
            handler_function=handler_function,
            **kwargs
        )
        
        recipe.event_handlers.append(handler)
        logger.info(f"Added {event_type.value} handler to workflow {recipe.name}")
        return handler
        
    def generate_workflow_documentation(self, recipe: WorkflowRecipe) -> str:
        """Generate comprehensive workflow documentation."""
        doc = [
            f'"""',
            f'{recipe.name} - Workflow Recipe',
            "",
            recipe.description,
            "",
            f"Workflow Type: {recipe.workflow_type.value.title()}",
            f"Total Steps: {len(recipe.steps)}",
            f"Event Handlers: {len(recipe.event_handlers)}",
        ]
        
        if recipe.quality_gates:
            doc.extend([
                "",
                "Quality Gates:",
            ])
            for gate in recipe.quality_gates:
                doc.append(f"- {gate}")
                
        doc.extend([
            '"""',
            "",
            "from phi.agent import Agent",
            "from phi.workflow import Workflow, WorkflowStep",
            "from typing import Dict, List, Any, Optional",
            "import asyncio",
            "import json",
            "from datetime import datetime",
            "from enum import Enum",
            ""
        ])
        
        # Generate state enum
        doc.extend([
            "class WorkflowState(Enum):",
            '    """Workflow execution states."""',
            "    PENDING = 'pending'",
            "    RUNNING = 'running'", 
            "    COMPLETED = 'completed'",
            "    FAILED = 'failed'",
            "    PAUSED = 'paused'",
            ""
        ])
        
        # Generate agents
        agents_created = set()
        for step in recipe.steps:
            if step.agent_name not in agents_created:
                doc.extend([
                    f"# {step.agent_name} Agent",
                    f"{step.agent_name.lower().replace(' ', '_')} = Agent(",
                    f'    name="{step.agent_name}",',
                    f'    description="Agent for {step.description.lower()}",',
                    '    instructions="""',
                    f'    You are responsible for: {step.description}',
                    '    Provide clear, structured output for the next workflow step.',
                    '    """',
                    ")",
                    ""
                ])
                agents_created.add(step.agent_name)
                
        # Generate workflow class
        class_name = f"{recipe.name.replace(' ', '')}Workflow"
        doc.extend([
            f"class {class_name}:",
            f'    """',
            f'    {recipe.name} implementation with state management and event handling.',
            f'    """',
            "",
            "    def __init__(self):",
            "        self.state = WorkflowState.PENDING",
            "        self.step_results = {}",
            "        self.current_step = 0",
            "        self.error_log = []",
            f"        self.total_steps = {len(recipe.steps)}",
            ""
        ])
        
        # Generate caching methods if configured
        if recipe.caching_config or any(step.caching != CachingStrategy.NONE for step in recipe.steps):
            doc.extend(self._generate_caching_methods(recipe))
            
        # Generate event handlers
        if recipe.event_handlers:
            doc.extend(self._generate_event_handler_methods(recipe))
            
        # Generate main execution method
        doc.extend(self._generate_execution_method(recipe))
        
        # Generate individual step methods
        for step in recipe.steps:
            doc.extend(self._generate_step_method(step))
            
        # Generate state management methods
        doc.extend(self._generate_state_management_methods(recipe))
        
        # Generate example usage
        doc.extend(self._generate_workflow_usage_example(recipe))
        
        return "\n".join(doc)
        
    def _generate_caching_methods(self, recipe: WorkflowRecipe) -> List[str]:
        """Generate caching methods for workflow."""
        caching_code = [
            "    def _get_cache_key(self, step_id: str, inputs: Dict[str, Any]) -> str:",
            '        """Generate cache key for step results."""',
            "        import hashlib",
            "        key_data = f'{step_id}:{json.dumps(inputs, sort_keys=True)}'",
            "        return hashlib.md5(key_data.encode()).hexdigest()",
            "",
            "    def _get_cached_result(self, cache_key: str) -> Optional[Any]:",
            '        """Retrieve cached result if available."""',
            "        # Implementation depends on caching strategy",
            "        return self.step_results.get(cache_key)",
            "",
            "    def _cache_result(self, cache_key: str, result: Any) -> None:",
            '        """Cache step result."""',
            "        self.step_results[cache_key] = result",
            ""
        ]
        
        return caching_code
        
    def _generate_event_handler_methods(self, recipe: WorkflowRecipe) -> List[str]:
        """Generate event handler methods."""
        handler_code = [
            "    async def _handle_event(self, event_type: str, data: Dict[str, Any]) -> None:",
            '        """Handle workflow events."""',
            "        handlers = {",
        ]
        
        for handler in recipe.event_handlers:
            handler_code.append(f"            '{handler.event_type.value}': self.{handler.handler_function},")
            
        handler_code.extend([
            "        }",
            "",
            "        if event_type in handlers:",
            "            await handlers[event_type](data)",
            ""
        ])
        
        # Generate individual handler methods
        for handler in recipe.event_handlers:
            handler_code.extend([
                f"    async def {handler.handler_function}(self, data: Dict[str, Any]) -> None:",
                f'        """Handle {handler.event_type.value} event."""',
                f'        print(f"Handling {handler.event_type.value} event: {{data}}")',
                "        # Custom event handling logic here",
                ""
            ])
            
        return handler_code
        
    def _generate_execution_method(self, recipe: WorkflowRecipe) -> List[str]:
        """Generate main workflow execution method."""
        exec_code = [
            "    async def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:",
            f'        """Execute {recipe.name} workflow."""',
            "        self.state = WorkflowState.RUNNING",
            "        await self._handle_event('start', {'input': initial_input})",
            "",
            "        try:",
        ]
        
        if recipe.workflow_type == WorkflowType.LINEAR:
            exec_code.extend([
                "            # Linear workflow execution",
                "            current_input = initial_input",
                "",
                "            for i, step in enumerate(self._get_workflow_steps()):",
                "                self.current_step = i",
                "                result = await self._execute_step(step, current_input)",
                "                current_input = result",
                "",
                "            final_result = current_input",
            ])
        elif recipe.workflow_type == WorkflowType.PARALLEL:
            exec_code.extend([
                "            # Parallel workflow execution",
                "            tasks = []",
                "            for step in self._get_workflow_steps():",
                "                task = asyncio.create_task(self._execute_step(step, initial_input))",
                "                tasks.append(task)",
                "",
                "            results = await asyncio.gather(*tasks)",
                "            final_result = {'step_results': results}",
            ])
        elif recipe.workflow_type == WorkflowType.EVENT_DRIVEN:
            exec_code.extend([
                "            # Event-driven workflow execution",
                "            final_result = await self._execute_event_driven_workflow(initial_input)",
            ])
        else:
            exec_code.extend([
                f"            # {recipe.workflow_type.value.title()} workflow execution",
                "            final_result = await self._execute_custom_workflow(initial_input)",
            ])
            
        exec_code.extend([
            "",
            "            self.state = WorkflowState.COMPLETED",
            "            await self._handle_event('complete', {'result': final_result})",
            "            return final_result",
            "",
            "        except Exception as e:",
            "            self.state = WorkflowState.FAILED",
            "            self.error_log.append({'error': str(e), 'timestamp': datetime.now()})",
            "            await self._handle_event('error', {'error': str(e)})",
            "            raise",
            ""
        ])
        
        return exec_code
        
    def _generate_step_method(self, step: WorkflowStep) -> List[str]:
        """Generate method for individual workflow step."""
        method_name = f"_execute_{step.step_id.lower()}"
        agent_var = step.agent_name.lower().replace(' ', '_')
        
        step_code = [
            f"    async def {method_name}(self, input_data: Dict[str, Any]) -> Dict[str, Any]:",
            f'        """Execute {step.name} step."""',
            f'        print(f"Executing step: {step.name}")',
            ""
        ]
        
        # Add caching logic if enabled
        if step.caching != CachingStrategy.NONE:
            step_code.extend([
                "        # Check cache first",
                f"        cache_key = self._get_cache_key('{step.step_id}', input_data)",
                "        cached_result = self._get_cached_result(cache_key)",
                "        if cached_result:",
                f'            print(f"Using cached result for {step.name}")',
                "            return cached_result",
                ""
            ])
            
        # Add validation if specified
        if step.validation_rules:
            step_code.extend([
                "        # Input validation",
                "        validation_errors = []",
            ])
            
            for rule in step.validation_rules:
                step_code.append(f'        # Validate: {rule}')
                
            step_code.extend([
                "        if validation_errors:",
                "            raise ValueError(f'Validation failed: {validation_errors}')",
                ""
            ])
            
        # Main step execution
        step_code.extend([
            "        # Execute step",
            f"        try:",
            f"            result = await {agent_var}.run(",
            f"                f\"Step: {step.description}\\nInput: {{input_data}}\"",
            "            )",
            "",
            "            # Process result",
            "            processed_result = {",
            f"                'step_id': '{step.step_id}',",
            f"                'step_name': '{step.name}',",
            "                'result': result,",
            "                'timestamp': datetime.now().isoformat()",
            "            }",
            ""
        ])
        
        # Add caching of result
        if step.caching != CachingStrategy.NONE:
            step_code.extend([
                "            # Cache result",
                "            self._cache_result(cache_key, processed_result)",
                ""
            ])
            
        # Add error handling
        step_code.extend([
            "            return processed_result",
            "",
            "        except Exception as e:",
            f"            error_msg = f'Step {step.name} failed: {{e}}'",
            "            self.error_log.append({",
            f"                'step_id': '{step.step_id}',",
            "                'error': error_msg,",
            "                'timestamp': datetime.now().isoformat()",
            "            })",
        ])
        
        # Add retry logic if configured
        if step.retry_policy:
            max_retries = step.retry_policy.get('max_retries', 3)
            step_code.extend([
                f"            # Retry logic (max {max_retries} retries)",
                "            # Implement retry mechanism here",
            ])
            
        step_code.extend([
            "            raise",
            ""
        ])
        
        return step_code
        
    def _generate_state_management_methods(self, recipe: WorkflowRecipe) -> List[str]:
        """Generate state management methods."""
        state_code = [
            "    def _get_workflow_steps(self) -> List[Dict[str, Any]]:",
            '        """Get workflow step definitions."""',
            "        return [",
        ]
        
        for step in recipe.steps:
            state_code.extend([
                "            {",
                f"                'id': '{step.step_id}',",
                f"                'name': '{step.name}',",
                f"                'agent': '{step.agent_name}',",
                f"                'dependencies': {step.dependencies},",
                "            },"
            ])
            
        state_code.extend([
            "        ]",
            "",
            "    def get_workflow_status(self) -> Dict[str, Any]:",
            '        """Get current workflow status."""',
            "        return {",
            "            'state': self.state.value,",
            "            'current_step': self.current_step,",
            "            'total_steps': self.total_steps,",
            "            'progress': (self.current_step / self.total_steps) * 100,",
            "            'errors': len(self.error_log)",
            "        }",
            ""
        ])
        
        return state_code
        
    def _generate_workflow_usage_example(self, recipe: WorkflowRecipe) -> List[str]:
        """Generate workflow usage example."""
        class_name = f"{recipe.name.replace(' ', '')}Workflow"
        
        usage_code = [
            "# Example Usage",
            "async def main():",
            f'    """Example usage of {recipe.name} workflow."""',
            f"    workflow = {class_name}()",
            "",
            "    # Example input",
            "    input_data = {",
        ]
        
        # Add example inputs based on first step
        if recipe.steps:
            first_step = recipe.steps[0]
            for input_name in first_step.inputs[:3]:  # Show first 3 inputs
                usage_code.append(f'        "{input_name}": "example_{input_name}",')
                
        usage_code.extend([
            "    }",
            "",
            "    try:",
            "        # Execute workflow",
            f'        print(f"Starting {recipe.name} workflow...")',
            "        result = await workflow.execute(input_data)",
            "        ",
            '        print("Workflow completed successfully!")',
            '        print(f"Result: {result}")',
            "",
            "        # Check final status",
            "        status = workflow.get_workflow_status()",
            '        print(f"Final status: {status}")',
            "",
            "    except Exception as e:",
            '        print(f"Workflow failed: {e}")',
            "        status = workflow.get_workflow_status()",
            '        print(f"Error status: {status}")',
            "",
            'if __name__ == "__main__":',
            "    asyncio.run(main())",
            ""
        ])
        
        return usage_code
        
    def create_default_workflow_recipes(self) -> None:
        """Create default workflow recipes based on PhiData patterns."""
        # Research Workflow
        research_workflow = self.create_workflow_recipe(
            "Comprehensive Research Workflow",
            "Multi-stage research process with validation and synthesis",
            WorkflowType.LINEAR,
            quality_gates=[
                "Minimum 5 credible sources",
                "Cross-validation of key facts",
                "Clear executive summary"
            ],
            execution_timeout=7200  # 2 hours
        )
        
        self.add_workflow_step(
            research_workflow,
            "gather_sources",
            "Source Gathering",
            "Collect initial research sources and validate credibility",
            "Web Researcher",
            inputs=["research_topic", "scope_parameters"],
            outputs=["source_list", "credibility_scores"],
            validation_rules=["At least 5 sources required", "Credibility score > 0.7"],
            caching=CachingStrategy.DISK
        )
        
        self.add_workflow_step(
            research_workflow,
            "extract_insights",
            "Data Extraction",
            "Extract key insights and data points from sources",
            "Data Analyst",
            inputs=["source_list"],
            outputs=["extracted_data", "insights_summary"],
            dependencies=["gather_sources"],
            timeout=1800  # 30 minutes
        )
        
        self.add_workflow_step(
            research_workflow,
            "synthesize_report",
            "Report Synthesis",
            "Create comprehensive research report with findings",
            "Report Writer",
            inputs=["extracted_data", "insights_summary"],
            outputs=["final_report", "executive_summary"],
            dependencies=["extract_insights"],
            validation_rules=["Report must have executive summary", "Minimum 1000 words"]
        )
        
        # Add event handlers
        self.add_event_handler(
            research_workflow,
            EventType.ERROR,
            "handle_research_error",
            priority=1,
            retry_on_failure=True
        )
        
        self.add_event_handler(
            research_workflow,
            EventType.COMPLETE,
            "handle_research_completion",
            priority=1
        )
        
        # Content Creation Workflow
        content_workflow = self.create_workflow_recipe(
            "Content Creation Pipeline",
            "Parallel content creation with review and optimization",
            WorkflowType.PARALLEL,
            max_parallel_steps=3,
            quality_gates=[
                "Grammar score > 90%",
                "SEO score > 80%",
                "Readability grade < 10"
            ]
        )
        
        self.add_workflow_step(
            content_workflow,
            "draft_content",
            "Content Drafting",
            "Create initial content draft based on requirements",
            "Content Writer",
            inputs=["content_brief", "target_audience"],
            outputs=["content_draft"],
            caching=CachingStrategy.MEMORY
        )
        
        self.add_workflow_step(
            content_workflow,
            "optimize_seo",
            "SEO Optimization",
            "Optimize content for search engines",
            "SEO Specialist",
            inputs=["content_draft", "target_keywords"],
            outputs=["seo_optimized_content"],
            validation_rules=["Keyword density 1-3%", "Meta description present"]
        )
        
        self.add_workflow_step(
            content_workflow,
            "review_quality",
            "Quality Review",
            "Review content for quality, grammar, and consistency",
            "Content Reviewer",
            inputs=["seo_optimized_content"],
            outputs=["quality_report", "final_content"],
            validation_rules=["Grammar score > 90%", "No plagiarism detected"]
        )
        
    def generate_workflow_comparison_table(self) -> str:
        """Generate comparison table of workflow recipes."""
        table = [
            "# Workflow Recipes Comparison",
            "",
            "| Workflow | Type | Steps | Caching | Quality Gates | Timeout |",
            "|----------|------|-------|---------|---------------|---------|"
        ]
        
        for recipe in self.workflow_recipes:
            caching_used = any(step.caching != CachingStrategy.NONE for step in recipe.steps)
            caching_str = "✅" if caching_used else "❌"
            quality_gates_str = "✅" if recipe.quality_gates else "❌"
            timeout_str = f"{recipe.execution_timeout}s"
            
            table.append(
                f"| {recipe.name} | {recipe.workflow_type.value} | "
                f"{len(recipe.steps)} | {caching_str} | {quality_gates_str} | {timeout_str} |"
            )
            
        return "\n".join(table)
        
    def export_workflow_recipes(self, output_dir: str) -> None:
        """Export all workflow recipe documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main index with comparison table
        index_content = [
            "# Workflow Recipe Documentation",
            "",
            "Complex orchestration patterns with state management and event handling.",
            "",
            self.generate_workflow_comparison_table(),
            "",
            "## Available Workflows",
            ""
        ]
        
        for recipe in self.workflow_recipes:
            index_content.extend([
                f"### [{recipe.name}]({recipe.name.lower().replace(' ', '_')}.py)",
                "",
                recipe.description,
                "",
                f"- **Type:** {recipe.workflow_type.value.title()}",
                f"- **Steps:** {len(recipe.steps)}",
                f"- **Event Handlers:** {len(recipe.event_handlers)}",
                f"- **Quality Gates:** {len(recipe.quality_gates)}",
                ""
            ])
            
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(index_content))
            
        # Generate individual workflow files
        for recipe in self.workflow_recipes:
            filename = f"{recipe.name.lower().replace(' ', '_')}.py"
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                f.write(self.generate_workflow_documentation(recipe))
                
        # Generate workflow patterns guide
        patterns_guide = [
            "# Workflow Patterns Guide",
            "",
            "Understanding different workflow execution patterns.",
            "",
            "## Pattern Types",
            ""
        ]
        
        for pattern_type in WorkflowType:
            patterns_guide.extend([
                f"### {pattern_type.value.title()}",
                "",
                self._get_pattern_description(pattern_type),
                ""
            ])
            
        with open(output_path / "workflow_patterns.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(patterns_guide))
            
        logger.info(f"Exported workflow recipes to {output_dir}")
        
    def _get_pattern_description(self, pattern_type: WorkflowType) -> str:
        """Get description for workflow pattern type."""
        descriptions = {
            WorkflowType.LINEAR: "Steps execute in sequence, each depending on the previous",
            WorkflowType.PARALLEL: "Steps execute simultaneously for maximum efficiency",
            WorkflowType.CONDITIONAL: "Execution path depends on runtime conditions",
            WorkflowType.LOOP: "Steps repeat until termination condition is met",
            WorkflowType.EVENT_DRIVEN: "Execution triggered by events and state changes",
            WorkflowType.STATE_MACHINE: "Complex state transitions with defined rules",
            WorkflowType.DAG: "Directed acyclic graph with complex dependencies",
            WorkflowType.RECURSIVE: "Self-referencing workflows for complex problems"
        }
        return descriptions.get(pattern_type, "Custom workflow pattern")