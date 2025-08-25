"""
Design-First Documentation

Creates conceptual documentation separate from implementation details
with middleware/event-driven patterns based on AutoGen's design-first approach.
"""

import os
import re
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConceptType(Enum):
    """Types of design concepts."""
    ARCHITECTURE = "architecture"
    PATTERN = "pattern" 
    PRINCIPLE = "principle"
    WORKFLOW = "workflow"
    PROTOCOL = "protocol"
    MIDDLEWARE = "middleware"
    EVENT_MODEL = "event_model"
    CONTRACT = "contract"


class DocumentationLayer(Enum):
    """Documentation abstraction layers."""
    CONCEPTUAL = "conceptual"
    LOGICAL = "logical"
    PHYSICAL = "physical"
    IMPLEMENTATION = "implementation"


@dataclass
class DesignConcept:
    """Represents a design concept or pattern."""
    name: str
    concept_type: ConceptType
    layer: DocumentationLayer
    description: str
    rationale: str = ""
    alternatives_considered: List[str] = field(default_factory=list)
    trade_offs: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    diagrams: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    implementation_notes: List[str] = field(default_factory=list)


@dataclass
class MiddlewareSpec:
    """Specification for middleware components."""
    name: str
    purpose: str
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    lifecycle: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    event_handling: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    integration_patterns: List[str] = field(default_factory=list)


@dataclass
class EventDrivenModel:
    """Event-driven architecture model."""
    name: str
    event_types: List[Dict[str, Any]] = field(default_factory=list)
    event_flow: List[Dict[str, Any]] = field(default_factory=list)
    handlers: List[Dict[str, Any]] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    cloud_events_spec: bool = True
    serialization: List[str] = field(default_factory=list)


@dataclass
class ArchitecturalDecision:
    """Architectural Decision Record (ADR)."""
    title: str
    status: str  # proposed, accepted, deprecated, superseded
    context: str
    decision: str
    consequences: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, str]] = field(default_factory=list)
    date_decided: str = ""
    decision_makers: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None


class DesignFirstDocs:
    """
    Design-first documentation generator inspired by AutoGen's
    approach of separating conceptual documentation from implementation.
    """
    
    def __init__(self, docs_root: str = "design-docs"):
        """Initialize design-first docs generator."""
        self.docs_root = Path(docs_root)
        self.concepts = []
        self.middleware_specs = []
        self.event_models = []
        self.decisions = []
        self.concept_hierarchy = {}
        logger.info(f"Design-first docs generator initialized at {docs_root}")
        
    def create_design_concept(self,
                            name: str,
                            concept_type: ConceptType,
                            layer: DocumentationLayer,
                            description: str,
                            **kwargs) -> DesignConcept:
        """Create a new design concept."""
        concept = DesignConcept(
            name=name,
            concept_type=concept_type,
            layer=layer,
            description=description,
            **kwargs
        )
        
        self.concepts.append(concept)
        logger.info(f"Created design concept: {name} ({concept_type.value})")
        return concept
        
    def create_middleware_spec(self,
                             name: str,
                             purpose: str,
                             **kwargs) -> MiddlewareSpec:
        """Create middleware specification."""
        spec = MiddlewareSpec(
            name=name,
            purpose=purpose,
            **kwargs
        )
        
        self.middleware_specs.append(spec)
        logger.info(f"Created middleware spec: {name}")
        return spec
        
    def create_event_driven_model(self,
                                name: str,
                                **kwargs) -> EventDrivenModel:
        """Create event-driven architecture model."""
        model = EventDrivenModel(
            name=name,
            **kwargs
        )
        
        self.event_models.append(model)
        logger.info(f"Created event model: {name}")
        return model
        
    def create_architectural_decision(self,
                                    title: str,
                                    context: str,
                                    decision: str,
                                    status: str = "proposed",
                                    **kwargs) -> ArchitecturalDecision:
        """Create architectural decision record."""
        adr = ArchitecturalDecision(
            title=title,
            context=context,
            decision=decision,
            status=status,
            **kwargs
        )
        
        self.decisions.append(adr)
        logger.info(f"Created ADR: {title}")
        return adr
        
    def generate_concept_overview(self) -> str:
        """Generate high-level conceptual overview."""
        overview = [
            "# System Design Overview",
            "",
            "This document provides a conceptual understanding of the system architecture",
            "and design principles, independent of implementation details.",
            "",
            "## Core Concepts",
            ""
        ]
        
        # Group concepts by type
        by_type = {}
        for concept in self.concepts:
            if concept.concept_type not in by_type:
                by_type[concept.concept_type] = []
            by_type[concept.concept_type].append(concept)
            
        # Generate sections for each concept type
        for concept_type, concepts in by_type.items():
            overview.extend([
                f"### {concept_type.value.title()}",
                ""
            ])
            
            for concept in concepts:
                overview.extend([
                    f"#### {concept.name}",
                    "",
                    concept.description,
                    ""
                ])
                
                if concept.rationale:
                    overview.extend([
                        f"**Why this approach?** {concept.rationale}",
                        ""
                    ])
                    
                if concept.trade_offs:
                    overview.extend([
                        "**Trade-offs:**",
                        ""
                    ])
                    for pro, con in concept.trade_offs.items():
                        overview.append(f"- **{pro}:** {con}")
                    overview.append("")
                    
        return "\n".join(overview)
        
    def generate_middleware_documentation(self) -> str:
        """Generate middleware concept documentation."""
        docs = [
            "# Middleware Architecture",
            "",
            "Understanding the middleware layer and its role in the system.",
            "",
            "## What is Middleware?",
            "",
            "Middleware sits between different system components and provides",
            "common services such as logging, authentication, request routing,",
            "and data transformation. It enables loose coupling and separation",
            "of concerns.",
            "",
            "## Middleware Pipeline",
            "",
            "Requests flow through a pipeline of middleware components:",
            "",
            "```",
            "Request → Auth → Logging → Transform → Route → Handler → Response",
            "```",
            "",
            "## Available Middleware",
            ""
        ]
        
        for spec in self.middleware_specs:
            docs.extend([
                f"### {spec.name}",
                "",
                f"**Purpose:** {spec.purpose}",
                ""
            ])
            
            # Add interfaces
            if spec.interfaces:
                docs.extend([
                    "**Interfaces:**",
                    ""
                ])
                for interface in spec.interfaces:
                    docs.append(f"- `{interface.get('name', 'Unknown')}`: {interface.get('description', '')}")
                docs.append("")
                
            # Add lifecycle
            if spec.lifecycle:
                docs.extend([
                    "**Lifecycle:**",
                    ""
                ])
                for i, stage in enumerate(spec.lifecycle, 1):
                    docs.append(f"{i}. {stage}")
                docs.append("")
                
            # Add configuration example
            if spec.configuration:
                docs.extend([
                    "**Configuration:**",
                    "",
                    "```yaml"
                ])
                docs.append(yaml.dump(spec.configuration, default_flow_style=False))
                docs.extend(["```", ""])
                
            # Add usage examples
            if spec.examples:
                docs.extend([
                    "**Usage Examples:**",
                    ""
                ])
                for example in spec.examples:
                    docs.extend([
                        f"**{example.get('title', 'Example')}:**",
                        "",
                        "```python",
                        example.get('code', '# No code provided'),
                        "```",
                        ""
                    ])
                    if example.get('description'):
                        docs.extend([
                            example['description'],
                            ""
                        ])
                        
        return "\n".join(docs)
        
    def generate_event_driven_documentation(self) -> str:
        """Generate event-driven architecture documentation."""
        docs = [
            "# Event-Driven Architecture",
            "",
            "The system uses an event-driven architecture where components",
            "communicate through events rather than direct calls.",
            "",
            "## Core Principles",
            "",
            "- **Loose Coupling:** Components don't need to know about each other",
            "- **Scalability:** Events can be processed asynchronously",
            "- **Resilience:** System continues working if components fail",
            "- **Auditability:** All actions are recorded as events",
            "",
            "## CloudEvents Specification",
            "",
            "We follow the [CloudEvents](https://cloudevents.io/) specification",
            "for event format and metadata:",
            "",
            "```json",
            "{",
            '  "specversion": "1.0",',
            '  "type": "com.example.user.created",',
            '  "source": "/user-service",',
            '  "id": "1234-5678-90ab",',
            '  "time": "2023-10-01T12:00:00Z",',
            '  "datacontenttype": "application/json",',
            '  "data": { "userId": "user123", "email": "user@example.com" }',
            "}",
            "```",
            "",
            "## Event Models",
            ""
        ]
        
        for model in self.event_models:
            docs.extend([
                f"### {model.name}",
                ""
            ])
            
            # Event types
            if model.event_types:
                docs.extend([
                    "**Event Types:**",
                    ""
                ])
                for event_type in model.event_types:
                    docs.extend([
                        f"#### {event_type.get('name', 'Unknown')}",
                        "",
                        event_type.get('description', 'No description provided.'),
                        ""
                    ])
                    
                    if event_type.get('schema'):
                        docs.extend([
                            "**Schema:**",
                            "",
                            "```json",
                            json.dumps(event_type['schema'], indent=2),
                            "```",
                            ""
                        ])
                        
            # Event flow
            if model.event_flow:
                docs.extend([
                    "**Event Flow:**",
                    ""
                ])
                for i, step in enumerate(model.event_flow, 1):
                    docs.append(f"{i}. **{step.get('stage', 'Stage')}:** {step.get('description', '')}")
                docs.append("")
                
            # Handlers
            if model.handlers:
                docs.extend([
                    "**Event Handlers:**",
                    ""
                ])
                for handler in model.handlers:
                    docs.extend([
                        f"- **{handler.get('name', 'Handler')}**",
                        f"  - Handles: {', '.join(handler.get('events', []))}",
                        f"  - Description: {handler.get('description', '')}",
                        ""
                    ])
                    
        return "\n".join(docs)
        
    def generate_adr_document(self, adr: ArchitecturalDecision) -> str:
        """Generate single ADR document."""
        doc = [
            f"# ADR: {adr.title}",
            "",
            f"**Status:** {adr.status}",
            ""
        ]
        
        if adr.date_decided:
            doc.extend([
                f"**Date:** {adr.date_decided}",
                ""
            ])
            
        if adr.decision_makers:
            doc.extend([
                f"**Decision Makers:** {', '.join(adr.decision_makers)}",
                ""
            ])
            
        doc.extend([
            "## Context",
            "",
            adr.context,
            "",
            "## Decision",
            "",
            adr.decision,
            ""
        ])
        
        # Alternatives considered
        if adr.alternatives:
            doc.extend([
                "## Alternatives Considered",
                ""
            ])
            for alt in adr.alternatives:
                doc.extend([
                    f"### {alt.get('name', 'Alternative')}",
                    "",
                    alt.get('description', ''),
                    ""
                ])
                if alt.get('rejected_because'):
                    doc.extend([
                        f"**Rejected because:** {alt['rejected_because']}",
                        ""
                    ])
                    
        # Consequences
        if adr.consequences:
            doc.extend([
                "## Consequences",
                ""
            ])
            for consequence in adr.consequences:
                doc.append(f"- {consequence}")
            doc.append("")
            
        # Supersession info
        if adr.superseded_by:
            doc.extend([
                "## Superseded By",
                "",
                f"This decision has been superseded by: {adr.superseded_by}",
                ""
            ])
            
        return "\n".join(doc)
        
    def generate_all_adrs_index(self) -> str:
        """Generate index of all ADRs."""
        index = [
            "# Architectural Decision Records",
            "",
            "This directory contains all architectural decisions made for the project.",
            "",
            "## Decision Index",
            ""
        ]
        
        # Group by status
        by_status = {}
        for adr in self.decisions:
            if adr.status not in by_status:
                by_status[adr.status] = []
            by_status[adr.status].append(adr)
            
        # Show active decisions first
        status_order = ["accepted", "proposed", "deprecated", "superseded"]
        
        for status in status_order:
            if status in by_status:
                index.extend([
                    f"### {status.title()} Decisions",
                    ""
                ])
                
                for adr in by_status[status]:
                    filename = self._adr_filename(adr.title)
                    index.append(f"- [{adr.title}]({filename})")
                    if adr.date_decided:
                        index.append(f"  - *Decided: {adr.date_decided}*")
                        
                index.append("")
                
        return "\n".join(index)
        
    def create_programming_model_docs(self) -> str:
        """Create programming model documentation."""
        docs = [
            "# Programming Model",
            "",
            "Understanding how to build applications with our framework.",
            "",
            "## Core Abstractions",
            "",
            "### Agents",
            "",
            "Agents are the primary building blocks of the system. They encapsulate",
            "behavior and can communicate with other agents through messages.",
            "",
            "```python",
            "class MyAgent(Agent):",
            "    async def handle_message(self, message):",
            "        # Process message and optionally respond",
            "        response = await self.process(message)",
            "        return response",
            "```",
            "",
            "### Messages",
            "",
            "Messages carry information between agents. They follow the CloudEvents",
            "specification for consistency and interoperability.",
            "",
            "```python",
            "message = Message(",
            "    type='user.query',",
            "    data={'query': 'What is the weather today?'},",
            "    source='/user-interface'",
            ")",
            "```",
            "",
            "### Workflows",
            "",
            "Workflows orchestrate multiple agents to complete complex tasks:",
            "",
            "```python",
            "workflow = Workflow()",
            "workflow.add_step('research', research_agent)",
            "workflow.add_step('analyze', analysis_agent)",
            "workflow.add_step('report', reporting_agent)",
            "",
            "result = await workflow.execute(initial_message)",
            "```",
            "",
            "## Event-Driven Communication",
            "",
            "Components communicate through events rather than direct calls:",
            "",
            "1. **Publish Events:** Components publish events when something happens",
            "2. **Subscribe to Events:** Components subscribe to events they care about", 
            "3. **Handle Events:** Components react to events asynchronously",
            "",
            "This pattern enables:",
            "- **Loose coupling** between components",
            "- **Scalability** through async processing",
            "- **Resilience** through retry and error handling",
            "",
            "## Middleware Integration",
            "",
            "Middleware components can be added to the processing pipeline:",
            "",
            "```python",
            "app = Application()",
            "app.use(AuthenticationMiddleware())",
            "app.use(LoggingMiddleware())",
            "app.use(RateLimitMiddleware())",
            "```",
            "",
            "Each middleware can:",
            "- Inspect and modify incoming messages",
            "- Add metadata or context",
            "- Implement cross-cutting concerns",
            "- Short-circuit the pipeline if needed",
            ""
        ]
        
        return "\n".join(docs)
        
    def build_concept_hierarchy(self) -> Dict[str, Any]:
        """Build hierarchy of concepts based on dependencies."""
        hierarchy = {}
        
        # Create nodes for each concept
        for concept in self.concepts:
            hierarchy[concept.name] = {
                "concept": concept,
                "children": [],
                "dependencies": concept.dependencies
            }
            
        # Build parent-child relationships
        for name, node in hierarchy.items():
            for dep in node["dependencies"]:
                if dep in hierarchy:
                    hierarchy[dep]["children"].append(name)
                    
        return hierarchy
        
    def generate_concept_map(self) -> str:
        """Generate visual concept map in Mermaid format."""
        mermaid = [
            "# System Concept Map",
            "",
            "```mermaid",
            "graph TD"
        ]
        
        # Add nodes
        for concept in self.concepts:
            node_id = self._concept_id(concept.name)
            mermaid.append(f"    {node_id}[{concept.name}]")
            
        # Add relationships
        for concept in self.concepts:
            node_id = self._concept_id(concept.name)
            
            # Dependencies
            for dep in concept.dependencies:
                dep_id = self._concept_id(dep)
                mermaid.append(f"    {dep_id} --> {node_id}")
                
            # Related concepts
            for related in concept.related_concepts:
                related_id = self._concept_id(related)
                mermaid.append(f"    {node_id} -.-> {related_id}")
                
        mermaid.extend(["```", ""])
        
        return "\n".join(mermaid)
        
    def _concept_id(self, name: str) -> str:
        """Generate Mermaid-safe ID for concept."""
        return re.sub(r'[^a-zA-Z0-9]', '_', name).upper()
        
    def _adr_filename(self, title: str) -> str:
        """Generate filename for ADR."""
        safe_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        safe_title = re.sub(r'\s+', '-', safe_title.strip()).lower()
        return f"adr-{safe_title}.md"
        
    def export_design_docs(self, output_dir: str) -> None:
        """Export all design documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_path / "concepts").mkdir(exist_ok=True)
        (output_path / "middleware").mkdir(exist_ok=True)
        (output_path / "events").mkdir(exist_ok=True)
        (output_path / "decisions").mkdir(exist_ok=True)
        
        # Generate overview
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_concept_overview())
            
        # Generate middleware docs
        with open(output_path / "middleware" / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_middleware_documentation())
            
        # Generate event docs
        with open(output_path / "events" / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_event_driven_documentation())
            
        # Generate programming model
        with open(output_path / "programming-model.md", 'w', encoding='utf-8') as f:
            f.write(self.create_programming_model_docs())
            
        # Generate ADRs
        with open(output_path / "decisions" / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_all_adrs_index())
            
        for adr in self.decisions:
            filename = self._adr_filename(adr.title)
            with open(output_path / "decisions" / filename, 'w', encoding='utf-8') as f:
                f.write(self.generate_adr_document(adr))
                
        # Generate concept map
        with open(output_path / "concept-map.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_concept_map())
            
        logger.info(f"Exported design docs to {output_dir}")
        
    def create_default_concepts(self) -> None:
        """Create default design concepts for common patterns."""
        # Agent architecture concept
        self.create_design_concept(
            "Agent-Oriented Architecture",
            ConceptType.ARCHITECTURE,
            DocumentationLayer.CONCEPTUAL,
            "System built around autonomous agents that can communicate and collaborate",
            rationale="Enables distributed processing and loose coupling",
            trade_offs={
                "Flexibility": "Increased complexity in coordination", 
                "Scalability": "Need for sophisticated message routing",
                "Resilience": "Debugging distributed interactions is harder"
            }
        )
        
        # Event-driven pattern
        self.create_design_concept(
            "Event-Driven Communication",
            ConceptType.PATTERN,
            DocumentationLayer.LOGICAL,
            "Components communicate through events following CloudEvents specification",
            rationale="Decouples producers from consumers, enables async processing",
            dependencies=["Agent-Oriented Architecture"]
        )
        
        # Middleware pattern
        self.create_design_concept(
            "Middleware Pipeline",
            ConceptType.MIDDLEWARE,
            DocumentationLayer.LOGICAL,
            "Request processing pipeline with pluggable middleware components",
            rationale="Separation of cross-cutting concerns from business logic",
            dependencies=["Event-Driven Communication"]
        )