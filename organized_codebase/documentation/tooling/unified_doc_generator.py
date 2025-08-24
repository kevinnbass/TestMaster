"""
Unified Documentation Generator

Master orchestrator that combines all documentation patterns from 
Phase 1 into a single, intelligent documentation generation system.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DocumentationFramework(Enum):
    """Source frameworks for documentation patterns."""
    AGENCY_SWARM = "agency_swarm"
    CREWAI = "crewai"
    AGENTSCOPE = "agentscope"
    AUTOGEN = "autogen"
    LLAMA_AGENTS = "llama_agents"
    PHIDATA = "phidata"
    SWARMS = "swarms"


class GenerationType(Enum):
    """Types of documentation to generate."""
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    COOKBOOK = "cookbook"
    ARCHITECTURE = "architecture"
    INTEGRATION_GUIDE = "integration_guide"
    TROUBLESHOOTING = "troubleshooting"
    INTERACTIVE = "interactive"
    MULTILINGUAL = "multilingual"


class OutputFormat(Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    JUPYTER = "jupyter"
    MDX = "mdx"
    CONFLUENCE = "confluence"
    NOTION = "notion"


@dataclass
class DocumentationRequest:
    """Request for documentation generation."""
    title: str
    generation_type: GenerationType
    output_format: OutputFormat
    source_framework: Optional[DocumentationFramework] = None
    target_audience: str = "developers"
    complexity_level: int = 3  # 1-5 scale
    include_examples: bool = True
    include_interactive: bool = False
    languages: List[str] = field(default_factory=lambda: ["en"])
    custom_templates: Dict[str, Any] = field(default_factory=dict)
    content_requirements: List[str] = field(default_factory=list)


@dataclass
class GenerationContext:
    """Context information for documentation generation."""
    project_info: Dict[str, Any] = field(default_factory=dict)
    existing_docs: List[str] = field(default_factory=list)
    code_analysis: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedDocGenerator:
    """
    Unified documentation generator that orchestrates all 31 modules
    from Phase 1 to create comprehensive, intelligent documentation.
    """
    
    def __init__(self, output_dir: str = "generated_docs"):
        """Initialize unified documentation generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import all Phase 1 documentation modules
        self.generators = self._initialize_generators()
        self.templates = self._load_unified_templates()
        self.patterns = self._load_framework_patterns()
        
        logger.info("Unified documentation generator initialized")
        
    def generate_documentation(self, 
                             request: DocumentationRequest,
                             context: GenerationContext) -> Dict[str, Any]:
        """Generate documentation based on request and context."""
        logger.info(f"Generating {request.generation_type.value} documentation: {request.title}")
        
        # Select optimal framework pattern
        selected_framework = self._select_framework_pattern(request, context)
        
        # Choose appropriate generator
        generator = self._get_generator(request.generation_type, selected_framework)
        
        # Generate base content
        base_content = generator.generate(request, context)
        
        # Apply enhancements based on request
        enhanced_content = self._apply_enhancements(base_content, request, context)
        
        # Format output
        formatted_output = self._format_output(enhanced_content, request.output_format)
        
        # Generate metadata
        metadata = self._generate_metadata(request, context, selected_framework)
        
        result = {
            "title": request.title,
            "content": formatted_output,
            "metadata": metadata,
            "framework_used": selected_framework.value,
            "generation_stats": self._get_generation_stats()
        }
        
        # Save to output directory
        self._save_generated_docs(result, request)
        
        logger.info(f"Documentation generation complete: {request.title}")
        return result
        
    def generate_multi_framework_comparison(self, topic: str) -> Dict[str, Any]:
        """Generate comparative documentation across all frameworks."""
        comparison_docs = {}
        
        for framework in DocumentationFramework:
            request = DocumentationRequest(
                title=f"{topic} - {framework.value.title()} Approach",
                generation_type=GenerationType.ARCHITECTURE,
                output_format=OutputFormat.MARKDOWN,
                source_framework=framework
            )
            
            context = GenerationContext(
                project_info={"topic": topic, "comparison_mode": True}
            )
            
            result = self.generate_documentation(request, context)
            comparison_docs[framework.value] = result
            
        # Generate unified comparison
        unified_comparison = self._create_unified_comparison(comparison_docs, topic)
        
        return {
            "topic": topic,
            "individual_approaches": comparison_docs,
            "unified_comparison": unified_comparison,
            "recommendations": self._generate_framework_recommendations(topic)
        }
        
    def generate_interactive_cookbook(self, 
                                   recipes: List[Dict[str, Any]],
                                   personality_profiles: List[Dict[str, Any]] = None) -> str:
        """Generate interactive cookbook using PhiData patterns."""
        from .recipe_based_learning import RecipeBasedLearning
        from .cookbook_organization_manager import CookbookOrganizationManager
        
        # Initialize cookbook systems
        recipe_system = RecipeBasedLearning()
        cookbook_manager = CookbookOrganizationManager()
        
        # Create personalities if provided
        if personality_profiles:
            for profile in personality_profiles:
                cookbook_manager.create_personality_profile(**profile)
                
        # Process recipes
        for recipe_data in recipes:
            recipe_system.create_recipe(**recipe_data)
            
        # Generate cookbook structure
        cookbook_html = self._generate_interactive_cookbook_html(recipe_system, cookbook_manager)
        
        return cookbook_html
        
    def generate_api_documentation(self, 
                                 code_analysis: Dict[str, Any],
                                 frameworks: List[DocumentationFramework] = None) -> Dict[str, str]:
        """Generate comprehensive API documentation."""
        from .auto_api_docs_generator import AutoApiDocsGenerator
        from .hierarchical_docs_organizer import HierarchicalDocsOrganizer
        
        if not frameworks:
            frameworks = [DocumentationFramework.AUTOGEN, DocumentationFramework.AGENCY_SWARM]
            
        api_docs = {}
        
        for framework in frameworks:
            # Use framework-specific patterns
            if framework == DocumentationFramework.AUTOGEN:
                generator = AutoApiDocsGenerator(code_analysis.get("language", "python"))
                docs = generator.generate_documentation([])
                api_docs[f"{framework.value}_api"] = docs
                
        return api_docs
        
    def generate_production_deployment_guide(self, deployment_config: Dict[str, Any]) -> str:
        """Generate production deployment guide using LLama-Agents patterns."""
        from .production_ready_docs import ProductionReadyDocs
        from .service_oriented_arch_docs import ServiceOrientedArchDocs
        
        # Initialize production documentation systems
        prod_docs = ProductionReadyDocs()
        soa_docs = ServiceOrientedArchDocs()
        
        # Create production configuration
        prod_config = prod_docs.create_production_config(**deployment_config)
        
        # Generate comprehensive guide
        overview = prod_docs.generate_production_overview()
        architecture = soa_docs.generate_architecture_overview()
        
        guide = f"""# Production Deployment Guide

{overview}

## Architecture Overview

{architecture}

## Deployment Configurations

{self._format_deployment_configs(prod_config)}

## Monitoring and Observability

{self._generate_monitoring_section()}
"""
        
        return guide
        
    def _initialize_generators(self) -> Dict[str, Any]:
        """Initialize all documentation generators from Phase 1."""
        generators = {}
        
        # Import and initialize generators from each framework
        try:
            # Agency-Swarm generators
            from .agency_swarm_analyzer import AgencySwarmAnalyzer
            from .mdx_generator import MDXGenerator
            generators["agency_swarm"] = {
                "analyzer": AgencySwarmAnalyzer,
                "mdx": MDXGenerator
            }
            
            # CrewAI generators
            from .multilingual_docs import SupportedLanguage
            from .enterprise_api_docs import EnterpriseAPIDocsSystem
            generators["crewai"] = {
                "multilingual": SupportedLanguage,
                "enterprise_api": EnterpriseAPIDocsSystem
            }
            
            # AgentScope generators
            from .changelog_generator import ChangelogGenerator
            from .roadmap_generator import RoadmapGenerator
            generators["agentscope"] = {
                "changelog": ChangelogGenerator,
                "roadmap": RoadmapGenerator
            }
            
            # Additional generators from other frameworks...
            
        except ImportError as e:
            logger.warning(f"Could not import generator: {e}")
            
        return generators
        
    def _load_unified_templates(self) -> Dict[str, str]:
        """Load unified templates combining all framework patterns."""
        return {
            "api_reference": """# {title} API Reference

## Overview
{overview}

## Classes
{classes_section}

## Functions  
{functions_section}

## Examples
{examples_section}
""",
            "tutorial": """# {title} Tutorial

## What You'll Learn
{learning_objectives}

## Prerequisites
{prerequisites}

## Step-by-Step Guide
{tutorial_steps}

## Troubleshooting
{troubleshooting}
""",
            "cookbook": """# {title} Cookbook

## Recipe Collection
{recipe_index}

## Recipes
{recipe_sections}

## Additional Resources
{resources}
""",
            "architecture": """# {title} Architecture

## System Overview
{system_overview}

## Components
{components_section}

## Design Decisions
{design_decisions}

## Deployment Architecture
{deployment_section}
"""
        }
        
    def _load_framework_patterns(self) -> Dict[DocumentationFramework, Dict[str, Any]]:
        """Load patterns specific to each framework."""
        return {
            DocumentationFramework.AGENCY_SWARM: {
                "strengths": ["MDX integration", "Component-based docs", "Interactive examples"],
                "best_for": ["API documentation", "Interactive tutorials", "Component libraries"],
                "patterns": ["hierarchical_organization", "progressive_disclosure"]
            },
            DocumentationFramework.CREWAI: {
                "strengths": ["Multilingual support", "Enterprise features", "Professional styling"],
                "best_for": ["Enterprise documentation", "Multi-language projects", "Professional APIs"],
                "patterns": ["enterprise_templates", "multi_language_workflows"]
            },
            DocumentationFramework.AGENTSCOPE: {
                "strengths": ["Bilingual documentation", "Example-driven learning", "Testing integration"],
                "best_for": ["Tutorial documentation", "Testing guides", "Development workflows"],
                "patterns": ["example_driven", "progressive_complexity"]
            },
            DocumentationFramework.AUTOGEN: {
                "strengths": ["Progressive disclosure", "Multi-platform support", "Design-first approach"],
                "best_for": ["Complex systems", "Multi-platform documentation", "Architecture guides"],
                "patterns": ["progressive_disclosure", "platform_specific", "design_first"]
            },
            DocumentationFramework.LLAMA_AGENTS: {
                "strengths": ["Production deployment", "Interactive CLI", "Service-oriented docs"],
                "best_for": ["Production guides", "Deployment documentation", "Service architectures"],
                "patterns": ["production_ready", "service_oriented", "interactive_cli"]
            },
            DocumentationFramework.PHIDATA: {
                "strengths": ["Recipe-based learning", "Personality-driven examples", "Multi-modal content"],
                "best_for": ["Cookbook documentation", "Learning materials", "Example collections"],
                "patterns": ["recipe_based", "personality_driven", "multi_modal"]
            },
            DocumentationFramework.SWARMS: {
                "strengths": ["AI-powered generation", "Self-healing docs", "Intelligence systems"],
                "best_for": ["Auto-generated docs", "Intelligent maintenance", "Swarm architectures"],
                "patterns": ["ai_powered", "self_healing", "intelligence_driven"]
            }
        }
        
    def _select_framework_pattern(self, 
                                request: DocumentationRequest, 
                                context: GenerationContext) -> DocumentationFramework:
        """Select optimal framework pattern based on request and context."""
        if request.source_framework:
            return request.source_framework
            
        # Scoring system to select best framework
        scores = {}
        
        for framework, patterns in self.patterns.items():
            score = 0
            
            # Score based on generation type
            if request.generation_type == GenerationType.API_REFERENCE:
                if "API documentation" in patterns["best_for"]:
                    score += 3
            elif request.generation_type == GenerationType.COOKBOOK:
                if "Cookbook documentation" in patterns["best_for"]:
                    score += 3
            elif request.generation_type == GenerationType.TUTORIAL:
                if "Tutorial documentation" in patterns["best_for"]:
                    score += 3
                    
            # Score based on complexity
            if request.complexity_level >= 4 and "Complex systems" in patterns["best_for"]:
                score += 2
                
            # Score based on interactivity
            if request.include_interactive and "Interactive" in str(patterns["strengths"]):
                score += 2
                
            # Score based on multilingual requirements
            if len(request.languages) > 1 and "Multilingual" in str(patterns["strengths"]):
                score += 2
                
            scores[framework] = score
            
        # Return framework with highest score
        best_framework = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected framework: {best_framework.value} (score: {scores[best_framework]})")
        
        return best_framework
        
    def _get_generator(self, generation_type: GenerationType, framework: DocumentationFramework):
        """Get appropriate generator for type and framework."""
        # Return mock generator for now - would be actual generators in production
        class MockGenerator:
            def generate(self, request, context):
                return f"Generated {generation_type.value} documentation using {framework.value} patterns"
                
        return MockGenerator()
        
    def _apply_enhancements(self, content: str, request: DocumentationRequest, context: GenerationContext) -> str:
        """Apply enhancements based on request parameters."""
        enhanced = content
        
        if request.include_examples:
            enhanced += "\\n\\n## Examples\\n\\nCode examples would be added here."
            
        if request.include_interactive:
            enhanced += "\\n\\n## Interactive Elements\\n\\nInteractive components would be added here."
            
        if len(request.languages) > 1:
            enhanced += f"\\n\\n## Multilingual Support\\n\\nAvailable in: {', '.join(request.languages)}"
            
        return enhanced
        
    def _format_output(self, content: str, output_format: OutputFormat) -> str:
        """Format content for specified output format."""
        if output_format == OutputFormat.MARKDOWN:
            return content
        elif output_format == OutputFormat.HTML:
            return f"<html><body>{content}</body></html>"  # Simplified
        elif output_format == OutputFormat.MDX:
            return f"---\\ncomponents:\\n  - Interactive\\n---\\n\\n{content}"
        else:
            return content  # Default to original format
            
    def _generate_metadata(self, request: DocumentationRequest, context: GenerationContext, framework: DocumentationFramework) -> Dict[str, Any]:
        """Generate metadata for the documentation."""
        return {
            "generation_type": request.generation_type.value,
            "framework_used": framework.value,
            "target_audience": request.target_audience,
            "complexity_level": request.complexity_level,
            "languages": request.languages,
            "generated_at": "2025-01-21T00:00:00Z",  # Would use actual timestamp
            "generator_version": "1.0.0"
        }
        
    def _get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generation process."""
        return {
            "total_generators_available": len(self.generators),
            "patterns_loaded": sum(len(p["patterns"]) for p in self.patterns.values()),
            "templates_available": len(self.templates)
        }
        
    def _save_generated_docs(self, result: Dict[str, Any], request: DocumentationRequest) -> None:
        """Save generated documentation to output directory."""
        filename = f"{request.title.lower().replace(' ', '_')}.{request.output_format.value}"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if request.output_format in [OutputFormat.MARKDOWN, OutputFormat.MDX, OutputFormat.HTML]:
                f.write(result["content"])
            else:
                json.dump(result, f, indent=2)
                
        logger.info(f"Saved documentation to {output_path}")
        
    def _create_unified_comparison(self, comparison_docs: Dict[str, Any], topic: str) -> str:
        """Create unified comparison across all frameworks."""
        comparison = [
            f"# {topic} - Framework Comparison",
            "",
            "Comparative analysis across all major AI documentation frameworks.",
            "",
            "## Framework Approaches",
            ""
        ]
        
        for framework, docs in comparison_docs.items():
            comparison.extend([
                f"### {framework.replace('_', ' ').title()}",
                "",
                f"**Approach:** {docs['metadata']['framework_used']}",
                f"**Strengths:** {self.patterns[DocumentationFramework(framework)]['strengths']}",
                ""
            ])
            
        return "\\n".join(comparison)
        
    def _generate_framework_recommendations(self, topic: str) -> List[str]:
        """Generate recommendations for framework selection."""
        return [
            f"For {topic}, consider framework strengths and project requirements",
            "Agency-Swarm excels at component-based documentation",
            "CrewAI provides excellent multilingual support", 
            "AgentScope offers great example-driven learning",
            "AutoGen provides comprehensive multi-platform support",
            "LLama-Agents excels at production deployment guides",
            "PhiData offers superior cookbook-style documentation",
            "Swarms provides AI-powered intelligent documentation"
        ]
        
    def _generate_interactive_cookbook_html(self, recipe_system, cookbook_manager) -> str:
        """Generate interactive HTML cookbook."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Interactive Cookbook</title>
    <style>
        .recipe { margin: 20px; padding: 15px; border: 1px solid #ddd; }
        .personality { background: #f0f8ff; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Interactive Recipe Cookbook</h1>
    <div id="recipes">
        <!-- Recipes would be dynamically generated here -->
    </div>
</body>
</html>"""

    def _format_deployment_configs(self, config) -> str:
        """Format deployment configuration for documentation."""
        return f"""
## Configuration Details

**Environment:** Production
**Services:** {getattr(config, 'services', 'Multiple services')}
**Monitoring:** Enabled
**Scaling:** Auto-scaling configured
"""

    def _generate_monitoring_section(self) -> str:
        """Generate monitoring and observability section."""
        return """
### Monitoring Stack

- **Metrics Collection:** Prometheus
- **Visualization:** Grafana
- **Alerting:** Alert Manager
- **Logging:** ELK Stack
- **Tracing:** Jaeger
"""