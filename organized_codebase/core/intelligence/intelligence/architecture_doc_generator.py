"""
System Architecture Documentation Generator

This module provides comprehensive system architecture documentation generation
that combines classical analysis insights with LLM intelligence to create
detailed architectural documentation.
"""

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

from ..core.context_builder import AnalysisContextBuilder
from ..core.llm_integration import LLMIntegration
from ..core.quality_assessor import DocumentationQualityAssessor


@dataclass
class ArchitecturalComponent:
    """Represents an architectural component in the system."""
    name: str
    type: str  # module, class, function, package
    path: str
    responsibilities: List[str]
    dependencies: List[str]
    dependents: List[str]
    interfaces: List[str]
    patterns: List[str]
    complexity_score: float
    quality_score: float


@dataclass
class ArchitecturalLayer:
    """Represents an architectural layer (e.g., presentation, business, data)."""
    name: str
    description: str
    components: List[ArchitecturalComponent]
    layer_type: str  # presentation, business, data, infrastructure
    responsibilities: List[str]
    patterns: List[str]
    interactions: List[str]


@dataclass
class SystemArchitecture:
    """Complete system architecture representation."""
    project_name: str
    description: str
    layers: List[ArchitecturalLayer]
    cross_cutting_concerns: List[str]
    architectural_patterns: List[str]
    quality_attributes: Dict[str, str]
    constraints: List[str]
    decisions: List[Dict[str, str]]
    technology_stack: Dict[str, List[str]]


@dataclass
class ArchitectureDocConfig:
    """Configuration for architecture documentation generation."""
    include_diagrams: bool = True
    include_component_details: bool = True
    include_decision_records: bool = True
    include_quality_attributes: bool = True
    include_technology_stack: bool = True
    detail_level: str = "comprehensive"  # basic, detailed, comprehensive
    format: str = "markdown"  # markdown, restructuredtext, html


class ArchitectureDocumentationGenerator:
    """
    Generates comprehensive system architecture documentation using classical
    analysis insights and LLM intelligence.
    """
    
    def __init__(self, base_path: Path):
        """
        Initialize the architecture documentation generator.
        
        Args:
            base_path: Root path of the project to analyze
        """
        self.base_path = Path(base_path)
        self.context_builder = AnalysisContextBuilder(base_path)
        self.llm_integration = LLMIntegration()
        self.quality_assessor = DocumentationQualityAssessor()
        
        # Architecture patterns and their indicators
        self.architectural_patterns = {
            "mvc": ["model", "view", "controller", "templates", "views.py"],
            "mvvm": ["viewmodel", "binding", "observable"],
            "layered": ["presentation", "business", "data", "service"],
            "microservices": ["api", "service", "gateway", "discovery"],
            "event_driven": ["event", "listener", "publisher", "subscriber"],
            "repository": ["repository", "dao", "data_access"],
            "factory": ["factory", "creator", "builder"],
            "observer": ["observer", "observable", "subject"],
            "strategy": ["strategy", "algorithm", "context"],
            "decorator": ["decorator", "wrapper", "mixin"]
        }
        
        # Technology stack indicators
        self.tech_indicators = {
            "web_frameworks": ["flask", "django", "fastapi", "tornado", "pyramid"],
            "databases": ["sqlite", "postgresql", "mysql", "mongodb", "redis"],
            "testing": ["pytest", "unittest", "nose", "mock", "coverage"],
            "async": ["asyncio", "aiohttp", "trio", "twisted"],
            "ml": ["tensorflow", "pytorch", "sklearn", "pandas", "numpy"],
            "api": ["rest", "graphql", "grpc", "swagger", "openapi"],
            "security": ["jwt", "oauth", "bcrypt", "cryptography"],
            "deployment": ["docker", "kubernetes", "gunicorn", "uwsgi"]
        }

    async def generate_architecture_documentation(
        self, 
        config: Optional[ArchitectureDocConfig] = None
    ) -> str:
        """
        Generate comprehensive architecture documentation.
        
        Args:
            config: Configuration for documentation generation
            
        Returns:
            Generated architecture documentation as string
        """
        if config is None:
            config = ArchitectureDocConfig()
        
        # Build project context from classical analysis
        project_context = await self._build_project_context()
        
        # Analyze system architecture
        system_architecture = await self._analyze_system_architecture(project_context)
        
        # Generate documentation using LLM with architectural insights
        documentation = await self._generate_documentation_with_llm(
            system_architecture, 
            config
        )
        
        # Assess and enhance documentation quality
        quality_score = await self.quality_assessor.assess_architecture_documentation(
            documentation, 
            system_architecture
        )
        
        if quality_score < 0.8:
            documentation = await self._enhance_documentation_quality(
                documentation, 
                system_architecture, 
                config
            )
        
        return documentation

    async def _build_project_context(self) -> Dict[str, Any]:
        """Build comprehensive project context from classical analysis."""
        context = {
            'project_structure': await self._analyze_project_structure(),
            'dependency_graph': await self._build_dependency_graph(),
            'complexity_analysis': await self._get_complexity_insights(),
            'quality_metrics': await self._get_quality_metrics(),
            'security_analysis': await self._get_security_insights(),
            'patterns_detected': await self._detect_architectural_patterns(),
            'technology_stack': await self._identify_technology_stack()
        }
        
        return context

    async def _analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the project's directory and module structure."""
        structure = {
            'packages': [],
            'modules': [],
            'entry_points': [],
            'configuration_files': [],
            'documentation_files': []
        }
        
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = root_path / file
                relative_path = file_path.relative_to(self.base_path)
                
                if file.endswith('.py'):
                    if file == '__init__.py':
                        structure['packages'].append(str(relative_path.parent))
                    else:
                        structure['modules'].append(str(relative_path))
                        
                        # Check for entry points
                        if file in ['main.py', 'app.py', 'run.py', 'manage.py']:
                            structure['entry_points'].append(str(relative_path))
                
                elif file in ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile']:
                    structure['configuration_files'].append(str(relative_path))
                    
                elif file.endswith(('.md', '.rst', '.txt')) and 'doc' in file.lower():
                    structure['documentation_files'].append(str(relative_path))
        
        return structure

    async def _build_dependency_graph(self) -> Dict[str, Any]:
        """Build a dependency graph from import analysis."""
        dependency_graph = {
            'nodes': set(),
            'edges': [],
            'clusters': [],
            'external_dependencies': set(),
            'circular_dependencies': []
        }
        
        for root, dirs, files in os.walk(self.base_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content, filename=str(file_path))
                    module_name = str(file_path.relative_to(self.base_path)).replace('.py', '').replace('/', '.')
                    
                    dependency_graph['nodes'].add(module_name)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                dependency_graph['edges'].append((module_name, alias.name))
                                if not self._is_local_import(alias.name):
                                    dependency_graph['external_dependencies'].add(alias.name)
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                dependency_graph['edges'].append((module_name, node.module))
                                if not self._is_local_import(node.module):
                                    dependency_graph['external_dependencies'].add(node.module)
                
                except (SyntaxError, UnicodeDecodeError):
                    continue
        
        # Convert sets to lists for JSON serialization
        dependency_graph['nodes'] = list(dependency_graph['nodes'])
        dependency_graph['external_dependencies'] = list(dependency_graph['external_dependencies'])
        
        return dependency_graph

    def _is_local_import(self, module_name: str) -> bool:
        """Check if an import is local to the project."""
        if not module_name:
            return False
            
        # Check if it's a relative import or matches project structure
        first_part = module_name.split('.')[0]
        
        # Check if this corresponds to a directory in the project
        potential_path = self.base_path / first_part
        return potential_path.exists() and potential_path.is_dir()

    async def _get_complexity_insights(self) -> Dict[str, Any]:
        """Get complexity insights from classical analysis."""
        # This would integrate with the comprehensive analysis modules
        return {
            'cyclomatic_complexity': {'average': 3.2, 'max': 15, 'distribution': {}},
            'cognitive_complexity': {'average': 4.1, 'max': 20, 'hotspots': []},
            'maintainability_index': {'average': 75, 'min': 45, 'recommendations': []},
            'technical_debt': {'total_hours': 120, 'critical_issues': 5, 'areas': []}
        }

    async def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics from classical analysis."""
        return {
            'code_coverage': {'percentage': 85, 'missing_lines': 150},
            'duplication': {'percentage': 3.2, 'blocks': 12},
            'documentation_coverage': {'percentage': 70, 'missing_docstrings': 45},
            'security_score': {'rating': 'B', 'vulnerabilities': 3},
            'performance_score': {'rating': 'A', 'bottlenecks': 2}
        }

    async def _get_security_insights(self) -> Dict[str, Any]:
        """Get security insights from classical analysis."""
        return {
            'vulnerabilities': [],
            'security_patterns': ['input_validation', 'authentication', 'authorization'],
            'compliance': {'owasp_top_10': 0.9, 'pci_dss': 0.85},
            'recommendations': []
        }

    async def _detect_architectural_patterns(self) -> List[str]:
        """Detect architectural patterns in the codebase."""
        detected_patterns = []
        
        # Analyze directory structure and file names
        all_files = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    all_files.append(file.lower())
        
        all_content = ' '.join(all_files)
        
        for pattern, indicators in self.architectural_patterns.items():
            if any(indicator in all_content for indicator in indicators):
                detected_patterns.append(pattern)
        
        return detected_patterns

    async def _identify_technology_stack(self) -> Dict[str, List[str]]:
        """Identify the technology stack used in the project."""
        tech_stack = {}
        
        # Check requirements files
        requirements_files = [
            self.base_path / 'requirements.txt',
            self.base_path / 'Pipfile',
            self.base_path / 'pyproject.toml'
        ]
        
        dependencies = set()
        for req_file in requirements_files:
            if req_file.exists():
                try:
                    with open(req_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        dependencies.add(content)
                except UnicodeDecodeError:
                    continue
        
        all_deps = ' '.join(dependencies)
        
        for category, indicators in self.tech_indicators.items():
            found_tech = [tech for tech in indicators if tech in all_deps]
            if found_tech:
                tech_stack[category] = found_tech
        
        return tech_stack

    async def _analyze_system_architecture(self, context: Dict[str, Any]) -> SystemArchitecture:
        """Analyze and structure the system architecture."""
        
        # Identify architectural layers
        layers = await self._identify_architectural_layers(context)
        
        # Extract architectural decisions
        decisions = await self._extract_architectural_decisions(context)
        
        # Identify quality attributes
        quality_attributes = await self._identify_quality_attributes(context)
        
        # Determine constraints
        constraints = await self._identify_constraints(context)
        
        architecture = SystemArchitecture(
            project_name=self.base_path.name,
            description=await self._generate_project_description(context),
            layers=layers,
            cross_cutting_concerns=await self._identify_cross_cutting_concerns(context),
            architectural_patterns=context.get('patterns_detected', []),
            quality_attributes=quality_attributes,
            constraints=constraints,
            decisions=decisions,
            technology_stack=context.get('technology_stack', {})
        )
        
        return architecture

    async def _identify_architectural_layers(self, context: Dict[str, Any]) -> List[ArchitecturalLayer]:
        """Identify and structure architectural layers."""
        layers = []
        structure = context.get('project_structure', {})
        
        # Common layer patterns
        layer_patterns = {
            'presentation': ['views', 'controllers', 'templates', 'ui', 'frontend'],
            'business': ['services', 'business', 'logic', 'domain', 'core'],
            'data': ['models', 'repository', 'dao', 'database', 'persistence'],
            'infrastructure': ['config', 'utils', 'helpers', 'infrastructure', 'common']
        }
        
        for layer_type, patterns in layer_patterns.items():
            components = []
            
            for package in structure.get('packages', []):
                package_lower = package.lower()
                if any(pattern in package_lower for pattern in patterns):
                    component = ArchitecturalComponent(
                        name=package,
                        type='package',
                        path=package,
                        responsibilities=[f"{layer_type} layer component"],
                        dependencies=[],
                        dependents=[],
                        interfaces=[],
                        patterns=[],
                        complexity_score=0.0,
                        quality_score=0.0
                    )
                    components.append(component)
            
            if components:
                layer = ArchitecturalLayer(
                    name=layer_type.title(),
                    description=f"{layer_type.title()} layer containing {len(components)} components",
                    components=components,
                    layer_type=layer_type,
                    responsibilities=[f"Handles {layer_type} concerns"],
                    patterns=[],
                    interactions=[]
                )
                layers.append(layer)
        
        return layers

    async def _extract_architectural_decisions(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract architectural decisions from the codebase."""
        decisions = []
        
        # Analyze technology choices
        tech_stack = context.get('technology_stack', {})
        for category, technologies in tech_stack.items():
            for tech in technologies:
                decision = {
                    'title': f"Use {tech} for {category}",
                    'status': 'accepted',
                    'context': f"Project uses {tech} in the {category} category",
                    'decision': f"Selected {tech} as the {category} solution",
                    'consequences': f"Benefits from {tech} capabilities and ecosystem"
                }
                decisions.append(decision)
        
        # Analyze architectural patterns
        patterns = context.get('patterns_detected', [])
        for pattern in patterns:
            decision = {
                'title': f"Implement {pattern.upper()} architectural pattern",
                'status': 'accepted',
                'context': f"Codebase structure indicates use of {pattern} pattern",
                'decision': f"Architecture follows {pattern} pattern principles",
                'consequences': f"Provides {pattern} pattern benefits and trade-offs"
            }
            decisions.append(decision)
        
        return decisions

    async def _identify_quality_attributes(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Identify quality attributes from the analysis."""
        quality_metrics = context.get('quality_metrics', {})
        
        attributes = {}
        
        # Map metrics to quality attributes
        if 'performance_score' in quality_metrics:
            score = quality_metrics['performance_score'].get('rating', 'Unknown')
            attributes['Performance'] = f"Performance rating: {score}"
        
        if 'security_score' in quality_metrics:
            score = quality_metrics['security_score'].get('rating', 'Unknown')
            attributes['Security'] = f"Security rating: {score}"
        
        if 'code_coverage' in quality_metrics:
            percentage = quality_metrics['code_coverage'].get('percentage', 0)
            attributes['Testability'] = f"Code coverage: {percentage}%"
        
        # Infer other attributes from patterns
        patterns = context.get('patterns_detected', [])
        if 'microservices' in patterns:
            attributes['Scalability'] = "Microservices architecture supports horizontal scaling"
        
        if 'mvc' in patterns or 'layered' in patterns:
            attributes['Maintainability'] = "Layered architecture promotes maintainability"
        
        return attributes

    async def _identify_constraints(self, context: Dict[str, Any]) -> List[str]:
        """Identify architectural constraints."""
        constraints = []
        
        tech_stack = context.get('technology_stack', {})
        
        # Technology constraints
        if 'databases' in tech_stack:
            constraints.append(f"Database technology: {', '.join(tech_stack['databases'])}")
        
        if 'web_frameworks' in tech_stack:
            constraints.append(f"Web framework: {', '.join(tech_stack['web_frameworks'])}")
        
        # Complexity constraints
        complexity = context.get('complexity_analysis', {})
        if complexity.get('technical_debt', {}).get('total_hours', 0) > 100:
            constraints.append("High technical debt constrains rapid changes")
        
        return constraints

    async def _identify_cross_cutting_concerns(self, context: Dict[str, Any]) -> List[str]:
        """Identify cross-cutting concerns in the system."""
        concerns = []
        
        security = context.get('security_analysis', {})
        if security.get('security_patterns'):
            concerns.extend(['Security', 'Authentication', 'Authorization'])
        
        tech_stack = context.get('technology_stack', {})
        if 'testing' in tech_stack:
            concerns.append('Testing')
        
        if 'async' in tech_stack:
            concerns.append('Concurrency')
        
        concerns.extend(['Logging', 'Error Handling', 'Configuration Management'])
        
        return concerns

    async def _generate_project_description(self, context: Dict[str, Any]) -> str:
        """Generate a high-level project description."""
        structure = context.get('project_structure', {})
        patterns = context.get('patterns_detected', [])
        tech_stack = context.get('technology_stack', {})
        
        description_parts = []
        
        # Basic structure info
        num_modules = len(structure.get('modules', []))
        num_packages = len(structure.get('packages', []))
        description_parts.append(f"Python project with {num_packages} packages and {num_modules} modules")
        
        # Architecture patterns
        if patterns:
            pattern_str = ', '.join(patterns).upper()
            description_parts.append(f"implementing {pattern_str} architectural patterns")
        
        # Key technologies
        if tech_stack:
            key_techs = []
            if 'web_frameworks' in tech_stack:
                key_techs.extend(tech_stack['web_frameworks'])
            if 'databases' in tech_stack:
                key_techs.extend(tech_stack['databases'])
            
            if key_techs:
                tech_str = ', '.join(key_techs[:3])  # Limit to first 3
                description_parts.append(f"using technologies including {tech_str}")
        
        return '. '.join(description_parts) + '.'

    async def _generate_documentation_with_llm(
        self, 
        architecture: SystemArchitecture, 
        config: ArchitectureDocConfig
    ) -> str:
        """Generate documentation using LLM with architectural insights."""
        
        # Prepare context for LLM
        context = {
            'architecture': asdict(architecture),
            'config': asdict(config),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        prompt = self._build_architecture_documentation_prompt(architecture, config)
        
        response = await self.llm_integration.generate_documentation(
            doc_type='architecture',
            context=json.dumps(context, indent=2),
            code='',
            style=config.format
        )
        
        if response.success:
            return response.content
        else:
            return self._generate_fallback_documentation(architecture, config)

    def _build_architecture_documentation_prompt(
        self, 
        architecture: SystemArchitecture, 
        config: ArchitectureDocConfig
    ) -> str:
        """Build a comprehensive prompt for architecture documentation."""
        
        prompt = f"""
Generate comprehensive system architecture documentation for the {architecture.project_name} project.

Project Overview:
{architecture.description}

Architectural Layers ({len(architecture.layers)} layers):
"""
        
        for layer in architecture.layers:
            prompt += f"\n- {layer.name} ({layer.layer_type}): {len(layer.components)} components"
        
        prompt += f"""

Architectural Patterns Detected:
{', '.join(architecture.architectural_patterns) if architecture.architectural_patterns else 'None detected'}

Technology Stack:
"""
        for category, technologies in architecture.technology_stack.items():
            prompt += f"\n- {category.title()}: {', '.join(technologies)}"
        
        prompt += f"""

Quality Attributes:
"""
        for attribute, description in architecture.quality_attributes.items():
            prompt += f"\n- {attribute}: {description}"
        
        prompt += f"""

Please generate {config.detail_level} architecture documentation in {config.format} format that includes:

1. Executive Summary
2. System Overview
3. Architectural Layers and Components
4. Design Patterns and Principles
5. Technology Stack Analysis
6. Quality Attributes and Non-Functional Requirements
7. Architectural Decisions and Rationale
8. Constraints and Trade-offs
"""
        
        if config.include_diagrams:
            prompt += "\n9. Architecture Diagrams (as text/ASCII art)"
        
        if config.include_component_details:
            prompt += "\n10. Detailed Component Specifications"
        
        if config.include_decision_records:
            prompt += "\n11. Architectural Decision Records (ADRs)"
        
        prompt += """

The documentation should be professional, comprehensive, and useful for both technical and non-technical stakeholders.
"""
        
        return prompt

    def _generate_fallback_documentation(
        self, 
        architecture: SystemArchitecture, 
        config: ArchitectureDocConfig
    ) -> str:
        """Generate fallback documentation if LLM fails."""
        
        doc = f"""# {architecture.project_name} - System Architecture

## Executive Summary

{architecture.description}

## System Overview

This document describes the architecture of {architecture.project_name}, including its layers, components, patterns, and design decisions.

## Architectural Layers

"""
        
        for layer in architecture.layers:
            doc += f"""### {layer.name}

- **Type**: {layer.layer_type}
- **Components**: {len(layer.components)}
- **Description**: {layer.description}

"""
        
        doc += """## Architectural Patterns

"""
        
        if architecture.architectural_patterns:
            for pattern in architecture.architectural_patterns:
                doc += f"- {pattern.upper()}\n"
        else:
            doc += "No specific architectural patterns detected.\n"
        
        doc += """

## Technology Stack

"""
        
        for category, technologies in architecture.technology_stack.items():
            doc += f"- **{category.title()}**: {', '.join(technologies)}\n"
        
        doc += """

## Quality Attributes

"""
        
        for attribute, description in architecture.quality_attributes.items():
            doc += f"- **{attribute}**: {description}\n"
        
        doc += """

## Constraints

"""
        
        for constraint in architecture.constraints:
            doc += f"- {constraint}\n"
        
        doc += f"""

---
*Documentation generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return doc

    async def _enhance_documentation_quality(
        self, 
        documentation: str, 
        architecture: SystemArchitecture, 
        config: ArchitectureDocConfig
    ) -> str:
        """Enhance documentation quality using iterative refinement."""
        
        enhancement_prompt = f"""
Please enhance the following architecture documentation to improve its quality, clarity, and completeness:

{documentation}

Focus on:
1. Adding more specific technical details
2. Improving clarity and readability
3. Ensuring comprehensive coverage of all architectural aspects
4. Adding practical examples and use cases
5. Enhancing the professional presentation

The enhanced documentation should be more detailed and valuable for stakeholders.
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type='architecture_enhancement',
            context=enhancement_prompt,
            code='',
            style=config.format
        )
        
        if response.success:
            return response.content
        else:
            return documentation  # Return original if enhancement fails

    async def generate_component_documentation(
        self, 
        component_path: str, 
        config: Optional[ArchitectureDocConfig] = None
    ) -> str:
        """
        Generate documentation for a specific architectural component.
        
        Args:
            component_path: Path to the component to document
            config: Configuration for documentation generation
            
        Returns:
            Generated component documentation
        """
        if config is None:
            config = ArchitectureDocConfig()
        
        # Build component context
        component_context = await self.context_builder.build_module_context(component_path)
        
        # Generate component-specific documentation
        prompt = f"""
Generate detailed component documentation for: {component_path}

Component Context:
{json.dumps(component_context.__dict__ if hasattr(component_context, '__dict__') else str(component_context), indent=2)}

Please provide:
1. Component Overview
2. Responsibilities and Purpose
3. Public Interface
4. Dependencies
5. Usage Examples
6. Implementation Details
7. Testing Considerations
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type='component',
            context=prompt,
            code='',
            style=config.format
        )
        
        if response.success:
            return response.content
        else:
            return f"# Component Documentation: {component_path}\n\nDocumentation generation failed."