"""
README.md Generator

Intelligent README.md generator that creates comprehensive project documentation
based on classical analysis insights and project structure understanding.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..core.context_builder import AnalysisContextBuilder, ProjectContext
from ..core.llm_integration import LLMIntegration
from ..core.quality_assessor import DocumentationQualityAssessor

logger = logging.getLogger(__name__)


@dataclass
class ReadmeConfig:
    """Configuration for README generation."""
    include_badges: bool = True
    include_installation: bool = True
    include_usage_examples: bool = True
    include_api_reference: bool = True
    include_contributing: bool = True
    include_license: bool = True
    include_architecture_overview: bool = True
    include_performance_notes: bool = True
    include_security_notes: bool = True
    project_logo_path: Optional[str] = None
    custom_sections: List[Dict[str, str]] = None


class ReadmeGenerator:
    """
    Intelligent README.md generator with comprehensive project analysis.
    
    Features:
    - Project structure analysis and documentation
    - Technology stack detection and documentation
    - Installation instruction generation
    - Usage example creation from analysis
    - API reference generation
    - Architecture overview from structural analysis
    - Performance and security insights integration
    - Customizable sections and styling
    """
    
    def __init__(self, 
                 llm_integration: LLMIntegration,
                 context_builder: AnalysisContextBuilder,
                 quality_assessor: Optional[DocumentationQualityAssessor] = None):
        """
        Initialize the README generator.
        
        Args:
            llm_integration: LLM integration for generation
            context_builder: Analysis context builder
            quality_assessor: Quality assessor for validation
        """
        self.llm_integration = llm_integration
        self.context_builder = context_builder
        self.quality_assessor = quality_assessor
        
    async def generate_readme(self, 
                            project_path: str,
                            config: Optional[ReadmeConfig] = None) -> str:
        """
        Generate a comprehensive README.md for a project.
        
        Args:
            project_path: Path to the project directory
            config: README generation configuration
            
        Returns:
            str: Generated README.md content
        """
        logger.info(f"Generating README.md for project: {project_path}")
        
        config = config or ReadmeConfig()
        
        # Build project context
        project_context = self.context_builder.build_project_context(project_path)
        
        # Generate README using structured approach
        readme_sections = await self._generate_readme_sections(project_context, config)
        
        # Assemble final README
        readme_content = self._assemble_readme(readme_sections, config)
        
        # Quality assessment and improvement
        if self.quality_assessor:
            quality_score = await self.quality_assessor.assess_readme_quality(
                readme_content, project_context
            )
            logger.info(f"README quality score: {quality_score:.2f}")
            
            if quality_score < 0.7:
                logger.info("Quality below threshold, enhancing README...")
                readme_content = await self._enhance_readme_quality(
                    readme_content, project_context, config
                )
        
        logger.info(f"Generated README.md for {project_context.name}")
        return readme_content
        
    async def _generate_readme_sections(self, 
                                      project_context: ProjectContext,
                                      config: ReadmeConfig) -> Dict[str, str]:
        """Generate individual README sections."""
        sections = {}
        
        # Project header and description
        sections["header"] = await self._generate_header_section(project_context, config)
        sections["description"] = await self._generate_description_section(project_context)
        
        # Badges (if enabled)
        if config.include_badges:
            sections["badges"] = self._generate_badges_section(project_context)
        
        # Table of contents
        sections["toc"] = self._generate_toc_section(config)
        
        # Features section
        sections["features"] = await self._generate_features_section(project_context)
        
        # Installation (if enabled)
        if config.include_installation:
            sections["installation"] = await self._generate_installation_section(project_context)
        
        # Usage examples (if enabled)
        if config.include_usage_examples:
            sections["usage"] = await self._generate_usage_section(project_context)
        
        # API reference (if enabled)
        if config.include_api_reference:
            sections["api_reference"] = await self._generate_api_reference_section(project_context)
        
        # Architecture overview (if enabled)
        if config.include_architecture_overview:
            sections["architecture"] = await self._generate_architecture_section(project_context)
        
        # Performance notes (if enabled)
        if config.include_performance_notes:
            sections["performance"] = await self._generate_performance_section(project_context)
        
        # Security notes (if enabled)
        if config.include_security_notes:
            sections["security"] = await self._generate_security_section(project_context)
        
        # Contributing (if enabled)
        if config.include_contributing:
            sections["contributing"] = await self._generate_contributing_section(project_context)
        
        # License (if enabled)
        if config.include_license:
            sections["license"] = self._generate_license_section(project_context)
        
        # Custom sections
        if config.custom_sections:
            for custom_section in config.custom_sections:
                sections[custom_section["name"]] = await self._generate_custom_section(
                    custom_section, project_context
                )
        
        return sections
        
    async def _generate_header_section(self, project_context: ProjectContext, config: ReadmeConfig) -> str:
        """Generate the header section with project name and logo."""
        header_parts = []
        
        # Logo (if provided)
        if config.project_logo_path:
            header_parts.append(f'<img src="{config.project_logo_path}" alt="{project_context.name} Logo" width="200">')
            header_parts.append("")
        
        # Project title
        header_parts.append(f"# {project_context.name}")
        header_parts.append("")
        
        # Brief tagline
        if project_context.description:
            tagline = project_context.description.split('.')[0] + "."
            header_parts.append(f"*{tagline}*")
            header_parts.append("")
        
        return "\n".join(header_parts)
        
    async def _generate_description_section(self, project_context: ProjectContext) -> str:
        """Generate the project description section."""
        context_str = self.context_builder.format_context_for_llm(project_context)
        
        prompt = f"""
Generate a comprehensive project description for a README.md based on the project analysis.

PROJECT CONTEXT:
{context_str}

Generate a detailed description that includes:
1. What the project does (main purpose and functionality)
2. Key features and capabilities
3. Target audience and use cases
4. Technology stack overview
5. What makes this project unique or valuable

Format as clean markdown without a section header.
"""
        
        response = await self.llm_integration.llm_integration.generate_documentation(
            doc_type="readme_section",
            context=prompt,
            code="",
            style="markdown"
        )
        
        return response.content.strip()
        
    def _generate_badges_section(self, project_context: ProjectContext) -> str:
        """Generate badges section for the README."""
        badges = []
        
        # Language badge
        badges.append("![Python](https://img.shields.io/badge/python-3.8+-blue.svg)")
        
        # License badge (placeholder)
        badges.append("![License](https://img.shields.io/badge/license-MIT-green.svg)")
        
        # Build status (placeholder)
        badges.append("![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)")
        
        # Coverage (placeholder)
        badges.append("![Coverage](https://img.shields.io/badge/coverage-85%25-yellowgreen.svg)")
        
        # Technology-specific badges
        for tech in project_context.tech_stack:
            if "web_framework" in tech:
                badges.append("![Framework](https://img.shields.io/badge/framework-web-blue.svg)")
            elif "data_science" in tech:
                badges.append("![Data Science](https://img.shields.io/badge/data-science-orange.svg)")
        
        return " ".join(badges) + "\n"
        
    def _generate_toc_section(self, config: ReadmeConfig) -> str:
        """Generate table of contents."""
        toc_items = [
            "- [Features](#features)",
        ]
        
        if config.include_installation:
            toc_items.append("- [Installation](#installation)")
        if config.include_usage_examples:
            toc_items.append("- [Usage](#usage)")
        if config.include_api_reference:
            toc_items.append("- [API Reference](#api-reference)")
        if config.include_architecture_overview:
            toc_items.append("- [Architecture](#architecture)")
        if config.include_performance_notes:
            toc_items.append("- [Performance](#performance)")
        if config.include_security_notes:
            toc_items.append("- [Security](#security)")
        if config.include_contributing:
            toc_items.append("- [Contributing](#contributing)")
        if config.include_license:
            toc_items.append("- [License](#license)")
        
        return "## Table of Contents\n\n" + "\n".join(toc_items) + "\n"
        
    async def _generate_features_section(self, project_context: ProjectContext) -> str:
        """Generate features section based on project analysis."""
        features = []
        
        # Extract features from modules and insights
        for module in project_context.modules:
            if module.module_type == "business_logic":
                features.append(f"**{module.name.replace('_', ' ').title()}**: Core business logic implementation")
            elif module.module_type == "interface":
                features.append(f"**{module.name.replace('_', ' ').title()}**: User interface and interaction")
            elif module.module_type == "utility":
                features.append(f"**{module.name.replace('_', ' ').title()}**: Utility functions and helpers")
        
        # Add insights-based features
        security_features = sum(1 for module in project_context.modules 
                              for insight in module.insights 
                              if insight.category == "security")
        if security_features > 0:
            features.append(f"**Security**: Advanced security analysis with {security_features} security checks")
        
        performance_features = sum(1 for module in project_context.modules 
                                 for insight in module.insights 
                                 if insight.category == "performance")
        if performance_features > 0:
            features.append(f"**Performance**: Performance optimization with {performance_features} analysis points")
        
        # Add architecture-based features
        if project_context.architecture_style:
            features.append(f"**Architecture**: {project_context.architecture_style.replace('_', ' ').title()} design pattern")
        
        # Add technology stack features
        if "data_science" in project_context.tech_stack:
            features.append("**Data Science**: Advanced data analysis and machine learning capabilities")
        if "web_framework" in project_context.tech_stack:
            features.append("**Web Framework**: Modern web application framework")
        
        features_text = "\n".join(f"- {feature}" for feature in features)
        return f"## Features\n\n{features_text}\n"
        
    async def _generate_installation_section(self, project_context: ProjectContext) -> str:
        """Generate installation instructions."""
        installation_parts = [
            "## Installation",
            "",
            "### Prerequisites",
            "",
            "- Python 3.8 or higher",
        ]
        
        # Add technology-specific prerequisites
        if "data_science" in project_context.tech_stack:
            installation_parts.extend([
                "- NumPy and Pandas for data processing",
                "- Scikit-learn for machine learning"
            ])
        if "web_framework" in project_context.tech_stack:
            installation_parts.append("- Web framework dependencies")
        
        installation_parts.extend([
            "",
            "### Install from PyPI",
            "",
            f"```bash",
            f"pip install {project_context.name.lower().replace('_', '-')}",
            "```",
            "",
            "### Install from source",
            "",
            "```bash",
            f"git clone https://github.com/yourusername/{project_context.name.lower()}.git",
            f"cd {project_context.name.lower()}",
            "pip install -e .",
            "```",
        ])
        
        return "\n".join(installation_parts) + "\n"
        
    async def _generate_usage_section(self, project_context: ProjectContext) -> str:
        """Generate usage examples section."""
        context_str = self.context_builder.format_context_for_llm(project_context)
        
        prompt = f"""
Generate practical usage examples for a README.md based on the project analysis.

PROJECT CONTEXT:
{context_str}

Generate usage examples that include:
1. Basic usage example
2. Advanced usage example
3. Common use cases
4. Code examples with proper imports
5. Expected outputs or results

Format as clean markdown with code blocks. Include the section header "## Usage".
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type="readme_section",
            context=prompt,
            code="",
            style="markdown"
        )
        
        return response.content.strip() + "\n"
        
    async def _generate_api_reference_section(self, project_context: ProjectContext) -> str:
        """Generate API reference section."""
        api_parts = [
            "## API Reference",
            "",
            "### Core Modules",
            ""
        ]
        
        # Document key modules
        for module in project_context.modules[:5]:  # Top 5 modules
            api_parts.append(f"#### {module.name}")
            api_parts.append("")
            
            if module.functions:
                api_parts.append("**Functions:**")
                for func in module.functions[:3]:  # Top 3 functions
                    api_parts.append(f"- `{func.name}()`: {func.name.replace('_', ' ').title()}")
                api_parts.append("")
            
            if module.classes:
                api_parts.append("**Classes:**")
                for cls in module.classes[:3]:  # Top 3 classes
                    api_parts.append(f"- `{cls.name}`: {cls.name.replace('_', ' ').title()}")
                api_parts.append("")
        
        # Add link to full documentation
        api_parts.extend([
            "### Full Documentation",
            "",
            "For complete API documentation, see [API Docs](docs/api.md).",
            ""
        ])
        
        return "\n".join(api_parts)
        
    async def _generate_architecture_section(self, project_context: ProjectContext) -> str:
        """Generate architecture overview section."""
        arch_parts = [
            "## Architecture",
            "",
            f"**Architecture Style**: {project_context.architecture_style.replace('_', ' ').title()}",
            f"**Primary Domain**: {project_context.primary_domain.replace('_', ' ').title()}",
            "",
            "### System Overview",
            ""
        ]
        
        # Add module breakdown
        arch_parts.append("```")
        arch_parts.append(f"{project_context.name}/")
        for module in project_context.modules:
            arch_parts.append(f"├── {module.name}/")
            arch_parts.append(f"│   ├── {len(module.functions)} functions")
            arch_parts.append(f"│   └── {len(module.classes)} classes")
        arch_parts.append("```")
        arch_parts.append("")
        
        # Add key patterns
        if project_context.key_patterns:
            arch_parts.append("### Design Patterns")
            arch_parts.append("")
            for pattern in project_context.key_patterns:
                arch_parts.append(f"- **{pattern.replace('_', ' ').title()}**")
            arch_parts.append("")
        
        return "\n".join(arch_parts)
        
    async def _generate_performance_section(self, project_context: ProjectContext) -> str:
        """Generate performance section."""
        perf_parts = [
            "## Performance",
            "",
        ]
        
        # Add quality metrics
        if project_context.quality_metrics:
            perf_parts.append("### Metrics")
            perf_parts.append("")
            for metric, value in project_context.quality_metrics.items():
                if isinstance(value, (int, float)):
                    perf_parts.append(f"- **{metric.replace('_', ' ').title()}**: {value}")
            perf_parts.append("")
        
        # Add performance insights
        performance_insights = []
        for module in project_context.modules:
            for insight in module.insights:
                if insight.category == "performance":
                    performance_insights.append(insight.description)
        
        if performance_insights:
            perf_parts.append("### Performance Notes")
            perf_parts.append("")
            for insight in performance_insights[:3]:  # Top 3 insights
                perf_parts.append(f"- {insight}")
            perf_parts.append("")
        
        return "\n".join(perf_parts) if len(perf_parts) > 3 else ""
        
    async def _generate_security_section(self, project_context: ProjectContext) -> str:
        """Generate security section."""
        security_parts = [
            "## Security",
            "",
        ]
        
        # Add security insights
        security_insights = []
        for module in project_context.modules:
            for insight in module.insights:
                if insight.category == "security":
                    security_insights.append(insight.description)
        
        if security_insights:
            security_parts.append("### Security Features")
            security_parts.append("")
            for insight in security_insights[:3]:  # Top 3 insights
                security_parts.append(f"- {insight}")
            security_parts.append("")
        
        # Add general security best practices
        security_parts.extend([
            "### Best Practices",
            "",
            "- Regular security audits and updates",
            "- Input validation and sanitization", 
            "- Secure coding practices",
            "- Dependency vulnerability scanning",
            ""
        ])
        
        return "\n".join(security_parts) if len(security_insights) > 0 else ""
        
    async def _generate_contributing_section(self, project_context: ProjectContext) -> str:
        """Generate contributing guidelines section."""
        contrib_parts = [
            "## Contributing",
            "",
            "We welcome contributions! Please see our contributing guidelines.",
            "",
            "### Development Setup",
            "",
            "1. Fork the repository",
            "2. Create a virtual environment",
            "3. Install development dependencies:",
            "",
            "```bash",
            "pip install -e .[dev]",
            "```",
            "",
            "4. Run tests:",
            "",
            "```bash",
            "pytest",
            "```",
            "",
            "### Code Style",
            "",
            "- Follow PEP 8",
            "- Use type hints",
            "- Write comprehensive docstrings",
            "- Add tests for new features",
            "",
            "### Pull Requests",
            "",
            "1. Create a feature branch",
            "2. Make your changes",
            "3. Add tests",
            "4. Update documentation",
            "5. Submit a pull request",
            ""
        ]
        
        return "\n".join(contrib_parts)
        
    def _generate_license_section(self, project_context: ProjectContext) -> str:
        """Generate license section."""
        return """## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

"""
        
    async def _generate_custom_section(self, custom_section: Dict[str, str], 
                                     project_context: ProjectContext) -> str:
        """Generate a custom section."""
        section_name = custom_section["name"]
        section_prompt = custom_section.get("prompt", "")
        
        if section_prompt:
            context_str = self.context_builder.format_context_for_llm(project_context)
            full_prompt = f"{section_prompt}\n\nPROJECT CONTEXT:\n{context_str}"
            
            response = await self.llm_integration.generate_documentation(
                doc_type="readme_section",
                context=full_prompt,
                code="",
                style="markdown"
            )
            
            return f"## {section_name}\n\n{response.content.strip()}\n"
        
        return f"## {section_name}\n\n*Section content to be added.*\n"
        
    def _assemble_readme(self, sections: Dict[str, str], config: ReadmeConfig) -> str:
        """Assemble the final README from sections."""
        readme_parts = []
        
        # Define section order
        section_order = [
            "header", "badges", "description", "toc", "features",
            "installation", "usage", "api_reference", "architecture",
            "performance", "security", "contributing", "license"
        ]
        
        # Add sections in order
        for section_name in section_order:
            if section_name in sections and sections[section_name]:
                readme_parts.append(sections[section_name])
        
        # Add custom sections
        for section_name, content in sections.items():
            if section_name not in section_order and content:
                readme_parts.append(content)
        
        return "\n".join(readme_parts)
        
    async def _enhance_readme_quality(self, 
                                    readme_content: str,
                                    project_context: ProjectContext,
                                    config: ReadmeConfig) -> str:
        """Enhance README quality based on assessment feedback."""
        enhancement_prompt = f"""
Improve the following README.md to make it more comprehensive and professional:

CURRENT README:
{readme_content}

PROJECT CONTEXT:
{self.context_builder.format_context_for_llm(project_context)}

Please enhance the README by:
1. Making descriptions more detailed and engaging
2. Adding more specific technical details
3. Improving code examples with better context
4. Adding more comprehensive installation instructions
5. Including better usage examples
6. Making the overall structure more professional

Return the complete improved README.md:
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type="readme_project",
            context=enhancement_prompt,
            code="",
            style="markdown",
            temperature=0.05
        )
        
        return response.content.strip()
        
    def save_readme(self, readme_content: str, project_path: str) -> str:
        """
        Save the generated README.md to the project directory.
        
        Args:
            readme_content: Generated README content
            project_path: Path to the project directory
            
        Returns:
            str: Path to the saved README file
        """
        readme_path = Path(project_path) / "README.md"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Saved README.md to {readme_path}")
        return str(readme_path)