"""
Template System Data Models

This module contains data models and structures for the template system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .enums import ProjectType


@dataclass
class ReadmeContext:
    """Context information for README generation."""
    project_name: str
    project_type: ProjectType
    description: str
    author: str
    license_type: Optional[str] = None
    version: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    installation_steps: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)
    api_endpoints: List[Dict[str, str]] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    contributing_guidelines: bool = True
    changelog_link: Optional[str] = None
    documentation_link: Optional[str] = None
    demo_link: Optional[str] = None
    badges: List[Dict[str, str]] = field(default_factory=list)
    tech_stack: List[str] = field(default_factory=list)


@dataclass
class TemplateMetadata:
    """Metadata for a template."""
    name: str
    description: str
    template_type: str
    format: str
    author: str
    version: str
    tags: List[str] = field(default_factory=list)
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    target_audience: str = "all"


@dataclass 
class Template:
    """Template structure."""
    metadata: TemplateMetadata
    content: str
    examples: List[str] = field(default_factory=list)