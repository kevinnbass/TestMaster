"""
Template System Enumerations

This module contains all enumerations used by the template system.
"""

from enum import Enum


class ProjectType(Enum):
    """Types of projects for README templates."""
    WEB_APPLICATION = "web_application"
    API = "api"
    LIBRARY = "library"
    CLI_TOOL = "cli_tool"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    DESKTOP_APPLICATION = "desktop_application"
    MOBILE_APPLICATION = "mobile_application"
    PLUGIN = "plugin"
    DOCUMENTATION = "documentation"
    GENERIC = "generic"


class TemplateStyle(Enum):
    """Template styles available."""
    COMPREHENSIVE = "comprehensive"
    MINIMAL = "minimal"
    DETAILED = "detailed"
    SIMPLE = "simple"