"""
TestMaster Template System

A modular template system for generating README files and other documentation.
"""

from .enums import ProjectType, TemplateStyle
from .models import ReadmeContext, Template, TemplateMetadata
from .template_manager import TemplateManager
from .template_engine import TemplateProcessor, TemplateLogicHandler

__all__ = [
    'ProjectType',
    'TemplateStyle',
    'ReadmeContext',
    'Template',
    'TemplateMetadata',
    'TemplateManager',
    'TemplateProcessor',
    'TemplateLogicHandler',
]

__version__ = '1.0.0'