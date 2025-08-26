#!/usr/bin/env python3
"""
API Templates Module Modularization Tool
Splits the large api_templates.py into organized, maintainable modules
"""

import os
import re
from pathlib import Path

def modularize_api_templates():
    """Modularize the large API templates file"""
    source_file = "C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/testmaster/intelligence/documentation/templates/api_templates.py"
    output_dir = Path("C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/testmaster/intelligence/documentation/templates/api")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Modularizing {source_file} ({len(content.split(chr(10)))} lines)")
    
    # 1. Create base types and enums module
    base_content = '''"""
API Documentation Base Types and Enums
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat


class ApiType(Enum):
    """Types of APIs for documentation templates."""
    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    RPC = "rpc"


class AuthenticationType(Enum):
    """Authentication types for APIs."""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM = "custom"
    NONE = "none"


@dataclass
class ApiEndpoint:
    """API endpoint information."""
    method: str
    path: str
    description: str
    parameters: List[Dict[str, str]] = None
    request_body: Optional[str] = None
    response_example: Optional[str] = None
    status_codes: List[Dict[str, str]] = None
    tags: List[str] = None
    deprecated: bool = False
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.status_codes is None:
            self.status_codes = []
        if self.tags is None:
            self.tags = []


@dataclass
class ApiModel:
    """API data model information."""
    name: str
    description: str
    properties: Dict[str, Dict[str, str]]
    required: List[str] = None
    example: Optional[str] = None
    
    def __post_init__(self):
        if self.required is None:
            self.required = []


@dataclass
class ApiContext:
    """Context information for API documentation generation."""
    api_name: str
    description: str
    version: str
    base_url: str
    authentication: AuthenticationType = AuthenticationType.NONE
    endpoints: List[ApiEndpoint] = None
    models: List[ApiModel] = None
    rate_limits: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None
    license_info: Optional[Dict[str, str]] = None
    servers: List[Dict[str, str]] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = []
        if self.models is None:
            self.models = []
        if self.servers is None:
            self.servers = []
        if self.tags is None:
            self.tags = []
'''
    
    base_file = output_dir / "base.py"
    with open(base_file, 'w', encoding='utf-8') as f:
        f.write(base_content)
    print(f"Created {base_file}")
    
    # Extract method sections from original file
    lines = content.split('\n')
    
    # Find all template creation methods
    method_sections = {
        'rest': {'start': None, 'end': None, 'content': []},
        'graphql': {'start': None, 'end': None, 'content': []},
        'grpc': {'start': None, 'end': None, 'content': []},
        'websocket': {'start': None, 'end': None, 'content': []},
        'webhook': {'start': None, 'end': None, 'content': []}
    }
    
    # Find method boundaries
    current_method = None
    for i, line in enumerate(lines):
        if 'def _create_rest_templates(self):' in line:
            current_method = 'rest'
            method_sections[current_method]['start'] = i
        elif 'def _create_graphql_templates(self):' in line:
            if current_method == 'rest':
                method_sections['rest']['end'] = i
            current_method = 'graphql'
            method_sections[current_method]['start'] = i
        elif 'def _create_grpc_templates(self):' in line:
            if current_method == 'graphql':
                method_sections['graphql']['end'] = i
            current_method = 'grpc'
            method_sections[current_method]['start'] = i
        elif 'def _create_websocket_templates(self):' in line:
            if current_method == 'grpc':
                method_sections['grpc']['end'] = i
            current_method = 'websocket'
            method_sections[current_method]['start'] = i
        elif 'def _create_webhook_templates(self):' in line:
            if current_method == 'websocket':
                method_sections['websocket']['end'] = i
            current_method = 'webhook'
            method_sections[current_method]['start'] = i
        elif current_method and line.strip().startswith('def ') and current_method in line:
            # End of current method section
            if method_sections[current_method]['end'] is None:
                method_sections[current_method]['end'] = i
    
    # Set end for webhook if not set
    if method_sections['webhook']['start'] and not method_sections['webhook']['end']:
        # Find next major section or end of class
        for i in range(method_sections['webhook']['start'], len(lines)):
            if (lines[i].strip().startswith('class ') or 
                lines[i].strip().startswith('def ') and 'webhook' not in lines[i]):
                method_sections['webhook']['end'] = i
                break
        else:
            method_sections['webhook']['end'] = len(lines)
    
    # Extract content for each method
    for method, info in method_sections.items():
        if info['start'] is not None and info['end'] is not None:
            info['content'] = lines[info['start']:info['end']]
    
    # Create individual template modules
    modules_created = []
    
    for api_type in ['rest', 'graphql', 'grpc', 'websocket', 'webhook']:
        if not method_sections[api_type]['content']:
            continue
            
        module_file = output_dir / f"{api_type}_templates.py"
        
        with open(module_file, 'w', encoding='utf-8') as f:
            f.write(f'''"""
{api_type.upper()} API Documentation Templates
"""

from typing import Dict
from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat
from .base import ApiContext, ApiEndpoint, ApiModel, ApiType, AuthenticationType


class {api_type.title()}TemplateProvider:
    """Provider for {api_type.upper()} API documentation templates."""
    
    def __init__(self):
        self.templates: Dict[str, Template] = {{}}
        self._create_{api_type}_templates()
    
''')
            
            # Write the extracted method content
            for line in method_sections[api_type]['content']:
                # Adjust indentation and method signature
                if f'def _create_{api_type}_templates(self):' in line:
                    f.write(f'    def _create_{api_type}_templates(self):\n')
                else:
                    f.write(line + '\n')
            
            # Add a getter method
            f.write(f'''
    def get_templates(self) -> Dict[str, Template]:
        """Get all {api_type.upper()} templates."""
        return self.templates
    
    def get_template(self, name: str) -> Template:
        """Get a specific {api_type.upper()} template by name."""
        return self.templates.get(name)
''')
        
        modules_created.append(module_file)
        print(f"Created {module_file} ({len(method_sections[api_type]['content'])} lines)")
    
    # Create a unified manager module
    manager_content = f'''"""
Unified API Template Manager
"""

from typing import Dict, Optional
from .base import ApiContext, ApiType
from .rest_templates import RestTemplateProvider
from .graphql_templates import GraphqlTemplateProvider
from .grpc_templates import GrpcTemplateProvider
from .websocket_templates import WebsocketTemplateProvider
from .webhook_templates import WebhookTemplateProvider
from ..template_engine import Template


class ApiTemplateManager:
    """
    Unified manager for all API documentation templates.
    """
    
    def __init__(self):
        """Initialize the API template manager."""
        self.providers = {{
            ApiType.REST: RestTemplateProvider(),
            ApiType.GRAPHQL: GraphqlTemplateProvider(),
            ApiType.GRPC: GrpcTemplateProvider(),
            ApiType.WEBSOCKET: WebsocketTemplateProvider(),
            ApiType.WEBHOOK: WebhookTemplateProvider(),
        }}
    
    def get_template(self, api_type: ApiType, template_name: str) -> Optional[Template]:
        """Get a specific template by API type and name."""
        provider = self.providers.get(api_type)
        if provider:
            return provider.get_template(template_name)
        return None
    
    def get_all_templates(self, api_type: ApiType) -> Dict[str, Template]:
        """Get all templates for a specific API type."""
        provider = self.providers.get(api_type)
        if provider:
            return provider.get_templates()
        return {{}}
    
    def list_available_templates(self, api_type: ApiType) -> list:
        """List all available template names for an API type."""
        provider = self.providers.get(api_type)
        if provider:
            return list(provider.get_templates().keys())
        return []
    
    def generate_documentation(self, api_type: ApiType, template_name: str, 
                             context: ApiContext) -> str:
        """Generate documentation using the specified template and context."""
        template = self.get_template(api_type, template_name)
        if template:
            return template.render(context.__dict__)
        raise ValueError(f"Template '{{template_name}}' not found for API type {{api_type.value}}")
'''
    
    manager_file = output_dir / "manager.py"
    with open(manager_file, 'w', encoding='utf-8') as f:
        f.write(manager_content)
    print(f"Created {manager_file}")
    modules_created.append(manager_file)
    
    # Create __init__.py
    init_content = '''"""
API Documentation Templates Module
"""

from .manager import ApiTemplateManager
from .base import ApiType, AuthenticationType, ApiEndpoint, ApiModel, ApiContext

__all__ = [
    'ApiTemplateManager',
    'ApiType', 
    'AuthenticationType',
    'ApiEndpoint',
    'ApiModel', 
    'ApiContext'
]
'''
    
    init_file = output_dir / "__init__.py"
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    print(f"Created {init_file}")
    
    # Archive the original file
    archive_dir = Path("C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/archive")
    archive_file = archive_dir / "api_templates_original_2813_lines.py"
    os.rename(source_file, str(archive_file))
    print(f"Archived original file to: {archive_file}")
    
    print(f"\\nModularization Complete!")
    print(f"- Created {len(modules_created) + 2} modular files")
    print(f"- Original: 1 file with 2,813 lines")
    print(f"- Result: {len(modules_created) + 2} focused modules")
    
    return len(modules_created) + 2

if __name__ == "__main__":
    modularize_api_templates()