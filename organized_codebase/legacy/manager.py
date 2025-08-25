"""
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
        self.providers = {
            ApiType.REST: RestTemplateProvider(),
            ApiType.GRAPHQL: GraphqlTemplateProvider(),
            ApiType.GRPC: GrpcTemplateProvider(),
            ApiType.WEBSOCKET: WebsocketTemplateProvider(),
            ApiType.WEBHOOK: WebhookTemplateProvider(),
        }
    
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
        return {}
    
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
        raise ValueError(f"Template '{template_name}' not found for API type {api_type.value}")
