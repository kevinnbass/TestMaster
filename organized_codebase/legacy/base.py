"""
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
