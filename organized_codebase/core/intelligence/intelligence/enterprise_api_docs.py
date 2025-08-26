"""
Enterprise API Documentation System Module
Handles enterprise-grade API documentation with advanced features
"""

from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
import re
from datetime import datetime


class AuthenticationType(Enum):
    """API authentication types"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic"
    CUSTOM = "custom"


class SecurityLevel(Enum):
    """Security levels for API endpoints"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN_ONLY = "admin_only"


@dataclass
class APISecurityScheme:
    """API security scheme definition"""
    name: str
    type: AuthenticationType
    description: str = ""
    header_name: str = "Authorization"
    token_prefix: str = ""
    scopes: List[str] = field(default_factory=list)
    oauth_flows: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIEnvironment:
    """API environment configuration"""
    name: str
    base_url: str
    description: str = ""
    variables: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    is_default: bool = False


@dataclass
class APIEndpointExample:
    """API endpoint example with request/response"""
    name: str
    description: str = ""
    request: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 200
    curl_example: str = ""
    language_examples: Dict[str, str] = field(default_factory=dict)


@dataclass
class APIEndpointDoc:
    """Enhanced API endpoint documentation"""
    path: str
    method: str
    summary: str
    description: str = ""
    operation_id: str = ""
    tags: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    examples: List[APIEndpointExample] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    deprecation_info: Optional[Dict[str, Any]] = None
    version_added: str = ""
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())


class EnterpriseAPIDocsSystem:
    """Enterprise-grade API documentation system"""
    
    def __init__(self):
        self.endpoints = {}
        self.security_schemes = {}
        self.environments = {}
        self.global_headers = {}
        self.rate_limits = {}
        self.webhooks = {}
        self.event_schemas = {}
        
    def add_security_scheme(self, scheme: APISecurityScheme) -> None:
        """Add a security scheme to the API documentation"""
        self.security_schemes[scheme.name] = scheme
    
    def add_environment(self, environment: APIEnvironment) -> None:
        """Add an API environment"""
        self.environments[environment.name] = environment
    
    def add_endpoint(self, endpoint: APIEndpointDoc) -> None:
        """Add an API endpoint to documentation"""
        endpoint_key = f"{endpoint.method.upper()}_{endpoint.path}"
        self.endpoints[endpoint_key] = endpoint
    
    def create_openapi_spec(self, version: str = "3.0.3",
                           api_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification"""
        if api_info is None:
            api_info = {
                "title": "Enterprise API",
                "version": "1.0.0",
                "description": "Enterprise API Documentation"
            }
        
        spec = {
            "openapi": version,
            "info": api_info,
            "servers": self._generate_servers(),
            "paths": self._generate_paths(),
            "components": {
                "securitySchemes": self._generate_security_schemes(),
                "schemas": self._generate_schemas(),
                "responses": self._generate_common_responses(),
                "parameters": self._generate_common_parameters(),
                "headers": self._generate_common_headers()
            },
            "security": self._generate_global_security(),
            "tags": self._generate_tags(),
            "webhooks": self._generate_webhooks()
        }
        
        return spec
    
    def _generate_servers(self) -> List[Dict[str, Any]]:
        """Generate server configurations"""
        servers = []
        
        for env in self.environments.values():
            server = {
                "url": env.base_url,
                "description": env.description or env.name
            }
            
            if env.variables:
                server["variables"] = {
                    var_name: {"default": var_value}
                    for var_name, var_value in env.variables.items()
                }
            
            servers.append(server)
        
        return servers
    
    def _generate_paths(self) -> Dict[str, Any]:
        """Generate OpenAPI paths from endpoints"""
        paths = {}
        
        for endpoint in self.endpoints.values():
            path = endpoint.path
            method = endpoint.method.lower()
            
            if path not in paths:
                paths[path] = {}
            
            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "operationId": endpoint.operation_id or self._generate_operation_id(endpoint),
                "tags": endpoint.tags,
                "parameters": self._format_parameters(endpoint.parameters),
                "responses": self._format_responses(endpoint.responses)
            }
            
            # Add request body if present
            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body
            
            # Add security requirements
            if endpoint.security_level != SecurityLevel.PUBLIC:
                operation["security"] = self._get_security_for_level(endpoint.security_level)
            
            # Add examples
            if endpoint.examples:
                operation["x-examples"] = [
                    {
                        "name": example.name,
                        "description": example.description,
                        "request": example.request,
                        "response": example.response
                    }
                    for example in endpoint.examples
                ]
            
            # Add deprecation info
            if endpoint.deprecation_info:
                operation["deprecated"] = True
                operation["x-deprecation"] = endpoint.deprecation_info
            
            # Add rate limiting info
            if endpoint.rate_limits:
                operation["x-rate-limits"] = endpoint.rate_limits
            
            paths[path][method] = operation
        
        return paths
    
    def _generate_security_schemes(self) -> Dict[str, Any]:
        """Generate security schemes for OpenAPI spec"""
        schemes = {}
        
        for scheme in self.security_schemes.values():
            if scheme.type == AuthenticationType.API_KEY:
                schemes[scheme.name] = {
                    "type": "apiKey",
                    "in": "header",
                    "name": scheme.header_name,
                    "description": scheme.description
                }
            elif scheme.type == AuthenticationType.BEARER_TOKEN:
                schemes[scheme.name] = {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": scheme.description
                }
            elif scheme.type == AuthenticationType.OAUTH2:
                schemes[scheme.name] = {
                    "type": "oauth2",
                    "flows": scheme.oauth_flows,
                    "description": scheme.description
                }
            elif scheme.type == AuthenticationType.BASIC_AUTH:
                schemes[scheme.name] = {
                    "type": "http",
                    "scheme": "basic",
                    "description": scheme.description
                }
        
        return schemes
    
    def _generate_schemas(self) -> Dict[str, Any]:
        """Generate common schemas"""
        return {
            "Error": {
                "type": "object",
                "required": ["code", "message"],
                "properties": {
                    "code": {
                        "type": "integer",
                        "format": "int32"
                    },
                    "message": {
                        "type": "string"
                    },
                    "details": {
                        "type": "object",
                        "additionalProperties": True
                    }
                }
            },
            "PaginationMeta": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "per_page": {"type": "integer"},
                    "total": {"type": "integer"},
                    "total_pages": {"type": "integer"}
                }
            },
            "PaginatedResponse": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {}
                    },
                    "meta": {
                        "$ref": "#/components/schemas/PaginationMeta"
                    }
                }
            }
        }
    
    def _generate_common_responses(self) -> Dict[str, Any]:
        """Generate common response definitions"""
        return {
            "BadRequest": {
                "description": "Bad Request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Unauthorized": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Forbidden": {
                "description": "Forbidden",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "NotFound": {
                "description": "Not Found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "RateLimitExceeded": {
                "description": "Rate limit exceeded",
                "headers": {
                    "X-RateLimit-Limit": {
                        "schema": {"type": "integer"},
                        "description": "Request limit per time window"
                    },
                    "X-RateLimit-Remaining": {
                        "schema": {"type": "integer"},
                        "description": "Remaining requests in current window"
                    },
                    "X-RateLimit-Reset": {
                        "schema": {"type": "integer"},
                        "description": "Time when rate limit resets"
                    }
                }
            }
        }
    
    def _generate_common_parameters(self) -> Dict[str, Any]:
        """Generate common parameter definitions"""
        return {
            "PageParam": {
                "name": "page",
                "in": "query",
                "description": "Page number for pagination",
                "schema": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                }
            },
            "LimitParam": {
                "name": "limit",
                "in": "query",
                "description": "Number of items per page",
                "schema": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20
                }
            },
            "SortParam": {
                "name": "sort",
                "in": "query",
                "description": "Sort field and direction (e.g., 'name:asc', 'created_at:desc')",
                "schema": {
                    "type": "string"
                }
            }
        }
    
    def _generate_common_headers(self) -> Dict[str, Any]:
        """Generate common header definitions"""
        return {
            "RequestId": {
                "description": "Unique identifier for the request",
                "schema": {
                    "type": "string",
                    "format": "uuid"
                }
            },
            "RateLimit": {
                "description": "Current rate limit status",
                "schema": {
                    "type": "integer"
                }
            }
        }
    
    def _generate_global_security(self) -> List[Dict[str, List[str]]]:
        """Generate global security requirements"""
        return [
            {scheme_name: scheme.scopes}
            for scheme_name, scheme in self.security_schemes.items()
        ]
    
    def _generate_tags(self) -> List[Dict[str, str]]:
        """Generate API tags from endpoints"""
        tags = {}
        
        for endpoint in self.endpoints.values():
            for tag in endpoint.tags:
                if tag not in tags:
                    tags[tag] = {"name": tag, "description": f"{tag} related operations"}
        
        return list(tags.values())
    
    def _generate_webhooks(self) -> Dict[str, Any]:
        """Generate webhook definitions"""
        return self.webhooks
    
    def create_postman_collection(self, collection_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate Postman collection from API documentation"""
        if collection_info is None:
            collection_info = {
                "name": "Enterprise API",
                "description": "Enterprise API Collection",
                "version": "1.0.0"
            }
        
        collection = {
            "info": {
                "name": collection_info["name"],
                "description": collection_info.get("description", ""),
                "version": collection_info.get("version", "1.0.0"),
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": self._generate_postman_auth(),
            "event": [],
            "variable": self._generate_postman_variables(),
            "item": self._generate_postman_items()
        }
        
        return collection
    
    def _generate_postman_auth(self) -> Dict[str, Any]:
        """Generate Postman authentication configuration"""
        # Use the first available security scheme
        for scheme in self.security_schemes.values():
            if scheme.type == AuthenticationType.BEARER_TOKEN:
                return {
                    "type": "bearer",
                    "bearer": [
                        {
                            "key": "token",
                            "value": "{{access_token}}",
                            "type": "string"
                        }
                    ]
                }
            elif scheme.type == AuthenticationType.API_KEY:
                return {
                    "type": "apikey",
                    "apikey": [
                        {
                            "key": "key",
                            "value": scheme.header_name,
                            "type": "string"
                        },
                        {
                            "key": "value",
                            "value": "{{api_key}}",
                            "type": "string"
                        }
                    ]
                }
        
        return {"type": "noauth"}
    
    def _generate_postman_variables(self) -> List[Dict[str, Any]]:
        """Generate Postman collection variables"""
        variables = []
        
        # Add environment variables
        for env in self.environments.values():
            if env.is_default:
                variables.append({
                    "key": "baseUrl",
                    "value": env.base_url,
                    "type": "string"
                })
                break
        
        # Add common variables
        variables.extend([
            {"key": "access_token", "value": "", "type": "string"},
            {"key": "api_key", "value": "", "type": "string"}
        ])
        
        return variables
    
    def _generate_postman_items(self) -> List[Dict[str, Any]]:
        """Generate Postman collection items"""
        # Group endpoints by tags
        groups = {}
        
        for endpoint in self.endpoints.values():
            primary_tag = endpoint.tags[0] if endpoint.tags else "Default"
            
            if primary_tag not in groups:
                groups[primary_tag] = []
            
            groups[primary_tag].append(endpoint)
        
        items = []
        
        for tag, tag_endpoints in groups.items():
            folder_items = []
            
            for endpoint in tag_endpoints:
                item = {
                    "name": endpoint.summary or f"{endpoint.method.upper()} {endpoint.path}",
                    "request": {
                        "method": endpoint.method.upper(),
                        "header": self._generate_postman_headers(endpoint),
                        "url": {
                            "raw": "{{baseUrl}}" + endpoint.path,
                            "host": ["{{baseUrl}}"],
                            "path": endpoint.path.strip('/').split('/')
                        }
                    },
                    "response": []
                }
                
                # Add request body if present
                if endpoint.request_body:
                    item["request"]["body"] = {
                        "mode": "raw",
                        "raw": json.dumps(endpoint.request_body.get("example", {}), indent=2),
                        "options": {
                            "raw": {
                                "language": "json"
                            }
                        }
                    }
                
                # Add query parameters
                if any(param.get("in") == "query" for param in endpoint.parameters):
                    item["request"]["url"]["query"] = [
                        {
                            "key": param["name"],
                            "value": str(param.get("example", "")),
                            "disabled": not param.get("required", False)
                        }
                        for param in endpoint.parameters
                        if param.get("in") == "query"
                    ]
                
                folder_items.append(item)
            
            items.append({
                "name": tag,
                "item": folder_items,
                "description": f"Endpoints related to {tag}"
            })
        
        return items
    
    def _generate_postman_headers(self, endpoint: APIEndpointDoc) -> List[Dict[str, str]]:
        """Generate headers for Postman request"""
        headers = [
            {"key": "Content-Type", "value": "application/json"},
            {"key": "Accept", "value": "application/json"}
        ]
        
        # Add headers from parameters
        for param in endpoint.parameters:
            if param.get("in") == "header":
                headers.append({
                    "key": param["name"],
                    "value": str(param.get("example", "")),
                    "disabled": not param.get("required", False)
                })
        
        return headers
    
    def generate_sdk_documentation(self, language: str = "python") -> str:
        """Generate SDK documentation for specific programming language"""
        if language.lower() == "python":
            return self._generate_python_sdk_docs()
        elif language.lower() == "javascript":
            return self._generate_javascript_sdk_docs()
        elif language.lower() == "curl":
            return self._generate_curl_examples()
        
        return f"SDK documentation for {language} not supported"
    
    def _generate_python_sdk_docs(self) -> str:
        """Generate Python SDK documentation"""
        docs = ["# Python SDK Documentation\n"]
        
        docs.append("## Installation\n")
        docs.append("```bash")
        docs.append("pip install enterprise-api-client")
        docs.append("```\n")
        
        docs.append("## Quick Start\n")
        docs.append("```python")
        docs.append("from enterprise_api import Client")
        docs.append("")
        docs.append("# Initialize client")
        docs.append("client = Client(api_key = os.getenv('KEY'))")
        docs.append("```\n")
        
        # Group endpoints by tags
        for tag in self._get_unique_tags():
            docs.append(f"## {tag}\n")
            
            for endpoint in self.endpoints.values():
                if tag in endpoint.tags:
                    docs.append(f"### {endpoint.summary}\n")
                    docs.append("```python")
                    
                    # Generate Python method call
                    method_name = self._python_method_name(endpoint)
                    params = self._python_method_params(endpoint)
                    
                    docs.append(f"response = client.{method_name}({params})")
                    docs.append("print(response)")
                    docs.append("```\n")
        
        return "\n".join(docs)
    
    def _generate_curl_examples(self) -> str:
        """Generate cURL examples for all endpoints"""
        docs = ["# cURL Examples\n"]
        
        for endpoint in self.endpoints.values():
            docs.append(f"## {endpoint.summary}\n")
            docs.append("```bash")
            
            curl_cmd = f"curl -X {endpoint.method.upper()}"
            
            # Add authentication
            if endpoint.security_level != SecurityLevel.PUBLIC:
                curl_cmd += " -H 'Authorization: Bearer $ACCESS_TOKEN'"
            
            # Add content type
            if endpoint.request_body:
                curl_cmd += " -H 'Content-Type: application/json'"
            
            # Add URL
            base_url = list(self.environments.values())[0].base_url if self.environments else "https://api.example.com"
            curl_cmd += f" '{base_url}{endpoint.path}'"
            
            # Add body
            if endpoint.request_body and endpoint.examples:
                example_request = endpoint.examples[0].request
                curl_cmd += f" -d '{json.dumps(example_request)}'"
            
            docs.append(curl_cmd)
            docs.append("```\n")
        
        return "\n".join(docs)
    
    def _format_parameters(self, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format parameters for OpenAPI spec"""
        formatted = []
        
        for param in parameters:
            formatted_param = {
                "name": param["name"],
                "in": param.get("in", "query"),
                "required": param.get("required", False),
                "schema": param.get("schema", {"type": "string"})
            }
            
            if "description" in param:
                formatted_param["description"] = param["description"]
            
            if "example" in param:
                formatted_param["example"] = param["example"]
            
            formatted.append(formatted_param)
        
        return formatted
    
    def _format_responses(self, responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Format responses for OpenAPI spec"""
        formatted = {}
        
        for status_code, response in responses.items():
            formatted[status_code] = {
                "description": response.get("description", f"Response {status_code}"),
                "content": response.get("content", {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                })
            }
        
        return formatted
    
    def _generate_operation_id(self, endpoint: APIEndpointDoc) -> str:
        """Generate operation ID for endpoint"""
        path_parts = [part for part in endpoint.path.split('/') if part and not part.startswith('{')]
        return f"{endpoint.method.lower()}{''.join(word.capitalize() for word in path_parts)}"
    
    def _get_security_for_level(self, level: SecurityLevel) -> List[Dict[str, List[str]]]:
        """Get security requirements for security level"""
        if level == SecurityLevel.PUBLIC:
            return []
        
        # Return first available security scheme
        for scheme_name, scheme in self.security_schemes.items():
            return [{scheme_name: scheme.scopes}]
        
        return []
    
    def _get_unique_tags(self) -> Set[str]:
        """Get unique tags from all endpoints"""
        tags = set()
        for endpoint in self.endpoints.values():
            tags.update(endpoint.tags)
        return sorted(tags)
    
    def _python_method_name(self, endpoint: APIEndpointDoc) -> str:
        """Generate Python method name from endpoint"""
        # Simple conversion - can be enhanced
        path_parts = [part for part in endpoint.path.split('/') if part and not part.startswith('{')]
        method_name = f"{endpoint.method.lower()}_{'_'.join(path_parts)}"
        return method_name.replace('-', '_')
    
    def _python_method_params(self, endpoint: APIEndpointDoc) -> str:
        """Generate Python method parameters"""
        params = []
        
        for param in endpoint.parameters:
            param_name = param["name"].replace('-', '_')
            if param.get("required", False):
                params.append(param_name)
            else:
                params.append(f"{param_name}=None")
        
        return ", ".join(params)